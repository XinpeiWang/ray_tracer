# GPU Rendering on macOS: Metal/MetalRT Feasibility Study

Answers one question: could this project's GPU renderer run on Apple
hardware, and what would that actually take? Written after getting the CPU
renderer and Qt GUI building/running on macOS for the first time (see the
"macOS (CPU-only)" section of [`README.md`](../README.md)) — this document
is the natural next question that work raised, not a commitment to build
anything. No code changes here.

**Bottom line up front**: this is not a portability fix like the CPU build
was. It is a full rewrite of a ~53,000-line, three-backend GPU renderer
against a different vendor's ray-tracing API and shading language, with no
source-level or binary compatibility between CUDA/OptiX and Metal at any
layer. Apple Silicon has no CUDA support at all (not "unsupported," not
present in the driver stack), and Apple dropped NVIDIA GPU support outright
years ago, so there is no shim, translation layer, or compatibility mode to
lean on — every line has to be re-expressed by hand.

---

## 1. What exists today

`gpu/optix/` is ~53,000 lines across three independently-compiled GPU
backends (see that directory's own [`README.md`](../gpu/optix/README.md)
for the file-naming convention):

| Backend | What it is | Rough scale |
|---|---|---|
| **Recursive** (`--gpu`) | Mega-kernel OptiX pipeline: raygen → intersect → closest-hit → miss, with real `traceRay()` recursion for bounces | `optix_programs.cu` + `optix_device_helpers*.h` + `optix_intersection_*.h` + `optix_renderer*.cpp/.h` |
| **Wavefront** (`--gpu --wavefront`) | Queue-based path tracer: separate compute kernels per work type (intersect/shade/shadow/accumulate), its own duplicated intersection code (OptiX pipeline-compile-option constraints force this — see the directory README) | `wavefront_*.cu/.h`, largest of the three |
| **SPPM** (`--gpu --sppm`) | GPU photon mapping | `sppm_*.cu/.h` |

Both the recursive and wavefront backends implement essentially the full
CPU feature set — every pbrt-v4 shape, material, light, camera, and medium
type the CPU backend has, per
[`docs/FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md)'s complete-feature
table. This is not a "GPU does the easy 80%" renderer; the two are close to
parity by design, and the project's own test suite has dedicated
CPU/GPU-recursive/GPU-wavefront parity tests
(`tests/unit/material_cpu_gpu_parity_tests.cpp`) to keep it that way.

On top of that shape/material/light parity, the **wavefront backend alone**
also carries a real-time "Live Preview" mode most path tracers this size
don't attempt at all — independently toggleable (`qt_gui/settings_keys.h`):

- **ReSTIR DI and GI** (`wavefront_kernels_restir.cu`, 824 lines) — reservoir
  resampling for direct and indirect lighting
- **SVGF denoising** (`wavefront_kernels_svgf.cu`, 446 lines) — spatiotemporal
  variance-guided filtering
- **NRC, a neural radiance cache** (`wavefront_kernels_nrc.cu` +
  `wavefront_nrc_mlp.h`, ~770 lines) — a small hand-rolled MLP run inline in
  the path tracer to cache/predict indirect radiance
- **Neural temporal upscaling** (`wavefront_upscale_mlp.h`, 256 lines) —
  a second small MLP for upsampling low-res frames
- A probe cache and path-guiding toggle beyond even that list

Plus, independent of Live Preview: the **OptiX AI Denoiser**
(`optix_denoiser.h`), spectral/hero-wavelength rendering
(`spectral_device.h`), tabulated BSSRDF subsurface scattering
(`optix_bssrdf.h`), and the pbrt-v4 GPU scene builder
(`scene_builder.cpp`, 6,191 lines; `pbrt_gpu_builder.h`, 1,338 lines) that
converts a loaded `.pbrt` scene into GPU-uploadable buffers and builds the
acceleration structures for all three backends.

None of this — not one file — compiles or runs on Apple hardware. CUDA has
no code path on Apple Silicon at all (unlike, say, an unsupported-but-present
driver); OptiX is an NVIDIA SDK with no macOS build. This is the exact gap
the README already calls out as "not a 'not ported yet' gap."

## 2. The target: Metal + MetalRT

Apple's ray-tracing stack is **Metal Performance Shaders Ray Tracing** /
inline ray tracing in Metal Shading Language (MSL), available since
Metal 3 (macOS 13+, which is already this project's `LSMinimumSystemVersion`
per the Qt GUI's `Info.plist`). At a high level it provides:

- `MTLAccelerationStructure` (primitive + instance) — OptiX's GAS/IAS
  equivalent, built from the GPU side, similarly opaque/vendor-managed
- Ray tracing called **from inside an ordinary compute kernel** via MSL's
  `raytracing::intersector` — there is no separate raygen/closest-hit/miss
  *pipeline* with a shader binding table the way OptiX has; a kernel calls
  `intersector.intersect(ray)` inline and branches on the result itself
- Custom (non-triangle) primitives via **bounding-box intersection
  functions** — conceptually the same idea as OptiX's custom intersection
  programs, different plumbing
- `MTLLibrary`/`.metallib` — compiled MSL, the `.metallib` counterpart to
  OptiX's PTX-at-runtime model

The architectural fit is *not* uniform across this project's own three
backends, which is the single most important finding of this study:

- **Wavefront maps well.** It's already a queue-of-compute-kernels design —
  "do intersection for this queue, then shading, then shadow rays, then
  accumulate" is close in shape to Metal's own "call `intersect()` inline
  from a compute kernel" model. Porting wavefront is mostly a language port
  (CUDA → MSL) plus a host-API port (CUDA/OptiX host calls → Metal), not an
  architecture redesign.
- **Recursive does not map well.** Its mega-kernel design leans on OptiX's
  actual pipeline dispatch and `traceRay()`-triggers-closest-hit-triggers-
  more-traceRay() recursion. Metal's inline model has no equivalent
  "the hardware calls my hit shader" mechanism — recursion has to become an
  explicit loop or be restructured as wavefront-style queues. In practice,
  porting the recursive backend to Metal means rebuilding it in the
  wavefront backend's shape, not a line-for-line translation.
- **SPPM is the smallest lift** (126-line kernel file) but still needs the
  same host-API and acceleration-structure rewrite as everything else.

## 3. Feature-by-feature port risk

| Piece | CUDA/OptiX → Metal mapping | Risk |
|---|---|---|
| Triangle mesh BVH | `OptixTraversableHandle` (GAS) → `MTLAccelerationStructure` (primitive) | **Low** — both are hardware-built, vendor-opaque, conceptually identical |
| Custom-primitive intersection (sphere/quad/disk/cylinder/bilinear patch) | OptiX intersection programs → MSL bounding-box intersection functions | **Medium** — same idea, real per-primitive porting work (5 shape types × 2 backends worth of intersection math to re-derive/re-verify in MSL) |
| Wavefront queue kernels (intersect/shade/shadow/accumulate) | CUDA `__global__` kernels → MSL compute kernels + inline `intersect()` | **Medium** — mostly mechanical language port; CUDA's `float3`/`make_float3` etc. become `simd::float3`, no `__device__`/`__global__` distinction, different memory-space qualifiers (`device`/`threadgroup`/`constant`) |
| Recursive mega-kernel | OptiX pipeline + SBT + real recursion | **High** — not a port, a redesign into wavefront's shape |
| Materials/BSDFs (all pbrt-v4 types, principled, measured, hair) | Pure math, `__device__` functions | **Low-Medium** — the math itself is platform-agnostic; the port is mechanical (CUDA math intrinsics → MSL equivalents), the volume (dozens of BxDF files) is the real cost, not the difficulty per-line |
| Spectral/hero-wavelength rendering | Pure math | **Low** |
| BSSRDF | Pure math + texture sampling | **Low-Medium** |
| ReSTIR DI/GI | Reservoir structs in device memory, resampling kernels | **Medium** — algorithm ports fine, needs careful re-verification (reservoir bias correction is easy to get subtly wrong on any new backend) |
| SVGF | Sequential compute passes over G-buffers | **Medium** — straightforward kernel-by-kernel port, more files than complexity per file |
| NRC / neural upscale (hand-rolled MLP) | Custom matmul in CUDA → custom matmul in MSL | **Low-Medium** — it's already a from-scratch MLP, not relying on cuDNN/Tensor Cores, so there's no vendor-library dependency to replace; if the CUDA version leans on Tensor Core intrinsics (`wmma`) that's real work, if it's plain FMA loops it's a direct port |
| OptiX AI Denoiser | `optixDenoiserInvoke()`, an NVIDIA pretrained model with **no Apple equivalent shipped as source or weights** | **Dead end as-is** — Metal Performance Shaders does ship its own ray-tracing-oriented denoising kernels (an SVGF-family filter), which is a plausible practical substitute given this project already has its own independent SVGF implementation for Live Preview to draw on — but this needs verifying against current MPS API docs before committing to it, not assumed |
| Scene builder / pbrt-v4 → GPU buffers (`scene_builder.cpp`, `pbrt_gpu_builder.h`) | `cudaMalloc`/`cudaMemcpy` host-side upload → `MTLBuffer`/`MTLDevice` | **Medium-High**, mostly from sheer size (7,500+ combined lines) rather than conceptual difficulty — this is "every scene primitive/material/light gets uploaded correctly," which is exactly the kind of code where an easy-to-miss field mismatch causes a silent wrong-looking render rather than a build error |
| Host renderer classes (`optix_renderer_init/_scene/_render.cpp`, `wavefront_path_tracer.cpp`) | OptiX/CUDA host API | **High** — full rewrite against `MTLDevice`/`MTLCommandQueue`/`MTLComputePipelineState` |
| Build system | `build_optix.targets` (nvcc → PTX, MSBuild `<Exec>` tasks) | **Medium** — needs a macOS-side equivalent compiling `.metal` → `.metallib` (Xcode's `xcrun metal`), and CMake's native support for this is thinner than its CUDA language support, so likely another hand-rolled custom-command setup like the existing one |

## 4. What this is realistically not

- **Not a recompile.** There is no `#ifdef __APPLE__` path through any of
  this. Every `.cu` file and every `__device__` helper header is CUDA
  source; MSL is a different, if related, C++ dialect (no virtual
  functions, no dynamic allocation in shaders, different template support,
  its own math-library namespace).
- **Not a small subset that "just needs stubbing."** Unlike the CPU-build
  fixes (missing include, missing source, a few `#ifdef _WIN32` guards),
  there is no existing fallback code path to complete here — Metal support
  doesn't exist in any form yet.
- **Not something CI or a single session can validate end-to-end without
  real Apple Silicon hardware and a real Metal debugger/profiler** —
  correctness bugs in a GPU path tracer (wrong BSDF sign, a reservoir bias
  bug, a race in a shared buffer) are exactly the class of bug that "looks
  plausible, renders *something*, is subtly wrong" — the same failure mode
  `docs/FEATURE_INVENTORY.md` already documents happening even *within*
  this project's existing two NVIDIA-only backends (see that doc's
  `BVHLightSampler2` gap: a real, shipped GPU-recursive bug that rendered
  scenes ~24x too dark and passed initial review before a parity test
  caught it). A from-scratch third backend without years of the existing
  CPU/GPU parity-testing infrastructure behind it yet is higher-risk, not
  lower-risk, than that history suggests.

## 5. Recommended approach, if pursued

1. **Start from wavefront, not recursive.** It's the closer architectural
   fit to Metal's inline ray-tracing model, and it's also where the
   highest-value features (Live Preview, ReSTIR, SVGF, NRC) already live —
   porting it first delivers the most capability for the effort, rather
   than porting the simpler recursive backend first and hitting the hard
   architectural mismatch second.
2. **Triangles + one or two basic materials first**, end to end (host
   buffer upload → acceleration structure → intersect → shade → present),
   before porting the rest of the material/light/shape catalogue. This
   validates the whole pipeline shape early, when a wrong assumption is
   cheap to fix, rather than after 40 material types are already ported.
3. **Defer ReSTIR/SVGF/NRC/denoiser until basic path tracing round-trips
   correctly** and has *some* parity coverage against the CPU backend
   (reusing the existing `MaterialCpuGpuParityTest` pattern, extended to a
   third backend, would directly reuse infrastructure this project already
   trusts).
4. **Treat the OptiX AI Denoiser replacement as its own open question**,
   not an assumed drop-in — verify current Metal Performance Shaders
   denoising APIs against what's actually needed before scoping that
   specific piece.
5. **New directory, not a rewrite-in-place**: `gpu/metal/`, mirroring
   `gpu/optix/`'s existing file-naming convention (this project already
   uses prefix-based backend separation inside one directory successfully),
   selected via a new CMake option analogous to `RT_BUILD_GPU` — e.g.
   `RT_BUILD_METAL`, additive, same posture the CPU-only `CMakeLists.txt`
   already takes toward the GPU build (never touches the Windows
   `.sln`/`.vcxproj`, and this shouldn't touch `gpu/optix/` either).

## 6. Effort estimate

Wide ranges, because the real unknowns are algorithmic correctness on a
third backend (not covered by any existing test) and how much of the CUDA
math intrinsic surface has a direct MSL equivalent versus needing
hand-derivation — both only resolve by actually doing the work, not by
estimating harder:

| Scope | Rough estimate |
|---|---|
| Triangles + BVH + one material, static camera, no lighting features beyond direct | Low single-digit weeks |
| Wavefront backend at CPU-backend material/light/shape parity (no ReSTIR/SVGF/NRC) | Several weeks to a couple months |
| Full Live Preview parity (ReSTIR DI/GI + SVGF + NRC + upscaling) | Additional weeks on top of the above — this is research-grade real-time GI, not routine porting, even once the base path tracer works |
| Recursive backend, SPPM backend, OptiX-AI-Denoiser-equivalent, full pbrt-v4 material catalogue (measured BSDFs, hair, subsurface, all media types) | Substantial additional time — realistically this is a multi-month project overall for real feature parity, not a single push |

## 7. Suggested next step

Given the above, the highest-value *next* piece of real work (not covered
by this document, which is deliberately design-only) would be a **triangle
+ one-material Metal proof of concept**: confirm the acceleration-structure
and inline-intersection pipeline actually works end-to-end on this
project's own scene data before committing to porting any of the harder
material/lighting code. That's a self-contained, time-boxed spike rather
than the start of the full port, and it would turn several of this
document's "should map" claims into "confirmed maps" or "doesn't, here's
why" before more effort goes in on the strength of an estimate alone.

## 8. Proof-of-concept results (done)

Built and run on real Apple Silicon hardware (M2) as `gpu/metal/metal_poc.mm`
+ `metal_poc.metal` — a standalone CLI tool, not yet wired into
`CMakeLists.txt`/`launcher/main.cpp`. Scene: a hardcoded Cornell-box-style
room (5 quads for floor/ceiling/back/left/right walls + one small tilted
object), one directional light, flat Lambertian shading, rendered to PNG.

**Confirmed working end to end**: `MTLAccelerationStructureTriangleGeometry
Descriptor` (primitive AS) wrapped in an `MTLInstanceAccelerationStructure
Descriptor` (instance AS) builds cleanly; a compute kernel using
`raytracing::intersector<instancing, triangle_data>` and inline
`.intersect(ray, accelStructure)` correctly finds hits; `primitive_id`
correctly indexes per-triangle material data (the tilted object renders in
its own distinct colour from the walls, proving indexing isn't just
"whichever mesh" but genuinely per-triangle); output writes to a
`texture2d` and reads back correctly. Section 3's "Low risk" calls for
triangle BVH and the wavefront-shaped kernel model both held up in
practice, not just in theory.

**Two concrete findings that update earlier sections:**

1. **`MTLCreateSystemDefaultDevice()` fails for command-line tools.**
   Confirmed via `log show`: *"Use of MTLCreateSystemDefaultDevice is not
   supported for non-interactive (commandline or daemon) apps. Use
   MTLCopyAllDevices(WithObserver) instead."* This wasn't anticipated
   anywhere in this document, wasn't sandboxing or permissions (verified:
   Claude.app itself carries no `com.apple.security.app-sandbox`
   entitlement — this is Apple's own device-creation API drawing a real
   distinction between GUI apps and CLI/daemon processes, not an
   environment quirk), and it matters beyond this one POC: **any future
   standalone test/CLI tooling for the real Metal backend** (parity tests
   mirroring `tests/unit/material_cpu_gpu_parity_tests.cpp`, a future
   `--gpu-metal` CLI flag on `ray_tracer` itself, anything invoked outside
   the Qt GUI's own app-bundle context) needs `MTLCopyAllDevices()` instead
   of the "obvious" API, or it silently gets no device with an error message
   that doesn't say why unless you go looking in the system log.
2. **`intersection_result` carries no face/shading normal.** Section 3
   didn't call this out specifically. A hit result gives you
   `primitive_id`/`instance_id`/barycentric coordinates, not a normal —
   computing one means either deriving it from the triangle's own raw
   vertex positions (what this POC does, and what a from-scratch geometric
   normal needs regardless of source) or, for the real port, carrying
   proper per-vertex normals as a second vertex attribute buffer
   alongside positions — the correct approach long-term, since flat
   per-triangle normals would visibly facet every curved surface the CPU
   backend currently smooth-shades. Worth deciding explicitly when the
   real vertex-buffer format for `gpu/metal/` gets designed, not
   discovered mid-port.

Net effect: this POC didn't surface anything that changes Section 6's
effort estimates or Section 4's "not a recompile" framing — the
architecture-level bets in Sections 2-3 held up — but it did surface two
API-level gotchas cheaply, exactly what a time-boxed spike is for.

## 9. Proof-of-concept, step 2: a real path integrator (done)

Step 1 (above) was a single-bounce raycast — no shadow rays, no GI, no
material branching. This step turned it into an actual minimal Monte Carlo
path tracer, still in the same standalone `gpu/metal/metal_poc.mm`/`.metal`
files:

- **Shadow-ray occlusion** against the one directional light (proper NEE,
  not just an unoccluded `dot(N,L)` term)
- **Multi-bounce indirect lighting** via cosine-weighted hemisphere sampling
  with Russian-roulette termination after depth 3 — the same unbiased
  early-termination shape this project's CPU integrator uses
  (`path_integrator.h`)
- **A second material** (mirror/perfect specular) to prove per-primitive
  *material-type* branching, not just per-primitive colour — a correct
  render shows the room genuinely reflected in the mirror object, not a
  flat grey quad
- **Multi-sample antialiasing** via jittered primary rays, with an
  SPP/max-depth CLI-configurable loop

**Result, visually confirmed**: correct reflections in the mirror (the
ceiling and side wall visibly reflected at the expected angle), a real
contact shadow cast by the mirror object onto the floor, and subtle red/
green colour bleeding onto the back wall near the corresponding walls —
genuine diffuse GI, not direct lighting alone. 600×600 at 256 samples/pixel,
max depth 8, rendered in ~2.2 seconds wall-clock (including acceleration-
structure build and runtime shader compilation) on an M2. Noise at low
sample counts (32 spp) converges to a clean image at higher counts, the
expected Monte Carlo behaviour.

**One more design decision this step forced, worth recording**: MSL's
`float3` inside a struct shared between host and device code is 16-byte
*aligned* but only 12 bytes in *size* — a host-side `simd::float3` and a
device-side `float3` field at the same struct offset are not guaranteed to
agree on where the *next* field starts. This is exactly the kind of silent
layout mismatch that "compiles fine, runs fine, renders wrong" — no crash,
no validation error, just data landing in the wrong field. Fixed by using
`packed_float3` (size 12, align 4, no padding) on both sides of every
shared struct (`Uniforms`, `TriangleMaterial`) instead, with explicit
`float3(...)` conversions at math call sites inside the shader. Worth
carrying forward as a rule for the real port: **any struct that crosses
the host/device boundary uses packed types, full stop** — don't rely on
reasoning through MSL's default packing rules per-field, design the
ambiguity out.

## 10. CMake integration (done)

`metal_poc` is now a real CMake target, not an ad-hoc `clang++` invocation:
`cmake -B build -DRT_BUILD_METAL=ON && cmake --build build --target
metal_poc`. Mirrors `RT_BUILD_GPU`'s existing posture in this file exactly
— off by default, purely additive (verified: a default `cmake -B build`
with no flags produces byte-identical `cpu_renderer`/`ray_tracer`/
`scene_metadata` targets, no OBJCXX language probe, no `metal_poc` target
at all), fails with a clear message rather than a raw CMake error if
`RT_BUILD_METAL=ON` is passed on a non-Apple platform. Needed enabling
CMake's `OBJCXX` language (Objective-C++, first-class since CMake 3.16)
conditionally, the same "only raise the version floor for the path that
actually needs it" shape `RT_BUILD_GPU` already uses for CUDA's 3.18
requirement. Also switched the shader-source lookup from a `__FILE__`-
relative path (fragile once a real build directory can live anywhere) to
an explicit `RT_METAL_SHADER_DIR` compile definition — falls back to the
old `__FILE__` behaviour when building outside CMake entirely, so the
original ad-hoc `clang++ metal_poc.mm ...` workflow from section 7 still
works unchanged. Verified: the CMake-built binary renders byte-plausible-
identical output to the hand-compiled one from section 9.

## 11. Proof-of-concept, step 3: a dielectric material and a custom (sphere) primitive (done)

Everything through section 10 used triangles only. This step added the
scene's first non-triangle primitive — a glass sphere — closing out
section 3's one genuinely "Medium risk, unconfirmed" line item
(custom-primitive intersection) with a real, working example instead of a
prediction. Also added a third material, dielectric (glass), alongside the
existing Lambertian/mirror pair — Schlick-approximated Fresnel choosing
reflect vs. refract stochastically each bounce, same approach pbrt-v4 and
this project's own CPU dielectric material use.

**Result, visually confirmed**: a correctly refracting glass sphere —
visible bending of the room's colours through it, a subtle caustic-like
light-focusing patch on the floor beneath it, and the existing mirror
still correctly reflecting both the sphere and the green wall behind the
new geometry. 600×600 @ 256spp @ depth 10 renders in ~3.3 seconds on an
M2.

**This took real debugging to get working, and the two bugs found are
worth recording in detail — this is exactly the kind of cost a
prediction-only feasibility study can't surface, only building something
can:**

1. **A custom-primitive geometry and a triangle geometry cannot safely
   share an instance acceleration structure without each geometry's
   `opaque` flag set explicitly.** Adding the sphere (a second instance,
   bounding-box geometry) made the *existing, previously-working* triangle
   room vanish entirely — not just the sphere. Root cause: once any
   intersection-function table is bound at trace time at all, a geometry
   without `opaque = YES` set explicitly can get routed through that table
   for hit confirmation instead of accepting the hardware triangle
   intersector's result directly — and the sphere's intersection function,
   invoked with a triangle's `primitive_id`, has no way to produce a
   sensible result. Fixed by setting `.opaque = YES` explicitly on *both*
   the triangle and bounding-box geometry descriptors, rather than relying
   on whatever Metal's default happens to be. **Rule for the real port**:
   every geometry descriptor sets `opaque` explicitly, full stop — the
   same "design the ambiguity out, don't reason through the default"
   lesson section 9 already drew for struct packing, now for a second,
   unrelated API surface.
2. **An intersection function's `[[intersection(...)]]` tag list must
   match the calling `intersector<...>`/`intersection_function_table<...>`
   tags exactly — not just declare the primitive type it handles.**
   `[[intersection(bounding_box)]]` compiles cleanly and *looks* complete
   (it names the right primitive type), but silently never gets dispatched
   at trace time when the intersector/table were declared with additional
   tags (`instancing`, `triangle_data`) that the function itself doesn't
   also declare. This is the more dangerous of the two bugs: it fails
   *silently* — no compile error, no runtime validation error even with
   `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` both enabled, just zero
   hits ever reported, indistinguishable from "the geometry genuinely
   isn't there" until isolated by bisection (confirmed via an
   unconditional-accept version of the function, which *still* produced
   zero hits until the tags were fixed — ruling out the geometric test
   itself before the tag mismatch was even suspected). Fixed:
   `[[intersection(bounding_box, triangle_data, instancing)]]`, tags
   copied verbatim from the intersector's own declaration.

Both fixes are now load-bearing comments in `metal_poc.metal`/`.mm`
directly, not just here — the real port should treat "geometry opacity"
and "intersection function tag parity" as two explicit checklist items
when any custom primitive enters the scene, not incidental details to
rediscover by the same bisection process this POC needed.
