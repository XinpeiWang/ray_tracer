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

## 12. Proof-of-concept, step 4: a real area light (done)

Every step through #11 used a single hardcoded directional (delta) light.
This step replaced it entirely with an actual area light: a small emissive
quad hanging just under the ceiling, sampled with proper area-to-solid-
angle PDF conversion, shadow-tested up to just short of the light's own
surface rather than to infinity. The light is real scene geometry (added
via the same `addQuad()` every other surface uses, now carrying a nonzero
`emission` field) rather than a special-cased light type - visible
directly to camera/GI rays that land on it, exactly like every other
project's own CPU integrator already treats area lights.

**Result, visually confirmed**: a visibly glowing ceiling panel, soft
shadow falloff (bright directly under the light, dimming with distance -
a delta light can't produce this at all, since every point either sees it
or doesn't), correct occlusion from both the sphere and mirror, and a
visible specular highlight of the light itself reflected in the glass
sphere's surface. 700×700 @ 512spp @ depth 10 renders in ~9.9 seconds on
an M2.

This one didn't surface a new API gotcha the way sections 9 and 11 did -
area-light NEE is pure math (uniform sample the quad, convert its area
PDF to solid angle via `distance² / (lightArea · cosθ_light)`, MIS-free
single-strategy NEE plus an unconditional emission check on every hit for
the indirect/direct-view case) layered on infrastructure sections 8-11
already proved works (shadow rays, the intersector, per-primitive material
data). Worth calling out for the opposite reason: not every increment in
a port this size hits new API territory - some are exactly as
straightforward as the underlying math suggests, and it's worth not
over-discounting those against effort estimates just because the harder
ones make for more interesting reading.

**What this still doesn't have**, worth being explicit about since a
glowing rectangle can look deceptively complete: no multiple-importance
sampling (BSDF-sampled rays that happen to hit the light aren't PDF-
weighted against the light-sampling strategy, just added at full value -
correct but higher-variance than MIS would give), no light-list
abstraction (the light's shape is hardcoded twice, once in each file, by
hand-kept-in-sync constants - a real port needs the CPU renderer's own
light-sampler abstraction, not a hardcoded quad), and still only one
light.

## 13. Proof-of-concept, step 5: multiple importance sampling (done)

Step 4 (above) explicitly flagged its NEE-only lighting as "correct but
higher-variance than MIS would give." This step closed that gap: the
light quad can now be reached two ways per bounce - explicit light
sampling (NEE) or landing on it by chance via a Lambertian BSDF-sampled
continuation ray - and both are power-heuristic-weighted (beta=2, the
same exponent this project's own CPU/GPU-OptiX integrators use per
`docs/FEATURE_INVENTORY.md`) instead of either double-counted or only one
strategy used. A `specularBounce`/`bsdfPdf` pair of loop-carried state
variables track whether the *previous* bounce was a Lambertian BSDF
sample (mirror/glass bounces and the camera ray itself always count as
"specular" for MIS purposes - no competing NEE sample could have produced
that exact hit, so their direct light-hits stay full weight, unweighted).

**Result, quantitatively confirmed, not just visually plausible**: an A/B
render at matched 8 samples/pixel - pre-MIS state (PR #6) vs. this step,
same scene, same seed pattern - shows visibly less noise in the MIS
version, most noticeably near the ceiling/light area where the NEE-only
version's variance was worst. This is exactly the outcome MIS is supposed
to produce, confirmed by direct comparison rather than assumed from the
math being textbook-correct.

Unlike step 4, this one also didn't need a new Metal API - it's a
restructuring of loop-carried state plus one more solid-angle PDF
evaluation (reusing the exact conversion step 4 already established) at
the point a BSDF-sampled ray happens to land on the light. Two changes
this size in a row without a new gotcha is itself informative for the
effort estimates in section 6: once the *infrastructure* (accel
structures, the intersector, per-primitive data, shadow rays) is in
place, some real integrator features really are as incremental as the
underlying algorithm suggests - the expensive, unpredictable part of this
POC so far has consistently been new Metal API surface (custom
primitives, intersection function tables), not new rendering math on top
of surface already proven to work.

## 14. Proof-of-concept, step 6: loading a real mesh (done)

Every object through step 5 was either hand-authored (quads, listed
vertex-by-vertex in source) or analytic (the sphere, defined by a centre
and radius, no vertex data at all). This step added the first genuinely
**data-driven** geometry: `models/suzanne.obj` (Blender's monkey mascot,
507 vertices, 500 faces - mostly quads, fan-triangulated to 968 triangles)
loaded through a minimal hand-written OBJ parser (`v`/`f` only - no
texcoords, materials, groups, or `vn`-driven smooth shading; every
triangle still gets the same flat face-normal treatment as every other
object in the scene) and auto-fit (bounding-box-normalized scale +
recentring) into the room, replacing the earlier flat "mirror test" quad
(mirror-MATERIAL coverage was already proven in #5's screenshots; what
this scene hadn't tested was real mesh SCALE and DATA-DRIVEN vertex
positions, not hand-listed axis-aligned corners).

**Result, visually confirmed**: unmistakably Suzanne, correctly flat-
shaded (visible per-triangle facets on the ears/cheeks/forehead - the
expected look without normal interpolation, not a bug), correctly
occluded behind/beside the glass sphere, and correctly refracted-through
where the sphere overlaps it. The acceleration structure now holds ~980
real triangles (vs. 14 in every prior step), built and traced with no
special-casing beyond "more triangles in the same vertex/material
buffers" - `MTLAccelerationStructureTriangleGeometryDescriptor` scales to
mesh-sized geometry the same way it handled a dozen hand-authored
triangles, exactly as expected (Section 3 rated this "Low" risk, and nothing
here changes that rating). 700×700 @ 256spp @ depth 10 renders in ~9.0
seconds on an M2 - triangle count roughly 70x higher than step 5, wall-
clock time barely higher, which is itself a useful confirmation that the
bottleneck in this scene is sample count/bounce depth, not primitive
count, at least at this modest scale.

**What this still isn't**: a real scene *loader* in the
`pbrt_gpu_builder.h`/`scene_builder.cpp` sense - no materials-per-face, no
smooth (vertex-normal-interpolated) shading, no instancing, no texture
coordinates, and it only reads Wavefront OBJ, not this project's actual
`.pbrt` scene format at all. Section 3's "Medium-High, mostly from sheer
size" rating for the real scene builder stands unchanged - this step
de-risks "can an arbitrary real mesh's vertex data reach a Metal
acceleration structure and render correctly," not "can this project's own
scene format be loaded," which remains a substantially larger, separate
piece of work.

## 15. Proof-of-concept, step 7: smooth (per-vertex-normal-interpolated) shading (done)

Step 6 explicitly flagged Suzanne rendering faceted as "a real (if
visually rougher) limitation, not a bug," deferred for exactly this
reason: it needed barycentric-coordinate plumbing the POC didn't have
yet. This step added it: `intersection_result<instancing, triangle_data>`
already carries `triangle_barycentric_coord` for a triangle hit - the
`triangle_data` tag has been on every `intersect()` call since the very
first working version of this POC, previously read only for
`primitive_id`. A new per-triangle-corner normal buffer (parallel to the
vertex buffer, same indexing) feeds
a barycentric blend (`shadingNormalFor()`) that replaces the flat
cross-product face normal used everywhere until now.

The OBJ loader now parses `vn` and each face token's own normal index
(`v//vn`), falling back to a computed flat normal per-triangle when a
face is missing one (a real files-are-inconsistent case the loader
handles rather than assumes away) - `suzanne.obj` turned out to have `vn`
on every one of its 500 faces, so this render uses zero fallbacks, all
real interpolated data. The hand-authored room quads get the *same* new
normal buffer, just with all three corners of each triangle carrying an
identical value (that quad's own flat face normal) - interpolating three
identical values trivially reproduces the old flat-shading result, so
every previously-correct object in the scene stays pixel-plausible-
identical; only Suzanne, the one object with genuinely different per-
corner values, looks different.

**Result, visually confirmed, and it's a real difference**: Suzanne now
renders with smooth, continuously-curving surfaces - no visible per-
triangle facets anywhere, including on the previously most-faceted areas
(ears, cheeks, forehead). Same 968 triangles as step 6, same scene,
same sample count - the *only* change is which normal gets used at each
shaded point. 700×700 @ 256spp @ depth 10 renders in ~9.9 seconds on an
M2, statistically indistinguishable in cost from step 6's ~9.0 seconds -
confirming (as step 6 itself predicted) that per-triangle-corner data
lookups and one extra buffer read are not where this scene's cost lives.

This is also the cleanest confirmation yet of section 13's observation
about where this POC's real costs are: `triangle_barycentric_coord` was
already sitting on every `intersection_result` this POC has produced
since the first working triangle intersection, unused. Turning it into a
visible feature took one new helper function, one new host-side buffer,
and reusing the existing per-corner indexing convention - genuinely
incremental, no new Metal API surface, no debugging odyssey like sections
9 and 11's custom-primitive work needed.

## 16. Proof-of-concept, step 8: texture mapping (done)

Every material so far has been a flat, hardcoded colour. This step adds
a fourth material type (`materialType == 3`, textured Lambertian) that
samples a real image instead: the back wall now shows `images/
earthmap.jpg`, decoded via the already-vendored `stb_image.h` (no new
third-party dependency - `metal_poc.mm` already carried `stb_image_write.h`
for PNG output since step 1; adding the matching read-side header is a
one-line include, `STB_IMAGE_IMPLEMENTATION`/`STB_IMAGE_WRITE_IMPLEMENTATION`
each guard their own single translation unit so there's no ODR conflict
having both in one file).

This is genuinely new Metal API surface for this POC: `texture2d<float,
access::sample>` as a kernel parameter, a `constexpr sampler` (`coord::
normalized, address::repeat, filter::linear`), and `MTLTexture` host-side
upload via `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` - none of
the previous seven steps touched texture objects at all (`outTexture`,
present since step 1, is `access::write`-only, the render target, not a
sampled input). UV coordinates needed their own new per-triangle-corner
buffer (`uvs`, buffer index 8), following the exact same indexing
convention steps 7's normal buffer established: parallel to `vertices`,
one value per triangle corner, read back in the shader via a new
`texCoordFor()` barycentric-interpolation helper that's structurally
identical to `shadingNormalFor()` but blends `float2` instead of `float3`.

`addQuad()` now generates a standard planar `(0,0)-(1,0)-(1,1)-(0,1)`
UV mapping across its four corners for every quad it builds (not just the
one that ends up textured) - harmless, since the shader only ever reads
`uvs` when `materialType == 3`, and it means any future quad can become
textured by changing one argument rather than re-deriving UVs later.
`loadObjMesh()` gained the same `uvs` output parameter for buffer-layout
parity, but pushes all-zero filler: `suzanne.obj` has no `vt` data and
stays `materialType 0`, so those UVs are declared but never sampled -
parsing real `vt`/`f v/vt/vn` tokens was judged out of scope for this
increment (the OBJ loader would need a third fan-triangulation-surviving
index alongside `posIdx`/`normalIdx`, doable but a separate, cleaner change
than bolting it onto a same-PR texture increment).

**Result, visually confirmed**: the back wall renders the Earth map
correctly - continents, cloud bands, and the equirectangular grid lines
all sharp and correctly oriented, with `address::repeat` + `filter::linear`
producing clean bilinear sampling and no visible seams. The glass sphere's
refraction/reflection correctly picks up the textured wall too, confirming
the texture is read through the same BSDF/path-tracing machinery as every
other material, not a separate special-cased code path. 700×700 @ 256spp @
depth 10 renders in ~8.5 seconds on an M2 - the texture sample itself adds
no measurable cost over step 7's ~9.9s (if anything faster, within this
POC's run-to-run noise, since a single `.sample()` call per bounce is
cheap relative to everything else already happening per-ray).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and run correctly, and both resolve `images/earthmap.jpg` via the
same `RT_MODELS_DIR`-relative-sibling-directory pattern (`RT_MODELS_DIR`
points at `<repo>/models`; `images/` is `<repo>/images`, so `..`-ing up
one level from `RT_MODELS_DIR` and appending `images` reaches it from an
out-of-tree build directory the same way the existing `RT_METAL_SHADER_DIR`/
`RT_MODELS_DIR` definitions already do for the shader source and Suzanne).
A missing-file fallback (a solid white 1×1 texture) keeps the shader's
always-bound texture slot valid even if `earthmap.jpg` can't be found,
rather than crashing or reading undefined data.

## 17. Proof-of-concept, step 9: a rough (GGX) conductor material (done)

Every specular-ish surface so far has been a delta BSDF (mirror, glass) -
sampled with a single deterministic direction, no NEE, because a delta
lobe has zero probability of a shadow ray landing exactly on it. This step
adds the first *glossy* (non-delta, non-diffuse) material: `materialType
== 4`, a rough conductor using the Trowbridge-Reitz/GGX microfacet
distribution with height-correlated Smith masking-shadowing, matching
pbrt-v4's own `TrowbridgeReitzDistribution` formulation (see
[`docs/FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md)'s materials table).
Unlike the previous two "no new API surface" steps, this one is a
genuinely new *algorithm* addition, not new Metal API surface - it's still
the same inline `intersector`/kernel machinery, applied to more elaborate
BSDF math.

`TriangleMaterial::ior` is reused as a perceptual roughness value for this
material type (squared into the GGX `alpha` parameter, the same
roughness→alpha remap pbrt-v4 uses) - dielectric IOR and conductor
roughness never coexist on one primitive, so sharing the slot avoids
adding a field that would sit unused on every other material. `color`
becomes the conductor's F0 (reflectance at normal incidence, an RGB
colour for a metal, not a scalar IOR) - Fresnel is Schlick-approximated
from it, the same "Schlick, not the full complex-IOR Fresnel equations"
simplification the existing dielectric material already makes.

Sampling uses Heitz (2018)'s "Sampling the GGX Distribution of Visible
Normals" (VNDF) rather than naive distribution-only importance sampling -
substantially lower variance at grazing angles, and what pbrt-v4's own
`Sample_wm` implements, so this isn't a simplified stand-in but the same
algorithm a production path tracer uses. The BSDF-sampled-continuation
weight (`f(wo,wi) * cosI / pdf(wi)`) collapses algebraically to `F *
G(wo,wi) / G1(wo)` for a VNDF-sampled direction - the distribution term
and the `4 * NdotO * NdotI` denominator cancel exactly against the same
terms in the Jacobian-converted pdf, leaving only Fresnel and a masking-
shadowing ratio. NEE against the area light evaluates the full BRDF (`D *
G * F / (4 * NdotO * NdotI)`) for the light-sampled direction and MIS-
weights it via the power heuristic against the BSDF strategy's own pdf for
that same direction - structurally identical to the Lambertian branch's
NEE/MIS shape, just with the microfacet math swapping in for the cosine-
weighted diffuse lobe.

The scene gained a second sphere (a gold-ish rough conductor, roughness
0.15) alongside the existing glass one - proving multiple custom
primitives share one bounding-box geometry/intersection function cleanly:
`sphereIntersectionFunction` already indexed into its `spheres` buffer by
`primitive_id`, so going from one sphere to an array of two was purely a
host-side change (`boundingBoxCount`, buffer sizes), no shader-side
modification at all.

**Result, visually confirmed**: the gold sphere shows blurred, glossy
reflections of the room (not perfect-mirror-sharp, not diffuse-flat) with
a clear specular highlight from the area light, correctly tinted toward
gold rather than reflecting colours at full saturation. An A/B render at
roughness 0.6 (vs. the committed 0.15) confirms the lobe genuinely widens
with roughness - the same sphere goes from a recognizable blurred
reflection of the red/green walls to a soft, almost-diffuse-looking
highlight with no distinguishable wall reflections at all, exactly the
qualitative behaviour a correct GGX implementation should show. 700×700 @
256spp @ depth 10 renders in ~14 seconds on an M2 - up from step 8's
~8.5s, the expected cost of a second NEE shadow ray + full BRDF evaluation
per bounce for the new sphere, plus one more object's intersections
across the whole scene, not a regression in the existing materials' own
cost.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 18. Proof-of-concept, step 10: a real light list (done)

Every step through 17 carried exactly one hardcoded light - `kLightCenter`/
`kLightHalfExtents`/`kLightNormal`/`kLightArea`/`kLightEmission` constants
baked directly into the shading kernel, with steps 4/5's own comments
explicitly flagging this as a known simplification: "a real port would
carry a proper light list... instead of one hardcoded light's shape baked
into the integrator." This step replaces those constants with an actual
`AreaLight` buffer (`center`/`edgeU`/`edgeV`/`normal`/`area`/`emission`,
precomputed host-side per light) and a `uniforms.lightCount`, matching the
abstraction this project's own CPU `src/TheRestOfYourLife/*light_sampler*.h`
already is.

NEE now **picks a light uniformly at random** from the list before
sampling a point on it (`sampleAreaLight()`, factored out once and shared
between the Lambertian and GGX-conductor branches, which previously had
their own near-duplicate copies of the single-light sampling code) - the
same one-sample-MIS-over-a-light-list approach pbrt-v4's own
`UniformLightSampler` uses, with the `1/lightCount` pick probability
folded into the existing area-to-solid-angle PDF conversion. The harder
half of this change is the OTHER direction: when a camera/GI ray lands
directly on an emissive triangle (rather than via NEE), the MIS weight
needs to know exactly which light that triangle belongs to, to look up
the right area/normal for the competing NEE strategy's pdf. Solved with a
new `TriangleMaterial::lightId` field (index into `lights`, -1 for every
non-emissive material) - set once, host-side, when a light's own quad is
authored (`addAreaLight()`'s lambda keeps a light's geometry, material
tag, and `AreaLight` buffer entry in sync by construction, instead of
needing hand-synchronized constants the way the single-light version's
comment on `kLightCenter`/etc. warned against).

The scene now has **two** lights (a warm one and a cool one, side by
side under the ceiling) instead of one - the smallest change that
actually exercises `lights` as a genuine list rather than a renamed
single constant, and one deliberately chosen to be visually falsifiable:
a bug in light-picking, per-light pdf, or the `lightId`-based MIS lookup
would very plausibly still "render something," just wrong (missing one
light's contribution, double-counting, or an incorrectly-biased result
that undersamples one light) - the same "looks plausible, is subtly
wrong" failure mode Section 4 calls out as the highest-risk bug class
for a from-scratch backend.

**Result, visually confirmed**: both the rough-conductor and dielectric
spheres show two distinct specular highlights (previously one), each
reflecting one of the two lights, and the ceiling shows two separate lit
panels rather than one - direct, easily-inspectable proof that both
lights are being intersected, sampled, and shaded independently rather
than one light silently dominating or the second light's contribution
going missing. 700×700 @ 256spp @ depth 10 renders in ~14 seconds on an
M2, statistically indistinguishable from step 9's own ~14s (same total
NEE ray count per bounce - one shadow ray either way, just toward a
randomly-chosen light instead of the sole light - so no cost increase
was expected).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 19. Proof-of-concept, step 11: thin-lens depth of field (done)

A lighter increment after the last two heavier ones (a new BSDF algorithm
in step 9, a structural light-list rewrite in step 10) - pure camera
math, touching only primary-ray generation, no new buffers, no shading-
loop changes. `Uniforms` gained `lensRadius`/`focusDistance`; the primary
ray's origin gets jittered across a simulated circular aperture
(`sampleUnitDisk()`, area-uniform via `r = sqrt(u1)`) and re-aimed through
the same fixed point on the focus plane the original pinhole ray would
have hit - the standard thin-lens camera model (same one pbrt-v4's
`PerspectiveCamera` and this project's own CPU `camera.h` implement).
`lensRadius == 0` skips the whole block, so this is strictly additive:
every earlier PR's pinhole camera behaviour is still reachable exactly,
not replaced.

The scene's camera now focuses on the gold conductor sphere (the nearest
object to the camera) with `lensRadius = 0.05`.

**Result, visually confirmed two ways**: first, an A/B render at
`lensRadius = 0` reproduces the original pinhole-sharp image essentially
identically (confirming the DOF code path is a true no-op when disabled,
not just visually close). Second, the depth-of-field render itself shows
the expected FALLOFF, not a uniform blur: the gold sphere at the focus
distance stays sharp, the dielectric sphere only ~0.3 units further back
stays nearly sharp, and the back wall/Suzanne - both well over a unit
further from the camera - show clearly increasing defocus blur the
further they are from the focus plane. That falloff shape (blur radius
scaling with distance from the focus plane, not a flat per-pixel blur) is
exactly what a real lens model produces and a fake post-process blur
would not, without deriving depth per pixel and doing the same math this
kernel already does inline.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 20. Proof-of-concept, step 12: real OBJ `vt` (UV) parsing (done)

The texture-mapping PR (step 8) explicitly deferred this: "parsing real
`vt`/`f v/vt/vn` tokens was judged out of scope for this increment," since
`suzanne.obj` - this POC's only mesh at the time - has no `vt` data at
all. This closes it: `loadObjMesh()` now parses `vt` lines and each face
token's own `vt` index (`v/vt` or `v/vt/vn`), the same per-corner-index-
survives-fan-triangulation approach the existing `vn` parsing already
established, falling back to `(0,0)` (tracked via a new
`uvFallbackCount`, mirroring `normalFallbackCount`'s own logging) when a
face is missing one.

Testing this needed a real mesh with actual `vt` data - `suzanne.obj` has
none, so `models/spot.obj` (Keenan Crane's textured cow model, a common
graphics-teaching asset, 5856 triangles, 3225 `vt` entries) was added to
the scene instead. It turns out to be the exact inverse case from
Suzanne: full `vt` coverage, **zero** `vn` data at all - loading it
exercises both `loadObjMesh()`'s real-UV path and its flat-normal-
fallback path in the same call, and confirms both meshes' own load logs
now show complementary fallback counts (`0 flat-normal fallback, 968
zero-uv fallback` for Suzanne; `5856 flat-normal fallback, 0 zero-uv
fallback` for Spot) - directly-inspectable proof neither path is
silently going unused. Spot is given `materialType 3` and reuses
`earthTexture` (`images/earthmap.jpg`) - wrapping a world map onto a cow
was never the texture's intended use, which is exactly what makes it a
meaningful test: a coincidentally-plausible result is much less likely
than with a texture actually designed for the mesh, so seeing the map's
coastlines and grid lines correctly follow the body's curvature (not
stretched, swum, or offset) is real evidence the per-corner UV data and
`texCoordFor()`'s barycentric interpolation are both working on genuine
mesh data, not just the hand-authored planar quad UVs step 8 originally
verified.

**Result, visually confirmed**: Spot renders with the earthmap texture
correctly wrapped around its curved body and head, following the mesh's
actual surface rather than looking flat-projected or misaligned. Placing
it required care - an early attempt positioned it almost directly behind
the dielectric sphere along the camera's sightline and it was nearly
invisible, caught by inspecting the render rather than assuming the
scene-graph math was right; the corrected position both models sit in
the same frame without one being an obviously camera-angle-dependent
accident. 700×700 @ 256spp @ depth 10 renders in ~15 seconds on an M2, up
slightly from step 11's ~14s - expected, the scene's triangle count
roughly septupled (982 -> 6838) with Spot's 5856 triangles, though BVH
traversal cost grows sub-linearly with triangle count, not the near-7x
the raw count increase might suggest.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 21. Proof-of-concept, step 13: a rough (frosted) dielectric material (done)

`materialType 5` extends the existing smooth dielectric (`materialType
2`)'s own Schlick-Fresnel reflect-vs-refract logic to a rough interface:
the reflect/refract decision and directions are computed about a GGX-
VNDF-sampled microfacet normal (`sampleGGXVNDF()`, the same helper the
conductor material uses) instead of the smooth geometric normal - the
standard way a rough surface's normal gets perturbed. A new
`TriangleMaterial::roughness` field was added (rather than reusing `ior`'s
slot the way materialType 4 does) since a rough dielectric genuinely
needs both a real refraction index AND a roughness value on the same
primitive at once.

**A deliberate, explicitly-documented simplification, not an oversight**:
this does NOT implement a full energy-conserving rough-BTDF derivation
the way the conductor material's own `G/G1(wo)` throughput correction did.
A correct one needs a transmission Jacobian plus an eta² radiance-scaling
term (Walter et al. 2007's rough refraction model) on top of the
reflection-side masking-shadowing ratio - real, genuinely tricky-to-get-
right math with no reference implementation in this codebase to check
results against, exactly the "looks plausible, renders something, is
subtly wrong" trap Section 4 warns about for a from-scratch backend.
Rather than ship that unverified, this keeps the smooth dielectric
branch's existing throughput accounting (plain `albedo`, no
masking-shadowing correction) and treats roughness as a direction-only
perturbation - the same kind of documented, deliberate approximation this
POC's Schlick-vs-full-Fresnel choice already represents, not a new kind
of shortcut.

A third sphere (small, `roughness = 0.35`) was added to the scene for
this. Placing it took two attempts: the first position (tucked in the
back-left corner) turned out to sit almost exactly along the camera-to-
gold-sphere sightline (both roughly 20° off-axis, with the much larger,
closer gold sphere fully hiding it) - a 3D bounding-region overlap check
alone wouldn't have caught this, since the two objects don't actually
intersect in space, only in 2D screen projection from this one camera
angle. Caught by rendering and inspecting the image rather than trusting
the placement math, the same lesson spot.obj's own placement (step 12)
surfaced.

**Result, visually confirmed two ways**: first, an A/B render at
`roughness = 0` reproduces materialType 2's own perfectly clear, sharp
glass sphere look - confirming `sampleGGXVNDF()` genuinely degenerates to
the exact geometric normal at zero roughness (traced through the
sampling math: at `alpha == 0`, the VNDF sample collapses to `(0,0,1)` in
the local frame regardless of the random numbers drawn, i.e. `hWorld ==
facingNormal` exactly), not just a visually-close approximation. Second,
the committed `roughness = 0.35` render shows the expected frosted-glass
character - a soft, milky, blurred-transmission look, clearly distinct
from both the sharp dielectric sphere and the diffuse Lambertian
materials elsewhere in the scene. 700×700 @ 256spp @ depth 10 renders in
~16 seconds on an M2, close to step 12's ~15s (one more, small, primitive
- not a meaningfully more expensive shading path than the smooth
dielectric it's based on).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.
