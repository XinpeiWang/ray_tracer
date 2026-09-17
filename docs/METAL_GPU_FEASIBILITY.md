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

## 22. Proof-of-concept, step 14: camera (shutter) motion blur (done)

A camera translation (`Uniforms::cameraVelocity`, world-space, full
displacement over the frame's simulated [0,1] shutter interval) applied
to the primary ray's origin, weighted by a per-SAMPLE uniform-random
shutter time drawn the same way the existing pixel jitter and DOF lens
sample already are - different samples for the same pixel see the camera
at different points along its path, and the multi-sample averaging loop
every other feature in this POC already reuses is what produces the
blur, no separate accumulation/reprojection pass needed.
`cameraVelocity == (0,0,0)` (every earlier PR's own scenes) makes every
sample use the identical origin regardless of its sampled time - the
original static camera exactly, purely additive like `lensRadius == 0`.

**Deliberately scoped to CAMERA motion, not object motion**: a moving
custom (bounding-box) primitive would need the per-sample shutter time
threaded into `sphereIntersectionFunction` itself, which has its own,
separate argument table (bound via `MTLIntersectionFunctionTable`, not
shared with the calling kernel's buffers - documented back in section 11)
- doable, but genuinely new, uninvestigated Metal API surface (whether/how
a custom intersection function's `[[buffer(N)]]` table can carry a value
that varies per-thread per-sample, not just per-primitive) rather than a
known quantity. Camera motion blur needed none of that: the whole effect
lives in primary-ray generation, touching nothing downstream of it - a
deliberately similar risk profile to step 11's own DOF addition.

Tuning the velocity's magnitude took two passes: the initial value
(0.12 world units over the shutter) produced an almost illegibly smeared
frame - the room is only ~2 units across and the camera is several units
away, so even a "small-looking" translation is a large angular sweep at
that scale. Settled on 0.015 for the committed scene, subtle enough to
read as a deliberate stylistic choice layered on top of the existing
depth-of-field blur rather than looking like a broken/noisy render.

**Result, verified two ways**: an isolated render (motion blur only,
`lensRadius = 0`) at a larger velocity (0.06, chosen purely to make the
effect unambiguous for this one verification image) shows a horizontal
smear across the ENTIRE frame regardless of depth - correctly distinct
from DOF's own depth-DEPENDENT blur (which leaves the focus plane sharp
and blurs progressively with distance from it), confirming this is
genuinely a different blur mechanism, not DOF's code path accidentally
doing double duty. The committed combined render (motion blur + DOF
together, both live in the same default scene now) shows a subtle
additional directional streak on top of the existing focus falloff,
most visible in the specular highlights on the metal/glass spheres and
the ceiling lights' own edges - exactly where motion blur reads most
clearly in real photography and other renderers. 700×700 @ 256spp @
depth 10 renders in ~16 seconds on an M2, statistically indistinguishable
from step 13's own ~16s (one more scalar multiply-add in ray generation,
not a new per-bounce cost).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 23. POC series status: what's been proven, and what a real port still needs

Sections 8-22 cover fourteen incremental steps (PRs #2-#16) built on top of
this document's own Section 7 "suggested next step" - a triangle + one-
material spike to validate the pipeline shape. That original, deliberately
minimal goal has been substantially exceeded; this section is the
synthesis Section 7 didn't have the hindsight to write, laying out what
that now actually adds up to and, as importantly, what it doesn't.

**What's been proven, concretely, on real Apple Silicon hardware:**

- **Geometry**: triangle meshes (hand-authored quads AND real loaded
  `.obj` files, with genuine per-vertex normals and UVs, smooth-shaded)
  mixed with a custom (non-triangle) bounding-box-intersected primitive
  (spheres, 1-N of them sharing one intersection function) in a single
  `instance_acceleration_structure` - Section 3's "Low risk" (triangle
  BVH) and "Medium risk" (custom-primitive intersection) predictions both
  held up, the second only after finding the two real bugs Section 11
  documents.
- **Materials**: Lambertian diffuse (flat AND textured), mirror, smooth
  dielectric, rough (frosted) dielectric, and a physically-based rough
  conductor (GGX/Trowbridge-Reitz with VNDF importance sampling and
  height-correlated Smith masking-shadowing) - six materialTypes
  spanning delta, glossy, and diffuse BSDF families, not just "one
  material to prove the pipeline."
- **Lighting**: a genuine multi-entry light LIST (not a single hardcoded
  light) with next-event estimation, uniform light-picking, and multiple
  importance sampling (power heuristic) between light- and BSDF-sampling
  strategies, correctly generalized across every non-delta material.
- **Camera**: thin-lens depth of field and shutter-time motion blur, both
  purely additive to the base pinhole/static-camera path.
- **Verified correctness methodology, not just "it renders something"**:
  A/B comparisons that toggle a new parameter to its degenerate value and
  confirm the OLD behaviour reproduces exactly (DOF's `lensRadius == 0`,
  motion blur's `cameraVelocity == 0`, rough dielectric's `roughness ==
  0`, MIS's own noise-comparison in step 5); load-time fallback counters
  that prove both a real-data code path and its fallback path actually
  execute (step 12's complementary Suzanne/Spot logs); visual
  falsifiability chosen deliberately when possible (step 10's two-light
  scene, chosen specifically because a light-list bug would plausibly
  still "render something," just wrong).

**What this explicitly is NOT, and Section 6's effort estimates for each
remain the right scale (weeks-to-months, not PR-sized) even now:**

- **Not the wavefront architecture.** Every step above still runs inside
  one compute kernel per sample (`primaryRayKernel`'s own bounce loop),
  not the queue-of-separate-compute-passes (intersect/shade/shadow/
  accumulate) design Section 2 identified as the real architectural
  target for a production port. This POC proves the underlying
  primitives (inline intersection, custom primitives, buffers) that a
  wavefront restructuring would be built FROM, not the restructuring
  itself.
- **Not scene-builder-integrated.** Every scene in this POC is hardcoded
  host-side C++ (`main()`'s own `addQuad()`/`loadObjMesh()`/sphere-array
  calls) - there is no path from a real `.pbrt` file to this renderer;
  `scene_builder.cpp`/`pbrt_gpu_builder.h`'s own 7,500+ lines remain
  entirely unaddressed, still Section 3's own "Medium-High" risk item.
- **Not app-integrated.** `metal_poc` remains a standalone CLI tool with
  its own `main()`, never called from `launcher/main.cpp` or the Qt GUI -
  by design (Section 5 point 5), but worth restating plainly: there is no
  `--gpu-metal` flag on the real `ray_tracer` binary.
- **Missing real feature-parity breadth**, even setting the above aside:
  no BSSRDF/subsurface scattering, no spectral/hero-wavelength rendering,
  no hair BSDF, no measured/tabulated BRDFs, no volumetric media, no
  additional shape types (disks, cylinders, bilinear patches), no
  ReSTIR DI/GI, no SVGF denoising, no NRC, no OptiX-AI-Denoiser
  equivalent - the entirety of Section 3's "Medium" through "Dead end
  as-is" rows beyond what's listed above as proven.

**Bottom line**: the architecture-level bets Sections 2-3 made have now
been tested across a genuinely broad slice of this project's own material/
light/shape/camera feature space, on real hardware, with real bugs found
and fixed along the way (Section 11's two, this section's own list of
verification methodology) - not just the single triangle + one material
Section 7 originally called for. That substantially de-risks a real port
at the ARCHITECTURE level. It does not shrink Section 6's own effort
estimates for what's still ahead: the wavefront restructuring, scene-
builder integration, and remaining feature-parity work are each
independently a multi-week-to-months undertaking, not a continuation of
this same PR-sized incremental pattern.

## 24. Proof-of-concept, step 15: non-identity instance transforms (done)

Every instance in every earlier step used the identity transform -
`instanceDescs[i].transformationMatrix` was always the same hardcoded
identity matrix, for both the room+mesh geometry and the sphere
primitives. A real port would place many object instances with real
per-instance transforms (pbrt's own `ObjectInstance` is exactly this),
and this POC had never actually exercised that: every "different
position" in every earlier scene was really a different WORLD-SPACE
vertex position baked in at load time, not an instance transform doing
real work.

Suzanne now gets her own, separate primitive acceleration structure
(previously her geometry was merged into the same buffer/AS as the room
quads and Spot) loaded once in OBJECT space (centred at the origin), then
referenced by **two** instance descriptors with **different** transforms:
instance A is a plain translation back to her original world position (a
direct continuation of every earlier screenshot's own placement, an
identity-rotation sanity check by construction); instance B is rotated 45
degrees about Y, uniformly scaled down, and translated to a new floating
position near the ceiling. One GPU-resident BVH, reused from two
different world-space placements - the actual point of instancing, not
demonstrated by anything in this POC before now.

**The real new piece, and the one Metal doesn't hand you for free**:
`intersection_result<instancing, triangle_data>` does NOT expose the
hit instance's own object-to-world transform as a queryable field -
confirmed by directly probing the compiler (`object_to_world_transform`,
`world_to_object_transform`, and several other plausible names all fail
with "no member named..."), unlike OptiX's own
`optixGetWorldToObjectTransformMatrix()`. Object-space per-vertex normals
still need transforming into world space for correct shading on a
rotated instance, so this POC built its own side-channel: an
`InstanceTransform` buffer (one entry per `instanceDescs[]` slot,
mirroring `MTLPackedFloat4x3`'s own 4-packed-column layout byte-for-
byte), indexed in the shader by `intersection_result::instance_id`
(which IS exposed) and populated host-side from the exact same
column/translation values used to build each instance descriptor's own
`transformationMatrix` - by construction, via one shared `addInstance()`
helper, so the two copies of the same transform can't drift apart. Only
the 3x3 linear part is applied to a normal (translation is meaningless
for a direction); this assumes a RIGID transform (rotation + uniform
scale, no shear/non-uniform scale) rather than implementing the general
inverse-transpose case - a real one would need that, but every transform
this POC's own scene ever constructs is rigid, and that's documented
rather than silently assumed away.

Placing the rotated instance needed two attempts, the same lesson
sections 20/21 already ran into: the first position (back-left corner,
floor level) sat almost exactly along the camera-to-gold-sphere sightline
and was nearly invisible - a 3D-non-overlap check doesn't catch 2D
screen-space occlusion from one particular camera angle. Moved to a
floating position near the ceiling instead.

**Result, verified two ways**: visually, the floating instance reads as
a clearly separate, smaller, rotated duplicate of the same face/mascot
shape sitting below it - unmistakably the same mesh, unmistakably a
different placement, with no obviously-wrong (inverted/flipped) shading
that a normal-transform bug would produce. An A/B render with the
rotation angle set to 0 (translation + scale only) shows the floating
instance snap back to the SAME front-facing orientation as the ground
instance - confirming the rotation matrix construction is actually doing
what it claims, not just producing a plausible-looking blob. 700×700 @
256spp @ depth 10 renders in ~20 seconds on an M2, up from step 14's
~16s - a real cost increase this time, not noise: a second triangle
acceleration structure plus a second traversed instance adds genuine
BVH-traversal overhead, unlike several earlier steps whose "new" work
was pure shading-side math with no extra scene complexity.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 25. Proof-of-concept, step 16: a second custom-primitive shape (a disk) (done)

Every custom (non-triangle) primitive in this POC before now - however
many spheres accumulated across earlier steps - has gone through the
SAME intersection function, at the SAME function-table slot (0). Section
3's own risk table lists "Custom-primitive intersection (sphere/quad/
disk/cylinder/bilinear patch)" as one item covering five shape types;
this POC had only ever proven ONE of them, and more importantly, had
never proven the mechanism a real multi-shape renderer actually needs:
more than one DISTINCT intersection function living in the same function
table at once, dispatched by `geometryDescriptor.intersectionFunction
TableOffset`.

A disk primitive (plane intersection + in-plane radius check -
`diskIntersectionFunction`, textbook math, same shape this project's own
CPU `disk.h` uses in spirit) is added as a SECOND `geometryDescriptor`
within the SAME primitive acceleration structure the spheres already
use (Metal supports multiple heterogeneous geometries in one AS - this
wasn't previously exercised either), at function-table slot 1. Two real
things had to be gotten right that the single-function case never
needed to think about:

1. **A shared argument namespace.** Both intersection functions live in
   the SAME `MTLIntersectionFunctionTable`, and `setBuffer:atIndex:N`
   binds buffer N for the WHOLE table, not per-function - so
   `sphereIntersectionFunction` and `diskIntersectionFunction` had to
   declare DIFFERENT `[[buffer(N)]]` indices in their own MSL signatures
   (0 and 1) for their host-side bindings not to collide.
2. **Disambiguating hits.** Both a sphere hit and a disk hit report
   `intersection_type::bounding_box` - identical to each other from the
   calling kernel's point of view - so `isSphere = (type ==
   bounding_box)` (every earlier step's own check) stopped being
   sufficient. `intersection_result::geometry_id` (the geometryDescriptors
   array index within the hit AS: spheres at 0, disk at 1) is what
   actually tells them apart - the first time this POC has needed that
   field for anything.

The disk is also, incidentally, the first object in this ENTIRE scene to
actually use `materialType 1` (mirror) - it's existed in the shader since
the very first multi-material step, but nothing rendered had used it
since the original mirror test quad was replaced by the dielectric
sphere back in step 6.

**Result, visually confirmed**: a circular mirror mounted on the right
(green) wall, correctly showing a crisp circular silhouette (not the
AABB's own square bounding shape, confirming the radius check is doing
real work, not just accepting anything inside the bounding box), a
plausible mirror reflection of the room's ceiling/back-wall area, and a
correct soft shadow/darkening on the wall directly behind it. Both
spheres and the disk coexist correctly in the same render - direct
confirmation the two-slot function table dispatches each geometry to
its own correct intersection function rather than one silently
overriding or only-sometimes-invoking the other. 700×700 @ 256spp @
depth 10 renders in ~20 seconds on an M2, statistically indistinguishable
from step 15's own ~20s (one more small, cheap-to-intersect primitive,
not a meaningful cost addition).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 26. Proof-of-concept, step 17: a homogeneous participating medium (fog) (done)

Section 23's own "what this explicitly is NOT" list named volumetric
media as one of the pieces missing from this POC's material/light/shape
coverage. This adds the simplest real version of it: a single, uniform
fog filling the ENTIRE scene volume (not attached to any one object's
geometry), avoiding the boundary-tracking problem (detecting when a ray
enters/exits a fog volume's own bounds) a more general per-object medium
would need and this POC has no other reason to build yet.

The technique is textbook free-flight distance sampling: each bounce
draws a random scattering distance `t = -ln(1-u)/sigmaT` from the
medium's own exponential transmittance distribution and compares it to
the surface hit's own distance. This single stochastic comparison is
what makes BOTH outcomes come out unbiased with NO extra transmittance
multiplier needed in either branch - a well-known result worth stating
precisely since it's easy to get backwards: reaching the surface without
scattering needs weight 1 exactly (the survival probability under an
exponential distribution IS the transmittance, so they cancel), and a
scattering event needs weight `sigmaS/sigmaT` exactly (the scattering
coefficient over the SAME pdf that generated `t`, with the transmittance
terms cancelling the same way). Neither of those derivations was taken on
faith - both were worked through algebraically before being committed to
code, the same "don't ship unverified math" discipline step 13's own
rough-dielectric writeup argued for (and, unlike that step, this one
didn't need to fall back to a documented simplification - isotropic
scattering has no analogous non-conservative shortcut to worry about).

A scattering event spawns a continuation ray via `sampleUniformSphere()`
(a new, genuinely different sampling domain from every existing
BSDF/phase function in this POC - every other direction sampler is a
HEMISPHERE sampler oriented around a surface normal; a volume scattering
point has no surface/normal to orient around at all) with an isotropic
phase function (constant `1/(4*pi)`, no cosine term), and does its own
NEE against the light list using the same `sampleAreaLight()`/MIS
machinery every surface material's NEE branch already shares. Getting
this consistent needed one more fix: shadow rays are a separate,
deterministic occlusion test, not routed through the same free-flight
sampling as the primary ray, so they don't automatically account for the
medium's own attenuation along their length - an explicit
`exp(-sigmaT*shadowDist)` transmittance factor was added to EVERY NEE
contribution in the shader (the new volume one, and both existing
surface ones - Lambertian/textured and the GGX conductor), or a foggy
scene would have looked inconsistent: correctly hazy in-scattering, but
surfaces lit as if the fog wasn't there at all.

**Result, verified three ways**: an A/B render with `fogSigmaT = 0`
reproduces step 16's own committed render pixel-plausibly identically,
confirming the medium code path is a true no-op when disabled. The
committed scene (`sigmaT = 0.05`, a faint cool-tinted albedo) shows a
subtle, physically-plausible atmospheric haze - soft depth-based
darkening, most visible in the room's far corners and in the mirror
disk's own reflection, without fighting every other material's own
visibility. A third render at `sigmaT = 0.25` (five times stronger, for
verification only, not committed) shows a dramatically thicker,
unmistakable haze with much steeper depth falloff - confirming the
effect's STRENGTH scales with the physical parameter the way it should,
not just "some haze appears at some arbitrary fixed strength." 700×700 @
384spp @ depth 10 renders in ~46 seconds on an M2, up from step 16's
~20s at 256spp - a real, expected cost increase (an extra shadow ray and
branch on every bounce, on top of the higher sample count used here to
keep the medium's own extra stochastic variance clean), not a regression
in any existing material's own cost.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly.

## 27. A CTest smoke test (done)

Seventeen incremental features (sections 8-26) had accumulated with
**zero automated regression coverage** - every one was verified by
rendering the scene and inspecting the image by hand, real verification
(it caught genuine bugs - see section 20's own account of the two review
findings), but nothing that runs automatically the next time an edit
touches shared code (a helper function, a buffer index, a struct layout)
and silently breaks something that isn't the specific feature being
changed. Section 5 point 3 of this document's own recommended approach
named "some parity coverage against the CPU backend... reusing the
existing `MaterialCpuGpuParityTest` pattern" as a real gap; full CPU/GPU
parity testing is its own separate undertaking, but the cheapest
possible version of "notice when something is badly broken" was still
missing entirely, so this adds it.

`metal_poc_validate.cpp` is a small, standalone second executable (not a
flag added to `metal_poc` itself - keeps the renderer's own code
untouched) that loads a rendered PNG via the already-vendored
`stb_image.h` and checks two things that need no knowledge of this
specific scene's own expected content: the image isn't all-black (mean
pixel value near zero - what a crashed or never-dispatched kernel would
leave in the output texture, still at its cleared/zero-initialized
state) and it isn't a flat single colour (near-zero standard deviation -
what a shading bug that collapses every pixel to the same value would
produce). Two `add_test()` entries wire this into `ctest`: one runs
`metal_poc` itself at a small, fast size (128×128, 16spp, depth 4 - a
smoke test, not a quality benchmark), the second validates its output,
both scoped inside the existing `if(RT_BUILD_METAL)` block so
`enable_testing()` and both tests are invisible to `ctest` entirely
unless `RT_BUILD_METAL=ON` - the same additive posture as every earlier
choice in that block.

**Verified working both directions**: the intended pass case
(`ctest --output-on-failure` after a normal build) passes both tests.
The failure paths were checked directly too, not assumed: running
`metal_poc_validate` against a nonexistent file reports "could not
load" and exits 1; running it with a deliberately wrong expected
width/height against the real rendered output correctly reports the
dimension mismatch and exits 1 - confirming the validator actually
fails when it should, not just passes vacuously. A full default CMake
configure (no `RT_BUILD_METAL`) was also re-run to confirm this change
has zero effect on the CPU/GPU-OptiX build path at all.

This CTest integration is **local-only for now** - the project's actual
CI workflow (`.github/workflows/*.yml`) builds `tests/unit_tests` via
`tests/CMakeLists.txt` directly on a Windows runner, not the root
`CMakeLists.txt` this test lives in, and Metal itself only exists on
Apple hardware a GitHub-hosted Windows/Linux runner doesn't have - so
this doesn't (and currently can't) run in CI. It's real coverage for
anyone building `RT_BUILD_METAL=ON` locally on a Mac, which is this
POC's entire audience today.

## 28. Proof-of-concept, step 18: environment-mapped sky (and a real fog bug, found and fixed) (done)

A miss ray (`result.type == intersection_type::none`) now optionally
samples `earthTexture` by DIRECTION (standard equirectangular mapping,
`equirectangularUV()`: longitude from `atan2`, latitude from `asin`)
instead of the flat two-colour `skyBottom`/`skyTop` gradient every
earlier step used - reusing the same texture already loaded for
`materialType 3`, sampled a genuinely different way (by ray direction,
not by a mesh's own per-vertex UVs). `uniforms.useEnvironmentMap == 0`
keeps the original gradient exactly, purely additive like every earlier
toggle.

**A real, pre-existing bug was found and fixed while verifying this.**
The room's 5 closed walls plus its ~40-degree camera FOV mean almost
every PRIMARY ray already hits something (the room's own open-front
"opening," as seen from the camera, almost exactly fills the frame -
this was noted back when depth of field was first added in step 11) -
so a dedicated wide-FOV, pulled-back test render was needed to actually
exercise miss rays directly, the way step 11/13's own out-of-band
verification renders did for their own features. That render came back
an unexplained near-black frame with sparse bright specks - not the
expected Earth backdrop. Tracing it down: step 17's fog code compared a
sampled scattering distance against `FLT_MAX` for a miss ray's own
"surface distance," intending to let fog still scatter a ray that would
otherwise have escaped. Since a sampled distance is some finite real
number with probability 1, `t < FLT_MAX` is true for EVERY miss ray -
meaning no ray could ever actually reach the sky/environment-map code
at all once fog was enabled: every escaping ray kept "scattering" in a
medium that should have already ended at the scene's own boundary,
wandering further and further from the room while its throughput slowly
decayed, the NEE-against-a-now-astronomically-distant-light contribution
these lost rays evaluate approaching zero with rare firefly-like
exceptions - exactly what the black-with-sparse-specks render showed.
Fixed by gating the fog-scattering branch on an actual surface hit
existing (`result.type != intersection_type::none`) - the fog fills the
scene's INTERIOR, implicitly bounded by the room's own geometry, not
empty space beyond a miss. This was invisible in every one of this
POC's own committed scenes up to this point (only secondary/GI bounces
escaping through the room's open front were ever affected, a small
enough fraction to not read as an obvious artifact in any of step 17's
own screenshots) - a genuine instance of the "looks plausible, renders
something, is subtly wrong" failure mode Section 4 warns about, caught
here specifically because THIS step's own verification method happened
to stress exactly the code path the bug lived in.

**Result, verified three ways**: the dedicated wide-FOV render, post-fix,
shows the Earth texture correctly wrapping the room in every direction -
recognizable continents, ocean, and grid lines, no flipping or mirroring
artifacts, confirming both the equirectangular mapping AND the fog fix
are correct. The committed scene (`useEnvironmentMap = 1`) shows a
subtle but real difference from before: the gold conductor sphere's own
reflection now shows a small patch of Earth-like blue/green where it
catches a GI ray that escapes toward the room's open front. An A/B
render with `useEnvironmentMap = 0` removes exactly that patch and
nothing else, confirming the toggle controls only what it claims to.
700×700 @ 384spp @ depth 10 renders in ~37 seconds on an M2 - actually
FASTER than step 17's own ~46s at the same sample count, consistent with
the fog-bug fix: fewer rays now spend their entire remaining depth
budget wandering through an unbounded medium that should have let them
terminate immediately instead.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` (added in the previous step)
continues to pass.

## 29. Proof-of-concept, step 19: Henyey-Greenstein phase function (directional fog) (done)

Step 17's fog scattered isotropically - a constant `1/(4*pi)` phase
function, uniform in every direction, chosen at the time because it
needed no analogous non-conservative shortcut the way rough dielectric
did (Section 21's own writeup made that comparison explicitly). Real fog
and haze don't scatter uniformly - light continues mostly FORWARD after
a scattering event (Mie scattering off water droplets is strongly
forward-peaked), which is what makes real fog show visible "god rays"
around a light source rather than a flat uniform glow. This replaces the
constant isotropic phase function with Henyey-Greenstein (`
henyeyGreensteinPhase()`/`sampleHenyeyGreenstein()`), the standard
analytic directional phase model - same one pbrt-v4's own
`HGPhaseFunction` implements - controlled by a new asymmetry parameter
`g` (`uniforms.fogAsymmetryG`, positive = forward scattering).

Both the phase VALUE (used for NEE) and its own importance-sampling
distribution (used for the continuation ray) are defined relative to
`wo` (the direction back toward where the ray came from, the exact same
convention `-rayDir` already used by the GGX conductor code) - a
property that made the MIS weighting simpler than the rough-dielectric
or GGX cases needed: HG's own sampling pdf EQUALS its own phase-function
value at the same angle exactly, a defining property of the
distribution, so `henyeyGreensteinPhase()` alone serves as both the
"BSDF" value AND its own competing MIS pdf, with no separate pdf
expression to derive or verify against it. `g == 0` is not a separate
code path either - it's the same formula's own documented degenerate
case, which reduces exactly to the same uniform-over-the-sphere
DISTRIBUTION step 17's isotropic phase function produced (a
rotationally-invariant distribution is identical whether sampled
relative to world axes or an arbitrary local frame like `wo`) - and
`sampleUniformSphere()`, now genuinely unused, was deleted rather than
left as dead code.

**Result, verified three ways**: an isolated render at `g = 0`
reproduces step 17's own isotropic haze look, no visible directional
character. A strong verification render (`g = 0.9`, denser fog for
legibility, not committed) shows an unmistakable glow/halo concentrated
around both ceiling lights - the classic forward-scattering "god ray"
signature no isotropic medium can produce, direct visual confirmation
the directionality is real, not just a scalar brightness change. The
committed scene (`g = 0.4`, a more modest asymmetry) shows a subtler
version of the same effect layered into the existing haze. 700×700 @
384spp @ depth 10 renders in ~37 seconds on an M2, statistically
indistinguishable from step 18's own ~37s (the same number of shadow
rays and bounces per scattering event either way - only which formula
computes the phase value/sampled direction changed, not how many times
it's evaluated).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 30. Proof-of-concept, step 20: Beer-Lambert absorption for dielectrics (done)

Every dielectric material (types 2 and 5) has multiplied `throughput` by
a flat per-bounce `albedo` on every reflect/refract event since step
6/11 - and since every scene up to this point set that colour to
`{1,1,1}`, this line has been a pure no-op the entire time, real glass
rendered perfectly clear rather than tinted. Real coloured glass doesn't
tint by a flat per-bounce factor - it absorbs proportionally to how far
light actually travels THROUGH the medium (Beer-Lambert's law), which is
why a thick paperweight of some glass reads far more saturated than a
thin windowpane of the identical material. `applyBeerLambertAbsorption()`
implements this: on the hit where the ray EXITS a dielectric surface
(`!frontFace`), it multiplies throughput by `exp(-absorption *
result.distance)`, where `result.distance` is exactly the in-medium path
length just travelled - correct with no separate distance-tracking state
needed, for one specific, explicitly-documented reason: every dielectric
shape this POC has is CONVEX (the sphere), so a ray hitting the surface
again after entering must have travelled the whole way through the
interior with nothing else in between. A concave dielectric could
re-enter/exit multiple times in a way this simple per-hit check wouldn't
correctly attribute - not handled, the same "state the assumption
explicitly rather than silently rely on it" approach this POC's other
simplifications (rough dielectric's own energy-conservation shortcut,
Section 21) already use.

`TriangleMaterial::color` now means something different for materialTypes
2/5 than it does everywhere else in the same struct: a per-unit-distance
absorption COEFFICIENT, not a reflectance/tint - `{0,0,0}` means ZERO
absorption (`exp(-0*dist) == 1` exactly, perfectly clear glass), the
opposite of what `{0,0,0}` would mean as a reflectance colour for every
other material type sharing this same field. Both dielectric spheres'
old placeholder `{1,1,1}` "clear glass" values (which, under the OLD
per-bounce-multiply interpretation, meant "no tint"; under absorption,
`{1,1,1}` would have meant "absorb essentially everything, render
black") were replaced with real per-channel absorption: the smooth glass
sphere is now emerald-tinted (absorbs red/blue faster than green), the
frosted sphere a much milder amber.

**Result, verified two ways**: the committed scene shows the glass
sphere clearly green-tinted, and - the specific signature that confirms
this is genuine Beer-Lambert absorption and not just a flat colour
multiply - visibly MORE saturated/darker toward the sphere's own thicker
centre and closer to clear/colourless near its thin edges, exactly the
gradient real coloured glass shows and a flat per-bounce tint cannot
produce. An A/B render with the glass sphere's absorption set back to
`{0,0,0}` reproduces the original perfectly clear glass look exactly,
confirming the code path is a true no-op at its own documented
degenerate value. 700×700 @ 384spp @ depth 10 renders in ~38 seconds on
an M2, statistically indistinguishable from step 19's own ~37s (one
`exp()` call added only on a dielectric's own exit hit, not a new
per-bounce cost category).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 31. POC series status update (sections 24-30)

Section 23 was written after step 14 (PR #16); seven more increments
(PRs #19-#25, sections 24-30) have landed since, adding real breadth
Section 23's own "what's been proven" list didn't yet cover. This is a
short update to that synthesis, not a replacement for it - everything
Section 23 said about the wavefront architecture, scene-builder
integration, app integration, and remaining feature-parity work still
being multi-week-to-months undertakings, not PR-sized ones, is
unchanged and still the right framing.

**Newly proven since Section 23:**

- **Non-identity instance transforms** (step 15) - every instance before
  this used the identity matrix; Suzanne is now instanced twice (one
  translated, one rotated + uniformly scaled) from ONE acceleration
  structure, with object-space normals correctly transformed per-instance
  for shading. Also surfaced a real Metal API gap worth remembering:
  `intersection_result` does not expose an instance's own object-to-world
  transform, unlike OptiX's equivalent query - this POC built its own
  side-channel buffer for it.
- **A second, distinct custom-primitive shape** (step 16) - a disk, via
  a genuinely second intersection function at function-table slot 1
  (every custom primitive before this shared slot 0), and multiple
  heterogeneous geometries coexisting in one acceleration structure.
- **A homogeneous participating medium** (step 17) - free-flight distance
  sampling, verified algebraically before being committed, not taken on
  faith - plus a real bug this itself introduced (miss rays trapped in
  infinite scattering) that a LATER step's own verification method
  happened to catch (step 18) and fixed.
- **Environment-mapped lighting** (step 18) - direction-based (not
  surface-UV-based) texture sampling, reusing an already-loaded texture
  a second way.
- **A CTest smoke test** (Section 27, PR #22, no "step" number of its
  own) - the first automated regression coverage this POC has ever had,
  even though it's local-only (Metal needs Apple hardware CI doesn't
  have).
- **A directional (Henyey-Greenstein) phase function** (step 19) -
  replacing the fog's own isotropic scattering, verified via the
  unmistakable "god ray" signature only real directional scattering
  produces.
- **Beer-Lambert absorption for dielectrics** (step 20) - real coloured
  glass, verified via the thickness-dependent saturation gradient a flat
  per-bounce tint cannot produce; also the point where `TriangleMaterial
  ::color` picked up its THIRD distinct meaning depending on
  materialType (reflectance for most types, F0 for the conductor,
  absorption coefficient for the two dielectrics) - worth knowing before
  reading any of this struct's fields without also checking which
  material type they belong to.

**Bottom line, updated**: the architecture-level de-risking Section 23
described has gotten broader still - instancing, multi-shape custom
primitives, volumetric media, and image-based lighting are now added to
the list of things proven on real hardware, not just materials/lights/
cameras/meshes. Automated regression coverage exists for the first time,
even if narrow and local-only. None of this changes Section 6's own
effort estimates for what a real port still needs - if anything, each
new proven piece is one more confirmation that the remaining gap is
breadth-of-effort (porting/rewriting a large, specific feature set) and
architecture-level integration (wavefront restructuring, scene-builder,
app integration), not remaining architectural risk in the Metal/MetalRT
approach itself.

## 32. Proof-of-concept, step 21: a procedural checkerboard texture (done)

Every textured surface so far (`materialType 3`) has sampled a real
IMAGE (`earthTexture`) - a genuinely different, and arguably more
fundamental, technique in production rendering is a PROCEDURAL texture:
computed analytically from the hit's own UV, no sampler/image involved
at all. `materialType 6` adds the textbook example of one -
`checkerColor()`, alternating between a colour and a fixed-fraction-
darker version of it based on the parity of `floor(u*scale) +
floor(v*scale)` - applied to the room's floor (previously plain white
Lambertian), reusing `addQuad()`'s own existing planar 0-1 UVs with no
scene-authoring changes needed beyond the one material-type flag.

**A real near-miss, caught before it shipped, not after**: the natural-
seeming choice for tile B's colour would have been reusing
`TriangleMaterial::emission` (an otherwise-unused field on a non-
emissive material, the exact kind of repurposing `ior`/`roughness`
already do for materialTypes 2/4/5). That would have been wrong -
`emission` is read UNCONDITIONALLY by the shading loop's own "is this
hit a light source" check (`any(mat.emission) > 0`), regardless of
materialType, so a nonzero tile-B colour stored there would have made
the checkerboard floor incorrectly glow as if it were an area light.
Caught while writing the code, before ever compiling or rendering it,
by re-reading how `emission` is actually consumed elsewhere rather than
assuming a field being "currently unused for this materialType" means
it's safe to repurpose - not every unused-looking field is actually
free. Fixed by deriving tile B analytically (a fixed darkening factor)
instead of storing a second colour at all.

**Result, verified two ways**: the committed scene shows a clean 8x8
checkerboard floor, visible directly and correctly reflected in both
the mirror disk and the glass/frosted spheres (confirming the pattern
participates in indirect light transport normally, not just primary
visibility). An A/B render with the floor's materialType reverted to 0
reproduces the original plain white floor exactly. 700×700 @ 384spp @
depth 10 renders in ~37 seconds on an M2, statistically indistinguishable
from step 20's own ~38s (one analytic `floor()`/`fmod()` computation
added to an existing albedo lookup, not a new cost category).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 33. Proof-of-concept, step 22: Fresnel-weighted mirror reflectance (done)

The mirror material (`materialType 1`) has multiplied `throughput` by a
flat `albedo` on every reflection since the very first multi-material
step - correct only exactly at normal incidence. A real mirror's
reflectance rises toward white/uncolored at grazing angles regardless of
its base tint (the same physical effect the GGX conductor material
already models); a flat multiply silently under-brightens every
grazing-angle reflection instead. Fixed by reusing
`fresnelSchlickConductor()` - already written, already verified, for the
GGX conductor material - with `albedo` doubling as this surface's own F0
(the same "colour IS the normal-incidence reflectance" convention that
material already established for its own F0 parameter). A small, low-
risk fix: no new sampling math, no new struct fields, just an existing,
already-trusted function applied to an existing material that hadn't
been using it.

**Why this had gone unnoticed for eight prior steps**: the scene's own
mirror (the wall disk added in step 16) uses a near-white albedo
(`{0.9, 0.9, 0.9}`), close enough to F0-at-white already that the
correction is real but visually subtle in the committed scene - Schlick
grazing-angle whitening asymptotically approaches white regardless of F0,
so a near-white F0 has little room left to visibly change.

**Result, verified with a dedicated colour choice, not the committed
one**: a temporary copper-coloured mirror (F0 ≈ `{0.7, 0.25, 0.1}`, chosen
specifically because it has plenty of room to show the effect) makes the
difference unambiguous - the OLD flat-multiply version renders uniformly
copper-tinted across the whole disk regardless of viewing angle; the NEW
Fresnel-weighted version shows the copper tint concentrated near the
disk's centre (closer to normal incidence from the camera) and visibly
whitened/desaturated toward its edges (closer to grazing), exactly the
signature real Fresnel reflectance produces and a flat multiply cannot.
The committed scene's own near-white disk shows the same effect, just
proportionally smaller given its own F0 choice. 700×700 @ 384spp @ depth
10 renders in ~36 seconds on an M2, statistically indistinguishable from
step 21's own ~37s (one existing function call added to an existing
material, not a new cost category).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 34. Proof-of-concept, step 23: anisotropic GGX conductor, checked against a real reference implementation (done)

The GGX conductor material (`materialType 4`) has been isotropic
(single `alpha`) since step 9 - real brushed/machined metal has a
genuinely directional highlight (elongated, not round), which an
isotropic BRDF cannot produce regardless of roughness value. This had
been deliberately deferred earlier in this series over two concerns:
correctly generalizing Heitz 2018's VNDF sampling to the anisotropic
(alphaX/alphaY) case, and needing a REAL, consistently-oriented tangent
direction across a curved surface (an isotropic BRDF is rotationally
symmetric around the normal, so the arbitrary tangent `buildOnb()`
produces was always fine before now; an anisotropic one visibly is not).

**What changed to make this tractable now**: a local, sparse reference
checkout of Blender Cycles' own Metal-backend renderer
(`intern/cycles/kernel/closure/bsdf_microfacet.h`) was pulled in as a
cross-check. `ggxD`/`ggxLambda`/`ggxG1`/`ggxG`/`sampleGGXVNDF` were all
generalized to take `alphaX`/`alphaY` and full local-frame vectors
(not just a `NdotX` scalar, which the anisotropic case genuinely needs),
and verified algebraically AND against Cycles' own `bsdf_aniso_D`/
`bsdf_aniso_lambda`/`microfacet_ggx_sample_vndf` before being committed -
matching, not just resembling, a real production implementation's
formulas. `buildAnisotropicOnb()` solves the tangent-direction problem
by projecting a fixed world-space reference axis (world-up, falling
back to world-X exactly at the poles) onto the local tangent plane via
Gram-Schmidt - the same construction Cycles' own
`make_orthonormals_tangent()` uses, adapted for this POC's analytic
sphere having no per-vertex tangent data of its own to begin with.

`alphaX == alphaY` reduces every one of these functions EXACTLY to the
isotropic formulas this POC used through step 22 - not a separate
code path, the same functions degenerating correctly at their own
boundary case (verified both algebraically and by a dedicated render,
below). `TriangleMaterial::roughness` - otherwise idle for
`materialType 4` since only `materialType 5` used it before now -
doubles as alphaY, with `roughness == 0.0` (the only value any earlier
scene ever set) falling back to the isotropic case automatically.

The gold conductor sphere's own roughness was changed from an isotropic
0.15 to a genuinely anisotropic `alphaX = 0.08, alphaY = 0.45` - a
deliberate showcase change, not a value chosen to preserve the previous
look (that comparison is what the dedicated verification renders below
are for).

**Result, verified two ways**: the committed scene shows the gold
sphere with a clearly elongated, "brushed metal" highlight - a real,
unmistakable streak, not a round one stretched by antialiasing or
noise. A dedicated render at the EXACT previous parameters (`alphaX =
0.15, alphaY = 0`, forcing the isotropic fallback) reproduces the
original round, symmetric highlight from step 9 near-identically,
confirming the anisotropic generalization's own degenerate case is a
true no-op, not just a plausible-looking approximation of one. 700×700 @
384spp @ depth 10 renders in ~36 seconds on an M2, statistically
indistinguishable from step 22's own ~36s (the same number of BRDF
evaluations per bounce either way - only which formula computes them
changed).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 35. Proof-of-concept, step 24: a true delta point light (done)

Every light in this POC has been an `AreaLight` - finite area, uniform-
sampled, needing the area-to-solid-angle pdf conversion and power-
heuristic MIS every NEE branch has carried since step 5. A point light
is a fundamentally simpler case, not just a smaller area light: it has
zero area and zero solid angle (a true delta distribution), which means
it has EXACTLY zero probability of ever being hit by a BSDF-sampled
continuation ray - the same "measure zero" reasoning the mirror/
dielectric materials already use to skip NEE for their OWN delta lobes,
mirrored here from the light's side instead of the material's. A point
light's own NEE contribution therefore needs no MIS weight at all
(always full weight) and no area-sampling pdf (just an idealized
`1/distSq` falloff) - the textbook simplest possible light type, and a
deliberately different one from every `AreaLight` so far, not a
variation on the same theme.

Point lights are NOT folded into the existing `lights[]`/light-picking
scheme - there's no variance-reduction benefit to stochastically picking
among a handful of always-fully-weighted point lights the way there is
for choosing among several area lights, so a SEPARATE, small
`pointLights` buffer/count is summed unconditionally each bounce instead
(every point light's contribution added, not one picked at random).
Added consistently to every existing NEE call site that already handles
area lights: the Lambertian branch, the GGX conductor branch (with its
own full anisotropic BRDF evaluation, reusing `ggxD`/`ggxG` exactly as
the area-light NEE already does), and the fog's own volume-scattering
NEE (phase-function value instead of a BRDF, same as its area-light
counterpart).

Placement took two iterations, the same "verify by rendering, not by
distance-math alone" lesson this POC keeps re-learning: the first
position/intensity was close enough to a wall that its own `1/distSq`
falloff blew the nearby geometry out to solid white - not a bug, exactly
the physically-correct (if impractical) behaviour a point light produces
near a surface, but not what this increment meant to demonstrate. Moved
to a position with generous clearance from every surface, well above the
existing objects.

**Result, verified two ways**: the committed scene shows a distinct
soft glow patch on the back wall, visibly separate from - not blended
into or hidden by - the two existing area lights' own illumination and
the two rectangular ceiling-light reflections, confirming the point
light is genuinely contributing radiance rather than sitting unused in
its buffer. An A/B render with `pointLightCount = 0` removes exactly
that glow patch and nothing else, reproducing step 23's own committed
render exactly. 700×700 @ 384spp @ depth 10 renders in ~62 seconds on an
M2, up from step 23's own ~36s - a REAL cost increase this time (one
extra shadow ray per bounce in three separate code paths, not
statistical noise), not glossed over the way several earlier steps'
truly negligible cost deltas were.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 36. Proof-of-concept, step 25: a spot light (done)

Step 24's point light is omnidirectional - its `1/distSq` falloff
depends only on distance, never on direction, so it cannot produce the
one signature effect a real spotlight is built around: a "pool of
light" with a sharp-ish edge, invisible outside its cone no matter how
close a surface is. A spot light is the same delta-light math as step
24 (still zero area, zero solid angle, still always full NEE weight, no
MIS, no pdf conversion) with one addition: a direction vector and an
inner/outer half-angle, multiplying the emission by an angular falloff
term instead of leaving it isotropic.

`PointLight`/`PointLightData` gained `direction`, `cosOuterAngle`, and
`cosInnerAngle` fields (comparing cosines instead of angles avoids an
`acos` per shadow ray). `cosOuterAngle <= -1.0` is the sentinel for "no
cone at all" - every point light added before this step left these new
fields at their default-initialized omnidirectional values, so step
24's own light and its A/B renders are reproduced exactly without
touching a single existing call site's own logic, only the falloff
factor each one already multiplies in. Between the two angles the
falloff is smoothstep-blended (`3t^2 - 2t^3`, Blinn/pbrt-style), not a
hard cutoff - a hard-edged cone aliases badly under Monte Carlo
integration, since every sample straddling the boundary either gets the
full contribution or none of it with no way to average toward a
correct in-between value.

Added a second point light, positioned above and aimed down at Spot-the-
cow's own floor area (25 degree outer / 15 degree inner cone) - a
genuinely different "shape" of illumination from step 24's glow patch,
not a recolored copy of it, and the same fog volume this scene already
has makes the cone itself visible as a shaft in the air, not just its
footprint on the floor.

**Verification took an extra step this time**: the committed scene's
own renders (700x700 @ 128spp and a 1000x1000 zoomed crop) didn't make
the cone unambiguous at a glance next to the existing area lights and
point light's own illumination, so - rather than assume the shader logic
was wrong OR right from a marginal image - an isolated diagnostic build
was compiled with the spot's emission boosted 10x and the first point
light's emission zeroed (both edits applied to a throwaway copy of
`metal_poc.mm`, never committed). That render shows an unambiguous,
correctly-shaped cone: narrow at the light, widening toward the target,
visible as a distinct fog-lit shaft with a matching bright patch where
it lands - confirming the falloff math and aim were correct all along,
and the ambiguity in the first renders was just intensity, not a bug.
The committed scene's own spot emission was then raised (`(3,2.6,4)` to
`(7.5,6.5,10)`, roughly 2.5x) to read clearly at normal exposure without
the 10x diagnostic boost's own oversaturation.

Re-verified two ways at the final intensity: the showcase render shows
the cone as a clear, distinctly-shaped shaft next to the existing point
light's glow and the two area lights, landing on Spot's floor area as
intended. An A/B render with the spot entry removed from `pointLights`
(keeping step 24's own light) reproduces that light's own committed
render exactly, with only the cone and its footprint missing - confirming
the two lights' contributions are genuinely independent, not entangled.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 37. Proof-of-concept, step 26: a directional ("sun") light (done)

Steps 24-25 added two delta lights, but both radiate from a finite
POINT - a spot is just a point light with an angular mask, not a
different underlying shape of illumination. A directional light is
genuinely different in kind: parallel rays with no position at all and
no `1/distSq` falloff, the idealized limit of a point light as its
distance goes to infinity and its emission grows to compensate. Still
the same delta-light integration math as steps 24-25 (zero solid angle,
always full NEE weight, no MIS, no area-sampling pdf) - only the
geometry of "which direction is the light in" changes, from
`normalize(lightPos - hitPoint)` to a single fixed vector shared by
every shading point in the scene.

`DirectionalLight`/`DirectionalLightData` carry just `direction` and
`emission` - no position, matching `PointLight`'s own `direction`
convention (the direction the light itself travels). Because a
directional light's shadow ray has no real target distance, its
`max_distance` uses a sentinel (`kDirectionalLightMaxDistance = 10.0`,
comfortably larger than this room's own `[-1,1]^3` extent) rather than a
computed one. This forced one explicit, documented simplification: a
point/spot light's fog attenuation (`exp(-fogSigmaT * plDist)`) uses a
REAL finite distance, but a directional light has no such distance
before it exits the room's own open front - rather than Beer-Lambert
across an arbitrary sentinel length (which would either silently
over- or under-attenuate depending on what constant was picked), this
light's own NEE contribution skips fog attenuation entirely. Documented
in the shader as a deliberate scope decision, the same judgement call
step 24's own doc applied to GGX energy compensation, not a
quietly-wrong approximation.

Added one directional light, aimed with a slight downward/sideways tilt
(`direction = (0.1, -0.15, -1.0)`, mostly -z) through the room's own
open front (the `z=1` face has no wall - see the floor/ceiling/wall
`addQuad()` calls in step 1's own scene). Verification surfaced a real,
non-obvious geometric interaction worth recording: an isolated
diagnostic render (emission boosted ~10x, the other two point lights
zeroed, never committed) showed the room's LEFT half staying dark while
the RIGHT half lit up brightly, a hard diagonal boundary between them -
not a bug, but genuine self-shadowing. The light's shallow entry angle
means a shading point near the red (`x=-1`) wall traces back toward the
light along a path that re-hits that SAME wall's own surface almost
immediately, before it can ever reach the open front at `z=1`; a point
near the green (`x=1`) wall traces back along a path moving AWAY from
any wall, reaching the opening freely. A real optical effect a shallow
directional light produces near a parallel surface, the same family of
"verify by rendering, not by geometry math alone" lesson step 24's own
placement iterations already taught this POC, applied to a new light
shape.

**Result, verified two ways**: the committed scene's showcase render
shows a clear, tasteful raking highlight across the back wall and green
wall's own right-hand portion (tuned to `(2.8, 2.6, 2.4)` after an
intermediate `(4.5, 4.2, 3.9)` clipped that patch to solid white -
dialed back once, the same "verify at the committed intensity, not just
the diagnostic one" step this POC's light PRs keep repeating), distinct
from the two area lights, the point light's own glow, and the spot's own
cone. An A/B render with `directionalLightCount = 0` reproduces the
prior committed (spot-light) render exactly, with only that raking
highlight missing - confirming the new light's contribution is genuinely
isolated, not entangled with the two point/spot lights already in the
scene.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 38. Proof-of-concept, step 27: procedural bump mapping (done)

Steps 24-26 were all lights; this step returns to materials with a new
Lambertian variant (materialType 7) that perturbs the SHADING normal in
tangent space rather than changing albedo the way materialTypes 3/6
already do - genuinely different from every earlier "same BSDF/NEE code
path, different albedo source" material (3's texture sample, 6's
analytic checker), since here the BSDF math is unchanged and what the
surface's own local frame POINTS is what varies instead. An analytic
"egg carton" height field (`h(u,v)`) generates the perturbation rather
than a sampled normal-map texture, so this needs no new image asset:
`proceduralBumpNormal()` perturbs the normal by that height field's own
partial derivatives along a tangent/bitangent basis, the textbook bump-
mapping construction (shading-only; this does NOT displace geometry, a
deliberately different, cheaper effect from real displacement mapping).
The true geometric normal is left untouched for ray-offset/self-
intersection purposes - only `facingNormal`, the value the BSDF/NEE math
downstream actually shades with, gets perturbed.

A new `tangentFor()` helper solves for the per-triangle tangent from the
standard position/UV partial-derivative construction (pbrt-v4's own
technique), reusing the primary triangle buffer's existing flat
`vertices`/`uvs` indexing rather than needing new per-triangle host-side
data - deliberately scoped to that buffer only (never a sphere/disk/
Suzanne-instance hit, guarded explicitly), since only it has both a
`vertices` and `uvs` array following the same flat `primId * 3 + i`
convention `tangentFor()` relies on. `roughness` picks up a FOURTH
distinct meaning for materialType 7 (bump strength) alongside its
existing 2/5 reuse - 0.0 there is a true no-op, exactly the unperturbed
normal, not an approximation of one.

**Two real bugs, both caught by inspecting a render rather than trusting
the math**, the same "verify by rendering" discipline this POC's light
PRs already established, now applied to a material:

1. The first version's derivative scaling literally included the
   height field's own `2*pi*frequency` amplitude factor - mathematically
   "correct" for a literal height-field derivative, but at this bump's
   own frequency that factor alone was ~63x, drowning the unit normal
   entirely regardless of how small the strength parameter was set. A
   render at even an extreme strength value showed zero visible change -
   traced to ground truth by temporarily replacing the shaded output
   with a direct false-colour visualization of `facingNormal` itself,
   which confirmed the normal WAS varying, just not translating into any
   visible shading difference. Fixed by dropping the redundant factor and
   re-deriving `strength` as a direct slope scale instead.
2. Even after that fix, the committed panel (originally flush-
   mounted on the red side wall) still rendered completely flat. The
   panel's own placement - hugging `x = -0.99`, a hair off the red wall
   at `x = -1` - put it in almost exactly the same near-total-self-
   shadow condition step 26's own directional-light doc describes for
   that wall, on top of receiving only a steep grazing angle from the
   overhead area lights: correct bump math, but with no real direct
   light ever reaching the surface to reveal it, since a Lambertian
   bump's visibility depends on a well-defined light direction, not
   ambient/GI-only illumination. Fixed by relocating the panel to the
   back wall, in the region the directional light hits closest to head-
   on - not a code change at all, a scene/lighting placement fix, the
   same category of fix step 24's own point-light placement iterations
   needed.

**Result, verified two ways** at the final placement/parameters (back
wall, `strength = 0.6`, bump frequency lowered from an initial, badly-
aliasing 10 cycles to 4 across the panel's own UV span - the first
frequency produced per-pixel noise indistinguishable from Monte Carlo
grain instead of a visible bump shape): the showcase render shows clear
diagonal bump-shading bands across the panel, a real "egg carton" look,
not a subtle hint. An A/B render with `strength = 0.0` reproduces a
perfectly flat, uniformly-shaded panel - confirming the perturbation is
a true, isolated, zero-at-zero effect.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 39. Proof-of-concept, step 28: ACES filmic tonemapping (done)

Every render this POC has ever produced went through the exact same
final step: read back the linear HDR float buffer, clamp straight to
`[0,1]`, then gamma-encode to 8-bit. That clamp is a hard cliff, not a
curve - any pixel at 1.3x "full white" and any pixel at 30x "full white"
both become identical solid white, discarding all the shape of exactly
how overexposed something was. Every light this POC has added (area,
point, spot, directional) makes this MORE visible, not less - more
things in the committed scene are now bright enough to clip. This step
replaces that clamp with `acesFilmicTonemap()`, Krzysztof Narkowicz's
widely-used fitted approximation of the ACES RRT+ODT curve - a smooth
compressive rolloff instead of a cliff, the same reason virtually every
production renderer/game engine tonemaps before display rather than
clamping raw linear radiance. Applied per channel, in linear space,
BEFORE the existing 1/2.2 gamma approximation - the correct ordering,
not the reverse (gamma-encoding first would feed the curve fit the wrong
input range entirely).

Purely a host-side, post-process change to `metal_poc.mm`'s own
readback loop - no shader, scene, or integrator code touched at all, the
narrowest possible scope for a change that still affects every pixel of
every future render.

**Result, verified via a direct before/after crop comparison** (same
scene, same seed, same sample count, only the readback function
changed): a genuinely emissive light source's own core (radiance far
above 1.0 by design) still reads as solid white either way - no
tonemapping operator un-clips a light that's ACTUALLY that many times
overexposed, and this doesn't try to. The real difference shows up in
MODERATELY overexposed regions - the directional light's own raking
highlight on the back wall (step 26), previously a flat, textureless
white patch, now retains a faint gradient and a hint of the mirror
disk's own colour bleeding through at its edge, exactly the kind of
subtle highlight detail a hard clamp discards and a filmic curve
preserves.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 40. Proof-of-concept, step 29: blackbody-temperature light colours (done)

Every light this POC has ever added picked its own emission colour as a
hand-tuned RGB tuple - reasonable for placement/intensity tuning (steps
24-26's own docs are full of exactly that kind of iteration), but the
COLOUR itself was always just "whatever looked right," with no physical
grounding. This step adds `blackbodyColor()` - Tanner Helland's widely-
used polynomial fit to the Planckian locus, converting an actual
temperature in Kelvin to an approximate RGB tint - and uses it to derive
three of this scene's existing light colours from a NAMED physical
quantity instead: 9000 K (cool, moonlight-ish) for the first point
light, 3000 K (warm tungsten) for the spot, and 5778 K - the Sun's own
real photosphere temperature - for the directional light. Each colour is
still scaled by a plain intensity multiplier chosen to land in roughly
the same brightness range the scene's own lights already used, so only
the HUE is newly derived, not the overall exposure balance this POC has
already tuned scene-by-scene.

Purely a host-side utility and three call-site changes in
`metal_poc.mm` - no shader or integration math touched at all, the same
narrow scope step 28's tonemap change had.

**Verified two ways**: first numerically, against a small standalone
C program computing the exact same formula (`bb(9000)`, `bb(3000)`,
`bb(5778)`) to confirm the in-repo function's own output before ever
rendering anything - `(0.822, 0.874, 1.000)`, `(1.000, 0.695, 0.431)`,
and `(1.000, 0.951, 0.904)` respectively, all matching visual
expectations for those temperatures (progressively bluer above ~6600 K,
progressively warmer/oranger below it). Then visually, via a direct
before/after crop of the first point light's own wall-glow patch (step
24's own signature verification image): the derived 9000 K colour reads
distinctly cooler/bluer than the hand-picked `(0.9, 0.65, 1.1)` it
replaced, confirming the hue shift actually reaches the final image, not
just the formula's own output. Also notable: the derived 5778 K sun
colour (`(1.0, 0.951, 0.904) * 2.7 ≈ (2.7, 2.57, 2.44)`) came out nearly
IDENTICAL to step 26's own hand-tuned `(2.8, 2.6, 2.4)` - a satisfying
confirmation that the earlier manual tuning had already converged on
something close to physically correct for a real sun-like light, not a
coincidence worth ignoring.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 41. Proof-of-concept, step 30: blackbody-derived area light colours (done)

A small, narrowly-scoped follow-on to step 29: the scene's own two
ceiling `AreaLight`s (added all the way back in step 4/section 12, "a
warm light and a cool light side by side") were the last hand-picked-RGB
lights left in the scene after step 29 converted the point/spot/
directional ones - this closes that out, so every light in the scene now
gets its colour from `blackbodyColor()` at a named temperature: 2700 K
(a standard incandescent bulb) for the warm one, 20000 K (near the top
of the fit's own valid range - a very hot, blue-white source) for the
cool one.

**A genuinely interesting, honestly-reported limitation surfaced here**:
20000 K's own derived colour (`(0.669, 0.778, 1.0)`) is nowhere near as
saturated a blue as the hand-picked `(6, 10, 18)` it replaces (a much
higher red:blue contrast ratio). Checked numerically before settling on
this: even at 40000 K, the very top of the fit's valid range, the
derived colour only reaches `(0.595, 0.728, 1.0)` - real blackbody
radiation never produces a deeply saturated blue the way an artistic RGB
pick can, since a thermal emitter's spectrum always retains substantial
energy across the visible range even as its peak shifts blue. This is a
genuine physical fact this derivation surfaces, not a bug to work around
- documented in the scene comment rather than swapping in a less-honest
temperature or abandoning the derivation for this one light.

**Result, verified**: numerically (both temperatures' own RGB output
checked against the same standalone reference program step 29 used,
before rendering), and visually (a full-scene render shows a visibly
less blue-saturated cool light and correspondingly less blue ambient
bounce on the green wall/ceiling than the previous hand-tuned version,
consistent with the numeric difference above - an honest, expected
change, not a regression).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 42. Bug fix: area lights emitted from both sides (found while reviewing shading code, done)

Every `AreaLight` NEE branch has always correctly checked `cosLight >
0.0` before contributing - a light only illuminates surfaces on the
side its own `normal` points toward. The unconditional "did a ray land
directly on the light quad" check (covering camera rays and BSDF-
sampled continuation rays, since NEE alone can't explain a ray that
happens to hit the light by chance) had NO equivalent check at all -
`if (any(mat.emission) > 0.0)` fired regardless of which face was hit,
meaning a ray reaching the light's own BACK side still read its full
emission. Found by re-reading this exact block while looking at
something else entirely, not by chasing a visible symptom - the
inconsistency with every NEE branch's own already-correct one-sidedness
was the tell.

**Invisible in the committed scene**: every light here is mounted flush
against the ceiling (`y = 0.98`, the ceiling quad itself at `y = 1.0`),
so its back face is physically inaccessible from inside the room - this
bug produced ZERO visible difference in any of this POC's own renders
to date. Fixed anyway, since a shading model that's only "one-sided" by
geometric accident rather than by its own logic will eventually bite a
scene that isn't this one (a light floating in open space, viewable from
both directions, would have shown it immediately).

Fixed by reusing `frontFace` (already computed a few lines above for the
dielectric branch's own eta selection) as an extra condition on the
emissive check - a one-line, minimal-risk change.

**Verified two ways**: an isolated diagnostic scene (a throwaway light
quad floating in open space, deliberately wound so its normal points
AWAY from the camera) makes the bug and fix unambiguous - the pre-fix
render shows the camera looking straight at a huge, blown-out white
glow from the quad's own BACK face; the post-fix render shows the exact
same quad correctly reading as a plain, unlit grey panel (ambient
GI-lit only, since it's still a valid Lambertian surface underneath -
just no longer glowing from the wrong side). Separately, a pixel-diff
of the actual committed scene (`stb_image`-based comparison tool,
64spp, same seed, before vs after) confirms the fix's real-world impact
here is exactly as small as expected: only 1504 of 750000 subpixels
differ at all, average difference 0.0028 (out of 255) - consistent with
rare grazing GI bounce rays clipping through the thin 0.02-unit gap
between a ceiling light and the ceiling quad above it, not a wholesale
change to the scene's own appearance. Both ceiling lights still glow
exactly as brightly as before from their own correct (downward-facing)
side, confirming the fix didn't accidentally invert which face is
"front."

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 43. Bug fix: the earth texture was never sRGB-decoded (found while reviewing texture setup, done)

Found the same way step 42's bug was: reading code for something else
entirely and noticing a real problem. `earthTexture` (step 8's own
texture-mapping addition, later reused for step 18's environment map)
has been created as `MTLPixelFormatRGBA8Unorm` since the very first
texture-mapping PR - a plain linear 8-bit format. An ordinary JPEG/PNG's
own stored bytes are sRGB-gamma-ENCODED, not linear (a well-known
convention: perceptually-spaced storage gives dark tones more of the
available 8-bit precision) - sampling those bytes directly as if they
were already linear, the way `RGBA8Unorm` does, silently darkens every
midtone this texture (or anything it bounces light onto via GI) ever
produced. This is a classic, well-known bug class in real-time and
offline renderers alike, not a subtle or exotic one - it's just been
sitting here unnoticed since step 8.

Fixed by switching both the real texture's descriptor and its 1x1 white
fallback to `MTLPixelFormatRGBA8Unorm_sRGB` - the hardware-accelerated,
standard fix (Metal's own texture sampler decodes sRGB to linear
automatically for this pixel format, before the shader ever sees a
value), not a manual `pow(c, 2.2)` after sampling in the shader. The
white fallback's own colour is unaffected either way (pure white
round-trips through sRGB<->linear unchanged) - included only for
consistency between the two descriptors.

**Result, verified via a direct before/after render comparison**: this
is a LARGE, obvious, immediately visible difference, unlike step 42's
own subtle one - the earth texture's oceans go from a washed-out, pale
purple-blue to a properly saturated deep navy, its landmasses from pale
washed green to a richer, more contrasted green, and Spot-the-cow's own
texture-mapped fur reads noticeably darker and more saturated overall.
Exactly the signature this bug class produces (uniformly washed-out
midtones), not a colour-balance shift or a lighting change - confirming
this was a real, meaningful correctness bug, not a cosmetic tweak.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 44. Closing a documented gap: directional-light fog attenuation (done)

Step 26's own `DirectionalLight` comment explicitly documented a
simplification: unlike the point/spot lights' real, finite shadow-ray
distance (`exp(-fogSigmaT * dist)`), a directional light's shadow ray
has no target distance at all (the light is at infinity), so its own
fog attenuation was skipped entirely rather than guessing at an
arbitrary sentinel length's worth of Beer-Lambert. This step closes that
gap properly instead of leaving it permanent: since this scene's own fog
fills exactly the room's solid geometry (a fixed, known `[-1,1]^3` box -
see the floor/ceiling/wall `addQuad()` calls in step 1's own scene), and
an UNOCCLUDED directional-light shadow ray (by definition, since this
code only runs when the real intersection test found nothing) can only
have exited through the room's own single gap (the open front - see
step 26's own aiming comment), the distance to where that same ray
crosses the room's bounds IS the real fog path length, not a guess.

`rayBoxExitDistance()` is the standard axis-aligned-box "far" slab-test
distance, assuming the ray origin is inside the box (true for every
shading point in this scene). Wired into all three directional-light NEE
branches (Lambertian, GGX conductor, fog volumetric) exactly the same
way the point/spot lights' own real distance already is - the two
"kinds" of delta light now share the identical fog-attenuation shape,
just computed differently (a real traced distance for one, an analytic
box-exit distance for the other).

**Result, verified two ways**: at this scene's own actual (fairly
subtle) fog density, an `stb_image`-based pixel-diff shows the expected
small, widespread effect - about 45% of subpixels differ, but by a small
amount (average 0.72/255, max 5/255), consistent with a modest,
physically-correct dimming spread across many surfaces rather than a
dramatic change. At a deliberately exaggerated fog density (`fogSigmaT`
raised 20x, non-committed, diagnostic only), the effect becomes
dramatic and unambiguous: virtually every pixel differs (99.97% of
subpixels, average difference 46.7/255) and the directional light's own
visible "god ray" fog beam - previously shining through the dense fog
completely unattenuated - is now correctly extinguished, exactly the
signature Beer-Lambert absorption is supposed to produce at high optical
depth.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 45. Proof-of-concept, step 31: procedurally roughness-mapped GGX conductor (done)

A new material (materialType 9) reusing materialType 4's own exact NEE +
BSDF-sampled-continuation + MIS structure, but with `alphaX`/`alphaY`
computed from an analytic UV-space checker pattern (`checkerColor()`,
picking between two roughness values instead of two colours) instead of
being one constant per primitive - patches of near-mirror-smooth and
rough microfacet regions on the SAME surface, a "worn/scratched metal"
look. Genuinely different in KIND from step 23's own anisotropic
conductor: that one varies alpha BY DIRECTION at a single point (one
alphaX, one alphaY, constant everywhere on the surface); this one varies
alpha BY LOCATION (isotropic at any single point, but which isotropic
value applies changes across the surface).

Restricted to sphere primitives, using `equirectangularUV()` on the
hit's own object-space normal for a texture-space coordinate - the same
technique the environment map already uses for direction-based
sampling, reused here for a second purpose. This sidesteps needing real
mesh UVs/`tangentFor()` the way step 27's bump-mapped Lambertian
(materialType 7, triangle-only) needed - a sphere's own hit normal
already gives a natural, free coordinate, no new per-primitive data
needed host-side at all. `ior`/`roughness` reused a FOURTH and FIFTH way
(after materialTypes 4/5/7's own reuses) as the smooth/rough patches'
own perceptual roughness values respectively.

Added as a new fourth sphere (copper-tinted), placed at the room's
front-left. Placement took a few iterations, the same "verify by
rendering" lesson this POC keeps re-learning for new geometry: the
first two positions tried landed the sphere fully out of camera view
(the first hidden behind Spot-the-cow along a near-identical sightline,
confirmed by the `4 spheres` scene-summary line printing correctly while
nothing new appeared on screen) - not a rendering bug, a placement one,
caught by actually looking rather than trusting the 3D coordinates
alone.

**Result, verified two ways**: the committed scene shows the new sphere
with a visibly different sheen between two regions of its own surface -
small in this framing (partially at the frame's own edge, the same
"partially cropped is fine" precedent the giraffe's own foot already
sets in every prior render). A dedicated, non-committed diagnostic
render (the same sphere enlarged and moved to fill the frame, verifying
composition rather than the committed scene) makes the effect
unambiguous: a real checkerboard of alternating sharp mirror
reflections and blurred, matte highlights on one continuous surface, not
a subtle hint.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 46. Proof-of-concept, step 32: polygonal aperture (bokeh) shape (done)

Step 19's own thin-lens DOF has sampled a perfectly circular aperture
(`sampleUnitDisk()`) unconditionally ever since - physically the
idealized "infinite aperture blades" limit, not what a real camera lens
actually does. A real lens's aperture is a finite-sided polygon (the
blades themselves), which is exactly why out-of-focus highlights in an
actual photograph read as hexagons/pentagons/etc. rather than perfect
circles - a well-known, purely cosmetic-but-recognizable signature of
real camera optics this POC's own DOF couldn't produce at all before
this step.

`samplePolygonAperture(sides, rngState)` samples a regular N-gon
uniformly: pick one of `sides` equal triangular wedges (origin to two
adjacent polygon vertices) uniformly at random, then a point within that
wedge via the standard sqrt-for-uniform-triangle-area construction - the
textbook regular-polygon sampling technique, not an approximation of
one. A new `apertureBlades` uniform selects it: `0/1/2` (every scene
before this one) keeps the exact original `sampleUnitDisk()` path -
purely additive, the same "0 reproduces prior behaviour exactly" shape
`lensRadius == 0` itself already has, and never touched at all when
`lensRadius == 0` regardless of this new field's value.

The committed scene now uses a 6-blade (hexagonal) aperture, the classic
photographic blade count.

**Verified via a dedicated diagnostic, not the committed scene**: the
committed scene's own existing DOF blur turned out too soft/diffuse to
show a recognizable aperture shape by itself (broad area-light
reflections and matte surfaces, not the small bright points against a
dark background real bokeh photos use to make the shape legible) - a
throwaway scene modification (five small, bright emissive quads
scattered on the defocused back wall, mimicking a classic "blurred
string of lights" bokeh test shot, `lensRadius` also temporarily
increased for a more pronounced blur) makes the effect unambiguous: a
crop of one blurred light shows a clean, distinct hexagon with the
6-blade aperture, and a perfect circle with `apertureBlades = 0` on the
identical scene - confirming both the new polygon path and the
fallback's own exactness.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 47. Proof-of-concept, step 33: natural lens vignetting (done)

A small companion to step 32's own polygonal aperture - another real
camera-lens signature this POC's own image had never modelled: a real
lens transmits less light to the sensor at the frame's own edges/corners
than at its centre (the classic natural/`cos^4` vignetting law),
darkening corners even with nothing physically blocking light the way a
lens hood or filter ring would ("mechanical" vignetting, which this is
NOT modelling). `vignetteFactor()` computes a radial falloff
(`1 - strength * r^4`, `r` the normalized distance from frame centre)
and multiplies it into each pixel's own LINEAR radiance, in
`metal_poc.mm`'s own readback loop - BEFORE tonemapping/gamma, the
physically correct place for it (the same reason tonemapping itself
operates on linear values, not gamma-encoded ones). `strength == 0.0`
is an exact no-op; the committed scene uses `0.18`, tuned down from an
initial `0.35` that read as an unnaturally heavy "tunnel vision" darkening
rather than a subtle, realistic lens characteristic - caught by
rendering both and comparing, not assumed from the formula alone.

Purely a host-side, post-process addition - no shader, scene, or
integrator code touched, the same narrow scope steps 28/29's own
tonemap/blackbody-colour changes had.

**Result, verified two ways**: visually, a before/after render shows a
natural-reading darkened frame border, most visible at the corners,
without an obvious hard edge or artificial-looking cutoff. Numerically,
a direct pixel comparison confirms the falloff shape exactly: the
FRAME'S OWN CENTRE pixel is byte-identical before and after (vignette
== 1.0 there by construction, `r == 0`), while a corner pixel shows
real, substantial darkening - confirming the effect is a true radial
falloff centred on the frame, not a uniform darkening or an
off-centre one.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 48. Proof-of-concept, step 34: lateral chromatic aberration (done)

The last piece of the "lens realism" set alongside steps 32/33's own
aperture-shape/vignette work: a real lens focuses different wavelengths
at very slightly different magnifications, so red and blue fringe
outward/inward from green toward the frame's own edges (worse toward
the corners, exactly zero at the optical centre) - the recognisable
colour fringing around high-contrast edges near a photo's own border
real camera lenses are well known for.

Modelled the simplest physically-motivated way: `chromaticAberration()`
resamples the red channel from a position scaled slightly OUTWARD from
frame centre and blue slightly INWARD, leaving green as the untouched
reference channel - a pure radial scale about the centre already gives
zero shift exactly at the centre and a shift growing with radius
everywhere else, with no need to compute a radius explicitly. Needed a
new building block, `sampleChannelBilinear()`, since (unlike
`vignetteFactor()`/`acesFilmicTonemap()`, which only ever touch a
pixel's own already-fetched value) this is the first post-process step
that needs to resample a NEIGHBOURING pixel's own value. `strength ==
0.0` is an exact no-op.

**A real bug, caught by a targeted test rather than a casual look**:
the first version computed the frame centre as `width * 0.5` but the
per-pixel offset as `float(px) + 0.5 - centre` (a pixel-CENTRE
convention), while `sampleChannelBilinear()` treats its own input
coordinates as pixel-INDEX-aligned (an integer coordinate means "exactly
that pixel," not "the corner before it") - two different, incompatible
conventions mixed in one calculation. The bug was invisible on this
POC's own default EVEN-width renders (900, with no single centre pixel
to check exactly) and even looked deceptively fine on a naive check of
the nearest-to-centre pixel, since the resulting sub-pixel misalignment
was small enough to often round to the same 8-bit value. Caught by
rendering at an ODD resolution instead (901, which has one true,
exactly-centred pixel) and confirming its own R/B channels had shifted
anyway - mathematically impossible at genuine zero shift, so a real bug,
not noise. Fixed by computing both the centre and the per-pixel offset
in the same index-aligned convention `sampleChannelBilinear()` itself
uses.

**Result, verified two ways**: after the fix, the same odd-resolution
test confirms the frame's own true centre pixel is exactly byte-identical
with the effect on vs. off - genuine zero shift, not an approximation of
one. Visually, the committed (even-resolution) scene shows a clear,
recognisable blue/purple fringe along the sharpest high-contrast
boundary near the frame's own edge (a ceiling light's own edge against
the dark ceiling corner) - the classic real-camera CA signature - while
staying subtle everywhere else, not garish or broken-looking.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 49. Firefly clamping (done, honestly partial)

A standard Monte Carlo path tracer variance-reduction technique this
POC had never added: a rare, extremely bright single-sample outlier (a
shadow ray grazing very close to a light's own edge, giving it a tiny
solid-angle pdf and therefore a huge NEE weight) dominates a pixel's
own average out of proportion to its real probability - the classic
"salt and pepper" bright-pixel noise visible at low sample counts.
Clamping each SAMPLE's own total radiance (scaled per-channel, so hue
is preserved and only brightness is capped) to `kFireflyClampLuminance`
before folding it into the accumulator trades a small, deliberately-
accepted bias (a true outlier's excess energy is discarded, not
redistributed) for faster-converging, less noisy images - the standard
practical trade-off production renderers already make.

**Threshold tuned by actually rendering at 16 samples/pixel and
comparing, not picked from theory**: an initial, "comfortably above
every light's own emission magnitude" value of 60 turned out to have
literally NO visible or measurable effect on this scene's own worst
noise cluster; an aggressive value of 3 visibly dimmed the ceiling
lights' own legitimate direct-view brightness - an unacceptable bias.
Settled on 20, a genuine middle ground (occasionally clips a legitimate
bright sample, not "guaranteed safe" the way a much higher threshold
would be, but the trade is worth it).

**Result, reported honestly, not oversold**: a local variance
measurement in the scene's own noisiest region (a fog/volumetric NEE
hotspot near the spot light's own cone) shows only a MODEST reduction
(std-dev 47.74 → 47.14 at 16spp) - this scene's worst noise turned out
to come from generally higher-variance volumetric sampling near a
bright, narrow-cone light, not from rare EXTREME single-sample spikes
the way firefly clamping is designed to catch, so this technique doesn't
fully clean it up on its own (a real limitation, not glossed over). What
IS confirmed: the mechanism itself works (an aggressive test value
visibly changed the image), the chosen threshold produces a real,
measurable, if modest, improvement at low sample counts, and the
scene's own full-quality (384spp) showcase render is visually
indistinguishable from before - no legitimate light source dimmed at
the sample counts this POC actually ships renders at.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 50. Bilateral (edge-preserving) denoise pass (done)

A direct follow-on to step 49's own honest finding: firefly clamping
barely helped this scene's own worst noise, because that noise turned
out to be generally high-variance fog/volumetric sampling near the spot
light's own cone, not the rare-extreme-outlier kind clamping targets.
Spatial denoising is the right tool for THAT kind of noise instead.
`bilateralDenoise()` computes each output pixel as a weighted average of
its own neighbours, weighted by both spatial distance (a Gaussian in
pixel distance, `sigmaSpatial`) and how similar each neighbour's own
LUMINANCE is to the centre pixel's (a Gaussian in luminance difference,
`sigmaRange`) - two nearby pixels with similar brightness (likely the
same underlying surface, differing only by noise) smooth together;
two nearby pixels with very different brightness (likely a real edge)
barely influence each other, which is what keeps this from being a
uniform blur. Applied to the final 8-bit LDR image (after tonemapping/
gamma, not the linear HDR buffer) - the standard display-referred
approach, since a range kernel compared against raw linear values would
be dominated by the huge magnitude gap between a light source and
everything else rather than meaningfully separating "real edge" from
noise. The SAME per-pixel weight (from luminance alone) is applied to
all three colour channels together, preserving each pixel's own hue
relationship to its neighbours.

**Tuned by rendering and comparing, the same way this POC's other post-
process knobs were**: settled on radius 3 (7×7), `sigmaSpatial = 2.5`,
`sigmaRange = 20.0`. A more aggressive setting (radius 4, `sigmaRange =
80`) was tried and rejected - it visibly softened the crystal ball's own
sharp specular highlight and the checkerboard floor's own tile edges,
confirming this knob really can wash out real detail if pushed too far,
not just in theory.

**Result, verified two ways**: at 16 samples/pixel, a local variance
measurement in the scene's own previously-identified worst region (the
same fog/volumetric hotspot step 49's own doc measured) shows a real,
substantially larger reduction than firefly clamping alone achieved -
std-dev 47.14 → 38.09 (roughly 19%), against clamping's own ~1.3% there.
Whole-image std-dev also dropped (57.21 → 54.55). Visually, walls read
noticeably smoother at both 16spp and the scene's own full-quality
384spp showcase render, while the checkerboard floor's tile seams and
the crystal ball's own sharp specular highlight both stay crisp -
confirming the edge-preserving behaviour is real, not just a blur in
disguise.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 51. Proof-of-concept, step 35: clearcoat (glossy plastic) material (done)

A genuinely new material (materialType 8, left unused since step 27's
own numbering skipped straight to 9) rather than a variation on an
existing one: "car paint"/glossy plastic - a colourless specular clear
coat sitting over a genuinely diffuse, coloured base, the one look
nothing already in this scene produces (the mirror and both GGX
conductors all TINT their own reflection by the surface's own colour;
clearcoat's reflection stays colourless regardless of the base colour
underneath, exactly the physical difference between a metal and a
coated dielectric).

Modelled as a stochastic MIX of two existing, already-proven lobes - a
smooth dielectric specular coat (fixed IOR 1.5, F0 = 0.04, the same
Schlick-Fresnel helper the mirror material already uses) and a
Lambertian diffuse base - rather than a genuinely new BRDF. At each hit,
ONE random draw against the coat's own Fresnel reflectance decides which
single lobe this bounce samples (never a blend of both at once): the
specular coat with probability equal to its own reflectance (no NEE,
same delta-lobe reasoning the mirror material already uses), or the
diffuse base with the complementary probability (full NEE + cosine-
sampling, the same code shape the Lambertian branch already uses).
Sampling each lobe with probability EXACTLY equal to its own weight is
what makes this unbiased with no extra scaling at the point of the
random decision - the probability and the true contribution cancel
exactly, for both branches.

**A deliberate implementation choice worth documenting**: the diffuse
half of this material duplicates the Lambertian branch's own NEE code
(area + point + directional lights) rather than falling through to that
shared branch, unlike materialTypes 3/6/7's own "share the code, vary
only how albedo/the normal is computed upstream" pattern. This is
necessary here, not just convenient - the stochastic coat-vs-diffuse
decision has to happen BEFORE any NEE, so there is no single shared
entry point left to reuse. A second, fully self-contained branch keeps
this addition from touching (and risking) any of the four already-
proven material types sharing that code.

Added as a fifth sphere, a deep red "car paint" look, placed close to
the camera on the room's front-right (deliberately at a different x
than the gold sphere it would otherwise sit right next to, and in FRONT
of rather than behind Spot-the-cow - a lesson learned from step 34's own
placement, where sharing an x coordinate with a nearer object hid the
new sphere completely).

**Result, verified via a dedicated diagnostic** (the same sphere
enlarged and moved to fill the frame, the same technique step 34's own
verification used): unambiguous - a rich, saturated matte red base with
two sharp, distinctly colourless specular highlights from the two
ceiling area lights riding on top, the textbook clearcoat signature, not
a subtle hint. The committed scene shows the same sphere, smaller and
partially at the frame's own edge (the same "partially cropped is fine"
precedent the giraffe/step 34's own sphere already set).

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 52. Proof-of-concept, step 36: patterned (spatially-varying) area light emission (done)

Every `AreaLight` so far has emitted a single FLAT colour, uniform
across its own surface - a real fixture (a diffuser panel, a stained-
glass window, a video screen) rarely does. This step adds spatially-
varying emission, reusing `checkerColor()` (materialType 6's own
albedo-checker helper) a THIRD time, now for light output instead of a
surface's own reflectance.

**Two genuinely separate places need this pattern evaluated, not one**,
because a light's own radiance reaches the camera two different ways:
directly (a camera/bounce ray landing ON the light quad, using the
hit's own barycentric-interpolated UV - `texCoordFor()`, the same
lookup materialType 3/6 already use) and via NEE (`sampleAreaLight()`
picking a RANDOM point on the light for a shadow ray, using that
sample's own `(u.x, u.y)` - a DIFFERENT point on the same quad, needing
the SAME pattern evaluated at ITS OWN coordinate, not the hit's). Both
call sites already had separate emission lookups before this step
(`mat.emission` for a direct hit, `light.emission` inside
`sampleAreaLight()`), so wiring the pattern into each was two small,
symmetric changes rather than one shared one. A new materialType (10)
flags which emissive triangles should evaluate the pattern on a direct
hit; `AreaLight` itself carries the matching `patternTileB`/
`patternScale` fields for the NEE side. `roughness` (already reused five
ways across materialTypes 4/5/7/9) picks up a SIXTH meaning here as the
pattern's own tile-B fraction. `patternScale <= 0.0` (every light before
this one) is an exact no-op on both paths - flat `emission`/
`light.emission`, matching every earlier render bit-for-bit.

Applied to the cool ceiling light only (the warm one stays flat, for
contrast): a 6×6 diffuser-grid pattern, tile B at 40% of tile A's own
brightness (a translucent grid, not opaque black bars).

**Result, verified two ways**: a direct render shows the cool light's
own surface with a visible checkerboard grid pattern, sharply distinct
from the warm light's own uniform flat rectangle right next to it - the
direct-hit half of this feature, unambiguous. Numerically, an exact
byte-for-byte pixel comparison confirms both ends of the "0 is a no-op"
claim: `patternScale = 0.0` on the SAME light reproduces the pre-step
render with ZERO differing pixels (a true no-op, not an approximation
of one), while `patternScale = 6.0` changes roughly 94% of the image's
own subpixels by a small amount each (the pattern's own influence on
this light's total NEE-sampled contribution, spread across every surface
it illuminates via direct and indirect light) - confirming the feature
has a real, measurable, widespread effect exactly where expected, and
none where it shouldn't.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.

## 53. Single-pass adaptive sampling (done - correct, but a real GPU-vs-CPU architecture lesson)

Ported directly from this project's own CPU integrator
(`src/shared/adaptive_sampling.h`'s own `pixel_convergence::
has_converged()`, wired into `camera.h`'s render loop): once a pixel's
own running per-sample LUMINANCE estimate is confident enough (its
relative standard error below a threshold - `0.01`, reused unchanged
from Cycles' own `adaptive_threshold` default the CPU integrator already
uses) that more samples wouldn't change its mean much, stop sampling
that pixel early instead of spending its full `samplesPerPixel` budget.
Uses Welford's online mean/variance update (matching `VarianceEstimator
::Add()`'s own formula exactly, reimplemented as plain scalars since
this is MSL, not the CPU's own C++ template class) and the identical
near-black fast-path (a pixel converging toward zero is treated as
converged immediately, matching Cycles' own behaviour of not endlessly
re-sampling background/shadow regions). The CPU version's own
`min(2 * sqrt_spp, 32)` minimum-samples-before-checking doesn't
translate directly (that integrator visits a STRATIFIED sqrt_spp x
sqrt_spp grid, this one a flat sample count), so a comparable flat
minimum (16) is used instead - the one deliberate adaptation, not a
faithfulness gap. `adaptiveSampling == 0` (available, though the
committed scene doesn't use it) is an exact no-op - none of the new
per-sample bookkeeping runs, and the final division is by the SAME
`samplesPerPixel` as before.

**A genuinely important finding, verified rather than assumed**: this
correctly REDUCES sample counts (confirmed - a full-quality on/off
comparison shows only 0.3% of the image's own subpixels differ at all,
by a negligible amount, exactly what "some pixels converged with fewer,
still-statistically-valid samples" should look like) but produces ZERO
measured wall-clock speedup on this GPU (two full 384spp renders,
identical scene, adaptive on vs off: 3m4.083s vs 3m4.507s - within
noise of each other). The reason is a real, worth-remembering GPU-vs-CPU
architectural difference this feasibility study exists to surface: the
CPU integrator's own per-pixel loop runs on independent threads where an
early `break` genuinely frees that thread to move on: a GPU compute
kernel schedules threads in SIMD-width execution groups that run in
LOCKSTEP, so an individual thread's own early exit does NOT free real
hardware time unless every OTHER thread sharing its execution group has
ALSO finished - and this scene's own per-pixel convergence times vary
too much across neighbouring pixels (a checkerboard tile edge next to a
converged flat wall, a DOF-blurred near/far boundary, a specular
highlight next to matte shadow) for that to happen in practice. Kept in
the codebase anyway - it is correct, harmless when on, and a genuine,
documented lesson about this architecture, not a discarded dead end.

Both the ad-hoc `clang++` build and the CMake `RT_BUILD_METAL` target
build and render correctly, and `ctest` continues to pass.
