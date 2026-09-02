# Feature Inventory & pbrt-v4 Gap Analysis

What this renderer actually has, backend by backend, and what it's still
missing relative to pbrt-v4. Built by reading the current source directly
(not from commit history or comments, several of which were found stale
while writing this) — see "Stale comments found" at the bottom for the ones
worth fixing.

**Scope note**: this is the broad "what features exist" survey. For the
narrower question of "what happens to each individual pbrt-v4 `.pbrt`
directive when loaded" (per-directive Full/Approx/Fallback/Unsupported,
CPU vs GPU), see [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) — that table is the
source of truth for loader fidelity; this doc doesn't repeat it.

Backends referenced throughout: **CPU** (`src/TheRestOfYourLife`,
`cpu_renderer/`), **GPU-recursive** (`gpu/optix/recursive_path_tracer.*`,
default `--gpu` mode), **GPU-wavefront** (`gpu/optix/wavefront_*`, opt-in
via `--wavefront`).

---

## Complete feature table

Quick-reference index of every feature this doc covers, in one place.
"Status" packs CPU/GPU-recursive/GPU-wavefront coverage as `Y/Y/Y`-style
shorthand where that shape fits, or plain text where it doesn't (CLI flags,
orphaned code, loader-only limits). "Fallback / Counterpart" is what
actually happens when the feature isn't there — `N/A` means the feature
works as stated and there's no fallback concept to speak of. See the
numbered sections below for the narrative detail behind any row.

| Category | Feature | Status | Fallback / Counterpart |
|---|---|---|---|
| Shapes | Sphere | Y / Y / Y | N/A |
| Shapes | Quad | Y / Y / Y | N/A |
| Shapes | Disk | Y / Y / Y | N/A (medium on a disk is structurally not meaningful, not merely unimplemented) |
| Shapes | Cylinder | Y / Y / Y | N/A |
| Shapes | Triangle / triangle mesh | Y / Y / Y | N/A |
| Shapes | Bilinear patch | Y / Y / Y | N/A |
| Shapes | Curves (hair/fiber strands) | Y (analytic) / Y (tessellated) / Y (tessellated) | GPU tessellates to a bilinear-patch tube instead of native curve intersection — matches pbrt-v4's own GPU strategy, not a degraded fallback |
| Shapes | Loop-subdivision surfaces | Y / Y / Y | N/A (refined to triangles once at load, before either backend sees it) |
| Shapes | OBJ mesh loader | Y / Y / Y | N/A |
| Shapes | PLY mesh loader | Y / Y / Y | N/A |
| Shapes | Instancing (`ObjectInstance`) | Y / Y / Y | N/A |
| Shapes | Non-cubic / non-Bezier curve (`splitdepth`, other basis/degree) | N | Dropped with a "shape not supported" warning — nothing rendered in its place |
| Materials | Lambertian | Y / Y / Y | N/A |
| Materials | Metal (fuzz) / Conductor (real complex-IOR GGX) | Y / Y / Y | N/A |
| Materials | Dielectric (smooth) | Y / Y / Y | N/A |
| Materials | Rough dielectric | Y / Y / Y | N/A |
| Materials | Thin dielectric | Y / Y / Y | N/A |
| Materials | Coated diffuse | Y / Y / Y | N/A |
| Materials | Coated conductor | Y / Y / Y | N/A |
| Materials | Diffuse transmission | Y / Y / Y | N/A |
| Materials | Mix material | Y / Y / Y | N/A |
| Materials | Subsurface (tabulated BSSRDF) | Y / Y / Y | N/A |
| Materials | Hair (Marschner/Chiang) | Y / Y / Y | N/A |
| Materials | Measured (`.bsdf` tensor) | Y / Y / Y | Unresolved filename → falls back to Lambertian (same gate on both backends) |
| Materials | Principled | Y / Y / Y | N/A |
| Materials | Dispersion (`dielectric`, `rough_dielectric`) | Y / Y / Y | GPU-recursive uses a simplified 3-representative-wavelength (R/G/B) stochastic scheme rather than CPU/GPU-wavefront's continuous hero-wavelength `SampledWavelengths` integration - a real chromatic fan, just coarser |
| Materials | Unrecognized pbrt `Material` kind | N | Falls back to flat Lambertian using base color, warned by name |
| Textures | Procedural (checker/noise/marble/windy/dots/etc.) | Y | N/A |
| Textures | Image textures + mipmap/EWA filter | Y | N/A |
| Textures | Texture mapping (UV/spherical/cylindrical/planar) | Y | N/A |
| Textures | `mix` texture with texture-bound `amount` | N (loader only — class works via native API) | Falls back to flat colour with a warning when loaded from `.pbrt` |
| Textures | checkerboard/mix tex1/tex2 nesting a further procedural texture (2nd level) | Y, CPU only | GPU approximates as a flat average colour, warned; a 3rd level of nesting still falls back with a warning on both |
| Lights | Point / spot / distant | Y / Y / Y | N/A |
| Lights | Goniometric | Y / Y / Y | Missing/non-square profile image → falls back to a uniform isotropic distribution |
| Lights | Projection | Y / Y / Y | Missing `filename` → warned (pbrt-v4 itself requires one) |
| Lights | Area (any NEE-samplable shape) | Y / Y / Y | N/A |
| Lights | Uniform infinite | Y / Y / Y | N/A |
| Lights | Image infinite (HDRI, importance-sampled) | Y / Y / Y | N/A |
| Lights | Portal light | Y / Y / Y | N/A |
| Lights | Unrecognized pbrt light kind | N | Dropped, **no fallback rendered** — the one case in the whole loader with no safety net |
| Light sampling | BVH light sampler (spatial + power) | CPU default | GPU counterpart is the flat `PowerLightSampler` — a real, permanent substitute, not a degraded fallback |
| Light sampling | Power light sampler (flat) | CPU (superseded) / GPU (default) | N/A |
| Light sampling | `UniformLightSampler` | Dead code, zero callers | N/A — orphaned, not part of any fallback chain |
| Light sampling | `BVHLightSampler2` | Dead code, zero callers | N/A — orphaned |
| Light sampling | `ExhaustiveLightSampler` | Dead code (only reachable from unwired `restir.h`) | N/A — orphaned |
| Light sampling | `PowerLightSampler` (`src/shared/power_light_sampler_scaffold.h`) | Dead code, zero callers | N/A — orphaned; not the same file as `src/TheRestOfYourLife/power_light_sampler.h`, which is real and CPU-default (see row above) |
| Media | Homogeneous | Y / Y | N/A |
| Media | Cloud (procedural Perlin-FBm) | Y / Y | N/A |
| Media | RGB grid | Y / Y | N/A |
| Media | Uniform grid | Y / Y | N/A |
| Media | NanoVDB | Y, CPU only | GPU falls back to flat homogeneous fog (whole boundary shape), warned - see §5 |
| Media | Medium on disk / triangle mesh / etc. (GPU) | N | No fallback — GPU medium dispatch is sphere/cylinder-triggered only |
| Cameras | Perspective (+ depth of field) | Y / Y / Y | N/A |
| Cameras | Orthographic | Y / Y / Y | N/A |
| Cameras | Spherical (equirect + equal-area) | Y / Y / Y | N/A |
| Cameras | Realistic (lens-file simulation) | Y / Y / Y | Missing/unreadable lens file → falls back to perspective, warned |
| Cameras | Motion blur (camera, default perspective camera) | Y / Y / Y | N/A |
| Cameras | Motion blur (object: Sphere) | Y / Y / Y | N/A |
| Cameras | Motion blur (object: Disk/Cylinder) | Y (CPU) | GPU (both backends) renders static at the StartTime position, warned |
| Cameras | Motion blur (camera, alt camera models: Orthographic/Spherical/Realistic) | Y (CPU) | GPU has no alt-camera-plus-motion-blur support; falls back to static, warned |
| Cameras | Motion blur (object: mesh/curve/bilinear patch) | N | No fallback — static transform only |
| Cameras | `ActiveTransform`/`TransformTimes` (`.pbrt`-authored animated CAMERA or Shape) | Y / Y / Y | Real directives, all three backends, for both camera and the shapes above |
| Samplers | Sobol / Z-Sobol / padded Sobol / stratified / PMJ02BN / Halton | Y (CPU) | N/A |
| Samplers | Blue noise (bonus, non-pbrt-v4) | Y (CPU) | N/A |
| Samplers | Independent | Y (CPU) | N/A |
| Samplers | `--sampler` under GPU/BDPT/MLT/SPPM/debug integrators | Out of scope by design | Warned and ignored, doesn't error |
| Integrators | Path (default) | Y / Y / Y | N/A |
| Integrators | VolPath (media-aware) | Y / folded into default / Y | N/A |
| Integrators | SPPM | Y / dedicated GPU pipeline / — | N/A |
| Integrators | BDPT | CPU only | `--gpu` request → forced onto CPU, warned, not an error |
| Integrators | MLT | CPU only | Same — forced onto CPU, warned |
| Integrators | RandomWalk / AO / SimplePath / SimpleVolPath / LightPath (debug) | CPU only | N/A (no GPU variant is ever attempted) |
| Integrators | `--denoise` (OptiX AI denoiser) | GPU-recursive and GPU-wavefront | Wavefront has its own independent denoiser/AOV-buffer implementation (WavefrontPathTracer::denoise(), not shared with OptiXRenderer's) - real support now, previously a silent (then warned) no-op |
| Spectral | `--spectral` (hero-wavelength Monte Carlo) | CPU, default path tracer only | Combined with GPU/SPPM/BDPT/MLT/debug → flag silently dropped, warned; render proceeds without it |
| Spectral | GPU-wavefront's internal spectral pipeline | Always-on, GPU-wavefront | N/A (not a flag, not togglable — this is just how the integrator works) |
| Spectral | Per-pixel (not per-sample) XYZ→RGB reduction, avoiding early gamut-clamp bias | Y | See §9 — `PixelSensor`/`SpectralFilm` themselves remain dead code, but the behavioral gap they existed to close is fixed directly |
| Acceleration | CPU hand-rolled BVH (`bvh.h`) | Y | N/A |
| Acceleration | CPU SAH/HLBVH BVH (`bvh_aggregate.h`) | Y | N/A |
| Acceleration | GPU hardware BVH (OptiX `OptixTraversableHandle`) | Y | N/A |
| Acceleration | Light BVH on GPU | N | Falls back to the flat `PowerLightSampler` (see Light sampling row above) — permanent, not degraded |
| Acceleration | `kd_tree.h` | Present, primary consumer unconfirmed from a source scan | N/A — worth a follow-up to confirm live vs orphaned |
| Tooling | Interactive/real-time progressive preview | N | No fallback — Qt GUI only launches the CLI as a subprocess and parses output; pbrt-v4's own reference implementation also lacks this, so it tracks upstream |
| Tooling | Tone mapping (ACES / Reinhard / none) | Y | N/A |
| Tooling | Video generation (`--video`) | Y | N/A |
| Tooling | System-compatibility diagnostics (`--diagnose`) | Y | N/A |
| Orphaned scaffolding | ReSTIR / reservoir sampler (`restir.h`) | Present, unwired, beyond pbrt-v4 book scope | N/A — not part of any fallback chain |
| Orphaned scaffolding | `PixelSensor` / `SpectralFilm` | Present, unwired | N/A |
| Loader-only | `ColorSpace` `.pbrt` directive | Y | N/A |
| Loader-only | `CoordinateSystem`/`CoordSysTransform` `.pbrt` directives | Y | N/A |
| Loader-only | `Accelerator` `.pbrt` directive | Y | See §10 and the "Summary: gaps" list's closed item 6 - real splitmethod/maxnodeprims, not just parsed-and-discarded |

---

## 1. Shapes

| Shape | CPU | GPU-recursive | GPU-wavefront | Notes |
|---|:-:|:-:|:-:|---|
| Sphere | Y | Y | Y | `sphere.h` |
| Quad | Y | Y | Y | `quad.h` |
| Disk | Y | Y | Y | `disk_cylinder_hittable.h` |
| Cylinder | Y | Y | Y | `disk_cylinder_hittable.h` |
| Triangle / triangle mesh | Y | Y | Y | `triangle.h` |
| Bilinear patch | Y | Y | Y | `src/shared/bilinear_patch.h` |
| Curves (hair/fiber strands) | Y — real analytic ray-Bezier | Y — tessellated to bilinear-patch tube | Y — tessellated | Matches pbrt-v4's own CPU-vs-GPU strategy split, not a gap. `type` (flat/cylinder/ribbon) is ignored on GPU — all tessellate to the same round tube. |
| Loop-subdivision surfaces | Y | Y | Y | Refined to triangles once at load (`loop_subdivide.h`), so both backends just see triangles. |
| OBJ mesh loader | Y | Y | Y | `mesh.h` (+ `mesh_mtl.h` for `.mtl`) — hand-written parser. |
| PLY mesh loader | Y | Y | Y | `src/shared/ply_mesh.h` — ASCII + binary LE/BE, arbitrary vertex properties. |
| Instancing (`ObjectInstance`) | Y | Y | Y | `transform_instance.h` + `pbrt_scene.h`'s `ObjectBegin/End/ObjectInstance`. |
| Cone | Y | N | N | `src/shared/shapes.h`'s `ConeShape<T>`, wrapped by `cone_hittable` (`cone_paraboloid_hittable.h`). Geometry-only (no `AreaLightSource`/`MediumInterface`); GPU warns and drops the shape at load time. |
| Paraboloid | Y | N | N | `ParaboloidShape<T>`, wrapped by `paraboloid_hittable` (`cone_paraboloid_hittable.h`). Same v1 scope as Cone. A ray exactly on the symmetry axis is an accepted miss, matching Cylinder's identical parallel-ray limitation. |
| Hyperboloid | N | N | N | Not implemented — a twisted ruled surface between two arbitrary points, meaningfully harder than the other quadrics and rare in practice; falls through to the generic "shape not supported" warning. |

Shape gap versus pbrt-v4's own set: Hyperboloid (unimplemented on every
backend), and Cone/Paraboloid are CPU-only (see table above).

## 2. Materials / BxDFs

Every pbrt-v4 material class has a real implementation on **every**
backend, including GPU-wavefront (the newest and most gap-prone target):
`lambertian`, `metal`(legacy fuzz model)/`conductor`(real complex-IOR GGX),
`dielectric`, `rough_dielectric`, `thin_dielectric`, `coated_diffuse`,
`coated_conductor`, `diffuse_transmission`, `mix_material`, `subsurface`
(real tabulated BSSRDF + device probe-walk), `hair` (real Marschner/Chiang
fiber scattering), `measured` (real Dupuy & Jakob tensor `.bsdf` loader +
eval on all three backends), `principled`, `diffuse_light`, `isotropic`
(medium phase-function material), normal/bump mapping wrappers.

**Dispersion** (wavelength-dependent IOR): both `dielectric` and
`rough_dielectric`, on **CPU and GPU-wavefront**. CPU: same two-term Cauchy
formula for both (`dielectric::make_dispersive(eta_d, abbe_number)` /
`rough_dielectric::make_dispersive(eta_d, abbe_number, roughness)`,
`material_simple.h` / `material_pbrt.h`), reachable through `--spectral`.
`rough_dielectric`'s CPU dispersive path is real NEE/MIS, not an
approximation reusing the smooth material's specular-only shortcut: a delta
light (point/spot/distant) is reachable only via NEE, never by chance
through BSDF sampling, so `scattering_pdf()` itself had to become
wavelength-aware too (`scattering_pdf_dispersive()`), not just the initial
`scatter()`. GPU-wavefront: `gpu/optix/scene_builder.cpp`'s
`add_dispersive_dielectric()`/`add_dispersive_rough_dielectric()` derive the
same Cauchy coefficients at scene-build time and store them in
`MaterialData`'s `dispersive_extra` union slot (`optix_types.h`);
`wavefront_kernels.cu`'s `evaluate_materials_dielectric()` resolves the
per-hit ior via `CauchyEta()` at the path's hero wavelength when dispersive,
for both its `Dielectric` and `RoughDielectric` cases, reusing wavefront's
own always-on `SampledWavelengths<4>` hero-wavelength pipeline (no new
wavelength-tracking infrastructure needed - unlike GPU-recursive, wavefront
already threads hero wavelengths through every path for its internal
spectral-upsampling pipeline). See `B23`/`B24` (Glass/Frosted Prism
Dispersion, Materials category) - both `gpu_compatible=true`, verified
against CPU's `--spectral` reference render on both GPU backends. Fixed a
real, pre-existing wavefront bug found while verifying this: the smooth
`Dielectric` case re-derived entering/exiting from
`dot(rayDir, normal) < 0` against the already-front-face-flipped `normal`
(always false-negative, always took the "entering" branch) instead of the
correctly-tracked `h.frontFace` the `RoughDielectric` case next to it
already used - harmless-looking for a single-bounce sphere (A1) but visibly
wrong for the prism's many internal total-internal-reflection bounces.
GPU-recursive now has real dispersion too, for both `dielectric` and
`rough_dielectric` - see §9 for the approximation it uses.

**Not a gap, just a scope note**: `--spectral`'s own material whitelist
(`cpu_interface.cpp`'s `spectral_scan_hittable()`, via the `spectral_scan_material()`
helper it calls per leaf primitive) covers 6 of the materials above —
`lambertian`/`metal`/`dielectric`/`rough_dielectric`/`conductor`/
`diffuse_light` — plus `mix_material`, which isn't itself a color/BSDF to
check but recurses into whichever two of the above it mixes. A scene using
`coated_diffuse`, `subsurface`, `hair`, `measured`, or any medium (or a
`mix_material` wrapping one of those) fails `--spectral` loudly at load
time rather than silently rendering wrong. GPU-wavefront's own internal
spectral pipeline (see §9) has no such restriction — it's always-on and
covers every material GPU-wavefront supports at all.

## 3. Textures

Procedural: checker (2D + 3D), noise/FBm, marble (cubic-Bezier color ramp,
matches pbrt-v4's 9 control points), windy, wrinkled (turbulence), dots,
scale, mix, bilerp, solid color — `src/shared/procedural_textures.h` +
`src/TheRestOfYourLife/texture.h`.

Image textures: Y, with a real mipmap + EWA filter
(`src/shared/mipmap.h`, ported from pbrt-v4's `mipmap.cpp` — 128-entry
Gaussian LUT), point/bilinear/trilinear/EWA all selectable.

Texture mapping: all 4 pbrt-v4 2D mapping types ported — UV, spherical,
cylindrical, planar (`src/shared/texture_mapping.h`).

**Gap (narrowed)**: `checkerboard`/`mix` texture nesting is now real up to
TWO levels on CPU (`tex1`/`tex2` binding to a further `checkerboard`/`mix`
Texture, not just a bare imagemap or flat literal — `pbrt_flatten::
NestedProceduralTexture`) — GPU still approximates a nested slot as a flat
average colour rather than rendering it for real (warned, not silent). A
THIRD level (that inner texture's own `tex1`/`tex2` naming yet another
procedural texture) is a deliberate, disclosed cap, not attempted. `mix`'s
own `amount` parameter separately supports one level of texture binding
(a spatially-varying blend mask) but not a second — see `PBRT_SUPPORT.md`
for the exact scope cut (this is a loader-parsing limit for the deeper
cases, not a missing texture *class*; the classes exist and work when
built through the native C++ scene API).

## 4. Lights & Light Sampling

All 5 pbrt-v4 punctual light kinds, plus infinite/portal, all real on both
CPU and GPU: point, spot, distant, goniometric (real IES-equirect image
decode), projection (real slide-image decode), area (any NEE-samplable
shape, with real per-point image emission + `twosided`), uniform infinite,
image-based infinite (real importance-sampled HDRI), portal (bounded image
infinite light).

**Light samplers** — this is where real backend asymmetry exists:

| Sampler | Where it's actually wired in |
|---|---|
| `bvh_light_sampler` (light BVH, spatial+power) | **CPU default**, `cpu_interface.cpp` |
| `power_light_sampler` (`src/TheRestOfYourLife/`) | CPU, superseded by the BVH sampler as default |
| pbrt-v4's PowerLightSampler algorithm (flat, power-only) | **GPU default (both recursive and wavefront)** - implemented natively in `gpu/optix/optix_renderer_scene.cpp`/`optix_types.h`, NOT via the C++ class below despite the shared algorithm name |
| `UniformLightSampler` | present, **zero callers** — dead code |
| `BVHLightSampler2` | **host-side tree build/upload now real** (`OptiXRenderer::buildScene()`, GPU-recursive only) - device-side traversal exists (`gpu_light_bvh_sample_index()`/`gpu_light_bvh_pmf()`, `optix_device_helpers.h`) but is unreachable by design, see the gap note below |
| `ExhaustiveLightSampler` | only referenced by the unwired `restir.h` (see §11) — not reachable from any render path |
| `PowerLightSampler<MaxLights>` class (`src/shared/power_light_sampler_scaffold.h`) | present, **zero callers** — dead code; not the GPU row above despite the name |

**Gap**: no light BVH on GPU — GPU light sampling is always the flat
power-weighted sampler, which scales worse (linear alias-table draw, no
spatial pruning) on scenes with many lights than CPU's spatial BVH sampler
does. This is a real perf-scaling gap, not a correctness one.

A GPU-recursive light-BVH port was attempted: `BVHLightSampler2`'s real
SAH tree build now runs for every scene (`OptiXRenderer::buildScene()`,
`optix_renderer_scene.cpp`) and uploads a genuinely correct, verified
(host-side-inspected) tree to the device. But every device-side read of
that uploaded tree - even a single harmless-looking `lightBvhNodes[0]`
dereference, with no traversal logic at all - reproducibly triggers a CUDA
700 illegal-memory-access on at least one real multi-light scene
(`pbrt_scenes/triangle-fan-light.pbrt`, 5 lights). Ruled out: a shared-
function-call codegen/inlining issue (hand-inlining the whole descent at
the call site, this codebase's own established fix for this exact symptom
class, made no difference); a reference-output-parameter miscompilation
(switching to a by-value struct return made no difference); an out-of-
range index (defensive bounds checks never fired before the crash still
happened). Root cause NOT established. `LaunchParams::lightBvhNodeCount`
is therefore deliberately left at its zero-init default at launch time
(see `OptiXRenderer::render()`'s own comment, `optix_renderer_render.cpp`)
so every NEE call site falls back to the alias table unconditionally -
the device code is written, present, and believed correct, but genuinely
unreachable until this is properly root-caused.

## 5. Media / Volumes

Homogeneous, cloud (procedural Perlin-FBm), RGB grid (per-voxel RGB
sigma_a/sigma_s), uniform grid (single-channel density × RGB sigma_s) — all
four real, on CPU and GPU, DDA majorant-grid delta tracking
(`src/shared/grid_medium.h`). NanoVDB (below) makes a fifth, CPU-only.

**Gap (narrowed)**: `MakeNamedMedium "nanovdb"` is now real, **CPU only** —
pbrt-v4's primary real-world-volumetric-data path (VDB files from
Houdini/etc.), previously the clearest gap in this codebase versus
pbrt-v4. `pbrt_cpu_builder.h` reads the named float density grid from the
`.nvdb` file via a vendored, header-only NanoVDB reader
(`src/external/nanovdb/` — NVIDIA's own `NanoVDB.h`/`io/IO.h`, Apache-2.0,
zero required third-party dependencies for uncompressed grids) and bakes
its active index region into a dense flat array, reusing
`GridMediumData<double>`/`grid_medium_hittable.h` (the same machinery
`"uniformgrid"` already used) completely unchanged. See
`pbrt_scenes/nanovdb-medium.pbrt` (scene `E9`) for a worked example, using
a small synthetic test grid (`pbrt_scenes/nanovdb-sphere.nvdb`) authored
directly via NanoVDB's own header-only grid-construction tools — no
external asset download, no OpenVDB dependency. Real blackbody emission is
supported via a second named `"string temperaturename"` grid (per-voxel
Kelvin, converted to RGB and weighted by `sigma_a/sigma_t` at each scatter
event, same convention as `"rgbgrid"`'s own per-voxel `"Le"`) - `sigma_a`
is only forced to 0 (pure scattering) when no temperature grid is given.
Real, disclosed scope cuts: only a single named `float` density grid
(`"gridname"`, default `"density"`); no animated/sequence grids; no other
NanoVDB build types (`Vec3f`/`Mask`/etc.); the sparse grid is densified
at load time rather than sampled natively sparse (a real memory/scope
tradeoff, not a NanoVDB limitation), capped at 512 voxels/axis
(`pbrt_cpu_builder.h`'s `kMaxVoxelsPerAxis` - also closes a real
integer-overflow-into-heap-overflow risk a code-review pass found for a
corrupt/malicious file claiming an extreme bbox). **GPU has no NanoVDB
support at all** — a nanovdb medium falls through to GPU's generic
homogeneous-
medium path, rendering as flat fog filling the whole boundary shape
(using the scene's own sigma_a/sigma_s) rather than the real sparse
density field, warned explicitly (`scene_builder.cpp`) since this is a
visibly *wrong* render on GPU, not merely an absent one. NanoVDB's own
format is explicitly designed to need no deserialization on GPU (the raw
file bytes already are the traversable structure), which could make GPU
support cheaper than a typical CPU-to-CUDA port if attempted later — but
this codebase has two prior unresolved GPU device-crash precedents on
non-trivial device call graphs (`CloudMedium::compute_density()`'s
member-call stall, the light-BVH CUDA 700 crash — see §4's own gap), so
it's scoped as a genuinely separate follow-up round, not attempted here.

**Gap**: GPU medium dispatch (all types) is sphere-hit-triggered only — a
`MediumInterface` on a disk/cylinder/trianglemesh has no effect on GPU
(cylinder gained real homogeneous-medium support since; disk is
structurally not meaningful — zero-thickness, no "inside" to bound).
**Trianglemesh specifically canary-tested and confirmed blocked** (same
session as §6's Disk/Cylinder motion-blur dead end): adding one unused
`int` field to `TriangleData` alone reproduced the identical 100x+
`optixModuleCreate` slowdown - see §6's own note on this being a module-
wide `optix_programs.cu` issue, not specific to any one struct. Not
attempted further pending diagnosis.

## 6. Cameras

All 4 pbrt-v4 camera types, full parity CPU/GPU: perspective (+ depth of
field), orthographic, spherical (equirect + equal-area), realistic
(lens-file-based, real multi-element simulation on both backends).

**Camera motion blur**: real, on CPU (default path tracer + SPPM,
both of which share `camera::get_ray()`) and on **both GPU backends**
(recursive and wavefront) — `src/TheRestOfYourLife/camera.h`'s
default perspective `camera` class now supports a `camera_is_animated`
keyframed camera-to-world, built on `src/shared/animated_transform.h`'s
`AnimatedTransform` (real keyframe interpolation, tested
`tests/unit/animated_transform_tests.cpp`), which was a complete but wholly
orphaned utility before. Design matches pbrt-v4's own `PerspectiveCamera`:
the per-pixel viewport geometry (`pixel00_loc`/`pixel_delta_u/v`/
`defocus_disk_u/v`) is computed once in **local camera space**, and each
ray's sampled shutter time picks a camera-to-world transform interpolated
between two keyframes (`lookfrom`/`lookat` at `shutter_open`,
`lookfrom1`/`lookat1` at `shutter_close`) — see `D13` (Cameras category) for
a native demo. Static cameras (`camera_is_animated=false`, every scene that
doesn't author motion blur) are a true no-op — the existing world-space fast
path is untouched. GPU (both backends) mirrors this exactly: the two
camera-to-world keyframes are decomposed into translate+rotate ONCE,
host-side, by the same `AnimatedTransform` class (`build_gpu_animated_
camera_params()`, `gpu/optix/scene_builder.cpp`), and each ray interpolates
them via a per-ray lerp/slerp (`GpuCameraParams::animated`, `gpu/optix/
optix_types.h`; `generate_primary_ray()`/`wf_generate_primary_ray()`) —
`GpuCameraParams::animated=0` (the zero-init default, every scene before
this feature existed) is the same true no-op. SPPM's blur is real but visually subtle compared to the
default path tracer's obvious streaking on the same scene — confirmed via
debug tracing (the visible point genuinely swings across the full keyframe
range per iteration) and a controlled crop comparison against a static SPPM
render (D13's sphere silhouette is measurably softer than A1's) — SPPM's own
photon-density smoothing partially masks the directional blur signal, not
a bug.

**Real `.pbrt` authoring, not just the native demo**: `ActiveTransform`
`"StartTime"`/`"EndTime"`/`"All"` and `TransformTimes` are now real,
recognized directives (`src/shared/pbrt_scene.h`) - `GraphicsState` carries a
second ("EndTime") CTM slot alongside the existing one, gated by
`activeTransformBits`, so a scene's own two `LookAt`/`Transform` blocks
(pbrt-v4's real authoring idiom) resolve into two genuinely different
`worldToCamera`/`worldToCameraEnd` matrices at `WorldBegin`, threaded through
`pbrt_flatten::Camera` (`isAnimated`/`lookfrom1`/`lookat1`/`shutterOpen`/
`shutterClose`, the last two read from the Camera directive's own real
`"shutteropen"`/`"shutterclose"` parameters, previously unparsed entirely)
and wired into a real `camera_is_animated` CPU camera by
`scene_registry.h`'s `setup_camera` hook, AND into `GpuCameraParams` for
both GPU backends (`build_loaded_pbrt_scene()`'s own `c.isAnimated` branch,
`gpu/optix/scene_builder.cpp`). `TransformTimes`'s own two floats
are read for real but have no separate effect from `shutteropen`/
`shutterclose` — this codebase's own `AnimatedTransform` construction uses
ONE pair of times for both the keyframes' own timestamps and the shutter's
random-sampling window (unlike real pbrt-v4, which keeps them conceptually
distinct), so a scene declaring `TransformTimes` with a value different from
`shutteropen`/`shutterclose` gets a loud warning rather than silently
following the wrong one. Verified end-to-end via a real `.pbrt` scene
(`ActiveTransform`-authored camera pan): CPU and both GPU backends all show
genuine motion-blur streaking.

**Object motion blur** (this is where the stale phrasing this doc used to
carry - "`ShapeDecl::xform` is still a single static `Matrix4`" - has been
corrected): `ActiveTransform "StartTime"/"EndTime"` around a `Shape` is a
real, recognized directive (`pbrt_scene.h`'s `ShapeDecl::xformEnd`, a second
CTM slot alongside the existing one), and both Sphere and Disk/Cylinder
resolve their real end-time transform at each ray's own sampled time -
**CPU only** for Disk/Cylinder (`disk_cylinder_hittable.h`'s `AnimatedTransform`
use - real TRS decomposition, not a naive per-element matrix lerp, so a
rotating keyframe pair doesn't visibly shear); Sphere has this on **all
three backends** (`Sphere::center1`, a simpler world-space-point lerp since
a sphere is rotation-invariant - see this doc's own earlier note on that).
**Gap**: GPU (both backends) renders a moving Disk/Cylinder **static**, at
its StartTime position, warned explicitly
(`BuildStats::animatedDiskCylinderCount`, `gpu/optix/scene_builder.cpp`) -
porting this to GPU means every intersection/closest-hit/NEE-sampling call
site that reads `DiskData`/`CylinderData`'s `o2w`/`w2o` (several sites, both
backends, since each has its own separately-compiled intersection program)
becoming per-ray-time-aware, a materially bigger port than Sphere's single
`center1` field ever needed.

**Attempted and abandoned**: a full port (host-side T/R/S decomposition via
`AnimatedTransform`, a shared device interpolation header, per-ray-time-
aware intersection/closest-hit/probe programs on both backends, a
conservative 17-sample swept AABB matching CPU's own bbox-under-rotation
fix) was implemented and unit-tested correctly, but hit a severe, rigorously
bisected OptiX compile-time regression: adding ANY new field to
`DiskData`/`CylinderData` in `optix_types.h` - even a single unused `int`,
independent of size/type/alignment (tested `float3`/`float4` vs. plain
float arrays, tested with/without the scale-matrix fields) - made
`optixModuleCreate` (PTX->SASS) take 100-400x longer (300ms baseline to
30s-116s) for EVERY render, regardless of whether the scene has any disk/
cylinder shapes at all. Root cause not found (no tooling available this
session to profile `ptxas`/`optixModuleCreate` internals - would need
NVIDIA Nsight or verbose `ptxas` diagnostics). Joins this codebase's other
undiagnosed GPU-compiler dead ends (`CloudMedium::compute_density()`'s
member-call stall, the light-BVH CUDA 700 crash, both elsewhere in this
doc) - don't re-attempt this exact approach (growing these two structs)
without new diagnostic capability; a design that avoids touching
`DiskData`/`CylinderData` at all (e.g. a separate parallel array indexed
by primitive id) is untried and might sidestep whatever this is.

**IMPORTANT, broader confirmed pattern** (same session): this is NOT
specific to Disk/Cylinder. Two follow-up canary tests - bumping
`OptixPipelineCompileOptions::numPayloadValues` by 1 (unused, no register
actually packed anywhere) in `optix_renderer_init.cpp`, and separately
adding one unused `int` field to `TriangleData` - each independently
reproduced the identical 100x+ slowdown, in isolation, with nothing else
changed. A control test (deleting and freshly rebuilding `optix_programs.
ptx` from completely unchanged source) stayed fast (helper `loadPTX()`
call to first pixel in ~300ms), ruling out "any freshly rebuilt PTX is
slow" as an explanation - this is genuinely content-sensitive. **Current
best understanding: `gpu/optix/optix_programs.cu` (the recursive
backend's single OptiX module) appears to be sitting at some threshold
where almost ANY additional growth - a per-primitive struct field, a
payload register, tested independently on three unrelated structures -
pushes `optixModuleCreate` into a ~100-400x slower compilation path.**
Until this is diagnosed (needs NVIDIA Nsight or verbose `ptxas`/OptiX
compiler diagnostics, unavailable this session), treat ANY struct/payload
growth to the recursive backend as high-risk and cheap to canary-test
first: add one trivial unused field, rebuild fresh, time a small render
before investing in real implementation - both blocked attempts above
were caught this way in under 10 minutes each rather than after a full
implementation.

Triangle meshes/bilinear patches/curves have no
motion-blur representation at all on any backend (a mesh bakes to static
world-space vertices at load time - real per-vertex motion would need a
second vertex-position keyframe carried all the way through, a separate,
larger feature).

The three **alternate** camera models (`src/shared/cameras.h`'s
Orthographic/Spherical/Realistic classes) now have real motion blur too,
CPU only - `ProjectiveCameraBase<T>`/`SphericalCamera<T>`/`RealisticCamera<T>`
each carry an optional `AnimatedTransform anim_camera_to_world` (same
TRS-decomposed interpolator, `src/shared/animated_transform.h`, as the
default perspective camera's own `camera::anim_cam_to_world_`), sampled at
each ray's own time and substituted for the static `camera_to_world` when
set. `scene_registry.h`'s `setup_camera()` lambda builds it from the same
`cam.lookfrom1/lookat1/shutter_open/shutter_close` state
`camera_is_animated` already populates from a scene's real `ActiveTransform
"StartTime"/"EndTime"` pair, and attaches it to whichever alt camera the
scene's own `Camera` type built - so `camera_is_animated` + an alt camera
model is no longer a silent drop (camera.h used to warn and ignore motion
blur whenever both were set; that warning is gone because the combination
now genuinely works). `camera::get_ray()`'s alt-camera dispatch also
switched from unscaled `random_double()` to a proper
`shutter_open + random_double()*(shutter_close-shutter_open)` time sample,
matching the default path's own formula - a real (if usually invisible,
since most scenes keep the [0,1] default) bug fix needed for a non-default
shutter window to sample the right portion of the animation. GPU has no
alt-camera-plus-motion-blur support at all (already warned/falls back
before this round; unaffected either way).

No other camera gap.

## 7. Samplers

Sobol, Z-order Sobol, padded Sobol, stratified, PMJ02BN, Halton, and
independent — all real, selectable via `--sampler` (`src/shared/
independent_sampler.h` for the last one: a plain per-pixel-seeded
`RNG`/PCG32 draw with no stratification, matching pbrt-v4's own
`IndependentSampler` exactly — Sobol still strictly dominates it in
practice, so it exists for `.pbrt`/`--sampler` fidelity, not as a
recommended choice). Bonus (non-pbrt-v4): blue-noise sampler.

**Gap**: `--sampler` is CPU-default-path-tracer-only — no effect on GPU, or
under BDPT/MLT/SPPM/debug integrators (warns, doesn't error).

## 8. Integrators / Render Modes

| Integrator | CPU | GPU-recursive | GPU-wavefront |
|---|:-:|:-:|:-:|
| Path (default) | Y | Y | Y |
| VolPath (media-aware default) | Y | folded into default | Y |
| SPPM | Y | dedicated GPU pipeline | — |
| BDPT | Y only | — | — |
| MLT | Y only | — | — |
| SimpleVolPath / RandomWalk / AmbientOcclusion / SimplePath / LightPath (debug integrators) | Y only | — | — |

**Gap**: BDPT, MLT, and every debug integrator are CPU-only — no GPU
implementation exists or is planned to exist for any of them (they're
diagnostic/reference-correctness tools, not the primary render path).
This mirrors pbrt-v4's own book scope reasonably closely (pbrt-v4's GPU
target, `pbrt_gpu`, is *also* path/volpath-only — it has no GPU BDPT/MLT
either), so this is arguably not a gap versus upstream at all.

**Note, not quite a gap**: `--denoise` (OptiX AI denoiser, AOV-guided) only
works on GPU-recursive — silently a no-op under `--wavefront`.

## 9. Spectral Rendering

Two independent, non-interacting spectral code paths exist:

- **CPU `--spectral`**: opt-in flag, default path tracer only, 6-material
  whitelist (fails closed on anything else), hero-wavelength Monte Carlo.
  Each sample reduces to CIE XYZ (`ray_color_spectral()`,
  `SampledSpectrumToXYZ`) — additive and always non-negative — and
  `render()`'s pixel loop filter-weight-averages XYZ across every sample in
  a pixel exactly like the RGB path averages RGB; the XYZ→RGB matrix
  multiply (which has negative coefficients — a single narrow
  hero-wavelength sample routinely lands outside the sRGB gamut) and its
  negative-clamp now happen exactly **once per pixel**, after averaging,
  not once per sample. Fixed a real, previously-unnoticed bias: clamping
  each sample's out-of-gamut RGB to 0 before averaging systematically
  desaturated/darkened near-gamut-boundary colors, since some of many
  additive samples got clipped away before they could combine with the
  rest. Supports dispersion (`dielectric` and `rough_dielectric`). See §2.
- **GPU-wavefront's internal spectral pipeline**: always-on, not a flag,
  not user-togglable — this is simply how the wavefront integrator itself
  is implemented internally (CIE tables + D65 + sRGB-upsampling table on
  device). Also supports dispersion now (`dielectric` and
  `rough_dielectric`, piggybacking on this same always-on hero-wavelength
  pipeline) — see §2.

**Gap (narrowed)**: `PixelSensor` (`src/shared/pixel_sensor.h`) and
`SpectralFilm` (`src/shared/film.h`) remain **dead code**, but the actual
behavioral gap they were built to close — early per-sample RGB reduction —
is now fixed above via `SampledSpectrumToXYZ`/`XYZToLinearRGB` directly,
which this codebase's GPU wavefront parity tests already prove correct.
Neither class is a net addition over that fix for this renderer's actual
deliverable (an RGB image, not a spectral one): `SpectralFilm`'s per-bucket
spectral storage has no consumer (nothing reads a spectral image out of
this renderer), and `PixelSensor::ToSensorRGB`'s output is scaled by
roughly `kCIE_Y_integral` (~107x) relative to `SampledSpectrumToXYZ`'s for
the same input — not confirmed as a bug (real pbrt-v4 may fold that factor
into `Film::Create()`'s calibrated `imagingRatio` rather than into
`ToSensorRGB` itself), but a real, unresolved discrepancy that makes wiring
it in with this port's simplified `imagingRatio=1` default
(`PixelSensor::CreateDefault()`) risky without deeper source verification
than this round did. Left untouched rather than guessed at.

**Gap**: GPU-recursive has no spectral path at all (RGB only) - its
dispersion support (see §2) is a separate, coarser 3-representative-
wavelength scheme layered directly into shading, not a real spectral path.

## 10. Acceleration Structures

CPU has two independent BVH implementations: a hand-rolled book-style one
(`bvh.h`, itself real SAH — pbrt-v4 §7.3 — not a naive median split) and a
full pbrt-v4-style SAH/HLBVH one (`src/shared/bvh_aggregate.h`,
`BvhSplitMethod::{SAH, Middle, EqualCounts, HLBVH}`). Both are now live: the
CPU pbrt-scene builder (`pbrt_cpu_builder.h`) reads the scene's `Accelerator
"bvh" "string splitmethod"` directive (via `FlatScene::
acceleratorSplitMethod`, resolved by `pbrt_flatten.h`) and uses `bvh_node`
for the default/`"sah"` case, or `bvh_aggregate_hittable.h`'s adapter around
`BvhTree<double,...>` for an explicit `"middle"`/`"equal"`/`"hlbvh"` — same
converged image either way, just a different build strategy. That adapter
has one disclosed limitation: `BvhTree::intersect()`'s slim primitive
interface carries no ray time, so a scene combining a non-`"sah"`
splitmethod with object motion blur (`Sphere::center1 != center`) falls
back to `"sah"`/`bvh_node` instead (warned, not silently frozen at time 0).
GPU uses OptiX's own hardware-accelerated BVH (`OptixTraversableHandle`)
for primitive traversal — not this project's own BVH code, and has no
`Accelerator`-directive concept at all. Light-sampling BVH exists
(`bvh_light_sampler.h`) but CPU-only (see §4's gap).

A `kd_tree.h` also exists in `src/shared/`; not confirmed as load-bearing
in the primary render path from a source scan alone — worth a targeted
follow-up if it turns out to be dead code too, matching the light-sampler
orphans in §11.

## 11. Present But Not Wired In (orphaned scaffolding)

These are real, apparently-complete implementations with **zero callers**
found anywhere in the render path — not gaps exactly (the code exists),
but not usable features either since nothing reaches them:

- `UniformLightSampler`, `BVHLightSampler2` (`src/shared/`) — superseded by
  `src/TheRestOfYourLife/bvh_light_sampler.h`/`power_light_sampler.h` (the
  CPU-side samplers actually wired into `cpu_interface.cpp` - not to be
  confused with the next item below, a different, same-named-until-recently
  file), left in place.
- `ExhaustiveLightSampler` — only referenced by `restir.h`, which is itself
  unwired.
- `PowerLightSampler<MaxLights>` (`src/shared/power_light_sampler_scaffold.h`
  - renamed from `power_light_sampler.h`, which collided with the unrelated,
  actually-used `src/TheRestOfYourLife/power_light_sampler.h` above) — a
  complete Vose-alias-table light sampler, zero callers; GPU's own
  power-weighted alias-table light sampling
  (`gpu/optix/optix_renderer_scene.cpp`/`optix_types.h`) implements the same
  algorithm natively rather than through this class.
- `restir.h` / `reservoir_sampler.h` (ReSTIR/RIS temporal+spatial reservoir
  reuse) — real-looking scaffolding, beyond pbrt-v4 book scope to begin
  with, never connected to any integrator.
- `PixelSensor` / `SpectralFilm` (§9) — real spectral-film classes, never
  called.

If any of these get finished and wired in, they'd be *additions beyond*
pbrt-v4 parity (ReSTIR especially — it's not part of the pbrt-v4 book at
all), not gap closures.

## 12. Tooling / CLI Surface

`--cpu`/`--gpu`, `--wavefront`, `--optix-validate`, `--denoise`, `--stats`,
`--diagnose` (system-compat report), `--exposure`, `--sampler`,
`--spectral`, `--sppm[-iterations|-photons]`, `--bdpt[-max-depth]`,
`--mlt[-bootstrap|-mutations|-max-depth]`, `--randomwalk`, `--ao[-max-dist|
-uniform|-illum-scale|-illum-rgb]`, `--simplepath[-no-lights|-no-bsdf]`,
`--simplevolpath`, `--lightpath`, `--video[-frames|-fps|-speed|
-camera-path|-preset]`. Full validation/mutual-exclusion logic in
`launcher/main.cpp`.

Tone mapping: ACES (Narkowicz approx, default), Reinhard, or none.

**Gap**: no real-time/interactive progressive preview. The Qt GUI launches
the CLI as a subprocess and parses its output — there's no live progressive
raster or interactive viewport (pbrt-v4 itself doesn't have one either in
the book/reference implementation, so this tracks upstream, not a gap
relative to it specifically).

---

## Summary: gaps from pbrt-v4, ranked by how much they'd actually matter

1. ~~NanoVDB heterogeneous media~~ (§5) — **closed, CPU only**. Was the
   clearest, most consequential gap (pbrt-v4's primary real-world
   volumetric path); now real via a vendored header-only NanoVDB reader,
   reusing the existing `GridMediumData`/DDA-majorant machinery
   unchanged. GPU has no NanoVDB support at all - see §5's own comment
   for why that's scoped as a separate follow-up, not attempted here.
2. **`PixelSensor`/`SpectralFilm` remain dead code** (§9) — the behavioral
   gap they were built to close (early per-sample RGB reduction, and the
   gamut-clamp bias it caused) is now fixed directly via
   `SampledSpectrumToXYZ`/`XYZToLinearRGB`, deferred to once per pixel; see
   §9 for why routing through either class wasn't worth the risk on top of
   that fix.
3. **GPU-recursive dispersion is approximate** (§2, §9) — GPU-wavefront has
   real continuous-wavelength dispersion (both `dielectric` and
   `rough_dielectric`, matching CPU's `SampledWavelengths` pipeline);
   GPU-recursive now has real dispersion for both `dielectric` and
   `rough_dielectric` too, but via a simplified 3-fixed-representative-
   wavelength scheme (stochastic confinement to one of R/G/B per path, not
   continuous hero-wavelength sampling) rather than porting
   `SampledWavelengths` into that backend's own separate OptiX module - a
   real chromatic fan, just coarser than the other two backends'.
   **Attempted a canary test** (same session as the Disk/Cylinder motion-
   blur dead end, §6): a real port needs ~8 new payload registers + a CIE
   table upload into the recursive module's device-constant memory; a
   minimal canary (`numPayloadValues` 25->26, unused) alone reproduced the
   identical 100x+ `optixModuleCreate` slowdown described in §6's own
   note. Blocked by the same undiagnosed module-wide issue - not attempted
   further.
4. ~~No object motion blur~~ — **narrowed further**. Sphere has real object
   motion blur on all three backends (`Sphere::center1`); Disk/Cylinder now
   do too on **CPU** (`AnimatedTransform`, real TRS interpolation, §6) — GPU
   (both backends) still renders a moving Disk/Cylinder static, warned. The
   default perspective camera's own motion blur (`AnimatedTransform`,
   previously wholly orphaned, now wired into `camera::get_ray()` and
   `GpuCameraParams::animated`) — and now the three **alternate** camera
   models too (Orthographic/Spherical/Realistic, CPU only, §6) — is
   unaffected by any of this. Meshes/curves/bilinear patches remain the one
   fully untouched case, on every backend.
5. **No GPU light BVH** (§4) — GPU light sampling doesn't spatially scale
   the way CPU's does on many-light scenes.
6. ~~`Accelerator` pbrt directive not parsed~~ — **closed**. `Accelerator
   "bvh" "string splitmethod"/"integer maxnodeprims"` is now real, parsed
   input (`pbrt_scene.h`); `pbrt_flatten.h` resolves it to `FlatScene::
   acceleratorSplitMethod` (falling back to `"sah"` for an unrecognized
   type/splitmethod, or for a non-`"sah"` splitmethod combined with object
   motion blur - see that field's own comment for why). The CPU builder
   (`pbrt_cpu_builder.h`) still uses `bvh_node` (this project's own
   pre-existing, real SAH build - also pbrt-v4's own default, so this
   covers the overwhelming majority of scenes unchanged) for `"sah"`, and
   routes an explicit `"middle"`/`"equal"`/`"hlbvh"` through a new
   `bvh_aggregate_hittable.h` wrapper around the previously-dead
   `BvhAggregate`/`BvhTree` (`src/shared/bvh_aggregate.h`) - both produce
   the exact same converged image over the same primitives (verified: a
   parameterized test renders an identical scattered-sphere scene through
   all 4 split methods and asserts pixel-identical hit results), since
   split method only changes build strategy/tree shape, never rendering
   behavior. GPU is untouched (no `Accelerator` concept there - OptiX
   builds its own hardware BVH regardless).
7. **BDPT/MLT/debug integrators CPU-only** — arguably tracks pbrt-v4's own
   GPU scope (path/volpath only), so more a parity-with-upstream item than
   a true gap.

## Fallback behavior: what happens when a feature is missing

Most gaps above degrade gracefully rather than failing outright — but not
all of them, and it's worth knowing which is which before relying on one.

**Falls back to something functional (warns, keeps rendering):**

- `MakeNamedMedium "nanovdb"` on **GPU** → falls back to a flat homogeneous
  medium filling the boundary shape, with a warning (CPU renders the real
  grid - see §5).
- A moving Disk/Cylinder (`ActiveTransform` motion) on **GPU** → falls back
  to its StartTime position, static, with a warning (CPU renders the real
  motion blur - see §6).
- An unrecognized `--sampler` name → silently falls back to Sobol
  (`independent` is now a real, recognized name — see §7).
- An unrecognized pbrt `Material` kind → falls back to flat Lambertian
  using the material's base color, with a named warning.
- `realistic` camera with a missing/unreadable lens file → falls back to
  perspective, with a warning.
- Goniometric/projection light with a missing or non-square image → falls
  back to a uniform/isotropic distribution.
- `--bdpt`/`--mlt` combined with `--gpu` → forced onto CPU with a warning
  ("--gpu is ignored under --bdpt/--mlt"), not an error.
- `--spectral` combined with `--gpu`/`--sppm`/`--bdpt`/`--mlt`/any debug
  integrator → the flag is silently dropped with a warning; you still get
  an ordinary render on whatever backend/mode you asked for.
- A `.pbrt` directive this loader doesn't recognize at all → warned and
  skipped; the rest of the scene still loads.
  (`CoordinateSystem`/`CoordSysTransform`/`ColorSpace`/`ActiveTransform`/
  `TransformTimes`/`Accelerator` are now real, recognized directives — see
  the "Loader-only" rows in the feature table above, §6's own camera-
  motion-blur entry for `ActiveTransform`/`TransformTimes` specifically,
  and §10 for `Accelerator`.)
- A `ColorSpace` directive naming something other than `srgb`/`dci-p3`/
  `rec2020`/`aces2065-1` → warned, the scene's working color space stays
  whatever it already was (`srgb` if never set).
- A `CoordSysTransform` naming a coordinate system no `CoordinateSystem`
  directive ever saved → warned, the current transform is left unchanged.

**No counterpart — genuinely absent, dropped rather than approximated:**

- An unrecognized pbrt *light* kind → dropped with a warning, and unlike
  materials there is no fallback light rendered in its place — it's simply
  invisible. This is the one case in the whole loader with no safety net:
  everything else either substitutes something visible or forces a backend/
  flag change loudly; an unsupported light just disappears.
- A `Shape` type this loader can't build (e.g. a non-cubic/non-Bezier
  `curve`) → dropped with a "shape not supported" warning; nothing is
  rendered in its place.
- The orphaned scaffolding (§11: `UniformLightSampler`, `BVHLightSampler2`,
  `ExhaustiveLightSampler`, `PowerLightSampler` (scaffold), ReSTIR,
  `PixelSensor`/`SpectralFilm`) — these aren't fallbacks *for* anything and
  don't *have* fallbacks either; they're unreachable code with zero
  callers, not part of any fallback chain.

**Working counterpart already in place (not a degraded fallback — just the
permanent behavior):**

- No GPU light BVH → GPU always uses the flat `PowerLightSampler` instead.
  That's a real, working counterpart, not a failure mode — it's the
  permanent GPU default, just with worse scaling on many-light scenes than
  CPU's spatial BVH sampler.
- No real spectral film accumulation → CPU `--spectral` reduces to RGB
  every sample rather than accumulating spectral radiance. This isn't a
  fallback triggered by failure — it's simply how the feature is
  architected; there's no better mode it's falling back *from*.

## Stale comments found while building this doc

Worth fixing separately (not done here — this is a survey, not a patch):

- (Previously noted here: `src/shared/pbrt_scene.h`'s top-of-file comment
  calling out subsurface/curves/instancing/media as unimplemented, and
  NanoVDB media as the sole remaining gap. Already fixed - the comment no
  longer exists, and NanoVDB itself is now real, CPU-side, as of this
  doc's own §5.)
