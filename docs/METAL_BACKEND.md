# Metal backend: how it is built and how to change it

The living reference for `gpu/metal/`, the macOS GPU renderer. Read this first; the other documents answer narrower questions:

| document | answers |
|---|---|
| this file | architecture, the host/shader contracts, how to add things, how to test, what the environment switches do |
| [`METAL_PARITY_STATUS.md`](METAL_PARITY_STATUS.md) | how closely Metal matches the CPU renderer, how that is measured, known differences, the large-scene sweep |
| [`MAC_LIVE_PREVIEW.md`](MAC_LIVE_PREVIEW.md) | the interactive Live Preview (the persistent session, performance) |
| [`history/METAL_GPU_FEASIBILITY.md`](history/METAL_GPU_FEASIBILITY.md) | the development diary: 200+ numbered sections of what was tried, measured and decided, in order. Source comments cite it as "section N". Search it for the *why* of an old decision; do not read it to learn how things work today |

## What it is

A Metal ray-tracing path tracer (hardware ray tracing on Apple GPUs) that renders the same flattened pbrt scene the CPU and OptiX renderers use
(`src/shared/pbrt_load.h` -> `pbrt_flatten.h`). It is one megakernel, `primaryRayKernel`, with the materials, lights and media as device functions
around it. The shader source is **compiled at run time** from the `.metal` files (no offline `.metallib`), so editing a shader needs no rebuild.

## Files

| file | role |
|---|---|
| `metal_poc_main.mm`, `metal_poc.mm` | the `metal_poc` CLI and `MetalPocApp`: argument parsing, the hand-built demo room (`buildScene`), post-processing and writing the image |
| `metal_poc_pbrt_loader.mm` | `loadPbrtScene()`: flattened pbrt scene -> host arrays (geometry, area/point/spot/distant/projection/goniometric/infinite lights, media, camera) |
| `metal_poc_pbrt_materials.mm` | pbrt material -> the shader's `TriangleMaterial` (`mapPbrtMaterial`, `mapPbrtDiffuseMaterial`), measured-BRDF tables |
| `metal_poc_gpu_resources.mm` | uploads buffers and builds the acceleration structures |
| `metal_poc_dispatch.mm` | `compileShaderAndDispatch()`: eight named stages (library, pipeline, targets, textures, sampling tables, uniforms, resource check, per-frame dispatch) over a `DispatchState` |
| `metal_poc_app.h`, `metal_poc_gpu_types.h`, `metal_poc_host_math.h`, `metal_poc_scene_helpers.h` | the host-side class, the host copies of the shader structs, math, scene-building helpers |
| `metal_poc_material_ids.metal` | the `METAL_MAT_*` names of every `materialType` (included by host and shader) |
| `metal_poc_types.metal` | the shader copies of the structs (`Uniforms`, `TriangleMaterial`, lights, media), intersection functions |
| `metal_poc_sampling.metal`, `metal_poc_layered_bxdf.metal` | sampling, microfacet, environment/portal light, layered BxDF code |
| `metal_poc_materials_*.metal` | one `shadeXxx()` per material family (specular, diffuse, layered, extra, medium, hair) |
| `metal_poc_kernel.metal` | `primaryRayKernel` |
| `metal_poc_shader_files.h` | the ordered list of shader files concatenated into one source string |
| `metal_live_preview.mm/.h` | the persistent per-scene session behind Live Preview (`realtime_renderer.dylib`) |
| `metal_poc_test_kernels.metal`, `metal_poc_shader_tests*.mm`, `metal_poc_math_tests.cpp`, `metal_poc_validate.cpp`, `metal_poc_crop_check.cpp`, `metal_realtime_dylib_check.cpp` | tests (see below) |
| `metal_cpu_gpu_parity_check.cpp`, `parity_golden.txt` | the CPU-vs-Metal sweep and its golden snapshot |

## Data flow of one render

1. `loadPbrtScene()` reads the scene and rescales it to ~2 units across (so the shader's fixed ray epsilons stay meaningful), then fills host vectors.
2. `buildGPUResources()` uploads them and builds the acceleration structures (triangles, spheres, disks, cylinders as custom intersection functions, instances).
3. `compileShaderAndDispatch()` concatenates and compiles the shader, builds the pipeline with its intersection function table, uploads textures and sampling
   tables, fills `Uniforms`, checks that no allocation failed (a nil buffer would otherwise silently bind nothing and render a *wrong* picture), then
   returns a re-runnable `renderFrame`. A single-image render runs it once, in row bands sized to about 0.3 s of GPU time (progress ticks and the GPU watchdog);
   Live Preview keeps it and calls it per frame after updating the camera/seed/sample count.
4. `postProcessAndWrite()` tone-maps and writes the PNG/EXR. Outside the `--crop` window the picture is black.

## The 31-slot ceiling and how data fits

Metal allows 31 buffer arguments per kernel and `primaryRayKernel` uses all of them (0-30), plus 9 textures:

| buffers | |
|---|---|
| 0 acceleration structure, 1 `Uniforms`, 6 intersection function table | |
| 2 triangle materials, 3 vertices, 7 normals, 8 uvs | triangle mesh |
| 4 sphere materials, 5 spheres, 13/14 disks, 26/27 cylinders | analytic shapes |
| 10/11 instance normals/materials, 12 instance transforms | instancing |
| 9 area lights, 15 point, 16 directional, 17 projection, 18 goniometric | lights |
| 19/20 environment marginal/conditional CDF, 22/23 the same for the pbrt image light | environment sampling |
| 21 GGX energy table, 24/25 lens elements / exit-pupil bounds | tables, realistic camera |
| 28 cloud media, 29 RGB-grid media, 30 shared float data | media and a general float buffer |

Textures: 0 output, 1 earth map, 2 goniometric, 3 pbrt environment, 4 pbrt goniometric, 5 projection, 6 area-light image, 7 diffuse image, 8 transmit image.

There is no spare slot, so new per-feature data **rides in existing buffers**: the portal light's tables live in the pbrt-environment buffers, measured
BRDF tables, bump-map images and grid media in `rgbGridData` (slot 30), and per-material parameters in unused `TriangleMaterial` fields. Each such
convention is documented where it is set (`metal_poc_pbrt_materials.mm`, `metal_poc_pbrt_loader.mm`) - for example `materialType` 25 is a *family* of
textures selected by `conductorK.y`, and glass bounding a medium carries its sigma_t in `conductorEta`.

## The host/shader contracts

Two copies of every shared struct exist, by hand: `metal_poc_gpu_types.h` (C++) and `metal_poc_types.metal` (MSL).

* **Add fields at the END** of `Uniforms` and of any other mirrored struct, on both sides, with a safe zero-initialised default. Zero-initialised `Uniforms`
  (the shader tests) must mean "feature off".
* `metal_poc_shader_tests` runs `test_structLayouts`: the shader reports `sizeof`/`alignof` of 14 structs and the offsets of 17 `Uniforms` fields, and the
  host compares. A field on one side only fails it. When you add a mirrored struct, add it to both probe lists
  (`metal_poc_test_kernels.metal`, `metal_poc_shader_tests_layout.mm`).
* `materialType` values are the `METAL_MAT_*` macros in `metal_poc_material_ids.metal`. Never reuse a number. Metal's numbering is its own; OptiX has a
  separate enum.
* `Uniforms` is read as a `constant` buffer; the loader and `dsFillUniforms` fill it. `liveRender` writes only camera/seed/sample fields per frame.

## Recipes

**A new material.** (1) Add `METAL_MAT_FOO` to `metal_poc_material_ids.metal`. (2) Write `shadeFoo()` in the right `metal_poc_materials_*.metal` (take
the closest existing one as the template: it must do the light sampling, MIS and continuation like its neighbours). (3) Add the dispatch branch in
`primaryRayKernel` (the `materialType ==` chain). (4) Map the pbrt material in `mapPbrtMaterial()` (or the CPU/hand-built scene helper). (5) Add a
`test_*` kernel and a `metal_poc_shader_tests_*.mm` check for the pdf/f when it has closed forms. (6) Add or pick a pbrt scene that uses it, run the
parity sweep, and re-snapshot the golden if the change is intended.

**A new `Uniforms` field.** Append it in both structs, give it a default, fill it in `dsFillUniforms`, add it to the layout probe if it is near the end, run the
shader tests.

**A new pbrt feature.** Teach `src/shared/pbrt_flatten*.h` first (the CPU and OptiX use it too), then map it in the loader. If Metal cannot support it yet, make
the loader warn and fall back rather than fail (see the existing "binds ... to a texture, which is not supported" warnings).

**A shader change** needs no rebuild: run the test or `metal_poc` directly. After a shader change the first Live Preview/GUI start recompiles the shaders (about 15 s once).

## Testing

| what | command | protects |
|---|---|---|
| everything | `cd build_macos && METAL_PARITY_STRICT=1 ctest` (~2 min, see [parity doc](METAL_PARITY_STATUS.md) for the build) | the rows below |
| device-side shader tests | `metal_poc_shader_tests` | BSDF/pdf/sampling functions against closed forms; the struct layouts |
| math tests | `metal_poc_math_tests` | host math |
| smoke/validate/crop/seed/exit-code | `metal_poc_smoke_*`, `metal_poc_crop_window`, `metal_poc_seed_reproducible`, `metal_poc_write_failure_exit_code` | the CLI contract |
| live-preview dylib | `metal_realtime_dylib`, `metal_realtime_dylib_other_cwd` | the C ABI the GUI loads; scene lookup from `/` |
| CPU vs Metal sweep + golden snapshot | `metal_poc_cpu_gpu_parity` | brightness/regions against the CPU, and drift against Metal's own previous output |
| large third-party scenes | `scripts/metal_large_scenes_sweep.sh` | the H1-H21 scenes load and render (needs the downloads) |
| the real GUI, headless | `python3 scripts/gui_selftest.py <App.app> --live-preview` | the packaged app, Live Preview frames and picture change |

The GitHub macOS runner has no hardware ray tracing, so CI builds and runs what it can and skips the device tests; on a developer Mac run the strict ctest
before every PR that touches `gpu/metal/` or shared headers (and rebuild after `git pull` first - a stale binary has produced a false "CPU bug" before).
A change that must not alter pictures (a refactor) should be checked with a byte compare of rendered PNGs against a build of the previous `main`.

## Environment switches

| variable | effect |
|---|---|
| `METAL_BANDS=N` | force N equal row bands instead of time-targeted ones |
| `METAL_REGEN=0\|1` | force path regeneration off/on (the host picks per scene: on below ~1500 triangles) |
| `METAL_CENSUS=1` | print per-scene lane utilisation and distinct materials per group-bounce |
| `METAL_TG_W`, `METAL_TG_H`, `METAL_TG_PRINT` | threadgroup shape override / print |
| `METAL_PARITY_*` | the sweep: `STRICT`, `ALL`, `MODELS`, `ONLY_SCENE_ID`, `SEED`, `SPP`, `DEPTH`, `ADAPTIVE`, `GOLDEN`, `DUMP`, `KEEP`, `TIMING`, `WORKERS`, `SHARD` (see the parity doc) |
| `RAY_TRACER_USER_ASSETS` | the per-user folder the loader also searches for downloaded scene files |
| `RT_METAL_SHADER_DIR` (compile-time) | where the shader sources are found when running from a build tree; an installed app finds them next to the executable |

## Conventions worth knowing

* Roughness: the shader squares the roughness-style value it is given; the loader must store *perceptual* roughness, not alpha.
* There is a path-throughput ceiling (50 per channel, like the CPU and OptiX) and adaptive sampling is **off** unless asked for.
* Shadow rays are re-aimed from their offset origin (`reaimShadowRay`); clear/thin/rough glass blocks shadow rays like pbrt-v4; `Material "interface"` does not.
* Image skies put image row 0 straight up on all three backends (`envMapUV`).
* **`primaryRayKernel` is one ~1,600-line function on purpose, and splitting it costs speed.** Measured 2026-10-07 on an M2: moving only its first block (the 168-line camera-ray generation: filter sample, shutter time, orthographic/spherical/realistic/pinhole/orbit-blur rays, thin-lens depth of field) into an `inline` helper that takes the RNG and the outputs by `thread` reference made 8 of 10 test scenes 3-10% slower, reproducibly (the original code reproduced its own timings to within 1-2%), while one scene (realistic camera) got 18% faster; renders also changed at the bit level in some scenes (those same scenes already differ between a cold and a warm shader cache, so that part is not conclusive). The cost is the compiler's register allocation and inlining across the helper boundary in a megakernel with ~40 live path-state variables. So the host-side stages are split (`metal_poc_dispatch.mm`, `metal_poc_gpu_resources.mm`, the loader) but the kernel stays whole; if you try again, measure best-of-3 wall times on a camera-mode set and the material scenes, and expect to need a `PathState` struct for the whole loop rather than extracting pieces. `scripts/check_code_size.py` lists the kernel in its baseline for this reason.
* A change to the code *shape* of the kernel can make bit-stable scenes nondeterministic (it has happened); measure with the byte compare above.
* When Metal and the CPU disagree, prove who is right with an independent integral or an explicit-mesh copy of the shape before "fixing" either.
