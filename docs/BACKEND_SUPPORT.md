# What each renderer supports

One table for all four renderers: the CPU renderer, the two OptiX renderers on Windows (**recursive** is the default `--gpu`, **wavefront** is `--gpu --wavefront`), and the **Metal** renderer on macOS. Every renderer reads the same parsed scene (`pbrt_flatten::FlatScene`), so "supported" here is about what each one *does* with it.

This page is the overview. The detail lives in:

* [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) - what happens to each individual pbrt-v4 directive (Full / Approx / Fallback / Unsupported).
* [`FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md) - the CPU / OptiX survey, row by row (it has no Metal column; this page does).
* [`METAL_BACKEND.md`](METAL_BACKEND.md) and [`METAL_PARITY_STATUS.md`](METAL_PARITY_STATUS.md) - how Metal works, and how closely it matches the CPU.

`Y` supported, `approx` supported with a documented approximation, `-` not supported (the scene still loads; the loader warns and falls back), `CPU` means the request is moved to the CPU with a warning. A `†` marks a Metal entry read from the source and the Metal docs rather than from a test run; the Mac session ran each of them on 2026-10-08 (the notes below say what was seen), so none is marked any more.

| Area | CPU | OptiX recursive | OptiX wavefront | Metal |
|---|:-:|:-:|:-:|:-:|
| **Shapes**: sphere, disk, cylinder, triangle meshes | Y | Y | Y | Y |
| bilinear patches, curves | Y (curves analytic) | Y (tessellated) | Y (tessellated) | Y (tessellated to triangles at load) |
| cones, paraboloids | Y | approx (tessellated) | approx (tessellated) | Y (tessellated to triangles at load) |
| instancing (`ObjectInstance`) | Y | Y | Y | Y (hardware instancing) |
| **Materials**: diffuse, conductor, dielectric (smooth, rough, thin), diffuse transmission, coated diffuse / coated conductor, hair, principled, mix | Y | Y | Y | Y |
| subsurface (tabulated BSSRDF) | Y | Y | Y | Y (pbrt-v4's SubsurfaceMaterial: dielectric boundary + probe-walk BSSRDF, the table from the CPU's own code; subsurface-ball and layered-materials match the CPU within 1-2% where an area light lights it. Under a constant sky only, Metal comes out 15-20% darker than the CPU inside the ball (the CPU samples that light by NEE through the ball, Metal does not sample a constant sky). Triangle meshes and spheres; not instanced meshes) |
| measured (`.bsdf`) | Y | Y | Y | Y (measured-brdf-showroom: means within 1% of the CPU) |
| dispersion | Y | approx (3 wavelengths) | Y | approx (3 wavelengths, like OptiX recursive: B23's colour fan matches the CPU's `--spectral` render; each path picks one of three channels at its first dispersive hit) |
| alpha cutout (a Shape's `"alpha"` / `"texture alpha"` mask, pbrt-v4's stochastic test) | Y | Y | Y | Y for ordinary (non-instanced) triangle meshes: an intersection function tests the mask (alpha-cutout: a square with a round hole and notches matches the CPU's picture; means within 3%). Not for an instanced mesh, and not for a mix's sub-material |
| **Textures**: image, checker, marble, fbm, windy, wrinkled, dots, bilerp | Y | Y | Y | Y |
| **Lights**: point, spot, distant, goniometric, projection, area (all samplable shapes), infinite (constant and image), portal | Y | Y | Y | Y |
| **Media**: homogeneous (incl. per-channel), cloud, RGB grid | Y | Y | Y | homogeneous and RGB grid Y (rgbgrid-medium matches the CPU; inside a medium Metal samples area, point/spot and distant lights, but not goniometric or projection lights); cloud Y (E2 cloud-medium-scene and E5 cloud-medium: the body matches the CPU to about 2%; the density, including the wispiness warp, is checked against the CPU's by a shader test; OptiX's density still has no wispiness warp. E5 agrees with the CPU to 0.5% over the whole cloud, and a white-furnace test (`cloud-furnace.pbrt`: a non-absorbing cloud under a uniform sky must be invisible) passes on Metal) |
| uniform grid, NanoVDB | Y | grid Y, NanoVDB Y (the density; a blackbody `temperaturename` glow is dropped with a warning) | grid Y, NanoVDB Y (the same) | grid Y, NanoVDB Y (read on the host by the shared `nanovdb_dense.h`, drawn by the RGB grid medium; E9 matches the CPU to 0.2% in brightness (worst 4x4 block 2.5-2.9%); the two absolute checks the CPU and OptiX pass, `nanovdb-furnace` (1.000 within 1.5%) and `nanovdb-thin-slab` (closed form 7.46e-4, Metal reads +0.7%), pass: ctest `metal_poc_nanovdb`; a `temperaturename` glow is dropped with a warning; grids over 64M voxels are not uploaded) |
| camera medium | Y | Y | Y | Y |
| **Cameras**: perspective (+ depth of field), orthographic, spherical, realistic lens | Y | Y | Y | Y |
| motion blur (camera, spheres) | Y | Y | Y | Y (cornell-camera-motion-blur, disk-cylinder-motion-blur: means within 1-4%) |
| motion blur (meshes, curves, patches, discs, cylinders) | Y | static | static | static (object-motion-blur: the moving object blurs less than on the CPU) |
| **Integrators**: path, volumetric path | Y | Y | Y | Y |
| SPPM | Y | Y (own pipeline) | - | CPU (`--sppm --gpu` warns and renders on the CPU; it used to stop with "OptiX is not available!") |
| BDPT, MLT, random walk, AO, debug integrators | Y | CPU | CPU | CPU |
| **Sampling options**: `--sampler`, `--lightsampler`, `--regularize`, `--spectral` | Y | ignored (warned) | partly (always-on spectral pipeline) | ignored (warned) |
| adaptive sampling | Y | - | Live Preview only | Y (off unless asked) |
| crop window, `--seed` (reproducible), firefly clamp, exposure, tone map | Y | Y | Y | Y |
| **Denoising** (`--denoise`) | Intel Open Image Denoise (default path tracer; GUI installs it on request; `--denoise-guides` adds surface colour and normal) | OptiX AI denoiser | OptiX AI denoiser | Intel Open Image Denoise (colour only) |
| **Render passes** (`--aovs`) | Y | Y (made on the CPU scene) | Y (same) | Y (same) |
| **Live Preview** | - | - | Y (ReSTIR, SVGF, probe cache, NRC, upscaling, ...) | Y (plain frames, more samples per frame once still, and two display-only aids: smoothing of new pixels, or AI denoise with Open Image Denoise) |
| **Video** (`--video`) | Y | Y | Y | Y |

## How the GUI uses this

The Live Preview controls are shown by what the loaded library says it has (`realtime_backend_features()`, see `src/shared/realtime_api.h`), not by the platform. Which options each renderer reads is one table in `src/shared/backend_capabilities.h`: the Render Options controls a renderer ignores are greyed out from it, and the launcher warns ("`--sampler` has no effect under the Metal renderer - ignoring.") from the same table. An option a renderer ignores is accepted and warned about, never an error.

## Checking that the renderers agree

The same method on every GPU backend: render each scene small, with the same seed, on the CPU and on the GPU, and compare the linear pictures (whole-image and per-channel means within 30%, a 6x6 grid of blocks within 50%, no NaN or Inf).

| Backend | Command | Known gaps |
|---|---|---|
| OptiX recursive and wavefront (Windows) | `python scripts/backend_parity.py` (103 scenes x 2 backends, about 6 minutes) | `scripts/backend_parity_known_gaps.txt` |
| Metal (Mac) | `ctest` runs `gpu/metal/metal_cpu_gpu_parity_check.cpp`, plus a committed snapshot of Metal's own earlier output | `kKnownGapScenes` (empty) |

To see whether a change moved a picture at all (not whether the picture is right), `python scripts/render_baseline.py capture <name>` before the change and `compare <name>` after: seeded renders are deterministic, so a pure refactor must come out bit-identical.

Neither check runs in CI (a hosted runner has no GPU or no hardware ray tracing), so they protect a developer machine.

## Keeping this page right

When a feature changes on any renderer, change its row here in the same commit; the sweeps above are what catch a renderer drifting from the table.
