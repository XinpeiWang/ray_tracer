# What each renderer supports

One table for all four renderers: the CPU renderer, the two OptiX renderers on Windows (**recursive** is the default `--gpu`, **wavefront** is `--gpu --wavefront`), and the **Metal** renderer on macOS. Every renderer reads the same parsed scene (`pbrt_flatten::FlatScene`), so "supported" here is about what each one *does* with it.

This page is the overview. The detail lives in:

* [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) - what happens to each individual pbrt-v4 directive (Full / Approx / Fallback / Unsupported).
* [`FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md) - the CPU / OptiX survey, row by row (it has no Metal column; this page does).
* [`METAL_BACKEND.md`](METAL_BACKEND.md) and [`METAL_PARITY_STATUS.md`](METAL_PARITY_STATUS.md) - how Metal works, and how closely it matches the CPU.

`Y` supported, `approx` supported with a documented approximation, `-` not supported (the scene still loads; the loader warns and falls back), `CPU` means the request is moved to the CPU with a warning. A `†` marks a Metal entry read from the source and the Metal docs rather than from a test run - the Mac session should confirm it.

| Area | CPU | OptiX recursive | OptiX wavefront | Metal |
|---|:-:|:-:|:-:|:-:|
| **Shapes**: sphere, disk, cylinder, triangle meshes | Y | Y | Y | Y |
| bilinear patches, curves | Y (curves analytic) | Y (tessellated) | Y (tessellated) | Y (tessellated to triangles at load) |
| cones, paraboloids | Y | - | - | Y (tessellated to triangles at load) |
| instancing (`ObjectInstance`) | Y | Y | Y | Y (hardware instancing) |
| **Materials**: diffuse, conductor, dielectric (smooth, rough, thin), diffuse transmission, coated diffuse / coated conductor, hair, principled, mix | Y | Y | Y | Y |
| subsurface (tabulated BSSRDF) | Y | Y | Y | - † |
| measured (`.bsdf`) | Y | Y | Y | Y † |
| dispersion | Y | approx (3 wavelengths) | Y | - † |
| **Textures**: image, checker, marble, fbm, windy, wrinkled, dots, bilerp | Y | Y | Y | Y |
| **Lights**: point, spot, distant, goniometric, projection, area (all samplable shapes), infinite (constant and image), portal | Y | Y | Y | Y |
| **Media**: homogeneous (incl. per-channel), cloud, RGB grid | Y | Y | Y | Y (homogeneous, cloud, RGB grid †) |
| uniform grid, NanoVDB | Y | grid Y, NanoVDB approx | grid Y, NanoVDB approx | - † |
| camera medium | Y | Y | Y | Y |
| **Cameras**: perspective (+ depth of field), orthographic, spherical, realistic lens | Y | Y | Y | Y |
| motion blur (camera, spheres) | Y | Y | Y | Y † |
| motion blur (meshes, curves, patches, discs, cylinders) | Y | static | static | static † |
| **Integrators**: path, volumetric path | Y | Y | Y | Y |
| SPPM | Y | Y (own pipeline) | - | CPU |
| BDPT, MLT, random walk, AO, debug integrators | Y | CPU | CPU | CPU |
| **Sampling options**: `--sampler`, `--lightsampler`, `--regularize`, `--spectral` | Y | ignored (warned) | partly (always-on spectral pipeline) | ignored (warned) |
| adaptive sampling | Y | - | Live Preview only | Y (off unless asked) |
| crop window, `--seed` (reproducible), firefly clamp, exposure, tone map | Y | Y | Y | Y |
| **Denoising** (`--denoise`) | none | OptiX AI denoiser | OptiX AI denoiser | Intel Open Image Denoise (colour only) |
| **Render passes** (`--aovs`) | Y | Y (made on the CPU scene) | Y (same) | Y (same) |
| **Live Preview** | - | - | Y (ReSTIR, SVGF, probe cache, NRC, upscaling, ...) | Y (plain frames + a display-only smoothing of new pixels) |
| **Video** (`--video`) | Y | Y | Y | Y |

## How the GUI uses this

The Live Preview controls are shown by what the loaded library says it has (`realtime_backend_features()`, see `src/shared/realtime_api.h`), not by the platform. Options a renderer ignores are accepted and warned about on the command line, never an error.

## Keeping this page right

When a feature changes on any renderer, change its row here in the same commit. The Metal parity sweep (`gpu/metal/metal_cpu_gpu_parity_check.cpp`, see `METAL_PARITY_STATUS.md`) and the OptiX golden check (`scripts/gpu_golden.py`) are what catch a renderer drifting from the table.
