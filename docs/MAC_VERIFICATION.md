# Mac verification checklist (2026-10-08)

The Windows session changed shared code whose macOS branches it could only syntax-check (`cl /Zs /DRT_HAVE_METAL launcher/main.cpp`). The Mac session ran these on 2026-10-08 (main at 2970f2a, an M2); results below. Delete this file when the open items are settled.

Build and automated checks

- [x] `cmake -S . -B build_macos -G Ninja -DCMAKE_BUILD_TYPE=Release -DRT_BUILD_METAL=ON && cmake --build build_macos` compiles, including `realtime_api.h`, `gpu_backend.h`, `backend_capabilities.h`, `realtime_renderer_mac.cpp` and `camera.h` with `oidn_runtime.h`.
- [x] `ctest` in `build_macos`: 12/12 (shader tests, the 95-scene parity sweep with its golden snapshot, seed reproducibility), and 13/13 with the new `metal_poc_sppm_gpu_falls_back`. `unit_tests` 3690 and `cpu_integrator_tests` 72 pass, built Release like CI and run from the repo root (with the launcher beside them, so the render tests run, none skipped).
  - A **Debug** (no build type) `cpu_integrator_tests` aborts on its first test: `scene_registry.h:328`, "a pbrt-backed scene was built before accelerator_override::set() ran". CI builds Release, where the assert is compiled out, so it never shows. Those test files need an `accelerator_override::set({"", ""})` before the first scene (the AOV tests already do it). Not fixed here.
- [x] `python3 scripts/gui_selftest.py RayTracer_Package_macOS/RayTracerGUI.app --live-preview`: all six modes pass.

Behaviour that changed on the Mac side

- [x] `ray_tracer --gpu --adaptive ...` prints no "has no effect"; it changes the picture (A1, 300 px, 1024 spp: 14.1 s without, 13.1 s with it, 6.6 s with `--adaptive-threshold 0.05`). The Adaptive sampling box is on for Metal.
- [x] Live Preview with the Metal library: Depth of Field, Exposure, Samples/Frame, Max Bounces, Firefly Clamp, Aperture, Focus Distance, Smooth noisy pixels, Auto exposure and **AI denoise** are shown; SVGF, ReSTIR and the rest are hidden (checked on a screenshot of the Render Options tab).
- [x] A single-image `--gpu` render with no Metal device says "ERROR: Metal is not available!" and exits 200 without a picture (tested by running the launcher under `sandbox-exec` with `(deny iokit-open)`).
- [x] `--cpu --denoise` with the denoiser installed: "Denoised with Open Image Denoise", and against a 512 spp reference the error of an 8 spp picture goes from 34.5 to 7.9 (mean abs diff, 0-255). In the GUI, with the CPU renderer selected the image/video AI denoiser box is enabled and ticking it without the library offers the download (declining unticks it).
- [x] **Found and fixed:** `--sppm --gpu` on a Mac stopped with "ERROR: OptiX is not available! (--sppm --gpu requires OptiX)" instead of rendering on the CPU like `--bdpt`, `--mlt` and the debug integrators. It now warns that `--gpu` is ignored and renders on the CPU (`launcher/main.cpp`, with a ctest).

For review

- [ ] `docs/GPU_SCENE_COMMON.md` **is not in the repository** (not on main, not on any branch or in the history the Mac checkout has), so there was nothing to review. Push it, or say where it is.
- [x] `docs/BACKEND_SUPPORT.md`, Metal column: every dagger entry was run and the table now says what was seen. Confirmed: subsurface `-`, dispersion `-` at the time (the shader existed but the pbrt loader never selected it; selected since, see CHANGELOG; B23 has no colour fan), uniform grid / NanoVDB `-` (the `interface` material is unsupported), measured `Y`, camera/sphere motion blur `Y`, mesh motion blur static, cones/paraboloids tessellated, RGB-grid medium `Y`. **Wrong in the old table:** the cloud medium - a pbrt scene's cloud is not rendered on Metal (E5 and `cloud-medium-scene.pbrt` show no cloud, a black disc in the second), the same before the kernel split; the parity sweep cannot see it (see `METAL_PARITY_STATUS.md`). **Fixed afterwards:** the loader now builds the cloud and the picture matches the CPU's (see `METAL_PARITY_STATUS.md`).
