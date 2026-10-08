# Mac verification checklist (2026-10-08)

The Windows session changed shared code whose macOS branches it could only syntax-check (`cl /Zs /DRT_HAVE_METAL launcher/main.cpp`). Please run these on a Mac and tick them off (delete this file when done).

Build and automated checks

- [ ] `cmake -S . -B build_macos -G Ninja -DCMAKE_BUILD_TYPE=Release -DRT_BUILD_METAL=ON && cmake --build build_macos` compiles (new/changed: `src/shared/realtime_api.h`, `launcher/gpu_backend.h`, `src/shared/backend_capabilities.h`, `realtime_renderer/realtime_renderer_mac.cpp`, `src/TheRestOfYourLife/camera.h` now includes `oidn_runtime.h`).
- [ ] `ctest` (unit tests, CPU integrator tests, the Metal parity sweep with its golden snapshot) passes.
- [ ] `python3 scripts/gui_selftest.py RayTracer_Package_macOS/RayTracerGUI.app` passes.

Behaviour that changed on the Mac side

- [ ] `ray_tracer --gpu --adaptive ...` no longer prints "has no effect"; the Adaptive sampling box is enabled when the GPU renderer is Metal (`backend_capabilities.h` says Metal reads it).
- [ ] Live Preview: with the Metal library the optional-feature controls (SVGF, ReSTIR, ...) are hidden as before, and the **AI denoise** box (row 15) is visible. The hiding now comes from `realtime_backend_features()`, not `#ifdef Q_OS_MAC`.
- [ ] A single-image `--gpu` render on a Mac without a usable Metal device now says "ERROR: Metal is not available!" (it used to go straight into `metal_render_main`).
- [ ] `--cpu --denoise` works on a Mac with the denoiser installed (the CPU renderer now denoises with Open Image Denoise; the GUI offers the download when the CPU renderer is selected).

For review

- [ ] `docs/GPU_SCENE_COMMON.md`: the proposal for one scene-to-GPU conversion shared by OptiX and Metal. Does the Mac side agree with the names and the stage order?
- [ ] `docs/BACKEND_SUPPORT.md`: the Metal column; entries marked with a dagger were read from source, not run.
