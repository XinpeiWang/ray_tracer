# Copilot Instructions

## Project Guidelines
- User asked to save current workspace work to memory: project 'ray_tracer' changes include multithreaded renderer (src/TheRestOfYourLife/camera.h), thread-local RNG (src/TheRestOfYourLife/rtweekend.h), OneDrive/Desktop output path changes for image output, automatic thread-count detection (RAY_TRACER_THREADS env var + Windows idle sampling), modified launcher (main.cpp) to prefer CUDA prototype, and CUDA prototype scaffold added under gpu/cuda (host.cu, README.md, cuda_renderer.vcxproj).
- The launcher project contains the unified entry point at launcher/main.cpp that switches between CPU and GPU rendering modes. GPU is the default mode (use_gpu = true). The output executable is ray_tracer.exe which should be set as the Visual Studio startup project. Set launcher as the startup project by right-clicking it in Solution Explorer and selecting "Set as Startup Project". The launcher project should appear in bold when correctly configured.
- The CPU renderer is now a static library project named `cpu_renderer` (formerly `raytracing_book`). It builds `cpu_renderer.lib` and is linked into the launcher via a ProjectReference. The CPU renderer is called in-process through the `cpu_render_main()` C API defined in `cpu_renderer/cpu_interface.h`.
- Build environment: CUDA compilation requires Visual Studio Developer Command Prompt (vcvars64.bat) on Windows. Regular PowerShell causes nvcc/cudafe++ access violations.

## Rendering Guidelines
- The shared Cornell box scene is defined in src/TheRestOfYourLife/cornell_box_scene.h and used by both CPU (main.cc) and GPU (scene_serializer.cpp) paths.
- The GPU path tracer uses naive path tracing while the CPU uses importance sampling (PDF-based). The GPU requires approximately 100x more samples (e.g., 1000 spp) to match CPU quality at low samples (e.g., 10 spp).
- Set the ray-origin offset to 0.01f for 555-unit Cornell box scenes to prevent self-intersection.
- Use a minimum t-value of 0.001f for sphere intersections instead of 0.0f.