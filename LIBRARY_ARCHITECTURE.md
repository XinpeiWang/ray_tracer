# Ray Tracer Library Architecture

## Overview

The ray tracer has a **hybrid library architecture** where the CPU renderer is a proper static library project and the GPU CUDA renderer uses custom MSBuild targets for compilation.

## Project Structure

```
ray_tracer/
├── launcher/                    # Unified launcher (produces ray_tracer.exe)
│   ├── main.cpp                # Entry point, switches between CPU/GPU
│   └── launcher.vcxproj        # References cpu_renderer
│
├── cpu_renderer/               # CPU renderer static library ✅
│   ├── cpu_interface.h         # C API: cpu_render_main()
│   ├── cpu_interface.cpp       # Implementation
│   └── cpu_renderer.vcxproj    # Builds as StaticLibrary → cpu_renderer.lib
│
├── gpu/cuda/                   # GPU renderer (custom build) ⚙️
│   ├── cuda_interface.h        # C API: gpu_render_main()
│   ├── host.cu                 # CUDA implementation
│   ├── scene_serializer.cpp    # Scene conversion
│   └── (compiled via build_cuda.targets)
│
├── build_cuda.targets          # Custom MSBuild targets for CUDA compilation
├── Directory.Build.targets     # Imports build_cuda.targets automatically
│
└── src/TheRestOfYourLife/      # Shared rendering core
	├── cornell_box_scene.h     # Shared scene definition
	├── camera.h                # CPU path tracing engine
	├── material.h, sphere.h, quad.h, etc.
	└── ...
```

## Why Hybrid Architecture?

- **CPU Renderer**: Proper Visual Studio static library project with clean ProjectReference
- **GPU Renderer**: Uses custom MSBuild targets because Visual Studio 2026's CUDA BuildCustomizations have compatibility issues (MSB3693/MSB4064 errors with duplicate parameters)

## APIs

### CPU Renderer Library

**Header:** `cpu_renderer/cpu_interface.h`
**Function:**
```c
int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path);
```

- **Features:**
  - Path tracing with importance sampling (PDF-based)
  - Multithreaded rendering (auto-detects thread count)
  - Produces high-quality results at low sample counts (e.g., 10-100 spp)

### GPU Renderer Library

**Header:** `gpu/cuda/cuda_interface.h`  
**Function:**
```c
int gpu_render_main(int width, int height, int spp, int max_depth, const char* output_path);
```

- **Features:**
  - CUDA-accelerated naive path tracing
  - Requires higher sample counts (~1000 spp) for quality comparable to CPU at 10 spp
  - Uses shared Cornell box scene definition

## Building

1. **Prerequisites:**
   - Visual Studio 2022+ with C++20 support
   - CUDA Toolkit 12.x+ (for GPU renderer)
   - Use **Visual Studio Developer Command Prompt** (vcvars64.bat) for CUDA builds

2. **Build:**
   ```powershell
   msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64
   ```

3. **Output:**
   - Executable: `x64\Release\ray_tracer.exe`
   - CPU Library: `x64\Release\cpu_renderer.lib`
   - GPU Objects: `gpu\cuda\*.obj` (linked directly into exe)

## Usage

```bash
# GPU mode (default)
ray_tracer.exe --gpu [width] [spp] [max_depth]

# CPU mode
ray_tracer.exe --cpu [width] [spp] [max_depth]

# Examples
ray_tracer.exe --gpu 600 1000 50    # GPU: 600×600, 1000 spp
ray_tracer.exe --cpu 600 100 50     # CPU: 600×600, 100 spp
```

## Design Benefits

1. **In-Process Execution:** Both renderers called directly (no subprocess spawning)
2. **Shared Scene:** CPU and GPU use the same Cornell box definition
3. **Clean CPU Library:** cpu_renderer is a proper static library with ProjectReference
4. **Reliable CUDA Build:** Custom targets avoid Visual Studio CUDA compatibility issues
5. **Flexible Integration:** CPU library can be used by other applications

## Rendering Quality Guidance

- **CPU Path (Importance Sampling):**
  - Use 10-100 samples per pixel for good quality
  - Efficient convergence due to light importance sampling

- **GPU Path (Naive Path Tracing):**
  - Use 1000+ samples per pixel to match CPU quality
  - Faster per-sample but no importance sampling yet

## Scene Configuration

The shared Cornell box scene is defined in `src/TheRestOfYourLife/cornell_box_scene.h`:
- Standard Cornell box: 555×555×555 units
- Ray origin offset: 0.01f (prevents self-intersection)
- Camera: `lookfrom=(278,278,-800)`, `lookat=(278,278,0)`, `vfov=40°`
