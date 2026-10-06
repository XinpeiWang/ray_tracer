# Build System Guide

This document describes how to build the entire Ray Tracer project on
**Windows** - the full build, including the GPU/OptiX renderer (CUDA/OptiX is
Windows+NVIDIA only, with no macOS equivalent). For building the CPU
renderer, CLI, and Qt GUI on **macOS** instead, see the
[macOS (CPU-only)](README.md#macos-cpu-only) section of README.md - it's a
separate, purely-additive CMake/qmake path that this document's MSBuild
instructions don't cover.

## Quick Start

### Windows - One-Command Build & Deploy (Recommended)

The easiest way to build everything and create a ready-to-run package:

```powershell
# From Visual Studio Developer PowerShell:
.\scripts\build_and_deploy.ps1
```

This single command:
- ✅ Builds all C++ components (CPU renderer, OptiX GPU renderer, launcher)
- ✅ Builds Qt GUI application
- ✅ Deploys everything to `RayTracer_Package\` with all dependencies
- ✅ Automatically runs `windeployqt` to include Qt DLLs
- ✅ Validates the package is complete and ready to run

**The output package includes:**
- `RayTracerGUI.exe` - Main Qt GUI application
- `ray_tracer.exe` - Console renderer backend
- `optix_programs.ptx` - GPU shader
- All Qt6 DLLs (Core, Gui, Widgets, Network, Svg)
- Qt plugins (platforms, styles, imageformats, etc.)

**Launch the GUI:**
```powershell
.\RayTracer_Package\RayTracerGUI.exe
```

### Windows - Traditional Build

Open a **Visual Studio Developer Command Prompt** or **Developer PowerShell** and run:

```batch
scripts\build_all.bat
```

This builds everything in Release mode by default (without deployment).

### Windows - Advanced Build Options

For more control, use the PowerShell script:

```powershell
# Build everything (default: Release)
.\scripts\build_all.ps1

# Build in Debug mode
.\scripts\build_all.ps1 -Configuration Debug

# Skip tests
.\scripts\build_all.ps1 -SkipTests

# Skip Qt GUI
.\scripts\build_all.ps1 -SkipGui

# Build and deploy Qt GUI package with all dependencies
.\scripts\build_all.ps1 -Deploy

# Clean and rebuild
.\scripts\build_all.ps1 -Clean

# Convenience: One-step build + deploy
.\scripts\build_and_deploy.ps1
```

### Visual Studio IDE

1. Open `ray_tracer.sln` in Visual Studio
2. Set the startup project:
   - Right-click `launcher` in Solution Explorer
   - Select "Set as Startup Project"
3. Select configuration: `Release` or `Debug`
4. Select platform: `x64`
5. Build → Build Solution (Ctrl+Shift+B)

**Automatic Deployment:** The `launcher` and `optix_renderer` projects now include post-build events that automatically copy:
- `ray_tracer.exe` → `RayTracer_Package/`
- `optix_programs.ptx` → `RayTracer_Package/`

**Qt GUI:** To complete the package, build the Qt GUI separately (see Qt GUI Build section) and run `.\scripts\deploy_qt_gui.ps1` to add Qt dependencies.

### Fast incremental builds and a quick test loop (Windows)

How long a rebuild takes depends on what you touched. Measured on this repo (Release, `/m:4`):

| You changed | Rebuild |
|---|---|
| nothing | ~2 s |
| one test `.cpp` | ~4 s |
| `src/TheRestOfYourLife/camera.h` (18 sources include it) | ~40 s |
| one wavefront kernel `.cu` | ~50 s (the biggest, `wavefront_kernels_materials.cu`; the others take 2-20 s) |
| a host-only header (e.g. `src/shared/pbrt_flatten.h`) | C++ dependents only, **no** CUDA recompile |
| a header the device code includes (e.g. `src/shared/volume_scattering.h`) | all CUDA files that include it (~2 min, the recursive `optix_programs.cu` is the longest) plus the C++ dependents |

Things that make this work, and what to keep in mind:

- **Change tracking.** `Directory.Build.props`/`Directory.Build.targets` use MSBuild's one-`cl`-per-file mode (`UseMultiToolTask`) instead of `/MP`. With `/MP` the tracking logs lost each project's first source, so every build recompiled the entire solution (~95 s with nothing changed). A project that mixes C and C++ files needs one command line for all of them (`miniz.c` is compiled as C++ for this reason), or the two compile tasks overwrite each other's tracking entries again.
- **CUDA.** `build_optix.targets` hands all 16 `.cu` files to `scripts/cuda_build.ps1`, which runs the compiles in parallel (`RT_CUDA_JOBS`, default 6) and rebuilds a file only when its source, a header it actually included (nvcc's own `-MD` dependency files, kept in `gpu/optix/.cudadeps/`) or its flags changed. `nvcc -t0` compiles the two target architectures concurrently.
- **PTX compile and the OptiX cache.** The three OptiX PTX modules compile with `--split-compile 0` (nvcc uses all cores; `optix_programs.ptx` 120 s -> 77 s, same output). A *cold* OptiX cache is a separate cost: OptiX compiles the PTX to machine code on first use (`%LOCALAPPDATA%\NVIDIA\OptixCache`), which took ~5 min for the recursive module on one core. `gpu/optix/optix_module_parallel.h` creates the modules with `optixModuleCreateWithTasks` on every core instead (cold ~2 min); a warm cache is unaffected. After editing device code the first GPU run or test pays that once. Making `shade_material` `__noinline__` was tried and rejected: PTX 45 MB -> 15 MB and nvcc 23 s, but the cold JIT got slower (531 s vs 307 s).
- **Development-only speed-up.** `msbuild /p:RtCudaLocalArchOnly=true` (or `RT_CUDA_LOCAL_ARCH_ONLY=1`) compiles the plain CUDA kernels only for the GPU in this machine. The result runs only on that GPU, so do not package it; the default builds sm_86 and sm_120.
- **Running one scene's parity test.** `MaterialCpuGpuParityTest` renders only the scenes your `--gtest_filter` selects (one scene ~4 s; the full sweep ~140 s is unchanged):

```powershell
.\bin\Release\ray_tracer_tests.exe --gtest_filter="*BrightnessAndChannelsConsistentAcrossBackends/Scene21_*"
```

- **Running the whole suite.** `scripts\run_tests_parallel.ps1` splits it by what the tests touch. `-Tier Fast` (about 80 s, 3930 tests) runs everything that does not use the GPU in 16 shards, the loop to use while editing. `-Tier Split` runs Fast and then Slow (the 507 GPU tests, one process) for the whole suite in about 245 s, against about 370 s for one `ray_tracer_tests.exe` process. Slow is deliberately not sharded: separate processes time-slice the GPU, and 2-4 shards measured slower (193-198 s) than one (170 s). Nearly all of Slow is the material parity sweep, where the wavefront backend takes about 125 s because a 60x60 frame is thousands of tiny, synchronised launches; only a change inside that renderer would shrink it. The runner counts failing tests itself (`--gtest_brief` hides gtest's own failure summary) and removes its scratch directory when everything passes.
- **Before a big change to rendering code: take a baseline.** The test suite reliably catches a rendering error of about 3-4% and can miss 1% (measured by injecting bugs). `python scripts\render_baseline.py capture before-refactor` renders 15 scenes on the CPU and both GPU backends (about 5 minutes) and stores every pixel under `baselines\` (git-ignored); after the change and a rebuild, `python scripts\render_baseline.py compare before-refactor` renders them again with the same seeds and reports which scene/backend pairs moved, exiting 1 if any did. It compares pixel by pixel, so noise is measured instead of assumed: a 1% loss in wavefront light sampling showed as a -0.3% to -1.0% shift in 12 of 15 scenes with a noise of 0.00-0.04%, and unchanged code reproduces 43 of 45 pairs bit for bit. For a change that must not alter any image (a pure refactor) add `--expect-identical`: it also fails any pair whose pixels differ by more than 0.01% RMS, which catches changes that move no mean, such as a 2% change to the roughness mapping (4 CPU scenes, 0.3-0.75% RMS); the last-bit nondeterminism of the wavefront backend stays under the limit. It answers "did my change move the image", not "is the image right"; a deliberate fix will show as moved. A baseline is only valid for the code it was captured from, so capture a new one after merging. Details are in the script's header; `--scenes`, `--backends` and `--tolerance` narrow or tighten a run.
- **If a build ever recompiles everything with no changes**, diagnose it instead of living with it: `msbuild ray_tracer.sln ... /v:diag` and search the log for `command line has changed` / `No output for` - those say which project's tracking is inconsistent. Also check for stale `cl.exe`/`Tracker.exe` processes left by a killed build (`Get-Process cl, Tracker`).

## Project Structure

The solution contains the following projects:

### Core Projects

1. **launcher** (`launcher/launcher.vcxproj`)
   - Main executable: `ray_tracer.exe`
   - Unified entry point for CPU and GPU rendering
   - Parses command-line arguments
   - Links CPU and OptiX renderers

2. **cpu_renderer** (`cpu_renderer/cpu_renderer.vcxproj`)
   - Static library: `cpu_renderer.lib`
   - Multithreaded CPU path tracer
   - Importance sampling with PDFs
   - Cornell box scene support

3. **optix_renderer** (`optix_renderer/optix_renderer.vcxproj`)
   - Static library: `optix_renderer.lib`
   - OptiX 9.1 GPU path tracer
   - PTX shader compilation: `optix_programs.ptx`
   - Requires NVIDIA GPU with OptiX support

4. **scene_metadata** (`scene_metadata/scene_metadata.vcxproj`)
   - Dynamic library: `scene_metadata.dll`
   - Exposes the C++ scene registry's live scene list/metadata to the Qt
     GUI (which shells out to the CLI rather than linking `cpu_renderer.lib`
     directly) - must be rebuilt whenever `scene_registry.h`/
     `scene_registry_data.h` changes, or the GUI will show a stale scene
     list

5. **realtime_renderer** (`realtime_renderer/realtime_renderer.vcxproj`)
   - Dynamic library: `realtime_renderer.dll`
   - Backs the GUI's GPU-only "Live Preview" tab (adaptive-sampling pixel
     skip, Neural Radiance Cache, Neural Temporal Upscale, DOF override) -
     `Full`-tier only, see `releases/README.md`'s packaging tiers

### Testing

6. **ray_tracer_tests** (`tests/ray_tracer_tests.vcxproj`)
   - Test executable: `ray_tracer_tests.exe`
   - Google Test framework
   - Unit and integration tests
   - Tests both CPU and GPU renderers

### GUI (External Qt Build)

7. **Qt GUI** (`qt_gui/RayTracerGUI.pro`)
   - Qt 6.11.1 application
   - MSVC 2022 64-bit build
   - Spawns `ray_tracer.exe` as subprocess
   - Built separately with qmake/nmake

## Prerequisites

### Required

- **Visual Studio 2022 or 2026** with C++ development tools
- **CUDA Toolkit 13.2+** (for OptiX runtime)
- **NVIDIA OptiX SDK 9.1+**
- **NVIDIA GPU** with OptiX support (RTX series recommended)
- **NVIDIA Driver 595.79+** for Blackwell architecture (e.g., RTX 5080)

`build_optix.targets` auto-detects both SDKs — it checks the standard
`CUDA_PATH`/`OPTIX_SDK_PATH` environment variables the installers set, then
falls back to their default install locations. You only need to set
`CudaToolkitPath` / `OptixSdkPath` yourself (as MSBuild properties, e.g.
`msbuild ... /p:CudaToolkitPath=...`) if you have multiple CUDA versions
installed or a non-standard install path:
```powershell
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64 `
  /p:CudaToolkitPath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" `
  /p:OptixSdkPath="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
```

### Optional

- **Qt 6.11.1** with MSVC 2022 64-bit component (for GUI build)
- **Google Test** (automatically included via FetchContent in tests)

## Build Configurations

### Release (Recommended)
- Optimizations enabled
- No debug symbols (fast execution)
- Output: `x64\Release\` or `launcher\x64\Release\`

### Debug
- Optimizations disabled
- Full debug symbols
- Output: `x64\Debug\` or `launcher\x64\Debug\`

## Build Outputs

After a successful build, the following files are automatically deployed:

**Build Artifacts:**
```
launcher\x64\Release\
  └─ ray_tracer.exe              # Main launcher (PRIMARY BUILD OUTPUT)

cpu_renderer\x64\Release\
  └─ cpu_renderer.lib            # CPU renderer static library

optix_renderer\x64\Release\
  └─ optix_renderer.lib          # OptiX renderer static library

gpu\optix\
  └─ optix_programs.ptx          # OptiX shader (compiled from .cu)

tests\x64\Release\
  └─ ray_tracer_tests.exe        # Test suite
```

**Auto-Deployed Package (via MSBuild post-build events):**
```
RayTracer_Package\              # Canonical deployment directory
  ├─ ray_tracer.exe             # Auto-copied from launcher build
  ├─ optix_programs.ptx         # Auto-copied from optix_renderer build
  ├─ RayTracerGUI.exe           # Added by Qt build + deploy_qt_gui.ps1
  ├─ Qt6Core.dll                # Qt dependencies (via deploy_qt_gui.ps1)
  ├─ Qt6Gui.dll
  ├─ Qt6Widgets.dll
  ├─ Qt6Network.dll
  ├─ Qt6Svg.dll
  ├─ libgcc_s_seh-1.dll
  ├─ libstdc++-6.dll
  ├─ libwinpthread-1.dll
  └─ [Qt plugins in subdirectories]
```

**Note:** Backend executables and shaders are now automatically copied to `RayTracer_Package/` by MSBuild post-build events. You only need to run `scripts\deploy_qt_gui.ps1` to add Qt dependencies after building the GUI.

## Deployment

### Automatic Deployment (Recommended)

The easiest way to create a complete, ready-to-run package:

```powershell
.\build_and_deploy.ps1
```

This automatically:
1. Builds all C++ components (launcher auto-deploys `ray_tracer.exe` and `optix_programs.ptx` to `RayTracer_Package/`)
2. Builds Qt GUI application
3. Copies `RayTracerGUI.exe` to `RayTracer_Package/`
4. Runs `windeployqt` to include all Qt DLLs
5. Validates the package completeness

### Build System Flow

**MSBuild Post-Build Events (Automatic):**
- `launcher` project → deploys `ray_tracer.exe` to `RayTracer_Package/`
- `optix_renderer` project → deploys `optix_programs.ptx` to `RayTracer_Package/`

**Manual Qt Deployment (after Qt build):**
```powershell
.\scripts\deploy_qt_gui.ps1
```

This script:
- Verifies backend files are present (auto-deployed by MSBuild)
- Copies `RayTracerGUI.exe` to `RayTracer_Package/`
- Runs `windeployqt` to add Qt dependencies

### Manual Deployment

If you've already built the project and want to deploy separately:

```powershell
# Deploy with auto-detection of Qt dependencies
.\scripts\deploy_qt_gui.ps1

# Deploy specific configuration
.\scripts\deploy_qt_gui.ps1 -Configuration Release
.\scripts\deploy_qt_gui.ps1 -Configuration Debug
```

The deployment script:
- ✅ Finds launcher executable automatically (checks multiple paths)
- ✅ Copies PTX shader
- ✅ Auto-detects and runs `windeployqt.exe`
- ✅ Validates all critical dependencies are present

### What Gets Deployed

The `RayTracer_Package\` directory becomes a self-contained package with:
- Main Qt GUI application
- Console renderer backend
- OptiX GPU shader
- All Qt6 runtime DLLs
- Qt plugins (platform integration, styles, image formats)
- MinGW runtime libraries

**This package can be zipped and distributed without requiring Qt installation on target machines.**

qt_gui\release\
  └─ RayTracerGUI.exe            # Qt GUI (if built)

RayTracer_Package\               # Deployment package (if deployed)
  ├─ RayTracerGUI.exe
  ├─ ray_tracer.exe
  ├─ optix_programs.ptx
  └─ (Qt DLLs)
```

## Common Issues

### MSBuild Not Found
**Symptom:** `'msbuild' is not recognized...`

**Solution:** Run from a Visual Studio Developer Command Prompt or Developer PowerShell:
- Start Menu → Visual Studio 2026 → Developer Command Prompt
- Or run `vcvars64.bat` from your VS installation

### OptiX Build Fails
**Symptom:** `fatal error: optix.h: No such file or directory`

**Solution:** 
1. Install NVIDIA OptiX SDK 9.1+
2. Set environment variable:
   ```powershell
   $env:OptixSdkPath = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
   ```

### CUDA Compilation Errors
**Symptom:** `nvcc.exe not found` or CUDA errors

**Solution:**
1. Install CUDA Toolkit 13.2+
2. Set environment variable:
   ```powershell
   $env:CudaToolkitPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
   ```
3. Run from VS Developer Command Prompt (nvcc requires vcvars)

### Link Errors: Unresolved External Symbols
**Symptom:** `LNK2001: unresolved external symbol cudaMemcpy`

**Solution:** The launcher project needs CUDA libraries. This should be automatic, but verify:
- `launcher.vcxproj` includes `cudart_static.lib` and `cuda.lib`
- `CudaToolkitPath` is set correctly

### Qt GUI Build Fails
**Symptom:** `qmake not found`

**Solution:**
1. Install Qt 6.11.1 with the MSVC 2022 64-bit component (via the Qt
   Maintenance Tool)
2. Add Qt bin directory to PATH:
   ```powershell
   $env:Path += ";C:\Qt\6.11.1\msvc2022_64\bin"
   ```
3. Make sure you're in a Visual Studio Developer Command Prompt/PowerShell
   (needed for `cl.exe`/`nmake.exe`, same requirement as `msbuild` above)
4. Or skip GUI: `.\scripts\build_all.ps1 -SkipGui`

## Advanced Topics

### Parallel Builds
MSBuild uses parallel compilation by default (`/m` flag).
To control thread count:
```batch
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64 /m:4
```

### Custom Build Properties
Pass MSBuild properties:
```batch
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64 /p:CudaToolkitPath="C:\Custom\CUDA"
```

### Clean Build
```batch
# Batch script
.\scripts\build_all.bat Release clean

# PowerShell script
.\scripts\build_all.ps1 -Clean

# Direct MSBuild
msbuild ray_tracer.sln /t:Clean /p:Configuration=Release /p:Platform=x64
```

### Build Individual Projects
```batch
# Build only launcher
msbuild launcher\launcher.vcxproj /p:Configuration=Release /p:Platform=x64

# Build only tests
msbuild tests\ray_tracer_tests.vcxproj /p:Configuration=Release /p:Platform=x64
```

## Next Steps

After building:

1. **Run the launcher:**
   ```batch
   .\launcher\x64\Release\ray_tracer.exe --help
   ```

2. **Run tests:**
   ```batch
   .\tests\x64\Release\ray_tracer_tests.exe
   ```

3. **Run Qt GUI:**
   ```batch
   .\qt_gui\release\RayTracerGUI.exe
   ```

4. **Deploy Qt package:**
   ```powershell
   .\scripts\deploy_qt_gui.ps1
   .\RayTracer_Package\RayTracerGUI.exe
   ```

## Related Documentation

- [Copilot Instructions](.github/copilot-instructions.md) - Project architecture and guidelines
- [OptiX README](gpu/optix/README.md) - GPU renderer details
- [Qt GUI Documentation](qt_gui/QT_GUI_DOCUMENTATION.md) - GUI application guide
