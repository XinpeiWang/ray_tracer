# Build System Guide

This document describes how to build the entire Ray Tracer project.

## Quick Start

### Windows - One-Command Build & Deploy (Recommended)

The easiest way to build everything and create a ready-to-run package:

```powershell
# From Visual Studio Developer PowerShell:
.\build_and_deploy.ps1
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
build_all.bat
```

This builds everything in Release mode by default (without deployment).

### Windows - Advanced Build Options

For more control, use the PowerShell script:

```powershell
# Build everything (default: Release)
.\build_all.ps1

# Build in Debug mode
.\build_all.ps1 -Configuration Debug

# Skip tests
.\build_all.ps1 -SkipTests

# Skip Qt GUI
.\build_all.ps1 -SkipGui

# Build and deploy Qt GUI package with all dependencies
.\build_all.ps1 -Deploy

# Clean and rebuild
.\build_all.ps1 -Clean

# Convenience: One-step build + deploy
.\build_and_deploy.ps1
```

### Visual Studio IDE

1. Open `ray_tracer.sln` in Visual Studio
2. Set the startup project:
   - Right-click `launcher` in Solution Explorer
   - Select "Set as Startup Project"
3. Select configuration: `Release` or `Debug`
4. Select platform: `x64`
5. Build → Build Solution (Ctrl+Shift+B)

**Note:** Building from Visual Studio IDE does not automatically deploy Qt dependencies. Use `.\deploy_qt_gui.ps1` or `.\build_and_deploy.ps1` after IDE builds to create a complete package.
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

### Testing

4. **ray_tracer_tests** (`tests/ray_tracer_tests.vcxproj`)
   - Test executable: `ray_tracer_tests.exe`
   - Google Test framework
   - Unit and integration tests
   - Tests both CPU and GPU renderers

### GUI (External Qt Build)

5. **Qt GUI** (`qt_gui/RayTracerGUI.pro`)
   - Qt 6.11.1 application
   - MinGW 64-bit build
   - Spawns `ray_tracer.exe` as subprocess
   - Built separately with qmake/nmake

## Prerequisites

### Required

- **Visual Studio 2022 or 2026** with C++ development tools
- **CUDA Toolkit 12.x+** (for OptiX runtime)
  - Set environment variable: `CudaToolkitPath=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- **NVIDIA OptiX SDK 9.1+**
  - Set environment variable: `OptixSdkPath=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0`
- **NVIDIA GPU** with OptiX support (RTX series recommended)
- **NVIDIA Driver 595.79+** for Blackwell architecture (e.g., RTX 5080)

### Optional

- **Qt 6.11.1** with MinGW 64-bit (for GUI build)
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

After a successful build:

```
x64\Release\
  ├─ ray_tracer.exe              # Main launcher executable
  └─ optix_renderer.lib          # OptiX renderer static library

cpu_renderer\x64\Release\
  └─ cpu_renderer.lib            # CPU renderer static library

gpu\optix\
  └─ optix_programs.ptx          # OptiX shader (compiled from .cu)

tests\x64\Release\
  └─ ray_tracer_tests.exe        # Test suite

qt_gui\
  └─ (build directory varies based on Qt toolchain)

RayTracer_Package\              # Deployment package (after running deploy script)
  ├─ RayTracerGUI.exe           # Qt GUI application
  ├─ ray_tracer.exe             # Console launcher
  ├─ optix_programs.ptx         # GPU shader
  ├─ Qt6Core.dll                # Qt dependencies (auto-deployed)
  ├─ Qt6Gui.dll
  ├─ Qt6Widgets.dll
  ├─ Qt6Network.dll
  ├─ Qt6Svg.dll
  ├─ libgcc_s_seh-1.dll
  ├─ libstdc++-6.dll
  ├─ libwinpthread-1.dll
  └─ [Qt plugins in subdirectories]
```

## Deployment

### Automatic Deployment (Recommended)

The easiest way to create a complete, ready-to-run package:

```powershell
.\build_and_deploy.ps1
```

This automatically:
1. Builds all components
2. Copies executables and shaders to `RayTracer_Package\`
3. Runs `windeployqt` to include all Qt DLLs
4. Validates the package completeness

### Manual Deployment

If you've already built the project and want to deploy separately:

```powershell
# Deploy with auto-detection of Qt dependencies
.\deploy_qt_gui.ps1

# Deploy specific configuration
.\deploy_qt_gui.ps1 -Configuration Release
.\deploy_qt_gui.ps1 -Configuration Debug
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
1. Install CUDA Toolkit 12.x+
2. Set environment variable:
   ```powershell
   $env:CudaToolkitPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
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
1. Install Qt 6.11.1 with MinGW 64-bit
2. Add Qt bin directory to PATH:
   ```powershell
   $env:Path += ";C:\Qt\6.11.1\mingw_64\bin"
   ```
3. Or skip GUI: `.\build_all.ps1 -SkipGui`

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
build_all.bat Release clean

# PowerShell script
.\build_all.ps1 -Clean

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
   .\deploy_qt_gui.ps1
   .\RayTracer_Package\RayTracerGUI.exe
   ```

## Related Documentation

- [Copilot Instructions](.github/copilot-instructions.md) - Project architecture and guidelines
- [OptiX README](gpu/optix/README.md) - GPU renderer details
- [Qt GUI README](qt_gui/README.md) - GUI application guide
