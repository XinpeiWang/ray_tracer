# Fresh Build - Quick Reference

## ✅ Build Status

**Last successful build:** $(Get-Date)

### Built Components
- ✅ **Launcher** - `x64\Release\ray_tracer.exe` (400 KB)
- ✅ **CPU Renderer** - `x64\Release\cpu_renderer.lib`
- ✅ **OptiX Renderer** - `x64\Release\optix_renderer.lib`
- ✅ **OptiX PTX Shader** - `gpu\optix\optix_programs.ptx`
- ⚠️ **Tests** - Requires toolset fix (v143 → v144)

### Test Render
- Rendered 200×200 @ 100 spp
- Time: 350 ms
- GPU: NVIDIA GeForce RTX 5080
- Output: test_build.png (111 KB)

## 🔧 How to Build from Fresh

### 1. Set Environment Variables
```powershell
$env:CudaToolkitPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:OptixSdkPath = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
```

### 2. Clean Previous Build (optional)
```batch
msbuild ray_tracer.sln /t:Clean /p:Configuration=Release /p:Platform=x64
```

### 3. Build Everything
```batch
# From Visual Studio Developer Command Prompt
build_all.bat

# Or with MSBuild directly
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64 /m
```

### 4. Test the Build
```batch
# Show help
.\x64\Release\ray_tracer.exe --help

# Quick GPU render
.\x64\Release\ray_tracer.exe --gpu 200 100 10

# CPU render
.\x64\Release\ray_tracer.exe --cpu 200 50 10
```

## 📁 Output Locations

After building, files are located in:

```
x64\Release\
├── ray_tracer.exe          # Main launcher
├── cpu_renderer.lib        # CPU renderer library
├── cpu_renderer.pdb        # CPU debug symbols
├── optix_renderer.lib      # OptiX renderer library
└── optix_renderer.pdb      # OptiX debug symbols

gpu\optix\
└── optix_programs.ptx      # OptiX shader (compiled .cu)
```

## 🚀 Quick Commands

### Build Commands
```batch
# Release build (optimized)
build_all.bat Release

# Debug build (with symbols)
build_all.bat Debug

# Build only launcher
msbuild launcher\launcher.vcxproj /p:Configuration=Release /p:Platform=x64

# Build only CPU renderer
msbuild cpu_renderer\cpu_renderer.vcxproj /p:Configuration=Release /p:Platform=x64

# Build only OptiX renderer
msbuild optix_renderer\optix_renderer.vcxproj /p:Configuration=Release /p:Platform=x64
```

### Render Commands
```batch
# Interactive mode (default)
.\x64\Release\ray_tracer.exe

# GPU render with custom settings
.\x64\Release\ray_tracer.exe --gpu 800 500 20

# CPU render
.\x64\Release\ray_tracer.exe --cpu 400 100 10

# Specify output file
.\x64\Release\ray_tracer.exe --output myimage.ppm 600 500 20
```

## 🐛 Troubleshooting

### If Build Fails

**"MSBuild not found"**
- Run from VS Developer Command Prompt
- Or run: `"C:\Program Files\Microsoft Visual Studio\2026\Community\Common7\Tools\VsDevCmd.bat"`

**"OptiX SDK not found"**
- Verify installation: `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0`
- Set environment variable: `$env:OptixSdkPath = "..."`

**"CUDA Toolkit not found"**
- Verify installation: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`
- Set environment variable: `$env:CudaToolkitPath = "..."`

**"PTX file not found" at runtime**
- The launcher expects `gpu\optix\optix_programs.ptx` relative to its location
- Copy `gpu\optix\optix_programs.ptx` to the executable directory if needed

### If Render Fails

**"OptiX not available"**
- Update NVIDIA driver (595.79+ for RTX 5080)
- Verify GPU supports OptiX (RTX series recommended)
- Run with `--cpu` flag to use CPU renderer instead

**"Cannot write output file"**
- Check write permissions in output directory
- Specify alternate path with `--output` flag

## 📝 Notes

- Build time: ~30-60 seconds (first build), ~5-10 seconds (incremental)
- Parallel compilation is enabled (`/m` flag)
- Tests project needs toolset update to v144 (currently broken with v143)
- OptiX PTX is compiled automatically during build
- PNG output is automatic (converted from PPM)

## 📚 Next Steps

1. **Read the documentation:**
   - [BUILD.md](BUILD.md) - Detailed build guide
   - [README.md](README.md) - Project overview
   - [.github/copilot-instructions.md](.github/copilot-instructions.md) - Architecture

2. **Try different scenes:**
   - Scene 0: Cornell Box (default)
   - Scene 1: Bouncing Spheres
   - Scene 2: Random spheres

3. **Experiment with parameters:**
   - Width: 200-1920 (higher = more detail)
   - SPP: 10-10000 (higher = less noise, GPU needs 10-100x more than CPU)
   - Max depth: 5-50 (higher = more realistic lighting)

4. **Deploy Qt GUI:**
   ```batch
   # Build Qt GUI
   cd qt_gui
   qmake
   nmake

   # Deploy package
   cd ..
   .\deploy_qt_gui.ps1
   ```

---

**Happy rendering! 🎨**
