# CUDA Ray Tracer

A GPU-accelerated path tracing renderer implementation using CUDA. This folder contains a complete GPU ray tracer with multiple rendering kernels for progressive development and testing.

## 🎯 Current Status

**✅ Fully Functional** - Complete path tracing implementation with:
- Multi-bounce ray tracing (configurable max depth)
- Multiple material types: Lambertian, Metal, Dielectric, Emissive
- Importance sampling (cosine-weighted hemisphere)
- Fresnel reflections using Schlick approximation
- Gamma correction and anti-aliasing
- Device-side random number generation

**Performance**: 800×450 @ 4 samples/pixel renders in **~10ms** on RTX 5080 (kernel only)

## 📋 Prerequisites

- **NVIDIA GPU** with CUDA compute capability 7.5+ (RTX series or newer)
- **CUDA Toolkit 13.2+** installed (https://developer.nvidia.com/cuda-downloads)
- **Visual Studio 2022 or 2026** with C++ development tools
- Windows 10/11 with updated NVIDIA drivers

## 🔧 Build Instructions

### ⚠️ Important: Build Environment Setup

**CUDA compilation requires Visual Studio Developer Command Prompt environment.** Regular PowerShell/CMD will fail with `cudafe++` crashes.

### Method 1: Visual Studio Developer Command Prompt (Recommended)

1. Open **Developer Command Prompt for VS 2026** from Start Menu
2. Navigate to project:
   ```cmd
   cd C:\Users\<YourUser>\source\repos\ray_tracer\gpu\cuda
   ```
3. Build:
   ```cmd
   nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -o cuda_renderer.exe
   ```

### Method 2: From Regular PowerShell

Initialize VS environment first, then build:
```powershell
cd gpu\cuda
$vsPath = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vsPath`" && nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -o cuda_renderer.exe"
```

### Build Options

```cmd
# Debug build with optimizations disabled
nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -O0 -g -G -o cuda_renderer_debug.exe

# Release build with optimizations
nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -O3 -o cuda_renderer.exe

# Specific GPU architecture (RTX 5080 example)
nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -arch=sm_89 -o cuda_renderer.exe
```

## 🚀 Usage

### Basic Rendering

```cmd
cuda_renderer.exe [width] [height] [samples_per_pixel]
```

**Examples:**
```cmd
# Quick preview (2 samples)
cuda_renderer.exe 800 450 2

# Medium quality (10 samples)
cuda_renderer.exe 1920 1080 10

# High quality (100 samples)
cuda_renderer.exe 1920 1080 100
```

**Output**: Writes `image_cuda.ppm` to Desktop (OneDrive Desktop if available, else User Desktop)

### Performance Metrics

The renderer prints timing information:
```
[CUDA LOG] image=800x450 samples=4 bytes=1080000 kernel=9.613ms memcpy=0.204ms
Wrote C:\Users\...\Desktop\image_cuda.ppm (1080000 bytes)
```

## 📁 Project Structure

```
gpu/cuda/
├── host.cu                 # Main CUDA kernels (3 variants)
├── cuda_interface.cu       # C API wrapper for in-process integration
├── cuda_interface.h        # External C interface
├── cuda_scene.h           # POD structures for GPU data transfer
├── scene_serializer.cpp   # Scene conversion to GPU-friendly format
├── scene_serializer.h     # Scene serializer interface
└── README.md              # This file
```

## 🎨 Available Kernels

The `host.cu` file contains **three rendering kernels** for different use cases:

### 1. `render_kernel` - Simple Test Kernel
- **Purpose**: Basic GPU verification
- **Scene**: Single sphere at (0,0,-1) with radius 0.5
- **Shading**: Surface normals mapped to RGB
- **Use when**: Testing GPU setup, quick smoke tests

### 2. `render_kernel_serial` - Single-Bounce Renderer
- **Purpose**: Simplified lighting with scene support
- **Scene**: Multiple spheres and quads
- **Shading**: Direct lighting only (no bounces)
- **Use when**: Debugging scene setup, fast previews

### 3. `render_kernel_path` - Full Path Tracer (Default)
- **Purpose**: Production-quality rendering
- **Scene**: Full scene support (spheres, quads, materials)
- **Shading**: Multi-bounce path tracing
- **Materials**: Lambertian (diffuse), Metal (reflective), Dielectric (glass), Emissive (lights)
- **Use when**: Final rendering, quality output

**Currently active**: `render_kernel_path` (line 521 in host.cu)

## 🐛 Troubleshooting

### Issue: `cudafe++ died with status 0xC0000005`

**Cause**: Building outside Visual Studio Developer environment

**Solution**: Use Developer Command Prompt or initialize VS environment (see Build Instructions)

### Issue: `No CUDA-capable device is detected`

**Cause**: Driver issue or GPU not recognized

**Solutions**:
1. Update NVIDIA drivers: `nvidia-smi` to check current version
2. Verify GPU in Device Manager
3. Reinstall CUDA Toolkit

### Issue: Black or incorrect output

**Cause**: Scene data not loaded or material issues

**Solutions**:
1. Check console output for CUDA errors
2. Verify `serialize_scene_arrays()` is working correctly
3. Test with `render_kernel` (simple sphere) first

### Issue: Slow performance

**Possible causes**:
- High sample count (100+ samples)
- Large resolution (4K+)
- Complex scene geometry
- Debug build (-O0)

**Solutions**:
- Reduce samples for preview
- Use release build with `-O3`
- Check GPU utilization with `nvidia-smi`

## 🔬 Technical Details

### Memory Layout

- **Image buffer**: Interleaved RGB bytes, row-major order
- **Scene data**: POD structures (SpherePOD, QuadPOD, MaterialPOD, CameraPOD)
- **No dynamic allocation on device**: All buffers pre-allocated

### Random Number Generation

- **Algorithm**: xorshift32 PRNG
- **Seed per pixel**: Based on pixel coordinates + time seed
- **Quality**: Good enough for anti-aliasing and sampling

### Camera Model

- **Type**: Pinhole camera with configurable FOV
- **Controls**: lookfrom, lookat, vup vectors
- **FOV**: Vertical field of view in degrees

### Material System

Materials are defined in `MaterialPOD`:
```cpp
struct MaterialPOD {
	int type;        // 0=lambertian, 1=metal, 2=dielectric, 3=diffuse_light
	float r, g, b;   // color/albedo or emission
	float fuzz;      // for metal roughness
	float ref_idx;   // for dielectric IOR
	int is_emissive; // 0=no, 1=yes
};
```

## 📊 Performance Benchmarks

Tested on **NVIDIA RTX 5080** (CUDA 13.2, Driver 595.79):

| Resolution | Samples | Kernel Time | Total Time | FPS (equiv) |
|------------|---------|-------------|------------|-------------|
| 400×225    | 2       | 2.5ms       | 2.7ms      | ~370 fps    |
| 800×450    | 4       | 9.6ms       | 9.8ms      | ~100 fps    |
| 1920×1080  | 10      | ~100ms      | ~102ms     | ~10 fps     |

*Note: Kernel time = GPU computation only; Total time includes memory transfer*

## 🚧 Known Limitations

- **No BVH acceleration**: Linear intersection tests (O(n) per ray)
- **No texture support**: Solid colors only
- **Single scene**: Hard-coded Cornell box scene (can be modified in serializer)
- **No adaptive sampling**: Fixed samples per pixel
- **Memory limited**: Large scenes may exceed GPU VRAM

## 🔜 Future Improvements

- [ ] BVH acceleration structure for faster intersections
- [ ] Texture mapping and normal maps
- [ ] Environment maps / HDR backgrounds
- [ ] Adaptive sampling based on variance
- [ ] Multi-GPU support
- [ ] Real-time interactive rendering mode
- [ ] Integration with CPU renderer for hybrid mode

## 📚 References

- **Ray Tracing in One Weekend** series (Peter Shirley)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PBRT Book](https://pbr-book.org/) for path tracing theory

## 📝 Version History

- **2026-07-18**: Fixed build system (VS Dev environment requirement documented)
- **2026-07-18**: Full path tracing kernel implemented
- **2026-07-17**: Initial CUDA prototype with simple sphere
- **2026-07-17**: Project started

---

**Last Updated**: July 18, 2026  
**Author**: Xinpei Wang  
**License**: See project root
