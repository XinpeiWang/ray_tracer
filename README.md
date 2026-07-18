# Ray Tracer

A high-performance ray tracing renderer with both **CPU** and **GPU (CUDA)** implementations. This project demonstrates physically-based rendering techniques including path tracing, material systems, and image-based lighting.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.2%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

## 🎯 Features

### Core Rendering
- ✅ **Path tracing** with multiple bounces
- ✅ **Material system**: Lambertian (diffuse), Metal (reflective), Dielectric (glass), Emissive (lights)
- ✅ **Importance sampling** (cosine-weighted hemisphere)
- ✅ **Fresnel reflections** (Schlick approximation)
- ✅ **Anti-aliasing** through multi-sampling
- ✅ **Gamma correction** (gamma=2.0)
- ✅ **Cornell box** and custom scene support

### Dual Rendering Modes
- **CPU Renderer**: Multi-threaded, portable, debugging-friendly
- **GPU Renderer**: CUDA-accelerated, **10-100x faster** for complex scenes

### Scene Primitives
- Spheres with arbitrary center/radius
- Quads (axis-aligned rectangles) for boxes and walls
- Extensible geometry system

## 📊 Performance

**Cornell Box Scene (800×450 resolution, 10 samples/pixel):**

| Renderer | Hardware | Time | Speedup |
|----------|----------|------|---------|
| CPU | AMD/Intel (multi-threaded) | ~5-10s | 1× |
| GPU | NVIDIA RTX 5080 | ~50ms | **100-200×** |

**GPU Performance by Resolution (RTX 5080):**

| Resolution | Samples | Kernel Time | FPS (equiv) |
|------------|---------|-------------|-------------|
| 400×225    | 2       | 2.5ms       | ~400 fps    |
| 800×450    | 4       | 9.6ms       | ~100 fps    |
| 1920×1080  | 10      | ~100ms      | ~10 fps     |

## 🚀 Quick Start

### Prerequisites

**Required:**
- **Windows 10/11** (64-bit)
- **Visual Studio 2022 or 2026** with C++ desktop development
- **C++17 compatible compiler**

**Optional (for GPU rendering):**
- **NVIDIA GPU** (RTX series or GTX 16xx+, Compute Capability 7.5+)
- **CUDA Toolkit 13.2+** ([download](https://developer.nvidia.com/cuda-downloads))
- **Updated NVIDIA drivers**

### Building

#### 1. Clone the Repository

```bash
git clone https://github.com/XinpeiWang/ray_tracer.git
cd ray_tracer
```

#### 2. Build with Visual Studio

**Option A: Using Visual Studio IDE**
1. Open `ray_tracer.sln` in Visual Studio
2. Select **Release** configuration
3. Build → Build Solution (Ctrl+Shift+B)
4. Executable: `x64/Release/ray_tracer.exe`

**Option B: Using MSBuild (Command Line)**
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64
```

#### 3. Build GPU Renderer (Optional)

**From VS Developer Command Prompt:**
```cmd
cd gpu\cuda
nvcc host.cu -DBUILD_CUDA_STANDALONE=1 -O3 -o cuda_renderer.exe
```

See [gpu/cuda/README.md](gpu/cuda/README.md) for detailed CUDA build instructions.

### Running

#### CPU Rendering (Default if no GPU)
```cmd
ray_tracer.exe --cpu
```

#### GPU Rendering (CUDA)
```cmd
ray_tracer.exe --gpu
```

#### Standalone GPU Renderer
```cmd
cd gpu\cuda
cuda_renderer.exe 1920 1080 10
```

**Output**: Generates `image.ppm` (CPU) or `image_cuda.ppm` (GPU) on your Desktop.

## 📁 Project Structure

```
ray_tracer/
├── src/                          # Ray tracing library (based on "Ray Tracing in One Weekend")
│   ├── InOneWeekend/            # Book 1 implementation
│   ├── TheNextWeek/             # Book 2 implementation
│   ├── TheRestOfYourLife/       # Book 3 implementation
│   └── external/                # Third-party headers (stb_image, etc.)
│
├── launcher/                     # Main launcher application
│   ├── main.cpp                 # Entry point with CPU/GPU mode switching
│   └── book_bridge.cpp          # Scene setup and CPU rendering interface
│
├── raytracing_book/             # Standalone examples from books
│   └── main.cc                  # Book example runner
│
├── gpu/cuda/                    # CUDA GPU renderer
│   ├── host.cu                  # GPU kernels (3 variants)
│   ├── cuda_interface.cu        # C API wrapper
│   ├── cuda_scene.h             # POD structures for GPU data
│   ├── scene_serializer.cpp     # Scene conversion to GPU format
│   └── README.md                # Detailed GPU documentation
│
└── README.md                    # This file
```

## 🎨 Rendering Modes

### CPU Renderer

**Pros:**
- Portable (runs anywhere)
- Easy to debug
- Stable and well-tested

**Cons:**
- Slower (5-10s for 800×450 @ 10 samples)

**Usage:**
```cmd
ray_tracer.exe --cpu
```

### GPU Renderer (CUDA)

**Pros:**
- **10-100× faster** than CPU
- Real-time preview capable
- High sample counts feasible

**Cons:**
- Requires NVIDIA GPU
- CUDA setup complexity

**Usage:**
```cmd
ray_tracer.exe --gpu
```

**Three Kernel Variants Available:**
1. **Simple Kernel** - Single sphere test (debugging)
2. **Serial Kernel** - Single-bounce lighting
3. **Path Tracing Kernel** - Full multi-bounce path tracer (default)

## 🖼️ Example Scenes

### Cornell Box (Default)
Classic Cornell box with:
- Two large spheres (metal and lambertian)
- Colored walls (red, green, white)
- White floor and ceiling
- Bright area light at top

### Custom Scenes
Modify `scene_serializer.cpp` (GPU) or scene setup in `book_bridge.cpp` (CPU) to create custom scenes.

## 🔧 Configuration

### Render Settings

Edit in source files:

**CPU (`book_bridge.cpp`):**
```cpp
const int image_width = 800;
const int image_height = 450;
const int samples_per_pixel = 10;
const int max_depth = 50;
```

**GPU (`host.cu` main or `cuda_interface.cu`):**
```cpp
int image_width = 800;
int image_height = 450;
int samples_per_pixel = 10;
int max_depth = 50;
```

Or via command line (standalone GPU):
```cmd
cuda_renderer.exe 1920 1080 100
```

## 🐛 Troubleshooting

### CUDA Build Issues

**Problem:** `cudafe++ died with status 0xC0000005`

**Solution:** Use Visual Studio Developer Command Prompt, not regular PowerShell.

See [gpu/cuda/README.md](gpu/cuda/README.md) for detailed troubleshooting.

### Black or Incorrect Output

1. Check console for error messages
2. Verify scene data is loading correctly
3. Try reducing samples for faster feedback
4. Test with simple kernel first (GPU)

### Performance Issues

**CPU:**
- Enable Release configuration (Debug is 10× slower)
- Reduce samples per pixel
- Lower resolution

**GPU:**
- Update NVIDIA drivers
- Check GPU utilization: `nvidia-smi`
- Verify not running debug build
- Ensure adequate VRAM

## 🔬 Technical Details

### Material Types

```cpp
// Lambertian (diffuse)
material mat_diffuse = make_shared<lambertian>(color(0.8, 0.2, 0.2));

// Metal (reflective)
material mat_metal = make_shared<metal>(color(0.8, 0.8, 0.8), 0.1); // fuzz=0.1

// Dielectric (glass)
material mat_glass = make_shared<dielectric>(1.5); // IOR=1.5

// Emissive (light)
material mat_light = make_shared<diffuse_light>(color(15, 15, 15));
```

### Camera Model

Pinhole camera with:
- Configurable field of view (vertical)
- Lookfrom/lookat/vup vectors
- Focus distance and aperture (depth of field capable)

### Ray Tracing Algorithm

1. **Ray Generation**: Cast rays from camera through each pixel
2. **Intersection**: Test ray against all scene geometry
3. **Shading**: Evaluate material at hit point
4. **Bouncing**: Recursively trace scattered rays (up to max_depth)
5. **Accumulation**: Average multiple samples per pixel
6. **Tone Mapping**: Apply gamma correction

### Random Number Generation

- **CPU**: C++ standard library (`<random>`)
- **GPU**: xorshift32 PRNG (device-side, per-pixel seeded)

## 📚 References

This project is based on the excellent **"Ray Tracing in One Weekend"** series by Peter Shirley:

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
- [Ray Tracing: The Rest of Your Life](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html)

### Additional Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Physically Based Rendering Book](https://pbr-book.org/)
- [Scratchapixel - Ray Tracing](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-ray-tracing/how-does-it-work)

## 🚧 Known Limitations

### Current Limitations

- **No BVH acceleration structure**: Linear O(n) intersection tests
- **No texture mapping**: Solid colors only
- **Fixed scene**: Hardcoded Cornell box (modifiable in source)
- **No adaptive sampling**: Fixed samples per pixel
- **Windows only**: Platform-specific code (file paths, CUDA)

### Planned Improvements

- [ ] BVH acceleration structure for faster rendering
- [ ] Texture mapping and normal maps
- [ ] Environment maps / HDRI backgrounds
- [ ] Adaptive sampling based on variance
- [ ] Multi-GPU support
- [ ] Real-time interactive preview mode
- [ ] Scene file format (JSON/XML)
- [ ] Cross-platform support (Linux, macOS)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Performance**: BVH, better sampling strategies
2. **Features**: Textures, volumes, participating media
3. **Scenes**: Scene file parser, more examples
4. **Portability**: Linux/macOS support
5. **Documentation**: Tutorials, code comments

## 📝 License

This project is inspired by and includes code from the "Ray Tracing in One Weekend" series, which is licensed under CC0 1.0 Universal (public domain).

GPU implementation and project structure are original work.

See individual source files for specific attributions.

## 👤 Author

**Xinpei Wang**
- GitHub: [@XinpeiWang](https://github.com/XinpeiWang)
- Project: [ray_tracer](https://github.com/XinpeiWang/ray_tracer)

## 🌟 Acknowledgments

- **Peter Shirley** for the amazing "Ray Tracing in One Weekend" book series
- **NVIDIA** for CUDA and GPU computing resources
- **stb libraries** for image I/O

---

**Last Updated:** July 18, 2026  
**Version:** 1.0.0

View the [GPU documentation](gpu/cuda/README.md) for detailed CUDA build instructions.
