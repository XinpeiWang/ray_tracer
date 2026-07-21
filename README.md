# Ray Tracer

A high-performance ray tracing renderer with both **CPU** and **GPU (OptiX)** implementations. This project demonstrates physically-based rendering techniques including path tracing, material systems, and physically-based lighting.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![OptiX](https://img.shields.io/badge/OptiX-9.1%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

## 📦 Download & Use (No Build Required!)

**Want to try it without building?** Download the portable release:

1. [Download RayTracer_v1.0_Portable.zip](../../releases) from the Releases page
2. Extract to any folder
3. **GUI Version:** Double-click `RayTracerGUI.exe` for a graphical interface
   - OR **Console Version:** Double-click `launcher.bat` or `RayTracer.exe` for interactive command-line

The portable version includes:
- ✅ **Graphical User Interface** - Easy point-and-click rendering
- ✅ All required runtime dependencies (CUDA, Visual C++)
- ✅ Automatic GPU/CPU detection
- ✅ Interactive parameter selection (GUI or console)
- ✅ No installation needed - fully portable!

See [INSTALL.md](INSTALL.md) for detailed usage instructions.

## 🔨 Building from Source

**Quick build:**
```batch
# From Visual Studio Developer Command Prompt
scripts\build_and_deploy.ps1
```

For detailed build instructions, see **[BUILD.md](BUILD.md)**.

## 🎯 Features

### Core Rendering
- ✅ **Path tracing** with multiple bounces
- ✅ **Material system**: Lambertian (diffuse), Metal (reflective), Dielectric (glass), Emissive (lights)
- ✅ **Importance sampling** (cosine-weighted hemisphere)
- ✅ **Fresnel reflections** (Schlick approximation)
- ✅ **Anti-aliasing** through multi-sampling
- ✅ **Gamma correction** (gamma=2.0)
- ✅ **Cornell box** and custom scene support

### Video Generation 🎬
- ✅ **Animated camera paths**: orbit, linear, figure-8, spiral
- ✅ **Multi-frame rendering** with automatic frame numbering
- ✅ **Direct MP4 video encoding** using OpenCV (no FFmpeg required!)
- ✅ **Configurable FPS and quality** settings
- 📖 See [VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md) for detailed usage

### Dual Rendering Modes
- **CPU Renderer**: Multi-threaded, portable, debugging-friendly
- **GPU Renderer**: OptiX-accelerated ray tracing, **10-100x faster** for complex scenes

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

### For End Users (No Build Required)

Download the portable package and run it directly - see the [📦 Download section](#-download--use-no-build-required) above.

### For Developers

#### Prerequisites

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
# Build the entire solution (includes OptiX renderer)
scripts\build_all.bat
```

See [BUILD.md](BUILD.md) for detailed build instructions and [Project Structure](#-project-structure) for directory layout.

### Running (Development)

#### Interactive Mode (Recommended)
```cmd
ray_tracer.exe
```
The app will auto-detect your GPU and prompt for rendering settings interactively.

#### CPU Rendering
```cmd
ray_tracer.exe --cpu [width] [samples] [max_depth]
```

#### GPU Rendering (CUDA)
```cmd
ray_tracer.exe --gpu [width] [samples] [max_depth]
```

**Examples:**
```cmd
ray_tracer.exe --gpu 800 1000 20   # GPU, 800x800, 1000 samples
ray_tracer.exe --cpu 600 100 15    # CPU, 600x600, 100 samples
```

#### Video Generation 🎬
```cmd
# Render 60 frames with orbit camera path (video auto-assembled with OpenCV)
ray_tracer.exe --video --frames 60 --fps 30 --camera-path orbit 600 100 50

# Output: output/image_video.mp4 (created automatically, no separate assembly step needed!)
```

**NEW**: Video generation now uses OpenCV for direct video encoding - **no FFmpeg required!** 🎉  
The launcher automatically assembles frames into an MP4 video file during rendering.

See [VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md) for complete video generation guide.

**Output**: Generates both `image.ppm` (raw) and `image.png` (lossless) in the `output/` folder next to the executable.

### Image Format Support

The ray tracer automatically generates multiple output formats for convenience:

- **PNG Format**: `image.png` - Lossless, widely supported, smaller file size (created automatically)
- **PPM Format**: `image.ppm` - Raw pixel data, useful for debugging and further processing

Both formats are generated after each render completes.

## 📦 Distribution & Release Process

### Creating a Distribution Package

After building in Release mode, you can create a portable package:

```powershell
powershell -ExecutionPolicy Bypass -File .\package.ps1
```

This will:
- Copy the executable and rename it to `RayTracer.exe`
- Bundle all required runtime DLLs (CUDA, Visual C++)
- Include launcher scripts and documentation
- Create a `RayTracer_Package` folder ready for distribution

Then create a ZIP for easy distribution:
```powershell
Compress-Archive -Path .\RayTracer_Package\* -DestinationPath RayTracer_v1.0_Portable.zip
```

### Creating a GitHub Release

**Prerequisites:**
- Build successful in Release|x64 configuration
- All tests passing
- Documentation updated
- Version number decided (e.g., `v1.0`, `v1.1`)

**Step-by-Step Process:**

1. **Build the Release**
   ```cmd
   # From VS Developer Command Prompt
   msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64
   ```

2. **Create the Package**
   ```powershell
   # Run packaging script
   powershell -ExecutionPolicy Bypass -File .\package.ps1

   # Verify package contents
   dir RayTracer_Package

   # Test the package
   cd RayTracer_Package
   .\launcher.bat
   cd ..
   ```

3. **Create Distribution ZIP**
   ```powershell
   # Update version number in the filename
   Compress-Archive -Path .\RayTracer_Package\* -DestinationPath RayTracer_v1.0_Portable.zip
   ```

4. **Create GitHub Release**

   a. Go to your repository: https://github.com/XinpeiWang/ray_tracer

   b. Click **Releases** → **Draft a new release**

   c. Fill in release details:
   - **Tag version**: `v1.0` (or your version number)
   - **Release title**: `Ray Tracer v1.0 - Portable Edition`
   - **Description** (example):
   ```markdown
   ## Ray Tracer v1.0 - Cornell Box Path Tracer

   First official release of the GPU/CPU hybrid ray tracer!

   ### Features
   - ✅ Automatic GPU detection with CPU fallback
   - ✅ Interactive parameter selection
   - ✅ CUDA-accelerated path tracing
   - ✅ Multi-threaded CPU renderer
   - ✅ Cornell Box scene included
   - ✅ Portable - no installation required

   ### What's Included
   - RayTracer.exe - Main application
   - launcher.bat - Easy double-click launcher
   - Full documentation (README.txt, INSTALL.md)
   - All runtime dependencies (CUDA, Visual C++)

   ### System Requirements
   - Windows 10/11 (64-bit)
   - 4 GB RAM minimum
   - NVIDIA GPU with CUDA support (optional, for GPU mode)

   ### Quick Start
   1. Download `RayTracer_v1.0_Portable.zip`
   2. Extract to any folder
   3. Double-click `launcher.bat`
   4. Follow the prompts and enjoy!

   ### Performance
   - GPU Mode: ~5-15 seconds for high-quality renders
   - CPU Mode: ~1-5 minutes for good quality

   See [INSTALL.md](INSTALL.md) for detailed instructions.
   ```

   d. **Attach the ZIP file**: Drag and drop `RayTracer_v1.0_Portable.zip`

   e. Click **Publish release**

5. **Update README Link**

   Once published, update the README download link:
   ```markdown
   [Download RayTracer_v1.0_Portable.zip](https://github.com/XinpeiWang/ray_tracer/releases/download/v1.0/RayTracer_v1.0_Portable.zip)
   ```

6. **Verify the Release**
   - Download the ZIP from the release page
   - Extract and test on a clean machine (or VM)
   - Verify GPU detection works
   - Test both interactive and command-line modes
   - Check documentation is complete

### Release Checklist

Before publishing a release:

- [ ] Build successful in Release configuration
- [ ] GPU renderer tested and working
- [ ] CPU renderer tested and working
- [ ] Interactive mode tested
- [ ] All dependencies included in package
- [ ] Documentation up to date (README.md, INSTALL.md)
- [ ] Version number updated in release materials
- [ ] Package tested on clean system
- [ ] Release notes written
- [ ] ZIP file created and named correctly
- [ ] GitHub release created with proper tag
- [ ] Download link in README updated

### Versioning Guidelines

Follow semantic versioning: `vMAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes, major new features
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, small improvements

Examples:
- `v1.0` - Initial release
- `v1.1` - Added denoising feature
- `v1.1.1` - Fixed GPU memory leak
- `v2.0` - Switched to Vulkan backend (breaking change)

## 📁 Project Structure

```
ray_tracer/
├── src/                          # Ray tracing library (based on "Ray Tracing in One Weekend" series)
│   ├── InOneWeekend/            # Book 1: Basic ray tracer
│   ├── TheNextWeek/             # Book 2: BVH, textures, volumes
│   ├── TheRestOfYourLife/       # Book 3: Path tracing, PDFs (ACTIVE CODEBASE)
│   └── external/                # Third-party headers (stb_image, etc.)
│
├── launcher/                     # Main executable project
│   ├── main.cpp                 # Entry point with CPU/GPU mode switching
│   └── launcher.vcxproj         # Visual Studio project (auto-deploys to RayTracer_Package/)
│
├── cpu_renderer/                 # CPU path tracer (static library)
│   ├── cpu_interface.cpp/.h     # C API for CPU rendering
│   └── cpu_renderer.vcxproj     # Visual Studio project
│
├── optix_renderer/               # OptiX GPU renderer (static library)
│   └── optix_renderer.vcxproj   # Visual Studio project (auto-deploys PTX)
│
├── gpu/optix/                    # OptiX GPU implementation
│   ├── optix_programs.cu        # OptiX ray tracing kernels (compiled to PTX)
│   ├── optix_renderer.cpp/.h    # OptiX host-side renderer
│   ├── optix_interface.cpp/.h   # C API wrapper
│   ├── scene_builder.cpp/.h     # Scene conversion to OptiX format
│   └── optix_types.h            # Shared structures
│
├── qt_gui/                       # Qt 6 graphical interface
│   ├── RayTracerGUI.pro         # Qt project file
│   ├── mainwindow.cpp/.h        # Main GUI window
│   └── (Qt build output)        # Builds to RayTracer_Package/
│
├── tests/                        # Google Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
├── scripts/                      # Build and deployment scripts
│   ├── build_all.bat/.ps1       # Build all components
│   ├── build_and_deploy.ps1     # One-command build + deploy
│   ├── deploy_qt_gui.ps1        # Qt dependency deployment
│   └── setup_env.bat/.ps1       # Environment setup
│
├── docs/                         # Documentation (fixes, guides, migration notes)
│   └── (21 historical .md files moved here for organization)
│
├── RayTracer_Package/            # Deployment output (single canonical location)
│   ├── RayTracerGUI.exe         # Qt GUI (built from qt_gui/)
│   ├── ray_tracer.exe           # Console launcher (auto-deployed from launcher/)
│   ├── optix_programs.ptx       # GPU shader (auto-deployed from optix_renderer/)
│   └── Qt6*.dll + plugins       # Qt dependencies (deployed by scripts/deploy_qt_gui.ps1)
│
├── README.md                     # This file
├── BUILD.md                      # Detailed build instructions
├── INSTALL.md                    # Installation and usage guide
├── CODING_STANDARDS.md           # Code style guidelines
└── ray_tracer.sln                # Visual Studio solution
```

**Key Directories:**
- **src/TheRestOfYourLife/** - Active production codebase (Cornell box, path tracing, PDFs)
- **RayTracer_Package/** - Single canonical deployment directory (auto-populated by builds)
- **scripts/** - All build/deploy automation
- **docs/** - Historical documentation and migration guides

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

### OptiX Build Issues

**Problem:** `OptiX SDK not found` or missing PTX file

**Solution:** 
1. Ensure OptiX SDK 9.1+ is installed
2. Run `setup_env.bat` to configure environment variables
3. Check that `gpu/optix/optix_programs.ptx` exists after build

See [BUILD.md](BUILD.md) for detailed troubleshooting.

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

**Last Updated:** July 20, 2026  
**Version:** 2.0.0 (OptiX)

View the [OptiX GPU documentation](gpu/optix/README.md) for detailed OptiX build instructions.
