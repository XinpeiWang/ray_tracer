# Ray Tracer

A physically-based renderer with parallel **CPU** and **GPU (OptiX)** implementations, built up from the "Ray Tracing in One Weekend" book series into a much broader pbrt-v4-style feature set: 78 scenes, a wide material library, multiple light types, real triangle-mesh/texture support, BVH acceleration, volumetrics, and an experimental SPPM (photon-mapping) integrator alongside standard path tracing.

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
```powershell
# From Visual Studio Developer PowerShell
.\scripts\build_and_deploy.ps1
```

For detailed build instructions, see **[BUILD.md](BUILD.md)**.

## 🎯 Features

### Core Rendering
- ✅ **Path tracing** with next-event estimation and multiple importance sampling (power heuristic)
- ✅ **BVH acceleration** on both CPU and GPU (SAH-based CPU BVH; OptiX's native BVH/GAS on GPU) — not a linear scan
- ✅ **78 built-in scenes** (ids 0-78, id 53 retired) spanning the "Ray Tracing" book series, a pbrt-v4-style material/light/camera showcase, ~25 real-world statue/object meshes, and nine "movie-level" environment scenes (Sponza, Amazon Lumberyard Bistro, Rungholt, Fireplace Room, San Miguel, Sibenik Cathedral, Breakfast Room, Salle de Bain, Gallery) — see [Scenes](#-scenes) below
- ✅ **Real triangle meshes**: OBJ loading with BVH, per-face `.mtl` materials, and real `map_Kd` image-texture sampling (not just flat colors) on both CPU and GPU
- ✅ **Stochastic Progressive Photon Mapping (SPPM)**, an alternative integrator for hard caustic/glass scenes a standard path tracer struggles to converge — CPU-verified broadly, GPU-verified on one reference scene (see [Known Limitations](#-known-limitations))
- ✅ **Volumetric media**: homogeneous participating media and procedural (Perlin-noise) cloud/fog
- ✅ **Anti-aliasing** through multi-sampling, **ACES filmic tone mapping** + sRGB output

### Materials
Lambertian, Metal, Dielectric (smooth and rough), Conductor (GGX + complex Fresnel), Coated Diffuse/Conductor (clear-coat layering), Thin Dielectric, Diffuse Transmission, Normalized Fresnel, Principled (Disney-style multi-lobe), Hair (Marschner/Chiang fiber scattering), Normal/Bump mapping, homogeneous participating media, and mixed materials — see `docs/` and `src/TheRestOfYourLife/material_*.h` for details.

### Lighting
Area lights (quad/sphere), point/spot/distant (sun) punctual lights, goniometric (IES-profile) lights, projection lights, procedural sky, and image-based HDRI environment lighting (including portal-sampled HDRI through a window).

### Cameras
Pinhole, depth-of-field (thin-lens), orthographic, spherical/equirectangular 360°, and a realistic multi-element lens camera (double-Gauss, real exit-pupil sampling) — all with both CPU and GPU support.

### Video Generation 🎬
- ✅ **Animated camera paths**: orbit, linear, figure-8, spiral
- ✅ **Multi-frame rendering** with automatic frame numbering
- ✅ **MP4 video assembly** via an `ffmpeg` subprocess (requires `ffmpeg` on `PATH`)
- ✅ **Configurable FPS, speed, and quality** settings
- 📖 See [docs/VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md) for detailed usage

### Dual Rendering Modes
- **CPU Renderer**: Multi-threaded, importance-sampled, the most feature-complete and battle-tested path
- **GPU Renderer**: OptiX-accelerated, dramatically faster for complex scenes — has near-complete feature parity with CPU (see [Known Limitations](#-known-limitations) for the remaining gaps), plus an alternate queue-based **wavefront** path tracer (opt-in via `--wavefront`)

### Qt GUI
Scene picker with live metadata (description, GPU compatibility, perf hint), camera presets, quality/resolution presets, GPU/CPU toggle, image vs. video mode with camera-path selection, and one-click render.

## 📊 Performance

**Cornell Box Scene (800×450 resolution, 10 samples/pixel):**

| Renderer | Hardware | Time | Speedup |
|----------|----------|------|---------|
| CPU | AMD/Intel (multi-threaded) | ~5-10s | 1× |
| GPU | NVIDIA RTX 5080 | ~50ms | **100-200×** |

**GPU Performance by Resolution (RTX 5080):**

| Resolution | Samples | Kernel Time | FPS (equiv) |
|------------|---------|-------------|--------------|
| 400×225    | 2       | 2.5ms       | ~400 fps    |
| 800×450    | 4       | 9.6ms       | ~100 fps    |
| 1920×1080  | 10      | ~100ms      | ~10 fps     |

Large environment scenes (Bistro's 2.84M triangles + ~1.9GB of texture data) are naturally much slower to build and render than the Cornell Box — expect tens of seconds for scene setup alone, independent of GPU speed.

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

**Optional (for video generation):**
- **ffmpeg** on `PATH` — video rendering assembles frames into MP4 via an `ffmpeg` subprocess; without it, frames are still rendered to disk but not muxed into a video.

### Building

#### 1. Clone the Repository

```bash
git clone https://github.com/XinpeiWang/ray_tracer.git
cd ray_tracer
```

Some large mesh/texture assets (Sponza, Bistro, Rungholt and their textures) are tracked via **Git LFS** — make sure `git lfs` is installed before cloning, or run `git lfs pull` afterward if large assets show up as small pointer files.

#### 2. Prerequisites

**Required:**
- **Visual Studio 2022 or 2026** with C++ desktop development workload

**Optional — Qt GUI:**
- **Qt 6.11.1** with MinGW 64-bit component
- Add Qt to PATH: `$env:Path += ";C:\Qt\6.11.1\mingw_64\bin"`

**Optional — GPU rendering:**
- **CUDA Toolkit 12.x+** and **NVIDIA OptiX SDK 9.1+**
- Auto-detected from the standard install locations; set `$env:CudaToolkitPath` /
  `$env:OptixSdkPath` yourself only if you have multiple CUDA versions or a
  non-standard install path

#### 3. Build Options

Open a **Visual Studio Developer PowerShell** and choose one of:

**Option A: One command — build everything + deploy (Recommended)**
```powershell
.\scripts\build_and_deploy.ps1
```
Builds all components, deploys to `RayTracer_Package\` with Qt DLLs included, then run:
```powershell
.\RayTracer_Package\RayTracerGUI.exe
```

**Option B: PowerShell script with flags**
```powershell
.\scripts\build_all.ps1                         # Release (default)
.\scripts\build_all.ps1 -Configuration Debug    # Debug build
.\scripts\build_all.ps1 -SkipGui                # Skip Qt GUI (no Qt required)
.\scripts\build_all.ps1 -SkipTests              # Skip test projects
.\scripts\build_all.ps1 -Clean                  # Clean rebuild
.\scripts\build_all.ps1 -Deploy                 # Build + deploy Qt package
```

**Option C: Visual Studio IDE**
1. Open `ray_tracer.sln`
2. Right-click `launcher` in Solution Explorer → **Set as Startup Project**
3. Select **Release** / **x64**
4. **Build → Build Solution** (`Ctrl+Shift+B`)

**Option D: MSBuild directly**
```cmd
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64
```

#### 4. Output locations

| Component | Path |
|---|---|
| Console renderer | `x64\Release\ray_tracer.exe` |
| Qt GUI | `qt_gui\release\RayTracerGUI.exe` |
| Tests | `bin\Release\ray_tracer_tests.exe` |
| Deployed package | `RayTracer_Package\RayTracerGUI.exe` |

See [BUILD.md](BUILD.md) for full details, advanced options, and troubleshooting common issues (missing MSBuild, OptiX/CUDA errors, Qt not found).

### Running Tests

The test suite uses **Google Test** and covers roughly **3,300+ unit and integration tests** across ~170 test files.

#### Option A: Automated script (builds + runs in one step)
```powershell
cd tests
.\build_and_run_tests.ps1
```
Or with the batch file:
```cmd
cd tests
build_and_run_tests.bat
```

#### Option B: CMake (manual)
```powershell
cd tests
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
ctest -C Release --output-on-failure
```

#### Option C: Run the pre-built test binary directly
```cmd
bin\Release\ray_tracer_tests.exe
```
Filter to specific tests with Google Test flags:
```cmd
# Run only a subset by name pattern
bin\Release\ray_tracer_tests.exe --gtest_filter=Camera*

# List all available tests without running
bin\Release\ray_tracer_tests.exe --gtest_list_tests

# Show brief pass/fail summary
bin\Release\ray_tracer_tests.exe --gtest_brief=1
```

#### Option D: Visual Studio Test Explorer
Open `ray_tracer.sln`, then **Test → Test Explorer** and click **Run All**.

> Tests requiring an NVIDIA GPU (OptiX) are automatically skipped if no compatible GPU is present. Mesh-scene tests requiring external assets not present on disk are skipped too (not failed).

See [tests/TESTING_GUIDE.md](tests/TESTING_GUIDE.md) for the full guide including test structure and how to add new tests.

### Running (Development)

#### Interactive Mode (Recommended)
```cmd
ray_tracer.exe
```
The app will auto-detect your GPU and prompt for rendering settings interactively.

#### CPU Rendering
```cmd
ray_tracer.exe --cpu [width] [samples] [max_depth] [scene_id]
```

#### GPU Rendering (CUDA/OptiX, default)
```cmd
ray_tracer.exe --gpu [width] [samples] [max_depth] [scene_id]
```

**Examples:**
```cmd
ray_tracer.exe --gpu 800 1000 20 0    # GPU, Cornell Box, 800x800, 1000 samples
ray_tracer.exe --cpu 600 100 15 42    # CPU, Stanford Dragon, 600x600, 100 samples
```

#### SPPM (Photon Mapping)
```cmd
ray_tracer.exe --sppm 600 300 10 11          # CPU SPPM, scene 11 (Cornell Rough Glass)
ray_tracer.exe --sppm --gpu 600 300 10 11    # GPU SPPM (scene 11 only, see Known Limitations)
```

#### Video Generation 🎬
```cmd
# Render 60 frames with orbit camera path (assembled into MP4 via ffmpeg)
ray_tracer.exe --video --frames 60 --fps 30 --camera-path orbit 600 100 50

# Output: output/image_video.mp4
```

See [docs/VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md) for complete video generation guide.

**Output**: Generates both `image.ppm` (raw) and `image.png` (lossless) in the `output/` folder next to the executable.

### Image Format Support

The ray tracer automatically generates multiple output formats for convenience:

- **PNG Format**: `image.png` - Lossless, widely supported, smaller file size (created automatically)
- **PPM Format**: `image.ppm` - Raw pixel data, useful for debugging and further processing

Both formats are generated after each render completes.

## 🖼️ Scenes

78 scenes (ids 0-78, id 53 retired), selected via the `scene_id` CLI argument or the GUI's scene dropdown. Full descriptions and camera defaults live in [src/TheRestOfYourLife/scene_registry.h](src/TheRestOfYourLife/scene_registry.h).

| Range | Category |
|---|---|
| 0-8, 22 | "Ray Tracing" book 1-3 progression (Cornell Box, Bouncing Spheres, Checkered/Perlin/Earth spheres, Cornell Smoke, depth of field, ...) |
| 9-21, 23 | pbrt-v4 material/BxDF showcase — mostly Cornell-box variants exercising rough metal, conductor, coated diffuse/conductor, thin glass, hair, subsurface, bilinear patches |
| 24-36 | pbrt-v4 light/camera/medium showcase — HDRI sky, spot/distant/point/goniometric/projection lights, homogeneous & cloud media, orthographic/spherical/realistic-lens cameras, measured BRDFs |
| 37-61 | Real-world statue/object meshes (Stanford Bunny, Armadillo, Happy Buddha, Lucy, XYZRGB Dragon, Utah Teapot, Suzanne, Nefertiti, and ~15 more) — one object per scene, checkered floor, overhead light |
| 62-64, 73-78 | "Movie-level" environment scenes: Crytek Sponza (262K tris), Amazon Lumberyard Bistro Exterior (2.84M tris), Rungholt (6.7M tris), Fireplace Room, San Miguel (9.9M tris), Sibenik Cathedral, Breakfast Room, Salle de Bain, Gallery — real per-face `.mtl` materials, with real `map_Kd` image textures on all nine |
| 65-68 | The classic Cornell box (same scene as A1/id 0), rendered by each of D1-D4's camera models in turn (defocus blur, orthographic, 360° spherical panorama, realistic-lens bokeh) — a fixed reference scene for directly comparing the four camera types |
| 69-70 | Additional volumetric-media showcases (a combined dielectric+medium sphere; a heterogeneous RGB-grid medium) — same category as 24-36's E1/E2, added later so non-contiguous |
| 71-72 | Additional geometry showcases (real object instancing; real Bezier curve/hair-fiber geometry, not the shading-normal proxy scene 19 uses) — same category as scene 23/37, added later so non-contiguous |

Scenes 37-64, 73-78 load external `.obj` assets from `models/` (Git LFS for the large ones); scenes 62-63 and 73-78 additionally need their own texture directories (`models/sponza_textures/`, `models/bistro_textures/`, `models/fireplace_room_textures/`, `models/san_miguel_textures/`, `models/sibenik_cathedral_textures/`, `models/breakfast_room_textures/`, `models/salle_de_bain_textures/`, `models/gallery_textures/`).

## 📦 Distribution & Release Process

### Creating a Distribution Package

After building in Release mode, you can create a portable package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package.ps1
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
   powershell -ExecutionPolicy Bypass -File .\scripts\package.ps1

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
   - **Description**: summarize what's new since the last release (new scenes, integrators, materials, fixes)

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

## 📁 Project Structure

```
ray_tracer/
├── src/                          # Ray tracing library (based on "Ray Tracing in One Weekend" series)
│   ├── InOneWeekend/             # Book 1: Basic ray tracer
│   ├── TheNextWeek/              # Book 2: BVH, textures, volumes
│   ├── TheRestOfYourLife/        # Book 3+: path tracing, PDFs, materials, scenes (ACTIVE CODEBASE)
│   ├── shared/                   # CPU/GPU-shared headers: pbrt-v4-style BxDFs, cameras, lights, sampling
│   └── external/                 # Third-party headers (stb_image, etc.)
│
├── launcher/                     # Unified launcher: the ray_tracer.exe entry point
│   ├── main.cpp                  # CPU/GPU/SPPM/video dispatch
│   ├── launcher_args.h           # Command-line parsing
│   ├── camera_path.h             # Video camera paths
│   └── launcher.vcxproj          # Visual Studio project (auto-deploys to RayTracer_Package/)
│
├── cpu_renderer/                  # CPU path tracer (static library)
│   ├── cpu_interface.cpp/.h      # C API for CPU rendering
│   └── cpu_renderer.vcxproj      # Visual Studio project
│
├── optix_renderer/                # OptiX GPU renderer (static library)
│   └── optix_renderer.vcxproj    # Visual Studio project (auto-deploys PTX)
│
├── gpu/optix/                     # OptiX GPU implementation
│   ├── optix_programs.cu         # Recursive (mega-kernel) OptiX ray tracing kernels
│   ├── wavefront_*.cu/.h         # Queue-based wavefront path tracer (alt. GPU backend)
│   ├── sppm_*.cu/.h              # GPU SPPM (photon mapping) backend
│   ├── optix_renderer.cpp/.h     # OptiX host-side renderer
│   ├── optix_interface.cpp/.h    # C API wrapper
│   ├── scene_builder.cpp/.h      # Scene conversion to OptiX format (all 78 scenes)
│   └── optix_types.h             # Shared structures (materials, geometry, launch params)
│
├── qt_gui/                        # Qt 6 graphical interface
│   ├── RayTracerGUI.pro          # Qt project file
│   ├── mainwindow.cpp/.h         # Main GUI window
│   └── (Qt build output)         # Builds to RayTracer_Package/
│
├── models/                        # Mesh (.obj) and texture assets, Git LFS for the large ones
│
├── tests/                         # Google Test suite (~3,300 tests)
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── scripts/                       # Build and deployment scripts
│   ├── build_all.bat/.ps1        # Build all components
│   ├── deploy_launcher.ps1       # Verify the launcher deployed to RayTracer_Package/
│   ├── build_and_deploy.ps1      # One-command build + deploy
│   ├── deploy_qt_gui.ps1         # Qt dependency deployment
│   └── setup_env.bat/.ps1        # Environment setup
│
├── docs/                          # Feature guides, migration notes, architecture docs
│
├── RayTracer_Package/              # Deployment output (single canonical location)
│   ├── RayTracerGUI.exe          # Qt GUI (built from qt_gui/)
│   ├── ray_tracer.exe            # Console launcher (auto-deployed from launcher/)
│   ├── optix_programs.ptx        # GPU shader (auto-deployed from optix_renderer/)
│   └── Qt6*.dll + plugins        # Qt dependencies (deployed by scripts/deploy_qt_gui.ps1)
│
├── README.md                      # This file
├── BUILD.md                       # Detailed build instructions
├── INSTALL.md                     # Installation and usage guide
├── CODING_STANDARDS.md            # Code style guidelines
└── ray_tracer.sln                 # Visual Studio solution
```

**Key Directories:**
- **src/TheRestOfYourLife/** and **src/shared/** - Active production codebase (materials, lights, cameras, scenes)
- **gpu/optix/** - GPU implementation, mirrors the CPU feature set (see [Known Limitations](#-known-limitations) for gaps)
- **models/** - External mesh/texture assets (Git LFS)
- **RayTracer_Package/** - Single canonical deployment directory (auto-populated by builds)
- **scripts/** - All build/deploy automation
- **docs/** - Feature guides and architecture notes

## 🎨 Rendering Modes

### CPU Renderer

**Pros:**
- Most feature-complete (all materials/lights/integrators, including SPPM broadly and BDPT/MLT via `--bdpt`/`--mlt`)
- Portable, easy to debug, stable and well-tested

**Cons:**
- Slower than GPU for complex scenes

**Usage:**
```cmd
ray_tracer.exe --cpu
```

### GPU Renderer (OptiX/CUDA)

**Pros:**
- **10-100×+ faster** than CPU for most scenes
- Near feature-complete: same material library, most lights/cameras, mesh+texture support, and SPPM on one reference scene
- Two backends: the default recursive mega-kernel path tracer, and an opt-in wavefront (queue-based) path tracer (`--wavefront`)

**Cons:**
- Requires NVIDIA GPU + CUDA/OptiX setup
- A handful of features remain CPU-only or scene-limited — see [Known Limitations](#-known-limitations)

**Usage:**
```cmd
ray_tracer.exe --gpu
```

## 🔧 Configuration

Rendering settings (resolution, samples, depth, scene) are passed via CLI arguments or the interactive/GUI prompts — see [Running (Development)](#running-development) above. There is no separate config file; scene definitions themselves live in [src/TheRestOfYourLife/scene_registry.h](src/TheRestOfYourLife/scene_registry.h) (CPU) and [gpu/optix/scene_builder.cpp](gpu/optix/scene_builder.cpp) (GPU).

## 🐛 Troubleshooting

### OptiX Build Issues

**Problem:** `OptiX SDK not found` or missing PTX file

**Solution:** 
1. Ensure OptiX SDK 9.1+ is installed
2. Run `scripts\setup_env.ps1` (or `.bat`) to configure environment variables
3. Check that `gpu/optix/optix_programs.ptx` exists after build, and that it was copied to `RayTracer_Package/`

See [BUILD.md](BUILD.md) for detailed troubleshooting.

### Black or Incorrect Output

1. Check console for error messages
2. Verify scene/mesh assets are present (mesh scenes need `models/`, some need Git LFS pulled)
3. Try reducing samples for faster feedback
4. For a new mesh scene, verify the camera isn't embedded in geometry or in a fully-enclosed, unlit room

### Performance Issues

**CPU:**
- Enable Release configuration (Debug is 10× slower)
- Reduce samples per pixel
- Lower resolution

**GPU:**
- Update NVIDIA drivers
- Check GPU utilization: `nvidia-smi`
- Verify not running debug build
- Ensure adequate VRAM (large environment scenes can use several GB of texture data alone)

## 🔬 Technical Details

### Material Types (partial list — see [Features](#materials) above for the full library)

```cpp
// Lambertian (diffuse)
auto mat_diffuse = make_shared<lambertian>(color(0.8, 0.2, 0.2));

// Metal (reflective)
auto mat_metal = make_shared<metal>(color(0.8, 0.8, 0.8), 0.1); // fuzz=0.1

// Dielectric (glass)
auto mat_glass = make_shared<dielectric>(1.5); // IOR=1.5

// Emissive (light)
auto mat_light = make_shared<diffuse_light>(color(15, 15, 15));

// Real per-material image texture (map_Kd), sampled via mesh UVs
auto mat_textured = make_shared<lambertian>(make_shared<image_texture>("brick_diff.png"));
```

### Camera Model

pbrt-v4-style camera abstraction with multiple implementations (pinhole, thin-lens, orthographic, spherical, realistic multi-element lens) — see [Cameras](#cameras) above. The default pinhole camera supports:
- Configurable field of view (vertical)
- Lookfrom/lookat/vup vectors
- Focus distance and aperture (depth of field capable)

### Ray Tracing Algorithm

1. **Ray Generation**: Cast rays from camera through each pixel
2. **Intersection**: BVH-accelerated traversal against scene geometry (CPU BVH / OptiX GAS on GPU)
3. **Shading**: Evaluate material BxDF at hit point, with next-event estimation + multiple importance sampling against scene lights
4. **Bouncing**: Recursively trace scattered rays (up to max_depth, Russian roulette on CPU)
5. **Accumulation**: Average multiple samples per pixel
6. **Tone Mapping**: ACES filmic tone mapping + sRGB OETF

An alternative SPPM (photon mapping) integrator is available for scenes with hard-to-converge caustics — see [SPPM (Photon Mapping)](#sppm-photon-mapping) above.

### Random Number Generation

- **CPU**: PCG-family generator, thread-local, stratified/Sobol low-discrepancy sampling for many integrators
- **GPU**: PCG hash-based PRNG (device-side, per-pixel/per-bounce seeded)

## 📚 References

This project is based on the excellent **"Ray Tracing in One Weekend"** series by Peter Shirley, and its material/light/camera library draws heavily on **pbrt-v4**:

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
- [Ray Tracing: The Rest of Your Life](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html)
- [Physically Based Rendering: From Theory to Implementation (pbrt-v4)](https://pbr-book.org/)

### Additional Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Scratchapixel - Ray Tracing](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-ray-tracing/how-does-it-work)

### Mesh & Texture Credits

External mesh/texture assets (`models/`) come from the Stanford 3D Scanning Repository, the McGuire Computer Graphics Archive (Crytek Sponza, Amazon Lumberyard Bistro, Rungholt), and the common-3d-test-models collection — see each model's own license/attribution where noted.

## 🚧 Known Limitations

Being upfront about what's incomplete rather than overselling:

- **GPU SPPM is scene-limited**: the GPU photon-mapping backend has only been verified end-to-end on one reference scene (Cornell Rough Glass). CPU SPPM works across a much broader set of materials/lights, though it too is primarily verified on lambertian + delta-BSDF scenes.
- **BDPT and MLT are CPU-only and narrow in scope**: selectable via `--bdpt`/`--mlt`, but there's no GPU/OptiX implementation (`--gpu` is ignored with a warning), only area lights are supported for NEE (no punctual/sky-light sampling yet), and both are verified end-to-end on scene A1 (Cornell Box) only — other scenes are unverified.
- **Hair/fur has two different fidelity levels**: scene F4 (Curve Fibers) uses real Bezier curve/strand geometry (`CurveShape`, exact ray-curve intersection on CPU, tessellated bilinear-patch tubes on GPU); the older scene B11 instead applies the Marschner/Chiang BxDF math via a shading-normal proxy on sphere primitives, not actual fiber geometry.
- **GPU wavefront path tracer is opt-in and less exercised**: enabled via the `--wavefront` flag; the default recursive GPU backend is the primary, best-tested GPU path.
- **Windows only**: platform-specific code throughout (file paths, CUDA/OptiX integration, build system).
- **No adaptive sampling**: fixed samples-per-pixel for standard path tracing (SPPM itself is progressive by design).

### Planned / possible future work

- [ ] GPU/OptiX implementation of BDPT/MLT, plus punctual/sky-light NEE support
- [ ] Broader GPU SPPM scene support
- [ ] Real curve/strand geometry for scene B11's hair fibers (matching scene F4's approach)
- [ ] Adaptive sampling based on variance
- [ ] Cross-platform support (Linux, macOS)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Integrators**: porting BDPT/MLT to GPU, broadening their light-sampling and scene coverage, broadening GPU SPPM scene support
2. **Geometry**: real curve/hair geometry for scene B11 (scene F4 already has it), more mesh formats
3. **Scenes**: more example scenes, a scene file format (JSON/XML) instead of hardcoded registry entries
4. **Portability**: Linux/macOS support
5. **Documentation**: tutorials, code comments

## 📝 License

This project is inspired by and includes code from the "Ray Tracing in One Weekend" series, which is licensed under CC0 1.0 Universal (public domain). Material/light/camera algorithms are original implementations informed by the publicly available pbrt-v4 book text.

GPU implementation and project structure are original work.

See individual source files for specific attributions, and the [Mesh & Texture Credits](#mesh--texture-credits) section above for external asset licensing.

## 👤 Author

**Xinpei Wang**
- GitHub: [@XinpeiWang](https://github.com/XinpeiWang)
- Project: [ray_tracer](https://github.com/XinpeiWang/ray_tracer)

## 🌟 Acknowledgments

- **Peter Shirley** for the "Ray Tracing in One Weekend" book series
- **Matt Pharr, Wenzel Jakob, and Greg Humphreys** for pbrt-v4, whose published algorithms informed much of this renderer's material/light/sampling library
- **NVIDIA** for CUDA, OptiX, and GPU computing resources
- **stb libraries** for image I/O
- **Morgan McGuire** and the McGuire Computer Graphics Archive for the Sponza/Bistro/Rungholt scenes

---

**Last Updated:** August 11, 2026
**Version:** 2.1.0 (Textured meshes + expanded scene library)

View the [OptiX GPU documentation](gpu/optix/README.md) for detailed OptiX build instructions.
