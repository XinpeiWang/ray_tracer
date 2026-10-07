# Ray Tracer

A physically-based renderer with parallel **CPU**, **GPU (OptiX)** (Windows + NVIDIA) and **GPU (Metal)** (macOS) implementations, built up from the "Ray Tracing in One Weekend" book series into a much broader pbrt-v4-style feature set: 151 built-in scenes and 58 test scenes, a wide material library, multiple light types, real triangle-mesh/texture support, BVH acceleration, volumetrics, and an experimental SPPM (photon-mapping) integrator alongside standard path tracing.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey.svg)
![OptiX](https://img.shields.io/badge/OptiX-9.1%2B-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

![A Cornell box with a glass sphere, the Stanford bunny in bronze, and glass spheres containing coloured fog, all rendered by this project](docs/gallery/hero.jpg)

## 🖼️ Gallery

Rendered with this project: 720 x 720, 2048 samples per pixel, max depth 12, the OptiX GPU backend on an RTX 5080, with the OptiX AI denoiser at blend 0.1 (so a little grain is kept). The time is the renderer's own "RENDER TIME" (denoising included). Every scene is built in; the id in brackets is the scene id the command line and the GUI use (for example `ray_tracer.exe --gpu --denoise 720 2048 12 A1`).

| | | |
|:-:|:-:|:-:|
| <img src="docs/gallery/A1-cornell-box.jpg" width="260"><br>**Cornell box (A1)**<br>glass sphere, aluminium box, caustic<br>6 s | <img src="docs/gallery/A9-final-scene.jpg" width="260"><br>**Final scene (A9)**<br>the "Next Week" cover scene: volumes, earth map, glass, noise<br>15 s | <img src="docs/gallery/G1-stanford-bunny.jpg" width="260"><br>**Stanford bunny (G1)**<br>69k triangles, polished bronze<br>5 s |
| <img src="docs/gallery/B10-principled-showcase.jpg" width="260"><br>**Principled BSDF (B10)**<br>matte plastic to clear-coated metal<br>4 s | <img src="docs/gallery/E3-glass-with-fog.jpg" width="260"><br>**Glass with fog (E3)**<br>coloured scattering media inside dielectrics<br>3 s | <img src="docs/gallery/A8-cornell-smoke.jpg" width="260"><br>**Cornell smoke (A8)**<br>participating media, multiple scattering<br>8 s |
| <img src="docs/gallery/B7-coated-conductor.jpg" width="260"><br>**Coated conductor (B7)**<br>pbrt-v4 layered BxDF: lacquered gold<br>22 s | <img src="docs/gallery/D1-depth-of-field.jpg" width="260"><br>**Depth of field (D1)**<br>thin-lens camera<br>3 s | <img src="docs/gallery/F4-curve-fibers.jpg" width="260"><br>**Curve fibres (F4)**<br>real Bezier strand geometry<br>10 s |
| <img src="docs/gallery/G12-trophy-room.jpg" width="260"><br>**Trophy room (G12)**<br>four meshes, mixed materials<br>4 s | <img src="docs/gallery/B3-rough-glass.jpg" width="260"><br>**Rough glass (B3)**<br>GGX rough dielectric<br>7 s | <img src="docs/gallery/B13-wax-and-jade.jpg" width="260"><br>**Wax and jade (B13)**<br>subsurface-like translucency<br>10 s |

## ✅ How the renders are checked

A rendering bug is quiet: the picture still looks plausible while being 20% too dark. So the test suite checks absolute numbers, not only that the backends agree with each other.

- **Closed forms.** Furnace scenes (a diffuse sphere under a white sky must read its albedo), a closed diffuse-transmission shell (0.2, 0.56, 0.632, 0.6464 ... 0.65), per-channel Beer-Lambert absorbers, and a point light over a plate are read by the CPU, both OptiX backends and the alternative integrators.
- **Cross-checks.** The CPU path tracer against the two OptiX backends and Metal, and BDPT, MLT, SPPM and the debug integrators against the path tracer, on purpose-built scenes.
- **pbrt-v4 as the reference.** Where behaviour is in doubt, the pbrt-v4 source is read, and an independent script (`scripts/pbrt_rough_glass_reference.py`) provides a path-level reference for rough glass.
- **Over 5,000 automated tests**, under ten minutes on a desktop GPU. [`docs/PBRT_SUPPORT.md`](docs/PBRT_SUPPORT.md) lists, feature by feature and backend by backend, what is supported and how closely it matches.

How this found more than a dozen bugs that a plain backend-versus-backend comparison could not see: [Closed forms found the bugs](docs/CLOSED_FORMS_FOUND_THE_BUGS.md).

## ⚡ Try it in a few minutes, no NVIDIA GPU needed

The CPU renderer is a complete renderer and builds with plain CMake and a C++17 compiler (on Windows, Visual Studio with the C++ workload). No CUDA, OptiX or Qt is needed:

```bash
git clone --depth 1 https://github.com/XinpeiWang/ray_tracer
cd ray_tracer
cmake -S . -B build
cmake --build build --config Release --target ray_tracer
# run from the repository root, so the scenes and textures are found:
build/Release/ray_tracer --cpu --output cornell.png 400 128 8 A1      # Windows (Visual Studio generator)
build/ray_tracer          --cpu --output cornell.png 400 128 8 A1      # macOS and other single-config generators
```

That renders the Cornell box (A1) at 400 px with 128 samples per pixel and depth 8: about 5 seconds on a 16-core desktop CPU (the build took under a minute). The arguments are `[width] [samples] [max depth] [scene]`, where the scene is a name such as `cornell-box`, an id such as `A1`, or the path of a `.pbrt` file; `--gpu` on a build without GPU support falls back to the CPU with a warning. Add `-DRT_BUILD_GPU=ON` (Windows with CUDA and OptiX) or `-DRT_BUILD_METAL=ON` (macOS) to the first `cmake` line for the GPU backends, or use the portable release below.

## 🧱 Build your own scene

The GUI has a **Scene Builder** tab: add spheres, boxes, quads, cylinders, cones, meshes and lights, pick materials (matte, metal, glass, glossy paint, translucent) and colours, drag things around in a top, front or side view, press Preview, and save the result as an ordinary `.pbrt` file that the renderer and any pbrt-compatible tool can read. See [docs/SCENE_BUILDER.md](docs/SCENE_BUILDER.md).

![The Scene Builder tab](docs/gallery/scene-builder.jpg)

## 📦 Download & Use (No Build Required!)

**Want to try it without building?** Download the portable release:

1. [Download the latest portable release](../../releases) from the Releases page
2. Extract to any folder
3. **GUI Version:** Double-click `RayTracerGUI.exe` for a graphical interface
   - OR **Console Version:** Run `RayTracer.exe` from a terminal for command-line rendering

The portable version includes:
- ✅ **Graphical User Interface** - Easy point-and-click rendering
- ✅ All required runtime dependencies (CUDA, Visual C++)
- ✅ Automatic GPU/CPU detection
- ✅ Interactive parameter selection (GUI or console)
- ✅ No installation needed - fully portable!

See [INSTALL.md](INSTALL.md) for detailed usage instructions.

**macOS:** a `RayTracerGUI.dmg` (Metal GPU rendering and Live Preview included; native on Apple silicon; `--arch universal` builds one dmg for Apple-silicon and Intel Macs) is built by `scripts/build_and_deploy_macos.sh` — see [BUILD.md](BUILD.md#macos-cpu-and-metal-gpu). It is not code-signed, so on first launch right-click the app → **Open**.

## 🔨 Building from Source

**Cloning:** the full history is about 440 MB, because older commits contained generated GPU code and release archives that are no longer tracked. If you only want the code, take a shallow clone: `git clone --depth 1 https://github.com/XinpeiWang/ray_tracer` downloads about 52 MB (150 MB checked out). A few large mesh and texture assets are tracked with Git LFS; run `git lfs pull` if they show up as small pointer files.

**Windows** (CPU, OptiX GPU and the Qt GUI), from a Visual Studio Developer PowerShell:
```powershell
.\scripts\build_and_deploy.ps1
```
You need Visual Studio 2022 or 2026 with the C++ workload. Add the CUDA Toolkit 13.2+ and the NVIDIA OptiX SDK 9.1+ for the GPU renderer, and Qt 6.11.1 (MSVC 2022 64-bit) for the GUI; `ffmpeg` on `PATH` is needed only to assemble videos.

**macOS** (the CPU renderer, CLI and Qt GUI, plus the Metal GPU backend): `cmake -B build && cmake --build build`, or `./scripts/build_and_deploy_macos.sh` for a `.app` and `.dmg`.

Both in full, with the output locations, the macOS Metal and universal-binary options, and troubleshooting: **[BUILD.md](BUILD.md)**. The tests: [tests/TESTING_GUIDE.md](tests/TESTING_GUIDE.md). Where things are: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## 🎯 Features

### Core Rendering
- ✅ **Path tracing** with next-event estimation and multiple importance sampling (power heuristic)
- ✅ **BVH acceleration** on both CPU and GPU (SAH-based CPU BVH; OptiX's native BVH/GAS on GPU) — not a linear scan
- ✅ **151 built-in scenes and 58 test scenes** (each has a stable name such as `cornell-box`, and a short id such as `A1`) spanning the "Ray Tracing" book series, a pbrt-v4-style material/light/camera showcase, dozens of real-world statue/object meshes, and several "movie-level" environment scenes (Sponza, Amazon Lumberyard Bistro, Rungholt, Fireplace Room, San Miguel, Sibenik Cathedral, Breakfast Room, Salle de Bain, Gallery) — see [Scenes](#-scenes) below
- ✅ **Real triangle meshes**: OBJ loading with BVH, per-face `.mtl` materials, and real `map_Kd` image-texture sampling (not just flat colors) on both CPU and GPU
- ✅ **Stochastic Progressive Photon Mapping (SPPM)**, an alternative integrator for hard caustic/glass scenes a standard path tracer struggles to converge — on the CPU, and on the GPU for sphere/quad scenes (see [Known Limitations](#-known-limitations))
- ✅ **Bidirectional Path Tracing (BDPT) and Metropolis Light Transport (MLT)**, additional alternative integrators (CPU-only, `--bdpt`/`--mlt`) for scenes with difficult light transport
- ✅ **Adaptive sampling** (`--adaptive`): stops sampling pixels that have already converged (CPU; opt-in on Metal)
- ✅ **Volumetric media**: homogeneous participating media, procedural (Perlin-noise) cloud/fog, and heterogeneous NanoVDB grid media (CPU-only)
- ✅ **Anti-aliasing** through multi-sampling, **ACES filmic tone mapping** + sRGB output

### Materials
Lambertian, Metal, Dielectric (smooth and rough), Conductor (GGX + complex Fresnel), Coated Diffuse/Conductor (clear-coat layering), Thin Dielectric, Diffuse Transmission, Normalized Fresnel, Principled (Disney-style multi-lobe), Hair (Marschner/Chiang fiber scattering), Measured (real importance-sampled pbrt-v4 `MeasuredBxDF`, loaded from a `.bsdf` tensor file), Subsurface (tabulated BSSRDF), Normal/Bump mapping, homogeneous participating media (plus rough/thin-dielectric+medium fusion for a single shape that's both a refractive boundary and a scattering volume), and mixed materials — see `docs/` and `src/TheRestOfYourLife/material_*.h` for details.

### Lighting
Area lights (quad/sphere), point/spot/distant (sun) punctual lights, goniometric (IES-profile) lights, projection lights, procedural sky, and image-based HDRI environment lighting (including portal-sampled HDRI through a window).

### Cameras
Pinhole, depth-of-field (thin-lens), orthographic, spherical/equirectangular 360°, and a realistic multi-element lens camera (double-Gauss, real exit-pupil sampling) — all with both CPU and GPU support.

### Video Generation 🎬
- ✅ **Animated camera paths**: orbit, linear, figure-8, spiral, interior tour and product showcase
- ✅ **Multi-frame rendering** with automatic frame numbering
- ✅ **MP4 video assembly** via an `ffmpeg` subprocess (requires `ffmpeg` on `PATH`)
- ✅ **Configurable FPS, speed, and quality** settings
- 📖 See [docs/VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md) for detailed usage

### Dual Rendering Modes
- **CPU Renderer**: Multi-threaded, importance-sampled, the most feature-complete and battle-tested path
- **GPU Renderer**: OptiX-accelerated, dramatically faster for complex scenes — has near-complete feature parity with CPU (see [Known Limitations](#-known-limitations) for the remaining gaps), plus an alternate queue-based **wavefront** path tracer (opt-in via `--wavefront`)
- **Metal GPU Renderer** (macOS): a Metal path tracer (`gpu/metal/`, documented in `docs/METAL_BACKEND.md`, `--gpu` on a `-DRT_BUILD_METAL=ON` build) checked against the CPU renderer scene by scene — see [`docs/METAL_PARITY_STATUS.md`](docs/METAL_PARITY_STATUS.md) for exactly what matches and what doesn't

### Qt GUI
A **Scene Builder** tab (add shapes, materials and lights, drag them around, preview, save a `.pbrt`; see [Build your own scene](#-build-your-own-scene)), a scene picker with live metadata (description, GPU compatibility, perf hint), camera presets, quality/resolution presets, GPU/CPU toggle, image vs. video mode with camera-path selection, a render queue, and one-click render. On macOS, **Live Preview** renders continuously on the Metal GPU so you can drag to orbit the camera (see [`docs/MAC_LIVE_PREVIEW.md`](docs/MAC_LIVE_PREVIEW.md)). The interface is translated into English, Spanish, French, Japanese and Simplified Chinese (Language menu, applied on restart) and has selectable themes and fonts.

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

## 🚀 Running it

```cmd
ray_tracer.exe [--cpu|--gpu] [--output PATH] [width] [samples] [max_depth] [scene] [cam_x cam_y cam_z]
```

`scene` is a scene's name (`cornell-box`), its id (`A1`), or the path of a `.pbrt` file. Every argument is optional (the defaults are a 600 px picture of the Cornell box at 500 samples and depth 20), and `ray_tracer.exe --help` lists every option. The GUI (`RayTracerGUI.exe`) does the same with a scene picker and settings tabs.

```cmd
ray_tracer.exe --gpu --output out.png 800 1000 20 cornell-box        :: OptiX GPU
ray_tracer.exe --cpu --output out.png 600 100 15 stanford-xyzrgb-dragon
ray_tracer.exe --cpu --output out.png 800 64 8 my-scene.pbrt          :: a scene file of your own
ray_tracer.exe --sppm --output out.png 600 300 10 cornell-rough-glass :: photon mapping (add --gpu for GPU SPPM)
ray_tracer.exe --video --frames 60 --fps 30 --camera-path orbit 600 100 50   :: MP4 via ffmpeg
```

Video: see [docs/VIDEO_GENERATION.md](docs/VIDEO_GENERATION.md). On macOS the command is `./build/ray_tracer`, and `--gpu` is the Metal backend (if built with `-DRT_BUILD_METAL=ON`).

**Output:** a `.ppm` (raw) and a `.png` (lossless) next to each other, in the `output/` folder beside the executable unless `--output` says otherwise. Give `--output` a `.exr` path instead (either backend) to get a linear, full-float-precision HDR EXR with no tonemapping or quantisation, useful for compositing. Combine it with `--denoise` to also get `<name>_albedo.exr` and `<name>_normal.exr` AOV files alongside the beauty image (GPU recursive backend only, reusing the denoiser's own guide-layer buffers).

## 🖼️ Scenes

151 built-in scenes and 58 test scenes, each with a stable name (`cornell-box`) and a
short id (a category letter plus a number: `A1`, `B10`, `G25`), selected via the
CLI's scene argument (either works, or the path of a `.pbrt` file) or the GUI's
scene dropdown. Categories: **A** Basics (the book progression), **B** Materials,
**C** Lights, **D** Cameras, **E** Volumes, **F** Geometry, **G** Models
(real-world statue/object meshes - Stanford Bunny, Armadillo, Sponza, Bistro,
San Miguel, and dozens more), **H** Large Scenes ("movie-level" fully
textured environments), **I** Education (curated demos of specific
render-option controls), **J** Textures (texture-system demos), **K** Custom
Scenes (your own `.pbrt` files and downloaded collections, found live in
`pbrt_scenes/` with no rebuild; see [`pbrt_scenes/README.md`](pbrt_scenes/README.md))
and **L** Test Scenes (the closed-form and regression scenes this project's tests
render). Every scene renders on the CPU renderer; the GUI's scene info shows
which ones the GPU backends also support - see
[`docs/SCENE_SELECTION.md`](docs/SCENE_SELECTION.md) for names and ids, the
header tags a scene file can carry, what "performance" means, and how to add a
scene, and
[`src/TheRestOfYourLife/scene_registry_data.h`](src/TheRestOfYourLife/scene_registry_data.h)
for the authoritative per-scene table (description, performance,
recommended SPP, camera defaults).

Scenes that import external mesh/texture assets (mostly categories **G** and **H**) load them from `models/` (Git LFS for the large ones), and the GUI can download the ones that are missing; see [Scenes that need files](docs/SCENE_SELECTION.md#scenes-that-need-files).

## 📁 Project Structure and Releases

`src/TheRestOfYourLife/` is the primary CPU path tracer (materials, lights, cameras, the pbrt-v4 scene builder, the BDPT/MLT/SPPM integrators), `src/shared/` the CPU/GPU-shared code, `gpu/optix/` and `gpu/metal/` the GPU backends, `launcher/` the `ray_tracer` command line, `qt_gui/` the GUI, `pbrt_scenes/` the scene files, and `tests/` the Google Test suite. The full tree is in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md), and the guides are listed in [docs/README.md](docs/README.md). Making a release package: [releases/README.md](releases/README.md) and [scripts/README.md](scripts/README.md).

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

## 🔬 How it renders

1. **Ray generation:** rays from the camera through each pixel (pinhole, thin-lens, orthographic, spherical or realistic-lens model).
2. **Intersection:** BVH-accelerated traversal of the scene (an SAH BVH on the CPU, OptiX's acceleration structures on the GPU).
3. **Shading:** the material's BxDF is evaluated at the hit, with next-event estimation and multiple importance sampling against the scene's lights.
4. **Bouncing:** scattered rays are traced up to the maximum depth, with Russian roulette on the CPU.
5. **Accumulation:** the samples of a pixel are averaged, then tone mapped (ACES filmic) and sRGB encoded.

SPPM, BDPT and MLT are alternative integrators for hard light transport (see the Features list). Random numbers: a thread-local PCG-family generator with stratified/Sobol low-discrepancy sampling on the CPU, a per-pixel, per-bounce seeded PCG hash on the GPU. The per-feature, per-backend detail is in [docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) and [docs/PBRT_SUPPORT.md](docs/PBRT_SUPPORT.md).

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

- **GPU SPPM is scene-limited**: it handles spheres and quads, quad/sphere area lights, a uniform sky and textured diffuse surfaces; anything else (meshes, instances, point lights, an image sky, ...) is rejected with a reason, so use `--cpu --sppm` instead. Where it runs it agrees with CPU SPPM to within about 2% on the scenes tested and matches closed forms (a furnace, a textured sphere under a sky). Its photon gather radius is a fixed 5 units (the CPU scales it to the scene).
- **SPPM has no volume model** (neither has pbrt-v4's): fog absorbs correctly but scattering inside it reads too dark, and the CPU warns for scenes with a medium. SPPM also emits photons from area lights only, so sky-lit scenes read 2-3% low against the path tracer.
- **BDPT and MLT are CPU-only**: selectable via `--bdpt`/`--mlt`, with no GPU (OptiX or Metal) implementation (`--gpu` is ignored with a warning). They are checked against the path tracer and against closed forms on many purpose-built scenes (furnaces, glass, rough conductors, disk/cylinder/cone/paraboloid lights, participating media: see `tests/unit/cpu_integrator_agreement_tests.cpp`) and agree to within about 1-2%. Portal lights and camera media are not supported by them (they warn).
- **Hair/fur has two different fidelity levels**: scene F4 (Curve Fibers) uses real Bezier curve/strand geometry (`CurveShape`, exact ray-curve intersection on CPU, tessellated bilinear-patch tubes on GPU); the older scene B11 instead applies the Marschner/Chiang BxDF math via a shading-normal proxy on sphere primitives, not actual fiber geometry.
- **The GPU wavefront path tracer is opt-in**: enabled via `--wavefront`. It is a spectral renderer (four hero wavelengths), so strongly chromatic media read a few percent off the per-channel closed form; the test suite compares it with the CPU and with the default recursive GPU backend, which remains the primary GPU path.
- **GPU/OptiX rendering is Windows+NVIDIA only, with no fallback**: CUDA/OptiX isn't available on macOS at all (Apple dropped NVIDIA GPU support; Apple Silicon has no CUDA), so that specific backend can't be ported there. macOS instead has its own separate Metal GPU backend (`gpu/metal/`, opt-in via `-DRT_BUILD_METAL=ON`) alongside the CPU renderer/CLI/Qt GUI (see [BUILD.md](BUILD.md#macos-cpu-and-metal-gpu)). It matches the CPU renderer on nearly every scene but is not at full parity — `docs/METAL_PARITY_STATUS.md` lists the remaining differences (for example one noise-limited rough-glass scene, and a few integrator-option gaps).
- **Scenes that need external files** (the Models and Large Scenes) stop with a "file not found" error if a mesh is missing, and the GUI can download them; see [Scenes that need files](docs/SCENE_SELECTION.md#scenes-that-need-files).

### Planned / possible future work

- [ ] GPU implementation of BDPT/MLT
- [ ] Broader GPU SPPM scene support
- [ ] Real curve/strand geometry for scene B11's hair fibers (matching scene F4's approach)
- [x] Native Apple Silicon (arm64) GUI build — `scripts/build_and_deploy_macos.sh` builds native arm64 by default and `--arch universal` builds arm64 + x86_64
- [ ] Linux support (likely a small extension of the same CMake/POSIX groundwork the macOS port added)

## 🤝 Contributing

Contributions are welcome. **[CONTRIBUTING.md](CONTRIBUTING.md)** says how to build and test, what a good change looks like (a closed form or a cross-backend check, and a test that fails without the fix), and lists small, self-contained things that need doing. A bug report with a scene that shows the problem is as useful as a patch. Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## 📝 License

The original code in this repository is released under the **[MIT licence](LICENSE)**.

It also contains code that comes from other projects, which keeps its own licence: ported and adapted code from **pbrt-v4** (Apache-2.0; many files under `src/shared/` carry the attribution header), the first scenes and classes of the "Ray Tracing in One Weekend" series (CC0), NanoVDB (Apache-2.0), tinyexr (BSD-3-Clause), stb (public domain) and miniz (public domain). Meshes, textures and scenes keep the licences of their sources and are **not** covered by the MIT licence. Everything, with the exact terms, is listed in **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**; see also the [Mesh & Texture Credits](#mesh--texture-credits) section above.

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

The OptiX GPU backend is described in [gpu/optix/README.md](gpu/optix/README.md); the version history is [CHANGELOG.md](CHANGELOG.md).
