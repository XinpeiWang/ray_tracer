# Build System Guide

This document describes how to build the Ray Tracer project: the full **Windows** build, including the GPU/OptiX renderer (CUDA/OptiX is
Windows+NVIDIA only), and the **macOS** build ([macOS](#macos-cpu-and-metal-gpu) below: the CPU renderer, CLI and Qt GUI, plus the Metal GPU
backend), a separate, purely additive CMake/qmake path that the MSBuild instructions in between do not cover. Linux is not supported yet.

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

- **Running the whole suite.** `scripts\run_tests_parallel.ps1` splits it by what the tests touch. `-Tier Fast` (a couple of minutes) runs everything that does not use the GPU in 16 shards, the loop to use while editing. `-Tier Split` runs Fast and then Slow (the GPU tests, one process) for the whole suite, noticeably faster than one `ray_tracer_tests.exe` process. Slow is deliberately not sharded: separate processes time-slice the GPU, and 2-4 shards measured slower (193-198 s) than one (170 s). Nearly all of Slow is the material parity sweep, where the wavefront backend takes about 125 s because a 60x60 frame is thousands of tiny, synchronised launches; only a change inside that renderer would shrink it. The runner counts failing tests itself (`--gtest_brief` hides gtest's own failure summary) and removes its scratch directory when everything passes.
- **Gallery scenes.** The parity and light-count suites skip every scene flagged `requires_files` (the model gallery, the downloaded pbrt collections), so for a long time nothing rendered them. `GalleryGpuRenderTest` (`tests/integration/gallery_scenes_render_test.cpp`, in the Slow tier) now renders each one that is on disk at 32x32 on the recursive GPU backend and fails on a failed or all-black render: 41 scenes, about 60 s. It skips the 12 Large Scenes (Power Plant, San Miguel, Rungholt, ...: 89 s, mostly parsing gigabyte OBJ files) unless `RT_GALLERY_ALL=1`, adds a CPU render and a CPU-vs-GPU brightness check with `RT_GALLERY_CPU=1`, and is skipped entirely by `RT_SKIP_GALLERY=1`. The GPU render alone is safe in one process (the OptiX scene builder keeps only the two most recent pbrt scenes; the whole Slow tier peaked at 23 GB before that and 8.5 GB now, `RT_GALLERY_ALL=1` at 10.5 GB). Do not combine `RT_GALLERY_ALL=1` and `RT_GALLERY_CPU=1` in one process: the CPU and GPU copies of the multi-million-triangle scenes accumulate (16 GB private memory after six of them) and renders start failing with "out of memory". `scripts\run_gallery_isolated.ps1 -Cpu` runs every gallery scene in its own process instead (62 scenes, about 15 minutes; without `-Cpu` about 4; `-SkipLarge` and `-Filter` narrow it); a run on 2026-10-06 passed 61 of the 62 on both backends, Large Scenes included, and the last (sportscar-area-lights) once both backends rendered the same number of samples.
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

**What GitHub CI runs** (`.github/workflows/unit-tests.yml`, a hosted runner with no GPU, CUDA or OptiX): the portable CMake target `unit_tests` (BVH, materials, cameras, sampling, BDPT/SPPM math, volumetrics, pbrt loading) and, since the CPU renderer itself needs no SDK, `cpu_integrator_tests` (`tests/unit/cpu_integrator_agreement_tests.cpp`): the CPU path tracer, BDPT, MLT, SPPM and `--simplepath`/`--randomwalk` rendered on small pbrt scenes and compared with each other and with closed forms - the tests that pinned the BDPT/MLT/SPPM/diffuse-transmission defects. Both are built by `cmake -B build -A x64` + `cmake --build build --config Release` in `tests/` and run from the repository root. Everything that renders on an OptiX backend (`pbrt_example_scenes_tests.cpp`, the CPU-vs-GPU parity sweeps, the gallery) runs only here, with `scripts\run_tests_parallel.ps1 -Tier Split`; the GPU is never exercised in CI.

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

qt_gui\release\
  └─ RayTracerGUI.exe            # Qt GUI (if built)
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
  └─ [Qt plugins in subdirectories]
```

**Note:** Backend executables and shaders are now automatically copied to `RayTracer_Package/` by MSBuild post-build events. You only need to run `scripts\deploy_qt_gui.ps1` to add Qt dependencies after building the GUI.

## Deployment

### Automatic Deployment (Recommended)

The easiest way to create a complete, ready-to-run package:

```powershell
.\scripts\build_and_deploy.ps1
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

**This package can be zipped and distributed without requiring Qt installation on target machines.**

## macOS (CPU, and Metal GPU)

No CUDA/OptiX support (the OptiX renderer is CUDA-only) — this builds the CPU path
tracer, the `ray_tracer` CLI, and (optionally) the Qt GUI, purely additive
alongside the Windows MSBuild solution. A plain `cmake` build is CPU-only;
see [Metal GPU (opt-in)](#metal-gpu-opt-in) just below for the Metal GPU
backend (always enabled in the one-command `.app`/`.dmg` build).

**CLI + CPU renderer**, via the root `CMakeLists.txt`:
```bash
cmake -B build && cmake --build build
./build/ray_tracer 800 100 50 A1   # width, spp, max_depth, scene_id
```
Produces `cpu_renderer` (static lib), `ray_tracer` (CLI - `--gpu` falls
back to a warning on this default build; pass `-DRT_BUILD_METAL=ON` for
a real macOS `--gpu` path instead, see below), and `scene_metadata.dylib`.

**Qt GUI**, via `qt_gui/RayTracerGUI.pro` (Qt 6, same as Windows):
```bash
cd qt_gui
qmake && make
```
Produces `RayTracerGUI.app`. Copy the `ray_tracer` binary and
`scene_metadata.dylib` built above into `RayTracerGUI.app/Contents/MacOS/`
(that exact path — it's where `QCoreApplication::applicationDirPath()`
resolves for a bundled Mac app, which is what both the GUI's subprocess
working directory and `scene_metadata_client.cpp`'s `dlopen()` call use to
find them). On a build without `RT_BUILD_METAL` the GUI's Renderer
dropdown only offers CPU and Live Preview is greyed out; with it, the
dropdown also offers **GPU (Metal)**.

**One-command build + `.app` + `.dmg`**, via `scripts/build_and_deploy_macos.sh`
(does all of the above, then runs `macdeployqt` to bundle Qt's frameworks and
produce a distributable disk image):
```bash
./scripts/build_and_deploy_macos.sh                        # this Mac's own CPU (arm64 on Apple silicon)
./scripts/build_and_deploy_macos.sh --arch universal       # arm64 + x86_64 in one dmg (about twice the build time)
# ./scripts/build_and_deploy_macos.sh --skip-dmg            # .app only, no .dmg
```
The script builds for the architecture(s) you ask for and checks that your Qt install has them (an official Qt 6 install is universal, so `--arch universal` works out of the box). A plain `cmake -B build` also builds for the Mac's real CPU even when `cmake` itself is an Intel binary running under Rosetta; pass `-DCMAKE_OSX_ARCHITECTURES=x86_64` to force an Intel build.
Output lands in `RayTracer_Package_macOS/` (`RayTracerGUI.app` and
`RayTracerGUI.dmg`). Pushing a `v*` tag (or running the "Release (macOS)" workflow by hand) builds the universal dmg on a GitHub runner, smoke-tests the packaged app headless (`scripts/gui_selftest.py`) and attaches the dmg to the release (`.github/workflows/release-macos.yml`); the Windows package still has to be built on a machine with CUDA and the OptiX SDK. **Scenes that need external mesh/texture files
(`requires_files=true` in `scene_registry.h` — Sponza, Bistro, every
"Large Scene", most single-model scenes) are deliberately NOT bundled into
the `.app`/`.dmg`**: many are hundreds of MB to 1GB+, and a few (Power
Plant) carry non-commercial-only licenses that make redistributing them in
an installer questionable regardless of size. Every scene that doesn't
require external files (Basics/Materials/Lights/Cameras/Volumes/Geometry/Textures —
most of the registry, all procedurally generated) works from the installed
app with no extra setup, as do the bundled `pbrt_scenes/` examples (category
K) that don't reference a missing mesh/texture. For the external-asset scenes, select the scene and press **Download missing files** (see above); you can also copy files into the installed app's `Contents/MacOS/models/` by hand.

The `.app`/`.dmg` are **not code-signed or notarized** (that needs an Apple
Developer account this project doesn't have) — macOS Gatekeeper will refuse
to open it with a plain double-click on first launch. Right-click the app →
**Open** (or System Settings → Privacy & Security → **Open Anyway**) once to
run it; this is a one-time step per machine, standard for any indie/unsigned
Mac app.

### Metal GPU (opt-in)

A real Metal GPU backend (`gpu/metal/`), separate from the
CPU-only build above - opt in with `-DRT_BUILD_METAL=ON`:
```bash
cmake -B build -DRT_BUILD_METAL=ON && cmake --build build
./build/ray_tracer 800 100 50 A1 --gpu   # real macOS GPU path, not a fallback warning
```
Also builds a standalone `metal_poc` CLI, the `realtime_renderer.dylib`
that backs the GUI's Live Preview, and (via `ctest`, once configured this
way) twelve regression tests: host-side math, real on-device shader
kernels, and a CPU-vs-Metal parity harness that renders every scene on
both and compares them
(`METAL_PARITY_STRICT=1 ctest`; `scripts/update_metal_golden.sh` refreshes the
golden snapshot in `gpu/metal/parity_golden.txt` after an intentional
change). Metal renders the great majority of scenes to within noise of the
CPU renderer; `docs/METAL_PARITY_STATUS.md` lists exactly which scenes and
features still differ, and `docs/history/METAL_GPU_FEASIBILITY.md` has the full
incremental history. Live Preview (interactive drag-to-orbit at a small
fixed size) is described in `docs/MAC_LIVE_PREVIEW.md`. CI builds and runs this on every push (`.github/workflows/
unit-tests.yml`'s own `metal-poc` job, `macos-14`) - the device-
dependent tests gracefully skip there (GitHub's own hosted runners don't
currently expose hardware-raytracing-capable Metal), so full local
verification on real Apple Silicon hardware remains the authoritative
check.

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
