# Ray Tracer - Installation Guide

This guide is for the **portable release package** (no build required). If
you're building from source instead, see [BUILD.md](BUILD.md).

## Quick Start (Windows 10/11)

1. [Download the latest release](../../releases) and extract the ZIP to
   any folder (e.g. `C:\RayTracer` or your Desktop) - nothing is installed
   to your system, so this is the whole setup.
2. **GUI (recommended):** double-click `RayTracerGUI.exe` - pick a scene,
   adjust resolution/samples/depth, and click Render.
   **CLI:** run `RayTracer.exe` from a terminal (see Command-Line Usage
   below) for scripting or automation.
3. Find your rendered image in the `output` folder.

## What You Get

After extraction:
- **RayTracerGUI.exe** - the graphical interface (if the GUI was built into
  this package)
- **RayTracer.exe** - the command-line renderer
- **Qt6\*.dll, platforms/, styles/** - Qt runtime, needed by the GUI
- **cudart64\*.dll** - NVIDIA CUDA runtime, needed for GPU mode
- **vcruntime140\*.dll, msvcp140.dll** - Visual C++ runtime libraries
- **output/** - folder where rendered images are saved

## System Requirements

**Minimum (CPU mode):**
- Windows 10/11 (64-bit)
- 4 GB RAM
- Any modern CPU

**Recommended (GPU mode):**
- Windows 10/11 (64-bit)
- NVIDIA GPU with OptiX support (RTX series recommended)
- 8 GB RAM

## Command-Line Usage

```
RayTracer.exe [--cpu|--gpu] [--output PATH] width spp max_depth SCENE_ID [cam_x cam_y cam_z]
```

Scenes are identified by a category letter + number (e.g. `A1` for the
Cornell Box) - see [docs/SCENE_SELECTION.md](docs/SCENE_SELECTION.md) for
the full id scheme and category list, or just use the GUI's scene dropdown
instead of memorizing ids.

Examples:
```
RayTracer.exe --gpu --output out.png 800 500 20 A1     # Cornell Box, GPU, 500 spp
RayTracer.exe --cpu --output out.png 600 100 15 B10    # A Materials-category scene, CPU
```

Camera position (`cam_x cam_y cam_z`) is optional - each scene has its own
recommended default camera if you omit it.

## Viewing Output Images

The rendered image is saved in two formats:
- **PNG** - lossless, opens in any image viewer or web browser
- **PPM** - raw pixel data, for advanced use (viewable with tools like
  IrfanView)

Give `--output` a `.exr` path instead for a linear full-float HDR EXR
(no tonemapping/quantization) - see the main [README.md](README.md) for
details.

## Troubleshooting

**"Missing DLL" error:**
- All required DLLs should be included in the package
- If you still get errors, install the Visual C++ Redistributable:
  https://aka.ms/vs/17/release/vc_redist.x64.exe

**No GPU / GPU rendering fails:**
- The app falls back to CPU mode automatically for most cases
- Update your NVIDIA drivers if you have a compatible GPU
- Older/non-RTX NVIDIA GPUs may not support the OptiX features this
  renderer needs

**Render takes too long:**
- Reduce samples per pixel (a scene's info panel/description shows its
  recommended SPP)
- Reduce resolution
- GPU mode is generally much faster than CPU for the same sample count

**Antivirus blocks the app:**
- Add an exception for the extracted folder if your antivirus flags it

## Uninstallation

Delete the extracted folder - nothing is installed to your system.

## More Information

See the main [README.md](README.md) for the full feature list, all
available scenes, and build-from-source instructions.
