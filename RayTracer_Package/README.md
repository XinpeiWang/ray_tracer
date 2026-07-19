# Ray Tracer - Path Tracing Renderer

A high-performance CPU/GPU path tracing renderer with a modern Qt6 GUI.

![Version](https://img.shields.io/badge/version-1.6-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

## Features

- 🎨 **Modern Gaming-Style GUI** - Intuitive Qt6 interface with dark theme
- 🚀 **Dual Rendering Modes** - Choose between CPU or GPU (CUDA) rendering
- 💎 **6 Quality Presets** - From Draft (25 samples) to Maximum (5,000 samples)
- 📐 **15 Resolution Options** - From 100x100 to 4K (4096x4096)
- ⏸️ **Stop Render Control** - Cancel long-running renders at any time
- 📊 **Real-time Progress** - Watch your render progress with scanline tracking
- 🖼️ **Auto-open Results** - Rendered images open automatically when complete
- 🎯 **Cornell Box Scene** - Classic test scene with realistic lighting

## Quick Start

### Option 1: Double-click GUI (Recommended)
1. Navigate to the `RayTracer_Package` folder
2. Double-click `RayTracerGUI.exe`
3. Choose your settings and click "Start Render"
4. Your rendered image will open automatically when complete!

### Option 2: Command Line
```bash
cd RayTracer_Package
ray_tracer.exe --gpu 800 100 50 --output my_render
```

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU** (optional): NVIDIA GPU with CUDA support for GPU mode
- **Disk Space**: ~50MB

## Quality Presets

| Preset | Samples | Max Depth | Render Time | Best For |
|--------|---------|-----------|-------------|----------|
| ⚡ Draft | 25 | 10 | Very Fast | Quick tests |
| 🚀 Preview | 50 | 20 | Fast | Composition preview |
| 📷 Good | 100 | 50 | Moderate | Standard renders |
| 💎 High | 500 | 50 | Slow | High quality |
| ✨ Ultra | 1,000 | 100 | Very Slow | Production quality |
| 🌟 Maximum | 5,000 | 100 | Extreme | Photorealistic final renders |

## Resolution Guide

- **100-400**: Quick tests (seconds)
- **512-800**: Standard renders (1-5 minutes)
- **1024-1080**: HD quality (5-15 minutes)
- **2048 (2K)**: Professional (20+ minutes)
- **3840-4096 (4K)**: Ultra HD (1+ hour)

*Times based on Good preset with CPU mode*

## GUI Usage

### Basic Tab
1. **Mode**: Choose CPU (slower, but works everywhere) or GPU (faster, needs NVIDIA)
2. **Quality**: Select a preset or choose Custom for manual control
3. **Resolution**: Pick your image size
4. **Output Path**: Where to save your render (default: output folder)
5. **Start Render**: Begin rendering!
6. **Stop Render**: Cancel anytime if it's taking too long

### Advanced Tab
- Fine-tune Width, Height, Samples per pixel, and Max Ray Depth
- Only active when "Custom" quality is selected

### Tips
- Start with **Preview** preset at **800x800** to test (takes ~30 seconds)
- Use **Stop Render** button if you need to cancel
- Higher samples = less noise but much longer render time
- GPU mode is ~10x faster but requires more samples for similar quality

## Command Line Options

```
ray_tracer.exe [OPTIONS] WIDTH SAMPLES MAX_DEPTH

Options:
  --gpu             Use GPU (CUDA) renderer (default)
  --cpu             Use CPU renderer
  --output PATH     Custom output path (default: output/image.ppm)
  --help            Show help message

Examples:
  ray_tracer.exe --gpu 1024 100 50
  ray_tracer.exe --cpu 800 500 50 --output my_render.ppm
```

## Output Files

The renderer creates two files:
- `image.ppm` - Raw PPM format (intermediate)
- `image.png` - Converted PNG (final, opens automatically)

## Troubleshooting

### "Failed to start renderer"
- Make sure `ray_tracer.exe` is in the same folder as `RayTracerGUI.exe`
- Check that all DLL files are present

### Slow Rendering
- Try a lower quality preset
- Reduce resolution
- Use GPU mode if you have an NVIDIA graphics card

### GPU Mode Not Working
- Requires NVIDIA GPU with CUDA support
- Fall back to CPU mode if GPU fails

### Icon Not Showing
- Windows may cache icons - restart Explorer or reboot

## Technical Details

- **Renderer**: Path tracing with importance sampling
- **Scene**: Cornell box with diffuse, metal, and dielectric materials
- **Output**: PPM → PNG conversion
- **GUI Framework**: Qt 6.11.1
- **Build**: MinGW 64-bit, C++17

## File Structure

```
RayTracer_Package/
├── RayTracerGUI.exe          # Main application (GUI)
├── ray_tracer.exe            # Renderer backend
├── Qt6*.dll                  # Qt runtime libraries
├── cuda*.dll                 # CUDA runtime (for GPU)
├── platforms/                # Qt platform plugins
├── styles/                   # Qt style plugins
├── imageformats/             # Image format plugins
└── output/                   # Default output folder
```

## Distribution

To share this app with others:

1. **Zip the entire `RayTracer_Package` folder**
2. Share the zip file
3. Recipients extract and run `RayTracerGUI.exe`

No installation needed - fully portable!

## Version History

### v1.6 (Current)
- Added modern Qt6 GUI with gaming-style theme
- Added custom game-like application icon
- Added stop render button for user control
- Expanded to 6 quality presets (including Maximum at 5000 samples)
- Expanded to 15 resolution options (100x100 to 4K)
- Added auto-open rendered images
- Improved progress tracking with real scanline parsing
- Fixed CPU renderer output path handling

### v1.5
- Added dual CPU/GPU rendering modes
- Implemented Cornell box scene
- Added command line interface

## Credits

Built using:
- Ray Tracing in One Weekend book series
- Qt 6.11.1 framework
- NVIDIA CUDA for GPU acceleration

## License

Free to use and share for educational and personal purposes.

## Support

For issues or questions, please open an issue on GitHub:
https://github.com/XinpeiWang/ray_tracer

---

**Enjoy ray tracing!** 🌟
