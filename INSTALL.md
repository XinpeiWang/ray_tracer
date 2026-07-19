# Ray Tracer - Installation Guide

## Quick Start (Windows 10/11)

### Option 1: Simple Double-Click Launch
1. Download `RayTracer_v1.0_Portable.zip`
2. Extract to any folder (e.g., `C:\RayTracer` or your Desktop)
3. Double-click `launcher.bat`
4. Follow the on-screen prompts
5. Find your rendered image in the `output` folder!

### Option 2: Direct Executable
1. Extract the ZIP as above
2. Double-click `RayTracer.exe` directly
3. Press ENTER to use default settings or type `custom` to change
4. Wait for rendering to complete
5. Check `output\image.ppm` for your rendered image

## What You Get

After extraction, you'll have:
- **RayTracer.exe** - Main application
- **launcher.bat** - Easy launcher with status messages
- **README.txt** - Full documentation and troubleshooting
- **output/** - Folder where images are saved
- **cudart64_13.dll** - NVIDIA CUDA runtime (for GPU mode)
- **vcruntime140*.dll, msvcp140.dll** - Visual C++ runtime libraries

## System Requirements

**Minimum (CPU Mode):**
- Windows 10/11 (64-bit)
- 4 GB RAM
- Any modern CPU

**Recommended (GPU Mode):**
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (GTX 600 series or newer)
- 8 GB RAM

## Command-Line Usage

For advanced users:
```
RayTracer.exe [--cpu|--gpu] [width] [samples] [depth]
```

Examples:
```
RayTracer.exe                    # Interactive mode
RayTracer.exe --gpu 800 1000 20  # GPU, 800x800, 1000 samples
RayTracer.exe --cpu 600 100 15   # CPU, 600x600, 100 samples
```

## Viewing Output Images

The rendered image is automatically saved in two formats in the `output\` folder:
- **image.png** - PNG format (lossless, widely supported) - Open with any image viewer
- **image.ppm** - PPM format (raw pixel data, for advanced use)

**Opening the image:**
- Windows Photos (built-in) opens PNG files  automatically
- Any modern image viewer or web browser can display PNG files
- The PPM file can be viewed with specialized tools like IrfanView

No conversion needed! The app automatically generates PNG files for easy viewing.

## Troubleshooting

**"Missing DLL" error:**
- All required DLLs should be included
- If you still get errors, install Visual C++ Redistributable:
  https://aka.ms/vs/17/release/vc_redist.x64.exe

**"No CUDA GPU detected" warning:**
- The app will automatically use CPU mode instead
- Update your NVIDIA drivers if you have a GPU
- Some older NVIDIA GPUs don't support CUDA

**Render takes too long:**
- Reduce samples (try 100 for GPU, 50 for CPU)
- Reduce resolution (try 400x400)
- Close other programs to free up resources

**Antivirus blocks the app:**
- The app is safe - it's just rendering math!
- Add an exception for the folder in your antivirus

## Rendering Quality Tips

**GPU Mode (faster but needs more samples):**
- Quick preview: 100-500 samples (~5-15 seconds)
- Good quality: 1000-2000 samples (~20-40 seconds)
- High quality: 5000+ samples (~1-3 minutes)

**CPU Mode (slower but more efficient sampling):**
- Quick preview: 10-50 samples (~10-30 seconds)
- Good quality: 100-500 samples (~1-5 minutes)
- High quality: 1000+ samples (~5-15 minutes)

## Uninstallation

Simply delete the extracted folder - nothing is installed to your system!

## Need Help?

See the full `README.txt` file for:
- Detailed feature descriptions
- Scene information
- Technical details
- More troubleshooting tips

## About

Cornell Box Path Tracer v1.0
A demonstration of GPU and CPU path tracing

Based on "Ray Tracing in One Weekend" series by Peter Shirley

---
**Enjoy creating beautiful renders!**
