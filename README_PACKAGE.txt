================================================================================
					RAY TRACER - Cornell Box Renderer
						 GPU & CPU Path Tracer
================================================================================

DESCRIPTION
-----------
A high-performance path tracer that renders the classic Cornell box scene
with a glass sphere. Supports both GPU (CUDA) and CPU rendering modes with
automatic hardware detection.

SYSTEM REQUIREMENTS
-------------------
Minimum (CPU Mode):
  - Windows 10/11 (64-bit)
  - 4 GB RAM
  - Visual C++ 2015-2026 Redistributable (included)
  - Any modern CPU

Recommended (GPU Mode):
  - Windows 10/11 (64-bit)
  - NVIDIA GPU with CUDA Compute Capability 3.0+ (GTX 600 series or newer)
  - 8 GB RAM
  - CUDA Runtime (included)
  - Visual C++ 2015-2026 Redistributable (included)

INSTALLATION
------------
1. Extract the ZIP file to any folder
2. No installation needed - the application is portable!

QUICK START
-----------
Double-click "RayTracer.exe" to launch in interactive mode.

The application will:
  1. Auto-detect if you have a CUDA-capable GPU
  2. Show current rendering settings
  3. Prompt you to press ENTER or customize settings

Output images are saved to the "output" folder as "image.ppm"

COMMAND-LINE USAGE
------------------
For advanced users and batch processing:

  RayTracer.exe [--cpu|--gpu] [width] [samples_per_pixel] [max_depth]

Examples:
  RayTracer.exe                    # Interactive mode
  RayTracer.exe --gpu 800 1000 20  # GPU, 800x800, 1000 samples, depth 20
  RayTracer.exe --cpu 600 100 15   # CPU, 600x600, 100 samples, depth 15

Parameters:
  --cpu          Force CPU rendering
  --gpu          Force GPU rendering (default if GPU detected)
  width          Image resolution (square, default: 600)
  samples        Samples per pixel (default: 500)
  max_depth      Maximum ray bounces (default: 20)
  --help         Show help message

RENDERING QUALITY GUIDE
-----------------------
GPU Mode (Naive Path Tracing):
  - Fast preview:     100-500 samples
  - Good quality:     1000-2000 samples
  - High quality:     5000-10000 samples
  - Render time:      ~10-60 seconds for 600x600

CPU Mode (Importance Sampling):
  - Fast preview:     10-50 samples
  - Good quality:     100-500 samples
  - High quality:     1000-5000 samples
  - Render time:      ~1-10 minutes for 600x600 (depends on CPU cores)

Note: GPU requires more samples than CPU due to different rendering algorithms

OUTPUT FORMAT
-------------
Images are saved in PPM (Portable Pixmap) format. To view:
  - Windows: Use IrfanView, GIMP, or Photoshop
  - Convert to PNG: Use ImageMagick or online converters
  - Many image viewers support PPM natively

TROUBLESHOOTING
---------------
Problem: "No CUDA GPU detected" but I have an NVIDIA GPU
Solution: 
  - Ensure your GPU is CUDA-capable (GTX 600 series or newer)
  - Update NVIDIA drivers to the latest version
  - Check if CUDA Runtime is working: nvidia-smi command
  - The application will automatically use CPU mode

Problem: Application won't start - missing DLL error
Solution:
  - Install Visual C++ Redistributable 2015-2026 from Microsoft
  - Included redistributable may not match your system
  - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

Problem: Render is too dark/noisy
Solution:
  - Increase samples per pixel (try 2000+ for GPU, 500+ for CPU)
  - Increase max depth if scene has lots of reflections
  - Wait for render to complete - progressive rendering not shown

Problem: Out of memory error
Solution:
  - Reduce image resolution (try 400 or 300)
  - Close other GPU-intensive applications
  - Use CPU mode if GPU has limited VRAM

SCENE DESCRIPTION
-----------------
The rendered scene is the classic Cornell Box:
  - Two boxes (one white diffuse, one glass sphere)
  - Red wall (left), Green wall (right)
  - White ceiling, floor, and back wall
  - Ceiling light (bright white emission)

The glass sphere demonstrates:
  - Fresnel reflection/refraction
  - Caustics (bright light patterns)
  - Total internal reflection

TECHNICAL INFO
--------------
Rendering Algorithms:
  - GPU: Naive path tracing (random sampling)
  - CPU: Importance sampling (direct light sampling)

This is why GPU needs more samples - it must randomly find the light,
while CPU explicitly samples the light source for efficiency.

Camera Settings:
  - Field of view: 40 degrees
  - Aspect ratio: 1:1 (square)
  - Position: Looking into Cornell box from front
  - No depth of field (infinite focus)

ABOUT
-----
Cornell Box Path Tracer
Version 1.0
Created as a demonstration of GPU and CPU path tracing techniques

Based on:
  - "Ray Tracing in One Weekend" series by Peter Shirley
  - Cornell Box scene (Cornell University)

LICENSE & USAGE
---------------
This software is provided "as-is" for educational and personal use.
Feel free to render, distribute images, and learn from the implementation.

CONTACT & RESOURCES
-------------------
For bug reports, suggestions, or questions visit:
  https://github.com/XinpeiWang/ray_tracer

================================================================================
					  Thank you for using Ray Tracer!
					Enjoy creating beautiful renders!
================================================================================
