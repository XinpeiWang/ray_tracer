Ray Tracer - {TIER} package
============================

This is the {TIER} distribution package. See INSTALL.md / README.md in the
main repository for full documentation; this file covers just what you need
to run this specific package.

Getting started
----------------
Double-click launcher.bat, or run RayTracerGUI.exe directly if this package
includes one.

If Windows blocks the app with a "protected your PC" SmartScreen warning
(expected for an unsigned executable from a small project - this is not a
sign of anything wrong), click "More info" then "Run anyway".

What's in this package
------------------------
- Lite package: RayTracer.exe (command-line renderer) only, CPU rendering.
  Run "RayTracer.exe --help" for usage. No GPU, no GUI.
- Medium package: adds RayTracerGUI.exe (the graphical interface) and the
  Qt runtime it needs. Still CPU rendering only - no GPU acceleration or
  Live Preview.
- Full package: adds GPU rendering (OptiX/CUDA) and the interactive Live
  Preview feature, on top of everything Medium has.

GPU requirements (Full package only)
--------------------------------------
GPU rendering needs a current NVIDIA driver and an RTX-capable GPU. The
CUDA runtime itself is bundled INTO the executables (no separate CUDA
Toolkit install needed on your machine) - only the driver matters. If no
compatible GPU/driver is found, the app automatically falls back to CPU
rendering, so it will still run correctly on a machine without an NVIDIA
GPU at all, just without GPU acceleration or Live Preview.

Troubleshooting
-----------------
- App won't start at all: install the Visual C++ Redistributable from
  https://aka.ms/vs/17/release/vc_redist.x64.exe
- GPU mode unavailable: update your NVIDIA driver, or use CPU mode (the
  Renderer dropdown in Settings, or "--cpu" on the command line).
