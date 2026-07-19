# Ray Tracer v1.5 - Clean UI Edition

## Release Date
July 18, 2026

## Overview
This release removes emoji characters from the GUI to ensure consistent, clean text display across all Windows systems. The dark theme and all functionality remain intact.

## Changes from v1.4

### Fixed
- **UI Text Display**: Removed all emoji characters that were causing garbled text on some systems
- **Font Consistency**: Reverted to standard Segoe UI font for reliable cross-system compatibility
- **Resource Encoding**: Cleaned up resource file to use only ASCII-safe characters

### UI Text Updates
- Title: "RAY TRACER" (clean, professional)
- Renderer options: "GPU (CUDA) - Recommended" / "CPU (Multi-threaded)"
- Action button: "RENDER"
- Tips and descriptions: Plain text without emoji decorations

### Retained Features
- ✓ Dark theme with modern styling
- ✓ Basic/Advanced tabbed interface
- ✓ Real-time progress bar
- ✓ Render time display
- ✓ PNG output generation
- ✓ Adaptive GPU fallback for compatibility
- ✓ Multi-threaded CPU rendering
- ✓ Quality presets (Low, Medium, High, Very High, Ultra)
- ✓ Square resolution options (512×512 to 4K)

## Package Contents
- `RayTracerGUI.exe` - Main GUI launcher with dark theme
- `ray_tracer.exe` - Console version for command-line use
- `cudart64_110.dll` - CUDA runtime (required for GPU rendering)
- `README.md` - Quick start guide
- `Output/` - Folder for rendered images

## System Requirements
- Windows 10/11 (64-bit)
- CPU: Multi-core processor recommended
- GPU: NVIDIA GPU with CUDA support (optional, for GPU mode)
- RAM: 4GB minimum, 8GB+ recommended for high-resolution renders
- Display: Any resolution (1920×1080 or higher recommended)

## Usage
1. Extract the ZIP file to any location
2. Run `RayTracerGUI.exe`
3. Select rendering engine (GPU or CPU)
4. Choose quality preset or customize settings
5. Click "RENDER"
6. Find your image in the Output folder

## Known Improvements
- Clean, readable text on all Windows systems
- No encoding or font-related display issues
- Professional appearance suitable for distribution

## Technical Notes
- GUI uses standard Win32 dialog controls with Segoe UI font
- All text is set programmatically using Unicode-safe APIs
- Resource file contains only ASCII characters for maximum compatibility

---

For issues or questions, visit: https://github.com/XinpeiWang/ray_tracer
