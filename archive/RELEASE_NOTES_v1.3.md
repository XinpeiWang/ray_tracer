# Ray Tracer v1.3 - Enhanced GUI with Basic/Advanced Modes

## 🎉 What's New in v1.3

### Major Features
- **🎛️ Basic/Advanced Tab Interface**: Choose your experience level!
  - **Basic Mode**: Simple quality presets for beginners
  - **Advanced Mode**: Full manual control for experts

- **✨ Quality Presets** (Basic Mode):
  - **Low** - Fast preview (400×400, 50 samples) - Perfect for testing
  - **Medium** - Balanced quality (600×600, 200 samples) - Good compromise
  - **High** - Recommended (800×800, 500 samples) - Great quality ⭐
  - **Very High** - Excellent (1080×1080, 1000 samples) - Takes time
  - **Extreme** - Production (2048×2048, 2000 samples) - Best quality

- **📊 Visual Progress Bar**: See rendering progress in real-time

### Bug Fixes & Improvements
- ✅ Fixed Unicode display issue in resolution dropdown
- ✅ All resolution options now properly square (400-2048)
- ✅ Added 2K Ultra resolution option (2048×2048)
- ✅ Improved dialog layout and spacing
- ✅ Better user guidance with quality descriptions

## 📦 Download & Use

### Quick Start (Beginners)
1. Extract the ZIP file
2. Double-click `RayTracerGUI.exe`
3. Select a quality preset from the **Basic** tab
4. Click **RENDER**
5. Find your image in the `output` folder as PNG and PPM

### Advanced Users
1. Switch to the **Advanced** tab
2. Fine-tune resolution, samples, and ray depth
3. Balance quality vs. render time
4. GPU mode recommended for high samples

## 🖼️ Output Formats
- **PNG** - Universal format, opens in any image viewer
- **PPM** - Raw format for technical analysis

## 📋 System Requirements
- Windows 10/11 (64-bit)
- Optional: NVIDIA GPU with CUDA support for faster rendering
- CPU mode works on any system (multi-threaded)

## 🎨 What This Renders
Classic Cornell Box scene:
- Path-traced global illumination
- Realistic materials (metal, glass, diffuse)
- Physically-based lighting
- Perfect for learning ray tracing!

## 📝 Full Changelog (v1.0 → v1.3)

### v1.3 (Current)
- Added Basic/Advanced tab interface
- Added 5 quality presets for easy use
- Fixed Unicode string display bugs
- All resolutions now square format
- Added 2K resolution option
- Added visual progress bar

### v1.2
- Added automatic PNG generation
- Image writer integration (stb_image_write)
- No manual conversion needed

### v1.1
- Added GUI launcher with icon
- Windows Forms-style interface
- One-click rendering

### v1.0
- Initial portable release
- Console launcher
- GPU/CPU rendering modes

## 🚀 How to Share
Just share the entire `RayTracer_Package` folder or the ZIP file. No installation needed!

## 🐛 Troubleshooting

**GUI shows random characters**: Fixed in v1.3!

**GPU not detected**: The app will automatically use CPU mode. Check if you have NVIDIA drivers installed.

**Render is very slow**: 
- Use **Basic** tab and select **Low** or **Medium** preset
- Enable GPU mode if available
- Lower resolution produces faster results

---

**Previous versions**: v1.2 (PNG Support), v1.1 (GUI), v1.0 (Portable Console)
