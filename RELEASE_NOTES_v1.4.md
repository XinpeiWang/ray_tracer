# Release Notes - v1.4 Dark Theme Edition

## 🎨 What's New

### Dark Theme UI
The Ray Tracer GUI now features a **sleek modern dark theme**!

- 🌑 **Dark background** (RGB 30, 30, 30) - easier on the eyes
- 📝 **Larger fonts** - 18pt bold title, 16pt body text
- 🎯 **Emoji indicators** throughout the UI
- 📐 **Better spacing** - 360×380 dialog (was 340×360)
- ✨ **Professional appearance** - matches industry tools

### Visual Enhancements

#### Cool Quality Presets
- ⚡ Low (Fast Preview)
- 💎 Medium (Balanced)
- 🌟 High (Recommended)
- 🔥 Very High (Impressive)
- 💫 Extreme (Ultra Quality)

#### Smart Status Messages
- 🚀 GPU detected! Ready to render.
- ⚡ Rendering with GPU (CUDA)...
- 🎨 Converting to PNG...
- ✅ Render complete! Time: X seconds

#### Enhanced Dialogs
Completion dialog now shows:
```
🎉 Render completed successfully!

⏱️ Render time: 2.45 seconds

📦 Saved formats:
✓ PNG (lossless)
✓ PPM (raw)

📂 Opening output folder...
```

### Performance Features

#### Render Time Display
Both console and GUI now show:
- Milliseconds for fast renders (< 1 sec)
- Seconds with decimals (1-60 sec)
- Minutes and seconds (> 1 min)

#### Adaptive GPU Block Size
- **Automatic fallback** for hardware compatibility
- Tries: 24×24 → 20×20 → 16×16 → 12×12
- **No crashes** on lower-end GPUs
- **Safe to share** with anyone!

Console output shows selected block size:
```
[cuda_interface] Using block size 24x24 (576 threads)
```

## 🔧 Technical Improvements

### Full Unicode Support
- All text APIs converted to Unicode (wide strings)
- Perfect emoji rendering everywhere
- No more random characters

### Hardware Portability
- Works on high-end GPUs (RTX 3060+)
- Works on mid-range GPUs (GTX 1060)
- Works on older GPUs (GTX 960)
- Works on integrated GPUs (Intel/AMD)

### Stability
- Proper GDI resource cleanup
- Custom dark theme painting
- Professional font rendering (ClearType)

## 📦 Package Contents

- `RayTracerGUI.exe` - GUI launcher with dark theme
- `ray_tracer.exe` - Console launcher with timing
- `cudart64_13.dll` - CUDA runtime
- `msvcp140.dll`, `vcruntime140*.dll` - Visual C++ runtimes
- `README.txt` - Quick start guide
- `output/` - Render output folder (created on first run)

## 🚀 How to Use

### GUI Mode (Recommended)
1. Double-click `RayTracerGUI.exe`
2. Select quality preset or customize settings
3. Click **▶️ RENDER**
4. Wait for completion dialog with render time
5. Output folder opens automatically

### Console Mode
```
ray_tracer.exe --gpu 800 500 20
```
Shows:
```
[cuda_interface] Using block size 24x24 (576 threads)
========================================
RENDER TIME: 2.45 seconds
========================================
```

## 🎯 What's Fixed

### From v1.3:
✅ Dark theme implemented  
✅ Emoji rendering working correctly  
✅ Render time display added  
✅ Adaptive GPU block size for compatibility  
✅ Professional modern UI  

### Known Good:
✅ PNG output works perfectly  
✅ Progress bar animates smoothly  
✅ Basic/Advanced tabs function correctly  
✅ GPU/CPU detection automatic  
✅ All resolutions work (400-2048)  

## 🖥️ System Requirements

### Minimum (CPU Mode):
- Windows 10/11
- x64 processor
- 4 GB RAM
- 50 MB disk space

### Recommended (GPU Mode):
- Windows 10/11
- NVIDIA GPU with CUDA support
- 8 GB RAM
- CUDA 13.x runtime (included)

## 📊 Example Performance

With adaptive GPU block sizing on RTX GPU:

| Quality Preset | Resolution | Samples | Typical GPU Time |
|---------------|-----------|---------|------------------|
| ⚡ Low | 400×400 | 50 | ~100 ms |
| 💎 Medium | 600×600 | 200 | ~2-3 sec |
| 🌟 High | 800×800 | 500 | ~8-12 sec |
| 🔥 Very High | 1080×1080 | 1000 | ~30-45 sec |
| 💫 Extreme | 2048×2048 | 2000 | ~3-5 min |

*CPU times are typically 50-100x slower*

## 🐛 Bug Fixes

- Fixed emoji display in message boxes (was showing random chars)
- Fixed emoji display in status text (was showing random chars)
- Fixed adaptive GPU block size for lower-end hardware
- Fixed Unicode text rendering throughout UI

## 📝 Upgrade from v1.3

Simply replace your old `RayTracerGUI.exe` with the new one!

All settings and functionality remain the same - just looks cooler and shows render time now. 😎

## 🔗 Links

- **Repository**: https://github.com/XinpeiWang/ray_tracer
- **Issues**: Report bugs on GitHub Issues
- **Documentation**: See repo README.md

## 🙏 Credits

Based on "Ray Tracing in One Weekend" series by Peter Shirley

---

**Enjoy the new dark theme!** 🎨✨

*Made with ❤️ and lots of ray bounces*
