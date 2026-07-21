# Video Generation - Quick Start Guide

> **⚠️ OUTDATED**: This guide describes the legacy FFmpeg-based workflow. The system now uses **OpenCV for automatic MP4 encoding**. No external tools are required. See `README.md` for current usage.

## ✅ Current Status

The video generation feature uses OpenCV and creates MP4 files automatically:

1. ✅ Renders all frames with animated camera paths
2. ✅ Automatically assembles frames into MP4 using OpenCV
3. ✅ No external dependencies (FFmpeg no longer needed)
4. ✅ Works from both command-line and Qt GUI

---

## Legacy Information (Historical Reference)

The video generation feature was **fully functional** with FFmpeg:

1. ✅ Rendered all 60 frames (800x800 @ 100 spp) in 21 seconds
2. ✅ Created orbital camera path around the Cornell box
3. ✅ Saved frames to `RayTracer_Package/output/frames/`
4. ✅ Fixed argument parsing bug (video flags no longer interfere with render settings)

## 📦 Next Step: Install FFmpeg

To complete the video assembly, you need FFmpeg installed:

### Windows Installation (Recommended):

**Option 1: Using Chocolatey (easiest)**
```powershell
choco install ffmpeg
```

**Option 2: Manual Installation**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Choose "ffmpeg-release-essentials.zip"
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system PATH:
   - Press Win + X → System → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK on all dialogs
5. **Restart your terminal/GUI application**

### Verify Installation:
```powershell
ffmpeg -version
```

## 🎬 Using Video Generation

### Via Qt GUI (Recommended):
1. Launch `RayTracerGUI.exe`
2. Select "Video Generation" mode at the top
3. Configure video settings:
   - Camera Path: orbit, linear, figure8, or spiral
   - Frames: 60 (default, ~2 second video at 30 FPS)
   - FPS: 30
4. Click "Start Render"
5. Wait for rendering to complete
6. **Video will automatically assemble and open** when FFmpeg is available

The GUI will show a helpful error message if FFmpeg is not installed.

### Via Command Line:
```powershell
cd RayTracer_Package

# Render frames (GPU mode)
.\ray_tracer.exe --gpu --video --frames 60 --fps 30 --camera-path orbit --output output/video.ppm 800 100 50

# Assemble video (requires FFmpeg)
PowerShell -ExecutionPolicy Bypass -File ..\scripts\assemble_video.ps1 -FramesDir output\frames -OutputPath output\video.mp4 -FPS 30
```

## 🎥 Camera Paths

- **orbit**: Circular path around the box (default)
- **linear**: Straight dolly movement
- **figure8**: Figure-8 pattern
- **spiral**: Spiral approach to center

## 📊 Performance

- Rendering: ~0.3-0.5s per frame (OptiX GPU renderer)
- 60 frames @ 800x800 @ 100 spp: ~21 seconds total
- Video assembly with FFmpeg: ~3-5 seconds

## 🐛 Troubleshooting

### "FFmpeg not found" error:
- Install FFmpeg using instructions above
- Restart your terminal/GUI after installation
- Verify with: `ffmpeg -version`

### Frames directory not found:
- Ensure you ran the render in video mode (`--video` flag)
- Check that frames exist in `RayTracer_Package/output/frames/`

### Video quality:
- Increase samples per pixel (spp) for better quality
- Default: 100 spp (good balance)
- High quality: 500-1000 spp (slow but beautiful)

## 📝 Files Modified Today

### Core Fix:
- `main.cpp`: Fixed argument parsing to track consumed arguments
  - Video flags (`--frames`, `--fps`, `--camera-path`) no longer treated as render settings
  - Added `std::set<int> consumed_args` tracking

### Video Assembly:
- `scripts/assemble_video.ps1`: Simplified, functional script
  - Converts PPM frames to PNG
  - Assembles MP4 video with FFmpeg
  - Cleans up temporary files

### Build:
- Launcher rebuilt successfully with MSBuild
- Updated `ray_tracer.exe` deployed to `RayTracer_Package/`

## 🎉 Summary

**Your video generation feature is ready to use!** Just install FFmpeg to enable automatic video assembly. The frames are already rendering perfectly, and the Qt GUI provides excellent user feedback throughout the process.
