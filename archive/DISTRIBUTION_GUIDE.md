# Ray Tracer - Installation & Sharing Guide

## For Users Receiving This App

### Installation (No Install Required!)

This is a **portable application** - no installation needed!

1. Extract the `RayTracer_Package` folder to any location on your computer
   - Desktop, Documents, or any folder will work
   - Keep all files together in the same folder

2. Open the `RayTracer_Package` folder

3. Double-click `RayTracerGUI.exe` to launch

That's it! 🎉

### First Run

1. When you first launch, Windows Defender might show a warning
   - Click "More info" → "Run anyway"
   - This is normal for apps without a digital signature

2. Try your first render:
   - Select "Preview (Fast)" quality
   - Choose "800 x 800" resolution
   - Click "Start Render"
   - Wait ~30 seconds
   - Your rendered image will open automatically!

## For Developers Sharing This App

### Method 1: Zip File (Recommended)

1. Right-click the `RayTracer_Package` folder
2. Choose "Send to" → "Compressed (zipped) folder"
3. Share the resulting ZIP file

**Recommended upload sites:**
- Google Drive
- Dropbox
- OneDrive
- WeTransfer
- GitHub Releases

### Method 2: GitHub Release

```bash
# From your repository root
git add RayTracer_Package/
git commit -m "Update distribution package with icon"
git push

# Then create a release on GitHub:
# 1. Go to your repo: https://github.com/XinpeiWang/ray_tracer
# 2. Click "Releases" → "Create a new release"
# 3. Tag: v1.6
# 4. Title: "Ray Tracer v1.6 - Modern GUI with Enhanced Options"
# 5. Upload the zipped RayTracer_Package folder
```

### Method 3: Installer (Advanced)

For a professional installer, use:
- **NSIS** (Nullsoft Scriptable Install System)
- **Inno Setup**
- **WiX Toolset**

### What's Included in the Package

```
Total size: ~50 MB

Essential files:
✓ RayTracerGUI.exe      - Main application
✓ ray_tracer.exe        - Rendering engine
✓ Qt6*.dll             - GUI framework (35 MB)
✓ cuda*.dll            - GPU support (15 MB)
✓ Platform plugins     - Windows integration
✓ Image format plugins - PNG/JPEG support
✓ README.md            - User documentation
```

### Distribution Checklist

Before sharing, verify:
- [ ] RayTracerGUI.exe launches and shows the icon
- [ ] Test a quick render (Preview at 800x800)
- [ ] Output image opens automatically
- [ ] Stop button works
- [ ] All quality presets selectable
- [ ] All resolutions selectable
- [ ] README.md is included
- [ ] Version number is correct

### Platform Requirements

Recipients need:
- **Windows 10 or 11** (64-bit)
- **4+ GB RAM** (8GB recommended)
- **NVIDIA GPU** (optional, for GPU mode)
- No other dependencies!

### File Size Optimization (Optional)

Current size: ~50 MB

To reduce:
1. Remove unused Qt plugins (saves ~5 MB)
2. Remove debug symbols (already done)
3. Compress with 7-Zip for better compression

### Troubleshooting for Recipients

**"VCRUNTIME140.dll is missing"**
→ Already included in package

**"Qt platform plugin could not be initialized"**
→ Ensure `platforms` folder is present

**"Failed to start renderer"**
→ Make sure `ray_tracer.exe` is in the same folder

**Antivirus flags the app**
→ False positive; add exception or ignore warning

### Legal Notes

- ✅ Free to share for educational/personal use
- ✅ No license required for recipients
- ✅ Open source on GitHub
- ❌ Do not claim as your own work
- ❌ Do not sell commercially without permission

### Share Links

Once uploaded, share with a message like:

```
🎨 Ray Tracer - Path Tracing Renderer
Modern GUI with real-time rendering

Features:
• 6 Quality presets (Draft to Maximum 5000 samples!)
• 15 Resolutions (100x100 to 4K)
• Stop render button
• Auto-open results
• No installation required!

Download: [YOUR-LINK-HERE]
Size: ~50 MB
Platform: Windows 10/11 (64-bit)

Just extract and run RayTracerGUI.exe - enjoy! 🌟
```

### Update Strategy

For version updates:
1. Update version number in README.md
2. Commit and tag: `git tag v1.7`
3. Create new GitHub Release
4. Share new download link

### Analytics (Optional)

Track downloads with:
- GitHub release download count
- URL shortener with analytics (bit.ly, tinyurl)
- Google Analytics on a landing page

---

## Quick Distribution Commands

```bash
# Navigate to project root
cd C:\Users\xinpe\source\repos\ray_tracer

# Create distribution ZIP
Compress-Archive -Path RayTracer_Package -DestinationPath RayTracer_v1.6.zip -Force

# Check file size
Get-Item RayTracer_v1.6.zip | Select-Object Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}

# Upload to your preferred platform and share!
```

---

**Happy Sharing!** 🚀
