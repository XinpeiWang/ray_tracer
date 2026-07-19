# Qt6 Installation Guide for Ray Tracer Project

## Overview
This guide walks you through installing Qt6 on Windows for the Ray Tracer GUI modernization.

## Installation Steps

### 1. Download Qt Online Installer

**Download Link:** https://www.qt.io/download-qt-installer-oss

- Click "Download the Qt Online Installer"
- Select the Windows version
- File name: `qt-unified-windows-x64-online.exe` (~50MB)

### 2. Run the Installer

1. **Launch** `qt-unified-windows-x64-online.exe`
2. **Sign in** (create a free Qt account if needed)
3. **Accept** the open-source obligations (LGPL)
4. **Choose Installation Path** (default is fine, e.g., `C:\Qt`)

### 3. Select Components (IMPORTANT)

On the "Select Components" screen, expand the tree and select:

```
Qt
├── Qt 6.8 (or latest stable 6.x version)
	├── MSVC 2022 64-bit ✅ (REQUIRED - matches your Visual Studio)
	├── Qt 5 Compatibility Module ✅ (recommended)
	├── Qt Debug Information Files (optional, skip to save space)
	└── Sources (optional, skip to save space)
├── Developer and Designer Tools
	├── Qt Creator (optional but recommended for UI design)
	├── CMake (optional, you already have it)
	└── Ninja (optional)
```

**Minimal Required Components:**
- ✅ Qt 6.x → MSVC 2022 64-bit
- ✅ Qt 5 Compatibility Module

**Total Download Size:** ~2-3GB  
**Installed Size:** ~4-5GB

### 4. Complete Installation

- Click "Next" and wait for download/installation (15-30 minutes depending on connection)
- Click "Finish" when done

### 5. Verify Installation

Open PowerShell and check:

```powershell
# Check if Qt6 is installed
$qtPath = "C:\Qt\6.8.0\msvc2022_64\bin\qmake.exe"
if (Test-Path $qtPath) {
	Write-Host "✅ Qt6 installed successfully!" -ForegroundColor Green
	& $qtPath -version
} else {
	Write-Host "❌ Qt6 not found at $qtPath" -ForegroundColor Red
	Write-Host "Check your installation path" -ForegroundColor Yellow
}
```

### 6. Set Environment Variables (Optional but Recommended)

Add Qt to your PATH for easier access:

```powershell
# Add Qt6 to PATH (adjust version number if different)
$qtBinPath = "C:\Qt\6.8.0\msvc2022_64\bin"
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$qtBinPath", "User")

# Set Qt6_DIR for CMake/tools
[Environment]::SetEnvironmentVariable("Qt6_DIR", "C:\Qt\6.8.0\msvc2022_64", "User")

Write-Host "✅ Environment variables set. Restart your terminal." -ForegroundColor Green
```

**After setting environment variables, restart your PowerShell and Visual Studio.**

## What Gets Installed

- **Qt Creator**: GUI designer and IDE (optional but useful)
- **qmake**: Qt build tool
- **Qt Libraries**: Qt6Core, Qt6Widgets, Qt6Gui, etc.
- **Headers**: C++ headers for Qt development
- **Tools**: moc, uic, rcc (Qt meta-object compiler, UI compiler, resource compiler)
- **windeployqt**: Tool to package Qt apps with required DLLs

## Next Steps

After installation completes:

1. **Verify** the installation using the PowerShell check above
2. **Restart** Visual Studio to pick up new environment variables
3. **Let me know** it's installed, and I'll proceed with creating the Qt GUI project

## Troubleshooting

### "MSVC 2022 64-bit not available"
- Your Qt version might list it as "MSVC 2019 64-bit" - that works too!
- If neither is available, install Visual Studio 2022 C++ tools first

### "Installation failed"
- Check disk space (need ~10GB free)
- Try the offline installer instead
- Disable antivirus temporarily

### "qmake not found"
- Double-check the installation path (might be 6.7.x or 6.9.x instead of 6.8.x)
- Manually browse to `C:\Qt` and find the installed version

## Licensing Note

Qt is available under:
- **LGPL v3**: Free for commercial use if dynamically linked (DLLs)
- **GPL v3**: Free for open source projects
- **Commercial**: Paid license for static linking or proprietary needs

Our ray tracer will use **dynamic linking (DLLs)**, which is LGPL-compliant. ✅

---

**Ready?** Once Qt6 is installed and verified, let me know and we'll build the new GUI! 🚀
