# Releases

Distribution packages and release artifacts.

## Contents

- **`RayTracer_v1.5_Clean.zip`** - Version 1.5 clean build
- **`RayTracer_v1.6_Distribution.zip`** - Version 1.6 distribution package
- **`RayTracer_Download.html`** - Download page template

## Creating New Releases

To create a new release package:

1. Build Release configuration:
   ```bash
   # Visual Studio
   Build → Batch Build → Select Release configurations → Build
   ```

2. Run packaging script:
   ```powershell
   .\scripts\package.ps1
   ```

3. Test the package:
   - Extract to clean directory
   - Run `RayTracerGUI.exe`
   - Verify GPU/CPU rendering works
   - Check all presets

4. Create release:
   - Tag the commit: `git tag v1.x.x`
   - Push tag: `git push origin v1.x.x`
   - Upload package to GitHub Releases

## Version History

See `/archive/RELEASE_NOTES_*.md` for historical release notes.

For current version info, see main `/README.md`.

## Package Structure

A typical release package contains:
```
RayTracer_vX.X/
├── RayTracerGUI.exe        # Qt GUI launcher
├── ray_tracer.exe          # Core renderer
├── Qt6*.dll                # Qt libraries
├── platforms/              # Qt platform plugins
├── styles/                 # Qt style plugins
├── README_PACKAGE.txt      # User instructions
└── examples/               # Example renders
```

## Distribution

Packages can be shared via:
- GitHub Releases (recommended)
- Direct download links
- File sharing services

Do not commit large ZIP files to the main repository.
Keep them in this `releases/` folder locally only.
