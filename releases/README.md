# Releases

Distribution packages and release artifacts.

## Contents

- **`RayTracer_v1.5_Clean.zip`** - Version 1.5 clean build
- **`RayTracer_v1.6_Distribution.zip`** - Version 1.6 distribution package
- **`RayTracer_Download.html`** - Download page template

## Creating New Releases

Releases come in three tiers - see `scripts/README.md`'s own "Packaging"
section for exactly what each contains and why. Pick the tier(s) you want to
publish; a release doesn't have to include all three.

1. Build and package (one step - `package.ps1` builds it itself, there's no
   separate manual build step anymore):
   ```powershell
   .\scripts\package.ps1 -Tier Full -Zip
   .\scripts\package.ps1 -Tier Medium -Zip
   .\scripts\package.ps1 -Tier Lite -Zip
   ```
   Each `-Zip` run produces `releases\RayTracer_<Tier>_<yyyyMMdd>.zip`.

2. Test each package you're publishing:
   - Extract its zip to a clean directory (not next to the repo - a stray
     relative path bug would otherwise still resolve against repo files and
     pass locally while failing on a real user's machine)
   - Run `launcher.bat` (or `RayTracer.exe --help` for the Lite tier)
   - Full/Medium: verify the GUI starts and renders; Full only: verify GPU
     mode and Live Preview both work
   - Check a few scene presets

3. Create the release:
   - Tag the commit: `git tag v1.x.x`
   - Push tag: `git push origin v1.x.x`
   - Upload the zip(s) you tested to GitHub Releases (`gh release create
     v1.x.x releases/RayTracer_Full_*.zip releases/RayTracer_Medium_*.zip
     releases/RayTracer_Lite_*.zip --title v1.x.x --notes "..."`, or via the
     GitHub web UI)

## Version History

For current version info, see main `/README.md`.

## Package Structure

```
RayTracer_Package_Lite/
├── RayTracer.exe           # CLI renderer, CPU-only
├── launcher.bat
├── README.txt
└── output/

RayTracer_Package_Medium/    # everything Lite has, plus:
├── RayTracerGUI.exe
├── scene_metadata.dll
├── Qt6*.dll, platforms/, styles/, imageformats/, multimedia/, ...

RayTracer_Package_Full/      # everything Medium has, plus:
├── realtime_renderer.dll    # GPU Live Preview
├── optix_programs.ptx       # GPU shaders, JIT-compiled against the
├── wavefront_programs.ptx   #   target machine's own NVIDIA driver at launch
```

## Distribution

Packages can be shared via:
- GitHub Releases (recommended)
- Direct download links
- File sharing services

Do not commit large ZIP files to the main repository.
Keep them in this `releases/` folder locally only.
