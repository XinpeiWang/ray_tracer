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
   ```
   Each `-Zip` run produces `releases\RayTracer_<Tier>_<yyyyMMdd>.zip`.

   All tiers assemble into the same `RayTracer_Package\` folder now (see
   `scripts/README.md`), additively - Medium adds files on top of Lite,
   Full adds files on top of Medium - and the folder itself is never wiped
   between runs. `-Zip` still always matches its own `-Tier` regardless of
   build order: it excludes any higher-tier leftovers at zip time rather
   than deleting them from the folder, so each `-Zip` run reflects exactly
   what its `-Tier` promises even if a larger tier was built there first.

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

All tiers assemble into the same `RayTracer_Package\` folder (the same one
`RayTracerGUI.pro`'s DESTDIR and every MSBuild post-build step already
write your local dev build into) - not a separate folder per tier:

```
RayTracer_Package/
├── RayTracer.exe             # CLI renderer, CPU-only - every tier
├── launcher.bat
├── README.txt
└── output/

# -Tier Medium or Full also adds:
├── RayTracerGUI.exe
├── scene_metadata.dll
├── Qt6*.dll, platforms/, styles/, imageformats/, multimedia/, ...

# -Tier Full also adds:
├── realtime_renderer.dll     # GPU Live Preview
├── optix_programs.ptx        # GPU shaders, JIT-compiled against the
├── wavefront_programs.ptx    #   target machine's own NVIDIA driver at launch
```

## Distribution

Packages can be shared via:
- GitHub Releases (recommended)
- Direct download links
- File sharing services

Do not commit large ZIP files to the main repository.
Keep them in this `releases/` folder locally only.
