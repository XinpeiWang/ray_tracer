# Releases

Distribution packages and release artifacts, for BOTH platforms this
project builds on - Windows (CLI + optional Qt GUI + optional GPU/OptiX)
and macOS (CLI + Qt GUI, CPU rendering only - there is no CUDA/OptiX on
macOS at all). The two platforms use separate build/package scripts (see
`scripts/README.md`) but share the same tag-and-upload release step at
the end, and can both be attached to the SAME GitHub Release (e.g.
`v1.x.x`) as separate, clearly-named assets - a user picks whichever
matches their own OS.

## Contents

This folder is gitignored except for this file and
`RayTracer_Download.html` (a download-page template) - built packages
(`.zip`/`.dmg`) land here locally when you run the steps below, but are
never committed (see "Distribution" below for why). Don't expect to find
built packages here in a fresh checkout.

## Creating New Releases (Windows)

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

3. Create (or add to) the release:
   - Tag the commit (skip if this is a second platform's package for a tag
     that's already pushed - see "Publishing both platforms together"
     below): `git tag v1.x.x` / `git push origin v1.x.x`
   - Upload the zip(s) you tested to GitHub Releases (`gh release create
     v1.x.x releases/RayTracer_Full_*.zip releases/RayTracer_Medium_*.zip
     releases/RayTracer_Lite_*.zip --title v1.x.x --notes "..."`, or via the
     GitHub web UI)

## Creating New Releases (macOS)

One combined build (CLI + Qt GUI together, always - there's no separate
tiering the way Windows has, since there's no GPU tier to opt into at all
on this platform):

1. Build and package (run this ON macOS, from the repo root - it is not
   usable from Windows or via cross-compilation):
   ```bash
   ./scripts/build_and_deploy_macos.sh
   ```
   This builds `cpu_renderer`/`ray_tracer`/`scene_metadata` via CMake, the
   Qt GUI via `qmake`, bundles both into `RayTracerGUI.app`, and runs
   `macdeployqt -dmg` to produce a distributable `.dmg` - both land in
   `RayTracer_Package_macOS/` (a separate folder from the Windows tiers'
   own `RayTracer_Package/`, so building both platforms in the same
   working copy - e.g. over a shared network drive, or in CI - can't clobber
   each other). Pass `--skip-dmg` to only produce the `.app` (e.g. for a
   quick local smoke test) without also invoking `macdeployqt -dmg`.
   Rename the `.dmg` to match the Windows zips' own naming convention
   before publishing it, e.g. `mv RayTracer_Package_macOS/RayTracerGUI.dmg
   releases/RayTracer_macOS_$(date +%Y%m%d).dmg`.

2. Test the package:
   - Copy the `.dmg` to a clean location (not next to the repo - same
     stray-relative-path concern the Windows step above already explains)
   - Mount it and drag `RayTracerGUI.app` to `/Applications` (or just
     launch it straight from the mounted volume for a quick check)
   - Verify the GUI starts and renders a few scene presets - CPU rendering
     only, there is no GPU/Live Preview tier to check here
   - If macOS Gatekeeper blocks the unsigned app (`.app` isn't
     notarized/code-signed by this project), right-click → Open once, or
     `xattr -cr RayTracerGUI.app` before testing - a real step a real
     downloader would also need until this project has an Apple Developer
     signing identity, worth calling out in the release notes

3. Create (or add to) the release - same tag-and-upload step as Windows,
   just with the `.dmg` instead of a `.zip`: `gh release create v1.x.x
   releases/RayTracer_macOS_*.dmg --title v1.x.x --notes "..."` (or `gh
   release upload v1.x.x releases/RayTracer_macOS_*.dmg` to add it to a
   release the Windows step above already created for the same tag).

### Publishing both platforms together

The two platforms build on different machines (Windows needs Visual
Studio/MSBuild; macOS needs Xcode command-line tools + Qt), so there is no
single machine that produces both packages in one run. To ship both under
the same version tag:
1. Tag and push ONCE, on whichever machine you build first:
   `git tag v1.x.x && git push origin v1.x.x`.
2. Build + test that platform's package (its own section above), then `gh
   release create v1.x.x <that platform's files> --title v1.x.x --notes
   "..."` - this creates the release.
3. On the OTHER machine, check out the SAME tag (`git fetch --tags && git
   checkout v1.x.x`), build + test that platform's package, then `gh
   release upload v1.x.x <that platform's files>` - this adds to the
   existing release rather than creating a second one for the same
   version.

## Version History

For current version info, see main `/README.md`.

## Package Structure

### Windows

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

### macOS

One combined bundle - no tiers, since there's no GPU backend to opt in or
out of on this platform at all:

```
RayTracer_Package_macOS/
├── RayTracerGUI.app/
│   └── Contents/
│       ├── MacOS/
│       │   ├── RayTracerGUI            # Qt GUI
│       │   ├── ray_tracer              # CLI renderer, CPU-only - the GUI's own subprocess
│       │   └── scene_metadata.dylib
│       └── Frameworks/                 # Qt frameworks, bundled by macdeployqt
└── RayTracerGUI.dmg                    # the actual file to publish/distribute
```

External mesh assets (Sponza, Bistro, and most other `requires_files=true`
scenes in the registry) are NOT bundled - see
`scripts/build_and_deploy_macos.sh`'s own header comment for why and for
the exact command to add them back in yourself after installing.

## Distribution

Packages can be shared via:
- GitHub Releases (recommended) - a single release/tag can carry both the
  Windows `.zip` file(s) and the macOS `.dmg` as separate assets; see
  "Publishing both platforms together" above
- Direct download links
- File sharing services

Do not commit large ZIP/DMG files to the main repository.
Keep them in this `releases/` folder locally only.
