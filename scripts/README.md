# Scripts

Utility scripts for building, packaging, and testing the ray tracer.

## Contents

### Building (local dev loop - shared RayTracer_Package/ folder)
- **`build_all.ps1`** - Build everything: launcher, renderers, tests, Qt GUI
  - Flags: `-Configuration Debug|Release`, `-SkipTests`, `-SkipGui`, `-Deploy`, `-Clean`
  - Stops a running RayTracerGUI.exe first; it locks the build output
  - Every individual-project MSBuild call passes `/p:SolutionDir` explicitly -
    see package.ps1's own comment on why (building a single .vcxproj instead
    of through ray_tracer.sln otherwise sends each project's post-build
    "deploy to RayTracer_Package" step to the wrong nested folder)
  - Usage: `.\scripts\build_all.ps1`
- **`build_all.bat`** - Batch equivalent
- **`build_and_deploy.ps1`** - build_all.ps1 followed by the Qt deployment
  step, for quickly running/testing your own local build - NOT for producing
  a release; use package.ps1 for that (below)
- **`build_and_deploy_macos.sh`** - macOS equivalent (CPU renderer + CLI + Qt
  GUI only - no GPU/OptiX backend on macOS). Run on macOS, not from Windows.
- **`setup_env.ps1`** / **`setup_env.bat`** - Auto-detect and set the CUDA/OptiX
  SDK paths a build needs; run once on a fresh machine before the first build.

### Deployment
- **`deploy_launcher.ps1`** - Verifies ray_tracer.exe and the PTX files reached
  RayTracer_Package/ after a build
- **`deploy_qt_gui.ps1`** - Copies the Qt runtime DLLs next to RayTracerGUI.exe

### Packaging (for sharing a release with others)
- **`package.ps1`** - Builds and assembles a clean, tiered distribution
  package directly into the same `RayTracer_Package\` folder as the local
  dev build above (windeployqt + VC++ redist DLLs + README/launcher on top
  of whatever's already there) - not a separate copy. The folder itself is
  never wiped/pruned (RayTracerGUI.exe and its Qt runtime have no other
  copy to rebuild from, so deleting them to "downgrade" a tier would
  destroy the only copy of that build), but `-Zip`'s output always matches
  the current `-Tier`: it excludes any higher-tier files left over from a
  previous run (e.g. `-Tier Lite -Zip` after a previous `-Tier Full` build
  excludes `RayTracerGUI.exe`, the Qt runtime, and the GPU DLLs/PTX from
  the zip). Without `-Zip`, running the folder directly can still launch
  more than the requested tier if a higher tier was built there before -
  the script warns when this applies.
  - `-Tier Lite|Medium|Full` (default `Full`):
    - `Lite` - RayTracer.exe (CLI) only, CPU rendering, no Qt/GUI
    - `Medium` - + RayTracerGUI.exe + Qt runtime, CPU rendering only
    - `Full` - + GPU rendering (OptiX/CUDA, Live Preview)
  - `-Configuration Debug|Release` (default `Release`)
  - `-SkipBuild` - assemble from whatever's already built, skip the build step
  - `-Zip` - also produce `releases\RayTracer_<Tier>_<date>.zip`
  - The CUDA runtime is statically linked into every binary that needs it
    (no `cudart64_*.dll` is ever required on the target machine - only the
    NVIDIA driver's own `nvcuda.dll`, never bundled). Medium cannot exclude
    Qt's multimedia/ffmpeg DLLs - they're a hard compile-time import of
    RayTracerGUI.exe (`QT += multimedia`), not a strippable runtime plugin;
    see the script's own header comment for why that would need a source
    change, not a packaging one.
  - Templates for the in-package `README.txt`/`launcher.bat` live in
    `scripts\templates\` - edit those, not this script, to change their wording.
  - Usage: `.\scripts\package.ps1 -Tier Lite` / `-Tier Full -Zip`

### Testing & Verification
- **`compare_images.py`** - Python script to compare rendered images
  - Compares CPU vs GPU output
  - Pixel-by-pixel difference analysis
  - Usage: `python scripts/compare_images.py image1.ppm image2.ppm`

### Maintenance
- **`clean_vs_cache.bat`** - Batch script to clean Visual Studio build cache
  - Removes intermediate files
  - Cleans .vs folder
  - Resets build state
  - Usage: `scripts\clean_vs_cache.bat`

## Usage Examples

### Create Distribution Package
```powershell
# From repository root - builds and packages in one step.
# Most setups only ever build one tier. If you need more than one tier's
# zip from a single build cycle, build smallest to largest as shown below -
# all tiers share one RayTracer_Package\ folder and are additive, so
# building Full first would leave its extra GPU/GUI files sitting in what's
# supposed to be a smaller Lite/Medium package.
.\scripts\package.ps1 -Tier Lite -Zip     # CLI only, smallest
.\scripts\package.ps1 -Tier Medium -Zip   # GUI, CPU rendering only
.\scripts\package.ps1 -Tier Full -Zip     # everything, GPU + GUI
```

### Compare Two Renders
```bash
# Compare CPU and GPU outputs
python scripts/compare_images.py output_cpu.ppm output_gpu.ppm
```

### Clean Build Cache
```cmd
# Reset Visual Studio build state
scripts\clean_vs_cache.bat
```

## Related Scripts

Other scripts are located in specific directories:
- **Qt GUI scripts:** `/qt_gui/verify_qt6.ps1`
- **Test scripts:** `/tests/build_and_run_tests.ps1`, `/tests/build_and_run_tests.bat`

## Dependencies

- **PowerShell scripts:** Require PowerShell 5.0+
- **Python scripts:** Require Python 3.6+ with PIL/Pillow
- **Batch scripts:** Run on Windows cmd.exe
