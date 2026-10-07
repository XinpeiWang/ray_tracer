## What and why
<!-- One paragraph. If a number changed (brightness, speed), put the before and after here. -->

## How it was checked
<!-- A closed form, backend agreement, a test, a render. Say which test fails without this change. -->

## Platforms
- [ ] Windows CPU
- [ ] Windows OptiX (recursive / wavefront)
- [ ] macOS CPU
- [ ] macOS Metal
- [ ] Not run on: <!-- say what you could not verify -->

## Checklist
- [ ] The test suite passes (`run_tests_parallel.ps1 -Tier Split` on Windows, or the CMake `unit_tests` target)
- [ ] `docs/PBRT_SUPPORT.md` updated if a backend's support or accuracy changed
- [ ] Ported code keeps its attribution header and is listed in `THIRD_PARTY_NOTICES.md`
- [ ] No new meshes, textures or scenes of unknown licence
- [ ] Touched `gpu/metal/` or a shared header the Metal build includes: `METAL_PARITY_STRICT=1 ctest` in `build_macos` passes on a Mac (CI's macOS runner has no hardware ray tracing and skips the device tests, so nothing else runs them)
- [ ] A refactor that must not change pictures: rendered PNGs compared byte for byte against a build of the previous `main` (say how many scenes)
- [ ] Added a test file: it is in both `tests/CMakeLists.txt` and `tests/ray_tracer_tests.vcxproj` (or in `tests/portable_build_exclusions.txt` with a reason) - `python3 scripts/check_build_lists.py`
- [ ] No function or file grew past `scripts/code_size_baseline.list` - `python3 scripts/check_code_size.py`
