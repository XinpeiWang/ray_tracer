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
