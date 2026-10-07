# Contributing

Thanks for looking. This is a renderer that tries to be *right*, not just pretty, so
the most useful contributions come with a way to check them. This page says how to
build, how to test, what a good change looks like, and where help is wanted.

## Build and run

* **Windows (CPU + OptiX GPU)**: `.\scripts\build_and_deploy.ps1` from a Visual Studio
  Developer PowerShell. Details, prerequisites (CUDA, the OptiX SDK, Qt) and
  incremental-build tips are in [BUILD.md](BUILD.md).
* **macOS (CPU, plus Metal with `-DRT_BUILD_METAL=ON`)** and **no NVIDIA GPU**: the root
  `CMakeLists.txt` builds the CPU renderer and the CLI without CUDA or OptiX; see the
  [macOS](README.md#macos) section of the README. The CPU renderer is a complete
  renderer, so you can work on most of the code without a GPU.
* Scenes are listed by id (`A1`, `B3`, `K42`, ...). `ray_tracer.exe --cpu 400 64 8 A1`
  renders one; `--gpu` uses OptiX (Windows) or Metal (macOS). Any `.pbrt` file you drop
  into `pbrt_scenes/` becomes a scene without a rebuild.

## Tests

* **Windows**: build `ray_tracer_tests`, then run
  `powershell -ExecutionPolicy Bypass -File scripts/run_tests_parallel.ps1 -Tier Split`
  (under ten minutes on a desktop GPU; `-Tier Fast` skips the GPU tests, `-Filter` narrows
  by name). The suite has over 5,000 tests and needs to stay green.
* **Anywhere, and on GitHub CI** (no GPU): the CMake targets `unit_tests` and
  `cpu_integrator_tests` (see `tests/CMakeLists.txt` and `.github/workflows/unit-tests.yml`).
* **macOS Metal**: `ctest` in a `-DRT_BUILD_METAL=ON` build runs the CPU-vs-Metal parity
  sweep ([docs/METAL_PARITY_STATUS.md](docs/METAL_PARITY_STATUS.md)).

## What a good change looks like

1. **A way to know it is right.** The best check is a closed form: a furnace scene whose
   answer is the albedo, a Beer-Lambert absorber, a point light over a plate
   (`rho * L * (r/d)^2 * cos`). `pbrt_scenes/*furnace*.pbrt`, `chromatic-absorber.pbrt` and
   `distant-sphere-light-probe.pbrt` are examples, and `tests/unit/pbrt_example_scenes_tests.cpp`
   shows how a scene becomes a test. The next best is agreement between backends (CPU path
   tracer, the two OptiX backends, Metal) or with BDPT/MLT/SPPM on the same scene. When something
   is in doubt, read what pbrt-v4 does (<https://github.com/mmp/pbrt-v4>) instead of inferring it
   from a difference between two of our own backends: two backends can be wrong in the same way.
2. **A test that fails without the fix.** Revert your fix locally and check the test goes red.
3. **Say what you could not verify.** Most of us can run only one platform. "Windows CPU and
   OptiX tested; Metal not run" is a perfectly good thing to write in a pull request.
4. **Keep the docs honest.** If a feature changes how well a backend matches the others, update
   [docs/PBRT_SUPPORT.md](docs/PBRT_SUPPORT.md), which lists what each backend supports and how well.
5. **Follow the style in [CODING_STANDARDS.md](CODING_STANDARDS.md)**, keep a commit to one
   subject, and explain the *why* in its message. If a number changed, put the measurement in it.

## Licence of what you contribute

Your contribution is under the project's [MIT licence](LICENSE). If you port or adapt code from
another project (pbrt-v4 is the usual one), keep that project's attribution header in the file and
add the component to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Do not add meshes, textures or
scenes unless you know their licence and say what it is in your pull request.

## Where help is wanted

Small, self-contained things, roughly from easiest:

* A **provenance table for `models/`** (source and licence of each mesh): documentation only, and
  it would close a gap in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* **GPU SPPM** (`--sppm --gpu`): its photon gather radius is a fixed 5 units, where the CPU scales it
  to the scene (`cpu_renderer/cpu_interface.cpp`).
* **SPPM sky photons**: SPPM emits photons from area lights only, so sky-lit scenes read 2-3% low
  against the path tracer (`src/TheRestOfYourLife/sppm_adapter.h`).
* A **~10% gap on the Perlin-sphere scene (A7) under SPPM**, on CPU and GPU alike, whose cause is
  not known. Start with a closed-form version of the scene.
* **`--lightpath` in participating media** has never been checked.
* **Linux** support (the macOS port added most of the CMake and POSIX groundwork).
* A **native arm64 macOS GUI** build (the current one is x86_64 under Rosetta).
* Bigger projects: a GPU port of BDPT/MLT, and the remaining Metal gaps listed in
  [docs/METAL_PARITY_STATUS.md](docs/METAL_PARITY_STATUS.md).

New scenes with a closed-form check, and bug reports with a scene that shows the problem, are always welcome.
Please open an issue before starting anything large so we can agree on the approach.
