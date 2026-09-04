# Ray Tracer Testing Guide

Complete guide for running and maintaining the ray tracer test suite:
**3,830 tests across 524 test suites**, in ~184 files under `tests/unit/`
and `tests/integration/`.

## Two build paths - pick one

This repo has **two separate ways to build the tests**, and they cover
different scope:

### The MSVC solution (`ray_tracer.sln`) - full coverage, recommended

```powershell
# From a VS Developer Command Prompt/PowerShell (needed for CUDA)
& "C:\Program Files\Microsoft Visual Studio\<version>\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" `
    ray_tracer.sln /t:ray_tracer_tests /p:Configuration=Release /p:Platform=x64

.\bin\Release\ray_tracer_tests.exe
```

This is `tests/ray_tracer_tests.vcxproj`, built as part of the full
solution alongside the CPU/GPU renderers. It links the OptiX SDK + CUDA
Toolkit and includes every test file in the suite, including the
OptiX-dependent ones (GPU material/scene builder tests, etc.). GPU tests
skip automatically if no compatible GPU is present rather than failing.
Run the executable directly from the repo root so relative asset paths
resolve correctly.

Or via Visual Studio's own Test Explorer: open `ray_tracer.sln`, build,
then **Test → Test Explorer → Run All**.

### The standalone CMake target (`tests/CMakeLists.txt`) - portable, no GPU SDK needed

```bash
cd tests
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
ctest -C Release --output-on-failure
```

This target (`unit_tests`) deliberately **excludes** the OptiX/CUDA-
dependent test files (anything that transitively `#include`s
`gpu/optix/optix_types.h` or links the full renderer) so it can build
without the OptiX SDK/CUDA Toolkit installed at all - see the exclusion
comments in `tests/CMakeLists.txt` for exactly which files and why. Use
this path when you specifically want a portable, SDK-independent build;
use the MSVC path above for full coverage.

`tests/build_and_run_tests.ps1` / `.bat` automate this CMake path (CMake
configure + Debug/Release build + run).

## Test Structure

```
tests/
├── unit/                  # 172 files - fast, isolated unit tests
├── integration/           # 12 files - slower, exercise real rendering
├── ray_tracer_tests.vcxproj  # MSVC target - full coverage, see above
├── CMakeLists.txt         # Standalone CMake target - portable subset
├── build_and_run_tests.ps1/.bat  # Automates the CMake path
└── README.md
```

Test files are organized by the subsystem they cover (materials, lights,
cameras, sampling, pbrt-loader directives, GPU parity, ...) - there are
too many to usefully enumerate here without this list drifting out of date
immediately; use `--gtest_list_tests` (see below) for the live, current
list, or `Glob`/`grep` the `tests/unit/`/`tests/integration/` directories.

## Running Specific Tests

```powershell
# Run a suite by name pattern
ray_tracer_tests.exe --gtest_filter=CameraTest.*

# Multiple patterns
ray_tracer_tests.exe --gtest_filter=CameraTest.*:MathTest.*

# Exclude a pattern
ray_tracer_tests.exe --gtest_filter=-*GPU*

# List every test without running (the authoritative current list)
ray_tracer_tests.exe --gtest_list_tests

# Repeat (find flaky tests) / shuffle order
ray_tracer_tests.exe --gtest_repeat=10
ray_tracer_tests.exe --gtest_shuffle

# Brief pass/fail summary only
ray_tracer_tests.exe --gtest_brief=1
```

### Quick dev-loop filter

A full run takes ~160s, but that's extremely concentrated: one suite
(`MaterialsAndVolumes/MaterialCpuGpuParityTest` - the name predates the
Materials/Volumes/Textures category split, it now covers all three)
accounts for ~55% of it by itself - a legitimate but heavy per-material
CPU/GPU-backend parity sweep, not wasted work. For fast local iteration:
```powershell
ray_tracer_tests.exe --gtest_filter=-MaterialsAndVolumes/*
```
cuts it to ~72s. See the main [README.md](../README.md)'s "Quick dev-loop
filter" section for a more aggressive variant. Always run the full,
unfiltered suite before pushing.

## GPU Tests

GPU tests skip automatically (not fail) if no compatible NVIDIA GPU/OptiX
is available:
```
[  SKIPPED ] GPUInterfaceTest.ValidParameters
```
Mesh-scene tests needing external assets not present on disk skip the same
way.

## Debugging Failed Tests

**Visual Studio debugger**: set `ray_tracer_tests` as the startup project,
set a breakpoint, and put `--gtest_filter=SuiteName.TestName` in Project
Properties → Debugging → Command Arguments, then F5.

**Command line**: `--gtest_break_on_failure` to break at the point of
failure; re-run a single failing test in isolation with
`--gtest_filter=ExactSuite.ExactTest` before assuming it's a real
regression rather than cross-test state leakage.

## Adding New Tests

```cpp
#include <gtest/gtest.h>
#include "your_header.h"

TEST(ComponentName, DescriptiveTestName) {
	// Arrange
	YourClass obj;

	// Act
	auto result = obj.doSomething();

	// Assert
	EXPECT_EQ(result, expected_value);
}
```

- Add to an existing file in `tests/unit/`/`tests/integration/` if it fits
  an existing subsystem, otherwise create a new file.
- A **new file** needs adding to `tests/ray_tracer_tests.vcxproj`'s
  `ClCompile` list (for the MSVC path) and, if it doesn't need the OptiX
  SDK, to `tests/CMakeLists.txt`'s `unit_tests` source list too (see that
  file's own exclusion comments for what disqualifies a file from the
  portable CMake target).
- Prefer `EXPECT_*` over `ASSERT_*` except where continuing after a
  failure would itself crash (e.g. `ASSERT_NE(ptr, nullptr)` before
  dereferencing).

## Troubleshooting

**GPU tests always skip** - install the CUDA Toolkit + NVIDIA OptiX SDK,
verify with `nvidia-smi`, and use the MSVC build path (the CMake path
never builds GPU tests at all).

**"CMake not found" / "Google Test download failed"** - install CMake
3.14+; Google Test is fetched via CMake `FetchContent`, so it needs
network access on first configure.

**A test hangs or crashes** - check scene construction for infinite loops,
verify render parameters (especially spp/depth aren't absurdly large),
and confirm the output path is writable.

## Resources

- Google Test docs: https://google.github.io/googletest/
- Project docs: [`../docs/`](../docs/)
