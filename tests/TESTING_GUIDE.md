# Ray Tracer Testing Guide

Complete guide for running and maintaining the ray tracer test suite.

## Quick Start

### Windows (Recommended)

```powershell
# Navigate to tests directory
cd tests

# Run the automated build and test script
.\build_and_run_tests.ps1
```

Or using batch file:

```cmd
cd tests
build_and_run_tests.bat
```

### Manual Build (CMake)

```bash
# From tests directory
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
ctest -C Release --output-on-failure
```

## Prerequisites

### Required Software

1. **CMake 3.14+**
   - Download: https://cmake.org/download/
   - Verify: `cmake --version`

2. **Visual Studio 2019 or 2022**
   - Community Edition is sufficient
   - Must include C++ Desktop Development workload

3. **Git** (for cloning Google Test)
   - Verify: `git --version`

### Optional

- **Visual Studio Test Explorer** (built into VS)
- **CUDA Toolkit** (for GPU tests, automatically skipped if unavailable)

## Test Structure

```
tests/
├── unit/                          # Unit tests (fast, isolated)
│   ├── camera_tests.cpp          # Camera positioning and ray generation
│   ├── scene_tests.cpp           # Cornell box scene construction
│   ├── math_tests.cpp            # Vector and color math
│   └── interface_tests.cpp       # Renderer interface validation
│
├── integration/                   # Integration tests (slower)
│   ├── serialization_tests.cpp   # GPU scene serialization
│   └── render_tests.cpp          # End-to-end rendering
│
├── CMakeLists.txt                # CMake build configuration
├── build_and_run_tests.ps1       # Automated PowerShell build script
├── build_and_run_tests.bat       # Automated batch build script
└── README.md                     # Test documentation
```

## Running Tests

### Option 1: Automated Scripts (Easiest)

**PowerShell (Recommended):**
```powershell
cd tests
.\build_and_run_tests.ps1
```

**Batch:**
```cmd
cd tests
build_and_run_tests.bat
```

These scripts will:
1. Download Google Test automatically
2. Build Debug and Release configurations
3. Run all tests
4. Display colored results

### Option 2: CMake + CTest

```bash
cd tests
mkdir build
cd build

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release

# Run all tests
ctest -C Release

# Run with verbose output
ctest -C Release --output-on-failure

# Run specific test
ctest -C Release -R CameraTest
```

### Option 3: Visual Studio Test Explorer

1. Open `ray_tracer.sln` in Visual Studio
2. Add the `tests/ray_tracer_tests.vcxproj` to solution (if not already added)
3. Build the solution
4. **View → Test Explorer** (Ctrl+E, T)
5. Click **Run All Tests** (green play button)

### Option 4: Run Test Executable Directly

```powershell
# After building with CMake
cd tests/build/Release
.\ray_tracer_tests.exe

# Run specific tests
.\ray_tracer_tests.exe --gtest_filter=CameraTest.*

# List all tests
.\ray_tracer_tests.exe --gtest_list_tests

# Verbose output
.\ray_tracer_tests.exe --gtest_verbose
```

## Test Categories

### Unit Tests (~100 tests, < 5 seconds)

Fast, isolated tests with no file I/O or rendering:

```bash
# Run only unit tests
ray_tracer_tests.exe --gtest_filter=*Test.*
```

**Camera Tests** (`camera_tests.cpp`)
- Camera position configuration
- Field of view calculations
- Ray generation
- Aspect ratio handling

**Math Tests** (`math_tests.cpp`)
- Vector operations (dot, cross, normalize)
- Color operations
- Interval clamping
- Utility functions

**Scene Tests** (`scene_tests.cpp`)
- Cornell box geometry
- Material properties
- Ray-scene intersection

**Interface Tests** (`interface_tests.cpp`)
- CPU renderer parameter validation
- GPU availability detection
- Return code checking

### Integration Tests (~30 tests, 30-60 seconds)

Slower tests that render actual images:

```bash
# Run only integration tests
ray_tracer_tests.exe --gtest_filter=*IntegrationTest.*
```

**Serialization Tests** (`serialization_tests.cpp`)
- Scene to POD conversion
- Camera serialization
- Material ID consistency

**Render Tests** (`render_tests.cpp`)
- End-to-end CPU rendering
- End-to-end GPU rendering (if available)
- Output file creation
- Camera preset validation

## Google Test Filters

### Common Patterns

```bash
# Run all camera tests
ray_tracer_tests.exe --gtest_filter=CameraTest.*

# Run a specific test
ray_tracer_tests.exe --gtest_filter=CameraTest.LookFromPosition

# Run multiple patterns
ray_tracer_tests.exe --gtest_filter=CameraTest.*:MathTest.*

# Exclude tests
ray_tracer_tests.exe --gtest_filter=-*GPU*

# Run only GPU tests
ray_tracer_tests.exe --gtest_filter=*GPU*

# Run fast tests only (exclude integration)
ray_tracer_tests.exe --gtest_filter=-*IntegrationTest.*
```

### Test Options

```bash
# Repeat tests (for finding flaky tests)
ray_tracer_tests.exe --gtest_repeat=10

# Shuffle test order
ray_tracer_tests.exe --gtest_shuffle

# Break on failure (debugging)
ray_tracer_tests.exe --gtest_break_on_failure

# List all tests without running
ray_tracer_tests.exe --gtest_list_tests
```

## GPU Tests

GPU tests are automatically skipped if CUDA is unavailable:

```
[ SKIPPED ] GPUInterfaceTest.ValidParameters (0 ms)
  GPU not available, skipping test
```

To run only GPU tests:
```bash
ray_tracer_tests.exe --gtest_filter=*GPU*
```

## Debugging Failed Tests

### Visual Studio Debugger

1. Set `tests/ray_tracer_tests` as startup project
2. Set breakpoint in failing test
3. Press F5 to debug
4. Use `--gtest_filter` in project properties to run specific test

**Project Properties → Debugging → Command Arguments:**
```
--gtest_filter=CameraTest.LookFromPosition
```

### Command Line Debugging

```bash
# Enable verbose output
ray_tracer_tests.exe --gtest_verbose

# Run only failed test
ray_tracer_tests.exe --gtest_filter=FailingTest.Name

# Break on failure
ray_tracer_tests.exe --gtest_break_on_failure
```

### Test Output Files

Integration tests create temporary output files:

```
tests/build/Release/
├── test_render_cpu_basic.ppm
├── test_render_gpu_basic.ppm
└── benchmark_*.ppm
```

Check these files if render tests fail:
1. Verify file exists
2. Open in image viewer (IrfanView, GIMP, etc.)
3. Check if image is blank, corrupted, or renders correctly

## Continuous Integration

Tests are designed for CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
	runs-on: windows-latest
	steps:
	  - uses: actions/checkout@v3

	  - name: Setup CMake
		uses: jwlawson/actions-setup-cmake@v1

	  - name: Build and Test
		run: |
		  cd tests
		  mkdir build
		  cd build
		  cmake .. -G "Visual Studio 17 2022" -A x64
		  cmake --build . --config Release
		  ctest -C Release --output-on-failure
```

## Performance Benchmarks

Some tests include performance benchmarks:

```
[ RUN      ] BenchmarkTest.SmallCPURender
Small CPU render (100x100, 1 spp): 342 ms
[       OK ] BenchmarkTest.SmallCPURender (345 ms)
```

Use these to:
- Detect performance regressions
- Compare CPU vs GPU performance
- Optimize hot paths

## Adding New Tests

### Step 1: Choose Test File

- **Unit test:** Add to existing file in `tests/unit/`
- **Integration test:** Add to existing file in `tests/integration/`
- **New component:** Create new test file

### Step 2: Write Test

```cpp
#include <gtest/gtest.h>
#include "your_header.h"

TEST(ComponentName, DescriptiveTestName) {
	// Arrange: Set up test data
	YourClass obj;

	// Act: Perform operation
	auto result = obj.doSomething();

	// Assert: Verify result
	EXPECT_EQ(result, expected_value);
}
```

### Step 3: Add to CMakeLists.txt (if new file)

```cmake
add_executable(ray_tracer_tests
	# ...existing files...
	unit/your_new_test.cpp
)
```

### Step 4: Build and Run

```bash
cd tests/build
cmake ..
cmake --build . --config Release
Release/ray_tracer_tests.exe --gtest_filter=ComponentName.*
```

## Best Practices

### Test Naming

- **Test Suite:** Noun describing component (`CameraTest`, `Vec3Test`)
- **Test Name:** Verb describing behavior (`LookFromPosition`, `AddsTwoVectors`)
- **Example:** `TEST(CameraTest, LookFromPosition)`

### Assertions

```cpp
// Prefer EXPECT over ASSERT (continues on failure)
EXPECT_EQ(actual, expected);      // Equality
EXPECT_NE(a, b);                  // Not equal
EXPECT_LT(a, b);                  // Less than
EXPECT_GT(a, b);                  // Greater than
EXPECT_NEAR(a, b, epsilon);       // Floating-point comparison
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Use ASSERT only when continuing is unsafe
ASSERT_NE(ptr, nullptr);  // Stop if pointer is null
ptr->doSomething();       // Would crash otherwise
```

### Test Structure

```cpp
TEST(SuiteName, TestName) {
	// Arrange: Set up objects and data
	camera cam;
	cam.lookfrom = point3(0, 0, 0);

	// Act: Perform the operation being tested
	cam.initialize();

	// Assert: Verify the result
	EXPECT_EQ(cam.center.x(), 0.0);
}
```

### Cleanup

```cpp
TEST(RenderTest, CreatesOutputFile) {
	const char* output = "test_file.ppm";

	// Render
	cpu_render_main(100, 100, 1, 5, output, 278, 278, -800);

	// Verify
	EXPECT_TRUE(file_exists(output));

	// Clean up (always!)
	std::remove(output);
}
```

## Troubleshooting

### "CMake not found"

Install CMake: https://cmake.org/download/

Add to PATH or use full path:
```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --version
```

### "Visual Studio generator not found"

Install Visual Studio with C++ Desktop Development workload.

Or try a different generator:
```bash
cmake .. -G "MinGW Makefiles"
cmake .. -G "Ninja"
```

### "Google Test download failed"

Check internet connection. Google Test is fetched from:
```
https://github.com/google/googletest/archive/...
```

Manual alternative:
1. Clone: `git clone https://github.com/google/googletest.git`
2. Build locally
3. Update CMakeLists.txt to use local copy

### Tests hang or crash

- Check for infinite loops in scene construction
- Verify render parameters (especially samples_per_pixel)
- Ensure output path is writable
- Check for memory leaks with large tests

### GPU tests always skip

- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Verify GPU: `nvidia-smi`
- Build GPU renderer: See main project README

## Test Coverage

Current coverage (estimated):

| Component | Coverage | Tests |
|-----------|----------|-------|
| Camera | 90% | 15 |
| Vec3/Math | 95% | 25 |
| Scene Construction | 75% | 10 |
| Renderer Interfaces | 80% | 12 |
| Serialization | 85% | 15 |
| End-to-End Rendering | 70% | 20 |

**Total:** ~100 tests covering core path tracer functionality.

## Future Improvements

- [ ] Add visual regression tests (compare rendered images)
- [ ] Benchmark suite for performance tracking
- [ ] Memory leak detection (Valgrind/AddressSanitizer)
- [ ] Fuzz testing for robustness
- [ ] Parallel test execution
- [ ] Test coverage reports (gcov/lcov)

## Resources

- **Google Test Documentation:** https://google.github.io/googletest/
- **CMake Documentation:** https://cmake.org/documentation/
- **Ray Tracing in One Weekend:** https://raytracing.github.io/
- **Project Documentation:** `../docs/`

## Support

If tests fail or you encounter issues:

1. Check this guide's Troubleshooting section
2. Review test output for specific error messages
3. Check `docs/CAMERA_SYSTEM.md` for camera behavior
4. Open an issue with:
   - Test name that failed
   - Full error output
   - Operating system and Visual Studio version
   - GPU availability (if relevant)
