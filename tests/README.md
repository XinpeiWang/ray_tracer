# Ray Tracer Unit Tests

This directory contains comprehensive unit tests for the ray tracer project using Google Test framework.

## Test Structure

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── camera_tests.cpp   # Camera position, lookat, field of view
│   ├── scene_tests.cpp    # Cornell box scene construction
│   ├── math_tests.cpp     # Vector math, color utilities
│   └── interface_tests.cpp # CPU/GPU interface parameter handling
├── integration/           # Integration tests across components
│   ├── render_tests.cpp   # End-to-end rendering tests
│   └── serialization_tests.cpp # Scene serialization for GPU
└── CMakeLists.txt         # Build configuration for Google Test
```

## Building Tests

### Prerequisites

1. **Google Test** (automatically downloaded by CMake)
2. **CMake 3.14+**
3. **Visual Studio 2019+** or MSVC compiler

### Build Commands

```bash
# From repo root
cd tests
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Run all tests
ctest -C Release

# Or run the test executable directly
./Release/ray_tracer_tests.exe
```

## Test Categories

### 1. Camera Tests (`unit/camera_tests.cpp`)
- ✅ Camera position configuration
- ✅ lookat target (fixed at Cornell box center)
- ✅ Field of view calculations
- ✅ Aspect ratio handling
- ✅ Ray generation from camera

### 2. Scene Tests (`unit/scene_tests.cpp`)
- ✅ Cornell box geometry construction
- ✅ Material properties (red, green, white walls)
- ✅ Light source configuration
- ✅ Object placement (glass sphere, rotated box)

### 3. Math Tests (`unit/math_tests.cpp`)
- ✅ Vector operations (dot, cross, normalize)
- ✅ Color operations (addition, multiplication, clamping)
- ✅ Ray-sphere intersection
- ✅ Ray-quad intersection
- ✅ Transform operations (translate, rotate)

### 4. Interface Tests (`unit/interface_tests.cpp`)
- ✅ CPU renderer parameter validation
- ✅ GPU renderer parameter validation
- ✅ Camera coordinate parsing
- ✅ Output path handling
- ✅ Error codes and return values

### 5. Serialization Tests (`integration/serialization_tests.cpp`)
- ✅ Scene POD conversion (spheres, quads, materials)
- ✅ Camera POD conversion
- ✅ Material ID consistency
- ✅ Transform application to geometry

### 6. Render Tests (`integration/render_tests.cpp`)
- ✅ Small image rendering (CPU + GPU)
- ✅ Camera position affects output
- ✅ Sample count affects quality
- ✅ Output file creation
- ✅ Image hash consistency

## Running Specific Tests

```bash
# Run only camera tests
./Release/ray_tracer_tests.exe --gtest_filter=CameraTest.*

# Run only GPU tests
./Release/ray_tracer_tests.exe --gtest_filter=*GPU*

# Verbose output
./Release/ray_tracer_tests.exe --gtest_verbose

# List all tests
./Release/ray_tracer_tests.exe --gtest_list_tests
```

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage of core components
- **Integration Tests**: All critical render paths
- **Performance Tests**: Ensure no major regressions

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Fast: Most tests complete in < 5 seconds
- Isolated: No external dependencies (except temp files)
- Deterministic: Same input → same output

## Adding New Tests

### Example Test Template

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

### Best Practices

1. **One concept per test**: Each test should verify one thing
2. **Descriptive names**: `CameraTest.LookFromSetsPosition` not `Test1`
3. **Arrange-Act-Assert**: Clear three-phase structure
4. **No side effects**: Tests should not depend on each other
5. **Fast execution**: Aim for < 100ms per unit test

## Debugging Failed Tests

```bash
# Run with Visual Studio debugger
devenv ray_tracer_tests.exe

# Or use the built-in VS Test Explorer
# View → Test Explorer → Run All
```

## Test Data

Test images and reference outputs are stored in:
```
tests/test_data/
├── reference/         # Known-good outputs for comparison
├── temp/             # Test-generated outputs (gitignored)
└── scenes/           # Small test scenes
```

## Known Limitations

- **GPU tests**: Require CUDA-capable GPU (skipped if unavailable)
- **Large renders**: Limited to small images (< 200x200) for speed
- **File I/O**: Tests create temporary files in system temp directory

## Future Test Additions

- [ ] Performance benchmarks
- [ ] Memory leak detection (Valgrind/AddressSanitizer)
- [ ] Fuzz testing for robustness
- [ ] Visual regression tests (image comparison)
