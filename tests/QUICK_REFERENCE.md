# Ray Tracer Tests - Quick Reference

## One-Line Commands

```powershell
# Run all tests (automatic build)
.\build_and_run_tests.ps1

# Run all tests (manual)
cmake -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release && build\Release\ray_tracer_tests.exe

# Run specific test
build\Release\ray_tracer_tests.exe --gtest_filter=CameraTest.LookFromPosition

# List all tests
build\Release\ray_tracer_tests.exe --gtest_list_tests
```

## Test Categories

| Category | Command | Count | Time |
|----------|---------|-------|------|
| All Tests | `ray_tracer_tests.exe` | ~100 | 30-60s |
| Unit Tests | `--gtest_filter=*Test.*` | ~70 | < 5s |
| Integration Tests | `--gtest_filter=*IntegrationTest.*` | ~30 | 30-60s |
| Camera Tests | `--gtest_filter=CameraTest.*` | ~15 | < 1s |
| Math Tests | `--gtest_filter=*Math*:*Vec3*` | ~25 | < 1s |
| Render Tests | `--gtest_filter=*RenderTest.*` | ~20 | 30s |
| GPU Tests | `--gtest_filter=*GPU*` | ~8 | 10s |

## Common Filters

```bash
# CPU tests only (no GPU)
--gtest_filter=-*GPU*

# Fast tests only (no rendering)
--gtest_filter=-*IntegrationTest.*

# Specific test
--gtest_filter=CameraTest.LookFromPosition

# Multiple patterns
--gtest_filter=CameraTest.*:MathTest.*
```

## Build Configurations

```powershell
# Debug (with debug symbols)
cmake --build build --config Debug
build\Debug\ray_tracer_tests.exe

# Release (optimized)
cmake --build build --config Release
build\Release\ray_tracer_tests.exe
```

## Test File Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── camera_tests.cpp
│   ├── math_tests.cpp
│   ├── scene_tests.cpp
│   └── interface_tests.cpp
├── integration/             # Slower, end-to-end tests
│   ├── serialization_tests.cpp
│   └── render_tests.cpp
└── CMakeLists.txt          # Build configuration
```

## Debugging

```powershell
# Verbose output
ray_tracer_tests.exe --gtest_verbose

# Break on failure
ray_tracer_tests.exe --gtest_break_on_failure

# Repeat test (find flaky tests)
ray_tracer_tests.exe --gtest_repeat=10
```

## Output Files

Integration tests create temporary PPM files:
- `test_render_cpu_basic.ppm`
- `test_render_gpu_basic.ppm`
- `benchmark_*.ppm`

These are automatically cleaned up (deleted after test).

## Prerequisites

✅ CMake 3.14+  
✅ Visual Studio 2019+  
✅ C++17 compiler  
⚠️ CUDA (optional, for GPU tests)

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "CMake not found" | Install CMake, add to PATH |
| "Visual Studio generator not found" | Install VS with C++ Desktop Development |
| "Google Test download failed" | Check internet connection |
| GPU tests always skip | Install CUDA Toolkit |
| Tests hang | Check render parameters (samples_per_pixel) |

## Exit Codes

- `0` = All tests passed ✅
- `1` = Some tests failed ❌

## Adding a New Test

```cpp
#include <gtest/gtest.h>

TEST(ComponentName, TestName) {
	// Arrange
	MyClass obj;

	// Act
	auto result = obj.method();

	// Assert
	EXPECT_EQ(result, expected);
}
```

## CI/CD Integration

```yaml
# GitHub Actions
- run: |
	cd tests
	mkdir build
	cd build
	cmake .. -G "Visual Studio 17 2022" -A x64
	cmake --build . --config Release
	ctest -C Release --output-on-failure
```

## Performance Benchmarks

```bash
# Run benchmarks
ray_tracer_tests.exe --gtest_filter=BenchmarkTest.*

# Example output:
# Small CPU render (100x100, 1 spp): 342 ms
```

## Documentation

- **Full Guide:** `TESTING_GUIDE.md`
- **Test Details:** `README.md`
- **Camera System:** `../docs/CAMERA_SYSTEM.md`
- **Google Test Docs:** https://google.github.io/googletest/

## Google Test Cheat Sheet

```cpp
// Assertions
EXPECT_EQ(a, b);              // a == b
EXPECT_NE(a, b);              // a != b
EXPECT_LT(a, b);              // a < b
EXPECT_GT(a, b);              // a > b
EXPECT_NEAR(a, b, epsilon);   // |a - b| <= epsilon
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Fatal assertions (stop test)
ASSERT_EQ(ptr, nullptr);

// Floating-point
EXPECT_FLOAT_EQ(a, b);
EXPECT_DOUBLE_EQ(a, b);

// Exceptions
EXPECT_THROW(statement, exception_type);
EXPECT_NO_THROW(statement);

// Skip test conditionally
if (!gpu_is_available()) {
	GTEST_SKIP() << "GPU not available";
}
```
