# Ray Tracer C++ Coding Standards

## Overview
This document defines the coding standards and best practices for the Ray Tracer project. These standards have been applied throughout the codebase and should be followed for all future development.

## General Principles
- **Clarity over cleverness**: Write code that is easy to read and understand
- **Modern C++**: Use C++17/20 features where appropriate
- **const-correctness**: Use `const` wherever possible
- **Type safety**: Prefer static_cast over C-style casts
- **RAII**: Prefer automatic resource management over manual cleanup

## Naming Conventions

### Variables
- **Local variables**: `snake_case`
  ```cpp
  int samples_per_pixel = 100;
  const std::string output_path = "image.ppm";
  ```
- **Member variables**: `snake_case_` with trailing underscore
  ```cpp
  class Renderer {
	  int frame_count_;
	  CUdeviceptr d_framebuffer_;
  };
  ```
- **Constants**: `kPascalCase` with k prefix
  ```cpp
  constexpr int kDefaultWidth = 600;
  constexpr float kPi = 3.14159265358979323846f;
  ```

### Functions
- **Free functions**: `snake_case`
  ```cpp
  void render_scene(const SceneData& scene);
  ```
- **Member functions**: `camelCase`
  ```cpp
  class Renderer {
	  bool initialize();
	  void buildScene();
  };
  ```

### Types
- **Classes/Structs**: `PascalCase`
  ```cpp
  class OptiXRenderer { };
  struct MaterialData { };
  ```
- **Enums**: `PascalCase` for type, `PascalCase` for values
  ```cpp
  enum class MaterialType {
	  Lambertian,
	  Metal,
	  Dielectric
  };
  ```

### Files
- **Headers**: `.h` extension
- **Implementation**: `.cpp` extension
- **CUDA/OptiX**: `.cu` extension for device code
- **Names**: `snake_case.extension`

## Code Style

### Indentation and Spacing
- **Tabs for indentation** (project convention)
- **Spaces around operators**: `a + b`, not `a+b`
- **No trailing whitespace**
- **One statement per line**

### Braces
- **Opening brace on same line** for functions, control flow
  ```cpp
  void function() {
	  if (condition) {
		  // code
	  }
  }
  ```

### Type Declarations
- **Modern type aliases**: Use `using` instead of `typedef`
  ```cpp
  // Good
  using RaygenRecord = SbtRecord<int>;

  // Avoid
  typedef SbtRecord<int> RaygenRecord;
  ```

### Initialization
- **Prefer uniform initialization** with `{}`
  ```cpp
  QuadData wall{};
  OptixDeviceContextOptions options{};
  std::array<char, 256> buffer{};
  ```

### Type Conversions
- **Always use static_cast** for explicit conversions
  ```cpp
  // Good
  int width = static_cast<int>(numeric_args[0]);

  // Avoid
  int width = (int)numeric_args[0];
  ```

### Constants
- **Use constexpr** for compile-time constants
  ```cpp
  constexpr float kBoxSize = 555.0f;
  constexpr int kDefaultLogLevel = 3;
  ```
- **Anonymous namespaces** for file-local constants
  ```cpp
  namespace {
	  constexpr int kMaxIterations = 1000;
  }
  ```

## Documentation

### File Headers
Every file should have a Doxygen-style header:
```cpp
/// @file renderer.cpp
/// @brief OptiX renderer implementation
/// @details Handles GPU-accelerated path tracing using NVIDIA OptiX.
///          Manages device memory, acceleration structures, and rendering pipeline.
```

### Function Documentation
All public functions should be documented:
```cpp
/// @brief Render a frame using path tracing
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param samplesPerPixel Number of samples per pixel
/// @return true if rendering succeeded, false otherwise
bool render(unsigned int width, unsigned int height, unsigned int samplesPerPixel);
```

### Inline Comments
- Use `//` for single-line comments
- Place comments above the code they describe
- Keep comments concise and meaningful
- Avoid obvious comments

```cpp
// Good
// Calculate viewport dimensions based on field of view
const float viewport_height = 2.0f * h;

// Avoid
// Set viewport height
const float viewport_height = 2.0f * h;
```

## Error Handling

### Return Codes
- Use `EXIT_SUCCESS` and `EXIT_FAILURE` instead of 0 and 1
- Return error codes consistently

### Exception Handling
- **Catch specific exceptions** when possible
  ```cpp
  // Good
  try {
	  value = std::stoi(input);
  } catch (const std::exception& e) {
	  std::cerr << "Invalid input: " << e.what() << "\n";
  }

  // Avoid
  try {
	  value = std::stoi(input);
  } catch (...) {
	  // Silent errors
  }
  ```

### Input Validation
Always validate input parameters:
```cpp
bool buildScene(SceneData& scene, float* camera_params) {
	if (camera_params == nullptr) {
		return false;  // Invalid input
	}
	// ... proceed with valid input
}
```

## Modern C++ Features

### Smart Pointers
- Prefer smart pointers over raw pointers for ownership
- Use `std::unique_ptr` for exclusive ownership
- Use `std::shared_ptr` only when shared ownership is needed
- **Note**: CUDA/OptiX memory must use raw pointers (API requirement)

### Auto Keyword
- Use `auto` for complex iterator types
- Use `auto` when type is obvious from initialization
- Avoid `auto` when type clarity is important

```cpp
// Good
auto it = container.begin();
const auto& value = getLargeObject();

// Avoid (type not obvious)
auto x = calculate();  // What type is x?
```

### Range-Based For Loops
Prefer range-based loops when iterating over containers:
```cpp
// Good
for (const auto& sphere : spheres) {
	process(sphere);
}

// Avoid (unless index is needed)
for (size_t i = 0; i < spheres.size(); ++i) {
	process(spheres[i]);
}
```

### Standard Containers
- Prefer `std::vector` over C arrays
- Use `std::array` for fixed-size arrays
- Use `std::string` instead of C strings (except for C APIs)

```cpp
// Good
std::array<char, 256> deviceName{};
std::vector<MaterialData> materials;

// Avoid
char deviceName[256];
MaterialData* materials = new MaterialData[count];
```

## Resource Management

### RAII Principle
Resources should be acquired in constructors and released in destructors:
```cpp
class Renderer {
public:
	Renderer() {
		// Acquire resources
		cudaMalloc(&d_buffer_, size);
	}

	~Renderer() {
		// Release resources
		if (d_buffer_) {
			cudaFree(d_buffer_);
		}
	}

	// Delete copy operations
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	// Allow move operations
	Renderer(Renderer&&) noexcept = default;
	Renderer& operator=(Renderer&&) noexcept = default;
};
```

### Rule of Five
If you define any of these, consider defining all five:
- Destructor
- Copy constructor
- Copy assignment operator
- Move constructor
- Move assignment operator

## Type Safety

### Size Type Conversions
When converting between `size_t` and `int`, use safe casting:
```cpp
// Good - with validation
inline int safe_cast_to_int(size_t value) {
	assert(value <= static_cast<size_t>(INT_MAX));
	return static_cast<int>(value);
}

int material_index = safe_cast_to_int(materials.size());

// Avoid - implicit conversion warns
int material_index = materials.size();  // Warning C4267
```

### Const Correctness
- Mark parameters `const` when they won't be modified
- Mark member functions `const` when they don't modify state
- Mark return values `const` when appropriate

```cpp
// Good
bool render(const Camera& camera, const Scene& scene) const;
const std::string& getName() const { return name_; }

// Avoid
bool render(Camera& camera, Scene& scene);
std::string& getName() { return name_; }
```

### noexcept Specifier
Mark functions `noexcept` when they guarantee not to throw:
```cpp
bool isAvailable() noexcept;
~Renderer() noexcept;  // Destructors should be noexcept
void cleanup() noexcept;
```

## GPU-Specific Guidelines

### CUDA Memory Management
- Always check CUDA error codes (use CUDA_CHECK macro)
- Zero device pointers after freeing
- Check for nullptr before freeing

```cpp
if (d_buffer_) {
	CUDA_CHECK(cudaFree(d_buffer_));
	d_buffer_ = 0;
}
```

### OptiX Programming
- Use OptiX error checking macros (OPTIX_CHECK)
- Document shader entry points clearly
- Keep device code in `.cu` files
- Use proper alignment for SBT records

## Performance Considerations

### Avoid Unnecessary Copies
```cpp
// Good - pass by const reference
void process(const std::vector<int>& data);

// Avoid - copies entire vector
void process(std::vector<int> data);
```

### Reserve Vector Capacity
When size is known, reserve capacity:
```cpp
std::vector<MaterialData> materials;
materials.reserve(expected_count);
```

### Use Structured Bindings
For tuple-like types (C++17):
```cpp
// Good
const auto [width, height] = getImageSize();

// Avoid
const auto size = getImageSize();
const int width = size.first;
const int height = size.second;
```

## Testing and Validation

### Compile-Time Checks
Use `static_assert` for compile-time validation:
```cpp
static_assert(sizeof(LaunchParams) % 8 == 0, "LaunchParams must be 8-byte aligned");
```

### Runtime Validation
- Validate all input parameters
- Check return values from library functions
- Use assertions for internal invariants

```cpp
assert(width > 0 && "Image width must be positive");
```

## Project-Specific Conventions

### Cornell Box Constants
- Box dimensions: 555 units
- Center point: (278, 278, 278)
- Use `kBoxSize` and `kCornellBoxCenter` constants

### Ray Offset
- Minimum t-value for ray intersections: 0.001f
- Ray origin offset for shadows: 0.01f

### Thread Safety
- CPU renderer uses thread-local RNG
- Document thread-safe vs. thread-unsafe functions
- Use `RAY_TRACER_THREADS` environment variable

## Code Review Checklist

Before committing code, verify:
- [ ] Follows naming conventions
- [ ] Has appropriate documentation
- [ ] Uses const-correctness
- [ ] Uses static_cast instead of C-style casts
- [ ] Error handling is robust
- [ ] Resources are managed with RAII
- [ ] No magic numbers (use named constants)
- [ ] Compiles without warnings
- [ ] Tested on target hardware (RTX 5080 for GPU code)

## Tools and Automation

### Recommended Tools
- **Compiler**: Visual Studio 2026 (18.8.0)
- **CUDA**: Version 13.2+
- **OptiX**: Version 9.1
- **Qt**: Version 6.11.1 (for GUI)

### Build System
- Use `build_all.ps1` for complete build
- Use `build_and_deploy.ps1` for build + package
- Run `deploy_qt_gui.ps1` after Qt GUI changes

## Additional Resources

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/)

## Revision History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2025-01-XX | Initial coding standards based on codebase review |

---

*This document is maintained as part of the Ray Tracer project. Suggestions for improvements can be submitted via issues or pull requests.*
