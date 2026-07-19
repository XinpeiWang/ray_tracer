cmake_minimum_required(VERSION 3.14)
project(RayTracerTests)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the latest available toolset (VS 2026 = v180, VS 2022 = v143)
# CMake will automatically detect the installed toolset

# Include directories from the main project
include_directories(
	${CMAKE_SOURCE_DIR}/../src
	${CMAKE_SOURCE_DIR}/../src/TheRestOfYourLife
	${CMAKE_SOURCE_DIR}/../cpu_renderer
	${CMAKE_SOURCE_DIR}/../gpu/cuda
)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Unit tests
add_executable(ray_tracer_tests
	# Unit tests
	unit/camera_tests.cpp
	unit/scene_tests.cpp
	unit/math_tests.cpp
	unit/interface_tests.cpp

	# Integration tests
	integration/serialization_tests.cpp
	integration/render_tests.cpp
)

# Link GoogleTest
target_link_libraries(ray_tracer_tests
	GTest::gtest_main
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(ray_tracer_tests)

# Add compile definitions
target_compile_definitions(ray_tracer_tests PRIVATE
	NOMINMAX
	_USE_MATH_DEFINES
)

# Windows-specific settings
if(WIN32)
	target_compile_options(ray_tracer_tests PRIVATE /W4 /EHsc)
endif()
