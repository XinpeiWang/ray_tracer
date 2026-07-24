cmake_minimum_required(VERSION 3.14)
project(RayTracerTests)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the latest available toolset (VS 2026 = v180, VS 2022 = v143)
# CMake will automatically detect the installed toolset

# Common include directories (header-only renderer sources)
set(RT_INCLUDE_DIRS
	${CMAKE_SOURCE_DIR}/../src
	${CMAKE_SOURCE_DIR}/../src/TheRestOfYourLife
	${CMAKE_SOURCE_DIR}/../cpu_renderer
	${CMAKE_SOURCE_DIR}/../gpu/cuda
	${CMAKE_SOURCE_DIR}/../gpu/optix
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

# ---------------------------------------------------------------------------
# unit_tests: header-only tests — no renderer libs required
# Covers math, camera, scene geometry, BVH (SAH), scene registry.
# ---------------------------------------------------------------------------
add_executable(unit_tests
	unit/camera_tests.cpp
	unit/scene_tests.cpp
	unit/math_tests.cpp
	unit/bvh_tests.cpp
)

target_include_directories(unit_tests PRIVATE ${RT_INCLUDE_DIRS})

target_link_libraries(unit_tests GTest::gtest_main)

target_compile_definitions(unit_tests PRIVATE
	NOMINMAX
	_USE_MATH_DEFINES
)

if(WIN32)
	target_compile_options(unit_tests PRIVATE /W3 /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)

# ---------------------------------------------------------------------------
# integration_tests: requires compiled renderer libs — built only when the
# libs are explicitly provided via cmake -DRENDERER_LIB_DIR=<path>.
# ---------------------------------------------------------------------------
if(DEFINED RENDERER_LIB_DIR)
	add_executable(integration_tests
		unit/interface_tests.cpp
		unit/gpu_render_tests.cpp
		unit/energy_conservation_tests.cpp
		unit/scene_registry_tests.cpp
		integration/render_tests.cpp
	)

	target_include_directories(integration_tests PRIVATE ${RT_INCLUDE_DIRS})

	target_link_libraries(integration_tests
		GTest::gtest_main
		${RENDERER_LIB_DIR}/cpu_renderer.lib
		${RENDERER_LIB_DIR}/optix_renderer.lib
	)

	target_compile_definitions(integration_tests PRIVATE
		NOMINMAX
		_USE_MATH_DEFINES
	)

	if(WIN32)
		target_compile_options(integration_tests PRIVATE /W3 /EHsc)
	endif()

	gtest_discover_tests(integration_tests)
endif()
