# Root CMakeLists.txt -- CPU-only, cross-platform build (macOS/Linux; also
# usable on Windows itself for a sanity check, though the Windows dev
# workflow stays the MSBuild solution: ray_tracer.sln).
#
# This is purely additive: it never touches the existing .sln/.vcxproj files,
# and does not build the GPU/OptiX renderer at all (gpu/optix/,
# optix_renderer/) - that backend is CUDA/OptiX-only with no fallback and
# stays Windows+NVIDIA via MSBuild. This file gives the CPU renderer, the CLI
# launcher, and the scene_metadata shared library (which the Qt GUI loads
# dynamically) a build path that doesn't require the CUDA/OptiX SDK, so they
# can be built on a machine that doesn't have it - notably macOS, which can
# never have it.
#
# tests/CMakeLists.txt is separate and unaffected by this file - it already
# proved this project's CPU/shared header-only code (src/TheRestOfYourLife/*.h,
# src/shared/*.h) compiles cleanly under CMake/C++17.
#
# Usage:
#   cmake -B build && cmake --build build
# Produces: cpu_renderer (static lib), ray_tracer (CLI executable, no GPU
# support), scene_metadata (shared lib: scene_metadata.dylib on macOS,
# scene_metadata.so on Linux).

cmake_minimum_required(VERSION 3.14)
project(ray_tracer_cpu CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

# ---------------------------------------------------------------------------
# cpu_renderer - static library. Mirrors cpu_renderer/cpu_renderer.vcxproj's
# source list exactly. No CUDA/OptiX dependency (confirmed by the .vcxproj
# itself having none), so this is a clean base for every other target below.
# ---------------------------------------------------------------------------
add_library(cpu_renderer STATIC
    cpu_renderer/cpu_interface.cpp
    cpu_renderer/cpu_interface_bdpt.cpp
    src/external/stb_image_impl.cpp
    src/external/tinyexr_impl.cpp
    src/external/miniz.c
)
target_include_directories(cpu_renderer PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/TheRestOfYourLife
    ${CMAKE_SOURCE_DIR}/src/shared
    ${CMAKE_SOURCE_DIR}/cpu_renderer
)

# ---------------------------------------------------------------------------
# ray_tracer - CLI executable (the macOS/Linux counterpart of launcher.vcxproj's
# ray_tracer.exe). Reuses launcher/main.cpp unmodified beyond its new
# RT_HAVE_OPTIX/_WIN32 guards - this build never defines RT_HAVE_OPTIX, so it
# links launcher/optix_stub.h instead of the real OptiX interface and always
# renders on CPU (see launcher_args.h's matching #ifndef RT_HAVE_OPTIX
# fallback). OUTPUT_NAME is deliberately exactly "ray_tracer" (no extension)
# to match what qt_gui/mainwindow.cpp looks for on non-Windows builds.
# ---------------------------------------------------------------------------
add_executable(ray_tracer
    launcher/main.cpp
    src/external/image_writer.cpp
)
target_include_directories(ray_tracer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(ray_tracer PRIVATE cpu_renderer)
set_target_properties(ray_tracer PROPERTIES OUTPUT_NAME "ray_tracer")

# ---------------------------------------------------------------------------
# scene_metadata - shared library. Mirrors scene_metadata/scene_metadata.vcxproj:
# a thin passthrough to cpu_interface.h that the Qt GUI loads at runtime
# (dlopen on this platform, see qt_gui/scene_metadata_client.cpp) rather than
# linking directly. PREFIX "" drops CMake's default "lib" prefix so the
# output is exactly "scene_metadata.dylib"/"scene_metadata.so", matching the
# filename scene_metadata_client.cpp's non-Windows dlopen() branch looks for.
# ---------------------------------------------------------------------------
add_library(scene_metadata SHARED
    scene_metadata/scene_metadata_dll.cpp
)
target_link_libraries(scene_metadata PRIVATE cpu_renderer)
set_target_properties(scene_metadata PROPERTIES PREFIX "")
