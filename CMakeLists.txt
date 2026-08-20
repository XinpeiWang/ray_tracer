# Root CMakeLists.txt -- CPU-only by default, cross-platform build (macOS/
# Linux; also usable on Windows itself for a sanity check, though the
# Windows dev workflow stays the MSBuild solution: ray_tracer.sln).
#
# This is purely additive: it never touches the existing .sln/.vcxproj files.
# By default it does not build the GPU/OptiX renderer at all (gpu/optix/,
# optix_renderer/) - that backend is CUDA/OptiX-only with no fallback, so it
# gives the CPU renderer, the CLI launcher, and the scene_metadata shared
# library (which the Qt GUI loads dynamically) a build path that doesn't
# require the CUDA/OptiX SDK, so they can be built on a machine that doesn't
# have it - notably macOS, which can never have it.
#
# On Windows with an NVIDIA GPU, pass -DRT_BUILD_GPU=ON to also build the
# optix_renderer static lib and link it into ray_tracer, mirroring every
# nvcc invocation build_optix.targets makes for the MSBuild solution (see
# that file's own comments for the flag-by-flag rationale) via CMake's
# native CUDA language support instead of hand-written <Exec> targets. This
# is a second, independent way to build the exact same GPU backend - not a
# replacement for the MSBuild solution, which stays the primary Windows dev
# workflow. See the RT_BUILD_GPU block below for what it does and does not
# replicate.
#
# tests/CMakeLists.txt is separate and unaffected by this file - it already
# proved this project's CPU/shared header-only code (src/TheRestOfYourLife/*.h,
# src/shared/*.h) compiles cleanly under CMake/C++17.
#
# Usage:
#   cmake -B build && cmake --build build
# Produces: cpu_renderer (static lib), ray_tracer (CLI executable, no GPU
# support unless -DRT_BUILD_GPU=ON), scene_metadata (shared lib:
# scene_metadata.dylib on macOS, scene_metadata.so on Linux).

cmake_minimum_required(VERSION 3.14)
option(RT_BUILD_GPU "Also build the CUDA/OptiX GPU renderer (Windows + NVIDIA GPU + CUDA/OptiX SDK only)" OFF)
project(ray_tracer_cpu CXX C)

# Checked (not unconditionally enable_language'd) so RT_BUILD_GPU=ON on a
# machine without CUDA fails with a clear message instead of a raw CMake
# language-detection error - same soft-optional posture pbrt-v4's own
# CMakeLists.txt uses for its CUDA/OptiX support.
if(RT_BUILD_GPU)
    # Raises the requirement only for the GPU path, which is the only part
    # of this file that actually needs it (CUDA_ARCHITECTURES' "86-real"/
    # "86-virtual" suffix syntax below is a CMake 3.18 feature) - the
    # CPU-only build this file exists for in the first place (see header
    # comment: macOS/Linux support) has no reason to require anything newer
    # than the 3.14 floor above. cmake_minimum_required() can be called more
    # than once to raise the requirement partway through a file.
    cmake_minimum_required(VERSION 3.18)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "RT_BUILD_GPU=ON but no CUDA compiler was found - install the CUDA Toolkit or build without -DRT_BUILD_GPU=ON")
    endif()
    enable_language(CUDA)
endif()

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
# RT_HAVE_OPTIX/_WIN32 guards - without RT_BUILD_GPU this build never defines
# RT_HAVE_OPTIX, so it links launcher/optix_stub.h instead of the real OptiX
# interface and always renders on CPU (see launcher_args.h's matching #ifndef
# RT_HAVE_OPTIX fallback). With -DRT_BUILD_GPU=ON, RT_HAVE_OPTIX is defined
# and this links the optix_renderer target below instead - see that target's
# comment. OUTPUT_NAME is deliberately exactly "ray_tracer" (no extension) to
# match what qt_gui/mainwindow.cpp looks for on non-Windows builds.
# ---------------------------------------------------------------------------
add_executable(ray_tracer
    launcher/main.cpp
    launcher/diagnostics.cpp
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

# =============================================================================
# optix_renderer - GPU/OptiX backend, only when -DRT_BUILD_GPU=ON.
#
# Mirrors build_optix.targets' nvcc invocations flag-for-flag (see that
# file's own per-target comments for the reasoning behind each one), using
# CMake's native CUDA language support instead of hand-written <Exec>
# targets - the same approach pbrt-v4's own CMakeLists.txt uses on this
# exact toolchain (checked directly against its source: check_language(CUDA)
# + CUDA_PTX_COMPILATION for its raygen/hit/miss programs,
# CUDA_SEPARABLE_COMPILATION for its device-linked compute kernels).
#
# What this DOES replicate: every compiler flag (-O3, --use_fast_math,
# --relocatable-device-code=true, --std=c++17, the sm_86 PTX target vs the
# sm_86+sm_120 real-SASS gencode pair for linked kernels - see
# build_optix.targets' own comment on why plain CUDA kernels need explicit
# per-arch SASS while OptiX PTX modules don't), the /MD vs /MDd runtime
# library split, and the PostBuild PTX copy into RayTracer_Package/.
#
# What this does NOT replicate: OptiX has no standard CMake find-module (true
# in this project's own build_optix.targets AND in pbrt-v4's CMakeLists.txt -
# both just take a raw SDK path), so OptixSdkPath below is a plain cache
# variable/env fallback mirroring build_optix.targets' own property exactly,
# not a real find_package(OptiX). And PTX is still loaded from disk at
# runtime relative to the working directory (optix_renderer.cpp's loadPTX())
# rather than embedded into the binary the way pbrt's bin2c approach does -
# intentionally: this replicates current behavior, it doesn't change it.
# =============================================================================
if(RT_BUILD_GPU)
    set(OptixSdkPath "" CACHE PATH "Path to the OptiX SDK (e.g. C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0)")
    if(NOT OptixSdkPath)
        if(DEFINED ENV{OPTIX_SDK_PATH})
            set(OptixSdkPath "$ENV{OPTIX_SDK_PATH}")
        else()
            set(OptixSdkPath "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
        endif()
    endif()
    if(NOT EXISTS "${OptixSdkPath}/include/optix.h")
        message(FATAL_ERROR "OptiX SDK not found at '${OptixSdkPath}/include/optix.h' - pass -DOptixSdkPath=... or set the OPTIX_SDK_PATH environment variable")
    endif()

    find_package(CUDAToolkit REQUIRED)

    # Real SASS for both architectures this project targets today (see
    # build_optix.targets' own comment: plain relocatable-device-code
    # kernels have no JIT fallback, unlike OptiX's always-PTX modules, so
    # each supported arch needs its own explicit gencode entry).
    set(RT_CUDA_OBJ_ARCHITECTURES "86-real;120-real")
    # PTX only, JIT-compiled by OptiX at pipeline-creation time - matches
    # build_optix.targets' plain "-arch=sm_86" (no gencode) on its three
    # --ptx targets.
    set(RT_CUDA_PTX_ARCHITECTURES "86-virtual")

    set(RT_OPTIX_INCLUDE "${OptixSdkPath}/include")

    # --- PTX-only device programs (loaded at runtime via optixModuleCreate,
    #     never linked into an executable) -----------------------------------
    foreach(ptx_name optix_programs wavefront_programs sppm_programs)
        add_library(${ptx_name}_ptx OBJECT gpu/optix/${ptx_name}.cu)
        target_include_directories(${ptx_name}_ptx PRIVATE
            ${RT_OPTIX_INCLUDE} gpu/optix src/shared)
        set_target_properties(${ptx_name}_ptx PROPERTIES
            CUDA_PTX_COMPILATION ON
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES "${RT_CUDA_PTX_ARCHITECTURES}")
        target_compile_options(${ptx_name}_ptx PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --relocatable-device-code=true>)
        # optix_renderer.cpp's loadPTX() searches gpu/optix/ (source tree,
        # not the build dir) first - copy there to match, then also into
        # RayTracer_Package/ to match the MSBuild PostBuild event exactly.
        # OBJECT library targets can't take POST_BUILD commands directly
        # (CMake error: "may not have PRE_BUILD, PRE_LINK, or POST_BUILD
        # commands"), so this uses the OUTPUT form plus a driving ALL target
        # instead - functionally the same "copy after this compiles" effect.
        add_custom_command(
            OUTPUT "${CMAKE_SOURCE_DIR}/gpu/optix/${ptx_name}.ptx"
                   "${CMAKE_SOURCE_DIR}/RayTracer_Package/${ptx_name}.ptx"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/RayTracer_Package"
            COMMAND ${CMAKE_COMMAND} -E copy
                "$<TARGET_OBJECTS:${ptx_name}_ptx>"
                "${CMAKE_SOURCE_DIR}/gpu/optix/${ptx_name}.ptx"
            COMMAND ${CMAKE_COMMAND} -E copy
                "$<TARGET_OBJECTS:${ptx_name}_ptx>"
                "${CMAKE_SOURCE_DIR}/RayTracer_Package/${ptx_name}.ptx"
            DEPENDS ${ptx_name}_ptx
            COMMENT "[OptiX] Deployed ${ptx_name}.ptx"
            VERBATIM)
        add_custom_target(${ptx_name}_ptx_deploy ALL
            DEPENDS "${CMAKE_SOURCE_DIR}/gpu/optix/${ptx_name}.ptx")
    endforeach()

    # --- wavefront/SPPM device-linked kernel pairs ------------------------
    # CMake's automatic CUDA_RESOLVE_DEVICE_SYMBOLS/PerformDeviceLink
    # machinery turned out not to fire for this shape (a plain-C++
    # executable linking a separately-built STATIC lib that contains
    # separable-compiled CUDA objects) - confirmed empirically across
    # several combinations (CUDA_RESOLVE_DEVICE_SYMBOLS, LINKER_LANGUAGE
    # CUDA, giving the exe a dummy .cu source): none made the Visual
    # Studio generator emit a device-link step covering objects that came
    # from another project's .lib. So this does exactly what
    # build_optix.targets' own DeviceLinkWavefront/DeviceLinkSppm targets
    # do: compile each pair to relocatable objects, then explicitly
    # "nvcc -dlink" them into one resolved object, and feed THAT into the
    # static lib as a prebuilt object - the same structure the MSBuild
    # solution already uses (a self-contained .lib with the device-link
    # already resolved, so the final exe needs no CUDA-specific link
    # handling of its own).
    foreach(pair wavefront sppm)
        add_library(${pair}_kernels_obj OBJECT
            gpu/optix/${pair}_kernels.cu
            gpu/optix/${pair}_launch.cu)
        target_include_directories(${pair}_kernels_obj PRIVATE
            ${RT_OPTIX_INCLUDE} gpu/optix src/shared)
        set_target_properties(${pair}_kernels_obj PROPERTIES
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${RT_CUDA_OBJ_ARCHITECTURES}"
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        target_compile_options(${pair}_kernels_obj PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --relocatable-device-code=true>)

        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/${pair}_dlink.obj"
            COMMAND "${CMAKE_CUDA_COMPILER}"
                -dlink -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_120,code=sm_120
                -Xcompiler "/MD$<$<CONFIG:Debug>:d>"
                $<TARGET_OBJECTS:${pair}_kernels_obj>
                -o "${CMAKE_BINARY_DIR}/${pair}_dlink.obj"
            DEPENDS ${pair}_kernels_obj
            COMMENT "[CUDA] Device-linking ${pair} kernels"
            VERBATIM
            COMMAND_EXPAND_LISTS)
    endforeach()

    # --- optix_renderer - C++ host code + the two pre-device-linked
    #     kernel pairs, mirroring optix_renderer.vcxproj's own ClCompile
    #     list plus its Lib AdditionalDependencies exactly. The dlink .obj
    #     and the original per-file objects are ALL included (matching
    #     build_optix.targets: the dlink output resolves the
    #     __cudaRegisterLinkedBinary_* symbols but doesn't replace the
    #     original objects' own kernel entry points, which optix_renderer.cpp/
    #     wavefront_path_tracer.cpp/sppm_path_tracer.cpp call via the
    #     wf_launch_*/sppm_launch_* extern "C" wrappers). ------------------
    add_library(optix_renderer STATIC
        gpu/optix/optix_renderer.cpp
        gpu/optix/optix_interface.cpp
        gpu/optix/scene_builder.cpp
        gpu/optix/renderer_config.cpp
        gpu/optix/recursive_path_tracer.cpp
        gpu/optix/path_tracing_strategy.cpp
        gpu/optix/wavefront_path_tracer.cpp
        gpu/optix/sppm_path_tracer.cpp
        $<TARGET_OBJECTS:wavefront_kernels_obj>
        $<TARGET_OBJECTS:sppm_kernels_obj>
        "${CMAKE_BINARY_DIR}/wavefront_dlink.obj"
        "${CMAKE_BINARY_DIR}/sppm_dlink.obj"
    )
    target_include_directories(optix_renderer PUBLIC
        ${RT_OPTIX_INCLUDE}
        ${CMAKE_SOURCE_DIR}/gpu/optix
        ${CMAKE_SOURCE_DIR}/src/shared
    )
    target_compile_definitions(optix_renderer PUBLIC OPTIX_RENDERER_AVAILABLE)
    set_target_properties(optix_renderer PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_compile_options(optix_renderer PRIVATE
        # ConformanceMode (/permissive-) on the .cpp files, matching
        # optix_renderer.vcxproj's own <ConformanceMode>true</ConformanceMode> -
        # CMake's own MSVC toolset default here is permissive (/permissive-
        # off), which rejects a real, valid lambda capture of a constexpr
        # variable (kPi in scene_builder.cpp) that ConformanceMode's
        # stricter/more-standard-conformant mode accepts.
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    add_dependencies(optix_renderer optix_programs_ptx wavefront_programs_ptx sppm_programs_ptx)

    # ray_tracer links optix_renderer instead of the CPU-only default and
    # gains RT_HAVE_OPTIX so launcher/main.cpp takes the real-OptiX branch
    # (see that file's #ifdef RT_HAVE_OPTIX / #else optix_stub.h split) -
    # same two libraries launcher.vcxproj links for the real ray_tracer.exe.
    # No CUDA-specific properties needed on ray_tracer itself: optix_renderer
    # is now a fully self-contained static lib (device-link already done
    # above), so from here it's an entirely ordinary C++ link.
    target_compile_definitions(ray_tracer PRIVATE RT_HAVE_OPTIX)
    target_link_libraries(ray_tracer PRIVATE optix_renderer CUDA::cudart_static CUDA::cuda_driver)
    set_target_properties(ray_tracer PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
