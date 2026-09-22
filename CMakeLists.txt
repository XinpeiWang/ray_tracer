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
# On macOS, pass -DRT_BUILD_METAL=ON to also build metal_poc - a standalone
# Metal ray tracing proof of concept (gpu/metal/), NOT part of the ray_tracer
# CLI/Qt GUI and nowhere near feature parity with either the CPU or
# CUDA/OptiX GPU backends. See docs/METAL_GPU_FEASIBILITY.md for what this
# is, why it's separate, and what porting the real GPU renderer to Metal
# would actually take.
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
option(RT_BUILD_METAL "Also build the standalone Metal ray tracing proof of concept (macOS + Apple GPU only)" OFF)
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

# Same soft-optional posture as RT_BUILD_GPU above: fail clearly, not with
# a raw language-detection error, and only raise requirements/enable the
# language when actually asked for.
if(RT_BUILD_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "RT_BUILD_METAL=ON but this is not macOS - Metal only exists on Apple platforms")
    endif()
    # OBJCXX (Objective-C++) became a first-class enable_language() target
    # in CMake 3.16 - the CPU-only build this file exists for has no reason
    # to require anything newer than the 3.14 floor above, so this only
    # raises the bar for the RT_BUILD_METAL path, same shape as RT_BUILD_GPU
    # raising it for CUDA above.
    cmake_minimum_required(VERSION 3.16)
    enable_language(OBJCXX)
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
    src/data/rgb_spectrum_table_data.cpp
)
target_include_directories(cpu_renderer PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/TheRestOfYourLife
    ${CMAKE_SOURCE_DIR}/src/shared
    ${CMAKE_SOURCE_DIR}/src/external
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

# =============================================================================
# metal_poc - standalone Metal ray tracing proof of concept, only when
# -DRT_BUILD_METAL=ON. See docs/METAL_GPU_FEASIBILITY.md for what this is
# and is not (not part of ray_tracer/scene_metadata, no CPU/GPU-OptiX
# feature parity, a design-and-feasibility spike, not the start of the real
# GPU-on-macOS backend). Entirely independent of RT_BUILD_GPU/optix_renderer
# above - the two can never both be relevant on the same machine (CUDA
# needs an NVIDIA GPU, which cannot exist on the Apple Silicon this targets)
# but nothing here assumes RT_BUILD_GPU is OFF, they just don't interact.
# =============================================================================
if(RT_BUILD_METAL)
    # metal_renderer: a static lib holding metal_poc.mm's own code
    # (MetalPocApp, metal_render_main(), metal_poc_cli_main() - everything
    # except main() itself, which lives in metal_poc_main.mm instead - see
    # that file's own header comment for why) - split out of the metal_poc
    # EXECUTABLE (phase 3 of real GPU integration, docs/METAL_GPU_
    # FEASIBILITY.md's own section on this) so this same, single-compiled
    # object can ALSO be linked into ray_tracer below, mirroring
    # optix_renderer's own static-lib-linked-into-ray_tracer shape exactly
    # (see RT_BUILD_GPU's own block further up) - not yet dispatched to
    # from launcher/main.cpp (that's a separate, later phase), just linked
    # in and proven to build correctly alongside the CPU-only code it
    # already contains.
    add_library(metal_renderer STATIC
        gpu/metal/metal_poc.mm
        # One .mm per hand-authored scene CATEGORY (MetalPocApp::buildXxx()
        # out-of-class definitions only - the shared class declaration
        # itself lives in gpu/metal/metal_poc_app.h, #included by all of
        # these plus metal_poc.mm) - split out of metal_poc.mm once its
        # own scene-builder count made that one file unwieldy (a pure
        # code-motion refactor, no behaviour change - see
        # docs/METAL_GPU_FEASIBILITY.md).
        gpu/metal/metal_poc_scenes_a.mm
        gpu/metal/metal_poc_scenes_b.mm
        gpu/metal/metal_poc_scenes_c.mm
        gpu/metal/metal_poc_scenes_d.mm
        gpu/metal/metal_poc_scenes_e.mm
        gpu/metal/metal_poc_scenes_f.mm
        gpu/metal/metal_poc_scenes_g.mm
        gpu/metal/metal_poc_scenes_i.mm
        # loadPbrtScene() and its 9 phase methods (real pbrt scene
        # loading) - split out of metal_poc.mm once that one file's own
        # "Stage 2.5" block grew past a third of it, same pure-code-motion
        # precedent as the metal_poc_scenes_*.mm split just above.
        gpu/metal/metal_poc_pbrt_loader.mm
        # Stage 3 (GPU buffer/acceleration-structure upload) and Stage 4
        # (shader compile + dispatch + readback) - split out of
        # metal_poc.mm once that pair grew past ~1200 combined lines,
        # same pure-code-motion precedent as the two splits just above.
        gpu/metal/metal_poc_gpu_resources.mm
        gpu/metal/metal_poc_dispatch.mm
        # tinyexr's own zip/pxr24 codecs need miniz's real mz_compress/
        # mz_uncompress symbols (see the target_include_directories
        # comment below on why metal_renderer now needs tinyexr's real
        # implementation at all) - cpu_renderer gets this from the SAME
        # source file, listed separately above.
        src/external/miniz.c
    )
    # RT_METAL_SHADER_DIR/RT_MODELS_DIR (not __FILE__-relative lookups) so
    # both the shader source and models/suzanne.obj resolve correctly
    # regardless of the build directory's location relative to the source
    # tree - out-of-tree builds are the CMake norm, and __FILE__ isn't
    # guaranteed to stay a path anyone can navigate back to gpu/metal/ or
    # models/ from an arbitrary build dir.
    target_compile_definitions(metal_renderer PUBLIC
        RT_METAL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gpu/metal"
        RT_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
    # Needed once metal_poc.mm started including src/shared/pbrt_load.h
    # (real pbrt scene loading, not just the relative-path #includes this
    # target used before) - several src/shared/src/data headers use BARE
    # (non-relative) #include "some_header.h", relying on the including
    # TARGET having that header's own directory on its include path, not
    # just being reachable via a relative ../../ chain from metal_poc.mm
    # itself. Same list cpu_renderer's own target_include_directories()
    # above uses, minus cpu_renderer/ (this target doesn't need that one).
    target_include_directories(metal_renderer PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/TheRestOfYourLife
        ${CMAKE_SOURCE_DIR}/src/shared
        ${CMAKE_SOURCE_DIR}/src/external)
    # cpu_renderer: metal_render_main() (section 80) resolves a scene_id to
    # its pbrt file path via cpu_scene_pbrt_path_by_id() - the same shared
    # C-ABI accessor gpu/optix/scene_builder.cpp already uses for the same
    # purpose - rather than re-scanning pbrt_scenes/ a second, independent
    # way. Pure C++, no Objective-C anything, so linking it into this
    # OBJCXX target is unremarkable. PUBLIC (not PRIVATE): both frameworks
    # and cpu_renderer's own symbols are needed again at FINAL link time by
    # whichever executable links this static lib (metal_poc below, and
    # ray_tracer further down) - a static lib's own PRIVATE link
    # dependencies don't propagate to its consumers the way a shared lib's
    # would, so this has to be PUBLIC for ray_tracer's own final link step
    # to resolve metal_renderer's symbols correctly.
    target_link_libraries(metal_renderer PUBLIC cpu_renderer
        "-framework Metal" "-framework Foundation")
    set_target_properties(metal_renderer PROPERTIES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
    target_compile_options(metal_renderer PRIVATE -fobjc-arc)

    # The standalone metal_poc executable itself is now just main() (see
    # metal_poc_main.mm's own header comment) plus metal_renderer above.
    # It also compiles its own copy of src/external/image_writer.cpp -
    # metal_poc.mm calls stbi_write_png() but (unlike ray_tracer) doesn't
    # otherwise link anything that provides STB_IMAGE_WRITE_IMPLEMENTATION,
    # so it needs that one translation unit itself. metal_renderer can't
    # provide it instead: ray_tracer ALSO links metal_renderer below, and
    # ray_tracer already compiles image_writer.cpp directly (see its own
    # add_executable further up) - putting image_writer.cpp in
    # metal_renderer would just move the duplicate-symbol conflict rather
    # than fix it. One copy per final executable, none shared, keeps each
    # link with exactly one definition.
    add_executable(metal_poc gpu/metal/metal_poc_main.mm src/external/image_writer.cpp)
    target_link_libraries(metal_poc PRIVATE metal_renderer)
    set_target_properties(metal_poc PROPERTIES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
    target_compile_options(metal_poc PRIVATE -fobjc-arc)

    # ray_tracer links metal_renderer and gains RT_HAVE_METAL, same shape
    # as RT_BUILD_GPU's own RT_HAVE_OPTIX block further up. launcher/
    # main.cpp's --gpu dispatch now branches on RT_HAVE_METAL (phase 3b of
    # real GPU integration, see docs/METAL_GPU_FEASIBILITY.md) - this link
    # step is what makes metal_render_main() resolve there. ray_tracer's
    # own pre-existing src/external/image_writer.cpp (added further up,
    # long before RT_BUILD_METAL existed) already provides
    # STB_IMAGE_WRITE_IMPLEMENTATION for the whole executable, including
    # metal_renderer's own stbi_write_png() call - metal_poc.mm itself only
    # declares, never defines, that macro (see its own include comment).
    target_compile_definitions(ray_tracer PRIVATE RT_HAVE_METAL)
    target_link_libraries(ray_tracer PRIVATE metal_renderer)

    # --- Smoke test: does metal_poc still run and produce a real image? -
    # Seventeen incremental features (docs/METAL_GPU_FEASIBILITY.md
    # sections 8-26) accumulated in this POC with zero automated
    # regression coverage - every one was verified by rendering and
    # looking at the image by hand, which catches real bugs but leaves
    # nothing running automatically the next time an edit silently
    # breaks something else. metal_poc_validate.cpp is deliberately NOT
    # full CPU/GPU parity testing (real, separate future work) - just a
    # cheap catch for "the renderer crashed" or "the whole image
    # collapsed to one flat colour." enable_testing()/add_test() are
    # scoped inside this RT_BUILD_METAL block only - `ctest` still finds
    # nothing here unless RT_BUILD_METAL=ON, same additive posture as
    # every other choice in this block.
    add_executable(metal_poc_validate gpu/metal/metal_poc_validate.cpp)

    # --- Real regression coverage, not just a smoke test - see
    # gpu/metal/metal_poc_host_math.h and gpu/metal/metal_poc_math_tests.cpp's
    # own comments for the full "why" (closing the gap the smoke test
    # above deliberately doesn't: every material/light/tonemap correctness
    # fix documented in docs/METAL_GPU_FEASIBILITY.md - Fresnel, the
    # power-light-sampler alias table, the PR #43 chromatic-aberration
    # bug, the sRGB OETF - was verified once by hand and never re-checked
    # automatically again). No Metal/Foundation link dependency at all
    # (every function under test is plain, GPU-independent host C++), so
    # this doesn't need an actual device or acceleration structure to run
    # - just plain math, checked against known-correct reference values.
    add_executable(metal_poc_math_tests gpu/metal/metal_poc_math_tests.cpp)

    # --- The device-side half of that same regression coverage - see
    # gpu/metal/metal_poc_shader_tests.mm's own comment. Dispatches small,
    # purpose-built test kernels (metal_poc.metal's own "Device-side
    # unit-test kernels" section) against frDielectric(), the GGX
    # microfacet math, checkerColor(), spotLightFalloff(),
    # fresnelSchlickConductor(), henyeyGreensteinPhase(),
    # projectionLightRadiance(), and sampleAreaLight()'s own real
    # alias-table lookup - everything metal_poc_math_tests above couldn't
    # reach, since none of it has a host-side equivalent at all. Needs
    # Metal/Foundation and a real device (unlike metal_poc_math_tests),
    # same requirement metal_poc/metal_poc_smoke_render already have, and
    # the same RT_METAL_SHADER_DIR definition so it finds metal_poc.metal's
    # own source to compile at runtime regardless of build directory.
    add_executable(metal_poc_shader_tests gpu/metal/metal_poc_shader_tests.mm)
    target_compile_definitions(metal_poc_shader_tests PRIVATE
        RT_METAL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gpu/metal")
    target_link_libraries(metal_poc_shader_tests PRIVATE
        "-framework Metal" "-framework Foundation")
    set_target_properties(metal_poc_shader_tests PROPERTIES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
    target_compile_options(metal_poc_shader_tests PRIVATE -fobjc-arc)

    enable_testing()
    add_test(NAME metal_poc_math_tests COMMAND metal_poc_math_tests)
    add_test(NAME metal_poc_shader_tests COMMAND metal_poc_shader_tests)
    set(METAL_POC_SMOKE_OUTPUT "${CMAKE_BINARY_DIR}/metal_poc_smoke_test.png")
    add_test(NAME metal_poc_smoke_render
        COMMAND metal_poc 128 128 "${METAL_POC_SMOKE_OUTPUT}" 16 4)
    add_test(NAME metal_poc_smoke_validate
        COMMAND metal_poc_validate "${METAL_POC_SMOKE_OUTPUT}" 128 128)
    set_tests_properties(metal_poc_smoke_validate PROPERTIES DEPENDS metal_poc_smoke_render)
endif()
