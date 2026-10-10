cmake_minimum_required(VERSION 3.14)
project(ray_tracer_tests CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src ${CMAKE_SOURCE_DIR}/../src/TheRestOfYourLife ${CMAKE_SOURCE_DIR}/../src/shared ${CMAKE_SOURCE_DIR}/../src/external ${CMAKE_SOURCE_DIR}/../cpu_renderer ${CMAKE_SOURCE_DIR}/../gpu/cuda ${CMAKE_SOURCE_DIR}/../gpu/optix)

add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)

add_executable(unit_tests
    # Registered in a later pass (they existed on disk but were never built by this target); each compiles without the
    # OptiX/CUDA headers or the cpu_renderer/optix_renderer libraries. The 20 files that need the OptiX/CUDA headers (pbrt_gpu_*, sppm_gpu_*, wavefront_{nrc,restir,svgf,upscale}_*,
    # pbrt_alpha_cutout, quad_light_estimator_bias, shadow_ray),
    # plus energy_conservation_tests, gpu_render_tests, interface_tests, pbrt_example_scenes_tests and scene_registry_tests (they call the renderers' C entry points, so they need those libraries linked) stay out, for the reason documented on pbrt_flatten_tests above.
    unit/accelerator_override_tests.cpp
    unit/adaptive_sampling_tests.cpp
    unit/area_light_sampling_tests.cpp
    unit/bilinear_patch_hittable_tests.cpp
    unit/camera_math_tests.cpp
    unit/object_drag_math_tests.cpp
    unit/camera_path_tests.cpp
    unit/curve_shape_tests.cpp
    unit/curve_tessellate_tests.cpp
    unit/error_codes_tests.cpp
    unit/exr_writer_tests.cpp
    unit/stb_image_16bit_tests.cpp
    unit/gzip_inflate_tests.cpp
    unit/hair_precision_tests.cpp
    unit/launcher_args_accelerator_tests.cpp
    unit/launcher_args_adaptive_tests.cpp
    unit/launcher_args_bdpt_mlt_tests.cpp
    unit/launcher_args_denoise_blend_tests.cpp
    unit/launcher_args_lightsampler_tests.cpp
    unit/launcher_args_time_limit_tests.cpp
    unit/launcher_args_keyframes_tests.cpp
    unit/launcher_args_scene_file_tests.cpp
    unit/photo_helper_report_tests.cpp
    unit/scene_view_math_tests.cpp
    unit/mesh_preview_tests.cpp
    unit/model_library_tests.cpp
    unit/scene_blocks_tests.cpp
    unit/log_format_tests.cpp
    unit/oidn_runtime_tests.cpp
    unit/job_queue_tests.cpp
    unit/live_spp_scheduler_tests.cpp
    unit/scene_gizmo_tests.cpp
    unit/snapshot_history_tests.cpp
    unit/gpu_tessellate_tests.cpp
    unit/gpu_scene_frame_tests.cpp
    unit/scene_size_tests.cpp
    unit/scene_array_tests.cpp
    unit/scene_selection_tests.cpp
    unit/scene_camera_views_tests.cpp
    unit/camera_keyframes_tests.cpp
    unit/gpu_scene_textures_tests.cpp
    unit/gpu_scene_materials_tests.cpp
    unit/nanovdb_dense_tests.cpp
    unit/backend_capabilities_tests.cpp
    unit/light_sampler_resolution_tests.cpp
    unit/mtl_material_tests.cpp
    unit/obj_negative_indices_tests.cpp
    unit/palette_data_tests.cpp
    unit/palette_file_tests.cpp
    unit/ply_mesh_tests.cpp
    unit/render_output_parser_tests.cpp
    unit/restir_tests.cpp
    unit/sampler_kind_tests.cpp
    unit/scatter_record_transmission_tests.cpp
    unit/sphere_cone_sampling_tests.cpp
    unit/spherical_rectangle_adjudication_tests.cpp
    unit/transform_instance_tests.cpp
    unit/wavefront_temporal_upscale_math_tests.cpp
    unit/animated_transform_tests.cpp
    unit/anisotropic_ggx_tests.cpp
    unit/area_lights_tests.cpp
    unit/backend_consistency_tests.cpp
    unit/bdpt_adapter_medium_emission_tests.cpp
    unit/bdpt_adapter_punctual_sky_tests.cpp
    unit/bdpt_adapter_two_sided_tests.cpp
    unit/bdpt_sppm_bvh_world_tests.cpp
    unit/bdpt_tests.cpp
    unit/bilinear_patch_tests.cpp
    unit/bilinear_sampling_tests.cpp
    unit/blue_noise_tests.cpp
    unit/bounds_tests.cpp
    unit/bsdf_chi2_tests.cpp
    unit/bssrdf_tests.cpp
    unit/bssrdf_vol_path_tests.cpp
    unit/bvh_aggregate_tests.cpp
    unit/bvh_light_sampler2_tests.cpp
    unit/bvh_light_sampler_tests.cpp
    unit/bvh_tests.cpp
    unit/bxdf_tests.cpp
    unit/camera_tests.cpp
    unit/cameras_tests.cpp
    unit/catmull_rom_tests.cpp
    unit/cie_data_tests.cpp
    unit/cloud_medium_tests.cpp
    unit/color_encoding_tests.cpp
    unit/color_utils_tests.cpp
    unit/compact_light_bounds_tests.cpp
    unit/compensated_float_tests.cpp
    unit/concentric_disk_tests.cpp
    unit/direction_cone_tests.cpp
    unit/equal_area_tests.cpp
    unit/exhaustive_light_sampler_tests.cpp
    unit/film_tests.cpp
    unit/function_integrator_tests.cpp
    unit/filter_sampler_tests.cpp
    unit/filter_tests.cpp
    unit/float_bits_tests.cpp
    unit/fresnel_tests.cpp
    unit/goniometric_light_tests.cpp
    unit/grid_medium_tests.cpp
    unit/hair_bxdf_tests.cpp
    unit/halton_sampler_tests.cpp
    unit/homogeneous_medium_tests.cpp
    unit/image_infinite_light_tests.cpp
    unit/interval_tests.cpp
    unit/interval_vec_tests.cpp
    unit/kd_tree_hittable_tests.cpp
    unit/kd_tree_tests.cpp
    unit/layered_bxdf_tests.cpp
    unit/light_bounds_tests.cpp
    unit/light_bvh_node_tests.cpp
    unit/light_path_tests.cpp
    unit/loop_subdivide_tests.cpp
    unit/lowdiscrepancy_tests.cpp
    unit/materials_tests.cpp
    unit/math_tests.cpp
    unit/measured_bxdf_pbrt_tests.cpp
    unit/measured_bxdf_tests.cpp
    unit/measured_material_tests.cpp
    unit/microfacet_tests.cpp
    unit/mipmap_tests.cpp
    unit/mis_tests.cpp
    unit/mix_material_tests.cpp
    unit/mlt_tests.cpp
    unit/noise_tests.cpp
    unit/normal_map_tests.cpp
    unit/octahedral_variance_tests.cpp
    unit/padded_sobol_tests.cpp
    unit/path_integrator_tests.cpp
    unit/path_regularization_tests.cpp
    unit/path_sampler_tests.cpp
    # pbrt_alpha_cutout_tests.cpp is deliberately NOT listed here either
    # (see the pbrt_gpu_*_tests.cpp comment below) - it #includes
    # pbrt_gpu_builder.h directly, which transitively #includes <optix.h>.
    unit/pbrt_cpu_builder_tests.cpp
    unit/pbrt_discover_tests.cpp
    # pbrt_example_scenes_tests.cpp is deliberately NOT listed here - it's a
    # real end-to-end render-smoke test calling cpu_render_main AND
    # optix_render_main directly (its CPU-only agreement tests live in
    # cpu_integrator_agreement_tests.cpp, built into the separate
    # cpu_integrator_tests target at the bottom of this file) (launcher/cpu_interface.cpp, gpu/optix/
    # optix_interface.cpp), which would need this target to link the full
    # cpu_renderer + optix_renderer surface (and, transitively, the OptiX
    # SDK + CUDA Toolkit for the latter) - the same class of dependency the
    # pbrt_gpu_*_tests.cpp comment below explains staying out of this
    # portable, SDK-independent target for.
    unit/pbrt_flatten_tests.cpp
    unit/live_object_edit_tests.cpp
    unit/pbrt_arrangement_tests.cpp
    # pbrt_gpu_*_tests.cpp (bilinear_light/disk_cylinder/instance_sphere/
    # light_coverage/mix_material/quad_light/texture/triangle_light) are
    # deliberately NOT listed here: they transitively #include
    # gpu/optix/optix_types.h, which unconditionally #includes <optix.h>
    # and <cuda_runtime.h> - wiring them in would make this entire
    # portable, SDK-independent unit_tests target require the OptiX SDK +
    # CUDA Toolkit installed just to build, unlike every other test here.
    # tests/ray_tracer_tests.vcxproj carries those SDK include paths and
    # does list and successfully build/run all of these files (confirmed:
    # a clean MSBuild rebuild + full run passes 3765/3830, 0 failed - the
    # rest skipped for lacking a GPU or external assets). An earlier version
    # of this comment claimed that project was broken (C1083 on gtest/
    # gtest.h) - that was wrong, or described a since-fixed transient state;
    # do not trust it without re-verifying. The real, current reason these
    # files are excluded from THIS target specifically is just what the
    # paragraph above says: keeping this portable CMake target
    # SDK-independent. Use the MSVC solution (ray_tracer.sln,
    # `MSBuild ray_tracer.sln /t:ray_tracer_tests`) to build and run these.
    unit/pbrt_hash_tests.cpp
    unit/pbrt_instance_tests.cpp
    unit/pbrt_load_tests.cpp
    unit/pbrt_quadify_tests.cpp
    unit/pbrt_sampling_tests.cpp
    unit/pbrt_scene_tests.cpp
    unit/pbrt_subdiv_tests.cpp
    unit/piecewise_dist_tests.cpp
    unit/piecewise_linear_2d_tests.cpp
    unit/pixel_sensor_tests.cpp
    unit/pmj02_sampler_tests.cpp
    unit/portal_image_infinite_light_tests.cpp
    unit/power_light_sampler_tests.cpp
    unit/uniform_light_sampler_tests.cpp
    unit/principled_bsdf_tests.cpp
    unit/procedural_textures_tests.cpp
    unit/projection_light_tests.cpp
    unit/punctual_light_tests.cpp
    unit/punctual_lights_tests.cpp
    unit/ratio_tracking_tests.cpp
    unit/realistic_camera_tests.cpp
    unit/reservoir_sampler_tests.cpp
    unit/rgb_colorspace_tests.cpp
    unit/rgb_grid_medium_tests.cpp
    unit/rgb_spectrum_types_tests.cpp
    # Two OTHER independent copies of this same wiring exist - a textual
    # #include in gpu/optix/wavefront_path_tracer.cpp and a raw ClCompile
    # in scene_metadata/scene_metadata.vcxproj (see its comment) - there is
    # no single shared target providing this dependency to every consumer.
    ${CMAKE_SOURCE_DIR}/../src/data/rgb_spectrum_table_data.cpp
    # Needed by the newly-added pbrt_*_tests.cpp files above that decode
    # real image files (measured_bxdf_pbrt/pbrt_cpu_builder/pbrt_instance/
    # pbrt_load) via pbrt_load.h/pbrt_cpu_builder.h - those only declare
    # stb_image's/tinyexr's functions, the *_IMPLEMENTATION macro that
    # actually defines them lives in these two .cpp files (see cpu_renderer/
    # cpu_renderer.vcxproj's own identical pair for the sibling copy this
    # mirrors).
    ${CMAKE_SOURCE_DIR}/../src/external/stb_image_impl.cpp
    ${CMAKE_SOURCE_DIR}/../src/external/tinyexr_impl.cpp
    # tinyexr's Pxr24 compression path needs miniz's mz_compress/
    # mz_compressBound/mz_uncompress - same pairing cpu_renderer.vcxproj
    # already builds tinyexr_impl.cpp alongside.
    ${CMAKE_SOURCE_DIR}/../src/external/miniz.c
    # The Qt-free palette parser/data the GUI's theme system is built on (palette_data_tests/palette_file_tests).
    ${CMAKE_SOURCE_DIR}/../qt_gui/palette_data.cpp
    ${CMAKE_SOURCE_DIR}/../qt_gui/palette_file.cpp
    unit/rng_tests.cpp
    unit/sample_function_tests.cpp
    unit/sampled_grid_tests.cpp
    unit/sampled_spectrum_tests.cpp
    unit/sampling_distributions_tests.cpp
    unit/sampling_inversion_tests.cpp
    unit/sampling_sph_cone_tests.cpp
    unit/sampling_tests.cpp
    unit/scalar_distributions_tests.cpp
    unit/scalar_math_tests.cpp
    unit/scene_tests.cpp
    unit/shapes_tests.cpp
    unit/simple_path_tests.cpp
    unit/simple_vol_path_tests.cpp
    unit/sky_light_tests.cpp
    unit/sobol_sampler_tests.cpp
    unit/sobol_strat_tests.cpp
    unit/spectrum_types_tests.cpp
    unit/splines_tests.cpp
    unit/sppm_adapter_bsdf_tests.cpp
    unit/sppm_tests.cpp
    unit/square_matrix_tests.cpp
    unit/stratified_sampler_tests.cpp
    unit/summed_area_table_tests.cpp
    unit/surface_interaction_tests.cpp
    unit/texture_differentials_tests.cpp
    unit/texture_mapping_tests.cpp
    unit/tone_map_tests.cpp
    unit/triangle_tests.cpp
    unit/utility_integrators_tests.cpp
    unit/vol_path_tests.cpp
    unit/volume_scattering_tests.cpp
    unit/wavefront_tests.cpp
    unit/z_sobol_sampler_tests.cpp
)
target_include_directories(unit_tests PRIVATE ${COMMON_INCLUDES})
target_link_libraries(unit_tests PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS})   # ${CMAKE_DL_LIBS}: oidn_runtime.h loads Open Image Denoise with dlopen
gtest_discover_tests(unit_tests)

# Precompiled header for the ~150 test translation units: gtest and the standard headers every one of them parses.
# C++ only - the project also enables C
# (miniz), and a C++ PCH is a compile error for C sources. target_precompile_headers needs CMake 3.16.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
    target_precompile_headers(unit_tests PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:<gtest/gtest.h$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<algorithm$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<cmath$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<functional$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<map$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<memory$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<sstream$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<string$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<vector$<ANGLE-R>>")
endif()

# The whole suite runs in ~2.6 s optimised, so compile time dominates: a clean build of the 150 translation units at -O3
# takes ~177 s on an 8-core M2, ~131 s at -O1 and ~76 s at -O0 (60 s with the PCH above), while the run goes from 2.6 s
# to ~9.6 s. Default ON for GCC/Clang (verified: all tests pass at -O0, and one FMA-sensitive test that fails at -O3 on
# clang passes); OFF for MSVC until it has been tried there (stack frames are larger at /Od, which matters for the
# deeply recursive BVH tests) - configure with -DUNIT_TESTS_FAST_COMPILE=ON to opt in, e.g. from CI.
if(MSVC)
    option(UNIT_TESTS_FAST_COMPILE "Compile the unit tests without optimisation (faster build, slower run)" OFF)
else()
    option(UNIT_TESTS_FAST_COMPILE "Compile the unit tests without optimisation (faster build, slower run)" ON)
endif()
if(UNIT_TESTS_FAST_COMPILE)
    target_compile_options(unit_tests PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/Od;/Ob0,-O0>)
endif()


# CPU integrator agreement tests: the CPU path tracer, BDPT, MLT, SPPM and the debug integrators rendered on small pbrt scenes and compared with each other and
# with closed forms (cpu_integrator_agreement_tests.cpp). They call cpu_render_main*() directly, which is why they are not in unit_tests above: this target
# compiles the CPU renderer itself (cpu_renderer/cpu_interface.cpp + cpu_interface_bdpt.cpp, the same two files cpu_renderer.vcxproj builds) and needs no OptiX
# SDK or CUDA Toolkit, so GitHub CI can run it. The other agreement tests (pbrt_example_scenes_tests.cpp) compare CPU against the OptiX backends and stay in
# tests/ray_tracer_tests.vcxproj. Run it from the repository root: the tests find pbrt_scenes/ by a relative path.
add_executable(cpu_integrator_tests
    unit/cpu_integrator_agreement_tests.cpp
    unit/scene_builder_tests.cpp
    unit/scene_sky_tests.cpp
    unit/aov_pass_tests.cpp
    ${CMAKE_SOURCE_DIR}/../cpu_renderer/cpu_interface.cpp
    ${CMAKE_SOURCE_DIR}/../cpu_renderer/cpu_interface_bdpt.cpp
    ${CMAKE_SOURCE_DIR}/../src/data/rgb_spectrum_table_data.cpp
    ${CMAKE_SOURCE_DIR}/../src/external/stb_image_impl.cpp
    ${CMAKE_SOURCE_DIR}/../src/external/tinyexr_impl.cpp
    ${CMAKE_SOURCE_DIR}/../src/external/miniz.c
)
target_include_directories(cpu_integrator_tests PRIVATE ${COMMON_INCLUDES})
# A Debug MSVC build of these heavily templated sources exceeds the default object-file section limit (error C1128).
if(MSVC)
    target_compile_options(cpu_integrator_tests PRIVATE /bigobj)
endif()
target_link_libraries(cpu_integrator_tests PRIVATE GTest::gtest_main)
gtest_discover_tests(cpu_integrator_tests)
