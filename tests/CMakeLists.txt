cmake_minimum_required(VERSION 3.14)
project(ray_tracer_tests CXX)

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
    unit/animated_transform_tests.cpp
    unit/anisotropic_ggx_tests.cpp
    unit/area_lights_tests.cpp
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
    # optix_render_main directly (launcher/cpu_interface.cpp, gpu/optix/
    # optix_interface.cpp), which would need this target to link the full
    # cpu_renderer + optix_renderer surface (and, transitively, the OptiX
    # SDK + CUDA Toolkit for the latter) - the same class of dependency the
    # pbrt_gpu_*_tests.cpp comment below explains staying out of this
    # portable, SDK-independent target for.
    unit/pbrt_flatten_tests.cpp
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
target_link_libraries(unit_tests PRIVATE GTest::gtest_main)
gtest_discover_tests(unit_tests)
