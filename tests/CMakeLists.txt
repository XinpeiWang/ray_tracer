cmake_minimum_required(VERSION 3.14)
project(ray_tracer_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src ${CMAKE_SOURCE_DIR}/../src/TheRestOfYourLife ${CMAKE_SOURCE_DIR}/../src/shared ${CMAKE_SOURCE_DIR}/../cpu_renderer ${CMAKE_SOURCE_DIR}/../gpu/cuda ${CMAKE_SOURCE_DIR}/../gpu/optix)

add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)

add_executable(unit_tests
    unit/animated_transform_tests.cpp
    unit/anisotropic_ggx_tests.cpp
    unit/area_lights_tests.cpp
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
    unit/pbrt_hash_tests.cpp
    unit/pbrt_sampling_tests.cpp
    unit/piecewise_dist_tests.cpp
    unit/piecewise_linear_2d_tests.cpp
    unit/pixel_sensor_tests.cpp
    unit/pmj02_sampler_tests.cpp
    unit/portal_image_infinite_light_tests.cpp
    unit/power_light_sampler_tests.cpp
    unit/uniform_light_sampler_tests.cpp
    unit/principled_bsdf_tests.cpp
    unit/proc_tex_tests.cpp
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
    ${CMAKE_SOURCE_DIR}/../src/data/rgb_spectrum_table_data.cpp
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
    unit/sppm_tests.cpp
    unit/square_matrix_tests.cpp
    unit/stratified_sampler_tests.cpp
    unit/summed_area_table_tests.cpp
    unit/surface_interaction_tests.cpp
    unit/texture_differentials_tests.cpp
    unit/texture_mapping_tests.cpp
    unit/textures_tests.cpp
    unit/tone_map_tests.cpp
    unit/triangle_tests_patched.cpp
    unit/utility_integrators_tests.cpp
    unit/vol_path_tests.cpp
    unit/volume_scattering_tests.cpp
    unit/wavefront_tests.cpp
    unit/z_sobol_sampler_tests.cpp
)
target_include_directories(unit_tests PRIVATE ${COMMON_INCLUDES})
target_link_libraries(unit_tests PRIVATE GTest::gtest_main)
gtest_discover_tests(unit_tests)
