// optix_device_helpers.h -- Shared device utilities for OptiX programs
// Included by optix_programs.cu

// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"    // Shared exact Fresnel (CPU+GPU)
#include "../../src/shared/volume_scattering.h" // HomogeneousMediumData/sample_homogeneous_event (CPU+GPU) - per-channel media
#include "../../src/shared/ray_hash.h"  // pbrt-v4 stochastic alpha test hash (CPU+GPU)
#include "../../src/shared/microfacet.h" // GGX TrowbridgeReitz (CPU+GPU)
#include "../../src/shared/bxdfs.h"      // HairBxDF<T> (CPU+GPU) - see MaterialType::Hair
#include "../../src/shared/noise.h"      // Perlin turbulence (CPU+GPU) - see sample_texture()
#include "../../src/shared/normal_map.h" // apply_normal_map (CPU+GPU) - see MaterialType::NormalMappedLambertian
#include "gpu_bump_map.h"               // grayscale "texture displacement" bump - see optix_intersection_triangle.h
#include "../../src/shared/bilinear_patch.h" // blp_sample/blp_pdf_wi (CPU+GPU) - see GpuLightKind::BilinearPatch
#include "../../src/shared/shading_frame.h"  // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured
#include "camera_motion_blur_device.h"        // gpu_camera_anim_rotation/apply (shared with wavefront_kernels.cu)

// Launch parameters (constant across all threads)
extern "C" { __constant__ LaunchParams params; }

// Device-side tabulated-BSSRDF evaluation (MaterialType::Subsurface,
// recursive backend only) - needs `params` above for the flat table arrays,
// so this include must stay below it.
#include "optix_bssrdf.h"

#include "optix_device_basics.h"
#include "optix_device_textures.h"
#include "optix_device_scatter_helpers.h"
#include "optix_device_shade.h"
#include "optix_device_mix_camera.h"
