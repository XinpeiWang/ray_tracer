// OptiX type definitions and shared structures
// Shared between host and device code

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

// Relative path (not a bare quoted include) so this resolves under both
// compile contexts that #include this file: MSBuild .cpp compiles (which do
// have src/shared on -I) and the recursive-path nvcc compile of
// optix_programs.cu (whose OptixFlags in build_optix.targets does NOT add
// src/shared to -I, unlike the wavefront compile flags). CPU_GPU-tagged and
// templated on T, so PointLightData<float>/SpotLightData<float>/
// DistantLightData<float> compile directly for device code - no reimplementation.
#include "../../src/shared/punctual_lights.h"
// Same relative-path reasoning as punctual_lights.h above - needed for the
// CPU_GPU tag on GpuLightSample/GpuReservoir's own accessor methods further
// down this file.
#include "../../src/shared/cpu_gpu.h"
#include "../../src/shared/filter_sampler.h"  // FilterSampler<float,32> - GpuCameraParams::filterSampler
// Same relative-path reasoning as punctual_lights.h above. CloudMedium<T> is
// CPU_GPU-tagged and templated, so CloudMedium<float> compiles directly for
// device code (it already calls perlin_noise<T> from noise.h, itself
// CPU_GPU-tagged and already used on-device by optix_device_helpers.h's
// sample_texture() for TextureKind::Noise) - no device-side reimplementation
// needed, same as the punctual lights above.
#include "../../src/shared/cloud_medium.h"
// Same relative-path reasoning as punctual_lights.h above. LightBVHNode/
// CompactLightBounds::Importance() are CPU_GPU-tagged and allocator-free
// (see light_bvh_node.h's own comment) - already exercised on the host by
// src/shared/bvh_light_sampler2.h (the tree BUILDER, used only host-side to
// construct LaunchParams::lightBvhNodes below), and now also compiled for
// device code here so gpu_light_bvh_sample_index()/gpu_light_bvh_pmf()
// (optix_device_helpers_lighting.h) can traverse the already-built tree
// directly - no device-side reimplementation of Importance()'s cone-angle
// math needed.
#include "../../src/shared/light_bvh_node.h"

#ifndef __CUDACC__
#include <stdexcept>
#include <string>
#endif

// Forward declarations
struct MaterialPOD;
struct LaunchParams;

#include "optix_types_geometry.h"
#include "optix_types_restir_media.h"
#include "optix_types_material.h"
#include "optix_types_lights_camera.h"
#include "optix_types_launch.h"
