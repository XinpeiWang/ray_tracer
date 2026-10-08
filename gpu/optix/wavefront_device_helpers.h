#pragma once
// wavefront_device_helpers.h -- shared __device__ helper functions for the
// wavefront GPU path tracer's CUDA kernels, split out of wavefront_kernels.cu
// (which had grown to 4707 lines, 1451 of them - the RNG, texture, light-
// sampling, and camera-ray helpers below - shared by every kernel in that
// file). All __device__ __forceinline__, so including this header into
// multiple separately-compiled .cu translation units is the same one-
// definition-rule-safe pattern as any other inline C++ header; no new
// build_optix.targets/vcxproj wiring is needed for a header (unlike adding
// a new .cu file, which needs updating the WavefrontObjSource item group,
// the DeviceLinkWavefront dlink inputs, both configs' AdditionalDependencies,
// and the CleanOptixPrograms cleanup list) - build_optix.targets' own
// CudaHeaderDeps already globs every *.h in this directory, so editing this
// file alone correctly invalidates and recompiles every kernel .cu that
// includes it.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_types.h"
#include "optix_types.h"
#include "probe_grid_types.h"
#include "wavefront_guiding.h"
#include "wavefront_nrc_types.h"
#include "wavefront_nrc_encoding.h"
#include "wavefront_nrc_mlp.h"
#include "spectral_device.h"
#include "sampled_spectrum.h"
#include "spectrum_types.h"
#include "color_utils.h"
#include "optix_math_helpers.h"
#include "fresnel.h"
#include "microfacet.h"
#include "math_utils.h"
#include "camera_motion_blur_device.h"  // gpu_camera_anim_rotation/apply (shared with optix_device_helpers.h)
#include "bxdfs.h"  // HairBxDF<T>/PrincipledBxDF<T> - see MaterialType::Hair/Principled
#include "../../src/shared/noise.h"       // turbulence_simple - see wf_sample_texture()
#include "../../src/shared/normal_map.h"  // apply_normal_map - see MaterialType::NormalMappedLambertian
#include "../../src/shared/bilinear_patch.h"  // blp_sample - see wf_sample_bilinear_patch_light()
#include "../../src/shared/shading_frame.h"   // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured
#include "../../src/shared/sampling_helpers.h"  // SampleUniformDiskConcentric - see wf_sample_disk_light()

#include "wavefront_device_common.h"
#include "wavefront_device_camera.h"
#include "wavefront_device_scatter.h"
