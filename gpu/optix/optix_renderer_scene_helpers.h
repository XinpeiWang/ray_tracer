#pragma once
// optix_renderer_scene_helpers.h -- what optix_renderer_scene.cpp and optix_renderer_upload.cpp share: the includes, a device-memory guard and the upload/AABB helpers.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
#include "optix_math_helpers.h"
#include "../../src/shared/bilinear_patch.h"  // blp_area - alias-table power for GpuLightKind::BilinearPatch
#include "optix_disk_cylinder_helpers.h"  // dc_area_disk/dc_area_cylinder - alias-table power for GpuLightKind::Disk/Cylinder
#include "../../src/shared/bvh_light_sampler2.h"  // BVHLightSampler2 - host-side light-BVH tree builder (see d_lightBvhNodes_'s own comment, optix_renderer.h)
#include <cuda.h>
#include <iostream>
#include <cstring>       // memcpy, for instance transform packing
#include <type_traits>   // remove_pointer_t, for the light-flag width assert
#include <cmath>         // ceil/isnan - buildProbeGrid()'s own spacing/dims derivation
#include <algorithm>     // min/max - buildProbeGrid()'s own spacing/dims derivation


// RAII device-memory guard for the temp/output buffers each accel build
// below allocates: frees the held pointer on scope exit (cudaFree(0) is a
// documented no-op, so an empty/already-released guard is harmless) unless
// disarmed via release() once the pointer has been safely handed off (an
// output buffer folded into a live GAS) or already freed manually. Exists
// because OPTIX_CHECK/CUDA_CHECK throw std::runtime_error on failure
// (optix_types.h), and every accel-build block below used to hold its temp/
// output buffers in bare CUdeviceptr locals with the cudaFree() calls placed
// AFTER optixAccelBuild() - an accel-build failure (bad input, driver error,
// OOM) unwound past those frees and permanently leaked the buffers. A
// destructor-based guard frees correctly on both the normal and the
// exception path with no change to the normal path's timing.
struct ScopedCudaFree {
	CUdeviceptr ptr = 0;
	ScopedCudaFree() = default;
	ScopedCudaFree(const ScopedCudaFree&) = delete;
	ScopedCudaFree& operator=(const ScopedCudaFree&) = delete;
	~ScopedCudaFree() { if (ptr) cudaFree(reinterpret_cast<void*>(ptr)); }
	void release() { ptr = 0; }
};

// World-space AABB for a disk/cylinder given its object-space extent and
// o2w transform - corner-by-corner (a naive transform of the object-space
// box's own min/max would clip the geometry the moment a rotation is
// involved), same technique as disk_cylinder_hittable.h's CPU-side
// transformed_bbox(). Hoisted to file scope (out of buildScene()'s own
// local lambda further down, which still delegates here) so buildProbeGrid()
// below can share it instead of falling back to a looser approximation.
inline OptixAabb wf_disk_cylinder_world_aabb(const float o2w[12],
											  float xlo, float xhi, float ylo, float yhi,
											  float zlo, float zhi) {
	OptixAabb box{};
	float lox = 0, loy = 0, loz = 0, hix = 0, hiy = 0, hiz = 0;
	bool first = true;
	for (int corner = 0; corner < 8; ++corner) {
		const float x = (corner & 1) ? xhi : xlo;
		const float y = (corner & 2) ? yhi : ylo;
		const float z = (corner & 4) ? zhi : zlo;
		const float wx = o2w[0] * x + o2w[1] * y + o2w[2]  * z + o2w[3];
		const float wy = o2w[4] * x + o2w[5] * y + o2w[6]  * z + o2w[7];
		const float wz = o2w[8] * x + o2w[9] * y + o2w[10] * z + o2w[11];
		if (first) { lox = hix = wx; loy = hiy = wy; loz = hiz = wz; first = false; continue; }
		lox = fminf(lox, wx); hix = fmaxf(hix, wx);
		loy = fminf(loy, wy); hiy = fmaxf(hiy, wy);
		loz = fminf(loz, wz); hiz = fmaxf(hiz, wz);
	}
	box.minX = lox; box.minY = loy; box.minZ = loz;
	box.maxX = hix; box.maxY = hiy; box.maxZ = hiz;
	return box;
}

// Shared by uploadSkyLight()/uploadPortalLight() below (both just free any
// previously-uploaded buffer, then malloc+memcpy the new one, skipped
// entirely for an empty source so dst stays null - matches
// GpuSkyDistribution::height<=0/GpuPortalLight::height<=0's own "absent"
// convention). Hoisted to file scope, same reasoning as
// wf_disk_cylinder_world_aabb above, so both methods can share one copy
// instead of each keeping its own local lambda. Templated (not float-only)
// so it also serves portalSatSum's double precision (SummedAreaTable's own
// comment on why that one stays double, not narrowed to float like every
// other GPU buffer here).
template <typename VecT>
static void wf_upload_gpu_buf(const VecT& src, CUdeviceptr& dst) {
	using ElemT = typename VecT::value_type;
	if (dst) { cudaFree(reinterpret_cast<void*>(dst)); dst = 0; }
	if (src.empty()) return;
	const size_t bytes = src.size() * sizeof(ElemT);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), bytes));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src.data(), bytes, cudaMemcpyHostToDevice));
}
