// wavefront_kernels_restir.cu
// CUDA compute kernel: restir_spatial_reuse - ReSTIR DI's spatial (cross-
// pixel) reservoir reuse pass for the wavefront GPU path tracer's real-time
// Live Preview (gpu/optix/wavefront_restir_helpers.h has the full picture).
//
// A SEPARATE kernel, indexed by PIXEL (not hit-queue position) - unlike
// evaluate_materials/_simple/_dielectric (Kernel 2's family), which process
// a QUEUE of compacted hits, so two adjacent entries there are NOT screen-
// space neighbors. Spatial reuse fundamentally needs the "which pixels are
// near which other pixels" relationship, so it must run over the full
// per-pixel reservoir buffer, after every evaluate_materials* launch this
// frame has already finished writing it - hence its own kernel, run once per
// render() call (WavefrontPathTracer::render(), after the whole sampleIdx
// loop), not once per hit.
//
// Reads WavefrontPathTracer::d_reservoirs_ (this call's fresh-plus-temporal
// reservoirs, one per pixel) + d_restirNormal_ + d_worldPos_ (both also this
// call's own, written by evaluate_materials* at depth==0), and writes into
// d_reservoirsHistory_ - which is also what temporal reuse reads as "last
// call's result" at the START of the NEXT render() call (see
// wavefront_path_tracer.cpp's own end-of-render() comment). This pixel's own
// reservoir is never directly re-shaded with its spatially-combined result
// THIS frame - only next frame's temporal reuse sees it - a direct
// consequence of spatial reuse needing this separate full-image pass: by the
// time it runs, the original per-thread OptiX hit context (material type,
// throughput, wavelengths, BSDF...) that evaluate_materials* had is long
// gone, so there is nothing here to re-shade WITH. This still gives real
// spatial diffusion of light samples across the image, just with a one-frame
// delay - imperceptible at Live Preview's frame rate, and still fully
// unbiased (every combine step re-evaluates the neighbor's target function
// fresh, at THIS pixel's own point - restir_reservoir_combine's own comment).

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

extern "C" __global__ void restir_spatial_reuse(
	const GpuReservoir* currentReservoirs,
	const float3*       currentNormals,
	const float4*       currentWorldPos,
	GpuReservoir*       outputReservoirs,
	int width, int height,
	unsigned int frameSeed,
	const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
	const MaterialData* materials, const TextureData* textures, const unsigned char* texturePixels
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	const float4 wp = currentWorldPos[idx];
	if (wp.w == 0.0f) {
		// This pixel had no depth==0 non-specular hit this frame (a miss, or
		// a specular material) - nothing to spatially combine here; pass its
		// (necessarily invalid/empty) reservoir through unchanged.
		outputReservoirs[idx] = currentReservoirs[idx];
		return;
	}
	const float3 hitPoint = make_float3(wp.x, wp.y, wp.z);
	const float3 normal = currentNormals[idx];
	const int px = idx % width;
	const int py = idx / width;

	GpuReservoir result = currentReservoirs[idx];
	unsigned int seed = wf_pcg(wf_pcg((unsigned int)idx) ^ frameSeed);

	for (int i = 0; i < kRestirSpatialNeighbors; ++i) {
		// Uniform sample within a disk of radius kRestirSpatialRadiusPixels
		// (SampleUniformDiskConcentric would be the textbook choice, but a
		// plain polar sample is simpler and this file's neighbor set doesn't
		// need the low-discrepancy properties a concentric mapping buys
		// elsewhere in this codebase - k is small and redrawn every frame).
		const float r = kRestirSpatialRadiusPixels * sqrtf(wf_rand(seed));
		const float theta = 6.283185307179586f * wf_rand(seed);
		const int nx = px + (int)(r * cosf(theta));
		const int ny = py + (int)(r * sinf(theta));
		if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
		const int nIdx = ny * width + nx;
		if (nIdx == idx) continue;

		const float4 nWp = currentWorldPos[nIdx];
		if (nWp.w == 0.0f) continue;  // neighbor had no valid hit this frame

		// Neighbor-rejection: reject a neighbor whose shading normal has
		// drifted too far from this pixel's own - the standard "don't blend
		// across a geometric edge" guard (this file's own header comment).
		const float3 nNormal = currentNormals[nIdx];
		if (dot(normal, nNormal) < kRestirSpatialNormalCosThreshold) continue;

		const GpuReservoir& neighbor = currentReservoirs[nIdx];
		if (!neighbor.valid()) continue;

		// Re-evaluate the neighbor's stored sample's geometry AND target
		// function fresh, AT THIS PIXEL's own hitPoint/normal - restir.h's
		// documented missing piece for unbiased reuse (wavefront_restir_
		// helpers.h's own header comment).
		float3 dirToSample; float dist; float geomPdf;
		if (!wf_reevaluate_light_geometry(neighbor.sample, hitPoint, spheres, quads, triangles,
										   bilinearPatches, disks, cylinders, dirToSample, dist, geomPdf) ||
			geomPdf <= 0.0f)
			continue;

		// Robustness guard: reject a neighbor whose light-sample geometry is
		// wildly different as seen from this pixel vs. the neighbor's own
		// (e.g. steeply grazing here but not there) - see wf_restir_jacobian's
		// own comment for why this is a variance guard, not a second
		// unbiasing correction on top of the re-evaluation above.
		const float3 nHitPoint = make_float3(nWp.x, nWp.y, nWp.z);
		const float3 nToSample = neighbor.sample.point - nHitPoint;
		const float nDist = sqrtf(fmaxf(dot(nToSample, nToSample), 1e-12f));
		const float3 nDir = nToSample / nDist;
		const float jacobian = wf_restir_jacobian(dirToSample, dist, nDir, nDist, neighbor.sample.normal);
		if (jacobian < 0.1f || jacobian > 10.0f) continue;

		const float3 rawEmission = wf_light_raw_emission(neighbor.sample, dirToSample, materials, spheres, quads,
														  triangles, bilinearPatches, disks, cylinders,
														  textures, texturePixels);
		const float pHatAtCurrent = wf_restir_target_proxy(rawEmission, dirToSample, normal);
		if (pHatAtCurrent <= 0.0f) continue;

		restir_reservoir_combine(result, neighbor, pHatAtCurrent, wf_rand(seed));
	}

	if (result.M > kRestirSpatialMaxM) result.M = kRestirSpatialMaxM;
	restir_finalize(result);
	outputReservoirs[idx] = result;
}
