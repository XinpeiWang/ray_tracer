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
#include "wavefront_restir_gi_math.h"
#include "wavefront_svgf_math.h"  // wf_checkerboard_pixel_active
#include "probe_grid_types.h"     // GpuProbe/GpuProbeGridMeta - probe_cache_accumulate below
#include "wavefront_guiding.h"    // GpuGuidingHistogram/wf_guiding_accumulate - probe_cache_accumulate below

// Uniform sample within a disk of radius kRestirSpatialRadiusPixels around
// pixel (px, py) (SampleUniformDiskConcentric would be the textbook choice,
// but a plain polar sample is simpler and this file's neighbor set doesn't
// need the low-discrepancy properties a concentric mapping buys elsewhere in
// this codebase - k is small and redrawn every frame). Returns the flat
// neighbor index, or -1 if the sampled pixel falls outside [0,width)x
// [0,height) or lands back on (px, py) itself. Shared by DI's own
// restir_spatial_reuse below and GI's own restir_gi_spatial_reuse (this
// file) instead of each keeping its own copy of this identical loop body.
__device__ __forceinline__ int wf_restir_pick_spatial_neighbor(
		int px, int py, int width, int height, unsigned int& seed) {
	const float r = kRestirSpatialRadiusPixels * sqrtf(wf_rand(seed));
	const float theta = 6.283185307179586f * wf_rand(seed);
	const int nx = px + (int)(r * cosf(theta));
	const int ny = py + (int)(r * sinf(theta));
	if (nx < 0 || nx >= width || ny < 0 || ny >= height) return -1;
	const int nIdx = ny * width + nx;
	const int idx = py * width + px;
	if (nIdx == idx) return -1;
	return nIdx;
}

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
	// worldPos.w==1.0 only means "some depth==0 hit happened here" - it is
	// written unconditionally for EVERY depth==0 hit including specular
	// materials and BSSRDF/Subsurface exits (which never populate a
	// reservoir/normal for this pixel at all - see wf_finish_material_
	// scatter's own restirReservoirs/restirCtx parameter comments). A real
	// ReSTIR-eligible hit always writes a genuine (non-degenerate) shading
	// normal, so a near-zero one here reliably means "no reservoir was ever
	// written for this pixel this frame" regardless of why - skip the whole
	// neighbor loop rather than running it against an empty reservoir that
	// the previous (worldPos-only) check let through.
	if (dot(normal, normal) < 1e-8f) {
		outputReservoirs[idx] = currentReservoirs[idx];
		return;
	}
	const int px = idx % width;
	const int py = idx / width;

	GpuReservoir result = currentReservoirs[idx];
	unsigned int seed = wf_pcg(wf_pcg((unsigned int)idx) ^ frameSeed);

	for (int i = 0; i < kRestirSpatialNeighbors; ++i) {
		const int nIdx = wf_restir_pick_spatial_neighbor(px, py, width, height, seed);
		if (nIdx < 0) continue;

		const float4 nWp = currentWorldPos[nIdx];
		if (nWp.w == 0.0f) continue;  // neighbor had no valid hit this frame

		// Neighbor-rejection: reject a neighbor whose shading normal has
		// drifted too far from this pixel's own - the standard "don't blend
		// across a geometric edge" guard (this file's own header comment).
		const float3 nNormal = currentNormals[nIdx];
		if (dot(normal, nNormal) < kRestirSpatialNormalCosThreshold) continue;

		GpuReservoir neighbor = currentReservoirs[nIdx];
		if (!neighbor.valid()) continue;
		// Clamp the neighbor's M before folding it in, not `result.M` after
		// the loop (as this code used to) - see wf_restir_temporal_combine's
		// own comment (wavefront_restir_helpers.h) for why post-hoc clamping
		// the sum (instead of the incoming candidate's own M) inflates W and
		// compounds across frames instead of just bounding staleness.
		if (neighbor.M > kRestirSpatialMaxM) neighbor.M = kRestirSpatialMaxM;

		// Re-evaluate the neighbor's stored sample's geometry AND target
		// function fresh, AT THIS PIXEL's own hitPoint/normal - restir.h's
		// documented missing piece for unbiased reuse (wavefront_restir_
		// helpers.h's own header comment).
		float3 dirToSample; float dist; float geomPdf;
		// neighbor.sample.time carries the neighbor's own actual draw time
		// (GpuLightSample::time's own comment) - no separate per-pixel
		// shutter-time buffer needed to recover it.
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

	restir_finalize(result);
	outputReservoirs[idx] = result;
}

// Per-frame clear of WavefrontPathTracer::d_reservoirs_ - a proper per-struct
// reinitialization instead of a raw cudaMemsetAsync zero-fill. A memset gives
// every reservoir's sample.lightIdx == 0, NOT GpuLightSample's documented -1
// "invalid" sentinel (optix_types.h) - harmless today only because every
// current consumer checks GpuReservoir::valid() (which also requires
// weightSum > 0.0f, still false after either a memset or this kernel), never
// GpuLightSample::valid() in isolation - but a genuine, avoidable mismatch
// between the documented sentinel and the actual cleared state. This kernel
// writes the real GpuReservoir{} default (lightIdx=-1) instead.
//
// Checkerboard temporal upsampling (WavefrontPathTracer::render()'s own
// checkerboardActive comment): skips clearing a checkerboard-inactive
// pixel's reservoir, exactly mirroring the fix restir_gi_finalize's own
// comment describes for ReSTIR GI - ReSTIR DI is unconditionally enabled for
// every Live Preview call with no way to disable it, and without this an
// inactive pixel's d_reservoirs_ entry would be wiped to empty every other
// frame, then passed straight through by restir_spatial_reuse's own
// zero-normal early-return (d_restirNormal_ is unconditionally cleared every
// call, so an inactive pixel's normal always reads as zero) into
// d_reservoirsHistory_ - defeating DI's own temporal accumulation for half
// the image every frame. Holding the reservoir here (skip the clear
// entirely, leaving whatever this pixel's own d_reservoirs_ entry already
// held from the last render() call it was active in) is sufficient - no
// separate reprojection is needed the way SVGF/GI's OWN per-frame-rewritten
// state needed, since d_reservoirs_ is already an incrementally-updated
// "current best estimate" buffer, not per-frame scratch.
extern "C" __global__ void restir_clear_reservoirs(
		GpuReservoir* reservoirs, int width, int height,
		unsigned int frameNumber, bool checkerboardActive) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;
	const int px = idx % width;
	const int py = idx / width;
	if (!wf_checkerboard_pixel_active(px, py, frameNumber, checkerboardActive)) return;
	reservoirs[idx] = GpuReservoir{};
}

// ===========================================================================
// ReSTIR GI (gpu/optix/wavefront_restir_gi_math.h) - restir_gi_finalize and
// restir_gi_spatial_reuse below. MVP scope: Lambertian x0 only (see
// wf_finish_material_scatter's own giOriginContext-stash comment,
// wavefront_device_helpers.h, for why - the other non-specular materials'
// own BSDF evaluation is a capturing lambda that can't be re-evaluated
// outside that function's call frame without a much larger refactor).
// ===========================================================================

// Runs once per sample, per pixel, right after the CURRENT sample's depth==1
// shadow rays have resolved (WavefrontPathTracer::render()'s per-depth loop -
// unlike DI's spatial reuse, which runs once per whole render() call, this
// must run once per SAMPLE, since d_giOriginContext_/d_giCandidateOut_ are
// this SAMPLE's own scratch, about to be overwritten by the next sampleIdx
// iteration's own depth-0/depth-1 processing).
//
// Builds this pixel's fresh M=1 RIS candidate from originContext[idx]/
// candidateIn[idx] (both already fully known - see those buffers' own
// comments for why), folds in the reprojected previous-frame GI reservoir
// (temporal reuse, same reprojection/disocclusion approach as DI's own
// wf_restir_temporal_combine, wavefront_restir_helpers.h - reusing the SAME
// worldPosHistory/prevCamera DI's own temporal reuse already maintains,
// since those only ever depend on x0's position and the camera, never on
// which ReSTIR technique is consuming them), finalizes, and immediately
// shades: evaluates x0's own (Lambertian-only) BSDF in the direction toward
// the winning reservoir's x1Point and adds the result to the real
// framebuffer - mirroring DI's own final-shading formula shape exactly
// (wf_finish_material_scatter's `Ld = mis_w * bsdf_val * cos_l * nee_norm *
// throughput * bsdf_color * light_emission_spec`), with no MIS weight here
// (GI has no competing BSDF-sampling estimator for the SAME vertex the way
// DI's NEE has to share with continuation-ray light hits - GI's own
// candidate IS the continuation ray).
// Checkerboard temporal upsampling (WavefrontPathTracer::render()'s own
// checkerboardActive comment): a checkerboard-inactive pixel never gets a
// primary ray this frame, so originContext[idx] is never populated and
// ctx.valid() is false for it - exactly the same shape as a genuine
// specular/miss/non-Lambertian hit that ALSO leaves originContext[idx]
// invalid on an ACTIVE frame. Those two causes must be told apart:
// specular/miss/non-Lambertian is a real "no GI candidate here right now"
// (existing behavior: wipe to an empty reservoir, correct - GI has nothing
// to show for this vertex this frame). Checkerboard-inactive is NOT that -
// this pixel's reservoir would otherwise be wiped to empty every other
// frame, defeating GI's own temporal accumulation for half the image
// (ReSTIR GI is unconditionally enabled for every Live Preview call, so
// this isn't a rare combination to guard against). weightBuffer[idx]==0.0f
// (the same "was this pixel sampled this frame" signal svgf_temporal_
// integrate uses) distinguishes the two; on that path, reproject via
// currentWorldPos[idx] (held over from this pixel's last active frame - see
// svgf_checkerboard_clear_frame's own comment) and carry the reprojected
// reservoir through UNCHANGED - no combine, since there is no fresh
// candidate to fold in.
extern "C" __global__ void restir_gi_finalize(
	const GpuGiOriginContext* originContext,
	const GpuGiSample*        candidateIn,
	const GpuGiReservoir*     history,
	const float4*             worldPosHistory,
	const float4*             currentWorldPos,
	const float*              weightBuffer,
	GpuReprojectBasis         prevCamera,
	bool                      historyValid,
	int                       imageWidth, int imageHeight,
	unsigned int              frameSeed,
	const MaterialData*       materials,
	float3*                   framebuffer,
	float                     maxComponentValue,
	GpuGiReservoir*           outputReservoirs
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = imageWidth * imageHeight;
	if (idx >= numPixels) return;

	const GpuGiOriginContext& ctx = originContext[idx];
	if (!ctx.valid()) {
		// weightBuffer[idx]==0.0f alone isn't exclusively a checkerboard
		// signal - a crop/pixelbounds-excluded pixel (gpu_in_crop(),
		// generate_camera_rays) reads the same way. The wp.w!=0.0f check
		// just below is what actually makes this safe: a crop-excluded
		// pixel is never dispatched a primary ray on ANY frame, so
		// currentWorldPos[idx].w stays permanently 0.0f for it and this
		// branch can never fire for one - it always falls through to the
		// unconditional wipe below, exactly like the old, pre-checkerboard
		// behavior (see svgf_temporal_integrate's own identical comment,
		// wavefront_kernels_svgf.cu, for the fuller version of this
		// reasoning).
		if (weightBuffer[idx] == 0.0f && historyValid) {
			GpuGiReservoir held;
			if (wf_checkerboard_try_hold(currentWorldPos[idx], prevCamera, worldPosHistory, imageWidth, imageHeight, history, held)) {
				outputReservoirs[idx] = held;
				return;
			}
		}
		outputReservoirs[idx] = GpuGiReservoir{};
		return;
	}

	unsigned int seed = wf_pcg(wf_pcg((unsigned int)idx) ^ frameSeed);

	GpuGiReservoir res;
	const GpuGiSample& cand = candidateIn[idx];
	{
		const float pHat = wf_restir_target_proxy(cand.radiance, ctx.dirX0ToX1, ctx.x0Normal);
		const float risWeight = (ctx.pdfAtX0 > 0.0f) ? (pHat / ctx.pdfAtX0) : 0.0f;
		restir_reservoir_add(res, cand, risWeight, 1, pHat, wf_rand(seed));
	}

	// Temporal reuse - reprojection/disocclusion via the same shared helper
	// DI's own wf_restir_temporal_combine uses (wavefront_restir_helpers.h),
	// instead of a second hand-duplicated copy of that logic.
	if (historyValid) {
		const int prevPixel = wf_restir_reproject_prev_pixel(ctx.x0Point, prevCamera, worldPosHistory,
															   imageWidth, imageHeight);
		if (prevPixel >= 0) {
			GpuGiReservoir prev = history[prevPixel];
			if (prev.valid()) {
				// Clamp the INCOMING reservoir's M before folding it
				// in, not the combined sum afterward - see DI's own
				// wf_restir_temporal_combine comment
				// (wavefront_restir_helpers.h) for exactly why the
				// other ordering silently inflates W and compounds
				// across frames instead of just bounding staleness.
				if (prev.M > kRestirTemporalMaxM) prev.M = kRestirTemporalMaxM;
				const float3 dirToPrevX1 = normalize(prev.sample.x1Point - ctx.x0Point);
				const float freshPHat = wf_restir_target_proxy(prev.sample.radiance, dirToPrevX1, ctx.x0Normal);
				if (freshPHat > 0.0f) {
					const float jacobian = wf_restir_gi_jacobian(ctx.x0Point, prev.sample);
					restir_reservoir_combine(res, prev, freshPHat * jacobian, wf_rand(seed));
				}
			}
		}
	}

	restir_finalize(res);
	outputReservoirs[idx] = res;

	if (!(res.valid() && res.W > 0.0f)) return;

	const float3 dirToWinner = res.sample.x1Point - ctx.x0Point;
	const float distToWinnerSq = dot(dirToWinner, dirToWinner);
	if (distToWinnerSq < 1e-12f) return;
	const float distToWinner = sqrtf(distToWinnerSq);
	const float3 dir = dirToWinner / distToWinner;
	const float cosTheta = fmaxf(dot(dir, ctx.x0Normal), 0.0f);
	if (cosTheta <= 0.0f) return;

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SWL swl;
	SS throughputX0;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = ctx.wavelengths[i];
		swl.pdf[i]    = ctx.wavelength_pdfs[i];
		throughputX0[i] = ctx.throughputX0[i];
	}

	// Defensive re-check of the Lambertian-only MVP scope this whole function
	// silently assumes below (the treatment of `albedo` as the COMPLETE BSDF
	// value, `albedo/pi`, is only correct for Lambertian): the only place
	// that scope is actually ENFORCED is the depth==0 stash gate in
	// wf_finish_material_scatter (`matType == MaterialType::Lambertian`,
	// wavefront_device_helpers.h), which this function has no direct way to
	// see - it only gets a materialIdx. If that gate is ever widened without
	// also updating the shading formula below, this re-check turns what
	// would otherwise be silently wrong shading (a mismatched BRDF for a
	// material this function was never taught to evaluate) into a clean
	// "no GI contribution this pixel" instead - the same safe fallback every
	// other MVP-scope exclusion in this feature already uses (a glowing or
	// specular x1, a BSSRDF exit that never populated a candidate).
	if (materials[ctx.materialIdx].type != MaterialType::Lambertian) return;

	// Lambertian albedo uplift - mirrors wavefront_kernels_materials.cu's own
	// albedoSpectrum lambda exactly (clamp to [0,1], no 2*max rescale, unlike
	// the unbounded uplift just below). MVP scope: flat (non-textured)
	// albedo only - x0's UV isn't stashed in GpuGiOriginContext, so a
	// textured Lambertian's per-point albedo can't be reproduced here; a
	// documented, bounded imprecision (wrong tint on a textured surface's own
	// GI contribution), not a correctness hazard the way this project's
	// earlier ReSTIR bugs were.
	const float3 albedoRgb = materials[ctx.materialIdx].albedo;
	SS albedoSpec(0.f);
	{
		const float r = albedoRgb.x < 0.f ? 0.f : (albedoRgb.x > 1.f ? 1.f : albedoRgb.x);
		const float g = albedoRgb.y < 0.f ? 0.f : (albedoRgb.y > 1.f ? 1.f : albedoRgb.y);
		const float b = albedoRgb.z < 0.f ? 0.f : (albedoRgb.z > 1.f ? 1.f : albedoRgb.z);
		float c0, c1, c2;
		dev_srgb_to_coeffs(r, g, b, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		for (int i = 0; i < kWFNWavelengths; ++i) albedoSpec[i] = poly(swl.lambda[i]);
	}

	// Cached radiance uplift - an UNBOUNDED positive RGB value (not a [0,1]
	// reflectance), same normalize-by-2*max/uplift/rescale technique as
	// wavefront_kernels_materials.cu's own unboundedSpectrum lambda (NOT
	// wf_finish_material_scatter's liftEmission, which additionally applies a
	// D65 illuminant factor appropriate for a light's own color swatch, not a
	// value that is already a computed linear-RGB radiance).
	const float3 rad = res.sample.radiance;
	const float radMax = fmaxf(rad.x, fmaxf(rad.y, rad.z));
	if (radMax <= 0.0f) return;
	const float sc = 2.0f * radMax;
	SS radianceSpec(0.f);
	{
		float c0, c1, c2;
		dev_srgb_to_coeffs(rad.x / sc, rad.y / sc, rad.z / sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		for (int i = 0; i < kWFNWavelengths; ++i) radianceSpec[i] = sc * poly(swl.lambda[i]);
	}

	const SS Ld = (cosTheta * res.W * (1.0f / 3.14159265f)) * throughputX0 * albedoSpec * radianceSpec;

	auto xyz = SampledSpectrumToXYZ(Ld, swl, d_cie_x, d_cie_y, d_cie_z, kDevCIEMin, kDevCIENSamples);
	float rr, gg, bb;
	wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, rr, gg, bb);
	const float m = fmaxf(rr, fmaxf(gg, bb));
	if (maxComponentValue > 0.0f && m > maxComponentValue) {
		const float s2 = maxComponentValue / m;
		rr *= s2; gg *= s2; bb *= s2;
	}
	atomicAdd(&framebuffer[idx].x, rr);
	atomicAdd(&framebuffer[idx].y, gg);
	atomicAdd(&framebuffer[idx].z, bb);
}

// GI's own spatial reuse - same one-frame-delay design as DI's own
// restir_spatial_reuse above (see that kernel's own header comment for why:
// by the time this runs, there is no per-thread shading context left to
// re-shade WITH this frame, only next frame's temporal reuse ever sees this
// pass's own output). Reuses originContext[idx].x0Point/x0Normal for both
// the neighbor-rejection test and the Jacobian/pHat re-evaluation, instead
// of a separate current-frame normal buffer - see GpuGiOriginContext's own
// comment for why it already carries everything needed.
extern "C" __global__ void restir_gi_spatial_reuse(
	const GpuGiReservoir*      currentReservoirs,
	const GpuGiOriginContext*  originContext,
	GpuGiReservoir*            outputReservoirs,
	int width, int height,
	unsigned int frameSeed
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	const GpuGiOriginContext& ctx = originContext[idx];
	if (!ctx.valid()) {
		outputReservoirs[idx] = currentReservoirs[idx];
		return;
	}

	const int px = idx % width;
	const int py = idx / width;
	GpuGiReservoir result = currentReservoirs[idx];
	unsigned int seed = wf_pcg(wf_pcg((unsigned int)idx) ^ frameSeed);

	for (int i = 0; i < kRestirSpatialNeighbors; ++i) {
		const int nIdx = wf_restir_pick_spatial_neighbor(px, py, width, height, seed);
		if (nIdx < 0) continue;

		const GpuGiOriginContext& nCtx = originContext[nIdx];
		if (!nCtx.valid()) continue;
		if (dot(ctx.x0Normal, nCtx.x0Normal) < kRestirSpatialNormalCosThreshold) continue;

		GpuGiReservoir neighbor = currentReservoirs[nIdx];
		if (!neighbor.valid()) continue;
		// Clamp the neighbor's M before folding it in, not `result.M` after
		// the loop - see restir_gi_finalize's own temporal-reuse comment
		// (and DI's own wf_restir_temporal_combine comment,
		// wavefront_restir_helpers.h) for why the other ordering inflates W.
		if (neighbor.M > kRestirSpatialMaxM) neighbor.M = kRestirSpatialMaxM;

		const float3 dirToSample = normalize(neighbor.sample.x1Point - ctx.x0Point);
		const float freshPHat = wf_restir_target_proxy(neighbor.sample.radiance, dirToSample, ctx.x0Normal);
		if (freshPHat <= 0.0f) continue;

		// Same robustness guard DI's own spatial reuse uses (this file's
		// area-light spatial-reuse block above) - reject a neighbor whose
		// geometric configuration is too different to reuse with acceptable
		// variance, not a second unbiasing correction on top of the Jacobian.
		const float jacobian = wf_restir_gi_jacobian(ctx.x0Point, neighbor.sample);
		if (jacobian < 0.1f || jacobian > 10.0f) continue;

		restir_reservoir_combine(result, neighbor, freshPHat * jacobian, wf_rand(seed));
	}

	restir_finalize(result);
	outputReservoirs[idx] = result;
}

// ============================================================================
// probe_cache_shade -- world-space irradiance probe cache (Live Preview
// only), shading half of the update ray. Consumes ProbeCacheHitWorkItems
// produced by __raygen__wf_probe_cache (wavefront_probe_cache.h, a SEPARATE
// OptiX-module translation unit that can only trace intersections, not call
// wf_generate_restir_candidate() - see that raygen's own header comment for
// the full cross-translation-unit rationale). This kernel lives in THIS file
// (not wavefront_kernels_materials.cu) because it, like restir_spatial_reuse
// above, is launched once per render() call rather than once per bounce.
//
// Deliberately simplified relative to wf_finish_material_scatter's own
// classic/ReSTIR NEE block: Lambertian-albedo-only BSDF (this project's own
// plan, ADR-equivalent to ReSTIR GI's own Lambertian-x0 MVP restriction) - a
// probe's SH-L1 storage is already a coarse, blurry approximation, so a
// glossy hit's own specular lobe would be lost in that blur anyway. Writes
// the hit surface's own emission directly into probeCacheRadianceOut (no
// occlusion to test for a surface lighting itself) and, for a valid one-
// light NEE draw, pushes a real spectral ShadowRayWorkItem into the ordinary
// shadow queue/pipeline for a fully correct occlusion test - no approximate
// second trace, no hand-rolled per-light-kind sampling: this reuses the
// exact same wf_generate_restir_candidate()/shadow-pipeline machinery every
// other NEE draw in this codebase already goes through.
// ============================================================================
extern "C" __global__ void probe_cache_shade(
	WorkQueue<ProbeCacheHitWorkItem> probeCacheHitQueue,
	int                          numProbeCacheHits,
	const MaterialData*          materials,
	const SphereData*            spheres,
	const QuadData*              quads,
	const TriangleData*          triangles,
	const BilinearPatchData*     bilinearPatches,
	const DiskData*              disks,
	const CylinderData*          cylinders,
	const TextureData*           textures,
	const unsigned char*         texturePixels,
	const int*                   lightIndices,
	const GpuLightKind*          lightKinds,
	const GpuAliasEntry*         aliasTable,
	unsigned int                 numLights,
	WfLightBvhContext            lightBvh,
	float3                       backgroundColor,
	float                        shadowEps,
	float3*                      probeCacheRadianceOut,  // [kProbesPerFrame], indexed by batchSlot
	float*                       probeCacheHitDistOut,   // [kProbesPerFrame], indexed by batchSlot
	WorkQueue<ShadowRayWorkItem> shadowQueue
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numProbeCacheHits || idx >= probeCacheHitQueue.capacity) return;

	const ProbeCacheHitWorkItem& item = probeCacheHitQueue.items[idx];
	probeCacheHitDistOut[item.batchSlot] = item.hitDist;

	if (!item.hit) {
		// Miss: treat as a flat-background contribution (same backgroundColor
		// every other escaped ray in this codebase adds - see accumulate_
		// miss's own comment for why a flat color, not a real importance-
		// sampled sky, matches what the CPU renderer does). No occlusion test
		// needed for a ray that hit nothing.
		atomicAdd(&probeCacheRadianceOut[item.batchSlot].x, backgroundColor.x);
		atomicAdd(&probeCacheRadianceOut[item.batchSlot].y, backgroundColor.y);
		atomicAdd(&probeCacheRadianceOut[item.batchSlot].z, backgroundColor.z);
		return;
	}

	const MaterialData& mat = materials[item.materialIdx];
	// A hit surface's own emission (DiffuseLight) - no occlusion to test,
	// this ray already IS at the emitting surface.
	atomicAdd(&probeCacheRadianceOut[item.batchSlot].x, mat.emission.x);
	atomicAdd(&probeCacheRadianceOut[item.batchSlot].y, mat.emission.y);
	atomicAdd(&probeCacheRadianceOut[item.batchSlot].z, mat.emission.z);

	unsigned int seed = item.seed;
	GpuLightSample cand;
	float3 toLight = make_float3(0.0f, 0.0f, 0.0f);
	float  maxDist = 0.0f, lightPdf = 0.0f;
	float3 rawEmission = make_float3(0.0f, 0.0f, 0.0f);
	if (!wf_generate_restir_candidate(item.hitPoint, seed, /*time=*/0.0f,
			spheres, quads, triangles, bilinearPatches, disks, cylinders,
			materials, lightIndices, lightKinds, aliasTable, numLights,
			textures, texturePixels, cand, toLight, maxDist, lightPdf, rawEmission,
			lightBvh)) {
		return;
	}
	const float cosTheta = dot(item.hitNormal, toLight);
	if (!(cosTheta > 0.0f) || !(lightPdf > 1e-6f)) return;

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	const SWL swl = SWL::SampleVisible(wf_rand(seed));

	const SS lightSpec  = wf_lift_rgb_to_spectrum(rawEmission, swl, /*isIlluminant=*/true);
	const SS albedoSpec = wf_lift_rgb_to_spectrum(mat.albedo, swl, /*isIlluminant=*/false);
	const float invPi = 1.0f / 3.14159265f;
	const SS Ld = (cosTheta * invPi / lightPdf) * albedoSpec * lightSpec;
	if (!(bool)Ld) return;

	// Same combined normal+direction shadow-ray-origin offset as wf_finish_
	// material_scatter's own classic NEE block - see that block's own long
	// comment (wavefront_device_helpers.h) for why both components matter.
	ShadowRayWorkItem sr;
	sr.origin    = item.hitPoint + shadowEps * item.hitNormal + shadowEps * normalize(toLight);
	sr.direction = toLight;
	sr.tMax      = maxDist - 0.002f;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		sr.Ld[i] = Ld[i];
		sr.wavelengths[i] = swl.lambda[i];
		sr.wavelength_pdfs[i] = swl.pdf[i];
	}
	sr.pixelIndex     = item.batchSlot;
	sr.time           = 0.0f;
	sr.isGiCandidate  = false;
	sr.seed           = seed;
	sr.isProbeCacheRay = true;
	shadowQueue.push(sr);
}

// ============================================================================
// probe_cache_accumulate -- EMA-blends this frame's resolved probe-update
// samples into the persistent GpuProbe array. Runs after accumulate_shadow
// has redirected each probe-update shadow ray's resolved Ld into
// probeCacheRadianceOut (ShadowRayWorkItem::isProbeCacheRay's own comment) -
// so by the time this launches, probeCacheRadianceOut[batchSlot] already
// holds the FULL resolved radiance (emission + occluded-tested NEE) for that
// batch slot's probe-update ray.
//
// One thread per this frame's batch slot (kProbesPerFrame threads, NOT one
// per probe in the grid) - see ProbeCacheRayWorkItem::batchSlot's own
// comment for the round-robin arithmetic recovering the real probe index.
// ============================================================================
extern "C" __global__ void probe_cache_accumulate(
	const float3*    probeCacheRadianceOut,  // [numBatchSlots]
	const float*     probeCacheHitDistOut,   // [numBatchSlots]
	const float3*    probeCacheDirections,   // [numBatchSlots] - this batch slot's own traced ray direction
	int              numBatchSlots,
	int              probeUpdateCursor,
	GpuProbeGridMeta gridMeta,
	GpuProbe*        probes,
	// Real-time path guiding (Live Preview only, gpu/optix/wavefront_guiding.h)
	// - nullptr (guiding disabled, or the probe cache's own totalProbes<=0
	// case) is a complete no-op, same "null pointer disables the whole
	// feature" shape as every other optional buffer in this codebase. When
	// non-null, sized gridMeta.totalProbes exactly like `probes` above -
	// SAME probe index, SAME already-traced direction/radiance this probe-
	// update ray already resolved for the SH-L1 projection above, at zero
	// extra ray-tracing cost (see this project's own plan).
	GpuGuidingHistogram* guidingHistograms = nullptr
) {
	const int batchSlot = blockIdx.x * blockDim.x + threadIdx.x;
	if (batchSlot >= numBatchSlots || gridMeta.totalProbes <= 0) return;

	const int probeIdx = (probeUpdateCursor + batchSlot) % gridMeta.totalProbes;
	GpuProbe& p = probes[probeIdx];

	const int kProbeHistoryCap = 64;
	const float alpha = 1.0f / (float)min(p.numRaysEverTraced + 1, kProbeHistoryCap);

	const float3 radiance = probeCacheRadianceOut[batchSlot];
	const float  hitDist  = probeCacheHitDistOut[batchSlot];
	const float3 d        = probeCacheDirections[batchSlot];

	float basis[4];
	wf_probe_sh_basis(d, basis);
	// wf_rand_unit()'s own distribution is uniform over the sphere, pdf =
	// 1/(4*pi) - dividing the projected sample by that pdf (equivalently,
	// multiplying by 4*pi) turns this single Monte-Carlo sample into an
	// unbiased estimator of the SH projection integral, matching this
	// project's own plan formula exactly.
	const float kInv4PiPdf = 4.0f * 3.14159265f;
	const float* radianceC[3] = {&radiance.x, &radiance.y, &radiance.z};
	float* shC[3] = {p.shR, p.shG, p.shB};
	for (int c = 0; c < 3; ++c) {
		for (int k = 0; k < 4; ++k) {
			const float sample = (*radianceC[c]) * basis[k] * kInv4PiPdf;
			shC[c][k] = shC[c][k] + alpha * (sample - shC[c][k]);
		}
	}
	// Skip the distance-EMA update entirely on a miss (hitDist < 0 - see
	// __raygen__wf_probe_cache's own -1.0f sentinel comment). A sky-miss has
	// no real "unoccluded reach" to report, and blending its old 1e4-style
	// placeholder in at the same weight as a real hit distance used to drag
	// meanDist toward that placeholder for any probe with a nontrivial miss
	// fraction (e.g. near an opening to the sky or a scene boundary),
	// inflating wf_query_probe_grid's own leak-test maxReach enough to
	// disable the leak guard entirely for exactly the probes nearest such
	// boundaries. The SH radiance/numRaysEverTraced update above still runs
	// unconditionally either way - a miss's background contribution is still
	// real, valid radiance data for the cache to learn from.
	if (hitDist >= 0.0f) {
		p.meanDist   = p.meanDist   + alpha * (hitDist - p.meanDist);
		p.meanDistSq = p.meanDistSq + alpha * (hitDist * hitDist - p.meanDistSq);
	}
	p.numRaysEverTraced += 1;

	// Real-time path guiding (Live Preview only) - bins this SAME already-
	// resolved sample into the matching probe's own directional histogram,
	// alongside (not instead of) the SH-L1 projection above. See
	// wavefront_guiding.h's own header comment for why this reuses the
	// probe-cache-update pipeline wholesale rather than a dedicated pass.
	if (guidingHistograms != nullptr) {
		wf_guiding_accumulate(guidingHistograms[probeIdx], d, wf_svgf_luminance(radiance));
	}
}
