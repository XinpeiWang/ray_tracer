// wavefront_kernels_svgf.cu
// CUDA compute kernels: SVGF (Schied et al. 2017, "Spatiotemporal Variance-
// Guided Filtering") for the wavefront GPU path tracer's real-time Live
// Preview. See this project's own SVGF plan for the full architecture
// writeup; summarized here:
//
//   svgf_temporal_integrate -> svgf_prepare_for_filter -> svgf_atrous_pass
//   (repeated with doubling step sizes, ping-ponging - pass count is a
//   host-side loop bound, WavefrontPathTracer::launchSvgf()'s own
//   svgfTuning_.atrousPasses (gpu/optix/svgf_tuning_params.h), not anything
//   in this file) -> svgf_finalize
//
// Runs once per render() call, after launchNormalizeFramebuffer has already
// produced this frame's final, normalized 1-spp radiance in the framebuffer
// - SVGF treats THAT as its raw noisy input, in place, overwriting it with
// the temporally-stable, spatially-filtered result by the time
// WavefrontPathTracer::render() returns (see launchSvgf()'s own comment,
// wavefront_path_tracer.h, for why this needs no new output parameter on
// rt_realtime_render_frame()).
//
// Reuses, rather than re-implements: wf_restir_reproject_prev_pixel
// (gpu/optix/wavefront_restir_helpers.h - the exact reprojection/
// disocclusion test DI's and GI's own temporal reuse already share) for
// this frame's own reprojection, and d_worldPos_/d_worldPosHistory_/
// d_restirNormal_ (generalized to be written whenever ANY of DI/GI/SVGF is
// enabled, not DI alone - see wavefront_path_tracer.cpp's own buffer-
// allocation gate comment) for the geometry every edge-stopping weight
// needs. wavefront_svgf_math.h supplies every pure formula (variance,
// temporal blend schedule, edge-stopping weights) - this file is purely the
// GPU-side plumbing around those formulas.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"
#include "wavefront_svgf_math.h"

// Formerly hardcoded `constexpr` literals here - now runtime parameters
// (gpu/optix/svgf_tuning_params.h's SvgfTuningParams, threaded through from
// WavefrontPathTracer::launchSvgf(), this project's own architecture-review
// follow-up). Each kernel below takes exactly the fields it needs as
// parameters instead of referencing a file-scope constant; the struct's own
// in-class defaults reproduce the original literature-default values
// (Schied et al. 2017 and the common reference implementations descended
// from it) exactly.
//
// kKernel[3] below (svgf_atrous_pass) is the one exception that stays a
// fixed-size compile-time table - see that kernel's own comment on why
// atrousRadius is clamped to [0,2].

// Derives this pixel's own linear "depth" (distance from the CURRENT
// camera) from the world-position buffer - no separate depth AOV exists or
// is needed (see this project's own SVGF plan for why). Returns false (hit
// == false) for a miss (worldPos.w == 0), matching every other consumer of
// this buffer's own validity convention.
__device__ __forceinline__ bool wf_svgf_depth_at(
		const float4* worldPos, int idx, const float3& cameraOrigin, float& outDepth) {
	const float4 wp = worldPos[idx];
	if (wp.w == 0.0f) { outDepth = 0.0f; return false; }
	const float3 delta = make_float3(wp.x, wp.y, wp.z) - cameraOrigin;
	outDepth = sqrtf(fmaxf(dot(delta, delta), 0.0f));
	return true;
}

// -----------------------------------------------------------------------
// Kernel 0 - svgf_checkerboard_clear_frame
// -----------------------------------------------------------------------
// Replaces the plain cudaMemsetAsync() this project's own render() used to
// run unconditionally, every call, on albedoAov/normalAov/worldPos - correct
// for every pixel when checkerboarding is off (byte-identical: every pixel
// is "active" per wf_checkerboard_pixel_active's own checkerboardActive==
// false fast path), but a checkerboard-INACTIVE pixel must instead survive
// this call untouched, holding whatever it was left at by the last render()
// call in which it WAS active - that held value is what makes the world-pos/
// albedo/normal buffers still valid inputs for svgf_temporal_integrate's own
// reprojection and the A-trous pass's edge-stopping weights on a frame this
// pixel isn't freshly traced. frameNumber passed in here is this render()
// call's frameNumber_ AFTER the increment generate_camera_rays will see
// later in the same call (this kernel runs earlier, in the pre-increment
// buffer-setup block) - see WavefrontPathTracer::render()'s own call site
// comment for why +1 is required to agree on parity.
// Each of the 3 buffers is independently nullable - callers pass whichever
// subset is actually allocated this call (denoise-only, or restirEnabled_/
// restirGiEnabled_ without SVGF, still only allocate a subset - see
// WavefrontPathTracer::render()'s own needsAovGuideBuffers/
// needsWorldPosHistory gates), unified into one kernel/one launch rather
// than 2 separate call sites that would otherwise need to stay in sync on
// which buffers each one is responsible for.
extern "C" __global__ void svgf_checkerboard_clear_frame(
	float3* albedoBuffer,
	float3* normalBuffer,
	float4* worldPosBuffer,
	int width, int height,
	unsigned int frameNumber,
	bool checkerboardActive
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;
	const int px = idx % width;
	const int py = idx / width;
	if (!wf_checkerboard_pixel_active(px, py, frameNumber, checkerboardActive)) return;
	if (albedoBuffer) albedoBuffer[idx] = make_float3(0.0f, 0.0f, 0.0f);
	if (normalBuffer) normalBuffer[idx] = make_float3(0.0f, 0.0f, 0.0f);
	if (worldPosBuffer) worldPosBuffer[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// -----------------------------------------------------------------------
// Kernel 1 - svgf_temporal_integrate
// -----------------------------------------------------------------------
// Reprojects each pixel into the previous frame (via the SAME shared helper
// DI/GI's own temporal reuse uses), blends this frame's raw radiance with
// the reprojected history's color/moments using wf_svgf_temporal_alpha's
// schedule, and writes this frame's own GpuSvgfState. A miss pixel (no
// world-space hit to reproject with) and any pixel that fails reprojection
// (disoccluded, off-screen, no history yet) simply starts fresh with
// historyLength=1 - deliberately NOT reprojected via plain screen-space
// pixel identity for misses, since a moving camera's sky/background would
// otherwise smear across frames exactly like a false "converged" surface.
//
// Checkerboard temporal upsampling (weightBuffer[idx]==0.0f - see
// wf_checkerboard_pixel_active's own comment for why this reuses
// generate_camera_rays' own per-pixel filter-weight accumulator as the "was
// this pixel sampled this frame" signal, rather than adding a second, easy-
// to-desync mask): currentRadiance[idx] is NOT a real sample for an inactive
// pixel (normalize_framebuffer already wrote a fabricated black there for
// weight==0 - see that kernel's own comment), so it must never be blended in
// as though it were. Instead, reproject via currentWorldPos[idx] (held over
// from this pixel's last active frame - see svgf_checkerboard_clear_frame's
// own comment) exactly as the active path does, and on success copy the
// reprojected history through UNCHANGED: no blend (zero new information
// arrived this frame) and historyLength NOT incremented (a held frame is not
// an additional independent sample). On reprojection failure - a pixel with
// no prior history yet, since checkerboarding only activates once both SVGF
// and GI history are already warm (WavefrontPathTracer::render()'s own
// checkerboardActive comment), this is a rare/defensive path, not steady
// state - fall through to historyLength=0 (NOT 1: nothing was actually
// sampled, so this must not look like a genuine fresh sample to the variance
// bootstrap in svgf_prepare_for_filter).
extern "C" __global__ void svgf_temporal_integrate(
	const float3* currentRadiance,
	const float4* currentWorldPos,
	const GpuSvgfState* history,
	const float4* worldPosHistory,
	const float* weightBuffer,
	GpuReprojectBasis prevCamera,
	bool historyValid,
	int width, int height,
	float temporalAlpha, float maxHistoryLength,
	GpuSvgfState* outputCurrent
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	// weightBuffer[idx]==0.0f is not EXCLUSIVELY a checkerboard-inactive
	// signal - a crop/pixelbounds-excluded pixel (gpu_in_crop(),
	// generate_camera_rays) never gets weightBuffer incremented either,
	// checkerboarding or not. This is safe regardless: a crop-excluded
	// pixel is never dispatched a primary ray on ANY frame, so it never
	// gets a real worldPos written either - currentWorldPos[idx].w stays
	// permanently 0.0f for it, which is exactly the SAME condition
	// (wp.w != 0.0f below) that already gates the reprojection/hold path a
	// checkerboard-inactive pixel needs. A crop-excluded pixel therefore
	// always falls through to the isActive==false, no-reprojection default
	// (historyLength=0) below - a harmless, permanent "no info" state for a
	// pixel that was never part of the rendered image region, and one that
	// svgf_prepare_for_filter's own variance-bootstrap neighbor loop already
	// excludes via that same wp.w==0.0f check when THIS pixel is someone
	// else's neighbor.
	const bool isActive = weightBuffer[idx] > 0.0f;
	const float3 color = isActive ? currentRadiance[idx] : make_float3(0.0f, 0.0f, 0.0f);
	const float lum = isActive ? wf_svgf_luminance(color) : 0.0f;

	GpuSvgfState result;
	result.color = color;
	result.moment1 = lum;
	result.moment2 = lum * lum;
	result.historyLength = isActive ? 1.0f : 0.0f;

	if (historyValid) {
		GpuSvgfState prev;
		if (wf_checkerboard_try_hold(currentWorldPos[idx], prevCamera, worldPosHistory, width, height, history, prev)) {
			if (isActive) {
				const float alpha = wf_svgf_temporal_alpha(prev.historyLength, temporalAlpha);
				result.color = prev.color + (color - prev.color) * alpha;
				result.moment1 = prev.moment1 + (lum - prev.moment1) * alpha;
				result.moment2 = prev.moment2 + (lum * lum - prev.moment2) * alpha;
				result.historyLength = fminf(prev.historyLength + 1.0f, maxHistoryLength);
			} else {
				// Hold: carry the reprojected history through unchanged - no
				// blend, since zero new information arrived this frame.
				result = prev;
			}
		}
	}

	outputCurrent[idx] = result;
}

// -----------------------------------------------------------------------
// Kernel 2 - svgf_prepare_for_filter
// -----------------------------------------------------------------------
// Reads THIS frame's just-written GpuSvgfState (safe: svgf_temporal_integrate
// has already fully completed and the launch synchronized - see
// launchSvgf()'s own comment), demodulates by albedo (dividing the
// illumination the spatial filter will smear by the surface's own albedo,
// so the filter never blurs across a genuine texture/albedo edge - the
// standard SVGF "denoise illumination, not final radiance" technique),
// and - for a pixel whose historyLength is still short (a fresh or recently
// disoccluded pixel, where the temporally-accumulated variance estimate
// alone is too noisy to be useful, per the paper's own Section 4.2) -
// spatially prefilters the variance with a small unweighted box average
// over neighboring pixels' own (already-written) moments first. Writes
// (demodulated color, prepared variance) into the first A-trous ping-pong
// buffer as float4 (xyz=color, w=variance).
extern "C" __global__ void svgf_prepare_for_filter(
	const GpuSvgfState* current,
	const float3* albedo,
	const float4* currentWorldPos,
	int width, int height,
	float varianceBootstrapFrames, int varianceBootstrapRadius, float minAlbedo,
	float4* outPingPong0
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	const GpuSvgfState s = current[idx];

	float variance;
	if (s.historyLength < varianceBootstrapFrames) {
		const int px = idx % width;
		const int py = idx / width;
		float sumMoment1 = 0.0f, sumMoment2 = 0.0f;
		int count = 0;
		for (int dy = -varianceBootstrapRadius; dy <= varianceBootstrapRadius; ++dy) {
			const int ny = py + dy;
			if (ny < 0 || ny >= height) continue;
			for (int dx = -varianceBootstrapRadius; dx <= varianceBootstrapRadius; ++dx) {
				const int nx = px + dx;
				if (nx < 0 || nx >= width) continue;
				const int nIdx = ny * width + nx;
				// Skip miss/background neighbors (worldPos.w == 0, same
				// validity convention wf_svgf_depth_at uses below) - a
				// silhouette-adjacent hit pixel's own variance bootstrap
				// must not be contaminated by unrelated sky/background
				// luminance statistics from the other side of the edge.
				if (currentWorldPos[nIdx].w == 0.0f) continue;
				const GpuSvgfState n = current[nIdx];
				// Skip a neighbor with historyLength==0 too - checkerboard
				// temporal upsampling can produce one: an inactive pixel
				// whose reprojection fails (freshly disoccluded while also
				// checkerboard-inactive this frame) keeps its OWN held,
				// still-nonzero worldPos (svgf_checkerboard_clear_frame only
				// clears active-this-frame slots) but gets a fabricated
				// all-zero color/moments (svgf_temporal_integrate's own
				// cold-start-when-inactive default) - a fake "valid, zero-
				// variance" neighbor the worldPos.w check above can't catch,
				// since its worldPos really is a genuine (just stale) hit.
				if (n.historyLength <= 0.0f) continue;
				sumMoment1 += n.moment1;
				sumMoment2 += n.moment2;
				++count;
			}
		}
		if (count > 0) {
			const float avgMoment1 = sumMoment1 / (float)count;
			const float avgMoment2 = sumMoment2 / (float)count;
			variance = wf_svgf_variance(avgMoment1, avgMoment2);
		} else {
			// Every neighbor (including self) is a miss - no hit-surface
			// statistics available to bootstrap from; fall back to this
			// pixel's own (short-history, noisy but not cross-contaminated)
			// moments rather than dividing by zero.
			variance = wf_svgf_variance(s.moment1, s.moment2);
		}
	} else {
		variance = wf_svgf_variance(s.moment1, s.moment2);
	}

	// Same floor used here (demodulate) and in svgf_finalize (remodulate) -
	// using the raw, un-floored albedo in one place and the floored value in
	// the other would round-trip a near-zero-albedo pixel (background/sky,
	// whose albedo AOV is never written) to zero regardless of its actual
	// filtered radiance.
	const float3 alb = albedo[idx];
	const float3 albedoFloor = make_float3(fmaxf(alb.x, minAlbedo), fmaxf(alb.y, minAlbedo), fmaxf(alb.z, minAlbedo));
	const float3 demodulated = make_float3(s.color.x / albedoFloor.x, s.color.y / albedoFloor.y, s.color.z / albedoFloor.z);

	outPingPong0[idx] = make_float4(demodulated.x, demodulated.y, demodulated.z, variance);
}

// -----------------------------------------------------------------------
// Kernel 3 - svgf_atrous_pass
// -----------------------------------------------------------------------
// One A-trous wavelet filter pass at the given step size - a small (default
// 5x5, atrousRadius=2 - see this kernel's own [0,2] clamp) footprint whose
// sample spacing is scaled by
// `stepSize`, so successive passes (stepSize 1,2,4,8) cover an exponentially
// growing effective radius without needing an exponentially growing kernel
// footprint (the "a trous" - "with holes" - trick the technique is named
// for). Depth/its screen-space gradient are derived fresh from
// currentWorldPos each pass (cheap: 4 extra buffer reads at the CENTER
// pixel, not per neighbor) rather than cached, matching the reference
// implementations' own approach. Filters the variance alongside the color
// using the same weights (squared, per the standard error-propagation
// formula for a weighted average's own variance - Schied et al.'s own
// supplementary material), not by recomputing it from the filtered color
// each pass.
extern "C" __global__ void svgf_atrous_pass(
	const float4* input,
	const float4* currentWorldPos,
	const float3* normals,
	float3 cameraOrigin,
	int width, int height,
	int stepSize,
	float sigmaNormal, float sigmaDepth, float sigmaLuminance, int atrousRadius,
	float4* output
) {
	// Defense-in-depth clamp: kKernel[3] below only has 3 entries, indexed by
	// abs(dx)/abs(dy) up to atrousRadius - the GUI's own spinbox range is the
	// primary guard (mainwindow_tabs.cpp), this just protects any other
	// caller (tests, future callers) from an out-of-bounds table read.
	atrousRadius = atrousRadius < 0 ? 0 : (atrousRadius > 2 ? 2 : atrousRadius);

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	const int px = idx % width;
	const int py = idx / width;

	const float4 centerIn = input[idx];
	const float3 centerColor = make_float3(centerIn.x, centerIn.y, centerIn.z);
	const float centerVariance = centerIn.w;
	const float3 centerNormal = normals[idx];
	float centerDepth;
	const bool centerHasDepth = wf_svgf_depth_at(currentWorldPos, idx, cameraOrigin, centerDepth);

	if (!centerHasDepth) {
		// A miss pixel (background/sky) - no meaningful geometry to compute
		// edge-stopping weights against; pass through unfiltered rather than
		// blending with unrelated neighboring surfaces' own depth/normal.
		output[idx] = centerIn;
		return;
	}

	// Screen-space depth gradient at the center pixel (central difference,
	// falling back to a one-sided difference at the image border) - see
	// wf_svgf_weight_depth's own comment for why this, not a fixed epsilon,
	// normalizes the raw depth difference term below.
	float depthGradX = 0.0f, depthGradY = 0.0f;
	{
		float dxPlus, dxMinus, dyPlus, dyMinus;
		const bool hasXPlus  = (px + 1 < width)  && wf_svgf_depth_at(currentWorldPos, py * width + (px + 1), cameraOrigin, dxPlus);
		const bool hasXMinus = (px - 1 >= 0)     && wf_svgf_depth_at(currentWorldPos, py * width + (px - 1), cameraOrigin, dxMinus);
		const bool hasYPlus  = (py + 1 < height) && wf_svgf_depth_at(currentWorldPos, (py + 1) * width + px, cameraOrigin, dyPlus);
		const bool hasYMinus = (py - 1 >= 0)     && wf_svgf_depth_at(currentWorldPos, (py - 1) * width + px, cameraOrigin, dyMinus);
		if (hasXPlus && hasXMinus)      depthGradX = (dxPlus - dxMinus) * 0.5f;
		else if (hasXPlus)              depthGradX = dxPlus - centerDepth;
		else if (hasXMinus)             depthGradX = centerDepth - dxMinus;
		if (hasYPlus && hasYMinus)      depthGradY = (dyPlus - dyMinus) * 0.5f;
		else if (hasYPlus)              depthGradY = dyPlus - centerDepth;
		else if (hasYMinus)             depthGradY = centerDepth - dyMinus;
	}
	const float centerLum = wf_svgf_luminance(centerColor);
	const float sqrtCenterVariance = sqrtf(centerVariance);

	float3 colorSum = make_float3(0.0f, 0.0f, 0.0f);
	float varianceSumSq = 0.0f;
	float weightSum = 0.0f;

	for (int dy = -atrousRadius; dy <= atrousRadius; ++dy) {
		const int ny = py + dy * stepSize;
		if (ny < 0 || ny >= height) continue;
		for (int dx = -atrousRadius; dx <= atrousRadius; ++dx) {
			const int nx = px + dx * stepSize;
			if (nx < 0 || nx >= width) continue;
			const int nIdx = ny * width + nx;

			float neighborDepth;
			if (!wf_svgf_depth_at(currentWorldPos, nIdx, cameraOrigin, neighborDepth)) continue;

			const float4 nIn = input[nIdx];
			const float3 nColor = make_float3(nIn.x, nIn.y, nIn.z);
			const float nVariance = nIn.w;
			const float3 nNormal = normals[nIdx];
			const float nLum = wf_svgf_luminance(nColor);

			const float depthGradDotOffset = depthGradX * (float)(dx * stepSize) + depthGradY * (float)(dy * stepSize);
			float w = wf_svgf_edge_weight(
				centerNormal, nNormal, sigmaNormal,
				centerDepth, neighborDepth, depthGradDotOffset, sigmaDepth,
				centerLum, nLum, sqrtCenterVariance, sigmaLuminance);
			// A-trous kernel weight itself (binomial-like [1,2,4,2,1] shape,
			// separable but applied jointly here since the footprint is
			// small) on top of the edge-stopping weight above - the standard
			// SVGF combination, not edge-stopping alone (which would treat a
			// flat, noisy region's own distant samples as equally important
			// as its near ones).
			const int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy;
			constexpr float kKernel[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};
			w *= kKernel[adx] * kKernel[ady] * 16.0f;  // normalized so center (0,0) contributes weight 1 before edge-stopping

			colorSum = colorSum + nColor * w;
			varianceSumSq += w * w * nVariance;
			weightSum += w;
		}
	}

	if (weightSum > 1e-6f) {
		output[idx] = make_float4(colorSum.x / weightSum, colorSum.y / weightSum, colorSum.z / weightSum,
								   varianceSumSq / (weightSum * weightSum));
	} else {
		output[idx] = centerIn;
	}
}

// -----------------------------------------------------------------------
// Kernel 4 - svgf_finalize
// -----------------------------------------------------------------------
// Re-multiplies the fully-filtered, still-demodulated illumination by
// albedo (undoing svgf_prepare_for_filter's own division) and writes the
// result into the real framebuffer - the LAST thing SVGF does each frame,
// after which WavefrontPathTracer::render()'s existing host-copy code path
// picks it up completely unchanged. Must re-multiply by the SAME floored
// albedo svgf_prepare_for_filter divided by (kSvgfMinAlbedo, not the raw
// AOV value) - otherwise any near-zero-albedo pixel (background/sky, whose
// albedo AOV is never written by accumulate_miss and so stays 0) would
// round-trip to black regardless of its actual filtered radiance.
extern "C" __global__ void svgf_finalize(
	const float4* filtered,
	const float3* albedo,
	int numPixels,
	float minAlbedo,
	float3* framebuffer
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	const float4 f = filtered[idx];
	const float3 alb = albedo[idx];
	const float3 albedoFloor = make_float3(fmaxf(alb.x, minAlbedo), fmaxf(alb.y, minAlbedo), fmaxf(alb.z, minAlbedo));
	framebuffer[idx] = make_float3(f.x * albedoFloor.x, f.y * albedoFloor.y, f.z * albedoFloor.z);
}
