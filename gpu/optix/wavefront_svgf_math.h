#pragma once
// wavefront_svgf_math.h -- SVGF's pure math: the variance estimate, the
// temporal blend schedule, and the A-trous edge-stopping weight functions -
// exactly the pieces that need no scene data or RNG draw, split out the same
// way and for the same reason wavefront_restir_gi_math.h already is (see
// that header's own comment): so this can be unit-tested from a plain host
// build, and so wavefront_kernels_svgf.cu's own kernels stay readable
// (formula-in-one-place, not reimplemented inline per call site).
//
// Reference: Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering:
// Real-Time Reconstruction for Path-Traced Global Illumination" (HPG). The
// formulas below match that paper's own eq. 4 (edge-stopping weights) and
// its temporal integration schedule (Section 4.1) - not re-derived from
// scratch.

#include "optix_types.h"
#include "optix_math_helpers.h"   // dot()/fabsf() via <cmath> already pulled in transitively

// Rec.709 relative luminance - the scalar SVGF's variance estimate and every
// edge-stopping weight below operate on (denoising a 3-channel signal
// per-channel would triple the work for no accuracy benefit at the
// TEMPORAL/spatial-WEIGHT-COMPUTATION level; the actual blended/filtered
// VALUES stay full RGB throughout - only the weights are luminance-based).
CPU_GPU inline float wf_svgf_luminance(float3 c) {
	return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Variance from the first and second moments of luminance - Schied et al.
// eq. 3 (Var = E[X^2] - E[X]^2). Clamped non-negative: floating-point
// cancellation between two nearly-equal accumulated means can otherwise
// produce a small negative value, which would poison every downstream
// sqrt(variance) edge-stopping weight (NaN) for what is actually a
// near-zero-variance (converged) pixel.
CPU_GPU inline float wf_svgf_variance(float moment1, float moment2) {
	return fmaxf(0.0f, moment2 - moment1 * moment1);
}

// Temporal blend weight - Schied et al. Section 4.1's own schedule: acts
// like a plain running mean (alpha = 1/(historyLength+1)) while history is
// still short, then clamps to a fixed rate (targetAlpha, typically ~0.2)
// once historyLength grows past 1/targetAlpha - 1 frames. This is what lets
// the SAME formula behave as an unbiased average early (when every sample
// is equally informative and there's no "past" to discard) and as an
// exponential moving average later (bounding how much influence an
// arbitrarily old frame can keep, so the image can still adapt if the scene
// itself changes under an unmoving camera - e.g. an animated light).
CPU_GPU inline float wf_svgf_temporal_alpha(float historyLength, float targetAlpha) {
	return fmaxf(targetAlpha, 1.0f / (historyLength + 1.0f));
}

// Checkerboard temporal-upsampling active-pixel predicate: with SVGF's own
// temporal reprojection/history already doing the reconstruction work, only
// half the pixels need a genuinely fresh primary-ray sample each frame - the
// other half hold last frame's (reprojected) value. `checkerboardActive`
// false is always the safe/default answer (every pixel active, byte-
// identical to the pre-checkerboard behavior) - see WavefrontPathTracer::
// render()'s own comment for the exact gating condition (SVGF AND ReSTIR GI
// history both already warm, so this never activates on the first frame
// after a scene/resolution change). Called identically from
// generate_camera_rays (gpu/optix/wavefront_kernels_camera.cu),
// svgf_checkerboard_clear_frame and svgf_temporal_integrate (wavefront_
// kernels_svgf.cu), and restir_gi_finalize (wavefront_kernels_restir.cu) -
// every consumer of "was this pixel supposed to get a fresh sample this
// frame" must agree on the exact same parity, so this is the one place that
// arithmetic lives. Lives here (not wavefront_device_helpers.h, this
// header's usual home for __device__-only helpers) specifically so it's
// unit-testable from a plain host build, same reasoning as every other
// function in this file - see tests/unit/wavefront_svgf_math_tests.cpp.
CPU_GPU inline bool wf_checkerboard_pixel_active(
		int px, int py, unsigned int frameNumber, bool checkerboardActive) {
	if (!checkerboardActive) return true;
	return ((px + py) & 1) == static_cast<int>(frameNumber & 1u);
}

// A-trous edge-stopping weights (Schied et al. eq. 4) - each in [0,1],
// multiplied together by the caller to get one neighbor's total filter
// weight. Un-normalized (the caller divides by the sum of all sampled
// neighbors' weights, standard weighted-average normalization) - returning
// a normalized value here would require a two-pass filter (sum first, then
// weight), which the paper's own single-pass formulation avoids.

// Normal term: tight cosine power (sigmaNormal is typically large, e.g. 128)
// so even a small shading-normal difference (a curved surface, or a genuine
// edge) sharply cuts the weight - prevents blurring across a geometric
// silhouette that depth/luminance alone might not catch.
CPU_GPU inline float wf_svgf_weight_normal(float3 normalP, float3 normalQ, float sigmaNormal) {
	const float cosTheta = fmaxf(dot(normalP, normalQ), 0.0f);
	return powf(cosTheta, sigmaNormal);
}

// Depth term: `depthGradientDotOffset` is the CENTER pixel's own screen-
// space depth gradient dotted with the (p-q) pixel offset to the neighbor -
// i.e. "how much would depth be expected to change over this offset if the
// surface were locally planar". Normalizing the raw depth difference by
// this (rather than a fixed epsilon) is what makes a grazing/oblique
// surface - where depth legitimately varies fast across neighboring pixels
// - not get its own real detail rejected as if it were a depth
// discontinuity; a genuine edge (silhouette, floating geometry) produces a
// depth difference far larger than the locally-planar prediction, so its
// weight still collapses correctly.
CPU_GPU inline float wf_svgf_weight_depth(float depthP, float depthQ, float depthGradientDotOffset,
										   float sigmaDepth) {
	const float denom = sigmaDepth * fabsf(depthGradientDotOffset) + 1e-8f;
	return expf(-fabsf(depthP - depthQ) / denom);
}

// Luminance term: normalizes the raw luminance difference by the CENTER
// pixel's own estimated standard deviation (sqrt(variance)) instead of a
// fixed constant - a still-noisy (high-variance) pixel tolerates a much
// larger luminance difference before rejecting a neighbor than an
// already-converged (low-variance) one does, which is the whole point of
// making this filter variance-GUIDED rather than a fixed-radius blur.
CPU_GPU inline float wf_svgf_weight_luminance(float lumP, float lumQ, float sqrtVariance, float sigmaLuminance) {
	const float denom = sigmaLuminance * sqrtVariance + 1e-8f;
	return expf(-fabsf(lumP - lumQ) / denom);
}

// Combined per-neighbor filter weight - product of all three terms, matching
// eq. 4's own w(p,q) = w_z(p,q) * w_n(p,q) * w_l(p,q).
CPU_GPU inline float wf_svgf_edge_weight(
		float3 normalP, float3 normalQ, float sigmaNormal,
		float depthP, float depthQ, float depthGradientDotOffset, float sigmaDepth,
		float lumP, float lumQ, float sqrtVariance, float sigmaLuminance) {
	return wf_svgf_weight_normal(normalP, normalQ, sigmaNormal) *
		   wf_svgf_weight_depth(depthP, depthQ, depthGradientDotOffset, sigmaDepth) *
		   wf_svgf_weight_luminance(lumP, lumQ, sqrtVariance, sigmaLuminance);
}
