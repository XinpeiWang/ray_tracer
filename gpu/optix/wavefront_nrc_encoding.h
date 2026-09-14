#pragma once
// wavefront_nrc_encoding.h -- Neural Radiance Cache (NRC) input feature
// encoding, wavefront GPU backend, Live Preview only. See this project's
// own plan and wavefront_nrc_types.h's own header comment for the full
// design rationale.
//
// wf_nrc_encode_features() turns 7 raw physical quantities (position,
// outgoing direction, normal, roughness, diffuse/specular albedo) into the
// kNrcInputDim=35 encoded floats wf_nrc_forward() (wavefront_nrc_mlp.h)
// actually consumes. Forward-only: nothing downstream ever needs a gradient
// with respect to the RAW inputs (they're observed data, not optimized
// parameters), so unlike wf_nrc_forward()/wf_nrc_backward()'s matched pair,
// this encoding has no backward counterpart.
//
// Encoding scheme: a normalized linear "hat" (triangular) basis over
// kNrcBinsPerAxis=4 evenly-spaced bin centers in [0,1], applied
// independently per scalar sub-feature. This is a simplified, exact-
// partition-of-unity relative of the original NRC paper's Gaussian
// "one-blob" encoding (Müller et al.), chosen over the paper's true
// Gaussian kernel for one concrete reason: partition-of-unity (weights sum
// to exactly 1 for any input in range) is trivial to state and unit-test
// for a hat basis, and isn't a natural property of an unnormalized Gaussian
// bump - a testable invariant matters here given the numerical-correctness
// bar the rest of this feature (wf_nrc_backward()'s gradient check) is held
// to. A learned multiresolution hash grid (Instant-NGP-style) was also
// considered and rejected - it needs its own trainable parameters and its
// own backward pass, doubling this feature's from-scratch-ML surface area
// for a network this small.
//
// Position is normalized against the scene's own AABB (passed in by the
// caller, mirroring how OptiXRenderer::buildProbeGrid(), optix_renderer_
// scene.cpp, already derives probe-grid spacing from the same scene AABB) -
// NOT read from any global/static state, so this header stays dependency-
// free and independently unit-testable, same convention wavefront_
// guiding.h's own header comment documents for itself.
//
// Deliberately dependency-free: no wf_rand()/wf_pcg() calls, no include of
// wavefront_device_helpers.h - can be included from anywhere, host or
// device side, in any order (same convention wavefront_guiding.h already
// established for the same reason).

#include <cuda_runtime.h>
// Relative path - see probe_grid_types.h's own identical comment for why.
#include "../../src/shared/cpu_gpu.h"
#include "wavefront_nrc_types.h"  // kNrcInputDim

static constexpr int kNrcBinsPerAxis = 4;

// Hat-basis encoding of a single scalar x (expected in [0,1], clamped if
// not) into kNrcBinsPerAxis weights summing to exactly 1. Bin centers are
// evenly spaced at i/(kNrcBinsPerAxis-1) for i in [0,kNrcBinsPerAxis) - the
// OUTER bin centers land exactly on 0 and 1, so a value at either boundary
// activates only its own outer bin at full weight, matching a linear-
// interpolation basis exactly (this is standard piecewise-linear
// interpolation over kNrcBinsPerAxis-1 equal-width segments, written out
// per-bin rather than as a single lerp so it generalizes to any bin count).
CPU_GPU void wf_nrc_encode_scalar(float x, float* outWeights) {
	if (x < 0.0f) x = 0.0f;
	if (x > 1.0f) x = 1.0f;
	const float binWidth = 1.0f / (float)(kNrcBinsPerAxis - 1);
	for (int i = 0; i < kNrcBinsPerAxis; ++i) {
		const float center = (float)i * binWidth;
		const float d = fabsf(x - center) / binWidth;
		outWeights[i] = d < 1.0f ? (1.0f - d) : 0.0f;
	}
}

// Maps a unit direction to (theta,phi) normalized into [0,1]^2 - theta =
// acos(z)/pi (polar angle from +Z), phi = (atan2(y,x)+pi)/(2*pi). Shared by
// direction/normal encoding below (both are unit vectors encoded the same
// way).
CPU_GPU void wf_nrc_dir_to_unit_square(float3 d, float& outTheta, float& outPhi) {
	const float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
	float3 n = len > 1e-8f ? make_float3(d.x / len, d.y / len, d.z / len) : make_float3(0.0f, 0.0f, 1.0f);
	if (n.z > 1.0f) n.z = 1.0f;
	if (n.z < -1.0f) n.z = -1.0f;
	outTheta = acosf(n.z) / 3.14159265358979323846f;
	outPhi = (atan2f(n.y, n.x) + 3.14159265358979323846f) / (2.0f * 3.14159265358979323846f);
}

// Full kNrcInputDim=35 encoding, in this fixed order:
//   [0:12)  position, one-blob per axis (x,y,z), 4 bins each
//   [12:20) outgoing direction, one-blob per (theta,phi), 4 bins each
//   [20:28) normal, one-blob per (theta,phi), 4 bins each
//   [28:32) roughness (remapped 1-exp(-r), matching the paper's own remap),
//           one-blob, 4 bins
//   [32:35) surface albedo, raw RGB (unencoded, matches the paper's own
//           choice to leave albedo unencoded) - see wavefront_nrc_types.h's
//           own kNrcInputDim comment for why this is a single albedo, not
//           separate diffuse/specular albedo
// `aabbMin`/`aabbExtent` give the scene's world-space bounding box (extent
// components must be > 0 - a degenerate/empty-scene AABB is the caller's
// responsibility to guard against, same as buildProbeGrid()'s own
// zero-extent handling).
CPU_GPU void wf_nrc_encode_features(float3 position, float3 outgoingDir, float3 normal,
									 float roughness, float3 albedo,
									 float3 aabbMin, float3 aabbExtent, float* outFeatures) {
	const float px = aabbExtent.x > 1e-8f ? (position.x - aabbMin.x) / aabbExtent.x : 0.5f;
	const float py = aabbExtent.y > 1e-8f ? (position.y - aabbMin.y) / aabbExtent.y : 0.5f;
	const float pz = aabbExtent.z > 1e-8f ? (position.z - aabbMin.z) / aabbExtent.z : 0.5f;
	wf_nrc_encode_scalar(px, outFeatures + 0);
	wf_nrc_encode_scalar(py, outFeatures + 4);
	wf_nrc_encode_scalar(pz, outFeatures + 8);

	float dirTheta, dirPhi;
	wf_nrc_dir_to_unit_square(outgoingDir, dirTheta, dirPhi);
	wf_nrc_encode_scalar(dirTheta, outFeatures + 12);
	wf_nrc_encode_scalar(dirPhi, outFeatures + 16);

	float nTheta, nPhi;
	wf_nrc_dir_to_unit_square(normal, nTheta, nPhi);
	wf_nrc_encode_scalar(nTheta, outFeatures + 20);
	wf_nrc_encode_scalar(nPhi, outFeatures + 24);

	const float roughnessRemapped = 1.0f - expf(-fmaxf(0.0f, roughness));
	wf_nrc_encode_scalar(roughnessRemapped, outFeatures + 28);

	outFeatures[32] = albedo.x;
	outFeatures[33] = albedo.y;
	outFeatures[34] = albedo.z;
}
