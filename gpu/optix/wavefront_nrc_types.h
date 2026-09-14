#pragma once
// wavefront_nrc_types.h -- Neural Radiance Cache (NRC) shared constants and
// POD types, wavefront GPU backend, Live Preview only. See this project's
// own plan for the full design rationale.
//
// Fills the gap the world-space irradiance probe cache (probe_grid_types.h)
// structurally cannot: that cache's SH-L1 coefficients are view-independent
// and Lambertian-only. NRC is a small MLP taking position + direction +
// material features as input, trained online from traced paths, usable for
// any non-specular material (Lambertian AND glossy/Conductor).
//
// Kept in its own header rather than probe_grid_types.h/wavefront_types.h -
// this feature shares neither the probe grid's spatial index nor the
// spectral-width-coupled GI structs. Included from both the host side
// (wavefront_path_tracer.h/.cpp, optix_renderer.h/optix_renderer_scene.cpp -
// plain floats/ints, no CUDA-only syntax) and the device side
// (wavefront_nrc_mlp.h, wavefront_nrc_encoding.h, wavefront_device_helpers.h,
// wavefront_kernels_nrc.cu).

#include <cuda_runtime.h>
// Relative path, not a bare quoted include - see optix_types.h's/
// probe_grid_types.h's own identical comment for why (resolves under both
// the plain MSBuild .cpp compile and nvcc's OptiX compile, whose -I flags
// differ).
#include "../../src/shared/cpu_gpu.h"

// ---------------------------------------------------------------------------
// Network architecture (see wavefront_nrc_encoding.h for how the 35 raw
// input dims below are produced, and wavefront_nrc_mlp.h for the
// forward/backward/Adam math that consumes these layout constants).
// ---------------------------------------------------------------------------
// one-blob-encoded position(12)+dir(8)+normal(8)+roughness(4)+albedo(3).
// v1 simplification: a single surface albedo, not separate diffuse/
// specular albedo - Conductor's tint is a complex eta_c/k_c pair, not a
// plain RGB reflectance, so deriving a uniform "specular albedo" across
// every eligible material type (Lambertian AND Conductor/RoughMetal) would
// need a per-material-type Fresnel-at-normal-incidence derivation with
// uncertain benefit; the single albedo below is populated from
// materials[matIdx].albedo directly for Lambertian/RoughMetal/Metal, and
// from a normal-incidence Fresnel reflectance computed from eta_c/k_c for
// Conductor (see the query/training call sites in wavefront_device_helpers.h/
// wavefront_kernels_nrc.cu).
static constexpr int kNrcInputDim = 35;
static constexpr int kNrcHiddenDim = 64;
static constexpr int kNrcOutputDim = 3;    // RGB radiance estimate

// Flat weight-buffer layout: [W1 | b1 | W2 | b2 | W3 | b3 | W4 | b4].
// Row-major, Wk[i][j] at offset kNrcWkOffset + i*outDim + j (in-features i,
// out-features j) - see wf_nrc_forward()'s own comment for the exact
// indexing convention this layout is written to match.
static constexpr int kNrcW1Count = kNrcInputDim * kNrcHiddenDim;
static constexpr int kNrcB1Count = kNrcHiddenDim;
static constexpr int kNrcW2Count = kNrcHiddenDim * kNrcHiddenDim;
static constexpr int kNrcB2Count = kNrcHiddenDim;
static constexpr int kNrcW3Count = kNrcHiddenDim * kNrcHiddenDim;
static constexpr int kNrcB3Count = kNrcHiddenDim;
static constexpr int kNrcW4Count = kNrcHiddenDim * kNrcOutputDim;
static constexpr int kNrcB4Count = kNrcOutputDim;

static constexpr int kNrcW1Offset = 0;
static constexpr int kNrcB1Offset = kNrcW1Offset + kNrcW1Count;
static constexpr int kNrcW2Offset = kNrcB1Offset + kNrcB1Count;
static constexpr int kNrcB2Offset = kNrcW2Offset + kNrcW2Count;
static constexpr int kNrcW3Offset = kNrcB2Offset + kNrcB2Count;
static constexpr int kNrcB3Offset = kNrcW3Offset + kNrcW3Count;
static constexpr int kNrcW4Offset = kNrcB3Offset + kNrcB3Count;
static constexpr int kNrcB4Offset = kNrcW4Offset + kNrcW4Count;
static constexpr int kNrcNumWeights = kNrcB4Offset + kNrcB4Count;  // 11,011

// ---------------------------------------------------------------------------
// Render-path query/termination tuning (wf_finish_material_scatter(), see
// wavefront_device_helpers.h).
// ---------------------------------------------------------------------------
static constexpr int kNrcMinDepth = 2;                  // matches kProbeCacheMinDepth - depth 0->1 is ReSTIR GI's territory
static constexpr float kNrcTerminationRamp = 0.3f;      // pTerm rises by this much per depth past kNrcMinDepth
static constexpr float kNrcMaxTerminationProb = 0.9f;   // never 1.0 - keeps the renderer asymptotically unbiased
static constexpr int kNrcWarmupSteps = 256;              // training steps (launchNrcTrainingUpdate() calls) before queries are trusted at all
// Sentinel "raw" roughness fed into wf_nrc_encode_features()'s own
// 1-exp(-r) remap for Lambertian vertices (which have no GGX alpha of their
// own) - large enough that the remapped value saturates to within 1e-3 of
// 1.0 (maximally rough), matching Lambertian's own infinitely-rough,
// uniform-hemisphere reflectance. Shared by the training-side write
// (wavefront_kernels_nrc.cu) and the render-path query
// (wavefront_device_helpers.h) so both agree on what "Lambertian
// roughness" encodes as.
static constexpr float kNrcLambertianRoughnessSentinel = 8.0f;

// ---------------------------------------------------------------------------
// Training-path budget (launchNrcTrainingUpdate(), fixed/resolution-
// independent - mirrors kProbesPerFrame_'s own fixed-budget convention).
// ---------------------------------------------------------------------------
static constexpr int kNrcTrainingPathsPerFrame = 4096;
static constexpr int kNrcTrainingMaxBounces = 5;
static constexpr int kNrcTrainingRecordCapacity = kNrcTrainingPathsPerFrame * kNrcTrainingMaxBounces;  // 20,480

// ---------------------------------------------------------------------------
// Optimizer tuning (wf_nrc_adam_step(), wavefront_nrc_mlp.h).
// ---------------------------------------------------------------------------
static constexpr float kNrcLearningRate = 1e-3f;
static constexpr float kNrcAdamBeta1 = 0.9f;
static constexpr float kNrcAdamBeta2 = 0.99f;
static constexpr float kNrcAdamEpsilon = 1e-8f;
static constexpr float kNrcGradClip = 10.0f;             // clamp per-weight averaged gradient before the Adam step
static constexpr float kNrcRelativeLossEpsilon = 1e-2f;  // denominator floor in the relative-L2 loss, see wf_nrc_loss_gradient()

// One training record: one visited vertex of one training path. Populated
// by nrc_training_shade (Phase A, forward in depth), consumed by
// nrc_bootstrap_and_train (Phase B, reverse depth) - see
// wavefront_kernels_nrc.cu. Stored in a flat, depth-major array of fixed
// size kNrcTrainingRecordCapacity (record for path p at depth d lives at
// index d*kNrcTrainingPathsPerFrame + p) - deterministic addressing, no
// atomics needed to find a vertex's own slot or its successor's.
struct NrcTrainingRecord {
	float3 position = make_float3(0.0f, 0.0f, 0.0f);
	float3 outgoingDir = make_float3(0.0f, 0.0f, 1.0f);   // direction this vertex scatters toward (i.e. toward vertex i+1)
	float3 normal = make_float3(0.0f, 0.0f, 1.0f);
	float  roughness = 0.0f;
	float3 albedo = make_float3(0.0f, 0.0f, 0.0f);
	// This vertex's own resolved direct lighting (NEE) plus, if the path
	// terminates here, any emission/escaped-background term - see `flags`
	// below. Filled by Phase A; Phase B adds the bootstrap term (throughput-
	// weighted L_hat of vertex i+1) on top of this for a non-terminal record
	// to get the full training target, or uses this value AS the target
	// unmodified for a terminal one.
	float3 directRadiance = make_float3(0.0f, 0.0f, 0.0f);
	// BSDF attenuation/pdf factor from this vertex to vertex i+1 - the
	// throughput multiplier the bootstrap term (throughputToNext * L_hat(i+1))
	// needs. Meaningless (never read) when nextIsTerminal.
	float3 throughputToNext = make_float3(0.0f, 0.0f, 0.0f);
	int    nextRecordIndex = -1;  // flat index of vertex i+1's own record, or -1 (this was the last allowed bounce)
	// bit0 (1): this record is valid (a real vertex was written here by
	// Phase A - Phase B skips any record left at its zero-initialized
	// default, exactly like GpuProbe::numRaysEverTraced==0 gates the probe
	// cache). Whether to bootstrap is decided at Phase-B time purely by
	// checking nextRecordIndex>=0 AND records[nextRecordIndex] is itself
	// Valid - there is no separate "terminal" flag: a record whose
	// continuation bounce missed the scene, hit an untrained material type,
	// or was never traced (max bounces reached) all collapse to the exact
	// same "no valid next record" case, needing no bit of their own to
	// distinguish (see nrc_bootstrap_and_train, wavefront_kernels_nrc.cu).
	unsigned int flags = 0u;
};

static constexpr unsigned int kNrcRecordFlagValid = 1u;
