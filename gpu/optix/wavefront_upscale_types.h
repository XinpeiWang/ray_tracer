#pragma once
// wavefront_upscale_types.h -- Neural temporal upscale shared constants and
// POD types, wavefront GPU backend, Live Preview only. See this project's
// own plan for the full design rationale.
//
// Replaces the existing CPU-side temporal-upscale reconstruction
// (qt_gui/realtime_preview_session.cpp's reprojectAccumulationHi()/write-step
// splat, gpu/optix/wavefront_temporal_upscale_math.h's own jitter sequence
// UNCHANGED and still driving primary-ray generation) with a small,
// online-trained MLP that predicts a learned per-candidate blend instead of
// a rigid block copy - see this project's own plan for why a masked-
// low-res self-supervision scheme was considered and rejected in favor of
// delayed real-sample supervision.
//
// Kept in its own header rather than wavefront_nrc_types.h - this feature
// shares NRC's own "small hand-written MLP + Adam, CPU_GPU-tagged,
// dimension-generic via named constants" idiom but is a completely
// different network (candidate-blend classifier, not a radiance regressor)
// with its own weight buffer, own forward-pass cache shape (needed for
// delayed-supervision backward - see NrcTrainingRecord's own comment,
// wavefront_nrc_types.h, for the identical "cache now, become supervised
// later" reasoning this mirrors). Included from both the host side
// (wavefront_path_tracer.h/.cpp) and the device side (wavefront_upscale_mlp.h,
// wavefront_kernels_upscale.cu).

#include <cuda_runtime.h>
// Relative path, not a bare quoted include - see optix_types.h's/
// wavefront_nrc_types.h's own identical comment for why (resolves under
// both the plain MSBuild .cpp compile and nvcc's OptiX compile, whose -I
// flags differ).
#include "../../src/shared/cpu_gpu.h"

// ---------------------------------------------------------------------------
// Candidate set - the blend targets both the input features and the final
// output are built from. Every candidate is a raw, already-plausible RGB
// color (never an invented value), so the network's own output - a convex
// combination of these six - can never leave the range of real samples
// already on screen, even before any training has happened (see this
// project's own plan for why this is the whole point of predicting BLEND
// WEIGHTS rather than raw RGB directly).
// ---------------------------------------------------------------------------
// Candidate indices, shared between the input-vector layout below and the
// blend step in wf_upscale_forward()/the inference kernel - MUST stay in
// this exact order, since the input vector's first kUpscaleCandidateCount*3
// floats ARE these six colors, in this order (see kUpscaleInputDim's own
// comment).
static constexpr int kUpscaleCandidateCenter = 0;  // this high-res cell's own parent low-res pixel's fresh sample (C)
static constexpr int kUpscaleCandidateNorth  = 1;
static constexpr int kUpscaleCandidateSouth  = 2;
static constexpr int kUpscaleCandidateEast   = 3;
static constexpr int kUpscaleCandidateWest   = 4;
static constexpr int kUpscaleCandidateHistory = 5;  // reprojected d_neuralUpscaleHistory_ color (H)
static constexpr int kUpscaleCandidateCount = 6;

// ---------------------------------------------------------------------------
// Network architecture (see wavefront_upscale_mlp.h for the forward/
// backward/Adam math that consumes these layout constants).
//
// Input layout (31 floats total):
//   [0..17]  the 6 candidate RGB colors themselves, in kUpscaleCandidate*
//            order (18) - present in the input AND used directly as the
//            blend targets, so the same values the network reads are what
//            it ultimately recombines; no separate "gather again for the
//            blend" step needed.
//   [18]     history age/confidence: how many frames old this cell's own
//            history entry is, normalized to [0,1] by the write cycle's own
//            period (upscaleFactor^2) - 0 for a cell fresh this very frame.
//   [19]     disocclusion confidence - the SAME world-space reprojection
//            test's own pass/fail signal wf_restir_reproject_prev_pixel's
//            camera-motion reprojection already computes, reused rather
//            than re-derived.
//   [20..27] sub-pixel jitter phase, one-blob encoded (2 axes x 4 bins) via
//            the existing wf_nrc_encode_scalar() idiom (wavefront_nrc_
//            encoding.h) - this is really a categorical "which of the
//            upscaleFactor^2 cells is being asked for" signal, not a
//            continuous quantity, so one-blob (a soft one-hot) is the right
//            encoding, matching NRC's own reasoning for encoding position/
//            direction/roughness the same way rather than feeding raw
//            floats.
//   [28..29] screen-space motion vector (this pixel's parent low-res
//            pixel's own d_motionVectors_ entry - a block-level
//            approximation shared by every high-res sub-cell in that
//            pixel's footprint, matching the granularity the PRE-existing
//            CPU reconstruction already accepted).
//   [30]     linear depth (|worldPos - cameraOrigin|, same derivation
//            wf_svgf_depth_at() already uses from d_worldPos_).
// ---------------------------------------------------------------------------
static constexpr int kUpscaleCandidateInputDim = kUpscaleCandidateCount * 3;  // 18
static constexpr int kUpscaleHistoryAgeDim = 1;
static constexpr int kUpscaleConfidenceDim = 1;
static constexpr int kUpscalePhaseBins = 4;                 // one-blob bins per axis, matches kNrcBinsPerAxis (wavefront_nrc_encoding.h) so wf_nrc_encode_scalar() can be reused as-is
static constexpr int kUpscalePhaseDim = 2 * kUpscalePhaseBins;  // 8 (2 axes)
static constexpr int kUpscaleMotionDim = 2;
static constexpr int kUpscaleDepthDim = 1;
static constexpr int kUpscaleInputDim = kUpscaleCandidateInputDim + kUpscaleHistoryAgeDim +
	kUpscaleConfidenceDim + kUpscalePhaseDim + kUpscaleMotionDim + kUpscaleDepthDim;  // 31

static constexpr int kUpscaleHiddenDim = 32;   // smaller than NRC's 64 - see this project's own plan for why
static constexpr int kUpscaleOutputDim = kUpscaleCandidateCount;  // one logit per candidate, softmaxed into blend weights

// Flat weight-buffer layout: [W1 | b1 | W2 | b2 | W3 | b3] (3 linear layers,
// 2 ReLU hidden - one fewer layer than NRC's 4, see wf_upscale_forward()'s
// own comment). Row-major, Wk[i][j] at offset kUpscaleWkOffset + i*outDim+j,
// same indexing convention as wavefront_nrc_types.h's own kNrcW1Offset etc.
static constexpr int kUpscaleW1Count = kUpscaleInputDim * kUpscaleHiddenDim;
static constexpr int kUpscaleB1Count = kUpscaleHiddenDim;
static constexpr int kUpscaleW2Count = kUpscaleHiddenDim * kUpscaleHiddenDim;
static constexpr int kUpscaleB2Count = kUpscaleHiddenDim;
static constexpr int kUpscaleW3Count = kUpscaleHiddenDim * kUpscaleOutputDim;
static constexpr int kUpscaleB3Count = kUpscaleOutputDim;

static constexpr int kUpscaleW1Offset = 0;
static constexpr int kUpscaleB1Offset = kUpscaleW1Offset + kUpscaleW1Count;
static constexpr int kUpscaleW2Offset = kUpscaleB1Offset + kUpscaleB1Count;
static constexpr int kUpscaleB2Offset = kUpscaleW2Offset + kUpscaleW2Count;
static constexpr int kUpscaleW3Offset = kUpscaleB2Offset + kUpscaleB2Count;
static constexpr int kUpscaleB3Offset = kUpscaleW3Offset + kUpscaleW3Count;
static constexpr int kUpscaleNumWeights = kUpscaleB3Offset + kUpscaleB3Count;  // 2,278

// ---------------------------------------------------------------------------
// Unlike NRC (whose untrained softplus output is meaningless noise, gated
// behind kNrcWarmupSteps until trusted), this network's output is a convex
// blend of already-plausible candidate colors BY CONSTRUCTION - even fully
// untrained (near-uniform blend weights), the result is a reasonable
// average of nearby real samples, not garbage. No warm-up gate is needed
// for this feature's inference to be safe to consume from frame 1 - see
// this project's own plan for why the candidate-blend architecture was
// chosen specifically to avoid needing one.
// ---------------------------------------------------------------------------
// Training budget (fixed/resolution-independent, mirrors kNrcTrainingPathsPerFrame's
// own fixed-budget convention - launchNeuralUpscaleUpdate(), wavefront_path_tracer.cpp).
// Needs ZERO extra ray tracing (unlike NRC): every "training record" here is
// a cell that ALREADY received a real low-res sample this frame as part of
// the ordinary write cycle, so this only bounds the BACKWARD-pass cost, not
// any tracing cost.
// ---------------------------------------------------------------------------
static constexpr int kUpscaleTrainingRecordsPerFrame = 8192;

// ---------------------------------------------------------------------------
// Optimizer tuning (wf_upscale_adam_step(), wavefront_upscale_mlp.h) - same
// constants as NRC's own (wavefront_nrc_types.h), duplicated here rather
// than shared so the two features' learning rates/clip thresholds can be
// tuned independently without coupling.
// ---------------------------------------------------------------------------
static constexpr float kUpscaleLearningRate = 1e-3f;
static constexpr float kUpscaleAdamBeta1 = 0.9f;
static constexpr float kUpscaleAdamBeta2 = 0.99f;
static constexpr float kUpscaleAdamEpsilon = 1e-8f;
static constexpr float kUpscaleGradClip = 10.0f;
static constexpr float kUpscaleRelativeLossEpsilon = 1e-2f;

// One forward pass's worth of saved intermediates for one high-res cell -
// written by the inference kernel every frame, consumed by the training
// kernel up to (upscaleFactor^2 - 1) frames later when that same cell's
// real low-res value finally arrives - see NrcTrainingRecord's own comment
// (wavefront_nrc_types.h) for the identical "cache now, become supervised
// later" shape this mirrors, and this project's own plan for why a cache
// (not just the final blended color) is required: Adam keeps changing the
// weights between when a prediction is made and when it's later supervised,
// so backward needs the ACTUAL activations from that original forward pass,
// not a fresh recompute under different weights.
struct UpscaleForwardCache {
	float input[kUpscaleInputDim] = {};
	float pre1[kUpscaleHiddenDim] = {};
	float h1[kUpscaleHiddenDim] = {};    // post-ReLU
	float pre2[kUpscaleHiddenDim] = {};
	float h2[kUpscaleHiddenDim] = {};    // post-ReLU
	float logits[kUpscaleOutputDim] = {};
	float3 blended = make_float3(0.0f, 0.0f, 0.0f);
	// Set by the inference kernel whenever this cell had valid inputs to
	// predict from at all (matches GpuProbe::numRaysEverTraced==0's own
	// "never written" gate) - the training kernel skips a record left at
	// its zero-initialized default the same way nrc_bootstrap_and_train
	// skips a record without kNrcRecordFlagValid set.
	unsigned int flags = 0u;
};

static constexpr unsigned int kUpscaleCacheFlagValid = 1u;
