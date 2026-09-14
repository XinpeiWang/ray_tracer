#pragma once
// wavefront_upscale_mlp.h -- neural temporal upscale's hand-written MLP
// forward pass, backward pass, softmax-blend combinator, and Adam optimizer
// step, wavefront GPU backend, Live Preview only. See this project's own
// plan and wavefront_upscale_types.h's own header comment for the full
// design rationale.
//
// This is the highest-bug-density code in the whole feature, same warning
// wavefront_nrc_mlp.h's own header carries for NRC: a subtly wrong gradient
// here won't crash anything, it'll silently train the blend toward garbage.
// tests/unit/wavefront_upscale_math_tests.cpp's numerical-gradient check
// (central-difference vs. wf_upscale_backward()'s analytic output) is the
// mandatory gate for this file. The hidden-layer backward chain below is a
// direct, dimension-adjusted copy of wf_nrc_backward()'s already-proven
// per-layer pattern; the ONE genuinely new piece of math relative to NRC is
// the softmax-blend combinator's own backward (see wf_upscale_backward()'s
// own comment) - that is where a new bug is most likely to hide and where
// the unit test should concentrate its scrutiny.
//
// One CUDA thread = one high-res cell's full inference (or, for a "due"
// cell, forward+backward) pass, entirely in registers/local arrays - same
// "one thread, one item, full eval in registers" pattern wf_nrc_forward()/
// wf_nrc_backward() already use (no shared memory, no warp cooperation, no
// fp16/WMMA).
//
// Network: Input(31) -> Linear+ReLU(32) -> Linear+ReLU(32) -> Linear(6)
// -> softmax -> blend against the 6 raw candidate colors already embedded
// in the input vector itself (see wavefront_upscale_types.h's own
// kUpscaleCandidate* comment). One fewer hidden layer, and half NRC's
// hidden width, than wavefront_nrc_mlp.h's own 3x64 - see this project's
// own plan for why (a local blend-weight decision over a handful of
// already-plausible candidates is lower-complexity than NRC's continuous
// radiance-field regression, and this network runs at full display
// resolution every frame - an order of magnitude+ more invocations than
// NRC's fixed per-frame training budget - so per-eval cost matters more).
// Weight layout/offsets: wavefront_upscale_types.h.
//
// Deliberately dependency-free beyond wavefront_upscale_types.h's layout
// constants - no wf_rand()/wf_pcg() calls, can be included from anywhere,
// host or device side, in any order (matches wavefront_nrc_mlp.h's own
// convention).

#include <cuda_runtime.h>
// Relative path - see probe_grid_types.h's own identical comment for why.
#include "../../src/shared/cpu_gpu.h"
#include "wavefront_upscale_types.h"

CPU_GPU float wf_upscale_relu(float x) { return x > 0.0f ? x : 0.0f; }
CPU_GPU float wf_upscale_relu_grad(float preActivation) { return preActivation > 0.0f ? 1.0f : 0.0f; }

// Accumulates `value` into gradAccum[idx] - atomicAdd on the device (many
// threads' backward passes share one gradAccum buffer across a training
// batch), plain += on the host (unit tests are single-threaded) - see
// wf_nrc_accumulate_grad()'s own identical comment (wavefront_nrc_mlp.h)
// for why __CUDA_ARCH__ is the correct guard here, not a platform check.
CPU_GPU void wf_upscale_accumulate_grad(float* gradAccum, int idx, float value) {
#if defined(__CUDA_ARCH__)
	atomicAdd(&gradAccum[idx], value);
#else
	gradAccum[idx] += value;
#endif
}

// Plain softmax over kUpscaleOutputDim logits - shared by the forward
// pass (to produce this frame's actual blend weights) and the backward
// pass (which needs the SAME weights again to build the softmax
// Jacobian - see wf_upscale_backward()'s own comment). Max-subtracted for
// numerical stability, standard practice for a small, always-finite logit
// vector like this one.
CPU_GPU void wf_upscale_softmax(const float* logits, float* outWeights) {
	float maxLogit = logits[0];
	for (int k = 1; k < kUpscaleOutputDim; ++k) {
		if (logits[k] > maxLogit) maxLogit = logits[k];
	}
	float sumExp = 0.0f;
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		outWeights[k] = expf(logits[k] - maxLogit);
		sumExp += outWeights[k];
	}
	const float invSum = 1.0f / fmaxf(sumExp, 1e-20f);
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		outWeights[k] *= invSum;
	}
}

// Reads candidate k's raw RGB straight out of the input vector - see
// wavefront_upscale_types.h's own kUpscaleCandidate* comment for why the
// first kUpscaleCandidateInputDim floats of the input ARE the six
// candidate colors, in kUpscaleCandidate* order.
CPU_GPU float3 wf_upscale_candidate(const float* input, int k) {
	return make_float3(input[3 * k + 0], input[3 * k + 1], input[3 * k + 2]);
}

// Forward pass: `weights` is a flat kUpscaleNumWeights buffer laid out per
// wavefront_upscale_types.h, `encodedInput` is kUpscaleInputDim floats.
// Fills every field of `cache` (including the final `blended` color) -
// callers doing pure inference (no backward pass planned, i.e. every cell
// NOT "due" for training this frame) only need cache.blended, but every
// cell's FULL cache is still written unconditionally (see wavefront_
// upscale_types.h's own UpscaleForwardCache comment for why: a cell not
// due today may become due on a future frame, at which point its cache
// from THIS frame is exactly what backward will need).
CPU_GPU void wf_upscale_forward(const float* weights, const float* encodedInput, UpscaleForwardCache& cache) {
	for (int i = 0; i < kUpscaleInputDim; ++i) cache.input[i] = encodedInput[i];

	for (int j = 0; j < kUpscaleHiddenDim; ++j) {
		float acc = weights[kUpscaleB1Offset + j];
		for (int i = 0; i < kUpscaleInputDim; ++i) {
			acc += cache.input[i] * weights[kUpscaleW1Offset + i * kUpscaleHiddenDim + j];
		}
		cache.pre1[j] = acc;
		cache.h1[j] = wf_upscale_relu(acc);
	}
	for (int j = 0; j < kUpscaleHiddenDim; ++j) {
		float acc = weights[kUpscaleB2Offset + j];
		for (int i = 0; i < kUpscaleHiddenDim; ++i) {
			acc += cache.h1[i] * weights[kUpscaleW2Offset + i * kUpscaleHiddenDim + j];
		}
		cache.pre2[j] = acc;
		cache.h2[j] = wf_upscale_relu(acc);
	}
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		float acc = weights[kUpscaleB3Offset + k];
		for (int i = 0; i < kUpscaleHiddenDim; ++i) {
			acc += cache.h2[i] * weights[kUpscaleW3Offset + i * kUpscaleOutputDim + k];
		}
		cache.logits[k] = acc;
	}

	float blendWeights[kUpscaleOutputDim];
	wf_upscale_softmax(cache.logits, blendWeights);
	float3 blended = make_float3(0.0f, 0.0f, 0.0f);
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		const float3 cand = wf_upscale_candidate(cache.input, k);
		blended.x += blendWeights[k] * cand.x;
		blended.y += blendWeights[k] * cand.y;
		blended.z += blendWeights[k] * cand.z;
	}
	cache.blended = blended;
}

// Backward pass: given dLoss/dBlended (from wf_upscale_loss_gradient()
// below) and the forward pass's own cache, accumulates dLoss/dWeight for
// every weight/bias into gradAccum (a flat kUpscaleNumWeights buffer, same
// layout as `weights`). Does NOT propagate a gradient past the input layer
// (the raw candidate colors/AOVs are observed data, not optimized
// parameters) - same convention wf_nrc_backward() already establishes.
//
// The softmax-blend combinator's own backward (the piece with no NRC
// analog): blended = sum_k weight_k * candidate_k, weight = softmax(logits).
//   dL/dWeight_k       = dot(dL/dBlended, candidate_k)   -- blended is LINEAR in weight_k
//   dL/dLogits_i       = weight_i * (dL/dWeight_i - sum_k dL/dWeight_k * weight_k)
// The second line is the standard softmax-Jacobian contraction
// (d(softmax)_k/d(logit)_i = weight_k*(delta_ki - weight_i)), written in
// its usual "avoid materializing the full K x K Jacobian" form. From
// dLogits onward this is a plain linear layer (kUpscaleW3/B3) feeding into
// the SAME two-hidden-layer ReLU backward chain wf_nrc_backward() already
// uses, just at this network's own (smaller) dimensions.
CPU_GPU void wf_upscale_backward(const float* weights, const UpscaleForwardCache& cache,
								  const float3& dLossDBlended, float* gradAccum) {
	float blendWeights[kUpscaleOutputDim];
	wf_upscale_softmax(cache.logits, blendWeights);

	float dWeight[kUpscaleOutputDim];
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		const float3 cand = wf_upscale_candidate(cache.input, k);
		dWeight[k] = dLossDBlended.x * cand.x + dLossDBlended.y * cand.y + dLossDBlended.z * cand.z;
	}
	float dWeightDotWeight = 0.0f;
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		dWeightDotWeight += dWeight[k] * blendWeights[k];
	}
	float dLogits[kUpscaleOutputDim];
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		dLogits[k] = blendWeights[k] * (dWeight[k] - dWeightDotWeight);
	}

	// Layer 3 (logits are this layer's own pre-activation - no elementwise
	// nonlinearity of its own, softmax already consumed above).
	for (int k = 0; k < kUpscaleOutputDim; ++k) {
		wf_upscale_accumulate_grad(gradAccum, kUpscaleB3Offset + k, dLogits[k]);
		for (int i = 0; i < kUpscaleHiddenDim; ++i) {
			wf_upscale_accumulate_grad(gradAccum, kUpscaleW3Offset + i * kUpscaleOutputDim + k, cache.h2[i] * dLogits[k]);
		}
	}

	float dPre2[kUpscaleHiddenDim];
	for (int i = 0; i < kUpscaleHiddenDim; ++i) {
		float dH2 = 0.0f;
		for (int k = 0; k < kUpscaleOutputDim; ++k) {
			dH2 += weights[kUpscaleW3Offset + i * kUpscaleOutputDim + k] * dLogits[k];
		}
		dPre2[i] = dH2 * wf_upscale_relu_grad(cache.pre2[i]);
	}
	for (int j = 0; j < kUpscaleHiddenDim; ++j) {
		wf_upscale_accumulate_grad(gradAccum, kUpscaleB2Offset + j, dPre2[j]);
		for (int i = 0; i < kUpscaleHiddenDim; ++i) {
			wf_upscale_accumulate_grad(gradAccum, kUpscaleW2Offset + i * kUpscaleHiddenDim + j, cache.h1[i] * dPre2[j]);
		}
	}

	float dPre1[kUpscaleHiddenDim];
	for (int i = 0; i < kUpscaleHiddenDim; ++i) {
		float dH1 = 0.0f;
		for (int j = 0; j < kUpscaleHiddenDim; ++j) {
			dH1 += weights[kUpscaleW2Offset + i * kUpscaleHiddenDim + j] * dPre2[j];
		}
		dPre1[i] = dH1 * wf_upscale_relu_grad(cache.pre1[i]);
	}
	for (int j = 0; j < kUpscaleHiddenDim; ++j) {
		wf_upscale_accumulate_grad(gradAccum, kUpscaleB1Offset + j, dPre1[j]);
		for (int i = 0; i < kUpscaleInputDim; ++i) {
			wf_upscale_accumulate_grad(gradAccum, kUpscaleW1Offset + i * kUpscaleHiddenDim + j, cache.input[i] * dPre1[j]);
		}
	}
}

// Symmetrized relative-L2 loss gradient - see this project's own plan for
// why NRC's own denominator (pred_detached^2 + eps alone) isn't reused
// unmodified: NRC's targets are always network-bootstrapped (already
// smooth), but this feature's target is a REAL raw 1-2spp Monte Carlo
// sample, which can be a firefly outlier - a denominator that only
// downweights relative to `pred` does nothing when a small `pred` meets a
// huge outlier `target`. Both pred and target are treated as stop-gradient
// in the denominator (matches NRC's own "detached" convention - the
// formula below simply never differentiates through the denominator).
CPU_GPU void wf_upscale_loss_gradient(const float3& pred, const float3& target, float3& outDLossDPred) {
	const float denomX = pred.x * pred.x + target.x * target.x + kUpscaleRelativeLossEpsilon;
	const float denomY = pred.y * pred.y + target.y * target.y + kUpscaleRelativeLossEpsilon;
	const float denomZ = pred.z * pred.z + target.z * target.z + kUpscaleRelativeLossEpsilon;
	outDLossDPred.x = 2.0f * (pred.x - target.x) / denomX;
	outDLossDPred.y = 2.0f * (pred.y - target.y) / denomY;
	outDLossDPred.z = 2.0f * (pred.z - target.z) / denomZ;
}

CPU_GPU float wf_upscale_clip_gradient(float g) {
	if (g > kUpscaleGradClip) return kUpscaleGradClip;
	if (g < -kUpscaleGradClip) return -kUpscaleGradClip;
	return g;
}

// One Adam update for a single parameter - called with one thread per
// weight (kUpscaleNumWeights threads total), the only place that ever
// writes the live weight buffer - see wf_nrc_adam_step()'s own identical
// comment (wavefront_nrc_mlp.h) for the full rationale (gradient-
// accumulate + single serial apply step, `grad` must already be batch-
// averaged, `stepCount` is 1-based).
CPU_GPU void wf_upscale_adam_step(float& weight, float& m, float& v, float grad, int stepCount) {
	const float g = wf_upscale_clip_gradient(grad);
	m = kUpscaleAdamBeta1 * m + (1.0f - kUpscaleAdamBeta1) * g;
	v = kUpscaleAdamBeta2 * v + (1.0f - kUpscaleAdamBeta2) * g * g;
	const float t = (float)stepCount;
	const float mHat = m / (1.0f - powf(kUpscaleAdamBeta1, t));
	const float vHat = v / (1.0f - powf(kUpscaleAdamBeta2, t));
	weight -= kUpscaleLearningRate * mHat / (sqrtf(vHat) + kUpscaleAdamEpsilon);
}
