#pragma once
// wavefront_nrc_mlp.h -- Neural Radiance Cache (NRC) hand-written MLP
// forward pass, backward pass, and Adam optimizer step, wavefront GPU
// backend, Live Preview only. See this project's own plan and
// wavefront_nrc_types.h's own header comment for the full design rationale.
//
// This is the highest-bug-density code in the whole feature: a subtly wrong
// gradient here won't crash anything, it'll silently train the cache toward
// garbage, which looks like "NRC just doesn't help" rather than an obvious
// bug. tests/unit/wavefront_nrc_math_tests.cpp's numerical-gradient check
// (central-difference vs. wf_nrc_backward()'s analytic output) is the
// mandatory gate for this file, not an optional nice-to-have.
//
// One CUDA thread = one full inference/training record's forward+backward
// pass, entirely in registers/local arrays - the same "one thread, one
// item, full eval in registers" pattern every evaluate_materials* kernel
// already uses for BSDF evaluation (no shared memory, no warp cooperation,
// no fp16/WMMA - see this project's own plan for why tensor cores don't
// help this workload's shape).
//
// Network: Input(38) -> Linear+ReLU(64) -> Linear+ReLU(64) -> Linear+ReLU(64)
// -> Linear(3) -> softplus. Weight layout/offsets: wavefront_nrc_types.h.
//
// Deliberately dependency-free beyond wavefront_nrc_types.h's layout
// constants - no wf_rand()/wf_pcg() calls, can be included from anywhere,
// host or device side, in any order (matches wavefront_guiding.h's own
// convention).

#include <cuda_runtime.h>
// Relative path - see probe_grid_types.h's own identical comment for why.
#include "../../src/shared/cpu_gpu.h"
#include "wavefront_nrc_types.h"

// Per-call activation cache, filled by wf_nrc_forward() and consumed by
// wf_nrc_backward() - the forward pass's pre-activation values (pre1/2/3/4)
// are needed for ReLU's/softplus' own derivatives, and its post-activation
// values (input/h1/h2/h3) are needed as the "left-hand side" of each
// layer's weight gradient (dL/dW[i][j] = leftActivation[i] * dL/dPre[j]).
struct NrcForwardCache {
	float input[kNrcInputDim];
	float pre1[kNrcHiddenDim];
	float h1[kNrcHiddenDim];
	float pre2[kNrcHiddenDim];
	float h2[kNrcHiddenDim];
	float pre3[kNrcHiddenDim];
	float h3[kNrcHiddenDim];
	float pre4[kNrcOutputDim];
	float out[kNrcOutputDim];
};

CPU_GPU float wf_nrc_relu(float x) { return x > 0.0f ? x : 0.0f; }
CPU_GPU float wf_nrc_relu_grad(float preActivation) { return preActivation > 0.0f ? 1.0f : 0.0f; }

// Numerically stable softplus: log(1+exp(x)) computed as
// max(x,0) + log1p(exp(-|x|)) - avoids overflow for large positive x and
// avoids the exp(x) term ever seeing a large positive argument.
CPU_GPU float wf_nrc_softplus(float x) {
	return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}
// d(softplus(x))/dx = sigmoid(x), computed directly rather than via
// softplus's own output (which would need an extra inverse) - stable for
// both signs of x since expf(-x) for very negative x just saturates to a
// huge (but finite for any x this network will realistically see) value.
CPU_GPU float wf_nrc_softplus_grad(float x) {
	return 1.0f / (1.0f + expf(-x));
}

// Accumulates `value` into gradAccum[idx] - atomicAdd on the device
// (many threads' backward passes share one gradAccum buffer across a
// training batch), plain += on the host (unit tests are single-threaded,
// and atomicAdd(float*, float) isn't available outside device code).
// __CUDA_ARCH__ is only ever defined during nvcc's DEVICE compilation
// pass, so this correctly resolves per translation-unit-pass, not per
// platform (cpu_gpu.h's own CPU_GPU macro uses the same __CUDACC__-vs-not
// split for the same reason).
CPU_GPU void wf_nrc_accumulate_grad(float* gradAccum, int idx, float value) {
#if defined(__CUDA_ARCH__)
	atomicAdd(&gradAccum[idx], value);
#else
	gradAccum[idx] += value;
#endif
}

// Forward pass: `weights` is a flat kNrcNumWeights buffer laid out per
// wavefront_nrc_types.h, `encodedInput` is kNrcInputDim floats (see
// wavefront_nrc_encoding.h). Fills every field of `cache` (including the
// final `out`) - callers doing pure inference (no backward pass planned)
// simply ignore everything but cache.out.
CPU_GPU void wf_nrc_forward(const float* weights, const float* encodedInput, NrcForwardCache& cache) {
	for (int i = 0; i < kNrcInputDim; ++i) cache.input[i] = encodedInput[i];

	for (int j = 0; j < kNrcHiddenDim; ++j) {
		float acc = weights[kNrcB1Offset + j];
		for (int i = 0; i < kNrcInputDim; ++i) {
			acc += cache.input[i] * weights[kNrcW1Offset + i * kNrcHiddenDim + j];
		}
		cache.pre1[j] = acc;
		cache.h1[j] = wf_nrc_relu(acc);
	}
	for (int j = 0; j < kNrcHiddenDim; ++j) {
		float acc = weights[kNrcB2Offset + j];
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			acc += cache.h1[i] * weights[kNrcW2Offset + i * kNrcHiddenDim + j];
		}
		cache.pre2[j] = acc;
		cache.h2[j] = wf_nrc_relu(acc);
	}
	for (int j = 0; j < kNrcHiddenDim; ++j) {
		float acc = weights[kNrcB3Offset + j];
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			acc += cache.h2[i] * weights[kNrcW3Offset + i * kNrcHiddenDim + j];
		}
		cache.pre3[j] = acc;
		cache.h3[j] = wf_nrc_relu(acc);
	}
	for (int k = 0; k < kNrcOutputDim; ++k) {
		float acc = weights[kNrcB4Offset + k];
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			acc += cache.h3[i] * weights[kNrcW4Offset + i * kNrcOutputDim + k];
		}
		cache.pre4[k] = acc;
		cache.out[k] = wf_nrc_softplus(acc);
	}
}

// Backward pass: given dLoss/dOut (kNrcOutputDim floats, from
// wf_nrc_loss_gradient() below) and the forward pass's own cache,
// accumulates dLoss/dWeight for every weight/bias into gradAccum (a flat
// kNrcNumWeights buffer, same layout as `weights` - see
// wf_nrc_accumulate_grad()'s own comment on why this is an accumulate, not
// an overwrite). Does NOT propagate a gradient past the input layer -
// nothing downstream ever needs dLoss/dEncodedInput, since the encoded
// input is observed data, not an optimized parameter.
CPU_GPU void wf_nrc_backward(const float* weights, const NrcForwardCache& cache,
							  const float* dLossDOut, float* gradAccum) {
	float dPre4[kNrcOutputDim];
	for (int k = 0; k < kNrcOutputDim; ++k) {
		dPre4[k] = dLossDOut[k] * wf_nrc_softplus_grad(cache.pre4[k]);
		wf_nrc_accumulate_grad(gradAccum, kNrcB4Offset + k, dPre4[k]);
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			wf_nrc_accumulate_grad(gradAccum, kNrcW4Offset + i * kNrcOutputDim + k, cache.h3[i] * dPre4[k]);
		}
	}

	float dPre3[kNrcHiddenDim];
	for (int i = 0; i < kNrcHiddenDim; ++i) {
		float dH3 = 0.0f;
		for (int k = 0; k < kNrcOutputDim; ++k) {
			dH3 += weights[kNrcW4Offset + i * kNrcOutputDim + k] * dPre4[k];
		}
		dPre3[i] = dH3 * wf_nrc_relu_grad(cache.pre3[i]);
	}
	for (int j = 0; j < kNrcHiddenDim; ++j) {
		wf_nrc_accumulate_grad(gradAccum, kNrcB3Offset + j, dPre3[j]);
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			wf_nrc_accumulate_grad(gradAccum, kNrcW3Offset + i * kNrcHiddenDim + j, cache.h2[i] * dPre3[j]);
		}
	}

	float dPre2[kNrcHiddenDim];
	for (int i = 0; i < kNrcHiddenDim; ++i) {
		float dH2 = 0.0f;
		for (int j = 0; j < kNrcHiddenDim; ++j) {
			dH2 += weights[kNrcW3Offset + i * kNrcHiddenDim + j] * dPre3[j];
		}
		dPre2[i] = dH2 * wf_nrc_relu_grad(cache.pre2[i]);
	}
	for (int j = 0; j < kNrcHiddenDim; ++j) {
		wf_nrc_accumulate_grad(gradAccum, kNrcB2Offset + j, dPre2[j]);
		for (int i = 0; i < kNrcHiddenDim; ++i) {
			wf_nrc_accumulate_grad(gradAccum, kNrcW2Offset + i * kNrcHiddenDim + j, cache.h1[i] * dPre2[j]);
		}
	}

	float dPre1[kNrcHiddenDim];
	for (int i = 0; i < kNrcHiddenDim; ++i) {
		float dH1 = 0.0f;
		for (int j = 0; j < kNrcHiddenDim; ++j) {
			dH1 += weights[kNrcW2Offset + i * kNrcHiddenDim + j] * dPre2[j];
		}
		dPre1[i] = dH1 * wf_nrc_relu_grad(cache.pre1[i]);
	}
	for (int j = 0; j < kNrcHiddenDim; ++j) {
		wf_nrc_accumulate_grad(gradAccum, kNrcB1Offset + j, dPre1[j]);
		for (int i = 0; i < kNrcInputDim; ++i) {
			wf_nrc_accumulate_grad(gradAccum, kNrcW1Offset + i * kNrcHiddenDim + j, cache.input[i] * dPre1[j]);
		}
	}
}

// Relative-L2 loss gradient (matches the NRC paper's own choice for keeping
// gradients well-scaled across HDR radiance's dynamic range): per channel,
// loss = (pred-target)^2 / (pred_detached^2 + eps), so
// dLoss/dPred = 2*(pred-target) / (pred_detached^2 + eps). `pred_detached`
// is numerically identical to `pred` (it's a stop-gradient, not a different
// value) - the gradient formula below simply doesn't differentiate through
// the denominator, which is what "detached" means here.
CPU_GPU void wf_nrc_loss_gradient(const float* pred, const float* target, float* outDLossDPred) {
	for (int k = 0; k < kNrcOutputDim; ++k) {
		const float denom = pred[k] * pred[k] + kNrcRelativeLossEpsilon;
		outDLossDPred[k] = 2.0f * (pred[k] - target[k]) / denom;
	}
}

CPU_GPU float wf_nrc_clip_gradient(float g) {
	if (g > kNrcGradClip) return kNrcGradClip;
	if (g < -kNrcGradClip) return -kNrcGradClip;
	return g;
}

// One Adam update for a single parameter - called with one thread per
// weight (kNrcNumWeights threads total) from nrc_apply_gradients
// (wavefront_kernels_nrc.cu), the only place that ever writes the live
// weight buffer (see this project's own plan's "gradient accumulation +
// single serial apply step" design). `grad` should already be the BATCH-
// AVERAGED gradient (gradAccum[idx] / numContributingRecords) - this
// function only clips and applies Adam, it does not average. `stepCount`
// is the 1-based global training-step counter (Adam's bias-correction
// divides by 1-beta^stepCount, which is undefined at stepCount==0 - callers
// must pass >=1).
CPU_GPU void wf_nrc_adam_step(float& weight, float& m, float& v, float grad, int stepCount) {
	const float g = wf_nrc_clip_gradient(grad);
	m = kNrcAdamBeta1 * m + (1.0f - kNrcAdamBeta1) * g;
	v = kNrcAdamBeta2 * v + (1.0f - kNrcAdamBeta2) * g * g;
	const float t = (float)stepCount;
	const float mHat = m / (1.0f - powf(kNrcAdamBeta1, t));
	const float vHat = v / (1.0f - powf(kNrcAdamBeta2, t));
	weight -= kNrcLearningRate * mHat / (sqrtf(vHat) + kNrcAdamEpsilon);
}
