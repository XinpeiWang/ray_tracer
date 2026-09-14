// wavefront_nrc_math_tests.cpp
// Validation for the Neural Radiance Cache (NRC) hand-written MLP math
// (gpu/optix/wavefront_nrc_mlp.h) and input feature encoding (gpu/optix/
// wavefront_nrc_encoding.h) - Live Preview's neural radiance cache feature,
// see this project's own plan. All CPU_GPU/plain-float functions, so no GPU
// hardware is required (mirrors wavefront_svgf_math_tests.cpp's own
// host-build-only approach).
//
// wf_nrc_backward()'s numerical-gradient check (NumericalGradientMatchesAnalytic
// below) is the single most important test in this file, per this project's
// own plan: a hand-written backward pass is the highest-bug-density code in
// the whole NRC feature, and a subtly wrong gradient won't crash anything -
// it'll silently train the cache toward garbage.

#include <gtest/gtest.h>
#include "wavefront_nrc_mlp.h"
#include "wavefront_nrc_encoding.h"
#include <cmath>
#include <random>

namespace {

// Deterministic pseudo-random weight/input fill - same seed every test run
// (reproducibility matters far more than true randomness here).
std::vector<float> makeRandomWeights(unsigned int seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
	std::vector<float> weights(kNrcNumWeights);
	for (float& w : weights) w = dist(rng);
	return weights;
}

std::vector<float> makeRandomInput(unsigned int seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	std::vector<float> input(kNrcInputDim);
	for (float& x : input) x = dist(rng);
	return input;
}

} // namespace

// ---------------------------------------------------------------------------
// wf_nrc_encode_scalar / wf_nrc_encode_features
// ---------------------------------------------------------------------------

TEST(WfNrcEncoding, ScalarEncodingIsPartitionOfUnity) {
	for (float x = 0.0f; x <= 1.0f; x += 0.05f) {
		float weights[kNrcBinsPerAxis];
		wf_nrc_encode_scalar(x, weights);
		float sum = 0.0f;
		for (int i = 0; i < kNrcBinsPerAxis; ++i) {
			EXPECT_GE(weights[i], 0.0f);
			sum += weights[i];
		}
		EXPECT_NEAR(sum, 1.0f, 1e-5f) << "x=" << x;
	}
}

TEST(WfNrcEncoding, ScalarEncodingClampsOutOfRange) {
	float below[kNrcBinsPerAxis], above[kNrcBinsPerAxis], atZero[kNrcBinsPerAxis];
	wf_nrc_encode_scalar(-5.0f, below);
	wf_nrc_encode_scalar(5.0f, above);
	wf_nrc_encode_scalar(0.0f, atZero);
	for (int i = 0; i < kNrcBinsPerAxis; ++i) {
		EXPECT_NEAR(below[i], atZero[i], 1e-6f);
	}
	float sum = 0.0f;
	for (int i = 0; i < kNrcBinsPerAxis; ++i) sum += above[i];
	EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(WfNrcEncoding, ScalarEncodingIsContinuousAcrossBinBoundaries) {
	const float binWidth = 1.0f / (float)(kNrcBinsPerAxis - 1);
	for (int bin = 0; bin < kNrcBinsPerAxis - 1; ++bin) {
		const float boundary = (float)bin * binWidth + binWidth * 0.5f;
		float wLeft[kNrcBinsPerAxis], wRight[kNrcBinsPerAxis];
		wf_nrc_encode_scalar(boundary - 1e-4f, wLeft);
		wf_nrc_encode_scalar(boundary + 1e-4f, wRight);
		for (int i = 0; i < kNrcBinsPerAxis; ++i) {
			EXPECT_NEAR(wLeft[i], wRight[i], 1e-3f) << "bin=" << bin << " i=" << i;
		}
	}
}

TEST(WfNrcEncoding, FullFeatureVectorHasExpectedLayoutAndRanges) {
	float features[kNrcInputDim];
	wf_nrc_encode_features(make_float3(5.0f, 5.0f, 5.0f), make_float3(0.0f, 0.0f, 1.0f),
							make_float3(0.0f, 1.0f, 0.0f), 0.5f,
							make_float3(0.8f, 0.2f, 0.1f),
							make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f),
							features);
	// Position (x=y=z=5 in a [0,10]^3 box -> normalized 0.5 on every axis)
	// one-blob blocks [0:4),[4:8),[8:12) should each be a partition of unity.
	for (int block = 0; block < 3; ++block) {
		float sum = 0.0f;
		for (int i = 0; i < kNrcBinsPerAxis; ++i) sum += features[block * kNrcBinsPerAxis + i];
		EXPECT_NEAR(sum, 1.0f, 1e-5f) << "position block " << block;
	}
	// Raw albedo passthrough at [32:35).
	EXPECT_FLOAT_EQ(features[32], 0.8f);
	EXPECT_FLOAT_EQ(features[33], 0.2f);
	EXPECT_FLOAT_EQ(features[34], 0.1f);
}

// ---------------------------------------------------------------------------
// wf_nrc_forward
// ---------------------------------------------------------------------------

TEST(WfNrcMlp, ForwardPassIsDeterministic) {
	std::vector<float> weights = makeRandomWeights(1);
	std::vector<float> input = makeRandomInput(2);
	NrcForwardCache cacheA, cacheB;
	wf_nrc_forward(weights.data(), input.data(), cacheA);
	wf_nrc_forward(weights.data(), input.data(), cacheB);
	for (int k = 0; k < kNrcOutputDim; ++k) {
		EXPECT_FLOAT_EQ(cacheA.out[k], cacheB.out[k]);
	}
}

TEST(WfNrcMlp, ForwardPassOutputIsNonNegative) {
	std::vector<float> weights = makeRandomWeights(3);
	for (unsigned int seed = 0; seed < 20; ++seed) {
		std::vector<float> input = makeRandomInput(seed + 100);
		NrcForwardCache cache;
		wf_nrc_forward(weights.data(), input.data(), cache);
		for (int k = 0; k < kNrcOutputDim; ++k) {
			EXPECT_GE(cache.out[k], 0.0f) << "seed=" << seed << " k=" << k;
			EXPECT_TRUE(std::isfinite(cache.out[k]));
		}
	}
}

// ---------------------------------------------------------------------------
// wf_nrc_backward - the mandatory correctness gate for this whole feature.
// ---------------------------------------------------------------------------

TEST(WfNrcMlp, NumericalGradientMatchesAnalytic) {
	std::vector<float> weights = makeRandomWeights(42);
	std::vector<float> input = makeRandomInput(7);
	std::vector<float> target = {0.3f, 0.6f, 0.1f};

	// Analytic gradient: forward, compute loss gradient wrt output, backward.
	NrcForwardCache cache;
	wf_nrc_forward(weights.data(), input.data(), cache);
	float dLossDOut[kNrcOutputDim];
	wf_nrc_loss_gradient(cache.out, target.data(), dLossDOut);
	std::vector<float> analyticGrad(kNrcNumWeights, 0.0f);
	wf_nrc_backward(weights.data(), cache, dLossDOut, analyticGrad.data());

	// Loss as a plain function of the weight vector - MUST match what
	// wf_nrc_loss_gradient() actually differentiates, which is a "detached"
	// relative-L2 (the denominator pred^2+eps is treated as a constant at
	// the CURRENT operating point, not re-differentiated - see that
	// function's own comment on why). The denominator below is therefore
	// fixed to its value at the original (unperturbed) weights, not
	// recomputed from the perturbed network each call - recomputing it
	// would numerically differentiate a DIFFERENT function (the full,
	// non-detached relative-L2) than the one the analytic gradient
	// represents, and the two disagree substantially whenever pred isn't
	// tiny relative to eps (an earlier version of this test recomputed the
	// denominator every call and failed here for exactly that reason - a
	// test bug, not a wf_nrc_backward() bug).
	float fixedDenom[kNrcOutputDim];
	for (int k = 0; k < kNrcOutputDim; ++k) {
		fixedDenom[k] = cache.out[k] * cache.out[k] + kNrcRelativeLossEpsilon;
	}
	auto loss = [&](const std::vector<float>& w) -> float {
		NrcForwardCache c;
		wf_nrc_forward(w.data(), input.data(), c);
		float total = 0.0f;
		for (int k = 0; k < kNrcOutputDim; ++k) {
			const float diff = c.out[k] - target[k];
			total += diff * diff / fixedDenom[k];
		}
		return total;
	};

	// Central-difference check on a representative subsample of weights
	// (every weight would be correct too, but this keeps the test fast) -
	// spans all 4 layers (W1/b1 .. W4/b4) so a bug isolated to one layer's
	// backward block can't hide behind another layer's correct one.
	const float eps = 1e-3f;
	const int sampleIndices[] = {
		kNrcW1Offset + 0, kNrcW1Offset + kNrcW1Count / 2, kNrcB1Offset,
		kNrcW2Offset + 0, kNrcW2Offset + kNrcW2Count / 2, kNrcB2Offset,
		kNrcW3Offset + 0, kNrcW3Offset + kNrcW3Count / 2, kNrcB3Offset,
		kNrcW4Offset + 0, kNrcW4Offset + kNrcW4Count - 1, kNrcB4Offset, kNrcB4Offset + kNrcOutputDim - 1,
	};
	for (int idx : sampleIndices) {
		std::vector<float> wPlus = weights, wMinus = weights;
		wPlus[idx] += eps;
		wMinus[idx] -= eps;
		const float numericalGrad = (loss(wPlus) - loss(wMinus)) / (2.0f * eps);
		const float relError = std::fabs(numericalGrad - analyticGrad[idx]) /
								std::fmax(1e-4f, std::fabs(numericalGrad));
		EXPECT_LT(relError, 5e-2f) << "weight idx=" << idx
									<< " numerical=" << numericalGrad
									<< " analytic=" << analyticGrad[idx];
	}
}

TEST(WfNrcMlp, BackwardAccumulatesRatherThanOverwrites) {
	std::vector<float> weights = makeRandomWeights(9);
	std::vector<float> input = makeRandomInput(11);
	NrcForwardCache cache;
	wf_nrc_forward(weights.data(), input.data(), cache);
	float dLossDOut[kNrcOutputDim] = {1.0f, 1.0f, 1.0f};

	std::vector<float> gradOnce(kNrcNumWeights, 0.0f);
	wf_nrc_backward(weights.data(), cache, dLossDOut, gradOnce.data());

	std::vector<float> gradTwice(kNrcNumWeights, 0.0f);
	wf_nrc_backward(weights.data(), cache, dLossDOut, gradTwice.data());
	wf_nrc_backward(weights.data(), cache, dLossDOut, gradTwice.data());

	for (int i = 0; i < kNrcNumWeights; ++i) {
		EXPECT_NEAR(gradTwice[i], 2.0f * gradOnce[i], 1e-4f) << "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// wf_nrc_adam_step
// ---------------------------------------------------------------------------

TEST(WfNrcMlp, AdamStepMatchesClosedFormFormula) {
	float weight = 1.0f, m = 0.0f, v = 0.0f;
	const float grad = 0.5f;
	wf_nrc_adam_step(weight, m, v, grad, /*stepCount=*/1);

	const float expectedM = (1.0f - kNrcAdamBeta1) * grad;
	const float expectedV = (1.0f - kNrcAdamBeta2) * grad * grad;
	const float expectedMHat = expectedM / (1.0f - kNrcAdamBeta1);
	const float expectedVHat = expectedV / (1.0f - kNrcAdamBeta2);
	const float expectedWeight = 1.0f - kNrcLearningRate * expectedMHat / (std::sqrt(expectedVHat) + kNrcAdamEpsilon);

	EXPECT_NEAR(m, expectedM, 1e-6f);
	EXPECT_NEAR(v, expectedV, 1e-6f);
	EXPECT_NEAR(weight, expectedWeight, 1e-5f);
}

TEST(WfNrcMlp, AdamStepClipsLargeGradients) {
	float weightClipped = 1.0f, mClipped = 0.0f, vClipped = 0.0f;
	float weightAtCap = 1.0f, mAtCap = 0.0f, vAtCap = 0.0f;
	wf_nrc_adam_step(weightClipped, mClipped, vClipped, kNrcGradClip * 100.0f, 1);
	wf_nrc_adam_step(weightAtCap, mAtCap, vAtCap, kNrcGradClip, 1);
	EXPECT_NEAR(weightClipped, weightAtCap, 1e-6f);
	EXPECT_NEAR(mClipped, mAtCap, 1e-6f);
}

TEST(WfNrcMlp, AdamMovesWeightTowardReducingLoss) {
	// A positive gradient (dLoss/dWeight > 0) means increasing the weight
	// increases the loss - Adam should therefore DECREASE the weight.
	float weight = 1.0f, m = 0.0f, v = 0.0f;
	wf_nrc_adam_step(weight, m, v, /*grad=*/2.0f, 1);
	EXPECT_LT(weight, 1.0f);
}

// ---------------------------------------------------------------------------
// End-to-end sanity: overfit a tiny synthetic dataset.
// ---------------------------------------------------------------------------

TEST(WfNrcMlp, TrainingReducesLossOnFixedTarget) {
	std::vector<float> weights = makeRandomWeights(123);
	std::vector<float> m(kNrcNumWeights, 0.0f), v(kNrcNumWeights, 0.0f);
	std::vector<float> input = makeRandomInput(456);
	const float target[kNrcOutputDim] = {0.5f, 0.25f, 0.1f};

	auto computeLoss = [&]() -> float {
		NrcForwardCache cache;
		wf_nrc_forward(weights.data(), input.data(), cache);
		float total = 0.0f;
		for (int k = 0; k < kNrcOutputDim; ++k) {
			const float diff = cache.out[k] - target[k];
			total += diff * diff;
		}
		return total;
	};

	const float initialLoss = computeLoss();

	for (int step = 1; step <= 200; ++step) {
		NrcForwardCache cache;
		wf_nrc_forward(weights.data(), input.data(), cache);
		float dLossDOut[kNrcOutputDim];
		wf_nrc_loss_gradient(cache.out, target, dLossDOut);
		std::vector<float> gradAccum(kNrcNumWeights, 0.0f);
		wf_nrc_backward(weights.data(), cache, dLossDOut, gradAccum.data());
		for (int i = 0; i < kNrcNumWeights; ++i) {
			wf_nrc_adam_step(weights[i], m[i], v[i], gradAccum[i], step);
		}
	}

	const float finalLoss = computeLoss();
	EXPECT_LT(finalLoss, initialLoss * 0.1f)
		<< "initial=" << initialLoss << " final=" << finalLoss;
}
