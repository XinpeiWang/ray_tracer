// sppm_gpu_mix_hash_tests.cpp
// Isolated host-driven verification of sppm_mix_branch_hash01_test_kernel
// (a test-only mirror of sppm_programs.cu's own sppm_mix_branch_hash01(),
// see that kernel's own comment in sppm_kernels.cu for why a mirror rather
// than a direct call - sppm_programs.cu is an OptiX device-program module,
// not host-callable/device-linkable the way this plain-kernel file is, the
// same reason tests/unit/sppm_gpu_hash_grid_tests.cpp exists for
// sppm_hash_grid_insert_kernel instead of exercising OptiX device code
// directly).
//
// This closes a real coverage gap a code-review pass found: GPU SPPM's
// MaterialType::Mix support had zero automated tests, unlike every other
// material generalization in tests/integration/sppm_gpu_first_slice_test.cpp
// (each got a dedicated end-to-end render test). A full end-to-end render
// test isn't practical yet - no bundled scene puts a Mix material on
// sphere/quad geometry under GPU SPPM's own sphere/quad-only capability gate
// (see optix_types.h's sppm_gpu_material_supported() comment) - so this
// targets the actual piece that was buggy instead: the hash's own
// `variant` parameter, added specifically to fix a real-render-verified bug
// where GPU SPPM's camera pass (unjittered primary ray, no per-iteration
// seed term) froze a Mix material's sub-material resolution to the same
// choice for a given pixel across the ENTIRE render, rather than the
// intended converging stochastic blend. These tests prove the mechanism
// that fixes that: varying `variant` for a FIXED point changes the hash.
#include "gtest/gtest.h"
#include "optix_types.h"
#include "sppm_types.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

extern "C" void sppm_launch_mix_branch_hash01_test(
	const float3* d_points, const unsigned int* d_variants, int n, float* d_outHashes,
	cudaStream_t stream);

namespace {

class SppmGpuMixHashTest : public ::testing::Test {
  protected:
	void SetUp() override {
		int deviceCount = 0;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if (err != cudaSuccess || deviceCount == 0) {
			GTEST_SKIP() << "No CUDA device available";
		}
	}
};

// Runs the kernel for a batch of (point, variant) pairs and returns the
// resulting hashes on the host.
std::vector<float> RunHashKernel(const std::vector<float3>& points, const std::vector<unsigned int>& variants) {
	const int n = static_cast<int>(points.size());
	float3* d_points = nullptr;
	unsigned int* d_variants = nullptr;
	float* d_out = nullptr;
	cudaMalloc(&d_points, n * sizeof(float3));
	cudaMalloc(&d_variants, n * sizeof(unsigned int));
	cudaMalloc(&d_out, n * sizeof(float));
	cudaMemcpy(d_points, points.data(), n * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_variants, variants.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);

	sppm_launch_mix_branch_hash01_test(d_points, d_variants, n, d_out, nullptr);
	cudaStreamSynchronize(nullptr);

	std::vector<float> out(n);
	cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_points);
	cudaFree(d_variants);
	cudaFree(d_out);
	return out;
}

} // namespace

// variant==0 must mathematically contribute nothing extra to the sin()
// argument (`float(0) * 0.618... == 0.0f` exactly, unsigned-int-to-float
// conversion has no rounding at 0) - i.e. the SAME structural property that
// makes variant==0 reproduce CPU's branch_hash01()/the recursive and
// wavefront backends' own mix-hash copies (no +variant term at all) on
// whichever platform evaluates it. Deliberately NOT comparing against a
// host-computed sinf()/floorf() reference value here: this hash is a
// classic GLSL-style "cheap chaotic hash" (a small sin() argument
// difference gets amplified enormously by the *43758.5453f multiplier
// before the frac()-style wraparound), and host libm sinf() vs the GPU's
// own hardware sine approximation can legitimately disagree by more than
// float epsilon on the *input* to that multiply - a PRE-EXISTING property
// of this exact hash trick (already true of CPU's own double-precision
// branch_hash01() vs any GPU backend's float sinf(), independent of this
// file's own `variant` extension), not something to assert bit-parity on.
TEST_F(SppmGpuMixHashTest, VariantZeroIsReproducibleOnDevice) {
	std::vector<float3> points = { make_float3(1.0f, 2.0f, 3.0f), make_float3(1.0f, 2.0f, 3.0f) };
	std::vector<unsigned int> variants = { 0u, 0u };
	std::vector<float> hashes = RunHashKernel(points, variants);
	ASSERT_EQ(hashes.size(), 2u);
	EXPECT_FLOAT_EQ(hashes[0], hashes[1]);
	EXPECT_GE(hashes[0], 0.0f);
	EXPECT_LT(hashes[0], 1.0f);
}

// Every output must land in [0,1) regardless of point or variant, matching
// the function's own documented range - a raw floorf(h)-subtraction bug
// (e.g. a sign error) would show up as a value outside this range.
TEST_F(SppmGpuMixHashTest, OutputsStayInZeroOneRange) {
	std::vector<float3> points;
	std::vector<unsigned int> variants;
	for (int i = 0; i < 64; ++i) {
		points.push_back(make_float3(float(i) * 3.7f - 100.0f, float(i) * -1.3f, float(i) * 0.9f + 50.0f));
		variants.push_back(static_cast<unsigned int>(i * 37));
	}
	std::vector<float> hashes = RunHashKernel(points, variants);
	for (float h : hashes) {
		EXPECT_GE(h, 0.0f);
		EXPECT_LT(h, 1.0f);
	}
}

// The core fix under test: for a FIXED point, the hash must not be constant
// across a run of consecutive iteration indices - if it were, GPU SPPM's
// Mix resolution would still freeze per-pixel across the whole render (the
// exact bug this parameter was added to fix). Not asserting exact values
// (the formula is a black box by design) - just that varying `variant`
// alone, with the point held fixed, changes the result at least once
// across a reasonably long run, which a variant-less (or a broken,
// variant-ignoring) implementation could never do.
TEST_F(SppmGpuMixHashTest, VaryingIterationChangesHashForFixedPoint) {
	const float3 fixedPoint = make_float3(12.5f, -7.25f, 3.0f);
	const int kIterations = 32;
	std::vector<float3> points(kIterations, fixedPoint);
	std::vector<unsigned int> variants(kIterations);
	for (int i = 0; i < kIterations; ++i) variants[i] = static_cast<unsigned int>(i);

	std::vector<float> hashes = RunHashKernel(points, variants);
	ASSERT_EQ(hashes.size(), static_cast<size_t>(kIterations));

	int distinctValues = 0;
	for (int i = 1; i < kIterations; ++i) {
		if (std::fabs(hashes[i] - hashes[0]) > 1e-4f) ++distinctValues;
	}
	EXPECT_GT(distinctValues, kIterations / 2)
		<< "Hash barely varies across iteration indices for a fixed point - "
		<< "GPU SPPM Mix resolution would still freeze per-pixel across the whole render";
}

// Determinism: the SAME (point, variant) pair must always resolve to the
// SAME hash - within one SPPM iteration, a radiance hit and any later
// shadow ray toward the same occluder point must agree on which
// sub-material a Mix resolved to (this file's own header comment on
// sppm_mix_branch_hash01, sppm_programs.cu).
TEST_F(SppmGpuMixHashTest, SameInputsAreDeterministic) {
	std::vector<float3> points = { make_float3(4.0f, 5.0f, 6.0f), make_float3(4.0f, 5.0f, 6.0f) };
	std::vector<unsigned int> variants = { 7u, 7u };
	std::vector<float> hashes = RunHashKernel(points, variants);
	ASSERT_EQ(hashes.size(), 2u);
	EXPECT_FLOAT_EQ(hashes[0], hashes[1]);
}
