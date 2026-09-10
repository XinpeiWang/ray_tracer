// wavefront_restir_gi_tests.cpp
// Validation for gpu/optix/wavefront_restir_gi_math.h's pure-math ReSTIR GI
// primitives - the GI counterpart to wavefront_restir_tests.cpp (ReSTIR DI).
// Covers exactly the ONE piece GI needs beyond DI's already-tested RIS/
// reservoir-combine core (reused here unmodified): the shift-mapping
// Jacobian that DI's own reuse never needed (see wavefront_restir_gi_math.h's
// own header comment for why) and its effect when folded into
// restir_reservoir_combine's `otherPHatAtDstContext` argument.
//
// Tests:
// wf_restir_gi_jacobian
//   1.  Returns 1 when the new origin IS the original x0 (identity reconnect)
//   2.  Scales with the cosine ratio alone when distances are equal
//   3.  Scales with the inverse-squared-distance ratio alone when cosines
//       are equal (mirrors wavefront_restir_tests.cpp's own
//       WfRestirJacobian.ScalesWithSquaredDistanceRatio shape)
//   4.  A worked example combining both a cosine AND a distance change,
//       checked against the formula computed independently by hand
//   5.  Returns 0 when the ORIGINAL view of x1 was grazing (cosOrig ~ 0)
//   6.  Returns 0 when the new origin coincides with x1 (degenerate)
//
// Jacobian-adjusted restir_reservoir_combine (reusing DI's own combine math
// unmodified, per this feature's own design - see wavefront_restir_gi_math.h)
//   7.  A Jacobian of 1 (no geometric change) reproduces DI's own
//       FoldsInOthersFullM behavior exactly
//   8.  A Jacobian < 1 shrinks the reused candidate's effective weight
//       relative to an identical call with jacobian == 1 (sanity check that
//       the fold-in actually has an effect, not a no-op)

#include <gtest/gtest.h>
#include "wavefront_restir_gi_math.h"
#include <cmath>

namespace {

GpuGiSample make_gi_sample(float3 x0, float3 x1, float3 x1Normal, float pdfAtX0 = 1.0f) {
	GpuGiSample s;
	s.x0Point = x0;
	s.x1Point = x1;
	s.x1Normal = x1Normal;
	s.pdfAtX0 = pdfAtX0;
	s.radiance = make_float3(1.0f, 1.0f, 1.0f);
	return s;
}

} // namespace

// ============================================================
// wf_restir_gi_jacobian
// ============================================================

TEST(WfRestirGiJacobian, OneWhenNewOriginIsOriginalX0) {
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 x0 = make_float3(0.3f, 1.0f, 0.2f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(x0, s);
	EXPECT_NEAR(j, 1.0f, 1e-5f);
}

TEST(WfRestirGiJacobian, ScalesWithCosineRatioWhenDistancesEqual) {
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	// x0 directly above x1 at distance 1 -> cosOrig = 1.
	float3 x0 = make_float3(0.0f, 1.0f, 0.0f);
	// newOrigin at the SAME distance (1) but 45 degrees off-normal.
	float3 newOrigin = make_float3(0.70710678f, 0.70710678f, 0.0f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(newOrigin, s);
	EXPECT_NEAR(j, 0.70710678f, 1e-4f);
}

TEST(WfRestirGiJacobian, ScalesWithInverseSquaredDistanceRatioWhenCosinesEqual) {
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	// x0 directly above x1 at distance 1; newOrigin directly above at
	// distance 3 - same cosine (1) on both sides.
	float3 x0 = make_float3(0.0f, 1.0f, 0.0f);
	float3 newOrigin = make_float3(0.0f, 3.0f, 0.0f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(newOrigin, s);
	EXPECT_NEAR(j, (1.0f * 1.0f) / (3.0f * 3.0f), 1e-5f);
}

TEST(WfRestirGiJacobian, WorkedExampleCombiningCosineAndDistanceChange) {
	// x1 at the origin, normal +Y. x0 directly above at distance 1
	// (cosOrig = 1, distOrig = 1). newOrigin at (1,1,0): distance sqrt(2),
	// cosNew = dot(normalize(1,1,0), (0,1,0)) = 1/sqrt(2).
	// Expected |J| = (cosNew/cosOrig) * (distOrig^2/distNew^2)
	//              = (1/sqrt(2)) * (1/2) = 1/(2*sqrt(2)).
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 x0 = make_float3(0.0f, 1.0f, 0.0f);
	float3 newOrigin = make_float3(1.0f, 1.0f, 0.0f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(newOrigin, s);
	const float expected = 1.0f / (2.0f * std::sqrt(2.0f));
	EXPECT_NEAR(j, expected, 1e-4f);
}

TEST(WfRestirGiJacobian, ZeroWhenOriginalViewWasGrazing) {
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	// x0 exactly in x1's own tangent plane - cosOrig == 0.
	float3 x0 = make_float3(1.0f, 0.0f, 0.0f);
	float3 newOrigin = make_float3(0.0f, 1.0f, 0.0f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(newOrigin, s);
	EXPECT_FLOAT_EQ(j, 0.0f);
}

TEST(WfRestirGiJacobian, ZeroWhenNewOriginCoincidesWithX1) {
	float3 x1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 x1Normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 x0 = make_float3(0.0f, 1.0f, 0.0f);
	GpuGiSample s = make_gi_sample(x0, x1, x1Normal);
	float j = wf_restir_gi_jacobian(x1 /*newOrigin == x1*/, s);
	EXPECT_FLOAT_EQ(j, 0.0f);
}

// ============================================================
// Jacobian-adjusted restir_reservoir_combine (DI's own math, unmodified)
// ============================================================

TEST(GiReservoirCombine, JacobianOfOneMatchesDiOwnFoldsInOthersFullM) {
	GpuGiReservoir dst;
	GpuGiSample s0 = make_gi_sample(make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0));
	restir_reservoir_add(dst, s0, 1.0f, 1, 1.0f, 0.0f);

	GpuGiReservoir other;
	GpuGiSample s1 = make_gi_sample(make_float3(0, 0, 0), make_float3(2, 0, 0), make_float3(0, 1, 0));
	restir_reservoir_add(other, s1, 3.0f, 7, 3.0f, 0.0f);  // M=7
	restir_finalize(other);

	const int dstMBefore = dst.M;
	const float jacobian = 1.0f;  // no geometric change - identity reconnect
	restir_reservoir_combine(dst, other, /*otherPHatAtDstContext=*/2.0f * jacobian, 0.99f);
	EXPECT_EQ(dst.M, dstMBefore + other.M);
}

TEST(GiReservoirCombine, SmallerJacobianShrinksReusedCandidatesEffectiveWeight) {
	// Two otherwise-identical combine calls, differing only in the Jacobian
	// folded into otherPHatAtDstContext - the smaller Jacobian must produce
	// a strictly smaller resulting weightSum contribution from `other`.
	auto runCombine = [](float jacobian) -> float {
		GpuGiReservoir dst;
		GpuGiSample s0 = make_gi_sample(make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0));
		restir_reservoir_add(dst, s0, 1.0f, 1, 1.0f, 0.0f);
		float weightSumBefore = dst.weightSum;

		GpuGiReservoir other;
		GpuGiSample s1 = make_gi_sample(make_float3(0, 0, 0), make_float3(2, 0, 0), make_float3(0, 1, 0));
		restir_reservoir_add(other, s1, 4.0f, 1, 4.0f, 0.0f);
		restir_finalize(other);  // W = 4/(1*4) = 1

		// rand01 only affects whether `other`'s sample gets SELECTED, never
		// weightSum itself (restir_reservoir_add always adds risWeight to
		// weightSum unconditionally) - any fixed value isolates the
		// weightSum delta this test actually checks.
		const float freshPHat = 5.0f;
		restir_reservoir_combine(dst, other, freshPHat * jacobian, 0.5f);
		return dst.weightSum - weightSumBefore;
	};

	float deltaFullJacobian = runCombine(1.0f);
	float deltaHalfJacobian = runCombine(0.5f);
	EXPECT_GT(deltaFullJacobian, 0.0f);
	EXPECT_NEAR(deltaHalfJacobian, deltaFullJacobian * 0.5f, 1e-4f);
}
