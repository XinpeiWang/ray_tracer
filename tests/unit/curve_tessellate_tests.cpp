// curve_tessellate_tests.cpp
// Unit tests for curve_tessellate::tessellate() in src/shared/curve_tessellate.h
//
// This file existed with no tests at all before this suite - the u/v axis
// mislabeling bug this pins down (p00->p10, the "u" edge, was actually the
// CIRCUMFERENTIAL direction instead of the documented "along the tube's
// length" direction) went unnoticed for exactly that reason: nothing before
// Round 7 Phase 3's hair-fiber-tangent code ever read dpdu directionally
// (only cross(dpdu,dpdv) for a face normal, which doesn't care which edge is
// "u" vs "v"), so a swapped u/v had no observable effect until then.

#include <gtest/gtest.h>
#include <cmath>

#include "curve_tessellate.h"

using curve_tessellate::Quad;
using curve_tessellate::tessellate;

namespace {
// A straight cubic Bezier segment along +Z from (0,0,0) to (0,0,3) - a
// degenerate (zero-curvature) curve, so every length-ring sits at a known,
// evenly-spaced z regardless of the Bezier basis math.
void straightZSegment(float cp[4][3]) {
	for (int i = 0; i < 4; ++i) {
		cp[i][0] = 0.0f;
		cp[i][1] = 0.0f;
		cp[i][2] = static_cast<float>(i);  // 0, 1, 2, 3
	}
}
} // namespace

TEST(CurveTessellate, UEdgeRunsAlongLengthNotCircumference) {
	// n_length=4, n_radial=8: ring spacing along Z is 3.0/4 = 0.75 per ring.
	float cp[4][3];
	straightZSegment(cp);

	std::vector<Quad> quads;
	tessellate(cp, 0.0f, 1.0f, /*width0=*/0.2f, /*width1=*/0.2f,
			   /*n_length=*/4, /*n_radial=*/8, quads);
	ASSERT_FALSE(quads.empty());

	const Quad &q = quads.front();  // ring i=0, angle j=0
	// p00->p10 is the documented "u" edge (BilinearPatchData: p10=(u=1,v=0)) -
	// it must move ALONG THE LENGTH (z changes by ~one ring spacing), not
	// around the circumference (which would leave z unchanged and only
	// rotate x/y at fixed radius).
	const float duz = q.p10[2] - q.p00[2];
	EXPECT_NEAR(duz, 0.75f, 1e-4f)
		<< "p00->p10 (the 'u' edge) should advance one length-ring (dz=0.75 "
		   "for a straight Z-axis curve with n_length=4), not stay on the "
		   "same ring";

	// p00->p01 is the "v" edge (p01=(u=0,v=1)) - it must be the
	// CIRCUMFERENTIAL direction: same length-ring (z unchanged), only the
	// angle around the tube changes.
	const float dvz = q.p01[2] - q.p00[2];
	EXPECT_NEAR(dvz, 0.0f, 1e-4f)
		<< "p00->p01 (the 'v' edge) should stay on the same length-ring "
		   "(dz=0) and only change the angle around the tube's "
		   "circumference, not advance along its length";

	// Sanity: p00 and p01 (same ring, different angle) sit at the same
	// distance from the Z axis (both on the tube's surface at radius
	// width/2), confirming p01 really is a rotation, not a length move in
	// disguise.
	const float r00 = std::sqrt(q.p00[0]*q.p00[0] + q.p00[1]*q.p00[1]);
	const float r01 = std::sqrt(q.p01[0]*q.p01[0] + q.p01[1]*q.p01[1]);
	EXPECT_NEAR(r00, r01, 1e-4f);
	EXPECT_NEAR(r00, 0.1f, 1e-4f);  // width/2 = 0.2/2
}

TEST(CurveTessellate, TaperedWidthAppliesAlongTheUEdgeNotTheVEdge) {
	// A strong taper (width0 != width1) should show up as the tube's radius
	// shrinking/growing along successive length-rings (the u direction) -
	// NOT as a radius change within a single ring (the v/circumference
	// direction, which must stay at a constant radius for a given ring).
	float cp[4][3];
	straightZSegment(cp);

	std::vector<Quad> quads;
	tessellate(cp, 0.0f, 1.0f, /*width0=*/0.4f, /*width1=*/0.05f,
			   /*n_length=*/4, /*n_radial=*/8, quads);
	ASSERT_FALSE(quads.empty());

	const Quad &q = quads.front();  // ring i=0 (should be near width0=0.4)
	const float r00 = std::sqrt(q.p00[0]*q.p00[0] + q.p00[1]*q.p00[1]);
	const float r01 = std::sqrt(q.p01[0]*q.p01[0] + q.p01[1]*q.p01[1]);
	const float r10 = std::sqrt(q.p10[0]*q.p10[0] + q.p10[1]*q.p10[1]);

	// Same ring (p00 vs p01, the v edge): radius unchanged.
	EXPECT_NEAR(r00, r01, 1e-4f);
	// Next ring along u (p00 vs p10): radius shrinks toward width1.
	EXPECT_LT(r10, r00)
		<< "the tube should taper along the 'u' edge (successive length-"
		   "rings), not within a single ring";
}
