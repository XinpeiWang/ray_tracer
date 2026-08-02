// ---------------------------------------------------------------------------
// animated_transform_tests.cpp
// Unit tests for animated_transform.h
//
// Mirrors pbrt-v4 AnimatedTransform (src/pbrt/util/transform.h/.cpp).
//
// Tests cover:
//   Construction:  default, static (single transform), animated pair
//   IsAnimated:    static vs animated
//   HasScale:      pure rotation/translation vs scaled matrix
//   Interpolate:   boundary clamp, midpoint, pure translation, pure rotation,
//                  TRS combined
//   apply_point:   static and animated
//   apply_vector:  no translation component
//   apply_normal:  inverse-transpose (orthogonal case)
//   apply_ray:     origin+direction both transformed
//   apply_inverse: round-trip point through forward+inverse
//   Slerp:         shortest path (dot<0 flip)
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/animated_transform.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static const double kEps = 1e-9;

static AT_Mat44 make_translate(double tx, double ty, double tz) {
	return at_translate(tx, ty, tz);
}

// Build a rotation matrix about Z by angle (radians).
static AT_Mat44 make_rot_z(double angle) {
	double c = std::cos(angle), s = std::sin(angle);
	AT_Mat44 m;
	m.m[0][0] = c;  m.m[0][1] = -s;
	m.m[1][0] = s;  m.m[1][1] =  c;
	return m;
}

// Build a uniform scale matrix.
static AT_Mat44 make_scale(double sx, double sy, double sz) {
	AT_Mat44 m;
	m.m[0][0] = sx; m.m[1][1] = sy; m.m[2][2] = sz;
	return m;
}

static bool near3(const double a[3], double x, double y, double z, double eps = kEps) {
	return std::fabs(a[0]-x) < eps && std::fabs(a[1]-y) < eps && std::fabs(a[2]-z) < eps;
}

// ---------------------------------------------------------------------------
// Construction & IsAnimated
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, DefaultIsNotAnimated) {
	AnimatedTransform at;
	EXPECT_FALSE(at.IsAnimated());
}

TEST(AnimatedTransform, StaticCtorIsNotAnimated) {
	AnimatedTransform at(make_translate(1, 2, 3));
	EXPECT_FALSE(at.IsAnimated());
}

TEST(AnimatedTransform, SameStartEndIsNotAnimated) {
	AT_Mat44 m = make_translate(5, 0, 0);
	AnimatedTransform at(m, 0.0, m, 1.0);
	EXPECT_FALSE(at.IsAnimated());
}

TEST(AnimatedTransform, DifferentStartEndIsAnimated) {
	AnimatedTransform at(make_translate(0,0,0), 0.0, make_translate(10,0,0), 1.0);
	EXPECT_TRUE(at.IsAnimated());
}

// ---------------------------------------------------------------------------
// HasScale
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, PureTranslationHasNoScale) {
	AnimatedTransform at(make_translate(1, 2, 3));
	EXPECT_FALSE(at.HasScale());
}

TEST(AnimatedTransform, PureRotationHasNoScale) {
	AnimatedTransform at(make_rot_z(0.5));
	EXPECT_FALSE(at.HasScale());
}

TEST(AnimatedTransform, ScaleMatrixHasScale) {
	AnimatedTransform at(make_scale(2, 1, 1));
	EXPECT_TRUE(at.HasScale());
}

// ---------------------------------------------------------------------------
// Interpolate -- boundary clamp
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, InterpolateBelowStartReturnsStart) {
	AT_Mat44 s = make_translate(0, 0, 0);
	AT_Mat44 e = make_translate(10, 0, 0);
	AnimatedTransform at(s, 1.0, e, 2.0);

	AT_Mat44 r = at.Interpolate(0.5); // below startTime=1
	EXPECT_NEAR(r.m[0][3], 0.0, kEps);
}

TEST(AnimatedTransform, InterpolateAboveEndReturnsEnd) {
	AT_Mat44 s = make_translate(0, 0, 0);
	AT_Mat44 e = make_translate(10, 0, 0);
	AnimatedTransform at(s, 0.0, e, 1.0);

	AT_Mat44 r = at.Interpolate(2.0); // above endTime=1
	EXPECT_NEAR(r.m[0][3], 10.0, kEps);
}

TEST(AnimatedTransform, InterpolateAtStartReturnsStart) {
	AT_Mat44 s = make_translate(3, 0, 0);
	AT_Mat44 e = make_translate(9, 0, 0);
	AnimatedTransform at(s, 0.0, e, 1.0);

	AT_Mat44 r = at.Interpolate(0.0);
	EXPECT_NEAR(r.m[0][3], 3.0, kEps);
}

TEST(AnimatedTransform, InterpolateAtEndReturnsEnd) {
	AT_Mat44 s = make_translate(3, 0, 0);
	AT_Mat44 e = make_translate(9, 0, 0);
	AnimatedTransform at(s, 0.0, e, 1.0);

	AT_Mat44 r = at.Interpolate(1.0);
	EXPECT_NEAR(r.m[0][3], 9.0, kEps);
}

// ---------------------------------------------------------------------------
// Interpolate -- pure translation midpoint
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, InterpolatePureTranslationMidpoint) {
	AT_Mat44 s = make_translate(0, 0, 0);
	AT_Mat44 e = make_translate(10, 4, -2);
	AnimatedTransform at(s, 0.0, e, 1.0);

	AT_Mat44 r = at.Interpolate(0.5);
	EXPECT_NEAR(r.m[0][3], 5.0,  1e-6);
	EXPECT_NEAR(r.m[1][3], 2.0,  1e-6);
	EXPECT_NEAR(r.m[2][3], -1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// apply_point -- static transform
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, ApplyPointStaticTranslation) {
	AnimatedTransform at(make_translate(3, 1, -2));
	double p[3] = {1, 0, 0}, out[3];
	at.apply_point(p, 0.5, out);
	EXPECT_NEAR(out[0], 4.0,  kEps);
	EXPECT_NEAR(out[1], 1.0,  kEps);
	EXPECT_NEAR(out[2], -2.0, kEps);
}

TEST(AnimatedTransform, ApplyPointStaticOriginUnchangedByIdentity) {
	AnimatedTransform at;
	double p[3] = {7, -3, 5}, out[3];
	at.apply_point(p, 0.0, out);
	EXPECT_NEAR(out[0], 7.0,  kEps);
	EXPECT_NEAR(out[1], -3.0, kEps);
	EXPECT_NEAR(out[2], 5.0,  kEps);
}

// ---------------------------------------------------------------------------
// apply_point -- animated (midpoint translation)
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, ApplyPointAnimatedTranslation) {
	AT_Mat44 s = make_translate(0,0,0);
	AT_Mat44 e = make_translate(6,0,0);
	AnimatedTransform at(s, 0.0, e, 1.0);
	double p[3] = {1, 0, 0}, out[3];
	at.apply_point(p, 0.5, out);
	// At t=0.5: translation = (3,0,0), so p=(1,0,0) -> (4,0,0)
	EXPECT_NEAR(out[0], 4.0, 1e-6);
	EXPECT_NEAR(out[1], 0.0, 1e-6);
	EXPECT_NEAR(out[2], 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// apply_vector -- no translation
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, ApplyVectorNoTranslation) {
	AnimatedTransform at(make_translate(100, 200, 300));
	double v[3] = {1, 0, 0}, out[3];
	at.apply_vector(v, 0.0, out);
	// Translation doesn't affect vectors
	EXPECT_NEAR(out[0], 1.0, kEps);
	EXPECT_NEAR(out[1], 0.0, kEps);
	EXPECT_NEAR(out[2], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// apply_normal -- inverse-transpose (for pure rotation, equals the rotation)
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, ApplyNormalPureRotationZAxis) {
	// Rotate 90° about Z: X-axis maps to Y-axis
	AnimatedTransform at(make_rot_z(M_PI / 2.0));
	double n[3] = {1, 0, 0}, out[3];
	at.apply_normal(n, 0.0, out);
	// For pure rotation, normal transforms the same as a vector
	EXPECT_NEAR(out[0], 0.0, 1e-9);
	EXPECT_NEAR(out[1], 1.0, 1e-9);
	EXPECT_NEAR(out[2], 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// apply_ray
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, ApplyRayTranslatesOriginNotDirection) {
	AnimatedTransform at(make_translate(1, 2, 3));
	double o[3] = {0, 0, 0}, d[3] = {0, 0, 1};
	double oo[3], od[3];
	at.apply_ray(o, d, 0.0, oo, od);
	EXPECT_NEAR(oo[0], 1.0, kEps);
	EXPECT_NEAR(oo[1], 2.0, kEps);
	EXPECT_NEAR(oo[2], 3.0, kEps);
	EXPECT_NEAR(od[0], 0.0, kEps);
	EXPECT_NEAR(od[1], 0.0, kEps);
	EXPECT_NEAR(od[2], 1.0, kEps);
}

// ---------------------------------------------------------------------------
// apply_inverse_point -- round-trip
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, InversePointRoundTrip) {
	AT_Mat44 m = make_translate(3, -1, 7);
	AnimatedTransform at(m);
	double p[3] = {5, 2, -4}, fwd[3], inv[3];
	at.apply_point(p, 0.5, fwd);
	at.apply_inverse_point(fwd, 0.5, inv);
	EXPECT_NEAR(inv[0], p[0], 1e-10);
	EXPECT_NEAR(inv[1], p[1], 1e-10);
	EXPECT_NEAR(inv[2], p[2], 1e-10);
}

TEST(AnimatedTransform, InverseVectorRoundTrip) {
	AnimatedTransform at(make_rot_z(0.7));
	double v[3] = {1, 1, 0}, fwd[3], inv_v[3];
	at.apply_vector(v, 0.0, fwd);
	at.apply_inverse_vector(fwd, 0.0, inv_v);
	EXPECT_NEAR(inv_v[0], v[0], 1e-10);
	EXPECT_NEAR(inv_v[1], v[1], 1e-10);
	EXPECT_NEAR(inv_v[2], v[2], 1e-10);
}

// ---------------------------------------------------------------------------
// Interpolate -- pure rotation slerp
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, InterpolatePureRotationMidpointVector) {
	// Animate from identity to 180° rotation about Z
	AT_Mat44 start = at_identity();
	AT_Mat44 end   = make_rot_z(M_PI);
	AnimatedTransform at(start, 0.0, end, 1.0);

	// At t=0.5 we expect 90° rotation about Z
	double v[3] = {1, 0, 0}, out[3];
	at.apply_vector(v, 0.5, out);
	EXPECT_NEAR(out[0],  0.0, 1e-6);
	EXPECT_NEAR(out[1],  1.0, 1e-6);
	EXPECT_NEAR(out[2],  0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Quaternion slerp shortest path (dot < 0 flip)
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, SlerpShortestPath) {
	// R[0] and R[1] in opposite hemispheres -- constructor should flip R[1]
	AT_Quat q1 = at_normalize(AT_Quat(0, 0, 0.1, 1));
	AT_Quat q2_neg = at_normalize(AT_Quat(0, 0, -0.1, -1)); // opposite hemisphere

	AT_Quat q_mid = at_slerp(0.5, q1, q2_neg);
	// Should converge toward identity (not cross through 180°)
	// Both are near-identity quats; result should also be near-identity
	EXPECT_GT(std::fabs(q_mid.w), 0.9);
}

// ---------------------------------------------------------------------------
// Static animated transform: Interpolate always returns startTransform
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, StaticInterpolateIgnoresTime) {
	AT_Mat44 m = make_translate(7, 8, 9);
	AnimatedTransform at(m);

	for (double t : {-1.0, 0.0, 0.5, 1.0, 2.0}) {
		AT_Mat44 r = at.Interpolate(t);
		EXPECT_NEAR(r.m[0][3], 7.0, kEps) << "t=" << t;
		EXPECT_NEAR(r.m[1][3], 8.0, kEps) << "t=" << t;
		EXPECT_NEAR(r.m[2][3], 9.0, kEps) << "t=" << t;
	}
}

// ---------------------------------------------------------------------------
// TRS composed animated transform
// ---------------------------------------------------------------------------

TEST(AnimatedTransform, TRSComposedMidpointPoint) {
	// Start: identity
	// End: translate (4,0,0) then rotate 90° about Z
	// Build end matrix as Translate * RotZ
	AT_Mat44 trans = make_translate(4, 0, 0);
	AT_Mat44 rot   = make_rot_z(M_PI / 2.0);
	AT_Mat44 end   = at_mul(trans, rot);

	AnimatedTransform at(at_identity(), 0.0, end, 1.0);

	// At t=0: point (1,0,0) -> (1,0,0)
	double p[3] = {1, 0, 0}, out0[3];
	at.apply_point(p, 0.0, out0);
	EXPECT_NEAR(out0[0], 1.0, 1e-9);
	EXPECT_NEAR(out0[1], 0.0, 1e-9);

	// At t=1: point (1,0,0) -> RotZ(90°)(1,0,0) + (4,0,0) = (0,1,0)+(4,0,0) = (4,1,0)
	double out1[3];
	at.apply_point(p, 1.0, out1);
	EXPECT_NEAR(out1[0], 4.0, 1e-9);
	EXPECT_NEAR(out1[1], 1.0, 1e-9);
	EXPECT_NEAR(out1[2], 0.0, 1e-9);
}
