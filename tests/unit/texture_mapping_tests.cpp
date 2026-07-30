// texture_mapping_tests.cpp
// Unit tests for src/shared/texture_mapping.h
// pbrt-v4 reference: src/pbrt/textures.h
//
// Tests cover:
//   TextureTransform   -- identity, scale, translate apply to points/vectors
//   TextureEvalContext -- construction
//   UVMapping          -- identity, scale, offset, differentials
//   SphericalMapping   -- +z axis maps to (0, 0) theta=0
//   CylindricalMapping -- known atan2 result for +x axis
//   PlanarMapping      -- dot-product projection
//   TextureMapping2D   -- variant dispatch
//   PointTransformMapping / TextureMapping3D -- 3D passthrough

#include <gtest/gtest.h>
#include "../../src/shared/texture_mapping.h"
#include <cmath>

static constexpr float kEps = 1e-5f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool near(float a, float b, float eps = kEps) {
	return std::fabs(a - b) <= eps;
}

static TextureEvalContext MakeCtx(
	TxPoint3f p = {0,0,0},
	TxPoint2f uv = {0.5f, 0.25f},
	float dudx = 0.1f, float dudy = 0.f,
	float dvdx = 0.f,  float dvdy = 0.1f,
	TxVector3f dpdx = {0.1f,0,0},
	TxVector3f dpdy = {0,0.1f,0})
{
	return TextureEvalContext(p, dpdx, dpdy,
							  {0,0,1}, uv,
							  dudx, dudy, dvdx, dvdy);
}

// ---------------------------------------------------------------------------
// TextureTransform tests
// ---------------------------------------------------------------------------

TEST(TextureTransformTest, IdentityLeavesPointUnchanged) {
	TextureTransform t;
	TxPoint3f p = {1.f, 2.f, 3.f};
	auto q = t(p);
	EXPECT_TRUE(near(q.x, 1.f));
	EXPECT_TRUE(near(q.y, 2.f));
	EXPECT_TRUE(near(q.z, 3.f));
}

TEST(TextureTransformTest, IdentityLeavesVectorUnchanged) {
	TextureTransform t;
	TxVector3f v = {4.f, 5.f, 6.f};
	auto w = t(v);
	EXPECT_TRUE(near(w.x, 4.f));
	EXPECT_TRUE(near(w.y, 5.f));
	EXPECT_TRUE(near(w.z, 6.f));
}

TEST(TextureTransformTest, ScaleAppliesCorrectly) {
	auto t = TextureTransform::Scale(2.f, 3.f, 4.f);
	TxPoint3f p = {1.f, 1.f, 1.f};
	auto q = t(p);
	EXPECT_TRUE(near(q.x, 2.f));
	EXPECT_TRUE(near(q.y, 3.f));
	EXPECT_TRUE(near(q.z, 4.f));
}

TEST(TextureTransformTest, TranslateAppliesCorrectly) {
	auto t = TextureTransform::Translate(10.f, 20.f, 30.f);
	TxPoint3f p = {1.f, 2.f, 3.f};
	auto q = t(p);
	EXPECT_TRUE(near(q.x, 11.f));
	EXPECT_TRUE(near(q.y, 22.f));
	EXPECT_TRUE(near(q.z, 33.f));
}

TEST(TextureTransformTest, TranslateDoesNotAffectVector) {
	// Translation should not move a direction vector (w=0)
	auto t = TextureTransform::Translate(10.f, 20.f, 30.f);
	TxVector3f v = {1.f, 0.f, 0.f};
	auto w = t(v);
	EXPECT_TRUE(near(w.x, 1.f));
	EXPECT_TRUE(near(w.y, 0.f));
	EXPECT_TRUE(near(w.z, 0.f));
}

TEST(TextureTransformTest, IsIdentityTrueForDefault) {
	TextureTransform t;
	EXPECT_TRUE(t.IsIdentity());
}

TEST(TextureTransformTest, IsIdentityFalseAfterScale) {
	auto t = TextureTransform::Scale(2.f, 2.f, 2.f);
	EXPECT_FALSE(t.IsIdentity());
}

// ---------------------------------------------------------------------------
// UVMapping tests
// ---------------------------------------------------------------------------

TEST(UVMappingTest, IdentityMappingReturnsUV) {
	UVMapping m;
	auto ctx = MakeCtx({}, {0.3f, 0.7f}, 0.1f, 0.f, 0.f, 0.1f);
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.3f));
	EXPECT_TRUE(near(tc.st.y, 0.7f));
}

TEST(UVMappingTest, ScaleAndOffsetApplied) {
	UVMapping m(2.f, 3.f, 0.1f, 0.2f);  // su=2, sv=3, du=0.1, dv=0.2
	auto ctx = MakeCtx({}, {0.5f, 0.5f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 2.f * 0.5f + 0.1f));
	EXPECT_TRUE(near(tc.st.y, 3.f * 0.5f + 0.2f));
}

TEST(UVMappingTest, DifferentialsScaledBySuSv) {
	// dsdx = su * dudx,  dtdy = sv * dvdy
	UVMapping m(2.f, 3.f);
	auto ctx = MakeCtx({}, {0,0}, 0.1f, 0.f, 0.f, 0.2f);
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.dsdx, 2.f * 0.1f));
	EXPECT_TRUE(near(tc.dtdy, 3.f * 0.2f));
}

TEST(UVMappingTest, MixedDifferentials) {
	UVMapping m(1.f, 1.f);
	// dudx=0.1, dudy=0.05, dvdx=0.03, dvdy=0.08
	TextureEvalContext ctx({0,0,0},{0,0,0},{0,0,0},{0,0,1},{0.f,0.f},
							0.1f, 0.05f, 0.03f, 0.08f);
	auto tc = m.Map(ctx);
	EXPECT_TRUE(near(tc.dsdx, 0.1f));
	EXPECT_TRUE(near(tc.dsdy, 0.05f));
	EXPECT_TRUE(near(tc.dtdx, 0.03f));
	EXPECT_TRUE(near(tc.dtdy, 0.08f));
}

// ---------------------------------------------------------------------------
// SphericalMapping tests
// ---------------------------------------------------------------------------

TEST(SphericalMappingTest, NorthPoleGivesTheta0) {
	// +z direction: theta = 0, s = theta/pi = 0  (pbrt-v4: s = SphericalTheta*InvPi)
	SphericalMapping m(TextureTransform{});
	auto ctx = MakeCtx({0.f, 0.f, 1.f});  // surface point on +z
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.f, 1e-4f));  // s = theta/pi, theta=0
}

TEST(SphericalMappingTest, SouthPoleGivesThetaPi) {
	// -z direction: theta = pi, s = theta/pi = 1
	SphericalMapping m(TextureTransform{});
	auto ctx = MakeCtx({0.f, 0.f, -1.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 1.f, 1e-4f));  // s = theta/pi = 1
}

TEST(SphericalMappingTest, PositiveXAxisPhi0) {
	// +x: phi = 0, t = phi/(2pi) = 0  (pbrt-v4: t = SphericalPhi*Inv2Pi)
	SphericalMapping m(TextureTransform{});
	auto ctx = MakeCtx({1.f, 0.f, 0.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.y, 0.f, 1e-4f));
}

TEST(SphericalMappingTest, PositiveYAxisPhiHalf) {
	// +y: phi = pi/2, t = phi/(2pi) = 0.25
	SphericalMapping m(TextureTransform{});
	auto ctx = MakeCtx({0.f, 1.f, 0.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.y, 0.25f, 1e-4f));
}

TEST(SphericalMappingTest, STInUnitRange) {
	// For any unit-length point, s and t should be in [0,1]
	SphericalMapping m(TextureTransform{});
	float inv_sqrt3 = 1.f / std::sqrt(3.f);
	auto ctx = MakeCtx({inv_sqrt3, inv_sqrt3, inv_sqrt3});
	auto tc  = m.Map(ctx);
	EXPECT_GE(tc.st.x, -kEps);
	EXPECT_LE(tc.st.x,  1.f + kEps);
	EXPECT_GE(tc.st.y, -kEps);
	EXPECT_LE(tc.st.y,  1.f + kEps);
}

// ---------------------------------------------------------------------------
// CylindricalMapping tests
// ---------------------------------------------------------------------------

TEST(CylindricalMappingTest, PositiveXAxisSIsHalf) {
	// +x: atan2(0,1)=0, s = (pi+0)/(2pi) = 0.5
	CylindricalMapping m(TextureTransform{});
	auto ctx = MakeCtx({1.f, 0.f, 2.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.5f, 1e-4f));
}

TEST(CylindricalMappingTest, ZComponentIsT) {
	CylindricalMapping m(TextureTransform{});
	auto ctx = MakeCtx({1.f, 0.f, 3.7f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.y, 3.7f, 1e-4f));
}

TEST(CylindricalMappingTest, NegativeYAxisSIsThreeQuarters) {
	// -y: atan2(-1,0) = -pi/2, phi = -pi/2 + 2pi = 3pi/2
	// s = (pi + atan2(-1,0)) / (2pi) = (pi - pi/2)/(2pi) = 0.25
	CylindricalMapping m(TextureTransform{});
	auto ctx = MakeCtx({0.f, -1.f, 0.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.25f, 1e-4f));
}

// ---------------------------------------------------------------------------
// PlanarMapping tests
// ---------------------------------------------------------------------------

TEST(PlanarMappingTest, XaxisVsYaxisVt) {
	// vs = (1,0,0), vt = (0,1,0), ds=dt=0
	TxVector3f vs{1,0,0}, vt{0,1,0};
	PlanarMapping m(TextureTransform{}, vs, vt);
	auto ctx = MakeCtx({3.f, 4.f, 5.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 3.f));
	EXPECT_TRUE(near(tc.st.y, 4.f));
}

TEST(PlanarMappingTest, OffsetAddedToST) {
	TxVector3f vs{1,0,0}, vt{0,1,0};
	PlanarMapping m(TextureTransform{}, vs, vt, 0.5f, -0.5f);
	auto ctx = MakeCtx({1.f, 2.f, 0.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 1.5f));
	EXPECT_TRUE(near(tc.st.y, 1.5f));
}

TEST(PlanarMappingTest, DifferentialsViaVsVt) {
	TxVector3f vs{2,0,0}, vt{0,3,0};
	// dpdx = (0.1, 0, 0), dpdy = (0, 0.1, 0)
	PlanarMapping m(TextureTransform{}, vs, vt);
	auto ctx = MakeCtx({0,0,0}, {0,0},
						0.f, 0.f, 0.f, 0.f,
						{0.1f,0,0}, {0,0.1f,0});
	auto tc = m.Map(ctx);
	// dsdx = dot(vs, dpdx) = 2*0.1 = 0.2
	// dtdy = dot(vt, dpdy) = 3*0.1 = 0.3
	EXPECT_TRUE(near(tc.dsdx, 0.2f));
	EXPECT_TRUE(near(tc.dtdy, 0.3f));
}

// ---------------------------------------------------------------------------
// TextureMapping2D variant dispatch
// ---------------------------------------------------------------------------

TEST(TextureMapping2DTest, DispatchesToUVMapping) {
	TextureMapping2D m{UVMapping(2.f, 2.f)};
	auto ctx = MakeCtx({}, {0.5f, 0.5f}, 0.1f, 0.f, 0.f, 0.1f);
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 1.f));
	EXPECT_TRUE(near(tc.st.y, 1.f));
}

TEST(TextureMapping2DTest, DispatchesToSphericalMapping) {
	TextureMapping2D m{SphericalMapping(TextureTransform{})};
	auto ctx = MakeCtx({0.f, 0.f, 1.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.f, 1e-4f));  // north pole: s = theta/pi = 0
}

TEST(TextureMapping2DTest, DispatchesToCylindricalMapping) {
	TextureMapping2D m{CylindricalMapping(TextureTransform{})};
	auto ctx = MakeCtx({1.f, 0.f, 5.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 0.5f, 1e-4f));
	EXPECT_TRUE(near(tc.st.y, 5.f, 1e-4f));
}

TEST(TextureMapping2DTest, DispatchesToPlanarMapping) {
	TextureMapping2D m{PlanarMapping(TextureTransform{}, {1,0,0}, {0,1,0})};
	auto ctx = MakeCtx({7.f, 8.f, 9.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.st.x, 7.f));
	EXPECT_TRUE(near(tc.st.y, 8.f));
}

// ---------------------------------------------------------------------------
// PointTransformMapping / TextureMapping3D tests
// ---------------------------------------------------------------------------

TEST(PointTransformMappingTest, IdentityPassthrough) {
	PointTransformMapping m(TextureTransform{});
	auto ctx = MakeCtx({1.f, 2.f, 3.f}, {0,0},
						0.f,0.f,0.f,0.f,
						{0.1f,0,0}, {0,0.1f,0});
	auto tc = m.Map(ctx);
	EXPECT_TRUE(near(tc.p.x,    1.f));
	EXPECT_TRUE(near(tc.p.y,    2.f));
	EXPECT_TRUE(near(tc.p.z,    3.f));
	EXPECT_TRUE(near(tc.dpdx.x, 0.1f));
	EXPECT_TRUE(near(tc.dpdy.y, 0.1f));
}

TEST(PointTransformMappingTest, ScaleTransform) {
	PointTransformMapping m(TextureTransform::Scale(2.f, 2.f, 2.f));
	auto ctx = MakeCtx({1.f, 1.f, 1.f}, {0,0},
						0.f,0.f,0.f,0.f,
						{1.f,0,0},{0,0,0});
	auto tc = m.Map(ctx);
	EXPECT_TRUE(near(tc.p.x, 2.f));
	EXPECT_TRUE(near(tc.p.y, 2.f));
	EXPECT_TRUE(near(tc.p.z, 2.f));
	EXPECT_TRUE(near(tc.dpdx.x, 2.f));
}

TEST(TextureMapping3DTest, VariantDispatch) {
	TextureMapping3D m{PointTransformMapping(TextureTransform{})};
	auto ctx = MakeCtx({4.f,5.f,6.f});
	auto tc  = m.Map(ctx);
	EXPECT_TRUE(near(tc.p.x, 4.f));
	EXPECT_TRUE(near(tc.p.y, 5.f));
	EXPECT_TRUE(near(tc.p.z, 6.f));
}
