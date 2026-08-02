// textures_tests.cpp -- unit tests for src/shared/textures.h
// Validates all texture types against pbrt-v4 behavior.

#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <vector>

// Include mipmap.h before textures.h so the MIPMAP_H guard in textures.h
// activates the FloatImageTexture / RGBImageTexture template classes.
#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/shared/mipmap.h"

#include "../../src/shared/textures.h"

// ===========================================================================
// Helpers
// ===========================================================================

// Build a minimal TextureEvalContext at a given uv
static TextureEvalContext make_ctx(float u, float v,
								   float px=0.f, float py=0.f, float pz=0.f) {
	TextureEvalContext ctx;
	ctx.p   = {px, py, pz};
	ctx.uv  = {u, v};
	ctx.n   = {0.f, 0.f, 1.f};
	ctx.dpdx = {0.01f, 0.f,  0.f};
	ctx.dpdy = {0.f,   0.f,  0.01f};
	ctx.dudx = 0.01f; ctx.dudy = 0.f;
	ctx.dvdx = 0.f;   ctx.dvdy = 0.01f;
	return ctx;
}

// ===========================================================================
// FloatConstantTexture
// ===========================================================================
TEST(FloatConstantTexture, ReturnsValue) {
	FloatConstantTexture<float> tex{0.75f};
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.f, 0.f)), 0.75f);
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.75f);
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(1.f, 1.f)), 0.75f);
}

TEST(FloatConstantTexture, ZeroDefault) {
	FloatConstantTexture<float> tex{};
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.3f, 0.7f)), 0.f);
}

TEST(FloatConstantTexture, DoublePrecision) {
	FloatConstantTexture<double> tex{0.123456789};
	EXPECT_DOUBLE_EQ(tex.Evaluate(make_ctx(0.f, 0.f)), 0.123456789);
}

// ===========================================================================
// RGBConstantTexture
// ===========================================================================
TEST(RGBConstantTexture, ReturnsRGB) {
	RGBConstantTexture<float> tex{{0.2f, 0.5f, 0.8f}};
	auto rgb = tex.Evaluate(make_ctx(0.3f, 0.7f));
	EXPECT_FLOAT_EQ(rgb[0], 0.2f);
	EXPECT_FLOAT_EQ(rgb[1], 0.5f);
	EXPECT_FLOAT_EQ(rgb[2], 0.8f);
}

TEST(RGBConstantTexture, IndependentOfUV) {
	RGBConstantTexture<float> tex{{1.f, 0.f, 0.f}};
	auto a = tex.Evaluate(make_ctx(0.f, 0.f));
	auto b = tex.Evaluate(make_ctx(1.f, 1.f));
	EXPECT_FLOAT_EQ(a[0], b[0]);
	EXPECT_FLOAT_EQ(a[1], b[1]);
	EXPECT_FLOAT_EQ(a[2], b[2]);
}

// ===========================================================================
// FloatBilerpTexture
// ===========================================================================
TEST(FloatBilerpTexture, CornersMatchValues) {
	// v00=0, v10=1, v01=2, v11=3
	FloatBilerpTexture<float> tex;
	tex.mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	tex.v00 = 0.f; tex.v10 = 1.f; tex.v01 = 2.f; tex.v11 = 3.f;

	// (0,0) -> v00
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 0.f, 1e-5f);
	// (1,0) -> v10
	EXPECT_NEAR(tex.Evaluate(make_ctx(1.f, 0.f)), 1.f, 1e-5f);
	// (0,1) -> v01
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 1.f)), 2.f, 1e-5f);
	// (1,1) -> v11
	EXPECT_NEAR(tex.Evaluate(make_ctx(1.f, 1.f)), 3.f, 1e-5f);
}

TEST(FloatBilerpTexture, CentreIsMean) {
	FloatBilerpTexture<float> tex;
	tex.v00 = 0.f; tex.v10 = 1.f; tex.v01 = 0.f; tex.v11 = 1.f;
	// At (0.5, 0.5): (0+1+0+1)/4 = 0.5
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.5f, 1e-5f);
}

TEST(FloatBilerpTexture, LinearAlongUAxis) {
	// All v-values 0 at v=0, so pure linear interpolation in u
	FloatBilerpTexture<float> tex;
	tex.v00 = 0.f; tex.v10 = 4.f; tex.v01 = 0.f; tex.v11 = 4.f;
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.25f, 0.5f)), 1.f, 1e-5f);
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.75f, 0.5f)), 3.f, 1e-5f);
}

TEST(FloatBilerpTexture, ScaledMappingHalvesUV) {
	// su=0.5 means u=0.5 in texel space -> sample at s=0.25 in texture
	FloatBilerpTexture<float> tex;
	tex.mapping = UVMapping(0.5f, 0.5f, 0.f, 0.f);
	tex.v00 = 0.f; tex.v10 = 1.f; tex.v01 = 0.f; tex.v11 = 1.f;
	// u=1 -> s=0.5 -> bilerp = 0.5
	EXPECT_NEAR(tex.Evaluate(make_ctx(1.f, 0.f)), 0.5f, 1e-5f);
}

// ===========================================================================
// RGBBilerpTexture
// ===========================================================================
TEST(RGBBilerpTexture, CornersMatchValues) {
	RGBBilerpTexture<float> tex;
	tex.v00 = {1.f,0.f,0.f};  // red
	tex.v10 = {0.f,1.f,0.f};  // green
	tex.v01 = {0.f,0.f,1.f};  // blue
	tex.v11 = {1.f,1.f,1.f};  // white

	auto r = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_NEAR(r[0], 1.f, 1e-5f); EXPECT_NEAR(r[1], 0.f, 1e-5f); EXPECT_NEAR(r[2], 0.f, 1e-5f);

	auto g = tex.Evaluate(make_ctx(1.f, 0.f));
	EXPECT_NEAR(g[0], 0.f, 1e-5f); EXPECT_NEAR(g[1], 1.f, 1e-5f); EXPECT_NEAR(g[2], 0.f, 1e-5f);
}

TEST(RGBBilerpTexture, CentreIsAverage) {
	RGBBilerpTexture<float> tex;
	tex.v00 = {0.f,0.f,0.f};
	tex.v10 = {1.f,0.f,0.f};
	tex.v01 = {0.f,1.f,0.f};
	tex.v11 = {0.f,0.f,1.f};
	auto c = tex.Evaluate(make_ctx(0.5f, 0.5f));
	EXPECT_NEAR(c[0], 0.25f, 1e-5f);
	EXPECT_NEAR(c[1], 0.25f, 1e-5f);
	EXPECT_NEAR(c[2], 0.25f, 1e-5f);
}

// ===========================================================================
// FloatScaleTexture
// ===========================================================================
TEST(FloatScaleTexture, MultipliesTwoConstants) {
	FloatConstantTexture<float> a{0.6f};
	FloatConstantTexture<float> b{0.5f};
	FloatScaleTexture<FloatConstantTexture<float>, FloatConstantTexture<float>, float> tex{a, b};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 0.3f, 1e-6f);
}

TEST(FloatScaleTexture, ZeroScaleGivesZero) {
	FloatConstantTexture<float> a{0.9f};
	FloatConstantTexture<float> b{0.f};
	FloatScaleTexture<FloatConstantTexture<float>, FloatConstantTexture<float>, float> tex{a, b};
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.3f, 0.7f)), 0.f);
}

TEST(FloatScaleTexture, ScaleBilerpByConstant) {
	FloatBilerpTexture<float> bilerp;
	bilerp.v00 = 0.f; bilerp.v10 = 2.f; bilerp.v01 = 0.f; bilerp.v11 = 2.f;
	FloatConstantTexture<float> scale{0.5f};
	FloatScaleTexture<FloatBilerpTexture<float>, FloatConstantTexture<float>, float> tex{bilerp, scale};
	// At u=1: bilerp=2, scale=0.5 -> result=1
	EXPECT_NEAR(tex.Evaluate(make_ctx(1.f, 0.f)), 1.f, 1e-5f);
}

// ===========================================================================
// RGBScaleTexture
// ===========================================================================
TEST(RGBScaleTexture, ScalesComponentsIndependently) {
	RGBConstantTexture<float> rgb{{1.f, 0.5f, 0.25f}};
	FloatConstantTexture<float> scale{2.f};
	RGBScaleTexture<RGBConstantTexture<float>, FloatConstantTexture<float>, float> tex{rgb, scale};
	auto result = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_NEAR(result[0], 2.f,  1e-5f);
	EXPECT_NEAR(result[1], 1.f,  1e-5f);
	EXPECT_NEAR(result[2], 0.5f, 1e-5f);
}

TEST(RGBScaleTexture, ScaleByZeroIsBlack) {
	RGBConstantTexture<float> rgb{{1.f, 1.f, 1.f}};
	FloatConstantTexture<float> scale{0.f};
	RGBScaleTexture<RGBConstantTexture<float>, FloatConstantTexture<float>, float> tex{rgb, scale};
	auto result = tex.Evaluate(make_ctx(0.3f, 0.7f));
	EXPECT_FLOAT_EQ(result[0], 0.f);
	EXPECT_FLOAT_EQ(result[1], 0.f);
	EXPECT_FLOAT_EQ(result[2], 0.f);
}

// ===========================================================================
// FloatMixTexture
// ===========================================================================
TEST(FloatMixTexture, WeightZeroGivesTex1) {
	FloatConstantTexture<float> t1{1.f};
	FloatConstantTexture<float> t2{0.f};
	FloatConstantTexture<float> w{0.f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{t1, t2, w};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 1.f, 1e-6f);
}

TEST(FloatMixTexture, WeightOneGivesTex2) {
	FloatConstantTexture<float> t1{0.f};
	FloatConstantTexture<float> t2{1.f};
	FloatConstantTexture<float> w{1.f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{t1, t2, w};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 1.f, 1e-6f);
}

TEST(FloatMixTexture, HalfWeightIsAverage) {
	FloatConstantTexture<float> t1{0.f};
	FloatConstantTexture<float> t2{1.f};
	FloatConstantTexture<float> w{0.5f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{t1, t2, w};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 0.5f, 1e-6f);
}

TEST(FloatMixTexture, BilerpWeightProducesVaryingMix) {
	FloatConstantTexture<float> t1{0.f};
	FloatConstantTexture<float> t2{1.f};
	// Weight bilerp: 0 at u=0, 1 at u=1
	FloatBilerpTexture<float> w;
	w.v00 = 0.f; w.v10 = 1.f; w.v01 = 0.f; w.v11 = 1.f;
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatBilerpTexture<float>, float> tex{t1, t2, w};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 0.f,  1e-5f);
	EXPECT_NEAR(tex.Evaluate(make_ctx(1.f, 0.f)), 1.f,  1e-5f);
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.f)), 0.5f, 1e-5f);
}

// ===========================================================================
// RGBMixTexture
// ===========================================================================
TEST(RGBMixTexture, WeightZeroGivesTex1) {
	RGBConstantTexture<float> t1{{1.f, 0.f, 0.f}};
	RGBConstantTexture<float> t2{{0.f, 1.f, 0.f}};
	FloatConstantTexture<float> w{0.f};
	RGBMixTexture<RGBConstantTexture<float>,
				  RGBConstantTexture<float>,
				  FloatConstantTexture<float>, float> tex{t1, t2, w};
	auto c = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_NEAR(c[0], 1.f, 1e-5f);
	EXPECT_NEAR(c[1], 0.f, 1e-5f);
}

TEST(RGBMixTexture, WeightOneGivesTex2) {
	RGBConstantTexture<float> t1{{1.f, 0.f, 0.f}};
	RGBConstantTexture<float> t2{{0.f, 0.f, 1.f}};
	FloatConstantTexture<float> w{1.f};
	RGBMixTexture<RGBConstantTexture<float>,
				  RGBConstantTexture<float>,
				  FloatConstantTexture<float>, float> tex{t1, t2, w};
	auto c = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_NEAR(c[0], 0.f, 1e-5f);
	EXPECT_NEAR(c[2], 1.f, 1e-5f);
}

TEST(RGBMixTexture, HalfBlendIsAverage) {
	RGBConstantTexture<float> t1{{0.f, 0.f, 0.f}};
	RGBConstantTexture<float> t2{{1.f, 1.f, 1.f}};
	FloatConstantTexture<float> w{0.5f};
	RGBMixTexture<RGBConstantTexture<float>,
				  RGBConstantTexture<float>,
				  FloatConstantTexture<float>, float> tex{t1, t2, w};
	auto c = tex.Evaluate(make_ctx(0.3f, 0.7f));
	EXPECT_NEAR(c[0], 0.5f, 1e-5f);
	EXPECT_NEAR(c[1], 0.5f, 1e-5f);
	EXPECT_NEAR(c[2], 0.5f, 1e-5f);
}

// ===========================================================================
// UVTexture
// ===========================================================================
TEST(UVTexture, UVCoordinatesAtCorners) {
	UVTexture<float> tex;
	auto c00 = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_NEAR(c00[0], 0.f, 1e-5f);
	EXPECT_NEAR(c00[1], 0.f, 1e-5f);
	EXPECT_NEAR(c00[2], 0.f, 1e-5f);

	auto c11 = tex.Evaluate(make_ctx(1.f, 1.f));
	// frac(1.0) = 0.0 (wraps)
	EXPECT_NEAR(c11[0], 0.f, 1e-5f);
	EXPECT_NEAR(c11[1], 0.f, 1e-5f);
}

TEST(UVTexture, MidpointUV) {
	UVTexture<float> tex;
	auto c = tex.Evaluate(make_ctx(0.3f, 0.7f));
	EXPECT_NEAR(c[0], 0.3f, 1e-5f);
	EXPECT_NEAR(c[1], 0.7f, 1e-5f);
	EXPECT_NEAR(c[2], 0.f,  1e-5f);
}

TEST(UVTexture, ScaledMapping) {
	// su=2 -> s = 2*u, so at u=0.3 -> s=0.6
	UVTexture<float> tex;
	tex.mapping = UVMapping(2.f, 1.f, 0.f, 0.f);
	auto c = tex.Evaluate(make_ctx(0.3f, 0.5f));
	EXPECT_NEAR(c[0], 0.6f, 1e-5f);
	EXPECT_NEAR(c[1], 0.5f, 1e-5f);
}

TEST(UVTexture, WrapsBeyondOne) {
	UVTexture<float> tex;
	// u=1.4 -> frac=0.4
	auto c = tex.Evaluate(make_ctx(1.4f, 0.2f));
	EXPECT_NEAR(c[0], 0.4f, 1e-5f);
	EXPECT_NEAR(c[1], 0.2f, 1e-5f);
}

// ===========================================================================
// CheckerboardTexture
// ===========================================================================
// ===========================================================================
// CheckerboardTexture -- antialiased (pbrt-v4 band-limited filter)
// ===========================================================================

// Helper: make a context with tiny differentials (nearly point-sampled)
static TextureEvalContext make_ctx_point(float u, float v,
										  float px=0.f, float py=0.f, float pz=0.f) {
	TextureEvalContext ctx = make_ctx(u, v, px, py, pz);
	// Very small footprint so the antialiased result closely matches binary
	ctx.dudx = 1e-6f; ctx.dudy = 0.f;
	ctx.dvdx = 0.f;   ctx.dvdy = 1e-6f;
	ctx.dpdx = {1e-6f, 0.f, 0.f};
	ctx.dpdy = {0.f, 0.f, 1e-6f};
	return ctx;
}

TEST(CheckerboardTexture, CellCentresMatchExpectedTexture) {
	FloatConstantTexture<float> light{1.f};
	FloatConstantTexture<float> dark{0.f};
	CheckerboardTexture<FloatConstantTexture<float>,
						FloatConstantTexture<float>, float> tex;
	tex.tex1 = light; tex.tex2 = dark;

	// Cell (0,0): even sum -> tex1 (light=1)
	float v00 = tex.Evaluate(make_ctx_point(0.25f, 0.25f));
	EXPECT_GT(v00, 0.9f);

	// Cell (1,0): odd sum -> tex2 (dark=0)
	float v10 = tex.Evaluate(make_ctx_point(1.25f, 0.25f));
	EXPECT_LT(v10, 0.1f);

	// Cell (1,1): even sum -> tex1 (light=1)
	float v11 = tex.Evaluate(make_ctx_point(1.25f, 1.25f));
	EXPECT_GT(v11, 0.9f);
}

TEST(CheckerboardTexture, NegativeCoordinatesCellCentre) {
	FloatConstantTexture<float> light{1.f};
	FloatConstantTexture<float> dark{0.f};
	CheckerboardTexture<FloatConstantTexture<float>,
						FloatConstantTexture<float>, float> tex;
	tex.tex1 = light; tex.tex2 = dark;
	// (-0.75, 0.25) -> cell (-1, 0) -> floor(-0.75)=-1, sum=-1 (odd) -> dark=0
	float v = tex.Evaluate(make_ctx_point(-0.75f, 0.25f));
	EXPECT_LT(v, 0.1f);
}

TEST(CheckerboardTexture, AntialiasedBoundaryBlendsColors) {
	FloatConstantTexture<float> light{1.f};
	FloatConstantTexture<float> dark{0.f};
	CheckerboardTexture<FloatConstantTexture<float>,
						FloatConstantTexture<float>, float> tex;
	tex.tex1 = light; tex.tex2 = dark;

	// At a cell boundary with a large footprint, the result should be
	// strictly between 0 and 1 (antialiased blend)
	TextureEvalContext ctx = make_ctx(1.0f, 0.5f);   // at boundary s=1
	ctx.dudx = 0.5f; ctx.dudy = 0.f;   // large footprint straddles boundary
	ctx.dvdx = 0.f;  ctx.dvdy = 0.5f;
	ctx.dpdx = {0.5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 0.f, 0.5f};
	float v = tex.Evaluate(ctx);
	EXPECT_GT(v, 0.01f);   // not pure dark
	EXPECT_LT(v, 0.99f);   // not pure light
}

// ===========================================================================
// RGBCheckerboardTexture
// ===========================================================================
TEST(RGBCheckerboardTexture, CellCentresMatchExpectedColor) {
	RGBConstantTexture<float> red  {{1.f, 0.f, 0.f}};
	RGBConstantTexture<float> blue {{0.f, 0.f, 1.f}};
	RGBCheckerboardTexture<RGBConstantTexture<float>,
						   RGBConstantTexture<float>, float> tex;
	tex.tex1 = red; tex.tex2 = blue;

	// cell (0,0) -> red
	auto r = tex.Evaluate(make_ctx_point(0.25f, 0.25f));
	EXPECT_GT(r[0], 0.9f); EXPECT_LT(r[2], 0.1f);

	// cell (1,0) -> blue
	auto b = tex.Evaluate(make_ctx_point(1.25f, 0.25f));
	EXPECT_LT(b[0], 0.1f); EXPECT_GT(b[2], 0.9f);
}

// ===========================================================================
// Composition tests (mirrors pbrt-v4 use patterns)
// ===========================================================================

// Scale a checkerboard by a constant (simulates "albedo texture * exposure")
TEST(TextureComposition, ScaleCheckerboardByHalf) {
	FloatConstantTexture<float> light{1.f};
	FloatConstantTexture<float> dark{0.f};
	CheckerboardTexture<FloatConstantTexture<float>,
						FloatConstantTexture<float>, float> checker;
	checker.tex1 = light; checker.tex2 = dark;
	FloatConstantTexture<float> half{0.5f};
	FloatScaleTexture<CheckerboardTexture<FloatConstantTexture<float>,
										   FloatConstantTexture<float>, float>,
					  FloatConstantTexture<float>, float> scaled{checker, half};

	// Cell centre: checker ~ 1, scale = 0.5 -> result ~ 0.5
	EXPECT_NEAR(scaled.Evaluate(make_ctx_point(0.25f, 0.25f)), 0.5f, 0.1f);
	// Dark cell centre: checker ~ 0, scale = 0.5 -> result ~ 0
	EXPECT_NEAR(scaled.Evaluate(make_ctx_point(1.25f, 0.25f)), 0.f, 0.1f);
}

// Mix between a constant and a bilerp (simulates a transition material)
TEST(TextureComposition, MixConstantAndBilerp) {
	FloatConstantTexture<float> base{0.f};
	FloatBilerpTexture<float> top;
	top.v00 = 0.f; top.v10 = 1.f; top.v01 = 0.f; top.v11 = 1.f;
	FloatConstantTexture<float> half{0.5f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatBilerpTexture<float>,
					FloatConstantTexture<float>, float> mix{base, top, half};

	// At u=0: top=0, base=0, mix=0
	EXPECT_NEAR(mix.Evaluate(make_ctx(0.f, 0.f)), 0.f, 1e-5f);
	// At u=1: top=1, base=0, mix=0.5
	EXPECT_NEAR(mix.Evaluate(make_ctx(1.f, 0.f)), 0.5f, 1e-5f);
}

// pbrt-v4 alignment: ScaleTexture evaluates scale first and short-circuits at 0
TEST(FloatScaleTexture, EarlyExitOnZeroScale) {
	// Use a bilerp as base -- at u=0.5 it returns 0.5;
	// but if scale=0, base should never be evaluated (and result = 0)
	FloatBilerpTexture<float> base;
	base.v00 = 1.f; base.v10 = 1.f; base.v01 = 1.f; base.v11 = 1.f;
	FloatConstantTexture<float> zero{0.f};
	FloatScaleTexture<FloatBilerpTexture<float>,
					  FloatConstantTexture<float>, float> tex{base, zero};
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.f);
}

TEST(RGBScaleTexture, EarlyExitOnZeroScale) {
	RGBConstantTexture<float> base{{1.f, 1.f, 1.f}};
	FloatConstantTexture<float> zero{0.f};
	RGBScaleTexture<RGBConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{base, zero};
	auto r = tex.Evaluate(make_ctx(0.5f, 0.5f));
	EXPECT_FLOAT_EQ(r[0], 0.f); EXPECT_FLOAT_EQ(r[1], 0.f); EXPECT_FLOAT_EQ(r[2], 0.f);
}

// pbrt-v4 alignment: MixTexture short-circuits tex1 when amt==1, tex2 when amt==0
TEST(FloatMixTexture, ShortCircuitAtZeroWeight) {
	FloatConstantTexture<float> t1{1.f};
	FloatConstantTexture<float> t2{2.f};
	FloatConstantTexture<float> w{0.f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{t1, t2, w};
	// amt=0 -> only tex1 evaluated -> result=1
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.f, 0.f)), 1.f);
}

TEST(FloatMixTexture, ShortCircuitAtOneWeight) {
	FloatConstantTexture<float> t1{1.f};
	FloatConstantTexture<float> t2{2.f};
	FloatConstantTexture<float> w{1.f};
	FloatMixTexture<FloatConstantTexture<float>,
					FloatConstantTexture<float>,
					FloatConstantTexture<float>, float> tex{t1, t2, w};
	// amt=1 -> only tex2 evaluated -> result=2
	EXPECT_FLOAT_EQ(tex.Evaluate(make_ctx(0.f, 0.f)), 2.f);
}

TEST(RGBMixTexture, ShortCircuitAtZeroWeight) {
	RGBConstantTexture<float> t1{{1.f, 0.f, 0.f}};
	RGBConstantTexture<float> t2{{0.f, 1.f, 0.f}};
	FloatConstantTexture<float> w{0.f};
	RGBMixTexture<RGBConstantTexture<float>,
				  RGBConstantTexture<float>,
				  FloatConstantTexture<float>, float> tex{t1, t2, w};
	auto c = tex.Evaluate(make_ctx(0.f, 0.f));
	EXPECT_FLOAT_EQ(c[0], 1.f); EXPECT_FLOAT_EQ(c[1], 0.f);
}

// ===========================================================================
// Double precision
// ===========================================================================
TEST(FloatConstantTexture, DoublePrecisionComposedScale) {
	FloatConstantTexture<double> a{0.3};
	FloatConstantTexture<double> b{0.7};
	FloatScaleTexture<FloatConstantTexture<double>,
					  FloatConstantTexture<double>, double> tex{a, b};
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.f, 0.f)), 0.21, 1e-12);
}

TEST(UVTexture, DoublePrecisionUV) {
	UVTexture<double> tex;
	auto c = tex.Evaluate(make_ctx(0.4f, 0.6f));
	EXPECT_NEAR(c[0], 0.4, 1e-5);
	EXPECT_NEAR(c[1], 0.6, 1e-5);
}

// ===========================================================================
// FloatImageTexture / RGBImageTexture  (pbrt-v4 ImageTextureBase)
// ===========================================================================

// Build a 2x2 synthetic mipmap: four distinct colors at each corner
//  (0,0)=red  (1,0)=green  (0,1)=blue  (1,1)=white
static std::vector<color> make_2x2_pixels() {
	return {
		color(1.0, 0.0, 0.0),  // (0,0) red
		color(0.0, 1.0, 0.0),  // (1,0) green
		color(0.0, 0.0, 1.0),  // (0,1) blue
		color(1.0, 1.0, 1.0),  // (1,1) white
	};
}

TEST(RGBImageTexture, ConstantOneMipReturnsApproximateColor) {
	// 1x1 mipmap (single red pixel) -- any UV should return red
	std::vector<color> pixels = { color(0.8, 0.3, 0.1) };
	MipMapOptions opts; opts.filter = MipFilter::Point;
	mipmap mip(pixels, 1, 1, opts);

	RGBImageTexture<float> tex;
	tex.mip = &mip;
	tex.mapping = UVMapping(1.f, 1.f, 0.f, 0.f);

	auto c = tex.Evaluate(make_ctx(0.5f, 0.5f));
	EXPECT_NEAR(c[0], 0.8f, 0.01f);
	EXPECT_NEAR(c[1], 0.3f, 0.01f);
	EXPECT_NEAR(c[2], 0.1f, 0.01f);
}

TEST(RGBImageTexture, ScaleMultipliesResult) {
	std::vector<color> pixels = { color(1.0, 1.0, 1.0) };
	MipMapOptions opts; opts.filter = MipFilter::Point;
	mipmap mip(pixels, 1, 1, opts);

	RGBImageTexture<float> tex;
	tex.mip = &mip;
	tex.scale = 0.5f;

	auto c = tex.Evaluate(make_ctx(0.5f, 0.5f));
	EXPECT_NEAR(c[0], 0.5f, 0.01f);
	EXPECT_NEAR(c[1], 0.5f, 0.01f);
	EXPECT_NEAR(c[2], 0.5f, 0.01f);
}

TEST(RGBImageTexture, InvertFlipsValues) {
	std::vector<color> pixels = { color(0.0, 0.0, 0.0) };
	MipMapOptions opts; opts.filter = MipFilter::Point;
	mipmap mip(pixels, 1, 1, opts);

	RGBImageTexture<float> tex;
	tex.mip = &mip;
	tex.invert = true;

	auto c = tex.Evaluate(make_ctx(0.5f, 0.5f));
	// 0 inverted -> max(0, 1-0) = 1
	EXPECT_NEAR(c[0], 1.f, 0.01f);
	EXPECT_NEAR(c[1], 1.f, 0.01f);
	EXPECT_NEAR(c[2], 1.f, 0.01f);
}

TEST(RGBImageTexture, NullMipReturnsZero) {
	RGBImageTexture<float> tex;
	tex.mip = nullptr;
	auto c = tex.Evaluate(make_ctx(0.5f, 0.5f));
	EXPECT_FLOAT_EQ(c[0], 0.f);
	EXPECT_FLOAT_EQ(c[1], 0.f);
	EXPECT_FLOAT_EQ(c[2], 0.f);
}

TEST(FloatImageTexture, SingleChannelLookup) {
	// 1x1 mipmap with (0.7, 0.4, 0.2)
	std::vector<color> pixels = { color(0.7, 0.4, 0.2) };
	MipMapOptions opts; opts.filter = MipFilter::Point;
	mipmap mip(pixels, 1, 1, opts);

	FloatImageTexture<float> tex;
	tex.mip = &mip;
	tex.channel = 0;  // R channel
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.7f, 0.01f);

	tex.channel = 1;  // G channel
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.4f, 0.01f);

	tex.channel = 2;  // B channel
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.2f, 0.01f);
}

TEST(FloatImageTexture, InvertAndScale) {
	std::vector<color> pixels = { color(0.8, 0.8, 0.8) };
	MipMapOptions opts; opts.filter = MipFilter::Point;
	mipmap mip(pixels, 1, 1, opts);

	FloatImageTexture<float> tex;
	tex.mip = &mip;
	tex.scale = 1.f; tex.invert = true; tex.channel = 0;
	// 0.8 inverted -> max(0, 1-0.8) = 0.2
	EXPECT_NEAR(tex.Evaluate(make_ctx(0.5f, 0.5f)), 0.2f, 0.01f);
}
