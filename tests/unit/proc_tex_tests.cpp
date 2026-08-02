// proc_tex_tests.cpp
// Unit tests for src/shared/procedural_textures.h
//
// Tests mirror pbrt-v4 behaviour and verify:
//   - ConstantTexture: returns value regardless of context
//   - BilerpTexture: corners, center, edge midpoints
//   - CheckerboardTexture: 2D and 3D cell alternation at cell centers
//   - MixTexture: amount=0/0.5/1
//   - ScaledTexture: child * scale
//   - FBmTexture: result in a plausible range, deterministic
//   - WrinkledTexture: non-negative (turbulence is ABS-based)
//   - WindyTexture: value varies with position
//   - DotsTexture: returns inside_val or outside_val depending on position
//   - MarbleTexture: luminance in [0, 1.5] range, deterministic
//   - MixFnTexture, ScaledFnTexture, DirectionMixFnTexture: callable wrappers
//   - checkerboard_weight helpers: filtered anti-aliased behavior
//   - inside_polka_dot: deterministic boolean

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include "../../src/shared/procedural_textures.h"

// ===========================================================================
// Helpers
// ===========================================================================

static TextureEvalContext MakeCtx(float px, float py, float pz,
								   float u = 0.f, float v = 0.f)
{
	TextureEvalContext ctx;
	ctx.p    = {px, py, pz};
	ctx.dpdx = {0.001f, 0.f, 0.f};
	ctx.dpdy = {0.f, 0.001f, 0.f};
	ctx.n    = {0.f, 0.f, 1.f};
	ctx.uv   = {u, v};
	ctx.dudx = 0.001f; ctx.dudy = 0.f;
	ctx.dvdx = 0.f;    ctx.dvdy = 0.001f;
	return ctx;
}

static TextureMapping2D MakeUV2D() {
	return UVMapping(1.f, 1.f, 0.f, 0.f);
}

static TextureMapping3D MakeIdentity3D() {
	return PointTransformMapping(TextureTransform{});
}

// ===========================================================================
// ConstantTexture
// ===========================================================================

TEST(ProcTex, ConstantTexture_ReturnsValue) {
	ConstantTexture<float> tex;
	tex.value = 0.75f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(1,2,3)), 0.75f);
}

TEST(ProcTex, ConstantTexture_Zero) {
	ConstantTexture<float> tex;
	tex.value = 0.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 0.f);
}

TEST(ProcTex, ConstantTexture_Double) {
	ConstantTexture<double> tex;
	tex.value = 3.14;
	EXPECT_DOUBLE_EQ(tex.Evaluate(MakeCtx(1,1,1)), 3.14);
}

TEST(ProcTex, ConstantTexture_Negative) {
	ConstantTexture<float> tex;
	tex.value = -1.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), -1.f);
}

// ===========================================================================
// BilerpTexture
// ===========================================================================

TEST(ProcTex, BilerpTexture_Corner00) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 1.f; tex.v01 = 0.f; tex.v10 = 0.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 0.f, 0.f);
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProcTex, BilerpTexture_Corner10) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 0.f; tex.v10 = 1.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 1.f, 0.f);
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProcTex, BilerpTexture_Corner01) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 1.f; tex.v10 = 0.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 0.f, 1.f);
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProcTex, BilerpTexture_Center) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 0.f; tex.v10 = 0.f; tex.v11 = 4.f;
	auto ctx = MakeCtx(0, 0, 0, 0.5f, 0.5f);
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProcTex, BilerpTexture_Uniform) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = tex.v01 = tex.v10 = tex.v11 = 0.5f;
	for (float u : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
		for (float v : {0.f, 0.5f, 1.f}) {
			EXPECT_NEAR(tex.Evaluate(MakeCtx(0, 0, 0, u, v)), 0.5f, 1e-5f);
		}
	}
}

// ===========================================================================
// CheckerboardTexture (2D)
// ===========================================================================

// Tiny filter footprint: use dudx/dvdy for UVMapping differentials
static TextureEvalContext MakeCheckerCtx(float u, float v, float filter = 1e-5f) {
	TextureEvalContext ctx = MakeCtx(0, 0, 0, u, v);
	ctx.dudx = filter; ctx.dudy = 0.f;
	ctx.dvdx = 0.f;    ctx.dvdy = filter;
	return ctx;
}

TEST(ProcTex, Checkerboard2D_EvenCellCenter) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	// cell (0,0) center: u=0.5, v=0.5 -> parity even -> tex0
	EXPECT_NEAR(tex.Evaluate(MakeCheckerCtx(0.5f, 0.5f)), 0.f, 0.05f);
}

TEST(ProcTex, Checkerboard2D_OddCellU1) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	// cell (1,0): u=1.5, v=0.5 -> parity odd -> tex1
	EXPECT_NEAR(tex.Evaluate(MakeCheckerCtx(1.5f, 0.5f)), 1.f, 0.05f);
}

TEST(ProcTex, Checkerboard2D_OddCellV1) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	// cell (0,1): u=0.5, v=1.5 -> parity odd -> tex1
	EXPECT_NEAR(tex.Evaluate(MakeCheckerCtx(0.5f, 1.5f)), 1.f, 0.05f);
}

TEST(ProcTex, Checkerboard2D_EvenCell11) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	// cell (1,1): u=1.5, v=1.5 -> parity even -> tex0
	EXPECT_NEAR(tex.Evaluate(MakeCheckerCtx(1.5f, 1.5f)), 0.f, 0.05f);
}

TEST(ProcTex, Checkerboard2D_BlurredBoundary) {
	// Wide filter in both u and v should produce weight near 0.5
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	// Large dudx and dvdy -> large dsdx/dtdy -> checker blurs toward 0.5
	float w = tex.Evaluate(MakeCheckerCtx(0.5f, 0.5f, 5.f));
	EXPECT_NEAR(w, 0.5f, 0.1f);
}

// ===========================================================================
// Checkerboard3DTexture
// ===========================================================================

TEST(ProcTex, Checkerboard3D_EvenCell) {
	Checkerboard3DTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	auto ctx = MakeCtx(0.5f, 0.5f, 0.5f);
	ctx.dpdx = {1e-5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 1e-5f, 0.f};
	EXPECT_NEAR(tex.Evaluate(ctx), 0.f, 0.05f);
}

TEST(ProcTex, Checkerboard3D_OddCell) {
	Checkerboard3DTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.tex0 = 0.f; tex.tex1 = 1.f;
	auto ctx = MakeCtx(1.5f, 0.5f, 0.5f);
	ctx.dpdx = {1e-5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 1e-5f, 0.f};
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 0.05f);
}

// ===========================================================================
// MixTexture
// ===========================================================================

TEST(ProcTex, MixTexture_Amount0) {
	MixTexture<float> tex;
	tex.tex0 = 3.f; tex.tex1 = 7.f; tex.amount = 0.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 3.f);
}

TEST(ProcTex, MixTexture_Amount1) {
	MixTexture<float> tex;
	tex.tex0 = 3.f; tex.tex1 = 7.f; tex.amount = 1.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 7.f);
}

TEST(ProcTex, MixTexture_HalfAmount) {
	MixTexture<float> tex;
	tex.tex0 = 0.f; tex.tex1 = 4.f; tex.amount = 0.5f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 2.f);
}

TEST(ProcTex, MixTexture_QuarterAmount) {
	MixTexture<float> tex;
	tex.tex0 = 0.f; tex.tex1 = 8.f; tex.amount = 0.25f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 2.f);
}

// ===========================================================================
// ScaledTexture
// ===========================================================================

TEST(ProcTex, ScaledTexture_BasicMultiply) {
	ScaledTexture<float> tex;
	tex.child_value = 3.f; tex.scale = 2.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 6.f);
}

TEST(ProcTex, ScaledTexture_ZeroScale) {
	ScaledTexture<float> tex;
	tex.child_value = 100.f; tex.scale = 0.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 0.f);
}

TEST(ProcTex, ScaledTexture_NegativeScale) {
	ScaledTexture<float> tex;
	tex.child_value = 5.f; tex.scale = -1.f;
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), -5.f);
}

// ===========================================================================
// FBmTexture
// ===========================================================================

TEST(ProcTex, FBmTexture_Deterministic) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f; tex.octaves = 6;
	auto ctx = MakeCtx(1.23f, 4.56f, 7.89f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProcTex, FBmTexture_PlausibleRange) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f; tex.octaves = 8;
	for (float x : {0.1f, 1.f, 3.7f, -2.f}) {
		for (float y : {0.f, 1.1f, -0.5f}) {
			float v = tex.Evaluate(MakeCtx(x, y, 0.f));
			EXPECT_GE(v, -4.f);
			EXPECT_LE(v,  4.f);
		}
	}
}

TEST(ProcTex, FBmTexture_VariesWithPosition) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	EXPECT_NE(tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f)),
			  tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f)));
}

// ===========================================================================
// WrinkledTexture
// ===========================================================================

TEST(ProcTex, WrinkledTexture_NonNegative) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f; tex.octaves = 6;
	for (float x : {0.f, 0.5f, 1.f, 2.3f, -1.f}) {
		for (float y : {0.f, 0.7f, -0.3f}) {
			EXPECT_GE(tex.Evaluate(MakeCtx(x, y, 0.f)), 0.f);
		}
	}
}

TEST(ProcTex, WrinkledTexture_Deterministic) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProcTex, WrinkledTexture_VariesWithPosition) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	EXPECT_NE(tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f)),
			  tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f)));
}

// ===========================================================================
// WindyTexture
// ===========================================================================

TEST(ProcTex, WindyTexture_Deterministic) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProcTex, WindyTexture_PlausibleRange) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	for (float x : {0.f, 1.f, 3.f}) {
		float v = tex.Evaluate(MakeCtx(x, x*0.5f, x*0.3f));
		EXPECT_GE(v, -4.f);
		EXPECT_LE(v,  4.f);
	}
}

TEST(ProcTex, WindyTexture_VariesWithPosition) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	EXPECT_NE(tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f)),
			  tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f)));
}

// ===========================================================================
// DotsTexture
// ===========================================================================

TEST(ProcTex, DotsTexture_ReturnsOneOfTwoValues) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.outside_val = 0.f; tex.inside_val = 1.f;
	for (float u = 0.f; u < 5.f; u += 0.3f) {
		for (float v = 0.f; v < 5.f; v += 0.3f) {
			float r = tex.Evaluate(MakeCtx(0,0,0, u, v));
			EXPECT_TRUE(r == 0.f || r == 1.f);
		}
	}
}

TEST(ProcTex, DotsTexture_CustomValues) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.outside_val = 0.3f; tex.inside_val = 0.9f;
	for (float u = 0.f; u < 3.f; u += 0.25f) {
		for (float v = 0.f; v < 3.f; v += 0.25f) {
			float r = tex.Evaluate(MakeCtx(0,0,0, u, v));
			EXPECT_TRUE(r == 0.3f || r == 0.9f);
		}
	}
}

TEST(ProcTex, DotsTexture_Deterministic) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	auto ctx = MakeCtx(0, 0, 0, 2.7f, 1.3f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

// ===========================================================================
// MarbleTexture
// ===========================================================================

TEST(ProcTex, MarbleTexture_Deterministic) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProcTex, MarbleTexture_PlausibleRange) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	for (float x : {0.f, 0.5f, 1.f, 3.3f, -1.f}) {
		for (float y : {0.f, 1.f, -0.5f}) {
			float v = tex.Evaluate(MakeCtx(x, y, 0.3f));
			EXPECT_GE(v, -0.01f);
			EXPECT_LE(v,  1.5f);
		}
	}
}

TEST(ProcTex, MarbleTexture_VariesWithPosition) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	EXPECT_NE(tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f)),
			  tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f)));
}

TEST(ProcTex, MarbleTexture_ScaleAffectsResult) {
	MarbleTexture<float> tex1, tex2;
	tex1.mapping = tex2.mapping = MakeIdentity3D();
	tex1.scale_ = 1.f; tex2.scale_ = 2.f;
	auto ctx = MakeCtx(0.5f, 0.5f, 0.5f);
	EXPECT_NE(tex1.Evaluate(ctx), tex2.Evaluate(ctx));
}

// ===========================================================================
// MixFnTexture
// ===========================================================================

TEST(ProcTex, MixFnTexture_Amount0) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 3.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 7.f; };
	tex.amount = [](const TextureEvalContext&){ return 0.f; };
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 3.f);
}

TEST(ProcTex, MixFnTexture_Amount1) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 3.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 7.f; };
	tex.amount = [](const TextureEvalContext&){ return 1.f; };
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 7.f);
}

TEST(ProcTex, MixFnTexture_HalfAmount) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 0.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 4.f; };
	tex.amount = [](const TextureEvalContext&){ return 0.5f; };
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 2.f);
}

// ===========================================================================
// ScaledFnTexture
// ===========================================================================

TEST(ProcTex, ScaledFnTexture_Basic) {
	ScaledFnTexture tex;
	tex.tex   = [](const TextureEvalContext&){ return 5.f; };
	tex.scale = [](const TextureEvalContext&){ return 3.f; };
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 15.f);
}

TEST(ProcTex, ScaledFnTexture_ZeroScale) {
	ScaledFnTexture tex;
	tex.tex   = [](const TextureEvalContext&){ return 99.f; };
	tex.scale = [](const TextureEvalContext&){ return 0.f; };
	EXPECT_FLOAT_EQ(tex.Evaluate(MakeCtx(0,0,0)), 0.f);
}

// ===========================================================================
// DirectionMixFnTexture
// ===========================================================================

TEST(ProcTex, DirectionMixFnTexture_ParallelNormal) {
	DirectionMixFnTexture tex;
	tex.tex0 = [](const TextureEvalContext&){ return 10.f; };
	tex.tex1 = [](const TextureEvalContext&){ return 20.f; };
	tex.dir  = {0.f, 0.f, 1.f};
	TextureEvalContext ctx = MakeCtx(0,0,0);
	ctx.n = {0.f, 0.f, 1.f};
	// amt = |dot(n,dir)| = 1 -> result = 1*tex0 = 10
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 10.f);
}

TEST(ProcTex, DirectionMixFnTexture_PerpendicularNormal) {
	DirectionMixFnTexture tex;
	tex.tex0 = [](const TextureEvalContext&){ return 10.f; };
	tex.tex1 = [](const TextureEvalContext&){ return 20.f; };
	tex.dir  = {0.f, 0.f, 1.f};
	TextureEvalContext ctx = MakeCtx(0,0,0);
	ctx.n = {1.f, 0.f, 0.f};
	// amt = 0 -> result = 0*tex0 + 1*tex1 = 20
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 20.f);
}

// ===========================================================================
// checkerboard_weight helpers (direct unit tests)
// ===========================================================================

TEST(ProcTex, CheckerboardWeight2D_EvenCellSmallFootprint) {
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	EXPECT_NEAR(checkerboard_weight_2d<float>(c), 0.f, 0.05f);
}

TEST(ProcTex, CheckerboardWeight2D_OddCellSmallFootprint) {
	TexCoord2D c;
	c.st   = {1.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	EXPECT_NEAR(checkerboard_weight_2d<float>(c), 1.f, 0.05f);
}

TEST(ProcTex, CheckerboardWeight2D_LargeFootprintNearHalf) {
	// With large dsdx AND dtdy, bf in both dimensions should be small,
	// so weight approaches 0.5.
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 5.f; c.dsdy = 0.f;
	c.dtdx = 0.f; c.dtdy = 5.f;
	EXPECT_NEAR(checkerboard_weight_2d<float>(c), 0.5f, 0.1f);
}

TEST(ProcTex, CheckerboardWeight3D_SmallFootprintEvenCell) {
	TexCoord3D c;
	c.p    = {0.5f, 0.5f, 0.5f};
	c.dpdx = {1e-5f, 0.f, 0.f};
	c.dpdy = {0.f, 1e-5f, 0.f};
	EXPECT_NEAR(checkerboard_weight_3d<float>(c), 0.f, 0.05f);
}

// ===========================================================================
// inside_polka_dot helpers
// ===========================================================================

TEST(ProcTex, InsidePolkaDot_ReturnsBool) {
	for (float s = -2.f; s < 3.f; s += 0.4f) {
		for (float t = -2.f; t < 3.f; t += 0.4f) {
			bool r = inside_polka_dot<float>(s, t);
			(void)r;  // just verify it doesn't crash
		}
	}
}

TEST(ProcTex, InsidePolkaDot_Deterministic) {
	bool a = inside_polka_dot<float>(0.7f, 1.3f);
	bool b = inside_polka_dot<float>(0.7f, 1.3f);
	EXPECT_EQ(a, b);
}
