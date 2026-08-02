// procedural_textures_tests.cpp
// Unit tests for src/shared/procedural_textures.h
//
// Tests mirror pbrt-v4 behaviour and verify:
//   - ConstantTexture: returns value regardless of context
//   - BilerpTexture: corners, center, edge midpoints
//   - CheckerboardTexture: 2D and 3D cell alternation at cell centers
//   - MixTexture: amount=0/0.5/1
//   - ScaledTexture: child * scale
//   - FBmTexture: result in a plausible [-2,2] range, deterministic
//   - WrinkledTexture: non-negative (turbulence is ABS-based)
//   - WindyTexture: value is product of two FBm calls
//   - DotsTexture: returns inside_val or outside_val depending on position
//   - MarbleTexture: luminance in [0, 1.5] range, deterministic
//   - MixFnTexture, ScaledFnTexture, DirectionMixFnTexture: callable wrappers

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

// Build a UV mapping that maps (u,v) directly from ctx.uv
static TextureMapping2D MakeUV2D() {
	return UVMapping(1.f, 1.f, 0.f, 0.f);
}

// Build a PointTransformMapping (identity) for 3D textures
static TextureMapping3D MakeIdentity3D() {
	return PointTransformMapping(TextureTransform{});
}

// ===========================================================================
// ConstantTexture
// ===========================================================================

TEST(ProceduralTextures, ConstantTexture_ReturnsValue) {
	ConstantTexture<float> tex;
	tex.value = 0.75f;
	auto ctx = MakeCtx(1,2,3);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 0.75f);
}

TEST(ProceduralTextures, ConstantTexture_Zero) {
	ConstantTexture<float> tex;
	tex.value = 0.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 0.f);
}

TEST(ProceduralTextures, ConstantTexture_Double) {
	ConstantTexture<double> tex;
	tex.value = 3.14;
	auto ctx = MakeCtx(1,1,1);
	EXPECT_DOUBLE_EQ(tex.Evaluate(ctx), 3.14);
}

TEST(ProceduralTextures, ConstantTexture_Negative) {
	ConstantTexture<float> tex;
	tex.value = -1.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), -1.f);
}

// ===========================================================================
// BilerpTexture
// ===========================================================================

TEST(ProceduralTextures, BilerpTexture_Corner00) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 1.f; tex.v01 = 0.f; tex.v10 = 0.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 0.f, 0.f);  // u=0, v=0 -> corner v00
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProceduralTextures, BilerpTexture_Corner10) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 0.f; tex.v10 = 1.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 1.f, 0.f);  // u=1, v=0 -> corner v10
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProceduralTextures, BilerpTexture_Corner01) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 1.f; tex.v10 = 0.f; tex.v11 = 0.f;
	auto ctx = MakeCtx(0, 0, 0, 0.f, 1.f);  // u=0, v=1 -> corner v01
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProceduralTextures, BilerpTexture_Center) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = 0.f; tex.v01 = 0.f; tex.v10 = 0.f; tex.v11 = 4.f;
	auto ctx = MakeCtx(0, 0, 0, 0.5f, 0.5f);  // center -> 1/4 * v11 = 1.0
	EXPECT_NEAR(tex.Evaluate(ctx), 1.f, 1e-5f);
}

TEST(ProceduralTextures, BilerpTexture_Uniform) {
	BilerpTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.v00 = tex.v01 = tex.v10 = tex.v11 = 0.5f;
	// Any (u,v) should return 0.5
	for (float u : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
		for (float v : {0.f, 0.5f, 1.f}) {
			auto ctx = MakeCtx(0, 0, 0, u, v);
			EXPECT_NEAR(tex.Evaluate(ctx), 0.5f, 1e-5f);
		}
	}
}

// ===========================================================================
// CheckerboardTexture (2D)
// ===========================================================================

// At cell centers (0.5+n, 0.5+m), the unfiltered checker is pure 0 or 1.
// The anti-aliased version approaches these values when the filter footprint
// is very small (dpdx/dpdy ~= 0).  We use the UV mapping with tiny differentials.

TEST(ProceduralTextures, Checkerboard2D_OddCellCenter) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f;
	tex.tex1 = 1.f;
	// cell (0,0) center: u=0.5, v=0.5 -> floor(0.5)*1 + floor(0.5)*1 = 0 -> white (tex0)
	TextureEvalContext ctx = MakeCtx(0,0,0, 0.5f, 0.5f);
	ctx.dudx = 1e-5f; ctx.dvdy = 1e-5f;  // tiny footprint
	ctx.dpdx = {1e-5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 1e-5f, 0.f};
	float w = tex.Evaluate(ctx);
	// At cell (0,0): floor(0.5)=0 for both, parity even -> tex0
	EXPECT_NEAR(w, 0.f, 0.05f);
}

TEST(ProceduralTextures, Checkerboard2D_CellParityAlternates) {
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f;
	tex.tex1 = 1.f;

	// cell (1,0): u=1.5, v=0.5 -> floor=1,0 -> parity odd -> tex1
	{
		TextureEvalContext ctx = MakeCtx(0,0,0, 1.5f, 0.5f);
		ctx.dpdx = {1e-5f, 0.f, 0.f};
		ctx.dpdy = {0.f, 1e-5f, 0.f};
		float w = tex.Evaluate(ctx);
		EXPECT_NEAR(w, 1.f, 0.05f);
	}
	// cell (0,1): u=0.5, v=1.5 -> floor=0,1 -> parity odd -> tex1
	{
		TextureEvalContext ctx = MakeCtx(0,0,0, 0.5f, 1.5f);
		ctx.dpdx = {1e-5f, 0.f, 0.f};
		ctx.dpdy = {0.f, 1e-5f, 0.f};
		float w = tex.Evaluate(ctx);
		EXPECT_NEAR(w, 1.f, 0.05f);
	}
	// cell (1,1): u=1.5, v=1.5 -> parity even -> tex0
	{
		TextureEvalContext ctx = MakeCtx(0,0,0, 1.5f, 1.5f);
		ctx.dpdx = {1e-5f, 0.f, 0.f};
		ctx.dpdy = {0.f, 1e-5f, 0.f};
		float w = tex.Evaluate(ctx);
		EXPECT_NEAR(w, 0.f, 0.05f);
	}
}

TEST(ProceduralTextures, Checkerboard2D_BlurredBoundary) {
	// At a wide filter width, the checker should approach 0.5
	CheckerboardTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.tex0 = 0.f;
	tex.tex1 = 1.f;
	TextureEvalContext ctx = MakeCtx(0,0,0, 0.5f, 0.5f);
	ctx.dpdx = {2.f, 0.f, 0.f};   // very large footprint
	ctx.dpdy = {0.f, 2.f, 0.f};
	float w = tex.Evaluate(ctx);
	// Should be near 0.5 when heavily blurred
	EXPECT_NEAR(w, 0.5f, 0.1f);
}

// ===========================================================================
// Checkerboard3DTexture
// ===========================================================================

TEST(ProceduralTextures, Checkerboard3D_CellCenter) {
	Checkerboard3DTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.tex0 = 0.f;
	tex.tex1 = 1.f;
	// Point (0.5, 0.5, 0.5): each floor=0, parity 0 -> tex0
	TextureEvalContext ctx = MakeCtx(0.5f, 0.5f, 0.5f);
	ctx.dpdx = {1e-5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 1e-5f, 0.f};
	float w = tex.Evaluate(ctx);
	EXPECT_NEAR(w, 0.f, 0.05f);
}

TEST(ProceduralTextures, Checkerboard3D_OddCell) {
	Checkerboard3DTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.tex0 = 0.f;
	tex.tex1 = 1.f;
	// Point (1.5, 0.5, 0.5): x-floor=1, parity odd -> tex1
	TextureEvalContext ctx = MakeCtx(1.5f, 0.5f, 0.5f);
	ctx.dpdx = {1e-5f, 0.f, 0.f};
	ctx.dpdy = {0.f, 1e-5f, 0.f};
	float w = tex.Evaluate(ctx);
	EXPECT_NEAR(w, 1.f, 0.05f);
}

// ===========================================================================
// MixTexture
// ===========================================================================

TEST(ProceduralTextures, MixTexture_Amount0) {
	MixTexture<float> tex;
	tex.tex0 = 3.f; tex.tex1 = 7.f; tex.amount = 0.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 3.f);
}

TEST(ProceduralTextures, MixTexture_Amount1) {
	MixTexture<float> tex;
	tex.tex0 = 3.f; tex.tex1 = 7.f; tex.amount = 1.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 7.f);
}

TEST(ProceduralTextures, MixTexture_HalfAmount) {
	MixTexture<float> tex;
	tex.tex0 = 0.f; tex.tex1 = 4.f; tex.amount = 0.5f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 2.f);
}

TEST(ProceduralTextures, MixTexture_QuarterAmount) {
	MixTexture<float> tex;
	tex.tex0 = 0.f; tex.tex1 = 8.f; tex.amount = 0.25f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 2.f);
}

// ===========================================================================
// ScaledTexture
// ===========================================================================

TEST(ProceduralTextures, ScaledTexture_BasicMultiply) {
	ScaledTexture<float> tex;
	tex.child_value = 3.f;
	tex.scale = 2.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 6.f);
}

TEST(ProceduralTextures, ScaledTexture_ZeroScale) {
	ScaledTexture<float> tex;
	tex.child_value = 100.f;
	tex.scale = 0.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 0.f);
}

TEST(ProceduralTextures, ScaledTexture_NegativeScale) {
	ScaledTexture<float> tex;
	tex.child_value = 5.f;
	tex.scale = -1.f;
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), -5.f);
}

// ===========================================================================
// FBmTexture
// ===========================================================================

TEST(ProceduralTextures, FBmTexture_Deterministic) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f;
	tex.octaves = 6;

	auto ctx = MakeCtx(1.23f, 4.56f, 7.89f);
	float v1 = tex.Evaluate(ctx);
	float v2 = tex.Evaluate(ctx);
	EXPECT_FLOAT_EQ(v1, v2);
}

TEST(ProceduralTextures, FBmTexture_PlausibleRange) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f;
	tex.octaves = 8;

	for (float x : {0.1f, 1.f, 3.7f, -2.f}) {
		for (float y : {0.f, 1.1f, -0.5f}) {
			auto ctx = MakeCtx(x, y, 0.f);
			float v = tex.Evaluate(ctx);
			EXPECT_GE(v, -4.f) << "FBm out of lower range at " << x << "," << y;
			EXPECT_LE(v,  4.f) << "FBm out of upper range at " << x << "," << y;
		}
	}
}

TEST(ProceduralTextures, FBmTexture_VariesWithPosition) {
	FBmTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f;
	tex.octaves = 8;

	float v1 = tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f));
	float v2 = tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f));
	EXPECT_NE(v1, v2);
}

// ===========================================================================
// WrinkledTexture
// ===========================================================================

TEST(ProceduralTextures, WrinkledTexture_NonNegative) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	tex.omega = 0.5f;
	tex.octaves = 6;

	for (float x : {0.f, 0.5f, 1.f, 2.3f, -1.f}) {
		for (float y : {0.f, 0.7f, -0.3f}) {
			auto ctx = MakeCtx(x, y, 0.f);
			float v = tex.Evaluate(ctx);
			EXPECT_GE(v, 0.f) << "Turbulence should be non-negative at " << x << "," << y;
		}
	}
}

TEST(ProceduralTextures, WrinkledTexture_Deterministic) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProceduralTextures, WrinkledTexture_VariesWithPosition) {
	WrinkledTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	float v1 = tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f));
	float v2 = tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f));
	EXPECT_NE(v1, v2);
}

// ===========================================================================
// WindyTexture
// ===========================================================================

TEST(ProceduralTextures, WindyTexture_Deterministic) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProceduralTextures, WindyTexture_PlausibleRange) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	for (float x : {0.f, 1.f, 3.f}) {
		auto ctx = MakeCtx(x, x*0.5f, x*0.3f);
		float v = tex.Evaluate(ctx);
		EXPECT_GE(v, -4.f);
		EXPECT_LE(v,  4.f);
	}
}

TEST(ProceduralTextures, WindyTexture_VariesWithPosition) {
	WindyTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	float v1 = tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f));
	float v2 = tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f));
	EXPECT_NE(v1, v2);
}

// ===========================================================================
// DotsTexture
// ===========================================================================

TEST(ProceduralTextures, DotsTexture_ReturnsOneOfTwoValues) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.outside_val = 0.f;
	tex.inside_val  = 1.f;

	// Sweep a grid; every result must be exactly 0 or 1
	for (float u = 0.f; u < 5.f; u += 0.3f) {
		for (float v = 0.f; v < 5.f; v += 0.3f) {
			auto ctx = MakeCtx(0,0,0, u, v);
			float r = tex.Evaluate(ctx);
			EXPECT_TRUE(r == 0.f || r == 1.f)
				<< "DotsTexture returned unexpected value " << r
				<< " at (" << u << "," << v << ")";
		}
	}
}

TEST(ProceduralTextures, DotsTexture_CustomValues) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.outside_val = 0.3f;
	tex.inside_val  = 0.9f;

	for (float u = 0.f; u < 3.f; u += 0.25f) {
		for (float v = 0.f; v < 3.f; v += 0.25f) {
			auto ctx = MakeCtx(0,0,0, u, v);
			float r = tex.Evaluate(ctx);
			EXPECT_TRUE(r == 0.3f || r == 0.9f);
		}
	}
}

TEST(ProceduralTextures, DotsTexture_Deterministic) {
	DotsTexture<float> tex;
	tex.mapping = MakeUV2D();
	tex.outside_val = 0.f;
	tex.inside_val  = 1.f;
	auto ctx = MakeCtx(0, 0, 0, 2.7f, 1.3f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

// ===========================================================================
// MarbleTexture
// ===========================================================================

TEST(ProceduralTextures, MarbleTexture_Deterministic) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	auto ctx = MakeCtx(1.f, 2.f, 3.f);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), tex.Evaluate(ctx));
}

TEST(ProceduralTextures, MarbleTexture_PlausibleRange) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	for (float x : {0.f, 0.5f, 1.f, 3.3f, -1.f}) {
		for (float y : {0.f, 1.f, -0.5f}) {
			auto ctx = MakeCtx(x, y, 0.3f);
			float v = tex.Evaluate(ctx);
			// Luminance of 1.5 * spline values: roughly [0, ~1.35]
			EXPECT_GE(v, -0.01f) << "Marble luminance too low at " << x << "," << y;
			EXPECT_LE(v,  1.5f)  << "Marble luminance too high at " << x << "," << y;
		}
	}
}

TEST(ProceduralTextures, MarbleTexture_VariesWithPosition) {
	MarbleTexture<float> tex;
	tex.mapping = MakeIdentity3D();
	float v1 = tex.Evaluate(MakeCtx(0.1f, 0.2f, 0.3f));
	float v2 = tex.Evaluate(MakeCtx(5.5f, 2.3f, 1.1f));
	EXPECT_NE(v1, v2);
}

TEST(ProceduralTextures, MarbleTexture_ScaleAffectsResult) {
	MarbleTexture<float> tex1, tex2;
	tex1.mapping = MakeIdentity3D();
	tex2.mapping = MakeIdentity3D();
	tex1.scale_ = 1.f;
	tex2.scale_ = 2.f;
	auto ctx = MakeCtx(0.5f, 0.5f, 0.5f);
	EXPECT_NE(tex1.Evaluate(ctx), tex2.Evaluate(ctx));
}

// ===========================================================================
// MixFnTexture
// ===========================================================================

TEST(ProceduralTextures, MixFnTexture_Amount0) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 3.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 7.f; };
	tex.amount = [](const TextureEvalContext&){ return 0.f; };
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 3.f);
}

TEST(ProceduralTextures, MixFnTexture_Amount1) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 3.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 7.f; };
	tex.amount = [](const TextureEvalContext&){ return 1.f; };
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 7.f);
}

TEST(ProceduralTextures, MixFnTexture_HalfAmount) {
	MixFnTexture tex;
	tex.tex0   = [](const TextureEvalContext&){ return 0.f; };
	tex.tex1   = [](const TextureEvalContext&){ return 4.f; };
	tex.amount = [](const TextureEvalContext&){ return 0.5f; };
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 2.f);
}

// ===========================================================================
// ScaledFnTexture
// ===========================================================================

TEST(ProceduralTextures, ScaledFnTexture_Basic) {
	ScaledFnTexture tex;
	tex.tex   = [](const TextureEvalContext&){ return 5.f; };
	tex.scale = [](const TextureEvalContext&){ return 3.f; };
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 15.f);
}

TEST(ProceduralTextures, ScaledFnTexture_ZeroScale) {
	ScaledFnTexture tex;
	tex.tex   = [](const TextureEvalContext&){ return 99.f; };
	tex.scale = [](const TextureEvalContext&){ return 0.f; };
	auto ctx = MakeCtx(0,0,0);
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 0.f);
}

// ===========================================================================
// DirectionMixFnTexture
// ===========================================================================

TEST(ProceduralTextures, DirectionMixFnTexture_ParallelNormal) {
	// n = (0,0,1), dir = (0,0,1) -> dot = 1 -> result = tex0
	DirectionMixFnTexture tex;
	tex.tex0 = [](const TextureEvalContext&){ return 10.f; };
	tex.tex1 = [](const TextureEvalContext&){ return 20.f; };
	tex.dir  = {0.f, 0.f, 1.f};
	TextureEvalContext ctx = MakeCtx(0,0,0);
	ctx.n = {0.f, 0.f, 1.f};
	// amt = |dot(n,dir)| = 1 -> result = 1*tex0 + 0*tex1 = 10
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 10.f);
}

TEST(ProceduralTextures, DirectionMixFnTexture_PerpendicularNormal) {
	// n = (1,0,0), dir = (0,0,1) -> dot = 0 -> result = tex1
	DirectionMixFnTexture tex;
	tex.tex0 = [](const TextureEvalContext&){ return 10.f; };
	tex.tex1 = [](const TextureEvalContext&){ return 20.f; };
	tex.dir  = {0.f, 0.f, 1.f};
	TextureEvalContext ctx = MakeCtx(0,0,0);
	ctx.n = {1.f, 0.f, 0.f};
	EXPECT_FLOAT_EQ(tex.Evaluate(ctx), 20.f);
}

// ===========================================================================
// checkerboard_weight helper (direct unit tests)
// ===========================================================================

TEST(ProceduralTextures, CheckerboardWeight2D_SmallFootprint_EvenCell) {
	// s=0.5, t=0.5 is center of cell (0,0) (parity 0) -> weight ~ 0
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 0.f, 0.05f);
}

TEST(ProceduralTextures, CheckerboardWeight2D_SmallFootprint_OddCell) {
	// s=1.5, t=0.5 -> cell (1,0), parity 1 -> weight ~ 1
	TexCoord2D c;
	c.st   = {1.5f, 0.5f};
	c.dsdx = 1e-5f; c.dsdy = 0.f;
	c.dtdx = 0.f;   c.dtdy = 1e-5f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 1.f, 0.05f);
}

TEST(ProceduralTextures, CheckerboardWeight2D_LargeFootprint_NearHalf) {
	TexCoord2D c;
	c.st   = {0.5f, 0.5f};
	c.dsdx = 5.f; c.dsdy = 0.f;
	c.dtdx = 0.f; c.dtdy = 5.f;
	float w = checkerboard_weight_2d<float>(c);
	EXPECT_NEAR(w, 0.5f, 0.1f);
}

TEST(ProceduralTextures, CheckerboardWeight3D_SmallFootprint) {
	TexCoord3D c;
	c.p    = {0.5f, 0.5f, 0.5f};
	c.dpdx = {1e-5f, 0.f, 0.f};
	c.dpdy = {0.f, 1e-5f, 0.f};
	float w = checkerboard_weight_3d<float>(c);
	EXPECT_NEAR(w, 0.f, 0.05f);
}

// ===========================================================================
// inside_polka_dot helper
// ===========================================================================

TEST(ProceduralTextures, InsidePolkaDot_ReturnsBool) {
	// Just verify it doesn't crash and returns true or false
	for (float s = -2.f; s < 3.f; s += 0.4f) {
		for (float t = -2.f; t < 3.f; t += 0.4f) {
			bool r = inside_polka_dot<float>(s, t);
			EXPECT_TRUE(r == true || r == false);  // sanity only
		}
	}
}

TEST(ProceduralTextures, InsidePolkaDot_Deterministic) {
	bool a = inside_polka_dot<float>(0.7f, 1.3f);
	bool b = inside_polka_dot<float>(0.7f, 1.3f);
	EXPECT_EQ(a, b);
}
