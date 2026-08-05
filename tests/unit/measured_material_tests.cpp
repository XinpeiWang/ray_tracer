// measured_material_tests.cpp
// Unit tests for the MeasuredMaterial wrapper in src/shared/materials.h.
//
// pbrt-v4 alignment reference: materials.h MeasuredMaterial (Apache-2.0)
//
// Tests:
//   1. DefaultConstruct: default-constructed MeasuredMaterial has null brdf pointer
//   2. WavelengthDefaults: default wavelengths match sRGB primaries (R~612, G~549, B~465)
//   3. ConstructFromPointer: brdf pointer and wavelengths are stored correctly
//   4. HasSubsurfaceScattering: always false (mirrors pbrt-v4)
//   5. GetBxdf_PropagatesPointer: get_bxdf() returns a MeasuredBxDF holding the same pointer
//   6. GetBxdf_PropagatesWavelengths: get_bxdf() propagates all three lambda values
//   7. GetBxdf_IgnoresContext: different contexts produce the same BxDF (no texture dependency)
//   8. NullBrdf_GetBxdfReturnsNullData: get_bxdf() on null-brdf material yields MeasuredBxDF with null data

#include <gtest/gtest.h>
#include <cmath>

#include "../../src/shared/materials.h"

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

// Build a minimal (non-usable for evaluation) MeasuredBRDFData so we can
// take its address without invoking Build().  Tests here only exercise the
// material wrapper layer, not BxDF evaluation.
static MeasuredBRDFData g_dummy_data;

static MaterialEvalContext<double> make_ctx(
	double px = 0, double py = 0, double pz = 0,
	double nx = 0, double ny = 1, double nz = 0,
	double wo_x = 0, double wo_y = 0, double wo_z = 1,
	double u = 0.5, double v = 0.5)
{
	MaterialEvalContext<double> ctx;
	ctx.px = px; ctx.py = py; ctx.pz = pz;
	ctx.nx = nx; ctx.ny = ny; ctx.nz = nz;
	ctx.wo_x = wo_x; ctx.wo_y = wo_y; ctx.wo_z = wo_z;
	ctx.u = u; ctx.v = v;
	return ctx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(MeasuredMaterialTest, DefaultConstruct) {
	MeasuredMaterial mat;
	EXPECT_EQ(mat.brdf, nullptr);
}

TEST(MeasuredMaterialTest, WavelengthDefaults) {
	MeasuredMaterial mat;
	// Default wavelengths mirror the sRGB primary approximations used in
	// pbrt-v4's measured material tests (R~612nm, G~549nm, B~465nm).
	EXPECT_NEAR(mat.lambda_r, 612.0f, 1.0f);
	EXPECT_NEAR(mat.lambda_g, 549.0f, 1.0f);
	EXPECT_NEAR(mat.lambda_b, 465.0f, 1.0f);
}

TEST(MeasuredMaterialTest, ConstructFromPointer) {
	MeasuredMaterial mat(&g_dummy_data, 630.0f, 532.0f, 468.0f);
	EXPECT_EQ(mat.brdf, &g_dummy_data);
	EXPECT_FLOAT_EQ(mat.lambda_r, 630.0f);
	EXPECT_FLOAT_EQ(mat.lambda_g, 532.0f);
	EXPECT_FLOAT_EQ(mat.lambda_b, 468.0f);
}

TEST(MeasuredMaterialTest, HasSubsurfaceScattering) {
	// pbrt-v4 MeasuredMaterial::HasSubsurfaceScattering() is always false.
	EXPECT_FALSE(MeasuredMaterial::has_subsurface_scattering());

	MeasuredMaterial mat(&g_dummy_data);
	// Also confirm via an instance (constexpr is accessible both ways).
	EXPECT_FALSE(mat.has_subsurface_scattering());
}

TEST(MeasuredMaterialTest, GetBxdf_PropagatesPointer) {
	MeasuredMaterial mat(&g_dummy_data, 612.0f, 549.0f, 465.0f);
	auto ctx = make_ctx();
	MeasuredBxDF<double> bxdf = mat.get_bxdf(ctx);
	EXPECT_EQ(bxdf.data, &g_dummy_data);
}

TEST(MeasuredMaterialTest, GetBxdf_PropagatesWavelengths) {
	const float lr = 700.0f, lg = 550.0f, lb = 440.0f;
	MeasuredMaterial mat(&g_dummy_data, lr, lg, lb);
	auto ctx = make_ctx();
	MeasuredBxDF<double> bxdf = mat.get_bxdf(ctx);
	EXPECT_FLOAT_EQ(bxdf.lambda[0], lr);
	EXPECT_FLOAT_EQ(bxdf.lambda[1], lg);
	EXPECT_FLOAT_EQ(bxdf.lambda[2], lb);
}

TEST(MeasuredMaterialTest, GetBxdf_IgnoresContext) {
	// MeasuredMaterial has no textures: different contexts must yield the
	// same BxDF (same pointer + wavelengths).
	MeasuredMaterial mat(&g_dummy_data, 612.0f, 549.0f, 465.0f);

	auto ctx1 = make_ctx(0, 0, 0,  0, 1, 0,  0, 0, 1,  0.1, 0.2);
	auto ctx2 = make_ctx(5, 3, 1,  0, 0, 1,  1, 0, 0,  0.9, 0.7);

	MeasuredBxDF<double> b1 = mat.get_bxdf(ctx1);
	MeasuredBxDF<double> b2 = mat.get_bxdf(ctx2);

	EXPECT_EQ(b1.data, b2.data);
	EXPECT_FLOAT_EQ(b1.lambda[0], b2.lambda[0]);
	EXPECT_FLOAT_EQ(b1.lambda[1], b2.lambda[1]);
	EXPECT_FLOAT_EQ(b1.lambda[2], b2.lambda[2]);
}

TEST(MeasuredMaterialTest, NullBrdf_GetBxdfReturnsNullData) {
	MeasuredMaterial mat; // brdf = nullptr
	auto ctx = make_ctx();
	MeasuredBxDF<double> bxdf = mat.get_bxdf(ctx);
	EXPECT_EQ(bxdf.data, nullptr);
}
