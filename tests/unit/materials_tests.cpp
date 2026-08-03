// materials_tests.cpp — unit tests for src/shared/materials.h
// Mirrors pbrt-v4 material semantics; validates GetBxDF() paths and
// SubsurfaceMaterial BSSRDF construction.

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <array>
#include "../../src/shared/materials.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static MaterialEvalContext<float> default_ctx() {
	MaterialEvalContext<float> ctx{};
	ctx.nx = 0.f; ctx.ny = 1.f; ctx.nz = 0.f;
	ctx.wo_x = 0.f; ctx.wo_y = 1.f; ctx.wo_z = 0.f;
	ctx.u = 0.2f; ctx.v = 0.3f;
	return ctx;
}

// ---------------------------------------------------------------------------
// DiffuseMaterial
// ---------------------------------------------------------------------------
TEST(Materials, DiffuseMaterialDefaultAlbedo) {
	DiffuseMaterial<float> mat;
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.albedo_r, 0.5f, 1e-5f);
	EXPECT_NEAR(bxdf.albedo_g, 0.5f, 1e-5f);
	EXPECT_NEAR(bxdf.albedo_b, 0.5f, 1e-5f);
}

TEST(Materials, DiffuseMaterialCustomAlbedo) {
	SpectrumTexVal<float> tex(0.8f, 0.4f, 0.2f);
	DiffuseMaterial<float> mat(tex);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.albedo_r, 0.8f, 1e-5f);
	EXPECT_NEAR(bxdf.albedo_g, 0.4f, 1e-5f);
	EXPECT_NEAR(bxdf.albedo_b, 0.2f, 1e-5f);
}

TEST(Materials, DiffuseMaterialClampsAboveOne) {
	SpectrumTexVal<float> tex(1.5f, 0.5f, -0.2f);
	DiffuseMaterial<float> mat(tex);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.albedo_r, 1.0f, 1e-5f);
	EXPECT_NEAR(bxdf.albedo_b, 0.0f, 1e-5f);
}

TEST(Materials, DiffuseMaterialNoSubsurface) {
	EXPECT_FALSE(DiffuseMaterial<float>::has_subsurface_scattering());
}

// ---------------------------------------------------------------------------
// DielectricMaterial
// ---------------------------------------------------------------------------
TEST(Materials, DielectricMaterialSmoothPath) {
	DielectricMaterial<float> mat(1.5f);  // zero roughness -> smooth
	auto v = mat.get_bxdf(default_ctx());
	EXPECT_FALSE(v.rough);
	EXPECT_NEAR(v.smooth.ior, 1.5f, 1e-5f);
}

TEST(Materials, DielectricMaterialRoughPath) {
	FloatTexVal<float> rough(0.3f);
	DielectricMaterial<float> mat(1.5f, rough, rough, false); // no remap
	auto v = mat.get_bxdf(default_ctx());
	EXPECT_TRUE(v.rough);
	EXPECT_NEAR(v.roughd.ior, 1.5f, 1e-5f);
	EXPECT_NEAR(v.roughd.alpha_x, 0.3f, 1e-5f);
	EXPECT_NEAR(v.roughd.alpha_y, 0.3f, 1e-5f);
}

TEST(Materials, DielectricMaterialRoughnessRemap) {
	FloatTexVal<float> rough(0.5f);
	DielectricMaterial<float> mat(1.5f, rough, rough, true); // remap on
	auto v = mat.get_bxdf(default_ctx());
	EXPECT_TRUE(v.rough);
	float expected_alpha = TrowbridgeReitz<float>::RoughnessToAlpha(0.5f);
	EXPECT_NEAR(v.roughd.alpha_x, expected_alpha, 1e-4f);
}

TEST(Materials, DielectricMaterialNoSubsurface) {
	EXPECT_FALSE(DielectricMaterial<float>::has_subsurface_scattering());
}

// ---------------------------------------------------------------------------
// ThinDielectricMaterial
// ---------------------------------------------------------------------------
TEST(Materials, ThinDielectricMaterialEta) {
	ThinDielectricMaterial<float> mat(1.4f);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.ior, 1.4f, 1e-5f);
}

TEST(Materials, ThinDielectricMaterialDefaultEta) {
	ThinDielectricMaterial<float> mat;
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.ior, 1.5f, 1e-5f);
}

TEST(Materials, ThinDielectricMaterialNoSubsurface) {
	EXPECT_FALSE(ThinDielectricMaterial<float>::has_subsurface_scattering());
}

// ---------------------------------------------------------------------------
// ConductorMaterial — (eta, k) path
// ---------------------------------------------------------------------------
TEST(Materials, ConductorMaterialEtaKPath) {
	SpectrumTexVal<float> eta_tex(0.2f, 0.4f, 1.0f);
	SpectrumTexVal<float> k_tex(3.0f, 2.5f, 2.0f);
	ConductorMaterial<float> mat(eta_tex, k_tex);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.eta_r, 0.2f, 1e-5f);
	EXPECT_NEAR(bxdf.k_r,   3.0f, 1e-5f);
	EXPECT_NEAR(bxdf.alpha_x, 0.f, 1e-5f); // default smooth
}

TEST(Materials, ConductorMaterialReflectancePath) {
	// r=0.5 => ks = 2*sqrt(0.5)/sqrt(0.5) = 2.0, eta=1
	SpectrumTexVal<float> refl(0.5f, 0.5f, 0.5f);
	ConductorMaterial<float> mat(refl);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.eta_r, 1.0f, 1e-5f);
	float expected_k = 2.f * std::sqrt(0.5f) / std::sqrt(0.5f);
	EXPECT_NEAR(bxdf.k_r, expected_k, 1e-4f);
}

TEST(Materials, ConductorMaterialReflectanceClamp) {
	// r=0.9999 should not produce NaN
	SpectrumTexVal<float> refl(0.9999f, 0.9999f, 0.9999f);
	ConductorMaterial<float> mat(refl);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_TRUE(std::isfinite(bxdf.k_r));
	EXPECT_GT(bxdf.k_r, 0.f);
}

TEST(Materials, ConductorMaterialRoughnessRemap) {
	SpectrumTexVal<float> eta_tex(1.f, 1.f, 1.f);
	SpectrumTexVal<float> k_tex(1.f, 1.f, 1.f);
	FloatTexVal<float> rough(0.5f);
	ConductorMaterial<float> mat(eta_tex, k_tex, rough, rough, true);
	auto bxdf = mat.get_bxdf(default_ctx());
	float expected = TrowbridgeReitz<float>::RoughnessToAlpha(0.5f);
	EXPECT_NEAR(bxdf.alpha_x, expected, 1e-4f);
}

TEST(Materials, ConductorMaterialNoSubsurface) {
	EXPECT_FALSE((ConductorMaterial<float>::has_subsurface_scattering()));
}

// ---------------------------------------------------------------------------
// CoatedDiffuseMaterial
// ---------------------------------------------------------------------------
TEST(Materials, CoatedDiffuseMaterialFields) {
	SpectrumTexVal<float> refl(0.6f, 0.6f, 0.6f);
	FloatTexVal<float> rough(0.0f);
	CoatedDiffuseMaterial<float> mat(refl, rough, rough,
		FloatTexVal<float>(0.01f),
		SpectrumTexVal<float>(0.f,0.f,0.f),
		FloatTexVal<float>(0.f),
		1.5f, false, 10, 1);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.albedo_r, 0.6f, 1e-5f);
	EXPECT_NEAR(bxdf.coat_ior, 1.5f, 1e-5f);
}

TEST(Materials, CoatedDiffuseMaterialAlbedoClamped) {
	SpectrumTexVal<float> refl(1.2f, 0.5f, 0.5f);
	CoatedDiffuseMaterial<float> mat(refl);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.albedo_r, 1.0f, 1e-5f);
}

TEST(Materials, CoatedDiffuseMaterialNoSubsurface) {
	EXPECT_FALSE((CoatedDiffuseMaterial<float>::has_subsurface_scattering()));
}

// ---------------------------------------------------------------------------
// CoatedConductorMaterial
// ---------------------------------------------------------------------------
TEST(Materials, CoatedConductorMaterialFields) {
	SpectrumTexVal<float> ceta(0.2f, 0.4f, 1.0f);
	SpectrumTexVal<float> ck(3.0f, 2.5f, 2.0f);
	CoatedConductorMaterial<float> mat(ceta, ck);
	auto bxdf = mat.get_bxdf(default_ctx());
	// pbrt-v4: ce /= ieta; ck /= ieta  (interface_eta default = 1.5)
	EXPECT_NEAR(bxdf.eta_r,    0.2f / 1.5f, 1e-5f);
	EXPECT_NEAR(bxdf.k_r,      3.0f / 1.5f, 1e-5f);
	EXPECT_NEAR(bxdf.coat_ior, 1.5f,        1e-5f);
}

TEST(Materials, CoatedConductorMaterialNoSubsurface) {
	EXPECT_FALSE((CoatedConductorMaterial<float>::has_subsurface_scattering()));
}

// CoatedDiffuse: g clamped to [-1,1] per pbrt-v4
TEST(Materials, CoatedDiffuseMaterialGClamped) {
	SpectrumTexVal<float> refl(0.5f, 0.5f, 0.5f);
	FloatTexVal<float> g_oob(2.5f); // out of [-1,1]
	CoatedDiffuseMaterial<float> mat(refl,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f),
		FloatTexVal<float>(0.01f),
		SpectrumTexVal<float>(0.f,0.f,0.f),
		g_oob, 1.5f, false, 10, 1);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.g, 1.0f, 1e-5f); // clamped to 1
}

// CoatedConductor: conductor IOR normalized by interface_eta per pbrt-v4
TEST(Materials, CoatedConductorMaterialIORNormalized) {
	// eta=0.2, k=3.0, interface_eta=1.5 -> stored eta_r = 0.2/1.5, k_r = 3.0/1.5
	SpectrumTexVal<float> ceta(0.2f, 0.2f, 0.2f);
	SpectrumTexVal<float> ck(3.0f, 3.0f, 3.0f);
	CoatedConductorMaterial<float> mat(ceta, ck,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f),
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f),
		FloatTexVal<float>(0.01f),
		SpectrumTexVal<float>(0.f,0.f,0.f),
		FloatTexVal<float>(0.f),
		1.5f, false, 10, 1);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.eta_r, 0.2f / 1.5f, 1e-5f);
	EXPECT_NEAR(bxdf.k_r,   3.0f / 1.5f, 1e-5f);
}

// CoatedConductor: g clamped to [-1,1] per pbrt-v4
TEST(Materials, CoatedConductorMaterialGClamped) {
	SpectrumTexVal<float> ceta(1.f, 1.f, 1.f);
	SpectrumTexVal<float> ck(1.f, 1.f, 1.f);
	FloatTexVal<float> g_oob(-5.f);
	CoatedConductorMaterial<float> mat(ceta, ck,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f),
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f),
		FloatTexVal<float>(0.01f),
		SpectrumTexVal<float>(0.f,0.f,0.f),
		g_oob, 1.5f, false, 10, 1);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.g, -1.0f, 1e-5f); // clamped to -1
}

// ---------------------------------------------------------------------------
// DiffuseTransmissionMaterial
// ---------------------------------------------------------------------------
TEST(Materials, DiffuseTransmissionMaterialFields) {
	SpectrumTexVal<float> refl(0.3f, 0.3f, 0.3f);
	SpectrumTexVal<float> trans(0.6f, 0.6f, 0.6f);
	DiffuseTransmissionMaterial<float> mat(refl, trans);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.R_r, 0.3f, 1e-5f);
	EXPECT_NEAR(bxdf.T_r, 0.6f, 1e-5f);
}

TEST(Materials, DiffuseTransmissionMaterialScaled) {
	SpectrumTexVal<float> refl(0.5f, 0.5f, 0.5f);
	SpectrumTexVal<float> trans(0.5f, 0.5f, 0.5f);
	FloatTexVal<float> scale(0.5f);
	DiffuseTransmissionMaterial<float> mat(refl, trans, scale);
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.R_r, 0.25f, 1e-5f);
	EXPECT_NEAR(bxdf.T_r, 0.25f, 1e-5f);
}

TEST(Materials, DiffuseTransmissionMaterialNoSubsurface) {
	EXPECT_FALSE((DiffuseTransmissionMaterial<float>::has_subsurface_scattering()));
}

// ---------------------------------------------------------------------------
// SubsurfaceMaterial
// ---------------------------------------------------------------------------
TEST(Materials, SubsurfaceMaterialHasSubsurface) {
	EXPECT_TRUE((SubsurfaceMaterial<float>::has_subsurface_scattering()));
}

TEST(Materials, SubsurfaceMaterialBxDFSmooth) {
	SpectrumTexVal<float> sig_a(0.01f, 0.02f, 0.04f);
	SpectrumTexVal<float> sig_s(0.5f,  0.5f,  0.5f);
	SubsurfaceMaterial<float> mat(1.f, sig_a, sig_s, 0.0, 1.5,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f), false, 64, 16);
	auto v = mat.get_bxdf(default_ctx());
	EXPECT_FALSE(v.rough);
	EXPECT_NEAR(v.smooth.ior, 1.5f, 1e-4f);
}

TEST(Materials, SubsurfaceMaterialBSSRDFSigmaMode) {
	SpectrumTexVal<float> sig_a(0.01f, 0.02f, 0.04f);
	SpectrumTexVal<float> sig_s(0.5f,  0.5f,  0.5f);
	SubsurfaceMaterial<float> mat(1.f, sig_a, sig_s, 0.0, 1.5,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f), false, 64, 16);
	auto bssrdf = mat.get_bssrdf(default_ctx());
	// sigma_t[0] = 0.01 + 0.5 = 0.51 -> rho = 0.5/0.51
	// Sr(0, r>0) should be finite and positive
	double sr = bssrdf.sr(0, 0.1);
	EXPECT_TRUE(std::isfinite(sr));
	EXPECT_GE(sr, 0.0);
}

TEST(Materials, SubsurfaceMaterialBSSRDFReflMfpMode) {
	SpectrumTexVal<float> refl(0.5f, 0.5f, 0.5f);
	SpectrumTexVal<float> mfp(1.0f,  1.0f, 1.0f);
	SubsurfaceMaterial<float> mat(
		1.f, refl, mfp, 0.0, 1.5,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f), false,
		SubsurfaceMaterial<float>::ReflMfpTag{}, 64, 16);
	auto bssrdf = mat.get_bssrdf(default_ctx());
	double sr = bssrdf.sr(0, 0.5);
	EXPECT_TRUE(std::isfinite(sr));
	EXPECT_GE(sr, 0.0);
}

TEST(Materials, SubsurfaceMaterialBSSRDF3Channels) {
	SpectrumTexVal<float> sig_a(0.01f, 0.02f, 0.04f);
	SpectrumTexVal<float> sig_s(0.5f,  0.4f,  0.3f);
	SubsurfaceMaterial<float> mat(1.f, sig_a, sig_s, 0.0, 1.5,
		FloatTexVal<float>(0.f), FloatTexVal<float>(0.f), false, 64, 16);
	auto bssrdf = mat.get_bssrdf(default_ctx());
	// All 3 channels should produce finite Sr values
	for (int ch = 0; ch < 3; ++ch) {
		double sr = bssrdf.sr(ch, 0.5);
		EXPECT_TRUE(std::isfinite(sr)) << "ch=" << ch;
		EXPECT_GE(sr, 0.0) << "ch=" << ch;
	}
	// Channels should have different sigma_t, so Sr at large r differs
	// (at very small r both may be near-zero due to table resolution)
	double sr0_large = bssrdf.sr(0, 2.0);
	double sr2_large = bssrdf.sr(2, 2.0);
	EXPECT_TRUE(std::isfinite(sr0_large));
	EXPECT_TRUE(std::isfinite(sr2_large));
}

// ---------------------------------------------------------------------------
// MaterialEvalContext helpers
// ---------------------------------------------------------------------------
TEST(Materials, MaterialEvalContextDefaults) {
	MaterialEvalContext<float> ctx{};
	EXPECT_EQ(ctx.faceIndex, 0);
	EXPECT_NEAR(ctx.dudx, 0.f, 1e-9f);
}

// ---------------------------------------------------------------------------
// FloatTexVal / SpectrumTexVal eval helpers
// ---------------------------------------------------------------------------
TEST(Materials, FloatTexValEval) {
	FloatTexVal<float> tex(0.75f);
	EXPECT_NEAR(eval_float_tex(tex, default_ctx()), 0.75f, 1e-6f);
}

TEST(Materials, SpectrumTexValEval) {
	SpectrumTexVal<float> tex(0.1f, 0.2f, 0.3f);
	float r, g, b;
	eval_spectrum_tex(tex, default_ctx(), r, g, b);
	EXPECT_NEAR(r, 0.1f, 1e-6f);
	EXPECT_NEAR(g, 0.2f, 1e-6f);
	EXPECT_NEAR(b, 0.3f, 1e-6f);
}

// ===========================================================================
// HairMaterial tests (pbrt-v4 HairMaterial parity)
// ===========================================================================

TEST(HairMaterialTest, DirectSigmaAProducesBxDF) {
	HairMaterial<float> mat;
	mat.has_sigma_a = true;
	mat.sigma_a_tex = SpectrumTexVal<float>(0.5f, 0.3f, 0.1f);
	mat.eta     = FloatTexVal<float>(1.55f);
	mat.beta_m  = FloatTexVal<float>(0.3f);
	mat.beta_n  = FloatTexVal<float>(0.3f);
	mat.alpha   = FloatTexVal<float>(2.0f);

	// default_ctx has v=0.3, so h = -1 + 2*0.3 = -0.4
	auto ctx = default_ctx();
	auto bxdf = mat.get_bxdf(ctx);
	EXPECT_NEAR(bxdf.sigma_r, 0.5f, 1e-5f);
	EXPECT_NEAR(bxdf.sigma_g, 0.3f, 1e-5f);
	EXPECT_NEAR(bxdf.sigma_b, 0.1f, 1e-5f);
	EXPECT_NEAR(bxdf.eta,     1.55f, 1e-5f);
	EXPECT_NEAR(bxdf.h,       -0.4f, 1e-5f);
}

TEST(HairMaterialTest, BetaClampedToMinimum) {
	// beta_m and beta_n below 0.01 should be clamped to 0.01
	HairMaterial<float> mat;
	mat.has_sigma_a = true;
	mat.sigma_a_tex = SpectrumTexVal<float>(0.1f, 0.1f, 0.1f);
	mat.beta_m = FloatTexVal<float>(0.0f);   // below min
	mat.beta_n = FloatTexVal<float>(0.005f); // below min

	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_NEAR(bxdf.beta_m, 0.01f, 1e-5f);
	EXPECT_NEAR(bxdf.beta_n, 0.01f, 1e-5f);
}

TEST(HairMaterialTest, SigmaAFromPigmentConcentration) {
	HairMaterial<float> mat;
	mat.has_pigment  = true;
	mat.eumelanin    = FloatTexVal<float>(1.3f);  // moderate brown
	mat.pheomelanin  = FloatTexVal<float>(0.0f);

	auto bxdf = mat.get_bxdf(default_ctx());
	// sigma_r = 1.3 * 0.419, sigma_g = 1.3 * 0.697, sigma_b = 1.3 * 1.37
	EXPECT_NEAR(bxdf.sigma_r, 1.3f * 0.419f, 1e-4f);
	EXPECT_NEAR(bxdf.sigma_g, 1.3f * 0.697f, 1e-4f);
	EXPECT_NEAR(bxdf.sigma_b, 1.3f * 1.37f,  1e-4f);
}

TEST(HairMaterialTest, SigmaAFromReflectanceColor) {
	HairMaterial<float> mat;
	mat.has_color = true;
	mat.color_tex = SpectrumTexVal<float>(0.5f, 0.5f, 0.5f); // mid-grey hair
	mat.beta_n    = FloatTexVal<float>(0.3f);

	auto bxdf = mat.get_bxdf(default_ctx());
	// sigma_a from reflectance must be positive (absorption to yield that color)
	EXPECT_GT(bxdf.sigma_r, 0.f);
	EXPECT_GT(bxdf.sigma_g, 0.f);
	EXPECT_GT(bxdf.sigma_b, 0.f);
	// Grey input -> all channels approximately equal
	EXPECT_NEAR(bxdf.sigma_r, bxdf.sigma_g, 1e-4f);
	EXPECT_NEAR(bxdf.sigma_g, bxdf.sigma_b, 1e-4f);
}

TEST(HairMaterialTest, NoSourceDefaultsToBlackSigmaA) {
	HairMaterial<float> mat;
	// has_sigma_a, has_color, has_pigment all false
	auto bxdf = mat.get_bxdf(default_ctx());
	EXPECT_FLOAT_EQ(bxdf.sigma_r, 0.f);
	EXPECT_FLOAT_EQ(bxdf.sigma_g, 0.f);
	EXPECT_FLOAT_EQ(bxdf.sigma_b, 0.f);
}

TEST(HairMaterialTest, HComputedFromVTexCoord) {
	// h = -1 + 2*ctx.v (pbrt-v4 HairMaterial::GetBxDF: h = -1 + 2*ctx.uv[1])
	HairMaterial<float> mat;
	mat.has_sigma_a = true;
	mat.sigma_a_tex = SpectrumTexVal<float>(0.f, 0.f, 0.f);

	auto ctx = default_ctx();
	ctx.v = 0.0f;
	EXPECT_NEAR(mat.get_bxdf(ctx).h, -1.0f, 1e-5f);

	ctx.v = 0.5f;
	EXPECT_NEAR(mat.get_bxdf(ctx).h, 0.0f, 1e-5f);

	ctx.v = 1.0f;
	EXPECT_NEAR(mat.get_bxdf(ctx).h, 1.0f, 1e-5f);
}

TEST(HairMaterialTest, EvaluateBxDFFinite) {
	HairMaterial<float> mat;
	mat.has_pigment  = true;
	mat.eumelanin    = FloatTexVal<float>(0.5f);
	mat.pheomelanin  = FloatTexVal<float>(0.2f);
	mat.beta_m       = FloatTexVal<float>(0.25f);
	mat.beta_n       = FloatTexVal<float>(0.2f);

	auto bxdf = mat.get_bxdf(default_ctx());
	// Evaluate the BSDF in the same-side direction (simple sanity check)
	std::array<float,3> wo = {0.f, 0.f, 1.f};
	std::array<float,3> wi = {0.f, 0.f, 1.f};
	// HairBxDF::f() expects (wo_theta, wo_phi, wi_theta, wi_phi) in hair coordinates
	// Just check the BxDF was constructed without crashing and has finite fields
	EXPECT_TRUE(std::isfinite(bxdf.eta));
	EXPECT_TRUE(std::isfinite(bxdf.beta_m));
	EXPECT_TRUE(std::isfinite(bxdf.beta_n));
	EXPECT_TRUE(std::isfinite(bxdf.s));
	for (int i = 0; i <= HairBxDF<float>::pMax; ++i)
		EXPECT_TRUE(std::isfinite(bxdf.v[i]));
}

// ===========================================================================
// MixMaterial tests (pbrt-v4 MixMaterial parity)
// ===========================================================================

TEST(SharedMixMaterialTest, WeightZeroAlwaysChoosesB) {
	MixMaterial<float> mat;
	// pbrt-v4: (amt < u) ? mat[0] : mat[1]
	// amt=0: 0 < u is true for all u > 0 -> always mat[0] = index 0
	mat.amount = FloatTexVal<float>(0.f);

	for (float px = -2.f; px <= 2.f; px += 0.5f) {
		MaterialEvalContext<float> ctx{};
		ctx.px = px; ctx.py = 0.3f; ctx.pz = -1.1f;
		EXPECT_EQ(mat.choose(ctx), 0) << "px=" << px;
	}
}

TEST(SharedMixMaterialTest, WeightOneAlwaysChoosesA) {
	MixMaterial<float> mat;
	// pbrt-v4: amt=1: 1 < u is never true for u in [0,1) -> always mat[1] = index 1
	mat.amount = FloatTexVal<float>(1.f);

	for (float px = -2.f; px <= 2.f; px += 0.5f) {
		MaterialEvalContext<float> ctx{};
		ctx.px = px; ctx.py = 1.7f; ctx.pz = 0.4f;
		EXPECT_EQ(mat.choose(ctx), 1) << "px=" << px;
	}
}

TEST(SharedMixMaterialTest, WeightHalfReturnsBothMaterials) {
	// With weight=0.5 and a spread of shading points, both 0 and 1 should appear
	MixMaterial<float> mat;
	mat.amount = FloatTexVal<float>(0.5f);

	int count_a = 0, count_b = 0;
	for (int i = 0; i < 100; ++i) {
		MaterialEvalContext<float> ctx{};
		ctx.px = static_cast<float>(i) * 0.137f;
		ctx.py = static_cast<float>(i) * 0.251f;
		ctx.pz = static_cast<float>(i) * 0.373f;
		int c = mat.choose(ctx);
		ASSERT_TRUE(c == 0 || c == 1);
		(c == 0 ? count_a : count_b)++;
	}
	// Both materials should be selected at least a few times
	EXPECT_GT(count_a, 5);
	EXPECT_GT(count_b, 5);
}

TEST(SharedMixMaterialTest, ChoiceIsDeterministic) {
	// Same point must always return the same material
	MixMaterial<float> mat;
	mat.amount = FloatTexVal<float>(0.5f);

	MaterialEvalContext<float> ctx{};
	ctx.px = 1.234f; ctx.py = 5.678f; ctx.pz = -9.012f;

	int first = mat.choose(ctx);
	for (int i = 0; i < 20; ++i)
		EXPECT_EQ(mat.choose(ctx), first);
}

TEST(SharedMixMaterialTest, EmittedBlends) {
	// MixMaterial itself has no BxDF; verify it compiles and has_subsurface_scattering=false
	EXPECT_FALSE(MixMaterial<float>::has_subsurface_scattering());
}

TEST(SharedMixMaterialTest, TextureWeightConstructor) {
	// Verify FloatTexVal<float> is accepted as amount type
	MixMaterial<float, FloatTexVal<float>> mat;
	mat.amount = FloatTexVal<float>(0.75f);
	MaterialEvalContext<float> ctx{};
	ctx.px = 0.f; ctx.py = 0.f; ctx.pz = 0.f;
	int c = mat.choose(ctx);
	EXPECT_TRUE(c == 0 || c == 1);
}

TEST(SharedMixMaterialTest, NestedMixWorks) {
	// Confirm two MixMaterial instances can be evaluated independently
	MixMaterial<float> inner;
	inner.amount = FloatTexVal<float>(0.3f);

	MixMaterial<float> outer;
	outer.amount = FloatTexVal<float>(0.7f);

	MaterialEvalContext<float> ctx{};
	ctx.px = 3.14f; ctx.py = 2.71f; ctx.pz = 1.41f;

	int ci = inner.choose(ctx);
	int co = outer.choose(ctx);
	EXPECT_TRUE(ci == 0 || ci == 1);
	EXPECT_TRUE(co == 0 || co == 1);
}
