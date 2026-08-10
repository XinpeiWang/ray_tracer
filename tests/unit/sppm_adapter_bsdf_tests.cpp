/**
 * @file sppm_adapter_bsdf_tests.cpp
 * @brief Unit tests for src/TheRestOfYourLife/sppm_adapter.h's BSDF bridge
 *
 * Tests sppm_is_delta_material, sppm_bsdf_f, sppm_bsdf_sample_f in
 * isolation, constructing SPPMShadingContext directly by hand -- no
 * hittable/world/Intersect() needed. This is the algorithmic crux of the
 * whole SPPM integration (bridging src/shared/sppm.h's duck-typed
 * BSDFf/BSDFSampleF interface to this codebase's real material::scatter()/
 * scattering_pdf()), so it gets its own dedicated, closed-form-verified
 * test file before any scene-level integration is attempted.
 *
 * Covers:
 *  - sppm_is_delta_material: correct classification for delta vs
 *    non-delta material classes.
 *  - sppm_bsdf_f: closed-form Lambertian f(wo,wi) = albedo/pi in-hemisphere,
 *    0 outside it.
 *  - sppm_bsdf_sample_f (non-specular): Monte-Carlo energy conservation --
 *    mean of f_val*cosI/pdf over many draws converges to albedo.
 *  - sppm_bsdf_sample_f (delta/specular): f_val*cosI == attenuation exactly
 *    and pdf==1.0 -- the test that catches a backwards cosI division.
 *  - rough_dielectric (scene 11's actual sphere material) sanity: reported
 *    as specular, produces valid finite unit-length directions.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable.h"
#include "material.h"
#include "sppm_adapter.h"
#include <cmath>
#include <random>

// ============================================================================
// sppm_is_delta_material
// ============================================================================

TEST(SppmIsDeltaMaterial, LambertianIsNotDelta) {
	lambertian m(color(0.5, 0.5, 0.5));
	EXPECT_FALSE(sppm_is_delta_material(&m));
}

TEST(SppmIsDeltaMaterial, MetalIsDelta) {
	metal m(color(0.8, 0.8, 0.8), 0.0);
	EXPECT_TRUE(sppm_is_delta_material(&m));
}

TEST(SppmIsDeltaMaterial, DielectricIsDelta) {
	dielectric m(1.5);
	EXPECT_TRUE(sppm_is_delta_material(&m));
}

TEST(SppmIsDeltaMaterial, DiffuseLightIsNotDelta) {
	diffuse_light m(color(4, 4, 4));
	EXPECT_FALSE(sppm_is_delta_material(&m));
}

// ============================================================================
// sppm_bsdf_f -- closed-form Lambertian check
// ============================================================================

TEST(SppmBsdfF, LambertianMatchesClosedForm) {
	auto mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.u = 0.5; ctx.v = 0.5;
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };   // arbitrary -- lambertian f is direction-of-wo-independent
	double wi[3] = { 0, 1, 0 };   // straight up, well within hemisphere
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	double expected = 0.5 / pi;
	EXPECT_NEAR(out[0], expected, 1e-9);
	EXPECT_NEAR(out[1], expected, 1e-9);
	EXPECT_NEAR(out[2], expected, 1e-9);
}

TEST(SppmBsdfF, LambertianZeroBelowHemisphere) {
	auto mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	double wi[3] = { 0, -1, 0 };   // below the surface
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	EXPECT_EQ(out[0], 0.0);
	EXPECT_EQ(out[1], 0.0);
	EXPECT_EQ(out[2], 0.0);
}

TEST(SppmBsdfF, ColoredAlbedoPerChannel) {
	auto mat = make_shared<lambertian>(color(0.9, 0.2, 0.4));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	double wi[3] = { 0, 1, 0 };
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	EXPECT_NEAR(out[0], 0.9 / pi, 1e-9);
	EXPECT_NEAR(out[1], 0.2 / pi, 1e-9);
	EXPECT_NEAR(out[2], 0.4 / pi, 1e-9);
}

// ============================================================================
// sppm_bsdf_sample_f -- non-specular Monte-Carlo energy conservation
// ============================================================================

TEST(SppmBsdfSampleF, LambertianEnergyConservationConvergesToAlbedo) {
	auto mat = make_shared<lambertian>(color(0.6, 0.6, 0.6));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };

	const int N = 20000;
	double sum[3] = { 0, 0, 0 };
	for (int i = 0; i < N; ++i) {
		double new_dir[3], f_val[3], pdf;
		bool is_specular;
		ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
		EXPECT_FALSE(is_specular);
		ASSERT_GT(pdf, 0.0);
		double cosI = std::fabs(new_dir[0]*n[0] + new_dir[1]*n[1] + new_dir[2]*n[2]);
		for (int c = 0; c < 3; ++c) sum[c] += f_val[c] * cosI / pdf;
	}
	for (int c = 0; c < 3; ++c) {
		double mean = sum[c] / N;
		EXPECT_NEAR(mean, 0.6, 0.02) << "channel " << c;   // MC error tolerance
	}
}

TEST(SppmBsdfSampleF, LambertianDirectionsStayInUpperHemisphere) {
	auto mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	for (int i = 0; i < 500; ++i) {
		double new_dir[3], f_val[3], pdf;
		bool is_specular;
		ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
		double cosI = new_dir[0]*n[0] + new_dir[1]*n[1] + new_dir[2]*n[2];
		EXPECT_GE(cosI, -1e-9);
	}
}

// ============================================================================
// sppm_bsdf_sample_f -- delta/specular regression test
// ============================================================================

TEST(SppmBsdfSampleF, MirrorMetalCollapsesToAttenuationExactly) {
	// fuzz=0 -> perfect mirror, deterministic reflection direction
	auto mat = make_shared<metal>(color(0.9, 0.7, 0.3), 0.0);
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	// wo pointing up-and-to-the-side (not straight up, so cosI != 1 -- a
	// more discriminating check for the cosI-division-direction bug than a
	// straight-up wo would be, since cosI=1 would hide a backwards division).
	vec3 wo_v = unit_vector(vec3(0.6, 0.8, 0.0));
	double wo[3] = { wo_v.x(), wo_v.y(), wo_v.z() };

	double new_dir[3], f_val[3], pdf;
	bool is_specular;
	ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
	EXPECT_TRUE(is_specular);
	EXPECT_DOUBLE_EQ(pdf, 1.0);

	double cosI = std::fabs(new_dir[0]*n[0] + new_dir[1]*n[1] + new_dir[2]*n[2]);
	EXPECT_NEAR(f_val[0] * cosI, 0.9, 1e-9);
	EXPECT_NEAR(f_val[1] * cosI, 0.7, 1e-9);
	EXPECT_NEAR(f_val[2] * cosI, 0.3, 1e-9);
}

TEST(SppmBsdfSampleF, MirrorMetalReflectsAcrossNormal) {
	auto mat = make_shared<metal>(color(1, 1, 1), 0.0);
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	// wo = incoming reflected back at 45 degrees -- reflected new_dir should
	// mirror it across the normal (same x/z magnitude, y matches wo's y).
	vec3 wo_v = unit_vector(vec3(0.7071, 0.7071, 0.0));
	double wo[3] = { wo_v.x(), wo_v.y(), wo_v.z() };

	double new_dir[3], f_val[3], pdf;
	bool is_specular;
	ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
	EXPECT_NEAR(new_dir[0], -wo_v.x(), 1e-6);
	EXPECT_NEAR(new_dir[1], wo_v.y(), 1e-6);
	EXPECT_NEAR(new_dir[2], wo_v.z(), 1e-6);
}

// ============================================================================
// rough_dielectric (scene 11's actual sphere material) sanity
// ============================================================================

TEST(SppmBsdfSampleF, RoughDielectricReportsSpecularAndValidDirections) {
	auto mat = make_shared<rough_dielectric>(1.5, 0.2);   // matches scene 11's actual sphere material
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };

	for (int i = 0; i < 200; ++i) {
		double new_dir[3], f_val[3], pdf;
		bool is_specular;
		bool ok = sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular);
		if (!ok) continue;   // TIR / degenerate sample is acceptable
		EXPECT_TRUE(is_specular) << "rough_dielectric must be treated as delta by this bridge";
		EXPECT_DOUBLE_EQ(pdf, 1.0);
		double len = std::sqrt(new_dir[0]*new_dir[0] + new_dir[1]*new_dir[1] + new_dir[2]*new_dir[2]);
		EXPECT_NEAR(len, 1.0, 1e-6);
		for (int c = 0; c < 3; ++c) EXPECT_TRUE(std::isfinite(f_val[c]));
	}
}
