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
// diffuse_transmission -- dedicated closed-form bridge (multi-lobe material)
// ============================================================================

TEST(SppmBsdfF, DiffuseTransmissionReflectionHemisphereMatchesR) {
	auto mat = make_shared<diffuse_transmission>(color(0.6, 0.3, 0.1), color(0.1, 0.2, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	double wi[3] = { 0, 1, 0 };   // same hemisphere as n -> reflection lobe
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	EXPECT_NEAR(out[0], 0.6 / pi, 1e-9);
	EXPECT_NEAR(out[1], 0.3 / pi, 1e-9);
	EXPECT_NEAR(out[2], 0.1 / pi, 1e-9);
}

TEST(SppmBsdfF, DiffuseTransmissionTransmissionHemisphereMatchesT) {
	auto mat = make_shared<diffuse_transmission>(color(0.6, 0.3, 0.1), color(0.1, 0.2, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	double wi[3] = { 0, -1, 0 };   // opposite hemisphere from n -> transmission lobe
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	EXPECT_NEAR(out[0], 0.1 / pi, 1e-9);
	EXPECT_NEAR(out[1], 0.2 / pi, 1e-9);
	EXPECT_NEAR(out[2], 0.5 / pi, 1e-9);
}

TEST(SppmBsdfSampleF, DiffuseTransmissionEnergyConservationConvergesToRPlusT) {
	// Grayscale R/T (equal across channels) so pr=R, pt=T as scalars and the
	// expected value of f*|cosI|/pdf simplifies to exactly R+T on every
	// channel (see sppm_adapter.h's diffuse_transmission special-case
	// comment for the derivation) -- a clean, closed-form MC target.
	double R = 0.5, T = 0.2;
	auto mat = make_shared<diffuse_transmission>(color(R, R, R), color(T, T, T));
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
		EXPECT_NEAR(mean, R + T, 0.02) << "channel " << c;
	}
}

TEST(SppmBsdfSampleF, DiffuseTransmissionSamplesBothHemispheres) {
	// Sanity that both lobes are actually reachable (not just one) - a
	// silent regression that always picked one hemisphere would still pass
	// the energy-conservation test above by coincidence if R==T, so this
	// checks the mechanism directly.
	auto mat = make_shared<diffuse_transmission>(color(0.5, 0.5, 0.5), color(0.5, 0.5, 0.5));
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	bool saw_reflection = false, saw_transmission = false;
	for (int i = 0; i < 500; ++i) {
		double new_dir[3], f_val[3], pdf;
		bool is_specular;
		ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
		double cosI = new_dir[0]*n[0] + new_dir[1]*n[1] + new_dir[2]*n[2];
		if (cosI > 0) saw_reflection = true; else saw_transmission = true;
	}
	EXPECT_TRUE(saw_reflection);
	EXPECT_TRUE(saw_transmission);
}

// ============================================================================
// normalized_fresnel -- confirms the EXISTING generic bridge already
// handles it correctly (its scatter() sets attenuation=(1,1,1)
// unconditionally, so it never needed a special case)
// ============================================================================

TEST(SppmBsdfF, NormalizedFresnelMatchesClosedForm) {
	auto mat = make_shared<normalized_fresnel>(1.5);
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };
	double wi[3] = { 0, 1, 0 };   // straight up: cos_wi = 1
	double out[3];
	sppm_bsdf_f(ctx, wo, wi, n, out);

	double fr = FrDielectric(1.0, 1.5);
	double expected = (1.0 - fr) / (mat->get_c() * pi);
	EXPECT_NEAR(out[0], expected, 1e-9);
	EXPECT_NEAR(out[1], expected, 1e-9);
	EXPECT_NEAR(out[2], expected, 1e-9);
}

TEST(SppmBsdfSampleF, NormalizedFresnelEnergyConservationConvergesToOne) {
	// (1-Fr)/c integrated over the cosine-weighted hemisphere converges to
	// exactly 1.0 by construction: c = 1 - 2*FresnelMoment1(1/eta) is
	// specifically the normalization constant that makes this BSDF conserve
	// all incoming energy (it's used at BSSRDF exit boundaries, where that
	// normalization is the whole point) - NOT "less than 1 like a lossy
	// reflectance" (an earlier version of this test wrongly assumed that
	// and failed at mean=1.0008, well within MC noise of the true 1.0).
	auto mat = make_shared<normalized_fresnel>(1.5);
	SPPMShadingContext ctx;
	ctx.p = point3(0, 0, 0);
	ctx.normal = vec3(0, 1, 0);
	ctx.mat = mat;

	double n[3] = { 0, 1, 0 };
	double wo[3] = { 0, 1, 0 };

	const int N = 5000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double new_dir[3], f_val[3], pdf;
		bool is_specular;
		ASSERT_TRUE(sppm_bsdf_sample_f(ctx, wo, n, 0.0, 0.0, new_dir, f_val, pdf, is_specular));
		EXPECT_FALSE(is_specular);
		ASSERT_GT(pdf, 0.0);
		double cosI = std::fabs(new_dir[0]*n[0] + new_dir[1]*n[1] + new_dir[2]*n[2]);
		double v = f_val[0] * cosI / pdf;
		EXPECT_TRUE(std::isfinite(v));
		sum += v;
	}
	double mean = sum / N;
	EXPECT_NEAR(mean, 1.0, 0.02);
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

// ============================================================================
// sppm_resolve_material (mix_material support)
// ============================================================================
// mix_material never reaches sppm_is_delta_material/sppm_bsdf_f/
// sppm_bsdf_sample_f directly -- SPPMSceneAdapter::Intersect() resolves it
// down to a concrete sub-material first via sppm_resolve_material(), which
// is what these tests target.

TEST(SppmResolveMaterial, NonMixMaterialPassesThroughUnchanged) {
	auto lam = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto resolved = sppm_resolve_material(lam, 0.0, 0.0, point3(0, 0, 0));
	EXPECT_EQ(resolved.get(), lam.get());
}

// weight=0.8 means mat_b is chosen with probability 0.8 (mix_material::
// scatter(): `random_double() >= w ? mat_a : mat_b`, so P(mat_b) = w) --
// mirrors that exact draw, so the resolved distribution should match.
TEST(SppmResolveMaterial, ResolvesToEachSubMaterialAccordingToWeight) {
	auto mat_a = make_shared<lambertian>(color(0.1, 0.1, 0.1));
	auto mat_b = make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
	auto mix = make_shared<mix_material>(mat_a, mat_b, 0.8);

	int a_count = 0, b_count = 0;
	const int N = 2000;
	for (int i = 0; i < N; ++i) {
		auto resolved = sppm_resolve_material(mix, 0.0, 0.0, point3(0, 0, 0));
		ASSERT_FALSE(std::dynamic_pointer_cast<mix_material>(resolved))
			<< "resolved material must never itself be a mix_material";
		if (resolved.get() == mat_a.get()) ++a_count;
		else if (resolved.get() == mat_b.get()) ++b_count;
		else FAIL() << "resolved to neither sub-material";
	}
	EXPECT_NEAR(static_cast<double>(b_count) / N, 0.8, 0.03);
	EXPECT_NEAR(static_cast<double>(a_count) / N, 0.2, 0.03);
}

// A mix nested inside a mix (pbrt-v4 allows arbitrary nesting) must resolve
// all the way down to a real, concrete leaf material, never stopping at an
// inner mix_material.
TEST(SppmResolveMaterial, ResolvesNestedMixToConcreteLeaf) {
	auto leaf_a = make_shared<lambertian>(color(0.1, 0.1, 0.1));
	auto leaf_b = make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
	auto leaf_c = make_shared<dielectric>(1.5);
	auto inner = make_shared<mix_material>(leaf_a, leaf_b, 0.5);
	auto outer = make_shared<mix_material>(inner, leaf_c, 0.5);

	for (int i = 0; i < 200; ++i) {
		auto resolved = sppm_resolve_material(outer, 0.0, 0.0, point3(0, 0, 0));
		ASSERT_FALSE(std::dynamic_pointer_cast<mix_material>(resolved));
		bool is_known_leaf = resolved.get() == leaf_a.get() ||
		                      resolved.get() == leaf_b.get() ||
		                      resolved.get() == leaf_c.get();
		EXPECT_TRUE(is_known_leaf);
	}
}

// The whole point of resolving before classification: sppm_is_delta_material
// must see the winning sub-material's real type, not "mix_material" (which
// isn't one of the checked delta classes and would otherwise always report
// false, silently mistreating a delta-lobe draw as diffuse).
TEST(SppmResolveMaterial, ResolvedDeltaSubMaterialIsClassifiedAsDelta) {
	auto lam = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto met = make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
	auto mix = make_shared<mix_material>(lam, met, 1.0);   // weight=1.0 -> always mat_b (metal)

	auto resolved = sppm_resolve_material(mix, 0.0, 0.0, point3(0, 0, 0));
	EXPECT_EQ(resolved.get(), met.get());
	EXPECT_TRUE(sppm_is_delta_material(resolved.get()));
}

TEST(SppmResolveMaterial, ResolvedNonDeltaSubMaterialIsClassifiedAsNonDelta) {
	auto lam = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto met = make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
	auto mix = make_shared<mix_material>(lam, met, 0.0);   // weight=0.0 -> always mat_a (lambertian)

	auto resolved = sppm_resolve_material(mix, 0.0, 0.0, point3(0, 0, 0));
	EXPECT_EQ(resolved.get(), lam.get());
	EXPECT_FALSE(sppm_is_delta_material(resolved.get()));
}
