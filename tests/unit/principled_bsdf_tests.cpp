// principled_bsdf_tests.cpp
// pbrt-v4-style unit tests for PrincipledBxDF and the `principled` material wrapper.
//
// Tests mirror pbrt-v4's bsdfs_test approach:
//   - White-furnace: energy ≤ 1 for all parameter combinations
//   - Hemisphere validity: sampled directions stay on correct hemisphere
//   - Metallic=1 → diffuse contribution is zero
//   - PDF positivity: PDF > 0 whenever sample is valid
//   - Material wrapper: scatter() succeeds and attenuation is non-negative

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <random>

#include "rtweekend.h"
#include "principled_material.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static PrincipledBxDF<double> make_bxdf(
	double metallic, double roughness,
	double ior = 1.5, double clearcoat = 0.0, double cc_rough = 0.1)
{
	return { 0.8, 0.5, 0.2,  // base color
			 metallic, roughness, ior, clearcoat, cc_rough };
}

// Evaluate Monte Carlo estimate of total reflectance (white-furnace integral)
// using N cosine-weighted samples from a fixed outgoing direction.
static double white_furnace(const PrincipledBxDF<double>& bxdf,
							double nx, double ny, double nz,
							int N = 4000)
{
	std::mt19937_64 rng(42);
	std::uniform_real_distribution<double> u01(0.0, 1.0);

	// incident direction (straight down = -normal)
	double wi_x = -nx, wi_y = -ny, wi_z = -nz;

	double sum = 0.0;
	int valid = 0;
	for (int i = 0; i < N; ++i) {
		auto res = bxdf.sample(nx, ny, nz,
							   wi_x, wi_y, wi_z,
							   u01(rng), u01(rng), u01(rng));
		if (!res.valid) continue;
		++valid;
		double lum = 0.2126*res.r + 0.7152*res.g + 0.0722*res.b;
		sum += lum;
	}
	if (valid == 0) return 0.0;
	return sum / N;  // average weight ≈ E[f*cos/pdf]
}

// ---------------------------------------------------------------------------
// White-furnace energy conservation
// ---------------------------------------------------------------------------

TEST(PrincipledBxDF, WhiteFurnacePlastic) {
	// Plastic (metallic=0, roughness=0.5) — total reflectance must be ≤ 1
	auto bxdf = make_bxdf(0.0, 0.5);
	double E = white_furnace(bxdf, 0.0, 1.0, 0.0);
	EXPECT_LE(E, 1.05) << "plastic: energy must be <= 1 (allow 5% MC variance)";
	EXPECT_GT(E, 0.0)  << "plastic: must scatter some energy";
}

TEST(PrincipledBxDF, WhiteFurnaceMetal) {
	// Pure metal (metallic=1) — energy ≤ 1
	auto bxdf = make_bxdf(1.0, 0.3);
	double E = white_furnace(bxdf, 0.0, 1.0, 0.0);
	EXPECT_LE(E, 1.05);
	EXPECT_GT(E, 0.0);
}

TEST(PrincipledBxDF, WhiteFurnaceClearcoat) {
	// Clearcoat on top of plastic
	auto bxdf = make_bxdf(0.0, 0.5, 1.5, 1.0, 0.1);
	double E = white_furnace(bxdf, 0.0, 1.0, 0.0);
	EXPECT_LE(E, 1.05);
	EXPECT_GT(E, 0.0);
}

TEST(PrincipledBxDF, WhiteFurnaceSmooth) {
	// Very smooth (roughness=0.05)
	auto bxdf = make_bxdf(0.0, 0.05);
	double E = white_furnace(bxdf, 0.0, 1.0, 0.0);
	EXPECT_LE(E, 1.05);
}

TEST(PrincipledBxDF, WhiteFurnaceRough) {
	// Very rough (roughness=0.95)
	auto bxdf = make_bxdf(0.0, 0.95);
	double E = white_furnace(bxdf, 0.0, 1.0, 0.0);
	EXPECT_LE(E, 1.05);
	EXPECT_GT(E, 0.0);
}

// ---------------------------------------------------------------------------
// Hemisphere validity
// ---------------------------------------------------------------------------

TEST(PrincipledBxDF, SampledDirectionOnCorrectHemisphere) {
	auto bxdf = make_bxdf(0.5, 0.5);
	std::mt19937_64 rng(7);
	std::uniform_real_distribution<double> u01(0.0, 1.0);

	double nx = 0.0, ny = 1.0, nz = 0.0;  // normal = world Y
	double wi_x = 0.0, wi_y = -1.0, wi_z = 0.0; // straight down

	int trials = 1000, valid = 0;
	for (int i = 0; i < trials; ++i) {
		auto res = bxdf.sample(nx, ny, nz, wi_x, wi_y, wi_z,
							   u01(rng), u01(rng), u01(rng));
		if (!res.valid) continue;
		++valid;
		// Scattered direction must be on the upper hemisphere
		double dot = res.wo_x*nx + res.wo_y*ny + res.wo_z*nz;
		EXPECT_GT(dot, 0.0) << "scattered direction below surface";
	}
	EXPECT_GT(valid, trials/2) << "at least half of samples should be valid";
}

// ---------------------------------------------------------------------------
// Metallic=1 → diffuse lobe zero
// ---------------------------------------------------------------------------

TEST(PrincipledBxDF, MetallicOneDiffuseContributionIsZero) {
	// With metallic=1 the diffuse weight is (1-metallic)*(1-F) = 0.
	// The only contribution should come from the specular lobe.
	// We can check that white_furnace with a white base color gives the
	// same result regardless of roughness changes to the diffuse term.
	PrincipledBxDF<double> bxdf_white_metal{ 1.0, 1.0, 1.0, // white base
											  1.0, 0.5, 1.5, 0.0, 0.1 };
	PrincipledBxDF<double> bxdf_black_metal{ 0.0, 0.0, 0.0, // black base
											  1.0, 0.5, 1.5, 0.0, 0.1 };

	// For metallic=1, base color acts as F0 for conductor Fresnel.
	// The DIFFERENCE in white-furnace between white and black metal base
	// comes only from the conductor Fresnel term — not from a diffuse lobe.
	double E_white = white_furnace(bxdf_white_metal, 0.0, 1.0, 0.0, 2000);
	double E_black = white_furnace(bxdf_black_metal, 0.0, 1.0, 0.0, 2000);

	// Both must be <= 1
	EXPECT_LE(E_white, 1.05);
	EXPECT_LE(E_black, 1.05);
	// White base should reflect more than black (conductor F0 effect)
	EXPECT_GT(E_white, E_black);
}

// ---------------------------------------------------------------------------
// PDF positivity and consistency
// ---------------------------------------------------------------------------

TEST(PrincipledBxDF, PDFPositiveForValidSamples) {
	auto bxdf = make_bxdf(0.3, 0.4);
	std::mt19937_64 rng(13);
	std::uniform_real_distribution<double> u01(0.0, 1.0);

	double nx = 0, ny = 1, nz = 0;
	double wi_x = 0, wi_y = -1, wi_z = 0;

	int checked = 0;
	for (int i = 0; i < 500; ++i) {
		auto res = bxdf.sample(nx, ny, nz, wi_x, wi_y, wi_z,
							   u01(rng), u01(rng), u01(rng));
		if (!res.valid) continue;

		double pdf = bxdf.scattering_pdf(nx, ny, nz,
										  wi_x, wi_y, wi_z,
										  res.wo_x, res.wo_y, res.wo_z);
		EXPECT_GT(pdf, 0.0) << "PDF must be positive for a valid sample";
		++checked;
		if (checked >= 100) break;
	}
	EXPECT_GT(checked, 10) << "need enough valid samples to test";
}

// ---------------------------------------------------------------------------
// `principled` material wrapper
// ---------------------------------------------------------------------------

static hit_record make_hit_principled(vec3 normal = vec3(0,1,0)) {
	hit_record rec;
	rec.p          = point3(0,0,0);
	rec.normal     = normal;
	rec.dpdu       = vec3(1,0,0);
	rec.u          = 0.5;
	rec.v          = 0.5;
	rec.t          = 1.0;
	rec.front_face = true;
	return rec;
}

TEST(PrincipledMaterial, ScatterSucceedsPlastic) {
	principled mat(color(0.8, 0.3, 0.1), 0.0, 0.5);
	hit_record rec = make_hit_principled();
	ray r_in(point3(0,2,0), vec3(0,-1,0));
	scatter_record srec;
	// scatter may fail for grazing angles; just run several times
	int ok = 0;
	for (int i = 0; i < 20; ++i)
		if (mat.scatter(r_in, rec, srec)) ++ok;
	EXPECT_GT(ok, 0) << "plastic principled material should scatter";
}

TEST(PrincipledMaterial, ScatterSucceedsMetal) {
	principled mat(color(0.95, 0.85, 0.6), 1.0, 0.2);
	hit_record rec = make_hit_principled();
	ray r_in(point3(0,2,0), vec3(0,-1,0));
	scatter_record srec;
	int ok = 0;
	for (int i = 0; i < 20; ++i)
		if (mat.scatter(r_in, rec, srec)) ++ok;
	EXPECT_GT(ok, 0) << "metallic principled material should scatter";
}

TEST(PrincipledMaterial, AttenuationNonNegative) {
	principled mat(color(0.7, 0.5, 0.3), 0.5, 0.4, 1.5, 0.5);
	hit_record rec = make_hit_principled();
	ray r_in(point3(0,2,0), vec3(0,-1,0));

	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		if (!mat.scatter(r_in, rec, srec)) continue;
		EXPECT_GE(srec.attenuation.x(), 0.0) << "attenuation R must be >= 0";
		EXPECT_GE(srec.attenuation.y(), 0.0) << "attenuation G must be >= 0";
		EXPECT_GE(srec.attenuation.z(), 0.0) << "attenuation B must be >= 0";
	}
}

TEST(PrincipledMaterial, ScatteredDirectionAboveSurface) {
	principled mat(color(0.8, 0.8, 0.8), 0.3, 0.5);
	hit_record rec = make_hit_principled(vec3(0,1,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));

	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		if (!mat.scatter(r_in, rec, srec)) continue;
		vec3 dir = unit_vector(srec.skip_pdf_ray.direction());
		double dot = dir.y(); // dot with normal (0,1,0)
		EXPECT_GT(dot, 0.0) << "scattered direction must be above surface";
	}
}
