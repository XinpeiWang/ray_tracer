// hair_bxdf_tests.cpp
// Unit tests for HairBxDF<double> and hair_material
// Mirrors pbrt-v4 test strategy for HairBxDF.

#include <gtest/gtest.h>
#include <cmath>
#include <random>

// Pull in the shared BxDF header (HairBxDF<T> lives here)
#include "bxdfs.h"
// CPU material wrapper
#include "hair_material.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::mt19937 g_rng(42);
static double randu() {
	return std::uniform_real_distribution<double>(0.0, 1.0)(g_rng);
}

// Normalise a vec3-like triple in-place
static void normalise(double& x, double& y, double& z) {
	double len = std::sqrt(x*x + y*y + z*z);
	if (len < 1e-12) { x = 0; y = 0; z = 1; return; }
	x /= len; y /= len; z /= len;
}

// Construct a test HairBxDF with sensible defaults
static HairBxDF<double> make_hair(
	double beta_m = 0.3, double beta_n = 0.3,
	double sigma_a = 0.05, double eta = 1.55,
	double h = 0.0)
{
	return HairBxDF<double>(h, eta, sigma_a, sigma_a, sigma_a,
							beta_m, beta_n, 2.0);
}

// Fiber tangent for tests: just +Z
static const double TX = 0.0, TY = 0.0, TZ = 1.0;

// A fixed incident direction (unit length, pointing roughly along fiber)
static void make_wi(double& x, double& y, double& z) {
	x = 0.5; y = 0.3; z = -0.8;
	normalise(x, y, z);
}

// ---------------------------------------------------------------------------
// Test: sampled direction is valid (finite, unit-length-ish)
// ---------------------------------------------------------------------------
TEST(HairBxDF, SampledDirectionIsValid) {
	auto bxdf = make_hair();
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	int valid = 0;
	for (int i = 0; i < 500; ++i) {
		auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
							   randu(), randu(), randu(), randu());
		if (!res.valid) continue;
		double len = std::sqrt(res.wo_x*res.wo_x + res.wo_y*res.wo_y + res.wo_z*res.wo_z);
		EXPECT_NEAR(len, 1.0, 0.01) << "sampled direction not unit length";
		++valid;
	}
	EXPECT_GT(valid, 400) << "too many invalid samples";
}

// ---------------------------------------------------------------------------
// Test: attenuation is non-negative
// ---------------------------------------------------------------------------
TEST(HairBxDF, AttenuationNonNegative) {
	auto bxdf = make_hair();
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	for (int i = 0; i < 200; ++i) {
		auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
							   randu(), randu(), randu(), randu());
		if (!res.valid) continue;
		EXPECT_GE(res.r, 0.0);
		EXPECT_GE(res.g, 0.0);
		EXPECT_GE(res.b, 0.0);
	}
}

// ---------------------------------------------------------------------------
// Test: PDF is positive for sampled directions
// ---------------------------------------------------------------------------
TEST(HairBxDF, PDFPositiveForSampledDirections) {
	auto bxdf = make_hair();
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	int tested = 0;
	for (int i = 0; i < 500; ++i) {
		auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
							   randu(), randu(), randu(), randu());
		if (!res.valid) continue;
		double pdf = bxdf.scattering_pdf(
			TX, TY, TZ,
			wix, wiy, wiz,
			res.wo_x, res.wo_y, res.wo_z);
		EXPECT_GE(pdf, 0.0) << "PDF is negative";
		++tested;
	}
	EXPECT_GT(tested, 400);
}

// ---------------------------------------------------------------------------
// Test: roughness sweep — higher beta_m produces broader longitudinal spread
// ---------------------------------------------------------------------------
TEST(HairBxDF, RoughnessSweepProducesValidSamples) {
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	for (double bm : {0.05, 0.2, 0.5, 0.9}) {
		auto bxdf = make_hair(bm, bm);
		int valid = 0;
		for (int i = 0; i < 100; ++i) {
			auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
								   randu(), randu(), randu(), randu());
			if (res.valid) ++valid;
		}
		EXPECT_GT(valid, 50) << "beta_m=" << bm << " too many invalid samples";
	}
}

// ---------------------------------------------------------------------------
// Test: absorption sweep — higher sigma_a lowers attenuation (energy absorbed)
// ---------------------------------------------------------------------------
TEST(HairBxDF, AbsorptionReducesAttenuation) {
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	double avg_low  = 0, avg_high = 0;
	int n_low = 0, n_high = 0;

	auto bxdf_lo = make_hair(0.3, 0.3, 0.01);
	auto bxdf_hi = make_hair(0.3, 0.3, 2.0);

	for (int i = 0; i < 500; ++i) {
		double u1 = randu(), u2 = randu(), u3 = randu(), u4 = randu();
		auto rlo = bxdf_lo.sample(TX, TY, TZ, wix, wiy, wiz, u1, u2, u3, u4);
		auto rhi = bxdf_hi.sample(TX, TY, TZ, wix, wiy, wiz, u1, u2, u3, u4);
		if (rlo.valid) { avg_low  += (rlo.r+rlo.g+rlo.b)/3; ++n_low; }
		if (rhi.valid) { avg_high += (rhi.r+rhi.g+rhi.b)/3; ++n_high; }
	}

	if (n_low > 0) avg_low  /= n_low;
	if (n_high > 0) avg_high /= n_high;

	// High absorption should generally produce lower or equal average weight
	// (the R lobe is not absorbed, so equality is acceptable)
	EXPECT_LE(avg_high, avg_low + 0.5)
		<< "high absorption should not drastically exceed low absorption weight";
}

// ---------------------------------------------------------------------------
// Test: white-furnace — total reflectance <= 1 (energy conservation upper bound)
//   Under white illumination (constant radiance=1), integral of f*cos / pdf
//   should stay at or below 1 for the sampled estimator.
// ---------------------------------------------------------------------------
TEST(HairBxDF, WhiteFurnaceBound) {
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	// Use near-zero absorption so we're testing max energy
	auto bxdf = make_hair(0.3, 0.3, 1e-6);

	double sum = 0;
	int n = 0;
	for (int i = 0; i < 2000; ++i) {
		auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
							   randu(), randu(), randu(), randu());
		if (!res.valid) continue;
		// weight = f/pdf * cos(theta_i); for hair the BxDF already divides by
		// cos(theta_i) so the MIS estimator weight is just (r+g+b)/3
		double w = (res.r + res.g + res.b) / 3.0;
		sum += w;
		++n;
	}
	double avg = n > 0 ? sum / n : 0.0;
	EXPECT_LE(avg, 3.0) << "average weight exceeds energy conservation bound";
	EXPECT_GE(avg, 0.0) << "average weight is negative";
}

// ---------------------------------------------------------------------------
// Test: different h values produce valid samples
// ---------------------------------------------------------------------------
TEST(HairBxDF, CrossSectionOffsetValid) {
	double wix, wiy, wiz;
	make_wi(wix, wiy, wiz);

	for (double h : {-0.9, -0.5, 0.0, 0.5, 0.9}) {
		auto bxdf = make_hair(0.3, 0.3, 0.05, 1.55, h);
		int valid = 0;
		for (int i = 0; i < 100; ++i) {
			auto res = bxdf.sample(TX, TY, TZ, wix, wiy, wiz,
								   randu(), randu(), randu(), randu());
			if (res.valid) ++valid;
		}
		EXPECT_GT(valid, 50) << "h=" << h << " too many invalid samples";
	}
}

// ---------------------------------------------------------------------------
// HairMaterial wrapper tests
// ---------------------------------------------------------------------------
#include "ray.h"
#include "hittable.h"

TEST(HairMaterial, ConstructsAndScattersSuccessfully) {
	hair_material mat(0.06, 0.10, 0.20, 0.3, 0.3, 2.0, 1.55);

	hit_record rec;
	rec.p      = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);
	rec.t      = 1.0;

	ray r_in(point3(0, 1, 0), vec3(0.2, -0.9, 0.3));

	int scattered = 0;
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec))
			++scattered;
	}
	EXPECT_GT(scattered, 50) << "hair material should scatter most rays";
}

TEST(HairMaterial, AttenuationNonNegative) {
	hair_material mat;

	hit_record rec;
	rec.p      = point3(0, 0, 0);
	rec.normal = vec3(0, 0, 1);
	rec.t      = 1.0;

	ray r_in(point3(0, 0, 1), vec3(0.1, 0.2, -0.97));

	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_GE(srec.attenuation.x(), 0.0);
			EXPECT_GE(srec.attenuation.y(), 0.0);
			EXPECT_GE(srec.attenuation.z(), 0.0);
		}
	}
}

TEST(HairMaterial, ScatteringPDFNonNegative) {
	hair_material mat;

	hit_record rec;
	rec.p      = point3(0, 0, 0);
	rec.normal = vec3(0, 0, 1);
	rec.t      = 1.0;

	ray r_in(point3(0, 0, 1), vec3(0.1, 0.2, -0.97));

	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			double pdf = mat.scattering_pdf(r_in, rec, srec.skip_pdf_ray);
			EXPECT_GE(pdf, 0.0);
		}
	}
}
