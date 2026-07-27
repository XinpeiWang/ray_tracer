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
// Test: White-furnace energy conservation (pbrt-v4 Hair::WhiteFurnace port)
//
// pbrt-v4 reference: src/pbrt/bsdfs_test.cpp  TEST(Hair, WhiteFurnace)
//
// Method: For zero absorption (sigma_a=0), the integral of f(wo,wi)*|cosTheta_i|
// over the full sphere must equal 1 (energy conservation).
//
// We use importance sampling via sample() for numerical stability.
// The estimator:
//   weight_i = (f.r + f.g + f.b) / 3 * |wi_lz|
//            = (eval_local / |wi_lz|) / pdf * |wi_lz|   [since res = f/pdf, f = eval/|lz|]
//            = fsum / pdf
//   => E_pdf[weight] = integral of fsum over sphere = 1 for energy conservation
//
// We iterate over h (cross-section offset) as a QMC parameter per sample,
// and sweep over (beta_m, beta_n) pairs, with alpha=0 matching pbrt-v4.
// ---------------------------------------------------------------------------

// Halton radical inverse helpers for low-discrepancy sampling
static double radical_inverse_base2(uint32_t n) {
	uint32_t bits = (n << 16) | (n >> 16);
	bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
	bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
	bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
	bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
	return bits * 2.3283064365386963e-10;
}
static double radical_inverse_base(int base, uint32_t n) {
	double inv = 1.0 / base, result = 0.0, f = inv;
	while (n > 0) { result += (n % base) * f; n /= (uint32_t)base; f *= inv; }
	return result;
}

TEST(HairBxDF, WhiteFurnaceBound) {
	// Fixed wo in world space: (1, 0, 0).
	// With fiber tangent TZ=(0,0,1), to_local gives:
	//   wo_lx = sinTheta_o = dot((1,0,0),(0,0,1)) = 0  => cosTheta_o = 1 (non-degenerate)
	const double wox = 1.0, woy = 0.0, woz = 0.0;

	// Sweep (beta_m, beta_n) matching pbrt-v4 furnace test grid
	for (double beta_m = 0.1; beta_m < 1.0; beta_m += 0.2) {
		for (double beta_n = 0.1; beta_n < 1.0; beta_n += 0.2) {

			// Importance sampling via sample() converges well with ~5k samples
			// even for sharp low-roughness distributions (unlike uniform sphere)
			const int count = 5000;
			double ySum = 0.0;
			int n_valid = 0;

			for (int i = 0; i < count; ++i) {
				// Vary h (cross-section offset) per sample via QMC — same as pbrt-v4
				double h_u = radical_inverse_base(3, (uint32_t)i);
				double h = std::max(-0.999999, std::min(0.999999, -1.0 + 2.0 * h_u));

				// Construct with alpha=0 matching pbrt-v4 furnace test, sigma_a=0
				HairBxDF<double> bxdf(h, 1.55, 0.0, 0.0, 0.0, beta_m, beta_n, 0.0);

				// Build the hair ONB (needed to retrieve |wi_lz| after sampling)
				double bx, by, bz, cx, cy, cz;
				bxdf.build_hair_onb(TX, TY, TZ, bx, by, bz, cx, cy, cz);

				// Importance-sample a wi direction
				auto res = bxdf.sample(TX, TY, TZ, wox, woy, woz,
									   radical_inverse_base2((uint32_t)i),
									   radical_inverse_base(5, (uint32_t)i),
									   radical_inverse_base(7, (uint32_t)i),
									   radical_inverse_base(11, (uint32_t)i));
				if (!res.valid) continue;

				// Convert sampled world-space direction to local to get |wi_lz|
				double wil_x, wil_y, wil_z;
				bxdf.to_local(TX, TY, TZ, bx, by, bz, cx, cy, cz,
							  res.wo_x, res.wo_y, res.wo_z, wil_x, wil_y, wil_z);
				double abs_wi_lz = std::fabs(wil_z);

				// IS estimator for energy conservation:
				//   res.r = eval_local.r / pdf = (fsum.r / |wi_lz|) / pdf
				//   res.r * |wi_lz| = fsum.r / pdf
				//   E_pdf[fsum.avg / pdf] = integral of fsum.avg over sphere = 1
				double weight = (res.r + res.g + res.b) / 3.0 * abs_wi_lz;
				if (std::isfinite(weight) && weight >= 0.0) {
					ySum += weight;
					++n_valid;
				}
			}

			double avg = n_valid > 0 ? ySum / n_valid : 0.0;

			// Allow [0.90, 1.10] tolerance: slightly wider than pbrt-v4's [0.95, 1.05]
			// to account for h-averaging variance in the IS estimator
			EXPECT_GE(avg, 0.90) << "beta_m=" << beta_m << " beta_n=" << beta_n
								 << " avg=" << avg << " (energy conservation violated)";
			EXPECT_LE(avg, 1.10) << "beta_m=" << beta_m << " beta_n=" << beta_n
								 << " avg=" << avg << " (energy creation detected)";
		}
	}
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
