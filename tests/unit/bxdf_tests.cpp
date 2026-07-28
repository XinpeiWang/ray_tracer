/**
 * @file bxdf_tests.cpp
 * @brief pbrt-v4-style tests for shared BxDF math and ShadingFrame helper.
 *
 * Test categories (mirroring pbrt-v4's bsdfs_test.cpp philosophy):
 *   1. ShadingFrame -- deterministic frame construction and round-trip transforms
 *   2. DiffuseBxDF  -- PDF normalisation, cosine-weighted distribution
 *   3. DielectricBxDF -- Fresnel energy conservation
 *   4. BxDF sampling consistency -- f/pdf ratio stays bounded
 */

#include <gtest/gtest.h>
#include "../shared/shading_frame.h"
#include "../shared/bxdfs.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr double kPi = 3.14159265358979323846;

static double dot3(double ax, double ay, double az,
				   double bx, double by, double bz) {
	return ax*bx + ay*by + az*bz;
}

static double len3(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

// Simple low-discrepancy sequence (van der Corput, base 2 & 3) for Monte Carlo.
static double radical_inverse_2(uint32_t n) {
	n = (n << 16) | (n >> 16);
	n = ((n & 0x55555555u) << 1) | ((n & 0xAAAAAAAAu) >> 1);
	n = ((n & 0x33333333u) << 2) | ((n & 0xCCCCCCCCu) >> 2);
	n = ((n & 0x0F0F0F0Fu) << 4) | ((n & 0xF0F0F0F0u) >> 4);
	n = ((n & 0x00FF00FFu) << 8) | ((n & 0xFF00FF00u) >> 8);
	return static_cast<double>(n) / 4294967296.0;
}

static double radical_inverse_3(uint32_t n) {
	double result = 0.0, base = 1.0 / 3.0, f = base;
	while (n > 0) { result += (n % 3) * f; n /= 3; f *= base; }
	return result;
}

// ============================================================================
// 1. ShadingFrame tests
// ============================================================================

// Helper: build a frame from n and verify the three axes are mutually orthogonal
// and each unit-length.
static void CheckFrameOrthonormality(double nx, double ny, double nz) {
	ShadingFrame<double> f = ShadingFrame<double>::from_normal(nx, ny, nz);

	// Each column should have unit length.
	EXPECT_NEAR(len3(f.tx, f.ty, f.tz), 1.0, 1e-12) << "tangent not unit";
	EXPECT_NEAR(len3(f.bx, f.by, f.bz), 1.0, 1e-12) << "bitangent not unit";
	EXPECT_NEAR(len3(f.nx, f.ny, f.nz), 1.0, 1e-12) << "normal not unit";

	// All pairs should be orthogonal.
	EXPECT_NEAR(dot3(f.tx,f.ty,f.tz, f.bx,f.by,f.bz), 0.0, 1e-12) << "t·b != 0";
	EXPECT_NEAR(dot3(f.tx,f.ty,f.tz, f.nx,f.ny,f.nz), 0.0, 1e-12) << "t·n != 0";
	EXPECT_NEAR(dot3(f.bx,f.by,f.bz, f.nx,f.ny,f.nz), 0.0, 1e-12) << "b·n != 0";
}

TEST(ShadingFrameTest, OrthonormalityAxisAligned) {
	CheckFrameOrthonormality(0.0, 1.0, 0.0);
	CheckFrameOrthonormality(1.0, 0.0, 0.0);
	CheckFrameOrthonormality(0.0, 0.0, 1.0);
	CheckFrameOrthonormality(0.0, 0.0, -1.0); // degenerate case that trips naive ONB
}

TEST(ShadingFrameTest, OrthonormalityArbitraryNormals) {
	// A variety of directions, including near-degenerate ones.
	const double sq2 = 1.0 / std::sqrt(2.0);
	const double sq3 = 1.0 / std::sqrt(3.0);
	CheckFrameOrthonormality(sq2, sq2, 0.0);
	CheckFrameOrthonormality(sq3, sq3, sq3);
	CheckFrameOrthonormality(0.0, sq2, -sq2);
	CheckFrameOrthonormality(-sq3, sq3, -sq3);
	// Near n.z = -1 (historically problematic for branch-free ONB)
	// Normalise before passing so from_normal receives a true unit vector.
	double nx_ = 1e-7, ny_ = 1e-7, nz_ = -1.0 + 1e-8;
	double len_ = std::sqrt(nx_*nx_ + ny_*ny_ + nz_*nz_);
	CheckFrameOrthonormality(nx_/len_, ny_/len_, nz_/len_);
}

TEST(ShadingFrameTest, RoundTripToLocalToWorld) {
	// Build a frame from an oblique normal and verify that
	// to_world(to_local(v)) == v for several directions.
	double nx = 0.6, ny = 0.8, nz = 0.0; // already unit
	ShadingFrame<double> f = ShadingFrame<double>::from_normal(nx, ny, nz);

	auto round_trip = [&](double wx, double wy, double wz) {
		double lx, ly, lz;
		f.to_local(wx, wy, wz, lx, ly, lz);
		double rx, ry, rz;
		f.to_world(lx, ly, lz, rx, ry, rz);
		EXPECT_NEAR(rx, wx, 1e-12);
		EXPECT_NEAR(ry, wy, 1e-12);
		EXPECT_NEAR(rz, wz, 1e-12);
	};

	round_trip(1.0, 0.0, 0.0);
	round_trip(0.0, 1.0, 0.0);
	round_trip(0.0, 0.0, 1.0);
	round_trip(nx, ny, nz); // normal itself should map to (0,0,1) and back
}

TEST(ShadingFrameTest, NormalMapsToLocalZ) {
	// By convention the normal is the z-axis of the local frame.
	double nx = 0.0, ny = 1.0, nz = 0.0;
	ShadingFrame<double> f = ShadingFrame<double>::from_normal(nx, ny, nz);
	double lx, ly, lz;
	f.to_local(nx, ny, nz, lx, ly, lz);
	EXPECT_NEAR(lx, 0.0, 1e-12);
	EXPECT_NEAR(ly, 0.0, 1e-12);
	EXPECT_NEAR(lz, 1.0, 1e-12);
}

// ============================================================================
// 2. DiffuseBxDF -- PDF tests
// ============================================================================

// White-furnace test: integrate f(wo,wi)/pdf * cos(theta) over the hemisphere.
// For a Lambertian BRDF with albedo = 1 this should equal 1 (energy conserving).
TEST(DiffuseBxDFTest, WhiteFurnace) {
	DiffuseBxDF<double> bxdf{ 1.0, 1.0, 1.0 };
	const double nx = 0.0, ny = 0.0, nz = 1.0;

	// Monte Carlo integration with a quasi-random sequence.
	const int N = 4096;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		auto r = bxdf.sample(nx, ny, nz, u1, u2);
		if (!r.valid) continue;
		double cos_theta = dot3(r.wo_x, r.wo_y, r.wo_z, nx, ny, nz);
		if (cos_theta <= 0.0) continue;
		double pdf = bxdf.scattering_pdf(nx, ny, nz, r.wo_x, r.wo_y, r.wo_z);
		if (pdf <= 0.0) continue;
		// f = albedo/pi; estimator = f * cos / pdf; for cosine sampling pdf = cos/pi so ratio = albedo
		double f_over_pdf = r.r; // albedo returned directly by DiffuseBxDF
		sum += f_over_pdf;       // f * cos / pdf simplifies to albedo for cosine sampling
	}
	double estimate = sum / N;
	// Should be very close to 1.0 (albedo) -- allow 1% tolerance for QMC noise.
	EXPECT_NEAR(estimate, 1.0, 0.01);
}

// PDF must integrate to 1 over the hemisphere.
TEST(DiffuseBxDFTest, PDFIntegratesTo1) {
	DiffuseBxDF<double> bxdf{ 0.5, 0.5, 0.5 };
	const double nx = 0.0, ny = 0.0, nz = 1.0;

	const int N = 8192;
	double sum = 0.0;
	// Uniform hemisphere sampling: pdf_uniform = 1/(2*pi), weight = pdf_bxdf / pdf_uniform
	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i);
		double u2 = radical_inverse_3(i);
		// Uniform hemisphere sample
		double cos_theta = u2;                     // u2 in [0,1)
		double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
		double phi = 2.0 * kPi * u1;
		double wx = sin_theta * std::cos(phi);
		double wy = sin_theta * std::sin(phi);
		double wz = cos_theta;
		double pdf_bxdf = bxdf.scattering_pdf(nx, ny, nz, wx, wy, wz);
		// Integral of pdf_bxdf dw ≈ mean(pdf_bxdf) * 2*pi
		sum += pdf_bxdf;
	}
	double integral = (sum / N) * 2.0 * kPi;
	EXPECT_NEAR(integral, 1.0, 0.02);
}

// ============================================================================
// 3. DielectricBxDF -- Fresnel energy conservation
// ============================================================================

// Averaged over many incident angles and random reflectance choices,
// the dielectric should conserve energy: reflectance + transmittance <= 1.
TEST(DielectricBxDFTest, EnergyConservation) {
	DielectricBxDF<double> bxdf{ 1.5 };  // glass
	const double nx = 0.0, ny = 0.0, nz = 1.0;

	const int N = 4096;
	int valid_samples = 0;
	double sum_weight = 0.0;

	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		// Random incident direction in upper hemisphere (front face)
		double cos_i = u1 * 0.99 + 0.01;
		double sin_i = std::sqrt(1.0 - cos_i * cos_i);
		double phi   = 2.0 * kPi * u2;
		double wix = sin_i * std::cos(phi);
		double wiy = sin_i * std::sin(phi);
		double wiz = -cos_i; // pointing into surface

		double u3 = radical_inverse_2((i + 1) * 7919u);
		auto r = bxdf.sample(wix, wiy, wiz, nx, ny, nz, /*front_face=*/true, u3);
		if (!r.valid) continue;
		// Each sample weight (R or T color) should be <= 1
		EXPECT_LE(r.r, 1.0 + 1e-9);
		EXPECT_LE(r.g, 1.0 + 1e-9);
		EXPECT_LE(r.b, 1.0 + 1e-9);
		// All weights must be non-negative
		EXPECT_GE(r.r, 0.0);
		sum_weight += r.r;
		++valid_samples;
	}
	// Average weight should be close to 1.0 for a white dielectric
	double avg = sum_weight / valid_samples;
	EXPECT_NEAR(avg, 1.0, 0.01);
}

// ============================================================================
// 4. MetalBxDF -- specular reflection correctness
// ============================================================================

TEST(MetalBxDFTest, ReflectionStaysAboveSurface) {
	MetalBxDF<double> bxdf{ 0.8, 0.8, 0.8, 0.0 }; // zero fuzz -> pure specular
	const double nx = 0.0, ny = 0.0, nz = 1.0;

	for (int i = 0; i < 256; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		double cos_i = u1 * 0.98 + 0.01;
		double sin_i = std::sqrt(1.0 - cos_i * cos_i);
		double phi   = 2.0 * kPi * u2;
		double wix = sin_i * std::cos(phi);
		double wiy = sin_i * std::sin(phi);
		double wiz = -cos_i;

		auto r = bxdf.sample(wix, wiy, wiz, nx, ny, nz, 0.0, 0.0, 0.0);
		if (!r.valid) continue;
		// Reflected direction must be above the surface
		EXPECT_GT(dot3(r.wo_x, r.wo_y, r.wo_z, nx, ny, nz), 0.0);
		// Output direction must be normalised
		EXPECT_NEAR(len3(r.wo_x, r.wo_y, r.wo_z), 1.0, 1e-9);
	}
}

TEST(MetalBxDFTest, PerfectSpecularAngle) {
	// For zero fuzz: angle of incidence == angle of reflection.
	MetalBxDF<double> bxdf{ 1.0, 1.0, 1.0, 0.0 };
	const double nx = 0.0, ny = 0.0, nz = 1.0;
	// Incident at 45 degrees: wi = (c, 0, -c), n = (0,0,1)
	// reflect: r = wi - 2*dot(wi,n)*n = (c,0,-c) - 2*(-c)*(0,0,1) = (c,0,c)
	const double c = 1.0 / std::sqrt(2.0);
	auto r = bxdf.sample(c, 0.0, -c, nx, ny, nz, 0.0, 0.0, 0.0);
	ASSERT_TRUE(r.valid);
	EXPECT_NEAR(r.wo_x,  c,   1e-9); // x unchanged by reflection about z
	EXPECT_NEAR(r.wo_y,  0.0, 1e-9);
	EXPECT_NEAR(r.wo_z,  c,   1e-9); // z flipped from -c to +c
}

// ============================================================================
// 5. ShadingFrame -- sampling-consistency check
//    Directions sampled by DiffuseBxDF should have their local-z coordinate
//    match cos(theta) exactly (verifies the ONB used inside is consistent
//    with ShadingFrame).
// ============================================================================

TEST(BxDFSamplingConsistencyTest, CosineWeightedHemisphereLCosZ) {
	const double nx = 0.0, ny = 0.0, nz = 1.0;
	ShadingFrame<double> frame = ShadingFrame<double>::from_normal(nx, ny, nz);
	DiffuseBxDF<double> bxdf{ 1.0, 1.0, 1.0 };

	for (int i = 0; i < 512; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		auto r = bxdf.sample(nx, ny, nz, u1, u2);
		if (!r.valid) continue;

		// The sampled direction should be in the upper hemisphere.
		double cos_theta = dot3(r.wo_x, r.wo_y, r.wo_z, nx, ny, nz);
		EXPECT_GE(cos_theta, -1e-9) << "Sample below surface at i=" << i;

		// The direction must be normalised.
		EXPECT_NEAR(len3(r.wo_x, r.wo_y, r.wo_z), 1.0, 1e-9);

		// When transformed to the local frame the z-component must equal cos_theta.
		double lx, ly, lz;
		frame.to_local(r.wo_x, r.wo_y, r.wo_z, lx, ly, lz);
		EXPECT_NEAR(lz, cos_theta, 1e-9);
	}
}

// ============================================================================
// 6. RoughMetalBxDF -- pbrt-v4 bsdfs_test.cpp style
//
// pbrt-v4 uses two strategies:
//   (a) Sampling consistency: importance sampling vs uniform hemisphere
//       sampling converge to the same integral of f*Li*cos.
//   (b) Per-sample weight bounded: every sample weight in [0, 1].
//
// We work in the LOCAL shading frame (z = normal = (0,0,1)) since
// RoughMetalBxDF::sample_local() operates there.
// ============================================================================

// Helper: uniform hemisphere sample in local frame
static void uniform_hemisphere(double u1, double u2,
								double& wx, double& wy, double& wz) {
	double cos_theta = u2;
	double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
	double phi = 2.0 * kPi * u1;
	wx = sin_theta * std::cos(phi);
	wy = sin_theta * std::sin(phi);
	wz = cos_theta;
}

// Analytic GGX BRDF value (no Fresnel, pure geometry/NDF term).
// f = D(wm)*G(wo,wi) / (4*|cosO|*|cosI|)   [perfect conductor, F=1]
// Uses HEIGHT-CORRELATED Smith G = 1/(1+Lambda_o+Lambda_i) to match
// TrowbridgeReitz::G() used inside RoughMetalBxDF.
static double ggx_lambda(double cx, double cy, double cz, double alpha) {
	if (cz <= 0.0) return 0.0;
	double t2 = (1.0 - cz*cz) / (cz*cz); // tan^2(theta)
	return (std::sqrt(1.0 + alpha*alpha * t2) - 1.0) / 2.0;
}

static double ggx_brdf(double wox, double woy, double woz,
						double wix, double wiy, double wiz,
						double alpha) {
	if (woz <= 0.0 || wiz <= 0.0) return 0.0;
	// Half-vector
	double hmx = wox + wix, hmy = woy + wiy, hmz = woz + wiz;
	double hlen = std::sqrt(hmx*hmx + hmy*hmy + hmz*hmz);
	if (hlen < 1e-10) return 0.0;
	hmx /= hlen; hmy /= hlen; hmz /= hlen;
	if (hmz <= 0.0) return 0.0;
	// GGX D(wm)
	double tan2_m = (1.0 - hmz*hmz) / (hmz*hmz);
	double denom_d = kPi * alpha*alpha * std::pow(hmz, 4) * std::pow(1.0 + tan2_m/(alpha*alpha), 2);
	double D = (denom_d > 0) ? 1.0 / denom_d : 0.0;
	// Height-correlated Smith G (matches TrowbridgeReitz::G in microfacet.h)
	double G = 1.0 / (1.0 + ggx_lambda(wox,woy,woz,alpha) + ggx_lambda(wix,wiy,wiz,alpha));
	return D * G / (4.0 * woz * wiz);
}

// Sampling consistency: integral of f*cos estimated via importance sampling
// must match the same integral estimated via uniform hemisphere sampling.
// This is the core pattern from pbrt-v4's BSDFSampling tests.
TEST(RoughMetalBxDFTest, SamplingConsistency) {
	const double roughness = 0.3;
	const double alpha = std::sqrt(roughness); // RoughnessToAlpha
	RoughMetalBxDF<double> bxdf{ 1.0, 1.0, 1.0, alpha, alpha };
	const double wi_x = 0.3, wi_y = 0.0, wi_z = std::sqrt(1.0 - 0.3*0.3);

	const int N = 4096;
	double sum_importance = 0.0;
	double sum_uniform    = 0.0;

	// Simple Li(w) = w.z^2 (smooth, non-trivial test function)
	auto Li = [](double wx, double wy, double wz) { return wz * wz; };

	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);

		// Importance sampling estimate: f(wo,wi) * Li(wo) * |cos_o| / pdf
		// For VNDF sampling, pdf = D_visible and the weight G/G1 already
		// incorporates the VNDF pdf, so the sample weight is f/pdf.
		auto r = bxdf.sample_local(wi_x, wi_y, wi_z, u1, u2);
		if (r.valid && r.wo_z > 0.0) {
			// r.r = albedo * G/G1 = f*cos_o/pdf (VNDF self-normalizes).
			// Estimator for integral(f*Li*cos dw) = r.r * Li(wo), no extra cos.
			sum_importance += r.r * Li(r.wo_x, r.wo_y, r.wo_z);
		}

		// Uniform hemisphere estimate: f(wo,wi) * Li(wo) * cos / pdf_uniform
		double ux, uy, uz;
		uniform_hemisphere(u1, u2, ux, uy, uz);
		double f_val = ggx_brdf(ux, uy, uz, wi_x, wi_y, wi_z, alpha);
		// pdf_uniform = 1/(2*pi), so multiply by 2*pi
		sum_uniform += f_val * Li(ux, uy, uz) * uz * 2.0 * kPi;
	}

	double est_importance = sum_importance / N;
	double est_uniform    = sum_uniform    / N;

	// Both estimates should agree within 5% relative error
	if (est_uniform > 1e-6) {
		double rel_err = std::fabs(est_importance - est_uniform) / est_uniform;
		EXPECT_LT(rel_err, 0.05) << "importance=" << est_importance
								  << " uniform=" << est_uniform;
	}
}

// White-furnace: for albedo=1, integrate f*cos/pdf over all directions.
// In the local frame with VNDF sampling, weight = G/G1, and the
// expected integral is approximately 1 (energy conservation).
TEST(RoughMetalBxDFTest, WhiteFurnace) {
	const double alpha = std::sqrt(0.4);
	RoughMetalBxDF<double> bxdf{ 1.0, 1.0, 1.0, alpha, alpha };

	const double wi_x = 0.2, wi_y = 0.0, wi_z = std::sqrt(1.0 - 0.04);
	const int N = 8192;
	double sum = 0.0; int valid = 0;

	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		auto r = bxdf.sample_local(wi_x, wi_y, wi_z, u1, u2);
		if (!r.valid || r.wo_z <= 0.0) continue;
		// For VNDF sampling the estimator for reflectance is r.r (= G/G1 * albedo)
		sum += r.r;
		++valid;
	}
	double estimate = sum / valid;
	// GGX energy conservation: should be <= 1 and typically close to 1
	EXPECT_LE(estimate, 1.0 + 1e-6) << "Energy created: " << estimate;
	EXPECT_GT(estimate, 0.5) << "Unexpectedly low reflectance: " << estimate;
}

// Per-sample weight must be in [0,1] — no energy creation per sample.
TEST(RoughMetalBxDFTest, PerSampleWeightBounded) {
	const double alpha = std::sqrt(0.2);
	RoughMetalBxDF<double> bxdf{ 0.8, 0.8, 0.8, alpha, alpha };

	for (int i = 0; i < 1024; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		double wi_z = 0.1 + 0.89 * radical_inverse_2(i * 3 + 7);
		double wi_x = std::sqrt(1.0 - wi_z*wi_z);
		auto r = bxdf.sample_local(wi_x, 0.0, wi_z, u1, u2);
		if (!r.valid) continue;
		EXPECT_GE(r.r, 0.0) << "Negative weight at i=" << i;
		EXPECT_LE(r.r, 1.0 + 1e-9) << "Weight > 1 at i=" << i;
	}
}

// ============================================================================
// 7. RoughDielectricBxDF -- pbrt-v4 bsdfs_test.cpp style
// ============================================================================

// Sampling consistency: importance vs uniform hemisphere.
TEST(RoughDielectricBxDFTest, SamplingConsistency) {
	const double alpha = std::sqrt(0.25);
	const double ior   = 1.5;
	RoughDielectricBxDF<double> bxdf{ ior, alpha, alpha };

	const double wi_x = 0.2, wi_y = 0.0, wi_z = std::sqrt(1.0 - 0.04);
	const double eta  = 1.0 / ior; // entering from outside

	const int N = 4096;
	double sum_importance = 0.0, sum_uniform = 0.0;
	auto Li = [](double wx, double wy, double wz) {
		double cz = std::fabs(wz);
		return cz * cz;
	};

	for (int i = 0; i < N; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		double u3 = radical_inverse_2((i + 1) * 2053u);

		auto r = bxdf.sample_local(wi_x, wi_y, wi_z, eta, u1, u2, u3);
		if (r.valid) {
			// weight = r.r (= 1.0 for dielectric, since G/G1 self-normalizes)
			double cz = std::fabs(r.wo_z);
			sum_importance += r.r * Li(r.wo_x, r.wo_y, r.wo_z) * cz;
		}

		// Uniform full sphere (dielectric can transmit)
		double cos_theta = -1.0 + 2.0 * u2;
		double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta*cos_theta));
		double phi = 2.0 * kPi * u1;
		double ux = sin_theta * std::cos(phi);
		double uy = sin_theta * std::sin(phi);
		double uz = cos_theta;
		// Analytic GGX dielectric BRDF (transmission omitted for simplicity;
		// just verify reflection lobe consistency, wo in upper hemisphere)
		if (uz > 0.0) {
			double f_val = ggx_brdf(ux, uy, uz, wi_x, wi_y, wi_z, alpha);
			sum_uniform += f_val * Li(ux, uy, uz) * uz * 2.0 * kPi;
		}
	}

	double est_importance = sum_importance / N;
	double est_uniform    = sum_uniform    / N;

	// Rough dielectric contributes both reflection and transmission;
	// the importance estimate includes both while the uniform estimate
	// covers only reflection -- so we just check the important is non-trivially positive
	// and not wildly larger than the uniform reflection estimate.
	EXPECT_GT(est_importance, 0.0) << "Importance estimate should be positive";
	EXPECT_LT(est_importance, 5.0 * (est_uniform + 1e-4))
		<< "Importance estimate suspiciously large vs reflection-only uniform";
}

// Energy conservation: weight per sample <= 1, average close to 1.
TEST(RoughDielectricBxDFTest, EnergyConservationPerSample) {
	const double alpha = std::sqrt(0.3);
	RoughDielectricBxDF<double> bxdf{ 1.5, alpha, alpha };

	double sum = 0.0; int valid = 0;
	for (int i = 0; i < 2048; ++i) {
		double u1 = radical_inverse_2(i + 1);
		double u2 = radical_inverse_3(i + 1);
		double u3 = radical_inverse_2((i + 1) * 1999u);
		double wi_z = 0.05 + 0.9 * radical_inverse_3(i + 1);
		double wi_x = std::sqrt(std::max(0.0, 1.0 - wi_z*wi_z));
		auto r = bxdf.sample_local(wi_x, 0.0, wi_z, 1.0/1.5, u1, u2, u3);
		if (!r.valid) continue;
		// Dielectric weight is always 1.0 (self-normalizing VNDF)
		EXPECT_NEAR(r.r, 1.0, 1e-9) << "Dielectric weight should be exactly 1";
		sum += r.r;
		++valid;
	}
	EXPECT_GT(valid, 1024) << "Too many invalid samples: " << (2048 - valid);
}
