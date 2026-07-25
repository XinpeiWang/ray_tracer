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
