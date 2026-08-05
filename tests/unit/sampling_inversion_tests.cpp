// sampling_inversion_tests.cpp
// Round-trip validation for InvertSphericalRectangleSample and
// InvertSphericalTriangleSample, ported from pbrt-v4 util/sampling.cpp.
//
// Methodology (mirrors pbrt-v4 sampling tests):
//   For each function, generate random forward samples (u0,u1) -> point/dir,
//   then invert back and verify ||recovered - original|| < tolerance.
//
// Tests:
//   1.  SphericalRectInvert_RoundTrip_Grid       -- 10x10 grid of (u0,u1)
//   2.  SphericalRectInvert_RoundTrip_Random     -- 500 random (u0,u1)
//   3.  SphericalRectInvert_DegenerateSmall      -- tiny rectangle falls back
//   4.  SphericalTriInvert_RoundTrip_Grid        -- 10x10 grid of (u0,u1)
//   5.  SphericalTriInvert_RoundTrip_Random      -- 500 random (u0,u1)
//   6.  SphericalTriInvert_OutputClamped         -- u0,u1 always in [0,1]
//   7.  SphericalRectInvert_OutputClamped        -- u0,u1 always in [0,1]
//   8.  SphericalRectInvert_VaryingObserverPos   -- different shading points
//   9.  SphericalTriInvert_ObliqueTriangle       -- non-planar triangle

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

#include "../../src/shared/sampling_patched.h"

// ---------------------------------------------------------------------------
// LCG for reproducible random numbers
// ---------------------------------------------------------------------------
struct LCG {
	uint64_t state;
	explicit LCG(uint64_t seed = 0xDEADBEEFCAFEULL) : state(seed) {}
	double next() {
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		return (state >> 11) * (1.0 / (1ULL << 53));
	}
};

// ---------------------------------------------------------------------------
// Helpers: forward sample wrappers that match the local API
// ---------------------------------------------------------------------------

// SampleSphericalRectangle returns a world-space point on the rectangle.
// We use the version from sampling_patched.h (double-precision).
static void fwd_rect(double px, double py, double pz,
					 double sx, double sy, double sz,
					 double exx, double exy, double exz,
					 double eyx, double eyy, double eyz,
					 double u0, double u1,
					 double& prx, double& pry, double& prz)
{
	double pdf = 0.0;
	SampleSphericalRectangle(px, py, pz,
							 sx, sy, sz,
							 exx, exy, exz,
							 eyx, eyy, eyz,
							 u0, u1,
							 &prx, &pry, &prz, &pdf);
}

// SampleSphericalTriangle returns barycentrics; we reconstruct the direction
// as the normalised vector from p to the surface point.
static void fwd_tri(double v0x, double v0y, double v0z,
					double v1x, double v1y, double v1z,
					double v2x, double v2y, double v2z,
					double px,  double py,  double pz,
					double u0,  double u1,
					double& wx, double& wy, double& wz)
{
	double b0, b1, b2, pdf;
	SampleSphericalTriangle(v0x, v0y, v0z,
							v1x, v1y, v1z,
							v2x, v2y, v2z,
							px, py, pz,
							u0, u1,
							&b0, &b1, &b2, &pdf);
	// Reconstruct direction from barycentrics
	double qx = b0*v0x + b1*v1x + b2*v2x - px;
	double qy = b0*v0y + b1*v1y + b2*v2y - py;
	double qz = b0*v0z + b1*v1z + b2*v2z - pz;
	double len = std::sqrt(qx*qx + qy*qy + qz*qz);
	if (len < 1e-30) { wx = wy = 0.0; wz = 1.0; return; }
	wx = qx/len; wy = qy/len; wz = qz/len;
}

// ---------------------------------------------------------------------------
// Test fixtures: a canonical unit-square rectangle and a unit triangle
// ---------------------------------------------------------------------------
// Rectangle: unit square in z=2 plane, from (0,0,2) with ex=(1,0,0), ey=(0,1,0)
// Observer at origin (0,0,0)
static const double kPx=0, kPy=0, kPz=0;
static const double kSx=0, kSy=0, kSz=2;   // corner
static const double kExx=1, kExy=0, kExz=0; // ex
static const double kEyx=0, kEyy=1, kEyz=0; // ey

// Triangle: equilateral in z=3 plane
static const double kV0x=0, kV0y=0, kV0z=3;
static const double kV1x=1, kV1y=0, kV1z=3;
static const double kV2x=0.5, kV2y=0.866, kV2z=3;

static const double kTol = 5e-4; // round-trip tolerance

// ===========================================================================
// Test 1: SphericalRectInvert_RoundTrip_Grid
// ===========================================================================
TEST(SphericalInversionTest, SphericalRectInvert_RoundTrip_Grid) {
	const int N = 10;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double u0 = (i + 0.5) / N;
			double u1 = (j + 0.5) / N;

			double prx, pry, prz;
			fwd_rect(kPx, kPy, kPz,
					 kSx, kSy, kSz,
					 kExx, kExy, kExz,
					 kEyx, kEyy, kEyz,
					 u0, u1, prx, pry, prz);

			double ru0, ru1;
			InvertSphericalRectangleSample(
				kPx, kPy, kPz,
				kSx, kSy, kSz,
				kExx, kExy, kExz,
				kEyx, kEyy, kEyz,
				prx, pry, prz,
				&ru0, &ru1);

			EXPECT_NEAR(ru0, u0, kTol) << "grid i=" << i << " j=" << j << " u0";
			EXPECT_NEAR(ru1, u1, kTol) << "grid i=" << i << " j=" << j << " u1";
		}
	}
}

// ===========================================================================
// Test 2: SphericalRectInvert_RoundTrip_Random
// ===========================================================================
TEST(SphericalInversionTest, SphericalRectInvert_RoundTrip_Random) {
	LCG rng(1);
	for (int i = 0; i < 500; ++i) {
		double u0 = rng.next(), u1 = rng.next();

		double prx, pry, prz;
		fwd_rect(kPx, kPy, kPz,
				 kSx, kSy, kSz,
				 kExx, kExy, kExz,
				 kEyx, kEyy, kEyz,
				 u0, u1, prx, pry, prz);

		double ru0, ru1;
		InvertSphericalRectangleSample(
			kPx, kPy, kPz,
			kSx, kSy, kSz,
			kExx, kExy, kExz,
			kEyx, kEyy, kEyz,
			prx, pry, prz,
			&ru0, &ru1);

		EXPECT_NEAR(ru0, u0, kTol) << "random sample " << i << " u0";
		EXPECT_NEAR(ru1, u1, kTol) << "random sample " << i << " u1";
	}
}

// ===========================================================================
// Test 3: SphericalRectInvert_DegenerateSmall
// Very small rectangle => solid angle < 1e-3 => planar fallback.
// ===========================================================================
TEST(SphericalInversionTest, SphericalRectInvert_DegenerateSmall) {
	// Tiny rectangle 1e-6 x 1e-6 at z=2
	double tiny = 1e-6;
	double u0 = 0.3, u1 = 0.7;
	double prx, pry, prz;
	fwd_rect(kPx, kPy, kPz,
			 kSx, kSy, kSz,
			 tiny, 0, 0,
			 0, tiny, 0,
			 u0, u1, prx, pry, prz);

	double ru0, ru1;
	InvertSphericalRectangleSample(
		kPx, kPy, kPz,
		kSx, kSy, kSz,
		tiny, 0, 0,
		0, tiny, 0,
		prx, pry, prz,
		&ru0, &ru1);

	// Just check it doesn't NaN/crash and is in [0,1]
	EXPECT_TRUE(std::isfinite(ru0)) << "ru0 not finite";
	EXPECT_TRUE(std::isfinite(ru1)) << "ru1 not finite";
	EXPECT_GE(ru0, 0.0); EXPECT_LE(ru0, 1.0);
	EXPECT_GE(ru1, 0.0); EXPECT_LE(ru1, 1.0);
}

// ===========================================================================
// Test 4: SphericalTriInvert_RoundTrip_Grid
// ===========================================================================
TEST(SphericalInversionTest, SphericalTriInvert_RoundTrip_Grid) {
	const int N = 10;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double u0 = (i + 0.5) / N;
			double u1 = (j + 0.5) / N;

			double wx, wy, wz;
			fwd_tri(kV0x, kV0y, kV0z,
					kV1x, kV1y, kV1z,
					kV2x, kV2y, kV2z,
					kPx, kPy, kPz,
					u0, u1, wx, wy, wz);

			double ru0, ru1;
			InvertSphericalTriangleSample(
				kV0x, kV0y, kV0z,
				kV1x, kV1y, kV1z,
				kV2x, kV2y, kV2z,
				kPx, kPy, kPz,
				wx, wy, wz,
				&ru0, &ru1);

			EXPECT_NEAR(ru0, u0, kTol) << "grid i=" << i << " j=" << j << " u0";
			EXPECT_NEAR(ru1, u1, kTol) << "grid i=" << i << " j=" << j << " u1";
		}
	}
}

// ===========================================================================
// Test 5: SphericalTriInvert_RoundTrip_Random
// Note: u0 values very close to 0 hit the Arvo near-degenerate guard
// (dot(a,cp) > 0.99999847691 => u0 clamped to 0), which is a known and
// acknowledged limitation in pbrt-v4 (marked CHECK_RARE). We therefore
// restrict u0 to [0.01, 0.99] to stay away from that region.
// ===========================================================================
TEST(SphericalInversionTest, SphericalTriInvert_RoundTrip_Random) {
	LCG rng(2);
	for (int i = 0; i < 500; ++i) {
		// Avoid the near-degenerate Arvo region (u0 near 0)
		double u0 = 0.01 + rng.next() * 0.98;
		double u1 = rng.next();

		double wx, wy, wz;
		fwd_tri(kV0x, kV0y, kV0z,
				kV1x, kV1y, kV1z,
				kV2x, kV2y, kV2z,
				kPx, kPy, kPz,
				u0, u1, wx, wy, wz);

		double ru0, ru1;
		InvertSphericalTriangleSample(
			kV0x, kV0y, kV0z,
			kV1x, kV1y, kV1z,
			kV2x, kV2y, kV2z,
			kPx, kPy, kPz,
			wx, wy, wz,
			&ru0, &ru1);

		EXPECT_NEAR(ru0, u0, kTol) << "random sample " << i << " u0";
		EXPECT_NEAR(ru1, u1, kTol) << "random sample " << i << " u1";
	}
}

// ===========================================================================
// Test 6: SphericalTriInvert_OutputClamped
// ===========================================================================
TEST(SphericalInversionTest, SphericalTriInvert_OutputClamped) {
	LCG rng(3);
	for (int i = 0; i < 200; ++i) {
		double u0 = rng.next(), u1 = rng.next();
		double wx, wy, wz;
		fwd_tri(kV0x, kV0y, kV0z,
				kV1x, kV1y, kV1z,
				kV2x, kV2y, kV2z,
				kPx, kPy, kPz,
				u0, u1, wx, wy, wz);
		double ru0, ru1;
		InvertSphericalTriangleSample(
			kV0x, kV0y, kV0z, kV1x, kV1y, kV1z, kV2x, kV2y, kV2z,
			kPx, kPy, kPz, wx, wy, wz, &ru0, &ru1);
		EXPECT_GE(ru0, 0.0); EXPECT_LE(ru0, 1.0);
		EXPECT_GE(ru1, 0.0); EXPECT_LE(ru1, 1.0);
		EXPECT_TRUE(std::isfinite(ru0));
		EXPECT_TRUE(std::isfinite(ru1));
	}
}

// ===========================================================================
// Test 7: SphericalRectInvert_OutputClamped
// ===========================================================================
TEST(SphericalInversionTest, SphericalRectInvert_OutputClamped) {
	LCG rng(4);
	for (int i = 0; i < 200; ++i) {
		double u0 = rng.next(), u1 = rng.next();
		double prx, pry, prz;
		fwd_rect(kPx, kPy, kPz,
				 kSx, kSy, kSz,
				 kExx, kExy, kExz,
				 kEyx, kEyy, kEyz,
				 u0, u1, prx, pry, prz);
		double ru0, ru1;
		InvertSphericalRectangleSample(
			kPx, kPy, kPz,
			kSx, kSy, kSz,
			kExx, kExy, kExz,
			kEyx, kEyy, kEyz,
			prx, pry, prz, &ru0, &ru1);
		EXPECT_GE(ru0, 0.0); EXPECT_LE(ru0, 1.0);
		EXPECT_GE(ru1, 0.0); EXPECT_LE(ru1, 1.0);
		EXPECT_TRUE(std::isfinite(ru0));
		EXPECT_TRUE(std::isfinite(ru1));
	}
}

// ===========================================================================
// Test 8: SphericalRectInvert_VaryingObserverPos
// Observer at different positions relative to rectangle.
// ===========================================================================
TEST(SphericalInversionTest, SphericalRectInvert_VaryingObserverPos) {
	// Observer positions to test
	struct Pos { double x, y, z; };
	Pos observers[] = {
		{ 0.5, 0.5, 0.0 },   // centred below
		{-1.0, 0.0, 0.0 },   // off to side
		{ 0.0, 0.0,-1.0 },   // behind
		{ 2.0, 3.0, 1.5 },   // arbitrary
	};
	LCG rng(5);
	for (auto& obs : observers) {
		for (int i = 0; i < 50; ++i) {
			double u0 = rng.next(), u1 = rng.next();
			double prx, pry, prz;
			fwd_rect(obs.x, obs.y, obs.z,
					 kSx, kSy, kSz,
					 kExx, kExy, kExz,
					 kEyx, kEyy, kEyz,
					 u0, u1, prx, pry, prz);
			double ru0, ru1;
			InvertSphericalRectangleSample(
				obs.x, obs.y, obs.z,
				kSx, kSy, kSz,
				kExx, kExy, kExz,
				kEyx, kEyy, kEyz,
				prx, pry, prz, &ru0, &ru1);
			EXPECT_NEAR(ru0, u0, kTol) << "obs(" << obs.x << "," << obs.z << ") u0";
			EXPECT_NEAR(ru1, u1, kTol) << "obs(" << obs.x << "," << obs.z << ") u1";
		}
	}
}

// ===========================================================================
// Test 9: SphericalTriInvert_ObliqueTriangle
// Non-axis-aligned triangle to stress the Arvo inversion.
// ===========================================================================
TEST(SphericalInversionTest, SphericalTriInvert_ObliqueTriangle) {
	// Tilted triangle
	double ax=1, ay=0, az=3;
	double bx=2, by=1, bz=4;
	double cx=0, cy=2, cz=3;
	double ox=0.5, oy=0.5, oz=0;  // observer

	LCG rng(6);
	for (int i = 0; i < 200; ++i) {
		double u0 = 0.01 + rng.next() * 0.98;  // avoid Arvo near-degenerate at u0~0
		double u1 = rng.next();
		double wx, wy, wz;
		fwd_tri(ax, ay, az, bx, by, bz, cx, cy, cz, ox, oy, oz,
				u0, u1, wx, wy, wz);
		double ru0, ru1;
		InvertSphericalTriangleSample(
			ax, ay, az, bx, by, bz, cx, cy, cz,
			ox, oy, oz, wx, wy, wz, &ru0, &ru1);
		EXPECT_NEAR(ru0, u0, kTol) << "oblique tri sample " << i << " u0";
		EXPECT_NEAR(ru1, u1, kTol) << "oblique tri sample " << i << " u1";
	}
}
