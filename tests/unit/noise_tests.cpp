// noise_tests.cpp -- Unit tests for pbrt-v4-aligned Perlin noise, FBm, Turbulence,
// and the CPU procedural texture wrappers (fbm_texture, marble_texture).
//
// Mirrors pbrt-v4 src/pbrt/util/noise.h coverage:
//   1. NoiseWeight / quintic C2 smoothstep
//   2. perlin_noise<double>: range, determinism, zero at integers
//   3. fbm_simple: range, octave attenuation
//   4. turbulence_simple: non-negative, range
//   5. perlin wrapper (CPU): noise(), turb(), fbm()
//   6. fbm_texture: colour range
//   7. marble_texture: colour range, spline coverage

#include <gtest/gtest.h>
#include <cmath>

// Shared noise math
#include "../../src/shared/noise.h"

// CPU wrappers -- rtweekend.h must come first (defines pi, infinity, etc.)
#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/TheRestOfYourLife/vec3.h"
#include "../../src/TheRestOfYourLife/color.h"
#include "../../src/TheRestOfYourLife/perlin.h"
#include "../../src/TheRestOfYourLife/texture.h"

// ---------------------------------------------------------------------------
// 1. NoiseWeight -- quintic C2 at boundary values and midpoint
// ---------------------------------------------------------------------------
TEST(NoiseWeight, BoundaryAndMidpoint) {
	// w(0) == 0, w(1) == 1 (pbrt-v4 property)
	EXPECT_DOUBLE_EQ(noise_detail::NoiseWeight(0.0), 0.0);
	EXPECT_DOUBLE_EQ(noise_detail::NoiseWeight(1.0), 1.0);
	// w(0.5) == 0.5  (symmetry of 6t^5-15t^4+10t^3)
	EXPECT_NEAR(noise_detail::NoiseWeight(0.5), 0.5, 1e-12);
	// Strictly monotone: w(0.25) < w(0.5) < w(0.75)
	EXPECT_LT(noise_detail::NoiseWeight(0.25), noise_detail::NoiseWeight(0.5));
	EXPECT_LT(noise_detail::NoiseWeight(0.5),  noise_detail::NoiseWeight(0.75));
}

// ---------------------------------------------------------------------------
// 2. perlin_noise<double>: range [-1,1], determinism, zero at integers
// ---------------------------------------------------------------------------
TEST(PerlinNoise, RangeIsBoundedPlusMinusOne) {
	// pbrt-v4: output is in [-1, 1] (gradient noise)
	for (double x = -3.0; x <= 3.0; x += 0.37) {
		for (double y = -3.0; y <= 3.0; y += 0.37) {
			for (double z = -3.0; z <= 3.0; z += 0.37) {
				double v = perlin_noise<double>(x, y, z);
				EXPECT_GE(v, -1.0) << "x=" << x << " y=" << y << " z=" << z;
				EXPECT_LE(v,  1.0) << "x=" << x << " y=" << y << " z=" << z;
			}
		}
	}
}

TEST(PerlinNoise, Deterministic) {
	// Same inputs must yield identical outputs (fixed perm table, no random state)
	double a = perlin_noise<double>(1.23, 4.56, 7.89);
	double b = perlin_noise<double>(1.23, 4.56, 7.89);
	EXPECT_EQ(a, b);
}

TEST(PerlinNoise, ZeroAtIntegerLattice) {
	// Gradient noise is 0 at every integer lattice point (gradient contribution = 0)
	for (int i = -2; i <= 2; ++i)
		for (int j = -2; j <= 2; ++j)
			for (int k = -2; k <= 2; ++k)
				EXPECT_NEAR(perlin_noise<double>((double)i,(double)j,(double)k),
							0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// 3. fbm_simple: range and octave attenuation
// ---------------------------------------------------------------------------
TEST(FBmSimple, RangeIsBounded) {
	// With omega=0.5, max |sum| <= 1/(1-0.5) = 2, so [-2,2]
	for (double x = -3.0; x <= 3.0; x += 0.5) {
		for (double y = -3.0; y <= 3.0; y += 0.5) {
			double v = fbm_simple<double>(x, y, 1.0, 0.5, 8);
			EXPECT_GE(v, -2.0) << "x=" << x << " y=" << y;
			EXPECT_LE(v,  2.0) << "x=" << x << " y=" << y;
		}
	}
}

TEST(FBmSimple, MoreOctavesAddsDetail) {
	// FBm with 1 octave should equal raw noise; adding octaves changes the value
	double v1 = fbm_simple<double>(1.1, 2.2, 3.3, 0.5, 1);
	double vN = fbm_simple<double>(1.1, 2.2, 3.3, 0.5, 8);
	// They should NOT be identical (octave additions change the result)
	EXPECT_NE(v1, vN);
}

// ---------------------------------------------------------------------------
// 4. turbulence_simple: non-negative output, bounded range
// ---------------------------------------------------------------------------
TEST(TurbulenceSimple, NonNegative) {
	for (double x = -3.0; x <= 3.0; x += 0.5) {
		for (double y = -3.0; y <= 3.0; y += 0.5) {
			double v = turbulence_simple<double>(x, y, 1.7, 0.5, 8);
			EXPECT_GE(v, 0.0) << "x=" << x << " y=" << y;
		}
	}
}

TEST(TurbulenceSimple, BoundedByGeometricSum) {
	// Upper bound: sum |omega^i| for i=0..n-1 = (1-omega^n)/(1-omega) < 1/(1-omega)
	// With omega=0.5 and 8 octaves: < 2.0
	for (double x = -3.0; x <= 3.0; x += 0.5) {
		for (double y = -3.0; y <= 3.0; y += 0.5) {
			double v = turbulence_simple<double>(x, y, 1.7, 0.5, 8);
			EXPECT_LE(v, 2.0) << "x=" << x << " y=" << y;
		}
	}
}

// ---------------------------------------------------------------------------
// 5. perlin wrapper: noise(), turb(), fbm()
// ---------------------------------------------------------------------------
TEST(PerlinWrapper, NoiseReturnsCpuGradientNoise) {
	perlin p;
	point3 pt(1.5, 2.7, 3.3);
	double v = p.noise(pt);
	EXPECT_GE(v, -1.0);
	EXPECT_LE(v,  1.0);
	// Verify it matches the shared perlin_noise directly
	EXPECT_DOUBLE_EQ(v, perlin_noise<double>(pt.x(), pt.y(), pt.z()));
}

TEST(PerlinWrapper, TurbIsNonNegative) {
	perlin p;
	for (int i = 0; i < 20; ++i) {
		point3 pt(i * 0.3, i * 0.7, i * 1.1);
		EXPECT_GE(p.turb(pt), 0.0);
	}
}

TEST(PerlinWrapper, FbmDelegatesToShared) {
	perlin p;
	point3 pt(0.5, 1.5, 2.5);
	double wrapper_val = p.fbm(pt, 8, 0.5);
	double shared_val  = fbm_simple<double>(pt.x(), pt.y(), pt.z(), 0.5, 8);
	EXPECT_DOUBLE_EQ(wrapper_val, shared_val);
}

// ---------------------------------------------------------------------------
// 6. fbm_texture: output colour per-channel in [0,1]
// ---------------------------------------------------------------------------
TEST(FbmTexture, ColourInRange) {
	fbm_texture tex(4.0, 8, 0.5);
	for (double x = -2.0; x <= 2.0; x += 0.5) {
		for (double y = -2.0; y <= 2.0; y += 0.5) {
			point3 p(x, y, 0.5);
			color c = tex.value(0.0, 0.0, p);
			EXPECT_GE(c.x(), 0.0);  EXPECT_LE(c.x(), 1.0);
			EXPECT_GE(c.y(), 0.0);  EXPECT_LE(c.y(), 1.0);
			EXPECT_GE(c.z(), 0.0);  EXPECT_LE(c.z(), 1.0);
		}
	}
}

TEST(FbmTexture, IsGrayscale) {
	// fbm_texture returns greyscale (r == g == b)
	fbm_texture tex(4.0);
	point3 p(1.1, 2.2, 3.3);
	color c = tex.value(0.0, 0.0, p);
	EXPECT_DOUBLE_EQ(c.x(), c.y());
	EXPECT_DOUBLE_EQ(c.y(), c.z());
}

// ---------------------------------------------------------------------------
// 7. marble_texture: colour per-channel in [0,1], covers a non-trivial range
// ---------------------------------------------------------------------------
TEST(MarbleTexture, ColourInRange) {
	marble_texture tex(1.0, 8, 0.5, 5.0);
	for (double x = -3.0; x <= 3.0; x += 0.5) {
		for (double y = -3.0; y <= 3.0; y += 0.5) {
			point3 p(x, y, 0.3);
			color c = tex.value(0.0, 0.0, p);
			EXPECT_GE(c.x(), 0.0);  EXPECT_LE(c.x(), 1.0);
			EXPECT_GE(c.y(), 0.0);  EXPECT_LE(c.y(), 1.0);
			EXPECT_GE(c.z(), 0.0);  EXPECT_LE(c.z(), 1.0);
		}
	}
}

TEST(MarbleTexture, VariesAcrossSpace) {
	// Marble should not be spatially constant
	marble_texture tex(1.0, 8, 0.5, 5.0);
	color c0 = tex.value(0, 0, point3(0, 0, 0));
	color c1 = tex.value(0, 0, point3(1, 1, 1));
	bool differs = (c0.x() != c1.x()) || (c0.y() != c1.y()) || (c0.z() != c1.z());
	EXPECT_TRUE(differs);
}

TEST(MarbleTexture, SplineCoversExpectedHue) {
	// pbrt-v4 marble colours are blue-grey; R and B should be close, G slightly less
	// (a visual sanity check, not an exact assert)
	marble_texture tex(1.0, 8, 0.5, 5.0);
	double sum_rb_diff = 0.0;
	int count = 0;
	for (double x = 0.0; x <= 4.0; x += 0.4) {
		for (double y = 0.0; y <= 4.0; y += 0.4) {
			color c = tex.value(0, 0, point3(x, y, 0.1));
			sum_rb_diff += std::abs(c.x() - c.z()); // |R-B|
			++count;
		}
	}
	// On average, marble should have R ≈ B (bluish-grey veins)
	EXPECT_LT(sum_rb_diff / count, 0.15);
}
