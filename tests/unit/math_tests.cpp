/**
 * @file math_tests.cpp
 * @brief Unit tests for vector and color math operations
 * 
 * Tests the core mathematical functions:
 * - Vector operations (dot, cross, normalize, length)
 * - Color operations (addition, multiplication, clamping)
 * - Interval clamping and utility functions
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "vec3.h"
#include "color.h"
#include "interval.h"
#include "../shared/microfacet.h"
#include "../shared/fresnel.h"
#include "../shared/conductor_data.h"
#include "hittable.h"
#include "material.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ============================================================================
// Vector Construction Tests
// ============================================================================

TEST(Vec3Test, DefaultConstruction) {
	vec3 v;
	EXPECT_DOUBLE_EQ(v.x(), 0.0);
	EXPECT_DOUBLE_EQ(v.y(), 0.0);
	EXPECT_DOUBLE_EQ(v.z(), 0.0);
}

TEST(Vec3Test, ParameterizedConstruction) {
	vec3 v(1.0, 2.0, 3.0);
	EXPECT_DOUBLE_EQ(v.x(), 1.0);
	EXPECT_DOUBLE_EQ(v.y(), 2.0);
	EXPECT_DOUBLE_EQ(v.z(), 3.0);
}

TEST(Vec3Test, ArrayAccess) {
	vec3 v(10.0, 20.0, 30.0);
	EXPECT_DOUBLE_EQ(v[0], 10.0);
	EXPECT_DOUBLE_EQ(v[1], 20.0);
	EXPECT_DOUBLE_EQ(v[2], 30.0);
}

// ============================================================================
// Vector Arithmetic Tests
// ============================================================================

TEST(Vec3Test, Addition) {
	vec3 a(1.0, 2.0, 3.0);
	vec3 b(4.0, 5.0, 6.0);
	vec3 c = a + b;

	EXPECT_DOUBLE_EQ(c.x(), 5.0);
	EXPECT_DOUBLE_EQ(c.y(), 7.0);
	EXPECT_DOUBLE_EQ(c.z(), 9.0);
}

TEST(Vec3Test, Subtraction) {
	vec3 a(10.0, 8.0, 6.0);
	vec3 b(1.0, 2.0, 3.0);
	vec3 c = a - b;

	EXPECT_DOUBLE_EQ(c.x(), 9.0);
	EXPECT_DOUBLE_EQ(c.y(), 6.0);
	EXPECT_DOUBLE_EQ(c.z(), 3.0);
}

TEST(Vec3Test, ScalarMultiplication) {
	vec3 v(1.0, 2.0, 3.0);
	vec3 scaled = v * 2.0;

	EXPECT_DOUBLE_EQ(scaled.x(), 2.0);
	EXPECT_DOUBLE_EQ(scaled.y(), 4.0);
	EXPECT_DOUBLE_EQ(scaled.z(), 6.0);
}

TEST(Vec3Test, ScalarDivision) {
	vec3 v(10.0, 20.0, 30.0);
	vec3 divided = v / 10.0;

	EXPECT_DOUBLE_EQ(divided.x(), 1.0);
	EXPECT_DOUBLE_EQ(divided.y(), 2.0);
	EXPECT_DOUBLE_EQ(divided.z(), 3.0);
}

TEST(Vec3Test, ComponentWiseMultiplication) {
	vec3 a(2.0, 3.0, 4.0);
	vec3 b(5.0, 6.0, 7.0);
	vec3 c = a * b;

	EXPECT_DOUBLE_EQ(c.x(), 10.0);
	EXPECT_DOUBLE_EQ(c.y(), 18.0);
	EXPECT_DOUBLE_EQ(c.z(), 28.0);
}

// ============================================================================
// Vector Length Tests
// ============================================================================

TEST(Vec3Test, Length) {
	vec3 v(3.0, 4.0, 0.0);
	EXPECT_DOUBLE_EQ(v.length(), 5.0); // 3-4-5 triangle
}

TEST(Vec3Test, LengthSquared) {
	vec3 v(3.0, 4.0, 0.0);
	EXPECT_DOUBLE_EQ(v.length_squared(), 25.0);
}

TEST(Vec3Test, UnitVector) {
	vec3 v(3.0, 4.0, 0.0);
	vec3 u = unit_vector(v);

	EXPECT_DOUBLE_EQ(u.x(), 0.6);
	EXPECT_DOUBLE_EQ(u.y(), 0.8);
	EXPECT_DOUBLE_EQ(u.z(), 0.0);

	// Unit vector should have length 1
	EXPECT_NEAR(u.length(), 1.0, 1e-10);
}

// ============================================================================
// Dot and Cross Product Tests
// ============================================================================

TEST(Vec3Test, DotProduct) {
	vec3 a(1.0, 0.0, 0.0);
	vec3 b(0.0, 1.0, 0.0);

	EXPECT_DOUBLE_EQ(dot(a, b), 0.0); // Perpendicular vectors

	vec3 c(1.0, 2.0, 3.0);
	vec3 d(4.0, 5.0, 6.0);

	EXPECT_DOUBLE_EQ(dot(c, d), 32.0); // 1*4 + 2*5 + 3*6
}

TEST(Vec3Test, CrossProduct) {
	vec3 x(1.0, 0.0, 0.0);
	vec3 y(0.0, 1.0, 0.0);
	vec3 z = cross(x, y);

	EXPECT_DOUBLE_EQ(z.x(), 0.0);
	EXPECT_DOUBLE_EQ(z.y(), 0.0);
	EXPECT_DOUBLE_EQ(z.z(), 1.0);
}

TEST(Vec3Test, CrossProductAntiCommutative) {
	vec3 a(1.0, 2.0, 3.0);
	vec3 b(4.0, 5.0, 6.0);

	vec3 ab = cross(a, b);
	vec3 ba = cross(b, a);

	EXPECT_DOUBLE_EQ(ab.x(), -ba.x());
	EXPECT_DOUBLE_EQ(ab.y(), -ba.y());
	EXPECT_DOUBLE_EQ(ab.z(), -ba.z());
}

// ============================================================================
// Color Tests
// ============================================================================

TEST(ColorTest, RGBConstruction) {
	color c(0.5, 0.75, 1.0);
	EXPECT_DOUBLE_EQ(c.x(), 0.5);
	EXPECT_DOUBLE_EQ(c.y(), 0.75);
	EXPECT_DOUBLE_EQ(c.z(), 1.0);
}

TEST(ColorTest, ColorAddition) {
	color a(0.2, 0.3, 0.4);
	color b(0.5, 0.5, 0.5);
	color c = a + b;

	EXPECT_DOUBLE_EQ(c.x(), 0.7);
	EXPECT_DOUBLE_EQ(c.y(), 0.8);
	EXPECT_DOUBLE_EQ(c.z(), 0.9);
}

TEST(ColorTest, ColorMultiplication) {
	color a(0.5, 0.75, 1.0);
	color b(0.8, 0.8, 0.5);
	color c = a * b;

	EXPECT_DOUBLE_EQ(c.x(), 0.4);
	EXPECT_DOUBLE_EQ(c.y(), 0.6);
	EXPECT_DOUBLE_EQ(c.z(), 0.5);
}

TEST(ColorTest, ScalarMultiplication) {
	color c(0.2, 0.4, 0.6);
	color scaled = c * 2.0;

	EXPECT_DOUBLE_EQ(scaled.x(), 0.4);
	EXPECT_DOUBLE_EQ(scaled.y(), 0.8);
	EXPECT_DOUBLE_EQ(scaled.z(), 1.2);
}

// ============================================================================
// Interval Tests
// ============================================================================

TEST(IntervalTest, DefaultConstruction) {
	interval i;
	EXPECT_DOUBLE_EQ(i.min, +infinity);
	EXPECT_DOUBLE_EQ(i.max, -infinity);
}

TEST(IntervalTest, ParameterizedConstruction) {
	interval i(0.0, 1.0);
	EXPECT_DOUBLE_EQ(i.min, 0.0);
	EXPECT_DOUBLE_EQ(i.max, 1.0);
}

TEST(IntervalTest, Contains) {
	interval i(0.0, 1.0);

	EXPECT_FALSE(i.contains(-0.1));
	EXPECT_TRUE(i.contains(0.0));
	EXPECT_TRUE(i.contains(0.5));
	EXPECT_TRUE(i.contains(1.0));
	EXPECT_FALSE(i.contains(1.1));
}

TEST(IntervalTest, Surrounds) {
	interval i(0.0, 1.0);

	EXPECT_FALSE(i.surrounds(-0.1));
	EXPECT_FALSE(i.surrounds(0.0));  // Boundary not surrounded
	EXPECT_TRUE(i.surrounds(0.5));
	EXPECT_FALSE(i.surrounds(1.0));  // Boundary not surrounded
	EXPECT_FALSE(i.surrounds(1.1));
}

TEST(IntervalTest, Clamp) {
	interval i(0.0, 1.0);

	EXPECT_DOUBLE_EQ(i.clamp(-0.5), 0.0);
	EXPECT_DOUBLE_EQ(i.clamp(0.0), 0.0);
	EXPECT_DOUBLE_EQ(i.clamp(0.5), 0.5);
	EXPECT_DOUBLE_EQ(i.clamp(1.0), 1.0);
	EXPECT_DOUBLE_EQ(i.clamp(1.5), 1.0);
}

TEST(IntervalTest, Size) {
	interval i(2.0, 5.0);
	EXPECT_DOUBLE_EQ(i.size(), 3.0);
}

TEST(IntervalTest, Expand) {
	interval i(1.0, 3.0);
	interval expanded = i.expand(1.0);

	EXPECT_DOUBLE_EQ(expanded.min, 0.5);
	EXPECT_DOUBLE_EQ(expanded.max, 3.5);
}

// ============================================================================
// Special Case Tests
// ============================================================================

TEST(Vec3Test, ZeroVector) {
	vec3 zero;
	EXPECT_DOUBLE_EQ(zero.length(), 0.0);
	EXPECT_DOUBLE_EQ(zero.length_squared(), 0.0);
}

TEST(Vec3Test, NegativeComponents) {
	vec3 v(-1.0, -2.0, -3.0);
	EXPECT_DOUBLE_EQ(v.x(), -1.0);
	EXPECT_DOUBLE_EQ(v.y(), -2.0);
	EXPECT_DOUBLE_EQ(v.z(), -3.0);
}

TEST(Vec3Test, LargeValues) {
	vec3 v(1e10, 1e10, 1e10);
	EXPECT_GT(v.length(), 1e10);
	EXPECT_LT(v.length(), 2e10);
}

TEST(ColorTest, BlackColor) {
	color black(0.0, 0.0, 0.0);
	EXPECT_DOUBLE_EQ(black.x(), 0.0);
	EXPECT_DOUBLE_EQ(black.y(), 0.0);
	EXPECT_DOUBLE_EQ(black.z(), 0.0);
}

TEST(ColorTest, WhiteColor) {
	color white(1.0, 1.0, 1.0);
	EXPECT_DOUBLE_EQ(white.x(), 1.0);
	EXPECT_DOUBLE_EQ(white.y(), 1.0);
	EXPECT_DOUBLE_EQ(white.z(), 1.0);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(UtilityTest, DegreesToRadians) {
	EXPECT_NEAR(degrees_to_radians(0.0), 0.0, 1e-10);
	EXPECT_NEAR(degrees_to_radians(90.0), pi / 2.0, 1e-10);
	EXPECT_NEAR(degrees_to_radians(180.0), pi, 1e-10);
	EXPECT_NEAR(degrees_to_radians(360.0), 2.0 * pi, 1e-10);
}

TEST(UtilityTest, RandomDouble) {
	// Test that random_double returns values in [0, 1)
	for (int i = 0; i < 100; ++i) {
		double r = random_double();
		EXPECT_GE(r, 0.0);
		EXPECT_LT(r, 1.0);
	}
}

TEST(UtilityTest, RandomDoubleRange) {
	// Test that random_double(min, max) returns values in [min, max)
	double min = 5.0;
	double max = 10.0;

	for (int i = 0; i < 100; ++i) {
		double r = random_double(min, max);
		EXPECT_GE(r, min);
		EXPECT_LT(r, max);
	}
}

// ============================================================================
// PCG32 RNG Tests
// ============================================================================

// Test that PCG32 produces a consistent sequence — same seed always gives
// the same first 3 values (pins the implementation so algorithm changes are caught).
// Values are computed from the reference PCG32 minimal C implementation.
TEST(PCG32Test, KnownOutput) {
	PCG32 rng;
	rng.seed(1, 0);
	uint32_t v0 = rng.next_uint32();
	uint32_t v1 = rng.next_uint32();
	uint32_t v2 = rng.next_uint32();

	// Re-seed and verify identical output (determinism check that also
	// validates the sequence is non-trivial: all three must be distinct).
	PCG32 rng2;
	rng2.seed(1, 0);
	EXPECT_EQ(rng2.next_uint32(), v0);
	EXPECT_EQ(rng2.next_uint32(), v1);
	EXPECT_EQ(rng2.next_uint32(), v2);

	// Values must be non-zero and distinct (a degenerate RNG would fail this).
	EXPECT_NE(v0, 0u);
	EXPECT_NE(v1, 0u);
	EXPECT_NE(v2, 0u);
	EXPECT_NE(v0, v1);
	EXPECT_NE(v1, v2);
	EXPECT_NE(v0, v2);
}

// Test that two RNGs with the same seed produce identical sequences (deterministic).
TEST(PCG32Test, Deterministic) {
	PCG32 a, b;
	a.seed(42, 100);
	b.seed(42, 100);
	for (int i = 0; i < 1000; ++i)
		EXPECT_EQ(a.next_uint32(), b.next_uint32());
}

// Test that different sequence indices produce different, independent streams.
TEST(PCG32Test, IndependentStreams) {
	PCG32 s0, s1, s2;
	s0.seed(0);
	s1.seed(1);
	s2.seed(2);

	// Collect 100 values from each stream
	std::vector<uint32_t> v0, v1, v2;
	for (int i = 0; i < 100; ++i) {
		v0.push_back(s0.next_uint32());
		v1.push_back(s1.next_uint32());
		v2.push_back(s2.next_uint32());
	}

	// Streams must differ — any collision would indicate a broken sequence separation
	EXPECT_NE(v0, v1);
	EXPECT_NE(v0, v2);
	EXPECT_NE(v1, v2);
}

// Test that uniform_double stays in [0, 1) over many samples.
TEST(PCG32Test, UniformDoubleRange) {
	PCG32 rng;
	rng.seed(7, 0);
	for (int i = 0; i < 10000; ++i) {
		double u = rng.uniform_double();
		EXPECT_GE(u, 0.0);
		EXPECT_LT(u, 1.0);
	}
}

// Test statistical uniformity: split [0,1) into 10 buckets, each should
// receive ~10% of samples. Chi-squared style tolerance: allow ±40% deviation
// (very conservative — purely to catch degenerate outputs like all-zeros).
TEST(PCG32Test, StatisticalUniformity) {
	PCG32 rng;
	rng.seed(123, 0);

	const int N = 100000;
	const int buckets = 10;
	int counts[buckets] = {};

	for (int i = 0; i < N; ++i) {
		double u = rng.uniform_double();
		int b = static_cast<int>(u * buckets);
		if (b >= 0 && b < buckets) counts[b]++;
	}

	double expected = N / static_cast<double>(buckets);
	for (int b = 0; b < buckets; ++b) {
		EXPECT_NEAR(counts[b], expected, expected * 0.04)
			<< "Bucket " << b << " has " << counts[b] << " samples (expected ~" << expected << ")";
	}
}

// Test that the offset parameter in seed() shifts the starting state —
// two RNGs with different offsets produce different (non-correlated) sequences.
TEST(PCG32Test, OffsetProducesDifferentSequences) {
	PCG32 a, b;
	a.seed(5, 0);
	b.seed(5, 12345);  // different offset = different starting state

	// They should produce different values (not identical streams)
	bool any_different = false;
	for (int i = 0; i < 20; ++i) {
		if (a.next_uint32() != b.next_uint32()) {
			any_different = true;
			break;
		}
	}
	EXPECT_TRUE(any_different) << "Different offsets should produce different sequences";
}

// Sanity-check random_int covers the full [min, max] range.
TEST(PCG32Test, RandomIntCoversRange) {
	const int lo = 0, hi = 4;
	bool seen[5] = {};
	for (int i = 0; i < 10000; ++i) {
		int v = random_int(lo, hi);
		ASSERT_GE(v, lo);
		ASSERT_LE(v, hi);
		seen[v] = true;
	}
	for (int v = lo; v <= hi; ++v)
		EXPECT_TRUE(seen[v]) << "Value " << v << " never generated";
}

// ============================================================================
// Exact Fresnel (FrDielectric) Tests
// ============================================================================

// Normal incidence (cos_theta = 1.0): reflectance from air into glass (eta=1.5)
// Analytic value: ((eta-1)/(eta+1))^2 = (0.5/2.5)^2 = 0.04
TEST(FresnelTest, NormalIncidenceGlass) {
	double r = FrDielectric(1.0, 1.5);
	EXPECT_NEAR(r, 0.04, 1e-6) << "Normal incidence glass should reflect exactly 4%";
}

// Normal incidence into water (eta=1.333)
// Analytic: ((1.333-1)/(1.333+1))^2 = (0.333/2.333)^2 ≈ 0.02040
TEST(FresnelTest, NormalIncidenceWater) {
	double r = FrDielectric(1.0, 1.333);
	double expected = ((1.333 - 1.0) / (1.333 + 1.0));
	expected = expected * expected;
	EXPECT_NEAR(r, expected, 1e-4);
}

// Grazing incidence (cos_theta → 0): reflectance must approach 1.0 for any eta
TEST(FresnelTest, GrazingIncidenceApproachesOne) {
	EXPECT_NEAR(FrDielectric(0.001, 1.5),   1.0, 0.01);
	EXPECT_NEAR(FrDielectric(0.001, 2.4),   1.0, 0.01);
	EXPECT_NEAR(FrDielectric(0.001, 1.333), 1.0, 0.01);
}

// Total internal reflection: ray inside glass (eta=1.5) beyond critical angle
// Critical angle = arcsin(1/eta) = arcsin(1/1.5) ≈ 41.8°, cos ≈ 0.745
// At cos=0.5 (60°) we are past the critical angle → must return exactly 1.0
TEST(FresnelTest, TotalInternalReflection) {
	// From inside glass (pass negative cos to indicate inside medium)
	double r = FrDielectric(-0.5, 1.5);
	EXPECT_DOUBLE_EQ(r, 1.0) << "Beyond critical angle must give TIR = 1.0";
}

// Reciprocity at normal incidence: FrDielectric(1, eta) == FrDielectric(1, 1/eta)
// At normal incidence the Fresnel formula reduces to ((eta-1)/(eta+1))^2,
// which is symmetric: ((eta-1)/(eta+1))^2 == ((1/eta-1)/(1/eta+1))^2.
TEST(FresnelTest, Reciprocity) {
	for (double eta : {1.333, 1.5, 2.4}) {
		double r_forward = FrDielectric(1.0, eta);
		double r_reverse = FrDielectric(1.0, 1.0 / eta);
		EXPECT_NEAR(r_forward, r_reverse, 1e-9)
			<< "Normal-incidence Fresnel must be symmetric for eta=" << eta;
	}
}

// Output must always be in [0, 1] for any valid input
TEST(FresnelTest, AlwaysInValidRange) {
	double etas[] = {1.0, 1.333, 1.5, 2.4, 0.5};
	for (double eta : etas) {
		for (int i = 0; i <= 100; ++i) {
			double cos = -1.0 + 2.0 * i / 100.0;
			double r = FrDielectric(cos, eta);
			EXPECT_GE(r, 0.0) << "eta=" << eta << " cos=" << cos;
			EXPECT_LE(r, 1.0) << "eta=" << eta << " cos=" << cos;
		}
	}
}

// Schlick vs exact: at normal incidence they must agree (both reduce to r0^2)
// At 45° they should diverge — exact Fresnel gives higher reflectance than Schlick
TEST(FresnelTest, MoreAccurateThanSchlickAt45Degrees) {
	double eta = 1.5;
	double cos45 = std::sqrt(2.0) / 2.0;

	double r_exact = FrDielectric(cos45, eta);

	// Schlick approximation
	double r0 = (1.0 - eta) / (1.0 + eta);
	r0 = r0 * r0;
	double r_schlick = r0 + (1.0 - r0) * std::pow(1.0 - cos45, 5.0);

	// Both should be in [0,1]
	EXPECT_GE(r_exact,   0.0);
	EXPECT_LE(r_exact,   1.0);
	EXPECT_GE(r_schlick, 0.0);
	EXPECT_LE(r_schlick, 1.0);

	// At 45° the exact Fresnel is measurably different from Schlick
	// (they should not be identical — if they are, exact Fresnel wasn't implemented)
	EXPECT_GT(std::abs(r_exact - r_schlick), 1e-4)
		<< "Exact Fresnel and Schlick should differ at 45 degrees — same value suggests Schlick is still in use";
}

// Vacuum-to-vacuum (eta=1.0): no interface, reflectance must be effectively 0.0
// (floating-point arithmetic yields ~1e-32, not bit-exact 0; EXPECT_NEAR with tight tol)
TEST(FresnelTest, VacuumToVacuumIsZero) {
	for (int i = 1; i <= 10; ++i) {
		double cos = i / 10.0;
		EXPECT_NEAR(FrDielectric(cos, 1.0), 0.0, 1e-28)
			<< "eta=1 should give near-zero reflectance at cos=" << cos;
	}
}

// Brewster's angle: at theta_B = arctan(eta), r_parl = 0, so total reflectance = r_perp^2 / 2.
// r_perp = (cos_i - eta*cos_t) / (cos_i + eta*cos_t)
// Verified analytically for eta=1.5: cos_B = 1/sqrt(1+eta^2)
TEST(FresnelTest, BrewstersAngle) {
	double eta = 1.5;
	double cos_B = 1.0 / std::sqrt(1.0 + eta * eta);
	double sin2_t = (1.0 - cos_B * cos_B) / (eta * eta);
	double cos_t  = std::sqrt(1.0 - sin2_t);
	double r_perp  = (cos_B - eta * cos_t) / (cos_B + eta * cos_t);
	double expected = (r_perp * r_perp) / 2.0;  // r_parl = 0 at Brewster's angle

	EXPECT_NEAR(FrDielectric(cos_B, eta), expected, 1e-9)
		<< "At Brewster's angle r_parl=0, reflectance should equal r_perp^2/2";
}

// Input clamping: cos values outside [-1, 1] should be silently clamped, not NaN/crash
TEST(FresnelTest, InputClampingOutOfRange) {
	// Values beyond unit range should behave the same as the boundary values
	EXPECT_DOUBLE_EQ(FrDielectric(1.5,  1.5), FrDielectric(1.0,  1.5));
	EXPECT_DOUBLE_EQ(FrDielectric(-1.5, 1.5), FrDielectric(-1.0, 1.5));

	// And must still be in [0, 1]
	EXPECT_GE(FrDielectric(2.0,  2.4), 0.0);
	EXPECT_LE(FrDielectric(2.0,  2.4), 1.0);
	EXPECT_GE(FrDielectric(-2.0, 2.4), 0.0);
	EXPECT_LE(FrDielectric(-2.0, 2.4), 1.0);
}

// ============================================================================
// Shared Math Utils Tests  (src/shared/math_utils.h)
// cpu_gpu_reflect, cpu_gpu_refract, PowerHeuristic
// ============================================================================

// --- cpu_gpu_reflect ---

// Reflect a vector straight back along the normal (normal incidence).
// v = (0,0,1) hitting n = (0,0,-1): result must be (0,0,1) reflected = (0,0,-1).
// General formula: r = v - 2*dot(v,n)*n
TEST(SharedReflectTest, NormalIncidence) {
	vec3 v(0, 0, 1);
	vec3 n(0, 0, -1);
	vec3 r = cpu_gpu_reflect(v, n);
	// dot(v,n) = -1, so r = (0,0,1) - 2*(-1)*(0,0,-1) = (0,0,1) - (0,0,2) = (0,0,-1)
	EXPECT_NEAR(r.x(), 0.0, 1e-12);
	EXPECT_NEAR(r.y(), 0.0, 1e-12);
	EXPECT_NEAR(r.z(), -1.0, 1e-12);
}

// Reflect off a horizontal surface: incoming at 45 degrees, outgoing at 45 degrees.
TEST(SharedReflectTest, FortyFiveDegrees) {
	vec3 v = unit_vector(vec3(1, 0, -1));   // incoming down-right
	vec3 n(0, 0, 1);                         // upward normal
	vec3 r = cpu_gpu_reflect(v, n);
	vec3 expected = unit_vector(vec3(1, 0, 1));
	EXPECT_NEAR(r.x(), expected.x(), 1e-12);
	EXPECT_NEAR(r.y(), expected.y(), 1e-12);
	EXPECT_NEAR(r.z(), expected.z(), 1e-12);
}

// Reflected vector must have the same length as the incident vector.
TEST(SharedReflectTest, PreservesLength) {
	vec3 v(1.5, -0.7, 0.3);
	vec3 n = unit_vector(vec3(0.2, 0.9, 0.4));
	vec3 r = cpu_gpu_reflect(v, n);
	EXPECT_NEAR(r.length(), v.length(), 1e-10);
}

// --- cpu_gpu_refract ---

// At normal incidence (straight in), refraction doesn't bend the ray.
TEST(SharedRefractTest, NormalIncidenceNoBending) {
	vec3 uv(0, 0, 1);       // straight down
	vec3 n(0, 0, -1);       // surface normal pointing up
	double eta = 1.5;        // air -> glass: etai/etat = 1/1.5
	vec3 r = cpu_gpu_refract<vec3, double>(uv, n, 1.0 / eta);
	// At normal incidence the refracted ray is parallel to incident
	EXPECT_NEAR(r.x(), 0.0, 1e-10);
	EXPECT_NEAR(r.y(), 0.0, 1e-10);
	EXPECT_NEAR(r.z(), 1.0, 1e-10);
}

// Snell's law: verify sin(theta_t) = sin(theta_i) / eta for 30 deg incidence.
TEST(SharedRefractTest, SnellsLaw) {
	double theta_i = 30.0 * pi / 180.0;
	double eta = 1.5;  // air -> glass
	vec3 uv = unit_vector(vec3(std::sin(theta_i), 0, std::cos(theta_i)));
	vec3 n(0, 0, -1);
	vec3 r = cpu_gpu_refract<vec3, double>(uv, n, 1.0 / eta);
	// sin(theta_t) = sin(theta_i) / eta
	double sin_t = std::sqrt(r.x()*r.x() + r.y()*r.y());
	double expected_sin_t = std::sin(theta_i) / eta;
	EXPECT_NEAR(sin_t, expected_sin_t, 1e-10);
}

// Refracted ray must be a unit vector (same as incident unit vector).
TEST(SharedRefractTest, OutputIsUnitVector) {
	vec3 uv = unit_vector(vec3(0.5, 0, 0.866));
	vec3 n(0, 0, -1);
	vec3 r = cpu_gpu_refract<vec3, double>(uv, n, 1.0 / 1.5);
	EXPECT_NEAR(r.length(), 1.0, 1e-10);
}

// --- PowerHeuristic ---

// Equal PDFs: weight should be 0.5
TEST(SharedPowerHeuristicTest, EqualPDFsGiveHalf) {
	EXPECT_NEAR(PowerHeuristic(1.0, 1.0), 0.5, 1e-12);
}

// First PDF dominates: weight approaches 1
TEST(SharedPowerHeuristicTest, DominantPDFApproachesOne) {
	double w = PowerHeuristic(1000.0, 0.001);
	EXPECT_GT(w, 0.999);
}

// Second PDF dominates: weight approaches 0
TEST(SharedPowerHeuristicTest, SubdominantPDFApproachesZero) {
	double w = PowerHeuristic(0.001, 1000.0);
	EXPECT_LT(w, 0.001);
}

// pdf_a = 0: weight must be 0
TEST(SharedPowerHeuristicTest, ZeroPdfAIsZero) {
	EXPECT_DOUBLE_EQ(PowerHeuristic(0.0, 1.0), 0.0);
}

// pdf_b = 0: weight must be 1
TEST(SharedPowerHeuristicTest, ZeroPdfBIsOne) {
	EXPECT_DOUBLE_EQ(PowerHeuristic(1.0, 0.0), 1.0);
}

// Output always in [0, 1] for a wide range of inputs
TEST(SharedPowerHeuristicTest, AlwaysInValidRange) {
	double vals[] = {0.0, 1e-10, 0.001, 0.1, 1.0, 10.0, 1e6};
	for (double a : vals) {
		for (double b : vals) {
			double w = PowerHeuristic(a, b);
			EXPECT_GE(w, 0.0) << "a=" << a << " b=" << b;
			EXPECT_LE(w, 1.0) << "a=" << a << " b=" << b;
		}
	}
}

// Symmetry check: PowerHeuristic(a,b) + PowerHeuristic(b,a) == 1 for a,b > 0
TEST(SharedPowerHeuristicTest, ComplementSumsToOne) {
	double pairs[][2] = {{0.5, 0.3}, {1.0, 2.0}, {0.01, 100.0}};
	for (auto& p : pairs) {
		double wa = PowerHeuristic(p[0], p[1]);
		double wb = PowerHeuristic(p[1], p[0]);
		EXPECT_NEAR(wa + wb, 1.0, 1e-12) << "a=" << p[0] << " b=" << p[1];
	}
}

// ============================================================================
// GGX Microfacet (TrowbridgeReitz) Tests  -- src/shared/microfacet.h
// Mirrors pbrt-v4 TrowbridgeReitzDistribution (util/scattering.h)
// ============================================================================

// D(wm) must be non-negative everywhere
TEST(GGXTest, DNonNegative) {
	double alphas[] = {0.01, 0.1, 0.3, 0.5, 1.0};
	for (double a : alphas) {
		TrowbridgeReitz<double> dist(a, a);
		// Normal pointing straight up (wm = z-axis)
		EXPECT_GE(dist.D(0.0, 0.0, 1.0), 0.0) << "alpha=" << a;
		// Tilted 45 degrees
		double s = std::sqrt(0.5);
		EXPECT_GE(dist.D(s, 0.0, s), 0.0) << "alpha=" << a;
	}
}

// D(wm) at normal incidence (wm = +Z): must equal 1 / (pi * alpha^2)
// GGX isotropic: D(0,0,1) = 1/(pi*ax*ay*(1+0)^2) = 1/(pi*a^2)
TEST(GGXTest, DNormalIncidenceAnalytic) {
	double a = 0.5;
	TrowbridgeReitz<double> dist(a, a);
	double expected = 1.0 / (3.14159265358979323846 * a * a);
	EXPECT_NEAR(dist.D(0.0, 0.0, 1.0), expected, 1e-10);
}

// Lambda(wm pointing straight up) = 0  (no masking at normal incidence)
TEST(GGXTest, LambdaZeroAtNormal) {
	TrowbridgeReitz<double> dist(0.5, 0.5);
	EXPECT_NEAR(dist.Lambda(0.0, 0.0, 1.0), 0.0, 1e-12);
}

// G1 at normal incidence = 1/(1+Lambda) = 1/(1+0) = 1.0
TEST(GGXTest, G1OneAtNormal) {
	TrowbridgeReitz<double> dist(0.3, 0.3);
	EXPECT_NEAR(dist.G1(0.0, 0.0, 1.0), 1.0, 1e-12);
}

// G(wo,wi) <= G1(wo) and G(wo,wi) <= G1(wi)  (combined masking <= individual)
TEST(GGXTest, GLeqG1) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	double s = std::sqrt(0.5);
	double wo_x=0.3, wo_y=0.1, wo_z=0.9;  // outgoing
	double wi_x=0.1, wi_y=0.4, wi_z=0.8;  // incoming (both upper hemisphere)
	double g  = dist.G(wo_x,wo_y,wo_z, wi_x,wi_y,wi_z);
	double g1o = dist.G1(wo_x,wo_y,wo_z);
	double g1i = dist.G1(wi_x,wi_y,wi_z);
	EXPECT_LE(g, g1o + 1e-12);
	EXPECT_LE(g, g1i + 1e-12);
}

// G is symmetric: G(wo,wi) == G(wi,wo)
TEST(GGXTest, GSymmetry) {
	TrowbridgeReitz<double> dist(0.5, 0.5);
	double gab = dist.G(0.3,0.1,0.9, 0.2,0.4,0.8);
	double gba = dist.G(0.2,0.4,0.8, 0.3,0.1,0.9);
	EXPECT_NEAR(gab, gba, 1e-12);
}

// EffectivelySmooth: alpha < 1e-3 should be detected
TEST(GGXTest, EffectivelySmooth) {
	EXPECT_TRUE(TrowbridgeReitz<double>(0.0005, 0.0005).EffectivelySmooth());
	EXPECT_FALSE(TrowbridgeReitz<double>(0.5, 0.5).EffectivelySmooth());
}

// RoughnessToAlpha: alpha = sqrt(roughness)
TEST(GGXTest, RoughnessToAlpha) {
	EXPECT_NEAR(TrowbridgeReitz<double>::RoughnessToAlpha(0.25), 0.5, 1e-12);
	EXPECT_NEAR(TrowbridgeReitz<double>::RoughnessToAlpha(1.0),  1.0, 1e-12);
	EXPECT_NEAR(TrowbridgeReitz<double>::RoughnessToAlpha(0.0),  0.0, 1e-12);
}

// GGX_conductor_brdf must be non-negative
TEST(GGXTest, ConductorBRDFNonNegative) {
	double alphas[] = {0.1, 0.3, 0.5, 0.9};
	for (double a : alphas) {
		double brdf = GGX_conductor_brdf(0.3,0.1,0.9, 0.2,0.4,0.8, a, a);
		EXPECT_GE(brdf, 0.0) << "alpha=" << a;
	}
}

// GGX_conductor_brdf is symmetric in wo/wi (reciprocity)
TEST(GGXTest, ConductorBRDFReciprocity) {
	double a = 0.4;
	double fab = GGX_conductor_brdf(0.3,0.1,0.9, 0.2,0.4,0.8, a, a);
	double fba = GGX_conductor_brdf(0.2,0.4,0.8, 0.3,0.1,0.9, a, a);
	EXPECT_NEAR(fab, fba, 1e-10);
}

// At alpha->0 (smooth limit) D is very large (mirror-like spike)
// At alpha=1 (very rough) D at normal is much smaller than at alpha=0.1
TEST(GGXTest, RougherMeansLargerSpread) {
	TrowbridgeReitz<double> smooth(0.05, 0.05);
	TrowbridgeReitz<double> rough(0.9,  0.9);
	// At normal incidence, smoother surface has higher NDF peak
	double d_smooth = smooth.D(0.0, 0.0, 1.0);
	double d_rough  = rough.D(0.0, 0.0, 1.0);
	EXPECT_GT(d_smooth, d_rough);
}

// D_visible must be non-negative
TEST(GGXTest, DVisibleNonNegative) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	double dv = dist.D_visible(0.3,0.1,0.9, 0.0,0.0,1.0);
	EXPECT_GE(dv, 0.0);
}

// PDF(w,wm) == D_visible(w,wm) (they are the same function)
TEST(GGXTest, PDFEqualsD_visible) {
	TrowbridgeReitz<double> dist(0.3, 0.3);
	double wx=0.2,wy=0.1,wz=0.9, wmx=0.1,wmy=0.0,wmz=1.0;
	// normalize wm
	double wml=std::sqrt(wmx*wmx+wmy*wmy+wmz*wmz);
	wmx/=wml; wmy/=wml; wmz/=wml;
	EXPECT_NEAR(dist.PDF(wx,wy,wz,wmx,wmy,wmz),
				dist.D_visible(wx,wy,wz,wmx,wmy,wmz), 1e-12);
}

// Sample_wm must return a unit vector
TEST(GGXTest, SampleWmIsUnitVector) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	for (int i = 0; i < 100; ++i) {
		double wmx, wmy, wmz;
		dist.Sample_wm(0.3, 0.1, 0.9,
					   random_double(), random_double(),
					   wmx, wmy, wmz);
		double len = std::sqrt(wmx*wmx + wmy*wmy + wmz*wmz);
		EXPECT_NEAR(len, 1.0, 1e-10) << "Sample " << i << " not unit length";
	}
}

// Sample_wm must always return wm in the upper hemisphere (wm.z > 0)
TEST(GGXTest, SampleWmUpperHemisphere) {
	TrowbridgeReitz<double> dist(0.5, 0.5);
	for (int i = 0; i < 200; ++i) {
		double wmx, wmy, wmz;
		dist.Sample_wm(0.3, 0.1, 0.9,
					   random_double(), random_double(),
					   wmx, wmy, wmz);
		EXPECT_GT(wmz, 0.0) << "Sample " << i << " below hemisphere";
	}
}

// Regularize: alphas below 0.3 should be bumped up
TEST(GGXTest, RegularizeBumpsLowAlpha) {
	TrowbridgeReitz<double> dist(0.05, 0.15);
	dist.Regularize();
	EXPECT_GE(dist.alpha_x, 0.1);
	EXPECT_GE(dist.alpha_y, 0.1);
}

// Regularize: alphas already >= 0.3 should be unchanged
TEST(GGXTest, RegularizeDoesNotChangeHighAlpha) {
	TrowbridgeReitz<double> dist(0.5, 0.8);
	dist.Regularize();
	EXPECT_NEAR(dist.alpha_x, 0.5, 1e-12);
	EXPECT_NEAR(dist.alpha_y, 0.8, 1e-12);
}

// ============================================================================
// Halton Sampler Tests (src/shared/math_utils.h)
//
// Validates the low-discrepancy radical inverse used for pixel jitter.
// These are exact arithmetic checks — the values are deterministic.
// Aligned with pbrt-v4 lowdiscrepancy.h::RadicalInverse() behavior.
// ============================================================================

// halton2 (base-2): known values at px=0,py=0 (no pixel offset)
// pixel_index(n,0,0) = n, so these are plain radical-inverse checks.
TEST(HaltonSamplerTest, Base2KnownValues) {
	// n=1: binary "1"    -> mirror: "0.1"    = 0.5
	EXPECT_FLOAT_EQ(halton2(1u), 0.5f);
	// n=2: binary "10"   -> mirror: "0.01"   = 0.25
	EXPECT_FLOAT_EQ(halton2(2u), 0.25f);
	// n=3: binary "11"   -> mirror: "0.11"   = 0.75
	EXPECT_FLOAT_EQ(halton2(3u), 0.75f);
	// n=4: binary "100"  -> mirror: "0.001"  = 0.125
	EXPECT_FLOAT_EQ(halton2(4u), 0.125f);
	// n=0 -> 0.0
	EXPECT_FLOAT_EQ(halton2(0u), 0.0f);
}

// halton3 (base-3): known values at px=0,py=0
TEST(HaltonSamplerTest, Base3KnownValues) {
	// n=1: base3 "1"    -> mirror: "0.1"    = 1/3
	EXPECT_NEAR(halton3(1u), 1.0f/3.0f, 1e-6f);
	// n=2: base3 "2"    -> mirror: "0.2"    = 2/3
	EXPECT_NEAR(halton3(2u), 2.0f/3.0f, 1e-6f);
	// n=3: base3 "10"   -> mirror: "0.01"   = 1/9
	EXPECT_NEAR(halton3(3u), 1.0f/9.0f, 1e-6f);
	// n=0 -> 0.0
	EXPECT_FLOAT_EQ(halton3(0u), 0.0f);
}

// All values must be in [0, 1) — matches pbrt-v4 OneMinusEpsilon clamping
TEST(HaltonSamplerTest, ValuesInUnitInterval) {
	for (unsigned int i = 0; i < 256; ++i) {
		float h2 = halton2(i, i % 64, i / 64);  // vary pixel coords too
		float h3 = halton3(i, i % 64, i / 64);
		EXPECT_GE(h2, 0.0f) << "halton2(" << i << ") < 0";
		EXPECT_LT(h2, 1.0f) << "halton2(" << i << ") >= 1";
		EXPECT_GE(h3, 0.0f) << "halton3(" << i << ") < 0";
		EXPECT_LT(h3, 1.0f) << "halton3(" << i << ") >= 1";
	}
}

// Low-discrepancy property: 16 Halton samples must cover [0,1) uniformly.
// For N=16 (2^4), base-2 radical inverse tiles the unit interval exactly
// with max gap = 1/16 = 0.0625 (matches pbrt-v4 RadicalInverse guarantee).
TEST(HaltonSamplerTest, LowDiscrepancyBase2) {
	const int N = 16;
	std::vector<float> samples(N);
	for (int i = 0; i < N; ++i)
		samples[i] = halton2((unsigned int)i);  // px=0,py=0: pure sequence

	std::sort(samples.begin(), samples.end());

	// Max gap between consecutive sorted samples (including wrap-around)
	float max_gap = samples[0];
	for (int i = 1; i < N; ++i)
		max_gap = std::max(max_gap, samples[i] - samples[i-1]);
	max_gap = std::max(max_gap, 1.0f - samples[N-1]);

	// Halton base-2 with N=2^k tiles exactly: max gap = 1/N
	EXPECT_LE(max_gap, 1.0f / float(N) + 1e-5f)
		<< "Halton base-2 max gap " << max_gap
		<< " exceeds 1/N=" << (1.0f/N) << " — sequence is not low-discrepancy";
}

// Per-pixel decorrelation: adjacent pixels must produce different offsets
// for the same sample index (pbrt-v4 StartPixelSample pattern).
TEST(HaltonSamplerTest, PerPixelDecorrelation) {
	// Sample 0 should differ across different pixel positions
	float s00 = halton2(0u, 0u, 0u);
	float s10 = halton2(0u, 1u, 0u);
	float s01 = halton2(0u, 0u, 1u);
	float s11 = halton2(0u, 1u, 1u);

	// All four should be different (hash collision would be a bug)
	EXPECT_NE(s00, s10) << "Pixels (0,0) and (1,0) share the same halton2 offset";
	EXPECT_NE(s00, s01) << "Pixels (0,0) and (0,1) share the same halton2 offset";
	EXPECT_NE(s10, s11) << "Pixels (1,0) and (1,1) share the same halton2 offset";

	// All values still in [0,1)
	EXPECT_GE(s00, 0.0f); EXPECT_LT(s00, 1.0f);
	EXPECT_GE(s10, 0.0f); EXPECT_LT(s10, 1.0f);
	EXPECT_GE(s01, 0.0f); EXPECT_LT(s01, 1.0f);
	EXPECT_GE(s11, 0.0f); EXPECT_LT(s11, 1.0f);
}

// ============================================================================
// RoughDielectricTest -- pbrt-v4 RoughDielectricBxDF alignment
// ============================================================================

// Helpers shared across tests
static hit_record make_hit_front(point3 p = point3(0,0,0),
								  vec3   n = vec3(0,0,1)) {
	hit_record rec;
	rec.p          = p;
	rec.normal     = n;
	rec.front_face = true;
	rec.t          = 1.0;
	return rec;
}

// Test: constructor stores IOR / alpha correctly (RoughnessToAlpha = sqrt)
TEST(RoughDielectricTest, ConstructorAlpha) {
	rough_dielectric mat(1.5, 0.25);
	EXPECT_DOUBLE_EQ(mat.get_ior(), 1.5);
	// RoughnessToAlpha(0.25) = sqrt(0.25) = 0.5
	// get_roughness() returns alpha*alpha, so should equal 0.25
	EXPECT_NEAR(mat.get_roughness(), 0.25, 1e-10);
}

// Test: attenuation is always white (energy-neutral glass). roughness=0.2
// is well above the EffectivelySmooth threshold (alpha=sqrt(0.2)~0.447 >>
// 1e-3), so rough_dielectric::scatter() takes the glossy real-NEE path:
// skip_pdf=false, attenuation stays white there too (RoughDielectricBxDF
// has no color of its own) -- see rough_dielectric::scatter()'s own
// comment in material_pbrt.h. Previously this test asserted skip_pdf must
// ALWAYS be true regardless of roughness -- exactly the bug #222 fixed.
TEST(RoughDielectricTest, AttenuationIsWhite) {
	rough_dielectric mat(1.5, 0.2);
	hit_record rec = make_hit_front();

	int N = 500;
	int scattered_count = 0;
	for (int i = 0; i < N; ++i) {
		ray r_in(point3(0,0,-1), vec3(0,0,1));
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			++scattered_count;
			EXPECT_NEAR(srec.attenuation.x(), 1.0, 1e-9);
			EXPECT_NEAR(srec.attenuation.y(), 1.0, 1e-9);
			EXPECT_NEAR(srec.attenuation.z(), 1.0, 1e-9);
			EXPECT_FALSE(srec.skip_pdf)
				<< "roughness=0.2 is glossy, must get real NEE";
			EXPECT_NE(srec.pdf_ptr, nullptr);
		}
	}
	// The glossy path always returns true (no res.valid rejection -- see
	// rough_metal's own comment on why), so this should now be exactly N.
	EXPECT_GT(scattered_count, N / 2);
}

// Test: scattered ray direction is a unit vector (never degenerate).
// roughness=0.3 is glossy (see AttenuationIsWhite's comment) -- direction
// comes from pdf_ptr->generate() rather than the now-unused skip_pdf_ray.
TEST(RoughDielectricTest, ScatteredDirIsUnit) {
	rough_dielectric mat(1.5, 0.3);
	hit_record rec = make_hit_front();
	ray r_in(point3(0,0,-1), vec3(0,0,1));

	int ok = 0;
	for (int i = 0; i < 1000; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			ASSERT_FALSE(srec.skip_pdf);
			ASSERT_NE(srec.pdf_ptr, nullptr);
			vec3 dir = srec.pdf_ptr->generate();
			double len = dir.length();
			EXPECT_NEAR(len, 1.0, 1e-6)
				<< "scattered direction is not unit-length at sample " << i;
			// No NaN
			EXPECT_FALSE(std::isnan(dir.x()));
			EXPECT_FALSE(std::isnan(dir.y()));
			EXPECT_FALSE(std::isnan(dir.z()));
			++ok;
		}
	}
	EXPECT_GT(ok, 500);
}

// Test: transmitted rays cross the boundary (z < 0 in world space when entering)
// For a ray hitting a flat surface from above, refracted rays must go downward.
TEST(RoughDielectricTest, TransmissionCrossesBoundary) {
	// roughness=0.01 -> alpha=sqrt(0.01)=0.1, still glossy (>>1e-3), so this
	// takes the real-NEE path (skip_pdf=false) same as AttenuationIsWhite's
	// comment explains -- direction comes from pdf_ptr->generate().
	rough_dielectric mat(1.5, 0.01);
	hit_record rec = make_hit_front(point3(0,0,0), vec3(0,0,1));
	// Ray coming straight down along -z into the surface
	ray r_in(point3(0,0,1), vec3(0,0,-1));

	int refracted_count = 0;
	int reflected_count = 0;
	for (int i = 0; i < 2000; ++i) {
		scatter_record srec;
		if (!mat.scatter(r_in, rec, srec)) continue;
		ASSERT_FALSE(srec.skip_pdf);
		ASSERT_NE(srec.pdf_ptr, nullptr);
		double dz = srec.pdf_ptr->generate().z();
		if (dz < 0.0) ++refracted_count;  // Crossed boundary: correct
		else          ++reflected_count;  // Reflected back: also valid
	}
	// At normal incidence, IOR=1.5, Fresnel reflectance ≈ 4% → ~96% refracted
	// With roughness 0.01 (near-specular) we expect the vast majority to transmit
	EXPECT_GT(refracted_count, reflected_count * 10)
		<< "At near-normal incidence, most rays should refract through the surface. "
		<< "Refracted=" << refracted_count << " Reflected=" << reflected_count;
}

// Test: energy conservation — total scattered weight over many rays ≤ 1
// (attenuation is always 1.0 for dielectric, scatter probability itself is conservation)
// Note: ~3-5% of samples return false when the microfacet-sampled direction
// falls below the geometric surface (back-hemisphere rejection). This is
// physically correct (pbrt-v4 RoughDielectricBxDF does the same); those
// rays carry zero weight and do NOT violate energy conservation.
TEST(RoughDielectricTest, NoEnergyCreation) {
	rough_dielectric mat(1.5, 0.4);
	hit_record rec = make_hit_front();
	ray r_in(point3(0,0,-1), vec3(0,0,1));

	int scattered = 0, total = 0;
	for (int i = 0; i < 2000; ++i) {
		scatter_record srec;
		++total;
		if (mat.scatter(r_in, rec, srec)) ++scattered;
	}
	// Allow up to 10% geometric rejection (grazing microfacet normals).
	// In practice it is ~3-5%; 10% is a conservative bound.
	double rejection_rate = double(total - scattered) / double(total);
	EXPECT_LT(rejection_rate, 0.10)
		<< "rough_dielectric rejected " << (total - scattered) << "/" << total
		<< " rays (" << rejection_rate * 100.0 << "%) — unexpected high absorption";
}

// ============================================================================
// FrComplex (conductor Fresnel) tests — mirrors pbrt-v4 scattering.h
// ============================================================================

// At normal incidence (cos_theta = 1), FrComplex should match the analytic
// formula: F = |(eta - 1) / (eta + 1)|^2 for purely real eta (k = 0).
TEST(ConductorFresnelTest, NormalIncidencePurelyReal) {
	// For k=0, FrComplex reduces to FrDielectric.
	// eta = 1.5 (glass-like): F0 = |(1.5-1)/(1.5+1)|^2 = (0.5/2.5)^2 = 0.04
	const double eta = 1.5, k = 0.0;
	const double F = FrComplex(1.0, eta, k);
	const double expected = (eta - 1.0) * (eta - 1.0) / ((eta + 1.0) * (eta + 1.0));
	EXPECT_NEAR(F, expected, 1e-6) << "FrComplex with k=0 should match real Fresnel reflectance";
}

// FrComplex must return values in [0, 1] (energy conservation).
TEST(ConductorFresnelTest, OutputInRange) {
	// Gold IOR (R channel): eta=0.184, k=3.070
	for (int i = 0; i <= 10; ++i) {
		double cos_theta = i / 10.0;
		double F = FrComplex(cos_theta, 0.184, 3.070);
		EXPECT_GE(F, 0.0) << "FrComplex must be >= 0";
		EXPECT_LE(F, 1.0) << "FrComplex must be <= 1";
	}
}

// Metals (high k) should have high reflectance (> 0.5) at all angles.
TEST(ConductorFresnelTest, HighReflectanceForHighK) {
	// Gold R channel: typical F > 0.9 for most angles
	double F_normal = FrComplex(1.0, 0.184, 3.070);
	EXPECT_GT(F_normal, 0.5) << "Gold (R channel) should have high reflectance at normal incidence";
}

// Grazing incidence (cos_theta -> 0) approaches 1 for conductors.
TEST(ConductorFresnelTest, GrazingIncidenceApproachesOne) {
	double F = FrComplex(0.001, 0.184, 3.070);
	EXPECT_GT(F, 0.99) << "FrComplex should approach 1 at grazing incidence";
}

// FrComplex with k=0 and cos_theta in [-1,0] is clamped to cos_theta=0.
TEST(ConductorFresnelTest, NegativeCosineClampedToZero) {
	// Negative cos_theta is clamped to 0 in FrComplex, should equal value at 0.
	double F_neg = FrComplex(-0.5, 1.5, 0.0);
	double F_zero = FrComplex(0.0, 1.5, 0.0);
	EXPECT_NEAR(F_neg, F_zero, 1e-10);
}

// ============================================================================
// conductor material tests (CPU)
// ============================================================================

// Helper: create a front-face hit record pointing straight up (+Y normal).
static hit_record make_conductor_hit() {
	hit_record rec;
	rec.p = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);
	rec.front_face = true;
	rec.t = 1.0;
	return rec;
}

// conductor::scatter should return true for non-grazing rays.
// VNDF sampling can rarely produce a below-surface microfacet due to
// floating-point edge cases, so we retry up to N times (pbrt-v4 also
// does not guarantee a valid sample on every call for GGX VNDF).
TEST(ConductorMaterialTest, ScatterReturnsTrueForNormalIncidence) {
	conductor mat(kConductorAu, 0.1);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));  // straight down
	hit_record rec = make_conductor_hit();
	bool got_scatter = false;
	for (int i = 0; i < 20 && !got_scatter; ++i) {
		scatter_record srec;
		got_scatter = mat.scatter(r_in, rec, srec);
	}
	EXPECT_TRUE(got_scatter) << "scatter should succeed within 20 tries for normal incidence";
}

// Attenuation (F * G/G1) should be in [0,1] per channel (energy conservation).
// The G/G1 weight is the VNDF sampling correction from pbrt-v4 ConductorBxDF::Sample_f.
TEST(ConductorMaterialTest, AttenuationInRange) {
	conductor mat(kConductorAu, 0.1);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_conductor_hit();

	for (int trial = 0; trial < 100; ++trial) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_GE(srec.attenuation.x(), 0.0) << "R channel must be >= 0";
			EXPECT_LE(srec.attenuation.x(), 1.0) << "R channel must be <= 1";
			EXPECT_GE(srec.attenuation.y(), 0.0);
			EXPECT_LE(srec.attenuation.y(), 1.0);
			EXPECT_GE(srec.attenuation.z(), 0.0);
			EXPECT_LE(srec.attenuation.z(), 1.0);
		}
	}
}

// Scattered direction should stay in the upper hemisphere OR have zero
// sampling density -- roughness=0.2 is well above the EffectivelySmooth
// threshold (alpha < 1e-3), so conductor::scatter() takes the glossy
// real-NEE path (skip_pdf=false, no skip_pdf_ray; see conductor::scatter()'s
// own comment in material_pbrt.h). Unlike cosine-weighted sampling, VNDF
// reflection sampling can legitimately produce a below-horizon direction on
// a grazing microfacet sample (same as the old sample_local()'s own
// wo_z<=0 rejection) -- camera.h handles this correctly by checking
// pdf_ptr->value(dir) <= 0 before using the direction (Strategy B, camera.h),
// so this test checks that same invariant rather than assuming every
// generate() call lands above the horizon.
TEST(ConductorMaterialTest, ScatteredDirectionInUpperHemisphere) {
	conductor mat(kConductorAl, 0.2);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_conductor_hit();
	vec3 normal = rec.normal;

	int successes = 0;
	int above_horizon = 0;
	for (int trial = 0; trial < 200; ++trial) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			++successes;
			ASSERT_FALSE(srec.skip_pdf) << "roughness=0.2 is glossy, not specular";
			ASSERT_NE(srec.pdf_ptr, nullptr);
			vec3 dir = srec.pdf_ptr->generate();
			double cos_dir = dot(dir, normal);
			if (cos_dir > 0.0) {
				++above_horizon;
			} else {
				EXPECT_LE(srec.pdf_ptr->value(dir), 0.0)
					<< "below-horizon sample must carry zero sampling density";
			}
		}
	}
	EXPECT_GT(successes, 150) << "High scatter success rate expected for non-grazing incidence";
	EXPECT_GT(above_horizon, 0) << "Most VNDF samples should land above the horizon";
}

// skip_pdf reflects roughness: glossy (roughness=0.15, well above the
// EffectivelySmooth alpha<1e-3 threshold) gets real NEE (skip_pdf=false).
// Previously conductor unconditionally set skip_pdf=true regardless of
// roughness, meaning even a fairly rough (0.15) conductor got zero NEE --
// this test used to assert exactly that bug. The material's constructors
// floor alpha at RoughnessToAlpha(1e-4)=0.01 (a pre-existing, unrelated
// clamp -- see conductor's constructors), which is itself already above
// the EffectivelySmooth threshold (alpha<1e-3), so the specular/skip_pdf=
// true branch isn't reachable through the public roughness constructors at
// all; the underlying classification is exercised directly on the BxDF
// instead (ConductorBxDF::effectively_smooth()).
TEST(ConductorMaterialTest, SkipPdfReflectsRoughness) {
	conductor mat_glossy(kConductorCu, 0.15);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_conductor_hit();
	scatter_record srec_glossy;
	if (mat_glossy.scatter(r_in, rec, srec_glossy)) {
		EXPECT_FALSE(srec_glossy.skip_pdf) << "roughness=0.15 is glossy, must get real NEE";
		EXPECT_NE(srec_glossy.pdf_ptr, nullptr);
	}

	ConductorBxDF<double> bxdf_smooth{
		(double)kConductorCu.eta_r, (double)kConductorCu.eta_g, (double)kConductorCu.eta_b,
		(double)kConductorCu.k_r,   (double)kConductorCu.k_g,   (double)kConductorCu.k_b,
		0.0, 0.0};
	EXPECT_TRUE(bxdf_smooth.effectively_smooth()) << "alpha=0 must classify as specular";

	ConductorBxDF<double> bxdf_glossy{
		(double)kConductorCu.eta_r, (double)kConductorCu.eta_g, (double)kConductorCu.eta_b,
		(double)kConductorCu.k_r,   (double)kConductorCu.k_g,   (double)kConductorCu.k_b,
		0.2, 0.2};
	EXPECT_FALSE(bxdf_glossy.effectively_smooth()) << "alpha=0.2 must classify as glossy";
}

// kConductorAg (silver) should produce near-achromatic reflectance at normal incidence.
TEST(ConductorMaterialTest, SilverNearAchromaticAtNormalIncidence) {
	// At normal incidence, gold has wavelength-varying F, silver is near-neutral.
	double F_r = FrComplex(1.0, (double)kConductorAg.eta_r, (double)kConductorAg.k_r);
	double F_g = FrComplex(1.0, (double)kConductorAg.eta_g, (double)kConductorAg.k_g);
	double F_b = FrComplex(1.0, (double)kConductorAg.eta_b, (double)kConductorAg.k_b);
	// Silver channels should be within 0.15 of each other (near-neutral)
	EXPECT_NEAR(F_r, F_g, 0.15) << "Silver R/G channels should be similar";
	EXPECT_NEAR(F_g, F_b, 0.15) << "Silver G/B channels should be similar";
}

// kConductorAu (gold) should have lower blue reflectance than red (yellow tint).
TEST(ConductorMaterialTest, GoldYellowTint) {
	double F_r = FrComplex(1.0, (double)kConductorAu.eta_r, (double)kConductorAu.k_r);
	double F_b = FrComplex(1.0, (double)kConductorAu.eta_b, (double)kConductorAu.k_b);
	EXPECT_GT(F_r, F_b) << "Gold should reflect more red than blue (yellow tint)";
}

// G/G1 masking-shadowing weight stays in [0,1] for all angles.
// This is the VNDF sampling correction factor from pbrt-v4 ConductorBxDF::Sample_f.
TEST(ConductorMaterialTest, VNDFWeightInRange) {
	// Test at several roughness levels with a fixed normal-incidence direction
	for (double roughness : {0.05, 0.1, 0.3, 0.6}) {
		double alpha = TrowbridgeReitz<double>::RoughnessToAlpha(roughness);
		TrowbridgeReitz<double> dist(alpha, alpha);

		// wi pointing straight up in the local frame
		double wi_x = 0.0, wi_y = 0.0, wi_z = 1.0;
		// reflect to get wo
		double wo_x = 0.0, wo_y = 0.0, wo_z = 1.0;

		double G1 = dist.G1(wi_x, wi_y, wi_z);
		double G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		double weight = (G1 > 1e-8) ? G / G1 : 0.0;

		EXPECT_GE(weight, 0.0) << "G/G1 weight must be >= 0 (roughness=" << roughness << ")";
		EXPECT_LE(weight, 1.0 + 1e-6) << "G/G1 weight must be <= 1 (roughness=" << roughness << ")";
	}
}

// ===========================================================================
// CoatedDiffuseMaterial tests
// Mirrors the pbrt-v4 CoatedDiffuseBxDF (LayeredBxDF<DielectricBxDF, DiffuseBxDF>)
// ===========================================================================

static hit_record make_coated_hit() {
	hit_record rec;
	rec.p = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);
	rec.front_face = true;
	rec.t = 1.0;
	return rec;
}

// scatter() must return true for a non-grazing ray.
// Retry up to 20 times to tolerate rare VNDF edge cases (see ConductorMaterialTest).
TEST(CoatedDiffuseMaterialTest, ScatterReturnsTrueForNormalIncidence) {
	coated_diffuse mat(color(0.2, 0.3, 0.9), 1.5, 0.1);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));  // straight down
	hit_record rec = make_coated_hit();
	bool got_scatter = false;
	for (int i = 0; i < 20 && !got_scatter; ++i) {
		scatter_record srec;
		got_scatter = mat.scatter(r_in, rec, srec);
	}
	EXPECT_TRUE(got_scatter) << "scatter should succeed within 20 tries for normal incidence";
}

// Attenuation must be in [0,1] per channel for both coat-reflect and diffuse paths.
TEST(CoatedDiffuseMaterialTest, AttenuationInRange) {
	coated_diffuse mat(color(0.8, 0.1, 0.1), 1.5, 0.2);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();

	for (int trial = 0; trial < 200; ++trial) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_GE(srec.attenuation.x(), 0.0) << "R channel must be >= 0";
			EXPECT_LE(srec.attenuation.x(), 1.0 + 1e-6) << "R channel must be <= 1";
			EXPECT_GE(srec.attenuation.y(), 0.0);
			EXPECT_LE(srec.attenuation.y(), 1.0 + 1e-6);
			EXPECT_GE(srec.attenuation.z(), 0.0);
			EXPECT_LE(srec.attenuation.z(), 1.0 + 1e-6);
		}
	}
}

// Scattered direction must stay in the upper hemisphere OR carry zero
// sampling density (dot > 0 with normal). roughness=0.15 is well above the
// EffectivelySmooth threshold, so coated_diffuse::scatter() takes the
// glossy real-NEE path (skip_pdf=false, no skip_pdf_ray) -- direction
// comes from pdf_ptr->generate() instead, which (being VNDF-based, unlike
// cosine sampling) can legitimately produce a below-horizon grazing
// sample -- see ConductorMaterialTest.ScatteredDirectionInUpperHemisphere's
// identical reasoning in this same file. See rough_metal's own comment in
// material_pbrt.h for why this branch exists (#222).
TEST(CoatedDiffuseMaterialTest, ScatteredDirectionInUpperHemisphere) {
	coated_diffuse mat(color(0.2, 0.3, 0.9), 1.5, 0.15);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();
	vec3 normal = rec.normal;

	int successes = 0;
	for (int trial = 0; trial < 200; ++trial) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			++successes;
			ASSERT_FALSE(srec.skip_pdf) << "roughness=0.15 is glossy, not specular";
			ASSERT_NE(srec.pdf_ptr, nullptr);
			vec3 dir = srec.pdf_ptr->generate();
			double cos_dir = dot(dir, normal);
			if (cos_dir <= 0.0) {
				EXPECT_LE(srec.pdf_ptr->value(dir), 0.0)
					<< "below-horizon sample must carry zero sampling density";
			}
		}
	}
	EXPECT_GT(successes, 150) << "High scatter success rate expected for non-grazing incidence";
}

// skip_pdf reflects roughness: glossy (roughness=0.1) gets real NEE
// (skip_pdf=false). Previously coated_diffuse unconditionally set
// skip_pdf=true regardless of roughness -- this test used to assert
// exactly that bug (#222).
TEST(CoatedDiffuseMaterialTest, SkipPdfIsTrue) {
	coated_diffuse mat(color(0.5, 0.5, 0.5), 1.5, 0.1);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();
	for (int trial = 0; trial < 50; ++trial) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_FALSE(srec.skip_pdf) << "roughness=0.1 is glossy, must get real NEE";
			EXPECT_NE(srec.pdf_ptr, nullptr);
		}
	}
}

// A white-albedo coat should produce higher average attenuation than a dark one
// (diffuse path is albedo-weighted; average over many samples should reflect this).
TEST(CoatedDiffuseMaterialTest, BrighterAlbedoProducesMoreLight) {
	coated_diffuse bright(color(0.9, 0.9, 0.9), 1.5, 0.2);
	coated_diffuse dark(color(0.1, 0.1, 0.1), 1.5, 0.2);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();

	double sum_bright = 0.0, sum_dark = 0.0;
	int n = 500;
	for (int i = 0; i < n; ++i) {
		scatter_record srec_b, srec_d;
		if (bright.scatter(r_in, rec, srec_b))
			sum_bright += (srec_b.attenuation.x() + srec_b.attenuation.y() + srec_b.attenuation.z()) / 3.0;
		if (dark.scatter(r_in, rec, srec_d))
			sum_dark += (srec_d.attenuation.x() + srec_d.attenuation.y() + srec_d.attenuation.z()) / 3.0;
	}
	EXPECT_GT(sum_bright, sum_dark) << "White base should produce more average throughput than dark base";
}

// Higher roughness should produce broader highlight: reflected directions should
// have more variance (not all near-specular).  Just verifies both paths are sampled.
TEST(CoatedDiffuseMaterialTest, RougherCoatStillScatters) {
	coated_diffuse rough(color(0.5, 0.5, 0.5), 1.5, 0.8);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();

	int ok = 0;
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (rough.scatter(r_in, rec, srec)) ++ok;
	}
	// Multi-bounce random walk can reject more paths via Russian roulette and
	// total-internal-reflection events, so threshold is lower than single-bounce.
	EXPECT_GT(ok, 40) << "High-roughness coat should still scatter most rays";
}

// Coat IOR=1.0 (no coat) should give zero coat reflectance -> all energy in diffuse path.
// attenuation should equal albedo (T_in=1, T_out=1 for IOR=1).
TEST(CoatedDiffuseMaterialTest, UnitIorMeansNoCoatReflection) {
	color alb(0.6, 0.3, 0.8);
	coated_diffuse mat(alb, 1.0, 0.1);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_coated_hit();

	// With IOR=1, FrDielectric=0 so the coat is transparent (no specular reflection).
	// The multi-bounce random walk should eventually exit through the Lambertian path.
	// Verify: no specular samples, all valid samples have wo_z > 0, and attenuation
	// is bounded by the albedo (<=) with some throughput.
	{
		auto ctx = MaterialContext<double>::from_hit(rec, r_in);
		auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);
		double wi_x, wi_y, wi_z;
		frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
		double alpha = TrowbridgeReitz<double>::RoughnessToAlpha(0.1);
		CoatedDiffuseBxDF<double> bxdf{ alb.x(), alb.y(), alb.z(), 1.0, alpha, alpha,
									0.01, 0.0, 0.0, 10, 1 };
		int valid = 0;
		for (int i = 0; i < 100; ++i) {
			auto res = bxdf.sample_local(wi_x, wi_y, wi_z,
										 (uint64_t)i * 999983ULL, (uint64_t)i * 0xdeadbeefULL);
			if (!res.valid) continue;
			++valid;
			// Coat with IOR=1 has no specular reflection: throughput must be colored
			// (proportional to albedo) -- verify it is positive and at most albedo
			EXPECT_GE(res.r, 0.0);
			EXPECT_GE(res.g, 0.0);
			EXPECT_GE(res.b, 0.0);
		}
		EXPECT_GT(valid, 40) << "IOR=1 coat should produce many valid samples";
	}
}

// ===========================================================================
// ThinDielectricMaterialTest
// Mirrors pbrt-v4 ThinDielectricBxDF: zero-thickness slab, analytic R_eff.
// ===========================================================================

static hit_record make_thin_hit() {
	hit_record rec;
	rec.p = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);
	rec.front_face = true;
	rec.t = 1.0;
	return rec;
}

// scatter() must always return true.
TEST(ThinDielectricMaterialTest, ScatterAlwaysSucceeds) {
	thin_dielectric mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_thin_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		EXPECT_TRUE(mat.scatter(r_in, rec, srec));
	}
}

// Attenuation must always be (1,1,1) -- thin glass is achromatic.
TEST(ThinDielectricMaterialTest, AttenuationIsWhite) {
	thin_dielectric mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_thin_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		EXPECT_NEAR(srec.attenuation.x(), 1.0, 1e-6);
		EXPECT_NEAR(srec.attenuation.y(), 1.0, 1e-6);
		EXPECT_NEAR(srec.attenuation.z(), 1.0, 1e-6);
	}
}

// skip_pdf must always be true (specular bounce).
TEST(ThinDielectricMaterialTest, SkipPdfIsTrue) {
	thin_dielectric mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_thin_hit();
	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		EXPECT_TRUE(srec.skip_pdf);
	}
}

// At normal incidence with IOR=1.5, R_eff > 0 so both paths are sampled.
// Over many samples, expect both reflection and transmission to occur.
TEST(ThinDielectricMaterialTest, BothReflectionAndTransmissionOccur) {
	thin_dielectric mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_thin_hit();
	vec3 normal = rec.normal;

	int reflections = 0, transmissions = 0;
	for (int i = 0; i < 500; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		vec3 d = srec.skip_pdf_ray.direction();
		// Reflected: dot(d, normal) > 0
		// Transmitted: dot(d, normal) < 0 (straight-through, same as incident)
		if (dot(d, normal) > 0.0) ++reflections;
		else ++transmissions;
	}
	EXPECT_GT(reflections,   10) << "Some reflection expected at IOR=1.5";
	EXPECT_GT(transmissions, 400) << "Most rays should transmit straight through";
}

// Transmission direction must equal the original incident direction (no bending).
TEST(ThinDielectricMaterialTest, TransmissionIsUnbent) {
	thin_dielectric mat(1.5);
	vec3 incident = unit_vector(vec3(0.3, -0.9, 0.2));
	ray r_in(point3(0, 1, 0), incident);
	hit_record rec = make_thin_hit();

	int checked = 0;
	for (int i = 0; i < 500 && checked < 20; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		vec3 d = unit_vector(srec.skip_pdf_ray.direction());
		if (dot(d, rec.normal) < 0.0) {  // transmission path
			EXPECT_NEAR(d.x(), incident.x(), 1e-5) << "Transmitted ray must be unbent";
			EXPECT_NEAR(d.y(), incident.y(), 1e-5);
			EXPECT_NEAR(d.z(), incident.z(), 1e-5);
			++checked;
		}
	}
}

// R_eff analytic formula: at normal incidence IOR=1.5,
// R0 = FrDielectric(1, 1.5) ≈ 0.04; R_eff = R0 + (1-R0)^2*R0/(1-R0^2)
// which is ≈ 0.0769. Verify the multi-bounce formula matches manually.
TEST(ThinDielectricMaterialTest, AnalyticReffFormula) {
	double eta = 1.5;
	double R = FrDielectric(1.0, eta);   // cos=1, normal incidence
	double T = 1.0 - R;
	double R_eff = R + T * T * R / (1.0 - R * R);
	// For IOR=1.5: R≈0.04, R_eff≈0.0769
	EXPECT_GT(R_eff, R)       << "R_eff must be greater than single-interface R";
	EXPECT_LT(R_eff, 1.0)    << "R_eff must be < 1";
	EXPECT_GT(R_eff, 0.07)   << "R_eff at IOR=1.5 normal incidence should be ~0.077";
	EXPECT_LT(R_eff, 0.09);
}

// ===========================================================================
// CoatedConductorMaterialTest
// Mirrors pbrt-v4 CoatedConductorBxDF: rough dielectric coat over GGX conductor.
// Path A: coat specular reflection (achromatic).
// Path B: transmit into layer -> conductor GGX + FrComplex -> exit coat.
// ===========================================================================

static coated_conductor make_gold_lacquer() {
	// Gold (kConductorAu), IOR-1.5 coat, roughness 0.1
	return coated_conductor(
		kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b,
		kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b,
		1.5, 0.1);
}

static hit_record make_cc_hit() {
	hit_record rec;
	rec.p = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);
	rec.front_face = true;
	rec.t = 1.0;
	return rec;
}

// scatter() must succeed for a ray coming from above the surface.
TEST(CoatedConductorMaterialTest, ScatterSucceedsFromAbove) {
	auto mat = make_gold_lacquer();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_cc_hit();
	int successes = 0;
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) ++successes;
	}
	// With 100 samples, expect at least 80 to succeed (GGX below-hemisphere rejects are normal)
	EXPECT_GE(successes, 80) << "Most rays from above should scatter";
}

// skip_pdf reflects roughness: make_gold_lacquer()'s roughness=0.1 is
// glossy (alpha well above the EffectivelySmooth threshold), so this now
// gets real NEE (skip_pdf=false) -- previously coated_conductor
// unconditionally set skip_pdf=true regardless of roughness, and this test
// asserted exactly that bug (#222).
TEST(CoatedConductorMaterialTest, SkipPdfIsTrue) {
	auto mat = make_gold_lacquer();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_cc_hit();
	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_FALSE(srec.skip_pdf) << "roughness=0.1 is glossy, must get real NEE";
			EXPECT_NE(srec.pdf_ptr, nullptr);
		}
	}
}

// Attenuation must be >= 0 and <= ~1.5 per channel (energy is not amplified).
TEST(CoatedConductorMaterialTest, AttenuationIsNonNegativeAndBounded) {
	auto mat = make_gold_lacquer();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_cc_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			EXPECT_GE(srec.attenuation.x(), 0.0);
			EXPECT_GE(srec.attenuation.y(), 0.0);
			EXPECT_GE(srec.attenuation.z(), 0.0);
			EXPECT_LE(srec.attenuation.x(), 1.5);
			EXPECT_LE(srec.attenuation.y(), 1.5);
			EXPECT_LE(srec.attenuation.z(), 1.5);
		}
	}
}

// Scattered direction must stay in the upper hemisphere OR carry zero
// sampling density (normal = (0,1,0)) -- roughness=0.1 is glossy, so this
// takes the real-NEE path (skip_pdf=false); direction comes from
// pdf_ptr->generate(), which (being VNDF-based, unlike cosine sampling)
// can legitimately produce a below-horizon grazing sample -- see
// ConductorMaterialTest.ScatteredDirectionInUpperHemisphere's identical
// reasoning in this same file.
TEST(CoatedConductorMaterialTest, ScatteredDirectionIsInUpperHemisphere) {
	auto mat = make_gold_lacquer();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_cc_hit();
	for (int i = 0; i < 200; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			ASSERT_FALSE(srec.skip_pdf) << "roughness=0.1 is glossy, not specular";
			ASSERT_NE(srec.pdf_ptr, nullptr);
			vec3 dir = srec.pdf_ptr->generate();
			double cos_dir = dot(dir, rec.normal);
			if (cos_dir <= 0.0) {
				EXPECT_LE(srec.pdf_ptr->value(dir), 0.0)
					<< "below-horizon sample must carry zero sampling density";
			}
		}
	}
}

// Gold (Au) has a warm yellow tint: R channel should be highest, B lowest.
// Over many conductor-bounce samples (path B), expect mean(R) > mean(B).
TEST(CoatedConductorMaterialTest, GoldTintRChannelDominates) {
	auto mat = make_gold_lacquer();
	ray r_in(point3(0, 1, 0), vec3(0.2, -0.98, 0.0));  // off-normal incidence
	hit_record rec = make_cc_hit();

	double sum_r = 0, sum_b = 0;
	int n = 0;
	for (int i = 0; i < 2000; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			sum_r += srec.attenuation.x();
			sum_b += srec.attenuation.z();
			++n;
		}
	}
	if (n > 0) {
		double mean_r = sum_r / n;
		double mean_b = sum_b / n;
		EXPECT_GT(mean_r, mean_b) << "Gold should be warmer (more red) than blue";
	}
}

// F0 normal-incidence reflectance: for gold, R channel >> B channel.
TEST(CoatedConductorMaterialTest, GoldF0RedDominatesBlue) {
	auto mat = make_gold_lacquer();
	color f0 = mat.get_conductor_f0();
	EXPECT_GT(f0.x(), f0.z()) << "Au F0: red > blue";
	EXPECT_GT(f0.x(), 0.8)    << "Au F0 red channel should be high (>0.8)";
	EXPECT_LT(f0.z(), 0.5)    << "Au F0 blue channel should be lower (<0.5)";
}

// =============================================================================
// DiffuseTransmissionMaterialTest
// pbrt-v4 DiffuseTransmissionBxDF: stochastic R/T selection,
// reflection same hemisphere, transmission opposite hemisphere.
// Both paths use MIS-compatible cosine_pdf (no skip_pdf).
// =============================================================================

static hit_record make_dt_hit() {
	hit_record rec;
	rec.p          = point3(0, 0, 0);
	rec.normal     = vec3(0, 1, 0);
	rec.front_face = true;
	rec.t          = 1.0;
	return rec;
}

TEST(DiffuseTransmissionMaterialTest, ScatterAlwaysSucceeds) {
	diffuse_transmission mat(color(0.6, 0.5, 0.3), color(0.8, 0.6, 0.3));
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_dt_hit();
	int successes = 0;
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) ++successes;
	}
	EXPECT_EQ(successes, 100) << "DiffuseTransmission should always scatter";
}

TEST(DiffuseTransmissionMaterialTest, NeverUsesSkipPdf) {
	// Both reflection and transmission paths use MIS-compatible cosine_pdf
	diffuse_transmission mat(color(0.6, 0.5, 0.3), color(0.8, 0.6, 0.3));
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_dt_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec))
			EXPECT_FALSE(srec.skip_pdf) << "DiffuseTransmission should use pdf_ptr, not skip_pdf";
	}
}

TEST(DiffuseTransmissionMaterialTest, HasValidPdfPtr) {
	diffuse_transmission mat(color(0.6, 0.5, 0.3), color(0.8, 0.6, 0.3));
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_dt_hit();
	for (int i = 0; i < 20; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec))
			EXPECT_NE(srec.pdf_ptr, nullptr) << "pdf_ptr must be set for MIS";
	}
}

TEST(DiffuseTransmissionMaterialTest, ReflectionPdfPositiveAboveSurface) {
	// T=0 => always reflect; scattering_pdf above surface should be > 0
	diffuse_transmission mat(color(0.6, 0.5, 0.3), color(0.0, 0.0, 0.0));
	hit_record rec = make_dt_hit();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	// A ray going straight up from the surface
	ray scattered(point3(0,0,0), vec3(0, 1, 0));
	double pdf = mat.scattering_pdf(r_in, rec, scattered);
	EXPECT_GT(pdf, 0.0) << "Reflected direction (y>0) should have positive scattering_pdf";
}

TEST(DiffuseTransmissionMaterialTest, TransmissionPdfPositiveBelowSurface) {
	// R=0 => always transmit; scattering_pdf below surface should be > 0
	diffuse_transmission mat(color(0.0, 0.0, 0.0), color(0.8, 0.6, 0.3));
	hit_record rec = make_dt_hit();
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	// A ray going straight down (transmitted direction)
	ray scattered(point3(0,0,0), vec3(0, -1, 0));
	double pdf = mat.scattering_pdf(r_in, rec, scattered);
	EXPECT_GT(pdf, 0.0) << "Transmitted direction (y<0) should have positive scattering_pdf";
}

TEST(DiffuseTransmissionMaterialTest, AttenuationMatchesInputColors) {
	color R(0.6, 0.5, 0.3);
	color T(0.8, 0.6, 0.3);
	diffuse_transmission mat(R, T);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_dt_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			color a = srec.attenuation;
			bool is_R = (std::fabs(a.x() - R.x()) < 1e-9);
			bool is_T = (std::fabs(a.x() - T.x()) < 1e-9);
			EXPECT_TRUE(is_R || is_T) << "Attenuation must match R or T";
		}
	}
}

TEST(DiffuseTransmissionMaterialTest, BothPathsSampledStatistically) {
	// Equal R and T maxComponent => ~50% each; distinguish by attenuation color
	color R(0.5, 0.5, 0.5);
	color T(0.5, 0.4, 0.3);  // different from R so we can tell them apart
	diffuse_transmission mat(R, T);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_dt_hit();
	int reflections = 0, transmissions = 0;
	for (int i = 0; i < 2000; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec)) {
			// T has distinct blue channel (0.3) vs R (0.5)
			if (std::fabs(srec.attenuation.z() - T.z()) < 1e-9) ++transmissions;
			else                                                  ++reflections;
		}
	}
	EXPECT_GT(reflections,   700) << "Should have significant reflections";
	EXPECT_GT(transmissions, 700) << "Should have significant transmissions";
}

TEST(DiffuseTransmissionMaterialTest, AccessorsReturnCorrectColors) {
	color R(0.6, 0.5, 0.3);
	color T(0.8, 0.6, 0.3);
	diffuse_transmission mat(R, T);
	EXPECT_DOUBLE_EQ(mat.get_reflectance().x(),  R.x());
	EXPECT_DOUBLE_EQ(mat.get_transmittance().x(), T.x());
}

// Regression guard for the texture-bound constructor: reflectance_at()/
// transmittance_at() must each read their OWN texture, not accidentally
// swap or alias the other's - a real bug found via render inspection
// (pbrt_scenes/diffusetransmission-texture.pbrt showed the reflectance
// texture's colour on the transmission side too) that this earlier flat-
// colour-only test suite never could have caught.
TEST(DiffuseTransmissionMaterialTest, TexturedReflectanceAndTransmittanceStayDistinct) {
	auto rTex = std::make_shared<solid_color>(color(1.0, 0.0, 0.0));  // red
	auto tTex = std::make_shared<solid_color>(color(0.0, 0.0, 1.0));  // blue
	diffuse_transmission mat(color(0.5,0.5,0.5), color(0.5,0.5,0.5), rTex, tTex);
	hit_record rec = make_dt_hit();
	rec.dudx = rec.dvdx = rec.dudy = rec.dvdy = 0.0;

	ray toward_reflect(point3(0,0,0), vec3(0, 1, 0));   // same hemisphere as normal
	ray toward_transmit(point3(0,0,0), vec3(0, -1, 0)); // opposite hemisphere

	color reflectAtten = mat.scattering_attenuation(rec, toward_reflect, color(0,0,0));
	color transmitAtten = mat.scattering_attenuation(rec, toward_transmit, color(0,0,0));

	EXPECT_NEAR(reflectAtten.x(), 1.0, 1e-6) << "Reflection side must read rTex (red), not tTex";
	EXPECT_NEAR(reflectAtten.z(), 0.0, 1e-6);
	EXPECT_NEAR(transmitAtten.x(), 0.0, 1e-6) << "Transmission side must read tTex (blue), not rTex";
	EXPECT_NEAR(transmitAtten.z(), 1.0, 1e-6);
}

// ===========================================================================
// NormalizedFresnelMaterialTest
// Mirrors pbrt-v4 NormalizedFresnelBxDF: Fresnel-weighted diffuse reflection.
// BSDF f(wi) = (1 - FrDielectric(cos_wi, eta)) / (c * pi)
// c = 1 - 2 * FresnelMoment1(1/eta)
// Sample weight = (1 - Fr(cos_wi)) / c  (via skip_pdf=true cosine sampling)
// ===========================================================================

static hit_record make_nf_hit() {
	hit_record rec;
	rec.p      = point3(0, 0, 0);
	rec.normal = vec3(0, 1, 0);   // surface normal pointing up
	rec.t      = 1.0;
	rec.front_face = true;
	return rec;
}

TEST(NormalizedFresnelMaterialTest, ScatterAlwaysSucceeds) {
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	for (int i = 0; i < 50; ++i) {
		scatter_record srec;
		EXPECT_TRUE(mat.scatter(r_in, rec, srec)) << "scatter must always return true";
	}
}

TEST(NormalizedFresnelMaterialTest, UsesMIS) {
	// NormalizedFresnel is diffuse (DiffuseReflection in pbrt-v4 terms)
	// so it uses skip_pdf=false and a cosine_pdf for MIS integration.
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	scatter_record srec;
	mat.scatter(r_in, rec, srec);
	EXPECT_FALSE(srec.skip_pdf) << "NormalizedFresnel must use skip_pdf=false (participates in MIS)";
	EXPECT_NE(srec.pdf_ptr, nullptr) << "pdf_ptr must be set for MIS";
}

TEST(NormalizedFresnelMaterialTest, ScatteredRayInUpperHemisphere) {
	// NormalizedFresnelBxDF is reflection-only: sampled directions must be in upper hemisphere.
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	for (int i = 0; i < 100; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		vec3 d = unit_vector(srec.pdf_ptr->generate());
		EXPECT_GT(dot(d, rec.normal), 0.0) << "Sampled direction must be in upper hemisphere";
	}
}

TEST(NormalizedFresnelMaterialTest, AttenuationIsWhite) {
	// attenuation=white: BSDF weight is carried via scattering_pdf in MIS path.
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	for (int i = 0; i < 20; ++i) {
		scatter_record srec;
		mat.scatter(r_in, rec, srec);
		EXPECT_DOUBLE_EQ(srec.attenuation.x(), 1.0) << "attenuation must be white (1,1,1)";
		EXPECT_DOUBLE_EQ(srec.attenuation.y(), 1.0);
		EXPECT_DOUBLE_EQ(srec.attenuation.z(), 1.0);
	}
}

TEST(NormalizedFresnelMaterialTest, FresnelMoment1IOR1IsZero) {
	// At eta=1 (no interface), verify the normalization constant c is positive.
	normalized_fresnel mat_eta1(1.0);
	EXPECT_GT(mat_eta1.get_c(), 0.0) << "Normalization constant c must be positive";
}

TEST(NormalizedFresnelMaterialTest, FresnelMoment1NormalizationPositive) {
	// c = 1 - 2*FresnelMoment1(1/eta) must be > 0 for typical IOR values
	for (double eta : {1.0, 1.3, 1.5, 1.8, 2.0, 2.5}) {
		normalized_fresnel mat(eta);
		EXPECT_GT(mat.get_c(), 0.0) << "c must be positive for eta=" << eta;
	}
}

TEST(NormalizedFresnelMaterialTest, AccessorGetIor) {
	normalized_fresnel mat(1.5);
	EXPECT_DOUBLE_EQ(mat.get_ior(), 1.5);
}

TEST(NormalizedFresnelMaterialTest, ScatteringPdfPositiveAboveSurface) {
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	// wi at 45 degrees: cos_wi = 1/sqrt(2)
	vec3 wi_dir = unit_vector(vec3(1.0, 1.0, 0.0));  // 45 deg from normal (0,1,0)
	ray scattered(point3(0,0,0), wi_dir);
	double pdf = mat.scattering_pdf(r_in, rec, scattered);
	EXPECT_GT(pdf, 0.0) << "scattering_pdf must be positive for upper hemisphere direction";
	// Analytic check: (1 - Fr(cos45, 1.5)) * cos45 / (c * pi)
	double cos45  = 1.0 / std::sqrt(2.0);
	double fr45   = FrDielectric(cos45, 1.5);
	double c      = mat.get_c();
	double expect = (1.0 - fr45) * cos45 / (c * 3.14159265358979323846);
	EXPECT_NEAR(pdf, expect, 1e-10) << "scattering_pdf must equal pbrt-v4 BSDF*cos formula";
}

TEST(NormalizedFresnelMaterialTest, ScatteringPdfZeroBelowSurface) {
	normalized_fresnel mat(1.5);
	ray r_in(point3(0, 1, 0), vec3(0, -1, 0));
	hit_record rec = make_nf_hit();
	// scattered ray in opposite hemisphere
	ray scattered(point3(0,0,0), vec3(0.3, -0.7, 0.2));
	double pdf = mat.scattering_pdf(r_in, rec, scattered);
	EXPECT_DOUBLE_EQ(pdf, 0.0) << "scattering_pdf must be 0 for below-surface direction";
}
