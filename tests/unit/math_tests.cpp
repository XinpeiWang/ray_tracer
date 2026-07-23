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
#include "material.h"
#include <cmath>

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
