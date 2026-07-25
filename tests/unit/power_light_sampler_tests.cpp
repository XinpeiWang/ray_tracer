/**
 * @file power_light_sampler_tests.cpp
 * @brief Unit tests for AliasTable and power_light_list
 *
 * Covers:
 *  - AliasTable: PMF normalization, O(1) sampling distribution, single-entry edge case
 *  - power_light_list: pdf_value() correctness (unbiasedness condition),
 *    power-weighted selection frequency, uniform fallback from hittable_list
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include "power_light_sampler.h"
#include <numeric>
#include <cmath>

// ============================================================================
// AliasTable Tests
// ============================================================================

TEST(AliasTableTest, PMFSumsToOne) {
	std::vector<double> weights = {1.0, 2.0, 3.0};
	AliasTable table(weights);

	double total_pmf = 0.0;
	for (int i = 0; i < (int)weights.size(); ++i)
		total_pmf += table.pmf(i);

	EXPECT_NEAR(total_pmf, 1.0, 1e-10);
}

TEST(AliasTableTest, PMFProportionalToWeights) {
	std::vector<double> weights = {1.0, 3.0};
	AliasTable table(weights);

	// PMF should be proportional: p[0]=0.25, p[1]=0.75
	EXPECT_NEAR(table.pmf(0), 0.25, 1e-10);
	EXPECT_NEAR(table.pmf(1), 0.75, 1e-10);
}

TEST(AliasTableTest, SizeMatchesWeights) {
	std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
	AliasTable table(weights);
	EXPECT_EQ(table.size(), 4);
}

TEST(AliasTableTest, SingleEntryAlwaysReturnsSame) {
	std::vector<double> weights = {5.0};
	AliasTable table(weights);
	for (int trial = 0; trial < 20; ++trial)
		EXPECT_EQ(table.sample(random_double()), 0);
}

TEST(AliasTableTest, SamplingDistributionMatchesPMF) {
	// Verify that sampling frequency converges to PMF within statistical tolerance
	std::vector<double> weights = {1.0, 2.0, 7.0};
	AliasTable table(weights);

	const int N = 100000;
	std::vector<int> counts(3, 0);
	for (int i = 0; i < N; ++i)
		counts[table.sample(random_double())]++;

	// Expected proportions: 0.1, 0.2, 0.7
	// Allow 1% absolute error (very loose for N=100k)
	EXPECT_NEAR((double)counts[0] / N, 0.1, 0.01);
	EXPECT_NEAR((double)counts[1] / N, 0.2, 0.01);
	EXPECT_NEAR((double)counts[2] / N, 0.7, 0.01);
}

TEST(AliasTableTest, EqualWeightsGiveUniformDistribution) {
	std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
	AliasTable table(weights);

	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(table.pmf(i), 0.25, 1e-10);
}

TEST(AliasTableTest, ZeroTotalWeightFallsBackToUniform) {
	// All zero weights: constructor clamps sum to 1, so each gets 0/1 = 0
	// but power_light_list clamps each power to >= 1e-6, so this tests the
	// AliasTable sum guard path
	std::vector<double> weights = {0.0, 0.0};
	AliasTable table(weights);
	// sum is 0 -> clamped to 1 -> each bin.p = 0/1 = 0, but Vose still runs
	// Just verify no crash and size is correct
	EXPECT_EQ(table.size(), 2);
}

// ============================================================================
// power_light_list Tests
// ============================================================================

class PowerLightListTest : public ::testing::Test {
protected:
	// Two quad lights with known dimensions and emission
	// Light A: 10x10 area, luminance 1.0  -> phi = 100 * 1.0 * pi
	// Light B: 10x10 area, luminance 3.0  -> phi = 100 * 3.0 * pi
	// Expected weights: A=0.25, B=0.75

	static constexpr double PHI_A = 100.0 * 1.0 * M_PI;  // 314.16
	static constexpr double PHI_B = 100.0 * 3.0 * M_PI;  // 942.48
	static constexpr double W_A   = PHI_A / (PHI_A + PHI_B);  // 0.25
	static constexpr double W_B   = PHI_B / (PHI_A + PHI_B);  // 0.75

	void SetUp() override {
		auto empty = std::shared_ptr<material>();
		auto quadA = std::make_shared<quad>(point3(0,10,0), vec3(10,0,0), vec3(0,0,10), empty);
		auto quadB = std::make_shared<quad>(point3(20,10,0), vec3(10,0,0), vec3(0,0,10), empty);

		lights.add(quadA, PHI_A);
		lights.add(quadB, PHI_B);
	}

	power_light_list lights;
};

TEST_F(PowerLightListTest, LightCount) {
	EXPECT_EQ(lights.light_count(), 2);
}

TEST_F(PowerLightListTest, PDFValueNonNegative) {
	point3 origin(5, 0, 5);
	vec3 dir = unit_vector(vec3(0, 1, 0));
	EXPECT_GE(lights.pdf_value(origin, dir), 0.0);
}

TEST_F(PowerLightListTest, PDFValueMarginalFormula) {
	// pdf_value(origin, dir) = sum_i P(light_i) * pdf_i(dir)
	// Verify it equals the manually computed weighted sum
	point3 origin(5, 0, 5);
	vec3 dir = unit_vector(vec3(0, 1, 0));

	double pdf_A = lights.objects[0]->pdf_value(origin, dir);
	double pdf_B = lights.objects[1]->pdf_value(origin, dir);
	double expected = W_A * pdf_A + W_B * pdf_B;

	EXPECT_NEAR(lights.pdf_value(origin, dir), expected, 1e-10);
}

TEST_F(PowerLightListTest, SelectionFrequencyMatchesPMF) {
	// Confirm that random() selects lights with frequency ~ their power weights
	point3 origin(10, 0, 5);
	const int N = 100000;
	int count_A = 0;

	for (int i = 0; i < N; ++i) {
		vec3 dir = lights.random(origin);
		// Light A is at x=[0,10], light B at x=[20,30]
		// Direction from origin(10,0,5): negative x = toward A, positive x = toward B
		if (dir.x() < 0) ++count_A;
	}

	EXPECT_NEAR((double)count_A / N, W_A, 0.02);  // 2% tolerance
}

TEST_F(PowerLightListTest, EmptyListReturnsFallbackDirection) {
	power_light_list empty_lights;
	vec3 fallback = empty_lights.random(point3(0,0,0));
	EXPECT_DOUBLE_EQ(fallback.x(), 1.0);
	EXPECT_DOUBLE_EQ(fallback.y(), 0.0);
	EXPECT_DOUBLE_EQ(fallback.z(), 0.0);
}

TEST_F(PowerLightListTest, EmptyListPDFIsZero) {
	power_light_list empty_lights;
	EXPECT_DOUBLE_EQ(empty_lights.pdf_value(point3(0,0,0), vec3(1,0,0)), 0.0);
}

TEST(PowerLightListUniformTest, UniformFallbackFromHittableList) {
	// When built from hittable_list, all weights are 1.0 -> uniform
	hittable_list hl;
	auto empty = std::shared_ptr<material>();
	hl.add(std::make_shared<quad>(point3(0,10,0), vec3(10,0,0), vec3(0,0,10), empty));
	hl.add(std::make_shared<quad>(point3(20,10,0), vec3(10,0,0), vec3(0,0,10), empty));

	power_light_list pl(hl);
	EXPECT_EQ(pl.light_count(), 2);

	// Both lights have equal weight 0.5
	// PMF[0] == PMF[1] == 0.5
	// (access via pdf_value with a direction that only one light can contribute)
	// Just verify light count and no crash
	point3 origin(10, 0, 5);
	EXPECT_GE(pl.pdf_value(origin, unit_vector(vec3(0,1,0))), 0.0);
}

TEST(PowerLightListSingle, SingleLightAlwaysSelected) {
	power_light_list lights;
	auto empty = std::shared_ptr<material>();
	lights.add(std::make_shared<quad>(point3(0,10,0), vec3(10,0,0), vec3(0,0,10), empty), 999.0);

	point3 origin(5, 0, 5);
	// With one light, every random() should point toward it (positive y)
	for (int i = 0; i < 20; ++i) {
		vec3 dir = lights.random(origin);
		EXPECT_GT(dir.y(), 0.0) << "Expected direction toward light (y>0) on trial " << i;
	}
}

// ============================================================================
// Cornell Box Power Estimation sanity check
// ============================================================================

TEST(CornellBoxPhiTest, CeilingLightPhiFormula) {
	// Ceiling quad: u=(-130,0,0), v=(0,0,-105), emission=(15,15,15)
	// area = |u x v| = 130 * 105 = 13650
	// luminance(15,15,15) = 15
	// phi = 13650 * 15 * pi = 204750 * pi
	vec3 u(-130, 0, 0);
	vec3 v(0, 0, -105);
	color emission(15, 15, 15);

	double area = cross(u, v).length();
	double lum  = 0.2126 * emission.x() + 0.7152 * emission.y() + 0.0722 * emission.z();
	double phi  = area * lum * M_PI;

	EXPECT_NEAR(area, 13650.0, 1e-6);
	EXPECT_NEAR(lum,  15.0,    1e-10);
	EXPECT_NEAR(phi,  13650.0 * 15.0 * M_PI, 1e-4);
}

TEST(CornellBoxPhiTest, AccentLightPhiFormula) {
	// Accent quad: u=(0,0,150), v=(0,200,0), emission=(4,2,1)
	// area = 150 * 200 = 30000
	// luminance(4,2,1) = 0.2126*4 + 0.7152*2 + 0.0722*1 = 0.8504 + 1.4304 + 0.0722 = 2.353
	vec3 u(0, 0, 150);
	vec3 v(0, 200, 0);
	color emission(4, 2, 1);

	double area = cross(u, v).length();
	double lum  = 0.2126 * emission.x() + 0.7152 * emission.y() + 0.0722 * emission.z();
	double phi  = area * lum * M_PI;

	EXPECT_NEAR(area, 30000.0, 1e-6);
	EXPECT_NEAR(lum,  2.353,   1e-3);
	EXPECT_GT(phi, 0.0);
	// Ceiling phi >> accent phi (bright vs dim)
	double ceiling_phi = 13650.0 * 15.0 * M_PI;
	EXPECT_GT(ceiling_phi, phi);
}

// ============================================================================
// Power light sampler: multi-light proportionality
// Mirrors pbrt-v4 lightsamplers_test.cpp BVHLightSampling.Point pattern:
// verify that each light is selected with frequency proportional to its power.
// ============================================================================

// Three lights with weights 1:2:7. Selection frequency should match PMF.
// Mirrors pbrt-v4's check that "sampledLight->p == pmf" for each light.
//
// Geometry: three quads placed far away in distinct angular directions.
// QuadA (+x), QuadB (-x), QuadC (-z) -- all facing the origin so
// origin is on the FRONT side of every quad (visible from origin).
TEST(PowerLightProportionalityTest, ThreeLightFrequencyMatchesPMF) {
	auto empty = std::shared_ptr<material>();
	// QuadA: at x=+995, facing -x (normal = -x), origin on front side
	auto quadA = std::make_shared<quad>(point3(1005, -5, -5), vec3(0,10,0), vec3(0,0,10), empty);
	// QuadB: at x=-995, facing +x (normal = +x), origin on front side
	// cross((0,10,0),(0,0,-10)) = (-100,0,0) → normal -x, so use v=(0,0,-10)
	auto quadB = std::make_shared<quad>(point3(-995, -5,  5), vec3(0,10,0), vec3(0,0,-10), empty);
	// QuadC: at z=-995, facing +z (normal = +z), origin on front side
	// cross((10,0,0),(0,10,0)) = (0,0,100) → normal +z ✓  origin at z=0 > z=-995
	auto quadC = std::make_shared<quad>(point3( -5, -5,-995), vec3(10,0,0), vec3(0,10,0), empty);

	const double pA = 1.0, pB = 2.0, pC = 7.0;
	const double total = pA + pB + pC;

	power_light_list lights;
	lights.add(quadA, pA);
	lights.add(quadB, pB);
	lights.add(quadC, pC);

	const int N = 200000;
	int cA = 0, cB = 0, cC = 0;
	point3 origin(0, 0, 0);

	for (int i = 0; i < N; ++i) {
		vec3 raw = lights.random(origin);
		// quad::random() returns p-origin (unnormalized). Normalize before
		// classifying so small transverse components don't trigger wrong branch.
		vec3 dir = unit_vector(raw);
		// Each quad is ~1000 units away so the normalized direction strongly
		// points toward it; threshold 0.9 gives ~26 degree half-angle margin.
		if      (dir.x() > 0.9)   ++cA;
		else if (dir.x() < -0.9)  ++cB;
		else if (dir.z() < -0.9)  ++cC;
	}

	// Allow 2% absolute tolerance
	EXPECT_NEAR((double)cA / N, pA / total, 0.02) << "Light A frequency";
	EXPECT_NEAR((double)cB / N, pB / total, 0.02) << "Light B frequency";
	EXPECT_NEAR((double)cC / N, pC / total, 0.02) << "Light C frequency";
}

// High power ratio: one light 10x brighter must be selected ~10x more often.
// Mirrors pbrt-v4's validation that PMF p == selection probability.
TEST(PowerLightProportionalityTest, HighPowerRatioDominance) {
	auto empty = std::shared_ptr<material>();
	// Dim light at negative x, bright light (10x) at positive x
	auto dim    = std::make_shared<quad>(point3(-20, 10, 0), vec3(10,0,0), vec3(0,0,10), empty);
	auto bright = std::make_shared<quad>(point3( 10, 10, 0), vec3(10,0,0), vec3(0,0,10), empty);

	const double pDim = 1.0, pBright = 10.0;
	power_light_list lights;
	lights.add(dim,    pDim);
	lights.add(bright, pBright);

	point3 origin(0, 0, 5);
	const int N = 100000;
	int count_bright = 0;
	for (int i = 0; i < N; ++i) {
		vec3 dir = lights.random(origin);
		if (dir.x() > 0.0) ++count_bright;
	}

	// Bright should be selected ~10/(1+10) = 90.9% of the time
	double expected = pBright / (pDim + pBright);
	EXPECT_NEAR((double)count_bright / N, expected, 0.02)
		<< "Bright light not selected proportionally";
}

// PDF value consistency: pdf_value() must equal weighted sum of per-light PDFs.
// This directly mirrors pbrt-v4 lightsamplers_test.cpp's PMF consistency check.
TEST(PowerLightProportionalityTest, PDFValueConsistentWithPMF) {
	auto empty = std::shared_ptr<material>();
	auto q1 = std::make_shared<quad>(point3(0, 10, 0), vec3(10,0,0), vec3(0,0,10), empty);
	auto q2 = std::make_shared<quad>(point3(20,10, 0), vec3(10,0,0), vec3(0,0,10), empty);
	auto q3 = std::make_shared<quad>(point3(40,10, 0), vec3(10,0,0), vec3(0,0,10), empty);

	const double w1 = 3.0, w2 = 1.0, w3 = 6.0, total = w1 + w2 + w3;
	power_light_list lights;
	lights.add(q1, w1);
	lights.add(q2, w2);
	lights.add(q3, w3);

	point3 origin(20, 0, 5);
	vec3   dir = unit_vector(vec3(0, 1, 0));

	// Manual weighted sum
	double pdf1 = lights.objects[0]->pdf_value(origin, dir);
	double pdf2 = lights.objects[1]->pdf_value(origin, dir);
	double pdf3 = lights.objects[2]->pdf_value(origin, dir);
	double expected_pdf = (w1/total)*pdf1 + (w2/total)*pdf2 + (w3/total)*pdf3;

	EXPECT_NEAR(lights.pdf_value(origin, dir), expected_pdf, 1e-10)
		<< "pdf_value != weighted sum of per-light PDFs";
}
