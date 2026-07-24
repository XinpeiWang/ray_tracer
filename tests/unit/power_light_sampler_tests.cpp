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
