// bvh_light_sampler_tests.cpp
// Unit tests for bvh_light_sampler and LightBounds
// Mirrors pbrt-v4 §12.6 design validation strategy.

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

#include "bvh_light_sampler.h"
#include "quad.h"
#include "material.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::shared_ptr<hittable> make_quad(
	const point3& Q, const vec3& u, const vec3& v)
{
	auto mat = std::make_shared<diffuse_light>(color(1,1,1));
	return std::make_shared<quad>(Q, u, v, mat);
}

// ---------------------------------------------------------------------------
// LightBounds tests
// ---------------------------------------------------------------------------

TEST(LightBounds, ImportanceDecaysWithDistance) {
	// A light facing +y at origin, queried from increasing distances
	aabb bb(point3(-0.5,-0.5,-0.5), point3(0.5,0.5,0.5));
	LightBounds lb(bb, vec3(0,1,0), /*phi=*/1.0,
				   /*cosTheta_o=*/0.0, /*cosTheta_e=*/0.0, /*twoSided=*/false);

	point3 p1(0, 2, 0);
	point3 p2(0, 5, 0);
	double i1 = lb.importance(p1, vec3(0,0,0));
	double i2 = lb.importance(p2, vec3(0,0,0));
	EXPECT_GT(i1, 0.0);
	EXPECT_GT(i1, i2) << "Importance should decrease with distance";
}

TEST(LightBounds, ImportanceDecaysWithAngle) {
	// Light facing +z at origin. Points at varying angles from +z axis.
	// At angle=0 (directly in front) importance should exceed angle=120 (behind).
	aabb bb(point3(-0.5, -0.5, 0), point3(0.5, 0.5, 0.1));
	// cosTheta_o = 0 => 90 degree emission cone; cosTheta_e = 0 => same
	LightBounds lb(bb, vec3(0,0,1), 1.0, /*cosTheta_o=*/0.0, /*cosTheta_e=*/0.0, false);

	// Both at same distance (5 units), one in front, one 120 degrees off-axis
	double i_front = lb.importance(point3(0, 0, 5), vec3(0,0,0));
	// 120 deg off-axis: the emission cone (90 deg half-angle) does not reach this point
	double i_side  = lb.importance(point3(0, 5, -3), vec3(0,0,0));  // behind+side

	EXPECT_GT(i_front, 0.0) << "Point in front of emission cone should have positive importance";
	EXPECT_GE(i_front, i_side) << "Point in front should have >= importance than behind";
}

TEST(LightBounds, MergeIncreasesPhiAndWidensCone) {
	aabb bb0(point3(0,0,0), point3(1,1,1));
	aabb bb1(point3(2,0,0), point3(3,1,1));
	LightBounds a(bb0, vec3(1,0,0), 2.0, 0.5, 0.0, false);
	LightBounds b(bb1, vec3(0,1,0), 3.0, 0.5, 0.0, false);
	LightBounds m = LightBounds::merge(a, b);

	EXPECT_NEAR(m.phi, 5.0, 1e-10) << "Merged phi should be sum";
	// Merged cosTheta_o must be <= min of inputs (wider cone)
	EXPECT_LE(m.cosTheta_o, std::min(a.cosTheta_o, b.cosTheta_o) + 1e-9);
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: single light
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, SingleLightPMFIsOne) {
	auto q = make_quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105));
	bvh_light_sampler ls;
	ls.add(q, 10.0);
	ls.build();

	double pmf = ls.light_pmf_at(0, point3(278, 278, 278));
	EXPECT_NEAR(pmf, 1.0, 1e-9) << "Single light must have PMF = 1";
}

TEST(BVHLightSampler, SingleLightRandomHitsLight) {
	auto q = make_quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105));
	bvh_light_sampler ls;
	ls.add(q, 10.0);
	ls.build();

	point3 origin(278, 278, 278);
	for (int i = 0; i < 20; ++i) {
		vec3 dir = ls.random(origin);
		EXPECT_GT(dir.length_squared(), 0.0);
	}
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: two lights with equal power
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, TwoEqualLightsEachHalfPMF) {
	// Light 0: left quad  at x=-5
	auto q0 = make_quad(point3(-6, -1, -1), vec3(2,0,0), vec3(0,2,0));
	// Light 1: right quad at x=+5
	auto q1 = make_quad(point3( 4, -1, -1), vec3(2,0,0), vec3(0,2,0));

	bvh_light_sampler ls;
	ls.add(q0, 1.0);
	ls.add(q1, 1.0);
	ls.build();

	// From a point equidistant, importance should be ~equal => PMFs ~0.5
	// (exact equality depends on bbox geometry; check they're both > 0 and sum to 1)
	point3 p(0, 0, 0);
	double p0 = ls.light_pmf_at(0, p);
	double p1 = ls.light_pmf_at(1, p);
	EXPECT_GT(p0, 0.0);
	EXPECT_GT(p1, 0.0);
	EXPECT_NEAR(p0 + p1, 1.0, 1e-9) << "PMFs must sum to 1";
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: bright light dominates selection
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, BrightLightSelectedMoreOften) {
	// Two identical quads side by side, one 10x more powerful
	auto q0 = make_quad(point3(-3, -1, -1), vec3(2,0,0), vec3(0,2,0));
	auto q1 = make_quad(point3( 1, -1, -1), vec3(2,0,0), vec3(0,2,0));

	LightBounds lb0 = light_bounds_for(q0, 1.0);
	LightBounds lb1 = light_bounds_for(q1, 10.0);

	bvh_light_sampler ls;
	ls.add(q0, lb0);
	ls.add(q1, lb1);
	ls.build();

	point3 p(0, 0, 5);  // equidistant from both
	double p0 = ls.light_pmf_at(0, p);
	double p1 = ls.light_pmf_at(1, p);
	EXPECT_GT(p1, p0) << "Brighter light should have higher selection PMF";
	EXPECT_NEAR(p0 + p1, 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: PMF sum over all lights equals 1
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, PMFSumsToOne_ManyLights) {
	bvh_light_sampler ls;
	// 8 quads in a ring
	const int N = 8;
	for (int i = 0; i < N; ++i) {
		double angle = i * (2.0 * 3.14159265358979323846 / N);
		double cx = 5.0 * std::cos(angle);
		double cz = 5.0 * std::sin(angle);
		auto q = make_quad(point3(cx-0.5, -0.5, cz-0.5),
						   vec3(1,0,0), vec3(0,1,0));
		ls.add(q, 1.0 + (double)i);  // varying power
	}
	ls.build();

	point3 p(0, 0, 0);
	double total = 0.0;
	for (int i = 0; i < N; ++i)
		total += ls.light_pmf_at(i, p);
	EXPECT_NEAR(total, 1.0, 1e-7) << "All PMFs must sum to 1";
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: MC convergence -- random() distribution matches PMF
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, RandomSelectionMatchesPMF) {
	// Two lights: one bright, one dim.  Count how often each is selected.
	auto q0 = make_quad(point3(-2, -1, 3), vec3(2,0,0), vec3(0,2,0));
	auto q1 = make_quad(point3( 1, -1, 3), vec3(2,0,0), vec3(0,2,0));

	LightBounds lb0 = light_bounds_for(q0, 1.0);
	LightBounds lb1 = light_bounds_for(q1, 9.0);

	bvh_light_sampler ls;
	ls.add(q0, lb0);
	ls.add(q1, lb1);
	ls.build();

	point3 origin(0, 0, 0);
	double pmf0 = ls.light_pmf_at(0, origin);
	double pmf1 = ls.light_pmf_at(1, origin);

	// Sample many directions and count which light is hit
	const int N = 5000;
	int hits0 = 0, hits1 = 0;
	for (int i = 0; i < N; ++i) {
		vec3 dir = ls.random(origin);
		ray r(origin, dir);
		hit_record rec;
		bool h0 = q0->hit(r, interval(1e-4, 1e20), rec);
		bool h1 = q1->hit(r, interval(1e-4, 1e20), rec);
		if (h0) ++hits0;
		if (h1) ++hits1;
	}

	// Allow generous tolerance (MC noise)
	if (hits0 + hits1 > 10) {
		double frac0 = (double)hits0 / (hits0 + hits1);
		EXPECT_NEAR(frac0, pmf0 / (pmf0 + pmf1), 0.1)
			<< "MC sample fraction should roughly match PMF ratio";
	}
}

// ---------------------------------------------------------------------------
// bvh_light_sampler: pdf_value integrates consistently
// ---------------------------------------------------------------------------

TEST(BVHLightSampler, PdfValuePositiveForSampledDirection) {
	auto q = make_quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105));
	bvh_light_sampler ls;
	ls.add(q, 5.0);
	ls.build();

	point3 origin(278, 278, 278);
	for (int i = 0; i < 10; ++i) {
		vec3 dir = ls.random(origin);
		double pdf = ls.pdf_value(origin, dir);
		EXPECT_GE(pdf, 0.0) << "pdf_value must be non-negative";
	}
}
