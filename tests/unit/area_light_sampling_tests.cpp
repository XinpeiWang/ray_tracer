/**
 * @file area_light_sampling_tests.cpp
 * @brief Unit tests for hittable::sample_area() (quad + sphere overrides)
 *
 * sample_area() is new infrastructure added for SPPM photon emission: it
 * samples a point uniformly over a shape's own surface (independent of any
 * reference/viewing point), unlike the existing pdf_value()/random() pair
 * which sample a *direction toward* the shape from a given reference point.
 *
 * Covers:
 *  - quad::sample_area(): sampled points lie in-plane and within the quad's
 *    footprint, UV in [0,1), mean pdf_pos matches the known analytic area.
 *  - sphere::sample_area(): sampled points lie exactly on the sphere,
 *    normals are unit length and radially outward, samples are uniformly
 *    distributed (octant bucket check), mean pdf_pos matches 1/(4*pi*r^2).
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include <cmath>
#include <random>

// ============================================================================
// quad::sample_area
// ============================================================================

TEST(QuadSampleArea, PointsLieInPlane) {
	point3 Q(1, 2, 3);
	vec3 u(4, 0, 0), v(0, 5, 0);
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	quad q(Q, u, v, mat);

	vec3 n = unit_vector(cross(u, v));
	std::mt19937 rng(1);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < 500; ++i) {
		AreaLightSample s;
		ASSERT_TRUE(q.sample_area(dist(rng), dist(rng), s));
		double plane_dist = std::fabs(dot(s.p - Q, n));
		EXPECT_NEAR(plane_dist, 0.0, 1e-9);
	}
}

TEST(QuadSampleArea, PointsWithinFootprintAndUVInRange) {
	point3 Q(0, 0, 0);
	vec3 u(2, 0, 0), v(0, 3, 0);
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	quad q(Q, u, v, mat);

	std::mt19937 rng(2);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < 500; ++i) {
		AreaLightSample s;
		ASSERT_TRUE(q.sample_area(dist(rng), dist(rng), s));
		EXPECT_GE(s.u, 0.0); EXPECT_LT(s.u, 1.0);
		EXPECT_GE(s.v, 0.0); EXPECT_LT(s.v, 1.0);
		// p = Q + u1*u + u2*v with u1,u2 in [0,1) -> x in [0,2), y in [0,3)
		EXPECT_GE(s.p.x(), 0.0); EXPECT_LT(s.p.x(), 2.0);
		EXPECT_GE(s.p.y(), 0.0); EXPECT_LT(s.p.y(), 3.0);
		EXPECT_NEAR(s.p.z(), 0.0, 1e-9);
	}
}

TEST(QuadSampleArea, NormalMatchesQuadNormal) {
	point3 Q(0, 0, 0);
	vec3 u(1, 0, 0), v(0, 1, 0);
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	quad q(Q, u, v, mat);

	AreaLightSample s;
	ASSERT_TRUE(q.sample_area(0.3, 0.7, s));
	vec3 expected_n = unit_vector(cross(u, v));
	EXPECT_NEAR(s.n.x(), expected_n.x(), 1e-9);
	EXPECT_NEAR(s.n.y(), expected_n.y(), 1e-9);
	EXPECT_NEAR(s.n.z(), expected_n.z(), 1e-9);
	EXPECT_NEAR(s.n.length(), 1.0, 1e-9);
}

TEST(QuadSampleArea, PdfMatchesKnownArea) {
	point3 Q(0, 0, 0);
	vec3 u(3, 0, 0), v(0, 4, 0);   // area = 12
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	quad q(Q, u, v, mat);

	AreaLightSample s;
	ASSERT_TRUE(q.sample_area(0.5, 0.5, s));
	EXPECT_NEAR(s.pdf_pos, 1.0 / 12.0, 1e-9);
}

// ============================================================================
// sphere::sample_area
// ============================================================================

TEST(SphereSampleArea, PointsLieOnSphere) {
	point3 center(1, 2, 3);
	double radius = 2.5;
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	sphere s(center, radius, mat);

	std::mt19937 rng(3);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < 500; ++i) {
		AreaLightSample smp;
		ASSERT_TRUE(s.sample_area(dist(rng), dist(rng), smp));
		double d = (smp.p - center).length();
		EXPECT_NEAR(d, radius, 1e-9);
	}
}

TEST(SphereSampleArea, NormalsAreUnitAndRadiallyOutward) {
	point3 center(0, 0, 0);
	double radius = 1.0;
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	sphere s(center, radius, mat);

	std::mt19937 rng(4);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < 200; ++i) {
		AreaLightSample smp;
		ASSERT_TRUE(s.sample_area(dist(rng), dist(rng), smp));
		EXPECT_NEAR(smp.n.length(), 1.0, 1e-9);
		vec3 expected_n = unit_vector(smp.p - center);
		EXPECT_NEAR(dot(smp.n, expected_n), 1.0, 1e-9);
	}
}

TEST(SphereSampleArea, SamplesAreUniformlyDistributedAcrossOctants) {
	point3 center(0, 0, 0);
	double radius = 1.0;
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	sphere s(center, radius, mat);

	int octant_counts[8] = {0};
	const int N = 20000;
	std::mt19937 rng(5);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < N; ++i) {
		AreaLightSample smp;
		ASSERT_TRUE(s.sample_area(dist(rng), dist(rng), smp));
		int idx = (smp.p.x() > 0 ? 1 : 0) | (smp.p.y() > 0 ? 2 : 0) | (smp.p.z() > 0 ? 4 : 0);
		octant_counts[idx]++;
	}
	// Each octant should get roughly N/8 samples; loose statistical bound
	// (this mirrors the style of other statistical sanity checks in this
	// codebase, e.g. sppm_tests.cpp, rather than an exact PRNG-sequence check).
	double expected = N / 8.0;
	for (int i = 0; i < 8; ++i) {
		EXPECT_GT(octant_counts[i], expected * 0.7) << "octant " << i;
		EXPECT_LT(octant_counts[i], expected * 1.3) << "octant " << i;
	}
}

TEST(SphereSampleArea, PdfMatchesKnownSurfaceArea) {
	point3 center(0, 0, 0);
	double radius = 2.0;   // surface area = 4*pi*4 = 16*pi
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	sphere s(center, radius, mat);

	AreaLightSample smp;
	ASSERT_TRUE(s.sample_area(0.4, 0.6, smp));
	EXPECT_NEAR(smp.pdf_pos, 1.0 / (16.0 * pi), 1e-9);
}

TEST(SphereSampleArea, UvInRange) {
	point3 center(0, 0, 0);
	double radius = 1.0;
	auto mat = make_shared<lambertian>(color(1, 1, 1));
	sphere s(center, radius, mat);

	std::mt19937 rng(6);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < 200; ++i) {
		AreaLightSample smp;
		ASSERT_TRUE(s.sample_area(dist(rng), dist(rng), smp));
		EXPECT_GE(smp.u, 0.0); EXPECT_LE(smp.u, 1.0);
		EXPECT_GE(smp.v, 0.0); EXPECT_LE(smp.v, 1.0);
	}
}

// ============================================================================
// Default hittable::sample_area (base class contract)
// ============================================================================

TEST(DefaultSampleArea, NonOverridingShapeReturnsFalse) {
	// bvh_node etc. don't override sample_area(); use a minimal stand-in
	// that only implements the pure-virtual interface to check the base
	// class's default without needing to construct a real un-samplable shape.
	struct MinimalHittable : public hittable {
		bool hit(const ray&, interval, hit_record&) const override { return false; }
		aabb bounding_box() const override { return aabb(); }
	};
	MinimalHittable h;
	AreaLightSample s;
	EXPECT_FALSE(h.sample_area(0.5, 0.5, s));
}
