// triangle_tests.cpp -- Unit tests for triangle.h and sampling.h
//   (SphericalTriangleSolidAngle, SampleSphericalTriangle, triangle hittable)
//
// pbrt-v4 references:
//   IntersectTriangle  -- shapes.cpp:168-268
//   SampleSphericalTriangle -- util/sampling.cpp:28-110
//
// Tests:
//   1. SphericalTriangleSolidAngle: hemisphere triangle SA = 2*pi / 4 for 1/4 of upper hemi
//   2. SphericalTriangleSolidAngle: degenerate (zero area) returns 0
//   3. SampleSphericalTriangle: barycentrics sum to 1, each in [0,1]
//   4. SampleSphericalTriangle: MC estimate of solid angle converges
//   5. triangle::hit: simple front-face hit
//   6. triangle::hit: miss (ray parallel to triangle)
//   7. triangle::hit: hit outside t range returns false
//   8. triangle::hit: back-face hit (two-sided by default via set_face_normal)
//   9. triangle::hit: UV interpolation
//  10. triangle::hit: normal interpolation via per-vertex normals
//  11. triangle::pdf_value: positive for a hitting direction
//  12. triangle::random: produces directions that hit the triangle

#include <gtest/gtest.h>
#include <cmath>

#include "../../src/shared/sampling.h"
using scalar_math_detail::kPi;
#define Pi kPi
#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/TheRestOfYourLife/vec3.h"
#include "../../src/TheRestOfYourLife/material.h"
#include "../../src/TheRestOfYourLife/triangle.h"

// Convenience: build a single-triangle mesh
static std::shared_ptr<triangle> make_tri(
		point3 p0, point3 p1, point3 p2,
		std::shared_ptr<material> mat = nullptr)
{
	if (!mat) mat = std::make_shared<lambertian>(color(0.5,0.5,0.5));
	auto mesh = std::make_shared<triangle_mesh_data>();
	mesh->positions = {p0, p1, p2};
	mesh->indices   = {0, 1, 2};
	return std::make_shared<triangle>(mesh, 0, mat);
}

// ---------------------------------------------------------------------------
// 1. SphericalTriangleSolidAngle -- simple sanity check
// ---------------------------------------------------------------------------
TEST(SphericalTriangle, SolidAnglePositiveForVisibleTriangle) {
	// Triangle on unit sphere, reference point at origin
	// v0=(1,0,0), v1=(0,1,0), v2=(0,0,1), p=(0,0,0)
	// These are three orthogonal unit vectors -> SA = pi/2
	double sa = SphericalTriangleSolidAngle(
		1,0,0,  0,1,0,  0,0,1,
		0,0,0);
	EXPECT_NEAR(sa, Pi / 2.0, 1e-8);
}

TEST(SphericalTriangle, DegenerateSolidAngleIsZero) {
	// Collinear vertices -> degenerate
	double sa = SphericalTriangleSolidAngle(
		0,0,1,  1,0,1,  2,0,1,
		0,0,0);
	EXPECT_EQ(sa, 0.0);
}

// ---------------------------------------------------------------------------
// 2. SampleSphericalTriangle: barycentrics valid
// ---------------------------------------------------------------------------
TEST(SphericalTriangle, BarycentricsValid) {
	// Triangle above origin
	for (int i = 0; i < 30; ++i) {
		double u0 = (i + 0.3) / 30.0, u1 = (i * 7 % 30 + 0.5) / 30.0;
		double b0, b1, b2, pdf;
		SampleSphericalTriangle(
			0,0,1,  1,0,1,  0,1,1,   // triangle z=1
			0,0,0,                     // ref point at origin
			u0, u1, &b0, &b1, &b2, &pdf);

		EXPECT_GE(b0, -1e-9) << i;
		EXPECT_GE(b1, -1e-9) << i;
		EXPECT_GE(b2, -1e-9) << i;
		EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-9) << i;
		EXPECT_GT(pdf, 0.0) << i;
	}
}

// ---------------------------------------------------------------------------
// 3. Monte Carlo: SA estimator converges to SphericalTriangleSolidAngle
// ---------------------------------------------------------------------------
TEST(SphericalTriangle, MonteCarloSolidAngleConverges) {
	double expected = SphericalTriangleSolidAngle(
		0,0,1,  1,0,1,  0,1,1,
		0,0,0);
	EXPECT_GT(expected, 0.0);

	const int N = 100000;
	double sum = 0.0;
	uint64_t state = 9876543210ULL;
	auto lcg = [&]() -> double {
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		return (state >> 11) / double(1ULL << 53);
	};
	for (int i = 0; i < N; ++i) {
		double b0, b1, b2, pdf;
		SampleSphericalTriangle(0,0,1, 1,0,1, 0,1,1, 0,0,0,
								lcg(), lcg(), &b0, &b1, &b2, &pdf);
		if (pdf > 0) sum += 1.0 / pdf;
	}
	EXPECT_NEAR(sum / N, expected, expected * 0.01);
}

// ---------------------------------------------------------------------------
// 4. triangle::hit -- basic front-face hit
// ---------------------------------------------------------------------------
TEST(TriangleHit, FrontFaceHit) {
	auto tri = make_tri({0,0,1}, {1,0,1}, {0,1,1});
	ray r(point3(0.25, 0.25, 0), vec3(0, 0, 1));
	hit_record rec;
	bool hit = tri->hit(r, interval(0.001, infinity), rec);
	EXPECT_TRUE(hit);
	EXPECT_NEAR(rec.t, 1.0, 1e-10);
	EXPECT_NEAR(rec.p.z(), 1.0, 1e-10);
	// set_face_normal: front_face = dot(ray.dir, outward_normal) < 0
	// Ray +z, triangle normal +z -> dot > 0 -> front_face = false (ray hits back)
	// This is correct: the triangle's geometric normal points toward +z (away from origin).
	// A ray from origin travelling +z hits the BACK face.
	EXPECT_FALSE(rec.front_face);
}

// ---------------------------------------------------------------------------
// 5. Miss when ray is parallel to triangle
// ---------------------------------------------------------------------------
TEST(TriangleHit, ParallelRayMisses) {
	auto tri = make_tri({0,0,1}, {1,0,1}, {0,1,1});
	ray r(point3(0, 0, 0), vec3(1, 0, 0));  // parallel to z=1 plane
	hit_record rec;
	EXPECT_FALSE(tri->hit(r, interval(0.001, infinity), rec));
}

// ---------------------------------------------------------------------------
// 6. Miss when t is outside interval
// ---------------------------------------------------------------------------
TEST(TriangleHit, OutsideTRange) {
	auto tri = make_tri({0,0,1}, {1,0,1}, {0,1,1});
	ray r(point3(0.25, 0.25, 0), vec3(0, 0, 1));
	hit_record rec;
	// Max t = 0.5, triangle is at t=1 -- should miss
	EXPECT_FALSE(tri->hit(r, interval(0.001, 0.5), rec));
}

// ---------------------------------------------------------------------------
// 7. Back-face hit (two-sided: set_face_normal flips normal)
// ---------------------------------------------------------------------------
TEST(TriangleHit, BackFaceHit) {
	auto tri = make_tri({0,0,1}, {1,0,1}, {0,1,1});
	// Ray from above shooting downward: ray.dir=-z, outward_normal=+z -> dot<0 -> front_face=true
	ray r(point3(0.25, 0.25, 5), vec3(0, 0, -1));
	hit_record rec;
	bool hit = tri->hit(r, interval(0.001, infinity), rec);
	EXPECT_TRUE(hit);
	EXPECT_NEAR(rec.t, 4.0, 1e-9);
	EXPECT_TRUE(rec.front_face);  // ray opposes outward normal -> front_face=true
}

// ---------------------------------------------------------------------------
// 8. UV interpolation
// ---------------------------------------------------------------------------
TEST(TriangleHit, UVInterpolation) {
	auto mesh = std::make_shared<triangle_mesh_data>();
	mesh->positions = {{0,0,1},{1,0,1},{0,1,1}};
	mesh->indices   = {0,1,2};
	mesh->uvs       = {0.0,0.0, 1.0,0.0, 0.0,1.0};  // corner UVs
	auto mat = std::make_shared<lambertian>(color(0.5,0.5,0.5));
	auto tri = std::make_shared<triangle>(mesh, 0, mat);

	// Hit at barycentric (0.5, 0.25, 0.25): UV = (0.25, 0.25)
	ray r(point3(0.25, 0.25, 0), vec3(0, 0, 1));
	hit_record rec;
	EXPECT_TRUE(tri->hit(r, interval(0.001, infinity), rec));
	EXPECT_NEAR(rec.u, 0.25, 1e-9);
	EXPECT_NEAR(rec.v, 0.25, 1e-9);
}

// ---------------------------------------------------------------------------
// 9. Normal interpolation via per-vertex normals
// ---------------------------------------------------------------------------
TEST(TriangleHit, NormalInterpolation) {
	auto mesh = std::make_shared<triangle_mesh_data>();
	mesh->positions = {{0,0,1},{1,0,1},{0,1,1}};
	mesh->indices   = {0,1,2};
	// All normals pointing in +z
	mesh->normals   = {vec3(0,0,1), vec3(0,0,1), vec3(0,0,1)};
	auto mat = std::make_shared<lambertian>(color(0.5,0.5,0.5));
	auto tri = std::make_shared<triangle>(mesh, 0, mat);

	ray r(point3(0.25, 0.25, 0), vec3(0, 0, 1));
	hit_record rec;
	EXPECT_TRUE(tri->hit(r, interval(0.001, infinity), rec));
	// set_face_normal flips normal to face the ray:
	// ray +z, outward +z -> dot > 0 -> normal = -outward = (0,0,-1)
	EXPECT_NEAR(std::abs(rec.normal.z()), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// 10. pdf_value is positive for a hitting direction
// ---------------------------------------------------------------------------
TEST(TrianglePDF, PositiveForHittingDirection) {
	auto tri = make_tri({0,0,2}, {1,0,2}, {0,1,2});
	point3 origin(0.25, 0.25, 0);
	vec3 dir = vec3(0.25, 0.25, 2) - origin;
	double pdf = tri->pdf_value(origin, dir);
	EXPECT_GT(pdf, 0.0);
	EXPECT_TRUE(std::isfinite(pdf));
}

TEST(TrianglePDF, ZeroForMissingDirection) {
	auto tri = make_tri({0,0,2}, {1,0,2}, {0,1,2});
	point3 origin(0.25, 0.25, 0);
	vec3 dir = vec3(0, 0, -1);  // away from triangle
	EXPECT_EQ(tri->pdf_value(origin, dir), 0.0);
}

// ---------------------------------------------------------------------------
// 11. random: sampled direction hits the triangle
// ---------------------------------------------------------------------------
TEST(TriangleRandom, SampledDirectionHits) {
	auto tri = make_tri({0,0,2}, {1,0,2}, {0,1,2});
	point3 origin(0.25, 0.25, 0);

	int hits = 0;
	const int N = 200;
	for (int i = 0; i < N; ++i) {
		vec3 dir = tri->random(origin);
		hit_record rec;
		if (tri->hit(ray(origin, dir), interval(1e-4, infinity), rec))
			++hits;
	}
	// Nearly all samples should hit (solid-angle sampling is exact)
	EXPECT_GE(hits, N * 95 / 100);
}
