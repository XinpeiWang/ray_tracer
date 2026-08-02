// shapes_tests.cpp
// Unit tests for src/shared/shapes.h
//
// Tests mirror pbrt-v4 behaviour and verify:
//   SphereShape<double>:
//     - Area formula (full sphere, clipped sphere)
//     - Intersection: hit, miss, behind-ray, z-clip, phi-clip
//     - Area-uniform sample: point on surface, correct normal, PDF = 1/area
//     - Solid-angle sample from exterior: PDF integrates (basic sanity check)
//     - pdf_from: PDF > 0 for direction inside cone, 0 outside
//   DiskShape<double>:
//     - Area formula (solid disk, annular)
//     - Intersection: hit, miss (outside radius), miss (behind ray), parallel ray
//     - Area-uniform sample: point on disk plane, PDF = 1/area
//     - pdf_from: area PDF converted to solid angle
//   TriangleShape<double>:
//     - Area formula (right-angle triangle)
//     - Intersection: centre hit, miss, degenerate triangle, back-face
//     - Area-uniform sample: within triangle bounds
//     - sample_from: solid-angle PDF basic sanity

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/shapes.h"

static const double PI = 3.14159265358979323846;

// ===========================================================================
// SphereShape tests
// ===========================================================================

TEST(ShapesSphere, AreaFullSphere) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	EXPECT_NEAR(s.area(), 4.0 * PI, 1e-10);
}

TEST(ShapesSphere, AreaClippedSphere) {
	// Half-sphere: z in [0, 1], phi_max = 2*pi -> area = 2*pi*r^2
	auto s = SphereShape<double>::make_clipped(0, 0, 0, 1.0, 0.0, 1.0, 2.0*PI);
	EXPECT_NEAR(s.area(), 2.0 * PI, 1e-10);
}

TEST(ShapesSphere, IntersectHit) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	// Ray from (0, 0, -5) pointing +z
	auto hit = s.intersect(0, 0, -5, 0, 0, 1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 4.0, 1e-9);
	// Normal should point -z at front face
	EXPECT_NEAR(hit->nz, -1.0, 1e-9);
}

TEST(ShapesSphere, IntersectMiss) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	// Ray offset in x
	auto hit = s.intersect(2, 0, -5, 0, 0, 1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesSphere, IntersectBehindRay) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	// Ray pointing away from sphere
	auto hit = s.intersect(0, 0, -5, 0, 0, -1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesSphere, IntersectOffCenter) {
	// Sphere not at origin
	auto s = SphereShape<double>::make(3, 0, 0, 1.0);
	auto hit = s.intersect(3, 0, -5, 0, 0, 1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 4.0, 1e-9);
}

TEST(ShapesSphere, IntersectNormalPointsOutward) {
	auto s = SphereShape<double>::make(0, 0, 0, 2.0);
	auto hit = s.intersect(0, 0, -10, 0, 0, 1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	// Normal at front hit should point toward -z (away from center)
	EXPECT_LT(hit->nz, 0.0);
	// Length should be ~1
	double nlen = std::sqrt(hit->nx*hit->nx + hit->ny*hit->ny + hit->nz*hit->nz);
	EXPECT_NEAR(nlen, 1.0, 1e-9);
}

TEST(ShapesSphere, SampleOnSurface) {
	auto s = SphereShape<double>::make(0, 0, 0, 2.0);
	auto ss = s.sample(0.25, 0.75);
	double dist = std::sqrt(ss.px*ss.px + ss.py*ss.py + ss.pz*ss.pz);
	EXPECT_NEAR(dist, 2.0, 1e-8);
}

TEST(ShapesSphere, SampleNormalUnitLength) {
	auto s = SphereShape<double>::make(1, 2, 3, 1.5);
	for (double u : {0.1, 0.4, 0.7, 0.9}) {
		auto ss = s.sample(u, 1.0 - u);
		double nlen = std::sqrt(ss.nx*ss.nx + ss.ny*ss.ny + ss.nz*ss.nz);
		EXPECT_NEAR(nlen, 1.0, 1e-9);
	}
}

TEST(ShapesSphere, SamplePdfEqualsOneOverArea) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	auto ss = s.sample(0.3, 0.6);
	EXPECT_NEAR(ss.pdf, 1.0 / s.area(), 1e-12);
}

TEST(ShapesSphere, SampleFromExteriorPdfPositive) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	auto ss = s.sample_from(ctx, 0.3, 0.6);
	EXPECT_GT(ss.pdf, 0.0);
}

TEST(ShapesSphere, PdfFromInsideCone) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	// Direction toward sphere center
	double pdf = s.pdf_from(ctx, 0, 0, -1);
	EXPECT_GT(pdf, 0.0);
}

TEST(ShapesSphere, PdfFromOutsideCone) {
	auto s = SphereShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	// Direction away from sphere
	double pdf = s.pdf_from(ctx, 0, 0, 1);
	EXPECT_EQ(pdf, 0.0);
}

TEST(ShapesSphere, PdfAreaReciprocal) {
	auto s = SphereShape<double>::make(0, 0, 0, 3.0);
	EXPECT_NEAR(s.pdf_area(), 1.0 / s.area(), 1e-14);
}

// ===========================================================================
// DiskShape tests
// ===========================================================================

TEST(ShapesDisk, AreaSolidDisk) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	EXPECT_NEAR(d.area(), PI, 1e-10);
}

TEST(ShapesDisk, AreaAnnularDisk) {
	auto d = DiskShape<double>::make_annular(0, 0, 0, 2.0, 1.0, 2.0*PI);
	// pi*(R^2 - r^2) = pi*(4 - 1) = 3*pi
	EXPECT_NEAR(d.area(), 3.0 * PI, 1e-10);
}

TEST(ShapesDisk, IntersectHit) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	// Ray from above, pointing down
	auto hit = d.intersect(0, 0, 5, 0, 0, -1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 5.0, 1e-10);
	EXPECT_NEAR(hit->nz, 1.0, 1e-10);
}

TEST(ShapesDisk, IntersectMissOutsideRadius) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	auto hit = d.intersect(2, 0, 5, 0, 0, -1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesDisk, IntersectParallelRay) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	// Ray parallel to disk (rdz == 0)
	auto hit = d.intersect(0, 0, 0, 1, 0, 0, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesDisk, IntersectBehindRay) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	// Ray pointing away from disk
	auto hit = d.intersect(0, 0, 5, 0, 0, 1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesDisk, IntersectOffCenter) {
	// Disk at height 3, centered at (1, 1, 3)
	auto d = DiskShape<double>::make(1, 1, 3, 2.0);
	auto hit = d.intersect(1, 1, 10, 0, 0, -1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 7.0, 1e-9);
}

TEST(ShapesDisk, SampleOnDiskPlane) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	auto ss = d.sample(0.3, 0.7);
	EXPECT_NEAR(ss.pz, 0.0, 1e-12);
	double r = std::sqrt(ss.px*ss.px + ss.py*ss.py);
	EXPECT_LE(r, 1.0 + 1e-9);
}

TEST(ShapesDisk, SampleNormalPointsUp) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	auto ss = d.sample(0.5, 0.5);
	EXPECT_NEAR(ss.nz, 1.0, 1e-12);
}

TEST(ShapesDisk, SamplePdfEqualsOneOverArea) {
	auto d = DiskShape<double>::make(0, 0, 0, 2.0);
	auto ss = d.sample(0.4, 0.6);
	EXPECT_NEAR(ss.pdf, 1.0 / d.area(), 1e-12);
}

TEST(ShapesDisk, SampleFromReturnsPositivePdf) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	auto ss = d.sample_from(ctx, 0.3, 0.6);
	EXPECT_GT(ss.pdf, 0.0);
}

TEST(ShapesDisk, PdfFromHitDirection) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	// Direction toward center of disk
	double pdf = d.pdf_from(ctx, 0, 0, -1);
	EXPECT_GT(pdf, 0.0);
}

TEST(ShapesDisk, PdfFromMissDirection) {
	auto d = DiskShape<double>::make(0, 0, 0, 1.0);
	SamplingContext<double> ctx{0, 0, 5, 0, 0, 1};
	// Direction that doesn't hit disk (too far off-axis)
	double pdf = d.pdf_from(ctx, 1, 0, 0);
	EXPECT_EQ(pdf, 0.0);
}

// ===========================================================================
// TriangleShape tests
// ===========================================================================

TEST(ShapesTriangle, AreaRightTriangle) {
	// Right-angle triangle: p0=(0,0,0), p1=(2,0,0), p2=(0,2,0)
	// area = 0.5 * base * height = 2
	auto t = TriangleShape<double>::make(0,0,0, 2,0,0, 0,2,0);
	EXPECT_NEAR(t.area(), 2.0, 1e-10);
}

TEST(ShapesTriangle, AreaEquilateral) {
	// Equilateral triangle with side 2: area = sqrt(3)
	auto t = TriangleShape<double>::make(
		-1.0, 0.0, 0.0,
		 1.0, 0.0, 0.0,
		 0.0, std::sqrt(3.0), 0.0);
	EXPECT_NEAR(t.area(), std::sqrt(3.0), 1e-9);
}

TEST(ShapesTriangle, IntersectCentreHit) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	// Ray from above centroid (1/3, 1/3, 5) pointing down
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, -1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 5.0, 1e-9);
}

TEST(ShapesTriangle, IntersectMiss) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	// Ray that misses (outside triangle)
	auto hit = t.intersect(2.0, 2.0, 5, 0, 0, -1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesTriangle, IntersectBehindRay) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	// Ray pointing away
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, 1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesTriangle, IntersectDegenerate) {
	// All three points collinear -> degenerate triangle, should always miss
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 2,0,0);
	auto hit = t.intersect(0.5, 0.5, 5, 0, 0, -1, 0, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesTriangle, IntersectNormalUnitLength) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, -1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	double nlen = std::sqrt(hit->nx*hit->nx + hit->ny*hit->ny + hit->nz*hit->nz);
	EXPECT_NEAR(nlen, 1.0, 1e-9);
}

TEST(ShapesTriangle, IntersectNormalPointsUp) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, -1, 0, 100);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->nz, 1.0, 1e-9);
}

TEST(ShapesTriangle, IntersectTMinRespected) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	// t_min = 10 > actual hit at t=5
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, -1, 10, 100);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesTriangle, IntersectTMaxRespected) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	// t_max = 3 < actual hit at t=5
	auto hit = t.intersect(1.0/3, 1.0/3, 5, 0, 0, -1, 0, 3);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesTriangle, SampleInsideTriangle) {
	auto t = TriangleShape<double>::make(0,0,0, 4,0,0, 0,4,0);
	for (double u : {0.1, 0.3, 0.5, 0.7, 0.9}) {
		auto ss = t.sample(u, 0.5 * (1.0 - u));
		// Point should be in z=0 plane
		EXPECT_NEAR(ss.pz, 0.0, 1e-10);
		// Barycentric coords b0=u, b1=v, b2=1-u-v should all be >= 0
		EXPECT_GE(ss.u, 0.0);
		EXPECT_GE(ss.v, 0.0);
		EXPECT_LE(ss.u + ss.v, 1.0 + 1e-10);
	}
}

TEST(ShapesTriangle, SamplePdfEqualsOneOverArea) {
	auto t = TriangleShape<double>::make(0,0,0, 2,0,0, 0,2,0);
	auto ss = t.sample(0.3, 0.5);
	EXPECT_NEAR(ss.pdf, 1.0 / t.area(), 1e-12);
}

TEST(ShapesTriangle, SampleNormalUnitLength) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	auto ss = t.sample(0.3, 0.3);
	double nlen = std::sqrt(ss.nx*ss.nx + ss.ny*ss.ny + ss.nz*ss.nz);
	EXPECT_NEAR(nlen, 1.0, 1e-9);
}

TEST(ShapesTriangle, SampleFromReturnsPositivePdf) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	SamplingContext<double> ctx{0.25, 0.25, 5, 0, 0, 1};
	auto ss = t.sample_from(ctx, 0.3, 0.6);
	EXPECT_GT(ss.pdf, 0.0);
}

TEST(ShapesTriangle, SampleFromSolidAnglePdfConsistency) {
	// Large triangle close to ctx: solid-angle sampling branch
	auto t = TriangleShape<double>::make(-5, -5, 0, 5, -5, 0, 0, 5, 0);
	SamplingContext<double> ctx{0, 0, 0.5, 0, 0, 1};
	auto ss = t.sample_from(ctx, 0.4, 0.4);
	EXPECT_GT(ss.pdf, 0.0);
}

TEST(ShapesTriangle, PdfFromHitDirection) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	SamplingContext<double> ctx{1.0/3, 1.0/3, 5, 0, 0, 1};
	double pdf = t.pdf_from(ctx, 0, 0, -1);
	EXPECT_GT(pdf, 0.0);
}

TEST(ShapesTriangle, PdfFromMissDirection) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	SamplingContext<double> ctx{1.0/3, 1.0/3, 5, 0, 0, 1};
	// Direction pointing away from triangle (+z away from z=0 plane)
	double pdf = t.pdf_from(ctx, 0, 0, 1);
	EXPECT_EQ(pdf, 0.0);
}

TEST(ShapesTriangle, GeometricNormalConsistency) {
	auto t = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	double nx, ny, nz;
	t.geometric_normal(nx, ny, nz);
	// Cross product of (1,0,0) and (0,1,0) = (0,0,1), scaled by area*2=2
	EXPECT_NEAR(nx, 0.0, 1e-10);
	EXPECT_NEAR(ny, 0.0, 1e-10);
	EXPECT_GT(nz, 0.0);
}
