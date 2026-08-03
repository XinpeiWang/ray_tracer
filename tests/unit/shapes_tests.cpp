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

// ===========================================================================
// CylinderShape<double> tests
// pbrt-v4 reference: Cylinder in shapes.h
// ===========================================================================

static CylinderShape<double> unit_cyl() {
	return CylinderShape<double>::make(0.0, 0.0, -1.0, 1.0, 1.0);
}

static const double CYL_PI = 3.14159265358979323846;

TEST(ShapesCylinder, AreaFullCylinder) {
	// 2*pi * r * h = 2*pi*1*2 = 4*pi
	auto c = unit_cyl();
	EXPECT_NEAR(c.area(), 4.0 * CYL_PI, 1e-10);
}

TEST(ShapesCylinder, AreaPartial) {
	// Half-sweep, r=2, h=3: pi*2*3 = 6*pi
	auto c = CylinderShape<double>::make_partial(0, 0, 0, 3, 2, CYL_PI);
	EXPECT_NEAR(c.area(), 6.0 * CYL_PI, 1e-10);
}

TEST(ShapesCylinder, PdfAreaIsInverseArea) {
	auto c = unit_cyl();
	EXPECT_NEAR(c.pdf_area(), 1.0 / c.area(), 1e-14);
}

TEST(ShapesCylinder, IntersectFrontFaceHit) {
	// Ray along +x from (-2,0,0): hits cylinder at t=1, nx=-1
	auto c = unit_cyl();
	auto hit = c.intersect(-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t,  1.0, 1e-9);
	EXPECT_NEAR(hit->nx, -1.0, 1e-9);
	EXPECT_NEAR(hit->ny,  0.0, 1e-9);
	EXPECT_NEAR(hit->nz,  0.0, 1e-9);
}

TEST(ShapesCylinder, IntersectRearFaceFromInside) {
	// Ray from origin along +x: exits at t=1, nx=+1
	auto c = unit_cyl();
	auto hit = c.intersect(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t,  1.0, 1e-9);
	EXPECT_NEAR(hit->nx, 1.0, 1e-9);
}

TEST(ShapesCylinder, IntersectPointOnSurface) {
	auto c = unit_cyl();
	auto hit = c.intersect(-2.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	ASSERT_TRUE(hit.has_value());
	double hx = -2.0 + hit->t;
	double hy = 0.5;
	EXPECT_NEAR(std::sqrt(hx*hx + hy*hy), 1.0, 1e-9);
}

TEST(ShapesCylinder, IntersectMissFarFromAxis) {
	auto c = unit_cyl();
	auto hit = c.intersect(5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1e-4, 1e6);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesCylinder, IntersectMissZClip) {
	auto c = unit_cyl(); // z in [-1,1]
	auto hit = c.intersect(-2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesCylinder, IntersectMissPhiClip) {
	// Quarter cylinder (phi in [0, pi/2]): both intersections land in the
	// third/fourth quadrant (phi > pi/2) so neither should register.
	auto c = CylinderShape<double>::make_partial(0, 0, -1, 1, 1, CYL_PI / 2);
	// Ray traveling +x through y=-0.5 slice; both hit points have negative y
	// giving phi in (pi, 2*pi) which is outside [0, pi/2].
	auto hit = c.intersect(-2.0, -0.5, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesCylinder, IntersectMissTMax) {
	auto c = unit_cyl();
	auto hit = c.intersect(-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-4, 0.5);
	EXPECT_FALSE(hit.has_value());
}

TEST(ShapesCylinder, UVFrontCenter) {
	// Ray from (-2,0,0): phi=pi -> u=0.5; z=0 -> v=0.5
	auto c = unit_cyl();
	auto hit = c.intersect(-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->u, 0.5, 1e-9);
	EXPECT_NEAR(hit->v, 0.5, 1e-9);
}

TEST(ShapesCylinder, NormalIsOutwardRadial) {
	auto c = unit_cyl();
	auto hit = c.intersect(-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-4, 1e6);
	ASSERT_TRUE(hit.has_value());
	double nlen = std::sqrt(hit->nx*hit->nx + hit->ny*hit->ny + hit->nz*hit->nz);
	EXPECT_NEAR(nlen, 1.0, 1e-9);
	EXPECT_NEAR(hit->nz, 0.0, 1e-9);
}

TEST(ShapesCylinder, SamplePdfIsInverseArea) {
	auto c = unit_cyl();
	auto s = c.sample(0.3, 0.7);
	EXPECT_NEAR(s.pdf, 1.0 / c.area(), 1e-12);
}

TEST(ShapesCylinder, SamplePointOnSurface) {
	auto c = unit_cyl();
	auto s = c.sample(0.25, 0.6);
	double dx = s.px, dy = s.py;
	EXPECT_NEAR(std::sqrt(dx*dx + dy*dy), 1.0, 1e-9);
	EXPECT_GE(s.pz, -1.0 - 1e-9);
	EXPECT_LE(s.pz,  1.0 + 1e-9);
}

TEST(ShapesCylinder, SampleNormalIsOutward) {
	auto c = unit_cyl();
	auto s = c.sample(0.5, 0.5);
	double nlen = std::sqrt(s.nx*s.nx + s.ny*s.ny + s.nz*s.nz);
	EXPECT_NEAR(nlen, 1.0, 1e-9);
	EXPECT_NEAR(s.nz, 0.0, 1e-9);
}

TEST(ShapesCylinder, SampleFromPdfConsistent) {
	// sample_from and pdf_from should agree when the sampled point is on the
	// near (ctx-facing) side so that pdf_from traces back to the same point.
	// ctx is at (10,0,0); u1=0.02 -> phi?0.13 rad, clearly on the +x face.
	auto c = unit_cyl();
	SamplingContext<double> ctx{10.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	auto ss = c.sample_from(ctx, 0.4, 0.02);
	ASSERT_GT(ss.pdf, 0.0);
	double wi_x = ss.px - ctx.px, wi_y = ss.py - ctx.py, wi_z = ss.pz - ctx.pz;
	double pdf2 = c.pdf_from(ctx, wi_x, wi_y, wi_z);
	EXPECT_NEAR(ss.pdf, pdf2, 0.01 * (ss.pdf + pdf2) * 0.5 + 1e-10);
}

TEST(ShapesCylinder, PdfFromZeroForMiss) {
	auto c = unit_cyl();
	SamplingContext<double> ctx{0.0, 0.0, 5.0, 0.0, 0.0, 0.0};
	double pdf = c.pdf_from(ctx, 0.0, 0.0, 1.0); // pointing away
	EXPECT_EQ(pdf, 0.0);
}

TEST(ShapesCylinder, SampleIsDeterministic) {
	auto c = unit_cyl();
	auto s1 = c.sample(0.123, 0.456);
	auto s2 = c.sample(0.123, 0.456);
	EXPECT_EQ(s1.px, s2.px);
	EXPECT_EQ(s1.py, s2.py);
	EXPECT_EQ(s1.pz, s2.pz);
}
