// area_lights_tests.cpp
// Unit tests for src/shared/area_lights.h
//
// Tests mirror pbrt-v4 DiffuseAreaLight and UniformInfiniteLight behaviour:
//
//   DiffuseAreaLight (backed by SphereShape, DiskShape, TriangleShape):
//     - L(): emits on front face, zero on back face; two_sided emits both ways
//     - sample_li(): sampled point on shape surface, correct wi direction,
//       PDF > 0, incident radiance = scale * Le on lit side
//     - pdf_li(): > 0 for direction that hits shape, 0 for miss direction
//     - power(): matches pi * area * Le * scale formula
//
//   UniformInfiniteLight:
//     - Le(): returns scale * (Lr,Lg,Lb)
//     - sample_li(): wi is unit, pdf = 1/(4*pi), radiance = scale * Le
//     - sample_li() returns empty when allowIncompletePDF = true
//     - pdf_li(): returns 1/(4*pi), 0 when allowIncompletePDF = true
//     - power(): matches 4*pi^2*r^2*scale*Le formula

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/area_lights.h"

static const double PI = 3.14159265358979323846;

// ===========================================================================
// DiffuseAreaLight + SphereShape tests
// ===========================================================================

static DiffuseAreaLight<double, SphereShape<double>> make_sphere_light(
	double cx=0, double cy=0, double cz=0, double r=1.0,
	double Le=10.0, double scale=1.0, bool two_sided=false)
{
	DiffuseAreaLight<double, SphereShape<double>> light;
	light.shape     = SphereShape<double>::make(cx, cy, cz, r);
	light.Lr = light.Lg = light.Lb = Le;
	light.scale     = scale;
	light.two_sided = two_sided;
	return light;
}

TEST(AreaLightSphere, LFrontFaceEmits) {
	auto light = make_sphere_light();
	double r, g, b;
	// Normal pointing up, direction pointing up (same side) -> emits
	light.L(0,0,1, 0,0,1, r, g, b);
	EXPECT_GT(r, 0.0);
}

TEST(AreaLightSphere, LBackFaceZero) {
	auto light = make_sphere_light();
	double r, g, b;
	// Normal pointing up, direction pointing down (back face) -> zero
	light.L(0,0,1, 0,0,-1, r, g, b);
	EXPECT_EQ(r, 0.0);
	EXPECT_EQ(g, 0.0);
	EXPECT_EQ(b, 0.0);
}

TEST(AreaLightSphere, LTwoSidedBackFaceEmits) {
	auto light = make_sphere_light(0,0,0,1.0,10.0,1.0,true);
	double r, g, b;
	light.L(0,0,1, 0,0,-1, r, g, b);
	EXPECT_GT(r, 0.0);
}

TEST(AreaLightSphere, LScaleApplied) {
	auto light = make_sphere_light(0,0,0,1.0,5.0,2.0);
	double r, g, b;
	light.L(0,0,1, 0,0,1, r, g, b);
	EXPECT_NEAR(r, 10.0, 1e-12);
}

TEST(AreaLightSphere, SampleLiReturnsValue) {
	auto light = make_sphere_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto sample = light.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(sample.has_value());
}

TEST(AreaLightSphere, SampleLiWiUnitLength) {
	auto light = make_sphere_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
	double len = std::sqrt(s->wi_x*s->wi_x + s->wi_y*s->wi_y + s->wi_z*s->wi_z);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

TEST(AreaLightSphere, SampleLiPdfPositive) {
	auto light = make_sphere_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
	EXPECT_GT(s->pdf, 0.0);
}

TEST(AreaLightSphere, SampleLiRadianceMatchesLe) {
	auto light = make_sphere_light(0,0,0,1.0, 7.0, 2.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.25, 0.75);
	ASSERT_TRUE(s.has_value());
	EXPECT_NEAR(s->Lr, 14.0, 1e-10);
}

TEST(AreaLightSphere, SampleLiPointOnSphereSurface) {
	auto light = make_sphere_light(0,0,0,2.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 10);
	auto s = light.sample_li(ctx, 0.4, 0.6);
	ASSERT_TRUE(s.has_value());
	double dist = std::sqrt(s->lx*s->lx + s->ly*s->ly + s->lz*s->lz);
	EXPECT_NEAR(dist, 2.0, 1e-6);
}

TEST(AreaLightSphere, PdfLiHitDirection) {
	auto light = make_sphere_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	double pdf = light.pdf_li(ctx, 0, 0, -1);
	EXPECT_GT(pdf, 0.0);
}

TEST(AreaLightSphere, PdfLiMissDirection) {
	auto light = make_sphere_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	double pdf = light.pdf_li(ctx, 0, 0, 1);  // away from sphere
	EXPECT_EQ(pdf, 0.0);
}

TEST(AreaLightSphere, PowerFormula) {
	// power = pi * area * Le * scale  (one-sided)
	// area of unit sphere = 4*pi
	// power = pi * 4*pi * 10 * 1 = 40*pi^2
	auto light = make_sphere_light(0,0,0,1.0, 10.0, 1.0, false);
	double pr, pg, pb;
	light.power(pr, pg, pb);
	EXPECT_NEAR(pr, PI * light.shape.area() * 10.0 * 1.0, 1e-8);
}

TEST(AreaLightSphere, PowerTwoSidedDoubles) {
	auto light_one  = make_sphere_light(0,0,0,1.0, 10.0, 1.0, false);
	auto light_two  = make_sphere_light(0,0,0,1.0, 10.0, 1.0, true);
	double r1,g1,b1, r2,g2,b2;
	light_one.power(r1,g1,b1);
	light_two.power(r2,g2,b2);
	EXPECT_NEAR(r2, 2.0 * r1, 1e-10);
}

// ===========================================================================
// DiffuseAreaLight + DiskShape tests
// ===========================================================================

static DiffuseAreaLight<double, DiskShape<double>> make_disk_light(
	double cx=0, double cy=0, double h=0, double r=1.0,
	double Le=5.0, double scale=1.0, bool two_sided=false)
{
	DiffuseAreaLight<double, DiskShape<double>> light;
	light.shape     = DiskShape<double>::make(cx, cy, h, r);
	light.Lr = light.Lg = light.Lb = Le;
	light.scale     = scale;
	light.two_sided = two_sided;
	return light;
}

TEST(AreaLightDisk, SampleLiReturnsValue) {
	auto light = make_disk_light();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
}

TEST(AreaLightDisk, SampleLiWiUnitLength) {
	auto light = make_disk_light();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.4, 0.7);
	ASSERT_TRUE(s.has_value());
	double len = std::sqrt(s->wi_x*s->wi_x + s->wi_y*s->wi_y + s->wi_z*s->wi_z);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

TEST(AreaLightDisk, SampleLiPdfPositive) {
	auto light = make_disk_light();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.5, 0.5);
	ASSERT_TRUE(s.has_value());
	EXPECT_GT(s->pdf, 0.0);
}

TEST(AreaLightDisk, SampleLiPointOnDiskPlane) {
	auto light = make_disk_light(0,0,0,1.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 5);
	auto s = light.sample_li(ctx, 0.3, 0.3);
	ASSERT_TRUE(s.has_value());
	EXPECT_NEAR(s->lz, 0.0, 1e-10);
}

TEST(AreaLightDisk, BackFaceZero) {
	auto light = make_disk_light();
	// Normal up (0,0,1); direction down toward disk = (0,0,-1)
	// The disk normal is +z; outgoing direction -z is back face
	double r, g, b;
	light.L(0,0,1, 0,0,-1, r, g, b);
	EXPECT_EQ(r, 0.0);
}

TEST(AreaLightDisk, PowerFormula) {
	auto light = make_disk_light(0,0,0,2.0, 3.0, 1.0);
	double pr, pg, pb;
	light.power(pr, pg, pb);
	EXPECT_NEAR(pr, PI * light.shape.area() * 3.0 * 1.0, 1e-9);
}

// ===========================================================================
// DiffuseAreaLight + TriangleShape tests
// ===========================================================================

static DiffuseAreaLight<double, TriangleShape<double>> make_tri_light(double Le=8.0)
{
	DiffuseAreaLight<double, TriangleShape<double>> light;
	light.shape     = TriangleShape<double>::make(0,0,0, 1,0,0, 0,1,0);
	light.Lr = light.Lg = light.Lb = Le;
	light.scale     = 1.0;
	light.two_sided = false;
	return light;
}

TEST(AreaLightTriangle, SampleLiReturnsValue) {
	auto light = make_tri_light();
	auto ctx = LightSampleContext<double>::from_point(0.25, 0.25, 5);
	auto s = light.sample_li(ctx, 0.3, 0.3);
	ASSERT_TRUE(s.has_value());
}

TEST(AreaLightTriangle, SampleLiWiUnitLength) {
	auto light = make_tri_light();
	auto ctx = LightSampleContext<double>::from_point(0.25, 0.25, 5);
	auto s = light.sample_li(ctx, 0.3, 0.3);
	ASSERT_TRUE(s.has_value());
	double len = std::sqrt(s->wi_x*s->wi_x + s->wi_y*s->wi_y + s->wi_z*s->wi_z);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

TEST(AreaLightTriangle, SampleLiPdfPositive) {
	auto light = make_tri_light();
	auto ctx = LightSampleContext<double>::from_point(0.25, 0.25, 5);
	auto s = light.sample_li(ctx, 0.4, 0.4);
	ASSERT_TRUE(s.has_value());
	EXPECT_GT(s->pdf, 0.0);
}

TEST(AreaLightTriangle, PdfLiHitDirection) {
	auto light = make_tri_light();
	auto ctx = LightSampleContext<double>::from_point(1.0/3, 1.0/3, 5);
	double pdf = light.pdf_li(ctx, 0, 0, -1);
	EXPECT_GT(pdf, 0.0);
}

TEST(AreaLightTriangle, PdfLiMissDirection) {
	auto light = make_tri_light();
	auto ctx = LightSampleContext<double>::from_point(1.0/3, 1.0/3, 5);
	double pdf = light.pdf_li(ctx, 0, 0, 1);  // away from triangle
	EXPECT_EQ(pdf, 0.0);
}

TEST(AreaLightTriangle, PowerFormula) {
	auto light = make_tri_light(8.0);
	double pr, pg, pb;
	light.power(pr, pg, pb);
	EXPECT_NEAR(pr, PI * light.shape.area() * 8.0, 1e-10);
}

// ===========================================================================
// UniformInfiniteLight tests
// ===========================================================================

static UniformInfiniteLight<double> make_sky(double Le=1.0, double scale=1.0,
											  double radius=100.0)
{
	UniformInfiniteLight<double> sky;
	sky.Lr = sky.Lg = sky.Lb = Le;
	sky.scale = scale;
	sky.scene_center_x = sky.scene_center_y = sky.scene_center_z = 0.0;
	sky.scene_radius = radius;
	return sky;
}

TEST(UniformInfiniteLight, LeReturnsScaledRadiance) {
	auto sky = make_sky(3.0, 2.0);
	double r, g, b;
	sky.Le(r, g, b);
	EXPECT_NEAR(r, 6.0, 1e-12);
}

TEST(UniformInfiniteLight, SampleLiReturnsValue) {
	auto sky = make_sky();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 0);
	auto s = sky.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
}

TEST(UniformInfiniteLight, SampleLiWiUnitLength) {
	auto sky = make_sky();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 0);
	auto s = sky.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
	double len = std::sqrt(s->wi_x*s->wi_x + s->wi_y*s->wi_y + s->wi_z*s->wi_z);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

TEST(UniformInfiniteLight, SampleLiPdfIsUniformSphere) {
	auto sky = make_sky();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 0);
	auto s = sky.sample_li(ctx, 0.3, 0.6);
	ASSERT_TRUE(s.has_value());
	EXPECT_NEAR(s->pdf, 1.0 / (4.0 * PI), 1e-12);
}

TEST(UniformInfiniteLight, SampleLiRadianceMatchesLe) {
	auto sky = make_sky(4.0, 3.0);
	auto ctx = LightSampleContext<double>::from_point(0, 0, 0);
	auto s = sky.sample_li(ctx, 0.5, 0.5);
	ASSERT_TRUE(s.has_value());
	EXPECT_NEAR(s->Lr, 12.0, 1e-12);
}

TEST(UniformInfiniteLight, SampleLiEmptyWhenAllowIncompletePDF) {
	auto sky = make_sky();
	auto ctx = LightSampleContext<double>::from_point(0, 0, 0);
	auto s = sky.sample_li(ctx, 0.3, 0.6, true);
	EXPECT_FALSE(s.has_value());
}

TEST(UniformInfiniteLight, PdfLiIsUniformSphere) {
	auto sky = make_sky();
	double pdf = sky.pdf_li(0, 0, 1);
	EXPECT_NEAR(pdf, 1.0 / (4.0 * PI), 1e-12);
}

TEST(UniformInfiniteLight, PdfLiZeroWhenAllowIncompletePDF) {
	auto sky = make_sky();
	double pdf = sky.pdf_li(0, 0, 1, true);
	EXPECT_EQ(pdf, 0.0);
}

TEST(UniformInfiniteLight, PdfLiSameForAllDirections) {
	auto sky = make_sky();
	double p1 = sky.pdf_li(1, 0, 0);
	double p2 = sky.pdf_li(0, 1, 0);
	double p3 = sky.pdf_li(0, 0, 1);
	EXPECT_NEAR(p1, p2, 1e-14);
	EXPECT_NEAR(p2, p3, 1e-14);
}

TEST(UniformInfiniteLight, PowerFormula) {
	// phi = 4*pi^2 * r^2 * scale * Le
	auto sky = make_sky(2.0, 3.0, 50.0);
	double pr, pg, pb;
	sky.power(pr, pg, pb);
	double expected = 4.0 * PI * PI * 50.0 * 50.0 * 3.0 * 2.0;
	EXPECT_NEAR(pr, expected, 1e-6);
}

TEST(UniformInfiniteLight, SampleLiConsistentWithPdfLi) {
	// For several samples, verify pdf_li agrees with the pdf field in sample_li
	auto sky = make_sky();
	auto ctx = LightSampleContext<double>::from_point(1, 2, 3);
	for (double u : {0.1, 0.3, 0.5, 0.7, 0.9}) {
		auto s = sky.sample_li(ctx, u, 1.0 - u);
		ASSERT_TRUE(s.has_value());
		double pdf_check = sky.pdf_li(s->wi_x, s->wi_y, s->wi_z);
		EXPECT_NEAR(s->pdf, pdf_check, 1e-12);
	}
}
