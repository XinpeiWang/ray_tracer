// ---------------------------------------------------------------------------
// punctual_light_tests.cpp
// Unit tests for punctual_lights.h and punctual_light_objects.h
// Mirrors pbrt-v4 PointLight / SpotLight / DistantLight behavior.
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/punctual_lights.h"
#include "../../src/TheRestOfYourLife/punctual_light_objects.h"

// ============================================================
// Helpers
// ============================================================
static double luminance(double r, double g, double b) {
	return 0.2126*r + 0.7152*g + 0.0722*b;
}

// ============================================================
// PointLightData<double>
// ============================================================
TEST(PointLightData, FalloffInverseSquare) {
	PointLightData<double> pl;
	pl.pos_x = 0; pl.pos_y = 0; pl.pos_z = 0;
	pl.ir = 1; pl.ig = 1; pl.ib = 1;
	pl.scale = 1.0;

	// Li at distance 2 should be 1/4 of Li at distance 1
	double Lr1, Lg1, Lb1, Lr2, Lg2, Lb2;
	pl.eval_Li(1, 0, 0, Lr1, Lg1, Lb1);
	pl.eval_Li(2, 0, 0, Lr2, Lg2, Lb2);
	EXPECT_NEAR(Lr1, 1.0, 1e-10);
	EXPECT_NEAR(Lr2, 0.25, 1e-10);
	EXPECT_NEAR(Lr1 / Lr2, 4.0, 1e-8);
}

TEST(PointLightData, SampleWiPointsTowardLight) {
	PointLightData<double> pl;
	pl.pos_x = 10; pl.pos_y = 0; pl.pos_z = 0;
	pl.ir = pl.ig = pl.ib = pl.scale = 1;
	double wx, wy, wz;
	pl.sample_wi(0, 0, 0, wx, wy, wz);
	EXPECT_NEAR(wx, 1.0, 1e-10);
	EXPECT_NEAR(wy, 0.0, 1e-10);
	EXPECT_NEAR(wz, 0.0, 1e-10);
}

TEST(PointLightData, PdfIsZero) {
	PointLightData<double> pl{};
	EXPECT_EQ(pl.pdf_Li(), 0.0);
}

TEST(PointLightData, PowerEquals4PiTimesAvgIntensity) {
	PointLightData<double> pl;
	pl.ir = 1; pl.ig = 1; pl.ib = 1; pl.scale = 1;
	double expected = 4.0 * 3.14159265358979323846;
	EXPECT_NEAR(pl.power(), expected, 1e-8);
}

TEST(PointLightData, ZeroDistanceIsBlack) {
	PointLightData<double> pl;
	pl.pos_x = 5; pl.pos_y = 5; pl.pos_z = 5;
	pl.ir = pl.ig = pl.ib = pl.scale = 1;
	double Lr, Lg, Lb;
	pl.eval_Li(5, 5, 5, Lr, Lg, Lb);
	EXPECT_EQ(Lr, 0.0);
	EXPECT_EQ(Lg, 0.0);
	EXPECT_EQ(Lb, 0.0);
}

// ============================================================
// SpotLightData<double>
// ============================================================
TEST(SpotLightData, InsideConeFullIntensity) {
	SpotLightData<double> sl;
	sl.pos_x = 0; sl.pos_y = 10; sl.pos_z = 0;
	sl.dir_x = 0; sl.dir_y = -1; sl.dir_z = 0;   // pointing down
	sl.ir = 1; sl.ig = 1; sl.ib = 1; sl.scale = 1;
	sl.cos_falloff_start = std::cos(0.1);  // ~5.7 deg inner
	sl.cos_falloff_end   = std::cos(0.4);  // ~22.9 deg outer

	// Point directly below: direction from light = (0,-1,0) aligns with cone axis
	double Lr, Lg, Lb;
	sl.eval_Li(0, 0, 0, Lr, Lg, Lb);
	// Should be full intensity / 100 (distance=10, r²=100)
	EXPECT_NEAR(Lr, 1.0/100.0, 1e-10);
}

TEST(SpotLightData, OutsideConeBlack) {
	SpotLightData<double> sl;
	sl.pos_x = 0; sl.pos_y = 10; sl.pos_z = 0;
	sl.dir_x = 0; sl.dir_y = -1; sl.dir_z = 0;
	sl.ir = 1; sl.ig = 1; sl.ib = 1; sl.scale = 1;
	sl.cos_falloff_start = std::cos(0.1);
	sl.cos_falloff_end   = std::cos(0.4);

	// Point far to the side (90 degrees from axis)
	double Lr, Lg, Lb;
	sl.eval_Li(1000, 10, 0, Lr, Lg, Lb);
	EXPECT_NEAR(Lr + Lg + Lb, 0.0, 1e-8);
}

TEST(SpotLightData, SmoothStepMonotonic) {
	// SmoothStep(x,a,b) should be monotonically increasing from 0 to 1
	double a = 0.5, b = 0.9;
	double prev = 0.0;
	for (int i = 0; i <= 20; ++i) {
		double x = a + (b - a) * i / 20.0;
		double s = SpotLightData<double>::smooth_step(x, a, b);
		EXPECT_GE(s, prev - 1e-12);
		EXPECT_GE(s, 0.0);
		EXPECT_LE(s, 1.0);
		prev = s;
	}
	EXPECT_NEAR(SpotLightData<double>::smooth_step(a - 0.01, a, b), 0.0, 1e-10);
	EXPECT_NEAR(SpotLightData<double>::smooth_step(b + 0.01, a, b), 1.0, 1e-10);
}

TEST(SpotLightData, PdfIsZero) {
	SpotLightData<double> sl{};
	EXPECT_EQ(sl.pdf_Li(), 0.0);
}

TEST(SpotLightData, PowerAnalytic) {
	// power = scale*avg_I*2*pi*((1-cosStart) + (cosStart-cosEnd)/2)
	SpotLightData<double> sl;
	sl.ir = 1; sl.ig = 1; sl.ib = 1; sl.scale = 1;
	sl.cos_falloff_start = std::cos(0.1);
	sl.cos_falloff_end   = std::cos(0.4);
	double p = sl.power();
	EXPECT_GT(p, 0.0);
	// narrower cone => less power
	sl.cos_falloff_start = std::cos(0.01);
	sl.cos_falloff_end   = std::cos(0.02);
	EXPECT_LT(sl.power(), p);
}

// ============================================================
// DistantLightData<double>
// ============================================================
TEST(DistantLightData, ConstantRadiance) {
	DistantLightData<double> dl;
	dl.dir_x = 0; dl.dir_y = 1; dl.dir_z = 0;
	dl.ir = 2; dl.ig = 3; dl.ib = 4;
	dl.scale = 0.5;
	dl.scene_radius = 100;
	double Lr, Lg, Lb;
	dl.eval_Li(Lr, Lg, Lb);
	EXPECT_NEAR(Lr, 1.0, 1e-10);
	EXPECT_NEAR(Lg, 1.5, 1e-10);
	EXPECT_NEAR(Lb, 2.0, 1e-10);
}

TEST(DistantLightData, SampleWiFixedDirection) {
	DistantLightData<double> dl;
	dl.dir_x = 0; dl.dir_y = 0; dl.dir_z = -1;
	dl.ir = dl.ig = dl.ib = dl.scale = 1;
	dl.scene_radius = 100;
	double wx, wy, wz;
	dl.sample_wi(wx, wy, wz);
	EXPECT_NEAR(wx, 0.0, 1e-10);
	EXPECT_NEAR(wy, 0.0, 1e-10);
	EXPECT_NEAR(wz, -1.0, 1e-10);
}

TEST(DistantLightData, PdfIsZero) {
	DistantLightData<double> dl{};
	EXPECT_EQ(dl.pdf_Li(), 0.0);
}

TEST(DistantLightData, PowerScalesWithRadiusSquared) {
	DistantLightData<double> dl;
	dl.dir_x = 0; dl.dir_y = 1; dl.dir_z = 0;
	dl.ir = 1; dl.ig = 1; dl.ib = 1; dl.scale = 1;
	dl.scene_radius = 10;
	double p10 = dl.power();
	dl.scene_radius = 20;
	double p20 = dl.power();
	EXPECT_NEAR(p20 / p10, 4.0, 1e-8);
}

// ============================================================
// CPU wrapper: point_light_obj
// ============================================================
TEST(PointLightObj, SampleDirectBasic) {
	point_light_obj pl(point3(0,10,0), color(1,1,1), 1.0);
	PunctualLiSample s = pl.sample_direct(point3(0,0,0));
	EXPECT_NEAR(s.wi.y(), 1.0, 1e-10);   // points straight up
	EXPECT_GT(luminance(s.Li.x(), s.Li.y(), s.Li.z()), 0.0);
	EXPECT_NEAR(s.pdf, 1.0, 1e-10);
	EXPECT_NEAR(s.t_max, 10.0, 1e-8);
}

TEST(PointLightObj, PowerPositive) {
	point_light_obj pl(point3(0,0,0), color(1,1,1), 1.0);
	EXPECT_GT(pl.power(), 0.0);
}

// ============================================================
// CPU wrapper: spot_light_obj
// ============================================================
TEST(SpotLightObj, SampleDirectInsideCone) {
	spot_light_obj sl(point3(0,10,0), vec3(0,-1,0), color(1,1,1), 30.0, 15.0, 1.0);
	PunctualLiSample s = sl.sample_direct(point3(0,0,0));
	EXPECT_GT(luminance(s.Li.x(), s.Li.y(), s.Li.z()), 0.0);
	EXPECT_NEAR(s.pdf, 1.0, 1e-10);
}

TEST(SpotLightObj, SampleDirectOutsideConeBlack) {
	spot_light_obj sl(point3(0,10,0), vec3(0,-1,0), color(1,1,1), 5.0, 3.0, 1.0);
	// Point far to the side: angle from cone axis is ~90 deg
	PunctualLiSample s = sl.sample_direct(point3(1000,10,0));
	EXPECT_NEAR(luminance(s.Li.x(), s.Li.y(), s.Li.z()), 0.0, 1e-8);
}

// ============================================================
// CPU wrapper: distant_light_obj
// ============================================================
TEST(DistantLightObj, SampleDirectConstantLi) {
	distant_light_obj dl(vec3(0,1,0), color(1,1,1), 500.0, 1.0);
	PunctualLiSample s1 = dl.sample_direct(point3(0,0,0));
	PunctualLiSample s2 = dl.sample_direct(point3(100,200,300));
	// Radiance is position-independent
	EXPECT_NEAR(s1.Li.x(), s2.Li.x(), 1e-10);
	EXPECT_NEAR(s1.wi.y(), 1.0, 1e-10);
	EXPECT_EQ(s1.t_max, infinity);
}

// ============================================================
// punctual_light_list
// ============================================================
TEST(PunctualLightList, EmptyByDefault) {
	punctual_light_list list;
	EXPECT_TRUE(list.empty());
}

TEST(PunctualLightList, AddAndIterate) {
	punctual_light_list list;
	list.add_point(point3(0,10,0), color(1,1,1), 1.0);
	list.add_distant(vec3(0,1,0), color(0.5,0.5,0.5), 100.0, 1.0);
	EXPECT_FALSE(list.empty());

	int count = 0;
	list.for_each_sample(point3(0,0,0), [&](const PunctualLiSample& s) {
		++count;
		EXPECT_NEAR(s.pdf, 1.0, 1e-10);
	});
	EXPECT_EQ(count, 2);
}
