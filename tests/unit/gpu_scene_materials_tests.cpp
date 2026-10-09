// gpu_scene_materials_tests.cpp - src/shared/gpu_scene_materials.h: the material rule the CPU builder and both GPU loaders share.
#include <gtest/gtest.h>

#include <cmath>

#include "../../src/shared/gpu_scene_materials.h"

using gpu_scene_materials::reflectanceToConductorK;

TEST(GpuSceneMaterialsTest, KIsSolvedSoTheNormalIncidenceReflectanceComesBack) {
	// With eta = 1, R = k^2 / (4 + k^2). Feeding k back must return the reflectance we started from.
	for (double r : {0.05, 0.2, 0.5, 0.75, 0.9, 0.97}) {
		const double k = reflectanceToConductorK(r);
		EXPECT_NEAR(k * k / (4.0 + k * k), r, 1e-12) << "r = " << r;
	}
}

TEST(GpuSceneMaterialsTest, OutOfRangeReflectanceIsClampedAndStaysFinite) {
	EXPECT_EQ(reflectanceToConductorK(0.0), 0.0);
	EXPECT_EQ(reflectanceToConductorK(-3.0), 0.0) << "negative reflectance is a black mirror, not a NaN";
	const double mirror = reflectanceToConductorK(1.0);
	EXPECT_TRUE(std::isfinite(mirror));
	EXPECT_EQ(reflectanceToConductorK(7.0), mirror) << "everything at or above 0.9999 is the same clamped mirror";
	EXPECT_NEAR(mirror, 2.0 * std::sqrt(0.9999) / 1e-2, 1e-9) << "the denominator is floored at sqrt(1e-4)";
}

TEST(GpuSceneMaterialsTest, FloatAndDoubleAgreeToFloatPrecisionAndDoubleIsNotRoundedDown) {
	for (float r : {0.1f, 0.4f, 0.8f, 0.95f})
		EXPECT_NEAR(reflectanceToConductorK(r), static_cast<float>(reflectanceToConductorK(static_cast<double>(r))), 1e-4f * reflectanceToConductorK(r));
	// The template keeps the caller's precision: a double result is not a rounded float.
	const double d = reflectanceToConductorK(0.123456789012345);
	EXPECT_NE(d, static_cast<double>(static_cast<float>(d)));
}

TEST(GpuSceneMaterialsTest, KGrowsWithReflectance) {
	double prev = -1.0;
	for (double r = 0.0; r <= 1.0; r += 0.05) {
		const double k = reflectanceToConductorK(r);
		EXPECT_GE(k, prev);
		prev = k;
	}
}
