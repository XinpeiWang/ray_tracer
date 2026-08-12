#include <gtest/gtest.h>

#include "../../qt_gui/camera_math.h"

#include <cmath>

using namespace camera_math;

namespace {

// Cornell Box's look-at point, the case every one of these paths was
// originally written against.
constexpr Vec3 kCornellLookAt{278.0, 278.0, 278.0};

void expectVec3Near(const Vec3 &actual, const Vec3 &expected, double tol = 1e-9) {
	EXPECT_NEAR(actual.x, expected.x, tol);
	EXPECT_NEAR(actual.y, expected.y, tol);
	EXPECT_NEAR(actual.z, expected.z, tol);
}

} // namespace

TEST(CameraMathTest, DistanceIsPlainEuclideanDistance) {
	EXPECT_NEAR(distanceFromTarget(Vec3{278.0, 278.0, -800.0}, kCornellLookAt),
				1078.0, 1e-9);
	EXPECT_NEAR(distanceFromTarget(kCornellLookAt, kCornellLookAt), 0.0, 1e-9);
}

TEST(CameraMathTest, RepositioningPreservesDirectionAndSetsDistance) {
	const Vec3 camera{278.0, 278.0, -800.0};   // straight in front, 1078 away
	const Vec3 moved = repositionAtDistance(camera, kCornellLookAt, 500.0);

	EXPECT_NEAR(distanceFromTarget(moved, kCornellLookAt), 500.0, 1e-9);
	// Direction preserved: still straight along -Z from the target.
	expectVec3Near(moved, Vec3{278.0, 278.0, -222.0});
}

TEST(CameraMathTest, RepositioningWorksOnAnObliqueDirection) {
	const Vec3 camera{278.0 + 3.0, 278.0 + 4.0, 278.0};   // 3-4-5 triangle, dist 5
	const Vec3 moved = repositionAtDistance(camera, kCornellLookAt, 10.0);

	EXPECT_NEAR(distanceFromTarget(moved, kCornellLookAt), 10.0, 1e-9);
	expectVec3Near(moved, Vec3{278.0 + 6.0, 278.0 + 8.0, 278.0});
}

// The case that would otherwise divide by zero: with the camera exactly on the
// look-at point there is no direction to preserve. Producing NaN here would be
// silently destructive - the value flows into the spin boxes and then onto the
// renderer's command line.
TEST(CameraMathTest, RepositioningFromTheTargetItselfFallsBackToMinusZ) {
	const Vec3 moved = repositionAtDistance(kCornellLookAt, kCornellLookAt, 42.0);

	EXPECT_FALSE(std::isnan(moved.x));
	EXPECT_FALSE(std::isnan(moved.y));
	EXPECT_FALSE(std::isnan(moved.z));
	expectVec3Near(moved, Vec3{278.0, 278.0, 278.0 - 42.0});
	EXPECT_NEAR(distanceFromTarget(moved, kCornellLookAt), 42.0, 1e-9);
}

// Same guard, approached from just inside the epsilon rather than exactly on it.
TEST(CameraMathTest, RepositioningIsStableJustInsideTheEpsilon) {
	const Vec3 almostOnTarget{278.0 + 1e-9, 278.0, 278.0};
	const Vec3 moved = repositionAtDistance(almostOnTarget, kCornellLookAt, 10.0);

	EXPECT_FALSE(std::isnan(moved.x));
	EXPECT_NEAR(distanceFromTarget(moved, kCornellLookAt), 10.0, 1e-9);
}

TEST(CameraMathTest, RepositioningToZeroDistanceLandsOnTheTarget) {
	const Vec3 camera{278.0, 278.0, -800.0};
	expectVec3Near(repositionAtDistance(camera, kCornellLookAt, 0.0), kCornellLookAt);
}

TEST(CameraMathTest, PresetScalesADirectionByTheScenesOwnDistance) {
	// "Front" style preset: straight back along -Z, full radius.
	const Vec3 dir{0.0, 0.0, -1.0};
	expectVec3Near(presetPosition(dir, kCornellLookAt, 1078.0),
				   Vec3{278.0, 278.0, 278.0 - 1078.0});
}

// The whole reason presets store a direction rather than a position: the same
// preset must land sensibly in a scene whose geometry is ~15 units across, not
// at Cornell Box's literal coordinates.
TEST(CameraMathTest, SamePresetAdaptsToASmallScene) {
	const Vec3 dir{0.0, 0.0, -1.0};
	const Vec3 smallSceneLookAt{0.0, 0.0, 0.0};

	const Vec3 pos = presetPosition(dir, smallSceneLookAt, 14.0);
	expectVec3Near(pos, Vec3{0.0, 0.0, -14.0});
	EXPECT_NEAR(distanceFromTarget(pos, smallSceneLookAt), 14.0, 1e-9);
}

TEST(CameraMathTest, PresetWithZeroSceneDistanceCollapsesToTheTarget) {
	// Degenerate but reachable if scene metadata is unavailable; must not NaN.
	const Vec3 pos = presetPosition(Vec3{1.0, 0.0, 0.0}, kCornellLookAt, 0.0);
	expectVec3Near(pos, kCornellLookAt);
}

// Round trip: read the distance off a position, then reposition to that same
// distance. The camera must not move. The GUI does exactly this every time a
// preset is applied and the distance display refreshes.
TEST(CameraMathTest, DistanceThenRepositionIsAnIdentity) {
	const Vec3 camera{278.0 + 120.0, 278.0 - 45.0, 278.0 + 300.0};
	const double d = distanceFromTarget(camera, kCornellLookAt);
	expectVec3Near(repositionAtDistance(camera, kCornellLookAt, d), camera, 1e-9);
}
