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

// Round trip: decompose a camera position into spherical coordinates, then
// rebuild the position from them. Live Preview's sub-tab's mouse-drag orbit
// control does exactly this every time it seeds orbit state from the camera
// spinboxes (MainWindow::startLivePreview()/onLivePreviewCameraChanged()).
TEST(CameraMathTest, CartesianToOrbitThenBackIsAnIdentity) {
	// Straight along +Z from the target - a plain, on-axis case.
	expectVec3Near(
		orbitToCartesian(cartesianToOrbit(Vec3{278.0, 278.0, 278.0 + 500.0}, kCornellLookAt), kCornellLookAt),
		Vec3{278.0, 278.0, 278.0 + 500.0});

	// An oblique position with all three axes offset from the target,
	// including a nonzero elevation.
	const Vec3 oblique{278.0 + 300.0, 278.0 + 150.0, 278.0 - 200.0};
	expectVec3Near(orbitToCartesian(cartesianToOrbit(oblique, kCornellLookAt), kCornellLookAt), oblique);

	// A different look-at point than Cornell Box's, to confirm the
	// conversion doesn't secretly assume a fixed target.
	const Vec3 smallSceneLookAt{0.0, 0.0, 0.0};
	const Vec3 nearSmallScene{5.0, -3.0, 7.0};
	expectVec3Near(orbitToCartesian(cartesianToOrbit(nearSmallScene, smallSceneLookAt), smallSceneLookAt),
				   nearSmallScene);
}

TEST(CameraMathTest, CartesianToOrbitRadiusMatchesDistanceFromTarget) {
	const Vec3 camera{278.0, 278.0, -800.0};
	EXPECT_NEAR(cartesianToOrbit(camera, kCornellLookAt).radius,
				distanceFromTarget(camera, kCornellLookAt), 1e-9);
}

// Same degenerate case repositionAtDistance() guards against: a camera
// sitting exactly on the look-at point has no direction to decompose into
// azimuth/elevation. Must not NaN.
TEST(CameraMathTest, CartesianToOrbitFromTheTargetItselfDoesNotNaN) {
	const OrbitCoordinates orbit = cartesianToOrbit(kCornellLookAt, kCornellLookAt);
	EXPECT_FALSE(std::isnan(orbit.radius));
	EXPECT_FALSE(std::isnan(orbit.azimuth));
	EXPECT_FALSE(std::isnan(orbit.elevation));

	const Vec3 rebuilt = orbitToCartesian(orbit, kCornellLookAt);
	EXPECT_FALSE(std::isnan(rebuilt.x));
	EXPECT_FALSE(std::isnan(rebuilt.y));
	EXPECT_FALSE(std::isnan(rebuilt.z));
}

// A camera directly "above" the look-at point (elevation = +90 degrees) is
// the other edge case worth naming explicitly - asin's domain is [-1, 1],
// and dy/radius lands exactly on that boundary here.
TEST(CameraMathTest, CartesianToOrbitHandlesStraightUp) {
	const Vec3 straightUp{278.0, 278.0 + 100.0, 278.0};
	const OrbitCoordinates orbit = cartesianToOrbit(straightUp, kCornellLookAt);
	EXPECT_FALSE(std::isnan(orbit.elevation));
	EXPECT_NEAR(orbit.elevation, 1.5707963267948966, 1e-9);  // +90 degrees, in radians
	expectVec3Near(orbitToCartesian(orbit, kCornellLookAt), straightUp);
}

// Vec3 arithmetic - added for Live Preview's free-fly WASD translation
// (MainWindow::applyTranslateDelta()).

TEST(CameraMathTest, Vec3AddSubtractAreComponentwise) {
	const Vec3 a{1.0, 2.0, 3.0};
	const Vec3 b{10.0, 20.0, 30.0};
	expectVec3Near(a + b, Vec3{11.0, 22.0, 33.0});
	expectVec3Near(b - a, Vec3{9.0, 18.0, 27.0});
}

TEST(CameraMathTest, Vec3ScaleMultipliesEachComponent) {
	expectVec3Near(Vec3{1.0, -2.0, 3.0} * 2.0, Vec3{2.0, -4.0, 6.0});
	expectVec3Near(Vec3{1.0, -2.0, 3.0} * 0.0, Vec3{0.0, 0.0, 0.0});
}

TEST(CameraMathTest, LengthIsPlainEuclideanNorm) {
	EXPECT_NEAR(length(Vec3{3.0, 4.0, 0.0}), 5.0, 1e-9);
	EXPECT_NEAR(length(Vec3{0.0, 0.0, 0.0}), 0.0, 1e-9);
}

TEST(CameraMathTest, NormalizedProducesAUnitVectorInTheSameDirection) {
	const Vec3 n = normalized(Vec3{3.0, 4.0, 0.0});
	EXPECT_NEAR(length(n), 1.0, 1e-9);
	expectVec3Near(n, Vec3{0.6, 0.8, 0.0});
}

// Same degenerate-case convention as repositionAtDistance()/cartesianToOrbit():
// a zero-length vector has no direction to normalize. Must not NaN.
TEST(CameraMathTest, NormalizedOfZeroVectorFallsBackToZeroNotNaN) {
	const Vec3 n = normalized(Vec3{0.0, 0.0, 0.0});
	EXPECT_FALSE(std::isnan(n.x));
	EXPECT_FALSE(std::isnan(n.y));
	EXPECT_FALSE(std::isnan(n.z));
	expectVec3Near(n, Vec3{0.0, 0.0, 0.0});
}

// Standard right-handed basis check: X cross Y = Z. cross() itself is still
// used elsewhere (e.g. deriving screen-space basis vectors); applyTranslateDelta()'s
// own right vector no longer goes through it directly - it derives the same
// direction straight from m_orbit.azimuth instead, since cross(forward,
// worldUp) degenerates to zero when looking straight up/down.
TEST(CameraMathTest, CrossProductOfUnitAxesMatchesRightHandRule) {
	const Vec3 xAxis{1.0, 0.0, 0.0};
	const Vec3 yAxis{0.0, 1.0, 0.0};
	const Vec3 zAxis{0.0, 0.0, 1.0};
	expectVec3Near(cross(xAxis, yAxis), zAxis);
	expectVec3Near(cross(yAxis, zAxis), xAxis);
	expectVec3Near(cross(zAxis, xAxis), yAxis);
	// Anti-commutative: swapping operands flips the sign.
	expectVec3Near(cross(yAxis, xAxis), zAxis * -1.0);
}

TEST(CameraMathTest, DotProductOfPerpendicularUnitAxesIsZero) {
	EXPECT_NEAR(dot(Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}), 0.0, 1e-9);
	EXPECT_NEAR(dot(Vec3{1.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}), 1.0, 1e-9);
	EXPECT_NEAR(dot(Vec3{2.0, 3.0, 4.0}, Vec3{5.0, 6.0, 7.0}), 56.0, 1e-9);
}

// Camera at (0,0,5) looking at the origin, 90 degree vertical FOV
// (h = tan(45) = 1, so viewport_height = 2*h*focus_dist = 2), square aspect
// (viewport_width = 2 too) - the simplest possible case to reason about by
// hand. Basis vectors are the literal values build_pinhole_camera_params()
// itself would compute (lookfrom - horizontal/2 - vertical/2 - focus_dist*w
// for lowerLeftCorner, hand-derived independently of projectToScreen() so
// the test doesn't share a bug with the code it's checking).
constexpr CameraBasis kSquare90DegBasis{
	Vec3{0.0, 0.0, 5.0},    // origin
	Vec3{-1.0, -1.0, 4.0},  // lowerLeftCorner
	Vec3{2.0, 0.0, 0.0},    // horizontal
	Vec3{0.0, 2.0, 0.0}};   // vertical

TEST(CameraMathTest, ProjectToScreenPutsTheLookAtPointAtScreenCenter) {
	const ScreenProjection p = projectToScreen(Vec3{0.0, 0.0, 0.0}, kSquare90DegBasis);
	EXPECT_TRUE(p.inFront);
	EXPECT_NEAR(p.s, 0.5, 1e-9);
	EXPECT_NEAR(p.t, 0.5, 1e-9);
}

TEST(CameraMathTest, ProjectToScreenPlacesAPointToTheRightPastScreenCenter) {
	// Same depth as the look-at point, offset 1 unit along world +X.
	const ScreenProjection p = projectToScreen(Vec3{1.0, 0.0, 0.0}, kSquare90DegBasis);
	EXPECT_TRUE(p.inFront);
	EXPECT_NEAR(p.s, 0.6, 1e-9);
	EXPECT_NEAR(p.t, 0.5, 1e-9);
}

TEST(CameraMathTest, ProjectToScreenPlacesAPointAbovePastScreenCenter) {
	const ScreenProjection p = projectToScreen(Vec3{0.0, 1.0, 0.0}, kSquare90DegBasis);
	EXPECT_TRUE(p.inFront);
	EXPECT_NEAR(p.s, 0.5, 1e-9);
	EXPECT_NEAR(p.t, 0.6, 1e-9);
}

TEST(CameraMathTest, ProjectToScreenRejectsPointsBehindTheCamera) {
	// Beyond the camera, along the same viewing axis - behind it, not in front.
	const ScreenProjection p = projectToScreen(Vec3{0.0, 0.0, 10.0}, kSquare90DegBasis);
	EXPECT_FALSE(p.inFront);
}

TEST(CameraMathTest, ProjectToScreenRejectsAPointExactlyOnTheCamera) {
	// c == 0 exactly - the degenerate case the ">= 0.0" gate (not "> 0.0")
	// exists for, matching this file's other "no division by zero" fallbacks.
	const ScreenProjection p = projectToScreen(kSquare90DegBasis.origin, kSquare90DegBasis);
	EXPECT_FALSE(p.inFront);
}

TEST(CameraMathTest, ProjectToScreenRejectsADegenerateZeroBasis) {
	const ScreenProjection p = projectToScreen(Vec3{1.0, 2.0, 3.0}, CameraBasis{});
	EXPECT_FALSE(p.inFront);
}

TEST(CameraMathTest, ProjectToScreenNarrowsScreenFractionWithAspectRatio) {
	// Same offset point as ProjectToScreenPlacesAPointToTheRightPastScreenCenter,
	// but a 2:1 aspect ratio (horizontal doubled) halves the screen-fraction
	// offset from center for the same world offset (a wider screen fits more
	// world-space per unit of screen fraction horizontally).
	const CameraBasis wideBasis{
		Vec3{0.0, 0.0, 5.0}, Vec3{-2.0, -1.0, 4.0}, Vec3{4.0, 0.0, 0.0}, Vec3{0.0, 2.0, 0.0}};
	const ScreenProjection p = projectToScreen(Vec3{1.0, 0.0, 0.0}, wideBasis);
	EXPECT_TRUE(p.inFront);
	EXPECT_NEAR(p.s, 0.55, 1e-9);
	EXPECT_NEAR(p.t, 0.5, 1e-9);
}
