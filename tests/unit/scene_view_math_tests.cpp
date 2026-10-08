// scene_view_math_tests.cpp - the geometry behind the Scene Builder's 3D view (src/shared/scene_view_math.h): the orbit camera's axes, perspective
// projection and its inverse (the ray under a pixel), the plane and axis queries that dragging uses, and near-plane clipping.
#include <gtest/gtest.h>

#include "../../src/shared/scene_view_math.h"

using namespace scene_view;

namespace {
void expectNear(const V3& a, const V3& b, double eps = 1e-9) {
	EXPECT_NEAR(a.x, b.x, eps);
	EXPECT_NEAR(a.y, b.y, eps);
	EXPECT_NEAR(a.z, b.z, eps);
}
View makeView() {
	View v;
	v.width = 800;
	v.height = 600;
	v.cam.target = {0, 1, 0};
	v.cam.distance = 8;
	v.cam.yawDeg = 30;
	v.cam.pitchDeg = 25;
	return v;
}
}  // namespace

TEST(SceneViewCameraTest, DefaultLooksDownMinusZWithXToTheRightAndYUp) {
	OrbitCamera c;
	c.target = {0, 0, 0};
	c.yawDeg = 0;
	c.pitchDeg = 0;
	expectNear(c.eye(), {0, 0, 10});
	expectNear(c.forward(), {0, 0, -1});
	expectNear(c.right(), {1, 0, 0});
	expectNear(c.up(), {0, 1, 0});
}

TEST(SceneViewCameraTest, AxesStayOrthonormalWhereverItOrbits) {
	OrbitCamera c;
	for (double yaw : {-170.0, -45.0, 0.0, 90.0, 200.0})
		for (double pitch : {-80.0, -10.0, 0.0, 33.0, 85.0}) {
			c.yawDeg = yaw;
			c.pitchDeg = pitch;
			EXPECT_NEAR(length(c.forward()), 1.0, 1e-9);
			EXPECT_NEAR(length(c.right()), 1.0, 1e-9);
			EXPECT_NEAR(length(c.up()), 1.0, 1e-9);
			EXPECT_NEAR(dot(c.forward(), c.right()), 0.0, 1e-9);
			EXPECT_NEAR(dot(c.forward(), c.up()), 0.0, 1e-9);
			EXPECT_NEAR(dot(c.right(), c.up()), 0.0, 1e-9);
			EXPECT_NEAR(length(c.eye() - c.target), c.distance, 1e-9);
			EXPECT_GT(c.up().y, 0.0) << "the camera is never upside down";
		}
}

TEST(SceneViewCameraTest, PitchIsClampedSoUpIsNeverParallelToTheViewDirection) {
	OrbitCamera c;
	c.orbit(0, 500);
	EXPECT_LE(c.pitchDeg, 89.0);
	c.orbit(0, -1000);
	EXPECT_GE(c.pitchDeg, -89.0);
	EXPECT_GT(length(cross(c.forward(), V3{0, 1, 0})), 1e-3);
}

TEST(SceneViewCameraTest, PanMovesTheTargetAlongTheScreenAxesAndDollyScalesTheDistance) {
	OrbitCamera c;
	const V3 before = c.target;
	c.pan(1.0, 0.0);
	expectNear(c.target - before, c.right());
	c.pan(0.0, 2.0);
	c.dolly(0.5);
	EXPECT_NEAR(c.distance, 5.0, 1e-9);
	c.dolly(1e-9);
	EXPECT_GE(c.distance, 0.2);
}

TEST(SceneViewProjectionTest, TheTargetProjectsToTheCentreAtItsDistance) {
	const View v = makeView();
	double sx = 0, sy = 0, depth = 0;
	ASSERT_TRUE(v.project(v.cam.target, sx, sy, &depth));
	EXPECT_NEAR(sx, 400.0, 1e-6);
	EXPECT_NEAR(sy, 300.0, 1e-6);
	EXPECT_NEAR(depth, v.cam.distance, 1e-9);
}

TEST(SceneViewProjectionTest, ARightwardPointLandsRightAndAnUpwardPointLandsHigher) {
	const View v = makeView();
	double cx, cy, rx, ry, ux, uy;
	ASSERT_TRUE(v.project(v.cam.target, cx, cy));
	ASSERT_TRUE(v.project(v.cam.target + v.cam.right(), rx, ry));
	ASSERT_TRUE(v.project(v.cam.target + v.cam.up(), ux, uy));
	EXPECT_GT(rx, cx);
	EXPECT_NEAR(ry, cy, 1e-6);
	EXPECT_LT(uy, cy) << "pixel y grows downwards";
	EXPECT_NEAR(ux, cx, 1e-6);
}

TEST(SceneViewProjectionTest, ThePixelSizeOfAUnitFollowsTheFieldOfView) {
	View v = makeView();
	v.cam.fovDeg = 90;  // focal length = half the height
	EXPECT_NEAR(v.focal(), 300.0, 1e-9);
	double sx, sy;
	ASSERT_TRUE(v.project(v.cam.target + v.cam.right() * v.cam.distance, sx, sy));  // a unit-per-unit-of-depth offset: tan(45) = 1 -> focal pixels
	EXPECT_NEAR(sx - 400.0, 300.0, 1e-6);
}

TEST(SceneViewProjectionTest, APointBehindTheCameraDoesNotProject) {
	const View v = makeView();
	double sx, sy;
	EXPECT_FALSE(v.project(v.cam.eye() - v.cam.forward() * 3.0, sx, sy));
}

TEST(SceneViewRayTest, TheRayUnderAProjectedPointPassesThroughIt) {
	const View v = makeView();
	for (const V3& p : {V3{1, 0.5, -2}, V3{-3, 2, 1}, V3{0, 0, 0}, V3{2, 3, 4}}) {
		double sx, sy, depth;
		ASSERT_TRUE(v.project(p, sx, sy, &depth));
		const Ray r = v.ray(sx, sy);
		EXPECT_NEAR(length(r.dir), 1.0, 1e-9);
		// distance from p to the ray's line
		const V3 w = p - r.origin;
		const double along = dot(w, r.dir);
		EXPECT_LT(length(w - r.dir * along), 1e-9);
		EXPECT_GT(along, 0);
	}
}

TEST(SceneViewRayTest, TheCentrePixelLooksAtTheTarget) {
	const View v = makeView();
	const Ray r = v.ray(400, 300);
	expectNear(r.dir, v.cam.forward());
	expectNear(r.origin, v.cam.eye());
}

TEST(SceneViewDragTest, RayPlaneFindsTheGroundPointUnderACornerOfTheScene) {
	const View v = makeView();
	const V3 ground{2.0, 0.0, -1.5};
	double sx, sy;
	ASSERT_TRUE(v.project(ground, sx, sy));
	V3 hit;
	ASSERT_TRUE(rayPlane(v.ray(sx, sy), {0, 0, 0}, {0, 1, 0}, hit));
	expectNear(hit, ground, 1e-9);
}

TEST(SceneViewDragTest, RayPlaneRefusesAParallelRayOrAHitBehindTheEye) {
	V3 hit;
	EXPECT_FALSE(rayPlane({{0, 5, 0}, {1, 0, 0}}, {0, 0, 0}, {0, 1, 0}, hit)) << "parallel";
	EXPECT_FALSE(rayPlane({{0, 5, 0}, {0, 1, 0}}, {0, 0, 0}, {0, 1, 0}, hit)) << "the plane is behind the ray";
}

TEST(SceneViewDragTest, DraggingAlongAnAxisFollowsTheMouseEvenWhenTheRayNeverTouchesTheAxis) {
	const View v = makeView();
	const V3 start{1.0, 0.5, -1.0};
	// The mouse is over where the object would be after moving 1.7 along +X, but 0.3 above that point: the ray misses the X axis line.
	const V3 target = start + V3{1.7, 0.3, 0.0};
	double sx, sy;
	ASSERT_TRUE(v.project(target, sx, sy));
	double t = 0;
	ASSERT_TRUE(closestOnLine(v.ray(sx, sy), start, {1, 0, 0}, t));
	EXPECT_NEAR(t, 1.7, 0.35);  // near, not exact: the closest point of the axis to that ray
	// Exactly on the axis it is exact.
	ASSERT_TRUE(v.project(start + V3{2.5, 0, 0}, sx, sy));
	ASSERT_TRUE(closestOnLine(v.ray(sx, sy), start, {1, 0, 0}, t));
	EXPECT_NEAR(t, 2.5, 1e-9);
	ASSERT_TRUE(v.project(start + V3{0, -1.25, 0}, sx, sy));
	ASSERT_TRUE(closestOnLine(v.ray(sx, sy), start, {0, 1, 0}, t));
	EXPECT_NEAR(t, -1.25, 1e-9);
}

TEST(SceneViewDragTest, ALineAlongTheViewDirectionCannotBeDraggedAndSaysSo) {
	const Ray r{{0, 0, 10}, {0, 0, -1}};
	double t;
	EXPECT_FALSE(closestOnLine(r, {0, 0, 0}, {0, 0, 1}, t));
}

TEST(SceneViewClipTest, AFaceWhollyInFrontIsUntouchedAndOneWhollyBehindIsDropped) {
	const std::vector<V3> front{{0, 0, 1}, {1, 0, 1}, {1, 1, 2}};
	EXPECT_EQ(clipNear(front, 0.05).size(), 3u);
	const std::vector<V3> behind{{0, 0, -1}, {1, 0, -1}, {1, 1, -2}};
	EXPECT_TRUE(clipNear(behind, 0.05).empty());
}

TEST(SceneViewClipTest, AFaceCrossingTheNearPlaneIsCutAtIt) {
	const std::vector<V3> quad{{-1, -1, -1}, {1, -1, -1}, {1, -1, 3}, {-1, -1, 3}};  // a floor strip running from behind the camera to in front
	const std::vector<V3> cut = clipNear(quad, 0.5);
	ASSERT_GE(cut.size(), 3u);
	for (const V3& p : cut) EXPECT_GE(p.z, 0.5 - 1e-9);
	bool onPlane = false;
	for (const V3& p : cut) onPlane = onPlane || std::abs(p.z - 0.5) < 1e-9;
	EXPECT_TRUE(onPlane);
}

TEST(SceneViewRotateTest, AngleAroundReadsWhereTheMouseIsOnARing) {
	const View v = makeView();
	const V3 centre{0, 1, 0};
	const V3 axis{0, 1, 0}, ref{1, 0, 0};  // a ring in the floor plane, measured from +X
	for (double deg : {0.0, 30.0, 90.0, 135.0, -60.0, -170.0}) {
		const double a = deg * kPi / 180.0;
		const V3 onRing = centre + (ref * std::cos(a) + cross(axis, ref) * std::sin(a)) * 1.5;
		double sx, sy;
		ASSERT_TRUE(v.project(onRing, sx, sy));
		double got = 0;
		ASSERT_TRUE(angleAround(v.ray(sx, sy), centre, axis, ref, got));
		EXPECT_NEAR(angleDelta(deg, got), 0.0, 1e-6) << "at " << deg;
	}
}

TEST(SceneViewRotateTest, AngleAroundRefusesAParallelRay) {
	double deg;
	EXPECT_FALSE(angleAround({{0, 5, 0}, {1, 0, 0}}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, deg));
}

TEST(SceneViewRotateTest, AngleDeltaTakesTheShortWayRound) {
	EXPECT_NEAR(angleDelta(170, -170), 20.0, 1e-9);
	EXPECT_NEAR(angleDelta(-170, 170), -20.0, 1e-9);
	EXPECT_NEAR(angleDelta(10, 40), 30.0, 1e-9);
	EXPECT_NEAR(angleDelta(0, 180), 180.0, 1e-9);
	EXPECT_NEAR(angleDelta(720, 725), 5.0, 1e-9);
}

namespace {
void expectSameMatrix(const Mat3& a, const Mat3& b, double eps = 1e-9) {
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) EXPECT_NEAR(a.m[i][j], b.m[i][j], eps) << i << "," << j;
}
}  // namespace

TEST(SceneViewEulerTest, RotationXYZMatchesWhatTheSceneFileDoes) {
	// Turn about X first, then Y, then Z: a point on +Y turned 90 about X goes to +Z, then 90 about Y to +X, then 90 about Z to +Y.
	const V3 p = rotationXYZ({90, 90, 90}) * V3{0, 1, 0};
	expectNear(p, {0, 1, 0}, 1e-9);
	expectNear(rotationXYZ({0, 0, 90}) * V3{1, 0, 0}, {0, 1, 0}, 1e-9);
	expectNear(rotationXYZ({90, 0, 0}) * V3{0, 1, 0}, {0, 0, 1}, 1e-9);
	expectNear(rotationXYZ({0, 90, 0}) * V3{0, 0, 1}, {1, 0, 0}, 1e-9);
}

TEST(SceneViewEulerTest, AnglesRoundTripThroughTheMatrix) {
	for (double x : {-170.0, -60.0, 0.0, 25.0, 140.0})
		for (double y : {-80.0, -20.0, 0.0, 45.0, 85.0})
			for (double z : {-150.0, 0.0, 33.0, 179.0}) {
				const Mat3 m = rotationXYZ({x, y, z});
				expectSameMatrix(rotationXYZ(eulerXYZ(m)), m, 1e-9);
			}
}

TEST(SceneViewEulerTest, StraightUpAndDownStillGiveTheSameTurn) {
	for (double y : {90.0, -90.0})
		for (double x : {0.0, 30.0, -100.0}) {
			const Mat3 m = rotationXYZ({x, y, 40.0});
			expectSameMatrix(rotationXYZ(eulerXYZ(m)), m, 1e-9);
		}
}

TEST(SceneViewEulerTest, TurningAboutAWorldAxisTurnsTheObjectAboutThatAxisWhateverItsAngles) {
	const V3 start{20, 35, -50};
	for (const V3& axis : {V3{1, 0, 0}, V3{0, 1, 0}, V3{0, 0, 1}}) {
		const V3 angles = turnAboutWorldAxis(start, axis, 30.0);
		// the object's own "forward" direction after, against turning the direction it had by 30 degrees about that world axis
		const V3 want = axisAngle(axis, 30.0) * (rotationXYZ(start) * V3{0.3, 0.5, 0.8});
		expectNear(rotationXYZ(angles) * V3{0.3, 0.5, 0.8}, want, 1e-9);
	}
	expectNear(turnAboutWorldAxis({0, 0, 0}, {0, 1, 0}, 90.0), {0, 90, 0}, 1e-9);
	expectNear(turnAboutWorldAxis({0, 0, 0}, {0, 1, 0}, 0.0), {0, 0, 0}, 1e-9);
}
