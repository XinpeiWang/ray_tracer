#include <gtest/gtest.h>

#include "../../qt_gui/object_drag_math.h"

#include <cmath>

using namespace object_drag;

namespace {

// A camera at (0, 5, 10) looking at the origin with a 90 degree vertical field of view and a square picture, in the form the renderer reports it.
CameraBasis testCamera() {
	const Vec3 origin{0.0, 5.0, 10.0};
	const Vec3 forward = camera_math::normalized(Vec3{0.0, 0.0, 0.0} - origin);
	const Vec3 right = camera_math::normalized(camera_math::cross(forward, Vec3{0.0, 1.0, 0.0}));
	const Vec3 up = camera_math::cross(right, forward);
	CameraBasis cb;
	cb.origin = origin;
	cb.horizontal = right * 2.0;
	cb.vertical = up * 2.0;
	cb.lowerLeftCorner = origin + forward - right - up;
	return cb;
}

// Where a world point shows in the picture.
void screenOf(const CameraBasis &cb, const Vec3 &p, double &s, double &t) {
	const camera_math::ScreenProjection sp = camera_math::projectToScreen(p, cb);
	ASSERT_TRUE(sp.inFront);
	s = sp.s;
	t = sp.t;
}

}  // namespace

TEST(ObjectDragMathTest, FloorDragFollowsTheCursorOnTheFloor) {
	const CameraBasis cb = testCamera();
	const Vec3 grab{2.0, 0.0, 0.0};
	double s, t;
	screenOf(cb, Vec3{5.0, 0.0, 3.0}, s, t);
	Vec3 delta;
	ASSERT_TRUE(dragDelta(cb, grab, s, t, false, 0.0, delta));
	EXPECT_NEAR(delta.x, 3.0, 1e-9);
	EXPECT_NEAR(delta.y, 0.0, 1e-12);
	EXPECT_NEAR(delta.z, 3.0, 1e-9);
}

TEST(ObjectDragMathTest, FloorDragOfAPointAboveTheFloorStaysInItsOwnPlane) {
	const CameraBasis cb = testCamera();
	const Vec3 grab{1.0, 2.0, 0.0};   // the top of a box
	double s, t;
	screenOf(cb, Vec3{-1.0, 2.0, 1.0}, s, t);
	Vec3 delta;
	ASSERT_TRUE(dragDelta(cb, grab, s, t, false, 0.0, delta));
	EXPECT_NEAR(delta.x, -2.0, 1e-9);
	EXPECT_NEAR(delta.y, 0.0, 1e-12) << "a floor drag never lifts";
	EXPECT_NEAR(delta.z, 1.0, 1e-9);
}

TEST(ObjectDragMathTest, VerticalDragMovesStraightUpAndDown) {
	const CameraBasis cb = testCamera();
	const Vec3 grab{0.0, 0.0, 0.0};
	double s, t;
	screenOf(cb, Vec3{0.0, 2.0, 0.0}, s, t);
	Vec3 delta;
	ASSERT_TRUE(dragDelta(cb, grab, s, t, true, 0.0, delta));
	EXPECT_NEAR(delta.x, 0.0, 1e-12);
	EXPECT_NEAR(delta.y, 2.0, 1e-9);
	EXPECT_NEAR(delta.z, 0.0, 1e-12);
	// A cursor to the side of the object does not move it sideways.
	screenOf(cb, Vec3{3.0, 2.0, 0.0}, s, t);
	ASSERT_TRUE(dragDelta(cb, grab, s, t, true, 0.0, delta));
	EXPECT_NEAR(delta.x, 0.0, 1e-12);
}

TEST(ObjectDragMathTest, ARayParallelToTheFloorGivesNoMove) {
	CameraBasis cb = testCamera();
	// Level camera at floor height looking along -z: the middle of the picture is parallel to the floor.
	cb.origin = Vec3{0.0, 0.0, 10.0};
	cb.lowerLeftCorner = Vec3{-1.0, -1.0, 9.0};
	cb.horizontal = Vec3{2.0, 0.0, 0.0};
	cb.vertical = Vec3{0.0, 2.0, 0.0};
	Vec3 delta;
	EXPECT_FALSE(dragDelta(cb, Vec3{0.0, 0.0, 0.0}, 0.5, 0.5, false, 0.0, delta));
}

TEST(ObjectDragMathTest, AFloorBehindTheCameraGivesNoMove) {
	const CameraBasis cb = testCamera();
	Vec3 delta;
	// The top edge of the picture looks above the horizon: the ray never meets the floor in front of the camera.
	EXPECT_FALSE(dragDelta(cb, Vec3{0.0, 0.0, 0.0}, 0.5, 1.0, false, 0.0, delta));
}

TEST(ObjectDragMathTest, MoveIsShortenedToTheLimit) {
	const CameraBasis cb = testCamera();
	double s, t;
	screenOf(cb, Vec3{20.0, 0.0, 0.0}, s, t);
	Vec3 delta;
	ASSERT_TRUE(dragDelta(cb, Vec3{0.0, 0.0, 0.0}, s, t, false, 5.0, delta));
	EXPECT_NEAR(camera_math::length(delta), 5.0, 1e-9);
	EXPECT_NEAR(delta.x, 5.0, 1e-9);
}

TEST(ObjectDragMathTest, NoVerticalPlaneFacesACameraLookingStraightDown) {
	CameraBasis cb = testCamera();
	cb.origin = Vec3{0.0, 10.0, 0.0};
	cb.lowerLeftCorner = Vec3{-1.0, 9.0, 1.0};
	cb.horizontal = Vec3{2.0, 0.0, 0.0};
	cb.vertical = Vec3{0.0, 0.0, -2.0};
	Vec3 delta;
	EXPECT_FALSE(dragDelta(cb, Vec3{0.0, 0.0, 0.0}, 0.5, 0.5, true, 0.0, delta));
}
