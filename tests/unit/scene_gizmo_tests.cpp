// scene_gizmo_tests.cpp - src/shared/scene_gizmo.h: which dimension each Scale handle changes for every shape, and which axis handle a press means (and that
// the middle of the selected object is not one).
#include <gtest/gtest.h>

#include "../../src/shared/scene_gizmo.h"

using namespace scene_doc;
using namespace scene_gizmo;

namespace {
Object make(ShapeKind k) {
	Object o;
	o.shape = k;
	o.radius = 2.0;
	o.radius2 = 0.5;
	o.height = 4.0;
	o.size = {2.0, 3.0, 5.0};
	o.meshScale = 1.5;
	return o;
}
}  // namespace

TEST(ScaledObjectTest, AllRoundShapesScaleTheirRadius) {
	for (ShapeKind k : {ShapeKind::Sphere, ShapeKind::Disk, ShapeKind::Dome})
		for (int axis = 0; axis < 3; ++axis) {
			const Object r = scaledObject(make(k), axis, 2.0);
			EXPECT_DOUBLE_EQ(r.radius, 4.0) << toString(k) << " axis " << axis;
			EXPECT_DOUBLE_EQ(r.height, 4.0);
			EXPECT_DOUBLE_EQ(r.size.x, 2.0);
		}
}

TEST(ScaledObjectTest, AMeshScalesItsMeshScaleOnEveryAxis) {
	for (int axis = 0; axis < 3; ++axis) EXPECT_DOUBLE_EQ(scaledObject(make(ShapeKind::Mesh), axis, 0.5).meshScale, 0.75);
}

TEST(ScaledObjectTest, CylindersConesAndCapsulesStretchHeightOnYAndRadiusOnXAndZ) {
	for (ShapeKind k : {ShapeKind::Cylinder, ShapeKind::Cone, ShapeKind::Capsule}) {
		const Object y = scaledObject(make(k), 1, 1.5);
		EXPECT_DOUBLE_EQ(y.height, 6.0) << toString(k);
		EXPECT_DOUBLE_EQ(y.radius, 2.0);
		for (int axis : {0, 2}) {
			const Object r = scaledObject(make(k), axis, 1.5);
			EXPECT_DOUBLE_EQ(r.radius, 3.0) << toString(k) << " axis " << axis;
			EXPECT_DOUBLE_EQ(r.height, 4.0);
		}
	}
}

TEST(ScaledObjectTest, BoxesWedgesAndStairsStretchOnlyThatSize) {
	for (ShapeKind k : {ShapeKind::Box, ShapeKind::Wedge, ShapeKind::Stairs}) {
		const Object x = scaledObject(make(k), 0, 2.0), y = scaledObject(make(k), 1, 2.0), z = scaledObject(make(k), 2, 2.0);
		EXPECT_DOUBLE_EQ(x.size.x, 4.0); EXPECT_DOUBLE_EQ(x.size.y, 3.0); EXPECT_DOUBLE_EQ(x.size.z, 5.0);
		EXPECT_DOUBLE_EQ(y.size.x, 2.0); EXPECT_DOUBLE_EQ(y.size.y, 6.0); EXPECT_DOUBLE_EQ(y.size.z, 5.0);
		EXPECT_DOUBLE_EQ(z.size.x, 2.0); EXPECT_DOUBLE_EQ(z.size.y, 3.0); EXPECT_DOUBLE_EQ(z.size.z, 10.0);
	}
}

TEST(ScaledObjectTest, APyramidHasABaseOnXAndZAndAHeightOnY) {
	const Object x = scaledObject(make(ShapeKind::Pyramid), 0, 2.0), y = scaledObject(make(ShapeKind::Pyramid), 1, 2.0), z = scaledObject(make(ShapeKind::Pyramid), 2, 2.0);
	EXPECT_DOUBLE_EQ(x.size.x, 4.0); EXPECT_DOUBLE_EQ(x.height, 4.0);
	EXPECT_DOUBLE_EQ(y.height, 8.0); EXPECT_DOUBLE_EQ(y.size.x, 2.0); EXPECT_DOUBLE_EQ(y.size.z, 5.0);
	EXPECT_DOUBLE_EQ(z.size.z, 10.0); EXPECT_DOUBLE_EQ(z.size.x, 2.0);
}

TEST(ScaledObjectTest, ATorusScalesBothRadiiAndATubeKeepsItsHoleInProportion) {
	const Object t = scaledObject(make(ShapeKind::Torus), 0, 2.0);
	EXPECT_DOUBLE_EQ(t.radius, 4.0);
	EXPECT_DOUBLE_EQ(t.radius2, 1.0);
	const Object around = scaledObject(make(ShapeKind::Tube), 2, 2.0);
	EXPECT_DOUBLE_EQ(around.radius, 4.0);
	EXPECT_DOUBLE_EQ(around.radius2, 1.0);
	EXPECT_DOUBLE_EQ(around.height, 4.0);
	const Object tall = scaledObject(make(ShapeKind::Tube), 1, 2.0);
	EXPECT_DOUBLE_EQ(tall.height, 8.0);
	EXPECT_DOUBLE_EQ(tall.radius, 2.0);
	EXPECT_DOUBLE_EQ(tall.radius2, 0.5);
}

TEST(ScaledObjectTest, AQuadHasAWidthAndADepthButNoHeight) {
	EXPECT_TRUE(scaleHandleUsed(make(ShapeKind::Quad), 0));
	EXPECT_FALSE(scaleHandleUsed(make(ShapeKind::Quad), 1));
	EXPECT_TRUE(scaleHandleUsed(make(ShapeKind::Quad), 2));
	for (ShapeKind k : {ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Mesh, ShapeKind::Torus})
		for (int axis = 0; axis < 3; ++axis) EXPECT_TRUE(scaleHandleUsed(make(k), axis));
	const Object y = scaledObject(make(ShapeKind::Quad), 1, 3.0);  // not reachable from the UI, but it must change nothing
	EXPECT_DOUBLE_EQ(y.size.x, 2.0);
	EXPECT_DOUBLE_EQ(y.size.z, 5.0);
	EXPECT_DOUBLE_EQ(scaledObject(make(ShapeKind::Quad), 0, 2.0).size.x, 4.0);
	EXPECT_DOUBLE_EQ(scaledObject(make(ShapeKind::Quad), 2, 2.0).size.z, 10.0);
}

TEST(ScaledObjectTest, EveryShapeIsHandledSoNoNewShapeIsSilentlyUnscalable) {
	// a handle that does nothing for a shape that has a size would be a dead tool: each shape must change SOMETHING on at least one used axis
	for (ShapeKind k : allShapeKinds()) {
		const Object before = make(k);
		bool changed = false;
		for (int axis = 0; axis < 3; ++axis) {
			if (!scaleHandleUsed(before, axis)) continue;
			const Object after = scaledObject(before, axis, 2.0);
			changed = changed || after.radius != before.radius || after.radius2 != before.radius2 || after.height != before.height || after.size.x != before.size.x ||
			          after.size.y != before.size.y || after.size.z != before.size.z || after.meshScale != before.meshScale;
		}
		EXPECT_TRUE(changed) << toString(k);
	}
}

namespace {
const bool kAll[3] = {true, true, true};
}

TEST(PickArrowTest, TheMiddleOfTheObjectIsNotAnArrow) {
	const P2 base{100, 100}, tips[3] = {{180, 100}, {100, 20}, {145, 145}};
	EXPECT_EQ(pickArrow({100, 100}, base, tips, kAll, 14, 9), -1);
	EXPECT_EQ(pickArrow({105, 103}, base, tips, kAll, 14, 9), -1) << "within the dead zone, near every shaft";
	EXPECT_EQ(pickArrow({94, 108}, base, tips, kAll, 14, 9), -1);
}

TEST(PickArrowTest, AShaftBeyondTheDeadZoneIsPickedAndTheNearestOneWins) {
	const P2 base{100, 100}, tips[3] = {{180, 100}, {100, 20}, {145, 145}};
	EXPECT_EQ(pickArrow({140, 103}, base, tips, kAll, 14, 9), 0);
	EXPECT_EQ(pickArrow({98, 60}, base, tips, kAll, 14, 9), 1);
	EXPECT_EQ(pickArrow({125, 125}, base, tips, kAll, 14, 9), 2);
	EXPECT_EQ(pickArrow({140, 120}, base, tips, kAll, 14, 9), -1) << "between two shafts, too far from both";
}

TEST(PickArrowTest, AnUnusedArrowCannotBePicked) {
	const P2 base{100, 100}, tips[3] = {{180, 100}, {100, 20}, {145, 145}};
	const bool noY[3] = {true, false, true};
	EXPECT_EQ(pickArrow({98, 60}, base, tips, noY, 14, 9), -1);
}

TEST(PickTipTest, OnlyTheSquareAtTheEndIsAHandle) {
	const P2 tips[3] = {{180, 100}, {100, 20}, {145, 145}};
	EXPECT_EQ(pickTip({182, 98}, tips, kAll, 13), 0);
	EXPECT_EQ(pickTip({100, 25}, tips, kAll, 13), 1);
	EXPECT_EQ(pickTip({140, 100}, tips, kAll, 13), -1) << "part-way along a shaft: not a handle (the scale would be measured from a tiny distance)";
	EXPECT_EQ(pickTip({100, 100}, tips, kAll, 13), -1) << "the middle of the object";
	const bool noZ[3] = {true, true, false};
	EXPECT_EQ(pickTip({145, 145}, tips, noZ, 13), -1);
}

TEST(PickArrowTest, OnlyTheOuterPartOfAShaftIsAnArrowTheInnerPartIsTheObject) {
	const P2 base{100, 100}, tips[3] = {{180, 100}, {100, 20}, {145, 145}};
	// Past the dead zone and right on the shafts, but in the inner 45%: a free move, not an axis drag.
	EXPECT_EQ(pickArrow({125, 100}, base, tips, kAll, 14, 9), -1) << "31% along X";
	EXPECT_EQ(pickArrow({100, 75}, base, tips, kAll, 14, 9), -1) << "31% along Y";
	EXPECT_EQ(pickArrow({118, 118}, base, tips, kAll, 14, 9), -1) << "40% along Z";
	// From 45% out it is the arrow, to the tip and a little beyond it.
	EXPECT_EQ(pickArrow({137, 100}, base, tips, kAll, 14, 9), 0) << "46% along X";
	EXPECT_EQ(pickArrow({100, 24}, base, tips, kAll, 14, 9), 1);
	EXPECT_EQ(pickArrow({184, 100}, base, tips, kAll, 14, 9), 0) << "just past the tip";
	// A different fraction can be asked for.
	EXPECT_EQ(pickArrow({125, 100}, base, tips, kAll, 14, 9, 0.2), 0);
	// An arrow that points straight at the camera has no length on screen: nothing to pick (and no division by zero).
	const P2 flat[3] = {{100, 100}, {100, 20}, {145, 145}};
	EXPECT_EQ(pickArrow({100, 100}, base, flat, kAll, 0, 9), -1);
}
