/**
 * @file live_object_edit_tests.cpp
 * @brief Unit tests for picking and moving the objects of a flattened pbrt scene (src/shared/live_object_edit.h), the Live Preview's object editing.
 */

#include <gtest/gtest.h>

#include "live_object_edit.h"
#include "pbrt_flatten.h"
#include "pbrt_scene.h"

#include <cmath>
#include <string>

using namespace live_objects;

namespace {

pbrt_flatten::FlatScene flattenText(const std::string &text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene, {});
}

// A sphere at the origin, a quad on the floor (z = 0, x and y from 3 to 5), a cylinder along z at x = -4.
const char *kScene =
	"WorldBegin\n"
	"Shape \"sphere\" \"float radius\" [ 1 ]\n"
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ] \"point3 P\" [ 3 3 0  5 3 0  5 5 0  3 5 0 ]\n"
	"AttributeBegin\n  Translate -4 0 0\n  Shape \"cylinder\" \"float radius\" [ 0.5 ] \"float zmin\" [ 0 ] \"float zmax\" [ 2 ]\nAttributeEnd\n";

}  // namespace

TEST(LiveObjectEditTest, PickFindsTheShapeASurfacePointBelongsTo) {
	const pbrt_flatten::FlatScene s = flattenText(kScene);
	ASSERT_EQ(objectsOf(s).size(), 3u);
	const PickIndex index = buildPickIndex(s, objectsOf(s));
	const double onSphere[3] = {0, 0, 1}, onQuad[3] = {4, 4, 0}, onCylinder[3] = {-3.5, 0, 1};
	EXPECT_EQ(pick(index, onSphere, 0.01).object, 0);
	EXPECT_EQ(pick(index, onQuad, 0.01).object, 1);
	EXPECT_EQ(pick(index, onCylinder, 0.05).object, 2);
}

TEST(LiveObjectEditTest, PickMissesEmptySpace) {
	const pbrt_flatten::FlatScene s = flattenText(kScene);
	const PickIndex index = buildPickIndex(s, objectsOf(s));
	const double far[3] = {20, 20, 20}, nearButOff[3] = {0, 0, 1.5};
	EXPECT_EQ(pick(index, far, 0.05).object, -1);
	EXPECT_EQ(pick(index, nearButOff, 0.05).object, -1) << "0.5 above the sphere is not on it";
}

TEST(LiveObjectEditTest, MovingAShapeMovesItsPickAndLeavesTheOthers) {
	pbrt_flatten::FlatScene s = flattenText(kScene);
	const double d[3] = {0, 0, 10};
	for (std::size_t i = 0; i < objectsOf(s).size(); ++i) {
		pbrt_flatten::FlatScene m = s;
		ASSERT_TRUE(translateObject(m, objectsOf(m), i, d));
		const PickIndex before = buildPickIndex(s, objectsOf(s)), after = buildPickIndex(m, objectsOf(m));
		for (std::size_t k = 0; k < before.shapes.size(); ++k) {
			const double shift = (k == i) ? 10.0 : 0.0;
			EXPECT_NEAR(after.shapes[k].lo[2], before.shapes[k].lo[2] + shift, 1e-9) << "moved " << i << " looked at " << k;
			EXPECT_NEAR(after.shapes[k].hi[2], before.shapes[k].hi[2] + shift, 1e-9);
			EXPECT_NEAR(after.shapes[k].lo[0], before.shapes[k].lo[0], 1e-9);
		}
	}
}

TEST(LiveObjectEditTest, MovedSphereIsPickedAtItsNewPlace) {
	pbrt_flatten::FlatScene s = flattenText(kScene);
	const double d[3] = {0, 0, 10};
	ASSERT_TRUE(translateObject(s, objectsOf(s), 0, d));
	const PickIndex index = buildPickIndex(s, objectsOf(s));
	const double oldTop[3] = {0, 0, 1}, newTop[3] = {0, 0, 11};
	EXPECT_EQ(pick(index, oldTop, 0.01).object, -1);
	EXPECT_EQ(pick(index, newTop, 0.01).object, 0);
}

TEST(LiveObjectEditTest, TranslateRejectsAnIndexOutOfRange) {
	pbrt_flatten::FlatScene s = flattenText(kScene);
	const double d[3] = {1, 0, 0};
	EXPECT_FALSE(translateObject(s, objectsOf(s), 99, d));
}

TEST(LiveObjectEditTest, ASmallObjectOnABigOneIsPickedNotTheBigOne) {
	// A ball resting on a floor: a point on the ball's surface where it touches the floor is on both; the smaller shape wins.
	const pbrt_flatten::FlatScene s = flattenText(
		"WorldBegin\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ] \"point3 P\" [ -10 -10 0  10 -10 0  10 10 0  -10 10 0 ]\n"
		"Translate 0 0 1\nShape \"sphere\" \"float radius\" [ 1 ]\n");
	const PickIndex index = buildPickIndex(s, objectsOf(s));
	const double touch[3] = {0, 0, 0};
	EXPECT_EQ(pick(index, touch, 0.01).object, 1);
}

TEST(LiveObjectEditTest, ShapesOfOneAttributeBlockAreOneObject) {
	// A box written as two quads in a block is one object; a second block is another; a bare shape outside any block is its own.
	pbrt_flatten::FlatScene s = flattenText(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ] \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ] \"point3 P\" [ 0 0 1  1 0 1  1 1 1  0 1 1 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n  Translate 5 0 0\n  Shape \"sphere\" \"float radius\" [ 1 ]\nAttributeEnd\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	const ObjectList objects = objectsOf(s);
	ASSERT_EQ(objects.size(), 3u);
	EXPECT_EQ(objects[0].size(), 2u);
	EXPECT_EQ(objects[1].size(), 1u);
	EXPECT_EQ(objects[2].size(), 1u);
	// Moving the first moves both quads (z of every vertex), and nothing else.
	const double d[3] = {0, 0, 7};
	ASSERT_TRUE(translateObject(s, objects, 0, d));
	for (const pbrt_flatten::Triangle &t : s.triangles)
		for (int v = 0; v < 3; ++v) EXPECT_GE(t.v[v * 3 + 2], 7.0);
	EXPECT_NEAR(s.spheres[0].center[2], 0.0, 1e-12);
	EXPECT_NEAR(s.spheres[1].center[2], 0.0, 1e-12);
}

TEST(LiveObjectEditTest, TheNestedBlockWinsOverItsParent) {
	const pbrt_flatten::FlatScene s = flattenText(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"  AttributeBegin\n    Translate 4 0 0\n    Shape \"sphere\" \"float radius\" [ 1 ]\n  AttributeEnd\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	const ObjectList objects = objectsOf(s);
	ASSERT_EQ(objects.size(), 2u) << "the outer block's two spheres (not adjacent in the file) are one object, the nested block's sphere another";
	EXPECT_EQ(objects[0].size(), 2u);
}
