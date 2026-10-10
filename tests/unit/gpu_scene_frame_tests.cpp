// gpu_scene_frame_tests.cpp - src/shared/gpu_scene_frame.h: the scale, centre and offset a GPU backend places a loaded pbrt scene in.
#include <gtest/gtest.h>

#include <cmath>

#include "../../src/shared/gpu_scene_frame.h"

using namespace gpu_scene_frame;

namespace {

void addBox(pbrt_flatten::FlatScene& s, double x0, double y0, double z0, double x1, double y1, double z1) {
	pbrt_flatten::Triangle a, b;
	const double p[6][3] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z1}, {x0, y0, z0}, {x1, y1, z1}, {x0, y1, z1}};
	for (int c = 0; c < 3; ++c)
		for (int k = 0; k < 3; ++k) {
			a.v[c * 3 + k] = p[c][k];
			b.v[c * 3 + k] = p[3 + c][k];
		}
	s.triangles.push_back(a);
	s.triangles.push_back(b);
}

}  // namespace

TEST(GpuSceneFrameTest, ACornellBoxSizedSceneIsScaledToTwoUnitsAndCentred) {
	pbrt_flatten::FlatScene s;
	addBox(s, 0, 0, 0, 555, 555, 555);
	const gpu_scene_frame::Frame f = computeSceneFrame(s);
	EXPECT_NEAR(f.extent, 555.0f, 1e-3f);
	EXPECT_NEAR(f.scale, 2.0f / 555.0f, 1e-7f);
	for (int a = 0; a < 3; ++a) EXPECT_NEAR(f.centre[a], 277.5f, 1e-3f);
	float world[3];
	f.toWorld(277.5f, 277.5f, 277.5f, world);   // the centre lands on the offset: +60 in X, nowhere else
	EXPECT_NEAR(world[0], 60.0f, 1e-4f);
	EXPECT_NEAR(world[1], 0.0f, 1e-4f);
	EXPECT_NEAR(world[2], 0.0f, 1e-4f);
	f.toWorld(555, 555, 555, world);   // a corner is one unit from the centre on each axis
	EXPECT_NEAR(world[0], 61.0f, 1e-4f);
	EXPECT_NEAR(world[1], 1.0f, 1e-4f);
}

TEST(GpuSceneFrameTest, FromWorldIsTheInverseOfToWorld) {
	pbrt_flatten::FlatScene s;
	addBox(s, -3, 1, 40, 17, 9, 90);
	const gpu_scene_frame::Frame f = computeSceneFrame(s);
	float w[3], back[3];
	f.toWorld(5.0f, 2.5f, 61.0f, w);
	f.fromWorld(w[0], w[1], w[2], back);
	EXPECT_NEAR(back[0], 5.0f, 1e-3f);
	EXPECT_NEAR(back[1], 2.5f, 1e-3f);
	EXPECT_NEAR(back[2], 61.0f, 1e-3f);
}

TEST(GpuSceneFrameTest, AHugeGroundSphereDoesNotSetTheScale) {
	pbrt_flatten::FlatScene s;
	pbrt_flatten::Sphere small, ground;
	small.center[1] = 1.0;       // radius 1 at height 1
	ground.center[1] = -1000.0;  // the radius-1000 ground idiom
	ground.radius = 1000.0;
	s.spheres = {small, ground};
	const gpu_scene_frame::Frame f = computeSceneFrame(s);
	EXPECT_TRUE(f.ignoredGiantSphere);
	EXPECT_NEAR(f.extent, 2.0f, 1e-4f) << "the small sphere's own box";
	EXPECT_GT(f.fullExtent, 1000.0f);
	EXPECT_NEAR(f.scale, 1.0f, 1e-4f);
}

TEST(GpuSceneFrameTest, ABigSphereThatIsMostOfTheSceneIsKept) {
	pbrt_flatten::FlatScene s;
	pbrt_flatten::Sphere big;
	big.radius = 100.0;
	s.spheres = {big};
	addBox(s, -90, -90, -90, 90, 90, 90);   // dropping the sphere would not shrink the extent 4x
	const gpu_scene_frame::Frame f = computeSceneFrame(s);
	EXPECT_FALSE(f.ignoredGiantSphere);
	EXPECT_NEAR(f.extent, 200.0f, 1e-3f);
}

TEST(GpuSceneFrameTest, AnEmptySceneIsLeftAlone) {
	const gpu_scene_frame::Frame f = computeSceneFrame(pbrt_flatten::FlatScene{});
	EXPECT_EQ(f.scale, 1.0f);
	for (int a = 0; a < 3; ++a) EXPECT_EQ(f.centre[a], 0.0f);
	EXPECT_EQ(f.offset[0], 60.0f);
}
