// scene_loader_transform_cache_test.cpp
//
// Regression test for the scene-loading caches (gpu/optix/scene_builder.cpp's
// get_or_build_cached()-based pbrt/image caches) - specifically the risk of one
// scene's per-placement transform leaking into another scene that shares the
// same underlying mesh file. Two scenes loading the *same* .obj at *different*
// scale/offset in one process would silently render one of them with the wrong
// geometry if a cache ever stored (or was corrupted into storing) TRANSFORMED
// vertex data keyed only by filename.
//
// This used to guard load_obj_triangles_gpu()'s process-lifetime raw-vertex
// cache; that loader was deleted with the G1-G24 mesh gallery, which is
// pbrt-backed now (pbrt_scenes/mesh-*.pbrt, each a Shape "plymesh" reading
// models/*.obj with its own Scale/Translate). The scenario is unchanged - the
// gallery still has two scenes on the same file at different transforms - and
// so is the contract, now exercised through the single shared pbrt mesh path.
//
// build_scene() is pure host-side C++ (no OptiX/CUDA device calls - see
// gpu_scene_light_count()'s own comment in optix_interface.h) and returns
// its SceneData by reference, so this test inspects the actual triangle
// vertex positions directly - no GPU, no rendering, no float noise from
// Monte Carlo sampling to tolerance around.
//
// Scenes G1 ("Stanford Bunny": mesh-stanford-bunny.pbrt) and G12 ("Trophy
// Room": mesh-trophy-room.pbrt) both read models/stanford-bunny.obj, at
// different scale/offset (19.4 / (0.3267,-0.6398,0.0298) vs 10.3467 /
// (-3.32576,-0.34123,0.01589)). G1 is only the bunny, and in G12 the bunny is
// the FIRST mesh in the file, so its triangles are the first
// bunnyG1.size() triangles of G12's list.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}
#include "scene_builder.h"

namespace {

bool SameTriangle(const TriangleData& a, const TriangleData& b) {
	return a.p0.x == b.p0.x && a.p0.y == b.p0.y && a.p0.z == b.p0.z &&
	       a.p1.x == b.p1.x && a.p1.y == b.p1.y && a.p1.z == b.p1.z &&
	       a.p2.x == b.p2.x && a.p2.y == b.p2.y && a.p2.z == b.p2.z;
}

} // namespace

TEST(SceneLoaderTransformCacheTest, SameObjFileDifferentTransformsDoNotCrossContaminate) {
	float cameraParams[16] = {};

	SceneData sceneG1First;
	ASSERT_TRUE(build_scene("G1", 64, 64, sceneG1First, cameraParams));
	const std::vector<TriangleData> bunnyG1First = sceneG1First.triangles;
	ASSERT_FALSE(bunnyG1First.empty()) << "G1 (Stanford Bunny) produced no triangles - is models/stanford-bunny.obj present?";

	// Load a DIFFERENT scene that shares the same underlying .obj file but
	// applies a different scale/offset.
	SceneData sceneG12;
	ASSERT_TRUE(build_scene("G12", 64, 64, sceneG12, cameraParams));
	ASSERT_GT(sceneG12.triangles.size(), bunnyG1First.size())
		<< "G12 (Trophy Room) should hold the bunny plus three more meshes";

	// Re-load G1 - if one scene's transform had leaked into a shared cache
	// entry, this second G1 load would now differ from the first.
	SceneData sceneG1Second;
	ASSERT_TRUE(build_scene("G1", 64, 64, sceneG1Second, cameraParams));
	const std::vector<TriangleData>& bunnyG1Second = sceneG1Second.triangles;

	ASSERT_EQ(bunnyG1First.size(), bunnyG1Second.size())
		<< "G1's triangle count changed after loading G12 in between";
	for (size_t i = 0; i < bunnyG1First.size(); ++i) {
		const TriangleData& a = bunnyG1First[i];
		const TriangleData& b = bunnyG1Second[i];
		EXPECT_FLOAT_EQ(a.p0.x, b.p0.x) << "triangle " << i << " p0.x";
		EXPECT_FLOAT_EQ(a.p0.y, b.p0.y) << "triangle " << i << " p0.y";
		EXPECT_FLOAT_EQ(a.p0.z, b.p0.z) << "triangle " << i << " p0.z";
		EXPECT_FLOAT_EQ(a.p1.x, b.p1.x) << "triangle " << i << " p1.x";
		EXPECT_FLOAT_EQ(a.p1.y, b.p1.y) << "triangle " << i << " p1.y";
		EXPECT_FLOAT_EQ(a.p1.z, b.p1.z) << "triangle " << i << " p1.z";
		EXPECT_FLOAT_EQ(a.p2.x, b.p2.x) << "triangle " << i << " p2.x";
		EXPECT_FLOAT_EQ(a.p2.y, b.p2.y) << "triangle " << i << " p2.y";
		EXPECT_FLOAT_EQ(a.p2.z, b.p2.z) << "triangle " << i << " p2.z";
	}

	// Sanity check that G12's own transform is genuinely different (not
	// coincidentally identical, which would make the test above vacuous):
	// G1's scale (19.4) and G12's scale (10.3467) for the same raw bunny must
	// put at least one of its triangles somewhere else. G12's bunny is its
	// first mesh, so compare the first bunnyG1First.size() triangles.
	size_t differing = 0;
	for (size_t i = 0; i < bunnyG1First.size(); ++i) {
		if (!SameTriangle(bunnyG1First[i], sceneG12.triangles[i])) ++differing;
	}
	EXPECT_GT(differing, bunnyG1First.size() / 2)
		<< "G12's bunny sits where G1's does - the two transforms should differ";
}
