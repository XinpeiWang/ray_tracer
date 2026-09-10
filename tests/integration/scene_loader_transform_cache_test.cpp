// scene_loader_transform_cache_test.cpp
//
// Regression test for the Live Preview large-scene performance fix
// (gpu/optix/scene_builder.cpp's get_or_build_cached()-based OBJ/pbrt/image
// caches) - specifically the risk that fix's own code review flagged but
// didn't have a test for: load_obj_triangles_gpu() caches an OBJ file's RAW,
// UNTRANSFORMED vertex data keyed only by filename, then re-applies each
// caller's own scale/offset fresh on every call (see that function's
// implementation). If a future change accidentally started caching the
// TRANSFORMED result instead - or let one caller's transform leak into
// another's - two scenes loading the *same* .obj file at *different*
// scale/offset in the same process (a real, already-existing situation: see
// below) would silently render one of them with the wrong geometry.
//
// build_scene() is pure host-side C++ (no OptiX/CUDA device calls - see
// gpu_scene_light_count()'s own comment in optix_interface.h) and returns
// its SceneData by reference, so this test inspects the actual triangle
// vertex positions directly - no GPU, no rendering, no float noise from
// Monte Carlo sampling to tolerance around.
//
// Scenes G1 ("Stanford Bunny", scene_builder_mesh_gallery.h's
// build_stanford_bunny_gpu) and G12 ("Trophy Room", build_trophy_room_gpu)
// both load models/stanford-bunny.obj via load_obj_triangles_gpu(), but at
// different scale/offset (19.4 / (0.3267,-0.6398,0.0298) vs 10.3467 /
// (-3.32576,-0.34123,0.01589)) - exactly the same-file-different-transform
// scenario this test needs, already present in the existing scene gallery
// rather than invented for this test. In both scenes the bunny is the
// second material added (index 1, right after the ground's checker
// lambertian at index 0), so its triangles are found by materialIdx == 1.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}
#include "scene_builder.h"

namespace {

constexpr int kBunnyMaterialIdx = 1;

std::vector<TriangleData> BunnyTriangles(const SceneData& scene) {
	std::vector<TriangleData> bunny;
	for (const TriangleData& tri : scene.triangles) {
		if (tri.materialIdx == kBunnyMaterialIdx) bunny.push_back(tri);
	}
	return bunny;
}

} // namespace

TEST(SceneLoaderTransformCacheTest, SameObjFileDifferentTransformsDoNotCrossContaminate) {
	float cameraParams[16] = {};

	SceneData sceneG1First;
	ASSERT_TRUE(build_scene("G1", 64, 64, sceneG1First, cameraParams));
	const std::vector<TriangleData> bunnyG1First = BunnyTriangles(sceneG1First);
	ASSERT_FALSE(bunnyG1First.empty()) << "G1 (Stanford Bunny) produced no bunny triangles at materialIdx "
										 << kBunnyMaterialIdx << " - scene layout may have changed";

	// Load a DIFFERENT scene that shares the same underlying .obj file but
	// applies a different scale/offset - this is what exercises the
	// process-lifetime s_objCache added for the large-scene perf fix.
	SceneData sceneG12;
	ASSERT_TRUE(build_scene("G12", 64, 64, sceneG12, cameraParams));
	const std::vector<TriangleData> bunnyG12 = BunnyTriangles(sceneG12);
	ASSERT_FALSE(bunnyG12.empty()) << "G12 (Trophy Room) produced no bunny triangles at materialIdx "
									 << kBunnyMaterialIdx << " - scene layout may have changed";

	// Re-load G1 - if the cache incorrectly stored (or was corrupted into
	// storing) transformed geometry, or if G12's own scale/offset leaked
	// into the shared cache entry, this second G1 load would now differ
	// from the first.
	SceneData sceneG1Second;
	ASSERT_TRUE(build_scene("G1", 64, 64, sceneG1Second, cameraParams));
	const std::vector<TriangleData> bunnyG1Second = BunnyTriangles(sceneG1Second);

	ASSERT_EQ(bunnyG1First.size(), bunnyG1Second.size())
		<< "G1's bunny triangle count changed after loading G12 in between - "
		<< "the shared OBJ cache is not correctly isolated per-caller.";
	// Same underlying raw file also means the same triangle count as the
	// scene that shares it, since load_obj_triangles_gpu() applies no
	// culling/decimation - only a per-face affine transform.
	EXPECT_EQ(bunnyG1First.size(), bunnyG12.size());

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
	// G1's scale (19.4) and G12's scale (10.3467) for the same raw mesh
	// must put at least one corresponding vertex at a visibly different
	// position.
	bool foundDifference = false;
	for (size_t i = 0; i < bunnyG1First.size() && i < bunnyG12.size(); ++i) {
		if (std::abs(bunnyG1First[i].p0.x - bunnyG12[i].p0.x) > 0.01f ||
			std::abs(bunnyG1First[i].p0.y - bunnyG12[i].p0.y) > 0.01f ||
			std::abs(bunnyG1First[i].p0.z - bunnyG12[i].p0.z) > 0.01f) {
			foundDifference = true;
			break;
		}
	}
	EXPECT_TRUE(foundDifference)
		<< "G1 and G12's bunny geometry is identical despite different scale/offset - "
		<< "this test would not catch a caching regression.";
}
