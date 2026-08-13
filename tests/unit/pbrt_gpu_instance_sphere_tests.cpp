/**
 * @file pbrt_gpu_instance_sphere_tests.cpp
 * @brief Spheres inside an ObjectBegin block reach the GPU builder.
 *
 * These used to be dropped. Spheres are custom AABB primitives, so instancing
 * them needs a second per-group acceleration structure and a hit record the
 * scene's own packed custom-primitive region cannot supply positionally - and
 * until that existed the builder counted them as unsupported and emitted
 * nothing. A scene whose only spheres were inside a definition therefore
 * rendered without them on GPU while rendering correctly on CPU.
 *
 * What is pinned here is the builder's half of the contract, which is what
 * the renderer reads: the spheres exist, they are stored ONCE however many
 * times they are placed, they stay in the definition's object space, and the
 * group's base/count actually address them. Object space is the load-bearing
 * part - it is what lets traversal apply a non-uniform placement scale and
 * produce an ellipsoid, which baking a world-space sphere per placement could
 * never express.
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

#include <cmath>

namespace {

// One definition holding a sphere AND a triangle, placed twice - the second
// placement with a non-uniform scale. The mixed content matters: the two kinds
// cannot share a GAS, so this is also the case that makes one placement become
// two entries in the instance hierarchy.
pbrt_flatten::FlatScene flattenInstancedSpheres() {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"ObjectBegin \"cairn\"\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ -10 0 -10   10 0 -10   10 0 10 ]\n"
		"  AttributeBegin\n"
		"    Translate 0 30 0\n"
		"    Shape \"sphere\" \"float radius\" [ 20 ]\n"
		"  AttributeEnd\n"
		"ObjectEnd\n"
		"AttributeBegin\n"
		"  Translate 100 0 100\n"
		"  ObjectInstance \"cairn\"\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 400 0 100\n"
		"  Scale 0.5 2 0.5\n"
		"  ObjectInstance \"cairn\"\n"
		"AttributeEnd\n");
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

} // namespace

TEST(PbrtGpuInstanceSphereTest, SpheresInADefinitionAreNotDropped) {
	const pbrt_flatten::FlatScene flat = flattenInstancedSpheres();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	EXPECT_EQ(scene.instanceSpheres.size(), 1u)
		<< "the definition's sphere should have reached the GPU scene";
	ASSERT_EQ(scene.instanceGroups.size(), 1u);
	EXPECT_EQ(scene.instanceGroups[0].sphereCount, 1);
}

TEST(PbrtGpuInstanceSphereTest, TwoPlacementsStillStoreTheSphereOnce) {
	const pbrt_flatten::FlatScene flat = flattenInstancedSpheres();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	// The whole point of instancing over baking: placing it twice must not
	// cost two spheres' worth of geometry.
	EXPECT_EQ(scene.instancePlacements.size(), 2u);
	EXPECT_EQ(scene.instanceSpheres.size(), 1u);
	EXPECT_TRUE(scene.spheres.empty())
		<< "an instanced sphere must not also be emitted as world-space geometry";
}

TEST(PbrtGpuInstanceSphereTest, TheStoredSphereIsInObjectSpaceNotAnyPlacement) {
	const pbrt_flatten::FlatScene flat = flattenInstancedSpheres();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_EQ(scene.instanceSpheres.size(), 1u);
	const SphereData &s = scene.instanceSpheres[0];

	// The definition puts it at (0,30,0) with radius 20. Either placement's
	// translation would move it hundreds of units away, and the second's scale
	// would change the radius - seeing the untouched values is what proves the
	// transform is left for traversal to apply rather than baked in here.
	EXPECT_NEAR(s.center.x, 0.0f, 1e-4f);
	EXPECT_NEAR(s.center.y, 30.0f, 1e-4f);
	EXPECT_NEAR(s.center.z, 0.0f, 1e-4f);
	EXPECT_NEAR(s.radius, 20.0f, 1e-4f);
}

TEST(PbrtGpuInstanceSphereTest, AMixedDefinitionCarriesBothKindsWithUsableBases) {
	const pbrt_flatten::FlatScene flat = flattenInstancedSpheres();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_EQ(scene.instanceGroups.size(), 1u);
	const SceneData::InstanceGroupGPU &g = scene.instanceGroups[0];

	EXPECT_EQ(g.triangleCount, 1) << "the definition's triangle should survive too";
	EXPECT_EQ(g.sphereCount, 1);

	// base + count must actually address the arrays - the renderer indexes
	// them directly when it builds each group's acceleration structure.
	EXPECT_GE(g.triangleBase, 0);
	EXPECT_LE(static_cast<std::size_t>(g.triangleBase + g.triangleCount),
			  scene.instanceTriangles.size());
	EXPECT_GE(g.sphereBase, 0);
	EXPECT_LE(static_cast<std::size_t>(g.sphereBase + g.sphereCount),
			  scene.instanceSpheres.size());
}

TEST(PbrtGpuInstanceSphereTest, PlacementTransformsReachTheRendererDistinctly) {
	const pbrt_flatten::FlatScene flat = flattenInstancedSpheres();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_EQ(scene.instancePlacements.size(), 2u);
	// OptiX 3x4 row-major: the translation is the last column of each row.
	EXPECT_NEAR(scene.instancePlacements[0].transform[3], 100.0f, 1e-4f);
	EXPECT_NEAR(scene.instancePlacements[1].transform[3], 400.0f, 1e-4f);

	// The second placement's non-uniform scale must survive as a scale, since
	// that is exactly what a baked sphere could not represent. Row 1 col 1 is
	// the y scale; rows 0/2 keep the 0.5 in x and z.
	EXPECT_NEAR(scene.instancePlacements[1].transform[0], 0.5f, 1e-4f);
	EXPECT_NEAR(scene.instancePlacements[1].transform[5], 2.0f, 1e-4f);
	EXPECT_NEAR(scene.instancePlacements[1].transform[10], 0.5f, 1e-4f);

	// Both placements point at the one group, rather than each having gained
	// a private copy of the geometry.
	EXPECT_EQ(scene.instancePlacements[0].group, 0);
	EXPECT_EQ(scene.instancePlacements[1].group, 0);
}
