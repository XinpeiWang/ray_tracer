/**
 * @file pbrt_gpu_light_lookup_tests.cpp
 * @brief The GPU light list is sorted by gpu_light_sort_key, so a hit emitter finds its light by binary search.
 *
 * The wavefront backend maps the primitive a bounce ray hit back to its light (wf_emitter_hit_mis_weight) with
 * gpu_find_light(), which only works while lightIndices/lightKinds are sorted by gpu_light_sort_key. The pbrt
 * builder registers lights in that order already and sorts as a safeguard; these pin both halves.
 */

#include <gtest/gtest.h>

#include <vector>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

TEST(PbrtGpuLightLookupTest, BuilderLightListIsSortedAndEveryLightIsFound) {
	// A sphere light, a quad light and a lone non-parallelogram triangle light: three kinds in one scene.
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 0 2 3 ] \"point3 P\" [ 0 5 0  1 5 0  1 5 1  0 5 1 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 3 3 3 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ] \"point3 P\" [ 4 5 0  6 5 0  4.5 5 3 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 2 2 2 ]\n"
		"  Shape \"sphere\" \"float radius\" [ 0.5 ]\n"
		"AttributeEnd\n");
	ASSERT_TRUE(r.ok) << r.error;
	SceneData scene;
	pbrt_gpu::build(pbrt_flatten::flatten(r.scene), scene);

	const size_t n = scene.lightIndices.size();
	ASSERT_GE(n, 3u);
	ASSERT_EQ(scene.lightKinds.size(), n);
	for (size_t i = 1; i < n; ++i)
		EXPECT_LT(gpu_light_sort_key(scene.lightKinds[i - 1], scene.lightIndices[i - 1]),
				  gpu_light_sort_key(scene.lightKinds[i], scene.lightIndices[i]))
			<< "lights " << i - 1 << " and " << i << " are out of order";
	for (size_t i = 0; i < n; ++i)
		EXPECT_EQ(gpu_find_light(scene.lightIndices.data(), scene.lightKinds.data(), (unsigned)n,
								 scene.lightKinds[i], scene.lightIndices[i]), (int)i);
}

TEST(PbrtGpuLightLookupTest, FindLightReturnsMinusOneForAPrimitiveThatIsNotALight) {
	const std::vector<int> idx = { 2, 7, 0, 3 };
	const std::vector<GpuLightKind> kinds = { GpuLightKind::Sphere, GpuLightKind::Sphere, GpuLightKind::Quad,
											  GpuLightKind::Triangle };
	// Sorted by key: Sphere 2, Sphere 7, Quad 0, Triangle 3.
	EXPECT_EQ(gpu_find_light(idx.data(), kinds.data(), 4, GpuLightKind::Sphere, 7), 1);
	EXPECT_EQ(gpu_find_light(idx.data(), kinds.data(), 4, GpuLightKind::Triangle, 3), 3);
	EXPECT_EQ(gpu_find_light(idx.data(), kinds.data(), 4, GpuLightKind::Quad, 3), -1)   // right index, wrong kind
		<< "the same primitive index in a different array is a different primitive";
	EXPECT_EQ(gpu_find_light(idx.data(), kinds.data(), 4, GpuLightKind::Sphere, 5), -1);
	EXPECT_EQ(gpu_find_light(idx.data(), kinds.data(), 0, GpuLightKind::Sphere, 2), -1);
}
