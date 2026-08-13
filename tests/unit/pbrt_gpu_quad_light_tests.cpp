/**
 * @file pbrt_gpu_quad_light_tests.cpp
 * @brief Regression test for QuadData::w's convention in the pbrt GPU builder
 *
 * Every loaded pbrt scene with an area light rendered pure black on GPU.
 * gpu/optix/optix_device_helpers.h's sample_quad_light() computes the light's
 * area as `length(quad.w)`, on the documented assumption "w = u x v, so
 * |w| = area" - true for every OTHER GPU quad builder in scene_builder.cpp
 * (grep `quad.w = quad_cross`), which all set w to the raw cross product.
 *
 * pbrt_gpu_builder.h instead set w = n / dot(n,n) - the CPU-side RTIOW
 * quad.h convention, used there for an unrelated barycentric-coordinate
 * trick. Its LENGTH is 1/area, not area. Reading that as area silently
 * inverted the light's solid-angle pdf, scaling it by area^2 (~1.86e8 for
 * a typical Cornell-box-scale light quad) - dividing radiance by a pdf a
 * hundred million times too large is indistinguishable from no light at
 * all once quantized to 8 bits, which is exactly what made this look like
 * "the scene is black" rather than "the light is wrong".
 *
 * Confirmed by dumping the raw pre-tonemap framebuffer during diagnosis:
 * values were finite, positive and real (not NaN, not exactly zero) but
 * capped at ~4.7e-7 for a light with L=(18,15,8) - light was reaching every
 * surface, just at roughly a hundred-millionth of its true magnitude.
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

#include <cmath>

namespace {

pbrt_flatten::FlatScene flattenQuadLight() {
	// A rectangular area light, sized like the bundled example scene's
	// ceiling light (130 x 105 world units - large enough that a 1/area vs
	// area mixup is a difference of ~1.86e8, not something float noise could
	// hide).
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 18 15 8 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ 213 548.7 227   343 548.7 227   343 548.7 332   213 548.7 332 ]\n"
		"AttributeEnd\n");
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

} // namespace

TEST(PbrtGpuQuadLightTest, QuadWLengthEqualsAreaNotItsReciprocal) {
	const pbrt_flatten::FlatScene flat = flattenQuadLight();
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);

	ASSERT_EQ(stats.quadLights, 1u) << "the two triangles should have merged into one quad";
	ASSERT_EQ(scene.quads.size(), 1u);

	const QuadData &q = scene.quads[0];
	const float3 crossProduct = cross(q.u, q.v);
	const float trueArea = length(crossProduct);
	const float wLength = length(q.w);

	// The actual regression: gpu/optix/optix_device_helpers.h's
	// sample_quad_light() reads `area = length(quad.w)` directly - so this
	// assertion is not an implementation detail, it is the exact contract
	// that function depends on.
	EXPECT_NEAR(wLength, trueArea, trueArea * 1e-4f)
		<< "quad.w's length must equal the quad's area (|u x v|), because "
		   "sample_quad_light() reads it as exactly that. Got |w|=" << wLength
		<< " but the true area is " << trueArea
		<< " (ratio " << (trueArea > 0 ? wLength / trueArea : 0.0f) << " - "
		<< "a ratio near 1/area^2 means w was built as the RTIOW n/dot(n,n) "
		   "reciprocal instead of the raw cross product).";

	// Sized like the bundled example's ceiling light: large enough that the
	// bug (1/area instead of area) cannot pass by coincidence.
	EXPECT_GT(trueArea, 1000.0f);
}

TEST(PbrtGpuQuadLightTest, WPointsTheSameWayAsTheCrossProduct) {
	// w's magnitude is the load-bearing property, but a correct-length vector
	// pointing the wrong way would still be a real bug (it would flip the
	// light-facing test in sample_quad_light/quad_light_pdf) - so direction is
	// checked too, not just magnitude.
	const pbrt_flatten::FlatScene flat = flattenQuadLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.quads.size(), 1u);

	const QuadData &q = scene.quads[0];
	const float3 crossProduct = cross(q.u, q.v);
	const float alignment = dot(q.w, crossProduct) / (length(q.w) * length(crossProduct));
	EXPECT_NEAR(alignment, 1.0f, 1e-4f) << "quad.w points the wrong way relative to u x v";
}

TEST(PbrtGpuQuadLightTest, NormalIsUnitLengthAndPerpendicularToTheQuad) {
	// normal and w come from the same cross product in pbrt_gpu_builder.h;
	// pinning normal's own correctness rules it out if w's test above ever
	// fails and someone "fixes" it by touching the wrong line.
	const pbrt_flatten::FlatScene flat = flattenQuadLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.quads.size(), 1u);

	const QuadData &q = scene.quads[0];
	EXPECT_NEAR(length(q.normal), 1.0f, 1e-5f);
	EXPECT_NEAR(dot(q.normal, q.u), 0.0f, 1e-3f);
	EXPECT_NEAR(dot(q.normal, q.v), 0.0f, 1e-3f);
}
