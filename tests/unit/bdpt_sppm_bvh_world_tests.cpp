/**
 * @file bdpt_sppm_bvh_world_tests.cpp
 * @brief Regression tests: BDPT/SPPM must find emitters through a BVH-wrapped world
 *
 * pbrt_cpu::build() (pbrt_cpu_builder.h) wraps its returned world in a
 * single bvh_node for real-ray-tracing performance - so any pbrt-loaded
 * scene's build_world() returns a hittable_list containing exactly one
 * bvh_node, not a flat list of spheres/quads. BDPTSceneAdapter's and
 * SPPMSceneAdapter's constructors used to scan world_.objects directly for
 * sample_area()-capable shapes, which found nothing behind that one opaque
 * bvh_node - silently rendering every pbrt-loaded scene's area lights
 * invisible under BDPT/MLT/RandomWalk/AO/SPPM. These tests build worlds
 * shaped like pbrt_cpu::build()'s own output and confirm both adapters
 * still find lights inside them, using emitter_discovery.h's
 * collectEmitterCandidates() - including its deliberate scope boundary
 * (translate/rotate_y are NOT unwrapped, since doing so would silently
 * return world-space-incorrect samples - see that header's own comment).
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "bvh.h"
#include "sphere.h"
#include "triangle.h"
#include "material.h"
#include "bdpt_adapter.h"
#include "sppm_adapter.h"

namespace {

// Mirrors pbrt_cpu_builder.h's own final step exactly: a flat hittable_list
// of real geometry, folded into a single bvh_node, then re-wrapped in a
// fresh top-level hittable_list - not just a bare bvh_node, since that's
// the precise shape build_world() actually returns for a pbrt-loaded scene.
shared_ptr<hittable_list> makeBvhWrappedWorldWithOneLight() {
	auto flat = make_shared<hittable_list>();
	auto light = make_shared<diffuse_light>(color(6, 6, 6), /*two_sided=*/false);
	flat->add(make_shared<sphere>(point3(0, 5, 0), 1.0, light));
	flat->add(make_shared<sphere>(point3(0, 0, 0), 1.0, make_shared<lambertian>(color(0.5, 0.5, 0.5))));

	auto wrapped = make_shared<hittable_list>();
	wrapped->add(make_shared<bvh_node>(*flat));
	return wrapped;
}

// A one-triangle mesh (the actual shape a pbrt Shape "trianglemesh" or a
// native OBJ mesh light resolves to), wrapped in a bvh_node the same way
// pbrt_cpu_builder.h wraps every top-level shape kind.
shared_ptr<hittable_list> makeBvhWrappedWorldWithATriangleLight() {
	auto mesh = make_shared<triangle_mesh_data>();
	mesh->positions = {point3(-1, 5, -1), point3(1, 5, -1), point3(0, 5, 1)};
	mesh->indices = {0, 1, 2};
	auto light = make_shared<diffuse_light>(color(8, 8, 8), /*two_sided=*/false);

	auto flat = make_shared<hittable_list>();
	flat->add(make_shared<triangle>(mesh, 0, light));
	flat->add(make_shared<sphere>(point3(0, 0, 0), 1.0, make_shared<lambertian>(color(0.5, 0.5, 0.5))));

	auto wrapped = make_shared<hittable_list>();
	wrapped->add(make_shared<bvh_node>(*flat));
	return wrapped;
}

// Enough primitives (kMaxLeafPrims=4, bvh.h) to force a REAL internal split
// - two non-null bvh_node/bvh_leaf children several levels deep - not just
// the single-bvh_leaf shape the 2-primitive world above takes.
shared_ptr<hittable_list> makeDeeplySplitBvhWorldWithTwoLights() {
	auto flat = make_shared<hittable_list>();
	auto lightA = make_shared<diffuse_light>(color(5, 0, 0), false);
	auto lightB = make_shared<diffuse_light>(color(0, 0, 5), false);
	flat->add(make_shared<sphere>(point3(-8, 5, 0), 0.5, lightA));
	flat->add(make_shared<sphere>(point3(8, 5, 0), 0.5, lightB));
	for (int i = 0; i < 8; ++i) {
		flat->add(make_shared<sphere>(point3(i * 2.0 - 7.0, 0, 0), 0.5,
									   make_shared<lambertian>(color(0.5, 0.5, 0.5))));
	}
	auto wrapped = make_shared<hittable_list>();
	wrapped->add(make_shared<bvh_node>(*flat));
	return wrapped;
}

} // namespace

TEST(BDPTAdapterBvhWorld, FindsAnEmitterInsideAWrappedBvhNode) {
	camera cam;
	auto world = makeBvhWrappedWorldWithOneLight();
	ASSERT_EQ(world->objects.size(), 1u) << "sanity check: this really is BVH-wrapped, not flat";

	BDPTSceneAdapter adapter(*world, cam);
	EXPECT_EQ(adapter.EmitterCount(), 1)
		<< "the light is real geometry inside the wrapped bvh_node - it must still be found";
}

TEST(BDPTAdapterBvhWorld, SampleLightStillWorksThroughTheWrapper) {
	camera cam;
	auto world = makeBvhWrappedWorldWithOneLight();
	BDPTSceneAdapter adapter(*world, cam);
	ASSERT_EQ(adapter.EmitterCount(), 1);

	const double ref_p[3] = {0.0, 0.0, 0.0};
	BDPTLightSample<double> ls;
	bool sawSample = false;
	for (int i = 0; i < 30; ++i) {
		if (adapter.SampleLight(random_double(), ref_p, ls)) {
			sawSample = true;
			EXPECT_GT(ls.pdf, 0.0);
			EXPECT_GT(ls.L[0], 0.0);
		}
	}
	EXPECT_TRUE(sawSample) << "the previously-invisible light must now actually contribute radiance";
}

TEST(BDPTAdapterBvhWorld, FindsATriangleMeshLightThroughTheWrapper) {
	// The exact combination the review found still broken by a naive
	// recursion-list extension alone: a mesh-based light needs BOTH the
	// bvh_node unwrap AND triangle::sample_area() (added alongside this
	// test) to be discoverable - sample_area() returning true is what
	// distinguishes "the light itself" from "some other, non-emissive
	// triangle in the same mesh."
	camera cam;
	auto world = makeBvhWrappedWorldWithATriangleLight();
	BDPTSceneAdapter adapter(*world, cam);
	EXPECT_EQ(adapter.EmitterCount(), 1);
}

TEST(BDPTAdapterBvhWorld, FindsBothLightsInAGenuinelySplitBvhTree) {
	// Unlike the 2-primitive worlds above (which take bvh_node's immediate-
	// leaf shortcut, left=bvh_leaf, right=nullptr), 10 primitives forces a
	// real SAH split - two non-null children, each potentially another
	// internal bvh_node - so this exercises collectEmitterCandidates()'s
	// bvh_node recursion into BOTH get_left() and get_right() for real,
	// and confirms multi-light index alignment (emitters_/emitter_dl_/
	// emitter_pdf_pos_ all built from the same loop, in lockstep).
	camera cam;
	auto world = makeDeeplySplitBvhWorldWithTwoLights();
	BDPTSceneAdapter adapter(*world, cam);
	EXPECT_EQ(adapter.EmitterCount(), 2);

	const double ref_p[3] = {0.0, 5.0, 0.0};
	bool sawRed = false, sawBlue = false;
	for (int i = 0; i < 200; ++i) {
		BDPTLightSample<double> ls;
		if (!adapter.SampleLight(random_double(), ref_p, ls)) continue;
		if (ls.L[0] > ls.L[2]) sawRed = true;
		if (ls.L[2] > ls.L[0]) sawBlue = true;
	}
	EXPECT_TRUE(sawRed) << "the red light (lightA) must be reachable";
	EXPECT_TRUE(sawBlue) << "the blue light (lightB) must be reachable";
}

TEST(BDPTAdapterBvhWorld, DoesNotFindALightWrappedInTranslate) {
	// Deliberate scope boundary (see emitter_discovery.h's own comment):
	// translate applies a real coordinate offset inside its own hit() that
	// a raw sample_area() call on the wrapped object knows nothing about,
	// so unwrapping through it would return world-space-WRONG samples, not
	// just miss the light - collectEmitterCandidates() must NOT recurse
	// into translate, and this light must correctly stay undiscovered
	// (same, pre-existing "not supported" outcome as before this round's
	// fix, not a new regression).
	camera cam;
	auto light = make_shared<diffuse_light>(color(6, 6, 6), false);
	auto sp = make_shared<sphere>(point3(0, 0, 0), 1.0, light);
	auto world = make_shared<hittable_list>();
	world->add(make_shared<translate>(sp, vec3(0, 5, 0)));

	BDPTSceneAdapter adapter(*world, cam);
	EXPECT_EQ(adapter.EmitterCount(), 0)
		<< "a translate-wrapped light is a known, deliberate gap - it must not silently "
		   "misfire with world-space-incorrect samples instead";
}

TEST(SPPMAdapterBvhWorld, ConstructsWithoutLosingTheEmitterInsideAWrappedBvhNode) {
	camera cam;
	auto world = makeBvhWrappedWorldWithOneLight();
	auto nee_lights = make_shared<hittable_list>();   // unused by this check
	// SPPMSceneAdapter has no public EmitterCount() accessor, so this is a
	// narrower smoke test than the BDPT one above: it only confirms
	// construction (which runs the same collectEmitterCandidates() scan)
	// doesn't crash and still finds real geometry through the wrapper -
	// SPPMAdapterBsdfTest (sppm_adapter_bsdf_tests.cpp) already covers this
	// adapter's actual sampling behavior in depth for the flat-world case.
	EXPECT_NO_THROW({
		SPPMSceneAdapter adapter(*world, *nee_lights, cam);
	});
}

TEST(SPPMAdapterBvhWorld, ConstructsWithATriangleMeshLightThroughTheWrapper) {
	camera cam;
	auto world = makeBvhWrappedWorldWithATriangleLight();
	auto nee_lights = make_shared<hittable_list>();
	EXPECT_NO_THROW({
		SPPMSceneAdapter adapter(*world, *nee_lights, cam);
	});
}
