/**
 * @file bdpt_adapter_medium_emission_tests.cpp
 * @brief Regression test for the MakeNamedMedium "rgb Le" / BDPT interaction
 *
 * A code-review pass on the emissive-medium feature found that letting a
 * medium-scatter hit's real hg_phase_material::emitted() reach BDPT's own
 * vertex classification unfiltered produces two real bugs: BDPT's front-face
 * Le() gate zeroes the contribution for half of all exit directions (since
 * constant_medium::hit() has no real geometric normal), and BDPT's MIS
 * weight misapplies its delta-distribution remap0() fallback to a
 * legitimately-zero (not delta) light-origin pdf, since the medium is never
 * a registered light - inflating the MIS denominator and dimming the s=0
 * strategy. The fix: BDPTSceneAdapter::Intersect() explicitly suppresses
 * emission for a medium-scatter hit (material::is_medium_scatter()) before
 * it ever reaches BDPT's vertex machinery, restoring BDPT's exact
 * pre-feature behavior for media. This test pins that suppression.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include "constant_medium.h"
#include "bdpt_adapter.h"

namespace {

// A sphere wrapped in a strongly-absorptive/scattering emissive medium -
// same "astronomically large sigma_t so a collision is effectively certain"
// trick as PbrtCpuBuildTest.EmissiveMediumBakesLeWeightedBySigmaAOverSigmaT
// (tests/unit/pbrt_cpu_builder_tests.cpp), sidestepping constant_medium's
// own stochastic collision test.
shared_ptr<hittable_list> makeEmissiveMediumWorld() {
	auto world = make_shared<hittable_list>();
	auto boundary = make_shared<sphere>(point3(0, 0, 0), 1.0, nullptr);
	auto medium = make_shared<constant_medium>(
		boundary, /*sigma_a=*/800.0, /*sigma_s=*/200.0,
		/*albedo=*/color(1, 1, 1), /*g=*/0.0, /*Le=*/color(5, 2, 0.5));
	world->add(medium);
	return world;
}

} // namespace

TEST(BDPTAdapterMediumEmission, MediumScatterHitReportsZeroAreaLe) {
	camera cam;
	auto world = makeEmissiveMediumWorld();
	BDPTSceneAdapter adapter(*world, cam);

	const double org[3] = {0.0, 0.0, -5.0};
	const double dir[3] = {0.0, 0.0, 1.0};
	BDPTHit<double> hit;
	ASSERT_TRUE(adapter.Intersect(org, dir, 1e30, hit))
		<< "a ray toward the emissive-medium-wrapped sphere should hit it";

	EXPECT_EQ(hit.area_Le[0], 0.0);
	EXPECT_EQ(hit.area_Le[1], 0.0);
	EXPECT_EQ(hit.area_Le[2], 0.0)
		<< "BDPTSceneAdapter::Intersect() must suppress a medium-scatter "
		   "hit's real emission, not feed it to BDPT's own surface-light "
		   "vertex machinery";
}
