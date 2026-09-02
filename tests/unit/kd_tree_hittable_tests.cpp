/**
 * @file kd_tree_hittable_tests.cpp
 * @brief Regression tests for src/TheRestOfYourLife/kd_tree_hittable.h
 *
 * Code review of the "Accelerator \"kdtree\"" wiring round found a real
 * hazard specific to kd_tree_hittable.h, not shared with its BVH sibling
 * bvh_aggregate_hittable.h: BvhTree (src/shared/bvh_aggregate.h) guarantees
 * exactly one leaf per primitive, so a plain "cache the first hit() result"
 * scheme is airtight there. KdTree (src/shared/kd_tree.h) does NOT have that
 * guarantee - build_tree()'s own SAH split classification can put a
 * primitive whose bounding box straddles the chosen split plane into BOTH
 * children, so one primitive can be tested from TWO different leaves within
 * a single traversal. For a STOCHASTIC hittable (constant_medium et al.,
 * which draws a fresh random free-path sample via random_double() inside
 * hit() itself), a naive re-query on the second leaf visit is exactly the
 * bug the whole cache-instead-of-requery pattern exists to prevent, just
 * moved one level down - see kd_tree_hittable.h's own top comment for the
 * fix (cache both hit AND miss outcomes, range-checked against the current,
 * possibly-narrower interval on reuse).
 *
 * These tests build the accelerated hittable_list DIRECTLY in C++ (not
 * through the pbrt scene-text loader, unlike pbrt_cpu_builder_tests.cpp's
 * own kd-tree medium tests) specifically to keep a competing, real-material
 * boundary shape OUT of the list: pbrt_cpu_builder.h's own loader always
 * adds a medium's boundary shape to the world as a SEPARATE, real hittable
 * (the "surface + fog inside" design real pbrt scenes use, e.g. a glass
 * shell around smoke) - since that boundary is always geometrically closer
 * than any scatter event drawn from inside the volume, it always wins the
 * "closest hit" race over the medium's own constant_medium hittable,
 * regardless of whether the cache logic under test is correct, making the
 * medium's own behavior unobservable through the loader's output. Testing
 * kd_tree_hittable directly, with the medium as the ONLY thing a ray can
 * hit, is the only way to actually isolate this.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "kd_tree_hittable.h"
#include "sphere.h"
#include "material.h"
#include "constant_medium.h"

#include <cmath>

namespace {

// A medium sphere (radius 3, centered at the origin) plus 6 tiny opaque
// decoy spheres at +/-2 along every axis - well inside the medium sphere's
// own bbox, so their own bbox edges force the SAH build (default
// max_prims_leaf=1, 7 primitives present) to carve splits through that
// region, genuinely duplicating the wide medium sphere into multiple
// leaves. Decoys are tiny (radius 0.05) purely so their bbox EDGES (a split
// candidate is a bbox min/max, independent of how small the primitive
// itself is) still force the same split behavior, while making it very
// unlikely (~0.1% of random directions) that a through-center diameter ray
// actually grazes one - keeping the "miss" bucket below a clean read on the
// medium's own scattering statistics, not contaminated by decoy-occlusion
// events.
shared_ptr<kd_tree_hittable> makeStraddlingMediumTree(double sigma_s) {
	auto flat = make_shared<hittable_list>();
	auto boundary = make_shared<sphere>(point3(0, 0, 0), 3.0, make_shared<lambertian>(color(1, 1, 1)));
	flat->add(make_shared<constant_medium>(boundary, /*sigma_a=*/0.0, sigma_s,
											color(1, 1, 1), /*g=*/0.0));
	auto decoyMat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	const double offsets[6][3] = {
		{-2, 0, 0}, {2, 0, 0}, {0, -2, 0}, {0, 2, 0}, {0, 0, -2}, {0, 0, 2},
	};
	for (const auto &o : offsets)
		flat->add(make_shared<sphere>(point3(o[0], o[1], o[2]), 0.05, decoyMat));
	return make_shared<kd_tree_hittable>(*flat, KdTreeAccelParams{});
}

} // namespace

TEST(KdTreeHittable, MediumWithNoCompetingBoundaryIsReachable) {
	// Sanity check for the test methodology itself, before the tree-forcing
	// complexity: with sigma_s high enough that scattering is all but
	// certain, and no decoys/competing boundary, a diameter ray through the
	// center must actually report an hg_phase_material hit - confirming
	// dynamic_cast<hg_phase_material*> is really the right signal to look
	// for (not, say, the medium's boundary material, or a builder default
	// this test would otherwise silently never observe).
	auto flat = make_shared<hittable_list>();
	auto boundary = make_shared<sphere>(point3(0, 0, 0), 3.0, make_shared<lambertian>(color(1, 1, 1)));
	flat->add(make_shared<constant_medium>(boundary, /*sigma_a=*/0.0, /*sigma_s=*/500.0,
											color(1, 1, 1), /*g=*/0.0));
	auto tree = make_shared<kd_tree_hittable>(*flat, KdTreeAccelParams{});

	hit_record rec;
	ASSERT_TRUE(tree->hit(ray(point3(0, 0, -10), vec3(0, 0, 1)), interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<hg_phase_material *>(rec.mat.get()), nullptr)
		<< "a diameter ray through an extreme-sigma_s medium, with nothing else in the "
		   "list to compete for closest-hit, must report the medium's own scatter event";
}

TEST(KdTreeHittable, MediumStraddlingASplitPlaneStillScattersAtTheRightRate) {
	// The real regression case: sigma_s chosen so a single, correctly-drawn
	// free-path sample has a substantial, statistically distinguishable
	// miss probability (tau = sigma_s * chord = 0.15 * 6 = 0.9,
	// exp(-0.9) ~= 40.66%) - an extreme sigma_s (near-certain scattering
	// either way) can't actually DETECT a re-query bug, since almost every
	// trial hits regardless of whether the fix is present.
	//
	// Rays are many randomly-directed diameters through the sphere's exact
	// center (origin = -20*dir, direction = dir) - every such ray has the
	// SAME chord length (a diameter), so one theoretical hit rate applies
	// to the whole trial set, and a random spread of directions means a
	// good fraction of them have a nonzero component along whatever split
	// axis/axes the tree actually used, genuinely exercising the double-
	// leaf-visit path (a single fixed-direction ray could easily approach
	// along a slice of space no chosen split plane ever divides, and
	// silently prove nothing - this is what the first, now-removed version
	// of this test got wrong).
	//
	// If KdHittablePrim::intersect() re-queried obj->hit() a second,
	// independent time instead of reusing its per-traversal cached outcome,
	// a ray whose first draw missed would get a second, independent chance
	// to hit on its second leaf visit, inflating the empirical hit rate
	// (deflating the miss rate) measurably below the true single-sample
	// rate.
	auto tree = makeStraddlingMediumTree(/*sigma_s=*/0.15);

	// Unlike the pbrt-loaded scenario (which always has a real, opaque
	// boundary shape backstopping every ray), this isolated scene has
	// NOTHING else for a ray to hit when the medium genuinely doesn't
	// scatter - constant_medium::hit() itself returns false in that case
	// (fully transparent, not "hit the boundary with no effect"), so a
	// tree->hit() returning false IS the expected, common miss outcome, not
	// a test-methodology error.
	const int kTrials = 4000;
	const double kExpectedMissFraction = std::exp(-0.9);
	int misses = 0;
	for (int i = 0; i < kTrials; ++i) {
		const vec3 dir = random_unit_vector();
		const point3 origin = point3(0, 0, 0) - 20.0 * dir;
		hit_record rec;
		if (!tree->hit(ray(origin, dir), interval(0.001, infinity), rec)) {
			++misses;  // no scatter, no decoy hit either - genuinely nothing
			continue;
		}
		if (dynamic_cast<hg_phase_material *>(rec.mat.get()) == nullptr)
			++misses;  // hit a decoy sphere instead (rare, ~0.1% - still not a real scatter)
	}
	const double observedMissFraction = static_cast<double>(misses) / kTrials;
	EXPECT_NEAR(observedMissFraction, kExpectedMissFraction, 0.05)
		<< "observed " << misses << "/" << kTrials << " misses (" << observedMissFraction
		<< ") vs theoretical single-sample exp(-tau)=" << kExpectedMissFraction
		<< " - a fraction pulled well below the theoretical rate means some rays got a "
		   "second, independent scattering roll from a second kd-tree leaf visit of the "
		   "same medium-wrapped primitive";
}
