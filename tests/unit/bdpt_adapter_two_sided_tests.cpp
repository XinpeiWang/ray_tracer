/**
 * @file bdpt_adapter_two_sided_tests.cpp
 * @brief Unit tests for BDPTSceneAdapter's is_two_sided() handling
 *
 * BDPTSceneAdapter::SampleLight/SampleLightEmission/SampleLightLe read a
 * diffuse_light's emitted texture directly, bypassing diffuse_light::
 * emitted()'s own front-face/two-sided gate -- these tests pin the adapter's
 * own replacement gate: a one-sided light must never be sampled from (or
 * emit toward) its back side, and a two-sided light must be reachable from
 * both.
 *
 * All lights below are a unit quad centered at the origin in the XZ plane
 * with outward normal +y (front/emitting side faces up), so "front" means
 * y>0 and "back" means y<0 for every reference point used here.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "quad.h"
#include "material.h"
#include "bdpt_adapter.h"

namespace {

shared_ptr<hittable_list> makeWorld(bool two_sided) {
	auto world = make_shared<hittable_list>();
	auto light = make_shared<diffuse_light>(color(4, 4, 4), two_sided);
	world->add(make_shared<quad>(point3(-1, 0, -1), vec3(0, 0, 2), vec3(2, 0, 0), light));
	return world;
}

} // namespace

TEST(BDPTAdapterTwoSided, OneSidedLightIlluminatesItsFront) {
	camera cam;
	auto world = makeWorld(/*two_sided=*/false);
	BDPTSceneAdapter adapter(*world, cam);
	ASSERT_EQ(adapter.EmitterCount(), 1);

	const double ref_p[3] = {0.0, 5.0, 0.0};   // above the quad -- front side
	BDPTLightSample<double> ls;
	bool sawSample = false;
	for (int i = 0; i < 20; ++i) {
		if (adapter.SampleLight(random_double(), ref_p, ls)) {
			sawSample = true;
			EXPECT_GT(ls.pdf, 0.0);
			EXPECT_GT(ls.L[0], 0.0);
		}
	}
	EXPECT_TRUE(sawSample) << "a one-sided light must illuminate a point in front of it";
}

TEST(BDPTAdapterTwoSided, OneSidedLightNeverIlluminatesItsBack) {
	camera cam;
	auto world = makeWorld(/*two_sided=*/false);
	BDPTSceneAdapter adapter(*world, cam);

	const double ref_p[3] = {0.0, -5.0, 0.0};   // below the quad -- back side
	BDPTLightSample<double> ls;
	for (int i = 0; i < 20; ++i)
		EXPECT_FALSE(adapter.SampleLight(random_double(), ref_p, ls))
			<< "a one-sided light's NEE sample must be rejected from behind it, "
			   "matching diffuse_light::emitted()'s own front-face gate";
}

TEST(BDPTAdapterTwoSided, TwoSidedLightIlluminatesBothFrontAndBack) {
	camera cam;
	auto world = makeWorld(/*two_sided=*/true);
	BDPTSceneAdapter adapter(*world, cam);

	const double front[3] = {0.0, 5.0, 0.0};
	const double back[3]  = {0.0, -5.0, 0.0};
	BDPTLightSample<double> ls;

	bool sawFront = false, sawBack = false;
	for (int i = 0; i < 20; ++i) {
		if (adapter.SampleLight(random_double(), front, ls)) sawFront = true;
		if (adapter.SampleLight(random_double(), back, ls)) sawBack = true;
	}
	EXPECT_TRUE(sawFront);
	EXPECT_TRUE(sawBack) << "a two-sided light must also illuminate a point behind it";
}

TEST(BDPTAdapterTwoSided, OneSidedLightEmissionNeverLeavesTheFrontHemisphere) {
	camera cam;
	auto world = makeWorld(/*two_sided=*/false);
	BDPTSceneAdapter adapter(*world, cam);

	LightEmissionSample<double> les;
	int accepted = 0;
	for (int i = 0; i < 200; ++i) {
		if (!adapter.SampleLightEmission(random_double(), random_double(), random_double(),
										  random_double(), random_double(), les))
			continue;
		++accepted;
		const vec3 dir(les.ray_d[0], les.ray_d[1], les.ray_d[2]);
		const vec3 n(les.surface_hit.geo_n[0], les.surface_hit.geo_n[1], les.surface_hit.geo_n[2]);
		EXPECT_GT(dot(dir, n), 0.0)
			<< "a one-sided light must never emit into its back hemisphere";
	}
	EXPECT_GT(accepted, 0);
}

TEST(BDPTAdapterTwoSided, TwoSidedLightEmissionUsesBothHemispheres) {
	camera cam;
	auto world = makeWorld(/*two_sided=*/true);
	BDPTSceneAdapter adapter(*world, cam);

	LightEmissionSample<double> les;
	bool sawFrontHemisphere = false, sawBackHemisphere = false;
	for (int i = 0; i < 200; ++i) {
		if (!adapter.SampleLightEmission(random_double(), random_double(), random_double(),
										  random_double(), random_double(), les))
			continue;
		const vec3 dir(les.ray_d[0], les.ray_d[1], les.ray_d[2]);
		const vec3 n(les.surface_hit.geo_n[0], les.surface_hit.geo_n[1], les.surface_hit.geo_n[2]);
		const double c = dot(dir, n);
		EXPECT_GT(les.pdf_dir, 0.0);
		if (c > 0.0) sawFrontHemisphere = true;
		if (c < 0.0) sawBackHemisphere = true;
	}
	EXPECT_TRUE(sawFrontHemisphere);
	EXPECT_TRUE(sawBackHemisphere)
		<< "a two-sided light must emit photons from its back face too, over enough draws";
}
