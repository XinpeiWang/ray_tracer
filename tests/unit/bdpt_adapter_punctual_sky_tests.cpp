/**
 * @file bdpt_adapter_punctual_sky_tests.cpp
 * @brief Unit tests for BDPTSceneAdapter's point/spot/distant/sky light
 * support (SampleLight/SampleLightLe/LightPMF/LightPDFLe unified across
 * area + punctual + sky lights - see bdpt_adapter.h's own "Scope" comment).
 *
 * Before this, BDPTSceneAdapter only ever sampled area (diffuse_light)
 * lights - a scene lit solely by a point/spot/distant/sky light rendered
 * entirely black under --bdpt/--mlt/--simplepath even though the scene had
 * a real light. These tests pin the basic contract for each newly-added
 * kind directly against the adapter, independent of any full scene render.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "bdpt_adapter.h"

namespace {

// A single diffuse sphere so world_.bounding_box() is non-degenerate and
// Unoccluded()/Intersect() have real geometry to trace shadow rays against.
// Placed well clear of every light's own position/direction below.
shared_ptr<hittable_list> makeGeometryOnlyWorld() {
	auto world = make_shared<hittable_list>();
	world->add(make_shared<sphere>(point3(0, 0, 0), 1.0,
		make_shared<lambertian>(color(0.5, 0.5, 0.5))));
	return world;
}

const double kRefP[3] = {0.0, 5.0, 0.0};

} // namespace

TEST(BDPTAdapterPunctualSky, PointLightIsSampledAsDelta) {
	camera cam;
	cam.punct_lights = make_shared<punctual_light_list>();
	cam.punct_lights->add_point(point3(10, 10, 10), color(50, 50, 50));
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);

	EXPECT_EQ(adapter.EmitterCount(), 1);
	EXPECT_EQ(adapter.AreaEmitterCount(), 0)
		<< "a point light is not an area emitter - LightPath's own "
		   "SampleLightEmission() must still see zero";

	BDPTLightSample<double> ls{};
	ASSERT_TRUE(adapter.SampleLight(random_double(), kRefP, ls));
	EXPECT_TRUE(ls.is_delta);
	EXPECT_FALSE(ls.is_infinite);
	EXPECT_GT(ls.pdf, 0.0);
	EXPECT_GT(ls.L[0] + ls.L[1] + ls.L[2], 0.0);

	// light_id must resolve back through LightPMF/LightPDFLe without
	// colliding with real bsdf_id space (see resolve_emitter_index's own
	// comment on why toLightId() offsets by kCtxPoolCapacity).
	EXPECT_GT(adapter.LightPMF(ls.light_id), 0.0);
	double pdfPos = -1.0, pdfDir = -1.0;
	adapter.LightPDFLe(ls.light_id, nullptr, nullptr, ls.wi, pdfPos, pdfDir);
	EXPECT_DOUBLE_EQ(pdfPos, 1.0) << "delta position: within-light density is 1";
	EXPECT_GT(pdfDir, 0.0);
}

TEST(BDPTAdapterPunctualSky, PointLightEmissionIsIsotropic) {
	camera cam;
	cam.punct_lights = make_shared<punctual_light_list>();
	cam.punct_lights->add_point(point3(10, 10, 10), color(50, 50, 50));
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);

	BDPTLightLeSample<double> les{};
	ASSERT_TRUE(adapter.SampleLightLe(0.0, nullptr, nullptr, les));
	EXPECT_TRUE(les.is_delta_dir);
	EXPECT_FALSE(les.is_infinite);
	EXPECT_TRUE(les.is_on_surface);
	EXPECT_DOUBLE_EQ(les.ray_o[0], 10.0);
	EXPECT_DOUBLE_EQ(les.ray_o[1], 10.0);
	EXPECT_DOUBLE_EQ(les.ray_o[2], 10.0);
	EXPECT_NEAR(les.pdf_dir, 1.0 / (4.0 * pi), 1e-9);
	EXPECT_GT(les.pdf_pos, 0.0);
}

TEST(BDPTAdapterPunctualSky, SpotLightRespectsItsCone) {
	camera cam;
	cam.punct_lights = make_shared<punctual_light_list>();
	// Narrow cone pointing straight down at the origin from above.
	cam.punct_lights->add_spot(point3(0, 10, 0), vec3(0, -1, 0), color(100, 100, 100),
	                            /*total_width_deg=*/10.0, /*falloff_start_deg=*/5.0);
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);
	ASSERT_EQ(adapter.EmitterCount(), 1);

	// A point directly under the spot, inside its cone.
	const double insideCone[3] = {0.0, 0.0, 0.0};
	BDPTLightSample<double> ls{};
	bool sawInside = false;
	for (int i = 0; i < 20; ++i)
		if (adapter.SampleLight(random_double(), insideCone, ls)) sawInside = true;
	EXPECT_TRUE(sawInside) << "a point inside the spot's cone must be reachable";

	// A point far to the side, well outside a 10-degree cone.
	const double outsideCone[3] = {100.0, 0.0, 0.0};
	for (int i = 0; i < 20; ++i)
		EXPECT_FALSE(adapter.SampleLight(random_double(), outsideCone, ls))
			<< "a point outside the spot's cone must never be illuminated";

	// Emission direction sampling must stay within the cone too.
	BDPTLightLeSample<double> les{};
	ASSERT_TRUE(adapter.SampleLightLe(0.0, nullptr, nullptr, les));
	const vec3 dir(les.ray_d[0], les.ray_d[1], les.ray_d[2]);
	EXPECT_GT(dot(dir, vec3(0, -1, 0)), std::cos(degrees_to_radians(10.0)) - 1e-6);
}

TEST(BDPTAdapterPunctualSky, DistantLightIsSampledAsDelta) {
	camera cam;
	cam.punct_lights = make_shared<punctual_light_list>();
	cam.punct_lights->add_distant(vec3(0, -1, 0), color(2, 2, 2));
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);
	ASSERT_EQ(adapter.EmitterCount(), 1);

	BDPTLightSample<double> ls{};
	ASSERT_TRUE(adapter.SampleLight(random_double(), kRefP, ls));
	EXPECT_TRUE(ls.is_delta);
	EXPECT_FALSE(ls.is_infinite);
	// distant_light_obj::sample_direct()'s wi is the constructor's own
	// `dir` argument unchanged (DistantLightData::sample_wi() returns
	// dir_x/y/z as-is - see punctual_lights.h) - no position dependence.
	EXPECT_NEAR(ls.wi[0], 0.0, 1e-9);
	EXPECT_NEAR(ls.wi[1], -1.0, 1e-9);
	EXPECT_NEAR(ls.wi[2], 0.0, 1e-9);

	BDPTLightLeSample<double> les{};
	ASSERT_TRUE(adapter.SampleLightLe(0.0, nullptr, nullptr, les));
	EXPECT_TRUE(les.is_delta_dir);
	EXPECT_FALSE(les.is_infinite);
	EXPECT_DOUBLE_EQ(les.pdf_dir, 1.0);
	EXPECT_GT(les.pdf_pos, 0.0);
	// The emitted photon travels the OPPOSITE way from "toward the light"
	// (SampleLightLe's own rayDir = -wiToLight - see its own comment).
	EXPECT_NEAR(les.ray_d[1], 1.0, 1e-9);
}

TEST(BDPTAdapterPunctualSky, SkyLightIsSampledAsRealDistributionNotDelta) {
	camera cam;
	cam.sky = std::make_shared<sky_light>(color(0.5, 0.6, 0.9));
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);
	ASSERT_EQ(adapter.EmitterCount(), 1);
	EXPECT_EQ(adapter.AreaEmitterCount(), 0);

	BDPTLightSample<double> ls{};
	ASSERT_TRUE(adapter.SampleLight(random_double(), kRefP, ls));
	EXPECT_FALSE(ls.is_delta) << "the sky is a real, continuous distribution - not delta";
	EXPECT_TRUE(ls.is_infinite);
	EXPECT_GT(ls.pdf, 0.0);
	EXPECT_GT(ls.L[0] + ls.L[1] + ls.L[2], 0.0);

	double pdfPos = -1.0, pdfDir = -1.0;
	adapter.LightPDFLe(ls.light_id, nullptr, nullptr, ls.wi, pdfPos, pdfDir);
	EXPECT_GT(pdfPos, 0.0);
	EXPECT_GT(pdfDir, 0.0);

	EXPECT_GT(adapter.InfiniteLightDensity(ls.wi), 0.0);
}

TEST(BDPTAdapterPunctualSky, SkyLightEmissionRoutesToInfiniteVertex) {
	camera cam;
	cam.sky = std::make_shared<sky_light>(color(0.5, 0.6, 0.9));
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);

	BDPTLightLeSample<double> les{};
	ASSERT_TRUE(adapter.SampleLightLe(0.0, nullptr, nullptr, les));
	EXPECT_TRUE(les.is_infinite);
	EXPECT_FALSE(les.is_on_surface)
		<< "must dispatch to MakeLightInfinite in GenerateLightSubpath, not MakeLightSurface";
	EXPECT_FALSE(les.is_delta_dir);
	EXPECT_GT(les.pdf_pos, 0.0);
	EXPECT_GT(les.pdf_dir, 0.0);
	// The emission ray must actually pass near the scene (not launched from
	// some degenerate/zero origin) - the sphere sits at the origin with
	// radius 1, so a ray aimed roughly at it from just past the bounding
	// sphere should be able to intersect it. Weaker, robust check: the
	// origin is far from the origin (on the bounding-sphere shell).
	const double originDist = std::sqrt(les.ray_o[0]*les.ray_o[0] +
	                                     les.ray_o[1]*les.ray_o[1] +
	                                     les.ray_o[2]*les.ray_o[2]);
	EXPECT_GT(originDist, 0.5);
}

TEST(BDPTAdapterPunctualSky, AreaAndPunctualLightsCoexistInOneDistribution) {
	camera cam;
	cam.punct_lights = make_shared<punctual_light_list>();
	cam.punct_lights->add_point(point3(10, 10, 10), color(50, 50, 50));
	auto world = makeGeometryOnlyWorld();
	auto light = make_shared<diffuse_light>(color(4, 4, 4));
	world->add(make_shared<sphere>(point3(0, 20, 0), 1.0, light));
	BDPTSceneAdapter adapter(*world, cam);

	EXPECT_EQ(adapter.EmitterCount(), 2);
	EXPECT_EQ(adapter.AreaEmitterCount(), 1);

	// Over enough draws, SampleLight must return both an area-light sample
	// (is_delta=false, is_infinite=false) and a point-light sample
	// (is_delta=true).
	bool sawArea = false, sawPoint = false;
	BDPTLightSample<double> ls{};
	for (int i = 0; i < 200; ++i) {
		if (!adapter.SampleLight(random_double(), kRefP, ls)) continue;
		if (ls.is_delta) sawPoint = true;
		else sawArea = true;
	}
	EXPECT_TRUE(sawArea);
	EXPECT_TRUE(sawPoint);
}

TEST(BDPTAdapterPunctualSky, NoLightsMeansNoSamples) {
	camera cam;
	auto world = makeGeometryOnlyWorld();
	BDPTSceneAdapter adapter(*world, cam);

	EXPECT_EQ(adapter.EmitterCount(), 0);
	EXPECT_EQ(adapter.AreaEmitterCount(), 0);

	BDPTLightSample<double> ls{};
	EXPECT_FALSE(adapter.SampleLight(random_double(), kRefP, ls));
	BDPTLightLeSample<double> les{};
	EXPECT_FALSE(adapter.SampleLightLe(0.0, nullptr, nullptr, les));
}
