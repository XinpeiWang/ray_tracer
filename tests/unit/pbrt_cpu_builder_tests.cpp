/**
 * @file pbrt_cpu_builder_tests.cpp
 * @brief Unit tests for building CPU hittables from a flattened pbrt scene
 *
 * These fire actual rays at the built world rather than inspecting the object
 * graph. Counting primitives proves objects were created; hitting them proves
 * they were created in the right place, which is what every earlier stage in
 * the chain exists to get right.
 */

#include <gtest/gtest.h>

#include "pbrt_cpu_builder.h"
#include "pbrt_flatten.h"
#include "pbrt_scene.h"

#include <string>

namespace {

pbrt_cpu::BuildResult buildFrom(const std::string &text) {
	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text);
	EXPECT_TRUE(parsed.ok) << parsed.error;
	return pbrt_cpu::build(pbrt_flatten::flatten(parsed.scene));
}

// Casts a ray and reports whether it hit, and how far along.
bool castRay(const pbrt_cpu::BuildResult &b, const point3 &origin,
			 const vec3 &direction, double &tOut) {
	hit_record rec;
	const ray r(origin, direction);
	if (!b.world->hit(r, interval(0.001, infinity), rec)) return false;
	tOut = rec.t;
	return true;
}

// A unit quad in the z=0 plane spanning (0,0)..(1,1).
const char *kQuad =
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
	"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n";

} // namespace

TEST(PbrtCpuBuildTest, GeometryIsWhereTheSceneSaysItIs) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.triangleCount, 2u);

	double t = 0.0;
	ASSERT_TRUE(castRay(b, point3(0.5, 0.5, -5), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 5.0, 1e-6);

	EXPECT_FALSE(castRay(b, point3(9, 9, -5), vec3(0, 0, 1), t))
		<< "a ray well outside the quad must miss";
}

TEST(PbrtCpuBuildTest, TranslationInTheSceneMovesTheGeometry) {
	const pbrt_cpu::BuildResult b = buildFrom(std::string("Translate 0 0 10\n") + kQuad);
	double t = 0.0;
	ASSERT_TRUE(castRay(b, point3(0.5, 0.5, -5), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 15.0, 1e-6) << "the quad moved 10 further away";
}

TEST(PbrtCpuBuildTest, SphereIsBuiltAtItsTransformedCentreAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 10\n"
		"Scale 2 2 2\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	// Centre at z=10, radius 2, so the near surface sits at z=8.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 8.0, 1e-6);
}

TEST(PbrtCpuBuildTest, SharedVerticesAreDeduplicated) {
	// The two triangles of a quad share two corners. FlatScene stores all six
	// vertices explicitly; the builder should recover the original four.
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.uniqueVertexCount, 4u);
}

TEST(PbrtCpuBuildTest, DistinctVerticesAreNotCollapsed) {
	// Guards the dedup against being too eager: two separate triangles sharing
	// no corners must keep all six vertices.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 5 5 5  6 5 5  5 6 5 ]\n");
	EXPECT_EQ(b.triangleCount, 2u);
	EXPECT_EQ(b.uniqueVertexCount, 6u);
}

TEST(PbrtCpuBuildTest, EmissiveShapesGoIntoTheLightList) {
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuad + "AttributeEnd\n" + kQuad);
	EXPECT_EQ(b.triangleCount, 4u);
	EXPECT_EQ(b.lights->objects.size(), 2u)
		<< "only the triangles inside the attribute scope are emissive";
}

TEST(PbrtCpuBuildTest, NonEmissiveSceneHasAnEmptyLightList) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_TRUE(b.lights->objects.empty());
}

TEST(PbrtCpuBuildTest, AnEmptySceneBuildsWithoutCrashing) {
	// flatten() drops unsupported shapes, so a scene can legitimately arrive
	// with nothing in it. A BVH must not be built over zero primitives.
	const pbrt_cpu::BuildResult b = buildFrom("Shape \"cylinder\"\n");
	EXPECT_EQ(b.triangleCount, 0u);
	EXPECT_EQ(b.sphereCount, 0u);
	ASSERT_NE(b.world, nullptr);
	EXPECT_TRUE(b.world->objects.empty());
}

TEST(PbrtCpuBuildTest, MaterialsAreSharedRatherThanCopiedPerPrimitive) {
	// A million-triangle mesh with one material should hold one material
	// object, not a million.
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n") + kQuad);
	ASSERT_EQ(b.triangleCount, 2u);

	// The world is BVH-wrapped, so reach the triangles by hitting them rather
	// than walking the object graph.
	hit_record a, c;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), a));
	ASSERT_TRUE(b.world->hit(ray(point3(0.75, 0.75, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), c));
	EXPECT_EQ(a.mat.get(), c.mat.get());
}

// ---------------------------------------------------------------------------
// makeMaterial() used to fall through to lambertian for these three kinds
// silently - no warning, unlike the genuinely-unsupported case - even though
// coated_diffuse, coated_conductor and diffuse_transmission all already
// existed in material_pbrt.h. That made CPU and GPU render the same pbrt
// file with different materials for any scene using one of them. These pin
// the real class getting built, not just that something renders.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, CoatedDiffuseBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"coateddiffuse\" \"rgb reflectance\" [ .2 .4 .6 ] "
		"\"float roughness\" [ .1 ]\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<coated_diffuse *>(rec.mat.get()), nullptr)
		<< "a pbrt coateddiffuse material must build the real coated_diffuse "
		   "class, not silently fall back to lambertian";
}

TEST(PbrtCpuBuildTest, CoatedConductorBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"coatedconductor\" \"rgb reflectance\" [ .8 .8 .8 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<coated_conductor *>(rec.mat.get()), nullptr)
		<< "a pbrt coatedconductor material must build the real "
		   "coated_conductor class, not silently fall back to lambertian";
}

TEST(PbrtCpuBuildTest, DiffuseTransmissionBuildsTheRealMaterialNotLambertian) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"diffusetransmission\"\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<diffuse_transmission *>(rec.mat.get()), nullptr)
		<< "a pbrt diffusetransmission material must build the real "
		   "diffuse_transmission class, not silently fall back to lambertian";
}

// ---------------------------------------------------------------------------
// LightSource point/spot/distant/goniometric/projection
//
// pbrt_flatten_tests.cpp already pins the parsing (parameter defaults, CTM
// handling); these pin the next stage - that a parsed pbrt_flatten::
// PunctualLight actually becomes a real punctual_light_objects.h light a
// shading point can sample, wired the same way build_point_light_punct() et
// al. wire the hand-built C2-C6 showcase scenes (scenes_advanced.h).
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, NoLightSourceLeavesPunctLightsNull) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_EQ(b.punctLights, nullptr);
}

TEST(PbrtCpuBuildTest, PointLightSourceIsSampleableAtAShadingPoint) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"point\" \"point3 from\" [ 0 10 0 ] \"rgb I\" [ 1 1 1 ] "
		"\"float scale\" [ 100 ]\n" + std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);
	ASSERT_FALSE(b.punctLights->empty());

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		// wi must point from the shading point TOWARD the light: straight up.
		EXPECT_NEAR(s.wi.x(), 0.0, 1e-9);
		EXPECT_NEAR(s.wi.y(), 1.0, 1e-9);
		EXPECT_NEAR(s.wi.z(), 0.0, 1e-9);
		// Li = I * scale / r^2 = 1 * 100 / 100 = 1.
		EXPECT_NEAR(s.Li.x(), 1.0, 1e-6);
		EXPECT_NEAR(s.t_max, 10.0, 1e-6);
	});
	EXPECT_EQ(samples, 1);
}

TEST(PbrtCpuBuildTest, SpotLightSourceAimsFromFromTowardTo) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"spot\" \"point3 from\" [ 0 0 -10 ] \"point3 to\" [ 0 0 0 ] "
		"\"rgb I\" [ 1 1 1 ] \"float scale\" [ 100 ] \"float coneangle\" [ 60 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	// A point directly in front of the spot (along its axis) sees full
	// intensity; the shadow-ray direction must point back toward the light.
	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_NEAR(s.wi.z(), -1.0, 1e-9);
		EXPECT_GT(s.Li.x(), 0.0) << "on-axis point should be lit, not in the falloff";
	});
	EXPECT_EQ(samples, 1);
}

TEST(PbrtCpuBuildTest, DistantLightSourceHasNoDistanceFalloff) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"LightSource \"distant\" \"point3 from\" [ 0 1 0 ] \"point3 to\" [ 0 0 0 ] "
		"\"rgb L\" [ 2 3 4 ] \"float scale\" [ 1 ]\n" + std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	// Radiance must be identical at two very different distances - that is
	// the entire point of a directional/parallel light.
	color nearL, farL;
	b.punctLights->for_each_sample(point3(0, 0, 0),
		[&](const PunctualLiSample &s) { nearL = s.Li; });
	b.punctLights->for_each_sample(point3(0, 1000, 0),
		[&](const PunctualLiSample &s) { farL = s.Li; });
	EXPECT_NEAR(nearL.x(), 2.0, 1e-6);
	EXPECT_NEAR(nearL.x(), farL.x(), 1e-9);
	EXPECT_NEAR(nearL.y(), farL.y(), 1e-9);
	EXPECT_NEAR(nearL.z(), farL.z(), 1e-9);
}

TEST(PbrtCpuBuildTest, GoniometricLightSourceIsSampleableAtAShadingPoint) {
	// pbrt-v4's GoniometricLight has no "from"/"to" of its own (unlike point/
	// spot/distant) - its position/orientation come purely from the CTM, so
	// this positions it with Translate rather than a "from" parameter (see
	// PunctualLight::pos's own comment). No "filename" - flatten() falls
	// back to an isotropic profile (see FlattenPunctualLightTest::
	// GoniometricLightWithNoFilenameIsIsotropicAndUnwarned in
	// pbrt_flatten_tests.cpp), so this should behave exactly like a plain
	// point light: nonzero, direction-independent intensity.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 10 0\n"
		"LightSource \"goniometric\" \"rgb I\" [ 1 1 1 ] \"float scale\" [ 100 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_GT(s.Li.x(), 0.0);
	});
	EXPECT_EQ(samples, 1);
}

TEST(PbrtCpuBuildTest, ProjectionLightSourceIsSampleableAtAShadingPoint) {
	// Like GoniometricLight, pbrt-v4's ProjectionLight has no "from"/"to" -
	// position/aim come purely from the CTM (Translate here places it at
	// (0,0,-10) looking down its local +z, which with no Rotate is world
	// +z - straight at the quad's shading point at the origin). No
	// "filename" either (see flatten()'s own warning for this case) - the
	// uniform-white-beam fallback should still light a point inside its fov.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 -10\n"
		"LightSource \"projection\" \"float scale\" [ 100 ] \"float fov\" [ 60 ]\n"
		+ std::string(kQuad));
	ASSERT_NE(b.punctLights, nullptr);

	int samples = 0;
	b.punctLights->for_each_sample(point3(0, 0, 0), [&](const PunctualLiSample &s) {
		++samples;
		EXPECT_GT(s.Li.x(), 0.0) << "on-axis point should fall inside the projected beam";
	});
	EXPECT_EQ(samples, 1);
}

// ---------------------------------------------------------------------------
// Material "dielectric" with/without roughness
//
// A nonzero "roughness" on a pbrt dielectric asks for the GGX microfacet
// variant (pbrt-v4 DielectricBxDF's rough path) - this codebase's real model
// for that is rough_dielectric, not plain dielectric. Pins that the roughness
// parameter actually routes to the different class rather than being parsed
// and silently dropped (a pre-existing gap fixed alongside the Metal/
// rough_metal GPU mismatch this session).
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, SmoothDielectricBuildsThePlainDielectricClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"dielectric\" \"float eta\" [ 1.5 ]\n" + std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<dielectric *>(rec.mat.get()), nullptr)
		<< "a pbrt dielectric with no roughness must build the plain smooth "
		   "dielectric class";
	EXPECT_EQ(dynamic_cast<rough_dielectric *>(rec.mat.get()), nullptr);
}

TEST(PbrtCpuBuildTest, RoughDielectricRoughnessBuildsTheRealRoughDielectricClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"dielectric\" \"float eta\" [ 1.5 ] \"float roughness\" [ 0.2 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<rough_dielectric *>(rec.mat.get()), nullptr)
		<< "a pbrt dielectric with nonzero roughness must build the real "
		   "rough_dielectric (GGX) class, not silently drop the roughness "
		   "and render a perfect mirror/refractor";
}

// ---------------------------------------------------------------------------
// Material "conductor"
//
// pbrt describes a conductor's complex IOR via "spectrum eta"/"spectrum k"
// bound to a NAMED spectrum ("metal-Ag-eta"/"metal-Ag-k") - a recognized
// metal name should build the real `conductor` class (GGX + complex
// Fresnel), not the flat-albedo `metal` fuzz-mirror approximation every
// pbrt conductor silently fell back to before.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, NamedMetalSpectrumConductorBuildsTheRealConductorClass) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"spectrum eta\" [ \"metal-Ag-eta\" ] "
		"\"spectrum k\" [ \"metal-Ag-k\" ] \"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<conductor *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor bound to a recognized named metal spectrum "
		   "(metal-Ag-eta/metal-Ag-k) must build the real conductor (GGX + "
		   "complex Fresnel) class, not fall back to the flat-albedo metal "
		   "fuzz-mirror approximation";
}

TEST(PbrtCpuBuildTest, UnrecognizedConductorSpectrumFallsBackToMetal) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"conductor\" \"rgb reflectance\" [ .8 .8 .8 ] "
		"\"float roughness\" [ 0.1 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<metal *>(rec.mat.get()), nullptr)
		<< "a pbrt conductor with no eta/k (or an explicit RGB k) must keep "
		   "the pre-existing metal fuzz-mirror approximation";
	EXPECT_EQ(dynamic_cast<conductor *>(rec.mat.get()), nullptr);
}

// ---------------------------------------------------------------------------
// Material "mix"
//
// pbrt_flatten_tests.cpp pins the name resolution; these pin the next stage -
// that a resolved pbrt_flatten::Material with MaterialKind::Mix actually
// becomes a real mix_material wrapping the real CPU classes its two named
// sub-materials asked for (not two more Lambertians), matching how the
// Coated/DiffuseTransmission tests above already pin their own real classes.
// ---------------------------------------------------------------------------

TEST(PbrtCpuBuildTest, MixMaterialBuildsTheRealMixOfItsTwoSubMaterials) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMaterial \"a\" \"string type\" [ \"conductor\" ] \"rgb reflectance\" [ .2 .4 .6 ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"dielectric\" ] \"float eta\" [ 1.7 ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ] \"float amount\" [ 0.75 ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *mix = dynamic_cast<mix_material *>(rec.mat.get());
	ASSERT_NE(mix, nullptr)
		<< "a pbrt mix material must build the real mix_material class, not "
		   "silently fall back to lambertian";
	EXPECT_NE(dynamic_cast<metal *>(mix->get_mat_a().get()), nullptr)
		<< "sub-material 'a' (conductor) must build the real metal class";
	EXPECT_NE(dynamic_cast<dielectric *>(mix->get_mat_b().get()), nullptr)
		<< "sub-material 'b' (dielectric) must build the real dielectric class";
	EXPECT_DOUBLE_EQ(mix->get_weight()->value(0, 0, point3(0, 0, 0)).x(), 0.75);
}

TEST(PbrtCpuBuildTest, MalformedMixFallsBackToLambertianLikeAnyOtherUnsupportedMaterial) {
	// flatten() already downgrades an unresolvable mix to MaterialKind::
	// Unsupported (pinned in pbrt_flatten_tests.cpp) - this just confirms
	// the builder's existing Unsupported->lambertian fallback still applies,
	// rather than the builder crashing or constructing a mix_material with
	// dangling sub-material indices.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Material \"mix\" \"string materials\" [ \"nope\" \"alsonope\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<lambertian *>(rec.mat.get()), nullptr);
}
