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
