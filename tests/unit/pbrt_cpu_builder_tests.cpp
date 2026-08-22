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
#include "pbrt_load.h"
#include "pbrt_scene.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

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

// Hand-authored minimal 1x1 24bpp BMP - same technique as pbrt_alpha_cutout_
// tests.cpp's own solidBmp1x1() (duplicated, not shared - small self-
// contained test-file helpers). A uniform-gray pixel decodes as a genuine
// grayscale bump/displacement map (is_grayscale_image()'s own convention).
std::string solidGrayBmp1x1() {
	std::string bytes;
	const auto u16 = [&](unsigned short v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
	};
	const auto u32 = [&](unsigned int v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 16) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 24) & 0xFF));
	};
	bytes.push_back('B'); bytes.push_back('M');
	u32(14 + 40 + 4);
	u32(0);
	u32(14 + 40);
	u32(40);
	u32(1);
	u32(1);
	u16(1);
	u16(24);
	u32(0);
	u32(4);
	u32(0); u32(0);
	u32(0); u32(0);
	bytes.push_back(static_cast<char>(128));
	bytes.push_back(static_cast<char>(128));
	bytes.push_back(static_cast<char>(128));
	bytes.push_back('\0');
	return bytes;
}

class CpuBuilderTempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_cpu_builder_tests/";
		std::string cmd = "if not exist \"" + root_ + "\" mkdir \"" + root_ + "\" >nul 2>&1";
		for (char &c : cmd) if (c == '/') c = '\\';
		std::system(cmd.c_str());
	}
	void TearDown() override {
		for (const std::string &f : written_) std::remove(f.c_str());
	}
	void write(const std::string &relative, const std::string &contents) {
		const std::string full = root_ + relative;
		std::ofstream out(full, std::ios::binary);
		out << contents;
		out.close();
		written_.push_back(full);
	}
	std::string path(const std::string &relative) const { return root_ + relative; }
private:
	std::string root_;
	std::vector<std::string> written_;
};

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

TEST(PbrtCpuBuildTest, DiskIsBuiltAtItsTransformedPositionAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 0 0 10\n"
		"Scale 2 2 2\n"
		"Shape \"disk\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.diskCount, 1u);
	double t = 0.0;
	// A disk at object-space z=0, radius 1, scaled by 2 and moved to z=10:
	// world-space disk of radius 2 centred at (0,0,10), still facing +Z.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(0, 0, 1), t));
	EXPECT_NEAR(t, 10.0, 1e-6);
}

TEST(PbrtCpuBuildTest, CylinderIsBuiltAtItsTransformedPositionAndRadius) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"Translate 10 0 0\n"
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	EXPECT_EQ(b.cylinderCount, 1u);
	double t = 0.0;
	// The cylinder's axis (object-space Z) moves to run through x=10, y=0.
	// A ray fired along +X hits the near wall at x=10-radius=9.
	ASSERT_TRUE(castRay(b, point3(0, 0, 0), vec3(1, 0, 0), t));
	EXPECT_NEAR(t, 9.0, 1e-6);
}

TEST(PbrtCpuBuildTest, CylinderHasNoEndCaps) {
	// pbrt's cylinder (and this project's CylinderShape<T> port) is an open
	// tube, not a capped can - matching DiskShape/CylinderShape's own
	// intersect() which only solves for the lateral surface. A ray travelling
	// straight down the axis must pass through untouched.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Shape \"cylinder\" \"float radius\" [ 1 ] \"float zmin\" [ -1 ] "
		"\"float zmax\" [ 1 ]\n");
	double t = 0.0;
	EXPECT_FALSE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "a ray down the cylinder's own axis should exit through the open "
		   "end, not bounce off a cap that doesn't exist";
}

TEST(PbrtCpuBuildTest, DiskUnderRotationHitsAtTheExactTransformedPosition) {
	// Disk/Cylinder deliberately keep their CTM unbaked (see
	// disk_cylinder_hittable.h's own header comment) instead of following
	// Sphere's "bake to world-space centre+radius, warn under anisotropic
	// scale" approximation - Sphere can get away with that because it's
	// rotation-invariant, but a disk's orientation is exactly what a rotation
	// changes. This proves a rotated disk is hit exactly, not approximated.
	//
	// Rotating 90 degrees about X leaves the local X axis fixed and swaps
	// local Y and Z, so a disk originally spanning the object-space XY plane
	// (its outward normal along +/-Z) ends up spanning the world-space XZ
	// plane instead - regardless of the handedness convention for the
	// rotation's sign, since X is untouched either way.
	const pbrt_cpu::BuildResult b =
		buildFrom("Rotate 90 1 0 0\nShape \"disk\" \"float radius\" [ 1 ]\n");
	EXPECT_EQ(b.diskCount, 1u);
	double t = 0.0;
	// Local point (x=0.5, z=0) is within the unit disk (radius 0.5 < 1) and
	// keeps x=0.5 under the rotation, landing in the world-space y=0 plane.
	ASSERT_TRUE(castRay(b, point3(0.5, -5, 0), vec3(0, 1, 0), t));
	EXPECT_NEAR(t, 5.0, 1e-6);
}

TEST(PbrtCpuBuildTest, MediumInterfaceWrapsTheSphereInAParticipatingMedium) {
	// The precise structural claim - the sphere resolves to the right
	// FlatScene::media index, with the right coefficients - lives in
	// pbrt_flatten_tests.cpp, which can check it before pbrt_cpu_builder.h's
	// own final BVH-folding step collapses every top-level hittable (sphere,
	// constant_medium wrapper, everything else) into a single bvh_node,
	// making world->objects.size() always 1 regardless of scene content and
	// unusable as a "how many things got added" signal here. This test only
	// confirms the CPU builder actually consumes that index without
	// crashing and the geometry stays reachable.
	const pbrt_cpu::BuildResult b = buildFrom(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.1 0.1 ] \"rgb sigma_s\" [ 2 2 2 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);

	// constant_medium stochastically decides whether a ray scatters inside
	// the volume, so this doesn't assert a specific outcome - only that the
	// medium-wrapped sphere is still reachable at all, the same "did it
	// build and can a ray find it" bar castRay() checks for plain geometry.
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t))
		<< "a ray toward the medium-wrapped sphere should still hit something";
}

TEST(PbrtCpuBuildTest, UnknownMediumInterfaceNameIsTreatedAsVacuum) {
	// No MakeNamedMedium declares "ghost" - the parser should warn and fall
	// back to insideMedium=-1 (vacuum) rather than crash or misindex. See
	// FlattenTest.UnresolvedMediumNameIsVacuumAndWarns for the precise
	// index-level check; this only confirms the CPU builder still produces
	// an ordinary, reachable sphere rather than failing to build at all.
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  MediumInterface \"ghost\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(0, 0, -5), vec3(0, 0, 1), t));
}

TEST(PbrtCpuBuildTest, CloudMediumIsReachable) {
	// Like MediumInterfaceWrapsTheSphereInAParticipatingMedium above, this
	// only confirms the standalone cloud_medium_hittable (added independently
	// of the declaring sphere - see addMediumIfPresent's own comment) builds
	// without crashing and stays reachable; CloudMedium's own density math is
	// covered by cloud_medium_hittable_tests.cpp / cloud_medium_tests.cpp.
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"puff\" \"string type\" [ \"cloud\" ]\n"
		"    \"rgb sigma_s\" [ 5 5 5 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"puff\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the cloud medium's world AABB should still hit something";
}

TEST(PbrtCpuBuildTest, RgbGridMediumIsReachable) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"nebula\" \"string type\" [ \"rgbgrid\" ]\n"
		"    \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"    \"rgb sigma_s\" [ 1 1 1  2 2 2 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"nebula\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the rgbgrid medium's world AABB should still hit something";
}

TEST(PbrtCpuBuildTest, UniformgridMediumIsReachable) {
	const pbrt_cpu::BuildResult b = buildFrom(
		"AttributeBegin\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"fog\" \"string type\" [ \"uniformgrid\" ]\n"
		"    \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"    \"float density\" [ 0.5 1.0 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  Translate 1 1 1\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 2 ]\n"
		"AttributeEnd\n");
	EXPECT_EQ(b.sphereCount, 1u);
	double t = 0.0;
	EXPECT_TRUE(castRay(b, point3(1, 1, -5), vec3(0, 0, 1), t))
		<< "a ray toward the uniformgrid medium's world AABB should still hit something";
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

TEST(PbrtCpuBuildTest, AreaLightFilenameBuildsATextureBackedTwoSidedDiffuseLight) {
	// buildFrom() only runs flatten(), not pbrt_load.h's resolution pass (see
	// DiffuseReflectanceImagemapBuildsATextureBackedLambertian's own comment
	// on the same trade-off), so this only checks that a non-empty
	// Emission::filename makes it all the way to a texture-backed,
	// is_two_sided() diffuse_light - makeMaterial()'s own filename branch -
	// rather than the flat-L solid_color-backed one.
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"string filename\" [ \"glow.png\" ]"
					" \"bool twosided\" [ true ]\n")
		+ kQuad + "AttributeEnd\n");
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dl = dynamic_cast<diffuse_light *>(rec.mat.get());
	ASSERT_NE(dl, nullptr);
	EXPECT_TRUE(dl->is_two_sided());
	EXPECT_EQ(dynamic_cast<mipmap_texture *>(dl->get_texture().get()), dl->get_texture().get())
		<< "a \"filename\" area light must build a mipmap_texture-backed "
		   "diffuse_light, not a flat solid_color one";
}

TEST(PbrtCpuBuildTest, AreaLightWithoutFilenameBuildsAPlainSolidColorDiffuseLight) {
	const pbrt_cpu::BuildResult b = buildFrom(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuad + "AttributeEnd\n");
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *dl = dynamic_cast<diffuse_light *>(rec.mat.get());
	ASSERT_NE(dl, nullptr);
	EXPECT_FALSE(dl->is_two_sided());
	EXPECT_EQ(dynamic_cast<mipmap_texture *>(dl->get_texture().get()), nullptr);
}

TEST(PbrtCpuBuildTest, NonEmissiveSceneHasAnEmptyLightList) {
	const pbrt_cpu::BuildResult b = buildFrom(kQuad);
	EXPECT_TRUE(b.lights->objects.empty());
}

TEST(PbrtCpuBuildTest, AnEmptySceneBuildsWithoutCrashing) {
	// flatten() drops unsupported shapes, so a scene can legitimately arrive
	// with nothing in it. A BVH must not be built over zero primitives.
	// "cone" - not "cylinder"/"disk", which this loader now supports.
	const pbrt_cpu::BuildResult b = buildFrom("Shape \"cone\"\n");
	EXPECT_EQ(b.triangleCount, 0u);
	EXPECT_EQ(b.sphereCount, 0u);
	EXPECT_EQ(b.diskCount, 0u);
	EXPECT_EQ(b.cylinderCount, 0u);
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

TEST(PbrtCpuBuildTest, DiffuseReflectanceImagemapBuildsATextureBackedLambertian) {
	// buildFrom() only runs flatten(), not pbrt_load.h's resolution pass, so
	// Material::textureFilename stays "as written" here ("t.png") rather than
	// an absolute path - fine for this test, which only checks that a
	// non-empty textureFilename makes it all the way to a mipmap_texture-
	// backed lambertian (this Diffuse case's own branch in makeMaterial()),
	// not a plain solid_color-backed one built from `reflectance`.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<mipmap_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with an imagemap-bound reflectance must build a "
		   "texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceCheckerboardBuildsAUVCheckerBackedLambertian) {
	// Round 5 Phase 2: hasCheckerReflectance must make it all the way to a
	// uv_checker_texture-backed lambertian, and that texture must actually
	// tile by UV (not the unrelated 3D world-space checker_texture) - probed
	// here via value() at two UVs one checker cell apart.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"chk\" \"spectrum\" \"checkerboard\" "
		"\"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ] "
		"\"float uscale\" [ 1 ] \"float vscale\" [ 1 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.25, 0.25, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *chk = dynamic_cast<uv_checker_texture *>(lam->get_texture().get());
	ASSERT_NE(chk, nullptr)
		<< "a diffuse material with a checkerboard-bound reflectance must "
		   "build a uv_checker_texture-backed lambertian";
	const color c00 = chk->value(0.25, 0.25, rec.p);
	const color c10 = chk->value(1.25, 0.25, rec.p);
	EXPECT_NE(c00.x(), c10.x())
		<< "adjacent UV checker cells (one uscale apart) must differ";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceFbmBuildsAnFbmBackedLambertian) {
	// Round 6 Phase 1: hasFbmReflectance must make it all the way to an
	// fbm_texture-backed lambertian, reusing the existing CPU class rather
	// than falling back to a flat colour.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"cloud\" \"float\" \"fbm\" \"integer octaves\" [ 4 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<fbm_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with an fbm-bound reflectance must build an "
		   "fbm_texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceMarbleBuildsAMarbleBackedLambertian) {
	// Round 6 Phase 1: hasMarbleReflectance must make it all the way to a
	// marble_texture-backed lambertian.
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"stone\" \"spectrum\" \"marble\"\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	EXPECT_NE(dynamic_cast<marble_texture *>(lam->get_texture().get()), nullptr)
		<< "a diffuse material with a marble-bound reflectance must build a "
		   "marble_texture-backed lambertian, not the flat-colour fallback";
}

TEST(PbrtCpuBuildTest, DiffuseReflectanceMixBuildsAMixBackedLambertian) {
	// Round 6 Phase 1: hasMixReflectance must make it all the way to a
	// mix_texture-backed lambertian that actually blends tex1/tex2 by
	// amount (deterministic, unlike fbm/marble - so the resulting colour
	// can be checked exactly).
	const pbrt_cpu::BuildResult b = buildFrom(
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float amount\" [ 0.25 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuad));
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	auto *lam = dynamic_cast<lambertian *>(rec.mat.get());
	ASSERT_NE(lam, nullptr);
	auto *mix = dynamic_cast<mix_texture *>(lam->get_texture().get());
	ASSERT_NE(mix, nullptr)
		<< "a diffuse material with a mix-bound reflectance must build a "
		   "mix_texture-backed lambertian, not the flat-colour fallback";
	const color c = mix->value(0.0, 0.0, rec.p);
	EXPECT_NEAR(c.x(), 0.75, 1e-9);  // (1-0.25)*1 + 0.25*0
	EXPECT_NEAR(c.z(), 0.25, 1e-9);  // (1-0.25)*0 + 0.25*1
}

TEST_F(CpuBuilderTempTree, DisplacementWithARealGrayscaleBumpMapWrapsInBumpMapMaterial) {
	// Round 5 Phase 1: Material::displacementTextureFilename (see that
	// field's own comment) must reach the CPU builder's materialFor()
	// lambda and wrap the base material in bump_map_material - the same
	// decorator mesh.h's own OBJ/MTL map_Bump dispatch already uses for a
	// genuine grayscale height/displacement image. Goes through the real
	// pbrt_load::loadFile() resolution pass (not just flatten()) so the
	// path is a real, existence-tested, absolute file pbrt_cpu_builder.h's
	// rtw_image probe can actually decode.
	write("scene.pbrt",
		  "Texture \"bmap\" \"float\" \"imagemap\" \"string filename\" [ \"bump.bmp\" ]\n"
		  "Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ] "
		  "\"texture displacement\" [ \"bmap\" ]\n"
		  + std::string(kQuad));
	write("bump.bmp", solidGrayBmp1x1());

	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;
	ASSERT_EQ(loaded.scene.materials.size(), 1u);
	ASSERT_EQ(loaded.scene.materials[0].displacementTextureFilename, path("bump.bmp"));

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	hit_record rec;
	ASSERT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
	EXPECT_NE(dynamic_cast<bump_map_material *>(rec.mat.get()), nullptr)
		<< "a material with a resolved, decodable displacement texture must "
		   "wrap the base material in bump_map_material";
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
