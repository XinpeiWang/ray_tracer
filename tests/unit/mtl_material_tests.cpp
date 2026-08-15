// mtl_material_tests.cpp
// Regression coverage for load_obj_mtl()/parse_mtl()/parse_mtl_textures()/
// resolve_mtl_texture_path() (mesh.h), added when Sponza/Bistro/Rungholt
// (scenes 62-64) were switched from one flat material per mesh to real
// per-face colors -- and, for Sponza/Bistro, real map_Kd image textures --
// sourced from the companion .mtl file. Uses small synthetic .obj/.mtl
// fixtures rather than the real multi-hundred-MB/GB scene assets, so this
// stays fast and deterministic.
#include <gtest/gtest.h>
#include "mesh.h"
#include "material_simple.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

namespace {

std::string write_temp_file(const std::string& path, const std::string& contents) {
	std::ofstream f(path, std::ios::binary);
	f << contents;
	f.close();
	return path;
}

// Writes a solid-color 1x1 RGB PNG -- used to build real alpha-cutout mask
// fixtures (map_d), since is_grayscale_image()/the alpha test both need to
// read actual decoded pixel data, not just a filename.
std::string write_solid_png(const std::string& path, unsigned char r, unsigned char g, unsigned char b) {
	unsigned char px[3] = { r, g, b };
	stbi_write_png(path.c_str(), 1, 1, 3, px, 3);
	return path;
}

// Reads back a lambertian triangle's flat color via its texture, the same
// route material_simple.h's own scatter() uses to fill srec.attenuation.
color lambertian_color(const std::shared_ptr<material>& mat) {
	auto lam = std::dynamic_pointer_cast<lambertian>(mat);
	if (!lam) return color(-1, -1, -1); // signals "not lambertian" to callers
	return lam->get_texture()->value(0, 0, point3(0, 0, 0));
}

} // namespace

// Two triangles sharing a quad, each under a different usemtl, resolved via
// a companion .mtl file's Kd colors -- confirms per-face (not per-mesh)
// material assignment actually works.
TEST(MtlMaterial, PerFaceMaterialsFromCompanionMtl) {
	write_temp_file("mtl_material_test_tmp.mtl",
		"newmtl Red\n"
		"Kd 1.0 0.0 0.0\n"
		"newmtl Blue\n"
		"Kd 0.0 0.0 1.0\n");
	std::string objPath = write_temp_file("mtl_material_test_tmp.obj",
		"mtllib mtl_material_test_tmp.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"v 1 1 0\n"
		"usemtl Red\n"
		"f 1 2 3\n"
		"usemtl Blue\n"
		"f 2 4 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto mesh = load_obj_mtl(objPath, fallback);
	ASSERT_NE(mesh, nullptr);

	// Ray through the first triangle's interior (centroid-ish (0.2,0.2)) --
	// should resolve to the Red material.
	ray rRed(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record recRed;
	ASSERT_TRUE(mesh->hit(rRed, interval(0.001, infinity), recRed));
	color cRed = lambertian_color(recRed.mat);
	EXPECT_NEAR(cRed.x(), 1.0, 1e-9);
	EXPECT_NEAR(cRed.y(), 0.0, 1e-9);
	EXPECT_NEAR(cRed.z(), 0.0, 1e-9);

	// Ray through the second triangle's interior (centroid-ish (0.7,0.7)) --
	// should resolve to the Blue material, proving the two faces got
	// genuinely different materials instead of one flat mesh-wide color.
	ray rBlue(point3(0.7, 0.7, -5), vec3(0, 0, 1));
	hit_record recBlue;
	ASSERT_TRUE(mesh->hit(rBlue, interval(0.001, infinity), recBlue));
	color cBlue = lambertian_color(recBlue.mat);
	EXPECT_NEAR(cBlue.x(), 0.0, 1e-9);
	EXPECT_NEAR(cBlue.y(), 0.0, 1e-9);
	EXPECT_NEAR(cBlue.z(), 1.0, 1e-9);

	std::remove("mtl_material_test_tmp.obj");
	std::remove("mtl_material_test_tmp.mtl");
}

// A face with no usemtl at all (mtl name is empty) must fall back to the
// caller-supplied fallback material rather than crash or pick an arbitrary
// entry from the .mtl.
TEST(MtlMaterial, FaceWithNoUsemtlUsesFallback) {
	write_temp_file("mtl_material_test_nousemtl.mtl",
		"newmtl Green\n"
		"Kd 0.0 1.0 0.0\n");
	std::string objPath = write_temp_file("mtl_material_test_nousemtl.obj",
		"mtllib mtl_material_test_nousemtl.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"f 1 2 3\n"); // no usemtl before this face

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.25, 0.75));
	auto mesh = load_obj_mtl(objPath, fallback);
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	ASSERT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));
	color c = lambertian_color(rec.mat);
	EXPECT_NEAR(c.x(), 0.5, 1e-9);
	EXPECT_NEAR(c.y(), 0.25, 1e-9);
	EXPECT_NEAR(c.z(), 0.75, 1e-9);

	std::remove("mtl_material_test_nousemtl.obj");
	std::remove("mtl_material_test_nousemtl.mtl");
}

// A usemtl name that isn't defined in the .mtl (or a missing .mtl file
// entirely) must degrade to the fallback material instead of throwing --
// this is the real-world case for Sponza/Bistro/Rungholt if their .mtl
// were ever absent from a deploy directory.
TEST(MtlMaterial, MissingMtlFileFallsBackForEveryFace) {
	std::string objPath = write_temp_file("mtl_material_test_missing.obj",
		"mtllib mtl_material_test_missing_DOES_NOT_EXIST.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl SomeUnknownMaterial\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.1, 0.2, 0.3));
	auto mesh = load_obj_mtl(objPath, fallback);
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	ASSERT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));
	color c = lambertian_color(rec.mat);
	EXPECT_NEAR(c.x(), 0.1, 1e-9);
	EXPECT_NEAR(c.y(), 0.2, 1e-9);
	EXPECT_NEAR(c.z(), 0.3, 1e-9);

	std::remove("mtl_material_test_missing.obj");
}

// parse_mtl() in isolation: reads Kd for every newmtl block, ignores
// unrelated tokens (Ka/Ns/map_Kd/...) between them.
TEST(MtlMaterial, ParseMtlReadsKdPerMaterial) {
	write_temp_file("mtl_material_test_parse.mtl",
		"newmtl Stone\n"
		"Ns 10.0\n"
		"Ka 0.1 0.1 0.1\n"
		"Kd 0.62 0.55 0.48\n"
		"map_Kd stone_diffuse.png\n"
		"newmtl Grass\n"
		"Kd 0.30 0.55 0.20\n");

	auto colors = parse_mtl("mtl_material_test_parse.mtl");
	ASSERT_EQ(colors.size(), 2u);
	ASSERT_TRUE(colors.count("Stone"));
	ASSERT_TRUE(colors.count("Grass"));
	EXPECT_NEAR(colors["Stone"].x(), 0.62, 1e-9);
	EXPECT_NEAR(colors["Stone"].y(), 0.55, 1e-9);
	EXPECT_NEAR(colors["Stone"].z(), 0.48, 1e-9);
	EXPECT_NEAR(colors["Grass"].x(), 0.30, 1e-9);
	EXPECT_NEAR(colors["Grass"].y(), 0.55, 1e-9);
	EXPECT_NEAR(colors["Grass"].z(), 0.20, 1e-9);

	std::remove("mtl_material_test_parse.mtl");
}

// parse_mtl_textures(): reads map_Kd per material, ignores materials with
// no map_Kd line, ignores other map_* lines (map_Ka, map_Bump, ...).
TEST(MtlMaterial, ParseMtlTexturesReadsMapKdPerMaterial) {
	write_temp_file("mtl_material_test_textures.mtl",
		"newmtl Textured\n"
		"Kd 1.0 1.0 1.0\n"
		"map_Ka some_ambient.png\n"
		"map_Kd textures\\brick_diff.png\n"
		"newmtl FlatOnly\n"
		"Kd 0.5 0.5 0.5\n");

	auto textures = parse_mtl_textures("mtl_material_test_textures.mtl");
	ASSERT_EQ(textures.size(), 1u);
	ASSERT_TRUE(textures.count("Textured"));
	EXPECT_EQ(textures["Textured"], "textures\\brick_diff.png");
	EXPECT_FALSE(textures.count("FlatOnly"));

	std::remove("mtl_material_test_textures.mtl");
}

// Regression test for a real bug found while adding Bistro's textures:
// some of its map_Kd paths contain literal spaces in a folder name (e.g.
// "..\OtherTextures\Tiling\Metal_ RollDoor_01\Metal_ RollDoor_01_diff.png"),
// which parse_mtl_textures()'s original `ss >> path` silently truncated at
// the first space -- the mesh loaded, but that material's texture quietly
// fell back to the solid-cyan missing-texture placeholder instead of erroring
// loudly, which is exactly the kind of failure that's easy to miss visually.
TEST(MtlMaterial, ParseMtlTexturesHandlesFilenamesWithSpaces) {
	write_temp_file("mtl_material_test_spaces.mtl",
		"newmtl Metal_RollDoor\n"
		"Kd 1.0 1.0 1.0\n"
		"map_Kd ..\\OtherTextures\\Tiling\\Metal_ RollDoor_01\\Metal_ RollDoor_01_diff.png\n");

	auto textures = parse_mtl_textures("mtl_material_test_spaces.mtl");
	ASSERT_TRUE(textures.count("Metal_RollDoor"));
	EXPECT_EQ(textures["Metal_RollDoor"],
		"..\\OtherTextures\\Tiling\\Metal_ RollDoor_01\\Metal_ RollDoor_01_diff.png");

	std::remove("mtl_material_test_spaces.mtl");
}

// Ke (emission): a material with a real, non-zero Ke gets a diffuse_light
// material instead of Kd/lambertian, and its triangles are collected into
// out_lights -- the NEE-light half of the OBJ/.mtl loader completeness work
// (see load_obj_mtl()'s own comment: emissive geometry doubles as a light,
// the same pattern pbrt_cpu_builder.h uses for .pbrt area lights). None of
// the three real large scenes exercise this path (Ke is zero throughout all
// of sponza.mtl/exterior.mtl/rungholt.mtl), hence this synthetic fixture.
TEST(MtlMaterial, EmissiveMaterialBecomesLightAndIsCollected) {
	write_temp_file("mtl_material_test_emission.mtl",
		"newmtl Lamp\n"
		"Kd 0.8 0.8 0.8\n"
		"Ke 5.0 5.0 5.0\n"
		"newmtl Wall\n"
		"Kd 0.5 0.5 0.5\n");
	std::string objPath = write_temp_file("mtl_material_test_emission.obj",
		"mtllib mtl_material_test_emission.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"v 1 1 0\n"
		"usemtl Lamp\n"
		"f 1 2 3\n"
		"usemtl Wall\n"
		"f 2 4 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	hittable_list lights;
	auto mesh = load_obj_mtl(objPath, fallback, 1.0, point3(0,0,0), false, "", &lights);
	ASSERT_NE(mesh, nullptr);

	// The Lamp triangle's material resolves to a diffuse_light, not the
	// usual Kd-backed lambertian.
	ray rLamp(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record recLamp;
	ASSERT_TRUE(mesh->hit(rLamp, interval(0.001, infinity), recLamp));
	EXPECT_NE(std::dynamic_pointer_cast<diffuse_light>(recLamp.mat), nullptr);

	// The Wall triangle (no Ke) stays an ordinary lambertian.
	ray rWall(point3(0.7, 0.7, -5), vec3(0, 0, 1));
	hit_record recWall;
	ASSERT_TRUE(mesh->hit(rWall, interval(0.001, infinity), recWall));
	EXPECT_NE(std::dynamic_pointer_cast<lambertian>(recWall.mat), nullptr);

	// Exactly the one emissive triangle was collected as a light.
	EXPECT_EQ(lights.objects.size(), 1u);

	std::remove("mtl_material_test_emission.obj");
	std::remove("mtl_material_test_emission.mtl");
}

// An explicit "Ke 0 0 0" (the boilerplate default many exporters always
// write -- confirmed to be the case for every material in the three real
// large-scene .mtl files) must NOT be treated as emissive: the material
// should resolve to its ordinary Kd lambertian and contribute nothing to
// out_lights, exactly as if no Ke line were present at all.
TEST(MtlMaterial, ZeroKeIsNotTreatedAsEmissive) {
	write_temp_file("mtl_material_test_zero_ke.mtl",
		"newmtl Boilerplate\n"
		"Kd 0.4 0.4 0.4\n"
		"Ke 0.0 0.0 0.0\n");
	std::string objPath = write_temp_file("mtl_material_test_zero_ke.obj",
		"mtllib mtl_material_test_zero_ke.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl Boilerplate\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	hittable_list lights;
	auto mesh = load_obj_mtl(objPath, fallback, 1.0, point3(0,0,0), false, "", &lights);
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	ASSERT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));
	EXPECT_NE(std::dynamic_pointer_cast<lambertian>(rec.mat), nullptr);
	EXPECT_EQ(lights.objects.size(), 0u);

	std::remove("mtl_material_test_zero_ke.obj");
	std::remove("mtl_material_test_zero_ke.mtl");
}

// parse_mtl_emission() in isolation: reads Ke per material, treats an
// explicit all-zero Ke the same as no Ke line, ignores unrelated tokens.
TEST(MtlMaterial, ParseMtlEmissionReadsKePerMaterialAndSkipsZero) {
	write_temp_file("mtl_material_test_parse_ke.mtl",
		"newmtl Lamp\n"
		"Kd 0.8 0.8 0.8\n"
		"Ke 3.0 2.5 1.0\n"
		"newmtl Wall\n"
		"Kd 0.5 0.5 0.5\n"
		"Ke 0 0 0\n"
		"newmtl NoKeLine\n"
		"Kd 0.2 0.2 0.2\n");

	auto emission = parse_mtl_emission("mtl_material_test_parse_ke.mtl");
	ASSERT_EQ(emission.size(), 1u);
	ASSERT_TRUE(emission.count("Lamp"));
	EXPECT_NEAR(emission["Lamp"].x(), 3.0, 1e-9);
	EXPECT_NEAR(emission["Lamp"].y(), 2.5, 1e-9);
	EXPECT_NEAR(emission["Lamp"].z(), 1.0, 1e-9);
	EXPECT_FALSE(emission.count("Wall"));
	EXPECT_FALSE(emission.count("NoKeLine"));

	std::remove("mtl_material_test_parse_ke.mtl");
}

// Specular dispatch (Ks/Ns/illum): boilerplate illum 2 with near-zero Ks
// stays an ordinary lambertian, illum 2 with a real Ks becomes a metal
// (albedo = Kd, fuzz = phong_to_roughness(Ns)), and illum 7 becomes a
// dielectric with Ni as its IOR. None of the three real large scenes
// exercise the metal/dielectric paths meaningfully (see load_obj_mtl()'s
// own comment history), hence this synthetic fixture.
TEST(MtlMaterial, SpecularDispatchByIllumAndKs) {
	write_temp_file("mtl_material_test_specular.mtl",
		"newmtl Plain\n"
		"Kd 0.5 0.5 0.5\n"
		"Ks 0.01 0.01 0.01\n"
		"Ns 10.0\n"
		"illum 2\n"
		"newmtl Glossy\n"
		"Kd 0.8 0.6 0.2\n"
		"Ks 0.50 0.50 0.50\n"
		"Ns 200.0\n"
		"illum 2\n"
		"newmtl Glass\n"
		"Kd 1.0 1.0 1.0\n"
		"Ni 1.45\n"
		"illum 7\n");
	std::string objPath = write_temp_file("mtl_material_test_specular.obj",
		"mtllib mtl_material_test_specular.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"v 1 1 0\n"
		"v 2 0 0\n"
		"v 2 1 0\n"
		"usemtl Plain\n"
		"f 1 2 3\n"
		"usemtl Glossy\n"
		"f 2 4 3\n"
		"usemtl Glass\n"
		"f 2 5 6\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto mesh = load_obj_mtl(objPath, fallback);
	ASSERT_NE(mesh, nullptr);

	// Plain: illum 2 but Ks is negligible (0.01, boilerplate-level) -- stays
	// lambertian, same as if illum/Ks weren't in the file at all.
	ray rPlain(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record recPlain;
	ASSERT_TRUE(mesh->hit(rPlain, interval(0.001, infinity), recPlain));
	EXPECT_NE(std::dynamic_pointer_cast<lambertian>(recPlain.mat), nullptr);

	// Glossy: illum 2 with a real Ks (0.5) -- becomes a metal(Kd, fuzz).
	ray rGlossy(point3(0.7, 0.7, -5), vec3(0, 0, 1));
	hit_record recGlossy;
	ASSERT_TRUE(mesh->hit(rGlossy, interval(0.001, infinity), recGlossy));
	auto glossyMetal = std::dynamic_pointer_cast<metal>(recGlossy.mat);
	ASSERT_NE(glossyMetal, nullptr);
	color glossyAlbedo = glossyMetal->get_albedo();
	EXPECT_NEAR(glossyAlbedo.x(), 0.8, 1e-9);
	EXPECT_NEAR(glossyAlbedo.y(), 0.6, 1e-9);
	EXPECT_NEAR(glossyAlbedo.z(), 0.2, 1e-9);
	EXPECT_NEAR(glossyMetal->get_fuzz(), phong_to_roughness(200.0), 1e-9);

	// Glass: illum 7 -- becomes a dielectric with Ni as its IOR. Face is
	// v2(1,0,0), v5(2,0,0), v6(2,1,0); (1.7,0.3) sits inside that triangle.
	ray rGlass(point3(1.7, 0.3, -5), vec3(0, 0, 1));
	hit_record recGlass;
	ASSERT_TRUE(mesh->hit(rGlass, interval(0.001, infinity), recGlass));
	EXPECT_NE(std::dynamic_pointer_cast<dielectric>(recGlass.mat), nullptr);

	std::remove("mtl_material_test_specular.obj");
	std::remove("mtl_material_test_specular.mtl");
}

// illum 7 with no Ni line at all must default to IOR 1.5 rather than an
// uninitialized/garbage value -- confirmed indirectly via a successful hit
// (a NaN/zero IOR dielectric would misbehave badly enough to be visually
// obvious, but this at least pins that the material resolves at all and
// stays a dielectric).
TEST(MtlMaterial, Illum7WithoutNiStillResolvesToDielectric) {
	write_temp_file("mtl_material_test_no_ni.mtl",
		"newmtl Glass\n"
		"Kd 1.0 1.0 1.0\n"
		"illum 7\n");
	std::string objPath = write_temp_file("mtl_material_test_no_ni.obj",
		"mtllib mtl_material_test_no_ni.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl Glass\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto mesh = load_obj_mtl(objPath, fallback);
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	ASSERT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));
	EXPECT_NE(std::dynamic_pointer_cast<dielectric>(rec.mat), nullptr);

	std::remove("mtl_material_test_no_ni.obj");
	std::remove("mtl_material_test_no_ni.mtl");
}

// parse_mtl_specular() in isolation: reads Ks/Ns/illum/Ni per material. A
// material with none of those four tokens (Kd-only) gets no entry at all --
// letting load_obj_mtl()'s .find() naturally skip specular dispatch for it,
// same as if the .mtl had no Ks/Ns/illum/Ni concept whatsoever. A material
// that specifies illum but no Ni keeps Ni at its -1 "not specified"
// sentinel, so callers can tell that apart from an explicit "Ni 0".
TEST(MtlMaterial, ParseMtlSpecularReadsFieldsPerMaterial) {
	write_temp_file("mtl_material_test_parse_specular.mtl",
		"newmtl Full\n"
		"Kd 0.5 0.5 0.5\n"
		"Ks 0.3 0.4 0.5\n"
		"Ns 64.0\n"
		"illum 2\n"
		"Ni 1.33\n"
		"newmtl IllumOnly\n"
		"Kd 0.4 0.4 0.4\n"
		"illum 1\n"
		"newmtl KdOnly\n"
		"Kd 0.2 0.2 0.2\n");

	auto spec = parse_mtl_specular("mtl_material_test_parse_specular.mtl");
	ASSERT_TRUE(spec.count("Full"));
	EXPECT_NEAR(spec["Full"].ks.x(), 0.3, 1e-9);
	EXPECT_NEAR(spec["Full"].ks.y(), 0.4, 1e-9);
	EXPECT_NEAR(spec["Full"].ks.z(), 0.5, 1e-9);
	EXPECT_NEAR(spec["Full"].ns, 64.0, 1e-9);
	EXPECT_EQ(spec["Full"].illum, 2);
	EXPECT_NEAR(spec["Full"].ni, 1.33, 1e-9);

	ASSERT_TRUE(spec.count("IllumOnly"));
	EXPECT_EQ(spec["IllumOnly"].illum, 1);
	EXPECT_LT(spec["IllumOnly"].ni, 0.0);

	EXPECT_FALSE(spec.count("KdOnly"));

	std::remove("mtl_material_test_parse_specular.mtl");
}

// resolve_mtl_texture_path(): backslashes normalize to forward slashes, and
// leading "../"/"./" segments are stripped (textures are relocated under
// texture_dir rather than mirroring the original archive's exact nesting).
TEST(MtlMaterial, ResolveMtlTexturePathNormalizesAndStripsRelativePrefix) {
	EXPECT_EQ(resolve_mtl_texture_path("textures\\foo.png", "models/sponza_textures"),
		"models/sponza_textures/textures/foo.png");
	EXPECT_EQ(resolve_mtl_texture_path("..\\BuildingTextures\\bar.png", "models/bistro_textures"),
		"models/bistro_textures/BuildingTextures/bar.png");
	EXPECT_EQ(resolve_mtl_texture_path("./flat.png", "models/x"),
		"models/x/flat.png");
}

// When texture_dir is given but the resolved map_Kd image doesn't actually
// exist on disk, load_obj_mtl() must degrade to the material's flat Kd
// color (or fallback_mat if it has neither) instead of crashing or silently
// producing a broken/cyan-debug texture -- this is the real-world case for
// any material whose specific texture file is missing from a deploy.
TEST(MtlMaterial, MissingTextureFileFallsBackToFlatKd) {
	write_temp_file("mtl_material_test_missing_tex.mtl",
		"newmtl HasBoth\n"
		"Kd 0.2 0.4 0.6\n"
		"map_Kd textures\\does_not_exist_anywhere.png\n");
	std::string objPath = write_temp_file("mtl_material_test_missing_tex.obj",
		"mtllib mtl_material_test_missing_tex.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl HasBoth\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.9, 0.9, 0.9));
	auto mesh = load_obj_mtl(objPath, fallback, 1.0, point3(0,0,0), false, "models/nonexistent_texture_dir");
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	ASSERT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));
	color c = lambertian_color(rec.mat);
	EXPECT_NEAR(c.x(), 0.2, 1e-9);
	EXPECT_NEAR(c.y(), 0.4, 1e-9);
	EXPECT_NEAR(c.z(), 0.6, 1e-9);

	std::remove("mtl_material_test_missing_tex.obj");
	std::remove("mtl_material_test_missing_tex.mtl");
}

// Alpha-cutout (map_d): a fully-opaque (white) mask must not change
// whether the triangle is hit -- confirms the >= threshold comparison in
// triangle::hit() keeps an opaque pixel, not just that it rejects a
// transparent one.
TEST(MtlMaterial, AlphaCutoutOpaqueMaskKeepsHit) {
	write_solid_png("mtl_material_test_alpha_opaque.png", 255, 255, 255);
	write_temp_file("mtl_material_test_alpha_opaque.mtl",
		"newmtl OpaqueMask\n"
		"Kd 0.4 0.6 0.8\n"
		"map_d mtl_material_test_alpha_opaque.png\n");
	std::string objPath = write_temp_file("mtl_material_test_alpha_opaque.obj",
		"mtllib mtl_material_test_alpha_opaque.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl OpaqueMask\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto mesh = load_obj_mtl(objPath, fallback, 1.0, point3(0,0,0), false, ".");
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	EXPECT_TRUE(mesh->hit(r, interval(0.001, infinity), rec));

	std::remove("mtl_material_test_alpha_opaque.obj");
	std::remove("mtl_material_test_alpha_opaque.mtl");
	std::remove("mtl_material_test_alpha_opaque.png");
}

// Alpha-cutout (map_d): a fully-transparent (black) mask must make the
// triangle unhittable at every point on its surface -- this is the actual
// leaf/foliage-cutout behavior (Sponza's leaf/vase_plant/chain materials,
// Bistro's tree/hedge foliage), confirmed here with a synthetic fixture
// since the real scenes only cut out PART of each texture, not the whole
// thing, making a targeted pass/fail assertion on a specific real pixel
// impractical without shipping a real texture file into the test tree.
TEST(MtlMaterial, AlphaCutoutTransparentMaskRejectsHit) {
	write_solid_png("mtl_material_test_alpha_cutout.png", 0, 0, 0);
	write_temp_file("mtl_material_test_alpha_cutout.mtl",
		"newmtl CutoutLeaf\n"
		"Kd 0.2 0.6 0.2\n"
		"map_d mtl_material_test_alpha_cutout.png\n");
	std::string objPath = write_temp_file("mtl_material_test_alpha_cutout.obj",
		"mtllib mtl_material_test_alpha_cutout.mtl\n"
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"usemtl CutoutLeaf\n"
		"f 1 2 3\n");

	auto fallback = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	auto mesh = load_obj_mtl(objPath, fallback, 1.0, point3(0,0,0), false, ".");
	ASSERT_NE(mesh, nullptr);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	EXPECT_FALSE(mesh->hit(r, interval(0.001, infinity), rec));

	std::remove("mtl_material_test_alpha_cutout.obj");
	std::remove("mtl_material_test_alpha_cutout.mtl");
	std::remove("mtl_material_test_alpha_cutout.png");
}

// parse_mtl_alpha_textures(): reads map_d per material, ignores materials
// with no map_d line, matches the OBJ/.mtl spec's "map_D" alternate casing
// too.
TEST(MtlMaterial, ParseMtlAlphaTexturesReadsMapDPerMaterial) {
	write_temp_file("mtl_material_test_parse_alpha.mtl",
		"newmtl Leaf\n"
		"Kd 0.2 0.6 0.2\n"
		"map_d leaf_mask.png\n"
		"newmtl AltCasing\n"
		"Kd 0.5 0.5 0.5\n"
		"map_D alt_mask.png\n"
		"newmtl FlatOnly\n"
		"Kd 0.3 0.3 0.3\n");

	auto alpha = parse_mtl_alpha_textures("mtl_material_test_parse_alpha.mtl");
	ASSERT_TRUE(alpha.count("Leaf"));
	EXPECT_EQ(alpha["Leaf"], "leaf_mask.png");
	ASSERT_TRUE(alpha.count("AltCasing"));
	EXPECT_EQ(alpha["AltCasing"], "alt_mask.png");
	EXPECT_FALSE(alpha.count("FlatOnly"));

	std::remove("mtl_material_test_parse_alpha.mtl");
}
