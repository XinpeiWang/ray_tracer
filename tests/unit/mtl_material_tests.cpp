// mtl_material_tests.cpp
// Regression coverage for load_obj_mtl()/parse_mtl() (mesh.h), added when
// Sponza/Bistro/Rungholt (scenes 62-64) were switched from one flat material
// per mesh to real per-face colors sourced from the companion .mtl file's
// Kd (diffuse) entries. Uses small synthetic .obj/.mtl fixtures rather than
// the real multi-hundred-MB scene assets, so this stays fast and
// deterministic.
#include <gtest/gtest.h>
#include "mesh.h"
#include "material_simple.h"
#include <cstdio>
#include <fstream>
#include <memory>

namespace {

std::string write_temp_file(const std::string& path, const std::string& contents) {
	std::ofstream f(path, std::ios::binary);
	f << contents;
	f.close();
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
