// obj_negative_indices_tests.cpp
// Regression test for a real bug found while adding the Rungholt scene
// (McGuire Computer Graphics Archive): mesh.h's load_obj() only handled
// positive (absolute, 1-based) face-vertex indices. The OBJ spec also
// allows negative (relative) indices -- -1 means "the most recently
// defined v/vt/vn", -2 the one before that, etc. -- which rungholt.obj
// uses throughout. Before the fix, `fv.p = pi - 1` on a negative pi
// produced a negative/nonsensical index, tripping the `if (fv.p >= 0)`
// guard and silently DROPPING that face vertex -- large chunks of
// geometry would vanish with no error, not even a crash.
#include <gtest/gtest.h>
#include "mesh.h"
#include "material_simple.h"
#include <cstdio>
#include <fstream>
#include <memory>

namespace {

std::string write_temp_obj(const std::string& contents) {
	std::string path = "obj_negative_indices_test_tmp.obj";
	std::ofstream f(path, std::ios::binary);
	f << contents;
	f.close();
	return path;
}

} // namespace

// A single triangle referencing all three vertices via negative (relative)
// indices: -1/-2/-3 immediately after the three `v` lines. Before the fix,
// every one of these resolved to a nonsense negative array index and the
// whole face was dropped -- the mesh would load with zero triangles instead
// of throwing (load_obj throws only when faces AND positions are both
// empty; positions alone being non-empty let the bug pass silently).
TEST(ObjNegativeIndices, RelativeIndicesResolveToTheSameVerticesAsPositive) {
	std::string path = write_temp_obj(
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"f -3 -2 -1\n");

	auto mat = std::make_shared<lambertian>(color(1, 1, 1));
	auto mesh = load_obj(path, mat);
	ASSERT_NE(mesh, nullptr);

	// A single hit test: a ray through the triangle's centroid-ish point
	// should hit if the face actually has 3 valid vertices (as it would if
	// resolved the same way as the equivalent positive-index face `f 1 2 3`
	// would). Before the fix, fv.p ended up negative for every vertex, the
	// `if (fv.p >= 0)` filter dropped all three, fverts stayed empty, and
	// the fan-triangulation loop (`i=1; i+1<fverts.size()`) never executed
	// -- this mesh would contain zero triangles and this ray would miss.
	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record rec;
	EXPECT_TRUE(mesh->hit(r, interval(0.001, infinity), rec))
		<< "Negative-index face vertices were dropped -- mesh has no geometry to hit";

	std::remove(path.c_str());
}

// Mixed positive/negative in the same face (some OBJ exporters do this) --
// both should resolve to the same absolute vertex.
TEST(ObjNegativeIndices, MixedPositiveAndNegativeIndicesAgree) {
	std::string pathPos = write_temp_obj(
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"f 1 2 3\n");
	std::string pathMixed = write_temp_obj(
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"f 1 -2 3\n");   // vertex 2 referenced as -2 instead of 2

	auto mat = std::make_shared<lambertian>(color(1, 1, 1));
	auto meshPos = load_obj(pathPos, mat);
	auto meshMixed = load_obj(pathMixed, mat);

	ray r(point3(0.2, 0.2, -5), vec3(0, 0, 1));
	hit_record recPos, recMixed;
	bool hitPos = meshPos->hit(r, interval(0.001, infinity), recPos);
	bool hitMixed = meshMixed->hit(r, interval(0.001, infinity), recMixed);

	ASSERT_TRUE(hitPos);
	ASSERT_TRUE(hitMixed);
	EXPECT_NEAR(recPos.t, recMixed.t, 1e-9);

	std::remove(pathPos.c_str());
	std::remove(pathMixed.c_str());
}
