// mesh_preview_tests.cpp - src/shared/mesh_preview.h: the bounding box and vertex sample the Scene Builder's 3D view draws for a mesh file.
#include <gtest/gtest.h>

#include "../../src/shared/mesh_preview.h"

#include <cstdio>
#include <fstream>

namespace {
std::string writeFile(const char* name, const std::string& text) {
	std::ofstream out(name, std::ios::binary);
	out << text;
	return name;
}
}  // namespace

TEST(MeshPreviewTest, AnObjGivesItsBoundsAndSamples) {
	const std::string path = writeFile("mesh_preview_test.obj",
	                                   "# a box 2 x 4 x 6 not centred on the origin\n"
	                                   "v 1 2 3\nv 3 2 3\nv 3 6 3\nv 1 6 3\nv 1 2 9\nv 3 2 9\nv 3 6 9\nv 1 6 9\n"
	                                   "f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\n");
	const mesh_preview::MeshPreview m = mesh_preview::load(path);
	std::remove(path.c_str());
	ASSERT_TRUE(m.ok) << m.error;
	EXPECT_DOUBLE_EQ(m.lo[0], 1); EXPECT_DOUBLE_EQ(m.hi[0], 3);
	EXPECT_DOUBLE_EQ(m.lo[1], 2); EXPECT_DOUBLE_EQ(m.hi[1], 6);
	EXPECT_DOUBLE_EQ(m.lo[2], 3); EXPECT_DOUBLE_EQ(m.hi[2], 9);
	EXPECT_EQ(m.vertexCount, 8u);
	EXPECT_EQ(m.triangleCount, 4u);
	EXPECT_EQ(m.samples.size(), 8u * 3u) << "fewer vertices than the sample limit: all of them";
}

TEST(MeshPreviewTest, ASampleIsCappedAndSpreadThroughTheFile) {
	std::string obj;
	for (int i = 0; i < 1000; ++i) obj += "v " + std::to_string(i) + " 0 0\n";
	obj += "f 1 2 3\n";
	const std::string path = writeFile("mesh_preview_test_big.obj", obj);
	const mesh_preview::MeshPreview m = mesh_preview::load(path, 10);
	std::remove(path.c_str());
	ASSERT_TRUE(m.ok) << m.error;
	ASSERT_EQ(m.samples.size(), 30u);
	EXPECT_FLOAT_EQ(m.samples[0], 0.0f);
	EXPECT_GT(m.samples[27], 800.0f) << "the last sample is from near the end of the file, not the start";
	EXPECT_DOUBLE_EQ(m.hi[0], 999);
}

TEST(MeshPreviewTest, AnAsciiPlyWorksToo) {
	const std::string path = writeFile("mesh_preview_test.ply",
	                                   "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
	                                   "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
	                                   "-1 0 -1\n1 0 -1\n1 3 1\n-1 3 1\n3 0 1 2\n3 0 2 3\n");
	const mesh_preview::MeshPreview m = mesh_preview::load(path);
	std::remove(path.c_str());
	ASSERT_TRUE(m.ok) << m.error;
	EXPECT_DOUBLE_EQ(m.lo[0], -1); EXPECT_DOUBLE_EQ(m.hi[0], 1);
	EXPECT_DOUBLE_EQ(m.hi[1], 3);
	EXPECT_EQ(m.triangleCount, 2u);
}

TEST(MeshPreviewTest, ProblemsAreReportedNotThrown) {
	EXPECT_FALSE(mesh_preview::load("mesh_preview_no_such_file.obj").ok);
	EXPECT_FALSE(mesh_preview::load("mesh_preview_no_such_file.obj").error.empty());
	const std::string empty = writeFile("mesh_preview_test_empty.obj", "# nothing here\n");
	const mesh_preview::MeshPreview e = mesh_preview::load(empty);
	std::remove(empty.c_str());
	EXPECT_FALSE(e.ok);
	const std::string big = writeFile("mesh_preview_test_limit.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
	const mesh_preview::MeshPreview tooBig = mesh_preview::load(big, 100, 10);  // a 10-byte limit
	std::remove(big.c_str());
	EXPECT_FALSE(tooBig.ok);
	EXPECT_NE(tooBig.error.find("too big"), std::string::npos);
}

TEST(MeshPreviewTest, ACorruptVertexDoesNotPoisonTheBounds) {
	// the FIRST vertex is NaN (it used to seed the box) and a later one is infinite: both are left out
	const std::string path = writeFile("mesh_preview_test_nan.obj", "v nan 0 0\nv 1 2 3\nv 3 4 5\nv inf 0 0\nv 2 3 4\nf 2 3 5\n");
	const mesh_preview::MeshPreview m = mesh_preview::load(path);
	std::remove(path.c_str());
	ASSERT_TRUE(m.ok) << m.error;
	for (int a = 0; a < 3; ++a) {
		EXPECT_TRUE(std::isfinite(m.lo[a]) && std::isfinite(m.hi[a])) << a;
		EXPECT_DOUBLE_EQ(m.lo[a], 1.0 + a);
		EXPECT_DOUBLE_EQ(m.hi[a], 3.0 + a);
	}
	EXPECT_EQ(m.samples.size(), 9u) << "the three usable vertices";
	for (float v : m.samples) EXPECT_TRUE(std::isfinite(v));
}

TEST(MeshPreviewTest, AFileWithNoUsableVertexIsAnErrorNotABoxOfNaN) {
	const std::string path = writeFile("mesh_preview_test_allnan.obj", "v nan nan nan\nv inf 0 0\nv 0 0 0\nf 1 2 3\n");
	// the third vertex is fine, so make it bad too
	{ std::ofstream out(path, std::ios::binary); out << "v nan nan nan\nv inf 0 0\nv -inf 1 1\nf 1 2 3\n"; }
	const mesh_preview::MeshPreview m = mesh_preview::load(path);
	std::remove(path.c_str());
	EXPECT_FALSE(m.ok);
	EXPECT_FALSE(m.error.empty());
}
