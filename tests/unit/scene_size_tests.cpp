// scene_size_tests.cpp - src/shared/scene_size.h and the "# @rt-size" header line: how big a scene is, so the Live Preview's keyboard step fits it.
// Run from the repository root: the bundled scenes are found by a relative path.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "../../src/shared/pbrt_discover.h"
#include "../../src/shared/scene_size.h"

namespace {

pbrt_flatten::Triangle triangle(double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2) {
	pbrt_flatten::Triangle t;
	const double v[9] = {x0, y0, z0, x1, y1, z1, x2, y2, z2};
	for (int i = 0; i < 9; ++i) t.v[i] = v[i];
	return t;
}

pbrt_flatten::Sphere sphere(double x, double y, double z, double r) {
	pbrt_flatten::Sphere s;
	s.center[0] = x; s.center[1] = y; s.center[2] = z;
	s.radius = r;
	return s;
}

}  // namespace

TEST(SceneSizeTest, ASceneIsAsBigAsTheLargestSideOfItsBox) {
	pbrt_flatten::FlatScene s;
	s.triangles.push_back(triangle(0, 0, 0, 555, 0, 0, 0, 100, 40));
	EXPECT_NEAR(scene_size::of(s), 555.0, 1e-3);
	pbrt_flatten::FlatScene balls;
	balls.spheres = {sphere(0, 0, 0, 1), sphere(10, 0, 0, 1)};
	EXPECT_NEAR(scene_size::of(balls), 12.0, 1e-4) << "from x = -1 to x = 11";
}

TEST(SceneSizeTest, AnEmptySceneHasNoSize) {
	EXPECT_EQ(scene_size::of(pbrt_flatten::FlatScene{}), 0.0);
}

TEST(SceneSizeTest, AHugeGroundSphereDoesNotMakeEverythingOnItTiny) {
	pbrt_flatten::FlatScene s;
	s.spheres = {sphere(0, 1, 0, 1), sphere(0, -1000, 0, 1000)};   // the radius-1000 ground idiom
	EXPECT_NEAR(scene_size::of(s), 2.0, 1e-3);
}

TEST(SceneSizeTest, ConesAndPatchesCountThroughTheirTessellation) {
	pbrt_flatten::FlatScene s;
	pbrt_flatten::Cone cone;
	cone.radius = 1.0;
	cone.height = 4.0;
	s.cones.push_back(cone);
	EXPECT_NEAR(scene_size::of(s), 4.0, 0.05) << "taller than wide";
	EXPECT_EQ(s.cones.size(), 1u) << "measuring does not rewrite the caller's scene";
	EXPECT_TRUE(s.triangles.empty());
}

TEST(SceneSizeTest, RoundingKeepsThreeSignificantDigits) {
	EXPECT_DOUBLE_EQ(scene_size::rounded(555.0000001), 555.0);
	EXPECT_NEAR(scene_size::rounded(1234.5), 1230.0, 1e-9);
	EXPECT_NEAR(scene_size::rounded(0.123456), 0.123, 1e-12);
	EXPECT_NEAR(scene_size::rounded(7.0), 7.0, 1e-12);
	EXPECT_EQ(scene_size::rounded(0.0), 0.0);
	EXPECT_EQ(scene_size::rounded(-3.0), 0.0);
}

TEST(SceneSizeTest, TheCornellBoxIs555AndAFileThatCannotBeLoadedIsNegative) {
	const std::string path = "pbrt_scenes/cornell-box-native.pbrt";
	if (!std::filesystem::exists(path)) GTEST_SKIP() << "run from the repository root";
	EXPECT_NEAR(scene_size::ofFile(path), 555.0, 1e-3);
	std::string error;
	EXPECT_LT(scene_size::ofFile("pbrt_scenes/there-is-no-such-scene.pbrt", &error), 0.0);
	EXPECT_FALSE(error.empty());
}

// ---- the header line --------------------------------------------------------------------------------------------------------------

TEST(SceneSizeHeaderTest, TheSizeLineIsReadFromTheHeader) {
	const std::string text = "# @rt-category Test Scenes\n# @rt-size 555\nLookAt 0 0 5 0 0 0 0 1 0\nWorldBegin\n";
	EXPECT_DOUBLE_EQ(pbrt_discover::detail::readHeaderTags(text).sceneSize, 555.0);
	EXPECT_DOUBLE_EQ(pbrt_discover::detail::readHeaderTags("# @rt-size 0.4\r\nWorldBegin\r\n").sceneSize, 0.4);
	EXPECT_DOUBLE_EQ(pbrt_discover::detail::readHeaderTags("# @rt-size 1e3\nWorldBegin\n").sceneSize, 1000.0);
}

TEST(SceneSizeHeaderTest, ABadOrMissingSizeMeansUnknown) {
	for (const char* line : {"# @rt-size\n", "# @rt-size none\n", "# @rt-size -4\n", "# @rt-size 0\n", "# @rt-size nan\n", "# @rt-size inf\n"}) {
		const std::string text = std::string(line) + "WorldBegin\n";
		EXPECT_EQ(pbrt_discover::detail::readHeaderTags(text).sceneSize, 0.0) << line;
	}
	EXPECT_EQ(pbrt_discover::detail::readHeaderTags("LookAt 0 0 5 0 0 0 0 1 0\nWorldBegin\n").sceneSize, 0.0);
	EXPECT_EQ(pbrt_discover::detail::readHeaderTags("WorldBegin\n# @rt-size 9\n").sceneSize, 0.0) << "only the header before WorldBegin is read";
}

TEST(SceneSizeHeaderTest, ASceneFilesSizeIsReadFromItsStartOnly) {
	const char* tmp = std::getenv("TEMP");
	if (!tmp) tmp = std::getenv("TMPDIR");
	const std::filesystem::path dir = std::filesystem::path(tmp ? tmp : ".") / "scene_size_header_test";
	std::filesystem::create_directories(dir);
	const std::string file = (dir / "sized.pbrt").string();
	{
		std::ofstream out(file, std::ios::binary);
		out << "# @rt-size 42.5\nLookAt 0 0 5 0 0 0 0 1 0\nCamera \"perspective\"\nWorldBegin\n";
		for (int i = 0; i < 100000; ++i) out << "# padding so the file is much longer than the part that is read\n";
	}
	EXPECT_DOUBLE_EQ(pbrt_discover::detail::sceneSizeOfFile(file), 42.5);
	EXPECT_EQ(pbrt_discover::detail::sceneSizeOfFile((dir / "missing.pbrt").string()), 0.0);
	std::filesystem::remove_all(dir);
}

// A new bundled scene should carry its size: scripts/stamp_scene_sizes.py adds the line. A handful of scenes cannot be measured (nothing but disks and
// cylinders, for one), so a few without it are fine.
TEST(SceneSizeBundledTest, BundledScenesCarryTheirSize) {
	if (!std::filesystem::exists("pbrt_scenes/cornell-box-native.pbrt")) GTEST_SKIP() << "run from the repository root";
	int files = 0;
	std::string unsized;
	for (const auto& entry : std::filesystem::directory_iterator("pbrt_scenes")) {
		if (entry.path().extension() != ".pbrt") continue;
		std::ifstream in(entry.path(), std::ios::binary);
		std::string head(65536, '\0');
		in.read(&head[0], static_cast<std::streamsize>(head.size()));
		head.resize(static_cast<size_t>(in.gcount()));
		if (head.find("WorldBegin") == std::string::npos) continue;   // an included library, not a scene
		++files;
		if (pbrt_discover::detail::readHeaderTags(head).sceneSize <= 0.0) unsized += " " + entry.path().filename().string();
	}
	ASSERT_GT(files, 100);
	size_t count = 0;
	for (char c : unsized) count += (c == ' ');
	EXPECT_LE(count, 8u) << "scenes without a '# @rt-size' line (run: python scripts/stamp_scene_sizes.py):" << unsized;
}
