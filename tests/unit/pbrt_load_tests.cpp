/**
 * @file pbrt_load_tests.cpp
 * @brief Unit tests for loading a .pbrt file and its references from disk
 *
 * Unlike the layers below it, this one is ABOUT the filesystem, so these tests
 * write real files to a temporary directory. The behaviour under test is
 * specifically that references resolve relative to the scene file rather than
 * the working directory - which cannot be tested with in-memory data, because
 * that is exactly the part in-memory resolvers stand in for.
 */

#include <gtest/gtest.h>

#include "pbrt_load.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

// A scratch directory that is created fresh and removed afterwards, so the
// tests cannot pass by accidentally finding a file left by a previous run.
class TempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_load_tests/";
		makeDir(root_);
		makeDir(root_ + "geometry/");
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
	static void makeDir(const std::string &d) {
	#ifdef _WIN32
		std::string cmd = "if not exist \"" + d + "\" mkdir \"" + d + "\" >nul 2>&1";
		for (char &c : cmd) if (c == '/') c = '\\';
		std::system(cmd.c_str());
	#else
		std::system(("mkdir -p '" + d + "'").c_str());
	#endif
	}
	std::string root_;
	std::vector<std::string> written_;
};

} // namespace

TEST_F(TempTree, LoadsASelfContainedScene) {
	write("simple.pbrt",
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("simple.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.triangles.size(), 1u);
}

TEST_F(TempTree, IncludeResolvesRelativeToTheSceneNotTheWorkingDirectory) {
	// The shape this proves: the test process's working directory is the build
	// output folder, nowhere near these files, so a cwd-relative resolver
	// cannot possibly find geometry/tri.pbrt.
	write("scene.pbrt", "Include \"geometry/tri.pbrt\"\n");
	write("geometry/tri.pbrt",
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.triangles.size(), 1u);
}

TEST_F(TempTree, NestedIncludeResolvesRelativeToTheIncludingFile) {
	// geometry/a.pbrt names its neighbour by bare filename, which is how real
	// scenes are laid out. Resolving that against the top-level scene would
	// look for ./b.pbrt and miss.
	write("scene.pbrt", "Include \"geometry/a.pbrt\"\n");
	write("geometry/a.pbrt", "Include \"b.pbrt\"\n");
	write("geometry/b.pbrt",
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.triangles.size(), 1u);
}

TEST_F(TempTree, PlyMeshResolvesRelativeToTheScene) {
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/tri.ply\" ]\n");
	write("geometry/tri.ply",
		  "ply\nformat ascii 1.0\n"
		  "element vertex 3\n"
		  "property float x\nproperty float y\nproperty float z\n"
		  "element face 1\nproperty list uchar int vertex_indices\n"
		  "end_header\n"
		  "0 0 0\n1 0 0\n0 1 0\n"
		  "3 0 1 2\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.triangles.size(), 1u);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].v[3], 1.0);
}

TEST_F(TempTree, SceneTransformsStillApplyToAnIncludedPlyMesh) {
	// Proves the whole chain composes: load -> include -> ply -> transform.
	write("scene.pbrt",
		  "Translate 100 0 0\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/tri.ply\" ]\n");
	write("geometry/tri.ply",
		  "ply\nformat ascii 1.0\n"
		  "element vertex 3\n"
		  "property float x\nproperty float y\nproperty float z\n"
		  "element face 1\nproperty list uchar int vertex_indices\n"
		  "end_header\n"
		  "0 0 0\n1 0 0\n0 1 0\n"
		  "3 0 1 2\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.triangles.size(), 1u);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].v[0], 100.0);
}

TEST_F(TempTree, DiffuseReflectanceImagemapResolvesRelativeToTheScene) {
	// Material::textureFilename's own comment: pbrt_load.h resolves it the
	// same scene-directory-first way as measuredFilename/plymesh/lensfile -
	// content doesn't matter here (mipmap_texture decodes it later, in
	// pbrt_cpu_builder.h), only that resolveExistingPath finds it.
	write("scene.pbrt",
		  "Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"geometry/t.png\" ]\n"
		  "Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	write("geometry/t.png", "not a real png, only existence is checked here");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_EQ(r.scene.materials[0].textureFilename, path("geometry/t.png"));
}

TEST_F(TempTree, MissingReflectanceImagemapWarnsAndFallsBackToConstantColour) {
	write("scene.pbrt",
		  "Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"nope.png\" ]\n"
		  "Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_TRUE(r.scene.materials[0].textureFilename.empty());
	bool warned = false;
	for (const pbrt_scene::Warning &w : r.scene.warnings)
		if (w.message.find("nope.png") != std::string::npos) warned = true;
	EXPECT_TRUE(warned);
}

TEST_F(TempTree, MissingSceneFileIsNamed) {
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("nope.pbrt"));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("nope.pbrt"), std::string::npos) << r.error;
}

TEST_F(TempTree, MissingIncludeNamesBothTheSceneAndTheMissingFile) {
	write("scene.pbrt", "Include \"geometry/gone.pbrt\"\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("gone.pbrt"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("scene.pbrt"), std::string::npos)
		<< "the error should say which scene was being loaded: " << r.error;
}

TEST_F(TempTree, MissingPlyMeshWarnsRatherThanFailingTheWholeScene) {
	// A missing mesh loses one object; a missing include loses everything
	// after it. They are deliberately not treated the same.
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/gone.ply\" ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.triangles.size(), 1u) << "the other shape still loads";

	bool warned = false;
	for (const pbrt_scene::Warning &w : r.scene.warnings)
		if (w.message.find("gone.ply") != std::string::npos) warned = true;
	EXPECT_TRUE(warned);
}

// ---------------------------------------------------------------------------
// loadFileNear / parseLensFile - a realistic camera's lensfile. Neither
// pbrt_scene.h nor pbrt_flatten.h can read it themselves (see pbrt_load.h's
// header comment on why file access lives only here), so scene_registry.h
// resolves and parses it separately, at the point it actually builds the
// camera rather than at flatten() time.
// ---------------------------------------------------------------------------

TEST_F(TempTree, LensFileResolvesRelativeToTheScene) {
	write("dgauss.dat", "35.98738 1.21638 1.54 23.716\n");
	std::string contents;
	EXPECT_TRUE(pbrt_load::loadFileNear(path("scene.pbrt"), "dgauss.dat", contents));
	EXPECT_NE(contents.find("35.98738"), std::string::npos);
}

TEST_F(TempTree, LensFileNotFoundIsReportedAsMissing) {
	std::string contents;
	EXPECT_FALSE(pbrt_load::loadFileNear(path("scene.pbrt"), "nope.dat", contents));
}

TEST(PbrtLensFileParseTest, FourNumberRowsAreReadInOrder) {
	const std::vector<double> lens = pbrt_load::parseLensFile(
		"35.98738  1.21638  1.54  23.716\n"
		"11.69718  9.9957   1.0   17.996\n");
	ASSERT_EQ(lens.size(), 8u);
	EXPECT_DOUBLE_EQ(lens[0], 35.98738);
	EXPECT_DOUBLE_EQ(lens[4], 11.69718);
	EXPECT_DOUBLE_EQ(lens[7], 17.996);
}

TEST(PbrtLensFileParseTest, CommentsAndBlankLinesAreIgnored) {
	const std::vector<double> lens = pbrt_load::parseLensFile(
		"# dgauss.22deg.dat - a Double-Gauss lens\n"
		"\n"
		"35.98738 1.21638 1.54 23.716  # first element\n"
		"\n"
		"11.69718 9.9957 1.0 17.996\n");
	EXPECT_EQ(lens.size(), 8u);
}

TEST(PbrtLensFileParseTest, AnApertureStopRowIsReadLikeAnyOther) {
	// radius=0, eta=0 marks the aperture stop in pbrt's own format -
	// RealisticCamera's constructor is what interprets that meaning; the
	// parser's only job is to hand every 4-number row through unchanged.
	const std::vector<double> lens = pbrt_load::parseLensFile("0.0 2.75 0.0 7.4\n");
	ASSERT_EQ(lens.size(), 4u);
	EXPECT_DOUBLE_EQ(lens[0], 0.0);
	EXPECT_DOUBLE_EQ(lens[2], 0.0);
	EXPECT_DOUBLE_EQ(lens[3], 7.4);
}

TEST(PbrtLensFileParseTest, ATrailingPartialRowIsDropped) {
	// Three numbers where a fourth was expected - malformed input, not a
	// scene worth guessing at. Dropping the partial row (rather than reading
	// it and shifting every row after it out of phase) is the safer failure.
	const std::vector<double> lens = pbrt_load::parseLensFile(
		"35.98738 1.21638 1.54 23.716\n"
		"11.69718 9.9957 1.0\n");
	EXPECT_EQ(lens.size(), 4u);
}

TEST(PbrtLensFileParseTest, EmptyTextYieldsNoRows) {
	EXPECT_TRUE(pbrt_load::parseLensFile("").empty());
}
