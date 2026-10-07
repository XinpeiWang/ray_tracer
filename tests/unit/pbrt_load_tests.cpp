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
#include "pbrt_asset_check.h"

#include <cstdio>
#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <unistd.h>
#endif
#include <filesystem>
#include <fstream>
#include <string>

namespace {

// A scratch directory that is created fresh and removed afterwards, so the
// tests cannot pass by accidentally finding a file left by a previous run.
class TempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		// Per-process directory: scripts/run_tests_parallel.ps1 runs shards of this suite concurrently.
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_load_tests_" + std::to_string(processId()) + "/";
		makeDir(root_);
		makeDir(root_ + "geometry/");
		makeDir(root_ + "textures/");
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
	static int processId() {
	#ifdef _WIN32
		return static_cast<int>(_getpid());
	#else
		return static_cast<int>(getpid());
	#endif
	}
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

TEST_F(TempTree, PlyMeshWithUvPropertiesThreadsRealUV) {
	// End-to-end: real PLY "u"/"v" vertex properties should survive the
	// whole load -> ply_mesh::parse -> MeshResolver -> flatten pipeline as
	// Triangle::uv/hasUVs, matching real pbrt-v4's own TriQuadMesh::ReadPLY
	// behavior (see docs/PBRT_SUPPORT.md).
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/tri.ply\" ]\n");
	write("geometry/tri.ply",
		  "ply\nformat ascii 1.0\n"
		  "element vertex 3\n"
		  "property float x\nproperty float y\nproperty float z\n"
		  "property float u\nproperty float v\n"
		  "element face 1\nproperty list uchar int vertex_indices\n"
		  "end_header\n"
		  "0 0 0 0 0\n1 0 0 1 0\n0 1 0 0 1\n"
		  "3 0 1 2\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.triangles.size(), 1u);
	ASSERT_TRUE(r.scene.triangles[0].hasUVs);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[0], 0.0);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[1], 0.0);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[2], 1.0);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[3], 0.0);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[4], 0.0);
	EXPECT_DOUBLE_EQ(r.scene.triangles[0].uv[5], 1.0);
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

TEST_F(TempTree, AreaLightFilenameResolvesRelativeToTheScene) {
	// Emission::filename's own comment: this needs the same scene-directory-
	// first resolution as textureFilename/alphaTextureFilename above - it
	// was added to Emission without a matching pbrt_load.h resolution pass,
	// so a relative "filename" silently failed to load whenever the working
	// directory wasn't the scene's own directory (exactly what this test's
	// working directory - the build output folder - proves).
	write("scene.pbrt",
		  "AttributeBegin\n"
		  "  AreaLightSource \"diffuse\" \"string filename\" [ \"textures/glow.png\" ]\n"
		  "  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "    \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		  "AttributeEnd\n");
	write("textures/glow.png", "not a real png, only existence is checked here");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.areaLights.size(), 1u);
	EXPECT_EQ(r.scene.areaLights[0].filename, path("textures/glow.png"));
}

TEST_F(TempTree, MissingAreaLightFilenameWarnsAndFallsBackToFlatColour) {
	write("scene.pbrt",
		  "AttributeBegin\n"
		  "  AreaLightSource \"diffuse\" \"string filename\" [ \"nope.png\" ]\n"
		  "  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "    \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		  "AttributeEnd\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.areaLights.size(), 1u);
	EXPECT_TRUE(r.scene.areaLights[0].filename.empty());
	bool warned = false;
	for (const pbrt_scene::Warning &w : r.scene.warnings)
		if (w.message.find("nope.png") != std::string::npos) warned = true;
	EXPECT_TRUE(warned);
}

TEST_F(TempTree, GoniometricLightFilenameResolvesRelativeToTheScene) {
	// PunctualLight::filename's own comment: same scene-directory-first
	// resolution as textureFilename/Emission::filename above. Uses
	// "textures/" (pre-created by SetUp(), unlike an arbitrary new
	// subdirectory) since write() itself does not create parent directories.
	write("scene.pbrt",
		  "LightSource \"goniometric\" \"string filename\" [ \"textures/lamp.exr\" ]\n");
	write("textures/lamp.exr", "not a real exr, only existence is checked here");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.punctualLights.size(), 1u);
	EXPECT_EQ(r.scene.punctualLights[0].filename, path("textures/lamp.exr"));
	EXPECT_TRUE(r.scene.punctualLights[0].hadImageFilename);
}

TEST_F(TempTree, MissingGoniometricLightFilenameWarnsAndFallsBackToIsotropic) {
	write("scene.pbrt",
		  "LightSource \"goniometric\" \"string filename\" [ \"nope.exr\" ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.punctualLights.size(), 1u);
	EXPECT_TRUE(r.scene.punctualLights[0].filename.empty());
	EXPECT_TRUE(r.scene.punctualLights[0].hadImageFilename)
		<< "still true - a name was given, just not found";
	bool warned = false;
	for (const pbrt_scene::Warning &w : r.scene.warnings)
		if (w.message.find("nope.exr") != std::string::npos) warned = true;
	EXPECT_TRUE(warned);
}

TEST_F(TempTree, ShapeAlphaImagemapResolvesRelativeToTheScene) {
	write("scene.pbrt",
		  "Texture \"leafAlpha\" \"float\" \"imagemap\" \"string filename\" [ \"geometry/leaf.png\" ]\n"
		  "Material \"diffusetransmission\"\n"
		  "Shape \"trianglemesh\" \"texture alpha\" [ \"leafAlpha\" ]\n"
		  "  \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	write("geometry/leaf.png", "not a real png, only existence is checked here");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_EQ(r.scene.materials[0].alphaTextureFilename, path("geometry/leaf.png"));
}

TEST_F(TempTree, MissingShapeAlphaImagemapWarnsAndRendersOpaque) {
	write("scene.pbrt",
		  "Texture \"leafAlpha\" \"float\" \"imagemap\" \"string filename\" [ \"nope.png\" ]\n"
		  "Material \"diffusetransmission\"\n"
		  "Shape \"trianglemesh\" \"texture alpha\" [ \"leafAlpha\" ]\n"
		  "  \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_TRUE(r.scene.materials[0].alphaTextureFilename.empty());
	bool warned = false;
	for (const pbrt_scene::Warning &w : r.scene.warnings)
		if (w.message.find("nope.png") != std::string::npos) warned = true;
	EXPECT_TRUE(warned);
}

TEST_F(TempTree, DisplacementImagemapResolvesRelativeToTheScene) {
	write("scene.pbrt",
		  "Texture \"bmap\" \"float\" \"imagemap\" \"string filename\" [ \"textures/bump.png\" ]\n"
		  "Material \"coateddiffuse\" \"texture displacement\" [ \"bmap\" ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	write("textures/bump.png", "not a real png, only existence is checked here");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_EQ(r.scene.materials[0].displacementTextureFilename, path("textures/bump.png"));
}

TEST_F(TempTree, MissingDisplacementImagemapWarnsAndRendersWithoutBumpDetail) {
	write("scene.pbrt",
		  "Texture \"bmap\" \"float\" \"imagemap\" \"string filename\" [ \"nope.png\" ]\n"
		  "Material \"coateddiffuse\" \"texture displacement\" [ \"bmap\" ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_TRUE(r.scene.materials[0].displacementTextureFilename.empty());
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

TEST_F(TempTree, SceneWhoseEveryMeshIsMissingFailsAndNamesTheFiles) {
	// A scene that skipped every shape used to load "successfully" and render an
	// empty (sky-only) picture - the K68 report. Now it is an error that says which
	// files to install, and the caller can tell it from a parse failure.
	write("scene.pbrt",
		  "LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/a.ply\" ]\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/b.ply\" ]\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/a.ply\" ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("geometry/a.ply"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("geometry/b.ply"), std::string::npos) << r.error;
	EXPECT_NE(r.error.find("scene.pbrt"), std::string::npos) << r.error;
	EXPECT_EQ(r.missingFiles.size(), 3u) << "one entry per skipped shape";
}

TEST_F(TempTree, MissingMeshErrorListIsDeduplicatedAndCapped) {
	std::string scene;
	for (int i = 0; i < 9; ++i)
		scene += "Shape \"plymesh\" \"string filename\" [ \"geometry/m" + std::to_string(i) + ".ply\" ]\n";
	write("scene.pbrt", scene);
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_FALSE(r.ok);
	EXPECT_NE(r.error.find("m0.ply"), std::string::npos) << r.error;
	EXPECT_EQ(r.error.find("m8.ply"), std::string::npos) << "only the first few are listed: " << r.error;
	EXPECT_NE(r.error.find("(+4 more)"), std::string::npos) << r.error;
}

TEST_F(TempTree, MissingMeshWithOtherGeometryStillLoads) {
	// Only the all-skipped case is an error; one surviving shape of any kind keeps
	// the lenient warn-and-continue behaviour.
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/gone.ply\" ]\n"
		  "Shape \"sphere\" \"float radius\" 1\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_TRUE(r.missingFiles.empty());
	EXPECT_EQ(r.scene.missingFiles.size(), 1u);
}

TEST_F(TempTree, SceneWithNoShapesAtAllIsNotTreatedAsMissingAssets) {
	// Nothing was skipped, there just is nothing: not this error's business.
	write("scene.pbrt", "LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	EXPECT_TRUE(r.ok) << r.error;
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

// ---- LightSource "infinite" CTM is baked into the environment image ---------------------
// Regression: commit b894923b ("windowed/portal infinite light") turned this branch into
// `else if (!sky.hasPortal) { <comment> }` and dropped the applyInfiniteLightOrientation()
// call, so a rotated/mirrored environment map was silently loaded un-rotated on every backend.

TEST_F(TempTree, InfiniteLightImageIsReorientedByItsTransform) {
	// A 4x2 Radiance .hdr (flat RGBE, no RLE: stb only compresses width >= 8): four distinct
	// pixels per row so a quarter-turn about +y visibly moves them.
	static const unsigned char kPixels[] = {
		255, 0, 0, 129,   0, 255, 0, 129,   0, 0, 255, 129,   255, 255, 255, 129,
		128, 0, 0, 129,   0, 128, 0, 129,   0, 0, 128, 129,   128, 128, 128, 129 };
	std::string hdr = R"(#?RADIANCE
FORMAT=32-bit_rle_rgbe

-Y 2 +X 4
)";
	hdr.append(reinterpret_cast<const char *>(kPixels), sizeof(kPixels));
	write("env.hdr", hdr);

	const std::string head = R"(LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective"
WorldBegin
)";
	write("identity.pbrt", head + R"(LightSource "infinite" "string filename" ["env.hdr"]
)");
	write("rotated.pbrt", head + R"(AttributeBegin
  Rotate 90 0 1 0
  LightSource "infinite" "string filename" ["env.hdr"]
AttributeEnd
)");

	const pbrt_load::LoadResult plain = pbrt_load::loadFile(path("identity.pbrt"));
	const pbrt_load::LoadResult turned = pbrt_load::loadFile(path("rotated.pbrt"));
	ASSERT_TRUE(plain.ok) << plain.error;
	ASSERT_TRUE(turned.ok) << turned.error;
	const auto &a = plain.scene.infiniteLight;
	const auto &b = turned.scene.infiniteLight;
	ASSERT_EQ(a.imageWidth, 4);
	ASSERT_EQ(a.imageHeight, 2);
	ASSERT_EQ(b.imagePixels.size(), a.imagePixels.size());
	EXPECT_NE(a.imagePixels, b.imagePixels) << "a 90 degree rotation left the environment image untouched";
}

// ---------------------------------------------------------------------------
// pbrt_asset_check - the GUI's "this scene's files are missing" warning. A text scan of the
// scene (and its Includes) for the files it refers to, using the loader's own lookups.
// ---------------------------------------------------------------------------

namespace {
bool endsWith(const std::string &s, const std::string &suffix) {
	return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

TEST_F(TempTree, AssetCheckCountsMissingAndFoundReferences) {
	write("textures/here.png", "x");
	write("scene.pbrt",
		  "Texture \"t\" \"spectrum\" \"imagemap\" \"string filename\" [ \"textures/here.png\" ]\n"
		  "Shape \"plymesh\" \"string filename\" \"geometry/gone.ply\"\n"
		  "# \"string filename\" [ \"textures/commented-out.png\" ]\n");
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("scene.pbrt"));
	ASSERT_EQ(r.missing.size(), 1u);
	EXPECT_EQ(r.missing[0], "geometry/gone.ply");
	EXPECT_EQ(r.referenced, 2) << "the commented-out line is not a reference";
	EXPECT_TRUE(endsWith(r.folder, "geometry")) << "the folder the file belongs in: " << r.folder;
}

TEST_F(TempTree, AssetCheckFollowsIncludesAndAcceptsGzippedMeshes) {
	write("geometry/part.pbrt", "Shape \"plymesh\" \"string filename\" [ \"gone.ply\" ]\n");
	write("geometry/zipped.ply.gz", "x");
	write("scene.pbrt",
		  "Include \"geometry/part.pbrt\"\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/zipped.ply\" ]\n");
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("scene.pbrt"));
	ASSERT_EQ(r.missing.size(), 1u) << "only the include's gone.ply; the .ply.gz stands in for zipped.ply";
	EXPECT_EQ(r.missing[0], "gone.ply");
}

TEST_F(TempTree, AssetCheckOnAMissingSceneFileReportsThatFile) {
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("nope.pbrt"));
	ASSERT_EQ(r.missing.size(), 1u);
	EXPECT_EQ(r.referenced, 0) << "referenced == 0 is how callers tell 'scene file absent' from 'its assets absent'";
	EXPECT_FALSE(r.folder.empty());
}

TEST_F(TempTree, AssetCheckOnACompleteSceneReportsNothing) {
	write("geometry/a.ply", "x");
	write("scene.pbrt", "Shape \"plymesh\" \"string filename\" [ \"geometry/a.ply\" ]\n");
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("scene.pbrt"));
	EXPECT_TRUE(r.missing.empty());
	EXPECT_TRUE(r.folder.empty());
}

TEST_F(TempTree, AssetCheckTreatsAGroupSuffixAsTheFileBeforeIt) {
	// "model.obj#group" is one group of an OBJ; the file existing means it is not missing.
	write("geometry/model.obj", "x");
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/model.obj#a\" ]\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/model.obj#b\" ]\n");
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("scene.pbrt"));
	EXPECT_TRUE(r.missing.empty());
	EXPECT_EQ(r.referenced, 1) << "two groups, one file";
}

TEST_F(TempTree, AssetCheckPointsAtTheFolderContainingEveryMissingFile) {
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"models/a.obj\" ]\n"
		  "Texture \"t\" \"spectrum\" \"imagemap\" \"string filename\" [ \"models/textures/b.png\" ]\n");
	const pbrt_asset_check::Result r = pbrt_asset_check::check(path("scene.pbrt"));
	ASSERT_EQ(r.missing.size(), 2u);
	EXPECT_TRUE(endsWith(r.folder, "models")) << "not the textures subfolder: " << r.folder;
}

TEST_F(TempTree, MissingMeshErrorNamesTheFileOnceNotOncePerObjGroup) {
	write("scene.pbrt",
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/model.obj#a\" ]\n"
		  "Shape \"plymesh\" \"string filename\" [ \"geometry/model.obj#b\" ]\n");
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_FALSE(r.ok);
	EXPECT_NE(r.error.find("geometry/model.obj"), std::string::npos) << r.error;
	EXPECT_EQ(r.error.find('#'), std::string::npos) << "the group suffix is not part of the file: " << r.error;
	EXPECT_EQ(r.error.find("more)"), std::string::npos) << "one file, nothing to abbreviate: " << r.error;
}

// ---------------------------------------------------------------------------
// RAY_TRACER_USER_ASSETS - the per-user folder the GUI downloads into. It stands in for the application folder, so a
// scene that names models/x.obj (relative) is satisfied by <root>/models/x.obj when nothing is next to the app.
// ---------------------------------------------------------------------------

namespace {
// setenv/unsetenv/getcwd/chdir with the Windows CRT's names, so the user-asset tests below run on every platform.
inline void testSetEnv(const char *name, const char *value) {
#ifdef _WIN32
	_putenv_s(name, value);
#else
	setenv(name, value, 1);
#endif
}
inline void testUnsetEnv(const char *name) {
#ifdef _WIN32
	_putenv_s(name, "");   // an empty value removes the variable
#else
	unsetenv(name);
#endif
}
inline const char *testGetCwd(char *buf, size_t n) {
#ifdef _WIN32
	return _getcwd(buf, static_cast<int>(n));
#else
	return getcwd(buf, n);
#endif
}
inline int testChdir(const char *dir) {
#ifdef _WIN32
	return _chdir(dir);
#else
	return chdir(dir);
#endif
}

// Sets RAY_TRACER_USER_ASSETS and the working directory for one test, and restores both.
class UserAssetEnv {
public:
	UserAssetEnv(const std::string &userRoot, const std::string &cwd) {
		const char *old = std::getenv("RAY_TRACER_USER_ASSETS");
		hadOld_ = old != nullptr;
		if (old) old_ = old;
		char buf[4096];
		oldCwd_ = testGetCwd(buf, sizeof buf) ? buf : "";
		testSetEnv("RAY_TRACER_USER_ASSETS", userRoot.c_str());
		if (testChdir(cwd.c_str()) != 0) ADD_FAILURE() << "chdir failed";
	}
	~UserAssetEnv() {
		if (!oldCwd_.empty() && testChdir(oldCwd_.c_str()) != 0) ADD_FAILURE() << "could not restore the working directory";
		if (hadOld_) testSetEnv("RAY_TRACER_USER_ASSETS", old_.c_str());
		else testUnsetEnv("RAY_TRACER_USER_ASSETS");
	}
private:
	bool hadOld_ = false;
	std::string old_, oldCwd_;
};
}  // namespace

TEST_F(TempTree, MeshFoundUnderTheUserAssetRootWhenNotNextToTheApp) {
	write("scene.pbrt", "Shape \"plymesh\" \"string filename\" [ \"models/x.obj\" ]\n");
	std::filesystem::create_directories(path("userassets/models/"));
	write("userassets/models/x.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
	UserAssetEnv env(path("userassets"), path(""));
	const pbrt_load::LoadResult r = pbrt_load::loadFile("scene.pbrt");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.triangles.size(), 1u);
}

TEST_F(TempTree, MeshStillMissingWhenTheUserAssetRootLacksIt) {
	write("scene.pbrt", "Shape \"plymesh\" \"string filename\" [ \"models/x.obj\" ]\n");
	std::filesystem::create_directories(path("userassets/"));
	UserAssetEnv env(path("userassets"), path(""));
	const pbrt_load::LoadResult r = pbrt_load::loadFile("scene.pbrt");
	EXPECT_FALSE(r.ok);
}

TEST_F(TempTree, AssetCheckCountsAFileInTheUserAssetRootAsPresent) {
	write("scene.pbrt", "Shape \"plymesh\" \"string filename\" [ \"models/x.obj\" ]\n");
	std::filesystem::create_directories(path("userassets/models/"));
	write("userassets/models/x.obj", "x");
	UserAssetEnv env(path("userassets"), path(""));
	EXPECT_TRUE(pbrt_asset_check::check("scene.pbrt").missing.empty());
}

TEST_F(TempTree, PartialMissingMeshMessageSaysPartAndNotNothing) {
	// The wording used when a scene that needs its files loads but some are absent (callers decide whether that is fatal).
	const std::string msg = pbrt_load::detail::missingMeshesMessage("scene.pbrt", {"models/a.obj#x", "models/a.obj#y"}, false);
	EXPECT_NE(msg.find("models/a.obj"), std::string::npos) << msg;
	EXPECT_NE(msg.find("missing part of the scene"), std::string::npos) << msg;
	EXPECT_EQ(msg.find("nothing to render"), std::string::npos) << msg;
}
