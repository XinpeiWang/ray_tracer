/**
 * @file pbrt_discover_tests.cpp
 * @brief Unit tests for describing .pbrt files without loading their geometry
 *
 * The value of this layer is entirely in what it DOESN'T do: it must produce a
 * usable description of a scene without reading the includes and meshes that
 * make a real scene expensive. Several of these tests therefore assert on the
 * absence of work - a header parse that succeeds even though the file names
 * an include that does not exist is the whole point, not a loophole.
 */

#include <gtest/gtest.h>

#include "pbrt_discover.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace {

const char *kHeaderAndWorld = R"PBRT(
LookAt 3 4 5   0 0 0   0 0 1
Camera "perspective" "float fov" [ 30 ]
Film "rgb" "integer xresolution" [ 800 ] "integer yresolution" [ 600 ]
Sampler "halton" "integer pixelsamples" [ 256 ]
WorldBegin
Include "geometry/does-not-exist.pbrt"
Shape "plymesh" "string filename" [ "missing.ply" ]
)PBRT";

} // namespace

TEST(PbrtDiscover, ReadsCameraAndFilmFromTheHeader) {
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/killeroo.pbrt", kHeaderAndWorld);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_EQ(d.xResolution, 800);
	EXPECT_EQ(d.yResolution, 600);
	EXPECT_EQ(d.samplesPerPixel, 256);
	EXPECT_NEAR(d.camera.lookfrom[0], 3.0, 1e-9);
	EXPECT_NEAR(d.camera.lookfrom[1], 4.0, 1e-9);
	EXPECT_NEAR(d.camera.lookfrom[2], 5.0, 1e-9);
}

TEST(PbrtDiscover, ReadsMaxComponentValueFromTheFilmDirective) {
	// Regression test for a code-review finding: no test exercised the
	// wiring one hop closer to real usage than pbrt_flatten_tests.cpp's own
	// FlatScene-level check - scene_registry.h's wire_pbrt_backed_scene()
	// reads camera::max_component_value from THIS struct (Discovered), not
	// from FlatScene directly, so a break here (e.g. forgetting to copy the
	// field in describe()) would compile and pass every parsing-level test
	// while still silently losing the real, live camera field.
	const char *kWithMaxComponentValue = R"PBRT(
LookAt 0 0 5   0 0 0   0 1 0
Camera "perspective" "float fov" [ 40 ]
Film "rgb" "float maxcomponentvalue" [ 12.5 ]
WorldBegin
)PBRT";
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/maxcomponentvalue-scene.pbrt", kWithMaxComponentValue);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_NEAR(d.maxComponentValue, 12.5, 1e-9);
}

TEST(PbrtDiscover, MaxComponentValueDefaultsToEffectivelyUnboundedWithNoDirective) {
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/plain.pbrt", kHeaderAndWorld);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_NEAR(d.maxComponentValue, 1e9, 1.0);
}

TEST(PbrtDiscover, ReadsIntegratorTypeAndMaxDepthFromTheHeader) {
	const char *kWithIntegrator = R"PBRT(
LookAt 0 0 5   0 0 0   0 1 0
Camera "perspective" "float fov" [ 40 ]
Integrator "bdpt" "integer maxdepth" [ 12 ]
WorldBegin
)PBRT";
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/bdpt-scene.pbrt", kWithIntegrator);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_EQ(d.integrator, "bdpt");
	EXPECT_EQ(d.maxDepth, 12);
}

TEST(PbrtDiscover, IntegratorWithNoParamsAtAllStillReadsItsType) {
	// The exact shape real bundled scenes use, e.g. pbrt_scenes/barcelona-
	// pavilion/pavilion-night.pbrt's bare "Integrator "bdpt"" with no
	// maxdepth override at all - a different parse path than a directive
	// with params (empty ParamList rather than one with entries).
	const char *kBareIntegrator = R"PBRT(
LookAt 0 0 5   0 0 0   0 1 0
Camera "perspective" "float fov" [ 40 ]
Integrator "bdpt"
WorldBegin
)PBRT";
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/pavilion-night.pbrt", kBareIntegrator);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_EQ(d.integrator, "bdpt");
	EXPECT_EQ(d.maxDepth, 5) << "no maxdepth override, so pbrt's own default";
}

TEST(PbrtDiscover, DefaultsToVolpathAndMaxDepth5WithNoIntegratorDirective) {
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/killeroo.pbrt", kHeaderAndWorld);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_EQ(d.integrator, "volpath")
		<< "matches pbrt_scene::Scene's own default, and what a real pbrt "
		   "reports for a file with no Integrator directive at all";
	EXPECT_EQ(d.maxDepth, 5);
}

TEST(PbrtDiscover, CarriesTheUpVectorThroughBecauseCameraConfigCannotHoldIt) {
	// A Z-up scene renders sideways if this is dropped, and CameraConfig has
	// no field for it - the registry has to route it through setup_camera.
	//
	// The value is NOT the raw LookAt up. pbrt treats that as a hint and
	// orthonormalizes it against the view direction, so the right assertions
	// are the two properties that survive: the result is perpendicular to the
	// view direction, and it points to the same side as the hint. Asserting
	// the literal (0,0,1) back would be asserting a bug.
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("s.pbrt", kHeaderAndWorld);
	ASSERT_TRUE(d.ok) << d.error;

	double fwd[3];
	for (int i = 0; i < 3; ++i) fwd[i] = d.camera.lookat[i] - d.camera.lookfrom[i];
	const double along = fwd[0] * d.camera.up[0] + fwd[1] * d.camera.up[1] +
						 fwd[2] * d.camera.up[2];
	EXPECT_NEAR(along, 0.0, 1e-9) << "up should be perpendicular to the view";
	EXPECT_GT(d.camera.up[2], 0.0)
		<< "the scene asked for Z-up, so the result must not be upside down: "
		<< d.camera.up[0] << "," << d.camera.up[1] << "," << d.camera.up[2];
}

TEST(PbrtDiscover, SurfacesActiveTransformCameraMotionBlurForSceneRegistryToConsume) {
	// scene_registry.h's wire_pbrt_backed_scene() reads exactly these fields
	// (d.camera.isAnimated/lookfrom1/lookat1/shutterOpen/shutterClose) to
	// populate CameraConfig::animated/lookfrom_t1_*/lookat_t1_*/shutter_open/
	// shutter_close - which cpu_scene_camera_is_animated_by_id() and main.cpp's
	// --video + animated-camera rejection guard both key off of. This
	// confirms the discovery layer actually surfaces them for a real
	// ActiveTransform-authored scene, the same idiom the CPU-render-level
	// FlattenCameraTest suite (pbrt_flatten_tests.cpp) exercises one layer
	// down.
	const char *kAnimatedCamera = R"PBRT(
ActiveTransform StartTime
LookAt 0 0 -5   0 0 0   0 1 0
ActiveTransform EndTime
LookAt 3 0 -5   0 0 0   0 1 0
ActiveTransform All
Camera "perspective" "float fov" [ 40 ] "float shutteropen" [ 0 ] "float shutterclose" [ 1 ]
WorldBegin
)PBRT";
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("scenes/animated-camera.pbrt", kAnimatedCamera);
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_TRUE(d.camera.isAnimated);
	EXPECT_NEAR(d.camera.lookfrom1[0], 3.0, 1e-9);
	EXPECT_NEAR(d.camera.lookfrom1[2], -5.0, 1e-9);
	EXPECT_NEAR(d.camera.shutterOpen, 0.0, 1e-9);
	EXPECT_NEAR(d.camera.shutterClose, 1.0, 1e-9);
}

TEST(PbrtDiscover, SucceedsEvenThoughTheWorldReferencesFilesThatDoNotExist) {
	// The reason this layer exists. Everything after WorldBegin is discarded,
	// so a missing include or .ply cannot make a scene un-listable, and no
	// disk access is paid for geometry nobody has asked to render yet.
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("s.pbrt", kHeaderAndWorld);
	EXPECT_TRUE(d.ok) << d.error;
}

TEST(PbrtDiscover, NameIsTheFilenameStemForBothSeparators) {
	EXPECT_EQ(pbrt_discover::describe("a/b/killeroo-simple.pbrt", "").name,
			  "killeroo-simple");
	EXPECT_EQ(pbrt_discover::describe("a\\b\\crown.pbrt", "").name, "crown");
	EXPECT_EQ(pbrt_discover::describe("bare.pbrt", "").name, "bare");
}

TEST(PbrtDiscover, DefaultsSurviveAFileWithNoCameraOrFilm) {
	// A fragment with only geometry still has to describe itself rather than
	// producing zeros that would render a degenerate camera.
	const pbrt_discover::Discovered d = pbrt_discover::describe("f.pbrt", "WorldBegin\n");
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_GT(d.xResolution, 0);
	EXPECT_GT(d.yResolution, 0);
	EXPECT_GT(d.samplesPerPixel, 0);
	EXPECT_GT(d.camera.vfov, 0.0);
}

TEST(PbrtDiscover, HeaderTruncationStopsExactlyAtWorldBegin) {
	// If the split were off, a Film directive placed after WorldBegin (illegal
	// in pbrt, but the kind of thing that appears in hand-edited files) would
	// leak into the description.
	const pbrt_discover::Discovered d = pbrt_discover::describe(
		"f.pbrt",
		"Film \"rgb\" \"integer xresolution\" [ 100 ]\n"
		"WorldBegin\n"
		"Film \"rgb\" \"integer xresolution\" [ 999 ]\n");
	ASSERT_TRUE(d.ok) << d.error;
	EXPECT_EQ(d.xResolution, 100);
}

TEST(PbrtDiscover, TheCameraSurvivesTruncationBecauseWorldBeginIsKept) {
	// A regression test for a bug this file's own tests caught. pbrt records
	// the camera transform as the CTM *at WorldBegin*, not when the Camera
	// directive is read - so cutting the text just before WorldBegin left the
	// transform at identity and reported every scene's camera as sitting at
	// the origin. Renders were unaffected (they take the full load), which is
	// what made it quiet: only the values shown and edited in the GUI were
	// wrong.
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("s.pbrt", kHeaderAndWorld);
	ASSERT_TRUE(d.ok) << d.error;
	const double distance = std::sqrt(d.camera.lookfrom[0] * d.camera.lookfrom[0] +
									 d.camera.lookfrom[1] * d.camera.lookfrom[1] +
									 d.camera.lookfrom[2] * d.camera.lookfrom[2]);
	EXPECT_GT(distance, 1.0)
		<< "the camera collapsed to the origin, so the transform was lost";
}

TEST(PbrtDiscover, MalformedHeaderIsReportedRatherThanThrown) {
	// Translate with two arguments instead of three - fatal in the parser,
	// because silently ignoring it would move everything afterwards.
	const pbrt_discover::Discovered d =
		pbrt_discover::describe("bad.pbrt", "Translate 1 2\nWorldBegin\n");
	EXPECT_FALSE(d.ok);
	EXPECT_FALSE(d.error.empty());
}

TEST(PbrtDiscover, MissingFileIsReportedWithItsPath) {
	const pbrt_discover::Discovered d =
		pbrt_discover::describeFile("no/such/scene-xyzzy.pbrt");
	EXPECT_FALSE(d.ok);
	EXPECT_NE(d.error.find("scene-xyzzy.pbrt"), std::string::npos) << d.error;
}

TEST(PbrtDiscover, ScanningAMissingDirectoryIsEmptyNotAnError) {
	EXPECT_TRUE(pbrt_discover::scanDirectory("no/such/directory/at/all").empty());
}

namespace {

// A real directory, because directory scanning is precisely the part that
// cannot be tested with in-memory data.
class ScanTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_discover_tests/";
		std::error_code ec;
		std::filesystem::create_directories(root_, ec);
	}
	void TearDown() override {
		std::error_code ec;
		std::filesystem::remove_all(root_, ec);
	}
	// Creates any missing parent directories under root_ so a test can write
	// straight into e.g. "ganesha/ganesha.pbrt" without a separate setup step.
	void write(const std::string &relative, const std::string &contents) {
		const std::filesystem::path full = root_ + relative;
		std::error_code ec;
		std::filesystem::create_directories(full.parent_path(), ec);
		std::ofstream out(full, std::ios::binary);
		out << contents;
	}
	std::string root_;
};

} // namespace

TEST_F(ScanTree, ListsPbrtFilesInSortedOrder) {
	// Sorted, not directory order: a scene's id is derived from its position,
	// and an id that moves when an unrelated file is added would invalidate
	// saved settings and any script that passes a scene number.
	write("zebra.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("apple.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("mango.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 3u);
	EXPECT_EQ(found[0].name, "apple");
	EXPECT_EQ(found[1].name, "mango");
	EXPECT_EQ(found[2].name, "zebra");
}

TEST_F(ScanTree, IgnoresNonPbrtFiles) {
	write("scene.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("notes.txt", "hello");
	write("mesh.ply", "ply");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].name, "scene");
}

TEST_F(ScanTree, DescendsOneLevelIntoSubdirectoriesToFindEachScenesOwnFolder) {
	// Published scene collections (e.g. github.com/mmp/pbrt-v4-scenes)
	// package each scene as its own folder, so a real scene one level down
	// must be found, not just ones sitting flat in the scanned directory.
	write("top.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("ganesha/ganesha.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 2u);
	EXPECT_EQ(found[0].name, "ganesha") << "sorts before top.pbrt by full path";
	EXPECT_EQ(found[1].name, "top");
}

TEST_F(ScanTree, NestedFlagDistinguishesDownloadedCollectionsFromBundledScenes) {
	// scene_registry.h reads this to set SceneDescriptor::requires_files: a
	// scene tucked into its own subdirectory (the "one folder per scene"
	// shape a downloaded collection like github.com/mmp/pbrt-v4-scenes
	// uses) needs assets beyond what ships in this repo; one sitting flat in
	// the scanned directory doesn't.
	write("top.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("ganesha/ganesha.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 2u);
	EXPECT_TRUE(found[0].nested) << "ganesha/ganesha.pbrt is one level down";
	EXPECT_FALSE(found[1].nested) << "top.pbrt sits directly in the scanned directory";
}

TEST_F(ScanTree, DoesNotDescendMoreThanOneLevel) {
	// A fragment's own nested asset folders (geometry/, textures/) must stay
	// unlisted - only the one level that separates a scene's own folder from
	// its shared parent is descended.
	write("scene/scene.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("scene/geometry/part.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].name, "scene");
}

TEST_F(ScanTree, ExcludesFilesThatParseCleanlyButNeverDeclareACamera) {
	// A materials/geometry library meant only to be Include'd from a real
	// scene file - e.g. mmp/pbrt-v4-scenes' own "materials.pbrt", which sits
	// flat alongside the real scene file(s) rather than in a subdirectory,
	// so location alone can't identify it as a fragment. Left unfiltered,
	// this parses "successfully" as a scene with zero geometry, which then
	// fails to render at all (ERR 101, no objects in the scene).
	write("materials.pbrt", "MakeNamedMaterial \"m\" \"string type\" [\"diffuse\"]\n");
	write("real-scene.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].name, "real-scene");
}

TEST_F(ScanTree, ExtensionMatchIsCaseInsensitive) {
	write("Shouty.PBRT", "Camera \"perspective\"\nFilm \"rgb\"\n");
	EXPECT_EQ(pbrt_discover::scanDirectory(root_).size(), 1u);
}

TEST_F(ScanTree, ExcludesAFileASiblingIncludesEvenWhenItFailsToParseStandalone) {
	// mmp/pbrt-v4-scenes' barcelona-pavilion ships exactly this shape:
	// geometry.pbrt has no WorldBegin of its own (so its whole body is read
	// as "header" text - see headerOf()'s fallback comment) and references
	// a NamedMaterial that materials.pbrt declares. Neither fragment is
	// meant to be loaded on its own; only real-scene.pbrt's own Include
	// order makes the reference resolvable. Being named in a sibling's
	// Include is proof enough to exclude it, independent of whether it
	// happens to parse standalone.
	write("geometry.pbrt", "NamedMaterial \"pavet\"\n");
	write("real-scene.pbrt",
		  "Camera \"perspective\"\nFilm \"rgb\"\nWorldBegin\nInclude \"geometry.pbrt\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].name, "real-scene");
}

TEST_F(ScanTree, AnUnparseableFileIsStillListedSoTheCallerCanReportIt) {
	// Silently dropping it would leave a user staring at a scene folder
	// wondering why one file never appears. The caller decides what to do;
	// this layer just refuses to hide the problem. Both declare a Camera so
	// the case under test is "fails to parse", not "filtered as a fragment".
	write("good.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	write("broken.pbrt", "Camera \"perspective\"\nTranslate 1 2\nWorldBegin\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 2u);
	EXPECT_FALSE(found[0].ok) << "broken.pbrt sorts first";
	EXPECT_TRUE(found[1].ok);
}

TEST_F(ScanTree, DiscoveredPathIsUsableToLoadTheFileLater) {
	// The path is the handle the registry keeps for the deferred full load,
	// so it has to be openable, not just descriptive.
	write("scene.pbrt", "Camera \"perspective\"\nFilm \"rgb\"\n");
	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	std::ifstream in(found[0].path, std::ios::binary);
	EXPECT_TRUE(in.good()) << "cannot reopen " << found[0].path;
}
