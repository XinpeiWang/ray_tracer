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
		std::filesystem::create_directories(root_ + "geometry", ec);
	}
	void TearDown() override {
		std::error_code ec;
		std::filesystem::remove_all(root_, ec);
	}
	void write(const std::string &relative, const std::string &contents) {
		std::ofstream out(root_ + relative, std::ios::binary);
		out << contents;
	}
	std::string root_;
};

} // namespace

TEST_F(ScanTree, ListsPbrtFilesInSortedOrder) {
	// Sorted, not directory order: a scene's id is derived from its position,
	// and an id that moves when an unrelated file is added would invalidate
	// saved settings and any script that passes a scene number.
	write("zebra.pbrt", "Film \"rgb\"\n");
	write("apple.pbrt", "Film \"rgb\"\n");
	write("mango.pbrt", "Film \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 3u);
	EXPECT_EQ(found[0].name, "apple");
	EXPECT_EQ(found[1].name, "mango");
	EXPECT_EQ(found[2].name, "zebra");
}

TEST_F(ScanTree, IgnoresNonPbrtFilesAndSubdirectories) {
	// Real scenes keep their includes in geometry/, and those are fragments
	// rather than scenes - listing them would offer the user a dozen entries
	// that each render a fraction of one picture.
	write("scene.pbrt", "Film \"rgb\"\n");
	write("notes.txt", "hello");
	write("mesh.ply", "ply");
	write("geometry/part.pbrt", "Film \"rgb\"\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].name, "scene");
}

TEST_F(ScanTree, ExtensionMatchIsCaseInsensitive) {
	write("Shouty.PBRT", "Film \"rgb\"\n");
	EXPECT_EQ(pbrt_discover::scanDirectory(root_).size(), 1u);
}

TEST_F(ScanTree, AnUnparseableFileIsStillListedSoTheCallerCanReportIt) {
	// Silently dropping it would leave a user staring at a scene folder
	// wondering why one file never appears. The caller decides what to do;
	// this layer just refuses to hide the problem.
	write("good.pbrt", "Film \"rgb\"\n");
	write("broken.pbrt", "Translate 1 2\nWorldBegin\n");

	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 2u);
	EXPECT_FALSE(found[0].ok) << "broken.pbrt sorts first";
	EXPECT_TRUE(found[1].ok);
}

TEST_F(ScanTree, DiscoveredPathIsUsableToLoadTheFileLater) {
	// The path is the handle the registry keeps for the deferred full load,
	// so it has to be openable, not just descriptive.
	write("scene.pbrt", "Film \"rgb\"\n");
	const std::vector<pbrt_discover::Discovered> found =
		pbrt_discover::scanDirectory(root_);
	ASSERT_EQ(found.size(), 1u);
	std::ifstream in(found[0].path, std::ios::binary);
	EXPECT_TRUE(in.good()) << "cannot reopen " << found[0].path;
}
