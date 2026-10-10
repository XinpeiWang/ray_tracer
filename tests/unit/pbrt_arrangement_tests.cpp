/**
 * @file pbrt_arrangement_tests.cpp
 * @brief The Live Preview's "Save arrangement": the saved text, flattened, must put every primitive exactly where moving the flattened scene by the same offsets does.
 */

#include <gtest/gtest.h>

#include "pbrt_arrangement.h"
#include "scene_document.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace pbrt_arrangement;

namespace {

pbrt_flatten::FlatScene flattenText(const std::string& text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene, {});
}

// Every coordinate of every primitive, in one list, in a fixed order.
std::vector<double> allCoordinates(const pbrt_flatten::FlatScene& s) {
	std::vector<double> v;
	for (const auto& t : s.triangles) v.insert(v.end(), t.v, t.v + 9);
	for (const auto& sp : s.spheres) { v.insert(v.end(), sp.center, sp.center + 3); v.push_back(sp.radius); }
	for (const auto& d : s.disks) v.insert(v.end(), d.xform, d.xform + 16);
	for (const auto& c : s.cylinders) v.insert(v.end(), c.xform, c.xform + 16);
	return v;
}

// Moves object `moved` of `original` by `d` both ways and compares: (a) the flattened scene translated in place, (b) the saved text flattened afresh.
void expectSavedMatchesTranslated(const std::string& original, const std::vector<std::array<double, 3>>& offsets, int expectedMoved) {
	pbrt_flatten::FlatScene a = flattenText(original);
	const live_objects::ObjectList objects = live_objects::objectsOf(a);
	ASSERT_EQ(objects.size(), offsets.size());
	const pbrt_flatten::FlatScene untouched = a;
	for (std::size_t o = 0; o < objects.size(); ++o) live_objects::translateObject(a, objects, o, offsets[o].data());
	const Result saved = write(original, a.shapeRanges, objects, offsets, "", [](const std::string&) { return false; }, "# saved\n");
	EXPECT_EQ(saved.movedShapes, expectedMoved);
	const pbrt_flatten::FlatScene b = flattenText(saved.text);
	const std::vector<double> want = allCoordinates(a), got = allCoordinates(b);
	ASSERT_EQ(want.size(), got.size()) << saved.text;
	bool anyMoved = false;
	const std::vector<double> before = allCoordinates(untouched);
	for (std::size_t i = 0; i < want.size(); ++i) {
		EXPECT_NEAR(got[i], want[i], 1e-9) << "coordinate " << i << " of\n" << saved.text;
		if (std::fabs(want[i] - before[i]) > 1e-9) anyMoved = true;
	}
	EXPECT_TRUE(anyMoved) << "the test moved nothing";
}

}  // namespace

TEST(PbrtArrangementTest, ShapesOutsideAnyBlockMoveInWorldSpace) {
	expectSavedMatchesTranslated(
		"WorldBegin\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ] \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n",
		{{0.0, 0.0, 0.0}, {2.0, -3.0, 4.5}}, 1);
}

TEST(PbrtArrangementTest, AGlobalScaleAndRotationBeforeTheShapeDoNotChangeTheWorldMove) {
	// The Translate is written in the shape's own frame: here scaled by 2 and turned 90 degrees about y, so the written numbers differ from the world offset.
	expectSavedMatchesTranslated(
		"WorldBegin\n"
		"Scale 2 2 2\n"
		"Rotate 90 0 1 0\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"Translate 3 0 0\n"
		"Shape \"disk\" \"float radius\" [ 1 ]\n",
		{{1.0, 2.0, 3.0}, {-1.5, 0.0, 2.0}}, 2);
}

TEST(PbrtArrangementTest, ABlockMovesAsOneObjectWithItsOwnTransform) {
	expectSavedMatchesTranslated(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  Translate 5 0 0\n  Rotate 30 0 1 0\n  Scale 1 2 1\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ] \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  Shape \"cylinder\" \"float radius\" [ 0.5 ]\n"
		"  AttributeBegin\n    Translate 0 0 3\n    Shape \"sphere\" \"float radius\" [ 1 ]\n  AttributeEnd\n"
		"AttributeEnd\n",
		{{0.5, 0.25, -2.0}, {4.0, 0.0, 0.0}}, 3);
}

TEST(PbrtArrangementTest, TheRestOfTheFileIsKeptByteForByte) {
	const std::string original =
		"# a comment\nWorldBegin\nMaterial \"diffuse\" \"rgb reflectance\" [ 0.5 0.5 0.5 ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]   # trailing\nShape \"sphere\" \"float radius\" [ 2 ]\n";
	const pbrt_flatten::FlatScene s = flattenText(original);
	const live_objects::ObjectList objects = live_objects::objectsOf(s);
	const Result r = write(original, s.shapeRanges, objects, {{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, "", [](const std::string&) { return false; }, "");
	// Removing what was inserted gives the original back.
	std::string stripped = r.text;
	const std::size_t openAt = stripped.find("\nAttributeBegin");
	stripped.erase(openAt, stripped.find("Shape \"sphere\" \"float radius\" [ 1 ]") - openAt);
	stripped.erase(stripped.find("\nAttributeEnd\n"), std::string("\nAttributeEnd\n").size());
	EXPECT_EQ(stripped, original);
}

TEST(PbrtArrangementTest, RelativeFileNamesBecomeAbsoluteWhenTheFileExists) {
	const std::string original =
		"WorldBegin\n"
		"Texture \"t\" \"spectrum\" \"imagemap\" \"string filename\" \"textures/wood.png\"\n"
		"Shape \"plymesh\" \"string filename\" \"meshes/bunny.ply\"\n"
		"Shape \"sphere\" \"string note\" \"textures/missing.png\" \"float radius\" [ 1 ]\n";
	const pbrt_flatten::FlatScene s = flattenText(original);
	const live_objects::ObjectList objects = live_objects::objectsOf(s);
	const Result r = write(original, s.shapeRanges, objects, std::vector<std::array<double, 3>>(objects.size()), "/scenes/",
	                       [](const std::string& p) { return p == "/scenes/textures/wood.png" || p == "/scenes/meshes/bunny.ply"; }, "");
	EXPECT_EQ(r.rewrittenPaths, 2);
	EXPECT_NE(r.text.find("\"/scenes/textures/wood.png\""), std::string::npos);
	EXPECT_NE(r.text.find("\"/scenes/meshes/bunny.ply\""), std::string::npos);
	EXPECT_NE(r.text.find("\"textures/missing.png\""), std::string::npos) << "a name that is not a file is left alone";
}

TEST(PbrtArrangementTest, AnObjectThatDidNotMoveIsNotTouched) {
	const std::string original = "WorldBegin\nShape \"sphere\" \"float radius\" [ 1 ]\n";
	const pbrt_flatten::FlatScene s = flattenText(original);
	const Result r = write(original, s.shapeRanges, live_objects::objectsOf(s), {{0.0, 0.0, 0.0}}, "", [](const std::string&) { return false; }, "");
	EXPECT_EQ(r.text, original);
	EXPECT_EQ(r.movedShapes, 0);
}

TEST(PbrtArrangementTest, AdjacentShapesWithNoWhitespaceBetweenThemStillNest) {
	// "]Shape": the first Shape's last token ends exactly where the next begins.
	expectSavedMatchesTranslated(
		"WorldBegin\nShape \"sphere\" \"float radius\" [ 1 ]Shape \"sphere\" \"float radius\" [ 2 ]\n",
		{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}}, 2);
}

// A Scene Builder scene: its embedded document is moved too, and the directives of the saved file put every primitive where regenerating the file from the moved
// document does.
TEST(PbrtArrangementTest, ASceneBuilderSceneKeepsItsDocumentInStepWithTheDirectives) {
	scene_doc::Document doc = scene_doc::makeStarterScene();
	ASSERT_GE(doc.objects.size(), 3u);
	const std::string original = scene_doc::toPbrt(doc);
	const pbrt_flatten::FlatScene flat = flattenText(original);
	const live_objects::ObjectList objects = live_objects::objectsOf(flat);
	ASSERT_EQ(objects.size(), doc.objects.size()) << "the Builder writes one block per object";
	std::vector<std::array<double, 3>> offsets(objects.size(), {0.0, 0.0, 0.0});
	offsets[1] = {1.25, 0.0, -0.5};
	offsets[2] = {-0.75, 0.5, 2.0};
	const Result saved = write(original, flat.shapeRanges, objects, offsets, "", [](const std::string&) { return false; }, "");
	EXPECT_TRUE(saved.builderDocumentUpdated);
	EXPECT_FALSE(saved.builderDocumentDropped);

	scene_doc::Document reread;
	std::string err;
	ASSERT_TRUE(scene_doc::fromPbrt(saved.text, reread, err)) << err;
	ASSERT_EQ(reread.objects.size(), doc.objects.size());
	for (std::size_t i = 0; i < doc.objects.size(); ++i) {
		EXPECT_NEAR(reread.objects[i].position.x, doc.objects[i].position.x + offsets[i][0], 1e-9) << "object " << i;
		EXPECT_NEAR(reread.objects[i].position.y, doc.objects[i].position.y + offsets[i][1], 1e-9);
		EXPECT_NEAR(reread.objects[i].position.z, doc.objects[i].position.z + offsets[i][2], 1e-9);
	}
	// The saved directives and a file regenerated from the moved document draw the same scene.
	scene_doc::Document moved = doc;
	for (std::size_t i = 0; i < moved.objects.size(); ++i) {
		moved.objects[i].position.x += offsets[i][0];
		moved.objects[i].position.y += offsets[i][1];
		moved.objects[i].position.z += offsets[i][2];
	}
	const std::vector<double> fromDirectives = allCoordinates(flattenText(saved.text)), fromDocument = allCoordinates(flattenText(scene_doc::toPbrt(moved)));
	ASSERT_EQ(fromDirectives.size(), fromDocument.size());
	for (std::size_t i = 0; i < fromDocument.size(); ++i) EXPECT_NEAR(fromDirectives[i], fromDocument[i], 1e-6) << "coordinate " << i;
}

TEST(PbrtArrangementTest, ABuilderDocumentThatDoesNotLineUpIsRemovedSoTheBuilderDoesNotOpenAStaleCopy) {
	scene_doc::Document doc = scene_doc::makeStarterScene();
	std::string text = scene_doc::toPbrt(doc);
	// A scene edited by hand after the Builder wrote it: one more shape than the document knows.
	text += "\nShape \"sphere\" \"float radius\" [ 0.2 ]\n";
	const pbrt_flatten::FlatScene flat = flattenText(text);
	const live_objects::ObjectList objects = live_objects::objectsOf(flat);
	ASSERT_NE(objects.size(), doc.objects.size());
	std::vector<std::array<double, 3>> offsets(objects.size(), {0.0, 0.0, 0.0});
	offsets[1] = {1.0, 0.0, 0.0};
	const Result saved = write(text, flat.shapeRanges, objects, offsets, "", [](const std::string&) { return false; }, "");
	EXPECT_TRUE(saved.builderDocumentDropped);
	scene_doc::Document reread;
	std::string err;
	EXPECT_FALSE(scene_doc::fromPbrt(saved.text, reread, err)) << "the Builder must not open a copy whose document is stale";
	EXPECT_EQ(saved.text.find("@rt-builder-doc"), std::string::npos);
}
