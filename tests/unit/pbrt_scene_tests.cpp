/**
 * @file pbrt_scene_tests.cpp
 * @brief Unit tests for the pbrt-v4 scene format subset parser
 *
 * The point of adopting pbrt's format rather than inventing one is to load
 * other people's scenes, so the tests are written against real pbrt syntax -
 * graphics-state scoping, CTM composition, typed parameter lists - rather than
 * a convenient subset of it.
 *
 * The warn-and-skip behaviour gets as much attention as the happy path. Any
 * real scene contains directives this parser does not implement, and the whole
 * design rests on those degrading to a warning instead of failing the load.
 */

#include <gtest/gtest.h>

#include "pbrt_scene.h"

#include <map>
#include <string>

using namespace pbrt_scene;

namespace {

// Convenience: parse and require success, reporting the parser's own message.
Scene parseOk(const std::string &text) {
	const ParseResult r = parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return r.scene;
}

bool hasWarningContaining(const Scene &s, const std::string &needle) {
	for (const Warning &w : s.warnings)
		if (w.message.find(needle) != std::string::npos) return true;
	return false;
}

} // namespace

// ===========================================================================
// Lexical
// ===========================================================================

TEST(PbrtTokenizerTest, CommentsAreIgnoredButStillCountLines) {
	const Scene s = parseOk(
		"# a comment\n"
		"# another\n"
		"Bogus\n"                       // unknown -> warning carrying the line
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.warnings.size(), 1u);
	EXPECT_EQ(s.warnings[0].line, 3) << "line number must survive comment skipping";
}

TEST(PbrtTokenizerTest, QuotedStringsMaySpanSpaces) {
	const Scene s = parseOk(
		"Film \"rgb\" \"string filename\" \"my output.exr\"\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.filmFilename, "my output.exr");
}

// ===========================================================================
// Parameter lists
// ===========================================================================

TEST(PbrtParamsTest, BracketedAndBareValuesBothParse) {
	const Scene s = parseOk(
		"Shape \"sphere\" \"float radius\" [ 2.5 ]\n"
		"Shape \"sphere\" \"float radius\" 3.5\n");
	ASSERT_EQ(s.shapes.size(), 2u);
	EXPECT_DOUBLE_EQ(s.shapes[0].params.getFloat("radius", -1), 2.5);
	EXPECT_DOUBLE_EQ(s.shapes[1].params.getFloat("radius", -1), 3.5);
}

TEST(PbrtParamsTest, MultiValueListsKeepEveryElement) {
	const Scene s = parseOk(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	const Param *idx = s.shapes[0].params.find("indices");
	const Param *P   = s.shapes[0].params.find("P");
	ASSERT_NE(idx, nullptr);
	ASSERT_NE(P, nullptr);
	EXPECT_EQ(idx->numbers.size(), 6u);
	EXPECT_EQ(P->numbers.size(), 12u);
	EXPECT_EQ(idx->type, "integer");
	EXPECT_EQ(P->type, "point3");
}

TEST(PbrtParamsTest, RgbAndStringAndBoolTypesAreDistinguished) {
	const Scene s = parseOk(
		"Material \"diffuse\" \"rgb reflectance\" [ .1 .2 .3 ]"
		" \"string filename\" \"tex.exr\" \"bool remaproughness\" false\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	const ParamList &p = s.materials[0].params;
	const Vec3 c = p.getVec3("reflectance", Vec3{});
	EXPECT_DOUBLE_EQ(c.x, 0.1);
	EXPECT_DOUBLE_EQ(c.z, 0.3);
	EXPECT_EQ(p.getString("filename", ""), "tex.exr");
	EXPECT_FALSE(p.getBool("remaproughness", true));
}

TEST(PbrtParamsTest, MissingParamsFallBackToTheCallersDefault) {
	const Scene s = parseOk("Shape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.shapes[0].params.getFloat("radius", 1.0), 1.0);
}

// ===========================================================================
// Graphics state
// ===========================================================================

TEST(PbrtStateTest, AttributeEndRestoresTheMaterial) {
	// The canonical example from pbrt's own file-format documentation.
	const Scene s = parseOk(
		"Material \"diffuse\"\n"
		"AttributeBegin\n"
		"  Material \"conductor\"\n"
		"  Shape \"sphere\"\n"
		"AttributeEnd\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 2u);
	ASSERT_GE(s.materials.size(), 2u);
	EXPECT_EQ(s.materials[s.shapes[0].materialIndex].type, "conductor");
	EXPECT_EQ(s.materials[s.shapes[1].materialIndex].type, "diffuse")
		<< "material must revert when the attribute scope closes";
}

TEST(PbrtStateTest, AttributeEndRestoresTheTransform) {
	const Scene s = parseOk(
		"AttributeBegin\n"
		"  Translate 10 0 0\n"
		"  Shape \"sphere\"\n"
		"AttributeEnd\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 2u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 10.0);
	EXPECT_DOUBLE_EQ(s.shapes[1].xform.m[3], 0.0);
}

TEST(PbrtStateTest, UnmatchedAttributeEndIsFatal) {
	// Silently tolerating this would let every later shape inherit the wrong
	// state, which is far harder to diagnose than a refusal.
	const ParseResult r = parse("AttributeEnd\nShape \"sphere\"\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("without a matching Begin"), std::string::npos) << r.error;
}

TEST(PbrtStateTest, AreaLightAppliesToFollowingShapesWithinScope) {
	const Scene s = parseOk(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ]\n"
		"  Shape \"sphere\"\n"
		"AttributeEnd\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 2u);
	EXPECT_EQ(s.shapes[0].areaLightIndex, 0) << "shape inside the scope is emissive";
	EXPECT_EQ(s.shapes[1].areaLightIndex, -1) << "shape outside it is not";
}

TEST(PbrtStateTest, WorldBeginCapturesTheCameraTransformAndResetsTheCtm) {
	const Scene s = parseOk(
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	// world-to-camera must be non-identity, and the world CTM must start clean
	EXPECT_NE(s.worldToCamera.m[11], 0.0);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 0.0);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[0], 1.0);
}

// ===========================================================================
// Transforms
// ===========================================================================

TEST(PbrtTransformTest, TransformsComposeInDirectiveOrder) {
	// Translate then Scale means the scale happens in the translated frame:
	// CTM = T * S, so the translation column is untouched by the scale.
	const Scene s = parseOk(
		"Translate 1 2 3\n"
		"Scale 2 2 2\n"
		"Shape \"sphere\"\n");
	const Matrix4 &m = s.shapes[0].xform;
	EXPECT_DOUBLE_EQ(m.m[0], 2.0);
	EXPECT_DOUBLE_EQ(m.m[3], 1.0);
	EXPECT_DOUBLE_EQ(m.m[7], 2.0);
	EXPECT_DOUBLE_EQ(m.m[11], 3.0);
}

TEST(PbrtTransformTest, RotateUsesDegrees) {
	const Scene s = parseOk("Rotate 90 0 0 1\nShape \"sphere\"\n");
	const Matrix4 &m = s.shapes[0].xform;
	EXPECT_NEAR(m.m[0], 0.0, 1e-9);
	EXPECT_NEAR(m.m[1], -1.0, 1e-9);
	EXPECT_NEAR(m.m[4], 1.0, 1e-9);
}

TEST(PbrtTransformTest, DegenerateRotationAxisIsIdentityNotNaN) {
	const Scene s = parseOk("Rotate 45 0 0 0\nShape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[0], 1.0);
}

TEST(PbrtTransformTest, TransformDirectiveReadsColumnMajor) {
	// pbrt writes matrices column-major; the translation therefore sits in the
	// last four numbers, not scattered through the rows.
	const Scene s = parseOk(
		"Transform [ 1 0 0 0  0 1 0 0  0 0 1 0  7 8 9 1 ]\n"
		"Shape \"sphere\"\n");
	const Matrix4 &m = s.shapes[0].xform;
	EXPECT_DOUBLE_EQ(m.m[3], 7.0);
	EXPECT_DOUBLE_EQ(m.m[7], 8.0);
	EXPECT_DOUBLE_EQ(m.m[11], 9.0);
}

TEST(PbrtTransformTest, TransformReplacesWhileConcatTransformComposes) {
	const Scene a = parseOk(
		"Translate 5 0 0\n"
		"Transform [ 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1 ]\n"
		"Shape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(a.shapes[0].xform.m[3], 0.0) << "Transform replaces the CTM";

	const Scene b = parseOk(
		"Translate 5 0 0\n"
		"ConcatTransform [ 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1 ]\n"
		"Shape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(b.shapes[0].xform.m[3], 5.0) << "ConcatTransform composes with it";
}

// ===========================================================================
// Materials and lights
// ===========================================================================

TEST(PbrtMaterialTest, NamedMaterialsResolveToTheirDeclaration) {
	const Scene s = parseOk(
		"MakeNamedMaterial \"gold\" \"string type\" \"conductor\" \"float roughness\" [ .1 ]\n"
		"NamedMaterial \"gold\"\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	ASSERT_GE(s.shapes[0].materialIndex, 0);
	const MaterialDecl &m = s.materials[s.shapes[0].materialIndex];
	EXPECT_EQ(m.name, "gold");
	EXPECT_EQ(m.type, "conductor") << "MakeNamedMaterial takes its type from the params";
	EXPECT_DOUBLE_EQ(m.params.getFloat("roughness", -1), 0.1);
}

TEST(PbrtMaterialTest, ReferencingAnUndeclaredMaterialIsFatal) {
	// Not a warning: everything after it would silently render with the wrong
	// material, which is exactly the kind of bug that survives a visual check.
	const ParseResult r = parse("NamedMaterial \"nope\"\nShape \"sphere\"\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("nope"), std::string::npos) << r.error;
}

TEST(PbrtLightTest, LightSourceAndAreaLightSourceAreKeptApart) {
	const Scene s = parseOk(
		"LightSource \"infinite\" \"string filename\" \"sky.exr\"\n"
		"AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.lights.size(), 1u);
	ASSERT_EQ(s.areaLights.size(), 1u);
	EXPECT_EQ(s.lights[0].type, "infinite");
	EXPECT_EQ(s.lights[0].params.getString("filename", ""), "sky.exr");
	EXPECT_EQ(s.areaLights[0].type, "diffuse");
}

// ===========================================================================
// Pre-world settings
// ===========================================================================

TEST(PbrtSettingsTest, CameraFilmSamplerAndIntegratorAreRead) {
	const Scene s = parseOk(
		"Camera \"perspective\" \"float fov\" 45\n"
		"Sampler \"halton\" \"integer pixelsamples\" [ 128 ]\n"
		"Film \"rgb\" \"integer xresolution\" [ 800 ] \"integer yresolution\" [ 600 ]\n"
		"Integrator \"volpath\" \"integer maxdepth\" [ 12 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.cameraType, "perspective");
	EXPECT_DOUBLE_EQ(s.cameraFov(), 45.0);
	EXPECT_EQ(s.samplesPerPixel, 128);
	EXPECT_EQ(s.xResolution, 800);
	EXPECT_EQ(s.yResolution, 600);
	EXPECT_EQ(s.integrator, "volpath");
	EXPECT_EQ(s.maxDepth, 12);
}

// ===========================================================================
// Unsupported directives degrade rather than fail
// ===========================================================================

TEST(PbrtSkipTest, UnknownDirectiveWarnsAndDoesNotFail) {
	const ParseResult r = parse(
		"Accelerator \"bvh\" \"integer maxnodeprims\" [ 4 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.shapes.size(), 1u) << "the shape after it must still load";
	EXPECT_TRUE(hasWarningContaining(r.scene, "Accelerator"));
}

TEST(PbrtSkipTest, SkippingConsumesTheDirectivesParametersNotTheNextDirective) {
	// The resync rule is "skip to the next known directive". If it over- or
	// under-shot, the Shape below would be lost or misparsed.
	const Scene s = parseOk(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\" \"rgb sigma_a\" [ 1 1 1 ]\n"
		"Material \"diffuse\"\n"
		"Shape \"sphere\" \"float radius\" [ 3 ]\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(s.shapes[0].params.getFloat("radius", -1), 3.0);
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].type, "diffuse");
}

TEST(PbrtSkipTest, InstancingIsReportedRatherThanSilentlyWrong) {
	// Emitting instanced geometry once in place is a real approximation, so it
	// has to be visible to the caller.
	const Scene s = parseOk(
		"ObjectBegin \"leaf\"\n"
		"  Shape \"sphere\"\n"
		"ObjectEnd\n"
		"ObjectInstance \"leaf\"\n");
	EXPECT_TRUE(hasWarningContaining(s, "instancing is not supported"));
}

TEST(PbrtSkipTest, StrayTokenWhereADirectiveBelongsIsFatal) {
	const ParseResult r = parse("Shape \"sphere\"\ngarbage\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("expected a directive"), std::string::npos) << r.error;
}

TEST(PbrtSkipTest, MalformedTransformArgumentsAreFatalWithALineNumber) {
	const ParseResult r = parse("Translate 1 2\nShape \"sphere\"\n");
	EXPECT_FALSE(r.ok);
	EXPECT_EQ(r.line, 2) << "reported at the point the numbers ran out";
	EXPECT_NE(r.error.find("Translate"), std::string::npos) << r.error;
}

// ===========================================================================
// A realistic fragment
// ===========================================================================

TEST(PbrtSceneTest, ParsesACornellBoxStyleFragmentEndToEnd) {
	// Shaped like a real pbrt-v4 file: pre-world settings, WorldBegin, an
	// emissive quad in its own attribute scope, then walls.
	const ParseResult r = parse(
		"Integrator \"volpath\" \"integer maxdepth\" [ 50 ]\n"
		"Sampler \"halton\" \"integer pixelsamples\" [ 256 ]\n"
		"Film \"rgb\" \"integer xresolution\" [ 512 ] \"integer yresolution\" [ 512 ]\n"
		"LookAt 278 278 -800  278 278 0  0 1 0\n"
		"Camera \"perspective\" \"float fov\" [ 40 ]\n"
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 17 12 4 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ 343 548 227  343 548 332  213 548 332  213 548 227 ]\n"
		"AttributeEnd\n"
		"MakeNamedMaterial \"white\" \"string type\" \"diffuse\" \"rgb reflectance\" [ .73 .73 .73 ]\n"
		"NamedMaterial \"white\"\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ 552 0 0  0 0 0  0 0 559  552 0 559 ]\n");

	ASSERT_TRUE(r.ok) << r.error;
	const Scene &s = r.scene;
	EXPECT_EQ(s.maxDepth, 50);
	EXPECT_EQ(s.samplesPerPixel, 256);
	EXPECT_EQ(s.xResolution, 512);
	EXPECT_DOUBLE_EQ(s.cameraFov(), 40.0);
	ASSERT_EQ(s.shapes.size(), 2u);

	// The light quad is emissive and the floor is not.
	EXPECT_EQ(s.shapes[0].areaLightIndex, 0);
	EXPECT_EQ(s.shapes[1].areaLightIndex, -1);
	EXPECT_EQ(s.materials[s.shapes[1].materialIndex].name, "white");
	EXPECT_EQ(s.shapes[1].params.find("P")->numbers.size(), 12u);
	EXPECT_TRUE(s.warnings.empty()) << "nothing here should need skipping";
}

// ===========================================================================
// Include
// ===========================================================================
// Real scenes put geometry in separate files - killeroo-simple.pbrt is little
// more than settings plus `Include "geometry/killeroo.pbrt"`. Skipping the
// directive loads an empty scene, which is a far more confusing symptom than
// a refusal, so these carry real weight.

namespace {

// An in-memory filesystem, which is the whole reason parse() takes a resolver
// callback instead of opening files itself.
pbrt_scene::FileResolver mapResolver(std::map<std::string, std::string> files) {
	return [files](const std::string &path, std::string &out) {
		auto it = files.find(path);
		if (it == files.end()) return false;
		out = it->second;
		return true;
	};
}

} // namespace

TEST(PbrtIncludeTest, IncludedGeometryIsSplicedIn) {
	const ParseResult r = parse(
		"Material \"diffuse\"\n"
		"Include \"geometry/mesh.pbrt\"\n",
		mapResolver({{"geometry/mesh.pbrt", "Shape \"sphere\" \"float radius\" [ 4 ]\n"}}));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(r.scene.shapes[0].params.getFloat("radius", -1), 4.0);
}

TEST(PbrtIncludeTest, IncludedFileInheritsTheCurrentGraphicsState) {
	// pbrt's Include is textual: the included file sees the material and CTM in
	// force at the point of inclusion.
	const ParseResult r = parse(
		"Material \"conductor\"\n"
		"Translate 3 0 0\n"
		"Include \"g.pbrt\"\n",
		mapResolver({{"g.pbrt", "Shape \"sphere\"\n"}}));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.shapes.size(), 1u);
	EXPECT_EQ(r.scene.materials[r.scene.shapes[0].materialIndex].type, "conductor");
	EXPECT_DOUBLE_EQ(r.scene.shapes[0].xform.m[3], 3.0);
}

TEST(PbrtIncludeTest, StateChangedInsideAnIncludeLeaksOutAsPbrtIntends) {
	const ParseResult r = parse(
		"Include \"m.pbrt\"\n"
		"Shape \"sphere\"\n",
		mapResolver({{"m.pbrt", "Material \"conductor\"\n"}}));
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.shapes.size(), 1u);
	EXPECT_EQ(r.scene.materials[r.scene.shapes[0].materialIndex].type, "conductor");
}

TEST(PbrtIncludeTest, IncludesNest) {
	const ParseResult r = parse(
		"Include \"a.pbrt\"\n",
		mapResolver({{"a.pbrt", "Include \"b.pbrt\"\n"},
					 {"b.pbrt", "Shape \"sphere\"\n"}}));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.shapes.size(), 1u);
}

TEST(PbrtIncludeTest, ImportIsTreatedLikeInclude) {
	const ParseResult r = parse(
		"Import \"g.pbrt\"\n",
		mapResolver({{"g.pbrt", "Shape \"sphere\"\n"}}));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.shapes.size(), 1u);
}

TEST(PbrtIncludeTest, MissingIncludedFileIsFatalAndNamesThePath) {
	// Not a warning. A scene missing its geometry renders empty, and an empty
	// render gives no clue which path was wrong.
	const ParseResult r = parse("Include \"geometry/gone.pbrt\"\n", mapResolver({}));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("geometry/gone.pbrt"), std::string::npos) << r.error;
}

TEST(PbrtIncludeTest, SelfReferentialIncludeIsRejectedNotRecursedForever) {
	const ParseResult r = parse(
		"Include \"loop.pbrt\"\n",
		mapResolver({{"loop.pbrt", "Include \"loop.pbrt\"\n"}}));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("cycle"), std::string::npos) << r.error;
}

TEST(PbrtIncludeTest, MutualIncludeCycleIsRejected) {
	const ParseResult r = parse(
		"Include \"a.pbrt\"\n",
		mapResolver({{"a.pbrt", "Include \"b.pbrt\"\n"},
					 {"b.pbrt", "Include \"a.pbrt\"\n"}}));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("cycle"), std::string::npos) << r.error;
}

TEST(PbrtIncludeTest, IncludeWithoutAFilenameIsRejected) {
	const ParseResult r = parse("Include\nShape \"sphere\"\n", mapResolver({}));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("quoted filename"), std::string::npos) << r.error;
}

TEST(PbrtIncludeTest, ErrorsInsideAnIncludeNameTheIncludedFile) {
	// "line 2" alone is useless when a scene pulls in six geometry files.
	const ParseResult r = parse(
		"Include \"bad.pbrt\"\n",
		mapResolver({{"bad.pbrt", "Shape \"sphere\"\nTranslate 1 2\n"}}));
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("bad.pbrt"), std::string::npos)
		<< "the failing file must be named: " << r.error;
}

TEST(PbrtIncludeTest, WithoutAResolverIncludeIsDroppedRatherThanFailing) {
	// A caller parsing text in isolation passed no resolver deliberately.
	const ParseResult r = parse("Include \"x.pbrt\"\nShape \"sphere\"\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.shapes.size(), 1u);
}
