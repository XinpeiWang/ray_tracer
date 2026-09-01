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

TEST(PbrtStateTest, CoordinateSystemAndCoordSysTransformRoundTrip) {
	const Scene s = parseOk(
		"Translate 10 0 0\n"
		"CoordinateSystem \"saved\"\n"
		"Translate 5 0 0\n"
		"Shape \"sphere\"\n"     // CTM is now translate(15,0,0)
		"CoordSysTransform \"saved\"\n"
		"Shape \"sphere\"\n");  // CTM restored to translate(10,0,0)
	ASSERT_EQ(s.shapes.size(), 2u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 15.0);
	EXPECT_DOUBLE_EQ(s.shapes[1].xform.m[3], 10.0)
		<< "CoordSysTransform must recall exactly the CTM CoordinateSystem saved, "
		   "not just undo the Translate that happened after it";
}

TEST(PbrtStateTest, CoordSysTransformUnknownNameWarnsAndLeavesCtmUnchanged) {
	const Scene s = parseOk(
		"Translate 10 0 0\n"
		"CoordSysTransform \"never-saved\"\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 10.0)
		<< "an unresolvable CoordSysTransform must not silently reset the CTM";
	EXPECT_TRUE(hasWarningContaining(s, "CoordSysTransform"));
}

TEST(PbrtStateTest, CoordinateSystemSavedInsideAttributeScopeSurvivesAttributeEnd) {
	// namedCoordinateSystems_ is deliberately parser-level, not stack-scoped
	// like gs_.ctm - a name saved inside an AttributeBegin/End block must
	// still be recallable after that block closes (unlike the CTM itself,
	// which AttributeEnd does reset).
	const Scene s = parseOk(
		"AttributeBegin\n"
		"  Translate 7 0 0\n"
		"  CoordinateSystem \"lightFrame\"\n"
		"AttributeEnd\n"                  // CTM reverts to identity here
		"CoordSysTransform \"lightFrame\"\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 7.0)
		<< "a CoordinateSystem saved inside a now-closed AttributeBegin/End scope "
		   "must still be recallable afterward";
}

TEST(PbrtStateTest, WorldBeginRegistersBuiltinWorldCoordinateSystem) {
	// pbrt-v4 builtin: "world" always names the CTM in effect right after
	// WorldBegin (identity), recallable without ever declaring it explicitly.
	const Scene s = parseOk(
		"WorldBegin\n"
		"Translate 3 0 0\n"
		"CoordSysTransform \"world\"\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 0.0)
		<< "CoordSysTransform \"world\" must reset to the CTM at WorldBegin (identity), "
		   "not warn as an unknown coordinate system";
	EXPECT_FALSE(hasWarningContaining(s, "unknown coordinate system"));
}

TEST(PbrtStateTest, CameraDirectiveRegistersBuiltinCameraCoordinateSystem) {
	// pbrt-v4 builtin: "camera" always names the CTM in effect at the Camera
	// statement - the standard camera-relative-light idiom.
	const Scene s = parseOk(
		"Translate 5 0 0\n"
		"Camera \"perspective\"\n"
		"WorldBegin\n"
		"CoordSysTransform \"camera\"\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.shapes.size(), 1u);
	EXPECT_DOUBLE_EQ(s.shapes[0].xform.m[3], 5.0)
		<< "CoordSysTransform \"camera\" must recall the CTM in effect at the Camera directive";
	EXPECT_FALSE(hasWarningContaining(s, "unknown coordinate system"));
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
	// worldToCameraEnd defaults to worldToCamera too when the scene never
	// declares ActiveTransform - see cameraIsAnimated()'s own comment.
	EXPECT_FALSE(s.cameraIsAnimated());
}

// ===========================================================================
// Camera motion blur: ActiveTransform / TransformTimes
// ===========================================================================

TEST(PbrtCameraMotionBlurTest, ActiveTransformStartEndProducesTwoDistinctKeyframes) {
	// pbrt-v4's real camera-motion-blur idiom - two LookAt blocks, one gated
	// to each time endpoint.
	const Scene s = parseOk(
		"ActiveTransform \"StartTime\"\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"EndTime\"\n"
		"LookAt 3 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"All\"\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_TRUE(s.cameraIsAnimated());
	// StartTime's own LookAt must be unaffected by EndTime's later one.
	// m[11] (not m[3]/m[7]) is the component guaranteed to differ here -
	// both LookAt calls target the world origin, so the mapped origin's
	// camera-space x/y (m[3]/m[7]) is always exactly 0 by construction
	// (the target always lies on the camera's own forward/z axis) - only
	// the eye-to-target DISTANCE (m[11], the z component) differs between
	// eye=(0,0,-5) (distance 5) and eye=(3,0,-5) (distance sqrt(34)).
	EXPECT_NE(s.worldToCamera.m[11], s.worldToCameraEnd.m[11])
		<< "the two keyframes' own eye-to-target distances must differ";
}

TEST(PbrtCameraMotionBlurTest, ActiveTransformDefaultsToAllAffectingBothSlots) {
	// A scene that never declares ActiveTransform at all keeps EVERY
	// transform directive affecting both slots identically (activeTransform
	// Bits defaults to 3, "All") - the common, non-animated case.
	const Scene s = parseOk(
		"Translate 1 2 3\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_FALSE(s.cameraIsAnimated());
	for (int i = 0; i < 16; ++i)
		EXPECT_DOUBLE_EQ(s.worldToCamera.m[i], s.worldToCameraEnd.m[i]);
}

TEST(PbrtCameraMotionBlurTest, ActiveTransformStartTimeAloneLeavesEndTimeAtItsPriorValue) {
	// Regression guard: while ActiveTransform "StartTime" is active, ONLY
	// the start-time slot should move - the end-time slot must stay exactly
	// where it was (identity, since nothing moved it yet), not silently pick
	// up the start-time-only transform too.
	const Scene s = parseOk(
		"ActiveTransform \"StartTime\"\n"
		"Translate 5 0 0\n"
		"ActiveTransform \"All\"\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.worldToCamera.m[3], 5.0);
	EXPECT_DOUBLE_EQ(s.worldToCameraEnd.m[3], 0.0);
	EXPECT_TRUE(s.cameraIsAnimated());
}

TEST(PbrtCameraMotionBlurTest, UnrecognizedActiveTransformStateWarnsAndIsIgnored) {
	const Scene s = parseOk(
		"ActiveTransform \"Bogus\"\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_TRUE(hasWarningContaining(s, "ActiveTransform"));
	// activeTransformBits must stay at its default (both slots still move
	// together), not get left in some partial/undefined state.
	EXPECT_FALSE(s.cameraIsAnimated());
}

TEST(PbrtCameraMotionBlurTest, TransformTimesIsParsedIntoSceneFields) {
	const Scene s = parseOk("TransformTimes 0.25 0.75\nWorldBegin\nShape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.transformTimeStart, 0.25);
	EXPECT_DOUBLE_EQ(s.transformTimeEnd, 0.75);
}

TEST(PbrtCameraMotionBlurTest, TransformTimesDefaultsToZeroOne) {
	const Scene s = parseOk("WorldBegin\nShape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.transformTimeStart, 0.0);
	EXPECT_DOUBLE_EQ(s.transformTimeEnd, 1.0);
}

TEST(PbrtCameraMotionBlurTest, ActiveTransformAcceptsRealPbrtV4UnquotedStateKeyword) {
	// Real pbrt-v4 syntax is UNQUOTED - `ActiveTransform StartTime`, not
	// `ActiveTransform "StartTime"` - confirmed against pbrt-v4's own file
	// format documentation. The quoted form (exercised by the tests above)
	// is accepted too, defensively, but every actual downloaded/authored
	// .pbrt scene uses this unquoted form, so it needs its own direct
	// regression coverage - a prior version of this parser required
	// `.quoted`, which meant a real pbrt-v4 scene using this exact idiom
	// would hit a fatal parse error rather than working at all.
	const Scene s = parseOk(
		"ActiveTransform StartTime\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform EndTime\n"
		"LookAt 3 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform All\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_TRUE(s.cameraIsAnimated());
	EXPECT_NE(s.worldToCamera.m[11], s.worldToCameraEnd.m[11]);
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
	EXPECT_EQ(s.samplerType, "halton");
	EXPECT_EQ(s.samplesPerPixel, 128);
	EXPECT_EQ(s.xResolution, 800);
	EXPECT_EQ(s.yResolution, 600);
	EXPECT_EQ(s.integrator, "volpath");
	EXPECT_EQ(s.maxDepth, 12);
}

TEST(PbrtSettingsTest, RegularizeDefaultsToFalseMatchingPbrtV4) {
	const Scene s = parseOk("WorldBegin\nShape \"sphere\"\n");
	EXPECT_FALSE(s.regularize);
}

TEST(PbrtSettingsTest, IntegratorRegularizeIsRead) {
	const Scene s = parseOk(
		"Integrator \"volpath\" \"integer maxdepth\" [ 8 ] \"bool regularize\" [ true ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.maxDepth, 8);
	EXPECT_TRUE(s.regularize);
}

TEST(PbrtSettingsTest, IntegratorRegularizeFalseIsReadExplicitly) {
	const Scene s = parseOk(
		"Integrator \"volpath\" \"bool regularize\" [ false ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_FALSE(s.regularize);
}

TEST(PbrtSettingsTest, CropWindowDefaultsToFullFrame) {
	const Scene s = parseOk("WorldBegin\nShape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.cropWindow[0], 0.0);
	EXPECT_DOUBLE_EQ(s.cropWindow[1], 1.0);
	EXPECT_DOUBLE_EQ(s.cropWindow[2], 0.0);
	EXPECT_DOUBLE_EQ(s.cropWindow[3], 1.0);
	EXPECT_FALSE(s.hasPixelBounds);
}

TEST(PbrtSettingsTest, FilmCropWindowIsRead) {
	const Scene s = parseOk(
		"Film \"rgb\" \"float cropwindow\" [ 0.25 0.75 0.1 0.9 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_DOUBLE_EQ(s.cropWindow[0], 0.25);
	EXPECT_DOUBLE_EQ(s.cropWindow[1], 0.75);
	EXPECT_DOUBLE_EQ(s.cropWindow[2], 0.1);
	EXPECT_DOUBLE_EQ(s.cropWindow[3], 0.9);
}

TEST(PbrtSettingsTest, FilmPixelBoundsIsRead) {
	const Scene s = parseOk(
		"Film \"rgb\" \"integer xresolution\" [ 800 ] \"integer yresolution\" [ 600 ] "
		"\"integer pixelbounds\" [ 100 700 50 550 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_TRUE(s.hasPixelBounds);
	EXPECT_EQ(s.pixelBounds[0], 100);
	EXPECT_EQ(s.pixelBounds[1], 700);
	EXPECT_EQ(s.pixelBounds[2], 50);
	EXPECT_EQ(s.pixelBounds[3], 550);
}

TEST(PbrtSettingsTest, LightSamplerDefaultsToBvhWithNoDirective) {
	const Scene s = parseOk("WorldBegin\nShape \"sphere\"\n");
	EXPECT_EQ(s.lightSamplerType, "bvh");
}

TEST(PbrtSettingsTest, IntegratorLightSamplerIsRead) {
	const Scene s = parseOk(
		"Integrator \"volpath\" \"string lightsampler\" [ \"power\" ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.lightSamplerType, "power");
}

TEST(PbrtSettingsTest, SamplerDefaultsToSobolWithNoDirective) {
	const Scene s = parseOk(
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.samplerType, "sobol");
}

TEST(PbrtSettingsTest, PixelFilterIsRead) {
	const Scene s = parseOk(
		"PixelFilter \"mitchell\" \"float B\" [ 0.2 ] \"float C\" [ 0.4 ]\n"
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.filterType, "mitchell");
	EXPECT_DOUBLE_EQ(s.filterParams.getFloat("B", -1.0), 0.2);
	EXPECT_DOUBLE_EQ(s.filterParams.getFloat("C", -1.0), 0.4);
}

TEST(PbrtSettingsTest, PixelFilterDefaultsToGaussianWithNoDirective) {
	// pbrt-v4's real default (see pbrt_flatten::PixelFilter's own comment) -
	// not box/triangle/mitchell.
	const Scene s = parseOk(
		"WorldBegin\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.filterType, "gaussian");
}

// ColorSpace is captured onto each LightDecl at declaration time (like
// xform), not stored as one Scene-wide setting - see GraphicsState::
// colorSpaceName's own comment (pbrt_scene.h) for why. These tests check
// a declared light's captured colorSpaceName, the only place it's visible.

TEST(PbrtSettingsTest, ColorSpaceDefaultsToSRGBWithNoDirective) {
	const Scene s = parseOk(
		"WorldBegin\n"
		"LightSource \"distant\"\n");
	ASSERT_EQ(s.lights.size(), 1u);
	EXPECT_EQ(s.lights[0].colorSpaceName, "srgb");
}

TEST(PbrtSettingsTest, ColorSpaceDirectiveIsRead) {
	for (const std::string &name : {"srgb", "dci-p3", "rec2020", "aces2065-1"}) {
		const Scene s = parseOk(
			"ColorSpace \"" + name + "\"\n"
			"WorldBegin\n"
			"LightSource \"distant\"\n");
		ASSERT_EQ(s.lights.size(), 1u);
		EXPECT_EQ(s.lights[0].colorSpaceName, name);
	}
}

TEST(PbrtSettingsTest, ColorSpaceUnrecognizedNameFallsBackToSRGBWithWarning) {
	const Scene s = parseOk(
		"ColorSpace \"not-a-real-colorspace\"\n"
		"WorldBegin\n"
		"LightSource \"distant\"\n");
	ASSERT_EQ(s.lights.size(), 1u);
	EXPECT_EQ(s.lights[0].colorSpaceName, "srgb");
	EXPECT_TRUE(hasWarningContaining(s, "ColorSpace"));
}

TEST(PbrtStateTest, ColorSpaceIsScopedByAttributeBeginEnd) {
	// The gap the code review caught: ColorSpace must behave like every
	// other piece of graphics state (ctm, materialIndex, ...) and revert at
	// AttributeEnd, not leak to lights declared afterward.
	const Scene s = parseOk(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  ColorSpace \"rec2020\"\n"
		"  LightSource \"distant\"\n"     // inside the scope: rec2020
		"AttributeEnd\n"
		"LightSource \"distant\"\n");     // after AttributeEnd: back to srgb
	ASSERT_EQ(s.lights.size(), 2u);
	EXPECT_EQ(s.lights[0].colorSpaceName, "rec2020");
	EXPECT_EQ(s.lights[1].colorSpaceName, "srgb")
		<< "AttributeEnd must restore the color space that was active before AttributeBegin, "
		   "not leave the rec2020 set inside the scope in effect for later lights";
}

// ===========================================================================
// Unsupported directives degrade rather than fail
// ===========================================================================

TEST(PbrtSkipTest, UnknownDirectiveWarnsAndDoesNotFail) {
	// "Accelerator" used to be this test's own example - it's real, parsed
	// input now (see the Accelerator tests below), so this uses a directive
	// name pbrt-v4 has no such grammar for at all.
	const ParseResult r = parse(
		"NotARealDirective \"foo\" \"integer bar\" [ 4 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.shapes.size(), 1u) << "the shape after it must still load";
	EXPECT_TRUE(hasWarningContaining(r.scene, "NotARealDirective"));
}

// ===========================================================================
// Accelerator - real splitmethod/maxnodeprims capture
// ===========================================================================

TEST(PbrtAcceleratorTest, DefaultsMatchPbrtV4) {
	const Scene s = parseOk("Shape \"sphere\"\n");
	EXPECT_EQ(s.acceleratorType, "bvh");
	EXPECT_EQ(s.acceleratorSplitMethod, "sah");
	EXPECT_EQ(s.acceleratorMaxNodePrims, 4);
}

TEST(PbrtAcceleratorTest, ExplicitSplitMethodAndMaxNodePrimsAreRead) {
	const Scene s = parseOk(
		"Accelerator \"bvh\" \"string splitmethod\" \"hlbvh\" "
		"\"integer maxnodeprims\" [ 8 ]\n"
		"Shape \"sphere\"\n");
	EXPECT_EQ(s.acceleratorType, "bvh");
	EXPECT_EQ(s.acceleratorSplitMethod, "hlbvh");
	EXPECT_EQ(s.acceleratorMaxNodePrims, 8);
	EXPECT_FALSE(hasWarningContaining(s, "Accelerator"))
		<< "a fully-supported Accelerator directive must not warn";
}

TEST(PbrtAcceleratorTest, UnrecognizedTypeIsCapturedRaw) {
	// This parser layer captures whatever string was given verbatim; the
	// "not bvh -> warn and fall back" decision happens in flatten() (see
	// FlattenTest.AcceleratorUnrecognizedTypeWarnsAndFallsBackToBvh,
	// pbrt_flatten_tests.cpp), matching every other Scene::*Type field's
	// "parse now, resolve later" split in this file.
	const Scene s = parseOk("Accelerator \"kdtree\"\nShape \"sphere\"\n");
	EXPECT_EQ(s.acceleratorType, "kdtree");
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

TEST(PbrtSkipTest, InstancingIsParsedRatherThanApproximated) {
	// This test used to assert the opposite: that instancing warned it was
	// unsupported and emitted the geometry once in place. That approximation
	// is gone - a definition is now recorded separately from the scene and
	// placed by its instances, which is what pbrt means by the directives.
	// See pbrt_instance_tests.cpp for the behaviour in full.
	const Scene s = parseOk(
		"ObjectBegin \"leaf\"\n"
		"  Shape \"sphere\"\n"
		"ObjectEnd\n"
		"ObjectInstance \"leaf\"\n");
	EXPECT_FALSE(hasWarningContaining(s, "instancing is not supported"));
	EXPECT_TRUE(s.shapes.empty()) << "the definition was emitted in place";
	ASSERT_EQ(s.objects.size(), 1u);
	EXPECT_EQ(s.objects[0].shapes.size(), 1u);
	EXPECT_EQ(s.instances.size(), 1u);
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
