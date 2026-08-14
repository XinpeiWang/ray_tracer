/**
 * @file pbrt_flatten_tests.cpp
 * @brief Unit tests for baking pbrt's transform hierarchy into world geometry
 *
 * The transform is applied once at load rather than per ray, so a mistake here
 * is permanent and silent: geometry ends up in the wrong place and every later
 * stage faithfully renders it there. These check actual coordinates, not just
 * that something was produced.
 */

#include <gtest/gtest.h>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"

#include <cmath>
#include <string>
#include <vector>

using namespace pbrt_flatten;

namespace {

FlatScene flattenSource(const std::string &text, const MeshResolver &meshes = {}) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return flatten(r.scene, meshes);
}

bool warnedAbout(const FlatScene &s, const std::string &needle) {
	for (const pbrt_scene::Warning &w : s.warnings)
		if (w.message.find(needle) != std::string::npos) return true;
	return false;
}

// A unit quad in the XY plane, as two triangles.
const char *kQuadMesh =
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
	"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n";

} // namespace

// ===========================================================================
// Triangles
// ===========================================================================

TEST(FlattenTest, UntransformedMeshKeepsItsCoordinates) {
	const FlatScene s = flattenSource(kQuadMesh);
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[0], 0.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[3], 1.0);   // second vertex x
	EXPECT_DOUBLE_EQ(s.triangles[0].v[7], 1.0);   // third vertex y
}

TEST(FlattenTest, TranslationIsBakedIntoEveryVertex) {
	const FlatScene s = flattenSource(std::string("Translate 10 20 30\n") + kQuadMesh);
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[0], 10.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[1], 20.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[2], 30.0);
	// the second triangle shares vertex 0 and must land in the same place
	EXPECT_DOUBLE_EQ(s.triangles[1].v[0], 10.0);
	EXPECT_DOUBLE_EQ(s.triangles[1].v[1], 20.0);
}

TEST(FlattenTest, ScaleAndRotationCompose) {
	// Scale by 2 then rotate 90 degrees about Z: (1,0,0) -> (2,0,0) -> (0,2,0).
	const FlatScene s = flattenSource(
		"Rotate 90 0 0 1\n"
		"Scale 2 2 2\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_NEAR(s.triangles[0].v[3], 0.0, 1e-9);
	EXPECT_NEAR(s.triangles[0].v[4], 2.0, 1e-9);
}

TEST(FlattenTest, TransformsRespectAttributeScope) {
	const FlatScene s = flattenSource(
		std::string("AttributeBegin\n  Translate 100 0 0\n") + kQuadMesh + "AttributeEnd\n" + kQuadMesh);
	ASSERT_EQ(s.triangles.size(), 4u);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[0], 100.0) << "inside the scope";
	EXPECT_DOUBLE_EQ(s.triangles[2].v[0], 0.0)   << "outside it";
}

TEST(FlattenTest, MaterialAndAreaLightIndicesSurvive) {
	const FlatScene s = flattenSource(
		std::string("Material \"conductor\"\n"
					"AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuadMesh + "AttributeEnd\n" + kQuadMesh);
	ASSERT_EQ(s.triangles.size(), 4u);
	EXPECT_EQ(s.triangles[0].areaLight, 0) << "emissive inside the scope";
	EXPECT_EQ(s.triangles[3].areaLight, -1) << "not emissive outside it";
	EXPECT_EQ(s.triangles[0].material, s.triangles[3].material);
	EXPECT_GE(s.triangles[0].material, 0);
}

// ===========================================================================
// Spheres
// ===========================================================================

TEST(FlattenTest, SphereCentreIsTransformedAndRadiusScaled) {
	const FlatScene s = flattenSource(
		"Translate 1 2 3\n"
		"Scale 4 4 4\n"
		"Shape \"sphere\" \"float radius\" [ 0.5 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_DOUBLE_EQ(s.spheres[0].center[0], 1.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].center[2], 3.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].radius, 2.0) << "0.5 scaled by 4";
}

TEST(FlattenTest, UniformScaleDoesNotWarn) {
	const FlatScene s = flattenSource(
		"Scale 3 3 3\nShape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_FALSE(warnedAbout(s, "ellipsoid"));
	EXPECT_DOUBLE_EQ(s.spheres[0].radius, 3.0);
}

TEST(FlattenTest, NonUniformScaleOnASphereIsReportedNotSilentlyRounded) {
	// A squashed sphere cannot be represented, and quietly emitting a round one
	// is exactly the difference nobody notices against a reference image.
	const FlatScene s = flattenSource(
		"Scale 1 5 1\nShape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(warnedAbout(s, "ellipsoid"));
	EXPECT_DOUBLE_EQ(s.spheres[0].radius, 5.0) << "largest axis is used";
}

TEST(FlattenTest, RotationAloneDoesNotCountAsNonUniformScale) {
	// A rotation leaves all three basis lengths at 1; a naive check that looked
	// at raw matrix entries rather than their lengths would warn here.
	const FlatScene s = flattenSource(
		"Rotate 37 0.3 0.5 0.8\nShape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_FALSE(warnedAbout(s, "ellipsoid"));
	EXPECT_NEAR(s.spheres[0].radius, 1.0, 1e-9);
}

// ===========================================================================
// PLY meshes
// ===========================================================================

TEST(FlattenTest, PlyMeshIsResolvedAndTransformed) {
	const MeshResolver res = [](const std::string &path,
								std::vector<float> &pos, std::vector<int> &idx) {
		if (path != "geometry/tri.ply") return false;
		pos = {0, 0, 0,  1, 0, 0,  0, 1, 0};
		idx = {0, 1, 2};
		return true;
	};
	const FlatScene s = flattenSource(
		"Translate 5 0 0\n"
		"Shape \"plymesh\" \"string filename\" [ \"geometry/tri.ply\" ]\n", res);
	ASSERT_EQ(s.triangles.size(), 1u);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[0], 5.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].v[3], 6.0);
}

TEST(FlattenTest, UnreadablePlyMeshWarnsRatherThanAborting) {
	const MeshResolver res = [](const std::string &, std::vector<float> &, std::vector<int> &) {
		return false;
	};
	const FlatScene s = flattenSource(
		"Shape \"plymesh\" \"string filename\" [ \"missing.ply\" ]\n" + std::string(kQuadMesh), res);
	EXPECT_TRUE(warnedAbout(s, "missing.ply"));
	EXPECT_EQ(s.triangles.size(), 2u) << "the rest of the scene still builds";
}

TEST(FlattenTest, PlyMeshWithoutAResolverIsReported) {
	const FlatScene s = flattenSource(
		"Shape \"plymesh\" \"string filename\" [ \"x.ply\" ]\n");
	EXPECT_TRUE(warnedAbout(s, "no mesh resolver"));
	EXPECT_TRUE(s.empty());
}

// ===========================================================================
// Malformed geometry
// ===========================================================================

TEST(FlattenTest, OutOfRangeFaceIndicesAreDroppedWithOneWarning) {
	const FlatScene s = flattenSource(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 1 99 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_EQ(s.triangles.size(), 1u) << "the valid face survives";
	EXPECT_TRUE(warnedAbout(s, "outside its vertex list"));
}

TEST(FlattenTest, TrailingIndicesThatDoNotFormATriangleAreReported) {
	const FlatScene s = flattenSource(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 1 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_EQ(s.triangles.size(), 1u);
	EXPECT_TRUE(warnedAbout(s, "multiple of 3"));
}

TEST(FlattenTest, MeshMissingItsParametersIsSkippedNotCrashed) {
	const FlatScene s = flattenSource("Shape \"trianglemesh\"\n");
	EXPECT_TRUE(s.empty());
	EXPECT_TRUE(warnedAbout(s, "missing its P or indices"));
}

TEST(FlattenTest, UnsupportedShapeTypesAreNamed) {
	const FlatScene s = flattenSource("Shape \"cylinder\" \"float radius\" [ 1 ]\n");
	EXPECT_TRUE(s.empty());
	EXPECT_TRUE(warnedAbout(s, "cylinder"));
}

TEST(FlattenTest, ParserWarningsAreCarriedThrough) {
	// The caller gets one list, not two - it should not have to check both the
	// parse result and the flatten result to learn what was approximated.
	const FlatScene s = flattenSource(
		"Accelerator \"bvh\"\n" + std::string(kQuadMesh));
	EXPECT_TRUE(warnedAbout(s, "Accelerator"));
}

// ===========================================================================
// Camera
// ===========================================================================
// pbrt hands over a world-to-camera matrix; our camera wants an eye, a target
// and a vertical fov. Getting the inversion or the fov convention wrong
// mis-frames every scene, and does so plausibly enough to look intentional.

TEST(FlattenCameraTest, EyeAndTargetAreRecoveredFromWorldToCamera) {
	const FlatScene s = flattenSource(
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"Camera \"perspective\" \"float fov\" [ 40 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_NEAR(s.camera.lookfrom[0], 0.0, 1e-9);
	EXPECT_NEAR(s.camera.lookfrom[1], 0.0, 1e-9);
	EXPECT_NEAR(s.camera.lookfrom[2], -5.0, 1e-9);
	// looking toward the origin means the target is further along +z
	EXPECT_NEAR(s.camera.lookat[2], -4.0, 1e-9);
	EXPECT_NEAR(s.camera.up[1], 1.0, 1e-9);
}

TEST(FlattenCameraTest, OffAxisEyePositionRoundTrips) {
	const FlatScene s = flattenSource(
		"LookAt 3 4 1.5   0.5 0.5 0   0 0 1\n"
		"Camera \"perspective\"\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_NEAR(s.camera.lookfrom[0], 3.0, 1e-9);
	EXPECT_NEAR(s.camera.lookfrom[1], 4.0, 1e-9);
	EXPECT_NEAR(s.camera.lookfrom[2], 1.5, 1e-9);

	// The view direction must point from the eye toward the stated target.
	const double dx = s.camera.lookat[0] - s.camera.lookfrom[0];
	const double dy = s.camera.lookat[1] - s.camera.lookfrom[1];
	const double dz = s.camera.lookat[2] - s.camera.lookfrom[2];
	const double tx = 0.5 - 3.0, ty = 0.5 - 4.0, tz = 0.0 - 1.5;
	const double tlen = std::sqrt(tx * tx + ty * ty + tz * tz);
	EXPECT_NEAR(dx, tx / tlen, 1e-9);
	EXPECT_NEAR(dy, ty / tlen, 1e-9);
	EXPECT_NEAR(dz, tz / tlen, 1e-9);
}

TEST(FlattenCameraTest, FovIsTakenAsVerticalOnALandscapeFrame) {
	const FlatScene s = flattenSource(
		"Film \"rgb\" \"integer xresolution\" [ 800 ] \"integer yresolution\" [ 600 ]\n"
		"Camera \"perspective\" \"float fov\" [ 45 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_NEAR(s.camera.vfov, 45.0, 1e-9);
}

TEST(FlattenCameraTest, FovIsConvertedOnAPortraitFrame) {
	// pbrt's fov covers the NARROWER axis, which on a portrait frame is the
	// width. Taking it as vertical unconditionally would leave vfov at 45 and
	// silently mis-frame the scene.
	const FlatScene s = flattenSource(
		"Film \"rgb\" \"integer xresolution\" [ 600 ] \"integer yresolution\" [ 800 ]\n"
		"Camera \"perspective\" \"float fov\" [ 45 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_GT(s.camera.vfov, 45.0) << "the vertical angle must widen, not stay put";
	// tan(v/2) = tan(h/2) / aspect, aspect = 600/800
	const double expect = 2.0 * std::atan(std::tan(45.0 * 0.5 * 3.14159265358979323846 / 180.0)
										  / 0.75) * 180.0 / 3.14159265358979323846;
	EXPECT_NEAR(s.camera.vfov, expect, 1e-9);
}

TEST(FlattenCameraTest, DepthOfFieldParametersAreCarriedOver) {
	const FlatScene s = flattenSource(
		"Camera \"perspective\" \"float lensradius\" [ 0.1 ] \"float focaldistance\" [ 7 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_NEAR(s.camera.aperture, 0.2, 1e-9) << "aperture is diameter, pbrt gives radius";
	EXPECT_NEAR(s.camera.focusDistance, 7.0, 1e-9);
}

TEST(FlattenCameraTest, CameraTypeDefaultsToPerspective) {
	const FlatScene s = flattenSource("WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.type, "perspective");
}

TEST(FlattenCameraTest, ScreenWindowOverrideIsCarriedThrough) {
	// Orthographic has no fov to derive a scale from any other way - a scene
	// authored larger than ~1 world unit across needs this to see anything.
	const FlatScene s = flattenSource(
		"Camera \"orthographic\" \"float screenwindow\" [ -8 8 -8 8 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	ASSERT_TRUE(s.camera.hasScreenWindow);
	EXPECT_DOUBLE_EQ(s.camera.screenWindow[0], -8.0);
	EXPECT_DOUBLE_EQ(s.camera.screenWindow[1], 8.0);
	EXPECT_DOUBLE_EQ(s.camera.screenWindow[3], 8.0);
}

TEST(FlattenCameraTest, MissingScreenWindowLeavesTheDefaultInPlace) {
	const FlatScene s = flattenSource(
		"Camera \"orthographic\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_FALSE(s.camera.hasScreenWindow);
}

TEST(FlattenCameraTest, ScreenWindowWithTooFewNumbersIsIgnoredAndWarned) {
	const FlatScene s = flattenSource(
		"Camera \"orthographic\" \"float screenwindow\" [ -8 8 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_FALSE(s.camera.hasScreenWindow);
	EXPECT_TRUE(warnedAbout(s, "screenwindow"));
}

TEST(FlattenCameraTest, OrthographicCameraTypeIsCarriedThrough) {
	// Before this, the type argument to Camera was parsed and then never
	// looked at again - every loaded scene rendered as perspective regardless.
	const FlatScene s = flattenSource(
		"Camera \"orthographic\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.type, "orthographic");
}

TEST(FlattenCameraTest, SphericalCameraReadsItsMappingParameter) {
	const FlatScene s = flattenSource(
		"Camera \"spherical\" \"string mapping\" [ \"equalarea\" ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.type, "spherical");
	EXPECT_EQ(s.camera.sphericalMapping, "equalarea");
}

TEST(FlattenCameraTest, SphericalCameraDefaultsToEquirectangular) {
	const FlatScene s = flattenSource(
		"Camera \"spherical\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.sphericalMapping, "equirectangular");
}

TEST(FlattenCameraTest, RealisticCameraReadsItsOwnParameters) {
	const FlatScene s = flattenSource(
		"Camera \"realistic\" \"string lensfile\" [ \"dgauss.dat\" ] "
		"\"float aperturediameter\" [ 4.5 ] \"float filmdiag\" [ 50 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.type, "realistic");
	EXPECT_EQ(s.camera.lensFile, "dgauss.dat");
	EXPECT_NEAR(s.camera.apertureDiameterMM, 4.5, 1e-9);
	EXPECT_NEAR(s.camera.filmDiagonalMM, 50.0, 1e-9);
}

TEST(FlattenCameraTest, RealisticCameraFocusDistanceSpellingIsAlsoRead) {
	// pbrt-v4 spells this "focaldistance" for perspective/orthographic but
	// "focusdistance" for realistic - reading only the first silently keeps
	// every realistic-camera scene at the 1e6 no-DOF sentinel.
	const FlatScene s = flattenSource(
		"Camera \"realistic\" \"string lensfile\" [ \"dgauss.dat\" ] "
		"\"float focusdistance\" [ 12 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_NEAR(s.camera.focusDistance, 12.0, 1e-9);
}

TEST(FlattenCameraTest, RealisticCameraWithNoLensfileFallsBackToPerspective) {
	const FlatScene s = flattenSource(
		"Camera \"realistic\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.camera.type, "perspective");
	EXPECT_TRUE(warnedAbout(s, "lensfile"));
}

// ===========================================================================
// Materials and emission
// ===========================================================================

TEST(FlattenMaterialTest, PbrtNamesMapStraightAcross) {
	const FlatScene s = flattenSource(
		"Material \"coateddiffuse\" \"rgb reflectance\" [ .2 .4 .6 ] \"float roughness\" [ .3 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::CoatedDiffuse);
	EXPECT_DOUBLE_EQ(s.materials[0].color[1], 0.4);
	EXPECT_DOUBLE_EQ(s.materials[0].roughness, 0.3);
}

TEST(FlattenMaterialTest, DielectricEtaIsReadAsIor) {
	const FlatScene s = flattenSource(
		"Material \"dielectric\" \"float eta\" [ 1.33 ]\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Dielectric);
	EXPECT_DOUBLE_EQ(s.materials[0].ior, 1.33);
}

TEST(FlattenMaterialTest, RoughnessFallsBackToUOrVRoughnessWhenIsotropicIsAbsent) {
	// A scene that only gives the anisotropic pair has no "roughness" key to
	// find, and without a fallback chain that silently reads as 0 - a
	// perfect mirror - regardless of what uroughness/vroughness said.
	const FlatScene s = flattenSource(
		"Material \"coateddiffuse\" \"float uroughness\" [ .25 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].roughness, 0.25);
}

TEST(FlattenMaterialTest, VRoughnessIsUsedWhenNeitherRoughnessNorURoughnessIsGiven) {
	const FlatScene s = flattenSource(
		"Material \"coateddiffuse\" \"float vroughness\" [ .4 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].roughness, 0.4);
}

TEST(FlattenMaterialTest, PlainRoughnessTakesPriorityOverTheAnisotropicPair) {
	const FlatScene s = flattenSource(
		"Material \"coateddiffuse\" \"float roughness\" [ .1 ] "
		"\"float uroughness\" [ .9 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].roughness, 0.1);
}

TEST(FlattenMaterialTest, DiffuseTransmissionReadsItsOwnTransmittanceColour) {
	// Distinct from "reflectance" on purpose - a diffusetransmission material
	// that let one silently stand in for the other would look identical from
	// both sides, which is not what the scene asked for.
	const FlatScene s = flattenSource(
		"Material \"diffusetransmission\" \"rgb reflectance\" [ .8 .1 .1 ] "
		"\"rgb transmittance\" [ .1 .1 .8 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::DiffuseTransmission);
	EXPECT_DOUBLE_EQ(s.materials[0].color[0], 0.8);
	EXPECT_DOUBLE_EQ(s.materials[0].transmittance[2], 0.8);
}

TEST(FlattenMaterialTest, DiffuseTransmissionDefaultsMatchPbrt) {
	// pbrt-v4's own default for this material is 0.25 for both channels, not
	// the generic 0.5 mid-grey every other material defaults its colour to.
	const FlatScene s = flattenSource(
		"Material \"diffusetransmission\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].transmittance[0], 0.25);
	EXPECT_DOUBLE_EQ(s.materials[0].transmittance[1], 0.25);
	EXPECT_DOUBLE_EQ(s.materials[0].transmittance[2], 0.25);
}

TEST(FlattenMaterialTest, UnsupportedMaterialIsFlaggedNotSilentlySubstituted) {
	// A subsurface material rendered as diffuse looks plausible and is wrong,
	// so the substitution has to be announced.
	const FlatScene s = flattenSource(
		"Material \"subsurface\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Unsupported);
	EXPECT_EQ(s.materials[0].pbrtType, "subsurface");
	EXPECT_TRUE(warnedAbout(s, "subsurface"));
}

TEST(FlattenMaterialTest, AreaLightRadianceAndScaleAreExtracted) {
	const FlatScene s = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 17 12 4 ] \"float scale\" [ 2 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	ASSERT_EQ(s.areaLights.size(), 1u);
	EXPECT_DOUBLE_EQ(s.areaLights[0].L[0], 17.0);
	EXPECT_DOUBLE_EQ(s.areaLights[0].L[2], 4.0);
	EXPECT_DOUBLE_EQ(s.areaLights[0].scale, 2.0);
}

TEST(FlattenMaterialTest, BlackbodyEmissionIsReportedAsApproximated) {
	const FlatScene s = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"blackbody L\" [ 6500 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	EXPECT_TRUE(warnedAbout(s, "blackbody"));
}

TEST(FlattenMaterialTest, MaterialIndicesOnGeometryStillLineUp) {
	const FlatScene s = flattenSource(
		"Material \"diffuse\"\n" + std::string(kQuadMesh)
		+ "Material \"conductor\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 2u);
	ASSERT_EQ(s.triangles.size(), 4u);
	EXPECT_EQ(s.materials[s.triangles[0].material].kind, MaterialKind::Diffuse);
	EXPECT_EQ(s.materials[s.triangles[3].material].kind, MaterialKind::Conductor);
}

// ---------------------------------------------------------------------------
// Focus distance
// ---------------------------------------------------------------------------
// focusDistanceFor() exists because pbrt's "no depth of field" default is the
// sentinel 1e6, and our camera uses focus_dist to size the viewport as well as
// to place the plane of focus. Handing it the sentinel scales the primary ray
// direction by a million, and the fixed t_min of 0.001 then rejects every hit
// within a thousand world units - which deleted near geometry from every
// loaded scene and looked like a broken material.

TEST(PbrtFocusTest, TheNoDepthOfFieldSentinelNeverReachesTheCamera) {
	pbrt_flatten::Camera c;                       // defaults: aperture 0, 1e6
	c.lookfrom[2] = -800; c.lookat[2] = 0;
	EXPECT_LT(pbrt_flatten::focusDistanceFor(c), 1e5)
		<< "pbrt's 1e6 sentinel was passed through as if it were a measurement";
}

TEST(PbrtFocusTest, WithNoApertureItFocusesOnTheSubject) {
	// Nothing is out of focus without an aperture, so the only requirement is
	// a sane viewport scale - and the distance to what the camera is aimed at
	// is the one number guaranteed to be on the scene's own scale.
	pbrt_flatten::Camera c;
	c.lookfrom[0] = 278; c.lookfrom[1] = 278; c.lookfrom[2] = -800;
	c.lookat[0] = 278;   c.lookat[1] = 278;   c.lookat[2] = 0;
	EXPECT_NEAR(pbrt_flatten::focusDistanceFor(c), 800.0, 1e-9);
}

TEST(PbrtFocusTest, AnExplicitFocalDistanceIsHonouredWhenThereIsAnAperture) {
	pbrt_flatten::Camera c;
	c.lookfrom[2] = -10; c.lookat[2] = 0;
	c.aperture = 0.5;
	c.focusDistance = 37.0;
	EXPECT_NEAR(pbrt_flatten::focusDistanceFor(c), 37.0, 1e-9)
		<< "a scene that set focaldistance meant it";
}

TEST(PbrtFocusTest, AnApertureWithNoFocalDistanceStillFallsBackRatherThanUsing1e6) {
	// pbrt lets a scene set lensradius without focaldistance, leaving the
	// sentinel in place. Honouring it literally would reintroduce the bug on
	// exactly the scenes that asked for depth of field.
	pbrt_flatten::Camera c;
	c.lookfrom[2] = -10; c.lookat[2] = 0;
	c.aperture = 0.5;
	EXPECT_NEAR(pbrt_flatten::focusDistanceFor(c), 10.0, 1e-9);
}

TEST(PbrtFocusTest, ADegenerateCameraStillYieldsAUsableDistance) {
	// lookfrom == lookat is nonsense, but returning 0 would make the viewport
	// zero-sized and every ray degenerate. Better a harmless default.
	pbrt_flatten::Camera c;
	for (int i = 0; i < 3; ++i) { c.lookfrom[i] = 5; c.lookat[i] = 5; }
	EXPECT_GT(pbrt_flatten::focusDistanceFor(c), 0.0);
}

// ===========================================================================
// LightSource "infinite"
// ===========================================================================
// Real pbrt-v4 scenes (pavilion, ganesha, sportscar-sky) lean on this as
// their main illumination. Dropping it - the old behaviour - renders black
// or near-black instead of failing loudly, so these pin what flatten() must
// carry through: presence, the constant-colour and filename forms, and the
// CTM (an environment map with its rotation dropped still renders, just with
// the sun/horizon facing the wrong way - easy to miss without a check).

TEST(FlattenInfiniteLightTest, AbsentByDefault) {
	const FlatScene s = flattenSource(kQuadMesh);
	EXPECT_FALSE(s.infiniteLight.present);
}

TEST(FlattenInfiniteLightTest, ConstantColorFormPopulatesLAndScale) {
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"rgb L\" [ 1 2 3 ] \"float scale\" [ 2 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[0], 1.0);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[1], 2.0);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[2], 3.0);
	EXPECT_DOUBLE_EQ(s.infiniteLight.scale, 2.0);
	EXPECT_TRUE(s.infiniteLight.imageFile.empty());
}

TEST(FlattenInfiniteLightTest, FilenameFormRecordsThePathButDoesNotDecodeIt) {
	// flatten() is filesystem-free by design (see the file comment) - decoding
	// filename into imagePixels is pbrt_load::loadFile()'s job, done AFTER
	// flatten() returns. This only pins the half flatten() itself owns.
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"string filename\" [ \"sky.exr\" ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	EXPECT_EQ(s.infiniteLight.imageFile, "sky.exr");
	EXPECT_EQ(s.infiniteLight.imageWidth, 0);
	EXPECT_EQ(s.infiniteLight.imageHeight, 0);
	EXPECT_TRUE(s.infiniteLight.imagePixels.empty());
}

TEST(FlattenInfiniteLightTest, RotationIsCarriedThroughInTheXform) {
	const FlatScene s = flattenSource(
		"Rotate 90 0 1 0\nLightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	// A 90 degree rotation about Y is not the identity matrix - if the CTM
	// were dropped (the old bug's shape) this would come back as identity
	// regardless of the Rotate directive above it.
	const pbrt_scene::Matrix4 &m = s.infiniteLight.xform;
	const pbrt_scene::Matrix4 id = pbrt_scene::Matrix4::identity();
	bool differsFromIdentity = false;
	for (int i = 0; i < 16; ++i)
		if (std::abs(m.m[i] - id.m[i]) > 1e-9) differsFromIdentity = true;
	EXPECT_TRUE(differsFromIdentity);
}

TEST(FlattenInfiniteLightTest, OnlyTheLastInfiniteLightWins) {
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n"
		"LightSource \"infinite\" \"rgb L\" [ 9 8 7 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[0], 9.0);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[1], 8.0);
	EXPECT_DOUBLE_EQ(s.infiniteLight.L[2], 7.0);
}

TEST(FlattenInfiniteLightTest, OtherLightKindsAreStillDroppedWithAWarning) {
	// Distant/point/spot are not carried through - only "infinite" is worth
	// the extra plumbing (see InfiniteLight's own comment on why).
	const FlatScene s = flattenSource(
		"LightSource \"point\" \"rgb I\" [ 1 1 1 ]\n");
	EXPECT_FALSE(s.infiniteLight.present);
	EXPECT_TRUE(warnedAbout(s, "point"));
}

// ===========================================================================
// Shape "bilinearmesh"
// ===========================================================================
// sportscar-area-lights.pbrt authors all 5 of its studio light panels as
// bilinear patches - dropping the shape (the old behaviour) loses 100% of
// that scene's light geometry and renders pure black.

TEST(FlattenBilinearMeshTest, FourPointFormPopulatesCorners) {
	const FlatScene s = flattenSource(
		"Shape \"bilinearmesh\" \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0 ]\n");
	ASSERT_EQ(s.bilinearPatches.size(), 1u);
	const pbrt_flatten::BilinearPatch &bp = s.bilinearPatches[0];
	EXPECT_DOUBLE_EQ(bp.p[0][0], 0.0); EXPECT_DOUBLE_EQ(bp.p[0][1], 0.0);
	EXPECT_DOUBLE_EQ(bp.p[1][0], 1.0); EXPECT_DOUBLE_EQ(bp.p[1][1], 0.0);
	EXPECT_DOUBLE_EQ(bp.p[2][0], 0.0); EXPECT_DOUBLE_EQ(bp.p[2][1], 1.0);
	EXPECT_DOUBLE_EQ(bp.p[3][0], 1.0); EXPECT_DOUBLE_EQ(bp.p[3][1], 1.0);
}

TEST(FlattenBilinearMeshTest, TransformIsBakedIntoEveryCorner) {
	const FlatScene s = flattenSource(
		"Translate 10 20 30\n"
		"Shape \"bilinearmesh\" \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0 ]\n");
	ASSERT_EQ(s.bilinearPatches.size(), 1u);
	const pbrt_flatten::BilinearPatch &bp = s.bilinearPatches[0];
	EXPECT_DOUBLE_EQ(bp.p[0][0], 10.0);
	EXPECT_DOUBLE_EQ(bp.p[0][1], 20.0);
	EXPECT_DOUBLE_EQ(bp.p[0][2], 30.0);
	EXPECT_DOUBLE_EQ(bp.p[1][0], 11.0) << "second corner translated too";
}

TEST(FlattenBilinearMeshTest, AreaLightSourceMarksItEmissive) {
	const FlatScene s = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"bilinearmesh\" \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0 ]\n"
		"AttributeEnd\n"
		"Shape \"bilinearmesh\" \"point3 P\" [ 0 0 5  1 0 5  0 1 5  1 1 5 ]\n");
	ASSERT_EQ(s.bilinearPatches.size(), 2u);
	EXPECT_GE(s.bilinearPatches[0].areaLight, 0) << "inside the AreaLightSource scope";
	EXPECT_EQ(s.bilinearPatches[1].areaLight, -1) << "outside it";
}

TEST(FlattenBilinearMeshTest, MultiPatchIndexedFormFallsThroughToTheGenericWarning) {
	// Only the single-patch "point3 P" (4 points) form is built - see
	// BilinearPatch's own comment on why the multi-patch "integer indices"
	// form is out of scope. It must not silently build a wrong/empty patch.
	const FlatScene s = flattenSource(
		"Shape \"bilinearmesh\" \"integer indices\" [ 0 1 2 3 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0  1 1 1  0 1 1 ]\n");
	EXPECT_TRUE(s.bilinearPatches.empty());
	EXPECT_TRUE(warnedAbout(s, "bilinearmesh"));
}

