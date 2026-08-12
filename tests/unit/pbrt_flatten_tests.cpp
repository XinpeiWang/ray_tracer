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
