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

TEST(FlattenTest, AreaLightFilenameAndTwoSidedAreParsed) {
	const FlatScene s = flattenSource(
		std::string("AttributeBegin\n"
					"  AreaLightSource \"diffuse\" \"string filename\" [ \"glow.png\" ]"
					" \"bool twosided\" [ true ]\n")
		+ kQuadMesh + "AttributeEnd\n");
	ASSERT_EQ(s.triangles[0].areaLight, 0);
	ASSERT_EQ(s.areaLights.size(), 1u);
	EXPECT_EQ(s.areaLights[0].filename, "glow.png");
	EXPECT_TRUE(s.areaLights[0].twoSided);
}

TEST(FlattenTest, AreaLightFilenameAndTwoSidedDefaultToUnsetAndFalse) {
	const FlatScene s = flattenSource(
		std::string("AttributeBegin\n  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n")
		+ kQuadMesh + "AttributeEnd\n");
	ASSERT_EQ(s.areaLights.size(), 1u);
	EXPECT_TRUE(s.areaLights[0].filename.empty());
	EXPECT_FALSE(s.areaLights[0].twoSided);
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

// ===========================================================================
// Disks / Cylinders
// ===========================================================================

TEST(FlattenTest, DiskParamsAreReadWithPbrtDefaults) {
	const FlatScene s = flattenSource("Shape \"disk\"\n");
	ASSERT_EQ(s.disks.size(), 1u);
	EXPECT_DOUBLE_EQ(s.disks[0].radius, 1.0);
	EXPECT_DOUBLE_EQ(s.disks[0].innerRadius, 0.0);
	EXPECT_DOUBLE_EQ(s.disks[0].height, 0.0);
	EXPECT_DOUBLE_EQ(s.disks[0].phiMaxDeg, 360.0);
}

TEST(FlattenTest, DiskParamsAreReadWhenGiven) {
	const FlatScene s = flattenSource(
		"Shape \"disk\" \"float radius\" [ 3 ] \"float innerradius\" [ 1 ] "
		"\"float height\" [ 2 ] \"float phimax\" [ 270 ]\n");
	ASSERT_EQ(s.disks.size(), 1u);
	EXPECT_DOUBLE_EQ(s.disks[0].radius, 3.0);
	EXPECT_DOUBLE_EQ(s.disks[0].innerRadius, 1.0);
	EXPECT_DOUBLE_EQ(s.disks[0].height, 2.0);
	EXPECT_DOUBLE_EQ(s.disks[0].phiMaxDeg, 270.0);
}

TEST(FlattenTest, DiskCarriesItsCTMUnbakedRatherThanBeingApproximated) {
	// Unlike Sphere (rotation-invariant, baked to world-space center+radius -
	// see NonUniformScaleOnASphereIsReportedNotSilentlyRounded above), a disk
	// is not rotation-invariant, so its xform is kept as-is for the CPU/GPU
	// builders to apply exactly at intersection time - see pbrt_flatten::
	// Disk's own comment for why. This just pins that the CTM in effect at
	// the Shape directive is what ends up on the flat struct.
	const FlatScene s = flattenSource("Translate 10 20 30\nShape \"disk\"\n");
	ASSERT_EQ(s.disks.size(), 1u);
	// Row-major affine: translation lives in column 3 of each row (m[3],
	// m[7], m[11] - see pbrt_scene::Matrix4's own layout, matching
	// transform_instance.h's apply_point()).
	EXPECT_DOUBLE_EQ(s.disks[0].xform[3], 10.0);
	EXPECT_DOUBLE_EQ(s.disks[0].xform[7], 20.0);
	EXPECT_DOUBLE_EQ(s.disks[0].xform[11], 30.0);
}

TEST(FlattenTest, CylinderParamsAreReadWithPbrtDefaults) {
	const FlatScene s = flattenSource("Shape \"cylinder\"\n");
	ASSERT_EQ(s.cylinders.size(), 1u);
	EXPECT_DOUBLE_EQ(s.cylinders[0].radius, 1.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].zMin, -1.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].zMax, 1.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].phiMaxDeg, 360.0);
}

TEST(FlattenTest, CylinderParamsAreReadWhenGiven) {
	const FlatScene s = flattenSource(
		"Shape \"cylinder\" \"float radius\" [ 2 ] \"float zmin\" [ -5 ] "
		"\"float zmax\" [ 5 ] \"float phimax\" [ 180 ]\n");
	ASSERT_EQ(s.cylinders.size(), 1u);
	EXPECT_DOUBLE_EQ(s.cylinders[0].radius, 2.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].zMin, -5.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].zMax, 5.0);
	EXPECT_DOUBLE_EQ(s.cylinders[0].phiMaxDeg, 180.0);
}

TEST(FlattenTest, DiskAndCylinderRespectMaterialAreaLightAndMedium) {
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"disk\"\n"
		"  Shape \"cylinder\"\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.disks.size(), 1u);
	ASSERT_EQ(s.cylinders.size(), 1u);
	EXPECT_EQ(s.disks[0].material, 0);
	EXPECT_EQ(s.disks[0].areaLight, 0);
	EXPECT_EQ(s.disks[0].medium, 0);
	EXPECT_EQ(s.cylinders[0].material, 0);
	EXPECT_EQ(s.cylinders[0].areaLight, 0);
	EXPECT_EQ(s.cylinders[0].medium, 0);
}

// ===========================================================================
// Curve
// ===========================================================================

namespace {
const char *kSingleSegmentCurve =
	"Shape \"curve\"\n"
	"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n";
} // namespace

TEST(FlattenTest, CurveParamsAreReadWithPbrtDefaults) {
	const FlatScene s = flattenSource(kSingleSegmentCurve);
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_EQ(s.curves[0].nSegments, 1);
	EXPECT_DOUBLE_EQ(s.curves[0].width0, 1.0);
	EXPECT_DOUBLE_EQ(s.curves[0].width1, 1.0);
	EXPECT_EQ(s.curves[0].curveType, "flat");
	ASSERT_EQ(s.curves[0].cp.size(), 12u);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[0], 0.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[9], 3.0);
}

TEST(FlattenTest, CurveWidthParamsAreReadWhenGiven) {
	const FlatScene s = flattenSource(
		"Shape \"curve\" \"string type\" [ \"cylinder\" ] "
		"\"float width0\" [ 0.2 ] \"float width1\" [ 0.05 ]\n"
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n");
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_DOUBLE_EQ(s.curves[0].width0, 0.2);
	EXPECT_DOUBLE_EQ(s.curves[0].width1, 0.05);
	EXPECT_EQ(s.curves[0].curveType, "cylinder");
}

TEST(FlattenTest, CurveWidthShorthandSetsBothEndpoints) {
	const FlatScene s = flattenSource(
		std::string("Shape \"curve\" \"float width\" [ 0.3 ]\n") +
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n");
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_DOUBLE_EQ(s.curves[0].width0, 0.3);
	EXPECT_DOUBLE_EQ(s.curves[0].width1, 0.3);
}

TEST(FlattenTest, CurveControlPointsAreTransformedByTheCTM) {
	const FlatScene s = flattenSource(std::string("Translate 10 20 30\n") + kSingleSegmentCurve);
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[0], 10.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[1], 20.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[2], 30.0);
	// Fourth (last) control point: (3,0,0) + translate.
	EXPECT_DOUBLE_EQ(s.curves[0].cp[9], 13.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[10], 20.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[11], 30.0);
}

TEST(FlattenTest, CurveWithSevenControlPointsSplitsIntoTwoSegments) {
	// (N-1)/3 = 2 segments; pbrt-v4's own Curve::Create splitting convention
	// (shapes.cpp) - segment 2 shares its first control point with segment
	// 1's last.
	const FlatScene s = flattenSource(
		"Shape \"curve\"\n"
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0  4 1 0  5 1 0  6 0 0 ]\n");
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_EQ(s.curves[0].nSegments, 2);
	ASSERT_EQ(s.curves[0].cp.size(), 24u);
	// Segment 0's last point (index 3) and segment 1's first point (index 4)
	// are both (3,0,0), the shared endpoint.
	EXPECT_DOUBLE_EQ(s.curves[0].cp[9], 3.0);
	EXPECT_DOUBLE_EQ(s.curves[0].cp[12], 3.0);
}

TEST(FlattenTest, CurveWithInvalidControlPointCountWarnsAndSkips) {
	// 5 points: (5-1) % 3 != 0.
	const FlatScene s = flattenSource(
		"Shape \"curve\"\n"
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0  4 0 0 ]\n");
	EXPECT_EQ(s.curves.size(), 0u);
	EXPECT_TRUE(warnedAbout(s, "control points"));
}

TEST(FlattenTest, CurveWithUnsupportedDegreeWarnsAndSkips) {
	const FlatScene s = flattenSource(
		std::string("Shape \"curve\" \"integer degree\" [ 2 ]\n") +
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0 ]\n");
	EXPECT_EQ(s.curves.size(), 0u);
	EXPECT_TRUE(warnedAbout(s, "degree"));
}

TEST(FlattenTest, RibbonCurveWithoutNormalsWarnsAndSkips) {
	const FlatScene s = flattenSource(
		std::string("Shape \"curve\" \"string type\" [ \"ribbon\" ]\n") +
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n");
	EXPECT_EQ(s.curves.size(), 0u);
	EXPECT_TRUE(warnedAbout(s, "normal"));
}

TEST(FlattenTest, RibbonCurveWithNormalsBuilds) {
	const FlatScene s = flattenSource(
		"Shape \"curve\" \"string type\" [ \"ribbon\" ]\n"
		"  \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n"
		"  \"normal N\" [ 0 0 1  0 0 1 ]\n");
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_EQ(s.curves[0].curveType, "ribbon");
	ASSERT_EQ(s.curves[0].n.size(), 6u);
	EXPECT_DOUBLE_EQ(s.curves[0].n[2], 1.0);
	EXPECT_DOUBLE_EQ(s.curves[0].n[5], 1.0);
}

TEST(FlattenTest, CurveRespectsMaterialAndAreaLight) {
	const FlatScene s = flattenSource(
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"curve\"\n"
		"    \"point3 P\" [ 0 0 0  1 1 0  2 1 0  3 0 0 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.curves.size(), 1u);
	EXPECT_EQ(s.curves[0].material, 0);
	EXPECT_EQ(s.curves[0].areaLight, 0);
}

// ===========================================================================
// MakeNamedMedium / MediumInterface
// ===========================================================================

TEST(FlattenTest, MediumInterfaceAttachesTheNamedMediumToTheSphere) {
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.2 0.3 ] \"rgb sigma_s\" [ 1 2 3 ] \"float scale\" [ 2 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	ASSERT_EQ(s.media.size(), 1u);
	ASSERT_EQ(s.spheres[0].medium, 0)
		<< "the sphere should resolve to media[0], the only declared medium";

	// scale multiplies both coefficients (pbrt-v4 semantics), so 0.1*2=0.2 etc.
	EXPECT_DOUBLE_EQ(s.media[0].sigma_a[0], 0.2);
	EXPECT_DOUBLE_EQ(s.media[0].sigma_a[1], 0.4);
	EXPECT_DOUBLE_EQ(s.media[0].sigma_a[2], 0.6);
	EXPECT_DOUBLE_EQ(s.media[0].sigma_s[0], 2.0);
	EXPECT_DOUBLE_EQ(s.media[0].sigma_s[1], 4.0);
	EXPECT_DOUBLE_EQ(s.media[0].sigma_s[2], 6.0);
}

TEST(FlattenTest, SphereWithNoMediumInterfaceIsVacuum) {
	const FlatScene s = flattenSource("Shape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_EQ(s.spheres[0].medium, -1);
}

TEST(FlattenTest, MediumInterfaceIsScopedByAttributeEnd) {
	// A sphere declared AFTER AttributeEnd must not inherit the medium the
	// first sphere picked up inside the block - the same scoping already
	// proven for materialIndex, now exercised for insideMedium.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_EQ(s.spheres.size(), 2u);
	EXPECT_EQ(s.spheres[0].medium, 0);
	EXPECT_EQ(s.spheres[1].medium, -1)
		<< "the second sphere is outside the AttributeBegin/End block";
}

TEST(FlattenTest, UnresolvedMediumNameIsVacuumAndWarns) {
	const FlatScene s = flattenSource(
		"AttributeBegin\n"
		"  MediumInterface \"ghost\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_EQ(s.spheres[0].medium, -1);
	EXPECT_TRUE(warnedAbout(s, "ghost"));
}

TEST(FlattenTest, UnsupportedMediumTypeFallsBackToHomogeneousAndWarns) {
	// "nanovdb" (or any other type pbrt-v4 supports that this loader
	// doesn't) still falls back to homogeneous with a warning - the
	// generic case cloud/rgbgrid/uniformgrid all now opt out of.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"g\" \"string type\" [ \"nanovdb\" ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	EXPECT_EQ(s.media[0].type, "homogeneous");
	EXPECT_TRUE(warnedAbout(s, "nanovdb"));
}

TEST(FlattenTest, UniformgridParsesDensityArray) {
	// nx=2,ny=1,nz=1: 2 voxels, "float density" holds exactly 2 numbers,
	// one per voxel.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"g\" \"string type\" [ \"uniformgrid\" ]\n"
		"  \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"  \"float density\" [ 0.25 0.75 ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	const Medium &m = s.media[0];
	EXPECT_EQ(m.type, "uniformgrid");
	EXPECT_EQ(m.nx, 2);
	ASSERT_EQ(m.gridDensity.size(), 2u);
	EXPECT_DOUBLE_EQ(m.gridDensity[0], 0.25);
	EXPECT_DOUBLE_EQ(m.gridDensity[1], 0.75);
}

TEST(FlattenTest, UniformgridMissingDensityIsEmptyAndWarns) {
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"g\" \"string type\" [ \"uniformgrid\" ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	EXPECT_EQ(s.media[0].type, "uniformgrid");
	EXPECT_TRUE(s.media[0].gridDensity.empty());
	EXPECT_TRUE(warnedAbout(s, "density"));
}

TEST(FlattenTest, UniformgridWrongDensityLengthIsEmptyAndWarns) {
	// nx=2,ny=1,nz=1 expects 2 numbers; giving only 1 must not be silently
	// truncated/padded - dropped entirely, with a warning.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"g\" \"string type\" [ \"uniformgrid\" ]\n"
		"  \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"  \"float density\" [ 0.5 ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	EXPECT_TRUE(s.media[0].gridDensity.empty());
	EXPECT_TRUE(warnedAbout(s, "g"));
}

TEST(FlattenTest, CloudMediumComputesWorldAabbAndInverseTransform) {
	// Default medium-space bounds are the unit cube; Translate+Scale places
	// it in world space. World box: (0,0,0)->(10,20,30), (1,1,1)->(12,22,32).
	const FlatScene s = flattenSource(
		"AttributeBegin\n"
		"  Translate 10 20 30\n"
		"  Scale 2 2 2\n"
		"  MakeNamedMedium \"puff\" \"string type\" [ \"cloud\" ]\n"
		"    \"rgb sigma_s\" [ 1 1 1 ] \"float density\" [ 0.5 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.media.size(), 1u);
	const Medium &m = s.media[0];
	EXPECT_EQ(m.type, "cloud");
	EXPECT_DOUBLE_EQ(m.p0[0], 0.0);
	EXPECT_DOUBLE_EQ(m.p1[0], 1.0);
	EXPECT_DOUBLE_EQ(m.density, 0.5);
	EXPECT_DOUBLE_EQ(m.wispiness, 1.0);
	EXPECT_DOUBLE_EQ(m.frequency, 5.0)
		<< "pbrt-v4's real default, not this codebase's own E2 showcase scene's 4.0";

	EXPECT_DOUBLE_EQ(m.worldMin[0], 10.0);
	EXPECT_DOUBLE_EQ(m.worldMin[1], 20.0);
	EXPECT_DOUBLE_EQ(m.worldMin[2], 30.0);
	EXPECT_DOUBLE_EQ(m.worldMax[0], 12.0);
	EXPECT_DOUBLE_EQ(m.worldMax[1], 22.0);
	EXPECT_DOUBLE_EQ(m.worldMax[2], 32.0);

	// world_to_medium = inverse(Translate(10,20,30)*Scale(2,2,2)):
	// mat = diag(0.5,0.5,0.5), translate = -(10,20,30)*0.5.
	EXPECT_DOUBLE_EQ(m.toMediumMat[0], 0.5);
	EXPECT_DOUBLE_EQ(m.toMediumMat[4], 0.5);
	EXPECT_DOUBLE_EQ(m.toMediumMat[8], 0.5);
	EXPECT_DOUBLE_EQ(m.toMediumTranslate[0], -5.0);
	EXPECT_DOUBLE_EQ(m.toMediumTranslate[1], -10.0);
	EXPECT_DOUBLE_EQ(m.toMediumTranslate[2], -15.0);
}

TEST(FlattenTest, CloudMediumWithNonzeroSigmaAWarns) {
	// cloud_medium_hittable/MaterialType::CloudMedium always force sigma_a
	// to 0 (pure scattering) - a scene that set a real one silently loses
	// it, which is worth a warning (see pbrt_flatten.h's own comment).
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"puff\" \"string type\" [ \"cloud\" ]\n"
		"  \"rgb sigma_a\" [ 0.1 0.1 0.1 ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	EXPECT_TRUE(warnedAbout(s, "sigma_a"));
}

TEST(FlattenTest, RgbGridDeinterleavesFlatTripleArray) {
	// nx=2,ny=1,nz=1: 2 voxels, "rgb sigma_s" holds 2*3=6 numbers, one RGB
	// triple per voxel (voxel 0 = (1,2,3), voxel 1 = (4,5,6)).
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"nebula\" \"string type\" [ \"rgbgrid\" ]\n"
		"  \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"  \"rgb sigma_s\" [ 1 2 3  4 5 6 ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	const Medium &m = s.media[0];
	EXPECT_EQ(m.type, "rgbgrid");
	ASSERT_EQ(m.sigma_s_r.size(), 2u);
	ASSERT_EQ(m.sigma_s_g.size(), 2u);
	ASSERT_EQ(m.sigma_s_b.size(), 2u);
	EXPECT_DOUBLE_EQ(m.sigma_s_r[0], 1.0);
	EXPECT_DOUBLE_EQ(m.sigma_s_g[0], 2.0);
	EXPECT_DOUBLE_EQ(m.sigma_s_b[0], 3.0);
	EXPECT_DOUBLE_EQ(m.sigma_s_r[1], 4.0);
	EXPECT_DOUBLE_EQ(m.sigma_s_g[1], 5.0);
	EXPECT_DOUBLE_EQ(m.sigma_s_b[1], 6.0);
	EXPECT_TRUE(m.sigma_a_r.empty())
		<< "sigma_a was never given, so its channel vectors stay absent";
}

TEST(FlattenTest, RgbGridWrongArrayLengthIsDroppedWithWarning) {
	// nx=2,ny=1,nz=1 expects 6 numbers; giving only 3 (one voxel) must not
	// be silently truncated/padded - it's dropped entirely, with a warning.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"nebula\" \"string type\" [ \"rgbgrid\" ]\n"
		"  \"integer nx\" [ 2 ] \"integer ny\" [ 1 ] \"integer nz\" [ 1 ]\n"
		"  \"rgb sigma_s\" [ 1 2 3 ]\n");
	ASSERT_EQ(s.media.size(), 1u);
	EXPECT_TRUE(s.media[0].sigma_s_r.empty());
	EXPECT_TRUE(warnedAbout(s, "nebula"));
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
	// "cone" - not "cylinder"/"disk", which this loader now supports (see
	// FlattenTest.MediumInterfaceAttachesTheNamedMediumToTheSphere's sibling
	// tests above and pbrt_cpu_builder_tests.cpp's disk/cylinder coverage).
	const FlatScene s = flattenSource("Shape \"cone\" \"float radius\" [ 1 ]\n");
	EXPECT_TRUE(s.empty());
	EXPECT_TRUE(warnedAbout(s, "cone"));
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

TEST(FlattenTest, PixelFilterDefaultsToGaussian) {
	// pbrt-v4's real default (see pbrt_flatten::PixelFilter's own comment) -
	// not box/triangle/mitchell.
	const FlatScene s = flattenSource("WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_EQ(s.filter.kind, "gaussian");
	EXPECT_DOUBLE_EQ(s.filter.sigma, 0.5);
}

TEST(FlattenTest, PixelFilterParamsAreCarriedThrough) {
	const FlatScene s = flattenSource(
		std::string("PixelFilter \"mitchell\" \"float B\" [ 0.2 ] \"float C\" [ 0.4 ]\nWorldBegin\n")
		+ kQuadMesh);
	EXPECT_EQ(s.filter.kind, "mitchell");
	EXPECT_DOUBLE_EQ(s.filter.B, 0.2);
	EXPECT_DOUBLE_EQ(s.filter.C, 0.4);
}

TEST(FlattenTest, SincPixelFilterTauIsCarriedThrough) {
	const FlatScene s = flattenSource(
		std::string("PixelFilter \"sinc\" \"float tau\" [ 2.5 ]\nWorldBegin\n") + kQuadMesh);
	EXPECT_EQ(s.filter.kind, "sinc");
	EXPECT_DOUBLE_EQ(s.filter.tau, 2.5);
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
	// A material rendered as diffuse looks plausible and is wrong, so the
	// substitution has to be announced. "subsurface" used to be the example
	// here, but it is now a real, CPU-supported MaterialKind (see
	// material_pbrt.h's `class subsurface` and camera.h::sample_bssrdf_exit) -
	// "hair" (still genuinely unimplemented on both backends) takes its place
	// as the still-Unsupported example.
	const FlatScene s = flattenSource(
		"Material \"hair\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Unsupported);
	EXPECT_EQ(s.materials[0].pbrtType, "hair");
	EXPECT_TRUE(warnedAbout(s, "hair"));
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
// Material "mix"
// ---------------------------------------------------------------------------
// A stochastic blend of two OTHER named materials (pbrt-v4 MixMaterial).
// src/TheRestOfYourLife/material_pbrt.h's `class mix_material` already backs
// this generically (added earlier for SPPM support, but not SPPM-specific -
// see MaterialKind::Mix's own comment) - these pin the name -> index
// resolution flatten() has to do to bridge onto it: forward AND backward
// references, and the "malformed mix" cases downgrading to the existing
// diffuse-fallback tier rather than a new, separate failure mode.

TEST(FlattenMaterialTest, MixResolvesBothNamedSubMaterials) {
	const FlatScene s = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ] \"rgb reflectance\" [ 1 0 0 ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"conductor\" ] \"rgb reflectance\" [ 0 0 1 ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ] \"float amount\" [ 0.25 ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 3u);
	const Material &mix = s.materials[2];
	EXPECT_EQ(mix.kind, MaterialKind::Mix);
	ASSERT_GE(mix.mixMaterialA, 0);
	ASSERT_GE(mix.mixMaterialB, 0);
	EXPECT_EQ(s.materials[mix.mixMaterialA].kind, MaterialKind::Diffuse);
	EXPECT_EQ(s.materials[mix.mixMaterialB].kind, MaterialKind::Conductor);
	EXPECT_DOUBLE_EQ(mix.mixWeight, 0.25);
	EXPECT_FALSE(warnedAbout(s, "not supported"));
}

TEST(FlattenMaterialTest, MixCanForwardReferenceAMaterialDeclaredLater) {
	// pbrt-v4 resolves MixMaterial's "materials" names against the whole
	// scene, not just what came textually before the mix directive itself -
	// this is the reason flatten() builds its name->index map in a pre-pass
	// rather than incrementally during the main per-material loop.
	const FlatScene s = flattenSource(
		"Material \"mix\" \"string materials\" [ \"later\" \"also_later\" ]\n"
		"MakeNamedMaterial \"later\" \"string type\" [ \"diffuse\" ]\n"
		"MakeNamedMaterial \"also_later\" \"string type\" [ \"conductor\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 3u);
	const Material &mix = s.materials[0];
	EXPECT_EQ(mix.kind, MaterialKind::Mix);
	ASSERT_GE(mix.mixMaterialA, 0);
	ASSERT_GE(mix.mixMaterialB, 0);
	EXPECT_EQ(s.materials[mix.mixMaterialA].kind, MaterialKind::Diffuse);
	EXPECT_EQ(s.materials[mix.mixMaterialB].kind, MaterialKind::Conductor);
}

TEST(FlattenMaterialTest, MixAmountDefaultsToOneHalf) {
	const FlatScene s = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"diffuse\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 3u);
	EXPECT_DOUBLE_EQ(s.materials[2].mixWeight, 0.5);
}

TEST(FlattenMaterialTest, MixWithAnUnknownNameFallsBackToDiffuseAndWarns) {
	const FlatScene s = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"does_not_exist\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 2u);
	EXPECT_EQ(s.materials[1].kind, MaterialKind::Unsupported);
	EXPECT_TRUE(warnedAbout(s, "mix"));
}

TEST(FlattenMaterialTest, MixWithFewerThanTwoNamesFallsBackToDiffuseAndWarns) {
	const FlatScene s = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 2u);
	EXPECT_EQ(s.materials[1].kind, MaterialKind::Unsupported);
	EXPECT_TRUE(warnedAbout(s, "mix"));
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
	// point/spot/distant/goniometric/projection all now carry through (see
	// the FlattenPunctualLightTest section below) - only a genuinely unknown
	// kind still hits this fallback.
	const FlatScene s = flattenSource(
		"LightSource \"bogus\" \"rgb I\" [ 1 1 1 ]\n");
	EXPECT_FALSE(s.infiniteLight.present);
	EXPECT_TRUE(s.punctualLights.empty());
	EXPECT_TRUE(warnedAbout(s, "bogus"));
}

// ===========================================================================
// LightSource "point"/"spot"/"distant"/"goniometric"/"projection"
// ===========================================================================
// pbrt-v4's five punctual (delta-distribution) light kinds. Rendering support
// for all five already exists on both backends, proven by this codebase's own
// C2-C6 showcase scenes (scenes_advanced.h) - these tests pin the
// parsing/bridging half: parameter defaults, the from/to -> position/
// direction derivation, and that the CTM is honoured the same way
// InfiniteLight::xform already is.

TEST(FlattenPunctualLightTest, PointLightReadsPositionIntensityAndScale) {
	const FlatScene s = flattenSource(
		"LightSource \"point\" \"point3 from\" [ 1 2 3 ] \"rgb I\" [ 0.5 0.6 0.7 ] "
		"\"float scale\" [ 10 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_EQ(pl.kind, PunctualLightKind::Point);
	EXPECT_DOUBLE_EQ(pl.pos[0], 1.0);
	EXPECT_DOUBLE_EQ(pl.pos[1], 2.0);
	EXPECT_DOUBLE_EQ(pl.pos[2], 3.0);
	EXPECT_DOUBLE_EQ(pl.intensity[0], 0.5);
	EXPECT_DOUBLE_EQ(pl.intensity[1], 0.6);
	EXPECT_DOUBLE_EQ(pl.intensity[2], 0.7);
	EXPECT_DOUBLE_EQ(pl.scale, 10.0);
}

TEST(FlattenPunctualLightTest, PointLightDefaultsMatchPbrt) {
	const FlatScene s = flattenSource("LightSource \"point\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_DOUBLE_EQ(pl.pos[0], 0.0);
	EXPECT_DOUBLE_EQ(pl.pos[1], 0.0);
	EXPECT_DOUBLE_EQ(pl.pos[2], 0.0);
	EXPECT_DOUBLE_EQ(pl.intensity[0], 1.0);
	EXPECT_DOUBLE_EQ(pl.scale, 1.0);
}

TEST(FlattenPunctualLightTest, PointLightPositionHonoursTheCtm) {
	// Same guard InfiniteLight's own RotationIsCarriedThroughInTheXform test
	// makes: a light's placement dropped the CTM used to render in the wrong
	// spot with nothing on screen to explain why.
	const FlatScene s = flattenSource(
		"Translate 10 20 30\nLightSource \"point\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_DOUBLE_EQ(s.punctualLights[0].pos[0], 10.0);
	EXPECT_DOUBLE_EQ(s.punctualLights[0].pos[1], 20.0);
	EXPECT_DOUBLE_EQ(s.punctualLights[0].pos[2], 30.0);
}

TEST(FlattenPunctualLightTest, MultiplePunctualLightsAreAllKept) {
	// Unlike "infinite", where only the last one wins (see
	// FlattenInfiniteLightTest::OnlyTheLastInfiniteLightWins), a scene with
	// three spotlights genuinely wants all three.
	const FlatScene s = flattenSource(
		"LightSource \"point\"\nLightSource \"point\"\nLightSource \"point\"\n");
	EXPECT_EQ(s.punctualLights.size(), 3u);
}

TEST(FlattenPunctualLightTest, SpotLightDerivesAxisFromFromAndTo) {
	const FlatScene s = flattenSource(
		"LightSource \"spot\" \"point3 from\" [ 0 0 0 ] \"point3 to\" [ 0 0 5 ] "
		"\"float coneangle\" [ 40 ] \"float conedeltaangle\" [ 10 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_EQ(pl.kind, PunctualLightKind::Spot);
	EXPECT_DOUBLE_EQ(pl.pos[0], 0.0);
	EXPECT_DOUBLE_EQ(pl.pos[1], 0.0);
	EXPECT_DOUBLE_EQ(pl.pos[2], 0.0);
	EXPECT_NEAR(pl.dir[0], 0.0, 1e-9);
	EXPECT_NEAR(pl.dir[1], 0.0, 1e-9);
	EXPECT_NEAR(pl.dir[2], 1.0, 1e-9);
	EXPECT_DOUBLE_EQ(pl.coneAngleDeg, 40.0);
	EXPECT_DOUBLE_EQ(pl.falloffStartAngleDeg, 30.0);   // coneangle - conedeltaangle
}

TEST(FlattenPunctualLightTest, SpotLightDefaultConeMatchesPbrt) {
	const FlatScene s = flattenSource("LightSource \"spot\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_DOUBLE_EQ(pl.coneAngleDeg, 30.0);
	EXPECT_DOUBLE_EQ(pl.falloffStartAngleDeg, 25.0);   // 30 - 5, pbrt-v4's own default delta
	// Default "to" is (0,0,1): the axis points straight down +z.
	EXPECT_NEAR(pl.dir[2], 1.0, 1e-9);
}

TEST(FlattenPunctualLightTest, SpotLightFalloffStartCanGoNegative) {
	// conedeltaangle larger than coneangle is legal pbrt-v4, not malformed:
	// pbrt-v4's own SpotLight::Create passes coneangle - conedeltaangle
	// through unclamped, and cos() being even means a negative start angle
	// still produces a real, meaningful full-intensity core (cos(-80) ==
	// cos(80)) rather than collapsing to "no core at all" the way clamping
	// to 0 would.
	const FlatScene s = flattenSource(
		"LightSource \"spot\" \"float coneangle\" [ 10 ] \"float conedeltaangle\" [ 90 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_DOUBLE_EQ(s.punctualLights[0].falloffStartAngleDeg, -80.0);
}

TEST(FlattenPunctualLightTest, DistantLightPointsFromTowardFrom) {
	// dir must be `wi` (toward the light) - see PunctualLight::dir's own
	// comment. With "from" above "to", the light sits in +y, so wi should
	// point in +y too.
	const FlatScene s = flattenSource(
		"LightSource \"distant\" \"point3 from\" [ 0 10 0 ] \"point3 to\" [ 0 0 0 ] "
		"\"rgb L\" [ 2 2 2 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_EQ(pl.kind, PunctualLightKind::Distant);
	EXPECT_NEAR(pl.dir[0], 0.0, 1e-9);
	EXPECT_NEAR(pl.dir[1], 1.0, 1e-9);
	EXPECT_NEAR(pl.dir[2], 0.0, 1e-9);
	EXPECT_DOUBLE_EQ(pl.intensity[0], 2.0);
}

TEST(FlattenPunctualLightTest, DistantLightDefaultDirectionMatchesPbrt) {
	// Default from=(0,0,0), to=(0,0,1): the light travels in +z, so wi
	// (toward the light) is -z.
	const FlatScene s = flattenSource("LightSource \"distant\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_NEAR(s.punctualLights[0].dir[2], -1.0, 1e-9);
}

TEST(FlattenPunctualLightTest, GoniometricLightWithNoFilenameIsIsotropicAndUnwarned) {
	const FlatScene s = flattenSource(
		"LightSource \"goniometric\" \"rgb I\" [ 3 3 3 ] \"float scale\" [ 5 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_EQ(pl.kind, PunctualLightKind::Goniometric);
	EXPECT_DOUBLE_EQ(pl.intensity[0], 3.0);
	EXPECT_DOUBLE_EQ(pl.scale, 5.0);
	EXPECT_FALSE(pl.hadImageFilename);
	EXPECT_FALSE(warnedAbout(s, "goniometric"));
}

TEST(FlattenPunctualLightTest, GoniometricLightWithFilenameWarnsAndFallsBackToIsotropic) {
	const FlatScene s = flattenSource(
		"LightSource \"goniometric\" \"string filename\" [ \"profile.exr\" ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_TRUE(s.punctualLights[0].hadImageFilename);
	EXPECT_TRUE(warnedAbout(s, "goniometric"));
	EXPECT_TRUE(warnedAbout(s, "profile.exr"));
}

TEST(FlattenPunctualLightTest, GoniometricLightRotationIsCarriedThroughInWorldToLight) {
	const FlatScene s = flattenSource(
		"Rotate 90 0 1 0\nLightSource \"goniometric\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const double *w2l = s.punctualLights[0].worldToLight;
	const double identity[9] = {1,0,0, 0,1,0, 0,0,1};
	bool differsFromIdentity = false;
	for (int i = 0; i < 9; ++i)
		if (std::abs(w2l[i] - identity[i]) > 1e-9) differsFromIdentity = true;
	EXPECT_TRUE(differsFromIdentity);
}

TEST(FlattenPunctualLightTest, ProjectionLightReadsFovAndScale) {
	const FlatScene s = flattenSource(
		"LightSource \"projection\" \"float fov\" [ 25 ] \"float scale\" [ 8 ] "
		"\"string filename\" [ \"slide.exr\" ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const PunctualLight &pl = s.punctualLights[0];
	EXPECT_EQ(pl.kind, PunctualLightKind::Projection);
	EXPECT_DOUBLE_EQ(pl.fovDeg, 25.0);
	EXPECT_DOUBLE_EQ(pl.scale, 8.0);
	EXPECT_TRUE(pl.hadImageFilename);
	EXPECT_TRUE(warnedAbout(s, "projection"));
}

TEST(FlattenPunctualLightTest, ProjectionLightDefaultFovMatchesPbrt) {
	const FlatScene s = flattenSource("LightSource \"projection\"\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_DOUBLE_EQ(s.punctualLights[0].fovDeg, 90.0);
	// pbrt-v4 requires "filename"; a scene without one still renders (as a
	// uniform beam) rather than being dropped outright, but is still
	// worth a warning since it is itself malformed.
	EXPECT_TRUE(warnedAbout(s, "projection"));
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

