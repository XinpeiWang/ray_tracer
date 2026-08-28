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

TEST(FlattenTest, TrianglemeshWithNoUVLeavesHasUVsFalse) {
	// pbrt-v4's own real "no uv given" default is a fixed (0,0)/(1,0)/(1,1)
	// per triangle CORNER, not shared across faces sharing a vertex -
	// deliberately not synthesized (see pbrt_flatten::Triangle::uv's own
	// comment: it would silently inflate vertex-dedup counts at any shared
	// vertex, a real cost paid by scenes that never read UV at all). Both
	// builders' own barycentric fallback covers the "solid black on GPU"
	// bug this loader used to have instead.
	const FlatScene s = flattenSource(kQuadMesh);
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_FALSE(s.triangles[0].hasUVs);
	EXPECT_FALSE(s.triangles[1].hasUVs);
}

TEST(FlattenTest, TrianglemeshWithRealUVIsThreadedThrough) {
	const FlatScene s = flattenSource(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n"
		"  \"point2 uv\" [ 0.1 0.2   0.3 0.4   0.5 0.6   0.7 0.8 ]\n");
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_TRUE(s.triangles[0].hasUVs);
	// Face 0 = indices (0,1,2) -> uv pairs 0,1,2.
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[0], 0.1); EXPECT_DOUBLE_EQ(s.triangles[0].uv[1], 0.2);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[2], 0.3); EXPECT_DOUBLE_EQ(s.triangles[0].uv[3], 0.4);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[4], 0.5); EXPECT_DOUBLE_EQ(s.triangles[0].uv[5], 0.6);
	// Face 1 = indices (0,2,3) -> uv pairs 0,2,3.
	EXPECT_DOUBLE_EQ(s.triangles[1].uv[0], 0.1); EXPECT_DOUBLE_EQ(s.triangles[1].uv[1], 0.2);
	EXPECT_DOUBLE_EQ(s.triangles[1].uv[2], 0.5); EXPECT_DOUBLE_EQ(s.triangles[1].uv[3], 0.6);
	EXPECT_DOUBLE_EQ(s.triangles[1].uv[4], 0.7); EXPECT_DOUBLE_EQ(s.triangles[1].uv[5], 0.8);
}

TEST(FlattenTest, TrianglemeshUVIsNotSpatiallyTransformedByTheCTM) {
	// A UV pair isn't a world-space point - Translate/Scale/Rotate on the
	// shape must never touch it, unlike "point3 P".
	const FlatScene s = flattenSource(
		"Translate 10 20 30\nScale 2 2 2\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  \"point2 uv\" [ 0.25 0.75   0.1 0.2   0.9 0.4 ]\n");
	ASSERT_EQ(s.triangles.size(), 1u);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[0], 0.25);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[1], 0.75);
}

TEST(FlattenTest, TrianglemeshWithTooFewUVPairsWarnsAndIgnoresUV) {
	const FlatScene s = flattenSource(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n"
		"  \"point2 uv\" [ 0.1 0.2   0.3 0.4 ]\n");   // only 2 pairs, needs 4
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_TRUE(warnedAbout(s, "uv"));
	// Same "refused wholesale" treatment as too-few normals - hasUVs stays
	// false, same as no "uv" param at all.
	EXPECT_FALSE(s.triangles[0].hasUVs);
}

TEST(FlattenTest, LoopsubdivDoesNotThreadUV) {
	// UV parsing is scoped to trianglemesh only - loopsubdiv doesn't thread
	// it through this loader at all yet (a separate, smaller, documented
	// gap); its own "point3 P"/"integer indices" parsing is entirely
	// separate code from trianglemesh's, so this just confirms hasUVs stays
	// at its default false rather than somehow picking up stale state.
	const FlatScene s = flattenSource(
		"Shape \"loopsubdiv\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ] \"integer levels\" [ 0 ]\n");
	ASSERT_EQ(s.triangles.size(), 2u);
	EXPECT_FALSE(s.triangles[0].hasUVs);
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

TEST(FlattenTest, SphereWithNoClippingParamsIsNotMarkedClipped) {
	// Every existing scene must leave Sphere::clipped false - the CPU builder
	// keys off exactly this to keep using the plain baked center/radius path.
	const FlatScene s = flattenSource("Shape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_FALSE(s.spheres[0].clipped);
}

TEST(FlattenTest, SphereZMinZMaxMarksItClippedAndCarriesObjectSpaceValues) {
	const FlatScene s = flattenSource(
		"Translate 1 2 3\n"
		"Shape \"sphere\" \"float radius\" [ 2 ] \"float zmin\" [ -1 ] \"float zmax\" [ 1 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(s.spheres[0].clipped);
	EXPECT_DOUBLE_EQ(s.spheres[0].radiusLocal, 2.0)
		<< "object-space, matching the declared radius directly - not scaled";
	EXPECT_DOUBLE_EQ(s.spheres[0].zMin, -1.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].zMax, 1.0);
	// The object-to-world transform (needed once clipping breaks rotation
	// invariance) must carry the real translation, not identity.
	EXPECT_DOUBLE_EQ(s.spheres[0].xform[3], 1.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].xform[11], 3.0);
}

TEST(FlattenTest, SphereZMinZMaxAreReorderedWhenAuthoredSwapped) {
	// Real pbrt-v4 tolerates zmin/zmax given in either order (Sphere::Create
	// does Clamp(min(zmin,zmax),-r,r)/Clamp(max(zmin,zmax),-r,r)) - a scene
	// (or a converter) that writes them swapped must still produce the same
	// cap, not an empty [zMin,zMax] range that clips away the entire sphere.
	const FlatScene s = flattenSource(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 1 ] \"float zmax\" [ -1 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_FALSE(s.spheres[0].clipped)
		<< "reordered to (-1,1), which exactly matches the full-sphere default";
	EXPECT_DOUBLE_EQ(s.spheres[0].zMin, -1.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].zMax, 1.0);
}

TEST(FlattenTest, SphereZMinZMaxAreClampedToTheRadius) {
	// A scene that writes an out-of-range zmax (e.g. authored against the
	// wrong radius after an edit) must not produce a [zMin,zMax] wider than
	// the sphere itself - real pbrt-v4 clamps to [-r,r].
	const FlatScene s = flattenSource(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float zmax\" [ 5 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_FALSE(s.spheres[0].clipped) << "clamped to 1.0, matching the full-sphere default";
	EXPECT_DOUBLE_EQ(s.spheres[0].zMax, 1.0);
}

TEST(FlattenTest, SpherePhiMaxAloneAlsoMarksItClipped) {
	const FlatScene s = flattenSource(
		"Shape \"sphere\" \"float radius\" [ 1 ] \"float phimax\" [ 90 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(s.spheres[0].clipped);
	EXPECT_DOUBLE_EQ(s.spheres[0].phiMaxDeg, 90.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].zMin, -1.0) << "zmin/zmax keep their pbrt defaults";
	EXPECT_DOUBLE_EQ(s.spheres[0].zMax, 1.0);
}

TEST(FlattenTest, SphereZMinZMaxMatchingTheFullSphereIsNotMarkedClipped) {
	// A scene that spells out the pbrt defaults explicitly (zmin=-radius,
	// zmax=radius, phimax=360) is still a full sphere - the equality check
	// must treat this the same as omitting the params entirely.
	const FlatScene s = flattenSource(
		"Shape \"sphere\" \"float radius\" [ 3 ] \"float zmin\" [ -3 ] "
		"\"float zmax\" [ 3 ] \"float phimax\" [ 360 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_FALSE(s.spheres[0].clipped);
}

TEST(FlattenTest, ClippedSphereWithMediumInterfaceWarnsButKeepsTheMediumFieldForGpu) {
	// A clipped sphere is an open shell (a hole where the clip cut it away),
	// which CPU's constant_medium can't bound (requires a watertight
	// boundary - two ordered hits per ray) - see sphere_clipped_hittable.h's
	// own comment. But GPU always renders every sphere (clipped or not) as
	// the same full, closed, baked-radius shape, which HAS no such problem
	// - so `medium` itself must stay the real resolved index (for GPU's own
	// independent read of it); only the new, CPU-only
	// `cpuMediumUnsupported` flag records the CPU-specific limitation,
	// consulted by pbrt_cpu_builder.h alone.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.2 0.3 ] \"rgb sigma_s\" [ 1 2 3 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(s.spheres[0].clipped);
	EXPECT_EQ(s.spheres[0].medium, 0) << "GPU still needs the real index";
	EXPECT_TRUE(s.spheres[0].cpuMediumUnsupported);
	EXPECT_TRUE(warnedAbout(s, "clipped sphere"));
}

TEST(FlattenTest, UnclippedSphereWithMediumInterfaceStillWorksNormally) {
	// Regression guard for the fix above: the new clipped-specific gate must
	// not accidentally catch the ordinary, already-working unclipped case.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"  \"rgb sigma_a\" [ 0.1 0.2 0.3 ] \"rgb sigma_s\" [ 1 2 3 ]\n"
		"AttributeBegin\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"AttributeEnd\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_FALSE(s.spheres[0].clipped);
	EXPECT_EQ(s.spheres[0].medium, 0);
	EXPECT_FALSE(s.spheres[0].cpuMediumUnsupported);
	EXPECT_FALSE(warnedAbout(s, "clipped sphere"));
}

TEST(FlattenTest, NonUniformScaleWarningDistinguishesTheClippedCpuExactPathFromGpu) {
	// The old, unconditional "would be an ellipsoid" wording is only true
	// for the baked (GPU) path - a clipped sphere's CPU rendering carries
	// the real transform and handles non-uniform scale exactly, so it needs
	// its own, differently-worded warning rather than the misleading one.
	const FlatScene clipped = flattenSource(
		"Scale 1 5 1\nShape \"sphere\" \"float radius\" [ 1 ] \"float zmin\" [ 0 ]\n");
	ASSERT_EQ(clipped.spheres.size(), 1u);
	EXPECT_TRUE(clipped.spheres[0].clipped);
	EXPECT_FALSE(warnedAbout(clipped, "ellipsoid"))
		<< "the ellipsoid/largest-radius wording is false for CPU's exact clipped path";
	EXPECT_TRUE(warnedAbout(clipped, "non-uniform scale"));

	const FlatScene unclipped = flattenSource(
		"Scale 1 5 1\nShape \"sphere\" \"float radius\" [ 1 ]\n");
	EXPECT_TRUE(warnedAbout(unclipped, "ellipsoid"))
		<< "unclipped spheres keep the original wording unchanged";
}

TEST(FlattenTest, ClippedSphereKeepsItsBakedCenterAndRadiusToo) {
	// GPU deliberately keeps rendering a clipped sphere as its full,
	// unclipped shape (OptiX's hardware sphere primitive has no clipping
	// support) - it reads center/radius exactly like an unclipped sphere,
	// so those must stay populated and correct even when clipped is true.
	const FlatScene s = flattenSource(
		"Translate 0 0 10\n"
		"Shape \"sphere\" \"float radius\" [ 2 ] \"float zmin\" [ 0 ]\n");
	ASSERT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(s.spheres[0].clipped);
	EXPECT_DOUBLE_EQ(s.spheres[0].center[2], 10.0);
	EXPECT_DOUBLE_EQ(s.spheres[0].radius, 2.0);
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
// Material "hair"
// ===========================================================================

TEST(FlattenTest, HairMaterialWithDirectSigmaA) {
	const FlatScene s = flattenSource(
		"Material \"hair\" \"rgb sigma_a\" [ 0.15 0.06 0.03 ] "
		"\"float beta_m\" [ 0.25 ] \"float beta_n\" [ 0.35 ] \"float alpha\" [ 3 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Hair);
	EXPECT_DOUBLE_EQ(s.materials[0].sigma_a[0], 0.15);
	EXPECT_DOUBLE_EQ(s.materials[0].sigma_a[1], 0.06);
	EXPECT_DOUBLE_EQ(s.materials[0].sigma_a[2], 0.03);
	EXPECT_DOUBLE_EQ(s.materials[0].betaM, 0.25);
	EXPECT_DOUBLE_EQ(s.materials[0].betaN, 0.35);
	EXPECT_DOUBLE_EQ(s.materials[0].alphaDeg, 3.0);
	// pbrt-v4's own Hair-specific eta default, not the generic 1.5.
	EXPECT_DOUBLE_EQ(s.materials[0].ior, 1.55);
}

TEST(FlattenTest, HairMaterialWithEumelaninPheomelanin) {
	// sigma_a = ce*(0.419,0.697,1.37) + cp*(0.187,0.4,1.05) - pbrt-v4's own
	// HairBxDF::SigmaAFromConcentration (bxdfs.cpp).
	const FlatScene s = flattenSource(
		"Material \"hair\" \"float eumelanin\" [ 1.0 ] \"float pheomelanin\" [ 0.5 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_NEAR(s.materials[0].sigma_a[0], 1.0 * 0.419 + 0.5 * 0.187, 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[1], 1.0 * 0.697 + 0.5 * 0.4, 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[2], 1.0 * 1.37 + 0.5 * 1.05, 1e-9);
}

TEST(FlattenTest, HairMaterialWithNothingSpecifiedUsesDefaultBrown) {
	// pbrt-v4's own default: SigmaAFromConcentration(1.3, 0.).
	const FlatScene s = flattenSource("Material \"hair\"\nShape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_NEAR(s.materials[0].sigma_a[0], 1.3 * 0.419, 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[1], 1.3 * 0.697, 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[2], 1.3 * 1.37, 1e-9);
}

namespace {
// pbrt-v4 HairBxDF::SigmaAFromReflectance (bxdfs.cpp) - mirrors
// pbrt_flatten.h's own sigmaAFromReflectanceChannel lambda exactly, so this
// test verifies the loader against the real formula rather than a magic
// literal.
double sigmaAFromReflectanceChannelRef(double c, double bn) {
	c = c < 1e-4 ? 1e-4 : (c > 1.0 - 1e-4 ? 1.0 - 1e-4 : c);
	const double bn2 = bn * bn, bn3 = bn2 * bn, bn4 = bn3 * bn, bn5 = bn4 * bn;
	const double denom = 5.969 - 0.215 * bn + 2.532 * bn2
		- 10.73 * bn3 + 5.574 * bn4 + 0.245 * bn5;
	const double x = std::log(c) / denom;
	return x * x;
}
} // namespace

TEST(FlattenTest, HairMaterialWithReflectanceInvertsToSigmaA) {
	// beta_n stays at its 0.3 default - matches pbrt-v4's own priority
	// (reflectance wins over eumelanin/pheomelanin, and no longer falls back
	// to the default brown preset the way it used to before this loader
	// implemented the closed-form inversion).
	const FlatScene s = flattenSource(
		"Material \"hair\" \"rgb reflectance\" [ 0.5 0.3 0.2 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_NEAR(s.materials[0].sigma_a[0], sigmaAFromReflectanceChannelRef(0.5, 0.3), 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[1], sigmaAFromReflectanceChannelRef(0.3, 0.3), 1e-9);
	EXPECT_NEAR(s.materials[0].sigma_a[2], sigmaAFromReflectanceChannelRef(0.2, 0.3), 1e-9);
	EXPECT_FALSE(warnedAbout(s, "reflectance"));
}

TEST(FlattenTest, HairMaterialWithReflectanceAndBetaNUsesTheGivenBetaN) {
	// beta_n must be read BEFORE sigma_a resolution, since the reflectance
	// inversion formula depends on it - this pins that ordering.
	const FlatScene s = flattenSource(
		"Material \"hair\" \"rgb reflectance\" [ 0.5 0.3 0.2 ] \"float beta_n\" [ 0.6 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_NEAR(s.materials[0].sigma_a[0], sigmaAFromReflectanceChannelRef(0.5, 0.6), 1e-9);
	// Sanity: using the WRONG (default 0.3) beta_n would give a visibly
	// different value, so this also proves beta_n is actually threaded
	// through rather than silently ignored.
	EXPECT_NE(s.materials[0].sigma_a[0], sigmaAFromReflectanceChannelRef(0.5, 0.3));
}

TEST(FlattenTest, HairMaterialWithReflectanceAndEumelaninWarnsReflectanceWins) {
	const FlatScene s = flattenSource(
		"Material \"hair\" \"rgb reflectance\" [ 0.5 0.3 0.2 ] \"float eumelanin\" [ 2.0 ]\n"
		"Shape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_NEAR(s.materials[0].sigma_a[0], sigmaAFromReflectanceChannelRef(0.5, 0.3), 1e-9);
	EXPECT_TRUE(warnedAbout(s, "reflectance"));
}

TEST(FlattenTest, HairMaterialDefaultBetaAndAlphaMatchHairMaterialHDefaults) {
	const FlatScene s = flattenSource("Material \"hair\"\nShape \"sphere\"\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].betaM, 0.3);
	EXPECT_DOUBLE_EQ(s.materials[0].betaN, 0.3);
	EXPECT_DOUBLE_EQ(s.materials[0].alphaDeg, 2.0);
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
								std::vector<float> &pos, std::vector<int> &idx,
								std::vector<float> &) {
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
	EXPECT_FALSE(s.triangles[0].hasUVs) << "the mock resolver supplied no UV";
}

TEST(FlattenTest, PlyMeshThreadsRealUVFromTheResolver) {
	const MeshResolver res = [](const std::string &path,
								std::vector<float> &pos, std::vector<int> &idx,
								std::vector<float> &uvs) {
		if (path != "geometry/tri.ply") return false;
		pos = {0, 0, 0,  1, 0, 0,  0, 1, 0};
		idx = {0, 1, 2};
		uvs = {0, 0,  1, 0,  0, 1};
		return true;
	};
	const FlatScene s = flattenSource(
		"Shape \"plymesh\" \"string filename\" [ \"geometry/tri.ply\" ]\n", res);
	ASSERT_EQ(s.triangles.size(), 1u);
	ASSERT_TRUE(s.triangles[0].hasUVs);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[0], 0.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[1], 0.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[2], 1.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[3], 0.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[4], 0.0);
	EXPECT_DOUBLE_EQ(s.triangles[0].uv[5], 1.0);
}

TEST(FlattenTest, UnreadablePlyMeshWarnsRatherThanAborting) {
	const MeshResolver res = [](const std::string &, std::vector<float> &, std::vector<int> &,
								std::vector<float> &) {
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
// Camera motion blur (ActiveTransform / TransformTimes / shutteropen /
// shutterclose)
// ===========================================================================

TEST(FlattenTest, RegularizeDefaultsToFalse) {
	const FlatScene s = flattenSource(
		"Camera \"perspective\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_FALSE(s.regularize);
}

TEST(FlattenTest, RegularizeIsCarriedThroughFromTheIntegratorDirective) {
	const FlatScene s = flattenSource(
		"Integrator \"volpath\" \"bool regularize\" [ true ]\n"
		"Camera \"perspective\"\nWorldBegin\n" + std::string(kQuadMesh));
	EXPECT_TRUE(s.regularize);
}

TEST(FlattenCameraTest, StaticSceneCameraIsNotAnimated) {
	const FlatScene s = flattenSource(
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"Camera \"perspective\"\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_FALSE(s.camera.isAnimated);
	EXPECT_DOUBLE_EQ(s.camera.shutterOpen, 0.0);
	EXPECT_DOUBLE_EQ(s.camera.shutterClose, 1.0);
}

TEST(FlattenCameraTest, AnimatedCameraCarriesTheEndKeyframeAndShutterWindow) {
	const FlatScene s = flattenSource(
		"ActiveTransform \"StartTime\"\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"EndTime\"\n"
		"LookAt 3 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"All\"\n"
		"Camera \"perspective\" \"float shutteropen\" [ 0.1 ] \"float shutterclose\" [ 0.9 ]\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_TRUE(s.camera.isAnimated);
	EXPECT_NEAR(s.camera.lookfrom[0], 0.0, 1e-9);    // start keyframe, unchanged
	EXPECT_NEAR(s.camera.lookfrom1[0], 3.0, 1e-9);   // end keyframe
	EXPECT_DOUBLE_EQ(s.camera.shutterOpen, 0.1);
	EXPECT_DOUBLE_EQ(s.camera.shutterClose, 0.9);
}

TEST(FlattenCameraTest, TransformTimesDistinctFromShutterWindowWarns) {
	// This codebase's own camera implementation uses shutteropen/
	// shutterclose for BOTH the keyframe times and the shutter sampling
	// window (Camera::shutterOpen's own comment) - a scene relying on
	// TransformTimes differing from them must find out, not silently get
	// the wrong timing.
	const FlatScene s = flattenSource(
		"TransformTimes 0.25 0.75\n"
		"ActiveTransform \"StartTime\"\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"EndTime\"\n"
		"LookAt 3 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"All\"\n"
		"Camera \"perspective\"\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_TRUE(s.camera.isAnimated);
	EXPECT_TRUE(warnedAbout(s, "TransformTimes"));
}

TEST(FlattenCameraTest, TransformTimesMatchingShutterWindowDoesNotWarn) {
	const FlatScene s = flattenSource(
		"TransformTimes 0 1\n"
		"ActiveTransform \"StartTime\"\n"
		"LookAt 0 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"EndTime\"\n"
		"LookAt 3 0 -5   0 0 0   0 1 0\n"
		"ActiveTransform \"All\"\n"
		"Camera \"perspective\"\n"
		"WorldBegin\n" + std::string(kQuadMesh));
	EXPECT_TRUE(s.camera.isAnimated);
	EXPECT_FALSE(warnedAbout(s, "TransformTimes"));
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

TEST(FlattenMaterialTest, CoatedDiffuseReflectanceImagemapIsThreadedThrough) {
	// pbrt's own ganesha scene's exact binding shape - Material::
	// textureFilename's own comment on why CoatedDiffuse is gated in
	// alongside Diffuse (previously Diffuse-only, warning for every other
	// kind).
	const FlatScene s = flattenSource(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"statue.png\" ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "statue.png");
	EXPECT_DOUBLE_EQ(s.materials[0].textureScale, 1.0);
	EXPECT_FALSE(warnedAbout(s, "coateddiffuse"));
}

TEST(FlattenMaterialTest, CoatedDiffuseReflectanceScaleWrappedImagemapCarriesTheScale) {
	// barcelona-pavilion's own dominant binding shape.
	const FlatScene s = flattenSource(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"concrete.png\" ]\n"
		"Texture \"tmap-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"tmap\" ] "
		"\"float scale\" [ 0.7 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "concrete.png");
	EXPECT_DOUBLE_EQ(s.materials[0].textureScale, 0.7);
}

TEST(FlattenMaterialTest, DiffuseReflectanceScaleWrappedImagemapCarriesTheScale) {
	// barcelona-pavilion's own dominant binding shape, now also supported for
	// plain "diffuse" surfaces, not just "coateddiffuse" - see
	// Material::textureScale's own comment.
	const FlatScene s = flattenSource(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"concrete.png\" ]\n"
		"Texture \"tmap-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"tmap\" ] "
		"\"float scale\" [ 0.7 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "concrete.png");
	EXPECT_DOUBLE_EQ(s.materials[0].textureScale, 0.7);
	EXPECT_FALSE(warnedAbout(s, "diffuse"));
}

TEST(FlattenMaterialTest, CoatedDiffuseReflectanceCheckerboardResolvesToAProceduralTexture) {
	// checkerboard/fbm/marble/mix used to be Diffuse-only (a documented
	// scope cut); now also resolved for CoatedDiffuse, same as the "scale"
	// unwrap and bare-imagemap cases already were - see
	// hasCheckerReflectance's own comment.
	const FlatScene s = flattenSource(
		"Texture \"chk\" \"spectrum\" \"checkerboard\"\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasCheckerReflectance);
	EXPECT_FALSE(warnedAbout(s, "coateddiffuse"));
}

TEST(FlattenMaterialTest, CoatedDiffuseReflectanceFbmResolvesToAProceduralTexture) {
	// Same broadening as checkerboard above, for fbm specifically - the fbm
	// gate at MaterialKind::Diffuse|CoatedDiffuse was previously untested for
	// CoatedDiffuse at this level.
	const FlatScene s = flattenSource(
		"Texture \"cloud\" \"float\" \"fbm\" \"integer octaves\" [ 4 ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasFbmReflectance);
	EXPECT_EQ(s.materials[0].fbmOctaves, 4);
	EXPECT_FALSE(warnedAbout(s, "coateddiffuse"));
}

TEST(FlattenMaterialTest, CoatedDiffuseReflectanceMarbleResolvesToAProceduralTexture) {
	// Same broadening as checkerboard above, for marble specifically - the
	// marble gate at MaterialKind::Diffuse|CoatedDiffuse was previously
	// untested for CoatedDiffuse at this level.
	const FlatScene s = flattenSource(
		"Texture \"stone\" \"spectrum\" \"marble\"\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasMarbleReflectance);
	EXPECT_FALSE(warnedAbout(s, "coateddiffuse"));
}

TEST(FlattenMaterialTest, MixAmountImagemapCarriesTheTextureFilename) {
	// pbrt-v4's real "amount" bound to its own Texture (e.g. an fbm-driven
	// dirt/wear mask) - previously always fell through to the generic "not
	// supported" warning; now resolved the same one-level-nested-bare-
	// imagemap way as tex1/tex2 - see Material::mixAmountTextureFilename's
	// own comment.
	const FlatScene s = flattenSource(
		"Texture \"wear\" \"float\" \"imagemap\" \"string filename\" [ \"wear.png\" ]\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ] "
		"\"texture amount\" [ \"wear\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasMixReflectance);
	EXPECT_EQ(s.materials[0].mixAmountTextureFilename, "wear.png");
	EXPECT_FALSE(warnedAbout(s, "diffuse"));
}

TEST(FlattenMaterialTest, MixAmountBoundToACheckerboardStillWarns) {
	// Regression guard for the "amount" nesting added alongside tex1/tex2's
	// own: "amount" naming a Texture that is itself NOT a bare imagemap
	// (here, a checkerboard - a second procedural texture, not a colour
	// sample) must still fall through to the generic warning, same scope
	// tex1/tex2 already had (see resolveNestedImagemap's own comment on why
	// this stays one level only).
	const FlatScene s = flattenSource(
		"Texture \"chk\" \"spectrum\" \"checkerboard\"\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ] "
		"\"texture amount\" [ \"chk\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasMixReflectance);
	EXPECT_TRUE(s.materials[0].mixAmountTextureFilename.empty());
	EXPECT_TRUE(warnedAbout(s, "diffuse"));
}

TEST(FlattenMaterialTest, DiffuseTransmissionReflectanceCheckerboardStillWarns) {
	// Regression guard for the Diffuse->CoatedDiffuse procedural-texture
	// broadening above: DiffuseTransmission must stay excluded (no bundled
	// scene binds a checkerboard/fbm/marble/mix to a diffusetransmission
	// reflectance), so this must still warn rather than the broadened OR
	// condition accidentally picking it up too.
	const FlatScene s = flattenSource(
		"Texture \"chk\" \"spectrum\" \"checkerboard\"\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"chk\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasCheckerReflectance);
	EXPECT_TRUE(warnedAbout(s, "diffusetransmission"));
}

TEST(FlattenMaterialTest, DiffuseReflectanceScaleWrappingACheckerboardStillWarns) {
	// A regression guard for the reflectance scale-unwrap added alongside
	// CoatedDiffuse support: unwrapping "scale" must only ever feed the
	// "imagemap" branch, never silently reveal a wrapped checkerboard/fbm/
	// marble/mix underneath and drop its own scale factor unnoticed - see
	// the unwrap's own "imgTex, not tex" comment in the flatten loop.
	const FlatScene s = flattenSource(
		"Texture \"chk\" \"spectrum\" \"checkerboard\"\n"
		"Texture \"chk-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"chk\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasCheckerReflectance);
	EXPECT_TRUE(warnedAbout(s, "diffuse"));
}

TEST(FlattenMaterialTest, DielectricEtaIsReadAsIor) {
	const FlatScene s = flattenSource(
		"Material \"dielectric\" \"float eta\" [ 1.33 ]\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Dielectric);
	EXPECT_DOUBLE_EQ(s.materials[0].ior, 1.33);
}

TEST(FlattenMaterialTest, InterfaceMaterialNoneMapsToInterfaceKind) {
	// pbrt-v4's real interface-material idiom (a shape bounding a
	// participating medium with no BSDF response of its own) - previously
	// fell to MaterialKind::Unsupported (opaque flat Lambertian, plus a
	// warning). Built as a real, dedicated pass-through material (CPU's
	// interface_material) - m.ior is irrelevant to it and no longer forced.
	const FlatScene s = flattenSource(
		"Material \"none\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Interface);
	EXPECT_FALSE(warnedAbout(s, "none"));
}

TEST(FlattenMaterialTest, InterfaceMaterialEmptyStringAlsoMapsToInterface) {
	// pbrt-v4 also accepts a bare empty type string for the same idiom.
	const FlatScene s = flattenSource(
		"Material \"\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Interface);
}

TEST(FlattenMaterialTest, ConstantTextureRgbValueResolvesReflectance) {
	// A real scene author's indirection for a colour reused across several
	// materials - previously fell through to the generic "texture not
	// supported" warning, and worse, to the WRONG fallback colour (the
	// pre-loop getVec3() read the raw texture-typed "reflectance" param,
	// which has no numbers, silently returning the generic default).
	const FlatScene s = flattenSource(
		"Texture \"c\" \"spectrum\" \"constant\" \"rgb value\" [ .2 .4 .6 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"c\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].color[0], 0.2);
	EXPECT_DOUBLE_EQ(s.materials[0].color[1], 0.4);
	EXPECT_DOUBLE_EQ(s.materials[0].color[2], 0.6);
	EXPECT_FALSE(warnedAbout(s, "diffuse"));
}

TEST(FlattenMaterialTest, ConstantTextureFloatValueBroadcastsToReflectance) {
	// A float-typed constant texture feeding an RGB-consuming parameter -
	// broadcasts to RGB, matching pbrt-v4's own convention. Deliberately NOT
	// 0.5: Material::color's own in-class default is {0.5,0.5,0.5}, so that
	// value would pass this test even if the broadcast silently failed and
	// getVec3() fell back to the default - which is exactly what happened
	// before this was fixed (the rewrite wrote a 1-element numbers vector,
	// getVec3() requires >=3 and returned its default unnoticed).
	const FlatScene s = flattenSource(
		"Texture \"c\" \"float\" \"constant\" \"float value\" [ .2 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"c\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].color[0], 0.2);
	EXPECT_DOUBLE_EQ(s.materials[0].color[1], 0.2);
	EXPECT_DOUBLE_EQ(s.materials[0].color[2], 0.2);
}

TEST(FlattenMaterialTest, ConstantTextureBoundToKResolvesOnConductor) {
	// Deliberately not gated on material kind - conductor's "k" needs the
	// same resolution as diffuse-family "reflectance".
	const FlatScene s = flattenSource(
		"Texture \"c\" \"spectrum\" \"constant\" \"rgb value\" [ .9 .8 .1 ]\n"
		"Material \"conductor\" \"texture k\" [ \"c\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].color[0], 0.9);
	EXPECT_DOUBLE_EQ(s.materials[0].color[1], 0.8);
	EXPECT_DOUBLE_EQ(s.materials[0].color[2], 0.1);
	EXPECT_FALSE(warnedAbout(s, "conductor"));
}

TEST(FlattenMaterialTest, ConstantTextureReflectanceWinsOverConstantTextureK) {
	// Both bound to constant textures - the pre-existing "reflectance wins
	// over k" precedence (materials.pbrt's own rule) must still hold
	// regardless of which param happens to be declared later in the file.
	// This is the general Param-rewrite pre-pass's whole point: reflectance/
	// k precedence is decided ONCE, by the same code whether the values
	// came from plain numbers or a resolved constant texture - not
	// re-decided by declaration order in a separate texture-handling branch.
	const FlatScene s = flattenSource(
		"Texture \"refl\" \"spectrum\" \"constant\" \"rgb value\" [ .9 .9 .9 ]\n"
		"Texture \"kk\" \"spectrum\" \"constant\" \"rgb value\" [ 3.9 2.4 1.7 ]\n"
		"Material \"conductor\" \"texture reflectance\" [ \"refl\" ] \"texture k\" [ \"kk\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].color[0], 0.9);
	EXPECT_DOUBLE_EQ(s.materials[0].color[1], 0.9);
	EXPECT_DOUBLE_EQ(s.materials[0].color[2], 0.9);
}

TEST(FlattenMaterialTest, ConstantTextureEtaAndKResolveRealConductorModel) {
	// Both eta and k texture-bound with real (>1) copper-like values - the
	// rewrite pre-pass must make the explicit-RGB conductor path (which
	// needs BOTH eta and k as real numbers to activate, see the "eta"/"k"
	// resolution block just below the reflectance/k precedence) trigger
	// correctly, instead of falling to the generic metal(albedo) path with
	// an un-clamped, energy-gaining (>1) albedo.
	const FlatScene s = flattenSource(
		"Texture \"cueta\" \"spectrum\" \"constant\" \"rgb value\" [ 0.2 0.92 1.1 ]\n"
		"Texture \"cuk\" \"spectrum\" \"constant\" \"rgb value\" [ 3.9 2.45 2.14 ]\n"
		"Material \"conductor\" \"texture eta\" [ \"cueta\" ] \"texture k\" [ \"cuk\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasConductorPreset);
	EXPECT_DOUBLE_EQ(s.materials[0].conductorEta[0], 0.2);
	EXPECT_DOUBLE_EQ(s.materials[0].conductorK[0], 3.9);
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

TEST(FlattenMaterialTest, DiffuseTransmissionReflectanceAndTransmittanceImagemapsAreThreadedThrough) {
	// barcelona-pavilion's foliage binding shape - both params bound to the
	// SAME bare imagemap - see Material::textureFilename/
	// transmittanceTextureFilename's own comments.
	const FlatScene s = flattenSource(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf\" ] "
		"\"texture transmittance\" [ \"leaf\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "leaf.png");
	EXPECT_EQ(s.materials[0].transmittanceTextureFilename, "leaf.png");
	EXPECT_FALSE(warnedAbout(s, "diffusetransmission"));
}

TEST(FlattenMaterialTest, DiffuseTransmissionReflectanceAndTransmittanceUseDistinctTextures) {
	// Regression guard: reflectance and transmittance must resolve to their
	// OWN respective texture filename, not accidentally alias each other's.
	const FlatScene s = flattenSource(
		"Texture \"leaf-r\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-r.png\" ]\n"
		"Texture \"leaf-t\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-t.png\" ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf-r\" ] "
		"\"texture transmittance\" [ \"leaf-t\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "leaf-r.png");
	EXPECT_EQ(s.materials[0].transmittanceTextureFilename, "leaf-t.png");
}

TEST(FlattenMaterialTest, DiffuseTransmissionReflectanceScaleWrappedImagemapCarriesTheScale) {
	// "scale"-unwrap used to be CoatedDiffuse-only; now also resolved for
	// DiffuseTransmission's own reflectance - see Material::textureScale's
	// own comment.
	const FlatScene s = flattenSource(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"leaf-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"leaf\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "leaf.png");
	EXPECT_DOUBLE_EQ(s.materials[0].textureScale, 0.5);
	EXPECT_FALSE(warnedAbout(s, "diffusetransmission"));
}

TEST(FlattenMaterialTest, DiffuseTransmissionTransmittanceScaleWrappedImagemapCarriesItsOwnScale) {
	// Same broadening, for transmittance specifically - and with a
	// DIFFERENT scale value than reflectance's own, to confirm the two are
	// resolved independently (Material::transmittanceTextureScale's own
	// comment) rather than accidentally sharing one scale.
	const FlatScene s = flattenSource(
		"Texture \"leaf-r\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-r.png\" ]\n"
		"Texture \"leaf-r-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"leaf-r\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Texture \"leaf-t\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf-t.png\" ]\n"
		"Texture \"leaf-t-scaled\" \"spectrum\" \"scale\" \"texture tex\" [ \"leaf-t\" ] "
		"\"float scale\" [ 0.8 ]\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"leaf-r-scaled\" ] "
		"\"texture transmittance\" [ \"leaf-t-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "leaf-r.png");
	EXPECT_DOUBLE_EQ(s.materials[0].textureScale, 0.5);
	EXPECT_EQ(s.materials[0].transmittanceTextureFilename, "leaf-t.png");
	EXPECT_DOUBLE_EQ(s.materials[0].transmittanceTextureScale, 0.8);
	EXPECT_FALSE(warnedAbout(s, "diffusetransmission"));
}

TEST(FlattenMaterialTest, DielectricRoughnessImagemapIsThreadedThrough) {
	// pbrt-v4 "texture roughness" on a Dielectric bound to a bare
	// imagemap - see Material::roughnessTextureFilename's own comment.
	// Bare imagemap only, same scope as DiffuseTransmission's own
	// transmittance texture-binding above.
	const FlatScene s = flattenSource(
		"Texture \"scratch\" \"float\" \"imagemap\" \"string filename\" [ \"scratch.png\" ]\n"
		"Material \"dielectric\" \"texture roughness\" [ \"scratch\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].roughnessTextureFilename, "scratch.png");
	EXPECT_FALSE(warnedAbout(s, "dielectric"));
}

TEST(FlattenMaterialTest, DielectricRoughnessConstantTextureResolvesToFlatNumber) {
	// The generic constant-texture rewrite pre-pass (flatten()'s own
	// comment, runs before any per-kind logic) is kind-agnostic - a
	// "texture roughness" bound to a "constant" Texture already resolves
	// to a real flat number here with NO dedicated Dielectric-roughness
	// code needed, unlike the imagemap case above which needed real
	// wiring. Regression guard confirming that pre-pass really does cover
	// this param on this material kind.
	const FlatScene s = flattenSource(
		"Texture \"fixedRough\" \"float\" \"constant\" \"float value\" [ 0.3 ]\n"
		"Material \"dielectric\" \"texture roughness\" [ \"fixedRough\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_DOUBLE_EQ(s.materials[0].roughness, 0.3);
	EXPECT_TRUE(s.materials[0].roughnessTextureFilename.empty());
	EXPECT_FALSE(warnedAbout(s, "dielectric"));
}

TEST(FlattenMaterialTest, DielectricRoughnessScaleWrappedImagemapStillWarns) {
	// Deliberate scope cut: no "scale"-wrap support for Dielectric's own
	// roughness texture-binding (unlike Diffuse/CoatedDiffuse/
	// DiffuseTransmission's own reflectance/transmittance, which all got
	// this above).
	const FlatScene s = flattenSource(
		"Texture \"scratch\" \"float\" \"imagemap\" \"string filename\" [ \"scratch.png\" ]\n"
		"Texture \"scratch-scaled\" \"float\" \"scale\" \"texture tex\" [ \"scratch\" ] "
		"\"float scale\" [ 0.5 ]\n"
		"Material \"dielectric\" \"texture roughness\" [ \"scratch-scaled\" ]\n"
		+ std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].roughnessTextureFilename.empty());
	EXPECT_TRUE(warnedAbout(s, "dielectric"));
}

TEST(FlattenMaterialTest, TrianglemeshWithMediumInterfaceWarnsMediumIsDropped) {
	// Triangle has no `medium` field (unlike Sphere/Disk/Cylinder), so a
	// mesh-bounded medium boundary - the most common real pbrt-v4 authoring
	// pattern for one - silently loses its medium. This must warn instead of
	// going quiet, especially now that Material "none" itself no longer
	// warns (MaterialKind::Interface, not Unsupported).
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"AttributeBegin\n"
		"  Material \"none\"\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ -1 -1 0   1 -1 0   1 1 0   -1 1 0 ]\n"
		"AttributeEnd\n");
	EXPECT_TRUE(warnedAbout(s, "MediumInterface"));
}

TEST(FlattenMaterialTest, SphereMediumInterfaceDoesNotWarn) {
	// Regression guard: the new trianglemesh warning must not fire for the
	// shape kinds that DO support an attached medium.
	const FlatScene s = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" \"homogeneous\"\n"
		"AttributeBegin\n"
		"  Material \"none\"\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\"\n"
		"AttributeEnd\n");
	EXPECT_FALSE(warnedAbout(s, "MediumInterface"));
}

TEST(FlattenMaterialTest, UnsupportedMaterialIsFlaggedNotSilentlySubstituted) {
	// A material rendered as diffuse looks plausible and is wrong, so the
	// substitution has to be announced. "subsurface" then "hair" used to be
	// the example here, in turn, until each became a real, CPU/GPU-supported
	// MaterialKind - every real pbrt-v4 material name now maps to a genuine
	// MaterialKind, so this uses a made-up, never-real type string instead
	// (matching materialKindFor()'s own "anything not recognized" fallback,
	// the actual case this Unsupported path exists for now - a typo'd or
	// genuinely nonexistent material name, not a real pbrt-v4 kind this
	// loader hasn't gotten to yet).
	const FlatScene s = flattenSource(
		"Material \"holographic\"\n" + std::string(kQuadMesh));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].kind, MaterialKind::Unsupported);
	EXPECT_EQ(s.materials[0].pbrtType, "holographic");
	EXPECT_TRUE(warnedAbout(s, "holographic"));
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

TEST(FlattenMaterialTest, BlackbodyEmissionIsConvertedToARealColour) {
	// Previously warned-and-discarded the temperature entirely (getVec3
	// requires >=3 numbers, a "blackbody L" param has exactly 1, so it
	// silently fell back to flat white regardless of temperature) - now
	// converted via resolveEmissionColor() to a real RGB colour. Verified
	// against the real physical characteristic every blackbody radiator has
	// (Planck's law + Wien's displacement law: low temperature -> warm/
	// orange, high temperature -> cool/blue) rather than a hand-computed
	// exact expected value - barcelona-pavilion's own night lighting uses
	// exactly this range (2500K-3500K).
	const FlatScene warmLight = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"blackbody L\" [ 2500 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	const FlatScene coolLight = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"blackbody L\" [ 9000 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	ASSERT_EQ(warmLight.areaLights.size(), 1u);
	ASSERT_EQ(coolLight.areaLights.size(), 1u);
	EXPECT_FALSE(warnedAbout(warmLight, "blackbody"))
		<< "a blackbody temperature is now converted, not warned-and-discarded";

	const double (&warm)[3] = warmLight.areaLights[0].L;
	const double (&cool)[3] = coolLight.areaLights[0].L;
	for (int c = 0; c < 3; ++c) {
		EXPECT_GE(warm[c], 0.0);
		EXPECT_GE(cool[c], 0.0);
	}
	EXPECT_FALSE(warm[0] == 1.0 && warm[1] == 1.0 && warm[2] == 1.0)
		<< "must not be the pre-fix flat-white {1,1,1} fallback";
	EXPECT_GT(warm[0] / warm[2], cool[0] / cool[2])
		<< "a 2500K light must read warmer (redder relative to blue) than a 9000K light";
}

TEST(FlattenMaterialTest, ColorSpaceDirectiveChangesBlackbodyConversion) {
	// The one place a ColorSpace directive's selection actually reaches
	// downstream (GraphicsState::colorSpaceName's own comment, pbrt_scene.h):
	// the same blackbody temperature must convert to a genuinely different
	// RGB triple under a different working color space's primaries, not
	// just have the directive parse without effect.
	const FlatScene srgbLight = flattenSource(
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"blackbody L\" [ 4000 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	const FlatScene rec2020Light = flattenSource(
		"ColorSpace \"rec2020\"\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"blackbody L\" [ 4000 ]\n"
		+ std::string(kQuadMesh) + "AttributeEnd\n");
	ASSERT_EQ(srgbLight.areaLights.size(), 1u);
	ASSERT_EQ(rec2020Light.areaLights.size(), 1u);
	const double (&srgb)[3] = srgbLight.areaLights[0].L;
	const double (&rec2020)[3] = rec2020Light.areaLights[0].L;
	EXPECT_FALSE(srgb[0] == rec2020[0] && srgb[1] == rec2020[1] && srgb[2] == rec2020[2])
		<< "same blackbody temperature must resolve to a different RGB triple "
		   "under Rec.2020's wider primaries than under the sRGB default";
}

TEST(FlattenPunctualLightTest, PointLightBlackbodyIntensityIsConverted) {
	// Punctual lights (point/spot/distant/goniometric) previously had NO
	// blackbody handling at all - not even a warning, "blackbody I" silently
	// became flat white with zero diagnostic. resolveEmissionColor() now
	// covers these the same way as AreaLightSource.
	const FlatScene s = flattenSource(
		"LightSource \"point\" \"point3 from\" [ 0 0 0 ] \"blackbody I\" [ 3000 ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	const double (&i)[3] = s.punctualLights[0].intensity;
	for (int c = 0; c < 3; ++c) EXPECT_GE(i[c], 0.0);
	EXPECT_FALSE(i[0] == 1.0 && i[1] == 1.0 && i[2] == 1.0)
		<< "must not be the pre-fix flat-white {1,1,1} fallback";
	EXPECT_GT(i[0], i[2]) << "a 3000K light must read warmer (more red than blue)";
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

TEST(FlattenInfiniteLightTest, NoPortalParamLeavesHasPortalFalse) {
	// The overwhelmingly common case - a plain (non-windowed) infinite
	// light - must not be misread as having a portal.
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	EXPECT_FALSE(s.infiniteLight.hasPortal);
}

TEST(FlattenInfiniteLightTest, PortalParamPopulatesHasPortalAndCorners) {
	// pbrt-v4's windowed infinite light ("point3 portal[4]") - 4 corners,
	// ordered p0, p1(=p0+right), p2(=p0+right+up), p3(=p0+up) (matches
	// PortalImageInfiniteLightData's own ordering comment).
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"string filename\" [ \"env.exr\" ] "
		"\"point3 portal\" [ -1 -1 0   1 -1 0   1 1 0   -1 1 0 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	ASSERT_TRUE(s.infiniteLight.hasPortal);
	const double expected[12] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
	for (int i = 0; i < 12; ++i)
		EXPECT_DOUBLE_EQ(s.infiniteLight.portal[i], expected[i]) << "component " << i;
}

TEST(FlattenInfiniteLightTest, PortalCornersAreTransformedByTheCTM) {
	// PortalImageInfiniteLightData itself applies no further transform (see
	// its own constructor comment) - the loader must bring the corners into
	// world/render space at parse time, the same way it already does for
	// point/spot/distant "from"/"to" (see FlattenPunctualLightTest below).
	const FlatScene s = flattenSource(
		"Translate 5 0 0\n"
		"LightSource \"infinite\" \"string filename\" [ \"env.exr\" ] "
		"\"point3 portal\" [ -1 -1 0   1 -1 0   1 1 0   -1 1 0 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	ASSERT_TRUE(s.infiniteLight.hasPortal);
	// Every corner's own x shifts by +5; y/z pass through unchanged.
	const double expectedX[4] = { 4.0, 6.0, 6.0, 4.0 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_DOUBLE_EQ(s.infiniteLight.portal[i*3+0], expectedX[i]) << "corner " << i;
		EXPECT_DOUBLE_EQ(s.infiniteLight.portal[i*3+1], (i==0||i==1) ? -1.0 : 1.0) << "corner " << i;
		EXPECT_DOUBLE_EQ(s.infiniteLight.portal[i*3+2], 0.0) << "corner " << i;
	}
}

TEST(FlattenInfiniteLightTest, PortalParamWithFewerThanTwelveNumbersIsIgnored) {
	// A malformed/truncated portal[] (e.g. hand-edited scene file) should
	// leave the light as an ordinary (non-windowed) infinite light rather
	// than reading out-of-bounds or half-populating the array.
	const FlatScene s = flattenSource(
		"LightSource \"infinite\" \"rgb L\" [ 1 1 1 ] \"point3 portal\" [ -1 -1 0 ]\n");
	ASSERT_TRUE(s.infiniteLight.present);
	EXPECT_FALSE(s.infiniteLight.hasPortal);
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

TEST(FlattenPunctualLightTest, GoniometricLightWithFilenameIsStoredForLaterResolution) {
	// pbrt_flatten.h stays filesystem-free (see PunctualLight::filename's own
	// comment) - resolving/decoding the name happens later, in pbrt_load.h/
	// the CPU-GPU builders, so this stage should neither warn nor decode,
	// just carry the raw name through unchanged.
	const FlatScene s = flattenSource(
		"LightSource \"goniometric\" \"string filename\" [ \"profile.exr\" ]\n");
	ASSERT_EQ(s.punctualLights.size(), 1u);
	EXPECT_TRUE(s.punctualLights[0].hadImageFilename);
	EXPECT_EQ(s.punctualLights[0].filename, "profile.exr");
	EXPECT_FALSE(warnedAbout(s, "goniometric"));
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
	EXPECT_EQ(pl.filename, "slide.exr");
	EXPECT_FALSE(warnedAbout(s, "projection"));
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

