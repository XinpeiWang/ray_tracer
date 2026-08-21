/**
 * @file pbrt_subdiv_tests.cpp
 * @brief Unit tests for pbrt `loopsubdiv` shapes
 *
 * These exist because of a specific, embarrassing discovery: the whole pbrt
 * chain was verified end to end, on both backends, against a Cornell box I
 * wrote myself - and then the first real published scene loaded, pbrt's own
 * killeroo-simple, rendered its floor and its light and silently dropped both
 * of its subjects. The models are `loopsubdiv`, and a hand-written test scene
 * was never going to contain one.
 *
 * Loop subdivision itself was already in the project (src/shared/
 * loop_subdivide.h, itself a port of pbrt's), so the gap was purely that
 * flatten() did not connect the two.
 */

#include <gtest/gtest.h>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"

#include <algorithm>
#include <string>

using namespace pbrt_flatten;

namespace {

FlatScene build(const std::string &text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return flatten(r.scene);
}

bool warned(const FlatScene &s, const std::string &needle) {
	for (const pbrt_scene::Warning &w : s.warnings)
		if (w.message.find(needle) != std::string::npos) return true;
	return false;
}

// A tetrahedron - the smallest closed mesh Loop subdivision can refine.
// Subdivision needs a closed surface to be meaningful; a single loose triangle
// is a boundary case that says nothing about whether refinement works.
const char *kTetra =
	"  \"integer indices\" [ 0 1 2  0 2 3  0 3 1  1 3 2 ]\n"
	"  \"point3 P\" [ 0 0 0   1 0 0   0 1 0   0 0 1 ]\n";

std::string subdiv(int levels) {
	return "Shape \"loopsubdiv\" \"integer levels\" [ " +
		   std::to_string(levels) + " ]\n" + kTetra;
}

} // namespace

TEST(PbrtSubdivTest, ALoopSubdivShapeProducesGeometryRatherThanAWarning) {
	const FlatScene s = build(subdiv(1));
	EXPECT_GT(s.triangles.size(), 0u)
		<< "loopsubdiv produced nothing - a real scene loses its subject";
	EXPECT_FALSE(warned(s, "not supported"));
}

TEST(PbrtSubdivTest, EachLevelRefinesRatherThanReturningTheControlCage) {
	// The failure this guards against is quiet: handing the control cage
	// straight through renders a faceted lump that looks like geometry and is
	// not the surface the scene asked for.
	EXPECT_GT(build(subdiv(1)).triangles.size(),
			  build(subdiv(0)).triangles.size());
	EXPECT_GT(build(subdiv(2)).triangles.size(),
			  build(subdiv(1)).triangles.size());
}

TEST(PbrtSubdivTest, AnAbsurdLevelCountIsClampedAndSaidOutLoud) {
	// Refinement is exponential. Honouring "levels 20" is not generosity, it
	// is an out-of-memory crash with a four-billion-fold triangle count.
	const FlatScene s = build(subdiv(20));
	EXPECT_GT(s.triangles.size(), 0u);
	EXPECT_TRUE(warned(s, "clamped"))
		<< "rendered something other than what was asked for, and said nothing";
}

TEST(PbrtSubdivTest, ANegativeLevelCountIsTreatedAsNoRefinement) {
	const FlatScene s = build(subdiv(-3));
	EXPECT_GT(s.triangles.size(), 0u) << "the control cage should still render";
}

TEST(PbrtSubdivTest, TheSceneTransformStillAppliesToARefinedSurface) {
	// Subdivision happens in object space. If the CTM were dropped on this
	// path the model would render at the origin no matter where the scene put
	// it - which is exactly how killeroo-simple places its two, by translating
	// between two Includes of the same geometry file.
	const FlatScene s = build("Translate 100 0 0\n" + subdiv(1));
	ASSERT_GT(s.triangles.size(), 0u);
	double minX = 1e9;
	for (const Triangle &t : s.triangles)
		for (int k = 0; k < 9; k += 3) minX = std::min(minX, t.v[k]);
	EXPECT_GT(minX, 99.0) << "the refined surface ignored Translate";
}

TEST(PbrtSubdivTest, RefiningDoesNotInflateTheSurfaceBeyondItsControlCage) {
	// Loop subdivision is an interpolating-ish scheme whose limit surface lies
	// within the convex hull of the cage. A result outside it means the
	// refinement is wrong in a way triangle counts alone would never reveal.
	const FlatScene s = build(subdiv(2));
	ASSERT_GT(s.triangles.size(), 0u);
	for (const Triangle &t : s.triangles) {
		for (int k = 0; k < 9; ++k) {
			EXPECT_GE(t.v[k], -0.001) << "vertex escaped the cage's lower bound";
			EXPECT_LE(t.v[k], 1.001) << "vertex escaped the cage's upper bound";
		}
	}
}

TEST(PbrtSubdivTest, AMalformedLoopSubdivIsSkippedWithItsOwnMessage) {
	const FlatScene s = build("Shape \"loopsubdiv\" \"integer levels\" [ 1 ]\n");
	EXPECT_TRUE(s.triangles.empty());
	EXPECT_TRUE(warned(s, "loopsubdiv"));
}

// ---------------------------------------------------------------------------
// Things that are dropped must say so
// ---------------------------------------------------------------------------
// pbrt's ganesha scene is lit almost entirely by an infinite light. It used
// to be dropped silently, producing a black statue that looked exactly like
// a shading bug - the render was correct given what had been discarded, and
// nothing on screen connected the two. A limitation that announces itself is
// a limitation; one that does not is a bug report waiting to happen.
//
// "infinite" itself is no longer one of these - it is now carried through
// (see FlattenInfiniteLightTest in pbrt_flatten_tests.cpp) rather than
// dropped-with-a-warning, which is what used to be pinned here. point, spot,
// distant, goniometric and projection are ALSO no longer dropped (see
// FlattenPunctualLightTest, also in pbrt_flatten_tests.cpp) - pbrt-v4 has no
// other standard light kinds left, so this test now exercises the generic
// fallback with a made-up type rather than a real pbrt one.

TEST(PbrtDroppedTest, EveryUnsupportedLightTypeIsNamedIndividually) {
	const FlatScene s = build(
		"WorldBegin\n"
		"LightSource \"bogus\"\n"
		"LightSource \"alsobogus\"\n");
	EXPECT_TRUE(warned(s, "bogus"));
	EXPECT_TRUE(warned(s, "alsobogus"));
}

TEST(PbrtDroppedTest, AnAreaLightIsNotWarnedAboutBecauseItIsSupported) {
	// The counterpart that keeps the warning meaningful: if everything warned,
	// the warnings would be noise and nobody would read the one that matters.
	const FlatScene s = build(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 1 1 1 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"AttributeEnd\n");
	EXPECT_FALSE(warned(s, "not supported and was dropped"));
}

TEST(PbrtDroppedTest, ATextureBoundToAMaterialIsReportedWithItsParameterName) {
	const FlatScene s = build(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"))
		<< "a textured material renders flat and says nothing about it";
	EXPECT_TRUE(warned(s, "constant colour"));
}

TEST(PbrtDroppedTest, ADiffuseReflectanceImagemapIsResolvedNotWarned) {
	// The one case ATextureBoundToAMaterialIsReportedWithItsParameterName
	// deliberately does NOT cover: a plain "diffuse" material (not
	// coateddiffuse) whose "reflectance" is bound to an "imagemap" texture -
	// pbrt's ganesha statue is exactly this, and it should resolve to a real
	// image reference (Material::textureFilename) instead of the generic
	// "not supported" warning.
	const FlatScene s = build(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].textureFilename, "t.png");
}

TEST(PbrtDroppedTest, APlainColourMaterialIsNotWarnedAbout) {
	const FlatScene s = build(
		"Material \"diffuse\" \"rgb reflectance\" [ 0.5 0.5 0.5 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "texture"));
}

// ---------------------------------------------------------------------------
// Shading normals
// ---------------------------------------------------------------------------
// The point of subdividing rather than tessellating is the limit surface, and
// the limit surface is carried by its normals. Refining without them renders
// smoother facets, which is not the same thing as a smooth surface.

TEST(PbrtNormalTest, ARefinedSurfaceCarriesUnitLengthShadingNormals) {
	const FlatScene s = build(subdiv(2));
	ASSERT_GT(s.triangles.size(), 0u);
	int withNormals = 0;
	for (const Triangle &t : s.triangles) {
		if (!t.hasNormals) continue;
		++withNormals;
		for (int k = 0; k < 9; k += 3) {
			const double len = std::sqrt(t.n[k] * t.n[k] + t.n[k + 1] * t.n[k + 1] +
										 t.n[k + 2] * t.n[k + 2]);
			EXPECT_NEAR(len, 1.0, 1e-9) << "a shading normal is not unit length";
		}
	}
	EXPECT_GT(withNormals, 0) << "refinement produced no shading normals at all";
}

TEST(PbrtNormalTest, AMeshWithNoNormalsSaysSoRatherThanInventingThem) {
	const FlatScene s = build(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.triangles.size(), 1u);
	EXPECT_FALSE(s.triangles[0].hasNormals);
}

TEST(PbrtNormalTest, AnExplicitNParameterIsCarriedThrough) {
	const FlatScene s = build(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  \"normal N\" [ 0 0 1  0 0 1  0 0 1 ]\n");
	ASSERT_EQ(s.triangles.size(), 1u);
	ASSERT_TRUE(s.triangles[0].hasNormals);
	EXPECT_NEAR(s.triangles[0].n[2], 1.0, 1e-9);
}

TEST(PbrtNormalTest, NormalsSurviveNonUniformScalePerpendicularToTheSurface) {
	// The whole reason normals need the inverse transpose rather than the
	// matrix itself. Under Scale 1 1 4 a directly-transformed normal tilts off
	// the surface; the correct one stays perpendicular to it.
	//
	// Asserting perpendicularity to the transformed edges - rather than a
	// hand-computed vector - means the test states the property that matters
	// and cannot be satisfied by reproducing my own arithmetic error.
	const FlatScene s = build(
		"Scale 1 1 4\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 1 ]\n"
		"  \"normal N\" [ 0 -1 1  0 -1 1  0 -1 1 ]\n");
	ASSERT_EQ(s.triangles.size(), 1u);
	const Triangle &t = s.triangles[0];
	ASSERT_TRUE(t.hasNormals);

	const double e1[3] = {t.v[3] - t.v[0], t.v[4] - t.v[1], t.v[5] - t.v[2]};
	const double e2[3] = {t.v[6] - t.v[0], t.v[7] - t.v[1], t.v[8] - t.v[2]};
	const double d1 = t.n[0] * e1[0] + t.n[1] * e1[1] + t.n[2] * e1[2];
	const double d2 = t.n[0] * e2[0] + t.n[1] * e2[1] + t.n[2] * e2[2];
	EXPECT_NEAR(d1, 0.0, 1e-9) << "normal is no longer perpendicular to edge 1";
	EXPECT_NEAR(d2, 0.0, 1e-9) << "normal is no longer perpendicular to edge 2";
}

TEST(PbrtNormalTest, TooFewNormalsAreRefusedWholesaleRatherThanUsedInPart) {
	// Half-smooth shading is a worse artefact than none: it looks like a
	// lighting bug rather than a data problem, and only on some faces.
	const FlatScene s = build(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  \"normal N\" [ 0 0 1 ]\n");
	ASSERT_EQ(s.triangles.size(), 1u);
	EXPECT_FALSE(s.triangles[0].hasNormals);
	EXPECT_TRUE(warned(s, "fewer normals than vertices"));
}

TEST(PbrtSubdivTest, MaterialAndAreaLightStillAttachToARefinedSurface) {
	// The refined triangles are new objects that never existed in the source
	// file, so their material and emission tags are assigned rather than
	// copied - worth pinning that they are assigned correctly.
	const FlatScene s = build(
		"AttributeBegin\n"
		"  Material \"diffuse\" \"rgb reflectance\" [ 0.1 0.2 0.3 ]\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n" +
		subdiv(1) +
		"AttributeEnd\n");
	ASSERT_GT(s.triangles.size(), 0u);
	for (const Triangle &t : s.triangles) {
		ASSERT_GE(t.material, 0);
		EXPECT_NEAR(s.materials[t.material].color[2], 0.3, 1e-9);
		EXPECT_GE(t.areaLight, 0) << "a refined triangle lost its emission";
	}
}
