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
	// "conductor" (not "coateddiffuse" - see
	// CoatedDiffuseReflectanceImagemapIsResolvedNotWarned below for why that
	// one now resolves instead of warning): a texture-bound "reflectance" on
	// a material kind neither Diffuse nor CoatedDiffuse gate handles must
	// still warn rather than silently rendering flat.
	const FlatScene s = build(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"conductor\" \"texture reflectance\" [ \"tmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"))
		<< "a textured material renders flat and says nothing about it";
	EXPECT_TRUE(warned(s, "constant colour"));
}

TEST(PbrtDroppedTest, CoatedDiffuseReflectanceImagemapIsResolvedNotWarned) {
	// Gap 4 (pbrt's own ganesha/barcelona-pavilion scenes): CoatedDiffuse
	// joined Diffuse in resolving a texture-bound "reflectance" bound to an
	// "imagemap" for real, instead of warning and falling back to a flat
	// colour - see pbrt_flatten::Material::textureFilename's own comment.
	const FlatScene s = build(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"t.png\" ]\n"
		"Material \"coateddiffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "reflectance"));
	EXPECT_EQ(s.materials[0].textureFilename, "t.png");
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

TEST(PbrtDroppedTest, DiffuseReflectanceCheckerboardIsResolvedNotWarned) {
	// Round 5 Phase 2: named-material-and-texture.pbrt's "floor-check" -
	// Texture "floor-check" "spectrum" "checkerboard" "float uscale" [8]
	// "float vscale" [8] - no tex1/tex2 given, so pbrt-v4's own defaults
	// (white/black) apply. Must resolve to Material::hasCheckerReflectance,
	// not the generic "not supported" warning.
	const FlatScene s = build(
		"Texture \"floor-check\" \"spectrum\" \"checkerboard\" "
		"\"float uscale\" [ 8 ] \"float vscale\" [ 8 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"floor-check\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasCheckerReflectance);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerColor1[0], 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerColor2[0], 0.0);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerUScale, 8.0);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerVScale, 8.0);
}

TEST(PbrtDroppedTest, CheckerboardWithExplicitColoursOverridesTheDefaults) {
	const FlatScene s = build(
		"Texture \"chk\" \"spectrum\" \"checkerboard\" "
		"\"rgb tex1\" [ 1 0 0 ] \"rgb tex2\" [ 0 0 1 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	ASSERT_TRUE(s.materials[0].hasCheckerReflectance);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerColor1[0], 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerColor1[2], 0.0);
	EXPECT_DOUBLE_EQ(s.materials[0].checkerColor2[2], 1.0);
}

TEST(PbrtDroppedTest, CheckerboardOnANonDiffuseMaterialStillWarns) {
	// Checkerboard/fbm/marble/mix reflectance resolution used to be gated on
	// MaterialKind::Diffuse only (coateddiffuse warned here); now also
	// resolved for CoatedDiffuse (see pbrt_flatten_tests.cpp's own
	// CoatedDiffuseReflectanceCheckerboardResolvesToAProceduralTexture), so
	// this regression guard moved to diffusetransmission - a kind that
	// genuinely still stays excluded (matches
	// ATextureBoundToAMaterialIsReportedWithItsParameterName's own scope
	// cut for imagemap).
	const FlatScene s = build(
		"Texture \"chk\" \"spectrum\" \"checkerboard\"\n"
		"Material \"diffusetransmission\" \"texture reflectance\" [ \"chk\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasCheckerReflectance);
}

TEST(PbrtDroppedTest, CheckerboardWithNestedImagemapTex1IsResolvedNotWarned) {
	// One level of nesting: tex1 bound to another Texture that is ITSELF a
	// bare "imagemap" now resolves to Material::checkerTex1Filename instead
	// of warning - see hasCheckerReflectance's own comment. tex2 stays the
	// flat literal default (white), independently.
	const FlatScene s = build(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"chk\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"leaf\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasCheckerReflectance);
	EXPECT_EQ(s.materials[0].checkerTex1Filename, "leaf.png");
	EXPECT_TRUE(s.materials[0].checkerTex2Filename.empty());
}

TEST(PbrtDroppedTest, CheckerboardNestedTex1UsesTheLastRedeclaredTexture) {
	// Regression guard: resolveNestedImagemap() must honour pbrt's own
	// "later Texture declaration with the same name overrides the earlier
	// one" convention, matching every OTHER texture-name lookup in this same
	// loop (e.g. the plain "reflectance" lookup) - it must NOT resolve to a
	// stale earlier declaration just because it happened to be an imagemap.
	const FlatScene s = build(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"a.png\" ]\n"
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"b.png\" ]\n"
		"Texture \"chk\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"leaf\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	ASSERT_TRUE(s.materials[0].hasCheckerReflectance);
	EXPECT_EQ(s.materials[0].checkerTex1Filename, "b.png");
}

TEST(PbrtDroppedTest, CheckerboardWithTex1NestedToANonImagemapStillWarns) {
	// TWO levels of nesting (tex1 -> a Texture that is itself a checkerboard,
	// not a bare imagemap) stays unsupported - a documented scope cut, not a
	// regression of the one-level case above.
	const FlatScene s = build(
		"Texture \"inner\" \"spectrum\" \"checkerboard\"\n"
		"Texture \"chk\" \"spectrum\" \"checkerboard\" \"texture tex1\" [ \"inner\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"chk\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasCheckerReflectance);
}

TEST(PbrtDroppedTest, DiffuseReflectanceFbmIsResolvedNotWarned) {
	// Round 6 Phase 1: pbrt-v4 FBmTexture bound to reflectance - param names
	// match pbrt-v4 exactly ("octaves", "roughness"), resolving to
	// Material::hasFbmReflectance rather than the generic "not supported"
	// warning.
	const FlatScene s = build(
		"Texture \"cloud\" \"float\" \"fbm\" \"integer octaves\" [ 4 ] "
		"\"float roughness\" [ 0.3 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasFbmReflectance);
	EXPECT_EQ(s.materials[0].fbmOctaves, 4);
	EXPECT_DOUBLE_EQ(s.materials[0].fbmRoughness, 0.3);
}

TEST(PbrtDroppedTest, DiffuseReflectanceFbmDefaultsMatchPbrtV4) {
	const FlatScene s = build(
		"Texture \"cloud\" \"float\" \"fbm\"\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"cloud\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	ASSERT_TRUE(s.materials[0].hasFbmReflectance);
	EXPECT_EQ(s.materials[0].fbmOctaves, 8);
	EXPECT_DOUBLE_EQ(s.materials[0].fbmRoughness, 0.5);
}

TEST(PbrtDroppedTest, DiffuseReflectanceMarbleIsResolvedNotWarned) {
	// Round 6 Phase 1: pbrt-v4 MarbleTexture bound to reflectance.
	const FlatScene s = build(
		"Texture \"stone\" \"spectrum\" \"marble\" \"integer octaves\" [ 6 ] "
		"\"float roughness\" [ 0.4 ] \"float scale\" [ 2.0 ] "
		"\"float variation\" [ 0.3 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasMarbleReflectance);
	EXPECT_EQ(s.materials[0].marbleOctaves, 6);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleRoughness, 0.4);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleScale, 2.0);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleVariation, 0.3);
}

TEST(PbrtDroppedTest, DiffuseReflectanceMarbleDefaultsMatchPbrtV4) {
	const FlatScene s = build(
		"Texture \"stone\" \"spectrum\" \"marble\"\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"stone\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	ASSERT_TRUE(s.materials[0].hasMarbleReflectance);
	EXPECT_EQ(s.materials[0].marbleOctaves, 8);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleRoughness, 0.5);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleScale, 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].marbleVariation, 0.2);
}

TEST(PbrtDroppedTest, DiffuseReflectanceMixIsResolvedNotWarned) {
	// Round 6 Phase 1: pbrt-v4 SpectrumMixTexture bound to reflectance,
	// flat-literal tex1/tex2/amount only (see Material::hasMixReflectance's
	// own comment) - same scope cut as checkerboard's tex1/tex2.
	const FlatScene s = build(
		"Texture \"dirt\" \"spectrum\" \"mix\" \"rgb tex1\" [ 1 0 0 ] "
		"\"rgb tex2\" [ 0 0 1 ] \"float amount\" [ 0.25 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasMixReflectance);
	EXPECT_DOUBLE_EQ(s.materials[0].mixColor1[0], 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].mixColor2[2], 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].mixAmount, 0.25);
}

TEST(PbrtDroppedTest, DiffuseReflectanceMixDefaultsMatchPbrtV4) {
	const FlatScene s = build(
		"Texture \"dirt\" \"spectrum\" \"mix\"\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	ASSERT_TRUE(s.materials[0].hasMixReflectance);
	EXPECT_DOUBLE_EQ(s.materials[0].mixColor1[0], 0.0);
	EXPECT_DOUBLE_EQ(s.materials[0].mixColor2[0], 1.0);
	EXPECT_DOUBLE_EQ(s.materials[0].mixAmount, 0.5);
}

TEST(PbrtDroppedTest, MixWithNestedImagemapTex2IsResolvedNotWarned) {
	// One level of nesting for mix's own tex1/tex2 - same support as
	// checkerboard's identical tex1/tex2 above.
	const FlatScene s = build(
		"Texture \"leaf\" \"spectrum\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"texture tex2\" [ \"leaf\" ] \"float amount\" [ 0.4 ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].hasMixReflectance);
	EXPECT_TRUE(s.materials[0].mixTex1Filename.empty());
	EXPECT_EQ(s.materials[0].mixTex2Filename, "leaf.png");
	EXPECT_DOUBLE_EQ(s.materials[0].mixAmount, 0.4);
}

TEST(PbrtDroppedTest, MixWithTex1NestedToANonImagemapStillWarns) {
	// TWO levels of nesting stays unsupported, same scope cut as
	// checkerboard's identical case above.
	const FlatScene s = build(
		"Texture \"inner\" \"float\" \"fbm\"\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"texture tex1\" [ \"inner\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasMixReflectance);
}

TEST(PbrtDroppedTest, MixWithNestedAmountStillWarns) {
	// "amount" bound to another Texture (rather than a float literal) isn't
	// supported - see Material::hasMixReflectance's own comment, same scope
	// cut as checkerboard's tex1/tex2.
	const FlatScene s = build(
		"Texture \"grime\" \"float\" \"fbm\"\n"
		"Texture \"dirt\" \"spectrum\" \"mix\" \"texture amount\" [ \"grime\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"dirt\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "reflectance"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_FALSE(s.materials[0].hasMixReflectance);
}

TEST(PbrtDroppedTest, ShapeAlphaImagemapResolvesOntoTheOwningMaterial) {
	// barcelona-pavilion's foliage: each leaf Shape "plymesh" gives its own
	// "texture alpha" naming a "float"/"imagemap" Texture, independent of
	// (but declared right alongside) its own Material - see Material::
	// alphaTextureFilename's own comment for why this lands on the Shape's
	// resolved material rather than a new per-triangle field.
	const FlatScene s = build(
		"Texture \"leafAlpha\" \"float\" \"imagemap\" \"string filename\" [ \"leaf.png\" ]\n"
		"Material \"diffusetransmission\"\n"
		"Shape \"trianglemesh\" \"texture alpha\" [ \"leafAlpha\" ]\n"
		"  \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].alphaTextureFilename, "leaf.png");
}

TEST(PbrtDroppedTest, AShapeWithNoAlphaParameterLeavesItsMaterialUnmasked) {
	const FlatScene s = build(
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].alphaTextureFilename.empty());
}

TEST(PbrtDroppedTest, DisplacementImagemapResolvesOntoTheMaterialRegardlessOfKind) {
	// barcelona-pavilion's "pavet" material: MakeNamedMaterial ... "string
	// type" ["coateddiffuse"] "texture displacement" ["pavet-bump"] - real
	// scenes bind displacement on coateddiffuse/dielectric/etc, not just
	// Diffuse, so this must NOT be gated on MaterialKind the way
	// textureFilename's reflectance binding is.
	const FlatScene s = build(
		"Texture \"bmap\" \"float\" \"imagemap\" \"string filename\" [ \"bump.png\" ]\n"
		"Material \"coateddiffuse\" \"texture displacement\" [ \"bmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_FALSE(warned(s, "not supported"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].displacementTextureFilename, "bump.png");
	EXPECT_DOUBLE_EQ(s.materials[0].displacementScale, 1.0);
}

TEST(PbrtDroppedTest, DisplacementWrappedInAScaleTextureCapturesTheScaleFactor) {
	// barcelona-pavilion's water material: "texture displacement"
	// ["water-bump"], where "water-bump" is itself "float" "scale"
	// wrapping a nested "texture tex" imagemap.
	const FlatScene s = build(
		"Texture \"waterBase\" \"float\" \"imagemap\" \"string filename\" [ \"water.png\" ]\n"
		"Texture \"waterBump\" \"float\" \"scale\" \"float scale\" [ 0.005 ]"
		" \"texture tex\" [ \"waterBase\" ]\n"
		"Material \"dielectric\" \"texture displacement\" [ \"waterBump\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_EQ(s.materials[0].displacementTextureFilename, "water.png");
	EXPECT_DOUBLE_EQ(s.materials[0].displacementScale, 0.005);
}

TEST(PbrtDroppedTest, AnUnresolvableDisplacementTextureFallsBackToTheGenericWarning) {
	const FlatScene s = build(
		"Texture \"notAnImage\" \"float\" \"checkerboard\"\n"
		"Material \"diffuse\" \"texture displacement\" [ \"notAnImage\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	EXPECT_TRUE(warned(s, "displacement"));
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].displacementTextureFilename.empty());
}

TEST(PbrtDroppedTest, AShapeWithNoDisplacementParameterLeavesItsMaterialUnbumped) {
	const FlatScene s = build(
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	ASSERT_EQ(s.materials.size(), 1u);
	EXPECT_TRUE(s.materials[0].displacementTextureFilename.empty());
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
