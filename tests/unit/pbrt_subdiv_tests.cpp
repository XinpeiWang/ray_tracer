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
