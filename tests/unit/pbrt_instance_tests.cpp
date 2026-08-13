/**
 * @file pbrt_instance_tests.cpp
 * @brief Unit tests for pbrt object instancing
 *
 * Instancing exists so a scene can place 10,000 copies of a tree without
 * storing 10,000 trees. That makes the central property a NEGATIVE one -
 * geometry must NOT be duplicated - which is invisible to any test that only
 * checks the picture looks right. Most of these assert on counts for that
 * reason.
 *
 * The one deliberate exception is emitters. Lights are enumerated into a flat
 * list and sampled from a distribution rather than traversed, so every copy
 * needs its own entry no matter how the geometry is stored. Those ARE
 * duplicated, on purpose, and tested for it.
 */

#include <gtest/gtest.h>

#include "pbrt_flatten.h"
#include "pbrt_scene.h"
#include "pbrt_cpu_builder.h"

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

// One triangle, defined once and placed twice at different offsets.
const char *kTwoInstances =
	"ObjectBegin \"tri\"\n"
	"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
	"    \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
	"ObjectEnd\n"
	"AttributeBegin\n"
	"  Translate 10 0 0\n"
	"  ObjectInstance \"tri\"\n"
	"AttributeEnd\n"
	"AttributeBegin\n"
	"  Translate 20 0 0\n"
	"  ObjectInstance \"tri\"\n"
	"AttributeEnd\n";

} // namespace

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

TEST(PbrtInstanceParseTest, ShapesInsideADefinitionAreNotSceneGeometry) {
	// The distinction the whole feature rests on: defining an object is not
	// drawing one. Before this, ObjectBegin's shapes were emitted in place,
	// so a definition that was never instanced still appeared in the render.
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"ObjectBegin \"unused\"\n"
		"  Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"ObjectEnd\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_TRUE(r.scene.shapes.empty()) << "a definition leaked into the scene";
	ASSERT_EQ(r.scene.objects.size(), 1u);
	EXPECT_EQ(r.scene.objects[0].shapes.size(), 1u);
	EXPECT_EQ(r.scene.objects[0].name, "unused");
}

TEST(PbrtInstanceParseTest, AnInstanceRecordsTheTransformWhereItAppeared) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(kTwoInstances);
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.instances.size(), 2u);
	EXPECT_EQ(r.scene.instances[0].name, "tri");
	EXPECT_NEAR(r.scene.instances[0].xform.m[3], 10.0, 1e-12);
	EXPECT_NEAR(r.scene.instances[1].xform.m[3], 20.0, 1e-12);
}

TEST(PbrtInstanceParseTest, NestedDefinitionsAreRefusedRatherThanFlattened) {
	// Silently absorbing the inner object would place its geometry at every
	// placement of the outer one - geometry appearing where the scene never
	// put it. pbrt rejects this too.
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"ObjectBegin \"outer\"\n"
		"  ObjectBegin \"inner\"\n"
		"  ObjectEnd\n"
		"ObjectEnd\n");
	EXPECT_FALSE(r.ok);
	EXPECT_NE(r.error.find("ObjectBegin"), std::string::npos) << r.error;
}

TEST(PbrtInstanceParseTest, ObjectEndWithoutObjectBeginIsFatal) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse("ObjectEnd\n");
	EXPECT_FALSE(r.ok);
}

TEST(PbrtInstanceParseTest, InstancingFromInsideADefinitionIsRefused) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"ObjectBegin \"a\"\n"
		"  ObjectInstance \"b\"\n"
		"ObjectEnd\n");
	EXPECT_FALSE(r.ok);
}

TEST(PbrtInstanceParseTest, ADefinitionRestoresTheGraphicsStateLikeAnyScope) {
	// ObjectBegin pushes attributes as well as opening a definition, so a
	// material set inside must not leak out and repaint the rest of the scene.
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"Material \"diffuse\" \"rgb reflectance\" [ 1 0 0 ]\n"
		"ObjectBegin \"o\"\n"
		"  Material \"diffuse\" \"rgb reflectance\" [ 0 1 0 ]\n"
		"  Shape \"sphere\"\n"
		"ObjectEnd\n"
		"Shape \"sphere\"\n");
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.shapes.size(), 1u);
	ASSERT_EQ(r.scene.objects.size(), 1u);
	EXPECT_NE(r.scene.shapes[0].materialIndex,
			  r.scene.objects[0].shapes[0].materialIndex)
		<< "the material declared inside the definition escaped it";
}

// ---------------------------------------------------------------------------
// Flattening
// ---------------------------------------------------------------------------

TEST(PbrtInstanceTest, GeometryIsStoredOnceNoMatterHowManyPlacements) {
	// The property instancing exists for. Two placements, one triangle.
	const FlatScene s = build(kTwoInstances);
	ASSERT_EQ(s.groups.size(), 1u);
	EXPECT_EQ(s.groups[0].triangles.size(), 1u)
		<< "the instanced geometry was duplicated, which defeats the point";
	EXPECT_EQ(s.instances.size(), 2u);
	EXPECT_TRUE(s.triangles.empty())
		<< "instanced geometry leaked into the world-space list";
}

TEST(PbrtInstanceTest, GroupGeometryStaysInObjectSpace) {
	// If the instance transform were baked in here, every placement would sit
	// on top of the first one.
	const FlatScene s = build(kTwoInstances);
	ASSERT_EQ(s.groups.size(), 1u);
	ASSERT_EQ(s.groups[0].triangles.size(), 1u);
	EXPECT_NEAR(s.groups[0].triangles[0].v[0], 0.0, 1e-12)
		<< "an instance transform was baked into shared geometry";
}

TEST(PbrtInstanceTest, EachPlacementCarriesItsOwnTransform) {
	const FlatScene s = build(kTwoInstances);
	ASSERT_EQ(s.instances.size(), 2u);
	EXPECT_EQ(s.instances[0].group, 0);
	EXPECT_EQ(s.instances[1].group, 0);
	EXPECT_NEAR(s.instances[0].xform[3], 10.0, 1e-12);
	EXPECT_NEAR(s.instances[1].xform[3], 20.0, 1e-12);
}

TEST(PbrtInstanceTest, TheInstanceTransformComposesWithTheShapesOwn) {
	// A shape placed off-origin inside the definition, then instanced with a
	// further translate. Getting the multiplication order backwards puts it at
	// the wrong place, and with translation-only transforms both orders look
	// plausible until you check.
	const FlatScene s = build(
		"ObjectBegin \"o\"\n"
		"  Scale 2 2 2\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 1 0 0  1 0 0  1 0 0 ]\n"
		"ObjectEnd\n"
		"Translate 100 0 0\n"
		"ObjectInstance \"o\"\n");
	ASSERT_EQ(s.groups.size(), 1u);
	ASSERT_EQ(s.groups[0].triangles.size(), 1u);
	// Scale is inside the definition, so it is baked into the group.
	EXPECT_NEAR(s.groups[0].triangles[0].v[0], 2.0, 1e-12);
	// The translate is the instance's, so it stays on the instance.
	ASSERT_EQ(s.instances.size(), 1u);
	EXPECT_NEAR(s.instances[0].xform[3], 100.0, 1e-12);
}

TEST(PbrtInstanceTest, RedefiningAnObjectMakesTheLaterDefinitionWin) {
	// The parser warns "the later definition replaces the earlier one" when
	// ObjectBegin reuses a name (pbrt_scene.h's ObjectBegin handling) - but
	// it does not erase the earlier ObjectDecl, only warns and appends a
	// second one under the same name, so flatten() is the one actually
	// responsible for honouring that promise when ObjectInstance resolves
	// the name. A vertex at a distinct x tells the two definitions apart.
	const FlatScene s = build(
		"ObjectBegin \"o\"\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 1 0 0  1 0 0  1 0 0 ]\n"
		"ObjectEnd\n"
		"ObjectBegin \"o\"\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 9 0 0  9 0 0  9 0 0 ]\n"
		"ObjectEnd\n"
		"ObjectInstance \"o\"\n");
	EXPECT_TRUE(warned(s, "redefined"));
	ASSERT_EQ(s.instances.size(), 1u);
	const int g = s.instances[0].group;
	ASSERT_GE(g, 0);
	ASSERT_LT(static_cast<std::size_t>(g), s.groups.size());
	ASSERT_EQ(s.groups[g].triangles.size(), 1u);
	EXPECT_NEAR(s.groups[g].triangles[0].v[0], 9.0, 1e-12)
		<< "ObjectInstance resolved to the earlier definition, contradicting "
		   "the parser's own warning about which one would win";
}

TEST(PbrtInstanceTest, AnInstanceOfSomethingUndefinedIsNamedAndSkipped) {
	const FlatScene s = build("ObjectInstance \"ghost\"\n");
	EXPECT_TRUE(s.instances.empty());
	EXPECT_TRUE(warned(s, "ghost"));
}

TEST(PbrtInstanceTest, ADefinitionThatIsNeverInstancedRendersNothing) {
	const FlatScene s = build(
		"ObjectBegin \"unused\"\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"ObjectEnd\n");
	EXPECT_TRUE(s.triangles.empty());
	EXPECT_TRUE(s.instances.empty());
	EXPECT_TRUE(s.empty()) << "an uninstanced definition still put geometry in the scene";
}

TEST(PbrtInstanceTest, PlainSceneGeometryIsUnaffectedByTheInstancePath) {
	// The regression guard: routing every shape through the work list must not
	// change what a scene with no instances at all produces.
	const FlatScene s = build(
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"Shape \"sphere\" \"float radius\" [ 2 ]\n");
	EXPECT_EQ(s.triangles.size(), 1u);
	EXPECT_EQ(s.spheres.size(), 1u);
	EXPECT_TRUE(s.groups.empty());
	EXPECT_TRUE(s.instances.empty());
}

// ---------------------------------------------------------------------------
// Emitters inside instances
// ---------------------------------------------------------------------------

TEST(PbrtInstanceTest, AnEmissiveShapeIsBakedOncePerPlacement) {
	// Deliberately duplicated. A light is sampled from a flat distribution, so
	// each copy needs its own entry; sharing them would light the scene from
	// one place and leave the others dark.
	const FlatScene s = build(
		"ObjectBegin \"lamp\"\n"
		"  AttributeBegin\n"
		"    AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"    Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"      \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  AttributeEnd\n"
		"ObjectEnd\n"
		"AttributeBegin\n Translate 10 0 0\n ObjectInstance \"lamp\"\n AttributeEnd\n"
		"AttributeBegin\n Translate 20 0 0\n ObjectInstance \"lamp\"\n AttributeEnd\n");

	ASSERT_EQ(s.triangles.size(), 2u)
		<< "emitters must exist once per placement to be samplable";
	for (const Triangle &t : s.triangles) EXPECT_GE(t.areaLight, 0);

	// And each copy must be where its instance put it.
	double xs[2] = {s.triangles[0].v[0], s.triangles[1].v[0]};
	if (xs[0] > xs[1]) std::swap(xs[0], xs[1]);
	EXPECT_NEAR(xs[0], 10.0, 1e-12);
	EXPECT_NEAR(xs[1], 20.0, 1e-12);
}

TEST(PbrtInstanceTest, AnEmissiveShapeIsNotAlsoLeftInTheSharedGroup) {
	// If it were in both, the geometry would be drawn twice at every
	// placement - once instanced, once baked - and z-fight with itself.
	const FlatScene s = build(
		"ObjectBegin \"lamp\"\n"
		"  AttributeBegin\n"
		"    AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"    Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"      \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n"
		"  AttributeEnd\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ 0 0 0  0 1 0  0 0 1 ]\n"
		"ObjectEnd\n"
		"ObjectInstance \"lamp\"\n");
	ASSERT_EQ(s.groups.size(), 1u);
	EXPECT_EQ(s.groups[0].triangles.size(), 1u)
		<< "the emissive triangle was left in the shared group as well";
	for (const Triangle &t : s.groups[0].triangles)
		EXPECT_LT(t.areaLight, 0);
	EXPECT_EQ(s.triangles.size(), 1u);
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------
// The CPU builder places instances with transform_instance rather than baking
// them. The property worth asserting is the negative one: the shared geometry
// is built once however many placements refer to it.

TEST(PbrtInstanceBuildTest, PlacementsShareOneBuiltCopyOfTheGeometry) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(kTwoInstances);
	ASSERT_TRUE(r.ok) << r.error;
	const FlatScene f = flatten(r.scene);
	const pbrt_cpu::BuildResult b = pbrt_cpu::build(f);

	EXPECT_EQ(b.instanceCount, 2u);
	// One triangle built, not two: triangleCount counts what was constructed,
	// and both placements point at it.
	EXPECT_EQ(b.triangleCount, 1u)
		<< "the geometry was built once per placement, which is baking";
}

TEST(PbrtInstanceBuildTest, AnInstancedTriangleIsActuallyHitWhereItWasPlaced) {
	// Counts alone would pass if the transform were dropped and everything sat
	// at the origin, so this fires a ray at where the second placement should
	// be and insists something is there.
	const pbrt_scene::ParseResult r = pbrt_scene::parse(kTwoInstances);
	ASSERT_TRUE(r.ok) << r.error;
	const pbrt_cpu::BuildResult b = pbrt_cpu::build(flatten(r.scene));
	ASSERT_TRUE(b.world != nullptr);

	// The triangle spans (0,0,0)-(1,0,0)-(0,1,0), placed at x+10 and x+20.
	// Aim down -z through a point inside the second copy.
	const ray hitting(point3(20.2, 0.2, 5), vec3(0, 0, -1));
	hit_record rec;
	EXPECT_TRUE(b.world->hit(hitting, interval(0.001, infinity), rec))
		<< "nothing at the second placement - the instance transform was lost";

	// And nothing at the origin, where un-transformed geometry would sit.
	const ray missing(point3(0.2, 0.2, 5), vec3(0, 0, -1));
	hit_record rec2;
	EXPECT_FALSE(b.world->hit(missing, interval(0.001, infinity), rec2))
		<< "geometry is at the origin, so the placement transform was ignored";
}

TEST(PbrtInstanceBuildTest, AnInstancedEmitterEndsUpInTheLightList) {
	// Baked per placement by flatten, so each copy should arrive as ordinary
	// world geometry AND be registered for next-event estimation. A light that
	// exists but is not in this list still glows when a ray happens to hit it
	// and contributes nothing to the lighting of anything else.
	const char *kLamp = R"PBRT(
ObjectBegin "lamp"
  AttributeBegin
    AreaLightSource "diffuse" "rgb L" [ 5 5 5 ]
    Shape "trianglemesh" "integer indices" [ 0 1 2 ]
      "point3 P" [ 0 0 0  1 0 0  0 1 0 ]
  AttributeEnd
ObjectEnd
AttributeBegin
  Translate 10 0 0
  ObjectInstance "lamp"
AttributeEnd
AttributeBegin
  Translate 20 0 0
  ObjectInstance "lamp"
AttributeEnd
)PBRT";

	const pbrt_scene::ParseResult r = pbrt_scene::parse(kLamp);
	ASSERT_TRUE(r.ok) << r.error;
	const pbrt_cpu::BuildResult b = pbrt_cpu::build(flatten(r.scene));
	ASSERT_TRUE(b.lights != nullptr);
	EXPECT_EQ(b.lights->objects.size(), 2u)
		<< "each placement of an emitter needs its own entry to be sampled";
	// And the shared group is empty, so nothing was instanced as well as baked.
	EXPECT_EQ(b.instanceCount, 0u)
		<< "an emitter-only definition should place no instanced geometry";
}

