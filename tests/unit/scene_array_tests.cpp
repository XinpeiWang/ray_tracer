// scene_array_tests.cpp - src/shared/scene_array.h: the Scene Builder's duplicate, grid and ring arrays and random scatter.
#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../src/shared/scene_array.h"

using namespace scene_doc;

namespace {

Object tree(const std::string& name = "Tree", double x = 0.0, double z = 0.0) {
	Object o = makeObject(ShapeKind::Cone, name);
	o.radius = 0.5;
	o.height = 2.0;
	o.position = {x, 1.0, z};
	o.rotation = {0.0, 10.0, 0.0};
	o.material.color = {0.1, 0.5, 0.1};
	return o;
}

double distance2d(const Object& a, const Object& b) {
	return std::hypot(a.position.x - b.position.x, a.position.z - b.position.z);
}

}  // namespace

// ---- names and sizes ----------------------------------------------------------------------------------------------------------------

TEST(SceneArrayNamesTest, ACopyIsNumberedFromTheSource) {
	std::vector<Object> existing = {tree("Chair")};
	EXPECT_EQ(uniqueObjectName("Chair", existing), "Chair 2");
	existing.push_back(tree("Chair 2"));
	EXPECT_EQ(uniqueObjectName("Chair", existing), "Chair 3") << "the first free number";
	EXPECT_EQ(uniqueObjectName("Chair 2", existing), "Chair 3") << "a copy of a copy is not 'Chair 2 copy'";
	EXPECT_EQ(uniqueObjectName("Chair copy", existing), "Chair 3") << "the old ' copy' suffix is dropped";
	EXPECT_EQ(uniqueObjectName("Table leg 1", {tree("Table leg 1"), tree("Table leg 2")}), "Table leg 3");
}

TEST(SceneArrayNamesTest, PendingNamesAreTakenToo) {
	const std::vector<Object> existing = {tree("Rock")};
	const std::vector<Object> pending = {tree("Rock 2"), tree("Rock 3")};
	EXPECT_EQ(uniqueObjectName("Rock", existing, pending), "Rock 4");
}

TEST(SceneArrayNamesTest, ANameThatIsOnlyDigitsOrEndsInOneWordIsKept) {
	EXPECT_EQ(uniqueObjectName("7", {}), "7 2") << "nothing but digits has no base to strip";
	EXPECT_EQ(uniqueObjectName("Box2", {}), "Box2 2") << "digits without a space are part of the name";
}

TEST(SceneArraySizeTest, ScalingChangesEveryLengthButNotTheShapeOrPlace) {
	Object o = tree();
	o.size = {1.0, 2.0, 3.0};
	o.radius2 = 0.1;
	o.meshScale = 2.0;
	const Float3 position = o.position;
	scaleObject(o, 1.5);
	EXPECT_DOUBLE_EQ(o.radius, 0.75);
	EXPECT_DOUBLE_EQ(o.radius2, 0.15);
	EXPECT_DOUBLE_EQ(o.height, 3.0);
	EXPECT_DOUBLE_EQ(o.size.x, 1.5);
	EXPECT_DOUBLE_EQ(o.size.y, 3.0);
	EXPECT_DOUBLE_EQ(o.size.z, 4.5);
	EXPECT_DOUBLE_EQ(o.meshScale, 3.0);
	EXPECT_EQ(o.shape, ShapeKind::Cone);
	EXPECT_EQ(o.position.x, position.x);
	EXPECT_EQ(o.position.y, position.y);
}

TEST(SceneArraySizeTest, AFootprintIsTheSidewaysReachOnly) {
	EXPECT_DOUBLE_EQ(footprintRadius(tree()), 0.5) << "a cone's height does not widen its footprint";
	Object box = makeObject(ShapeKind::Box, "Box");
	box.size = {3.0, 10.0, 4.0};
	EXPECT_DOUBLE_EQ(footprintRadius(box), 2.5) << "half the diagonal on the floor";
	Object sphere = makeObject(ShapeKind::Sphere, "Ball");
	sphere.radius = 0.7;
	EXPECT_DOUBLE_EQ(footprintRadius(sphere), 0.7);
}

// ---- duplicate ------------------------------------------------------------------------------------------------------------------------

TEST(SceneArrayDuplicateTest, ACopyIsNumberedMovedAside_AndLeavesTheSourceAlone) {
	const Object src = tree("Pine");
	const std::vector<Object> existing = {src};
	const Object copy = duplicateOf(src, existing);
	EXPECT_EQ(copy.name, "Pine 2");
	EXPECT_GT(copy.position.x, src.position.x) << "moved along +X";
	EXPECT_DOUBLE_EQ(std::fmod(copy.position.x * 4.0, 1.0), 0.0) << "to the 0.25 grid";
	EXPECT_EQ(copy.position.y, src.position.y);
	EXPECT_EQ(copy.position.z, src.position.z);
	EXPECT_EQ(copy.shape, src.shape);
	EXPECT_EQ(copy.rotation.y, src.rotation.y);
	EXPECT_EQ(copy.material.color.g, src.material.color.g);
	EXPECT_EQ(existing[0].position.x, src.position.x);
}

TEST(SceneArrayDuplicateTest, TheStepFollowsTheWidthWithinLimits) {
	Object tiny = makeObject(ShapeKind::Sphere, "Bead");
	tiny.radius = 0.05;
	EXPECT_DOUBLE_EQ(duplicateOf(tiny, {}).position.x - tiny.position.x, 0.5) << "never less than half a unit";
	Object floor = makeObject(ShapeKind::Quad, "Floor");
	floor.size = {40.0, 1.0, 40.0};
	EXPECT_DOUBLE_EQ(duplicateOf(floor, {}).position.x - floor.position.x, 4.0) << "never more than four";
	Object ball = makeObject(ShapeKind::Sphere, "Ball");
	ball.radius = 1.0;
	EXPECT_DOUBLE_EQ(duplicateOf(ball, {}).position.x - ball.position.x, 2.25) << "about a diameter and a bit, on the grid";
}

// ---- grid -------------------------------------------------------------------------------------------------------------------------------

TEST(SceneArrayGridTest, ALineHasTheSourceAndThenOneStepEach) {
	GridParams p;
	p.countX = 4;
	p.spacing = {1.5, 0, 0};
	const Object src = tree("Fence post", 2.0, -1.0);
	const std::vector<Object> copies = makeGrid(src, p, {src});
	ASSERT_EQ(copies.size(), 3u) << "four in all, the source is one of them";
	for (size_t i = 0; i < copies.size(); ++i) {
		EXPECT_DOUBLE_EQ(copies[i].position.x, 2.0 + 1.5 * static_cast<double>(i + 1));
		EXPECT_DOUBLE_EQ(copies[i].position.z, -1.0);
		EXPECT_DOUBLE_EQ(copies[i].position.y, src.position.y);
		EXPECT_EQ(copies[i].shape, ShapeKind::Cone);
		EXPECT_DOUBLE_EQ(copies[i].rotation.y, 10.0);
	}
}

TEST(SceneArrayGridTest, ABlockFillsAllThreeAxes) {
	GridParams p;
	p.countX = 2; p.countY = 2; p.countZ = 3;
	p.spacing = {1.0, 10.0, 100.0};
	const Object src = tree();
	const std::vector<Object> copies = makeGrid(src, p, {src});
	ASSERT_EQ(copies.size(), 11u);
	EXPECT_EQ(gridRequested(p), 11);
	std::set<std::string> names;
	std::set<std::tuple<double, double, double>> spots = {{src.position.x, src.position.y, src.position.z}};
	for (const Object& o : copies) {
		names.insert(o.name);
		spots.insert({o.position.x, o.position.y, o.position.z});
	}
	EXPECT_EQ(names.size(), 11u) << "every copy has its own name";
	EXPECT_EQ(spots.size(), 12u) << "and its own place";
	EXPECT_EQ(names.count("Tree"), 0u) << "none reuses the source's name";
}

TEST(SceneArrayGridTest, ACountOfOneAddsNothing_AndTheLimitHolds) {
	GridParams one;
	one.countX = 1;
	EXPECT_TRUE(makeGrid(tree(), one, {}).empty());
	GridParams zero;
	zero.countX = 0; zero.countY = -4;
	EXPECT_TRUE(makeGrid(tree(), zero, {}).empty()) << "a nonsense count is one";
	GridParams huge;
	huge.countX = 50; huge.countY = 50; huge.countZ = 50;
	EXPECT_EQ(makeGrid(tree(), huge, {}).size(), static_cast<size_t>(kMaxArrayCopies));
	EXPECT_EQ(gridRequested(huge), 125000 - 1) << "the dialog can say how many it asked for";
}

// ---- ring ---------------------------------------------------------------------------------------------------------------------------------

TEST(SceneArrayRingTest, CopiesSitOnTheCircleSpacedFromTheSource) {
	RingParams p;
	p.count = 4;
	p.radius = 3.0;
	p.turnWithRing = false;
	const Object src = tree("Chair", 3.0, 0.0);   // at angle 0 around the origin
	const std::vector<Object> copies = makeRing(src, p, {src});
	ASSERT_EQ(copies.size(), 3u) << "four in all, the source included";
	EXPECT_NEAR(copies[0].position.x, 0.0, 1e-9);  EXPECT_NEAR(copies[0].position.z, 3.0, 1e-9);
	EXPECT_NEAR(copies[1].position.x, -3.0, 1e-9); EXPECT_NEAR(copies[1].position.z, 0.0, 1e-9);
	EXPECT_NEAR(copies[2].position.x, 0.0, 1e-9);  EXPECT_NEAR(copies[2].position.z, -3.0, 1e-9);
	for (const Object& o : copies) {
		EXPECT_EQ(o.position.y, src.position.y);
		EXPECT_EQ(o.rotation.y, src.rotation.y) << "not turned when 'face the centre' is off";
	}
}

TEST(SceneArrayRingTest, TheRingCanBeAroundAnyPointAndHaveAnyRadius) {
	RingParams p;
	p.count = 3;
	p.radius = 2.0;
	p.centreX = 10.0; p.centreZ = 5.0;
	p.turnWithRing = false;
	const std::vector<Object> copies = makeRing(tree("Chair", 12.0, 5.0), p, {});
	ASSERT_EQ(copies.size(), 2u);
	for (const Object& o : copies) EXPECT_NEAR(std::hypot(o.position.x - 10.0, o.position.z - 5.0), 2.0, 1e-9);
	p.radius = 5.0;   // the source is not on this ring: the copies still go round it, from the source's angle
	const std::vector<Object> wide = makeRing(tree("Chair", 12.0, 5.0), p, {});
	for (const Object& o : wide) EXPECT_NEAR(std::hypot(o.position.x - 10.0, o.position.z - 5.0), 5.0, 1e-9);
	EXPECT_NEAR(wide[0].position.z, 5.0 + 5.0 * std::sin(120.0 * 3.14159265358979 / 180.0), 1e-9) << "a third of the way round from angle 0";
}

TEST(SceneArrayRingTest, TurningWithTheRingKeepsEachCopyFacingTheCentreAsTheOriginalDoes) {
	RingParams p;
	p.count = 6;
	p.radius = 4.0;
	p.turnWithRing = true;
	Object src = tree("Chair", 4.0, 0.0);
	src.rotation.y = 30.0;   // the original faces the centre at some angle of its own
	const auto facingAngle = [](const Object& o) {   // the angle from where +Z points to the direction of the centre, in degrees
		const double t = o.rotation.y * 3.14159265358979 / 180.0;
		const double fx = std::sin(t), fz = std::cos(t);
		const double cx = -o.position.x, cz = -o.position.z;
		return std::atan2(fx * cz - fz * cx, fx * cx + fz * cz) * 180.0 / 3.14159265358979;
	};
	const double original = facingAngle(src);
	for (const Object& o : makeRing(src, p, {})) EXPECT_NEAR(facingAngle(o), original, 1e-9) << o.name;
	p.turnWithRing = false;
	for (const Object& o : makeRing(src, p, {})) EXPECT_EQ(o.rotation.y, 30.0) << "not turned";
}

TEST(SceneArrayRingTest, AStartAngleSlidesTheWholeRing_AndSmallCountsAddNothing) {
	RingParams p;
	p.count = 4;
	p.radius = 3.0;
	p.turnWithRing = false;
	p.startAngle = 45.0;
	const std::vector<Object> copies = makeRing(tree("Chair", 3.0, 0.0), p, {});
	EXPECT_NEAR(std::atan2(copies[0].position.z, copies[0].position.x) * 180.0 / 3.14159265358979, 135.0, 1e-9) << "one quarter turn plus 45 degrees";
	p.count = 1;
	EXPECT_TRUE(makeRing(tree(), p, {}).empty());
	p.count = 0;
	EXPECT_TRUE(makeRing(tree(), p, {}).empty());
	p.count = 100000;
	EXPECT_EQ(makeRing(tree(), p, {}).size(), static_cast<size_t>(kMaxArrayCopies));
}

// ---- scatter --------------------------------------------------------------------------------------------------------------------------

TEST(SceneArrayScatterTest, TheSameSeedGivesTheSameScatterAndAnotherSeedADifferentOne) {
	ScatterParams p;
	p.count = 30;
	p.seed = 7;
	const Object src = tree();
	const ScatterResult a = scatter(src, p, {src}), b = scatter(src, p, {src});
	ASSERT_EQ(a.objects.size(), b.objects.size());
	for (size_t i = 0; i < a.objects.size(); ++i) {
		EXPECT_EQ(a.objects[i].position.x, b.objects[i].position.x);
		EXPECT_EQ(a.objects[i].position.z, b.objects[i].position.z);
		EXPECT_EQ(a.objects[i].rotation.y, b.objects[i].rotation.y);
		EXPECT_EQ(a.objects[i].radius, b.objects[i].radius);
	}
	p.seed = 8;
	const ScatterResult c = scatter(src, p, {src});
	bool anyDifferent = false;
	for (size_t i = 0; i < std::min(a.objects.size(), c.objects.size()); ++i) anyDifferent = anyDifferent || a.objects[i].position.x != c.objects[i].position.x;
	EXPECT_TRUE(anyDifferent);
}

TEST(SceneArrayScatterTest, CopiesStayInsideTheDiskOrTheRectangle) {
	ScatterParams p;
	p.count = 200;
	p.keepApart = false;
	p.centreX = 5.0; p.centreZ = -3.0;
	p.width = 8.0;
	const Object src = tree();
	for (const Object& o : scatter(src, p, {}).objects) EXPECT_LE(std::hypot(o.position.x - 5.0, o.position.z + 3.0), 4.0 + 1e-9);
	p.disk = false;
	p.depth = 2.0;
	for (const Object& o : scatter(src, p, {}).objects) {
		EXPECT_LE(std::fabs(o.position.x - 5.0), 4.0 + 1e-9);
		EXPECT_LE(std::fabs(o.position.z + 3.0), 1.0 + 1e-9);
	}
}

TEST(SceneArrayScatterTest, WithoutKeepApartExactlyTheRequestedCountIsPlaced) {
	ScatterParams p;
	p.count = 50;
	p.keepApart = false;
	p.width = 1.0;   // far too small for fifty trees to stand apart
	const ScatterResult r = scatter(tree(), p, {});
	EXPECT_EQ(r.objects.size(), 50u);
	EXPECT_EQ(r.requested, 50);
}

TEST(SceneArrayScatterTest, KeepApartStopsFootprintsOverlapping_AndPlacesFewerWhenThereIsNoRoom) {
	ScatterParams p;
	p.count = 40;
	p.width = 12.0;
	p.scaleMin = 0.8; p.scaleMax = 1.2;
	p.keepApart = true;
	const Object src = tree("Pine", 1.0, 1.0);
	const ScatterResult r = scatter(src, p, {src});
	ASSERT_GE(r.objects.size(), 20u) << "plenty of room";
	std::vector<Object> all = r.objects;
	all.push_back(src);
	for (size_t i = 0; i < all.size(); ++i)
		for (size_t j = i + 1; j < all.size(); ++j)
			EXPECT_GE(distance2d(all[i], all[j]), footprintRadius(all[i]) + footprintRadius(all[j]) - 1e-9) << i << " and " << j << " overlap";
	ScatterParams crowded = p;
	crowded.width = 3.0;
	crowded.count = 100;
	const ScatterResult small = scatter(src, crowded, {src});
	EXPECT_LT(small.objects.size(), 100u);
	EXPECT_EQ(small.requested, 100);
	EXPECT_GT(small.objects.size(), 0u);
}

TEST(SceneArrayScatterTest, ScaleSpinAndTiltFollowTheirSettings) {
	ScatterParams p;
	p.count = 100;
	p.keepApart = false;
	p.groundY = 0.0;
	p.scaleMin = 0.5; p.scaleMax = 2.0;
	p.tiltDegrees = 5.0;
	const Object src = tree();   // a cone 2 high, 0.5 radius, standing on y = 0 (its middle is at y = 1)
	double minScale = 1e9, maxScale = 0.0;
	bool spun = false;
	for (const Object& o : scatter(src, p, {}).objects) {
		const double s = o.radius / src.radius;
		minScale = std::min(minScale, s);
		maxScale = std::max(maxScale, s);
		EXPECT_NEAR(o.height, src.height * s, 1e-9);
		EXPECT_NEAR(o.position.y, 1.0 * s, 1e-9) << "still standing on the ground";
		EXPECT_LE(std::fabs(o.rotation.x), 5.0 + 1e-9);
		EXPECT_LE(std::fabs(o.rotation.z), 5.0 + 1e-9);
		spun = spun || std::fabs(o.rotation.y - 10.0) > 1.0;
	}
	EXPECT_GE(minScale, 0.5 - 1e-9);
	EXPECT_LE(maxScale, 2.0 + 1e-9);
	EXPECT_LT(minScale, 0.7) << "the range is used, not just one end";
	EXPECT_GT(maxScale, 1.8);
	EXPECT_TRUE(spun);
	p.randomSpin = false;
	p.tiltDegrees = 0.0;
	for (const Object& o : scatter(src, p, {}).objects) {
		EXPECT_EQ(o.rotation.y, 10.0) << "no spin: the source's own turn";
		EXPECT_EQ(o.rotation.x, 0.0);
		EXPECT_EQ(o.rotation.z, 0.0);
	}
}

TEST(SceneArrayScatterTest, GroundHeightIsWhatCopiesStandOn) {
	ScatterParams p;
	p.count = 10;
	p.keepApart = false;
	p.groundY = 2.0;
	p.scaleMin = p.scaleMax = 2.0;
	Object src = tree();
	src.position.y = 3.0;   // 1 above the ground at y = 2
	for (const Object& o : scatter(src, p, {}).objects) EXPECT_NEAR(o.position.y, 2.0 + 1.0 * 2.0, 1e-9);
}

TEST(SceneArrayScatterTest, NamesAreUnique_AndTheCountIsCapped) {
	ScatterParams p;
	p.count = 100000;
	p.keepApart = false;
	const std::vector<Object> existing = {tree("Tree"), tree("Tree 2")};
	const ScatterResult r = scatter(existing[0], p, existing);
	EXPECT_EQ(r.objects.size(), static_cast<size_t>(kMaxArrayCopies));
	EXPECT_EQ(r.requested, kMaxArrayCopies);
	std::set<std::string> names = {"Tree", "Tree 2"};
	for (const Object& o : r.objects) EXPECT_TRUE(names.insert(o.name).second) << o.name << " is used twice";
	p.count = -5;
	EXPECT_TRUE(scatter(existing[0], p, existing).objects.empty());
}


// ---- units: an object and the parts that go with it ----------------------------------------------------------------------------------

namespace {
// A two-part tree: a trunk (the anchor, selected) and a crown above it, a little to one side.
std::vector<Object> twoPartTree(double x = 0.0, double z = 0.0) {
	Object trunk = makeObject(ShapeKind::Cylinder, "Tree trunk");
	trunk.radius = 0.15; trunk.height = 1.0; trunk.position = {x, 0.5, z};
	Object crown = makeObject(ShapeKind::Cone, "Tree crown");
	crown.radius = 0.8; crown.height = 2.4; crown.position = {x + 0.4, 2.2, z};
	return {trunk, crown};
}
}  // namespace

TEST(SceneArrayUnitTest, AGridCopiesTheWholeUnitKeepingItsParts) {
	GridParams p;
	p.countX = 3;
	p.spacing = {5.0, 0, 0};
	const std::vector<Object> unit = twoPartTree(1.0, 2.0);
	const std::vector<Object> copies = makeGrid(unit, p, unit);
	ASSERT_EQ(copies.size(), 4u) << "two more trees of two parts each";
	for (size_t tree = 0; tree < 2; ++tree) {
		const Object& trunk = copies[tree * 2];
		const Object& crown = copies[tree * 2 + 1];
		EXPECT_DOUBLE_EQ(trunk.position.x, 1.0 + 5.0 * static_cast<double>(tree + 1));
		EXPECT_NEAR(crown.position.x - trunk.position.x, 0.4, 1e-12) << "the crown stays where it was relative to the trunk";
		EXPECT_NEAR(crown.position.y - trunk.position.y, 1.7, 1e-12);
		EXPECT_EQ(trunk.shape, ShapeKind::Cylinder);
		EXPECT_EQ(crown.shape, ShapeKind::Cone);
	}
	std::set<std::string> names = {"Tree trunk", "Tree crown"};
	for (const Object& o : copies) EXPECT_TRUE(names.insert(o.name).second) << o.name;
	EXPECT_EQ(copies[0].name, "Tree trunk 2");
	EXPECT_EQ(copies[1].name, "Tree crown 2");
}

TEST(SceneArrayUnitTest, ARingTurnsTheWholeUnitAboutItsAnchor) {
	Object seat = makeObject(ShapeKind::Box, "Chair seat");
	seat.position = {3.0, 0.5, 0.0};
	Object back = makeObject(ShapeKind::Box, "Chair back");
	back.position = {3.0, 1.0, 0.5};   // half a unit "behind" the seat along +Z
	RingParams p;
	p.count = 4;
	p.radius = 3.0;
	p.turnWithRing = true;
	const std::vector<Object> ring = makeRing({seat, back}, p, {});
	ASSERT_EQ(ring.size(), 6u);
	const auto cross2 = [](double ax, double az, double bx, double bz) { return ax * bz - az * bx; };
	for (size_t i = 0; i < 3; ++i) {
		const Object& s = ring[i * 2];
		const Object& b = ring[i * 2 + 1];
		EXPECT_NEAR(std::hypot(b.position.x - s.position.x, b.position.z - s.position.z), 0.5, 1e-9) << "the back stays half a unit from the seat";
		EXPECT_DOUBLE_EQ(b.position.y - s.position.y, 0.5);
		// The back's offset from the seat makes the same angle with the way to the centre as it did for the original (a quarter turn: 90 degrees).
		const double ox = b.position.x - s.position.x, oz = b.position.z - s.position.z, cx = -s.position.x, cz = -s.position.z;
		EXPECT_NEAR(std::atan2(cross2(ox, oz, cx, cz), ox * cx + oz * cz), std::atan2(cross2(0.0, 0.5, -3.0, 0.0), 0.0), 1e-9) << i;
		EXPECT_DOUBLE_EQ(s.rotation.y, b.rotation.y) << "both parts are turned the same";
	}
}

TEST(SceneArrayUnitTest, AScatterScalesAndTurnsAUnitAsOneThing) {
	ScatterParams p;
	p.count = 30;
	p.keepApart = false;
	p.scaleMin = 0.5; p.scaleMax = 2.0;
	p.groundY = 0.0;
	const std::vector<Object> unit = twoPartTree();
	const ScatterResult r = scatter(unit, p, unit);
	ASSERT_EQ(r.objects.size(), 60u);
	EXPECT_EQ(r.placed, 30);
	for (size_t i = 0; i < r.objects.size(); i += 2) {
		const Object& trunk = r.objects[i];
		const Object& crown = r.objects[i + 1];
		const double s = trunk.radius / 0.15;
		EXPECT_NEAR(crown.radius / 0.8, s, 1e-9) << "one scale for the whole tree";
		EXPECT_NEAR(trunk.position.y, 0.5 * s, 1e-9) << "standing on the ground";
		EXPECT_NEAR(crown.position.y, 2.2 * s, 1e-9);
		EXPECT_NEAR(std::hypot(crown.position.x - trunk.position.x, crown.position.z - trunk.position.z), 0.4 * s, 1e-9) << "the crown stays 0.4 (scaled) to one side, whichever way the tree turned";
		EXPECT_DOUBLE_EQ(trunk.rotation.y, crown.rotation.y);
	}
}

TEST(SceneArrayUnitTest, KeepApartUsesTheWholeUnitsFootprint) {
	ScatterParams p;
	p.count = 40;
	p.width = 14.0;
	p.keepApart = true;
	p.scaleMin = p.scaleMax = 1.0;
	const std::vector<Object> unit = twoPartTree(0.0, 0.0);   // reaches 0.4 + 0.8 = 1.2 from the trunk
	const ScatterResult r = scatter(unit, p, unit);
	ASSERT_GT(r.placed, 5);
	std::vector<std::pair<double, double>> anchors = {{0.0, 0.0}};
	for (size_t i = 0; i < r.objects.size(); i += 2) anchors.push_back({r.objects[i].position.x, r.objects[i].position.z});
	for (size_t i = 0; i < anchors.size(); ++i)
		for (size_t j = i + 1; j < anchors.size(); ++j)
			EXPECT_GE(std::hypot(anchors[i].first - anchors[j].first, anchors[i].second - anchors[j].second), 2.4 - 1e-9) << "two trees' canopies overlap";
}

TEST(SceneArrayUnitTest, ABigUnitIsLimitedByTheObjectCount) {
	Object part = makeObject(ShapeKind::Box, "Block");
	std::vector<Object> unit(40, part);
	for (size_t i = 0; i < unit.size(); ++i) unit[i].name = "Block " + std::to_string(i);
	GridParams g;
	g.countX = 100; g.countY = 100;
	EXPECT_EQ(unitLimit(unit.size()), 50);
	EXPECT_EQ(makeGrid(unit, g, unit).size(), 50u * 40u) << "fifty copies of forty blocks: two thousand objects, not twenty thousand";
	EXPECT_EQ(unitLimit(1), kMaxArrayCopies);
	EXPECT_EQ(unitLimit(0), kMaxArrayCopies);
	EXPECT_EQ(unitLimit(5000), 1) << "even a huge unit may be copied once";
	EXPECT_TRUE(makeGrid(std::vector<Object>(), g, {}).empty());
	EXPECT_TRUE(makeRing(std::vector<Object>(), RingParams(), {}).empty());
	EXPECT_TRUE(scatter(std::vector<Object>(), ScatterParams(), {}).objects.empty());
}

TEST(SceneArrayUnitTest, LikelyCompanionsAreTheSameNamedPartsCloseBy) {
	Object trunk = makeObject(ShapeKind::Cylinder, "Tree trunk");
	trunk.position = {0, 0.5, 0};
	Object crown = makeObject(ShapeKind::Cone, "Tree crown");
	crown.position = {0, 2.0, 0};
	Object farTree = makeObject(ShapeKind::Cylinder, "Tree trunk 2");
	farTree.position = {20, 0.5, 0};
	Object table = makeObject(ShapeKind::Box, "Table top");
	table.position = {0.5, 0.5, 0};
	const std::vector<Object> all = {trunk, crown, farTree, table};
	const std::vector<int> found = likelyCompanions(trunk, all, 0);
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0], 1) << "the crown: the same first word, close by";
	EXPECT_TRUE(likelyCompanions(makeObject(ShapeKind::Sphere, "Ball"), {makeObject(ShapeKind::Sphere, "Ball 2")}, 5).size() == 1u) << "no object of the document is the source here";
	EXPECT_TRUE(likelyCompanions(trunk, {trunk}, 0).empty()) << "never the source itself";
}
