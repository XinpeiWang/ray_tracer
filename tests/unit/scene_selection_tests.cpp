// scene_selection_tests.cpp - src/shared/scene_selection.h: acting on several Scene Builder items at once (move, copy, delete, group, copy a look).
#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "../../src/shared/scene_selection.h"

using namespace scene_doc;

namespace {

Object ball(const std::string& name, double x, double z = 0.0) {
	Object o = makeObject(ShapeKind::Sphere, name);
	o.radius = 0.5;
	o.position = {x, 0.5, z};
	return o;
}

// A table: four named parts in the group "Table", and a loose ball and a lamp.
Document sceneWithTable() {
	Document d;
	d.objects = {ball("Top", 0.0), ball("Leg", 0.5), ball("Leg 2", -0.5), ball("Ball", 5.0)};
	for (int i = 0; i < 3; ++i) d.objects[i].group = "Table";
	Light lamp;
	lamp.name = "Lamp";
	lamp.kind = LightKind::Spot;
	lamp.position = {1.0, 4.0, 0.0};
	lamp.target = {1.0, 0.0, 0.0};
	d.lights.push_back(lamp);
	return d;
}

}  // namespace

TEST(SceneSelection, ItemSetIsSortedUniqueAndInRange) {
	const Document d = sceneWithTable();
	const ItemSet s = makeItemSet(d, {3, 1, 1, 9, -1}, {0, 5});
	EXPECT_EQ(s.objects, (std::vector<int>{1, 3}));
	EXPECT_EQ(s.lights, (std::vector<int>{0}));
	EXPECT_EQ(s.size(), 3u);
}

TEST(SceneSelection, PickingOneLegPicksTheWholeTable) {
	const Document d = sceneWithTable();
	EXPECT_EQ(withGroupMates(d, makeItemSet(d, {1})).objects, (std::vector<int>{0, 1, 2}));
	EXPECT_EQ(withGroupMates(d, makeItemSet(d, {3})).objects, (std::vector<int>{3})) << "a loose object is just itself";
	EXPECT_EQ(withGroupMates(d, makeItemSet(d, {1, 3})).objects, (std::vector<int>{0, 1, 2, 3}));
}

TEST(SceneSelection, GroupNeedsTwoObjectsAndGetsAFreeName) {
	Document d = sceneWithTable();
	EXPECT_EQ(groupObjects(d, {3}), "") << "one object is not a group";
	EXPECT_TRUE(d.objects[3].group.empty());
	d.objects[3].group = "Group";   // the plain name is taken
	d.objects.push_back(ball("Other", 7.0));
	d.objects.push_back(ball("Another", 8.0));
	const std::string name = groupObjects(d, {4, 5}, "Group");
	EXPECT_EQ(name, "Group 2");
	EXPECT_EQ(d.objects[4].group, "Group 2");
	EXPECT_EQ(d.objects[5].group, "Group 2");
	EXPECT_TRUE(d.objects[3].group.empty()) << "the old 'Group' now has one member, so it is dissolved";
}

TEST(SceneSelection, GroupingAnObjectOfAnotherGroupTakesItOut) {
	Document d = sceneWithTable();
	const std::string name = groupObjects(d, {2, 3});   // a leg and the ball
	EXPECT_FALSE(name.empty());
	EXPECT_EQ(d.objects[2].group, name);
	EXPECT_EQ(d.objects[3].group, name);
	EXPECT_EQ(d.objects[0].group, "Table");
	EXPECT_EQ(d.objects[1].group, "Table") << "the rest of the table is still a table";
}

TEST(SceneSelection, UngroupDissolvesTheWholeGroup) {
	Document d = sceneWithTable();
	EXPECT_EQ(ungroupObjects(d, {1}), 3);
	for (const Object& o : d.objects) EXPECT_TRUE(o.group.empty());
	EXPECT_EQ(ungroupObjects(d, {3}), 0) << "nothing to dissolve";
}

TEST(SceneSelection, TranslateMovesObjectsAndALightWithItsTarget) {
	Document d = sceneWithTable();
	translateItems(d, makeItemSet(d, {0, 3}, {0}), {1.0, 0.0, -2.0});
	EXPECT_DOUBLE_EQ(d.objects[0].position.x, 1.0);
	EXPECT_DOUBLE_EQ(d.objects[0].position.z, -2.0);
	EXPECT_DOUBLE_EQ(d.objects[3].position.x, 6.0);
	EXPECT_DOUBLE_EQ(d.objects[1].position.x, 0.5) << "not picked: not moved";
	EXPECT_DOUBLE_EQ(d.lights[0].position.x, 2.0);
	EXPECT_DOUBLE_EQ(d.lights[0].target.x, 2.0) << "a spot keeps pointing the same way";
	EXPECT_DOUBLE_EQ(d.lights[0].target.z, -2.0);
}

TEST(SceneSelection, DuplicatingOneObjectMatchesTheDuplicateButton) {
	Document d = sceneWithTable();
	const Object expected = duplicateOf(d.objects[3], d.objects);
	const ItemSet added = duplicateItems(d, makeItemSet(d, {3}));
	ASSERT_EQ(added.objects.size(), 1u);
	const Object& copy = d.objects[added.objects[0]];
	EXPECT_EQ(copy.name, expected.name);
	EXPECT_DOUBLE_EQ(copy.position.x, expected.position.x);
	EXPECT_DOUBLE_EQ(copy.position.y, expected.position.y);
}

TEST(SceneSelection, DuplicatingATableMakesAnotherTableBesideIt) {
	Document d = sceneWithTable();
	const ItemSet added = duplicateItems(d, withGroupMates(d, makeItemSet(d, {0})));
	ASSERT_EQ(added.objects.size(), 3u);
	// The copies keep their places relative to each other...
	const Object &top = d.objects[added.objects[0]], &leg = d.objects[added.objects[1]], &leg2 = d.objects[added.objects[2]];
	EXPECT_NEAR(leg.position.x - top.position.x, 0.5, 1e-12);
	EXPECT_NEAR(leg2.position.x - top.position.x, -0.5, 1e-12);
	// ...clear of the originals (the table is 2 wide: 1.5 across the legs plus a leg's radius each side)...
	EXPECT_GT(top.position.x - 0.5, 0.5 + 0.5 - 1e-9) << "the copy's left edge is past the original's right edge";
	// ...numbered, and a group of their own.
	EXPECT_EQ(top.name, "Top 2");
	EXPECT_EQ(leg.name, "Leg 3") << "'Leg 2' was already taken";
	EXPECT_FALSE(top.group.empty());
	EXPECT_NE(top.group, "Table");
	EXPECT_EQ(top.group, leg.group);
	EXPECT_EQ(top.group, leg2.group);
	EXPECT_EQ(d.objects[0].group, "Table") << "the original table is untouched";
}

TEST(SceneSelection, TwoGroupsCopiedTogetherGetTwoNewGroups) {
	Document d;
	d.objects = {ball("A", 0.0), ball("B", 1.0), ball("C", 4.0), ball("D", 5.0)};
	d.objects[0].group = d.objects[1].group = "First";
	d.objects[2].group = d.objects[3].group = "Second";
	const ItemSet added = duplicateItems(d, makeItemSet(d, {0, 1, 2, 3}));
	ASSERT_EQ(added.objects.size(), 4u);
	std::set<std::string> groups;
	for (int i : added.objects) groups.insert(d.objects[i].group);
	EXPECT_EQ(groups.size(), 2u);
	EXPECT_FALSE(groups.count("First") || groups.count("Second"));
}

TEST(SceneSelection, DuplicatingALightNumbersItAndMovesItsTarget) {
	Document d = sceneWithTable();
	const ItemSet added = duplicateItems(d, makeItemSet(d, {}, {0}));
	ASSERT_EQ(added.lights.size(), 1u);
	EXPECT_EQ(d.lights[added.lights[0]].name, "Lamp 2");
	EXPECT_DOUBLE_EQ(d.lights[added.lights[0]].position.x, 1.5);
	EXPECT_DOUBLE_EQ(d.lights[added.lights[0]].target.x, 1.5);
}

TEST(SceneSelection, EraseRemovesTheItemsAndDissolvesALoneMember) {
	Document d = sceneWithTable();
	eraseItems(d, makeItemSet(d, {0, 1, 3}, {0}));
	ASSERT_EQ(d.objects.size(), 1u);
	EXPECT_EQ(d.objects[0].name, "Leg 2");
	EXPECT_TRUE(d.objects[0].group.empty()) << "one leg is not a table";
	EXPECT_TRUE(d.lights.empty());
}

TEST(SceneSelection, CopyLookGivesTheOthersTheMaterialOnly) {
	Document d = sceneWithTable();
	d.objects[3].material.color = {1.0, 0.0, 0.0};
	d.objects[3].material.kind = MaterialKind::Conductor;
	EXPECT_TRUE(copyLook(d, 3, {0, 1, 3}));
	EXPECT_EQ(d.objects[0].material.kind, MaterialKind::Conductor);
	EXPECT_DOUBLE_EQ(d.objects[1].material.color.r, 1.0);
	EXPECT_DOUBLE_EQ(d.objects[0].position.x, 0.0) << "positions are not copied";
	EXPECT_EQ(d.objects[0].name, "Top");
	EXPECT_FALSE(copyLook(d, 9, {0}));
}

TEST(SceneSelection, TheUnitPutsThePrimaryFirst) {
	const Document d = sceneWithTable();
	const std::vector<Object> unit = unitOfObjects(d, 2, {0, 1, 2});
	ASSERT_EQ(unit.size(), 3u);
	EXPECT_EQ(unit[0].name, "Leg 2");
	EXPECT_EQ(unit[1].name, "Top");
	EXPECT_EQ(unit[2].name, "Leg");
	EXPECT_TRUE(unitOfObjects(d, 9, {0}).empty());
}

TEST(SceneSelection, GroupsSurviveTheJsonCopyAMissingOneReadsAsNone) {
	Document d = sceneWithTable();
	Document back;
	std::string error;
	ASSERT_TRUE(fromJson(toJson(d), back, error)) << error;
	ASSERT_EQ(back.objects.size(), d.objects.size());
	EXPECT_EQ(back.objects[1].group, "Table");
	EXPECT_TRUE(back.objects[3].group.empty());
	EXPECT_EQ(toJson(back), toJson(d));
	EXPECT_EQ(toJson(sceneWithTable()).find("\"group\":\"Table\"") != std::string::npos, true);
	Document plain;
	plain.objects = {ball("A", 0.0)};
	EXPECT_EQ(toJson(plain).find("group"), std::string::npos) << "an ungrouped object writes nothing, so old files and new ones agree";
}

TEST(SceneSelection, CopiesOfAGroupUnitAreGroupsOfTheirOwn) {
	Document d = sceneWithTable();
	const std::vector<Object> unit = unitOfObjects(d, 0, {0, 1, 2});
	std::vector<Object> copies;   // two copies of the three-part unit
	for (int copy = 0; copy < 2; ++copy)
		for (const Object& part : unit) copies.push_back(part);
	EXPECT_EQ(groupCopies(d, copies, unit), 2);
	EXPECT_EQ(copies[0].group, copies[2].group);
	EXPECT_EQ(copies[3].group, copies[5].group);
	EXPECT_NE(copies[0].group, copies[3].group);
	EXPECT_NE(copies[0].group, "Table");
	EXPECT_NE(copies[3].group, "Table");
}

TEST(SceneSelection, AUnitThatWasNotOneGroupMakesNoGroups) {
	Document d = sceneWithTable();
	std::vector<Object> unit = {d.objects[3], d.objects[0]};   // the loose ball and a table top
	std::vector<Object> copies = unit;
	EXPECT_EQ(groupCopies(d, copies, unit), 0);
	EXPECT_TRUE(copies[0].group.empty());
	std::vector<Object> alone = {d.objects[0]};
	std::vector<Object> aloneCopies = alone;
	EXPECT_EQ(groupCopies(d, aloneCopies, alone), 0) << "one part is not a group";
}

// ---- turning and scaling several things as one ------------------------------------------------------------------------------------------

namespace {

double dist(const Float3& a, const Float3& b) { return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)); }

void expectNearFloat3(const Float3& a, const Float3& b, double eps = 1e-9) {
	EXPECT_NEAR(a.x, b.x, eps);
	EXPECT_NEAR(a.y, b.y, eps);
	EXPECT_NEAR(a.z, b.z, eps);
}

// The turn an object's three angles stand for, as a matrix (so two sets of angles can be compared whatever form they are in).
scene_view::Mat3 orientation(const Object& o) { return scene_view::rotationXYZ({o.rotation.x, o.rotation.y, o.rotation.z}); }

void expectSameOrientation(const Object& a, const Object& b) {
	const scene_view::Mat3 x = orientation(a), y = orientation(b);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) EXPECT_NEAR(x.m[i][j], y.m[i][j], 1e-9);
}

}  // namespace

TEST(SceneSelection, TheCentroidIsTheMeanOfThePositionsLightsIncluded) {
	Document d = sceneWithTable();   // tops and legs at x 0, 0.5, -0.5 (y 0.5), the ball at x 5, a lamp at (1, 4, 0)
	expectNearFloat3(centroidOf(d, makeItemSet(d, {0, 1, 2})), {0.0, 0.5, 0.0});
	expectNearFloat3(centroidOf(d, makeItemSet(d, {1, 3}, {0})), {(0.5 + 5.0 + 1.0) / 3.0, (0.5 + 0.5 + 4.0) / 3.0, 0.0});
	expectNearFloat3(centroidOf(d, ItemSet{}), {0.0, 0.0, 0.0});
}

TEST(SceneSelection, RotatingMovesPositionsRoundThePivotAndTurnsEveryObjectTheSame) {
	Document d = sceneWithTable();
	d.objects[1].rotation = {10.0, 20.0, 30.0};
	const Object before1 = d.objects[1];
	const ItemSet table = makeItemSet(d, {0, 1, 2});
	const Float3 pivot = centroidOf(d, table);
	rotateItems(d, table, 1, 90.0, pivot);   // a quarter turn about the vertical
	// A leg at +x of the pivot goes to -z (counter-clockwise looking down the axis towards the origin: x -> -z).
	expectNearFloat3(d.objects[1].position, {0.0, 0.5, -0.5});
	expectNearFloat3(d.objects[2].position, {0.0, 0.5, 0.5});
	expectNearFloat3(d.objects[0].position, {0.0, 0.5, 0.0});
	expectNearFloat3(centroidOf(d, table), pivot, 1e-12);
	// The orientation is the quarter turn on top of what it was, whatever angles say so.
	const scene_view::Mat3 turn = scene_view::axisAngle({0, 1, 0}, 90.0), was = orientation(before1), now = orientation(d.objects[1]);
	const scene_view::Mat3 expected = turn * was;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) EXPECT_NEAR(now.m[i][j], expected.m[i][j], 1e-9);
	EXPECT_DOUBLE_EQ(d.objects[3].position.x, 5.0) << "not picked: not moved";
}

TEST(SceneSelection, ARotationKeepsEveryDistanceAndUndoesWithTheOppositeTurn) {
	Document d = sceneWithTable();
	d.objects[3].position = {5.0, 2.0, -1.0};
	const Document original = d;
	const ItemSet all = makeItemSet(d, {0, 1, 2, 3}, {0});
	const Float3 pivot = centroidOf(d, all);
	for (int axis = 0; axis < 3; ++axis) {
		d = original;
		rotateItems(d, all, axis, 37.0, pivot);
		for (int i = 0; i < 4; ++i)
			for (int j = i + 1; j < 4; ++j) EXPECT_NEAR(dist(d.objects[i].position, d.objects[j].position), dist(original.objects[i].position, original.objects[j].position), 1e-9);
		EXPECT_NEAR(dist(d.lights[0].position, d.lights[0].target), dist(original.lights[0].position, original.lights[0].target), 1e-9) << "a spot keeps its length";
		rotateItems(d, all, axis, -37.0, pivot);
		for (int i = 0; i < 4; ++i) {
			expectNearFloat3(d.objects[i].position, original.objects[i].position, 1e-9);
			expectSameOrientation(d.objects[i], original.objects[i]);
		}
		expectNearFloat3(d.lights[0].target, original.lights[0].target, 1e-9);
	}
	d = original;
	rotateItems(d, all, 2, 360.0, pivot);
	expectNearFloat3(d.objects[3].position, original.objects[3].position, 1e-9);
}

TEST(SceneSelection, ARotationOfALightTurnsItsTargetAndLeavesTheSkyAlone) {
	Document d = sceneWithTable();
	Light sky;
	sky.name = "Sky";
	sky.kind = LightKind::Infinite;
	sky.position = {9, 9, 9};
	d.lights.push_back(sky);
	rotateItems(d, makeItemSet(d, {}, {0}), 1, 180.0, {0.0, 0.0, 0.0});   // the lamp at (1, 4, 0) aiming at (1, 0, 0), half a turn about the vertical
	expectNearFloat3(d.lights[0].position, {-1.0, 4.0, 0.0});
	expectNearFloat3(d.lights[0].target, {-1.0, 0.0, 0.0});
	rotateItems(d, makeItemSet(d, {}, {1}), 1, 90.0, {0.0, 0.0, 0.0});
	expectNearFloat3(d.lights[1].target, Light{}.target);   // a sky has no target to turn
}

TEST(SceneSelection, ScalingGrowsPositionsAndObjectsAboutThePivot) {
	Document d = sceneWithTable();
	d.objects[0].shape = ShapeKind::Box;
	d.objects[0].size = {2.0, 0.2, 1.0};
	d.objects[1].shape = ShapeKind::Mesh;
	d.objects[1].meshScale = 0.5;
	const Object ball = d.objects[2];
	const ItemSet table = makeItemSet(d, {0, 1, 2});
	const Float3 pivot = centroidOf(d, table);
	EXPECT_TRUE(scaleItems(d, table, 2.0, pivot));
	EXPECT_NEAR(dist(d.objects[1].position, d.objects[2].position), 2.0 * 1.0, 1e-9) << "the legs were 1 apart";
	expectNearFloat3(centroidOf(d, table), pivot, 1e-12);
	EXPECT_DOUBLE_EQ(d.objects[0].size.x, 4.0);
	EXPECT_DOUBLE_EQ(d.objects[0].size.y, 0.4);
	EXPECT_DOUBLE_EQ(d.objects[1].meshScale, 1.0);
	EXPECT_DOUBLE_EQ(d.objects[2].radius, ball.radius * 2.0);
	EXPECT_DOUBLE_EQ(d.objects[3].radius, 0.5) << "not picked: not scaled";
	EXPECT_TRUE(scaleItems(d, table, 0.5, pivot));
	expectNearFloat3(d.objects[2].position, ball.position, 1e-12);
	EXPECT_DOUBLE_EQ(d.objects[2].radius, ball.radius);
}

TEST(SceneSelection, ScalingAlsoMovesALightAndItsTargetButNotItsBrightness) {
	Document d = sceneWithTable();
	const double intensity = d.lights[0].intensity;
	EXPECT_TRUE(scaleItems(d, makeItemSet(d, {}, {0}), 3.0, {0.0, 0.0, 0.0}));
	expectNearFloat3(d.lights[0].position, {3.0, 12.0, 0.0});
	expectNearFloat3(d.lights[0].target, {3.0, 0.0, 0.0});
	EXPECT_DOUBLE_EQ(d.lights[0].intensity, intensity);
}

TEST(SceneSelection, ScalingRefusesAFactorThatWouldFlipOrVanish) {
	Document d = sceneWithTable();
	const Document original = d;
	for (double factor : {0.0, -1.0, 0.001, 1000.0, std::nan("")}) EXPECT_FALSE(scaleItems(d, makeItemSet(d, {0, 1}), factor, {0.0, 0.0, 0.0})) << factor;
	EXPECT_EQ(toJson(d), toJson(original));
}
