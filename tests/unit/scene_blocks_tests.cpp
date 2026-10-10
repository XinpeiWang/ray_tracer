// scene_blocks_tests.cpp - src/shared/scene_blocks.h: the blocky blocks, creatures and things, and the greedy merge of voxels into boxes they are drawn with.
#include <gtest/gtest.h>

#include "../../src/shared/scene_blocks.h"
#include "../../src/shared/scene_document.h"

#include <cmath>
#include <set>
#include <string>

using namespace scene_doc;
using namespace scene_doc::blocks_detail;

namespace {
// A box's extent along one axis, as [lo, hi].
struct Span { double lo, hi; };
Span spanOf(const Object& o, int axis) {
	const double c = axis == 0 ? o.position.x : axis == 1 ? o.position.y : o.position.z;
	const double s = axis == 0 ? o.size.x : axis == 1 ? o.size.y : o.size.z;
	return {c - s / 2, c + s / 2};
}
bool overlap(const Object& a, const Object& b) {
	for (int axis = 0; axis < 3; ++axis) {
		const Span x = spanOf(a, axis), y = spanOf(b, axis);
		if (x.hi <= y.lo + 1e-9 || y.hi <= x.lo + 1e-9) return false;
	}
	return true;
}
}  // namespace

TEST(SceneBlocksTest, TheGreedyMergeCoversExactlyTheFilledVoxelsOnce) {
	// A pseudo-random grid of three colours with holes: the boxes' volume equals the number of filled voxels, no two boxes overlap, and each box holds one colour.
	Voxels v(7, 5, 6);
	unsigned state = 12345;
	for (char& c : v.cell) {
		state = state * 1664525u + 1013904223u;
		const unsigned r = (state >> 24) % 5;
		c = r < 3 ? static_cast<char>('a' + r) : 0;
	}
	const std::vector<Paint> palette = {{'a', "a", {1, 0, 0}, Look::Matte, 0}, {'b', "b", {0, 1, 0}, Look::Matte, 0}, {'c', "c", {0, 0, 1}, Look::Matte, 0}};
	const std::vector<Object> boxes = boxesFromVoxels("Test", v, palette, 0.5);
	double volume = 0;
	for (const Object& o : boxes) volume += o.size.x * o.size.y * o.size.z;
	EXPECT_NEAR(volume, static_cast<double>(v.filled()) * 0.125, 1e-9);
	for (size_t i = 0; i < boxes.size(); ++i)
		for (size_t j = i + 1; j < boxes.size(); ++j) EXPECT_FALSE(overlap(boxes[i], boxes[j])) << boxes[i].name << " and " << boxes[j].name;
	EXPECT_LT(boxes.size(), v.filled()) << "the merge should join neighbours";
}

TEST(SceneBlocksTest, AFullCubeIsOneBoxStandingOnTheFloorAroundTheOrigin) {
	Voxels v(4, 4, 4);
	v.fill(0, 0, 0, 4, 4, 4, 'a');
	const std::vector<Object> boxes = boxesFromVoxels("Cube", v, {{'a', "stone", {0.5, 0.5, 0.5}, Look::Matte, 0}}, 0.5);
	ASSERT_EQ(boxes.size(), 1u);
	EXPECT_EQ(boxes[0].name, "Cube stone");
	EXPECT_DOUBLE_EQ(boxes[0].size.x, 2.0);
	EXPECT_DOUBLE_EQ(boxes[0].position.x, 0.0);
	EXPECT_DOUBLE_EQ(boxes[0].position.y, 1.0);   // centre of a 2-high box on the floor
	EXPECT_DOUBLE_EQ(boxes[0].position.z, 0.0);
}

TEST(SceneBlocksTest, EveryBlockyObjectIsAValidSetOfBoxesStandingOnTheFloor) {
	std::set<std::string> labels;
	for (BlockyKind kind : allBlockyKinds()) {
		EXPECT_TRUE(labels.insert(toString(kind)).second) << "duplicate name " << toString(kind);
		const std::vector<Object> parts = makeBlocky(kind);
		ASSERT_GE(parts.size(), 1u) << toString(kind);
		EXPECT_LE(parts.size(), 120u) << toString(kind) << " is too many objects for a prop";
		Document d = makeStarterScene();
		const size_t first = d.objects.size();
		double lowest = 1e9, highest = -1e9;
		std::set<std::string> names;
		for (const Object& o : parts) {
			EXPECT_EQ(o.shape, ShapeKind::Box) << toString(kind);
			EXPECT_GT(o.size.x, 0) << o.name; EXPECT_GT(o.size.y, 0) << o.name; EXPECT_GT(o.size.z, 0) << o.name;
			EXPECT_TRUE(names.insert(o.name).second) << "duplicate part name " << o.name;
			lowest = std::min(lowest, o.position.y - o.size.y / 2);
			highest = std::max(highest, o.position.y + o.size.y / 2);
			d.objects.push_back(o);
		}
		EXPECT_NEAR(lowest, 0.0, 1e-9) << toString(kind) << " does not stand on the floor";
		EXPECT_LE(highest, 4.0 + 1e-9) << toString(kind) << " is taller than a cottage";
		EXPECT_FALSE(hasErrors(validate(d))) << toString(kind);
		Document back;
		std::string error;
		ASSERT_TRUE(fromPbrt(toPbrt(d), back, error)) << toString(kind) << ": " << error;
		ASSERT_EQ(back.objects.size(), d.objects.size());
		for (size_t i = first; i < d.objects.size(); ++i) EXPECT_EQ(back.objects[i].name, d.objects[i].name);
	}
}

TEST(SceneBlocksTest, TheLightGivingOnesGiveLightAndTheOthersDoNot) {
	auto emitters = [](BlockyKind k) {
		int n = 0;
		for (const Object& o : makeBlocky(k)) n += o.emissive ? 1 : 0;
		return n;
	};
	for (BlockyKind k : {BlockyKind::GlowBlock, BlockyKind::LavaBlock, BlockyKind::Torch, BlockyKind::LanternPumpkin, BlockyKind::PurplePortal, BlockyKind::Furnace})
		EXPECT_GE(emitters(k), 1) << toString(k);
	for (BlockyKind k : {BlockyKind::StoneBlock, BlockyKind::GreenMonster, BlockyKind::Pig, BlockyKind::OakTree, BlockyKind::Cottage, BlockyKind::GlassBlock})
		EXPECT_EQ(emitters(k), 0) << toString(k);
}
