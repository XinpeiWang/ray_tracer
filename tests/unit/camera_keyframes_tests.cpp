// camera_keyframes_tests.cpp - src/shared/camera_keyframes.h: the flythrough camera path (parsing the file, and the camera at a point along it).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "../../src/shared/camera_keyframes.h"

using namespace camera_keyframes;

namespace {

Path straight(bool ease) {
	Path p;
	p.ease = ease;
	for (double x : {0.0, 10.0, 20.0}) {
		Key k;
		k.pos[0] = x;
		k.target[0] = x;
		k.target[2] = 5;
		p.keys.push_back(k);
	}
	return p;
}

double dist(const double a[3], const double b[3]) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); }

}  // namespace

TEST(CameraKeyframes, ParsesKeysEaseAndComments) {
	const ParseResult r = parse(
		"# a flythrough\n"
		"ease 0   # constant pace\n"
		"\n"
		"key 0 1 8   0 1 0\n"
		"key 4 2 6.5  0 1 0 # closer\n"
		"key -3 2e0 5  0 1 0\n");
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FALSE(r.path.ease);
	ASSERT_EQ(r.path.keys.size(), 3u);
	EXPECT_DOUBLE_EQ(r.path.keys[1].pos[0], 4);
	EXPECT_DOUBLE_EQ(r.path.keys[1].pos[2], 6.5);
	EXPECT_DOUBLE_EQ(r.path.keys[2].pos[1], 2);
	EXPECT_DOUBLE_EQ(r.path.keys[2].target[1], 1);
	EXPECT_TRUE(parse("key 0 0 1 0 0 0\nkey 1 0 1 0 0 0\n").path.ease) << "ease is on unless the file says 0";
}

TEST(CameraKeyframes, RejectsWhatCannotBeAPath) {
	EXPECT_FALSE(parse("").ok);
	EXPECT_FALSE(parse("key 0 0 1 0 0 0\n").ok) << "one key is not a path";
	const ParseResult bad = parse("key 0 0 1 0 0 0\nkey 1 2 3\n");
	EXPECT_FALSE(bad.ok);
	EXPECT_NE(bad.error.find("line 2"), std::string::npos) << bad.error;
	EXPECT_FALSE(parse("key 0 0 1 0 0 0\nkey 1 0 1 0 0 0 7\n").ok) << "seven numbers";
	EXPECT_FALSE(parse("key 0 0 1 0 0 0\nkey a b c 0 0 0\n").ok);
	EXPECT_FALSE(parse("key 0 0 1 0 0 0\nkey 1 0 1 nan 0 0\n").ok);
	EXPECT_FALSE(parse("key 0 0 1 0 0 0\nkey 1 0 1 1 0 1\n").ok) << "a camera looking at its own place";
	EXPECT_FALSE(parse("ease 2\nkey 0 0 1 0 0 0\nkey 1 0 1 0 0 0\n").ok);
	EXPECT_FALSE(parse("fly 1 2 3\nkey 0 0 1 0 0 0\nkey 1 0 1 0 0 0\n").ok);
}

TEST(CameraKeyframes, TheFileWrittenIsTheFileRead) {
	Path p = straight(false);
	p.keys[1].pos[1] = 0.1234567890123456;
	const ParseResult r = parse(toText(p));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_FALSE(r.path.ease);
	ASSERT_EQ(r.path.keys.size(), 3u);
	for (std::size_t i = 0; i < 3; ++i) {
		EXPECT_DOUBLE_EQ(dist(r.path.keys[i].pos, p.keys[i].pos), 0.0);
		EXPECT_DOUBLE_EQ(dist(r.path.keys[i].target, p.keys[i].target), 0.0);
	}
}

TEST(CameraKeyframes, TheFirstFrameIsTheFirstKeyAndTheLastTheLast) {
	for (bool ease : {false, true}) {
		const Path p = straight(ease);
		EXPECT_DOUBLE_EQ(dist(at(p, 0.0).pos, p.keys.front().pos), 0.0);
		EXPECT_DOUBLE_EQ(dist(at(p, 1.0).pos, p.keys.back().pos), 0.0);
		EXPECT_DOUBLE_EQ(dist(at(p, 1.0).target, p.keys.back().target), 0.0);
		EXPECT_DOUBLE_EQ(dist(at(p, -3.0).pos, p.keys.front().pos), 0.0) << "clamped";
		EXPECT_DOUBLE_EQ(dist(at(p, 9.0).pos, p.keys.back().pos), 0.0) << "clamped";
	}
}

TEST(CameraKeyframes, ItGoesThroughEveryKeyWhenTheTimeIsProportionalToTheDistance) {
	Path p = straight(false);
	p.keys[2].pos[0] = 50;   // the stretches are 10 and 40 long: the middle key is a fifth of the way
	p.keys[2].target[0] = 50;
	// Each stretch counts the camera's and its target's distance: 20 and 80, so 20 / (20 + 80) = 0.2 of the time is spent reaching the middle key.
	const Key reached = at(p, 0.2);
	EXPECT_NEAR(reached.pos[0], 10.0, 1e-9);
	EXPECT_NEAR(reached.target[0], 10.0, 1e-9);
}

TEST(CameraKeyframes, AStraightLineStaysStraightAndNeverGoesBackwards) {
	const Path p = straight(false);
	double last = -1.0;
	for (int i = 0; i <= 100; ++i) {
		const Key k = at(p, i / 100.0);
		EXPECT_NEAR(k.pos[1], 0.0, 1e-9);
		EXPECT_NEAR(k.pos[2], 0.0, 1e-9);
		EXPECT_GE(k.pos[0], last - 1e-12);
		last = k.pos[0];
	}
	EXPECT_NEAR(at(p, 0.5).pos[0], 10.0, 1e-9) << "equal stretches: half way is the middle key";
}

TEST(CameraKeyframes, EasingStartsAndEndsSlowerThanTheConstantPace) {
	const double slowStart = at(straight(true), 0.1).pos[0], steady = at(straight(false), 0.1).pos[0];
	EXPECT_LT(slowStart, steady * 0.5);
	const double slowEnd = 20.0 - at(straight(true), 0.9).pos[0], steadyEnd = 20.0 - at(straight(false), 0.9).pos[0];
	EXPECT_LT(slowEnd, steadyEnd * 0.5);
	EXPECT_NEAR(at(straight(true), 0.5).pos[0], 10.0, 1e-9) << "the middle is the middle either way";
}

TEST(CameraKeyframes, TheCameraTurnsSmoothlyThroughACorner) {
	Path p;
	p.ease = false;
	for (const double* v : {new double[6]{0, 0, 0, 0, 0, 5}, new double[6]{10, 0, 0, 10, 0, 5}, new double[6]{10, 0, 10, 10, 0, 15}}) {
		Key k;
		for (int i = 0; i < 3; ++i) {
			k.pos[i] = v[i];
			k.target[i] = v[3 + i];
		}
		p.keys.push_back(k);
		delete[] v;
	}
	Key prev = at(p, 0.0);
	double biggest = 0.0, smallest = 1e9;
	for (int i = 1; i <= 200; ++i) {
		const Key k = at(p, i / 200.0);
		const double step = dist(k.pos, prev.pos);
		biggest = std::max(biggest, step);
		smallest = std::min(smallest, step);
		prev = k;
	}
	EXPECT_LT(biggest, 0.2) << "no jump anywhere: the corner is rounded, not cut";
	EXPECT_GT(smallest, 0.0);
	EXPECT_LT(biggest / smallest, 3.0) << "the pace stays about the same through the corner";
}

TEST(CameraKeyframes, ARepeatedKeyAddsNoTimeAndNoJump) {
	Path p = straight(false);
	p.keys.insert(p.keys.begin() + 1, p.keys[1]);   // 0, 10, 10, 20
	EXPECT_NEAR(at(p, 0.5).pos[0], 10.0, 1e-9);
	Key prev = at(p, 0.0);
	for (int i = 1; i <= 100; ++i) {
		const Key k = at(p, i / 100.0);
		EXPECT_LT(dist(k.pos, prev.pos), 1.0);
		prev = k;
	}
	Path same;
	Key k;
	k.target[2] = 1;
	same.keys = {k, k};
	EXPECT_DOUBLE_EQ(dist(at(same, 0.5).pos, k.pos), 0.0) << "two identical keys: the camera stays";
}

TEST(CameraKeyframes, ProgressOfAFrameRunsFromZeroToOne) {
	EXPECT_DOUBLE_EQ(progressOf(0, 120), 0.0);
	EXPECT_DOUBLE_EQ(progressOf(119, 120), 1.0);
	EXPECT_NEAR(progressOf(60, 121), 0.5, 1e-12);
	EXPECT_DOUBLE_EQ(progressOf(0, 1), 0.0);
	EXPECT_DOUBLE_EQ(progressOf(5, 0), 0.0);
	EXPECT_DOUBLE_EQ(progressOf(500, 10), 1.0);
}

TEST(CameraKeyframes, LoadReadsAFileAndSaysWhichFileIsWrong) {
	const std::string missing = (std::filesystem::temp_directory_path() / "rt_no_such_keyframes.txt").string();
	const ParseResult none = load(missing);
	EXPECT_FALSE(none.ok);
	EXPECT_TRUE(none.unreadable);
	EXPECT_NE(none.error.find("rt_no_such_keyframes.txt"), std::string::npos);

	const std::filesystem::path file = std::filesystem::temp_directory_path() / "rt_keyframes_test.txt";
	{
		std::ofstream out(file);
		out << "key 0 1 8  0 1 0\nkey 4 1 6  0 1 0\n";
	}
	const ParseResult good = load(file.string());
	EXPECT_TRUE(good.ok) << good.error;
	EXPECT_EQ(good.path.keys.size(), 2u);
	{
		std::ofstream out(file);
		out << "key 0 1 8  0 1 0\nkey oops\n";
	}
	const ParseResult bad = load(file.string());
	EXPECT_FALSE(bad.ok);
	EXPECT_FALSE(bad.unreadable) << "the file opened; its contents are wrong";
	EXPECT_NE(bad.error.find("rt_keyframes_test.txt"), std::string::npos);
	EXPECT_NE(bad.error.find("line 2"), std::string::npos);
	std::error_code ec;
	std::filesystem::remove(file, ec);
}
