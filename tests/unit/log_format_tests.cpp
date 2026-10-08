/**
 * @file log_format_tests.cpp
 * @brief The GUI log file's Qt-free parts (src/shared/log_format.h) and the Scene Builder's change description (src/shared/scene_doc_diff.h)
 */

#include <gtest/gtest.h>

#include "../../src/shared/log_format.h"
#include "../../src/shared/scene_doc_diff.h"
#include "../../src/shared/window_placement.h"

#include <filesystem>
#include <fstream>

namespace {
std::string tempDir() {
	const char* t = std::getenv("TEMP");
	std::filesystem::path p = std::filesystem::path(t ? t : ".") / ("log_format_tests_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
	std::filesystem::create_directories(p);
	return p.string();
}
void writeFile(const std::string& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }
}  // namespace

TEST(LogFormatTest, AnEntryIsOneLineWithTimeLevelAndCategory) {
	EXPECT_EQ(log_format::formatLine("2026-10-07 21:30:01.123", log_format::Level::Info, "builder", "added Sphere"),
	          "2026-10-07 21:30:01.123 [INFO ] builder: added Sphere");
	EXPECT_EQ(log_format::formatLine("t", log_format::Level::Error, "ui", "x"), "t [ERROR] ui: x");
	EXPECT_EQ(log_format::formatLine("t", log_format::Level::Warn, "c", "m"), "t [WARN ] c: m");
}

TEST(LogFormatTest, LineBreaksBecomeSeparatorsAndLongMessagesAreCutWithANote) {
	EXPECT_EQ(log_format::oneLine("a\nb\r\nc\n\n"), "a | b | c");
	EXPECT_EQ(log_format::oneLine("\n\nstart"), "start");
	EXPECT_EQ(log_format::oneLine("trailing   "), "trailing");
	const std::string cut = log_format::oneLine(std::string(100, 'x'), 40);
	EXPECT_EQ(cut, std::string(40, 'x') + "... (+60 characters)");
	EXPECT_EQ(log_format::formatLine("t", log_format::Level::Info, "c", "one\ntwo").find('\n'), std::string::npos);
}

TEST(LogFormatTest, ALargeFileRotatesAndOldOnesAreKeptUpToTheLimit) {
	const std::string dir = tempDir();
	const std::string file = dir + "/rot.log";
	for (int round = 1; round <= 5; ++round) {
		writeFile(file, std::string(100, static_cast<char>('0' + round)));
		EXPECT_TRUE(log_format::rotateIfLarge(file, 50, 3)) << round;
		EXPECT_FALSE(std::filesystem::exists(file)) << "the big file moved aside";
	}
	// newest old file is .1; only three are kept
	auto first = [](const std::string& p) { std::ifstream in(p); char c = 0; in >> c; return c; };
	EXPECT_EQ(first(file + ".1"), '5');
	EXPECT_EQ(first(file + ".2"), '4');
	EXPECT_EQ(first(file + ".3"), '3');
	EXPECT_FALSE(std::filesystem::exists(file + ".4"));
	writeFile(file, "small");
	EXPECT_FALSE(log_format::rotateIfLarge(file, 50, 3));
	EXPECT_FALSE(log_format::rotateIfLarge(dir + "/missing.log", 50, 3));
	std::filesystem::remove_all(dir);
}

TEST(LogFormatTest, TailGivesTheLastLinesOfAFile) {
	const std::string dir = tempDir();
	const std::string file = dir + "/tail.log";
	std::string text;
	for (int i = 1; i <= 500; ++i) text += "line " + std::to_string(i) + (i % 2 ? "\r\n" : "\n");
	writeFile(file, text);
	const std::vector<std::string> last = log_format::tailLines(file, 3);
	ASSERT_EQ(last.size(), 3u);
	EXPECT_EQ(last[0], "line 498");
	EXPECT_EQ(last[2], "line 500");
	EXPECT_EQ(log_format::tailLines(file, 1000).size(), 500u);
	EXPECT_TRUE(log_format::tailLines(dir + "/nope.log", 5).empty());
	std::filesystem::remove_all(dir);
}

TEST(SceneDocDiffTest, NamesTheItemThatWasDeletedOrAdded) {
	using namespace scene_doc;
	Document a = makeStarterScene();
	ASSERT_GE(a.objects.size(), 3u);
	for (std::size_t i = 0; i < a.objects.size(); ++i) a.objects[i].name = "item" + std::to_string(i);
	// Deleting one from the middle names it, not the last one in the list.
	Document b = a;
	b.objects.erase(b.objects.begin() + 1);
	const std::string removed = describeChange(a, b);
	EXPECT_NE(removed.find("-1 object (item1)"), std::string::npos) << removed;
	EXPECT_EQ(removed.find("item" + std::to_string(a.objects.size() - 1)), std::string::npos) << removed;
	// The first and the last name themselves too, and an addition at the end is named.
	b = a;
	b.objects.erase(b.objects.begin());
	EXPECT_NE(describeChange(a, b).find("-1 object (item0)"), std::string::npos);
	b = a;
	b.objects.pop_back();
	EXPECT_NE(describeChange(a, b).find("-1 object (item" + std::to_string(a.objects.size() - 1) + ")"), std::string::npos);
	b = a;
	Object extra = b.objects[0];
	extra.name = "extra";
	b.objects.push_back(extra);
	EXPECT_NE(describeChange(a, b).find("+1 object (extra)"), std::string::npos);
}

TEST(SceneDocDiffTest, DescribesWhatAUserChanged) {
	using namespace scene_doc;
	Document a = makeStarterScene();
	Document b = a;
	EXPECT_EQ(describeChange(a, b), "");
	b.objects[1].position.x += 2;
	const std::string moved = describeChange(a, b);
	EXPECT_NE(moved.find("object '" + a.objects[1].name + "': position"), std::string::npos) << moved;
	EXPECT_NE(moved.find("->"), std::string::npos) << moved;

	b = a;
	b.objects[1].radius = 3;
	b.objects[1].material.color = {0.1, 0.2, 0.3};
	const std::string two = describeChange(a, b);
	EXPECT_NE(two.find("radius,color"), std::string::npos) << two;

	b = a;
	b.title = "Renamed";
	b.camera.fov = 55;
	b.render.samples = 128;
	const std::string more = describeChange(a, b);
	EXPECT_NE(more.find("title '" + a.title + "' -> 'Renamed'"), std::string::npos) << more;
	EXPECT_NE(more.find("camera: fov 40 -> 55"), std::string::npos) << more;
	EXPECT_NE(more.find("render: samples"), std::string::npos) << more;

	b = a;
	b.objects.push_back(makeObject(ShapeKind::Pyramid, "Roof"));
	EXPECT_EQ(describeChange(a, b), "+1 object (Roof)");
	EXPECT_EQ(describeChange(b, a), "-1 object (Roof)");
	b = a;
	b.lights.pop_back();
	EXPECT_EQ(describeChange(a, b).substr(0, 9), "-1 light ");
}

TEST(SceneDocDiffTest, ManyChangesAreCutAfterAFew) {
	using namespace scene_doc;
	Document a = makeStarterScene();
	Document b = a;
	for (Object& o : b.objects) o.position.y += 1;
	b.title = "t";
	b.camera.fov = 90;
	b.render.width = 99;
	const std::string text = describeChange(a, b);
	EXPECT_NE(text.find("... (+"), std::string::npos) << text;
}

TEST(WindowPlacementTest, ASavedWindowIsUsableOnlyIfItsTitleBarCanBeReached) {
	using window_placement::Rect;
	const std::vector<Rect> one = {{0, 0, 1920, 1080}};
	EXPECT_TRUE(window_placement::reachable({100, 100, 1000, 800}, one));
	EXPECT_TRUE(window_placement::reachable({-880, 50, 1000, 800}, one)) << "mostly off to the left, but 120 pixels of title bar show";
	EXPECT_FALSE(window_placement::reachable({-990, 50, 1000, 800}, one)) << "only 10 pixels of title bar show";
	EXPECT_FALSE(window_placement::reachable({3000, 100, 1000, 800}, one)) << "on a monitor that is gone";
	EXPECT_FALSE(window_placement::reachable({100, 2000, 1000, 800}, one)) << "below the screen";
	EXPECT_FALSE(window_placement::reachable({100, -780, 1000, 800}, one)) << "title bar above the screen, only the bottom shows";
	EXPECT_FALSE(window_placement::reachable({0, 0, 0, 0}, one));
	EXPECT_FALSE(window_placement::reachable({100, 100, 1000, 800}, {}));
	// A second screen to the right makes the same window reachable.
	const std::vector<Rect> two = {{0, 0, 1920, 1080}, {1920, 0, 2560, 1440}};
	EXPECT_TRUE(window_placement::reachable({3000, 100, 1000, 800}, two));
	// A small window narrower than the minimum counts as reachable when all of it shows.
	EXPECT_TRUE(window_placement::reachable({10, 10, 80, 60}, one));
}

TEST(WindowPlacementTest, ASizeIsNeverLargerThanTheScreen) {
	using window_placement::Rect;
	const Rect clamped = window_placement::clampSize({0, 0, 3000, 2000}, {0, 0, 1920, 1080});
	EXPECT_EQ(clamped.w, 1920);
	EXPECT_EQ(clamped.h, 1080);
	EXPECT_EQ(window_placement::clampSize({0, 0, 800, 600}, {0, 0, 1920, 1080}).w, 800);
}
