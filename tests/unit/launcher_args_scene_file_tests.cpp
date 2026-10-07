// launcher_args_scene_file_tests.cpp -- CLI parsing of a .pbrt path in the scene slot and of --height (launcher/launcher_args.h), the two things the Scene
// Builder needs to render a scene it saved: its own file, at its own aspect ratio. Same direct parse_launch_args() style as launcher_args_time_limit_tests.cpp.

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <string>
#include <vector>

namespace {

bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage{"ray_tracer.exe"};
	for (auto& t : tokens) argv_storage.push_back(t);
	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));
	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

}  // namespace

TEST(LauncherArgsSceneFile, APbrtPathInTheSceneSlotIsKeptAsAFile) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"400", "16", "4", "C:\scenes\my room.pbrt"}, args));
	EXPECT_EQ(args.scene_file, "C:\scenes\my room.pbrt");
	EXPECT_EQ(args.scene_id, "A1") << "the id stays the default until main registers the file";
	EXPECT_EQ(args.image_width, 400);
}

TEST(LauncherArgsSceneFile, TheExtensionIsCaseInsensitive) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"400", "16", "4", "scene.PBRT"}, args));
	EXPECT_EQ(args.scene_file, "scene.PBRT");
}

TEST(LauncherArgsSceneFile, AnIdStillMeansAnId) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"400", "16", "4", "B3"}, args));
	EXPECT_TRUE(args.scene_file.empty());
	EXPECT_EQ(args.scene_id, "B3");
}

TEST(LauncherArgsSceneFile, ANonsenseSceneArgumentIsStillRejected) {
	LaunchArgs args;
	EXPECT_FALSE(parse({"400", "16", "4", "scene.txt"}, args));
}

TEST(LauncherArgsHeight, DefaultsToSquare) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"320"}, args));
	EXPECT_EQ(args.image_width, 320);
	EXPECT_EQ(args.image_height, 320);
}

TEST(LauncherArgsHeight, HeightFlagGivesANonSquareImage) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--height", "180", "320", "8"}, args));
	EXPECT_EQ(args.image_width, 320);
	EXPECT_EQ(args.image_height, 180);
	EXPECT_EQ(args.samples_per_pixel, 8) << "the flag's value must not be read as a positional argument";
}

TEST(LauncherArgsHeight, ABadHeightFallsBackToSquare) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--height", "-5", "320"}, args));
	EXPECT_EQ(args.image_height, 320);
}

TEST(LauncherArgsSceneName, ASlugIsAcceptedAsTheScene) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"400", "16", "4", "cornell-box"}, args));
	EXPECT_EQ(args.scene_id, "cornell-box");
	EXPECT_TRUE(args.scene_file.empty());
}

TEST(LauncherArgsSceneName, ASlugWithDigitsAndHyphensIsAccepted) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"400", "16", "4", "dragon-10"}, args));
	EXPECT_EQ(args.scene_id, "dragon-10");
}

TEST(LauncherArgsSceneName, MalformedNamesAreStillRejected) {
	LaunchArgs args;
	EXPECT_FALSE(parse({"400", "16", "4", "Cornell Box"}, args));
	EXPECT_FALSE(parse({"400", "16", "4", "bad_name"}, args));
	EXPECT_FALSE(parse({"400", "16", "4", "-x"}, args));
}
