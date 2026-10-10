// launcher_args_keyframes_tests.cpp -- CLI flag parsing for --camera-keyframes (launcher/launcher_args.h), in the same style as launcher_args_time_limit_tests.cpp. The path
// itself is src/shared/camera_keyframes.h (camera_keyframes_tests.cpp).

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <string>
#include <vector>

namespace {

bool parseArgs(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> storage;
	storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) storage.push_back(t);
	std::vector<char*> ptrs;
	for (auto& s : storage) ptrs.push_back(const_cast<char*>(s.c_str()));
	return parse_launch_args(static_cast<int>(ptrs.size()), ptrs.data(), out_args);
}

}  // namespace

TEST(LauncherArgsKeyframes, NoFileByDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parseArgs({}, args));
	EXPECT_TRUE(args.camera_keyframes_file.empty());
}

TEST(LauncherArgsKeyframes, TakesTheFileAndCountsAsAChosenPath) {
	LaunchArgs args;
	ASSERT_TRUE(parseArgs({"--video", "--camera-keyframes", "C:/work/fly path.txt", "--frames", "60"}, args));
	EXPECT_EQ(args.camera_keyframes_file, "C:/work/fly path.txt");
	EXPECT_TRUE(args.camera_path_explicit) << "the scene's own recommended path must not replace it";
	EXPECT_TRUE(args.video_mode);
	EXPECT_EQ(args.video_frames, 60);
}
