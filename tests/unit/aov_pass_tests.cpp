/**
 * @file aov_pass_tests.cpp
 * @brief The render passes (src/TheRestOfYourLife/aov_pass.h, cpu_render_main_aovs) and the multi-channel EXR helpers they are written with (src/shared/exr_writer.h)
 *
 * Runs from the repository root (it renders the built-in Cornell box, scene A1).
 */

#include <gtest/gtest.h>

#include "../../cpu_renderer/cpu_interface.h"
#include "../../src/shared/accelerator_override.h"
#include "../../src/shared/exr_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {
// A pbrt-backed scene may only be built after the launcher has said which accelerator to use (here: the defaults).
void ensureAcceleratorChosen() { if (!accelerator_override::was_set()) accelerator_override::set({"", ""}); }
std::string tempPath(const char* name) {
	return (std::filesystem::temp_directory_path() / (std::string("aov_pass_tests_") + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" + name)).string();
}
size_t index(const std::vector<std::string>& names, const std::string& n) {
	const auto it = std::find(names.begin(), names.end(), n);
	return it == names.end() ? names.size() : static_cast<size_t>(it - names.begin());
}
}  // namespace

TEST(ExrChannelsTest, ChannelsAreWrittenSortedAndReadBackByName) {
	const std::string path = tempPath("channels.exr");
	const int w = 5, h = 3;
	std::vector<std::string> names = {"normal.Z", "R", "depth.Z", "A", "B", "G"};
	std::vector<std::vector<float>> planes;
	for (size_t c = 0; c < names.size(); ++c) {
		planes.emplace_back(w * h);
		for (int i = 0; i < w * h; ++i) planes[c][i] = static_cast<float>(c * 100 + i);
	}
	std::string error;
	ASSERT_TRUE(write_exr_channels(path, w, h, names, planes, error)) << error;
	int rw = 0, rh = 0;
	std::vector<std::string> rn;
	std::vector<std::vector<float>> rp;
	ASSERT_TRUE(read_exr_channels(path, rw, rh, rn, rp, error)) << error;
	EXPECT_EQ(rw, w);
	EXPECT_EQ(rh, h);
	ASSERT_EQ(rn.size(), names.size());
	EXPECT_TRUE(std::is_sorted(rn.begin(), rn.end())) << "the file lists its channels alphabetically, as EXR requires";
	for (size_t c = 0; c < names.size(); ++c) EXPECT_EQ(rp[index(rn, names[c])], planes[c]) << names[c];
	std::remove(path.c_str());
	EXPECT_FALSE(write_exr_channels(path, w, h, {}, {}, error)) << "no channels";
	EXPECT_FALSE(write_exr_channels(path, w, h, {"R"}, {std::vector<float>(3)}, error)) << "a plane of the wrong size";
	EXPECT_FALSE(read_exr_channels(tempPath("missing.exr"), rw, rh, rn, rp, error));
}

TEST(AovPassTest, TheCornellBoxPassesDescribeItsSurfaces) {
	if (!std::filesystem::exists("pbrt_scenes")) GTEST_SKIP() << "run from the repository root (needs pbrt_scenes/)";
	ensureAcceleratorChosen();
	const std::string path = tempPath("aov_a1.exr");
	const int w = 32, h = 32;
	ASSERT_EQ(cpu_render_main_aovs(w, h, 8, path.c_str(), "A1", 278, 278, -800, 0), 0);
	int rw = 0, rh = 0;
	std::vector<std::string> names;
	std::vector<std::vector<float>> planes;
	std::string error;
	ASSERT_TRUE(read_exr_channels(path, rw, rh, names, planes, error)) << error;
	EXPECT_EQ(rw, w);
	for (const char* c : {"albedo.R", "albedo.G", "albedo.B", "normal.X", "normal.Y", "normal.Z", "depth.Z", "uv.U", "uv.V", "A"}) ASSERT_LT(index(names, c), names.size()) << "missing channel " << c;
	auto at = [&](const char* c, int x, int y) { return planes[index(names, c)][static_cast<size_t>(y) * w + x]; };

	double coverage = 0, minAlbedo = 9, maxAlbedo = -9;
	int surfacePixels = 0;
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			coverage += at("A", x, y);
			if (at("A", x, y) < 0.5f) continue;
			++surfacePixels;
			const double len = std::sqrt(at("normal.X", x, y) * at("normal.X", x, y) + at("normal.Y", x, y) * at("normal.Y", x, y) + at("normal.Z", x, y) * at("normal.Z", x, y));
			EXPECT_NEAR(len, 1.0, 0.02) << "a unit normal at " << x << "," << y;
			EXPECT_GT(at("depth.Z", x, y), 0.0f) << "a positive distance at " << x << "," << y;
			for (const char* c : {"albedo.R", "albedo.G", "albedo.B"}) {
				minAlbedo = std::min<double>(minAlbedo, at(c, x, y));
				maxAlbedo = std::max<double>(maxAlbedo, at(c, x, y));
			}
		}
	EXPECT_GT(coverage / (w * h), 0.9) << "the camera sits inside a closed box: nearly every pixel hits a surface";
	EXPECT_GT(surfacePixels, w * h * 9 / 10);
	EXPECT_GE(minAlbedo, 0.0);
	EXPECT_LE(maxAlbedo, 1.0001);
	// The Cornell box's two side walls are different colours: opposite sides of the frame differ in which of red / green dominates.
	const float leftRG = at("albedo.R", 1, h / 2) - at("albedo.G", 1, h / 2);
	const float rightRG = at("albedo.R", w - 2, h / 2) - at("albedo.G", w - 2, h / 2);
	EXPECT_LT(leftRG * rightRG, 0.0f) << "left wall red-minus-green " << leftRG << ", right wall " << rightRG;
	// The centre of the back wall faces the camera (a normal along +-Z) and is farther away than a side wall's edge.
	EXPECT_GT(std::fabs(at("normal.Z", w / 2, h / 2)), 0.9f);
	EXPECT_GT(at("depth.Z", w / 2, h / 2), at("depth.Z", 1, h / 2) * 0.9f);
	std::remove(path.c_str());
}

TEST(AovPassTest, ThePassesMergeIntoARenderedImageAndKeepItsChannels) {
	if (!std::filesystem::exists("pbrt_scenes")) GTEST_SKIP() << "run from the repository root (needs pbrt_scenes/)";
	ensureAcceleratorChosen();
	const std::string image = tempPath("beauty.exr"), passes = tempPath("passes.exr");
	const int w = 16, h = 16;
	std::vector<float> rgb(w * h * 3, 0.25f);
	std::string error;
	ASSERT_TRUE(write_exr_image(image, rgb.data(), w, h, error)) << error;
	ASSERT_EQ(cpu_render_main_aovs(w, h, 2, passes.c_str(), "A1", 278, 278, -800, 0), 0);
	char message[256] = {0};
	ASSERT_EQ(cpu_merge_exr_passes(image.c_str(), passes.c_str(), message, sizeof message), 0) << message;
	int rw = 0, rh = 0;
	std::vector<std::string> names;
	std::vector<std::vector<float>> planes;
	ASSERT_TRUE(read_exr_channels(image, rw, rh, names, planes, error)) << error;
	for (const char* c : {"R", "G", "B", "A", "albedo.R", "normal.X", "depth.Z", "uv.U"}) EXPECT_LT(index(names, c), names.size()) << c;
	EXPECT_FLOAT_EQ(planes[index(names, "R")][5], 0.25f) << "the image's own R/G/B survive the merge";
	// A size mismatch is refused with a message, and the image is left alone.
	const std::string small = tempPath("small.exr");
	ASSERT_EQ(cpu_render_main_aovs(8, 8, 1, small.c_str(), "A1", 278, 278, -800, 0), 0);
	EXPECT_NE(cpu_merge_exr_passes(image.c_str(), small.c_str(), message, sizeof message), 0);
	EXPECT_NE(std::string(message).find("size"), std::string::npos) << message;
	for (const std::string& p : {image, passes, small}) std::remove(p.c_str());
}
