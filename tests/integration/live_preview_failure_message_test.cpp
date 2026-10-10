// live_preview_failure_message_test.cpp
//
// When a Live Preview frame cannot be rendered, the renderer library says why (rt_realtime_get_last_error()), and the GUI shows that instead of a generic guess
// ("scene may not be GPU-supported"), which was wrong for an empty scene (the problem is that it has nothing in it) and for a scene whose mesh file is missing
// (the user needs to download it). The scenes here are .pbrt files saved into a throwaway per-user scenes folder, which the scene list picks up while running.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../cpu_renderer/cpu_interface.h"
extern "C" {
	#include "optix_interface.h"
}

namespace {

namespace fs = std::filesystem;

const char* kHeader =
	"LookAt 0 1 6  0 1 0  0 1 0\n"
	"Camera \"perspective\" \"float fov\" [ 40 ]\n"
	"Film \"rgb\" \"integer xresolution\" [ 32 ] \"integer yresolution\" [ 32 ]\n"
	"Sampler \"halton\" \"integer pixelsamples\" [ 4 ]\n"
	"WorldBegin\n"
	"LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n";

class LivePreviewFailureMessageTest : public ::testing::Test {
protected:
	void SetUp() override {
		if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
		const char* tmp = std::getenv("TEMP");
		if (!tmp) tmp = std::getenv("TMPDIR");
		root_ = fs::path(tmp ? tmp : ".") / "live_preview_failure_message_test";
		fs::remove_all(root_);
		fs::create_directories(root_ / "user_scenes");
#ifdef _WIN32
		_putenv_s("RAY_TRACER_USER_ASSETS", root_.string().c_str());
#else
		setenv("RAY_TRACER_USER_ASSETS", root_.string().c_str(), 1);
#endif
	}
	void TearDown() override {
#ifdef _WIN32
		_putenv_s("RAY_TRACER_USER_ASSETS", "");
#else
		unsetenv("RAY_TRACER_USER_ASSETS");
#endif
		std::error_code ec;
		fs::remove_all(root_, ec);
	}

	// Saves a scene into the per-user folder and returns the id the scene list gives it.
	std::string addScene(const std::string& name, const std::string& body) {
		const fs::path file = root_ / "user_scenes" / (name + ".pbrt");
		std::ofstream(file) << kHeader << body;
		cpu_refresh_user_scenes();
		const char* id = cpu_scene_id_for_file(file.string().c_str());
		return id ? std::string(id) : std::string();
	}

	// One Live Preview frame the way the GUI asks for it (the Windows defaults: restir on, svgf off).
	static bool renderFrame(const std::string& id, std::vector<float>& rgb) {
		return rt_realtime_render_frame(id.c_str(), 32, 32, /*samples_per_pixel=*/1, /*max_depth=*/4, 0.0, 1.0, 6.0,
			/*has_custom_lookat=*/true, 0.0, 1.0, 0.0, /*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr, rgb.data(), /*enable_svgf=*/false);
	}

	fs::path root_;
};

}  // namespace

TEST_F(LivePreviewFailureMessageTest, AnEmptySceneSaysItHasNoGeometry) {
	const std::string id = addScene("empty-world", "");
	ASSERT_FALSE(id.empty()) << "the scene list did not pick up the new scene";
	std::vector<float> rgb(32 * 32 * 3);
	EXPECT_FALSE(renderFrame(id, rgb));
	const std::string why = rt_realtime_get_last_error();
	EXPECT_NE(why.find("no geometry"), std::string::npos) << "got: " << why;
	// Asking again gives the same answer (the failed build did not leave the renderer half-built), and nothing from the first call is appended.
	EXPECT_FALSE(renderFrame(id, rgb));
	EXPECT_EQ(std::string(rt_realtime_get_last_error()), why);
}

TEST_F(LivePreviewFailureMessageTest, AMissingMeshFileNamesTheFile) {
	const std::string id = addScene("broken-mesh", "Shape \"plymesh\" \"string filename\" [ \"there-is-no-such-mesh.ply\" ]\n");
	ASSERT_FALSE(id.empty());
	std::vector<float> rgb(32 * 32 * 3);
	EXPECT_FALSE(renderFrame(id, rgb));
	const std::string why = rt_realtime_get_last_error();
	EXPECT_NE(why.find("there-is-no-such-mesh.ply"), std::string::npos) << "got: " << why;
	EXPECT_EQ(why.find("[OptiX]"), std::string::npos) << "the library's own tag is not part of the message";
}

TEST_F(LivePreviewFailureMessageTest, ASceneThatRendersLeavesNoMessage) {
	const std::string id = addScene("one-ball", "Shape \"sphere\" \"float radius\" [ 1 ]\n");
	ASSERT_FALSE(id.empty());
	std::vector<float> rgb(32 * 32 * 3);
	EXPECT_TRUE(renderFrame(id, rgb)) << rt_realtime_get_last_error();
	EXPECT_STREQ(rt_realtime_get_last_error(), "");
}

TEST_F(LivePreviewFailureMessageTest, AnUnknownSceneStillGetsAnExplanation) {
	std::vector<float> rgb(32 * 32 * 3);
	EXPECT_FALSE(renderFrame("ZZ9999", rgb));
	EXPECT_NE(std::string(rt_realtime_get_last_error()), "") << "even the fallback reason is better than nothing";
}
