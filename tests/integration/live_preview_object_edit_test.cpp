// live_preview_object_edit_test.cpp
//
// Live Preview's "Move objects" on OptiX (gpu/optix/optix_live_edit.h): click a surface to find which object it belongs to, give the object an offset and the next frame draws
// the scene with it moved, reset puts everything back, and "Save arrangement" writes the moved scene to a pbrt file. The scene is a .pbrt file with two spheres, each in its
// own AttributeBegin/End block, saved into a throwaway per-user scenes folder (the way live_preview_failure_message_test.cpp does it).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../cpu_renderer/cpu_interface.h"
extern "C" {
	#include "optix_interface.h"
}

namespace {

namespace fs = std::filesystem;

constexpr int kSize = 64;

const char* kScene =
	"LookAt 0 1 6  0 1 0  0 1 0\n"
	"Camera \"perspective\" \"float fov\" [ 40 ]\n"
	"Film \"rgb\" \"integer xresolution\" [ 64 ] \"integer yresolution\" [ 64 ]\n"
	"Sampler \"halton\" \"integer pixelsamples\" [ 4 ]\n"
	"WorldBegin\n"
	"LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n"
	"AttributeBegin\n"
	"  Material \"diffuse\" \"rgb reflectance\" [ 0.8 0.1 0.1 ]\n"
	"  Translate -1.5 1 0\n"
	"  Shape \"sphere\" \"float radius\" [ 0.8 ]\n"
	"AttributeEnd\n"
	"AttributeBegin\n"
	"  Material \"diffuse\" \"rgb reflectance\" [ 0.1 0.1 0.8 ]\n"
	"  Translate 1.5 1 0\n"
	"  Shape \"sphere\" \"float radius\" [ 0.8 ]\n"
	"AttributeEnd\n";

struct LiveFrame {
	std::vector<float> rgb = std::vector<float>(kSize * kSize * 3);
	std::vector<float> pos = std::vector<float>(kSize * kSize * 4);
	// The first-hit position of the pixel at (column, row), or false for a miss.
	bool hit(int col, int row, double p[3]) const {
		const float* q = &pos[(static_cast<std::size_t>(row) * kSize + col) * 4];
		if (q[3] < 0.5f) return false;
		for (int i = 0; i < 3; ++i) p[i] = q[i];
		return true;
	}
};

double meanAbsDiff(const LiveFrame& a, const LiveFrame& b) {
	double sum = 0.0;
	for (std::size_t i = 0; i < a.rgb.size(); ++i) sum += std::fabs(a.rgb[i] - b.rgb[i]);
	return sum / static_cast<double>(a.rgb.size());
}

class LivePreviewObjectEditTest : public ::testing::Test {
protected:
	void SetUp() override {
		if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
		const char* tmp = std::getenv("TEMP");
		if (!tmp) tmp = std::getenv("TMPDIR");
		root_ = fs::path(tmp ? tmp : ".") / "live_preview_object_edit_test";
		fs::remove_all(root_);
		fs::create_directories(root_ / "user_scenes");
#ifdef _WIN32
		_putenv_s("RAY_TRACER_USER_ASSETS", root_.string().c_str());
#else
		setenv("RAY_TRACER_USER_ASSETS", root_.string().c_str(), 1);
#endif
		const fs::path file = root_ / "user_scenes" / "two-balls.pbrt";
		std::ofstream(file) << kScene;
		cpu_refresh_user_scenes();
		const char* found = cpu_scene_id_for_file(file.string().c_str());
		id_ = found ? found : "";
		ASSERT_FALSE(id_.empty()) << "the scene list did not pick up the new scene";
	}
	void TearDown() override {
		if (!id_.empty() && optix_is_available()) rt_realtime_reset_objects(id_.c_str());
#ifdef _WIN32
		_putenv_s("RAY_TRACER_USER_ASSETS", "");
#else
		unsetenv("RAY_TRACER_USER_ASSETS");
#endif
		std::error_code ec;
		fs::remove_all(root_, ec);
	}

	bool draw(LiveFrame& f) {
		return rt_realtime_render_frame(id_.c_str(), kSize, kSize, /*spp=*/64, /*max_depth=*/4, 0.0, 1.0, 6.0,
			/*has_custom_lookat=*/true, 0.0, 1.0, 0.0, /*denoise=*/false, /*denoise_blend=*/0.0,
			f.pos.data(), /*out_camera_basis=*/nullptr, f.rgb.data(), /*enable_svgf=*/false);
	}

	// The object under a point, or -1.
	int pickAt(const double p[3], double* lo = nullptr, double* hi = nullptr, double* off = nullptr, std::string* label = nullptr) {
		char text[64] = {};
		const int object = rt_realtime_pick_object(id_.c_str(), p[0], p[1], p[2], lo, hi, off, text, sizeof text);
		if (label) *label = text;
		return object;
	}

	// A hit on the left sphere and one on the right sphere, from a drawn frame (a row above the middle crosses both).
	static void spheresIn(const LiveFrame& f, double left[3], double right[3]) {
		bool haveL = false, haveR = false;
		for (int col = 0; col < kSize; ++col) {
			double p[3];
			if (!f.hit(col, kSize / 2 - 4, p)) continue;
			if (p[0] < -1.0 && !haveL) { std::copy(p, p + 3, left); haveL = true; }
			if (p[0] > 1.0) { std::copy(p, p + 3, right); haveR = true; }
		}
		ASSERT_TRUE(haveL && haveR) << "both spheres should be visible";
	}

	fs::path root_;
	std::string id_;
};

}  // namespace

TEST_F(LivePreviewObjectEditTest, NothingIsPickableBeforeAFrameIsDrawn) {
	const double p[3] = {-1.5, 1.0, 0.8};
	EXPECT_EQ(pickAt(p), -1);
	EXPECT_FALSE(rt_realtime_set_object_offset(id_.c_str(), 0, 1, 0, 0)) << "no scene has been built yet, so no object is known";
}

TEST_F(LivePreviewObjectEditTest, AClickFindsTheObjectItHit) {
	LiveFrame f;
	ASSERT_TRUE(draw(f)) << rt_realtime_get_last_error();
	double left[3], right[3];
	spheresIn(f, left, right);
	ASSERT_FALSE(HasFatalFailure());

	double lo[3], hi[3], off[3];
	std::string label;
	EXPECT_EQ(pickAt(left, lo, hi, off, &label), 0);
	EXPECT_LT(lo[0], -1.9);
	EXPECT_GT(hi[0], -1.1);
	EXPECT_LT(hi[0], -0.5) << "the box is the left sphere's, not both";
	EXPECT_EQ(off[0], 0.0);
	EXPECT_FALSE(label.empty());
	EXPECT_EQ(pickAt(right), 1);

	const double empty[3] = {0.0, 5.0, 0.0};
	EXPECT_EQ(pickAt(empty), -1);
	EXPECT_EQ(rt_realtime_pick_object("ZZ9999", 0, 0, 0, nullptr, nullptr, nullptr, nullptr, 0), -1);
}

TEST_F(LivePreviewObjectEditTest, MovingAnObjectChangesTheNextFrameAndResetRestoresIt) {
	LiveFrame before;
	ASSERT_TRUE(draw(before)) << rt_realtime_get_last_error();
	double left[3], right[3];
	spheresIn(before, left, right);
	ASSERT_FALSE(HasFatalFailure());
	ASSERT_EQ(pickAt(left), 0);

	EXPECT_FALSE(rt_realtime_set_object_offset(id_.c_str(), 7, 1, 0, 0)) << "there are only two objects";
	ASSERT_TRUE(rt_realtime_set_object_offset(id_.c_str(), 0, 0.0, 2.0, 0.0));

	// Until the next frame the picture on screen is the old one, so a click still lands on the left sphere where it was drawn, and the box it reports is where the sphere will be.
	double lo[3], hi[3], off[3];
	EXPECT_EQ(pickAt(left, lo, hi, off), 0);
	EXPECT_NEAR(off[1], 2.0, 1e-9);
	EXPECT_GT(lo[1], 2.0) << "the box already follows the move: the sphere spans y 0.2..1.8, moved up by 2";

	LiveFrame moved;
	ASSERT_TRUE(draw(moved)) << rt_realtime_get_last_error();
	EXPECT_GT(meanAbsDiff(before, moved), 0.01) << "the left sphere is somewhere else now";
	EXPECT_EQ(pickAt(left), -1) << "nothing is where the left sphere was";
	const double up[3] = {left[0], left[1] + 2.0, left[2]};
	EXPECT_EQ(pickAt(up), 0);
	EXPECT_EQ(pickAt(right), 1) << "the other object did not move";

	rt_realtime_reset_objects(id_.c_str());
	LiveFrame back;
	ASSERT_TRUE(draw(back)) << rt_realtime_get_last_error();
	EXPECT_LT(meanAbsDiff(before, back), 0.5 * meanAbsDiff(before, moved)) << "reset draws the file's own arrangement again";
	EXPECT_EQ(pickAt(left), 0);
}

TEST_F(LivePreviewObjectEditTest, SaveArrangementWritesTheMovedScene) {
	LiveFrame f;
	ASSERT_TRUE(draw(f)) << rt_realtime_get_last_error();
	const fs::path out = root_ / "saved.pbrt";
	char message[256] = {};
	EXPECT_FALSE(rt_realtime_export_arrangement(id_.c_str(), out.string().c_str(), message, sizeof message)) << "nothing has been moved yet";
	EXPECT_NE(std::string(message), "");

	ASSERT_TRUE(rt_realtime_set_object_offset(id_.c_str(), 1, 0.5, 0.0, -1.0));
	ASSERT_TRUE(rt_realtime_export_arrangement(id_.c_str(), out.string().c_str(), message, sizeof message)) << message;
	EXPECT_NE(std::string(message).find("1 object"), std::string::npos) << message;

	std::ifstream in(out);
	std::stringstream text;
	text << in.rdbuf();
	const std::string saved = text.str();
	EXPECT_NE(saved.find("Translate"), std::string::npos);
	EXPECT_NE(saved.find("0.5"), std::string::npos);
	EXPECT_NE(saved.find("-1"), std::string::npos);
}
