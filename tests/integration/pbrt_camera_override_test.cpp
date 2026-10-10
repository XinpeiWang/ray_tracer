// pbrt_camera_override_test.cpp
//
// A whole camera per render (RenderOptions::has_lookat_override / lookat_override / vfov_override, src/shared/render_options.h): what a --video --camera-keyframes
// flythrough frame needs, so each frame looks where its keyframe says with its keyframe's field of view, instead of only moving the camera's position while it kept
// looking at the scene's own target.
//
// The OptiX side is checked on the host (build_scene() is host-side C++, as in pbrt_dof_override_test.cpp): the camera vectors it fills in. The CPU path tracer is checked
// by rendering small pictures and comparing them.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}
#include "../../cpu_renderer/cpu_interface.h"
#include "scene_builder.h"
#include "scene_registry.h"
#include "../../src/shared/render_options.h"

namespace {

struct V { double x, y, z; };
V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double len(V a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
V unit(V a) { const double l = len(a); return {a.x / l, a.y / l, a.z / l}; }
double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// camera_params: origin(3), lower_left(3), horizontal(3), vertical(3). The direction through the middle of the picture, and the vertical extent at depth 1.
struct Pinhole {
	V origin, centre;
	double verticalExtent;
};
Pinhole readPinhole(const float* p) {
	const V origin{p[0], p[1], p[2]}, lowerLeft{p[3], p[4], p[5]}, h{p[6], p[7], p[8]}, v{p[9], p[10], p[11]};
	const V middle{lowerLeft.x + 0.5 * (h.x + v.x), lowerLeft.y + 0.5 * (h.y + v.y), lowerLeft.z + 0.5 * (h.z + v.z)};
	return {origin, unit(sub(middle, origin)), len(v)};
}

const SceneDescriptor* findScene() { return find_scene_by_file_stem("checkered-spheres"); }

bool build(const SceneDescriptor& s, bool lookatOverride, V lookat, double vfov, float* params) {
	SceneData scene;
	GpuCameraParams extra{};
	return build_scene(s.id.c_str(), 64, 64, scene, params, 0.0, 2.0, 8.0, &extra, /*force_camera_override=*/true, lookatOverride, lookat.x, lookat.y, lookat.z,
	                   /*has_dof_override=*/false, 0.0, 0.0, vfov);
}

// A text PPM's pixels as 0..255 numbers.
std::vector<int> readPpm(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	std::string magic;
	int w = 0, h = 0, maxv = 0;
	in >> magic >> w >> h >> maxv;
	std::vector<int> pixels;
	if (magic != "P3") return pixels;
	int v;
	while (in >> v) pixels.push_back(v);
	return pixels;
}

double meanAbsDiff(const std::vector<int>& a, const std::vector<int>& b) {
	if (a.empty() || a.size() != b.size()) return -1.0;
	double sum = 0.0;
	for (std::size_t i = 0; i < a.size(); ++i) sum += std::abs(a[i] - b[i]);
	return sum / static_cast<double>(a.size());
}

}  // namespace

TEST(PbrtCameraOverrideTest, OptixLooksWhereItIsToldAndWithTheFieldOfViewItIsGiven) {
	const SceneDescriptor* s = findScene();
	if (!s) GTEST_SKIP() << "checkered-spheres.pbrt was not discovered - is pbrt_scenes/ present?";
	float own[12], moved[12], zoomed[12];
	ASSERT_TRUE(build(*s, false, {0, 0, 0}, 0.0, own));
	const Pinhole base = readPinhole(own);

	// Looking at a point to the side: the middle of the picture points at it.
	const V target{base.origin.x + 5.0, base.origin.y, base.origin.z - 5.0};
	ASSERT_TRUE(build(*s, true, target, 0.0, moved));
	const Pinhole turned = readPinhole(moved);
	const V want = unit(sub(target, turned.origin));
	EXPECT_NEAR(dot(turned.centre, want), 1.0, 1e-4) << "the middle of the picture looks at the override's target";
	EXPECT_LT(dot(turned.centre, base.centre), 0.99) << "and that is not where the scene's own camera looked";
	EXPECT_NEAR(turned.verticalExtent, base.verticalExtent, 1e-4) << "the field of view did not change";

	// A wider field of view: the picture spans more at the same depth. 90 degrees: 2 * tan(45) = 2.
	ASSERT_TRUE(build(*s, false, {0, 0, 0}, 90.0, zoomed));
	const Pinhole wide = readPinhole(zoomed);
	EXPECT_NEAR(wide.verticalExtent, 2.0, 1e-4);
	EXPECT_NEAR(dot(wide.centre, base.centre), 1.0, 1e-5) << "the direction did not change";
}

TEST(PbrtCameraOverrideTest, CpuRendersWhereItIsToldAndWithTheFieldOfViewItIsGiven) {
	const SceneDescriptor* s = findScene();
	if (!s) GTEST_SKIP() << "checkered-spheres.pbrt was not discovered - is pbrt_scenes/ present?";
	const std::filesystem::path dir = std::filesystem::temp_directory_path() / "rt_camera_override_test";
	std::filesystem::create_directories(dir);
	const auto render = [&](const char* name, const RenderOptions& options) {
		const std::string path = (dir / name).string();
		EXPECT_EQ(cpu_render_main(48, 36, 8, 4, path.c_str(), s->id.c_str(), 0.0, 2.0, 8.0, /*force_camera_override=*/1, options), 0) << name;
		return readPpm(path);
	};
	RenderOptions plain;
	const std::vector<int> a = render("a.ppm", plain), again = render("again.ppm", plain);
	ASSERT_FALSE(a.empty());
	const double noise = meanAbsDiff(a, again);   // the same camera twice: only the sampling noise differs (zero when seeded)

	RenderOptions turned;
	turned.has_lookat_override = true;
	turned.lookat_override[0] = 6.0;
	turned.lookat_override[1] = 1.0;
	turned.lookat_override[2] = 0.0;
	const double looking = meanAbsDiff(a, render("turned.ppm", turned));
	EXPECT_GT(looking, noise + 3.0) << "looking somewhere else gives another picture";

	RenderOptions wide;
	wide.vfov_override = 100.0;
	const double zoom = meanAbsDiff(a, render("wide.ppm", wide));
	EXPECT_GT(zoom, noise + 3.0) << "a different field of view gives another picture";

	RenderOptions same;
	same.has_lookat_override = false;
	same.vfov_override = -1.0;
	EXPECT_LE(meanAbsDiff(a, render("same.ppm", same)), noise + 3.0) << "no override: the scene's own camera";
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);
}
