/**
 * @file pbrt_scene_render_test.cpp
 * @brief End-to-end: a pbrt scene text actually renders
 *
 * The unit tests along the chain each prove one link. The ray-cast tests in
 * pbrt_cpu_builder_tests.cpp go furthest, showing geometry ends up where the
 * scene said - but hitting a triangle is not the same as shading it. Nothing
 * so far has driven a pbrt-derived world through ray_color(), so nothing has
 * shown that the materials attach, that emissive shapes actually emit, or that
 * the camera derived from pbrt's world-to-camera matrix points at the geometry
 * rather than away from it.
 *
 * A camera aimed 180 degrees wrong renders a perfectly clean black image and
 * passes every test that only checks for finite values, which is exactly why
 * this one insists on light arriving.
 */

#include <gtest/gtest.h>

#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "color.h"
#include "power_light_sampler.h"

#include "pbrt_cpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

#include <cmath>
#include <string>

namespace {

// A miniature Cornell box in pbrt syntax: an emissive quad on the ceiling, a
// diffuse floor and back wall, and a sphere between them. Small enough to
// render in a fraction of a second, complete enough that light has to bounce
// off a surface to reach the camera.
const char *kMiniCornell = R"PBRT(
LookAt 0 1 -4    0 1 0    0 1 0
Camera "perspective" "float fov" [ 50 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 32 ]
WorldBegin

# Ceiling light, emissive and facing down into the box
AttributeBegin
  AreaLightSource "diffuse" "rgb L" [ 12 12 12 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ -0.6 2.0 -0.6   0.6 2.0 -0.6   0.6 2.0 0.6   -0.6 2.0 0.6 ]
AttributeEnd

Material "diffuse" "rgb reflectance" [ 0.75 0.75 0.75 ]

# Floor
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -2   2 0 -2   2 0 2   -2 0 2 ]

# Back wall
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]

# A sphere sitting on the floor
Shape "sphere" "float radius" [ 0.5 ]
)PBRT";

// Mean Rec.709 luminance over the whole frame, and whether anything was
// non-finite. Sampling every pixel rather than a window: the point here is
// "the scene is lit and sane", not the brightness of one object.
struct FrameStats {
	double meanLuminance = 0.0;
	double maxLuminance = 0.0;
	bool allFinite = true;
	int litPixels = 0;
};

FrameStats renderStats(const hittable &world, const hittable &lights,
					   const camera &cam, int spp) {
	FrameStats st;
	double sum = 0.0;
	int count = 0;
	for (int j = 0; j < cam.image_height; ++j) {
		for (int i = 0; i < cam.image_width; ++i) {
			for (int s = 0; s < spp; ++s) {
				const ray r = cam.get_ray(i, j, 0, 0, vec3(0, 0, 0));
				SobolSampler ps(s * (cam.image_width * cam.image_height)
								+ j * cam.image_width + i, i, j);
				const color c = cam.ray_color(r, cam.max_depth, world, lights, ps);
				const double lum = 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
				if (!std::isfinite(lum)) st.allFinite = false;
				if (lum > 0.01) ++st.litPixels;
				if (lum > st.maxLuminance) st.maxLuminance = lum;
				sum += lum;
				++count;
			}
		}
	}
	st.meanLuminance = (count > 0) ? sum / count : 0.0;
	return st;
}

// Runs the whole chain: pbrt text -> description -> world geometry ->
// hittables, and configures our camera from the scene's own camera.
struct Loaded {
	pbrt_cpu::BuildResult built;
	camera cam;
};

Loaded loadAndAim(const std::string &text, int spp) {
	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text);
	EXPECT_TRUE(parsed.ok) << parsed.error;
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(parsed.scene);

	Loaded out;
	out.built = pbrt_cpu::build(flat);

	// Everything here comes from the scene, not from the test - that is the
	// point. If the camera derivation is wrong, this aims somewhere useless.
	out.cam.aspect_ratio = 1.0;
	out.cam.image_width = parsed.scene.xResolution;
	out.cam.samples_per_pixel = spp;
	out.cam.max_depth = 8;
	out.cam.background = color(0, 0, 0);
	out.cam.vfov = flat.camera.vfov;
	out.cam.lookfrom = point3(flat.camera.lookfrom[0], flat.camera.lookfrom[1],
							  flat.camera.lookfrom[2]);
	out.cam.lookat = point3(flat.camera.lookat[0], flat.camera.lookat[1],
							flat.camera.lookat[2]);
	out.cam.vup = vec3(flat.camera.up[0], flat.camera.up[1], flat.camera.up[2]);
	out.cam.initialize();
	return out;
}

} // namespace

TEST(PbrtSceneRender, MiniCornellBoxRendersLit) {
	Loaded l = loadAndAim(kMiniCornell, 8);

	ASSERT_GT(l.built.triangleCount, 0u);
	ASSERT_EQ(l.built.sphereCount, 1u);
	ASSERT_FALSE(l.built.lights->objects.empty())
		<< "the ceiling quad should have been collected as a light";

	// Wrapped exactly as cpu_interface.cpp does it - a raw hittable_list is
	// not what the renderer is given, and an empty one is not safe to sample.
	const power_light_list lights(*l.built.lights);
	const FrameStats st = renderStats(*l.built.world, lights, l.cam, 8);

	EXPECT_TRUE(st.allFinite) << "a NaN or infinity reached the film";
	EXPECT_GT(st.meanLuminance, 0.005)
		<< "the frame is essentially black - the camera may be pointed away "
		   "from the geometry, or the light is not emitting";
	EXPECT_GT(st.litPixels, 100)
		<< "only " << st.litPixels << " pixels received light, which suggests "
		   "the camera is clipping the scene rather than framing it";
	// The emitter itself is in view, so something must be much brighter than
	// the diffuse walls - a uniformly dim frame means emission never landed.
	EXPECT_GT(st.maxLuminance, 1.0)
		<< "nothing in frame is brighter than diffuse bounce light";
}

TEST(PbrtSceneRender, RemovingTheLightMakesTheFrameBlack) {
	// The control for the test above. Without it, a bug that made every ray
	// return a constant non-zero colour would pass "the frame is lit" happily.
	std::string noLight = kMiniCornell;
	const std::string tag = "AreaLightSource \"diffuse\" \"rgb L\" [ 12 12 12 ]";
	const std::size_t at = noLight.find(tag);
	ASSERT_NE(at, std::string::npos);
	noLight.erase(at, tag.size());

	Loaded l = loadAndAim(noLight, 4);
	EXPECT_TRUE(l.built.lights->objects.empty());

	const power_light_list lights(*l.built.lights);
	const FrameStats st = renderStats(*l.built.world, lights, l.cam, 4);
	EXPECT_TRUE(st.allFinite);
	EXPECT_LT(st.meanLuminance, 1e-6)
		<< "with no emitter and a black background the frame must be black; "
		   "a non-zero result means light is coming from somewhere unintended";
}
