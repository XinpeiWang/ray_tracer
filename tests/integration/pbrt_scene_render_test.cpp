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

// A closed box: the same walls as kMiniCornell plus a front wall behind the
// camera. The distinction matters for anything specular. A mirror compresses
// the whole environment into its silhouette, and the part it puts in the
// MIDDLE of its visible disc is whatever lies behind the viewer - so in an
// open-fronted box a mirror sphere is legitimately black across most of its
// face, and "the metal renders black" is not evidence of a broken material.
// Closing the box removes that confound entirely.
const char *kClosedBoxMetal = R"PBRT(
LookAt 0 1 -4    0 1 0    0 1 0
Camera "perspective" "float fov" [ 50 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 32 ]
WorldBegin

AttributeBegin
  AreaLightSource "diffuse" "rgb L" [ 12 12 12 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ -0.6 2.0 -0.6   0.6 2.0 -0.6   0.6 2.0 0.6   -0.6 2.0 0.6 ]
AttributeEnd

Material "diffuse" "rgb reflectance" [ 0.75 0.75 0.75 ]
# floor, back wall, ceiling, both sides, and a wall behind the camera
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   2 0 -6   2 0 2   -2 0 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 3 -6   2 3 -6   2 3 2   -2 3 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   -2 0 2   -2 3 2   -2 3 -6 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 2 0 -6   2 0 2   2 3 2   2 3 -6 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   2 0 -6   2 3 -6   -2 3 -6 ]

AttributeBegin
  Translate 0 0.5 0
  Material "conductor" "float roughness" [ 0.3 ]
  Shape "sphere" "float radius" [ 0.5 ]
AttributeEnd
)PBRT";

// The same closed box, every coordinate multiplied by 100. Real pbrt scenes
// are authored at whatever scale suited their modeller - the canonical
// Cornell box is 555 units across, not 5 - so a renderer that only works
// near unit scale works on almost nothing. Separating this from the scene
// above is the point: if the small one passes and this one fails, the fault
// is scale-dependent, which is a very different bug from a broken material.
const char *kClosedBoxMetalLarge = R"PBRT(
LookAt 0 100 -400    0 100 0    0 1 0
Camera "perspective" "float fov" [ 50 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 32 ]
WorldBegin

AttributeBegin
  AreaLightSource "diffuse" "rgb L" [ 12 12 12 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ -60 200 -60   60 200 -60   60 200 60   -60 200 60 ]
AttributeEnd

Material "diffuse" "rgb reflectance" [ 0.75 0.75 0.75 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -200 0 -600   200 0 -600   200 0 200   -200 0 200 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -200 0 200   200 0 200   200 300 200   -200 300 200 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -200 300 -600   200 300 -600   200 300 200   -200 300 200 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -200 0 -600   -200 0 200   -200 300 200   -200 300 -600 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 200 0 -600   200 0 200   200 300 200   200 300 -600 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -200 0 -600   200 0 -600   200 300 -600   -200 300 -600 ]

AttributeBegin
  Translate 0 50 0
  Material "conductor" "float roughness" [ 0.3 ]
  Shape "sphere" "float radius" [ 50 ]
AttributeEnd
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
	// The lens settings are configured exactly as scene_registry.h does it,
	// including going through focusDistanceFor() rather than reading
	// focusDistance directly. Without this the harness was quietly kinder to
	// the renderer than the real path is, and missed a bug that deleted
	// geometry in every loaded scene.
	//
	// defocus_angle also goes through defocusAngleDegreesFor() rather than
	// using flat.camera.aperture directly - that field is a world-space lens
	// DIAMETER, not a degrees value (see that function's own comment); this
	// test used to carry the same unit-mismatch bug scene_registry.h had.
	out.cam.focus_dist = pbrt_flatten::focusDistanceFor(flat.camera);
	out.cam.defocus_angle = pbrt_flatten::defocusAngleDegreesFor(flat.camera, out.cam.focus_dist);
	// nullptr (no punctual lights) unless the scene declared LightSource
	// point/spot/distant/goniometric/projection - same wiring
	// cpu_interface.cpp does for every other pbrt scene (see scene_registry.h's
	// build_punct). Harmless for the tests above, which have none.
	out.cam.punct_lights = out.built.punctLights;
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

TEST(PbrtSceneRender, MediumInterfaceSphereRendersWithoutNaN) {
	// kMiniCornell's sphere, but wrapped in a homogeneous fog medium via
	// MakeNamedMedium/MediumInterface - the end-to-end path this segment's
	// loader work added (pbrt_scene.h's dispatch -> pbrt_flatten::Medium ->
	// pbrt_cpu_builder.h's constant_medium wrap). The structural claim (one
	// extra hittable) is already covered in pbrt_cpu_builder_tests.cpp; this
	// proves the medium survives contact with the real multithreaded-capable
	// render path without producing non-finite pixels.
	const char *kMiniCornellWithFog = R"PBRT(
LookAt 0 1 -4    0 1 0    0 1 0
Camera "perspective" "float fov" [ 50 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 32 ]
WorldBegin

AttributeBegin
  AreaLightSource "diffuse" "rgb L" [ 12 12 12 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ -0.6 2.0 -0.6   0.6 2.0 -0.6   0.6 2.0 0.6   -0.6 2.0 0.6 ]
AttributeEnd

Material "diffuse" "rgb reflectance" [ 0.75 0.75 0.75 ]

Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -2   2 0 -2   2 0 2   -2 0 2 ]

Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]

MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [ 0.05 0.05 0.05 ] "rgb sigma_s" [ 3 3 3 ]

AttributeBegin
  MediumInterface "fog" ""
  Shape "sphere" "float radius" [ 0.5 ]
AttributeEnd
)PBRT";

	Loaded l = loadAndAim(kMiniCornellWithFog, 8);
	ASSERT_EQ(l.built.sphereCount, 1u);

	const power_light_list lights(*l.built.lights);
	const FrameStats st = renderStats(*l.built.world, lights, l.cam, 8);

	EXPECT_TRUE(st.allFinite) << "a NaN or infinity reached the film";
	EXPECT_GT(st.meanLuminance, 0.0) << "the frame is entirely black";
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

// A conductor sphere is not black. This exists because a render of one in an
// OPEN-fronted box looked like proof of a broken material, and was not: a
// mirror shows the environment behind the viewer in the centre of its disc,
// and there the environment behind the viewer is an unlit void. Closing the
// box is what turns "looks black" into a claim that can be tested at all.
//
// The assertion is on the sphere's own pixels rather than the whole frame,
// because a lit room around a black sphere passes any frame-wide check.
TEST(PbrtSceneRender, AConductorInAClosedBoxReflectsTheRoomRatherThanRenderingBlack) {
	Loaded l = loadAndAim(kClosedBoxMetal, 24);
	ASSERT_EQ(l.built.sphereCount, 1u);

	const power_light_list lights(*l.built.lights);

	// The sphere sits at the centre of frame; sample the middle of its disc,
	// which is exactly the region an open box would have made black.
	double sum = 0.0;
	int samples = 0;
	const int cx = l.cam.image_width / 2, cy = l.cam.image_height / 2;
	for (int j = cy - 2; j <= cy + 2; ++j) {
		for (int i = cx - 2; i <= cx + 2; ++i) {
			for (int s = 0; s < 24; ++s) {
				const ray r = l.cam.get_ray(i, j, 0, 0, vec3(0, 0, 0));
				SobolSampler ps(s * 4096 + j * 64 + i, i, j);
				const color c = l.cam.ray_color(r, l.cam.max_depth, *l.built.world,
												lights, ps);
				sum += 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
				++samples;
			}
		}
	}
	const double mean = sum / samples;
	EXPECT_GT(mean, 0.01)
		<< "the conductor's own pixels are black (mean luminance " << mean
		<< ") even with a wall behind the camera for it to reflect";
}

// Same assertion, same scene, 100x the size. Split out from the test above so
// a failure says WHICH property is broken rather than just "metal is black".
TEST(PbrtSceneRender, AConductorStillReflectsWhenTheSceneIsAuthoredAtLargeScale) {
	Loaded l = loadAndAim(kClosedBoxMetalLarge, 24);
	ASSERT_EQ(l.built.sphereCount, 1u);
	const power_light_list lights(*l.built.lights);

	double sum = 0.0;
	int samples = 0;
	const int cx = l.cam.image_width / 2, cy = l.cam.image_height / 2;
	for (int j = cy - 2; j <= cy + 2; ++j) {
		for (int i = cx - 2; i <= cx + 2; ++i) {
			for (int s = 0; s < 24; ++s) {
				const ray r = l.cam.get_ray(i, j, 0, 0, vec3(0, 0, 0));
				SobolSampler ps(s * 4096 + j * 64 + i, i, j);
				const color c = l.cam.ray_color(r, l.cam.max_depth, *l.built.world,
												lights, ps);
				sum += 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
				++samples;
			}
		}
	}
	const double mean = sum / samples;
	EXPECT_GT(mean, 0.01)
		<< "identical scene to the previous test, scaled 100x, and the metal "
		   "went black (mean luminance " << mean << ")";
}

// Reproduces the exact configuration that renders a metal sphere as a hole:
// the bundled example scene's geometry, minus the glass sphere. The two tests
// above pass, so whatever breaks it is in the difference between them - which
// is why this one is a copy of the real thing rather than another variation.
const char *kOpenBoxMetal = R"PBRT(
LookAt  278 278 -800    278 278 0    0 1 0
Camera "perspective" "float fov" [ 40 ]
Film "rgb" "integer xresolution" [ 48 ] "integer yresolution" [ 48 ]
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "rgb L" [ 18 15 8 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ 213 548.7 227   343 548.7 227   343 548.7 332   213 548.7 332 ]
AttributeEnd
Material "diffuse" "rgb reflectance" [ 0.73 0.73 0.73 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 0 0 0   555 0 0   555 0 555   0 0 555 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 0 555 0   555 555 0   555 555 555   0 555 555 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 0 0 555   555 0 555   555 555 555   0 555 555 ]
AttributeBegin
  Material "diffuse" "rgb reflectance" [ 0.65 0.05 0.05 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ 555 0 0   555 0 555   555 555 555   555 555 0 ]
AttributeEnd
AttributeBegin
  Material "diffuse" "rgb reflectance" [ 0.12 0.45 0.15 ]
  Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
    "point3 P" [ 0 0 0   0 0 555   0 555 555   0 555 0 ]
AttributeEnd
AttributeBegin
  Translate 185 120 169
  Material "conductor" "float roughness" [ 0.15 ]
  Shape "sphere" "float radius" [ 120 ]
AttributeEnd
)PBRT";

TEST(PbrtSceneRender, MetalSphereIsNotAHoleInTheOpenCornellBox) {
	Loaded l = loadAndAim(kOpenBoxMetal, 16);
	ASSERT_EQ(l.built.sphereCount, 1u);
	const power_light_list lights(*l.built.lights);

	// The sphere's screen position is hardcoded rather than found by tracing a
	// probe ray, and that is deliberate. The first version of this test used
	// world->hit() to decide which pixels covered the sphere - but that probe
	// travels down the same distorted ray as the render, so when the bug was
	// present it missed the sphere too and happily measured the lit floor
	// behind it. The test passed with the bug in place, which is worse than
	// having no test. Fixed camera plus fixed geometry means these coordinates
	// are a constant of the scene, and a constant cannot collude with the bug.
	const int x0 = static_cast<int>(l.cam.image_width * 0.52);
	const int x1 = static_cast<int>(l.cam.image_width * 0.74);
	const int y0 = static_cast<int>(l.cam.image_height * 0.55);
	const int y1 = static_cast<int>(l.cam.image_height * 0.85);

	double brightest = 0.0;
	int samples = 0, lit = 0;
	for (int j = y0; j <= y1; ++j) {
		for (int i = x0; i <= x1; ++i) {
			for (int s = 0; s < 16; ++s) {
				const ray r = l.cam.get_ray(i, j, 0, 0, vec3(0, 0, 0));
				SobolSampler ps(s * 4096 + j * 64 + i, i, j);
				const color c = l.cam.ray_color(r, l.cam.max_depth, *l.built.world,
												lights, ps);
				const double lum = 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
				if (lum > 0.0) ++lit;
				if (lum > brightest) brightest = lum;
				++samples;
			}
		}
	}

	// The middle of the disc is legitimately black - that is where a mirror
	// puts whatever is behind the viewer, and this box is open there - so the
	// assertion is that the sphere REFLECTS SOMETHING, not that it is
	// uniformly lit. When the bug was present, every sample here was exactly
	// 0.0 and the floor one pixel outside read ~100/255; nothing physical
	// steps from lit to absolute zero across a single pixel.
	EXPECT_GT(lit, samples / 10)
		<< "only " << lit << " of " << samples << " samples over the metal "
		   "sphere are non-zero - it is rendering as a hole, not a mirror";
	EXPECT_GT(brightest, 0.05)
		<< "nothing in the sphere reflects the lit walls beside it";
}

// ---------------------------------------------------------------------------
// LightSource point/spot/distant/goniometric/projection: end-to-end
//
// Everything above proves AreaLightSource works end-to-end. These are the
// same shape of test for the five punctual (delta) light kinds Priority 1
// added: a box with NO emissive geometry at all, lit purely by one
// LightSource directive - if flatten()'s parsing, pbrt_cpu_builder.h's
// wiring, or camera.h's punct_lights NEE block disagreed about any of these,
// the box renders black and every test below catches it exactly the way
// RemovingTheLightMakesTheFrameBlack catches a missing area light.
//
// The box itself is kClosedBoxMetal's wall layout with no ceiling light quad
// and a plain diffuse sphere instead of a conductor, so "is it lit" is a
// direct, unambiguous read rather than something that also depends on a
// mirror finding something to reflect.
// ---------------------------------------------------------------------------

const char *kPunctualLightBoxHeader = R"PBRT(
LookAt 0 1 -4    0 1 0    0 1 0
Camera "perspective" "float fov" [ 50 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 32 ]
WorldBegin

Material "diffuse" "rgb reflectance" [ 0.75 0.75 0.75 ]
# floor, back wall, ceiling, both side walls, and a wall behind the camera -
# fully closed, so a light anywhere inside always has something to light.
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   2 0 -6   2 0 2   -2 0 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 2   2 0 2   2 3 2   -2 3 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 3 -6   2 3 -6   2 3 2   -2 3 2 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   -2 0 2   -2 3 2   -2 3 -6 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ 2 0 -6   2 0 2   2 3 2   2 3 -6 ]
Shape "trianglemesh" "integer indices" [ 0 1 2  0 2 3 ]
  "point3 P" [ -2 0 -6   2 0 -6   2 3 -6   -2 3 -6 ]

# A sphere sitting on the floor, in frame.
Shape "sphere" "float radius" [ 0.5 ]
)PBRT";

// Renders `lightDirective` (a single LightSource line) inside the closed box
// above and returns whether the frame is lit. Shared by all five kinds
// below so a failure's cause is "this one light kind" rather than a copy-
// pasted assertion block silently drifting between them.
FrameStats renderPunctualLightBox(const std::string &lightDirective, int spp) {
	const std::string text = std::string(kPunctualLightBoxHeader) + lightDirective + "\n";
	Loaded l = loadAndAim(text, spp);
	EXPECT_TRUE(l.built.lights->objects.empty())
		<< "this scene has no AreaLightSource - anything in the light list "
		   "would mean geometry was misclassified as emissive";
	// EXPECT rather than ASSERT: this function returns a value, and ASSERT_*
	// expands to a bare `return;`, which does not typecheck against
	// FrameStats. A null punctLights still safely renders (as an unlit black
	// box) below, so the meanLuminance/litPixels checks each caller makes
	// catch this failure mode too - just with a less specific message.
	EXPECT_NE(l.built.punctLights, nullptr)
		<< "the LightSource directive was not recognised as a punctual light";

	const power_light_list lights(*l.built.lights);
	return renderStats(*l.built.world, lights, l.cam, spp);
}

TEST(PbrtPunctualLightRender, PointLightLitsTheBox) {
	const FrameStats st = renderPunctualLightBox(
		"LightSource \"point\" \"point3 from\" [ 0 2.5 0 ] \"rgb I\" [ 1 1 1 ] "
		"\"float scale\" [ 40 ]", 8);
	EXPECT_TRUE(st.allFinite);
	EXPECT_GT(st.meanLuminance, 0.01)
		<< "the box is essentially black under a point light overhead";
	EXPECT_GT(st.litPixels, 100);
}

TEST(PbrtPunctualLightRender, SpotLightLitsTheBox) {
	// CTM = Translate * Rotate, and pbrt applies the LAST-written transform
	// to a point FIRST (gs_.ctm = gs_.ctm * newTransform in pbrt_scene.h) -
	// so this rotates the light's default (0,0,1) aim 90 degrees about x
	// BEFORE translating the whole thing up to y=2.5, turning "look toward
	// +z" into "look toward -y": straight down at the sphere/floor below.
	const FrameStats st = renderPunctualLightBox(
		"Translate 0 2.5 0\nRotate 90 1 0 0\n"
		"LightSource \"spot\" \"rgb I\" [ 1 1 1 ] \"float scale\" [ 60 ] "
		"\"float coneangle\" [ 60 ]", 8);
	EXPECT_TRUE(st.allFinite);
	EXPECT_GT(st.meanLuminance, 0.01)
		<< "the box is essentially black under an overhead spotlight";
	EXPECT_GT(st.litPixels, 50);
}

// Distant is the one punctual kind renderPunctualLightBox's fully-closed six-
// wall room cannot exercise: its shadow ray has t_max = infinity (see
// distant_light_obj::sample_direct(), punctual_light_objects.h), and a ray
// cast in ANY direction from inside a sealed room hits a wall before
// infinity - a real physical fact (a sun has no way into a room with no
// windows), not a bug. This reuses kMiniCornell's OPEN layout instead (floor
// and back wall only - four full walls short of renderPunctualLightBox's
// room), the same way scenes_advanced.h's own build_distant_light_punct()
// needs cornell_walls_no_light()'s open front for the identical reason.
TEST(PbrtPunctualLightRender, DistantLightLitsTheBox) {
	std::string text = kMiniCornell;
	// Strip the emissive ceiling quad (kMiniCornell's own light) so the only
	// illumination in this render comes from the LightSource below - same
	// removal RemovingTheLightMakesTheFrameBlack does, reused here to turn a
	// lit fixture into an unlit one rather than authoring a third variant.
	const std::string tag = "AreaLightSource \"diffuse\" \"rgb L\" [ 12 12 12 ]";
	const std::size_t at = text.find(tag);
	ASSERT_NE(at, std::string::npos);
	text.erase(at, tag.size());
	text += "LightSource \"distant\" \"point3 from\" [ 0 10 -2 ] \"point3 to\" [ 0 0 0 ] "
			"\"rgb L\" [ 3 3 3 ]\n";

	Loaded l = loadAndAim(text, 8);
	EXPECT_TRUE(l.built.lights->objects.empty());
	EXPECT_NE(l.built.punctLights, nullptr);
	const power_light_list lights(*l.built.lights);
	const FrameStats st = renderStats(*l.built.world, lights, l.cam, 8);

	EXPECT_TRUE(st.allFinite);
	EXPECT_GT(st.meanLuminance, 0.01)
		<< "the box is essentially black under a distant (directional) light";
	EXPECT_GT(st.litPixels, 100);
}

TEST(PbrtPunctualLightRender, GoniometricLightLitsTheBox) {
	// No "filename" - falls back to an isotropic profile (see
	// FlattenPunctualLightTest::GoniometricLightWithNoFilenameIsIsotropicAndUnwarned,
	// pbrt_flatten_tests.cpp) - so this should light the box exactly like the
	// plain point-light test above.
	const FrameStats st = renderPunctualLightBox(
		"Translate 0 2.5 0\n"
		"LightSource \"goniometric\" \"rgb I\" [ 1 1 1 ] \"float scale\" [ 40 ]", 8);
	EXPECT_TRUE(st.allFinite);
	EXPECT_GT(st.meanLuminance, 0.01)
		<< "the box is essentially black under an (isotropic-fallback) "
		   "goniometric light";
	EXPECT_GT(st.litPixels, 100);
}

TEST(PbrtPunctualLightRender, ProjectionLightLitsTheBox) {
	// No "filename" either - falls back to a uniform white beam (see
	// flatten()'s own warning for this case) - aimed from the camera's
	// position straight down +z at the sphere/back wall.
	const FrameStats st = renderPunctualLightBox(
		"Translate 0 1 -4\n"
		"LightSource \"projection\" \"float scale\" [ 30 ] \"float fov\" [ 60 ]", 8);
	EXPECT_TRUE(st.allFinite);
	EXPECT_GT(st.meanLuminance, 0.005)
		<< "the box is essentially black under a (uniform-fallback) "
		   "projection light aimed into it";
	EXPECT_GT(st.litPixels, 30);
}
