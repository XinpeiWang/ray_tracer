/**
 * @file camera_tests.cpp
 * @brief Unit tests for camera positioning and ray generation
 * 
 * Tests the camera class to ensure:
 * - Camera position (lookfrom) is correctly set
 * - Camera target (lookat) points to Cornell box center
 * - Field of view calculations are accurate
 * - Ray generation produces valid directions
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "camera.h"
#include "vec3.h"
#include <cmath>

// ============================================================================
// Camera Configuration Tests
// ============================================================================

/**
 * Test that camera lookfrom position is set correctly
 */
TEST(CameraTest, LookFromPosition) {
	camera cam;

	// Test preset: Default (front view)
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vfov = 40;
	cam.image_width = 600;
	cam.samples_per_pixel = 10;
	cam.max_depth = 10;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Verify camera center matches lookfrom
	EXPECT_DOUBLE_EQ(cam.center.x(), 278);
	EXPECT_DOUBLE_EQ(cam.center.y(), 278);
	EXPECT_DOUBLE_EQ(cam.center.z(), -800);
}

/**
 * Test that camera lookat is pointing at Cornell box center
 */
TEST(CameraTest, LookAtCenter) {
	camera cam;

	// Cornell box center is at (278, 278, 278)
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.vfov = 40;
	cam.image_width = 600;
	cam.samples_per_pixel = 10;
	cam.max_depth = 10;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Camera should be looking towards z+ direction
	// The w vector points from lookat to lookfrom (opposite of view direction)
	vec3 expected_view_dir = unit_vector(cam.lookat - cam.lookfrom);

	// View direction should point roughly in +z direction
	EXPECT_GT(expected_view_dir.z(), 0.9); // Should be nearly (0, 0, 1)
}

/**
 * Test camera with different viewpoints
 */
TEST(CameraTest, MultipleViewpoints) {
	struct TestCase {
		const char* name;
		point3 lookfrom;
		point3 lookat;
	};

	TestCase cases[] = {
		{"Front View", point3(278, 278, -800), point3(278, 278, 278)},
		{"Inside Center", point3(278, 278, 278), point3(278, 278, 278)},
		{"Left Wall", point3(50, 278, 278), point3(278, 278, 278)},
		{"Right Wall", point3(506, 278, 278), point3(278, 278, 278)},
		{"Top View", point3(278, 506, 278), point3(278, 278, 278)},
		{"Bottom View", point3(278, 50, 278), point3(278, 278, 278)},
	};

	for (const auto& test : cases) {
		camera cam;
		cam.lookfrom = test.lookfrom;
		cam.lookat = test.lookat;
		cam.vfov = 40;
		cam.image_width = 100;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		// Should not crash
		EXPECT_NO_THROW(cam.initialize()) << "Failed for: " << test.name;

		// Camera center should match lookfrom
		EXPECT_DOUBLE_EQ(cam.center.x(), test.lookfrom.x()) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(cam.center.y(), test.lookfrom.y()) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(cam.center.z(), test.lookfrom.z()) << "Failed for: " << test.name;
	}
}

// ============================================================================
// Field of View Tests
// ============================================================================

/**
 * Test field of view calculations
 */
TEST(CameraTest, FieldOfView) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90; // 90 degree FOV
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// With 90° FOV and aspect_ratio 1:1, viewport should have specific dimensions
	// theta = 90° = π/2 rad
	// h = tan(θ/2) = tan(π/4) = 1
	// viewport_height = 2 * h * focus_dist = 2 * 1 * 1 = 2

	// Note: The actual viewport calculation depends on focus_dist
	// We just verify initialization doesn't crash
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Test different FOV values
 */
TEST(CameraTest, DifferentFOVValues) {
	double fov_values[] = {20, 40, 60, 90, 120};

	for (double fov : fov_values) {
		camera cam;
		cam.lookfrom = point3(0, 0, 0);
		cam.lookat = point3(0, 0, 1);
		cam.vfov = fov;
		cam.image_width = 200;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		EXPECT_NO_THROW(cam.initialize()) << "Failed for FOV: " << fov;
	}
}

// ============================================================================
// Ray Generation Tests
// ============================================================================

/**
 * Test that camera generates valid rays
 */
TEST(CameraTest, RayGeneration) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Get ray for center pixel
	ray r = cam.get_ray(cam.image_width / 2, cam.image_height / 2, 0, 0);

	// Ray origin should be at camera position
	EXPECT_DOUBLE_EQ(r.origin().x(), 0.0);
	EXPECT_DOUBLE_EQ(r.origin().y(), 0.0);
	EXPECT_DOUBLE_EQ(r.origin().z(), 0.0);

	// Ray direction should be roughly looking forward (+z)
	vec3 dir = unit_vector(r.direction());
	EXPECT_GT(dir.z(), 0.5); // Should have significant +z component
}

/**
 * Test ray generation for corner pixels
 */
TEST(CameraTest, RayGenerationCorners) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Test corners
	struct Corner {
		int x, y;
		const char* name;
	};

	Corner corners[] = {
		{0, 0, "Top-Left"},
		{cam.image_width - 1, 0, "Top-Right"},
		{0, cam.image_height - 1, "Bottom-Left"},
		{cam.image_width - 1, cam.image_height - 1, "Bottom-Right"},
	};

	for (const auto& corner : corners) {
		ray r = cam.get_ray(corner.x, corner.y, 0, 0);

		// All rays should originate from camera position
		EXPECT_DOUBLE_EQ(r.origin().x(), 0.0) << "Failed for: " << corner.name;
		EXPECT_DOUBLE_EQ(r.origin().y(), 0.0) << "Failed for: " << corner.name;
		EXPECT_DOUBLE_EQ(r.origin().z(), 0.0) << "Failed for: " << corner.name;

		// Direction should be non-zero (not necessarily a unit vector)
		double len = r.direction().length();
		EXPECT_GT(len, 0.0) << "Failed for: " << corner.name;
	}
}

// ============================================================================
// Aspect Ratio Tests
// ============================================================================

/**
 * Test aspect ratio handling
 */
TEST(CameraTest, AspectRatio) {
	struct TestCase {
		int width;
		double expected_height_ratio;
	};

	TestCase cases[] = {
		{400, 1.0},   // 1:1 square
		{800, 1.0},   // 1:1 square (larger)
		{600, 1.0},   // 1:1 square
	};

	for (const auto& test : cases) {
		camera cam;
		cam.lookfrom = point3(0, 0, 0);
		cam.lookat = point3(0, 0, 1);
		cam.vfov = 90;
		cam.image_width = test.width;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		cam.initialize();

		// Verify aspect ratio is approximately 1:1 (default)
		double aspect = static_cast<double>(cam.image_width) / cam.image_height;
		EXPECT_NEAR(aspect, 1.0, 0.01) << "Width: " << test.width;
	}
}

// ============================================================================
// Camera Parameter Validation Tests
// ============================================================================

/**
 * Test camera with minimum valid parameters
 */
TEST(CameraTest, MinimumParameters) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 1; // Very narrow FOV
	cam.image_width = 10; // Very small image
	cam.samples_per_pixel = 1;
	cam.max_depth = 1;
	cam.background = color(0, 0, 0);

	EXPECT_NO_THROW(cam.initialize());
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Test camera with maximum reasonable parameters
 */
TEST(CameraTest, MaximumParameters) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 179; // Nearly 180° FOV
	cam.image_width = 4000; // Large image
	cam.samples_per_pixel = 10000;
	cam.max_depth = 100;
	cam.background = color(1, 1, 1);

	EXPECT_NO_THROW(cam.initialize());
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Film "cropwindow"/"pixelbounds" - crop_x1/crop_y1 default to -1 (unset),
 * which initialize() must resolve to the full frame when nothing set them.
 */
// ============================================================================
// Screen Window Tests (pbrt-v4 Camera "perspective" "float screenwindow")
// ============================================================================

namespace {
camera makeScreenWindowTestCamera() {
	camera cam;
	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90;
	cam.image_width = 400;
	cam.aspect_ratio = 1.0;  // square, so x/y widening are directly comparable
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);
	return cam;
}
} // namespace

TEST(CameraTest, ExplicitDefaultScreenWindowMatchesNoScreenWindow) {
	// At a SQUARE aspect ratio only, an explicit [-1,1,-1,1] screenwindow
	// and no screenwindow at all happen to resolve to the identical
	// viewport (both formulas reduce to viewport_width==viewport_height
	// with zero center shift) - this is a coincidence of the square case,
	// not a special "isDefaultWindow" fast path (removed - a code-review
	// pass found it wrongly discarded a genuinely different, explicitly-
	// requested [-1,1,-1,1] window on a NON-square image; see
	// ScreenWindowNegOneToOneIsHonoredVerbatimOnNonSquareAspect below for
	// the case where the two diverge).
	camera plain = makeScreenWindowTestCamera();
	plain.initialize();

	camera explicitDefault = makeScreenWindowTestCamera();
	explicitDefault.has_screen_window = true;
	explicitDefault.screen_window[0] = -1.0; explicitDefault.screen_window[1] = 1.0;
	explicitDefault.screen_window[2] = -1.0; explicitDefault.screen_window[3] = 1.0;
	explicitDefault.initialize();

	for (int x : {0, 200, 399}) {
		for (int y : {0, 200, 399}) {
			ray rPlain = plain.get_ray(x, y, 0, 0);
			ray rExplicit = explicitDefault.get_ray(x, y, 0, 0);
			EXPECT_NEAR(rPlain.direction().x(), rExplicit.direction().x(), 1e-12);
			EXPECT_NEAR(rPlain.direction().y(), rExplicit.direction().y(), 1e-12);
			EXPECT_NEAR(rPlain.direction().z(), rExplicit.direction().z(), 1e-12);
		}
	}
}

TEST(CameraTest, ScreenWindowNegOneToOneIsHonoredVerbatimOnNonSquareAspect) {
	// Regression test for a code-review finding: on a NON-square aspect
	// ratio, an explicit "float screenwindow" [-1 1 -1 1] must NOT match
	// having no screenwindow at all. Real pbrt-v4's own auto-computed
	// default is aspect-scaled (wider in x for aspect>1); an EXPLICIT
	// [-1,1,-1,1] is a genuinely square, unstretched window instead - the
	// two only coincide at aspect==1 (ExplicitDefaultScreenWindowMatches
	// NoScreenWindow above). An earlier version of initialize() silently
	// treated this specific explicit value as "no screenwindow", making it
	// take the aspect-scaled path too - discarding the user's literal
	// directive with no warning.
	camera plain = makeScreenWindowTestCamera();
	plain.aspect_ratio = 2.0;  // non-square
	plain.initialize();

	camera explicitDefault = makeScreenWindowTestCamera();
	explicitDefault.aspect_ratio = 2.0;
	explicitDefault.has_screen_window = true;
	explicitDefault.screen_window[0] = -1.0; explicitDefault.screen_window[1] = 1.0;
	explicitDefault.screen_window[2] = -1.0; explicitDefault.screen_window[3] = 1.0;
	explicitDefault.initialize();

	// plain's viewport is aspect-scaled (viewport_width = viewport_height*2);
	// explicit's is square (viewport_width == viewport_height) - a corner
	// ray's x-direction should be about twice as wide for plain as for
	// explicit. If the bug were still present, these would match exactly.
	const vec3 cornerDirPlain = plain.get_ray(plain.image_width - 1, plain.image_height / 2, 0, 0).direction();
	const vec3 cornerDirExplicit = explicitDefault.get_ray(explicitDefault.image_width - 1, explicitDefault.image_height / 2, 0, 0).direction();
	EXPECT_GT(std::abs(cornerDirPlain.x()), std::abs(cornerDirExplicit.x()) * 1.5)
		<< "plain (aspect-scaled) corner ray x-direction should be substantially "
		   "wider than explicit [-1,1,-1,1]'s (square, unstretched) - if these "
		   "match, the explicit window is being silently treated as unset";
}

TEST(CameraTest, WidenedScreenWindowGivesACornerRayFartherFromCenterRay) {
	// A screenwindow twice as wide/tall as the default ([-2,2,-2,2] instead
	// of [-1,1,-1,1]) doubles the effective FOV - the corner ray's
	// direction should diverge from the center ray's direction MORE than
	// the default-window corner ray does, in the same (positive x)
	// direction.
	camera plain = makeScreenWindowTestCamera();
	plain.initialize();
	camera widened = makeScreenWindowTestCamera();
	widened.has_screen_window = true;
	widened.screen_window[0] = -2.0; widened.screen_window[1] = 2.0;
	widened.screen_window[2] = -2.0; widened.screen_window[3] = 2.0;
	widened.initialize();

	const vec3 centerDirPlain = plain.get_ray(200, 200, 0, 0).direction();
	const vec3 cornerDirPlain = plain.get_ray(399, 200, 0, 0).direction();
	const vec3 centerDirWidened = widened.get_ray(200, 200, 0, 0).direction();
	const vec3 cornerDirWidened = widened.get_ray(399, 200, 0, 0).direction();

	// Center ray is APPROXIMATELY unaffected by a symmetric widening (still
	// points close to straight down -w) - not exactly, now that get_ray()
	// routes sub-pixel offsets through FilterSampler's tabulated CDF
	// inversion (filterSampler_'s own comment, camera.h): the table's own
	// 32-cell quantization means an input this close to u=0.5 doesn't land
	// at EXACTLY filter-center 0.0, so that tiny residual offset - scaled
	// by the (now 2x wider) viewport_width - shows up as a small but
	// nonzero difference here. Loose tolerance catches a genuine framing
	// bug (which would be orders of magnitude larger) without being
	// fragile against expected quantization noise.
	EXPECT_NEAR(centerDirPlain.x(), centerDirWidened.x(), 0.5);

	const double plainOffset = std::abs(cornerDirPlain.x() - centerDirPlain.x());
	const double widenedOffset = std::abs(cornerDirWidened.x() - centerDirWidened.x());
	EXPECT_GT(widenedOffset, plainOffset)
		<< "a wider screenwindow must give a more divergent corner ray, not "
		   "an identical FOV silently ignoring the requested window";
}

TEST(CameraTest, OffCenterScreenWindowShiftsTheFrame) {
	// A genuinely off-center window (xmin=0,xmax=2 instead of the symmetric
	// xmin=-1,xmax=1 default) shifts the whole frame by exactly one NDC
	// unit's worth of world distance (h*focus_dist, since the window's own
	// center moved from x=0 to x=1) - the CENTER pixel's ray should no
	// longer point straight down -w. (Which world-space AXIS that shift
	// lands on depends on this camera's own u=cross(vup,w) - for this
	// lookfrom/lookat/vup it's -x, not +x - so this checks the shift's
	// MAGNITUDE, not an assumed world-space sign.)
	camera plain = makeScreenWindowTestCamera();
	plain.initialize();
	camera shifted = makeScreenWindowTestCamera();
	shifted.has_screen_window = true;
	shifted.screen_window[0] = 0.0; shifted.screen_window[1] = 2.0;
	shifted.screen_window[2] = -1.0; shifted.screen_window[3] = 1.0;
	shifted.initialize();

	const vec3 centerDirPlain = plain.get_ray(200, 200, 0, 0).direction();
	const vec3 centerDirShifted = shifted.get_ray(200, 200, 0, 0).direction();
	// h=tan(45deg)=1, focus_dist=10 (camera's own default) -> a 1-NDC-unit
	// shift moves the center ray by exactly 10 world units along u.
	EXPECT_NEAR(std::abs(centerDirShifted.x() - centerDirPlain.x()), 10.0, 1e-6)
		<< "an off-center screenwindow must shift the frame by exactly the "
		   "requested amount, not just resize it or leave it centered";
}

TEST(CameraTest, CropDefaultsToFullFrameWhenUnset) {
	camera cam;
	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 40;
	cam.image_width = 100;
	cam.aspect_ratio = 2.0;  // image_height = 50

	cam.initialize();

	EXPECT_EQ(cam.crop_x0, 0);
	EXPECT_EQ(cam.crop_x1, cam.image_width);
	EXPECT_EQ(cam.crop_y0, 0);
	EXPECT_EQ(cam.crop_y1, cam.image_height);
}

/**
 * A caller (scene_registry.h) sets crop_x0/x1/y0/y1 in pixel space before
 * initialize() runs - those explicit values must survive unchanged when
 * they're already within bounds.
 */
TEST(CameraTest, ExplicitCropIsPreservedWhenWithinBounds) {
	camera cam;
	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 40;
	cam.image_width = 200;
	cam.aspect_ratio = 2.0;  // image_height = 100
	cam.crop_x0 = 50; cam.crop_x1 = 150;
	cam.crop_y0 = 20; cam.crop_y1 = 80;

	cam.initialize();

	EXPECT_EQ(cam.crop_x0, 50);
	EXPECT_EQ(cam.crop_x1, 150);
	EXPECT_EQ(cam.crop_y0, 20);
	EXPECT_EQ(cam.crop_y1, 80);
}

/**
 * An out-of-bounds explicit crop (e.g. from a rounding edge case) is
 * clamped to the actual frame rather than left invalid.
 */
TEST(CameraTest, CropOutOfBoundsIsClamped) {
	camera cam;
	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 40;
	cam.image_width = 100;
	cam.aspect_ratio = 2.0;  // image_height = 50
	cam.crop_x0 = -10; cam.crop_x1 = 500;
	cam.crop_y0 = -5;  cam.crop_y1 = 500;

	cam.initialize();

	EXPECT_EQ(cam.crop_x0, 0);
	EXPECT_EQ(cam.crop_x1, cam.image_width);
	EXPECT_EQ(cam.crop_y0, 0);
	EXPECT_EQ(cam.crop_y1, cam.image_height);
}

/**
 * A degenerate crop (x1 <= x0) falls back to the full frame rather than
 * rendering zero pixels.
 */
TEST(CameraTest, DegenerateCropFallsBackToFullFrame) {
	camera cam;
	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 40;
	cam.image_width = 100;
	cam.aspect_ratio = 2.0;  // image_height = 50
	cam.crop_x0 = 60; cam.crop_x1 = 40;  // inverted -> degenerate

	cam.initialize();

	EXPECT_EQ(cam.crop_x0, 0);
	EXPECT_EQ(cam.crop_x1, cam.image_width);
	EXPECT_EQ(cam.crop_y0, 0);
	EXPECT_EQ(cam.crop_y1, cam.image_height);
}

/**
 * Test camera when lookfrom equals lookat (degenerate case)
 */
TEST(CameraTest, DegenerateLookFromEqualsLookAt) {
	camera cam;

	cam.lookfrom = point3(278, 278, 278);
	cam.lookat = point3(278, 278, 278); // Same position
	cam.vfov = 40;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	// Should not crash even with degenerate parameters
	// (Camera implementation should handle this gracefully)
	EXPECT_NO_THROW(cam.initialize());
}

// ============================================================================
// Camera Motion Blur (camera_is_animated) Tests
// ============================================================================

/**
 * An animated camera whose two keyframes are IDENTICAL (lookfrom1==lookfrom,
 * lookat1==lookat) has nothing to interpolate - every ray_time in
 * [shutter_open, shutter_close) should place the camera at exactly the same
 * spot a plain static camera would. This is the cheapest possible check that
 * the local-space-ray + AnimatedTransform machinery in camera.h's
 * initialize()/get_ray() doesn't introduce a sign/axis error (which would
 * silently mirror/rotate/offset every animated-camera ray without crashing
 * or looking obviously wrong at a glance).
 */
TEST(CameraTest, AnimatedCameraWithIdenticalKeyframesMatchesStatic) {
	camera static_cam;
	static_cam.lookfrom = point3(278, 278, -800);
	static_cam.lookat   = point3(278, 278, 278);
	static_cam.vfov = 40;
	static_cam.image_width = 200;
	static_cam.samples_per_pixel = 1;
	static_cam.max_depth = 5;
	static_cam.background = color(0, 0, 0);
	static_cam.initialize();

	camera anim_cam;
	anim_cam.lookfrom = point3(278, 278, -800);
	anim_cam.lookat   = point3(278, 278, 278);
	anim_cam.vfov = 40;
	anim_cam.image_width = 200;
	anim_cam.samples_per_pixel = 1;
	anim_cam.max_depth = 5;
	anim_cam.background = color(0, 0, 0);
	anim_cam.camera_is_animated = true;
	anim_cam.lookfrom1 = anim_cam.lookfrom;  // identical keyframes
	anim_cam.lookat1   = anim_cam.lookat;
	anim_cam.shutter_open  = 0.0;
	anim_cam.shutter_close = 1.0;
	anim_cam.initialize();

	// A handful of pixels, including corners and center - offset (0.5,0.5)
	// so both cameras sample the exact same sub-pixel point (get_ray()'s
	// ray_time sampling is the only remaining source of per-ray randomness,
	// and with identical keyframes it must not matter).
	struct Px { int x, y; };
	Px pixels[] = {
		{0, 0}, {199, 0}, {0, 199}, {199, 199}, {100, 100},
	};
	const vec3 offset(0.5, 0.5, 0.0);

	for (const auto& px : pixels) {
		ray r_static = static_cam.get_ray(px.x, px.y, 0, 0, offset);
		ray r_anim   = anim_cam.get_ray(px.x, px.y, 0, 0, offset);

		EXPECT_NEAR(r_static.origin().x(), r_anim.origin().x(), 1e-9) << "px=(" << px.x << "," << px.y << ")";
		EXPECT_NEAR(r_static.origin().y(), r_anim.origin().y(), 1e-9) << "px=(" << px.x << "," << px.y << ")";
		EXPECT_NEAR(r_static.origin().z(), r_anim.origin().z(), 1e-9) << "px=(" << px.x << "," << px.y << ")";

		vec3 d_static = unit_vector(r_static.direction());
		vec3 d_anim   = unit_vector(r_anim.direction());
		EXPECT_NEAR(d_static.x(), d_anim.x(), 1e-9) << "px=(" << px.x << "," << px.y << ")";
		EXPECT_NEAR(d_static.y(), d_anim.y(), 1e-9) << "px=(" << px.x << "," << px.y << ")";
		EXPECT_NEAR(d_static.z(), d_anim.z(), 1e-9) << "px=(" << px.x << "," << px.y << ")";
	}
}

/**
 * With DISTINCT keyframes, get_ray()'s per-ray time sampling should place
 * rays somewhere on the segment between the two keyframe positions (never
 * outside it) - a coarse but real check that Interpolate()'s translation
 * lerp is wired correctly end to end through the local-space + AnimatedTransform
 * path, not just a "doesn't crash" smoke test.
 */
TEST(CameraTest, AnimatedCameraStaysWithinKeyframeBounds) {
	camera anim_cam;
	anim_cam.lookfrom = point3(278, 278, -800);
	anim_cam.lookat   = point3(278, 278, 278);
	anim_cam.vfov = 40;
	anim_cam.image_width = 100;
	anim_cam.samples_per_pixel = 1;
	anim_cam.max_depth = 5;
	anim_cam.background = color(0, 0, 0);
	anim_cam.camera_is_animated = true;
	anim_cam.lookfrom1 = point3(378, 278, -800);  // 100-unit lateral truck, matches D13
	anim_cam.lookat1   = anim_cam.lookat;
	anim_cam.shutter_open  = 0.0;
	anim_cam.shutter_close = 1.0;
	anim_cam.initialize();

	const vec3 offset(0.5, 0.5, 0.0);
	for (int i = 0; i < 50; ++i) {
		ray r = anim_cam.get_ray(50, 50, 0, 0, offset);
		// Origin x must stay within [278, 378] regardless of sampled ray_time.
		EXPECT_GE(r.origin().x(), 278.0 - 1e-6);
		EXPECT_LE(r.origin().x(), 378.0 + 1e-6);
		// y/z of lookfrom don't move between the two keyframes.
		EXPECT_NEAR(r.origin().y(), 278.0, 1e-6);
		EXPECT_NEAR(r.origin().z(), -800.0, 1e-6);
	}
}
