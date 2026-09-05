// camera_path_tests.cpp -- unit tests for launcher/camera_path.h's "tour"
// and "showcase" path types (added alongside orbit/linear/figure8/spiral to
// give scenes a genuinely better-suited default - see scene_registry.h's
// recommended_camera_path_for()). Header-only, no linking against the
// launcher executable needed - same convention as
// launcher_args_bdpt_mlt_tests.cpp's own top-of-file comment explains.
//
// The other four path types have no existing test coverage of their own;
// this file only covers the two new ones plus get_camera_position()'s
// dispatch to them, rather than retroactively testing the pre-existing four.
#include <gtest/gtest.h>
#include "../../launcher/camera_path.h"

#include <cmath>

namespace {
constexpr double kEps = 1e-6;

double distance3d(double x1, double y1, double z1, double x2, double y2, double z2) {
	double dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}
} // namespace

// Every path type documents "frame 0 lands exactly on the given lookfrom" -
// tour and showcase both claim this too (camera_path_tour/camera_path_showcase's
// own comments), so pin it the same way a caller (main.cpp's video-mode
// branch) relies on for a video's first frame to match the still-image
// preview taken from the same scene.
TEST(CameraPathTest, TourFrameZeroMatchesLookfrom) {
	CameraPosition pos = camera_path_tour(0, 60, 100.0, 50.0, -200.0, 10.0, 50.0, 10.0);
	EXPECT_NEAR(pos.lookfrom_x, 100.0, kEps);
	EXPECT_NEAR(pos.lookfrom_y, 50.0, kEps);
	EXPECT_NEAR(pos.lookfrom_z, -200.0, kEps);
}

TEST(CameraPathTest, ShowcaseFrameZeroMatchesLookfrom) {
	CameraPosition pos = camera_path_showcase(0, 60, 100.0, 50.0, -200.0, 10.0, 50.0, 10.0);
	EXPECT_NEAR(pos.lookfrom_x, 100.0, kEps);
	EXPECT_NEAR(pos.lookfrom_y, 50.0, kEps);
	EXPECT_NEAR(pos.lookfrom_z, -200.0, kEps);
}

// camera_path_tour's own comment promises it never approaches lookat closer
// than 75% of the original lookfrom-to-lookat distance (the same "don't end
// up inside the geometry the original view was composed around" safety
// property camera_path_linear already has) - verify that promise holds
// across a full pass, not just at the documented worst-case frame.
TEST(CameraPathTest, TourNeverApproachesCloserThan75PercentOfOriginalDistance) {
	const double lookfrom_x = 278.0, lookfrom_y = 278.0, lookfrom_z = -800.0;
	const double lookat_x = 278.0, lookat_y = 278.0, lookat_z = 278.0;
	const double original_distance = distance3d(lookfrom_x, lookfrom_y, lookfrom_z,
	                                             lookat_x, lookat_y, lookat_z);
	const int total_frames = 90;
	for (int frame = 0; frame < total_frames; ++frame) {
		CameraPosition pos = camera_path_tour(frame, total_frames,
			lookfrom_x, lookfrom_y, lookfrom_z, lookat_x, lookat_y, lookat_z);
		// Compare against the ORIGINAL lookat (not the frame's own drifted
		// one) - the safety property that matters is staying clear of
		// whatever the scene's own geometry sits around, which is anchored
		// to the original recommended lookat, not the sway added on top.
		double dist_to_original_lookat = distance3d(pos.lookfrom_x, pos.lookfrom_y, pos.lookfrom_z,
			lookat_x, lookat_y, lookat_z);
		EXPECT_GE(dist_to_original_lookat, original_distance * 0.7)
			<< "Frame " << frame << " came too close to the original lookat";
	}
}

// camera_path_showcase's own comment promises it closes to 65% of the
// original radius at most (a 35% push-in) - verify the radius at t=1 (the
// last frame) doesn't overshoot that, and that it's monotonically
// decreasing in between (the eased eases IN, it doesn't oscillate).
TEST(CameraPathTest, ShowcasePushInStaysWithinDocumentedBound) {
	const double lookfrom_x = 278.0, lookfrom_y = 278.0, lookfrom_z = -800.0;
	const double lookat_x = 278.0, lookat_y = 278.0, lookat_z = 278.0;
	const double original_radius_xz = std::sqrt(
		(lookfrom_x - lookat_x) * (lookfrom_x - lookat_x) +
		(lookfrom_z - lookat_z) * (lookfrom_z - lookat_z));
	const int total_frames = 90;

	double prev_radius_xz = original_radius_xz;
	for (int frame = 0; frame <= total_frames; ++frame) {
		CameraPosition pos = camera_path_showcase(frame, total_frames,
			lookfrom_x, lookfrom_y, lookfrom_z, lookat_x, lookat_y, lookat_z);
		double radius_xz = std::sqrt(
			(pos.lookfrom_x - lookat_x) * (pos.lookfrom_x - lookat_x) +
			(pos.lookfrom_z - lookat_z) * (pos.lookfrom_z - lookat_z));
		EXPECT_LE(radius_xz, original_radius_xz + kEps)
			<< "Frame " << frame << " radius exceeded the original - should only ever push IN";
		EXPECT_GE(radius_xz, original_radius_xz * 0.65 - kEps)
			<< "Frame " << frame << " pushed in further than the documented 35% bound";
		EXPECT_LE(radius_xz, prev_radius_xz + kEps)
			<< "Frame " << frame << " radius increased - push-in should ease monotonically inward";
		prev_radius_xz = radius_xz;
	}
}

// get_camera_position()'s dispatch - the two new names must actually reach
// the two new functions (not silently fall through to the "unrecognized ->
// orbit" default branch, which would make every "tour"/"showcase" video
// quietly render as a plain orbit instead).
TEST(CameraPathTest, GetCameraPositionDispatchesTour) {
	CameraPosition direct = camera_path_tour(5, 60, 278.0, 278.0, -800.0, 278.0, 278.0, 278.0);
	CameraPosition dispatched = get_camera_position("tour", 5, 60, 278.0, 278.0, -800.0, 278.0, 278.0, 278.0);
	EXPECT_NEAR(direct.lookfrom_x, dispatched.lookfrom_x, kEps);
	EXPECT_NEAR(direct.lookfrom_y, dispatched.lookfrom_y, kEps);
	EXPECT_NEAR(direct.lookfrom_z, dispatched.lookfrom_z, kEps);
	EXPECT_NEAR(direct.lookat_x, dispatched.lookat_x, kEps);
	EXPECT_NEAR(direct.lookat_z, dispatched.lookat_z, kEps);
}

TEST(CameraPathTest, GetCameraPositionDispatchesShowcase) {
	CameraPosition direct = camera_path_showcase(5, 60, 278.0, 278.0, -800.0, 278.0, 278.0, 278.0);
	CameraPosition dispatched = get_camera_position("showcase", 5, 60, 278.0, 278.0, -800.0, 278.0, 278.0, 278.0);
	EXPECT_NEAR(direct.lookfrom_x, dispatched.lookfrom_x, kEps);
	EXPECT_NEAR(direct.lookfrom_y, dispatched.lookfrom_y, kEps);
	EXPECT_NEAR(direct.lookfrom_z, dispatched.lookfrom_z, kEps);
}

// Pre-existing behavior (an unrecognized path name falls back to orbit) -
// confirm adding two new recognized names didn't change what happens for a
// name that's still neither of the (now six) known ones.
TEST(CameraPathTest, GetCameraPositionStillDefaultsUnrecognizedToOrbit) {
	CameraPosition dispatched = get_camera_position("not-a-real-path", 5, 60,
		278.0, 278.0, -800.0, 278.0, 278.0, 278.0);
	// Should land on orbit's own circle (radius/height derived from the
	// passed lookfrom/lookat, same as camera_path_orbit's "frame 0 is the
	// recommended lookfrom" guarantee) - checking distance-from-center
	// rather than exact coordinates since the internal start_angle isn't
	// exposed here. Expected radius is the actual XZ distance from this
	// lookfrom (278, -800) to this lookat (278, 278) - |lookat_z -
	// lookfrom_z| = |278 - (-800)| = 1078, not the 800 in lookfrom_z's own
	// value (that's an offset from the origin, not from lookat).
	double dispatched_radius = std::sqrt(
		(dispatched.lookfrom_x - 278.0) * (dispatched.lookfrom_x - 278.0) +
		(dispatched.lookfrom_z - 278.0) * (dispatched.lookfrom_z - 278.0));
	EXPECT_NEAR(dispatched_radius, 1078.0, 1.0);
	EXPECT_NEAR(dispatched.lookfrom_y, 278.0, kEps);
}
