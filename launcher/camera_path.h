#ifndef CAMERA_PATH_H
#define CAMERA_PATH_H

// ============================================================================
// Camera Path Animation
// ============================================================================
// Provides parametric camera animation paths for video generation.
// Each function takes (frame_number, total_frames) and returns camera position.
//
// These functions are speed-agnostic: given a frame index and a total frame
// count, they always place the camera at the corresponding point along one
// full baseline traversal (1 full rotation for orbit/figure8, 2 for spiral,
// the whole start->end sweep for linear). "Movement speed" for video
// rendering is implemented by the caller choosing how many actual frames to
// render for a given total_frames - see main.cpp's video-mode branch, which
// derives an expanded frame count from a --speed multiplier so a slower
// video spreads the same complete path over more frames (and more real
// time) instead of covering less of the path in the same number of frames.

#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Camera position result
struct CameraPosition {
	double lookfrom_x, lookfrom_y, lookfrom_z;
	double lookat_x, lookat_y, lookat_z;
	double vup_x, vup_y, vup_z;
};

// ============================================================================
// Circular Orbit Path
// Camera orbits around the lookAt point in a circle on the XZ plane.
// start_angle (radians) sets where frame 0 sits on that circle - passing the
// angle of a specific point (see get_camera_position()) makes frame 0 land
// exactly on that point instead of always starting at angle 0.
// ============================================================================
inline CameraPosition camera_path_orbit(int frame, int total_frames,
										double radius = 800.0,
										double center_x = 278.0,
										double center_y = 278.0,
										double center_z = 278.0,
										double height = 278.0,
										double start_angle = 0.0) {
	CameraPosition pos;

	// Compute angle (full 360° rotation over total_frames)
	double t = static_cast<double>(frame) / static_cast<double>(total_frames);
	double angle = start_angle + 2.0 * M_PI * t;

	// Circular motion in XZ plane
	pos.lookfrom_x = center_x + radius * std::cos(angle);
	pos.lookfrom_y = height;
	pos.lookfrom_z = center_z + radius * std::sin(angle);

	// Always look at center
	pos.lookat_x = center_x;
	pos.lookat_y = center_y;
	pos.lookat_z = center_z;

	// Up vector
	pos.vup_x = 0.0;
	pos.vup_y = 1.0;
	pos.vup_z = 0.0;

	return pos;
}

// ============================================================================
// Linear Path
// Camera translates in a straight line, starting exactly at lookfrom and
// sweeping sideways (perpendicular to the original lookfrom->lookat view
// direction) rather than straight toward/through lookat.
//
// The original version lerped all the way to lookfrom's point-reflection
// through lookat - i.e. it flew directly at the subject, through it (zero
// distance at the halfway frame, an undefined view direction), and out the
// other side. That is safe only for a scene with nothing solid along that
// line; for anything with an enclosing or extended ground/wall (e.g. scene
// A2's checker ground sphere), the camera ends up passing through or
// inside geometry partway along the flight, rendering pure black for every
// remaining frame - confirmed by inspecting real output frames from the
// "one-weekend-flyby" video preset (video_preset.h), where frame 1 already
// shows the camera grazing a sphere's surface and frames 2+ are solid
// black.
//
// A lateral sweep avoids this by construction: the offset is added
// perpendicular to the start->lookat direction, so distance from lookat is
// sqrt(original_distance^2 + offset^2) - never less than the original
// distance the scene's own recommended camera was already placed at.
// ============================================================================
inline CameraPosition camera_path_linear(int frame, int total_frames,
										  double lookfrom_x = 278.0, double lookfrom_y = 278.0, double lookfrom_z = -800.0,
										  double lookat_x = 278.0, double lookat_y = 278.0, double lookat_z = 278.0) {
	CameraPosition pos;

	// Guard against a 1-frame "video", which would otherwise divide by zero.
	double t = (total_frames > 1)
		? static_cast<double>(frame) / static_cast<double>(total_frames - 1)
		: 0.0;

	double dx = lookfrom_x - lookat_x;
	double dz = lookfrom_z - lookat_z;
	double radius_xz = std::sqrt(dx * dx + dz * dz);

	// Perpendicular to (dx, dz) in the XZ plane - a 90-degree rotation.
	// Falls back to a fixed axis when lookfrom sits directly above/below
	// lookat (radius_xz == 0), where "perpendicular to the view direction"
	// has no unique answer.
	double lateral_x, lateral_z;
	if (radius_xz > 1e-6) {
		lateral_x = -dz / radius_xz;
		lateral_z = dx / radius_xz;
	} else {
		lateral_x = 1.0;
		lateral_z = 0.0;
	}

	// Sweeps from 0 to one full radius_xz of lateral offset - t=0 lands
	// exactly on lookfrom, matching every other path's "frame 0 is the
	// scene's own recommended camera" guarantee.
	double offset = radius_xz * t;
	pos.lookfrom_x = lookfrom_x + lateral_x * offset;
	pos.lookfrom_y = lookfrom_y;
	pos.lookfrom_z = lookfrom_z + lateral_z * offset;

	// Fixed lookAt point
	pos.lookat_x = lookat_x;
	pos.lookat_y = lookat_y;
	pos.lookat_z = lookat_z;

	// Up vector
	pos.vup_x = 0.0;
	pos.vup_y = 1.0;
	pos.vup_z = 0.0;

	return pos;
}

// ============================================================================
// Figure-8 Path
// Camera moves in a figure-8 pattern on the XZ plane
// ============================================================================
inline CameraPosition camera_path_figure8(int frame, int total_frames,
										   double radius = 400.0,
										   double center_x = 278.0,
										   double center_y = 278.0,
										   double center_z = 278.0,
										   double height = 278.0) {
	CameraPosition pos;

	double t = static_cast<double>(frame) / static_cast<double>(total_frames);
	double angle = 2.0 * M_PI * t;

	// Lemniscate of Gerono (figure-8) parametric equations
	double scale = radius * 1.5;
	pos.lookfrom_x = center_x + scale * std::cos(angle);
	pos.lookfrom_y = height;
	pos.lookfrom_z = center_z + scale * std::sin(angle) * std::cos(angle);

	// Look at center
	pos.lookat_x = center_x;
	pos.lookat_y = center_y;
	pos.lookat_z = center_z;

	// Up vector
	pos.vup_x = 0.0;
	pos.vup_y = 1.0;
	pos.vup_z = 0.0;

	return pos;
}

// ============================================================================
// Spiral Path
// Camera spirals in while orbiting around the scene. start_angle works the
// same way as camera_path_orbit's.
// ============================================================================
inline CameraPosition camera_path_spiral(int frame, int total_frames,
										  double start_radius = 1000.0,
										  double end_radius = 400.0,
										  double center_x = 278.0,
										  double center_y = 278.0,
										  double center_z = 278.0,
										  double start_height = 500.0,
										  double end_height = 278.0,
										  double start_angle = 0.0) {
	CameraPosition pos;

	double t = static_cast<double>(frame) / static_cast<double>(total_frames);
	double angle = start_angle + 2.0 * M_PI * t * 2.0;  // Two full rotations

	// Interpolate radius and height
	double radius = start_radius + t * (end_radius - start_radius);
	double height = start_height + t * (end_height - start_height);

	pos.lookfrom_x = center_x + radius * std::cos(angle);
	pos.lookfrom_y = height;
	pos.lookfrom_z = center_z + radius * std::sin(angle);

	pos.lookat_x = center_x;
	pos.lookat_y = center_y;
	pos.lookat_z = center_z;

	pos.vup_x = 0.0;
	pos.vup_y = 1.0;
	pos.vup_z = 0.0;

	return pos;
}

// ============================================================================
// Get Camera Position by Path Name
// ============================================================================
// lookfrom_x/y/z and lookat_x/y/z are the scene's actual recommended camera
// (see cpu_interface.h's cpu_scene_recommended_camera(), which reads the
// same scene_registry.h CameraConfig the renderers themselves use) - passed
// through by main.cpp's video-mode branch so each path adapts to that
// scene's real coordinate scale AND starts from that exact camera position,
// instead of every video using the same Cornell-Box-scale orbit (radius 800
// around (278,278,278), starting at a fixed angle unrelated to any scene's
// actual default view) regardless of scene. The defaults below are Cornell
// Box's own registry values, so omitting these arguments still gives a
// sensible Cornell-scale path.
//
// orbit/spiral: decomposed into an XZ-plane radius, a height (the
// lookfrom's own Y), and a start_angle such that frame 0 lands exactly on
// the given lookfrom - the path then sweeps a full circle (or two, for
// spiral) from there. spiral's end_radius/end_height scale down from that
// starting radius/height toward the lookat point, preserving its original
// "zooms in as it spirals" character.
// linear: starts exactly at lookfrom and sweeps sideways from there,
// perpendicular to the lookfrom->lookat direction - see
// camera_path_linear's own comment for why it's a lateral pass rather than
// a fly-through straight at (and past) the subject.
// figure8: the lemniscate's shape can't be phase-aligned to an arbitrary
// start point with just an angle offset, so it's scaled (via the lookfrom-
// to-lookat distance) and centered on lookat, but doesn't start exactly at
// lookfrom the way the other three paths do.
inline CameraPosition get_camera_position(const std::string& path_type, int frame, int total_frames,
											double lookfrom_x = 278.0, double lookfrom_y = 278.0, double lookfrom_z = -800.0,
											double lookat_x = 278.0, double lookat_y = 278.0, double lookat_z = 278.0) {
	double dx = lookfrom_x - lookat_x;
	double dy = lookfrom_y - lookat_y;
	double dz = lookfrom_z - lookat_z;
	double radius_xz = std::sqrt(dx * dx + dz * dz);
	double start_angle = std::atan2(dz, dx);
	double dist3d = std::sqrt(dx * dx + dy * dy + dz * dz);
	double scale_factor = dist3d / 800.0;  // Cornell Box's own registry distance

	if (path_type == "orbit") {
		return camera_path_orbit(frame, total_frames, radius_xz, lookat_x, lookat_y, lookat_z, lookfrom_y, start_angle);
	} else if (path_type == "linear") {
		return camera_path_linear(frame, total_frames,
			lookfrom_x, lookfrom_y, lookfrom_z,
			lookat_x, lookat_y, lookat_z);
	} else if (path_type == "figure8") {
		return camera_path_figure8(frame, total_frames, 400.0 * scale_factor, lookat_x, lookat_y, lookat_z, lookfrom_y);
	} else if (path_type == "spiral") {
		return camera_path_spiral(frame, total_frames,
			radius_xz, radius_xz * 0.4,
			lookat_x, lookat_y, lookat_z,
			lookfrom_y, lookat_y,
			start_angle);
	} else {
		// Default to orbit
		return camera_path_orbit(frame, total_frames, radius_xz, lookat_x, lookat_y, lookat_z, lookfrom_y, start_angle);
	}
}

#endif // CAMERA_PATH_H
