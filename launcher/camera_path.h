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
// Camera orbits around the lookAt point in a circle on the XZ plane
// ============================================================================
inline CameraPosition camera_path_orbit(int frame, int total_frames,
										double radius = 800.0,
										double center_x = 278.0,
										double center_y = 278.0,
										double center_z = 278.0,
										double height = 278.0) {
	CameraPosition pos;

	// Compute angle (full 360° rotation over total_frames)
	double t = static_cast<double>(frame) / static_cast<double>(total_frames);
	double angle = 2.0 * M_PI * t;

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
// Camera moves linearly from start to end position
// ============================================================================
inline CameraPosition camera_path_linear(int frame, int total_frames,
										  double start_x = 278.0, double start_y = 278.0, double start_z = -800.0,
										  double end_x = 278.0, double end_y = 278.0, double end_z = 800.0,
										  double lookat_x = 278.0, double lookat_y = 278.0, double lookat_z = 278.0) {
	CameraPosition pos;

	// Linear interpolation parameter (guard against a 1-frame "video", which
	// would otherwise divide by zero and produce a NaN camera position)
	double t = (total_frames > 1)
		? static_cast<double>(frame) / static_cast<double>(total_frames - 1)
		: 0.0;

	// Lerp camera position
	pos.lookfrom_x = start_x + t * (end_x - start_x);
	pos.lookfrom_y = start_y + t * (end_y - start_y);
	pos.lookfrom_z = start_z + t * (end_z - start_z);

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
// Camera spirals in while orbiting around the scene
// ============================================================================
inline CameraPosition camera_path_spiral(int frame, int total_frames,
										  double start_radius = 1000.0,
										  double end_radius = 400.0,
										  double center_x = 278.0,
										  double center_y = 278.0,
										  double center_z = 278.0,
										  double start_height = 500.0,
										  double end_height = 278.0) {
	CameraPosition pos;

	double t = static_cast<double>(frame) / static_cast<double>(total_frames);
	double angle = 2.0 * M_PI * t * 2.0;  // Two full rotations

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
// center_x/y/z and scale let the caller adapt these paths to a scene's
// actual coordinate scale, rather than every video using the same
// Cornell-Box-scale defaults (radius 800 around (278,278,278)) regardless
// of how large or small that scene's own geometry is - see main.cpp's
// video-mode branch, which derives these from
// cpu_scene_recommended_camera() (cpu_interface.h). The defaults below
// exactly reproduce each path's original Cornell-scale behavior, since
// Cornell Box's own registry distance-from-lookat happens to already be
// 800 - so passing scale=800/center=(278,278,278) (or omitting them
// entirely) is behaviorally identical to before this parameter existed.
//
// scale_factor rescales orbit/figure8/spiral's radii proportionally,
// preserving their relative sizes to each other (figure8 stays half of
// orbit's implied radius, spiral's start/end stay in the same ratio) -
// only linear's start/end don't reduce to their *exact* prior hardcoded
// values at the Cornell defaults (those were two independently hand-picked
// points, not derivable from a center+scale formula to begin with), though
// they're still a reasonable, similarly-scaled sweep.
inline CameraPosition get_camera_position(const std::string& path_type, int frame, int total_frames,
											double center_x = 278.0, double center_y = 278.0, double center_z = 278.0,
											double scale = 800.0) {
	double scale_factor = scale / 800.0;

	if (path_type == "orbit") {
		return camera_path_orbit(frame, total_frames, scale, center_x, center_y, center_z, center_y);
	} else if (path_type == "linear") {
		return camera_path_linear(frame, total_frames,
			center_x, center_y, center_z - scale,
			center_x, center_y, center_z + scale,
			center_x, center_y, center_z);
	} else if (path_type == "figure8") {
		return camera_path_figure8(frame, total_frames, 400.0 * scale_factor, center_x, center_y, center_z, center_y);
	} else if (path_type == "spiral") {
		return camera_path_spiral(frame, total_frames,
			1000.0 * scale_factor, 400.0 * scale_factor,
			center_x, center_y, center_z,
			center_y + 222.0 * scale_factor, center_y);
	} else {
		// Default to orbit
		return camera_path_orbit(frame, total_frames, scale, center_x, center_y, center_z, center_y);
	}
}

#endif // CAMERA_PATH_H
