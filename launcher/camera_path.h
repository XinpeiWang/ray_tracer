#ifndef CAMERA_PATH_H
#define CAMERA_PATH_H

// ============================================================================
// Camera Path Animation
// ============================================================================
// Provides parametric camera animation paths for video generation.
// Each function takes (frame_number, total_frames) and returns camera position.

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

	// Linear interpolation parameter
	double t = static_cast<double>(frame) / static_cast<double>(total_frames - 1);

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
inline CameraPosition get_camera_position(const std::string& path_type, int frame, int total_frames) {
	if (path_type == "orbit") {
		return camera_path_orbit(frame, total_frames);
	} else if (path_type == "linear") {
		return camera_path_linear(frame, total_frames);
	} else if (path_type == "figure8") {
		return camera_path_figure8(frame, total_frames);
	} else if (path_type == "spiral") {
		return camera_path_spiral(frame, total_frames);
	} else {
		// Default to orbit
		return camera_path_orbit(frame, total_frames);
	}
}

#endif // CAMERA_PATH_H
