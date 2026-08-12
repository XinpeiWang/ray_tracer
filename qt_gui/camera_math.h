#ifndef CAMERA_MATH_H
#define CAMERA_MATH_H

#include <cmath>

// ============================================================================
// Camera positioning arithmetic
// ============================================================================
// The GUI lets you move the camera two ways: by typing a distance from the
// scene's look-at point, and by picking a named preset ("Right Wall", "Front",
// ...). Both are small pieces of vector arithmetic that used to sit inline in
// MainWindow's slots, tangled with spin-box plumbing, where nothing could test
// them - including the degenerate cases that actually matter (camera sitting
// exactly on the look-at point, a scene whose recommended distance is zero).
//
// Qt-free for the same reason as render_output_parser.h: the Qt install here
// is MinGW-only while the gtest binary is MSVC, so anything the tests need to
// reach cannot depend on Qt.
//
// See tests/unit/camera_math_tests.cpp.
// ============================================================================
namespace camera_math {

struct Vec3 {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
};

// Distance from the look-at point to the camera.
inline double distanceFromTarget(const Vec3 &camera, const Vec3 &lookAt) {
	const double dx = camera.x - lookAt.x;
	const double dy = camera.y - lookAt.y;
	const double dz = camera.z - lookAt.z;
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Moves the camera to `distance` from lookAt while preserving its current
// viewing direction.
//
// The degenerate case is the interesting one: if the camera sits exactly on
// the look-at point there is no direction to preserve, and normalising would
// divide by zero. It falls back to looking down -Z, which matches the
// launcher's own generic default direction, rather than producing NaNs that
// would propagate into the spin boxes and then onto the renderer's command
// line.
inline Vec3 repositionAtDistance(const Vec3 &camera, const Vec3 &lookAt, double distance) {
	double dx = camera.x - lookAt.x;
	double dy = camera.y - lookAt.y;
	double dz = camera.z - lookAt.z;
	double current = std::sqrt(dx * dx + dy * dy + dz * dz);

	if (current < 1e-6) {
		dx = 0.0;
		dy = 0.0;
		dz = -1.0;
		current = 1.0;
	}

	const double scale = distance / current;
	return Vec3{lookAt.x + dx * scale,
				lookAt.y + dy * scale,
				lookAt.z + dz * scale};
}

// Turns a preset's stored direction*ratio vector into an absolute position for
// the active scene.
//
// Presets deliberately store a direction rather than a position: an absolute
// one would be Cornell Box's literal (500,278,278) for every scene, which is
// wildly outside the geometry of most of them (scene 1's spheres sit within
// roughly +-15 units of the origin). Scaling by the scene's own recommended
// camera distance makes "Right Wall" land somewhere sensible everywhere.
inline Vec3 presetPosition(const Vec3 &direction, const Vec3 &lookAt, double sceneDistance) {
	return Vec3{lookAt.x + direction.x * sceneDistance,
				lookAt.y + direction.y * sceneDistance,
				lookAt.z + direction.z * sceneDistance};
}

} // namespace camera_math

#endif // CAMERA_MATH_H
