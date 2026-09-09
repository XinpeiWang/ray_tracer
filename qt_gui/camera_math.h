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

// Basic vector arithmetic - added for Live Preview's free-fly WASD
// translation (applyTranslateDelta(), mainwindow_tabs_render.cpp), which
// needs a real forward/right/up basis (cross product) rather than the
// hand-written per-axis dx/dy/dz arithmetic every function above already
// used - fine for a single distance/direction formula, awkward for a
// three-vector basis computation with a cross product in the middle.
inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(const Vec3 &v, double s) { return Vec3{v.x * s, v.y * s, v.z * s}; }

inline double length(const Vec3 &v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// Same degenerate-case convention as repositionAtDistance()/cartesianToOrbit()
// above: a zero-length input has no direction to normalize, so this returns
// the zero vector rather than propagating NaN.
inline Vec3 normalized(const Vec3 &v) {
	const double len = length(v);
	if (len < 1e-9) return Vec3{0.0, 0.0, 0.0};
	return v * (1.0 / len);
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
	return Vec3{a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x};
}

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

// Spherical coordinates of a camera around a look-at point: distance
// (radius), horizontal angle (azimuth, radians, measured from +Z rotating
// toward +X), and vertical angle (elevation, radians - positive is above
// the look-at point's horizontal plane). Used by Live Preview's sub-tab's
// mouse-drag orbit control (MainWindow::onLivePreviewOrbitDragged()/
// onLivePreviewZoomRequested()): dragging changes azimuth/elevation, the
// wheel changes radius, and orbitToCartesian() below converts the result
// back into an actual camera position on every step.
struct OrbitCoordinates {
	double radius = 0.0;
	double azimuth = 0.0;
	double elevation = 0.0;
};

// Decomposes an absolute camera position into spherical coordinates around
// lookAt - the inverse of orbitToCartesian() below (round-trips exactly,
// away from the poles - see camera_math_tests.cpp).
//
// Same degenerate case as repositionAtDistance(): a camera sitting exactly
// on the look-at point has no direction to decompose at all (both azimuth
// and elevation would be feeding a zero-length vector into atan2/asin) -
// falls back to a fixed radius/angle pair rather than propagating NaN,
// matching repositionAtDistance()'s own fallback shape.
inline OrbitCoordinates cartesianToOrbit(const Vec3 &camera, const Vec3 &lookAt) {
	const double dx = camera.x - lookAt.x;
	const double dy = camera.y - lookAt.y;
	const double dz = camera.z - lookAt.z;
	const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
	if (radius < 1e-6) {
		return OrbitCoordinates{1.0, 0.0, 0.0};
	}
	double sinElevation = dy / radius;
	if (sinElevation > 1.0) sinElevation = 1.0;
	if (sinElevation < -1.0) sinElevation = -1.0;
	return OrbitCoordinates{radius, std::atan2(dx, dz), std::asin(sinElevation)};
}

// The inverse of cartesianToOrbit() above: turns spherical coordinates
// around lookAt back into an absolute camera position.
inline Vec3 orbitToCartesian(const OrbitCoordinates &orbit, const Vec3 &lookAt) {
	const double cosElevation = std::cos(orbit.elevation);
	return Vec3{lookAt.x + orbit.radius * cosElevation * std::sin(orbit.azimuth),
				lookAt.y + orbit.radius * std::sin(orbit.elevation),
				lookAt.z + orbit.radius * cosElevation * std::cos(orbit.azimuth)};
}

} // namespace camera_math

#endif // CAMERA_MATH_H
