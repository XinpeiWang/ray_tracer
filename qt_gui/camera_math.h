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

inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

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

// A pinhole camera's ray-generation basis, EXACTLY mirroring the GPU's own
// camera_params[12] output layout (build_pinhole_camera_params(),
// scene_builder.cpp): a ray for screen fraction (s,t) in [0,1]x[0,1] is
// origin + k*(lowerLeftCorner + s*horizontal + t*vertical - origin) for
// some k > 0. Live Preview's rt_realtime_render_frame() exposes exactly
// these 4 vectors (GpuCameraParams::origin/lower_left_corner/horizontal/
// vertical - the same fields build_scene() already fills for every camera
// kind it supports) rather than a vertical-FOV-and-aspect pair: vfov/aspect
// are never actually available to ANY caller of rt_realtime_render_frame() -
// every scene case bakes its own vfov into a hardcoded literal deep inside
// build_scene(), with nothing surfacing it - so re-deriving a basis from a
// guessed vfov was never an option; reading back the exact basis the GPU
// already computed is both simpler and exact, not an approximation.
struct CameraBasis {
	Vec3 origin;
	Vec3 lowerLeftCorner;
	Vec3 horizontal;
	Vec3 vertical;
};

// Where a world-space point lands on a pinhole camera's screen - used by
// Live Preview's temporal reprojection (qt_gui/realtime_preview_session.cpp)
// to find where a surface point visible in the CURRENT frame would have
// appeared in a PREVIOUS frame's camera, so that frame's already-accumulated
// sample can be reused instead of starting over from noise.
struct ScreenProjection {
	double s = 0.0;  // [0, 1], left to right
	double t = 0.0;  // [0, 1], BOTTOM to top - matches build_pinhole_camera_
					 // params()'s own lower-left-origin convention (see
					 // projectToScreen()'s own comment on converting this to
					 // a pixel row, which needs a top/bottom flip)
	bool inFront = false;  // false if the point is behind (or exactly on)
							// the camera's image plane - s/t are
							// meaningless in that case
};

// Projects worldPoint into the screen space of the pinhole camera described
// by basis (see CameraBasis's own comment).
//
// Derivation: a point at parameter (s,t) lies along
// direction(s,t) = lowerLeftCorner + s*horizontal + t*vertical - origin
//                = (s-0.5)*horizontal + (t-0.5)*vertical - wScaled
// where wScaled = origin - lowerLeftCorner - 0.5*horizontal - 0.5*vertical
// (algebraically equal to focus_dist*w in build_pinhole_camera_params()'s
// own notation, without needing to know focus_dist separately). Solving
// worldPoint - origin = k*(s-0.5)*horizontal + k*(t-0.5)*vertical - k*wScaled
// for (a,b,c) = (k*(s-0.5), k*(t-0.5), -k) is a plain 3x3 linear solve
// (Cramer's rule via scalar triple products) - NOT simplified to a
// dot-product-over-squared-length shortcut, because horizontal/vertical/
// wScaled are only mutually orthogonal for a CENTERED viewport. A scene
// using pbrt's Camera "perspective" "float screenwindow" with an off-center
// window (build_pinhole_camera_params()'s own center_shift_u/v,
// scene_builder.cpp) bakes that shift into lowerLeftCorner, which tilts
// wScaled out of alignment with horizontal/vertical - an earlier version of
// this function assumed orthogonality and silently computed the wrong s/t
// for exactly that case. This general solve is exact regardless, and
// reduces to that same simpler shortcut's result whenever the basis IS
// orthogonal (verified in camera_math_tests.cpp), so one code path handles
// both rather than needing to detect which one applies. k <= 0 means
// worldPoint is behind (or exactly on) the camera's image plane, mirroring
// how a negative/zero ray parameter means "no intersection in the
// direction the ray was actually cast" everywhere else in this codebase's
// own ray-tracing math.
inline ScreenProjection projectToScreen(const Vec3 &worldPoint, const CameraBasis &basis) {
	const Vec3 toPoint = worldPoint - basis.origin;
	const Vec3 wScaled = basis.origin - basis.lowerLeftCorner
						  - basis.horizontal * 0.5 - basis.vertical * 0.5;
	// Cramer's rule for M*(a,b,c) = toPoint where M's columns are
	// (horizontal, vertical, wScaled): x_i = det(M with column i replaced
	// by toPoint) / det(M), each det expressed as a scalar triple product.
	const Vec3 vCrossW = cross(basis.vertical, wScaled);
	const double denom = dot(basis.horizontal, vCrossW);
	if (std::fabs(denom) < 1e-18) return ScreenProjection{0.0, 0.0, false};
	const double a = dot(toPoint, vCrossW) / denom;
	const double b = dot(basis.horizontal, cross(toPoint, wScaled)) / denom;
	const double c = dot(basis.horizontal, cross(basis.vertical, toPoint)) / denom;
	if (c >= 0.0) return ScreenProjection{0.0, 0.0, false};
	return ScreenProjection{0.5 - a / c, 0.5 - b / c, true};
}

} // namespace camera_math

#endif // CAMERA_MATH_H
