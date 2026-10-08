// scene_view_math.h - the geometry behind the Scene Builder's 3D view, with no Qt in it so it can be unit-tested: an orbit camera, perspective
// projection, the ray under a pixel, ray/plane and ray/line queries (what dragging an object or an axis arrow needs) and near-plane clipping.
//
// Coordinates are the renderer's: X to the right, Y up, Z towards a viewer looking down -Z (right-handed).
#pragma once

#include <cmath>
#include <vector>

namespace scene_view {

constexpr double kPi = 3.14159265358979323846;

struct V3 {
	double x = 0, y = 0, z = 0;
};
inline V3 operator+(const V3& a, const V3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(const V3& a, const V3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(const V3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double length(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(const V3& a) {
	const double l = length(a);
	return l > 1e-300 ? a * (1.0 / l) : V3{0, 0, 1};
}

struct Ray {
	V3 origin, dir;  // dir is a unit vector
};

// A camera that orbits `target`: yaw turns it round the vertical axis (0 looks from +Z towards -Z), pitch lifts it above the horizon.
struct OrbitCamera {
	V3 target{0, 1, 0};
	double distance = 10;
	double yawDeg = 30;
	double pitchDeg = 25;   // kept inside +-89 so "up" is never parallel to the view direction
	double fovDeg = 40;     // vertical

	V3 eye() const {
		const double y = yawDeg * kPi / 180.0, p = pitchDeg * kPi / 180.0;
		return target + V3{std::cos(p) * std::sin(y), std::sin(p), std::cos(p) * std::cos(y)} * distance;
	}
	V3 forward() const { return normalize(target - eye()); }
	V3 right() const { return normalize(cross(forward(), V3{0, 1, 0})); }
	V3 up() const { return cross(right(), forward()); }

	void orbit(double dxDeg, double dyDeg) {
		yawDeg += dxDeg;
		pitchDeg += dyDeg;
		if (pitchDeg > 89) pitchDeg = 89;
		if (pitchDeg < -89) pitchDeg = -89;
	}
	// Moves the target sideways and up/down on the screen by (dx, dy) world units in the camera's own axes.
	void pan(double dx, double dy) { target = target + right() * dx + up() * dy; }
	void dolly(double factor) {
		distance *= factor;
		if (distance < 0.2) distance = 0.2;
		if (distance > 2000) distance = 2000;
	}
};

// A camera seen through a width x height widget.
struct View {
	OrbitCamera cam;
	double width = 640, height = 480;
	double nearPlane = 0.05;

	double focal() const { return (height / 2.0) / std::tan(cam.fovDeg * kPi / 360.0); }  // pixels

	// World -> camera space: x right, y up, z DEPTH (positive in front of the camera).
	V3 toView(const V3& p) const {
		const V3 v = p - cam.eye();
		return {dot(v, cam.right()), dot(v, cam.up()), dot(v, cam.forward())};
	}
	// Camera space -> pixels (y down). The point must be in front of the camera.
	void toPixel(const V3& v, double& sx, double& sy) const {
		const double f = focal();
		sx = width / 2.0 + v.x / v.z * f;
		sy = height / 2.0 - v.y / v.z * f;
	}
	// False when the point is behind (or too near) the camera.
	bool project(const V3& p, double& sx, double& sy, double* depth = nullptr) const {
		const V3 v = toView(p);
		if (v.z < nearPlane) return false;
		toPixel(v, sx, sy);
		if (depth) *depth = v.z;
		return true;
	}
	// The ray from the eye through a pixel.
	Ray ray(double sx, double sy) const {
		const double f = focal();
		const V3 d = cam.forward() + cam.right() * ((sx - width / 2.0) / f) - cam.up() * ((sy - height / 2.0) / f);
		return {cam.eye(), normalize(d)};
	}
	// World units per pixel at a given depth.
	double unitsPerPixel(double depth) const { return depth / focal(); }
};

// Where a ray meets a plane (False when it is parallel to it or the hit is behind the ray's origin).
inline bool rayPlane(const Ray& r, const V3& pointOnPlane, const V3& normal, V3& hit) {
	const double denom = dot(r.dir, normal);
	if (std::abs(denom) < 1e-12) return false;
	const double t = dot(pointOnPlane - r.origin, normal) / denom;
	if (t < 0) return false;
	hit = r.origin + r.dir * t;
	return true;
}

// The t for which p + t*dir is the point of that line closest to the ray (False when the ray runs parallel to the line). This is what
// dragging along an axis arrow needs: the mouse ray rarely meets the axis, but it has a closest point on it.
inline bool closestOnLine(const Ray& r, const V3& p, const V3& dir, double& t) {
	const V3 w = p - r.origin;
	const double a = dot(dir, dir), b = dot(dir, r.dir), c = dot(r.dir, r.dir);
	const double d = dot(dir, w), e = dot(r.dir, w);
	const double denom = a * c - b * b;
	if (std::abs(denom) < 1e-12 * a * c) return false;
	t = (b * e - c * d) / denom;
	return true;
}

// Clips a polygon given in camera space to the depth >= nearPlane half space (Sutherland-Hodgman), so a face that passes behind the camera is cut
// instead of being projected with a flipped sign.
inline std::vector<V3> clipNear(const std::vector<V3>& poly, double nearPlane) {
	std::vector<V3> out;
	const size_t n = poly.size();
	for (size_t i = 0; i < n; ++i) {
		const V3& a = poly[i];
		const V3& b = poly[(i + 1) % n];
		const bool ain = a.z >= nearPlane, bin = b.z >= nearPlane;
		if (ain) out.push_back(a);
		if (ain != bin) {
			const double t = (nearPlane - a.z) / (b.z - a.z);
			out.push_back(a + (b - a) * t);
		}
	}
	return out;
}

}  // namespace scene_view
