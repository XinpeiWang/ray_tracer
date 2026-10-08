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

// A 3x3 rotation matrix, row-major: m[row][col].
struct Mat3 {
	double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	V3 operator*(const V3& v) const {
		return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z, m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
	}
	Mat3 operator*(const Mat3& o) const {
		Mat3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
		return r;
	}
};

// A turn of `deg` degrees about a unit `axis` (right-handed: counter-clockwise looking back down the axis).
inline Mat3 axisAngle(const V3& axis, double deg) {
	const double a = deg * kPi / 180.0, c = std::cos(a), s = std::sin(a), t = 1.0 - c;
	const V3 u = normalize(axis);
	Mat3 r;
	r.m[0][0] = t * u.x * u.x + c;       r.m[0][1] = t * u.x * u.y - s * u.z; r.m[0][2] = t * u.x * u.z + s * u.y;
	r.m[1][0] = t * u.x * u.y + s * u.z; r.m[1][1] = t * u.y * u.y + c;       r.m[1][2] = t * u.y * u.z - s * u.x;
	r.m[2][0] = t * u.x * u.z - s * u.y; r.m[2][1] = t * u.y * u.z + s * u.x; r.m[2][2] = t * u.z * u.z + c;
	return r;
}

// What the scene file does with an object's three rotation angles: a turn about world X, then one about Y, then one about Z (p' = Rz * Ry * Rx * p).
inline Mat3 rotationXYZ(const V3& deg) { return axisAngle({0, 0, 1}, deg.z) * axisAngle({0, 1, 0}, deg.y) * axisAngle({1, 0, 0}, deg.x); }

// The three angles (degrees) that rotationXYZ turns into `r`. Straight up or down (|y| = 90) many triples give the same matrix; this picks z = 0 then.
inline V3 eulerXYZ(const Mat3& r) {
	const double sy = -r.m[2][0];
	const double y = std::asin(sy < -1.0 ? -1.0 : (sy > 1.0 ? 1.0 : sy));
	double x, z;
	if (std::abs(sy) < 0.999999) {
		x = std::atan2(r.m[2][1], r.m[2][2]);
		z = std::atan2(r.m[1][0], r.m[0][0]);
	} else {
		x = std::atan2(-r.m[1][2], r.m[1][1]);
		z = 0.0;
	}
	return {x * 180.0 / kPi, y * 180.0 / kPi, z * 180.0 / kPi};
}

// The angles an object ends up with after being turned `deg` degrees about a fixed WORLD axis (the ring being dragged), whatever its current angles are.
inline V3 turnAboutWorldAxis(const V3& currentDeg, const V3& worldAxis, double deg) { return eulerXYZ(axisAngle(worldAxis, deg) * rotationXYZ(currentDeg)); }

// The angle, in degrees and round `axis` from `refDir` (a unit vector in the plane), of the point where the ray meets the plane through `center`
// with normal `axis`. This is what turning an object by dragging a ring needs. False when the ray runs parallel to the plane or meets it behind the eye.
inline bool angleAround(const Ray& r, const V3& center, const V3& axis, const V3& refDir, double& deg) {
	V3 hit;
	if (!rayPlane(r, center, axis, hit)) return false;
	const V3 v = hit - center;
	const double x = dot(v, refDir), y = dot(v, cross(axis, refDir));
	if (x == 0.0 && y == 0.0) return false;
	deg = std::atan2(y, x) * 180.0 / kPi;
	return true;
}

// The difference between two angles in degrees, wrapped to (-180, 180]: turning from 170 to -170 is +20, not -340.
inline double angleDelta(double fromDeg, double toDeg) {
	double d = std::fmod(toDeg - fromDeg, 360.0);
	if (d > 180.0) d -= 360.0;
	if (d <= -180.0) d += 360.0;
	return d;
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
