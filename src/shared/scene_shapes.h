// Triangle meshes for the Scene Builder's ready-made shapes that pbrt has no shape of its own for: pyramid, wedge (ramp), stairs, torus, capsule, dome
// and tube. Everything is in the object's own space, centred on its position, Y up (like the box and cylinder), and std-only so the unit tests and the
// pbrt writer share it with the GUI's previews.
//
// A mesh has outward-facing triangles (counter-clockwise seen from outside, the pbrt convention) and one normal per vertex: flat faces get their own
// vertices (sharp edges), curved ones share vertices (smooth shading), so the same mesh renders correctly on every backend.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace scene_doc {

struct ShapeMesh {
	std::vector<double> P;    // 3 per vertex
	std::vector<double> N;    // 3 per vertex, unit
	std::vector<double> UV;   // 2 per vertex
	std::vector<int> indices; // 3 per triangle
	std::size_t vertexCount() const { return P.size() / 3; }
	std::size_t triangleCount() const { return indices.size() / 3; }
};

namespace shapes_detail {

struct V { double x, y, z; };
inline V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double length(V a) { return std::sqrt(dot(a, a)); }
inline V unit(V a) { const double l = length(a); return l > 0 ? V{a.x / l, a.y / l, a.z / l} : V{0, 1, 0}; }

inline int addVertex(ShapeMesh& m, V p, V n, double u, double v) {
	m.P.insert(m.P.end(), {p.x, p.y, p.z});
	m.N.insert(m.N.end(), {n.x, n.y, n.z});
	m.UV.insert(m.UV.end(), {u, v});
	return static_cast<int>(m.vertexCount()) - 1;
}

// One triangle over existing vertices, wound so its geometric normal agrees with `outward`; a sliver with no area is dropped.
inline void addTriangle(ShapeMesh& m, int a, int b, int c, V outward) {
	const V pa{m.P[a * 3], m.P[a * 3 + 1], m.P[a * 3 + 2]}, pb{m.P[b * 3], m.P[b * 3 + 1], m.P[b * 3 + 2]}, pc{m.P[c * 3], m.P[c * 3 + 1], m.P[c * 3 + 2]};
	const V g = cross(sub(pb, pa), sub(pc, pa));
	if (length(g) < 1e-12) return;
	if (dot(g, outward) >= 0) m.indices.insert(m.indices.end(), {a, b, c});
	else m.indices.insert(m.indices.end(), {a, c, b});
}

// A flat polygon (3 or 4 corners, any order of winding) with a single normal: `outward` says which way it faces.
inline void addFlat(ShapeMesh& m, const std::vector<V>& corners, V outward) {
	const V n = unit(outward);
	std::vector<int> ids;
	static const double uvs[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
	for (std::size_t i = 0; i < corners.size(); ++i) ids.push_back(addVertex(m, corners[i], n, uvs[i][0], uvs[i][1]));
	for (std::size_t i = 1; i + 1 < ids.size(); ++i) addTriangle(m, ids[0], ids[i], ids[i + 1], n);
}

struct ProfilePoint { double r, y, nr, ny; };   // a point of a revolved outline, with its surface normal in the (r, y) plane

// Revolves `profile` (listed from the bottom up, or round a closed outline) about the Y axis. Two consecutive points at one place with different normals
// make a sharp edge.
inline void lathe(ShapeMesh& m, const std::vector<ProfilePoint>& profile, int segments) {
	const double pi = 3.14159265358979323846;
	std::vector<double> along(profile.size(), 0.0);
	for (std::size_t j = 1; j < profile.size(); ++j) along[j] = along[j - 1] + std::hypot(profile[j].r - profile[j - 1].r, profile[j].y - profile[j - 1].y);
	const double total = std::max(along.empty() ? 0.0 : along.back(), 1e-9);
	const std::size_t first = m.vertexCount();
	const std::size_t rows = profile.size();
	for (int i = 0; i <= segments; ++i) {
		const double phi = 2 * pi * i / segments, c = std::cos(phi), s = std::sin(phi);
		for (std::size_t j = 0; j < rows; ++j) {
			const ProfilePoint& p = profile[j];
			addVertex(m, {p.r * c, p.y, p.r * s}, unit({p.nr * c, p.ny, p.nr * s}), static_cast<double>(i) / segments, along[j] / total);
		}
	}
	for (int i = 0; i < segments; ++i)
		for (std::size_t j = 0; j + 1 < rows; ++j) {
			const int a = static_cast<int>(first + i * rows + j), b = static_cast<int>(first + (i + 1) * rows + j);
			const int c = static_cast<int>(first + (i + 1) * rows + j + 1), d = static_cast<int>(first + i * rows + j + 1);
			const auto outward = [&](int k) { return V{m.N[k * 3], m.N[k * 3 + 1], m.N[k * 3 + 2]}; };
			const V o1{outward(a).x + outward(b).x + outward(c).x, outward(a).y + outward(b).y + outward(c).y, outward(a).z + outward(b).z + outward(c).z};
			const V o2{outward(a).x + outward(c).x + outward(d).x, outward(a).y + outward(c).y + outward(d).y, outward(a).z + outward(c).z + outward(d).z};
			addTriangle(m, a, b, c, o1);
			addTriangle(m, a, c, d, o2);
		}
}

}  // namespace shapes_detail

// Base `w` (X) by `d` (Z), apex over the middle of it; `h` tall.
inline ShapeMesh pyramidMesh(double w, double h, double d) {
	using namespace shapes_detail;
	ShapeMesh m;
	const double x = w / 2, z = d / 2, lo = -h / 2, hi = h / 2;
	const V b0{-x, lo, -z}, b1{x, lo, -z}, b2{x, lo, z}, b3{-x, lo, z}, apex{0, hi, 0};
	addFlat(m, {b3, b2, b1, b0}, {0, -1, 0});
	// Each side leans out by half its base and up by the height: the normal of a slope that rises `h` over a run of `half`.
	addFlat(m, {b0, b1, apex}, {0, z, -h});
	addFlat(m, {b1, b2, apex}, {h, x, 0});
	addFlat(m, {b2, b3, apex}, {0, z, h});
	addFlat(m, {b3, b0, apex}, {-h, x, 0});
	return m;
}

// A ramp: the box w x h x d cut along its diagonal, low at the front (-Z) and `h` high at the back (+Z).
inline ShapeMesh wedgeMesh(double w, double h, double d) {
	using namespace shapes_detail;
	ShapeMesh m;
	const double x = w / 2, y = h / 2, z = d / 2;
	const V fl{-x, -y, -z}, fr{x, -y, -z}, bl{-x, -y, z}, br{x, -y, z}, tl{-x, y, z}, tr{x, y, z};
	addFlat(m, {fl, fr, br, bl}, {0, -1, 0});
	addFlat(m, {bl, br, tr, tl}, {0, 0, 1});
	addFlat(m, {fl, fr, tr, tl}, {0, d, -h});
	addFlat(m, {fl, bl, tl}, {-1, 0, 0});
	addFlat(m, {fr, br, tr}, {1, 0, 0});
	return m;
}

// `steps` steps climbing from the front (-Z) to the back (+Z), `w` wide, `h` high in all and `d` deep.
inline ShapeMesh stairsMesh(double w, double h, double d, int steps) {
	using namespace shapes_detail;
	ShapeMesh m;
	steps = std::max(1, steps);
	const double x = w / 2, lo = -h / 2;
	for (int i = 0; i < steps; ++i) {
		const double z0 = -d / 2 + d * i / steps, z1 = -d / 2 + d * (i + 1) / steps;
		const double top = lo + h * (i + 1) / steps, prev = lo + h * i / steps;
		addFlat(m, {{-x, top, z0}, {x, top, z0}, {x, top, z1}, {-x, top, z1}}, {0, 1, 0});                  // the tread
		addFlat(m, {{-x, prev, z0}, {x, prev, z0}, {x, top, z0}, {-x, top, z0}}, {0, 0, -1});               // the riser
		addFlat(m, {{x, lo, z0}, {x, lo, z1}, {x, top, z1}, {x, top, z0}}, {1, 0, 0});                      // the sides
		addFlat(m, {{-x, lo, z0}, {-x, lo, z1}, {-x, top, z1}, {-x, top, z0}}, {-1, 0, 0});
	}
	addFlat(m, {{-x, lo, -d / 2}, {x, lo, -d / 2}, {x, lo, d / 2}, {-x, lo, d / 2}}, {0, -1, 0});
	addFlat(m, {{-x, lo, d / 2}, {x, lo, d / 2}, {x, lo + h, d / 2}, {-x, lo + h, d / 2}}, {0, 0, 1});
	return m;
}

// A ring (doughnut) lying flat: `major` from its centre to the middle of the tube, `minor` the tube's radius.
inline ShapeMesh torusMesh(double major, double minor) {
	using namespace shapes_detail;
	const double pi = 3.14159265358979323846;
	ShapeMesh m;
	std::vector<ProfilePoint> profile;
	const int around = 20;
	for (int k = 0; k <= around; ++k) {
		const double a = 2 * pi * k / around;
		profile.push_back({major + minor * std::cos(a), minor * std::sin(a), std::cos(a), std::sin(a)});
	}
	lathe(m, profile, 40);
	return m;
}

// A cylinder with rounded ends: `radius` across, `height` in all (never less than the two ends need).
inline ShapeMesh capsuleMesh(double radius, double height) {
	using namespace shapes_detail;
	const double pi = 3.14159265358979323846;
	ShapeMesh m;
	const double straight = std::max(0.0, height - 2 * radius) / 2;
	std::vector<ProfilePoint> profile;
	const int arc = 10;
	for (int k = 0; k <= arc; ++k) {   // the bottom cap, from the pole round to the side
		const double a = -pi / 2 + (pi / 2) * k / arc;
		profile.push_back({radius * std::cos(a), -straight + radius * std::sin(a), std::cos(a), std::sin(a)});
	}
	for (int k = 0; k <= arc; ++k) {   // the top cap, from the side up to the pole
		const double a = (pi / 2) * k / arc;
		profile.push_back({radius * std::cos(a), straight + radius * std::sin(a), std::cos(a), std::sin(a)});
	}
	lathe(m, profile, 32);
	return m;
}

// Half a sphere with a flat bottom, `radius` across; the whole thing is `radius` high, centred on its position.
inline ShapeMesh domeMesh(double radius) {
	using namespace shapes_detail;
	const double pi = 3.14159265358979323846;
	ShapeMesh m;
	const double lo = -radius / 2;
	std::vector<ProfilePoint> profile;
	profile.push_back({0, lo, 0, -1});
	profile.push_back({radius, lo, 0, -1});
	const int arc = 12;
	for (int k = 0; k <= arc; ++k) {
		const double a = (pi / 2) * k / arc;
		profile.push_back({radius * std::cos(a), lo + radius * std::sin(a), std::cos(a), std::sin(a)});
	}
	lathe(m, profile, 32);
	return m;
}

// A pipe: a cylinder `height` tall with a round hole through it; `radius` outside, `inner` inside.
inline ShapeMesh tubeMesh(double radius, double inner, double height) {
	using namespace shapes_detail;
	ShapeMesh m;
	const double hy = height / 2;
	std::vector<ProfilePoint> profile = {
		{radius, -hy, 1, 0}, {radius, hy, 1, 0},     // the outside
		{radius, hy, 0, 1},  {inner, hy, 0, 1},      // the top rim
		{inner, hy, -1, 0},  {inner, -hy, -1, 0},    // the inside
		{inner, -hy, 0, -1}, {radius, -hy, 0, -1},   // the bottom rim
	};
	lathe(m, profile, 36);
	return m;
}

}  // namespace scene_doc
