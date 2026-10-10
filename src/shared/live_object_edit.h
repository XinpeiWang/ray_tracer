#pragma once
// live_object_edit.h -- what the Live Preview needs to pick and move the objects of a pbrt scene: which shape a point on a surface belongs to, and a shape moved by
// an offset. Both work on the flattened scene (pbrt_flatten::FlatScene, world space) through FlatScene::shapeRanges, so a backend applies the same edit to the
// scene it is about to build, and the same pick reads the picture back. Standard library only.
//
// An "object" here is what a person would call one thing in the scene: all the top-level `Shape`s of one AttributeBegin/End block (a box written as six quads is
// one object), or a `Shape` outside any block on its own. Only shapes that became triangles, spheres, disks or cylinders count. (A shape turned into another kind,
// an instance, or geometry the loader tessellates later is not an object: it cannot be picked or moved.)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pbrt_flatten.h"

namespace live_objects {

// The objects of a scene: each one is the list of its indices into FlatScene::shapeRanges, in the order the objects first appear in the file.
using ObjectList = std::vector<std::vector<std::size_t>>;

inline ObjectList objectsOf(const pbrt_flatten::FlatScene& scene) {
	ObjectList objects;
	std::vector<std::pair<int, std::size_t>> byGroup;   // block number -> index into objects
	for (std::size_t i = 0; i < scene.shapeRanges.size(); ++i) {
		const int group = scene.shapeRanges[i].group;
		if (group >= 0) {
			bool found = false;
			for (const auto& g : byGroup)
				if (g.first == group) { objects[g.second].push_back(i); found = true; break; }
			if (found) continue;
			byGroup.emplace_back(group, objects.size());
		}
		objects.push_back({i});
	}
	return objects;
}

namespace detail {

// Moves one range of primitives by d, in world space. Triangles move by their vertices (normals are unchanged), spheres by their centre (and their end-of-motion
// centre), disks and cylinders by the translation of their object-to-world matrix.
inline void translateRange(pbrt_flatten::FlatScene& scene, const pbrt_flatten::ShapeRange& r, const double d[3]) {
	for (std::size_t i = r.triBegin; i < r.triEnd && i < scene.triangles.size(); ++i)
		for (int v = 0; v < 3; ++v)
			for (int a = 0; a < 3; ++a) scene.triangles[i].v[v * 3 + a] += d[a];
	for (std::size_t i = r.sphereBegin; i < r.sphereEnd && i < scene.spheres.size(); ++i)
		for (int a = 0; a < 3; ++a) {
			scene.spheres[i].center[a] += d[a];
			scene.spheres[i].center1[a] += d[a];
			scene.spheres[i].xform[3 + 4 * a] += d[a];
		}
	for (std::size_t i = r.diskBegin; i < r.diskEnd && i < scene.disks.size(); ++i)
		for (int a = 0; a < 3; ++a) {
			scene.disks[i].xform[3 + 4 * a] += d[a];
			scene.disks[i].xformEnd[3 + 4 * a] += d[a];
		}
	for (std::size_t i = r.cylinderBegin; i < r.cylinderEnd && i < scene.cylinders.size(); ++i)
		for (int a = 0; a < 3; ++a) {
			scene.cylinders[i].xform[3 + 4 * a] += d[a];
			scene.cylinders[i].xformEnd[3 + 4 * a] += d[a];
		}
}

}  // namespace detail

// Moves object `index` of `objects` by d, in world space. False for an index out of range.
inline bool translateObject(pbrt_flatten::FlatScene& scene, const ObjectList& objects, std::size_t index, const double d[3]) {
	if (index >= objects.size()) return false;
	for (std::size_t r : objects[index])
		if (r < scene.shapeRanges.size()) detail::translateRange(scene, scene.shapeRanges[r], d);
	return true;
}

// One pickable object: its box and, for triangles and spheres, the exact surface to measure a distance to.
struct PickShape {
	int object = -1;                   // index into the ObjectList
	std::string label;                 // "sphere", "trianglemesh", "3 shapes"
	double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
	std::vector<float> triangles;      // 9 floats per triangle; empty when the object has too many to keep (then only its box is used)
	std::vector<float> spheres;        // 4 floats per sphere: centre and radius
	bool exact = false;                // the surface above is complete: distance is measured to it, not to the box
};

struct PickIndex {
	std::vector<PickShape> shapes;     // the objects that have geometry
};

namespace detail {

inline void grow(PickShape& s, const double p[3], bool first) {
	for (int a = 0; a < 3; ++a) {
		s.lo[a] = first ? p[a] : std::min(s.lo[a], p[a]);
		s.hi[a] = first ? p[a] : std::max(s.hi[a], p[a]);
	}
}

// The distance from p to the box (0 inside).
inline double boxDistance(const PickShape& s, const double p[3]) {
	double sq = 0;
	for (int a = 0; a < 3; ++a) {
		const double d = std::max({s.lo[a] - p[a], 0.0, p[a] - s.hi[a]});
		sq += d * d;
	}
	return std::sqrt(sq);
}

// The distance from p to the triangle abc (the closest point, after Ericson's "Real-Time Collision Detection").
inline double triangleDistance(const double p[3], const float* t) {
	auto sub = [](const double* x, const double* y, double* o) { for (int i = 0; i < 3; ++i) o[i] = x[i] - y[i]; };
	auto dot = [](const double* x, const double* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
	const double a[3] = {t[0], t[1], t[2]}, b[3] = {t[3], t[4], t[5]}, c[3] = {t[6], t[7], t[8]};
	double ab[3], ac[3], ap[3];
	sub(b, a, ab); sub(c, a, ac); sub(p, a, ap);
	const double d1 = dot(ab, ap), d2 = dot(ac, ap);
	double q[3];
	auto fromPoint = [&](double u, double v, double w) { for (int i = 0; i < 3; ++i) q[i] = u * a[i] + v * b[i] + w * c[i]; };
	if (d1 <= 0 && d2 <= 0) fromPoint(1, 0, 0);
	else {
		double bp[3]; sub(p, b, bp);
		const double d3 = dot(ab, bp), d4 = dot(ac, bp);
		double cp[3]; sub(p, c, cp);
		const double d5 = dot(ab, cp), d6 = dot(ac, cp);
		const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
		if (d3 >= 0 && d4 <= d3) fromPoint(0, 1, 0);
		else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const double v = d1 / (d1 - d3); fromPoint(1 - v, v, 0); }
		else if (d6 >= 0 && d5 <= d6) fromPoint(0, 0, 1);
		else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const double w = d2 / (d2 - d6); fromPoint(1 - w, 0, w); }
		else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); fromPoint(0, 1 - w, w); }
		else {
			const double denom = 1.0 / (va + vb + vc);
			const double v = vb * denom, w = vc * denom;
			fromPoint(1 - v - w, v, w);
		}
	}
	double diff[3]; sub(p, q, diff);
	return std::sqrt(dot(diff, diff));
}

}  // namespace detail

// Indexes every object of the scene. An object with more than `maxTrianglesKept` triangles keeps only its box (a picked point is then matched to the box).
inline PickIndex buildPickIndex(const pbrt_flatten::FlatScene& scene, const ObjectList& objects, std::size_t maxTrianglesKept = 200000) {
	PickIndex index;
	for (std::size_t o = 0; o < objects.size(); ++o) {
		PickShape s;
		s.object = static_cast<int>(o);
		s.label = objects[o].size() == 1 ? scene.shapeRanges[objects[o][0]].type : std::to_string(objects[o].size()) + " shapes";
		std::size_t tris = 0;
		for (std::size_t ri : objects[o]) tris += scene.shapeRanges[ri].triEnd - scene.shapeRanges[ri].triBegin;
		const bool keep = tris <= maxTrianglesKept;
		bool first = true, boxOnly = false;
		auto addTransformedBox = [&](const double* m, double rad, double z0, double z1) {
			for (int corner = 0; corner < 8; ++corner) {
				const double x = (corner & 1) ? rad : -rad, y = (corner & 2) ? rad : -rad, z = (corner & 4) ? z1 : z0;
				const double p[3] = {m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7], m[8] * x + m[9] * y + m[10] * z + m[11]};
				detail::grow(s, p, first);
				first = false;
			}
			boxOnly = true;
		};
		for (std::size_t ri : objects[o]) {
			const pbrt_flatten::ShapeRange& r = scene.shapeRanges[ri];
			for (std::size_t i = r.triBegin; i < r.triEnd && i < scene.triangles.size(); ++i)
				for (int v = 0; v < 3; ++v) {
					const double p[3] = {scene.triangles[i].v[v * 3], scene.triangles[i].v[v * 3 + 1], scene.triangles[i].v[v * 3 + 2]};
					detail::grow(s, p, first);
					first = false;
					if (keep) for (int a = 0; a < 3; ++a) s.triangles.push_back(static_cast<float>(p[a]));
				}
			for (std::size_t i = r.sphereBegin; i < r.sphereEnd && i < scene.spheres.size(); ++i) {
				const pbrt_flatten::Sphere& sp = scene.spheres[i];
				const double rad = std::fabs(sp.radius);
				for (int corner = 0; corner < 8; ++corner) {
					const double p[3] = {sp.center[0] + ((corner & 1) ? rad : -rad), sp.center[1] + ((corner & 2) ? rad : -rad), sp.center[2] + ((corner & 4) ? rad : -rad)};
					detail::grow(s, p, first);
					first = false;
				}
				for (int a = 0; a < 3; ++a) s.spheres.push_back(static_cast<float>(sp.center[a]));
				s.spheres.push_back(static_cast<float>(rad));
			}
			for (std::size_t i = r.diskBegin; i < r.diskEnd && i < scene.disks.size(); ++i)
				addTransformedBox(scene.disks[i].xform, scene.disks[i].radius, scene.disks[i].height, scene.disks[i].height);
			for (std::size_t i = r.cylinderBegin; i < r.cylinderEnd && i < scene.cylinders.size(); ++i)
				addTransformedBox(scene.cylinders[i].xform, scene.cylinders[i].radius, scene.cylinders[i].zMin, scene.cylinders[i].zMax);
		}
		if (first) continue;   // no usable geometry
		s.exact = keep && !boxOnly;
		index.shapes.push_back(std::move(s));
	}
	return index;
}

struct PickResult {
	int object = -1;                   // index into the ObjectList; -1 = nothing within tolerance
	int slot = -1;                     // index into PickIndex::shapes
	double distance = 0;
};

// The object whose surface is nearest to p, if that is within `tolerance` (a picked point lies on its surface up to the picture's rounding).
// Among objects equally near (a ball resting on a floor), the one with the smaller box wins.
inline PickResult pick(const PickIndex& index, const double p[3], double tolerance) {
	PickResult best;
	best.distance = std::numeric_limits<double>::infinity();
	double bestSize = std::numeric_limits<double>::infinity();
	for (std::size_t i = 0; i < index.shapes.size(); ++i) {
		const PickShape& s = index.shapes[i];
		double d = detail::boxDistance(s, p);
		if (d > tolerance * 4.0) continue;
		if (s.exact) {
			double m = std::numeric_limits<double>::infinity();
			for (std::size_t t = 0; t + 8 < s.triangles.size(); t += 9) m = std::min(m, detail::triangleDistance(p, &s.triangles[t]));
			for (std::size_t q = 0; q + 3 < s.spheres.size(); q += 4) {
				const double dx = p[0] - s.spheres[q], dy = p[1] - s.spheres[q + 1], dz = p[2] - s.spheres[q + 2];
				m = std::min(m, std::fabs(std::sqrt(dx * dx + dy * dy + dz * dz) - s.spheres[q + 3]));
			}
			d = m;
		}
		const double size = std::sqrt(std::pow(s.hi[0] - s.lo[0], 2) + std::pow(s.hi[1] - s.lo[1], 2) + std::pow(s.hi[2] - s.lo[2], 2));   // the box's diagonal: a flat floor is not "smaller" than a ball
		if (d < best.distance - 1e-12 || (std::fabs(d - best.distance) <= 1e-12 && size < bestSize)) {
			best.distance = d;
			bestSize = size;
			best.slot = static_cast<int>(i);
			best.object = s.object;
		}
	}
	if (best.distance > tolerance) { best.object = -1; best.slot = -1; }
	return best;
}

}  // namespace live_objects
