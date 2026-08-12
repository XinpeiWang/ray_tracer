#pragma once
// pbrt_flatten.h -- turns a parsed pbrt scene into world-space geometry.
//
// pbrt describes geometry in local coordinates under a current transformation
// matrix. Neither backend can consume that directly: the CPU side has only
// `translate` and `rotate_y` hittables (hittable.h) - no general 4x4 - and the
// GPU side builds flat AABB primitive arrays. So the CTM is BAKED into vertex
// positions here, once, instead of being applied per ray at render time.
//
// That is the cheaper answer as well as the simpler one. A transform wrapper
// costs a matrix multiply on every ray-object test forever; baking costs one
// multiply per vertex at load.
//
// This sits between pbrt_scene.h (text -> description) and the backends
// (geometry -> renderer), and like both of its neighbours it is Qt-free and
// free of renderer types, so the MSVC test binary can reach it.
//
// WHAT IT APPROXIMATES, AND SAYS SO
// ---------------------------------
// Baking works exactly for triangles: any affine transform maps a triangle to
// a triangle. It does NOT work for spheres. A non-uniform scale turns a sphere
// into an ellipsoid, which this cannot represent, so such a sphere is emitted
// with its largest axis and a warning. Silently emitting a round sphere where
// the scene wanted a squashed one is the kind of difference nobody spots
// against a reference image.

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "pbrt_scene.h"

namespace pbrt_flatten {

struct Triangle {
	double v[9];              // three vertices, world space, xyz each
	int material = -1;        // index into Scene::materials, -1 = pbrt default
	int areaLight = -1;       // index into Scene::areaLights, -1 = not emissive
};

struct Sphere {
	double center[3] = {0, 0, 0};
	double radius = 1.0;
	int material = -1;
	int areaLight = -1;
};

struct FlatScene {
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<pbrt_scene::Warning> warnings;

	bool empty() const { return triangles.empty() && spheres.empty(); }
};

// Supplies a PLY mesh's positions and indices for `Shape "plymesh"`. Same
// callback shape, and for the same reason, as pbrt_scene::FileResolver: keeps
// this a pure function and lets the caller decide how a path resolves.
// Returning false means the mesh could not be read.
using MeshResolver = std::function<bool(const std::string &path,
										std::vector<float> &positions,
										std::vector<int> &indices)>;

namespace detail {

inline void transformPoint(const pbrt_scene::Matrix4 &m,
						   double x, double y, double z, double *out) {
	out[0] = m.m[0] * x + m.m[1] * y + m.m[2]  * z + m.m[3];
	out[1] = m.m[4] * x + m.m[5] * y + m.m[6]  * z + m.m[7];
	out[2] = m.m[8] * x + m.m[9] * y + m.m[10] * z + m.m[11];
}

// The three basis vectors' lengths are the scale along each axis. Comparing
// them is how a non-uniform scale is detected without decomposing the matrix
// properly - enough to know a sphere cannot survive it.
inline void axisScales(const pbrt_scene::Matrix4 &m, double *out) {
	for (int c = 0; c < 3; ++c) {
		const double a = m.m[0 + c], b = m.m[4 + c], d = m.m[8 + c];
		out[c] = std::sqrt(a * a + b * b + d * d);
	}
}

} // namespace detail

inline FlatScene flatten(const pbrt_scene::Scene &scene,
						 const MeshResolver &meshes = {}) {
	using namespace detail;
	FlatScene out;
	out.warnings = scene.warnings;   // carry the parser's own warnings through

	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	for (const pbrt_scene::ShapeDecl &shape : scene.shapes) {
		if (shape.type == "sphere") {
			const double r = shape.params.getFloat("radius", 1.0);
			Sphere s;
			transformPoint(shape.xform, 0.0, 0.0, 0.0, s.center);

			double sc[3];
			axisScales(shape.xform, sc);
			const double lo = std::fmin(sc[0], std::fmin(sc[1], sc[2]));
			const double hi = std::fmax(sc[0], std::fmax(sc[1], sc[2]));
			if (hi - lo > 1e-6 * std::fmax(1.0, hi)) {
				warn("a sphere carries a non-uniform scale and would be an "
					 "ellipsoid; emitted with its largest radius instead");
			}
			s.radius = r * hi;
			s.material = shape.materialIndex;
			s.areaLight = shape.areaLightIndex;
			out.spheres.push_back(s);
			continue;
		}

		if (shape.type == "trianglemesh" || shape.type == "plymesh") {
			std::vector<double> P;
			std::vector<int> indices;

			if (shape.type == "trianglemesh") {
				const pbrt_scene::Param *pp = shape.params.find("P");
				const pbrt_scene::Param *pi = shape.params.find("indices");
				if (!pp || !pi) {
					warn("a trianglemesh is missing its P or indices parameter; skipped");
					continue;
				}
				P = pp->numbers;
				indices.reserve(pi->numbers.size());
				for (double d : pi->numbers) indices.push_back(static_cast<int>(d));
			} else {
				const std::string file = shape.params.getString("filename", "");
				if (file.empty()) { warn("a plymesh has no filename; skipped"); continue; }
				if (!meshes) {
					warn("plymesh '" + file + "' skipped: no mesh resolver supplied");
					continue;
				}
				std::vector<float> pos;
				if (!meshes(file, pos, indices)) {
					warn("plymesh '" + file + "' could not be read; skipped");
					continue;
				}
				P.assign(pos.begin(), pos.end());
			}

			const std::size_t vertexCount = P.size() / 3;
			if (indices.size() % 3 != 0)
				warn("a mesh has an index count that is not a multiple of 3; "
					 "the trailing indices are ignored");

			// Transform once per vertex rather than once per index: a shared
			// vertex is referenced by several triangles, and transforming it
			// repeatedly is both slower and a source of tiny inconsistencies
			// between the same point on adjacent faces.
			std::vector<double> world(vertexCount * 3);
			for (std::size_t v = 0; v < vertexCount; ++v)
				transformPoint(shape.xform, P[v * 3], P[v * 3 + 1], P[v * 3 + 2],
							   &world[v * 3]);

			bool reportedRange = false;
			for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
				const int a = indices[i], b = indices[i + 1], c = indices[i + 2];
				if (a < 0 || b < 0 || c < 0
					|| static_cast<std::size_t>(a) >= vertexCount
					|| static_cast<std::size_t>(b) >= vertexCount
					|| static_cast<std::size_t>(c) >= vertexCount) {
					if (!reportedRange) {
						warn("a mesh has face indices outside its vertex list; "
							 "those faces are dropped");
						reportedRange = true;
					}
					continue;
				}
				Triangle t;
				for (int k = 0; k < 3; ++k) {
					t.v[0 + k] = world[static_cast<std::size_t>(a) * 3 + k];
					t.v[3 + k] = world[static_cast<std::size_t>(b) * 3 + k];
					t.v[6 + k] = world[static_cast<std::size_t>(c) * 3 + k];
				}
				t.material = shape.materialIndex;
				t.areaLight = shape.areaLightIndex;
				out.triangles.push_back(t);
			}
			continue;
		}

		// disk, cylinder, curve, bilinearmesh, ... Recognised as geometry we
		// cannot build, which is worth saying: the scene will render with a
		// hole in it rather than looking subtly wrong.
		warn("shape '" + shape.type + "' is not supported; skipped");
	}

	return out;
}

} // namespace pbrt_flatten
