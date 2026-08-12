#pragma once
// pbrt_quadify.h -- recovers parallelogram quads from pairs of triangles.
//
// WHY THIS EXISTS
// ---------------
// The GPU renderer samples area lights as spheres or quads and nothing else
// (see optix_renderer.cpp's light list, which carries one is-it-a-sphere flag
// per light). pbrt has no quad shape: an area light is written as a
// `trianglemesh` with an AreaLightSource attached, which is two triangles
// sharing a diagonal. Handed to the GPU as triangles, such a light is
// geometry that glows when a ray happens to hit it but that next-event
// estimation cannot aim at - so the render is either black or converges at a
// rate that makes it useless.
//
// Rejoining those triangles is therefore not a nicety, it is what makes a
// loaded pbrt scene renderable on the GPU at all.
//
// SCOPE, DELIBERATELY NARROW
// --------------------------
// Only EMISSIVE triangles need this: ordinary geometry is perfectly happy as
// triangles, and merging it would be work with no beneficiary. Triangles that
// cannot be paired into a true parallelogram are handed back untouched rather
// than approximated - a light that is a general quadrilateral, a triangle fan,
// or a genuine single triangle is a real thing to encounter, and silently
// pretending it is a parallelogram would put its light in the wrong place.
//
// Qt-free and CUDA-free on purpose, so the geometry can be tested without a
// GPU present. The conversion to QuadData happens at the GPU boundary.

#include <cmath>
#include <cstddef>
#include <vector>

#include "pbrt_flatten.h"

namespace pbrt_quadify {

// A parallelogram: origin corner plus two edge vectors, matching the GPU's
// QuadData and our CPU quad alike.
struct Quad {
	double Q[3] = {0, 0, 0};
	double u[3] = {0, 0, 0};
	double v[3] = {0, 0, 0};
	int material = -1;
	int areaLight = -1;
};

struct Result {
	std::vector<Quad> quads;
	// Triangles that were not merged, in their original order. Callers should
	// still add these to the scene - they are real geometry - but a caller
	// that needs quads for light sampling should say so about any emissive
	// ones left here rather than assume the list is empty.
	std::vector<pbrt_flatten::Triangle> leftover;
};

namespace detail {

inline bool samePoint(const double *a, const double *b, double tol) {
	for (int i = 0; i < 3; ++i)
		if (std::fabs(a[i] - b[i]) > tol) return false;
	return true;
}

// A scale-relative tolerance. A fixed epsilon is wrong here: pbrt scenes are
// authored at whatever scale suited their modeller, and vertices that agree to
// within a millionth at unit scale differ by a whole unit at 1e6. The
// magnitude of the coordinates involved is the only sane reference.
inline double toleranceFor(const pbrt_flatten::Triangle &t) {
	double biggest = 1.0;
	for (int i = 0; i < 9; ++i) biggest = std::max(biggest, std::fabs(t.v[i]));
	return biggest * 1e-9;
}

} // namespace detail

// Merges pairs of emissive triangles that share an edge and form a
// parallelogram. Non-emissive triangles are passed through untouched.
//
// The pairing is between ADJACENT triangles in the list, which is not a
// shortcut: pbrt writes a mesh's index list as consecutive faces, so the two
// halves of a light quad are always neighbours. Searching all pairs would be
// quadratic in the triangle count for scenes with millions of them, to catch
// an ordering that pbrt does not produce.
inline Result quadify(const std::vector<pbrt_flatten::Triangle> &triangles) {
	using namespace detail;
	Result out;

	std::size_t i = 0;
	while (i < triangles.size()) {
		const pbrt_flatten::Triangle &a = triangles[i];

		// Only lights benefit, and only a neighbour can be the other half.
		if (a.areaLight < 0 || i + 1 >= triangles.size()) {
			out.leftover.push_back(a);
			++i;
			continue;
		}
		const pbrt_flatten::Triangle &b = triangles[i + 1];
		if (b.areaLight != a.areaLight || b.material != a.material) {
			out.leftover.push_back(a);
			++i;
			continue;
		}

		const double tol = toleranceFor(a);

		// Find the two vertices they share. Those form the diagonal; the two
		// they do not share are the opposite corners of the quad.
		int sharedA[3], sharedB[3], nShared = 0;
		int uniqueA = -1, uniqueB = -1;
		for (int ia = 0; ia < 3; ++ia) {
			int match = -1;
			for (int ib = 0; ib < 3; ++ib)
				if (samePoint(&a.v[ia * 3], &b.v[ib * 3], tol)) { match = ib; break; }
			if (match >= 0) {
				if (nShared < 3) { sharedA[nShared] = ia; sharedB[nShared] = match; }
				++nShared;
			} else {
				uniqueA = ia;
			}
		}
		if (nShared != 2 || uniqueA < 0) {
			out.leftover.push_back(a);
			++i;
			continue;
		}
		for (int ib = 0; ib < 3; ++ib)
			if (ib != sharedB[0] && ib != sharedB[1]) uniqueB = ib;

		// Corners in order around the quad: shared, uniqueA, shared, uniqueB.
		const double *c0 = &a.v[sharedA[0] * 3];
		const double *c1 = &a.v[uniqueA * 3];
		const double *c2 = &a.v[sharedA[1] * 3];
		const double *c3 = &b.v[uniqueB * 3];

		// A parallelogram satisfies c0 + (c1-c0) + (c3-c0) == c2. Checking it
		// rather than assuming is the difference between merging a light and
		// moving one: a quadrilateral that is not a parallelogram cannot be
		// represented by Q/u/v at all, and forcing it would put emission where
		// the scene never put any.
		double eu[3], ev[3], predicted[3];
		for (int k = 0; k < 3; ++k) {
			eu[k] = c1[k] - c0[k];
			ev[k] = c3[k] - c0[k];
			predicted[k] = c0[k] + eu[k] + ev[k];
		}
		if (!samePoint(predicted, c2, tol * 1e3)) {
			out.leftover.push_back(a);
			++i;
			continue;
		}

		Quad q;
		for (int k = 0; k < 3; ++k) { q.Q[k] = c0[k]; q.u[k] = eu[k]; q.v[k] = ev[k]; }
		q.material = a.material;
		q.areaLight = a.areaLight;
		out.quads.push_back(q);
		i += 2;                       // both triangles consumed
	}
	return out;
}

// How many emissive triangles survived as triangles. Non-zero means the scene
// has lights the GPU cannot aim at, which is worth telling the user about
// rather than letting them wonder why a render is dark.
inline std::size_t unmergedEmissiveCount(const Result &r) {
	std::size_t n = 0;
	for (const pbrt_flatten::Triangle &t : r.leftover)
		if (t.areaLight >= 0) ++n;
	return n;
}

} // namespace pbrt_quadify
