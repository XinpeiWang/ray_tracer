#pragma once
// affine_transform_apply.h -- shared point/vector/normal/AABB transform math
// for a pbrt_scene::Matrix4, used by every CPU hittable that carries its own
// object<->world transform and intersects by carrying the ray into object
// space rather than baking the transform into world-space parameters (see
// transform_instance.h's own header comment for the full "why t survives"
// and "normals need the inverse transpose" reasoning - both apply verbatim
// here, this file just gives that math one shared home instead of a private
// copy per hittable).

#include <cmath>

#include "../shared/pbrt_scene.h"

#include "aabb.h"
#include "vec3.h"

namespace affine_transform {

inline point3 apply_point(const pbrt_scene::Matrix4& m, const point3& p) {
	return point3(
		m.m[0] * p.x() + m.m[1] * p.y() + m.m[2]  * p.z() + m.m[3],
		m.m[4] * p.x() + m.m[5] * p.y() + m.m[6]  * p.z() + m.m[7],
		m.m[8] * p.x() + m.m[9] * p.y() + m.m[10] * p.z() + m.m[11]);
}

// A direction: the translation column is deliberately not applied. Used for
// both ray directions and tangent vectors (dpdu/dpdv transform the same way
// a direction does).
inline vec3 apply_vector(const pbrt_scene::Matrix4& m, const vec3& v) {
	return vec3(
		m.m[0] * v.x() + m.m[1] * v.y() + m.m[2]  * v.z(),
		m.m[4] * v.x() + m.m[5] * v.y() + m.m[6]  * v.z(),
		m.m[8] * v.x() + m.m[9] * v.y() + m.m[10] * v.z());
}

// Object -> world for a normal is the TRANSPOSE of world -> object (a normal
// is not a direction; under a non-uniform scale, pushing it through the same
// matrix as a point tilts it off the surface), so this reads w2o column-wise
// rather than taking o2w row-wise.
inline vec3 apply_normal(const pbrt_scene::Matrix4& w2o, const vec3& n) {
	return vec3(
		w2o.m[0] * n.x() + w2o.m[4] * n.y() + w2o.m[8]  * n.z(),
		w2o.m[1] * n.x() + w2o.m[5] * n.y() + w2o.m[9]  * n.z(),
		w2o.m[2] * n.x() + w2o.m[6] * n.y() + w2o.m[10] * n.z());
}

// World-space AABB of an object-space box, transformed corner-by-corner (not
// the transform of the box's own min/max) - those differ the moment a
// rotation is involved, and transforming just the corners of the
// untransformed box produces a box that clips the geometry it is supposed to
// contain.
inline aabb transformed_bbox(const pbrt_scene::Matrix4& o2w,
							  double xlo, double xhi, double ylo, double yhi,
							  double zlo, double zhi) {
	point3 lo(0, 0, 0), hi(0, 0, 0);
	bool first = true;
	for (int corner = 0; corner < 8; ++corner) {
		const double x = (corner & 1) ? xhi : xlo;
		const double y = (corner & 2) ? yhi : ylo;
		const double z = (corner & 4) ? zhi : zlo;
		const point3 p = apply_point(o2w, point3(x, y, z));
		if (first) { lo = hi = p; first = false; continue; }
		lo = point3(std::fmin(lo.x(), p.x()), std::fmin(lo.y(), p.y()), std::fmin(lo.z(), p.z()));
		hi = point3(std::fmax(hi.x(), p.x()), std::fmax(hi.y(), p.y()), std::fmax(hi.z(), p.z()));
	}
	return aabb(lo, hi);
}

} // namespace affine_transform
