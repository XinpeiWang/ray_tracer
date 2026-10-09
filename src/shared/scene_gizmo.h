// scene_gizmo.h - the parts of the Scene Builder's 3D tools that decide things, without Qt so they can be unit-tested: which dimension each Scale handle
// changes for each shape, and which axis handle a mouse press means (and which it does not: the middle of an object is for moving it freely).
#pragma once

#include <cmath>

#include "scene_document.h"

namespace scene_gizmo {

// Does the Scale handle on the object's own `axis` (0 X, 1 Y, 2 Z) change anything? A quad is flat: it has no height handle.
inline bool scaleHandleUsed(const scene_doc::Object& o, int axis) { return !(o.shape == scene_doc::ShapeKind::Quad && axis == 1); }

// The object with its `axis` handle dragged to `factor` times its length: a box's, wedge's or stairs' size on that axis, a pyramid's base or height, a
// cylinder's, cone's or capsule's height (Y) or radius (X, Z), a tube's height or both radii (the hole keeps its proportion), a quad's width or depth;
// a sphere, a disk, a dome, a torus and a mesh scale all round.
inline scene_doc::Object scaledObject(const scene_doc::Object& o, int axis, double factor) {
	using scene_doc::ShapeKind;
	scene_doc::Object r = o;
	switch (o.shape) {
		case ShapeKind::Sphere:
		case ShapeKind::Disk:
		case ShapeKind::Dome: r.radius = o.radius * factor; break;
		case ShapeKind::Mesh: r.meshScale = o.meshScale * factor; break;
		case ShapeKind::Cylinder:
		case ShapeKind::Cone:
		case ShapeKind::Capsule:
			if (axis == 1) r.height = o.height * factor;
			else r.radius = o.radius * factor;
			break;
		case ShapeKind::Box:
		case ShapeKind::Wedge:
		case ShapeKind::Stairs: (axis == 0 ? r.size.x : axis == 1 ? r.size.y : r.size.z) = (axis == 0 ? o.size.x : axis == 1 ? o.size.y : o.size.z) * factor; break;
		case ShapeKind::Pyramid:
			if (axis == 0) r.size.x = o.size.x * factor;
			else if (axis == 1) r.height = o.height * factor;
			else r.size.z = o.size.z * factor;
			break;
		case ShapeKind::Torus:
			r.radius = o.radius * factor;
			r.radius2 = o.radius2 * factor;
			break;
		case ShapeKind::Tube:
			if (axis == 1) r.height = o.height * factor;
			else { r.radius = o.radius * factor; r.radius2 = o.radius2 * factor; }
			break;
		case ShapeKind::Quad:
			if (axis == 0) r.size.x = o.size.x * factor;
			else if (axis == 2) r.size.z = o.size.z * factor;
			break;
	}
	return r;
}

struct P2 {
	double x = 0, y = 0;
};

inline double distanceToSegment(const P2& p, const P2& a, const P2& b) {
	const double abx = b.x - a.x, aby = b.y - a.y, len2 = abx * abx + aby * aby;
	double t = len2 > 0 ? ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2 : 0.0;
	t = t < 0 ? 0 : (t > 1 ? 1 : t);
	return std::hypot(p.x - (a.x + abx * t), p.y - (a.y + aby * t));
}

// Which axis arrow (a segment on screen from `base` to tips[a]) a press at `p` means, or -1. Only the outer part of an arrow, from `minAlong` (a fraction of
// its length) out to the tip, takes hold: the three arrows all start at the middle of the selected object and fan out over its body, so if the whole shaft
// were an arrow most of the object would be an axis drag, and "drag the object" would only ever move it along an axis. The inner part and the middle (within
// `deadPx` of `base`, whatever the arrow's length on screen) start the free move on the floor instead. Otherwise the nearest used arrow within `tolPx`.
inline int pickArrow(const P2& p, const P2& base, const P2 tips[3], const bool used[3], double deadPx, double tolPx, double minAlong = 0.45) {
	if (std::hypot(p.x - base.x, p.y - base.y) < deadPx) return -1;
	int best = -1;
	double bestD = tolPx;
	for (int a = 0; a < 3; ++a) {
		if (!used[a]) continue;
		const double abx = tips[a].x - base.x, aby = tips[a].y - base.y, len2 = abx * abx + aby * aby;
		if (len2 < 1.0) continue;   // an arrow that points at the camera has no length on screen: nothing to grab
		if (((p.x - base.x) * abx + (p.y - base.y) * aby) / len2 < minAlong) continue;   // the inner part of the shaft: the object itself
		const double d = distanceToSegment(p, base, tips[a]);
		if (d < bestD) { bestD = d; best = a; }
	}
	return best;
}

// Which Scale handle (a square at tips[a]) a press at `p` means, or -1: only the square itself, so the drag always starts where the handle's length is
// known (a grab part-way along the shaft would measure the scale from a tiny distance and turn a small move into a huge change).
inline int pickTip(const P2& p, const P2 tips[3], const bool used[3], double radiusPx) {
	int best = -1;
	double bestD = radiusPx;
	for (int a = 0; a < 3; ++a) {
		if (!used[a]) continue;
		const double d = std::hypot(p.x - tips[a].x, p.y - tips[a].y);
		if (d < bestD) { bestD = d; best = a; }
	}
	return best;
}

}  // namespace scene_gizmo
