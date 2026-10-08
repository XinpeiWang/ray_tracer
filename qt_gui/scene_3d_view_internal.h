#pragma once
// scene_3d_view_internal.h - what the parts of scene_3d_view*.cpp share and nothing else needs: the tool constants, small helpers, and the two
// private types the view draws with. Only the Scene3DView source files include this.

#include "scene_3d_view.h"
#include "scene_builder_common.h"

#include <QColor>
#include <QFont>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QPolygonF>

#include <cmath>
#include <cstdint>
#include <vector>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::ShapeKind;
using scene_view::V3;

using namespace scene_builder_ui;

namespace {

constexpr double kPi = scene_view::kPi;
const QColor kAxisColor[3] = {QColor(232, 72, 72), QColor(84, 200, 96), QColor(72, 124, 232)};
const V3 kAxisDir[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
// For the turning rings: a direction in each ring's plane to measure angles from.
const V3 kRingRef[3] = {{0, 1, 0}, {1, 0, 0}, {1, 0, 0}};
constexpr int kRingSegments = 64;
constexpr double kArrowDeadPx = 14.0;   // the middle of the selected object is for moving it freely, not for the arrows that start there
constexpr double kArrowPickPx = 9.0;
constexpr double kHandlePickPx = 13.0;
constexpr double kMeshRecheckMs = 1500.0;   // a mesh file's date is looked at this often, not at every repaint
constexpr double kMaxDropDistance = 1.5;    // a new object is dropped within this many camera distances of what the camera looks at

V3 toV3(const Float3 &f) { return {f.x, f.y, f.z}; }
Float3 toFloat3(const V3 &v) { return {v.x, v.y, v.z}; }

double snapQuarter(double v) { return std::round(v / 0.25) * 0.25; }

scene_gizmo::P2 toP2(const QPointF &p) { return {p.x(), p.y()}; }

// The point on a ring: its centre, the ring's axis (0 X, 1 Y, 2 Z), a radius and an angle from the reference direction.
V3 ringAt(const V3 &centre, int axis, double radius, double deg) {
	const V3 ref = kRingRef[axis], side = scene_view::cross(kAxisDir[axis], ref);
	const double a = deg * kPi / 180.0;
	return centre + (ref * std::cos(a) + side * std::sin(a)) * radius;
}

// FNV-1a over bytes: the signature of what shapes the faces.
void mix(std::uint64_t &h, const void *data, std::size_t n) {
	const unsigned char *p = static_cast<const unsigned char *>(data);
	for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
}
template <class T> void mix(std::uint64_t &h, const T &v) { mix(h, &v, sizeof(T)); }

}  // namespace

// One flat polygon of an object, in world space, with what it takes to draw and pick it.
struct Scene3DView::Face {
	std::vector<V3> world;
	QColor color;
	int owner = -1;          // the object's index
	bool bigFlat = false;    // a floor-like panel: drawn first so things standing on it are not hidden by it
	bool flat = false;       // a single big flat panel (quad, disk): sorted by its farthest point, since its centre says little about what is in front of it
	double depth = 0;        // centroid depth (camera space)
	double farDepth = 0;     // depth of its farthest vertex
	QPolygonF screen;        // after clipping and projection; empty when behind the camera
};

// What a paint needs: the painter, the camera, the colours.
struct Scene3DView::Ctx {
	QPainter *p;
	scene_view::View v;
	QColor text, accent;
	QFont small, normal;

	// A line between two world points, clipped at the camera.
	void line(const V3 &a, const V3 &b, const QPen &pen) const {
		V3 va = v.toView(a), vb = v.toView(b);
		if (va.z < v.nearPlane && vb.z < v.nearPlane) return;
		if (va.z < v.nearPlane) va = va + (vb - va) * ((v.nearPlane - va.z) / (vb.z - va.z));
		else if (vb.z < v.nearPlane) vb = vb + (va - vb) * ((v.nearPlane - vb.z) / (va.z - vb.z));
		double x0, y0, x1, y1;
		v.toPixel(va, x0, y0);
		v.toPixel(vb, x1, y1);
		p->setPen(pen);
		p->drawLine(QPointF(x0, y0), QPointF(x1, y1));
	}
	bool pixel(const V3 &w, QPointF &out) const {
		double sx, sy;
		if (!v.project(w, sx, sy)) return false;
		out = QPointF(sx, sy);
		return true;
	}
};
