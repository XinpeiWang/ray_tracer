// scene_3d_view.cpp - see scene_3d_view.h.
#include "scene_3d_view.h"

#include "scene_builder_common.h"

#include <QFileInfo>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

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

V3 toV3(const Float3 &f) { return {f.x, f.y, f.z}; }
Float3 toFloat3(const V3 &v) { return {v.x, v.y, v.z}; }

double snapQuarter(double v) { return std::round(v / 0.25) * 0.25; }

double distanceToSegment(const QPointF &p, const QPointF &a, const QPointF &b) {
	const QPointF ab = b - a;
	const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
	double t = len2 > 0 ? ((p.x() - a.x()) * ab.x() + (p.y() - a.y()) * ab.y()) / len2 : 0.0;
	t = std::clamp(t, 0.0, 1.0);
	const QPointF q = a + ab * t;
	return std::hypot(p.x() - q.x(), p.y() - q.y());
}

// The point on a ring: its centre, the ring's axis (0 X, 1 Y, 2 Z), a radius and an angle from the reference direction.
V3 ringAt(const V3 &centre, int axis, double radius, double deg) {
	const V3 ref = kRingRef[axis], side = scene_view::cross(kAxisDir[axis], ref);
	const double a = deg * kPi / 180.0;
	return centre + (ref * std::cos(a) + side * std::sin(a)) * radius;
}

}  // namespace

// One flat polygon of an object, in world space, with what it takes to draw and pick it.
struct Scene3DView::Face {
	std::vector<V3> world;
	QColor color;
	int owner = -1;          // the object's index
	bool bigFlat = false;    // a floor-like panel: drawn first so things standing on it are not hidden by it
	double depth = 0;        // centroid depth (camera space), for sorting
	QPolygonF screen;        // after clipping and projection; empty when behind the camera
};

Scene3DView::Scene3DView(QWidget *parent) : QWidget(parent) {
	setMouseTracking(false);
	setFocusPolicy(Qt::ClickFocus);
	setMinimumSize(280, 220);
	setAutoFillBackground(false);
}

scene_view::View Scene3DView::view() const {
	scene_view::View v;
	v.cam = m_cam;
	v.width = std::max(1, width());
	v.height = std::max(1, height());
	return v;
}

const Float3 *Scene3DView::handle(const BuilderSelection &s, int which) const { return builderHandle(m_doc, s, which); }

const Object *Scene3DView::selectedObject() const {
	if (!m_doc || m_sel.kind != BuilderSelection::Kind::Object || m_sel.index < 0 || m_sel.index >= static_cast<int>(m_doc->objects.size())) return nullptr;
	return &m_doc->objects[m_sel.index];
}

Scene3DView::GizmoMode Scene3DView::effectiveGizmo() const { return selectedObject() ? m_gizmo : GizmoMode::Move; }

void Scene3DView::setGizmoMode(GizmoMode m) {
	if (m == m_gizmo) return;
	m_gizmo = m;
	emit gizmoModeChanged(static_cast<int>(m));
	update();
}

void Scene3DView::keyPressEvent(QKeyEvent *e) {
	switch (e->key()) {
		case Qt::Key_W: setGizmoMode(GizmoMode::Move); break;
		case Qt::Key_E: setGizmoMode(GizmoMode::Rotate); break;
		case Qt::Key_R: setGizmoMode(GizmoMode::Scale); break;
		default: QWidget::keyPressEvent(e); return;
	}
	e->accept();
}

void Scene3DView::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	if (!m_userView) frameAll();
}

// The object's mesh file, read once (and again if the file changes); the cache keeps a failure too, so a bad file is not retried at every repaint.
const mesh_preview::MeshPreview *Scene3DView::meshPreview(const std::string &path) const {
	if (path.empty()) return nullptr;
	const QFileInfo info(QString::fromStdString(path));
	const qint64 modified = info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1, size = info.size();
	auto it = m_meshes.find(path);
	if (it == m_meshes.end() || it->second.modified != modified || it->second.size != size) {
		CachedMesh c;
		c.modified = modified;
		c.size = size;
		c.preview = mesh_preview::load(path);
		it = m_meshes.insert_or_assign(path, std::move(c)).first;
	}
	return it->second.preview.ok ? &it->second.preview : nullptr;
}

void Scene3DView::frameAll() {
	m_userView = false;
	if (!m_doc) return;
	V3 lo{1e18, 1e18, 1e18}, hi{-1e18, -1e18, -1e18};
	auto add = [&](const Float3 &p, double pad) {
		lo = {std::min(lo.x, p.x - pad), std::min(lo.y, p.y - pad), std::min(lo.z, p.z - pad)};
		hi = {std::max(hi.x, p.x + pad), std::max(hi.y, p.y + pad), std::max(hi.z, p.z + pad)};
	};
	add(m_doc->camera.position, 0.3);
	for (const Object &o : m_doc->objects) {
		double pad = o.shape == ShapeKind::Quad ? std::max(o.size.x, o.size.z) * 0.5 : std::max({o.radius, o.size.x, o.size.y, o.size.z, o.height});
		if (o.shape == ShapeKind::Mesh)
			if (const mesh_preview::MeshPreview *m = meshPreview(o.meshFile))
				pad = std::max({m->hi[0] - m->lo[0], m->hi[1] - m->lo[1], m->hi[2] - m->lo[2]}) * o.meshScale * 0.7;
		add(o.position, std::min(pad, 4.0));  // a big floor would make everything else tiny: frame a bounded part of it
	}
	for (const Light &l : m_doc->lights)
		if (l.kind != LightKind::Infinite) add(l.position, 0.3);
	if (lo.x > hi.x) { lo = {-4, 0, -4}; hi = {4, 3, 4}; }
	const V3 centre = (lo + hi) * 0.5;
	const double radius = std::max(1.5, scene_view::length(hi - lo) * 0.5);
	m_cam.target = centre;
	m_cam.distance = radius / std::sin(m_cam.fovDeg * kPi / 360.0) * 0.55;
	update();
}

Float3 Scene3DView::centerInWorld() const {
	const scene_view::View v = view();
	V3 hit;
	if (scene_view::rayPlane(v.ray(width() / 2.0, height() / 2.0), {0, 0, 0}, {0, 1, 0}, hit)) return toFloat3(hit);
	return toFloat3(m_cam.target);
}

QPointF Scene3DView::itemScreenPos(const BuilderSelection &s) const {
	const Float3 *p = handle(s, 0);
	double sx, sy;
	if (!p || !view().project(toV3(*p), sx, sy)) return QPointF();
	return QPointF(sx, sy);
}

double Scene3DView::gizmoLength(const scene_view::View &v, const V3 &at) const {
	const double depth = std::max(v.nearPlane * 2, v.toView(at).z);
	return 80.0 * v.unitsPerPixel(depth);  // about 80 pixels, whatever the zoom
}

V3 Scene3DView::localAxis(const Object &o, int axis) const {
	return toV3(rotateXYZ(Float3{kAxisDir[axis].x, kAxisDir[axis].y, kAxisDir[axis].z}, o.rotation));
}

bool Scene3DView::scaleHandleUsed(const Object &o, int axis) const { return !(o.shape == ShapeKind::Quad && axis == 1); }

QPointF Scene3DView::axisArrowPoint(int axis, double fraction) const {
	const Float3 *p = handle(m_sel, 0);
	if (!p || axis < 0 || axis > 2) return QPointF();
	const scene_view::View v = view();
	const V3 base = toV3(*p);
	double sx, sy;
	if (!v.project(base + kAxisDir[axis] * (gizmoLength(v, base) * fraction), sx, sy)) return QPointF();
	return QPointF(sx, sy);
}

QPointF Scene3DView::ringPoint(int axis, double deg) const {
	const Float3 *p = handle(m_sel, 0);
	if (!p || axis < 0 || axis > 2) return QPointF();
	const scene_view::View v = view();
	const V3 base = toV3(*p);
	double sx, sy;
	if (!v.project(ringAt(base, axis, gizmoLength(v, base) * 0.9, deg), sx, sy)) return QPointF();
	return QPointF(sx, sy);
}

QPointF Scene3DView::scaleHandlePoint(int axis, double fraction) const {
	const Object *o = selectedObject();
	if (!o || axis < 0 || axis > 2) return QPointF();
	const scene_view::View v = view();
	const V3 base = toV3(o->position);
	double sx, sy;
	if (!v.project(base + localAxis(*o, axis) * (gizmoLength(v, base) * fraction), sx, sy)) return QPointF();
	return QPointF(sx, sy);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Geometry of the shapes
// ---------------------------------------------------------------------------------------------------------------------------------
QList<Scene3DView::Face> Scene3DView::buildFaces(const scene_view::View &) const {
	QList<Face> faces;
	if (!m_doc) return faces;
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		QColor base = toQColor(o.material.color);
		if (o.material.kind == MaterialKind::Dielectric) base = QColor(150, 200, 240);
		else if (o.material.kind == MaterialKind::Conductor) base = base.lighter(115);
		if (o.emissive) base = QColor(255, 220, 120);
		const bool glass = o.material.kind == MaterialKind::Dielectric;
		if (glass) base.setAlpha(150);

		auto addFace = [&](std::vector<Float3> local) {
			Face f;
			f.owner = i;
			f.color = base;
			for (const Float3 &q : local) {
				Float3 w = rotateXYZ(q, o.rotation);
				f.world.push_back({w.x + o.position.x, w.y + o.position.y, w.z + o.position.z});
			}
			faces.append(f);
			return &faces.last();
		};
		auto ring = [&](double r, double y, int n, int k) { return Float3{r * std::cos(2 * kPi * k / n), y, r * std::sin(2 * kPi * k / n)}; };
		auto boxFaces = [&](double x0, double x1, double y0, double y1, double z0, double z1) {
			const Float3 c[8] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}, {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}};
			const int idx[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}};
			for (const auto &q : idx) addFace({c[q[0]], c[q[1]], c[q[2]], c[q[3]]});
		};

		switch (o.shape) {
			case ShapeKind::Sphere: {
				const int lat = 10, lon = 18;
				auto pt = [&](int a, int b) {
					const double th = kPi * a / lat, ph = 2 * kPi * b / lon;
					return Float3{o.radius * std::sin(th) * std::cos(ph), o.radius * std::cos(th), o.radius * std::sin(th) * std::sin(ph)};
				};
				for (int a = 0; a < lat; ++a)
					for (int b = 0; b < lon; ++b) {
						if (a == 0) addFace({pt(0, 0), pt(1, b + 1), pt(1, b)});
						else if (a == lat - 1) addFace({pt(a, b), pt(a, b + 1), pt(lat, 0)});
						else addFace({pt(a, b), pt(a, b + 1), pt(a + 1, b + 1), pt(a + 1, b)});
					}
				break;
			}
			case ShapeKind::Box: boxFaces(-o.size.x / 2, o.size.x / 2, -o.size.y / 2, o.size.y / 2, -o.size.z / 2, o.size.z / 2); break;
			case ShapeKind::Quad: {
				const double x = o.size.x / 2, z = o.size.z / 2;
				Face *f = addFace({{-x, 0, -z}, {x, 0, -z}, {x, 0, z}, {-x, 0, z}});
				f->bigFlat = o.size.x * o.size.z > 30;
				break;
			}
			case ShapeKind::Disk: {
				std::vector<Float3> pts;
				for (int k = 0; k < 28; ++k) pts.push_back(ring(o.radius, 0, 28, k));
				addFace(pts);
				break;
			}
			case ShapeKind::Cylinder: {
				const int n = 24;
				for (int k = 0; k < n; ++k) addFace({ring(o.radius, -o.height / 2, n, k), ring(o.radius, -o.height / 2, n, k + 1), ring(o.radius, o.height / 2, n, k + 1), ring(o.radius, o.height / 2, n, k)});
				std::vector<Float3> lo, hi;
				for (int k = 0; k < n; ++k) { lo.push_back(ring(o.radius, -o.height / 2, n, k)); hi.push_back(ring(o.radius, o.height / 2, n, k)); }
				addFace(lo);
				addFace(hi);
				break;
			}
			case ShapeKind::Cone: {
				const int n = 24;
				for (int k = 0; k < n; ++k) addFace({ring(o.radius, -o.height / 2, n, k), ring(o.radius, -o.height / 2, n, k + 1), {0, o.height / 2, 0}});
				std::vector<Float3> lo;
				for (int k = 0; k < n; ++k) lo.push_back(ring(o.radius, -o.height / 2, n, k));
				addFace(lo);
				break;
			}
			case ShapeKind::Mesh: {
				const double s = o.meshScale;
				if (const mesh_preview::MeshPreview *m = meshPreview(o.meshFile)) {
					// The file's own bounding box, at the object's scale: a see-through box (its vertices are drawn as dots over it).
					const size_t before = faces.size();
					boxFaces(m->lo[0] * s, m->hi[0] * s, m->lo[1] * s, m->hi[1] * s, m->lo[2] * s, m->hi[2] * s);
					for (qsizetype k = before; k < faces.size(); ++k) faces[k].color.setAlpha(60);
				} else {
					// The file could not be read: an octahedron marks the object.
					const double r = 0.5 * s;
					const Float3 px{r, 0, 0}, nx{-r, 0, 0}, py{0, r, 0}, ny{0, -r, 0}, pz{0, 0, r}, nz{0, 0, -r};
					const Float3 tris[8][3] = {{px, py, pz}, {py, nx, pz}, {nx, ny, pz}, {ny, px, pz}, {py, px, nz}, {nx, py, nz}, {ny, nx, nz}, {px, ny, nz}};
					for (const auto &t : tris) addFace({t[0], t[1], t[2]});
				}
				break;
			}
		}
	}
	return faces;
}

// Camera space, near-clipped, projected; `depth` is the centroid's, for sorting.
void Scene3DView::projectFaces(QList<Face> &faces, const scene_view::View &v) const {
	for (Face &f : faces) {
		std::vector<V3> cam;
		cam.reserve(f.world.size());
		for (const V3 &p : f.world) cam.push_back(v.toView(p));
		cam = scene_view::clipNear(cam, v.nearPlane);
		f.screen.clear();
		if (cam.size() < 3) { f.depth = 1e18; continue; }
		double sum = 0;
		for (const V3 &p : cam) {
			double sx, sy;
			v.toPixel(p, sx, sy);
			f.screen << QPointF(sx, sy);
			sum += p.z;
		}
		f.depth = sum / static_cast<double>(cam.size());
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------------------------------------------------------------
Scene3DView::Hit Scene3DView::hitTest(const QPointF &px) const {
	Hit h;
	if (!m_doc) return h;
	const scene_view::View v = view();
	auto withinPx = [&](const Float3 &p, double r) {
		double sx, sy;
		return v.project(toV3(p), sx, sy) && std::hypot(sx - px.x(), sy - px.y()) <= r;
	};

	// The selected item's tool first: it overlaps the item it belongs to.
	if (const Float3 *p = handle(m_sel, 0)) {
		const V3 base = toV3(*p);
		const double len = gizmoLength(v, base);
		double bx, by;
		if (v.project(base, bx, by)) {
			const GizmoMode mode = effectiveGizmo();
			double best = 9.0;
			if (mode == GizmoMode::Move) {
				for (int a = 0; a < 3; ++a) {
					double ex, ey;
					if (!v.project(base + kAxisDir[a] * len, ex, ey)) continue;
					const double d = distanceToSegment(px, QPointF(bx, by), QPointF(ex, ey));
					if (d < best) { best = d; h.kind = Hit::Kind::Axis; h.sel = m_sel; h.axis = a; h.which = 0; }
				}
			} else if (mode == GizmoMode::Rotate) {
				for (int a = 0; a < 3; ++a) {
					QPointF prev;
					bool havePrev = false;
					for (int k = 0; k <= kRingSegments; ++k) {
						double sx, sy;
						const bool ok = v.project(ringAt(base, a, len * 0.9, 360.0 * k / kRingSegments), sx, sy);
						if (ok && havePrev) {
							const double d = distanceToSegment(px, prev, QPointF(sx, sy));
							if (d < best) { best = d; h.kind = Hit::Kind::Ring; h.sel = m_sel; h.axis = a; h.which = 0; }
						}
						prev = QPointF(sx, sy);
						havePrev = ok;
					}
				}
			} else if (const Object *o = selectedObject()) {
				for (int a = 0; a < 3; ++a) {
					if (!scaleHandleUsed(*o, a)) continue;
					double ex, ey;
					if (!v.project(base + localAxis(*o, a) * len, ex, ey)) continue;
					const double d = std::min(std::hypot(px.x() - ex, px.y() - ey) - 2.0, distanceToSegment(px, QPointF(bx, by), QPointF(ex, ey)) + 3.0);
					if (d < best) { best = d; h.kind = Hit::Kind::ScaleHandle; h.sel = m_sel; h.axis = a; h.which = 0; }
				}
			}
			if (h.kind != Hit::Kind::None) return h;
		}
	}
	// The selected camera's or light's target.
	if (handle(m_sel, 1) && withinPx(*handle(m_sel, 1), 11)) { h.kind = Hit::Kind::Item; h.sel = m_sel; h.which = 1; return h; }
	// Lights and the camera: small, so they come before the faces.
	for (int i = static_cast<int>(m_doc->lights.size()) - 1; i >= 0; --i) {
		const Light &l = m_doc->lights[i];
		if (l.kind != LightKind::Infinite && withinPx(l.position, 13)) { h.kind = Hit::Kind::Item; h.sel = {BuilderSelection::Kind::Light, i}; return h; }
	}
	if (withinPx(m_doc->camera.position, 15)) { h.kind = Hit::Kind::Item; h.sel = {BuilderSelection::Kind::Camera, 0}; return h; }

	// Objects: the nearest face under the pointer.
	QList<Face> faces = buildFaces(v);
	projectFaces(faces, v);
	const Face *best = nullptr;
	for (const Face &f : faces) {
		if (f.screen.size() < 3 || !f.screen.containsPoint(px, Qt::OddEvenFill)) continue;
		if (!best || f.depth < best->depth) best = &f;
	}
	if (best) { h.kind = Hit::Kind::Item; h.sel = {BuilderSelection::Kind::Object, best->owner}; }
	return h;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------------------------------------------------------------
void Scene3DView::paintEvent(QPaintEvent *) {
	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing, true);
	p.fillRect(rect(), palette().color(QPalette::Base));
	const QColor text = palette().color(QPalette::Text);
	const QColor accent = palette().color(QPalette::Highlight);
	const scene_view::View v = view();

	// A line between two world points, clipped at the camera.
	auto line = [&](const V3 &a, const V3 &b, const QPen &pen) {
		V3 va = v.toView(a), vb = v.toView(b);
		if (va.z < v.nearPlane && vb.z < v.nearPlane) return;
		if (va.z < v.nearPlane) va = va + (vb - va) * ((v.nearPlane - va.z) / (vb.z - va.z));
		else if (vb.z < v.nearPlane) vb = vb + (va - vb) * ((v.nearPlane - vb.z) / (va.z - vb.z));
		double x0, y0, x1, y1;
		v.toPixel(va, x0, y0);
		v.toPixel(vb, x1, y1);
		p.setPen(pen);
		p.drawLine(QPointF(x0, y0), QPointF(x1, y1));
	};
	auto pixel = [&](const V3 &w, QPointF &out) {
		double sx, sy;
		if (!v.project(w, sx, sy)) return false;
		out = QPointF(sx, sy);
		return true;
	};

	// Floor grid and the world axes
	QColor gridColor = text; gridColor.setAlpha(26);
	const int half = 10;
	for (int i = -half; i <= half; ++i) {
		line({double(i), 0, double(-half)}, {double(i), 0, double(half)}, QPen(i == 0 ? QColor(72, 124, 232, 150) : gridColor, i == 0 ? 1.6 : 1.0));
		line({double(-half), 0, double(i)}, {double(half), 0, double(i)}, QPen(i == 0 ? QColor(232, 72, 72, 150) : gridColor, i == 0 ? 1.6 : 1.0));
	}
	line({0, 0, 0}, {0, 2, 0}, QPen(QColor(84, 200, 96, 150), 1.6));

	if (!m_doc) return;

	// Faces, back to front (floor-like panels first)
	QList<Face> faces = buildFaces(v);
	projectFaces(faces, v);
	std::stable_sort(faces.begin(), faces.end(), [](const Face &a, const Face &b) {
		if (a.bigFlat != b.bigFlat) return a.bigFlat;
		return a.depth > b.depth;
	});
	const V3 lightDir = scene_view::normalize({0.4, 0.8, 0.5});
	for (const Face &f : faces) {
		if (f.screen.size() < 3) continue;
		V3 n = scene_view::cross(f.world[1] - f.world[0], f.world[2] - f.world[0]);
		n = scene_view::normalize(n);
		const double shade = 0.40 + 0.60 * std::abs(scene_view::dot(n, lightDir));
		QColor c = f.color;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == f.owner;
		const bool emissive = m_doc->objects[f.owner].emissive;
		const double k = emissive ? 1.0 : shade;
		c = QColor(std::min(255, int(c.red() * k)), std::min(255, int(c.green() * k)), std::min(255, int(c.blue() * k)), c.alpha());
		QColor edge = c.darker(160);
		edge.setAlpha(120);
		p.setBrush(c);
		p.setPen(QPen(selected ? accent : edge, selected ? 1.6 : 0.8));
		p.drawPolygon(f.screen);
	}

	// A mesh's sampled vertices, over its bounding box, so its shape shows
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		if (o.shape != ShapeKind::Mesh) continue;
		const mesh_preview::MeshPreview *m = meshPreview(o.meshFile);
		if (!m) continue;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == i;
		QColor dot = selected ? accent : toQColor(o.material.color).lighter(130);
		dot.setAlpha(210);
		p.setPen(QPen(dot, 2.0, Qt::SolidLine, Qt::RoundCap));
		QList<QPointF> pts;
		for (size_t k = 0; k + 2 < m->samples.size(); k += 3) {
			const Float3 w = rotateXYZ(Float3{m->samples[k] * o.meshScale, m->samples[k + 1] * o.meshScale, m->samples[k + 2] * o.meshScale}, o.rotation);
			double sx, sy;
			if (v.project({w.x + o.position.x, w.y + o.position.y, w.z + o.position.z}, sx, sy)) pts.append(QPointF(sx, sy));
		}
		p.drawPoints(QPolygonF(pts.toVector()));
	}

	QFont small = font();
	small.setPointSizeF(std::max(7.0, font().pointSizeF() - 1.5));
	// Object names
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		QPointF c;
		if (!pixel(toV3(o.position), c)) continue;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == i;
		p.setBrush(selected ? accent : text);
		p.setPen(Qt::NoPen);
		p.drawEllipse(c, 2.5, 2.5);
		p.setFont(small);
		p.setPen(text);
		p.drawText(c + QPointF(6, -6), QString::fromStdString(o.name));
		p.setFont(font());
	}

	// Lights
	for (int i = 0; i < static_cast<int>(m_doc->lights.size()); ++i) {
		const Light &l = m_doc->lights[i];
		if (l.kind == LightKind::Infinite) continue;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Light && m_sel.index == i;
		QPointF c;
		if (!pixel(toV3(l.position), c)) continue;
		if (l.kind == LightKind::Spot || l.kind == LightKind::Distant) {
			line(toV3(l.position), toV3(l.target), QPen(selected ? accent : QColor(230, 180, 40), 1.2, Qt::DashLine));
			QPointF t;
			if (pixel(toV3(l.target), t)) {
				p.setPen(QPen(selected ? accent : QColor(230, 180, 40), 1.2));
				p.drawLine(t - QPointF(4, 4), t + QPointF(4, 4));
				p.drawLine(t - QPointF(4, -4), t + QPointF(4, -4));
			}
		}
		p.setPen(QPen(selected ? accent : QColor(200, 150, 20), selected ? 2.5 : 1.5));
		p.setBrush(toQColor(l.color).lighter(130));
		p.drawEllipse(c, 6, 6);
		for (int k = 0; k < 8; ++k) {
			const double a = k * kPi / 4;
			p.drawLine(c + QPointF(std::cos(a) * 9, std::sin(a) * 9), c + QPointF(std::cos(a) * 13, std::sin(a) * 13));
		}
		p.setPen(text);
		p.setFont(small);
		p.drawText(c + QPointF(10, -10), QString::fromStdString(l.name));
		p.setFont(font());
	}

	// Camera: its position, what it looks at, and the frame it sees
	{
		const bool selected = m_sel.kind == BuilderSelection::Kind::Camera;
		const QColor camColor = selected ? accent : QColor(70, 130, 200);
		const V3 pos = toV3(m_doc->camera.position), tgt = toV3(m_doc->camera.target);
		const V3 f = scene_view::normalize(tgt - pos);
		V3 r = scene_view::cross(f, toV3(m_doc->camera.up));
		if (scene_view::length(r) > 1e-6) {
			r = scene_view::normalize(r);
			const V3 u = scene_view::cross(r, f);
			const double dist = std::min(1.3, scene_view::length(tgt - pos));
			const double aspect = m_doc->render.height > 0 ? double(m_doc->render.width) / m_doc->render.height : 1.0;
			const double hh = dist * std::tan(m_doc->camera.fov * kPi / 360.0), hw = hh * aspect;
			const V3 centre = pos + f * dist;
			const V3 corner[4] = {centre - r * hw - u * hh, centre + r * hw - u * hh, centre + r * hw + u * hh, centre - r * hw + u * hh};
			const QPen pen(camColor, selected ? 2.0 : 1.2);
			for (int k = 0; k < 4; ++k) {
				line(pos, corner[k], pen);
				line(corner[k], corner[(k + 1) % 4], pen);
			}
			line(pos, tgt, QPen(camColor, 1.0, Qt::DashLine));
		}
		QPointF c, t;
		if (pixel(pos, c)) {
			p.setPen(QPen(camColor, selected ? 2.5 : 1.5));
			p.setBrush(camColor.lighter(160));
			p.drawRoundedRect(QRectF(c.x() - 9, c.y() - 6, 18, 12), 2, 2);
			p.drawEllipse(c, 4, 4);
			p.setPen(text);
			p.setFont(small);
			p.drawText(c + QPointF(10, 18), tr("Camera"));
			p.setFont(font());
		}
		if (pixel(tgt, t)) {
			p.setPen(QPen(camColor, 1.5));
			p.setBrush(Qt::NoBrush);
			p.drawEllipse(t, 5, 5);
		}
	}

	// The selected item's tool
	if (const Float3 *hp = handle(m_sel, 0)) {
		const V3 base = toV3(*hp);
		const double len = gizmoLength(v, base);
		const GizmoMode mode = effectiveGizmo();
		QPointF b;
		if (pixel(base, b)) {
			for (int a = 0; a < 3; ++a) {
				const bool active = (m_mode == Mode::Axis || m_mode == Mode::Rotate || m_mode == Mode::Scale) && m_drag.axis == a;
				if (mode == GizmoMode::Rotate) {
					const QPen pen(kAxisColor[a], active ? 3.6 : 2.2);
					V3 prev = ringAt(base, a, len * 0.9, 0);
					for (int k = 1; k <= kRingSegments; ++k) {
						const V3 cur = ringAt(base, a, len * 0.9, 360.0 * k / kRingSegments);
						line(prev, cur, pen);
						prev = cur;
					}
					QPointF e;
					if (pixel(ringAt(base, a, len * 0.9, 45), e)) {
						p.setPen(kAxisColor[a]);
						p.setFont(small);
						p.drawText(e + QPointF(6, -4), QString(QChar('X' + a)));
						p.setFont(font());
					}
					continue;
				}
				const Object *o = selectedObject();
				if (mode == GizmoMode::Scale && o && !scaleHandleUsed(*o, a)) continue;
				const V3 dir = (mode == GizmoMode::Scale && o) ? localAxis(*o, a) : kAxisDir[a];
				const V3 tip = base + dir * len;
				QPointF e;
				if (!pixel(tip, e)) continue;
				line(base, tip, QPen(kAxisColor[a], active ? 4.0 : 2.6));
				QPointF d = e - b;
				const double dl = std::hypot(d.x(), d.y());
				p.setPen(Qt::NoPen);
				p.setBrush(kAxisColor[a]);
				if (mode == GizmoMode::Scale) {
					p.drawRect(QRectF(e.x() - 5, e.y() - 5, 10, 10));
				} else if (dl > 6) {
					d /= dl;
					const QPointF n(-d.y(), d.x());
					QPolygonF head;
					head << e + d * 4 << e - d * 8 + n * 5 << e - d * 8 - n * 5;
					p.drawPolygon(head);
				}
				p.setPen(kAxisColor[a]);
				p.setFont(small);
				p.drawText(e + QPointF(8, 4), QString(QChar('X' + a)));
				p.setFont(font());
			}
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Mouse
// ---------------------------------------------------------------------------------------------------------------------------------
void Scene3DView::mousePressEvent(QMouseEvent *e) {
	setFocus();
	const QPointF px = e->position();
	m_last = px;
	if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
		m_mode = Mode::Pan;
		return;
	}
	if (e->button() != Qt::LeftButton) return;
	const Hit h = hitTest(px);
	if (h.kind == Hit::Kind::None) {
		// A click on nothing deselects; a drag on nothing orbits.
		m_mode = Mode::Orbit;
		emit selectionRequested(BuilderSelection{});
		return;
	}
	m_drag = h;
	if (h.kind == Hit::Kind::Item) emit selectionRequested(h.sel);
	const Float3 *pos = handle(h.sel, h.which);
	if (!pos) { m_mode = Mode::None; return; }
	m_dragStart = toV3(*pos);
	emit dragBegan();
	const scene_view::View v = view();
	const scene_view::Ray ray = v.ray(px.x(), px.y());
	if (h.kind == Hit::Kind::Axis) {
		m_mode = Mode::Axis;
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, kAxisDir[h.axis], t)) m_mode = Mode::None;
		m_axisT0 = t;
		return;
	}
	if (h.kind == Hit::Kind::Ring) {
		m_mode = Mode::Rotate;
		m_dragObject = *selectedObject();
		if (!scene_view::angleAround(ray, m_dragStart, kAxisDir[h.axis], kRingRef[h.axis], m_angle0)) m_mode = Mode::None;
		return;
	}
	if (h.kind == Hit::Kind::ScaleHandle) {
		m_mode = Mode::Scale;
		m_dragObject = *selectedObject();
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, localAxis(m_dragObject, h.axis), t) || std::abs(t) < 1e-3) m_mode = Mode::None;
		m_axisT0 = t;
		return;
	}
	if (e->modifiers() & Qt::ShiftModifier) {
		m_mode = Mode::Vertical;
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, kAxisDir[1], t)) m_mode = Mode::None;
		m_axisT0 = t;
		return;
	}
	m_mode = Mode::Ground;
	V3 hit;
	if (scene_view::rayPlane(ray, m_dragStart, {0, 1, 0}, hit)) m_grabOffset = m_dragStart - hit;
	else m_mode = Mode::None;  // looking along the floor: nothing to drag on
}

// The object with `axis` (0 X, 1 Y, 2 Z, in its own frame) stretched by `factor`: a box's size, a cylinder's or cone's height (Y) or radius (X, Z), a quad's
// width or depth; a sphere, a disk and a mesh scale all round.
Object Scene3DView::scaled(const Object &o, int axis, double factor) const {
	Object r = o;
	switch (o.shape) {
		case ShapeKind::Sphere:
		case ShapeKind::Disk: r.radius = o.radius * factor; break;
		case ShapeKind::Mesh: r.meshScale = o.meshScale * factor; break;
		case ShapeKind::Cylinder:
		case ShapeKind::Cone:
			if (axis == 1) r.height = o.height * factor;
			else r.radius = o.radius * factor;
			break;
		case ShapeKind::Box: (axis == 0 ? r.size.x : axis == 1 ? r.size.y : r.size.z) = (axis == 0 ? o.size.x : axis == 1 ? o.size.y : o.size.z) * factor; break;
		case ShapeKind::Quad:
			if (axis == 0) r.size.x = o.size.x * factor;
			else if (axis == 2) r.size.z = o.size.z * factor;
			break;
	}
	return r;
}

void Scene3DView::applyDrag(const QPointF &px, Qt::KeyboardModifiers mods) {
	const scene_view::View v = view();
	const scene_view::Ray ray = v.ray(px.x(), px.y());
	const bool snap = m_snap && !(mods & Qt::AltModifier);

	if (m_mode == Mode::Rotate) {
		double angle = 0;
		if (!scene_view::angleAround(ray, m_dragStart, kAxisDir[m_drag.axis], kRingRef[m_drag.axis], angle)) return;
		double delta = scene_view::angleDelta(m_angle0, angle);
		if (snap) delta = std::round(delta / 5.0) * 5.0;
		Object o = m_dragObject;
		const V3 turned = scene_view::turnAboutWorldAxis(toV3(m_dragObject.rotation), kAxisDir[m_drag.axis], delta);
		o.rotation = toFloat3(turned);
		emit objectEdited(m_drag.sel, o);
		return;
	}
	if (m_mode == Mode::Scale) {
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, localAxis(m_dragObject, m_drag.axis), t)) return;
		double factor = std::clamp(t / m_axisT0, 0.05, 50.0);
		if (snap) factor = std::max(0.05, std::round(factor * 20.0) / 20.0);  // steps of 5 %
		emit objectEdited(m_drag.sel, scaled(m_dragObject, m_drag.axis, factor));
		return;
	}

	const Float3 *current = handle(m_drag.sel, m_drag.which);
	if (!current) return;
	Float3 w = *current;
	if (m_mode == Mode::Ground) {
		V3 hit;
		if (!scene_view::rayPlane(ray, m_dragStart, {0, 1, 0}, hit)) return;
		const V3 target = hit + m_grabOffset;
		w.x = snap ? snapQuarter(target.x) : target.x;
		w.z = snap ? snapQuarter(target.z) : target.z;
		w.y = m_dragStart.y;
	} else if (m_mode == Mode::Axis || m_mode == Mode::Vertical) {
		const int axis = m_mode == Mode::Vertical ? 1 : m_drag.axis;
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, kAxisDir[axis], t)) return;
		double value = (axis == 0 ? m_dragStart.x : axis == 1 ? m_dragStart.y : m_dragStart.z) + (t - m_axisT0);
		if (snap) value = snapQuarter(value);
		(axis == 0 ? w.x : axis == 1 ? w.y : w.z) = value;
	} else {
		return;
	}
	emit positionDragged(m_drag.sel, m_drag.which, w);
}

void Scene3DView::mouseMoveEvent(QMouseEvent *e) {
	const QPointF px = e->position();
	const QPointF d = px - m_last;
	if (m_mode == Mode::Orbit) {
		m_userView = true;
		m_cam.orbit(-d.x() * 0.4, d.y() * 0.4);
		m_last = px;
		update();
	} else if (m_mode == Mode::Pan) {
		m_userView = true;
		const double upp = view().unitsPerPixel(m_cam.distance);
		m_cam.pan(-d.x() * upp, d.y() * upp);
		m_last = px;
		update();
	} else if (m_mode != Mode::None) {
		applyDrag(px, e->modifiers());
	}
}

void Scene3DView::mouseReleaseEvent(QMouseEvent *) {
	m_mode = Mode::None;
	m_drag = Hit{};
	update();
}

void Scene3DView::wheelEvent(QWheelEvent *e) {
	m_userView = true;
	m_cam.dolly(std::pow(0.88, e->angleDelta().y() / 120.0));
	update();
	e->accept();
}
