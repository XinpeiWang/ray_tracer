// scene_3d_view.cpp - see scene_3d_view.h.
#include "scene_3d_view.h"

#include "scene_builder_common.h"

#include <QFileInfo>
#include <QKeyEvent>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPointer>
#include <QPolygonF>
#include <QThreadPool>
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

Scene3DView::Scene3DView(QWidget *parent) : QWidget(parent) {
	setMouseTracking(false);
	setFocusPolicy(Qt::ClickFocus);
	setMinimumSize(280, 220);
	setAutoFillBackground(false);
	m_clock.start();
}

Scene3DView::~Scene3DView() = default;

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

// ---------------------------------------------------------------------------------------------------------------------------------
// Mesh files: read on a worker thread, so painting and picking never wait for a file
// ---------------------------------------------------------------------------------------------------------------------------------
const mesh_preview::MeshPreview *Scene3DView::meshPreview(const std::string &path) const {
	if (path.empty()) return nullptr;
	const qint64 now = m_clock.elapsed();
	auto it = m_meshes.find(path);
	if (it == m_meshes.end()) {
		const QFileInfo info(QString::fromStdString(path));
		startMeshLoad(path, info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1, info.size());
		return nullptr;
	}
	CachedMesh &c = it->second;
	if (!c.pending && now - c.checkedAt > kMeshRecheckMs) {
		c.checkedAt = now;
		const QFileInfo info(QString::fromStdString(path));
		const qint64 modified = info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1;
		if (modified != c.modified || info.size() != c.size) startMeshLoad(path, modified, info.size());
	}
	return (!c.pending && c.preview.ok) ? &c.preview : nullptr;
}

void Scene3DView::startMeshLoad(const std::string &path, qint64 modified, qint64 size) const {
	CachedMesh &c = m_meshes[path];
	c.pending = true;
	c.modified = modified;
	c.size = size;
	c.checkedAt = m_clock.elapsed();
	QPointer<Scene3DView> self(const_cast<Scene3DView *>(this));
	QThreadPool::globalInstance()->start([self, path, modified, size]() {
		mesh_preview::MeshPreview preview = mesh_preview::load(path);
		if (!self) return;
		QMetaObject::invokeMethod(
		    self.data(), [self, path, modified, size, p = std::move(preview)]() mutable {
			    if (self) self->meshLoaded(path, std::move(p), modified, size);
		    },
		    Qt::QueuedConnection);
	});
}

void Scene3DView::meshLoaded(const std::string &path, mesh_preview::MeshPreview preview, qint64 modified, qint64 size) {
	CachedMesh &c = m_meshes[path];
	if (c.modified != modified || c.size != size) return;  // the file changed again meanwhile: a newer read is on its way
	c.preview = std::move(preview);
	c.pending = false;
	c.checkedAt = m_clock.elapsed();
	++m_meshVersion;
	if (!m_userView) frameAll();
	update();
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Framing and positions
// ---------------------------------------------------------------------------------------------------------------------------------
void Scene3DView::frameAll() {
	m_userView = false;
	if (!m_doc) return;
	V3 lo{1e18, 1e18, 1e18}, hi{-1e18, -1e18, -1e18};
	auto add = [&](const Float3 &p, double pad) {
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(pad)) return;
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
	if (scene_view::rayPlane(v.ray(width() / 2.0, height() / 2.0), {0, 0, 0}, {0, 1, 0}, hit) && scene_view::length(hit - m_cam.target) <= kMaxDropDistance * m_cam.distance)
		return toFloat3(hit);
	return toFloat3(V3{m_cam.target.x, 0.0, m_cam.target.z});  // a level camera: the floor under what it looks at
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
// Everything that shapes a face: each object's kind, place, size, colour and mesh file, and how many mesh files have finished loading.
std::uint64_t Scene3DView::geometrySignature() const {
	std::uint64_t h = 14695981039346656037ull;
	if (!m_doc) return h;
	mix(h, m_doc->objects.size());
	mix(h, m_meshVersion);
	for (const Object &o : m_doc->objects) {
		mix(h, o.shape);
		mix(h, o.position); mix(h, o.rotation); mix(h, o.size);
		mix(h, o.radius); mix(h, o.radius2); mix(h, o.height); mix(h, o.steps); mix(h, o.meshScale);
		mix(h, o.material.kind); mix(h, o.material.color);
		mix(h, o.emissive);
		mix(h, o.meshFile.data(), o.meshFile.size());
	}
	return h;
}

std::vector<Scene3DView::Face> &Scene3DView::faces() const {
	const std::uint64_t sig = geometrySignature();
	if (m_facesValid && sig == m_facesSignature) return m_faces;
	m_faces.clear();
	m_facesSignature = sig;
	m_facesValid = true;
	if (!m_doc) return m_faces;
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		QColor base = toQColor(o.material.color);
		if (o.material.kind == MaterialKind::Dielectric) base = QColor(150, 200, 240);
		else if (o.material.kind == MaterialKind::Conductor) base = base.lighter(115);
		if (o.emissive) base = QColor(255, 220, 120);
		if (o.material.kind == MaterialKind::Dielectric) base.setAlpha(150);

		auto addFace = [&](std::vector<Float3> local) {
			Face f;
			f.owner = i;
			f.color = base;
			for (const Float3 &q : local) {
				Float3 w = rotateXYZ(q, o.rotation);
				f.world.push_back({w.x + o.position.x, w.y + o.position.y, w.z + o.position.z});
			}
			m_faces.push_back(std::move(f));
			return &m_faces.back();
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
				f->flat = true;
				break;
			}
			case ShapeKind::Disk: {
				std::vector<Float3> pts;
				for (int k = 0; k < 28; ++k) pts.push_back(ring(o.radius, 0, 28, k));
				addFace(pts)->flat = true;
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
			case ShapeKind::Pyramid:
			case ShapeKind::Wedge:
			case ShapeKind::Stairs:
			case ShapeKind::Torus:
			case ShapeKind::Capsule:
			case ShapeKind::Dome:
			case ShapeKind::Tube: {
				const scene_doc::ShapeMesh mesh = scene_doc::generatedMesh(o);
				auto vertex = [&mesh](int k) { return Float3{mesh.P[k * 3], mesh.P[k * 3 + 1], mesh.P[k * 3 + 2]}; };
				for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3) addFace({vertex(mesh.indices[t]), vertex(mesh.indices[t + 1]), vertex(mesh.indices[t + 2])});
				break;
			}
			case ShapeKind::Mesh: {
				const double s = o.meshScale;
				if (const mesh_preview::MeshPreview *m = meshPreview(o.meshFile)) {
					// The file's own bounding box, at the object's scale: a see-through box (its vertices are drawn as dots over it).
					const std::size_t before = m_faces.size();
					boxFaces(m->lo[0] * s, m->hi[0] * s, m->lo[1] * s, m->hi[1] * s, m->lo[2] * s, m->hi[2] * s);
					for (std::size_t k = before; k < m_faces.size(); ++k) m_faces[k].color.setAlpha(60);
				} else {
					// Not read yet (or unreadable): an octahedron marks the object.
					const double r = 0.5 * s;
					const Float3 px{r, 0, 0}, nx{-r, 0, 0}, py{0, r, 0}, ny{0, -r, 0}, pz{0, 0, r}, nz{0, 0, -r};
					const Float3 tris[8][3] = {{px, py, pz}, {py, nx, pz}, {nx, ny, pz}, {ny, px, pz}, {py, px, nz}, {nx, py, nz}, {ny, nx, nz}, {px, ny, nz}};
					for (const auto &t : tris) addFace({t[0], t[1], t[2]});
				}
				break;
			}
		}
	}
	return m_faces;
}

// Camera space, near-clipped, projected (in place; the world polygons stay). depth is the centroid's, farDepth the farthest vertex's.
void Scene3DView::projectFaces(const scene_view::View &v) const {
	for (Face &f : faces()) {
		std::vector<V3> cam;
		cam.reserve(f.world.size());
		for (const V3 &p : f.world) cam.push_back(v.toView(p));
		cam = scene_view::clipNear(cam, v.nearPlane);
		f.screen.clear();
		if (cam.size() < 3) { f.depth = f.farDepth = 1e18; continue; }
		double sum = 0, far = 0;
		for (const V3 &p : cam) {
			double sx, sy;
			v.toPixel(p, sx, sy);
			f.screen << QPointF(sx, sy);
			sum += p.z;
			far = std::max(far, p.z);
		}
		f.depth = sum / static_cast<double>(cam.size());
		f.farDepth = far;
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
			if (mode == GizmoMode::Move || mode == GizmoMode::Scale) {
				// Arrows (Move) or squares at the same ends (Scale, along the object's own axes).
				const Object *o = selectedObject();
				scene_gizmo::P2 tips[3];
				bool used[3] = {true, true, true};
				for (int a = 0; a < 3; ++a) {
					const V3 dir = (mode == GizmoMode::Scale && o) ? localAxis(*o, a) : kAxisDir[a];
					double ex = 0, ey = 0;
					used[a] = v.project(base + dir * len, ex, ey) && !(mode == GizmoMode::Scale && o && !scene_gizmo::scaleHandleUsed(*o, a));
					tips[a] = {ex, ey};
				}
				const int axis = mode == GizmoMode::Move ? scene_gizmo::pickArrow(toP2(px), {bx, by}, tips, used, kArrowDeadPx, kArrowPickPx)
				                                         : scene_gizmo::pickTip(toP2(px), tips, used, kHandlePickPx);
				if (axis >= 0) {
					h.kind = mode == GizmoMode::Move ? Hit::Kind::Axis : Hit::Kind::ScaleHandle;
					h.sel = m_sel;
					h.axis = axis;
					return h;
				}
			} else {
				double best = 9.0;
				for (int a = 0; a < 3; ++a) {
					QPointF prev;
					bool havePrev = false;
					for (int k = 0; k <= kRingSegments; ++k) {
						double sx, sy;
						const bool ok = v.project(ringAt(base, a, len * 0.9, 360.0 * k / kRingSegments), sx, sy);
						if (ok && havePrev) {
							const double d = scene_gizmo::distanceToSegment(toP2(px), toP2(prev), {sx, sy});
							if (d < best) { best = d; h.kind = Hit::Kind::Ring; h.sel = m_sel; h.axis = a; h.which = 0; }
						}
						prev = QPointF(sx, sy);
						havePrev = ok;
					}
				}
				if (h.kind != Hit::Kind::None) return h;
			}
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
	projectFaces(v);
	const Face *best = nullptr;
	for (const Face &f : faces()) {
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
	Ctx c{&p, view(), palette().color(QPalette::Text), palette().color(QPalette::Highlight), font(), font()};
	c.small.setPointSizeF(std::max(7.0, font().pointSizeF() - 1.5));
	drawGrid(c);
	if (!m_doc) return;
	drawFaces(c);
	drawNames(c);
	drawLights(c);
	drawCamera(c);
	drawTool(c);
}

// Floor grid and the world axes
void Scene3DView::drawGrid(Ctx &c) const {
	QColor gridColor = c.text;
	gridColor.setAlpha(26);
	const int half = 10;
	for (int i = -half; i <= half; ++i) {
		c.line({double(i), 0, double(-half)}, {double(i), 0, double(half)}, QPen(i == 0 ? QColor(72, 124, 232, 150) : gridColor, i == 0 ? 1.6 : 1.0));
		c.line({double(-half), 0, double(i)}, {double(half), 0, double(i)}, QPen(i == 0 ? QColor(232, 72, 72, 150) : gridColor, i == 0 ? 1.6 : 1.0));
	}
	c.line({0, 0, 0}, {0, 2, 0}, QPen(QColor(84, 200, 96, 150), 1.6));
}

// Faces back to front, with each mesh's sampled points among them (so a mesh behind something is hidden by it, like its box). Floor-like panels first; a
// flat panel goes by its farthest point, everything else by its centre.
void Scene3DView::drawFaces(Ctx &c) const {
	projectFaces(c.v);
	const std::vector<Face> &all = faces();
	struct Item { double key; bool floor; int face; int meshObject; };
	std::vector<Item> order;
	order.reserve(all.size() + 4);
	std::map<int, double> nearestOfMesh;  // object index -> depth of its nearest face: its points are drawn after the box faces
	for (int i = 0; i < static_cast<int>(all.size()); ++i) {
		const Face &f = all[i];
		if (f.screen.size() < 3) continue;
		order.push_back({f.flat ? f.farDepth : f.depth, f.bigFlat, i, -1});
		if (m_doc->objects[f.owner].shape == ShapeKind::Mesh) {
			auto it = nearestOfMesh.find(f.owner);
			if (it == nearestOfMesh.end()) nearestOfMesh[f.owner] = f.depth;
			else it->second = std::min(it->second, f.depth);
		}
	}
	for (const auto &kv : nearestOfMesh) order.push_back({kv.second - 1e-6, false, -1, kv.first});
	std::stable_sort(order.begin(), order.end(), [](const Item &a, const Item &b) {
		if (a.floor != b.floor) return a.floor;
		return a.key > b.key;
	});
	const V3 lightDir = scene_view::normalize({0.4, 0.8, 0.5});
	for (const Item &it : order) {
		if (it.face < 0) { drawMeshPoints(c, it.meshObject); continue; }
		const Face &f = all[it.face];
		V3 n = scene_view::cross(f.world[1] - f.world[0], f.world[2] - f.world[0]);
		n = scene_view::normalize(n);
		const double shade = 0.40 + 0.60 * std::abs(scene_view::dot(n, lightDir));
		QColor col = f.color;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == f.owner;
		const double k = m_doc->objects[f.owner].emissive ? 1.0 : shade;
		col = QColor(std::min(255, int(col.red() * k)), std::min(255, int(col.green() * k)), std::min(255, int(col.blue() * k)), col.alpha());
		QColor edge = col.darker(160);
		edge.setAlpha(120);
		c.p->setBrush(col);
		c.p->setPen(QPen(selected ? c.accent : edge, selected ? 1.6 : 0.8));
		c.p->drawPolygon(f.screen);
	}
}

// A mesh's sampled vertices, over its bounding box, so its shape shows
void Scene3DView::drawMeshPoints(Ctx &c, int i) const {
	const Object &o = m_doc->objects[i];
	const mesh_preview::MeshPreview *m = meshPreview(o.meshFile);
	if (!m) return;
	const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == i;
	QColor dot = selected ? c.accent : toQColor(o.material.color).lighter(130);
	dot.setAlpha(210);
	c.p->setPen(QPen(dot, 2.0, Qt::SolidLine, Qt::RoundCap));
	QList<QPointF> pts;
	for (std::size_t k = 0; k + 2 < m->samples.size(); k += 3) {
		const Float3 w = rotateXYZ(Float3{m->samples[k] * o.meshScale, m->samples[k + 1] * o.meshScale, m->samples[k + 2] * o.meshScale}, o.rotation);
		QPointF px;
		if (c.pixel({w.x + o.position.x, w.y + o.position.y, w.z + o.position.z}, px)) pts.append(px);
	}
	c.p->drawPoints(QPolygonF(pts.toVector()));
}

void Scene3DView::drawNames(Ctx &c) const {
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		QPointF at;
		if (!c.pixel(toV3(o.position), at)) continue;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == i;
		c.p->setBrush(selected ? c.accent : c.text);
		c.p->setPen(Qt::NoPen);
		c.p->drawEllipse(at, 2.5, 2.5);
		c.p->setFont(c.small);
		c.p->setPen(c.text);
		c.p->drawText(at + QPointF(6, -6), QString::fromStdString(o.name));
		c.p->setFont(c.normal);
	}
}

void Scene3DView::drawLights(Ctx &c) const {
	for (int i = 0; i < static_cast<int>(m_doc->lights.size()); ++i) {
		const Light &l = m_doc->lights[i];
		if (l.kind == LightKind::Infinite) continue;
		const bool selected = m_sel.kind == BuilderSelection::Kind::Light && m_sel.index == i;
		QPointF at;
		if (!c.pixel(toV3(l.position), at)) continue;
		if (l.kind == LightKind::Spot || l.kind == LightKind::Distant) {
			c.line(toV3(l.position), toV3(l.target), QPen(selected ? c.accent : QColor(230, 180, 40), 1.2, Qt::DashLine));
			QPointF t;
			if (c.pixel(toV3(l.target), t)) {
				c.p->setPen(QPen(selected ? c.accent : QColor(230, 180, 40), 1.2));
				c.p->drawLine(t - QPointF(4, 4), t + QPointF(4, 4));
				c.p->drawLine(t - QPointF(4, -4), t + QPointF(4, -4));
			}
		}
		c.p->setPen(QPen(selected ? c.accent : QColor(200, 150, 20), selected ? 2.5 : 1.5));
		c.p->setBrush(toQColor(l.color).lighter(130));
		c.p->drawEllipse(at, 6, 6);
		for (int k = 0; k < 8; ++k) {
			const double a = k * kPi / 4;
			c.p->drawLine(at + QPointF(std::cos(a) * 9, std::sin(a) * 9), at + QPointF(std::cos(a) * 13, std::sin(a) * 13));
		}
		c.p->setPen(c.text);
		c.p->setFont(c.small);
		c.p->drawText(at + QPointF(10, -10), QString::fromStdString(l.name));
		c.p->setFont(c.normal);
	}
}

// The camera: its position, what it looks at, and the frame it sees
void Scene3DView::drawCamera(Ctx &c) const {
	const bool selected = m_sel.kind == BuilderSelection::Kind::Camera;
	const QColor camColor = selected ? c.accent : QColor(70, 130, 200);
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
			c.line(pos, corner[k], pen);
			c.line(corner[k], corner[(k + 1) % 4], pen);
		}
		c.line(pos, tgt, QPen(camColor, 1.0, Qt::DashLine));
	}
	QPointF at, t;
	if (c.pixel(pos, at)) {
		c.p->setPen(QPen(camColor, selected ? 2.5 : 1.5));
		c.p->setBrush(camColor.lighter(160));
		c.p->drawRoundedRect(QRectF(at.x() - 9, at.y() - 6, 18, 12), 2, 2);
		c.p->drawEllipse(at, 4, 4);
		c.p->setPen(c.text);
		c.p->setFont(c.small);
		c.p->drawText(at + QPointF(10, 18), tr("Camera"));
		c.p->setFont(c.normal);
	}
	if (c.pixel(tgt, t)) {
		c.p->setPen(QPen(camColor, 1.5));
		c.p->setBrush(Qt::NoBrush);
		c.p->drawEllipse(t, 5, 5);
	}
}

// The selected item's tool: arrows (Move), rings (Rotate) or squares (Scale)
void Scene3DView::drawTool(Ctx &c) const {
	const Float3 *hp = handle(m_sel, 0);
	if (!hp) return;
	const V3 base = toV3(*hp);
	const double len = gizmoLength(c.v, base);
	const GizmoMode mode = effectiveGizmo();
	const Object *o = selectedObject();
	QPointF b;
	if (!c.pixel(base, b)) return;
	for (int a = 0; a < 3; ++a) {
		const bool active = (m_mode == Mode::Axis || m_mode == Mode::Rotate || m_mode == Mode::Scale) && m_drag.axis == a;
		if (mode == GizmoMode::Rotate) {
			const QPen pen(kAxisColor[a], active ? 3.6 : 2.2);
			V3 prev = ringAt(base, a, len * 0.9, 0);
			for (int k = 1; k <= kRingSegments; ++k) {
				const V3 cur = ringAt(base, a, len * 0.9, 360.0 * k / kRingSegments);
				c.line(prev, cur, pen);
				prev = cur;
			}
			QPointF e;
			if (c.pixel(ringAt(base, a, len * 0.9, 45), e)) {
				c.p->setPen(kAxisColor[a]);
				c.p->setFont(c.small);
				c.p->drawText(e + QPointF(6, -4), QString(QChar('X' + a)));
				c.p->setFont(c.normal);
			}
			continue;
		}
		if (mode == GizmoMode::Scale && o && !scene_gizmo::scaleHandleUsed(*o, a)) continue;
		const V3 dir = (mode == GizmoMode::Scale && o) ? localAxis(*o, a) : kAxisDir[a];
		const V3 tip = base + dir * len;
		QPointF e;
		if (!c.pixel(tip, e)) continue;
		c.line(base, tip, QPen(kAxisColor[a], active ? 4.0 : 2.6));
		QPointF d = e - b;
		const double dl = std::hypot(d.x(), d.y());
		c.p->setPen(Qt::NoPen);
		c.p->setBrush(kAxisColor[a]);
		if (mode == GizmoMode::Scale) {
			c.p->drawRect(QRectF(e.x() - 5, e.y() - 5, 10, 10));
		} else if (dl > 6) {
			d /= dl;
			const QPointF n(-d.y(), d.x());
			QPolygonF head;
			head << e + d * 4 << e - d * 8 + n * 5 << e - d * 8 - n * 5;
			c.p->drawPolygon(head);
		}
		c.p->setPen(kAxisColor[a]);
		c.p->setFont(c.small);
		c.p->drawText(e + QPointF(8, 4), QString(QChar('X' + a)));
		c.p->setFont(c.normal);
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
		// A click on nothing deselects; a drag on nothing orbits, or pans with Shift (for a trackpad, where a right-drag is awkward).
		m_mode = (e->modifiers() & Qt::ShiftModifier) ? Mode::Pan : Mode::Orbit;
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
		// Only the square at the end of a handle starts this (see pickTip), so t is about the handle's length and the scale starts at 1.
		m_mode = Mode::Scale;
		m_dragObject = *selectedObject();
		double t = 0;
		if (!scene_view::closestOnLine(ray, m_dragStart, localAxis(m_dragObject, h.axis), t) || std::abs(t) < 0.25 * gizmoLength(v, m_dragStart)) m_mode = Mode::None;
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

void Scene3DView::applyDrag(const QPointF &px, Qt::KeyboardModifiers mods) {
	const scene_view::View v = view();
	const scene_view::Ray ray = v.ray(px.x(), px.y());
	const bool snap = m_snap && !(mods & Qt::AltModifier);

	if (m_mode == Mode::Rotate) {
		double angle = 0;
		if (!scene_view::angleAround(ray, m_dragStart, kAxisDir[m_drag.axis], kRingRef[m_drag.axis], angle)) return;  // the ring is seen too edge-on: hold still
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
		emit objectEdited(m_drag.sel, scene_gizmo::scaledObject(m_dragObject, m_drag.axis, factor));
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
