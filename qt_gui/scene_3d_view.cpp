// scene_3d_view.cpp - see scene_3d_view.h.
#include "scene_3d_view.h"

#include "scene_3d_view_internal.h"
#include "app_log.h"

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
	GizmoMode wanted = m_gizmo;
	const char *name = "";
	switch (e->key()) {
		case Qt::Key_W: wanted = GizmoMode::Move; name = "Move (W key)"; break;
		case Qt::Key_E: wanted = GizmoMode::Rotate; name = "Rotate (E key)"; break;
		case Qt::Key_R: wanted = GizmoMode::Scale; name = "Scale (R key)"; break;
		default: QWidget::keyPressEvent(e); return;
	}
	if (wanted != m_gizmo) AppLog::info(QStringLiteral("builder"), QStringLiteral("3D view: tool %1").arg(QString::fromLatin1(name)));
	setGizmoMode(wanted);
	e->accept();
}

void Scene3DView::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	if (!m_userView) frameAll();
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

