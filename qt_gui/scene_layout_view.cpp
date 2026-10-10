// scene_layout_view.cpp - see scene_layout_view.h.
#include "scene_layout_view.h"

#include "scene_builder_common.h"

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

using namespace scene_builder_ui;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Convex hull of 2D points (monotone chain); returns fewer than 3 points for a degenerate (edge-on) outline.
QList<QPointF> convexHull(QList<QPointF> pts) {
	std::sort(pts.begin(), pts.end(), [](const QPointF &a, const QPointF &b) { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });
	pts.erase(std::unique(pts.begin(), pts.end(), [](const QPointF &a, const QPointF &b) { return a == b; }), pts.end());
	if (pts.size() < 3) return pts;
	auto cross = [](const QPointF &o, const QPointF &a, const QPointF &b) {
		return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
	};
	QList<QPointF> hull;
	for (const QPointF &p : pts) {
		while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0) hull.removeLast();
		hull.append(p);
	}
	const int lower = hull.size() + 1;
	for (int i = pts.size() - 2; i >= 0; --i) {
		while (hull.size() >= lower && cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0) hull.removeLast();
		hull.append(pts[i]);
	}
	hull.removeLast();
	return hull;
}

double distanceToSegment(const QPointF &p, const QPointF &a, const QPointF &b) {
	const QPointF ab = b - a;
	const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
	double t = len2 > 0 ? ((p.x() - a.x()) * ab.x() + (p.y() - a.y()) * ab.y()) / len2 : 0.0;
	t = std::clamp(t, 0.0, 1.0);
	const QPointF q = a + ab * t;
	return std::hypot(p.x() - q.x(), p.y() - q.y());
}

double polygonArea(const QList<QPointF> &poly) {
	double a = 0;
	for (int i = 0; i < poly.size(); ++i) {
		const QPointF &p = poly[i], &q = poly[(i + 1) % poly.size()];
		a += p.x() * q.y() - q.x() * p.y();
	}
	return std::abs(a) / 2.0;
}

} // namespace

// =====================================================================================================================================
// SceneLayoutView
// =====================================================================================================================================

SceneLayoutView::SceneLayoutView(QWidget *parent) : QWidget(parent) {
	setMouseTracking(false);
	setFocusPolicy(Qt::ClickFocus);
	setMinimumSize(280, 220);
	setAutoFillBackground(false);
}

void SceneLayoutView::setPlane(Plane p) {
	m_plane = p;
	frameAll();
}

QPointF SceneLayoutView::toUV(const Float3 &p) const {
	switch (m_plane) {
		case Plane::Top: return QPointF(p.x, p.z);
		case Plane::Front: return QPointF(p.x, -p.y);
		case Plane::Side: return QPointF(-p.z, -p.y);
	}
	return QPointF();
}

QPointF SceneLayoutView::toScreen(const Float3 &p) const {
	const QPointF uv = toUV(p);
	return QPointF(width() / 2.0 + (uv.x() - m_cu) * m_scale, height() / 2.0 + (uv.y() - m_cv) * m_scale);
}

Float3 SceneLayoutView::fromScreen(const QPointF &px, const Float3 &keep) const {
	const double u = (px.x() - width() / 2.0) / m_scale + m_cu;
	const double v = (px.y() - height() / 2.0) / m_scale + m_cv;
	Float3 w = keep;
	switch (m_plane) {
		case Plane::Top: w.x = u; w.z = v; break;
		case Plane::Front: w.x = u; w.y = -v; break;
		case Plane::Side: w.z = -u; w.y = -v; break;
	}
	return w;
}

Float3 SceneLayoutView::centerInWorld() const {
	return fromScreen(QPointF(width() / 2.0, height() / 2.0), Float3{0, 0, 0});
}

// The widget has no real size when the scene is first framed (the tab is built before it is laid out), so frame again as it is resized, until the
// user takes over with the wheel or by panning.
void SceneLayoutView::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	if (!m_userView) frameAll();
}

void SceneLayoutView::frameAll() {
	m_userView = false;
	if (!m_doc) return;
	double u0 = 1e18, u1 = -1e18, v0 = 1e18, v1 = -1e18;
	auto add = [&](const Float3 &p, double pad) {
		const QPointF uv = toUV(p);
		u0 = std::min(u0, uv.x() - pad); u1 = std::max(u1, uv.x() + pad);
		v0 = std::min(v0, uv.y() - pad); v1 = std::max(v1, uv.y() + pad);
	};
	add(m_doc->camera.position, 0.5);
	add(m_doc->camera.target, 0.2);
	for (const Object &o : m_doc->objects) {
		// A big floor would make everything else tiny: frame a bounded part of it.
		const double pad = std::min(o.shape == ShapeKind::Quad ? std::max(o.size.x, o.size.z) * 0.5 : std::max({o.radius, o.size.x, o.size.y, o.size.z}), 4.0);
		add(o.position, pad);
	}
	for (const Light &l : m_doc->lights)
		if (l.kind != LightKind::Infinite) add(l.position, 0.5);
	if (u0 > u1) { u0 = -4; u1 = 4; v0 = -3; v1 = 3; }
	m_cu = (u0 + u1) / 2.0;
	m_cv = (v0 + v1) / 2.0;
	const double sx = (width() - 40) / std::max(0.5, u1 - u0), sy = (height() - 40) / std::max(0.5, v1 - v0);
	m_scale = std::clamp(std::min(sx, sy), 6.0, 200.0);
	update();
}

QList<QPointF> SceneLayoutView::silhouette(const Object &o) const {
	QList<Float3> local;
	auto circle = [&](double r, double y, int n) {
		for (int i = 0; i < n; ++i) local.append(Float3{r * std::cos(2 * kPi * i / n), y, r * std::sin(2 * kPi * i / n)});
	};
	switch (o.shape) {
		case ShapeKind::Sphere: {
			QList<QPointF> poly;
			const QPointF c = toScreen(o.position);
			for (int i = 0; i < 40; ++i)
				poly.append(QPointF(c.x() + o.radius * m_scale * std::cos(2 * kPi * i / 40), c.y() + o.radius * m_scale * std::sin(2 * kPi * i / 40)));
			return poly;
		}
		case ShapeKind::Box:
			for (int i = 0; i < 8; ++i)
				local.append(Float3{(i & 1 ? 0.5 : -0.5) * o.size.x, (i & 2 ? 0.5 : -0.5) * o.size.y, (i & 4 ? 0.5 : -0.5) * o.size.z});
			break;
		case ShapeKind::Quad:
			for (int i = 0; i < 4; ++i) local.append(Float3{(i & 1 ? 0.5 : -0.5) * o.size.x, 0.0, (i & 2 ? 0.5 : -0.5) * o.size.z});
			break;
		case ShapeKind::Disk: circle(o.radius, 0.0, 36); break;
		case ShapeKind::Cylinder: circle(o.radius, -o.height / 2, 28); circle(o.radius, o.height / 2, 28); break;
		case ShapeKind::Cone: circle(o.radius, -o.height / 2, 28); local.append(Float3{0, o.height / 2, 0}); break;
		case ShapeKind::Pyramid:
		case ShapeKind::Wedge:
		case ShapeKind::Stairs:
		case ShapeKind::Torus:
		case ShapeKind::Capsule:
		case ShapeKind::Dome:
		case ShapeKind::Tube: {
			const scene_doc::ShapeMesh mesh = scene_doc::generatedMesh(o);
			for (std::size_t i = 0; i + 2 < mesh.P.size(); i += 3) local.append(Float3{mesh.P[i], mesh.P[i + 1], mesh.P[i + 2]});
			break;
		}
		case ShapeKind::Mesh: {
			const double r = 0.4 * o.meshScale;
			local = {Float3{-r, 0, 0}, Float3{r, 0, 0}, Float3{0, 0, -r}, Float3{0, 0, r}, Float3{0, r, 0}, Float3{0, -r, 0}};
			break;
		}
	}
	QList<QPointF> pts;
	for (const Float3 &p : local) {
		Float3 w = rotateXYZ(p, o.rotation);
		w.x += o.position.x; w.y += o.position.y; w.z += o.position.z;
		pts.append(toScreen(w));
	}
	return convexHull(pts);
}

QPointF SceneLayoutView::itemScreenPos(const BuilderSelection &s) const {
	const Float3 *p = handlePosition(s, 0);
	return p ? toScreen(*p) : QPointF();
}

const Float3 *builderHandle(const Document *doc, const BuilderSelection &s, int which) {
	if (!doc) return nullptr;
	switch (s.kind) {
		case BuilderSelection::Kind::Camera: return which == 0 ? &doc->camera.position : &doc->camera.target;
		case BuilderSelection::Kind::Object:
			return (which == 0 && s.index >= 0 && s.index < static_cast<int>(doc->objects.size())) ? &doc->objects[s.index].position : nullptr;
		case BuilderSelection::Kind::Light:
			if (s.index < 0 || s.index >= static_cast<int>(doc->lights.size())) return nullptr;
			{
				const Light &l = doc->lights[s.index];
				if (l.kind == LightKind::Infinite) return nullptr;
				if (which == 0) return &l.position;
				return (l.kind == LightKind::Spot || l.kind == LightKind::Distant) ? &l.target : nullptr;
			}
		case BuilderSelection::Kind::None: break;
	}
	return nullptr;
}

const Float3 *SceneLayoutView::handlePosition(const BuilderSelection &s, int which) const {
	return builderHandle(m_doc, s, which);
}

bool SceneLayoutView::isPicked(const BuilderSelection &s) const {
	if (s.kind == BuilderSelection::Kind::None) return false;
	if (s == m_sel) return true;
	return std::find(m_extra.begin(), m_extra.end(), s) != m_extra.end();
}

SceneLayoutView::Hit SceneLayoutView::hitTest(const QPointF &px) const {
	Hit h;
	if (!m_doc) return h;
	auto withinPx = [&](const Float3 &p, double r) { return std::hypot(toScreen(p).x() - px.x(), toScreen(p).y() - px.y()) <= r; };

	// The selected item's own handles first, so a target can be grabbed even when something is drawn over it.
	for (int which = 1; which >= 0; --which) {
		if (const Float3 *p = handlePosition(m_sel, which)) {
			if (withinPx(*p, 11)) { h.sel = m_sel; h.which = which; h.valid = true; return h; }
		}
	}
	// Lights and the camera next: they are small and would otherwise be hidden behind a floor.
	for (int i = static_cast<int>(m_doc->lights.size()) - 1; i >= 0; --i) {
		const Light &l = m_doc->lights[i];
		if (l.kind == LightKind::Infinite) continue;
		if (withinPx(l.position, 12)) { h.sel = {BuilderSelection::Kind::Light, i}; h.valid = true; return h; }
	}
	if (withinPx(m_doc->camera.position, 14)) { h.sel = {BuilderSelection::Kind::Camera, 0}; h.valid = true; return h; }
	if (withinPx(m_doc->camera.target, 10) && m_sel.kind == BuilderSelection::Kind::Camera) { h.sel = {BuilderSelection::Kind::Camera, 0}; h.which = 1; h.valid = true; return h; }

	// Objects: of everything under the pointer, the smallest outline (a ball on a floor picks the ball).
	double best = 1e30;
	for (int i = static_cast<int>(m_doc->objects.size()) - 1; i >= 0; --i) {
		const QList<QPointF> poly = silhouette(m_doc->objects[i]);
		bool inside = false;
		if (poly.size() >= 3) inside = QPolygonF(poly.toVector()).containsPoint(px, Qt::OddEvenFill);
		if (!inside) {
			for (int k = 0; k < poly.size() && !inside; ++k) {
				const QPointF &a = poly[k], &b = poly[(k + 1) % poly.size()];
				if (distanceToSegment(px, a, b) <= 5.0) inside = true;
			}
		}
		if (!inside) continue;
		const double area = poly.size() >= 3 ? polygonArea(poly) : 0.0;
		if (area < best) { best = area; h.sel = {BuilderSelection::Kind::Object, i}; h.valid = true; }
	}
	return h;
}

void SceneLayoutView::paintEvent(QPaintEvent *) {
	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing, true);
	p.fillRect(rect(), palette().color(QPalette::Base));
	const QColor text = palette().color(QPalette::Text);
	const QColor accent = palette().color(QPalette::Highlight);

	// Grid
	double step = 1.0;
	for (double s : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
		step = s;
		if (s * m_scale >= 26) break;
	}
	QColor gridColor = text; gridColor.setAlpha(28);
	QColor axisColor = text; axisColor.setAlpha(90);
	const double uLeft = m_cu - width() / 2.0 / m_scale, uRight = m_cu + width() / 2.0 / m_scale;
	const double vTop = m_cv - height() / 2.0 / m_scale, vBottom = m_cv + height() / 2.0 / m_scale;
	p.setPen(QPen(gridColor, 1));
	for (double u = std::floor(uLeft / step) * step; u <= uRight; u += step) {
		const double x = width() / 2.0 + (u - m_cu) * m_scale;
		p.drawLine(QPointF(x, 0), QPointF(x, height()));
	}
	for (double v = std::floor(vTop / step) * step; v <= vBottom; v += step) {
		const double y = height() / 2.0 + (v - m_cv) * m_scale;
		p.drawLine(QPointF(0, y), QPointF(width(), y));
	}
	p.setPen(QPen(axisColor, 1.5));
	{
		const double x0 = width() / 2.0 - m_cu * m_scale, y0 = height() / 2.0 - m_cv * m_scale;
		p.drawLine(QPointF(x0, 0), QPointF(x0, height()));
		p.drawLine(QPointF(0, y0), QPointF(width(), y0));
	}
	const QString planeText = m_plane == Plane::Top ? tr("Top view: X to the right, Z towards you (down)")
	                         : m_plane == Plane::Front ? tr("Front view: X to the right, Y up")
	                                                    : tr("Side view: Z to the left, Y up");
	p.setPen(text);
	p.drawText(8, 16, planeText);
	p.drawText(8, height() - 8, step == 1.0 ? tr("Grid: 1 unit") : tr("Grid: %1 units").arg(step));

	if (!m_doc) return;

	// Objects
	QFont small = font();
	small.setPointSizeF(std::max(7.0, font().pointSizeF() - 1.5));
	for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i) {
		const Object &o = m_doc->objects[i];
		const QList<QPointF> poly = silhouette(o);
		QColor fill = toQColor(o.material.color);
		if (o.material.kind == MaterialKind::Dielectric) fill = QColor(150, 200, 240);
		if (o.emissive) fill = QColor(255, 214, 110);
		fill.setAlpha(o.shape == ShapeKind::Quad && o.size.x * o.size.z > 30 ? 60 : 140);
		const bool selected = isPicked({BuilderSelection::Kind::Object, i});
		p.setBrush(fill);
		p.setPen(QPen(selected ? accent : text, selected ? 2.5 : 1.0));
		if (poly.size() >= 3) p.drawPolygon(QPolygonF(poly.toVector()));
		else if (poly.size() == 2) { p.setPen(QPen(selected ? accent : fill.darker(), selected ? 4.0 : 3.0)); p.drawLine(poly[0], poly[1]); }
		const QPointF c = toScreen(o.position);
		p.setBrush(selected ? accent : text);
		p.setPen(Qt::NoPen);
		p.drawEllipse(c, 2.5, 2.5);
		if (m_scale >= 14 || selected) {
			p.setFont(small);
			p.setPen(text);
			p.drawText(c + QPointF(6, -6), QString::fromStdString(o.name));
			p.setFont(font());
		}
	}

	// Lights
	for (int i = 0; i < static_cast<int>(m_doc->lights.size()); ++i) {
		const Light &l = m_doc->lights[i];
		if (l.kind == LightKind::Infinite) continue;
		const bool selected = isPicked({BuilderSelection::Kind::Light, i});
		const QPointF c = toScreen(l.position);
		if (l.kind == LightKind::Spot || l.kind == LightKind::Distant) {
			const QPointF t = toScreen(l.target);
			p.setPen(QPen(selected ? accent : QColor(230, 180, 40), 1.2, Qt::DashLine));
			p.drawLine(c, t);
			p.drawLine(t - QPointF(4, 4), t + QPointF(4, 4));
			p.drawLine(t - QPointF(4, -4), t + QPointF(4, -4));
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

	// Camera, with its view direction and field of view
	{
		const bool selected = m_sel.kind == BuilderSelection::Kind::Camera;
		const QPointF c = toScreen(m_doc->camera.position), t = toScreen(m_doc->camera.target);
		QPointF dir = t - c;
		const double len = std::hypot(dir.x(), dir.y());
		const QColor camColor = selected ? accent : QColor(70, 130, 200);
		if (len > 1e-3) {
			dir /= len;
			const double aspect = m_doc->render.height > 0 ? double(m_doc->render.width) / m_doc->render.height : 1.0;
			const double half = (m_plane == Plane::Top ? std::atan(std::tan(m_doc->camera.fov * kPi / 360.0) * aspect) : m_doc->camera.fov * kPi / 360.0);
			const double reach = std::min(len, 90.0);
			p.setPen(QPen(camColor, 1.0, Qt::DashLine));
			for (double sgn : {-1.0, 1.0}) {
				const double a = sgn * half, ca = std::cos(a), sa = std::sin(a);
				const QPointF d(dir.x() * ca - dir.y() * sa, dir.x() * sa + dir.y() * ca);
				p.drawLine(c, c + d * reach);
			}
			p.setPen(QPen(camColor, 1.5));
			p.drawLine(c, t);
			p.drawEllipse(t, 5, 5);
		}
		QPolygonF body;
		const QPointF n = len > 1e-3 ? QPointF(-dir.y(), dir.x()) : QPointF(0, 1);
		const QPointF d = len > 1e-3 ? dir : QPointF(1, 0);
		body << c - d * 9 + n * 7 << c - d * 9 - n * 7 << c + d * 9 - n * 4 << c + d * 9 + n * 4;
		p.setPen(QPen(camColor, selected ? 2.5 : 1.5));
		p.setBrush(camColor.lighter(160));
		p.drawPolygon(body);
		p.setPen(text);
		p.setFont(small);
		p.drawText(c + QPointF(10, 18), tr("Camera"));
		p.setFont(font());
	}

	if (m_banding) {
		QColor fill = accent;
		fill.setAlpha(40);
		p.setBrush(fill);
		p.setPen(QPen(accent, 1.0, Qt::DashLine));
		p.drawRect(QRectF(m_bandStart, m_bandEnd).normalized());
	}
}

void SceneLayoutView::mousePressEvent(QMouseEvent *e) {
	setFocus();
	const QPointF px = e->position();
	if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
		m_panning = true;
		m_lastPan = px;
		return;
	}
	if (e->button() != Qt::LeftButton) return;
	const bool additive = (e->modifiers() & (Qt::ShiftModifier | Qt::ControlModifier)) != 0;   // Control is the Command key on a Mac
	const Hit h = hitTest(px);
	if (additive && (!h.valid || h.sel.kind != BuilderSelection::Kind::Camera)) {
		// Ctrl or Shift: a click on an item adds it or takes it out; a drag from anywhere (a floor under everything included) is a box that picks what it holds.
		m_banding = false;
		m_bandAdditive = true;
		m_bandStart = m_bandEnd = px;
		m_toggleCandidate = h.valid ? h.sel : BuilderSelection{};
		m_maybeToggle = true;
		return;
	}
	if (!h.valid) {
		// A click on nothing deselects; a drag on nothing pans.
		m_panning = true;
		m_lastPan = px;
		emit selectionRequested(BuilderSelection{});
		return;
	}
	// Pressing one of several picked items drags them all; a click that does not move leaves just that one picked, as in a file manager.
	if (!m_extra.empty() && isPicked(h.sel) && h.which == 0) {
		m_collapseTo = h.sel;
		m_collapsePending = true;
	} else {
		emit selectionRequested(h.sel);
	}
	m_drag = h;
	m_dragging = true;
	emit dragBegan();
	if (const Float3 *pos = handlePosition(h.sel, h.which)) {
		const QPointF itemPx = toScreen(*pos);
		m_dragOffsetUV = QPointF((itemPx.x() - px.x()) / m_scale, (itemPx.y() - px.y()) / m_scale);
	}
}

void SceneLayoutView::mouseMoveEvent(QMouseEvent *e) {
	const QPointF px = e->position();
	if (m_maybeToggle) {
		m_bandEnd = px;
		if (std::hypot(px.x() - m_bandStart.x(), px.y() - m_bandStart.y()) < 4.0) return;   // still a click
		m_maybeToggle = false;
		m_banding = true;
	}
	if (m_banding) {
		m_bandEnd = px;
		update();
		return;
	}
	if (m_panning) {
		m_userView = true;
		m_cu -=(px.x() - m_lastPan.x()) / m_scale;
		m_cv -= (px.y() - m_lastPan.y()) / m_scale;
		m_lastPan = px;
		update();
		return;
	}
	if (!m_dragging || !m_drag.valid) return;
	const Float3 *current = handlePosition(m_drag.sel, m_drag.which);
	if (!current) return;
	m_collapsePending = false;   // it moved: the picked items stay picked
	QPointF target(px.x() + m_dragOffsetUV.x() * m_scale, px.y() + m_dragOffsetUV.y() * m_scale);
	Float3 w = fromScreen(target, *current);
	if (m_snap && !(e->modifiers() & Qt::AltModifier)) {
		const double q = 0.25;
		auto snap = [q](double v) { return std::round(v / q) * q; };
		switch (m_plane) {
			case Plane::Top: w.x = snap(w.x); w.z = snap(w.z); break;
			case Plane::Front: w.x = snap(w.x); w.y = snap(w.y); break;
			case Plane::Side: w.z = snap(w.z); w.y = snap(w.y); break;
		}
	}
	emit positionDragged(m_drag.sel, m_drag.which, w);
}

void SceneLayoutView::mouseReleaseEvent(QMouseEvent *) {
	if (m_maybeToggle) {
		m_maybeToggle = false;
		if (m_toggleCandidate.kind != BuilderSelection::Kind::None) emit selectionToggled(m_toggleCandidate);
		return;
	}
	if (m_banding) {
		m_banding = false;
		const QRectF box = QRectF(m_bandStart, m_bandEnd).normalized();
		QList<BuilderSelection> inside;
		if (m_doc) {
			for (int i = 0; i < static_cast<int>(m_doc->objects.size()); ++i)
				if (box.contains(toScreen(m_doc->objects[i].position))) inside.append({BuilderSelection::Kind::Object, i});
			for (int i = 0; i < static_cast<int>(m_doc->lights.size()); ++i)
				if (m_doc->lights[i].kind != LightKind::Infinite && box.contains(toScreen(m_doc->lights[i].position))) inside.append({BuilderSelection::Kind::Light, i});
		}
		update();
		emit boxSelected(inside, m_bandAdditive);
		return;
	}
	if (m_collapsePending) {
		m_collapsePending = false;
		emit selectionRequested(m_collapseTo);
	}
	m_panning = false;
	m_dragging = false;
	m_drag = Hit{};
}

void SceneLayoutView::wheelEvent(QWheelEvent *e) {
	m_userView = true;
	const double factor = std::pow(1.15, e->angleDelta().y() / 120.0);
	const QPointF px = e->position();
	// Keep the point under the pointer where it is.
	const double uBefore = (px.x() - width() / 2.0) / m_scale + m_cu, vBefore = (px.y() - height() / 2.0) / m_scale + m_cv;
	m_scale = std::clamp(m_scale * factor, 6.0, 400.0);
	m_cu = uBefore - (px.x() - width() / 2.0) / m_scale;
	m_cv = vBefore - (px.y() - height() / 2.0) / m_scale;
	update();
	e->accept();
}
