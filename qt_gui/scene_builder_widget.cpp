// scene_builder_widget.cpp - see scene_builder_widget.h.
#include "scene_builder_widget.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
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

namespace {

constexpr double kPi = 3.14159265358979323846;

double toSrgb(double c) {
	c = std::clamp(c, 0.0, 1.0);
	return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
double toLinear(double s) {
	s = std::clamp(s, 0.0, 1.0);
	return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}
// Colours are stored linear (what the renderer wants) and picked and drawn as sRGB (what a colour picker shows).
QColor toQColor(const Rgb &c) { return QColor::fromRgbF(toSrgb(c.r), toSrgb(c.g), toSrgb(c.b)); }
Rgb fromQColor(const QColor &q) { return Rgb{toLinear(q.redF()), toLinear(q.greenF()), toLinear(q.blueF())}; }

// Rotates p by the object's three angles, in the order the writer applies them: about X, then Y, then Z.
Float3 rotateXYZ(Float3 p, const Float3 &deg) {
	auto rad = [](double d) { return d * kPi / 180.0; };
	if (deg.x != 0.0) {
		const double c = std::cos(rad(deg.x)), s = std::sin(rad(deg.x));
		const double y = p.y * c - p.z * s, z = p.y * s + p.z * c;
		p.y = y; p.z = z;
	}
	if (deg.y != 0.0) {
		const double c = std::cos(rad(deg.y)), s = std::sin(rad(deg.y));
		const double x = p.x * c + p.z * s, z = -p.x * s + p.z * c;
		p.x = x; p.z = z;
	}
	if (deg.z != 0.0) {
		const double c = std::cos(rad(deg.z)), s = std::sin(rad(deg.z));
		const double x = p.x * c - p.y * s, y = p.x * s + p.y * c;
		p.x = x; p.y = y;
	}
	return p;
}

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

QString shapeLabel(ShapeKind k) {
	switch (k) {
		case ShapeKind::Sphere: return QObject::tr("Sphere");
		case ShapeKind::Box: return QObject::tr("Box");
		case ShapeKind::Quad: return QObject::tr("Quad (flat panel)");
		case ShapeKind::Disk: return QObject::tr("Disk");
		case ShapeKind::Cylinder: return QObject::tr("Cylinder");
		case ShapeKind::Cone: return QObject::tr("Cone");
		case ShapeKind::Mesh: return QObject::tr("Mesh (.ply file)");
	}
	return QString();
}
QString lightLabel(LightKind k) {
	switch (k) {
		case LightKind::Point: return QObject::tr("Point light");
		case LightKind::Spot: return QObject::tr("Spotlight");
		case LightKind::Distant: return QObject::tr("Sun (distant light)");
		case LightKind::Infinite: return QObject::tr("Sky (surrounds the scene)");
	}
	return QString();
}
QString materialLabel(MaterialKind k) {
	switch (k) {
		case MaterialKind::Diffuse: return QObject::tr("Matte (diffuse)");
		case MaterialKind::Conductor: return QObject::tr("Metal");
		case MaterialKind::Dielectric: return QObject::tr("Glass");
		case MaterialKind::CoatedDiffuse: return QObject::tr("Glossy paint (coated)");
		case MaterialKind::DiffuseTransmission: return QObject::tr("Translucent (paper, leaves)");
	}
	return QString();
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

void SceneLayoutView::frameAll() {
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

const Float3 *SceneLayoutView::handlePosition(const BuilderSelection &s, int which) const {
	if (!m_doc) return nullptr;
	switch (s.kind) {
		case BuilderSelection::Kind::Camera: return which == 0 ? &m_doc->camera.position : &m_doc->camera.target;
		case BuilderSelection::Kind::Object:
			return (which == 0 && s.index >= 0 && s.index < static_cast<int>(m_doc->objects.size())) ? &m_doc->objects[s.index].position : nullptr;
		case BuilderSelection::Kind::Light:
			if (s.index < 0 || s.index >= static_cast<int>(m_doc->lights.size())) return nullptr;
			{
				const Light &l = m_doc->lights[s.index];
				if (l.kind == LightKind::Infinite) return nullptr;
				if (which == 0) return &l.position;
				return (l.kind == LightKind::Spot || l.kind == LightKind::Distant) ? &l.target : nullptr;
			}
		case BuilderSelection::Kind::None: break;
	}
	return nullptr;
}

SceneLayoutView::Hit SceneLayoutView::hitTest(const QPointF &px) const {
	Hit h;
	if (!m_doc) return h;
	auto near = [&](const Float3 &p, double r) { return std::hypot(toScreen(p).x() - px.x(), toScreen(p).y() - px.y()) <= r; };

	// The selected item's own handles first, so a target can be grabbed even when something is drawn over it.
	for (int which = 1; which >= 0; --which) {
		if (const Float3 *p = handlePosition(m_sel, which)) {
			if (near(*p, 11)) { h.sel = m_sel; h.which = which; h.valid = true; return h; }
		}
	}
	// Lights and the camera next: they are small and would otherwise be hidden behind a floor.
	for (int i = static_cast<int>(m_doc->lights.size()) - 1; i >= 0; --i) {
		const Light &l = m_doc->lights[i];
		if (l.kind == LightKind::Infinite) continue;
		if (near(l.position, 12)) { h.sel = {BuilderSelection::Kind::Light, i}; h.valid = true; return h; }
	}
	if (near(m_doc->camera.position, 14)) { h.sel = {BuilderSelection::Kind::Camera, 0}; h.valid = true; return h; }
	if (near(m_doc->camera.target, 10) && m_sel.kind == BuilderSelection::Kind::Camera) { h.sel = {BuilderSelection::Kind::Camera, 0}; h.which = 1; h.valid = true; return h; }

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
	const char *planeText = m_plane == Plane::Top ? "Top view: X to the right, Z towards you (down)"
	                       : m_plane == Plane::Front ? "Front view: X to the right, Y up"
	                                                  : "Side view: Z to the left, Y up";
	p.setPen(text);
	p.drawText(8, 16, tr(planeText));
	p.drawText(8, height() - 8, tr("Grid: %1 unit%2").arg(step).arg(step == 1.0 ? "" : "s"));

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
		const bool selected = m_sel.kind == BuilderSelection::Kind::Object && m_sel.index == i;
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
		const bool selected = m_sel.kind == BuilderSelection::Kind::Light && m_sel.index == i;
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
			const double reach = std::min(len, 90.0 + len * 0.0);
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
	const Hit h = hitTest(px);
	if (!h.valid) {
		// A click on nothing deselects; a drag on nothing pans.
		m_panning = true;
		m_lastPan = px;
		emit selectionRequested(BuilderSelection{});
		return;
	}
	emit selectionRequested(h.sel);
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
	if (m_panning) {
		m_cu -= (px.x() - m_lastPan.x()) / m_scale;
		m_cv -= (px.y() - m_lastPan.y()) / m_scale;
		m_lastPan = px;
		update();
		return;
	}
	if (!m_dragging || !m_drag.valid) return;
	const Float3 *current = handlePosition(m_drag.sel, m_drag.which);
	if (!current) return;
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
	m_panning = false;
	m_dragging = false;
	m_drag = Hit{};
}

void SceneLayoutView::wheelEvent(QWheelEvent *e) {
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

// =====================================================================================================================================
// SceneBuilderWidget
// =====================================================================================================================================

SceneBuilderWidget::SceneBuilderWidget(QWidget *parent) : QWidget(parent) {
	m_autosaveEnabled = !qEnvironmentVariableIsSet("RT_GUI_SELFTEST");
	m_autosaveTimer = new QTimer(this);
	m_autosaveTimer->setSingleShot(true);
	m_autosaveTimer->setInterval(1500);
	connect(m_autosaveTimer, &QTimer::timeout, this, &SceneBuilderWidget::writeAutosave);

	buildUi();
	if (!(m_autosaveEnabled && loadAutosave())) newScene();
}

SceneBuilderWidget::~SceneBuilderWidget() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
	if (m_dirty && m_autosaveEnabled) writeAutosave();
}

QString SceneBuilderWidget::workFolder() const {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/ray_tracer_scene_builder";
	QDir().mkpath(dir);
	return dir;
}

QString SceneBuilderWidget::launcherPath() const {
#ifdef Q_OS_WIN
	return QCoreApplication::applicationDirPath() + "/ray_tracer.exe";
#else
	return QCoreApplication::applicationDirPath() + "/ray_tracer";
#endif
}

void SceneBuilderWidget::buildUi() {
	auto *root = new QVBoxLayout(this);

	// ---- top bar
	auto *bar = new QHBoxLayout;
	auto button = [this](const QString &text, const QString &tip) {
		auto *b = new QPushButton(text, this);
		b->setToolTip(tip);
		b->setAutoDefault(false);
		return b;
	};
	auto *newB = button(tr("New"), tr("Start again from the example scene"));
	auto *openB = button(tr("Open..."), tr("Open a .pbrt file saved by the Scene Builder"));
	auto *saveB = button(tr("Save"), tr("Save the scene as a .pbrt file"));
	auto *saveAsB = button(tr("Save As..."), tr("Save the scene under a new name"));
	auto *listB = button(tr("Add to scene list"), tr("Save the scene into the scenes folder so it shows up in the Settings tab (after a restart)"));
	m_undoButton = button(tr("Undo"), tr("Undo the last change (Ctrl+Z)"));
	m_redoButton = button(tr("Redo"), tr("Redo (Ctrl+Y)"));
	m_titleLabel = new QLabel(this);
	m_titleLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
	for (QPushButton *b : {newB, openB, saveB, saveAsB, listB}) bar->addWidget(b);
	bar->addSpacing(12);
	bar->addWidget(m_undoButton);
	bar->addWidget(m_redoButton);
	bar->addStretch(1);
	bar->addWidget(m_titleLabel);
	root->addLayout(bar);
	connect(newB, &QPushButton::clicked, this, [this]() { if (confirmDiscard()) newScene(); });
	connect(openB, &QPushButton::clicked, this, &SceneBuilderWidget::onOpenClicked);
	connect(saveB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveClicked);
	connect(saveAsB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveAsClicked);
	connect(listB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveToSceneListClicked);
	connect(m_undoButton, &QPushButton::clicked, this, [this]() { undo(); });
	connect(m_redoButton, &QPushButton::clicked, this, [this]() { redo(); });

	auto *split = new QSplitter(Qt::Horizontal, this);
	root->addWidget(split, 1);

	// ---- left: scene contents
	auto *left = new QWidget(split);
	auto *leftLayout = new QVBoxLayout(left);
	leftLayout->setContentsMargins(0, 0, 0, 0);
	auto *addB = new QPushButton(tr("Add"), left);
	addB->setAutoDefault(false);
	auto *addMenu = new QMenu(addB);
	addMenu->addSection(tr("Objects"));
	for (ShapeKind k : {ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Quad, ShapeKind::Disk, ShapeKind::Cylinder, ShapeKind::Cone, ShapeKind::Mesh})
		addMenu->addAction(shapeLabel(k), this, [this, k]() { addObject(k); });
	addMenu->addAction(tr("Light panel (emitting quad)"), this, [this]() {
		edit(QString(), [this]() {
			scene_doc::Object o = scene_doc::makeAreaLightPanel(QString("Light panel %1").arg(m_doc.objects.size() + 1).toStdString());
			m_doc.objects.push_back(o);
			m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
		});
		rebuildList();
		setSelection(m_sel);
	});
	addMenu->addSection(tr("Lights"));
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite})
		addMenu->addAction(lightLabel(k), this, [this, k]() { addLight(k); });
	addB->setMenu(addMenu);
	m_duplicateButton = new QPushButton(tr("Duplicate"), left);
	m_deleteButton = new QPushButton(tr("Delete"), left);
	m_duplicateButton->setAutoDefault(false);
	m_deleteButton->setAutoDefault(false);
	auto *row = new QHBoxLayout;
	row->addWidget(addB);
	row->addWidget(m_duplicateButton);
	row->addWidget(m_deleteButton);
	leftLayout->addLayout(row);
	m_list = new QListWidget(left);
	leftLayout->addWidget(m_list, 1);
	left->setMinimumWidth(300);
	connect(m_list, &QListWidget::currentRowChanged, this, [this](int) { onListSelectionChanged(); });
	connect(m_deleteButton, &QPushButton::clicked, this, [this]() { deleteSelected(); });
	connect(m_duplicateButton, &QPushButton::clicked, this, [this]() {
		if (m_sel.kind == SelKind::Object && m_sel.index < static_cast<int>(m_doc.objects.size())) {
			edit(QString(), [this]() {
				Object o = m_doc.objects[m_sel.index];
				o.name += " copy";
				o.position.x += 0.5;
				m_doc.objects.push_back(o);
				m_sel.index = static_cast<int>(m_doc.objects.size()) - 1;
			});
		} else if (m_sel.kind == SelKind::Light && m_sel.index < static_cast<int>(m_doc.lights.size())) {
			edit(QString(), [this]() {
				Light l = m_doc.lights[m_sel.index];
				l.name += " copy";
				l.position.x += 0.5;
				m_doc.lights.push_back(l);
				m_sel.index = static_cast<int>(m_doc.lights.size()) - 1;
			});
		} else {
			return;
		}
		rebuildList();
		setSelection(m_sel);
	});
	// Delete and undo work from the list and the layout view (not inside a text box, where Delete edits text).
	auto *delList = new QShortcut(QKeySequence::Delete, m_list);
	delList->setContext(Qt::WidgetShortcut);
	connect(delList, &QShortcut::activated, this, [this]() { deleteSelected(); });

	// ---- centre: layout view above the preview
	auto *centre = new QSplitter(Qt::Vertical, split);
	auto *layoutBox = new QWidget(centre);
	auto *layoutLayout = new QVBoxLayout(layoutBox);
	layoutLayout->setContentsMargins(0, 0, 0, 0);
	auto *planeRow = new QHBoxLayout;
	auto *planeGroup = new QButtonGroup(this);
	planeGroup->setExclusive(true);
	const std::pair<QString, SceneLayoutView::Plane> planes[] = {{tr("Top"), SceneLayoutView::Plane::Top}, {tr("Front"), SceneLayoutView::Plane::Front}, {tr("Side"), SceneLayoutView::Plane::Side}};
	m_view = new SceneLayoutView(layoutBox);
	for (const auto &pl : planes) {
		auto *b = new QPushButton(pl.first, layoutBox);
		b->setCheckable(true);
		b->setAutoDefault(false);
		b->setChecked(pl.second == SceneLayoutView::Plane::Top);
		planeGroup->addButton(b);
		planeRow->addWidget(b);
		connect(b, &QPushButton::clicked, this, [this, pl]() { m_view->setPlane(pl.second); });
	}
	auto *snap = new QCheckBox(tr("Snap to grid"), layoutBox);
	snap->setChecked(true);
	snap->setToolTip(tr("Dragging moves things in steps of 0.25. Hold Alt to drag freely."));
	connect(snap, &QCheckBox::toggled, this, [this](bool on) { m_view->setSnap(on); });
	auto *frame = new QPushButton(tr("Frame all"), layoutBox);
	frame->setAutoDefault(false);
	connect(frame, &QPushButton::clicked, this, [this]() { m_view->frameAll(); });
	auto *hint = new QLabel(tr("Wheel: zoom. Right-drag: pan."), layoutBox);
	planeRow->addSpacing(8);
	planeRow->addWidget(snap);
	planeRow->addWidget(frame);
	planeRow->addStretch(1);
	planeRow->addWidget(hint);
	layoutLayout->addLayout(planeRow);
	layoutLayout->addWidget(m_view, 1);
	m_view->setDocument(&m_doc);
	connect(m_view, &SceneLayoutView::selectionRequested, this, [this](const BuilderSelection &s) { setSelection(s); });
	connect(m_view, &SceneLayoutView::dragBegan, this, [this]() {
		// A drag is one undo step however many mouse events it takes; the first move takes the snapshot.
		m_lastEditKey.clear();
		m_editCounter++;
	});
	connect(m_view, &SceneLayoutView::positionDragged, this, [this](const BuilderSelection &s, int which, const Float3 &w) {
		edit(QString("drag#%1").arg(m_editCounter), [&]() {
			Float3 *target = nullptr;
			switch (s.kind) {
				case SelKind::Camera: target = which == 0 ? &m_doc.camera.position : &m_doc.camera.target; break;
				case SelKind::Object: if (s.index < static_cast<int>(m_doc.objects.size())) target = &m_doc.objects[s.index].position; break;
				case SelKind::Light:
					if (s.index < static_cast<int>(m_doc.lights.size())) target = which == 0 ? &m_doc.lights[s.index].position : &m_doc.lights[s.index].target;
					break;
				case SelKind::None: break;
			}
			if (target) *target = w;
		});
		refreshInspectorValues();
	});
	auto *undoView = new QShortcut(QKeySequence::Undo, m_view);
	undoView->setContext(Qt::WidgetShortcut);
	connect(undoView, &QShortcut::activated, this, [this]() { undo(); });
	auto *redoView = new QShortcut(QKeySequence::Redo, m_view);
	redoView->setContext(Qt::WidgetShortcut);
	connect(redoView, &QShortcut::activated, this, [this]() { redo(); });
	auto *delView = new QShortcut(QKeySequence::Delete, m_view);
	delView->setContext(Qt::WidgetShortcut);
	connect(delView, &QShortcut::activated, this, [this]() { deleteSelected(); });

	auto *previewBox = new QWidget;
	auto *previewLayout = new QVBoxLayout(previewBox);
	previewLayout->setContentsMargins(0, 0, 0, 0);
	auto *renderRow = new QHBoxLayout;
	m_qualityCombo = new QComboBox(previewBox);
	m_qualityCombo->addItem(tr("Draft"), 0);
	m_qualityCombo->addItem(tr("Good"), 1);
	m_qualityCombo->addItem(tr("Best"), 2);
	m_qualityCombo->setCurrentIndex(0);
	m_qualityCombo->setToolTip(tr("Draft: 320 pixels wide, 16 samples. Good: 480 wide, 64 samples. Best: 640 wide, 256 samples."));
	m_gpuCheck = new QCheckBox(tr("Use the GPU"), previewBox);
	m_gpuCheck->setToolTip(tr("Render on the graphics card (NVIDIA OptiX on Windows, Metal on a Mac). Much faster for large pictures; needs a supported GPU."));
	m_previewButton = new QPushButton(tr("Preview"), previewBox);
	m_previewButton->setAutoDefault(false);
	m_finalButton = new QPushButton(tr("Render picture..."), previewBox);
	m_finalButton->setAutoDefault(false);
	m_finalButton->setToolTip(tr("Render at the image size and sample count set under Camera, and save the picture as a PNG"));
	m_previewStatus = new QLabel(previewBox);
	renderRow->addWidget(new QLabel(tr("Quality:"), previewBox));
	renderRow->addWidget(m_qualityCombo);
	renderRow->addWidget(m_gpuCheck);
	renderRow->addWidget(m_previewButton, 1);
	renderRow->addWidget(m_finalButton, 1);
	m_previewStatus->setWordWrap(true);
	previewLayout->addLayout(renderRow);
	m_previewLabel = new QLabel(previewBox);
	m_previewLabel->setAlignment(Qt::AlignCenter);
	m_previewLabel->setMinimumHeight(120);
	m_previewLabel->installEventFilter(this);  // rescale the picture when the pane is resized
	m_previewLabel->setText(tr("Press Preview to see the scene."));
	m_previewLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	previewLayout->addWidget(m_previewLabel, 1);
	previewLayout->addWidget(m_previewStatus);
	connect(m_previewButton, &QPushButton::clicked, this, [this]() {
		if (m_process) {
			m_cancelRequested = true;
			m_process->kill();  // the button reads Cancel while a render runs
			return;
		}
		startPreview();
	});
	connect(m_finalButton, &QPushButton::clicked, this, &SceneBuilderWidget::onRenderFinalClicked);

	// ---- right: inspector
	auto *right = new QWidget(split);
	auto *inspectorPane = right;
	auto *rightLayout = new QVBoxLayout(inspectorPane);
	rightLayout->setContentsMargins(0, 0, 0, 0);
	m_inspectorScroll = new QScrollArea(inspectorPane);
	m_inspectorScroll->setWidgetResizable(true);
	m_inspectorScroll->setFrameShape(QFrame::NoFrame);
	rightLayout->addWidget(m_inspectorScroll, 1);
	m_problemsLabel = new QLabel(inspectorPane);
	m_problemsLabel->setWordWrap(true);
	m_problemsLabel->setTextFormat(Qt::RichText);
	rightLayout->addWidget(m_problemsLabel);
	right->setMinimumWidth(390);
	centre->addWidget(previewBox);
	centre->setStretchFactor(0, 3);
	centre->setStretchFactor(1, 2);
	centre->setSizes({500, 280});

	split->addWidget(left);
	split->addWidget(centre);
	split->addWidget(right);
	split->setStretchFactor(0, 0);
	split->setStretchFactor(1, 1);
	split->setStretchFactor(2, 0);
	split->setSizes({310, 600, 420});
}

// ---- document state -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::newScene() {
	m_doc = scene_doc::makeStarterScene();
	m_path.clear();
	m_dirty = false;
	m_undo.clear();
	m_redo.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	m_view->frameAll();
	refreshProblems();
	updateTitle();
	updateActions();
}

bool SceneBuilderWidget::openFile(const QString &path, QString *error) {
	QFile f(path);
	if (!f.open(QIODevice::ReadOnly)) {
		if (error) *error = tr("Cannot open %1.").arg(path);
		return false;
	}
	const std::string text = f.readAll().toStdString();
	Document d;
	std::string err;
	if (!scene_doc::fromPbrt(text, d, err)) {
		if (error) *error = QString::fromStdString(err);
		return false;
	}
	m_doc = std::move(d);
	m_path = path;
	m_dirty = false;
	m_undo.clear();
	m_redo.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	m_view->frameAll();
	refreshProblems();
	updateTitle();
	updateActions();
	return true;
}

bool SceneBuilderWidget::saveFile(const QString &path) {
	QFile f(path);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
	const std::string text = scene_doc::toPbrt(m_doc);
	if (f.write(text.data(), static_cast<qint64>(text.size())) != static_cast<qint64>(text.size())) return false;
	f.close();
	m_path = path;
	m_dirty = false;
	clearAutosave();
	updateTitle();
	return true;
}

bool SceneBuilderWidget::confirmDiscard() {
	if (!m_dirty) return true;
	const auto answer = QMessageBox::question(this, tr("Unsaved changes"), tr("The scene has changes that are not saved. Save them first?"),
	                                          QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
	if (answer == QMessageBox::Cancel) return false;
	if (answer == QMessageBox::Save) {
		onSaveClicked();
		return !m_dirty;
	}
	return true;
}

void SceneBuilderWidget::onOpenClicked() {
	if (!confirmDiscard()) return;
	const QString path = QFileDialog::getOpenFileName(this, tr("Open a Scene Builder scene"), m_path.isEmpty() ? QDir::homePath() : QFileInfo(m_path).absolutePath(),
	                                                  tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	QString error;
	if (!openFile(path, &error)) QMessageBox::warning(this, tr("Cannot open the scene"), error);
}

void SceneBuilderWidget::onSaveClicked() {
	if (m_path.isEmpty()) {
		onSaveAsClicked();
		return;
	}
	if (!saveFile(m_path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(m_path));
	else emit statusMessage(tr("Saved %1").arg(m_path));
}

void SceneBuilderWidget::onSaveAsClicked() {
	QString start = m_path;
	if (start.isEmpty()) start = QDir::homePath() + "/" + QString::fromStdString(m_doc.title).replace(QRegularExpression("[^A-Za-z0-9_-]+"), "-") + ".pbrt";
	QString path = QFileDialog::getSaveFileName(this, tr("Save the scene"), start, tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	if (!path.endsWith(".pbrt", Qt::CaseInsensitive)) path += ".pbrt";
	if (!saveFile(path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(path));
	else emit statusMessage(tr("Saved %1").arg(path));
}

// The scene list is built from the first existing, non-empty pbrt_scenes folder (the same search order as the renderer's own); a new folder created
// ahead of an existing one would hide every other scene, so this only ever uses a folder that is already there.
QString SceneBuilderWidget::sceneListFolder() {
	QStringList candidates;
	const QString env = qEnvironmentVariable("RAY_TRACER_PBRT_DIR");
	if (!env.isEmpty()) candidates << env;
	const QString app = QCoreApplication::applicationDirPath();
	candidates << app + "/pbrt_scenes" << app + "/../pbrt_scenes" << app + "/../../pbrt_scenes";
	for (const QString &c : candidates) {
		QDir d(c);
		if (d.exists() && !d.entryList(QStringList() << "*.pbrt", QDir::Files).isEmpty()) return d.absolutePath();
	}
	return QString();
}

void SceneBuilderWidget::onSaveToSceneListClicked() {
	const QString folder = sceneListFolder();
	if (folder.isEmpty()) {
		QMessageBox::information(this, tr("No scenes folder"),
		                         tr("The scenes folder (pbrt_scenes) was not found next to the program. Use Save As to put the file where you like, and set the "
		                            "environment variable RAY_TRACER_PBRT_DIR to that folder to have the program list it."));
		return;
	}
	QString name = QString::fromStdString(m_doc.title).trimmed().toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
	name.remove(QRegularExpression("^-+|-+$"));
	if (name.isEmpty()) name = "my-scene";
	const QString path = folder + "/" + name + ".pbrt";
	if (QFileInfo::exists(path) && QFileInfo(path) != QFileInfo(m_path)) {
		if (QMessageBox::question(this, tr("Replace the scene?"), tr("%1 already exists. Replace it?").arg(path)) != QMessageBox::Yes) return;
	}
	if (!saveFile(path)) {
		QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(path));
		return;
	}
	QMessageBox::information(this, tr("Added to the scene list"),
	                         tr("Saved as %1.\n\nRestart the program to see it in the scene list (Settings tab, Custom Scenes).").arg(path));
}

// ---- undo, autosave -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::pushUndo() {
	m_undo.append(QString::fromStdString(scene_doc::toJson(m_doc)));
	if (m_undo.size() > 200) m_undo.removeFirst();
	m_redo.clear();
}

void SceneBuilderWidget::edit(const QString &key, const std::function<void()> &mutate) {
	const bool merge = !key.isEmpty() && key == m_lastEditKey && (key.startsWith("drag#") || (m_editClock.isValid() && m_editClock.elapsed() < 1200));
	if (!merge) pushUndo();
	mutate();
	m_lastEditKey = key;
	m_editClock.restart();
	documentChanged();
}

void SceneBuilderWidget::restore(const QString &json) {
	Document d;
	std::string err;
	if (!scene_doc::fromJson(json.toStdString(), d, err)) return;
	m_doc = std::move(d);
	if (m_sel.kind == SelKind::Object && m_sel.index >= static_cast<int>(m_doc.objects.size())) m_sel = {SelKind::None, 0};
	if (m_sel.kind == SelKind::Light && m_sel.index >= static_cast<int>(m_doc.lights.size())) m_sel = {SelKind::None, 0};
	m_lastEditKey.clear();
	rebuildList();
	setSelection(m_sel);
	documentChanged();
}

bool SceneBuilderWidget::undo() {
	if (m_undo.isEmpty()) return false;
	const QString now = QString::fromStdString(scene_doc::toJson(m_doc));
	const QString prev = m_undo.takeLast();
	m_redo.append(now);
	restore(prev);
	return true;
}

bool SceneBuilderWidget::redo() {
	if (m_redo.isEmpty()) return false;
	const QString now = QString::fromStdString(scene_doc::toJson(m_doc));
	const QString next = m_redo.takeLast();
	m_undo.append(now);
	restore(next);
	return true;
}

void SceneBuilderWidget::documentChanged() {
	m_dirty = true;
	refreshListLabels();
	refreshProblems();
	updateTitle();
	updateActions();
	m_view->update();
	scheduleAutosave();
}

static QString autosavePath() {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
	QDir().mkpath(dir);
	return dir + "/scene_builder_autosave.pbrt";
}

void SceneBuilderWidget::scheduleAutosave() {
	if (m_autosaveEnabled) m_autosaveTimer->start();
}

void SceneBuilderWidget::writeAutosave() {
	if (!m_autosaveEnabled || !m_dirty) return;
	QFile f(autosavePath());
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
	const std::string text = scene_doc::toPbrt(m_doc);
	f.write(text.data(), static_cast<qint64>(text.size()));
}

void SceneBuilderWidget::clearAutosave() {
	if (m_autosaveEnabled) QFile::remove(autosavePath());
}

// A scene that was being edited when the program closed comes back, marked unsaved.
bool SceneBuilderWidget::loadAutosave() {
	QFile f(autosavePath());
	if (!f.exists() || !f.open(QIODevice::ReadOnly)) return false;
	Document d;
	std::string err;
	if (!scene_doc::fromPbrt(f.readAll().toStdString(), d, err)) return false;
	m_doc = std::move(d);
	m_path.clear();
	m_dirty = true;
	rebuildList();
	setSelection({SelKind::None, 0});
	m_view->frameAll();
	refreshProblems();
	updateTitle();
	updateActions();
	return true;
}

// ---- list, selection ------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::rebuildList() {
	QSignalBlocker block(m_list);
	m_list->clear();
	auto add = [this](const QString &text, SelKind kind, int index) {
		auto *item = new QListWidgetItem(text, m_list);
		item->setData(Qt::UserRole, static_cast<int>(kind));
		item->setData(Qt::UserRole + 1, index);
	};
	add(tr("Camera and image"), SelKind::Camera, 0);
	for (int i = 0; i < static_cast<int>(m_doc.objects.size()); ++i) add(QString(), SelKind::Object, i);
	for (int i = 0; i < static_cast<int>(m_doc.lights.size()); ++i) add(QString(), SelKind::Light, i);
	refreshListLabels();
	// reselect
	for (int r = 0; r < m_list->count(); ++r) {
		const auto *it = m_list->item(r);
		if (it->data(Qt::UserRole).toInt() == static_cast<int>(m_sel.kind) && it->data(Qt::UserRole + 1).toInt() == m_sel.index) {
			m_list->setCurrentRow(r);
			break;
		}
	}
}

void SceneBuilderWidget::refreshListLabels() {
	for (int r = 0; r < m_list->count(); ++r) {
		QListWidgetItem *it = m_list->item(r);
		const auto kind = static_cast<SelKind>(it->data(Qt::UserRole).toInt());
		const int i = it->data(Qt::UserRole + 1).toInt();
		if (kind == SelKind::Object && i < static_cast<int>(m_doc.objects.size())) {
			const Object &o = m_doc.objects[i];
			it->setText(QString("%1  -  %2%3").arg(QString::fromStdString(o.name), shapeLabel(o.shape).section(' ', 0, 0), o.emissive ? tr(", light") : QString()));
		} else if (kind == SelKind::Light && i < static_cast<int>(m_doc.lights.size())) {
			const Light &l = m_doc.lights[i];
			const QString type = lightLabel(l.kind).section(" (", 0, 0);
			const QString name = QString::fromStdString(l.name);
			it->setText(name.compare(type, Qt::CaseInsensitive) == 0 || name.startsWith(type.section(' ', 0, 0)) ? name : QString("%1  -  %2").arg(name, type));
		}
	}
}

void SceneBuilderWidget::onListSelectionChanged() {
	QListWidgetItem *it = m_list->currentItem();
	if (!it) {
		setSelection({SelKind::None, 0}, true);
		return;
	}
	setSelection({static_cast<SelKind>(it->data(Qt::UserRole).toInt()), it->data(Qt::UserRole + 1).toInt()}, true);
}

void SceneBuilderWidget::setSelection(const BuilderSelection &s, bool fromList) {
	m_sel = s;
	if (!fromList) {
		QSignalBlocker block(m_list);
		m_list->setCurrentRow(-1);
		for (int r = 0; r < m_list->count(); ++r) {
			const auto *it = m_list->item(r);
			if (it->data(Qt::UserRole).toInt() == static_cast<int>(s.kind) && it->data(Qt::UserRole + 1).toInt() == s.index) {
				m_list->setCurrentRow(r);
				break;
			}
		}
	}
	m_view->setSelection(s);
	rebuildInspector();
	updateActions();
}

// Presses on object `index` in the layout view, drags by `deltaPx` pixels and releases, with real mouse events; returns whether the object moved.
bool SceneBuilderWidget::dragObjectForTest(int index, const QPointF &deltaPx) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	const Float3 before = m_doc.objects[index].position;
	const QPointF start = m_view->itemScreenPos({SelKind::Object, index});
	auto send = [this](QEvent::Type type, const QPointF &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
		QMouseEvent e(type, pos, m_view->mapToGlobal(pos), button, buttons, Qt::NoModifier);
		QApplication::sendEvent(m_view, &e);
	};
	send(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) send(QEvent::MouseMove, start + deltaPx * (step / 4.0), Qt::NoButton, Qt::LeftButton);
	send(QEvent::MouseButtonRelease, start + deltaPx, Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].position;
	return before.x != after.x || before.y != after.y || before.z != after.z;
}

void SceneBuilderWidget::selectObject(int index) {
	setSelection({SelKind::Object, index});
}

void SceneBuilderWidget::addObject(ShapeKind shape) {
	std::string fileName;
	if (shape == ShapeKind::Mesh) {
		const QString f = QFileDialog::getOpenFileName(this, tr("Choose a .ply mesh"), QString(), tr("PLY meshes (*.ply)"));
		if (f.isEmpty()) return;
		fileName = f.toStdString();
	}
	edit(QString(), [&]() {
		// "Sphere 2", "Sphere 3", ...: the first free number.
		const QString base = shapeLabel(shape).section(' ', 0, 0);
		QString name = base;
		for (int n = 2;; ++n) {
			bool taken = false;
			for (const Object &o : m_doc.objects) taken = taken || QString::fromStdString(o.name) == name;
			if (!taken) break;
			name = QString("%1 %2").arg(base).arg(n);
		}
		Object o = scene_doc::makeObject(shape, name.toStdString());
		o.meshFile = fileName;
		// Drop it at the middle of the layout view, keeping the shape's own height above the floor where the view does not show it.
		const Float3 c = m_view->centerInWorld();
		o.position.x = c.x;
		o.position.z = c.z;
		if (c.y != 0.0) o.position.y = c.y;
		m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::addLight(LightKind kind) {
	edit(QString(), [&]() {
		Light l;
		l.kind = kind;
		const QString base = lightLabel(kind).section(' ', 0, 0);
		QString name = base;
		for (int n = 2;; ++n) {
			bool taken = false;
			for (const Light &x : m_doc.lights) taken = taken || QString::fromStdString(x.name) == name;
			if (!taken) break;
			name = QString("%1 %2").arg(base).arg(n);
		}
		l.name = name.toStdString();
		switch (kind) {
			case LightKind::Point: l.position = {1.0, 4.0, 2.0}; l.intensity = 40.0; break;
			case LightKind::Spot: l.position = {0.0, 5.0, 2.0}; l.target = {0.0, 0.0, 0.0}; l.intensity = 120.0; break;
			case LightKind::Distant: l.position = {4.0, 6.0, 3.0}; l.target = {0.0, 0.0, 0.0}; l.color = {1.0, 0.95, 0.85}; l.intensity = 3.0; break;
			case LightKind::Infinite: l.color = {0.55, 0.70, 1.0}; l.intensity = 0.5; break;
		}
		const Float3 c = m_view->centerInWorld();
		if (kind == LightKind::Point || kind == LightKind::Spot) { l.position.x = c.x; l.position.z = c.z; }
		m_doc.lights.push_back(l);
		m_sel = {SelKind::Light, static_cast<int>(m_doc.lights.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::deleteSelected() {
	if (m_sel.kind == SelKind::Object && m_sel.index < static_cast<int>(m_doc.objects.size())) {
		edit(QString(), [this]() { m_doc.objects.erase(m_doc.objects.begin() + m_sel.index); });
	} else if (m_sel.kind == SelKind::Light && m_sel.index < static_cast<int>(m_doc.lights.size())) {
		edit(QString(), [this]() { m_doc.lights.erase(m_doc.lights.begin() + m_sel.index); });
	} else {
		return;
	}
	m_sel = {SelKind::None, 0};
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::updateActions() {
	const bool item = (m_sel.kind == SelKind::Object || m_sel.kind == SelKind::Light);
	m_deleteButton->setEnabled(item);
	m_duplicateButton->setEnabled(item);
	m_undoButton->setEnabled(!m_undo.isEmpty());
	m_redoButton->setEnabled(!m_redo.isEmpty());
	const bool ok = !scene_doc::hasErrors(scene_doc::validate(m_doc));
	m_previewButton->setEnabled(ok || m_process);
	m_finalButton->setEnabled(ok && !m_process);
}

void SceneBuilderWidget::updateTitle() {
	const QString file = m_path.isEmpty() ? tr("not saved yet") : QFileInfo(m_path).fileName();
	m_titleLabel->setText(tr("%1 (%2)%3  |  %4 objects, %5 lights")
	                          .arg(QString::fromStdString(m_doc.title), file, m_dirty ? " *" : "")
	                          .arg(m_doc.objects.size())
	                          .arg(m_doc.lights.size()));
}

QString SceneBuilderWidget::problemsText() const {
	QString s;
	for (const auto &p : scene_doc::validate(m_doc))
		s += (p.severity == scene_doc::Problem::Severity::Error ? "Error: " : "Note: ") + QString::fromStdString(p.message) + "\n";
	return s;
}

void SceneBuilderWidget::refreshProblems() {
	const auto problems = scene_doc::validate(m_doc);
	if (problems.empty()) {
		m_problemsLabel->setText(tr("No problems found."));
		return;
	}
	QString html;
	for (const auto &p : problems) {
		const bool error = p.severity == scene_doc::Problem::Severity::Error;
		html += QString("<p style='margin:2px 0'><b>%1</b> %2</p>").arg(error ? tr("Fix this:") : tr("Note:"), QString::fromStdString(p.message).toHtmlEscaped());
	}
	m_problemsLabel->setText(html);
}

// ---- inspector ------------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::rebuildInspector() {
	m_refreshers.clear();
	auto *host = new QWidget;
	// The application stylesheet pads spin boxes generously; three of them in a row need less.
	host->setStyleSheet("QAbstractSpinBox { padding: 3px 3px; min-width: 40px; }");
	auto *form = new QFormLayout(host);
	form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
	m_loading = true;
	switch (m_sel.kind) {
		case SelKind::Camera: inspectCamera(form); break;
		case SelKind::Object:
			if (m_sel.index < static_cast<int>(m_doc.objects.size())) inspectObject(form, m_sel.index);
			break;
		case SelKind::Light:
			if (m_sel.index < static_cast<int>(m_doc.lights.size())) inspectLight(form, m_sel.index);
			break;
		case SelKind::None: {
			auto *help = new QLabel(tr("Pick something in the list or the layout view to edit it.\n\n"
			                           "Add shapes and lights with the Add button. Drag them in the layout view, then press Preview to see the picture. "
			                           "Save writes an ordinary .pbrt file that the renderer (and this tab) can open."),
			                        host);
			help->setWordWrap(true);
			form->addRow(help);
			break;
		}
	}
	m_loading = false;
	m_inspectorScroll->setWidget(host);  // the scroll area deletes the previous page
}

void SceneBuilderWidget::refreshInspectorValues() {
	m_loading = true;
	for (const auto &r : m_refreshers) r();
	m_loading = false;
}

static void addHeading(QFormLayout *f, const QString &text) {
	auto *l = new QLabel("<b>" + text.toHtmlEscaped() + "</b>");
	l->setContentsMargins(0, 8, 0, 0);
	f->addRow(l);
}

void SceneBuilderWidget::addNum(QFormLayout *f, const QString &label, const std::function<double *()> &ref, double lo, double hi, double step, int decimals,
                                const QString &suffix) {
	auto *box = new QDoubleSpinBox;
	box->setRange(lo, hi);
	box->setDecimals(decimals);
	box->setSingleStep(step);
	box->setKeyboardTracking(false);
	if (!suffix.isEmpty()) box->setSuffix(suffix);
	if (double *v = ref()) box->setValue(*v);
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, ref, key](double x) {
		if (m_loading) return;
		edit(key, [&]() { if (double *v = ref()) *v = x; });
	});
	m_refreshers.push_back([box, ref]() {
		QSignalBlocker b(box);
		if (double *v = ref()) box->setValue(*v);
	});
	f->addRow(label, box);
}

void SceneBuilderWidget::addInt(QFormLayout *f, const QString &label, const std::function<int *()> &ref, int lo, int hi) {
	auto *box = new QSpinBox;
	box->setRange(lo, hi);
	box->setKeyboardTracking(false);
	if (int *v = ref()) box->setValue(*v);
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(box, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, ref, key](int x) {
		if (m_loading) return;
		edit(key, [&]() { if (int *v = ref()) *v = x; });
	});
	m_refreshers.push_back([box, ref]() {
		QSignalBlocker b(box);
		if (int *v = ref()) box->setValue(*v);
	});
	f->addRow(label, box);
}

void SceneBuilderWidget::addVec3(QFormLayout *f, const QString &label, const std::function<Float3 *()> &ref, double step) {
	auto *row = new QWidget;
	auto *h = new QHBoxLayout(row);
	h->setContentsMargins(0, 0, 0, 0);
	h->setSpacing(4);
	const char *axes[3] = {"X", "Y", "Z"};
	double Float3::*members[3] = {&Float3::x, &Float3::y, &Float3::z};
	for (int a = 0; a < 3; ++a) {
		auto *box = new QDoubleSpinBox;
		box->setRange(-10000, 10000);
		box->setDecimals(3);
		box->setSingleStep(step);
		box->setKeyboardTracking(false);
		box->setPrefix(QString(axes[a]) + " ");
		box->setButtonSymbols(QAbstractSpinBox::NoButtons);  // three boxes in a row have no room for arrows; type, or use the wheel
		box->setMinimumWidth(60);
		auto member = members[a];
		if (Float3 *v = ref()) box->setValue((*v).*member);
		const QString key = QString("%1:%2:%3:%4").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label).arg(a);
		connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, ref, member, key](double x) {
			if (m_loading) return;
			edit(key, [&]() { if (Float3 *v = ref()) (*v).*member = x; });
		});
		m_refreshers.push_back([box, ref, member]() {
			QSignalBlocker b(box);
			if (Float3 *v = ref()) box->setValue((*v).*member);
		});
		h->addWidget(box, 1);
	}
	f->addRow(label, row);
}

void SceneBuilderWidget::addColor(QFormLayout *f, const QString &label, const std::function<Rgb *()> &ref) {
	auto *b = new QPushButton;
	b->setAutoDefault(false);
	b->setToolTip(tr("Click to choose a colour. Colours are picked as ordinary (sRGB) colours and stored as linear values for the renderer."));
	auto paint = [b, ref]() {
		if (Rgb *c = ref()) {
			const QColor q = toQColor(*c);
			const QColor fg = q.lightness() > 128 ? Qt::black : Qt::white;
			b->setText(q.name().toUpper());
			b->setStyleSheet(QString("QPushButton { background-color: %1; color: %2; border: 1px solid #666; padding: 4px; }").arg(q.name(), fg.name()));
		}
	};
	paint();
	connect(b, &QPushButton::clicked, this, [this, ref, paint, b]() {
		Rgb *c = ref();
		if (!c) return;
		const QColor picked = QColorDialog::getColor(toQColor(*c), this, tr("Choose a colour"));
		if (!picked.isValid()) return;
		edit(QString(), [&]() { if (Rgb *v = ref()) *v = fromQColor(picked); });
		paint();
	});
	m_refreshers.push_back(paint);
	f->addRow(label, b);
}

void SceneBuilderWidget::addBool(QFormLayout *f, const QString &label, const std::function<bool *()> &ref, bool rebuildAfter) {
	auto *c = new QCheckBox;
	if (bool *v = ref()) c->setChecked(*v);
	connect(c, &QCheckBox::toggled, this, [this, ref, rebuildAfter](bool on) {
		if (m_loading) return;
		edit(QString(), [&]() { if (bool *v = ref()) *v = on; });
		if (rebuildAfter) QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(label, c);
}

void SceneBuilderWidget::addText(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref, bool) {
	auto *e = new QLineEdit;
	if (std::string *v = ref()) e->setText(QString::fromStdString(*v));
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(e, &QLineEdit::textEdited, this, [this, ref, key](const QString &t) {
		if (m_loading) return;
		edit(key, [&]() { if (std::string *v = ref()) *v = t.toStdString(); });
	});
	f->addRow(label, e);
}

void SceneBuilderWidget::addFile(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref, const QString &filter) {
	auto *row = new QWidget;
	auto *h = new QHBoxLayout(row);
	h->setContentsMargins(0, 0, 0, 0);
	auto *e = new QLineEdit;
	e->setReadOnly(true);
	if (std::string *v = ref()) e->setText(QString::fromStdString(*v));
	auto *browse = new QPushButton(tr("Browse..."));
	browse->setAutoDefault(false);
	connect(browse, &QPushButton::clicked, this, [this, ref, e, filter]() {
		const QString p = QFileDialog::getOpenFileName(this, tr("Choose a file"), e->text(), filter);
		if (p.isEmpty()) return;
		edit(QString(), [&]() { if (std::string *v = ref()) *v = p.toStdString(); });
		e->setText(p);
	});
	h->addWidget(e, 1);
	h->addWidget(browse);
	f->addRow(label, row);
}

void SceneBuilderWidget::inspectCamera(QFormLayout *f) {
	addHeading(f, tr("Scene"));
	addText(f, tr("Title"), [this]() { return &m_doc.title; }, false);
	addHeading(f, tr("Camera"));
	addVec3(f, tr("Position"), [this]() { return &m_doc.camera.position; }, 0.25);
	addVec3(f, tr("Looks at"), [this]() { return &m_doc.camera.target; }, 0.25);
	addNum(f, tr("Field of view"), [this]() { return &m_doc.camera.fov; }, 1, 170, 1, 1, QString::fromUtf8(" \xC2\xB0"));
	addNum(f, tr("Lens radius"), [this]() { return &m_doc.camera.lensRadius; }, 0, 10, 0.01, 3);
	addNum(f, tr("Focus distance"), [this]() { return &m_doc.camera.focusDistance; }, 0.01, 10000, 0.5, 2);
	auto *hint = new QLabel(tr("A lens radius above 0 blurs what is not at the focus distance (depth of field)."));
	hint->setWordWrap(true);
	f->addRow(hint);
	addHeading(f, tr("Picture"));
	addInt(f, tr("Width (pixels)"), [this]() { return &m_doc.render.width; }, 1, 16384);
	addInt(f, tr("Height (pixels)"), [this]() { return &m_doc.render.height; }, 1, 16384);
	auto *aspect = new QComboBox;
	aspect->addItem(tr("Set height from width..."), QVariant());
	for (const char *a : {"4:3", "16:9", "3:2", "1:1", "21:9", "9:16"}) aspect->addItem(a, a);
	connect(aspect, QOverload<int>::of(&QComboBox::activated), this, [this, aspect](int i) {
		if (i <= 0) return;
		const QStringList parts = aspect->itemData(i).toString().split(':');
		const double w = parts[0].toDouble(), h = parts[1].toDouble();
		edit(QString(), [&]() { m_doc.render.height = std::max(1, static_cast<int>(std::lround(m_doc.render.width * h / w))); });
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Aspect ratio"), aspect);
	addInt(f, tr("Samples per pixel"), [this]() { return &m_doc.render.samples; }, 1, 100000);
	addInt(f, tr("Light bounces (max depth)"), [this]() { return &m_doc.render.maxDepth; }, 1, 100);
}

void SceneBuilderWidget::inspectMaterial(QFormLayout *f, int i) {
	addHeading(f, tr("Material"));
	auto *kind = new QComboBox;
	for (MaterialKind k : {MaterialKind::Diffuse, MaterialKind::Conductor, MaterialKind::Dielectric, MaterialKind::CoatedDiffuse, MaterialKind::DiffuseTransmission})
		kind->addItem(materialLabel(k), static_cast<int>(k));
	kind->setCurrentIndex(kind->findData(static_cast<int>(m_doc.objects[i].material.kind)));
	connect(kind, QOverload<int>::of(&QComboBox::activated), this, [this, i, kind](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.objects.size())) return;
		edit(QString(), [&]() {
			scene_doc::Material &m = m_doc.objects[i].material;
			m.kind = static_cast<MaterialKind>(kind->itemData(idx).toInt());
			// Starting values that look right for the new kind.
			if (m.kind == MaterialKind::Conductor && m.roughness == 0.0) m.roughness = 0.1;
			if (m.kind == MaterialKind::CoatedDiffuse && m.roughness == 0.0) m.roughness = 0.05;
		});
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Type"), kind);
	auto mat = [this, i]() -> scene_doc::Material * { return i < static_cast<int>(m_doc.objects.size()) ? &m_doc.objects[i].material : nullptr; };
	const scene_doc::Material &m = m_doc.objects[i].material;
	switch (m.kind) {
		case MaterialKind::Diffuse:
			addColor(f, m.checker ? tr("Colour A") : tr("Colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addBool(f, tr("Checker pattern"), [mat]() { return mat() ? &mat()->checker : nullptr; }, true);
			if (m.checker) {
				addColor(f, tr("Colour B"), [mat]() { return mat() ? &mat()->color2 : nullptr; });
				addNum(f, tr("Checks across"), [mat]() { return mat() ? &mat()->checkerCount : nullptr; }, 1, 1000, 1, 0);
			}
			break;
		case MaterialKind::Conductor:
			addColor(f, tr("Colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addNum(f, tr("Roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::Dielectric:
			addNum(f, tr("Index of refraction"), [mat]() { return mat() ? &mat()->ior : nullptr; }, 1.0, 3.0, 0.05, 2);
			addNum(f, tr("Roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::CoatedDiffuse:
			addColor(f, tr("Paint colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addNum(f, tr("Coat index of refraction"), [mat]() { return mat() ? &mat()->ior : nullptr; }, 1.0, 3.0, 0.05, 2);
			addNum(f, tr("Coat roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::DiffuseTransmission:
			addColor(f, tr("Reflects"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addColor(f, tr("Lets through"), [mat]() { return mat() ? &mat()->transmittance : nullptr; });
			break;
	}
}

void SceneBuilderWidget::inspectObject(QFormLayout *f, int i) {
	auto obj = [this, i]() -> Object * { return i < static_cast<int>(m_doc.objects.size()) ? &m_doc.objects[i] : nullptr; };
	addHeading(f, tr("Object"));
	addText(f, tr("Name"), [obj]() { return obj() ? &obj()->name : nullptr; }, true);
	auto *shape = new QComboBox;
	for (ShapeKind k : {ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Quad, ShapeKind::Disk, ShapeKind::Cylinder, ShapeKind::Cone, ShapeKind::Mesh})
		shape->addItem(shapeLabel(k), static_cast<int>(k));
	shape->setCurrentIndex(shape->findData(static_cast<int>(m_doc.objects[i].shape)));
	connect(shape, QOverload<int>::of(&QComboBox::activated), this, [this, i, shape](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.objects.size())) return;
		edit(QString(), [&]() { m_doc.objects[i].shape = static_cast<ShapeKind>(shape->itemData(idx).toInt()); });
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Shape"), shape);
	addVec3(f, tr("Position"), [obj]() { return obj() ? &obj()->position : nullptr; }, 0.25);
	addVec3(f, tr("Rotation (degrees)"), [obj]() { return obj() ? &obj()->rotation : nullptr; }, 5.0);

	const Object &o = m_doc.objects[i];
	switch (o.shape) {
		case ShapeKind::Sphere:
		case ShapeKind::Disk:
			addNum(f, tr("Radius"), [obj]() { return obj() ? &obj()->radius : nullptr; }, 0.001, 10000, 0.1);
			break;
		case ShapeKind::Cylinder:
		case ShapeKind::Cone:
			addNum(f, tr("Radius"), [obj]() { return obj() ? &obj()->radius : nullptr; }, 0.001, 10000, 0.1);
			addNum(f, tr("Height"), [obj]() { return obj() ? &obj()->height : nullptr; }, 0.001, 10000, 0.1);
			break;
		case ShapeKind::Box:
			addVec3(f, tr("Size"), [obj]() { return obj() ? &obj()->size : nullptr; }, 0.25);
			break;
		case ShapeKind::Quad:
			addNum(f, tr("Width (X)"), [obj]() { return obj() ? &obj()->size.x : nullptr; }, 0.001, 10000, 0.5);
			addNum(f, tr("Depth (Z)"), [obj]() { return obj() ? &obj()->size.z : nullptr; }, 0.001, 10000, 0.5);
			break;
		case ShapeKind::Mesh:
			addFile(f, tr("Mesh file"), [obj]() { return obj() ? &obj()->meshFile : nullptr; }, tr("PLY meshes (*.ply)"));
			addNum(f, tr("Scale"), [obj]() { return obj() ? &obj()->meshScale : nullptr; }, 0.0001, 10000, 0.1, 4);
			break;
	}

	inspectMaterial(f, i);

	addHeading(f, tr("Light"));
	addBool(f, tr("Gives off light"), [obj]() { return obj() ? &obj()->emissive : nullptr; }, true);
	if (o.emissive) {
		addColor(f, tr("Light colour"), [obj]() { return obj() ? &obj()->emission : nullptr; });
		addNum(f, tr("Strength"), [obj]() { return obj() ? &obj()->emissionStrength : nullptr; }, 0, 100000, 1, 2);
		addBool(f, tr("Both sides"), [obj]() { return obj() ? &obj()->twoSided : nullptr; });
		if (o.shape == ShapeKind::Quad || o.shape == ShapeKind::Disk) {
			auto *hint = new QLabel(tr("A quad or disk lights the side that faces up. Rotate it 180 degrees about X to make a ceiling light."));
			hint->setWordWrap(true);
			f->addRow(hint);
		}
	}
}

void SceneBuilderWidget::inspectLight(QFormLayout *f, int i) {
	auto lt = [this, i]() -> Light * { return i < static_cast<int>(m_doc.lights.size()) ? &m_doc.lights[i] : nullptr; };
	addHeading(f, tr("Light"));
	addText(f, tr("Name"), [lt]() { return lt() ? &lt()->name : nullptr; }, true);
	auto *kind = new QComboBox;
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite}) kind->addItem(lightLabel(k), static_cast<int>(k));
	kind->setCurrentIndex(kind->findData(static_cast<int>(m_doc.lights[i].kind)));
	connect(kind, QOverload<int>::of(&QComboBox::activated), this, [this, i, kind](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.lights.size())) return;
		edit(QString(), [&]() { m_doc.lights[i].kind = static_cast<LightKind>(kind->itemData(idx).toInt()); });
		rebuildList();
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Type"), kind);
	const Light &l = m_doc.lights[i];
	if (l.kind == LightKind::Point || l.kind == LightKind::Spot) addVec3(f, tr("Position"), [lt]() { return lt() ? &lt()->position : nullptr; }, 0.25);
	if (l.kind == LightKind::Spot) {
		addVec3(f, tr("Aims at"), [lt]() { return lt() ? &lt()->target : nullptr; }, 0.25);
		addNum(f, tr("Cone angle"), [lt]() { return lt() ? &lt()->coneAngle : nullptr; }, 1, 89, 1, 1, QString::fromUtf8(" \xC2\xB0"));
		addNum(f, tr("Soft edge"), [lt]() { return lt() ? &lt()->coneDelta : nullptr; }, 0, 89, 1, 1, QString::fromUtf8(" \xC2\xB0"));
	}
	if (l.kind == LightKind::Distant) {
		addVec3(f, tr("Shines from"), [lt]() { return lt() ? &lt()->position : nullptr; }, 0.25);
		addVec3(f, tr("Towards"), [lt]() { return lt() ? &lt()->target : nullptr; }, 0.25);
	}
	if (l.kind == LightKind::Infinite) {
		addFile(f, tr("Sky image"), [lt]() { return lt() ? &lt()->imageFile : nullptr; }, tr("Images (*.exr *.hdr *.png *.jpg *.jpeg)"));
		auto *hint = new QLabel(tr("Leave the image empty for a plain colour sky. An image is an equirectangular (lat-long) panorama."));
		hint->setWordWrap(true);
		f->addRow(hint);
	}
	if (l.kind != LightKind::Infinite || l.imageFile.empty()) addColor(f, tr("Colour"), [lt]() { return lt() ? &lt()->color : nullptr; });
	addNum(f, l.kind == LightKind::Point || l.kind == LightKind::Spot ? tr("Strength") : tr("Brightness"), [lt]() { return lt() ? &lt()->intensity : nullptr; }, 0, 100000, 0.5, 2);
}

// ---- rendering -------------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::startPreview(const std::function<void(bool, const QString &)> &done) {
	static const int widths[3] = {320, 480, 640};
	static const int spps[3] = {16, 64, 256};
	const int q = std::clamp(m_qualityCombo->currentData().toInt(), 0, 2);
	const double aspect = m_doc.render.width > 0 ? double(m_doc.render.height) / m_doc.render.width : 0.75;
	const int w = widths[q];
	const int h = std::max(1, static_cast<int>(std::lround(w * aspect)));
	runRender(w, h, spps[q], false, QString(), done);
}

void SceneBuilderWidget::onRenderFinalClicked() {
	QString start = m_path.isEmpty() ? QDir::homePath() + "/render.png" : QFileInfo(m_path).absolutePath() + "/" + QFileInfo(m_path).completeBaseName() + ".png";
	QString png = QFileDialog::getSaveFileName(this, tr("Save the rendered picture"), start, tr("PNG images (*.png)"));
	if (png.isEmpty()) return;
	if (!png.endsWith(".png", Qt::CaseInsensitive)) png += ".png";
	runRender(m_doc.render.width, m_doc.render.height, m_doc.render.samples, true, png, [this](bool ok, const QString &msg) {
		if (!ok) QMessageBox::warning(this, tr("The render failed"), msg);
	});
}

void SceneBuilderWidget::runRender(int width, int height, int samples, bool toFinalFile, const QString &finalPng,
                                   const std::function<void(bool, const QString &)> &done) {
	auto fail = [&](const QString &msg) {
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
	};
	if (m_process) return fail(tr("A render is already running."));
	const auto problems = scene_doc::validate(m_doc);
	if (scene_doc::hasErrors(problems)) return fail(tr("Fix the problems listed under the properties first."));
	if (!QFileInfo::exists(launcherPath())) return fail(tr("The renderer (%1) was not found next to the program.").arg(launcherPath()));

	const QString dir = workFolder();
	const QString scene = dir + "/scene.pbrt";
	const QString base = dir + (toFinalFile ? "/final" : "/preview");
	QFile::remove(base + ".ppm");
	QFile::remove(base + ".png");
	{
		QFile f(scene);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return fail(tr("Could not write %1.").arg(scene));
		const std::string text = scene_doc::toPbrt(m_doc);
		f.write(text.data(), static_cast<qint64>(text.size()));
	}

	QStringList args;
	args << (m_gpuCheck->isChecked() ? "--gpu" : "--cpu") << "--output" << base + ".ppm" << "--height" << QString::number(height)
	     << QString::number(width) << QString::number(samples) << QString::number(m_doc.render.maxDepth) << scene;

	m_pendingFinalPng = toFinalFile ? finalPng : QString();
	m_previewPng = base + ".png";
	m_pendingDone = done;
	m_renderLog.clear();
	m_cancelRequested = false;
	m_renderClock.start();
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setWorkingDirectory(QCoreApplication::applicationDirPath());
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
		m_renderLog += QString::fromUtf8(m_process->readAll());
		if (m_renderLog.size() > 20000) m_renderLog.remove(0, m_renderLog.size() - 20000);
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) onPreviewFinished(-1);
	});
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		onPreviewFinished(st == QProcess::NormalExit ? code : -2);
	});
	m_previewButton->setText(tr("Cancel"));
	m_previewButton->setEnabled(true);
	m_finalButton->setEnabled(false);
	m_previewStatus->setText(tr("Rendering %1 x %2, %3 samples...").arg(width).arg(height).arg(samples));
	m_process->start(launcherPath(), args);
}

void SceneBuilderWidget::onPreviewFinished(int exitCode) {
	QProcess *p = m_process;
	if (!p) return;
	m_process = nullptr;
	p->disconnect(this);
	p->deleteLater();
	const auto done = m_pendingDone;
	m_pendingDone = nullptr;
	m_previewButton->setText(tr("Preview"));
	updateActions();

	const double secs = m_renderClock.elapsed() / 1000.0;
	QPixmap pix(m_previewPng);
	if (exitCode != 0 || pix.isNull()) {
		QString tail;
		const QStringList lines = m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts);
		for (int i = std::max(0, static_cast<int>(lines.size()) - 6); i < lines.size(); ++i) tail += lines[i] + "\n";
		const QString msg = (exitCode == -2 && m_cancelRequested) ? tr("The render was cancelled.") : exitCode == -2 ? tr("The renderer stopped unexpectedly.") : tr("The renderer did not produce a picture (exit code %1).\n%2").arg(exitCode).arg(tail.trimmed());
		m_previewStatus->setText((exitCode == -2 && m_cancelRequested) ? msg : tr("The render failed."));
		if (done) done(false, msg);
		return;
	}
	m_previewPixmap = pix;
	updatePreviewPixmap();
	QString message = tr("Done in %1 s (%2 x %3).").arg(secs, 0, 'f', 1).arg(pix.width()).arg(pix.height());
	if (!m_pendingFinalPng.isEmpty()) {
		QFile::remove(m_pendingFinalPng);
		if (QFile::copy(m_previewPng, m_pendingFinalPng)) message += " " + tr("Saved %1.").arg(m_pendingFinalPng);
		else message += " " + tr("Could not save to %1.").arg(m_pendingFinalPng);
	}
	m_previewStatus->setText(message);
	if (done) done(true, message);
}

void SceneBuilderWidget::updatePreviewPixmap() {
	if (m_previewPixmap.isNull()) return;
	m_previewLabel->setPixmap(m_previewPixmap.scaled(m_previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

bool SceneBuilderWidget::eventFilter(QObject *watched, QEvent *event) {
	if (watched == m_previewLabel && event->type() == QEvent::Resize) updatePreviewPixmap();
	return QWidget::eventFilter(watched, event);
}

void SceneBuilderWidget::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	updatePreviewPixmap();
}
