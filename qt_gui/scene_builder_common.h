#pragma once
// scene_builder_common.h - colour conversion and display names shared by the Scene Builder's layout view, property panel and widget.

#include <QColor>
#include <QObject>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <cmath>

#include "../src/shared/scene_document.h"

namespace scene_builder_ui {

using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

inline double toSrgb(double c) {
	c = std::clamp(c, 0.0, 1.0);
	return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
inline double toLinear(double s) {
	s = std::clamp(s, 0.0, 1.0);
	return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}
// Colours are stored linear (what the renderer wants) and picked and drawn as sRGB (what a colour picker shows).
inline QColor toQColor(const Rgb &c) { return QColor::fromRgbF(toSrgb(c.r), toSrgb(c.g), toSrgb(c.b)); }
inline Rgb fromQColor(const QColor &q) { return Rgb{toLinear(q.redF()), toLinear(q.greenF()), toLinear(q.blueF())}; }

inline QString shapeLabel(ShapeKind k) {
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
inline QString lightLabel(LightKind k) {
	switch (k) {
		case LightKind::Point: return QObject::tr("Point light");
		case LightKind::Spot: return QObject::tr("Spotlight");
		case LightKind::Distant: return QObject::tr("Sun (distant light)");
		case LightKind::Infinite: return QObject::tr("Sky (surrounds the scene)");
	}
	return QString();
}
// "Sphere", then "Sphere 2", "Sphere 3", ...: the first of those not already in `taken`.
inline QString uniqueName(const QString &base, const QStringList &taken) {
	QString name = base;
	for (int n = 2; taken.contains(name); ++n) name = QString("%1 %2").arg(base).arg(n);
	return name;
}

inline QString materialLabel(MaterialKind k) {
	switch (k) {
		case MaterialKind::Diffuse: return QObject::tr("Matte (diffuse)");
		case MaterialKind::Conductor: return QObject::tr("Metal");
		case MaterialKind::Dielectric: return QObject::tr("Glass");
		case MaterialKind::CoatedDiffuse: return QObject::tr("Glossy paint (coated)");
		case MaterialKind::DiffuseTransmission: return QObject::tr("Translucent (paper, leaves)");
	}
	return QString();
}



// Rotates p by the object's three angles, in the order the writer applies them: about X, then Y, then Z.
inline scene_doc::Float3 rotateXYZ(scene_doc::Float3 p, const scene_doc::Float3 &deg) {
	constexpr double kPi = 3.14159265358979323846;
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

} // namespace scene_builder_ui
