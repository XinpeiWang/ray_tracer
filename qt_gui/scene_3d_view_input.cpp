// scene_3d_view_input.cpp - what is under the mouse, and what dragging, wheel and keys do in the 3D view (see scene_3d_view.h).
#include "scene_3d_view_internal.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>

#include <algorithm>

using namespace scene_builder_ui;

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
			if (mode == GizmoMode::Move && std::hypot(bx - px.x(), by - px.y()) <= kFreeHandlePx) {   // the dot in the middle: a free move (comes before the arrows that start there)
				h.kind = Hit::Kind::FreeHandle;
				h.sel = m_sel;
				return h;
			}
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
	if (h.kind == Hit::Kind::FreeHandle || (e->modifiers() & Qt::ControlModifier)) {
		// A free move: the item follows the pointer on the plane through it that faces the camera, so it goes up, down, sideways, nearer or further as the mouse
		// goes (the white dot in the middle of the Move tool, or Ctrl - Command on a Mac - while dragging anything).
		m_mode = Mode::Free;
		V3 hit;
		if (scene_view::rayPlane(ray, m_dragStart, v.cam.forward(), hit)) m_grabOffset = m_dragStart - hit;
		else m_mode = Mode::None;
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
	} else if (m_mode == Mode::Free) {
		V3 hit;
		if (!scene_view::rayPlane(ray, m_dragStart, v.cam.forward(), hit)) return;
		const V3 target = hit + m_grabOffset;
		w.x = snap ? snapQuarter(target.x) : target.x;
		w.y = snap ? snapQuarter(target.y) : target.y;
		w.z = snap ? snapQuarter(target.z) : target.z;
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
