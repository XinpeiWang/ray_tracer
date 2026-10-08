// The Scene Builder's views of the scene: the 2D layout view (Top / Front / Side) and the 3D view, one shown at a time, and what they share (the
// snap switch, Frame all, selection, drags as undo steps, and the keyboard shortcuts that work while a view has focus).
#include "scene_builder_widget.h"

#include "scene_3d_view.h"
#include "scene_builder_common.h"
#include "scene_layout_view.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QShortcut>
#include <QStackedWidget>
#include <QVBoxLayout>

#include <cmath>

using scene_doc::Float3;

void SceneBuilderWidget::createViews(QWidget *layoutBox, QVBoxLayout *layoutLayout) {
	auto *planeRow = new QHBoxLayout;
	auto *planeGroup = new QButtonGroup(this);
	planeGroup->setExclusive(true);
	m_view = new SceneLayoutView(layoutBox);
	m_view3d = new Scene3DView(layoutBox);
	m_viewStack = new QStackedWidget(layoutBox);
	m_viewStack->addWidget(m_view);
	m_viewStack->addWidget(m_view3d);
	m_viewHint = new QLabel(layoutBox);
	const QString hint2d = tr("Wheel: zoom. Right-drag: pan.");
	const QString hint3d = tr("Drag the background: orbit. Right-drag: pan. Wheel: zoom. Pick Move, Rotate or Scale (W, E, R) and drag the arrows, rings or squares; Shift-drag an object to lift it.");
	m_viewHint->setText(hint2d);
	m_gizmoBar = new QWidget(layoutBox);  // the Move / Rotate / Scale tools of the 3D view (W, E, R); shown with it
	auto *gizmoLayout = new QHBoxLayout(m_gizmoBar);
	gizmoLayout->setContentsMargins(0, 0, 0, 0);
	auto *gizmoGroup = new QButtonGroup(this);
	gizmoGroup->setExclusive(true);
	const std::pair<QString, QString> tools[] = {{tr("Move"), tr("Move (W): drag an object, or an arrow to move along one axis")},
	                                             {tr("Rotate"), tr("Rotate (E): drag a ring to turn the object about that axis")},
	                                             {tr("Scale"), tr("Scale (R): drag a square handle to stretch the object along that axis")}};
	for (int i = 0; i < 3; ++i) {
		auto *b = new QPushButton(tools[i].first, m_gizmoBar);
		b->setCheckable(true);
		b->setAutoDefault(false);
		b->setChecked(i == 0);
		b->setToolTip(tools[i].second);
		gizmoGroup->addButton(b, i);
		gizmoLayout->addWidget(b);
		connect(b, &QPushButton::clicked, this, [this, i]() { m_view3d->setGizmoMode(static_cast<Scene3DView::GizmoMode>(i)); });
	}
	connect(m_view3d, &Scene3DView::gizmoModeChanged, this, [gizmoGroup](int mode) {
		if (auto *b = gizmoGroup->button(mode)) b->setChecked(true);  // the W / E / R keys
	});
	m_gizmoBar->setVisible(false);

	const std::pair<QString, SceneLayoutView::Plane> planes[] = {{tr("Top"), SceneLayoutView::Plane::Top}, {tr("Front"), SceneLayoutView::Plane::Front}, {tr("Side"), SceneLayoutView::Plane::Side}};
	for (const auto &pl : planes) {
		auto *b = new QPushButton(pl.first, layoutBox);
		b->setCheckable(true);
		b->setAutoDefault(false);
		b->setChecked(pl.second == SceneLayoutView::Plane::Top);
		planeGroup->addButton(b);
		planeRow->addWidget(b);
		connect(b, &QPushButton::clicked, this, [this, pl, hint2d]() {
			m_viewStack->setCurrentWidget(m_view);
			m_view->setPlane(pl.second);
			m_viewHint->setText(hint2d);
			m_gizmoBar->setVisible(false);
		});
	}
	auto *b3d = new QPushButton(tr("3D"), layoutBox);
	b3d->setCheckable(true);
	b3d->setAutoDefault(false);
	b3d->setToolTip(tr("Look at the scene from any side, and move things in 3D"));
	planeGroup->addButton(b3d);
	planeRow->addWidget(b3d);
	connect(b3d, &QPushButton::clicked, this, [this, hint3d]() {
		m_viewStack->setCurrentWidget(m_view3d);
		m_viewHint->setText(hint3d);
		m_gizmoBar->setVisible(true);
	});

	auto *snap = new QCheckBox(tr("Snap to grid"), layoutBox);
	snap->setChecked(true);
	snap->setToolTip(tr("Dragging moves things in steps of 0.25. Hold Alt to drag freely."));
	connect(snap, &QCheckBox::toggled, this, [this](bool on) {
		m_view->setSnap(on);
		m_view3d->setSnap(on);
	});
	auto *frame = new QPushButton(tr("Frame all"), layoutBox);
	frame->setAutoDefault(false);
	connect(frame, &QPushButton::clicked, this, [this]() {
		m_view->frameAll();
		m_view3d->frameAll();
	});
	planeRow->addSpacing(8);
	planeRow->addWidget(m_gizmoBar);
	planeRow->addSpacing(8);
	planeRow->addWidget(snap);
	planeRow->addWidget(frame);
	planeRow->addStretch(1);
	layoutLayout->addLayout(planeRow);
	layoutLayout->addWidget(m_viewStack, 1);
	m_viewHint->setAlignment(Qt::AlignRight);  // on its own line: a translated hint is too long to share the button row
	m_viewHint->setWordWrap(true);
	layoutLayout->addWidget(m_viewHint);
	m_view->setDocument(&m_doc);
	m_view3d->setDocument(&m_doc);

	// Both views speak the same way: a click selects, a drag is one undo step, and a drag says where the item now is.
	const auto onSelection = [this](const BuilderSelection &s) { setSelection(s); };
	const auto onDragBegan = [this]() {
		// A drag is one undo step however many mouse events it takes; the first move takes the snapshot.
		m_lastEditKey.clear();
		m_editCounter++;
	};
	const auto onDragged = [this](const BuilderSelection &s, int which, const Float3 &w) {
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
	};
	connect(m_view, &SceneLayoutView::selectionRequested, this, onSelection);
	connect(m_view, &SceneLayoutView::dragBegan, this, onDragBegan);
	connect(m_view, &SceneLayoutView::positionDragged, this, onDragged);
	connect(m_view3d, &Scene3DView::selectionRequested, this, onSelection);
	connect(m_view3d, &Scene3DView::dragBegan, this, onDragBegan);
	connect(m_view3d, &Scene3DView::positionDragged, this, onDragged);
	// A turn or a stretch arrives as the whole edited object; the angles and dimensions are taken from it.
	connect(m_view3d, &Scene3DView::objectEdited, this, [this](const BuilderSelection &s, const scene_doc::Object &updated) {
		if (s.kind != SelKind::Object || s.index < 0 || s.index >= static_cast<int>(m_doc.objects.size())) return;
		edit(QString("drag#%1").arg(m_editCounter), [&]() {
			scene_doc::Object &o = m_doc.objects[s.index];
			o.rotation = updated.rotation;
			o.radius = updated.radius;
			o.radius2 = updated.radius2;
			o.height = updated.height;
			o.size = updated.size;
			o.meshScale = updated.meshScale;
		});
		refreshInspectorValues();
	});

	for (QWidget *view : {static_cast<QWidget *>(m_view), static_cast<QWidget *>(m_view3d)}) {
		auto *undoView = new QShortcut(QKeySequence::Undo, view);
		undoView->setContext(Qt::WidgetShortcut);
		connect(undoView, &QShortcut::activated, this, [this]() { undo(); });
		auto *redoView = new QShortcut(QKeySequence::Redo, view);
		redoView->setContext(Qt::WidgetShortcut);
		connect(redoView, &QShortcut::activated, this, [this]() { redo(); });
		auto *delView = new QShortcut(QKeySequence::Delete, view);
		delView->setContext(Qt::WidgetShortcut);
		connect(delView, &QShortcut::activated, this, [this]() { deleteSelected(); });
	}
}

// The views follow the document: both are redrawn, and the one on show tells where a new object goes.
void SceneBuilderWidget::updateViews() {
	m_view->update();
	m_view3d->update();
}

void SceneBuilderWidget::selectInViews(const BuilderSelection &s) {
	m_view->setSelection(s);
	m_view3d->setSelection(s);
}

Float3 SceneBuilderWidget::dropPoint() const {
	return m_viewStack->currentWidget() == m_view3d ? m_view3d->centerInWorld() : m_view->centerInWorld();
}

void SceneBuilderWidget::frameViews() {
	m_view->frameAll();
	m_view3d->frameAll();
}

// ---- self-test helpers for the 3D view (real mouse events, like dragObjectForTest for the 2D one) -----------------------------------------
void SceneBuilderWidget::show3dView(bool on) {
	m_viewStack->setCurrentWidget(on ? static_cast<QWidget *>(m_view3d) : static_cast<QWidget *>(m_view));
	m_viewHint->setText(on ? tr("Drag the background: orbit. Right-drag: pan. Wheel: zoom. Pick Move, Rotate or Scale (W, E, R) and drag the arrows, rings or squares; Shift-drag an object to lift it.")
	                       : tr("Wheel: zoom. Right-drag: pan."));
	m_gizmoBar->setVisible(on);
}

static void sendMouse(QWidget *w, QEvent::Type type, const QPointF &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
	QMouseEvent e(type, pos, w->mapToGlobal(pos), button, buttons, Qt::NoModifier);
	QApplication::sendEvent(w, &e);
}

bool SceneBuilderWidget::dragObject3dForTest(int index, const QPointF &deltaPx) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	const Float3 before = m_doc.objects[index].position;
	const QPointF start = m_view3d->itemScreenPos({SelKind::Object, index});
	sendMouse(m_view3d, QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) sendMouse(m_view3d, QEvent::MouseMove, start + deltaPx * (step / 4.0), Qt::NoButton, Qt::LeftButton);
	sendMouse(m_view3d, QEvent::MouseButtonRelease, start + deltaPx, Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].position;
	return before.x != after.x || before.y != after.y || before.z != after.z;
}

// Selects the object, then drags its `axis` arrow (0 X, 1 Y, 2 Z) by `pixels` along the arrow's own direction on screen; true if that coordinate (and only
// that one) changed.
bool SceneBuilderWidget::dragAxis3dForTest(int index, int axis, double pixels) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	setSelection({SelKind::Object, index});
	const Float3 before = m_doc.objects[index].position;
	const QPointF base = m_view3d->itemScreenPos({SelKind::Object, index});
	const QPointF mid = m_view3d->axisArrowPoint(axis, 0.6);
	QPointF dir = mid - base;
	const double len = std::hypot(dir.x(), dir.y());
	if (len < 1.0) return false;
	dir /= len;
	sendMouse(m_view3d, QEvent::MouseButtonPress, mid, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) sendMouse(m_view3d, QEvent::MouseMove, mid + dir * (pixels * step / 4.0), Qt::NoButton, Qt::LeftButton);
	sendMouse(m_view3d, QEvent::MouseButtonRelease, mid + dir * pixels, Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].position;
	const double d[3] = {after.x - before.x, after.y - before.y, after.z - before.z};
	for (int a = 0; a < 3; ++a)
		if (a != axis && std::abs(d[a]) > 1e-9) return false;
	return std::abs(d[axis]) > 1e-9;
}

// Selects the object, then drags its `axis` ring from 45 degrees round (halfway between where the three rings cross) to 45 + `degrees`; true if the object now has the angles a turn of (about) that much
// about that world axis gives.
bool SceneBuilderWidget::dragRotate3dForTest(int index, int axis, double degrees) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	m_view3d->setGizmoMode(Scene3DView::GizmoMode::Rotate);
	setSelection({SelKind::Object, index});
	const Float3 before = m_doc.objects[index].rotation;
	const QPointF from = m_view3d->ringPoint(axis, 45.0);
	sendMouse(m_view3d, QEvent::MouseButtonPress, from, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 6; ++step) sendMouse(m_view3d, QEvent::MouseMove, m_view3d->ringPoint(axis, 45.0 + degrees * step / 6.0), Qt::NoButton, Qt::LeftButton);
	sendMouse(m_view3d, QEvent::MouseButtonRelease, m_view3d->ringPoint(axis, 45.0 + degrees), Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].rotation;
	return before.x != after.x || before.y != after.y || before.z != after.z;
}

// Selects the object, then drags the end of its `axis` handle to `ratio` times its distance from the object; true if something changed.
bool SceneBuilderWidget::dragScale3dForTest(int index, int axis, double ratio) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	m_view3d->setGizmoMode(Scene3DView::GizmoMode::Scale);
	setSelection({SelKind::Object, index});
	const scene_doc::Object before = m_doc.objects[index];
	const QPointF from = m_view3d->scaleHandlePoint(axis, 1.0);
	sendMouse(m_view3d, QEvent::MouseButtonPress, from, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) sendMouse(m_view3d, QEvent::MouseMove, m_view3d->scaleHandlePoint(axis, 1.0 + (ratio - 1.0) * step / 4.0), Qt::NoButton, Qt::LeftButton);
	sendMouse(m_view3d, QEvent::MouseButtonRelease, m_view3d->scaleHandlePoint(axis, ratio), Qt::LeftButton, Qt::NoButton);
	const scene_doc::Object& after = m_doc.objects[index];
	return before.radius != after.radius || before.radius2 != after.radius2 || before.height != after.height || before.meshScale != after.meshScale || before.size.x != after.size.x ||
	       before.size.y != after.size.y || before.size.z != after.size.z;
}
