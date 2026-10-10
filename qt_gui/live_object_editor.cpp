#include "live_object_editor.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVector>

#include "mainwindow_widgets.h"   // OrbitPreviewLabel
#include "object_drag_math.h"

LiveObjectEditor::LiveObjectEditor(RealtimePreviewSession *session, OrbitPreviewLabel *label, double sceneSize, QObject *parent)
	: QObject(parent), m_session(session), m_label(label), m_sceneSize(sceneSize) {
	connect(label, &OrbitPreviewLabel::objectPressed, this, &LiveObjectEditor::onPressed);
	connect(label, &OrbitPreviewLabel::objectDragged, this, &LiveObjectEditor::onDragged);
	connect(label, &OrbitPreviewLabel::objectReleased, this, &LiveObjectEditor::onReleased);
}

QWidget *LiveObjectEditor::createControls(QWidget *parent) {
	QWidget *row = new QWidget(parent);
	QHBoxLayout *layout = new QHBoxLayout(row);
	layout->setContentsMargins(0, 0, 0, 0);
	m_toggle = new QPushButton(tr("Move objects"), row);
	m_toggle->setCheckable(true);
	m_toggle->setToolTip(tr("Click an object in the picture and drag it to move it. Hold Shift while dragging to lift or lower it. "
	                        "Drag on empty space to orbit as usual."));
	m_reset = new QPushButton(tr("Reset objects"), row);
	m_reset->setObjectName("liveResetObjectsButton");   // the self-test finds the button by this, whatever the language
	m_reset->setToolTip(tr("Put every object back where the scene file puts it."));
	m_hint = new QLabel(row);
	m_hint->setWordWrap(true);
	layout->addWidget(m_toggle);
	layout->addWidget(m_reset);
	layout->addWidget(m_hint, /*stretch=*/1);
	connect(m_toggle, &QPushButton::toggled, this, &LiveObjectEditor::onModeToggled);
	connect(m_reset, &QPushButton::clicked, this, &LiveObjectEditor::onResetClicked);
	return row;
}

void LiveObjectEditor::setMode(bool on) {
	if (m_toggle) m_toggle->setChecked(on);
}

QString LiveObjectEditor::hint() const {
	return m_hint ? m_hint->text() : QString();
}

void LiveObjectEditor::setHint(const QString &text) {
	if (m_hint) m_hint->setText(text);
}

void LiveObjectEditor::onModeToggled(bool on) {
	m_label->setObjectMode(on);
	m_haveSelection = false;
	setHint(on ? tr("Click an object and drag it. Shift: up and down.") : QString());
	if (on) m_label->setFocus();
}

void LiveObjectEditor::cameraMoved() {
	if (!m_haveSelection) return;
	m_haveSelection = false;
	m_label->clearSelection();
}

void LiveObjectEditor::onPressed(double s, double t) {
	if (!m_session) return;
	const LiveObjectPick pick = m_session->pickObjectAt(s, t);
	if (!pick.valid) {
		m_haveSelection = false;
		m_label->clearSelection();
		setHint(tr("Nothing movable there: drag to orbit."));
		return;
	}
	m_selected = pick;
	m_haveSelection = true;
	m_dragging = true;
	m_label->setObjectGrabbed(true);
	showSelectionBox(camera_math::Vec3{0.0, 0.0, 0.0});
	setHint(tr("Moving: %1. Shift: up and down.").arg(pick.label));
}

void LiveObjectEditor::onDragged(double s, double t, bool vertical) {
	if (!m_haveSelection || !m_dragging || !m_session) return;
	const double *b = m_selected.cameraBasis;
	const camera_math::CameraBasis basis{{b[0], b[1], b[2]}, {b[3], b[4], b[5]}, {b[6], b[7], b[8]}, {b[9], b[10], b[11]}};
	const camera_math::Vec3 grab{m_selected.hit[0], m_selected.hit[1], m_selected.hit[2]};
	camera_math::Vec3 delta;
	// One drag never throws the object further than a few scene sizes (a ray that grazes the floor meets it very far away).
	if (!object_drag::dragDelta(basis, grab, s, t, vertical, m_sceneSize > 0.0 ? 3.0 * m_sceneSize : 0.0, delta)) return;
	m_session->setObjectOffset(m_selected.object, m_selected.offset[0] + delta.x, m_selected.offset[1] + delta.y, m_selected.offset[2] + delta.z);
	showSelectionBox(delta);
}

void LiveObjectEditor::onReleased() {
	m_dragging = false;
}

void LiveObjectEditor::onResetClicked() {
	if (!m_session) return;
	m_session->resetObjects();
	m_haveSelection = false;
	m_label->clearSelection();
	setHint(tr("Every object is back where the scene file puts it."));
}

// The 12 edges of the selected object's box, shifted by `shift`, as the picture shows them (the camera that drew the picture the object was picked in).
void LiveObjectEditor::showSelectionBox(const camera_math::Vec3 &shift) {
	const double *b = m_selected.cameraBasis;
	const camera_math::CameraBasis basis{{b[0], b[1], b[2]}, {b[3], b[4], b[5]}, {b[6], b[7], b[8]}, {b[9], b[10], b[11]}};
	QPointF corner[8];
	bool visible[8];
	for (int i = 0; i < 8; ++i) {
		const camera_math::Vec3 p{(i & 1 ? m_selected.hi[0] : m_selected.lo[0]) + shift.x, (i & 2 ? m_selected.hi[1] : m_selected.lo[1]) + shift.y,
		                          (i & 4 ? m_selected.hi[2] : m_selected.lo[2]) + shift.z};
		const camera_math::ScreenProjection sp = camera_math::projectToScreen(p, basis);
		visible[i] = sp.inFront;
		corner[i] = QPointF(sp.s, sp.t);
	}
	QVector<QPointF> segments;
	for (int i = 0; i < 8; ++i)
		for (int axis = 0; axis < 3; ++axis) {
			const int j = i | (1 << axis);
			if (j == i || !visible[i] || !visible[j]) continue;
			segments.push_back(corner[i]);
			segments.push_back(corner[j]);
		}
	m_label->setSelectionSegments(segments);
}
