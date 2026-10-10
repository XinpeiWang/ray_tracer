#include "live_object_editor.h"

#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QRegularExpression>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QVector>

#include "mainwindow_widgets.h"   // OrbitPreviewLabel
#include "object_drag_math.h"
#include "../src/shared/pbrt_asset_check.h"

LiveObjectEditor::LiveObjectEditor(RealtimePreviewSession *session, OrbitPreviewLabel *label, const QString &sceneId, const QString &sceneName, double sceneSize,
                                   QObject *parent)
	: QObject(parent), m_session(session), m_label(label), m_sceneId(sceneId), m_sceneName(sceneName), m_sceneSize(sceneSize) {
	connect(label, &OrbitPreviewLabel::objectPressed, this, &LiveObjectEditor::onPressed);
	connect(label, &OrbitPreviewLabel::objectDragged, this, &LiveObjectEditor::onDragged);
	connect(label, &OrbitPreviewLabel::objectReleased, this, &LiveObjectEditor::onReleased);
	connect(label, &OrbitPreviewLabel::objectHovered, this, &LiveObjectEditor::onHovered);
	connect(label, &OrbitPreviewLabel::objectHoverEnded, this, &LiveObjectEditor::onHoverEnded);
	connect(session, &RealtimePreviewSession::hoverPicked, this, &LiveObjectEditor::onHoverPicked);
	m_hoverTimer = new QTimer(this);
	m_hoverTimer->setSingleShot(true);
	m_hoverTimer->setInterval(70);
	connect(m_hoverTimer, &QTimer::timeout, this, &LiveObjectEditor::onHoverTimer);
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
	m_save = new QPushButton(tr("Save arrangement"), row);
	m_save->setObjectName("liveSaveArrangementButton");
	m_save->setToolTip(tr("Save the scene with the objects where they are now as a new scene in the scene list (My Scenes). The original scene is not changed."));
	m_hint = new QLabel(row);
	m_hint->setWordWrap(true);
	layout->addWidget(m_toggle);
	layout->addWidget(m_reset);
	layout->addWidget(m_save);
	layout->addWidget(m_hint, /*stretch=*/1);
	connect(m_toggle, &QPushButton::toggled, this, &LiveObjectEditor::onModeToggled);
	connect(m_reset, &QPushButton::clicked, this, &LiveObjectEditor::onResetClicked);
	connect(m_save, &QPushButton::clicked, this, &LiveObjectEditor::onSaveClicked);
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
	setHint(on ? tr("Click an object and drag it (Shift: up and down). Then W A S D and Up/Down move it too.") : QString());
	if (on) m_label->setFocus();
}

void LiveObjectEditor::frameShown() {
	if (m_haveSelection) showSelectionBox();
	if (m_hovering) showHoverBox();
}

// The camera of the picture on screen (it moves while the user orbits or flies), or, before the first frame, the one the object was picked with.
void LiveObjectEditor::currentBasis(camera_math::CameraBasis &basis) const {
	double b[12];
	if (!m_session || !m_session->cameraBasisNow(b)) std::copy(m_selected.cameraBasis, m_selected.cameraBasis + 12, b);
	basis = camera_math::CameraBasis{{b[0], b[1], b[2]}, {b[3], b[4], b[5]}, {b[6], b[7], b[8]}, {b[9], b[10], b[11]}};
}

bool LiveObjectEditor::nudge(int forwardSteps, int rightSteps, int upSteps, double step) {
	if (!m_haveSelection || !m_label->objectMode() || !m_session) return false;
	if (m_dragging) return true;
	camera_math::CameraBasis basis;
	camera_math::Vec3 delta;
	currentBasis(basis);
	if (!object_drag::keyMove(basis, forwardSteps, rightSteps, upSteps, step, delta)) return true;
	for (int a = 0; a < 3; ++a) {
		const double d = a == 0 ? delta.x : a == 1 ? delta.y : delta.z;
		m_selected.lo[a] += d;
		m_selected.hi[a] += d;
		m_selected.hit[a] += d;
		m_selected.offset[a] += d;
	}
	m_session->setObjectOffset(m_selected.object, m_selected.offset[0], m_selected.offset[1], m_selected.offset[2]);
	showSelectionBox();
	return true;
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
	m_label->clearHover();
	m_label->setObjectGrabbed(true);
	m_pending = camera_math::Vec3{0.0, 0.0, 0.0};
	showSelectionBox();
	setHint(tr("Selected: %1. Drag to move it (Shift: up and down); W A S D and Up/Down move it too.").arg(pick.label));
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
	m_pending = delta;
	showSelectionBox();
}

// The drag is over: the object is where the drag put it, which is where later drags and key presses start from.
void LiveObjectEditor::onReleased() {
	m_dragging = false;
	if (!m_haveSelection) return;
	const double d[3] = {m_pending.x, m_pending.y, m_pending.z};
	for (int a = 0; a < 3; ++a) {
		m_selected.lo[a] += d[a];
		m_selected.hi[a] += d[a];
		m_selected.hit[a] += d[a];
		m_selected.offset[a] += d[a];
	}
	m_pending = camera_math::Vec3{0.0, 0.0, 0.0};
}

void LiveObjectEditor::onResetClicked() {
	if (!m_session) return;
	m_session->resetObjects();
	m_haveSelection = false;
	m_label->clearSelection();
	setHint(tr("Every object is back where the scene file puts it."));
}

// Saves the scene with the moved objects as a new file in the per-user scenes folder (never over an existing one), and tells the main window to list it.
void LiveObjectEditor::onSaveClicked() {
	if (!m_session) return;
	const QString folder = QString::fromStdString(pbrt_asset_check::userSceneDir());
	if (folder.isEmpty() || !QDir().mkpath(folder)) {
		setHint(tr("Cannot save: there is no scenes folder to write to."));
		return;
	}
	QString name = m_sceneName.trimmed().toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
	name.remove(QRegularExpression("^-+|-+$"));
	if (name.isEmpty()) name = "scene";
	QString path = folder + "/" + name + "-arranged.pbrt";
	for (int n = 2; QFileInfo::exists(path); ++n) path = folder + "/" + name + "-arranged-" + QString::number(n) + ".pbrt";
	QString message;
	if (!m_session->exportArrangement(m_sceneId, path, &message)) {
		setHint(tr("Not saved: %1.").arg(message));
		return;
	}
	setHint(tr("Saved as %1 (%2). It is in the scene list under My Scenes.").arg(QFileInfo(path).completeBaseName(), message));
	emit arrangementSaved(path);
}

// The 12 edges of the selected object's box (plus the drag so far), as the picture on screen shows them.
void LiveObjectEditor::showSelectionBox() {
	camera_math::CameraBasis basis;
	currentBasis(basis);
	m_label->setSelectionSegments(boxSegments(m_selected, m_pending, basis));
}

// The hover box: the object a click would grab, drawn as it is in the picture on screen (not while one is being dragged, and not on the selected object, which has
// its own box).
void LiveObjectEditor::showHoverBox() {
	if (!m_hovering || m_dragging || (m_haveSelection && m_hover.object == m_selected.object)) {
		m_label->clearHover();
		return;
	}
	camera_math::CameraBasis basis;
	currentBasis(basis);
	m_label->setHoverSegments(boxSegments(m_hover, camera_math::Vec3{0.0, 0.0, 0.0}, basis));
}

QVector<QPointF> LiveObjectEditor::boxSegments(const LiveObjectPick &pick, const camera_math::Vec3 &shift, const camera_math::CameraBasis &basis) {
	QPointF corner[8];
	bool visible[8];
	for (int i = 0; i < 8; ++i) {
		const camera_math::Vec3 p{(i & 1 ? pick.hi[0] : pick.lo[0]) + shift.x, (i & 2 ? pick.hi[1] : pick.lo[1]) + shift.y, (i & 4 ? pick.hi[2] : pick.lo[2]) + shift.z};
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
	return segments;
}

// Hover: the cursor stopped (or moved) over the picture. A pick is asked for at most every 70 ms, one at a time, so a fast mouse never queues up work for the worker.
void LiveObjectEditor::onHovered(double s, double t) {
	if (m_dragging || !m_session) return;
	m_hoverAt = QPointF(s, t);
	m_hoverWanted = true;
	if (!m_hoverTimer->isActive() && !m_hoverBusy) m_hoverTimer->start();
}

void LiveObjectEditor::onHoverTimer() {
	if (!m_hoverWanted || m_hoverBusy || !m_session) return;
	m_hoverWanted = false;
	m_hoverBusy = true;
	m_session->requestHover(m_hoverAt.x(), m_hoverAt.y());
}

void LiveObjectEditor::onHoverPicked(LiveObjectPick pick) {
	m_hoverBusy = false;
	if (!m_label->objectMode()) return;
	m_hovering = pick.valid;
	if (pick.valid) m_hover = pick;
	showHoverBox();
	if (m_hoverWanted) m_hoverTimer->start();   // the cursor moved while that pick was running
}

void LiveObjectEditor::onHoverEnded() {
	m_hoverWanted = false;
	m_hovering = false;
	m_label->clearHover();
}
