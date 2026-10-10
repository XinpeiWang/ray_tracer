#ifndef LIVE_OBJECT_EDITOR_H
#define LIVE_OBJECT_EDITOR_H
// live_object_editor.h -- pick and move the objects of the scene in the Live Preview picture with the mouse.
//
// The Live Preview tab owns one of these while a preview runs (MainWindow::addLivePreviewTab()), beside its OrbitPreviewLabel. Switched on with the "Move objects"
// button, a press on an object grabs it (the picture's pick comes from the renderer, RealtimePreviewSession::pickObjectAt()), a drag slides it across the floor, a
// drag with Shift held lifts and lowers it, W/A/S/D and Up/Down move it in the camera's own directions while it is selected (the keys that otherwise fly the
// camera), and a press on empty space orbits the camera as before. The selection, and its box, survive camera moves. "Reset objects" puts everything back. The arithmetic is
// object_drag_math.h; the objects themselves live in the renderer (src/shared/realtime_api.h).

#include <QObject>
#include <QString>

#include "camera_math.h"
#include "realtime_preview_session.h"

class OrbitPreviewLabel;
class QLabel;
class QPushButton;
class QWidget;

class LiveObjectEditor : public QObject {
	Q_OBJECT
public:
	// `sceneSize`: roughly how big the scene is (0 = unknown), which limits how far one drag may throw an object.
	LiveObjectEditor(RealtimePreviewSession *session, OrbitPreviewLabel *label, const QString &sceneId, const QString &sceneName, double sceneSize, QObject *parent);

	// The row of controls to put under the picture (the toggle, the reset button, and a hint line); owned by `parent`.
	QWidget *createControls(QWidget *parent);

	// Switches the mode on or off as the "Move objects" button does (the self-test uses it).
	void setMode(bool on);
	// What the hint line says now (the self-test reads it).
	QString hint() const;

	// A new picture is on screen: redraw the selection's box for the camera that drew it (so it follows the object while the camera moves).
	void frameShown();
	bool hasSelection() const { return m_haveSelection; }
	// A free-fly key (W/S forward/back, A/D left/right, Up/Down up/down, as steps of -1/0/+1) while an object is selected in object mode moves the object, by
	// `step` world units per step, in the directions the camera currently shows; true when it took the key (then the camera must not move).
	bool nudge(int forwardSteps, int rightSteps, int upSteps, double step);

signals:
	// "Save arrangement" wrote a scene file (into the per-user scenes folder); the main window lists it.
	void arrangementSaved(QString path);

private slots:
	void onSaveClicked();
	void onModeToggled(bool on);
	void onPressed(double s, double t);
	void onDragged(double s, double t, bool vertical);
	void onReleased();
	void onResetClicked();

private:
	void showSelectionBox();
	void currentBasis(camera_math::CameraBasis &basis) const;
	void setHint(const QString &text);

	RealtimePreviewSession *m_session;
	OrbitPreviewLabel *m_label;
	QString m_sceneId, m_sceneName;
	double m_sceneSize;
	QPushButton *m_toggle = nullptr;
	QPushButton *m_reset = nullptr;
	QPushButton *m_save = nullptr;
	QLabel *m_hint = nullptr;
	LiveObjectPick m_selected;   // the object grabbed or last grabbed: where it was when the drag began, and the camera that drew that picture
	bool m_haveSelection = false;
	bool m_dragging = false;
	camera_math::Vec3 m_pending;   // how far the current drag has moved the object so far (added to m_selected when the mouse is released)
};

#endif  // LIVE_OBJECT_EDITOR_H
