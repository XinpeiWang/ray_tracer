#ifndef LIVE_OBJECT_EDITOR_H
#define LIVE_OBJECT_EDITOR_H
// live_object_editor.h -- pick and move the objects of the scene in the Live Preview picture with the mouse.
//
// The Live Preview tab owns one of these while a preview runs (MainWindow::addLivePreviewTab()), beside its OrbitPreviewLabel. Switched on with the "Move objects"
// button, a press on an object grabs it (the picture's pick comes from the renderer, RealtimePreviewSession::pickObjectAt()), a drag slides it across the floor, a
// drag with Shift held lifts and lowers it, and a press on empty space orbits the camera as before. "Reset objects" puts everything back. The arithmetic is
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
	LiveObjectEditor(RealtimePreviewSession *session, OrbitPreviewLabel *label, double sceneSize, QObject *parent);

	// The row of controls to put under the picture (the toggle, the reset button, and a hint line); owned by `parent`.
	QWidget *createControls(QWidget *parent);

	// Switches the mode on or off as the "Move objects" button does (the self-test uses it).
	void setMode(bool on);
	// What the hint line says now (the self-test reads it).
	QString hint() const;

	// The camera moved: a box drawn for the old view would be in the wrong place, so it goes.
	void cameraMoved();

private slots:
	void onModeToggled(bool on);
	void onPressed(double s, double t);
	void onDragged(double s, double t, bool vertical);
	void onReleased();
	void onResetClicked();

private:
	void showSelectionBox(const camera_math::Vec3 &shift);
	void setHint(const QString &text);

	RealtimePreviewSession *m_session;
	OrbitPreviewLabel *m_label;
	double m_sceneSize;
	QPushButton *m_toggle = nullptr;
	QPushButton *m_reset = nullptr;
	QLabel *m_hint = nullptr;
	LiveObjectPick m_selected;   // the object grabbed or last grabbed: where it was when the drag began, and the camera that drew that picture
	bool m_haveSelection = false;
	bool m_dragging = false;
};

#endif  // LIVE_OBJECT_EDITOR_H
