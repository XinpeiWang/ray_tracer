#pragma once
// scene_builder_widget.h - the GUI's "Scene Builder" tab: build a scene from shapes, materials and lights, move things around in a
// layout view, preview it, and save it as an ordinary .pbrt file.
//
// The model, the pbrt writer and the checks are in src/shared/scene_document.h (header-only, unit-tested without Qt); this file is
// only the editing surface. The widget owns one scene_doc::Document and never renders in-process: a preview runs the same
// ray_tracer launcher the Render tab runs, on a .pbrt file saved to a temporary folder.

#include <QElapsedTimer>
#include <QList>
#include <QPixmap>
#include <QPointF>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <vector>

#include "../src/shared/scene_document.h"
#include "../src/shared/snapshot_history.h"
#include "../src/shared/scene_props.h"

class QButtonGroup;
class QCheckBox;
class QSplitter;
class QLineEdit;
class QComboBox;
class QFormLayout;
class QLabel;
class QListWidget;
class QProcess;
class QPushButton;
class QScrollArea;
class QStackedWidget;
class QVBoxLayout;
class Scene3DView;
class QTimer;

#include "scene_layout_view.h"


class SceneBuilderWidget : public QWidget {
	Q_OBJECT
public:
	explicit SceneBuilderWidget(QWidget *parent = nullptr);
	~SceneBuilderWidget() override;

	// For the automated self-test (mainwindow_selftest.cpp) and tests: drive the editor without a mouse.
	const scene_doc::Document &document() const { return m_doc; }
	void newScene();
	bool openFile(const QString &path, QString *error = nullptr);
	bool saveFile(const QString &path);
	void selectObject(int index);
	void selectLight(int index);   // (the self-test picks the sky)
	bool dragObjectForTest(int index, const QPointF &deltaPx);
	// The 3D view, for the self-test: show it (or go back to the 2D views), drag an object on the floor, or along one axis arrow of the selected item.
	void show3dView(bool on);
	void resetPaneSizes();   // View > Reset Window Layout
	bool dragObject3dForTest(int index, const QPointF &deltaPx);
	QPointF shiftPanBackground3dForTest(int index, const QPointF &deltaPx);
	bool dragAxis3dForTest(int index, int axis, double pixels);
	bool dragRotate3dForTest(int index, int axis, double degrees);
	bool dragScale3dForTest(int index, int axis, double ratio);
	void orbit3dForTest(double yawDeg, double pitchDeg);
	scene_doc::Float3 dropPointForTest() const { return dropPoint(); }
	bool meshReadyForTest(const QString &path) const;
	bool gizmoButtonEnabled(int tool) const;   // 0 Move, 1 Rotate, 2 Scale
	int gizmoButtonChecked() const;
	void selectCameraForTest() { setSelection({SelKind::Camera, 0}); }
	void addObject(scene_doc::ShapeKind shape);
	void addProp(scene_doc::PropKind kind);   // a few ordinary objects at once (a table, a tree...), one undo step
	void addLight(scene_doc::LightKind kind);
	void addObjectFromPhoto();  // scene_builder_photo.cpp: needs the optional photo helper
	// Runs the helper on one photo (progress dialog) and adds the mesh. False with `error` set (empty when cancelled) if nothing was added.
	bool importPhoto(const QString &photo, QString *error);
	// See scene_builder_inspector.cpp: an upright copy of a picture whose EXIF tag asks for a rotation (the renderer ignores that tag).
	static QString uprightPictureCopy(const QString &path, QSize *size);
	void deleteSelected();
	bool undo();
	bool redo();
	// Starts a preview render (no file dialog) and calls `done` when the picture is loaded or it failed.
	void startPreview(const std::function<void(bool ok, const QString &message)> &done = nullptr);
	QString previewImagePath() const { return m_previewPng; }
	// The "Use the GPU" checkbox (OptiX on Windows, Metal on a Mac); the self-test drives it.
	void setUseGpu(bool on);
	bool isDirty() const { return m_dirty; }
	QString problemsText() const;
	// Where a scene is saved so the scene list finds it (empty if no scene folder exists); see saveToSceneList().
	static QString sceneListFolder();
	static bool hasAutosave();           // an unsaved scene from the last run is waiting to be restored
	static QString setAsideAutosave();   // renames it (never deletes it); returns the new path, "" if there was none
	// Saves a copy of the scene into the scene-list folder (see sceneListFolder()); no dialogs. Returns the path, or "" with `error` set. An existing file
	// of that name is replaced.
	// With `update` false (the default) a scene of that name already in the folder is never touched: the copy gets the next free name ("my-scene-2").
	// With `update` true and this document already listed (listedPath()), that listing is overwritten.
	QString addToSceneList(QString *error, bool update = false);
	// The scene-list file this document was last added as, or opened from (empty if none).
	QString listedPath() const { return m_listedPath; }
	// The scene's name (its title), as typed in the toolbar; setSceneName() is an undoable edit like typing there.
	QString sceneName() const { return QString::fromStdString(m_doc.title); }
	void setSceneName(const QString &name);

signals:
	void statusMessage(const QString &text);
	// A copy of the scene was saved into the scene-list folder ("Add to scene list"); the main window lists it without a restart.
	void sceneListed(const QString &path);

protected:
	void resizeEvent(QResizeEvent *e) override;
	void showEvent(QShowEvent *e) override;
	bool eventFilter(QObject *watched, QEvent *event) override;

private slots:
	void onListSelectionChanged();
	void onOpenClicked();
	void onSaveClicked();
	void onSaveAsClicked();
	void onSaveToSceneListClicked();
	void onRenderFinalClicked();
	void onPreviewFinished(int exitCode);

private:
	using SelKind = BuilderSelection::Kind;

	void buildUi();
	void rebuildList();
	void refreshListLabels();
	void rebuildInspector();
	void refreshInspectorValues();
	void updatePreviewPixmap();
	void setSelection(const BuilderSelection &s, bool fromList = false);
	void logSelection(const BuilderSelection &s, const char *where);   // what the user picked, and where (the list, the layout view, the 3D view)
	void documentChanged();
	void refreshProblems();
	void updateTitle();
	void updateActions();

	// An edit: takes an undo snapshot (merging a run of edits with the same key, such as a spin box being dragged), applies `mutate`, and refreshes.
	void edit(const QString &key, const std::function<void()> &mutate);
	// Sun & sky (src/shared/scene_sky.h): while an Infinite light has it on, its picture and the scene's Sun follow its sun settings. The sun is brought into line
	// right after an edit that changed those settings (skySignature() differs from the one the document had before), and only then, so a Sun the user moved by
	// hand stays where it is until the sky is changed again.
	std::string skySignature() const;
	void syncSunAndSky();
	static QString skyImageFolder();   // where the generated sky pictures go: <user assets>/skies
	void pushUndo();
	bool restore(const std::string &json);   // false (and nothing changed) if the snapshot cannot be read
	void scheduleAutosave();
	void writeAutosave();
	void clearAutosave();
	bool loadAutosave();
	bool confirmDiscard();

	// Inspector building blocks. Each binds to the live value through a pointer-returning function, so a rebuilt document is never dangling.
	void addNum(QFormLayout *f, const QString &label, const std::function<double *()> &ref, double lo, double hi, double step, int decimals = 3,
	            const QString &suffix = QString());
	void addInt(QFormLayout *f, const QString &label, const std::function<int *()> &ref, int lo, int hi);
	void addVec3(QFormLayout *f, const QString &label, const std::function<scene_doc::Float3 *()> &ref, double step);
	void addColor(QFormLayout *f, const QString &label, const std::function<scene_doc::Rgb *()> &ref);
	void addBool(QFormLayout *f, const QString &label, const std::function<bool *()> &ref, bool rebuildAfter = false);
	void addText(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref);
	// A file row: read-only path, Browse (and Clear, when `clearable`). `alsoApply`, if given, runs inside the same undo step as the choice.
	void addFile(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref, const QString &filter, bool clearable = false,
	             const std::function<void(const QString &)> &alsoApply = nullptr);
	void inspectCamera(QFormLayout *f);
	void inspectObject(QFormLayout *f, int i);
	void inspectLight(QFormLayout *f, int i);
	void inspectMaterial(QFormLayout *f, int i);

	QString workFolder() const;
	QString launcherPath() const;
	void runRender(int width, int height, int samples, bool toFinalFile, const QString &finalPng,
	               const std::function<void(bool, const QString &)> &done);

	scene_doc::Document m_doc;
	BuilderSelection m_sel;
	QString m_path;                  // the .pbrt file this scene was opened from or saved to; empty for an unsaved one
	bool m_dirty = false;
	bool m_loading = false;          // true while the inspector is being filled, so setting a value does not count as an edit
	SnapshotHistory m_history;   // undo and redo
	QStringList m_loggedProblems;   // the problems already written to the log, so each is said once when it appears
	std::vector<scene_doc::Problem> m_problems;   // validate() of the current document, refreshed by refreshProblems()
	QString m_lastEditKey;
	std::string m_skySignature;
	// What changed since the last undo step began, for the log file: one line per step ("object 'Ball': position 0,1,0 -> 2,1,0"), not one per drag movement.
	scene_doc::Document m_logBase;
	bool m_logPending = false;
	QTimer *m_logTimer = nullptr;
	void flushEditLog();
	QElapsedTimer m_editClock;
	int m_editCounter = 0;
	bool m_autosaveEnabled = true;
	QTimer *m_autosaveTimer = nullptr;

	// UI
	QLabel *m_titleLabel = nullptr;
	QLineEdit *m_titleEdit = nullptr;   // the scene's name, editable in the toolbar
	QString m_listedPath;
	QListWidget *m_list = nullptr;
	SceneLayoutView *m_view = nullptr;
	Scene3DView *m_view3d = nullptr;
	QStackedWidget *m_viewStack = nullptr;  // m_view (Top / Front / Side) or m_view3d, one at a time
	QLabel *m_viewHint = nullptr;
	QWidget *m_gizmoBar = nullptr;           // the 3D view's Move / Rotate / Scale buttons
	// scene_builder_views.cpp: building the views, and doing one thing to both (they follow the same document and selection).
	void createViews(QWidget *layoutBox, QVBoxLayout *layoutLayout);
	void updateViews();
	void selectInViews(const BuilderSelection &s);
	void updateGizmoButtons(const BuilderSelection &s);
	QButtonGroup *m_gizmoGroup = nullptr;  // the Move / Rotate / Scale buttons
	void frameViews();
	scene_doc::Float3 dropPoint() const;  // where a new object goes: the middle of the view on show
	QWidget *m_inspectorHost = nullptr;
	QScrollArea *m_inspectorScroll = nullptr;
	QLabel *m_problemsLabel = nullptr;
	QLabel *m_previewLabel = nullptr;
	QLabel *m_previewStatus = nullptr;
	QComboBox *m_qualityCombo = nullptr;
	QCheckBox *m_gpuCheck = nullptr;
	QPushButton *m_previewButton = nullptr;
	QPushButton *m_finalButton = nullptr;
	QPushButton *m_deleteButton = nullptr;
	QPushButton *m_duplicateButton = nullptr;
	QPushButton *m_addButton = nullptr;
	QWidget *m_inspectorPanel = nullptr;  // the properties column
	QSplitter *m_mainSplit = nullptr;     // list | view and preview | properties; its sizes are remembered between runs (window_geometry.h)
	QSplitter *m_centreSplit = nullptr;   // view above preview
	QWidget *m_leftPanel = nullptr;  // the list column; widened to fit its buttons when shown
	QPushButton *m_undoButton = nullptr;
	QPushButton *m_redoButton = nullptr;

	// A running render
	QProcess *m_process = nullptr;
	QElapsedTimer m_renderClock;
	QString m_previewPng;
	QString m_pendingFinalPng;
	std::function<void(bool, const QString &)> m_pendingDone;
	QString m_renderLog;
	bool m_cancelRequested = false;
	QPixmap m_previewPixmap;
	std::vector<std::function<void()>> m_refreshers;  // re-read each inspector field from the document (after a drag moves something)
};
