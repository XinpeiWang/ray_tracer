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
#include "../src/shared/scene_blocks.h"
#include "../src/shared/scene_selection.h"
#include "../src/shared/scene_camera_views.h"

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

#include "scene_3d_view.h"
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
	// Several items at once (scene_builder_multi.cpp): Ctrl- or Shift-click and a box in the views, Ctrl/Shift-click in the list, and groups. Everything acts on all of them.
	void pickObjects(const std::vector<int> &objects, const std::vector<int> &lights = {});   // replaces the selection; the first one given is the main one
	scene_doc::ItemSet pickedItems() const;   // the main item and the others picked along with it (never the camera)
	void selectAll();                          // every object and light (Ctrl+A)
	void groupSelected();                      // the picked objects become one group (Ctrl+G); needs two or more
	void ungroupSelected();                    // dissolves the groups of the picked objects (Ctrl+Shift+G)
	void moveSelectedBy(const scene_doc::Float3 &delta);   // the picked items, together, one undo step
	void useLookOfMainObject();                // the other picked objects get the main one's material
	// Saved camera views (scene_builder_camera_views.cpp). Each change is one undo step.
	int saveCameraViewNamed(const QString &name, bool sceneCamera = false);   // the 3D view's place (or the camera, looking through it; or always the camera if `sceneCamera`); the new index, -1 if full
	void useCameraView(int index);            // the scene's camera goes to the saved view
	void updateCameraViewFromNow(int index, bool sceneCamera = false);
	void deleteCameraView(int index);
	void renameCameraView(int index, const QString &name);
	void cameraFromView();                    // the scene's camera goes where the 3D view is
	void lookThroughCamera(bool on);          // the 3D view looks through the scene's camera
	bool lookingThroughCameraForTest() const { return m_view3d && m_view3d->throughCamera(); }
	scene_doc::Float3 viewEyeForTest() const {   // where the 3D view's eye is now
		const scene_view::V3 e = m_view3d->currentPose().eye();
		return {e.x, e.y, e.z};
	}
	int viewComboCountForTest() const;
	bool dragPickedForTest(int index, const QPointF &deltaPx);   // a real drag in the layout view of one picked object (moves them all)
	void togglePickedForTest(const BuilderSelection &s) { togglePicked(s); }
	void clickItemForTest(const BuilderSelection &s, Qt::KeyboardModifiers mods = Qt::NoModifier);   // a real click on an item in the layout view
	int pickedRowsForTest() const;   // how many rows of the list are highlighted
	QPointF itemScreenPosForTest(const BuilderSelection &s) const;   // where the layout view draws an item
	// A box dragged in the layout view with Ctrl held, from one pixel position to another (real mouse events); the layout view must be the one showing.
	void boxSelectForTest(const QPointF &from, const QPointF &to);
	bool dragObjectForTest(int index, const QPointF &deltaPx);
	// The 3D view, for the self-test: show it (or go back to the 2D views), drag an object on the floor, or along one axis arrow of the selected item.
	void show3dView(bool on);
	void resetPaneSizes();   // View > Reset Window Layout
	bool dragObject3dForTest(int index, const QPointF &deltaPx, Qt::KeyboardModifiers mods = Qt::NoModifier);
	QPointF shiftPanBackground3dForTest(int index, const QPointF &deltaPx);
	bool dragAxis3dForTest(int index, int axis, double pixels);
	// Selects the object, presses on the inner part of its `axis` arrow (a `fraction` of the way to the tip: that is the object itself, not the arrow) and drags by
	// `deltaPx`; true if it moved in both X and Z, on the floor.
	bool dragInnerArrow3dForTest(int index, int axis, double fraction, const QPointF &deltaPx);
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
	void addBlocky(scene_doc::BlockyKind kind);   // a block, creature or thing made of boxes (scene_blocks.h), one undo step
	void duplicateSelected();                                  // a numbered copy beside the selected object or light (the Duplicate button, Ctrl+D)
	void addCopies(const std::vector<scene_doc::Object> &copies);   // objects made from another one (an array, a scatter): one undo step, the first selected
	void showArrayDialog();                                    // the Array... button: a grid, ring or scatter of copies of the selected object
	// The objects go in together at the drop point, named apart from what is there, as a group of that name (two or more) and all picked.
	void addParts(std::vector<scene_doc::Object> parts, const QString &groupName = QString());
	void addLight(scene_doc::LightKind kind);
	void addObjectFromPhoto();  // scene_builder_photo.cpp: needs the optional photo helper
	void showModelLibrary();    // scene_builder_models.cpp: the dialog behind Add > Model library...
	// Adds one library model (the stem of models/<stem>.obj) at a handy size, standing on the floor under the drop point, as one undo step. False when it is not found.
	bool addLibraryModel(const QString &stem);
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
	// Shows or hides the "Preview live" button (the main window knows whether a Live Preview library is there).
	void setLivePreviewAvailable(bool available);
	// Opens a scene file the Scene Builder saved, asking first what to do with unsaved changes; false when the user cancels or the file cannot be opened (then a message says why).
	bool openForEditing(const QString &path);
	// The scene's name (its title), as typed in the toolbar; setSceneName() is an undoable edit like typing there.
	QString sceneName() const { return QString::fromStdString(m_doc.title); }
	void setSceneName(const QString &name);

signals:
	void statusMessage(const QString &text);
	// A copy of the scene was saved into the scene-list folder ("Add to scene list"); the main window lists it without a restart.
	void sceneListed(const QString &path);
	// "Preview live": the scene was saved as a listing of its own (livePreviewCopy()) and should be opened in Live Preview; the main window does that.
	void livePreviewRequested(const QString &path);

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
	void onPreviewLiveClicked();
	// Saves the scene, titled "<name> (live preview)", as <name>-live-preview.pbrt in the scene-list folder (replaced each time: one preview copy per scene name), so
	// Live Preview can open it by a scene id without touching the scene's own listing or file. Returns the path, or "" with `error` set.
	QString writeLivePreviewCopy(QString *error);
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
	void setSelection(const BuilderSelection &s, bool fromList = false);   // one item (or none): the others are let go
	void applySelection(const BuilderSelection &s, bool fromList);      // m_sel is the main item; m_extra the others, as they are
	void syncListSelection();                                           // highlights the picked items' rows
	// scene_builder_multi.cpp
	void pickItems(const scene_doc::ItemSet &items, const BuilderSelection &mainItem = BuilderSelection{});
	void selectWithGroup(const BuilderSelection &s);   // a click in a view: the whole group when the object is in one
	void togglePicked(const BuilderSelection &s);      // Ctrl- or Shift-click: add or remove (a group goes in or out whole)
	void boxPicked(const QList<BuilderSelection> &items, bool additive);
	void inspectMultiple(QFormLayout *f);
	void inspectCameraViews(QFormLayout *f);
	void createCameraBar(QWidget *layoutBox);
	void refreshViewCombo();
	scene_doc::Camera cameraNow() const;
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
	std::vector<BuilderSelection> m_extra;   // the other items picked along with m_sel
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
	QWidget *m_cameraBar = nullptr;          // the 3D view's camera buttons: Through camera, Camera from view, the saved views
	QPushButton *m_throughButton = nullptr;
	QComboBox *m_viewCombo = nullptr;
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
	QPushButton *m_arrayButton = nullptr;
	QPushButton *m_addButton = nullptr;
	QWidget *m_inspectorPanel = nullptr;  // the properties column
	QSplitter *m_mainSplit = nullptr;     // list | view and preview | properties; its sizes are remembered between runs (window_geometry.h)
	QSplitter *m_centreSplit = nullptr;   // view above preview
	QWidget *m_leftPanel = nullptr;  // the list column; widened to fit its buttons when shown
	QPushButton *m_liveButton = nullptr;   // "Preview live"; hidden until the main window says Live Preview is there
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
