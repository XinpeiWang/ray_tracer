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

class QCheckBox;
class QComboBox;
class QFormLayout;
class QLabel;
class QListWidget;
class QProcess;
class QPushButton;
class QScrollArea;
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
	bool dragObjectForTest(int index, const QPointF &deltaPx);
	void addObject(scene_doc::ShapeKind shape);
	void addLight(scene_doc::LightKind kind);
	void deleteSelected();
	bool undo();
	bool redo();
	// Starts a preview render (no file dialog) and calls `done` when the picture is loaded or it failed.
	void startPreview(const std::function<void(bool ok, const QString &message)> &done = nullptr);
	QString previewImagePath() const { return m_previewPng; }
	bool isDirty() const { return m_dirty; }
	QString problemsText() const;
	// Where a scene is saved so the scene list finds it (empty if no scene folder exists); see saveToSceneList().
	static QString sceneListFolder();

signals:
	void statusMessage(const QString &text);

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
	void documentChanged();
	void refreshProblems();
	void updateTitle();
	void updateActions();

	// An edit: takes an undo snapshot (merging a run of edits with the same key, such as a spin box being dragged), applies `mutate`, and refreshes.
	void edit(const QString &key, const std::function<void()> &mutate);
	void pushUndo();
	void restore(const QString &json);
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
	void addFile(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref, const QString &filter);
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
	QStringList m_undo, m_redo;
	QString m_lastEditKey;
	QElapsedTimer m_editClock;
	int m_editCounter = 0;
	bool m_autosaveEnabled = true;
	QTimer *m_autosaveTimer = nullptr;

	// UI
	QLabel *m_titleLabel = nullptr;
	QListWidget *m_list = nullptr;
	SceneLayoutView *m_view = nullptr;
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
