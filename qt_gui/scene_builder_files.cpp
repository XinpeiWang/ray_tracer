// scene_builder_files.cpp - the Scene Builder's scene files: new, open, save, and adding a scene to the scene list (see scene_builder_widget.h).
#include "scene_builder_widget.h"

#include "scene_builder_common.h"
#include "app_log.h"
#include "window_geometry.h"
#include "atomic_file.h"
#include "../src/shared/scene_doc_diff.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

using namespace scene_builder_ui;

// ---- document state -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::newScene() {
	flushEditLog();
	m_logPending = false;
	AppLog::info(QStringLiteral("builder"), QStringLiteral("new scene (the starter scene)"));
	m_doc = scene_doc::makeStarterScene();
	m_path.clear();
	m_listedPath.clear();
	m_dirty = false;
	m_history.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	frameViews();
	refreshProblems();
	updateTitle();
	updateActions();
}

bool SceneBuilderWidget::openFile(const QString &path, QString *error) {
	flushEditLog();
	QFile f(path);
	if (!f.open(QIODevice::ReadOnly)) {
		if (error) *error = tr("Cannot open %1.").arg(path);
		AppLog::error(QStringLiteral("builder"), QStringLiteral("open %1: cannot open the file (%2)").arg(path, f.errorString()));
		return false;
	}
	const std::string text = f.readAll().toStdString();
	Document d;
	std::string err;
	if (!scene_doc::fromPbrt(text, d, err)) {
		if (error) *error = QString::fromStdString(err);
		AppLog::error(QStringLiteral("builder"), QStringLiteral("open %1: not a Scene Builder scene: %2").arg(path, QString::fromStdString(err)));
		return false;
	}
	AppLog::info(QStringLiteral("builder"), QStringLiteral("open %1: \"%2\", %3 objects, %4 lights").arg(path, QString::fromStdString(d.title)).arg(d.objects.size()).arg(d.lights.size()));
	m_logPending = false;
	m_doc = std::move(d);
	m_path = path;
	// A file opened from the scene-list folder is a listed scene: adding it again offers to update it.
	m_listedPath = QFileInfo(path).absolutePath() == QFileInfo(sceneListFolder() + "/x").absolutePath() ? path : QString();
	m_dirty = false;
	m_history.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	frameViews();
	refreshProblems();
	updateTitle();
	updateActions();
	return true;
}

// Writes the scene as pbrt text to `path`, without touching which file the document belongs to.
static bool writeSceneText(const Document &doc, const QString &path) {
	const std::string text = scene_doc::toPbrt(doc);
	return writeFileAtomically(path, QByteArray::fromRawData(text.data(), static_cast<qsizetype>(text.size())));   // the old file survives a failed write
}

bool SceneBuilderWidget::saveFile(const QString &path) {
	flushEditLog();
	if (!writeSceneText(m_doc, path)) {
		AppLog::error(QStringLiteral("builder"), QStringLiteral("save %1: could not write the file").arg(path));
		return false;
	}
	AppLog::info(QStringLiteral("builder"), QStringLiteral("save %1: \"%2\", %3 objects, %4 lights").arg(path, QString::fromStdString(m_doc.title)).arg(m_doc.objects.size()).arg(m_doc.lights.size()));
	m_path = path;
	m_dirty = false;
	clearAutosave();
	updateTitle();
	return true;
}

bool SceneBuilderWidget::confirmDiscard() {
	if (!m_dirty) return true;
	const auto answer = QMessageBox::question(this, tr("Unsaved changes"), tr("The scene has changes that are not saved. Save them first?"),
	                                          QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
	if (answer == QMessageBox::Cancel) return false;
	if (answer == QMessageBox::Save) {
		onSaveClicked();
		return !m_dirty;
	}
	return true;
}

void SceneBuilderWidget::onOpenClicked() {
	if (!confirmDiscard()) return;
	const QString path = QFileDialog::getOpenFileName(this, tr("Open a Scene Builder scene"), m_path.isEmpty() ? QDir::homePath() : QFileInfo(m_path).absolutePath(),
	                                                  tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	QString error;
	if (!openFile(path, &error)) QMessageBox::warning(this, tr("Cannot open the scene"), error);
}

void SceneBuilderWidget::onSaveClicked() {
	if (m_path.isEmpty()) {
		onSaveAsClicked();
		return;
	}
	if (!saveFile(m_path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(m_path));
	else emit statusMessage(tr("Saved %1").arg(m_path));
}

void SceneBuilderWidget::onSaveAsClicked() {
	QString start = m_path;
	if (start.isEmpty()) start = QDir::homePath() + "/" + QString::fromStdString(m_doc.title).replace(QRegularExpression("[^A-Za-z0-9_-]+"), "-") + ".pbrt";
	QString path = QFileDialog::getSaveFileName(this, tr("Save the scene"), start, tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	if (!path.endsWith(".pbrt", Qt::CaseInsensitive)) path += ".pbrt";
	if (!saveFile(path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(path));
	else emit statusMessage(tr("Saved %1").arg(path));
}

// Where "Add to scene list" saves: the per-user folder <data>/user_scenes (pbrt_discover::userSceneDir), always. That is the only folder the scene list can grow
// from while the program runs (refresh_user_scenes() rescans it, and gives each scene a persistent id), it is always writable, and it never writes into the
// program's own folder (a macOS .app bundle's seal breaks if you do; a disk image is read-only). A scene saved into the program's pbrt_scenes folder would
// only be listed after a restart, and its id would shift the ids of the scenes found after it. RAY_TRACER_PBRT_DIR still names a folder to use instead.
QString SceneBuilderWidget::sceneListFolder() {
	const QString env = qEnvironmentVariable("RAY_TRACER_PBRT_DIR");
	if (!env.isEmpty() && QDir(env).exists() && QFileInfo(env).isWritable()) return QDir(env).absolutePath();
	const QString user = QString::fromStdString(pbrt_asset_check::userSceneDir());
	if (!user.isEmpty() && QDir().mkpath(user)) return user;
	return QString();
}

// The generated sky pictures (Sun & sky) live beside the per-user scenes: <data>/skies, found from the same per-user root.
QString SceneBuilderWidget::skyImageFolder() {
	const QString scenes = QString::fromStdString(pbrt_asset_check::userSceneDir());
	const QString folder = !scenes.isEmpty() ? QFileInfo(scenes).dir().filePath(QStringLiteral("skies"))
	                                         : QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/user_assets/skies");
	QDir().mkpath(folder);
	return folder;
}

QString SceneBuilderWidget::addToSceneList(QString *error, bool update) {
	const QString folder = sceneListFolder();
	flushEditLog();
	if (folder.isEmpty()) {
		AppLog::error(QStringLiteral("builder"), QStringLiteral("add to scene list: no writable scenes folder"));
		if (error) *error = tr("The scenes folder (pbrt_scenes) was not found next to the program. Use Save As to put the file where you like, and set the "
		                       "environment variable RAY_TRACER_PBRT_DIR to that folder to have the program list it.");
		return QString();
	}
	QString name = QString::fromStdString(m_doc.title).trimmed().toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
	name.remove(QRegularExpression("^-+|-+$"));
	if (name.isEmpty()) name = "my-scene";
	QString path = folder + "/" + name + ".pbrt";
	if (update && !m_listedPath.isEmpty() && QFileInfo::exists(m_listedPath)) {
		path = m_listedPath;   // the listing this document already has
	} else {
		// A name already taken (by another scene, or by an earlier listing of this one) is never overwritten: the copy gets the next free name.
		for (int n = 2; QFileInfo::exists(path); ++n) path = folder + "/" + name + "-" + QString::number(n) + ".pbrt";
	}
	// A copy for the scene list: the document keeps its own file and its unsaved state.
	if (!writeSceneText(m_doc, path)) {
		AppLog::error(QStringLiteral("builder"), QStringLiteral("add to scene list: could not write %1").arg(path));
		if (error) *error = tr("Could not write %1.").arg(path);
		return QString();
	}
	AppLog::info(QStringLiteral("builder"), QStringLiteral("add to scene list: wrote %1 (%2)").arg(path, update ? QStringLiteral("updating the existing listing") : QStringLiteral("as a new scene")));
	m_listedPath = path;
	return path;
}

void SceneBuilderWidget::setLivePreviewAvailable(bool available) {
	if (m_liveButton) m_liveButton->setVisible(available);
}

QString SceneBuilderWidget::writeLivePreviewCopy(QString *error) {
	const QString folder = sceneListFolder();
	flushEditLog();
	if (folder.isEmpty()) {
		if (error) *error = tr("There is no scenes folder to write the preview copy to.");
		return QString();
	}
	QString name = QString::fromStdString(m_doc.title).trimmed().toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
	name.remove(QRegularExpression("^-+|-+$"));
	if (name.isEmpty()) name = "my-scene";
	const QString path = folder + "/" + name + "-live-preview.pbrt";
	Document copy = m_doc;
	copy.title += " (live preview)";
	if (!writeSceneText(copy, path)) {
		if (error) *error = tr("Could not write %1.").arg(path);
		return QString();
	}
	AppLog::info(QStringLiteral("builder"), QStringLiteral("preview live: wrote %1").arg(path));
	return path;
}

void SceneBuilderWidget::onPreviewLiveClicked() {
	QString error;
	const QString path = writeLivePreviewCopy(&error);
	if (path.isEmpty()) {
		QMessageBox::warning(this, tr("Cannot preview"), error);
		return;
	}
	emit sceneListed(path);   // the main window lists the copy ...
	emit livePreviewRequested(path);   // ... and opens it in Live Preview
}

bool SceneBuilderWidget::openForEditing(const QString &path) {
	if (!confirmDiscard()) return false;
	QString error;
	if (!openFile(path, &error)) {
		QMessageBox::warning(this, tr("Cannot open the scene"), error);
		return false;
	}
	return true;
}

void SceneBuilderWidget::onSaveToSceneListClicked() {
	const QString folder = sceneListFolder();
	if (folder.isEmpty()) {
		QString why;
		addToSceneList(&why);
		QMessageBox::information(this, tr("No scenes folder"), why);
		return;
	}
	// Already listed once (or opened from the list): ask whether this is an update of that scene or a new one. Otherwise there is nothing to ask - a
	// name that is taken just gets the next free one.
	bool update = false;
	if (!m_listedPath.isEmpty() && QFileInfo::exists(m_listedPath)) {
		QMessageBox box(QMessageBox::Question, tr("Add to the scene list"),
		                tr("This scene is already in the list as \"%1\".").arg(QFileInfo(m_listedPath).completeBaseName()), QMessageBox::NoButton, this);
		QPushButton *asNew = box.addButton(tr("Add as a new scene"), QMessageBox::AcceptRole);
		QPushButton *updateIt = box.addButton(tr("Update the existing one"), QMessageBox::DestructiveRole);
		box.addButton(QMessageBox::Cancel);
		box.setDefaultButton(asNew);
		box.exec();
		if (box.clickedButton() == updateIt) update = true;
		else if (box.clickedButton() != asNew) return;
	}
	QString error;
	const QString path = addToSceneList(&error, update);
	if (path.isEmpty()) {
		QMessageBox::warning(this, tr("Cannot save"), error);
		return;
	}
	emit sceneListed(path);   // the main window lists it and selects it - no restart needed
	QMessageBox::information(this, tr("Added to the scene list"),
	                         tr("Saved a copy as %1.\n\nIt is in the scene list now (Settings tab, My Scenes).").arg(path));
}
