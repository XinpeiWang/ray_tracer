// Deleting the scenes the user made (Settings tab, "My Scenes" category): one scene, or all of them. The files are moved to the Trash (the Recycle Bin on
// Windows), not erased, so a mistake can be undone from there; then the scene list is refreshed (the registry forgets scenes whose file is gone,
// scene_registry.h's refresh_user_scenes(true)) and the selection moves to a neighbouring scene.
#include "mainwindow.h"

#include "app_log.h"
#include "scene_builder_widget.h"
#include "scene_metadata_client.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>

namespace {
const char *const kMyScenesCategory = "My Scenes";   // SceneCategories::MyScenes: what SceneMetadataClient::sceneCategory() answers for them

QString trashName() {
#ifdef Q_OS_WIN
	return QObject::tr("the Recycle Bin");
#else
	return QObject::tr("the Trash");
#endif
}
}  // namespace

// The .pbrt file a scene was made from, if it is one of the files the Scene Builder saved into its scene-list folder (only those can be deleted from here);
// "" for any other scene.
QString MainWindow::userSceneFileForId(const QString &id) const {
	const QString folder = SceneBuilderWidget::sceneListFolder();
	if (folder.isEmpty() || id.isEmpty()) return QString();
	for (const QFileInfo &info : QDir(folder).entryInfoList({QStringLiteral("*.pbrt")}, QDir::Files))
		if (SceneMetadataClient::sceneIdForFile(info.absoluteFilePath()) == id) return info.absoluteFilePath();
	return QString();
}

// Every scene in the My Scenes category, not narrowed by the search box or the grid.
QStringList MainWindow::myScenesIds() const {
	QStringList ids;
	const int count = SceneMetadataClient::sceneCount();
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		if (SceneMetadataClient::sceneCategory(id) == QLatin1String(kMyScenesCategory)) ids << id;
	}
	return ids;
}

// Shown only under the My Scenes category; the buttons say what they would do.
void MainWindow::updateMyScenesButtons(const QString &category) {
	if (!m_myScenesRow) return;
	const bool mine = category == QLatin1String(kMyScenesCategory);
	m_myScenesRow->setVisible(mine);
	if (!mine) return;
	const QString id = m_sceneCombo ? m_sceneCombo->currentData().toString() : QString();
	const bool canDelete = !userSceneFileForId(id).isEmpty();
	m_deleteSceneButton->setEnabled(canDelete);
	m_deleteSceneButton->setToolTip(canDelete ? tr("Move the selected scene's file to %1").arg(trashName())
	                                          : tr("This scene is not one the Scene Builder saved into the scene list, so it cannot be deleted here"));
	m_deleteAllScenesButton->setEnabled(!myScenesIds().isEmpty());
}

// Moves `files` to the trash. Returns how many were moved; a file that cannot be moved there is reported, never erased behind the user's back.
static int moveFilesToTrash(const QStringList &files, QStringList *failed) {
	int moved = 0;
	// (Under the self-test the files are erased instead: the Trash belongs to the real user, and a test must not fill it.)
	const bool eraseInstead = qEnvironmentVariableIsSet("RT_GUI_SELFTEST");
	for (const QString &path : files) {
		if (eraseInstead ? QFile::remove(path) : QFile::moveToTrash(path)) {
			++moved;
			AppLog::info(QStringLiteral("scenes"), QStringLiteral("moved %1 to the trash").arg(path));
		} else {
			if (failed) *failed << path;
			AppLog::warn(QStringLiteral("scenes"), QStringLiteral("could not move %1 to the trash").arg(path));
		}
	}
	return moved;
}

// After files were removed: list what is left, drop the removed scenes' thumbnails, and select a neighbour (or the first scene of the whole list when none of the
// user's scenes remain: the My Scenes tab is gone then).
void MainWindow::afterUserScenesRemoved(const QStringList &removedIds, const QString &nextHintId) {
	for (const QString &id : removedIds) QFile::remove(thumbnailCachePath(id));   // before the registry forgets the id (the cache is keyed by its slug)
	SceneMetadataClient::refreshUserScenes();
	QString next = nextHintId;
	if (next.isEmpty() || SceneMetadataClient::sceneCategory(next).isEmpty()) {
		const QStringList mine = myScenesIds();
		next = !mine.isEmpty() ? mine.first() : SceneMetadataClient::sceneIdAtIndex(0);
	}
	selectSceneById(next);
}

// The deletion itself, without questions: the files of `ids` go to the trash and the list follows. Returns how many were removed; `failedIds` gets the ones
// whose file could not be moved (or that are not scenes of the Scene Builder's folder).
int MainWindow::deleteUserScenes(const QStringList &ids, const QString &nextHintId, QStringList *failedIds) {
	QStringList files, usable;
	for (const QString &id : ids) {
		const QString file = userSceneFileForId(id);
		if (file.isEmpty()) { if (failedIds) *failedIds << id; continue; }
		files << file;
		usable << id;
	}
	QStringList failedFiles;
	const int moved = moveFilesToTrash(files, &failedFiles);
	QStringList removedIds;
	for (int i = 0; i < usable.size(); ++i) {
		if (failedFiles.contains(files.at(i))) { if (failedIds) *failedIds << usable.at(i); }
		else removedIds << usable.at(i);
	}
	afterUserScenesRemoved(removedIds, nextHintId);
	AppLog::info(QStringLiteral("scenes"), QStringLiteral("deleted %1 of %2 scene(s)").arg(moved).arg(ids.size()));
	return moved;
}

void MainWindow::onDeleteSceneClicked() {
	const QString id = m_sceneCombo ? m_sceneCombo->currentData().toString() : QString();
	const QString file = userSceneFileForId(id);
	if (file.isEmpty()) return;
	const QString name = SceneMetadataClient::sceneName(id);
	const auto choice = QMessageBox::question(this, tr("Delete scene"),
		tr("Move the scene \"%1\" to %2?\n\nIts file is %3.").arg(name, trashName(), QDir::toNativeSeparators(file)),
		QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (choice != QMessageBox::Yes) return;

	// Neighbour to select afterwards: the next My Scenes entry (or the one before the last).
	const QStringList mine = myScenesIds();
	const int at = mine.indexOf(id);
	QString next;
	if (at >= 0 && mine.size() > 1) next = mine.at(at + 1 < mine.size() ? at + 1 : at - 1);

	if (deleteUserScenes({id}, next) == 0) {
		QMessageBox::warning(this, tr("Delete scene"), tr("Could not move %1 to %2. The scene was left as it is.").arg(QDir::toNativeSeparators(file), trashName()));
		return;
	}
	statusBar()->showMessage(tr("Moved \"%1\" to %2.").arg(name, trashName()), 8000);
}

void MainWindow::onDeleteAllMyScenesClicked() {
	QStringList ids, files;
	for (const QString &id : myScenesIds()) {
		const QString file = userSceneFileForId(id);
		if (!file.isEmpty()) { ids << id; files << file; }
	}
	if (files.isEmpty()) return;
	const auto choice = QMessageBox::question(this, tr("Delete all my scenes"),
		tr("Move all %n scene(s) you made to %1?\n\nThe scenes in the other categories are not touched. You can get them back from %1.", "", files.size()).arg(trashName()),
		QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (choice != QMessageBox::Yes) return;

	QStringList failed;
	const int moved = deleteUserScenes(ids, QString(), &failed);
	if (!failed.isEmpty())
		QMessageBox::warning(this, tr("Delete all my scenes"), tr("%n scene(s) could not be moved to %1 and were left as they are.", "", failed.size()).arg(trashName()));
	statusBar()->showMessage(tr("Moved %n scene(s) to %1.", "", moved).arg(trashName()), 8000);
}
