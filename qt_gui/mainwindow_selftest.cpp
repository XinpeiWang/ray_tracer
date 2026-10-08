// mainwindow_selftest.cpp - an opt-in automated smoke test of the real GUI (the project had none): run the app with
//   RT_GUI_SELFTEST=livepreview RT_GUI_SELFTEST_OUT=/tmp/gui_selftest  (optionally QT_QPA_PLATFORM=offscreen)
// and it drives the actual MainWindow - selects the Live Preview output mode, starts it, lets it render, saves a screenshot
// of its own window (never the screen) and a text log, and exits with 0 on success, 1 on failure, 2 if Live Preview is not
// available in this build.
#include "mainwindow.h"
#include "app_log.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPixmap>
#include <QProcess>
#include <QSysInfo>
#include <QPainter>
#include <QList>
#include <QScrollArea>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QTextStream>
#include <QTimer>
#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSpinBox>
#include <functional>
#include <memory>
#include <cmath>
#include "scene_builder_widget.h"
#include "scene_technique_notes.h"
#include "scene_metadata_client.h"

// The Scene Builder's "Add to scene list" checks: where the copy goes (never into a .app bundle), that it is in the scene list at once under its
// persistent id and category, that the renderer the GUI starts knows the id, that adding again never overwrites, and renaming + updating.
void MainWindow::selfTestSceneList(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	QString listError;
	const QString listed = sb->addToSceneList(&listError);
	check(!listed.isEmpty() && QFile::exists(listed), "added to the scene list: " + (listed.isEmpty() ? listError : listed));
	check(!listed.contains(".app/Contents/") && listed.contains("user_scenes"), "the scene went to the per-user folder (never into the program folder or a .app bundle)");
	// It must be in the scene list NOW, without a restart: the library lists it, under one id that the renderer this GUI starts also knows
	// (the id is a persistent number, not a position, so a separate process agrees on it).
	const int addedNow = SceneMetadataClient::refreshUserScenes();
	const QString listedId = SceneMetadataClient::sceneIdForFile(listed);
	check(addedNow >= 0 && !listedId.isEmpty(), "the scene is in the scene list without a restart: " + listedId);
	if (!listedId.isEmpty()) {
		check(SceneMetadataClient::sceneName(listedId) == QString("My scene"), "its name in the list is the title the builder saved");
		check(SceneMetadataClient::sceneCategory(listedId) == QString("My Scenes") && listedId.startsWith(QLatin1Char('M')), "it is in the My Scenes category: " + SceneMetadataClient::sceneCategory(listedId) + " / " + listedId);
		QProcess renderer;
		const QString exe = QCoreApplication::applicationDirPath() + QStringLiteral("/ray_tracer") + (QSysInfo::productType() == "windows" ? ".exe" : "");
		renderer.setWorkingDirectory(QCoreApplication::applicationDirPath());
		const QString outPpm = QFileInfo(listed).absolutePath() + "/listed_render.ppm";
		renderer.start(exe, QStringList() << "--cpu" << "--output" << outPpm << "64" << "4" << "3" << listedId);
		const bool ran = renderer.waitForFinished(180000) && renderer.exitCode() == 0;
		check(ran && QFileInfo(outPpm.left(outPpm.size() - 4) + ".png").exists() || ran && QFileInfo(outPpm).exists(),
		      "the renderer the GUI starts renders that scene by its id (exit " + QString::number(renderer.exitCode()) + ")");
		QFile::remove(outPpm);
		QFile::remove(outPpm.left(outPpm.size() - 4) + ".png");
		// What "Add to scene list" does in the real window: the main window lists the scene and selects it in the Settings tab's scene picker.
		emit sb->sceneListed(listed);
		check(m_sceneCombo && m_sceneCombo->currentData().toString() == listedId, "the Settings tab's scene picker now shows it selected");
		check(m_sceneCategoryTabs && m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString() == QString("My Scenes"), "the My Scenes tab is the one showing");
}
// Adding the same scene again never overwrites: the copy gets the next free name. Renaming it and choosing "Update the existing one"
// rewrites that listing, and the list shows the new name without a restart.
QString again;
const QString second = sb->addToSceneList(&again);
check(!second.isEmpty() && second != listed && QFile::exists(listed) && QFile::exists(second),
      "adding it again made a new file (" + QFileInfo(second).fileName() + "); the first one is untouched");
sb->setSceneName("Renamed room");
check(sb->sceneName() == "Renamed room", "the scene name can be edited");
const QString updated = sb->addToSceneList(&again, /*update=*/true);
check(updated == second, "'update the existing one' rewrites the listing it was added as");
SceneMetadataClient::refreshUserScenes();
check(SceneMetadataClient::sceneName(SceneMetadataClient::sceneIdForFile(second)) == QString("Renamed room"), "the scene list shows the new name without a restart");
QFile::remove(second);
QFile::remove(listed);
}

void MainWindow::runSelfTest(const QString &mode, const QString &outPrefix) {
	auto log = [outPrefix](const QString &line) {
		QFile f(outPrefix + ".txt");
		if (f.open(QIODevice::Append | QIODevice::Text)) QTextStream(&f) << line << "\n";
		QTextStream(stdout) << "[selftest] " << line << "\n";
	};
	auto shot = [this, outPrefix, log](const QString &tag) {
		const QString path = outPrefix + "_" + tag + ".png";
		const bool ok = grab().save(path);   // this window's own contents only
		log(QString("screenshot %1: %2").arg(path, ok ? "saved" : "FAILED"));
	};

	// Every Output Mode item and whether it can be selected.
	for (int i = 0; i < m_modeCombo->count(); ++i) {
		bool enabled = true;
		if (auto *model = qobject_cast<QStandardItemModel *>(m_modeCombo->model()))
			if (QStandardItem *item = model->item(i)) enabled = item->isEnabled();
		log(QString("output mode item %1: \"%2\" enabled=%3").arg(i).arg(m_modeCombo->itemText(i)).arg(enabled ? 1 : 0));
	}

	if (mode == "options") { runOptionsSelfTest(log, shot); return; }

	if (mode == "builder") { runBuilderSelfTest(log, shot, outPrefix); return; }

	// RT_GUI_SELFTEST=photo: see mainwindow_selftest_photo.cpp
	if (mode == "photo") { runPhotoSelfTest(log, shot); return; }
	if (mode == "builder3d") { runBuilder3dSelfTest(log, shot); return; }
	if (mode == "tour") { runTourSelfTest(log, shot); return; }
	if (mode == "queue") { runQueueSelfTest(log, shot); return; }
	if (mode == "installphoto") { runInstallPhotoSelfTest(log, shot); return; }

	if (mode == "diagnostics") { runDiagnosticsSelfTest(log, shot); return; }

	if (mode == "scenekeys") { runSceneKeysSelfTest(log, shot); return; }

	if (mode == "download") { runDownloadSelfTest(log, shot); return; }

	if (mode == "ui") { runUiSelfTest(log, shot); return; }

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	if (mode == "livepreview_sweep") { runLivePreviewSweepSelfTest(log, shot, outPrefix); return; }

	if (mode == "livepreview_drag") { runLivePreviewDragSelfTest(log, shot, outPrefix); return; }

	if (mode == "livepreview") { runLivePreviewSelfTest(log, shot, outPrefix); return; }
#endif
	log("unknown or unavailable self-test mode: " + mode);
	QApplication::exit(2);
}
