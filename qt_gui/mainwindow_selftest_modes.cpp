// mainwindow_selftest_modes.cpp - the smaller self-test modes (options, builder, diagnostics, scenekeys, download, ui), each its own function called by
// runSelfTest() in mainwindow_selftest.cpp. A pure split of what used to be one 500-line function.

#include "mainwindow.h"
#include "../src/shared/oidn_runtime.h"
#include "denoiser_installer.h"
#include "live_ai_denoise.h"
#include "app_log.h"
#include <iostream>
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QAbstractSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
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

void MainWindow::runOptionsSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	// Every button, drop-down, line edit and spin box is the same height (46px: a 40px box and a 3px margin above and below - see the QPushButton rule in
	// mainwindow_style.cpp), on every tab. Looks at the visible ones; a control that sets its own height shows up here.
	{
		resize(1500, 1000);
		show();
		QStringList wrong;
		int checked = 0;
		for (int t = 0; t < m_tabWidget->count(); ++t) {
			m_tabWidget->setCurrentIndex(t);
			QApplication::processEvents();
			for (QWidget *w : m_tabWidget->currentWidget()->findChildren<QWidget *>()) {
				if (!w->isVisible()) continue;
				QString kind;
				if (qobject_cast<QPushButton *>(w)) kind = "button";
				else if (qobject_cast<QComboBox *>(w)) kind = "drop-down";
				else if (qobject_cast<QAbstractSpinBox *>(w)) kind = "spin box";
				else if (qobject_cast<QLineEdit *>(w) && !qobject_cast<QAbstractSpinBox *>(w->parentWidget()) && !qobject_cast<QComboBox *>(w->parentWidget())) kind = "line edit";
				else continue;
				++checked;
				if (w->height() != 46) {
					const QString text = qobject_cast<QPushButton *>(w) ? qobject_cast<QPushButton *>(w)->text() : QString();
					wrong << QString("%1 \"%2\" %3 (%4): %5px").arg(kind, text, w->objectName(), m_tabWidget->tabText(t)).arg(w->height());
				}
			}
		}
		log(QString("%1: %2 controls checked, every one 46px tall%3").arg(wrong.isEmpty() ? "ok" : "FAIL").arg(checked).arg(wrong.isEmpty() ? QString() : QString(" - not: ") + wrong.join("; ")));
		if (!wrong.isEmpty()) { QApplication::exit(1); return; }
	}
	// The Render Options tab with Live Preview selected, large enough to read: which live controls does this platform show?
	resize(1100, 1500);
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
	if (idx >= 0) m_modeCombo->setCurrentIndex(idx);
#endif
	for (int i = 0; i < m_tabWidget->count(); ++i) {
		log(QString("tab %1: %2").arg(i).arg(m_tabWidget->tabText(i)));
		if (m_tabWidget->tabText(i).contains("Render Options")) m_tabWidget->setCurrentIndex(i);
	}
	QTimer::singleShot(800, this, [this, shot, log]() {
		shot("options");
#ifdef Q_OS_MAC
		// Ticking "AI denoise" (Live Preview) with no denoiser library present offers to download it; declining must untick the box again and not leave it
		// remembered as on. (Skipped when this machine has the library, or the app cannot install it.)
		QCheckBox *aiBox = nullptr;
		for (QCheckBox *c : findChildren<QCheckBox *>())
			if (c->text() == tr("AI denoise")) aiBox = c;
		if (aiBox && !oidn_runtime::available() && denoiser_installer::isSupportedHere()) {
			QTimer::singleShot(400, this, []() {
				if (auto *question = qobject_cast<QMessageBox *>(QApplication::activeModalWidget()))
					for (QAbstractButton *b : question->buttons())
						if (question->buttonRole(b) == QMessageBox::NoRole) b->click();
			});
			aiBox->setChecked(true);   // returns once the question is answered
			const bool untickedAgain = !aiBox->isChecked() && !live_ai_denoise::savedEnabled();
			log(QString("%1: declining the denoiser download unticks \"AI denoise\" and does not remember it").arg(untickedAgain ? "ok" : "FAIL"));
			if (!untickedAgain) { QApplication::exit(1); return; }
		} else {
			log("skipped: the AI denoise download prompt (the denoiser library is present here, or not installable on this platform)");
		}
#endif
		// ...and scrolled to the bottom, where the Live Preview Settings group is.
		QList<QScrollArea *> areas = m_tabWidget->currentWidget()->findChildren<QScrollArea *>();
		if (auto *self = qobject_cast<QScrollArea *>(m_tabWidget->currentWidget())) areas.prepend(self);
		for (QScrollArea *sa : areas)
			sa->verticalScrollBar()->setValue(sa->verticalScrollBar()->maximum());
		QTimer::singleShot(500, this, [shot]() { shot("options_bottom"); QApplication::exit(0); });
	});
	return;
}

	// RT_GUI_SELFTEST=builder: drives the Scene Builder tab through an edit, undo/redo, a save and re-open, and a real preview render (needs
	// ray_tracer next to the GUI); saves screenshots <out>_builder_edit.png / _builder_preview.png and exits 0 if every step held.
void MainWindow::runBuilderSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot, const QString &outPrefix) {
	resize(qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") : 1500, qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") : 950);  // (..._WIDTH / ..._HEIGHT: a smaller window)
	if (m_sceneBuilder) m_tabWidget->setCurrentWidget(m_sceneBuilder);  // by widget, so it works in every language
	SceneBuilderWidget *sb = m_sceneBuilder;
	bool ok = sb != nullptr;
	auto check = [&ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		ok = ok && cond;
	};
	check(sb && sb->document().objects.size() == 5 && sb->document().lights.size() == 1, "starter scene has 5 objects and 1 light");
	const size_t objects0 = sb->document().objects.size();
	sb->addObject(scene_doc::ShapeKind::Cylinder);
	sb->addLight(scene_doc::LightKind::Spot);
	check(sb->document().objects.size() == objects0 + 1 && sb->document().lights.size() == 2, "added a cylinder and a spotlight");
	check(sb->isDirty(), "the scene is marked unsaved");
	check(sb->undo() && sb->document().lights.size() == 1, "undo removes the spotlight");
	check(sb->redo() && sb->document().lights.size() == 2, "redo brings it back");
	// A real mouse drag in the layout view moves the object (snapped to the 0.25 grid) as one undo step.
	const scene_doc::Float3 ball0 = sb->document().objects[1].position;
	check(sb->dragObjectForTest(1, QPointF(80, -40)), "dragging the glass ball in the layout view moves it");
	const scene_doc::Float3 ball1 = sb->document().objects[1].position;
	check(ball1.x > ball0.x + 0.3 && ball1.z < ball0.z - 0.1 && ball1.y == ball0.y, "it moved right and away from the camera, height unchanged");
	check(std::fabs(ball1.x * 4 - std::round(ball1.x * 4)) < 1e-9, "the new position is on the grid");
	check(sb->undo() && sb->document().objects[1].position.x == ball0.x && sb->document().objects[1].position.z == ball0.z, "one undo puts it back");
	check(sb->redo(), "redo moves it again");
	selfTestShapes(sb, check);
	sb->selectObject(1);
	check(sb->problemsText().isEmpty(), "the scene has no problems or notes");
	const QString pbrt = outPrefix + "_builder.pbrt";
	check(sb->saveFile(pbrt) && !sb->isDirty(), "saved " + pbrt);
	const std::string before = scene_doc::toJson(sb->document());
	QString err;
	sb->newScene();
	check(sb->openFile(pbrt, &err) && scene_doc::toJson(sb->document()) == before, "re-opened the saved file unchanged " + err);
	selfTestSceneList(sb, check);
	selfTestDeleteScenes(sb, check);
	selfTestLog(sb, check);
	selfTestWheelGuard(check);
	selfTestWindowGeometry(check);
	selfTestCrashRecovery(check);
	selfTestRenderQueue(check);
	// The screenshots show the starter scene (the edits above are done), with the gold ball picked.
	sb->newScene();
	sb->selectObject(2);
	QTimer::singleShot(600, this, [this, shot, log, sb, ok]() mutable {
		shot("builder_edit");
		// One preview on the CPU, then (RT_GUI_SELFTEST_GPU=1, set by scripts/gui_selftest.py --live-preview on a machine with a GPU)
		// one through "Use the GPU" - Metal on a Mac - which must also give a lit picture of about the same brightness.
		const bool alsoGpu = qEnvironmentVariableIsSet("RT_GUI_SELFTEST_GPU");
		auto meanGrey = [](const QString &path) {
			const QImage img(path);
			double sum = 0;
			for (int y = 0; y < img.height(); ++y)
				for (int x = 0; x < img.width(); ++x) sum += qGray(img.pixel(x, y));
			return img.isNull() ? 0.0 : sum / (double(img.width()) * img.height());
		};
		log("starting a preview render");
		sb->startPreview([this, shot, log, sb, ok, alsoGpu, meanGrey](bool done, const QString &message) mutable {
			log(QString("preview: %1 - %2").arg(done ? "ok" : "FAIL", message));
			bool good = ok && done;
			double cpuMean = 0.0;
			if (done) {
				cpuMean = meanGrey(sb->previewImagePath());
				log(QString("preview picture, mean grey %1").arg(cpuMean));
				good = good && cpuMean > 10.0 && cpuMean < 245.0;
			}
			const auto finish = [this, shot, log](bool result) {
				QTimer::singleShot(300, this, [shot, log, result]() {
					shot("builder_preview");
					log(result ? "RESULT: OK" : "RESULT: FAIL");
					QApplication::exit(result ? 0 : 1);
				});
			};
			if (!alsoGpu || !good) { finish(good); return; }
			sb->setUseGpu(true);
			log("starting a preview render on the GPU");
			sb->startPreview([log, sb, cpuMean, meanGrey, finish](bool gpuDone, const QString &gpuMessage) {
				log(QString("GPU preview: %1 - %2").arg(gpuDone ? "ok" : "FAIL", gpuMessage));
				bool gpuGood = gpuDone;
				if (gpuDone) {
					const double gpuMean = meanGrey(sb->previewImagePath());
					log(QString("GPU preview picture, mean grey %1 (CPU %2)").arg(gpuMean).arg(cpuMean));
					// Different renderers and sample noise, but the same scene: the same order of brightness.
					gpuGood = gpuMean > 10.0 && gpuMean < 245.0 && std::abs(gpuMean - cpuMean) < 40.0;
				}
				sb->setUseGpu(false);
				finish(gpuGood);
			});
		});
	});
	return;
}

	// RT_GUI_SELFTEST=diagnostics: runs the Diagnostics tab's action (the CLI report plus the GUI's Network section) and logs the
	// finished report. RT_ASSET_BASE_URL can point the network check at a local or dead address.
void MainWindow::runDiagnosticsSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	for (int i = 0; i < m_tabWidget->count(); ++i)
		if (m_tabWidget->tabText(i).contains("Diagnostics")) m_tabWidget->setCurrentIndex(i);
	onRunDiagnosticsClicked();
	auto *poll = new QTimer(this);
	auto *waited = new int(0);
	connect(poll, &QTimer::timeout, this, [this, poll, waited, log]() {
		const bool done = !diagnosticsBusy() && m_lastDiagReport.contains("=== Photo helper");
		if (!done && ++*waited < 240) return;
		poll->stop();
		log(m_lastDiagReport);
		const bool sections = m_lastDiagReport.contains("=== Network ===") && m_lastDiagReport.contains("=== Photo helper");
		log(sections ? "RESULT: OK" : "RESULT: FAIL (no Network or Photo helper section)");
		QApplication::exit(sections ? 0 : 1);
	});
	poll->start(500);
	return;
}

	// RT_GUI_SELFTEST=scenekeys: the scene keys the GUI saves and looks up by (slugs, with ids still accepted) round-trip through the metadata library, the
	// technique notes are found by either, and the thumbnail cache is keyed by slug. Exit 0 if every check held.
void MainWindow::runSceneKeysSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	bool ok = true;
	auto check = [&ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		ok = ok && cond;
	};
	const QString slug = SceneMetadataClient::sceneSlug("A1");
	check(slug == "cornell-box", "A1's slug is cornell-box (got \"" + slug + "\")");
	check(SceneMetadataClient::sceneIdForKey("cornell-box") == "A1", "the slug resolves to A1");
	check(SceneMetadataClient::sceneIdForKey("A1") == "A1", "an id resolves to itself");
	check(SceneMetadataClient::sceneIdForKey("no-such-scene").isEmpty(), "an unknown key resolves to nothing");
	check(scene_technique_notes::hasNote("A1") && scene_technique_notes::hasNote("cornell-box"), "the Cornell Box note is found by id and by slug");
	check(scene_technique_notes::forScene("A1") == scene_technique_notes::forScene("cornell-box"), "both give the same note");
	check(thumbnailCachePath("A1").endsWith("/thumbnails/cornell-box.png"), "the thumbnail cache file is named by slug: " + thumbnailCachePath("A1"));
	// every compiled-in scene's slug is unique and resolves back to its own id
	QSet<QString> slugs;
	int count = SceneMetadataClient::sceneCount(), bad = 0;
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		const QString s = SceneMetadataClient::sceneSlug(id);
		if (s == id || slugs.contains(s) || SceneMetadataClient::sceneIdForKey(s) != id) ++bad;
		slugs.insert(s);
	}
	check(bad == 0, QString("%1 scenes: every slug is unique and resolves back to its id (%2 bad)").arg(count).arg(bad));
	// The technique notes (scene_technique_notes.h, translated GUI text, so kept out of the Qt-free registry) and the registry must describe the
	// same scenes: every self-contained built-in scene has a note, and every note belongs to one.
	QSet<QString> expected, missingNote, staleNote;
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		const QString category = SceneMetadataClient::sceneCategory(id);
		if (category == "Custom Scenes" || category == "Test Scenes" || category == "My Scenes" || SceneMetadataClient::sceneRequiresFiles(id)) continue;
		expected.insert(SceneMetadataClient::sceneSlug(id));
	}
	for (const QString &slug : expected)
		if (!scene_technique_notes::notes().contains(slug)) missingNote.insert(slug);
	for (auto it = scene_technique_notes::notes().constBegin(); it != scene_technique_notes::notes().constEnd(); ++it)
		if (!expected.contains(it.key())) staleNote.insert(it.key());
	check(missingNote.isEmpty(), QString("%1 self-contained scenes, %2 notes: missing a note: %3").arg(expected.size()).arg(scene_technique_notes::notes().size()).arg(QStringList(missingNote.begin(), missingNote.end()).join(", ")));
	check(staleNote.isEmpty(), "notes for no self-contained scene: " + QStringList(staleNote.begin(), staleNote.end()).join(", "));
	log(ok ? "RESULT: OK" : "RESULT: FAIL");
	QApplication::exit(ok ? 0 : 1);
	return;
}

	// RT_GUI_SELFTEST=download RT_GUI_SELFTEST_SCENE=<id>: selects the scene, runs the "Download missing files" action
	// without dialogs (point RT_ASSET_BASE_URL at a local server to avoid the network), and checks the files arrived and
	// the scene no longer reports them missing. Exit 0 on success.
void MainWindow::runDownloadSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	selectSceneById(qEnvironmentVariable("RT_GUI_SELFTEST_SCENE"));
	const QList<asset_downloader::Job> jobs = m_downloadableAssetJobs;
	const scene_packs::Pack *pack = m_downloadablePack;
	if (jobs.isEmpty() && pack) {
		log(QString("scene=%1, pack \"%2\": %3 file(s), %4 MB to download").arg(m_sceneCombo->currentData().toString(), pack->title).arg(pack->fileCount()).arg(pack->downloadBytes() / 1.0e6, 0, 'f', 1));
		QTimer::singleShot(1800000, this, [log]() { log("RESULT: FAIL (timed out)"); QApplication::exit(1); });
		startPackDownload(*pack, false, [this, log](bool ok, const QString &error) {
			const bool stillMissing = SceneMetadataClient::missingAssets(m_sceneCombo->currentData().toString()).any;
			log(QString("download ok=%1 error=\"%2\" scene-still-reports-missing=%3").arg(ok).arg(error).arg(stillMissing));
			log(ok && !stillMissing ? "RESULT: OK" : "RESULT: FAIL");
			QApplication::exit(ok && !stillMissing ? 0 : 1);
		});
		return;
	}
	log(QString("scene=%1, %2 downloadable missing file(s), button visible: %3")
		.arg(m_sceneCombo->currentData().toString()).arg(jobs.size()).arg(m_downloadAssetsButton && !m_downloadAssetsButton->isHidden()));
	if (jobs.isEmpty()) { log("RESULT: FAIL (nothing to download for this scene)"); QApplication::exit(1); return; }
	QTimer::singleShot(120000, this, [log]() { log("RESULT: FAIL (timed out)"); QApplication::exit(1); });
	startAssetDownload(jobs, false, [this, jobs, log](bool ok, const QString &error) {
		bool allThere = ok;
		for (const auto &j : jobs) allThere = allThere && QFileInfo(j.destination).size() == j.entry.size;
		const bool stillMissing = SceneMetadataClient::missingAssets(m_sceneCombo->currentData().toString()).any;
		log(QString("download ok=%1 error=\"%2\" files-present=%3 scene-still-reports-missing=%4").arg(ok).arg(error).arg(allThere).arg(stillMissing));
		log(allThere && !stillMissing ? "RESULT: OK" : "RESULT: FAIL");
		QApplication::exit(allThere && !stillMissing ? 0 : 1);
	});
	return;
}

void MainWindow::runUiSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	// RT_GUI_SELFTEST_SCENE=<id> selects that scene first, so the scene info (including the
	// missing-files warning) can be checked; its text is logged as well as pictured.
	const QString sceneOverride = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE");
	if (!sceneOverride.isEmpty()) {
		selectSceneById(sceneOverride);
		resize(1100, 900);   // large enough to see the scene group, including the download button
	}
	log(QString("scene info: %1").arg(m_sceneInfoLabel ? m_sceneInfoLabel->text() : QString()));
	// The renderer libraries print their errors to std::cerr. This program's stderr used to be left CLOSED on Windows (the redirect into the log failed after closing
	// it), so the first such line aborted the whole program (0xC0000409). Writing one here makes that regression a crash of the self-test; when the session log says
	// stderr goes into it, the line must have arrived there.
	std::cerr << "[selftest] a line written to stderr" << std::endl;
	const QStringList logTail = AppLog::tail(400);
	const bool redirected = logTail.join('\n').contains(QStringLiteral("stderr of this process"));
	const bool arrived = logTail.join('\n').contains(QStringLiteral("[selftest] a line written to stderr"));
	log(QString("stderr: %1").arg(redirected ? (arrived ? "written to the log" : "FAIL (redirected, but the line is not in the log)") : "left where it was (a terminal)"));
	if (redirected && !arrived) { QApplication::exit(1); return; }
	QTimer::singleShot(600, this, [shot]() { shot("ui"); QApplication::exit(0); });
	return;
}
