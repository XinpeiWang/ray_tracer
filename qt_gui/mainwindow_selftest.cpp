// mainwindow_selftest.cpp - an opt-in automated smoke test of the real GUI (the project had none): run the app with
//   RT_GUI_SELFTEST=livepreview RT_GUI_SELFTEST_OUT=/tmp/gui_selftest  (optionally QT_QPA_PLATFORM=offscreen)
// and it drives the actual MainWindow - selects the Live Preview output mode, starts it, lets it render, saves a screenshot
// of its own window (never the screen) and a text log, and exits with 0 on success, 1 on failure, 2 if Live Preview is not
// available in this build.
#include "mainwindow.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QComboBox>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPixmap>
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

	if (mode == "options") {
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
		QTimer::singleShot(800, this, [this, shot]() {
			shot("options");
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
	if (mode == "builder") {
		resize(1500, 950);
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
		sb->selectObject(1);
		check(sb->problemsText().isEmpty(), "the scene has no problems or notes");
		const QString pbrt = outPrefix + "_builder.pbrt";
		check(sb->saveFile(pbrt) && !sb->isDirty(), "saved " + pbrt);
		const std::string before = scene_doc::toJson(sb->document());
		QString err;
		sb->newScene();
		check(sb->openFile(pbrt, &err) && scene_doc::toJson(sb->document()) == before, "re-opened the saved file unchanged " + err);
		// "Add to scene list": from inside a macOS .app bundle (or any read-only place) it must save to the per-user folder, never into the bundle, and
		// scene discovery must then list it as a user scene.
		{
			QString listError;
			const QString listed = sb->addToSceneList(&listError);
			check(!listed.isEmpty() && QFile::exists(listed), "added to the scene list: " + (listed.isEmpty() ? listError : listed));
			if (QCoreApplication::applicationDirPath().contains(".app/Contents/"))
				check(!listed.contains(".app/Contents/") && listed.contains("user_scenes"), "from a .app bundle the scene went to the per-user folder, not into the bundle");
			// (That scene discovery lists a file in that folder is covered by the unit test ScanTree.ScenesInTheUserFolder...)
			QFile::remove(listed);
		}
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

	// RT_GUI_SELFTEST=photo: see mainwindow_selftest_photo.cpp
	if (mode == "photo") { runPhotoSelfTest(log, shot); return; }
	if (mode == "installphoto") { runInstallPhotoSelfTest(log, shot); return; }

	// RT_GUI_SELFTEST=diagnostics: runs the Diagnostics tab's action (the CLI report plus the GUI's Network section) and logs the
	// finished report. RT_ASSET_BASE_URL can point the network check at a local or dead address.
	if (mode == "diagnostics") {
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
	if (mode == "scenekeys") {
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
			if (category == "Custom Scenes" || category == "Test Scenes" || SceneMetadataClient::sceneRequiresFiles(id)) continue;
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
	if (mode == "download") {
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

	if (mode == "ui") {
		// RT_GUI_SELFTEST_SCENE=<id> selects that scene first, so the scene info (including the
		// missing-files warning) can be checked; its text is logged as well as pictured.
		const QString sceneOverride = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE");
		if (!sceneOverride.isEmpty()) {
			selectSceneById(sceneOverride);
			resize(1100, 900);   // large enough to see the scene group, including the download button
		}
		log(QString("scene info: %1").arg(m_sceneInfoLabel ? m_sceneInfoLabel->text() : QString()));
		QTimer::singleShot(600, this, [shot]() { shot("ui"); QApplication::exit(0); });
		return;
	}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	// Starts Live Preview on every Metal-compatible scene in turn (one process, one scene at a time) and reports, per scene, the
	// frame count and how much of the tile is lit - to catch scenes that fail to load or start with a bad camera. Writes each
	// scene's displayed picture to <out>_sweep_<id>.png. Optional: RT_GUI_SELFTEST_SCENES=id1,id2,... to restrict it.
	if (mode == "livepreview_sweep") {
		const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
		if (idx < 0) { log("Live Preview mode missing"); QApplication::exit(2); return; }
		m_modeCombo->setCurrentIndex(idx);
		QStringList ids;
		const QStringList only = qEnvironmentVariable("RT_GUI_SELFTEST_SCENES").split(',', Qt::SkipEmptyParts);
		for (int i = 0, n = SceneMetadataClient::sceneCount(); i < n; ++i) {
			const QString id = SceneMetadataClient::sceneIdAtIndex(i);
			bool metal = false;
			if (!SceneMetadataClient::metalCompatible(id, metal) || !metal) continue;
			if (!only.isEmpty() && !only.contains(id)) continue;
			ids << id;
		}
		log(QString("sweep: %1 Metal-compatible scenes").arg(ids.size()));
		auto index = std::make_shared<int>(0);
		auto bad = std::make_shared<int>(0);
		auto step = std::make_shared<std::function<void()>>();
		*step = [this, ids, index, bad, step, log, outPrefix]() {
			if (*index >= ids.size()) {
				log(QString("sweep done: %1 of %2 scenes flagged").arg(*bad).arg(ids.size()));
				QApplication::exit(0);
				return;
			}
			const QString id = ids[*index];
			stopLivePreview();
			selectSceneById(id);
			startLivePreview();
			QTimer::singleShot(3000, this, [this, id, index, bad, step, log, outPrefix]() {
				const qint64 frames = m_livePreviewFrameCount;
				double lit = -1.0;
				if (m_livePreviewLabel) {
					const QImage shown = m_livePreviewLabel->pixmap().toImage();
					if (!shown.isNull()) {
						qint64 n = 0;
						for (int y = 0; y < shown.height(); ++y)
							for (int x = 0; x < shown.width(); ++x) {
								const QRgb p = shown.pixel(x, y);
								if (qRed(p) + qGreen(p) + qBlue(p) > 12) ++n;
							}
						lit = 100.0 * n / (double(shown.width()) * shown.height());
						shown.save(outPrefix + "_sweep_" + id + ".png");
					}
				}
				const QString status = m_livePreviewStatusLabel ? m_livePreviewStatusLabel->text() : QString();
				const bool flagged = frames < 5 || lit < 30.0;
				if (flagged) ++*bad;
				log(QString("%1 %2: frames=%3 lit=%4% status=\"%5\" camera=(%6, %7, %8) recommendedExposure=%9")
					.arg(flagged ? "FLAG" : "ok  ", id).arg(frames).arg(lit, 0, 'f', 1).arg(status)
					.arg(m_cameraPosX->value(), 0, 'g', 5).arg(m_cameraPosY->value(), 0, 'g', 5).arg(m_cameraPosZ->value(), 0, 'g', 5).arg(m_exposureSpin ? m_exposureSpin->value() : -1.0, 0, 'g', 4));
				++*index;
				(*step)();
			});
		};
		(*step)();
		return;
	}

	// Simulates a mouse drag (many small orbit steps over ~1.5 s) and writes a filmstrip of the displayed tile before, during and after
	// it: <out>_drag_strip.png. For judging what a user sees while orbiting: ghosting, streaks, how fast the picture settles.
	if (mode == "livepreview_drag") {
		const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
		if (idx < 0) { QApplication::exit(2); return; }
		m_modeCombo->setCurrentIndex(idx);
		const QString sceneOverride = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE");
		if (!sceneOverride.isEmpty()) selectSceneById(sceneOverride);
		if (qEnvironmentVariable("RT_GUI_SELFTEST_SMOOTH") == "0" && m_liveSmoothNoiseCheck) m_liveSmoothNoiseCheck->setChecked(false);
		if (qEnvironmentVariable("RT_GUI_SELFTEST_AUTOEXP") == "0" && m_liveAutoExposureCheck) m_liveAutoExposureCheck->setChecked(false);
		startLivePreview();
		auto frames = std::make_shared<QList<QImage>>();
		auto grabTile = [this, frames]() { if (m_livePreviewLabel) frames->append(m_livePreviewLabel->pixmap().toImage()); };
		const double total = qEnvironmentVariable("RT_GUI_SELFTEST_ORBIT", "0.5").toDouble();
		for (int step = 0; step < 36; ++step)   // 36 steps x 40 ms
			QTimer::singleShot(5000 + step * 40, this, [this, total]() {
				m_orbit.azimuth += total / 36.0;
				m_orbit.elevation += 0.1 * total / 36.0;
				updateLivePreviewCameraFromOrbit();
			});
		// before the drag, mid drag, end of drag, then 100 ms, 300 ms, 1 s, 4 s after
		const int marks[] = {4900, 5500, 6430, 6550, 6750, 7450, 10450};
		for (int t : marks) QTimer::singleShot(t, this, grabTile);
		QTimer::singleShot(10600, this, [this, frames, outPrefix, log]() {
			stopLivePreview();
			if (frames->isEmpty() || frames->first().isNull()) { log("no frames captured"); QApplication::exit(1); return; }
			const int w = frames->first().width(), h = frames->first().height(), n = frames->size();
			QImage strip(w * n + 4 * (n - 1), h, QImage::Format_RGB32);
			strip.fill(QColor(40, 40, 40));
			for (int i = 0; i < n; ++i) { QPainter p(&strip); p.drawImage(i * (w + 4), 0, (*frames)[i]); }
			strip.save(outPrefix + "_drag_strip.png");
			log("wrote the drag filmstrip");
			QApplication::exit(0);
		});
		return;
	}

	if (mode == "livepreview") {
		const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
		bool enabled = false;
		if (idx >= 0)
			if (auto *model = qobject_cast<QStandardItemModel *>(m_modeCombo->model()))
				if (QStandardItem *item = model->item(idx)) enabled = item->isEnabled();
		if (!enabled) {
			log("Live Preview is present=" + QString::number(idx >= 0) + " but NOT enabled (realtime_renderer library not found?)");
			shot("unavailable");
			QApplication::exit(2);
			return;
		}
		m_modeCombo->setCurrentIndex(idx);
		if (qEnvironmentVariable("RT_GUI_SELFTEST_SMOOTH") == "0" && m_liveSmoothNoiseCheck) m_liveSmoothNoiseCheck->setChecked(false);
		if (qEnvironmentVariable("RT_GUI_SELFTEST_AUTOEXP") == "0" && m_liveAutoExposureCheck) m_liveAutoExposureCheck->setChecked(false);
		// Optional overrides so the same test can cover other scenes / Resolution settings:
		//   RT_GUI_SELFTEST_SCENE=<scene id>   RT_GUI_SELFTEST_RES=<W>x<H>
		const QString sceneOverride = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE");
		if (!sceneOverride.isEmpty()) selectSceneById(sceneOverride);
		const QString resOverride = qEnvironmentVariable("RT_GUI_SELFTEST_RES");
		if (const QRegularExpressionMatch rm = QRegularExpression(R"(^(\d+)x(\d+)$)").match(resOverride); rm.hasMatch()) {
			m_widthSpinBox->setValue(rm.captured(1).toInt());
			m_heightSpinBox->setValue(rm.captured(2).toInt());
		}
		log(QString("scene=%1, Resolution setting %2x%3").arg(m_sceneCombo->currentData().toString()).arg(m_widthSpinBox->value()).arg(m_heightSpinBox->value()));
		log(QString("camera spinboxes (%1, %2, %3), look-at (%4, %5, %6)").arg(m_cameraPosX->value()).arg(m_cameraPosY->value()).arg(m_cameraPosZ->value())
			.arg(m_currentLookatX).arg(m_currentLookatY).arg(m_currentLookatZ));
		startLivePreview();
		log(QString("started: running=%1").arg(m_livePreviewRunning ? 1 : 0));
		// The preview must frame the scene like the image render does: same aspect ratio as the Resolution setting.
		{
			const QString text = m_logTextEdit ? m_logTextEdit->toPlainText() : QString();
			QRegularExpressionMatch m = QRegularExpression(R"(\[Live Preview\] Starting: scene=\S+, (\d+)x(\d+))").match(text);
			if (m.hasMatch()) {
				const double pw = m.captured(1).toDouble(), ph = m.captured(2).toDouble();
				const double rw = m_widthSpinBox->value(), rh = m_heightSpinBox->value();
				log(QString("preview %1x%2 (aspect %3) vs Resolution setting %4x%5 (aspect %6)").arg(pw).arg(ph).arg(pw / ph, 0, 'f', 3)
					.arg(rw).arg(rh).arg(rw / rh, 0, 'f', 3));
				if (qAbs(pw / ph - rw / rh) > 0.02) { log("RESULT: FAIL (preview aspect differs from the Resolution setting)"); QApplication::exit(1); return; }
			} else {
				log("could not read the preview size from the log");
			}
		}
		// Let it render for a while, then look at what happened.
		// Timeline: look at 4 s, orbit the camera at 5 s (what a mouse drag does), look again at 9 s, report at 10 s.
		auto before = std::make_shared<QImage>();
		auto justAfterMove = std::make_shared<QImage>();
		auto litPercent = std::make_shared<double>(-1.0);
		QTimer::singleShot(4000, this, [this, log, shot, before, litPercent]() {
			log(QString("after 4 s: frames=%1").arg(m_livePreviewFrameCount));
			// What the tile really shows: the displayed picture's size and the bounding box of its non-black pixels.
			if (m_livePreviewLabel) {
				const QImage shown = m_livePreviewLabel->pixmap().toImage();
				int x0 = shown.width(), y0 = shown.height(), x1 = -1, y1 = -1;
				qint64 lit = 0;
				for (int y = 0; y < shown.height(); ++y)
					for (int x = 0; x < shown.width(); ++x) {
						const QRgb p = shown.pixel(x, y);
						if (qRed(p) + qGreen(p) + qBlue(p) > 12) { ++lit; x0 = qMin(x0, x); x1 = qMax(x1, x); y0 = qMin(y0, y); y1 = qMax(y1, y); }
					}
				if (!shown.isNull()) *litPercent = 100.0 * lit / (double(shown.width()) * shown.height());
				log(QString("displayed picture %1x%2, lit pixels %3%, lit box x[%4..%5] y[%6..%7]")
					.arg(shown.width()).arg(shown.height()).arg(shown.isNull() ? 0 : 100.0 * lit / (double(shown.width()) * shown.height()), 0, 'f', 1)
					.arg(x0).arg(x1).arg(y0).arg(y1));
			}
			shot("live_4s");
			*before = grab().toImage();
		});
		// How much of the old picture survives a camera move: the displayed tile a fixed number of frames after the move against the
		// settled one. With temporal reprojection the old accumulation is carried over, so the two are close; restarting from noise
		// (a few samples per pixel) puts them far apart. Counted in frames, not milliseconds, so it does not depend on the frame rate.
		QTimer::singleShot(5000, this, [this, log, justAfterMove]() {
			m_orbit.azimuth += qEnvironmentVariable("RT_GUI_SELFTEST_ORBIT", "0.15").toDouble();
			updateLivePreviewCameraFromOrbit();
			log(QString("orbited the camera by %1 rad").arg(qEnvironmentVariable("RT_GUI_SELFTEST_ORBIT", "0.15")));
			const qint64 frameAtMove = m_livePreviewFrameCount;
			auto poll = std::make_shared<std::function<void()>>();
			*poll = [this, justAfterMove, frameAtMove, poll]() {
				if (m_livePreviewFrameCount >= frameAtMove + 8) {   // the first couple of frames after the move may still be in flight
					if (m_livePreviewLabel) *justAfterMove = m_livePreviewLabel->pixmap().toImage();
					return;
				}
				QTimer::singleShot(2, this, *poll);
			};
			(*poll)();
		});
		QTimer::singleShot(9500, this, [this, log, justAfterMove, outPrefix]() {
			justAfterMove->save(outPrefix + "_after_move.png");
			if (!m_livePreviewLabel || justAfterMove->isNull()) return;
			const QImage settled = m_livePreviewLabel->pixmap().toImage();
			if (settled.size() != justAfterMove->size()) return;
			qint64 sum = 0, n = 0;
			for (int y = 0; y < settled.height(); ++y)
				for (int x = 0; x < settled.width(); ++x) {
					const QRgb a = justAfterMove->pixel(x, y), b = settled.pixel(x, y);
					sum += qAbs(qRed(a) - qRed(b)) + qAbs(qGreen(a) - qGreen(b)) + qAbs(qBlue(a) - qBlue(b));
					n += 3;
				}
			log(QString("recovery after the move: tile 8 frames after vs settled differ by %1 (mean abs diff per channel, 0-255)").arg(n ? double(sum) / n : 0.0, 0, 'f', 2));
		});
		QTimer::singleShot(10000, this, [this, log, shot, before, litPercent]() {
			const qint64 frames = m_livePreviewFrameCount;
			const double secs = m_livePreviewSessionTimer.elapsed() / 1000.0;
			log(QString("after 10 s: frames=%1 (%2 fps)").arg(frames).arg(secs > 0 ? frames / secs : 0.0, 0, 'f', 1));
			shot("live_10s");
			const QImage after = grab().toImage();
			double diff = 0;
			if (!before->isNull() && before->size() == after.size()) {
				qint64 sum = 0, n = 0;
				for (int y = 0; y < after.height(); y += 2)
					for (int x = 0; x < after.width(); x += 2) {
						const QRgb a = before->pixel(x, y), b = after.pixel(x, y);
						sum += qAbs(qRed(a) - qRed(b)) + qAbs(qGreen(a) - qGreen(b)) + qAbs(qBlue(a) - qBlue(b));
						n += 3;
					}
				diff = n ? double(sum) / n : 0.0;
			}
			log(QString("picture change after the camera move: %1 (mean abs diff per channel, 0-255)").arg(diff, 0, 'f', 2));
			// The picture must fill the tile (a wrong start camera showed a small patch in a black tile) - checked for the default scene only,
			// since other scenes can legitimately be dark.
			const bool fills = !qEnvironmentVariableIsSet("RT_GUI_SELFTEST_SCENE") ? *litPercent > 50.0 : true;
			if (!fills) log(QString("picture fills only %1% of the tile").arg(*litPercent, 0, 'f', 1));
			const bool ok = frames > 20 && diff > 1.0 && fills;
			// Depth of field: switch it on with a wide aperture focused well in front of the scene; the picture must change.
			const QImage beforeDof = m_livePreviewLabel ? m_livePreviewLabel->pixmap().toImage() : QImage();
			if (m_liveApertureSpin) m_liveApertureSpin->setValue(150.0);
			if (m_liveFocusDistanceSpin) m_liveFocusDistanceSpin->setValue(300.0);
			if (m_liveDofCheck) m_liveDofCheck->setChecked(true);
			log("enabled depth of field (aperture 150, focus 300)");
			QTimer::singleShot(2500, this, [this, log, shot, ok, beforeDof]() {
				shot("dof");
				double dofDiff = 0.0;
				const QImage afterDof = m_livePreviewLabel ? m_livePreviewLabel->pixmap().toImage() : QImage();
				if (!beforeDof.isNull() && beforeDof.size() == afterDof.size()) {
					qint64 sum = 0, n = 0;
					for (int y = 0; y < afterDof.height(); ++y)
						for (int x = 0; x < afterDof.width(); ++x) {
							const QRgb a = beforeDof.pixel(x, y), b = afterDof.pixel(x, y);
							sum += qAbs(qRed(a) - qRed(b)) + qAbs(qGreen(a) - qGreen(b)) + qAbs(qBlue(a) - qBlue(b));
							n += 3;
						}
					dofDiff = n ? double(sum) / n : 0.0;
				}
				log(QString("picture change after enabling depth of field: %1").arg(dofDiff, 0, 'f', 2));
				stopLivePreview();
				const bool dofOk = dofDiff > 1.0;
				log((ok && dofOk) ? "RESULT: OK" : "RESULT: FAIL (too few frames, no picture change on the camera move, a mostly empty tile, or no change from depth of field)");
				QApplication::exit((ok && dofOk) ? 0 : 1);
			});
		});
		return;
	}
#endif
	log("unknown or unavailable self-test mode: " + mode);
	QApplication::exit(2);
}
