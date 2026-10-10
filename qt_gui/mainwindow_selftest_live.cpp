// mainwindow_selftest_live.cpp - the Live Preview self-test modes (livepreview, livepreview_sweep, livepreview_drag) (see mainwindow_selftest.cpp).

#include "mainwindow.h"
#include "live_ai_denoise.h"
#include "live_object_editor.h"
#include "app_log.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QCheckBox>
#include <QDir>
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
#include <QStatusBar>
#include <functional>
#include <memory>
#include <cmath>
#include "scene_builder_widget.h"
#include "scene_technique_notes.h"
#include "scene_metadata_client.h"

// The Scene Builder's "Add to scene list" checks: where the copy goes (never into a .app bundle), that it is in the scene list at once under its

#ifdef RT_GUI_HAVE_LIVE_PREVIEW

	// Starts Live Preview on every Metal-compatible scene in turn (one process, one scene at a time) and reports, per scene, the
	// frame count and how much of the tile is lit - to catch scenes that fail to load or start with a bad camera. Writes each
	// scene's displayed picture to <out>_sweep_<id>.png. Optional: RT_GUI_SELFTEST_SCENES=id1,id2,... to restrict it.
void MainWindow::runLivePreviewSweepSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot, const QString &outPrefix) {
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
// Starts Live Preview on each scene in turn (RT_GUI_SELFTEST_SCENES=id1,id2,...; default A1, the 555-unit Cornell box) and presses the Up key and the W key once: the
// camera must move by the step camera_math::keyboardStep() gives for that scene (2% of its size), and for a scene of any size that is a small fraction of it.
// Logs the scene's size, the distance moved and that distance as a percentage of the size; exits 1 if a press moved anything else or more than 5% of the scene.
void MainWindow::runLivePreviewKeysSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &, const QString &) {
	const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
	if (idx < 0) { log("Live Preview mode missing"); QApplication::exit(2); return; }
	m_modeCombo->setCurrentIndex(idx);
	QStringList ids = qEnvironmentVariable("RT_GUI_SELFTEST_SCENES").split(',', Qt::SkipEmptyParts);
	if (ids.isEmpty()) ids << "A1";
	auto index = std::make_shared<int>(0);
	auto bad = std::make_shared<int>(0);
	auto step = std::make_shared<std::function<void()>>();
	*step = [this, ids, index, bad, step, log]() {
		if (*index >= ids.size()) {
			log(QString("keys done: %1 of %2 scenes flagged").arg(*bad).arg(ids.size()));
			stopLivePreview();
			QApplication::exit(*bad ? 1 : 0);
			return;
		}
		const QString id = ids[*index];
		stopLivePreview();
		selectSceneById(id);
		startLivePreview();
		QTimer::singleShot(2500, this, [this, id, index, bad, step, log]() {
			if (!m_livePreviewRunning) { log(QString("FLAG %1: the preview did not start").arg(id)); ++*bad; ++*index; (*step)(); return; }
			SceneMetadataClient::SceneMetadata meta;
			SceneMetadataClient::sceneMetadata(id, meta);
			const camera_math::Vec3 before = m_livePreviewLookAt;
			onLivePreviewTranslate(0, 0, 1);   // the Up key
			const double up = camera_math::length(m_livePreviewLookAt - before);
			const camera_math::Vec3 mid = m_livePreviewLookAt;
			onLivePreviewTranslate(1, 0, 0);   // the W key
			const double forward = camera_math::length(m_livePreviewLookAt - mid);
			const double expected = camera_math::keyboardStep(m_livePreviewSceneSize, m_keyboardSensitivity);
			const double size = m_livePreviewSceneSize;
			const bool flagged = std::abs(up - expected) > 1e-6 * std::max(1.0, expected) || std::abs(forward - expected) > 1e-6 * std::max(1.0, expected)
				|| (size > 0.0 && up > 0.05 * size);
			if (flagged) ++*bad;
			log(QString("%1 %2: declared size %3, size used %4, one press moves %5 (%6% of the scene), expected %7")
				.arg(flagged ? "FLAG" : "ok  ", id).arg(meta.sceneSize, 0, 'g', 5).arg(size, 0, 'g', 5).arg(up, 0, 'g', 5)
				.arg(size > 0.0 ? 100.0 * up / size : 0.0, 0, 'f', 2).arg(expected, 0, 'g', 5));
			++*index;
			(*step)();
		});
	};
	(*step)();
}

// Moves an object with the mouse: starts Live Preview (scene RT_GUI_SELFTEST_SCENE, default A1), switches on "Move objects", presses on the middle of the picture
// and drags to the right, as real mouse events sent to the picture. The picture must change, the hint must say which object is being moved, and a "Reset objects"
// click must bring the picture back. Writes <out>_objects_before.png / _moved.png / _reset.png; exits 1 on a failure.
void MainWindow::runLivePreviewObjectsSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &, const QString &outPrefix) {
	const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
	if (idx < 0) { log("Live Preview mode missing"); QApplication::exit(2); return; }
	m_modeCombo->setCurrentIndex(idx);
	const QString scene = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE", "A1");
	selectSceneById(scene);
	if (m_liveSmoothNoiseCheck) m_liveSmoothNoiseCheck->setChecked(false);
	startLivePreview();
	auto shown = [this]() { return m_livePreviewLabel ? m_livePreviewLabel->pixmap().toImage() : QImage(); };
	auto diff = [](const QImage &a, const QImage &b) {   // mean absolute difference per channel, 0..255
		if (a.isNull() || b.isNull()) return -1.0;
		// The picture is shown scaled to the label, and a hint line under it can change the label's size: compare at one size.
		const QImage x = a.scaled(200, 200, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_RGB32);
		const QImage y = b.scaled(200, 200, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_RGB32);
		double sum = 0;
		for (int py = 0; py < x.height(); ++py)
			for (int px = 0; px < x.width(); ++px) {
				const QRgb p = x.pixel(px, py), q = y.pixel(px, py);
				sum += std::abs(qRed(p) - qRed(q)) + std::abs(qGreen(p) - qGreen(q)) + std::abs(qBlue(p) - qBlue(q));
			}
		return sum / (3.0 * x.width() * x.height());
	};
	auto before = std::make_shared<QImage>();
	auto moved = std::make_shared<QImage>();
	auto fail = [log](const QString &why) { log("FAIL: " + why); QApplication::exit(1); };
	QTimer::singleShot(4000, this, [this, shown, before, outPrefix, log, fail]() {
		if (!m_livePreviewRunning || !m_livePreviewLabel || !m_liveObjectEditor) { fail("the preview did not start with object editing available"); return; }
		*before = shown();
		before->save(outPrefix + "_objects_before.png");
		m_liveObjectEditor->setMode(true);
		const QRect image = m_livePreviewLabel->displayedImageRect();
		auto send = [this](QEvent::Type type, const QPoint &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
			QMouseEvent event(type, QPointF(pos), QPointF(m_livePreviewLabel->mapToGlobal(pos)), button, buttons, Qt::NoModifier);
			QApplication::sendEvent(m_livePreviewLabel, &event);
		};
		const QPoint start = image.center() + QPoint(0, image.height() / 5);   // a little below the middle: floor, a box or a ball rather than the back wall
		// Hovering first: with no button held, a box shows round the object under the cursor, and goes when the cursor leaves the picture.
		send(QEvent::MouseMove, start, Qt::NoButton, Qt::NoButton);
		QTimer::singleShot(900, this, [this, send, start, image, log, fail]() {
			if (!m_livePreviewLabel->hasHover()) { fail("hovering over an object did not highlight it"); return; }
			log("hovering highlights the object under the cursor");
			QEvent leave(QEvent::Leave);
			QApplication::sendEvent(m_livePreviewLabel, &leave);
			if (m_livePreviewLabel->hasHover()) { fail("the highlight stayed after the cursor left the picture"); return; }
			send(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
			log("after the press: \"" + m_liveObjectEditor->hint() + "\"");
			if (!m_liveObjectEditor->hint().startsWith("Selected")) { fail("the press did not grab an object"); return; }
			for (int i = 1; i <= 12; ++i) send(QEvent::MouseMove, start + QPoint(image.width() * i / 40, 0), Qt::NoButton, Qt::LeftButton);
			send(QEvent::MouseButtonRelease, start + QPoint(image.width() * 12 / 40, 0), Qt::LeftButton, Qt::NoButton);
		});
	});
	QTimer::singleShot(8000, this, [this, shown, before, moved, diff, outPrefix, log, fail]() {
		*moved = shown();
		moved->save(outPrefix + "_objects_moved.png");
		const double change = diff(*before, *moved);
		log(QString("the picture changed by %1 (of 255) after the drag").arg(change, 0, 'f', 3));
		if (!(change > 1.0)) { fail("dragging an object did not change the picture"); return; }
		// The camera moves with the object still selected: the selection must survive it.
		m_orbit.azimuth += 0.3;
		updateLivePreviewCameraFromOrbit();
	});
	auto beforeKeys = std::make_shared<QImage>();
	auto lookAtBefore = std::make_shared<camera_math::Vec3>();
	QTimer::singleShot(9500, this, [this, shown, beforeKeys, lookAtBefore, outPrefix, fail]() {
		if (!m_liveObjectEditor->hasSelection()) { fail("moving the camera cleared the selection"); return; }
		*beforeKeys = shown();
		m_livePreviewPage->grab().save(outPrefix + "_objects_selection.png");   // the tab as the window shows it: the box over the picture, the button looking pressed
		*lookAtBefore = m_livePreviewLookAt;
		for (int i = 0; i < 4; ++i) onLivePreviewTranslate(1, 0, 0);   // the W key: with an object selected it moves the object, not the camera
	});
	QTimer::singleShot(11500, this, [this, shown, beforeKeys, lookAtBefore, diff, log, fail]() {
		const double keyChange = diff(*beforeKeys, shown());
		const double cameraMoved = camera_math::length(m_livePreviewLookAt - *lookAtBefore);
		log(QString("W key with the object selected: the picture changed by %1, the camera moved %2").arg(keyChange, 0, 'f', 3).arg(cameraMoved, 0, 'g', 4));
		if (cameraMoved > 1e-9) { fail("a W key press moved the camera although an object was selected"); return; }
		if (!(keyChange > 1.0)) { fail("a W key press did not move the selected object"); return; }
		m_orbit.azimuth -= 0.3;   // back to the camera of the first picture
		updateLivePreviewCameraFromOrbit();
		m_liveObjectEditor->setMode(false);
	});
	// Save arrangement, then Reset: the saved scene is listed, has the object moved, and the picture of the original scene goes back.
	auto arranged = std::make_shared<QImage>();
	auto savedId = std::make_shared<QString>();
	QTimer::singleShot(13500, this, [this, shown, arranged, savedId, log, fail]() {
		*arranged = shown();
		if (auto *button = m_livePreviewPage->findChild<QPushButton *>("liveSaveArrangementButton")) button->click();
		const QString hint = m_liveObjectEditor->hint();
		log("after Save arrangement: \"" + hint + "\"");
		if (!hint.startsWith("Saved as")) { fail("Save arrangement did not save"); return; }
		const QRegularExpression idPattern("scene (\\S+) \\(");
		const auto match = idPattern.match(statusBar()->currentMessage());
		if (!match.hasMatch()) { fail("the saved arrangement was not listed: \"" + statusBar()->currentMessage() + "\""); return; }
		*savedId = match.captured(1);
		log("listed as scene " + *savedId);
		if (auto *button = m_livePreviewPage->findChild<QPushButton *>("liveResetObjectsButton")) button->click();
	});
	QTimer::singleShot(16500, this, [this, shown, before, moved, diff, outPrefix, log, fail]() {
		const QImage reset = shown();
		reset.save(outPrefix + "_objects_reset.png");
		const double back = diff(*before, reset), away = diff(*moved, reset);
		log(QString("after Reset objects: %1 from the first picture, %2 from the moved one").arg(back, 0, 'f', 3).arg(away, 0, 'f', 3));
		if (!(back < away)) { fail("Reset objects did not bring the first picture back"); return; }
	});
	QTimer::singleShot(17000, this, [this, savedId, fail]() {
		stopLivePreview();
		selectSceneById(*savedId);
		startLivePreview();   // the saved scene, drawn from its own file
	});
	QTimer::singleShot(30000, this, [this, shown, before, arranged, diff, outPrefix, log, fail]() {
		const QImage fromSaved = shown();
		log("the saved scene: status \"" + (m_livePreviewStatusLabel ? m_livePreviewStatusLabel->text() : QString()) + "\", picture " + QString::number(fromSaved.width()) + "x" + QString::number(fromSaved.height()));
		fromSaved.save(outPrefix + "_objects_saved_scene.png");
		const double likeArranged = diff(*arranged, fromSaved), likeOriginal = diff(*before, fromSaved);
		log(QString("the saved scene is %1 from the arrangement it was saved from, %2 from the original").arg(likeArranged, 0, 'f', 3).arg(likeOriginal, 0, 'f', 3));
		if (!(likeArranged < likeOriginal)) { fail("the saved scene does not show the arrangement"); return; }
		stopLivePreview();
		log("objects ok");
		QApplication::exit(0);
	});
}

// The Scene Builder <-> Live Preview round trip, end to end: the Builder's starter scene is opened in Live Preview with the "Preview live" button, an object is dragged with real
// mouse events, Save arrangement is pressed, and "Edit in Builder" brings the arrangement back into the Scene Builder: exactly the moved object's position differs from the original
// document, by about the drag. Also checks that clicking "Preview live" again after a change shows the new scene (the running preview restarts on the changed file). Exits 1 on a failure.
void MainWindow::runLivePreviewArrangeSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &, const QString &) {
	const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
	if (idx < 0 || !m_sceneBuilder) { log("Live Preview mode or the Scene Builder is missing"); QApplication::exit(2); return; }
	auto fail = [log](const QString &why) { log("FAIL: " + why); QApplication::exit(1); };
	if (!m_livePreviewSession) { fail("no Live Preview session: the Live Preview library is missing"); return; }
	m_sceneBuilder->newScene();
	const scene_doc::Document original = m_sceneBuilder->document();
	auto *previewLive = m_sceneBuilder->findChild<QPushButton *>("builderPreviewLiveButton");
	if (!previewLive || previewLive->isHidden()) { fail("the Scene Builder has no visible \"Preview live\" button"); return; }
	if (m_liveSmoothNoiseCheck) m_liveSmoothNoiseCheck->setChecked(false);
	previewLive->click();
	const QString id = m_sceneCombo ? m_sceneCombo->currentData().toString() : QString();
	log("previewing the starter scene as " + id + " (" + SceneMetadataClient::sceneName(id) + ")");
	if (!SceneMetadataClient::sceneName(id).endsWith("(live preview)")) { fail("Preview live did not select its own preview copy"); return; }
	if (!m_livePreviewRunning) { fail("Preview live did not start Live Preview"); return; }
	QTimer::singleShot(4500, this, [this, log, fail]() {
		if (!m_livePreviewRunning || !m_livePreviewLabel || !m_liveObjectEditor) { fail("the preview did not start with object editing available"); return; }
		m_liveObjectEditor->setMode(true);
		const QRect image = m_livePreviewLabel->displayedImageRect();
		auto send = [this](QEvent::Type type, const QPoint &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
			QMouseEvent event(type, QPointF(pos), QPointF(m_livePreviewLabel->mapToGlobal(pos)), button, buttons, Qt::NoModifier);
			QApplication::sendEvent(m_livePreviewLabel, &event);
		};
		// Try a few places for an object (the starter scene has a floor, two balls, a box and a light): the first press that grabs one wins.
		const QPoint spots[] = {image.center() + QPoint(-image.width() / 6, image.height() / 6), image.center() + QPoint(image.width() / 5, image.height() / 5),
		                        image.center() + QPoint(0, image.height() / 3), image.center()};
		QPoint start;
		bool grabbed = false;
		for (const QPoint &spot : spots) {
			send(QEvent::MouseButtonPress, spot, Qt::LeftButton, Qt::LeftButton);
			if (m_liveObjectEditor->hint().startsWith("Selected")) { start = spot; grabbed = true; break; }
			send(QEvent::MouseButtonRelease, spot, Qt::LeftButton, Qt::NoButton);
		}
		if (!grabbed) { fail("no press on the starter scene grabbed an object"); return; }
		log("grabbed: \"" + m_liveObjectEditor->hint() + "\"");
		for (int i = 1; i <= 10; ++i) send(QEvent::MouseMove, start + QPoint(image.width() * i / 40, 0), Qt::NoButton, Qt::LeftButton);
		send(QEvent::MouseButtonRelease, start + QPoint(image.width() * 10 / 40, 0), Qt::LeftButton, Qt::NoButton);
	});
	QTimer::singleShot(8000, this, [this, original, log, fail]() {
		if (auto *button = m_livePreviewPage->findChild<QPushButton *>("liveSaveArrangementButton")) button->click();
		const QString hint = m_liveObjectEditor->hint();
		log("after Save arrangement: \"" + hint + "\"");
		if (!hint.startsWith("Saved as")) { fail("Save arrangement did not save"); return; }
		if (!hint.contains("Scene Builder opens it")) { fail("the summary does not say the Scene Builder can open the copy"); return; }
		auto *edit = m_livePreviewPage->findChild<QPushButton *>("liveEditInBuilderButton");
		if (!edit || edit->isHidden()) { fail("no visible \"Edit in Builder\" button after the save"); return; }
		edit->click();
		if (m_tabWidget->currentWidget() != m_sceneBuilder) { fail("Edit in Builder did not bring up the Scene Builder tab"); return; }
		const scene_doc::Document &now = m_sceneBuilder->document();
		if (now.objects.size() != original.objects.size()) { fail("the arrangement has a different number of objects"); return; }
		int moved = 0;
		double shift = 0.0;
		for (std::size_t i = 0; i < now.objects.size(); ++i) {
			const double dx = now.objects[i].position.x - original.objects[i].position.x, dy = now.objects[i].position.y - original.objects[i].position.y,
			             dz = now.objects[i].position.z - original.objects[i].position.z;
			const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
			if (d > 1e-6) { ++moved; shift = d; }
		}
		log(QString("the Scene Builder opened the arrangement: %1 object(s) moved, by %2").arg(moved).arg(shift, 0, 'f', 3));
		if (moved != 1) { fail("expected exactly one moved object in the opened arrangement"); return; }
		if (!(shift > 0.05)) { fail("the object moved by almost nothing"); return; }
		// Change the scene in the Builder (a new name) and preview live again: the running preview restarts on the changed preview copy.
		m_sceneBuilder->setSceneName("Round trip");
		if (auto *previewLive = m_sceneBuilder->findChild<QPushButton *>("builderPreviewLiveButton")) previewLive->click();
		const QString newId = m_sceneCombo->currentData().toString();
		log("previewing again as " + newId + " (" + SceneMetadataClient::sceneName(newId) + ")");
		if (SceneMetadataClient::sceneName(newId) != QString("Round trip (live preview)")) { fail("the second Preview live did not use the changed scene"); return; }
		if (!m_livePreviewRunning) { fail("the second Preview live did not start Live Preview"); return; }
		QTimer::singleShot(2500, this, [this, log, fail]() {
			stopLivePreview();
			// Tidy: the files this test made in the (throwaway) scene-list folder.
			const QString folder = QString::fromStdString(pbrt_asset_check::userSceneDir());
			for (const QString &name : QDir(folder).entryList(QStringList() << "*-live-preview.pbrt" << "*-arranged*.pbrt", QDir::Files)) QFile::remove(folder + "/" + name);
			log("arrange ok");
			QApplication::exit(0);
		});
	});
}

void MainWindow::runLivePreviewDragSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot, const QString &outPrefix) {
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

void MainWindow::runLivePreviewSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot, const QString &outPrefix) {
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
	if (qEnvironmentVariable("RT_GUI_SELFTEST_AIDENOISE") == "1") live_ai_denoise::saveEnabled(true);   // (needs RT_OIDN_DIR or an installed library)
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
		log(QString("after 10 s: frames=%1 (%2 fps), status \"%3\"").arg(frames).arg(secs > 0 ? frames / secs : 0.0, 0, 'f', 1)
		        .arg(m_livePreviewStatusLabel ? m_livePreviewStatusLabel->text() : QString()));
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
