#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include "camera_math.h"
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QDir>
#include <QTimer>
#include <QDateTime>
#include <QDesktopServices>
#include <QUrl>
#include <QScrollBar>
#include <QCoreApplication>
#include <QSignalBlocker>
#include <QIcon>
#include <QStyle>
#include <QThread>
#include <array>
#include <cmath>
#include <optional>

namespace {

// Log-line classification lives in render_output_parser.h, which is Qt-free so
// the rules can be unit tested against real captured renderer output (see
// tests/unit/render_output_parser_tests.cpp). It previously lived here as ~15
// classifier functions in this anonymous namespace, which no test could reach
// - and writing those tests immediately turned up a misclassification that had
// shipped: the launcher's settings echo ("... height=80 spp=4 ...") was being
// labelled a performance measurement.

QString styleLogLine(const render_output::LogCategory &cat, const QString &colour,
					 const QString &timestampColour, const QString &timestamp,
					 const QString &escaped) {
	switch (cat.style) {
	case render_output::LineStyle::Banner:
		return QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
					   "<b>%2</b></span>").arg(colour, escaped);
	case render_output::LineStyle::BoldLabeled:
		return QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
					   "<b><span style='color:%5;'>%2</span> "
					   "<span style='color:%1;'>[%3]</span> %4</b></span>")
			.arg(colour, timestamp, QString::fromLatin1(cat.label), escaped, timestampColour);
	case render_output::LineStyle::Normal:
		break;
	}
	return QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
				   "<span style='color:%5;'>%2</span> "
				   "<span style='color:%1;'>[%3]</span> %4"
				   "</span>")
		.arg(colour, timestamp, QString::fromLatin1(cat.label), escaped, timestampColour);
}

} // namespace

void MainWindow::onRenderClicked() {
	if (m_isRendering) {
		QMessageBox::warning(this, "Render In Progress", "A render is already in progress!");
		return;
	}

	// ========================================================================
	// Collect Render Parameters
	// ========================================================================

	// Render mode: GPU (true) or CPU (false)
	bool useGPU = m_renderModeCombo->currentData().toBool();
	// GPU backend: recursive (false, default) or wavefront (true). Meaningless
	// under CPU, so only honored when useGPU is also true.
	bool useWavefront = useGPU && m_gpuBackendCombo->currentData().toBool();

	// Resolution: either from preset dropdown or custom values from Advanced tab
	int width, height;
	if (m_qualityPresetCombo->currentIndex() == 6) {
		// Custom quality preset - use manual width/height from Advanced tab
		width = m_widthSpinBox->value();
		height = m_heightSpinBox->value();
	} else {
		// Standard quality preset - use resolution from dropdown
		QSize res = m_resolutionCombo->currentData().toSize();
		width = res.width();
		height = res.height();
	}

	// Ray tracing quality parameters
	int samples = m_samplesSpinBox->value();    // Samples per pixel (higher = smoother but slower)
	int maxDepth = m_maxDepthSpinBox->value();  // Max ray bounce depth (higher = more realistic lighting)

	// Output file path (timestamped by default to avoid overwriting)
	QString outputPath = m_outputPathEdit->text();

	// Camera position (lookfrom) - read from spinboxes
	// These reflect either the selected preset or custom user input
	int sceneId = m_sceneCombo->currentData().toInt();
	double camX = m_cameraPosX->value();
	double camY = m_cameraPosY->value();
	double camZ = m_cameraPosZ->value();

	// Only treat the camera as "explicit" (see RenderController::setParameters's
	// comment) if it actually differs from this scene's own recommended
	// camera - queried live, same as onSceneChanged. If the query fails,
	// default to explicit: the worse outcome is an unnecessary (but
	// harmless, since it'd be the same value anyway) cam_x/y/z on the
	// command line, not a silently wrong camera.
	bool camExplicit = true;
	double recCamX = 0.0, recCamY = 0.0, recCamZ = 0.0, recLookatX = 0.0, recLookatY = 0.0, recLookatZ = 0.0;
	if (SceneMetadataClient::recommendedCamera(sceneId, recCamX, recCamY, recCamZ, recLookatX, recLookatY, recLookatZ)) {
		// m_cameraPosX/Y/Z are QDoubleSpinBoxes with the default 2 decimal
		// places, so a recommended value round-trips through setValue()/
		// value() rounded to the nearest 0.01 - the epsilon has to be
		// looser than that (half a step) or a scene whose recommended
		// camera ever needs more precision than 2 decimals would silently
		// never compare equal, permanently forcing camExplicit=true (still
		// harmless, just pointlessly defeats this check for that scene).
		constexpr double kEpsilon = 0.005;
		camExplicit = std::abs(camX - recCamX) > kEpsilon
			|| std::abs(camY - recCamY) > kEpsilon
			|| std::abs(camZ - recCamZ) > kEpsilon;
	}

	// ========================================================================
	// Launch Render
	// ========================================================================
	// RenderController spawns ray_tracer.exe as a subprocess with all
	// parameters and reports its output back via signals (no worker thread -
	// QProcess is already asynchronous). The executable will call either the
	// CPU or GPU renderer based on the useGPU flag.
	m_renderController = new RenderController(this);
	m_renderController->setParameters(useGPU, width, height, samples, maxDepth, sceneId, camX, camY, camZ, camExplicit, outputPath, useWavefront);

	// Set video parameters if in video mode
	if (m_videoMode) {
		int videoFrames = m_videoFramesSpinBox->value();
		int videoFPS = m_videoFPSSpinBox->value();
		double videoSpeed = m_videoSpeedSpinBox->value();
		QString cameraPath = m_cameraPathCombo->currentData().toString();
		m_renderController->setVideoParameters(true, videoFrames, videoFPS, cameraPath, videoSpeed);
	} else {
		m_renderController->setVideoParameters(false, 0, 0, "", 1.0);
	}

	connect(m_renderController, &RenderController::progressUpdate, this, &MainWindow::onProgressUpdate);
	connect(m_renderController, &RenderController::renderComplete, this, &MainWindow::onRenderComplete);
	connect(m_renderController, &RenderController::logMessage, this, &MainWindow::onLogMessage);

	// The controller is done once it reports completion; drop it so
	// m_renderController is only non-null while a render is actually active.
	connect(m_renderController, &RenderController::renderComplete, this, [this]() {
		if (!m_renderController) return;
		m_renderController->deleteLater();
		m_renderController = nullptr;
	});

	m_isRendering = true;
	m_renderButton->setEnabled(false);
	m_stopButton->setEnabled(true);
	updateActionStates();
	refreshStatusBarInfo();
	m_progressBar->setValue(0);
	m_statusLabel->setText(m_videoMode ? "Rendering video frames..." : "Rendering...");

	// Clear any previous render's preview so a stale image doesn't linger
	// confusingly while this one is in progress.
	if (m_previewLabel) m_previewLabel->clearPreviewPixmap();
	if (m_previewInfoLabel) m_previewInfoLabel->clear();
	m_lastOutputPath.clear();
	m_lastPreviewImagePath.clear();

	// Start elapsed timer. Its 1 Hz tick doubles as the ETA sampling clock,
	// which is the same cadence HandBrake samples at.
	m_renderStartTime = QDateTime::currentDateTime();
	resetProgressSamples();
	setProgressResultState("");
	// Indeterminate until the first real progress line: scene loading and
	// BVH/pipeline build report nothing, and a bar pinned at 0% reads as
	// "stuck" rather than "working".
	m_lastTaskbarPercent = -1;
	win_taskbar::setState(this, win_taskbar::State::Indeterminate);
	if (!m_elapsedTimer) {
		m_elapsedTimer = new QTimer(this);
		connect(m_elapsedTimer, &QTimer::timeout, this, &MainWindow::onElapsedTick);
	}
	m_elapsedTimer->start(1000);

	// Auto-switch to Log tab so user sees live output immediately
	if (m_logTabIndex >= 0) m_tabWidget->setCurrentIndex(m_logTabIndex);

	m_renderController->start();
}

void MainWindow::onStopClicked() {
	if (!m_isRendering || !m_renderController) {
		return;
	}

	m_statusLabel->setText("Stopping render...");
	m_stopButton->setEnabled(false);

	m_renderController->stopRender();

	// The controller emits renderComplete once the process actually exits,
	// which resets the UI.
}

void MainWindow::onQualityPresetChanged(int index) {
	// Draft: 25 samples, Preview: 50 samples, Good: 100 samples, High: 500 samples, Ultra: 1000 samples, Maximum: 5000 samples
	const int presetSamples[] = {25, 50, 100, 500, 1000, 5000, m_samplesSpinBox->value()};
	const int presetDepth[] = {10, 20, 50, 50, 100, 100, m_maxDepthSpinBox->value()};

	if (index >= 0 && index < 7) {
		m_samplesSpinBox->setValue(presetSamples[index]);
		m_maxDepthSpinBox->setValue(presetDepth[index]);
	}
}

// ============================================================================
// Camera Preset Change Handler
// ============================================================================
// Called when user selects a different camera preset from the dropdown
// or when initializing the GUI with the default preset
// 
// Behavior:
//   - If "Custom" is selected (index 7): enables X/Y/Z spinboxes for manual input
//   - Otherwise: disables spinboxes but updates them to show the preset's position
//   - The spinbox values are always visible to show where the camera is positioned
// ============================================================================
void MainWindow::onCameraPresetChanged(int index) {
	// Check if "Custom" preset is selected (index 7 = 8th item in the combo box)
	// Custom preset allows user to manually adjust camera position via spinboxes
	bool isCustom = (index == 7); // "Custom" is the 8th item (index 7)

	// Enable spinboxes only for Custom preset; disable for all other presets
	m_cameraPosX->setEnabled(isCustom);
	m_cameraPosY->setEnabled(isCustom);
	m_cameraPosZ->setEnabled(isCustom);
	m_cameraDistance->setEnabled(isCustom);

	// Update spinbox values to reflect the selected preset's camera position.
	// Skip this for Custom: its stored itemData is just a fixed starting
	// point, and overwriting the spinboxes here would silently discard
	// whatever position the user already typed in whenever they switch away
	// from Custom and back. Non-Custom presets always show their own
	// scaled position, so overwriting is correct (and expected) for those.
	// itemData holds a direction*ratio vector, not an absolute position (see
	// the combo's setup comment in mainwindow_tabs.cpp) - scale by the
	// current scene's own recommended-camera distance and offset from its
	// lookat, so e.g. "Right Wall" lands somewhere sensible for whatever
	// scene is active instead of always landing at Cornell Box's own literal
	// (500,278,278).
	if (!isCustom && index >= 0 && index < m_cameraPresetCombo->count()) {
		const QVector3D dir = m_cameraPresetCombo->itemData(index).value<QVector3D>();
		const camera_math::Vec3 pos = camera_math::presetPosition(
			camera_math::Vec3{dir.x(), dir.y(), dir.z()},
			currentLookAt(), m_currentSceneCamDistance);
		m_cameraPosX->setValue(pos.x);
		m_cameraPosY->setValue(pos.y);
		m_cameraPosZ->setValue(pos.z);
	}

	// Keep the Distance display in sync with wherever X/Y/Z just landed
	// (either the preset's fixed position, or whatever Custom was already
	// showing) so it never displays a stale value.
	refreshCameraDistanceDisplay();
}

// ============================================================================
// MainWindow::onCameraDistanceChanged
// ============================================================================
// Called when the user edits the "Distance from Center" spinbox (only
// enabled for the "Custom" camera preset). Repositions the camera along its
// EXISTING viewing direction from the current scene's look-at point
// (m_currentLookatX/Y/Z) to the new distance - i.e. a zoom control that
// preserves viewing angle, rather than resetting to some fixed direction.
// ============================================================================
// Reading the camera position and look-at point out of the widgets was
// repeated verbatim at three call sites; these keep that in one place.
camera_math::Vec3 MainWindow::currentCameraPosition() const {
	return camera_math::Vec3{m_cameraPosX->value(),
							 m_cameraPosY->value(),
							 m_cameraPosZ->value()};
}

camera_math::Vec3 MainWindow::currentLookAt() const {
	return camera_math::Vec3{m_currentLookatX, m_currentLookatY, m_currentLookatZ};
}

void MainWindow::onCameraDistanceChanged(double distance) {
	// Arithmetic (including the camera-sits-on-the-target degenerate case)
	// lives in camera_math.h so it can be unit tested - see
	// tests/unit/camera_math_tests.cpp. This slot is only plumbing.
	const camera_math::Vec3 moved = camera_math::repositionAtDistance(
		currentCameraPosition(), currentLookAt(), distance);

	// setValue() below each fire valueChanged; onSceneChanged is not connected
	// to X/Y/Z, so there is no re-entrant loop, and we deliberately do not
	// write back to m_cameraDistance here.
	m_cameraPosX->setValue(moved.x);
	m_cameraPosY->setValue(moved.y);
	m_cameraPosZ->setValue(moved.z);
}

// ============================================================================
// MainWindow::refreshCameraDistanceDisplay
// ============================================================================
// Recomputes the Distance spinbox's displayed value from the current X/Y/Z
// spinboxes and m_currentLookatX/Y/Z, without re-triggering
// onCameraDistanceChanged (which would otherwise try to reposition X/Y/Z
// right back, a harmless but wasteful no-op loop).
// ============================================================================
void MainWindow::refreshCameraDistanceDisplay() {
	const double dist = camera_math::distanceFromTarget(currentCameraPosition(),
														currentLookAt());
	const QSignalBlocker blocker(m_cameraDistance);
	m_cameraDistance->setValue(dist);
}

void MainWindow::onSceneChanged(int index) {
	int scene_id = (m_sceneCombo && index >= 0) ? m_sceneCombo->itemData(index).toInt() : index;

	// Every field here is queried live from scene_metadata.dll (see
	// scene_metadata_client.h) instead of a locally-duplicated table, so it
	// can't drift from scene_registry.h the way scene_descriptor.h's old
	// copy already had once. An empty description means the DLL couldn't
	// be queried or scene_id wasn't found.
	QString description = SceneMetadataClient::sceneDescription(scene_id);
	if (description.isEmpty()) return;

	// GPU-compatibility defaults to "supported" if the DLL can't be
	// queried (missing, wrong architecture, etc.) - the worse outcome
	// there is an avoidable GPU render failure, not a misleadingly-blocked
	// CPU-only scene.
	bool gpuSupported = true;
	SceneMetadataClient::gpuCompatible(scene_id, gpuSupported);

	int recommendedSpp = SceneMetadataClient::sceneRecommendedSpp(scene_id);

	QString infoText = QString("<b>Description:</b> %1<br>").arg(description);
	infoText += QString("<b>Performance:</b> %1<br>").arg(SceneMetadataClient::scenePerformance(scene_id));
	infoText += QString("<b>Recommended SPP:</b> %1<br>").arg(recommendedSpp);
	infoText += QString("<b>GPU Support:</b> %1<br>").arg(gpuSupported ? "Yes" : "CPU only");
	// These two warnings are the only coloured text in the label, so they take
	// their colours from the theme's log severities rather than fixed hex - a
	// gold-on-cream warning is unreadable on the light schemes.
	if (SceneMetadataClient::sceneRequiresFiles(scene_id))
		infoText += QString("<br><b style='color: %1;'>&#9888; Requires external files</b>")
			.arg(m_activeTheme.logWarning.name());
	if (!gpuSupported)
		infoText += QString("<br><b style='color: %1;'>&#9888; CPU renderer only</b>")
			.arg(m_activeTheme.logError.name());
	m_sceneInfoLabel->setText(infoText);
	if (m_samplesSpinBox->value() == 100 || m_samplesSpinBox->value() == 200 || m_samplesSpinBox->value() == 500)
		m_samplesSpinBox->setValue(recommendedSpp);

	// Auto-switch to CPU when scene doesn't support GPU
	if (!gpuSupported && m_renderModeCombo->currentData().toBool()) {
		m_renderModeCombo->setCurrentIndex(1); // index 1 = CPU
	}

	// Reset the camera position to this scene's own recommended default -
	// every preset in m_cameraPresetCombo is now a direction*ratio relative
	// to m_currentSceneCamDistance rather than an absolute Cornell-Box-scale
	// position (see the combo's setup comment in mainwindow_tabs.cpp), but a
	// stale position from a previously viewed scene could still be wildly
	// wrong-scale for a much smaller scene's actual geometry (e.g. scene 1's
	// spheres sit within roughly +-15 units of the origin) until
	// m_currentSceneCamDistance itself is refreshed below. Switches to
	// "Custom" (index 7) first so onCameraPresetChanged() enables the
	// spinboxes for editing without also overwriting the values we're about
	// to set (Custom is specifically exempted from that overwrite - see its
	// own comment). The user can still freely adjust the camera afterward,
	// same as for every other scene.
	//
	// Also queried live from scene_metadata.dll rather than a duplicated
	// table - if the query fails, leave the camera spinboxes exactly as
	// they were (skip the reset) rather than falling back to a stale or
	// wrong-scale default.
	double cam_x, cam_y, cam_z, lookat_x, lookat_y, lookat_z;
	if (SceneMetadataClient::recommendedCamera(scene_id, cam_x, cam_y, cam_z, lookat_x, lookat_y, lookat_z)) {
		m_cameraPresetCombo->setCurrentIndex(7);
		m_currentLookatX = lookat_x;
		m_currentLookatY = lookat_y;
		m_currentLookatZ = lookat_z;
		// This scene's own default viewing distance - the reference every
		// named preset in m_cameraPresetCombo scales against (see its setup
		// comment). Falls back to the Cornell-scale default (1078) if the
		// recommended camera sits exactly on its own lookat (distance 0),
		// which would otherwise collapse every preset to the lookat point.
		double dx = cam_x - lookat_x, dy = cam_y - lookat_y, dz = cam_z - lookat_z;
		double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		m_currentSceneCamDistance = (dist > 1e-6) ? dist : 1078.0;
		m_cameraPosX->setValue(cam_x);
		m_cameraPosY->setValue(cam_y);
		m_cameraPosZ->setValue(cam_z);
		refreshCameraDistanceDisplay();

		// Scale the spinboxes' arrow-key/scroll step to this scene's scale
		// too - a fixed 10-unit step (Cornell Box's own scale) is unusably
		// coarse for e.g. scene 39's Stanford Armadillo, whose whole ~3-unit
		// model would be crossed in well under one click. 1/100th of the
		// scene's own default viewing distance keeps Cornell Box's step at
		// its original ~10.8, matching prior behavior there.
		const double step = m_currentSceneCamDistance / 100.0;
		m_cameraPosX->setSingleStep(step);
		m_cameraPosY->setSingleStep(step);
		m_cameraPosZ->setSingleStep(step);
		m_cameraDistance->setSingleStep(step);
	}
}

void MainWindow::onProgressUpdate(int percentage) {
	m_progressBar->setValue(percentage);

	// Mirror onto the taskbar button so progress is readable while the window
	// is behind something else. Only pushed when the integer percent actually
	// changes - the COM call is not free, and progress lines arrive far more
	// often than once per percent.
	if (percentage != m_lastTaskbarPercent) {
		m_lastTaskbarPercent = percentage;
		win_taskbar::setProgress(this, percentage / 100.0);
	}

	// Let onElapsedTick handle status label text during rendering;
	// just keep the progress bar updated here.
	if (!m_isRendering)
		m_statusLabel->setText(QString("Rendering... %1%").arg(percentage));
}

void MainWindow::onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath) {
	m_isRendering = false;
	m_renderButton->setEnabled(true);
	m_stopButton->setEnabled(false);
	if (m_elapsedTimer) m_elapsedTimer->stop();
	updateActionStates();

	// A failed render leaves the taskbar button red so the outcome is visible
	// without switching to the window; anything else clears it. Leaving a
	// progress state set would make it stick until the process exits.
	notifyRenderFinished(success, message, totalTime);

	if (success) {
		m_progressBar->setValue(100);
		// A finished bar keeps its fill and turns green rather than resetting -
		// the outcome stays visible after the fact (Qt Creator's behaviour).
		setProgressResultState("success");
		m_statusLabel->setText(QString("✅ %1 - Total time: %2 seconds").arg(message).arg(totalTime, 0, 'f', 2));

		// If video mode, automatically assemble the video
		if (m_videoMode) {
			onLogMessage("Video frames rendered successfully. Starting video assembly...");
			m_statusLabel->setText("⚙️ Assembling video from frames...");

			// Trigger automatic video assembly
			QTimer::singleShot(500, this, &MainWindow::assembleVideoAutomatically);
		} else {
			// Image mode: show the rendered image inline on the Preview tab
			// instead of shelling out to the OS's default image viewer.
			// main.cpp's Format Conversion step always writes a same-
			// basename .png next to a successful render's .ppm output -
			// load that (smaller, simpler than parsing PPM by hand).
			if (!outputPath.isEmpty()) {
				QFileInfo fileInfo(outputPath);
				QString pngPath = fileInfo.absolutePath() + "/" + fileInfo.completeBaseName() + ".png";
				QFileInfo pngInfo(pngPath);

				m_lastOutputPath = outputPath;

				if (pngInfo.exists()) {
					QPixmap pixmap(pngPath);
					if (!pixmap.isNull() && m_previewLabel) {
						m_lastPreviewImagePath = pngPath;
						m_previewLabel->setPreviewPixmap(pixmap);
						if (m_previewInfoLabel) {
							m_previewInfoLabel->setText(QString("%1  •  %2×%3  •  %4 KB  •  %5s")
								.arg(pngInfo.fileName())
								.arg(pixmap.width()).arg(pixmap.height())
								.arg(pngInfo.size() / 1024)
								.arg(totalTime, 0, 'f', 2));
						}
						if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
						// Open Folder / Open Viewer only become meaningful once
						// there is actually something to open.
						updateActionStates();
					} else {
						m_statusLabel->setText(QString("✅ Render complete (%1s) - Warning: preview image failed to load at %2")
							.arg(totalTime, 0, 'f', 2).arg(pngPath));
					}
				} else if (fileInfo.exists()) {
					// PNG conversion failed but the raw PPM output exists -
					// fall back to the old external-viewer behavior rather
					// than showing nothing.
					QDesktopServices::openUrl(QUrl::fromLocalFile(outputPath));
				} else {
					m_statusLabel->setText(QString("✅ Render complete (%1s) - Warning: output file not found at %2")
						.arg(totalTime, 0, 'f', 2).arg(outputPath));
				}
			}
		}
	} else {
		const bool stoppedByUser = message.contains("stopped by user", Qt::CaseInsensitive);

		// A user-requested stop isn't a failure, so it clears back to neutral;
		// a genuine failure leaves the bar where it died and turns it red, so
		// the outcome is still readable after the dialog is dismissed.
		if (stoppedByUser) {
			m_progressBar->setValue(0);
			setProgressResultState("");
		} else {
			setProgressResultState("error");
		}
		m_statusLabel->setText(QString("❌ %1").arg(message));

		// Only show error popup for actual failures, not for user-stopped renders
		if (!stoppedByUser) {
			QMessageBox::critical(this, "Render Failed", message);
		}
	}
}

void MainWindow::onLogMessage(const QString &message) {
	if (!m_logTextEdit) return;

	QString msg = message.trimmed();
	if (msg.isEmpty()) return;

	// Timestamp prefix (HH:mm:ss)
	QString ts = QTime::currentTime().toString("HH:mm:ss");

	// HTML-escape so < > & don't break the rich-text display
	QString escaped = msg.toHtmlEscaped();

	// Classification is the tested, Qt-free implementation; this function is
	// left with nothing but presentation.
	const render_output::LogCategory category =
		render_output::classifyLogLine(msg.toStdString());

	m_logHistory.push_back({ts, escaped, category});
	m_logTextEdit->append(styleLogLine(category,
									   m_activeTheme.colourFor(category.severity).name(),
									   m_activeTheme.logSeparator.name(),
									   ts, escaped));

	qDebug() << msg;
}

// Re-renders every line the log has shown, in the current scheme. Called on a
// theme change: the pane's contents are HTML with the previous scheme's
// colours already written into each span, so they can only be replaced, not
// recoloured.
void MainWindow::rebuildLogPane() {
	if (!m_logTextEdit || m_logHistory.isEmpty()) return;

	// Re-appending line by line rather than assembling one HTML document keeps
	// this on exactly the same code path as normal logging, so a rebuilt pane
	// cannot drift in appearance from a freshly written one. Updates are held
	// off because otherwise every append repaints and reflows the whole
	// document.
	const bool scrolledToBottom =
		m_logTextEdit->verticalScrollBar()->value() ==
		m_logTextEdit->verticalScrollBar()->maximum();

	m_logTextEdit->setUpdatesEnabled(false);
	m_logTextEdit->clear();
	for (const LoggedLine &line : m_logHistory) {
		m_logTextEdit->append(styleLogLine(line.category,
										   m_activeTheme.colourFor(line.category.severity).name(),
										   m_activeTheme.logSeparator.name(),
										   line.timestamp, line.escaped));
	}
	m_logTextEdit->setUpdatesEnabled(true);

	// append() leaves the cursor at the end, which scrolls the view there.
	// Restore the top for a user who had scrolled up to read something - a
	// theme change should not move them.
	if (!scrolledToBottom)
		m_logTextEdit->moveCursor(QTextCursor::Start);
}

namespace {

// H:MM:SS once past an hour, else M:SS. Always at least M:SS so the field
// width stays stable and the status line doesn't jitter as digits change -
// the same reason HandBrake's status string uses fixed-width specifiers.
QString formatDuration(qint64 seconds) {
	if (seconds < 0) return QStringLiteral("--:--");
	const qint64 h = seconds / 3600;
	const qint64 m = (seconds % 3600) / 60;
	const qint64 s = seconds % 60;
	if (h > 0)
		return QString("%1:%2:%3").arg(h).arg(m, 2, 10, QChar('0')).arg(s, 2, 10, QChar('0'));
	return QString("%1:%2").arg(m).arg(s, 2, 10, QChar('0'));
}

} // namespace

void MainWindow::resetProgressSamples() {
	m_progressRingCount = 0;
	for (ProgressSample &sample : m_progressRing)
		sample = ProgressSample{};
}

QString MainWindow::formatProgressStatus(qint64 elapsedMs, int percent) {
	// Append this tick's sample, shifting the ring when it's full.
	if (m_progressRingCount < kProgressSamples) {
		m_progressRing[m_progressRingCount++] = ProgressSample{elapsedMs, percent};
	} else {
		for (int i = 0; i < kProgressSamples - 1; ++i)
			m_progressRing[i] = m_progressRing[i + 1];
		m_progressRing[kProgressSamples - 1] = ProgressSample{elapsedMs, percent};
	}

	QString text = QString("Rendering  ·  %1%  ·  elapsed %2")
		.arg(percent, 3)
		.arg(formatDuration(elapsedMs / 1000));

	// Instantaneous rate across the ring - shown, never divided by.
	const ProgressSample &oldest = m_progressRing[0];
	const qint64 windowMs = elapsedMs - oldest.elapsedMs;
	if (m_progressRingCount == kProgressSamples && windowMs > 0) {
		const double pctPerSec = 1000.0 * (percent - oldest.percent) / windowMs;
		if (pctPerSec > 0.0)
			text += QString("  ·  %1 %/s").arg(pctPerSec, 0, 'f', 1);
	}

	// ETA from the cumulative rate, suppressed during warm-up.
	if (elapsedMs >= kEtaWarmupMs && percent > 0 && percent < 100) {
		const double pctPerMs = double(percent) / double(elapsedMs);
		if (pctPerMs > 0.0) {
			const qint64 remainingMs = qint64((100.0 - percent) / pctPerMs);
			text += QString("  ·  ETA %1").arg(formatDuration(remainingMs / 1000));
		}
	} else if (percent < 100) {
		text += QStringLiteral("  ·  ETA --:--");
	}

	return text;
}

void MainWindow::notifyRenderFinished(bool success, const QString &message, double totalTime) {
	const bool stoppedByUser = message.contains("stopped by user", Qt::CaseInsensitive);

	if (success || stoppedByUser) {
		win_taskbar::setState(this, win_taskbar::State::NoProgress);
	} else {
		win_taskbar::setState(this, win_taskbar::State::Error);
	}

	// Only pull attention when the user is plausibly elsewhere. OBS gates its
	// completion toast on the window not being visible for the same reason -
	// notifying someone who is already watching the progress bar is pure
	// noise. QApplication::alert is a no-op when the window is active, so it
	// is safe to call unconditionally (Qt Creator does exactly this after a
	// build).
	QApplication::alert(this, 3000);

	if (isActiveWindow()) return;

	// Log why a notification was or wasn't raised - a silently-swallowed
	// showMessage() (which is what an invisible tray icon does) is otherwise
	// indistinguishable from the feature simply not being wired up.
	if (!m_trayIcon) {
		onLogMessage("[DEBUG] No system tray available; skipping completion notification");
		return;
	}
	if (!QSystemTrayIcon::supportsMessages()) {
		onLogMessage("[DEBUG] System tray does not support messages; skipping notification");
		return;
	}

	const QString title = success ? "Render complete"
								  : (stoppedByUser ? "Render stopped" : "Render failed");
	const QString body = success
		? QString("Finished in %1 seconds").arg(totalTime, 0, 'f', 2)
		: message.section('<', 0, 0).left(120);
	m_trayIcon->showMessage(title, body,
		success ? QSystemTrayIcon::Information : QSystemTrayIcon::Warning, 10000);
}


void MainWindow::setProgressResultState(const char *state) {
	if (!m_progressBar) return;
	// Dynamic property + repolish is the standard way to switch a QSS rule
	// at runtime; the stylesheet carries matching
	// QProgressBar[resultState="..."]::chunk selectors.
	m_progressBar->setProperty("resultState", state);
	m_progressBar->style()->unpolish(m_progressBar);
	m_progressBar->style()->polish(m_progressBar);
	m_progressBar->update();
}

void MainWindow::onElapsedTick() {
	if (!m_isRendering) return;
	const qint64 elapsedMs = m_renderStartTime.msecsTo(QDateTime::currentDateTime());
	m_statusLabel->setText(formatProgressStatus(elapsedMs, m_progressBar->value()));
}

void MainWindow::onModeChanged(int index) {
	m_videoMode = (index == 1); // 0 = Image, 1 = Video

	// Update render button text based on mode
	if (m_videoMode) {
		// Keep the same Alt+R mnemonic as the single-image label below, so the
		// keyboard shortcut doesn't move when the output mode changes. The
		// leading glyph is gone from both labels: the button carries a real
		// QIcon now, and a text-embedded emoji would sit next to it as a
		// second, differently-styled icon.
		m_renderButton->setText("START VIDEO &RENDER");
		icon_tint::apply(m_renderButton, ":/icons/video.svg",
		                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
		m_statusLabel->setText("Ready to render video frames");
	} else {
		m_renderButton->setText("START &RENDER");
		icon_tint::apply(m_renderButton, ":/icons/render.svg",
		                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
		m_statusLabel->setText("Ready to render");
	}

	// Log mode change
	onLogMessage(QString("Mode changed to: %1").arg(m_videoMode ? "Video Generation" : "Single Image"));
}

void MainWindow::assembleVideoAutomatically() {
	// ray_tracer.exe assembles the video itself (via ffmpeg) before it exits.
	// This just finds and opens the resulting file. In the normal case
	// ray_tracer.exe already exits non-zero if ffmpeg failed - see
	// onRenderComplete()'s failure branch - so this is mainly a defensive
	// fallback for the case where the process exited 0 but the expected
	// video filename wasn't where we expect it.

	// Wait a moment for file to be fully written
	QThread::msleep(500);

	// Search for any *_video.mp4 file in the output directory
	QString outputDir = QCoreApplication::applicationDirPath() + "/output";
	QDir dir(outputDir);
	QStringList filters;
	filters << "*_video.mp4" << "video.mp4";
	QFileInfoList videoFiles = dir.entryInfoList(filters, QDir::Files, QDir::Time);

	QString videoPath;
	QFileInfo videoInfo;

	// Get the most recently modified video file
	if (!videoFiles.isEmpty()) {
		videoInfo = videoFiles.first();
		videoPath = videoInfo.absoluteFilePath();
	}

	if (videoPath.isEmpty()) {
		m_statusLabel->setText("⚠️ Video file not found, checking for frames...");
		onLogMessage("WARNING: Video file not found at any of the expected locations");

		// Check if frames exist (fallback diagnostic)
		QString framesDir = QCoreApplication::applicationDirPath() + "/output/frames";
		QDir framesDirObj(framesDir);

		if (framesDirObj.exists()) {
			QStringList frames = framesDirObj.entryList(QStringList() << "frame_*.ppm", QDir::Files);
			if (!frames.isEmpty()) {
				m_statusLabel->setText(QString("⚠️ Found %1 frames but no video file").arg(frames.count()));
				onLogMessage(QString("Frames were rendered (%1 files) but video assembly may have failed.").arg(frames.count()));
				QMessageBox::warning(this, "Video Not Created",
					QString("Frames were rendered successfully (%1 files), but the video file was not created.\n\n"
							"Expected video in: %2\n"
							"with pattern: *_video.mp4 or video.mp4\n\n"
							"Please check the render log for ffmpeg errors.").arg(frames.count()).arg(outputDir));
			} else {
				m_statusLabel->setText("❌ No frames or video found");
				onLogMessage("ERROR: No frames or video file found");
				QMessageBox::critical(this, "Render Failed", 
					"Neither frames nor video file were created.\n\nPlease check the render log for errors.");
			}
		} else {
			m_statusLabel->setText("❌ Frames directory not found");
			onLogMessage(QString("ERROR: Frames directory not found: %1").arg(framesDir));
			QMessageBox::critical(this, "Directory Not Found", 
				QString("Frames directory not found:\n%1\n\nThe render may have failed to create output.").arg(framesDir));
		}
		return;
	}

	// Video file found! Open it
	m_statusLabel->setText("✅ Video created successfully!");
	onLogMessage(QString("✅ Video assembled successfully: %1").arg(videoPath));
	onLogMessage(QString("Video size: %1 MB").arg(videoInfo.size() / (1024.0 * 1024.0), 0, 'f', 2));

	// Auto-open the video
	onLogMessage(QString("Opening video: %1").arg(videoPath));
	QDesktopServices::openUrl(QUrl::fromLocalFile(videoPath));
}
