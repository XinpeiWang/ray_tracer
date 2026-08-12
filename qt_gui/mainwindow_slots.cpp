#include "mainwindow.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
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

// ============================================================================
// Log line classification
// ============================================================================
// onLogMessage() below used to be one long if/else chain matching keywords
// against each incoming line - it had already needed a dedicated bug-fix
// pass once (mis-categorized lines stealing each other's rules) and kept
// growing harder to safely extend. This splits it into small, independently
// readable classifier functions tried in order (first match wins), same
// "line parser chain" shape Qt Creator's Core::OutputWindow uses for its
// own process-output categorization. Adding a new category is now "add one
// function + one array entry" instead of finding the right spot to splice
// an else-if into a 150-line function.
// ============================================================================

enum class LogLineStyle {
	Normal,       // "HH:mm:ss [LABEL] message" - most lines
	BoldLabeled,  // same as Normal but bold (render-start banners)
	Banner,       // bold, no timestamp/label (separators, error banner)
};

struct LogCategory {
	QString colour;
	QString label;  // ignored for Banner style
	LogLineStyle style = LogLineStyle::Normal;
};

using LogClassifierFn = std::optional<LogCategory> (*)(const QString&);

// Would otherwise match classifySeparator's "===" rule below and render as a
// plain grey line, burying the one banner meant to flag a render failure.
std::optional<LogCategory> classifyErrorDetailsBanner(const QString& msg) {
	if (msg.contains("ERROR DETAILS"))
		return LogCategory{"#FF6B6B", QString(), LogLineStyle::Banner};
	return std::nullopt;
}

std::optional<LogCategory> classifySeparator(const QString& msg) {
	if (msg.startsWith("═") || msg.startsWith("─") ||
		msg.startsWith("===") || msg.startsWith("---"))
		return LogCategory{"#555555", QString(), LogLineStyle::Banner};
	return std::nullopt;
}

// Matches on the ASCII words, not the leading glyph. ray_tracer.exe's console
// output decorates lines with box-drawing/emoji characters, and matching those
// literally turned them into an undocumented wire protocol between its stdout
// and this classifier: restyling either side (or the mojibake class of bug
// already documented in RenderController::onReadyRead) would silently drop
// lines back to the plain default category. The words are the contract; the
// glyphs are decoration.
std::optional<LogCategory> classifyRenderStart(const QString& msg) {
	if (msg.contains("RENDER START") || msg.startsWith("Starting render"))
		return LogCategory{"#74C0FC", "INFO", LogLineStyle::BoldLabeled};
	return std::nullopt;
}

std::optional<LogCategory> classifyErrorOrFatal(const QString& msg) {
	if (!(msg.contains("error", Qt::CaseInsensitive) ||
		  msg.contains("FAILED",  Qt::CaseSensitive)  ||
		  msg.contains("fatal",   Qt::CaseInsensitive) ||
		  msg.contains("ERR_",    Qt::CaseSensitive)   ||
		  msg.startsWith("Result: FAILED")))
		return std::nullopt;

	// Keep the red "this is an error" signal, but preserve which subsystem
	// it came from instead of collapsing every GPU/CPU line that happens to
	// contain "error"/"failed" into the generic ERR tag.
	QString label;
	if (msg.contains("[OptiX]",     Qt::CaseSensitive)   ||
		msg.contains("[optix]",     Qt::CaseSensitive)   ||
		msg.contains("OptiX",       Qt::CaseSensitive)   ||
		msg.contains("GPU mode",    Qt::CaseInsensitive)) {
		label = "GPU ";
	} else if (msg.contains("[cpu_interface]", Qt::CaseSensitive) ||
			   msg.contains("CPU mode",        Qt::CaseInsensitive)) {
		label = "CPU ";
	} else {
		label = "ERR ";
	}
	return LogCategory{"#FF6B6B", label, LogLineStyle::Normal};
}

std::optional<LogCategory> classifyWarning(const QString& msg) {
	if (msg.contains("warning", Qt::CaseInsensitive) ||
		msg.contains("WARN",    Qt::CaseSensitive)    ||
		msg.contains("Requires external files", Qt::CaseInsensitive))
		return LogCategory{"#FFD700", "WARN"};
	return std::nullopt;
}

// The check-mark prefixes are kept as an extra hint (ray_tracer.exe emits
// them), but every case they cover is also matched by an ASCII phrase below,
// so success lines stay correctly categorised even if the glyphs change - see
// classifyRenderStart()'s comment.
std::optional<LogCategory> classifySuccess(const QString& msg) {
	if (msg.startsWith("Result: SUCCESS") ||
		msg.startsWith("✅") ||
		msg.startsWith("✓")  ||
		msg.contains("[OK]",                Qt::CaseSensitive)   ||
		msg.contains("Render completed",    Qt::CaseInsensitive) ||
		msg.contains("render complete",     Qt::CaseInsensitive) ||
		msg.contains("rendered successfully", Qt::CaseInsensitive) ||
		msg.contains("saved successfully",  Qt::CaseInsensitive) ||
		msg.contains("PNG saved",           Qt::CaseInsensitive))
		return LogCategory{"#51CF66", " OK "};
	return std::nullopt;
}

std::optional<LogCategory> classifyGpuOptix(const QString& msg) {
	if (msg.contains("[OptiX]",         Qt::CaseSensitive) ||
		msg.contains("[optix]",          Qt::CaseSensitive) ||
		msg.contains("OptiX",            Qt::CaseSensitive) ||
		msg.contains("optix_render",     Qt::CaseInsensitive) ||
		msg.contains("GPU mode",         Qt::CaseInsensitive) ||
		msg.contains("NVIDIA",           Qt::CaseSensitive) ||
		msg.contains("RTX",              Qt::CaseSensitive) ||
		msg.contains("Launching renderer (GPU", Qt::CaseInsensitive))
		return LogCategory{"#A9E34B", "GPU "};
	return std::nullopt;
}

std::optional<LogCategory> classifyCpuRenderer(const QString& msg) {
	if (msg.contains("[cpu_interface]",  Qt::CaseSensitive) ||
		msg.contains("CPU mode",         Qt::CaseInsensitive) ||
		msg.contains("Launching renderer (CPU", Qt::CaseInsensitive))
		return LogCategory{"#74C0FC", "CPU "};
	return std::nullopt;
}

std::optional<LogCategory> classifyPerfTiming(const QString& msg) {
	if (msg.contains("RENDER TIME",  Qt::CaseSensitive) ||
		msg.contains("time=",          Qt::CaseInsensitive) ||
		msg.contains(" ms",           Qt::CaseSensitive)   ||
		msg.contains(" spp",          Qt::CaseInsensitive) ||
		msg.contains("Rendered ",     Qt::CaseSensitive)   ||
		msg.contains("Pipeline stat", Qt::CaseInsensitive))
		return LogCategory{"#FFA94D", "PERF"};
	return std::nullopt;
}

std::optional<LogCategory> classifySceneInfo(const QString& msg) {
	if (msg.contains("Building scene",   Qt::CaseInsensitive) ||
		msg.contains("Uploaded ",         Qt::CaseInsensitive) ||
		msg.contains("Built ",            Qt::CaseSensitive)   ||
		msg.contains("materials to GPU",  Qt::CaseInsensitive) ||
		msg.contains("spheres to GPU",    Qt::CaseInsensitive) ||
		msg.contains("quads to GPU",      Qt::CaseInsensitive) ||
		msg.contains("light sources",     Qt::CaseInsensitive) ||
		msg.contains("acceleration struct", Qt::CaseInsensitive))
		return LogCategory{"#CC99FF", "SCN "};
	return std::nullopt;
}

std::optional<LogCategory> classifyPipelineInit(const QString& msg) {
	if (msg.contains("Pipeline",         Qt::CaseSensitive) ||
		msg.contains("Initializ",        Qt::CaseInsensitive) ||
		msg.contains("Loaded PTX",       Qt::CaseInsensitive) ||
		msg.contains("Created program",  Qt::CaseInsensitive) ||
		msg.contains("Using GPU:",       Qt::CaseSensitive))
		return LogCategory{"#63E6BE", "INIT"};
	return std::nullopt;
}

std::optional<LogCategory> classifyTechSummary(const QString& msg) {
	if (msg.startsWith("[TECH]"))
		return LogCategory{"#E599F7", "TECH"};
	return std::nullopt;
}

std::optional<LogCategory> classifyCommandLine(const QString& msg) {
	if (msg.startsWith("Command:"))
		return LogCategory{"#CCCCCC", "CMD "};
	return std::nullopt;
}

std::optional<LogCategory> classifyDebugSettings(const QString& msg) {
	if (msg.contains("[DEBUG]",           Qt::CaseSensitive) ||
		msg.contains("Parsed ",           Qt::CaseSensitive)  ||
		msg.contains("Using command-line",Qt::CaseInsensitive)||
		msg.contains("Writing output to", Qt::CaseInsensitive))
		return LogCategory{"#A8A8A8", "DBG "};
	return std::nullopt;
}

std::optional<LogCategory> classifyProcessResult(const QString& msg) {
	if (msg.startsWith("Process finished") ||
		msg.startsWith("Result:"))
		return LogCategory{"#D0D0D0", "INFO"};
	return std::nullopt;
}

// Tried in order, first match wins - matches the original if/else chain's
// precedence exactly (e.g. classifyErrorOrFatal before classifyGpuOptix, so
// an OptiX line containing "error" still gets flagged red, not green).
constexpr std::array<LogClassifierFn, 15> kLogClassifiers = {
	classifyErrorDetailsBanner,
	classifySeparator,
	classifyRenderStart,
	classifyErrorOrFatal,
	classifyWarning,
	classifySuccess,
	classifyGpuOptix,
	classifyCpuRenderer,
	classifyPerfTiming,
	classifySceneInfo,
	classifyPipelineInit,
	classifyTechSummary,
	classifyCommandLine,
	classifyDebugSettings,
	classifyProcessResult,
};

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
	m_renderController->setParameters(useGPU, width, height, samples, maxDepth, sceneId, camX, camY, camZ, camExplicit, outputPath);

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
		QVector3D dir = m_cameraPresetCombo->itemData(index).value<QVector3D>();
		m_cameraPosX->setValue(m_currentLookatX + dir.x() * m_currentSceneCamDistance);
		m_cameraPosY->setValue(m_currentLookatY + dir.y() * m_currentSceneCamDistance);
		m_cameraPosZ->setValue(m_currentLookatZ + dir.z() * m_currentSceneCamDistance);
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
void MainWindow::onCameraDistanceChanged(double distance) {
	double dx = m_cameraPosX->value() - m_currentLookatX;
	double dy = m_cameraPosY->value() - m_currentLookatY;
	double dz = m_cameraPosZ->value() - m_currentLookatZ;
	double currentDist = std::sqrt(dx * dx + dy * dy + dz * dz);

	// If the camera is sitting exactly on the look-at point, there's no
	// direction to preserve - fall back to looking down -Z, matching the
	// launcher's own generic default direction.
	if (currentDist < 1e-6) {
		dx = 0.0;
		dy = 0.0;
		dz = -1.0;
		currentDist = 1.0;
	}

	const double scale = distance / currentDist;

	// setValue() below will each fire valueChanged -> onSceneChanged is not
	// connected to X/Y/Z directly, so no re-entrant loop here; only guard
	// against this spinbox re-triggering itself is unnecessary since we
	// don't write back to m_cameraDistance in this slot.
	m_cameraPosX->setValue(m_currentLookatX + dx * scale);
	m_cameraPosY->setValue(m_currentLookatY + dy * scale);
	m_cameraPosZ->setValue(m_currentLookatZ + dz * scale);
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
	double dx = m_cameraPosX->value() - m_currentLookatX;
	double dy = m_cameraPosY->value() - m_currentLookatY;
	double dz = m_cameraPosZ->value() - m_currentLookatZ;
	double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

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
	if (SceneMetadataClient::sceneRequiresFiles(scene_id))
		infoText += "<br><b style='color: #FFD700;'>&#9888; Requires external files</b>";
	if (!gpuSupported)
		infoText += "<br><b style='color: #FF6B6B;'>&#9888; CPU renderer only</b>";
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

	// First matching classifier wins; falls back to plain INFO if none
	// match, same as the original if/else chain's final "else" branch.
	LogCategory category{"#D0D0D0", "INFO", LogLineStyle::Normal};
	for (LogClassifierFn classify : kLogClassifiers) {
		if (std::optional<LogCategory> match = classify(msg)) {
			category = *match;
			break;
		}
	}

	QString html;
	switch (category.style) {
	case LogLineStyle::Banner:
		html = QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
						"<b>%2</b></span>").arg(category.colour, escaped);
		break;
	case LogLineStyle::BoldLabeled:
		html = QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
						"<b><span style='color:#888888;'>%2</span> "
						"<span style='color:%1;'>[%3]</span> %4</b></span>")
			.arg(category.colour, ts, category.label, escaped);
		break;
	case LogLineStyle::Normal:
		html = QString("<span style='color:%1;font-family:Consolas,monospace;font-size:9pt;'>"
						"<span style='color:#555555;'>%2</span> "
						"<span style='color:%1;'>[%3]</span> %4"
						"</span>")
			.arg(category.colour, ts, category.label, escaped);
		break;
	}
	m_logTextEdit->append(html);

	qDebug() << msg;
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
		m_renderButton->setIcon(QIcon(":/icons/video_accent.svg"));
		m_statusLabel->setText("Ready to render video frames");
	} else {
		m_renderButton->setText("START &RENDER");
		m_renderButton->setIcon(QIcon(":/icons/render_accent.svg"));
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
