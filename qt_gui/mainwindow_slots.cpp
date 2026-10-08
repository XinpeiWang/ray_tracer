#include "mainwindow.h"
#include "icon_tint.h"
#include "photo_import.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include "app_log.h"
#include "camera_math.h"
#include "../src/shared/video_preset.h"
#include "../src/shared/scene_descriptor.h"
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QLocale>
#include <QProgressDialog>
#include <QStorageInfo>
#include <QProcess>
#include <QDir>
#include <QTimer>
#include <QDateTime>
#include <QDesktopServices>
#include <QUrl>
#include <QScrollBar>
#include <QStatusBar>
#include <QCoreApplication>
#include <QSignalBlocker>
#include <QIcon>
#include <QStyle>
#include <QThread>
#include <QHash>
#include <array>
#include <cmath>
#include <optional>


void MainWindow::onStopClicked() {
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	// Checked first (though §3's mutual-exclusion rule means m_isRendering
	// and m_livePreviewRunning are never both true, so the order isn't
	// actually load-bearing) - RenderController::stopRender() exists to
	// kill a QProcess, which doesn't apply to an in-process live session.
	if (m_livePreviewRunning) {
		stopLivePreview();
		return;
	}
#endif
	if (!m_isRendering || !m_renderController) {
		return;
	}

	m_statusLabel->setText(tr("Stopping render..."));
	m_stopButton->setEnabled(false);
	m_pauseButton->setEnabled(false);
	m_abandonButton->setEnabled(false);

	m_renderController->stopRender();

	// The controller emits renderComplete once the process actually exits,
	// which resets the UI.
}

void MainWindow::onPauseClicked() {
	if (!m_isRendering || !m_renderController) return;

	// A single button toggles both directions - isPaused() (not a locally
	// tracked bool) is the source of truth, so this can never drift out of
	// sync with what the process is actually doing.
	if (m_renderController->isPaused()) {
		m_renderController->resumeRender();
	} else {
		m_renderController->pauseRender();
	}
	// onControllerPauseStateChanged() (connected in startRenderJob()) flips
	// the button's own label/icon and the status text once the controller
	// confirms the change.
}

void MainWindow::onAbandonClicked() {
	if (!m_isRendering || !m_renderController) return;

	m_statusLabel->setText(tr("Abandoning render..."));
	m_stopButton->setEnabled(false);
	m_pauseButton->setEnabled(false);
	m_abandonButton->setEnabled(false);

	m_renderController->abandonRender();

	// onRenderComplete() picks this up once the process actually exits -
	// unlike a plain Stop, it advances straight to the next queued job
	// (see that method's own stoppedByUser/abandonedByUser comment) instead
	// of leaving the queue waiting.
}

void MainWindow::onControllerPauseStateChanged(bool paused) {
	if (!m_pauseButton) return;

	if (paused) {
		m_pauseStartedAt = QDateTime::currentDateTime();
		if (m_elapsedTimer) m_elapsedTimer->stop();
		m_pauseButton->setText(tr("&RESUME RENDER"));
		icon_tint::apply(m_pauseButton, ":/icons/render.svg", icon_tint::Role::Body, m_activeTheme.textBody);
		m_pauseButton->setToolTip(tr("Resume the paused render from the exact same pixels"));
		if (m_actPause) {
			m_actPause->setText(tr("&Resume Render"));
			icon_tint::apply(m_actPause, ":/icons/render.svg", icon_tint::Role::Body, m_activeTheme.textBody);
		}
		m_statusLabel->setText(tr("⏸ Paused"));
	} else {
		// Shifts the elapsed-time origin forward by however long the pause
		// lasted, so onElapsedTick()'s plain wall-clock formula
		// (m_renderStartTime to now) keeps reading correctly without needing
		// its own separate paused-time bookkeeping - the same trick
		// RenderController uses for the totalTime it reports on completion.
		if (m_pauseStartedAt.isValid()) {
			m_renderStartTime = m_renderStartTime.addMSecs(m_pauseStartedAt.msecsTo(QDateTime::currentDateTime()));
			m_pauseStartedAt = QDateTime();
		}
		if (m_elapsedTimer) m_elapsedTimer->start(1000);
		m_pauseButton->setText(tr("&PAUSE RENDER"));
		icon_tint::apply(m_pauseButton, ":/icons/pause.svg", icon_tint::Role::Body, m_activeTheme.textBody);
		m_pauseButton->setToolTip(tr("Pause the running render in place - Resume continues from the exact same pixels"));
		if (m_actPause) {
			m_actPause->setText(tr("&Pause Render"));
			icon_tint::apply(m_actPause, ":/icons/pause.svg", icon_tint::Role::Body, m_activeTheme.textBody);
		}
	}
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
// MainWindow::onVideoPresetChanged
// ============================================================================
// Called when the user picks a named bundle from the Video Generation
// Settings group's Preset combo (see video_preset.h and createSettingsTab()'s
// own setup comment).
// Index 0 is the always-present "(custom)" placeholder with empty itemData -
// selecting it is a no-op, since its whole point is "I'm choosing the four
// controls below myself" rather than pointing at anything to apply.
// ============================================================================
void MainWindow::onVideoPresetChanged(int index) {
	if (index <= 0 || !m_videoPresetCombo) return;
	const QString id = m_videoPresetCombo->itemData(index).toString();
	const video_preset::VideoPreset* preset = video_preset::find(id.toUtf8().constData());
	if (!preset) return;

	// Switch to Generate Video first: onModeChanged()'s own side effects
	// (render button text/icon, status label) should already be in place
	// before selectSceneById() below runs onSceneChanged(), which also
	// touches status-adjacent labels.
	selectOutputMode(OutputMode::Video);

	selectSceneById(QString::fromUtf8(preset->scene_id));

	const int pathIndex = m_cameraPathCombo ? m_cameraPathCombo->findData(QString::fromUtf8(preset->camera_path)) : -1;
	if (pathIndex >= 0) m_cameraPathCombo->setCurrentIndex(pathIndex);

	if (m_videoFramesSpinBox) m_videoFramesSpinBox->setValue(preset->frames);
	if (m_videoFPSSpinBox) m_videoFPSSpinBox->setValue(preset->fps);
	if (m_videoSpeedSpinBox) m_videoSpeedSpinBox->setValue(preset->speed);
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

void MainWindow::selectOutputMode(OutputMode mode) {
	const int index = m_modeCombo->findData(static_cast<int>(mode));
	if (index < 0) return;  // e.g. LivePreview's item doesn't exist on a non-GPU build
	if (m_modeCombo->currentIndex() != index) m_modeCombo->setCurrentIndex(index);
}

void MainWindow::onModeChanged(int index) {
	m_outputMode = static_cast<OutputMode>(m_modeCombo->itemData(index).toInt());
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	// Live Preview only ever runs while it's the selected mode - leaving it
	// stops any in-progress session, the same "only costs anything while
	// actually being watched" intent stopLivePreviewIfNavigatedAway()
	// already applies to navigating away from the live sub-tab. This also
	// means starting a batch render (which forces the mode
	// combo to Image/Video first, via m_actRender/m_actRenderVideo) always
	// cleanly stops a running preview first, with no separate error/prompt.
	// m_outputMode is updated above, before this call, so stopLivePreview()'s
	// own updateTransportButtons() call already sees the new mode instead of
	// evaluating isLiveMode() against the mode being left.
	if (m_livePreviewRunning && !isLiveMode())
		stopLivePreview();
#endif

	// Every control in Video Generation Settings is inert unless Output Mode
	// is "Generate Video" - the warning label (see its own comment,
	// mainwindow.h) surfaces that, rather than disabling the group outright
	// and blocking browsing/configuring it ahead of switching modes (which
	// would also silently break onVideoPresetChanged()'s auto-switch-to-
	// Video-mode behavior below, since a disabled combo can't be opened to
	// pick a preset from in the first place). Same convention for Live
	// Preview's own warning label(s).
	if (m_videoModeWarningLabel) m_videoModeWarningLabel->setVisible(!isVideoMode());
	// Dims (never disables - see setGroupDimmed()'s own comment) the two
	// groups whose fields the banners above are talking about, so
	// irrelevance also reads at a glance instead of only via the text.
	setGroupDimmed(m_videoGroupBox, !isVideoMode());
	setGroupDimmed(m_advancedParamsGroupBox, isLiveMode());
	// The Render Options tab's "Denoiser" group's two subsections - same
	// dimming, one level deeper (mainwindow_tabs_render.cpp's own comment on
	// m_denoiserGroupBox).
	setGroupDimmed(m_denoiserImageVideoGroupBox, isLiveMode());
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	if (m_liveModeWarningLabel) m_liveModeWarningLabel->setVisible(isLiveMode());
	if (m_liveModeOptionsWarningLabel) m_liveModeOptionsWarningLabel->setVisible(isLiveMode());
	setGroupDimmed(m_liveModeSettingsGroupBox, !isLiveMode());
	setGroupDimmed(m_denoiserLivePreviewGroupBox, !isLiveMode());
	setGroupDimmed(m_liveRenderSettingsGroupBox, !isLiveMode());
#endif

	// --video hard-rejects any non-Default integrator (see
	// m_integratorVideoWarningLabel's own comment, mainwindow.h) - also
	// toggled from onIntegratorChanged() below, since either control can
	// create or resolve the conflict.
	const auto currentIntegrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	const bool showIntegratorVideoWarning = isVideoMode() && currentIntegrator != IntegratorMode::Default;
	m_integratorVideoWarningLabel->setVisible(showIntegratorVideoWarning);
	m_integratorVideoWarningLabelBasic->setVisible(showIntegratorVideoWarning);

	// Update the pinned Render button's label/icon/tooltip and the status
	// line to match the selected mode. Every label below keeps the same
	// Alt+R mnemonic, so the keyboard shortcut doesn't move when the
	// output mode changes.
	switch (m_outputMode) {
	case OutputMode::Video:
		m_renderButton->setText(tr("START VIDEO &RENDER"));
		icon_tint::apply(m_renderButton, ":/icons/video.svg",
		                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
		m_renderButton->setToolTip(tr("Renders the camera path frame by frame and assembles a video. "
		                              "Queues behind it instead if a render is already running."));
		m_statusLabel->setText(tr("Ready to render video frames"));
		break;
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	case OutputMode::LivePreview:
		m_renderButton->setText(tr("START LIVE &PREVIEW"));
		icon_tint::apply(m_renderButton, ":/icons/gpu.svg",
		                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
		m_renderButton->setToolTip(tr("Starts an interactive GPU preview you can orbit/zoom with the mouse. "
		                              "Disabled while a batch render is running."));
		m_statusLabel->setText(tr("Ready to start live preview"));
		break;
#endif
	case OutputMode::Image:
	default:
		m_renderButton->setText(tr("START &RENDER"));
		icon_tint::apply(m_renderButton, ":/icons/render.svg",
		                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
		m_renderButton->setToolTip(tr("Renders the selected scene with the current settings. "
		                              "Queues behind it instead if a render is already running."));
		m_statusLabel->setText(tr("Ready to render"));
		break;
	}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	updateTransportButtons();
#endif

	// Log mode change
	QString modeName = tr("Single Image");
	if (isVideoMode()) modeName = tr("Video Generation");
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	else if (isLiveMode()) modeName = tr("Live Preview");
#endif
	onLogMessage(tr("Mode changed to: %1").arg(modeName));
}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
// Single source of truth for the pinned Render/Stop/Pause/Abandon buttons'
// enabled state whenever Live Preview is (or was just) involved - called
// from onModeChanged(), startRenderJob(), onRenderComplete(), and both
// startLivePreview()/stopLivePreview(). Image/Video mode's own existing
// enablement rules (m_renderButton always stays enabled to queue another
// job; Stop/Pause/Abandon follow m_isRendering) are untouched and set
// directly at their own call sites - this function only needs to add the
// Live-Preview-aware overrides on top, and only actually changes anything
// when Live Preview is the selected mode or a preview is still running.
void MainWindow::updateTransportButtons() {
	if (isLiveMode()) {
		// No queue for Live Preview: only one preview can run, and it can't
		// even start while a batch render (queued from before the mode was
		// switched) still owns the GPU.
		m_renderButton->setEnabled(!m_isRendering && !m_livePreviewRunning);
		// If a batch render from before the mode switch is still running,
		// its own Stop/Pause/Abandon state (set by startRenderJob()) is what
		// the buttons must reflect - Live Preview being selected doesn't
		// mean nothing else is happening. Only override them for what Live
		// Preview itself controls once that batch job is out of the way.
		if (!m_isRendering) {
			m_stopButton->setEnabled(m_livePreviewRunning);
			m_pauseButton->setEnabled(false);
			m_abandonButton->setEnabled(false);
		}
		return;
	}
	if (m_livePreviewRunning) {
		// Image/Video mode is selected (or just switched to) but a preview
		// from before the mode change hasn't finished stopping yet -
		// shouldn't normally be observable, since onModeChanged() stops it
		// synchronously, but stay consistent regardless.
		m_renderButton->setEnabled(false);
		m_stopButton->setEnabled(true);
		m_pauseButton->setEnabled(false);
		m_abandonButton->setEnabled(false);
	}
	// Otherwise: Image/Video mode, no preview involved - leave whatever
	// startRenderJob()/onStopClicked()/onPauseClicked()/onRenderComplete()
	// already set.
}
#endif

void MainWindow::onIntegratorChanged(int) {
	const auto integrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());

	// 7 of the 8 alternate integrators are CPU-only (the CLI just warns
	// and forces CPU under --gpu, never rejects - see launcher/main.cpp's
	// own gpu_flag_explicit warnings); SPPM is the one exception with a
	// real, scene-dependent GPU path - but that path is OptiX's own
	// optix_render_main_sppm() specifically (launcher/main.cpp:1189-1201),
	// which unlike every other integrator's own GPU fallback does NOT warn
	// and continue on CPU - it hard-errors ("OptiX is not available!
	// (--sppm --gpu requires OptiX)") and aborts the render, since GPU
	// SPPM has no equivalent Metal implementation at all. kGpuOptionAvailable
	// (true only on the Windows/OptiX build - mainwindow.h) is what SPPM's
	// own GPU capability actually depends on; without it (macOS/Metal,
	// or any non-OptiX build) SPPM is exactly as CPU-only as the other 7 -
	// found via a direct audit of the actual failure path, not assumed
	// from the "one real exception" framing this comment used to make
	// unconditionally. Mirrors onSceneChanged()'s existing GPU-compat
	// auto-switch (same "no failure, just a stale/misleading control"
	// class of problem this fixes for the other 7 integrators).
	const bool sppmGpuCapable = kGpuOptionAvailable;
	const bool cpuOnly = (integrator != IntegratorMode::Default)
		&& !(integrator == IntegratorMode::Sppm && sppmGpuCapable);
	m_renderModeCombo->setEnabled(!cpuOnly);
	if (cpuOnly && m_renderModeCombo->currentData().toBool()) {
		m_renderModeCombo->setCurrentIndex(m_renderModeCombo->count() - 1); // last item = CPU; fires its own lambda -> updateRenderOptionsEnabled()
	}

	// Switch the "Integrator Options" stack to this integrator's own
	// sub-flag page - see m_integratorOptionsStack's own comment
	// (mainwindow.h) for why only 5 of the 8 modes get a real page.
	// Default/RandomWalk/SimpleVolPath/LightPath (no sub-flags) share page
	// 0, whose text is the same integratorDescription() content the other
	// 5 pages now show at their own top (mainwindow_tabs.cpp) and the
	// per-item combo tooltips already use - one description per mode, not
	// a separately-maintained shorter duplicate here.
	int page = 0;
	switch (integrator) {
		case IntegratorMode::Sppm: page = 1; break;
		case IntegratorMode::Bdpt: page = 2; break;
		case IntegratorMode::Mlt: page = 3; break;
		case IntegratorMode::Ao: page = 4; break;
		case IntegratorMode::SimplePath: page = 5; break;
		case IntegratorMode::Default:
		case IntegratorMode::RandomWalk:
		case IntegratorMode::SimpleVolPath:
		case IntegratorMode::LightPath:
		default:
			page = 0;
			break;
	}
	if (page == 0) m_integratorNoOptionsLabel->setText(integratorDescription(integrator));
	m_integratorOptionsStack->setCurrentIndex(page);

	updateRenderOptionsEnabled();

	// See onModeChanged()'s own comment - either control can create or
	// resolve the --video + non-Default-integrator conflict.
	const bool showIntegratorVideoWarning = isVideoMode() && integrator != IntegratorMode::Default;
	m_integratorVideoWarningLabel->setVisible(showIntegratorVideoWarning);
	m_integratorVideoWarningLabelBasic->setVisible(showIntegratorVideoWarning);

	refreshStatusBarInfo();
	if (isVisible()) onLogMessage(tr("Integrator changed to: %1").arg(m_integratorCombo->currentText()));   // not the start-up default being applied

	if (m_sceneCombo) updateSceneRecommendedSettingsHint(m_sceneCombo->currentData().toString());
}

void MainWindow::assembleVideoAutomatically(const QString &baseOutputPath, const RenderJob &job) {
	// ray_tracer.exe assembles the video itself (via ffmpeg) before it exits.
	// This just finds and opens the resulting file. In the normal case
	// ray_tracer.exe already exits non-zero if ffmpeg failed - see
	// onRenderComplete()'s failure branch - so the directory-glob fallback
	// below is mainly a defensive path for the case where the process
	// exited 0 but the expected video filename wasn't where we expect it.

	// Wait a moment for file to be fully written
	QThread::msleep(500);

	// main.cpp derives the final video's name from the SAME stem it was
	// given via --output (see its own "Output Video" step: "<stem>_video.
	// mp4"), so the expected path can be computed directly from
	// baseOutputPath - QFileInfo::completeBaseName() strips exactly one
	// extension, matching std::filesystem::path::stem(), so this lands on
	// the same name regardless of whether baseOutputPath still has its
	// original .ppm extension or was already normalized to .png upstream.
	QFileInfo baseInfo(baseOutputPath);
	const QString outputDir = baseInfo.absolutePath();
	const QString expectedVideoPath = outputDir + "/" + baseInfo.completeBaseName() + "_video.mp4";

	QString videoPath;
	QFileInfo videoInfo;

	if (!baseOutputPath.isEmpty() && QFileInfo::exists(expectedVideoPath)) {
		videoPath = expectedVideoPath;
		videoInfo = QFileInfo(videoPath);
	} else {
		// Fallback: search the same directory the render actually wrote to
		// (not a hardcoded app-relative one - now that every render's base
		// path is unique, that would never find anything).
		QDir dir(outputDir);
		QStringList filters;
		filters << "*_video.mp4" << "video.mp4";
		QFileInfoList videoFiles = dir.entryInfoList(filters, QDir::Files, QDir::Time);
		if (!videoFiles.isEmpty()) {
			videoInfo = videoFiles.first();
			videoPath = videoInfo.absoluteFilePath();
		}
	}

	if (videoPath.isEmpty()) {
		m_statusLabel->setText(tr("⚠️ Video file not found, checking for frames..."));
		onLogMessage(tr("WARNING: Video file not found at any of the expected locations"));

		// Check if frames exist (fallback diagnostic)
		QString framesDir = outputDir + "/frames";
		QDir framesDirObj(framesDir);

		if (framesDirObj.exists()) {
			QStringList frames = framesDirObj.entryList(QStringList() << "frame_*.ppm", QDir::Files);
			if (!frames.isEmpty()) {
				m_statusLabel->setText(tr("⚠️ Found %1 frames but no video file").arg(frames.count()));
				onLogMessage(tr("Frames were rendered (%1 files) but video assembly may have failed.").arg(frames.count()));
				QMessageBox::warning(this, tr("Video Not Created"),
					tr("Frames were rendered successfully (%1 files), but the video file was not created.\n\n"
							"Expected video at: %2\n\n"
							"Please check the render log for ffmpeg errors.").arg(frames.count()).arg(expectedVideoPath));
			} else {
				m_statusLabel->setText(tr("❌ No frames or video found"));
				onLogMessage(tr("ERROR: No frames or video file found"));
				QMessageBox::critical(this, tr("Render Failed"),
					tr("Neither frames nor video file were created.\n\nPlease check the render log for errors."));
			}
		} else {
			m_statusLabel->setText(tr("❌ Frames directory not found"));
			onLogMessage(tr("ERROR: Frames directory not found: %1").arg(framesDir));
			QMessageBox::critical(this, tr("Directory Not Found"),
				tr("Frames directory not found:\n%1\n\nThe render may have failed to create output.").arg(framesDir));
		}
		return;
	}

	// Video file found! Open it
	m_statusLabel->setText(tr("✅ Video created successfully!"));
	onLogMessage(tr("✅ Video assembled successfully: %1").arg(videoPath));
	onLogMessage(tr("Video size: %1 MB").arg(videoInfo.size() / (1024.0 * 1024.0), 0, 'f', 2));

	// Add it as a new Preview sub-tab (see addVideoPreviewTab()) - its own
	// player, its own tab, sitting alongside whatever earlier renders are
	// already there rather than replacing a single shared pane.
	//
	// Frame count for the info label comes from the enc_*.png sequence -
	// main.cpp's BackgroundPngConverter (launcher/main.cpp) converts each
	// frame_NNNN.ppm straight to a contiguously-renumbered enc_NNNN.png for
	// ffmpeg's sequential-input requirement, and those files are never
	// cleaned up after a successful assembly, so they're still there to
	// count.
	QDir framesDirForPreview(outputDir + "/frames");
	QStringList frameFiles = framesDirForPreview.entryList(QStringList() << "enc_*.png", QDir::Files, QDir::Name);

	// Named scene+path+frames/fps/speed bundle if one was selected (see
	// video_preset.h and captureRenderJob()'s own comment on job.
	// videoPresetName); a custom (non-preset) render falls back to the
	// scene's own name, distinguished with a "(Video)" suffix so it can't
	// be confused with an image-mode tab of the same scene.
	QString title = job.videoPresetName;
	if (title.isEmpty()) title = tr("%1 (Video)").arg(job.displayTitle);

	const QString infoText = tr("%1  •  %2 MB  •  %3 frames  •  %4spp · %5%6")
		.arg(videoInfo.fileName())
		.arg(videoInfo.size() / (1024.0 * 1024.0), 0, 'f', 1)
		.arg(frameFiles.count())
		.arg(job.samples)
		.arg(rendererLabel(job.useGPU, job.useWavefront), integratorSuffixTag(job.integratorOptions.mode));
	addVideoPreviewTab(title, job.sceneDescription, videoPath, infoText,
						{job.sceneId, renderTechniqueHtml(job.integratorOptions, job.advancedFlags)});
	saveRecentRender(job, videoPath, /*isVideo=*/true, title);
	refreshRecentRendersList();
	if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);

	onLogMessage(tr("Playing video inline: %1").arg(videoPath));
}
