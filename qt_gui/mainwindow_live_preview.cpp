// mainwindow_live_preview.cpp - the Live Preview output mode: starting and stopping the session, frames and status, and pushing the Render Options
// controls to the running session (split out of mainwindow_tabs_render.cpp; nothing changed). Compiled only where Live Preview exists.

#include "mainwindow.h"
#include "live_ai_denoise.h"
#include "live_object_editor.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"
#include "settings_keys.h"

#include "../src/shared/scene_descriptor.h"

#include <cmath>

#include <QTabBar>
#include <QStatusBar>
#include "scene_builder_widget.h"
#include "scene_metadata_client.h"
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
#include "realtime_preview_session.h"
#endif
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QStyleFactory>
#include <QPalette>
#include <QProcess>
#include <QDir>
#include <QDateTime>
#include <QScrollArea>
#include <QScrollBar>
#include <QScreen>
#include <QTimer>
#include <QAbstractItemView>
#include <QIcon>
#include <QDesktopServices>
#include <QUrl>
#include <QSplitter>
#include <QStackedWidget>
#include <QSlider>
#include <QStandardPaths>
#include <QFile>
#include <QToolButton>
#include <QSettings>
#include <cmath>
#include <algorithm>


#ifdef RT_GUI_HAVE_LIVE_PREVIEW

// Live Preview - GPU progressive-refinement preview (see this project's own
// real-time-preview plan), just another Output Mode (see that enum's own
// comment, mainwindow_jobtypes.h) driven by the same pinned Render/Stop
// button pair every other mode uses - no button of its own. Its running
// image shows up as an ordinary sub-tab under the Preview tab, exactly like
// a finished Image/Video render (addLivePreviewTab() mirrors
// addImagePreviewTab()/addVideoPreviewTab()'s shape), rather than a
// dedicated top-level tab of its own. Camera comes from the existing
// m_cameraPosX/Y/Z spinboxes (read at Start time, and live-forwarded via
// onLivePreviewCameraChanged() while running) as a starting point, PLUS
// click-drag-to-orbit/wheel-to-zoom directly in the preview image
// (OrbitPreviewLabel, mainwindow_widgets.h) - see m_orbit's own comment
// (mainwindow.h) for how those two camera representations stay in sync.
// The actual spherical-coordinate math lives in camera_math.h (Qt-free,
// unit-tested - see tests/unit/camera_math_tests.cpp), matching this
// codebase's own existing convention for camera arithmetic
// (onCameraDistanceChanged()'s own comment, mainwindow_slots.cpp); the
// functions here are just plumbing that reads/writes m_orbit and forwards
// the result to RealtimePreviewSession.
void MainWindow::initLivePreviewSession() {
	if (!RealtimePreviewSession::isAvailable()) {
		// Same "fail quiet, explain why" pattern as every scene_metadata.dll
		// query - realtime_renderer.dll missing/wrong-arch/etc. shouldn't
		// crash the GUI, just leave the session null. The Output Mode
		// combo's own "Live Preview" item is separately disabled with a
		// matching tooltip - see createSettingsTab()'s own comment - so
		// startLivePreview()'s own null guard is unreachable through
		// normal UI in this case.
		return;
	}

	m_livePreviewSession = new RealtimePreviewSession(this);
	connect(m_livePreviewSession, &RealtimePreviewSession::frameReady,
	        this, &MainWindow::onLivePreviewFrameReady);
	connect(m_livePreviewSession, &RealtimePreviewSession::statusChanged,
	        this, &MainWindow::onLivePreviewStatus);
}

void MainWindow::addLivePreviewTab(const QString &sceneId, const QString &sceneName) {
	if (!m_previewSubTabs) return;

	// A previous session's sub-tab, if still open (Stop leaves it in place,
	// showing the frozen last frame, until closed - see stopLivePreview()'s
	// own comment), would otherwise pile up indefinitely across repeated
	// Start/Stop cycles instead of being replaced by this one. Reuses
	// closePreviewSubTab()'s own stop-before-clear teardown rather than
	// duplicating it here; its stopLivePreview() call is a harmless no-op
	// since the old session is already stopped by this point (startLivePreview()'s
	// own guard wouldn't have let a second Start through otherwise).
	if (m_livePreviewPage) {
		const int oldIndex = m_previewSubTabs->indexOf(m_livePreviewPage);
		if (oldIndex >= 0) closePreviewSubTab(oldIndex);
	}

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(12, 12, 12, 12);

	m_livePreviewLabel = new OrbitPreviewLabel(page);
	m_livePreviewLabel->setMinimumSize(200, 200);
	m_livePreviewLabel->setPlaceholderText(
		tr("Waiting for first frame...\n\nDrag to orbit, scroll or +/- to zoom, "
		   "WASD to move, Up/Down to fly"));
	connect(m_livePreviewLabel, &OrbitPreviewLabel::orbitDragged, this, &MainWindow::onLivePreviewOrbitDragged);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::zoomRequested, this, &MainWindow::onLivePreviewZoomRequested);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::keyOrbitRequested, this, &MainWindow::onLivePreviewKeyOrbit);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::keyZoomRequested, this, &MainWindow::onLivePreviewKeyZoom);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::translateRequested, this, &MainWindow::onLivePreviewTranslate);
	layout->addWidget(m_livePreviewLabel, /*stretch=*/1);
	// Grabs keyboard focus immediately so arrow-key/+/- navigation works
	// without an extra click first (which would itself start an orbit
	// drag) - safe to call before the page is even shown, since a fresh
	// label is created on every Start (see this function's own guard
	// above), so there's always exactly one live-preview label at a time
	// that legitimately wants focus.
	m_livePreviewLabel->setFocus();

	m_livePreviewStatusLabel = new QLabel(page);
	m_livePreviewStatusLabel->setAlignment(Qt::AlignCenter);
	layout->addWidget(m_livePreviewStatusLabel);
	m_liveObjectEditor = nullptr;
	if (m_livePreviewSession && RealtimePreviewSession::objectEditingAvailable()) {
		m_liveObjectEditor = new LiveObjectEditor(m_livePreviewSession, m_livePreviewLabel, m_livePreviewSceneSize, page);
		layout->addWidget(m_liveObjectEditor->createControls(page));
	}

	// No outputPath/previewPath/techniqueHtml - there's no file on disk and
	// no completed-render settings summary, so Open Folder/Open Viewer
	// correctly stay disabled (currentPreviewProperty() on an unset
	// property just yields an empty string) rather than needing special-
	// casing. sceneId IS set, so updatePreviewSidebarForActiveTab()'s
	// existing scene_technique_notes lookup works for this page too.
	page->setProperty("infoText", tr("Live Preview — %1").arg(sceneName));
	page->setProperty("sceneId", sceneId);

	m_livePreviewPage = page;
	addPreviewSubTabPage(page, tr("Live Preview"),
		tr("Interactive GPU preview - drag to orbit, scroll or +/- to zoom, "
		"WASD to move, Up/Down to fly, Left/Right to orbit"));
}

bool MainWindow::isLivePreviewSubTabVisible() const {
	return m_tabWidget->currentIndex() == m_previewTabIndex
		&& m_previewSubTabs && m_livePreviewPage
		&& m_previewSubTabs->currentWidget() == m_livePreviewPage;
}

void MainWindow::stopLivePreviewIfNavigatedAway() {
	if (isLivePreviewSubTabVisible()) return;
	if (m_livePreviewLabel) m_livePreviewLabel->cancelDrag();
	if (m_livePreviewRunning) stopLivePreview();
}

void MainWindow::startLivePreview() {
	if (!m_livePreviewSession || m_livePreviewRunning) return;

	const QString sceneId = m_sceneCombo->currentData().toString();
	if (sceneId.isEmpty()) {
		// No live sub-tab exists yet at this point (it's only created
		// below, once a preview actually starts) - m_statusLabel is the
		// one status surface that's always available regardless of which
		// tab is open, matching how every other "can't start" message
		// (Stopping/Abandoning/Paused/etc.) already reports through it.
		m_statusLabel->setText(tr("Select a scene first"));
		return;
	}
	// A modest size keeps the per-frame cost low regardless of the Settings tab's own width/height (this preview is about interactive feedback, not a final-quality
	// render at the requested output size), but it keeps that setting's ASPECT RATIO, so the preview frames the scene like the image render will (camera_math::
	// previewSizeFor(): the longer side is 400 pixels). On every platform; Windows used a fixed 400x300, which showed a square scene small with bars at the sides.
	const camera_math::PreviewSize previewSize = camera_math::previewSizeFor(m_widthSpinBox ? m_widthSpinBox->value() : 0.0, m_heightSpinBox ? m_heightSpinBox->value() : 0.0);
	const int kPreviewWidth = previewSize.width, kPreviewHeight = previewSize.height;
	// Seed both the orbit state AND the free-fly pivot from wherever the
	// camera spinboxes currently point, around the CURRENT scene's own
	// lookAt point (currentLookAt()) - a fresh starting pivot every
	// session, exactly like m_orbit itself. From here on, m_livePreviewLookAt
	// (not currentLookAt()) is what orbiting/zooming/translating actually
	// revolves around - see its own comment (mainwindow.h).
	const camera_math::Vec3 camera = currentCameraPosition();
	m_livePreviewLookAt = currentLookAt();
	m_orbit = camera_math::cartesianToOrbit(camera, m_livePreviewLookAt);
	// How big this scene is, which the keyboard step (WASD, Up/Down) is a fraction of: the size its file declares ("# @rt-size", a bounding-box side in the
	// scene's own units), else the camera's distance to its target, which is at least the right order of magnitude for any scene whose author framed it.
	{
		SceneMetadataClient::SceneMetadata sizeMeta;
		const double declared = SceneMetadataClient::sceneMetadata(sceneId, sizeMeta) ? sizeMeta.sceneSize : 0.0;
		m_livePreviewSceneSize = camera_math::effectiveSceneSize(declared, m_orbit.radius);
	}
	// Exposure and SVGF tuning aren't part of start()'s own parameter list
	// (neither has an accumulation-structure side effect - see setExposure()'s
	// own comment - so it's simplest to push them separately); pushed BEFORE
	// start() (not after) so the very first rendered frame already uses them,
	// not just frame 2 onward - start()'s own queued call synchronously
	// renders frame 1 as part of the SAME worker-thread event, so a push
	// queued AFTER start() would only take effect starting with frame 2.
	m_livePreviewSession->setExposure(m_liveExposure);
	pushLiveSvgfTuningToSession();
	pushLiveAdaptiveSamplingToSession();
	pushLiveSmoothNoiseToSession();
	pushLiveAutoExposureToSession();
	m_livePreviewSession->start(sceneId, kPreviewWidth, kPreviewHeight, camera.x, camera.y, camera.z,
								 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z,
								 m_liveDenoiseEnabled, m_liveDenoiseBlend, m_liveDenoiseShowLatest,
								 m_liveSvgfEnabled, m_liveRestirGiEnabled, m_liveRestirDiEnabled,
								 m_liveProbeCacheEnabled, m_livePathGuidingEnabled,
								 m_liveSamples, m_liveMaxDepth, m_liveFireflyClamp,
								 m_liveTemporalUpscaleFactor > 1, m_liveTemporalUpscaleFactor, m_liveNrcEnabled,
								 m_liveNeuralUpscaleEnabled,
							 m_liveDofEnabled, m_liveAperture, m_liveFocusDistance);
	// A settings snapshot at the moment of start - the only record of what a
	// given session actually ran with, since every m_live*Enabled member is
	// live app state (not persisted per-session) and the toggle checkboxes
	// keep going after this point. Reset alongside it: m_livePreviewFrameCount/
	// m_livePreviewSessionTimer back this session's own stopLivePreview()
	// duration/fps log, not a running total across repeated Start/Stop cycles.
	onLogMessage(QString("[Live Preview] Starting: scene=%1, %2x%3, denoise=%4, svgf=%5, "
		"restirGI=%6, restirDI=%7, probeCache=%8, pathGuiding=%9, nrc=%10, upscale=%11, dof=%12")
		.arg(sceneId).arg(kPreviewWidth).arg(kPreviewHeight)
		.arg(m_liveDenoiseEnabled ? "on" : "off", m_liveSvgfEnabled ? "on" : "off",
			 m_liveRestirGiEnabled ? "on" : "off", m_liveRestirDiEnabled ? "on" : "off")
		.arg(m_liveProbeCacheEnabled ? "on" : "off", m_livePathGuidingEnabled ? "on" : "off",
			 m_liveNrcEnabled ? "on" : "off",
			 m_liveTemporalUpscaleFactor > 1 ? QString("%1x").arg(m_liveTemporalUpscaleFactor) : QString("off"))
		.arg(m_liveDofEnabled ? "on" : "off"));
	m_livePreviewFrameCount = 0;
	m_livePreviewSessionTimer.start();
	// m_livePreviewRunning stays false until BOTH tab switches below have
	// happened. addLivePreviewTab()'s own m_previewSubTabs->setCurrentIndex()
	// call (and the m_tabWidget switch after it) synchronously re-emit
	// currentChanged - QTabBar's own signal is direct, not queued - which
	// reaches stopLivePreviewIfNavigatedAway() reentrantly, inside this very
	// function call, before either switch has fully landed. Setting this
	// flag only once both are done means that reentrant call's own
	// "if (m_livePreviewRunning) stopLivePreview();" guard is still false
	// and a no-op, instead of killing the session this function just
	// started before it ever got to run.
	addLivePreviewTab(sceneId, SceneMetadataClient::sceneName(sceneId));
	// Same "click Render -> land where you watch it happen" behavior every
	// other Output Mode already gets from startRenderJob()'s own switch to
	// the Progress tab - just a different destination tab for this mode.
	if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
	m_livePreviewRunning = true;
	m_livePreviewStatusLabel->setText(tr("Starting..."));
	updateTransportButtons();
	updateActionStates();  // Escape (m_actStop) becomes enabled - see its own comment
}

void MainWindow::stopLivePreview() {
	if (!m_livePreviewSession || !m_livePreviewRunning) return;
	m_livePreviewSession->stop();
	m_livePreviewRunning = false;
	// Defensively ends an in-progress orbit drag the same way
	// stopLivePreviewIfNavigatedAway() already does - Escape (m_actStop's
	// shortcut) and the Stop button both route here directly, and neither
	// waits for a mouseReleaseEvent that may never arrive if the mouse
	// button is still held when Stop fires. Also covers closePreviewSubTab(),
	// which calls this function before tearing the page down.
	if (m_livePreviewLabel) m_livePreviewLabel->cancelDrag();
	m_livePreviewStatusLabel->setText(tr("Stopped"));
	// The only record of a session's actual duration/throughput -
	// m_livePreviewFrameCount/m_livePreviewSessionTimer are reset together
	// at the top of startLivePreview(), so this always reports against just
	// the session that's ending, not a running total.
	const double elapsedSec = m_livePreviewSessionTimer.elapsed() / 1000.0;
	const double avgFps = elapsedSec > 0.0 ? m_livePreviewFrameCount / elapsedSec : 0.0;
	onLogMessage(QString("[Live Preview] Stopped after %1s (%2 frames, avg %3 fps)")
		.arg(elapsedSec, 0, 'f', 1).arg(m_livePreviewFrameCount).arg(avgFps, 0, 'f', 1));
	updateTransportButtons();
	updateActionStates();  // Escape (m_actStop) becomes disabled again
}

void MainWindow::onLivePreviewFrameReady(QImage image, int sampleCount) {
	// m_livePreviewLabel/m_livePreviewStatusLabel are only ever set together
	// (addLivePreviewTab()) or cleared together (closePreviewSubTab()), but
	// guarding both here rather than relying on that invariant costs
	// nothing and doesn't assume a future edit can't decouple them.
	if (!m_livePreviewLabel || !m_livePreviewStatusLabel) return;
	++m_livePreviewFrameCount;
	m_livePreviewLabel->setPreviewPixmap(QPixmap::fromImage(image));
	// While effectively showing the latest frame instead of accumulating
	// (RealtimePreviewWorker::renderLoop()'s own gating condition), sampleCount
	// is really just a frame counter - no real accumulation is happening, so
	// "N samples" would misleadingly imply ongoing convergence.
	if (m_liveDenoiseEnabled && m_liveDenoiseShowLatest) {
		m_livePreviewStatusLabel->setText(tr("Live (denoised, not accumulating)"));
	} else {
		m_livePreviewStatusLabel->setText(tr("%1 samples").arg(sampleCount));
	}
}

void MainWindow::onLivePreviewStatus(QString text) {
	// Both statusChanged() call sites (RealtimePreviewWorker::renderLoop())
	// are genuine failures, not routine progress - unlike onLivePreviewFrameReady()'s
	// own per-frame sample-count text just above, which is never routed to
	// the Log tab. Logged only when the text actually CHANGES (compared
	// against what the label is already showing, BEFORE overwriting it) -
	// renderLoop() reposts itself every frame even after a failure and would
	// otherwise re-emit the identical message every frame indefinitely,
	// flooding the Log tab with duplicate lines for one single underlying
	// problem.
	if (m_livePreviewStatusLabel) {
		if (m_livePreviewStatusLabel->text() != text) {
			// "ERROR" (not just the message text on its own) guarantees
			// render_output_parser.h's classifyLogLine() flags this as an
			// error line (its containsNoCase(line, "error") rule) - the two
			// actual messages ("realtime_renderer.dll not found...",
			// "Render failed - scene may not be GPU-supported...") don't
			// reliably match any of its rules on their own (lowercase
			// "failed" misses the case-sensitive "FAILED" check), so without
			// this a live preview failure rendered as a plain, unflagged
			// line - much easier to miss than a real render's own failures.
			onLogMessage(tr("[Live Preview] ERROR: %1").arg(text));
		}
		m_livePreviewStatusLabel->setText(text);
	}
}

void MainWindow::onLivePreviewCameraChanged() {
	if (!m_livePreviewRunning || !m_livePreviewSession) return;
	const camera_math::Vec3 camera = currentCameraPosition();
	m_orbit = camera_math::cartesianToOrbit(camera, m_livePreviewLookAt);
	if (m_liveObjectEditor) m_liveObjectEditor->cameraMoved();
	m_livePreviewSession->setCamera(camera.x, camera.y, camera.z,
									 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z);
}

void MainWindow::updateLivePreviewCameraFromOrbit() {
	if (!m_livePreviewRunning || !m_livePreviewSession) return;
	const camera_math::Vec3 camera = camera_math::orbitToCartesian(m_orbit, m_livePreviewLookAt);
	if (m_liveObjectEditor) m_liveObjectEditor->cameraMoved();
	m_livePreviewSession->setCamera(camera.x, camera.y, camera.z,
									 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z);
}

// Each pushLive*ToSession() below is the one choke point its setting's
// checkbox/combo goes through regardless of whether it's being adjusted
// before a session even exists (nothing logged - see m_livePreviewSession
// guard) or live, mid-session (m_livePreviewRunning below) - only the
// latter is interesting for debugging "why did the image just change", so
// only that case logs. Exposure/samples/max-depth/firefly-clamp/SVGF-tuning
// have no equivalent log line: those are continuous sliders that fire
// repeatedly while being dragged, and would flood the Log Output tab.
void MainWindow::pushLiveDenoiseToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setDenoise(m_liveDenoiseEnabled, m_liveDenoiseBlend, m_liveDenoiseShowLatest);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] Denoise: %1").arg(m_liveDenoiseEnabled ? "on" : "off"));
}

void MainWindow::pushLiveSvgfToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSvgf(m_liveSvgfEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] SVGF: %1").arg(m_liveSvgfEnabled ? "on" : "off"));
}

void MainWindow::pushLiveRestirGiToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setRestirGi(m_liveRestirGiEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] ReSTIR GI: %1").arg(m_liveRestirGiEnabled ? "on" : "off"));
}

void MainWindow::pushLiveRestirDiToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setRestirDi(m_liveRestirDiEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] ReSTIR DI: %1").arg(m_liveRestirDiEnabled ? "on" : "off"));
}

void MainWindow::pushLiveProbeCacheToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setProbeCache(m_liveProbeCacheEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] Probe Cache: %1").arg(m_liveProbeCacheEnabled ? "on" : "off"));
}

void MainWindow::pushLivePathGuidingToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setPathGuiding(m_livePathGuidingEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] Path Guiding: %1").arg(m_livePathGuidingEnabled ? "on" : "off"));
}

void MainWindow::pushLiveNrcToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setNrc(m_liveNrcEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] NRC: %1").arg(m_liveNrcEnabled ? "on" : "off"));
}

void MainWindow::pushLiveNeuralUpscaleToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setNeuralUpscale(m_liveNeuralUpscaleEnabled);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] Neural Upscale: %1").arg(m_liveNeuralUpscaleEnabled ? "on" : "off"));
}

void MainWindow::pushLiveAutoExposureToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setAutoExposure(m_liveAutoExposure);
}

void MainWindow::pushLiveSmoothNoiseToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSmoothLowSample(m_liveSmoothNoise);
	m_livePreviewSession->setAiDenoise(live_ai_denoise::savedEnabled());   // (the Mac-only display options travel together)
}

void MainWindow::pushLiveDofToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setDof(m_liveDofEnabled, m_liveAperture, m_liveFocusDistance);
	if (m_livePreviewRunning)
		onLogMessage(QString("[Live Preview] Depth of Field: %1").arg(m_liveDofEnabled ? "on" : "off"));
}

void MainWindow::pushLiveTemporalUpscaleToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setTemporalUpscale(m_liveTemporalUpscaleFactor > 1, m_liveTemporalUpscaleFactor);
	if (m_livePreviewRunning) {
		onLogMessage(QString("[Live Preview] Temporal Upscale: %1")
			.arg(m_liveTemporalUpscaleFactor > 1 ? QString("%1x").arg(m_liveTemporalUpscaleFactor) : QString("off")));
	}
}

void MainWindow::pushLiveExposureToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setExposure(m_liveExposure);
}

void MainWindow::pushLiveSppMaxDepthToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSppAndMaxDepth(m_liveSamples, m_liveMaxDepth);
}

void MainWindow::pushLiveFireflyClampToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setFireflyClamp(m_liveFireflyClamp);
}

void MainWindow::pushLiveAdaptiveSamplingToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setAdaptiveSampling(m_liveAdaptiveSamplingEnabled, m_liveAdaptiveSamplingThreshold);
}

void MainWindow::pushLiveSvgfTuningToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSvgfTuning(
		m_liveSvgfTemporalAlpha, m_liveSvgfMaxHistoryLength, m_liveSvgfVarianceBootstrapFrames,
		m_liveSvgfVarianceBootstrapRadius, m_liveSvgfSigmaNormal, m_liveSvgfSigmaDepth,
		m_liveSvgfSigmaLuminance, m_liveSvgfAtrousRadius, m_liveSvgfMinAlbedo, m_liveSvgfAtrousPasses);
}

// Applies a rotation to m_orbit and clamps elevation short of the true
// poles (+-90deg): AT a pole, azimuth becomes meaningless (every azimuth
// points the same direction), which would make the very next step's
// horizontal component do nothing/jump - matches the standard orbit-camera
// convention (Blender, Maya, etc.). Shared by the mouse-drag and keyboard
// arrow-key paths, which differ only in how they convert their own raw
// input (pixel deltas vs. discrete steps) into radians before calling this.
void MainWindow::applyOrbitDelta(double azimuthDelta, double elevationDelta) {
	m_orbit.azimuth += azimuthDelta;
	m_orbit.elevation += elevationDelta;
	constexpr double kMaxElevation = 1.5533;  // 89 degrees in radians
	if (m_orbit.elevation > kMaxElevation) m_orbit.elevation = kMaxElevation;
	if (m_orbit.elevation < -kMaxElevation) m_orbit.elevation = -kMaxElevation;
	updateLivePreviewCameraFromOrbit();
}

// Applies a zoom factor to m_orbit.radius and clamps it away from the
// lookAt point, where the view direction becomes degenerate. Shared by the
// mouse-wheel and keyboard +/- paths.
void MainWindow::applyZoomDelta(double factor) {
	m_orbit.radius *= factor;
	// Closest the camera may get to its pivot: 1 unit in the 555-unit Cornell box, in proportion for any other scene (a fixed 1 unit was most of a small room).
	const double kMinRadius = camera_math::minOrbitRadius(m_livePreviewSceneSize);
	if (m_orbit.radius < kMinRadius) m_orbit.radius = kMinRadius;
	updateLivePreviewCameraFromOrbit();
}

// Free-fly WASD/Up-Down translation - unlike applyOrbitDelta()/
// applyZoomDelta() above (which rotate/scale around a FIXED
// m_livePreviewLookAt), this MOVES the pivot together with the camera by
// the same world-space delta, so the camera keeps facing the same
// direction it already was rather than snapping to re-aim at wherever the
// pivot used to be. Basis is derived from the camera's CURRENT facing
// direction (forward = toward the pivot; right = perpendicular to forward
// in the horizontal plane; up = world +Y, confirmed the vertical axis
// throughout camera_math.h and the GPU renderer's own vup convention -
// scene_builder.cpp hardcodes make_float3(0,1,0) everywhere), not a
// separately-tracked orientation, so there's no drift between this and
// what orbitToCartesian() would compute from m_orbit right now.
//
// Matches applyOrbitDelta()/applyZoomDelta()'s own division of labor: the
// caller (onLivePreviewTranslate()) has already turned raw step counts into
// real world-space distances via kUnitsPerStep/m_keyboardSensitivity, so
// forwardDelta/rightDelta/upDelta here are plain distances along each basis
// vector, not step counts this function would need to scale itself.
void MainWindow::applyTranslateDelta(double forwardDelta, double rightDelta, double upDelta) {
	const camera_math::Vec3 camera = camera_math::orbitToCartesian(m_orbit, m_livePreviewLookAt);
	const camera_math::Vec3 forward = camera_math::normalized(m_livePreviewLookAt - camera);
	constexpr camera_math::Vec3 kWorldUp{0.0, 1.0, 0.0};
	// Derived directly from azimuth rather than normalized(cross(forward,
	// kWorldUp)): that cross product is forward's horizontal component
	// scaled by cos(elevation), which shrinks to the zero vector (silently
	// killing A/D strafing - normalized() of a near-zero vector is defined
	// to return zero, see camera_math.h) once the camera looks near-straight
	// up or down. The formula below is that same horizontal direction with
	// the cos(elevation) factor divided back out, so it stays unit-length
	// and well-defined at every elevation, including both poles, while
	// matching cross(forward, kWorldUp)'s own direction everywhere else.
	const camera_math::Vec3 right{std::cos(m_orbit.azimuth), 0.0, -std::sin(m_orbit.azimuth)};
	const camera_math::Vec3 delta = forward * forwardDelta + right * rightDelta + kWorldUp * upDelta;
	// Camera and pivot translate by the IDENTICAL delta, preserving
	// distance/orientation between them - re-deriving m_orbit via
	// cartesianToOrbit() rather than assuming it stays numerically
	// unchanged matches this file's own existing practice (e.g.
	// onLivePreviewCameraChanged()) of never assuming float-exact
	// preservation across a recomputation.
	m_livePreviewLookAt = m_livePreviewLookAt + delta;
	const camera_math::Vec3 newCamera = camera + delta;
	m_orbit = camera_math::cartesianToOrbit(newCamera, m_livePreviewLookAt);
	updateLivePreviewCameraFromOrbit();
}

void MainWindow::onLivePreviewOrbitDragged(int dxPixels, int dyPixels) {
	if (!m_livePreviewRunning) return;
	// Radians per pixel of drag - chosen so a full 180-degree turn takes
	// roughly the preview panel's own width in drag distance (~400px, this
	// pass's fixed preview resolution), a comfortable, not-too-twitchy feel
	// for a panel this size.
	constexpr double kRadiansPerPixel = 0.008;
	// Screen Y grows downward, so dragging UP (dyPixels negative) should
	// raise the camera (increase elevation) - hence the negation.
	applyOrbitDelta(dxPixels * kRadiansPerPixel * m_mouseSensitivity,
					 -dyPixels * kRadiansPerPixel * m_mouseSensitivity);
}

void MainWindow::onLivePreviewZoomRequested(int angleDeltaY) {
	if (!m_livePreviewRunning) return;
	// ~5.8% radius change per standard wheel notch (angleDelta of +-120) -
	// a smooth, moderate zoom step; std::pow with a negative exponent
	// (scrolling the other way) naturally inverts it. Scaling the EXPONENT
	// (not the base) by sensitivity keeps 1.0x exactly today's behavior and
	// preserves "higher sensitivity = bigger effect, same direction" -
	// scaling the base instead would need a second formula to keep values
	// above/below 1.0 behaving symmetrically.
	constexpr double kZoomFactorPerUnit = 0.9995;
	applyZoomDelta(std::pow(kZoomFactorPerUnit, static_cast<double>(angleDeltaY) * m_mouseSensitivity));
}

// Keyboard equivalents of the two mouse handlers above - see
// OrbitPreviewLabel::keyOrbitRequested()/keyZoomRequested()'s own
// comments for why these are separate slots rather than routed through
// the mouse ones, and mainwindow.h's comment on m_keyboardSensitivity
// for why it's one multiplier per device rather than per axis.
void MainWindow::onLivePreviewKeyOrbit(int azimuthSteps, int elevationSteps) {
	if (!m_livePreviewRunning) return;
	// Deliberately coarser per-press than the mouse's per-pixel rate (which
	// fires many times over one drag) - a single key press should be a
	// noticeable, discrete nudge, not an imperceptible fraction of one.
	constexpr double kRadiansPerKeyStep = 0.05;
	applyOrbitDelta(azimuthSteps * kRadiansPerKeyStep * m_keyboardSensitivity,
					 elevationSteps * kRadiansPerKeyStep * m_keyboardSensitivity);
}

void MainWindow::onLivePreviewKeyZoom(int radiusSteps) {
	if (!m_livePreviewRunning) return;
	// ~15% radius change per press at 1.0x - a single press should read as
	// a deliberate zoom step, matching kRadiansPerKeyStep's own "noticeable
	// per press" intent above rather than the mouse wheel's much finer
	// per-notch granularity.
	constexpr double kZoomFactorPerKeyStep = 0.85;
	applyZoomDelta(std::pow(kZoomFactorPerKeyStep, radiusSteps * m_keyboardSensitivity));
}

void MainWindow::onLivePreviewTranslate(int forwardSteps, int rightSteps, int upSteps) {
	if (!m_livePreviewRunning) return;
	// World-space distance per step: a fraction of the scene's size, so one press is a sensible distance in a 6-unit room and in the 555-unit Cornell box alike.
	// (It used to be a flat 20 units, tuned on the Cornell box, which flew straight out of any small scene.) 2% of the scene is about 11 units in the Cornell box,
	// a little over half the old step: a handful of presses crosses the room. The Settings tab's keyboard sensitivity still scales it, 0.25x to 3x.
	// Scaled here rather than inside applyTranslateDelta(), matching onLivePreviewKeyOrbit()/onLivePreviewKeyZoom()'s own "resolve steps to a real delta before
	// calling the shared apply* helper" convention.
	const double scale = camera_math::keyboardStep(m_livePreviewSceneSize, m_keyboardSensitivity);
	applyTranslateDelta(forwardSteps * scale, rightSteps * scale, upSteps * scale);
}

#endif
