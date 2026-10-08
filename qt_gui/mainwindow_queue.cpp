// mainwindow_queue.cpp - starting a render, and the render queue: capturing a job from the controls, running the next one, the queue panel and its buttons
// (split out of mainwindow_slots.cpp; nothing changed).

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


void MainWindow::onRenderClicked() {
	// m_sceneCombo can legitimately be empty - a search term that matches
	// nothing in the current category tab leaves it with no items and
	// currentIndex()==-1 (see onSceneChanged()'s own comment on this case) -
	// and the button itself is never disabled for that state. Without this
	// guard, captureRenderJob() below would read an empty scene id and
	// enqueue a render with no scene specified at all.
	if (!m_sceneCombo || m_sceneCombo->currentIndex() < 0) {
		setStatusWarning(tr("Can't start a render - no scene is selected (try clearing the search box)."));
		return;
	}

	// ThumbnailGenerator's own design intent (see its comment in mainwindow.h)
	// is to never compete with user-requested work - onGenerateThumbnailsClicked()
	// already enforces that direction by refusing to start while a render is
	// active/queued, but nothing enforced it the other way until now. A real,
	// user-requested render always wins: stop() drops the rest of the
	// thumbnail queue cleanly (already-generated thumbnails stay cached, and
	// re-clicking "Generate Thumbnails" later just picks up where it left
	// off), rather than letting two ray_tracer.exe processes run at once.
	if (m_thumbnailGenerator && m_thumbnailGenerator->isRunning()) {
		m_thumbnailGenerator->stop();
		onLogMessage("Paused background thumbnail generation to start this render.");
	}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	// Live Preview never becomes a RenderJob - it has no output file, no
	// completion state, and runs in-process rather than as a
	// RenderController/QProcess job (see OutputMode's own comment). Branch
	// away before touching captureRenderJob()/m_renderQueue at all, rather
	// than giving RenderJob a "kind" field: the queue is a full user-facing
	// feature (its own panel, describeRenderJob(), ETA sampler, recent-
	// renders list) built entirely around concepts a live session has none
	// of, and "queued but instantly started" is self-contradictory anyway -
	// a live preview queued behind a real render would sit inert, the
	// opposite of what clicking the button asked for.
	if (isLiveMode()) {
		startLivePreview();
		return;
	}
#endif

	// Always enqueue, then start the front of the queue if nothing is
	// currently running - the everyday single-render case is just "enqueue
	// one job into an empty, immediately-idle queue", so there is no
	// separate "start immediately" branch whose behaviour could drift from
	// the queued path. See processQueueIfIdle()'s own comment for what
	// happens after a render this triggers actually finishes.
	m_queueModel->add(captureRenderJob());
	refreshQueuePanel();
	processQueueIfIdle();
}

RenderJob MainWindow::captureRenderJob() {
	RenderJob job;

	// ========================================================================
	// Collect Render Parameters
	// ========================================================================

	// Render mode: GPU (true) or CPU (false)
	job.useGPU = m_renderModeCombo->currentData().toBool();
	// GPU backend: recursive (false, default) or wavefront (true). Meaningless
	// under CPU, so only honored when useGPU is also true. m_gpuBackendCombo
	// is always constructed now (see mainwindow_tabs.cpp's Renderer combo
	// setup, kGpuOptionAvailable), but on a build with no GPU support at
	// all job.useGPU is already always false (the GPU Renderer item is
	// disabled/unreachable) and the short-circuit below never reaches this
	// combo - the null check just stays as cheap, harmless defensiveness.
	// Also gated on isEnabled(), same reasoning as denoise/optixValidate/etc below -
	// updateRenderOptionsEnabled() now disables m_gpuBackendCombo under any
	// non-Default integrator (e.g. SPPM+GPU, the one alternate integrator
	// that still allows GPU), and Qt doesn't clear a disabled combo's
	// selection, so an earlier Wavefront choice would otherwise leak through
	// as `--wavefront` under an integrator that never asked for it.
	job.useWavefront = job.useGPU && m_gpuBackendCombo && m_gpuBackendCombo->isEnabled() && m_gpuBackendCombo->currentData().toBool();

	// Render Options tab - see AdvancedRenderFlags's own comment.
	// sampler/tonemap use currentData() (empty for the "default" item)
	// rather than currentText(), matching every other flag combo in this
	// file (e.g. job.useWavefront above).
	//
	// denoise/optixValidate/sampler/spectral/exposure/tonemap/stats are all
	// gated on isEnabled() (see updateRenderOptionsEnabled(),
	// mainwindow_tabs.cpp, for what disables each): Qt does not clear a
	// checkbox's checked state (or a combo's selection, or a spinbox's
	// value) just because setEnabled(false) grayed it out, so a value set
	// before a backend/integrator switch would otherwise survive into the
	// CLI invocation even though the control now shows as inactive.
	job.advancedFlags.denoise = m_denoiseCheck->isEnabled() && m_denoiseCheck->isChecked();
	job.advancedFlags.denoiseBlend = m_denoiseBlendSpin->isEnabled() ? m_denoiseBlendSpin->value() : 0.0;
	job.advancedFlags.stats = m_statsCheck->isEnabled() && m_statsCheck->isChecked();
	job.advancedFlags.aovs = m_aovsCheck && m_aovsCheck->isEnabled() && m_aovsCheck->isChecked();
	job.advancedFlags.optixValidate = m_optixValidateCheck->isEnabled() && m_optixValidateCheck->isChecked();
	job.advancedFlags.exposure = m_exposureSpin->isEnabled() ? m_exposureSpin->value() : 1.0;
	job.advancedFlags.sampler = m_samplerCombo->isEnabled() ? m_samplerCombo->currentData().toString() : QString();
	job.advancedFlags.lightSampler = m_lightSamplerCombo->isEnabled() ? m_lightSamplerCombo->currentData().toString() : QString();
	job.advancedFlags.accelerator = m_acceleratorCombo->isEnabled() ? m_acceleratorCombo->currentData().toString() : QString();
	job.advancedFlags.splitMethod = m_splitMethodCombo->isEnabled() ? m_splitMethodCombo->currentData().toString() : QString();
	job.advancedFlags.adaptiveSampling = m_adaptiveSamplingCheck->isEnabled() && m_adaptiveSamplingCheck->isChecked();
	job.advancedFlags.adaptiveThreshold = m_adaptiveThresholdSpin->value();
	job.advancedFlags.timeLimitSeconds =
		(m_timeLimitCheck->isEnabled() && m_timeLimitCheck->isChecked()) ? m_timeLimitSpin->value() : 0.0;
	job.advancedFlags.spectral = m_spectralCheck->isEnabled() && m_spectralCheck->isChecked();
	job.advancedFlags.tonemap = m_tonemapCombo->isEnabled() ? m_tonemapCombo->currentData().toString() : QString();
	job.advancedFlags.regularize = m_regularizeCheck->isEnabled() && m_regularizeCheck->isChecked();
	job.advancedFlags.maxComponentValue =
		(m_maxComponentValueCheck->isEnabled() && m_maxComponentValueCheck->isChecked())
			? m_maxComponentValueSpin->value() : 0.0;
	job.advancedFlags.cropEnabled = m_cropCheck->isEnabled() && m_cropCheck->isChecked();
	job.advancedFlags.cropX0 = m_cropX0Spin->value();
	job.advancedFlags.cropY0 = m_cropY0Spin->value();
	job.advancedFlags.cropX1 = m_cropX1Spin->value();
	job.advancedFlags.cropY1 = m_cropY1Spin->value();
	job.advancedFlags.dofOverrideEnabled = m_dofOverrideCheck->isEnabled() && m_dofOverrideCheck->isChecked();
	job.advancedFlags.apertureOverride = m_apertureSpin->value();
	job.advancedFlags.focusDistanceOverride = m_focusDistanceSpin->value();
	job.advancedFlags.seed =
		(m_seedCheck->isEnabled() && m_seedCheck->isChecked())
			? static_cast<long long>(m_seedSpin->value()) : -1;

	// Integrator combo + its sub-flags, both in the "Integrator" group on
	// the Render Options tab - see
	// IntegratorOptions's own comment (mainwindow.h). No isEnabled()
	// gating needed on the sub-flag widgets themselves: only the
	// currently-selected integrator's own fields are ever read by
	// RenderController::start()'s switch, so a stale value from a hidden
	// stack page is never emitted regardless.
	job.integratorOptions.mode = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	job.integratorOptions.sppmIterations = m_sppmIterationsSpin->value();
	job.integratorOptions.sppmPhotons = m_sppmPhotonsSpin->value();
	job.integratorOptions.bdptMaxDepth = m_bdptMaxDepthSpin->value();
	job.integratorOptions.mltBootstrap = m_mltBootstrapSpin->value();
	job.integratorOptions.mltMutations = static_cast<long long>(m_mltMutationsSpin->value());
	job.integratorOptions.mltMaxDepth = m_mltMaxDepthSpin->value();
	job.integratorOptions.aoMaxDist = m_aoMaxDistSpin->value();
	job.integratorOptions.aoUniform = m_aoUniformCheck->isChecked();
	job.integratorOptions.aoIllumScale = m_aoIllumScaleSpin->value();
	job.integratorOptions.aoIllumR = m_aoIllumRSpin->value();
	job.integratorOptions.aoIllumG = m_aoIllumGSpin->value();
	job.integratorOptions.aoIllumB = m_aoIllumBSpin->value();
	job.integratorOptions.simplepathNoLights = m_simplepathNoLightsCheck->isChecked();
	job.integratorOptions.simplepathNoBsdf = m_simplepathNoBsdfCheck->isChecked();

	// Resolution: either from preset dropdown or the manual Advanced Parameters fields
	if (m_qualityPresetCombo->currentIndex() == 6) {
		// Custom quality preset - use manual width/height from Advanced Parameters
		job.width = m_widthSpinBox->value();
		job.height = m_heightSpinBox->value();
	} else {
		// Standard quality preset - use resolution from dropdown
		QSize res = m_resolutionCombo->currentData().toSize();
		job.width = res.width();
		job.height = res.height();
	}

	// Ray tracing quality parameters
	job.samples = m_samplesSpinBox->value();    // Samples per pixel (higher = smoother but slower)
	job.maxDepth = m_maxDepthSpinBox->value();  // Max ray bounce depth (higher = more realistic lighting)

	// Camera position (lookfrom) - read from spinboxes
	// These reflect either the selected preset or custom user input
	job.sceneId = m_sceneCombo->currentData().toString();
	// One fetch for name/description below AND the recommended-camera
	// comparison further down, instead of three separate scene_metadata.dll
	// round-trips (each its own find_scene() scan) for the same sceneId.
	SceneMetadataClient::SceneMetadata jobMeta;
	const bool jobMetaFound = SceneMetadataClient::sceneMetadata(job.sceneId, jobMeta);
	job.displayTitle = jobMeta.name;
	if (job.displayTitle.isEmpty())
		job.displayTitle = job.sceneId.isEmpty() ? QStringLiteral("Render") : job.sceneId;
	job.sceneDescription = jobMeta.description;

	// Output file path - the field's own placeholder text promises a
	// timestamp "to avoid overwriting", but that timestamp was only ever
	// generated once, when the field was created at app startup - every
	// render in the same session reused that same name. Refreshed here
	// instead, on every capture (including ones that only get queued rather
	// than started immediately - each still needs its own unique path),
	// keeping whatever directory the user chose (via Browse) but replacing
	// the filename - so each render gets its own file and an earlier
	// render's Preview sub-tab (see addImagePreviewTab/addVideoPreviewTab)
	// still has something real to point "Open Folder"/"Open Viewer" at
	// after a later render completes. Video mode reuses this same field
	// rather than its own fixed "output/video.ppm", for the same reason.
	{
		QFileInfo prevInfo(m_outputPathEdit->text());
		QString dir = prevInfo.absolutePath();
		QString ext = isVideoMode() ? "ppm" : (prevInfo.suffix().isEmpty() ? "png" : prevInfo.suffix());
		QString base = isVideoMode() ? "video" : "render";
		QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz");
		// The scene's slug, not its id: it stays the same scene when the registry changes, and has no underscore to confuse recent_renders.cpp's parsing.
		QString newName = QString("%1_%2_%3.%4").arg(base, SceneMetadataClient::sceneSlug(job.sceneId), timestamp, ext);
		m_outputPathEdit->setText(QDir::toNativeSeparators(dir + "/" + newName));
	}
	job.outputPath = m_outputPathEdit->text();
	job.camX = m_cameraPosX->value();
	job.camY = m_cameraPosY->value();
	job.camZ = m_cameraPosZ->value();

	// Only treat the camera as "explicit" (see RenderController::setParameters's
	// comment) if it actually differs from this scene's own recommended
	// camera - queried live, same as onSceneChanged. If the query fails,
	// default to explicit: the worse outcome is an unnecessary (but
	// harmless, since it'd be the same value anyway) cam_x/y/z on the
	// command line, not a silently wrong camera.
	job.camExplicit = true;
	if (jobMetaFound) {
		// m_cameraPosX/Y/Z are QDoubleSpinBoxes with the default 2 decimal
		// places, so a recommended value round-trips through setValue()/
		// value() rounded to the nearest 0.01 - the epsilon has to be
		// looser than that (half a step) or a scene whose recommended
		// camera ever needs more precision than 2 decimals would silently
		// never compare equal, permanently forcing camExplicit=true (still
		// harmless, just pointlessly defeats this check for that scene).
		constexpr double kEpsilon = 0.005;
		job.camExplicit = std::abs(job.camX - jobMeta.camLookfromX) > kEpsilon
			|| std::abs(job.camY - jobMeta.camLookfromY) > kEpsilon
			|| std::abs(job.camZ - jobMeta.camLookfromZ) > kEpsilon;
	}

	job.videoMode = isVideoMode();
	if (isVideoMode()) {
		job.videoFrames = m_videoFramesSpinBox->value();
		job.videoFPS = m_videoFPSSpinBox->value();
		job.videoSpeed = m_videoSpeedSpinBox->value();
		job.cameraPath = m_cameraPathCombo->currentData().toString();

		// Named scene+path+frames/fps/speed bundle, if one is selected - see
		// video_preset.h. Captured now (rather than re-read by
		// assembleVideoAutomatically() later) since that runs on a ~500ms
		// deferred timer, by which point the user may already have changed
		// this combo for a different queued job.
		if (m_videoPresetCombo && m_videoPresetCombo->currentIndex() > 0) {
			const QString id = m_videoPresetCombo->currentData().toString();
			if (const video_preset::VideoPreset *preset = video_preset::find(id.toUtf8().constData()))
				job.videoPresetName = QString::fromUtf8(preset->name);
		}
	}

	return job;
}

void MainWindow::startRenderJob(const RenderJob &job) {
	// Recorded before anything else so onRenderComplete() - which fires
	// asynchronously, possibly after the user has already changed the scene/
	// mode combo for a job queued behind this one - describes the job that
	// actually ran, not whatever the form happens to show by then.
	m_currentJob = job;
	m_currentJobLabel->setText(describeRenderJob(job));

	// ========================================================================
	// Launch Render
	// ========================================================================
	// RenderController spawns ray_tracer.exe as a subprocess with all
	// parameters and reports its output back via signals (no worker thread -
	// QProcess is already asynchronous). The executable will call either the
	// CPU or GPU renderer based on the useGPU flag.
	m_renderController = new RenderController(this);
	m_renderController->setParameters(job.useGPU, job.width, job.height, job.samples, job.maxDepth,
	                                   job.sceneId, job.camX, job.camY, job.camZ, job.camExplicit,
	                                   job.outputPath, job.useWavefront);
	m_renderController->setAdvancedFlags(job.advancedFlags);
	m_renderController->setIntegratorOptions(job.integratorOptions);

	if (job.videoMode) {
		m_renderController->setVideoParameters(true, job.videoFrames, job.videoFPS, job.cameraPath, job.videoSpeed);
	} else {
		m_renderController->setVideoParameters(false, 0, 0, "", 1.0);
	}

	connect(m_renderController, &RenderController::progressUpdate, this, &MainWindow::onProgressUpdate);
	connect(m_renderController, &RenderController::renderComplete, this, &MainWindow::onRenderComplete);
	connect(m_renderController, &RenderController::logMessage, this, &MainWindow::onLogMessage);
	connect(m_renderController, &RenderController::pauseStateChanged, this, &MainWindow::onControllerPauseStateChanged);

	// The controller is done once it reports completion; drop it so
	// m_renderController is only non-null while a render is actually active.
	// Captures the controller by value rather than reading the m_renderController
	// member at delete time: onRenderComplete (connected above, so it runs first)
	// can synchronously start the next queued job before this lambda runs, which
	// would otherwise repoint m_renderController at the new job's controller and
	// make this delete the wrong (brand new, still-running) instance out from
	// under it.
	RenderController *controllerToRetire = m_renderController;
	connect(m_renderController, &RenderController::renderComplete, this, [this, controllerToRetire]() {
		if (m_renderController == controllerToRetire) m_renderController = nullptr;
		controllerToRetire->deleteLater();
	});

	m_isRendering = true;
	// m_renderButton stays enabled (unlike m_stopButton) - it's still valid
	// to click while a render is running, since doing so now just queues
	// another job instead of starting one immediately. (Not true in Live
	// Preview mode, which has no queue - updateTransportButtons() below
	// overrides this when that's the currently-selected mode, e.g. if this
	// job was queued while Image/Video was selected and only started
	// executing after the user switched to Live Preview.)
	m_stopButton->setEnabled(true);
	m_pauseButton->setEnabled(true);
	m_abandonButton->setEnabled(true);
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	updateTransportButtons();
#endif
	// Always starts in "Pause" state regardless of how the previous job
	// ended - a paused job is never left running unattended when
	// onRenderComplete() fires (it only fires once the process has actually
	// exited), so there is no leftover "Resume" state to inherit.
	m_pauseButton->setText(tr("&PAUSE RENDER"));
	icon_tint::apply(m_pauseButton, ":/icons/pause.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	if (m_actPause) {
		m_actPause->setText(tr("&Pause Render"));
		icon_tint::apply(m_actPause, ":/icons/pause.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	}
	m_pauseStartedAt = QDateTime();
	updateActionStates();
	refreshStatusBarInfo();
	m_progressBar->setValue(0);
	startProgressGlow();
	QString statusText = job.videoMode ? tr("Rendering video frames...") : tr("Rendering...");
	if (m_queueModel->hasWaiting()) statusText += tr(" (%1 more queued)").arg(m_queueModel->waitingCount());
	m_statusLabel->setText(statusText);

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

	// Auto-switch to the Progress tab so the user sees render status
	// immediately, without stealing focus from whichever results tab
	// (Preview/Log) they may already be looking at from an earlier job.
	if (m_progressTabIndex >= 0) m_tabWidget->setCurrentIndex(m_progressTabIndex);

	m_renderController->start();
}

void MainWindow::processQueueIfIdle() {
	if (m_isRendering || !m_queueModel->hasWaiting()) return;
	RenderJob job;
	int id = 0;
	if (!m_queueModel->takeNext(id, job)) return;
	m_currentQueueId = id;
	refreshQueuePanel();
	startRenderJob(job);
}

QString MainWindow::integratorSuffixTag(IntegratorMode mode) {
	switch (mode) {
		case IntegratorMode::Default: return QString();
		case IntegratorMode::Sppm: return tr(" · SPPM");
		case IntegratorMode::Bdpt: return tr(" · BDPT");
		case IntegratorMode::Mlt: return tr(" · MLT");
		case IntegratorMode::RandomWalk: return tr(" · RandomWalk");
		case IntegratorMode::Ao: return tr(" · AO");
		case IntegratorMode::SimplePath: return tr(" · SimplePath");
		case IntegratorMode::SimpleVolPath: return tr(" · SimpleVolPath");
		case IntegratorMode::LightPath: return tr(" · LightPath");
	}
	return QString();
}

QString MainWindow::rendererLabel(bool useGPU, bool useWavefront) {
	return useGPU ? (useWavefront ? tr("GPU-WF") : tr("GPU")) : tr("CPU");
}

QString MainWindow::describeRenderJob(const RenderJob &job) {
	const QString renderer = rendererLabel(job.useGPU, job.useWavefront);
	const QString modeSuffix = job.videoMode ? tr(" · Video (%1f)").arg(job.videoFrames) : QString();
	// A queued/current job's integrator materially changes both algorithm
	// and render time - worth a short tag here even though it's blank for
	// the common (Default) case, same as modeSuffix above being blank
	// outside Video mode.
	const QString integratorSuffix = integratorSuffixTag(job.integratorOptions.mode);
	return tr("%1 — %2×%3 · %4spp · %5%6%7")
		.arg(job.displayTitle)
		.arg(job.width).arg(job.height)
		.arg(job.samples)
		.arg(renderer, modeSuffix, integratorSuffix);
}

void MainWindow::refreshQueuePanel() {
	if (!m_queueGroup || !m_queueModel) return;
	m_queueGroup->setTitle(m_queueModel->hasWaiting() ? tr("Render Queue (%1 waiting)").arg(m_queueModel->waitingCount()) : tr("Render Queue"));
	const bool empty = m_queueModel->rowCount() == 0;
	if (m_queueView) m_queueView->setVisible(!empty);
	if (m_queueEmptyLabel) m_queueEmptyLabel->setVisible(empty);
	updateQueueButtons();
}

// Each button is available only where it means something: a running job is stopped with Stop, not removed; only waiting jobs move; only a failed or
// cancelled one can be run again.
void MainWindow::updateQueueButtons() {
	if (!m_queueView || !m_queueModel) return;
	const QModelIndex current = m_queueView->currentIndex();
	const int row = current.isValid() ? current.row() : -1;
	const job_queue::State state = m_queueModel->stateAt(row);
	const bool has = row >= 0;
	m_queueRemoveButton->setEnabled(has && state != job_queue::State::Running);
	m_queueUpButton->setEnabled(has && state == job_queue::State::Waiting);
	m_queueDownButton->setEnabled(has && state == job_queue::State::Waiting);
	m_queueRetryButton->setEnabled(has && (state == job_queue::State::Failed || state == job_queue::State::Cancelled));
	m_queueClearFinishedButton->setEnabled(m_queueModel->finishedCount() > 0);
	m_queueClearButton->setEnabled(m_queueModel->hasWaiting());
}

void MainWindow::onRemoveSelectedQueueItem() {
	if (!m_queueView) return;
	const int id = m_queueModel->idAt(m_queueView->currentIndex().row());
	if (id && m_queueModel->remove(id)) refreshQueuePanel();
}

void MainWindow::onMoveQueueItemUp() {
	const int id = m_queueModel->idAt(m_queueView->currentIndex().row());
	if (id && m_queueModel->moveUp(id)) m_queueView->selectRow(m_queueModel->queue().rowOf(id));
	refreshQueuePanel();
}

void MainWindow::onMoveQueueItemDown() {
	const int id = m_queueModel->idAt(m_queueView->currentIndex().row());
	if (id && m_queueModel->moveDown(id)) m_queueView->selectRow(m_queueModel->queue().rowOf(id));
	refreshQueuePanel();
}

void MainWindow::onRetryQueueItem() {
	const int id = m_queueModel->idAt(m_queueView->currentIndex().row());
	if (!id || !m_queueModel->retry(id)) return;
	refreshQueuePanel();
	processQueueIfIdle();   // nothing running: it starts at once, like a new render would
}

void MainWindow::onClearFinishedJobs() {
	m_queueModel->clearFinished();
	refreshQueuePanel();
}

void MainWindow::onClearQueue() {
	// The one destructive, irreversible action in this app with no undo -
	// worth a confirmation given "Clear Queue" sits right next to "Remove
	// Selected" in the same row (mainwindow_tabs.cpp) and a misclick would
	// silently discard every queued job's configuration. Skipped when nothing
	// is waiting - nothing destructive to confirm. Finished jobs and the one
	// that is rendering are not touched (Clear Finished and Stop do those).
	if (!m_queueModel->hasWaiting()) return;
	const auto choice = QMessageBox::question(this, tr("Clear Render Queue"),
		tr("Remove all %n queued render(s)? This can't be undone.", "", m_queueModel->waitingCount()),
		QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (choice != QMessageBox::Yes) return;

	m_queueModel->clearWaiting();
	refreshQueuePanel();
}

