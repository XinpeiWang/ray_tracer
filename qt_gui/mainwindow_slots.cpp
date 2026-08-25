#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include "camera_math.h"
#include "../src/shared/video_preset.h"
#include "../src/shared/scene_descriptor.h"
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

// Styles one line of the --diagnose report (see launcher/diagnostics.cpp,
// which emits exactly one "Key: value" fact per line for this to key off).
// Unlike styleLogLine() above, a diagnostics line carries no timestamp or
// [LABEL] - it's a plain fact, so only two things vary: the "===" banner
// rule, and whether the fact reports a problem (missing/unavailable/not
// writable), a healthy state (available/writable/present), or a neutral
// measurement (a count, a size, a version string).
QString styleDiagnosticsLine(const theme::Palette &p, const QString &line) {
	const QString mono = "font-family:Consolas,monospace;font-size:9pt;";

	if (line.startsWith("===")) {
		return QString("<span style='color:%1;%2'><b>%3</b></span>")
			.arg(p.logSeparator.name(), mono, line.toHtmlEscaped());
	}

	int colon = line.indexOf(':');
	if (colon < 0) {
		return QString("<span style='color:%1;%2'>%3</span>")
			.arg(p.textBody.name(), mono, line.toHtmlEscaped());
	}

	// Classified on the WHOLE line, not just the value: a fact like "Pbrt
	// Scene Files Missing: a.pbrt, b.pbrt" carries its problem-word in the
	// key, not the value. Order matters within the check itself - the
	// negative phrasings ("not available", "NOT writable") must be tested
	// before the positive ones they contain as a substring ("available",
	// "writable"), or a missing GPU would render green.
	const QString lower = line.toLower();
	QColor factColour;
	if (lower.contains("not available") || lower.contains("not detected") ||
		lower.contains("not usable") || lower.contains("not writable") ||
		lower.contains("missing")) {
		factColour = p.logWarning;
	} else if (lower.contains("failed")) {
		factColour = p.logError;
	} else if (lower.contains("available") || lower.contains("writable") ||
			   lower.contains("present")) {
		factColour = p.logSuccess;
	} else {
		factColour = p.textBody;
	}

	QString key = line.left(colon).toHtmlEscaped();
	QString value = line.mid(colon + 1).toHtmlEscaped();
	// The key stays a muted label (like a log line's [LABEL] tag) so the
	// coloured value is what draws the eye; a neutral fact keeps the value
	// in the same muted-adjacent body colour instead of standing out.
	return QString("<span style='%1'><span style='color:%2;'>%3:</span>"
				   "<span style='color:%4;'>%5</span></span>")
		.arg(mono, p.textMuted.name(), key, factColour.name(), value);
}

} // namespace

void MainWindow::onRenderClicked() {
	// m_sceneCombo can legitimately be empty - a search term that matches
	// nothing in the current category tab leaves it with no items and
	// currentIndex()==-1 (see onSceneChanged()'s own comment on this case) -
	// and the button itself is never disabled for that state. Without this
	// guard, captureRenderJob() below would read an empty scene id and
	// enqueue a render with no scene specified at all.
	if (!m_sceneCombo || m_sceneCombo->currentIndex() < 0) {
		setStatusWarning("Can't start a render - no scene is selected (try clearing the search box).");
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

	// Always enqueue, then start the front of the queue if nothing is
	// currently running - the everyday single-render case is just "enqueue
	// one job into an empty, immediately-idle queue", so there is no
	// separate "start immediately" branch whose behaviour could drift from
	// the queued path. See processQueueIfIdle()'s own comment for what
	// happens after a render this triggers actually finishes.
	m_renderQueue.enqueue(captureRenderJob());
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
	// is nullptr on a build with no GPU support at all (RT_GUI_HAVE_GPU
	// undefined), in which case job.useGPU is already always false and the
	// short-circuit below never reaches it - the explicit null check just
	// makes that safety non-fragile against future reordering.
	job.useWavefront = job.useGPU && m_gpuBackendCombo && m_gpuBackendCombo->currentData().toBool();

	// Render Options tab - see AdvancedRenderFlags's own comment.
	// sampler/tonemap use currentData() (empty for the "default" item)
	// rather than currentText(), matching every other flag combo in this
	// file (e.g. job.useWavefront above).
	//
	// denoise/optixValidate/sampler/spectral are gated on isEnabled(): Qt
	// does not clear a checkbox's checked state (or a combo's selection)
	// just because setEnabled(false) grayed it out, so a value checked
	// before a backend switch would otherwise survive into the CLI
	// invocation even though the control now shows as inactive.
	job.advancedFlags.denoise = m_denoiseCheck->isEnabled() && m_denoiseCheck->isChecked();
	job.advancedFlags.stats = m_statsCheck->isChecked();
	job.advancedFlags.optixValidate = m_optixValidateCheck->isEnabled() && m_optixValidateCheck->isChecked();
	job.advancedFlags.exposure = m_exposureSpin->value();
	job.advancedFlags.sampler = m_samplerCombo->isEnabled() ? m_samplerCombo->currentData().toString() : QString();
	job.advancedFlags.spectral = m_spectralCheck->isEnabled() && m_spectralCheck->isChecked();
	job.advancedFlags.tonemap = m_tonemapCombo->currentData().toString();

	// Resolution: either from preset dropdown or custom values from Advanced tab
	if (m_qualityPresetCombo->currentIndex() == 6) {
		// Custom quality preset - use manual width/height from Advanced tab
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
	job.displayTitle = SceneMetadataClient::sceneName(job.sceneId);
	if (job.displayTitle.isEmpty())
		job.displayTitle = job.sceneId.isEmpty() ? QStringLiteral("Render") : job.sceneId;
	job.sceneDescription = SceneMetadataClient::sceneDescription(job.sceneId);

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
		QString ext = m_videoMode ? "ppm" : (prevInfo.suffix().isEmpty() ? "png" : prevInfo.suffix());
		QString base = m_videoMode ? "video" : "render";
		QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz");
		QString newName = QString("%1_%2_%3.%4").arg(base, job.sceneId, timestamp, ext);
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
	double recCamX = 0.0, recCamY = 0.0, recCamZ = 0.0, recLookatX = 0.0, recLookatY = 0.0, recLookatZ = 0.0;
	if (SceneMetadataClient::recommendedCamera(job.sceneId, recCamX, recCamY, recCamZ, recLookatX, recLookatY, recLookatZ)) {
		// m_cameraPosX/Y/Z are QDoubleSpinBoxes with the default 2 decimal
		// places, so a recommended value round-trips through setValue()/
		// value() rounded to the nearest 0.01 - the epsilon has to be
		// looser than that (half a step) or a scene whose recommended
		// camera ever needs more precision than 2 decimals would silently
		// never compare equal, permanently forcing camExplicit=true (still
		// harmless, just pointlessly defeats this check for that scene).
		constexpr double kEpsilon = 0.005;
		job.camExplicit = std::abs(job.camX - recCamX) > kEpsilon
			|| std::abs(job.camY - recCamY) > kEpsilon
			|| std::abs(job.camZ - recCamZ) > kEpsilon;
	}

	job.videoMode = m_videoMode;
	if (m_videoMode) {
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

	if (job.videoMode) {
		m_renderController->setVideoParameters(true, job.videoFrames, job.videoFPS, job.cameraPath, job.videoSpeed);
	} else {
		m_renderController->setVideoParameters(false, 0, 0, "", 1.0);
	}

	connect(m_renderController, &RenderController::progressUpdate, this, &MainWindow::onProgressUpdate);
	connect(m_renderController, &RenderController::renderComplete, this, &MainWindow::onRenderComplete);
	connect(m_renderController, &RenderController::logMessage, this, &MainWindow::onLogMessage);

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
	// another job instead of starting one immediately.
	m_stopButton->setEnabled(true);
	updateActionStates();
	refreshStatusBarInfo();
	m_progressBar->setValue(0);
	startProgressGlow();
	QString statusText = job.videoMode ? "Rendering video frames..." : "Rendering...";
	if (!m_renderQueue.isEmpty()) statusText += QString(" (%1 more queued)").arg(m_renderQueue.size());
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
	if (m_isRendering || m_renderQueue.isEmpty()) return;
	RenderJob job = m_renderQueue.dequeue();
	refreshQueuePanel();
	startRenderJob(job);
}

QString MainWindow::describeRenderJob(const RenderJob &job) {
	const QString renderer = job.useGPU ? (job.useWavefront ? "GPU-WF" : "GPU") : "CPU";
	const QString modeSuffix = job.videoMode ? QString(" · Video (%1f)").arg(job.videoFrames) : QString();
	return QString("%1 — %2×%3 · %4spp · %5%6")
		.arg(job.displayTitle)
		.arg(job.width).arg(job.height)
		.arg(job.samples)
		.arg(renderer, modeSuffix);
}

void MainWindow::refreshQueuePanel() {
	if (!m_queueGroup || !m_queueListWidget) return;
	m_queueListWidget->clear();
	for (const RenderJob &job : m_renderQueue) {
		m_queueListWidget->addItem(describeRenderJob(job));
	}
	m_queueGroup->setTitle(QString("Render Queue (%1)").arg(m_renderQueue.size()));
}

void MainWindow::onRemoveSelectedQueueItem() {
	if (!m_queueListWidget) return;
	const int row = m_queueListWidget->currentRow();
	if (row < 0 || row >= m_renderQueue.size()) return;
	m_renderQueue.removeAt(row);
	refreshQueuePanel();
}

void MainWindow::onClearQueue() {
	// The one destructive, irreversible action in this app with no undo -
	// worth a confirmation given "Clear Queue" sits right next to "Remove
	// Selected" in the same row (mainwindow_tabs.cpp) and a misclick would
	// silently discard every queued job's configuration. Skipped when the
	// queue is already empty - nothing destructive to confirm.
	if (m_renderQueue.isEmpty()) return;
	const auto choice = QMessageBox::question(this, "Clear Render Queue",
		QString("Remove all %1 queued render%2? This can't be undone.")
			.arg(m_renderQueue.size()).arg(m_renderQueue.size() == 1 ? "" : "s"),
		QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (choice != QMessageBox::Yes) return;

	m_renderQueue.clear();
	refreshQueuePanel();
}

void MainWindow::onRunDiagnosticsClicked() {
	// A stray click while one is already running would leak a second
	// QProcess and race both sets of signals into the same text edit.
	if (m_diagnosticsRunner) return;

	m_lastDiagReport.clear();  // no report to recolour until reportReady fires
	if (m_diagTextEdit) {
		m_diagTextEdit->clear();
		m_diagTextEdit->setPlainText("Running diagnostics...");
	}
	if (m_runDiagnosticsButton) m_runDiagnosticsButton->setEnabled(false);

	m_diagnosticsRunner = new DiagnosticsRunner(this);
	connect(m_diagnosticsRunner, &DiagnosticsRunner::reportReady,
			this, &MainWindow::onDiagnosticsReportReady);
	connect(m_diagnosticsRunner, &DiagnosticsRunner::reportFailed,
			this, &MainWindow::onDiagnosticsFailed);

	// Same "retire by captured value" reasoning as RenderController's own
	// cleanup lambda above (mainwindow.cpp) - reportReady/reportFailed have
	// already run by the time either lambda below fires (Qt delivers queued
	// connections in connection order), so m_diagnosticsRunner is only
	// nulled out if it's still pointing at the instance that just finished.
	DiagnosticsRunner *runnerToRetire = m_diagnosticsRunner;
	auto retire = [this, runnerToRetire]() {
		if (m_diagnosticsRunner == runnerToRetire) m_diagnosticsRunner = nullptr;
		if (m_runDiagnosticsButton) m_runDiagnosticsButton->setEnabled(true);
		runnerToRetire->deleteLater();
	};
	connect(m_diagnosticsRunner, &DiagnosticsRunner::reportReady, this, retire);
	connect(m_diagnosticsRunner, &DiagnosticsRunner::reportFailed, this, retire);

	m_diagnosticsRunner->start();
}

void MainWindow::onDiagnosticsReportReady(const QString &report) {
	m_lastDiagReport = report;
	rebuildDiagPane();
}

void MainWindow::onDiagnosticsFailed(const QString &message) {
	// Not a report - plain text, nothing to colour, and clearing
	// m_lastDiagReport keeps a later theme change from trying to recolour
	// a report that isn't showing anymore.
	m_lastDiagReport.clear();
	if (m_diagTextEdit) m_diagTextEdit->setPlainText("Diagnostics failed:\n\n" + message);
}

// Fills in m_sceneGrid's preview tiles for the curated, self-contained,
// fast-rendering subset (Basics/Materials/Cameras) - see the scene-gallery
// plan's phased-coverage decision for why the rest of the ~154-scene
// registry isn't covered yet. Disabled (see createBasicTab()'s button
// tooltip) while a real render is in flight so thumbnail generation can
// never compete with the user's own queued work - m_thumbnailGenerator owns
// a private RenderController instead of reusing m_renderController/
// m_renderQueue precisely so it never needs to cooperate with those at all,
// only avoid running alongside them.
void MainWindow::onGenerateThumbnailsClicked() {
	if (m_isRendering || !m_renderQueue.isEmpty()) {
		setStatusWarning("Can't generate thumbnails while a render is in progress or queued.");
		return;
	}
	if (m_thumbnailGenerator && m_thumbnailGenerator->isRunning()) return;

	if (!m_thumbnailGenerator) {
		m_thumbnailGenerator = new ThumbnailGenerator(this);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::thumbnailReady,
				this, &MainWindow::onThumbnailReady);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::allDone,
				this, &MainWindow::onThumbnailsAllDone);
	}

	static const QStringList kThumbnailCategories = {
		SceneCategories::Basics, SceneCategories::Materials, SceneCategories::Cameras
	};
	QStringList ids;
	const int count = SceneMetadataClient::sceneCount();
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		if (SceneMetadataClient::sceneRequiresFiles(id)) continue;
		if (!kThumbnailCategories.contains(SceneMetadataClient::sceneCategory(id))) continue;
		ids << id;
	}

	if (m_generateThumbnailsButton) m_generateThumbnailsButton->setEnabled(false);
	onLogMessage(QString("Generating thumbnails for up to %1 scene(s)...").arg(ids.size()));
	m_thumbnailGenerator->start(ids, [this](const QString &id) { return thumbnailCachePath(id); });
}

void MainWindow::onThumbnailReady(const QString &sceneId, bool success, const QString &outputPath) {
	if (!success) {
		onLogMessage(QString("Thumbnail generation failed for scene %1").arg(sceneId));
		return;
	}
	if (!m_sceneGrid) return;
	for (int i = 0; i < m_sceneGrid->count(); ++i) {
		QListWidgetItem *item = m_sceneGrid->item(i);
		if (item->data(Qt::UserRole).toString() == sceneId) {
			item->setIcon(QIcon(outputPath));
			break;
		}
	}
}

void MainWindow::onThumbnailsAllDone() {
	if (m_generateThumbnailsButton) m_generateThumbnailsButton->setEnabled(true);
	onLogMessage("Thumbnail generation finished.");
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
// MainWindow::onVideoPresetChanged
// ============================================================================
// Called when the user picks a named bundle from the Video tab's Preset
// combo (see video_preset.h and createVideoTab()'s own setup comment).
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
	if (m_modeCombo && m_modeCombo->currentIndex() != 1)
		m_modeCombo->setCurrentIndex(1);

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

void MainWindow::refreshSceneInfoLabel() {
	if (!m_sceneCombo || !m_sceneInfoLabel) return;
	const int index = m_sceneCombo->currentIndex();
	if (index < 0) return;
	const QString scene_id = m_sceneCombo->itemData(index).toString();
	const QString description = SceneMetadataClient::sceneDescription(scene_id);
	if (description.isEmpty()) return;

	bool gpuSupported = true;
	SceneMetadataClient::gpuCompatible(scene_id, gpuSupported);

	QString infoText = QString("<b>Description:</b> %1<br>").arg(description);
	infoText += QString("<b>Performance:</b> %1<br>").arg(SceneMetadataClient::scenePerformance(scene_id));
	infoText += QString("<b>Recommended SPP:</b> %1<br>").arg(SceneMetadataClient::sceneRecommendedSpp(scene_id));
	infoText += QString("<b>GPU Support:</b> %1<br>").arg(gpuSupported ? "Yes" : "CPU only");
	// These two warnings are the only coloured text in the label, so they take
	// their colours from the theme's log severities rather than fixed hex - a
	// gold-on-cream warning is unreadable on the light schemes. Rebuilt fresh
	// on every call (rather than cached) specifically so restyleThemedWidgets()
	// calling this on a theme switch picks up the new theme's colours instead
	// of leaving an already-shown badge stuck in the old one.
	if (SceneMetadataClient::sceneRequiresFiles(scene_id))
		infoText += QString("<br><b style='color: %1;'>&#9888; Requires external files</b>")
			.arg(m_activeTheme.logWarning.name());
	if (!gpuSupported)
		infoText += QString("<br><b style='color: %1;'>&#9888; CPU renderer only</b>")
			.arg(m_activeTheme.logError.name());
	m_sceneInfoLabel->setText(infoText);
}

void MainWindow::onSceneChanged(int index) {
	if (index < 0) {
		// The only realistic way to reach this once scenes have loaded is
		// m_sceneSearchBox narrowing the current category to zero matches -
		// every category tab always holds at least one scene otherwise
		// (rebuildCategoryTabs() excludes empty categories outright).
		// Leaving the description panel showing the PREVIOUS scene made a
		// zero-match search look like nothing had happened.
		if (m_sceneInfoLabel && m_sceneCombo && m_sceneCombo->count() == 0) {
			const QString term = m_sceneSearchBox ? m_sceneSearchBox->text().trimmed() : QString();
			m_sceneInfoLabel->setText(term.isEmpty()
				? "No scenes in this category."
				: QString("No scenes match \"%1\" in this category.").arg(term));
		}
		return;
	}
	// A raw combo-row index is no longer a valid id on its own (ids are
	// category letter + number now - see scene_registry.h's
	// SceneDescriptor::id comment), so the m_sceneCombo-null fallback
	// resolves the id via the registry position instead.
	QString scene_id = m_sceneCombo ? m_sceneCombo->itemData(index).toString()
									 : SceneMetadataClient::sceneIdAtIndex(index);

	// Keeps m_sceneGrid's highlighted tile in sync with whatever scene just
	// became current, regardless of which view (combo, grid, search, tab
	// switch, selectSceneById) drove the change - every one of those paths
	// already funnels through here. Blocked so this never re-enters via the
	// grid's own currentItemChanged handler (mainwindow_tabs.cpp).
	if (m_sceneGrid) {
		const QSignalBlocker blocker(m_sceneGrid);
		bool found = false;
		for (int i = 0; i < m_sceneGrid->count(); ++i) {
			if (m_sceneGrid->item(i)->data(Qt::UserRole).toString() == scene_id) {
				m_sceneGrid->setCurrentRow(i);
				found = true;
				break;
			}
		}
		if (!found) m_sceneGrid->setCurrentRow(-1);
	}

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

	refreshSceneInfoLabel();
	// Same description text, shown in the Preview tab's sidebar too - see
	// createPreviewTab()'s own comment on why this is kept in sync here
	// rather than only refreshed on render completion.
	m_previewSceneDescLabel->setText(description);
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
	animateProgressTo(percentage);

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
	// Snapshotted once, up front: this function (indirectly, via the
	// deferred singleShot below) can end up running after m_currentJob has
	// already been overwritten by a queued next job - see m_currentJob's own
	// comment. Everything below reads finishedJob, never m_currentJob
	// directly.
	const RenderJob finishedJob = m_currentJob;

	m_isRendering = false;
	m_renderButton->setEnabled(true);
	m_stopButton->setEnabled(false);
	if (m_elapsedTimer) m_elapsedTimer->stop();
	updateActionStates();
	// Cleared unconditionally, before either branch below runs: a warning
	// left over from a previous job's preview failure must not linger next
	// to this job's own (possibly unrelated) outcome.
	clearStatusWarning();
	stopProgressGlow();

	const bool stoppedByUser = !success && message.contains("stopped by user", Qt::CaseInsensitive);

	// A failed render leaves the taskbar button red so the outcome is visible
	// without switching to the window; anything else clears it. Leaving a
	// progress state set would make it stick until the process exits.
	notifyRenderFinished(success, message, totalTime);

	if (success) {
		animateProgressTo(100);
		// A finished bar keeps its fill and turns green rather than resetting -
		// the outcome stays visible after the fact (Qt Creator's behaviour).
		setProgressResultState("success");
		m_statusLabel->setText(QString("✅ %1 - Total time: %2 seconds").arg(message).arg(totalTime, 0, 'f', 2));

		if (finishedJob.videoMode) {
			onLogMessage("Video frames rendered successfully. Starting video assembly...");
			m_statusLabel->setText("⚙️ Assembling video from frames...");

			// Trigger automatic video assembly. outputPath and finishedJob
			// are both threaded through explicitly (rather than read back
			// from m_currentJob when this fires) so assembleVideoAutomatically()
			// - which runs on a ~500ms deferred timer - still describes the
			// job that actually rendered even if a queued next job has
			// already started and overwritten m_currentJob by then. It can
			// derive the expected "<stem>_video.mp4" directly from
			// outputPath (see main.cpp's own stem-based naming) instead of
			// globbing a hardcoded directory - necessary now that each
			// render's base path is unique (see captureRenderJob()'s own
			// comment on m_outputPathEdit) rather than always landing in the
			// same app-relative "output/" folder.
			QTimer::singleShot(500, this, [this, outputPath, finishedJob]() {
				assembleVideoAutomatically(outputPath, finishedJob);
			});
		} else {
			// Image mode: show the rendered image inline as a new Preview
			// sub-tab instead of shelling out to the OS's default image
			// viewer. main.cpp's Format Conversion step always writes a
			// same-basename .png next to a successful render's .ppm output -
			// load that (smaller, simpler than parsing PPM by hand).
			if (!outputPath.isEmpty()) {
				QFileInfo fileInfo(outputPath);
				QString pngPath = fileInfo.absolutePath() + "/" + fileInfo.completeBaseName() + ".png";
				QFileInfo pngInfo(pngPath);

				if (pngInfo.exists()) {
					QPixmap pixmap(pngPath);
					if (!pixmap.isNull()) {
						const QString infoText = QString("%1  •  %2×%3  •  %4 KB  •  %5s")
							.arg(pngInfo.fileName())
							.arg(pixmap.width()).arg(pixmap.height())
							.arg(pngInfo.size() / 1024)
							.arg(totalTime, 0, 'f', 2);
						addImagePreviewTab(finishedJob.displayTitle, finishedJob.sceneDescription,
											pixmap, infoText, outputPath, pngPath);
						if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
					} else {
						m_statusLabel->setText(QString("✅ Render complete (%1s)").arg(totalTime, 0, 'f', 2));
						setStatusWarning(QString("Warning: preview image failed to load at %1").arg(pngPath));
					}
				} else if (fileInfo.exists()) {
					// PNG conversion failed but the raw PPM output exists -
					// fall back to the old external-viewer behavior rather
					// than showing nothing.
					QDesktopServices::openUrl(QUrl::fromLocalFile(outputPath));
				} else {
					m_statusLabel->setText(QString("✅ Render complete (%1s)").arg(totalTime, 0, 'f', 2));
					setStatusWarning(QString("Warning: output file not found at %1").arg(outputPath));
				}
			}
		}
	} else {
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

	// Cleared unconditionally rather than only on the "nothing queued" path:
	// if processQueueIfIdle() below does start the next job, startRenderJob()
	// overwrites this again before the next repaint, so there's no visible
	// flicker - and it means this code doesn't have to duplicate the queue's
	// own idle/non-idle logic to decide whether to clear it.
	m_currentJobLabel->clear();

	// Continue automatically on natural completion (success or a genuine
	// failure); a user-requested Stop pauses the queue instead of skipping
	// straight to the next job. The remaining jobs stay queued, and the next
	// click of Start Render both resumes them and appends whatever's in the
	// form as one more job at the back - see onRenderClicked()'s own comment.
	if (!stoppedByUser) {
		processQueueIfIdle();
	} else if (!m_renderQueue.isEmpty()) {
		m_statusLabel->setText(QString("Stopped - %1 more queued (click Start Render to resume)").arg(m_renderQueue.size()));
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

// Re-renders the diagnostics report in the current scheme, same reasoning
// and same "replace, don't recolour" constraint as rebuildLogPane() above.
// A no-op when the pane isn't currently showing a report (m_lastDiagReport
// empty - e.g. it's showing "Running diagnostics..." or a failure message).
void MainWindow::rebuildDiagPane() {
	if (!m_diagTextEdit || m_lastDiagReport.isEmpty()) return;

	const bool scrolledToBottom =
		m_diagTextEdit->verticalScrollBar()->value() ==
		m_diagTextEdit->verticalScrollBar()->maximum();

	m_diagTextEdit->setUpdatesEnabled(false);
	m_diagTextEdit->clear();
	const QStringList lines = m_lastDiagReport.split('\n');
	for (const QString &line : lines) {
		if (line.isEmpty()) continue;
		m_diagTextEdit->append(styleDiagnosticsLine(m_activeTheme, line));
	}
	m_diagTextEdit->setUpdatesEnabled(true);

	if (!scrolledToBottom)
		m_diagTextEdit->moveCursor(QTextCursor::Start);
	else
		m_diagTextEdit->moveCursor(QTextCursor::End);
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

	const QString title = success ? "Render complete"
								  : (stoppedByUser ? "Render stopped" : "Render failed");
	const QString body = success
		? QString("Finished in %1 seconds").arg(totalTime, 0, 'f', 2)
		: message.section('<', 0, 0).left(120);

	// While the window IS active, a toast inside it is the completion cue
	// instead of a tray balloon - the user is plausibly looking right at the
	// app already, so a corner-of-the-eye tray message is easy to miss and
	// would just be a second, redundant notification for the same event.
	if (isActiveWindow()) {
		if (m_toast) {
			const QColor fill = success ? m_activeTheme.success
										: (stoppedByUser ? m_activeTheme.textMuted : m_activeTheme.error);
			// Mirrors textOn()'s own threshold (mainwindow_style.cpp, internal
			// linkage - not reachable from this file) rather than sharing it:
			// one ternary isn't worth a cross-TU declaration.
			const QColor onFill = fill.lightness() > 170 ? m_activeTheme.surface0 : QColor(Qt::white);
			m_toast->showToast(QString("%1 – %2").arg(title, body), fill, onFill);
		}
		return;
	}

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

void MainWindow::startProgressGlow() {
	if (!m_progressBar) return;
	// Applied lazily here rather than at the bar's creation - it only ever
	// needs to exist once a render has actually started once. Left in place
	// afterward (stopProgressGlow() only stops the pulse, never removes the
	// effect), so a second render job reuses the same glow object rather
	// than replacing it.
	constexpr qreal kRestGlow = 10.0, kPeakGlow = 24.0;
	auto *glow = qobject_cast<QGraphicsDropShadowEffect *>(m_progressBar->graphicsEffect());
	if (!glow) {
		applyGlow(m_progressBar, kRestGlow, m_activeTheme.accentPrimary);
		glow = qobject_cast<QGraphicsDropShadowEffect *>(m_progressBar->graphicsEffect());
	}
	if (!glow) return;

	if (!m_progressGlowAnim) {
		m_progressGlowAnim = new QPropertyAnimation(glow, "blurRadius", this);
		m_progressGlowAnim->setDuration(1400);
		m_progressGlowAnim->setLoopCount(-1);
		m_progressGlowAnim->setEasingCurve(QEasingCurve::InOutSine);
		m_progressGlowAnim->setKeyValueAt(0.0, kRestGlow);
		m_progressGlowAnim->setKeyValueAt(0.5, kPeakGlow);
		m_progressGlowAnim->setKeyValueAt(1.0, kRestGlow);
	}
	m_progressGlowAnim->stop();
	m_progressGlowAnim->start();
}

void MainWindow::stopProgressGlow() {
	if (m_progressGlowAnim) m_progressGlowAnim->stop();
	// Settles to the same low ambient level the pulse breathes around,
	// rather than 0 - a bar that only ever glows while active would pop
	// abruptly back to flat the moment a render finishes; staying lit at a
	// low level reads as "idle", not "broken".
	if (auto *glow = qobject_cast<QGraphicsDropShadowEffect *>(
			m_progressBar ? m_progressBar->graphicsEffect() : nullptr))
		glow->setBlurRadius(10.0);
}

void MainWindow::animateProgressTo(int value) {
	if (!m_progressBar) return;
	if (!m_progressValueAnim) {
		m_progressValueAnim = new QPropertyAnimation(m_progressBar, "value", this);
		m_progressValueAnim->setEasingCurve(QEasingCurve::OutCubic);
	}
	m_progressValueAnim->stop();
	m_progressValueAnim->setDuration(300);
	m_progressValueAnim->setStartValue(m_progressBar->value());
	m_progressValueAnim->setEndValue(value);
	m_progressValueAnim->start();
}

void MainWindow::setStatusWarning(const QString &text) {
	if (!m_statusWarningLabel) return;
	m_statusWarningLabel->setText(text);
	m_statusWarningLabel->show();
}

void MainWindow::clearStatusWarning() {
	if (!m_statusWarningLabel) return;
	m_statusWarningLabel->clear();
	m_statusWarningLabel->hide();
}

void MainWindow::onElapsedTick() {
	if (!m_isRendering) return;
	const qint64 elapsedMs = m_renderStartTime.msecsTo(QDateTime::currentDateTime());
	m_statusLabel->setText(formatProgressStatus(elapsedMs, m_progressBar->value()));
}

void MainWindow::onModeChanged(int index) {
	m_videoMode = (index == 1); // 0 = Image, 1 = Video

	// Every control on Video Settings is inert unless Output Mode is
	// "Generate Video" - the banner (see createVideoTab()'s own comment) is
	// what surfaces that, rather than disabling the tab outright and
	// blocking the user from browsing/configuring it ahead of switching
	// modes.
	if (m_videoModeWarningLabel) m_videoModeWarningLabel->setVisible(!m_videoMode);

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
		m_statusLabel->setText("⚠️ Video file not found, checking for frames...");
		onLogMessage("WARNING: Video file not found at any of the expected locations");

		// Check if frames exist (fallback diagnostic)
		QString framesDir = outputDir + "/frames";
		QDir framesDirObj(framesDir);

		if (framesDirObj.exists()) {
			QStringList frames = framesDirObj.entryList(QStringList() << "frame_*.ppm", QDir::Files);
			if (!frames.isEmpty()) {
				m_statusLabel->setText(QString("⚠️ Found %1 frames but no video file").arg(frames.count()));
				onLogMessage(QString("Frames were rendered (%1 files) but video assembly may have failed.").arg(frames.count()));
				QMessageBox::warning(this, "Video Not Created",
					QString("Frames were rendered successfully (%1 files), but the video file was not created.\n\n"
							"Expected video at: %2\n\n"
							"Please check the render log for ffmpeg errors.").arg(frames.count()).arg(expectedVideoPath));
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
	if (title.isEmpty()) title = job.displayTitle + " (Video)";

	const QString infoText = QString("%1  •  %2 MB  •  %3 frames")
		.arg(videoInfo.fileName())
		.arg(videoInfo.size() / (1024.0 * 1024.0), 0, 'f', 1)
		.arg(frameFiles.count());
	addVideoPreviewTab(title, job.sceneDescription, videoPath, infoText);
	if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);

	onLogMessage(QString("Playing video inline: %1").arg(videoPath));
}
