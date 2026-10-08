// mainwindow_thumbnails.cpp - generating the scene-list thumbnails: which scenes qualify, the batch, pause, stop and progress (split out of
// mainwindow_slots.cpp; nothing changed).

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


bool MainWindow::isThumbnailEligible(const QString &sceneId) {
	SceneMetadataClient::SceneMetadata meta;
	// Same "can't query -> don't include" caution as every other
	// SceneMetadataClient call site in this file: a scene this couldn't even
	// fetch metadata for is not one to blindly hand to a CPU render.
	//
	// Used to also exclude "Very Slow" scenes here, on the theory that a bulk
	// "Generate Thumbnails" click could look hung for a very long time on a
	// multi-million-triangle whole-environment mesh (Bistro, Rungholt, ...) -
	// loading and BVH-building geometry that size is a real, substantial
	// fixed cost independent of the thumbnail's own tiny 128x128/16spp
	// render settings. Removed at the user's own request: the pause/stop
	// controls and per-scene progress log (onThumbnailProgress()'s own
	// "(N/M) Rendering <scene>..." line) already give a way to see what's
	// happening and cancel if a particular category turns out to take too
	// long, so excluding these scenes entirely was more caution than the
	// existing UI actually needed.
	return SceneMetadataClient::sceneMetadata(sceneId, meta);
}

// The scenes a "Generate Thumbnails" click on `category` would actually
// attempt - filteredSceneIds() already applies the same category/
// availability-tab/search-box filter the visible grid itself uses (so a
// "Requires External Files" scene is only ever included while that tab is
// the one showing - if a required asset turns out to still be missing
// locally, the render just fails and gets logged/counted like any other
// failure, same as a real render of that scene would), narrowed further by
// isThumbnailEligible()'s own check. The one list both
// onGenerateThumbnailsClicked() and updateGenerateThumbnailsButtonState()
// read, so the button's enabled state can never drift out of sync with
// what a click on it actually does.
QStringList MainWindow::eligibleThumbnailIds(const QString &category) const {
	QStringList ids;
	for (const QString &id : filteredSceneIds(category)) {
		if (isThumbnailEligible(id)) ids << id;
	}
	return ids;
}

// Fills in m_sceneGrid's preview tiles for whichever category/availability/
// search combination is CURRENTLY showing - see eligibleThumbnailIds()'s own
// comment for exactly what qualifies. Scoped to the current category alone
// (not every category at once, which an earlier version of this did)
// because the button sits directly under that one category's grid - generating
// thumbnails for scenes the user isn't even looking at, while the ones
// actually on screen stay placeholders, was the surprising part. Disabled
// (see createSettingsTab()'s button tooltip) while a real render is in
// flight so thumbnail generation can never compete with the user's own
// queued work - m_thumbnailGenerator owns a private RenderController instead
// of reusing m_renderController/m_renderQueue precisely so it never needs to
// cooperate with those at all, only avoid running alongside them.
void MainWindow::onGenerateThumbnailsClicked() {
	if (m_isRendering || m_queueModel->hasWaiting()) {
		setStatusWarning(tr("Can't generate thumbnails while a render is in progress or queued."));
		return;
	}
	if (m_thumbnailGenerator && m_thumbnailGenerator->isRunning()) return;

	const QString currentCategory = (m_sceneCategoryTabs && m_sceneCategoryTabs->count() > 0)
		? m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString() : QString();
	const QStringList ids = eligibleThumbnailIds(currentCategory);
	if (ids.isEmpty()) {
		setStatusWarning(tr("Nothing to generate thumbnails for in the current view."));
		return;
	}

	if (!m_thumbnailGenerator) {
		m_thumbnailGenerator = new ThumbnailGenerator(this);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::thumbnailReady,
				this, &MainWindow::onThumbnailReady);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::progress,
				this, &MainWindow::onThumbnailProgress);
		// Forwards this generator's own private RenderController's log
		// lines (see ThumbnailGenerator::logMessage()'s own comment) into
		// the same Log Output tab a user-requested render uses - without
		// this, a failed thumbnail gave no way to tell WHY.
		connect(m_thumbnailGenerator, &ThumbnailGenerator::logMessage,
				this, &MainWindow::onLogMessage);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::pauseStateChanged,
				this, &MainWindow::onThumbnailPauseStateChanged);
		connect(m_thumbnailGenerator, &ThumbnailGenerator::allDone,
				this, &MainWindow::onThumbnailsAllDone);
	}

	if (m_generateThumbnailsButton) m_generateThumbnailsButton->setEnabled(false);
	if (m_thumbnailPauseButton) {
		m_thumbnailPauseButton->setText(tr("Pause"));
		m_thumbnailPauseButton->setVisible(true);
	}
	if (m_thumbnailStopButton) m_thumbnailStopButton->setVisible(true);
	m_thumbnailSucceededCount = 0;
	m_thumbnailFailedCount = 0;
	m_thumbnailBatchStartTime = QDateTime::currentDateTime();
	onLogMessage(QString("Generating thumbnails for up to %1 scene(s) in \"%2\"...")
		.arg(ids.size()).arg(currentCategory));
	m_thumbnailGenerator->start(ids, [this](const QString &id) { return thumbnailCachePath(id); });
}

void MainWindow::onThumbnailStopClicked() {
	if (!m_thumbnailGenerator) return;
	// Resume first if paused - stop() itself works fine on a suspended
	// process (RenderController::stopRender()'s own comment), but skipping
	// this would leave onThumbnailPauseStateChanged()'s elapsed-time shift
	// never applied for this final pause segment, over-counting it as
	// elapsed work time in onThumbnailsAllDone()'s summary.
	if (m_thumbnailGenerator->isPaused()) m_thumbnailGenerator->resume();
	// onThumbnailsAllDone() (connected to ThumbnailGenerator::allDone) does
	// the actual UI teardown once the killed process's renderComplete
	// signal actually arrives - same async shape as the real render's own
	// onStopClicked()/RenderController::stopRender().
	m_thumbnailGenerator->stop();
}

void MainWindow::onThumbnailPauseClicked() {
	if (!m_thumbnailGenerator || !m_thumbnailGenerator->isRunning()) return;
	if (m_thumbnailGenerator->isPaused()) {
		m_thumbnailGenerator->resume();
	} else {
		m_thumbnailGenerator->pause();
	}
	// onThumbnailPauseStateChanged() (connected above) flips the button's
	// own label once ThumbnailGenerator confirms the change.
}

void MainWindow::onThumbnailPauseStateChanged(bool paused) {
	if (m_thumbnailPauseButton) m_thumbnailPauseButton->setText(paused ? tr("Resume") : tr("Pause"));
	if (paused) {
		m_thumbnailPauseStartedAt = QDateTime::currentDateTime();
	} else if (m_thumbnailPauseStartedAt.isValid()) {
		m_thumbnailBatchStartTime = m_thumbnailBatchStartTime.addMSecs(
			m_thumbnailPauseStartedAt.msecsTo(QDateTime::currentDateTime()));
		m_thumbnailPauseStartedAt = QDateTime();
	}
}

void MainWindow::onThumbnailReady(const QString &sceneId, bool success, const QString &outputPath) {
	if (!success) {
		++m_thumbnailFailedCount;
		onLogMessage(QString("Thumbnail generation failed for scene %1").arg(sceneId));
		return;
	}
	++m_thumbnailSucceededCount;
	if (!m_sceneGrid) return;
	bool matched = false;
	for (int i = 0; i < m_sceneGrid->count(); ++i) {
		QListWidgetItem *item = m_sceneGrid->item(i);
		if (item->data(Qt::UserRole).toString() == sceneId) {
			item->setIcon(QIcon(outputPath));
			matched = true;
			break;
		}
	}
	// Not a failure - just means the grid moved on to a different category
	// tab while this scene was still rendering in the background - but
	// silently doing nothing here previously looked identical to a real bug
	// ("why didn't my thumbnail show up?") from the Log Output tab alone.
	if (!matched)
		onLogMessage(QString("Thumbnail for %1 saved, but it's not in the currently-shown grid.").arg(sceneId));
}

void MainWindow::onThumbnailProgress(int completed, int total, const QString &sceneId) {
	// A permanent record of the same text the progress bar shows only while
	// it's visible - onThumbnailsAllDone() hides the bar, so this is the
	// only place that sequence survives to be scrolled back through.
	onLogMessage(QString("[Thumbnail] (%1/%2) Rendering %3 (%4)...")
		.arg(completed + 1).arg(total).arg(SceneMetadataClient::sceneName(sceneId), sceneId));
	if (!m_thumbnailProgressBar) return;
	m_thumbnailProgressBar->setRange(0, total);
	m_thumbnailProgressBar->setValue(completed);
	m_thumbnailProgressBar->setFormat(tr("Generating thumbnail %1 of %2: %3")
		.arg(completed + 1).arg(total).arg(SceneMetadataClient::sceneName(sceneId)));
	m_thumbnailProgressBar->setVisible(true);
}

void MainWindow::onThumbnailsAllDone() {
	// Not an unconditional setEnabled(true): the user may have switched to
	// an unsupported category (e.g. "Models") while this run - started on a
	// supported one - was still working in the background. Re-evaluating
	// against whatever category is CURRENTLY showing avoids leaving the
	// button wrongly enabled there.
	updateGenerateThumbnailsButtonState();
	if (m_thumbnailProgressBar) m_thumbnailProgressBar->setVisible(false);
	if (m_thumbnailPauseButton) m_thumbnailPauseButton->setVisible(false);
	if (m_thumbnailStopButton) m_thumbnailStopButton->setVisible(false);
	// m_thumbnailBatchStartTime's origin was shifted forward by however long
	// any pause lasted (onThumbnailPauseStateChanged()), so this plain
	// wall-clock diff already excludes paused time - same trick
	// RenderController uses for a single render's own totalTime.
	const double elapsedSec = m_thumbnailBatchStartTime.msecsTo(QDateTime::currentDateTime()) / 1000.0;
	onLogMessage(QString("Thumbnail generation finished: %1 succeeded, %2 failed, %3s elapsed.")
		.arg(m_thumbnailSucceededCount).arg(m_thumbnailFailedCount)
		.arg(elapsedSec, 0, 'f', 1));
	statusBar()->showMessage(m_thumbnailFailedCount > 0
		? tr("Thumbnail generation finished - %1 failed.").arg(m_thumbnailFailedCount)
		: tr("Thumbnail generation finished."), 5000);
}

void MainWindow::updateGenerateThumbnailsButtonState() {
	if (!m_generateThumbnailsButton) return;
	// A tab switch mid-generation must not fight onGenerateThumbnailsClicked()'s
	// own disable - the button stays disabled until allDone() (which calls
	// this function itself, see its own comment) re-evaluates for real.
	if (m_thumbnailGenerator && m_thumbnailGenerator->isRunning()) return;

	const QString currentCategory = (m_sceneCategoryTabs && m_sceneCategoryTabs->count() > 0)
		? m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString() : QString();
	// Same call onGenerateThumbnailsClicked() itself makes - the button's
	// enabled state can never drift out of sync with what a click actually
	// does, whether that's an empty category, a search term matching
	// nothing, or (now that eligibleThumbnailIds() no longer excludes
	// requires-files OR Very Slow scenes) simply every scene in view
	// already having a cached thumbnail.
	const bool supported = !eligibleThumbnailIds(currentCategory).isEmpty();
	m_generateThumbnailsButton->setEnabled(supported);
	m_generateThumbnailsButton->setToolTip(supported
		? tr("Creates a small preview image for each ready-to-render scene in the CURRENT view that\n"
		"doesn't already have one saved. Runs on the CPU only, at low resolution - it can still take a\n"
		"while for a \"Very Slow\" whole-environment scene, since loading and BVH-building a\n"
		"multi-million-triangle mesh costs the same regardless of the thumbnail's own small size.\n"
		"Use the pause/stop controls if a category turns out to take too long.")
		: tr("Nothing to generate thumbnails for in the current view."));
}
