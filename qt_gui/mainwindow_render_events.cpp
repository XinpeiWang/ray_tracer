// mainwindow_render_events.cpp - what happens while and after a render runs: progress, completion, the status line (elapsed, ETA, glow) and the
// finished notification (split out of mainwindow_slots.cpp; nothing changed).

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
		m_statusLabel->setText(tr("Rendering... %1%").arg(percentage));
}

void MainWindow::onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath) {
	// Snapshotted once, up front: this function (indirectly, via the
	// deferred singleShot below) can end up running after m_currentJob has
	// already been overwritten by a queued next job - see m_currentJob's own
	// comment. Everything below reads finishedJob, never m_currentJob
	// directly.
	const RenderJob finishedJob = m_currentJob;
	const int finishedQueueId = m_currentQueueId;   // the queue row to give the result to (the next job will take m_currentQueueId over)
	m_currentQueueId = 0;

	m_isRendering = false;
	m_renderButton->setEnabled(true);
	m_stopButton->setEnabled(false);
	m_pauseButton->setEnabled(false);
	m_abandonButton->setEnabled(false);
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	updateTransportButtons();
#endif
	m_pauseStartedAt = QDateTime();
	if (m_elapsedTimer) m_elapsedTimer->stop();
	updateActionStates();
	// Cleared unconditionally, before either branch below runs: a warning
	// left over from a previous job's preview failure must not linger next
	// to this job's own (possibly unrelated) outcome.
	clearStatusWarning();
	stopProgressGlow();

	const bool stoppedByUser = !success && message.contains("stopped by user", Qt::CaseInsensitive);
	// See RenderController::abandonRender()'s own comment: same "process was
	// killed on purpose, not a real failure" shape as stoppedByUser, but the
	// queue-advance decision at the bottom of this function treats the two
	// oppositely - a plain Stop pauses the queue, an Abandon skips ahead.
	const bool abandonedByUser = !success && message.contains("abandoned by user", Qt::CaseInsensitive);
	const bool userEndedWithoutFailure = stoppedByUser || abandonedByUser;
	// The two user-ended messages are matched by their English text above and
	// in notifyRenderFinished(), so RenderController keeps them canonical and
	// the translation happens only here, for display.
	const QString shownMessage = stoppedByUser ? tr("Render stopped by user")
		: abandonedByUser ? tr("Render abandoned by user") : message;

	if (finishedQueueId)
		m_queueModel->finish(finishedQueueId, success ? job_queue::State::Done : userEndedWithoutFailure ? job_queue::State::Cancelled : job_queue::State::Failed,
		                     totalTime, shownMessage);
	refreshQueuePanel();

	// A failed render leaves the taskbar button red so the outcome is visible
	// without switching to the window; anything else clears it. Leaving a
	// progress state set would make it stick until the process exits.
	notifyRenderFinished(success, message, totalTime);

	if (success) {
		animateProgressTo(100);
		// A finished bar keeps its fill and turns green rather than resetting -
		// the outcome stays visible after the fact (Qt Creator's behaviour).
		setProgressResultState("success");
		m_statusLabel->setText(tr("✅ %1 - Total time: %2 seconds").arg(shownMessage).arg(totalTime, 0, 'f', 2));

		if (finishedJob.videoMode) {
			onLogMessage(tr("Video frames rendered successfully. Starting video assembly..."));
			m_statusLabel->setText(tr("⚙️ Assembling video from frames..."));

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
						const QString infoText = tr("%1  •  %2×%3  •  %4 KB  •  %5s  •  %6spp · %7%8")
							.arg(pngInfo.fileName())
							.arg(pixmap.width()).arg(pixmap.height())
							.arg(pngInfo.size() / 1024)
							.arg(totalTime, 0, 'f', 2)
							.arg(finishedJob.samples)
							.arg(rendererLabel(finishedJob.useGPU, finishedJob.useWavefront),
								 integratorSuffixTag(finishedJob.integratorOptions.mode));
						addImagePreviewTab(finishedJob.displayTitle, finishedJob.sceneDescription,
											pixmap, infoText, outputPath, pngPath,
											{finishedJob.sceneId, renderTechniqueHtml(finishedJob.integratorOptions, finishedJob.advancedFlags)});
						saveRecentRender(finishedJob, pngPath, /*isVideo=*/false);
						refreshRecentRendersList();
						if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
					} else {
						m_statusLabel->setText(tr("✅ Render complete (%1s)").arg(totalTime, 0, 'f', 2));
						setStatusWarning(tr("Warning: preview image failed to load at %1").arg(pngPath));
					}
				} else if (fileInfo.exists()) {
					// Either PNG conversion failed but the raw PPM output
					// still exists, OR outputPath is itself a .exr (main.cpp's
					// own Format Conversion step deliberately never generates
					// a PNG for those - see is_exr_output_path()'s callers) -
					// fall back to the OS's own default handler for the file
					// rather than showing nothing/erroring on an unsupported
					// format. Does not add a Recent Renders entry or an inline
					// preview tab in either case; only the PNG-found branch
					// above does that.
					QDesktopServices::openUrl(QUrl::fromLocalFile(outputPath));
				} else {
					m_statusLabel->setText(tr("✅ Render complete (%1s)").arg(totalTime, 0, 'f', 2));
					setStatusWarning(tr("Warning: output file not found at %1").arg(outputPath));
				}
			}
		}
	} else {
		// A user-requested stop or abandon isn't a failure, so it clears back
		// to neutral; a genuine failure leaves the bar where it died and turns
		// it red, so the outcome is still readable after the dialog is
		// dismissed.
		if (userEndedWithoutFailure) {
			m_progressBar->setValue(0);
			setProgressResultState("");
		} else {
			setProgressResultState("error");
		}
		m_statusLabel->setText(tr("❌ %1").arg(shownMessage));

		// Only show error popup for actual failures, not for user-stopped/
		// abandoned renders.
		if (!userEndedWithoutFailure) {
			QMessageBox::critical(this, tr("Render Failed"), shownMessage);
		}
	}

	// Cleared unconditionally rather than only on the "nothing queued" path:
	// if processQueueIfIdle() below does start the next job, startRenderJob()
	// overwrites this again before the next repaint, so there's no visible
	// flicker - and it means this code doesn't have to duplicate the queue's
	// own idle/non-idle logic to decide whether to clear it.
	m_currentJobLabel->clear();

	// Continue automatically on natural completion (success or a genuine
	// failure) AND on an explicit Abandon - the whole point of "Abandon &
	// Next" is skipping straight ahead. A plain Stop is the one case that
	// pauses the queue instead: the remaining jobs stay queued, and the next
	// click of Start Render both resumes them and appends whatever's in the
	// form as one more job at the back - see onRenderClicked()'s own comment.
	if (!stoppedByUser) {
		processQueueIfIdle();
	} else if (m_queueModel->hasWaiting()) {
		m_statusLabel->setText(tr("Stopped - %1 more queued (click Start Render to resume)").arg(m_queueModel->waitingCount()));
	}
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

	QString text = tr("Rendering  ·  %1%  ·  elapsed %2")
		.arg(percent, 3)
		.arg(formatDuration(elapsedMs / 1000));

	// Instantaneous rate across the ring - shown, never divided by.
	const ProgressSample &oldest = m_progressRing[0];
	const qint64 windowMs = elapsedMs - oldest.elapsedMs;
	if (m_progressRingCount == kProgressSamples && windowMs > 0) {
		const double pctPerSec = 1000.0 * (percent - oldest.percent) / windowMs;
		if (pctPerSec > 0.0)
			text += tr("  ·  %1 %/s").arg(pctPerSec, 0, 'f', 1);
	}

	// ETA from the cumulative rate, suppressed during warm-up.
	if (elapsedMs >= kEtaWarmupMs && percent > 0 && percent < 100) {
		const double pctPerMs = double(percent) / double(elapsedMs);
		if (pctPerMs > 0.0) {
			const qint64 remainingMs = qint64((100.0 - percent) / pctPerMs);
			text += tr("  ·  ETA %1").arg(formatDuration(remainingMs / 1000));
		}
	} else if (percent < 100) {
		text += tr("  ·  ETA --:--");
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

	const QString title = success ? tr("Render complete")
								  : (stoppedByUser ? tr("Render stopped") : tr("Render failed"));
	const QString body = success
		? tr("Finished in %1 seconds").arg(totalTime, 0, 'f', 2)
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
			m_toast->showToast(tr("%1 – %2").arg(title, body), fill, onFill);
		}
		return;
	}

	// Log why a notification was or wasn't raised - a silently-swallowed
	// showMessage() (which is what an invisible tray icon does) is otherwise
	// indistinguishable from the feature simply not being wired up.
	if (!m_trayIcon) {
		onLogMessage(tr("[DEBUG] No system tray available; skipping completion notification"));
		return;
	}
	if (!QSystemTrayIcon::supportsMessages()) {
		onLogMessage(tr("[DEBUG] System tray does not support messages; skipping notification"));
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
