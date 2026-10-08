// mainwindow_diagnostics_log.cpp - the Diagnostics run and its report pane, and the Log pane (colouring renderer output lines) (split out of
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
		return QString("<span style='color:%1;font-family:Consolas,monospace;'>"
					   "<b>%2</b></span>").arg(colour, escaped);
	case render_output::LineStyle::BoldLabeled:
		return QString("<span style='color:%1;font-family:Consolas,monospace;'>"
					   "<b><span style='color:%5;'>%2</span> "
					   "<span style='color:%1;'>[%3]</span> %4</b></span>")
			.arg(colour, timestamp, QString::fromLatin1(cat.label), escaped, timestampColour);
	case render_output::LineStyle::Normal:
		break;
	}
	return QString("<span style='color:%1;font-family:Consolas,monospace;'>"
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
	const QString mono = "font-family:Consolas,monospace;";

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

void MainWindow::onRunDiagnosticsClicked() {
	// A stray click while one is already running would leak a second
	// QProcess and race both sets of signals into the same text edit.
	if (diagnosticsBusy()) return;

	m_lastDiagReport.clear();  // no report to recolour until reportReady fires
	if (m_diagTextEdit) {
		m_diagShowsHint = false;
		m_diagTextEdit->clear();
		m_diagTextEdit->setPlainText(tr("Running diagnostics..."));
	}
	if (m_runDiagnosticsButton) m_runDiagnosticsButton->setEnabled(false);
	if (m_installPhotoHelperButton) m_installPhotoHelperButton->setEnabled(false);

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
	for (const QString &line : report.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) AppLog::info(QStringLiteral("diagnostics"), line);
	rebuildDiagPane();

	// The CLI's report has no network facts (and cannot - the point is whether THIS app can reach the download server), so
	// the GUI adds its own section: shown as soon as the check finishes, and part of the saved/copied report from then on.
	if (!m_connectionCheck) {
		m_connectionCheck = new asset_downloader::ConnectionCheck(asset_downloader::builtInManifest(), this);
		connect(m_connectionCheck, &asset_downloader::ConnectionCheck::finished, this, [this](const QString &section) {
			if (!m_lastDiagReport.isEmpty()) {
				if (!m_lastDiagReport.endsWith(QLatin1Char('\n'))) m_lastDiagReport += QLatin1Char('\n');
				m_lastDiagReport += section;
				rebuildDiagPane();
			}
			startPhotoHelperCheck();  // the next section; the button stays off until it is in
		});
	}
	m_connectionCheck->start();
	// The runner's own cleanup (connected after this slot) re-enables the button; keep it off until the checks are done too.
	QTimer::singleShot(0, this, [this]() {
		if (m_runDiagnosticsButton && diagnosticsBusy()) m_runDiagnosticsButton->setEnabled(false);
	});
}

bool MainWindow::diagnosticsBusy() const {
	return m_diagnosticsRunner || (m_connectionCheck && m_connectionCheck->isRunning()) || (m_photoCheck && m_photoCheck->isRunning());
}

// The report's last section. Run after the network check so the sections always come in the same order.
void MainWindow::startPhotoHelperCheck() {
	if (!m_photoCheck) {
		m_photoCheck = new PhotoHelperCheck(this);
		connect(m_photoCheck, &PhotoHelperCheck::finished, this, [this](const QString &section) {
			if (!m_lastDiagReport.isEmpty()) {
				if (!m_lastDiagReport.endsWith(QLatin1Char('\n'))) m_lastDiagReport += QLatin1Char('\n');
				m_lastDiagReport += section;
				// The last section: where this program's own log is, so a report and its log travel together.
				m_lastDiagReport += QStringLiteral("\n=== Log file ===\n  %1\n  Help > Show Log Folder opens it. Send it with a bug report: it lists what the program did, in order.\n")
				                        .arg(QDir::toNativeSeparators(AppLog::filePath().isEmpty() ? tr("(could not be created)") : AppLog::filePath()));
				if (AppLog::previousSessionEndedUnexpectedly())
					m_lastDiagReport += QStringLiteral("  The previous session did not exit cleanly (see the log).\n");
				for (const QString &line : section.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) AppLog::info(QStringLiteral("diagnostics"), line);
				rebuildDiagPane();
			}
			if (m_runDiagnosticsButton && !diagnosticsBusy()) m_runDiagnosticsButton->setEnabled(true);
			updateInstallPhotoButton();
		});
	}
	m_photoCheck->start();
}

void MainWindow::onDiagnosticsFailed(const QString &message) {
	// Not a report - plain text, nothing to colour, and clearing
	// m_lastDiagReport keeps a later theme change from trying to recolour
	// a report that isn't showing anymore.
	m_lastDiagReport.clear();
	updateInstallPhotoButton();
	m_diagShowsHint = false;
	if (m_diagTextEdit) m_diagTextEdit->setPlainText(tr("Diagnostics failed:\n\n%1").arg(message));
}

void MainWindow::onLogMessage(const QString &message) {
	QString msg = message.trimmed();
	if (msg.isEmpty()) return;

	// Every line the pane shows is also kept in the log file, which outlives the pane. A render prints a "Scanlines remaining" line per scanline: one in
	// twenty-five is enough there.
	{
		static int scanlineLines = 0;
		const bool progressTick = msg.startsWith(QLatin1String("Scanlines remaining:"));
		if (!progressTick || (++scanlineLines % 25) == 1) {
			const render_output::LogCategory c = render_output::classifyLogLine(msg.toStdString());
			if (c.severity != render_output::LogSeverity::Separator)
				AppLog::write(c.severity == render_output::LogSeverity::Error ? log_format::Level::Error
				              : c.severity == render_output::LogSeverity::Warning ? log_format::Level::Warn
				              : c.severity == render_output::LogSeverity::Debug ? log_format::Level::Debug : log_format::Level::Info,
				              QStringLiteral("pane"), msg);
		}
	}
	if (!m_logTextEdit) return;

	// Timestamp prefix (HH:mm:ss)
	QString ts = QTime::currentTime().toString("HH:mm:ss");

	// HTML-escape so < > & don't break the rich-text display
	QString escaped = msg.toHtmlEscaped();

	// Classification is the tested, Qt-free implementation; this function is
	// left with nothing but presentation.
	const render_output::LogCategory category =
		render_output::classifyLogLine(msg.toStdString());

	m_logHistory.push_back({ts, escaped, category});
	if (m_logHistory.size() > kMaxLogHistoryLines) {
		// Batch-trim back to 90% of the cap rather than popping one line
		// every push once at the cap - that would turn every single log
		// line for the rest of the session into an O(n) QVector::remove().
		m_logHistory.remove(0, m_logHistory.size() - (kMaxLogHistoryLines * 9 / 10));
	}
	m_logTextEdit->append(styleLogLine(category,
									   m_activeTheme.colourFor(category.severity).name(),
									   m_activeTheme.logSeparator.name(),
									   ts, escaped));
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
// The empty Diagnostics pane says what to do, as real text in the theme's muted colour (the view's placeholder text was drawn in a colour the style sheet left
// unreadable, so the pane looked broken instead of waiting for a click).
void MainWindow::showDiagnosticsHint() {
	if (!m_diagTextEdit) return;
	m_diagShowsHint = true;
	m_diagTextEdit->setHtml(QStringLiteral("<p style=\"color:%1\">%2</p>")
	                            .arg(m_activeTheme.textMuted.name(), tr("Click \"Run Diagnostics\" to check GPU/CUDA/OptiX availability, CPU/RAM, "
	                                                                    "disk space, and scene asset availability.").toHtmlEscaped()));
}

void MainWindow::rebuildDiagPane() {
	if (m_diagTextEdit && m_lastDiagReport.isEmpty() && m_diagShowsHint) showDiagnosticsHint();   // a theme change re-colours the hint
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
