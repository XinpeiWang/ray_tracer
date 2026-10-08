// Progress, Log, and Diagnostics tabs - split out of mainwindow_tabs.cpp;
// see mainwindow_tabs_render.cpp for the Render Options/Preview tabs this
// file's content used to sit between, in both the old file's function order
// and the tab bar's own left-to-right order.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"

#include "../src/shared/scene_descriptor.h"
#include "../src/shared/video_preset.h"

#include <QTabBar>
#include <QTextDocument>
#include "scene_metadata_client.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
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
#include <QHeaderView>
#include <QTableView>
#include <QSettings>
#include <QStatusBar>
#include <QWheelEvent>
#include <functional>
#include "settings_keys.h"
#include <cmath>
#include <algorithm>

void MainWindow::createProgressTab() {
	QWidget *progressWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(progressWidget);

	InfoGroupBox *progressGroup = new InfoGroupBox(tr("Progress"), progressWidget);
	progressGroup->setInfoIcon(createInfoIcon(
		tr("Shows what's rendering right now - which job it is, how far "
		"along it is (as a percentage), and how much time has passed and "
		"is left. Pause, Stop, and Abandon only affect this job.")));
	QVBoxLayout *progressLayout = new QVBoxLayout(progressGroup);

	// Which job is actually running (scene/resolution/samples/renderer, same
	// one-line format as a queue row) - set from m_currentJob in
	// startRenderJob(), not the live Settings form, since the user may
	// have already changed the form for a job queued behind this one.
	m_currentJobLabel = new QLabel(progressGroup);
	m_currentJobLabel->setAlignment(Qt::AlignCenter);
	m_currentJobLabel->setObjectName("currentJobLabel");
	m_currentJobLabel->setWordWrap(true);

	m_progressBar = new QProgressBar(progressGroup);
	m_progressBar->setRange(0, 100);
	m_progressBar->setValue(0);
	m_progressBar->setTextVisible(true);

	m_statusLabel = new QLabel(tr("Ready to render"), progressGroup);
	m_statusLabel->setAlignment(Qt::AlignCenter);

	// Caveat line below the main status (e.g. a succeeded render whose
	// preview image couldn't be shown) - see its own comment in mainwindow.h.
	// Hidden by default so it costs no layout space until there's something
	// to say; setStatusWarning()/clearStatusWarning() show/hide it.
	m_statusWarningLabel = new QLabel(progressGroup);
	m_statusWarningLabel->setObjectName("statusWarning");
	m_statusWarningLabel->setAlignment(Qt::AlignCenter);
	m_statusWarningLabel->setWordWrap(true);
	m_statusWarningLabel->hide();

	progressLayout->addWidget(m_currentJobLabel);
	progressLayout->addWidget(m_progressBar);
	progressLayout->addWidget(m_statusLabel);
	progressLayout->addWidget(m_statusWarningLabel);
	layout->addWidget(progressGroup);

	// Render queue - stays visible even when empty (unlike when this lived
	// outside the tabs alongside other always-on controls, hiding it here
	// would leave the Progress tab looking like it lost a section every
	// time the queue drains, rather than like a stable panel).
	m_queueGroup = new InfoGroupBox(tr("Render Queue"), progressWidget);
	m_queueGroup->setInfoIcon(createInfoIcon(
		tr("Every render of this session, one row each: waiting, running, and what became of the finished ones. "
		"Clicking Render while something is already in progress adds another job here instead of interrupting it; "
		"waiting jobs start automatically, one after another, as each one finishes. Waiting jobs can be moved up or down, "
		"and a failed or cancelled one can be run again.")));
	QVBoxLayout *queueLayout = new QVBoxLayout(m_queueGroup);

	m_queueModel = new RenderQueueModel(
		[](const RenderJob &job) { return MainWindow::rendererLabel(job.useGPU, job.useWavefront); },
		[](const RenderJob &job) { return MainWindow::describeRenderJob(job); }, this);
	m_queueView = new QTableView(m_queueGroup);
	m_queueView->setModel(m_queueModel);
	m_queueView->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_queueView->setSelectionMode(QAbstractItemView::SingleSelection);
	m_queueView->setEditTriggers(QAbstractItemView::NoEditTriggers);
	m_queueView->setShowGrid(false);
	m_queueView->setMaximumHeight(170);
	m_queueView->setMinimumHeight(90);
	m_queueView->verticalHeader()->hide();
	m_queueView->verticalHeader()->setDefaultSectionSize(24);
	m_queueView->horizontalHeader()->setStretchLastSection(false);
	m_queueView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	m_queueView->horizontalHeader()->setSectionResizeMode(RenderQueueModel::SceneColumn, QHeaderView::Stretch);
	// Clicking empty space below the rows otherwise leaves whatever was
	// selected stuck selected - see ListEmptyAreaDeselectFilter's comment.
	m_queueView->viewport()->installEventFilter(new ListEmptyAreaDeselectFilter(m_queueView));
	queueLayout->addWidget(m_queueView);

	auto queueButton = [this](const QString &text, const QString &tip, void (MainWindow::*slot)()) {
		auto *b = new QPushButton(text, m_queueGroup);
		b->setToolTip(tip);
		connect(b, &QPushButton::clicked, this, slot);
		return b;
	};
	QHBoxLayout *queueButtonLayout = new QHBoxLayout();
	m_queueRemoveButton = queueButton(tr("Re&move Selected"), tr("Remove the selected job from the list (a running job is stopped with Stop)"), &MainWindow::onRemoveSelectedQueueItem);
	m_queueUpButton = queueButton(tr("Move &Up"), tr("Run the selected waiting job earlier"), &MainWindow::onMoveQueueItemUp);
	m_queueDownButton = queueButton(tr("Move &Down"), tr("Run the selected waiting job later"), &MainWindow::onMoveQueueItemDown);
	m_queueRetryButton = queueButton(tr("&Retry"), tr("Queue the selected failed or cancelled job again"), &MainWindow::onRetryQueueItem);
	m_queueClearFinishedButton = queueButton(tr("Clear &Finished"), tr("Forget the jobs that are done, failed or cancelled"), &MainWindow::onClearFinishedJobs);
	// Discards every waiting job at once (unlike m_queueRemoveButton above,
	// which only drops the one job you selected), so it gets the danger
	// styling too.
	m_queueClearButton = queueButton(tr("Clear &Queue"), tr("Remove every waiting job from the render queue"), &MainWindow::onClearQueue);
	m_queueClearButton->setObjectName("dangerAction");
	applyElevation(m_queueClearButton, /*blurRadius=*/14, /*offsetY=*/3, /*alpha=*/90);
	m_queueClearButton->installEventFilter(
		new HoverLiftFilter(m_queueClearButton, 14, 22, 6, /*idlePulse=*/false, m_queueClearButton));
	for (QPushButton *b : {m_queueRemoveButton, m_queueUpButton, m_queueDownButton, m_queueRetryButton, m_queueClearFinishedButton, m_queueClearButton}) queueButtonLayout->addWidget(b);
	queueLayout->addLayout(queueButtonLayout);
	connect(m_queueView->selectionModel(), &QItemSelectionModel::currentChanged, this, [this]() { updateQueueButtons(); });
	connect(m_queueModel, &RenderQueueModel::countsChanged, this, [this]() { refreshQueuePanel(); });
	updateQueueButtons();

	layout->addWidget(m_queueGroup);
	layout->addStretch(1);

	m_progressTabIndex = m_tabWidget->addTab(progressWidget, tr("Progress"));
}

namespace {
constexpr int kLogBasePt = 9;        // what both views were hardcoded to before this was adjustable
constexpr int kLogMinDeltaPt = -3;   // 6pt floor
constexpr int kLogMaxDeltaPt = 30;   // 39pt ceiling

// Ctrl/Cmd + scroll wheel over a log view steps its font size instead of
// QTextEdit's built-in zoom - that one changes the document's own default font
// (never persisted, and stacking on top of the size set below), so leaving it
// would give two competing "zoom" states. A trackpad delivers many small
// angleDelta values per gesture, so they're accumulated to one step per 120
// (a classic mouse wheel notch) rather than stepping on every event.
class CtrlWheelFontStep : public QObject {
public:
	CtrlWheelFontStep(std::function<void(int)> onStep, QObject *parent)
		: QObject(parent), m_onStep(std::move(onStep)) {}
protected:
	bool eventFilter(QObject *watched, QEvent *event) override {
		if (event->type() != QEvent::Wheel) return QObject::eventFilter(watched, event);
		auto *wheel = static_cast<QWheelEvent *>(event);
		if (!(wheel->modifiers() & Qt::ControlModifier)) return false;
		m_accum += wheel->angleDelta().y();
		while (m_accum >= 120)  { m_accum -= 120; m_onStep(+1); }
		while (m_accum <= -120) { m_accum += 120; m_onStep(-1); }
		return true;  // consumed, even while accumulating - never falls through to QTextEdit's own zoom
	}
private:
	std::function<void(int)> m_onStep;
	int m_accum = 0;
};
}  // namespace

// A widget-level stylesheet rather than setFont(): the app-wide QTextEdit rule
// (mainwindow_style.cpp) sets font-size itself, and a QSS font-size beats a
// widget's own QFont size - which is why the old QFont("Consolas", 9) here was
// never actually what controlled the size either. This local rule wins over the
// app-wide one for just these two views, and survives applyTheme() re-setting
// the app stylesheet.
void MainWindow::applyLogFontSize() {
	const QString qss = QStringLiteral("QTextEdit { font-size: %1pt; }").arg(kLogBasePt + m_logFontDelta);
	if (m_logTextEdit) m_logTextEdit->setStyleSheet(qss);
	if (m_diagTextEdit) m_diagTextEdit->setStyleSheet(qss);
}

void MainWindow::changeLogFontSize(int deltaPoints) {
	const int next = std::clamp(m_logFontDelta + deltaPoints, kLogMinDeltaPt, kLogMaxDeltaPt);
	if (next == m_logFontDelta) return;
	m_logFontDelta = next;
	QSettings(settings_keys::kOrg, settings_keys::kApp).setValue(settings_keys::kLogFontDeltaKey, m_logFontDelta);
	applyLogFontSize();
	statusBar()->showMessage(tr("Log font size: %1 pt").arg(kLogBasePt + m_logFontDelta), 2000);
}

void MainWindow::resetLogFontSize() {
	changeLogFontSize(-m_logFontDelta);
}

void MainWindow::createLogTab() {
	QWidget *logWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(logWidget);

	// Log output text area
	m_logTextEdit = new QTextEdit();
	m_logTextEdit->setReadOnly(true);
	m_logTextEdit->setFont(QFont("Consolas", 9));
	m_logTextEdit->setLineWrapMode(QTextEdit::NoWrap);
	m_logFontDelta = std::clamp(
		QSettings(settings_keys::kOrg, settings_keys::kApp).value(settings_keys::kLogFontDeltaKey, 0).toInt(),
		kLogMinDeltaPt, kLogMaxDeltaPt);
	// Bounds the pane's own memory/reflow cost the same way m_logHistory is
	// bounded (see kMaxLogHistoryLines's own comment, mainwindow.h) - drops
	// oldest blocks automatically as new ones are appended past this count.
	m_logTextEdit->document()->setMaximumBlockCount(MainWindow::kMaxLogHistoryLines);
	// No stylesheet here: the global QTextEdit rule already supplies the
	// surface, border and radius, and this local copy only duplicated it.

	layout->addWidget(m_logTextEdit);
	m_logTextEdit->viewport()->installEventFilter(
		new CtrlWheelFontStep([this](int step) { changeLogFontSize(step); }, m_logTextEdit));

	// Button bar: Copy | Save Log | Clear Log
	QHBoxLayout *btnLayout = new QHBoxLayout();
	btnLayout->setContentsMargins(0, 4, 0, 0);

	// Geometry only - see previewBtnStyle's comment.
	QString logBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; min-width: 160px; padding: 0px 20px; font-size: 11pt; }";

	// These three share their implementation with the File menu's actions
	// (see createActions()), so the bodies live in slots rather than lambdas
	// here - otherwise the menu entry and the button would be two separate
	// copies of the same behaviour, free to drift apart.
	QPushButton *copyButton = new QPushButton(tr("&Copy All"));
	icon_tint::apply(copyButton, ":/icons/copy.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	copyButton->setStyleSheet(logBtnStyle);
	connect(copyButton, &QPushButton::clicked, this, &MainWindow::copyLogToClipboard);

	QPushButton *saveButton = new QPushButton(tr("&Save Log…"));
	icon_tint::apply(saveButton, ":/icons/save.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	saveButton->setStyleSheet(logBtnStyle);
	connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveLogToFile);

	QPushButton *clearButton = new QPushButton(tr("C&lear Log"));
	icon_tint::apply(clearButton, ":/icons/clear.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	clearButton->setStyleSheet(logBtnStyle);
	connect(clearButton, &QPushButton::clicked, this, &MainWindow::clearLog);

	btnLayout->addWidget(copyButton);
	btnLayout->addWidget(saveButton);
	btnLayout->addStretch();
	btnLayout->addWidget(clearButton);
	layout->addLayout(btnLayout);

	m_logTabIndex = m_tabWidget->addTab(logWidget, tr("Log Output"));
}

// Modeled directly on createLogTab() above: a read-only monospace text area
// (no QScrollArea wrapper - it scrolls itself) plus a button bar. "Run
// Diagnostics" launches ray_tracer.exe --diagnose via DiagnosticsRunner
// (see mainwindow.h/.cpp) exactly the way RenderController launches a
// render, just without any progress parsing - it's a one-shot report.
void MainWindow::createDiagnosticsTab() {
	QWidget *diagWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(diagWidget);

	m_diagTextEdit = new QTextEdit();
	m_diagTextEdit->setReadOnly(true);
	m_diagTextEdit->setFont(QFont("Consolas", 9));
	m_diagTextEdit->setLineWrapMode(QTextEdit::NoWrap);
	m_diagTextEdit->setPlaceholderText(
		tr("Click \"Run Diagnostics\" to check GPU/CUDA/OptiX availability, CPU/RAM, "
		"disk space, and scene asset availability."));
	layout->addWidget(m_diagTextEdit);
	m_diagTextEdit->viewport()->installEventFilter(
		new CtrlWheelFontStep([this](int step) { changeLogFontSize(step); }, m_diagTextEdit));
	applyLogFontSize();  // both views exist now; the log tab is created first so its delta is already loaded

	QHBoxLayout *btnLayout = new QHBoxLayout();
	btnLayout->setContentsMargins(0, 4, 0, 0);

	// Same geometry-only style as the Log tab's own button bar.
	QString diagBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; min-width: 160px; padding: 0px 20px; font-size: 11pt; }";

	m_runDiagnosticsButton = new QPushButton(tr("&Run Diagnostics"));
	icon_tint::apply(m_runDiagnosticsButton, ":/icons/gpu.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	m_runDiagnosticsButton->setStyleSheet(diagBtnStyle);
	connect(m_runDiagnosticsButton, &QPushButton::clicked, this, &MainWindow::onRunDiagnosticsClicked);

	// Enabled by updateInstallPhotoButton() once a report shows the optional photo helper is missing something.
	m_installPhotoHelperButton = new QPushButton(tr("&Install Photo Helper..."));
	icon_tint::apply(m_installPhotoHelperButton, ":/icons/save.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	m_installPhotoHelperButton->setStyleSheet(diagBtnStyle);
	connect(m_installPhotoHelperButton, &QPushButton::clicked, this, &MainWindow::onInstallPhotoHelperClicked);
	updateInstallPhotoButton();

	QPushButton *copyButton = new QPushButton(tr("&Copy All"));
	icon_tint::apply(copyButton, ":/icons/copy.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	copyButton->setStyleSheet(diagBtnStyle);
	connect(copyButton, &QPushButton::clicked, this, &MainWindow::copyDiagToClipboard);

	QPushButton *saveButton = new QPushButton(tr("&Save Report…"));
	icon_tint::apply(saveButton, ":/icons/save.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	saveButton->setStyleSheet(diagBtnStyle);
	connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveDiagReportToFile);

	btnLayout->addWidget(m_runDiagnosticsButton);
	btnLayout->addWidget(m_installPhotoHelperButton);
	btnLayout->addStretch();
	btnLayout->addWidget(copyButton);
	btnLayout->addWidget(saveButton);
	layout->addLayout(btnLayout);

	m_diagnosticsTabIndex = m_tabWidget->addTab(diagWidget, tr("Diagnostics"));
}

