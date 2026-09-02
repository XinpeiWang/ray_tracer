// Progress, Log, and Diagnostics tabs - split out of mainwindow_tabs.cpp;
// see mainwindow_tabs_render.cpp for the Render Options/Preview/Video tabs
// this file's content used to sit between, in both the old file's function
// order and the tab bar's own left-to-right order.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"

#include "../src/shared/scene_descriptor.h"
#include "../src/shared/video_preset.h"

#include <QTabBar>
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
#include <cmath>
#include <algorithm>

void MainWindow::createProgressTab() {
	QWidget *progressWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(progressWidget);

	QGroupBox *progressGroup = new QGroupBox(tr("Progress"), progressWidget);
	QVBoxLayout *progressLayout = new QVBoxLayout(progressGroup);

	// Which job is actually running (scene/resolution/samples/renderer, same
	// one-line format as a queue row) - set from m_currentJob in
	// startRenderJob(), not the live Basic Settings form, since the user may
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
	m_queueGroup = new QGroupBox(tr("Render Queue"), progressWidget);
	QVBoxLayout *queueLayout = new QVBoxLayout(m_queueGroup);

	m_queueListWidget = new QListWidget(m_queueGroup);
	m_queueListWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	m_queueListWidget->setMaximumHeight(120);
	// Clicking empty space below the rows otherwise leaves whatever was
	// selected stuck selected - see ListEmptyAreaDeselectFilter's comment.
	m_queueListWidget->viewport()->installEventFilter(new ListEmptyAreaDeselectFilter(m_queueListWidget));
	queueLayout->addWidget(m_queueListWidget);

	QHBoxLayout *queueButtonLayout = new QHBoxLayout();
	QPushButton *removeQueueItemButton = new QPushButton(tr("Re&move Selected"), m_queueGroup);
	removeQueueItemButton->setToolTip(tr("Remove the selected job from the render queue"));
	connect(removeQueueItemButton, &QPushButton::clicked, this, &MainWindow::onRemoveSelectedQueueItem);
	// Discards every queued job at once (unlike removeQueueItemButton above,
	// which only drops the one job you selected), so it gets the danger
	// styling too.
	QPushButton *clearQueueButton = new QPushButton(tr("Clear &Queue"), m_queueGroup);
	clearQueueButton->setObjectName("dangerAction");
	clearQueueButton->setToolTip(tr("Remove every job from the render queue"));
	connect(clearQueueButton, &QPushButton::clicked, this, &MainWindow::onClearQueue);
	applyElevation(clearQueueButton, /*blurRadius=*/14, /*offsetY=*/3, /*alpha=*/90);
	clearQueueButton->installEventFilter(
		new HoverLiftFilter(clearQueueButton, 14, 22, 6, /*idlePulse=*/false, clearQueueButton));
	queueButtonLayout->addWidget(removeQueueItemButton);
	queueButtonLayout->addWidget(clearQueueButton);
	queueLayout->addLayout(queueButtonLayout);

	layout->addWidget(m_queueGroup);
	layout->addStretch(1);

	m_progressTabIndex = m_tabWidget->addTab(progressWidget, tr("Progress"));
}

void MainWindow::createLogTab() {
	QWidget *logWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(logWidget);

	// Log output text area
	m_logTextEdit = new QTextEdit();
	m_logTextEdit->setReadOnly(true);
	m_logTextEdit->setFont(QFont("Consolas", 9));
	m_logTextEdit->setLineWrapMode(QTextEdit::NoWrap);
	// No stylesheet here: the global QTextEdit rule already supplies the
	// surface, border and radius, and this local copy only duplicated it.

	layout->addWidget(m_logTextEdit);

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

	QHBoxLayout *btnLayout = new QHBoxLayout();
	btnLayout->setContentsMargins(0, 4, 0, 0);

	// Same geometry-only style as the Log tab's own button bar.
	QString diagBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; min-width: 160px; padding: 0px 20px; font-size: 11pt; }";

	m_runDiagnosticsButton = new QPushButton(tr("&Run Diagnostics"));
	icon_tint::apply(m_runDiagnosticsButton, ":/icons/gpu.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	m_runDiagnosticsButton->setStyleSheet(diagBtnStyle);
	connect(m_runDiagnosticsButton, &QPushButton::clicked, this, &MainWindow::onRunDiagnosticsClicked);

	QPushButton *copyButton = new QPushButton(tr("&Copy All"));
	icon_tint::apply(copyButton, ":/icons/copy.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	copyButton->setStyleSheet(diagBtnStyle);
	connect(copyButton, &QPushButton::clicked, this, &MainWindow::copyDiagToClipboard);

	QPushButton *saveButton = new QPushButton(tr("&Save Report…"));
	icon_tint::apply(saveButton, ":/icons/save.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	saveButton->setStyleSheet(diagBtnStyle);
	connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveDiagReportToFile);

	btnLayout->addWidget(m_runDiagnosticsButton);
	btnLayout->addStretch();
	btnLayout->addWidget(copyButton);
	btnLayout->addWidget(saveButton);
	layout->addLayout(btnLayout);

	m_diagnosticsTabIndex = m_tabWidget->addTab(diagWidget, tr("Diagnostics"));
}

