#include "mainwindow.h"
#include "icon_tint.h"

#include <QApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QIcon>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QTextStream>
#include <QUrl>

// ============================================================================
// Actions, menus and status bar
// ============================================================================
// Shortcut choices follow the two traditions this app straddles - Blender on
// the render side and Qt Creator / VS Code on the long-running-job side:
//
//   F12          render (Blender's Render Image)
//   Ctrl+R       render (Qt Creator's Run)
//   Ctrl+F12     render animation (Blender's Render Animation)
//   Esc          cancel the running render
//
// F12 and Ctrl+R are both bound to the same action, which Qt supports
// directly via setShortcuts() - there's no reason to make users of either
// tradition learn the other's key.
//
// Esc deliberately does NOT use QKeySequence::Cancel unconditionally: a
// QMainWindow ignores Esc by default (unlike QDialog), so binding it makes it
// a real key, and a key that silently does nothing most of the time is worse
// than no key at all. It is therefore enabled only while a render is running
// (see updateActionStates()).
//
// Plain Return is deliberately NOT bound to render. This window is a form
// full of spin boxes; a default button would fire on any stray Return while
// editing a value.
// ============================================================================

void MainWindow::createActions() {
	constexpr auto kBody = icon_tint::Role::Body;

	m_actRender = new QAction("&Render Image", this);
	icon_tint::apply(m_actRender, ":/icons/render.svg", kBody, m_activeTheme.textBody);
	m_actRender->setShortcuts({QKeySequence(Qt::Key_F12), QKeySequence("Ctrl+R")});
	m_actRender->setStatusTip("Render the selected scene with the current settings");
	connect(m_actRender, &QAction::triggered, this, [this]() {
		if (m_modeCombo && m_modeCombo->currentIndex() != 0)
			m_modeCombo->setCurrentIndex(0);   // single image
		onRenderClicked();
	});

	m_actRenderVideo = new QAction("Render &Video", this);
	icon_tint::apply(m_actRenderVideo, ":/icons/video.svg", kBody, m_activeTheme.textBody);
	m_actRenderVideo->setShortcut(QKeySequence("Ctrl+F12"));
	m_actRenderVideo->setStatusTip("Render the camera path frame by frame and assemble a video");
	connect(m_actRenderVideo, &QAction::triggered, this, [this]() {
		if (m_modeCombo && m_modeCombo->currentIndex() != 1)
			m_modeCombo->setCurrentIndex(1);   // video
		onRenderClicked();
	});

	m_actStop = new QAction("&Stop Render", this);
	icon_tint::apply(m_actStop, ":/icons/stop.svg", kBody, m_activeTheme.textBody);
	m_actStop->setShortcut(QKeySequence(Qt::Key_Escape));
	m_actStop->setShortcutContext(Qt::WindowShortcut);
	m_actStop->setStatusTip("Stop the running render and discard its output");
	connect(m_actStop, &QAction::triggered, this, &MainWindow::onStopClicked);

	m_actOpenFolder = new QAction("Open Output &Folder", this);
	icon_tint::apply(m_actOpenFolder, ":/icons/folder.svg", kBody, m_activeTheme.textBody);
	m_actOpenFolder->setShortcut(QKeySequence("Ctrl+Shift+O"));
	m_actOpenFolder->setStatusTip("Show the folder containing the last render");
	connect(m_actOpenFolder, &QAction::triggered, this, [this]() {
		if (m_lastOutputPath.isEmpty()) return;
		QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(m_lastOutputPath).absolutePath()));
	});

	m_actOpenViewer = new QAction("Open in Default &Viewer", this);
	icon_tint::apply(m_actOpenViewer, ":/icons/image.svg", kBody, m_activeTheme.textBody);
	m_actOpenViewer->setStatusTip("Open the rendered image in the system image viewer");
	connect(m_actOpenViewer, &QAction::triggered, this, [this]() {
		if (m_lastPreviewImagePath.isEmpty()) return;
		QDesktopServices::openUrl(QUrl::fromLocalFile(m_lastPreviewImagePath));
	});

	m_actCopyLog = new QAction("&Copy Log", this);
	icon_tint::apply(m_actCopyLog, ":/icons/copy.svg", kBody, m_activeTheme.textBody);
	m_actCopyLog->setShortcut(QKeySequence("Ctrl+Shift+C"));   // Ctrl+C stays "copy selection"
	m_actCopyLog->setStatusTip("Copy the entire log to the clipboard");
	connect(m_actCopyLog, &QAction::triggered, this, &MainWindow::copyLogToClipboard);

	m_actSaveLog = new QAction("&Save Log…", this);
	icon_tint::apply(m_actSaveLog, ":/icons/save.svg", kBody, m_activeTheme.textBody);
	// Deliberately NOT QKeySequence::Save. In a renderer, Ctrl+S reads as
	// "save the image", and handing someone a log-save dialog when they meant
	// their render is worse than having no shortcut. Renders already write
	// themselves to the output path, so there is no save-image command for
	// Ctrl+S to mean - leaving it unbound is the honest option.
	m_actSaveLog->setShortcut(QKeySequence("Ctrl+Shift+S"));
	m_actSaveLog->setStatusTip("Write the log to a text file");
	connect(m_actSaveLog, &QAction::triggered, this, &MainWindow::saveLogToFile);

	m_actClearLog = new QAction("C&lear Log", this);
	icon_tint::apply(m_actClearLog, ":/icons/clear.svg", kBody, m_activeTheme.textBody);
	m_actClearLog->setShortcut(QKeySequence("Ctrl+L"));        // terminal convention
	m_actClearLog->setStatusTip("Clear the log pane");
	connect(m_actClearLog, &QAction::triggered, this, &MainWindow::clearLog);

	m_actAbout = new QAction("&About Ray Tracer", this);
	m_actAbout->setShortcut(QKeySequence::HelpContents);       // F1
	m_actAbout->setStatusTip("Version and project information");
	connect(m_actAbout, &QAction::triggered, this, &MainWindow::showAboutDialog);

	m_actAboutQt = new QAction("About &Qt", this);
	connect(m_actAboutQt, &QAction::triggered, qApp, &QApplication::aboutQt);

	m_actQuit = new QAction("&Quit", this);
	m_actQuit->setShortcut(QKeySequence::Quit);
	m_actQuit->setStatusTip("Exit the application");
	connect(m_actQuit, &QAction::triggered, this, &QWidget::close);

	updateActionStates();
}

void MainWindow::createMenus() {
	QMenu *fileMenu = menuBar()->addMenu("&File");
	fileMenu->addAction(m_actOpenFolder);
	fileMenu->addAction(m_actOpenViewer);
	fileMenu->addSeparator();
	fileMenu->addAction(m_actSaveLog);
	fileMenu->addSeparator();
	fileMenu->addAction(m_actQuit);

	QMenu *renderMenu = menuBar()->addMenu("&Render");
	renderMenu->addAction(m_actRender);
	renderMenu->addAction(m_actRenderVideo);
	renderMenu->addSeparator();
	renderMenu->addAction(m_actStop);

	QMenu *viewMenu = menuBar()->addMenu("&View");
	// Tab switching by name, so the menu documents the shortcuts rather than
	// leaving Ctrl+1..5 as undiscoverable trivia.
	for (int i = 0; i < m_tabWidget->count(); ++i) {
		QAction *tabAction = viewMenu->addAction(m_tabWidget->tabText(i));
		tabAction->setShortcut(QKeySequence(QString("Ctrl+%1").arg(i + 1)));
		connect(tabAction, &QAction::triggered, this, [this, i]() {
			m_tabWidget->setCurrentIndex(i);
		});
	}
	viewMenu->addSeparator();
	createThemeMenu(viewMenu);
	viewMenu->addSeparator();
	viewMenu->addAction(m_actCopyLog);
	viewMenu->addAction(m_actClearLog);

	QMenu *helpMenu = menuBar()->addMenu("&Help");
	helpMenu->addAction(m_actAbout);
	helpMenu->addAction(m_actAboutQt);
}

void MainWindow::createStatusBar() {
	// Permanent widgets only - these are ambient mode indicators, which is
	// exactly what Qt documents addPermanentWidget() for. Transient text
	// (action status tips) uses the same bar automatically.
	m_statusDevice = new QLabel(this);
	m_statusSettings = new QLabel(this);
	m_statusDevice->setObjectName("statusInfo");
	m_statusSettings->setObjectName("statusInfo");

	statusBar()->addPermanentWidget(m_statusSettings);
	statusBar()->addPermanentWidget(m_statusDevice);
	refreshStatusBarInfo();
}

void MainWindow::refreshStatusBarInfo() {
	if (!m_statusDevice || !m_statusSettings) return;
	const bool useGPU = m_renderModeCombo && m_renderModeCombo->currentData().toBool();
	m_statusDevice->setText(useGPU ? "GPU (OptiX)" : "CPU");
	if (m_widthSpinBox && m_heightSpinBox && m_samplesSpinBox) {
		m_statusSettings->setText(QString("%1x%2  ·  %3 spp")
			.arg(m_widthSpinBox->value())
			.arg(m_heightSpinBox->value())
			.arg(m_samplesSpinBox->value()));
	}
}

void MainWindow::updateActionStates() {
	if (m_actRender)      m_actRender->setEnabled(!m_isRendering);
	if (m_actRenderVideo) m_actRenderVideo->setEnabled(!m_isRendering);
	// Esc only exists while there's something to cancel - see the header
	// comment in this file.
	if (m_actStop)        m_actStop->setEnabled(m_isRendering);
	if (m_actOpenFolder)  m_actOpenFolder->setEnabled(!m_lastOutputPath.isEmpty());
	if (m_actOpenViewer)  m_actOpenViewer->setEnabled(!m_lastPreviewImagePath.isEmpty());
}

// ----------------------------------------------------------------------------
// Log commands - shared by the log tab's buttons and the menu actions
// ----------------------------------------------------------------------------

void MainWindow::copyLogToClipboard() {
	if (!m_logTextEdit) return;
	m_logTextEdit->selectAll();
	m_logTextEdit->copy();
	m_logTextEdit->moveCursor(QTextCursor::End);
}

void MainWindow::saveLogToFile() {
	if (!m_logTextEdit) return;
	const QString path = QFileDialog::getSaveFileName(this, "Save Log",
		QDir::homePath() + "/render_log.txt",
		"Text Files (*.txt);;All Files (*.*)");
	if (path.isEmpty()) return;

	QFile file(path);
	if (!file.open(QFile::WriteOnly | QFile::Text)) {
		// Previously a failed open was silently ignored, which looked
		// identical to a successful save.
		onLogMessage(QString("[ERROR] Could not write log to %1: %2")
			.arg(path, file.errorString()));
		return;
	}
	QTextStream out(&file);
	out << m_logTextEdit->toPlainText();
	file.close();
	onLogMessage(QString("[INFO] Log saved to %1").arg(path));
}

void MainWindow::clearLog() {
	if (m_logTextEdit) m_logTextEdit->clear();
}

void MainWindow::showAboutDialog() {
	QMessageBox::about(this, "About Ray Tracer",
		"<h3>Ray Tracer</h3>"
		"<p>A physically-based path tracer with parallel CPU and GPU (OptiX) "
		"backends, built up from the <i>Ray Tracing in One Weekend</i> series "
		"into a pbrt-v4-style feature set.</p>"
		"<p>65 scenes, a wide BxDF library, multiple light and camera types, "
		"triangle-mesh and texture support, BVH acceleration, volumetrics, and "
		"an SPPM photon-mapping integrator alongside standard path tracing.</p>"
		"<p>This window drives <code>ray_tracer.exe</code> as a subprocess.</p>");
}
