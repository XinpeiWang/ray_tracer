#include "mainwindow.h"
#include "icon_tint.h"
#include "error_handler.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QStyleFactory>
#include <QStyle>
#include <QPalette>
#include <QProcess>
#include <QDir>
#include <QTime>
#include <QDateTime>
#include <QRegularExpression>
#include <QDesktopServices>
#include <QUrl>
#include <QFont>
#include <QFontInfo>
#include <QAbstractItemView>
#include <QPainter>
#include <QPolygon>
#include <QIcon>
#include <QPixmap>
#include <QScrollArea>
#include <QScreen>
#include <QTimer>

// RenderController Implementation
RenderController::RenderController(QObject *parent)
	: QObject(parent), m_useGPU(true), m_width(800), m_height(800), m_samples(100), m_maxDepth(50),
	  m_sceneId(0), m_camX(278), m_camY(278), m_camZ(-800), m_camExplicit(false),
	  m_videoMode(false), m_videoFrames(60), m_videoFPS(30), m_videoSpeed(1.0), m_cameraPath("orbit") {
}

void RenderController::setParameters(bool useGPU, int width, int height, int samples, int maxDepth,
								  int sceneId, double camX, double camY, double camZ, bool camExplicit,
								  const QString &outputPath) {
	m_useGPU = useGPU;
	m_width = width;
	m_height = height;
	m_samples = samples;
	m_maxDepth = maxDepth;
	m_sceneId = sceneId;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_camExplicit = camExplicit;
	m_outputPath = outputPath;
}

void RenderController::setVideoParameters(bool enabled, int frames, int fps, const QString &cameraPath, double speed) {
	m_videoMode = enabled;
	m_videoFrames = frames;
	m_videoFPS = fps;
	m_videoSpeed = speed;
	m_cameraPath = cameraPath;
}

bool RenderController::isRunning() const {
	return m_renderProcess && m_renderProcess->state() != QProcess::NotRunning;
}

void RenderController::stopRender() {
	if (!isRunning()) return;
	emit logMessage("Stopping render...");
	m_stopRequested = true;
	// Everything lives on the GUI thread now, so the process can be killed
	// directly here - onProcessFinished() will pick it up from the event loop.
	m_renderProcess->kill();
}

void RenderController::start() {
	emit logMessage(QString("Starting render..."));

	// Look up scene name live from scene_metadata.dll
	QString sceneName = SceneMetadataClient::sceneName(m_sceneId);
	if (sceneName.isEmpty()) sceneName = QString("Scene %1").arg(m_sceneId);

	// Emit a separator so each render is visually distinct in the log
	QString sep = QString("─").repeated(60);
	emit logMessage(sep);
	emit logMessage(QString("▶ RENDER START  %1")
		.arg(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss")));
	emit logMessage(QString("  Scene: %1 (%2) | %3x%4 | %5 spp | depth %6 | %7")
		.arg(m_sceneId)
		.arg(sceneName)
		.arg(m_width).arg(m_height)
		.arg(m_samples)
		.arg(m_maxDepth)
		.arg(m_useGPU ? "GPU" : "CPU"));
	emit logMessage(sep);

	// ========================================================================
	// Build Command Line for ray_tracer.exe
	// ========================================================================
	// The ray_tracer.exe is the unified launcher that calls CPU or GPU renderer
	// Command format: ray_tracer.exe [--gpu|--cpu] [--output path] width samples depth cam_x cam_y cam_z
	//
	// Camera parameters:
	//   - cam_x, cam_y, cam_z: Camera position (lookfrom) in world space
	//   - lookat is always (278, 278, 278) - fixed in renderer code
	//   - These are passed to both CPU and GPU renderers through their interfaces
	// ========================================================================

	QString exePath = QCoreApplication::applicationDirPath() + "/ray_tracer.exe";
	QStringList args;

	// Render mode flag: --gpu or --cpu
	if (m_useGPU) {
		args << "--gpu";
	} else {
		args << "--cpu";
	}

	// Video mode flags (if enabled)
	if (m_videoMode) {
		args << "--video";
		args << "--frames" << QString::number(m_videoFrames);
		args << "--fps" << QString::number(m_videoFPS);
		args << "--speed" << QString::number(m_videoSpeed);
		args << "--camera-path" << m_cameraPath;

		// In video mode, explicitly set output path to ensure frames go to the right directory
		// The launcher will create frames in <output_dir>/frames/
		QString videoOutputPath = QCoreApplication::applicationDirPath() + "/output/video.ppm";
		args << "--output" << videoOutputPath;
	} else {
		// Image mode: use custom output path if provided
		if (!m_outputPath.isEmpty()) {
			args << "--output" << m_outputPath;
		}
	}

	// Numeric positional arguments (order matters!):
	// 1. width: image width in pixels (height = width for square aspect ratio)
	// 2. samples: samples per pixel for anti-aliasing and noise reduction
	// 3. depth: maximum ray bounce depth for recursive ray tracing
	// 4. scene_id: scene selector (0=Cornell Box, 1=Bouncing Spheres, etc.)
	// 5. cam_x: camera X position (lookfrom X coordinate) - direct in image mode, path start in video mode
	// 6. cam_y: camera Y position (lookfrom Y coordinate) - direct in image mode, path start in video mode
	// 7. cam_z: camera Z position (lookfrom Z coordinate) - direct in image mode, path start in video mode
	args << QString::number(m_width);
	args << QString::number(m_samples);
	args << QString::number(m_maxDepth);

	// Scene ID applies in both modes - dropping it in video mode used to
	// silently fall back to the launcher's default (scene 0/Cornell Box)
	// regardless of what the user picked in the Scene dropdown.
	args << QString::number(m_sceneId);

	// Camera position is only sent when it's explicit (see setParameters()'s
	// comment) - when the user hasn't touched it, omitting it lets
	// ray_tracer.exe fall back to the scene's own recommended camera
	// (main.cpp, driven by cam_explicit) exactly as a bare CLI invocation
	// would, instead of this GUI forcing every render through the "explicit
	// camera" path regardless of whether the user asked for one. When
	// explicit, it applies in both modes: single-image uses it directly,
	// and video mode uses it as the camera path's starting point (see
	// main.cpp's cam_explicit override of path_lookfrom) so that adjusting
	// X/Y/Z or "Distance from Center" in the GUI actually changes the
	// video, not just the single-image preview.
	if (m_camExplicit) {
		args << QString::number(m_camX);     // Camera position X
		args << QString::number(m_camY);     // Camera position Y
		args << QString::number(m_camZ);     // Camera position Z
	}

	emit logMessage(QString("Command: %1 %2").arg(exePath, args.join(" ")));

	// ========================================================================
	// Launch ray_tracer.exe as Subprocess
	// ========================================================================
	m_stopRequested = false;
	m_finished = false;
	m_lastProgress = 0;
	m_lineBuffer.clear();
	m_elapsed.start();

	m_renderProcess = new QProcess(this);
	m_renderProcess->setProcessChannelMode(QProcess::MergedChannels);

	// Set working directory to the application directory (RayTracer_Package)
	// This ensures output/frames directories are created in the correct location
	m_renderProcess->setWorkingDirectory(QCoreApplication::applicationDirPath());

	connect(m_renderProcess, &QProcess::readyReadStandardOutput,
			this, &RenderController::onReadyRead);
	connect(m_renderProcess, &QProcess::errorOccurred,
			this, &RenderController::onProcessErrorOccurred);
	connect(m_renderProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
			this, &RenderController::onProcessFinished);

	m_renderProcess->start(exePath, args);
}

void RenderController::onReadyRead() {
	// ray_tracer.exe's console output (box-drawing banners, checkmarks) is
	// UTF-8 - fromLocal8Bit() would decode it against the system ANSI
	// codepage instead, mangling every non-ASCII character (e.g. "─" ->
	// "â”€"). QProcess pipes raw bytes with no codepage translation, so the
	// bytes read here are exactly what the child process wrote.
	handleOutputChunk(QString::fromUtf8(m_renderProcess->readAll()));
}

void RenderController::handleOutputChunk(const QString &chunk) {
	m_lineBuffer += chunk;

	// The renderer writes its in-place progress counter with a bare '\r' and
	// no newline, so '\r' has to count as a line terminator here - waiting
	// for '\n' would stall progress updates until the next real line.
	QStringList parts = m_lineBuffer.split(QRegularExpression("[\r\n]"));

	// Whatever follows the last terminator is an incomplete line: hold it
	// back until the rest of it arrives in a later chunk.
	m_lineBuffer = parts.takeLast();

	for (const QString &part : parts) {
		QString line = part.trimmed();
		if (line.isEmpty()) continue;
		emit logMessage(line);
		parseProgressLine(line);
	}
}

void RenderController::parseProgressLine(const QString &line) {
	// The actual matching lives in render_output_parser.h so it can be unit
	// tested against real captured renderer output - see
	// tests/unit/render_output_parser_tests.cpp. Doing it here meant the
	// wording of ray_tracer.exe's std::cout calls was an untested protocol,
	// and a reworded line silently froze the progress bar.
	const std::string raw = line.toStdString();

	// Video mode: rendering (0-95%), then a fixed bump once ffmpeg assembly
	// starts (no per-frame progress available from ffmpeg cheaply);
	// onRenderComplete snaps to 100% once the subprocess actually exits.
	if (m_videoMode) {
		const int framePct = render_output::parseVideoFrameProgress(raw);
		if (framePct != render_output::kNoProgress && framePct > m_lastProgress) {
			emit progressUpdate(framePct);
			m_lastProgress = framePct;
		}
		if (render_output::isVideoAssemblyStart(raw) && m_lastProgress < 97) {
			emit progressUpdate(97);
			m_lastProgress = 97;
		}
		return;
	}

	const int pct = render_output::parseScanlineProgress(raw, m_height);
	if (pct != render_output::kNoProgress && pct > m_lastProgress) {
		emit progressUpdate(pct);   // only ever moves forward
		m_lastProgress = pct;
	}
}

void RenderController::onProcessErrorOccurred(QProcess::ProcessError error) {
	// Only FailedToStart is terminal on its own - every other error is
	// followed by finished(), which does the reporting.
	if (error != QProcess::FailedToStart || m_finished) return;
	finish(false, QString("Failed to start renderer: %1").arg(m_renderProcess->errorString()), QString());
}

void RenderController::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
	if (m_finished) return;

	const double totalTime = m_elapsed.elapsed() / 1000.0;

	// Drain whatever is left, then flush any final partial line.
	handleOutputChunk(QString::fromUtf8(m_renderProcess->readAll()));
	QString finalOutput = m_lineBuffer.trimmed();
	if (!finalOutput.isEmpty()) {
		emit logMessage(finalOutput);
		m_lineBuffer.clear();
	}

	// Structured finish footer
	emit logMessage(QString("─").repeated(60));

	// Format elapsed time as mm:ss or Xs depending on magnitude
	QString elapsedStr;
	if (totalTime >= 60.0)
		elapsedStr = QString("%1m %2s").arg((int)totalTime / 60).arg((int)totalTime % 60);
	else
		elapsedStr = QString("%1 ms").arg((int)(totalTime * 1000));

	emit logMessage(QString("Process finished: exit=%1  status=%2  time=%3")
		.arg(exitCode)
		.arg(exitStatus == QProcess::NormalExit ? "Normal" : "Crash")
		.arg(elapsedStr));

	if (m_stopRequested) {
		emit logMessage("Result: STOPPED BY USER");
		finish(false, "Render stopped by user", QString());
	} else if (exitCode == 0) {
		// Determine actual output path (default if not specified)
		QString actualOutputPath = m_outputPath;
		if (actualOutputPath.isEmpty()) {
			actualOutputPath = QCoreApplication::applicationDirPath() + "/output/image.png";
		} else {
			// If user specified .ppm, the renderer also creates .png
			if (actualOutputPath.endsWith(".ppm", Qt::CaseInsensitive)) {
				actualOutputPath.replace(actualOutputPath.length() - 4, 4, ".png");
			}
		}

		emit logMessage(QString("Result: SUCCESS  |  Output: %1").arg(actualOutputPath));
		emit progressUpdate(100);
		finish(true, "Render completed successfully!", actualOutputPath);
	} else {
		// Render failed with specific error code
		emit logMessage(QString("Result: FAILED (exit code %1)").arg(exitCode));

		QString errorTitle = ErrorHandler::getErrorTitle(exitCode);
		QString errorMessage = ErrorHandler::getErrorMessage(exitCode);
		QString hint = ErrorHandler::getTroubleshootingHint(exitCode);
		QString category = ErrorHandler::getCategoryName(exitCode);

		emit logMessage(QString("=== ERROR DETAILS ==="));
		emit logMessage(QString("Category: %1").arg(category));
		emit logMessage(QString("Error: %1").arg(errorTitle));
		emit logMessage(QString("Message: %1").arg(errorMessage));
		if (!hint.isEmpty() && hint != "Check the Log Output tab for detailed error information.") {
			emit logMessage(QString("Troubleshooting:\n%1").arg(hint));
		}

		// Build comprehensive error message for user
		QString fullErrorMsg = QString("<b>%1</b><br><br>%2").arg(errorTitle, errorMessage);
		if (!hint.isEmpty() && hint != "Check the Log Output tab for detailed error information.") {
			fullErrorMsg += QString("<br><br><b>Troubleshooting:</b><br>%1").arg(hint.replace("\n", "<br>"));
		}
		fullErrorMsg += QString("<br><br><small>Error Code: %1 | Category: %2</small>").arg(exitCode).arg(category);
		if (!finalOutput.isEmpty()) {
			fullErrorMsg += "\n\nOutput:\n" + finalOutput;
		}
		finish(false, fullErrorMsg, QString());
	}
}

void RenderController::finish(bool success, const QString &message, const QString &outputPath) {
	m_finished = true;

	const double totalTime = m_elapsed.elapsed() / 1000.0;

	if (m_renderProcess) {
		m_renderProcess->disconnect(this);
		m_renderProcess->deleteLater();
		m_renderProcess = nullptr;
	}

	emit renderComplete(success, message, totalTime, outputPath);
}

// MainWindow Implementation
MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), m_renderController(nullptr), m_isRendering(false), m_videoMode(false),
	  m_elapsedTimer(nullptr) {

	// Create shared wheel filter (blocks accidental scroll on all controls)
	m_wheelFilter = new WheelIgnoreFilter(this);

	setWindowTitle("Ray Tracer - Path Tracing Renderer");
	setMinimumSize(600, 500);

	// Auto-adjust initial size based on screen resolution
	QScreen *screen = QApplication::primaryScreen();
	if (screen) {
		QRect screenGeometry = screen->availableGeometry();
		int screenWidth = screenGeometry.width();
		int screenHeight = screenGeometry.height();

		// Use 50% of screen width, 70% of screen height (good balance for this UI)
		int initialWidth = qMin(1000, screenWidth * 50 / 100);
		int initialHeight = qMin(900, screenHeight * 70 / 100);

		resize(initialWidth, initialHeight);

		// Center the window on screen
		move(screenGeometry.center() - rect().center());
	} else {
		// Fallback if screen info unavailable
		resize(800, 700);
	}

	// The theme must exist before setupUI(), because createThemeMenu() ticks
	// the entry matching the active scheme.
	m_activeTheme = theme::byId(loadSavedThemeId());
	setupUI();
	applyTheme(m_activeTheme);
	restyleThemedWidgets();

	// Notification-only tray icon: the app has no tray menu and never hides
	// into the tray, this exists purely so a render finishing while the user
	// is in another window can say so. Skipped entirely where the platform
	// has no tray.
	if (QSystemTrayIcon::isSystemTrayAvailable()) {
		// A QSystemTrayIcon with a null icon does not become visible, and an
		// invisible tray icon silently swallows showMessage() - which is
		// exactly how this failed the first time round. windowIcon() is empty
		// this early (the .rc icon is an executable resource, not necessarily
		// the widget's), so fall back until something non-null is found.
		QIcon icon = windowIcon();
		if (icon.isNull()) icon = qApp->windowIcon();
		if (icon.isNull()) icon = style()->standardIcon(QStyle::SP_ComputerIcon);
		m_trayIcon = new QSystemTrayIcon(icon, this);
		m_trayIcon->setToolTip("Ray Tracer");
		// Shown for the app's whole lifetime. Showing it lazily instead (only
		// around a render, to avoid parking a do-nothing icon in the tray) was
		// tried and reverted, but NOT because it was proven broken: after the
		// change no notification arrived, and after reverting it none arrived
		// either, so the failure was environmental - Windows throttles
		// repeated toasts from the same app, which a burst of testing will
		// trigger. This configuration is kept only because it is the one with
		// positive evidence behind it (a delivered "Render complete" toast).
		// If the lazy variant is ever revisited, verify it on a fresh boot or
		// a different app identity, not immediately after other toasts.
		m_trayIcon->show();
		connect(m_trayIcon, &QSystemTrayIcon::activated, this,
				[this](QSystemTrayIcon::ActivationReason reason) {
			// An icon that ignores clicks is its own small annoyance.
			if (reason == QSystemTrayIcon::Trigger || reason == QSystemTrayIcon::DoubleClick) {
				showNormal();
				raise();
				activateWindow();
			}
		});
	}

	// The taskbar button only exists once the window has been realised, so
	// defer the COM setup to the event loop rather than doing it here.
	QTimer::singleShot(0, this, []() { win_taskbar::init(); });
}

MainWindow::~MainWindow() {
	if (m_renderController && m_renderController->isRunning()) {
		m_renderController->stopRender();
	}
	// A progress state left set outlives the window on the taskbar button,
	// so it must be cleared explicitly.
	win_taskbar::setState(this, win_taskbar::State::NoProgress);
	win_taskbar::shutdown();
}

void MainWindow::setupUI() {
	QWidget *centralWidget = new QWidget(this);
	QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

	// Create tab widget
	m_tabWidget = new QTabWidget(this);
	createBasicTab();
	createAdvancedTab();
	createVideoTab();
	createPreviewTab();
	createLogTab();

	// Initialize scene info AFTER tabs are created (onSceneChanged uses m_samplesSpinBox)
	onSceneChanged(0);

	// Actions/menus come after the tabs because the View menu enumerates the
	// tabs, and the status bar reads the renderer/size/samples widgets.
	createActions();
	createMenus();
	createStatusBar();

	// Keep the status bar's ambient readout honest when the user changes any
	// of the settings it reports.
	connect(m_renderModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, [this](int) { refreshStatusBarInfo(); });
	connect(m_widthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			this, [this](int) { refreshStatusBarInfo(); });
	connect(m_heightSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			this, [this](int) { refreshStatusBarInfo(); });
	connect(m_samplesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			this, [this](int) { refreshStatusBarInfo(); });

	mainLayout->addWidget(m_tabWidget);

	// Progress section
	QGroupBox *progressGroup = new QGroupBox("Progress", this);
	progressGroup->setContentsMargins(0, 16, 0, 0);
	QVBoxLayout *progressLayout = new QVBoxLayout(progressGroup);

	m_progressBar = new QProgressBar(this);
	m_progressBar->setRange(0, 100);
	m_progressBar->setValue(0);
	m_progressBar->setTextVisible(true);

	m_statusLabel = new QLabel("Ready to render", this);
	m_statusLabel->setAlignment(Qt::AlignCenter);

	progressLayout->addWidget(m_progressBar);
	progressLayout->addWidget(m_statusLabel);

	mainLayout->addWidget(progressGroup);

	// Render button.
	// Mnemonics ("&R" -> Alt+R) are assigned so every button is reachable
	// without the mouse. They're deliberately collision-free across the whole
	// window, including buttons that live on different tabs: R=Render,
	// T=sTop, B=Browse, C=Copy, S=Save, L=cLear, F=Folder, V=Viewer.
	// Icons come from resources.qrc rather than emoji in the label text.
	// A colour emoji carries its own palette, so it cannot be tinted to the
	// theme and stays fully saturated when the button is disabled; QIcon
	// derives a proper greyed variant automatically. Emoji also render
	// inconsistently across fonts and are read aloud by their CLDR name
	// ("wastebasket Clear Log, button") by screen readers.
	// The primary button's label is drawn in accentPrimary rather than body
	// colour, so its icon takes the same role - a body-coloured icon beside
	// accent-coloured bold text reads as orphaned. This used to be a second
	// hand-authored SVG with the accent baked in; tinting made it unnecessary.
	m_renderButton = new QPushButton("START &RENDER", this);
	icon_tint::apply(m_renderButton, ":/icons/render.svg",
	                 icon_tint::Role::Primary, m_activeTheme.accentPrimary);
	// Singles this out as the primary action in the stylesheet (2px accent
	// border + bold), so it isn't visually tied with every other button.
	m_renderButton->setObjectName("primaryAction");
	m_renderButton->setMinimumHeight(50);
	// The style's default 16px icon is dwarfed by a 50px-tall button with
	// 13pt bold text; 20px sits correctly against the cap height.
	m_renderButton->setIconSize(QSize(20, 20));
	m_renderButton->setToolTip("Render the selected scene with the current settings");
	connect(m_renderButton, &QPushButton::clicked, this, &MainWindow::onRenderClicked);

	// Stop button
	m_stopButton = new QPushButton("S&TOP RENDER", this);
	icon_tint::apply(m_stopButton, ":/icons/stop.svg",
	                 icon_tint::Role::Body, m_activeTheme.textBody);
	m_stopButton->setMinimumHeight(50);
	m_stopButton->setIconSize(QSize(20, 20));
	m_stopButton->setEnabled(false);
	m_stopButton->setToolTip("Stop the running render and discard its output");
	connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::onStopClicked);

	// Button layout
	QHBoxLayout *buttonLayout = new QHBoxLayout();
	buttonLayout->addWidget(m_renderButton);
	buttonLayout->addWidget(m_stopButton);
	mainLayout->addLayout(buttonLayout);

	setCentralWidget(centralWidget);
}

