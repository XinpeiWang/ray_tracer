#include "mainwindow.h"
#include "error_handler.h"
#include "scene_descriptor.h"
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
#include <QPixmap>
#include <QScrollArea>
#include <QScreen>
#include <QTimer>

// RenderThread Implementation
RenderThread::RenderThread(QObject *parent)
	: QThread(parent), m_useGPU(true), m_width(800), m_height(800), m_samples(100), m_maxDepth(50),
	  m_sceneId(0), m_camX(278), m_camY(278), m_camZ(-800), m_renderProcess(nullptr),
	  m_videoMode(false), m_videoFrames(60), m_videoFPS(30), m_cameraPath("orbit") {
}

void RenderThread::setParameters(bool useGPU, int width, int height, int samples, int maxDepth,
								  int sceneId, double camX, double camY, double camZ, const QString &outputPath) {
	m_useGPU = useGPU;
	m_width = width;
	m_height = height;
	m_samples = samples;
	m_maxDepth = maxDepth;
	m_sceneId = sceneId;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_outputPath = outputPath;
}

void RenderThread::setVideoParameters(bool enabled, int frames, int fps, const QString &cameraPath) {
	m_videoMode = enabled;
	m_videoFrames = frames;
	m_videoFPS = fps;
	m_cameraPath = cameraPath;
}

void RenderThread::stopRender() {
	if (m_renderProcess && m_renderProcess->state() == QProcess::Running) {
		emit logMessage("Stopping render...");
		m_renderProcess->kill();
		m_renderProcess->waitForFinished(3000); // Wait up to 3 seconds
	}
}

void RenderThread::run() {
	emit logMessage(QString("Starting render..."));

	// Emit a separator so each render is visually distinct in the log
	QString sep = QString("─").repeated(60);
	emit logMessage(sep);
	emit logMessage(QString("▶ RENDER START  %1")
		.arg(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss")));
	emit logMessage(QString("  Scene: %1 | %2x%3 | %4 spp | depth %5 | %6")
		.arg(m_sceneId)
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
	// 5. cam_x: camera X position (lookfrom X coordinate) - only used in single-image mode
	// 6. cam_y: camera Y position (lookfrom Y coordinate) - only used in single-image mode
	// 7. cam_z: camera Z position (lookfrom Z coordinate) - only used in single-image mode
	args << QString::number(m_width);
	args << QString::number(m_samples);
	args << QString::number(m_maxDepth);

	// Camera position only applies to single-image mode
	// In video mode, camera path animation overrides these values
	if (!m_videoMode) {
		args << QString::number(m_sceneId);  // Scene ID
		args << QString::number(m_camX);     // Camera position X
		args << QString::number(m_camY);     // Camera position Y
		args << QString::number(m_camZ);     // Camera position Z
	}

	emit logMessage(QString("Command: %1 %2").arg(exePath, args.join(" ")));

	QTime startTime = QTime::currentTime();

	// ========================================================================
	// Launch ray_tracer.exe as Subprocess
	// ========================================================================
	m_renderProcess = new QProcess();
	m_renderProcess->setProcessChannelMode(QProcess::MergedChannels);

	// Set working directory to the application directory (RayTracer_Package)
	// This ensures output/frames directories are created in the correct location
	m_renderProcess->setWorkingDirectory(QCoreApplication::applicationDirPath());

	m_renderProcess->start(exePath, args);

	if (!m_renderProcess->waitForStarted()) {
		QString error = m_renderProcess->errorString();
		emit renderComplete(false, QString("Failed to start renderer: %1").arg(error), 0.0, QString());
		m_renderProcess->deleteLater();
		m_renderProcess = nullptr;
		return;
	}

	// Read output and parse progress
	int lastProgress = 0;
	QRegularExpression scanlinesRegex("Scanlines remaining:\\s*(\\d+)");
	QRegularExpression videoFrameRegex("\\[(\\d+)/(\\d+)\\] Rendering frame_");  // Matches "[5/60] Rendering frame_"
	QRegularExpression assemblyProgressRegex("Progress:\\s*(\\d+)/(\\d+)\\s*frames written");  // Matches "Progress: 10/60 frames written"
	int totalScanlines = m_height; // Track total height for percentage calculation
	QString accumulatedOutput;

	while (m_renderProcess->state() == QProcess::Running) {
		m_renderProcess->waitForReadyRead(50); // Check more frequently
		QByteArray rawData = m_renderProcess->readAll();

		if (!rawData.isEmpty()) {
			QString output = QString::fromLocal8Bit(rawData);
			accumulatedOutput += output;
			emit logMessage(output.trimmed());

			// Parse progress - look for "Scanlines remaining: X" in the most recent output
			// Handle both \r and \n line endings
			QStringList lines = output.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);
			for (const QString& line : lines) {
				// Video mode: look for "[X/Y] Rendering frame_" pattern (0-90%)
				if (m_videoMode) {
					QRegularExpressionMatch frameMatch = videoFrameRegex.match(line);
					if (frameMatch.hasMatch()) {
						int currentFrame = frameMatch.captured(1).toInt();
						int totalFrames = frameMatch.captured(2).toInt();
						// Scale rendering to 0-90%
						int progress = (totalFrames > 0) ? (currentFrame * 90) / totalFrames : 0;
						progress = std::max(0, std::min(progress, 90)); // Clamp to 0-90
						if (progress != lastProgress && progress >= lastProgress) {
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
					// Also check for assembly progress (90-100%)
					QRegularExpressionMatch assemblyMatch = assemblyProgressRegex.match(line);
					if (assemblyMatch.hasMatch()) {
						int writtenFrames = assemblyMatch.captured(1).toInt();
						int totalFrames = assemblyMatch.captured(2).toInt();
						// Scale assembly to 90-100%
						int progress = 90 + ((totalFrames > 0) ? (writtenFrames * 10) / totalFrames : 0);
						progress = std::max(90, std::min(progress, 100)); // Clamp to 90-100
						if (progress != lastProgress && progress >= lastProgress) {
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
				}
				// Single-image mode: look for "Scanlines remaining: X"
				else {
					QRegularExpressionMatch scanlinesMatch = scanlinesRegex.match(line);
					if (scanlinesMatch.hasMatch()) {
						int remaining = scanlinesMatch.captured(1).toInt();
						int completed = totalScanlines - remaining;
						int progress = (totalScanlines > 0) ? (completed * 100) / totalScanlines : 0;
						progress = std::max(0, std::min(progress, 100)); // Clamp to 0-100
						if (progress != lastProgress && progress >= lastProgress) { // Only update forward
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
				}
			}

			// Also check the accumulated buffer for the last scanline message (single-image mode only)
			if (!m_videoMode) {
				int lastCR = accumulatedOutput.lastIndexOf('\r');
				if (lastCR >= 0) {
					QString lastLine = accumulatedOutput.mid(lastCR + 1);
					QRegularExpressionMatch match = scanlinesRegex.match(lastLine);
					if (match.hasMatch()) {
						int remaining = match.captured(1).toInt();
						int completed = totalScanlines - remaining;
						int progress = (totalScanlines > 0) ? (completed * 100) / totalScanlines : 0;
						progress = std::max(0, std::min(progress, 100));
						if (progress != lastProgress && progress >= lastProgress) {
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
				}
			}
		}
	}

	// Wait for process to finish
	// Use -1 (infinite timeout) because complex scenes can take a long time
	// The progress updates will keep the GUI responsive
	if (m_renderProcess->state() != QProcess::NotRunning) {
		m_renderProcess->waitForFinished(-1); // Wait indefinitely
	}

	QTime endTime = QTime::currentTime();
	double totalTime = startTime.msecsTo(endTime) / 1000.0;

	int exitCode = m_renderProcess->exitCode();
	QProcess::ExitStatus exitStatus = m_renderProcess->exitStatus();
	QString finalOutput = m_renderProcess->readAll();

	// Clean summary line
	emit logMessage(QString("Process finished: exit=%1  status=%2  time=%3s")
		.arg(exitCode)
		.arg(exitStatus == QProcess::NormalExit ? "Normal" : "Crash")
		.arg(totalTime, 0, 'f', 1));
	if (!finalOutput.isEmpty()) {
		emit logMessage("Trailing output: " + finalOutput);
	}

	// Check if process was killed (user stopped it)
	bool wasKilled = (exitStatus == QProcess::CrashExit && exitCode != 0);

	// Clean up process
	m_renderProcess->deleteLater();
	m_renderProcess = nullptr;

	if (wasKilled) {
		emit logMessage("Result: STOPPED BY USER");

		// Show user-friendly error dialog for crashes
		QString errorTitle = ErrorHandler::getErrorTitle(exitCode);
		QString errorMessage = ErrorHandler::getErrorMessage(exitCode);
		QString hint = ErrorHandler::getTroubleshootingHint(exitCode);

		emit logMessage(QString("Error Category: %1").arg(ErrorHandler::getCategoryName(exitCode)));
		emit logMessage(QString("Error: %1").arg(errorTitle));

		emit renderComplete(false, "Render stopped by user", totalTime, QString());
	} else if (exitCode == 0) {
		emit logMessage("Result: SUCCESS");
		emit progressUpdate(100);

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

		emit renderComplete(true, "Render completed successfully!", totalTime, actualOutputPath);
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
		emit renderComplete(false, fullErrorMsg, totalTime, QString());
	}
}

// MainWindow Implementation
MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), m_renderThread(nullptr), m_isRendering(false), m_videoMode(false) {

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

	setupUI();
	applyDarkTheme();
}

MainWindow::~MainWindow() {
	if (m_renderThread && m_renderThread->isRunning()) {
		m_renderThread->terminate();
		m_renderThread->wait();
	}
}

void MainWindow::setupUI() {
	QWidget *centralWidget = new QWidget(this);
	QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

	// Create tab widget
	m_tabWidget = new QTabWidget(this);
	createBasicTab();
	createAdvancedTab();
	createVideoTab();
	createLogTab();

	// Initialize scene info AFTER tabs are created (onSceneChanged uses m_samplesSpinBox)
	onSceneChanged(0);

	mainLayout->addWidget(m_tabWidget);

	// Progress section
	QGroupBox *progressGroup = new QGroupBox("Progress", this);
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

	// Render button
	m_renderButton = new QPushButton("▶ START RENDER", this);
	m_renderButton->setMinimumHeight(50);
	connect(m_renderButton, &QPushButton::clicked, this, &MainWindow::onRenderClicked);

	// Stop button
	m_stopButton = new QPushButton("■ STOP RENDER", this);
	m_stopButton->setMinimumHeight(50);
	m_stopButton->setEnabled(false);
	connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::onStopClicked);

	// Button layout
	QHBoxLayout *buttonLayout = new QHBoxLayout();
	buttonLayout->addWidget(m_renderButton);
	buttonLayout->addWidget(m_stopButton);
	mainLayout->addLayout(buttonLayout);

	setCentralWidget(centralWidget);
}

