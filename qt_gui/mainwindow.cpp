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
	// Called from the GUI thread. m_renderProcess is created and owned by the
	// worker thread (see run()), so it must not be touched from here - just
	// flag the request and let run()'s polling loop do the actual kill() on
	// its own thread.
	emit logMessage("Stopping render...");
	m_stopRequested.store(true, std::memory_order_relaxed);
}

void RenderThread::run() {
	emit logMessage(QString("Starting render..."));

	// Look up scene name from the shared descriptor table
	const SceneDesc* sceneInfo = find_scene_desc(m_sceneId);
	QString sceneName = sceneInfo ? QString(sceneInfo->name) : QString("Scene %1").arg(m_sceneId);

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

	// Scene ID applies in both modes - dropping it in video mode used to
	// silently fall back to the launcher's default (scene 0/Cornell Box)
	// regardless of what the user picked in the Scene dropdown.
	args << QString::number(m_sceneId);

	// Camera position only applies to single-image mode; in video mode the
	// camera path animation overrides these values every frame.
	if (!m_videoMode) {
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
	QRegularExpression conversionProgressRegex("Progress:\\s*(\\d+)/(\\d+)\\s*frames converted");  // Matches "Progress: 10/60 frames converted"
	int totalScanlines = m_height; // Track total height for percentage calculation
	QString accumulatedOutput;

	while (m_renderProcess->state() == QProcess::Running) {
		if (m_stopRequested.load(std::memory_order_relaxed)) {
			// This thread owns m_renderProcess (created above), so it's the
			// only one allowed to call kill() on it.
			m_renderProcess->kill();
			m_renderProcess->waitForFinished(3000); // Wait up to 3 seconds
			break;
		}
		m_renderProcess->waitForReadyRead(50); // Check more frequently
		QByteArray rawData = m_renderProcess->readAll();

		if (!rawData.isEmpty()) {
			// ray_tracer.exe's console output (box-drawing banners, checkmarks)
			// is UTF-8 - fromLocal8Bit() decoded it against the system ANSI
			// codepage instead, mangling every non-ASCII character (e.g. "─"
			// -> "â”€"). QProcess pipes raw bytes with no codepage translation,
			// so the bytes captured here are exactly what the child process
			// wrote - decode them as UTF-8 to match.
			QString output = QString::fromUtf8(rawData);
			accumulatedOutput += output;

			// Split into individual lines and emit each separately so every line
			// gets its own timestamp and category label in the log panel.
			QStringList lines = output.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);
			for (const QString& line : lines) {
				QString trimmed = line.trimmed();
				if (!trimmed.isEmpty())
					emit logMessage(trimmed);
			}

			// Parse progress from same lines
			for (const QString& line : lines) {
				// Video mode: rendering (0-80%), PNG conversion (80-95%), then a
				// fixed bump once ffmpeg assembly starts (no per-frame progress
				// available from ffmpeg cheaply); onRenderComplete snaps to 100%
				// once the subprocess actually exits.
				if (m_videoMode) {
					QRegularExpressionMatch frameMatch = videoFrameRegex.match(line);
					if (frameMatch.hasMatch()) {
						int currentFrame = frameMatch.captured(1).toInt();
						int totalFrames = frameMatch.captured(2).toInt();
						int progress = (totalFrames > 0) ? (currentFrame * 80) / totalFrames : 0;
						progress = std::max(0, std::min(progress, 80));
						if (progress > lastProgress) {
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
					QRegularExpressionMatch conversionMatch = conversionProgressRegex.match(line);
					if (conversionMatch.hasMatch()) {
						int convertedFrames = conversionMatch.captured(1).toInt();
						int totalFrames = conversionMatch.captured(2).toInt();
						int progress = 80 + ((totalFrames > 0) ? (convertedFrames * 15) / totalFrames : 0);
						progress = std::max(80, std::min(progress, 95));
						if (progress > lastProgress) {
							emit progressUpdate(progress);
							lastProgress = progress;
						}
					}
					if (line.contains("ASSEMBLING VIDEO WITH FFMPEG") && lastProgress < 97) {
						emit progressUpdate(97);
						lastProgress = 97;
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
						if (progress > lastProgress) { // Only update forward
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
						if (progress > lastProgress) {
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

	// Flush any trailing lines
	if (!finalOutput.isEmpty()) {
		QStringList trailing = finalOutput.split(QRegularExpression("[\\r\\n]"), Qt::SkipEmptyParts);
		for (const QString& line : trailing) {
			QString trimmed = line.trimmed();
			if (!trimmed.isEmpty())
				emit logMessage(trimmed);
		}
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

	// Whether this was a user-requested stop, tracked explicitly via
	// m_stopRequested rather than inferred from exit status/code - a real
	// crash (e.g. access violation) also produces CrashExit + a nonzero exit
	// code on Windows, so that heuristic can't tell the two apart and used to
	// mislabel genuine crashes as "stopped by user", hiding the error dialog.
	bool wasKilled = m_stopRequested.load(std::memory_order_relaxed);

	// Clean up process
	m_renderProcess->deleteLater();
	m_renderProcess = nullptr;

	if (wasKilled) {
		emit logMessage("Result: STOPPED BY USER");
		emit renderComplete(false, "Render stopped by user", totalTime, QString());
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
	: QMainWindow(parent), m_renderThread(nullptr), m_isRendering(false), m_videoMode(false),
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

