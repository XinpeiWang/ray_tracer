#include "mainwindow.h"
#include "error_handler.h"
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

// RenderThread Implementation
RenderThread::RenderThread(QObject *parent)
	: QThread(parent), m_useGPU(true), m_width(800), m_height(800), m_samples(100), m_maxDepth(50),
	  m_sceneId(0), m_camX(278), m_camY(278), m_camZ(-800), m_renderProcess(nullptr) {
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

void RenderThread::stopRender() {
	if (m_renderProcess && m_renderProcess->state() == QProcess::Running) {
		emit logMessage("Stopping render...");
		m_renderProcess->kill();
		m_renderProcess->waitForFinished(3000); // Wait up to 3 seconds
	}
}

void RenderThread::run() {
	emit logMessage("Starting render...");

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

	// Output file path (optional, has default)
	if (!m_outputPath.isEmpty()) {
		args << "--output" << m_outputPath;
	}

	// Numeric positional arguments (order matters!):
	// 1. width: image width in pixels (height = width for square aspect ratio)
	// 2. samples: samples per pixel for anti-aliasing and noise reduction
	// 3. depth: maximum ray bounce depth for recursive ray tracing
	// 4. scene_id: scene selector (0=Cornell Box, 1=Bouncing Spheres, etc.)
	// 5. cam_x: camera X position (lookfrom X coordinate)
	// 6. cam_y: camera Y position (lookfrom Y coordinate)
	// 7. cam_z: camera Z position (lookfrom Z coordinate)
	args << QString::number(m_width);
	args << QString::number(m_samples);
	args << QString::number(m_maxDepth);
	args << QString::number(m_sceneId);  // Scene ID
	args << QString::number(m_camX);   // Camera position X
	args << QString::number(m_camY);   // Camera position Y
	args << QString::number(m_camZ);   // Camera position Z

	emit logMessage(QString("Command: %1 %2").arg(exePath, args.join(" ")));

	QTime startTime = QTime::currentTime();

	// ========================================================================
	// Launch ray_tracer.exe as Subprocess
	// ========================================================================
	m_renderProcess = new QProcess();
	m_renderProcess->setProcessChannelMode(QProcess::MergedChannels);
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

			// Also check the accumulated buffer for the last scanline message
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

	// Detailed logging for debugging
	emit logMessage(QString("=== Process Exit Details ==="));
	emit logMessage(QString("Exit Code: %1").arg(exitCode));
	emit logMessage(QString("Exit Status: %1").arg(exitStatus == QProcess::NormalExit ? "NormalExit" : "CrashExit"));
	emit logMessage(QString("Total Time: %1 seconds").arg(totalTime, 0, 'f', 2));
	if (!finalOutput.isEmpty()) {
		emit logMessage("Final output: " + finalOutput);
	}

	// Check if process was killed (user stopped it)
	// Only treat as "stopped by user" if it crashed AND had a non-zero exit code
	// Normal exits with code 0 should be treated as success, even if the exit status is CrashExit
	bool wasKilled = (exitStatus == QProcess::CrashExit && exitCode != 0);

	emit logMessage(QString("wasKilled determination: exitStatus=%1, exitCode=%2, wasKilled=%3")
		.arg(exitStatus == QProcess::NormalExit ? "NormalExit" : "CrashExit")
		.arg(exitCode)
		.arg(wasKilled ? "TRUE" : "FALSE"));

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
	: QMainWindow(parent), m_renderThread(nullptr), m_isRendering(false) {

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
	createLogTab();

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

void MainWindow::createBasicTab() {
	QWidget *basicTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(basicTab);

	// Render settings group
	QGroupBox *renderGroup = new QGroupBox("Render Settings", basicTab);
	QFormLayout *renderLayout = new QFormLayout(renderGroup);
	renderLayout->setVerticalSpacing(15);
	renderLayout->setHorizontalSpacing(10);
	renderLayout->setContentsMargins(15, 20, 15, 15);

	// Render mode
	m_renderModeCombo = new QComboBox(basicTab);
	m_renderModeCombo->addItem("🎮 GPU (CUDA) - Fast", true);
	m_renderModeCombo->addItem("🖥️ CPU - High Quality", false);
	styleComboBox(m_renderModeCombo);
	renderLayout->addRow("Render Mode:", m_renderModeCombo);

	// Quality preset
	m_qualityPresetCombo = new QComboBox(basicTab);
	m_qualityPresetCombo->addItem("⚡ Draft (Very Fast)", 0);
	m_qualityPresetCombo->addItem("🚀 Preview (Fast)", 1);
	m_qualityPresetCombo->addItem("📷 Good (Balanced)", 2);
	m_qualityPresetCombo->addItem("💎 High (Slow)", 3);
	m_qualityPresetCombo->addItem("✨ Ultra (Very Slow)", 4);
	m_qualityPresetCombo->addItem("🌟 Maximum (Extreme)", 5);
	m_qualityPresetCombo->addItem("🎨 Custom", 6);
	m_qualityPresetCombo->setCurrentIndex(2); // Default to Good
	connect(m_qualityPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onQualityPresetChanged);
	styleComboBox(m_qualityPresetCombo);
	renderLayout->addRow("Quality:", m_qualityPresetCombo);

	// Resolution
	m_resolutionCombo = new QComboBox(basicTab);
	m_resolutionCombo->addItem("100 x 100 (Tiny)", QSize(100, 100));
	m_resolutionCombo->addItem("200 x 200", QSize(200, 200));
	m_resolutionCombo->addItem("400 x 400", QSize(400, 400));
	m_resolutionCombo->addItem("512 x 512", QSize(512, 512));
	m_resolutionCombo->addItem("600 x 600", QSize(600, 600));
	m_resolutionCombo->addItem("800 x 800", QSize(800, 800));
	m_resolutionCombo->addItem("1024 x 1024 (1K)", QSize(1024, 1024));
	m_resolutionCombo->addItem("1080 x 1080 (Full HD)", QSize(1080, 1080));
	m_resolutionCombo->addItem("1200 x 1200", QSize(1200, 1200));
	m_resolutionCombo->addItem("1440 x 1440", QSize(1440, 1440));
	m_resolutionCombo->addItem("1920 x 1920", QSize(1920, 1920));
	m_resolutionCombo->addItem("2048 x 2048 (2K)", QSize(2048, 2048));
	m_resolutionCombo->addItem("2560 x 2560", QSize(2560, 2560));
	m_resolutionCombo->addItem("3840 x 3840 (4K)", QSize(3840, 3840));
	m_resolutionCombo->addItem("4096 x 4096", QSize(4096, 4096));
	m_resolutionCombo->setCurrentIndex(5); // Default to 800x800
	styleComboBox(m_resolutionCombo);
	renderLayout->addRow("Resolution:", m_resolutionCombo);

	layout->addWidget(renderGroup);

	// Output group
	QGroupBox *outputGroup = new QGroupBox("Output", basicTab);
	QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);
	outputLayout->setSpacing(10);
	outputLayout->setContentsMargins(15, 20, 15, 15);

	QHBoxLayout *pathLayout = new QHBoxLayout();
	// Use timestamped filename to avoid caching issues
	QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
	QString defaultPath = QDir::homePath() + "/Desktop/render_" + timestamp + ".png";
	m_outputPathEdit = new QLineEdit(QDir::toNativeSeparators(defaultPath), basicTab);
	m_browseButton = new QPushButton("Browse...", basicTab);
	connect(m_browseButton, &QPushButton::clicked, [this]() {
		QString path = QFileDialog::getSaveFileName(this, "Save Render Output",
			m_outputPathEdit->text(), "PNG Image (*.png);;PPM Image (*.ppm)");
		if (!path.isEmpty()) {
			m_outputPathEdit->setText(QDir::toNativeSeparators(path));
		}
	});

	pathLayout->addWidget(m_outputPathEdit);
	pathLayout->addWidget(m_browseButton);
	outputLayout->addLayout(pathLayout);

	layout->addWidget(outputGroup);
	layout->addStretch();

	// Wrap the tab content in a scroll area for better responsiveness
	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(basicTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, "Basic Settings");
}

void MainWindow::createAdvancedTab() {
	QWidget *advancedTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(advancedTab);
	layout->setSpacing(25);  // Add spacing between group boxes
	layout->setContentsMargins(15, 15, 15, 15);

	QGroupBox *advancedGroup = new QGroupBox("Advanced Parameters", advancedTab);
	QFormLayout *formLayout = new QFormLayout(advancedGroup);
	formLayout->setVerticalSpacing(15);
	formLayout->setHorizontalSpacing(10);
	formLayout->setContentsMargins(15, 30, 15, 15);

	// Width
	m_widthSpinBox = new QSpinBox(advancedTab);
	m_widthSpinBox->setRange(100, 4096);
	m_widthSpinBox->setValue(800);
	styleSpinBox(m_widthSpinBox);
	formLayout->addRow("Width:", m_widthSpinBox);

	// Height
	m_heightSpinBox = new QSpinBox(advancedTab);
	m_heightSpinBox->setRange(100, 4096);
	m_heightSpinBox->setValue(800);
	styleSpinBox(m_heightSpinBox);
	formLayout->addRow("Height:", m_heightSpinBox);

	// Samples
	m_samplesSpinBox = new QSpinBox(advancedTab);
	m_samplesSpinBox->setRange(1, 10000);
	m_samplesSpinBox->setValue(100);
	styleSpinBox(m_samplesSpinBox);
	formLayout->addRow("Samples per Pixel:", m_samplesSpinBox);

	// Max depth
	m_maxDepthSpinBox = new QSpinBox(advancedTab);
	m_maxDepthSpinBox->setRange(1, 100);
	m_maxDepthSpinBox->setValue(50);
	styleSpinBox(m_maxDepthSpinBox);
	formLayout->addRow("Max Ray Depth:", m_maxDepthSpinBox);

	layout->addWidget(advancedGroup);

	// ============================================================================
	// Scene Selection Group
	// ============================================================================
	// Choose from different pre-built scenes from the Ray Tracing book series
	// Each scene has different complexity, performance characteristics, and features
	// ============================================================================

	QGroupBox *sceneGroup = new QGroupBox("Scene Selection", advancedTab);
	QVBoxLayout *sceneLayout = new QVBoxLayout(sceneGroup);
	sceneLayout->setSpacing(10);
	sceneLayout->setContentsMargins(15, 25, 15, 15);

	// Scene selector combo box
	m_sceneCombo = new QComboBox(advancedTab);
	m_sceneCombo->addItem("Cornell Box", 0);
	m_sceneCombo->addItem("Bouncing Spheres", 1);
	m_sceneCombo->addItem("Checkered Spheres", 2);
	m_sceneCombo->addItem("Earth (requires earthmap.jpg)", 3);
	m_sceneCombo->addItem("Perlin Spheres", 4);
	m_sceneCombo->addItem("Colored Quads", 5);
	m_sceneCombo->addItem("Simple Light", 6);
	m_sceneCombo->addItem("Cornell Smoke", 7);
	m_sceneCombo->addItem("Final Scene (very slow!)", 8);
	styleComboBox(m_sceneCombo);

	// Scene info label - shows description and performance info
	m_sceneInfoLabel = new QLabel(advancedTab);
	m_sceneInfoLabel->setWordWrap(true);
	m_sceneInfoLabel->setStyleSheet(
		"QLabel {"
		"  color: #B0B0B0;"
		"  background-color: #2E2E2E;"
		"  border: 1px solid #404040;"
		"  border-radius: 4px;"
		"  padding: 10px;"
		"  font-size: 11px;"
		"}"
	);

	sceneLayout->addWidget(new QLabel("Scene:"));
	sceneLayout->addWidget(m_sceneCombo);
	sceneLayout->addWidget(m_sceneInfoLabel);

	// Connect scene change handler
	connect(m_sceneCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onSceneChanged);

	// Initialize with default scene (Cornell Box)
	onSceneChanged(0);

	layout->addWidget(sceneGroup);

	// ============================================================================
	// Camera Position Group
	// ============================================================================
	// The Cornell box scene has fixed geometry:
	//   - Box dimensions: X[0,555], Y[0,555], Z[0,555]
	//   - Center point: (278, 278, 278)
	//   - Front opening: Z=0 (no wall, viewer can look in from outside)
	//   - Back wall: Z=555 (white)
	//   - Left wall: X=0 (red)
	//   - Right wall: X=555 (green)
	//   - Floor: Y=0 (white)
	//   - Ceiling: Y=555 (white), with light source at center
	//
	// Camera system:
	//   - lookfrom: camera position in 3D space (set by user via presets or custom values)
	//   - lookat: always points to center (278, 278, 278) - fixed in renderer
	//   - The camera can be positioned anywhere, inside or outside the box
	// ============================================================================

	QGroupBox *cameraGroup = new QGroupBox("Camera Position", advancedTab);
	QFormLayout *cameraLayout = new QFormLayout(cameraGroup);
	cameraLayout->setVerticalSpacing(15);
	cameraLayout->setHorizontalSpacing(10);
	cameraLayout->setContentsMargins(15, 30, 15, 15);

	// Camera preset combo box
	// Each preset stores a QVector3D with the camera position (lookfrom)
	// All presets maintain ~220-300 units from center for consistent viewing distance
	m_cameraPresetCombo = new QComboBox(advancedTab);

	// Default view: outside the box looking in through the open front (Z=0)
	m_cameraPresetCombo->addItem("Front View (Outside)", QVariant::fromValue(QVector3D(278, 278, -300)));

	// Inside views: camera positioned near walls, all looking toward center
	m_cameraPresetCombo->addItem("Inside Front", QVariant::fromValue(QVector3D(278, 278, 50)));    // Near Z=0 opening
	m_cameraPresetCombo->addItem("Inside Back", QVariant::fromValue(QVector3D(278, 278, 500)));    // Near Z=555 back wall
	m_cameraPresetCombo->addItem("Right Wall (Green)", QVariant::fromValue(QVector3D(500, 278, 278))); // Near X=555 green wall
	m_cameraPresetCombo->addItem("Left Wall (Red)", QVariant::fromValue(QVector3D(50, 278, 278)));     // Near X=0 red wall

	// Corner views: diagonal perspectives from inside the box
	m_cameraPresetCombo->addItem("Floor Corner", QVariant::fromValue(QVector3D(100, 50, 100)));    // Low angle, near floor
	m_cameraPresetCombo->addItem("Ceiling Corner", QVariant::fromValue(QVector3D(450, 500, 450))); // High angle, near ceiling

	// Custom: allows manual X/Y/Z input via spinboxes below
	m_cameraPresetCombo->addItem("Custom", QVariant::fromValue(QVector3D(278, 278, -300)));

	styleComboBox(m_cameraPresetCombo);
	cameraLayout->addRow("Preset:", m_cameraPresetCombo);

	// Camera position spinboxes (X, Y, Z coordinates)
	// These are disabled by default; only enabled when "Custom" preset is selected
	// Range: -2000 to 2000 allows positioning far outside the box if needed

	m_cameraPosX = new QDoubleSpinBox(advancedTab);
	m_cameraPosX->setRange(-2000, 2000);
	m_cameraPosX->setValue(278);  // Default X: centered horizontally
	m_cameraPosX->setSingleStep(10);
	m_cameraPosX->setEnabled(false);  // Disabled until "Custom" is selected
	cameraLayout->addRow("Camera X:", m_cameraPosX);

	m_cameraPosY = new QDoubleSpinBox(advancedTab);
	m_cameraPosY->setRange(-2000, 2000);
	m_cameraPosY->setValue(278);  // Default Y: centered vertically
	m_cameraPosY->setSingleStep(10);
	m_cameraPosY->setEnabled(false);  // Disabled until "Custom" is selected
	cameraLayout->addRow("Camera Y:", m_cameraPosY);

	m_cameraPosZ = new QDoubleSpinBox(advancedTab);
	m_cameraPosZ->setRange(-2000, 2000);
	m_cameraPosZ->setValue(-300);  // Default Z: outside front, matches default preset
	m_cameraPosZ->setSingleStep(10);
	m_cameraPosZ->setEnabled(false);  // Disabled until "Custom" is selected
	cameraLayout->addRow("Camera Z:", m_cameraPosZ);

	// Connect preset combo to handler that updates spinboxes and enables/disables manual input
	// Connection made AFTER all widgets are created to avoid null pointer issues
	connect(m_cameraPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onCameraPresetChanged);

	// Initialize the spinboxes with the default preset (index 0: "Front View (Outside)")
	onCameraPresetChanged(0);

	layout->addWidget(cameraGroup);
	layout->addStretch();

	// Wrap the tab content in a scroll area for better responsiveness
	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(advancedTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, "Advanced Settings");
}

void MainWindow::createLogTab() {
	QWidget *logWidget = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(logWidget);

	// Log output text area
	m_logTextEdit = new QTextEdit();
	m_logTextEdit->setReadOnly(true);
	m_logTextEdit->setFont(QFont("Consolas", 9));
	m_logTextEdit->setLineWrapMode(QTextEdit::NoWrap);

	layout->addWidget(m_logTextEdit);

	// Clear button
	QPushButton *clearButton = new QPushButton("Clear Log");
	clearButton->setMaximumWidth(120);
	connect(clearButton, &QPushButton::clicked, m_logTextEdit, &QTextEdit::clear);
	layout->addWidget(clearButton, 0, Qt::AlignRight);

	m_tabWidget->addTab(logWidget, "Log Output");
}

void MainWindow::applyDarkTheme() {
	// Cyberpunk theme with neon colors
	QPalette cyberpunkPalette;
	cyberpunkPalette.setColor(QPalette::Window, QColor(10, 10, 15));           // Deep black with subtle blue
	cyberpunkPalette.setColor(QPalette::WindowText, QColor(0, 255, 255));      // Bright cyan neon
	cyberpunkPalette.setColor(QPalette::Base, QColor(5, 5, 10));               // Almost pure black
	cyberpunkPalette.setColor(QPalette::AlternateBase, QColor(15, 10, 20));    // Dark purple tint
	cyberpunkPalette.setColor(QPalette::ToolTipBase, QColor(20, 0, 30));       // Dark purple
	cyberpunkPalette.setColor(QPalette::ToolTipText, QColor(255, 0, 255));     // Magenta neon
	cyberpunkPalette.setColor(QPalette::Text, QColor(0, 255, 255));            // Cyan neon
	cyberpunkPalette.setColor(QPalette::Button, QColor(30, 15, 50));           // Deep purple
	cyberpunkPalette.setColor(QPalette::ButtonText, QColor(255, 0, 255));      // Magenta neon
	cyberpunkPalette.setColor(QPalette::BrightText, QColor(255, 255, 0));      // Yellow neon
	cyberpunkPalette.setColor(QPalette::Link, QColor(0, 200, 255));            // Bright blue
	cyberpunkPalette.setColor(QPalette::Highlight, QColor(200, 0, 255));       // Purple-pink highlight
	cyberpunkPalette.setColor(QPalette::HighlightedText, QColor(255, 255, 255)); // White

	// Disabled state colors
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::WindowText, QColor(60, 60, 80));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(60, 60, 80));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(80, 40, 100));

	qApp->setPalette(cyberpunkPalette);
	qApp->setStyle(QStyleFactory::create("Fusion"));

	// Set cyberpunk-style font
	QFont cyberpunkFont;
	// Try futuristic/tech fonts, fallback to system fonts
	QStringList fontFamilies = {"Orbitron", "Rajdhani", "Exo 2", "Michroma", "Audiowide",
								 "Chakra Petch", "Saira", "Teko", "Electrolize",
								 "Bahnschrift", "Segoe UI", "Arial"};
	bool fontSet = false;
	for (const QString& fontFamily : fontFamilies) {
		cyberpunkFont.setFamily(fontFamily);
		if (QFontInfo(cyberpunkFont).family() == fontFamily) {
			fontSet = true;
			break;
		}
	}
	if (!fontSet) {
		cyberpunkFont.setFamily("Arial");
	}
	cyberpunkFont.setPointSize(11);  // Increased from 10
	cyberpunkFont.setWeight(QFont::Bold);
	cyberpunkFont.setLetterSpacing(QFont::AbsoluteSpacing, 0.8);
	qApp->setFont(cyberpunkFont);

	// Apply cyberpunk stylesheet for enhanced neon effects
	QString stylesheet = R"(
		QGroupBox {
			border: 3px solid #00FFFF;
			border-radius: 8px;
			margin-top: 12px;
			margin-bottom: 8px;
			padding: 20px 10px 10px 10px;
			color: #00FFFF;
			font-size: 12pt;
		}
		QGroupBox::title {
			subcontrol-origin: margin;
			subcontrol-position: top left;
			padding: 2px 15px;
			left: 10px;
			top: -10px;
			color: #FF00FF;
			font-size: 13pt;
			font-weight: bold;
		}
		QPushButton {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #3A1050, stop:1 #1E0832);
			border: 3px solid #FF00FF;
			border-radius: 8px;
			color: #FF00FF;
			padding: 12px 20px;
			font-weight: bold;
			font-size: 13pt;
			min-height: 45px;
		}
		QPushButton:hover {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #5A1570, stop:1 #3E1552);
			border: 3px solid #00FFFF;
			color: #00FFFF;
		}
		QPushButton:pressed {
			background-color: #2A0A40;
			border: 3px solid #C800FF;
		}
		QPushButton:disabled {
			background-color: #1A0A2A;
			border: 2px solid #503060;
			color: #503060;
		}
		QProgressBar {
			border: 3px solid #00FFFF;
			border-radius: 8px;
			text-align: center;
			background-color: #0A0A0F;
			color: #00FFFF;
			font-size: 12pt;
			min-height: 35px;
		}
		QProgressBar::chunk {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 #FF00FF, stop:0.5 #C800FF, stop:1 #00FFFF);
			border-radius: 5px;
		}
		QTabWidget::pane {
			border: 3px solid #FF00FF;
			border-radius: 8px;
			background-color: #0A0A0F;
			top: -3px;
		}
		QTabBar::tab {
			background-color: #1E0832;
			border: 3px solid #FF00FF;
			border-bottom: none;
			border-top-left-radius: 8px;
			border-top-right-radius: 8px;
			padding: 12px 20px;
			color: #FF00FF;
			font-size: 12pt;
			min-width: 100px;
			margin-right: 2px;
		}
		QTabBar::tab:selected {
			background-color: #3A1050;
			border-color: #00FFFF;
			color: #00FFFF;
			margin-bottom: -3px;
			padding-bottom: 15px;
		}
		QTabBar::tab:hover {
			background-color: #2A0A40;
			color: #00FFFF;
		}
		QSpinBox, QDoubleSpinBox, QComboBox {
			background-color: #0A0A0F;
			border: 3px solid #00FFFF;
			border-radius: 5px;
			padding: 8px;
			color: #00FFFF;
			font-size: 11pt;
			min-height: 30px;
			margin: 5px 2px;
		}
		QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
			background-color: #1A0A2A;
			border: 3px solid #FF00FF;
			color: #FF00FF;
		}
		QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
			background-color: #2A1040;
			border: 3px solid #C800FF;
			color: #FF00FF;
		}
		QSpinBox, QDoubleSpinBox {
			padding-right: 30px;
		}
		QComboBox::drop-down {
			border: none;
			padding-right: 5px;
		}
		QComboBox::down-arrow {
			image: none;
			border-left: 5px solid transparent;
			border-right: 5px solid transparent;
			border-top: 6px solid #FF00FF;
		}
		QComboBox::down-arrow:hover {
			border-top: 6px solid #00FFFF;
		}
		QComboBox QAbstractItemView {
			background-color: #0A0A0F;
			border: 3px solid #FF00FF;
			border-radius: 5px;
			selection-background-color: #FF00FF;
			selection-color: #000000;
			color: #00FFFF;
			outline: none;
			padding: 2px;
		}
		QComboBox QAbstractItemView::item {
			padding: 8px;
			min-height: 30px;
			border: 1px solid transparent;
			background-color: transparent;
			color: #00FFFF;
		}
		QComboBox QAbstractItemView::item:hover {
			background-color: #5A1570;
			border: 1px solid #FF00FF;
			color: #FF00FF;
		}
		QComboBox QAbstractItemView::item:selected {
			background-color: #FF00FF;
			color: #000000;
			border: 1px solid #00FFFF;
		}
		QComboBox QAbstractItemView::item:selected:hover {
			background-color: #00FFFF;
			color: #000000;
			border: 1px solid #FF00FF;
		}
		QListView {
			background-color: #0A0A0F;
			border: 3px solid #FF00FF;
			color: #00FFFF;
		}
		QListView::item {
			padding: 8px;
			min-height: 30px;
		}
		QListView::item:hover {
			background-color: #5A1570;
			color: #FF00FF;
		}
		QListView::item:selected {
			background-color: #FF00FF;
			color: #000000;
		}
		QListView::item:selected:hover {
			background-color: #00FFFF;
			color: #000000;
		}
		QLabel {
			color: #00FFFF;
			font-size: 11pt;
			padding: 8px 5px;
			margin: 5px 2px;
		}
		QFormLayout {
			spacing: 15px;
		}
		QTextEdit {
			background-color: #0A0A0F;
			border: 3px solid #00FFFF;
			border-radius: 5px;
			color: #00FFFF;
			selection-background-color: #C800FF;
			font-size: 10pt;
			padding: 5px;
		}
		QScrollBar:vertical {
			background-color: #0A0A0F;
			width: 16px;
			margin: 0px;
			border: 2px solid #00FFFF;
			border-radius: 8px;
		}
		QScrollBar::handle:vertical {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 #FF00FF, stop:1 #C800FF);
			border-radius: 6px;
			min-height: 30px;
			margin: 2px;
		}
		QScrollBar::handle:vertical:hover {
			background-color: #00FFFF;
		}
		QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
			height: 0px;
		}
		QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
			background: none;
		}
		QScrollBar:horizontal {
			background-color: #0A0A0F;
			height: 16px;
			margin: 0px;
			border: 2px solid #00FFFF;
			border-radius: 8px;
		}
		QScrollBar::handle:horizontal {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #FF00FF, stop:1 #C800FF);
			border-radius: 6px;
			min-width: 30px;
			margin: 2px;
		}
		QScrollBar::handle:horizontal:hover {
			background-color: #00FFFF;
		}
		QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
			width: 0px;
		}
		QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
			background: none;
		}
	)";
	qApp->setStyleSheet(stylesheet);
}

void MainWindow::styleComboBox(QComboBox *combo) {
	// Get the view (popup list) of the combo box
	QAbstractItemView *view = combo->view();

	// CRITICAL: Enable mouse tracking for hover events
	view->setMouseTracking(true);
	view->viewport()->setMouseTracking(true);

	// Force the view to update on hover
	view->setAttribute(Qt::WA_Hover, true);
	view->viewport()->setAttribute(Qt::WA_Hover, true);

	// Set up custom palette for the dropdown with bright hover colors
	QPalette viewPalette;
	viewPalette.setColor(QPalette::Base, QColor(10, 10, 15));              // Background
	viewPalette.setColor(QPalette::Text, QColor(0, 255, 255));             // Text - cyan
	viewPalette.setColor(QPalette::AlternateBase, QColor(20, 15, 25));

	// IMPORTANT: Set highlight colors for hover state
	viewPalette.setColor(QPalette::Highlight, QColor(255, 0, 255));        // Hover background - BRIGHT MAGENTA
	viewPalette.setColor(QPalette::HighlightedText, QColor(255, 255, 255)); // Hover text - WHITE for contrast

	// Disabled inactive states
	viewPalette.setColor(QPalette::Inactive, QPalette::Highlight, QColor(255, 0, 255));
	viewPalette.setColor(QPalette::Inactive, QPalette::HighlightedText, QColor(255, 255, 255));

	view->setPalette(viewPalette);

	// Simple stylesheet focusing on palette usage
	QString itemStyle = R"(
		QAbstractItemView {
			background-color: rgb(10, 10, 15);
			border: 3px solid rgb(255, 0, 255);
			border-radius: 5px;
			outline: none;
			show-decoration-selected: 1;
		}
		QAbstractItemView::item {
					padding: 12px;
					min-height: 40px;
					border: none;
				}
				)";
				view->setStyleSheet(itemStyle);
			}

			void MainWindow::styleSpinBox(QSpinBox *spinBox) {
				// Apply custom stylesheet with CSS triangle arrows
				QString spinBoxStyle = R"(
					QSpinBox {
						padding-right: 30px;
						background-color: #0A0A0F;
						border: 3px solid #FF00FF;
						border-radius: 5px;
						color: #00FFFF;
						font-size: 12pt;
						padding: 8px;
					}
					QSpinBox::up-button {
						subcontrol-origin: border;
						subcontrol-position: top right;
						width: 28px;
						border-left: 2px solid #FF00FF;
						border-top: 2px solid #FF00FF;
						border-right: 2px solid #FF00FF;
						border-top-right-radius: 3px;
						background-color: #1E0832;
					}
					QSpinBox::down-button {
						subcontrol-origin: border;
						subcontrol-position: bottom right;
						width: 28px;
						border-left: 2px solid #FF00FF;
						border-bottom: 2px solid #FF00FF;
						border-right: 2px solid #FF00FF;
						border-bottom-right-radius: 3px;
						background-color: #1E0832;
					}
					QSpinBox::up-button:hover {
						background-color: #3A1050;
						border-color: #00FFFF;
					}
					QSpinBox::down-button:hover {
						background-color: #3A1050;
						border-color: #00FFFF;
					}
					QSpinBox::up-arrow {
						width: 0;
						height: 0;
						border-left: 5px solid transparent;
						border-right: 5px solid transparent;
						border-bottom: 7px solid #FF00FF;
					}
					QSpinBox::up-arrow:hover {
						border-bottom-color: #00FFFF;
					}
					QSpinBox::down-arrow {
						width: 0;
						height: 0;
						border-left: 5px solid transparent;
						border-right: 5px solid transparent;
						border-top: 7px solid #FF00FF;
					}
					QSpinBox::down-arrow:hover {
						border-top-color: #00FFFF;
					}
				)";
				spinBox->setStyleSheet(spinBoxStyle);
			}

			void MainWindow::onRenderClicked() {
	if (m_isRendering) {
		QMessageBox::warning(this, "Render In Progress", "A render is already in progress!");
		return;
	}

	// ========================================================================
	// Collect Render Parameters
	// ========================================================================

	// Render mode: GPU (true) or CPU (false)
	bool useGPU = m_renderModeCombo->currentData().toBool();

	// Resolution: either from preset dropdown or custom values from Advanced tab
	int width, height;
	if (m_qualityPresetCombo->currentIndex() == 6) {
		// Custom quality preset - use manual width/height from Advanced tab
		width = m_widthSpinBox->value();
		height = m_heightSpinBox->value();
	} else {
		// Standard quality preset - use resolution from dropdown
		QSize res = m_resolutionCombo->currentData().toSize();
		width = res.width();
		height = res.height();
	}

	// Ray tracing quality parameters
	int samples = m_samplesSpinBox->value();    // Samples per pixel (higher = smoother but slower)
	int maxDepth = m_maxDepthSpinBox->value();  // Max ray bounce depth (higher = more realistic lighting)

	// Output file path (timestamped by default to avoid overwriting)
	QString outputPath = m_outputPathEdit->text();

	// Camera position (lookfrom) - read from spinboxes
	// These reflect either the selected preset or custom user input
	// The camera will always look toward the center (278, 278, 278) - lookat is fixed in renderer
	double camX = m_cameraPosX->value();
	double camY = m_cameraPosY->value();
	double camZ = m_cameraPosZ->value();

	// Scene selection - determines which scene to render
	int sceneId = m_sceneCombo->currentData().toInt();

	// ========================================================================
	// Launch Render Thread
	// ========================================================================
	// RenderThread spawns ray_tracer.exe as a subprocess with all parameters
	// The executable will call either CPU or GPU renderer based on useGPU flag
	m_renderThread = new RenderThread(this);
	m_renderThread->setParameters(useGPU, width, height, samples, maxDepth, sceneId, camX, camY, camZ, outputPath);

	connect(m_renderThread, &RenderThread::progressUpdate, this, &MainWindow::onProgressUpdate);
	connect(m_renderThread, SIGNAL(renderComplete(bool,QString,double,QString)), 
			this, SLOT(onRenderComplete(bool,QString,double,QString)));
	connect(m_renderThread, &RenderThread::logMessage, this, &MainWindow::onLogMessage);
	connect(m_renderThread, &QThread::finished, m_renderThread, &QObject::deleteLater);

	m_isRendering = true;
	m_renderButton->setEnabled(false);
	m_stopButton->setEnabled(true);
	m_progressBar->setValue(0);
	m_statusLabel->setText("Rendering...");

	m_renderThread->start();
}

void MainWindow::onStopClicked() {
	if (!m_isRendering || !m_renderThread) {
		return;
	}

	m_statusLabel->setText("Stopping render...");
	m_stopButton->setEnabled(false);

	// Stop the render process - this will cause the thread to finish
	m_renderThread->stopRender();

	// The thread will emit renderComplete when done, which will reset the UI
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

	// Update spinbox values to reflect the selected preset's camera position
	// This happens even when spinboxes are disabled, so user can see the coordinates
	// QVector3D is stored in each combo box item's data as a QVariant
	if (index >= 0 && index < m_cameraPresetCombo->count()) {
		QVector3D pos = m_cameraPresetCombo->itemData(index).value<QVector3D>();
		m_cameraPosX->setValue(pos.x());
		m_cameraPosY->setValue(pos.y());
		m_cameraPosZ->setValue(pos.z());
	}
}

void MainWindow::onSceneChanged(int index) {
	// Scene metadata with descriptions and recommendations
	struct SceneInfo {
		const char* description;
		const char* performance;
		int recommendedSpp;
		bool requiresFiles;
		bool gpuCompatible;
	};

	static const SceneInfo sceneInfos[] = {
		{"Classic Cornell box with glass sphere and white box", "Medium", 100, false, true},
		{"Random spheres with checker ground (In One Weekend final)", "Slow", 100, false, false},
		{"Two spheres with procedural checker texture", "Fast", 100, false, false},
		{"Globe with earth texture mapping (requires earthmap.jpg)", "Fast", 100, true, false},
		{"Spheres with Perlin noise marble texture", "Fast", 100, false, false},
		{"Five colored quad primitives", "Fast", 100, false, false},
		{"Perlin spheres with emissive light sources", "Fast", 100, false, false},
		{"Cornell box with volumetric fog", "Slow", 200, false, false},
		{"Complex scene from The Next Week (very slow!)", "Very Slow", 500, false, false}
	};

	if (index >= 0 && index < 9) {
		const SceneInfo& info = sceneInfos[index];

		QString infoText = QString("<b>Description:</b> %1<br>").arg(info.description);
		infoText += QString("<b>Performance:</b> %1<br>").arg(info.performance);
		infoText += QString("<b>Recommended SPP:</b> %1<br>").arg(info.recommendedSpp);
		infoText += QString("<b>GPU Support:</b> %1<br>").arg(info.gpuCompatible ? "Yes" : "CPU only");

		if (info.requiresFiles) {
			infoText += "<br><b style='color: #FFD700;'>⚠ Requires external files</b>";
		}

		if (!info.gpuCompatible) {
			infoText += "<br><b style='color: #FF6B6B;'>⚠ CPU renderer only</b>";
		}

		m_sceneInfoLabel->setText(infoText);

		// Update recommended samples if user hasn't manually changed it
		if (m_samplesSpinBox->value() == 100 || m_samplesSpinBox->value() == 200 || m_samplesSpinBox->value() == 500) {
			m_samplesSpinBox->setValue(info.recommendedSpp);
		}
	}
}

void MainWindow::onProgressUpdate(int percentage) {
	m_progressBar->setValue(percentage);
	m_statusLabel->setText(QString("Rendering... %1%").arg(percentage));
}

void MainWindow::onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath) {
	m_isRendering = false;
	m_renderButton->setEnabled(true);
	m_stopButton->setEnabled(false);

	if (success) {
		m_progressBar->setValue(100);
		m_statusLabel->setText(QString("✅ %1 - Total time: %2 seconds").arg(message).arg(totalTime, 0, 'f', 2));

		// Auto-open the output file if it exists
		if (!outputPath.isEmpty()) {
			QFileInfo fileInfo(outputPath);

			if (fileInfo.exists()) {
				QDesktopServices::openUrl(QUrl::fromLocalFile(outputPath));
			} else {
				m_statusLabel->setText(QString("✅ Render complete (%1s) - Warning: output file not found at %2")
					.arg(totalTime, 0, 'f', 2).arg(outputPath));
			}
		}
	} else {
		// Reset progress bar on failure/stop
		m_progressBar->setValue(0);
		m_statusLabel->setText(QString("❌ %1").arg(message));

		// Only show error popup for actual failures, not for user-stopped renders
		if (!message.contains("stopped by user", Qt::CaseInsensitive)) {
			QMessageBox::critical(this, "Render Failed", message);
		}
	}
}

void MainWindow::onLogMessage(const QString &message) {
	if (m_logTextEdit) {
		m_logTextEdit->append(message);
	}
	qDebug() << message;
}
