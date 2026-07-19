#include "mainwindow.h"
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
#include <QRegularExpression>
#include <QDesktopServices>
#include <QUrl>
#include <QFont>
#include <QFontInfo>

// RenderThread Implementation
RenderThread::RenderThread(QObject *parent)
	: QThread(parent), m_useGPU(true), m_width(800), m_height(800), m_samples(100), m_maxDepth(50), m_renderProcess(nullptr) {
}

void RenderThread::setParameters(bool useGPU, int width, int height, int samples, int maxDepth, const QString &outputPath) {
	m_useGPU = useGPU;
	m_width = width;
	m_height = height;
	m_samples = samples;
	m_maxDepth = maxDepth;
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

	// Build command line arguments
	QString exePath = QCoreApplication::applicationDirPath() + "/ray_tracer.exe";
	QStringList args;

	// Mode flag
	if (m_useGPU) {
		args << "--gpu";
	} else {
		args << "--cpu";
	}

	// Output path if specified
	if (!m_outputPath.isEmpty()) {
		args << "--output" << m_outputPath;
	}

	// Numeric arguments: width samples depth
	// Note: ray_tracer.exe uses square aspect ratio (height = width)
	args << QString::number(m_width);
	args << QString::number(m_samples);
	args << QString::number(m_maxDepth);

	emit logMessage(QString("Command: %1 %2").arg(exePath, args.join(" ")));

	QTime startTime = QTime::currentTime();

	// Launch render process
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

	// Wait for process to finish (with timeout in case of issues)
	if (m_renderProcess->state() != QProcess::NotRunning) {
		m_renderProcess->waitForFinished(5000); // 5 second timeout
	}

	QTime endTime = QTime::currentTime();
	double totalTime = startTime.msecsTo(endTime) / 1000.0;

	int exitCode = m_renderProcess->exitCode();
	QString finalOutput = m_renderProcess->readAll();

	emit logMessage(QString("Process exited with code %1").arg(exitCode));
	if (!finalOutput.isEmpty()) {
		emit logMessage("Final output: " + finalOutput);
	}

	// Check if process was killed (user stopped it)
	bool wasKilled = (m_renderProcess->exitStatus() == QProcess::CrashExit);

	// Clean up process
	m_renderProcess->deleteLater();
	m_renderProcess = nullptr;

	if (wasKilled) {
		emit renderComplete(false, "Render stopped by user", totalTime, QString());
	} else if (exitCode == 0) {
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
		QString errorMsg = QString("Render failed with exit code %1").arg(exitCode);
		if (!finalOutput.isEmpty()) {
			errorMsg += "\n\nOutput:\n" + finalOutput;
		}
		emit renderComplete(false, errorMsg, totalTime, QString());
	}
}

// MainWindow Implementation
MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), m_renderThread(nullptr), m_isRendering(false) {

	setWindowTitle("Ray Tracer - Path Tracing Renderer");
	setMinimumSize(600, 500);

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
	m_renderButton = new QPushButton("Start Render", this);
	m_renderButton->setMinimumHeight(40);
	m_renderButton->setStyleSheet("font-size: 14pt; font-weight: bold;");
	connect(m_renderButton, &QPushButton::clicked, this, &MainWindow::onRenderClicked);

	// Stop button
	m_stopButton = new QPushButton("Stop Render", this);
	m_stopButton->setMinimumHeight(40);
	m_stopButton->setStyleSheet("font-size: 14pt; font-weight: bold;");
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

	// Render mode
	m_renderModeCombo = new QComboBox(basicTab);
	m_renderModeCombo->addItem("🎮 GPU (CUDA) - Fast", true);
	m_renderModeCombo->addItem("🖥️ CPU - High Quality", false);
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
	renderLayout->addRow("Resolution:", m_resolutionCombo);

	layout->addWidget(renderGroup);

	// Output group
	QGroupBox *outputGroup = new QGroupBox("Output", basicTab);
	QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);

	QHBoxLayout *pathLayout = new QHBoxLayout();
	m_outputPathEdit = new QLineEdit(QDir::toNativeSeparators(
		QDir::homePath() + "/Desktop/render_output.png"), basicTab);
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

	m_tabWidget->addTab(basicTab, "Basic Settings");
}

void MainWindow::createAdvancedTab() {
	QWidget *advancedTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(advancedTab);

	QGroupBox *advancedGroup = new QGroupBox("Advanced Parameters", advancedTab);
	QFormLayout *formLayout = new QFormLayout(advancedGroup);

	// Width
	m_widthSpinBox = new QSpinBox(advancedTab);
	m_widthSpinBox->setRange(100, 4096);
	m_widthSpinBox->setValue(800);
	formLayout->addRow("Width:", m_widthSpinBox);

	// Height
	m_heightSpinBox = new QSpinBox(advancedTab);
	m_heightSpinBox->setRange(100, 4096);
	m_heightSpinBox->setValue(800);
	formLayout->addRow("Height:", m_heightSpinBox);

	// Samples
	m_samplesSpinBox = new QSpinBox(advancedTab);
	m_samplesSpinBox->setRange(1, 10000);
	m_samplesSpinBox->setValue(100);
	formLayout->addRow("Samples per Pixel:", m_samplesSpinBox);

	// Max depth
	m_maxDepthSpinBox = new QSpinBox(advancedTab);
	m_maxDepthSpinBox->setRange(1, 100);
	m_maxDepthSpinBox->setValue(50);
	formLayout->addRow("Max Ray Depth:", m_maxDepthSpinBox);

	layout->addWidget(advancedGroup);
	layout->addStretch();

	m_tabWidget->addTab(advancedTab, "Advanced Settings");
}

void MainWindow::applyDarkTheme() {
	// Gaming theme with bold font and darker muted colors
	QPalette darkPalette;
	darkPalette.setColor(QPalette::Window, QColor(8, 10, 16));           // Nearly black with blue tint
	darkPalette.setColor(QPalette::WindowText, QColor(80, 150, 130));    // Darker teal
	darkPalette.setColor(QPalette::Base, QColor(12, 15, 22));            // Very dark blue-black
	darkPalette.setColor(QPalette::AlternateBase, QColor(8, 10, 16));
	darkPalette.setColor(QPalette::ToolTipBase, QColor(18, 22, 32));
	darkPalette.setColor(QPalette::ToolTipText, QColor(100, 160, 140));
	darkPalette.setColor(QPalette::Text, QColor(90, 160, 140));          // Muted teal
	darkPalette.setColor(QPalette::Button, QColor(25, 35, 70));          // Deep navy blue
	darkPalette.setColor(QPalette::ButtonText, QColor(140, 80, 160));    // Darker purple
	darkPalette.setColor(QPalette::BrightText, QColor(160, 140, 100));   // Muted gold
	darkPalette.setColor(QPalette::Link, QColor(70, 110, 160));          // Darker blue
	darkPalette.setColor(QPalette::Highlight, QColor(80, 50, 120));      // Darker purple highlight
	darkPalette.setColor(QPalette::HighlightedText, QColor(180, 180, 160)); // Muted highlight

	qApp->setPalette(darkPalette);
	qApp->setStyle(QStyleFactory::create("Fusion"));

	// Set bold gaming font
	QFont gameFont;
	// Try gaming-style fonts, fallback to system bold fonts
	QStringList fontFamilies = {"Orbitron", "Rajdhani", "Teko", "Press Start 2P", 
								 "Chakra Petch", "Saira", "Exo 2", "Michroma",
								 "Arial Black", "Impact", "Bahnschrift", "Segoe UI Black"};
	bool fontSet = false;
	for (const QString& fontFamily : fontFamilies) {
		gameFont.setFamily(fontFamily);
		if (QFontInfo(gameFont).family() == fontFamily) {
			fontSet = true;
			break;
		}
	}
	if (!fontSet) {
		gameFont.setFamily("Arial");
	}
	gameFont.setPointSize(10);
	gameFont.setWeight(QFont::Bold);
	gameFont.setLetterSpacing(QFont::AbsoluteSpacing, 0.5);
	qApp->setFont(gameFont);
}

void MainWindow::onRenderClicked() {
	if (m_isRendering) {
		QMessageBox::warning(this, "Render In Progress", "A render is already in progress!");
		return;
	}

	// Get parameters
	bool useGPU = m_renderModeCombo->currentData().toBool();

	int width, height;
	if (m_qualityPresetCombo->currentIndex() == 6) {
		// Custom - use advanced tab values
		width = m_widthSpinBox->value();
		height = m_heightSpinBox->value();
	} else {
		// Use resolution combo
		QSize res = m_resolutionCombo->currentData().toSize();
		width = res.width();
		height = res.height();
	}

	int samples = m_samplesSpinBox->value();
	int maxDepth = m_maxDepthSpinBox->value();
	QString outputPath = m_outputPathEdit->text();

	// Create and start render thread
	m_renderThread = new RenderThread(this);
	m_renderThread->setParameters(useGPU, width, height, samples, maxDepth, outputPath);

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
	// Could add a log window in the future
	qDebug() << message;
}
