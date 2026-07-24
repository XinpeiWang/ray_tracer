#include "mainwindow.h"
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
#include <QDateTime>
#include <QScrollArea>
#include <QScreen>
#include <QTimer>
#include <QAbstractItemView>

void MainWindow::createBasicTab() {
	QWidget *basicTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(basicTab);

	// --- Scene selection ---
	QGroupBox *sceneGroup = new QGroupBox("Scene", basicTab);
	QVBoxLayout *sceneGroupLayout = new QVBoxLayout(sceneGroup);
	sceneGroupLayout->setContentsMargins(12, 24, 12, 12);
	sceneGroupLayout->setSpacing(10);

	QHBoxLayout *sceneRow = new QHBoxLayout();
	m_sceneCombo = new QComboBox(basicTab);
	{
		int count = 0;
		const SceneDesc* scenes = get_all_scenes(&count);
		for (int i = 0; i < count; ++i)
			m_sceneCombo->addItem(scenes[i].name, scenes[i].id);
	}
	styleComboBox(m_sceneCombo);
	sceneRow->addWidget(new QLabel("Scene:"));
	sceneRow->addWidget(m_sceneCombo, 1);
	sceneGroupLayout->addLayout(sceneRow);

	m_sceneInfoLabel = new QLabel(basicTab);
	m_sceneInfoLabel->setWordWrap(true);
	m_sceneInfoLabel->setStyleSheet(
		"QLabel {"
		"  color: #B0B0B0;"
		"  background-color: #2E2E2E;"
		"  border: 1px solid #404040;"
		"  border-radius: 4px;"
		"  padding: 6px 10px;"
		"  font-size: 11px;"
		"}"
	);
	sceneGroupLayout->addWidget(m_sceneInfoLabel);

	connect(m_sceneCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onSceneChanged);

	layout->addWidget(sceneGroup);

	// --- Render mode (Image vs Video) + GPU/CPU ---
	QGroupBox *modeGroup = new QGroupBox("Render Mode", basicTab);
	QFormLayout *modeFormLayout = new QFormLayout(modeGroup);
	modeFormLayout->setVerticalSpacing(14);
	modeFormLayout->setHorizontalSpacing(10);
	modeFormLayout->setContentsMargins(15, 28, 15, 15);

	m_modeCombo = new QComboBox(basicTab);
	m_modeCombo->addItem("🖼️ Render Single Image");
	m_modeCombo->addItem("🎬 Generate Video");
	m_modeCombo->setCurrentIndex(0);
	styleComboBox(m_modeCombo);
	connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onModeChanged);
	modeFormLayout->addRow("Output Mode:", m_modeCombo);

	m_renderModeCombo = new QComboBox(basicTab);
	m_renderModeCombo->addItem("🎮 GPU (CUDA) - Fast", true);
	m_renderModeCombo->addItem("🖥️ CPU - High Quality", false);
	styleComboBox(m_renderModeCombo);
	modeFormLayout->addRow("Renderer:", m_renderModeCombo);

	layout->addWidget(modeGroup);

	// --- Render settings (quality / resolution / samples) ---
	QGroupBox *renderGroup = new QGroupBox("Render Settings", basicTab);
	QFormLayout *renderLayout = new QFormLayout(renderGroup);
	renderLayout->setVerticalSpacing(15);
	renderLayout->setHorizontalSpacing(10);
	renderLayout->setContentsMargins(15, 28, 15, 15);

	// (Render Mode row already moved above — skip re-adding it)

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
	outputLayout->setContentsMargins(15, 28, 15, 15);

	QHBoxLayout *pathLayout = new QHBoxLayout();
	// Use timestamped filename to avoid caching issues
	QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
	QString defaultPath = QDir::homePath() + "/Desktop/render_" + timestamp + ".png";
	m_outputPathEdit = new QLineEdit(QDir::toNativeSeparators(defaultPath), basicTab);
	m_outputPathEdit->setStyleSheet(
		"QLineEdit { font-size: 11pt; padding: 6px 8px; min-height: 32px; }"
	);
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
	// Far back view for full scene visibility
	m_cameraPresetCombo->addItem("Front View (Outside)", QVariant::fromValue(QVector3D(278, 278, -800)));

	// Inside views: camera positioned near walls, all looking toward center
	m_cameraPresetCombo->addItem("Inside Front", QVariant::fromValue(QVector3D(278, 278, 50)));    // Near Z=0 opening
	m_cameraPresetCombo->addItem("Inside Back", QVariant::fromValue(QVector3D(278, 278, 500)));    // Near Z=555 back wall
	m_cameraPresetCombo->addItem("Right Wall (Green)", QVariant::fromValue(QVector3D(500, 278, 278))); // Near X=555 green wall
	m_cameraPresetCombo->addItem("Left Wall (Red)", QVariant::fromValue(QVector3D(50, 278, 278)));     // Near X=0 red wall

	// Corner views: diagonal perspectives from inside the box
	m_cameraPresetCombo->addItem("Floor Corner", QVariant::fromValue(QVector3D(100, 50, 100)));    // Low angle, near floor
	m_cameraPresetCombo->addItem("Ceiling Corner", QVariant::fromValue(QVector3D(450, 500, 450))); // High angle, near ceiling

	// Custom: allows manual X/Y/Z input via spinboxes below
	m_cameraPresetCombo->addItem("Custom", QVariant::fromValue(QVector3D(278, 278, -800)));

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
	m_cameraPosX->installEventFilter(m_wheelFilter);
	cameraLayout->addRow("Camera X:", m_cameraPosX);

	m_cameraPosY = new QDoubleSpinBox(advancedTab);
	m_cameraPosY->setRange(-2000, 2000);
	m_cameraPosY->setValue(278);  // Default Y: centered vertically
	m_cameraPosY->setSingleStep(10);
	m_cameraPosY->setEnabled(false);  // Disabled until "Custom" is selected
	m_cameraPosY->installEventFilter(m_wheelFilter);
	cameraLayout->addRow("Camera Y:", m_cameraPosY);

	m_cameraPosZ = new QDoubleSpinBox(advancedTab);
	m_cameraPosZ->setRange(-2000, 2000);
	m_cameraPosZ->setValue(-800);  // Default Z: far back view to match default preset
	m_cameraPosZ->setSingleStep(10);
	m_cameraPosZ->setEnabled(false);  // Disabled until "Custom" is selected
	m_cameraPosZ->installEventFilter(m_wheelFilter);
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
	// Dark background matches the app theme
	m_logTextEdit->setStyleSheet(
		"QTextEdit {"
		"  background-color: #1A1A1A;"
		"  border: 1px solid #404040;"
		"  border-radius: 4px;"
		"}"
	);

	layout->addWidget(m_logTextEdit);

	// Button bar: Copy | Save Log | Clear Log
	QHBoxLayout *btnLayout = new QHBoxLayout();
	btnLayout->setContentsMargins(0, 4, 0, 0);

	QString logBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; min-width: 160px; padding: 0px 20px; font-size: 11pt; }"
		"QPushButton:hover { background-color: #3A1050; border-color: #00FFFF; color: #00FFFF; }";

	QPushButton *copyButton = new QPushButton("📋 Copy All");
	copyButton->setStyleSheet(logBtnStyle);
	connect(copyButton, &QPushButton::clicked, [this]() {
		m_logTextEdit->selectAll();
		m_logTextEdit->copy();
		m_logTextEdit->moveCursor(QTextCursor::End);
	});

	QPushButton *saveButton = new QPushButton("💾 Save Log");
	saveButton->setStyleSheet(logBtnStyle);
	connect(saveButton, &QPushButton::clicked, [this]() {
		QString path = QFileDialog::getSaveFileName(this, "Save Log",
			QDir::homePath() + "/render_log.txt",
			"Text Files (*.txt);;All Files (*.*)");
		if (!path.isEmpty()) {
			QFile file(path);
			if (file.open(QFile::WriteOnly | QFile::Text)) {
				QTextStream out(&file);
				out << m_logTextEdit->toPlainText();
				file.close();
				onLogMessage(QString("[INFO] Log saved to %1").arg(path));
			}
		}
	});

	QPushButton *clearButton = new QPushButton("🗑 Clear Log");
	clearButton->setStyleSheet(logBtnStyle);
	connect(clearButton, &QPushButton::clicked, m_logTextEdit, &QTextEdit::clear);

	btnLayout->addWidget(copyButton);
	btnLayout->addWidget(saveButton);
	btnLayout->addStretch();
	btnLayout->addWidget(clearButton);
	layout->addLayout(btnLayout);

	m_tabWidget->addTab(logWidget, "Log Output");
}

void MainWindow::createVideoTab() {
	QWidget *videoTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(videoTab);

	// Video parameters group
	QGroupBox *videoGroup = new QGroupBox("Video Generation Settings", videoTab);
	QFormLayout *videoLayout = new QFormLayout(videoGroup);
	videoLayout->setVerticalSpacing(14);
	videoLayout->setHorizontalSpacing(10);
	videoLayout->setContentsMargins(15, 28, 15, 15);

	// Camera path selector
	m_cameraPathCombo = new QComboBox();
	m_cameraPathCombo->addItem("🔄 Orbit (Circular rotation)", "orbit");
	m_cameraPathCombo->addItem("➡️ Linear (Straight path)", "linear");
	m_cameraPathCombo->addItem("∞ Figure-8 (Lemniscate)", "figure8");
	m_cameraPathCombo->addItem("🌀 Spiral (Zoom-in)", "spiral");
	m_cameraPathCombo->setCurrentIndex(0);
	styleComboBox(m_cameraPathCombo);
	videoLayout->addRow("Camera Path:", m_cameraPathCombo);

	// Frame count
	m_videoFramesSpinBox = new QSpinBox();
	m_videoFramesSpinBox->setRange(10, 1000);
	m_videoFramesSpinBox->setValue(60);
	m_videoFramesSpinBox->setSuffix(" frames");
	styleSpinBox(m_videoFramesSpinBox);
	videoLayout->addRow("Frame Count:", m_videoFramesSpinBox);

	// FPS (frames per second)
	m_videoFPSSpinBox = new QSpinBox();
	m_videoFPSSpinBox->setRange(15, 120);
	m_videoFPSSpinBox->setValue(30);
	m_videoFPSSpinBox->setSuffix(" fps");
	styleSpinBox(m_videoFPSSpinBox);
	videoLayout->addRow("Frames Per Second:", m_videoFPSSpinBox);

	// Video duration info (calculated from frames/fps)
	m_videoInfoLabel = new QLabel();
	m_videoInfoLabel->setWordWrap(true);
	m_videoInfoLabel->setStyleSheet("QLabel { color: #00FFFF; font-style: italic; padding: 10px; }");

	// Update duration display when frames or FPS changes
	auto updateVideoDuration = [this]() {
		int frames = m_videoFramesSpinBox->value();
		int fps = m_videoFPSSpinBox->value();
		double duration = static_cast<double>(frames) / fps;
		QString cameraPath = m_cameraPathCombo->currentData().toString();

		m_videoInfoLabel->setText(QString(
			"<b>Video Duration:</b> %1 seconds<br>"
			"<b>Camera Path:</b> %2<br>"
			"<b>Output:</b> Frames will be saved to <code>output/frames/</code>"
		).arg(QString::number(duration, 'f', 1), cameraPath));
	};

	connect(m_videoFramesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoFPSSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_cameraPathCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateVideoDuration);
	updateVideoDuration();

	videoLayout->addRow("", m_videoInfoLabel);

	layout->addWidget(videoGroup);

	// Requirements info
	QGroupBox *requirementsGroup = new QGroupBox("ℹ️ Requirements", videoTab);
	QVBoxLayout *requirementsLayout = new QVBoxLayout(requirementsGroup);

	QLabel *requirementsInfo = new QLabel(
		"<b>Built-in Video Encoding:</b> Videos are automatically created using OpenCV (MP4V codec).<br>"
		"<small>No external tools required!</small><br><br>"
		"<b>Automatic Assembly:</b> After rendering all frames, the video will be automatically assembled and opened."
	);
	requirementsInfo->setOpenExternalLinks(true);
	requirementsInfo->setWordWrap(true);
	requirementsInfo->setStyleSheet("QLabel { color: #00FFFF; padding: 10px; }");
	requirementsLayout->addWidget(requirementsInfo);

	layout->addWidget(requirementsGroup);

	// Usage instructions
	QGroupBox *usageGroup = new QGroupBox("📖 Usage Instructions", videoTab);
	QVBoxLayout *usageLayout = new QVBoxLayout(usageGroup);

	QLabel *usageText = new QLabel(
		"<b>Step 1:</b> Configure video settings (camera path, frames, FPS)<br>"
		"<b>Step 2:</b> Configure quality settings in Basic/Advanced tabs<br>"
		"<b>Step 3:</b> Click START VIDEO RENDER and wait<br>"
		"<b>Step 4:</b> Video automatically assembles and opens when done!<br><br>"
		"<b>Tips:</b><br>"
		"• Use GPU mode for faster rendering<br>"
		"• Lower samples/pixel for quick previews (10-50)<br>"
		"• Higher samples/pixel for production quality (100-500)<br>"
		"• Typical render time: 1-5 minutes (GPU), 15-60 minutes (CPU)"
	);
	usageText->setWordWrap(true);
	usageText->setStyleSheet("QLabel { color: #00FFFF; padding: 10px; line-height: 1.5; }");
	usageLayout->addWidget(usageText);

	layout->addWidget(usageGroup);

	layout->addStretch();

	// Scroll area for video tab
	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(videoTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);

	m_tabWidget->addTab(scrollArea, "Video Settings");
}
