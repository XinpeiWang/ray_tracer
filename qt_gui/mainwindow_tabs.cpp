#include "mainwindow.h"
#include "icon_tint.h"

#include "../src/shared/scene_descriptor.h"

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
#include <QScreen>
#include <QTimer>
#include <QAbstractItemView>
#include <QIcon>
#include <QDesktopServices>
#include <QUrl>
#include <QSplitter>
#include <cmath>

// Refills the scene dropdown with just one category's scenes.
//
// Ids are category letter + number now (e.g. "B10"), not contiguous ints
// (see scene_registry.h's SceneDescriptor::id comment), so walking registry
// POSITIONS 0..sceneCount() and resolving each one's id via
// sceneIdAtIndex() is how enumeration works now. The id is stored as item
// data and everything downstream reads THAT, never the row index, which is
// what makes filtering the list safe: a scene keeps its identity no matter
// which position it lands in.
void MainWindow::populateSceneCombo(const QString &category) {
	if (!m_sceneCombo) return;

	// Silent while refilling: clear() plus one addItem() per scene would emit
	// currentIndexChanged repeatedly, running onSceneChanged - which rewrites
	// the SPP box and camera - once per insertion, on scenes the user never
	// chose. The caller issues exactly one update for the final selection.
	const QSignalBlocker blocker(m_sceneCombo);
	m_sceneCombo->clear();

	// requiresFiles==true on the "Requires External Files" tab, false on
	// "Self-Contained" - matches rebuildCategoryTabs()'s own reading of the
	// same tab bar, and defaults to false (self-contained) if the bar
	// somehow isn't built yet, the safer of the two defaults for a fresh
	// checkout.
	const bool wantRequiresFiles = m_sceneAvailabilityTabs && m_sceneAvailabilityTabs->currentIndex() == 1;

	const int count = SceneMetadataClient::sceneCount();
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		if (SceneMetadataClient::sceneCategory(id) != category) continue;
		if (SceneMetadataClient::sceneRequiresFiles(id) != wantRequiresFiles) continue;
		m_sceneCombo->addItem(
			QString("[%1] %2").arg(id).arg(SceneMetadataClient::sceneName(id)), id);
	}
}

// Rebuilds m_sceneCategoryTabs for the given availability filter - see this
// function's own declaration comment (mainwindow.h) for the two-level filter
// design. Tries to keep the same letter category selected across a rebuild
// (e.g. toggling availability while on "Geometry" should land back on
// "Geometry" if it still has a qualifying scene), falling back to index 0
// otherwise - mirrors createBasicTab()'s own initial-fill fallback.
void MainWindow::rebuildCategoryTabs(bool requiresFiles) {
	if (!m_sceneCategoryTabs) return;

	const QString previousCategory = m_sceneCategoryTabs->count() > 0
		? m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString()
		: QString();

	const QSignalBlocker blocker(m_sceneCategoryTabs);
	while (m_sceneCategoryTabs->count() > 0)
		m_sceneCategoryTabs->removeTab(0);

	const int sceneCount = SceneMetadataClient::sceneCount();
	int restoredTab = -1;
	for (std::size_t i = 0; i < SceneCategories::kAllCount; ++i) {
		const QString category = QString::fromUtf8(SceneCategories::kAll[i]);
		int inCategory = 0;
		for (int j = 0; j < sceneCount; ++j) {
			const QString id = SceneMetadataClient::sceneIdAtIndex(j);
			if (SceneMetadataClient::sceneCategory(id) != category) continue;
			if (SceneMetadataClient::sceneRequiresFiles(id) != requiresFiles) continue;
			++inCategory;
		}
		if (inCategory == 0) continue;

		const int tab = m_sceneCategoryTabs->addTab(category);
		m_sceneCategoryTabs->setTabData(tab, category);
		m_sceneCategoryTabs->setTabToolTip(tab,
			QString("%1 scene%2").arg(inCategory).arg(inCategory == 1 ? "" : "s"));
		if (category == previousCategory) restoredTab = tab;
	}

	if (m_sceneCategoryTabs->count() > 0)
		m_sceneCategoryTabs->setCurrentIndex(restoredTab >= 0 ? restoredTab : 0);
}

void MainWindow::createBasicTab() {
	QWidget *basicTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(basicTab);
	layout->setSpacing(12);
	layout->setContentsMargins(12, 12, 12, 12);

	// --- Scene selection ---
	QGroupBox *sceneGroup = new QGroupBox("Scene", basicTab);
	styleGroupBox(sceneGroup);
	QVBoxLayout *sceneGroupLayout = new QVBoxLayout(sceneGroup);
	sceneGroupLayout->setContentsMargins(12, 20, 12, 10);
	sceneGroupLayout->setSpacing(8);

	// A category filter above the dropdown. With 69 scenes and counting, one
	// flat list had become a scroll-and-hunt exercise; the tabs cut it to at
	// most a couple of dozen at a time. The categories come from the registry
	// itself (SceneDescriptor::category, served by scene_metadata.dll), not
	// from a table here - a GUI-local copy is exactly the duplication that
	// drifted before and got scene_descriptor.h's mirror table deleted.
	const int sceneCount = SceneMetadataClient::sceneCount();
	if (sceneCount <= 0) {
		QMessageBox::critical(basicTab, "Scene Metadata Unavailable",
			"Could not load scene_metadata.dll, so the scene list is empty. "
			"Make sure scene_metadata.dll is present alongside RayTracerGUI.exe.");
	}

	// A second, higher-level filter above the letter-category tabs: every
	// scene splits into "Self-Contained" (renders in a fresh checkout) or
	// "Requires External Files" (SceneMetadataClient::sceneRequiresFiles()),
	// independently of SceneCategories - see mainwindow.h's own comment on
	// m_sceneAvailabilityTabs for why this is a per-scene split layered on
	// top of the categories rather than a coarser replacement for them.
	// Defaults to "Self-Contained" (index 0) - the bucket that reliably
	// renders for a user who just cloned the repo.
	m_sceneAvailabilityTabs = new QTabBar(basicTab);
	m_sceneAvailabilityTabs->setObjectName("sceneCategoryTabs");
	m_sceneAvailabilityTabs->setDrawBase(false);
	m_sceneAvailabilityTabs->setExpanding(false);
	{
		int selfContainedCount = 0, requiresFilesCount = 0;
		for (int i = 0; i < sceneCount; ++i) {
			const QString id = SceneMetadataClient::sceneIdAtIndex(i);
			if (SceneMetadataClient::sceneRequiresFiles(id)) ++requiresFilesCount;
			else ++selfContainedCount;
		}
		const int selfTab = m_sceneAvailabilityTabs->addTab("Self-Contained");
		m_sceneAvailabilityTabs->setTabToolTip(selfTab,
			QString("%1 scene%2 - no extra downloads needed")
				.arg(selfContainedCount).arg(selfContainedCount == 1 ? "" : "s"));
		const int filesTab = m_sceneAvailabilityTabs->addTab("Requires External Files");
		m_sceneAvailabilityTabs->setTabToolTip(filesTab,
			QString("%1 scene%2 - needs assets not included in a fresh checkout")
				.arg(requiresFilesCount).arg(requiresFilesCount == 1 ? "" : "s"));
	}
	sceneGroupLayout->addWidget(m_sceneAvailabilityTabs);

	m_sceneCategoryTabs = new QTabBar(basicTab);
	m_sceneCategoryTabs->setObjectName("sceneCategoryTabs");
	m_sceneCategoryTabs->setDrawBase(false);
	m_sceneCategoryTabs->setExpanding(false);
	// Eight categories fit at this window's normal width, but the tab bar is
	// inside a resizable group box - scroll buttons beat silently clipping the
	// last category off the right edge when it isn't.
	m_sceneCategoryTabs->setUsesScrollButtons(true);
	// SceneCategories::kAll drives the ORDER (a curated reading order, not
	// the order categories happen to first appear in the registry).
	// Categories with no scenes IN THE CURRENT AVAILABILITY BUCKET are
	// skipped rather than shown as an empty tab - same reasoning
	// createBasicTab() already applied for categories with zero scenes at
	// all (see rebuildCategoryTabs()'s own comment), just re-evaluated
	// per bucket instead of once.
	rebuildCategoryTabs(/*requiresFiles=*/false);
	sceneGroupLayout->addWidget(m_sceneCategoryTabs);

	QHBoxLayout *sceneRow = new QHBoxLayout();
	m_sceneCombo = new QComboBox(basicTab);
	styleComboBox(m_sceneCombo);
	sceneRow->addWidget(new QLabel("Scene:"));
	sceneRow->addWidget(m_sceneCombo, 1);
	sceneGroupLayout->addLayout(sceneRow);

	// Fill the dropdown for whichever category the bar opened on.
	if (m_sceneCategoryTabs->count() > 0)
		populateSceneCombo(m_sceneCategoryTabs->tabData(0).toString());

	connect(m_sceneAvailabilityTabs, &QTabBar::currentChanged, this, [this](int tab) {
		if (tab < 0) return;
		rebuildCategoryTabs(/*requiresFiles=*/tab == 1);
		// rebuildCategoryTabs() picks a category tab but (like the category
		// bar's own currentChanged handler below) does not refill the combo
		// itself - QTabBar::currentChanged only fires on an actual index
		// CHANGE, which rebuildCategoryTabs() causes most of the time (tab
		// counts/order shift between buckets) but not always (e.g. toggling
		// back to a bucket that happens to restore the same tab index by
		// coincidence) - so this always refills explicitly rather than
		// relying on that signal firing.
		if (m_sceneCategoryTabs->count() > 0)
			populateSceneCombo(m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString());
		else
			m_sceneCombo->clear();
		onSceneChanged(m_sceneCombo->currentIndex());
	});

	connect(m_sceneCategoryTabs, &QTabBar::currentChanged, this, [this](int tab) {
		if (tab < 0) return;
		populateSceneCombo(m_sceneCategoryTabs->tabData(tab).toString());
		// populateSceneCombo() deliberately stays silent, so the one update for
		// the newly selected scene is issued here - otherwise switching category
		// would leave the description, SPP and camera describing the old scene.
		onSceneChanged(m_sceneCombo->currentIndex());
	});

	m_sceneInfoLabel = new QLabel(basicTab);
	m_sceneInfoLabel->setWordWrap(true);
	// Appearance lives in the global stylesheet under this name, so it follows
	// the active theme without anything here having to know a colour.
	m_sceneInfoLabel->setObjectName("sceneInfo");
	sceneGroupLayout->addWidget(m_sceneInfoLabel);

	connect(m_sceneCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onSceneChanged);

	layout->addWidget(sceneGroup);

	// --- Render settings: output mode, GPU/CPU, quality, resolution ---
	// One group instead of separate "Render Mode" + "Render Settings" boxes -
	// they're all "how do I want this rendered" and splitting them just cost
	// an extra group box's worth of border/title chrome for no real benefit.
	QGroupBox *renderGroup = new QGroupBox("Render Settings", basicTab);
	styleGroupBox(renderGroup);
	QFormLayout *renderLayout = new QFormLayout(renderGroup);
	renderLayout->setVerticalSpacing(10);
	renderLayout->setHorizontalSpacing(10);
	renderLayout->setContentsMargins(15, 22, 15, 12);

	m_modeCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_modeCombo, ":/icons/image.svg", "Render Single Image", {}, m_activeTheme.textBody);
	icon_tint::addItem(m_modeCombo, ":/icons/video.svg", "Generate Video", {}, m_activeTheme.textBody);
	m_modeCombo->setCurrentIndex(0);
	styleComboBox(m_modeCombo);
	connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onModeChanged);
	renderLayout->addRow("Output Mode:", m_modeCombo);

	m_modeCombo->setToolTip(
		"Single Image renders one frame.\n"
		"Generate Video renders a camera path frame by frame and assembles an MP4.");

	m_renderModeCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_renderModeCombo, ":/icons/gpu.svg", "GPU (CUDA) - Fast", true, m_activeTheme.textBody);
	icon_tint::addItem(m_renderModeCombo, ":/icons/cpu.svg", "CPU - High Quality", false, m_activeTheme.textBody);
	styleComboBox(m_renderModeCombo);
	// Tooltips carry what the label cannot: the actual trade-off, not a repeat
	// of the visible text.
	m_renderModeCombo->setToolTip(
		"GPU: OptiX hardware ray tracing — typically orders of magnitude faster.\n"
		"CPU: importance-sampled path tracer — supports every scene and material,\n"
		"including the handful the GPU backend does not implement.");
	renderLayout->addRow("Renderer:", m_renderModeCombo);

	m_gpuBackendCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_gpuBackendCombo, ":/icons/gpu.svg", "Recursive (Default)", false, m_activeTheme.textBody);
	icon_tint::addItem(m_gpuBackendCombo, ":/icons/gpu.svg", "Wavefront (Experimental)", true, m_activeTheme.textBody);
	m_gpuBackendCombo->setCurrentIndex(0);
	styleComboBox(m_gpuBackendCombo);
	m_gpuBackendCombo->setToolTip(
		"Recursive: one thread per pixel, the default GPU path tracer — broad, battle-tested coverage.\n"
		"Wavefront: splits each bounce into separate queue-passed kernel launches — better GPU\n"
		"utilization on complex/divergent scenes, but a newer, less exercised code path.\n"
		"Only applies when Renderer is set to GPU.");
	// Starts disabled/enabled in sync with the initial Renderer selection (GPU,
	// index 0/true above) - the connect() in the constructor keeps it synced
	// afterwards whenever the user changes Renderer.
	m_gpuBackendCombo->setEnabled(m_renderModeCombo->currentData().toBool());
	renderLayout->addRow("GPU Backend:", m_gpuBackendCombo);

	// Quality preset
	m_qualityPresetCombo = new QComboBox(basicTab);
	m_qualityPresetCombo->addItem("Draft (Very Fast)", 0);
	m_qualityPresetCombo->addItem("Preview (Fast)", 1);
	m_qualityPresetCombo->addItem("Good (Balanced)", 2);
	m_qualityPresetCombo->addItem("High (Slow)", 3);
	m_qualityPresetCombo->addItem("Ultra (Very Slow)", 4);
	m_qualityPresetCombo->addItem("Maximum (Extreme)", 5);
	m_qualityPresetCombo->addItem("Custom", 6);
	m_qualityPresetCombo->setCurrentIndex(2); // Default to Good
	connect(m_qualityPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onQualityPresetChanged);
	styleComboBox(m_qualityPresetCombo);
	// The preset names are relative ("Ultra", "Maximum") and say nothing
	// quantitative; spell out what each actually sets. Keep in sync with
	// onQualityPresetChanged()'s presetSamples/presetDepth tables.
	m_qualityPresetCombo->setToolTip(
		"Samples per pixel / max ray depth:\n"
		"  Draft    25 spp,  depth 10\n"
		"  Preview  50 spp,  depth 20\n"
		"  Good    100 spp,  depth 50\n"
		"  High    500 spp,  depth 50\n"
		"  Ultra  1000 spp,  depth 100\n"
		"  Maximum 5000 spp, depth 100\n"
		"Custom leaves the Advanced tab values untouched.\n"
		"Render time scales roughly linearly with samples per pixel.");
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
	styleGroupBox(outputGroup);
	QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);
	outputLayout->setSpacing(8);
	outputLayout->setContentsMargins(15, 20, 15, 12);

	QHBoxLayout *pathLayout = new QHBoxLayout();
	// Use timestamped filename to avoid caching issues
	QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
	QString defaultPath = QDir::homePath() + "/Desktop/render_" + timestamp + ".png";
	m_outputPathEdit = new QLineEdit(QDir::toNativeSeparators(defaultPath), basicTab);
	m_outputPathEdit->setStyleSheet(
		"QLineEdit { font-size: 11pt; padding: 6px 8px; min-height: 32px; }"
	);
	m_outputPathEdit->setToolTip(
		"Where the rendered image is written. A .png is always saved alongside\n"
		"the raw .ppm, and it is the .png the Preview tab displays.");
	// Trailing ellipsis (U+2026, not three periods) marks an action that needs
	// further input before it completes - a file dialog here. Buttons that act
	// immediately (Open Output Folder, Clear Log) deliberately have none.
	m_browseButton = new QPushButton("&Browse…", basicTab);
	m_browseButton->setToolTip("Choose the output file name and location");
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
	// Named so the global stylesheet can paint a theme's decorative motif here.
	// It has to be the scroll area rather than QTabWidget::pane: the pane is
	// covered edge to edge by this widget, so a background set on it is never
	// seen. QAbstractScrollArea is also the one thing Qt documents as
	// supporting background-attachment, which is what keeps the motif still
	// while the settings scroll past.
	scrollArea->setObjectName("tabScroll");
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, "Basic Settings");
}

void MainWindow::createAdvancedTab() {
	QWidget *advancedTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(advancedTab);
	layout->setSpacing(14);  // Space between group boxes
	layout->setContentsMargins(12, 12, 12, 12);

	QGroupBox *advancedGroup = new QGroupBox("Advanced Parameters", advancedTab);
	styleGroupBox(advancedGroup);
	QFormLayout *formLayout = new QFormLayout(advancedGroup);
	formLayout->setVerticalSpacing(10);
	formLayout->setHorizontalSpacing(10);
	formLayout->setContentsMargins(15, 22, 15, 12);

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
	m_samplesSpinBox->setToolTip(
		"Rays traced per pixel. This is the main quality/time dial: noise falls\n"
		"as the square root of this value, so halving the noise costs about 4x\n"
		"the render time. Setting it here switches Quality to Custom.");
	formLayout->addRow("Samples per Pixel:", m_samplesSpinBox);

	// Max depth
	m_maxDepthSpinBox = new QSpinBox(advancedTab);
	m_maxDepthSpinBox->setRange(1, 100);
	m_maxDepthSpinBox->setValue(50);
	styleSpinBox(m_maxDepthSpinBox);
	m_maxDepthSpinBox->setToolTip(
		"How many times a ray may bounce before it is terminated. Low values\n"
		"darken glass and mirrors, which need many bounces to resolve; scenes\n"
		"of plain diffuse surfaces look the same well below the maximum.");
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
	styleGroupBox(cameraGroup);
	QFormLayout *cameraLayout = new QFormLayout(cameraGroup);
	cameraLayout->setVerticalSpacing(10);
	cameraLayout->setHorizontalSpacing(10);
	cameraLayout->setContentsMargins(15, 22, 15, 12);

	// Camera preset combo box
	// Each preset stores a direction*ratio QVector3D, NOT an absolute world
	// position: onCameraPresetChanged() scales it by m_currentSceneCamDistance
	// (the CURRENT scene's own recommended-camera distance from its lookat)
	// and offsets it from m_currentLookat*, so "Right Wall" lands at a
	// sensible position for whatever scene is active. These vectors were
	// derived from Cornell Box's own original hardcoded positions - e.g.
	// "Front View (Outside)" used to be the literal point (278,278,-800),
	// which is offset (0,0,-1078) from Cornell's lookat (278,278,278); divide
	// by Cornell's own recommended-camera distance (1078, the default
	// m_currentSceneCamDistance below) to get this preset's direction*ratio
	// vector (0,0,-1.0) - a pure "straight back, at 1x the scene's own
	// default viewing distance" direction that means the same thing
	// regardless of scene scale. The others below were derived the same way,
	// which is why "Front View" ends up at ratio 1.0 (it WAS the reference
	// distance) while the inside/corner views are fractions of it. Previously
	// every preset stored its literal Cornell-Box position directly, so
	// selecting e.g. "Right Wall" while viewing a much smaller scene (like
	// scene 1's spheres, which sit within roughly +-15 units of the origin)
	// put the camera at a literal (500,278,278) - wildly outside that
	// scene's geometry.
	m_cameraPresetCombo = new QComboBox(advancedTab);

	// Default view: straight back from lookat, at the scene's own default
	// viewing distance (ratio 1.0) - matches Cornell Box's own recommended
	// camera exactly, since that's what this ratio was derived from.
	m_cameraPresetCombo->addItem("Front View (Outside)", QVariant::fromValue(QVector3D(0.0f, 0.0f, -1.0f)));

	// Inside views: camera positioned near walls, all looking toward center
	m_cameraPresetCombo->addItem("Inside Front", QVariant::fromValue(QVector3D(0.0f, 0.0f, -0.211503f)));   // Near Z=0 opening
	m_cameraPresetCombo->addItem("Inside Back", QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.205937f)));     // Near Z=555 back wall
	m_cameraPresetCombo->addItem("Right Wall (Green)", QVariant::fromValue(QVector3D(0.205937f, 0.0f, 0.0f))); // Near X=555 green wall
	m_cameraPresetCombo->addItem("Left Wall (Red)", QVariant::fromValue(QVector3D(-0.211503f, 0.0f, 0.0f)));   // Near X=0 red wall

	// Corner views: diagonal perspectives from inside the box
	m_cameraPresetCombo->addItem("Floor Corner", QVariant::fromValue(QVector3D(-0.165121f, -0.211503f, -0.165121f)));  // Low angle, near floor
	m_cameraPresetCombo->addItem("Ceiling Corner", QVariant::fromValue(QVector3D(0.159555f, 0.205937f, 0.159555f)));   // High angle, near ceiling

	// Custom: allows manual X/Y/Z input via spinboxes below. Its itemData is
	// never read (onCameraPresetChanged skips the overwrite for Custom - see
	// its own comment), so this value is unused, but keep it a plausible
	// starting direction rather than leaving it as leftover absolute-position
	// data of a different shape than every other item now stores.
	m_cameraPresetCombo->addItem("Custom", QVariant::fromValue(QVector3D(0.0f, 0.0f, -1.0f)));

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
	styleSpinBox(m_cameraPosX);
	cameraLayout->addRow("Camera X:", m_cameraPosX);

	m_cameraPosY = new QDoubleSpinBox(advancedTab);
	m_cameraPosY->setRange(-2000, 2000);
	m_cameraPosY->setValue(278);  // Default Y: centered vertically
	m_cameraPosY->setSingleStep(10);
	m_cameraPosY->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosY);
	cameraLayout->addRow("Camera Y:", m_cameraPosY);

	m_cameraPosZ = new QDoubleSpinBox(advancedTab);
	m_cameraPosZ->setRange(-2000, 2000);
	m_cameraPosZ->setValue(-800);  // Default Z: far back view to match default preset
	m_cameraPosZ->setSingleStep(10);
	m_cameraPosZ->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosZ);
	cameraLayout->addRow("Camera Z:", m_cameraPosZ);

	// Distance from the current scene's look-at point. Adjusting this moves
	// the camera along its EXISTING viewing direction to the new distance
	// (see onCameraDistanceChanged) - a quick way to zoom in/out without
	// having to work out new X/Y/Z coordinates by hand. Only meaningful (and
	// only enabled) alongside the X/Y/Z spinboxes for "Custom"; its value is
	// kept in sync (not user-editable-then-stale) whenever the scene or
	// preset changes, via refreshCameraDistanceDisplay().
	m_cameraDistance = new QDoubleSpinBox(advancedTab);
	m_cameraDistance->setRange(0.01, 5000);
	m_cameraDistance->setValue(1078);  // Matches the default preset's distance from Cornell Box's lookat
	m_cameraDistance->setSingleStep(10);
	m_cameraDistance->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraDistance);
	cameraLayout->addRow("Distance from Center:", m_cameraDistance);

	// Connect preset combo to handler that updates spinboxes and enables/disables manual input
	// Connection made AFTER all widgets are created to avoid null pointer issues
	connect(m_cameraPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onCameraPresetChanged);
	connect(m_cameraDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			this, &MainWindow::onCameraDistanceChanged);

	// Initialize the spinboxes with the default preset (index 0: "Front View (Outside)")
	onCameraPresetChanged(0);

	layout->addWidget(cameraGroup);
	layout->addStretch();

	// Wrap the tab content in a scroll area for better responsiveness
	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(advancedTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	// Named so the global stylesheet can paint a theme's decorative motif here.
	// It has to be the scroll area rather than QTabWidget::pane: the pane is
	// covered edge to edge by this widget, so a background set on it is never
	// seen. QAbstractScrollArea is also the one thing Qt documents as
	// supporting background-attachment, which is what keeps the motif still
	// while the settings scroll past.
	scrollArea->setObjectName("tabScroll");
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, "Advanced Settings");
}

void MainWindow::createPreviewTab() {
	QWidget *previewWidget = new QWidget();
	QHBoxLayout *outerLayout = new QHBoxLayout(previewWidget);
	outerLayout->setContentsMargins(12, 12, 12, 12);
	outerLayout->setSpacing(0);

	// Image on the left in a large, dominant pane; info/buttons in a narrow
	// sidebar on the right, so the image gets most of the tab's space instead
	// of splitting height with a full-width info/button strip underneath it.
	// A QSplitter (not a fixed QHBoxLayout split) so the user can still drag
	// the sidebar narrower/wider if they want even more image space.
	QSplitter *splitter = new QSplitter(Qt::Horizontal, previewWidget);
	splitter->setChildrenCollapsible(false);
	outerLayout->addWidget(splitter);

	// Shows the rendered PNG scaled to fit (see main.cpp's Format Conversion
	// step, which always writes a same-basename .png next to a successful
	// render's .ppm output) - populated by onRenderComplete() instead of
	// this app shelling out to the OS's default image viewer for every render.
	m_previewLabel = new ScaledImageLabel();
	m_previewLabel->setPlaceholderText("No render yet — start a render to see a preview here.");
	m_previewLabel->setMinimumSize(200, 200);
	// Styled globally by class name - see the ScaledImageLabel rule.
	splitter->addWidget(m_previewLabel);

	QWidget *sidebar = new QWidget();
	sidebar->setMinimumWidth(200);
	sidebar->setMaximumWidth(320);
	QVBoxLayout *sideLayout = new QVBoxLayout(sidebar);
	sideLayout->setContentsMargins(12, 4, 0, 0);
	sideLayout->setSpacing(10);

	m_previewInfoLabel = new QLabel(sidebar);
	m_previewInfoLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewInfoLabel->setWordWrap(true);
	m_previewInfoLabel->setObjectName("previewInfo");
	sideLayout->addWidget(m_previewInfoLabel);

	// Selected scene's description - same text/source as the Basic Settings
	// tab's #sceneInfo box (see onSceneChanged()), kept in sync with the
	// scene combo rather than tied to a completed render, so it's already
	// showing what you're about to render before the first click.
	m_previewSceneDescLabel = new QLabel(sidebar);
	m_previewSceneDescLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewSceneDescLabel->setWordWrap(true);
	m_previewSceneDescLabel->setObjectName("previewSceneDesc");
	sideLayout->addWidget(m_previewSceneDescLabel);

	// Geometry only - colour and hover/focus states come from the global
	// theme so every secondary button behaves identically. Full sidebar
	// width and stacked vertically now that they're beside the image, not
	// centered in a horizontal strip underneath it.
	QString previewBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; padding: 0px 20px; font-size: 11pt; }";

	QPushButton *openFolderButton = new QPushButton("Open Output &Folder");
	icon_tint::apply(openFolderButton, ":/icons/folder.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openFolderButton->setStyleSheet(previewBtnStyle);
	openFolderButton->setToolTip("Show the folder containing the last render in Explorer");
	connect(openFolderButton, &QPushButton::clicked, this, [this]() {
		if (m_lastOutputPath.isEmpty()) return;
		QFileInfo fileInfo(m_lastOutputPath);
		QDesktopServices::openUrl(QUrl::fromLocalFile(fileInfo.absolutePath()));
	});
	sideLayout->addWidget(openFolderButton);

	QPushButton *openViewerButton = new QPushButton("Open in Default &Viewer");
	icon_tint::apply(openViewerButton, ":/icons/image.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openViewerButton->setStyleSheet(previewBtnStyle);
	openViewerButton->setToolTip("Open the rendered image in the system image viewer");
	connect(openViewerButton, &QPushButton::clicked, this, [this]() {
		if (m_lastPreviewImagePath.isEmpty()) return;
		QDesktopServices::openUrl(QUrl::fromLocalFile(m_lastPreviewImagePath));
	});
	sideLayout->addWidget(openViewerButton);

	sideLayout->addStretch(1);
	splitter->addWidget(sidebar);

	// Bias initial space toward the image - the sidebar only needs enough
	// width for its buttons/info text, everything else goes to the render.
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 0);
	splitter->setSizes({700, 220});

	m_previewTabIndex = m_tabWidget->addTab(previewWidget, "Preview");
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
	QPushButton *copyButton = new QPushButton("&Copy All");
	icon_tint::apply(copyButton, ":/icons/copy.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	copyButton->setStyleSheet(logBtnStyle);
	connect(copyButton, &QPushButton::clicked, this, &MainWindow::copyLogToClipboard);

	QPushButton *saveButton = new QPushButton("&Save Log…");
	icon_tint::apply(saveButton, ":/icons/save.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	saveButton->setStyleSheet(logBtnStyle);
	connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveLogToFile);

	QPushButton *clearButton = new QPushButton("C&lear Log");
	icon_tint::apply(clearButton, ":/icons/clear.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	clearButton->setStyleSheet(logBtnStyle);
	connect(clearButton, &QPushButton::clicked, this, &MainWindow::clearLog);

	btnLayout->addWidget(copyButton);
	btnLayout->addWidget(saveButton);
	btnLayout->addStretch();
	btnLayout->addWidget(clearButton);
	layout->addLayout(btnLayout);

	m_logTabIndex = m_tabWidget->addTab(logWidget, "Log Output");
}

void MainWindow::createVideoTab() {
	QWidget *videoTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(videoTab);
	layout->setSpacing(12);
	layout->setContentsMargins(12, 12, 12, 12);

	// Video parameters group
	QGroupBox *videoGroup = new QGroupBox("Video Generation Settings", videoTab);
	styleGroupBox(videoGroup);
	QFormLayout *videoLayout = new QFormLayout(videoGroup);
	videoLayout->setVerticalSpacing(10);
	videoLayout->setHorizontalSpacing(10);
	videoLayout->setContentsMargins(15, 22, 15, 12);

	// Camera path selector
	m_cameraPathCombo = new QComboBox();
	m_cameraPathCombo->addItem("Orbit (Circular rotation)", "orbit");
	m_cameraPathCombo->addItem("Linear (Straight path)", "linear");
	m_cameraPathCombo->addItem("Figure-8 (Lemniscate)", "figure8");
	m_cameraPathCombo->addItem("Spiral (Zoom-in)", "spiral");
	m_cameraPathCombo->setToolTip(
		"How the camera moves over the frame sequence:\n"
		"  Orbit     — full circle around the scene, always looking at its centre\n"
		"  Linear    — straight sweep past the scene\n"
		"  Figure-8  — lemniscate, crossing back through the middle\n"
		"  Spiral    — orbits while moving steadily closer\n"
		"Every path starts from the camera position on the Advanced tab.");
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

	// Movement speed multiplier - does NOT change the camera path itself (it
	// always completes the exact same full sweep: 1 rotation for
	// orbit/figure8, 2 for spiral, the whole start->end traversal for
	// linear). Instead it expands the actual number of rendered frames:
	// speed 0.5x renders 2x the Frame Count above, spreading the same
	// journey over more frames (and more real video time at the same fps),
	// so it looks slower without ever cutting the path short. speed 2x
	// renders half as many frames, covering the same journey faster.
	m_videoSpeedSpinBox = new QDoubleSpinBox();
	m_videoSpeedSpinBox->setRange(0.1, 5.0);
	m_videoSpeedSpinBox->setSingleStep(0.1);
	m_videoSpeedSpinBox->setDecimals(2);
	m_videoSpeedSpinBox->setValue(1.0);
	m_videoSpeedSpinBox->setSuffix("x");
	styleSpinBox(m_videoSpeedSpinBox);
	videoLayout->addRow("Movement Speed:", m_videoSpeedSpinBox);

	// Video duration info (calculated from frames/fps)
	m_videoInfoLabel = new QLabel();
	m_videoInfoLabel->setWordWrap(true);
	m_videoInfoLabel->setObjectName("videoInfo");

	// Update duration display when frames, FPS, speed, or path changes
	auto updateVideoDuration = [this]() {
		int baseFrames = m_videoFramesSpinBox->value();
		int fps = m_videoFPSSpinBox->value();
		double speed = m_videoSpeedSpinBox->value();
		QString cameraPath = m_cameraPathCombo->currentData().toString();

		// Mirrors main.cpp's render_frame_count derivation exactly, so this
		// preview matches what will actually be rendered.
		int actualFrames = qMax(1, static_cast<int>(std::llround(baseFrames / speed)));
		const int kMaxVideoFrames = 5000;
		bool capped = actualFrames > kMaxVideoFrames;
		if (capped) actualFrames = kMaxVideoFrames;
		double duration = static_cast<double>(actualFrames) / fps;

		QString framesLine = (actualFrames == baseFrames)
			? QString("%1 frames").arg(actualFrames)
			: QString("%1 frames (base %2 × 1/%3x speed)%4")
				.arg(actualFrames).arg(baseFrames).arg(QString::number(speed, 'f', 2))
				.arg(capped ? " - capped at 5000" : "");

		m_videoInfoLabel->setText(QString(
			"<b>Video Duration:</b> %1 seconds (%2)<br>"
			"<b>Camera Path:</b> %3, always completes its full sweep regardless of speed<br>"
			"<b>Output:</b> Frames will be saved to <code>output/frames/</code>"
		).arg(QString::number(duration, 'f', 1), framesLine, cameraPath));
	};

	connect(m_videoFramesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoFPSSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoSpeedSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), updateVideoDuration);
	connect(m_cameraPathCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateVideoDuration);
	updateVideoDuration();

	videoLayout->addRow("", m_videoInfoLabel);

	layout->addWidget(videoGroup);

	// Requirements info
	QGroupBox *requirementsGroup = new QGroupBox("ℹ️ Requirements", videoTab);
	QVBoxLayout *requirementsLayout = new QVBoxLayout(requirementsGroup);

	QLabel *requirementsInfo = new QLabel(
		"<b>Requires ffmpeg:</b> Video encoding uses ffmpeg (libx264), which must be installed and on your PATH.<br>"
		"<small>Get it from <a href=\"https://ffmpeg.org/download.html\">ffmpeg.org</a> if the render log reports it's missing.</small><br><br>"
		"<b>Automatic Assembly:</b> After rendering all frames, the video will be automatically assembled and opened."
	);
	requirementsInfo->setOpenExternalLinks(true);
	requirementsInfo->setWordWrap(true);
	requirementsInfo->setObjectName("mutedInfo");
	requirementsLayout->addWidget(requirementsInfo);

	layout->addWidget(requirementsGroup);

	// Usage instructions
	QGroupBox *usageGroup = new QGroupBox("Usage Instructions", videoTab);
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
	usageText->setObjectName("mutedInfo");
	usageLayout->addWidget(usageText);

	layout->addWidget(usageGroup);

	layout->addStretch();

	// Scroll area for video tab
	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(videoTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	// Named so the global stylesheet can paint a theme's decorative motif here.
	// It has to be the scroll area rather than QTabWidget::pane: the pane is
	// covered edge to edge by this widget, so a background set on it is never
	// seen. QAbstractScrollArea is also the one thing Qt documents as
	// supporting background-attachment, which is what keeps the motif still
	// while the settings scroll past.
	scrollArea->setObjectName("tabScroll");

	m_tabWidget->addTab(scrollArea, "Video Settings");
}
