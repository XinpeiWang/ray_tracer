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

// Refills the scene dropdown with just one category's scenes, further
// narrowed by m_sceneSearchBox's current text if any (name/id/description
// substring match) - see that member's own comment in mainwindow.h.
//
// Ids are category letter + number now (e.g. "B10"), not contiguous ints
// (see scene_registry.h's SceneDescriptor::id comment), so walking registry
// POSITIONS 0..sceneCount() and resolving each one's id via
// sceneIdAtIndex() is how enumeration works now. The id is stored as item
// data and everything downstream reads THAT, never the row index, which is
// what makes filtering the list safe: a scene keeps its identity no matter
// which position it lands in.
QStringList MainWindow::filteredSceneIds(const QString &category) const {
	QStringList result;

	// requiresFiles==true on the "Requires External Files" tab, false on
	// "Self-Contained" - matches rebuildCategoryTabs()'s own reading of the
	// same tab bar, and defaults to false (self-contained) if the bar
	// somehow isn't built yet, the safer of the two defaults for a fresh
	// checkout.
	const bool wantRequiresFiles = m_sceneAvailabilityTabs && m_sceneAvailabilityTabs->currentIndex() == 1;

	// A further substring narrowing on top of category/availability, not a
	// replacement for them - see m_sceneSearchBox's own comment.
	const QString searchTerm = m_sceneSearchBox ? m_sceneSearchBox->text().trimmed() : QString();

	const int count = SceneMetadataClient::sceneCount();
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		if (SceneMetadataClient::sceneCategory(id) != category) continue;
		if (SceneMetadataClient::sceneRequiresFiles(id) != wantRequiresFiles) continue;
		const QString name = SceneMetadataClient::sceneName(id);
		if (!searchTerm.isEmpty() &&
			!name.contains(searchTerm, Qt::CaseInsensitive) &&
			!id.contains(searchTerm, Qt::CaseInsensitive) &&
			!SceneMetadataClient::sceneDescription(id).contains(searchTerm, Qt::CaseInsensitive))
			continue;
		result << id;
	}
	return result;
}

void MainWindow::populateSceneCombo(const QString &category) {
	if (!m_sceneCombo) return;

	// Silent while refilling: clear() plus one addItem() per scene would emit
	// currentIndexChanged repeatedly, running onSceneChanged - which rewrites
	// the SPP box and camera - once per insertion, on scenes the user never
	// chose. The caller issues exactly one update for the final selection.
	const QSignalBlocker blocker(m_sceneCombo);
	m_sceneCombo->clear();

	for (const QString &id : filteredSceneIds(category)) {
		const QString text = QString("[%1] %2").arg(id).arg(SceneMetadataClient::sceneName(id));
		// The same "(i)" mark createInfoIcon() uses elsewhere, one per row -
		// so the dropdown itself shows there's a rendering-technique note to
		// read, not just the single info icon next to the "Scene:" label
		// (which only ever shows whichever scene is already selected). Only
		// for self-contained scenes though: scene_technique_notes.h is
		// explicitly scoped to those (requires_files == false) and has no
		// entry at all for the rest, so a mark on a "Requires External
		// Files" row would promise a note every hover could only ever answer
		// with the generic "not written yet" fallback - a plain addItem()
		// for those instead, same as before this per-row mark existed.
		if (SceneMetadataClient::sceneRequiresFiles(id)) {
			m_sceneCombo->addItem(text, id);
			continue;
		}
		// icon_tint::addItem() (not a plain combo->addItem()) so a theme
		// switch's restyleThemedWidgets() -> retintItems() sweep recolours
		// these the same way every other combo's icons already do.
		icon_tint::addItem(m_sceneCombo, ":/icons/info.svg", text, id, m_activeTheme.textBody);
		m_sceneCombo->setItemData(m_sceneCombo->count() - 1,
			sceneTooltipHtml(id, /*includeHeading=*/false), Qt::ToolTipRole);
	}
}

void MainWindow::populateSceneGrid(const QString &category) {
	if (!m_sceneGrid) return;

	const QSignalBlocker blocker(m_sceneGrid);
	m_sceneGrid->clear();

	static QIcon placeholderIcon(":/icons/image.svg");

	for (const QString &id : filteredSceneIds(category)) {
		QListWidgetItem *item = new QListWidgetItem(SceneMetadataClient::sceneName(id));
		item->setData(Qt::UserRole, id);
		const QString cachePath = thumbnailCachePath(id);
		item->setIcon(QFile::exists(cachePath) ? QIcon(cachePath) : placeholderIcon);
		// Heading (this tile's own label is just the name, not the id) plus
		// the technique note, same self-contained-only scoping as
		// populateSceneCombo() above - a "Requires External Files" scene has
		// no note to show, so its tooltip stays the plain id/name heading
		// instead of promising one that can only ever fall back to
		// "not written yet".
		item->setToolTip(SceneMetadataClient::sceneRequiresFiles(id)
			? wrapTooltipHtml(QString("[%1] %2").arg(id, SceneMetadataClient::sceneName(id)))
			: sceneTooltipHtml(id, /*includeHeading=*/true));
		m_sceneGrid->addItem(item);
	}
}

void MainWindow::populateSceneViews(const QString &category) {
	populateSceneCombo(category);
	populateSceneGrid(category);
}

QString MainWindow::thumbnailCachePath(const QString &sceneId) const {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + "/thumbnails";
	return QDir(dir).filePath(sceneId + ".png");
}

// Drives the availability tab, category tab, and m_sceneCombo to the scene
// matching `id` - see this function's own declaration comment (mainwindow.h)
// for why it exists (the video preset combo needs it). Unconditionally
// (re)builds the category tab set and repopulates the combo at the end
// rather than trying to predict which of the intermediate setCurrentIndex()
// calls below actually changed anything and fired their own signal chain -
// same "don't trust signal timing, just do the work explicitly" approach
// m_sceneCategoryTabs' own currentChanged handler already takes, just
// applied one level further out. A little redundant work on the (common)
// case where the target scene is already in the current bucket/category is
// a small, one-time cost for a user-initiated action, not a hot path.
void MainWindow::selectSceneById(const QString &id) {
	if (!m_sceneCombo || id.isEmpty()) return;

	const QString category = SceneMetadataClient::sceneCategory(id);
	if (category.isEmpty()) {
		onLogMessage(QString("WARNING: video preset points at unknown scene id \"%1\"").arg(id));
		return;
	}
	const bool requiresFiles = SceneMetadataClient::sceneRequiresFiles(id);

	// Blocked: both tab bars' own currentChanged handlers repopulate the
	// combo/grid and call onSceneChanged() on whatever scene the newly
	// selected bucket/category happens to default to - NOT `id` - so an
	// unblocked cascade here could transiently evaluate the wrong scene
	// (e.g. tripping onSceneChanged()'s one-directional GPU->CPU auto-
	// downgrade for an intermediate CPU-only scene, which would then stick
	// even once the real, GPU-capable target scene is selected below).
	// populateSceneViews()+onSceneChanged() at the end of this function
	// already do everything those handlers would have, for the correct
	// scene, so nothing is lost by silencing them here.
	if (m_sceneAvailabilityTabs) {
		const QSignalBlocker blocker(m_sceneAvailabilityTabs);
		m_sceneAvailabilityTabs->setCurrentIndex(requiresFiles ? 1 : 0);
	}
	rebuildCategoryTabs(requiresFiles);
	if (!m_sceneCategoryTabs) return;
	{
		const QSignalBlocker blocker(m_sceneCategoryTabs);
		for (int i = 0; i < m_sceneCategoryTabs->count(); ++i) {
			if (m_sceneCategoryTabs->tabData(i).toString() == category) {
				m_sceneCategoryTabs->setCurrentIndex(i);
				break;
			}
		}
	}

	populateSceneViews(category);
	const int itemIndex = m_sceneCombo->findData(id);
	if (itemIndex < 0) {
		onLogMessage(QString("WARNING: video preset's scene \"%1\" not found under category \"%2\"")
			.arg(id, category));
		return;
	}
	m_sceneCombo->setCurrentIndex(itemIndex);
	onSceneChanged(itemIndex);  // also syncs m_sceneGrid's highlighted tile
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
			tr("%n scene(s)", "", inCategory));
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

	// A category filter above the dropdown. With 78 scenes and counting, one
	// flat list had become a scroll-and-hunt exercise; the tabs cut it to at
	// most a couple of dozen at a time. The categories come from the registry
	// itself (SceneDescriptor::category, served by scene_metadata.dll), not
	// from a table here - a GUI-local copy is exactly the duplication that
	// drifted before and got scene_descriptor.h's mirror table deleted.
	const int sceneCount = SceneMetadataClient::sceneCount();
	if (sceneCount <= 0) {
#ifdef Q_OS_WIN
		QMessageBox::critical(basicTab, tr("Scene Metadata Unavailable"),
			tr("Could not load scene_metadata.dll, so the scene list is empty. "
			"Make sure scene_metadata.dll is present alongside RayTracerGUI.exe."));
#else
		QMessageBox::critical(basicTab, tr("Scene Metadata Unavailable"),
			tr("Could not load scene_metadata.dylib/.so, so the scene list is empty. "
			"Make sure scene_metadata.dylib/.so is present alongside RayTracerGUI."));
#endif
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
		int requiresFilesCount = 0;
		QStringList selfContainedIds;
		for (int i = 0; i < sceneCount; ++i) {
			const QString id = SceneMetadataClient::sceneIdAtIndex(i);
			if (SceneMetadataClient::sceneRequiresFiles(id)) ++requiresFilesCount;
			else selfContainedIds << id;
		}
		const int selfTab = m_sceneAvailabilityTabs->addTab(tr("Self-Contained"));
		m_sceneAvailabilityTabs->setTabToolTip(selfTab,
			tr("%n scene(s) - no extra downloads needed", "", selfContainedIds.size()));
		const int filesTab = m_sceneAvailabilityTabs->addTab(tr("Requires External Files"));
		m_sceneAvailabilityTabs->setTabToolTip(filesTab,
			tr("%n scene(s) - needs assets not included in a fresh checkout", "", requiresFilesCount));

#ifndef QT_NO_DEBUG
		// One-time drift guard against scene_technique_notes.h - see that
		// header's own comment on warnIfOutOfSync() for why this lives here
		// (debug-only qWarning, not a gtest assertion) rather than beside
		// scene_registry_tests.cpp's equivalent GuiSceneCountMatchesRegistry
		// check for the scene count.
		scene_technique_notes::warnIfOutOfSync(selfContainedIds);
#endif
	}
	sceneGroupLayout->addWidget(m_sceneAvailabilityTabs);

	m_sceneCategoryTabs = new QTabBar(basicTab);
	m_sceneCategoryTabs->setObjectName("sceneCategoryTabs");
	m_sceneCategoryTabs->setDrawBase(false);
	m_sceneCategoryTabs->setExpanding(false);
	// The compiled-in categories fit at this window's normal width, but the
	// tab bar is inside a resizable group box - scroll buttons beat silently
	// clipping the last category off the right edge when it doesn't.
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

	// Narrows the combo/grid below by substring, on top of (not instead of)
	// the availability/category tabs above - see m_sceneSearchBox's own
	// comment in mainwindow.h for why. setClearButtonEnabled gives it Qt's
	// own built-in inline "x" rather than a hand-drawn one. The grid/list
	// toggle sits on the same row, and "Generate Thumbnails" (grid-only, so
	// it lives in the grid page rather than here) fills in cached preview
	// images for it - see populateSceneGrid()'s own comment.
	QHBoxLayout *searchRow = new QHBoxLayout();
	m_sceneSearchBox = new QLineEdit(basicTab);
	m_sceneSearchBox->setPlaceholderText(tr("Search scenes by name or id..."));
	m_sceneSearchBox->setClearButtonEnabled(true);
	searchRow->addWidget(m_sceneSearchBox, 1);

	m_sceneViewToggle = new QToolButton(basicTab);
	m_sceneViewToggle->setCheckable(true);
	m_sceneViewToggle->setText(tr("Grid"));
	m_sceneViewToggle->setToolTip(tr("Switch between the dropdown list and a thumbnail gallery grid"));
	searchRow->addWidget(m_sceneViewToggle);
	sceneGroupLayout->addLayout(searchRow);

	m_sceneViewStack = new QStackedWidget(basicTab);

	QWidget *comboPage = new QWidget(m_sceneViewStack);
	QHBoxLayout *sceneRow = new QHBoxLayout(comboPage);
	sceneRow->setContentsMargins(0, 0, 0, 0);
	m_sceneCombo = new QComboBox(basicTab);
	styleComboBox(m_sceneCombo);
	sceneRow->addWidget(new QLabel(tr("Scene:")));
	sceneRow->addWidget(createInfoIcon(
		tr("Every render starts from a scene - a description of what's in the "
		"world: the geometry (shapes and meshes), materials (what surfaces "
		"are made of), lights, and a camera.\n\n"
		"This app ships with dozens of built-in scenes covering the basics "
		"(a simple Cornell box) up through complex conductor/dielectric "
		"materials, volumetric fog, and real photogrammetry-scale models - "
		"pick one to render, or browse by category using the tabs above.")));
	sceneRow->addWidget(m_sceneCombo, 1);
	m_sceneViewStack->addWidget(comboPage);

	QWidget *gridPage = new QWidget(m_sceneViewStack);
	QVBoxLayout *gridPageLayout = new QVBoxLayout(gridPage);
	gridPageLayout->setContentsMargins(0, 0, 0, 0);
	m_sceneGrid = new QListWidget(basicTab);
	m_sceneGrid->setViewMode(QListView::IconMode);
	m_sceneGrid->setResizeMode(QListView::Adjust);
	m_sceneGrid->setMovement(QListView::Static);
	m_sceneGrid->setSelectionMode(QAbstractItemView::SingleSelection);
	m_sceneGrid->setIconSize(QSize(96, 96));
	m_sceneGrid->setGridSize(QSize(120, 132));
	m_sceneGrid->setUniformItemSizes(true);
	m_sceneGrid->setWordWrap(true);
	m_sceneGrid->setMinimumHeight(260);
	gridPageLayout->addWidget(m_sceneGrid, 1);
	m_generateThumbnailsButton = new QPushButton(tr("Generate Thumbnails"), gridPage);
	m_generateThumbnailsButton->setToolTip(
		tr("Renders a small preview image for each self-contained Basics/Materials/Cameras\n"
		"scene not already cached. CPU-only, low resolution - takes a while the first time."));
	gridPageLayout->addWidget(m_generateThumbnailsButton);
	m_sceneViewStack->addWidget(gridPage);

	sceneGroupLayout->addWidget(m_sceneViewStack);

	// QStackedWidget sizes itself to fit the largest of ALL its pages by
	// default, not just the current one - m_sceneGrid's 260px minimum
	// height would otherwise force this whole area to stay that tall even
	// while the much shorter combo page is showing (the common case). Only
	// the currently-visible page keeps its natural size policy; the other
	// is set to Ignored so it drops out of the stack's own size-hint
	// calculation - the standard Qt workaround for this exact behavior.
	// comboPage starts current (index 0), so gridPage starts Ignored.
	gridPage->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	connect(m_sceneViewToggle, &QToolButton::toggled, this, [this, comboPage, gridPage](bool gridChecked) {
		m_sceneViewStack->setCurrentIndex(gridChecked ? 1 : 0);
		comboPage->setSizePolicy(gridChecked ? QSizePolicy::Ignored : QSizePolicy::Preferred,
		                          gridChecked ? QSizePolicy::Ignored : QSizePolicy::Preferred);
		gridPage->setSizePolicy(gridChecked ? QSizePolicy::Preferred : QSizePolicy::Ignored,
		                         gridChecked ? QSizePolicy::Preferred : QSizePolicy::Ignored);
		m_sceneViewStack->updateGeometry();
	});
	connect(m_generateThumbnailsButton, &QPushButton::clicked, this, &MainWindow::onGenerateThumbnailsClicked);
	connect(m_sceneGrid, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *current, QListWidgetItem *) {
		if (!current) return;
		const QString id = current->data(Qt::UserRole).toString();
		const int comboIndex = m_sceneCombo->findData(id);
		if (comboIndex < 0) return;
		if (m_sceneCombo->currentIndex() == comboIndex) return;
		const QSignalBlocker blocker(m_sceneCombo);
		m_sceneCombo->setCurrentIndex(comboIndex);
		onSceneChanged(comboIndex);
	});

	// Fill the dropdown/grid for whichever category the bar opened on.
	if (m_sceneCategoryTabs->count() > 0)
		populateSceneViews(m_sceneCategoryTabs->tabData(0).toString());

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
			populateSceneViews(m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString());
		else {
			m_sceneCombo->clear();
			if (m_sceneGrid) m_sceneGrid->clear();
		}
		onSceneChanged(m_sceneCombo->currentIndex());
	});

	connect(m_sceneCategoryTabs, &QTabBar::currentChanged, this, [this](int tab) {
		if (tab < 0) return;
		populateSceneViews(m_sceneCategoryTabs->tabData(tab).toString());
		// populateSceneViews() deliberately stays silent, so the one update for
		// the newly selected scene is issued here - otherwise switching category
		// would leave the description, SPP and camera describing the old scene.
		onSceneChanged(m_sceneCombo->currentIndex());
	});

	// Re-narrows the current category's combo/grid on every keystroke -
	// populateSceneViews() reads m_sceneSearchBox->text() itself, so this
	// only needs to trigger the same repopulate the tab handlers above
	// already use, not duplicate the filtering logic here.
	connect(m_sceneSearchBox, &QLineEdit::textChanged, this, [this](const QString &) {
		if (m_sceneCategoryTabs->count() == 0) return;
		populateSceneViews(m_sceneCategoryTabs->tabData(m_sceneCategoryTabs->currentIndex()).toString());
		onSceneChanged(m_sceneCombo->currentIndex());
	});

	m_sceneInfoLabel = new QLabel(basicTab);
	m_sceneInfoLabel->setWordWrap(true);
	// Appearance lives in the global stylesheet under this name, so it follows
	// the active theme without anything here having to know a colour.
	m_sceneInfoLabel->setObjectName("sceneInfo");
	sceneGroupLayout->addWidget(m_sceneInfoLabel);

	// Rendering-technique icon: same look as every other info icon, but its
	// tooltip is rewritten per scene by refreshSceneInfoLabel() rather than
	// fixed at construction - see scene_technique_notes.h. The placeholder
	// text here is overwritten before the window is ever shown (the initial
	// onSceneChanged(0) call further down the constructor triggers it).
	{
		QWidget *techRow = new QWidget(basicTab);
		QHBoxLayout *techRowLayout = new QHBoxLayout(techRow);
		techRowLayout->setContentsMargins(0, 0, 0, 0);
		techRowLayout->setSpacing(4);
		techRowLayout->addWidget(new QLabel(tr("Rendering Technique:"), techRow));
		m_sceneTechInfoIcon = createInfoIcon(tr("Select a scene to see the rendering technique it demonstrates."));
		techRowLayout->addWidget(m_sceneTechInfoIcon);
		techRowLayout->addStretch();
		sceneGroupLayout->addWidget(techRow);
	}

	connect(m_sceneCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onSceneChanged);

	layout->addWidget(sceneGroup);

	// --- Render settings: output mode, GPU/CPU, quality, resolution ---
	// One group instead of separate "Render Mode" + "Render Settings" boxes -
	// they're all "how do I want this rendered" and splitting them just cost
	// an extra group box's worth of border/title chrome for no real benefit.
	QGroupBox *renderGroup = new QGroupBox(tr("Render Settings"), basicTab);
	styleGroupBox(renderGroup);
	QFormLayout *renderLayout = new QFormLayout(renderGroup);
	renderLayout->setVerticalSpacing(10);
	renderLayout->setHorizontalSpacing(10);
	renderLayout->setContentsMargins(15, 22, 15, 12);

	m_modeCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_modeCombo, ":/icons/image.svg", tr("Render Single Image"), {}, m_activeTheme.textBody);
	icon_tint::addItem(m_modeCombo, ":/icons/video.svg", tr("Generate Video"), {}, m_activeTheme.textBody);
	m_modeCombo->setCurrentIndex(0);
	styleComboBox(m_modeCombo);
	connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onModeChanged);
	renderLayout->addRow(tr("Output Mode:"), m_modeCombo);

	m_modeCombo->setToolTip(
		tr("Single Image renders one frame.\n"
		"Generate Video renders a camera path frame by frame and assembles an MP4."));

	m_renderModeCombo = new QComboBox(basicTab);
#ifdef RT_GUI_HAVE_GPU
	icon_tint::addItem(m_renderModeCombo, ":/icons/gpu.svg", tr("GPU (CUDA) - Fast"), true, m_activeTheme.textBody);
#endif
	icon_tint::addItem(m_renderModeCombo, ":/icons/cpu.svg", tr("CPU - High Quality"), false, m_activeTheme.textBody);
	styleComboBox(m_renderModeCombo);
	// Tooltips carry what the label cannot: the actual trade-off, not a repeat
	// of the visible text.
#ifdef RT_GUI_HAVE_GPU
	m_renderModeCombo->setToolTip(
		tr("GPU: OptiX hardware ray tracing — typically orders of magnitude faster.\n"
		"CPU: importance-sampled path tracer — supports every scene and material,\n"
		"including the handful the GPU backend does not implement."));
#else
	// This build's CLI (ray_tracer, from root CMakeLists.txt) has no
	// CUDA/OptiX support at all - see launcher/optix_stub.h - so GPU was
	// never a real option here and isn't offered as one.
	m_renderModeCombo->setToolTip(
		tr("Importance-sampled CPU path tracer — supports every scene and material.\n"
		"GPU rendering is not available in this build."));
#endif
	renderLayout->addRow(labelWithInfo(tr("Renderer:"),
		tr("Both trace the exact same rays and produce the same image - the "
		"difference is speed and hardware, not physics.\n\n"
		"GPU (OptiX) uses NVIDIA's dedicated ray-tracing cores to trace "
		"thousands of rays in parallel, typically far faster. CPU uses "
		"ordinary processor cores instead: much slower, but works on any "
		"machine and supports every material this app implements, "
		"including a couple the GPU path hasn't caught up to yet.")),
		m_renderModeCombo);

#ifdef RT_GUI_HAVE_GPU
	m_gpuBackendCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_gpuBackendCombo, ":/icons/gpu.svg", tr("Recursive (Default)"), false, m_activeTheme.textBody);
	icon_tint::addItem(m_gpuBackendCombo, ":/icons/gpu.svg", tr("Wavefront (Experimental)"), true, m_activeTheme.textBody);
	m_gpuBackendCombo->setCurrentIndex(0);
	styleComboBox(m_gpuBackendCombo);
	m_gpuBackendCombo->setToolTip(
		tr("Recursive: one thread per pixel, the default GPU path tracer — broad, battle-tested coverage.\n"
		"Wavefront: splits each bounce into separate queue-passed kernel launches — better GPU\n"
		"utilization on complex/divergent scenes, but a newer, less exercised code path.\n"
		"Only applies when Renderer is set to GPU."));
	// Starts disabled/enabled in sync with the initial Renderer selection (GPU,
	// index 0/true above) - the connect() in the constructor keeps it synced
	// afterwards whenever the user changes Renderer.
	m_gpuBackendCombo->setEnabled(m_renderModeCombo->currentData().toBool());
	renderLayout->addRow(labelWithInfo(tr("GPU Backend:"),
		tr("Two different ways of organizing the SAME ray-tracing work on the "
		"GPU.\n\n"
		"Recursive traces one ray per thread from start to finish, "
		"bouncing recursively - simple and battle-tested. Wavefront "
		"instead groups all rays currently doing the same kind of work "
		"(e.g. \"just hit glass\") into a batch and processes them "
		"together - better use of the GPU's parallel hardware on complex "
		"scenes with lots of different materials, at the cost of being a "
		"newer, less-tested code path.")),
		m_gpuBackendCombo);
#else
	// No GPU support in this build (see above) - the combo simply doesn't
	// exist, rather than existing permanently disabled. Every other file
	// that touches m_gpuBackendCombo (mainwindow.cpp's connect(), etc.) is
	// itself gated the same way - see those sites for the matching #ifdef.
	m_gpuBackendCombo = nullptr;
#endif

	// Integrator: which render algorithm to use instead of the default
	// path tracer. See IntegratorMode's own comment (mainwindow.h) for why
	// this is a combo (structural mutual exclusion) where the CLI itself
	// uses 8 independent bool flags with hand-written guards.
	m_integratorCombo = new QComboBox(basicTab);
	{
		// Same "(i)" mark + per-row tooltip as populateSceneCombo() gives
		// each of its own rows - icon_tint::addItem() (not a plain
		// combo->addItem()) so a theme switch's restyleThemedWidgets() ->
		// retintItems() sweep recolours these the same way every other
		// combo's icons already do, and setItemData(..., Qt::ToolTipRole)
		// so hovering a row in the OPEN dropdown shows what that specific
		// integrator does. Qt also shows the current item's icon natively
		// inside the closed combo box, so this single mechanism covers
		// both "browsing the list" and "at a glance, what's selected".
		const IntegratorMode modes[] = {
			IntegratorMode::Default, IntegratorMode::Sppm, IntegratorMode::Bdpt,
			IntegratorMode::Mlt, IntegratorMode::RandomWalk, IntegratorMode::Ao,
			IntegratorMode::SimplePath, IntegratorMode::SimpleVolPath, IntegratorMode::LightPath,
		};
		const QString labels[] = {
			tr("Path Tracer (default)"), tr("SPPM (Photon Mapping)"), tr("BDPT (Bidirectional)"),
			tr("MLT (Metropolis Light Transport)"), tr("RandomWalk (reference, unbiased)"),
			tr("Ambient Occlusion (debug)"), tr("SimplePath (reference)"),
			tr("SimpleVolPath (reference, volumetric)"), tr("LightPath (light tracer)"),
		};
		for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
			icon_tint::addItem(m_integratorCombo, ":/icons/info.svg", labels[i],
				static_cast<int>(modes[i]), m_activeTheme.textBody);
			m_integratorCombo->setItemData(m_integratorCombo->count() - 1,
				wrapTooltipHtml(integratorDescription(modes[i])), Qt::ToolTipRole);
		}
	}
	styleComboBox(m_integratorCombo);
	m_integratorCombo->setToolTip(
		tr("Which rendering algorithm to use. Path Tracer (the default) is the\n"
		"well-tested, general-purpose choice - the alternates below trade\n"
		"generality for a specific technique (photon mapping, bidirectional/\n"
		"Metropolis light transport, or a handful of unbiased reference and\n"
		"debug integrators). All alternates are CPU-only except SPPM, and none\n"
		"can be combined with Generate Video mode.\n\n"
		"Sampler/Spectral/Exposure/Tonemap/Stats above only affect the\n"
		"default Path Tracer - see each control's own tooltip."));
	connect(m_integratorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onIntegratorChanged);
	// Same two-column labelWithInfo() row shape every sibling control here
	// uses (Renderer/GPU Backend/Quality/Resolution), so the label column
	// stays aligned across all of them. A SEPARATE dynamic icon here (an
	// earlier version of this row) turned out to duplicate the per-item
	// icon Qt already shows natively inside the closed combo box (every
	// item now carries its own "(i)" + tooltip, added above via
	// icon_tint::addItem/setItemData(Qt::ToolTipRole)) - that already
	// covers "what does the currently-selected integrator do" without a
	// second, differently-aligned icon widget.
	renderLayout->addRow(labelWithInfo(tr("Integrator:"),
		tr("The rendering algorithm itself, not just how fast it runs. Path "
		"Tracer (the default) is the general-purpose, well-tested choice "
		"used everywhere else in this app.\n\n"
		"SPPM (Stochastic Progressive Photon Mapping) handles hard caustics/"
		"glass scenes path tracing struggles with. BDPT and MLT (built on "
		"BDPT) trace light paths from both the camera and the light source "
		"and connect them - better for some difficult lighting, area lights "
		"only. RandomWalk, Ambient Occlusion, SimplePath, SimpleVolPath, "
		"and LightPath are reference/debug integrators - simpler, often "
		"noisier or narrower in scope (e.g. Ambient Occlusion isn't a lit "
		"render at all), useful for isolating what a specific technique "
		"contributes.\n\nHover any item in the dropdown for details on that "
		"specific integrator.")),
		m_integratorCombo);

	m_integratorVideoWarningLabel = new QLabel(
		tr("⚠ Generate Video cannot be combined with an alternate integrator - "
		"switch back to Path Tracer, or to Single Image output."), basicTab);
	m_integratorVideoWarningLabel->setObjectName("statusWarning");
	m_integratorVideoWarningLabel->setWordWrap(true);
	m_integratorVideoWarningLabel->setVisible(false);
	renderLayout->addRow(QString(), m_integratorVideoWarningLabel);

	// Quality preset
	m_qualityPresetCombo = new QComboBox(basicTab);
	m_qualityPresetCombo->addItem(tr("Draft (Very Fast)"), 0);
	m_qualityPresetCombo->addItem(tr("Preview (Fast)"), 1);
	m_qualityPresetCombo->addItem(tr("Good (Balanced)"), 2);
	m_qualityPresetCombo->addItem(tr("High (Slow)"), 3);
	m_qualityPresetCombo->addItem(tr("Ultra (Very Slow)"), 4);
	m_qualityPresetCombo->addItem(tr("Maximum (Extreme)"), 5);
	m_qualityPresetCombo->addItem(tr("Custom"), 6);
	m_qualityPresetCombo->setCurrentIndex(2); // Default to Good
	connect(m_qualityPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onQualityPresetChanged);
	styleComboBox(m_qualityPresetCombo);
	// The preset names are relative ("Ultra", "Maximum") and say nothing
	// quantitative; spell out what each actually sets. Keep in sync with
	// onQualityPresetChanged()'s presetSamples/presetDepth tables.
	m_qualityPresetCombo->setToolTip(
		tr("Samples per pixel / max ray depth:\n"
		"  Draft    25 spp,  depth 10\n"
		"  Preview  50 spp,  depth 20\n"
		"  Good    100 spp,  depth 50\n"
		"  High    500 spp,  depth 50\n"
		"  Ultra  1000 spp,  depth 100\n"
		"  Maximum 5000 spp, depth 100\n"
		"Custom leaves the Advanced tab values untouched.\n"
		"Render time scales roughly linearly with samples per pixel."));
	renderLayout->addRow(labelWithInfo(tr("Quality:"),
		tr("A shortcut that sets both Samples per Pixel and Max Ray Depth "
		"together, since they're the two dials that trade render time "
		"for image quality.\n\n"
		"Each step up roughly doubles the render time in exchange for a "
		"cleaner, less noisy image - Draft is for quickly checking a "
		"scene looks right, Ultra/Maximum are for a final image you'd "
		"actually want to look at closely.")),
		m_qualityPresetCombo);

	// Resolution
	m_resolutionCombo = new QComboBox(basicTab);
	m_resolutionCombo->addItem(tr("100 x 100 (Tiny)"), QSize(100, 100));
	m_resolutionCombo->addItem(tr("200 x 200"), QSize(200, 200));
	m_resolutionCombo->addItem(tr("400 x 400"), QSize(400, 400));
	m_resolutionCombo->addItem(tr("512 x 512"), QSize(512, 512));
	m_resolutionCombo->addItem(tr("600 x 600"), QSize(600, 600));
	m_resolutionCombo->addItem(tr("800 x 800"), QSize(800, 800));
	m_resolutionCombo->addItem(tr("1024 x 1024 (1K)"), QSize(1024, 1024));
	m_resolutionCombo->addItem(tr("1080 x 1080 (Full HD)"), QSize(1080, 1080));
	m_resolutionCombo->addItem(tr("1200 x 1200"), QSize(1200, 1200));
	m_resolutionCombo->addItem(tr("1440 x 1440"), QSize(1440, 1440));
	m_resolutionCombo->addItem(tr("1920 x 1920"), QSize(1920, 1920));
	m_resolutionCombo->addItem(tr("2048 x 2048 (2K)"), QSize(2048, 2048));
	m_resolutionCombo->addItem(tr("2560 x 2560"), QSize(2560, 2560));
	m_resolutionCombo->addItem(tr("3840 x 3840 (4K)"), QSize(3840, 3840));
	m_resolutionCombo->addItem(tr("4096 x 4096"), QSize(4096, 4096));
	m_resolutionCombo->setCurrentIndex(5); // Default to 800x800
	styleComboBox(m_resolutionCombo);
	renderLayout->addRow(labelWithInfo(tr("Resolution:"),
		tr("How many pixels wide and tall the final image is.\n\n"
		"Higher resolution means more individual pixels to trace - each "
		"one independently sampled - so render time scales up roughly in "
		"proportion to the pixel count (double the width AND height and "
		"you're tracing about 4x as many pixels), independent of the "
		"Samples per Pixel or Max Ray Depth settings.")),
		m_resolutionCombo);

	layout->addWidget(renderGroup);

	// Output group
	QGroupBox *outputGroup = new QGroupBox(tr("Output"), basicTab);
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
		tr("Where the rendered image is written. A .png is always saved alongside\n"
		"the raw .ppm, and it is the .png the Preview tab displays."));
	// Trailing ellipsis (U+2026, not three periods) marks an action that needs
	// further input before it completes - a file dialog here. Buttons that act
	// immediately (Open Output Folder, Clear Log) deliberately have none.
	m_browseButton = new QPushButton(tr("&Browse…"), basicTab);
	m_browseButton->setToolTip(tr("Choose the output file name and location"));
	connect(m_browseButton, &QPushButton::clicked, [this]() {
		QString path = QFileDialog::getSaveFileName(this, tr("Save Render Output"),
			m_outputPathEdit->text(), tr("PNG Image (*.png);;PPM Image (*.ppm)"));
		if (!path.isEmpty()) {
			m_outputPathEdit->setText(QDir::toNativeSeparators(path));
		}
	});

	pathLayout->addWidget(createInfoIcon(
		tr("Where the finished image is saved.\n\n"
		"A raw .ppm file is always written, and a .png copy is generated "
		"alongside it automatically - the Preview tab always shows the "
		".png, since most image viewers (and this app's own preview) "
		"can't open .ppm directly.")));
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

	m_tabWidget->addTab(scrollArea, tr("Basic Settings"));
}

void MainWindow::createAdvancedTab() {
	QWidget *advancedTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(advancedTab);
	layout->setSpacing(14);  // Space between group boxes
	layout->setContentsMargins(12, 12, 12, 12);

	QGroupBox *advancedGroup = new QGroupBox(tr("Advanced Parameters"), advancedTab);
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
	formLayout->addRow(labelWithInfo(tr("Width:"),
		tr("The image's pixel width.\n\n"
		"Paired with Height below to set the resolution manually, "
		"overriding whatever the Quality preset on the Basic tab would "
		"otherwise use.")),
		m_widthSpinBox);

	// Height
	m_heightSpinBox = new QSpinBox(advancedTab);
	m_heightSpinBox->setRange(100, 4096);
	m_heightSpinBox->setValue(800);
	styleSpinBox(m_heightSpinBox);
	formLayout->addRow(labelWithInfo(tr("Height:"),
		tr("The image's pixel height.\n\n"
		"Paired with Width above - together they set the resolution "
		"manually, overriding the Basic tab's Quality preset.")),
		m_heightSpinBox);

	// Samples
	m_samplesSpinBox = new QSpinBox(advancedTab);
	m_samplesSpinBox->setRange(1, 10000);
	m_samplesSpinBox->setValue(100);
	styleSpinBox(m_samplesSpinBox);
	m_samplesSpinBox->setToolTip(
		tr("Rays traced per pixel. This is the main quality/time dial: noise falls\n"
		"as the square root of this value, so halving the noise costs about 4x\n"
		"the render time. Setting it here switches Quality to Custom."));
	formLayout->addRow(labelWithInfo(tr("Samples per Pixel:"),
		tr("Ray tracing estimates each pixel's color by firing many random "
		"rays and averaging the results, like polling a lot of people and "
		"averaging their guesses.\n\n"
		"More samples means a more accurate average, which shows up as "
		"less speckly \"noise\" in the image - but each extra sample "
		"costs render time. Doubling this value roughly halves the "
		"noise, but takes about twice as long to render.")),
		m_samplesSpinBox);

	// Max depth
	m_maxDepthSpinBox = new QSpinBox(advancedTab);
	m_maxDepthSpinBox->setRange(1, 100);
	m_maxDepthSpinBox->setValue(50);
	styleSpinBox(m_maxDepthSpinBox);
	m_maxDepthSpinBox->setToolTip(
		tr("How many times a ray may bounce before it is terminated. Low values\n"
		"darken glass and mirrors, which need many bounces to resolve; scenes\n"
		"of plain diffuse surfaces look the same well below the maximum."));
	formLayout->addRow(labelWithInfo(tr("Max Ray Depth:"),
		tr("A depth of 1 means a ray only sees what it hits directly, with "
		"no bounced light at all - like a scene with no reflections or "
		"indirect lighting.\n\n"
		"Each extra bounce lets light travel one more surface before "
		"giving up, which is what makes glass, mirrors, and soft "
		"indirect lighting look correct. Most scenes look \"finished\" "
		"well before the maximum - beyond that, extra depth mostly "
		"traces light too dim to matter.")),
		m_maxDepthSpinBox);

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

	QGroupBox *cameraGroup = new QGroupBox(tr("Camera Position"), advancedTab);
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
	m_cameraPresetCombo->addItem(tr("Front View (Outside)"), QVariant::fromValue(QVector3D(0.0f, 0.0f, -1.0f)));

	// Inside views: camera positioned near walls, all looking toward center
	m_cameraPresetCombo->addItem(tr("Inside Front"), QVariant::fromValue(QVector3D(0.0f, 0.0f, -0.211503f)));   // Near Z=0 opening
	m_cameraPresetCombo->addItem(tr("Inside Back"), QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.205937f)));     // Near Z=555 back wall
	m_cameraPresetCombo->addItem(tr("Right Wall (Green)"), QVariant::fromValue(QVector3D(0.205937f, 0.0f, 0.0f))); // Near X=555 green wall
	m_cameraPresetCombo->addItem(tr("Left Wall (Red)"), QVariant::fromValue(QVector3D(-0.211503f, 0.0f, 0.0f)));   // Near X=0 red wall

	// Corner views: diagonal perspectives from inside the box
	m_cameraPresetCombo->addItem(tr("Floor Corner"), QVariant::fromValue(QVector3D(-0.165121f, -0.211503f, -0.165121f)));  // Low angle, near floor
	m_cameraPresetCombo->addItem(tr("Ceiling Corner"), QVariant::fromValue(QVector3D(0.159555f, 0.205937f, 0.159555f)));   // High angle, near ceiling

	// Custom: allows manual X/Y/Z input via spinboxes below. Its itemData is
	// never read (onCameraPresetChanged skips the overwrite for Custom - see
	// its own comment), so this value is unused, but keep it a plausible
	// starting direction rather than leaving it as leftover absolute-position
	// data of a different shape than every other item now stores.
	m_cameraPresetCombo->addItem(tr("Custom"), QVariant::fromValue(QVector3D(0.0f, 0.0f, -1.0f)));

	styleComboBox(m_cameraPresetCombo);
	cameraLayout->addRow(labelWithInfo(tr("Preset:"),
		tr("A handful of hand-picked camera positions for this scene, framed "
		"to show off something specific (e.g. looking in through the "
		"front, or from inside a Cornell-box-style enclosure).\n\n"
		"Choosing \"Custom\" unlocks the X/Y/Z fields below so you can "
		"fly the camera anywhere you like instead.")),
		m_cameraPresetCombo);

	// Camera position spinboxes (X, Y, Z coordinates)
	// These are disabled by default; only enabled when "Custom" preset is selected
	// Range: -2000 to 2000 allows positioning far outside the box if needed

	m_cameraPosX = new QDoubleSpinBox(advancedTab);
	m_cameraPosX->setRange(-2000, 2000);
	m_cameraPosX->setValue(278);  // Default X: centered horizontally
	m_cameraPosX->setSingleStep(10);
	m_cameraPosX->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosX);
	cameraLayout->addRow(labelWithInfo(tr("Camera X:"),
		tr("The camera's position along the world's X axis (left/right).\n\n"
		"Only editable when the preset above is set to Custom - the "
		"camera always looks toward the scene's own fixed look-at point, "
		"so moving X/Y/Z changes the viewing angle and distance, not "
		"just a straight left-right pan.")),
		m_cameraPosX);

	m_cameraPosY = new QDoubleSpinBox(advancedTab);
	m_cameraPosY->setRange(-2000, 2000);
	m_cameraPosY->setValue(278);  // Default Y: centered vertically
	m_cameraPosY->setSingleStep(10);
	m_cameraPosY->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosY);
	cameraLayout->addRow(labelWithInfo(tr("Camera Y:"),
		tr("The camera's position along the world's Y axis (up/down).\n\n"
		"Same Custom-preset-only editing rule as Camera X - the camera "
		"keeps looking at the scene's fixed look-at point as you move "
		"it.")),
		m_cameraPosY);

	m_cameraPosZ = new QDoubleSpinBox(advancedTab);
	m_cameraPosZ->setRange(-2000, 2000);
	m_cameraPosZ->setValue(-800);  // Default Z: far back view to match default preset
	m_cameraPosZ->setSingleStep(10);
	m_cameraPosZ->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosZ);
	cameraLayout->addRow(labelWithInfo(tr("Camera Z:"),
		tr("The camera's position along the world's Z axis (forward/back, "
		"into or out of the scene).\n\n"
		"Same Custom-preset-only editing rule as Camera X/Y.")),
		m_cameraPosZ);

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
	cameraLayout->addRow(labelWithInfo(tr("Distance from Center:"),
		tr("Moves the camera directly toward or away from the scene's "
		"look-at point along whatever direction it's currently facing, "
		"without changing which way it's pointed.\n\n"
		"The quickest way to zoom in or pull back once you've already "
		"found an angle you like via the X/Y/Z fields or a preset.")),
		m_cameraDistance);

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

	m_tabWidget->addTab(scrollArea, tr("Advanced Settings"));
}

// ============================================================================
// Render Options Tab
// ============================================================================
// Exposes CLI flags RenderController::start() (mainwindow.cpp) already knows
// how to emit but that no earlier tab surfaced: --sampler, --spectral,
// --exposure, --tonemap, --stats, --denoise, --optix-validate. Deliberately
// a separate tab from "Advanced Settings" above (resolution/samples/depth/
// camera) rather than folded into it, since that name already means
// something else and this tab is exclusively about render BEHAVIOR flags,
// not image/camera parameters.
void MainWindow::createRenderOptionsTab() {
	QWidget *optionsTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(optionsTab);
	layout->setSpacing(14);
	layout->setContentsMargins(12, 12, 12, 12);

	// ------------------------------------------------------------------
	// Integrator Options group - sub-flags for whichever alternate
	// integrator the Basic Settings tab's Integrator combo selects. Comes
	// first (rather than after Sampling & Spectral/Output) since it's the
	// direct continuation of that combo's choice, not a separate axis.
	// ------------------------------------------------------------------
	m_integratorOptionsGroup = new QGroupBox(tr("Integrator Options"), optionsTab);
	styleGroupBox(m_integratorOptionsGroup);
	QVBoxLayout *integratorGroupLayout = new QVBoxLayout(m_integratorOptionsGroup);
	integratorGroupLayout->setContentsMargins(15, 22, 15, 12);

	m_integratorOptionsStack = new QStackedWidget(m_integratorOptionsGroup);

	// Prepends a word-wrapped description of `mode` as the first (full-
	// width) row of `pageLayout` - reuses the exact same text the per-item
	// combo tooltips already show (integratorDescription(),
	// mainwindow_style.cpp), so SPPM/BDPT/MLT/AO/SimplePath's pages get the
	// same persistent, always-visible explanation the shared placeholder
	// page (Default/RandomWalk/SimpleVolPath/LightPath, below) already has
	// via m_integratorNoOptionsLabel - not just a tooltip you'd only see by
	// opening the dropdown and hovering.
	auto addIntegratorDescription = [this](QFormLayout *pageLayout, IntegratorMode mode) {
		QLabel *desc = new QLabel(integratorDescription(mode));
		desc->setWordWrap(true);
		pageLayout->addRow(desc);
	};

	// Page 0: shared placeholder for Default/RandomWalk/SimpleVolPath/
	// LightPath - none of these four have any sub-flags, so they share
	// one page instead of each getting a near-duplicate empty one. Text
	// is swapped per-mode in onIntegratorChanged() (mainwindow_slots.cpp).
	m_integratorNoOptionsLabel = new QLabel(tr("The default Path Tracer has no integrator-specific options here - see the Render Options above."), m_integratorOptionsStack);
	m_integratorNoOptionsLabel->setWordWrap(true);
	m_integratorOptionsStack->addWidget(m_integratorNoOptionsLabel);

	// Page 1: SPPM
	QWidget *sppmPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *sppmLayout = new QFormLayout(sppmPage);
	sppmLayout->setVerticalSpacing(10);
	sppmLayout->setHorizontalSpacing(10);
	addIntegratorDescription(sppmLayout, IntegratorMode::Sppm);
	m_sppmIterationsSpin = new QSpinBox(sppmPage);
	m_sppmIterationsSpin->setRange(1, 1000000);
	m_sppmIterationsSpin->setValue(100);
	styleSpinBox(m_sppmIterationsSpin);
	sppmLayout->addRow(labelWithInfo(tr("Iterations:"),
		tr("How many camera-pass + photon-pass rounds SPPM runs. More "
		"iterations converge to a cleaner result, at a roughly linear "
		"cost in render time.")),
		m_sppmIterationsSpin);
	m_sppmPhotonsSpin = new QSpinBox(sppmPage);
	m_sppmPhotonsSpin->setRange(1, 100000000);
	m_sppmPhotonsSpin->setValue(5000);
	styleSpinBox(m_sppmPhotonsSpin);
	sppmLayout->addRow(labelWithInfo(tr("Photons per iteration:"),
		tr("How many photons are shot from the lights each iteration. "
		"More photons reduce noise in indirect/caustic lighting at the "
		"cost of a slower photon pass.")),
		m_sppmPhotonsSpin);
	m_integratorOptionsStack->addWidget(sppmPage);

	// Page 2: BDPT
	QWidget *bdptPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *bdptLayout = new QFormLayout(bdptPage);
	bdptLayout->setVerticalSpacing(10);
	bdptLayout->setHorizontalSpacing(10);
	addIntegratorDescription(bdptLayout, IntegratorMode::Bdpt);
	m_bdptMaxDepthSpin = new QSpinBox(bdptPage);
	m_bdptMaxDepthSpin->setRange(1, 100);
	m_bdptMaxDepthSpin->setValue(5);
	styleSpinBox(m_bdptMaxDepthSpin);
	bdptLayout->addRow(labelWithInfo(tr("Max path depth:"),
		tr("Maximum bounces for each of the two subpaths (camera side and "
		"light side) that BDPT connects together.")),
		m_bdptMaxDepthSpin);
	m_integratorOptionsStack->addWidget(bdptPage);

	// Page 3: MLT
	QWidget *mltPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *mltLayout = new QFormLayout(mltPage);
	mltLayout->setVerticalSpacing(10);
	mltLayout->setHorizontalSpacing(10);
	addIntegratorDescription(mltLayout, IntegratorMode::Mlt);
	m_mltBootstrapSpin = new QSpinBox(mltPage);
	m_mltBootstrapSpin->setRange(1, 10000000);
	m_mltBootstrapSpin->setValue(100000);
	styleSpinBox(m_mltBootstrapSpin);
	mltLayout->addRow(labelWithInfo(tr("Bootstrap samples:"),
		tr("How many candidate light paths MLT samples up front, per "
		"depth, to seed its Markov chains - more gives a better-informed "
		"starting distribution.")),
		m_mltBootstrapSpin);
	m_mltMutationsSpin = new QSpinBox(mltPage);
	m_mltMutationsSpin->setRange(1, 1000000000);
	m_mltMutationsSpin->setValue(4000000);
	styleSpinBox(m_mltMutationsSpin);
	mltLayout->addRow(labelWithInfo(tr("Mutations:"),
		tr("Total Metropolis mutations across all chains combined - the "
		"main knob for render time/quality, analogous to samples per "
		"pixel in the default path tracer.")),
		m_mltMutationsSpin);
	m_mltMaxDepthSpin = new QSpinBox(mltPage);
	m_mltMaxDepthSpin->setRange(1, 100);
	m_mltMaxDepthSpin->setValue(5);
	styleSpinBox(m_mltMaxDepthSpin);
	mltLayout->addRow(labelWithInfo(tr("Max path depth:"),
		tr("Same meaning as BDPT's max path depth (MLT is built directly "
		"on BDPT's subpath machinery).")),
		m_mltMaxDepthSpin);
	m_integratorOptionsStack->addWidget(mltPage);

	// Page 4: Ambient Occlusion
	QWidget *aoPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *aoLayout = new QFormLayout(aoPage);
	aoLayout->setVerticalSpacing(10);
	aoLayout->setHorizontalSpacing(10);
	addIntegratorDescription(aoLayout, IntegratorMode::Ao);
	m_aoMaxDistSpin = new QDoubleSpinBox(aoPage);
	m_aoMaxDistSpin->setRange(0.01, 1.0e12);
	m_aoMaxDistSpin->setDecimals(2);
	m_aoMaxDistSpin->setValue(1.0e10);
	styleSpinBox(m_aoMaxDistSpin);
	aoLayout->addRow(labelWithInfo(tr("Max occlusion distance:"),
		tr("How far an occlusion test ray can reach before counting as "
		"unoccluded. The default (10 billion) is effectively unbounded - "
		"lower it to only count nearby geometry as occluding.")),
		m_aoMaxDistSpin);
	m_aoUniformCheck = new QCheckBox(tr("Uniform-hemisphere sampling (instead of cosine)"), aoPage);
	styleCheckBox(m_aoUniformCheck);
	aoLayout->addRow(checkboxWithInfo(m_aoUniformCheck,
		tr("The default samples occlusion rays weighted toward the "
		"surface normal (cosine-hemisphere), matching how a Lambertian "
		"surface would actually be lit. Uniform-hemisphere spreads "
		"samples evenly instead - a different, unweighted estimator.")));
	m_aoIllumScaleSpin = new QDoubleSpinBox(aoPage);
	m_aoIllumScaleSpin->setRange(0.0, 1000.0);
	m_aoIllumScaleSpin->setValue(1.0);
	styleSpinBox(m_aoIllumScaleSpin);
	aoLayout->addRow(labelWithInfo(tr("Illumination scale:"),
		tr("Flat multiplier on the occlusion color below.")),
		m_aoIllumScaleSpin);
	QWidget *aoIllumRgbRow = new QWidget(aoPage);
	QHBoxLayout *aoIllumRgbLayout = new QHBoxLayout(aoIllumRgbRow);
	aoIllumRgbLayout->setContentsMargins(0, 0, 0, 0);
	aoIllumRgbLayout->setSpacing(6);
	m_aoIllumRSpin = new QDoubleSpinBox(aoIllumRgbRow);
	m_aoIllumGSpin = new QDoubleSpinBox(aoIllumRgbRow);
	m_aoIllumBSpin = new QDoubleSpinBox(aoIllumRgbRow);
	for (QDoubleSpinBox *spin : {m_aoIllumRSpin, m_aoIllumGSpin, m_aoIllumBSpin}) {
		spin->setRange(0.0, 1.0);
		spin->setSingleStep(0.05);
		spin->setValue(1.0);
		styleSpinBox(spin);
		aoIllumRgbLayout->addWidget(spin);
	}
	aoLayout->addRow(labelWithInfo(tr("Occlusion color (R, G, B):"),
		tr("The color ambient occlusion is visualized in - not a lit "
		"render, so this is a visualization choice, not a light color. "
		"Default is white (1, 1, 1).")),
		aoIllumRgbRow);
	m_integratorOptionsStack->addWidget(aoPage);

	// Page 5: SimplePath
	QWidget *simplepathPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *simplepathLayout = new QFormLayout(simplepathPage);
	simplepathLayout->setVerticalSpacing(10);
	simplepathLayout->setHorizontalSpacing(10);
	addIntegratorDescription(simplepathLayout, IntegratorMode::SimplePath);
	m_simplepathNoLightsCheck = new QCheckBox(tr("Disable next-event estimation (direct light sampling)"), simplepathPage);
	styleCheckBox(m_simplepathNoLightsCheck);
	simplepathLayout->addRow(checkboxWithInfo(m_simplepathNoLightsCheck,
		tr("On by default. Direct light sampling explicitly aims shadow "
		"rays at lights each bounce, sharply reducing noise on scenes "
		"with small/bright lights. Disabling it falls back to finding "
		"lights only by chance, the way a purely unbiased path tracer "
		"would.")));
	m_simplepathNoBsdfCheck = new QCheckBox(tr("Disable BSDF importance sampling"), simplepathPage);
	styleCheckBox(m_simplepathNoBsdfCheck);
	simplepathLayout->addRow(checkboxWithInfo(m_simplepathNoBsdfCheck,
		tr("On by default. Samples each bounce's new direction weighted "
		"toward where the surface's material actually reflects light. "
		"Disabling it falls back to uniform hemisphere sampling.")));
	m_integratorOptionsStack->addWidget(simplepathPage);

	integratorGroupLayout->addWidget(m_integratorOptionsStack);
	layout->addWidget(m_integratorOptionsGroup);

	// ------------------------------------------------------------------
	// Sampling & Spectral group - CPU default path tracer only
	// ------------------------------------------------------------------
	// "&&" (not "&") - a single "&" is a Qt mnemonic-accelerator marker,
	// which would eat the "&" and underline the next letter instead of
	// showing a literal ampersand.
	QGroupBox *samplingGroup = new QGroupBox(tr("Sampling && Spectral"), optionsTab);
	styleGroupBox(samplingGroup);
	QFormLayout *samplingLayout = new QFormLayout(samplingGroup);
	samplingLayout->setVerticalSpacing(10);
	samplingLayout->setHorizontalSpacing(10);
	samplingLayout->setContentsMargins(15, 22, 15, 12);

	m_samplerCombo = new QComboBox(optionsTab);
	m_samplerCombo->addItem(tr("Sobol (default)"), QString());
	m_samplerCombo->addItem(tr("Z-Sobol"), QStringLiteral("zsobol"));
	m_samplerCombo->addItem(tr("Padded Sobol"), QStringLiteral("paddedsobol"));
	m_samplerCombo->addItem(tr("Stratified"), QStringLiteral("stratified"));
	m_samplerCombo->addItem(tr("PMJ02BN"), QStringLiteral("pmj02bn"));
	m_samplerCombo->addItem(tr("Halton"), QStringLiteral("halton"));
	m_samplerCombo->addItem(tr("Independent (no stratification)"), QStringLiteral("independent"));
	m_samplerCombo->setToolTip(
		tr("Which sampler drives random decisions (all but Independent are\n"
		"low-discrepancy). CPU default path tracer only - no effect on GPU\n"
		"or under BDPT/MLT/SPPM/the debug integrators."));
	styleComboBox(m_samplerCombo);
	samplingLayout->addRow(labelWithInfo(tr("Sampler:"),
		tr("Ray tracing needs a lot of random numbers - which direction to "
		"bounce a ray, which point on a light to sample, and so on - and "
		"HOW those \"random\" numbers are generated changes how quickly "
		"the image converges to a clean result.\n\n"
		"A naive random-number generator clusters and leaves gaps; most "
		"samplers here (Sobol, Halton, etc.) are low-discrepancy sequences, "
		"deliberately spread out to cover the sampling space more evenly, "
		"which converges to a clean image faster than true randomness "
		"would for the same sample count. Independent is the exception - "
		"plain uncorrelated random numbers, included for fidelity to a "
		"loaded .pbrt scene's own Sampler directive rather than as a "
		"recommended choice.\n\n"
		"Grayed out? This only affects the CPU renderer's default path "
		"tracer - switch Renderer to CPU on the Basic Settings tab to "
		"use it.")),
		m_samplerCombo);

	m_spectralCheck = new QCheckBox(tr("Spectral rendering (--spectral)"), optionsTab);
	m_spectralCheck->setToolTip(
		tr("Real hero-wavelength spectral rendering instead of flat RGB.\n"
		"CPU default path tracer only. Only lambertian, metal, dielectric,\n"
		"rough_dielectric, conductor, and diffuse_light materials are\n"
		"supported - a scene using anything else fails to render rather\n"
		"than silently rendering wrong colors. Noticeably slower per-sample."));
	styleCheckBox(m_spectralCheck);
	samplingLayout->addRow(checkboxWithInfo(m_spectralCheck,
		tr("Ordinary rendering tracks light as three numbers - red, "
		"green, blue - the same way a screen displays color.\n\n"
		"Real light is a continuous spectrum of wavelengths, and a "
		"few physical effects (like a prism splitting white light "
		"into a rainbow) only happen because different wavelengths "
		"refract by different amounts - RGB alone can't represent "
		"that. Spectral rendering tracks a handful of actual "
		"wavelengths per ray instead of just RGB, at the cost of "
		"being noisier and slower per sample.\n\n"
		"Grayed out? This only exists on the CPU renderer's default "
		"path tracer - switch Renderer to CPU on the Basic Settings "
		"tab to use it.")));

	m_exposureSpin = new QDoubleSpinBox(optionsTab);
	m_exposureSpin->setRange(0.01, 100.0);
	m_exposureSpin->setValue(1.0);
	m_exposureSpin->setSingleStep(0.1);
	m_exposureSpin->setToolTip(
		tr("Flat multiplier on linear color before tone-mapping (1.0 = no-op).\n"
		"Both CPU and GPU default path tracer only."));
	styleSpinBox(m_exposureSpin);
	samplingLayout->addRow(labelWithInfo(tr("Exposure:"),
		tr("A flat brightness multiplier applied to the whole image, the "
		"same knob a camera's exposure setting is.\n\n"
		"1.0 leaves the image unchanged; below 1.0 darkens it, above 1.0 "
		"brightens it - useful for a scene that's rendering correctly "
		"but is just too dark or too bright to see clearly, without "
		"changing any actual light in the scene.")),
		m_exposureSpin);

	layout->addWidget(samplingGroup);

	// ------------------------------------------------------------------
	// Output group
	// ------------------------------------------------------------------
	QGroupBox *outputGroup = new QGroupBox(tr("Output"), optionsTab);
	styleGroupBox(outputGroup);
	QFormLayout *outputLayout = new QFormLayout(outputGroup);
	outputLayout->setVerticalSpacing(10);
	outputLayout->setHorizontalSpacing(10);
	outputLayout->setContentsMargins(15, 22, 15, 12);

	m_tonemapCombo = new QComboBox(optionsTab);
	m_tonemapCombo->addItem(tr("ACES (default)"), QString());
	m_tonemapCombo->addItem(tr("Reinhard"), QStringLiteral("reinhard"));
	m_tonemapCombo->addItem(tr("None"), QStringLiteral("none"));
	m_tonemapCombo->setToolTip(
		tr("Which tone-mapping operator to apply before the sRGB curve.\n"
		"Applies to both CPU and GPU (recursive and wavefront) - no\n"
		"effect under BDPT/MLT/SPPM/the debug integrators."));
	styleComboBox(m_tonemapCombo);
	outputLayout->addRow(labelWithInfo(tr("Tone mapping:"),
		tr("A raytraced scene's true brightness values are unbounded - a "
		"light bulb might be a hundred times brighter than a wall - but "
		"a screen can only display a fixed range. Tone mapping is the "
		"curve that compresses that huge range down into something "
		"displayable.\n\n"
		"ACES rolls off bright highlights gently, the way film does; "
		"Reinhard is a simpler, older compression; None just clips "
		"anything too bright to flat white, which can look harsh.")),
		m_tonemapCombo);

	m_statsCheck = new QCheckBox(tr("Print render stats"), optionsTab);
	m_statsCheck->setToolTip(
		tr("Print a small end-of-render stats block (rays cast, bounces,\n"
		"shadow rays, samples/sec) to the Log tab. Observation-only -\n"
		"never changes the rendered image."));
	styleCheckBox(m_statsCheck);
	outputLayout->addRow(checkboxWithInfo(m_statsCheck,
		tr("Prints a short summary after the render finishes - how many "
		"rays were cast, how many bounces happened, how many shadow "
		"rays were traced, and samples per second.\n\n"
		"Purely informational: it never changes the rendered image, "
		"just tells you what the renderer actually did.")));

	m_denoiseCheck = new QCheckBox(tr("OptiX AI denoiser (GPU recursive only)"), optionsTab);
	m_denoiseCheck->setToolTip(
		tr("Run the OptiX AI denoiser on the finished render, guided by\n"
		"albedo + normal buffers. GPU recursive backend only - silently\n"
		"has no effect under the wavefront backend."));
	styleCheckBox(m_denoiseCheck);
	outputLayout->addRow(checkboxWithInfo(m_denoiseCheck,
		tr("Ray tracing is noisy by nature - low sample counts leave a "
		"grainy, speckled image, which is why more samples usually "
		"means a cleaner picture.\n\n"
		"A denoiser is a machine-learning model trained to recognize "
		"that speckle pattern and smooth it away after the fact, "
		"without needing to trace additional rays - a way to get a "
		"clean-looking image faster, at some cost in fine detail.\n\n"
		"Grayed out? This needs the GPU recursive backend - switch "
		"Renderer to GPU (and GPU Backend to Recursive) on the Basic "
		"Settings tab to use it.")));

	m_optixValidateCheck = new QCheckBox(tr("OptiX validation mode (slower, debugging only)"), optionsTab);
	m_optixValidateCheck->setToolTip(
		tr("Enable OptiX validation mode - extra device-side checks with a\n"
		"real per-launch cost. GPU only, for debugging, not routine use."));
	styleCheckBox(m_optixValidateCheck);
	outputLayout->addRow(checkboxWithInfo(m_optixValidateCheck,
		tr("Turns on extra correctness checks inside the GPU ray-tracing "
		"pipeline itself, catching certain classes of bugs that would "
		"otherwise silently produce a wrong image or crash "
		"unpredictably.\n\n"
		"It's a debugging aid for people working on the renderer's "
		"own GPU code, not something a normal render benefits from - "
		"it has a real performance cost and doesn't change what a "
		"correct render looks like.\n\n"
		"Grayed out? This is GPU-only - switch Renderer to GPU on "
		"the Basic Settings tab to use it.")));

	layout->addWidget(outputGroup);
	layout->addStretch();

	// Initial enabled state matches whatever m_renderModeCombo/
	// m_gpuBackendCombo/m_integratorCombo already hold at this point in
	// construction - updateRenderOptionsEnabled() is the single source of
	// truth for this, also called live from each of those combos' own
	// change handlers (mainwindow.cpp/mainwindow_slots.cpp).
	updateRenderOptionsEnabled();

	QScrollArea *scrollArea = new QScrollArea();
	scrollArea->setWidget(optionsTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea->setObjectName("tabScroll");
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, tr("Render Options"));
}

// Single source of truth for m_samplerCombo/m_spectralCheck/m_exposureSpin/
// m_tonemapCombo/m_statsCheck/m_denoiseCheck/m_optixValidateCheck/
// m_gpuBackendCombo's enabled state, replacing what used to be a
// hand-duplicated 2-input (GPU/CPU x wavefront/recursive) condition
// copy-pasted at construction and in two separate connect() lambdas -
// adding Integrator as a third input would have made that a 3-site
// hand-copy of an increasingly complex condition, so it's factored here
// instead and called from all four controls' own change handlers
// (construction, m_renderModeCombo's lambda, m_gpuBackendCombo's lambda,
// onIntegratorChanged() - mainwindow.cpp/mainwindow_slots.cpp).
void MainWindow::updateRenderOptionsEnabled() {
	const bool gpuSelected = m_renderModeCombo->currentData().toBool();
	const bool wavefrontSelected = gpuSelected && m_gpuBackendCombo && m_gpuBackendCombo->currentData().toBool();
	const auto integrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	const bool isDefault = (integrator == IntegratorMode::Default);

	if (m_gpuBackendCombo) m_gpuBackendCombo->setEnabled(isDefault && gpuSelected);
	m_samplerCombo->setEnabled(isDefault && !gpuSelected);
	m_spectralCheck->setEnabled(isDefault && !gpuSelected);
	m_exposureSpin->setEnabled(isDefault);
	m_tonemapCombo->setEnabled(isDefault);
	m_statsCheck->setEnabled(isDefault);
	m_denoiseCheck->setEnabled(isDefault && gpuSelected && !wavefrontSelected);
	m_optixValidateCheck->setEnabled(isDefault && gpuSelected);
}

void MainWindow::createPreviewTab() {
	QWidget *previewWidget = new QWidget();
	QHBoxLayout *outerLayout = new QHBoxLayout(previewWidget);
	outerLayout->setContentsMargins(12, 12, 12, 12);
	outerLayout->setSpacing(0);

	// Sub-tabs on the left in a large, dominant pane; info/buttons in a
	// narrow sidebar on the right, so the render gets most of the tab's
	// space instead of splitting height with a full-width info/button strip
	// underneath it. A QSplitter (not a fixed QHBoxLayout split) so the
	// user can still drag the sidebar narrower/wider if they want even more
	// render space.
	QSplitter *splitter = new QSplitter(Qt::Horizontal, previewWidget);
	splitter->setChildrenCollapsible(false);
	outerLayout->addWidget(splitter);

	// Each completed render gets its own closable sub-tab (see
	// addImagePreviewTab()/addVideoPreviewTab()) rather than a single
	// shared pane the next render overwrites - switching between sub-tabs
	// keeps every past render's image/video around. Not movable: tab order
	// (render order) is itself informative, and reordering would also
	// complicate m_previewTitleCounts' de-duplication.
	m_previewSubTabs = new SplitPreviewTabs();
	m_previewSubTabs->setMinimumSize(200, 200);
	// Qt's own auto-managed close button is deliberately left off - its
	// placement follows a style hint (SH_TabBar_CloseButtonPosition) that
	// this app's active stylesheet does not reliably let a per-widget style
	// override, so it always landed on the wrong side regardless.
	// HorizontalTabBar instead hand-paints and hand-hit-tests its own close
	// glyph directly on the left of each tab's label (see paintEvent()/
	// mousePressEvent() in mainwindow.h), which sidesteps that negotiation
	// entirely. Not movable either (QTabBar's own default): tab order
	// (render order) is itself informative, and reordering would also
	// complicate m_previewTitleCounts' de-duplication.
	//
	// Left side rather than across the top - a render can accumulate many
	// of these over a session, and a vertical list scales far better than
	// a widening horizontal strip. setUsesScrollButtons() is what keeps the
	// list reachable once it overflows the pane's height (Qt's tab bar has
	// no literal scrollbar, but this is its own equivalent: small up/down
	// arrow buttons appear once the tabs no longer fit). objectName'd so
	// its QSS rule (mainwindow_style.cpp) can give it left-rounded corners
	// and a right-edge accent border instead of the main tab strip's
	// top-rounded/bottom-underline styling, which is built for a
	// horizontal bar and would land on the wrong edges here. The West
	// shape itself is set in SplitPreviewTabs's own constructor
	// (mainwindow.h), since that's a QTabBar property independent of
	// QTabWidget - SplitPreviewTabs doesn't use one.
	m_previewSubTabs->tabBar()->setObjectName("previewSubTabsBar");
	m_previewSubTabs->tabBar()->setUsesScrollButtons(true);
	m_previewSubTabs->setElideMode(Qt::ElideRight);
	connect(m_previewSubTabs, &SplitPreviewTabs::currentChanged, this, [this](int) {
		updatePreviewSidebarForActiveTab();
	});
	connect(m_previewSubTabs->tabBar(), &HorizontalTabBar::closeRequested,
	        this, &MainWindow::closePreviewSubTab);
	splitter->addWidget(m_previewSubTabs);

	// Recent Renders: past renders' output files are never deleted when
	// their tab is closed (see closePreviewSubTab()), just no longer
	// reachable from the UI once the session that created them ends - this
	// list, inserted into the empty-state prompt (visible in exactly the
	// state where recovering a past render matters), reopens one with a
	// double-click via the same addImagePreviewTab()/addVideoPreviewTab()
	// calls a fresh render itself uses. Built here (see
	// refreshRecentRendersList(), recent_renders.cpp) and rebuilt again
	// after every save and every tab close, so renders from earlier this
	// session show up without an app restart.
	refreshRecentRendersList();

	QWidget *sidebar = new QWidget();
	m_previewSidebar = sidebar;
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

	// The active render's own technique note (what the scene demonstrates
	// and why it looks the way it does) - see updatePreviewSidebarForActiveTab(),
	// which populates this from the tab's "sceneId" property and hides it
	// when scene_technique_notes::hasNote() says there's nothing authored.
	m_previewTechniqueLabel = new QLabel(sidebar);
	m_previewTechniqueLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewTechniqueLabel->setWordWrap(true);
	m_previewTechniqueLabel->setObjectName("previewTechniqueNote");
	m_previewTechniqueLabel->setVisible(false);
	sideLayout->addWidget(m_previewTechniqueLabel);

	// Geometry only - colour and hover/focus states come from the global
	// theme so every secondary button behaves identically. Full sidebar
	// width and stacked vertically now that they're beside the image, not
	// centered in a horizontal strip underneath it. Both act on whichever
	// sub-tab is currently active, not just the most recent render - see
	// currentPreviewProperty().
	QString previewBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; padding: 0px 20px; font-size: 11pt; }";

	QPushButton *openFolderButton = new QPushButton(tr("Open Output &Folder"));
	icon_tint::apply(openFolderButton, ":/icons/folder.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openFolderButton->setStyleSheet(previewBtnStyle);
	openFolderButton->setToolTip(tr("Show the folder containing the active tab's render in Explorer"));
	connect(openFolderButton, &QPushButton::clicked, this, [this]() {
		const QString path = currentPreviewProperty("outputPath");
		if (path.isEmpty()) return;
		QFileInfo fileInfo(path);
		QDesktopServices::openUrl(QUrl::fromLocalFile(fileInfo.absolutePath()));
	});
	sideLayout->addWidget(openFolderButton);

	QPushButton *openViewerButton = new QPushButton(tr("Open in Default &Viewer"));
	icon_tint::apply(openViewerButton, ":/icons/image.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openViewerButton->setStyleSheet(previewBtnStyle);
	openViewerButton->setToolTip(tr("Open the active tab's render in the system viewer"));
	connect(openViewerButton, &QPushButton::clicked, this, [this]() {
		const QString path = currentPreviewProperty("previewPath");
		if (path.isEmpty()) return;
		QDesktopServices::openUrl(QUrl::fromLocalFile(path));
	});
	sideLayout->addWidget(openViewerButton);

	sideLayout->addStretch(1);
	splitter->addWidget(sidebar);

	// Bias initial space toward the render - the sidebar only needs enough
	// width for its buttons/info text, everything else goes to the render.
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 0);
	splitter->setSizes({700, 220});

	// Starts hidden - no tabs exist yet at construction time, so there is
	// nothing for the sidebar to describe. updatePreviewSidebarForActiveTab()
	// takes over from here once real tabs come and go.
	sidebar->setVisible(false);

	m_previewTabIndex = m_tabWidget->addTab(previewWidget, tr("Preview"));
}

QString MainWindow::uniquePreviewTabTitle(const QString &baseTitle) {
	int &count = m_previewTitleCounts[baseTitle];
	++count;
	return count == 1 ? baseTitle : QString("%1 (%2)").arg(baseTitle).arg(count);
}

QString MainWindow::currentPreviewProperty(const char *name) const {
	if (!m_previewSubTabs) return QString();
	QWidget *page = m_previewSubTabs->currentWidget();
	if (!page) return QString();
	return page->property(name).toString();
}

void MainWindow::updatePreviewSidebarForActiveTab() {
	if (m_previewInfoLabel) m_previewInfoLabel->setText(currentPreviewProperty("infoText"));
	if (m_previewTechniqueLabel) {
		const QString sceneId = currentPreviewProperty("sceneId");
		const bool hasNote = !sceneId.isEmpty() && scene_technique_notes::hasNote(sceneId);
		m_previewTechniqueLabel->setVisible(hasNote);
		if (hasNote) {
			m_previewTechniqueLabel->setText(
				tr("<b>Why it looks this way</b><br>%1").arg(scene_technique_notes::forScene(sceneId)));
		}
	}
	// Nothing in the sidebar (render info, Open Folder/Viewer) means anything
	// without an active render tab to point at - hidden rather than left
	// showing stale info/dead buttons alongside the empty-state prompt (see
	// SplitPreviewTabs's own empty-state widget in mainwindow.h), and it also
	// lets that prompt use the pane's full width instead of being squeezed
	// down to whatever the splitter left it.
	if (m_previewSidebar && m_previewSubTabs) {
		m_previewSidebar->setVisible(m_previewSubTabs->tabBar()->count() > 0);
	}
	updateActionStates();
}

void MainWindow::closePreviewSubTab(int index) {
	if (!m_previewSubTabs) return;
	QWidget *page = m_previewSubTabs->widget(index);
	m_previewSubTabs->removeTab(index);
	// Any QMediaPlayer/QVideoWidget a video tab owns is a CHILD of `page`
	// (see addVideoPreviewTab()), so deleting it tears those down too
	// rather than leaking a player per closed tab.
	if (page) page->deleteLater();
	updatePreviewSidebarForActiveTab();
	// The file this tab pointed at is still on disk (only the tab itself
	// closed) - if closing this was the last tab, the empty state (and its
	// Recent Renders list) is about to show, so make sure it includes this
	// one rather than waiting for the next render or app restart.
	refreshRecentRendersList();
}

void MainWindow::addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
									 const QString &infoText, const QString &outputPath, const QString &previewPath,
									 const QString &sceneId) {
	if (!m_previewSubTabs) return;

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(0, 0, 0, 0);

	ScaledImageLabel *label = new ScaledImageLabel();
	label->setMinimumSize(200, 200);
	label->setPreviewPixmap(pixmap);
	// Styled globally by class name - see the ScaledImageLabel rule.
	layout->addWidget(label);

	page->setProperty("outputPath", outputPath);
	page->setProperty("previewPath", previewPath);
	page->setProperty("infoText", infoText);
	page->setProperty("sceneId", sceneId);

	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);
}

void MainWindow::addVideoPreviewTab(const QString &title, const QString &tooltip,
									 const QString &videoPath, const QString &infoText,
									 const QString &sceneId) {
	if (!m_previewSubTabs) return;

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	QVideoWidget *videoWidget = new QVideoWidget();
	videoWidget->setMinimumSize(200, 200);
	layout->addWidget(videoWidget, 1);

	// Parented to `page`, not `this` - a self-contained player per tab that
	// only ever loads ONE file, once. Besides being the natural way to give
	// each tab independent playback state, this sidesteps the QMediaPlayer/
	// FFmpeg-backend failure mode a single shared, reused player used to
	// hit (reloading a DIFFERENT file at a path it had already opened
	// before could come back "Invalid data found when processing input"
	// even though the file itself was perfectly valid) - with one player
	// per unique file, that scenario can no longer arise.
	QMediaPlayer *player = new QMediaPlayer(page);
	QAudioOutput *audioOutput = new QAudioOutput(page);
	player->setAudioOutput(audioOutput);
	player->setVideoOutput(videoWidget);

	QWidget *controls = new QWidget();
	QHBoxLayout *controlsLayout = new QHBoxLayout(controls);
	controlsLayout->setContentsMargins(0, 0, 0, 0);
	controlsLayout->setSpacing(8);

	QPushButton *playPauseButton = new QPushButton(tr("Pause"));
	playPauseButton->setFixedWidth(70);
	connect(playPauseButton, &QPushButton::clicked, page, [player]() {
		if (player->playbackState() == QMediaPlayer::PlayingState) player->pause();
		else player->play();
	});
	controlsLayout->addWidget(playPauseButton);

	QSlider *positionSlider = new QSlider(Qt::Horizontal);
	controlsLayout->addWidget(positionSlider, 1);
	layout->addWidget(controls);

	connect(player, &QMediaPlayer::playbackStateChanged, page, [playPauseButton](QMediaPlayer::PlaybackState state) {
		playPauseButton->setText(state == QMediaPlayer::PlayingState ? tr("Pause") : tr("Play"));
	});
	connect(player, &QMediaPlayer::durationChanged, page, [positionSlider](qint64 duration) {
		positionSlider->setRange(0, static_cast<int>(duration));
	});
	connect(player, &QMediaPlayer::positionChanged, page, [positionSlider](qint64 position) {
		if (!positionSlider->isSliderDown())
			positionSlider->setValue(static_cast<int>(position));
	});
	connect(player, &QMediaPlayer::errorOccurred, page, [this](QMediaPlayer::Error error, const QString &errorString) {
		onLogMessage(tr("Video playback error (%1): %2").arg(static_cast<int>(error)).arg(errorString));
	});
	// A small value bubble that follows the handle while scrubbing, showing
	// the position being dragged to as M:SS - Fluent-style sliders do this
	// by default; QSlider has no built-in equivalent. A plain child widget
	// rather than a real QToolTip, which auto-hides on mouse movement -
	// exactly what a drag never stops doing. Parented to positionSlider so
	// it's destroyed with it automatically; each video preview tab gets its
	// own slider (and so its own bubble), never a shared one.
	QLabel *scrubBubble = new QLabel(positionSlider);
	scrubBubble->setObjectName("scrubBubble");
	scrubBubble->setAlignment(Qt::AlignCenter);
	scrubBubble->hide();

	const auto formatMs = [](qint64 ms) {
		const qint64 totalSeconds = ms / 1000;
		return QString("%1:%2").arg(totalSeconds / 60).arg(totalSeconds % 60, 2, 10, QChar('0'));
	};
	// Horizontal position only (handle height doesn't vary), placed just
	// above the groove. Proportional to (value-min)/(max-min) across the
	// slider's own current width - not a hand-tuned pixel offset - so it
	// tracks correctly regardless of the tab's width or DPI.
	const auto moveBubbleTo = [positionSlider, scrubBubble](int value) {
		scrubBubble->adjustSize();
		const int span = std::max(0, positionSlider->width() - scrubBubble->width());
		const int range = positionSlider->maximum() - positionSlider->minimum();
		const double t = range > 0
			? double(value - positionSlider->minimum()) / range : 0.0;
		scrubBubble->move(static_cast<int>(t * span), -scrubBubble->height() - 4);
	};

	// Pausing before setPosition() (and resuming after, if it was playing)
	// is the standard fix for scrubbing that "doesn't seem to do anything":
	// while playing, the player's own clock keeps advancing on its own
	// timeline in parallel with each setPosition() call from the drag, so
	// the seek and normal playback fight over what frame gets shown next -
	// on some backends the dragged-to frame never visibly lands at all.
	// Seeking while paused is a single deterministic jump with nothing
	// racing it. "Was playing" rides as a property on the slider itself
	// rather than a captured variable, since this tab's own connections are
	// the only thing that needs it.
	connect(positionSlider, &QSlider::sliderPressed, page,
			[player, positionSlider, scrubBubble, formatMs, moveBubbleTo]() {
		positionSlider->setProperty("wasPlaying", player->playbackState() == QMediaPlayer::PlayingState);
		player->pause();
		scrubBubble->setText(formatMs(positionSlider->value()));
		moveBubbleTo(positionSlider->value());
		scrubBubble->show();
		scrubBubble->raise();
	});
	connect(positionSlider, &QSlider::sliderMoved, page,
			[player, scrubBubble, formatMs, moveBubbleTo](int position) {
		player->setPosition(position);
		scrubBubble->setText(formatMs(position));
		moveBubbleTo(position);
	});
	connect(positionSlider, &QSlider::sliderReleased, page, [player, positionSlider, scrubBubble]() {
		if (positionSlider->property("wasPlaying").toBool()) player->play();
		scrubBubble->hide();
	});

	page->setProperty("outputPath", videoPath);
	page->setProperty("previewPath", videoPath);
	page->setProperty("infoText", infoText);
	page->setProperty("sceneId", sceneId);

	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);

	player->setSource(QUrl::fromLocalFile(videoPath));
	player->play();
}

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

void MainWindow::createVideoTab() {
	QWidget *videoTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(videoTab);
	layout->setSpacing(12);
	layout->setContentsMargins(12, 12, 12, 12);

	// The tab stays fully interactive regardless of Output Mode (a prior
	// version disabled it outright when Output Mode wasn't "Generate Video",
	// but that blocked browsing/configuring these settings ahead of
	// switching modes) - this banner is the substitute warning, so a
	// configured preset/camera path/frame count never gets silently ignored
	// without the user having seen why. onModeChanged() toggles its
	// visibility; the initial state matches m_videoMode's own default
	// (false - see the MainWindow constructor), so no extra sync call is
	// needed here.
	m_videoModeWarningLabel = new QLabel(
		tr("⚠ These settings only take effect when Output Mode (Basic Settings tab) is set to \"Generate Video\"."),
		videoTab);
	m_videoModeWarningLabel->setObjectName("videoModeWarning");
	m_videoModeWarningLabel->setWordWrap(true);
	m_videoModeWarningLabel->setVisible(!m_videoMode);
	layout->addWidget(m_videoModeWarningLabel);

	// Video parameters group
	QGroupBox *videoGroup = new QGroupBox(tr("Video Generation Settings"), videoTab);
	styleGroupBox(videoGroup);
	QFormLayout *videoLayout = new QFormLayout(videoGroup);
	videoLayout->setVerticalSpacing(10);
	videoLayout->setHorizontalSpacing(10);
	videoLayout->setContentsMargins(15, 22, 15, 12);

	// Preset selector - sets the scene picker (on the Basic tab), camera
	// path, and the three spinboxes below all at once from one of
	// video_preset.h's named bundles. First row, above Camera Path, since
	// picking one is meant to replace tuning the other four controls, not
	// sit alongside them as a fifth independent setting.
	m_videoPresetCombo = new QComboBox();
	m_videoPresetCombo->addItem(tr("(custom - choose settings below)"), QString());
	for (const video_preset::VideoPreset& p : video_preset::kAll)
		m_videoPresetCombo->addItem(
			QString("[%1] %2").arg(QString::fromUtf8(p.id), QString::fromUtf8(p.name)),
			QString::fromUtf8(p.id));
	m_videoPresetCombo->setToolTip(
		tr("Famous ray-tracing reference scenes and motions, pre-tuned so you don't\n"
		"have to set the scene, camera path, frame count, fps, and speed by hand.\n"
		"Selecting one changes the scene on the Basic tab too. Choosing any of the\n"
		"other controls on this tab afterward is fine - they simply stop matching\n"
		"the preset, the same as if you had built the same settings by hand."));
	styleComboBox(m_videoPresetCombo);
	connect(m_videoPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onVideoPresetChanged);
	videoLayout->addRow(labelWithInfo(tr("Preset:"),
		tr("A ready-made bundle of scene + camera path + frame count + fps "
		"+ speed, tuned so the resulting video actually looks good "
		"without hand-picking every setting yourself.\n\n"
		"Picking one fills in every field below (and the scene on the "
		"Basic tab) - you can still change anything afterward, it just "
		"stops matching the preset once you do.")),
		m_videoPresetCombo);

	// Camera path selector
	m_cameraPathCombo = new QComboBox();
	m_cameraPathCombo->addItem(tr("Orbit (Circular rotation)"), "orbit");
	m_cameraPathCombo->addItem(tr("Linear (Straight path)"), "linear");
	m_cameraPathCombo->addItem(tr("Figure-8 (Lemniscate)"), "figure8");
	m_cameraPathCombo->addItem(tr("Spiral (Zoom-in)"), "spiral");
	m_cameraPathCombo->setToolTip(
		tr("How the camera moves over the frame sequence:\n"
		"  Orbit     — full circle around the scene, always looking at its centre\n"
		"  Linear    — straight sweep past the scene\n"
		"  Figure-8  — lemniscate, crossing back through the middle\n"
		"  Spiral    — orbits while moving steadily closer\n"
		"Every path starts from the camera position on the Advanced tab."));
	m_cameraPathCombo->setCurrentIndex(0);
	styleComboBox(m_cameraPathCombo);
	videoLayout->addRow(labelWithInfo(tr("Camera Path:"),
		tr("How the camera moves across the sequence of frames.\n\n"
		"Orbit circles fully around the scene, always facing its center "
		"- the classic \"turntable\" shot. Linear sweeps past in a "
		"straight line. Figure-8 traces a lemniscate, crossing back "
		"through the middle. Spiral orbits while steadily moving closer. "
		"Every path starts from wherever the camera is positioned on "
		"the Advanced tab.")),
		m_cameraPathCombo);

	// Frame count
	m_videoFramesSpinBox = new QSpinBox();
	m_videoFramesSpinBox->setRange(10, 1000);
	m_videoFramesSpinBox->setValue(60);
	m_videoFramesSpinBox->setSuffix(tr(" frames"));
	styleSpinBox(m_videoFramesSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Frame Count:"),
		tr("How many individual images make up the video - each one is a "
		"full, independent render, so this multiplies total render time "
		"directly (100 frames takes roughly 100x as long as one image "
		"at the same settings).\n\n"
		"Paired with Frames Per Second below to determine the video's "
		"total length in seconds.")),
		m_videoFramesSpinBox);

	// FPS (frames per second)
	m_videoFPSSpinBox = new QSpinBox();
	m_videoFPSSpinBox->setRange(15, 120);
	m_videoFPSSpinBox->setValue(30);
	m_videoFPSSpinBox->setSuffix(tr(" fps"));
	styleSpinBox(m_videoFPSSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Frames Per Second:"),
		tr("How many of the rendered frames play per second of video.\n\n"
		"Doesn't change how many frames get rendered (that's Frame "
		"Count above) - only how fast they play back, and therefore how "
		"many seconds long the finished video is (Frame Count divided "
		"by FPS).")),
		m_videoFPSSpinBox);

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
	m_videoSpeedSpinBox->setSuffix(tr("x"));
	styleSpinBox(m_videoSpeedSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Movement Speed:"),
		tr("A multiplier on how many frames the camera's full path is "
		"spread across - not a change to the path itself, which always "
		"completes the same full sweep.\n\n"
		"Speed 0.5x renders twice as many frames to cover the same "
		"journey more slowly and smoothly; speed 2x renders half as "
		"many frames, covering the same journey faster.")),
		m_videoSpeedSpinBox);

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
			? tr("%1 frames").arg(actualFrames)
			: tr("%1 frames (base %2 × 1/%3x speed)%4")
				.arg(actualFrames).arg(baseFrames).arg(QString::number(speed, 'f', 2))
				.arg(capped ? tr(" - capped at 5000") : QString());

		m_videoInfoLabel->setText(tr(
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
	QGroupBox *requirementsGroup = new QGroupBox(tr("ℹ️ Requirements"), videoTab);
	styleGroupBox(requirementsGroup);
	QVBoxLayout *requirementsLayout = new QVBoxLayout(requirementsGroup);

	QLabel *requirementsInfo = new QLabel(
		tr("<b>Requires ffmpeg:</b> Video encoding uses ffmpeg (libx264), which must be installed and on your PATH.<br>"
		"<small>Get it from <a href=\"https://ffmpeg.org/download.html\">ffmpeg.org</a> if the render log reports it's missing.</small><br><br>"
		"<b>Automatic Assembly:</b> After rendering all frames, the video will be automatically assembled and opened.")
	);
	requirementsInfo->setOpenExternalLinks(true);
	requirementsInfo->setWordWrap(true);
	requirementsInfo->setObjectName("mutedInfo");
	requirementsLayout->addWidget(requirementsInfo);

	layout->addWidget(requirementsGroup);

	// Usage instructions
	QGroupBox *usageGroup = new QGroupBox(tr("Usage Instructions"), videoTab);
	styleGroupBox(usageGroup);
	QVBoxLayout *usageLayout = new QVBoxLayout(usageGroup);

	QLabel *usageText = new QLabel(
		tr("<b>Step 1:</b> Configure video settings (camera path, frames, FPS)<br>"
		"<b>Step 2:</b> Configure quality settings in Basic/Advanced tabs<br>"
		"<b>Step 3:</b> Click START VIDEO RENDER and wait<br>"
		"<b>Step 4:</b> Video automatically assembles and opens when done!<br><br>"
		"<b>Tips:</b><br>"
		"• Use GPU mode for faster rendering<br>"
		"• Lower samples/pixel for quick previews (10-50)<br>"
		"• Higher samples/pixel for production quality (100-500)<br>"
		"• Typical render time: 1-5 minutes (GPU), 15-60 minutes (CPU)")
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

	m_videoTabIndex = m_tabWidget->addTab(scrollArea, tr("Video Settings"));
}
