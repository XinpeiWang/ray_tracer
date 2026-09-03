// Basic Settings and Advanced Settings tabs, plus the scene-list helpers
// they share (filteredSceneIds/populateSceneCombo/populateSceneGrid/
// populateSceneViews/thumbnailCachePath/selectSceneById/rebuildCategoryTabs)
// - split out into mainwindow_tabs_render.cpp (Render Options/Preview/Video)
// and mainwindow_tabs_output.cpp (Progress/Log/Diagnostics), which used to
// live in this same file.
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
	searchRow->addWidget(createInfoIcon(
		tr("Narrows the scene list/grid below by substring match against "
		"each scene's name, id, or description - on top of, not instead "
		"of, the availability and category tabs above.\n\n"
		"Clear it (the small \"x\" inside the field) to see every scene "
		"in the current category again.")));

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
	renderLayout->addRow(labelWithInfo(tr("Output Mode:"),
		tr("Whether this render produces a single still frame, or a "
		"sequence of frames stitched into a video.\n\n"
		"Single Image renders the scene once, from the camera set on "
		"this tab (or Advanced Settings). Generate Video instead moves "
		"the camera along a path (Video Settings, on the Render Options "
		"tab) and renders one frame per step, then assembles them into "
		"an MP4 - taking roughly Frame Count times as long as a single "
		"image.\n\nGenerate Video cannot be combined with an alternate "
		"Integrator - see the warning below if that combination is "
		"picked.")),
		m_modeCombo);

	// See m_integratorVideoWarningLabelBasic's own comment (mainwindow.h) -
	// a second copy of the Render Options tab's warning, here next to the
	// control (Output Mode) that actually triggers the conflict.
	m_integratorVideoWarningLabelBasic = new QLabel(
		tr("⚠ Generate Video cannot be combined with an alternate integrator - "
		"switch back to Path Tracer, or to Single Image output."), basicTab);
	m_integratorVideoWarningLabelBasic->setObjectName("statusWarning");
	m_integratorVideoWarningLabelBasic->setWordWrap(true);
	m_integratorVideoWarningLabelBasic->setVisible(false);
	renderLayout->addRow(QString(), m_integratorVideoWarningLabelBasic);

	m_modeCombo->setToolTip(
		tr("Single Image renders one frame.\n"
		"Generate Video renders a camera path frame by frame and assembles an MP4."));

	m_renderModeCombo = new QComboBox(basicTab);
#ifdef RT_GUI_HAVE_GPU
	icon_tint::addItem(m_renderModeCombo, ":/icons/gpu.svg", tr("GPU (CUDA) - Fast"), true, m_activeTheme.textBody);
	m_renderModeCombo->setItemData(m_renderModeCombo->count() - 1, wrapTooltipHtml(
		tr("NVIDIA OptiX hardware ray tracing. Typically orders of "
		"magnitude faster than CPU, but needs a CUDA-capable NVIDIA GPU "
		"and doesn't yet implement every material the CPU path does.")),
		Qt::ToolTipRole);
#endif
	icon_tint::addItem(m_renderModeCombo, ":/icons/cpu.svg", tr("CPU - High Quality"), false, m_activeTheme.textBody);
	m_renderModeCombo->setItemData(m_renderModeCombo->count() - 1, wrapTooltipHtml(
		tr("The full importance-sampled path tracer. Runs on any machine "
		"and supports every scene and material this app implements, "
		"including the handful the GPU backend hasn't caught up to yet - "
		"at the cost of being much slower.")),
		Qt::ToolTipRole);
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
	m_gpuBackendCombo->setItemData(m_gpuBackendCombo->count() - 1, wrapTooltipHtml(
		tr("One thread per pixel, tracing each ray recursively bounce by "
		"bounce. The default GPU path tracer - broad, battle-tested "
		"coverage of scenes and materials.")),
		Qt::ToolTipRole);
	icon_tint::addItem(m_gpuBackendCombo, ":/icons/gpu.svg", tr("Wavefront (Experimental)"), true, m_activeTheme.textBody);
	m_gpuBackendCombo->setItemData(m_gpuBackendCombo->count() - 1, wrapTooltipHtml(
		tr("Splits each bounce into separate queue-passed kernel launches, "
		"batching rays doing the same kind of work together. Better GPU "
		"utilization on complex, divergent scenes - but a newer, less "
		"exercised code path.")),
		Qt::ToolTipRole);
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

	// Integrator selector lives on the Render Options tab now (colocated
	// with its own per-mode Integrator Options group, immediately below
	// it) - see createRenderOptionsTab() for m_integratorCombo/
	// m_integratorVideoWarningLabel's construction.

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
	// <exe_dir>/output/ - matches launcher/main.cpp's own default for the CLI
	// (<exe_dir>/output/image.ppm) rather than the Desktop this used to
	// default to. applicationDirPath() is RayTracerGUI.exe's own directory,
	// the same RayTracer_Package/ the CLI exe is deployed into, so this
	// lands in the exact same place a bare `ray_tracer.exe` invocation
	// (no --output) would. See recent_renders.cpp's own comment - its
	// Desktop-scan backfill was updated to match this new default too.
	QString defaultPath = QApplication::applicationDirPath() + "/output/render_" + timestamp + ".png";
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
	ThemedScrollArea *scrollArea = new ThemedScrollArea();  // theme motif support - see that class's own comment
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
	ThemedScrollArea *scrollArea = new ThemedScrollArea();  // theme motif support - see that class's own comment
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
