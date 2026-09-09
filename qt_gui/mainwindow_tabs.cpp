// The Settings tab (scene selection, render mode/quality/resolution, video
// generation settings, manual width/height/samples/depth overrides, camera
// position, output path - formerly split across separate "Basic Settings",
// "Advanced Settings", and "Video Settings" tabs with no documented reason
// for the split; merged into one scrollable tab - Video Generation Settings
// stays fully interactive regardless of Output Mode, same as it always has,
// with m_videoModeWarningLabel explaining when it takes effect rather than
// disabling it outright - see that group's own construction comment for
// why), plus the scene-list helpers it uses
// (filteredSceneIds/populateSceneCombo/populateSceneGrid/populateSceneViews/
// thumbnailCachePath/selectSceneById/rebuildCategoryTabs) - split out into
// mainwindow_tabs_render.cpp (Render Options/Preview) and
// mainwindow_tabs_output.cpp (Progress/Log/Diagnostics), which used to live
// in this same file.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"

#include "../src/shared/scene_descriptor.h"
#include "../src/shared/video_preset.h"

#include <QTabBar>
#include "scene_metadata_client.h"
#ifdef RT_GUI_HAVE_GPU
#include "realtime_preview_session.h"
#endif
#include <QStandardItemModel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
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
// otherwise - mirrors createSettingsTab()'s own initial-fill fallback.
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

void MainWindow::createSettingsTab() {
	QWidget *basicTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(basicTab);
	layout->setSpacing(12);
	layout->setContentsMargins(12, 12, 12, 12);

	// --- Scene selection ---
	InfoGroupBox *sceneGroup = new InfoGroupBox("Scene", basicTab);
	styleGroupBox(sceneGroup);
	sceneGroup->setInfoIcon(createInfoIcon(
		tr("Pick which scene to render. Scenes are grouped by category and "
		"searchable; switch to the grid view for thumbnail previews. "
		"Selecting a scene here also seeds its recommended camera/settings "
		"hint below, if it has one.")));
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
	// createSettingsTab() already applied for categories with zero scenes at
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
		tr("Renders a small preview image for each self-contained Basics/Materials/Textures/Cameras\n"
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

	// Non-blocking heads-up when a loaded .pbrt scene's own Sampler/
	// Integrator/light-sampler directive differs from what's currently
	// selected on the Render Options tab - see updateSceneRecommendedSettingsHint()'s
	// own comment (mainwindow_slots.cpp) for the exact mismatch logic, kept
	// in lockstep with cpu_render_main()'s own (console-only) warning.
	// Same objectName/wordWrap/hidden-by-default shape as
	// m_integratorVideoWarningLabelBasic just above, plus a one-click Apply
	// button (applyRecommendedSettings(), mainwindow_slots.cpp) - the one
	// deliberate exception to this hint's own "not applied automatically"
	// text, since clicking it is a real user action, not an automatic
	// override. Both widgets share one row so Apply sits right next to the
	// text it applies, and both toggle visibility together.
	QWidget *recommendedSettingsRow = new QWidget(basicTab);
	QHBoxLayout *recommendedSettingsLayout = new QHBoxLayout(recommendedSettingsRow);
	recommendedSettingsLayout->setContentsMargins(0, 0, 0, 0);
	m_sceneRecommendedSettingsHint = new QLabel(recommendedSettingsRow);
	m_sceneRecommendedSettingsHint->setObjectName("statusWarning");
	m_sceneRecommendedSettingsHint->setWordWrap(true);
	m_sceneRecommendedSettingsHint->setVisible(false);
	recommendedSettingsLayout->addWidget(m_sceneRecommendedSettingsHint, 1);
	m_applyRecommendedSettingsButton = new QPushButton(tr("Apply"), recommendedSettingsRow);
	m_applyRecommendedSettingsButton->setToolTip(
		tr("Set Sampler/Integrator/Light Sampler (Render Options tab) to "
		"this scene's own recommended values."));
	m_applyRecommendedSettingsButton->setVisible(false);
	connect(m_applyRecommendedSettingsButton, &QPushButton::clicked,
			this, &MainWindow::applyRecommendedSettings);
	recommendedSettingsLayout->addWidget(m_applyRecommendedSettingsButton, 0, Qt::AlignTop);
	sceneGroupLayout->addWidget(recommendedSettingsRow);

	connect(m_sceneCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onSceneChanged);

	layout->addWidget(sceneGroup);

	// --- Render settings: output mode, GPU/CPU, quality, resolution ---
	// One group instead of separate "Render Mode" + "Render Settings" boxes -
	// they're all "how do I want this rendered" and splitting them just cost
	// an extra group box's worth of border/title chrome for no real benefit.
	InfoGroupBox *renderGroup = new InfoGroupBox(tr("Render Settings"), basicTab);
	styleGroupBox(renderGroup);
	renderGroup->setInfoIcon(createInfoIcon(
		tr("Choose Output Mode (Single Image, Video, or Live Preview) and "
		"the renderer (GPU or CPU) here, plus a Quality/Resolution preset "
		"or a manual override further down. Video- and Live-Preview-only "
		"fields stay visible and editable even in Image mode, dimmed with "
		"a note - so you can pre-configure them before switching modes.")));
	QFormLayout *renderLayout = new QFormLayout(renderGroup);
	renderLayout->setVerticalSpacing(10);
	renderLayout->setHorizontalSpacing(10);
	renderLayout->setContentsMargins(15, 22, 15, 12);

	m_modeCombo = new QComboBox(basicTab);
	icon_tint::addItem(m_modeCombo, ":/icons/image.svg", tr("Render Single Image"),
					   static_cast<int>(OutputMode::Image), m_activeTheme.textBody);
	icon_tint::addItem(m_modeCombo, ":/icons/video.svg", tr("Generate Video"),
					   static_cast<int>(OutputMode::Video), m_activeTheme.textBody);
#ifdef RT_GUI_HAVE_GPU
	// Added even when realtime_renderer.dll isn't found (RealtimePreviewSession::
	// isAvailable() == false) - disabled with an explanatory tooltip instead
	// of omitted, so the feature is at least discoverable rather than
	// silently missing. Same "fail quiet, explain why" convention
	// initLivePreviewSession()'s own unavailable-session path uses.
	icon_tint::addItem(m_modeCombo, ":/icons/gpu.svg", tr("Live Preview (interactive)"),
					   static_cast<int>(OutputMode::LivePreview), m_activeTheme.textBody);
	if (!RealtimePreviewSession::isAvailable()) {
		const int liveIndex = m_modeCombo->count() - 1;
		// QComboBox uses a QStandardItemModel by default - disabling the
		// underlying QStandardItem (not just adding a tooltip) is what
		// actually greys the row out and blocks selecting it.
		if (auto *model = qobject_cast<QStandardItemModel *>(m_modeCombo->model())) {
			if (QStandardItem *item = model->item(liveIndex)) item->setEnabled(false);
		}
		m_modeCombo->setItemData(liveIndex, tr("realtime_renderer.dll wasn't found next to the application."), Qt::ToolTipRole);
	}
#endif
	m_modeCombo->setCurrentIndex(0);
	styleComboBox(m_modeCombo);
	connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onModeChanged);
	renderLayout->addRow(labelWithInfo(tr("Output Mode:"),
		tr("Whether this render produces a single still frame, a "
		"sequence of frames stitched into a video, or an interactive GPU "
		"preview.\n\n"
		"Single Image renders the scene once, from the camera set on "
		"this tab. Generate Video instead moves "
		"the camera along a path (Video Generation Settings, further "
		"down this tab) and renders one frame per step, then assembles "
		"them into an MP4 - taking roughly Frame Count times as long as "
		"a single image. Live Preview instead renders continuously at a "
		"fixed, small resolution so you can click-drag/scroll to orbit "
		"the camera and see the result converge in real time - it never "
		"writes an output file.\n\nGenerate Video cannot be combined with an alternate "
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

#ifdef RT_GUI_HAVE_GPU
	// Same "banner, don't hide" convention as m_videoModeWarningLabel below -
	// see its own comment (mainwindow.h) for why nothing here gets disabled
	// instead. Visible only in Live Preview mode; toggled by onModeChanged().
	m_liveModeWarningLabel = makeModeWarningBanner(basicTab,
		tr("⚠ Live Preview renders at a fixed, small resolution on the GPU and writes no "
		"output file - Resolution, Samples per Pixel, Max Ray Depth, and Output Path "
		"don't apply. Scene and Camera Position do."));
	renderLayout->addRow(QString(), m_liveModeWarningLabel);
#endif

	m_modeCombo->setToolTip(
		tr("Single Image renders one frame.\n"
		"Generate Video renders a camera path frame by frame and assembles an MP4.\n"
		"Live Preview renders continuously with an orbitable camera - GPU only."));

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
		"Custom leaves the Samples/Max Depth fields below untouched.\n"
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

	// --- Video Generation Settings: only meaningful when Output Mode above
	// is "Generate Video" - formerly its own "Video Settings" tab, kept
	// fully interactive regardless of mode rather than disabled outright
	// (an earlier version of this app DID disable it, and that turned out
	// to block browsing/configuring these settings ahead of switching modes
	// - see commit 9e1c7df8's own message - which also silently breaks
	// onVideoPresetChanged()'s auto-switch-to-Video-mode behavior below,
	// since a disabled combo can't be opened to pick a preset from in the
	// first place). m_videoModeWarningLabel is the same warning-banner
	// pattern that fix introduced, just now scoped to this group instead of
	// a whole standalone tab.
	m_videoGroupBox = new InfoGroupBox(tr("Video Generation Settings"), basicTab);
	styleGroupBox(m_videoGroupBox);
	// Empty for now - populated (and kept up to date) by updateVideoDuration()
	// below, which is where the fuller, dynamic version of this text lives.
	m_videoGroupBox->setInfoIcon(createInfoIcon(QString()));
	// Dimmed (not disabled - see setGroupDimmed()'s own comment) whenever
	// Output Mode isn't "Generate Video", alongside the existing warning
	// banner right below - the banner explains WHY, the dim reinforces AT A
	// GLANCE that this whole group is currently inert, without blocking a
	// user who wants to pre-configure it before switching modes.
	setGroupDimmed(m_videoGroupBox, !isVideoMode());
	QFormLayout *videoLayout = new QFormLayout(m_videoGroupBox);
	videoLayout->setVerticalSpacing(10);
	videoLayout->setHorizontalSpacing(10);
	videoLayout->setContentsMargins(15, 22, 15, 12);

	m_videoModeWarningLabel = makeModeWarningBanner(m_videoGroupBox,
		tr("⚠ These settings only take effect when Output Mode above is set to \"Generate Video\"."));
	m_videoModeWarningLabel->setVisible(!isVideoMode());
	videoLayout->addRow(m_videoModeWarningLabel);

	// Preset selector - sets the scene picker above, camera path, and the
	// three spinboxes below all at once from one of video_preset.h's named
	// bundles. First row, above Camera Path, since picking one is meant to
	// replace tuning the other four controls, not sit alongside them as a
	// fifth independent setting.
	m_videoPresetCombo = new QComboBox();
	m_videoPresetCombo->addItem(tr("(custom - choose settings below)"), QString());
	for (const video_preset::VideoPreset& p : video_preset::kAll)
		m_videoPresetCombo->addItem(
			QString("[%1] %2").arg(QString::fromUtf8(p.id), QString::fromUtf8(p.name)),
			QString::fromUtf8(p.id));
	m_videoPresetCombo->setToolTip(
		tr("Famous ray-tracing reference scenes and motions, pre-tuned so you don't\n"
		"have to set the scene, camera path, frame count, fps, and speed by hand.\n"
		"Selecting one changes the Scene above too. Choosing any of the\n"
		"other controls on this tab afterward is fine - they simply stop matching\n"
		"the preset, the same as if you had built the same settings by hand."));
	styleComboBox(m_videoPresetCombo);
	connect(m_videoPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onVideoPresetChanged);
	videoLayout->addRow(labelWithInfo(tr("Preset:"),
		tr("A ready-made bundle of scene + camera path + frame count + fps "
		"+ speed, tuned so the resulting video actually looks good "
		"without hand-picking every setting yourself.\n\n"
		"Picking one fills in every field below (and the Scene above) - "
		"you can still change anything afterward, it just "
		"stops matching the preset once you do.")),
		m_videoPresetCombo);

	// Camera path selector
	m_cameraPathCombo = new QComboBox();
	m_cameraPathCombo->addItem(tr("Orbit (Circular rotation)"), "orbit");
	m_cameraPathCombo->addItem(tr("Linear (Straight path)"), "linear");
	m_cameraPathCombo->addItem(tr("Figure-8 (Lemniscate)"), "figure8");
	m_cameraPathCombo->addItem(tr("Spiral (Zoom-in)"), "spiral");
	m_cameraPathCombo->addItem(tr("Tour (Room walkthrough)"), "tour");
	m_cameraPathCombo->addItem(tr("Showcase (Product reveal)"), "showcase");
	m_cameraPathCombo->setToolTip(
		tr("How the camera moves over the frame sequence:\n"
		"  Orbit     — full circle around the scene, always looking at its centre\n"
		"  Linear    — straight sweep past the scene\n"
		"  Figure-8  — lemniscate, crossing back through the middle\n"
		"  Spiral    — orbits while moving steadily closer\n"
		"  Tour      — sways side to side and glides forward while looking around, like walking through a room\n"
		"  Showcase  — one eased turn that pushes in and arcs up-then-down, like a product ad\n"
		"Every path starts from the camera position set below."));
	m_cameraPathCombo->setCurrentIndex(0);
	styleComboBox(m_cameraPathCombo);
	videoLayout->addRow(labelWithInfo(tr("Camera Path:"),
		tr("How the camera moves across the sequence of frames.\n\n"
		"Orbit circles fully around the scene, always facing its center "
		"- the classic \"turntable\" shot. Linear sweeps past in a "
		"straight line. Figure-8 traces a lemniscate, crossing back "
		"through the middle. Spiral orbits while steadily moving closer. "
		"Tour sways side to side and glides forward while its look-at "
		"point drifts too, like an actual visitor walking through and "
		"looking around a room. Showcase turns once around the subject "
		"with an eased push-in and a gentle rise-and-fall, like a "
		"product advertisement's hero shot. Every path starts from "
		"wherever the camera is positioned further down this tab.")),
		m_cameraPathCombo);

	// Frame Count + Frames Per Second on one line - same 4-column-grid-as-
	// a-single-row trick used for the sensitivity pair in Live Preview
	// Settings below, rather than converting this whole group from
	// QFormLayout to QGridLayout (like Advanced Parameters/Camera Position
	// do) just for this one pair.
	QWidget *frameRateRow = new QWidget();
	QGridLayout *frameRateGrid = new QGridLayout(frameRateRow);
	frameRateGrid->setContentsMargins(0, 0, 0, 0);
	frameRateGrid->setHorizontalSpacing(10);
	frameRateGrid->setColumnStretch(1, 1);
	frameRateGrid->setColumnStretch(3, 1);

	m_videoFramesSpinBox = new QSpinBox();
	m_videoFramesSpinBox->setRange(10, 1000);
	m_videoFramesSpinBox->setValue(60);
	m_videoFramesSpinBox->setSuffix(tr(" frames"));
	styleSpinBox(m_videoFramesSpinBox);
	frameRateGrid->addWidget(labelWithInfo(tr("Frame Count:"),
		tr("How many individual images make up the video - each one is a "
		"full, independent render, so this multiplies total render time "
		"directly (100 frames takes roughly 100x as long as one image "
		"at the same settings).\n\n"
		"Paired with Frames Per Second to determine the video's total "
		"length in seconds.")),
		0, 0);
	frameRateGrid->addWidget(m_videoFramesSpinBox, 0, 1);

	m_videoFPSSpinBox = new QSpinBox();
	m_videoFPSSpinBox->setRange(15, 120);
	m_videoFPSSpinBox->setValue(30);
	m_videoFPSSpinBox->setSuffix(tr(" fps"));
	styleSpinBox(m_videoFPSSpinBox);
	frameRateGrid->addWidget(labelWithInfo(tr("Frames Per Second:"),
		tr("How many of the rendered frames play per second of video.\n\n"
		"Doesn't change how many frames get rendered (that's Frame "
		"Count) - only how fast they play back, and therefore how many "
		"seconds long the finished video is (Frame Count divided by "
		"FPS).")),
		0, 2);
	frameRateGrid->addWidget(m_videoFPSSpinBox, 0, 3);

	videoLayout->addRow(frameRateRow);

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

	// Video duration/summary info (calculated from frames/fps), plus the
	// static ffmpeg-requirement/step-by-step usage text a separate
	// "Video render requirements & usage:" row used to show below this
	// group - both live on the group's OWN header icon (set up above)
	// rather than a dedicated row/icon of their own, so there's exactly
	// one place to check for info about this group instead of two a user
	// has to notice are both worth hovering. Tooltip is rewritten on every
	// recompute, same "content changes after construction" pattern
	// m_sceneTechInfoIcon already uses (updateSceneTechInfoIcon(),
	// mainwindow_style.cpp).
	m_videoInfoIcon = m_videoGroupBox->infoIcon();

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
			: tr("%1 frames (base %2 x 1/%3x speed)%4")
				.arg(actualFrames).arg(baseFrames).arg(QString::number(speed, 'f', 2))
				.arg(capped ? tr(" - capped at 5000") : QString());

		// Plain text, not HTML - createInfoIcon()'s helpText is escaped
		// before display (see wrapTooltipHtml()'s own comment), so the
		// <b>/<code> tags the old always-visible label used would show up
		// as literal text here instead of formatting.
		// The Requirements/Usage half below never changes, only the Duration/
		// Camera Path/Output half above does - re-concatenated on every
		// recompute rather than split into two tooltips/icons (see this
		// icon's own comment above for why one icon covers both).
		m_videoInfoIcon->setToolTip(wrapTooltipHtml(tr(
			"Video Duration: %1 seconds (%2)\n\n"
			"Camera Path: %3, always completes its full sweep regardless of speed\n\n"
			"Output: frames will be saved to output/frames/\n\n"
			"Requires ffmpeg: video encoding uses ffmpeg (libx264), which must "
			"be installed and on your PATH - get it from ffmpeg.org if the "
			"render log reports it's missing.\n\n"
			"After rendering all frames, the video is automatically assembled "
			"and opened.\n\n"
			"Step 1: Configure Video Generation Settings above (camera path, "
			"frames, FPS) and set Output Mode to Generate Video.\n\n"
			"Step 2: Configure quality settings further down this tab.\n\n"
			"Step 3: Click START VIDEO RENDER and wait.\n\n"
			"Step 4: Video automatically assembles and opens when done!\n\n"
			"Tips: use GPU mode for faster rendering. Lower samples/pixel "
			"(10-50) for quick previews, higher (100-500) for production "
			"quality. Typical render time is 1-5 minutes on GPU, 15-60 minutes "
			"on CPU."
		).arg(QString::number(duration, 'f', 1), framesLine, cameraPath)));
	};

	connect(m_videoFramesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoFPSSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoSpeedSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), updateVideoDuration);
	connect(m_cameraPathCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateVideoDuration);
	updateVideoDuration();

	layout->addWidget(m_videoGroupBox);

#ifdef RT_GUI_HAVE_GPU
	// --- Live Preview Settings: mouse/keyboard sensitivity - its own
	// independent group rather than living inline in Render Settings, so
	// it's as discoverable as Video's own settings are. No separate warning
	// banner (unlike Video Generation Settings) - the header icon's own
	// tooltip already states the "only takes effect when..." caveat, so a
	// banner repeating the same sentence would be pure duplication;
	// setGroupDimmed() (keyed on isLiveMode() instead of isVideoMode())
	// still gives the same at-a-glance "inert right now" visual cue. Placed
	// right below Video Generation Settings - the two other Output Mode
	// options' own dedicated settings sit next to each other, immediately
	// under the combo that picks between them.
	m_liveModeSettingsGroupBox = new InfoGroupBox(tr("Live Preview Settings"), basicTab);
	styleGroupBox(m_liveModeSettingsGroupBox);
	m_liveModeSettingsGroupBox->setInfoIcon(createInfoIcon(
		tr("Tune how responsive mouse orbit/zoom and keyboard WASD/Up/Down "
		"movement + Left/Right/+/- feel in Live Preview. Only takes effect "
		"when Output Mode above is \"Live Preview (interactive)\", but "
		"stays editable in any mode.")));
	setGroupDimmed(m_liveModeSettingsGroupBox, !isLiveMode());
	QFormLayout *liveModeSettingsLayout = new QFormLayout(m_liveModeSettingsGroupBox);
	liveModeSettingsLayout->setVerticalSpacing(10);
	liveModeSettingsLayout->setHorizontalSpacing(10);
	liveModeSettingsLayout->setContentsMargins(15, 22, 15, 12);

	// One multiplier per INPUT DEVICE (not one per axis - azimuth/
	// elevation/radius all scale together per device), loaded from/saved
	// to QSettings immediately on change (loadSavedMouseSensitivity()/
	// saveMouseSensitivity() etc., mainwindow_tabs_render.cpp) the same
	// way theme/font/language already are. Both on one line, same
	// 4-column-grid-as-a-single-row trick as Video Generation Settings'
	// Frame Count/FPS pair above.
	QWidget *sensitivityRow = new QWidget();
	QGridLayout *sensitivityGrid = new QGridLayout(sensitivityRow);
	sensitivityGrid->setContentsMargins(0, 0, 0, 0);
	sensitivityGrid->setHorizontalSpacing(10);
	sensitivityGrid->setColumnStretch(1, 1);
	sensitivityGrid->setColumnStretch(3, 1);

	m_mouseSensitivitySpinBox = new QDoubleSpinBox();
	m_mouseSensitivitySpinBox->setRange(0.25, 3.0);
	m_mouseSensitivitySpinBox->setSingleStep(0.25);
	m_mouseSensitivitySpinBox->setDecimals(2);
	m_mouseSensitivitySpinBox->setValue(m_mouseSensitivity);
	m_mouseSensitivitySpinBox->setSuffix(tr("x"));
	styleSpinBox(m_mouseSensitivitySpinBox);
	connect(m_mouseSensitivitySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_mouseSensitivity = value;
		saveMouseSensitivity(value);
	});
	sensitivityGrid->addWidget(labelWithInfo(tr("Mouse Sensitivity:"),
		tr("Scales click-drag-to-orbit and scroll-to-zoom speed in Live "
		"Preview. 1x matches the original feel; lower is gentler, higher "
		"is more responsive.")),
		0, 0);
	sensitivityGrid->addWidget(m_mouseSensitivitySpinBox, 0, 1);

	m_keyboardSensitivitySpinBox = new QDoubleSpinBox();
	m_keyboardSensitivitySpinBox->setRange(0.25, 3.0);
	m_keyboardSensitivitySpinBox->setSingleStep(0.25);
	m_keyboardSensitivitySpinBox->setDecimals(2);
	m_keyboardSensitivitySpinBox->setValue(m_keyboardSensitivity);
	m_keyboardSensitivitySpinBox->setSuffix(tr("x"));
	styleSpinBox(m_keyboardSensitivitySpinBox);
	connect(m_keyboardSensitivitySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_keyboardSensitivity = value;
		saveKeyboardSensitivity(value);
	});
	sensitivityGrid->addWidget(labelWithInfo(tr("Keyboard Sensitivity:"),
		tr("Scales WASD/Up/Down movement, Left/Right-arrow orbit, and +/- "
		"zoom step size in Live Preview. 1x is a moderate per-press nudge; "
		"lower is finer, higher moves further per press.")),
		0, 2);
	sensitivityGrid->addWidget(m_keyboardSensitivitySpinBox, 0, 3);

	liveModeSettingsLayout->addRow(sensitivityRow);

	// OptiX AI denoiser - same denoiser/blend the Render Options tab's
	// m_denoiseCheck/m_denoiseBlendSpin already run for batch/video
	// rendering, applied to the realtime path too (rt_realtime_render_frame()'s
	// own comment). "Show latest frame" only makes sense alongside denoise
	// (a single raw noisy frame has no redeeming value over accumulating),
	// so it's enabled/disabled in lockstep with the main checkbox exactly
	// like the blend spinbox already is.
	QWidget *liveDenoiseRow = new QWidget();
	QHBoxLayout *liveDenoiseRowLayout = new QHBoxLayout(liveDenoiseRow);
	liveDenoiseRowLayout->setContentsMargins(0, 0, 0, 0);
	liveDenoiseRowLayout->setSpacing(10);

	m_liveDenoiseCheck = new QCheckBox(tr("OptiX AI Denoiser"));
	m_liveDenoiseCheck->setChecked(m_liveDenoiseEnabled);
	m_liveDenoiseCheck->setToolTip(
		tr("Run the OptiX AI denoiser on every Live Preview frame - the same "
		"one the Render Options tab's own Denoiser checkbox runs for "
		"finished renders."));
	styleCheckBox(m_liveDenoiseCheck);

	m_liveDenoiseBlendSpin = new QDoubleSpinBox();
	m_liveDenoiseBlendSpin->setRange(0.0, 1.0);
	m_liveDenoiseBlendSpin->setDecimals(2);
	m_liveDenoiseBlendSpin->setSingleStep(0.05);
	m_liveDenoiseBlendSpin->setValue(m_liveDenoiseBlend);
	m_liveDenoiseBlendSpin->setEnabled(m_liveDenoiseEnabled);
	m_liveDenoiseBlendSpin->setToolTip(
		tr("Blend between the noisy input and the fully denoised output\n"
		"(0.0 = 100% denoised, 1.0 = original noisy image), same meaning as "
		"the Render Options tab's own blend control."));
	styleSpinBox(m_liveDenoiseBlendSpin);
	connect(m_liveDenoiseCheck, &QCheckBox::toggled, m_liveDenoiseBlendSpin, &QDoubleSpinBox::setEnabled);

	m_liveDenoiseShowLatestCheck = new QCheckBox(tr("Show latest frame instead of accumulating"));
	m_liveDenoiseShowLatestCheck->setChecked(m_liveDenoiseShowLatest);
	m_liveDenoiseShowLatestCheck->setEnabled(m_liveDenoiseEnabled);
	styleCheckBox(m_liveDenoiseShowLatestCheck);
	connect(m_liveDenoiseCheck, &QCheckBox::toggled, m_liveDenoiseShowLatestCheck, &QCheckBox::setEnabled);

	// All three push straight to the running session (if any) as well as
	// QSettings, via the shared pushLiveDenoiseToSession() helper - see
	// RealtimePreviewSession::setDenoise()'s own comment on when this does
	// and doesn't reset accumulation.
	connect(m_liveDenoiseCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveDenoiseEnabled = checked;
		saveLiveDenoiseEnabled(checked);
		pushLiveDenoiseToSession();
	});
	connect(m_liveDenoiseBlendSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveDenoiseBlend = value;
		saveLiveDenoiseBlend(value);
		pushLiveDenoiseToSession();
	});
	connect(m_liveDenoiseShowLatestCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveDenoiseShowLatest = checked;
		saveLiveDenoiseShowLatest(checked);
		pushLiveDenoiseToSession();
	});

	liveDenoiseRowLayout->addWidget(checkboxWithInfo(m_liveDenoiseCheck,
		tr("Cleans up Live Preview's noisy low-sample image using the same "
		"OptiX AI denoiser the Render Options tab's own Denoiser checkbox "
		"runs for finished renders - lets the view look reasonable almost "
		"immediately instead of waiting many frames to converge. Costs a "
		"small amount of GPU time per frame. The number to its right blends "
		"between the noisy original and the fully denoised result, same as "
		"the Render Options tab's own blend control.")));
	liveDenoiseRowLayout->addWidget(m_liveDenoiseBlendSpin);
	liveDenoiseRowLayout->addWidget(checkboxWithInfo(m_liveDenoiseShowLatestCheck,
		tr("Displays each denoised frame as-is instead of averaging it into "
		"a running mean with earlier frames. Trades away the extra quality "
		"accumulating more samples would eventually reach, in exchange for "
		"a view that always reflects only the most recent frame - useful "
		"while flying around with WASD, where older accumulated frames are "
		"from a camera position you've already left.")));
	liveDenoiseRowLayout->addStretch(1);

	liveModeSettingsLayout->addRow(liveDenoiseRow);

	layout->addWidget(m_liveModeSettingsGroupBox);
#endif

	// --- Advanced Parameters: manual width/height/samples/depth overrides ---
	// Formerly its own "Advanced Settings" tab - folded in here since there
	// was never a documented reason for the split (see git history), and
	// keeping the Quality preset above and the exact values it writes into
	// Width/Height/Samples/Max Depth below on the same tab reads more like
	// one coherent "how big and how clean" decision than two.
	m_advancedParamsGroupBox = new InfoGroupBox(tr("Advanced Parameters"), basicTab);
	styleGroupBox(m_advancedParamsGroupBox);
	m_advancedParamsGroupBox->setInfoIcon(createInfoIcon(
		tr("Manually override resolution, samples per pixel, and max ray "
		"depth instead of using the Quality/Resolution presets above. "
		"Shared by Image and Video (Video reuses these as its per-frame "
		"settings) - Live Preview always uses its own fixed, small "
		"resolution instead.")));
	// Dimmed (not disabled) whenever Live Preview is selected - see
	// m_liveModeWarningLabel's own comment for why these specifically don't
	// apply there, and setGroupDimmed()'s comment for why dim rather than
	// hide/disable. Image and Video both use this group, so this is keyed
	// on isLiveMode(), not isVideoMode() - the opposite condition from
	// m_videoGroupBox just above.
	setGroupDimmed(m_advancedParamsGroupBox, isLiveMode());
	// A 4-column grid (label+info, field, label+info, field) instead of
	// QFormLayout's one-pair-per-row - two related dials per line (Width/
	// Height, then Samples/Max Depth) halves this group's height without
	// losing anything: each field keeps its own label, info icon, and
	// tooltip exactly as before, just packed two to a row.
	QGridLayout *advancedGrid = new QGridLayout(m_advancedParamsGroupBox);
	advancedGrid->setVerticalSpacing(10);
	advancedGrid->setHorizontalSpacing(10);
	advancedGrid->setContentsMargins(15, 22, 15, 12);
	advancedGrid->setColumnStretch(1, 1);
	advancedGrid->setColumnStretch(3, 1);

	// Width
	m_widthSpinBox = new QSpinBox(basicTab);
	m_widthSpinBox->setRange(100, 4096);
	m_widthSpinBox->setValue(800);
	styleSpinBox(m_widthSpinBox);
	advancedGrid->addWidget(labelWithInfo(tr("Width:"),
		tr("The image's pixel width.\n\n"
		"Paired with Height to set the resolution manually, "
		"overriding whatever the Quality preset above would "
		"otherwise use.")),
		0, 0);
	advancedGrid->addWidget(m_widthSpinBox, 0, 1);

	// Height
	m_heightSpinBox = new QSpinBox(basicTab);
	m_heightSpinBox->setRange(100, 4096);
	m_heightSpinBox->setValue(800);
	styleSpinBox(m_heightSpinBox);
	advancedGrid->addWidget(labelWithInfo(tr("Height:"),
		tr("The image's pixel height.\n\n"
		"Paired with Width - together they set the resolution "
		"manually, overriding the Quality preset above.")),
		0, 2);
	advancedGrid->addWidget(m_heightSpinBox, 0, 3);

	// Samples
	m_samplesSpinBox = new QSpinBox(basicTab);
	m_samplesSpinBox->setRange(1, 10000);
	m_samplesSpinBox->setValue(100);
	styleSpinBox(m_samplesSpinBox);
	m_samplesSpinBox->setToolTip(
		tr("Rays traced per pixel. This is the main quality/time dial: noise falls\n"
		"as the square root of this value, so halving the noise costs about 4x\n"
		"the render time. Setting it here switches Quality to Custom."));
	advancedGrid->addWidget(labelWithInfo(tr("Samples per Pixel:"),
		tr("Ray tracing estimates each pixel's color by firing many random "
		"rays and averaging the results, like polling a lot of people and "
		"averaging their guesses.\n\n"
		"More samples means a more accurate average, which shows up as "
		"less speckly \"noise\" in the image - but each extra sample "
		"costs render time. Doubling this value roughly halves the "
		"noise, but takes about twice as long to render.")),
		1, 0);
	advancedGrid->addWidget(m_samplesSpinBox, 1, 1);

	// Max depth
	m_maxDepthSpinBox = new QSpinBox(basicTab);
	m_maxDepthSpinBox->setRange(1, 100);
	m_maxDepthSpinBox->setValue(50);
	styleSpinBox(m_maxDepthSpinBox);
	m_maxDepthSpinBox->setToolTip(
		tr("How many times a ray may bounce before it is terminated. Low values\n"
		"darken glass and mirrors, which need many bounces to resolve; scenes\n"
		"of plain diffuse surfaces look the same well below the maximum."));
	advancedGrid->addWidget(labelWithInfo(tr("Max Ray Depth:"),
		tr("A depth of 1 means a ray only sees what it hits directly, with "
		"no bounced light at all - like a scene with no reflections or "
		"indirect lighting.\n\n"
		"Each extra bounce lets light travel one more surface before "
		"giving up, which is what makes glass, mirrors, and soft "
		"indirect lighting look correct. Most scenes look \"finished\" "
		"well before the maximum - beyond that, extra depth mostly "
		"traces light too dim to matter.")),
		1, 2);
	advancedGrid->addWidget(m_maxDepthSpinBox, 1, 3);

	layout->addWidget(m_advancedParamsGroupBox);

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

	InfoGroupBox *cameraGroup = new InfoGroupBox(tr("Camera Position"), basicTab);
	styleGroupBox(cameraGroup);
	cameraGroup->setInfoIcon(createInfoIcon(
		tr("Set the camera's world position directly, or pick a named "
		"preset. Used as-is for Image mode, as the starting point Video's "
		"camera path animates from, and as Live Preview's initial "
		"position before you orbit/zoom it interactively.")));
	// Same 4-column grid as Advanced Parameters above: X/Y and Z/Distance
	// pack two fields per row instead of QFormLayout's one-pair-per-row.
	// Preset spans the field columns on its own row since there's nothing
	// to pair it with.
	QGridLayout *cameraLayout = new QGridLayout(cameraGroup);
	cameraLayout->setVerticalSpacing(10);
	cameraLayout->setHorizontalSpacing(10);
	cameraLayout->setContentsMargins(15, 22, 15, 12);
	cameraLayout->setColumnStretch(1, 1);
	cameraLayout->setColumnStretch(3, 1);

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
	m_cameraPresetCombo = new QComboBox(basicTab);

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
	cameraLayout->addWidget(labelWithInfo(tr("Preset:"),
		tr("A handful of hand-picked camera positions for this scene, framed "
		"to show off something specific (e.g. looking in through the "
		"front, or from inside a Cornell-box-style enclosure).\n\n"
		"Choosing \"Custom\" unlocks the X/Y/Z fields below so you can "
		"fly the camera anywhere you like instead.")),
		0, 0);
	cameraLayout->addWidget(m_cameraPresetCombo, 0, 1, 1, 3);

	// Camera position spinboxes (X, Y, Z coordinates)
	// These are disabled by default; only enabled when "Custom" preset is selected
	// Range: -2000 to 2000 allows positioning far outside the box if needed

	m_cameraPosX = new QDoubleSpinBox(basicTab);
	m_cameraPosX->setRange(-2000, 2000);
	m_cameraPosX->setValue(278);  // Default X: centered horizontally
	m_cameraPosX->setSingleStep(10);
	m_cameraPosX->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosX);
	cameraLayout->addWidget(labelWithInfo(tr("Camera X:"),
		tr("The camera's position along the world's X axis (left/right).\n\n"
		"Only editable when the preset above is set to Custom - the "
		"camera always looks toward the scene's own fixed look-at point, "
		"so moving X/Y/Z changes the viewing angle and distance, not "
		"just a straight left-right pan.")),
		1, 0);
	cameraLayout->addWidget(m_cameraPosX, 1, 1);

	m_cameraPosY = new QDoubleSpinBox(basicTab);
	m_cameraPosY->setRange(-2000, 2000);
	m_cameraPosY->setValue(278);  // Default Y: centered vertically
	m_cameraPosY->setSingleStep(10);
	m_cameraPosY->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosY);
	cameraLayout->addWidget(labelWithInfo(tr("Camera Y:"),
		tr("The camera's position along the world's Y axis (up/down).\n\n"
		"Same Custom-preset-only editing rule as Camera X - the camera "
		"keeps looking at the scene's fixed look-at point as you move "
		"it.")),
		1, 2);
	cameraLayout->addWidget(m_cameraPosY, 1, 3);

	m_cameraPosZ = new QDoubleSpinBox(basicTab);
	m_cameraPosZ->setRange(-2000, 2000);
	m_cameraPosZ->setValue(-800);  // Default Z: far back view to match default preset
	m_cameraPosZ->setSingleStep(10);
	m_cameraPosZ->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraPosZ);
	cameraLayout->addWidget(labelWithInfo(tr("Camera Z:"),
		tr("The camera's position along the world's Z axis (forward/back, "
		"into or out of the scene).\n\n"
		"Same Custom-preset-only editing rule as Camera X/Y.")),
		2, 0);
	cameraLayout->addWidget(m_cameraPosZ, 2, 1);

#ifdef RT_GUI_HAVE_GPU
	// Live Preview (addLivePreviewTab()) forwards a camera move onto its
	// already-running session - a no-op whenever that session isn't running
	// (onLivePreviewCameraChanged()'s own guard), so this connect is always
	// safe to make here regardless of whether the live sub-tab happens to
	// be open right now.
	connect(m_cameraPosX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::onLivePreviewCameraChanged);
	connect(m_cameraPosY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::onLivePreviewCameraChanged);
	connect(m_cameraPosZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::onLivePreviewCameraChanged);
#endif

	// Distance from the current scene's look-at point. Adjusting this moves
	// the camera along its EXISTING viewing direction to the new distance
	// (see onCameraDistanceChanged) - a quick way to zoom in/out without
	// having to work out new X/Y/Z coordinates by hand. Only meaningful (and
	// only enabled) alongside the X/Y/Z spinboxes for "Custom"; its value is
	// kept in sync (not user-editable-then-stale) whenever the scene or
	// preset changes, via refreshCameraDistanceDisplay().
	m_cameraDistance = new QDoubleSpinBox(basicTab);
	m_cameraDistance->setRange(0.01, 5000);
	m_cameraDistance->setValue(1078);  // Matches the default preset's distance from Cornell Box's lookat
	m_cameraDistance->setSingleStep(10);
	m_cameraDistance->setEnabled(false);  // Disabled until "Custom" is selected
	styleSpinBox(m_cameraDistance);
	cameraLayout->addWidget(labelWithInfo(tr("Distance from Center:"),
		tr("Moves the camera directly toward or away from the scene's "
		"look-at point along whatever direction it's currently facing, "
		"without changing which way it's pointed.\n\n"
		"The quickest way to zoom in or pull back once you've already "
		"found an angle you like via the X/Y/Z fields or a preset.")),
		2, 2);
	cameraLayout->addWidget(m_cameraDistance, 2, 3);

	// Connect preset combo to handler that updates spinboxes and enables/disables manual input
	// Connection made AFTER all widgets are created to avoid null pointer issues
	connect(m_cameraPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onCameraPresetChanged);
	connect(m_cameraDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			this, &MainWindow::onCameraDistanceChanged);

	// Initialize the spinboxes with the default preset (index 0: "Front View (Outside)")
	onCameraPresetChanged(0);

	layout->addWidget(cameraGroup);

	// Output group
	InfoGroupBox *outputGroup = new InfoGroupBox(tr("Output"), basicTab);
	styleGroupBox(outputGroup);
	outputGroup->setInfoIcon(createInfoIcon(
		tr("Where the rendered file is saved. Video mode appends the "
		"correct extension automatically; Live Preview ignores this "
		"entirely since it never writes a file.")));
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

	m_tabWidget->addTab(scrollArea, tr("Settings"));
}

