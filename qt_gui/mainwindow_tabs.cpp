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
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
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
#include <QTemporaryFile>
#include <QFile>
#include <QToolButton>
#include <QButtonGroup>
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
		setRichItemTooltip(m_sceneCombo, m_sceneCombo->count() - 1,
			sceneTooltipPlainText(id, /*includeHeading=*/false));
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
		setRichItemTooltip(item, SceneMetadataClient::sceneRequiresFiles(id)
			? QString("[%1] %2").arg(id, SceneMetadataClient::sceneName(id))
			: sceneTooltipPlainText(id, /*includeHeading=*/true));
		m_sceneGrid->addItem(item);
	}
}

void MainWindow::populateSceneViews(const QString &category) {
	populateSceneCombo(category);
	populateSceneGrid(category);
	updateMyScenesButtons(category);
}

QString MainWindow::thumbnailCachePath(const QString &sceneId) const {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + "/thumbnails";
	// Keyed by slug: an id such as "K37" names a different scene once a file is added to pbrt_scenes/, which would show the wrong picture.
	return QDir(dir).filePath(SceneMetadataClient::sceneSlug(sceneId) + ".png");
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
void MainWindow::selectSceneById(const QString &key) {
	if (!m_sceneCombo || key.isEmpty()) return;
	// `key` may be a scene's name ("cornell-box") as well as its id ("A1"); the rest works with the id.
	const QString resolvedId = SceneMetadataClient::sceneIdForKey(key);
	const QString id = resolvedId.isEmpty() ? key : resolvedId;

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
	// Both setCurrentIndex() calls above ran signal-blocked, so neither tab
	// bar's own currentChanged handler (mainwindow_tabs.cpp's
	// createSettingsTab()) fired to refresh this - do it explicitly, for
	// both the early "scene not found" return below and the normal exit.
	updateGenerateThumbnailsButtonState();
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
	for (const char *categoryName : SceneCategories::kDisplayOrder) {
		const QString category = QString::fromUtf8(categoryName);
		int inCategory = 0;
		for (int j = 0; j < sceneCount; ++j) {
			const QString id = SceneMetadataClient::sceneIdAtIndex(j);
			if (SceneMetadataClient::sceneCategory(id) != category) continue;
			if (SceneMetadataClient::sceneRequiresFiles(id) != requiresFiles) continue;
			++inCategory;
		}
		if (inCategory == 0) continue;

		const int tab = m_sceneCategoryTabs->addTab(SceneMetadataClient::displayCategory(category));
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

	buildSceneGroup(basicTab, layout);

	buildRenderSettingsGroup(basicTab, layout);

	buildVideoGroup(basicTab, layout);

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	buildLiveControlsGroup(basicTab, layout);
#endif

	buildAdvancedParamsGroup(basicTab, layout);

	buildCameraGroup(basicTab, layout);

	buildOutputGroup(basicTab, layout);

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

