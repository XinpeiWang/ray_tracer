// Render Options tab and the Preview tab (and its sub-tab management) -
// split out of mainwindow_tabs.cpp to keep that file to the Settings tab
// (formerly separate Basic/Advanced settings tabs, and since then Video
// Settings too - all three merged into one scrollable tab, see that file's
// own header comment) and its scene-list helpers; see
// mainwindow_tabs_output.cpp for the Progress/Log/Diagnostics tabs.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"
#include "settings_keys.h"

#include "../src/shared/scene_descriptor.h"

#include <cmath>

#include <QTabBar>
#include <QStatusBar>
#include "scene_builder_widget.h"
#include "scene_metadata_client.h"
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
#include "realtime_preview_session.h"
#endif
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
#include <QSettings>
#include <cmath>
#include <algorithm>

// ============================================================================
// Render Options Tab
// ============================================================================
// Exposes CLI flags RenderController::start() (mainwindow.cpp) already knows
// how to emit but that no earlier tab surfaced: --sampler, --spectral,
// --exposure, --tonemap, --stats, --denoise, --optix-validate. Deliberately
// a separate tab from Settings (resolution/samples/depth/camera, among
// other things) rather than folded into it, since this tab is exclusively
// about render BEHAVIOR flags, not image/camera parameters.
void MainWindow::createRenderOptionsTab() {
	QWidget *optionsTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(optionsTab);
	layout->setSpacing(14);
	layout->setContentsMargins(12, 12, 12, 12);

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	// Same "banner, don't hide" convention as the Settings tab's own
	// m_videoModeWarningLabel/m_liveModeWarningLabel - see their comments
	// (mainwindow.h). Live Preview always renders via the GPU wavefront path
	// tracer directly, so almost nothing on this tab applies to it - except
	// the Denoiser group's own "Live Preview" subsection further down,
	// which is specifically FOR it (and is dimmed/undimmed the same way
	// this banner is shown/hidden).
#ifdef Q_OS_MAC
	m_liveModeOptionsWarningLabel = makeModeWarningBanner(optionsTab,
		tr("⚠ Live Preview uses the Metal progressive path tracer directly - none of the "
		"settings on this tab apply to it, except the \"Live Preview Settings\" group below."));
#else
	m_liveModeOptionsWarningLabel = makeModeWarningBanner(optionsTab,
		tr("⚠ Live Preview uses the GPU progressive path tracer directly - none of the "
		"settings on this tab apply to it, except the Denoiser section's own "
		"\"Live Preview\" subsection below."));
#endif
	layout->addWidget(m_liveModeOptionsWarningLabel);
#endif

	buildIntegratorGroup(optionsTab, layout);
	buildSamplingGroup(optionsTab, layout);
	buildAcceleratorGroup(optionsTab, layout);
	buildPostProcessingGroup(optionsTab, layout);
	buildDenoiserGroup(optionsTab, layout);
	buildCropGroup(optionsTab, layout);
	buildDepthOfFieldGroup(optionsTab, layout);
	buildSeedGroup(optionsTab, layout);
	layout->addStretch();

	// Initial enabled state matches whatever m_renderModeCombo/
	// m_gpuBackendCombo/m_integratorCombo already hold at this point in
	// construction - updateRenderOptionsEnabled() is the single source of
	// truth for this, also called live from each of those combos' own
	// change handlers (mainwindow.cpp/mainwindow_slots.cpp).
	updateRenderOptionsEnabled();

	ThemedScrollArea *scrollArea = new ThemedScrollArea();  // theme motif support - see that class's own comment
	scrollArea->setWidget(optionsTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea->setObjectName("tabScroll");
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, tr("Render Options"));
}

// The Scene Builder tab: build a scene from shapes and lights, preview it, save it as a .pbrt file (scene_builder_widget.h).
void MainWindow::createSceneBuilderTab() {
	m_sceneBuilder = new SceneBuilderWidget(this);
	connect(m_sceneBuilder, &SceneBuilderWidget::statusMessage, this, [this](const QString &text) { statusBar()->showMessage(text, 5000); });
	// "Add to scene list": list the saved file now (the registry grows while the program runs) and select it in the Settings tab.
	connect(m_sceneBuilder, &SceneBuilderWidget::sceneListed, this, [this](const QString &path) {
		SceneMetadataClient::refreshUserScenes();
		const QString id = SceneMetadataClient::sceneIdForFile(path);
		if (id.isEmpty()) {
			statusBar()->showMessage(tr("Saved, but the scene list could not list it until the program is restarted."), 8000);
			return;
		}
		selectSceneById(id);
		statusBar()->showMessage(tr("Added to the scene list as %1 (Settings tab).").arg(id), 8000);
	});
	m_tabWidget->addTab(m_sceneBuilder, tr("Scene Builder"));
}

// Single source of truth for m_samplerCombo/m_lightSamplerCombo/
// m_spectralCheck/m_timeLimitCheck/m_timeLimitSpin/m_exposureSpin/
// m_tonemapCombo/m_statsCheck/m_denoiseCheck/m_denoiseBlendSpin/m_optixValidateCheck/
// m_seedCheck/m_gpuBackendCombo/m_acceleratorCombo/m_splitMethodCombo's
// enabled state, replacing what used to be a
// hand-duplicated 2-input (GPU/CPU x wavefront/recursive) condition
// copy-pasted at construction and in two separate connect() lambdas -
// adding Integrator as a third input would have made that a 3-site
// hand-copy of an increasingly complex condition, so it's factored here
// instead and called from all four controls' own change handlers
// (construction, m_renderModeCombo's lambda, m_gpuBackendCombo's lambda,
// onIntegratorChanged() - mainwindow.cpp/mainwindow_slots.cpp).
void MainWindow::updateRenderOptionsEnabled() {
	const bool gpuSelected = m_renderModeCombo->currentData().toBool();
	// kGpuOptionAvailable (mainwindow.h) is true only on a Windows/OptiX
	// build - GPU is never offered at all on Linux (no GPU renderer exists
	// there), so "GPU selected AND not an OptiX build" can only mean the
	// macOS/Metal backend. Several RenderOptions fields below are real,
	// working features under OptiX but documented no-ops under Metal
	// (gpu/metal/metal_interface.h's own field-by-field comment) - this
	// pair lets each control below gate on the BACKEND that's actually
	// going to run, not just "is GPU selected" the way this function used
	// to (a real, previously-unnoticed bug: several no-op-under-Metal
	// controls stayed clickable whenever GPU was selected on macOS, silently
	// letting a Mac user toggle a setting that render_options.h's own
	// RenderOptions get passed into metal_render_main() unchanged with no
	// effect at all - found via a direct audit, not assumed).
	const bool gpuIsOptix = gpuSelected && kGpuOptionAvailable;
	const bool gpuIsMetal = gpuSelected && !kGpuOptionAvailable;
	const auto integrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	const bool isDefault = (integrator == IntegratorMode::Default);

	if (m_gpuBackendCombo) m_gpuBackendCombo->setEnabled(isDefault && gpuSelected);
	m_samplerCombo->setEnabled(isDefault && !gpuSelected);
	m_lightSamplerCombo->setEnabled(isDefault && !gpuSelected);
	m_spectralCheck->setEnabled(isDefault && !gpuSelected);
	m_adaptiveSamplingCheck->setEnabled(isDefault && !gpuSelected);
	m_adaptiveThresholdSpin->setEnabled(isDefault && !gpuSelected && m_adaptiveSamplingCheck->isChecked());
	m_timeLimitCheck->setEnabled(isDefault && !gpuSelected);
	m_timeLimitSpin->setEnabled(isDefault && !gpuSelected && m_timeLimitCheck->isChecked());
	m_exposureSpin->setEnabled(isDefault);
	m_tonemapCombo->setEnabled(isDefault);
	m_statsCheck->setEnabled(isDefault);
	// Both GPU backends have their own real denoiser now (WavefrontPathTracer::
	// denoise(), gpu/optix/wavefront_path_tracer.cpp) - no longer gated on
	// !wavefrontSelected. Metal has no denoiser at all though (not in
	// metal_interface.h's own list of fields it reads) - gated on
	// gpuIsOptix, not the broader gpuSelected, so this stays correctly
	// disabled when the selected GPU is actually Metal.
	m_denoiseCheck->setEnabled(isDefault && gpuIsOptix);
	m_denoiseBlendSpin->setEnabled(isDefault && gpuIsOptix && m_denoiseCheck->isChecked());
	// OptiX-specific by definition (the checkbox's own label says so) -
	// meaningless, and previously left clickable, under Metal.
	m_optixValidateCheck->setEnabled(isDefault && gpuIsOptix);
	// "Both backends" per render_options.h's own comment, but that predates
	// Metal - metal_interface.h's own field list confirms Metal doesn't
	// actually read regularize/max_component_value at all, so each is
	// gated on !gpuIsMetal (CPU or OptiX-GPU, not Metal-GPU) rather than left
	// unconditional. (seed and crop are NOT in this list - Metal reads both
	// now, see gpu/metal/metal_interface.h, so they stay enabled below.)
	m_regularizeCheck->setEnabled(isDefault && !gpuIsMetal);
	m_maxComponentValueCheck->setEnabled(isDefault && !gpuIsMetal);
	m_maxComponentValueSpin->setEnabled(isDefault && !gpuIsMetal && m_maxComponentValueCheck->isChecked());
	m_cropCheck->setEnabled(isDefault);
	const bool cropSpinsEnabled = isDefault && m_cropCheck->isChecked();
	m_cropX0Spin->setEnabled(cropSpinsEnabled);
	m_cropY0Spin->setEnabled(cropSpinsEnabled);
	m_cropX1Spin->setEnabled(cropSpinsEnabled);
	m_cropY1Spin->setEnabled(cropSpinsEnabled);
	m_seedCheck->setEnabled(isDefault);
	m_seedSpin->setEnabled(isDefault && m_seedCheck->isChecked());
	// Unlike every field above, NOT gated on isDefault - accelerator/
	// splitmethod affect scene construction, shared by every integrator
	// (default path tracer, BDPT/MLT, SPPM), not one integrator's own logic.
	// Still CPU-only (GPU always builds its own fixed BVH) - correctly
	// gated on the broad gpuSelected already, no gpuIsMetal split needed
	// since this one's disabled for EITHER GPU backend alike.
	m_acceleratorCombo->setEnabled(!gpuSelected);
	const bool splitMethodMeaningful =
		!gpuSelected && m_acceleratorCombo->currentData().toString() != QStringLiteral("kdtree");
	m_splitMethodCombo->setEnabled(splitMethodMeaningful);
	// m_dofOverrideCheck/m_apertureSpin/m_focusDistanceSpin are likewise NOT
	// gated on isDefault here, same reasoning as accelerator/splitmethod
	// just above: depth-of-field lens sampling lives in camera ray
	// generation itself (camera.h's get_ray()/GPU's wf_generate_primary_ray),
	// shared unconditionally by every integrator (AO, SimplePath, BDPT/MLT,
	// SPPM, the default path tracer), not gated behind any one integrator's
	// own logic the way maxComponentValue/crop/seed above are. IS gated on
	// !gpuIsMetal though (added, previously unconditional) - the override is
	// only wired for gpu/optix/scene_builder.cpp's own pbrt-camera branch,
	// a real OptiX-only feature (CPU has its own separate, always-available
	// DOF support), not one metal_interface.h lists Metal as reading at all.
	m_dofOverrideCheck->setEnabled(!gpuIsMetal);
	const bool dofSpinsEnabled = !gpuIsMetal && m_dofOverrideCheck->isChecked();
	m_apertureSpin->setEnabled(dofSpinsEnabled);
	m_focusDistanceSpin->setEnabled(dofSpinsEnabled);
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
		stopLivePreviewIfNavigatedAway();
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

	// Selected scene's description - same text/source as the Settings
	// tab's #sceneInfo box (see onSceneChanged()), kept in sync with the
	// scene combo rather than tied to a completed render, so it's already
	// showing what you're about to render before the first click.
	m_previewSceneDescLabel = new QLabel(sidebar);
	m_previewSceneDescLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewSceneDescLabel->setWordWrap(true);
	m_previewSceneDescLabel->setObjectName("previewSceneDesc");
	sideLayout->addWidget(m_previewSceneDescLabel);

	// The active render's own technique note (what the scene demonstrates
	// and why it looks the way it does, plus the render's own technique/
	// settings summary) - see updatePreviewSidebarForActiveTab(), which
	// populates this from the tab's "sceneId"/"techniqueHtml" properties
	// and hides the scroll wrapper when there's nothing to show at all.
	// Scrollable (rather than a plain QLabel like previewInfo/
	// previewSceneDesc above) since the combined content can run long -
	// stretch factor 1 so it absorbs whatever vertical space is left in
	// the sidebar instead of the whole sidebar growing past the visible
	// area and pushing the buttons below out of view.
	m_previewTechniqueLabel = new QLabel();
	m_previewTechniqueLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewTechniqueLabel->setWordWrap(true);
	m_previewTechniqueLabel->setObjectName("previewTechniqueNote");

	m_previewTechniqueScroll = new QScrollArea(sidebar);
	m_previewTechniqueScroll->setWidget(m_previewTechniqueLabel);
	m_previewTechniqueScroll->setWidgetResizable(true);
	m_previewTechniqueScroll->setFrameShape(QFrame::NoFrame);
	m_previewTechniqueScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_previewTechniqueScroll->setObjectName("previewTechniqueScroll");
	m_previewTechniqueScroll->setVisible(false);
	sideLayout->addWidget(m_previewTechniqueScroll, /*stretch=*/1);

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

	// No trailing addStretch() here - m_previewTechniqueScroll's own
	// stretch factor (set above) already absorbs whatever vertical space
	// is left when it's visible; when it's hidden (nothing to show), the
	// buttons simply sit right after the info/scene-desc labels instead
	// of being pushed toward the bottom of an otherwise-empty sidebar.
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

	// Stop the running live preview (rather than let it keep burning GPU
	// cycles unseen) whenever the live sub-tab stops being visibly active -
	// leaving the Preview tab entirely fires here; leaving the live
	// sub-tab for a different one while staying on Preview fires via the
	// m_previewSubTabs::currentChanged connection just above instead. A
	// no-op on a non-GPU build - see stopLivePreviewIfNavigatedAway()'s own
	// comment (mainwindow.h).
	connect(m_tabWidget, &QTabWidget::currentChanged, this, &MainWindow::stopLivePreviewIfNavigatedAway);
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
	if (m_previewTechniqueLabel && m_previewTechniqueScroll) {
		const QString sceneId = currentPreviewProperty("sceneId");
		// Scene note first: it's what actually answers "what's special
		// about this image / why does it look this way", so it leads:
		// the generic integrator/settings text below is useful background,
		// not the headline. techniqueHtml is precomputed once at
		// tab-creation time (see PreviewTechniqueInfo/
		// MainWindow::renderTechniqueHtml()) since a completed render's
		// own settings never change; the scene note stays a live lookup
		// instead, since scene_technique_notes.h is the single source of
		// truth for that text.
		QString html;
		if (!sceneId.isEmpty() && scene_technique_notes::hasNote(sceneId)) {
			html = tr("<b>Why it looks this way</b><br>%1")
						.arg(plainTextToHtmlParagraphs(scene_technique_notes::forScene(sceneId)));
		}
		const QString techniqueHtml = currentPreviewProperty("techniqueHtml");
		if (!techniqueHtml.isEmpty()) {
			if (!html.isEmpty()) html += "<br><br>";
			html += techniqueHtml;
		}
		m_previewTechniqueScroll->setVisible(!html.isEmpty());
		if (!html.isEmpty()) {
			m_previewTechniqueLabel->setText(html);
			// QScrollArea only clamps an out-of-range scroll value to the
			// new content's height on relayout, it doesn't zero it - without
			// this, switching from a long-scrolled tab to a shorter one
			// would open already scrolled toward the bottom instead of at
			// the top.
			m_previewTechniqueScroll->verticalScrollBar()->setValue(0);
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
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	if (page == m_livePreviewPage) {
		// Stop BEFORE clearing the tracking pointers below - stopLivePreview()
		// itself dereferences m_livePreviewStatusLabel, so it must run while
		// that pointer is still valid rather than relying on removeTab()'s
		// currentChanged -> stopLivePreviewIfNavigatedAway() to catch this
		// indirectly (which would fire only after the pointers were already null).
		stopLivePreview();
		m_livePreviewPage = nullptr;
		m_livePreviewLabel = nullptr;
		m_livePreviewStatusLabel = nullptr;
	}
#endif
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

// Trailing three lines shared by every addXPreviewTab() - the page's own
// contents (image label / video player / orbit label) genuinely differ per
// caller and stay in each function, but registering the finished page and
// making it current is identical for all three.
void MainWindow::addPreviewSubTabPage(QWidget *page, const QString &title, const QString &tooltip) {
	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);
}

void MainWindow::addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
									 const QString &infoText, const QString &outputPath, const QString &previewPath,
									 const PreviewTechniqueInfo &technique) {
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
	page->setProperty("sceneId", technique.sceneId);
	page->setProperty("techniqueHtml", technique.techniqueHtml);

	addPreviewSubTabPage(page, title, tooltip);
}

void MainWindow::addVideoPreviewTab(const QString &title, const QString &tooltip,
									 const QString &videoPath, const QString &infoText,
									 const PreviewTechniqueInfo &technique) {
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
	page->setProperty("sceneId", technique.sceneId);
	page->setProperty("techniqueHtml", technique.techniqueHtml);

	addPreviewSubTabPage(page, title, tooltip);

	player->setSource(QUrl::fromLocalFile(videoPath));
	player->play();
}
