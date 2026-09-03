#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QComboBox>
#include <QAbstractSpinBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QAction>
#include <QGroupBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QProcess>
#include <QVector3D>
#include <QTextEdit>
#include <QEvent>
#include <QMouseEvent>
#include <QTimer>
#include <QDateTime>
#include <QElapsedTimer>
#include <QPixmap>
#include <QResizeEvent>
#include <QSystemTrayIcon>
#include <QStackedWidget>
#include <QSlider>
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QVideoWidget>
#include <QMap>
#include <QStylePainter>
#include <QStyleOptionTab>
#include <QHBoxLayout>
#include <QSplitter>
#include <QSignalBlocker>
#include <QQueue>
#include <QListWidget>
#include <QToolButton>
#include <QScrollArea>
#include <QDebug>
#include <QFile>
#include <functional>

#include "camera_math.h"
#include "render_output_parser.h"
#include "theme.h"
#include "icon_tint.h"

#include <QGraphicsDropShadowEffect>
#include <QGraphicsOpacityEffect>
#include <QPropertyAnimation>
#include <QEasingCurve>

// Small standalone UI helper widgets, and render-job driver classes/data
// structs - see each file's own header comment for why they're split out.
#include "mainwindow_widgets.h"
#include "mainwindow_jobtypes.h"


// ============================================================================
// ExpandingTabBar / ExpandingTabWidget
// ============================================================================
// Makes the main tab strip's 8 tabs fill the window's full width instead of
// sitting left-aligned with blank space to the right of "Diagnostics" once
// the window is wider than the tabs' own natural size.
//
// QTabBar::setExpanding(true) - the documented-sounding way to do this - does
// NOT do this: confirmed empirically (a standalone repro, logging every
// tabSizeHint() call) that it only equalizes tab widths when tabs must
// SHRINK to fit an overflowing bar, never grows them to fill idle space.
// QTabWidget also never stretches its tab bar to the widget's own width in
// the first place - the bar only ever claims its own sizeHint (the sum of
// its tabs' natural widths), which is the real reason the strip left-aligns.
//
// The fix is a tabSizeHint() override that hands out width()/count() per
// tab whenever that's wider than the tab's natural hint - but width() itself
// is circular (the bar's width is DERIVED from summing these same hints), so
// naively reading width() here just converges to some in-between value, not
// the full window width (also confirmed empirically). Reading
// parentWidget()->width() instead - the QTabWidget itself, stable and set
// independently of the tab bar's own size - breaks that circularity.
class ExpandingTabBar : public QTabBar {
public:
	explicit ExpandingTabBar(QWidget *parent = nullptr) : QTabBar(parent) {}

protected:
	QSize tabSizeHint(int index) const override {
		QSize hint = QTabBar::tabSizeHint(index);
		const int n = count();
		if (n > 0 && parentWidget()) {
			// The "- n * 4" isn't cosmetic: each tab's own QSS margin/border
			// (mainwindow_style.cpp's QTabBar::tab rule) adds a few pixels
			// this per-tab hint doesn't otherwise account for. Without this
			// slack, the summed hints land a handful of pixels OVER the
			// parent's actual width, tipping the whole bar into scroll-arrow
			// mode - which then reserves its own space for the arrows and
			// never revisits this hint, so only the first few (oversized)
			// tabs end up visible at all. A few pixels of unused margin at
			// the right edge is a far smaller cost than that cliff.
			const int evenWidth = (parentWidget()->width() - n * 4) / n;
			if (evenWidth > hint.width()) hint.setWidth(evenWidth);
		}
		return hint;
	}
};

class ExpandingTabWidget : public QTabWidget {
public:
	explicit ExpandingTabWidget(QWidget *parent = nullptr) : QTabWidget(parent) {
		setTabBar(new ExpandingTabBar(this));
	}
};

// ============================================================================
// MainWindow
// ============================================================================
// Main GUI window with tabbed interface for render controls
// Basic Tab: Quick presets and render mode selection
// Advanced Tab: Detailed controls including camera position presets
// ============================================================================
class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	// startupLanguageCode: main.cpp already has to call loadSavedLanguageCode()
	// itself, before any MainWindow exists, to decide which QTranslator to
	// install - passing that same value through here instead of reading it a
	// second time inside the constructor avoids a second QSettings round trip
	// for one value on the startup path. Left empty, the constructor reads it
	// itself (keeps this callable the old way if ever needed).
	explicit MainWindow(QWidget *parent = nullptr, const QString &startupLanguageCode = QString());
	~MainWindow();

	// Called from main.cpp, before any MainWindow exists, to decide which
	// QTranslator (if any) to install - see language_switch.cpp's own
	// comment for why language switching is restart-to-apply.
	static QString loadSavedLanguageCode();

private slots:
	void onRenderClicked();
	void onStopClicked();
	void onQualityPresetChanged(int index);
	void onCameraPresetChanged(int index);  // Updates camera spinboxes when preset changes
	void onVideoPresetChanged(int index);   // Points scene/camera-path/frames/fps/speed at a named preset
	void onCameraDistanceChanged(double distance);  // Repositions camera X/Y/Z along its current direction from lookat
	void onSceneChanged(int index);         // Updates UI when scene selection changes
	void onModeChanged(int index);          // Switches between Image and Video modes
	void onIntegratorChanged(int index);    // Switches the Integrator Options stack page, auto-corrects GPU/CPU, updates gating
	void onProgressUpdate(int percentage);
	void onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void onLogMessage(const QString &message);
	void onElapsedTick();        // fires every second during render to update status label
	void onRemoveSelectedQueueItem();  // Removes the currently-selected row from m_renderQueue
	void onClearQueue();               // Empties m_renderQueue entirely
	void onRunDiagnosticsClicked();
	void onDiagnosticsReportReady(const QString &report);
	void onDiagnosticsFailed(const QString &message);
	void onGenerateThumbnailsClicked();
	void onThumbnailReady(const QString &sceneId, bool success, const QString &outputPath);
	void onThumbnailsAllDone();

	// Shared by the log tab's buttons and the File menu's actions.
	void copyLogToClipboard();
	void saveLogToFile();
	void clearLog();
	void showAboutDialog();

	// Shared by the Diagnostics tab's buttons.
	void copyDiagToClipboard();
	void saveDiagReportToFile();

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void createRenderOptionsTab();
	// Single source of truth for m_samplerCombo/m_lightSamplerCombo/
	// m_spectralCheck/m_exposureSpin/m_tonemapCombo/m_statsCheck/
	// m_regularizeCheck/m_maxComponentValueCheck/m_cropCheck/m_denoiseCheck/
	// m_optixValidateCheck/m_gpuBackendCombo's enabled state - depends on
	// m_renderModeCombo (GPU/CPU), m_gpuBackendCombo (recursive/wavefront),
	// and m_integratorCombo (Default vs an alternate integrator), so it's
	// called from all of those controls' own change handlers instead of
	// hand-duplicating the same condition at each call site.
	void updateRenderOptionsEnabled();
	void createVideoTab();
	void createPreviewTab();
	void createProgressTab();
	void createLogTab();
	void createDiagnosticsTab();

	// ------------------------------------------------------------------
	// Action layer
	// ------------------------------------------------------------------
	// Every command is a QAction first. A QAction carries its shortcut,
	// tooltip and status tip in one object, so the menu entry, the keyboard
	// shortcut and the status-bar hint can't drift out of sync the way three
	// separate hand-wired copies would. Qt Creator, OBS, Kate and Wireshark
	// are all built this way.
	void createActions();
	void createMenus();
	void createStatusBar();
	// Enables/disables the render-related actions to match m_isRendering.
	void updateActionStates();

	QAction *m_actRender = nullptr;
	QAction *m_actRenderVideo = nullptr;
	QAction *m_actStop = nullptr;
	QAction *m_actOpenFolder = nullptr;
	QAction *m_actOpenViewer = nullptr;
	QAction *m_actCopyLog = nullptr;
	QAction *m_actSaveLog = nullptr;
	QAction *m_actClearLog = nullptr;
	QAction *m_actAbout = nullptr;
	QAction *m_actAboutQt = nullptr;
	QAction *m_actQuit = nullptr;

	// Status bar: permanent widgets carry ambient state that would otherwise
	// have no home (which renderer, what size/quality). Deliberately NOT the
	// progress bar - that stays large and central where it belongs.
	QLabel *m_statusDevice = nullptr;
	QLabel *m_statusSettings = nullptr;

	// The scheme currently in force. Widgets that are styled individually
	// (the log pane, preview, scene info panel) re-read this when it changes.
	theme::Palette m_activeTheme;
	QVector<QAction *> m_themeActions;
	void restyleThemedWidgets();
	void refreshStatusBarInfo();
	// Applies a colour scheme to the whole app. Safe to call repeatedly: the
	// theme menu re-invokes it to switch live rather than asking for a restart,
	// and applyFont() below re-invokes it too - the stylesheet it builds reads
	// its font-size rules from m_activeFontId, so a font change needs the same
	// full rebuild-and-reapply a theme change does, not a cheaper substitute.
	void applyTheme(const theme::Palette &palette);
	void switchTheme(const QString &themeId);
	void createThemeMenu();
	QString loadSavedThemeId() const;
	void saveThemeId(const QString &themeId) const;

	// Recent Renders persistence (recent_renders.cpp) - same
	// QSettings(settings_keys::kOrg, settings_keys::kApp) location as the
	// theme/font/language prefs above, its own group
	// (settings_keys::kRecentRendersGroup) since this is the first
	// list-shaped value this app persists. saveRecentRender() is called
	// from onRenderComplete()'s image branch and
	// assembleVideoAutomatically() (mainwindow_slots.cpp), right after
	// their existing addImagePreviewTab()/addVideoPreviewTab() calls.
	// loadRecentRenders() filters out entries whose files no longer exist
	// on disk (QFileInfo::exists()) - a correctness step, not polish,
	// since files may have moved/been deleted between sessions.
	// displayTitleOverride: assembleVideoAutomatically() computes its own
	// tab title (named preset, or "<scene> (Video)") distinct from
	// job.displayTitle - passed through here so a reopened entry shows the
	// same title its live tab did, rather than the plain scene name.
	// Defaults to job.displayTitle when empty (the image call site's case).
	void saveRecentRender(const RenderJob &job, const QString &previewPath, bool isVideo,
	                       const QString &displayTitleOverride = QString()) const;
	QList<RecentRenderEntry> loadRecentRenders() const;
	QString describeRecentRenderEntry(const RecentRenderEntry &entry) const;
	// Rebuilds m_recentRendersList (creating it on first use if this is the
	// very first entry the app has ever recorded) from a fresh
	// loadRecentRenders() call. Called after every saveRecentRender() and
	// after closePreviewSubTab(), so a render completed - or a tab closed -
	// earlier this session shows up without needing an app restart.
	void refreshRecentRendersList();

	// Language selection and persistence - same shape as the theme menu just
	// above, except switching still needs a fresh process under the hood
	// (see language_switch.cpp's own comment for why: no retranslateUi()
	// split exists between building this app's widget tree and setting its
	// text, unlike theme colour which is a re-derivable property) -
	// switchLanguage() now does that relaunch itself instead of asking the
	// user to do it. main.cpp reads loadSavedLanguageCode() and installs
	// the matching QTranslator before MainWindow is ever constructed, on
	// both a genuine cold start and the relaunched process alike.
	QVector<QAction *> m_languageActions;
	// The relaunch a language switch triggers is destructive to an in-progress
	// render/queue and to a launch that fails to spawn - see switchLanguage()'s
	// own comment. m_languageSwitchPending guards against a second click
	// scheduling a second relaunch while the first is still in its delay window.
	bool m_languageSwitchPending = false;
	// Read once at startup (main.cpp already reads it before MainWindow exists,
	// to decide which QTranslator to install) and reused here so
	// createLanguageMenu()'s initial checkmark doesn't cost a second QSettings
	// round trip for the same value.
	QString m_startupLanguageCode;
	void switchLanguage(const QString &code);
	void createLanguageMenu();
	static void saveLanguageCode(const QString &code);

	// Font selection and persistence - live, like the theme menu (see
	// font_switch.cpp's own comment). Decoupled from theme in the sense that
	// switching one never resets the other's saved choice or menu checkmark -
	// but applyTheme()'s stylesheet does read the active font's base point
	// size (m_activeFontId) to scale its own font-size rules, so a font
	// switch still goes through applyTheme() to make that visible everywhere.
	QVector<QAction *> m_fontActions;
	// Kept in sync by applyFont(); read by applyTheme() so the stylesheet's
	// font-size tokens scale with whichever font is actually active, not a
	// hardcoded baseline. Defaults to the same choice loadSavedFontId() does.
	QString m_activeFontId = QStringLiteral("cyberpunk");
	// Read once at startup and reused by createFontMenu(), same reasoning as
	// m_startupLanguageCode above.
	QString m_startupFontId;
	void applyFont(const QString &id);
	void switchFont(const QString &id);
	void createFontMenu();
	static QString loadSavedFontId();
	static void saveFontId(const QString &id);
	// Base point size for a FontChoice id, defined in font_switch.cpp;
	// exposed here so applyTheme() (mainwindow_style.cpp) can scale its own
	// font-size rules to the active choice without reaching into font_switch.cpp's
	// anonymous namespace. Same "unknown id falls back to the default choice"
	// rule fontChoiceById() uses internally.
	static int fontPointSizeForId(const QString &id);

	// Shared by createThemeMenu()/createFontMenu()/createLanguageMenu(): checks
	// the one action in `actions` whose stored data matches `activeValue`,
	// unchecking the rest. All three menus are an exclusive QActionGroup of
	// QActions carrying their choice's id/code as setData(), so this one loop
	// replaces three copies of the same four lines.
	static void syncCheckedAction(const QVector<QAction *> &actions, const QString &activeValue);

	void styleComboBox(QComboBox *combo);
	void applyComboPopupPalette(QComboBox *combo);
	void styleSpinBox(QAbstractSpinBox *spinBox);
	void styleGroupBox(QGroupBox *box);
	void styleCheckBox(QCheckBox *box);
	// Beginner-facing "(i)" info marks - see their own doc comments
	// (mainwindow_style.cpp) for the full design rationale.
	QToolButton* createInfoIcon(const QString &helpText);
	QWidget* labelWithInfo(const QString &labelText, const QString &helpText);
	QWidget* checkboxWithInfo(QCheckBox *checkBox, const QString &helpText);
	// Escapes and wraps plain, blank-line-separated paragraphs into `<p>`
	// tags - the shared core of wrapTooltipHtml() below and the Preview
	// tab's technique box (renderTechniqueHtml()/updatePreviewSidebarForActiveTab(),
	// mainwindow_tabs.cpp), which need the same paragraph handling without
	// the fixed tooltip-width body wrapper.
	QString plainTextToHtmlParagraphs(const QString &plainText);
	// Wraps plain, blank-line-separated paragraphs into width-constrained HTML
	// for a rich-text tooltip - see its own definition (mainwindow_style.cpp)
	// for the full rationale. Shared rather than file-local so both the
	// standalone scene-tech info icon (updateSceneTechInfoIcon() below) and
	// the scene combo/grid's per-row tooltips (mainwindow_tabs.cpp) format
	// scene_technique_notes.h's text the same way.
	QString wrapTooltipHtml(const QString &plainText);
	// Builds the wrapTooltipHtml()-formatted technique-note tooltip for
	// `sceneId` - the one place that decides how to combine an id/name
	// heading with scene_technique_notes::forScene()'s text, shared by the
	// 3 call sites that used to each assemble this independently
	// (mainwindow_tabs.cpp's populateSceneCombo()/populateSceneGrid(), and
	// updateSceneTechInfoIcon() below). includeHeading=false when the id/name
	// is already visible right next to the tooltip's own widget (the combo
	// row's own text, the standalone info icon's neighboring scene picker);
	// true when it isn't (the grid's tiles only show the scene's name, not
	// its id, as their own label).
	QString sceneTooltipHtml(const QString &sceneId, bool includeHeading);
	// Rewrites m_sceneTechInfoIcon's tooltip for `sceneId` (see
	// scene_technique_notes.h) - called from refreshSceneInfoLabel().
	void updateSceneTechInfoIcon(const QString &sceneId);
	// Shows/hides m_sceneRecommendedSettingsHint for `sceneId` - visible iff
	// the loaded scene's own recommended Sampler/Integrator/light-sampler
	// (SceneMetadataClient::sceneRecommended*(), sourced from
	// SceneDescriptor::recommended_* in scene_registry.h) differs from what
	// m_samplerCombo/m_integratorCombo/m_lightSamplerCombo currently have
	// selected. Mirrors cpu_render_main()'s own mismatch check
	// (cpu_interface.cpp) exactly - same "only when the default/no-flag
	// value is in effect" scope, same "volpath"/"sobol"/"bvh" skip values -
	// so the GUI never claims a mismatch the CLI itself wouldn't warn about.
	// Called from onSceneChanged() AND from each of those three combos' own
	// change handlers, so switching Sampler/Integrator/Light Sampler after
	// picking the scene updates the hint live instead of leaving it stale.
	void updateSceneRecommendedSettingsHint(const QString &sceneId);
	// m_applyRecommendedSettingsButton's slot - sets m_integratorCombo/
	// m_samplerCombo/m_lightSamplerCombo to the CURRENT scene's recommended
	// values (SceneMetadataClient::sceneRecommended*()), skipping any one
	// of the three that's empty or doesn't match a real combo entry (e.g.
	// an Integrator directive string this GUI's IntegratorMode enum has no
	// mapping for - see this function's own definition, mainwindow_slots.cpp).
	// The one deliberate exception to this file's "not auto-applied,
	// warn rather than switch" rule (see updateSceneRecommendedSettingsHint's
	// own comment) - here the user explicitly clicked Apply, so it's a real
	// action, not an automatic override of an unrelated selection.
	void applyRecommendedSettings();
	// Plain-text description of `mode` - used by createBasicTab()'s
	// per-item combo tooltips (each Integrator dropdown row's own "(i)").
	QString integratorDescription(IntegratorMode mode);
	// " · "-joined list of only the AdvancedRenderFlags/IntegratorOptions
	// fields that differ from their default - mirrors
	// RenderController::start()'s own emission conditions (mainwindow.cpp
	// ~line 143-211) field-for-field, so a displayed setting never
	// contradicts what was actually passed to the CLI. Keep both in sync
	// if either gains/loses a field. Empty string when everything's
	// default. Used by renderTechniqueHtml() below.
	QString advancedFlagsSummary(const AdvancedRenderFlags &flags);
	QString integratorSettingsSummary(const IntegratorOptions &opts);
	// Combines integratorDescription() with the two settings summaries
	// above into the Preview tab's technique-box HTML (see
	// updatePreviewSidebarForActiveTab(), mainwindow_tabs_render.cpp) - computed
	// once per tab at creation time (see PreviewTechniqueInfo below), not
	// live, since a completed render's own settings never change.
	QString renderTechniqueHtml(const IntegratorOptions &integratorOptions, const AdvancedRenderFlags &advancedFlags);
	// A subtle "elevated card" drop shadow (QSS alone cannot do box-shadow) -
	// neutral black at low alpha rather than theme-tinted, the same choice
	// every real elevation system (Material, Fluent, CSS itself) makes,
	// since a shadow reads as "surface above surface" on any hue, dark or
	// light theme alike, without needing a per-palette shadow colour.
	void applyElevation(QWidget *widget, qreal blurRadius, qreal offsetY, int alpha);
	// Same QGraphicsDropShadowEffect mechanism as applyElevation(), but
	// centred (no offset) and theme-coloured rather than neutral black - a
	// glow of energy around the widget rather than a cast shadow beneath
	// it. Used for the progress bar's own pulse while actively rendering
	// (see startProgressGlow()) - a different visual language deliberately
	// kept separate from applyElevation() rather than folding a colour
	// parameter into that one, since "surface depth" and "energy glow" read
	// as two different things even though the underlying Qt mechanism is
	// identical.
	void applyGlow(QWidget *widget, qreal blurRadius, const QColor &color);
	// Starts/stops the progress bar's pulsing glow - see applyGlow()'s own
	// comment. Called from startRenderJob()/onRenderComplete() so the pulse
	// runs exactly while m_isRendering is true, settling to a fixed glow
	// once finished (the QSS resultState colouring already handles success/
	// error at that point - this only ever controls whether it's animating).
	void startProgressGlow();
	void stopProgressGlow();
	// Smoothly animates m_progressBar's own `value` property (a real
	// QProgressBar Q_PROPERTY) toward `value` instead of snapping - Qt
	// redraws the bar's built-in percentage text from that same live
	// property each animation frame, so this animates the fill AND the
	// number simultaneously for free, no separate text animation needed.
	void animateProgressTo(int value);
	// baseOutputPath is the render's own --output argument (see
	// onRenderClicked()) so the assembled video's expected path
	// ("<stem>_video.mp4") can be derived directly instead of guessing at a
	// shared directory - necessary now that every render's base path is
	// unique. Automatically assembles/locates the video after frames are
	// rendered and adds it as a new Preview sub-tab. `job` is the RenderJob
	// that was actually rendered, captured by value into the caller's
	// deferred QTimer::singleShot rather than read back from m_currentJob -
	// this runs ~500ms after onRenderComplete() returns, by which time a
	// queued next job may already have overwritten it (see m_currentJob's
	// own comment).
	void assembleVideoAutomatically(const QString &baseOutputPath, const RenderJob &job);
	void refreshCameraDistanceDisplay(); // Recomputes m_cameraDistance's shown value from X/Y/Z and m_currentLookat*, without re-triggering onCameraDistanceChanged

	// Preview sub-tabs (see m_previewSubTabs's own comment). Both add*
	// functions build a self-contained tab page, register it, and make it
	// current; tooltip is the full scene/preset description, shown on
	// hover since the tab bar itself only has room for a short title.
	void addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
							 const QString &infoText, const QString &outputPath, const QString &previewPath,
							 const PreviewTechniqueInfo &technique);
	void addVideoPreviewTab(const QString &title, const QString &tooltip, const QString &videoPath,
							 const QString &infoText, const PreviewTechniqueInfo &technique);
	// First use of a title returns it unchanged; each repeat appends " (N)".
	QString uniquePreviewTabTitle(const QString &baseTitle);
	// Reads a property (see m_previewSubTabs's comment) off the currently
	// active sub-tab's page widget; empty string if there is no active tab.
	QString currentPreviewProperty(const char *name) const;
	// Refreshes m_previewInfoLabel and the Open Folder/Viewer actions from
	// whichever sub-tab just became active - connected to m_previewSubTabs's
	// currentChanged signal.
	void updatePreviewSidebarForActiveTab();
	// Removes and deletes a Preview sub-tab. Called from each tab's own
	// left-side close button (see addImagePreviewTab()/addVideoPreviewTab())
	// rather than QTabWidget::tabCloseRequested, since Qt's own auto-managed
	// close button is turned off entirely - see createPreviewTab()'s comment
	// on why.
	void closePreviewSubTab(int index);

	// Camera arithmetic lives in camera_math.h (Qt-free, unit tested); these
	// just read the current values out of the widgets for it.
	camera_math::Vec3 currentCameraPosition() const;
	camera_math::Vec3 currentLookAt() const;

	// UI Components
	QTabWidget *m_tabWidget;

	// Mode selector (Image vs Video)
	QComboBox *m_modeCombo;             // Render mode: Image or Video

	// Basic Tab
	QComboBox *m_renderModeCombo;       // GPU vs CPU selection
	QComboBox *m_gpuBackendCombo;       // Recursive vs wavefront GPU path tracer (only meaningful under GPU)
	QComboBox *m_qualityPresetCombo;    // Quality preset dropdown
	QComboBox *m_resolutionCombo;       // Resolution preset dropdown
	QLineEdit *m_outputPathEdit;        // Output file path (timestamped by default)
	QPushButton *m_browseButton;
	QPushButton *m_renderButton;
	QPushButton *m_stopButton;
	QProgressBar *m_progressBar;
	// See applyGlow()/startProgressGlow()'s own comments. Both lazily
	// created on first use, not at construction - the progress bar's
	// QGraphicsDropShadowEffect only needs to exist once rendering has
	// actually started once.
	QPropertyAnimation *m_progressGlowAnim = nullptr;
	QPropertyAnimation *m_progressValueAnim = nullptr;
	QLabel *m_statusLabel;
	// A secondary caveat line below m_statusLabel (e.g. "render succeeded but
	// the preview image couldn't be shown") - kept separate rather than
	// appended into m_statusLabel's own text so it can carry the theme's
	// warning colour without needing m_statusLabel's other ~15 call sites
	// (success/failure/progress messages) to switch to rich text and start
	// HTML-escaping arbitrary renderer output. Hidden whenever there is
	// nothing to caveat - see setStatusWarning()/clearStatusWarning().
	QLabel *m_statusWarningLabel;
	QLabel *m_currentJobLabel;          // Which job is actually rendering - see startRenderJob()/describeRenderJob()
	int m_progressTabIndex = -1;        // Index of the Progress tab within m_tabWidget (see createProgressTab())
	int m_videoTabIndex = -1;           // Index of the Video Settings tab within m_tabWidget (see createVideoTab()/onModeChanged())

	// Advanced Tab - Manual Controls
	QSpinBox *m_widthSpinBox;           // Custom width
	QSpinBox *m_heightSpinBox;          // Custom height
	QSpinBox *m_samplesSpinBox;         // Samples per pixel
	QSpinBox *m_maxDepthSpinBox;        // Max ray depth

	// Render Options tab (createRenderOptionsTab()) - one widget per CLI
	// flag RenderController::start() can emit; see setAdvancedFlags()'s own
	// comment. m_samplerCombo/m_lightSamplerCombo/m_spectralCheck are
	// CPU-default-path-tracer only; m_denoiseCheck/m_optixValidateCheck are GPU-only (both GPU
	// backends have their own real denoiser - see WavefrontPathTracer::
	// denoise()); m_exposureSpin/m_tonemapCombo/
	// m_statsCheck are default-path-tracer-only (inert, not rejected,
	// under any alternate integrator). Enabled state kept in sync with
	// m_renderModeCombo/m_gpuBackendCombo/m_integratorCombo by
	// updateRenderOptionsEnabled() (mainwindow_tabs_render.cpp), the single
	// source of truth for this cross-product - called from all four
	// controls' own change handlers.
	QComboBox *m_samplerCombo;          // --sampler (CPU default path tracer only)
	QComboBox *m_lightSamplerCombo;     // --lightsampler (CPU default path tracer only)
	QCheckBox *m_spectralCheck;         // --spectral (CPU default path tracer only)
	QDoubleSpinBox *m_exposureSpin;     // --exposure (default path tracer only)
	QComboBox *m_tonemapCombo;          // --tonemap (default path tracer only)
	QCheckBox *m_statsCheck;            // --stats (default path tracer only)
	QCheckBox *m_denoiseCheck;          // --denoise (GPU only, both backends)
	QCheckBox *m_optixValidateCheck;    // --optix-validate (GPU only)
	QCheckBox *m_regularizeCheck;       // --regularize (default path tracer only, both backends)
	// --maxcomponentvalue (CPU default path tracer only) - spinbox only
	// enabled/emitted when the checkbox is checked, since the CLI's own
	// "1e9 = not requested" sentinel would be a confusing default value to
	// show in a spinbox (see m_maxComponentValueSpin's own construction
	// comment, mainwindow_tabs_render.cpp).
	QCheckBox *m_maxComponentValueCheck;
	QDoubleSpinBox *m_maxComponentValueSpin;
	// --crop (default path tracer only, both backends) - four fractional
	// [0,1] spinboxes, only enabled/emitted when the checkbox is checked,
	// same "avoid showing a confusing sentinel" reasoning as maxcomponentvalue.
	QCheckBox *m_cropCheck;
	QDoubleSpinBox *m_cropX0Spin;
	QDoubleSpinBox *m_cropY0Spin;
	QDoubleSpinBox *m_cropX1Spin;
	QDoubleSpinBox *m_cropY1Spin;

	// "Integrator" group (createRenderOptionsTab()): the algorithm selector
	// itself, plus sub-flag widgets for the 5 alternate integrators that
	// have any (SPPM/BDPT/MLT/AO/SimplePath); RandomWalk/SimpleVolPath/
	// LightPath/Default share one placeholder page (m_integratorNoOptionsLabel)
	// instead of an empty page each. m_integratorOptionsStack's page index
	// per IntegratorMode is defined in onIntegratorChanged() (mainwindow_slots.cpp).
	//
	// Which alternate integrator (--sppm/--bdpt/--mlt/--randomwalk/--ao/
	// --simplepath/--simplevolpath/--lightpath) to use instead of the
	// default path tracer - see IntegratorMode's own comment and
	// onIntegratorChanged(). All 7 non-SPPM alternates are CPU-only (the
	// CLI just warns and forces CPU under --gpu, never rejects), so
	// onIntegratorChanged() auto-switches m_renderModeCombo to CPU and
	// disables it whenever one of those 7 is picked.
	QComboBox *m_integratorCombo;
	// --video hard-rejects any non-Default integrator (an animated
	// camera-path flythrough and an alternate integrator's own render
	// loop can't be combined) - same class of guaranteed CLI rejection as
	// an animated-camera scene + --video, which this GUI already doesn't
	// block at the click, just lets fail through renderComplete(). This
	// label adds a non-blocking heads-up before that happens, toggled by
	// both onModeChanged() and onIntegratorChanged().
	QLabel *m_integratorVideoWarningLabel;
	// Second copy of the same warning, on Basic Settings directly under
	// m_modeCombo (Output Mode) - that's the control that actually
	// triggers the conflict, but it's on a different tab from
	// m_integratorVideoWarningLabel above (which lives in the Integrator
	// group, Render Options tab, next to the OTHER control that can
	// trigger it). Whichever tab the user is looking at when the conflict
	// is created, they see it there - both kept in sync by the same
	// onModeChanged()/onIntegratorChanged() toggle.
	QLabel *m_integratorVideoWarningLabelBasic;
	// No isEnabled() gating needed on these in captureRenderJob() - only
	// the currently-selected integrator's own fields are ever read by
	// RenderController::start()'s switch, so a stale value from a hidden
	// page is never emitted.
	QGroupBox *m_integratorOptionsGroup;
	// CurrentPageSizedStackedWidget, not a plain QStackedWidget - see that
	// class's own comment (mainwindow_widgets.h): this group's shortest page
	// (the shared placeholder) and tallest (Ambient Occlusion) differ by
	// several form rows, and a plain QStackedWidget always reserves the
	// tallest page's height even while showing the shortest one.
	CurrentPageSizedStackedWidget *m_integratorOptionsStack;
	QLabel *m_integratorNoOptionsLabel; // Placeholder page text, swapped per-mode
	QSpinBox *m_sppmIterationsSpin;
	QSpinBox *m_sppmPhotonsSpin;
	QSpinBox *m_bdptMaxDepthSpin;
	QSpinBox *m_mltBootstrapSpin;
	QSpinBox *m_mltMutationsSpin;
	QSpinBox *m_mltMaxDepthSpin;
	QDoubleSpinBox *m_aoMaxDistSpin;
	QCheckBox *m_aoUniformCheck;
	QDoubleSpinBox *m_aoIllumScaleSpin;
	QDoubleSpinBox *m_aoIllumRSpin;
	QDoubleSpinBox *m_aoIllumGSpin;
	QDoubleSpinBox *m_aoIllumBSpin;
	QCheckBox *m_simplepathNoLightsCheck;
	QCheckBox *m_simplepathNoBsdfCheck;

	// Camera controls
	// Camera position (lookfrom) can be set via presets or custom X/Y/Z
	// values; lookat is fixed per-scene in the renderer (not the same point
	// for every scene - see scene_registry.h's CameraConfig::lookat_x/y/z,
	// queried live via SceneMetadataClient::recommendedCamera() and
	// mirrored into m_currentLookatX/Y/Z below whenever the scene changes).
	QComboBox *m_cameraPresetCombo;     // Preset camera positions (includes "Custom" option)
	QDoubleSpinBox *m_cameraPosX;       // Camera X position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosY;       // Camera Y position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosZ;       // Camera Z position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraDistance;   // Distance from the current scene's look-at point (enabled only for "Custom")
	// Currently selected scene's look-at point (updated in onSceneChanged).
	// Used by onCameraDistanceChanged to reposition the camera along its
	// existing viewing direction from this point, rather than needing to
	// know the scene's geometry to type new X/Y/Z values by hand.
	double m_currentLookatX = 278.0, m_currentLookatY = 278.0, m_currentLookatZ = 278.0;
	// Distance from m_currentLookat* to the CURRENT scene's own recommended
	// camera (updated in onSceneChanged alongside the lookat fields above).
	// m_cameraPresetCombo's items store direction*ratio vectors (see its
	// setup comment in mainwindow_tabs.cpp) that get scaled by this value in
	// onCameraPresetChanged, so a named preset like "Right Wall" lands at a
	// sensible position for whatever scene is active instead of always
	// landing at Cornell Box's own literal (500,278,278), which used to be
	// wildly outside the geometry of any other scene (e.g. scene 1's
	// spheres sit within roughly +-15 units of the origin).
	double m_currentSceneCamDistance = 1078.0;

	// Scene selection
	// Two-level filter: m_sceneAvailabilityTabs splits every scene into
	// "Self-Contained" (renders in a fresh checkout, no extra downloads) vs.
	// "Requires External Files" (SceneMetadataClient::sceneRequiresFiles()),
	// independently of SceneCategories - a category like Basics or Geometry
	// has scenes in both buckets (e.g. Basics' A4 Earth needs an image file,
	// A1-A9 mostly don't), so this is a per-scene split layered on top of the
	// existing letter categories, not a coarser replacement for them.
	// m_sceneCategoryTabs is then rebuilt to show only the letter categories
	// that have at least one scene in whichever availability bucket is
	// selected (rebuildCategoryTabs()) - mirroring how it already skips any
	// category with zero scenes at all.
	QTabBar *m_sceneAvailabilityTabs = nullptr;  // "Self-Contained" / "Requires External Files"
	QTabBar *m_sceneCategoryTabs = nullptr;  // Category filter above the scene dropdown
	// Narrows the current category's own combo list further by name/id/
	// description substring - independent of, and applied on top of, the
	// availability/category tab filters above. Scenes went from 78 to 154
	// (curated pbrt examples added under real topic categories, see
	// scene_registry.h), so a category tab alone can now hold 20+ scenes -
	// exactly the point a flat dropdown starts wanting search.
	QLineEdit *m_sceneSearchBox = nullptr;
	QComboBox *m_sceneCombo;            // Scene selector dropdown, showing one category at a time

	// Grid ("gallery") alternative to m_sceneCombo - same filtered scene set,
	// shown as thumbnail tiles instead of text rows. Toggled against
	// m_sceneCombo via m_sceneViewStack/m_sceneViewToggle; see
	// populateSceneGrid()'s own comment. Thumbnails come from
	// thumbnailCachePath() - a generated PNG if one exists, else a shared
	// placeholder icon.
	QListWidget *m_sceneGrid = nullptr;
	QStackedWidget *m_sceneViewStack = nullptr;   // page 0 = m_sceneCombo, page 1 = m_sceneGrid
	QToolButton *m_sceneViewToggle = nullptr;     // checked = grid page showing
	QPushButton *m_generateThumbnailsButton = nullptr;
	// Lazily created (and reused across multiple "Generate Thumbnails"
	// clicks) by onGenerateThumbnailsClicked() the first time it's needed -
	// see that slot's own comment.
	ThumbnailGenerator *m_thumbnailGenerator = nullptr;

	// Rebuilds m_sceneCategoryTabs' tab set for the given availability filter
	// (true = only scenes with sceneRequiresFiles()==true, false = only
	// scenes with sceneRequiresFiles()==false), skipping any letter category
	// left with zero matching scenes - same skip-if-empty rule
	// createBasicTab() already applies for categories with zero scenes at
	// all. Does not touch m_sceneCombo; callers follow up with
	// populateSceneCombo() for whichever category tab ends up selected.
	void rebuildCategoryTabs(bool requiresFiles);

	// The scene ids matching `category` and m_sceneAvailabilityTabs' current
	// selection, further narrowed by m_sceneSearchBox's text if any (name/id/
	// description substring match) - the shared filter both
	// populateSceneCombo() and populateSceneGrid() build their contents from,
	// so category/availability/search apply identically to both views with
	// no duplicated filter logic.
	QStringList filteredSceneIds(const QString &category) const;

	// Refills m_sceneCombo with just the scenes in `category` that also match
	// m_sceneAvailabilityTabs' current selection. Does NOT emit
	// currentIndexChanged per insertion - callers apply the resulting selection
	// themselves with a single onSceneChanged() call.
	void populateSceneCombo(const QString &category);

	// Refills m_sceneGrid the same way populateSceneCombo() refills
	// m_sceneCombo - same filteredSceneIds(category), one QListWidgetItem per
	// scene (Qt::UserRole holds the scene id, mirroring the combo's item-data
	// convention), icon set to the cached thumbnail if thumbnailCachePath(id)
	// exists on disk, else a shared placeholder built once in
	// createBasicTab(). Does NOT emit itemSelectionChanged - callers behave
	// like populateSceneCombo()'s own callers.
	void populateSceneGrid(const QString &category);

	// Calls populateSceneCombo() and populateSceneGrid() together - every
	// call site that used to call populateSceneCombo() alone now calls this,
	// so both views always agree on what's currently filtered in.
	void populateSceneViews(const QString &category);

	// Absolute path a scene's cached thumbnail PNG would live at, under
	// QStandardPaths::CacheLocation + "/thumbnails" (mirrors theme_load.cpp's
	// own QStandardPaths convention for per-user files) - does not check
	// whether the file actually exists.
	QString thumbnailCachePath(const QString &sceneId) const;

	// Drives all three scene-selection widgets (availability tab, category
	// tab, m_sceneCombo) to the scene matching `id`, as if the user had
	// clicked through to it by hand - switching availability/category tabs
	// first if `id` isn't already visible under the current ones. Used by
	// the video preset combo (see onVideoPresetChanged()) to point the
	// scene picker at a preset's scene without duplicating this tab-
	// cascading logic. No-op (logs a warning) if `id` isn't a real scene id.
	void selectSceneById(const QString &id);
	QLabel *m_sceneInfoLabel;           // Scene description and performance info
	// Unlike every other info icon (fixed tooltip at construction), this
	// one's tooltip is rewritten per selection by refreshSceneInfoLabel() -
	// see scene_technique_notes.h for the per-scene content it reads from.
	QToolButton *m_sceneTechInfoIcon = nullptr;
	// Hidden unless the current scene's own recommended Sampler/Integrator/
	// light-sampler differs from the Render Options tab's current
	// selection - see updateSceneRecommendedSettingsHint()'s own comment.
	QLabel *m_sceneRecommendedSettingsHint = nullptr;
	// Shown/hidden together with m_sceneRecommendedSettingsHint - see
	// applyRecommendedSettings()'s own comment.
	QPushButton *m_applyRecommendedSettingsButton = nullptr;

	// Rebuilds m_sceneInfoLabel's text (description/performance/SPP/GPU-
	// support, plus the requires-files/CPU-only warning badges) for
	// whichever scene m_sceneCombo currently has selected. Split out of
	// onSceneChanged() so restyleThemedWidgets() can call this alone on a
	// theme switch - the warning badges' colours are baked into inline HTML
	// at build time (QSS can't reach them), so without a way to rebuild just
	// the label, an already-shown badge would keep the previous theme's
	// colour until the user reselected a scene. No-op if nothing is selected
	// or the scene's metadata can't be queried.
	void refreshSceneInfoLabel();

	// Video Tab
	QComboBox *m_videoPresetCombo;      // Named scene+path+frames/fps/speed bundle - see video_preset.h
	QComboBox *m_cameraPathCombo;       // Camera animation path selector
	QSpinBox *m_videoFramesSpinBox;     // Number of frames to render
	QSpinBox *m_videoFPSSpinBox;        // Target FPS for video
	QDoubleSpinBox *m_videoSpeedSpinBox; // Camera movement speed multiplier
	QLabel *m_videoInfoLabel;           // Video duration and path info
	// Visible whenever Output Mode (Basic Settings) isn't "Generate Video" -
	// see createVideoTab()'s own comment for why this tab stays enabled and
	// clickable rather than being disabled outright. Toggled by onModeChanged().
	QLabel *m_videoModeWarningLabel = nullptr;

	// Preview tab - each completed render gets its own closable sub-tab
	// (see addImagePreviewTab()/addVideoPreviewTab()) instead of a single
	// shared pane that the next render overwrites, so switching between
	// sub-tabs keeps every past render's image/video around. Each sub-tab
	// page widget carries its own "outputPath"/"previewPath"/"infoText"
	// Qt properties (image mode: a ScaledImageLabel; video mode: its own
	// QVideoWidget/QMediaPlayer/QAudioOutput, all parented to the page so
	// closing the tab tears them down too) - see createPreviewTab() and
	// currentPreviewProperty()/updatePreviewSidebarForActiveTab().
	SplitPreviewTabs *m_previewSubTabs = nullptr;
	// Recent Renders list, inserted into m_previewSubTabs' empty-state
	// prompt (SplitPreviewTabs::addToEmptyState()) - see createPreviewTab(),
	// which builds it via the first refreshRecentRendersList() call. Null
	// until the app has recorded at least one entry (see
	// refreshRecentRendersList()'s own comment).
	QListWidget *m_recentRendersList = nullptr;
	QWidget *m_previewSidebar = nullptr; // Info/buttons pane; hidden while there are no sub-tabs - see updatePreviewSidebarForActiveTab()
	QLabel *m_previewInfoLabel;         // Filename / resolution / size / render time - reflects whichever sub-tab is active
	QLabel *m_previewSceneDescLabel;    // Selected scene's description - see onSceneChanged()
	// The active sub-tab's own technique note (scene_technique_notes::
	// forScene(), keyed by the "sceneId" tab property, plus the render's
	// own technique/settings summary - see renderTechniqueHtml()) - unlike
	// m_previewSceneDescLabel above, tied to the actual render shown, not
	// whichever scene is currently selected in the picker. Wrapped in
	// m_previewTechniqueScroll (below) since combined content can run
	// long; hidden when there's nothing to show at all - see
	// updatePreviewSidebarForActiveTab().
	QLabel *m_previewTechniqueLabel = nullptr;
	// Scrolls m_previewTechniqueLabel internally, sized policy Expanding
	// with a layout stretch factor, so long technique/scene-note text
	// scrolls in its own bounded area instead of growing the whole
	// sidebar and pushing the Open Folder/Viewer buttons out of view.
	QScrollArea *m_previewTechniqueScroll = nullptr;
	int m_previewTabIndex = -1;         // Index of the Preview tab within m_tabWidget
	// Counts repeat sub-tab titles ("Cornell Box" -> "Cornell Box (2)") so
	// re-rendering the same scene/preset in one session doesn't produce
	// indistinguishable tabs - see uniquePreviewTabTitle().
	QMap<QString, int> m_previewTitleCounts;

	// Log output
	QTextEdit *m_logTextEdit;           // Log output display
	int m_logTabIndex = -1;             // Index of the Log Output tab within m_tabWidget

	// Every line the log has shown, kept so a theme change can re-render it.
	// The QTextEdit holds only the RESULT of styling - HTML with one scheme's
	// colours already baked into each span - which cannot be recoloured after
	// the fact. Keeping the inputs (what the line said, and how it classified)
	// is what makes a rebuild possible; the classification is cached rather
	// than recomputed because it is regex work and a long render log has tens
	// of thousands of lines.
	struct LoggedLine {
		QString timestamp;
		QString escaped;
		render_output::LogCategory category;
	};
	QVector<LoggedLine> m_logHistory;
	void rebuildLogPane();

	// Diagnostics
	QTextEdit *m_diagTextEdit;              // Diagnostics report display
	QPushButton *m_runDiagnosticsButton;    // Disabled while a probe is running
	int m_diagnosticsTabIndex = -1;         // Index of the Diagnostics tab within m_tabWidget
	DiagnosticsRunner *m_diagnosticsRunner = nullptr;  // nullptr when not running
	// Raw (unstyled) report text, kept for the same reason m_logHistory is:
	// the pane holds styled HTML with the previous scheme's colours already
	// baked in, so a theme change re-renders from this rather than trying to
	// recolour existing spans in place. Empty when the pane is showing a
	// plain (non-report) message, e.g. "Running diagnostics..." or a failure
	// - see onDiagnosticsReportReady()/onDiagnosticsFailed().
	QString m_lastDiagReport;
	void rebuildDiagPane();

	// Render driver (nullptr when not rendering)
	RenderController *m_renderController;

	// State
	bool m_isRendering;                 // true when a render is in progress
	bool m_videoMode;                   // true = video generation mode, false = single image mode

	// ------------------------------------------------------------------
	// Render queue
	// ------------------------------------------------------------------
	// onRenderClicked() (mainwindow_slots.cpp) always enqueues a captured
	// RenderJob and then calls processQueueIfIdle() - which starts the front
	// job only if nothing is currently running. The everyday single-render
	// case is just "enqueue one job into an empty, immediately-idle queue",
	// so there is no separate "start now" code path to keep in sync with
	// the queued one.
	QQueue<RenderJob> m_renderQueue;
	// The job actually passed to the RenderController currently running (or
	// most recently run). onRenderComplete() reads its videoMode/sceneId/
	// displayTitle from here, not from the live UI widgets - once renders
	// can queue, the user may have already changed the scene/mode combo for
	// the *next* job by the time an earlier one's completion signal arrives.
	RenderJob m_currentJob;
	QGroupBox *m_queueGroup = nullptr;       // Hidden whenever m_renderQueue is empty
	QListWidget *m_queueListWidget = nullptr;
	RenderJob captureRenderJob();             // Snapshots every render field currently in the UI
	void startRenderJob(const RenderJob &job); // Builds a RenderController for `job` and starts it
	void processQueueIfIdle();                // Dequeues and starts the front job if nothing is running
	void refreshQueuePanel();                 // Rebuilds m_queueListWidget from m_renderQueue
	static QString describeRenderJob(const RenderJob &job); // One-line queue-row summary
	// Shared by describeRenderJob(), describeRecentRenderEntry()
	// (recent_renders.cpp), and refreshStatusBarInfo() (mainwindow_actions.cpp) -
	// previously identical switch/ternary logic copy-pasted in all three.
	static QString integratorSuffixTag(IntegratorMode mode); // " · SPPM" etc., blank for Default
	static QString rendererLabel(bool useGPU, bool useWavefront); // "CPU"/"GPU"/"GPU-WF"

	// Elapsed render timer
	QTimer *m_elapsedTimer;             // fires every second during render
	QDateTime m_renderStartTime;        // wall-clock time when render began

	// ------------------------------------------------------------------
	// Progress sampling for the time-remaining estimate
	// ------------------------------------------------------------------
	// Modelled on HandBrake's UpdateState() (libhb/sync.c), which is the
	// most carefully-built ETA of the render/encode tools surveyed. Two
	// separate rates are derived from the same samples:
	//
	//   * an INSTANTANEOUS rate over the short sliding window, which is
	//     responsive enough to be worth showing, but far too noisy to
	//     divide by; and
	//   * a CUMULATIVE rate over the whole render, whose sensitivity to
	//     new noise keeps shrinking, so the ETA drifts smoothly instead
	//     of oscillating every tick.
	//
	// The ETA uses the cumulative rate for exactly that reason. Nothing is
	// shown at all until kEtaWarmupMs has passed - an early estimate from
	// two samples is worse than admitting we don't know yet, so the label
	// reads "--:--" rather than a wild (or zero) guess.
	struct ProgressSample {
		qint64 elapsedMs = 0;
		int percent = 0;
	};
	static constexpr int kProgressSamples = 4;     // ring depth (~3s window at 1Hz)
	static constexpr qint64 kEtaWarmupMs = 4000;   // no estimate before this
	ProgressSample m_progressRing[kProgressSamples];
	int m_progressRingCount = 0;        // how many slots are actually filled

	void resetProgressSamples();
	// Appends a sample and returns the formatted "elapsed / ETA" status
	// text for the current tick.
	QString formatProgressStatus(qint64 elapsedMs, int percent);
	// Paints the progress bar green on success / red on failure, matching
	// Qt Creator's ProgressBarColorFinished / ProgressBarColorError.
	void setProgressResultState(const char *state);
	// Shows/hides m_statusWarningLabel - see that member's own comment.
	void setStatusWarning(const QString &text);
	void clearStatusWarning();
	// Clears or reddens the taskbar button and, if the window isn't the
	// active one, raises a tray notification.
	void notifyRenderFinished(bool success, const QString &message, double totalTime);

	// Last percentage pushed to the taskbar button, so the COM call only
	// fires when the integer percent actually changes.
	int m_lastTaskbarPercent = -1;

	// Used only for completion notifications - the app has no tray UI.
	// Null if the platform has no system tray.
	QSystemTrayIcon *m_trayIcon = nullptr;
	// The active-window complement to m_trayIcon's showMessage() below - see
	// notifyRenderFinished()'s own comment for which one fires when.
	ToastNotification *m_toast = nullptr;
	// Shared event filter that blocks accidental wheel-scroll on controls
	WheelIgnoreFilter *m_wheelFilter;
};

#endif // MAINWINDOW_H
