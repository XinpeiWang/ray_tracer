#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"
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
#include <QFile>
#include <QDebug>
#include <QStyledItemDelegate>


// ============================================================================
// Design tokens
// ============================================================================
// The theme keeps its cyberpunk identity, but routes every colour through a
// small named set instead of ad-hoc hex literals per rule. Two principles do
// most of the work here:
//
//   1. Saturated neon is an ACCENT, never body text. Pure #00FFFF/#FF00FF are
//      the harshest colours an sRGB display can produce; they used to be the
//      colour of every QLabel, log line and combo item, which made long
//      render logs genuinely tiring to read. Body text is now near-white and
//      the neon is reserved for titles, the primary action, focus rings and
//      selection.
//   2. Border weight encodes HIERARCHY. Everything used to carry a 2px glowing
//      border - buttons, tabs, panes, spin boxes, the progress bar, even the
//      scroll bar - so nothing stood out. Ordinary chrome is now a 1px subtle
//      line; only the primary action (START RENDER) keeps a 2px accent border.
// ============================================================================
namespace {

// Radius scale - two values, not five. Unlike colours these are not part of a
// theme: every scheme shares the same geometry, only the palette changes.
constexpr const char *kRadius       = "6px";      // Controls
constexpr const char *kRadiusLarge  = "10px";     // Containers

// Qt stylesheets want "#RRGGBB" strings, palettes hold QColor.
QString hex(const QColor &c) { return c.name(QColor::HexRgb); }

// Text drawn ON a filled element (a selected row, a completed progress bar)
// has to contrast with THAT fill, not with the window - which is why a single
// body-text colour cannot serve both. White is right for every dark scheme and
// for mid-tone fills; only a genuinely pale fill needs dark text instead. The
// threshold is high because white-on-mid-tone stays readable a good way up the
// range - flipping at the midpoint sent Solarized Light's violet accent to
// cream text, which was the worse of the two options.
QColor textOn(const QColor &fill, const theme::Palette &p) {
	return fill.lightness() > 170 ? p.surface0 : QColor(Qt::white);
}

// Builds the QSS fragment that paints a theme's decorative motif on the tab
// pane. Empty for the eight schemes that have none, which is why the tokens are
// substituted rather than the rule being written inline.
//
// Qt's stylesheet engine has no background-size, so a non-tiled motif is drawn
// at whatever size its SVG declares and anchored to a corner - that is why the
// artwork is authored at a deliberate size rather than expected to scale.
QString paneBackgroundRule(const theme::Palette &p) {
	if (p.backgroundImage.isEmpty()) return QString();

	// A missing or unregistered resource makes QSS silently skip the
	// declaration - the same failure mode that once made the SVG icons render
	// as nothing at all. Check rather than trust, and say so.
	if (!QFile::exists(p.backgroundImage)) {
		qWarning() << "Theme" << p.id << "declares background" << p.backgroundImage
				   << "but it is not in the resource bundle - motif skipped."
				   << "Check resources.qrc.";
		return QString();
	}

	// One line: whitespace inside a QSS block is insignificant, and keeping the
	// fragment flat avoids having to match the surrounding rule's indentation.
	//
	// background-attachment: fixed anchors the motif to the viewport rather
	// than to the scrolled content, so it stays put while the settings panel
	// scrolls past it - a motif that slid around with the content would read as
	// a rendering glitch.
	return QString("background-image: url(%1); background-repeat: %2; "
				   "background-position: %3; background-attachment: fixed;")
		.arg(p.backgroundImage,
			 p.backgroundTiled ? "repeat" : "no-repeat",
			 p.backgroundTiled ? "top left" : p.backgroundPosition);
}


} // namespace

void MainWindow::applyTheme(const theme::Palette &p) {
	m_activeTheme = p;

	// HorizontalTabBar's label paints itself rather than going through the
	// stylesheet (see that class's own comment on why), so it needs these
	// three colours handed to it directly instead of picking them up from
	// QTabBar#previewSubTabsBar::tab the way the tab's shape/background
	// still does. Matches that QSS rule's normal/hover/selected colours.
	if (m_previewSubTabs) {
		m_previewSubTabs->tabBar()->setColors(p.textMuted, p.textBody, p.accentSecondary);
		m_previewSubTabs->setEmptyStateColors(p.textMuted, p.textBody, p.textMuted);
	}

	// Qt's own palette still matters: it is what non-stylesheet painting and
	// native dialogs read, so it has to track the scheme too rather than being
	// left on whatever the first theme set.
	QPalette appPalette;
	appPalette.setColor(QPalette::Window, p.surface0);
	appPalette.setColor(QPalette::WindowText, p.textBody);
	appPalette.setColor(QPalette::Base, p.surface1);
	appPalette.setColor(QPalette::AlternateBase, p.surface2);
	appPalette.setColor(QPalette::ToolTipBase, p.surface2);
	appPalette.setColor(QPalette::ToolTipText, p.textBody);
	appPalette.setColor(QPalette::Text, p.textBody);
	appPalette.setColor(QPalette::Button, p.surface2);
	appPalette.setColor(QPalette::ButtonText, p.textBody);
	appPalette.setColor(QPalette::BrightText, p.accentPrimary);
	appPalette.setColor(QPalette::Link, p.accentSecondary);
	appPalette.setColor(QPalette::Highlight, p.accentDim);
	appPalette.setColor(QPalette::HighlightedText, textOn(p.accentDim, p));

	// Disabled state colors
	appPalette.setColor(QPalette::Disabled, QPalette::WindowText, p.textDisabled);
	appPalette.setColor(QPalette::Disabled, QPalette::Text, p.textDisabled);
	appPalette.setColor(QPalette::Disabled, QPalette::ButtonText, p.textDisabled);

	qApp->setPalette(appPalette);
	qApp->setStyle(QStyleFactory::create("Fusion"));

	// The font FAMILY is a fully independent choice (see font_switch.cpp) -
	// applied once at startup and whenever the Font menu changes it, never
	// here, so switching themes can no longer silently reset it back to
	// Cyberpunk. Font SIZE is different: the rules below still set it per
	// selector (that hierarchy - group box titles vs. captions vs. the
	// primary button - is themeing, not something font_switch.cpp should
	// know about), but every value is now an offset from m_activeFontId's
	// own base size rather than a hardcoded number, so a font switch's point
	// size is still visible everywhere instead of only in qApp->font().
	const int basePt = fontPointSizeForId(m_activeFontId);

	// Apply cyberpunk stylesheet for enhanced neon effects
	QString stylesheet = QString(R"(
		/* The menu bar and its dropdown menus previously had no rules at
		   all here, so they rendered as plain native OS chrome - a bright
		   white Windows menu bar across the top of an otherwise fully
		   dark-themed window. Selected-state colours match the existing
		   convention (QListView::item:hover/selected below) rather than
		   inventing a new one: %HOVER_ROW%/%ACCENT_2% for a hovered top-
		   level item, %ACCENT_DIM%/%SELECTED_TEXT% for a highlighted
		   dropdown entry. */
		QMenuBar {
			background-color: %SURFACE0%;
			color: %TEXT%;
			border-bottom: 1px solid %BORDER%;
			padding: 2px;
		}
		QMenuBar::item {
			background: transparent;
			padding: 6px 12px;
			border-radius: 4px;
		}
		QMenuBar::item:selected {
			background-color: %HOVER_ROW%;
			color: %ACCENT_2%;
		}
		QMenuBar::item:pressed {
			background-color: %SURFACE2%;
		}
		QMenu {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			padding: 4px;
		}
		QMenu::item {
			background: transparent;
			color: %TEXT%;
			padding: 6px 24px 6px 12px;
			border-radius: 4px;
		}
		QMenu::item:selected {
			background-color: %ACCENT_DIM%;
			color: %SELECTED_TEXT%;
		}
		QMenu::item:disabled {
			color: %TEXT_DISABLED%;
		}
		QMenu::separator {
			height: 1px;
			background-color: %BORDER%;
			margin: 4px 8px;
		}
		/* QMessageBox::critical/::warning (used for save/error dialogs) are
		   real Qt-painted QDialog subclasses on Windows, not native OS
		   popups, so this reaches them - unlike QFileDialog elsewhere in
		   this app, which uses the native picker (getSaveFileName() with
		   no DontUseNativeDialog flag) and QSS genuinely cannot touch.
		   QLabel's own global rule below already covers the message text
		   colour, so only the background needs setting here. */
		QDialog {
			background-color: %SURFACE1%;
		}
		QGroupBox {
			border: 1px solid %BORDER%;
			border-radius: %RADIUS_LG%;
			margin-top: 26px;
			margin-bottom: 6px;
			padding: 4px 10px 10px 10px;
			background-color: %SURFACE1%;
			color: %TEXT%;
			font-size: %FS_P1%;
		}
		QGroupBox::title {
			subcontrol-origin: margin;
			subcontrol-position: top left;
			padding: 2px 12px;
			left: 12px;
			top: 2px;
			color: %ACCENT_1%;
			font-size: %FS_P1%;
			font-weight: bold;
			background-color: %SURFACE1%;
		}
		/* Ordinary (secondary) buttons: a quiet 1px outline. The primary
		   action is singled out by object name further down. */
		QPushButton {
			background-color: %SURFACE2%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			color: %TEXT%;
			padding: 8px 18px;
			font-size: %FS_P1%;
			min-height: 34px;
		}
		QPushButton:hover {
			background-color: %SURFACE3%;
			border-color: %ACCENT_2%;
			color: %ACCENT_2%;
		}
		QPushButton:pressed {
			background-color: %SURFACE1%;
		}
		QPushButton:focus {
			border: 1px solid %ACCENT_2%;
		}
		QPushButton:disabled {
			background-color: %SURFACE1%;
			border-color: %BORDER%;
			color: %TEXT_DISABLED%;
		}
		/* Primary action - the one element allowed a 2px accent border and
		   bold text, so it reads as the main thing to click. */
		QPushButton#primaryAction {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 %PRIMARY_TOP%, stop:1 %PRIMARY_BOTTOM%);
			border: 2px solid %ACCENT_DIM%;
			color: %ACCENT_1%;
			font-weight: bold;
			font-size: %FS_P2%;
		}
		QPushButton#primaryAction:hover {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 %PRIMARY_TOP_HOVER%, stop:1 %PRIMARY_BOTTOM_HOVER%);
			border-color: %ACCENT_2%;
			color: %ACCENT_2%;
		}
		QPushButton#primaryAction:pressed {
			background-color: %PRIMARY_BOTTOM%;
		}
		/* Without this, keyboard focus on the app's single most-clicked
		   button is invisible: an ID selector's own (unstyled-for-focus)
		   rules outrank the generic QPushButton:focus rule above in Qt's
		   stylesheet cascade, so that rule alone never reaches this button. */
		QPushButton#primaryAction:focus {
			border-color: %ACCENT_2%;
		}
		QPushButton#primaryAction:disabled {
			background-color: %SURFACE1%;
			border: 2px solid %BORDER%;
			color: %TEXT_DISABLED%;
		}
		/* Destructive action (Stop Render, Clear Queue) - a quiet outline in
		   the theme's error colour at rest, filling solid on hover so the
		   irreversible click reads as a deliberate choice rather than an
		   ordinary button. Reuses %ERROR%/%ON_ERROR% (already defined for
		   the progress bar's failed state) rather than introducing a new
		   token pair. */
		QPushButton#dangerAction {
			background-color: %SURFACE2%;
			border: 1px solid %ERROR%;
			color: %ERROR%;
		}
		QPushButton#dangerAction:hover {
			background-color: %ERROR%;
			border-color: %ERROR%;
			color: %ON_ERROR%;
		}
		QPushButton#dangerAction:pressed {
			background-color: %SURFACE1%;
		}
		/* Wider border (matches how #primaryAction's rest state already differs
		   from #dangerAction's by width) so tabbing onto Stop/Clear Queue - this
		   app's one unconfirmed destructive action - is visibly distinguishable
		   from its unfocused rest state, same reasoning as #primaryAction's own
		   :focus rule above. */
		QPushButton#dangerAction:focus {
			border: 2px solid %ERROR%;
		}
		QPushButton#dangerAction:disabled {
			background-color: %SURFACE1%;
			border-color: %BORDER%;
			color: %TEXT_DISABLED%;
		}
		QProgressBar {
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			text-align: center;
			background-color: %SURFACE0%;
			color: %TEXT%;
			font-size: %FS_0%;
			min-height: 28px;
		}
		QProgressBar::chunk {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 %ACCENT_1%, stop:1 %ACCENT_2%);
			border-radius: 5px;
		}
		/* Outcome colouring, driven by the "resultState" dynamic property
		   (see MainWindow::setProgressResultState). The finished bar keeps
		   its fill and just changes colour, so a completed or failed render
		   stays readable instead of snapping back to an empty bar - the
		   same convention Qt Creator uses, including its exact colours. */
		QProgressBar[resultState="success"]::chunk {
			background-color: %SUCCESS%;
		}
		QProgressBar[resultState="error"]::chunk {
			background-color: %ERROR%;
		}
		/* A finished bar is filled edge to edge, so its percentage text sits on
		   the outcome colour and has to contrast with THAT. Leaving it on body
		   text put a dark olive "100%" on Solarized Light's dark olive success
		   fill - technically drawn, effectively invisible. The in-progress bar
		   deliberately keeps body text: its text spans both the filled and
		   unfilled halves, so no single colour is right and the trough is the
		   safer one to match. */
		QProgressBar[resultState="success"] {
			color: %ON_SUCCESS%;
		}
		QProgressBar[resultState="error"] {
			color: %ON_ERROR%;
		}
		QTabWidget::pane {
			border: 1px solid %BORDER%;
			border-radius: 0px %RADIUS% %RADIUS% %RADIUS%;
			background-color: %SURFACE0%;
			top: -1px;
		}
		/* The scrolling tab pages, and where a theme's decorative motif is
		   painted. Not on QTabWidget::pane, which this covers completely. */
		QScrollArea#tabScroll {
			background-color: %SURFACE0%;
			border: none;
			%PANE_BACKGROUND%
		}
		/* A scroll area's own background is covered by two widgets Qt creates
		   for it - the viewport, and the content widget inside that - both of
		   which fill themselves opaquely by default. Punching both through is
		   the documented way to let a scroll area's background show. The child
		   selectors are deliberately direct (">"): group boxes sit deeper and
		   keep their own opaque QGroupBox rule, which is what stops the motif
		   from ever appearing behind text. */
		QScrollArea#tabScroll > QWidget {
			background: transparent;
		}
		QScrollArea#tabScroll > QWidget > QWidget {
			background: transparent;
		}
		QTabBar::tab {
			background-color: transparent;
			border: 1px solid transparent;
			border-bottom: 1px solid %BORDER%;
			border-top-left-radius: %RADIUS%;
			border-top-right-radius: %RADIUS%;
			/* Tighter than the old 10px 18px - every pixel of per-tab padding
			   here is pixels sooner the whole strip gives up stretching (see
			   ExpandingTabBar, mainwindow.h) and switches to scroll arrows as
			   the window narrows. */
			padding: 8px 12px;
			color: %TEXT_MUTED%;
			font-size: %FS_P1%;
			/* No minimum width: ExpandingTabBar (mainwindow.h) already grows
			   each tab to fill the bar when there's room. A fixed 100px floor
			   used to force the strip's natural width past the window's
			   typical size once a 7th tab (Diagnostics) was added, tipping it
			   into scroll-button mode instead of just spreading tabs out -
			   same reasoning sceneCategoryTabs below already applies with its
			   own zero minimum. */
			min-width: 0px;
			margin-right: 2px;
		}
		/* The selected tab is marked by an accent underline rather than a
		   full glowing outline - the tab strip is navigation, not the
		   loudest thing on screen. */
		QTabBar::tab:selected {
			background-color: %SURFACE0%;
			border-color: %BORDER%;
			border-bottom: 2px solid %ACCENT_2%;
			color: %ACCENT_2%;
			font-weight: bold;
		}
		QTabBar::tab:hover:!selected {
			background-color: %SURFACE1%;
			color: %TEXT%;
		}
		/* The scene-category bar is a filter inside a group box, not top-level
		   navigation, so it is deliberately quieter than the main tab strip
		   above: smaller text, tighter padding, no minimum width (eight
		   categories at the main strip's 100px each would not fit), and a
		   1px underline rather than 2px. Same accent, less weight - it should
		   read as subordinate to the tabs it sits under. */
		QTabBar#sceneCategoryTabs::tab {
			padding: 5px 12px;
			font-size: %FS_M1%;
			min-width: 0px;
			margin-right: 1px;
		}
		QTabBar#sceneCategoryTabs::tab:selected {
			border-bottom: 1px solid %ACCENT_2%;
		}
		/* The Preview tab's per-render sub-tabs (createPreviewTab()) sit on
		   the LEFT (QTabWidget::West) rather than across the top, so the
		   corners and the selected-tab accent move to match: left-rounded
		   (not top-rounded) and a right-side border (not bottom) - the main
		   tab strip's rule above is built for a horizontal bar and its
		   bottom-edge underline would land on the wrong side entirely here. */
		QTabBar#previewSubTabsBar::tab {
			background-color: transparent;
			border: 1px solid transparent;
			border-right: 1px solid %BORDER%;
			border-top-left-radius: %RADIUS%;
			border-bottom-left-radius: %RADIUS%;
			padding: 10px 14px;
			color: %TEXT_MUTED%;
			font-size: %FS_M1%;
			min-width: 0px;
			min-height: 32px;
			margin-bottom: 2px;
		}
		QTabBar#previewSubTabsBar::tab:selected {
			background-color: %SURFACE0%;
			border-color: %BORDER%;
			border-right: 2px solid %ACCENT_2%;
			color: %ACCENT_2%;
			font-weight: bold;
		}
		QTabBar#previewSubTabsBar::tab:hover:!selected {
			background-color: %SURFACE1%;
			color: %TEXT%;
		}
		QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			padding: 6px 8px;
			color: %TEXT%;
			font-size: %FS_0%;
			min-height: 26px;
			margin: 3px 2px;
		}
		/* Fusion splits a spin box's contents-rect height in half for its
		   up/down step buttons, so at the shared 26px min-height above
		   (~12px per button once the border is subtracted) the PlusMinus
		   primitives' cross-bars are proportionally too thick for the
		   space: the "+" blobs into a diamond and only the "-" reads
		   cleanly (see styleSpinBox()'s comment for why PlusMinus is used
		   at all). Taller spin boxes only, so each button gets room for a
		   legible glyph; the combo box keeps the 26px height. */
		QSpinBox, QDoubleSpinBox {
			min-height: 40px;
		}
		/* Sizes only the clickable button rect, not ::up-arrow/
		   ::down-arrow - those stay completely unstyled so Fusion keeps
		   drawing its native PE_IndicatorSpinPlus/Minus fill inside the
		   wider rect (styling the arrow subcontrol itself is what
		   collapsed to an empty box, see styleSpinBox()'s comment).
		   Widens len = min(buttonWidth, buttonHeight)'s width side to
		   match the min-height bump above, so neither dimension caps it
		   down to a thick, blobby cross. */
		QSpinBox::up-button, QDoubleSpinBox::up-button,
		QSpinBox::down-button, QDoubleSpinBox::down-button {
			width: 20px;
		}
		QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {
			background-color: %SURFACE2%;
			border-color: %TEXT_MUTED%;
		}
		/* Focus is the one state that gets the accent colour, so keyboard
		   navigation is unmistakable without every control glowing at rest. */
		QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
			background-color: %SURFACE2%;
			border: 1px solid %ACCENT_2%;
			color: %TEXT%;
		}
		/* Disabled state was previously unstyled, so a disabled combo/spin
		   box (e.g. Render Options' sampler/exposure fields under GPU mode)
		   looked nearly identical to an enabled one at a glance - only
		   discoverable by actually clicking it. Matches QPushButton:disabled's
		   own dimming below. */
		QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {
			background-color: %SURFACE1%;
			border-color: %BORDER%;
			color: %TEXT_DISABLED%;
		}
		QCheckBox {
			color: %TEXT%;
			font-size: %FS_0%;
			spacing: 8px;
			padding: 4px 2px;
		}
		QCheckBox::indicator {
			width: 18px;
			height: 18px;
			border: 1px solid %BORDER_STRONG%;
			border-radius: 4px;
			background-color: %SURFACE1%;
		}
		QCheckBox::indicator:hover {
			border-color: %TEXT_MUTED%;
			background-color: %SURFACE2%;
		}
		QCheckBox::indicator:checked {
			background-color: %ACCENT_2%;
			border-color: %ACCENT_2%;
		}
		QCheckBox:disabled {
			color: %TEXT_DISABLED%;
		}
		QCheckBox::indicator:disabled {
			background-color: %SURFACE1%;
			border-color: %BORDER%;
		}
		/* No custom ::up-button/::down-button/::up-arrow/::down-arrow rules:
		   every attempt to restyle them (custom background+border, image
		   icons via border-triangle/data-URI/PNG resource, even native
		   PlusMinus text) rendered as a completely empty box with this
		   style/Qt version - something about customizing those subcontrols
		   collapses their content area to nothing. Leaving them alone gets
		   Qt's stock Fusion step buttons, which are the one thing
		   confirmed to actually render. */
		QComboBox QAbstractItemView {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			selection-background-color: %ACCENT_DIM%;
			selection-color: %SELECTED_TEXT%;
			color: %TEXT%;
			outline: none;
			padding: 2px;
		}
		QComboBox QAbstractItemView::item {
			padding: 8px;
			min-height: 30px;
			border: none;
			background-color: transparent;
			color: %TEXT%;
		}
		QComboBox QAbstractItemView::item:hover {
			background-color: %HOVER_ROW%;
			color: %ACCENT_2%;
		}
		QComboBox QAbstractItemView::item:selected {
			background-color: %ACCENT_DIM%;
			color: %SELECTED_TEXT%;
		}
		QListView {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			color: %TEXT%;
		}
		QListView::item {
			padding: 8px;
			min-height: 30px;
		}
		QListView::item:hover {
			background-color: %HOVER_ROW%;
			color: %ACCENT_2%;
		}
		QListView::item:selected {
			background-color: %ACCENT_DIM%;
			color: %SELECTED_TEXT%;
		}
		QLabel {
			color: %TEXT%;
			font-size: %FS_0%;
			padding: 4px 5px;
			margin: 3px 2px;
			background: transparent;
		}
		/* Secondary and explanatory surfaces, addressed by object name rather
		   than by a stylesheet set on the widget itself. A per-widget sheet is
		   invisible to this one, so it has to be rebuilt by hand on every
		   theme change - which is exactly how these ended up frozen on the
		   first scheme's colours. Rules here re-theme for free. */
		QLabel#sceneInfo {
			color: %TEXT_MUTED%;
			background-color: %SURFACE2%;
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			padding: 8px 12px;
			font-size: %FS_0%;
		}
		QLabel#previewInfo {
			color: %TEXT_MUTED%;
			font-size: %FS_M2%;
		}
		QLabel#currentJobLabel {
			color: %ACCENT_1%;
			font-size: %FS_0%;
			font-weight: bold;
			padding-bottom: 4px;
		}
		QLabel#previewSceneDesc, QLabel#previewTechniqueNote {
			color: %TEXT_MUTED%;
			background-color: %SURFACE2%;
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			padding: 8px 10px;
			font-size: %FS_M1%;
		}
		QLabel#videoInfo {
			color: %TEXT_MUTED%;
			font-style: italic;
			padding: 10px;
		}
		/* Shared by the long help/requirements blurbs on the Video tab. */
		QLabel#mutedInfo {
			color: %TEXT_MUTED%;
			padding: 10px;
		}
		QLabel#statusInfo {
			color: %TEXT_MUTED%;
			padding: 0 8px;
		}
		/* A caveat below the main status line (e.g. render succeeded but its
		   preview image failed to load) - reuses the theme's own logWarning
		   colour (already tuned per-palette to read against %SURFACE0%,
		   which the status bar/log share) rather than adding a new palette
		   field just for this one label. */
		QLabel#statusWarning {
			color: %WARNING%;
			font-size: %FS_M1%;
		}
		/* Video Settings' "these controls are currently ignored" banner - see
		   createVideoTab()'s own comment. Same %WARNING%/%SURFACE1%/%RADIUS%
		   tokens the rest of the app's warning/surface treatment already
		   uses, just with a border and padding since this sits at the top of
		   a whole tab rather than trailing a status line. */
		QLabel#videoModeWarning {
			color: %WARNING%;
			background-color: %SURFACE1%;
			border: 1px solid %WARNING%;
			border-radius: %RADIUS%;
			padding: 8px 12px;
			font-size: %FS_M1%;
		}
		ScaledImageLabel {
			background-color: %SURFACE0%;
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			color: %TEXT_MUTED%;
			font-size: %FS_0%;
		}
		QStatusBar {
			background-color: %SURFACE0%;
			color: %TEXT_MUTED%;
		}
		QStatusBar::item {
			border: none;
		}
		QFormLayout {
			spacing: 10px;
		}
		QTextEdit {
			background-color: %SURFACE0%;
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			color: %TEXT%;
			selection-background-color: %ACCENT_DIM%;
			selection-color: %SELECTED_TEXT%;
			font-size: %FS_M1%;
			padding: 5px;
		}
		/* Scroll bars are chrome: no border, transparent trough, and a
		   muted handle that only picks up the accent on hover. Previously
		   they carried a 2px cyan border and a magenta gradient handle,
		   which made them one of the loudest things on screen. */
		QScrollBar:vertical {
			background: transparent;
			width: 12px;
			margin: 0px;
			border: none;
		}
		QScrollBar::handle:vertical {
			background-color: %BORDER_STRONG%;
			border-radius: 4px;
			min-height: 30px;
			margin: 2px;
		}
		QScrollBar::handle:vertical:hover {
			background-color: %ACCENT_DIM%;
		}
		QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
			height: 0px;
		}
		QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
			background: none;
		}
		QScrollBar:horizontal {
			background: transparent;
			height: 12px;
			margin: 0px;
			border: none;
		}
		QScrollBar::handle:horizontal {
			background-color: %BORDER_STRONG%;
			border-radius: 4px;
			min-width: 30px;
			margin: 2px;
		}
		QScrollBar::handle:horizontal:hover {
			background-color: %ACCENT_DIM%;
		}
		QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
			width: 0px;
		}
		QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
			background: none;
		}
		/* Video scrubber (Preview tab's position slider). A plain drawn
		   handle rather than an image asset - a themed circle over a
		   two-tone groove, filled portion in the accent colour like the
		   progress bar's own fill. */
		QSlider::groove:horizontal {
			height: 4px;
			border-radius: 2px;
			background-color: %SURFACE1%;
		}
		QSlider::sub-page:horizontal {
			height: 4px;
			border-radius: 2px;
			background-color: %ACCENT_1%;
		}
		QSlider::handle:horizontal {
			width: 14px;
			height: 14px;
			margin: -6px 0;
			border-radius: 7px;
			background-color: %ACCENT_1%;
			border: 1px solid %ACCENT_DIM%;
		}
		QSlider::handle:horizontal:hover {
			background-color: %ACCENT_2%;
			border-color: %ACCENT_2%;
		}
		/* Scrub-position bubble that follows the video slider's handle while
		   dragging - see addVideoPreviewTab()'s own comment. */
		QLabel#scrubBubble {
			background-color: %SURFACE3%;
			color: %TEXT%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: 4px;
			padding: 2px 6px;
			font-size: %FS_M2%;
		}
		/* Preview tab's image/video splitter - a slim themed grip instead
		   of the OS's default flat grey bar. */
		QSplitter::handle:horizontal {
			width: 6px;
			background-color: %BORDER%;
		}
		QSplitter::handle:horizontal:hover {
			background-color: %ACCENT_2%;
		}
		QSplitter::handle:vertical {
			height: 6px;
			background-color: %BORDER%;
		}
		QSplitter::handle:vertical:hover {
			background-color: %ACCENT_2%;
		}
		QToolTip {
			background-color: %SURFACE2%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: 4px;
			color: %TEXT%;
			padding: 4px 8px;
		}
	)")
		.replace("%PANE_BACKGROUND%", paneBackgroundRule(p))
		.replace("%SURFACE0%",      hex(p.surface0))
		.replace("%SURFACE1%",      hex(p.surface1))
		.replace("%SURFACE2%",      hex(p.surface2))
		.replace("%SURFACE3%",      hex(p.surface3))
		.replace("%SELECTED_TEXT%", hex(textOn(p.accentDim, p)))
		.replace("%HOVER_ROW%",     hex(p.hoverRow))
		.replace("%TEXT%",          hex(p.textBody))
		.replace("%TEXT_MUTED%",    hex(p.textMuted))
		.replace("%TEXT_DISABLED%", hex(p.textDisabled))
		.replace("%ACCENT_2%",      hex(p.accentSecondary))
		.replace("%ACCENT_1%",      hex(p.accentPrimary))
		.replace("%ACCENT_DIM%",    hex(p.accentDim))
		.replace("%BORDER_STRONG%", hex(p.borderStrong))
		.replace("%BORDER%",        hex(p.border))
		.replace("%ON_SUCCESS%",    hex(textOn(p.success, p)))
		.replace("%ON_ERROR%",      hex(textOn(p.error, p)))
		.replace("%SUCCESS%",       hex(p.success))
		.replace("%ERROR%",         hex(p.error))
		.replace("%WARNING%",       hex(p.logWarning))
		.replace("%PRIMARY_TOP_HOVER%",    hex(p.primaryTopHover))
		.replace("%PRIMARY_BOTTOM_HOVER%", hex(p.primaryBottomHover))
		.replace("%PRIMARY_TOP%",          hex(p.primaryTop))
		.replace("%PRIMARY_BOTTOM%",       hex(p.primaryBottom))
		.replace("%RADIUS_LG%",     kRadiusLarge)
		.replace("%RADIUS%",        kRadius)
		// Every font-size rule above is basePt plus a fixed offset from the
		// hierarchy this stylesheet was originally authored with (group box
		// titles/buttons one step up, captions/badges one or two steps down),
		// not a hardcoded point size - so a font switch's point size actually
		// scales the whole UI instead of only qApp->font()'s own default.
		.replace("%FS_M2%", QStringLiteral("%1pt").arg(basePt - 2))
		.replace("%FS_M1%", QStringLiteral("%1pt").arg(basePt - 1))
		.replace("%FS_0%",  QStringLiteral("%1pt").arg(basePt))
		.replace("%FS_P1%", QStringLiteral("%1pt").arg(basePt + 1))
		.replace("%FS_P2%", QStringLiteral("%1pt").arg(basePt + 2));

	qApp->setStyleSheet(stylesheet);
}

void MainWindow::styleGroupBox(QGroupBox *box) {
	// WA_StyledBackground forces Qt to paint the CSS background+border
	// on QGroupBox (which by default ignores background-color in stylesheets).
	box->setAttribute(Qt::WA_StyledBackground, true);
	// A quiet elevation shadow - noticeably subtler than the buttons' own
	// (see applyElevation() call sites in mainwindow.cpp/mainwindow_tabs.cpp),
	// since there are 6 of these panels visible at once on some tabs and a
	// strong shadow on all of them would read as clutter rather than depth.
	applyElevation(box, /*blurRadius=*/16, /*offsetY=*/3, /*alpha=*/55);
}

void MainWindow::applyElevation(QWidget *widget, qreal blurRadius, qreal offsetY, int alpha) {
	auto *shadow = new QGraphicsDropShadowEffect(widget);
	shadow->setBlurRadius(blurRadius);
	shadow->setOffset(0, offsetY);
	shadow->setColor(QColor(0, 0, 0, alpha));
	widget->setGraphicsEffect(shadow);
}

void MainWindow::applyGlow(QWidget *widget, qreal blurRadius, const QColor &color) {
	auto *glow = new QGraphicsDropShadowEffect(widget);
	glow->setBlurRadius(blurRadius);
	glow->setOffset(0, 0);
	glow->setColor(color);
	widget->setGraphicsEffect(glow);
}

void MainWindow::styleComboBox(QComboBox *combo) {
	QAbstractItemView *view = combo->view();

	// Cap the popup's visible rows so it scrolls instead of growing past
	// the screen - matters most for m_sceneCombo (81 scenes and counting);
	// harmless for the shorter dropdowns (mode/quality/resolution/camera
	// presets), which never reach this count anyway.
	// setMaxVisibleItems() alone doesn't reliably constrain the popup here
	// (confirmed live - the popup still grew to fit every item, no
	// scrollbar) - the Fusion style forced onto the view below, combined
	// with this QSS's `min-height: 32px` per item, apparently isn't what
	// Qt's own maxVisibleItems sizing math measures against on this style/
	// platform combination. A hard pixel cap on the view itself sidesteps
	// that: QAbstractItemView shows a scrollbar automatically once content
	// exceeds its viewport height, regardless of how that height was set.
	combo->setMaxVisibleItems(12);
	view->setMaximumHeight(420);
	// The view alone scrolling correctly wasn't enough - confirmed live,
	// the popup's own top-level container window (a separate widget
	// QComboBox lazily creates the first time view() is called, sized from
	// its own pre-QSS sizeHint estimate) stayed at the full uncapped
	// height, leaving a large dead/empty area around the now-properly-
	// clamped-and-scrollable view floating inside it. Cap the container
	// too, once it exists (immediately, since accessing view() above is
	// exactly what creates it).
	if (QWidget *popupContainer = view->parentWidget())
		popupContainer->setMaximumHeight(420);

	// QComboBox installs a plain QItemDelegate on its popup, and QItemDelegate
	// paints items itself without consulting the stylesheet - which silently
	// discards every QAbstractItemView::item rule below. Only QStyledItemDelegate
	// routes item painting back through the style, so without this line the
	// hover and selection styling is dead code that renders nothing.
	view->setItemDelegate(new QStyledItemDelegate(view));

	view->setMouseTracking(true);
	view->viewport()->setMouseTracking(true);
	view->setAttribute(Qt::WA_Hover, true);
	view->viewport()->setAttribute(Qt::WA_Hover, true);

	applyComboPopupPalette(combo);

	combo->installEventFilter(m_wheelFilter);
}

// The popup is styled separately from the global sheet (Qt does not reach into
// it reliably), so it reads the active scheme directly. Split out from
// styleComboBox() so a theme switch can re-run just this part: the sizing and
// delegate setup above is one-time, but the colours are not, and leaving them
// baked at construction was exactly what made popups keep the previous scheme
// after switching.
void MainWindow::applyComboPopupPalette(QComboBox *combo) {
	QAbstractItemView *view = combo->view();

	view->setStyleSheet(QString(R"(
		QAbstractItemView {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			outline: none;
			color: %TEXT%;
			padding: 2px;
		}
		QAbstractItemView::item {
			padding: 8px 12px;
			min-height: 32px;
			border: none;
			border-left: 3px solid transparent;
			color: %TEXT%;
		}
		/* Only :selected is styled here, deliberately. In a combo popup,
		   moving the mouse over the list moves the CURRENT item rather than
		   setting a hover state, so the row under the cursor arrives as
		   selected and :hover never fires (verified live with probe colours).
		   A :hover rule here would be dead code.

		   The marker is a 3px accent bar plus a quiet background rather than a
		   full-width accent fill: this list is 81 scenes long, and a saturated
		   band sweeping down it as the cursor moves is exhausting to read
		   against. The reserved-transparent border above keeps the text from
		   shifting sideways when the bar appears. */
		QAbstractItemView::item:selected {
			background-color: %HOVER_ROW%;
			color: %ACCENT_2%;
			border-left: 3px solid %ACCENT_1%;
		}
	)")
		.replace("%SURFACE1%",      hex(m_activeTheme.surface1))
		.replace("%HOVER_ROW%",     hex(m_activeTheme.hoverRow))
		.replace("%TEXT%",          hex(m_activeTheme.textBody))
		.replace("%ACCENT_2%",      hex(m_activeTheme.accentSecondary))
		.replace("%ACCENT_1%",      hex(m_activeTheme.accentPrimary))
		.replace("%BORDER_STRONG%", hex(m_activeTheme.borderStrong))
		.replace("%RADIUS%",        kRadius));
}

void MainWindow::styleSpinBox(QAbstractSpinBox *spinBox) {
	// UpDownArrows (the unstyled default - no setButtonSymbols() call) was
	// tried first, reusing the same "leave the subcontrol alone" approach
	// that works for m_sceneCombo's own drop-down arrow - it rendered as a
	// completely empty box instead (confirmed live), matching this file's
	// prior history of every ::up-arrow/::down-arrow customization attempt
	// collapsing to nothing with this style/Qt version. Back to PlusMinus:
	// Fusion's PE_IndicatorSpinPlus/Minus fill-rect primitives
	// (QCommonStyle::drawPrimitive) are the one thing confirmed to render
	// something. Geometry: horizontal bar len x step, plus a crossing
	// vertical bar for "+", where len = min(buttonWidth, buttonHeight) and
	// step = round-up-to-even((len+4)/5) - see applyDarkTheme()'s
	// ::up-button/::down-button width rule, which widens the button rect
	// (without touching ::up-arrow/::down-arrow, so the native fill
	// primitive keeps drawing) so len isn't capped by Fusion's narrow
	// default button width, keeping the step/len ratio low enough to read
	// as a cross instead of a blob.
	spinBox->setButtonSymbols(QAbstractSpinBox::PlusMinus);

	// Styling otherwise comes entirely from the global stylesheet in
	// applyDarkTheme() now, so every QSpinBox/QDoubleSpinBox looks the
	// same - including the Camera X/Y/Z fields, which now go through this
	// function too instead of a bare installEventFilter() call.
	spinBox->installEventFilter(m_wheelFilter);
}

void MainWindow::styleCheckBox(QCheckBox *box) {
	// No wheel-filter/scroll-guard needed (unlike combo/spin boxes, a
	// checkbox has no scrollable value a stray wheel event could change) -
	// styling comes entirely from applyTheme()'s global QCheckBox rules, so
	// this function exists mainly for symmetry with styleComboBox()/
	// styleSpinBox()/styleGroupBox() and as the one place to add per-widget
	// checkbox behavior later if it's ever needed.
	(void)box;
}

// Wraps plain, blank-line-separated paragraphs into HTML so Qt's rich-text
// tooltip renderer word-wraps at a fixed width, instead of the unwrapped
// single line (or manually-inserted \n breaks, like this file's other
// setToolTip() calls) a plain-text tooltip would otherwise produce - the
// beginner explanations createInfoIcon() carries, and the per-scene notes
// scene_technique_notes.h supplies (used here and from the scene combo/grid
// in mainwindow_tabs.cpp), are real paragraphs, long enough that manual line
// breaks would be brittle to edit. Escaped via QString::toHtmlEscaped()
// first so a "<" or "&" in the explanation text itself can't be misread as
// markup. A MainWindow member rather than file-local, so mainwindow_tabs.cpp
// can reuse it instead of hand-rolling a second HTML-wrapping helper.
QString MainWindow::plainTextToHtmlParagraphs(const QString &plainText) {
	QStringList paragraphs = plainText.split("\n\n", Qt::SkipEmptyParts);
	QString body;
	for (const QString &para : paragraphs) {
		body += "<p>" + para.toHtmlEscaped() + "</p>";
	}
	return body;
}

QString MainWindow::wrapTooltipHtml(const QString &plainText) {
	return "<html><body style=\"width:320px;\">" + plainTextToHtmlParagraphs(plainText) + "</body></html>";
}

// A small, flat, icon-only "(i)" mark - the beginner-facing explanation
// layer this app didn't have before: every existing setToolTip() call in
// this file describes what a control DOES, not the computer-graphics
// concept behind it. Deliberately not connected to any click handler and
// not focusable (setFocusPolicy(Qt::NoFocus)) - it exists purely as a
// hover affordance, not an action, so it shouldn't behave like a button in
// any way beyond being themed/tinted like one.
QToolButton* MainWindow::createInfoIcon(const QString &helpText) {
	QToolButton *icon = new QToolButton(this);
	icon->setAutoRaise(true);
	icon->setFixedSize(18, 18);
	icon->setIconSize(QSize(14, 14));
	icon->setFocusPolicy(Qt::NoFocus);
	icon->setCursor(Qt::WhatsThisCursor);
	// Role::Body, not Role::textMuted-as-initial-colour - retint()
	// (theme_switch.cpp) hardcodes Role::Body to p.textBody on every theme
	// switch, so initializing with anything else here would make the icon
	// visibly change shade the first time the user switches themes.
	icon_tint::apply(icon, ":/icons/info.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	icon->setToolTip(wrapTooltipHtml(helpText));
	return icon;
}

// Composite {QLabel, info icon} widget for use as a QFormLayout::addRow()
// label argument - addRow("text", field) only accepts a plain string, so a
// row that wants an info icon next to its label needs the label built as
// an actual widget instead. Tight spacing/zero margins so it reads as one
// unit ("Samples per Pixel: (i)") rather than two separately-spaced items.
QWidget* MainWindow::labelWithInfo(const QString &labelText, const QString &helpText) {
	QWidget *container = new QWidget(this);
	QHBoxLayout *layout = new QHBoxLayout(container);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(4);
	layout->addWidget(new QLabel(labelText, container));
	layout->addWidget(createInfoIcon(helpText));
	layout->addStretch();
	return container;
}

// Same row shape as labelWithInfo() above, for the label-less checkbox rows
// on the Render Options tab (checkBox already carries its own label text, so
// there's no separate QLabel to build) - addRow(checkboxWithInfo(...)) in
// place of addRow(checkBox).
QWidget* MainWindow::checkboxWithInfo(QCheckBox *checkBox, const QString &helpText) {
	QWidget *row = new QWidget(this);
	QHBoxLayout *rowLayout = new QHBoxLayout(row);
	rowLayout->setContentsMargins(0, 0, 0, 0);
	rowLayout->setSpacing(4);
	rowLayout->addWidget(checkBox);
	rowLayout->addWidget(createInfoIcon(helpText));
	rowLayout->addStretch();
	return row;
}

// Builds the technique-note tooltip for `sceneId`, shared by every widget
// that shows one (this file's updateSceneTechInfoIcon() below, and
// mainwindow_tabs.cpp's populateSceneCombo()/populateSceneGrid()) so the
// choice of "heading + note, or just the note" lives in one place instead
// of each call site independently deciding how to combine them.
QString MainWindow::sceneTooltipHtml(const QString &sceneId, bool includeHeading) {
	const QString note = scene_technique_notes::forScene(sceneId);
	if (!includeHeading) return wrapTooltipHtml(note);
	return wrapTooltipHtml(
		QString("[%1] %2\n\n%3").arg(sceneId, SceneMetadataClient::sceneName(sceneId), note));
}

// Rewrites m_sceneTechInfoIcon's tooltip for the given scene - the one
// standalone info icon in this app whose content changes after construction
// instead of being fixed for its whole lifetime (see scene_technique_notes.h
// for the per-scene text; the scene combo/grid in mainwindow_tabs.cpp show
// the same text per-row instead, via sceneTooltipHtml() above). Routed
// through here rather than called directly from refreshSceneInfoLabel()
// (mainwindow_slots.cpp) so the "which widget does this icon belong to"
// knowledge stays in one place.
void MainWindow::updateSceneTechInfoIcon(const QString &sceneId) {
	if (!m_sceneTechInfoIcon) return;
	m_sceneTechInfoIcon->setToolTip(sceneTooltipHtml(sceneId, /*includeHeading=*/false));
}

// Plain-text (not HTML-wrapped) description of what `mode` does - used by
// createBasicTab()'s populateSceneCombo()-style per-item tooltips (one
// fixed entry per combo row, mirroring how the Scene combo gives each of
// its own rows an "(i)" icon + tooltip via icon_tint::addItem()/
// setItemData(Qt::ToolTipRole)).
// Held inline (not an external file like scene_technique_notes.h) since
// there are only 9 fixed entries, one per IntegratorMode value, rather
// than a per-scene lookup keyed by an open-ended id. Content mirrors the
// CLI's own --help text (launcher/launcher_args.h) - scope caveats
// (CPU-only, area-lights-only, verification status) included since those
// are exactly what a user picking an integrator needs to know before
// relying on it.
QString MainWindow::integratorDescription(IntegratorMode mode) {
	QString text;
	switch (mode) {
		case IntegratorMode::Default:
			text = tr("The general-purpose importance-sampled path tracer used "
			"everywhere else in this app - next-event estimation plus BSDF "
			"importance sampling, combined via the power heuristic (MIS). "
			"The well-tested default; start here unless you have a specific "
			"reason not to.\n\n"
			"The alternates below trade that generality for a specific "
			"technique - photon mapping, bidirectional/Metropolis light "
			"transport, or a handful of unbiased reference and debug "
			"integrators. All are CPU-only except SPPM, and none can be "
			"combined with Generate Video mode. Sampler/Spectral/Exposure/"
			"Tonemap/Stats above only affect this default Path Tracer.");
			break;
		case IntegratorMode::Sppm:
			text = tr("Stochastic Progressive Photon Mapping (pbrt-v4 style). "
			"Best for hard caustic/glass scenes ordinary path tracing "
			"struggles to resolve.\n\n"
			"CPU: verified end-to-end on the Cornell Rough Glass scene; "
			"other scenes are unverified and only support lambertian + "
			"delta-BSDF materials.\n\n"
			"GPU: capability-checked per scene - Lambertian/DiffuseLight, "
			"RoughDielectric, Metal, Dielectric, Conductor, RoughMetal, and "
			"DiffuseTransmission are supported (area lights only); an "
			"unsupported scene falls back to an error - use CPU SPPM "
			"instead.");
			break;
		case IntegratorMode::Bdpt:
			text = tr("Bidirectional Path Tracing - traces subpaths from both "
			"the camera and the light source and connects every pair, "
			"better for some difficult lighting configurations path "
			"tracing alone struggles with.\n\n"
			"CPU only. Area lights only (no punctual/sky-light NEE yet). "
			"Verified end-to-end on the Cornell Box scene only; other "
			"scenes are unverified.");
			break;
		case IntegratorMode::Mlt:
			text = tr("Metropolis Light Transport, built directly on BDPT's "
			"subpath machinery - uses a Markov chain to concentrate "
			"samples on light paths that already contribute, useful for "
			"scenes with hard-to-find bright paths.\n\n"
			"CPU only. Same area-lights-only scope and single-scene "
			"(Cornell Box) verification as BDPT.");
			break;
		case IntegratorMode::RandomWalk:
			text = tr("pbrt-v4's unbiased reference path tracer - "
			"uniform-sphere sampling, no next-event estimation or "
			"multiple importance sampling. Simpler and noisier than the "
			"default path tracer; useful as a ground-truth reference to "
			"check other integrators against.\n\n"
			"CPU only.");
			break;
		case IntegratorMode::Ao:
			text = tr("A visualization/debug mode, not a lit render - "
			"measures how occluded each point is by nearby geometry, "
			"ignoring material color and indirect lighting entirely.\n\n"
			"CPU only.");
			break;
		case IntegratorMode::SimplePath:
			text = tr("pbrt-v4's canonical reference path tracer - optional "
			"next-event estimation and optional BSDF importance sampling, "
			"both on by default (see the toggles below).\n\n"
			"CPU only. NEE, when enabled, is area-lights-only, the same "
			"scope as BDPT/MLT.");
			break;
		case IntegratorMode::SimpleVolPath:
			text = tr("pbrt-v4's simplest volumetric path tracer - pure "
			"delta tracking, no NEE/MIS/surface BSDFs.\n\n"
			"Reachable but medium-free in this integration, so it renders "
			"mostly black on ordinary solid-geometry scenes except where "
			"a camera ray lands directly on a light - matches pbrt-v4's "
			"own upstream behavior on medium-free scenes.\n\n"
			"CPU only.");
			break;
		case IntegratorMode::LightPath:
			text = tr("A pure light tracer - the opposite direction of "
			"every other integrator here: every sample starts at a light "
			"and splats camera-connection contributions into the film, "
			"instead of starting at the camera.\n\n"
			"CPU only. Area lights only.");
			break;
	}
	return text;
}

// Mirrors RenderController::start()'s own "only emit if it differs from
// default" conditions (mainwindow.cpp ~line 205-211) field-for-field, so
// a displayed setting can never contradict what was actually passed to
// the CLI for this render. Keep both in sync if either gains/loses a
// field.
QString MainWindow::advancedFlagsSummary(const AdvancedRenderFlags &flags) {
	QStringList parts;
	if (flags.denoise) parts << tr("Denoiser: on");
	if (flags.stats) parts << tr("Stats: on");
	if (flags.optixValidate) parts << tr("OptiX validation: on");
	if (flags.exposure != 1.0) parts << tr("Exposure: %1").arg(flags.exposure);
	if (!flags.sampler.isEmpty()) parts << tr("Sampler: %1").arg(flags.sampler);
	if (flags.spectral) parts << tr("Spectral: on");
	if (!flags.tonemap.isEmpty()) parts << tr("Tonemap: %1").arg(flags.tonemap);
	return parts.join(" · ");
}

// Same "only if non-default" mirroring as advancedFlagsSummary() above,
// against RenderController::start()'s switch(m_integratorOptions.mode)
// (mainwindow.cpp ~line 143-198) - keep both in sync.
QString MainWindow::integratorSettingsSummary(const IntegratorOptions &opts) {
	QStringList parts;
	switch (opts.mode) {
		case IntegratorMode::Sppm:
			if (opts.sppmIterations != 100) parts << tr("Iterations: %1").arg(opts.sppmIterations);
			if (opts.sppmPhotons != 5000) parts << tr("Photons/iter: %1").arg(opts.sppmPhotons);
			break;
		case IntegratorMode::Bdpt:
			if (opts.bdptMaxDepth != 5) parts << tr("Max depth: %1").arg(opts.bdptMaxDepth);
			break;
		case IntegratorMode::Mlt:
			if (opts.mltBootstrap != 100000) parts << tr("Bootstrap: %1").arg(opts.mltBootstrap);
			if (opts.mltMutations != 4000000) parts << tr("Mutations: %1").arg(opts.mltMutations);
			if (opts.mltMaxDepth != 5) parts << tr("Max depth: %1").arg(opts.mltMaxDepth);
			break;
		case IntegratorMode::Ao:
			// 'f', 2: same precision choice as start()'s own CLI-arg
			// formatting, for the same reason (default 'g' formatting can
			// silently round a large aoMaxDist value).
			if (opts.aoMaxDist != 1.0e10) parts << tr("Max distance: %1").arg(opts.aoMaxDist, 0, 'f', 2);
			if (opts.aoUniform) parts << tr("Uniform-hemisphere sampling");
			if (opts.aoIllumScale != 1.0) parts << tr("Illumination scale: %1").arg(opts.aoIllumScale);
			if (opts.aoIllumR != 1.0 || opts.aoIllumG != 1.0 || opts.aoIllumB != 1.0)
				parts << tr("Occlusion color: (%1, %2, %3)").arg(opts.aoIllumR).arg(opts.aoIllumG).arg(opts.aoIllumB);
			break;
		case IntegratorMode::SimplePath:
			if (opts.simplepathNoLights) parts << tr("NEE disabled");
			if (opts.simplepathNoBsdf) parts << tr("BSDF importance sampling disabled");
			break;
		case IntegratorMode::Default:
		case IntegratorMode::RandomWalk:
		case IntegratorMode::SimpleVolPath:
		case IntegratorMode::LightPath:
			break;
	}
	return parts.join(" · ");
}

// The Preview tab's technique-box HTML (see updatePreviewSidebarForActiveTab(),
// mainwindow_tabs.cpp) - the integrator's own description, plus whichever
// settings were actually customized for this specific render. Computed
// once per tab at creation time (see PreviewTechniqueInfo, mainwindow.h),
// not live, since a completed render's settings never change afterward.
QString MainWindow::renderTechniqueHtml(const IntegratorOptions &integratorOptions, const AdvancedRenderFlags &advancedFlags) {
	QString html = tr("<b>Rendering technique</b><br>%1")
		.arg(plainTextToHtmlParagraphs(integratorDescription(integratorOptions.mode)));

	QStringList settingsParts;
	const QString flagsSummary = advancedFlagsSummary(advancedFlags);
	if (!flagsSummary.isEmpty()) settingsParts << flagsSummary;
	const QString integratorSummary = integratorSettingsSummary(integratorOptions);
	if (!integratorSummary.isEmpty()) settingsParts << integratorSummary;

	if (!settingsParts.isEmpty()) {
		html += tr("<br><br><b>Settings used</b><br>%1").arg(settingsParts.join(" · "));
	}
	return html;
}

