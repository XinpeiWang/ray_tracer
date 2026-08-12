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
#include <QDateTime>
#include <QScrollArea>
#include <QScreen>
#include <QTimer>
#include <QAbstractItemView>


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

// Surface ramp - darkest (window) to lightest (raised controls)
constexpr const char *kSurface0     = "#0E0E14";  // Window / deepest background
constexpr const char *kSurface1     = "#16161F";  // Panels, group boxes, inputs
constexpr const char *kSurface2     = "#1E1E2A";  // Raised / hovered controls
constexpr const char *kSurface3     = "#2A2A3A";  // Pressed / selected controls

// Text
constexpr const char *kTextBody     = "#E6E6F0";  // Default readable text
constexpr const char *kTextMuted    = "#9A9AB0";  // Secondary / helper text
constexpr const char *kTextDisabled = "#5A5A70";

// Accents (small doses only)
constexpr const char *kAccentCyan   = "#00E5FF";  // Focus, active tab, headings
constexpr const char *kAccentMag    = "#FF3DFF";  // Primary action, titles
constexpr const char *kAccentDim    = "#C93FE8";  // Primary border at rest

// Lines
constexpr const char *kBorder       = "#2A2A3A";  // Ordinary 1px chrome
constexpr const char *kBorderStrong = "#3A3A50";  // Interactive controls

// Hover fill for list/menu rows. Deliberately purple-tinted and a clear step
// lighter than kSurface3 - a plain neutral hover at this contrast level is
// nearly invisible against kSurface1, which loses the "this row is under the
// cursor" feedback the old high-saturation theme did give.
constexpr const char *kHoverRow     = "#3A2E56";

// Radius scale - two values, not five
constexpr const char *kRadius       = "6px";      // Controls
constexpr const char *kRadiusLarge  = "10px";     // Containers

} // namespace

void MainWindow::applyDarkTheme() {
	// Cyberpunk theme with neon colors
	QPalette cyberpunkPalette;
	cyberpunkPalette.setColor(QPalette::Window, QColor(0x0E, 0x0E, 0x14));
	cyberpunkPalette.setColor(QPalette::WindowText, QColor(0xE6, 0xE6, 0xF0));
	cyberpunkPalette.setColor(QPalette::Base, QColor(0x16, 0x16, 0x1F));
	cyberpunkPalette.setColor(QPalette::AlternateBase, QColor(0x1E, 0x1E, 0x2A));
	cyberpunkPalette.setColor(QPalette::ToolTipBase, QColor(0x1E, 0x1E, 0x2A));
	cyberpunkPalette.setColor(QPalette::ToolTipText, QColor(0xE6, 0xE6, 0xF0));
	cyberpunkPalette.setColor(QPalette::Text, QColor(0xE6, 0xE6, 0xF0));
	cyberpunkPalette.setColor(QPalette::Button, QColor(0x1E, 0x1E, 0x2A));
	cyberpunkPalette.setColor(QPalette::ButtonText, QColor(0xE6, 0xE6, 0xF0));
	cyberpunkPalette.setColor(QPalette::BrightText, QColor(0xFF, 0x3D, 0xFF));
	cyberpunkPalette.setColor(QPalette::Link, QColor(0x00, 0xE5, 0xFF));
	cyberpunkPalette.setColor(QPalette::Highlight, QColor(0xC9, 0x3F, 0xE8));
	cyberpunkPalette.setColor(QPalette::HighlightedText, QColor(0xFF, 0xFF, 0xFF));

	// Disabled state colors
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::WindowText, QColor(0x5A, 0x5A, 0x70));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(0x5A, 0x5A, 0x70));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(0x5A, 0x5A, 0x70));

	qApp->setPalette(cyberpunkPalette);
	qApp->setStyle(QStyleFactory::create("Fusion"));

	// Set cyberpunk-style font
	QFont cyberpunkFont;
	// Try futuristic/tech fonts, fallback to system fonts
	QStringList fontFamilies = {"Orbitron", "Rajdhani", "Exo 2", "Michroma", "Audiowide",
								 "Chakra Petch", "Saira", "Teko", "Electrolize",
								 "Bahnschrift", "Segoe UI", "Arial"};
	bool fontSet = false;
	for (const QString& fontFamily : fontFamilies) {
		cyberpunkFont.setFamily(fontFamily);
		if (QFontInfo(cyberpunkFont).family() == fontFamily) {
			fontSet = true;
			break;
		}
	}
	if (!fontSet) {
		cyberpunkFont.setFamily("Arial");
	}
	cyberpunkFont.setPointSize(11);
	// Normal weight app-wide: bold was previously applied to EVERY widget,
	// which meant nothing was actually emphasised. Bold is now opt-in, on
	// group-box titles and the primary button only. Letter-spacing is also
	// gone from body text (it belongs on short display strings, not on log
	// lines and labels, where it measurably slows reading).
	cyberpunkFont.setWeight(QFont::Normal);
	qApp->setFont(cyberpunkFont);

	// Apply cyberpunk stylesheet for enhanced neon effects
	QString stylesheet = QString(R"(
		QGroupBox {
			border: 1px solid %BORDER%;
			border-radius: %RADIUS_LG%;
			margin-top: 26px;
			margin-bottom: 6px;
			padding: 4px 10px 10px 10px;
			background-color: %SURFACE1%;
			color: %TEXT%;
			font-size: 12pt;
		}
		QGroupBox::title {
			subcontrol-origin: margin;
			subcontrol-position: top left;
			padding: 2px 12px;
			left: 12px;
			top: 2px;
			color: %ACCENT_MAG%;
			font-size: 12pt;
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
			font-size: 12pt;
			min-height: 34px;
		}
		QPushButton:hover {
			background-color: %SURFACE3%;
			border-color: %ACCENT_CYAN%;
			color: %ACCENT_CYAN%;
		}
		QPushButton:pressed {
			background-color: %SURFACE1%;
		}
		QPushButton:focus {
			border: 1px solid %ACCENT_CYAN%;
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
				stop:0 #3A1050, stop:1 #240C38);
			border: 2px solid %ACCENT_DIM%;
			color: %ACCENT_MAG%;
			font-weight: bold;
			font-size: 13pt;
		}
		QPushButton#primaryAction:hover {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #4E1668, stop:1 #341048);
			border-color: %ACCENT_CYAN%;
			color: %ACCENT_CYAN%;
		}
		QPushButton#primaryAction:pressed {
			background-color: #240C38;
		}
		QPushButton#primaryAction:disabled {
			background-color: %SURFACE1%;
			border: 2px solid %BORDER%;
			color: %TEXT_DISABLED%;
		}
		QProgressBar {
			border: 1px solid %BORDER%;
			border-radius: %RADIUS%;
			text-align: center;
			background-color: %SURFACE0%;
			color: %TEXT%;
			font-size: 11pt;
			min-height: 28px;
		}
		QProgressBar::chunk {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 %ACCENT_MAG%, stop:1 %ACCENT_CYAN%);
			border-radius: 5px;
		}
		QTabWidget::pane {
			border: 1px solid %BORDER%;
			border-radius: 0px %RADIUS% %RADIUS% %RADIUS%;
			background-color: %SURFACE0%;
			top: -1px;
		}
		QTabBar::tab {
			background-color: transparent;
			border: 1px solid transparent;
			border-bottom: 1px solid %BORDER%;
			border-top-left-radius: %RADIUS%;
			border-top-right-radius: %RADIUS%;
			padding: 10px 18px;
			color: %TEXT_MUTED%;
			font-size: 12pt;
			min-width: 100px;
			margin-right: 2px;
		}
		/* The selected tab is marked by an accent underline rather than a
		   full glowing outline - the tab strip is navigation, not the
		   loudest thing on screen. */
		QTabBar::tab:selected {
			background-color: %SURFACE0%;
			border-color: %BORDER%;
			border-bottom: 2px solid %ACCENT_CYAN%;
			color: %ACCENT_CYAN%;
			font-weight: bold;
		}
		QTabBar::tab:hover:!selected {
			background-color: %SURFACE1%;
			color: %TEXT%;
		}
		QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
			background-color: %SURFACE1%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: %RADIUS%;
			padding: 6px 8px;
			color: %TEXT%;
			font-size: 11pt;
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
			border: 1px solid %ACCENT_CYAN%;
			color: %TEXT%;
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
			selection-color: #FFFFFF;
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
			color: %ACCENT_CYAN%;
		}
		QComboBox QAbstractItemView::item:selected {
			background-color: %ACCENT_DIM%;
			color: #FFFFFF;
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
			color: %ACCENT_CYAN%;
		}
		QListView::item:selected {
			background-color: %ACCENT_DIM%;
			color: #FFFFFF;
		}
		QLabel {
			color: %TEXT%;
			font-size: 11pt;
			padding: 4px 5px;
			margin: 3px 2px;
			background: transparent;
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
			selection-color: #FFFFFF;
			font-size: 10pt;
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
		QToolTip {
			background-color: %SURFACE2%;
			border: 1px solid %BORDER_STRONG%;
			border-radius: 4px;
			color: %TEXT%;
			padding: 4px 8px;
		}
	)")
		.replace("%SURFACE0%",      kSurface0)
		.replace("%SURFACE1%",      kSurface1)
		.replace("%SURFACE2%",      kSurface2)
		.replace("%SURFACE3%",      kSurface3)
		.replace("%HOVER_ROW%",     kHoverRow)
		.replace("%TEXT%",          kTextBody)
		.replace("%TEXT_MUTED%",    kTextMuted)
		.replace("%TEXT_DISABLED%", kTextDisabled)
		.replace("%ACCENT_CYAN%",   kAccentCyan)
		.replace("%ACCENT_MAG%",    kAccentMag)
		.replace("%ACCENT_DIM%",    kAccentDim)
		.replace("%BORDER_STRONG%", kBorderStrong)
		.replace("%BORDER%",        kBorder)
		.replace("%RADIUS_LG%",     kRadiusLarge)
		.replace("%RADIUS%",        kRadius);

	qApp->setStyleSheet(stylesheet);
}

void MainWindow::styleGroupBox(QGroupBox *box) {
	// WA_StyledBackground forces Qt to paint the CSS background+border
	// on QGroupBox (which by default ignores background-color in stylesheets).
	box->setAttribute(Qt::WA_StyledBackground, true);
}

void MainWindow::styleComboBox(QComboBox *combo) {
	QAbstractItemView *view = combo->view();

	// Cap the popup's visible rows so it scrolls instead of growing past
	// the screen - matters most for m_sceneCombo (65 scenes and counting);
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

	// Force Fusion style on the popup so Qt honours the stylesheet
	// instead of deferring to the Windows native list-box renderer.
	view->setStyle(QStyleFactory::create("Fusion"));

	view->setMouseTracking(true);
	view->viewport()->setMouseTracking(true);
	view->setAttribute(Qt::WA_Hover, true);
	view->viewport()->setAttribute(Qt::WA_Hover, true);

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
		/* A 3px accent bar on the left edge marks the hovered row instead of
		   recolouring the whole row - the reserved-transparent border above
		   keeps the text from shifting sideways when it appears. */
		QAbstractItemView::item:hover {
			background-color: %HOVER_ROW%;
			color: %ACCENT_CYAN%;
			border-left: 3px solid %ACCENT_CYAN%;
		}
		QAbstractItemView::item:selected {
			background-color: %ACCENT_DIM%;
			color: #FFFFFF;
			border-left: 3px solid %ACCENT_MAG%;
		}
	)")
		.replace("%SURFACE1%",      kSurface1)
		.replace("%SURFACE3%",      kSurface3)
		.replace("%HOVER_ROW%",     kHoverRow)
		.replace("%TEXT%",          kTextBody)
		.replace("%ACCENT_CYAN%",   kAccentCyan)
		.replace("%ACCENT_MAG%",    kAccentMag)
		.replace("%ACCENT_DIM%",    kAccentDim)
		.replace("%BORDER_STRONG%", kBorderStrong)
		.replace("%RADIUS%",        kRadius));

	combo->installEventFilter(m_wheelFilter);
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

