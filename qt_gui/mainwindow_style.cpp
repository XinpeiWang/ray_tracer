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

// Text drawn on a selected row sits on accentDim, so it has to contrast with
// THAT, not with the window. White is right for every dark scheme and for
// mid-tone accents; only a genuinely pale accent needs dark text instead. The
// threshold is high because white-on-mid-tone stays readable a good way up the
// range - flipping at the midpoint sent Solarized Light's violet accent to
// cream text, which was the worse of the two options.
QColor selectedRowText(const theme::Palette &p) {
	return p.accentDim.lightness() > 170 ? p.surface0 : QColor(Qt::white);
}

} // namespace

void MainWindow::applyTheme(const theme::Palette &p) {
	m_activeTheme = p;

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
	appPalette.setColor(QPalette::HighlightedText, selectedRowText(p));

	// Disabled state colors
	appPalette.setColor(QPalette::Disabled, QPalette::WindowText, p.textDisabled);
	appPalette.setColor(QPalette::Disabled, QPalette::Text, p.textDisabled);
	appPalette.setColor(QPalette::Disabled, QPalette::ButtonText, p.textDisabled);

	qApp->setPalette(appPalette);
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
			color: %ACCENT_1%;
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
			font-size: 13pt;
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
			border-bottom: 2px solid %ACCENT_2%;
			color: %ACCENT_2%;
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
			border: 1px solid %ACCENT_2%;
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
			selection-color: %SELECTED_TEXT%;
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
		.replace("%SURFACE0%",      hex(p.surface0))
		.replace("%SURFACE1%",      hex(p.surface1))
		.replace("%SURFACE2%",      hex(p.surface2))
		.replace("%SURFACE3%",      hex(p.surface3))
		.replace("%SELECTED_TEXT%", hex(selectedRowText(p)))
		.replace("%HOVER_ROW%",     hex(p.hoverRow))
		.replace("%TEXT%",          hex(p.textBody))
		.replace("%TEXT_MUTED%",    hex(p.textMuted))
		.replace("%TEXT_DISABLED%", hex(p.textDisabled))
		.replace("%ACCENT_2%",      hex(p.accentSecondary))
		.replace("%ACCENT_1%",      hex(p.accentPrimary))
		.replace("%ACCENT_DIM%",    hex(p.accentDim))
		.replace("%BORDER_STRONG%", hex(p.borderStrong))
		.replace("%BORDER%",        hex(p.border))
		.replace("%SUCCESS%",       hex(p.success))
		.replace("%ERROR%",         hex(p.error))
		.replace("%PRIMARY_TOP_HOVER%",    hex(p.primaryTopHover))
		.replace("%PRIMARY_BOTTOM_HOVER%", hex(p.primaryBottomHover))
		.replace("%PRIMARY_TOP%",          hex(p.primaryTop))
		.replace("%PRIMARY_BOTTOM%",       hex(p.primaryBottom))
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
		   full-width accent fill: this list is 65 scenes long, and a saturated
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

