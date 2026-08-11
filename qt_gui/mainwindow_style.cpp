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


void MainWindow::applyDarkTheme() {
	// Cyberpunk theme with neon colors
	QPalette cyberpunkPalette;
	cyberpunkPalette.setColor(QPalette::Window, QColor(10, 10, 15));           // Deep black with subtle blue
	cyberpunkPalette.setColor(QPalette::WindowText, QColor(0, 255, 255));      // Bright cyan neon
	cyberpunkPalette.setColor(QPalette::Base, QColor(5, 5, 10));               // Almost pure black
	cyberpunkPalette.setColor(QPalette::AlternateBase, QColor(15, 10, 20));    // Dark purple tint
	cyberpunkPalette.setColor(QPalette::ToolTipBase, QColor(20, 0, 30));       // Dark purple
	cyberpunkPalette.setColor(QPalette::ToolTipText, QColor(255, 0, 255));     // Magenta neon
	cyberpunkPalette.setColor(QPalette::Text, QColor(0, 255, 255));            // Cyan neon
	cyberpunkPalette.setColor(QPalette::Button, QColor(30, 15, 50));           // Deep purple
	cyberpunkPalette.setColor(QPalette::ButtonText, QColor(255, 0, 255));      // Magenta neon
	cyberpunkPalette.setColor(QPalette::BrightText, QColor(255, 255, 0));      // Yellow neon
	cyberpunkPalette.setColor(QPalette::Link, QColor(0, 200, 255));            // Bright blue
	cyberpunkPalette.setColor(QPalette::Highlight, QColor(200, 0, 255));       // Purple-pink highlight
	cyberpunkPalette.setColor(QPalette::HighlightedText, QColor(255, 255, 255)); // White

	// Disabled state colors
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::WindowText, QColor(60, 60, 80));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(60, 60, 80));
	cyberpunkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(80, 40, 100));

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
	cyberpunkFont.setPointSize(11);  // Increased from 10
	cyberpunkFont.setWeight(QFont::Bold);
	cyberpunkFont.setLetterSpacing(QFont::AbsoluteSpacing, 0.8);
	qApp->setFont(cyberpunkFont);

	// Apply cyberpunk stylesheet for enhanced neon effects
	QString stylesheet = R"(
		QGroupBox {
			border: 1px solid #2E5A5E;
			border-radius: 10px;
			margin-top: 26px;
			margin-bottom: 6px;
			padding: 4px 10px 10px 10px;
			background-color: #14141F;
			color: #00FFFF;
			font-size: 12pt;
		}
		QGroupBox::title {
			subcontrol-origin: margin;
			subcontrol-position: top left;
			padding: 2px 12px;
			left: 12px;
			top: 2px;
			color: #FF00FF;
			font-size: 13pt;
			font-weight: bold;
			background-color: #14141F;
		}
		QPushButton {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #3A1050, stop:1 #1E0832);
			border: 2px solid #C93FE8;
			border-radius: 8px;
			color: #FF00FF;
			padding: 10px 20px;
			font-weight: bold;
			font-size: 13pt;
			min-height: 42px;
		}
		QPushButton:hover {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #5A1570, stop:1 #3E1552);
			border: 2px solid #00FFFF;
			color: #00FFFF;
		}
		QPushButton:pressed {
			background-color: #2A0A40;
			border: 2px solid #C800FF;
		}
		QPushButton:disabled {
			background-color: #1A0A2A;
			border: 2px solid #503060;
			color: #503060;
		}
		QProgressBar {
			border: 2px solid #2E5A5E;
			border-radius: 8px;
			text-align: center;
			background-color: #0A0A0F;
			color: #00FFFF;
			font-size: 12pt;
			min-height: 32px;
		}
		QProgressBar::chunk {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 #FF00FF, stop:0.5 #C800FF, stop:1 #00FFFF);
			border-radius: 5px;
		}
		QTabWidget::pane {
			border: 2px solid #C93FE8;
			border-radius: 0px 8px 8px 8px;
			background-color: #0A0A0F;
			top: -2px;
		}
		QTabBar::tab {
			background-color: #1E0832;
			border: 2px solid #5A3A6E;
			border-bottom: 2px solid #C93FE8;
			border-top-left-radius: 8px;
			border-top-right-radius: 8px;
			padding: 10px 18px;
			color: #D090E8;
			font-size: 12pt;
			min-width: 100px;
			margin-right: 2px;
		}
		QTabBar::tab:selected {
			background-color: #3A1050;
			border-color: #00FFFF;
			border-bottom-color: #0A0A0F;
			color: #00FFFF;
			margin-bottom: -2px;
			padding-bottom: 12px;
		}
		QTabBar::tab:hover {
			background-color: #2A0A40;
			color: #00FFFF;
			border-color: #FF00FF;
		}
		QSpinBox, QDoubleSpinBox, QComboBox {
			background-color: #0A0A0F;
			border: 2px solid #2E5A5E;
			border-radius: 5px;
			padding: 6px 8px;
			color: #00FFFF;
			font-size: 11pt;
			min-height: 26px;
			margin: 3px 2px;
		}
		/* Fusion splits a spin box's contents-rect height in half for its
		   up/down step buttons, so at the shared 26px min-height above
		   (~11px per button once the 2px border is subtracted) the
		   PlusMinus primitives' cross-bars are proportionally too thick
		   for the space: the "+" blobs into a diamond and only the "-"
		   reads cleanly (see styleSpinBox()'s comment for why PlusMinus
		   is used at all). Taller spin boxes only, so each button gets
		   room for a legible glyph; the combo box keeps the 26px height. */
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
		QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
			background-color: #1A0A2A;
			border: 2px solid #FF00FF;
			color: #FF00FF;
		}
		QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
			background-color: #2A1040;
			border: 2px solid #C800FF;
			color: #FF00FF;
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
			background-color: #0A0A0F;
			border: 2px solid #FF00FF;
			border-radius: 5px;
			selection-background-color: #FF00FF;
			selection-color: #000000;
			color: #00FFFF;
			outline: none;
			padding: 2px;
		}
		QComboBox QAbstractItemView::item {
			padding: 8px;
			min-height: 30px;
			border: 1px solid transparent;
			background-color: transparent;
			color: #00FFFF;
		}
		QComboBox QAbstractItemView::item:hover {
			background-color: #5A1570;
			border: 1px solid #FF00FF;
			color: #FF00FF;
		}
		QComboBox QAbstractItemView::item:selected {
			background-color: #FF00FF;
			color: #000000;
			border: 1px solid #00FFFF;
		}
		QComboBox QAbstractItemView::item:selected:hover {
			background-color: #00FFFF;
			color: #000000;
			border: 1px solid #FF00FF;
		}
		QListView {
			background-color: #0A0A0F;
			border: 2px solid #FF00FF;
			color: #00FFFF;
		}
		QListView::item {
			padding: 8px;
			min-height: 30px;
		}
		QListView::item:hover {
			background-color: #5A1570;
			color: #FF00FF;
		}
		QListView::item:selected {
			background-color: #FF00FF;
			color: #000000;
		}
		QListView::item:selected:hover {
			background-color: #00FFFF;
			color: #000000;
		}
		QLabel {
			color: #00FFFF;
			font-size: 11pt;
			padding: 4px 5px;
			margin: 3px 2px;
		}
		QFormLayout {
			spacing: 10px;
		}
		QTextEdit {
			background-color: #0A0A0F;
			border: 2px solid #2E5A5E;
			border-radius: 5px;
			color: #00FFFF;
			selection-background-color: #C800FF;
			font-size: 10pt;
			padding: 5px;
		}
		QScrollBar:vertical {
			background-color: #0A0A0F;
			width: 16px;
			margin: 0px;
			border: 2px solid #00FFFF;
			border-radius: 8px;
		}
		QScrollBar::handle:vertical {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 #FF00FF, stop:1 #C800FF);
			border-radius: 6px;
			min-height: 30px;
			margin: 2px;
		}
		QScrollBar::handle:vertical:hover {
			background-color: #00FFFF;
		}
		QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
			height: 0px;
		}
		QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
			background: none;
		}
		QScrollBar:horizontal {
			background-color: #0A0A0F;
			height: 16px;
			margin: 0px;
			border: 2px solid #00FFFF;
			border-radius: 8px;
		}
		QScrollBar::handle:horizontal {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #FF00FF, stop:1 #C800FF);
			border-radius: 6px;
			min-width: 30px;
			margin: 2px;
		}
		QScrollBar::handle:horizontal:hover {
			background-color: #00FFFF;
		}
		QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
			width: 0px;
		}
		QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
			background: none;
		}
	)";
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
	// the screen - matters most for m_sceneCombo (51 scenes and counting);
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

	view->setStyleSheet(R"(
		QAbstractItemView {
			background-color: #0A0A0F;
			border: 3px solid #FF00FF;
			border-radius: 5px;
			outline: none;
			color: #00FFFF;
			padding: 2px;
		}
		QAbstractItemView::item {
			padding: 8px 12px;
			min-height: 32px;
			border: none;
			color: #00FFFF;
		}
		QAbstractItemView::item:hover {
			background-color: #5A1570;
			color: #FF00FF;
			border-left: 3px solid #FF00FF;
		}
		QAbstractItemView::item:selected {
			background-color: #FF00FF;
			color: #000000;
		}
		QAbstractItemView::item:selected:hover {
			background-color: #00FFFF;
			color: #000000;
		}
	)");

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

