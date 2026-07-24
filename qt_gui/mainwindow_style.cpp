#include "mainwindow.h"
#include "scene_descriptor.h"
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
			border: 2px solid #00FFFF;
			border-radius: 0px;
			margin-top: 20px;
			margin-bottom: 8px;
			padding: 20px 10px 12px 10px;
			background-color: #1A1A2E;
			color: #00FFFF;
			font-size: 12pt;
		}
		QGroupBox::title {
			subcontrol-origin: margin;
			subcontrol-position: top left;
			padding: 2px 12px;
			left: 12px;
			top: -12px;
			color: #FF00FF;
			font-size: 13pt;
			font-weight: bold;
			background-color: #1A1A2E;
		}
		QPushButton {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #3A1050, stop:1 #1E0832);
			border: 3px solid #FF00FF;
			border-radius: 8px;
			color: #FF00FF;
			padding: 12px 20px;
			font-weight: bold;
			font-size: 13pt;
			min-height: 45px;
		}
		QPushButton:hover {
			background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
				stop:0 #5A1570, stop:1 #3E1552);
			border: 3px solid #00FFFF;
			color: #00FFFF;
		}
		QPushButton:pressed {
			background-color: #2A0A40;
			border: 3px solid #C800FF;
		}
		QPushButton:disabled {
			background-color: #1A0A2A;
			border: 2px solid #503060;
			color: #503060;
		}
		QProgressBar {
			border: 3px solid #00FFFF;
			border-radius: 8px;
			text-align: center;
			background-color: #0A0A0F;
			color: #00FFFF;
			font-size: 12pt;
			min-height: 35px;
		}
		QProgressBar::chunk {
			background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
				stop:0 #FF00FF, stop:0.5 #C800FF, stop:1 #00FFFF);
			border-radius: 5px;
		}
		QTabWidget::pane {
			border: 3px solid #FF00FF;
			border-radius: 0px 8px 8px 8px;
			background-color: #0A0A0F;
			top: -3px;
		}
		QTabBar::tab {
			background-color: #1E0832;
			border: 3px solid #FF00FF;
			border-bottom: 3px solid #FF00FF;
			border-top-left-radius: 8px;
			border-top-right-radius: 8px;
			padding: 12px 20px;
			color: #FF00FF;
			font-size: 12pt;
			min-width: 100px;
			margin-right: 2px;
		}
		QTabBar::tab:selected {
			background-color: #3A1050;
			border-color: #00FFFF;
			border-bottom-color: #0A0A0F;
			color: #00FFFF;
			margin-bottom: -3px;
			padding-bottom: 15px;
		}
		QTabBar::tab:hover {
			background-color: #2A0A40;
			color: #00FFFF;
		}
		QSpinBox, QDoubleSpinBox, QComboBox {
			background-color: #0A0A0F;
			border: 3px solid #00FFFF;
			border-radius: 5px;
			padding: 8px;
			color: #00FFFF;
			font-size: 11pt;
			min-height: 30px;
			margin: 5px 2px;
		}
		QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
			background-color: #1A0A2A;
			border: 3px solid #FF00FF;
			color: #FF00FF;
		}
		QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
			background-color: #2A1040;
			border: 3px solid #C800FF;
			color: #FF00FF;
		}
		QSpinBox, QDoubleSpinBox {
			padding-right: 30px;
		}
		QComboBox::drop-down {
			border: none;
			padding-right: 5px;
		}
		QComboBox::down-arrow {
			image: none;
			border-left: 5px solid transparent;
			border-right: 5px solid transparent;
			border-top: 6px solid #FF00FF;
		}
		QComboBox::down-arrow:hover {
			border-top: 6px solid #00FFFF;
		}
		QComboBox QAbstractItemView {
			background-color: #0A0A0F;
			border: 3px solid #FF00FF;
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
			border: 3px solid #FF00FF;
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
			padding: 8px 5px;
			margin: 5px 2px;
		}
		QFormLayout {
			spacing: 15px;
		}
		QTextEdit {
			background-color: #0A0A0F;
			border: 3px solid #00FFFF;
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

void MainWindow::styleComboBox(QComboBox *combo) {
	QAbstractItemView *view = combo->view();

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

			void MainWindow::styleSpinBox(QSpinBox *spinBox) {
				// Apply custom stylesheet with CSS triangle arrows
				QString spinBoxStyle = R"(
					QSpinBox {
						padding-right: 30px;
						background-color: #0A0A0F;
						border: 3px solid #FF00FF;
						border-radius: 5px;
						color: #00FFFF;
						font-size: 12pt;
						padding: 8px;
					}
					QSpinBox::up-button {
						subcontrol-origin: border;
						subcontrol-position: top right;
						width: 28px;
						border-left: 2px solid #FF00FF;
						border-top: 2px solid #FF00FF;
						border-right: 2px solid #FF00FF;
						border-top-right-radius: 3px;
						background-color: #1E0832;
					}
					QSpinBox::down-button {
						subcontrol-origin: border;
						subcontrol-position: bottom right;
						width: 28px;
						border-left: 2px solid #FF00FF;
						border-bottom: 2px solid #FF00FF;
						border-right: 2px solid #FF00FF;
						border-bottom-right-radius: 3px;
						background-color: #1E0832;
					}
					QSpinBox::up-button:hover {
						background-color: #3A1050;
						border-color: #00FFFF;
					}
					QSpinBox::down-button:hover {
						background-color: #3A1050;
						border-color: #00FFFF;
					}
					QSpinBox::up-arrow {
						width: 0;
						height: 0;
						border-left: 5px solid transparent;
						border-right: 5px solid transparent;
						border-bottom: 7px solid #FF00FF;
					}
					QSpinBox::up-arrow:hover {
						border-bottom-color: #00FFFF;
					}
					QSpinBox::down-arrow {
						width: 0;
						height: 0;
						border-left: 5px solid transparent;
						border-right: 5px solid transparent;
						border-top: 7px solid #FF00FF;
					}
					QSpinBox::down-arrow:hover {
						border-top-color: #00FFFF;
					}
				)";
				spinBox->setStyleSheet(spinBoxStyle);
				spinBox->installEventFilter(m_wheelFilter);
			}

