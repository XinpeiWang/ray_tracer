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

#include "camera_math.h"
#include "render_output_parser.h"
#include "theme.h"
#include "icon_tint.h"

class QMenu;
class QTabBar;

// ============================================================================
// WheelIgnoreFilter
// ============================================================================
// Prevents accidental value changes when the user scrolls the page.
// Install on any QComboBox / QSpinBox / QDoubleSpinBox to block wheel events
// unless the widget already has keyboard focus.
// ============================================================================
class WheelIgnoreFilter : public QObject {
    Q_OBJECT
public:
    explicit WheelIgnoreFilter(QObject *parent = nullptr) : QObject(parent) {}
protected:
    bool eventFilter(QObject *obj, QEvent *event) override {
        if (event->type() == QEvent::Wheel)
            return true;  // always block wheel on these controls
        return QObject::eventFilter(obj, event);
    }
};

// ============================================================================
// ScaledImageLabel
// ============================================================================
// A QLabel that shows an image scaled to fit its current size (preserving
// aspect ratio, re-scaled on every resize) instead of QLabel's default
// native-size-or-nothing behavior. Used by the Preview tab to show the
// rendered image inline - see MainWindow::createPreviewTab() - instead of
// shelling out to the OS's default image viewer for every render.
// ============================================================================
class ScaledImageLabel : public QLabel {
	Q_OBJECT
public:
	explicit ScaledImageLabel(QWidget *parent = nullptr) : QLabel(parent) {
		setAlignment(Qt::AlignCenter);
		setMinimumSize(1, 1);
	}

	void setPreviewPixmap(const QPixmap &pixmap) {
		m_original = pixmap;
		updateScaledPixmap();
	}

	// Drops the currently displayed image (if any) and restores whatever
	// text was last set via setPlaceholderText().
	void clearPreviewPixmap() {
		m_original = QPixmap();
		setPixmap(QPixmap());
		setText(m_placeholderText);
	}

	void setPlaceholderText(const QString &text) {
		m_placeholderText = text;
		if (m_original.isNull()) setText(text);
	}

protected:
	void resizeEvent(QResizeEvent *event) override {
		QLabel::resizeEvent(event);
		updateScaledPixmap();
	}

private:
	void updateScaledPixmap() {
		if (m_original.isNull()) return;
		setPixmap(m_original.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}

	QPixmap m_original;
	QString m_placeholderText;
};

// ============================================================================
// HorizontalTabBar / SplitPreviewTabs
// ============================================================================
// A West-positioned (left-side) QTabBar whose labels stay upright/horizontal
// instead of Qt's default rendering for that position, which rotates the
// text 90 degrees to read top-to-bottom - readable for a handful of tabs,
// but not what "tabs on the left" means for the Preview tab's per-render
// sub-tabs (createPreviewTab()), which are titled with real scene/preset
// names.
//
// The label is painted by hand (setColors() + paintEvent() below) rather
// than by counter-rotating the painter and calling the style's own
// CE_TabBarTabLabel - that is the textbook technique for this, but with a
// stylesheet active it draws nothing: QStyleSheetStyle's CE_TabBarTabLabel
// matches each tab against its own cached geometry to pick a rule, and this
// bar's very own transposed opt.rect (needed for the rotation) is exactly
// what breaks that match, silently producing invisible (not just
// mis-rotated) text. Drawing the shape via CE_TabBarTabShape still works
// fine - only the label needs the hand-rolled path - so setColors() is fed
// the same three colours the (still QSS-driven) shape rule already keys
// off of (mainwindow_style.cpp's own QTabBar#previewSubTabsBar::tab rule),
// applied by MainWindow::applyTheme() whenever the scheme changes.
//
// tabSizeHint() still double-transposes for the same reason it did with the
// rotation approach: QCommonStyle already transposes its own size hint once
// for a West-shaped bar (narrow width, height sized to fit the label
// lengthwise), and undoing that gets back a normal horizontal-tab footprint
// (width sized to the label, one line tall) arranged in a vertical stack
// instead of a horizontal row - independent of how the label itself ends up
// painted.
class HorizontalTabBar : public QTabBar {
	Q_OBJECT
public:
	explicit HorizontalTabBar(QWidget *parent = nullptr) : QTabBar(parent) {}

	// Kept in sync with the active theme's palette by MainWindow::
	// applyTheme() - see this class's own comment on why the label can't
	// just read colours out of the stylesheet the way the shape does.
	void setColors(const QColor &normal, const QColor &hover, const QColor &selected) {
		m_normalColor = normal;
		m_hoverColor = hover;
		m_selectedColor = selected;
		update();
	}

signals:
	// Emitted on a left-click inside a tab's close glyph. Connected to
	// MainWindow::closePreviewSubTab() by createPreviewTab() - a plain
	// signal/slot connection, in place of the QTabBar::tabCloseRequested a
	// normal closable tab bar would emit, since this bar manages its close
	// affordance by hand (see closeGlyphRect()'s own comment on why).
	void closeRequested(int index);

protected:
	QSize tabSizeHint(int index) const override {
		QSize s = QTabBar::tabSizeHint(index);
		s.transpose();
		return s;
	}

	void paintEvent(QPaintEvent * /*event*/) override {
		QStylePainter painter(this);
		for (int i = 0; i < count(); ++i) {
			QStyleOptionTab opt;
			initStyleOption(&opt, i);
			// Widened to the bar's actual current width rather than Qt's own
			// (natural, text-width-only) tabRect(i) - stretchedTabRect()'s
			// own comment explains why this can't be done by feeding
			// width() back through tabSizeHint() instead. Background/border
			// only here - QSS-aware (QTabBar#previewSubTabsBar::tab in
			// mainwindow_style.cpp), and unaffected by the label's own
			// rect-mismatch problem since there is no text to hide here.
			opt.rect = stretchedTabRect(i);
			painter.drawControl(QStyle::CE_TabBarTabShape, opt);

			QColor color = m_normalColor;
			QFont font = painter.font();
			if (opt.state & QStyle::State_Selected) {
				color = m_selectedColor;
				font.setBold(true);
			} else if (opt.state & QStyle::State_MouseOver) {
				color = m_hoverColor;
			}
			painter.setPen(color);
			painter.setFont(font);

			// Close glyph, drawn (and hit-tested - see mousePressEvent())
			// by hand for the same reason the label is: Qt's automatic
			// close-button placement (setTabButton()/tabsClosable(true))
			// positions relative to the tab bar's shape/orientation
			// semantics, not the screen - for a West-shaped bar,
			// QTabBar::LeftSide lands the button at the tab's physical
			// TOP edge, not its visual left, no matter what this bar's own
			// painting does to make the tab read as a normal horizontal
			// one. A real child widget positioned by Qt's layout can't be
			// told "screen-left" directly, so the glyph is drawn as plain
			// text instead, exactly where this bar's own coordinate space
			// says left actually is.
			painter.setPen(color);
			painter.drawText(closeGlyphRect(i), Qt::AlignCenter, QStringLiteral("×"));

			// Drawn upright in the bar's own (unrotated) coordinate space -
			// there is no rotation to undo here in the first place, unlike
			// the style's own CE_TabBarTabLabel handling for West tabs.
			// Left margin clears the close glyph; right margin is just
			// breathing room before the tab's own edge. Elided against
			// stretchedTabRect()'s width, not tabRect()'s, so widening the
			// bar (dragging the splitter - see SplitPreviewTabs) actually
			// reveals more of the label instead of leaving it truncated in
			// newly opened dead space to its right.
			const QRect textRect = stretchedTabRect(i).adjusted(30, 0, -8, 0);
			const QString elided = QFontMetrics(font).elidedText(tabText(i), Qt::ElideRight, textRect.width());
			painter.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, elided);
		}
	}

	void mousePressEvent(QMouseEvent *event) override {
		if (event->button() == Qt::LeftButton) {
			for (int i = 0; i < count(); ++i) {
				if (closeGlyphRect(i).contains(event->pos())) {
					emit closeRequested(i);
					return; // Swallowed - a close click is not a tab-select click.
				}
			}
		}
		QTabBar::mousePressEvent(event);
	}

private:
	// Qt's own tabRect(index) only ever spans the label's natural text
	// width (see tabSizeHint() above, deliberately left alone) - stretching
	// that would mean feeding width() back through tabSizeHint(), but
	// tabSizeHint() output feeds Qt's own layout/sizeHint bookkeeping for
	// this bar, and QSplitter consults that bookkeeping too (see
	// SplitPreviewTabs), so a self-referential width() there turned live
	// splitter drags into a feedback loop that collapsed the bar instead of
	// growing it. Painting, close-glyph hit-testing, and text elision all
	// read this instead: same rect, width swapped for the bar's actual
	// current width - a pure paint/hit-test-time adjustment that never
	// reaches anything Qt's layout system queries.
	QRect stretchedTabRect(int index) const {
		QRect r = tabRect(index);
		r.setWidth(qMax(60, width() - r.x()));
		return r;
	}

	QRect closeGlyphRect(int index) const {
		const QRect r = stretchedTabRect(index);
		return QRect(r.left() + 6, r.center().y() - 8, 18, 18);
	}

	QColor m_normalColor;
	QColor m_hoverColor;
	QColor m_selectedColor;
};

// A hand-rolled stand-in for QTabWidget, built from a HorizontalTabBar and a
// QStackedWidget placed in a QSplitter instead of QTabWidget's fixed internal
// layout. QTabWidget offers no way to make the boundary between its tab bar
// and its pages user-draggable - that boundary isn't a QSplitter at all, just
// a plain layout - so there was no way to give the tab strip's width the same
// drag-to-resize affordance as the splitter on the sidebar's edge (see
// createPreviewTab()). Splitting the two halves out into real QSplitter
// children gets that same handle "for free", matching the sidebar one
// exactly since both come from the same default QSplitter::handle styling.
// Only the subset of QTabWidget's API createPreviewTab() and friends
// actually use is reimplemented here.
class SplitPreviewTabs : public QWidget {
	Q_OBJECT
public:
	explicit SplitPreviewTabs(QWidget *parent = nullptr) : QWidget(parent) {
		auto *layout = new QHBoxLayout(this);
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		// Swaps between the empty-state prompt (no renders yet, or every
		// tab closed) and the real tab bar/page splitter below - rather
		// than leaving the whole pane blank but for the sidebar, which is
		// what a bare, always-present splitter would do while count() == 0.
		m_outerStack = new QStackedWidget(this);
		layout->addWidget(m_outerStack);

		m_emptyState = new QWidget(m_outerStack);
		QVBoxLayout *emptyLayout = new QVBoxLayout(m_emptyState);
		emptyLayout->addStretch(1);
		m_emptyIcon = new QLabel(m_emptyState);
		m_emptyIcon->setAlignment(Qt::AlignCenter);
		emptyLayout->addWidget(m_emptyIcon);
		m_emptyTitle = new QLabel(tr("No renders yet"), m_emptyState);
		m_emptyTitle->setAlignment(Qt::AlignCenter);
		QFont titleFont = m_emptyTitle->font();
		titleFont.setPointSizeF(titleFont.pointSizeF() * 1.3);
		titleFont.setBold(true);
		m_emptyTitle->setFont(titleFont);
		emptyLayout->addWidget(m_emptyTitle);
		m_emptySubtitle = new QLabel(
			tr("Start a render from Basic Settings - each finished image\n"
			   "or video opens in its own tab here, so past renders stay\n"
			   "around while you compare or tweak settings."),
			m_emptyState);
		m_emptySubtitle->setAlignment(Qt::AlignCenter);
		m_emptySubtitle->setWordWrap(true);
		emptyLayout->addWidget(m_emptySubtitle);
		emptyLayout->addStretch(1);
		m_outerStack->addWidget(m_emptyState);

		m_splitter = new QSplitter(Qt::Horizontal, m_outerStack);
		m_splitter->setChildrenCollapsible(false);
		m_outerStack->addWidget(m_splitter);

		// QSplitter always stretches a pane to the splitter's full
		// perpendicular extent (here, full height) - but QTabBar, given more
		// height than its tabs need, centers the tab group in the extra
		// space rather than pinning it to the top the way QTabWidget's own
		// (non-QSplitter) internal layout does. Wrapping the bar in its own
		// top-anchoring container - tab bar, then a stretch - keeps the
		// splitter pane full height (so the drag handle still runs the
		// whole column) while restoring the top-anchored tab list.
		QWidget *tabBarContainer = new QWidget(m_splitter);
		QVBoxLayout *tabBarLayout = new QVBoxLayout(tabBarContainer);
		tabBarLayout->setContentsMargins(0, 0, 0, 0);
		tabBarLayout->setSpacing(0);
		m_tabBar = new HorizontalTabBar(tabBarContainer);
		m_tabBar->setShape(QTabBar::RoundedWest);
		tabBarLayout->addWidget(m_tabBar);
		tabBarLayout->addStretch(1);
		m_splitter->addWidget(tabBarContainer);

		m_stack = new QStackedWidget(m_splitter);
		m_splitter->addWidget(m_stack);

		// The tab strip keeps its own width on a splitter drag; the page
		// area absorbs the rest - mirrors setStretchFactor() on the outer
		// splitter between this widget and the sidebar.
		m_splitter->setStretchFactor(0, 0);
		m_splitter->setStretchFactor(1, 1);
		// setStretchFactor() only governs how space is redistributed on a
		// live resize; it does nothing for the very first layout, which
		// QSplitter seeds from each pane's sizeHint() - and m_stack's hint
		// is whatever an empty QStackedWidget reports (tiny), since this
		// runs at construction time, long before the first render ever adds
		// a page. Without this, that tiny initial split sticks: the page
		// area never claims the rest of the width on its own, so the first
		// image/video added would render small and centered in a pane far
		// smaller than what's actually available. A lopsided requested
		// split (mostly to the stack) forces QSplitter to hand it virtually
		// all the space up front instead.
		m_splitter->setSizes({1, 1'000'000});

		connect(m_tabBar, &QTabBar::currentChanged, this, [this](int index) {
			m_stack->setCurrentIndex(index);
			emit currentChanged(index);
		});
	}

	HorizontalTabBar *tabBar() const { return m_tabBar; }

	int addTab(QWidget *page, const QString &label) {
		const int stackIndex = m_stack->addWidget(page);
		const int tabIndex = m_tabBar->addTab(label);
		Q_ASSERT(stackIndex == tabIndex);
		m_outerStack->setCurrentWidget(m_splitter);
		return tabIndex;
	}

	void setTabToolTip(int index, const QString &tip) { m_tabBar->setTabToolTip(index, tip); }
	void setCurrentIndex(int index) { m_tabBar->setCurrentIndex(index); }
	QWidget *currentWidget() const { return m_stack->currentWidget(); }
	QWidget *widget(int index) const { return m_stack->widget(index); }
	void setElideMode(Qt::TextElideMode mode) { m_tabBar->setElideMode(mode); }

	// Removes the tab/page pair at index without deleting the page - same
	// ownership contract as QTabWidget::removeTab(), so closePreviewSubTab()
	// (mainwindow_tabs.cpp) still does its own deleteLater() afterward.
	// The tab bar's own removeTab() can auto-select and emit currentChanged
	// before the stack has caught up (index i+1's old page still sitting at
	// index i) - blocked here and re-emitted once both sides agree.
	void removeTab(int index) {
		QWidget *page = m_stack->widget(index);
		{
			const QSignalBlocker blocker(m_tabBar);
			m_tabBar->removeTab(index);
		}
		if (page) m_stack->removeWidget(page);
		const int newIndex = m_tabBar->currentIndex();
		m_stack->setCurrentIndex(newIndex);
		if (m_tabBar->count() == 0) m_outerStack->setCurrentWidget(m_emptyState);
		emit currentChanged(newIndex);
	}

	// Kept in sync with the active theme by MainWindow::applyTheme(), same
	// as HorizontalTabBar::setColors() - the empty-state prompt is plain
	// QLabels, not QSS-styled chrome, so it needs its colours (and the
	// icon's tint, baked into the pixmap at paint time rather than
	// stylesheet-recolourable) hand-fed the same way.
	void setEmptyStateColors(const QColor &iconColor, const QColor &titleColor, const QColor &subtitleColor) {
		m_emptyIcon->setPixmap(icon_tint::tinted(":/icons/image.svg", iconColor).pixmap(56, 56));
		m_emptyTitle->setStyleSheet(QStringLiteral("color: %1;").arg(titleColor.name()));
		m_emptySubtitle->setStyleSheet(QStringLiteral("color: %1;").arg(subtitleColor.name()));
	}

signals:
	void currentChanged(int index);

private:
	QStackedWidget *m_outerStack;
	QWidget *m_emptyState;
	QLabel *m_emptyIcon;
	QLabel *m_emptyTitle;
	QLabel *m_emptySubtitle;
	QSplitter *m_splitter;
	HorizontalTabBar *m_tabBar;
	QStackedWidget *m_stack;
};

// ============================================================================
// RenderController
// ============================================================================
// Spawns ray_tracer.exe as a subprocess and turns its stdout into progress
// and log signals.
//
// This deliberately is NOT a QThread. QProcess is already fully asynchronous
// (it's a QObject that emits readyReadStandardOutput/finished/errorOccurred
// on the event loop), so it needs no worker thread of its own - this class
// lives on the GUI thread and is driven entirely by those signals. The
// earlier design subclassed QThread and ran a blocking
// `while (running) { waitForReadyRead(50); readAll(); }` loop, which cost a
// fixed up-to-50ms latency on every progress update and forced the
// stop-request flag to be an atomic polled across threads. Being
// single-threaded now, stopRender() can just kill the process directly.
// ============================================================================
class RenderController : public QObject {
	Q_OBJECT

public:
	explicit RenderController(QObject *parent = nullptr);

	// Set all render parameters before calling start()
	// Camera parameters (camX, camY, camZ) define the camera position (lookfrom)
	// The camera always looks at the Cornell box center (278, 278, 278) - lookat is fixed
	// sceneId selects which scene to render (category letter + number, e.g.
	// "A1"=Cornell Box, "A2"=Bouncing Spheres - see
	// src/TheRestOfYourLife/scene_registry.h's SceneDescriptor::id)
	// camExplicit: true if camX/Y/Z differ from sceneId's own recommended camera
	// (computed by the caller via SceneMetadataClient - see onRenderClicked) - only
	// then are cam_x/y/z actually passed on the command line, so ray_tracer.exe's
	// own "use the scene's recommended camera" fallback (main.cpp, driven by
	// LaunchArgs::cam_explicit) stays a real code path for untouched renders
	// instead of being permanently short-circuited by this GUI.
	void setParameters(bool useGPU, int width, int height, int samples, int maxDepth,
					   const QString &sceneId, double camX, double camY, double camZ, bool camExplicit,
					   const QString &outputPath = QString(), bool useWavefront = false);

	// Set video generation parameters
	void setVideoParameters(bool enabled, int frames, int fps, const QString &cameraPath, double speed = 1.0);

	// Builds the command line and launches ray_tracer.exe. Returns immediately;
	// everything after this point is driven by the process's own signals.
	void start();

	// Kill the render process. Safe to call when nothing is running.
	void stopRender();

	bool isRunning() const;

signals:
	void progressUpdate(int percentage);
	void renderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void logMessage(const QString &message);

private slots:
	void onReadyRead();
	void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void onProcessErrorOccurred(QProcess::ProcessError error);

private:
	// Splits a raw output chunk into complete lines (holding any trailing
	// partial line back in m_lineBuffer until the rest arrives), emitting
	// each as a log message and feeding it to parseProgressLine().
	void handleOutputChunk(const QString &chunk);
	void parseProgressLine(const QString &line);
	void finish(bool success, const QString &message, const QString &outputPath);

	// Render configuration
	bool m_useGPU;          // true = GPU renderer, false = CPU renderer
	bool m_useWavefront;    // true = wavefront (queue-based) GPU backend; ignored unless m_useGPU
	int m_width;            // Image width in pixels
	int m_height;           // Image height in pixels
	int m_samples;          // Samples per pixel (anti-aliasing quality)
	int m_maxDepth;         // Max ray bounce depth (lighting quality)
	QString m_sceneId;      // Scene selector ID (category letter + number, e.g. "A1")

	// Camera position (lookfrom) - always looking at center (278, 278, 278)
	double m_camX;          // Camera X coordinate
	double m_camY;          // Camera Y coordinate
	double m_camZ;          // Camera Z coordinate
	bool m_camExplicit;     // See setParameters()'s comment

	QString m_outputPath;   // Output file path for rendered image
	QProcess *m_renderProcess = nullptr;  // Subprocess handle (child of this object)

	// Whether the user asked to stop, tracked explicitly rather than inferred
	// from exit status/code - a real crash (e.g. access violation) also
	// produces CrashExit + a nonzero exit code on Windows, so that heuristic
	// can't tell the two apart and used to mislabel genuine crashes as
	// "stopped by user", hiding the error dialog.
	bool m_stopRequested = false;

	// Guards against emitting renderComplete twice when both errorOccurred
	// and finished fire for the same run.
	bool m_finished = false;

	QElapsedTimer m_elapsed;   // Wall-clock timer for the render
	QString m_lineBuffer;      // Incomplete trailing line awaiting the rest of its chunk
	int m_lastProgress = 0;    // Last emitted percentage (progress only moves forward)

	// Video generation parameters
	bool m_videoMode;       // true = video generation, false = single image
	int m_videoFrames;      // Number of frames to render
	int m_videoFPS;         // Target frames per second
	double m_videoSpeed;    // Camera movement speed multiplier (1.0 = default path speed)
	QString m_cameraPath;   // Camera animation path (orbit, linear, figure8, spiral)
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
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();

private slots:
	void onRenderClicked();
	void onStopClicked();
	void onQualityPresetChanged(int index);
	void onCameraPresetChanged(int index);  // Updates camera spinboxes when preset changes
	void onVideoPresetChanged(int index);   // Points scene/camera-path/frames/fps/speed at a named preset
	void onCameraDistanceChanged(double distance);  // Repositions camera X/Y/Z along its current direction from lookat
	void onSceneChanged(int index);         // Updates UI when scene selection changes
	void onModeChanged(int index);          // Switches between Image and Video modes
	void onProgressUpdate(int percentage);
	void onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void onLogMessage(const QString &message);
	void onElapsedTick();        // fires every second during render to update status label

	// Shared by the log tab's buttons and the File menu's actions.
	void copyLogToClipboard();
	void saveLogToFile();
	void clearLog();
	void showAboutDialog();

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void createVideoTab();
	void createPreviewTab();
	void createLogTab();

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
	// theme menu re-invokes it to switch live rather than asking for a restart.
	void applyTheme(const theme::Palette &palette);
	void switchTheme(const QString &themeId);
	void createThemeMenu();
	QString loadSavedThemeId() const;
	void saveThemeId(const QString &themeId) const;
	void styleComboBox(QComboBox *combo);
	void applyComboPopupPalette(QComboBox *combo);
	void styleSpinBox(QAbstractSpinBox *spinBox);
	void styleGroupBox(QGroupBox *box);
	// baseOutputPath is the render's own --output argument (see
	// onRenderClicked()) so the assembled video's expected path
	// ("<stem>_video.mp4") can be derived directly instead of guessing at a
	// shared directory - necessary now that every render's base path is
	// unique. Automatically assembles/locates the video after frames are
	// rendered and adds it as a new Preview sub-tab.
	void assembleVideoAutomatically(const QString &baseOutputPath);
	void refreshCameraDistanceDisplay(); // Recomputes m_cameraDistance's shown value from X/Y/Z and m_currentLookat*, without re-triggering onCameraDistanceChanged

	// Preview sub-tabs (see m_previewSubTabs's own comment). Both add*
	// functions build a self-contained tab page, register it, and make it
	// current; tooltip is the full scene/preset description, shown on
	// hover since the tab bar itself only has room for a short title.
	void addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
							 const QString &infoText, const QString &outputPath, const QString &previewPath);
	void addVideoPreviewTab(const QString &title, const QString &tooltip, const QString &videoPath,
							 const QString &infoText);
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
	QLabel *m_statusLabel;

	// Advanced Tab - Manual Controls
	QSpinBox *m_widthSpinBox;           // Custom width
	QSpinBox *m_heightSpinBox;          // Custom height
	QSpinBox *m_samplesSpinBox;         // Samples per pixel
	QSpinBox *m_maxDepthSpinBox;        // Max ray depth

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
	QComboBox *m_sceneCombo;            // Scene selector dropdown, showing one category at a time

	// Rebuilds m_sceneCategoryTabs' tab set for the given availability filter
	// (true = only scenes with sceneRequiresFiles()==true, false = only
	// scenes with sceneRequiresFiles()==false), skipping any letter category
	// left with zero matching scenes - same skip-if-empty rule
	// createBasicTab() already applies for categories with zero scenes at
	// all. Does not touch m_sceneCombo; callers follow up with
	// populateSceneCombo() for whichever category tab ends up selected.
	void rebuildCategoryTabs(bool requiresFiles);

	// Refills m_sceneCombo with just the scenes in `category` that also match
	// m_sceneAvailabilityTabs' current selection. Does NOT emit
	// currentIndexChanged per insertion - callers apply the resulting selection
	// themselves with a single onSceneChanged() call.
	void populateSceneCombo(const QString &category);

	// Drives all three scene-selection widgets (availability tab, category
	// tab, m_sceneCombo) to the scene matching `id`, as if the user had
	// clicked through to it by hand - switching availability/category tabs
	// first if `id` isn't already visible under the current ones. Used by
	// the video preset combo (see onVideoPresetChanged()) to point the
	// scene picker at a preset's scene without duplicating this tab-
	// cascading logic. No-op (logs a warning) if `id` isn't a real scene id.
	void selectSceneById(const QString &id);
	QLabel *m_sceneInfoLabel;           // Scene description and performance info

	// Video Tab
	QComboBox *m_videoPresetCombo;      // Named scene+path+frames/fps/speed bundle - see video_preset.h
	QComboBox *m_cameraPathCombo;       // Camera animation path selector
	QSpinBox *m_videoFramesSpinBox;     // Number of frames to render
	QSpinBox *m_videoFPSSpinBox;        // Target FPS for video
	QDoubleSpinBox *m_videoSpeedSpinBox; // Camera movement speed multiplier
	QLabel *m_videoInfoLabel;           // Video duration and path info

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
	QWidget *m_previewSidebar = nullptr; // Info/buttons pane; hidden while there are no sub-tabs - see updatePreviewSidebarForActiveTab()
	QLabel *m_previewInfoLabel;         // Filename / resolution / size / render time - reflects whichever sub-tab is active
	QLabel *m_previewSceneDescLabel;    // Selected scene's description - see onSceneChanged()
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

	// Render driver (nullptr when not rendering)
	RenderController *m_renderController;

	// State
	bool m_isRendering;                 // true when a render is in progress
	bool m_videoMode;                   // true = video generation mode, false = single image mode

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
	// Clears or reddens the taskbar button and, if the window isn't the
	// active one, raises a tray notification.
	void notifyRenderFinished(bool success, const QString &message, double totalTime);

	// Last percentage pushed to the taskbar button, so the COM call only
	// fires when the integer percent actually changes.
	int m_lastTaskbarPercent = -1;

	// Used only for completion notifications - the app has no tray UI.
	// Null if the platform has no system tray.
	QSystemTrayIcon *m_trayIcon = nullptr;

	// Shared event filter that blocks accidental wheel-scroll on controls
	WheelIgnoreFilter *m_wheelFilter;
};

#endif // MAINWINDOW_H
