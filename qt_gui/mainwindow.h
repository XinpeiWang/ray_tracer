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
#include <QTimer>
#include <QDateTime>
#include <QElapsedTimer>
#include <QPixmap>
#include <QResizeEvent>
#include <QSystemTrayIcon>

#include "camera_math.h"
#include "render_output_parser.h"
#include "theme.h"

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
	// sceneId selects which scene to render (0=Cornell Box, 1=Bouncing Spheres, etc.)
	// camExplicit: true if camX/Y/Z differ from sceneId's own recommended camera
	// (computed by the caller via SceneMetadataClient - see onRenderClicked) - only
	// then are cam_x/y/z actually passed on the command line, so ray_tracer.exe's
	// own "use the scene's recommended camera" fallback (main.cpp, driven by
	// LaunchArgs::cam_explicit) stays a real code path for untouched renders
	// instead of being permanently short-circuited by this GUI.
	void setParameters(bool useGPU, int width, int height, int samples, int maxDepth,
					   int sceneId, double camX, double camY, double camZ, bool camExplicit,
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
	int m_sceneId;          // Scene selector ID (0=Cornell Box, 1=Bouncing Spheres, etc.)

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
	void assembleVideoAutomatically();  // Automatically assembles video after frames are rendered
	void refreshCameraDistanceDisplay(); // Recomputes m_cameraDistance's shown value from X/Y/Z and m_currentLookat*, without re-triggering onCameraDistanceChanged

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
	QTabBar *m_sceneCategoryTabs = nullptr;  // Category filter above the scene dropdown
	QComboBox *m_sceneCombo;            // Scene selector dropdown, showing one category at a time

	// Refills m_sceneCombo with just the scenes in `category`. Does NOT emit
	// currentIndexChanged per insertion - callers apply the resulting selection
	// themselves with a single onSceneChanged() call.
	void populateSceneCombo(const QString &category);
	QLabel *m_sceneInfoLabel;           // Scene description and performance info

	// Video Tab
	QComboBox *m_cameraPathCombo;       // Camera animation path selector
	QSpinBox *m_videoFramesSpinBox;     // Number of frames to render
	QSpinBox *m_videoFPSSpinBox;        // Target FPS for video
	QDoubleSpinBox *m_videoSpeedSpinBox; // Camera movement speed multiplier
	QLabel *m_videoInfoLabel;           // Video duration and path info

	// Preview tab - shows the rendered image inline instead of shelling out
	// to the OS's default image viewer (see onRenderComplete()).
	ScaledImageLabel *m_previewLabel;   // Displays the rendered PNG, scaled to fit
	QLabel *m_previewInfoLabel;         // Filename / resolution / size / render time
	int m_previewTabIndex = -1;         // Index of the Preview tab within m_tabWidget
	QString m_lastOutputPath;           // Most recent render's raw output path (.ppm)
	QString m_lastPreviewImagePath;     // Most recent render's displayed image path (.png)

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
