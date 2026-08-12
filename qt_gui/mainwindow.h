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
					   const QString &outputPath = QString());

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

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void createVideoTab();
	void createPreviewTab();
	void createLogTab();
	void applyDarkTheme();
	void styleComboBox(QComboBox *combo);
	void styleSpinBox(QAbstractSpinBox *spinBox);
	void styleGroupBox(QGroupBox *box);
	void assembleVideoAutomatically();  // Automatically assembles video after frames are rendered
	void refreshCameraDistanceDisplay(); // Recomputes m_cameraDistance's shown value from X/Y/Z and m_currentLookat*, without re-triggering onCameraDistanceChanged

	// UI Components
	QTabWidget *m_tabWidget;

	// Mode selector (Image vs Video)
	QComboBox *m_modeCombo;             // Render mode: Image or Video

	// Basic Tab
	QComboBox *m_renderModeCombo;       // GPU vs CPU selection
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
	QComboBox *m_sceneCombo;            // Scene selector dropdown (Cornell Box, Bouncing Spheres, etc.)
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

	// Render driver (nullptr when not rendering)
	RenderController *m_renderController;

	// State
	bool m_isRendering;                 // true when a render is in progress
	bool m_videoMode;                   // true = video generation mode, false = single image mode

	// Elapsed render timer
	QTimer *m_elapsedTimer;             // fires every second during render
	QDateTime m_renderStartTime;        // wall-clock time when render began

	// Shared event filter that blocks accidental wheel-scroll on controls
	WheelIgnoreFilter *m_wheelFilter;
};

#endif // MAINWINDOW_H
