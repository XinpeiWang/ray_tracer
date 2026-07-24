#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QThread>
#include <QLineEdit>
#include <QProcess>
#include <QVector3D>
#include <QTextEdit>
#include <QEvent>

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
        if (event->type() == QEvent::Wheel) {
            // Only allow scroll if the widget has focus (user clicked it first)
            QWidget *w = qobject_cast<QWidget*>(obj);
            if (w && !w->hasFocus())
                return true;  // consume / block the wheel event
        }
        return QObject::eventFilter(obj, event);
    }
};

// ============================================================================
// RenderThread
// ============================================================================
// Background thread that spawns ray_tracer.exe as a subprocess
// Passes render parameters including camera position via command line
// Monitors stdout for progress updates and completion status
// ============================================================================
class RenderThread : public QThread {
	Q_OBJECT

public:
	RenderThread(QObject *parent = nullptr);

	// Set all render parameters before starting the thread
	// Camera parameters (camX, camY, camZ) define the camera position (lookfrom)
	// The camera always looks at the Cornell box center (278, 278, 278) - lookat is fixed
	// sceneId selects which scene to render (0=Cornell Box, 1=Bouncing Spheres, etc.)
	void setParameters(bool useGPU, int width, int height, int samples, int maxDepth, 
					   int sceneId, double camX, double camY, double camZ, const QString &outputPath = QString());

	// Set video generation parameters
	void setVideoParameters(bool enabled, int frames, int fps, const QString &cameraPath);

	// Request the render process to stop (sends terminate signal)
	void stopRender();

protected:
	// Thread entry point - builds command line and launches ray_tracer.exe
	void run() override;

signals:
	void progressUpdate(int percentage);
	void renderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void logMessage(const QString &message);

private:
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

	QString m_outputPath;   // Output file path for rendered image
	QProcess *m_renderProcess; // Subprocess handle for ray_tracer.exe

	// Video generation parameters
	bool m_videoMode;       // true = video generation, false = single image
	int m_videoFrames;      // Number of frames to render
	int m_videoFPS;         // Target frames per second
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
	void onSceneChanged(int index);         // Updates UI when scene selection changes
	void onModeChanged(int index);          // Switches between Image and Video modes
	void onProgressUpdate(int percentage);
	void onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void onLogMessage(const QString &message);

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void createVideoTab();
	void createLogTab();
	void applyDarkTheme();
	void styleComboBox(QComboBox *combo);
	void styleSpinBox(QSpinBox *spinBox);
	void assembleVideoAutomatically();  // Automatically assembles video after frames are rendered

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
	// Camera position (lookfrom) can be set via presets or custom X/Y/Z values
	// All cameras look at the Cornell box center (278, 278, 278) - lookat is fixed in renderer
	QComboBox *m_cameraPresetCombo;     // Preset camera positions (includes "Custom" option)
	QDoubleSpinBox *m_cameraPosX;       // Camera X position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosY;       // Camera Y position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosZ;       // Camera Z position (enabled only for "Custom" preset)

	// Scene selection
	QComboBox *m_sceneCombo;            // Scene selector dropdown (Cornell Box, Bouncing Spheres, etc.)
	QLabel *m_sceneInfoLabel;           // Scene description and performance info

	// Video Tab
	QComboBox *m_cameraPathCombo;       // Camera animation path selector
	QSpinBox *m_videoFramesSpinBox;     // Number of frames to render
	QSpinBox *m_videoFPSSpinBox;        // Target FPS for video
	QLabel *m_videoInfoLabel;           // Video duration and path info

	// Log output
	QTextEdit *m_logTextEdit;           // Log output display

	// Render thread
	RenderThread *m_renderThread;       // Background render thread (nullptr when not rendering)

	// State
	bool m_isRendering;                 // true when a render is in progress
	bool m_videoMode;                   // true = video generation mode, false = single image mode

	// Shared event filter that blocks accidental wheel-scroll on controls
	WheelIgnoreFilter *m_wheelFilter;
};

#endif // MAINWINDOW_H
