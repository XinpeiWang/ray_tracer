#ifndef MAINWINDOW_JOBTYPES_H
#define MAINWINDOW_JOBTYPES_H
// mainwindow_jobtypes.h -- render-job driver classes and their small data
// structs (RenderController, DiagnosticsRunner, ThumbnailGenerator, and
// the plain structs they pass around). Split out of mainwindow.h, which
// #includes this file at the point this content used to live.
//
// Every Q_OBJECT class here needs its own moc output - see qt_gui/
// RayTracerGUI.pro's HEADERS list, which lists this file alongside
// mainwindow.h for exactly that reason (qmake only runs moc on headers
// explicitly listed there, not on anything textually #include'd) - and
// that in turn means THIS file must be self-sufficient (its own full Qt
// include list, not "borrowed" from mainwindow.h's), since moc generates
// and compiles moc_mainwindow_jobtypes.cpp as its own independent
// translation unit that #includes only this header, standalone. Kept the
// same include list mainwindow.h's own top used before the split, rather
// than trimming to each class's exact minimum - over-including a Qt
// header is harmless, and this avoids re-deriving per-class minimums by
// hand only to get one wrong.
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
// "Render Options" tab flags - one struct instead of 7 positional
// parameters. Each field maps 1:1 to a CLI flag (see RenderController::
// start()'s own arg-building); defaults here match the CLI's own
// defaults, so code that never touches this (ThumbnailGenerator, in
// particular, via a default-constructed RenderJob) renders exactly as it
// always has. sampler/lightSampler/tonemap empty = use the CLI's own
// default (sobol/bvh/aces) rather than passing the flag at all.
struct AdvancedRenderFlags {
	bool denoise = false;
	bool stats = false;
	bool optixValidate = false;
	double exposure = 1.0;
	QString sampler;
	QString lightSampler;
	bool spectral = false;
	QString tonemap;
};

// Which alternate integrator (--sppm/--bdpt/--mlt/--randomwalk/--ao/
// --simplepath/--simplevolpath/--lightpath) to use instead of the default
// path tracer - see the Integrator combo and its per-mode sub-flags, both
// in the "Integrator" group on the Render Options tab. No CLI analog exists
// for this as a single enum - launcher_args.h uses 8 independent bool
// use_X fields with hand-written mutual exclusion in main.cpp - but a
// combo box is naturally single-selection, so the GUI gets a real enum
// instead of mirroring that shape.
enum class IntegratorMode {
	Default = 0, Sppm, Bdpt, Mlt, RandomWalk, Ao, SimplePath, SimpleVolPath, LightPath
};

// "Integrator Options" group's per-integrator sub-flags - one struct
// instead of growing AdvancedRenderFlags or setParameters() further, same
// reasoning as AdvancedRenderFlags's own comment. Defaults match
// launcher_args.h's kDefaultXxx constants exactly, so a render with
// Integrator left on Default (mode's own default) reproduces today's
// command line unchanged.
struct IntegratorOptions {
	IntegratorMode mode = IntegratorMode::Default;
	int sppmIterations = 100;          // kDefaultSppmIterations
	int sppmPhotons = 5000;            // kDefaultSppmPhotons
	int bdptMaxDepth = 5;              // kDefaultBdptMaxDepth
	int mltBootstrap = 100000;         // kDefaultMltBootstrap
	long long mltMutations = 4000000;  // kDefaultMltMutations
	int mltMaxDepth = 5;               // kDefaultMltMaxDepth
	double aoMaxDist = 1e10;           // kDefaultAoMaxDist
	bool aoUniform = false;            // false = cosine (CLI ao_cosine default true)
	double aoIllumScale = 1.0;         // kDefaultAoIllumScale
	double aoIllumR = 1.0, aoIllumG = 1.0, aoIllumB = 1.0;
	bool simplepathNoLights = false;   // CLI simplepath_sample_lights default true
	bool simplepathNoBsdf = false;     // CLI simplepath_sample_bsdf default true
};

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

	// Set the "Render Options" tab's flags - a separate setter rather than
	// growing setParameters()'s already-large signature further, and one
	// struct rather than 7 positional parameters (see AdvancedRenderFlags's
	// own comment above for why: three adjacent bools then two adjacent
	// strings is exactly the shape a transposed-argument bug hides in).
	void setAdvancedFlags(const AdvancedRenderFlags &flags);

	// Which alternate integrator to use and its sub-flags - see
	// IntegratorOptions's own comment. Separate setter for the same reason
	// setAdvancedFlags() is separate from setParameters().
	void setIntegratorOptions(const IntegratorOptions &options);

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

	// "Render Options" tab flags - see AdvancedRenderFlags's own comment.
	// Default-constructed matches every existing caller's prior behavior
	// (nothing passed = CLI default), so ThumbnailGenerator (which never
	// calls setAdvancedFlags()) is unaffected.
	AdvancedRenderFlags m_advancedFlags;

	// Integrator combo + "Integrator Options" group's flags - see
	// IntegratorOptions's own comment. Default-constructed (mode=Default)
	// matches every existing caller's prior behavior, same reasoning as
	// m_advancedFlags above.
	IntegratorOptions m_integratorOptions;

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
// DiagnosticsRunner
// ============================================================================
// Spawns `ray_tracer.exe --diagnose` and captures its full stdout as one
// report string. A stripped-down sibling of RenderController above (same
// QProcess-is-already-async reasoning, same not-a-QThread shape) - no
// progress parsing, no video/scene parameters, just "run it, hand back
// whatever it printed."
// ============================================================================
class DiagnosticsRunner : public QObject {
	Q_OBJECT

public:
	explicit DiagnosticsRunner(QObject *parent = nullptr);

	// Builds the command line and launches ray_tracer.exe --diagnose.
	// Returns immediately; everything after is driven by the process's own
	// signals.
	void start();

	// Kill the diagnostics process. Safe to call when nothing is running.
	// Mirrors RenderController::stopRender()/isRunning() - added so
	// ~MainWindow() has a graceful way to end an in-flight diagnostics run
	// at app exit instead of relying solely on QProcess's own destructor
	// (which does forcibly kill a still-running child, but with a blocking
	// wait and a logged warning rather than this app's own quieter path).
	void stop();

	bool isRunning() const;

signals:
	void reportReady(const QString &report);
	void reportFailed(const QString &message);

private slots:
	void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void onProcessErrorOccurred(QProcess::ProcessError error);

private:
	QProcess *m_process = nullptr;
	bool m_finished = false;   // guards against double-emitting on error+finished
};

// ============================================================================
// ThumbnailGenerator
// ============================================================================
// Renders the scene gallery grid's preview PNGs, one scene at a time, via its
// own RenderController - deliberately NOT m_renderController/m_renderQueue.
// Those are wired end-to-end into the visible progress bar, status label,
// queue panel and stop button; routing thumbnail jobs through them would
// hijack that chrome for invisible background work. This class owns a
// private RenderController instead and only ever READS the real render
// state (see MainWindow::onGenerateThumbnailsClicked()) to decide whether
// it's safe to start - it never competes with a user-requested render.
//
// Generation is manual/one-shot (triggered by "Generate Thumbnails"), not a
// background daemon, so there is no pause/resume logic here: start() walks
// its whole queue to completion or until stop() is called.
// ============================================================================
class ThumbnailGenerator : public QObject {
	Q_OBJECT

public:
	explicit ThumbnailGenerator(QObject *parent = nullptr);

	// Renders every id in `sceneIds` that doesn't already have a cached PNG
	// at outputPathForId(id), one at a time, low-res/low-spp/CPU-only (see
	// .cpp for the exact RenderController::setParameters() call). Ignored if
	// already running.
	void start(const QStringList &sceneIds, std::function<QString(const QString &)> outputPathForId);

	// Kills the in-flight render (if any) and drops the rest of the queue.
	void stop();

	bool isRunning() const;

signals:
	// One per finished scene (success or failure) - MainWindow uses this to
	// update that scene's grid item icon in place and advance status text.
	void thumbnailReady(const QString &sceneId, bool success, const QString &outputPath);
	// Fires once after the last queued scene finishes (or stop() is called).
	void allDone();

private:
	void startNext();

	RenderController *m_controller;
	QQueue<QString> m_pending;
	std::function<QString(const QString &)> m_outputPathForId;
	QString m_currentId;
	bool m_stopRequested = false;
};

// A snapshot of every render-affecting UI field at the moment "Start Render"
// was clicked, so a queued job is unaffected by any further changes the user
// makes to the form while an earlier job is still running. Mirrors
// RenderController::setParameters()/setVideoParameters()'s own parameter
// list exactly - see MainWindow::captureRenderJob()/startRenderJob()
// (mainwindow_slots.cpp).
struct RenderJob {
	bool useGPU = false;
	bool useWavefront = false;
	int width = 0;
	int height = 0;
	int samples = 0;
	int maxDepth = 0;
	QString sceneId;
	QString outputPath;
	double camX = 0.0;
	double camY = 0.0;
	double camZ = 0.0;
	bool camExplicit = true;
	bool videoMode = false;
	int videoFrames = 0;
	int videoFPS = 0;
	double videoSpeed = 1.0;
	QString cameraPath;
	QString displayTitle;      // Scene name, for the queue list row - see describeRenderJob()
	QString sceneDescription;  // Scene's own description, for the finished Preview tab's tooltip
	QString videoPresetName;   // Named video preset's display name, if one was selected; empty otherwise

	// "Render Options" tab fields - see AdvancedRenderFlags's own comment.
	AdvancedRenderFlags advancedFlags;

	// Integrator combo + "Integrator Options" group's fields - see
	// IntegratorOptions's own comment.
	IntegratorOptions integratorOptions;
};

// A finished render's metadata, persisted across app sessions (see
// recent_renders.cpp) so the Preview tab's empty-state list can reopen
// it later - deliberately smaller than RenderJob (which is never itself
// persisted): only what a list row needs to display
// (describeRecentRenderEntry()) and what reopening needs to pass back
// into addImagePreviewTab()/addVideoPreviewTab().
struct RecentRenderEntry {
	QString outputPath;   // image: raw .ppm/original render path; video: same as previewPath
	QString previewPath;  // image: converted .png; video: the .mp4 itself
	bool isVideo = false;
	QString sceneId;
	QString displayTitle;      // for video, the preset name or "<scene> (Video)" title actually shown
	QString sceneDescription;
	int width = 0, height = 0, samples = 0;
	bool useGPU = false, useWavefront = false;
	// Full structs (not just the mode enum) so a reopened entry can show
	// the same technique-box settings summary a fresh completion does -
	// see MainWindow::renderTechniqueHtml(). Persisted via new QSettings
	// keys in recent_renders.cpp's writeEntry()/readEntry(), guarded by
	// settings.contains() on read so an entry saved before these fields
	// existed keeps its correct struct defaults instead of reading back
	// fabricated zeros.
	IntegratorOptions integratorOptions;
	AdvancedRenderFlags advancedFlags;
	qint64 timestampEpochSecs = 0;
	// False for a best-effort entry backfilled by scanning the default
	// output folder for pre-existing render_*.png/*_video.mp4 files this
	// app produced before Recent Renders existed (or whose own persisted
	// entry has since aged out) - see loadRecentRenders(). Only
	// outputPath/previewPath/isVideo/sceneId (parsed from the filename,
	// possibly empty)/displayTitle/timestampEpochSecs are meaningful then;
	// describeRecentRenderEntry() omits the rest rather than showing
	// fabricated zeros.
	bool metadataKnown = true;
};

// Bundled rather than more bare QString parameters on
// addImagePreviewTab()/addVideoPreviewTab(). techniqueHtml is
// pre-formatted (see MainWindow::renderTechniqueHtml()) and left empty
// when there's nothing meaningful to report (a scanned/best-effort
// Recent Renders entry - see RecentRenderEntry::metadataKnown).
struct PreviewTechniqueInfo {
	QString sceneId;
	QString techniqueHtml;
};
#endif // MAINWINDOW_JOBTYPES_H
