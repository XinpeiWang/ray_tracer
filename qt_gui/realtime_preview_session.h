#ifndef REALTIME_PREVIEW_SESSION_H
#define REALTIME_PREVIEW_SESSION_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QThread>
#include <vector>

// ============================================================================
// RealtimePreviewSession
// ============================================================================
// GPU progressive-refinement live preview (see this project's own real-time-
// preview plan): renders very-low-spp frames in a tight in-process loop via
// realtime_renderer.dll (gpu/optix/optix_interface.h's rt_realtime_render_
// frame(), re-exported across the MinGW/MSVC ABI boundary the same way
// scene_metadata.dll already re-exports cpu_renderer's scene registry - see
// scene_metadata_client.cpp's own header comment for why a DLL boundary is
// needed at all here), accumulates the raw linear samples into a running
// mean, and emits a tonemapped QImage once per frame for the GUI to display.
//
// Deliberately scoped to progressive refinement only for this first pass:
// the camera is supplied by the caller (from the existing camera spinboxes),
// not driven by mouse/orbit input - see setCamera()'s own comment for what
// "changing" it does.
//
// RealtimePreviewWorker does the actual work and lives on its own QThread
// (moveToThread() pattern, not a QThread subclass - avoids the classic
// override-run()-and-touch-members-created-in-the-wrong-thread pitfall).
// RealtimePreviewSession is the GUI-thread-owned handle: it owns the QThread,
// forwards start/stop/setScene/setCamera as queued calls onto the worker,
// and re-emits the worker's frameReady/statusChanged signals (Qt's
// cross-thread signal/slot connections already marshal these automatically
// since the worker lives on a different QThread).
// ============================================================================

class RealtimePreviewWorker : public QObject {
	Q_OBJECT
public:
	explicit RealtimePreviewWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
	// Starts the render loop for sceneId at (camX,camY,camZ), width x height.
	// Safe to call again while already running - restarts with the new
	// scene/resolution and a fresh accumulation buffer, same effect as
	// stop() then start(). Runs until stop() is called. Bumps m_epoch (see
	// its own comment) so any renderLoop() continuation still queued from
	// a PRIOR start()/stop() cycle recognizes itself as stale and exits
	// instead of running alongside the new chain this call starts.
	void start(QString sceneId, int width, int height, double camX, double camY, double camZ);

	// Stops the loop after the in-flight frame (if any) finishes. Safe to
	// call even if not running.
	void stop();

	// Moves the camera and resets accumulation (a real "the view changed,
	// start converging again" reset - see the class-level comment on why
	// this project's chosen progressive-preview design resets rather than
	// tries to reproject/reuse samples across a camera change, matching
	// three-gpu-pathtracer/GLSL-PathTracer's own approach). No-op if not
	// currently running.
	void setCamera(double camX, double camY, double camZ);

signals:
	// Emitted once per accumulated frame - already tonemapped (ACES + sRGB,
	// matching this project's own CPU/GPU display convention) and ready to
	// hand straight to a QLabel/QPixmap on the GUI thread.
	void frameReady(QImage image, int sampleCount);

	// Human-readable one-liner for a status label (e.g. "128 samples" or an
	// error message) - separate from frameReady so a failure can be reported
	// even on a frame that produced no image.
	void statusChanged(QString text);

private:
	// `epoch` is the value m_epoch held when THIS continuation was posted -
	// see m_epoch's own comment for why renderLoop() needs to know that,
	// not just read the current m_running/m_epoch.
	void renderLoop(int epoch);
	void resetAccumulation();

	QString m_sceneId;
	int m_width = 0;
	int m_height = 0;
	double m_camX = 0.0, m_camY = 0.0, m_camZ = 0.0;
	std::vector<float> m_accum;   // linear RGB running mean, width*height*3
	// Per-frame scratch buffers, persisted across renderLoop() calls and
	// only resized in resetAccumulation() (same resolution-keyed reuse
	// shape as m_accum itself, and the same GPU-side d_fb_/d_weight_
	// persistence this project's own plan already applied to
	// WavefrontPathTracer::render() - avoids a heap alloc/free pair on
	// every single frame for buffers that are the same size every time).
	std::vector<float> m_tmp;     // raw per-call sample from the DLL, width*height*3
	QImage m_displayImage;        // tonemapped result, re-filled in place each frame
	int m_sampleCount = 0;
	// Every access to the fields below happens only inside a method
	// invoked via Qt::QueuedConnection onto this worker's own QThread
	// (start()/stop()/setCamera() from RealtimePreviewSession, renderLoop()'s
	// own self-continuation) - Qt's per-thread event queue already
	// serializes all of them, so plain bool/int (not std::atomic) is
	// correct here, not just adequate.
	bool m_running = false;
	bool m_cameraDirty = false;
	// Bumped by every start() call. A renderLoop(epoch) continuation
	// compares its own captured epoch against the CURRENT m_epoch before
	// doing anything - a stale continuation left over from a stop()+start()
	// cycle that raced ahead of it (see start()'s own comment) carries an
	// OLDER epoch and safely no-ops instead of running as a second,
	// independent render chain alongside the new one start() just began.
	int m_epoch = 0;
};

class RealtimePreviewSession : public QObject {
	Q_OBJECT
public:
	explicit RealtimePreviewSession(QObject *parent = nullptr);
	~RealtimePreviewSession() override;

	// True if realtime_renderer.dll was found and exports the expected
	// symbol - callers should grey out/hide the Live Preview UI entirely
	// when this is false rather than let start() silently fail every frame.
	static bool isAvailable();

	void start(const QString &sceneId, int width, int height, double camX, double camY, double camZ);
	void stop();
	void setCamera(double camX, double camY, double camZ);

signals:
	void frameReady(QImage image, int sampleCount);
	void statusChanged(QString text);

private:
	QThread m_thread;
	RealtimePreviewWorker *m_worker;
};

#endif // REALTIME_PREVIEW_SESSION_H
