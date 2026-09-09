#ifndef REALTIME_PREVIEW_SESSION_H
#define REALTIME_PREVIEW_SESSION_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QThread>
#include <cstdint>
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
// This class just accumulates whatever camera/look-at it's given into a
// converging image - it has no idea whether the caller got there from
// camera spinboxes, mouse-drag orbit, or WASD free-fly (all three drive it,
// via MainWindow's own orbit/translate handlers calling setCamera()) - see
// setCamera()'s own comment for what "changing" it does. A camera move
// reprojects rather than discards the existing accumulation (see
// reprojectAccumulation()'s own comment) - an EARLIER version of this class
// reset to zero noise on every move instead (matching three-gpu-pathtracer/
// GLSL-PathTracer's own simpler approach), until temporal reprojection
// replaced that with real reuse of still-valid samples.
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
	// Starts the render loop for sceneId at (camX,camY,camZ) looking at
	// (lookX,lookY,lookZ), width x height, with the given denoise settings
	// (see setDenoise()'s own comment). Safe to call again while already
	// running - restarts with the new scene/resolution and a fresh
	// accumulation buffer, same effect as stop() then start(). Runs until
	// stop() is called. Bumps m_epoch (see its own comment) so any
	// renderLoop() continuation still queued from a PRIOR start()/stop()
	// cycle recognizes itself as stale and exits instead of running
	// alongside the new chain this call starts.
	void start(QString sceneId, int width, int height, double camX, double camY, double camZ,
			   double lookX, double lookY, double lookZ,
			   bool denoise, double denoiseBlend, bool denoiseShowLatest);

	// Stops the loop after the in-flight frame (if any) finishes. Safe to
	// call even if not running.
	void stop();

	// Moves the camera and/or where it's looking. Marks accumulation dirty
	// (m_cameraDirty) rather than resetting it immediately - renderLoop()
	// renders the NEXT frame with this new camera first, then reprojects
	// the still-valid parts of the OLD accumulation into it (see
	// reprojectAccumulation()'s own comment) instead of discarding
	// everything. No-op if not currently running.
	void setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ);

	// Toggles the OptiX AI denoiser (same one --denoise/--denoise-blend use
	// for batch/video rendering, see optix_interface.h's own comment) for
	// every subsequent frame. A plain denoise-enabled/blend change does NOT
	// reset accumulation - that's a post-process decision on the same
	// converging signal, not a "this is a different image now" event, so a
	// momentary blend of old/new-style samples in m_accum is an acceptable,
	// self-correcting cosmetic blip.
	// denoiseShowLatest, when EFFECTIVE (denoise && denoiseShowLatest both
	// true - see renderLoop()'s own comment), replaces the running-mean
	// accumulation with "always display the latest frame" instead - unlike
	// a plain denoise/blend change, THIS changes what m_accum structurally
	// IS (a single frame vs. a genuine multi-sample average), so a change
	// to the EFFECTIVE flag (denoise && denoiseShowLatest) - from either
	// side flipping - does reset accumulation, the same "different image
	// now" treatment setCamera() gives an actual camera move. No-op if not
	// running.
	void setDenoise(bool denoise, double denoiseBlend, bool denoiseShowLatest);

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
	void reprojectAccumulation();

	QString m_sceneId;
	int m_width = 0;
	int m_height = 0;
	double m_camX = 0.0, m_camY = 0.0, m_camZ = 0.0;
	// Always passed to the DLL as an explicit look-at override (has_custom_lookat=true)
	// - Live Preview's whole point is letting the caller drive where the
	// camera looks, not just where it stands. See optix_interface.h's own
	// comment on rt_realtime_render_frame()'s lookat parameters.
	double m_lookX = 0.0, m_lookY = 0.0, m_lookZ = 0.0;
	// See setDenoise()'s own comment. m_denoiseShowLatest only changes how
	// renderLoop() folds m_tmp into m_accum below - it never crosses the
	// DLL boundary (the GPU side only needs to know whether/how much to
	// denoise, not what the Qt side does with the result afterward).
	// renderLoop() and setDenoise() both gate on `m_denoise &&
	// m_denoiseShowLatest` (never m_denoiseShowLatest alone), so leaving
	// m_denoiseShowLatest true while m_denoise is false is inert rather
	// than stuck showing raw, never-converging noise.
	bool m_denoise = false;
	double m_denoiseBlend = 0.0;
	bool m_denoiseShowLatest = false;
	std::vector<float> m_accum;   // linear RGB running mean, width*height*3
	// Per-frame scratch buffers, persisted across renderLoop() calls and
	// only resized in resetAccumulation() (same resolution-keyed reuse
	// shape as m_accum itself, and the same GPU-side d_fb_/d_weight_
	// persistence this project's own plan already applied to
	// WavefrontPathTracer::render() - avoids a heap alloc/free pair on
	// every single frame for buffers that are the same size every time).
	std::vector<float> m_tmp;     // raw per-call sample from the DLL, width*height*3
	// Temporal reprojection state - see reprojectAccumulation()'s own
	// comment. m_worldPos/m_cameraBasis are THIS frame's own (just rendered
	// with the CURRENT camera); m_worldPosPrev/m_prevCameraBasis are
	// whichever frame's data currently backs m_accum, updated to match at
	// the end of every successful frame (see renderLoop()) - so they always
	// hold exactly what a reprojection FROM m_accum needs, regardless of
	// how many frames (with or without a camera move) have happened since.
	std::vector<float> m_worldPos;         // xyz + validity, width*height*4
	std::vector<float> m_worldPosPrev;     // same layout, previous frame's
	std::vector<float> m_cameraBasis;      // origin/lowerLeft/horiz/vert, 12 floats
	std::vector<float> m_prevCameraBasis;  // same layout, previous frame's
	// Per-pixel effective sample count - NOT uniform once reprojection is in
	// play (a freshly-disoccluded pixel starts over at 0 while a
	// successfully-reprojected neighbor carries its whole history forward),
	// unlike the single scalar this replaced. m_sampleCount (below) becomes
	// the MINIMUM across all pixels once rendering starts - a conservative
	// "worst-converged pixel" indicator for the status label (see
	// MainWindow::onLivePreviewFrameReady()) rather than a literal count
	// that stopped being uniform.
	std::vector<uint16_t> m_sampleCounts;
	// Scratch buffers reprojectAccumulation() writes its remapped result
	// into before swapping with m_accum/m_sampleCounts - persisted and
	// only resized in resetAccumulation() (same reuse shape as m_accum/
	// m_tmp above) rather than allocated fresh every call, since a camera
	// drag/held WASD key runs reprojectAccumulation() on every rendered
	// frame for the whole gesture, not just once per discrete move.
	std::vector<float> m_accumScratch;
	std::vector<uint16_t> m_sampleCountsScratch;
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

	void start(const QString &sceneId, int width, int height, double camX, double camY, double camZ,
			   double lookX, double lookY, double lookZ,
			   bool denoise, double denoiseBlend, bool denoiseShowLatest);
	void stop();
	void setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ);
	void setDenoise(bool denoise, double denoiseBlend, bool denoiseShowLatest);

signals:
	void frameReady(QImage image, int sampleCount);
	void statusChanged(QString text);

private:
	QThread m_thread;
	RealtimePreviewWorker *m_worker;
};

#endif // REALTIME_PREVIEW_SESSION_H
