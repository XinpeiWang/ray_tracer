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
			   bool denoise, double denoiseBlend, bool denoiseShowLatest, bool svgf,
			   bool restirGi, bool restirDi, int spp, int maxDepth, double fireflyClamp);

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

	// Toggles SVGF (gpu/optix/wavefront_svgf_math.h) - an alternative to
	// setDenoise()'s own OptiX AI denoiser, not layered on top of it (the
	// GUI is expected to present these as mutually-exclusive modes). Unlike
	// a plain denoise/blend change, enabling or disabling SVGF DOES reset
	// accumulation: SVGF's own output is unconditionally treated as
	// "already final" (renderLoop()'s own effectiveShowLatest, generalized
	// to `(m_denoise && m_denoiseShowLatest) || m_svgf`) - toggling it
	// changes what m_accum structurally IS, the same "different image now"
	// reasoning setDenoise()'s own denoiseShowLatest-effective-change already
	// gets. No-op if not running.
	void setSvgf(bool svgf);

	// Multiplies the accumulated linear radiance right before the ACES+sRGB
	// tonemap - same step (and formula) the batch/CLI path applies via its
	// own --exposure option (optix_interface.cpp). Live Preview needs its
	// OWN, independently-set value rather than reusing whatever the Settings
	// tab's exposure spinbox holds for batch rendering, since the two paths
	// converge to different images (Live Preview always renders via the
	// wavefront/ReSTIR backend). Defaults to 1.0, matching batch's own
	// implicit default - see m_exposure's own comment for why an earlier,
	// lower default existed and why it no longer needs to. A plain exposure
	// change does NOT reset accumulation, same "post-process on the same
	// converging signal" reasoning as setDenoise()'s own comment.
	void setExposure(double exposure);

	// Toggles ReSTIR GI (gpu/optix/wavefront_restir_gi_math.h) resampled
	// one-bounce indirect lighting - independent from setSvgf() above. No
	// accumulation reset needed: unlike denoise/SVGF, this doesn't change
	// what m_accum structurally holds, only what each new sample contains.
	// No-op if not running.
	void setRestirGi(bool restirGi);

	// Toggles ReSTIR DI (gpu/optix/wavefront_restir_helpers.h) resampled
	// direct-light sampling - independent from setRestirGi() above (that one
	// resamples one-bounce INDIRECT lighting; this one resamples the
	// direct-light draw classic NEE would otherwise do with a single
	// alias-table sample). Same "no accumulation reset needed" reasoning as
	// setRestirGi(). No-op if not running.
	void setRestirDi(bool restirDi);

	// Samples-per-frame / max ray depth for each low-spp render() call - see
	// renderLoop()'s own comment on why a small per-call cost is used at all.
	// No accumulation reset needed, same reasoning as setRestirGi() above.
	// No-op if not running.
	void setSppAndMaxDepth(int spp, int maxDepth);

	// Firefly clamp (GpuCameraParams::maxComponentValue) - see
	// optix_interface.h's rt_realtime_render_frame() comment. No accumulation
	// reset needed. No-op if not running.
	void setFireflyClamp(double fireflyClamp);

	// SVGF advanced tuning (gpu/optix/svgf_tuning_params.h's SvgfTuningParams,
	// one scalar param per field, in the SAME order) - NOT gated on m_running,
	// same reasoning as setExposure() above: the caller is expected to push
	// this (and setExposure()) BEFORE start() so the very first rendered
	// frame already reflects it, not just frame 2 onward (start()'s own
	// queued call synchronously renders frame 1 as part of the same
	// worker-thread event). No accumulation reset needed - a filter-tuning
	// change doesn't alter what m_accum structurally holds.
	void setSvgfTuning(double temporalAlpha, double maxHistoryLength, double varianceBootstrapFrames,
						int varianceBootstrapRadius, double sigmaNormal, double sigmaDepth,
						double sigmaLuminance, int atrousRadius, double minAlbedo, int atrousPasses);

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
	// `(m_denoise && m_denoiseShowLatest) || m_svgf` - the single "treat
	// m_tmp as already-final, don't blend into m_accum" condition, computed
	// in one place and reused by setDenoise()/setSvgf()/renderLoop() instead
	// of each recomputing it (a prior version had setDenoise() recompute it
	// without the `|| m_svgf` term, which only happened to be harmless
	// because the GUI never enables both denoise and SVGF at once - see
	// setSvgf()'s own comment on why that invariant isn't backend-enforced).
	bool effectiveShowLatest() const { return (m_denoise && m_denoiseShowLatest) || m_svgf; }

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
	// See setSvgf()'s own comment. Crosses the DLL boundary (unlike
	// m_denoiseShowLatest above) - the GPU side needs to know whether to run
	// SVGF at all, not just how Qt should treat its own accumulation buffer.
	bool m_svgf = false;
	// See setRestirGi()'s own comment. Crosses the DLL boundary like m_svgf
	// (the GPU side needs to know whether to run GI at all). Defaults true,
	// matching rt_realtime_render_frame()'s own previously-hardcoded-on
	// behavior.
	bool m_restirGi = true;
	// See setRestirDi()'s own comment. Crosses the DLL boundary like m_restirGi
	// above. Defaults true, matching rt_realtime_render_frame()'s own
	// previously-hardcoded-on behavior (ReSTIR DI shipped as always-on before
	// this parameter existed).
	bool m_restirDi = true;
	// See setSppAndMaxDepth()'s own comment. Both cross the DLL boundary
	// (they're renderFrame()'s own 4th/5th positional args). Defaults match
	// renderLoop()'s own previous hardcoded locals exactly.
	int m_spp = 1;
	int m_maxDepth = 8;
	// See setFireflyClamp()'s own comment. Crosses the DLL boundary. Default
	// matches rt_realtime_render_frame()'s own previous hardcoded literal.
	double m_fireflyClamp = 50.0;
	// See setSvgfTuning()'s own comment. Bundled into a local SvgfTuningParams
	// (mirroring gpu/optix/svgf_tuning_params.h - see this file's own
	// anonymous-namespace mirror struct) only at the renderFrame() call site
	// in renderLoop(), the point where it actually crosses the DLL boundary.
	// Defaults match SvgfTuningParams' own literature defaults exactly.
	double m_svgfTemporalAlpha = 0.2;
	double m_svgfMaxHistoryLength = 32.0;
	double m_svgfVarianceBootstrapFrames = 4.0;
	int m_svgfVarianceBootstrapRadius = 3;
	double m_svgfSigmaNormal = 128.0;
	double m_svgfSigmaDepth = 1.0;
	double m_svgfSigmaLuminance = 4.0;
	int m_svgfAtrousRadius = 2;
	double m_svgfMinAlbedo = 0.02;
	int m_svgfAtrousPasses = 4;
	// Neutral default (matches batch rendering's own implicit 1.0) - an
	// earlier version of this default was 0.5, added to mask a genuine ReSTIR
	// correctness bug (gpu/optix/wavefront_restir_helpers.h's temporal
	// combine and wavefront_kernels_restir.cu's spatial reuse both clamped a
	// reservoir's M AFTER folding its weight into weightSum, leaving
	// weightSum built from a larger M than restir_finalize() then divided by
	// - inflating W, which compounded every frame since that inflated W fed
	// forward as next frame's own input) that made ReSTIR's converged image
	// mean ~44x classic NEE's for the same scene. With that bug fixed at the
	// source, Live Preview's own output is back at parity with classic NEE
	// (confirmed via a direct instrumented comparison) and no longer needs a
	// darkening workaround - setExposure() itself stays, as a legitimate,
	// generically useful control.
	double m_exposure = 1.0;
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
	// Per-pixel effective BATCH count - NOT uniform once reprojection is in
	// play (a freshly-disoccluded pixel starts over at 0 while a
	// successfully-reprojected neighbor carries its whole history forward),
	// unlike the single scalar this replaced. m_sampleCount (below) becomes
	// the MINIMUM across all pixels once rendering starts - a conservative
	// "worst-converged pixel" indicator for the status label (see
	// MainWindow::onLivePreviewFrameReady()) rather than a literal count
	// that stopped being uniform. "Batch" because each increment is one
	// render call's own m_spp-sample average, not one raw sample - the
	// frameReady() emit site multiplies by m_spp before it reaches the
	// label, so the label shows real samples traced while this stays the
	// batch-weighted count the running-mean math (renderLoop()) actually
	// needs as its divisor.
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
			   bool denoise, double denoiseBlend, bool denoiseShowLatest, bool svgf,
			   bool restirGi, bool restirDi, int spp, int maxDepth, double fireflyClamp);
	void stop();
	void setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ);
	void setDenoise(bool denoise, double denoiseBlend, bool denoiseShowLatest);
	void setSvgf(bool svgf);
	void setExposure(double exposure);
	void setRestirGi(bool restirGi);
	void setRestirDi(bool restirDi);
	void setSppAndMaxDepth(int spp, int maxDepth);
	void setFireflyClamp(double fireflyClamp);
	void setSvgfTuning(double temporalAlpha, double maxHistoryLength, double varianceBootstrapFrames,
						int varianceBootstrapRadius, double sigmaNormal, double sigmaDepth,
						double sigmaLuminance, int atrousRadius, double minAlbedo, int atrousPasses);

signals:
	void frameReady(QImage image, int sampleCount);
	void statusChanged(QString text);

private:
	QThread m_thread;
	RealtimePreviewWorker *m_worker;
};

#endif // REALTIME_PREVIEW_SESSION_H
