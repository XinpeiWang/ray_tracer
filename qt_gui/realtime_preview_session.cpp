#include "realtime_preview_session.h"
#include "../src/shared/tone_map.h"
#include "cross_abi_library.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <cmath>
#include <mutex>

namespace {

// bool(const char* scene_id, int w, int h, int spp, int max_depth,
//      double camX, double camY, double camZ,
//      bool has_custom_lookat, double lookX, double lookY, double lookZ,
//      float* out_rgb)
typedef bool (*RenderFrameFn)(const char*, int, int, int, int, double, double, double,
							   bool, double, double, double, float*);

struct DllHandle {
	void* module = nullptr;
	RenderFrameFn renderFrameFn = nullptr;
};

#ifdef Q_OS_WIN
constexpr const char* kLibraryFileName = "realtime_renderer.dll";
#elif defined(Q_OS_MAC)
constexpr const char* kLibraryFileName = "realtime_renderer.dylib";
#else
constexpr const char* kLibraryFileName = "realtime_renderer.so";
#endif

// Same LoadLibrary/GetProcAddress-across-the-MinGW/MSVC-ABI-boundary
// approach as scene_metadata_client.cpp - see that file's own header
// comment for why a DLL boundary is needed at all here. std::call_once
// since RealtimePreviewWorker::start()/renderLoop() all run on the worker
// thread, not the GUI thread that (re)creates RealtimePreviewSession.
DllHandle& handle() {
	static DllHandle h;
	static std::once_flag loadOnce;
	std::call_once(loadOnce, [&h]() {
		h.module = cross_abi_library::loadLibrary(QCoreApplication::applicationDirPath(), kLibraryFileName);
		if (!h.module) return;
		h.renderFrameFn = reinterpret_cast<RenderFrameFn>(
			cross_abi_library::lookupSymbol(h.module, "realtime_render_frame"));
	});
	return h;
}

} // namespace

bool RealtimePreviewSession::isAvailable() {
	return handle().renderFrameFn != nullptr;
}

// ============================================================================
// RealtimePreviewWorker
// ============================================================================

void RealtimePreviewWorker::resetAccumulation() {
	m_accum.assign(static_cast<size_t>(m_width) * m_height * 3, 0.0f);
	m_tmp.assign(static_cast<size_t>(m_width) * m_height * 3, 0.0f);
	m_displayImage = QImage(m_width, m_height, QImage::Format_RGB888);
	m_sampleCount = 0;
}

void RealtimePreviewWorker::start(QString sceneId, int width, int height, double camX, double camY, double camZ,
								   double lookX, double lookY, double lookZ) {
	m_sceneId = sceneId;
	m_width = width;
	m_height = height;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_lookX = lookX;
	m_lookY = lookY;
	m_lookZ = lookZ;
	m_cameraDirty = false;
	resetAccumulation();
	m_running = true;
	renderLoop(++m_epoch);
}

void RealtimePreviewWorker::stop() {
	m_running = false;
}

void RealtimePreviewWorker::setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ) {
	if (!m_running) return;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_lookX = lookX;
	m_lookY = lookY;
	m_lookZ = lookZ;
	m_cameraDirty = true;
}

void RealtimePreviewWorker::renderLoop(int epoch) {
	// epoch != m_epoch means a stop()+start() cycle already happened since
	// THIS continuation was posted (start() bumps m_epoch) - it belongs to
	// an old, already-superseded chain and must not run at all, let alone
	// repost itself, or it would keep running forever alongside the new
	// chain start() began. See m_epoch's own header comment for the exact
	// race this closes.
	if (!m_running || epoch != m_epoch) return;

	if (m_cameraDirty) {
		m_cameraDirty = false;
		resetAccumulation();
	}

	RenderFrameFn renderFrame = handle().renderFrameFn;
	bool ok = false;
	if (!renderFrame) {
		emit statusChanged(QStringLiteral("realtime_renderer.dll not found or missing its export"));
		m_running = false;
	} else {
		// One low-spp sample per loop iteration - see this class's own header
		// comment on why: a fixed small per-call cost keeps the loop responsive
		// to stop()/setCamera() between frames, and frameNumber_'s own
		// self-incrementing seed (WavefrontPathTracer, see optix_interface.h's
		// rt_realtime_render_frame() comment) already decorrelates noise across
		// calls, so accumulating many 1-spp calls converges the same way one
		// big N-spp call would.
		const int spp = 1;
		const int maxDepth = 8;
		ok = renderFrame(m_sceneId.toUtf8().constData(), m_width, m_height, spp, maxDepth,
						  m_camX, m_camY, m_camZ,
						  /*has_custom_lookat=*/true, m_lookX, m_lookY, m_lookZ,
						  m_tmp.data());
		if (!ok) {
			emit statusChanged(QStringLiteral("Render failed - scene may not be GPU-supported, "
											   "or the wavefront backend is unavailable"));
		}
	}

	if (ok) {
		// Running mean: accum += (sample - accum) / (n+1). Both buffers are
		// linear RGB (rt_realtime_render_frame()'s own contract), so this is
		// a plain per-channel average - no dividing/multiplying needed
		// beyond this, unlike CPU/GPU's own filter-weighted reconstruction
		// (this preview uses a trivial 1-sample-per-pixel box filter, no
		// splatting).
		const int n = m_sampleCount;
		for (size_t i = 0; i < m_accum.size(); ++i) {
			m_accum[i] += (m_tmp[i] - m_accum[i]) / static_cast<float>(n + 1);
		}
		++m_sampleCount;

		// Tonemap the ACCUMULATED result (ACES + sRGB, matching this
		// project's CPU/GPU display convention exactly - see tone_map.h)
		// into m_displayImage IN PLACE. Tonemapping the per-call noisy
		// sample instead would defeat the whole point of accumulating in
		// linear space first.
		for (int y = 0; y < m_height; ++y) {
			uchar* row = m_displayImage.scanLine(y);
			for (int x = 0; x < m_width; ++x) {
				const size_t idx = (static_cast<size_t>(y) * m_width + x) * 3;
				double r = m_accum[idx + 0], g = m_accum[idx + 1], b = m_accum[idx + 2];
				if (!std::isfinite(r)) r = 0.0;
				if (!std::isfinite(g)) g = 0.0;
				if (!std::isfinite(b)) b = 0.0;
				r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
				g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
				b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
				row[x * 3 + 0] = static_cast<uchar>(std::fmin(std::fmax(r, 0.0), 1.0) * 255.0 + 0.5);
				row[x * 3 + 1] = static_cast<uchar>(std::fmin(std::fmax(g, 0.0), 1.0) * 255.0 + 0.5);
				row[x * 3 + 2] = static_cast<uchar>(std::fmin(std::fmax(b, 0.0), 1.0) * 255.0 + 0.5);
			}
		}

		// .copy() rather than emitting m_displayImage directly: QImage is
		// implicitly shared, and frameReady() carries this image across a
		// queued cross-thread signal to the GUI thread - without a detach,
		// the NEXT loop iteration's in-place scanLine() writes (above,
		// possibly before the GUI thread has consumed the queued copy)
		// would race with whatever the GUI thread reads from the "same"
		// shared buffer. .copy() gives this emit its own buffer up front,
		// letting m_displayImage keep being mutated in place next iteration
		// with no such hazard.
		emit frameReady(m_displayImage.copy(), m_sampleCount);
	}

	if (m_running) QMetaObject::invokeMethod(this, [this, epoch]() { renderLoop(epoch); }, Qt::QueuedConnection);
}

// ============================================================================
// RealtimePreviewSession
// ============================================================================

RealtimePreviewSession::RealtimePreviewSession(QObject *parent) : QObject(parent) {
	m_worker = new RealtimePreviewWorker();
	m_worker->moveToThread(&m_thread);
	// Worker is parent-less and owned by the thread's own finished() cleanup
	// below, not by this QObject - it must not be destroyed from the GUI
	// thread while the worker thread might still be running a queued call
	// against it.
	connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
	connect(m_worker, &RealtimePreviewWorker::frameReady, this, &RealtimePreviewSession::frameReady);
	connect(m_worker, &RealtimePreviewWorker::statusChanged, this, &RealtimePreviewSession::statusChanged);
	m_thread.start();
}

RealtimePreviewSession::~RealtimePreviewSession() {
	stop();
	m_thread.quit();
	m_thread.wait();
}

void RealtimePreviewSession::start(const QString &sceneId, int width, int height, double camX, double camY, double camZ,
									double lookX, double lookY, double lookZ) {
	QMetaObject::invokeMethod(m_worker, "start", Qt::QueuedConnection,
		Q_ARG(QString, sceneId), Q_ARG(int, width), Q_ARG(int, height),
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ),
		Q_ARG(double, lookX), Q_ARG(double, lookY), Q_ARG(double, lookZ));
}

void RealtimePreviewSession::stop() {
	QMetaObject::invokeMethod(m_worker, "stop", Qt::QueuedConnection);
}

void RealtimePreviewSession::setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ) {
	QMetaObject::invokeMethod(m_worker, "setCamera", Qt::QueuedConnection,
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ),
		Q_ARG(double, lookX), Q_ARG(double, lookY), Q_ARG(double, lookZ));
}
