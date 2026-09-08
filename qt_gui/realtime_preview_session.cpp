#include "realtime_preview_session.h"
#include "../src/shared/tone_map.h"

#ifdef Q_OS_WIN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <QCoreApplication>
#include <QDir>
#include <QMetaObject>
#include <cmath>
#include <mutex>

namespace {

// bool(const char* scene_id, int w, int h, int spp, int max_depth,
//      double camX, double camY, double camZ, float* out_rgb)
typedef bool (*RenderFrameFn)(const char*, int, int, int, int, double, double, double, float*);

struct DllHandle {
	void* module = nullptr;
	RenderFrameFn renderFrameFn = nullptr;
};

#ifdef Q_OS_WIN
void* loadLibraryFrom(const QString& dir, const QString& name) {
	QString path = QDir(dir).filePath(name);
	return static_cast<void*>(LoadLibraryW(reinterpret_cast<const wchar_t*>(path.utf16())));
}
void* lookupSymbol(void* module, const char* name) {
	return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
}
#else
void* loadLibraryFrom(const QString& dir, const QString& name) {
	QString path = QDir(dir).filePath(name);
	return dlopen(path.toUtf8().constData(), RTLD_NOW | RTLD_LOCAL);
}
void* lookupSymbol(void* module, const char* name) {
	return dlsym(module, name);
}
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
#ifdef Q_OS_WIN
		h.module = loadLibraryFrom(QCoreApplication::applicationDirPath(), "realtime_renderer.dll");
#elif defined(Q_OS_MAC)
		h.module = loadLibraryFrom(QCoreApplication::applicationDirPath(), "realtime_renderer.dylib");
#else
		h.module = loadLibraryFrom(QCoreApplication::applicationDirPath(), "realtime_renderer.so");
#endif
		if (!h.module) return;
		h.renderFrameFn = reinterpret_cast<RenderFrameFn>(lookupSymbol(h.module, "realtime_render_frame"));
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
	m_sampleCount = 0;
}

void RealtimePreviewWorker::start(QString sceneId, int width, int height, double camX, double camY, double camZ) {
	m_sceneId = sceneId;
	m_width = width;
	m_height = height;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_cameraDirty = false;
	resetAccumulation();
	m_running = true;
	renderLoop();
}

void RealtimePreviewWorker::stop() {
	m_running = false;
}

void RealtimePreviewWorker::setCamera(double camX, double camY, double camZ) {
	if (!m_running) return;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_cameraDirty = true;
}

void RealtimePreviewWorker::renderLoop() {
	if (!m_running) return;

	if (m_cameraDirty.exchange(false)) resetAccumulation();

	RenderFrameFn renderFrame = handle().renderFrameFn;
	if (!renderFrame) {
		emit statusChanged(QStringLiteral("realtime_renderer.dll not found or missing its export"));
		m_running = false;
		return;
	}

	// One low-spp sample per loop iteration - see this class's own header
	// comment on why: a fixed small per-call cost keeps the loop responsive
	// to stop()/setCamera() between frames, and frameNumber_'s own
	// self-incrementing seed (WavefrontPathTracer, see optix_interface.h's
	// rt_realtime_render_frame() comment) already decorrelates noise across
	// calls, so accumulating many 1-spp calls converges the same way one
	// big N-spp call would.
	const int spp = 1;
	const int maxDepth = 8;
	std::vector<float> tmp(static_cast<size_t>(m_width) * m_height * 3);
	const bool ok = renderFrame(m_sceneId.toUtf8().constData(), m_width, m_height, spp, maxDepth,
								 m_camX, m_camY, m_camZ, tmp.data());
	if (!ok) {
		emit statusChanged(QStringLiteral("Render failed - scene may not be GPU-supported"));
		if (m_running) QMetaObject::invokeMethod(this, &RealtimePreviewWorker::renderLoop, Qt::QueuedConnection);
		return;
	}

	// Running mean: accum += (sample - accum) / (n+1). Both buffers are
	// linear RGB (rt_realtime_render_frame()'s own contract), so this is a
	// plain per-channel average - no dividing/multiplying needed beyond
	// this, unlike CPU/GPU's own filter-weighted reconstruction (this
	// preview uses a trivial 1-sample-per-pixel box filter, no splatting).
	const int n = m_sampleCount;
	for (size_t i = 0; i < m_accum.size(); ++i) {
		m_accum[i] += (tmp[i] - m_accum[i]) / static_cast<float>(n + 1);
	}
	++m_sampleCount;

	// Tonemap the ACCUMULATED result (ACES + sRGB, matching this project's
	// CPU/GPU display convention exactly - see tone_map.h) into a QImage.
	// Tonemapping the per-call noisy sample instead would defeat the whole
	// point of accumulating in linear space first.
	QImage image(m_width, m_height, QImage::Format_RGB888);
	for (int y = 0; y < m_height; ++y) {
		uchar* row = image.scanLine(y);
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

	emit frameReady(image, m_sampleCount);

	if (m_running) QMetaObject::invokeMethod(this, &RealtimePreviewWorker::renderLoop, Qt::QueuedConnection);
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

void RealtimePreviewSession::start(const QString &sceneId, int width, int height, double camX, double camY, double camZ) {
	QMetaObject::invokeMethod(m_worker, "start", Qt::QueuedConnection,
		Q_ARG(QString, sceneId), Q_ARG(int, width), Q_ARG(int, height),
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ));
}

void RealtimePreviewSession::stop() {
	QMetaObject::invokeMethod(m_worker, "stop", Qt::QueuedConnection);
}

void RealtimePreviewSession::setCamera(double camX, double camY, double camZ) {
	QMetaObject::invokeMethod(m_worker, "setCamera", Qt::QueuedConnection,
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ));
}
