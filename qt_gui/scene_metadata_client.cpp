#include "scene_metadata_client.h"

#include <windows.h>
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <mutex>

namespace {

typedef int (*GpuCompatibleFn)(const char*);
typedef int (*RecommendedCameraFn)(const char*, double*, double*, double*, double*, double*, double*);
typedef int (*CountFn)();
typedef const char* (*IdAtIndexFn)(int);
typedef const char* (*StringByIdFn)(const char*);
typedef int (*IntByIdFn)(const char*);

struct DllHandle {
	HMODULE module = nullptr;
	GpuCompatibleFn gpuCompatibleFn = nullptr;
	RecommendedCameraFn recommendedCameraFn = nullptr;
	CountFn countFn = nullptr;
	IdAtIndexFn idAtIndexFn = nullptr;
	StringByIdFn nameFn = nullptr;
	StringByIdFn categoryFn = nullptr;
	StringByIdFn descriptionFn = nullptr;
	StringByIdFn performanceFn = nullptr;
	IntByIdFn recommendedSppFn = nullptr;
	IntByIdFn requiresFilesFn = nullptr;
};

// All current callers are on the GUI thread, but std::call_once (rather than
// a plain "attempted" bool) keeps the one-time DLL load race-free if this is
// ever called from another thread again - it previously was, from the render
// worker thread that RenderController replaced.
DllHandle& handle() {
	static DllHandle h;
	static std::once_flag loadOnce;
	std::call_once(loadOnce, [&h]() {
		QString dllPath = QDir(QCoreApplication::applicationDirPath()).filePath("scene_metadata.dll");
		h.module = LoadLibraryW(reinterpret_cast<const wchar_t*>(dllPath.utf16()));
		if (!h.module) return;

		h.gpuCompatibleFn = reinterpret_cast<GpuCompatibleFn>(
			GetProcAddress(h.module, "scene_metadata_gpu_compatible"));
		h.recommendedCameraFn = reinterpret_cast<RecommendedCameraFn>(
			GetProcAddress(h.module, "scene_metadata_recommended_camera"));
		h.countFn = reinterpret_cast<CountFn>(
			GetProcAddress(h.module, "scene_metadata_count"));
		h.idAtIndexFn = reinterpret_cast<IdAtIndexFn>(
			GetProcAddress(h.module, "scene_metadata_id_at_index"));
		h.nameFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_name"));
		h.categoryFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_category"));
		h.descriptionFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_description"));
		h.performanceFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_performance"));
		h.recommendedSppFn = reinterpret_cast<IntByIdFn>(
			GetProcAddress(h.module, "scene_metadata_recommended_spp"));
		h.requiresFilesFn = reinterpret_cast<IntByIdFn>(
			GetProcAddress(h.module, "scene_metadata_requires_files"));

		// Every export is required, including newer ones: a DLL missing any of
		// them is a stale build sitting next to a newer exe, and half-working
		// metadata (a scene list with no categories, say) is harder to diagnose
		// than the outright "couldn't load" the caller already handles.
		if (!h.gpuCompatibleFn || !h.recommendedCameraFn || !h.countFn || !h.idAtIndexFn ||
			!h.nameFn || !h.categoryFn || !h.descriptionFn || !h.performanceFn ||
			!h.recommendedSppFn || !h.requiresFilesFn) {
			FreeLibrary(h.module);
			h = DllHandle{};
		}
	});
	return h;
}

} // namespace

namespace SceneMetadataClient {

bool ensureLoaded() {
	return handle().module != nullptr;
}

bool gpuCompatible(const QString& scene_id, bool& out_compatible) {
	if (!ensureLoaded()) return false;
	out_compatible = handle().gpuCompatibleFn(scene_id.toUtf8().constData()) != 0;
	return true;
}

bool recommendedCamera(const QString& scene_id,
	double& cam_x, double& cam_y, double& cam_z,
	double& lookat_x, double& lookat_y, double& lookat_z) {
	if (!ensureLoaded()) return false;
	return handle().recommendedCameraFn(scene_id.toUtf8().constData(),
		&cam_x, &cam_y, &cam_z, &lookat_x, &lookat_y, &lookat_z) != 0;
}

int sceneCount() {
	if (!ensureLoaded()) return 0;
	return handle().countFn();
}

QString sceneIdAtIndex(int index) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().idAtIndexFn(index));
}

QString sceneName(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().nameFn(scene_id.toUtf8().constData()));
}

QString sceneCategory(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().categoryFn(scene_id.toUtf8().constData()));
}

QString sceneDescription(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().descriptionFn(scene_id.toUtf8().constData()));
}

QString scenePerformance(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().performanceFn(scene_id.toUtf8().constData()));
}

int sceneRecommendedSpp(const QString& scene_id) {
	if (!ensureLoaded()) return 100;
	return handle().recommendedSppFn(scene_id.toUtf8().constData());
}

bool sceneRequiresFiles(const QString& scene_id) {
	if (!ensureLoaded()) return false;
	return handle().requiresFilesFn(scene_id.toUtf8().constData()) != 0;
}

} // namespace SceneMetadataClient
