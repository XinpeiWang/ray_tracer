#include "scene_metadata_client.h"

#include <windows.h>
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <mutex>

namespace {

typedef int (*GpuCompatibleFn)(int);
typedef int (*RecommendedCameraFn)(int, double*, double*, double*, double*, double*, double*);
typedef int (*CountFn)();
typedef const char* (*StringByIdFn)(int);
typedef int (*IntByIdFn)(int);

struct DllHandle {
	HMODULE module = nullptr;
	GpuCompatibleFn gpuCompatibleFn = nullptr;
	RecommendedCameraFn recommendedCameraFn = nullptr;
	CountFn countFn = nullptr;
	StringByIdFn nameFn = nullptr;
	StringByIdFn descriptionFn = nullptr;
	StringByIdFn performanceFn = nullptr;
	IntByIdFn recommendedSppFn = nullptr;
	IntByIdFn requiresFilesFn = nullptr;
};

// Callers span both the GUI thread (onSceneChanged, onRenderClicked) and
// RenderThread's worker thread (mainwindow.cpp's run(), via
// ErrorHandler::getTroubleshootingHint -> gpuSupportedSceneList for exit
// code 211) - std::call_once, not a plain "attempted" bool, makes the
// first load race-free regardless of which thread gets there first.
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
		h.nameFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_name"));
		h.descriptionFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_description"));
		h.performanceFn = reinterpret_cast<StringByIdFn>(
			GetProcAddress(h.module, "scene_metadata_performance"));
		h.recommendedSppFn = reinterpret_cast<IntByIdFn>(
			GetProcAddress(h.module, "scene_metadata_recommended_spp"));
		h.requiresFilesFn = reinterpret_cast<IntByIdFn>(
			GetProcAddress(h.module, "scene_metadata_requires_files"));

		if (!h.gpuCompatibleFn || !h.recommendedCameraFn || !h.countFn || !h.nameFn ||
			!h.descriptionFn || !h.performanceFn || !h.recommendedSppFn || !h.requiresFilesFn) {
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

bool gpuCompatible(int scene_id, bool& out_compatible) {
	if (!ensureLoaded()) return false;
	out_compatible = handle().gpuCompatibleFn(scene_id) != 0;
	return true;
}

bool recommendedCamera(int scene_id,
	double& cam_x, double& cam_y, double& cam_z,
	double& lookat_x, double& lookat_y, double& lookat_z) {
	if (!ensureLoaded()) return false;
	return handle().recommendedCameraFn(scene_id,
		&cam_x, &cam_y, &cam_z, &lookat_x, &lookat_y, &lookat_z) != 0;
}

int sceneCount() {
	if (!ensureLoaded()) return 0;
	return handle().countFn();
}

QString sceneName(int scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().nameFn(scene_id));
}

QString sceneDescription(int scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().descriptionFn(scene_id));
}

QString scenePerformance(int scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().performanceFn(scene_id));
}

int sceneRecommendedSpp(int scene_id) {
	if (!ensureLoaded()) return 100;
	return handle().recommendedSppFn(scene_id);
}

bool sceneRequiresFiles(int scene_id) {
	if (!ensureLoaded()) return false;
	return handle().requiresFilesFn(scene_id) != 0;
}

} // namespace SceneMetadataClient
