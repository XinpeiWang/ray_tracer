#include "scene_metadata_client.h"

#include <windows.h>
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <mutex>

namespace {

typedef int (*GpuCompatibleFn)(int);
typedef int (*RecommendedCameraFn)(int, double*, double*, double*, double*, double*, double*);

struct DllHandle {
	HMODULE module = nullptr;
	GpuCompatibleFn gpuCompatibleFn = nullptr;
	RecommendedCameraFn recommendedCameraFn = nullptr;
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

		if (!h.gpuCompatibleFn || !h.recommendedCameraFn) {
			FreeLibrary(h.module);
			h.module = nullptr;
			h.gpuCompatibleFn = nullptr;
			h.recommendedCameraFn = nullptr;
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

} // namespace SceneMetadataClient
