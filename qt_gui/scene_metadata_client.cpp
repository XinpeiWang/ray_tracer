#include "scene_metadata_client.h"

#ifdef Q_OS_WIN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
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
typedef double (*DoubleByIdFn)(const char*);

// Raw mirror of cpu_interface.h's SceneMetadataSnapshot - field-for-field,
// same order, same types. Both sides are plain data (ints/doubles/const
// char*), so this is standard-layout and safe across the DLL boundary the
// same way every individual int/double/const char* parameter above already
// is (see this file's own top-of-file ABI comment) - there's no shared
// header between the MSVC-built DLL and this MinGW-built client to enforce
// that agreement, so keep this in sync by hand if SceneMetadataSnapshot
// ever changes.
struct RawSceneMetadataSnapshot {
	const char* name;
	const char* category;
	const char* description;
	const char* performance;
	int recommended_spp;
	int requires_files;
	int gpu_compatible;
	double recommended_exposure;
	const char* recommended_integrator;
	const char* recommended_sampler;
	const char* recommended_light_sampler;
	double cam_lookfrom_x, cam_lookfrom_y, cam_lookfrom_z;
	double cam_lookat_x, cam_lookat_y, cam_lookat_z;
};
typedef int (*SnapshotFn)(const char*, RawSceneMetadataSnapshot*);

struct DllHandle {
	// HMODULE (Windows) and dlopen()'s return type are both opaque handles
	// that fit in a void* - stored as void* here so the struct itself needs
	// no platform-conditional field, only handle()'s load/lookup/close calls
	// below do.
	void* module = nullptr;
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
	StringByIdFn recommendedIntegratorFn = nullptr;
	StringByIdFn recommendedSamplerFn = nullptr;
	StringByIdFn recommendedLightSamplerFn = nullptr;
	DoubleByIdFn recommendedExposureFn = nullptr;
	SnapshotFn snapshotFn = nullptr;
};

#ifdef Q_OS_WIN
void* loadSceneMetadataLibrary(const QString& dir) {
	QString path = QDir(dir).filePath("scene_metadata.dll");
	return static_cast<void*>(LoadLibraryW(reinterpret_cast<const wchar_t*>(path.utf16())));
}
void* lookupSymbol(void* module, const char* name) {
	return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
}
void closeLibrary(void* module) {
	FreeLibrary(static_cast<HMODULE>(module));
}
#else
void* loadSceneMetadataLibrary(const QString& dir) {
	// scene_metadata.dylib on macOS, scene_metadata.so on Linux - matches
	// CMakeLists.txt's `set_target_properties(scene_metadata PROPERTIES
	// PREFIX "")`, which drops CMake's default "lib" prefix so this exact
	// filename is what actually gets built.
#ifdef Q_OS_MAC
	QString path = QDir(dir).filePath("scene_metadata.dylib");
#else
	QString path = QDir(dir).filePath("scene_metadata.so");
#endif
	return dlopen(path.toUtf8().constData(), RTLD_NOW | RTLD_LOCAL);
}
void* lookupSymbol(void* module, const char* name) {
	return dlsym(module, name);
}
void closeLibrary(void* module) {
	dlclose(module);
}
#endif

// All current callers are on the GUI thread, but std::call_once (rather than
// a plain "attempted" bool) keeps the one-time DLL load race-free if this is
// ever called from another thread again - it previously was, from the render
// worker thread that RenderController replaced.
DllHandle& handle() {
	static DllHandle h;
	static std::once_flag loadOnce;
	std::call_once(loadOnce, [&h]() {
		h.module = loadSceneMetadataLibrary(QCoreApplication::applicationDirPath());
		if (!h.module) return;

		h.gpuCompatibleFn = reinterpret_cast<GpuCompatibleFn>(
			lookupSymbol(h.module, "scene_metadata_gpu_compatible"));
		h.recommendedCameraFn = reinterpret_cast<RecommendedCameraFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_camera"));
		h.countFn = reinterpret_cast<CountFn>(
			lookupSymbol(h.module, "scene_metadata_count"));
		h.idAtIndexFn = reinterpret_cast<IdAtIndexFn>(
			lookupSymbol(h.module, "scene_metadata_id_at_index"));
		h.nameFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_name"));
		h.categoryFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_category"));
		h.descriptionFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_description"));
		h.performanceFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_performance"));
		h.recommendedSppFn = reinterpret_cast<IntByIdFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_spp"));
		h.requiresFilesFn = reinterpret_cast<IntByIdFn>(
			lookupSymbol(h.module, "scene_metadata_requires_files"));
		h.recommendedIntegratorFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_integrator"));
		h.recommendedSamplerFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_sampler"));
		h.recommendedLightSamplerFn = reinterpret_cast<StringByIdFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_light_sampler"));
		h.recommendedExposureFn = reinterpret_cast<DoubleByIdFn>(
			lookupSymbol(h.module, "scene_metadata_recommended_exposure"));
		h.snapshotFn = reinterpret_cast<SnapshotFn>(
			lookupSymbol(h.module, "scene_metadata_snapshot"));

		// Every export is required, including newer ones: a library missing
		// any of them is a stale build sitting next to a newer exe, and
		// half-working metadata (a scene list with no categories, say) is
		// harder to diagnose than the outright "couldn't load" the caller
		// already handles.
		if (!h.gpuCompatibleFn || !h.recommendedCameraFn || !h.countFn || !h.idAtIndexFn ||
			!h.nameFn || !h.categoryFn || !h.descriptionFn || !h.performanceFn ||
			!h.recommendedSppFn || !h.requiresFilesFn || !h.recommendedIntegratorFn ||
			!h.recommendedSamplerFn || !h.recommendedLightSamplerFn || !h.recommendedExposureFn ||
			!h.snapshotFn) {
			closeLibrary(h.module);
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

QString sceneRecommendedIntegrator(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().recommendedIntegratorFn(scene_id.toUtf8().constData()));
}

QString sceneRecommendedSampler(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().recommendedSamplerFn(scene_id.toUtf8().constData()));
}

QString sceneRecommendedLightSampler(const QString& scene_id) {
	if (!ensureLoaded()) return QString();
	return QString::fromUtf8(handle().recommendedLightSamplerFn(scene_id.toUtf8().constData()));
}

double sceneRecommendedExposure(const QString& scene_id) {
	if (!ensureLoaded()) return 1.0;
	return handle().recommendedExposureFn(scene_id.toUtf8().constData());
}

bool sceneMetadata(const QString& scene_id, SceneMetadata& out) {
	if (!ensureLoaded()) return false;
	RawSceneMetadataSnapshot raw{};
	if (!handle().snapshotFn(scene_id.toUtf8().constData(), &raw)) return false;

	out.name = QString::fromUtf8(raw.name);
	out.category = QString::fromUtf8(raw.category);
	out.description = QString::fromUtf8(raw.description);
	out.performance = QString::fromUtf8(raw.performance);
	out.recommendedSpp = raw.recommended_spp;
	out.requiresFiles = raw.requires_files != 0;
	out.gpuCompatible = raw.gpu_compatible != 0;
	out.recommendedExposure = raw.recommended_exposure;
	out.recommendedIntegrator = QString::fromUtf8(raw.recommended_integrator);
	out.recommendedSampler = QString::fromUtf8(raw.recommended_sampler);
	out.recommendedLightSampler = QString::fromUtf8(raw.recommended_light_sampler);
	out.camLookfromX = raw.cam_lookfrom_x;
	out.camLookfromY = raw.cam_lookfrom_y;
	out.camLookfromZ = raw.cam_lookfrom_z;
	out.camLookatX = raw.cam_lookat_x;
	out.camLookatY = raw.cam_lookat_y;
	out.camLookatZ = raw.cam_lookat_z;
	return true;
}

} // namespace SceneMetadataClient
