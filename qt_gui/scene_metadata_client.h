#ifndef SCENE_METADATA_CLIENT_H
#define SCENE_METADATA_CLIENT_H

#include <QString>

// ============================================================================
// SceneMetadataClient
// ============================================================================
// The GUI is built with MinGW (see RayTracerGUI.pro) and cannot link
// cpu_renderer.lib (MSVC-built, ABI-incompatible) to query the actual
// per-scene camera/GPU-support data in scene_registry.h. scene_metadata.dll
// (see ../scene_metadata/scene_metadata_dll.cpp) re-exports that data as
// plain extern "C" functions - x64 Windows has one calling convention
// regardless of compiler, so this process can call into an MSVC-built DLL
// dynamically via LoadLibrary/GetProcAddress even though it can't link its
// import lib at build time.
//
// If the DLL can't be loaded (missing from the app directory, wrong
// architecture, etc.) every query here simply fails - callers must handle
// that by leaving whatever value they already had, not by crashing.
// ============================================================================
namespace SceneMetadataClient {

// Loads scene_metadata.dll from the application directory. Safe to call
// repeatedly; only actually loads once. Returns false (and logs nothing -
// callers decide whether/how to surface this) if the DLL isn't found or
// doesn't export the expected functions.
bool ensureLoaded();

// Returns true and sets out_compatible if scene_id's GPU-compatibility
// could be queried; false (out_compatible left untouched) otherwise.
bool gpuCompatible(const QString& scene_id, bool& out_compatible);

// Returns true and fills every output if scene_id's recommended camera
// could be queried; false (all outputs left untouched) otherwise.
bool recommendedCamera(const QString& scene_id,
	double& cam_x, double& cam_y, double& cam_z,
	double& lookat_x, double& lookat_y, double& lookat_z);

// Presentational scene metadata (name/description/performance/recommended
// SPP/requires_files) - the sole source now for what used to be a separate
// GUI-local copy in src/shared/scene_descriptor.h (see that header's
// comment). Every scene's row is queried by id, not enumerated as a batch
// like the old get_all_scenes() did - callers loop index in
// [0, sceneCount()) and resolve each position's id via sceneIdAtIndex()
// first, since ids are category letter + number now, not contiguous ints
// (see scene_registry.h's SceneDescriptor::id comment).
// All return their type's "empty" value (0 / "" / false) if the DLL isn't
// loaded or scene_id isn't found - same "fail quiet, let the caller's
// existing state stand" contract as gpuCompatible/recommendedCamera above.

// Total number of registered scenes, or 0 if the DLL isn't loaded.
int sceneCount();

// The id (category letter + number, e.g. "B10") of the scene at registry
// position index, or "" if not loaded/index out of range. The only bridge
// from "position in registry" to "id" - every other function below takes
// an id, not a position.
QString sceneIdAtIndex(int index);

// scene_id's display name, or "" if not loaded/found.
QString sceneName(const QString& scene_id);

// scene_id's category ("Basics", "Materials", ...), or "" if not
// loaded/found. Matches one of the SceneCategories:: constants in
// src/shared/scene_descriptor.h; the GUI groups its scene list by this.
QString sceneCategory(const QString& scene_id);

// scene_id's short description, or "" if not loaded/found.
QString sceneDescription(const QString& scene_id);

// scene_id's performance hint ("Fast"/"Medium"/"Slow"/"Very Slow"), or ""
// if not loaded/found.
QString scenePerformance(const QString& scene_id);

// scene_id's recommended samples-per-pixel, or 100 (the same fallback
// cpu_interface.cpp's own accessors use) if not loaded/found.
int sceneRecommendedSpp(const QString& scene_id);

// True if scene_id requires external asset files (e.g. earthmap.jpg);
// false if not loaded/found - erring toward "no special files needed"
// rather than surfacing a spurious warning.
bool sceneRequiresFiles(const QString& scene_id);

// scene_id's recommended Sampler/Integrator/light sampler, as raw pbrt
// directive strings (e.g. "halton", "bdpt", "power") directly comparable
// to m_samplerCombo/m_integratorCombo/m_lightSamplerCombo's own
// currentData() values - or "" if not loaded/found/the scene's file
// doesn't declare that directive (every hand-built scene always returns
// ""). Not auto-applied anywhere; see cpu_interface.cpp's own comment on
// why the CLI only warns rather than switching settings for the user.
QString sceneRecommendedIntegrator(const QString& scene_id);
QString sceneRecommendedSampler(const QString& scene_id);
QString sceneRecommendedLightSampler(const QString& scene_id);

// scene_id's curated recommended Exposure multiplier, or 1.0 (neutral, same
// fallback cpu_interface.cpp's own accessor uses) if not loaded/found.
// UNLIKE sceneRecommendedIntegrator/Sampler/LightSampler above, this one IS
// meant to be auto-applied - see onSceneChanged()'s call site.
double sceneRecommendedExposure(const QString& scene_id);

} // namespace SceneMetadataClient

#endif // SCENE_METADATA_CLIENT_H
