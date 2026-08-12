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
bool gpuCompatible(int scene_id, bool& out_compatible);

// Returns true and fills every output if scene_id's recommended camera
// could be queried; false (all outputs left untouched) otherwise.
bool recommendedCamera(int scene_id,
	double& cam_x, double& cam_y, double& cam_z,
	double& lookat_x, double& lookat_y, double& lookat_z);

// Presentational scene metadata (name/description/performance/recommended
// SPP/requires_files) - the sole source now for what used to be a separate
// GUI-local copy in src/shared/scene_descriptor.h (see that header's
// comment). Every scene's row is queried by id, not enumerated as a batch
// like the old get_all_scenes() did - callers loop id in
// [0, sceneCount()) themselves, relying on the registry's ids being
// contiguous from 0 (tests/unit/scene_registry_tests.cpp enforces this).
// All return their type's "empty" value (0 / "" / false) if the DLL isn't
// loaded or scene_id isn't found - same "fail quiet, let the caller's
// existing state stand" contract as gpuCompatible/recommendedCamera above.

// Total number of registered scenes, or 0 if the DLL isn't loaded.
int sceneCount();

// scene_id's display name, or "" if not loaded/found.
QString sceneName(int scene_id);

// scene_id's category ("Basics", "Materials", ...), or "" if not
// loaded/found. Matches one of the SceneCategories:: constants in
// src/shared/scene_descriptor.h; the GUI groups its scene list by this.
QString sceneCategory(int scene_id);

// scene_id's short description, or "" if not loaded/found.
QString sceneDescription(int scene_id);

// scene_id's performance hint ("Fast"/"Medium"/"Slow"/"Very Slow"), or ""
// if not loaded/found.
QString scenePerformance(int scene_id);

// scene_id's recommended samples-per-pixel, or 100 (the same fallback
// cpu_interface.cpp's own accessors use) if not loaded/found.
int sceneRecommendedSpp(int scene_id);

// True if scene_id requires external asset files (e.g. earthmap.jpg);
// false if not loaded/found - erring toward "no special files needed"
// rather than surfacing a spurious warning.
bool sceneRequiresFiles(int scene_id);

} // namespace SceneMetadataClient

#endif // SCENE_METADATA_CLIENT_H
