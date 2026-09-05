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
// could be queried; false (out_compatible left untouched) otherwise. Kept
// as its own accessor (unlike recommendedCamera/scenePerformance/etc.,
// folded into SceneMetadata below once their last individual caller was
// converted to it) because error_handler.h's build-a-warning-list use is a
// genuine single-field need, not part of a "several fields, one scene"
// cluster.
bool gpuCompatible(const QString& scene_id, bool& out_compatible);

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
// existing state stand" contract as gpuCompatible above.

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

// True if scene_id requires external asset files (e.g. earthmap.jpg);
// false if not loaded/found - erring toward "no special files needed"
// rather than surfacing a spurious warning. Kept standalone: every
// remaining caller (mainwindow_tabs.cpp's category/search filtering,
// mainwindow_slots.cpp's thumbnail-generation loop) checks this ALONE for
// many scenes in a loop, not alongside other fields for one scene, so
// folding it into SceneMetadata would add a full bulk fetch's cost to a
// tight loop for no benefit.
bool sceneRequiresFiles(const QString& scene_id);

// scenePerformance/sceneRecommendedSpp/sceneRecommendedExposure/
// recommendedCamera/sceneRecommendedIntegrator/sceneRecommendedSampler/
// sceneRecommendedLightSampler used to live here as their own individual
// accessors, mirroring cpu_interface.h's by-id functions of the same
// names/shapes one-for-one. Removed once SceneMetadata (below) covered
// every one of their fields AND their own last individual caller was
// converted to fetch a SceneMetadata once instead (onSceneChanged(),
// applyRecommendedSettings(), captureRenderJob(), updateSceneRecommended-
// SettingsHint() - all in mainwindow_slots.cpp) - keeping them as an
// unused parallel API alongside SceneMetadata would have been exactly the
// "two APIs to keep in sync, only one of which anything still calls"
// problem a design review flagged this file for. The underlying
// cpu_interface.h/.cpp accessors and scene_metadata_dll.cpp exports these
// used to wrap are UNTOUCHED - they're still real, tested, general-purpose
// C API surface (scene_registry_tests.cpp exercises them directly), this
// removal is scoped to this GUI-side client's own now-dead wrappers only.
// If a future single-field-only need for one of them comes back, add a
// thin one-line wrapper here again rather than reintroducing the whole
// Fn-pointer/DllHandle/lookupSymbol/required-check machinery for it.

// Qt-friendly mirror of cpu_interface.h's SceneMetadataSnapshot (the raw C
// struct scene_metadata_snapshot() fills in across the DLL boundary) - one
// call's worth of everything a scene-selection UI refresh needs, instead of
// calling several of the single-field functions above in a row. Every
// string/bool/int/double field here defaults to the same "not found"
// fallback its single-field equivalent above uses, so a caller that gets
// `false` back can still use a default-constructed SceneMetadata safely
// (e.g. treating its empty recommendedIntegrator as "no recommendation",
// the same "empty means none declared" meaning every recommended_* field
// on SceneDescriptor itself already carries - scene_registry.h).
struct SceneMetadata {
	QString name;
	QString category;
	QString description;
	QString performance;
	int recommendedSpp = 100;
	bool requiresFiles = false;
	bool gpuCompatible = true;
	double recommendedExposure = 1.0;
	QString recommendedIntegrator;
	QString recommendedSampler;
	QString recommendedLightSampler;
	double camLookfromX = 0.0, camLookfromY = 0.0, camLookfromZ = 0.0;
	double camLookatX = 0.0, camLookatY = 0.0, camLookatZ = 0.0;
};

// Fills `out` with everything SceneMetadata holds for scene_id in one DLL
// call and one registry lookup, instead of the several separate calls (each
// its own lookup) the individual accessors above would take together. See
// SceneMetadata's own comment for the "not found" fallback shape.
// @return true on success (out fully populated), false if not loaded/found
//         (out left at whatever it already held - same "don't crash, don't
//         silently reset" contract every other query in this file follows)
bool sceneMetadata(const QString& scene_id, SceneMetadata& out);

} // namespace SceneMetadataClient

#endif // SCENE_METADATA_CLIENT_H
