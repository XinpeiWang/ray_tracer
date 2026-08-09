#ifndef SCENE_METADATA_CLIENT_H
#define SCENE_METADATA_CLIENT_H

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

} // namespace SceneMetadataClient

#endif // SCENE_METADATA_CLIENT_H
