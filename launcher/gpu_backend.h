#pragma once
// gpu_backend.h -- the launcher's one way to call "the GPU renderer", whichever this build has.
//
// A Windows build has OptiX (gpu/optix/optix_interface.h), a macOS build has Metal (gpu/metal/metal_interface.h); the two entry points take exactly
// the same arguments (the shared RenderOptions struct included), and a build never has both. main.cpp used to repeat an #ifdef RT_HAVE_METAL / #else
// block, with its own copy of the argument list and its own "not available" message, at every place it renders on the GPU. Those now call this, so
// they cannot drift apart: a new argument or a changed message is made once.
//
// A build with neither backend gets the OptiX stub (launcher/optix_stub.h), whose availability check says no - so `available()` is false and the caller
// reports "no GPU", exactly as before.

#include "../src/shared/render_options.h"

#ifdef RT_HAVE_METAL
#include "../gpu/metal/metal_interface.h"
#elif defined(RT_HAVE_OPTIX)
#include "../gpu/optix/optix_interface.h"
#else
#include "optix_stub.h"
#endif

namespace gpu_backend {

#ifdef RT_HAVE_METAL
inline const char* name() { return "Metal"; }                       // how the backend is spoken of to the user
inline const char* entryPoint() { return "metal_render_main"; }     // for log lines and error reports
inline const char* reportTag() { return "METAL"; }
inline const char* failureHint() { return "See metal_render_main()'s own stderr message above for why.\n"; }
inline bool available() {
	MetalDiagnostics d{};
	return metal_get_diagnostics(&d);
}
inline int render(int width, int height, int samples, int maxDepth, const char* outputPath, const char* sceneId, double camX, double camY, double camZ,
                  int forceCameraOverride, const RenderOptions& options) {
	return metal_render_main(width, height, samples, maxDepth, outputPath, sceneId, camX, camY, camZ, forceCameraOverride, options);
}
#else
inline const char* name() { return "OptiX"; }
inline const char* entryPoint() { return "optix_render_main"; }
inline const char* reportTag() { return "OptiX"; }
inline const char* failureHint() { return nullptr; }   // nullptr: report_render_result() prints the error code's own text
inline bool available() { return optix_is_available(); }
inline int render(int width, int height, int samples, int maxDepth, const char* outputPath, const char* sceneId, double camX, double camY, double camZ,
                  int forceCameraOverride, const RenderOptions& options) {
	return optix_render_main(width, height, samples, maxDepth, outputPath, sceneId, camX, camY, camZ, forceCameraOverride, options);
}
#endif

}  // namespace gpu_backend
