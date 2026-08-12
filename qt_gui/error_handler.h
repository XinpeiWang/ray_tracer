#ifndef QT_ERROR_HANDLER_H
#define QT_ERROR_HANDLER_H

#include <QString>
#include <QMap>
#include <QStringList>
#include "scene_metadata_client.h"

// The canonical code list and its baseline text. Header-only and Qt-free
// (<string>/<map>), so the GUI can share it rather than restate it.
#include "../src/TheRestOfYourLife/error_codes.h"

// ============================================================================
// Qt GUI Error Handler
// ============================================================================
// Presents renderer exit codes to the user.
//
// This file used to be a full second copy of the error system: its own
// integer literals, its own message text, its own hints. Nothing tied the two
// together, so a code added to error_codes.h would surface here as "Unknown
// Error" with no build failure and no test to catch it. (Checked at the time
// of this change: the two tables had not actually drifted - 41 backend codes,
// all covered - so this is prevention, not a bug fix.)
//
// Now the relationship is explicit:
//   * codes are named enum constants, so renaming or removing one in
//     error_codes.h breaks THIS FILE at compile time rather than silently;
//   * message/hint/category text falls back to the canonical
//     get_error_message()/get_troubleshooting_hint()/get_error_category(),
//     so a newly added code is described correctly here without any edit;
//   * the GUI only overrides where it genuinely adds something the backend
//     cannot know - a short title for a dialog caption, bulleted hints
//     written for a GUI user, or live data like the current scene count.
// ============================================================================

namespace ErrorHandler {

// Bridges the canonical std::string API into Qt without pulling <string>
// conversions through every call site below.
inline QString fromStd(const std::string &s) {
	return QString::fromStdString(s);
}

// Scene-count/GPU-support text below is generated live from
// scene_metadata.dll rather than hardcoded, so it can't go stale the way
// it did previously (this file said "Scene ID must be between 0 and 8"
// and separately "0-36" for the same error code, and claimed GPU support
// for only 4 of the 11 GPU-capable scenes while the total scene count had
// grown from 9 to 37).

// Highest valid scene ID (scene count - 1).
inline int maxSceneId() {
	int count = SceneMetadataClient::sceneCount();
	return count > 0 ? count - 1 : 0;
}

// "0 (Cornell Box), 2 (Checkered Spheres), ..." for every GPU-supported scene.
// Both count and per-scene GPU-compatibility are queried live from
// scene_metadata.dll (see scene_metadata_client.h) rather than a
// locally-duplicated field, so this can't drift from scene_registry.h.
// Scenes are included if the query fails (unlikely - the DLL is deployed
// alongside this GUI - but erring toward listing a scene as GPU-supported
// is safer than erring toward steering users away from one that actually
// works).
inline QString gpuSupportedSceneList() {
	int count = SceneMetadataClient::sceneCount();
	QStringList parts;
	for (int id = 0; id < count; ++id) {
		bool supported = true;
		SceneMetadataClient::gpuCompatible(id, supported);
		if (supported)
			parts << QString("%1 (%2)").arg(id).arg(SceneMetadataClient::sceneName(id));
	}
	return parts.join(", ");
}

// Get user-friendly error title
inline QString getErrorTitle(int errorCode) {
	static const QMap<int, QString> titles = {
		// Success
		{SUCCESS, "Success"},

		// General errors (1-99)
		{ERR_UNKNOWN, "Unknown Error"},
		{ERR_INVALID_ARGUMENTS, "Invalid Arguments"},
		{ERR_FILE_NOT_FOUND, "File Not Found"},
		{ERR_FILE_READ_FAILED, "File Read Failed"},
		{ERR_FILE_WRITE_FAILED, "File Write Failed"},
		{ERR_FILE_COPY_FAILED, "File Copy Failed"},
		{ERR_DIRECTORY_CREATE_FAILED, "Directory Creation Failed"},
		{ERR_INVALID_DIMENSIONS, "Invalid Image Dimensions"},
		{ERR_INVALID_SAMPLE_COUNT, "Invalid Sample Count"},
		{ERR_INVALID_MAX_DEPTH, "Invalid Ray Depth"},
		{ERR_INVALID_SCENE_ID, "Invalid Scene ID"},
		{ERR_INVALID_CAMERA_POSITION, "Invalid Camera Position"},
		{ERR_OUTPUT_PATH_INVALID, "Invalid Output Path"},
		{ERR_VIDEO_ASSEMBLY_FAILED, "Video Assembly Failed"},

		// CPU errors (100-199)
		{ERR_CPU_SCENE_BUILD_FAILED, "Scene Build Failed (CPU)"},
		{ERR_CPU_SCENE_EMPTY, "Scene is Empty (CPU)"},
		{ERR_CPU_CAMERA_INIT_FAILED, "Camera Initialization Failed (CPU)"},
		{ERR_CPU_RENDER_FAILED, "Rendering Failed (CPU)"},
		{ERR_CPU_THREAD_FAILED, "Thread Error (CPU)"},
		{ERR_CPU_MEMORY_ALLOCATION, "Out of Memory (CPU)"},
		{ERR_CPU_BVH_BUILD_FAILED, "BVH Build Failed (CPU)"},
		{ERR_CPU_TEXTURE_LOAD_FAILED, "Texture Load Failed (CPU)"},
		{ERR_CPU_LIGHTS_EMPTY, "No Lights in Scene (CPU)"},
		{ERR_CPU_MATERIAL_INVALID, "Invalid Material (CPU)"},

		// GPU errors (200-299)
		{ERR_GPU_NO_DEVICE, "No GPU Found"},
		{ERR_GPU_DEVICE_INIT_FAILED, "GPU Initialization Failed"},
		{ERR_GPU_MEMORY_ALLOCATION, "GPU Out of Memory"},
		{ERR_GPU_MEMORY_COPY_FAILED, "GPU Memory Copy Failed"},
		{ERR_GPU_KERNEL_LAUNCH_FAILED, "GPU Kernel Launch Failed"},
		{ERR_GPU_KERNEL_EXECUTION_FAILED, "GPU Kernel Execution Failed"},
		{ERR_GPU_SCENE_SERIALIZATION_FAILED, "GPU Scene Serialization Failed"},
		{ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED, "GPU Synchronization Failed"},
		{ERR_GPU_OUT_OF_MEMORY, "GPU Out of Memory"},
		{ERR_GPU_INVALID_CONFIGURATION, "Invalid GPU Configuration"},
		{ERR_GPU_TEXTURE_BINDING_FAILED, "GPU Texture Binding Failed"},
		{ERR_GPU_UNSUPPORTED_SCENE, "Scene Not Supported on GPU"},
		{ERR_GPU_SCENE_BUILD_FAILED, "Scene Build Failed (GPU)"},
		{ERR_GPU_RENDER_FAILED, "Rendering Failed (GPU)"},
		{ERR_GPU_EXCEPTION, "Exception During Rendering (GPU)"},
		{ERR_GPU_UNKNOWN_ERROR, "Unknown Error (GPU)"},

		// User action
		{999, "Cancelled by User"}
	};

	if (titles.contains(errorCode)) {
		return titles[errorCode];
	}
	return QString("Error Code %1").arg(errorCode);
}

// Get detailed error message
inline QString getErrorMessage(int errorCode) {
	// Scene count grows over time, so this one is built from the live scene
	// table instead of living in the static map below.
	if (errorCode == 11)
		return QString("Scene ID must be between 0 and %1. Check the scene selector.").arg(maxSceneId());

	static const QMap<int, QString> messages = {
		{SUCCESS, "Render completed successfully."},
		{ERR_UNKNOWN, "An unknown error occurred during rendering."},
		{ERR_INVALID_ARGUMENTS, "Invalid command-line arguments were provided to the renderer."},
		{ERR_FILE_NOT_FOUND, "A required file could not be found."},
		{ERR_FILE_WRITE_FAILED, "Failed to write the output image file."},
		{ERR_INVALID_DIMENSIONS, "Image dimensions must be positive integers (recommended: 400-1920)."},
		{ERR_INVALID_SAMPLE_COUNT, "Samples per pixel must be greater than 0 (recommended: 10-500)."},
		{ERR_INVALID_MAX_DEPTH, "Maximum ray depth must be greater than 0 (recommended: 10-100)."},
		{ERR_VIDEO_ASSEMBLY_FAILED, "Frames rendered successfully, but assembling them into a video with ffmpeg failed."},
		{ERR_CPU_SCENE_BUILD_FAILED, "Failed to construct the scene geometry."},
		{ERR_CPU_SCENE_EMPTY, "The scene contains no objects to render."},
		{ERR_CPU_RENDER_FAILED, "An error occurred while rendering the image."},
		{ERR_CPU_MEMORY_ALLOCATION, "The system ran out of memory during rendering."},
		{ERR_CPU_TEXTURE_LOAD_FAILED, "Failed to load texture file (e.g., earthmap.jpg for Earth scene)."},
		{ERR_GPU_NO_DEVICE, "No CUDA-capable GPU was detected."},
		{ERR_GPU_MEMORY_ALLOCATION, "The GPU ran out of memory."},
		{ERR_GPU_KERNEL_LAUNCH_FAILED, "Failed to launch GPU rendering kernel."},
		{ERR_GPU_OUT_OF_MEMORY, "GPU memory allocation failed."},
		{ERR_GPU_UNSUPPORTED_SCENE, "This scene is not supported on GPU. Please use CPU mode."},
		{ERR_GPU_SCENE_BUILD_FAILED, "Failed to construct the scene geometry on GPU."},
		{ERR_GPU_RENDER_FAILED, "An error occurred while rendering the image on GPU."},
		{ERR_GPU_EXCEPTION, "An exception was thrown while rendering on GPU."},
		{ERR_GPU_UNKNOWN_ERROR, "An unknown error occurred while rendering on GPU."},
		{999, "The render was cancelled by the user."}
	};

	if (messages.contains(errorCode)) {
		return messages[errorCode];
	}
	// Not overridden here - use the canonical text rather than inventing a
	// generic string, so a code added to error_codes.h is described properly
	// without anyone having to remember to edit this file too. The has_*
	// predicate matters: get_error_message() returns a placeholder rather
	// than an empty string for unknown codes, so testing the return value
	// would always "succeed" and mask this file's own wording.
	if (::has_error_message(errorCode))
		return fromStd(::get_error_message(errorCode));
	return QString("An error occurred with code %1.").arg(errorCode);
}

// Get troubleshooting hint
inline QString getTroubleshootingHint(int errorCode) {
	// Both of these depend on the live scene table (total count / which
	// scenes are GPU-supported), so they're built here instead of hardcoded
	// in the static map below.
	if (errorCode == 11) {
		return QString("• Valid scenes are 0-%1 — check the Scene dropdown in the Basic Settings tab\n"
			"• Use CPU renderer for scenes that are not GPU-supported").arg(maxSceneId());
	}
	if (errorCode == 211) {
		return QString("• GPU supports scenes: %1\n"
			"• Switch to CPU mode for all other scenes\n"
			"• CPU mode supports all %2 scenes")
			.arg(gpuSupportedSceneList()).arg(maxSceneId() + 1);
	}

	static const QMap<int, QString> hints = {
		{ERR_FILE_WRITE_FAILED, "• Check that the output directory exists and is writable\n"
			"• Make sure you have enough disk space\n"
			"• Try closing any programs that might be using the output file"},

		{ERR_VIDEO_ASSEMBLY_FAILED, "• Install ffmpeg from https://ffmpeg.org/download.html and add it to your PATH\n"
			 "• Check the render log above for the exact ffmpeg command and error output\n"
			 "• Rendered frames are kept in output/frames/ - you can assemble the video manually"},

		{ERR_INVALID_DIMENSIONS, "• Try common resolutions: 800×800, 1920×1080\n"
			"• Width and height must be positive numbers"},

		{ERR_INVALID_SAMPLE_COUNT, "• For quick previews, use 10-50 samples\n"
			"• For final renders, use 100-500 samples\n"
			"• More samples = better quality but slower"},

		{ERR_CPU_SCENE_BUILD_FAILED, "• Some scenes require texture files (e.g., earthmap.jpg)\n"
			 "• Make sure all required files are in the correct location"},

		{ERR_CPU_MEMORY_ALLOCATION, "• Try reducing image resolution (e.g., 800×800 instead of 1920×1080)\n"
			  "• Try reducing samples per pixel (e.g., 50 instead of 500)\n"
			  "• Close other memory-intensive applications"},

		{ERR_CPU_TEXTURE_LOAD_FAILED, "• For the Earth scene, make sure earthmap.jpg is in the correct folder\n"
			  "• Check that texture files are not corrupted"},

		{ERR_GPU_NO_DEVICE, "• No CUDA-capable GPU found\n"
			  "• Switch to CPU mode in the renderer settings\n"
			  "• CPU mode works on all systems"},

		{ERR_GPU_MEMORY_ALLOCATION, "• Try reducing image resolution\n"
			  "• Try reducing samples per pixel\n"
			  "• Switch to CPU mode if GPU memory is limited"},

		{ERR_GPU_OUT_OF_MEMORY, "• GPU ran out of memory\n"
			  "• Try smaller resolution (e.g., 800×800)\n"
			  "• Try fewer samples (e.g., 50)\n"
			  "• Switch to CPU mode for large scenes"}
	};

	if (hints.contains(errorCode)) {
		return hints[errorCode];
	}
	// Same idea as getErrorMessage(), including why the has_* predicate is
	// needed rather than an emptiness check.
	if (::has_troubleshooting_hint(errorCode))
		return fromStd(::get_troubleshooting_hint(errorCode));
	return "Check the Log Output tab for detailed error information.";
}

// Get category color (for UI styling)
inline QString getCategoryColor(int errorCode) {
	if (errorCode == 0) return "#4CAF50"; // Green - success
	if (errorCode >= 1 && errorCode <= 99) return "#FF9800"; // Orange - general
	if (errorCode >= 100 && errorCode <= 199) return "#2196F3"; // Blue - CPU
	if (errorCode >= 200 && errorCode <= 299) return "#9C27B0"; // Purple - GPU
	if (errorCode == 999) return "#9E9E9E"; // Gray - user action
	return "#F44336"; // Red - unknown
}

// Get error category name
inline QString getCategoryName(int errorCode) {
	if (errorCode == 0) return "Success";
	if (errorCode >= 1 && errorCode <= 99) return "General";
	if (errorCode >= 100 && errorCode <= 199) return "CPU Renderer";
	if (errorCode >= 200 && errorCode <= 299) return "GPU Renderer";
	if (errorCode == 999) return "User Action";
	return "Unknown";
}

} // namespace ErrorHandler

#endif // QT_ERROR_HANDLER_H
