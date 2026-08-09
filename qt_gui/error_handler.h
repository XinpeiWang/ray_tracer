#ifndef QT_ERROR_HANDLER_H
#define QT_ERROR_HANDLER_H

#include <QString>
#include <QMap>
#include <QStringList>
#include "scene_metadata_client.h"

// ============================================================================
// Qt GUI Error Handler
// ============================================================================
// Maps numeric error codes to user-friendly messages and troubleshooting hints
// for display in the GUI when renders fail.
//
// This is a Qt-compatible wrapper around the C++ error_codes.h system
// ============================================================================

namespace ErrorHandler {

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
		{0, "Success"},

		// General errors (1-99)
		{1, "Unknown Error"},
		{2, "Invalid Arguments"},
		{3, "File Not Found"},
		{4, "File Read Failed"},
		{5, "File Write Failed"},
		{6, "File Copy Failed"},
		{7, "Directory Creation Failed"},
		{8, "Invalid Image Dimensions"},
		{9, "Invalid Sample Count"},
		{10, "Invalid Ray Depth"},
		{11, "Invalid Scene ID"},
		{12, "Invalid Camera Position"},
		{13, "Invalid Output Path"},
		{14, "Video Assembly Failed"},

		// CPU errors (100-199)
		{100, "Scene Build Failed (CPU)"},
		{101, "Scene is Empty (CPU)"},
		{102, "Camera Initialization Failed (CPU)"},
		{103, "Rendering Failed (CPU)"},
		{104, "Thread Error (CPU)"},
		{105, "Out of Memory (CPU)"},
		{106, "BVH Build Failed (CPU)"},
		{107, "Texture Load Failed (CPU)"},
		{108, "No Lights in Scene (CPU)"},
		{109, "Invalid Material (CPU)"},

		// GPU errors (200-299)
		{200, "No GPU Found"},
		{201, "GPU Initialization Failed"},
		{202, "GPU Out of Memory"},
		{203, "GPU Memory Copy Failed"},
		{204, "GPU Kernel Launch Failed"},
		{205, "GPU Kernel Execution Failed"},
		{206, "GPU Scene Serialization Failed"},
		{207, "GPU Synchronization Failed"},
		{208, "GPU Out of Memory"},
		{209, "Invalid GPU Configuration"},
		{210, "GPU Texture Binding Failed"},
		{211, "Scene Not Supported on GPU"},
		{212, "Scene Build Failed (GPU)"},
		{213, "Rendering Failed (GPU)"},
		{214, "Exception During Rendering (GPU)"},
		{215, "Unknown Error (GPU)"},

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
		{0, "Render completed successfully."},
		{1, "An unknown error occurred during rendering."},
		{2, "Invalid command-line arguments were provided to the renderer."},
		{3, "A required file could not be found."},
		{5, "Failed to write the output image file."},
		{8, "Image dimensions must be positive integers (recommended: 400-1920)."},
		{9, "Samples per pixel must be greater than 0 (recommended: 10-500)."},
		{10, "Maximum ray depth must be greater than 0 (recommended: 10-100)."},
		{14, "Frames rendered successfully, but assembling them into a video with ffmpeg failed."},
		{100, "Failed to construct the scene geometry."},
		{101, "The scene contains no objects to render."},
		{103, "An error occurred while rendering the image."},
		{105, "The system ran out of memory during rendering."},
		{107, "Failed to load texture file (e.g., earthmap.jpg for Earth scene)."},
		{200, "No CUDA-capable GPU was detected."},
		{202, "The GPU ran out of memory."},
		{204, "Failed to launch GPU rendering kernel."},
		{208, "GPU memory allocation failed."},
		{211, "This scene is not supported on GPU. Please use CPU mode."},
		{212, "Failed to construct the scene geometry on GPU."},
		{213, "An error occurred while rendering the image on GPU."},
		{214, "An exception was thrown while rendering on GPU."},
		{215, "An unknown error occurred while rendering on GPU."},
		{999, "The render was cancelled by the user."}
	};

	if (messages.contains(errorCode)) {
		return messages[errorCode];
	}
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
		{5, "• Check that the output directory exists and is writable\n"
			"• Make sure you have enough disk space\n"
			"• Try closing any programs that might be using the output file"},

		{14, "• Install ffmpeg from https://ffmpeg.org/download.html and add it to your PATH\n"
			 "• Check the render log above for the exact ffmpeg command and error output\n"
			 "• Rendered frames are kept in output/frames/ - you can assemble the video manually"},

		{8, "• Try common resolutions: 800×800, 1920×1080\n"
			"• Width and height must be positive numbers"},

		{9, "• For quick previews, use 10-50 samples\n"
			"• For final renders, use 100-500 samples\n"
			"• More samples = better quality but slower"},

		{100, "• Some scenes require texture files (e.g., earthmap.jpg)\n"
			 "• Make sure all required files are in the correct location"},

		{105, "• Try reducing image resolution (e.g., 800×800 instead of 1920×1080)\n"
			  "• Try reducing samples per pixel (e.g., 50 instead of 500)\n"
			  "• Close other memory-intensive applications"},

		{107, "• For the Earth scene, make sure earthmap.jpg is in the correct folder\n"
			  "• Check that texture files are not corrupted"},

		{200, "• No CUDA-capable GPU found\n"
			  "• Switch to CPU mode in the renderer settings\n"
			  "• CPU mode works on all systems"},

		{202, "• Try reducing image resolution\n"
			  "• Try reducing samples per pixel\n"
			  "• Switch to CPU mode if GPU memory is limited"},

		{208, "• GPU ran out of memory\n"
			  "• Try smaller resolution (e.g., 800×800)\n"
			  "• Try fewer samples (e.g., 50)\n"
			  "• Switch to CPU mode for large scenes"}
	};

	if (hints.contains(errorCode)) {
		return hints[errorCode];
	}
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
