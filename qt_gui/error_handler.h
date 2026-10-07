#ifndef QT_ERROR_HANDLER_H
#define QT_ERROR_HANDLER_H

#include <QCoreApplication>
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

// Total scene count, for messaging like "N scenes available". Ids are
// category letter + number now (e.g. "B10"), not contiguous ints, so there
// is no more "highest valid id" - see scene_registry.h's SceneDescriptor::id
// comment.
inline int sceneCount() {
	return SceneMetadataClient::sceneCount();
}

// "A1 (Cornell Box), C1 (Colored Quads), ..." for every GPU-supported scene.
// Both count and per-scene GPU-compatibility are queried live from
// scene_metadata.dll (see scene_metadata_client.h) rather than a
// locally-duplicated field, so this can't drift from scene_registry.h.
// Scenes are included if the query fails (unlikely - the DLL is deployed
// alongside this GUI - but erring toward listing a scene as GPU-supported
// is safer than erring toward steering users away from one that actually
// works).
//
// useMetal picks which of the two genuinely different compatibility fields
// to consult (see SceneMetadata::metalCompatible's own comment,
// scene_metadata_client.h) - the caller knows which GPU backend the failed
// render actually used, this function doesn't. Defaults to false (OptiX/
// gpuCompatible) since that was this function's only behavior before Metal
// existed.
inline QString gpuSupportedSceneList(bool useMetal = false) {
	int count = SceneMetadataClient::sceneCount();
	QStringList parts;
	for (int i = 0; i < count; ++i) {
		QString id = SceneMetadataClient::sceneIdAtIndex(i);
		bool supported = true;
		if (useMetal)
			SceneMetadataClient::metalCompatible(id, supported);
		else
			SceneMetadataClient::gpuCompatible(id, supported);
		if (supported)
			parts << QString("%1 (%2)").arg(id).arg(SceneMetadataClient::sceneName(id));
	}
	return parts.join(", ");
}

// Get user-friendly error title
inline QString getErrorTitle(int errorCode) {
	static const QMap<int, const char *> titles = {
		// Success
		{SUCCESS, QT_TRANSLATE_NOOP("ErrorHandler", "Success")},

		// General errors (1-99)
		{ERR_UNKNOWN, QT_TRANSLATE_NOOP("ErrorHandler", "Unknown Error")},
		{ERR_INVALID_ARGUMENTS, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Arguments")},
		{ERR_FILE_NOT_FOUND, QT_TRANSLATE_NOOP("ErrorHandler", "File Not Found")},
		{ERR_FILE_READ_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "File Read Failed")},
		{ERR_FILE_WRITE_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "File Write Failed")},
		// ERR_FILE_COPY_FAILED: kept for numeric-code stability (a script
		// checking the CLI's exit code shouldn't have code 6 silently
		// change meaning), but no code path returns it anymore - the
		// Desktop-write-then-copy step that used to produce it was removed
		// (see docs/ERROR_CODE_REFERENCE.md's own entry for the full story;
		// a total CPU write failure now returns ERR_FILE_WRITE_FAILED
		// above instead).
		{ERR_FILE_COPY_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "File Copy Failed")},
		{ERR_DIRECTORY_CREATE_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Directory Creation Failed")},
		{ERR_INVALID_DIMENSIONS, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Image Dimensions")},
		{ERR_INVALID_SAMPLE_COUNT, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Sample Count")},
		{ERR_INVALID_MAX_DEPTH, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Ray Depth")},
		{ERR_INVALID_SCENE_ID, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Scene ID")},
		{ERR_INVALID_CAMERA_POSITION, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Camera Position")},
		{ERR_OUTPUT_PATH_INVALID, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Output Path")},
		{ERR_VIDEO_ASSEMBLY_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Video Assembly Failed")},

		// CPU errors (100-199)
		{ERR_CPU_SCENE_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Scene Build Failed (CPU)")},
		{ERR_CPU_SCENE_EMPTY, QT_TRANSLATE_NOOP("ErrorHandler", "Scene is Empty (CPU)")},
		{ERR_CPU_CAMERA_INIT_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Camera Initialization Failed (CPU)")},
		{ERR_CPU_RENDER_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Rendering Failed (CPU)")},
		{ERR_CPU_THREAD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Thread Error (CPU)")},
		{ERR_CPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "Out of Memory (CPU)")},
		{ERR_CPU_BVH_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "BVH Build Failed (CPU)")},
		{ERR_CPU_TEXTURE_LOAD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Texture Load Failed (CPU)")},
		{ERR_CPU_LIGHTS_EMPTY, QT_TRANSLATE_NOOP("ErrorHandler", "No Lights in Scene (CPU)")},
		{ERR_CPU_MATERIAL_INVALID, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid Material (CPU)")},

		// GPU errors (200-299)
		{ERR_GPU_NO_DEVICE, QT_TRANSLATE_NOOP("ErrorHandler", "No GPU Found")},
		{ERR_GPU_DEVICE_INIT_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Initialization Failed")},
		{ERR_GPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Out of Memory")},
		{ERR_GPU_MEMORY_COPY_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Memory Copy Failed")},
		{ERR_GPU_KERNEL_LAUNCH_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Kernel Launch Failed")},
		{ERR_GPU_KERNEL_EXECUTION_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Kernel Execution Failed")},
		{ERR_GPU_SCENE_SERIALIZATION_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Scene Serialization Failed")},
		{ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Synchronization Failed")},
		{ERR_GPU_OUT_OF_MEMORY, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Out of Memory")},
		{ERR_GPU_INVALID_CONFIGURATION, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid GPU Configuration")},
		{ERR_GPU_TEXTURE_BINDING_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "GPU Texture Binding Failed")},
		{ERR_GPU_UNSUPPORTED_SCENE, QT_TRANSLATE_NOOP("ErrorHandler", "Scene Not Supported on GPU")},
		{ERR_GPU_SCENE_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Scene Build Failed (GPU)")},
		{ERR_GPU_RENDER_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Rendering Failed (GPU)")},
		{ERR_GPU_EXCEPTION, QT_TRANSLATE_NOOP("ErrorHandler", "Exception During Rendering (GPU)")},
		{ERR_GPU_UNKNOWN_ERROR, QT_TRANSLATE_NOOP("ErrorHandler", "Unknown Error (GPU)")},

		// User action
		{999, QT_TRANSLATE_NOOP("ErrorHandler", "Cancelled by User")}
	};

	if (titles.contains(errorCode)) {
		return QCoreApplication::translate("ErrorHandler", titles[errorCode]);
	}
	return QCoreApplication::translate("ErrorHandler", "Error Code %1").arg(errorCode);
}

// Get detailed error message. useMetal: same meaning as getTroubleshootingHint()'s
// own parameter below - which GPU backend the failed render actually used.
// Unlike that function, this one DOES need it beyond just errorCode 211:
// ERR_GPU_NO_DEVICE's own text names a specific GPU vendor/API, which is
// backend-dependent in a way none of this map's other codes are.
inline QString getErrorMessage(int errorCode, bool useMetal = false) {
	// Scene count grows over time, so this one is built from the live scene
	// table instead of living in the static map below.
	if (errorCode == 11)
		return QCoreApplication::translate("ErrorHandler", "Scene ID must be a valid scene identifier (e.g. \"A1\"). Check the scene selector.");
	if (errorCode == ERR_GPU_NO_DEVICE) {
		return useMetal ? QCoreApplication::translate("ErrorHandler", "No usable Metal GPU was found on this Mac.")
			: QCoreApplication::translate("ErrorHandler", "No CUDA-capable GPU was detected.");
	}

	static const QMap<int, const char *> messages = {
		{SUCCESS, QT_TRANSLATE_NOOP("ErrorHandler", "Render completed successfully.")},
		{ERR_UNKNOWN, QT_TRANSLATE_NOOP("ErrorHandler", "An unknown error occurred during rendering.")},
		{ERR_INVALID_ARGUMENTS, QT_TRANSLATE_NOOP("ErrorHandler", "Invalid command-line arguments were provided to the renderer.")},
		{ERR_FILE_NOT_FOUND, QT_TRANSLATE_NOOP("ErrorHandler", "A required file could not be found.")},
		{ERR_FILE_WRITE_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Failed to write the output image file.")},
		{ERR_INVALID_DIMENSIONS, QT_TRANSLATE_NOOP("ErrorHandler", "Image dimensions must be positive integers (recommended: 400-1920).")},
		{ERR_INVALID_SAMPLE_COUNT, QT_TRANSLATE_NOOP("ErrorHandler", "Samples per pixel must be greater than 0 (recommended: 10-500).")},
		{ERR_INVALID_MAX_DEPTH, QT_TRANSLATE_NOOP("ErrorHandler", "Maximum ray depth must be greater than 0 (recommended: 10-100).")},
		{ERR_VIDEO_ASSEMBLY_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Frames rendered successfully, but assembling them into a video with ffmpeg failed.")},
		{ERR_CPU_SCENE_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Failed to construct the scene geometry.")},
		{ERR_CPU_SCENE_EMPTY, QT_TRANSLATE_NOOP("ErrorHandler", "The scene contains no objects to render.")},
		{ERR_CPU_RENDER_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "An error occurred while rendering the image.")},
		{ERR_CPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "The system ran out of memory during rendering.")},
		{ERR_CPU_TEXTURE_LOAD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Failed to load texture file (e.g., earthmap.jpg for Earth scene).")},
		// ERR_GPU_NO_DEVICE is NOT here - handled above, before this map,
		// since its text is backend-dependent (see this function's own
		// header comment).
		{ERR_GPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "The GPU ran out of memory.")},
		{ERR_GPU_KERNEL_LAUNCH_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Failed to launch GPU rendering kernel.")},
		{ERR_GPU_OUT_OF_MEMORY, QT_TRANSLATE_NOOP("ErrorHandler", "GPU memory allocation failed.")},
		{ERR_GPU_UNSUPPORTED_SCENE, QT_TRANSLATE_NOOP("ErrorHandler", "This scene is not supported on GPU. Please use CPU mode.")},
		{ERR_GPU_SCENE_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "Failed to construct the scene geometry on GPU.")},
		{ERR_GPU_RENDER_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "An error occurred while rendering the image on GPU.")},
		{ERR_GPU_EXCEPTION, QT_TRANSLATE_NOOP("ErrorHandler", "An exception was thrown while rendering on GPU.")},
		{ERR_GPU_UNKNOWN_ERROR, QT_TRANSLATE_NOOP("ErrorHandler", "An unknown error occurred while rendering on GPU.")},
		{999, QT_TRANSLATE_NOOP("ErrorHandler", "The render was cancelled by the user.")}
	};

	if (messages.contains(errorCode)) {
		return QCoreApplication::translate("ErrorHandler", messages[errorCode]);
	}
	// Not overridden here - use the canonical text rather than inventing a
	// generic string, so a code added to error_codes.h is described properly
	// without anyone having to remember to edit this file too. The has_*
	// predicate matters: get_error_message() returns a placeholder rather
	// than an empty string for unknown codes, so testing the return value
	// would always "succeed" and mask this file's own wording.
	if (::has_error_message(errorCode))
		return fromStd(::get_error_message(errorCode));
	return QCoreApplication::translate("ErrorHandler", "An error occurred with code %1.").arg(errorCode);
}

// The fallback hint when a code has neither an override nor a canonical one.
// Callers compare against this to avoid printing the same generic line twice,
// so they must call here rather than restate the English text: once it is
// translated, a literal comparison would silently stop matching.
inline QString genericHint() {
	return QCoreApplication::translate("ErrorHandler", "Check the Log Output tab for detailed error information.");
}

// Get troubleshooting hint. useMetal: which GPU backend the failed render
// actually used - affects errorCode 211's scene list (see
// gpuSupportedSceneList()'s own comment) and ERR_GPU_NO_DEVICE's wording
// below, both genuinely backend-dependent; every other code's text is
// backend-agnostic.
inline QString getTroubleshootingHint(int errorCode, bool useMetal = false) {
	// Both of these depend on the live scene table (total count / which
	// scenes are GPU-supported), so they're built here instead of hardcoded
	// in the static map below.
	if (errorCode == 11) {
		return QCoreApplication::translate("ErrorHandler", "• Check the Scene dropdown in the Settings tab for a valid scene id\n"
			"• Use CPU renderer for scenes that are not GPU-supported");
	}
	if (errorCode == 211) {
		return QCoreApplication::translate("ErrorHandler", "• GPU supports scenes: %1\n"
			"• Switch to CPU mode for all other scenes\n"
			"• CPU mode supports all %2 scenes")
			.arg(gpuSupportedSceneList(useMetal)).arg(sceneCount());
	}
	if (errorCode == ERR_GPU_NO_DEVICE) {
		return useMetal
			? QCoreApplication::translate("ErrorHandler", "• No usable Metal GPU found on this Mac\n"
				"• Switch to CPU mode in the renderer settings\n"
				"• CPU mode works on all systems")
			: QCoreApplication::translate("ErrorHandler", "• No CUDA-capable GPU found\n"
				"• Switch to CPU mode in the renderer settings\n"
				"• CPU mode works on all systems");
	}

	if (errorCode == ERR_FILE_WRITE_FAILED) {
		QString hint = QCoreApplication::translate("ErrorHandler", "• Check that the output directory exists and is writable\n"
			"• Make sure you have enough disk space\n"
			"• Try closing any programs that might be using the output file");
#ifdef Q_OS_WIN
		// The ACLs on a protected folder still say "writable", so nothing in
		// the generic list above points at this - it was found via Defender's
		// event log, not from this dialog, when every default GUI render into
		// Pictures failed with only the three bullets above to go on.
		hint += QCoreApplication::translate("ErrorHandler", "\n• Windows Security's \"Controlled folder access\" blocks apps it doesn't "
			"recognize from writing to protected folders (Pictures, Documents, Desktop). "
			"Either choose a different Output Path, or allow RayTracerGUI.exe and ray_tracer.exe "
			"under Virus & threat protection > Ransomware protection > Controlled folder access "
			"> Allow an app through");
#endif
		return hint;
	}

	static const QMap<int, const char *> hints = {
		{ERR_VIDEO_ASSEMBLY_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "• Install ffmpeg from https://ffmpeg.org/download.html and add it to your PATH\n"
			 "• Check the render log above for the exact ffmpeg command and error output\n"
			 "• Rendered frames are kept in output/frames/ - you can assemble the video manually")},

		{ERR_INVALID_DIMENSIONS, QT_TRANSLATE_NOOP("ErrorHandler", "• Try common resolutions: 800×800, 1920×1080\n"
			"• Width and height must be positive numbers")},

		{ERR_INVALID_SAMPLE_COUNT, QT_TRANSLATE_NOOP("ErrorHandler", "• For quick previews, use 10-50 samples\n"
			"• For final renders, use 100-500 samples\n"
			"• More samples = better quality but slower")},

		{ERR_CPU_SCENE_BUILD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "• Some scenes require texture files (e.g., earthmap.jpg)\n"
			 "• Make sure all required files are in the correct location")},

		{ERR_CPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "• Try reducing image resolution (e.g., 800×800 instead of 1920×1080)\n"
			  "• Try reducing samples per pixel (e.g., 50 instead of 500)\n"
			  "• Close other memory-intensive applications")},

		{ERR_CPU_TEXTURE_LOAD_FAILED, QT_TRANSLATE_NOOP("ErrorHandler", "• For the Earth scene, make sure earthmap.jpg is in the correct folder\n"
			  "• Check that texture files are not corrupted")},

		// ERR_GPU_NO_DEVICE is NOT here - handled above, before this map,
		// since its text is backend-dependent (see this function's own
		// header comment).

		{ERR_GPU_MEMORY_ALLOCATION, QT_TRANSLATE_NOOP("ErrorHandler", "• Try reducing image resolution\n"
			  "• Try reducing samples per pixel\n"
			  "• Switch to CPU mode if GPU memory is limited")},

		{ERR_GPU_OUT_OF_MEMORY, QT_TRANSLATE_NOOP("ErrorHandler", "• GPU ran out of memory\n"
			  "• Try smaller resolution (e.g., 800×800)\n"
			  "• Try fewer samples (e.g., 50)\n"
			  "• Switch to CPU mode for large scenes")}
	};

	if (hints.contains(errorCode)) {
		return QCoreApplication::translate("ErrorHandler", hints[errorCode]);
	}
	// Same idea as getErrorMessage(), including why the has_* predicate is
	// needed rather than an emptiness check.
	if (::has_troubleshooting_hint(errorCode))
		return fromStd(::get_troubleshooting_hint(errorCode));
	return genericHint();
}

// Get error category name
inline QString getCategoryName(int errorCode) {
	if (errorCode == 0) return QCoreApplication::translate("ErrorHandler", "Success");
	if (errorCode >= 1 && errorCode <= 99) return QCoreApplication::translate("ErrorHandler", "General");
	if (errorCode >= 100 && errorCode <= 199) return QCoreApplication::translate("ErrorHandler", "CPU Renderer");
	if (errorCode >= 200 && errorCode <= 299) return QCoreApplication::translate("ErrorHandler", "GPU Renderer");
	if (errorCode == 999) return QCoreApplication::translate("ErrorHandler", "User Action");
	return QCoreApplication::translate("ErrorHandler", "Unknown");
}

} // namespace ErrorHandler

#endif // QT_ERROR_HANDLER_H
