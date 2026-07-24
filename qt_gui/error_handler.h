#ifndef QT_ERROR_HANDLER_H
#define QT_ERROR_HANDLER_H

#include <QString>
#include <QMap>

// ============================================================================
// Qt GUI Error Handler
// ============================================================================
// Maps numeric error codes to user-friendly messages and troubleshooting hints
// for display in the GUI when renders fail.
// 
// This is a Qt-compatible wrapper around the C++ error_codes.h system
// ============================================================================

namespace ErrorHandler {

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
	static const QMap<int, QString> messages = {
		{0, "Render completed successfully."},
		{1, "An unknown error occurred during rendering."},
		{2, "Invalid command-line arguments were provided to the renderer."},
		{3, "A required file could not be found."},
		{5, "Failed to write the output image file."},
		{8, "Image dimensions must be positive integers (recommended: 400-1920)."},
		{9, "Samples per pixel must be greater than 0 (recommended: 10-500)."},
		{10, "Maximum ray depth must be greater than 0 (recommended: 10-100)."},
		{11, "Scene ID must be between 0 and 8. Check the scene selector."},
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
		{999, "The render was cancelled by the user."}
	};

	if (messages.contains(errorCode)) {
		return messages[errorCode];
	}
	return QString("An error occurred with code %1.").arg(errorCode);
}

// Get troubleshooting hint
inline QString getTroubleshootingHint(int errorCode) {
	static const QMap<int, QString> hints = {
		{5, "• Check that the output directory exists and is writable\n"
			"• Make sure you have enough disk space\n"
			"• Try closing any programs that might be using the output file"},

		{8, "• Try common resolutions: 800×800, 1920×1080\n"
			"• Width and height must be positive numbers"},

		{9, "• For quick previews, use 10-50 samples\n"
			"• For final renders, use 100-500 samples\n"
			"• More samples = better quality but slower"},

		{11, "• Valid scenes are 0-10 (Cornell Box through Cornell Rough Metal)\n"
			"• Check the Basic Settings tab for scene selection"},

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
			  "• Switch to CPU mode for large scenes"},

		{211, "• GPU supports scenes: 0 (Cornell Box), 2 (Checkered Spheres), 5 (Colored Quads), 10 (Cornell Rough Metal)\n"
			  "• Switch to CPU mode for all other scenes\n"
			  "• CPU mode supports all 11 scenes"}
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
