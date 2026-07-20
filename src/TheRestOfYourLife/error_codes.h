#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#include <string>
#include <map>

// ============================================================================
// RAY TRACER ERROR CODE SYSTEM
// ============================================================================
// Centralized error codes for easy debugging across GUI/launcher/CPU/GPU
//
// Error Code Ranges:
//   0          - Success
//   1-99       - General errors (file I/O, invalid parameters)
//   100-199    - CPU renderer errors (scene building, rendering)
//   200-299    - GPU renderer errors (CUDA, memory, kernel)
//   300-399    - Reserved for future use
// ============================================================================

enum RenderErrorCode {
	// ===== SUCCESS =====
	SUCCESS = 0,

	// ===== GENERAL ERRORS (1-99) =====
	ERR_UNKNOWN = 1,
	ERR_INVALID_ARGUMENTS = 2,
	ERR_FILE_NOT_FOUND = 3,
	ERR_FILE_READ_FAILED = 4,
	ERR_FILE_WRITE_FAILED = 5,
	ERR_FILE_COPY_FAILED = 6,
	ERR_DIRECTORY_CREATE_FAILED = 7,
	ERR_INVALID_DIMENSIONS = 8,
	ERR_INVALID_SAMPLE_COUNT = 9,
	ERR_INVALID_MAX_DEPTH = 10,
	ERR_INVALID_SCENE_ID = 11,
	ERR_INVALID_CAMERA_POSITION = 12,
	ERR_OUTPUT_PATH_INVALID = 13,

	// ===== CPU RENDERER ERRORS (100-199) =====
	ERR_CPU_SCENE_BUILD_FAILED = 100,
	ERR_CPU_SCENE_EMPTY = 101,
	ERR_CPU_CAMERA_INIT_FAILED = 102,
	ERR_CPU_RENDER_FAILED = 103,
	ERR_CPU_THREAD_FAILED = 104,
	ERR_CPU_MEMORY_ALLOCATION = 105,
	ERR_CPU_BVH_BUILD_FAILED = 106,
	ERR_CPU_TEXTURE_LOAD_FAILED = 107,
	ERR_CPU_LIGHTS_EMPTY = 108,
	ERR_CPU_MATERIAL_INVALID = 109,

	// ===== GPU RENDERER ERRORS (200-299) =====
	ERR_GPU_NO_DEVICE = 200,
	ERR_GPU_DEVICE_INIT_FAILED = 201,
	ERR_GPU_MEMORY_ALLOCATION = 202,
	ERR_GPU_MEMORY_COPY_FAILED = 203,
	ERR_GPU_KERNEL_LAUNCH_FAILED = 204,
	ERR_GPU_KERNEL_EXECUTION_FAILED = 205,
	ERR_GPU_SCENE_SERIALIZATION_FAILED = 206,
	ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED = 207,
	ERR_GPU_OUT_OF_MEMORY = 208,
	ERR_GPU_INVALID_CONFIGURATION = 209,
	ERR_GPU_TEXTURE_BINDING_FAILED = 210,
	ERR_GPU_UNSUPPORTED_SCENE = 211,

	// ===== USER CANCELLATION =====
	ERR_USER_CANCELLED = 999
};

// ============================================================================
// Error Code to String Message
// ============================================================================
inline std::string get_error_message(int error_code) {
	static const std::map<int, std::string> error_messages = {
		// Success
		{SUCCESS, "Success"},

		// General errors (1-99)
		{ERR_UNKNOWN, "Unknown error occurred"},
		{ERR_INVALID_ARGUMENTS, "Invalid command-line arguments"},
		{ERR_FILE_NOT_FOUND, "Required file not found"},
		{ERR_FILE_READ_FAILED, "Failed to read file"},
		{ERR_FILE_WRITE_FAILED, "Failed to write output file"},
		{ERR_FILE_COPY_FAILED, "Failed to copy output file"},
		{ERR_DIRECTORY_CREATE_FAILED, "Failed to create output directory"},
		{ERR_INVALID_DIMENSIONS, "Invalid image dimensions (must be > 0)"},
		{ERR_INVALID_SAMPLE_COUNT, "Invalid sample count (must be > 0)"},
		{ERR_INVALID_MAX_DEPTH, "Invalid max depth (must be > 0)"},
		{ERR_INVALID_SCENE_ID, "Invalid scene ID (must be 0-8)"},
		{ERR_INVALID_CAMERA_POSITION, "Invalid camera position coordinates"},
		{ERR_OUTPUT_PATH_INVALID, "Output path is invalid or not writable"},

		// CPU renderer errors (100-199)
		{ERR_CPU_SCENE_BUILD_FAILED, "CPU: Failed to build scene"},
		{ERR_CPU_SCENE_EMPTY, "CPU: Scene contains no objects"},
		{ERR_CPU_CAMERA_INIT_FAILED, "CPU: Failed to initialize camera"},
		{ERR_CPU_RENDER_FAILED, "CPU: Rendering failed during execution"},
		{ERR_CPU_THREAD_FAILED, "CPU: Worker thread encountered an error"},
		{ERR_CPU_MEMORY_ALLOCATION, "CPU: Out of memory"},
		{ERR_CPU_BVH_BUILD_FAILED, "CPU: BVH acceleration structure build failed"},
		{ERR_CPU_TEXTURE_LOAD_FAILED, "CPU: Failed to load texture file"},
		{ERR_CPU_LIGHTS_EMPTY, "CPU: Scene has no lights for importance sampling"},
		{ERR_CPU_MATERIAL_INVALID, "CPU: Invalid material configuration"},

		// GPU renderer errors (200-299)
		{ERR_GPU_NO_DEVICE, "GPU: No CUDA-capable device found"},
		{ERR_GPU_DEVICE_INIT_FAILED, "GPU: Failed to initialize CUDA device"},
		{ERR_GPU_MEMORY_ALLOCATION, "GPU: Failed to allocate device memory"},
		{ERR_GPU_MEMORY_COPY_FAILED, "GPU: Failed to copy data to/from device"},
		{ERR_GPU_KERNEL_LAUNCH_FAILED, "GPU: Failed to launch kernel"},
		{ERR_GPU_KERNEL_EXECUTION_FAILED, "GPU: Kernel execution failed"},
		{ERR_GPU_SCENE_SERIALIZATION_FAILED, "GPU: Failed to serialize scene data"},
		{ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED, "GPU: Device synchronization failed"},
		{ERR_GPU_OUT_OF_MEMORY, "GPU: Out of device memory"},
		{ERR_GPU_INVALID_CONFIGURATION, "GPU: Invalid kernel configuration"},
		{ERR_GPU_TEXTURE_BINDING_FAILED, "GPU: Failed to bind texture"},
		{ERR_GPU_UNSUPPORTED_SCENE, "GPU: Scene not supported on GPU (use CPU mode)"},

		// User cancellation
		{ERR_USER_CANCELLED, "Render cancelled by user"}
	};

	auto it = error_messages.find(error_code);
	if (it != error_messages.end()) {
		return it->second;
	}

	// Unknown error code
	return "Unknown error code (" + std::to_string(error_code) + ")";
}

// ============================================================================
// Error Code to Troubleshooting Hint
// ============================================================================
inline std::string get_troubleshooting_hint(int error_code) {
	static const std::map<int, std::string> hints = {
		{ERR_FILE_WRITE_FAILED, "Check output directory permissions and disk space."},
		{ERR_FILE_NOT_FOUND, "Verify that required texture files (e.g., earthmap.jpg) are in the correct location."},
		{ERR_INVALID_DIMENSIONS, "Width and height must be positive integers (recommended: 400-1920)."},
		{ERR_INVALID_SAMPLE_COUNT, "Samples per pixel must be > 0 (recommended: 10-500)."},
		{ERR_INVALID_SCENE_ID, "Valid scene IDs are 0-8. Check docs/SCENE_SELECTION.md for details."},
		{ERR_CPU_SCENE_BUILD_FAILED, "Scene construction failed. Check scene ID and texture file availability."},
		{ERR_CPU_MEMORY_ALLOCATION, "Out of memory. Try reducing resolution, sample count, or scene complexity."},
		{ERR_CPU_TEXTURE_LOAD_FAILED, "Texture file missing or corrupted. For Earth scene, ensure earthmap.jpg exists."},
		{ERR_GPU_NO_DEVICE, "No CUDA-capable GPU found. Use CPU mode instead."},
		{ERR_GPU_OUT_OF_MEMORY, "GPU out of memory. Try reducing resolution or sample count, or use CPU mode."},
		{ERR_GPU_UNSUPPORTED_SCENE, "GPU currently only supports Cornell Box (scene 0). Use CPU mode for other scenes."},
		{ERR_GPU_MEMORY_ALLOCATION, "GPU memory allocation failed. Try reducing resolution or switching to CPU mode."},
		{ERR_INVALID_ARGUMENTS, "Check command-line syntax: ray_tracer.exe [--cpu|--gpu] [--output path] <width> <spp> <depth> <scene_id> <cam_x> <cam_y> <cam_z>"}
	};

	auto it = hints.find(error_code);
	if (it != hints.end()) {
		return it->second;
	}

	return "No additional information available. Check logs for details.";
}

// ============================================================================
// Error Category Helper
// ============================================================================
inline std::string get_error_category(int error_code) {
	if (error_code == SUCCESS) return "SUCCESS";
	if (error_code >= 1 && error_code <= 99) return "GENERAL";
	if (error_code >= 100 && error_code <= 199) return "CPU_RENDERER";
	if (error_code >= 200 && error_code <= 299) return "GPU_RENDERER";
	if (error_code == ERR_USER_CANCELLED) return "USER_ACTION";
	return "UNKNOWN";
}

// ============================================================================
// Complete Error Info Structure
// ============================================================================
struct ErrorInfo {
	int code;
	std::string category;
	std::string message;
	std::string hint;

	ErrorInfo(int error_code) 
		: code(error_code)
		, category(get_error_category(error_code))
		, message(get_error_message(error_code))
		, hint(get_troubleshooting_hint(error_code))
	{}

	std::string to_string() const {
		std::string result = "[" + category + " ERROR " + std::to_string(code) + "] " + message;
		if (!hint.empty() && hint != "No additional information available. Check logs for details.") {
			result += "\n  Hint: " + hint;
		}
		return result;
	}
};

#endif // ERROR_CODES_H
