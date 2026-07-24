// ============================================================================
// CPU Renderer Implementation
// ============================================================================
// This file implements the CPU-based ray tracer entry point.
// 
// Renderer features:
//   - Multithreaded C++ path tracing
//   - Importance sampling using PDFs (probability density functions)
//   - Cornell box scene with glass sphere and rotated box
//   - Configurable camera position
//
// Camera behavior:
//   - Position (lookfrom) is set by caller via (cam_x, cam_y, cam_z)
//   - Target (lookat) is fixed at Cornell box center: (278, 278, 278)
//   - This ensures camera always points toward the center of the scene
//
// Output handling:
//   - Camera class writes to OneDrive/Desktop by default
//   - This interface copies the result to the requested output path
// ============================================================================

#include "cpu_interface.h"
#include "../src/TheRestOfYourLife/rtweekend.h"
#include "../src/TheRestOfYourLife/camera.h"
#include "../src/TheRestOfYourLife/scene_registry.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
#include "../src/TheRestOfYourLife/error_codes.h"
#include <iostream>
#include <fstream>
#include <filesystem>

// ============================================================================
// cpu_render_main - CPU Render Entry Point
// ============================================================================
// C-linkage function that can be called from the launcher (main.cpp)
// Builds the Cornell box scene, configures camera, renders, and copies output
// ============================================================================

extern "C" int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path,
								 int scene_id, double cam_x, double cam_y, double cam_z) {
	try {
		// ====================================================================
		// Parameter Validation
		// ====================================================================

		// Validate dimensions
		if (width <= 0 || height <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_DIMENSIONS).to_string() << std::endl;
			return ERR_INVALID_DIMENSIONS;
		}

		// Validate sample count
		if (spp <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_SAMPLE_COUNT).to_string() << std::endl;
			return ERR_INVALID_SAMPLE_COUNT;
		}

		// Validate max depth
		if (max_depth <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_MAX_DEPTH).to_string() << std::endl;
			return ERR_INVALID_MAX_DEPTH;
		}

		// Validate scene ID against registry
		const SceneDescriptor* scene_desc = find_scene(scene_id);
		if (!scene_desc) {
			std::cerr << ErrorInfo(ERR_INVALID_SCENE_ID).to_string() << " (received: " << scene_id << ")" << std::endl;
			return ERR_INVALID_SCENE_ID;
		}

		// Log render start with all parameters
		std::clog << "[cpu_interface] cpu_render_main start: " << width << "x" << height 
				  << " spp=" << spp << " scene_id=" << scene_id << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		// ====================================================================
		// Scene Construction
		// ====================================================================
		// Build the selected scene using the centralized scene library

		// Build world and lights via registry -- no switch needed
		std::cout << "[cpu_interface] Building scene " << scene_id << " (" << scene_desc->name << ")..." << std::endl;
		hittable_list world  = scene_desc->build_world();
		hittable_list lights = scene_desc->build_lights();

			// Validate that scene was built successfully
			if (world.objects.size() == 0) {
				std::cerr << ErrorInfo(ERR_CPU_SCENE_EMPTY).to_string() << std::endl;
				return ERR_CPU_SCENE_EMPTY;
			}

		// ====================================================================
		// Camera Configuration
		// ====================================================================
		// Set up camera with scene-specific settings

		camera cam;
		cam.aspect_ratio      = double(width) / double(height);
		cam.image_width       = width;
		cam.samples_per_pixel = spp;
		cam.max_depth         = max_depth;
		cam.vup               = vec3(0, 1, 0);  // Up direction is +Y
		cam.defocus_angle     = 0;              // No depth of field blur

		// Apply camera config from registry
		const CameraConfig& cc = scene_desc->camera;
		cam.vfov       = cc.vfov;
		cam.background = color(cc.bg_r, cc.bg_g, cc.bg_b);
		if (cc.mode == CameraMode::UserControlled) {
			// Let caller override lookfrom (camera presets in UI)
			cam.lookfrom = point3(cam_x, cam_y, cam_z);
		} else {
			cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
		}
		cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
		std::cout << "[cpu_interface] Camera: vfov=" << cc.vfov
				  << " lookfrom=(" << cam.lookfrom.x() << "," << cam.lookfrom.y() << "," << cam.lookfrom.z() << ")"
				  << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;

		// ====================================================================
		// Output Path Handling
		// ====================================================================
		// Save original requested path for later file copy

		std::string orig_path = std::string(output_path);

		// Ensure the output directory exists
		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		// ====================================================================
		// Render Execution
		// ====================================================================
		// Camera class handles multithreaded rendering and writes to a default
		// location (OneDrive/Desktop). We'll copy the file afterward.

		std::cout << "[cpu_interface] Starting render..." << std::endl;
		cam.render(world, lights);

		// ====================================================================
		// Output File Copy
		// ====================================================================
		// The camera wrote to its default Desktop location; copy to requested path
		// This is necessary because the camera class has hardcoded output logic

		// Determine where the camera actually wrote the file
		// Priority: OneDrive Desktop > regular Desktop > current directory
		std::string actual_output;
		if (const char* od = std::getenv("OneDrive")) {
			actual_output = std::string(od) + "\\Desktop\\image.ppm";
		} else if (const char* up = std::getenv("USERPROFILE")) {
			if (std::filesystem::exists(std::filesystem::path(std::string(up) + "\\OneDrive"))) {
				actual_output = std::string(up) + "\\OneDrive\\Desktop\\image.ppm";
			} else {
				actual_output = std::string(up) + "\\Desktop\\image.ppm";
			}
		} else {
			actual_output = "image.ppm";
		}

		// Copy the file to the requested location if different
		if (actual_output != output_path) {
			try {
				std::filesystem::copy_file(actual_output, output_path, std::filesystem::copy_options::overwrite_existing);
				std::clog << "[cpu_interface] Copied " << actual_output << " to " << output_path << std::endl;
			} catch (const std::exception& e) {
				std::cerr << "[cpu_interface] " << ErrorInfo(ERR_FILE_COPY_FAILED).to_string() 
						  << " - " << e.what() << std::endl;
				return ERR_FILE_COPY_FAILED;
			}
		}

		std::clog << "[cpu_interface] CPU render complete: " << output_path << std::endl;
		return SUCCESS; // Success code 0

	} catch (const std::bad_alloc& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
				  << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
				  << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

// ============================================================================
// Scene metadata C API -- read from the registry, safe to call from C / Qt
// ============================================================================

extern "C" int cpu_scene_count() {
	return scene_count();
}

extern "C" int cpu_scene_id(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return -1;
	return reg[index].id;
}

extern "C" const char* cpu_scene_name(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].name;
}

extern "C" const char* cpu_scene_description(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].description;
}

extern "C" const char* cpu_scene_performance(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].performance;
}

extern "C" int cpu_scene_recommended_spp(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 100;
	return reg[index].recommended_spp;
}

extern "C" int cpu_scene_requires_files(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 0;
	return reg[index].requires_files ? 1 : 0;
}

extern "C" int cpu_scene_gpu_compatible(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 0;
	return reg[index].gpu_compatible ? 1 : 0;
}
