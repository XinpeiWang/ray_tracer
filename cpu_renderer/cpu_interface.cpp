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
#include "../src/TheRestOfYourLife/power_light_sampler.h"
#include "../src/TheRestOfYourLife/error_codes.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

// ============================================================================
// cpu_render_main - CPU Render Entry Point
// ============================================================================
// C-linkage function that can be called from the launcher (main.cpp)
// Builds the Cornell box scene, configures camera, renders, and copies output
// ============================================================================

extern "C" int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path,
								 int scene_id, double cam_x, double cam_y, double cam_z,
								 int force_camera_override) {
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
		hittable_list world      = scene_desc->build_world();
		hittable_list lights_raw = scene_desc->build_lights();

		// For Cornell box scenes use explicitly-weighted power_light_list;
		// for all others wrap uniformly (equal weights = same as old hittable_list).
		// Scene 7 (Cornell Smoke) used to be included here too, but its light
		// is a different size/color than build_cornell_box_power_lights()
		// assumes and it has no glass sphere - the uniform-weight path below
		// (scene_desc->build_lights() -> build_cornell_smoke_lights(), a
		// single correctly-sized light) is both correct and, with only one
		// light in the list, equivalent to power weighting anyway.
		power_light_list lights;
		if (scene_id == 0 || scene_id == 10) {
			lights = build_cornell_box_power_lights();
		} else {
			lights = power_light_list(lights_raw);
		}

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
		// image_height is normally computed by camera::initialize() (see its
		// comment at camera.h's image_height declaration), but setup_camera()
		// below runs before initialize() - scenes 32/33's alt-camera lambdas
		// read cam.image_height directly (compute_screen_window,
		// SphericalCamera's constructor), so it must already be valid here.
		// Mirrors camera::initialize()'s exact formula so the value
		// initialize() computes later is identical (harmless recompute).
		cam.image_height      = int(cam.image_width / cam.aspect_ratio);
		cam.image_height      = (cam.image_height < 1) ? 1 : cam.image_height;
		cam.samples_per_pixel = spp;
		cam.max_depth         = max_depth;
		cam.vup               = vec3(0, 1, 0);  // Up direction is +Y

		// Apply camera config from registry
		const CameraConfig& cc = scene_desc->camera;
		cam.vfov          = cc.vfov;
		cam.background    = color(cc.bg_r, cc.bg_g, cc.bg_b);
		cam.defocus_angle = cc.defocus_angle;  // 0 = no DOF blur
		cam.focus_dist    = cc.focus_dist;
		if (cc.mode == CameraMode::UserControlled || force_camera_override) {
			// Let caller override lookfrom (camera presets in UI, or a
			// video-mode frame's animated per-frame position - the latter
			// must be honored regardless of CameraMode, since a "video" that
			// silently ignores its own animated camera and stays frozen
			// defeats the point of video mode)
			cam.lookfrom = point3(cam_x, cam_y, cam_z);
		} else {
			cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
		}
		cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
		std::cout << "[cpu_interface] Camera: vfov=" << cc.vfov
				  << " lookfrom=(" << cam.lookfrom.x() << "," << cam.lookfrom.y() << "," << cam.lookfrom.z() << ")"
				  << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;

		// Apply optional sky light and punctual lights from scene descriptor
		if (scene_desc->build_sky)
			cam.sky = scene_desc->build_sky();
		if (scene_desc->build_punct)
			cam.punct_lights = scene_desc->build_punct();
		// Apply optional alternate camera model from scene descriptor
		if (scene_desc->setup_camera)
			scene_desc->setup_camera(cam);

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
		// Technique Summary
		// ====================================================================
		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : Iterative path tracer  (pbrt-v4 PathIntegrator::Li style)" << std::endl;
		std::cout << "[TECH] Sampler        : Stratified grid + Halton LDS (base-2/3, per-pixel decorrelated) + Sobol-Owen per bounce" << std::endl;
		std::cout << "[TECH] Reconstruction : Mitchell-Netravali filter  B=1/3  C=1/3  radius=0.5 px" << std::endl;
		std::cout << "[TECH] Acceleration   : SAH BVH  |  12 buckets  |  max 4 prims/leaf  |  C_trav=1  C_isect=2" << std::endl;
		std::cout << "[TECH] Light sampling : Power-weighted alias table (Vose method)  phi = area * Le * pi" << std::endl;
		std::cout << "[TECH] MIS            : Power heuristic  beta=2  (BSDF sample + NEE light sample)" << std::endl;
		std::cout << "[TECH] Path termination: Russian Roulette per-bounce, etaScale-aware" << std::endl;
		std::cout << "[TECH] Firefly guard  : NaN / Inf samples clamped to 0" << std::endl;
		std::cout << "[TECH] Tone mapping   : ACES Narkowicz 2015 --> sRGB OETF (IEC 61966-2-1)" << std::endl;
		std::cout << "[TECH] Threading      : " << std::thread::hardware_concurrency() << " logical cores (auto idle-adjusted on Windows)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

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

extern "C" int cpu_scene_recommended_camera(int scene_id,
	double* lookfrom_x, double* lookfrom_y, double* lookfrom_z,
	double* lookat_x, double* lookat_y, double* lookat_z) {
	const SceneDescriptor* s = find_scene(scene_id);
	if (!s) return 0;
	const CameraConfig& cc = s->camera;
	// Every scene, including CameraMode::UserControlled ones (the
	// Cornell-box family), has a documented default lookfrom in the
	// registry (e.g. Cornell Box's (278,278,-800) front view - see main.cpp's
	// header comment) - it's a perfectly good video camera-path starting
	// point even though single-image rendering lets the user freely move
	// away from it, so there's no need to special-case UserControlled here.
	if (lookfrom_x) *lookfrom_x = cc.lookfrom_x;
	if (lookfrom_y) *lookfrom_y = cc.lookfrom_y;
	if (lookfrom_z) *lookfrom_z = cc.lookfrom_z;
	if (lookat_x) *lookat_x = cc.lookat_x;
	if (lookat_y) *lookat_y = cc.lookat_y;
	if (lookat_z) *lookat_z = cc.lookat_z;
	return 1;
}
