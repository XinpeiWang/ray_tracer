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
#include "../src/TheRestOfYourLife/cornell_box_scene.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
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
								 double cam_x, double cam_y, double cam_z) {
	try {
		// Log render start with all parameters
		std::clog << "[cpu_interface] cpu_render_main start: " << width << "x" << height 
				  << " spp=" << spp << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		// ====================================================================
		// Scene Construction
		// ====================================================================
		// Build the Cornell box scene using shared scene definition
		// This ensures CPU and GPU renderers produce identical scenes

		hittable_list world = build_cornell_box_scene();   // Geometry: walls, sphere, box
		hittable_list lights = build_cornell_box_lights(); // Light sources for importance sampling

		// ====================================================================
		// Camera Configuration
		// ====================================================================
		// Set up camera with user-specified position and fixed target

		camera cam;
		cam.aspect_ratio      = double(width) / double(height);  // Usually 1:1 for Cornell box
		cam.image_width       = width;
		cam.samples_per_pixel = spp;       // Higher = smoother but slower
		cam.max_depth         = max_depth; // Max ray bounces (recursion limit)
		cam.background        = color(0, 0, 0); // Black background (closed room)

		cam.vfov     = 40;                           // Vertical field of view in degrees
		cam.lookfrom = point3(cam_x, cam_y, cam_z);  // Camera position (user-specified)
		cam.lookat   = point3(278, 278, 278);        // Look at center of Cornell box (fixed)
		cam.vup      = vec3(0, 1, 0);                // Up direction is +Y (fixed)
		cam.defocus_angle = 0;                       // No depth of field blur

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
				std::clog << "[cpu_interface] Failed to copy output: " << e.what() << std::endl;
				// Still return success since render worked, file is just in different location
			}
		}

		std::clog << "[cpu_interface] CPU render complete: " << output_path << std::endl;
		return 0; // Success

	} catch (const std::exception& e) {
		std::clog << "[cpu_interface] Exception: " << e.what() << std::endl;
		return 1; // Failure
	} catch (...) {
		std::clog << "[cpu_interface] Unknown exception" << std::endl;
		return 1; // Failure
	}
}
