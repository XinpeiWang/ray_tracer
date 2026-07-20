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
#include "../src/TheRestOfYourLife/scenes.h"
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

		// Validate scene ID
		if (scene_id < 0 || scene_id > 8) {
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

		hittable_list world;
		hittable_list lights;

		std::cout << "[cpu_interface] Building scene " << scene_id << "..." << std::endl;

		// Dispatch to the appropriate scene builder based on scene_id
		switch (scene_id) {
			case 0:  // Cornell Box
				std::cout << "[cpu_interface] Calling build_cornell_box()..." << std::endl;
				world = build_cornell_box();
				std::cout << "[cpu_interface] Calling build_cornell_box_lights()..." << std::endl;
				lights = build_cornell_box_lights();
				std::cout << "[cpu_interface] Cornell Box scene built successfully" << std::endl;
				break;
			case 1:  // Bouncing Spheres
				std::cout << "[cpu_interface] Calling build_bouncing_spheres()..." << std::endl;
				world = build_bouncing_spheres();
				std::cout << "[cpu_interface] Bouncing Spheres scene built successfully" << std::endl;
				// No explicit lights, but add a dummy "sky" light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
			case 2:  // Checkered Spheres
				world = build_checkered_spheres();
				// Add dummy light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
			case 3:  // Earth
				world = build_earth();
				// Add dummy light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
			case 4:  // Perlin Spheres
				world = build_perlin_spheres();
				// Add dummy light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
			case 5:  // Quads
				world = build_quads();
				// Add dummy light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
			case 6:  // Simple Light
				world = build_simple_light();
				// TODO: Add light sources for importance sampling if needed
				break;
			case 7:  // Cornell Smoke
				world = build_cornell_smoke();
				lights = build_cornell_box_lights();
				break;
			case 8:  // Final Scene
				world = build_final_scene();
				// Add dummy light for PDF sampling
				{
					auto empty_mat = shared_ptr<material>();
					lights.add(make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
				}
				break;
				default:
					std::cerr << "[cpu_interface] " << ErrorInfo(ERR_INVALID_SCENE_ID).to_string() << " (received: " << scene_id << ")" << std::endl;
					return ERR_INVALID_SCENE_ID;
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
		cam.samples_per_pixel = spp;
		cam.max_depth         = max_depth;
		cam.vup               = vec3(0, 1, 0);  // Up direction is +Y
		cam.defocus_angle     = 0;              // No depth of field blur

		// Scene-specific camera settings
		switch (scene_id) {
			case 0:  // Cornell Box
				cam.background = color(0, 0, 0);           // Black background
				cam.vfov       = 40;
				cam.lookfrom   = point3(cam_x, cam_y, cam_z);
				cam.lookat     = point3(278, 278, 278);    // Cornell box center
				std::cout << "[cpu_interface] Cornell Box camera: lookfrom=(" << cam_x << "," << cam_y << "," << cam_z << ") lookat=(278,278,278)" << std::endl;
				break;

			case 1:  // Bouncing Spheres
				cam.background = color(0.70, 0.80, 1.00);  // Sky gradient
				cam.vfov       = 20;
				cam.lookfrom   = point3(13, 2, 3);         // Default position
				cam.lookat     = point3(0, 0, 0);          // Look at origin
				std::cout << "[cpu_interface] Bouncing Spheres camera: lookfrom=(13,2,3) lookat=(0,0,0)" << std::endl;
				break;

			case 2:  // Checkered Spheres
				cam.background = color(0.70, 0.80, 1.00);  // Sky gradient
				cam.vfov       = 20;
				cam.lookfrom   = point3(13, 2, 3);
				cam.lookat     = point3(0, 0, 0);
				std::cout << "[cpu_interface] Checkered Spheres camera: lookfrom=(13,2,3) lookat=(0,0,0)" << std::endl;
				break;

			case 3:  // Earth
				cam.background = color(0.70, 0.80, 1.00);  // Sky gradient
				cam.vfov       = 20;
				cam.lookfrom   = point3(0, 0, 12);
				cam.lookat     = point3(0, 0, 0);
				std::cout << "[cpu_interface] Earth camera: lookfrom=(0,0,12) lookat=(0,0,0)" << std::endl;
				break;

			case 4:  // Perlin Spheres
				cam.background = color(0.70, 0.80, 1.00);  // Sky gradient
				cam.vfov       = 20;
				cam.lookfrom   = point3(13, 2, 3);
				cam.lookat     = point3(0, 0, 0);
				std::cout << "[cpu_interface] Perlin Spheres camera: lookfrom=(13,2,3) lookat=(0,0,0)" << std::endl;
				break;

			case 5:  // Quads
				cam.background = color(0.70, 0.80, 1.00);  // Sky gradient
				cam.vfov       = 80;
				cam.lookfrom   = point3(0, 0, 9);
				cam.lookat     = point3(0, 0, 0);
				std::cout << "[cpu_interface] Quads camera: lookfrom=(0,0,9) lookat=(0,0,0)" << std::endl;
				break;

			case 6:  // Simple Light
				cam.background = color(0, 0, 0);           // Black background
				cam.vfov       = 20;
				cam.lookfrom   = point3(26, 3, 6);
				cam.lookat     = point3(0, 2, 0);
				std::cout << "[cpu_interface] Simple Light camera: lookfrom=(26,3,6) lookat=(0,2,0)" << std::endl;
				break;

			case 7:  // Cornell Smoke
				cam.background = color(0, 0, 0);           // Black background
				cam.vfov       = 40;
				cam.lookfrom   = point3(cam_x, cam_y, cam_z);
				cam.lookat     = point3(278, 278, 278);
				std::cout << "[cpu_interface] Cornell Smoke camera: lookfrom=(" << cam_x << "," << cam_y << "," << cam_z << ") lookat=(278,278,278)" << std::endl;
				break;

			case 8:  // Final Scene
				cam.background = color(0, 0, 0);           // Black background
				cam.vfov       = 40;
				cam.lookfrom   = point3(478, 278, -600);
				cam.lookat     = point3(278, 278, 0);
				std::cout << "[cpu_interface] Final Scene camera: lookfrom=(478,278,-600) lookat=(278,278,0)" << std::endl;
				break;

			default:  // Fallback to Cornell Box
				cam.background = color(0, 0, 0);
				cam.vfov       = 40;
				cam.lookfrom   = point3(cam_x, cam_y, cam_z);
				cam.lookat     = point3(278, 278, 278);
				std::cout << "[cpu_interface] Default camera (Cornell Box style): lookfrom=(" << cam_x << "," << cam_y << "," << cam_z << ") lookat=(278,278,278)" << std::endl;
				break;
		}

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
