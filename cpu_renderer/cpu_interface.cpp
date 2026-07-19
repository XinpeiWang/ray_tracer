#include "cpu_interface.h"
#include "../src/TheRestOfYourLife/rtweekend.h"
#include "../src/TheRestOfYourLife/camera.h"
#include "../src/TheRestOfYourLife/cornell_box_scene.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
#include <iostream>
#include <fstream>
#include <filesystem>

// C linkage for external call from launcher
extern "C" int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path) {
	try {
		std::clog << "[cpu_interface] cpu_render_main start: " << width << "x" << height 
				  << " spp=" << spp << " out=" << output_path << std::endl;

		// Build the Cornell box scene (shared with GPU)
		hittable_list world = build_cornell_box_scene();
		hittable_list lights = build_cornell_box_lights();

		// Configure camera
		camera cam;
		cam.aspect_ratio      = double(width) / double(height);
		cam.image_width       = width;
		cam.samples_per_pixel = spp;
		cam.max_depth         = max_depth;
		cam.background        = color(0, 0, 0); // Cornell box closed room

		cam.vfov     = 40;
		cam.lookfrom = point3(278, 278, -800);
		cam.lookat   = point3(278, 278, 0);
		cam.vup      = vec3(0, 1, 0);
		cam.defocus_angle = 0;

		// Save the original output path and temporarily redirect to the specified path
		std::string orig_path = std::string(output_path);

		// Ensure the output directory exists
		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		// Redirect stdout to the output file for the camera render
		std::streambuf* original_cout = std::cout.rdbuf();
		std::ofstream output_file(output_path);
		if (!output_file.is_open()) {
			std::clog << "[cpu_interface] Failed to open output file: " << output_path << std::endl;
			return 1;
		}
		std::cout.rdbuf(output_file.rdbuf());

		// Render the scene (camera writes to stdout in PPM format)
		cam.render(world, lights);

		// Restore stdout
		std::cout.rdbuf(original_cout);
		output_file.close();

		std::clog << "[cpu_interface] CPU render complete: " << output_path << std::endl;
		return 0;

	} catch (const std::exception& e) {
		std::clog << "[cpu_interface] Exception: " << e.what() << std::endl;
		return 1;
	} catch (...) {
		std::clog << "[cpu_interface] Unknown exception" << std::endl;
		return 1;
	}
}
