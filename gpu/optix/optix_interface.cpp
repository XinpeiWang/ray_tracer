// OptiX Interface Implementation
// C API wrapper matching gpu/cuda/gpu_interface.h signature

#include "optix_interface.h"
#include "optix_renderer.h"
#include "scene_builder.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

// Global renderer instance
static std::unique_ptr<OptiXRenderer> g_renderer;

extern "C" bool optix_is_available() {
	return OptiXRenderer::isAvailable();
}

extern "C" int optix_render_main(
	int image_width,
	int image_height,
	int samples_per_pixel,
	int max_depth,
	const char* output_path
) {
	try {
		// Initialize renderer on first call
		if (!g_renderer) {
			std::cout << "[OptiX] Initializing renderer...\n";
			g_renderer = std::make_unique<OptiXRenderer>();
			if (!g_renderer->initialize()) {
				std::cerr << "[OptiX] Failed to initialize renderer\n";
				return 100;  // Init error
			}
		}

		// Build Cornell Box scene (scene_id=0)
		std::cout << "[OptiX] Building Cornell Box scene...\n";

		SceneData scene;
		float camera_params[12];  // origin(3) + lower_left(3) + horizontal(3) + vertical(3)

		if (!build_scene(0, image_width, image_height, scene, camera_params)) {
			std::cerr << "[OptiX] Failed to build scene\n";
			return 101;  // Scene build error
		}

		if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials)) {
			std::cerr << "[OptiX] Failed to upload scene to GPU\n";
			return 102;  // GPU upload error
		}

		// Allocate float framebuffer
		size_t pixelCount = image_width * image_height;
		std::vector<float> framebuffer(pixelCount * 3);

		// Render
		std::cout << "[OptiX] Rendering...\n";

		if (!g_renderer->render(
			image_width,
			image_height,
			samples_per_pixel,
			max_depth,
			&camera_params[0],   // origin
			&camera_params[3],   // lower_left
			&camera_params[6],   // horizontal
			&camera_params[9],   // vertical
			framebuffer.data()
		)) {
			std::cerr << "[OptiX] Render failed\n";
			return 103;  // Render error
		}

		// Write to PPM file
		std::ofstream outFile(output_path, std::ios::binary);
		if (!outFile) {
			std::cerr << "[OptiX] Failed to open output file: " << output_path << "\n";
			return 104;  // File error
		}

		// PPM header
		outFile << "P3\n" << image_width << " " << image_height << "\n255\n";

		// Write pixels with gamma correction
		for (size_t i = 0; i < pixelCount; ++i) {
			float r = framebuffer[i * 3 + 0];
			float g = framebuffer[i * 3 + 1];
			float b = framebuffer[i * 3 + 2];

			// Gamma correction (gamma = 2.0)
			r = sqrtf(fmaxf(r, 0.0f));
			g = sqrtf(fmaxf(g, 0.0f));
			b = sqrtf(fmaxf(b, 0.0f));

			// Clamp and convert to byte
			int ir = static_cast<int>(fminf(r * 255.999f, 255.0f));
			int ig = static_cast<int>(fminf(g * 255.999f, 255.0f));
			int ib = static_cast<int>(fminf(b * 255.999f, 255.0f));

			outFile << ir << " " << ig << " " << ib << "\n";
		}

		outFile.close();

		std::cout << "[OptiX] Render complete! Output: " << output_path << "\n";
		return 0;  // Success

	} catch (const std::exception& e) {
		std::cerr << "[OptiX] Exception: " << e.what() << "\n";
		return 105;  // Exception
	} catch (...) {
		std::cerr << "[OptiX] Unknown error\n";
		return 106;  // Unknown error
	}
}
