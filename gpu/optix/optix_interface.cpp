// OptiX Interface Implementation
// C API wrapper matching gpu/cuda/gpu_interface.h signature

#include "optix_interface.h"
#include "optix_renderer.h"
#include "scene_builder.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>

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
	const char* output_path,
	int scene_id,
	double cam_x,
	double cam_y,
	double cam_z
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

		// Build scene with specified scene_id
		std::cout << "[OptiX] Building scene " << scene_id << "...\n";
		std::cout << "[OptiX] Camera position: (" << cam_x << ", " << cam_y << ", " << cam_z << ")\n";

		SceneData scene;
		float camera_params[12];  // origin(3) + lower_left(3) + horizontal(3) + vertical(3)
		GpuCameraParams cameraExtra{};  // zero-init: kind=Perspective, DOF/spherical fields all zero

		if (!build_scene(scene_id, image_width, image_height, scene, camera_params, cam_x, cam_y, cam_z, &cameraExtra)) {
			std::cerr << "[OptiX] Failed to build scene\n";
			return 101;  // Scene build error
		}

		// Scenes that don't use a non-default camera model leave cameraExtra
		// untouched by build_scene() (still zero-init'd: kind=Perspective,
		// defocus disk zero) - fill it from the plain 12-float camera_params
		// in that case, matching every scene's prior behavior. Scenes that
		// DID request a non-default model (Orthographic/Spherical, or
		// Perspective with DOF) set kind and/or a nonzero defocus disk
		// themselves in build_scene(), so this check reliably tells the two
		// cases apart.
		bool defocusDiskZero = cameraExtra.defocus_disk_u.x == 0.0f && cameraExtra.defocus_disk_u.y == 0.0f &&
								cameraExtra.defocus_disk_u.z == 0.0f;
		if (cameraExtra.kind == CameraKind::Perspective && defocusDiskZero) {
			cameraExtra.origin = make_float3(camera_params[0], camera_params[1], camera_params[2]);
			cameraExtra.lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			cameraExtra.horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			cameraExtra.vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
		}

		if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials,
									 scene.lightIndices, scene.isLightSphere,
									 scene.punctualLights, scene.bilinearPatches,
									 scene.triangles, scene.lensElements,
									 scene.exitPupilBounds)) {
			std::cerr << "[OptiX] Failed to upload scene to GPU\n";
			return 102;  // GPU upload error
		}

		// Enable wavefront mode if requested via env var RAY_TRACER_WAVEFRONT=1
#pragma warning(suppress: 4996)
		const char* wfEnv = std::getenv("RAY_TRACER_WAVEFRONT");
		if (wfEnv && std::string(wfEnv) == "1") {
			// Derive PTX path: same directory as output_path, or executable directory
			std::string ptxPath;
			std::string outStr(output_path);
			size_t sep = outStr.rfind('\\');
			if (sep == std::string::npos) sep = outStr.rfind('/');
			if (sep != std::string::npos)
				ptxPath = outStr.substr(0, sep + 1) + "wavefront_programs.ptx";
			else
				ptxPath = "wavefront_programs.ptx";
			std::cout << "[OptiX] Wavefront mode enabled (PTX: " << ptxPath << ")\n";
			g_renderer->enableWavefront(true, ptxPath);
		}

		// Allocate float framebuffer
		size_t pixelCount = image_width * image_height;
		std::vector<float> framebuffer(pixelCount * 3);

		// Technique summary
		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : OptiX mega-kernel path tracer  (raygen + closest-hit + miss programs)" << std::endl;
		std::cout << "[TECH] Sampler        : PCG hash-based pseudo-random (per-pixel seed, per-bounce offset)" << std::endl;
		std::cout << "[TECH] Reconstruction : Box filter (1 sample per pixel sub-region, averaged in kernel)" << std::endl;
		std::cout << "[TECH] Acceleration   : OptiX BVH  |  custom AABB primitives  (spheres + quads)" << std::endl;
		std::cout << "[TECH] Light sampling : Power-weighted alias table (Vose method)  phi = area * Le * pi" << std::endl;
		std::cout << "[TECH] MIS            : Power heuristic  beta=2  (BSDF sample + NEE light sample)" << std::endl;
		std::cout << "[TECH] Path termination: fixed max_depth=" << max_depth << "  (no Russian Roulette on GPU)" << std::endl;
		std::cout << "[TECH] Tone mapping   : sqrt gamma-2.0  (linear gamma correction, no filmic tone map)" << std::endl;
		std::cout << "[TECH] Device         : CUDA / OptiX 7+  (NVIDIA GPU)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		// Render
		std::cout << "[OptiX] Rendering...\n";

		if (!g_renderer->render(
			image_width,
			image_height,
			samples_per_pixel,
			max_depth,
			cameraExtra,
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
