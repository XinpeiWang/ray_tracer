// OptiX Interface Implementation
// C API wrapper matching gpu/cuda/gpu_interface.h signature

#include "optix_interface.h"
#include "optix_renderer.h"
#include "scene_builder.h"
#include "../../src/TheRestOfYourLife/error_codes.h"
#include "../../src/shared/tone_map.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>

// Global renderer instance
static std::unique_ptr<OptiXRenderer> g_renderer;

// Which scene_id is currently uploaded to the GPU (materials/geometry/BVH/SBT
// all live in g_renderer's device memory, keyed only by scene_id - see the
// skip-reupload check below). -1 = nothing uploaded yet this process.
static int g_uploaded_scene_id = -1;

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
	double cam_z,
	int force_camera_override
) {
	try {
		// Initialize renderer on first call
		if (!g_renderer) {
			std::cout << "[OptiX] Initializing renderer...\n";
			g_renderer = std::make_unique<OptiXRenderer>();
			if (!g_renderer->initialize()) {
				std::cerr << "[OptiX] Failed to initialize renderer\n";
				return ERR_GPU_DEVICE_INIT_FAILED;
			}
		}

		// Build scene with specified scene_id
		std::cout << "[OptiX] Building scene " << scene_id << "...\n";
		std::cout << "[OptiX] Camera position: (" << cam_x << ", " << cam_y << ", " << cam_z << ")\n";

		SceneData scene;
		float camera_params[12];  // origin(3) + lower_left(3) + horizontal(3) + vertical(3)
		GpuCameraParams cameraExtra{};  // zero-init: kind=Perspective, DOF/spherical fields all zero

		if (!build_scene(scene_id, image_width, image_height, scene, camera_params, cam_x, cam_y, cam_z, &cameraExtra, force_camera_override != 0)) {
			// build_scene() only ever returns false for an unrecognized/
			// unimplemented scene_id (its default: case) - the other
			// return-false path (a null camera_params buffer) is
			// unreachable in practice, since every caller here passes a
			// valid on-stack array. ERR_GPU_UNSUPPORTED_SCENE gives the
			// user the actionable "switch to CPU mode" message
			// (error_handler.h); the old ERR_GPU_SCENE_BUILD_FAILED here
			// was misleading - it reads as a genuine build/geometry
			// failure, not "this scene was never ported to GPU."
			std::cerr << "[OptiX] Scene not supported on GPU\n";
			return ERR_GPU_UNSUPPORTED_SCENE;
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

		// buildScene() only touches geometry/material/light device memory -
		// it never sees camera state (that's render()'s cameraExtra param,
		// computed fresh above on every call regardless) - so re-uploading
		// and rebuilding the BVH/SBT is only actually necessary when the
		// scene_id changes from the last call in this process. This is what
		// makes video rendering (same scene, moving camera, many frames in
		// one process) expensive: skip it when nothing but the camera moved.
		if (scene_id != g_uploaded_scene_id) {
			// Instanced geometry travels separately - see setInstanceData().
			// Called inside this same cache-skip guard as buildScene() itself,
			// so instance data is part of what "this scene is already
			// uploaded" means: a scene switch that reuses geometry must not
			// keep a PREVIOUS scene's placements around.
			g_renderer->setInstanceData(scene.instanceTriangles, scene.instanceSpheres,
										scene.instanceGroups, scene.instancePlacements);
			if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials,
										 scene.lightIndices, scene.lightKinds,
										 scene.punctualLights, scene.bilinearPatches,
										 scene.triangles, scene.lensElements,
										 scene.exitPupilBounds, scene.textures,
										 scene.texturePixels)) {
				std::cerr << "[OptiX] Failed to upload scene to GPU\n";
				return ERR_GPU_MEMORY_COPY_FAILED;
			}
			g_uploaded_scene_id = scene_id;
		} else {
			std::cout << "[OptiX] Reusing already-uploaded scene " << scene_id << " (skipping GPU rebuild)\n";
		}

		// Enable wavefront mode if requested via env var RAY_TRACER_WAVEFRONT=1
#pragma warning(suppress: 4996)
		const char* wfEnv = std::getenv("RAY_TRACER_WAVEFRONT");
		if (wfEnv && std::string(wfEnv) == "1" && g_renderer->hasInstancePlacements()) {
			// The wavefront tracer builds its OWN shader binding table from
			// primitive counts alone, so it knows nothing about the extra
			// records and per-instance base offsets that placed geometry
			// needs - it would read the wrong primitives rather than fail.
			// Declining is the honest answer; silently rendering nonsense is
			// not.
			std::cerr << "[OptiX] warning: wavefront mode does not support object "
					     "instancing; rendering this scene with the default "
					     "path tracer instead.\n";
		} else if (wfEnv && std::string(wfEnv) == "1") {
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
		std::cout << "[TECH] Tone mapping   : ACES filmic (Narkowicz) + sRGB OETF  (matches CPU's write_color())" << std::endl;
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
			return ERR_GPU_RENDER_FAILED;
		}

		// Write to PPM file
		std::ofstream outFile(output_path, std::ios::binary);
		if (!outFile) {
			std::cerr << "[OptiX] Failed to open output file: " << output_path << "\n";
			return ERR_FILE_WRITE_FAILED;
		}

		// PPM header
		outFile << "P3\n" << image_width << " " << image_height << "\n255\n";

		// Write pixels with the same ACES filmic tone map + sRGB OETF CPU's
		// write_color() (src/TheRestOfYourLife/color.h) uses - previously
		// this was a plain sqrt gamma-2.0 approximation, the exact thing
		// tone_map.h's own header comment says CPU "replaced" at some point
		// without GPU's output path being updated to match, which showed up
		// as a real, visible color/contrast mismatch between the two
		// renderers (ACES's filmic S-curve shifts shadow/midtone hue and
		// softens contrast compared to a flat gamma curve).
		for (size_t i = 0; i < pixelCount; ++i) {
			double r = framebuffer[i * 3 + 0];
			double g = framebuffer[i * 3 + 1];
			double b = framebuffer[i * 3 + 2];

			// NaN/Inf guard (firefly guard), matching write_color() exactly.
			if (!std::isfinite(r)) r = 0.0;
			if (!std::isfinite(g)) g = 0.0;
			if (!std::isfinite(b)) b = 0.0;

			r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
			g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
			b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));

			// Clamp and convert to byte (matches write_color()'s
			// interval(0.000, 0.999) clamp-then-*256 convention exactly).
			// fmin/fmax, not std::min/std::max: this TU pulls in <windows.h>
			// transitively (via optix.h/cuda headers) without NOMINMAX, and
			// std::min/std::max collide with its min/max macros here.
			int ir = static_cast<int>(256.0 * std::fmin(std::fmax(r, 0.0), 0.999));
			int ig = static_cast<int>(256.0 * std::fmin(std::fmax(g, 0.0), 0.999));
			int ib = static_cast<int>(256.0 * std::fmin(std::fmax(b, 0.0), 0.999));

			outFile << ir << " " << ig << " " << ib << "\n";
		}

		outFile.close();

		std::cout << "[OptiX] Render complete! Output: " << output_path << "\n";
		return 0;  // Success

	} catch (const std::exception& e) {
		std::cerr << "[OptiX] Exception: " << e.what() << "\n";
		return ERR_GPU_EXCEPTION;
	} catch (...) {
		std::cerr << "[OptiX] Unknown error\n";
		return ERR_GPU_UNKNOWN_ERROR;
	}
}

// GPU SPPM entry point (sub-phase 1e) -- mirrors optix_render_main()'s own
// scene-build/camera-setup preamble and ACES-tonemap PPM write, but calls
// OptiXRenderer::renderSPPM() (the real multi-iteration loop, sub-phase 1d)
// instead of render(). Phase 1 scope only: any scene_id other than 11
// (CornellRoughGlass) is rejected up front, matching the plan's own
// "Explicitly out of scope for Phase 1" section.
extern "C" int optix_render_main_sppm(
	int image_width,
	int image_height,
	int iterations,
	int photons,
	int max_depth,
	const char* output_path,
	int scene_id,
	double cam_x,
	double cam_y,
	double cam_z,
	int force_camera_override
) {
	if (scene_id != 11) {
		std::cerr << "[OptiX] GPU SPPM (Phase 1) only supports scene 11 (CornellRoughGlass); got scene "
		          << scene_id << ". Use CPU SPPM (--sppm without --gpu) for other scenes.\n";
		return ERR_GPU_UNSUPPORTED_SCENE;
	}

	try {
		if (!g_renderer) {
			std::cout << "[OptiX] Initializing renderer...\n";
			g_renderer = std::make_unique<OptiXRenderer>();
			if (!g_renderer->initialize()) {
				std::cerr << "[OptiX] Failed to initialize renderer\n";
				return ERR_GPU_DEVICE_INIT_FAILED;
			}
		}

		std::cout << "[OptiX] Building scene " << scene_id << " (SPPM)...\n";
		std::cout << "[OptiX] Camera position: (" << cam_x << ", " << cam_y << ", " << cam_z << ")\n";

		SceneData scene;
		float camera_params[12];
		GpuCameraParams cameraExtra{};

		if (!build_scene(scene_id, image_width, image_height, scene, camera_params, cam_x, cam_y, cam_z, &cameraExtra, force_camera_override != 0)) {
			std::cerr << "[OptiX] Scene not supported on GPU\n";
			return ERR_GPU_UNSUPPORTED_SCENE;
		}

		bool defocusDiskZero = cameraExtra.defocus_disk_u.x == 0.0f && cameraExtra.defocus_disk_u.y == 0.0f &&
		                       cameraExtra.defocus_disk_u.z == 0.0f;
		if (cameraExtra.kind == CameraKind::Perspective && defocusDiskZero) {
			cameraExtra.origin = make_float3(camera_params[0], camera_params[1], camera_params[2]);
			cameraExtra.lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			cameraExtra.horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			cameraExtra.vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
		}

		if (scene_id != g_uploaded_scene_id) {
			// Instanced geometry travels separately - see setInstanceData().
			// Called inside this same cache-skip guard as buildScene() itself,
			// so instance data is part of what "this scene is already
			// uploaded" means: a scene switch that reuses geometry must not
			// keep a PREVIOUS scene's placements around.
			g_renderer->setInstanceData(scene.instanceTriangles, scene.instanceSpheres,
										scene.instanceGroups, scene.instancePlacements);
			if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials,
			                             scene.lightIndices, scene.lightKinds,
			                             scene.punctualLights, scene.bilinearPatches,
			                             scene.triangles, scene.lensElements,
			                             scene.exitPupilBounds, scene.textures,
			                             scene.texturePixels)) {
				std::cerr << "[OptiX] Failed to upload scene to GPU\n";
				return ERR_GPU_MEMORY_COPY_FAILED;
			}
			g_uploaded_scene_id = scene_id;
		} else {
			std::cout << "[OptiX] Reusing already-uploaded scene " << scene_id << " (skipping GPU rebuild)\n";
		}

		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : GPU SPPM (Stochastic Progressive Photon Mapping, Phase 1)" << std::endl;
		std::cout << "[TECH] Iterations     : " << iterations << "  |  Photons/iteration: " << photons << std::endl;
		std::cout << "[TECH] Materials      : Lambertian + RoughDielectric (GGX)  |  Lights: area only" << std::endl;
		std::cout << "[TECH] Hash grid      : atomic-head-swap linked list over a fixed device node pool" << std::endl;
		std::cout << "[TECH] Tone mapping   : ACES filmic (Narkowicz) + sRGB OETF  (matches CPU's write_color())" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		size_t pixelCount = static_cast<size_t>(image_width) * image_height;
		std::vector<float> framebuffer(pixelCount * 3);

		std::string outStr(output_path);
		size_t sep = outStr.rfind('\\');
		if (sep == std::string::npos) sep = outStr.rfind('/');
		std::string ptxPath = (sep != std::string::npos)
			? outStr.substr(0, sep + 1) + "sppm_programs.ptx"
			: "sppm_programs.ptx";

		std::cout << "[OptiX] Rendering (SPPM)...\n";
		// initialRadius: fixed at Phase 1 (5.0, matching CPU SPPM's own
		// scene-11 default -- see cpu_render_main_sppm's call site) rather
		// than exposed as a CLI parameter yet; --sppm's existing CLI surface
		// (main.cpp/launcher_args.h) has no radius flag for the CPU path
		// either, so GPU staying consistent with that is the faithful port,
		// not a gap introduced here.
		if (!g_renderer->renderSPPM(
				static_cast<unsigned int>(image_width), static_cast<unsigned int>(image_height),
				iterations, photons, static_cast<unsigned int>(max_depth), 5.0f,
				cameraExtra, framebuffer.data(), ptxPath)) {
			std::cerr << "[OptiX] SPPM render failed\n";
			return ERR_GPU_RENDER_FAILED;
		}

		std::ofstream outFile(output_path, std::ios::binary);
		if (!outFile) {
			std::cerr << "[OptiX] Failed to open output file: " << output_path << "\n";
			return ERR_FILE_WRITE_FAILED;
		}
		outFile << "P3\n" << image_width << " " << image_height << "\n255\n";
		for (size_t i = 0; i < pixelCount; ++i) {
			double r = framebuffer[i * 3 + 0];
			double g = framebuffer[i * 3 + 1];
			double b = framebuffer[i * 3 + 2];
			if (!std::isfinite(r)) r = 0.0;
			if (!std::isfinite(g)) g = 0.0;
			if (!std::isfinite(b)) b = 0.0;
			r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
			g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
			b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
			int ir = static_cast<int>(256.0 * std::fmin(std::fmax(r, 0.0), 0.999));
			int ig = static_cast<int>(256.0 * std::fmin(std::fmax(g, 0.0), 0.999));
			int ib = static_cast<int>(256.0 * std::fmin(std::fmax(b, 0.0), 0.999));
			outFile << ir << " " << ig << " " << ib << "\n";
		}
		outFile.close();

		std::cout << "[OptiX] SPPM render complete! Output: " << output_path << "\n";
		return 0;

	} catch (const std::exception& e) {
		std::cerr << "[OptiX] Exception: " << e.what() << "\n";
		return ERR_GPU_EXCEPTION;
	} catch (...) {
		std::cerr << "[OptiX] Unknown error\n";
		return ERR_GPU_UNKNOWN_ERROR;
	}
}

extern "C" int gpu_scene_light_count(int scene_id, int image_width, int image_height) {
	try {
		SceneData scene;
		float camera_params[12];
		GpuCameraParams cameraExtra{};
		if (!build_scene(scene_id, image_width, image_height, scene, camera_params,
						  278.0, 278.0, -800.0, &cameraExtra)) {
			return -1;
		}
		return static_cast<int>(scene.lightIndices.size());
	} catch (...) {
		return -1;
	}
}
