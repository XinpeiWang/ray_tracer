#pragma once
// launcher_args.h -- Command-line argument parsing for the ray tracer launcher.
//
// Provides LaunchArgs struct and parse_launch_args() function.
// Call parse_launch_args(argc, argv) to get a fully populated LaunchArgs.
// Returns false if the program should exit (e.g., --help was passed).

#include <iostream>
#include <string>
#include <vector>
#include <set>

namespace {
	constexpr int kDefaultWidth = 600;
	constexpr int kDefaultHeight = 600;
	constexpr int kDefaultSamplesPerPixel = 500;
	constexpr int kDefaultMaxDepth = 20;
	constexpr int kDefaultSceneId = 0;
	constexpr double kCornellBoxCenter = 278.0;
	constexpr double kDefaultCameraX = 278.0;
	constexpr double kDefaultCameraY = 278.0;
	constexpr double kDefaultCameraZ = -800.0;
	constexpr int kDefaultSppmIterations = 100;
	constexpr int kDefaultSppmPhotons = 5000;
}

struct LaunchArgs {
	bool use_gpu            = true;
	bool force_cpu          = false;
	// GPU-only: use the wavefront (queue-based) path tracer instead of the
	// default recursive (megakernel) one - see gpu/optix/wavefront_path_tracer.h.
	// Ignored under --cpu/--sppm.
	bool use_wavefront      = false;
	// GPU-only: enable OptiX validation mode (extra device-side checks, real
	// per-launch cost) - see OptiXRenderer::createContext()'s own comment.
	// Ignored under --cpu/--sppm.
	bool optix_validate     = false;
	bool video_mode         = false;
	// Stochastic Progressive Photon Mapping - a separate CPU-only render
	// mode (see cpu_renderer/cpu_interface.h's cpu_render_main_sppm() doc
	// comment), not a flag on the existing --cpu/--gpu path tracer. Takes
	// priority over use_gpu/force_cpu when set (main.cpp checks it first).
	bool   use_sppm         = false;
	int    sppm_iterations  = kDefaultSppmIterations;
	int    sppm_photons     = kDefaultSppmPhotons;
	int  video_frames       = 120;
	int  video_fps          = 30;
	double video_speed      = 1.0;
	std::string camera_path = "orbit";
	std::string custom_output_path;

	int    image_width       = kDefaultWidth;
	int    image_height      = kDefaultHeight;
	int    samples_per_pixel = kDefaultSamplesPerPixel;
	int    max_ray_depth     = kDefaultMaxDepth;
	int    scene_id          = kDefaultSceneId;
	double cam_x             = kDefaultCameraX;
	double cam_y             = kDefaultCameraY;
	double cam_z             = kDefaultCameraZ;
	// True only if the user actually passed cam_x/y/z positional args - lets
	// main.cpp tell "user explicitly chose this camera position" (honor it
	// regardless of scene) apart from "cam_x/y/z are just sitting at their
	// generic Cornell-Box-scale struct defaults" (fall back to the selected
	// scene's own recommended camera instead of forcing this default onto
	// every scene, which would misplace it just as badly as a stale GUI
	// spinbox value would for a much smaller scene).
	//
	// How this relates to the renderer's other two camera-authority
	// concepts, so all three don't have to be puzzled out independently:
	//   - CameraConfig::mode (Fixed/UserControlled, scene_registry.h) is
	//     the SCENE's own default posture: does this scene want manual
	//     camera control at all, absent any override?
	//   - cam_explicit (here) and force_camera_override (cpu_interface.h /
	//     optix_interface.h / scene_builder.h) are the same question - "does
	//     THIS render call override that default?" - asked at two different
	//     layers: cam_explicit at CLI-parse time (did the user supply
	//     cam_x/y/z at all), force_camera_override at render-call time
	//     (main.cpp sets it from cam_explicit for single images, and
	//     unconditionally for every video frame, which must always honor
	//     its own per-frame animated camera regardless of scene mode).
	// The Qt GUI (qt_gui/mainwindow_slots.cpp's onRenderClicked) computes
	// its own explicit-ness by comparing the camera spinboxes against the
	// scene's recommended default (via scene_metadata.dll) and only sends
	// cam_x/y/z when they actually differ - otherwise this fallback path
	// would never run for GUI-launched renders, since every GUI render
	// would otherwise arrive with cam_x/y/z always present.
	bool   cam_explicit      = false;
};

// Parse command-line arguments into a LaunchArgs struct.
// Returns false if the caller should exit immediately (e.g. --help printed).
inline bool parse_launch_args(int argc, char** argv, LaunchArgs& out) {
	std::set<int> consumed_args;

	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];

		if (arg == "--cpu" || arg == "-cpu") {
			out.force_cpu = true;
			out.use_gpu   = false;
			consumed_args.insert(i);
		} else if (arg == "--gpu" || arg == "-gpu") {
			out.use_gpu   = true;
			out.force_cpu = false;
			consumed_args.insert(i);
		} else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
			out.custom_output_path = argv[i + 1];
			consumed_args.insert(i);
			consumed_args.insert(i + 1);
			++i;
		} else if (arg == "--wavefront") {
			out.use_wavefront = true;
			consumed_args.insert(i);
		} else if (arg == "--optix-validate") {
			out.optix_validate = true;
			consumed_args.insert(i);
		} else if (arg == "--sppm") {
			out.use_sppm = true;
			consumed_args.insert(i);
		} else if (arg == "--sppm-iterations" && i + 1 < argc) {
			try {
				out.sppm_iterations = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --sppm-iterations value, using default\n";
			}
		} else if (arg == "--sppm-photons" && i + 1 < argc) {
			try {
				out.sppm_photons = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --sppm-photons value, using default\n";
			}
		} else if (arg == "--video") {
			out.video_mode = true;
			consumed_args.insert(i);
		} else if ((arg == "--frames" || arg == "-f") && i + 1 < argc) {
			try {
				out.video_frames = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid frame count, using default\n";
			}
		} else if (arg == "--fps" && i + 1 < argc) {
			try {
				out.video_fps = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid FPS, using default\n";
			}
		} else if (arg == "--speed" && i + 1 < argc) {
			try {
				out.video_speed = std::stod(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid speed, using default\n";
			}
		} else if ((arg == "--camera-path" || arg == "-p") && i + 1 < argc) {
			out.camera_path = argv[i + 1];
			consumed_args.insert(i);
			consumed_args.insert(i + 1);
			++i;
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0]
					  << " [--cpu|--gpu] [--output PATH] [width] [spp] [max_depth] [scene_id] [cam_x] [cam_y] [cam_z]\n"
					  << "  --cpu      : Force CPU rendering\n"
					  << "  --gpu      : Force GPU rendering (default)\n"
					  << "  --wavefront: Use the wavefront (queue-based) GPU path tracer instead of\n"
					  << "               the default recursive one. GPU-only, ignored under --cpu/--sppm.\n"
					  << "  --optix-validate: Enable OptiX validation mode (extra device-side checks,\n"
					  << "               real per-launch cost - for debugging, not routine use).\n"
					  << "               GPU-only, ignored under --cpu/--sppm.\n"
					  << "  --sppm     : Render with Stochastic Progressive Photon Mapping instead of\n"
					  << "               the path tracer (incompatible with --video). Best for hard\n"
					  << "               caustic/glass scenes. CPU: verified end-to-end on scene 11\n"
					  << "               (Cornell Rough Glass); other scenes are unverified and only\n"
					  << "               support lambertian + delta-BSDF materials. GPU (--sppm --gpu):\n"
					  << "               Phase 1, scene 11 ONLY (lambertian + rough-dielectric, area\n"
					  << "               lights only) -- any other scene falls back to an error, use\n"
					  << "               CPU SPPM (--sppm without --gpu) instead.\n"
					  << "  --sppm-iterations N: SPPM iteration count (default " << kDefaultSppmIterations << ")\n"
					  << "  --sppm-photons N   : Photons shot per SPPM iteration (default " << kDefaultSppmPhotons << ")\n"
					  << "  --output,-o: Output file path (default: ./output/image.ppm)\n"
					  << "  --video    : Enable video generation mode\n"
					  << "  --frames,-f: Number of frames for video (default: 120)\n"
					  << "  --fps      : Frames per second for video (default: 30)\n"
					  << "  --speed    : Camera movement speed multiplier for video (default: 1.0)\n"
					  << "  --camera-path,-p: Camera animation path (orbit|linear|figure8|spiral)\n"
					  << "  --help,-h  : Show this help message\n"
					  << "  width      : Image width (default " << kDefaultWidth << ", square aspect)\n"
					  << "  spp        : Samples per pixel (default " << kDefaultSamplesPerPixel << ")\n"
					  << "  max_depth  : Max ray depth (default " << kDefaultMaxDepth << ")\n"
					  << "  scene_id   : Scene selector (0=Cornell Box, default " << kDefaultSceneId << ")\n"
					  << "  cam_x/y/z  : Camera position - if omitted, uses the selected scene's own\n"
					  << "               recommended camera (see src/TheRestOfYourLife/scene_registry.h),\n"
					  << "               not a single fixed default across every scene\n";
			return false;
		}
	}

	// Parse positional numeric arguments: width spp depth scene_id cam_x cam_y cam_z
	std::vector<double> numeric_args;
	for (int i = 1; i < argc; ++i) {
		if (consumed_args.count(i) > 0) continue;
		const std::string arg = argv[i];
		if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') continue;
		try { numeric_args.push_back(std::stod(arg)); } catch (const std::exception&) {}
	}

	if (numeric_args.size() >= 1 && numeric_args[0] > 0) {
		out.image_width = out.image_height = static_cast<int>(numeric_args[0]);
	}
	if (numeric_args.size() >= 2 && numeric_args[1] > 0)
		out.samples_per_pixel = static_cast<int>(numeric_args[1]);
	if (numeric_args.size() >= 3 && numeric_args[2] > 0)
		out.max_ray_depth = static_cast<int>(numeric_args[2]);
	if (numeric_args.size() >= 4 && numeric_args[3] >= 0)
		out.scene_id = static_cast<int>(numeric_args[3]);
	if (numeric_args.size() >= 5) { out.cam_x = numeric_args[4]; out.cam_explicit = true; }
	if (numeric_args.size() >= 6) out.cam_y = numeric_args[5];
	if (numeric_args.size() >= 7) out.cam_z = numeric_args[6];

	return true;
}
