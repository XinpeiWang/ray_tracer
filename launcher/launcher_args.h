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
}

struct LaunchArgs {
	bool use_gpu            = true;
	bool force_cpu          = false;
	bool video_mode         = false;
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
