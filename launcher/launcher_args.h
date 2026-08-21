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
#include <algorithm>
#include <cctype>

#include "../src/shared/video_preset.h"

namespace {
	constexpr int kDefaultWidth = 600;
	constexpr int kDefaultHeight = 600;
	constexpr int kDefaultSamplesPerPixel = 500;
	constexpr int kDefaultMaxDepth = 20;
	constexpr const char* kDefaultSceneId = "A1";
	constexpr double kCornellBoxCenter = 278.0;
	constexpr double kDefaultCameraX = 278.0;
	constexpr double kDefaultCameraY = 278.0;
	constexpr double kDefaultCameraZ = -800.0;
	constexpr int kDefaultSppmIterations = 100;
	constexpr int kDefaultSppmPhotons = 5000;
	// BDPT: smaller than kDefaultMaxDepth deliberately -- BDPT's per-sample
	// cost is O(maxDepth^2) (every (s,t) strategy pair up to that depth is
	// connected and MIS-weighted, unlike the path tracer's O(maxDepth) single
	// walk), so pbrt-v4's own CLI default (5) is a much better starting point
	// than reusing the path tracer's 20.
	constexpr int kDefaultBdptMaxDepth = 5;
	// MLT: pbrt-v4's own ballpark defaults (Render.cpp), scaled down for a
	// CPU-only render in this codebase's existing time budget -- see
	// launcher_args.h's --mlt-* help text for how to raise these for a
	// higher-quality (slower) render.
	constexpr int kDefaultMltBootstrap = 100000;
	constexpr long long kDefaultMltMutations = 4000000;
	constexpr int kDefaultMltMaxDepth = 5;
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
	// GPU-only, recursive backend only: run the OptiX AI denoiser on the
	// finished render - see OptiXRenderer::enableDenoise()'s comment
	// (albedo/normal-guided AOV denoising). Silently has no effect under
	// --wavefront (see optix_render_main()'s own comment); ignored under
	// --cpu/--sppm/--bdpt/--mlt like use_wavefront/optix_validate above.
	bool denoise            = false;
	// Flat post-multiply on linear color right before tone-mapping, applied
	// on both CPU and GPU output paths (camera.h / optix_interface.cpp) -
	// the default path tracer only. Mirrors pbrt-v4's PixelSensor::
	// imagingRatio = exposureTime * ISO / 100 (film.cpp) collapsed to a
	// single scalar - the only brightness knob this project has outside of
	// changing scene light intensity itself. Default 1.0 matches pbrt's own
	// passthrough default (no-op). Has no effect under --bdpt/--mlt/--sppm
	// (main.cpp warns) - none of those entry points take an exposure
	// parameter, same scope cut as denoise's own --wavefront exclusion above.
	double exposure         = 1.0;
	// Which ported pbrt-v4 sampler (src/shared/sobol_sampler.h,
	// stratified_sampler.h, pmj02_sampler.h, halton_sampler.h) drives
	// camera.h's random decisions this render - see camera.h's SamplerKind/
	// sampler_kind_from_name(). Empty (default) means "use sobol", this
	// project's pre-existing hardcoded default - CPU only, same "only the
	// default path tracer supports it" scope cut as exposure above (no GPU
	// sampler-selection exists yet, and BDPT/MLT/SPPM each have their own
	// sampling scheme already).
	std::string sampler     = "";
	bool video_mode         = false;
	// Stochastic Progressive Photon Mapping - a separate CPU-only render
	// mode (see cpu_renderer/cpu_interface.h's cpu_render_main_sppm() doc
	// comment), not a flag on the existing --cpu/--gpu path tracer. Takes
	// priority over use_gpu/force_cpu when set (main.cpp checks it first).
	bool   use_sppm         = false;
	int    sppm_iterations  = kDefaultSppmIterations;
	int    sppm_photons     = kDefaultSppmPhotons;
	// Bidirectional Path Tracing - another separate CPU-only render mode
	// (see cpu_renderer/cpu_interface.h's cpu_render_main_bdpt() doc
	// comment), same "takes priority over use_gpu/force_cpu, mutually
	// exclusive with the other special render modes" shape as use_sppm.
	// CPU only - no GPU/OptiX implementation exists (see --gpu handling in
	// main.cpp and this struct's own validation below).
	bool   use_bdpt         = false;
	int    bdpt_max_depth   = kDefaultBdptMaxDepth;
	// Metropolis Light Transport - same shape as use_bdpt/use_sppm above.
	bool   use_mlt          = false;
	int    mlt_bootstrap    = kDefaultMltBootstrap;
	long long mlt_mutations = kDefaultMltMutations;
	int    mlt_max_depth    = kDefaultMltMaxDepth;
	// System-compatibility report (see launcher/diagnostics.h) instead of a
	// render - prints OS/CPU/GPU/CUDA/OptiX/disk/scene-asset info and exits.
	// Reuses custom_output_path (below) for --output rather than a second
	// flag: main.cpp checks this before any of the render-mode fields are
	// otherwise used.
	bool   diagnose         = false;
	int  video_frames       = 120;
	int  video_fps          = 30;
	double video_speed      = 1.0;
	std::string camera_path = "orbit";
	std::string custom_output_path;

	int    image_width       = kDefaultWidth;
	int    image_height      = kDefaultHeight;
	int    samples_per_pixel = kDefaultSamplesPerPixel;
	int    max_ray_depth     = kDefaultMaxDepth;
	std::string scene_id     = kDefaultSceneId;
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

	// True only if the user actually passed --gpu/-gpu - lets main.cpp warn
	// specifically when --gpu was a deliberate choice that --bdpt/--mlt (no
	// GPU implementation at all - see cpu_interface.h's cpu_render_main_bdpt()
	// doc comment) can't honor, without spamming that warning on the common
	// `--bdpt` invocation, where use_gpu just sits at its own struct default.
	bool   gpu_flag_explicit = false;
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
			out.gpu_flag_explicit = true;
			consumed_args.insert(i);
		} else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
			out.custom_output_path = argv[i + 1];
			consumed_args.insert(i);
			consumed_args.insert(i + 1);
			++i;
		} else if (arg == "--diagnose") {
			out.diagnose = true;
			consumed_args.insert(i);
		} else if (arg == "--wavefront") {
			out.use_wavefront = true;
			consumed_args.insert(i);
		} else if (arg == "--optix-validate") {
			out.optix_validate = true;
			consumed_args.insert(i);
		} else if (arg == "--denoise") {
			out.denoise = true;
			consumed_args.insert(i);
		} else if (arg == "--exposure" && i + 1 < argc) {
			try {
				out.exposure = std::stod(argv[i + 1]);
				// exposure <= 0 is a valid double but not a valid exposure -
				// linear_to_srgb clamps non-positive input to 0, so this
				// would otherwise silently render solid black with nothing
				// telling the user their value was nonsensical.
				if (out.exposure <= 0.0) {
					std::cerr << "Warning: --exposure " << out.exposure
							  << " is <= 0, image will render solid black\n";
				}
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --exposure value, using default\n";
			}
		} else if (arg == "--sampler" && i + 1 < argc) {
			std::string name = argv[i + 1];
			std::transform(name.begin(), name.end(), name.begin(),
							[](unsigned char c) { return std::tolower(c); });
			static const std::set<std::string> kValidSamplers = {
				"sobol", "zsobol", "paddedsobol", "stratified", "pmj02bn", "halton"};
			if (kValidSamplers.count(name)) {
				out.sampler = name;
			} else {
				std::cerr << "Invalid --sampler \"" << argv[i + 1] << "\", using default (sobol). "
							 "Valid: sobol, zsobol, paddedsobol, stratified, pmj02bn, halton\n";
			}
			consumed_args.insert(i);
			consumed_args.insert(i + 1);
			++i;
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
		} else if (arg == "--bdpt") {
			out.use_bdpt = true;
			consumed_args.insert(i);
		} else if (arg == "--bdpt-max-depth" && i + 1 < argc) {
			try {
				out.bdpt_max_depth = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --bdpt-max-depth value, using default\n";
			}
		} else if (arg == "--mlt") {
			out.use_mlt = true;
			consumed_args.insert(i);
		} else if (arg == "--mlt-bootstrap" && i + 1 < argc) {
			try {
				out.mlt_bootstrap = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --mlt-bootstrap value, using default\n";
			}
		} else if (arg == "--mlt-mutations" && i + 1 < argc) {
			try {
				out.mlt_mutations = std::stoll(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --mlt-mutations value, using default\n";
			}
		} else if (arg == "--mlt-max-depth" && i + 1 < argc) {
			try {
				out.mlt_max_depth = std::stoi(argv[i + 1]);
				consumed_args.insert(i);
				consumed_args.insert(i + 1);
				++i;
			} catch (const std::exception&) {
				std::cerr << "Invalid --mlt-max-depth value, using default\n";
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
		} else if (arg == "--video-preset" && i + 1 < argc) {
			// Sets video_mode plus all five video_preset::VideoPreset fields
			// in one shot - see src/shared/video_preset.h's own comment for
			// why this lives there rather than being duplicated per-CLI/GUI.
			// Applied immediately, same as every other flag here, so a flag
			// placed AFTER --video-preset on the command line overrides just
			// that one field - e.g. `--video-preset cornell-orbit --fps 60`
			// keeps the preset's scene/path/frames/speed but renders at 60fps.
			if (const video_preset::VideoPreset* preset = video_preset::find(argv[i + 1])) {
				out.video_mode = true;
				out.scene_id = preset->scene_id;
				out.camera_path = preset->camera_path;
				out.video_frames = preset->frames;
				out.video_fps = preset->fps;
				out.video_speed = preset->speed;
			} else {
				std::cerr << "Invalid --video-preset \"" << argv[i + 1] << "\" - valid presets:\n";
				for (const video_preset::VideoPreset& p : video_preset::kAll)
					std::cerr << "  " << p.id << " / " << p.key << " - " << p.name << "\n";
				return false;
			}
			consumed_args.insert(i);
			consumed_args.insert(i + 1);
			++i;
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0]
					  << " [--cpu|--gpu] [--output PATH] [width] [spp] [max_depth] [scene_id] [cam_x] [cam_y] [cam_z]\n"
					  << "  --cpu      : Force CPU rendering\n"
					  << "  --gpu      : Force GPU rendering (default)\n"
					  << "  --diagnose : Print a system-compatibility report (OS/CPU/RAM, GPU/CUDA/\n"
					  << "               OptiX, disk space, scene asset availability) instead of\n"
					  << "               rendering, and exit. Combine with --output PATH to also write\n"
					  << "               the report to a file.\n"
					  << "  --wavefront: Use the wavefront (queue-based) GPU path tracer instead of\n"
					  << "               the default recursive one. GPU-only, ignored under --cpu/--sppm.\n"
					  << "  --optix-validate: Enable OptiX validation mode (extra device-side checks,\n"
					  << "               real per-launch cost - for debugging, not routine use).\n"
					  << "               GPU-only, ignored under --cpu/--sppm.\n"
					  << "  --denoise  : Run the OptiX AI denoiser on the finished render, guided by\n"
					  << "               albedo + normal AOV buffers. GPU-only, recursive backend only -\n"
					  << "               silently has no effect under --wavefront; ignored under\n"
					  << "               --cpu/--sppm.\n"
					  << "  --exposure VALUE: Flat multiplier on linear color before tone-mapping\n"
					  << "               (default 1.0 = no-op). CPU and GPU default path tracer only.\n"
					  << "               E.g. 0.5 = darker, 2.0 = brighter. No effect under\n"
					  << "               --bdpt/--mlt/--sppm (warns).\n"
					  << "  --sampler NAME: Which ported pbrt-v4 sampler drives random decisions\n"
					  << "               (default sobol, this project's pre-existing behavior).\n"
					  << "               One of sobol, zsobol, paddedsobol, stratified, pmj02bn, halton.\n"
					  << "               CPU default path tracer only.\n"
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
					  << "  --bdpt     : Render with Bidirectional Path Tracing instead of the path\n"
					  << "               tracer (incompatible with --video, --sppm, --mlt). CPU only -\n"
					  << "               no GPU/OptiX implementation exists; --gpu is ignored (with a\n"
					  << "               warning) if combined with --bdpt. Area lights only (no\n"
					  << "               punctual/sky-light NEE yet). Verified end-to-end on scene A1\n"
					  << "               (Cornell Box) only; other scenes are unverified.\n"
					  << "  --bdpt-max-depth N : Maximum BDPT path depth (default " << kDefaultBdptMaxDepth << ")\n"
					  << "  --mlt      : Render with Metropolis Light Transport instead of the path\n"
					  << "               tracer (incompatible with --video, --sppm, --bdpt). CPU only -\n"
					  << "               no GPU/OptiX implementation exists; --gpu is ignored (with a\n"
					  << "               warning) if combined with --mlt. Same area-lights-only scope\n"
					  << "               and scene A1 verification as --bdpt (MLT is built directly on\n"
					  << "               BDPT's subpath machinery).\n"
					  << "  --mlt-bootstrap N  : MLT bootstrap samples per depth (default " << kDefaultMltBootstrap << ")\n"
					  << "  --mlt-mutations N  : Total Metropolis mutations, all chains combined\n"
					  << "                       (default " << kDefaultMltMutations << ")\n"
					  << "  --mlt-max-depth N  : Maximum BDPT path depth per MLT sample (default " << kDefaultMltMaxDepth << ")\n"
					  << "  --output,-o: Output file path (default: ./output/image.ppm)\n"
					  << "  --video    : Enable video generation mode\n"
					  << "  --frames,-f: Number of frames for video (default: 120)\n"
					  << "  --fps      : Frames per second for video (default: 30)\n"
					  << "  --speed    : Camera movement speed multiplier for video (default: 1.0)\n"
					  << "  --camera-path,-p: Camera animation path (orbit|linear|figure8|spiral)\n"
					  << "  --video-preset ID: Sets --video plus scene_id/camera-path/frames/fps/speed\n"
					  << "               together from one of src/shared/video_preset.h's named bundles.\n"
					  << "               Accepts either short id or descriptive key:\n"
					  << "               V1/cornell-orbit, V2/teapot-spin, V3/one-weekend-flyby,\n"
					  << "               V4/next-week-finale, V5/glass-dragon-caustics,\n"
					  << "               V6/sponza-flythrough. Any of those five flags placed AFTER\n"
					  << "               --video-preset on the command line overrides just that one field.\n"
					  << "  --help,-h  : Show this help message\n"
					  << "  width      : Image width (default " << kDefaultWidth << ", square aspect)\n"
					  << "  spp        : Samples per pixel (default " << kDefaultSamplesPerPixel << ")\n"
					  << "  max_depth  : Max ray depth (default " << kDefaultMaxDepth << ")\n"
					  << "  scene_id   : Scene selector, category letter + number (e.g. \"A1\"=Cornell Box,\n"
					  << "               default " << kDefaultSceneId << " - see src/TheRestOfYourLife/scene_registry.h)\n"
					  << "  cam_x/y/z  : Camera position - if omitted, uses the selected scene's own\n"
					  << "               recommended camera (see src/TheRestOfYourLife/scene_registry.h),\n"
					  << "               not a single fixed default across every scene\n";
			return false;
		}
	}

	// Parse positional arguments: width spp depth scene_id cam_x cam_y cam_z
	// Kept as raw strings rather than blindly std::stod-ing every token like
	// the old flat-int scheme did - scene_id (position 3) is a category
	// letter + number now (e.g. "A2"), not a number, so each position is
	// parsed according to what it's actually expected to hold - see
	// scene_registry.h's SceneDescriptor::id comment for the id format.
	std::vector<std::string> positional_args;
	for (int i = 1; i < argc; ++i) {
		if (consumed_args.count(i) > 0) continue;
		const std::string arg = argv[i];
		if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') continue;
		positional_args.push_back(arg);
	}

	auto parse_positive_int = [](const std::string& s, int& out_value) {
		try {
			double v = std::stod(s);
			if (v > 0) out_value = static_cast<int>(v);
		} catch (const std::exception&) {}
	};

	if (positional_args.size() >= 1) parse_positive_int(positional_args[0], out.image_width);
	out.image_height = out.image_width;
	if (positional_args.size() >= 2) parse_positive_int(positional_args[1], out.samples_per_pixel);
	if (positional_args.size() >= 3) parse_positive_int(positional_args[2], out.max_ray_depth);

	if (positional_args.size() >= 4) {
		const std::string& id = positional_args[3];
		// Category letter + one or more digits (e.g. "A1", "B10").
		const bool valid = id.size() >= 2 && id[0] >= 'A' && id[0] <= 'Z' &&
			std::all_of(id.begin() + 1, id.end(),
				[](unsigned char c) { return std::isdigit(c) != 0; });
		if (valid) {
			out.scene_id = id;
		} else {
			std::cerr << "Invalid scene_id \"" << id
				<< "\" - expected a category letter followed by a number "
				   "(e.g. \"A1\"), see src/TheRestOfYourLife/scene_registry.h\n";
			return false;
		}
	}

	if (positional_args.size() >= 5) {
		try { out.cam_x = std::stod(positional_args[4]); out.cam_explicit = true; }
		catch (const std::exception&) { std::cerr << "Invalid cam_x value, using default\n"; }
	}
	if (positional_args.size() >= 6) {
		try { out.cam_y = std::stod(positional_args[5]); }
		catch (const std::exception&) { std::cerr << "Invalid cam_y value, using default\n"; }
	}
	if (positional_args.size() >= 7) {
		try { out.cam_z = std::stod(positional_args[6]); }
		catch (const std::exception&) { std::cerr << "Invalid cam_z value, using default\n"; }
	}

	return true;
}
