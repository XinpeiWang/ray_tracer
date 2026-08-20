// ============================================================================
// Ray Tracer Unified Launcher
// ============================================================================
// This is the main entry point for the ray tracer executable (ray_tracer.exe)
// 
// Features:
//   - Detects CUDA GPU availability at runtime
//   - Routes to GPU or CPU renderer based on command-line flags or availability
//   - Accepts camera position parameters for configurable viewpoints
//   - Supports both interactive and command-line modes
//   - Converts output to multiple formats (PPM, PNG)
//
// Camera System:
//   - Camera position (lookfrom) is configurable via command-line args
//   - Camera target (lookat) is fixed at Cornell box center: (278, 278, 278)
//   - Default camera position: (278, 278, -800) - outside front of box
//
// Command-line format:
//   ray_tracer.exe [--gpu|--cpu] [--output path] [width] [spp] [depth] [cam_x] [cam_y] [cam_z]
//
// Example:
//   ray_tracer.exe --gpu 800 100 50 278 500 200
//   (GPU mode, 800x800 pixels, 100 samples/pixel, 50 ray depth, camera at (278,500,200))
// ============================================================================

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif
#include "cpu_renderer/cpu_interface.h"
#ifdef RT_HAVE_OPTIX
#include "gpu/optix/optix_interface.h"
#else
#include "launcher/optix_stub.h"
#endif
#include "src/external/image_writer.h"
#include "src/TheRestOfYourLife/error_codes.h"
#include "src/TheRestOfYourLife/thread_count.h"
#include "launcher/camera_path.h"
#include "launcher/launcher_args.h"   // Argument parsing
#include "launcher/diagnostics.h"     // --diagnose

// Run a subprocess with the given argv directly (no shell, so there's no
// cmd.exe/sh quoting or percent/glob expansion to worry about - important
// since ffmpeg's "%04d" frame-pattern argument would otherwise be at the
// mercy of the shell's own parsing rules). Returns the child's exit code, or
// -1 if it could not be launched (e.g. the executable isn't found on PATH).
//
// Windows: CreateProcess re-tokenizes a single command-line string, so every
// argument is wrapped in quotes (safe here since none of the arguments we
// build contain literal quote characters).
//
// POSIX (macOS/Linux): posix_spawnp takes argv[] directly with no
// re-parsing, so no quoting is needed at all - the same argv vector is
// passed through byte-for-byte. posix_spawnp (not fork+exec) is used
// deliberately: this process already runs a background thread
// (BackgroundPngConverter below), and fork() in a multithreaded process only
// clones the calling thread, risking the child deadlocking on a libc lock
// another thread held at fork time. No file_actions/chdir setup is needed -
// the child inherits fds 0/1/2 and the CWD by default, matching
// CreateProcess's behavior above. A signal-killed child is mapped to
// 128+signal so it can't collide with this function's own -1 "launch
// failed" sentinel.
static int run_subprocess(const std::vector<std::string>& argv) {
#ifdef _WIN32
    std::string cmdline;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) cmdline += ' ';
        cmdline += '"' + argv[i] + '"';
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    // CreateProcess may write into this buffer, so it can't be a literal/const string.
    std::vector<char> mutableCmd(cmdline.begin(), cmdline.end());
    mutableCmd.push_back('\0');

    BOOL ok = CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr, FALSE,
                              0, nullptr, nullptr, &si, &pi);
    if (!ok) {
        return -1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exitCode);
#else
    if (argv.empty()) return -1;

    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);

    pid_t pid = 0;
    int spawnRc = posix_spawnp(&pid, cargv[0], nullptr, nullptr, cargv.data(), environ);
    if (spawnRc != 0) {
        return -1;
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
#endif
}

// Converts rendered PPM frames to PNG on a background thread while the main
// thread renders the next frame, instead of converting everything in a
// separate pass after the whole video has finished rendering. Especially
// valuable in GPU mode, where the CPU otherwise sits idle during rendering -
// this overlap is "free" wall-clock time. Frames are renumbered contiguously
// (enc_0000.png, enc_0001.png, ...) in the order they're pushed, independent
// of the original frame index, so a mid-run render failure (which just skips
// pushing that frame) doesn't leave a gap in the sequence - ffmpeg's
// sequential image2 demuxer stops at the first gap in a numbered sequence.
class BackgroundPngConverter {
public:
    BackgroundPngConverter(std::filesystem::path frames_dir)
        : frames_dir_(std::move(frames_dir)), worker_(&BackgroundPngConverter::run, this) {}

    // Queue a successfully-rendered PPM for conversion. Safe to call from the
    // render loop while the worker thread is running.
    void push(std::string ppm_path) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(std::move(ppm_path));
        }
        cv_.notify_one();
    }

    // Signals no more frames are coming, waits for the queue to drain, and
    // returns how many frames converted successfully.
    int finish() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            done_ = true;
        }
        cv_.notify_one();
        worker_.join();
        return converted_;
    }

private:
    void run() {
        int next_index = 0;
        while (true) {
            std::string ppm_path;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&] { return !queue_.empty() || done_; });
                if (queue_.empty() && done_) break;
                ppm_path = std::move(queue_.front());
                queue_.pop();
            }

            char enc_filename[32];
            std::snprintf(enc_filename, sizeof(enc_filename), "enc_%04d.png", next_index);
            std::filesystem::path png_path = frames_dir_ / enc_filename;
            if (convert_ppm_to_png(ppm_path.c_str(), png_path.string().c_str())) {
                ++next_index;
                converted_ = next_index;
            } else {
                std::lock_guard<std::mutex> lock(mtx_);
                std::cerr << "\nWARNING: PNG conversion failed for " << ppm_path << std::endl;
            }
        }
    }

    std::filesystem::path frames_dir_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<std::string> queue_;
    bool done_ = false;
    int converted_ = 0;
    std::thread worker_;  // must be the last member so it's constructed last
};

int main(int argc, char** argv) {
	std::cout << "========================================" << std::endl;
	std::cout << "RAY TRACER LAUNCHER (Unified GPU/CPU)" << std::endl;
	std::cout << "========================================" << std::endl;

	// Parse command-line arguments
	LaunchArgs args;
	if (!parse_launch_args(argc, argv, args)) {
		return EXIT_SUCCESS;
	}

	// Unpack for readability in the rest of main
	bool use_gpu            = args.use_gpu;
#ifndef RT_HAVE_OPTIX
	// This build has no CUDA/OptiX SDK - GPU rendering was never compiled in
	// (see launcher/optix_stub.h). Deliberately applied here, after parsing,
	// rather than inside parse_launch_args() itself: that function's
	// LaunchArgs::use_gpu default (true) is covered by its own unit tests
	// (tests/unit/launcher_args_bdpt_mlt_tests.cpp) independent of platform,
	// so the platform-capability decision belongs at the call site, next to
	// where every other use_gpu-affecting concern in main is applied - not
	// baked into the parser itself. Force CPU rendering unconditionally
	// rather than letting the true default reach the optix_is_available()
	// check below and hard-fail every plain invocation; only warn when the
	// user actually typed --gpu, so a build known ahead of time to be
	// CPU-only degrades gracefully instead of refusing to render at all.
	if (use_gpu) {
		if (args.gpu_flag_explicit) {
			std::cerr << "Warning: --gpu was requested, but this build has no "
						 "GPU/OptiX support - rendering on CPU instead.\n";
		}
		use_gpu = false;
	}
#endif
	int  image_width        = args.image_width;
	int  image_height       = args.image_height;
	int  samples_per_pixel  = args.samples_per_pixel;
	int  max_ray_depth      = args.max_ray_depth;
	std::string scene_id    = args.scene_id;
	double cam_x            = args.cam_x;
	double cam_y            = args.cam_y;
	double cam_z            = args.cam_z;
	bool video_mode         = args.video_mode;
	int  video_frames       = args.video_frames;
	int  video_fps          = args.video_fps;
	double video_speed      = args.video_speed;
	std::string camera_path = args.camera_path;
	bool use_sppm           = args.use_sppm;
	bool use_bdpt           = args.use_bdpt;
	bool use_mlt            = args.use_mlt;

	// optix_render_main() reads this env var itself (gpu/optix/optix_interface.cpp)
	// to pick the wavefront GPU backend over the default recursive one - set it
	// once here so it's in effect for both single-image and per-frame video
	// renders below. Meaningless under CPU/SPPM, so only set for a plain GPU render.
	if (use_gpu && !use_sppm && args.use_wavefront) {
		_putenv_s("RAY_TRACER_WAVEFRONT", "1");
	}
	if (use_gpu && !use_sppm && args.optix_validate) {
		_putenv_s("RAY_TRACER_OPTIX_VALIDATION", "1");
	}

	// System-compatibility report instead of a render - see
	// launcher/diagnostics.h. Runs before any scene-loading or
	// output-directory side effects below, so this has no effect beyond
	// what it itself prints/writes.
	if (args.diagnose) {
		return run_diagnostics(args);
	}

	// SPPM has no defined per-frame semantics (its progressive radius state
	// doesn't reset cleanly frame-to-frame the way the path tracer's
	// stateless per-pixel sampling does) - reject the combination outright
	// rather than silently mis-rendering a video.
	if (use_sppm && video_mode) {
		std::cerr << ErrorInfo(ERR_INVALID_ARGUMENTS).to_string()
				  << " - --sppm and --video cannot be combined" << std::endl;
		return ERR_INVALID_ARGUMENTS;
	}

	// BDPT/MLT are each their own distinct render mode, same footing as
	// SPPM - reject any combination of two-or-more special modes outright
	// (which one would even "win" is ambiguous) rather than silently
	// picking one, mirroring --sppm/--video's own rejection above. Neither
	// has a defined per-frame video semantics either (a fresh independent
	// render per frame would technically work for either, unlike SPPM's
	// genuinely stateful radius, but wiring that per-frame path through
	// main.cpp's video loop is unimplemented scope here) - rejected for the
	// same "don't silently do something unexpected" reasoning.
	if ((use_bdpt && use_mlt) || (use_bdpt && use_sppm) || (use_mlt && use_sppm)) {
		std::cerr << ErrorInfo(ERR_INVALID_ARGUMENTS).to_string()
				  << " - --bdpt, --mlt, and --sppm are mutually exclusive render modes" << std::endl;
		return ERR_INVALID_ARGUMENTS;
	}
	if ((use_bdpt || use_mlt) && video_mode) {
		std::cerr << ErrorInfo(ERR_INVALID_ARGUMENTS).to_string()
				  << " - --bdpt/--mlt and --video cannot be combined" << std::endl;
		return ERR_INVALID_ARGUMENTS;
	}
	// --bdpt/--mlt have no GPU/OptiX implementation at all (CPU only - see
	// cpu_interface.h's cpu_render_main_bdpt()/cpu_render_main_mlt() doc
	// comments) - always render via the CPU path below regardless of
	// use_gpu, but only warn about it when the user actually typed --gpu
	// (use_gpu's own struct default is true, so warning unconditionally
	// would fire on the plain, common `--bdpt` invocation too).
	if ((use_bdpt || use_mlt) && args.gpu_flag_explicit) {
		std::cerr << "WARNING: --gpu is ignored under --bdpt/--mlt (CPU-only, no GPU/OptiX implementation exists) - rendering on CPU" << std::endl;
	}

    if (video_mode) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO GENERATION MODE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Frames: " << video_frames << std::endl;
        std::cout << "FPS: " << video_fps << std::endl;
        std::cout << "Speed: " << video_speed << "x" << std::endl;
        std::cout << "Camera path: " << camera_path << std::endl;
        std::cout << "Renderer: " << (use_gpu ? (args.use_wavefront ? "GPU (wavefront)" : "GPU") : "CPU") << std::endl;
    } else {
        std::cout << "\nLaunching renderer ("
                   << (use_bdpt ? "CPU BDPT"
                       : use_mlt ? "CPU MLT"
                       : use_sppm ? (use_gpu ? "GPU SPPM" : "CPU SPPM")
                                  : (use_gpu ? (args.use_wavefront ? "GPU wavefront" : "GPU") : "CPU"))
                   << " mode)..." << std::endl;
    }

    // ========================================================================
    // Output Path Configuration
    // ========================================================================
    // Priority: command-line --output > default (./output/image.ppm)
    // Ensures output directory exists before rendering

    std::string out_path;

    if (!args.custom_output_path.empty()) {
        // Use custom output path from command line
        out_path = args.custom_output_path;
    } else {
        // Default: <executable_directory>/output/image.ppm
        std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
        std::filesystem::path exe_dir = exe_path.parent_path();
        std::filesystem::path outPath = exe_dir / "output" / "image.ppm";
        out_path = outPath.string();
    }

    // Ensure the output directory exists (create if needed)
    try {
        std::filesystem::path out_path_obj(out_path);
        if (!out_path_obj.parent_path().empty() && !std::filesystem::exists(out_path_obj.parent_path())) {
            std::filesystem::create_directories(out_path_obj.parent_path());
        }
    } catch (...) {
        // Directory creation errors are non-fatal (will fail later if needed)
    }

    // Write a runtime marker file to verify executable is running correctly
    // This is useful for debugging build/deployment issues
    try {
        std::string marker_path = out_path + ".run_marker.txt";
        FILE* mf = std::fopen(marker_path.c_str(), "w");
        if (mf) {
            std::fprintf(mf, "launcher started\n");
            std::fflush(mf);
            std::fclose(mf);
        }
    } catch (...) {
        // Marker creation is optional, ignore errors
    }

    // Print configuration summary before rendering
    std::cout << "Using command-line settings: width=" << image_width << " height=" << image_height << " spp=" << samples_per_pixel << " max_depth=" << max_ray_depth 
              << " scene_id=" << scene_id << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ")" << std::endl;
    std::cout << "Writing output to: " << out_path << std::endl;

    // ========================================================================
    // Video Generation Mode
    // ========================================================================
    // If --video flag is set, render multiple frames with animated camera
    // Output video is saved directly using OpenCV VideoWriter

    if (video_mode) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO RENDERING" << std::endl;
        std::cout << "========================================" << std::endl;

        if (video_frames < 1) {
            std::cerr << "ERROR: --frames must be >= 1 (got " << video_frames << ")" << std::endl;
            return ERR_INVALID_ARGUMENTS;
        }
        if (video_speed <= 0.0) {
            std::cerr << "ERROR: --speed must be > 0 (got " << video_speed << ")" << std::endl;
            return ERR_INVALID_ARGUMENTS;
        }

        // --speed does not change the camera path itself (still always one
        // full baseline traversal - see camera_path.h) - it changes how many
        // frames that traversal is spread across, so a slower video takes
        // more frames (and more real time at the same fps) to cover the
        // exact same path, instead of covering less of the path in the same
        // frame count. --frames is the "1.0x" baseline frame count.
        int render_frame_count = static_cast<int>(std::llround(video_frames / video_speed));
        if (render_frame_count < 1) render_frame_count = 1;
        const int kMaxVideoFrames = 5000;
        if (render_frame_count > kMaxVideoFrames) {
            std::cout << "WARNING: --speed " << video_speed << " with --frames " << video_frames
                       << " would need " << render_frame_count << " frames; capping at "
                       << kMaxVideoFrames << " frames." << std::endl;
            render_frame_count = kMaxVideoFrames;
        }
        if (render_frame_count != video_frames) {
            std::cout << "Speed " << video_speed << "x expands " << video_frames << " base frames to "
                       << render_frame_count << " actual frames ("
                       << std::fixed << std::setprecision(1)
                       << (static_cast<double>(render_frame_count) / video_fps) << "s at " << video_fps << " fps)."
                       << std::endl;
        }

        // cam.render() (called once per CPU frame, inside cpu_render_main())
        // auto-detects a "free core" count by sampling system idle time over
        // 200ms - fine for a single image, but that's 200ms wasted on every
        // single video frame for an answer that isn't going to meaningfully
        // change between frames of the same render. Sample it once here and
        // pin it via RAY_TRACER_THREADS so every frame's render() call hits
        // the explicit-override fast path instead of re-sampling. Skipped
        // entirely if the user already set RAY_TRACER_THREADS themselves.
        if (!use_gpu && !std::getenv("RAY_TRACER_THREADS")) {
            unsigned int nthreads = determine_render_thread_count();
            std::string nthreads_str = std::to_string(nthreads);
            _putenv_s("RAY_TRACER_THREADS", nthreads_str.c_str());
            std::cout << "Pinned CPU thread count to " << nthreads << " for the whole video (skips per-frame detection)." << std::endl;
        }

        // Prepare output directory for temporary frames
        std::filesystem::path out_path_obj(out_path);
        std::filesystem::path frames_dir = out_path_obj.parent_path() / "frames";
        std::filesystem::path video_path = out_path_obj.parent_path() / (out_path_obj.stem().string() + "_video.mp4");

        try {
            // Clear any frames left over from a previous video render - otherwise
            // stale files can mix with (or mask) this run's sequence, especially
            // when this run has fewer frames than the last one.
            if (std::filesystem::exists(frames_dir)) {
                std::filesystem::remove_all(frames_dir);
            }
            std::filesystem::create_directories(frames_dir);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to create frames directory: " << e.what() << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }

        std::cout << "Temporary frame directory: " << frames_dir << std::endl;
        std::cout << "Output video: " << video_path << std::endl;
        std::cout << "Rendering " << render_frame_count << " frames..." << std::endl;

        // Scale and phase-align the camera-path animation to this scene's
        // actual recommended camera, rather than every video using the same
        // Cornell-Box-scale orbit (radius 800 around (278,278,278), starting
        // at a fixed angle unrelated to any scene's actual default view)
        // regardless of scene - e.g. scene 1's spheres are clustered within
        // roughly +-15 units of the origin, so an 800-unit orbit radius
        // would show nothing but a tiny distant speck. Frame 0 of orbit/
        // linear/spiral lands exactly on this recommended camera position -
        // see camera_path.h's get_camera_position() - so switching between
        // Image and Video preview starts from the same view.
        double path_lookfrom_x = 278.0, path_lookfrom_y = 278.0, path_lookfrom_z = -800.0;
        double path_lookat_x = 278.0, path_lookat_y = 278.0, path_lookat_z = 278.0;
        if (cpu_scene_recommended_camera(scene_id.c_str(), &path_lookfrom_x, &path_lookfrom_y, &path_lookfrom_z,
                                          &path_lookat_x, &path_lookat_y, &path_lookat_z)) {
            std::cout << "Camera path starts at (" << path_lookfrom_x << ", " << path_lookfrom_y << ", " << path_lookfrom_z
                       << ") looking at (" << path_lookat_x << ", " << path_lookat_y << ", " << path_lookat_z
                       << ") (scene " << scene_id << "'s recommended camera)." << std::endl;
        }

        // If the user explicitly set a camera position (e.g. via the GUI's
        // X/Y/Z spinboxes or "Distance from Center" control), start the path
        // there instead of the scene's generic recommended camera - the
        // look-at point stays the scene's own, so this still preserves
        // get_camera_position()'s scale/height/start_angle derivation, just
        // decomposed from the user's chosen viewpoint rather than the
        // recommended one.
        if (args.cam_explicit) {
            path_lookfrom_x = cam_x;
            path_lookfrom_y = cam_y;
            path_lookfrom_z = cam_z;
            std::cout << "Camera path overridden to start at (" << path_lookfrom_x << ", " << path_lookfrom_y
                       << ", " << path_lookfrom_z << ") (explicit camera position)." << std::endl;
        }

        auto video_start_time = std::chrono::high_resolution_clock::now();
        int successful_frames = 0;

        // Converts each frame to PNG on a background thread as soon as it's
        // rendered, instead of after the whole video finishes - overlaps
        // conversion with the next frame's render (particularly valuable in
        // GPU mode, where the CPU is otherwise idle while the GPU renders).
        BackgroundPngConverter png_converter(frames_dir);

        // Render each frame with animated camera position
        for (int frame = 0; frame < render_frame_count; ++frame) {
            // Get camera position for this frame
            CameraPosition cam_pos = get_camera_position(camera_path, frame, render_frame_count,
                                                            path_lookfrom_x, path_lookfrom_y, path_lookfrom_z,
                                                            path_lookat_x, path_lookat_y, path_lookat_z);

            // Generate frame filename (e.g., frame_0001.ppm)
            char frame_filename[256];
            std::snprintf(frame_filename, sizeof(frame_filename), "frame_%04d.ppm", frame);
            std::filesystem::path frame_path = frames_dir / frame_filename;

            // Progress indicator
            std::cout << "\n[" << (frame + 1) << "/" << render_frame_count << "] Rendering "
                      << frame_filename << " (camera: "
                      << std::fixed << std::setprecision(1)
                      << cam_pos.lookfrom_x << ", " 
                      << cam_pos.lookfrom_y << ", " 
                      << cam_pos.lookfrom_z << ")..." << std::flush;

            auto frame_start = std::chrono::high_resolution_clock::now();
            int render_result = -1;

            // Render frame with GPU or CPU. force_camera_override=1: some
            // scenes (currently 1 and 2) otherwise ignore cam_x/y/z and
            // always use their own fixed single-image lookfrom - a video
            // must honor its per-frame animated camera regardless, or it
            // would render the same static frame video_frames times.
            if (use_gpu) {
                if (optix_is_available()) {
                    render_result = optix_render_main(
                        image_width,
                        image_height,
                        samples_per_pixel,
                        max_ray_depth,
                        frame_path.string().c_str(),
                        scene_id.c_str(),
                        cam_pos.lookfrom_x,
                        cam_pos.lookfrom_y,
                        cam_pos.lookfrom_z,
                        1,  // force_camera_override
                        args.denoise ? 1 : 0
                    );
                } else {
                    std::cerr << "\nERROR: OptiX is not available!" << std::endl;
                    return ERR_GPU_NO_DEVICE;
                }
            } else {
                render_result = cpu_render_main(
                    image_width,
                    image_height,
                    samples_per_pixel,
                    max_ray_depth,
                    frame_path.string().c_str(),
                    scene_id.c_str(),
                    cam_pos.lookfrom_x,
                    cam_pos.lookfrom_y,
                    cam_pos.lookfrom_z,
                    1  // force_camera_override
                );
            }

            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);

            if (render_result == SUCCESS) {
                successful_frames++;
                png_converter.push(frame_path.string());
                std::cout << " ✓ (" << (frame_duration.count() / 1000.0) << "s)" << std::endl;
            } else {
                std::cerr << " ✗ FAILED (error code: " << render_result << ")" << std::endl;
            }
        }

        auto video_end_time = std::chrono::high_resolution_clock::now();
        auto video_duration = std::chrono::duration_cast<std::chrono::seconds>(video_end_time - video_start_time);

        std::cout << "\n========================================" << std::endl;
        std::cout << "CONVERTING FRAMES" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Successfully rendered: " << successful_frames << "/" << render_frame_count << " frames" << std::endl;
        std::cout << "Rendering time: " << video_duration.count() << " seconds" << std::endl;

        if (successful_frames == 0) {
            std::cerr << "ERROR: No frames rendered successfully!" << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }
        if (successful_frames < render_frame_count) {
            std::cout << "WARNING: " << (render_frame_count - successful_frames) << " of " << render_frame_count
                       << " frames failed to render and will be skipped - the video will be shorter than requested."
                       << std::endl;
        }

        // Most frames were already converted to PNG in the background while
        // later frames rendered (see png_converter.push() above) - this just
        // waits for the queue to drain (typically near-instant, since
        // conversion is far faster than a render) and gets the final count.
        std::cout << "Finishing PNG conversion..." << std::endl;
        int converted = png_converter.finish();

        std::cout << "Rendered: " << successful_frames << " frames, converted: " << converted << " PNG files" << std::endl;

        if (converted == 0) {
            std::cerr << "ERROR: No frames converted to PNG successfully!" << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }

        // Assemble the PNG sequence into an MP4 with ffmpeg
        std::cout << "\n========================================" << std::endl;
        std::cout << "ASSEMBLING VIDEO WITH FFMPEG" << std::endl;
        std::cout << "========================================" << std::endl;

        // -g (one keyframe per second) and -movflags +faststart: without
        // them libx264's default ~250-frame GOP can leave a short render
        // with just its very first frame as a keyframe, and the moov atom
        // (seek index) lands at the end of the file - together those made
        // seeking backward in the GUI's embedded QMediaPlayer preview
        // unreliable (seeking forward mostly worked by chance since it
        // could just keep decoding ahead to the target).
        std::string enc_pattern = (frames_dir / "enc_%04d.png").string();
        std::vector<std::string> ffmpeg_argv = {
            "ffmpeg", "-y",
            "-r", std::to_string(video_fps),
            "-i", enc_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-g", std::to_string(video_fps),
            "-movflags", "+faststart",
            video_path.string()
        };
        std::string manual_cmd = "ffmpeg -y -r " + std::to_string(video_fps) + " -i \"" + enc_pattern +
                                  "\" -c:v libx264 -pix_fmt yuv420p -g " + std::to_string(video_fps) +
                                  " -movflags +faststart \"" + video_path.string() + "\"";

        std::cout << "Running: " << manual_cmd << std::endl;
        int ffmpeg_result = run_subprocess(ffmpeg_argv);

        if (ffmpeg_result == -1) {
            std::cerr << "ERROR: Could not launch ffmpeg - is it installed and on PATH?" << std::endl;
            std::cerr << "Install ffmpeg (https://ffmpeg.org/download.html), add it to PATH, then assemble manually:" << std::endl;
            std::cerr << "  " << manual_cmd << std::endl;
            return ERR_VIDEO_ASSEMBLY_FAILED;
        }
        if (ffmpeg_result != 0) {
            std::cerr << "ERROR: ffmpeg exited with code " << ffmpeg_result << std::endl;
            std::cerr << "Rendered frames are still available in: " << frames_dir << std::endl;
            std::cerr << "You can retry assembly manually with:" << std::endl;
            std::cerr << "  " << manual_cmd << std::endl;
            return ERR_VIDEO_ASSEMBLY_FAILED;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO COMPLETE!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Output video: " << video_path << std::endl;
        std::cout << "Resolution: " << image_width << "x" << image_height << std::endl;
        std::cout << "========================================" << std::endl;

        return SUCCESS;
    }

    // ========================================================================
    // Single Frame Rendering Mode
    // ========================================================================
    // Standard single-image render (default behavior when --video is not set)

    // If the caller didn't explicitly pass cam_x/y/z, cam_x/y/z are just
    // sitting at LaunchArgs' generic Cornell-Box-scale struct defaults
    // (278,278,-800) - forcing that onto every scene regardless of its own
    // actual coordinate scale would misplace the camera exactly like a
    // stale leftover GUI spinbox value would (e.g. scene 1's spheres sit
    // within roughly +-15 units of the origin). Use the selected scene's
    // own recommended camera instead in that case. Either way,
    // force_camera_override=1 below makes sure the renderer actually uses
    // whichever of these two is now in cam_x/y/z, regardless of the
    // scene's CameraConfig.mode (Fixed scenes would otherwise silently
    // ignore it and use their own registry lookfrom unconditionally).
    if (!args.cam_explicit) {
        cpu_scene_recommended_camera(scene_id.c_str(), &cam_x, &cam_y, &cam_z, nullptr, nullptr, nullptr);
    }

    // Start timing for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    int render_result = -1; // 0 = success, non-zero = error

    if (use_bdpt) {
        // BDPT Renderer, CPU path (Bidirectional Path Tracing). Implemented
        // in cpu_renderer/cpu_interface_bdpt.cpp - checked before use_sppm/
        // use_gpu below since it's a distinct render mode, not a path-tracer
        // variant (same shape as use_sppm's own priority - see this file's
        // earlier mutual-exclusion validation, which already rejected
        // --bdpt combined with --sppm/--mlt/--video, and the gpu_flag_explicit
        // warning above, which already covered --bdpt --gpu).
        std::cout << "Calling cpu_render_main_bdpt(...) in-process..." << std::endl;
        render_result = cpu_render_main_bdpt(image_width, image_height, samples_per_pixel,
                                              args.bdpt_max_depth, out_path.c_str(),
                                              scene_id.c_str(), cam_x, cam_y, cam_z, 1);  // force_camera_override
        std::cout << "cpu_render_main_bdpt returned: " << render_result << std::endl;
        if (render_result == SUCCESS) {
            std::cout << "Rendered with BDPT renderer, output: " << out_path << std::endl;
        } else {
            ErrorInfo err(render_result);
            std::cerr << "\n" << std::string(60, '=') << std::endl;
            std::cerr << "BDPT RENDER FAILED" << std::endl;
            std::cerr << std::string(60, '=') << std::endl;
            std::cerr << err.to_string() << std::endl;
            std::cerr << std::string(60, '=') << "\n" << std::endl;
            return render_result;
        }
    } else if (use_mlt) {
        // MLT Renderer, CPU path (Metropolis Light Transport). Implemented
        // in cpu_renderer/cpu_interface_bdpt.cpp - same priority reasoning
        // as use_bdpt above.
        std::cout << "Calling cpu_render_main_mlt(...) in-process..." << std::endl;
        render_result = cpu_render_main_mlt(image_width, image_height, args.mlt_bootstrap,
                                             args.mlt_mutations, args.mlt_max_depth, out_path.c_str(),
                                             scene_id.c_str(), cam_x, cam_y, cam_z, 1);  // force_camera_override
        std::cout << "cpu_render_main_mlt returned: " << render_result << std::endl;
        if (render_result == SUCCESS) {
            std::cout << "Rendered with MLT renderer, output: " << out_path << std::endl;
        } else {
            ErrorInfo err(render_result);
            std::cerr << "\n" << std::string(60, '=') << std::endl;
            std::cerr << "MLT RENDER FAILED" << std::endl;
            std::cerr << std::string(60, '=') << std::endl;
            std::cerr << err.to_string() << std::endl;
            std::cerr << std::string(60, '=') << "\n" << std::endl;
            return render_result;
        }
    } else if (use_sppm && use_gpu) {
        // SPPM Renderer, GPU/OptiX path. Scope (which scenes are actually
        // supported) is determined dynamically from the built scene's real
        // materials/geometry/lights -- see gpu/optix/optix_interface.cpp's
        // sppm_gpu_unsupported_reason() for the exact, current rule set and
        // optix_render_main_sppm() returns a reason-specific
        // ERR_GPU_UNSUPPORTED_SCENE for anything still out of scope. Checked
        // before the CPU-only use_sppm branch below so --sppm --gpu actually
        // reaches the GPU path instead of silently falling back to CPU.
        if (!optix_is_available()) {
            std::cerr << "ERROR: OptiX is not available! (--sppm --gpu requires OptiX)" << std::endl;
            return ERR_GPU_NO_DEVICE;
        }
        std::cout << "Calling optix_render_main_sppm(...) in-process (OptiX)..." << std::endl;
        render_result = optix_render_main_sppm(image_width, image_height, args.sppm_iterations,
                                                args.sppm_photons, max_ray_depth, out_path.c_str(),
                                                scene_id.c_str(), cam_x, cam_y, cam_z, 1);  // force_camera_override
        std::cout << "optix_render_main_sppm returned: " << render_result << std::endl;
        if (render_result == SUCCESS) {
            std::cout << "Rendered with GPU SPPM renderer, output: " << out_path << std::endl;
        } else {
            ErrorInfo err(render_result);
            std::cerr << "\n" << std::string(60, '=') << std::endl;
            std::cerr << "GPU SPPM RENDER FAILED" << std::endl;
            std::cerr << std::string(60, '=') << std::endl;
            std::cerr << err.to_string() << std::endl;
            std::cerr << std::string(60, '=') << "\n" << std::endl;
            return render_result;
        }
    } else if (use_sppm) {
        // SPPM Renderer, CPU path (Stochastic Progressive Photon Mapping).
        // Implemented in cpu_renderer/cpu_interface.cpp - takes priority
        // over --gpu/--cpu since SPPM is a distinct render mode, not a
        // path-tracer variant.
        std::cout << "Calling cpu_render_main_sppm(...) in-process..." << std::endl;
        render_result = cpu_render_main_sppm(image_width, image_height, args.sppm_iterations,
                                              args.sppm_photons, max_ray_depth, out_path.c_str(),
                                              scene_id.c_str(), cam_x, cam_y, cam_z, 1);  // force_camera_override
        std::cout << "cpu_render_main_sppm returned: " << render_result << std::endl;
        if (render_result == SUCCESS) {
            std::cout << "Rendered with SPPM renderer, output: " << out_path << std::endl;
        } else {
            ErrorInfo err(render_result);
            std::cerr << "\n" << std::string(60, '=') << std::endl;
            std::cerr << "SPPM RENDER FAILED" << std::endl;
            std::cerr << std::string(60, '=') << std::endl;
            std::cerr << err.to_string() << std::endl;
            std::cerr << std::string(60, '=') << "\n" << std::endl;
            return render_result;
        }
    } else if (use_gpu) {
        // GPU Renderer (OptiX)
        if (optix_is_available()) {
            std::cout << "[OptiX] OptiX is available!" << std::endl;
            std::cout << "Calling optix_render_main(...) in-process (OptiX)..." << std::endl;
            std::cout << "[DEBUG] Camera: (" << cam_x << ", " << cam_y << ", " << cam_z << ")" << std::endl;
            render_result = optix_render_main(
                image_width,
                image_height,
                samples_per_pixel,
                max_ray_depth,
                out_path.c_str(),
                scene_id.c_str(),
                cam_x,
                cam_y,
                cam_z,
                1,  // force_camera_override - see the comment above this section
                args.denoise ? 1 : 0
            );
            std::cout << "optix_render_main returned: " << render_result << std::endl;
            if (render_result == SUCCESS) {
                std::cout << "Rendered with OptiX renderer, output: " << out_path << std::endl;
            } else {
                ErrorInfo err(render_result);
                std::cerr << "\n" << std::string(60, '=') << std::endl;
                std::cerr << "OptiX RENDER FAILED" << std::endl;
                std::cerr << std::string(60, '=') << std::endl;
                std::cerr << err.to_string() << std::endl;
                std::cerr << std::string(60, '=') << "\n" << std::endl;
                return render_result;
            }
        } else {
            std::cerr << "ERROR: OptiX is not available!" << std::endl;
            return ERR_GPU_NO_DEVICE;
        }
    } else {
        // CPU Renderer (multithreaded C++)
        // Implemented in cpu_renderer/cpu_interface.cpp
        std::cout << "Calling cpu_render_main(...) in-process..." << std::endl;
        render_result = cpu_render_main(image_width, image_height, samples_per_pixel, max_ray_depth, out_path.c_str(), scene_id.c_str(), cam_x, cam_y, cam_z, 1);  // force_camera_override
        std::cout << "cpu_render_main returned: " << render_result << std::endl;
        if (render_result == SUCCESS) {
            std::cout << "Rendered with in-process CPU renderer, output: " << out_path << std::endl;
        } else {
            ErrorInfo err(render_result);
            std::cerr << "\n" << std::string(60, '=') << std::endl;
            std::cerr << "CPU RENDER FAILED" << std::endl;
            std::cerr << std::string(60, '=') << std::endl;
            std::cerr << err.to_string() << std::endl;
            std::cerr << std::string(60, '=') << "\n" << std::endl;
            return render_result;
        }
    }

    // ========================================================================
    // Performance Reporting
    // ========================================================================
    // Calculate and display render time in appropriate units

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;

    std::cout << "\n========================================" << std::endl;
    std::cout << "RENDER TIME: ";
    if (seconds < 1.0) {
        std::cout << duration.count() << " ms" << std::endl;
    } else if (seconds < 60.0) {
        std::cout << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
    } else {
        int minutes = (int)(seconds / 60);
        double remainingSeconds = seconds - (minutes * 60);
        std::cout << minutes << " min " << std::fixed << std::setprecision(1) << remainingSeconds << " sec" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // ========================================================================
    // Format Conversion
    // ========================================================================
    // Convert PPM (raw format) to PNG (compressed, widely supported)
    // PNG is more convenient for viewing and sharing

    if (render_result == 0) {
        std::cout << "\nConverting to PNG format..." << std::endl;

        // Generate output path for PNG
        std::filesystem::path ppm_path_obj(out_path);
        std::filesystem::path png_path = ppm_path_obj.parent_path() / (ppm_path_obj.stem().string() + ".png");

        // Convert PPM to PNG using external image_writer library
        if (convert_ppm_to_png(out_path.c_str(), png_path.string().c_str())) {
            std::cout << "✓ PNG saved: " << png_path << std::endl;
        } else {
            std::cout << "✗ PNG conversion failed" << std::endl;
        }

        std::cout << "\nRender complete! You can now open:" << std::endl;
        std::cout << "  - " << png_path.filename() << " (PNG - lossless, widely supported)" << std::endl;
        std::cout << "  - " << ppm_path_obj.filename() << " (PPM - raw data)" << std::endl;
    }

    return render_result;
}
