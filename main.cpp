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
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "cpu_renderer/cpu_interface.h"
#include "gpu/optix/optix_interface.h"
#include "src/external/image_writer.h"
#include "src/TheRestOfYourLife/error_codes.h"
#include "launcher/camera_path.h"
#include "launcher/launcher_args.h"   // Argument parsing

// Run a subprocess with the given argv directly via CreateProcess (no shell,
// so there's no cmd.exe quoting/percent-expansion to worry about - important
// since ffmpeg's "%04d" frame-pattern argument would otherwise be at the
// mercy of cmd.exe's own "%" parsing rules). Every argument is wrapped in
// quotes; safe here since none of the arguments we build contain literal
// quote characters. Returns the child's exit code, or -1 if it could not be
// launched (e.g. the executable isn't found on PATH).
static int run_subprocess(const std::vector<std::string>& argv) {
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
}

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
	int  image_width        = args.image_width;
	int  image_height       = args.image_height;
	int  samples_per_pixel  = args.samples_per_pixel;
	int  max_ray_depth      = args.max_ray_depth;
	int  scene_id           = args.scene_id;
	double cam_x            = args.cam_x;
	double cam_y            = args.cam_y;
	double cam_z            = args.cam_z;
	bool video_mode         = args.video_mode;
	int  video_frames       = args.video_frames;
	int  video_fps          = args.video_fps;
	double video_speed      = args.video_speed;
	std::string camera_path = args.camera_path;

    if (video_mode) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO GENERATION MODE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Frames: " << video_frames << std::endl;
        std::cout << "FPS: " << video_fps << std::endl;
        std::cout << "Speed: " << video_speed << "x" << std::endl;
        std::cout << "Camera path: " << camera_path << std::endl;
        std::cout << "Renderer: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    } else {
        std::cout << "\nLaunching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;
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
        std::cout << "Rendering " << video_frames << " frames..." << std::endl;

        auto video_start_time = std::chrono::high_resolution_clock::now();
        int successful_frames = 0;
        std::vector<std::string> frame_paths;

        // Render each frame with animated camera position
        for (int frame = 0; frame < video_frames; ++frame) {
            // Get camera position for this frame
            CameraPosition cam_pos = get_camera_position(camera_path, frame, video_frames, video_speed);

            // Generate frame filename (e.g., frame_0001.ppm)
            char frame_filename[256];
            std::snprintf(frame_filename, sizeof(frame_filename), "frame_%04d.ppm", frame);
            std::filesystem::path frame_path = frames_dir / frame_filename;

            // Progress indicator
            std::cout << "\n[" << (frame + 1) << "/" << video_frames << "] Rendering " 
                      << frame_filename << " (camera: " 
                      << std::fixed << std::setprecision(1)
                      << cam_pos.lookfrom_x << ", " 
                      << cam_pos.lookfrom_y << ", " 
                      << cam_pos.lookfrom_z << ")..." << std::flush;

            auto frame_start = std::chrono::high_resolution_clock::now();
            int render_result = -1;

            // Render frame with GPU or CPU
            if (use_gpu) {
                if (optix_is_available()) {
                    render_result = optix_render_main(
                        image_width,
                        image_height,
                        samples_per_pixel,
                        max_ray_depth,
                        frame_path.string().c_str(),
                        scene_id,
                        cam_pos.lookfrom_x,
                        cam_pos.lookfrom_y,
                        cam_pos.lookfrom_z
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
                    scene_id,
                    cam_pos.lookfrom_x,
                    cam_pos.lookfrom_y,
                    cam_pos.lookfrom_z
                );
            }

            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);

            if (render_result == SUCCESS) {
                successful_frames++;
                frame_paths.push_back(frame_path.string());
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
        std::cout << "Successfully rendered: " << successful_frames << "/" << video_frames << " frames" << std::endl;
        std::cout << "Rendering time: " << video_duration.count() << " seconds" << std::endl;

        if (successful_frames == 0) {
            std::cerr << "ERROR: No frames rendered successfully!" << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }
        if (successful_frames < video_frames) {
            std::cout << "WARNING: " << (video_frames - successful_frames) << " of " << video_frames
                       << " frames failed to render and will be skipped - the video will be shorter than requested."
                       << std::endl;
        }

        // Convert each successfully-rendered PPM frame to PNG, numbering the
        // PNGs contiguously (0000, 0001, ...) rather than reusing the frame's
        // original index - ffmpeg's sequential image2 demuxer stops at the
        // first gap in a numbered sequence, so a mid-run failure would
        // otherwise silently truncate the video at that point.
        std::cout << "Converting frames to PNG..." << std::endl;
        int converted = 0;
        for (size_t i = 0; i < frame_paths.size(); ++i) {
            char enc_filename[32];
            std::snprintf(enc_filename, sizeof(enc_filename), "enc_%04d.png", converted);
            std::filesystem::path png_p = frames_dir / enc_filename;
            if (convert_ppm_to_png(frame_paths[i].c_str(), png_p.string().c_str())) {
                ++converted;
            } else {
                std::cerr << "WARNING: PNG conversion failed for " << frame_paths[i] << std::endl;
            }
            if ((i + 1) % 10 == 0 || i == frame_paths.size() - 1) {
                std::cout << "  Progress: " << (i + 1) << "/" << frame_paths.size() << " frames converted" << std::endl;
            }
        }

        std::cout << "Rendered: " << successful_frames << " frames, converted: " << converted << " PNG files" << std::endl;

        if (converted == 0) {
            std::cerr << "ERROR: No frames converted to PNG successfully!" << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }

        // Assemble the PNG sequence into an MP4 with ffmpeg
        std::cout << "\n========================================" << std::endl;
        std::cout << "ASSEMBLING VIDEO WITH FFMPEG" << std::endl;
        std::cout << "========================================" << std::endl;

        std::string enc_pattern = (frames_dir / "enc_%04d.png").string();
        std::vector<std::string> ffmpeg_argv = {
            "ffmpeg", "-y",
            "-r", std::to_string(video_fps),
            "-i", enc_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path.string()
        };
        std::string manual_cmd = "ffmpeg -y -r " + std::to_string(video_fps) + " -i \"" + enc_pattern +
                                  "\" -c:v libx264 -pix_fmt yuv420p \"" + video_path.string() + "\"";

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

    // Start timing for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    int render_result = -1; // 0 = success, non-zero = error

    if (use_gpu) {
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
                scene_id,
                cam_x,
                cam_y,
                cam_z
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
        render_result = cpu_render_main(image_width, image_height, samples_per_pixel, max_ray_depth, out_path.c_str(), scene_id, cam_x, cam_y, cam_z);
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
