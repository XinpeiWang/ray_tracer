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
#include <set>
#include <filesystem>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "cpu_renderer/cpu_interface.h" // CPU renderer interface (multithreaded C++)
#include "gpu/optix/optix_interface.h"  // OptiX renderer interface (OptiX)
#include "src/external/image_writer.h"  // PPM to PNG conversion utilities
#include "src/TheRestOfYourLife/error_codes.h" // Centralized error code system
#include "launcher/camera_path.h"        // Camera animation paths for video generation

namespace {
	/// Default rendering parameters
	constexpr int kDefaultWidth = 600;
	constexpr int kDefaultHeight = 600;
	constexpr int kDefaultSamplesPerPixel = 500;
	constexpr int kDefaultMaxDepth = 20;
	constexpr int kDefaultSceneId = 0;  // Cornell Box

	/// Cornell box center coordinates
	constexpr double kCornellBoxCenter = 278.0;

	/// Default camera position (outside front of Cornell box)
	constexpr double kDefaultCameraX = 278.0;
	constexpr double kDefaultCameraY = 278.0;
	constexpr double kDefaultCameraZ = -800.0;  // Far back view for full scene

	/// Helper function to load PPM file into OpenCV Mat
	/// Supports both P3 (ASCII) and P6 (binary) PPM formats
	/// Returns empty Mat on failure
	cv::Mat load_ppm_to_mat(const std::string& filepath) {
		FILE* file = std::fopen(filepath.c_str(), "rb");
		if (!file) {
			std::cerr << "ERROR: Cannot open PPM file: " << filepath << std::endl;
			return cv::Mat();
		}

		// Read PPM header
		char format[3];
		int width, height, maxval;
		if (std::fscanf(file, "%2s\n%d %d\n%d\n", format, &width, &height, &maxval) != 4) {
			std::cerr << "ERROR: Invalid PPM header in: " << filepath << std::endl;
			std::fclose(file);
			return cv::Mat();
		}

		bool isP3 = (format[0] == 'P' && format[1] == '3');  // ASCII
		bool isP6 = (format[0] == 'P' && format[1] == '6');  // Binary

		if (!isP3 && !isP6) {
			std::cerr << "ERROR: Unsupported PPM format in: " << filepath << " (expected P3 or P6, got " << format << ")" << std::endl;
			std::fclose(file);
			return cv::Mat();
		}

		// Read pixel data (PPM is RGB, OpenCV uses BGR)
		cv::Mat img(height, width, CV_8UC3);

		if (isP3) {
			// ASCII format - read integers
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int r, g, b;
					if (std::fscanf(file, "%d %d %d", &r, &g, &b) != 3) {
						std::cerr << "ERROR: Incomplete pixel data in: " << filepath << std::endl;
						std::fclose(file);
						return cv::Mat();
					}
					// Convert RGB to BGR
					img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>(b), 
														static_cast<unsigned char>(g), 
														static_cast<unsigned char>(r));
				}
			}
		} else {
			// Binary format (P6)
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					unsigned char rgb[3];
					if (std::fread(rgb, 1, 3, file) != 3) {
						std::cerr << "ERROR: Incomplete pixel data in: " << filepath << std::endl;
						std::fclose(file);
						return cv::Mat();
					}
					// Convert RGB to BGR
					img.at<cv::Vec3b>(y, x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
				}
			}
		}

		std::fclose(file);
		return img;
	}
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "RAY TRACER LAUNCHER (Unified GPU/CPU)" << std::endl;
    std::cout << "========================================" << std::endl;

    // ========================================================================
    // Command-Line Argument Parsing
    // ========================================================================
    // Parse flags: --gpu, --cpu, --output, --help
    // Default to GPU mode (OptiX)

    bool use_gpu = true; // Default to GPU (OptiX)
    bool force_cpu = false;
    std::string custom_output_path;

    // Video generation parameters
    bool video_mode = false;
    int video_frames = 120;  // Default: 120 frames
    int video_fps = 30;      // Default: 30 FPS (4 second video)
    std::string camera_path = "orbit";  // Default path type

    // Track which argument indices have been consumed by flags
    std::set<int> consumed_args;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        // Force CPU rendering
        if (arg == "--cpu" || arg == "-cpu") {
            force_cpu = true;
            use_gpu = false;
            consumed_args.insert(i);
        }
        // Force GPU rendering
        else if (arg == "--gpu" || arg == "-gpu") {
            use_gpu = true;
            force_cpu = false;
            consumed_args.insert(i);
        }
        // Custom output path
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            custom_output_path = argv[i + 1];
            consumed_args.insert(i);
            consumed_args.insert(i + 1);
            ++i; // Skip next argument as it's the output path
        }
        // Video mode
        else if (arg == "--video") {
            video_mode = true;
            consumed_args.insert(i);
        }
        // Video frames
        else if ((arg == "--frames" || arg == "-f") && i + 1 < argc) {
            try {
                video_frames = std::stoi(argv[i + 1]);
                consumed_args.insert(i);
                consumed_args.insert(i + 1);
                ++i;
            } catch (const std::exception&) {
                std::cerr << "Invalid frame count, using default\n";
            }
        }
        // Video FPS
        else if (arg == "--fps" && i + 1 < argc) {
            try {
                video_fps = std::stoi(argv[i + 1]);
                consumed_args.insert(i);
                consumed_args.insert(i + 1);
                ++i;
            } catch (const std::exception&) {
                std::cerr << "Invalid FPS, using default\n";
            }
        }
        // Camera path type
        else if ((arg == "--camera-path" || arg == "-p") && i + 1 < argc) {
            camera_path = argv[i + 1];
            consumed_args.insert(i);
            consumed_args.insert(i + 1);
            ++i;
        }
        // Help message
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--cpu|--gpu] [--output PATH] [width] [spp] [max_depth] [scene_id] [cam_x] [cam_y] [cam_z]\n";
            std::cout << "  --cpu      : Force CPU rendering\n";
            std::cout << "  --gpu      : Force GPU rendering (default)\n";
            std::cout << "  --output,-o: Output file path (default: ./output/image.ppm)\n";
            std::cout << "  --video    : Enable video generation mode (renders multiple frames)\n";
            std::cout << "  --frames,-f: Number of frames for video (default: 120)\n";
            std::cout << "  --fps      : Frames per second for video (default: 30)\n";
            std::cout << "  --camera-path,-p: Camera animation path (orbit|linear|figure8|spiral, default: orbit)\n";
            std::cout << "  --help,-h  : Show this help message\n";
            std::cout << "  width      : Image width (default " << kDefaultWidth << ", square aspect)\n";
            std::cout << "  spp        : Samples per pixel (default " << kDefaultSamplesPerPixel << ")\n";
            std::cout << "  max_depth  : Max ray depth (default " << kDefaultMaxDepth << ")\n";
            std::cout << "  scene_id   : Scene selector (0=Cornell Box, 1=Bouncing Spheres, etc., default " << kDefaultSceneId << ")\n";
            std::cout << "  cam_x      : Camera X position (default " << kDefaultCameraX << ")\n";
            std::cout << "  cam_y      : Camera Y position (default " << kDefaultCameraY << ")\n";
            std::cout << "  cam_z      : Camera Z position (default " << kDefaultCameraZ << ")\n";
            std::cout << "\nCamera always looks at Cornell box center: (" << kCornellBoxCenter << ", " << kCornellBoxCenter << ", " << kCornellBoxCenter << ")\n";
            return EXIT_SUCCESS;
        }
    }

    // ========================================================================
    // Default Rendering Settings
    // ========================================================================
    // Cornell box is 555x555x555 units, so square aspect ratio is appropriate

    int image_width = kDefaultWidth;    // Image width in pixels
    int image_height = kDefaultHeight;  // Image height (1:1 aspect for Cornell box)
    int samples_per_pixel = kDefaultSamplesPerPixel;  // Samples per pixel (anti-aliasing quality)
    int max_ray_depth = kDefaultMaxDepth;  // Max ray bounce depth (lighting quality)
    int scene_id = kDefaultSceneId;  // Scene selector (0=Cornell Box, 1=Bouncing Spheres, etc.)

    // ========================================================================
    // Interactive Mode (if no arguments provided)
    // ========================================================================
    // Allows user to customize settings via prompts

    if (argc == 1) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "INTERACTIVE MODE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Current settings:" << std::endl;
        std::cout << "  Renderer: " << (use_gpu ? "GPU" : "CPU") << std::endl;
        std::cout << "  Resolution: " << image_width << "x" << image_height << std::endl;
        std::cout << "  Samples per pixel: " << samples_per_pixel << std::endl;
        std::cout << "  Max ray depth: " << max_ray_depth << std::endl;
        std::cout << "\nPress ENTER to render with these settings, or type 'custom' to change: ";

        std::string response;
        std::getline(std::cin, response);

        // User typed 'custom' - enter customization prompts
        if (response == "custom" || response == "c") {

            // Renderer mode selection
            std::cout << "\nRenderer mode (gpu/cpu) [" << (use_gpu ? "gpu" : "cpu") << "]: ";
            std::getline(std::cin, response);
            if (response == "cpu" || response == "c") {
                use_gpu = false;
            } else if (response == "gpu" || response == "g" || response.empty()) {
                use_gpu = true;
            }

            // Resolution (square aspect ratio maintained)
            std::cout << "Image width [" << image_width << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try {
                    image_width = std::stoi(response);
                    image_height = image_width; // Maintain square aspect
                } catch (const std::exception&) {
                    std::cerr << "Invalid width, using default\n";
                }
            }

            // Samples per pixel
            std::cout << "Samples per pixel [" << samples_per_pixel << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try {
                    samples_per_pixel = std::stoi(response);
                } catch (const std::exception&) {
                    std::cerr << "Invalid sample count, using default\n";
                }
            }

            // Max ray depth
            std::cout << "Max ray depth [" << max_ray_depth << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try {
                    max_ray_depth = std::stoi(response);
                } catch (const std::exception&) {
                    std::cerr << "Invalid depth, using default\n";
                }
            }
        }
    }

    if (video_mode) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO GENERATION MODE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Frames: " << video_frames << std::endl;
        std::cout << "FPS: " << video_fps << std::endl;
        std::cout << "Camera path: " << camera_path << std::endl;
        std::cout << "Renderer: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    } else {
        std::cout << "\nLaunching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;
    }

    // ========================================================================
    // Parse Numeric Positional Arguments
    // ========================================================================
    // Command-line format: width spp depth scene_id cam_x cam_y cam_z
    // All numeric arguments are collected into a vector for ordered parsing
    // Uses std::stod() for floating-point camera coordinate support
    // Skip arguments that were already consumed by flags

    std::vector<double> numeric_args;
    for (int i = 1; i < argc; ++i) {
        // Skip arguments that were consumed by flags
        if (consumed_args.count(i) > 0) {
            continue;
        }

        const std::string arg = argv[i];
        // Skip flag arguments (those starting with '--')
        // But allow negative numbers like -800 (check if it's a number)
        if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') {
            continue; // Skip flags like --gpu, --cpu, --output
        }
        // Try to parse as number (including negative numbers like -800)
        try {
            numeric_args.push_back(std::stod(arg));
        } catch (const std::exception&) {
            // Ignore non-numeric args (e.g., output path after --output)
        }
    }

    // Apply numeric arguments in order:
    // [0] = width (also sets height for square aspect)
    // [1] = samples per pixel
    // [2] = max ray depth
    // [3] = scene ID
    // [4] = camera X position
    // [5] = camera Y position
    // [6] = camera Z position

    if (numeric_args.size() >= 1 && numeric_args[0] > 0) {
        image_width = static_cast<int>(numeric_args[0]);
        image_height = static_cast<int>(numeric_args[0]); // Keep square aspect ratio
    }
    if (numeric_args.size() >= 2 && numeric_args[1] > 0) {
        samples_per_pixel = static_cast<int>(numeric_args[1]);
    }
    if (numeric_args.size() >= 3 && numeric_args[2] > 0) {
        max_ray_depth = static_cast<int>(numeric_args[2]);
    }
    if (numeric_args.size() >= 4 && numeric_args[3] >= 0) {
        scene_id = static_cast<int>(numeric_args[3]);
    }

    // ========================================================================
    // Camera Position Configuration
    // ========================================================================
    // Default camera position: (278, 278, -800) - outside front of Cornell box
    // lookat target is fixed at (278, 278, 278) in the renderer code
    // User can override position via command-line args

    double cam_x = kDefaultCameraX;
    double cam_y = kDefaultCameraY;
    double cam_z = kDefaultCameraZ;

    // Override with command-line camera coordinates if provided
    if (numeric_args.size() >= 7) {
        cam_x = numeric_args[4];
        cam_y = numeric_args[5];
        cam_z = numeric_args[6];
        std::cout << "[DEBUG] Parsed camera from args[4-6]: (" << cam_x << ", " << cam_y << ", " << cam_z << ")" << std::endl;
    } else {
        std::cout << "[DEBUG] Not enough args for camera, using defaults. numeric_args.size()=" << numeric_args.size() << std::endl;
    }

    // ========================================================================
    // Output Path Configuration
    // ========================================================================
    // Priority: command-line --output > default (./output/image.ppm)
    // Ensures output directory exists before rendering

    std::string out_path;

    if (!custom_output_path.empty()) {
        // Use custom output path from command line
        out_path = custom_output_path;
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
        std::cout << "VIDEO RENDERING WITH OPENCV" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare output directory for temporary frames
        std::filesystem::path out_path_obj(out_path);
        std::filesystem::path frames_dir = out_path_obj.parent_path() / "frames";
        std::filesystem::path video_path = out_path_obj.parent_path() / (out_path_obj.stem().string() + "_video.mp4");

        try {
            if (!std::filesystem::exists(frames_dir)) {
                std::filesystem::create_directories(frames_dir);
            }
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
            CameraPosition cam_pos = get_camera_position(camera_path, frame, video_frames);

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
        std::cout << "ASSEMBLING VIDEO WITH OPENCV" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Successfully rendered: " << successful_frames << "/" << video_frames << " frames" << std::endl;
        std::cout << "Rendering time: " << video_duration.count() << " seconds" << std::endl;

        if (successful_frames == 0) {
            std::cerr << "ERROR: No frames rendered successfully!" << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }

        // Initialize OpenCV VideoWriter
        // Codec: MP4V (MPEG-4), FourCC code
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter video_writer(
            video_path.string(),
            fourcc,
            static_cast<double>(video_fps),
            cv::Size(image_width, image_height),
            true  // isColor
        );

        if (!video_writer.isOpened()) {
            std::cerr << "ERROR: Failed to open video writer for: " << video_path << std::endl;
            return ERR_FILE_WRITE_FAILED;
        }

        std::cout << "Writing frames to video..." << std::endl;

        // Load and write each rendered frame
        for (size_t i = 0; i < frame_paths.size(); ++i) {
            cv::Mat frame = load_ppm_to_mat(frame_paths[i]);
            if (frame.empty()) {
                std::cerr << "\nERROR: Failed to load frame: " << frame_paths[i] << std::endl;
                video_writer.release();
                return ERR_FILE_WRITE_FAILED;
            }

            video_writer.write(frame);

            // Progress indicator
            if ((i + 1) % 10 == 0 || i == frame_paths.size() - 1) {
                std::cout << "  Progress: " << (i + 1) << "/" << frame_paths.size() << " frames written" << std::endl;
            }
        }

        video_writer.release();

        std::cout << "\n========================================" << std::endl;
        std::cout << "VIDEO COMPLETE!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Output: " << video_path << std::endl;
        std::cout << "Resolution: " << image_width << "x" << image_height << std::endl;
        std::cout << "FPS: " << video_fps << std::endl;
        std::cout << "Duration: " << (static_cast<double>(successful_frames) / video_fps) << " seconds" << std::endl;
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
