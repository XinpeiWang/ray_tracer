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
#include "cpu_renderer/cpu_interface.h" // CPU renderer interface (multithreaded C++)
#include "gpu/optix/optix_interface.h"  // OptiX renderer interface (OptiX)
#include "src/external/image_writer.h"  // PPM to PNG conversion utilities
#include "src/TheRestOfYourLife/error_codes.h" // Centralized error code system

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
    std::string custom_output_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Force CPU rendering
        if (arg == "--cpu" || arg == "-cpu") {
            force_cpu = true;
            use_gpu = false;
        }
        // Force GPU rendering
        else if (arg == "--gpu" || arg == "-gpu") {
            use_gpu = true;
            force_cpu = false;
        }
        // Custom output path
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            custom_output_path = argv[i + 1];
            i++; // Skip next argument as it's the output path
        }
        // Help message
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--cpu|--gpu] [--output PATH] [width] [spp] [max_depth] [scene_id] [cam_x] [cam_y] [cam_z]\n";
            std::cout << "  --cpu      : Force CPU rendering\n";
            std::cout << "  --gpu      : Force GPU rendering (default)\n";
            std::cout << "  --output,-o: Output file path (default: ./output/image.ppm)\n";
            std::cout << "  --help,-h  : Show this help message\n";
            std::cout << "  width      : Image width (default 600, square aspect)\n";
            std::cout << "  spp        : Samples per pixel (default 500)\n";
            std::cout << "  max_depth  : Max ray depth (default 20)\n";
            std::cout << "  scene_id   : Scene selector (0=Cornell Box, 1=Bouncing Spheres, etc., default 0)\n";
            std::cout << "  cam_x      : Camera X position (default 278)\n";
            std::cout << "  cam_y      : Camera Y position (default 278)\n";
            std::cout << "  cam_z      : Camera Z position (default -800)\n";
            std::cout << "\nCamera always looks at Cornell box center: (278, 278, 278)\n";
            return 0;
        }
    }

    // ========================================================================
    // Default Rendering Settings
    // ========================================================================
    // Cornell box is 555x555x555 units, so square aspect ratio is appropriate

    int hard_width = 600;    // Image width in pixels
    int hard_height = 600;   // Image height (1:1 aspect for Cornell box)
    int hard_spp   = 500;    // Samples per pixel (anti-aliasing quality)
    int hard_depth = 20;     // Max ray bounce depth (lighting quality)
    int scene_id   = 0;      // Scene selector (0=Cornell Box, 1=Bouncing Spheres, etc.)

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
        std::cout << "  Resolution: " << hard_width << "x" << hard_height << std::endl;
        std::cout << "  Samples per pixel: " << hard_spp << std::endl;
        std::cout << "  Max ray depth: " << hard_depth << std::endl;
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
            std::cout << "Image width [" << hard_width << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_width = std::stoi(response); hard_height = hard_width; } catch(...) {}
            }

            // Samples per pixel
            std::cout << "Samples per pixel [" << hard_spp << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_spp = std::stoi(response); } catch(...) {}
            }

            // Max ray depth
            std::cout << "Max ray depth [" << hard_depth << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_depth = std::stoi(response); } catch(...) {}
            }
        }
    }

    std::cout << "\nLaunching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;

    // ========================================================================
    // Parse Numeric Positional Arguments
    // ========================================================================
    // Command-line format: width spp depth cam_x cam_y cam_z
    // All numeric arguments are collected into a vector for ordered parsing
    // Uses std::stod() for floating-point camera coordinate support

    std::vector<double> numeric_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // Skip flag arguments (those starting with '--')
        // But allow negative numbers like -800 (check if it's a number)
        if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') {
            continue; // Skip flags like --gpu, --cpu, --output
        }
        // Try to parse as number (including negative numbers like -800)
        try {
            numeric_args.push_back(std::stod(arg));
        } catch (...) {
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
        hard_width = (int)numeric_args[0];
        hard_height = (int)numeric_args[0]; // Keep square aspect ratio
    }
    if (numeric_args.size() >= 2 && numeric_args[1] > 0) {
        hard_spp = (int)numeric_args[1];
    }
    if (numeric_args.size() >= 3 && numeric_args[2] > 0) {
        hard_depth = (int)numeric_args[2];
    }
    if (numeric_args.size() >= 4 && numeric_args[3] >= 0) {
        scene_id = (int)numeric_args[3];
    }

    // ========================================================================
    // Camera Position Configuration
    // ========================================================================
    // Default camera position: (278, 278, -800) - outside front of Cornell box
    // lookat target is fixed at (278, 278, 278) in the renderer code
    // User can override position via command-line args

    double cam_x = 278.0;   // Default X: horizontally centered
    double cam_y = 278.0;   // Default Y: vertically centered
    double cam_z = -800.0;  // Default Z: outside the box looking in

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
    std::cout << "Using command-line settings: width=" << hard_width << " height=" << hard_height << " spp=" << hard_spp << " max_depth=" << hard_depth 
              << " scene_id=" << scene_id << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ")" << std::endl;
    std::cout << "Writing output to: " << out_path << std::endl;

    // ========================================================================
    // Render Execution
    // ========================================================================
    // Call either GPU or CPU renderer in-process (linked libraries)
    // Both renderers accept the same parameters including camera position
    // Camera always looks at Cornell box center (278, 278, 278) - fixed in renderer

    // Start timing for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    int render_result = -1; // 0 = success, non-zero = error

    if (use_gpu) {
        // GPU Renderer (OptiX)
        if (optix_is_available()) {
            std::cout << "[OptiX] OptiX is available!" << std::endl;
            std::cout << "Calling optix_render_main(...) in-process (OptiX)..." << std::endl;
            render_result = optix_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
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
        render_result = cpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str(), scene_id, cam_x, cam_y, cam_z);
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
