#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include "gpu/cuda/gpu_interface.h"
#include "cpu_renderer/cpu_interface.h"
#include "src/external/image_writer.h"

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "RAY TRACER LAUNCHER (Unified GPU/CPU)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Detect GPU availability
    bool gpu_available = (gpu_is_available() == 1);
    if (gpu_available) {
        std::cout << "✓ CUDA-capable GPU detected" << std::endl;
    } else {
        std::cout << "⚠ No CUDA GPU detected - CPU mode only" << std::endl;
    }

    // Parse command-line arguments for CPU/GPU switch
    bool use_gpu = gpu_available; // Default to GPU if available
    bool force_cpu = false;
    std::string custom_output_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu" || arg == "-cpu") {
            force_cpu = true;
            use_gpu = false;
        } else if (arg == "--gpu" || arg == "-gpu") {
            if (!gpu_available) {
                std::cout << "ERROR: GPU mode requested but no CUDA GPU detected!" << std::endl;
                std::cout << "Falling back to CPU mode..." << std::endl;
                use_gpu = false;
            } else {
                use_gpu = true;
                force_cpu = false;
            }
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            custom_output_path = argv[i + 1];
            i++; // Skip next argument as it's the output path
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--cpu|--gpu] [--output PATH] [width] [spp] [max_depth]\n";
            std::cout << "  --cpu      : Force CPU rendering\n";
            std::cout << "  --gpu      : Force GPU rendering (default)\n";
            std::cout << "  --output,-o: Output file path (default: ./output/image.ppm)\n";
            std::cout << "  --help     : Show this help message\n";
            std::cout << "  width      : Image width (default 600, square aspect)\n";
            std::cout << "  spp        : Samples per pixel (default 500)\n";
            std::cout << "  max_depth  : Max ray depth (default 20)\n";
            return 0;
        }
    }

    // Default rendering settings
    int hard_width = 600;    // Cornell box is square, so use same width
    int hard_height = 600;   // Cornell box should be 1:1 aspect ratio
    int hard_spp   = 500;    // samples per pixel
    int hard_depth = 20;     // max ray bounces

    // If no arguments provided, enter interactive mode
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

        if (response == "custom" || response == "c") {
            // Mode selection
            if (gpu_available) {
                std::cout << "\nRenderer mode (gpu/cpu) [" << (use_gpu ? "gpu" : "cpu") << "]: ";
                std::getline(std::cin, response);
                if (response == "cpu" || response == "c") {
                    use_gpu = false;
                } else if (response == "gpu" || response == "g" || response.empty()) {
                    use_gpu = true;
                }
            }

            // Resolution
            std::cout << "Image width [" << hard_width << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_width = std::stoi(response); hard_height = hard_width; } catch(...) {}
            }

            // Samples
            std::cout << "Samples per pixel [" << hard_spp << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_spp = std::stoi(response); } catch(...) {}
            }

            // Depth
            std::cout << "Max ray depth [" << hard_depth << "]: ";
            std::getline(std::cin, response);
            if (!response.empty()) {
                try { hard_depth = std::stoi(response); } catch(...) {}
            }
        }
    }

    std::cout << "\nLaunching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;

    // Parse numeric arguments (after mode flags)
    std::vector<int> numeric_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] != '-') {
            try {
                numeric_args.push_back(std::stoi(arg));
            } catch (...) {
                // Ignore non-numeric args
            }
        }
    }

    // Apply numeric arguments in order: width, spp, max_depth
    if (numeric_args.size() >= 1 && numeric_args[0] > 0) {
        hard_width = numeric_args[0];
        hard_height = numeric_args[0]; // Keep square aspect ratio
    }
    if (numeric_args.size() >= 2 && numeric_args[1] > 0) {
        hard_spp = numeric_args[1];
    }
    if (numeric_args.size() >= 3 && numeric_args[2] > 0) {
        hard_depth = numeric_args[2];
    }

    // Setup output path (use executable directory for portability)
    std::string out_path;

    if (!custom_output_path.empty()) {
        // Use custom output path from command line
        out_path = custom_output_path;
    } else {
        // Default: use executable directory
        std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
        std::filesystem::path exe_dir = exe_path.parent_path();
        std::filesystem::path outPath = exe_dir / "output" / "image.ppm";
        out_path = outPath.string();
    }

    // Ensure the directory exists
    try {
        std::filesystem::path out_path_obj(out_path);
        if (!out_path_obj.parent_path().empty() && !std::filesystem::exists(out_path_obj.parent_path())) {
            std::filesystem::create_directories(out_path_obj.parent_path());
        }
    } catch (...) {
        // Ignore directory creation errors
    }

    // Write a simple runtime marker next to the output image to prove this binary is being executed
    try {
        std::string marker_path = out_path + ".run_marker.txt";
        FILE* mf = std::fopen(marker_path.c_str(), "w");
        if (mf) {
            std::fprintf(mf, "launcher started\n");
            std::fflush(mf);
            std::fclose(mf);
        }
    } catch (...) {
        // ignore
    }

    std::cout << "Using command-line settings: width=" << hard_width << " height=" << hard_height << " spp=" << hard_spp << " max_depth=" << hard_depth << std::endl;
    std::cout << "Writing output to: " << out_path << std::endl;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use GPU or CPU based on flag
    int render_result = -1;
    if (use_gpu) {
        // Try GPU in-process call
        std::cout << "Calling gpu_render_main(...) in-process..." << std::endl;
        render_result = gpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
        std::cout << "gpu_render_main returned: " << render_result << std::endl;
        if (render_result == 0) {
            std::cout << "Rendered with in-process GPU renderer, output: " << out_path << std::endl;
        } else {
            std::cout << "In-process GPU renderer failed." << std::endl;
            return render_result;
        }
    } else {
        // Use CPU renderer (in-process library call)
        std::cout << "Calling cpu_render_main(...) in-process..." << std::endl;
        render_result = cpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
        std::cout << "cpu_render_main returned: " << render_result << std::endl;
        if (render_result == 0) {
            std::cout << "Rendered with in-process CPU renderer, output: " << out_path << std::endl;
        } else {
            std::cout << "In-process CPU renderer failed." << std::endl;
            return render_result;
        }
    }

    // Calculate elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;

    // Print render time
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

    // Convert PPM to PNG and BMP
    if (render_result == 0) {
        std::cout << "\nConverting to PNG format..." << std::endl;

        // Generate output paths
        std::filesystem::path ppm_path_obj(out_path);
        std::filesystem::path png_path = ppm_path_obj.parent_path() / (ppm_path_obj.stem().string() + ".png");

        // Convert to PNG
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
