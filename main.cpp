#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "gpu/cuda/gpu_interface.h"
#include "cpu_renderer/cpu_interface.h"

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "RAY TRACER LAUNCHER (Unified GPU/CPU)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Parse command-line arguments for CPU/GPU switch
    bool use_gpu = true; // Default to GPU
    bool force_cpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu" || arg == "-cpu") {
            force_cpu = true;
            use_gpu = false;
        } else if (arg == "--gpu" || arg == "-gpu") {
            use_gpu = true;
            force_cpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--cpu|--gpu] [width] [spp] [max_depth]\n";
            std::cout << "  --cpu      : Force CPU rendering\n";
            std::cout << "  --gpu      : Force GPU rendering (default)\n";
            std::cout << "  --help     : Show this help message\n";
            std::cout << "  width      : Image width (default 600, square aspect)\n";
            std::cout << "  spp        : Samples per pixel (default 500)\n";
            std::cout << "  max_depth  : Max ray depth (default 20)\n";
            return 0;
        }
    }

    std::cout << "Launching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;

    // Default rendering settings
    int hard_width = 600;    // Cornell box is square, so use same width
    int hard_height = 600;   // Cornell box should be 1:1 aspect ratio
    int hard_spp   = 500;    // samples per pixel
    int hard_depth = 20;     // max ray bounces

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

    // Setup output path (repo-local)
    std::filesystem::path repoRoot = "C:/Users/xinpe/source/repos/ray_tracer";
    std::filesystem::path outPath = repoRoot / "x64" / "Release" / "image.ppm";

    // Ensure the directory exists
    try {
        if (!outPath.parent_path().empty() && !std::filesystem::exists(outPath.parent_path())) {
            std::filesystem::create_directories(outPath.parent_path());
        }
    } catch (...) {
        // Ignore directory creation errors
    }
    std::string out_path = outPath.string();

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

    // Use GPU or CPU based on flag
    if (use_gpu) {
        // Try GPU in-process call
        std::cout << "Calling gpu_render_main(...) in-process..." << std::endl;
        int gpuRc = gpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
        std::cout << "gpu_render_main returned: " << gpuRc << std::endl;
        if (gpuRc == 0) {
            std::cout << "Rendered with in-process GPU renderer, output: " << out_path << std::endl;
            return 0;
        } else {
            std::cout << "In-process GPU renderer failed." << std::endl;
            return gpuRc;
        }
    } else {
        // Use CPU renderer (in-process library call)
        std::cout << "Calling cpu_render_main(...) in-process..." << std::endl;
        int cpuRc = cpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
        std::cout << "cpu_render_main returned: " << cpuRc << std::endl;
        if (cpuRc == 0) {
            std::cout << "Rendered with in-process CPU renderer, output: " << out_path << std::endl;
            return 0;
        } else {
            std::cout << "In-process CPU renderer failed." << std::endl;
            return cpuRc;
        }
    }
}
