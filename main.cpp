#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem>
#include "gpu/cuda/cuda_interface.h"

int main(int argc, char** argv) {
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
            std::cout << "Usage: " << argv[0] << " [--cpu|--gpu]\n";
            std::cout << "  --cpu  : Force CPU rendering\n";
            std::cout << "  --gpu  : Force GPU rendering (default)\n";
            std::cout << "  --help : Show this help message\n";
            return 0;
        }
    }

    std::cout << "Launching renderer (" << (use_gpu ? "GPU" : "CPU") << " mode)..." << std::endl;

    // Hard-coded rendering settings: <image_width> <samples_per_pixel> <max_depth>
    const int hard_width = 600;    // Cornell box is square, so use same width
    const int hard_height = 600;   // Cornell box should be 1:1 aspect ratio
    const int hard_spp   = 500;    // samples per pixel
    const int hard_depth = 20;     // max ray bounces

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
        // Use CPU renderer (external executable)
        std::filesystem::path cpuExePath = repoRoot / "x64" / "Release" / "raytracing_book.exe";
        if (!std::filesystem::exists(cpuExePath)) {
            std::cout << "CPU renderer not found at: " << cpuExePath << std::endl;
            return 1;
        }

        std::string args = " " + std::to_string(hard_width) + " " + std::to_string(hard_spp) + " " + std::to_string(hard_depth);
        std::string cmd = std::string("\"") + cpuExePath.string() + "\"" + args;
        std::cout << "Executing CPU renderer: " << cmd << std::endl;

        int rc = std::system(cmd.c_str());
        if (rc == 0) {
            std::cout << "CPU renderer completed successfully." << std::endl;
        }
        return rc;
    }
}
