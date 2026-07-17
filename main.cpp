#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem>
#include "gpu/cuda/cuda_interface.h"

int main(int /*argc*/, char** /*argv*/) {
    // Try to prefer a CUDA renderer executable if present (so the launcher behaves as before but uses GPU when available)
    std::cout << "Launching renderer (prefers GPU if available)..." << std::endl;

    // Hard-coded rendering settings: <image_width> <samples_per_pixel> <max_depth>
    const int hard_width = 500;    // preview width
    const int hard_height = hard_width * 9 / 16; // default 16:9 aspect
    const int hard_spp   = 500;    // samples per pixel
    const int hard_depth = 20;     // max ray bounces

    std::filesystem::path gpuCandidate = std::filesystem::current_path() / "gpu" / "cuda" / "cuda_renderer.exe";
    std::filesystem::path cpuCandidate = std::filesystem::current_path() / "x64" / "Release" / "raytracing_book.exe";

    // Also try repository-absolute path as a fallback
    std::filesystem::path repoRootCpu = "C:/Users/xinpe/source/repos/ray_tracer/x64/Release/raytracing_book.exe";
    std::filesystem::path repoRootGpu = "C:/Users/xinpe/source/repos/ray_tracer/gpu/cuda/cuda_renderer.exe";

    std::string exePathStr;
    if (std::filesystem::exists(gpuCandidate)) {
        exePathStr = gpuCandidate.string();
        std::cout << "Using GPU renderer: " << exePathStr << std::endl;
    } else if (std::filesystem::exists(repoRootGpu)) {
        exePathStr = repoRootGpu.string();
        std::cout << "Using GPU renderer: " << exePathStr << std::endl;
    } else if (std::filesystem::exists(cpuCandidate)) {
        exePathStr = cpuCandidate.string();
        std::cout << "GPU not found, using CPU renderer: " << exePathStr << std::endl;
    } else if (std::filesystem::exists(repoRootCpu)) {
        exePathStr = repoRootCpu.string();
        std::cout << "GPU not found, using CPU renderer: " << exePathStr << std::endl;
    } else {
        // Last resort: try the cpuCandidate path string even if missing (so error from system shows)
        exePathStr = cpuCandidate.string();
        std::cout << "Renderer executable not found in known locations. Will attempt: " << exePathStr << std::endl;
    }

    // First try in-process GPU render if the CUDA interface is available (linked)
    std::string out_path = "image.ppm";
    if (const char* od = std::getenv("OneDrive")) out_path = std::string(od) + "\\Desktop\\image.ppm";
    else if (const char* up = std::getenv("USERPROFILE")) out_path = std::string(up) + "\\Desktop\\image.ppm";

    std::cout << "Using command-line settings: width=" << hard_width << " height=" << hard_height << " spp=" << hard_spp << " max_depth=" << hard_depth << std::endl;

    // Try GPU in-process call
    bool usedGpuInProcess = false;
    int gpuRc = -1;
    try {
        gpuRc = gpu_render_main(hard_width, hard_height, hard_spp, hard_depth, out_path.c_str());
        if (gpuRc == 0) {
            std::cout << "Rendered with in-process GPU renderer, output: " << out_path << std::endl;
            return 0;
        } else {
            std::cout << "In-process GPU renderer returned non-zero (" << gpuRc << "). Falling back to external renderer." << std::endl;
        }
    } catch (...) {
        std::cout << "In-process GPU renderer not available (not linked). Falling back to external renderer." << std::endl;
    }

    std::string args = " " + std::to_string(hard_width) + " " + std::to_string(hard_spp) + " " + std::to_string(hard_depth);
    std::string cmd = std::string("\"") + exePathStr + "\"" + args;
    std::cout << "Executing external: " << cmd << std::endl;

    int rc = std::system(cmd.c_str());
    return rc;
}
