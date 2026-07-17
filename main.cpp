#include <cstdlib>
#include <iostream>
#include <string>

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "Launching raytracing_book executable..." << std::endl;
    // Adjust the path if your build output directory differs
    // Use Release build for better performance
    const char* exePath = "C:\\Users\\xinpe\\source\\repos\\ray_tracer\\x64\\Release\\raytracing_book.exe";

    // Hard-coded rendering settings: <image_width> <samples_per_pixel> <max_depth>
    const int hard_width = 1000;    // preview width
    const int hard_spp   = 1000;     // samples per pixel
    const int hard_depth = 20;      // max ray bounces

    std::string args = " " + std::to_string(hard_width) + " " + std::to_string(hard_spp) + " " + std::to_string(hard_depth);
    std::string cmd = std::string("\"") + exePath + "\"" + args;
    int rc = std::system(cmd.c_str());
    return rc;
}
