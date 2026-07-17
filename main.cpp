#include <cstdlib>
#include <iostream>

int main() {
    std::cout << "Launching raytracing_book executable..." << std::endl;
    // Adjust the path if your build output directory differs
    const char* exePath = "C:\\Users\\xinpe\\source\\repos\\ray_tracer\\x64\\Debug\\raytracing_book.exe";
    int rc = std::system(exePath);
    return rc;
}
