#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include <string>
#include <vector>

// Write image in various formats
// RGB data should be in range [0, 255]
bool write_png(const char* filename, int width, int height, const unsigned char* rgb_data);
bool write_bmp(const char* filename, int width, int height, const unsigned char* rgb_data);

// Convert PPM file to PNG/BMP
bool convert_ppm_to_png(const char* ppm_path, const char* png_path);
bool convert_ppm_to_bmp(const char* ppm_path, const char* bmp_path);

// Read PPM and return RGB data
std::vector<unsigned char> read_ppm(const char* filename, int& width, int& height);

#endif // IMAGE_WRITER_H
