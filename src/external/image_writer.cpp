#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image_writer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

bool write_png(const char* filename, int width, int height, const unsigned char* rgb_data) {
	// stbi_write_png expects RGB data, 3 bytes per pixel
	int result = stbi_write_png(filename, width, height, 3, rgb_data, width * 3);
	return result != 0;
}

bool write_bmp(const char* filename, int width, int height, const unsigned char* rgb_data) {
	// stbi_write_bmp expects RGB data, 3 bytes per pixel
	int result = stbi_write_bmp(filename, width, height, 3, rgb_data);
	return result != 0;
}

std::vector<unsigned char> read_ppm(const char* filename, int& width, int& height) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open PPM file: " << filename << std::endl;
		return {};
	}

	std::string magic;
	file >> magic;

	if (magic != "P3" && magic != "P6") {
		std::cerr << "Unsupported PPM format: " << magic << std::endl;
		return {};
	}

	// Read width and height (skip comments)
	std::string line;
	width = height = 0;
	while (width == 0 || height == 0) {
		if (!std::getline(file, line)) {
			std::cerr << "Failed to read PPM dimensions" << std::endl;
			return {};
		}
		if (line.empty() || line[0] == '#') continue;
		std::istringstream iss(line);
		if (!(iss >> width >> height)) {
			// Try reading one at a time if both aren't on the same line
			width = 0;
			height = 0;
		}
	}

	int max_val = 0;
	while (max_val == 0) {
		if (!std::getline(file, line)) {
			std::cerr << "Failed to read PPM max value" << std::endl;
			return {};
		}
		if (line.empty() || line[0] == '#') continue;
		std::istringstream iss(line);
		iss >> max_val;
	}

	if (max_val != 255) {
		std::cerr << "Unsupported max value: " << max_val << std::endl;
		return {};
	}

	std::vector<unsigned char> pixels(width * height * 3);

	if (magic == "P3") {
		// ASCII format
		for (int i = 0; i < width * height * 3; i++) {
			int val;
			if (!(file >> val)) {
				std::cerr << "Failed to read pixel value at index " <<  i << std::endl;
				return {};
			}
			pixels[i] = static_cast<unsigned char>(val);
		}
	} else {
		// Binary format (P6)
		file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
		if (!file) {
			std::cerr << "Failed to read PPM binary pixel data" << std::endl;
			return {};
		}
	}

	return pixels;
}

bool convert_ppm_to_png(const char* ppm_path, const char* png_path) {
	int width, height;
	auto pixels = read_ppm(ppm_path, width, height);

	if (pixels.empty()) {
		return false;
	}

	return write_png(png_path, width, height, pixels.data());
}

bool convert_ppm_to_bmp(const char* ppm_path, const char* bmp_path) {
	int width, height;
	auto pixels = read_ppm(ppm_path, width, height);

	if (pixels.empty()) {
		return false;
	}

	return write_bmp(bmp_path, width, height, pixels.data());
}
