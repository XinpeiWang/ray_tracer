// ppm_test_utils.h -- Shared P3 (ASCII) PPM loading/analysis helpers for
// tests. Previously duplicated independently in tests/unit/gpu_render_tests.cpp
// and tests/integration/render_tests.cpp; hoisted here so both share one
// implementation. Header-only (inline) since it's included by multiple test
// translation units.
#pragma once

#include <fstream>
#include <string>
#include <vector>

struct PPMImage {
	int width = 0;
	int height = 0;
	std::vector<float> pixels;  // RGB floats, normalized to [0,1]
	bool valid = false;
};

/// Load a PPM (P3 ASCII) file and normalize pixel values to [0,1]
inline PPMImage load_ppm(const char* path) {
	PPMImage img;
	std::ifstream file(path);
	if (!file.good()) return img;

	std::string magic;
	int maxVal;
	file >> magic >> img.width >> img.height >> maxVal;

	if (magic != "P3" || maxVal <= 0) return img;

	int total = img.width * img.height * 3;
	img.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v;
		file >> v;
		img.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	img.valid = (file.good() || file.eof());
	return img;
}

/// Average brightness across all pixels (0=black, 1=white)
inline float average_brightness(const PPMImage& img) {
	if (img.pixels.empty()) return 0.0f;
	float sum = 0.0f;
	for (float v : img.pixels) sum += v;
	return sum / static_cast<float>(img.pixels.size());
}
