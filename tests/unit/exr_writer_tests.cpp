/**
 * @file exr_writer_tests.cpp
 * @brief Unit tests for src/shared/exr_writer.h - the thin SaveEXR wrapper
 *        shared by camera.h (CPU) and optix_interface.cpp (GPU). End-to-end
 *        coverage against real renders lives in tests/integration/
 *        render_tests.cpp; this pins the wrapper's own contract in
 *        isolation (round-trip fidelity, and the failure path).
 */

#include <gtest/gtest.h>

#include "../../src/shared/exr_writer.h"
#include "../../src/external/tinyexr.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

TEST(ExrWriterTest, RoundTripsPixelValuesWithinFloatPrecision) {
	const int width = 4, height = 3;
	std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
	for (size_t i = 0; i < pixels.size(); ++i)
		pixels[i] = static_cast<float>(i) * 0.1f;

	const std::string path = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") +
							  "/exr_writer_tests_roundtrip.exr";
	std::string error;
	ASSERT_TRUE(write_exr_image(path, pixels.data(), width, height, error)) << error;
	ASSERT_EQ(IsEXR(path.c_str()), TINYEXR_SUCCESS);

	float* loaded = nullptr;
	int loadedWidth = 0, loadedHeight = 0;
	const char* loadErr = nullptr;
	const int ret = LoadEXR(&loaded, &loadedWidth, &loadedHeight, path.c_str(), &loadErr);
	ASSERT_EQ(ret, TINYEXR_SUCCESS) << (loadErr ? loadErr : "unknown LoadEXR error");
	ASSERT_EQ(loadedWidth, width);
	ASSERT_EQ(loadedHeight, height);

	// LoadEXR always returns RGBA regardless of the source's channel count -
	// compare only the RGB channels SaveEXR(components=3) actually wrote.
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			for (int c = 0; c < 3; ++c) {
				const size_t srcIdx = (static_cast<size_t>(y) * width + x) * 3 + c;
				const size_t dstIdx = (static_cast<size_t>(y) * width + x) * 4 + c;
				EXPECT_NEAR(loaded[dstIdx], pixels[srcIdx], 1e-5f);
			}
		}
	}
	free(loaded);
	std::remove(path.c_str());
}

TEST(ExrWriterTest, FailsGracefullyOnAnUnwritablePath) {
	const std::vector<float> pixels(1 * 1 * 3, 0.0f);
	// A directory that doesn't exist - SaveEXR cannot create it.
	std::string error;
	EXPECT_FALSE(write_exr_image(
		"nonexistent_directory_xyz/unwritable.exr", pixels.data(), 1, 1, error));
	EXPECT_FALSE(error.empty()) << "a failed write should still explain why";
}
