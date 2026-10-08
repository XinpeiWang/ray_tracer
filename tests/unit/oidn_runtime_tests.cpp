/**
 * @file oidn_runtime_tests.cpp
 * @brief Intel Open Image Denoise loaded at run time (src/shared/oidn_runtime.h): a missing library is reported, not fatal; with the library present (RT_OIDN_DIR,
 *        or the per-user folder) a noisy image gets clearly closer to the clean one
 */

#include <gtest/gtest.h>

#include "../../src/shared/oidn_runtime.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

TEST(OidnRuntimeTest, AMissingLibraryIsReportedNotFatal) {
	// Only meaningful when no OIDN is installed here; with one, the next test covers the real thing.
	if (oidn_runtime::available()) GTEST_SKIP() << "an Open Image Denoise library is installed: " << oidn_runtime::libraryPath();
	std::vector<float> img(16 * 16 * 3, 0.5f);
	const std::vector<float> before = img;
	std::string error;
	EXPECT_FALSE(oidn_runtime::denoiseHdr(img.data(), 16, 16, 3, 1.0f, error));
	EXPECT_FALSE(error.empty());
	EXPECT_EQ(img, before) << "the pixels are left alone";
	EXPECT_FALSE(oidn_runtime::unavailableReason().empty());
}

TEST(OidnRuntimeTest, BadArgumentsAreRefused) {
	std::string error;
	EXPECT_FALSE(oidn_runtime::denoiseHdr(nullptr, 16, 16, 3, 1.0f, error));
	std::vector<float> img(16 * 16 * 3, 0.5f);
	EXPECT_FALSE(oidn_runtime::denoiseHdr(img.data(), 0, 16, 3, 1.0f, error));
	EXPECT_FALSE(oidn_runtime::denoiseHdr(img.data(), 16, 16, 5, 1.0f, error));
}

TEST(OidnRuntimeTest, ANoisyGradientGetsMuchCloserToTheCleanOne) {
	if (!oidn_runtime::available()) GTEST_SKIP() << "no Open Image Denoise library (set RT_OIDN_DIR to run this): " << oidn_runtime::unavailableReason();
	const int w = 96, h = 64;
	std::vector<float> clean(w * h * 4), noisy;
	std::mt19937 rng(7);
	std::normal_distribution<float> gauss(0.0f, 0.35f);
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			const float base = 0.2f + 0.8f * x / (w - 1);   // a smooth ramp, brighter to the right
			const size_t i = (static_cast<size_t>(y) * w + x) * 4;
			clean[i] = base; clean[i + 1] = base * 0.7f; clean[i + 2] = base * 0.4f; clean[i + 3] = 1.0f;
		}
	noisy = clean;
	for (size_t i = 0; i < noisy.size(); i += 4)
		for (int c = 0; c < 3; ++c) noisy[i + c] = std::max(0.0f, noisy[i + c] + gauss(rng));
	auto mse = [&](const std::vector<float>& a) {
		double s = 0;
		for (size_t i = 0; i < a.size(); i += 4)
			for (int c = 0; c < 3; ++c) { const double d = a[i + c] - clean[i + c]; s += d * d; }
		return s / (static_cast<double>(w) * h * 3);
	};
	const double before = mse(noisy);
	std::vector<float> out = noisy;
	std::string error;
	ASSERT_TRUE(oidn_runtime::denoiseHdr(out.data(), w, h, 4, 1.0f, error)) << error;
	const double after = mse(out);
	EXPECT_LT(after, before * 0.35) << "the error to the clean image should drop to a fraction (" << before << " -> " << after << ")";
	for (size_t i = 3; i < out.size(); i += 4) ASSERT_EQ(out[i], noisy[i]) << "the alpha channel is not touched";

	std::vector<float> half = noisy;
	ASSERT_TRUE(oidn_runtime::denoiseHdr(half.data(), w, h, 4, 0.5f, error)) << error;
	const double mid = mse(half);
	EXPECT_GT(mid, after) << "a half blend sits between the original and the denoised image";
	EXPECT_LT(mid, before);
}
