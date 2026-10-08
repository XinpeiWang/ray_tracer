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

TEST(OidnRuntimeTest, TheLibraryIsLookedForInTheFolderAWindowsReleaseUses) {
	// RT_OIDN_DIR may name the unpacked release itself: the DLL is in bin/ there, the dylib/so in lib/.
#ifdef _WIN32
	_putenv_s("RT_OIDN_DIR", "C:/oidn-test");
#else
	setenv("RT_OIDN_DIR", "/oidn-test", 1);
#endif
	const std::vector<std::string> c = oidn_runtime::detail::candidates();
	auto has = [&](const std::string& tail) {
		for (const std::string& p : c)
			if (p.size() >= tail.size() && p.compare(p.size() - tail.size(), tail.size(), tail) == 0 && p.find("oidn-test") != std::string::npos) return true;
		return false;
	};
	const std::string name = oidn_runtime::detail::libraryName();
	EXPECT_TRUE(has("oidn-test/" + name));
	EXPECT_TRUE(has("oidn-test/lib/" + name));
	EXPECT_TRUE(has("oidn-test/bin/" + name));
#ifdef _WIN32
	_putenv_s("RT_OIDN_DIR", "");
#else
	unsetenv("RT_OIDN_DIR");
#endif
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

// ---- Session: the denoiser kept open for Live Preview ----

TEST(OidnSessionTest, WithoutALibraryOpenFailsCleanlyAndRunRefuses) {
	if (oidn_runtime::available()) GTEST_SKIP() << "an Open Image Denoise library is installed: " << oidn_runtime::libraryPath();
	oidn_runtime::Session s;
	std::string error;
	EXPECT_FALSE(s.open(32, 32, oidn_runtime::Session::Quality::Fast, error));
	EXPECT_FALSE(error.empty());
	EXPECT_FALSE(s.isOpen());
	std::vector<float> in(32 * 32 * 3, 0.5f), out(in.size());
	EXPECT_FALSE(s.run(in.data(), out.data(), error));
}

TEST(OidnSessionTest, BadSizesAreRefusedAndRunNeedsAnOpenSession) {
	oidn_runtime::Session s;
	std::string error;
	EXPECT_FALSE(s.open(0, 10, oidn_runtime::Session::Quality::Fast, error));
	std::vector<float> in(3, 0.5f), out(3);
	EXPECT_FALSE(s.run(in.data(), out.data(), error));
	EXPECT_NE(error.find("not open"), std::string::npos);
}

TEST(OidnSessionTest, AnOpenSessionDenoisesRepeatedlyAndCanBeReopenedAtAnotherSize) {
	if (!oidn_runtime::available()) GTEST_SKIP() << "no Open Image Denoise library (set RT_OIDN_DIR to run this): " << oidn_runtime::unavailableReason();
	auto noisyRamp = [](int w, int h, std::vector<float>& clean, unsigned seed) {
		std::vector<float> noisy(static_cast<size_t>(w) * h * 3);
		clean.resize(noisy.size());
		std::mt19937 rng(seed);
		std::normal_distribution<float> gauss(0.0f, 0.35f);
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x)
				for (int c = 0; c < 3; ++c) {
					const float base = (0.2f + 0.8f * x / (w - 1)) * (1.0f - 0.2f * c);
					const size_t i = (static_cast<size_t>(y) * w + x) * 3 + c;
					clean[i] = base;
					noisy[i] = std::max(0.0f, base + gauss(rng));
				}
		return noisy;
	};
	auto mse = [](const std::vector<float>& a, const std::vector<float>& b) {
		double s = 0;
		for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
		return s / static_cast<double>(a.size());
	};
	oidn_runtime::Session s;
	std::string error;
	for (auto quality : {oidn_runtime::Session::Quality::Fast, oidn_runtime::Session::Quality::High}) {
		ASSERT_TRUE(s.open(96, 64, quality, error)) << error;
		EXPECT_TRUE(s.isOpen());
		EXPECT_EQ(s.width(), 96);
		for (unsigned frame = 0; frame < 3; ++frame) {   // the same session, different pictures
			std::vector<float> clean, out(96 * 64 * 3);
			const std::vector<float> noisy = noisyRamp(96, 64, clean, 10 + frame);
			ASSERT_TRUE(s.run(noisy.data(), out.data(), error)) << error;
			EXPECT_LT(mse(out, clean), mse(noisy, clean) * 0.35) << "frame " << frame;
		}
	}
	// A different size: reopen, and a non-finite or negative input value is treated as 0 instead of poisoning the picture.
	ASSERT_TRUE(s.open(48, 32, oidn_runtime::Session::Quality::Fast, error)) << error;
	EXPECT_EQ(s.height(), 32);
	std::vector<float> clean, out(48 * 32 * 3);
	std::vector<float> noisy = noisyRamp(48, 32, clean, 3);
	noisy[10] = std::nanf("");
	noisy[11] = -5.0f;
	ASSERT_TRUE(s.run(noisy.data(), out.data(), error)) << error;
	for (float v : out) ASSERT_TRUE(std::isfinite(v));
	s.close();
	EXPECT_FALSE(s.isOpen());
}
