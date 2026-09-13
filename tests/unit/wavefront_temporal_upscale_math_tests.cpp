// wavefront_temporal_upscale_math_tests.cpp
// Validation for gpu/optix/wavefront_temporal_upscale_math.h's deterministic
// sub-pixel jitter sequence (Live Preview's temporal upscale feature - see
// this project's own plan). All CPU_GPU/plain-float functions, so no GPU
// hardware is required (mirrors wavefront_svgf_math_tests.cpp's own
// host-build-only approach).
//
// Tests:
// wf_temporal_upscale_jitter
//   1. Jitter stays within [0,1) for many sample indices
//   2. Jitter is periodic with period upscaleFactor*upscaleFactor
//   3. Jitter is consistent with wf_temporal_upscale_subcell() - the cell
//      floor(rx*U),floor(ry*U) recovers is exactly the cell subcell()
//      itself returns for the same sample index
// wf_temporal_upscale_subcell
//   4. Sub-cell coordinates stay within [0, upscaleFactor)
//   5. One full period visits every sub-cell EXACTLY once (an earlier,
//      quantized-Halton-sequence version of this file failed this exact
//      test - see wavefront_temporal_upscale_math.h's own header comment)
// Cross-check against qt_gui/camera_math.h's HAND-DUPLICATED copy
//   6. Both copies agree exactly on jitter (u,v) and sub-cell (cx,cy) for
//      every sample index across one full period, at both upscaleFactor
//      values the UI exposes (2 and 4) - guards against the two copies
//      drifting apart (see wavefront_temporal_upscale_math.h's own header
//      comment on why this hand-duplication exists at all).

#include <gtest/gtest.h>
#include "wavefront_temporal_upscale_math.h"
#include "../../qt_gui/camera_math.h"
#include <cmath>
#include <set>

TEST(WfTemporalUpscaleMath, JitterStaysInUnitRange) {
	for (unsigned int i = 0; i < 64; ++i) {
		for (int factor : {2, 4}) {
			float rx, ry;
			wf_temporal_upscale_jitter(i, factor, rx, ry);
			EXPECT_GE(rx, 0.0f);
			EXPECT_LT(rx, 1.0f);
			EXPECT_GE(ry, 0.0f);
			EXPECT_LT(ry, 1.0f);
		}
	}
}

TEST(WfTemporalUpscaleMath, JitterIsPeriodic) {
	for (int factor : {2, 4}) {
		const unsigned int period = static_cast<unsigned int>(factor * factor);
		for (unsigned int i = 0; i < period; ++i) {
			float rx0, ry0, rx1, ry1;
			wf_temporal_upscale_jitter(i, factor, rx0, ry0);
			wf_temporal_upscale_jitter(i + period, factor, rx1, ry1);
			EXPECT_NEAR(rx0, rx1, 1e-6f);
			EXPECT_NEAR(ry0, ry1, 1e-6f);
		}
	}
}

TEST(WfTemporalUpscaleMath, JitterIsConsistentWithSubcell) {
	for (int factor : {2, 4}) {
		const unsigned int period = static_cast<unsigned int>(factor * factor);
		for (unsigned int i = 0; i < period; ++i) {
			float rx, ry;
			wf_temporal_upscale_jitter(i, factor, rx, ry);
			int cx, cy;
			wf_temporal_upscale_subcell(i, factor, cx, cy);
			EXPECT_EQ(static_cast<int>(rx * static_cast<float>(factor)), cx)
				<< "i=" << i << " factor=" << factor;
			EXPECT_EQ(static_cast<int>(ry * static_cast<float>(factor)), cy)
				<< "i=" << i << " factor=" << factor;
		}
	}
}

TEST(WfTemporalUpscaleMath, SubcellStaysInBounds) {
	for (unsigned int i = 0; i < 64; ++i) {
		for (int factor : {2, 4}) {
			int cx, cy;
			wf_temporal_upscale_subcell(i, factor, cx, cy);
			EXPECT_GE(cx, 0);
			EXPECT_LT(cx, factor);
			EXPECT_GE(cy, 0);
			EXPECT_LT(cy, factor);
		}
	}
}

TEST(WfTemporalUpscaleMath, OnePeriodVisitsEverySubcell) {
	for (int factor : {2, 4}) {
		const unsigned int period = static_cast<unsigned int>(factor * factor);
		std::set<std::pair<int, int>> seen;
		for (unsigned int i = 0; i < period; ++i) {
			int cx, cy;
			wf_temporal_upscale_subcell(i, factor, cx, cy);
			seen.insert({cx, cy});
		}
		EXPECT_EQ(seen.size(), static_cast<size_t>(factor * factor))
			<< "factor=" << factor << " did not visit every sub-cell over one full period";
	}
}

TEST(WfTemporalUpscaleMath, MatchesHandDuplicatedCameraMathCopy) {
	for (int factor : {2, 4}) {
		const unsigned int period = static_cast<unsigned int>(factor * factor);
		for (unsigned int i = 0; i < period; ++i) {
			float gpuRx, gpuRy;
			wf_temporal_upscale_jitter(i, factor, gpuRx, gpuRy);
			double qtRx, qtRy;
			camera_math::temporalUpscaleJitter(i, factor, qtRx, qtRy);
			EXPECT_NEAR(static_cast<double>(gpuRx), qtRx, 1e-6)
				<< "i=" << i << " factor=" << factor << " (rx)";
			EXPECT_NEAR(static_cast<double>(gpuRy), qtRy, 1e-6)
				<< "i=" << i << " factor=" << factor << " (ry)";

			int gpuCx, gpuCy, qtCx, qtCy;
			wf_temporal_upscale_subcell(i, factor, gpuCx, gpuCy);
			camera_math::temporalUpscaleSubcell(i, factor, qtCx, qtCy);
			EXPECT_EQ(gpuCx, qtCx) << "i=" << i << " factor=" << factor << " (cx)";
			EXPECT_EQ(gpuCy, qtCy) << "i=" << i << " factor=" << factor << " (cy)";
		}
	}
}
