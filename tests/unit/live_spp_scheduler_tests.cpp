// Tests for the Live Preview samples-per-frame scheduler (src/shared/live_spp_scheduler.h).
#include <gtest/gtest.h>

#include "../../src/shared/live_spp_scheduler.h"

using live_preview::SppScheduler;

TEST(LiveSppSchedulerTest, StartsAtOneBatchAndStaysThereWhileTheCameraMoves) {
	SppScheduler s;
	EXPECT_EQ(s.batch(), 1);
	for (int i = 0; i < 50; ++i) {
		s.frameDone(/*cameraMoved=*/true, 10.0);
		EXPECT_EQ(s.batch(), 1);
	}
}

TEST(LiveSppSchedulerTest, GrowsByDoublingOnceStillAndStopsAtTheMaximum) {
	SppScheduler s;
	int last = 1;
	for (int i = 0; i < 40; ++i) {
		s.frameDone(false, 10.0 * s.batch());   // a frame costs 10 ms per batch: always fast enough
		EXPECT_GE(s.batch(), last);
		EXPECT_TRUE(s.batch() == last || s.batch() == last * 2) << "grows by doubling only";
		last = s.batch();
	}
	EXPECT_EQ(s.batch(), SppScheduler::kMaxBatch);
	EXPECT_EQ(s.batch() & (s.batch() - 1), 0) << "always a power of two";
}

TEST(LiveSppSchedulerTest, WaitsForAFewStillFramesBeforeGrowing) {
	SppScheduler s;
	for (int i = 0; i < SppScheduler::kStillFramesBeforeGrowing - 1; ++i) {
		s.frameDone(false, 5.0);
		EXPECT_EQ(s.batch(), 1);
	}
	s.frameDone(false, 5.0);
	EXPECT_EQ(s.batch(), 2);
}

TEST(LiveSppSchedulerTest, ADoublingThatWouldBeTooSlowIsNotTaken) {
	SppScheduler s;
	for (int i = 0; i < 20; ++i) s.frameDone(false, 70.0 * s.batch());   // one batch already costs 70 ms: two would be 140 ms, over the target
	EXPECT_EQ(s.batch(), 1);
	// 40 ms per batch: 2 batches (80 ms) fit, 4 (160 ms) do not.
	for (int i = 0; i < 20; ++i) s.frameDone(false, 40.0 * s.batch());
	EXPECT_EQ(s.batch(), 2);
}

TEST(LiveSppSchedulerTest, ASlowFrameShrinksTheBatchAndAMoveResetsIt) {
	SppScheduler s;
	for (int i = 0; i < 20; ++i) s.frameDone(false, 5.0 * s.batch());
	ASSERT_EQ(s.batch(), SppScheduler::kMaxBatch);
	s.frameDone(false, 400.0);   // e.g. another program took the GPU
	EXPECT_EQ(s.batch(), SppScheduler::kMaxBatch / 2);
	s.frameDone(true, 5.0);
	EXPECT_EQ(s.batch(), 1);
	// After a move it takes the still-frame wait again.
	s.frameDone(false, 5.0);
	EXPECT_EQ(s.batch(), 1);
}

TEST(LiveSppSchedulerTest, ResetGoesBackToOne) {
	SppScheduler s;
	for (int i = 0; i < 20; ++i) s.frameDone(false, 5.0 * s.batch());
	ASSERT_GT(s.batch(), 1);
	s.reset();
	EXPECT_EQ(s.batch(), 1);
}
