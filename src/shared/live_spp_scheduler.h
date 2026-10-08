#pragma once
// live_spp_scheduler.h -- how many batches of samples Live Preview renders per frame.
//
// While the camera is moving every frame must come back fast, so a frame is one batch (the "Samples/Frame" setting). Once the picture has been still for a
// few frames it is the noise that matters, not the latency, and each frame also costs a fixed amount besides the samples (accumulating, tone mapping,
// showing the picture), so a frame of 2, 4 or 8 batches converges faster per second than 8 frames of one batch. The scheduler doubles the batch count while
// the last frame was fast enough to stay well under the target frame time, halves it if a frame ran long, and drops back to one the moment the camera moves
// or the accumulation is reset. (Cycles' viewport does the same in spirit: a small, quick pass while navigating, bigger ones when idle.)
//
// std-only. The caller supplies the measured time of the render call and weights the running mean by `batch()`.

namespace live_preview {

class SppScheduler {
public:
	static constexpr int kMaxBatch = 8;
	static constexpr int kStillFramesBeforeGrowing = 3;   // lets the first frames after a move settle (reprojection) before spending more per frame
	static constexpr double kTargetFrameMs = 100.0;        // a frame of the next size is allowed if it should still finish within this
	static constexpr double kTooSlowFrameMs = 160.0;       // a frame longer than this shrinks the batch again

	// Batches of Samples/Frame to render in the next frame (1, 2, 4 or 8).
	int batch() const { return m_batch; }

	// Camera moved, accumulation was reset, or a setting changed: back to one batch.
	void reset() {
		m_batch = 1;
		m_stillFrames = 0;
	}

	// Call after every frame with whether the camera moved for it and how long the render call took.
	void frameDone(bool cameraMoved, double renderMs) {
		if (cameraMoved) {
			reset();
			return;
		}
		++m_stillFrames;
		if (renderMs > kTooSlowFrameMs && m_batch > 1) {
			m_batch /= 2;
			return;
		}
		if (m_stillFrames >= kStillFramesBeforeGrowing && m_batch < kMaxBatch && renderMs * 2.0 <= kTargetFrameMs) m_batch *= 2;
	}

private:
	int m_batch = 1;
	int m_stillFrames = 0;
};

}  // namespace live_preview
