#pragma once
// ============================================================================
// render_stats.h
// ============================================================================
// Minimal, opt-in ray/bounce counters for the CPU default path tracer - see
// launcher/main.cpp's own "[STATS]" block for how these get printed, and
// gpu/optix/wavefront_path_tracer.cpp's own "[WF-STATS]" block (gated by the
// same RAY_TRACER_STATS env var, following this project's established
// same-process env-var flag pattern - see RAY_TRACER_WAVEFRONT in main.cpp)
// for the GPU-wavefront equivalent. The recursive GPU backend has no
// per-ray counting infra yet - main.cpp prints only its deterministic
// primary-ray count (width*height*spp) for that backend.
//
// Not a general profiling framework - three counters, incremented with
// relaxed atomics from camera.h's multithreaded worker threads, gated behind
// enabled() so a render without --stats pays only one predictable, always-
// false branch per call site instead of the atomic increment itself.
// ============================================================================

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace render_stats {

// Cached once per process rather than re-checked on every bounce/shadow-ray
// - std::getenv's result can't change mid-render, and this sits in camera.h's
// hot per-ray loop.
inline bool enabled() {
	static const bool e = (std::getenv("RAY_TRACER_STATS") != nullptr);
	return e;
}

inline std::atomic<uint64_t> &bounce_rays() {
	static std::atomic<uint64_t> v{0};
	return v;
}

inline std::atomic<uint64_t> &shadow_rays() {
	static std::atomic<uint64_t> v{0};
	return v;
}

// Call once before a render starts - these are process-lifetime statics, so
// a leftover count from an earlier render (e.g. the GUI's render queue,
// or main.cpp's own per-frame video loop) would otherwise bleed into the
// next one's printed stats.
inline void reset() {
	bounce_rays().store(0, std::memory_order_relaxed);
	shadow_rays().store(0, std::memory_order_relaxed);
}

} // namespace render_stats
