#ifndef THREAD_COUNT_H
#define THREAD_COUNT_H

// Auto-detects how many worker threads the CPU renderer should use, honoring
// an explicit RAY_TRACER_THREADS override (a positive integer, or "auto" to
// force re-detection). Shared between camera.h's per-render call and
// main.cpp's video-mode loop, which calls this once up front and then sets
// RAY_TRACER_THREADS for the rest of the process - see the comment at its
// call site for why.

#include <thread>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cmath>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

inline unsigned int determine_render_thread_count() {
	unsigned int hw = std::thread::hardware_concurrency();
	if (hw == 0) hw = 4;

	// Respect explicit override via env var RAY_TRACER_THREADS
	if (const char* env = std::getenv("RAY_TRACER_THREADS")) {
		std::string s(env);
		if (s == "auto") {
			// fallthrough to auto-detect below
		} else {
			try {
				int v = std::stoi(s);
				if (v > 0) return static_cast<unsigned int>(v);
			} catch (...) {
				// ignore parse errors
			}
		}
	}

	// Auto-detect free cores by sampling system idle fraction (Windows only).
#ifdef _WIN32
	FILETIME idle1, kernel1, user1;
	FILETIME idle2, kernel2, user2;
	if (GetSystemTimes(&idle1, &kernel1, &user1)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		if (GetSystemTimes(&idle2, &kernel2, &user2)) {
			auto toULL = [](const FILETIME &ft) -> unsigned long long {
				ULARGE_INTEGER ui; ui.LowPart = ft.dwLowDateTime; ui.HighPart = ft.dwHighDateTime; return ui.QuadPart;
			};
			unsigned long long idleDiff = toULL(idle2) - toULL(idle1);
			unsigned long long kernelDiff = toULL(kernel2) - toULL(kernel1);
			unsigned long long userDiff = toULL(user2) - toULL(user1);
			unsigned long long total = kernelDiff + userDiff;
			if (total == 0) total = 1;
			double busy = double(total - idleDiff) / double(total);
			// estimate free cores = hw * (1 - busy)
			int recommend = int(std::round(hw * (1.0 - busy)));
			if (recommend < 1) recommend = 1;
			if (recommend > (int)hw) recommend = hw;
			return static_cast<unsigned int>(recommend);
		}
	}
#endif
	// Fallback: use all logical cores
	return hw;
}

#endif // THREAD_COUNT_H
