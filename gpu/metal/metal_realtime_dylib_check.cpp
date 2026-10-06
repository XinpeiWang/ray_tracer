// metal_realtime_dylib_check.cpp - calls realtime_renderer.dylib the way the Qt GUI's Live Preview does: dlopen, look up
// "realtime_render_frame" and call it through the GUI's hand-duplicated function-pointer type
// (qt_gui/realtime_preview_session.cpp's RenderFrameFn). If either side's signature drifts, this breaks.
//   metal_realtime_dylib_check <path/to/realtime_renderer.dylib>
// Prints REALTIME_DYLIB_OK on success; exit code 0 and a SKIP line when this machine has no usable Metal device.
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Must stay identical to qt_gui/realtime_preview_session.cpp's RenderFrameFn.
typedef bool (*RenderFrameFn)(const char*, int, int, int, int, double, double, double,
                               bool, double, double, double, bool, double, float*, float*, float*, bool,
                               bool, float, const void*, bool, bool, bool,
                               bool, int, unsigned int, bool,
                               bool, float*,
                               double, double,
                               bool, const unsigned char*);
typedef const char* (*GetLastErrorFn)();

static double meanOf(const std::vector<float>& v) { double s = 0; for (float x : v) s += x; return v.empty() ? 0 : s / v.size(); }
static double meanAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
	double s = 0; for (size_t i = 0; i < a.size(); ++i) s += std::fabs(a[i] - b[i]); return s / a.size();
}

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: %s realtime_renderer.dylib\n", argv[0]); return 2; }
	void* lib = dlopen(argv[1], RTLD_NOW);
	if (!lib) { fprintf(stderr, "FAIL: dlopen: %s\n", dlerror()); return 1; }
	auto render = reinterpret_cast<RenderFrameFn>(dlsym(lib, "realtime_render_frame"));
	auto lastError = reinterpret_cast<GetLastErrorFn>(dlsym(lib, "realtime_get_last_error"));
	if (!render || !lastError) { fprintf(stderr, "FAIL: missing export(s)\n"); return 1; }

	const int w = 96, h = 96;
	std::vector<float> a((size_t)w * h * 3), b(a.size()), c(a.size()), worldPos((size_t)w * h * 4, -1.0f), basis(12, -1.0f);
	auto frame = [&](std::vector<float>& out, double camX, bool lookat) {
		return render("A1", w, h, /*spp*/16, /*depth*/5, camX, 278.0, -800.0, lookat, 278.0, 278.0, 278.0,
		              false, 0.0, worldPos.data(), basis.data(), out.data(), false, true, 50.0f, nullptr, true, false, false,
		              false, 2, 0u, false, false, nullptr, -1.0, -1.0, false, nullptr);
	};
	if (!frame(a, 278.0, true)) {
		const char* e = lastError();
		if (e && std::strstr(e, "Metal device")) { printf("SKIP: no usable Metal device (%s)\n", e); return 0; }
		fprintf(stderr, "FAIL: first frame: %s\n", e ? e : "(no detail)"); return 1;
	}
	if (!frame(b, 278.0, true)) { fprintf(stderr, "FAIL: second frame: %s\n", lastError()); return 1; }   // same camera, new seed
	if (!frame(c, -250.0, true)) { fprintf(stderr, "FAIL: moved-camera frame: %s\n", lastError()); return 1; }

	const double meanA = meanOf(a), noise = meanAbsDiff(a, b), moved = meanAbsDiff(a, c);
	printf("mean radiance %.4f, frame-to-frame noise %.4f, camera-move change %.4f\n", meanA, noise, moved);
	if (!(meanA > 0.02)) { fprintf(stderr, "FAIL: the frame is black (mean %.5f)\n", meanA); return 1; }
	if (!(noise > 1e-4)) { fprintf(stderr, "FAIL: two frames with the same camera are identical - the seed is not changing\n"); return 1; }
	if (!(moved > noise * 1.5)) { fprintf(stderr, "FAIL: moving the camera did not change the picture\n"); return 1; }
	for (float v : a) if (!std::isfinite(v) || v < 0) { fprintf(stderr, "FAIL: non-finite or negative radiance\n"); return 1; }
	// reprojection buffers are zero-filled ("no data") rather than left uninitialised
	for (float v : basis) if (v != 0.0f) { fprintf(stderr, "FAIL: camera basis not zero-filled\n"); return 1; }
	printf("REALTIME_DYLIB_OK\n");
	dlclose(lib);
	return 0;
}
