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

#include "../../qt_gui/camera_math.h"   // header-only: the GUI's own projectToScreen(), used by its temporal reprojection

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
	auto frame = [&](std::vector<float>& out, double camX, bool lookat, double aperture = -1.0, double focus = -1.0) {
		return render("A1", w, h, /*spp*/16, /*depth*/5, camX, 278.0, -800.0, lookat, 278.0, 278.0, 278.0,
		              false, 0.0, worldPos.data(), basis.data(), out.data(), false, true, 50.0f, nullptr, true, false, false,
		              false, 2, 0u, false, false, nullptr, aperture, focus, false, nullptr);
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
	// Reprojection data (of the last, moved-camera frame): every pixel that hit something must project, through the returned camera
	// basis, back onto that pixel - the exact relation the GUI's reprojection relies on to reuse an old accumulation.
	{
		camera_math::CameraBasis cb{{basis[0], basis[1], basis[2]}, {basis[3], basis[4], basis[5]},
		                            {basis[6], basis[7], basis[8]}, {basis[9], basis[10], basis[11]}};
		int hits = 0; double errSum = 0, errMax = 0;
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x) {
				const float* p = &worldPos[((size_t)y * w + x) * 4];
				if (p[3] == 0.0f) continue;
				const camera_math::ScreenProjection sp = camera_math::projectToScreen({p[0], p[1], p[2]}, cb);
				if (!sp.inFront) { fprintf(stderr, "FAIL: a hit point projects behind the camera (pixel %d,%d)\n", x, y); return 1; }
				const double ex = sp.s * w - (x + 0.5), ey = (1.0 - sp.t) * h - (y + 0.5);   // pixels - exact: the position is the pixel centre's, not a jittered sample's
				const double e = std::sqrt(ex * ex + ey * ey);
				errSum += e; errMax = std::max(errMax, e); ++hits;
			}
		printf("world positions: %d of %d pixels hit, mean reprojection error %.4f px, max %.4f px\n", hits, w * h, hits ? errSum / hits : 0.0, errMax);
		if (hits < w * h / 2) { fprintf(stderr, "FAIL: too few pixels have a world position\n"); return 1; }
		if (!(errMax < 0.05)) { fprintf(stderr, "FAIL: world positions do not project onto their pixels\n"); return 1; }
	}
	// Depth of field: a wide aperture focused far from the scene must blur it (a different picture than the pinhole one).
	{
		std::vector<float> pin((size_t)w * h * 3), blur(pin.size()), blur2(pin.size());
		if (!frame(pin, 278.0, true) || !frame(blur, 278.0, true, 60.0, 400.0) || !frame(blur2, 278.0, true, 60.0, 400.0)) {
			fprintf(stderr, "FAIL: depth-of-field frame: %s\n", lastError()); return 1;
		}
		const double dofChange = meanAbsDiff(pin, blur), baseline = meanAbsDiff(blur, blur2);
		printf("depth of field: change %.4f vs noise %.4f\n", dofChange, baseline);
		if (!(dofChange > baseline * 1.3)) { fprintf(stderr, "FAIL: an aperture override did not change the picture\n"); return 1; }
	}
	printf("REALTIME_DYLIB_OK\n");
	dlclose(lib);
	return 0;
}
