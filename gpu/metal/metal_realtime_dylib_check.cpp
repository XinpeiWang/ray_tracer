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
// Must stay identical to qt_gui/realtime_preview_session.cpp's object editing types (src/shared/realtime_api.h).
typedef int (*PickObjectFn)(const char*, double, double, double, double*, double*, double*, char*, int);
typedef bool (*SetObjectOffsetFn)(const char*, int, double, double, double);
typedef void (*ResetObjectsFn)(const char*);

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
		// And a frame without the override puts the scene's own (pinhole) lens back: it looks like the first pinhole frame, not the blurred one.
		std::vector<float> off(pin.size());
		if (!frame(off, 278.0, true)) { fprintf(stderr, "FAIL: frame without the override: %s\n", lastError()); return 1; }
		const double offChange = meanAbsDiff(pin, off);
		printf("depth of field: pinhole again %.4f\n", offChange);
		if (!(offChange < dofChange * 0.7)) { fprintf(stderr, "FAIL: a frame without an aperture override kept the last override's blur\n"); return 1; }
	}
	// Object editing, through the exported functions: pick what the centre pixel shows, move it, and the picture must change; a reset brings it back.
	{
		auto pick = reinterpret_cast<PickObjectFn>(dlsym(lib, "realtime_pick_object"));
		auto setOffset = reinterpret_cast<SetObjectOffsetFn>(dlsym(lib, "realtime_set_object_offset"));
		auto reset = reinterpret_cast<ResetObjectsFn>(dlsym(lib, "realtime_reset_objects"));
		if (!pick || !setOffset || !reset) { fprintf(stderr, "FAIL: missing object editing export(s)\n"); return 1; }
		std::vector<float> before((size_t)w * h * 3), after(before.size()), restored(before.size());
		if (!frame(before, 278.0, true)) { fprintf(stderr, "FAIL: frame before the move: %s\n", lastError()); return 1; }
		const std::vector<float> beforeWorld = worldPos;
		const float* centre = &beforeWorld[((size_t)(h / 2) * w + w / 2) * 4];
		if (centre[3] == 0.0f) { fprintf(stderr, "FAIL: the centre pixel shows no surface\n"); return 1; }
		double lo[3], hi[3], off[3];
		char label[64];
		const int object = pick("A1", centre[0], centre[1], centre[2], lo, hi, off, label, sizeof label);
		if (object < 0) { fprintf(stderr, "FAIL: the surface at the centre pixel belongs to no object\n"); return 1; }
		printf("object editing: the centre pixel shows object %d (%s)\n", object, label);
		if (!setOffset("A1", object, 150.0, 0.0, 0.0)) { fprintf(stderr, "FAIL: set_object_offset refused\n"); return 1; }
		if (!frame(after, 278.0, true)) { fprintf(stderr, "FAIL: frame after the move: %s\n", lastError()); return 1; }
		reset("A1");
		if (!frame(restored, 278.0, true)) { fprintf(stderr, "FAIL: frame after the reset: %s\n", lastError()); return 1; }
		// Compare only where the object was: that is where a move shows, and the rest of the picture is just noise.
		std::vector<char> inObject((size_t)w * h, 0);
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x) {
				const float* q = &beforeWorld[((size_t)y * w + x) * 4];
				inObject[(size_t)y * w + x] = q[3] != 0.0f && pick("A1", q[0], q[1], q[2], nullptr, nullptr, nullptr, nullptr, 0) == object;
			}
		auto diffInObject = [&](const std::vector<float>& p, const std::vector<float>& q, int& count) {
			double sum = 0; count = 0;
			for (size_t px = 0; px < inObject.size(); ++px)
				if (inObject[px]) { ++count; for (int c = 0; c < 3; ++c) sum += std::fabs(p[px * 3 + c] - q[px * 3 + c]); }
			return count ? sum / (3.0 * count) : 0.0;
		};
		int count = 0;
		const double moveChange = diffInObject(before, after, count), resetChange = diffInObject(before, restored, count);
		printf("object editing: over its %d pixels, a move changes the picture by %.4f, after a reset %.4f\n", count, moveChange, resetChange);
		if (count < 20) { fprintf(stderr, "FAIL: the object covers too few pixels to judge\n"); return 1; }
		if (!(moveChange > resetChange * 2.0)) { fprintf(stderr, "FAIL: moving an object did not change the picture where it was\n"); return 1; }
	}
	printf("REALTIME_DYLIB_OK\n");
	dlclose(lib);
	return 0;
}
