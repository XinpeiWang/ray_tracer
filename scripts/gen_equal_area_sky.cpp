// gen_equal_area_sky.cpp -- writes pbrt_scenes/portal-light-sky.exr, the equal-area (octahedral) environment map
// that pbrt_scenes/portal-light.pbrt uses.
//
// portal-light.pbrt needs an EQUAL-AREA image (pbrt-v4's PortalImageInfiniteLight does not accept an equirectangular
// one). It used to name sssdragon/textures/small_rural_road_equiarea.exr, which belongs to a third-party scene
// collection that is not in a checkout, so a fresh clone silently rendered the scene against a black sky. This
// generates a small original sky instead - a gradient, a ground colour below the horizon and a sun - so the scene
// is self-contained and carries no licence constraints.
//
// Colours are a function of the WORLD direction (+y up), which is how the scene is authored: the light has no
// transform, so an image direction is a world direction. The sun sits in the sky seen through the portal window
// (the window is in the z = -3 wall, looking toward -z).
//
// Build and run from the repo root (any C++17 compiler; writes next to the scenes):
//   c++ -std=c++17 -O2 -I src/shared -I src/external scripts/gen_equal_area_sky.cpp src/external/tinyexr_impl.cpp \
//       src/external/miniz.c -o /tmp/gen_equal_area_sky   (compile miniz.c as C separately if your compiler needs it)
//   /tmp/gen_equal_area_sky pbrt_scenes/portal-light-sky.exr
#include "exr_writer.h"
#include "sampling_sphere.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
	const char *out = argc > 1 ? argv[1] : "portal-light-sky.exr";
	const int n = 256;
	std::vector<float> rgb(static_cast<std::size_t>(n) * n * 3);

	// Unit direction of the sun, as seen from inside the room through the portal (toward -z, a little up and to the left).
	double sx = -0.25, sy = 0.45, sz = -0.85;
	const double sl = std::sqrt(sx * sx + sy * sy + sz * sz);
	sx /= sl; sy /= sl; sz /= sl;
	const double sunCos = std::cos(0.06);   // angular radius ~3.4 degrees

	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			double x, y, z;
			EqualAreaSquareToSphere((i + 0.5) / n, (j + 0.5) / n, x, y, z);
			double r, g, b;
			if (y >= 0.0) {
				// Sky: pale near the horizon, deeper blue toward the zenith.
				const double t = std::pow(y, 0.6);
				r = 0.80 + (0.20 - 0.80) * t;
				g = 0.88 + (0.42 - 0.88) * t;
				b = 1.00 + (0.85 - 1.00) * t;
			} else {
				// Ground below the horizon: warm grey, darker with depth.
				const double t = std::min(1.0, -y * 2.0);
				r = 0.42 * (1.0 - 0.4 * t); g = 0.38 * (1.0 - 0.4 * t); b = 0.32 * (1.0 - 0.4 * t);
			}
			if (x * sx + y * sy + z * sz > sunCos) { r = 60.0; g = 52.0; b = 40.0; }   // the sun: far brighter than the sky
			float *px = &rgb[(static_cast<std::size_t>(j) * n + i) * 3];
			px[0] = static_cast<float>(r); px[1] = static_cast<float>(g); px[2] = static_cast<float>(b);
		}
	}
	std::string error;
	if (!write_exr_image(out, rgb.data(), n, n, error)) {
		std::fprintf(stderr, "writing %s failed: %s\n", out, error.c_str());
		return 1;
	}
	std::printf("wrote %s (%dx%d, equal-area)\n", out, n, n);
	return 0;
}
