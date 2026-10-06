// metal_live_bench.cpp - drives the Live Preview API (metal_live_preview.h) the way the GUI does: one call per frame with
// an orbiting camera. Prints frame times and writes a few frames as PNG so they can be looked at.
//   metal_live_bench [scene=A1] [width=480] [height=480] [spp=2] [frames=120] [out_prefix=live_bench]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "metal_live_preview.h"
#include "../../cpu_renderer/cpu_interface.h"
#include "../../src/external/stb_image_write.h"

static unsigned char to8(float v) {   // simple gamma 2.2 + clamp, enough to look at a frame
    v = v < 0 ? 0 : v;
    v = std::pow(v / (1.0f + v * 0.15f), 1.0f / 2.2f);
    return (unsigned char)(std::fmin(v, 1.0f) * 255.0f + 0.5f);
}

int main(int argc, char** argv) {
    const std::string scene = argc > 1 ? argv[1] : "A1";
    const int w = argc > 2 ? atoi(argv[2]) : 480, h = argc > 3 ? atoi(argv[3]) : 480;
    const int spp = argc > 4 ? atoi(argv[4]) : 2, frames = argc > 5 ? atoi(argv[5]) : 120;
    const std::string prefix = argc > 6 ? argv[6] : "live_bench";

    double fx = 0, fy = 0, fz = 0, ax = 0, ay = 0, az = 1;
    if (!cpu_scene_recommended_camera(scene.c_str(), &fx, &fy, &fz, &ax, &ay, &az)) { fprintf(stderr, "no camera for %s\n", scene.c_str()); return 1; }
    const double dx = fx - ax, dz = fz - az, radius = std::sqrt(dx * dx + dz * dz), a0 = std::atan2(dz, dx);
    printf("scene %s: camera (%.1f,%.1f,%.1f) looking at (%.1f,%.1f,%.1f); %dx%d, %d spp/frame, %d frames\n", scene.c_str(), fx, fy, fz, ax, ay, az, w, h, spp, frames);

    std::vector<float> rgb((size_t)w * h * 3);
    std::vector<double> ms;
    for (int f = 0; f < frames; ++f) {
        const double ang = a0 + 0.35 * std::sin(f * 0.08);   // sway +-20 degrees around the starting view
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = metal_live_render_frame(scene.c_str(), w, h, spp, 6, ax + radius * std::cos(ang), fy, az + radius * std::sin(ang),
                                                true, ax, ay, az, 50.0f, (unsigned)f, rgb.data(), nullptr, nullptr);
        const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { fprintf(stderr, "frame %d failed: %s\n", f, metal_live_last_error()); return 1; }
        ms.push_back(t);
        if (f == 0 || f == frames / 2 || f == frames - 1) {
            std::vector<unsigned char> px((size_t)w * h * 3);
            for (size_t i = 0; i < px.size(); ++i) px[i] = to8(rgb[i]);
            const std::string name = prefix + "_" + std::to_string(f) + ".png";
            stbi_write_png(name.c_str(), w, h, 3, px.data(), w * 3);
        }
    }
    double sum = 0, best = 1e9, worst = 0;
    for (size_t i = 1; i < ms.size(); ++i) { sum += ms[i]; best = std::fmin(best, ms[i]); worst = std::fmax(worst, ms[i]); }
    const double mean = ms.size() > 1 ? sum / (ms.size() - 1) : ms[0];
    printf("first frame (scene load + pipeline): %.0f ms\nsteady state over %zu frames: mean %.1f ms (%.1f fps), best %.1f ms, worst %.1f ms\n",
           ms[0], ms.size() - 1, mean, 1000.0 / mean, best, worst);
    metal_live_shutdown();
    return 0;
}
