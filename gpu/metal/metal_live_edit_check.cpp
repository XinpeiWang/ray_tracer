// metal_live_edit_check.cpp - drives the Live Preview's object editing (metal_live_preview.h) the way the GUI does and checks that it does what it says:
// every pixel that shows a surface is attributed to an object, moving an object changes the picture where that object was and where it goes, the pick
// finds it again at its new place, and a reset restores the first picture exactly.
//   metal_live_edit_check [scene=A1]
// Prints LIVE_EDIT_OK on success (or SKIP when there is no usable Metal device); anything else is a failure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "metal_live_preview.h"
#include "../../cpu_renderer/cpu_interface.h"

namespace {

constexpr int W = 240, H = 180;
std::string gScene;
double gCam[3], gLook[3];

struct Frame {
    std::vector<float> rgb = std::vector<float>(W * H * 3), world = std::vector<float>(W * H * 4);
};

bool render(Frame& f) {
    return metal_live_render_frame(gScene.c_str(), W, H, 4, 6, gCam[0], gCam[1], gCam[2], true, gLook[0], gLook[1], gLook[2], 50.0f, 7u,
                                   f.rgb.data(), f.world.data(), nullptr, -1.0, -1.0);
}

double meanAbsDiff(const Frame& a, const Frame& b) {
    double sum = 0;
    for (size_t i = 0; i < a.rgb.size(); ++i) sum += std::fabs(a.rgb[i] - b.rgb[i]);
    return sum / a.rgb.size();
}

// LIVE_EDIT_DUMP=<prefix> also writes the pictures as <prefix>_<name>.ppm, to look at them.
void dump(const Frame& f, const char* name) {
    const char* prefix = getenv("LIVE_EDIT_DUMP");
    if (!prefix) return;
    const std::string path = std::string(prefix) + "_" + name + ".ppm";
    if (FILE* out = fopen(path.c_str(), "wb")) {
        fprintf(out, "P6\n%d %d\n255\n", W, H);
        for (float v : f.rgb) {
            const float g = std::pow(std::fmin(std::fmax(v, 0.0f), 1.0f), 1.0f / 2.2f);
            fputc((int)(g * 255.0f + 0.5f), out);
        }
        fclose(out);
    }
}

int fail(const char* what) {
    fprintf(stderr, "LIVE_EDIT_FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    gScene = argc > 1 ? argv[1] : "A1";
    if (!cpu_scene_recommended_camera(gScene.c_str(), &gCam[0], &gCam[1], &gCam[2], &gLook[0], &gLook[1], &gLook[2])) return fail("no camera for the scene");
    Frame first;
    if (!render(first)) {
        if (strstr(metal_live_last_error(), "Metal device")) { printf("SKIP: no usable Metal device\n"); return 0; }
        fprintf(stderr, "first frame: %s\n", metal_live_last_error());
        return fail("the first frame failed");
    }

    // 1. Every few pixels: a pixel that shows a surface is attributed to some object. Count the objects seen and their pixel counts.
    int surfaces = 0, attributed = 0;
    std::vector<int> pixelsOfObject;
    for (int y = 4; y < H; y += 6)
        for (int x = 4; x < W; x += 6) {
            const float* q = &first.world[((size_t)y * W + x) * 4];
            if (q[3] < 0.5f) continue;
            ++surfaces;
            const int obj = metal_live_pick_object(gScene.c_str(), q[0], q[1], q[2], nullptr, nullptr, nullptr, nullptr, 0);
            if (obj < 0) continue;
            ++attributed;
            if ((size_t)obj >= pixelsOfObject.size()) pixelsOfObject.resize(obj + 1, 0);
            ++pixelsOfObject[obj];
        }
    printf("%d surface samples, %d attributed to an object\n", surfaces, attributed);
    if (surfaces < 100) return fail("the picture shows almost no surface");
    if (attributed < surfaces * 9 / 10) return fail("fewer than 90% of the visible surface points were attributed to an object");

    // 2. Move the object with the fewest (but some) pixels: it is the small one, not a wall.
    int target = -1;
    for (size_t i = 0; i < pixelsOfObject.size(); ++i)
        if (pixelsOfObject[i] >= 8 && (target < 0 || pixelsOfObject[i] < pixelsOfObject[target])) target = (int)i;
    if (target < 0) return fail("no object with enough pixels to move");
    int px = -1, py = -1;
    for (int y = 4; y < H && px < 0; y += 6)
        for (int x = 4; x < W; x += 6) {
            const float* q = &first.world[((size_t)y * W + x) * 4];
            if (q[3] > 0.5f && metal_live_pick_object(gScene.c_str(), q[0], q[1], q[2], nullptr, nullptr, nullptr, nullptr, 0) == target) { px = x; py = y; break; }
        }
    const float* hit = &first.world[((size_t)py * W + px) * 4];
    double lo[3], hi[3], off[3];
    char label[64];
    if (metal_live_pick_object(gScene.c_str(), hit[0], hit[1], hit[2], lo, hi, off, label, sizeof label) != target) return fail("the pick is not repeatable");
    printf("moving object %d (%s), box (%.1f %.1f %.1f)-(%.1f %.1f %.1f), seen at pixel (%d,%d)\n", target, label, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], px, py);
    if (off[0] != 0 || off[1] != 0 || off[2] != 0) return fail("an untouched object reports an offset");

    const double dx = 0.8 * (hi[0] - lo[0]), dz = 0.8 * (hi[2] - lo[2]);   // sideways and back by most of its size
    if (!metal_live_set_object_offset(gScene.c_str(), target, dx, 0.0, dz)) return fail("set_object_offset refused");
    Frame moved;
    if (!render(moved)) { fprintf(stderr, "%s\n", metal_live_last_error()); return fail("the frame after the move failed"); }
    dump(first, "first");
    dump(moved, "moved");
    const double change = meanAbsDiff(first, moved);
    printf("mean absolute change of the picture after the move: %.5f\n", change);
    if (change < 1e-4) return fail("moving the object did not change the picture");

    // 3. The pick follows the object: the point it was seen at is no longer on it, the same point shifted is.
    if (metal_live_pick_object(gScene.c_str(), hit[0], hit[1], hit[2], nullptr, nullptr, nullptr, nullptr, 0) == target) {
        // The old spot can still be on the object when the object is big and the shift small: only fail if the whole box did not move.
        double lo2[3], hi2[3], off2[3];
        metal_live_pick_object(gScene.c_str(), hit[0], hit[1], hit[2], lo2, hi2, off2, nullptr, 0);
        if (std::fabs(lo2[0] - lo[0]) < 1e-6 && std::fabs(lo2[2] - lo[2]) < 1e-6) return fail("the object's box did not move");
    }
    double lo2[3], hi2[3], off2[3];
    const int again = metal_live_pick_object(gScene.c_str(), hit[0] + dx, hit[1], hit[2] + dz, lo2, hi2, off2, nullptr, 0);
    if (again != target) return fail("the object is not found at its new place");
    if (std::fabs(off2[0] - dx) > 1e-9 || std::fabs(off2[2] - dz) > 1e-9) return fail("the reported offset is not the one set");
    if (std::fabs((lo2[0] - lo[0]) - dx) > 1e-4 * (1 + std::fabs(dx))) return fail("the box did not move by the offset");

    // 4. Reset: the first picture comes back exactly (same seed, same scene).
    metal_live_reset_objects(gScene.c_str());
    Frame back;
    if (!render(back)) return fail("the frame after the reset failed");
    const double restored = meanAbsDiff(first, back);
    printf("mean absolute difference from the first picture after the reset: %.7f\n", restored);
    if (restored > 1e-6) return fail("a reset did not restore the first picture");

    metal_live_shutdown();
    printf("LIVE_EDIT_OK\n");
    return 0;
}
