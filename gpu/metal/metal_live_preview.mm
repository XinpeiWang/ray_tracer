// metal_live_preview.mm - see metal_live_preview.h.
#include "metal_live_preview.h"

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "metal_poc_app.h"
#include "../../cpu_renderer/cpu_interface.h"   // cpu_scene_pbrt_path_by_id

namespace {

struct LiveSession {
    std::unique_ptr<MetalPocApp> app;
    std::string sceneId;
    int width = 0, height = 0;
    // MetalPocApp keeps pointers into the argv strings it was started with (outPath), so they live as long as it does.
    std::string widthStr, heightStr, sppStr, depthStr, pbrtPath;
};

std::mutex gMutex;
std::unique_ptr<LiveSession> gSession;
thread_local std::string tlsError;

bool fail(const std::string& msg) {
    tlsError = msg;
    fprintf(stderr, "metal_live_render_frame: %s\n", msg.c_str());
    return false;
}

// The registry hands back scene files as relative paths ("pbrt_scenes/x.pbrt"), which only resolve when the current directory
// is the app's own. A GUI process launched from Finder has cwd "/" (Live Preview runs IN the GUI process, unlike a render job,
// whose subprocess the GUI starts with the right working directory), so look in the places the files actually live: the
// current directory, the host executable's directory (Contents/MacOS in the app bundle), this library's directory, and their
// parents (a development build keeps pbrt_scenes at the repository root, one level above build_macos/).
std::string resolveSceneFile(const std::string& rel, std::string& tried) {
    namespace fs = std::filesystem;
    if (fs::path(rel).is_absolute()) return rel;
    std::vector<fs::path> dirs;
    dirs.push_back(fs::current_path());
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        char real[PATH_MAX];
        const fs::path exeDir = fs::path(realpath(buf, real) ? real : buf).parent_path();
        dirs.push_back(exeDir);
        dirs.push_back(exeDir.parent_path());
    }
    Dl_info info;
    if (dladdr(reinterpret_cast<const void*>(&resolveSceneFile), &info) && info.dli_fname) {
        const fs::path libDir = fs::path(info.dli_fname).parent_path();
        dirs.push_back(libDir);
        dirs.push_back(libDir.parent_path());
    }
    std::error_code ec;
    for (const fs::path& d : dirs) {
        const fs::path candidate = d / rel;
        tried += (tried.empty() ? "" : ", ") + d.string();
        if (fs::exists(candidate, ec)) return candidate.string();
    }
    return rel;   // not found: the loader reports it
}

std::unique_ptr<LiveSession> createSession(const char* sceneId, int width, int height, int spp, int maxDepth) {
    const char* pbrtPath = cpu_scene_pbrt_path_by_id(sceneId);
    if (!pbrtPath || !pbrtPath[0]) {
        fail(std::string("scene '") + sceneId + "' has no pbrt file backing it; Metal renders only pbrt-backed scenes");
        return nullptr;
    }
    auto s = std::make_unique<LiveSession>();
    s->sceneId = sceneId;
    s->width = width;
    s->height = height;
    s->widthStr = std::to_string(width);
    s->heightStr = std::to_string(height);
    s->sppStr = std::to_string(spp);
    s->depthStr = std::to_string(maxDepth);
    std::string tried;
    s->pbrtPath = resolveSceneFile(pbrtPath, tried);
    if (!std::filesystem::exists(s->pbrtPath)) {
        fail(std::string("cannot find the scene file ") + pbrtPath + " (looked in: " + tried + ")");
        return nullptr;
    }
    static const char* kOutPath = "live_preview.png";   // never written: a live session renders into memory only
    const char* args[8] = {"metal_live_preview", s->widthStr.c_str(), s->heightStr.c_str(), kOutPath,
                            s->sppStr.c_str(), s->depthStr.c_str(), "none", s->pbrtPath.c_str()};

    s->app = std::make_unique<MetalPocApp>();
    MetalPocApp& app = *s->app;
    if (!app.parseArgsAndCreateDevice(8, args)) { fail("no usable Metal device"); return nullptr; }
    app.isolatePbrtLighting = true;
    app.skipDemoRoom = true;
    app.liveSession = true;
    app.buildScene();
    if (!app.havePbrtCamera) { fail("the pbrt scene failed to load (see the loader's message above)"); return nullptr; }
    if (!app.buildGPUResources()) { fail("could not build the GPU scene resources"); return nullptr; }
    if (!app.compileShaderAndDispatch(8, args)) { fail("shader compile / first frame failed"); return nullptr; }   // also fills app.liveRender
    if (!app.liveRender) { fail("internal error: the live dispatch was not captured"); return nullptr; }
    return s;
}

}  // namespace

bool metal_live_render_frame(const char* scene_id, int width, int height, int spp, int max_depth,
                             double cam_x, double cam_y, double cam_z,
                             bool has_custom_lookat, double lookat_x, double lookat_y, double lookat_z,
                             float max_component_value, unsigned int frame_seed,
                             float* out_rgb, float* out_world_pos, float* out_camera_basis) {
    if (!scene_id || !out_rgb || width <= 0 || height <= 0) return fail("bad arguments");
    std::lock_guard<std::mutex> lock(gMutex);
    @autoreleasepool {
        if (!gSession || gSession->sceneId != scene_id || gSession->width != width || gSession->height != height) {
            gSession.reset();   // free the old session's GPU memory before building the new one
            gSession = createSession(scene_id, width, height, spp, max_depth);
            if (!gSession) return false;
        }
        MetalPocApp& app = *gSession->app;
        app.applyCameraLookAt(cam_x, cam_y, cam_z, has_custom_lookat, lookat_x, lookat_y, lookat_z);
        const PackedFloat3 pos{app.pbrtCameraPos.x, app.pbrtCameraPos.y, app.pbrtCameraPos.z};
        const PackedFloat3 fwd{app.pbrtCameraForward.x, app.pbrtCameraForward.y, app.pbrtCameraForward.z};
        const PackedFloat3 right{app.pbrtCameraRight.x, app.pbrtCameraRight.y, app.pbrtCameraRight.z};
        const PackedFloat3 up{app.pbrtCameraUp.x, app.pbrtCameraUp.y, app.pbrtCameraUp.z};
        // A well-mixed seed: the kernel derives each pixel's stream from frameSeed linearly, so consecutive small
        // integers would give strongly correlated noise between frames.
        const uint32_t seed = (frame_seed + 1u) * 2654435761u;
        const bool ok = app.liveRender([&](Uniforms& u) {
            u.cameraPos = pos;
            u.cameraForward = fwd;
            u.cameraRight = right;
            u.cameraUp = up;
            u.samplesPerPixel = (uint32_t)std::max(1, spp);
            u.maxDepth = (uint32_t)std::max(1, max_depth);
            u.frameSeed = seed;
            u.adaptiveSampling = 0u;   // a fixed few samples per frame; the GUI does the accumulating
            if (max_component_value > 0.0f) u.fireflyClamp = max_component_value;
        });
        if (!ok) return fail("the frame failed to render");
        const size_t n = (size_t)width * height;
        if (app.pixels.size() < n * 4) return fail("the frame produced no pixels");
        for (size_t i = 0; i < n; ++i) {
            out_rgb[i * 3 + 0] = app.pixels[i * 4 + 0];
            out_rgb[i * 3 + 1] = app.pixels[i * 4 + 1];
            out_rgb[i * 3 + 2] = app.pixels[i * 4 + 2];
        }
        if (out_world_pos) std::memset(out_world_pos, 0, n * 4 * sizeof(float));
        if (out_camera_basis) std::memset(out_camera_basis, 0, 12 * sizeof(float));
    }
    return true;
}

const char* metal_live_last_error() { return tlsError.c_str(); }

void metal_live_shutdown() {
    std::lock_guard<std::mutex> lock(gMutex);
    gSession.reset();
}
