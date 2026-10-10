// metal_live_preview.mm - see metal_live_preview.h.
#include "metal_live_preview.h"

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "metal_poc_app.h"
#include "../../src/shared/live_object_edit.h"
#include "../../src/shared/pbrt_arrangement.h"
#include "../../cpu_renderer/cpu_interface.h"   // cpu_scene_pbrt_path_by_id

namespace {

struct LiveSession {
    std::unique_ptr<MetalPocApp> app;
    std::string sceneId;
    int width = 0, height = 0;
    // MetalPocApp keeps pointers into the argv strings it was started with (outPath), so they live as long as it does.
    std::string widthStr, heightStr, sppStr, depthStr, pbrtPath;
    live_objects::PickIndex pick;   // of the scene as it was built (edits applied); filled by the scene-edit hook
    live_objects::ObjectList objects;                     // the scene's objects, and where each shape sits in the scene's text, for "Save arrangement"
    std::vector<pbrt_flatten::ShapeRange> ranges;
};

// Object moves for the current scene: one offset (pbrt world units) per object (see live_object_edit.h), applied while the scene is built.
// They outlive a session (a new picture size rebuilds the session, not the edits) and end with a different scene or a reset.
struct ObjectEdits {
    std::string sceneId;
    std::vector<std::array<double, 3>> offsets;
};
ObjectEdits gEdits;

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
    app.pbrtRequireAllMeshes = cpu_scene_requires_files_by_id(sceneId) != 0;
    if (gEdits.sceneId != sceneId) gEdits = ObjectEdits{sceneId, {}};
    LiveSession* const session = s.get();
    app.pbrtSceneEdit = [session](pbrt_flatten::FlatScene& scene) {
        const live_objects::ObjectList objects = live_objects::objectsOf(scene);
        gEdits.offsets.resize(objects.size(), {0.0, 0.0, 0.0});
        for (size_t i = 0; i < objects.size(); ++i)
            if (gEdits.offsets[i][0] != 0.0 || gEdits.offsets[i][1] != 0.0 || gEdits.offsets[i][2] != 0.0)
                live_objects::translateObject(scene, objects, i, gEdits.offsets[i].data());
        session->pick = live_objects::buildPickIndex(scene, objects);
        session->objects = objects;
        session->ranges = scene.shapeRanges;
    };
    app.buildScene();
    if (!app.pbrtMissingAssetsError.empty()) { fail(app.pbrtMissingAssetsError); return nullptr; }
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
                             float* out_rgb, float* out_world_pos, float* out_camera_basis,
                             double aperture, double focus_distance) {
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
        const float sceneScale = app.pbrtSceneScale;
        const float sceneLensRadius = app.pbrtLensRadius, sceneFocusDistance = app.pbrtFocusDistance;
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
            // The dispatch keeps its own copy of the uniforms from frame to frame, so a frame without an override must put the scene's own lens
            // back, not leave the last override in place.
            if (aperture >= 0.0) {   // thin-lens depth of field, scene units -> this scene's internal units
                u.lensRadius = (float)(0.5 * aperture) * sceneScale;
                u.focusDistance = (float)focus_distance * sceneScale;
            } else {
                u.lensRadius = sceneLensRadius;
                u.focusDistance = sceneFocusDistance;
            }
        });
        if (!ok) return fail("the frame failed to render");
        const size_t n = (size_t)width * height;
        if (app.pixels.size() < n * 4) return fail("the frame produced no pixels");
        for (size_t i = 0; i < n; ++i) {
            out_rgb[i * 3 + 0] = app.pixels[i * 4 + 0];
            out_rgb[i * 3 + 1] = app.pixels[i * 4 + 1];
            out_rgb[i * 3 + 2] = app.pixels[i * 4 + 2];
        }
        if (out_world_pos) {
            // Back from this scene's internal (recentred, rescaled) units to the caller's: p = (q - offset) / scale + bboxCenter.
            const bool haveData = app.liveWorldPos.size() >= n * 4;
            for (size_t i = 0; i < n; ++i) {
                const float* q = haveData ? &app.liveWorldPos[i * 4] : nullptr;
                float* o = &out_world_pos[i * 4];
                if (q && q[3] > 0.5f) {
                    o[0] = (q[0] - app.pbrtSceneOffset.x) / sceneScale + app.pbrtBboxCenter.x;
                    o[1] = (q[1] - app.pbrtSceneOffset.y) / sceneScale + app.pbrtBboxCenter.y;
                    o[2] = (q[2] - app.pbrtSceneOffset.z) / sceneScale + app.pbrtBboxCenter.z;
                    o[3] = 1.0f;
                } else {
                    o[0] = o[1] = o[2] = o[3] = 0.0f;
                }
            }
        }
        if (out_camera_basis) {
            // The view plane at distance 1 along forward: the kernel casts normalize(forward + sx * right * tanHalfFov * aspect +
            // sy * up * tanHalfFov), sx and sy in [-1, 1]. Same units as the caller's camera position (directions are unchanged by
            // the uniform rescale).
            const double aspect = (double)width / (double)height, halfH = app.pbrtTanHalfFov, halfW = halfH * aspect;
            const double o[3] = {cam_x, cam_y, cam_z};
            const double f[3] = {app.pbrtCameraForward.x, app.pbrtCameraForward.y, app.pbrtCameraForward.z};
            const double r[3] = {app.pbrtCameraRight.x, app.pbrtCameraRight.y, app.pbrtCameraRight.z};
            const double uu[3] = {app.pbrtCameraUp.x, app.pbrtCameraUp.y, app.pbrtCameraUp.z};
            for (int k = 0; k < 3; ++k) {
                const double h = 2.0 * halfW * r[k], v = 2.0 * halfH * uu[k];
                out_camera_basis[0 + k] = (float)o[k];
                out_camera_basis[3 + k] = (float)(o[k] + f[k] - 0.5 * h - 0.5 * v);
                out_camera_basis[6 + k] = (float)h;
                out_camera_basis[9 + k] = (float)v;
            }
        }
    }
    return true;
}

int metal_live_pick_object(const char* scene_id, double x, double y, double z,
                           double* out_lo, double* out_hi, double* out_offset, char* out_label, int label_size) {
    if (!scene_id) return -1;
    std::lock_guard<std::mutex> lock(gMutex);
    if (!gSession || gSession->sceneId != scene_id) return -1;   // nothing has been drawn yet (or it is being rebuilt after a move)
    // A first-hit position lies on the surface up to the picture's rounding; half a percent of the scene's size covers that.
    const double tolerance = 0.005 / gSession->app->pbrtSceneScale;
    const double p[3] = {x, y, z};
    const live_objects::PickResult hit = live_objects::pick(gSession->pick, p, tolerance);
    if (hit.object < 0) return -1;
    const live_objects::PickShape& shape = gSession->pick.shapes[hit.slot];
    for (int a = 0; a < 3; ++a) {
        if (out_lo) out_lo[a] = shape.lo[a];
        if (out_hi) out_hi[a] = shape.hi[a];
        if (out_offset) out_offset[a] = (size_t)hit.object < gEdits.offsets.size() ? gEdits.offsets[hit.object][a] : 0.0;
    }
    if (out_label && label_size > 0) snprintf(out_label, (size_t)label_size, "%s", shape.label.c_str());
    return hit.object;
}

bool metal_live_set_object_offset(const char* scene_id, int object, double dx, double dy, double dz) {
    if (!scene_id || object < 0) return false;
    std::lock_guard<std::mutex> lock(gMutex);
    if (gEdits.sceneId != scene_id || (size_t)object >= gEdits.offsets.size()) return false;
    std::array<double, 3>& o = gEdits.offsets[object];
    if (o[0] == dx && o[1] == dy && o[2] == dz) return true;
    o = {dx, dy, dz};
    gSession.reset();   // the next frame rebuilds the scene with the new offsets
    return true;
}

void metal_live_reset_objects(const char* scene_id) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!scene_id || gEdits.sceneId != scene_id) return;
    bool any = false;
    for (std::array<double, 3>& o : gEdits.offsets) {
        any = any || o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0;
        o = {0.0, 0.0, 0.0};
    }
    if (any) gSession.reset();
}

bool metal_live_export_arrangement(const char* scene_id, const char* out_path, char* message, int message_size) {
    auto say = [&](const std::string& text) { if (message && message_size > 0) snprintf(message, (size_t)message_size, "%s", text.c_str()); };
    if (!scene_id || !out_path) { say("bad arguments"); return false; }
    std::lock_guard<std::mutex> lock(gMutex);
    if (!gSession || gSession->sceneId != scene_id || gEdits.sceneId != scene_id) { say("nothing has been drawn yet, so there is nothing to save"); return false; }
    namespace fs = std::filesystem;
    std::ifstream in(gSession->pbrtPath, std::ios::binary);
    if (!in) { say("cannot read the scene file " + gSession->pbrtPath); return false; }
    std::ostringstream text;
    text << in.rdbuf();
    const fs::path original = fs::absolute(gSession->pbrtPath);
    const std::string dir = original.parent_path().string() + "/";
    int objectsMoved = 0;
    for (const std::array<double, 3>& o : gEdits.offsets) objectsMoved += (o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0) ? 1 : 0;
    const pbrt_arrangement::Result result = pbrt_arrangement::write(text.str(), gSession->ranges, gSession->objects, gEdits.offsets, dir,
        [](const std::string& path) { std::error_code ec; return fs::exists(path, ec); },
        "# " + original.filename().string() + " with " + std::to_string(objectsMoved) + " object(s) moved, saved from Live Preview.\n");
    if (objectsMoved == 0) { say("no object has been moved"); return false; }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { say(std::string("cannot write ") + out_path); return false; }
    out << result.text;
    out.close();
    if (!out) { say(std::string("cannot write ") + out_path); return false; }
    std::string summary = std::to_string(objectsMoved) + " object(s) moved";
    if (result.skippedShapes > 0) summary += "; " + std::to_string(result.skippedShapes) + " shape(s) sit in included files and stayed where they were";
    say(summary);
    return true;
}

const char* metal_live_last_error() { return tlsError.c_str(); }

void metal_live_shutdown() {
    std::lock_guard<std::mutex> lock(gMutex);
    gSession.reset();
}
