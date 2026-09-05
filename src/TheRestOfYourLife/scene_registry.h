#pragma once
// scene_registry.h -- CPU scene registry (pbrt-v4 SceneEntity pattern),
// the sole source of truth for scene metadata (see SceneDescriptor's
// gpu_compatible field comment below for why there's only one table).
//
// Scene names are defined as constants in src/shared/scene_descriptor.h
// (SceneNames::). Always use SceneNames:: constants here -- never raw
// string literals -- so a name can never drift between call sites.
//
// To add a new scene:
//   1. Add a SceneNames:: constant in scene_descriptor.h
//   2. Add a builder function in scenes.h (or scenes_book.h/scenes_advanced.h)
//   3. Add one SceneDescriptor entry in get_scene_registry() below using SceneNames::
//   Done -- cpu_interface.cpp's C API and scene_metadata.dll (and through
//   it, the GUI) pick it up automatically, no other file to touch unless
//   you're also adding GPU support (see gpu/optix/scene_builder.cpp).

#include "../shared/scene_descriptor.h"
#include "scenes.h"
#include "cornell_box_scene.h"
#include "sky_light.h"
#include "punctual_light_objects.h"
#include "camera.h"
#include "pbrt_cpu_builder.h"
#include "../shared/pbrt_discover.h"
#include "../shared/pbrt_load.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Whether the camera lookfrom is overridden by the user (cam_x/y/z params)
enum class CameraMode { Fixed, UserControlled };

struct CameraConfig {
    double vfov;
    double lookfrom_x, lookfrom_y, lookfrom_z;
    double lookat_x,   lookat_y,   lookat_z;
    double bg_r, bg_g, bg_b;
    CameraMode mode = CameraMode::Fixed;
    double defocus_angle = 0.0;  // 0 = no DOF
    double focus_dist    = 10.0;
    // Camera motion blur (camera::camera_is_animated - see that field's own
    // comment, camera.h). animated=false (default) leaves every existing
    // scene's positional brace-initializer untouched - these fields simply
    // zero-fill. An animated scene always uses its own keyframes regardless
    // of `mode`/cam_x/y/z overrides (see cpu_interface.cpp's own comment) -
    // a moving camera has no single "current position" for an override to
    // mean.
    bool   animated       = false;
    double lookfrom_t1_x = 0.0, lookfrom_t1_y = 0.0, lookfrom_t1_z = 0.0;
    double lookat_t1_x   = 0.0, lookat_t1_y   = 0.0, lookat_t1_z   = 0.0;
    double shutter_open  = 0.0, shutter_close = 1.0;
};

// Shared CameraConfig rows for scenes that intentionally reuse another
// scene's exact framing (Education/B24 - see their own comments in
// get_builtin_scene_registry() below for why: same geometry, only the id/
// category/description/material differ). Named constants instead of each
// entry repeating the same literal row make "same camera as X" a fact the
// compiler enforces rather than one only asserted by comment - if A1's or
// B23's own row is ever retuned, every sibling below picks up the change
// automatically instead of silently going out of sync with it.
constexpr CameraConfig kCornellBoxCamera =
    { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled };
constexpr CameraConfig kPrismCamera =
    { 30, 75, 60, -400,  75, 75, 250,  0, 0, 0, CameraMode::UserControlled };

// Forward alias so std::function<void(camera_t&)> inside SceneDescriptor
// doesn't conflict with the CameraConfig field named 'camera'.
using camera_t = camera;

struct SceneDescriptor {
    // Category letter + number within category (e.g. "B10" = 10th
    // Materials scene) - the id used everywhere outside this file: CLI,
    // scene_metadata.dll, the GUI, tests. std::string (not const char*,
    // unlike this struct's other string fields) because pbrt-loaded scenes
    // generate their id at runtime with unbounded count - see
    // pbrt_scene_registry::append() below.
    std::string id;
    // The OLD flat 0..68 scene_id, kept only so gpu/optix/scene_builder.cpp's
    // large switch(scene_id) doesn't need rewriting into a string-keyed
    // dispatch - build_scene() looks a scene up by `id` via find_scene()
    // and switches on `legacy_id` internally. Never exposed outside that
    // one call site; do not add new uses of this field.
    int         legacy_id;
    const char* name;
    // Which group this scene appears under in the GUI's category tabs. Always
    // a SceneCategories:: constant, never a literal - a typo would put the
    // scene under no tab at all, which is silent (see the category tests in
    // tests/unit/scene_registry_tests.cpp, which pin exactly that).
    const char* category;
    const char* description;
    const char* performance;   // "Fast" | "Medium" | "Slow" | "Very Slow"
    int         recommended_spp;
    bool        requires_files;
    // Every field above and this one are queried live by the Qt GUI via
    // scene_metadata.dll (see qt_gui/scene_metadata_client.h) rather than
    // a separate, independently-maintained copy - this struct is now the
    // sole source of truth for scene metadata (src/shared/scene_descriptor.h
    // used to duplicate id/name/description/performance/recommended_spp/
    // requires_files in its own kScenes[] table; that drifted out of sync
    // once already - scene 1 got gpu_compatible=true here without its
    // scene_descriptor.h row being updated, so the GUI kept showing "CPU
    // only" until fixed separately - which is why that table was removed).
    bool        gpu_compatible;
    CameraConfig camera;
    std::function<hittable_list()>                       build_world;
    std::function<hittable_list()>                       build_lights;   // may return empty list
    std::function<std::shared_ptr<sky_light>()>          build_sky;      // nullptr = flat bg
    std::function<std::shared_ptr<punctual_light_list>()> build_punct;   // nullptr = none
    std::function<void(camera_t&)>                       setup_camera;   // nullptr = default perspective

    // A pbrt-loaded scene's own Integrator directive - 0/empty for every
    // hand-built scene above (none of them were ever declared via an
    // Integrator directive to read in the first place). Deliberately last:
    // the ~65 hand-built entries in get_builtin_scene_registry() below
    // construct this struct with POSITIONAL brace-init, so a field inserted
    // anywhere earlier silently reassigns every value after it to the wrong
    // member. Not applied automatically (this renderer's CLI integrator/
    // depth selection is a positional argument + explicit flags, not a
    // per-scene default lookup) - cpu_interface.cpp compares against these
    // to warn when what a render actually does diverges from what the scene
    // asked for, rather than silently doing something else. See
    // pbrt_scene_registry::wire_pbrt_backed_scene() below for where this
    // gets set, and pbrt_discover::Discovered::maxDepth/integrator for
    // where the values themselves come from.
    int         recommended_max_depth = 0;
    std::string recommended_integrator;
    // Same shape as recommended_integrator above (empty for every hand-built
    // scene; not auto-applied; cpu_interface.cpp warns rather than switches
    // samplers) - the loaded scene's own Sampler directive type name. Also
    // deliberately last, same positional-brace-init reason as
    // recommended_integrator's own comment.
    std::string recommended_sampler;

    // LightSource "infinite" "point3 portal[4]" (pbrt-v4's windowed
    // infinite light, visible only through a finite rectangular window) -
    // nullptr unless the loaded scene declared a real portal[4] (see
    // pbrt_cpu_builder.h's BuildResult::portal comment). Deliberately
    // last, same positional-brace-init reason as recommended_integrator/
    // recommended_sampler above - every hand-built scene above leaves this
    // at its default (empty std::function = nullptr), since none of them
    // are pbrt-loaded portal-light scenes.
    std::function<std::shared_ptr<PortalImageInfiniteLightData<double>>()> build_portal;

    // Integrator "string lightsampler" - same "empty for every hand-built
    // scene; not auto-applied; cpu_interface.cpp warns rather than
    // switches samplers" shape as recommended_sampler above. Deliberately
    // LAST (after build_portal, not next to recommended_sampler) - same
    // positional-brace-init fragility reasoning as every other field
    // added here after the struct's own initial fields: appending keeps
    // every existing hand-built scene's positional initializer list
    // referring to the same members it always did.
    std::string recommended_light_sampler;

    // pbrt-v4's own "camera medium" (see pbrt_flatten::FlatScene::
    // cameraMediumIndex's own comment for what this requests, and
    // camera_t::camera_medium's own comment, camera.h, for what consumes
    // it) - nullptr unless the loaded scene declared a MediumInterface
    // before its Camera directive (see pbrt_cpu_builder.h's BuildResult::
    // cameraMedium comment). Deliberately LAST, same positional-brace-init
    // fragility reasoning as every other field added here after the
    // struct's own initial fields - every hand-built scene above leaves
    // this at its default (empty std::function = nullptr), since none of
    // them are pbrt-loaded camera-medium scenes.
    std::function<std::shared_ptr<ambient_medium>()> build_camera_medium;

    // A CURATED (not derived from the .pbrt file - see the comment below on
    // why not) flat multiplier auto-applied to the Render Options tab's
    // Exposure control when this scene is selected, the same way
    // recommended_spp is auto-applied to the Samples field (mainwindow_
    // slots.cpp's onSceneChanged()). Defaults to 1.0 (no-op) for every
    // hand-built scene and the vast majority of pbrt-backed ones.
    //
    // pbrt-v4 files that DO specify exposure-relevant metadata (Film's
    // "float iso", occasionally paired with a camera shutter time) encode
    // it for pbrt-v4's own PixelSensor model - a real, but partial, physical
    // camera response pipeline this project's tone-mapping does not
    // reproduce. Computing pbrt's own imagingRatio = exposureTime * ISO/100
    // from those fields was tried and rejected: for H19 (Crown, iso 150) it
    // works out to 1.5x, but the render needs roughly 30x to look like
    // anything other than a near-black silhouette - the gap is dominated by
    // something else entirely (most likely this renderer's blackbody-arealight
    // radiance scale not matching pbrt-v4's own convention, though that
    // wasn't tracked down further - out of scope for a scene-registry
    // addition). A formula-derived value here would therefore be
    // confidently wrong; an empirically-found one that actually produces a
    // viewable image is more honest, even though it can't be traced back to
    // a specific line in the source file the way recommended_spp can.
    // Set only for H13/H14/H19/H20 so far (the four curated pbrt-v4-scenes
    // bundles found to render unusably dark at the engine's neutral
    // exposure=1.0) - see their own build_curated_external_pbrt_scene_
    // descriptor() call sites in scene_registry_data.h for the values and
    // how each was determined.
    double recommended_exposure = 1.0;
};

// Dummy sphere light used by scenes that have no explicit light geometry
static inline hittable_list sky_dummy_lights() {
    hittable_list l;
    auto empty_mat = std::shared_ptr<material>();
    l.add(std::make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
    return l;
}

static inline hittable_list no_lights() { return hittable_list{}; }

// paths() is defined at the bottom of this file; forward-declared here so a
// curated builtin entry can register its own .pbrt path the same way
// pbrt_scene_registry::append() does for dynamically-discovered ones.
// Loaded and wire_pbrt_backed_scene are defined here (not at the bottom)
// because build_instanced_spheres_descriptor() below needs them - having
// their only definition up here, rather than forward-declaring them too,
// keeps append() (which reopens this namespace at the bottom of the file)
// from being tempted to redefine its own copy.
namespace pbrt_scene_registry {
    std::map<std::string, std::string>& paths();

    // Loading is deferred AND shared between build_world()/build_lights()/
    // etc (see wire_pbrt_backed_scene below), which is why callers keep
    // this behind a shared_ptr rather than a plain local.
    struct Loaded {
        bool attempted = false;
        pbrt_cpu::BuildResult built;
    };

    // Wires every field of `s` that must be derived from - and stay
    // consistent with - a pbrt-v4 file on disk: the lazy-load/cache,
    // world/lights/sky/punct accessors, recommended_spp, the camera itself,
    // and setup_camera's ortho/spherical/realistic dispatch (read from the
    // file's own Camera directive, never hand-transcribed). Shared by both
    // append() (below), which discovers files at startup, and any curated
    // builtin entry that wraps a specific bundled file under its own id/
    // category/description (build_instanced_spheres_descriptor() today).
    //
    // This exists because "the same file, wired independently by hand
    // twice" is exactly what let those two wirings drift out of sync
    // before this function did: once on SceneDescriptor::requires_files
    // (a curated entry hardcoded true for a file the discovery path had
    // just correctly started reporting false for), and again on the CPU/
    // GPU light-count parity test's assumptions about build_lights()'s
    // shape. One implementation, called from both places, makes that
    // specific class of drift impossible rather than merely caught later.
    //
    // `path` only needs to be valid for the duration of this call: every
    // lambda below captures it BY VALUE (`[state, path]`, `[..., path]`),
    // so each gets its own copy and neither `s` nor the closures it holds
    // reference the caller's string after this function returns.
    inline void wire_pbrt_backed_scene(SceneDescriptor& s,
                                        const pbrt_discover::Discovered& d,
                                        const std::string& path) {
        const auto state = std::make_shared<Loaded>();
        const auto ensure = [state, path]() -> pbrt_cpu::BuildResult& {
            if (!state->attempted) {
                state->attempted = true;
                const pbrt_load::LoadResult r = pbrt_load::loadFile(path);
                if (!r.ok) {
                    std::cerr << "error: " << r.error << "\n";
                } else {
                    for (const pbrt_scene::Warning& w : r.scene.warnings)
                        std::cerr << "warning: " << path << ": " << w.message << "\n";
                    state->built = pbrt_cpu::build(r.scene);
                    // Worth printing rather than inferring from the
                    // picture: instanced geometry that failed to be placed
                    // looks identical to geometry the scene never had.
                    std::cerr << "[pbrt] " << path << ": "
                              << state->built.triangleCount << " triangles, "
                              << state->built.sphereCount << " spheres, "
                              << state->built.instanceCount << " instance placements\n";
                }
            }
            return state->built;
        };

        s.recommended_spp = d.samplesPerPixel;
        s.recommended_max_depth = d.maxDepth;
        s.recommended_integrator = d.integrator;
        s.recommended_sampler = d.samplerType;
        s.recommended_light_sampler = d.lightSamplerType;
        s.camera = CameraConfig{
            d.camera.vfov,
            d.camera.lookfrom[0], d.camera.lookfrom[1], d.camera.lookfrom[2],
            d.camera.lookat[0],   d.camera.lookat[1],   d.camera.lookat[2],
            0.0, 0.0, 0.0,                      // pbrt has no flat background
            CameraMode::Fixed,
            // NOT d.camera.aperture directly - see defocusAngleDegreesFor()'s
            // comment: that field is a world-space lens DIAMETER (pbrt's
            // lensradius*2), not a degrees value, and camera.h's
            // defocus_angle is a full angle in degrees.
            pbrt_flatten::defocusAngleDegreesFor(d.camera, pbrt_flatten::focusDistanceFor(d.camera)),
            // NOT d.camera.focusDistance - see focusDistanceFor()'s comment.
            // Passing pbrt's raw default here made near geometry vanish.
            pbrt_flatten::focusDistanceFor(d.camera),
            // Camera motion blur (pbrt-v4's real ActiveTransform "StartTime"/
            // "EndTime" idiom - see pbrt_flatten::Camera::isAnimated's own
            // comment). Routed through CameraConfig itself, not just
            // setup_camera below, so cpu_scene_camera_is_animated_by_id()
            // (cpu_interface.cpp) - which reads exactly CameraConfig::animated
            // - and main.cpp's --video + animated-camera rejection guard both
            // see a pbrt-authored animated camera the same way they already
            // see the native-demo (D13) one.
            d.camera.isAnimated,
            d.camera.lookfrom1[0], d.camera.lookfrom1[1], d.camera.lookfrom1[2],
            d.camera.lookat1[0],   d.camera.lookat1[1],   d.camera.lookat1[2],
            d.camera.shutterOpen, d.camera.shutterClose
        };

        // A failed load yields an empty world rather than a crash - the
        // error above already said why, and an empty render is a legible
        // symptom.
        s.build_world = [ensure]() {
            pbrt_cpu::BuildResult& b = ensure();
            return b.world ? *b.world : hittable_list{};
        };
        s.build_lights = [ensure]() {
            pbrt_cpu::BuildResult& b = ensure();
            return b.lights ? *b.lights : hittable_list{};
        };
        // nullptr (flat background) unless the scene declared its own
        // LightSource "infinite" - see pbrt_cpu_builder.h's build() for how
        // b.sky gets populated (constant-colour form now; image-backed form
        // once the image resolver lands).
        s.build_sky = [ensure]() -> std::shared_ptr<sky_light> {
            pbrt_cpu::BuildResult& b = ensure();
            return b.sky;
        };
        // nullptr unless the scene declared LightSource "infinite" with a
        // real "point3 portal[4]" - see pbrt_cpu_builder.h's build() for
        // how b.portal gets populated (mutually exclusive with b.sky - see
        // that field's own comment).
        s.build_portal = [ensure]() -> std::shared_ptr<PortalImageInfiniteLightData<double>> {
            pbrt_cpu::BuildResult& b = ensure();
            return b.portal;
        };
        // nullptr (no punctual lights) unless the file declared LightSource
        // point/spot/distant/goniometric/projection - see
        // pbrt_cpu_builder.h's build() for how b.punctLights gets populated.
        s.build_punct = [ensure]() -> std::shared_ptr<punctual_light_list> {
            pbrt_cpu::BuildResult& b = ensure();
            return b.punctLights;
        };
        // nullptr unless the file declared a MediumInterface before its
        // Camera directive - see pbrt_cpu_builder.h's build() for how
        // b.cameraMedium gets populated.
        s.build_camera_medium = [ensure]() -> std::shared_ptr<ambient_medium> {
            pbrt_cpu::BuildResult& b = ensure();
            return b.cameraMedium;
        };

        // CameraConfig cannot express an up vector, but pbrt's LookAt can,
        // and a scene shot in Z-up renders sideways without this.
        // setup_camera runs after every config field is applied, so it is
        // the right place.
        //
        // It is also the only hook that can give a pbrt scene a non-
        // perspective camera: camera_t's alt_ortho_cam/alt_spherical_cam/
        // alt_realistic_cam slots (see camera.h) are the same generic
        // mechanism the built-in OrthographicCamera/SphericalCamera/
        // RealisticCamera scenes already use, populated here from whatever
        // pbrt_flatten.h read off the Camera directive instead of a
        // hardcoded scene. Before this, `Camera "orthographic"` (or
        // "realistic"/"spherical"/"environment") in a loaded .pbrt file was
        // parsed and then silently never looked at again - every pbrt scene
        // rendered through the standard perspective path regardless of what
        // its own Camera directive asked for, with no warning, even though
        // this renderer already has real support for all three.
        const double ux = d.camera.up[0], uy = d.camera.up[1], uz = d.camera.up[2];
        const pbrt_flatten::Camera pcam = d.camera;
        const pbrt_flatten::PixelFilter pfilter = d.filter;
        const bool pregularize = d.regularize;
        const double pcropX0 = d.cropX0, pcropX1 = d.cropX1;
        const double pcropY0 = d.cropY0, pcropY1 = d.cropY1;
        const double pmaxComponentValue = d.maxComponentValue;
        s.setup_camera = [ux, uy, uz, pcam, pfilter, pregularize, pcropX0, pcropX1, pcropY0, pcropY1, pmaxComponentValue, path](camera_t& cam) {
            cam.vup = vec3(ux, uy, uz);
            // PixelFilter - see that struct's own comment (pbrt_flatten.h)
            // for why this is applied unconditionally (unlike sampler_kind)
            // and regardless of camera type, unlike the ortho/spherical/
            // realistic dispatch below.
            cam.filter_kind   = pfilter.kind;
            cam.filter_radius = pfilter.radius;
            cam.filter_B      = pfilter.B;
            cam.filter_C      = pfilter.C;
            cam.filter_sigma  = pfilter.sigma;
            cam.filter_tau    = pfilter.tau;

            // Film "float maxcomponentvalue" - same "applied unconditionally
            // from the scene's own declaration" shape as PixelFilter above
            // (see camera::max_component_value's own comment).
            cam.max_component_value = pmaxComponentValue;

            // Integrator "bool regularize" - same "applied unconditionally"
            // shape as PixelFilter above (see pbrt_discover::Discovered::
            // regularize's own comment).
            cam.regularize = pregularize;

            // Film "cropwindow"/"pixelbounds" - pcropX0/X1/Y0/Y1 are NDC
            // fractions (see pbrt_discover::Discovered::cropX0's own comment
            // on why fractions, not scene-relative pixel indices); converted
            // here against cam.image_width/image_height, which by this point
            // already hold the ACTUAL render resolution (cpu_interface.cpp
            // sets both before calling setup_camera - see camera::crop_x0's
            // own comment) - correct even when a CLI width/height argument
            // overrides what the scene itself declared.
            cam.crop_x0 = static_cast<int>(std::lround(pcropX0 * cam.image_width));
            cam.crop_x1 = static_cast<int>(std::lround(pcropX1 * cam.image_width));
            cam.crop_y0 = static_cast<int>(std::lround(pcropY0 * cam.image_height));
            cam.crop_y1 = static_cast<int>(std::lround(pcropY1 * cam.image_height));

            // Camera motion blur (pbrt-v4's real ActiveTransform "StartTime"/
            // "EndTime" idiom) is wired through CameraConfig itself now (see
            // this scene's own s.camera brace-init above) rather than here -
            // applyCameraConfig() (cpu_interface.cpp) already sets
            // cam.camera_is_animated/lookfrom1/lookat1/shutter_open/close
            // from CameraConfig, and runs before setup_camera at every call
            // site, so duplicating that wiring here would just be two places
            // to keep in sync. Kept in CameraConfig alone so
            // cpu_scene_camera_is_animated_by_id() sees it too (see that
            // brace-init's own comment for why that matters).
            //
            // That said, camera_is_animated only wires real motion blur into
            // the DEFAULT perspective path (camera::initialize()'s own
            // anim_cam_to_world_) - an alt camera model built below gets its
            // own separate AnimatedTransform, attached to whichever
            // alt_*_cam is constructed just below (see cameras.h's own
            // anim_camera_to_world comment for why each alt camera class
            // carries this independently rather than sharing camera::get_ray()'s
            // dispatch, and camera.h's own "alt camera model takes priority"
            // warning - now genuinely honoring motion blur too instead of
            // silently dropping it).
            std::optional<AnimatedTransform> animCtw;
            if (cam.camera_is_animated) {
                const Mat4<double> ctw0 = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    ux, uy, uz);
                const Mat4<double> ctw1 = make_look_at<double>(
                    cam.lookfrom1.x(), cam.lookfrom1.y(), cam.lookfrom1.z(),
                    cam.lookat1.x(),   cam.lookat1.y(),   cam.lookat1.z(),
                    ux, uy, uz);
                animCtw = AnimatedTransform(
                    mat4_to_at_mat44(ctw0), cam.shutter_open,
                    mat4_to_at_mat44(ctw1), cam.shutter_close);
            }

            if (pcam.type == "perspective") {
                // Camera "perspective" "float screenwindow" - previously
                // parsed by pbrt_flatten.h (pcam.hasScreenWindow/
                // screenWindow) but silently discarded here: this function
                // returned before ever consulting it, so only the
                // Orthographic branch below ever honored an explicit
                // screenwindow, even though real pbrt-v4 scenes (anamorphic/
                // cropped/off-center framing) can bind it to a perspective
                // camera just as validly - see camera::has_screen_window's
                // own comment (camera.h) for how it's actually applied.
                if (pcam.hasScreenWindow) {
                    cam.has_screen_window = true;
                    cam.screen_window[0] = pcam.screenWindow[0];
                    cam.screen_window[1] = pcam.screenWindow[1];
                    cam.screen_window[2] = pcam.screenWindow[2];
                    cam.screen_window[3] = pcam.screenWindow[3];
                }
                return;
            }

            Mat4<double> ctw = make_look_at<double>(
                cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                ux, uy, uz);

            if (pcam.type == "orthographic") {
                double xmin, xmax, ymin, ymax;
                if (pcam.hasScreenWindow) {
                    // The scene gave its own extent explicitly - orthographic
                    // has no fov to derive one from any other way, so without
                    // this a scene authored at any scale other than roughly
                    // 1 world unit across needs it to see anything at all.
                    xmin = pcam.screenWindow[0]; xmax = pcam.screenWindow[1];
                    ymin = pcam.screenWindow[2]; ymax = pcam.screenWindow[3];
                } else {
                    compute_screen_window<double>(cam.image_width, cam.image_height,
                                                  xmin, xmax, ymin, ymax);
                }
                cam.alt_ortho_cam = std::make_shared<OrthographicCamera<double>>(
                    xmin, xmax, ymin, ymax, cam.image_width, cam.image_height, ctw);
                cam.alt_ortho_cam->anim_camera_to_world = animCtw;
            } else if (pcam.type == "spherical" || pcam.type == "environment") {
                const auto mapping = (pcam.sphericalMapping == "equalarea")
                    ? SphericalCamera<double>::EqualArea
                    : SphericalCamera<double>::EquiRectangular;
                cam.alt_spherical_cam = std::make_shared<SphericalCamera<double>>(
                    cam.image_width, cam.image_height, mapping, ctw);
                cam.alt_spherical_cam->anim_camera_to_world = animCtw;
            } else if (pcam.type == "realistic") {
                // flatten() already turned a missing lensfile into a warning
                // and fell back to "perspective" (see pbrt_flatten.h), so
                // reaching this branch means a filename really was given -
                // what's left to fail is the file itself, checked here
                // rather than at flatten() time because reading it needs a
                // resolver, and pbrt_flatten.h deliberately has no file
                // access of its own (see pbrt_load.h's header comment).
                std::string lensText;
                if (!pbrt_load::loadFileNear(path, pcam.lensFile, lensText)) {
                    std::cerr << "warning: " << path << ": realistic camera's lensfile '"
                              << pcam.lensFile << "' could not be found; "
                                 "rendering with a perspective camera instead\n";
                    return;
                }
                const std::vector<double> lens = pbrt_load::parseLensFile(lensText);
                if (lens.empty()) {
                    std::cerr << "warning: " << path << ": lensfile '" << pcam.lensFile
                              << "' had no usable lens elements; "
                                 "rendering with a perspective camera instead\n";
                    return;
                }
                // Film half-extents from the diagonal and the image's own
                // aspect ratio - pbrt-v4's own convention, and the same
                // relationship the built-in RealisticCamera scene's
                // hardcoded 18mm x 12mm half-extents satisfy for its 3:2 frame.
                const double aspect = (cam.image_height > 0)
                    ? static_cast<double>(cam.image_width) / cam.image_height : 1.0;
                const double halfY = pcam.filmDiagonalMM / (2.0 * std::sqrt(aspect * aspect + 1.0));
                const double halfX = aspect * halfY;
                cam.alt_realistic_cam = std::make_shared<RealisticCamera<double>>(
                    ctw, halfX, halfY, cam.focus_dist, pcam.apertureDiameterMM, lens);
                cam.alt_realistic_cam->anim_camera_to_world = animCtw;
            }
        };
    }

    // Builds a curated SceneDescriptor for a specific bundled pbrt_scenes/
    // file, under its own real topic id/category/description rather than an
    // auto-generated "I<N>" Custom Scene - the same idea
    // build_instanced_spheres_descriptor() above hand-wrote once, factored
    // out so the ~30 recently-added self-contained example scenes (see their
    // own call sites in get_builtin_scene_registry() below) don't each need
    // their own copy of the same path-resolution/wiring boilerplate. The
    // file still ALSO auto-discovers as a generic "I<N>" Custom Scene too
    // (append() below has no way to know a curated entry exists for it, and
    // doesn't need to - see this function's own callers for why that's
    // intentional, matching the instanced-spheres/F3 precedent).
    //
    // Every caller here passes a self-contained, git-tracked file (no
    // external assets beyond the repo checkout), so requires_files is always
    // false and gpu_compatible always true - a curated entry exists
    // specifically BECAUSE the scene is worth surfacing under a real topic
    // tab, which only makes sense for a scene that's always present and
    // renders on both backends.
    inline SceneDescriptor build_curated_pbrt_scene_descriptor(
            const char* id, int legacy_id, const char* name, const char* category,
            const char* description, const char* performance, const char* filename) {
        // Same search-path walk as build_instanced_spheres_descriptor()'s own
        // comment explains: the working directory differs between running
        // from the repo root (development) and from RayTracer_Package/ (GUI/
        // CLI launch), so this can't be a single hardcoded relative path.
        std::string path;
        for (const std::string& dir : pbrt_discover::defaultSearchPaths()) {
            std::filesystem::path candidate = std::filesystem::path(dir) / filename;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) { path = candidate.string(); break; }
        }
        if (path.empty()) path = std::string("pbrt_scenes/") + filename;  // not found anywhere - fails loudly below

        const pbrt_discover::Discovered d = pbrt_discover::describeFile(path);

        SceneDescriptor s;
        s.id = id;
        s.legacy_id = legacy_id;
        s.name = name;
        s.category = category;
        s.description = description;
        s.performance = performance;
        s.requires_files = false;
        s.gpu_compatible = true;

        wire_pbrt_backed_scene(s, d, path);

        paths()[id] = path;
        return s;
    }

    // Sibling of build_curated_pbrt_scene_descriptor() above for a curated
    // entry whose .pbrt file is NOT git-tracked - a full pbrt-v4-scenes
    // bundle (github.com/mmp/pbrt-v4-scenes), downloaded locally into its
    // own pbrt_scenes/<name>/ subdirectory (see pbrt_scenes/README.md's
    // "Getting scenes" section and .gitignore's own entries for those
    // subdirectories - several GB combined across all of them, and
    // individually licensed non-commercial/no-derivatives in some cases,
    // so committing them isn't an option the way the ~30 small, self-
    // contained example scenes above are). Otherwise identical: same
    // search-path walk (these ALSO live directly under pbrt_scenes/, same
    // as every self-contained example - collectPbrtFiles() already
    // recurses one level into subdirectories, so these files are already
    // where a user's own download naturally lands), same
    // wire_pbrt_backed_scene() wiring, same "the file also auto-discovers
    // as a generic Custom Scene too" acceptance - just requires_files=true,
    // so the GUI's Self-Contained/Requires-External-Files toggle honestly
    // represents that a manual download is needed first, matching how
    // H1-H12 (the hand-written OBJ "whole environment" scenes) already
    // disclose the same thing for their own external assets. A missing
    // file isn't a crash either way - wire_pbrt_backed_scene()'s own
    // ensure() lambda renders an empty world and prints why, the same
    // graceful failure every pbrt-backed scene already has.
    inline SceneDescriptor build_curated_external_pbrt_scene_descriptor(
            const char* id, int legacy_id, const char* name, const char* category,
            const char* description, const char* performance, const char* filename,
            double recommended_exposure = 1.0) {
        std::string path;
        for (const std::string& dir : pbrt_discover::defaultSearchPaths()) {
            std::filesystem::path candidate = std::filesystem::path(dir) / filename;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) { path = candidate.string(); break; }
        }
        if (path.empty()) path = std::string("pbrt_scenes/") + filename;  // not downloaded - fails gracefully below

        const pbrt_discover::Discovered d = pbrt_discover::describeFile(path);

        SceneDescriptor s;
        s.id = id;
        s.legacy_id = legacy_id;
        s.name = name;
        s.category = category;
        s.description = description;
        s.performance = performance;
        s.requires_files = true;
        s.gpu_compatible = true;

        wire_pbrt_backed_scene(s, d, path);

        // Set after wire_pbrt_backed_scene() (which only touches the
        // recommended_spp/max_depth/integrator/sampler/light_sampler fields
        // it derives from `d`) so a caller's curated override always wins.
        s.recommended_exposure = recommended_exposure;

        paths()[id] = path;
        return s;
    }
} // namespace pbrt_scene_registry

// F3: Instanced Spheres. Loads pbrt_scenes/instanced-spheres.pbrt through
// the same lazy-load pbrt pipeline pbrt_scene_registry::append() uses for
// files discovered at startup, but as a curated Geometry-category entry
// with its own id/name/description rather than an auto-generated "I<N>"
// Custom Scene. See that file's own header comment for why it exists: one
// object definition ("cairn": slab + metal ball + glass ball) placed three
// times - plain, rotated, and non-uniformly scaled. The scaled placement's
// spheres only render as true ellipsoids under real instancing (the ray is
// carried into the instance's local space before intersecting); baking the
// placement into world-space spheres cannot express a non-uniform scale
// and would quietly draw round balls instead. GPU needs no new code for
// this scene - build_scene()'s generic pbrt fallback (see the default:
// case in gpu/optix/scene_builder.cpp) already handles arbitrary loaded
// .pbrt files, provided paths()["F3"] resolves, which this sets below.
inline SceneDescriptor build_instanced_spheres_descriptor() {
    // Resolved the same way pbrt_discover::scanDefaultPaths() finds every
    // other .pbrt file: the working directory differs between running from
    // the repo root (development) and from RayTracer_Package/ (GUI/CLI
    // launch), so a single hardcoded relative path is wrong for one of
    // them. This walks the same candidate directories, in the same order,
    // and keeps whichever one actually holds the file.
    static const std::string path = []() -> std::string {
        for (const std::string& dir : pbrt_discover::defaultSearchPaths()) {
            std::filesystem::path candidate =
                std::filesystem::path(dir) / "instanced-spheres.pbrt";
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec))
                return candidate.string();
        }
        return "pbrt_scenes/instanced-spheres.pbrt";  // not found anywhere - fails loudly below
    }();
    // Same lightweight header-only parse pbrt_discover::scanDefaultPaths()
    // uses for every other file - not the full geometry load, just camera/
    // resolution/sample-count, cheap enough to do unconditionally here.
    // This is what lets wire_pbrt_backed_scene() below derive the camera
    // and recommended_spp from the file itself instead of a second,
    // hand-transcribed copy of the same numbers that could silently drift
    // if the file were ever edited.
    static const pbrt_discover::Discovered d = pbrt_discover::describeFile(path);

    SceneDescriptor s;
    s.id = "F3";
    s.legacy_id = 71;
    s.name = SceneNames::InstancedSpheres;
    s.category = SceneCategories::Geometry;
    s.description =
        "Object instancing with spheres: one 'cairn' (slab + metal ball + "
        "glass ball) defined once, placed three times - plain, rotated, and "
        "non-uniformly scaled. The scaled placement's spheres read as true "
        "ellipsoids, which only real instancing (not baked world-space "
        "placement) can produce";
    s.performance = "Fast";
    // pbrt_scenes/instanced-spheres.pbrt is git-tracked and self-contained
    // (no external assets beyond the repo checkout) - the same file's
    // auto-discovered "I<N>" Custom Scene copy correctly reports false via
    // pbrt_discover::Discovered::nested (see that field's comment), so this
    // curated entry needs to agree rather than hardcode the opposite.
    s.requires_files = false;
    s.gpu_compatible = true;

    // Camera, recommended_spp, lazy-load/cache, and world/lights/sky/punct
    // accessors all come from the file itself - see this function's own
    // comment for why sharing this with append()'s per-discovered-file
    // wiring, instead of each hand-writing its own copy, is the point.
    pbrt_scene_registry::wire_pbrt_backed_scene(s, d, path);

    pbrt_scene_registry::paths()["F3"] = path;
    return s;
}

#include "scene_registry_data.h"


inline int builtin_scene_count() {
    return static_cast<int>(get_builtin_scene_registry().size());
}

// -----------------------------------------------------------------------
// Loaded scenes
// -----------------------------------------------------------------------
// Turns each .pbrt file found on disk into a SceneDescriptor appended after
// the built-ins. Doing it here, in the one function every consumer already
// reads through, is what makes loaded scenes appear in the CLI, in
// scene_metadata.dll and in the GUI's scene list without any of those three
// knowing that pbrt exists.
//
// GEOMETRY IS LOADED WHEN THE SCENE IS RENDERED, NOT WHEN IT IS LISTED
// --------------------------------------------------------------------
// pbrt_discover only read each file's header, so listing a hundred scenes
// costs a hundred small reads. The full load happens inside build_world(),
// the first time that particular scene is actually rendered.
//
// build_world() and build_lights() are separate calls against one scene, and
// cpu_interface.cpp invokes both, so the result is cached: without that, a
// render would parse and triangulate the entire file twice.
namespace pbrt_scene_registry {

std::map<std::string, std::string>& paths();   // defined below, used by append()

inline void append(std::vector<SceneDescriptor>& registry) {
    std::vector<pbrt_discover::Discovered> found = pbrt_discover::scanDefaultPaths();

    // Bundled (non-nested) scenes are numbered first, in their own stable
    // alphabetical group; downloaded collections (nested) are numbered
    // after. stable_partition preserves each group's existing alphabetical
    // order (scanDefaultPaths() already sorted the whole list by path)
    // while making a bundled scene's id depend only on what else is
    // bundled - never on what unrelated downloaded collections happen to
    // be sitting alongside it locally. Confirmed empirically: without
    // this, a bundled example scene got id "I14" instead of "I1" purely
    // because other locally-downloaded pbrt collections happened to sort
    // before it on that machine. A downloaded collection's own ids are
    // still free to shift when collections are added/removed - expected,
    // since those genuinely aren't the same fixed set on every machine.
    std::stable_partition(found.begin(), found.end(),
        [](const pbrt_discover::Discovered& d) { return !d.nested; });

    // SceneDescriptor holds `const char*`, so the strings have to outlive the
    // registry. A deque is used rather than a vector because it never
    // reallocates its elements, so pointers taken here stay valid as more
    // scenes are appended.
    static std::deque<std::string> names;
    static std::deque<std::string> descriptions;

    // legacy_id keeps counting up from one past the HIGHEST legacy_id any
    // builtin scene actually uses (see SceneDescriptor::legacy_id's comment)
    // - values past every `case N:` in scene_builder.cpp's switch always
    // miss and land on `default:`, which is all that's required of it for
    // pbrt scenes.
    //
    // NOT builtin_scene_count(): the builtin registry has a permanent gap at
    // legacy_id 53 (G16 was removed - see scene_registry_tests.cpp's
    // RegistryHasExpectedCount comment), so the array's SIZE (78) undercounts
    // the highest id in use by one - the last builtin entry (H9 Gallery)
    // itself has legacy_id 78. Starting the counter at builtin_scene_count()
    // therefore handed the first pbrt-loaded scene ("I1") legacy_id=78 too,
    // a direct collision with Gallery's - build_scene()'s switch matched
    // `case 78:` before ever reaching `default:`, so GPU silently rendered
    // Gallery's own geometry (a real photographic gallery of framed
    // paintings) for whatever the first discovered .pbrt file was, while CPU
    // (which never switches on legacy_id - see that field's own comment)
    // rendered the correct file. Computed as a max here, not hardcoded,
    // so any future gap can't reopen the same collision silently.
    //
    // user_number is the NEW id's counter, independent and always 1-based -
    // all pbrt scenes share category CustomScenes (letter derived via
    // SceneCategories::letter_for_category() below, not hardcoded - see
    // that function's own comment for why), so this is simply "the Nth
    // pbrt scene loaded this run", with no static CustomScenes entries in
    // the builtin registry to continue from today.
    int legacy_id = 0;
    for (const SceneDescriptor& b : get_builtin_scene_registry())
        legacy_id = std::max(legacy_id, b.legacy_id + 1);
    int user_number = 1;
    for (const pbrt_discover::Discovered& d : found) {
        // A file that will not even parse its header is skipped rather than
        // listed: offering a scene that cannot possibly render is worse than
        // not offering it. The warning goes to stderr so the CLI and the GUI
        // log both surface it.
        if (!d.ok) {
            std::cerr << "warning: skipping " << d.path << ": " << d.error << "\n";
            continue;
        }

        names.push_back(d.name);
        descriptions.push_back(
            "Loaded from " + d.path + " (pbrt-v4 scene). Camera, resolution and "
            "sample count come from the file itself; geometry is read on first "
            "render, so the first frame of a large scene starts slowly.");

        SceneDescriptor s;
        // Uses SceneCategories::letter_for_category() rather than a
        // hardcoded 'I', so this can't silently drift from
        // SceneCategories::kAll's declared order (see that function's
        // comment in scene_descriptor.h).
        s.id = std::string(1, SceneCategories::letter_for_category(SceneCategories::CustomScenes))
             + std::to_string(user_number++);
        s.legacy_id = legacy_id++;
        s.name = names.back().c_str();
        s.category = SceneCategories::CustomScenes;
        s.description = descriptions.back().c_str();
        // Honest rather than flattering: a .pbrt file can hold anything from
        // three triangles to ten million, and nothing in the header says
        // which. "Unknown" is the truthful answer at this point.
        s.performance = "Unknown";
        // Not unconditionally true: a hand-authored scene bundled directly
        // in pbrt_scenes/ (git-tracked, no assets beyond the repo checkout)
        // is exactly as self-contained as a builtin scene, but a scene from
        // a downloaded per-scene-folder collection (gitignored, possibly
        // hundreds of MB of its own geometry/textures) is not - see
        // pbrt_discover::Discovered::nested's comment for how that
        // distinction is made.
        s.requires_files = d.nested;
        // gpu/optix/pbrt_gpu_builder.h consumes the same FlatScene this does,
        // so both backends render the same file, and they now agree on what
        // they can sample: sphere, parallelogram AND individual-triangle area
        // lights (GpuLightKind), plus instanced triangles and spheres. The GPU
        // used to be the weaker of the two - it could only sample lights that
        // were spheres or parallelograms - which is no longer true and is why
        // this no longer carries a caveat.
        s.gpu_compatible = true;

        // Camera, recommended_spp, lazy-load/cache, and world/lights/sky/
        // punct accessors all come from the file itself - see
        // wire_pbrt_backed_scene's own comment for why this is shared with
        // build_instanced_spheres_descriptor()'s curated entry instead of
        // each hand-writing its own copy.
        wire_pbrt_backed_scene(s, d, d.path);

        paths()[s.id] = d.path;
        registry.push_back(s);
    }
}

// The .pbrt file each loaded scene came from, keyed by scene id. Kept beside
// the registry rather than inside SceneDescriptor because only loaded scenes
// have one, and the GPU builder is the only consumer: it is handed a scene id
// and has to find the same file the CPU side found. Re-scanning the directory
// there would be a second implementation of the search order, free to drift.
inline std::map<std::string, std::string>& paths() {
    static std::map<std::string, std::string> byId;
    return byId;
}

} // namespace pbrt_scene_registry

// The full registry: built-in scenes, then whatever was found on disk.
inline const std::vector<SceneDescriptor>& get_scene_registry() {
    static const std::vector<SceneDescriptor> registry = []() {
        std::vector<SceneDescriptor> all = get_builtin_scene_registry();
        pbrt_scene_registry::append(all);
        return all;
    }();
    return registry;
}

// Lookup by id -- returns nullptr if not found
inline const SceneDescriptor* find_scene(const std::string& id) {
    for (const auto& s : get_scene_registry())
        if (s.id == id) return &s;
    return nullptr;
}

inline int scene_count() {
    return static_cast<int>(get_scene_registry().size());
}
