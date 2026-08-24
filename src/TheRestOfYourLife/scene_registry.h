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
};

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
            pbrt_flatten::focusDistanceFor(d.camera)
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
        // nullptr (no punctual lights) unless the file declared LightSource
        // point/spot/distant/goniometric/projection - see
        // pbrt_cpu_builder.h's build() for how b.punctLights gets populated.
        s.build_punct = [ensure]() -> std::shared_ptr<punctual_light_list> {
            pbrt_cpu::BuildResult& b = ensure();
            return b.punctLights;
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
        s.setup_camera = [ux, uy, uz, pcam, pfilter, path](camera_t& cam) {
            cam.vup = vec3(ux, uy, uz);
            // PixelFilter - see that struct's own comment (pbrt_flatten.h)
            // for why this is applied unconditionally (unlike sampler_kind)
            // and regardless of camera type, unlike the ortho/spherical/
            // realistic dispatch below.
            cam.filter_kind  = pfilter.kind;
            cam.filter_B     = pfilter.B;
            cam.filter_C     = pfilter.C;
            cam.filter_sigma = pfilter.sigma;
            cam.filter_tau   = pfilter.tau;
            if (pcam.type == "perspective") return;

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
            } else if (pcam.type == "spherical" || pcam.type == "environment") {
                const auto mapping = (pcam.sphericalMapping == "equalarea")
                    ? SphericalCamera<double>::EqualArea
                    : SphericalCamera<double>::EquiRectangular;
                cam.alt_spherical_cam = std::make_shared<SphericalCamera<double>>(
                    cam.image_width, cam.image_height, mapping, ctw);
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

// -----------------------------------------------------------------------
// Scenes loaded from .pbrt files (see append_pbrt_scenes below)
// -----------------------------------------------------------------------

// The scenes compiled into this binary. Everything the full registry holds
// beyond these came from a .pbrt file found on disk at startup.
inline const std::vector<SceneDescriptor>& get_builtin_scene_registry() {
    static const std::vector<SceneDescriptor> registry = {
        {
            "A1", 0, SceneNames::CornellBox, SceneCategories::Basics,
            "Classic Cornell box with glass sphere and aluminum box",
            "Medium", 100, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            "A2", 1, SceneNames::BouncingSpheres, SceneCategories::Basics,
            "Random spheres with checker ground (In One Weekend final)",
            "Slow", 100, false, true,
            // defocus_angle/focus_dist: the book's own final-render values
            // for this exact scene - a subtle depth-of-field "beauty shot"
            // focused on the 3 hero spheres near the origin, without
            // redesigning the iconic grid composition itself.
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00, CameraMode::Fixed, 0.6, 10.0 },
            build_bouncing_spheres,
            sky_dummy_lights
        },
        {
            "A3", 2, SceneNames::CheckeredSpheres, SceneCategories::Basics,
            "Two spheres with procedural checker texture",
            "Fast", 100, false, true,
            // Warm sunset-ish flat background instead of generic sky-blue -
            // fits the "planet" motif better and gives the new accent
            // spheres something to contrast against.
            { 20, 13, 2, 3,  0, 0, 0,  0.90, 0.75, 0.55 },
            build_checkered_spheres,
            sky_dummy_lights
        },
        {
            "A4", 3, SceneNames::Earth, SceneCategories::Basics,
            "Globe with earth texture mapping (requires earthmap.jpg)",
            "Fast", 100, true, true,
            // vfov widened 20->25 to leave room for the new moon accent
            // sphere near the frame edge without cropping it.
            { 25, 0, 0, 12,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_earth,
            build_earth_lights
        },
        {
            "A5", 4, SceneNames::PerlinSpheres, SceneCategories::Basics,
            "Spheres with Perlin noise marble texture",
            "Fast", 100, false, true,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_perlin_spheres,
            build_perlin_spheres_lights
        },
        {
            "A6", 5, SceneNames::ColoredQuads, SceneCategories::Basics,
            "Five colored quad primitives",
            "Fast", 100, false, true,
            { 80, 0, 0, 9,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_quads,
            build_quads_lights
        },
        {
            "A7", 6, SceneNames::SimpleLight, SceneCategories::Basics,
            "Perlin spheres with emissive light sources",
            "Fast", 100, false, true,
            { 20, 26, 3, 6,  0, 2, 0,  0, 0, 0 },
            build_simple_light,
            no_lights
        },
        {
            "A8", 7, SceneNames::CornellSmoke, SceneCategories::Basics,
            "Cornell box with volumetric fog",
            "Slow", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_smoke,
            build_cornell_smoke_lights
        },
        {
            "A9", 8, SceneNames::FinalScene, SceneCategories::Basics,
            "Complex scene from The Next Week",
            "Very Slow", 500, false, true,
            // Subtle deep ambient instead of pure black - the box-grid
            // ground and negative space used to render into a stark void.
            { 40, 478, 278, -600,  278, 278, 0,  0.03, 0.025, 0.02 },
            build_final_scene,
            build_final_scene_lights
        },
        {
            "B1", 9, SceneNames::RoughMetalSpheres, SceneCategories::Materials,
            "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
            "Medium", 200, false, true,
            { 42, 0, 2.7, 17,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_rough_metal_spheres,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-6,6,-4), vec3(12,0,0), vec3(0,0,8), empty_mat));
                return l;
            }
        },
        {
            "B2", 10, SceneNames::CornellRoughMetal, SceneCategories::Materials,
            "Cornell box with rough aluminum box and rough gold sphere",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0.05, 0.055, 0.07, CameraMode::UserControlled },
            build_cornell_rough_metal,
            build_cornell_box_lights
        },
        {
            "B3", 11, SceneNames::CornellRoughGlass, SceneCategories::Materials,
            "Cornell box with a GGX rough-dielectric sphere (pbrt-v4 RoughDielectricBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_rough_glass,
            build_cornell_box_lights
        },
        {
            "B4", 12, SceneNames::CornellConductor, SceneCategories::Materials,
            "Cornell box with polished gold sphere and aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0.05, 0.055, 0.07, CameraMode::UserControlled },
            build_cornell_conductor,
            build_cornell_box_lights
        },
        {
            "B5", 13, SceneNames::CornellCoatedDiffuse, SceneCategories::Materials,
            "Cornell box with blue coated-diffuse sphere and red coated-diffuse box (pbrt-v4 CoatedDiffuseBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_coated_diffuse,
            build_cornell_box_lights
        },
        {
            "B6", 14, SceneNames::CornellThinGlass, SceneCategories::Materials,
            "Cornell box with a vertical thin-glass panel, analytic multi-bounce Fresnel (pbrt-v4 ThinDielectricBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_thin_glass,
            build_cornell_thin_glass_lights
        },
        {
            "B7", 15, SceneNames::CornellCoatedConductor, SceneCategories::Materials,
            "Cornell box with lacquered-gold sphere and lacquered-copper box (pbrt-v4 CoatedConductorBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0.05, 0.055, 0.07, CameraMode::UserControlled },
            build_cornell_coated_conductor,
            build_cornell_box_lights
        },
        {
            "B8", 16, SceneNames::CornellWaxSlab, SceneCategories::Materials,
            "Cornell box with a wax sphere that diffusely reflects and transmits light (pbrt-v4 DiffuseTransmissionBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_wax_slab,
            build_cornell_box_lights
        },
        {
            "B9", 17, SceneNames::CornellCrystal, SceneCategories::Materials,
            "Cornell box with a crystal sphere using Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_crystal,
            build_cornell_box_lights
        },
        {
            "B10", 18, SceneNames::PrincipledShowcase, SceneCategories::Materials,
            "Row of spheres from matte plastic to metallic with clearcoat (pbrt-v4 PrincipledBxDF)",
            "Medium", 200, false, true,
            { 45, 0, 2.7, 17,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_principled_showcase,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-7,7,-5), vec3(14,0,0), vec3(0,0,10), empty_mat));
                return l;
            }
        },
        {
            "B11", 19, SceneNames::HairFibers, SceneCategories::Materials,
            "Sphere cluster with hair/fur fiber scattering (pbrt-v4 HairBxDF)",
            "Medium", 200, false, true,
            { 45, 0, 2.5, 14,  0, 1, 0,  0.05, 0.05, 0.07 },
            build_hair_fibers,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-5,6,-5), vec3(10,0,0), vec3(0,0,7), empty_mat));
                return l;
            }
        },
        {
            "B12", 20, SceneNames::NormalMappedCornell, SceneCategories::Materials,
            "Cornell box with procedural bump-mapped back wall and normal-mapped sphere (pbrt-v4 NormalMap/BumpMap)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_normal_mapped_cornell,
            build_cornell_box_lights
        },
        {
            "B13", 21, SceneNames::SubsurfaceSlab, SceneCategories::Materials,
            "Cornell box with translucent wax slab and jade sphere using subsurface-like scattering",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0.05, 0.055, 0.07, CameraMode::UserControlled },
            build_subsurface_slab,
            build_cornell_box_lights
        },
        {
            "D1", 22, SceneNames::DepthOfField, SceneCategories::Cameras,
            "Row of spheres with defocus blur showing depth-of-field from the thin-lens camera model",
            "Medium", 200, false, true,
            { 62, 0, 2, 9,  0, 1, 0,  0.70, 0.80, 1.00, CameraMode::Fixed, 10.0, 9.0 },
            build_depth_of_field,
            sky_dummy_lights
        },
        {
            "F1", 23, SceneNames::BilinearPatchScene, SceneCategories::Geometry,
            "Cornell box with curved bilinear patch saddle surface (pbrt-v4 BilinearPatch shape)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_bilinear_patch_scene,
            build_bilinear_patch_lights
        },
        // ---- pbrt-v4 light / camera / medium showcase ----
        {
            "C1", 24, SceneNames::HdriSky, SceneCategories::Lights,
            "Open scene lit by a procedural gradient sky (pbrt-v4 ImageInfiniteLight / sky_light)",
            "Medium", 200, false, true,
            { 42, 0, 2.3, 15,  0, 1, 0,  0, 0, 0 },
            build_hdri_sky_world,
            no_lights,
            build_hdri_sky,
            nullptr
        },
        {
            "C2", 25, SceneNames::SpotlightCornell, SceneCategories::Lights,
            "Cornell box lit by a spotlight with smooth penumbra (pbrt-v4 SpotLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_spotlight_cornell,
            no_lights,
            nullptr,
            build_spotlight_punct
        },
        {
            "C3", 26, SceneNames::DistantLightCornell, SceneCategories::Lights,
            "Cornell box lit by a parallel sun-like distant light (pbrt-v4 DistantLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_distant_light_cornell,
            no_lights,
            nullptr,
            build_distant_light_punct
        },
        {
            "C4", 27, SceneNames::PointLightCornell, SceneCategories::Lights,
            "Cornell box lit by a single overhead point light with 1/r^2 falloff (pbrt-v4 PointLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_point_light_cornell,
            no_lights,
            nullptr,
            build_point_light_punct
        },
        {
            "C5", 28, SceneNames::GoniometricLight, SceneCategories::Lights,
            "Cornell box lit by a goniometric (IES-profile) point light (pbrt-v4 GoniometricLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_goniometric_light_scene,
            no_lights,
            nullptr,
            build_goniometric_punct
        },
        {
            "C6", 29, SceneNames::ProjectionLight, SceneCategories::Lights,
            "Cornell box with a slide-projector beam casting a checkerboard pattern (pbrt-v4 ProjectionLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_projection_light_scene,
            no_lights,
            nullptr,
            build_projection_punct
        },
        {
            "E1", 30, SceneNames::HomogeneousMedium, SceneCategories::Volumes,
            "Cornell box filled with a homogeneous scattering fog (pbrt-v4 HomogeneousMedium / HenyeyGreenstein)",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_homogeneous_medium_scene,
            build_homogeneous_medium_lights
        },
        {
            "E2", 31, SceneNames::CloudMedium, SceneCategories::Volumes,
            "Open scene with a procedural Perlin-noise cloud volume (pbrt-v4 CloudMedium)",
            "Slow", 300, false, true,
            // vfov widened/camera pulled back (was 20 deg at (0,5,20)) - the
            // cloud's world AABB (x:[-4,4]) alone overflowed that framing,
            // and the two "context" spheres at x=+-5 were entirely outside
            // it. See build_cloud_medium_scene's comment for the cloud's
            // actual extent.
            { 40, 0, 4, 26,  0, 2, 0,  0.5, 0.7, 1.0 },
            build_cloud_medium_scene,
            sky_dummy_lights
        },
        {
            "E3", 69, SceneNames::DielectricMediumShowcase, SceneCategories::Volumes,
            "Three glass spheres containing colored internal fog at varying density - dielectric surface + participating medium combined (pbrt-v4 style)",
            "Medium", 200, false, true,
            // vfov/lookfrom chosen so all 3 spheres (x:[-5.5,5.5] incl.
            // radius) fit comfortably in frame - see build_dielectric_medium_scene.
            { 40, 0, 3, 18,  0, 1.5, 0,  0.5, 0.7, 1.0 },
            build_dielectric_medium_scene,
            sky_dummy_lights
        },
        {
            "E4", 70, SceneNames::RgbGridMedium, SceneCategories::Volumes,
            "Heterogeneous nebula with an independent per-voxel R/G/B scattering grid (pbrt-v4 RGBGridMedium)",
            "Slow", 300, false, true,
            // Pulled back further than E2's cloud camera - this box is
            // taller (world y:[1,5] vs E2's [1,4]) and the context spheres
            // sit further out (x:+-6) - see build_rgb_grid_medium_scene.
            { 45, 0, 5, 30,  0, 3, 0,  0.5, 0.7, 1.0 },
            build_rgb_grid_medium_scene,
            sky_dummy_lights
        },
        {
            "D2", 32, SceneNames::OrthographicCamera, SceneCategories::Cameras,
            "Geometric showcase rendered with an orthographic (parallel-projection) camera (pbrt-v4 OrthographicCamera)",
            "Fast", 100, false, true,
            { 30, 0, 10, 20,  0, 1, 0,  0, 0, 0 },
            build_ortho_camera_scene,
            no_lights,
            build_ortho_sky,
            nullptr,
            [](camera_t& cam) {
                // cam.lookfrom/lookat are already set (from CameraConfig, or
                // overridden by the caller - e.g. a video-mode frame's
                // animated position) by the time setup_camera() runs - see
                // cpu_interface.cpp. Read them here instead of hardcoding the
                // registry's default (0,3,12)/(0,1,0), so this alt camera
                // actually moves for video mode instead of silently staying
                // frozen on every frame.
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    0, 1, 0     // up
                );
                double xmin, xmax, ymin, ymax;
                compute_screen_window<double>(cam.image_width, cam.image_height,
                                              xmin, xmax, ymin, ymax);
                // Screen-window scale (was 8, uniform in x AND y): the row of
                // spheres needs wide horizontal coverage, but scaling y by
                // the same large factor pushed ray origins for the bottom
                // rows below the giant ground sphere's surface (radius 100,
                // centered at y=-100) - those rays hit the sphere from
                // inside/behind, the Lambertian material's cosine check
                // failed, and the path terminated with zero radiance,
                // rendering as a solid black region with a curved boundary
                // (traced from that sphere's silhouette). Camera moved
                // higher/farther back (was lookfrom (0,3,12)) so a scale
                // that's still wide enough for the spheres doesn't dip
                // below ground.
                cam.alt_ortho_cam = std::make_shared<OrthographicCamera<double>>(
                    xmin*5, xmax*5, ymin*5, ymax*5,
                    cam.image_width, cam.image_height,
                    ctw
                );
            }
        },
        {
            "D3", 33, SceneNames::SphericalCamera, SceneCategories::Cameras,
            "360-degree equirectangular panorama from a spherical camera (pbrt-v4 SphericalCamera)",
            "Medium", 200, false, true,
            { 90, 0, 1, 0,  0, 0, 0,  0, 0, 0 },
            build_spherical_camera_scene,
            no_lights,
            build_spherical_sky,
            nullptr,
            [](camera_t& cam) {
                // SphericalCamera captures the full 360-degree sphere around
                // its origin, so its orientation doesn't gate a field of
                // view the way lookat does for other cameras - this scene's
                // own registry entry sets lookat=(0,0,0) directly below
                // lookfrom=(0,1,0), which would make the polar (up) axis of
                // the equirect mapping parallel to world up, a degenerate
                // input to make_look_at (cross(up,forward) == 0). Use a
                // fixed horizontal forward reference (+Z, matching the old
                // hardcoded identity transform's own forward axis, so the
                // panorama's default orientation is unchanged) instead, so
                // it stays stable and well-defined regardless of the
                // scene's lookat value; only the origin needs to track
                // cam.lookfrom (previously hardcoded to the world origin
                // via an identity transform, so video mode's animated
                // camera position had no effect and every frame was
                // identical).
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(),     cam.lookfrom.z(),
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z() + 1.0,
                    0, 1, 0     // up
                );
                cam.alt_spherical_cam = std::make_shared<SphericalCamera<double>>(
                    cam.image_width, cam.image_height,
                    SphericalCamera<double>::EquiRectangular,
                    ctw
                );
            }
        },
        {
            "B14", 34, SceneNames::MeasuredBrdf, SceneCategories::Materials,
            "Sphere cluster with measured BRDF material using tabulated RGL data (pbrt-v4 MeasuredBxDF)",
            "Medium", 200, false, true,
            { 42, 0, 3.2, 17,  0, 1, 0,  0, 0, 0 },
            build_measured_brdf_scene,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 1.5,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "C7", 35, SceneNames::PortalInfiniteLight, SceneCategories::Lights,
            "Room scene with a portal window sampling the sky through a planar quad (pbrt-v4 PortalImageInfiniteLight)",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_portal_light_scene,
            no_lights,
            build_portal_sky
        },
        {
            "D4", 36, SceneNames::RealisticCamera, SceneCategories::Cameras,
            "Spheres rendered through a thin-lens with realistic lens-element bokeh (pbrt-v4 RealisticCamera)",
            "Medium", 200, false, true,
            { 50, 1.65, 1.07, -6.85,  1.4, 1, 5.5,  0, 0, 0 },
            build_realistic_camera_scene,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,5), 2,
                      std::shared_ptr<material>()));
                return l;
            },
            nullptr,
            nullptr,
            [](camera_t& cam) {
                // Double-Gauss lens
                // Lens data from pbrt-v4 sample scene: dgauss.22deg.dat (simplified)
                // Format: curvature_mm, thickness_mm, ior, aperture_mm
                std::vector<double> lens = {
                     35.98738,  1.21638, 1.54,  23.716,
                     11.69718,  9.9957,  1.0,   17.996,
                     13.08714, 15.9948,  1.77,  12.364,
                    -22.63294,  2.7757,  1.617, 9.812,
                      0.0,      2.75,    0.0,   7.4,     // aperture stop
                     36.3581,   8.9722,  1.617, 12.7,
                    -17.8595,   1.2,     1.0,   12.7,
                    100.0,      2.9804,  1.567, 14.478,
                    -24.5656,   0.0,     1.0,   15.0
                };
                // camera_to_world: read cam.lookfrom/lookat (already set from
                // CameraConfig, or overridden by the caller - e.g. a
                // video-mode frame's animated position - by the time
                // setup_camera() runs, see cpu_interface.cpp) instead of the
                // registry's original default (0,2,-2)/(0,1,5) directly, so
                // this alt camera actually moves for video mode instead of
                // silently staying frozen on every frame.
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    0, 1,  0    // up
                );
                // Film half-extents shrunk from a full 35mm frame (18/12mm) to
                // 3.0/2.0mm. This "simplified" 9-element lens prescription
                // (curvatures/thicknesses copied from pbrt-v4's dgauss sample
                // but hand-trimmed) has a genuine back-focal-distance of only
                // ~4mm - far short of what a real Double-Gauss 35mm-format
                // lens needs (~30-40mm) - so its actual working image circle
                // is much smaller than a 35mm frame. At 18/12mm, over 70% of
                // the film radius fell entirely outside every lens element's
                // combined aperture (confirmed by sweeping exit-pupil-bounds
                // degeneracy directly against this exact lens data - neither
                // focus_distance nor a wider requested aperture changed it),
                // rendering as solid black outside a small central disc.
                // Verified empirically: this lens gives full, non-vignetted
                // coverage starting around a 3.6mm RADIUS from the film
                // center; 3.0/2.0mm keeps every corner (sqrt(3.0^2+2.0^2)
                // = 3.6mm) within that verified-safe radius - see the
                // half-width/half-height comment below for why the corners
                // specifically matter here.
                //
                // Two more real bugs were found and fixed alongside the film
                // size, both upstream of this file:
                //  1. src/TheRestOfYourLife/camera.h's get_ray() was
                //     discarding RealisticCamera::generate_ray()'s `weight`
                //     (the pbrt-v4 cos^4(theta)/(pdf*LensRearZ^2) exposure
                //     factor - see cameras.h's class comment). With this
                //     lens's tiny rear-Z, 1/LensRearZ^2 is a large multiplier;
                //     dropping it made every CPU RealisticCamera render come
                //     out catastrophically underexposed regardless of camera
                //     position (GPU already applied the equivalent weight
                //     correctly - see optix_device_helpers.h's
                //     generate_primary_ray). Fixed by threading the weight
                //     out of get_ray() and multiplying it into the sample
                //     before filter accumulation.
                //  2. The scene's 5 spheres sit at x=0, differing only in
                //     depth (z=2..8) - i.e. exactly on the camera's original
                //     on-axis viewing line. An opaque near sphere on that
                //     same line fully occludes the ones behind it from every
                //     lens-aperture sample, so the original dead-on
                //     lookfrom/lookat rendered as one oversized near sphere
                //     with the rest invisible no matter the distance. Fixed
                //     by moving the camera to an oblique angle (found via a
                //     temporary ray-vs-known-sphere projection sweep, see
                //     git history for tests/unit/realistic_camera_tests.cpp)
                //     so all 5 spheres are visible side by side with
                //     depth-increasing defocus blur - the actual bokeh demo
                //     this scene is meant to show.
                cam.alt_realistic_cam = std::make_shared<RealisticCamera<double>>(
                    ctw,
                    3.0,    // film half-width mm
                    2.0,    // film half-height mm (3:2 aspect) - chosen so the frame's
                            // CORNER radius (sqrt(hx^2+hy^2) = 3.6mm) sits right at the
                            // empirically-verified non-vignetted radius; the corners of
                            // a rectangular film reach further than the half-width/
                            // half-height alone, and a larger size here left them a
                            // grainy high-variance patch (a handful of valid-but-near-
                            // degenerate exit-pupil samples getting a very small pdf,
                            // hence a very large 1/pdf weight spike).
                    12.4,   // focus distance meters
                    8.0,    // aperture diameter mm
                    lens,
                    512     // pupil samples
                );
            }
        },
        // D5-D8: the exact same classic Cornell box as A1 (build_cornell_box /
        // build_cornell_box_lights, backed by src/shared/cornell_box_data.h),
        // rendered by each of D1-D4's camera models in turn. Keeping the scene
        // fixed and only varying the camera makes the actual differences
        // between the four models (defocus blur, parallel projection, 360
        // panorama, real lens bokeh) directly comparable, which D1-D4's own
        // bespoke per-scene geometry doesn't support. D1-D4 are left
        // unchanged - these are additive, not replacements.
        {
            "D5", 65, SceneNames::DepthOfFieldCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D6-D8) with defocus blur from the thin-lens perspective camera",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::Fixed, 2.0, 800.0 },
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            "D6", 66, SceneNames::OrthographicCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5/D7/D8) rendered with a parallel-projection orthographic camera",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0 },
            build_cornell_box,
            build_cornell_box_lights,
            nullptr,
            nullptr,
            [](camera_t& cam) {
                // cam.lookfrom/lookat already set from CameraConfig (or a
                // video-mode override) by the time setup_camera() runs.
                //
                // Same lookfrom/lookat as A1/D5 (dead-on, straight down the
                // box's own z-axis). An angled "isometric dollhouse" vantage
                // was tried first to also reveal a side wall, but this
                // codebase's camera_to_world convention (see cameras.h's
                // Mat4/make_look_at - the ray direction ends up built from
                // the basis vectors' individual components rather than
                // behaving like a textbook "rotate camera-space forward into
                // world space" transform) makes an oblique orthographic
                // frame's actual on-screen position hard to predict by hand;
                // empirical search (see git history for this file and
                // tests/unit/realistic_camera_tests.cpp) never found an
                // angled position that centered the box as reliably as this
                // dead-on one does. The trade-off is real - a dead-on
                // orthographic view shows only the back wall/floor/ceiling,
                // not the red/green side walls - but it still demonstrates
                // parallel projection's defining trait unambiguously: unlike
                // D5's converging perspective lines, this box's edges stay
                // perfectly parallel no matter how far from center they are.
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    0, 1, 0     // up
                );
                double xmin, xmax, ymin, ymax;
                compute_screen_window<double>(cam.image_width, cam.image_height,
                                              xmin, xmax, ymin, ymax);
                // The box spans 555 world units per side; a screen-window
                // half-extent of ~320 (with the unscaled +-1/+-aspect window
                // from compute_screen_window) fills close to half the frame
                // with the box, centered - confirmed via a ray-vs-AABB sweep
                // (see git history for tests/unit/realistic_camera_tests.cpp)
                // rather than by eye, since this codebase's camera_to_world
                // convention doesn't map screen position to world position
                // as directly as the naive "u/v offset from lookfrom" model
                // suggests (see this lambda's opening comment).
                cam.alt_ortho_cam = std::make_shared<OrthographicCamera<double>>(
                    xmin*320, xmax*320, ymin*320, ymax*320,
                    cam.image_width, cam.image_height,
                    ctw
                );
            }
        },
        {
            "D7", 67, SceneNames::SphericalCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5/D6/D8), toured from its center as a 360-degree equirectangular panorama",
            "Medium", 200, false, true,
            { 90, 278, 278, 278,  278, 278, 279,  0, 0, 0 },
            build_cornell_box,
            build_cornell_box_lights,
            nullptr,
            nullptr,
            [](camera_t& cam) {
                // SphericalCamera captures the full 360-degree sphere around
                // its origin - see D3's setup_camera lambda for why lookat
                // isn't used directly (degenerate cross(up,forward) if it
                // were fed straight into make_look_at) and a fixed +Z
                // forward reference is used instead, with only the origin
                // tracking cam.lookfrom (the box's center, so the panorama
                // shows all 5 walls, the ceiling light, the glass sphere,
                // and the rotated box wrapped around the viewer).
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(),       cam.lookfrom.z(),
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z() + 1.0,
                    0, 1, 0     // up
                );
                cam.alt_spherical_cam = std::make_shared<SphericalCamera<double>>(
                    cam.image_width, cam.image_height,
                    SphericalCamera<double>::EquiRectangular,
                    ctw
                );
            }
        },
        {
            "D8", 68, SceneNames::RealisticCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5-D7), rendered through a multi-element lens for realistic bokeh (pbrt-v4 RealisticCamera)",
            "Medium", 200, false, true,
            { 40, 278, 278, -420,  278, 278, 278,  0, 0, 0 },
            build_cornell_box,
            build_cornell_box_lights,
            nullptr,
            nullptr,
            [](camera_t& cam) {
                // Same simplified 9-element dgauss lens as D4 (see that
                // scene's setup_camera lambda for the full derivation of its
                // vignetting-safe 3.0/2.13mm film size). D4's lens/film pair
                // was tuned for a scene a few world-units from the camera;
                // this box is ~550 units across, so at D4's own aperture
                // (8mm = 0.008 world units) the defocus cone would be far
                // too narrow to see any blur at a comparable framing
                // distance. Scaling the aperture up to 350mm (0.35 world
                // units) restores a defocus cone of roughly the same
                // angular size as D4's - this is a display convenience for
                // an already-non-physical "simplified" lens, not meant to
                // model a real 350mm-aperture lens.
                std::vector<double> lens = {
                     35.98738,  1.21638, 1.54,  23.716,
                     11.69718,  9.9957,  1.0,   17.996,
                     13.08714, 15.9948,  1.77,  12.364,
                    -22.63294,  2.7757,  1.617, 9.812,
                      0.0,      2.75,    0.0,   7.4,     // aperture stop
                     36.3581,   8.9722,  1.617, 12.7,
                    -17.8595,   1.2,     1.0,   12.7,
                    100.0,      2.9804,  1.567, 14.478,
                    -24.5656,   0.0,     1.0,   15.0
                };
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    0, 1, 0     // up
                );
                cam.alt_realistic_cam = std::make_shared<RealisticCamera<double>>(
                    ctw,
                    3.0,    // film half-width mm
                    2.0,    // film half-height mm - same vignetting-safe size as D4
                    420.0,  // focus distance world-units, ~= lookfrom-to-lookat distance
                    350.0,  // aperture diameter mm (see comment above for why this is
                            // scaled way up from D4's 8mm)
                    lens,
                    512     // pupil samples
                );
            }
        },
        {
            "F2", 37, SceneNames::TriangleMesh, SceneCategories::Geometry,
            "Procedurally-generated icosahedron showcasing real triangle-mesh geometry (watertight Moller-Trumbore intersection)",
            "Fast", 100, false, true,
            { 35, 0, 4, 8,  0, 2.5, 0,  0.05, 0.05, 0.08 },
            build_triangle_mesh_scene,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        build_instanced_spheres_descriptor(),
        {
            "F4", 72, SceneNames::CurveFibers, SceneCategories::Geometry,
            "A windswept tuft of real Bezier curve strands (CurveShape, tapered Cylinder cross-section) - genuine ray-curve intersection on CPU, not the sphere+HairBxDF trick scene B11 uses. GPU renders the same 70 strands tessellated into tapered tubes of bilinear patches (matches pbrt-v4's own GPU curve strategy) rather than an exact curve intersection, so the tube surface reads slightly faceted up close.",
            "Fast", 150, false, true,
            { 38, 0, 2.0, 6.5,  0, 0.7, 0,  0.04, 0.045, 0.06 },
            build_curve_fibers_scene,
            []() {
                hittable_list l;
                l.add(std::make_shared<quad>(point3(-2.5,4.0,-2.5), vec3(5,0,0), vec3(0,0,5),
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G1", 38, SceneNames::StanfordBunny, SceneCategories::Models,
            "Classic Stanford bunny scan (69,451 triangles) in polished bronze, loaded from an external .obj file (requires models/stanford-bunny.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_bunny,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G2", 39, SceneNames::StanfordArmadillo, SceneCategories::Models,
            "Stanford armadillo scan (99,976 triangles) in gunmetal, loaded from an external .obj file (requires models/armadillo.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_armadillo,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G3", 40, SceneNames::StanfordHappyBuddha, SceneCategories::Models,
            "Stanford happy buddha scan (98,601 triangles) in polished gold, loaded from an external .obj file (requires models/happy-buddha.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_happy_buddha,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G4", 41, SceneNames::StanfordLucy, SceneCategories::Models,
            "Stanford Lucy angel figure (99,970 triangles) in bright silver, loaded from an external .obj file (requires models/lucy.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_lucy,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G5", 42, SceneNames::StanfordDragon, SceneCategories::Models,
            "Stanford XYZRGB Dragon (249,882 triangles) in bright silver, loaded from an external .obj file (requires models/xyzrgb_dragon.obj). Camera pulled back/up further than the other mesh scenes' default (0,3,7): the dragon's lunging pose is much wider than tall (~5.4 units wide vs ~3 tall after normalization, similar to scene 43's teapot), and the default statue framing cropped the head and tail.",
            "Very Slow", 150, true, true,
            { 35, 0, 4, 12,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_dragon,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G6", 43, SceneNames::UtahTeapot, SceneCategories::Models,
            "The classic Utah Teapot (6,320 triangles) in bright silver, loaded from an external .obj file (requires models/teapot.obj)",
            "Medium", 150, true, true,
            // Camera pulled back further than the other mesh scenes (0,3,7)
            // because the teapot's spout+handle make it much wider than it
            // is tall (~9 units wide vs ~3 tall after normalization) -
            // the statue framing crops the spout/handle at this aspect.
            { 35, 0, 6, 20,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_utah_teapot,
            []() {
                // Matches build_utah_teapot()'s light sphere, raised to
                // y=20 - see that function's comment for why (this scene's
                // raised/pulled-back camera brought the standard y=8 light
                // into frame as a blown-out disc).
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,20,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G7", 44, SceneNames::SpotCow, SceneCategories::Models,
            "Keenan Crane's Spot the Cow (5,856 triangles) in bright silver, loaded from an external .obj file (requires models/spot.obj)",
            "Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_spot_cow,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G8", 45, SceneNames::Suzanne, SceneCategories::Models,
            "Blender's Suzanne monkey-head mascot (968 triangles after fan-triangulating its mostly-quad faces) in bright silver, loaded from an external .obj file (requires models/suzanne.obj). Unlike every other mesh scene, Suzanne is a disembodied head with no neck/shoulders/pedestal, so grounding its chin at y=0 (the shared statue convention) puts its face well above the generic eye-level camera - the camera below is raised and pulled in closer to look at roughly the model's own eye height instead.",
            "Fast", 150, true, true,
            { 35, 0, 2.1, 6.5,  0, 1.9, 0,  0.05, 0.05, 0.08 },
            build_suzanne,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G9", 46, SceneNames::NefertitiBust, SceneCategories::Models,
            "Scanned bust of Nefertiti (99,938 triangles) in bright silver, loaded from an external .obj file (requires models/nefertiti.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_nefertiti,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G10", 47, SceneNames::Horse, SceneCategories::Models,
            "Classic geometry-processing test horse head/neck bust (96,966 triangles) in bright silver, loaded from an external .obj file (requires models/horse.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_horse,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G11", 48, SceneNames::Cheburashka, SceneCategories::Models,
            "Beloved cartoon-character bust from Keenan Crane's geometry-processing course (13,334 triangles) in bright silver, loaded from an external .obj file (requires models/cheburashka.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_cheburashka,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G12", 49, SceneNames::TrophyRoom, SceneCategories::Models,
            "Four already-loaded meshes (bunny, teapot, Suzanne, Spot the Cow) lined up in bronze/chrome/gold/gunmetal, the first scene to combine multiple external .obj meshes in one composition (requires models/stanford-bunny.obj, teapot.obj, suzanne.obj, spot.obj)",
            "Very Slow", 200, true, true,
            { 34, 0, 2.3, 14,  0, 0.9, 0,  0.05, 0.05, 0.08 },
            build_trophy_room,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G13", 50, SceneNames::GlassDragon, SceneCategories::Models,
            "Stanford XYZRGB Dragon (249,882 triangles) in clear glass (dielectric, IOR 1.5), loaded from an external .obj file (requires models/xyzrgb_dragon.obj). The dragon's own surface renders persistently noisy at any sample count under EITHER the regular path tracer OR --sppm -- refraction through this deeply concave mesh is a hard case for any unidirectional camera-side estimator (SPPM's photon-density gather only ever helps non-delta/diffuse surfaces, and the dragon is 100% delta-BSDF glass), not a bug. --sppm's real benefit here is a genuine floor caustic from the dragon (CPU only -- GPU SPPM currently supports scene 11 only) that the regular path tracer's NEE can't resolve; a fully clean render of the glass surface itself would need bidirectional path tracing or MLT.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_glass_dragon,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G14", 51, SceneNames::Beast, SceneCategories::Models,
            "Fantasy creature bust (common-3d-test-models) in bronze, loaded from an external .obj file (requires models/beast.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_beast,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G15", 52, SceneNames::VWBeetle, SceneCategories::Models,
            "Classic CAD-style Volkswagen Beetle in bright chrome, loaded from an external .obj file (requires models/beetle.obj). Elongated along Z after normalization, so the camera is pulled back further than the other mesh scenes, same reasoning as scene 43's Utah Teapot.",
            "Medium", 150, true, true,
            { 35, 0, 3, 16,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_beetle,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G17", 54, SceneNames::Bimba, SceneCategories::Models,
            "Smooth abstract bust/statue (AIM@SHAPE repository test model) in gold, loaded from an external .obj file (requires models/bimba.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_bimba,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G18", 55, SceneNames::Cow, SceneCategories::Models,
            "Classic Viewpoint/Alias Cow test model (distinct from scene 44's Spot the Cow) in brass, loaded from an external .obj file (requires models/cow.obj)",
            "Medium", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_cow,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G19", 56, SceneNames::Fandisk, SceneCategories::Models,
            "Classic CAD mechanical-engineering test model with sharp creases, in gunmetal, loaded from an external .obj file (requires models/fandisk.obj). Camera moved to a three-quarter elevated angle rather than the usual eye-level statue framing - this mesh's proportions are shallow along the default view axis, and a face-on shot showed only a smooth, featureless wedge with none of the sharp creases the model is known for.",
            "Medium", 150, true, true,
            { 35, 4, 9, 4,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_fandisk,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G20", 57, SceneNames::Homer, SceneCategories::Models,
            "Homer Simpson bust in gold, loaded from an external .obj file (requires models/homer.obj)",
            "Medium", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_homer,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G21", 58, SceneNames::Igea, SceneCategories::Models,
            "Classical Italian bust (Igea, Roman goddess of health) in bright silver, loaded from an external .obj file (requires models/igea.obj). An earlier camera here (raised and looking steeply down) was meant to compensate for this scan's upward-tilted face, but actually framed the shiny crown of the skull instead of the face - lowered/pulled back closer to the other mesh scenes' eye-level convention, which shows the face (eyes, nose, tilted-up chin) correctly.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 5,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_igea,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G22", 59, SceneNames::MaxPlanck, SceneCategories::Models,
            "Scanned bust of physicist Max Planck in aged bronze, loaded from an external .obj file (requires models/max-planck.obj). This scan's face points toward -Z, so the camera sits on that side (unlike the other mesh scenes' +Z default) to actually see the face instead of the back of the head.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, -7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_max_planck,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G23", 60, SceneNames::Ogre, SceneCategories::Models,
            "Fantasy ogre head in dark olive metal, loaded from an external .obj file (requires models/ogre.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_ogre,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G24", 61, SceneNames::RockerArm, SceneCategories::Models,
            "Mechanical engine-part test model in gunmetal, loaded from an external .obj file (requires models/rocker-arm.obj). Elongated along Z after normalization like the Beetle scene (G15), but much smaller overall and taller than that comparison suggested - the camera is pulled back/up further than originally set, which cropped the two boss/lobe cylinders at the top of the part. Now visible, those bosses' flat tops catch a strong mirror-like specular highlight from the overhead light - a legitimate result of a flat, low-roughness surface facing a point-ish light, confirmed by testing (repositioning/brightening the light didn't change it), not a bug.",
            "Slow", 150, true, true,
            { 35, 0, 4, 12,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_rocker_arm,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "H1", 62, SceneNames::CrytekSponza, SceneCategories::LargeScene,
            "Crytek Sponza (262K triangles) - the classic architectural global-illumination benchmark scene, with real per-face .mtl materials and image textures (curtains, columns, floor) loaded from models/sponza_textures/, lit by an open sky, loaded from an external .obj file (requires models/sponza.obj). First 'whole environment' mesh scene here rather than a single statue -- see build_sponza()'s own comment for the full design rationale.",
            "Very Slow", 150, true, true,
            { 70, -800, 300, 0,  800, 300, 0,  0, 0, 0 },
            build_sponza,
            build_sponza_lights,
            build_sponza_sky,
            nullptr
        },
        {
            "H2", 63, SceneNames::AmazonBistro, SceneCategories::LargeScene,
            "Amazon Lumberyard Bistro, Exterior (2.84M triangles) - a full outdoor street block (multiple buildings + plaza), with real per-face .mtl materials and image textures (windows, doors, foliage) loaded from models/bistro_textures/, lit by an open sky, loaded from an external .obj file (requires models/bistro_exterior.obj). Second 'whole environment' mesh scene, same design rationale as scene 62 (Crytek Sponza) -- see build_bistro_exterior()'s own comment. Camera nudged 300 units in Z from the original verified-clear-sightline position: a decorative streetlamp post sat directly in the foreground as a fully-black silhouette blocking most of the frame; the shift turns it into a pleasant framing element instead (visible tree/building behind it) rather than eliminating it.",
            "Very Slow", 150, true, true,
            { 60, 1500, 700, 1700,  4000, 700, 2000,  0, 0, 0 },
            build_bistro_exterior,
            build_bistro_exterior_lights,
            build_bistro_exterior_sky,
            nullptr
        },
        {
            "H3", 64, SceneNames::Rungholt, SceneCategories::LargeScene,
            "Rungholt (6.7M triangles) - a giant blocky Minecraft-style town, with real per-face .mtl material colors (no image textures for this one, unlike scenes 62/63's Sponza/Bistro), loaded from an external .obj file (requires models/rungholt.obj). Third 'whole environment' mesh scene, same design rationale as scenes 62-63 -- see build_rungholt()'s own comment (including a real OBJ-loader bug this mesh exposed and fixed: negative/relative face indices).",
            "Very Slow", 150, true, true,
            { 45, 400, 300, 400,  0, 40, 0,  0, 0, 0 },
            build_rungholt,
            build_rungholt_lights,
            build_rungholt_sky,
            nullptr
        },
        {
            "H4", 73, SceneNames::FireplaceRoom, SceneCategories::LargeScene,
            "Fireplace Room - a small, human-scale furnished living room (fireplace, wood floor, framed pictures, a potted plant), with real per-face .mtl materials and image textures loaded from models/fireplace_room_textures/, lit by an open sky through its windows, loaded from an external .obj file (requires models/fireplace_room.obj). Fourth 'whole environment' mesh scene, same design rationale as scenes 62-64 -- see build_fireplace_room()'s own comment. A furnished interior rather than a building/street/town-scale environment.",
            "Slow", 150, true, true,
            { 55, -2.0, 1.6, -1.5,  0, 1.3, 0,  0, 0, 0 },
            build_fireplace_room,
            build_fireplace_room_lights,
            build_fireplace_room_sky,
            nullptr
        },
        {
            "H5", 74, SceneNames::SanMiguel, SceneCategories::LargeScene,
            "San Miguel (9.9M triangles) - a dense Mexican hacienda courtyard/villa, the classic 'hero' benchmark scene with real per-face .mtl materials and image textures (tile, wood, fabric, foliage) loaded from models/san_miguel_textures/, lit by an open sky, loaded from an external .obj file (requires models/san_miguel.obj). Fifth 'whole environment' mesh scene, same design rationale as scenes 62-64/73 -- see build_san_miguel()'s own comment.",
            "Very Slow", 150, true, true,
            { 45, 10, 3, 5,  0, 3, 0,  0, 0, 0 },
            build_san_miguel,
            build_san_miguel_lights,
            build_san_miguel_sky,
            nullptr
        },
        {
            "H6", 75, SceneNames::SibenikCathedral, SceneCategories::LargeScene,
            "Sibenik Cathedral - a Gothic cathedral interior (vaulted nave, stone columns, a rose window, colored stained glass), with real per-face .mtl materials, image textures, and real bump maps loaded from models/sibenik_cathedral_textures/, lit through its open doorway/arches, loaded from an external .obj file (requires models/sibenik_cathedral.obj). Sixth 'whole environment' mesh scene, same design rationale as scenes 62-64/73/74 -- see build_sibenik_cathedral()'s own comment.",
            "Very Slow", 400, true, true,
            { 60, -15, 1.7, 0,  15, 5, 0,  0, 0, 0 },
            build_sibenik_cathedral,
            build_sibenik_cathedral_lights,
            build_sibenik_cathedral_sky,
            nullptr
        },
        {
            "H7", 76, SceneNames::BreakfastRoom, SceneCategories::LargeScene,
            "Breakfast Room - a cozy furnished dining interior with glassware, table settings, and marble/tile textures, with real per-face .mtl materials and image textures loaded from models/breakfast_room_textures/, lit by an open sky through its windows, loaded from an external .obj file (requires models/breakfast_room.obj). Seventh 'whole environment' mesh scene, same design rationale as scenes 62-64/73/74/75 -- see build_breakfast_room()'s own comment.",
            "Very Slow", 300, true, true,
            { 70, -3.0, 1.5, 3.0,  2.5, 1.3, 0,  0, 0, 0 },
            build_breakfast_room,
            build_breakfast_room_lights,
            build_breakfast_room_sky,
            nullptr
        },
        {
            "H8", 77, SceneNames::SalleDeBain, SceneCategories::LargeScene,
            "Salle de Bain - a tiled bathroom interior with a mirror, tub, and a real ceiling light fixture (genuine Ke emission -- exercises the NEE-light path a second time, after Fireplace Room), with real per-face .mtl materials and image textures loaded from models/salle_de_bain_textures/, loaded from an external .obj file (requires models/salle_de_bain.obj). Eighth 'whole environment' mesh scene, same design rationale as scenes 62-64/73-75 -- see build_salle_de_bain()'s own comment.",
            "Slow", 150, true, true,
            { 50, 10, 15, -5,  -10, 12, 5,  0, 0, 0 },
            build_salle_de_bain,
            build_salle_de_bain_lights,
            build_salle_de_bain_sky,
            nullptr
        },
        {
            "H9", 78, SceneNames::Gallery, SceneCategories::LargeScene,
            "Gallery - the Hallwyl Museum picture gallery in Stockholm, an ornate room of framed paintings, chandeliers, and a parquet floor, with a real per-face .mtl material and an image texture loaded from models/gallery_textures/, lit by an open sky, loaded from an external .obj file (requires models/gallery.obj). Ninth 'whole environment' mesh scene, same design rationale as scenes 62-64/73-76 -- see build_gallery()'s own comment.",
            "Very Slow", 300, true, true,
            { 55, 0, 2.2, -5,  0, 2.2, 0,  0, 0, 0 },
            build_gallery,
            build_gallery_lights,
            build_gallery_sky,
            nullptr
        },
        {
            "H10", 79, SceneNames::LostEmpire, SceneCategories::LargeScene,
            "Lost Empire - a large half-buried ancient city exported from a Minecraft world, with temple platforms, staircases, and a lava chamber, with real per-face .mtl materials and an image texture loaded from models/lost_empire_textures/, lit by an open sky, loaded from an external .obj file (requires models/lost_empire.obj). Tenth 'whole environment' mesh scene, and the first at a scale (165 units deep) that suits a long video flythrough -- see build_lost_empire()'s own comment.",
            "Slow", 150, true, true,
            { 55, 0, 60, 100,  0, 10, 0,  0, 0, 0 },
            build_lost_empire,
            build_lost_empire_lights,
            build_lost_empire_sky,
            nullptr
        },
        {
            "H11", 80, SceneNames::VokseliaSpawn, SceneCategories::LargeScene,
            "Vokselia Spawn - a small floating voxel island, exported from the same Minecraft world as Lost Empire from its spawn point, with a real per-face .mtl material and an image texture loaded from models/vokselia_spawn_textures/, lit by an open sky, loaded from an external .obj file (requires models/vokselia_spawn.obj). Eleventh 'whole environment' mesh scene -- see build_vokselia_spawn()'s own comment.",
            "Medium", 100, true, true,
            { 40, 4.5, 0.9, 4.5,  0, 0.25, 0,  0, 0, 0 },
            build_vokselia_spawn,
            build_vokselia_spawn_lights,
            build_vokselia_spawn_sky,
            nullptr
        },
        {
            "H12", 81, SceneNames::PowerPlant, SceneCategories::LargeScene,
            "Power Plant - a complete model of an actual coal-fired power plant (12.76M triangles, 5.98M vertices), the largest scene in this collection by triangle count, with flat per-face .mtl colors (no image textures), lit by an open sky, loaded from an external .obj file (requires models/powerplant.obj). Twelfth 'whole environment' mesh scene, and the first needing a real coordinate rescale rather than raw OBJ units -- see build_power_plant()'s own comment.",
            "Slow", 150, true, true,
            { 40, 130, 85, 130,  -55, 40, -35,  0, 0, 0 },
            build_power_plant,
            build_power_plant_lights,
            build_power_plant_sky,
            nullptr
        },

        // ---------------------------------------------------------------
        // Curated pbrt_scenes/*.pbrt example scenes, under their real topic
        // tab instead of only the generic "Custom Scenes" bucket every
        // loaded .pbrt file auto-discovers into (see
        // pbrt_scene_registry::build_curated_pbrt_scene_descriptor()'s own
        // comment). Legacy ids 100+ - past every real case in
        // gpu/optix/scene_builder.cpp's switch, same reasoning as
        // append()'s own dynamically-assigned ids below.
        // ---------------------------------------------------------------

        // -- Materials --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B15", 100, SceneNames::MixMaterialPbrtExample, SceneCategories::Materials,
            "Real per-shading-point stochastic pbrt \"mix\" material resolution on all three backends -- a fine-grained speckle of matte red diffuse and a conductor's real specular highlights, not one flat averaged color.",
            "Fast", "mix-material.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B16", 101, SceneNames::LayeredMaterialsPbrtExample, SceneCategories::Materials,
            "Four pbrt material kinds no other bundled example scene touches: thindielectric, coatedconductor, diffusetransmission, and subsurface via a named measured-scattering preset (\"Marble\", no external file needed).",
            "Fast", "layered-materials.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B17", 102, SceneNames::CoatedDiffuseTexturePbrtExample, SceneCategories::Materials,
            "Real texture-bound reflectance for pbrt's CoatedDiffuse material, which previously silently dropped to a flat color on both backends.",
            "Fast", "coateddiffuse-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B18", 103, SceneNames::ConductorRgbEtaKPbrtExample, SceneCategories::Materials,
            "Explicit RGB eta/k for pbrt's conductor material, plus named-metal-spectrum resolution for coatedconductor -- real complex-IOR GGX highlights instead of the flat fuzz-mirror/reflectance-only fallback.",
            "Fast", "conductor-rgb-eta-k.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B19", 104, SceneNames::DiffuseTransmissionTexturePbrtExample, SceneCategories::Materials,
            "Texture-bound reflectance/transmittance for pbrt's DiffuseTransmission material, threaded through both backends.",
            "Fast", "diffusetransmission-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B20", 105, SceneNames::HairMaterialPbrtExample, SceneCategories::Materials,
            "pbrt's Material \"hair\" (Marschner/Chiang fiber scattering) applied to ordinary spheres, matching this project's own native Hair Fibers demo for a fair comparison.",
            "Fast", "hair-material.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B21", 106, SceneNames::NestedCheckerTexturePbrtExample, SceneCategories::Materials,
            "One level of nested imagemap texture reference inside a pbrt checkerboard/mix texture -- tex1/tex2 bound to a real image instead of only a flat literal color.",
            "Fast", "nested-checker-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B22", 107, SceneNames::NamedMaterialAndTexturePbrtExample, SceneCategories::Materials,
            "pbrt's NamedMaterial referenced directly for a shape (not just as a \"mix\" sub-material), plus a texture-bound material parameter and AreaLightSource's twosided flag.",
            "Fast", "named-material-and-texture.pbrt"),

        // -- Lights --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C8", 108, SceneNames::PunctualLightsPbrtExample, SceneCategories::Lights,
            "All five of pbrt-v4's punctual (delta-distribution) light kinds in one scene: point, spot, distant, goniometric, and projection.",
            "Fast", "punctual-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C9", 109, SceneNames::GoniometricProjectionPbrtExample, SceneCategories::Lights,
            "Real image decoding for pbrt's goniometric and projection lights, which previously silently ignored their own filename and fell back to a uniform beam.",
            "Fast", "goniometric-projection.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C10", 110, SceneNames::BlackbodyLightPbrtExample, SceneCategories::Lights,
            "pbrt's \"blackbody L\" colour-temperature area lights -- two identical panels at 2500K and 9000K, so a regression back to flat-white emission would be immediately visible.",
            "Fast", "blackbody-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C11", 111, SceneNames::TexturedTwoSidedLightsPbrtExample, SceneCategories::Lights,
            "A real filename-textured, two-sided AreaLightSource on a non-triangle (sphere/quad) shape, on both backends.",
            "Fast", "textured-twosided-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C12", 112, SceneNames::InfiniteLightPbrtExample, SceneCategories::Lights,
            "pbrt's \"infinite\" constant-colour sky light in open geometry, actually lighting the scene from every direction rather than being blocked by a room.",
            "Fast", "infinite-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C13", 113, SceneNames::DiskCylinderLightPbrtExample, SceneCategories::Lights,
            "Disk and cylinder shapes as real NEE-samplable area lights on both GPU backends, converging as cleanly as CPU's solid-angle sampling instead of noisier hit-only emission.",
            "Fast", "disk-cylinder-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C14", 114, SceneNames::TwoSphereLightsPbrtExample, SceneCategories::Lights,
            "Two sphere area lights in one scene, pinning a GPU light-type-table width bug where every light after the first silently misread its own type.",
            "Fast", "two-sphere-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C15", 115, SceneNames::TriangleFanLightPbrtExample, SceneCategories::Lights,
            "An area light that is NOT a parallelogram -- an irregular 5-triangle fan the quad-merge pass can't rejoin, exercising the GPU's per-triangle light sampling.",
            "Fast", "triangle-fan-light.pbrt"),

        // -- Cameras --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D9", 116, SceneNames::DepthOfFieldPbrtExample, SceneCategories::Cameras,
            "A perspective camera's thin-lens depth-of-field (lensradius/focaldistance) loaded from a pbrt file, on both backends.",
            "Fast", "depth-of-field.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D10", 117, SceneNames::OrthographicCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's orthographic (parallel-projection) camera loaded from a file -- two same-size spheres at different depths read as equal size, not perspective-foreshortened.",
            "Fast", "orthographic-camera.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D11", 118, SceneNames::SphericalCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's spherical (equal-area) camera loaded from a file, positioned inside an enclosed room so it actually captures every direction at once.",
            "Fast", "spherical-camera.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D12", 119, SceneNames::RealisticCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's realistic multi-element lens camera loaded from a file, including the lensfile-loading path a compiled-in scene never exercised.",
            "Fast", "realistic-camera.pbrt"),

        // -- Volumes --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E5", 120, SceneNames::CloudMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"cloud\" (Perlin-noise heterogeneous scattering) loaded from a file, on both backends.",
            "Fast", "cloud-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E6", 121, SceneNames::CylinderMediumPbrtExample, SceneCategories::Volumes,
            "A homogeneous fog medium on pbrt's Shape \"cylinder\", now real on both GPU backends instead of silently rendering as ordinary empty geometry.",
            "Fast", "cylinder-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E7", 122, SceneNames::RgbGridMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"rgbgrid\" (an RGB voxel grid) rendering as a soft coloured nebula, on both backends.",
            "Fast", "rgbgrid-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E8", 123, SceneNames::UniformGridMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"uniformgrid\" (a single-channel density voxel grid) rendering as a soft glowing blob, on both backends.",
            "Fast", "uniformgrid-medium.pbrt"),

        // -- Geometry --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F5", 124, SceneNames::PlymeshUvPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"plymesh\" real per-vertex UV data, threaded through both backends -- previously GPU-recursive rendered this exact scene solid black.",
            "Fast", "plymesh-uv.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F6", 125, SceneNames::PlymeshGeometryPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"plymesh\" loading a real external .ply file, including fan-triangulation of a non-triangular base face.",
            "Fast", "plymesh-geometry.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F7", 126, SceneNames::CurveTuftPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"curve\" (real cubic-Bezier fiber geometry, tessellated for GPU) compared against this project's own native curve-tuft demo.",
            "Fast", "curve-tuft.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F8", 127, SceneNames::CurveHairTuftPbrtExample, SceneCategories::Geometry,
            "Real curve geometry paired with Material \"hair\" for the first time -- the exact combination that motivated HairBxDF's own fiber-tangent fix.",
            "Fast", "curve-hair-tuft.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F9", 128, SceneNames::TrianglemeshUvPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"trianglemesh\" \"point2 uv\" parameter threaded through both backends -- previously GPU-recursive rendered this exact scene solid black.",
            "Fast", "trianglemesh-uv.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F10", 129, SceneNames::PixelFilterBoxPbrtExample, SceneCategories::Geometry,
            "pbrt's PixelFilter directive end-to-end on both backends -- a box filter's harder, more aliased silhouette edges compared to the default Gaussian.",
            "Fast", "pixel-filter-box.pbrt"),

        // -- Models --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "G25", 130, SceneNames::KillerooSimplePbrtExample, SceneCategories::Models,
            "The classic pbrt-v4 \"killeroo\" statue example scene, loaded end-to-end from its own .pbrt file rather than a compiled-in scene.",
            "Medium", "killeroo-simple.pbrt"),
    };
    return registry;
}

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
    // all pbrt scenes share category CustomScenes ("I"), so this is simply
    // "the Nth pbrt scene loaded this run", with no static CustomScenes
    // entries in the builtin registry to continue from today.
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
