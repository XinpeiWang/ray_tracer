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
#include <deque>
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
    // The OLD flat 0..64 scene_id, kept only so gpu/optix/scene_builder.cpp's
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
};

// Dummy sphere light used by scenes that have no explicit light geometry
static inline hittable_list sky_dummy_lights() {
    hittable_list l;
    auto empty_mat = std::shared_ptr<material>();
    l.add(std::make_shared<sphere>(point3(0, 1000, 0), 500, empty_mat));
    return l;
}

static inline hittable_list no_lights() { return hittable_list{}; }

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
            build_cornell_box_lights
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
            build_cornell_box_lights
        },
        {
            "E2", 31, SceneNames::CloudMedium, SceneCategories::Volumes,
            "Open scene with a procedural Perlin-noise cloud volume (pbrt-v4 CloudMedium)",
            "Slow", 300, false, true,
            { 20, 0, 5, 20,  0, 2, 0,  0.5, 0.7, 1.0 },
            build_cloud_medium_scene,
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
            "Stanford XYZRGB Dragon (249,882 triangles) in bright silver, loaded from an external .obj file (requires models/xyzrgb_dragon.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
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
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
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
            "Blender's Suzanne monkey-head mascot (968 triangles after fan-triangulating its mostly-quad faces) in bright silver, loaded from an external .obj file (requires models/suzanne.obj)",
            "Fast", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
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
            "G16", 53, SceneNames::VWBeetleAlt, SceneCategories::Models,
            "Alternate resolution/topology of scene 52's Volkswagen Beetle mesh, in gunmetal, loaded from an external .obj file (requires models/beetle-alt.obj). Same elongated-Z proportions and pulled-back camera as scene 52.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 16,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_beetle_alt,
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
            "Classic CAD mechanical-engineering test model with sharp creases, in gunmetal, loaded from an external .obj file (requires models/fandisk.obj)",
            "Medium", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
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
            "Classical Italian bust (Igea, Roman goddess of health) in bright silver, loaded from an external .obj file (requires models/igea.obj). This particular scan's face is tilted upward rather than forward -- camera positioned higher and closer, looking down at the face, rather than the other mesh scenes' eye-level framing.",
            "Very Slow", 150, true, true,
            { 35, 0, 5, 3,  0, 1.5, 0,  0.05, 0.05, 0.08 },
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
            "Mechanical engine-part test model in gunmetal, loaded from an external .obj file (requires models/rocker-arm.obj). Elongated along Z after normalization like the Beetle scenes, but much smaller overall, so the camera is only modestly pulled back (not as far as scenes 52/53's cars).",
            "Slow", 150, true, true,
            { 35, 0, 2.5, 7,  0, 1.2, 0,  0.05, 0.05, 0.08 },
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
            "Crytek Sponza (262K triangles) - the classic architectural global-illumination benchmark scene, in warm sandstone lambertian (this renderer's OBJ loader has no per-face/.mtl material support) lit by an open sky, loaded from an external .obj file (requires models/sponza.obj). First 'whole environment' mesh scene here rather than a single statue -- see build_sponza()'s own comment for the full design rationale.",
            "Very Slow", 150, true, true,
            { 70, -800, 300, 0,  800, 300, 0,  0, 0, 0 },
            build_sponza,
            no_lights,
            build_sponza_sky,
            nullptr
        },
        {
            "H2", 63, SceneNames::AmazonBistro, SceneCategories::LargeScene,
            "Amazon Lumberyard Bistro, Exterior (2.84M triangles) - a full outdoor street block (multiple buildings + plaza), in warm plaster/terracotta lambertian (this renderer's OBJ loader has no per-face/.mtl material support) lit by an open sky, loaded from an external .obj file (requires models/bistro_exterior.obj). Second 'whole environment' mesh scene, same design rationale as scene 62 (Crytek Sponza) -- see build_bistro_exterior()'s own comment.",
            "Very Slow", 150, true, true,
            { 60, 1500, 700, 2000,  4000, 700, 2000,  0, 0, 0 },
            build_bistro_exterior,
            no_lights,
            build_bistro_exterior_sky,
            nullptr
        },
        {
            "H3", 64, SceneNames::Rungholt, SceneCategories::LargeScene,
            "Rungholt (6.7M triangles) - a giant blocky Minecraft-style town, in warm wood-tone lambertian, loaded from an external .obj file (requires models/rungholt.obj). Third 'whole environment' mesh scene, same design rationale as scenes 62-63 -- see build_rungholt()'s own comment (including a real OBJ-loader bug this mesh exposed and fixed: negative/relative face indices).",
            "Very Slow", 150, true, true,
            { 45, 400, 300, 400,  0, 40, 0,  0, 0, 0 },
            build_rungholt,
            no_lights,
            build_rungholt_sky,
            nullptr
        },
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

// Loading is deferred AND shared between the two callbacks below, which is
// why this is a shared_ptr the lambdas capture rather than a local.
struct Loaded {
    bool attempted = false;
    pbrt_cpu::BuildResult built;
};

std::map<std::string, std::string>& paths();   // defined below, used by append()

inline void append(std::vector<SceneDescriptor>& registry) {
    const std::vector<pbrt_discover::Discovered> found = pbrt_discover::scanDefaultPaths();

    // SceneDescriptor holds `const char*`, so the strings have to outlive the
    // registry. A deque is used rather than a vector because it never
    // reallocates its elements, so pointers taken here stay valid as more
    // scenes are appended.
    static std::deque<std::string> names;
    static std::deque<std::string> descriptions;

    // legacy_id keeps counting up from the builtins exactly as `id` used to
    // (see SceneDescriptor::legacy_id's comment) - values >= builtin count
    // always miss every `case N:` in scene_builder.cpp's switch and land on
    // `default:`, which is all that's required of it for pbrt scenes.
    // user_number is the NEW id's counter, independent and always 1-based -
    // all pbrt scenes share category UserScenes ("I"), so this is simply
    // "the Nth pbrt scene loaded this run", with no static UserScenes
    // entries in the builtin registry to continue from today.
    int legacy_id = builtin_scene_count();
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

        const auto state = std::make_shared<Loaded>();
        const std::string path = d.path;

        // Both callbacks run the load exactly once between them.
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
                    // Worth printing rather than inferring from the picture:
                    // instanced geometry that failed to be placed looks
                    // identical to geometry the scene never had.
                    std::cerr << "[pbrt] " << path << ": "
                              << state->built.triangleCount << " triangles, "
                              << state->built.sphereCount << " spheres, "
                              << state->built.instanceCount << " instance placements\n";

                }
            }
            return state->built;
        };

        SceneDescriptor s;
        // Uses SceneCategories::letter_for_category() rather than a
        // hardcoded 'I', so this can't silently drift from
        // SceneCategories::kAll's declared order (see that function's
        // comment in scene_descriptor.h).
        s.id = std::string(1, SceneCategories::letter_for_category(SceneCategories::UserScenes))
             + std::to_string(user_number++);
        s.legacy_id = legacy_id++;
        s.name = names.back().c_str();
        s.category = SceneCategories::UserScenes;
        s.description = descriptions.back().c_str();
        // Honest rather than flattering: a .pbrt file can hold anything from
        // three triangles to ten million, and nothing in the header says
        // which. "Unknown" is the truthful answer at this point.
        s.performance = "Unknown";
        s.recommended_spp = d.samplesPerPixel;
        s.requires_files = true;
        // gpu/optix/pbrt_gpu_builder.h consumes the same FlatScene this does,
        // so both backends render the same file, and they now agree on what
        // they can sample: sphere, parallelogram AND individual-triangle area
        // lights (GpuLightKind), plus instanced triangles and spheres. The GPU
        // used to be the weaker of the two - it could only sample lights that
        // were spheres or parallelograms - which is no longer true and is why
        // this no longer carries a caveat.
        s.gpu_compatible = true;
        s.camera = CameraConfig{
            d.camera.vfov,
            d.camera.lookfrom[0], d.camera.lookfrom[1], d.camera.lookfrom[2],
            d.camera.lookat[0],   d.camera.lookat[1],   d.camera.lookat[2],
            0.0, 0.0, 0.0,                      // pbrt has no flat background
            CameraMode::Fixed,
            d.camera.aperture,
            // NOT d.camera.focusDistance - see focusDistanceFor()'s comment.
            // Passing pbrt's raw default here made near geometry vanish.
            pbrt_flatten::focusDistanceFor(d.camera)
        };

        // A failed load yields an empty world rather than a crash - the error
        // above already said why, and an empty render is a legible symptom.
        s.build_world  = [ensure]() {
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
        s.build_punct = nullptr;
        // CameraConfig cannot express an up vector, but pbrt's LookAt can, and
        // a scene shot in Z-up renders sideways without this. setup_camera
        // runs after every config field is applied, so it is the right place.
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
        s.setup_camera = [ux, uy, uz, pcam, path](camera_t& cam) {
            cam.vup = vec3(ux, uy, uz);
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

        paths()[s.id] = path;
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
