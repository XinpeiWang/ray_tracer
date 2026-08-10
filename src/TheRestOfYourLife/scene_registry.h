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
#include <functional>
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
    int         id;
    const char* name;
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

inline const std::vector<SceneDescriptor>& get_scene_registry() {
    static const std::vector<SceneDescriptor> registry = {
        {
            0, SceneNames::CornellBox,
            "Classic Cornell box with glass sphere and aluminum box",
            "Medium", 100, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            1, SceneNames::BouncingSpheres,
            "Random spheres with checker ground (In One Weekend final)",
            "Slow", 100, false, true,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_bouncing_spheres,
            sky_dummy_lights
        },
        {
            2, SceneNames::CheckeredSpheres,
            "Two spheres with procedural checker texture",
            "Fast", 100, false, true,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_checkered_spheres,
            sky_dummy_lights
        },
        {
            3, SceneNames::Earth,
            "Globe with earth texture mapping (requires earthmap.jpg)",
            "Fast", 100, true, true,
            { 20, 0, 0, 12,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_earth,
            sky_dummy_lights
        },
        {
            4, SceneNames::PerlinSpheres,
            "Spheres with Perlin noise marble texture",
            "Fast", 100, false, true,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_perlin_spheres,
            sky_dummy_lights
        },
        {
            5, SceneNames::ColoredQuads,
            "Five colored quad primitives",
            "Fast", 100, false, true,
            { 80, 0, 0, 9,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_quads,
            sky_dummy_lights
        },
        {
            6, SceneNames::SimpleLight,
            "Perlin spheres with emissive light sources",
            "Fast", 100, false, true,
            { 20, 26, 3, 6,  0, 2, 0,  0, 0, 0 },
            build_simple_light,
            no_lights
        },
        {
            7, SceneNames::CornellSmoke,
            "Cornell box with volumetric fog",
            "Slow", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_smoke,
            build_cornell_smoke_lights
        },
        {
            8, SceneNames::FinalScene,
            "Complex scene from The Next Week",
            "Very Slow", 500, false, true,
            { 40, 478, 278, -600,  278, 278, 0,  0, 0, 0 },
            build_final_scene,
            build_final_scene_lights
        },
        {
            9, SceneNames::RoughMetalSpheres,
            "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
            "Medium", 200, false, true,
            { 35, 0, 2.5, 10,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_rough_metal_spheres,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-6,6,-4), vec3(12,0,0), vec3(0,0,8), empty_mat));
                return l;
            }
        },
        {
            10, SceneNames::CornellRoughMetal,
            "Cornell box with rough aluminum box and rough gold sphere",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_rough_metal,
            build_cornell_box_lights
        },
        {
            11, SceneNames::CornellRoughGlass,
            "Cornell box with a GGX rough-dielectric sphere (pbrt-v4 RoughDielectricBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_rough_glass,
            build_cornell_box_lights
        },
        {
            12, SceneNames::CornellConductor,
            "Cornell box with polished gold sphere and aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_conductor,
            build_cornell_box_lights
        },
        {
            13, SceneNames::CornellCoatedDiffuse,
            "Cornell box with blue coated-diffuse sphere and red coated-diffuse box (pbrt-v4 CoatedDiffuseBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_coated_diffuse,
            build_cornell_box_lights
        },
        {
            14, SceneNames::CornellThinGlass,
            "Cornell box with a vertical thin-glass panel, analytic multi-bounce Fresnel (pbrt-v4 ThinDielectricBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_thin_glass,
            build_cornell_thin_glass_lights
        },
        {
            15, SceneNames::CornellCoatedConductor,
            "Cornell box with lacquered-gold sphere and lacquered-copper box (pbrt-v4 CoatedConductorBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_coated_conductor,
            build_cornell_box_lights
        },
        {
            16, SceneNames::CornellWaxSlab,
            "Cornell box with a wax sphere that diffusely reflects and transmits light (pbrt-v4 DiffuseTransmissionBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_wax_slab,
            build_cornell_box_lights
        },
        {
            17, SceneNames::CornellCrystal,
            "Cornell box with a crystal sphere using Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_crystal,
            build_cornell_box_lights
        },
        {
            18, SceneNames::PrincipledShowcase,
            "Row of spheres from matte plastic to metallic with clearcoat (pbrt-v4 PrincipledBxDF)",
            "Medium", 200, false, true,
            { 35, 0, 2.5, 10,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_principled_showcase,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-6,6,-4), vec3(12,0,0), vec3(0,0,8), empty_mat));
                return l;
            }
        },
        {
            19, SceneNames::HairFibers,
            "Sphere cluster with hair/fur fiber scattering (pbrt-v4 HairBxDF)",
            "Medium", 200, false, true,
            { 30, 0, 2, 8,  0, 1, 0,  0.05, 0.05, 0.07 },
            build_hair_fibers,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-6,6,-4), vec3(12,0,0), vec3(0,0,8), empty_mat));
                return l;
            }
        },
        {
            20, SceneNames::NormalMappedCornell,
            "Cornell box with procedural bump-mapped back wall and normal-mapped sphere (pbrt-v4 NormalMap/BumpMap)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_normal_mapped_cornell,
            build_cornell_box_lights
        },
        {
            21, SceneNames::SubsurfaceSlab,
            "Cornell box with translucent wax slab and jade sphere using subsurface-like scattering",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_subsurface_slab,
            build_cornell_box_lights
        },
        {
            22, SceneNames::DepthOfField,
            "Row of spheres with defocus blur showing depth-of-field from the thin-lens camera model",
            "Medium", 200, false, true,
            { 20, 0, 2, 9,  0, 1, 0,  0.70, 0.80, 1.00, CameraMode::Fixed, 10.0, 9.0 },
            build_depth_of_field,
            sky_dummy_lights
        },
        {
            23, SceneNames::BilinearPatchScene,
            "Cornell box with curved bilinear patch saddle surface (pbrt-v4 BilinearPatch shape)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_bilinear_patch_scene,
            build_cornell_box_lights
        },
        // ---- pbrt-v4 light / camera / medium showcase ----
        {
            24, SceneNames::HdriSky,
            "Open scene lit by a procedural gradient sky (pbrt-v4 ImageInfiniteLight / sky_light)",
            "Medium", 200, false, true,
            { 30, 0, 2, 10,  0, 1, 0,  0, 0, 0 },
            build_hdri_sky_world,
            no_lights,
            build_hdri_sky,
            nullptr
        },
        {
            25, SceneNames::SpotlightCornell,
            "Cornell box lit by a spotlight with smooth penumbra (pbrt-v4 SpotLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_spotlight_cornell,
            no_lights,
            nullptr,
            build_spotlight_punct
        },
        {
            26, SceneNames::DistantLightCornell,
            "Cornell box lit by a parallel sun-like distant light (pbrt-v4 DistantLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_distant_light_cornell,
            no_lights,
            nullptr,
            build_distant_light_punct
        },
        {
            27, SceneNames::PointLightCornell,
            "Cornell box lit by a single overhead point light with 1/r^2 falloff (pbrt-v4 PointLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_point_light_cornell,
            no_lights,
            nullptr,
            build_point_light_punct
        },
        {
            28, SceneNames::GoniometricLight,
            "Cornell box lit by a goniometric (IES-profile) point light (pbrt-v4 GoniometricLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_goniometric_light_scene,
            no_lights,
            nullptr,
            build_goniometric_punct
        },
        {
            29, SceneNames::ProjectionLight,
            "Cornell box with a slide-projector beam casting a checkerboard pattern (pbrt-v4 ProjectionLight)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_projection_light_scene,
            no_lights,
            nullptr,
            build_projection_punct
        },
        {
            30, SceneNames::HomogeneousMedium,
            "Cornell box filled with a homogeneous scattering fog (pbrt-v4 HomogeneousMedium / HenyeyGreenstein)",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_homogeneous_medium_scene,
            build_cornell_box_lights
        },
        {
            31, SceneNames::CloudMedium,
            "Open scene with a procedural Perlin-noise cloud volume (pbrt-v4 CloudMedium)",
            "Slow", 300, false, true,
            { 20, 0, 5, 20,  0, 2, 0,  0.5, 0.7, 1.0 },
            build_cloud_medium_scene,
            sky_dummy_lights
        },
        {
            32, SceneNames::OrthographicCamera,
            "Geometric showcase rendered with an orthographic (parallel-projection) camera (pbrt-v4 OrthographicCamera)",
            "Fast", 100, false, true,
            { 30, 0, 3, 12,  0, 1, 0,  0, 0, 0 },
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
                cam.alt_ortho_cam = std::make_shared<OrthographicCamera<double>>(
                    xmin*8, xmax*8, ymin*8, ymax*8,   // screen window scaled to scene
                    cam.image_width, cam.image_height,
                    ctw
                );
            }
        },
        {
            33, SceneNames::SphericalCamera,
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
            34, SceneNames::MeasuredBrdf,
            "Sphere cluster with measured BRDF material using tabulated RGL data (pbrt-v4 MeasuredBxDF)",
            "Medium", 200, false, true,
            { 25, 0, 3, 12,  0, 1, 0,  0, 0, 0 },
            build_measured_brdf_scene,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 1.5,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            35, SceneNames::PortalInfiniteLight,
            "Room scene with a portal window sampling the sky through a planar quad (pbrt-v4 PortalImageInfiniteLight)",
            "Slow", 300, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_portal_light_scene,
            no_lights,
            build_portal_sky
        },
        {
            36, SceneNames::RealisticCamera,
            "Spheres rendered through a thin-lens with realistic lens-element bokeh (pbrt-v4 RealisticCamera)",
            "Medium", 200, false, true,
            { 50, 0, 2, -2,  0, 1, 5,  0, 0, 0 },
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
                // registry's default (0,2,-2)/(0,1,5) directly, so this alt
                // camera actually moves for video mode instead of silently
                // staying frozen on every frame.
                Mat4<double> ctw = make_look_at<double>(
                    cam.lookfrom.x(), cam.lookfrom.y(), cam.lookfrom.z(),
                    cam.lookat.x(),   cam.lookat.y(),   cam.lookat.z(),
                    0, 1,  0    // up
                );
                cam.alt_realistic_cam = std::make_shared<RealisticCamera<double>>(
                    ctw,
                    18.0,   // film half-width mm (matches a real 35mm frame's
                    12.0,   // film half-height mm  half-extents, realistic_camera_tests.cpp's
                            // FILM_HX/FILM_HY convention)
                    7.0,    // focus distance meters
                    8.0,    // aperture diameter mm
                    lens,
                    512     // pupil samples
                );
            }
        },
        {
            37, SceneNames::TriangleMesh,
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
            38, SceneNames::StanfordBunny,
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
            39, SceneNames::StanfordArmadillo,
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
            40, SceneNames::StanfordHappyBuddha,
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
            41, SceneNames::StanfordLucy,
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
    };
    return registry;
}

// Lookup by id -- returns nullptr if not found
inline const SceneDescriptor* find_scene(int id) {
    for (const auto& s : get_scene_registry())
        if (s.id == id) return &s;
    return nullptr;
}

inline int scene_count() {
    return static_cast<int>(get_scene_registry().size());
}
