#pragma once
// scene_registry.h -- CPU scene registry (pbrt-v4 SceneEntity pattern)
//
// Scene names are defined as constants in src/shared/scene_descriptor.h (SceneNames::).
// Always use SceneNames:: constants here -- never raw string literals -- so both
// tables can never drift out of sync.
//
// To add a new scene:
//   1. Add a SceneNames:: constant in scene_descriptor.h
//   2. Add a row in scene_descriptor.h kScenes[]
//   3. Add a builder function in scenes.h
//   4. Add one SceneDescriptor entry in get_scene_registry() below using SceneNames::
//   Done -- cpu_interface and GUI pick it up automatically.

#include "../shared/scene_descriptor.h"
#include "scenes.h"
#include "cornell_box_scene.h"
#include <functional>
#include <vector>
#include <string>

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

struct SceneDescriptor {
    int         id;
    const char* name;
    const char* description;
    const char* performance;   // "Fast" | "Medium" | "Slow" | "Very Slow"
    int         recommended_spp;
    bool        requires_files;
    bool        gpu_compatible;
    CameraConfig camera;
    std::function<hittable_list()>              build_world;
    std::function<hittable_list()>              build_lights;  // may return empty list
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
            "Slow", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_bouncing_spheres,
            sky_dummy_lights
        },
        {
            2, SceneNames::CheckeredSpheres,
            "Two spheres with procedural checker texture",
            "Fast", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_checkered_spheres,
            sky_dummy_lights
        },
        {
            3, SceneNames::Earth,
            "Globe with earth texture mapping (requires earthmap.jpg)",
            "Fast", 100, true, false,
            { 20, 0, 0, 12,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_earth,
            sky_dummy_lights
        },
        {
            4, SceneNames::PerlinSpheres,
            "Spheres with Perlin noise marble texture",
            "Fast", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_perlin_spheres,
            sky_dummy_lights
        },
        {
            5, SceneNames::ColoredQuads,
            "Five colored quad primitives",
            "Fast", 100, false, false,
            { 80, 0, 0, 9,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_quads,
            sky_dummy_lights
        },
        {
            6, SceneNames::SimpleLight,
            "Perlin spheres with emissive light sources",
            "Fast", 100, false, false,
            { 20, 26, 3, 6,  0, 2, 0,  0, 0, 0 },
            build_simple_light,
            no_lights
        },
        {
            7, SceneNames::CornellSmoke,
            "Cornell box with volumetric fog",
            "Slow", 200, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_smoke,
            build_cornell_box_lights
        },
        {
            8, SceneNames::FinalScene,
            "Complex scene from The Next Week",
            "Very Slow", 500, false, false,
            { 40, 478, 278, -600,  278, 278, 0,  0, 0, 0 },
            build_final_scene,
            build_final_scene_lights
        },
        {
            9, SceneNames::RoughMetalSpheres,
            "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
            "Medium", 200, false, false,
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
            build_cornell_box_lights
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
            "Medium", 200, false, false,
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
            "Medium", 200, false, false,
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
            "Medium", 200, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_normal_mapped_cornell,
            build_cornell_box_lights
        },
        {
            21, SceneNames::SubsurfaceSlab,
            "Cornell box with translucent wax slab and jade sphere using subsurface-like scattering",
            "Slow", 300, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_subsurface_slab,
            build_cornell_box_lights
        },
        {
            22, SceneNames::DepthOfField,
            "Row of spheres with defocus blur showing depth-of-field from the thin-lens camera model",
            "Medium", 200, false, false,
            { 20, 0, 2, 9,  0, 1, 0,  0.70, 0.80, 1.00, CameraMode::Fixed, 10.0, 9.0 },
            build_depth_of_field,
            sky_dummy_lights
        },
        {
            23, SceneNames::BilinearPatchScene,
            "Cornell box with curved bilinear patch saddle surface (pbrt-v4 BilinearPatch shape)",
            "Medium", 200, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_bilinear_patch_scene,
            build_cornell_box_lights
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
