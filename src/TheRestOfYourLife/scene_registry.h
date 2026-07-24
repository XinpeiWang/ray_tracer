#pragma once
// scene_registry.h -- Single source of truth for all scenes (pbrt-v4 SceneEntity pattern)
//
// To add a new scene:
//   1. Add a builder function in scenes.h
//   2. Add one SceneDescriptor entry in get_scene_registry() below
//   Done -- cpu_interface and GUI pick it up automatically.

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
            0, "Cornell Box",
            "Classic Cornell box with glass sphere and aluminum box",
            "Medium", 100, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            1, "Bouncing Spheres",
            "Random spheres with checker ground (In One Weekend final)",
            "Slow", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_bouncing_spheres,
            sky_dummy_lights
        },
        {
            2, "Checkered Spheres",
            "Two spheres with procedural checker texture",
            "Fast", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_checkered_spheres,
            sky_dummy_lights
        },
        {
            3, "Earth",
            "Globe with earth texture mapping (requires earthmap.jpg)",
            "Fast", 100, true, false,
            { 20, 0, 0, 12,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_earth,
            sky_dummy_lights
        },
        {
            4, "Perlin Spheres",
            "Spheres with Perlin noise marble texture",
            "Fast", 100, false, false,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_perlin_spheres,
            sky_dummy_lights
        },
        {
            5, "Colored Quads",
            "Five colored quad primitives",
            "Fast", 100, false, false,
            { 80, 0, 0, 9,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_quads,
            sky_dummy_lights
        },
        {
            6, "Simple Light",
            "Perlin spheres with emissive light sources",
            "Fast", 100, false, false,
            { 20, 26, 3, 6,  0, 2, 0,  0, 0, 0 },
            build_simple_light,
            no_lights
        },
        {
            7, "Cornell Smoke",
            "Cornell box with volumetric fog",
            "Slow", 200, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_smoke,
            build_cornell_box_lights
        },
        {
            8, "Final Scene",
            "Complex scene from The Next Week (very slow!)",
            "Very Slow", 500, false, false,
            { 40, 478, 278, -600,  278, 278, 0,  0, 0, 0 },
            build_final_scene,
            sky_dummy_lights
        },
        {
            9, "Rough Metal Spheres (GGX)",
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
            10, "Cornell Rough Metal (GGX)",
            "Cornell box with rough aluminum box and rough gold sphere",
            "Medium", 200, false, false,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_rough_metal,
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
