#pragma once
#include <cstring>

// video_preset.h -- Named bundles of video-mode parameters (scene + camera
// path + frame count/fps/speed), the video-mode analog of SceneDescriptor
// (scene_registry.h): pick one thing instead of tuning five.
//
// Deliberately header-only, plain-data, and dependency-free - unlike scene
// metadata, a preset needs nothing from the CPU renderer's hittable/material
// class hierarchy (no build_world-style callback), so there is no reason to
// duplicate it behind scene_metadata.dll the way SceneDescriptor's fields
// are. This one file is #include'd directly by both launcher/ (MSVC, CLI)
// and qt_gui/ (MinGW, GUI) - see mainwindow_tabs.cpp's existing
// `#include "../src/shared/scene_descriptor.h"` for the established
// precedent of a qt_gui source file reaching directly into src/shared/.
//
// Each preset's scene_id/camera_path must be valid values accepted
// elsewhere already (scene_id: a real SceneDescriptor::id from
// scene_registry.h; camera_path: one of camera_path.h's named path types) -
// nothing here re-validates that; an invalid scene_id fails exactly the way
// a hand-typed --scene_id would, and an unrecognized camera_path silently
// falls back to "orbit" (see camera_path.h's own get_camera_position()).

namespace video_preset {

struct VideoPreset {
    const char* id;           // stable key, e.g. "cornell-orbit" - used by --video-preset
    const char* name;         // "Cornell Box Orbit" - shown in the GUI combo
    const char* description;  // one line, shown as a tooltip
    const char* scene_id;     // SceneDescriptor::id, e.g. "A1"
    const char* camera_path;  // one of camera_path.h's named types
    int frames;
    int fps;
    double speed;
};

// Six presets, each pairing a distinct scene already in the registry with a
// camera path already implemented in camera_path.h - no new rendering
// features, just known-good bundles of existing ones. Ordered roughly by
// render cost (fastest first) so the GUI combo's default selection is cheap
// to try.
inline const VideoPreset kAll[] = {
    {
        "cornell-orbit", "Cornell Box Orbit",
        "A slow 360-degree orbit around ray tracing's most iconic reference scene.",
        "A1", "orbit", 90, 30, 1.0
    },
    {
        "teapot-spin", "Utah Teapot Spin",
        "The Utah teapot - computer graphics' most iconic test object - rotating in place.",
        "G6", "orbit", 90, 30, 1.0
    },
    {
        "one-weekend-flyby", "One Weekend Flyby",
        "A flythrough of the Ray Tracing in One Weekend cover scene - hundreds of "
        "random spheres with real per-object motion blur.",
        "A2", "linear", 90, 30, 0.8
    },
    {
        "next-week-finale", "Next Week Finale",
        "The Ray Tracing: The Next Week finale scene - moving spheres, volumes, and "
        "a subsurface sphere over a box-grid ground.",
        "A9", "figure8", 120, 30, 0.6
    },
    {
        "glass-dragon-caustics", "Glass Dragon Caustics",
        "A spiral zoom into a refractive glass dragon, showing off dielectric "
        "caustics as the camera closes in.",
        "G13", "spiral", 120, 30, 0.5
    },
    {
        "sponza-flythrough", "Sponza Flythrough",
        "A flythrough of Crytek Sponza, the rendering community's canonical "
        "global-illumination showcase scene.",
        "H1", "linear", 120, 24, 0.5
    },
};
inline constexpr int kAllCount = sizeof(kAll) / sizeof(kAll[0]);

inline const VideoPreset* find(const char* id) {
    if (!id) return nullptr;
    for (const VideoPreset& p : kAll) {
        if (std::strcmp(p.id, id) == 0) return &p;
    }
    return nullptr;
}

} // namespace video_preset
