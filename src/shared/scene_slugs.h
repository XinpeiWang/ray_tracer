#pragma once
// scene_slugs.h -- stable string keys for scenes, and the rule that makes them.
//
// A scene's slug ("cornell-box") is its durable identity: it does not change when a category is added, when a file is dropped into pbrt_scenes/, or
// when a scene is renamed for display. The id ("A1", "K37") is a short code that is only as stable as the registry order, so anything that is saved
// or typed by a person (recent renders, scripts, the command line, notes keyed to a scene) should use the slug; the id stays accepted everywhere as
// an alias. SceneDescriptor::slug is filled from kBuiltin below for compiled-in scenes and from the file name for .pbrt files found on disk.
//
// Slugs are lower-case, digits and single hyphens only (no underscore, so a name like "render_<slug>_<time>.png" splits unambiguously).

#include <cctype>
#include <cstddef>
#include <set>
#include <string>

namespace scene_slugs {

// "Depth of Field (pbrt file)" -> "depth-of-field-pbrt-file"; any run of characters outside a-z and 0-9 becomes one hyphen; "&" reads as "and".
inline std::string slugify(const std::string& text) {
    std::string spaced;
    for (char c : text) {
        if (c == '&') spaced += " and ";
        else spaced += c;
    }
    std::string out;
    bool pendingHyphen = false;
    for (unsigned char c : spaced) {
        if (std::isalnum(c) && c < 128) {
            if (pendingHyphen && !out.empty()) out += '-';
            pendingHyphen = false;
            out += static_cast<char>(std::tolower(c));
        } else {
            pendingHyphen = true;
        }
    }
    return out.empty() ? std::string("scene") : out;
}

// The base slug, or base-2, base-3, ...: the first one not already in the taken set (it is added to the set).
inline std::string uniqueSlug(const std::string& base, std::set<std::string>& taken) {
    std::string slug = base;
    for (int n = 2; taken.count(slug); ++n) slug = base + "-" + std::to_string(n);
    taken.insert(slug);
    return slug;
}

// True for a string shaped like a slug: starts with a letter, then lower-case letters, digits and hyphens. (An id such as "B10" is upper case and
// never matches, which is how the two kinds of key are told apart.)
inline bool looksLikeSlug(const std::string& s) {
    if (s.empty() || !(s[0] >= 'a' && s[0] <= 'z')) return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return false;
    return true;
}

struct Entry {
    const char* id;
    const char* slug;
};

// The compiled-in scenes. Written out rather than derived from the names so that renaming a scene for display never changes its slug.
constexpr Entry kBuiltin[] = {
        {"A1", "cornell-box"},   // Cornell Box
        {"A2", "bouncing-spheres"},   // Bouncing Spheres
        {"A3", "checkered-spheres"},   // Checkered Spheres
        {"A4", "earth"},   // Earth
        {"A5", "perlin-spheres"},   // Perlin Spheres
        {"A6", "colored-quads"},   // Colored Quads
        {"A7", "simple-light"},   // Simple Light
        {"A8", "cornell-smoke"},   // Cornell Smoke
        {"A9", "final-scene"},   // Final Scene
        {"B1", "rough-metal-spheres"},   // Rough Metal Spheres
        {"B2", "cornell-rough-metal"},   // Cornell Rough Metal
        {"B3", "cornell-rough-glass"},   // Cornell Rough Glass
        {"B4", "cornell-conductor"},   // Cornell Conductor
        {"B5", "cornell-coated-diffuse"},   // Cornell Coated Diffuse
        {"B6", "cornell-thin-glass"},   // Cornell Thin Glass
        {"B7", "cornell-coated-conductor"},   // Cornell Coated Conductor
        {"B8", "cornell-wax-slab"},   // Cornell Wax Slab
        {"B9", "cornell-crystal"},   // Cornell Crystal
        {"B10", "principled-showcase"},   // Principled Showcase
        {"B11", "hair-fibers"},   // Hair Fibers
        {"B12", "normal-mapped-cornell"},   // Normal Mapped Cornell
        {"B13", "subsurface-slab"},   // Subsurface Slab
        {"B14", "measured-brdf"},   // Measured BRDF
        {"B15", "mix-material"},   // Mix Material
        {"B16", "layered-materials"},   // Layered Materials
        {"B18", "conductor-rgb-eta-k"},   // Conductor RGB Eta/K
        {"B20", "hair-material"},   // Hair Material
        {"B22", "named-material-and-texture"},   // Named Material & Texture
        {"B23", "glass-prism-dispersion"},   // Glass Prism Dispersion
        {"B24", "frosted-prism-dispersion"},   // Frosted Prism Dispersion
        {"B25", "glass-presets"},   // Glass Presets
        {"C1", "hdri-sky"},   // HDRI Sky
        {"C2", "spotlight-cornell"},   // Spotlight Cornell
        {"C3", "distant-light-cornell"},   // Distant Light Cornell
        {"C4", "point-light-cornell"},   // Point Light Cornell
        {"C5", "goniometric-light"},   // Goniometric Light
        {"C6", "projection-light"},   // Projection Light
        {"C7", "portal-infinite-light"},   // Portal Infinite Light
        {"C8", "punctual-lights"},   // Punctual Lights
        {"C9", "goniometric-and-projection-lights"},   // Goniometric & Projection Lights
        {"C10", "blackbody-light"},   // Blackbody Light
        {"C11", "textured-two-sided-lights"},   // Textured Two-Sided Lights
        {"C12", "infinite-light"},   // Infinite Light
        {"C13", "disk-and-cylinder-lights"},   // Disk & Cylinder Lights
        {"C14", "two-sphere-lights"},   // Two Sphere Lights
        {"C15", "triangle-fan-light"},   // Triangle Fan Light
        {"C16", "colorspace-blackbody-light"},   // ColorSpace + Blackbody Light
        {"C17", "portal-light"},   // Portal Light
        {"C18", "light-power-parameter"},   // Light Power Parameter
        {"C19", "projection-light-non-square"},   // Projection Light: Non-Square
        {"C20", "spectral-gamut-saturation"},   // Spectral Gamut Saturation
        {"D1", "depth-of-field"},   // Depth of Field
        {"D2", "orthographic-camera"},   // Orthographic Camera
        {"D3", "spherical-camera"},   // Spherical Camera
        {"D4", "realistic-camera"},   // Realistic Camera
        {"D5", "depth-of-field-cornell-box"},   // Depth of Field (Cornell Box)
        {"D6", "orthographic-camera-cornell-box"},   // Orthographic Camera (Cornell Box)
        {"D7", "spherical-camera-cornell-box"},   // Spherical Camera (Cornell Box)
        {"D8", "realistic-camera-cornell-box"},   // Realistic Camera (Cornell Box)
        {"D9", "depth-of-field-pbrt-file"},   // Depth of Field (pbrt file)
        {"D10", "orthographic-camera-pbrt-file"},   // Orthographic Camera (pbrt file)
        {"D11", "spherical-camera-pbrt-file"},   // Spherical Camera (pbrt file)
        {"D12", "realistic-camera-pbrt-file"},   // Realistic Camera (pbrt file)
        {"D13", "camera-motion-blur-cornell-box"},   // Camera Motion Blur (Cornell Box)
        {"E1", "homogeneous-medium"},   // Homogeneous Medium
        {"E2", "cloud-medium"},   // Cloud Medium
        {"E3", "dielectric-medium-showcase"},   // Dielectric Medium Showcase
        {"E4", "rgb-grid-medium"},   // RGB Grid Medium
        {"E5", "cloud-medium-pbrt-file"},   // Cloud Medium (pbrt file)
        {"E6", "cylinder-medium"},   // Cylinder Medium
        {"E7", "rgb-grid-medium-pbrt-file"},   // RGB Grid Medium (pbrt file)
        {"E8", "uniform-grid-medium"},   // Uniform Grid Medium
        {"E9", "nanovdb-medium"},   // NanoVDB Medium
        {"E10", "camera-medium"},   // Camera Medium
        {"E11", "thin-dielectric-medium"},   // Thin Dielectric Medium
        {"E12", "rough-dielectric-medium"},   // Rough Dielectric Medium
        {"F1", "bilinear-patch"},   // Bilinear Patch
        {"F2", "triangle-mesh"},   // Triangle Mesh
        {"F3", "instanced-spheres"},   // Instanced Spheres
        {"F4", "curve-fibers"},   // Curve Fibers
        {"F5", "ply-mesh-uv"},   // PLY Mesh UV
        {"F6", "ply-mesh-geometry"},   // PLY Mesh Geometry
        {"F7", "curve-tuft"},   // Curve Tuft
        {"F8", "curve-hair-tuft"},   // Curve + Hair Tuft
        {"F9", "triangle-mesh-uv"},   // Triangle Mesh UV
        {"F10", "pixel-filter-box"},   // Pixel Filter: Box
        {"F11", "object-motion-blur"},   // Object Motion Blur
        {"F12", "disk-and-cylinder-motion-blur"},   // Disk & Cylinder Motion Blur
        {"F13", "reverseorientation"},   // ReverseOrientation
        {"F14", "cone-and-paraboloid-gallery"},   // Cone & Paraboloid Gallery
        {"G1", "stanford-bunny"},   // Stanford Bunny
        {"G2", "stanford-armadillo"},   // Stanford Armadillo
        {"G3", "stanford-happy-buddha"},   // Stanford Happy Buddha
        {"G4", "stanford-lucy"},   // Stanford Lucy
        {"G5", "stanford-xyzrgb-dragon"},   // Stanford XYZRGB Dragon
        {"G6", "utah-teapot"},   // Utah Teapot
        {"G7", "spot-the-cow"},   // Spot the Cow
        {"G8", "suzanne"},   // Suzanne
        {"G9", "nefertiti-bust"},   // Nefertiti Bust
        {"G10", "horse"},   // Horse
        {"G11", "cheburashka"},   // Cheburashka
        {"G12", "trophy-room"},   // Trophy Room
        {"G13", "glass-dragon"},   // Glass Dragon
        {"G14", "beast"},   // Beast
        {"G15", "vw-beetle"},   // VW Beetle
        {"G17", "bimba"},   // Bimba
        {"G18", "cow"},   // Cow
        {"G19", "fandisk"},   // Fandisk
        {"G20", "homer"},   // Homer
        {"G21", "igea"},   // Igea
        {"G22", "max-planck"},   // Max Planck
        {"G23", "ogre"},   // Ogre
        {"G24", "rocker-arm"},   // Rocker Arm
        {"G25", "killeroo"},   // Killeroo
        {"H1", "crytek-sponza"},   // Crytek Sponza
        {"H2", "amazon-lumberyard-bistro"},   // Amazon Lumberyard Bistro
        {"H3", "rungholt"},   // Rungholt
        {"H4", "fireplace-room"},   // Fireplace Room
        {"H5", "san-miguel"},   // San Miguel
        {"H6", "sibenik-cathedral"},   // Sibenik Cathedral
        {"H7", "breakfast-room"},   // Breakfast Room
        {"H8", "salle-de-bain"},   // Salle de Bain
        {"H9", "gallery"},   // Gallery
        {"H10", "lost-empire"},   // Lost Empire
        {"H11", "vokselia-spawn"},   // Vokselia Spawn
        {"H12", "power-plant"},   // Power Plant
        {"H13", "contemporary-bathroom"},   // Contemporary Bathroom
        {"H14", "barcelona-pavilion"},   // Barcelona Pavilion
        {"H15", "subsurface-dragon"},   // Subsurface Dragon
        {"H16", "ganesha"},   // Ganesha
        {"H17", "sports-car"},   // Sports Car
        {"H18", "zero-day"},   // Zero Day
        {"H19", "crown"},   // Crown
        {"H20", "villa"},   // Villa
        {"H21", "transparent-machines"},   // Transparent Machines
        {"I1", "sampler-comparison"},   // Sampler Comparison (Low Sample Count)
        {"I2", "spectral-prism-dispersion"},   // Spectral Rendering: Prism Dispersion
        {"I3", "exposure-tone-mapping"},   // Exposure & Tone Mapping (HDR Sky)
        {"I4", "gpu-denoiser-demo"},   // GPU Denoiser: Before & After
        {"I5", "sppm-rough-glass-caustic"},   // SPPM: Rough Glass Caustic
        {"I6", "bdpt-mlt-demo"},   // BDPT / MLT: Bidirectional Light Transport
        {"I7", "light-transport-strategies"},   // Light Transport Strategies (NEE / BSDF / MIS)
        {"I8", "light-sampler-comparison"},   // Light Sampler Strategy (Uniform / Power / BVH)
        {"I9", "ambient-occlusion-demo"},   // Ambient Occlusion (Debug Integrator)
        {"I10", "firefly-suppression"},   // Firefly Suppression (Regularize / Clamp)
        {"J1", "coateddiffuse-texture"},   // CoatedDiffuse Texture
        {"J2", "diffusetransmission-texture"},   // DiffuseTransmission Texture
        {"J3", "nested-checker-texture"},   // Nested Checker Texture
        {"J4", "texture-encoding-and-wrap"},   // Texture Encoding & Wrap
        {"J5", "procedural-texture-gallery"},   // Procedural Texture Gallery
        {"J6", "nested-texture-2-levels"},   // Nested Texture: 2 Levels
};
constexpr std::size_t kBuiltinCount = sizeof(kBuiltin) / sizeof(kBuiltin[0]);

// The slug of a compiled-in scene, or "" for an id that has none (a scene found on disk).
inline std::string builtinSlugForId(const std::string& id) {
    for (const Entry& e : kBuiltin)
        if (id == e.id) return e.slug;
    return std::string();
}

} // namespace scene_slugs
