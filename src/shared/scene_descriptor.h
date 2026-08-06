// scene_descriptor.h -- Single source of truth for all scene metadata
// Shared by the Qt GUI, CPU renderer, and GPU renderer.
//
// To add a scene:
//   1. Add a name constant in SceneNames below
//   2. Add a row in kAllScenes[] using that constant
//   3. Add a builder in src/TheRestOfYourLife/scenes.h
//   4. Add the scene to scene_registry.h (CPU) using the same SceneNames constant
//   5. Add a case in gpu/optix/scene_builder.cpp (GPU), and set gpu_supported = true

#pragma once

#ifdef __cplusplus
#include <cstddef>

// -----------------------------------------------------------------------
// Canonical scene name constants
// Use these everywhere (registry, GPU builder, tests) so names can never
// drift between the two tables.
// -----------------------------------------------------------------------
namespace SceneNames {
    constexpr const char* CornellBox          = "Cornell Box";
    constexpr const char* BouncingSpheres     = "Bouncing Spheres";
    constexpr const char* CheckeredSpheres    = "Checkered Spheres";
    constexpr const char* Earth               = "Earth";
    constexpr const char* PerlinSpheres       = "Perlin Spheres";
    constexpr const char* ColoredQuads        = "Colored Quads";
    constexpr const char* SimpleLight         = "Simple Light";
    constexpr const char* CornellSmoke        = "Cornell Smoke";
    constexpr const char* FinalScene          = "Final Scene";
    constexpr const char* RoughMetalSpheres   = "Rough Metal Spheres";
    constexpr const char* CornellRoughMetal   = "Cornell Rough Metal";
    constexpr const char* CornellRoughGlass   = "Cornell Rough Glass";
    constexpr const char* CornellConductor    = "Cornell Conductor";
    constexpr const char* CornellCoatedDiffuse   = "Cornell Coated Diffuse";
    constexpr const char* CornellThinGlass    = "Cornell Thin Glass";
    constexpr const char* CornellCoatedConductor = "Cornell Coated Conductor";
    constexpr const char* CornellWaxSlab      = "Cornell Wax Slab";
    constexpr const char* CornellCrystal      = "Cornell Crystal";
} // namespace SceneNames

struct SceneDesc {
	int         id;
	const char* name;
	const char* description;
	const char* performance;   // "Fast" | "Medium" | "Slow" | "Very Slow"
	int         recommended_spp;
	bool        requires_files; // needs external assets (e.g. earthmap.jpg)
	bool        gpu_supported;  // implemented in gpu/optix/scene_builder.cpp
};

// -----------------------------------------------------------------------
// Canonical list — edit ONLY here when adding / changing scenes
// -----------------------------------------------------------------------
inline const SceneDesc* get_all_scenes(int* out_count = nullptr) {
	static const SceneDesc kScenes[] = {
		{  0, SceneNames::CornellBox,
		   "Classic Cornell box with glass sphere and aluminum box",
		   "Medium",    100, false, true  },
		{  1, SceneNames::BouncingSpheres,
		   "Random spheres with checker ground (In One Weekend final)",
		   "Slow",      100, false, false },
		{  2, SceneNames::CheckeredSpheres,
		   "Two spheres with procedural checker texture",
		   "Fast",      100, false, true  },
		{  3, SceneNames::Earth,
		   "Globe with earth texture mapping (requires earthmap.jpg)",
		   "Fast",      100, true,  false },
		{  4, SceneNames::PerlinSpheres,
		   "Spheres with Perlin noise marble texture",
		   "Fast",      100, false, false },
		{  5, SceneNames::ColoredQuads,
		   "Five colored quad primitives",
		   "Fast",      100, false, true  },
		{  6, SceneNames::SimpleLight,
		   "Perlin spheres with emissive light sources",
		   "Fast",      100, false, false },
		{  7, SceneNames::CornellSmoke,
		   "Cornell box with volumetric fog",
		   "Slow",      200, false, false },
		{  8, SceneNames::FinalScene,
		   "Complex scene from The Next Week",
		   "Very Slow", 500, false, false },
		{  9, SceneNames::RoughMetalSpheres,
		   "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
		   "Medium",    200, false, false },
		{ 10, SceneNames::CornellRoughMetal,
		   "Cornell box with rough aluminum box and rough gold sphere",
		   "Medium",    200, false, true  },
		{ 11, SceneNames::CornellRoughGlass,
		   "Cornell box with a GGX rough-dielectric sphere (pbrt-v4 RoughDielectricBxDF)",
		   "Medium",    200, false, true  },
		{ 12, SceneNames::CornellConductor,
		   "Cornell box with polished gold sphere and aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)",
		   "Medium",    200, false, true  },
		{ 13, SceneNames::CornellCoatedDiffuse,
		   "Cornell box with blue coated-diffuse sphere and red coated-diffuse box (pbrt-v4 CoatedDiffuseBxDF)",
		   "Medium",    200, false, true  },
		{ 14, SceneNames::CornellThinGlass,
		   "Cornell box with a vertical thin-glass panel, analytic multi-bounce Fresnel (pbrt-v4 ThinDielectricBxDF)",
		   "Medium",    200, false, true  },
		{ 15, SceneNames::CornellCoatedConductor,
		   "Cornell box with lacquered-gold sphere and lacquered-copper box (pbrt-v4 CoatedConductorBxDF)",
		   "Medium",    200, false, true  },
		{ 16, SceneNames::CornellWaxSlab,
		   "Cornell box with a wax sphere that diffusely reflects and transmits light (pbrt-v4 DiffuseTransmissionBxDF)",
		   "Medium",    200, false, true  },
		{ 17, SceneNames::CornellCrystal,
		   "Cornell box with a crystal sphere using Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)",
		   "Medium",    200, false, true  },
	};
	static const int kCount = (int)(sizeof(kScenes) / sizeof(kScenes[0]));
	if (out_count) *out_count = kCount;
	return kScenes;
}

inline const SceneDesc* find_scene_desc(int id) {
	int count = 0;
	const SceneDesc* scenes = get_all_scenes(&count);
	for (int i = 0; i < count; ++i)
		if (scenes[i].id == id) return &scenes[i];
	return nullptr;
}

inline int scene_desc_count() {
	int count = 0;
	get_all_scenes(&count);
	return count;
}

#endif // __cplusplus
