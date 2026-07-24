// scene_descriptor.h -- Single source of truth for all scene metadata
// Shared by the Qt GUI, CPU renderer, and GPU renderer.
//
// To add a scene:
//   1. Add a row here in kAllScenes[]
//   2. Add a builder in src/TheRestOfYourLife/scenes.h
//   3. Add the scene to scene_registry.h (CPU)
//   4. Add a case in gpu/optix/scene_builder.cpp (GPU), and set gpu_supported = true

#pragma once

#ifdef __cplusplus
#include <cstddef>

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
		{  0, "Cornell Box",
		   "Classic Cornell box with glass sphere and aluminum box",
		   "Medium",    100, false, true  },
		{  1, "Bouncing Spheres",
		   "Random spheres with checker ground (In One Weekend final)",
		   "Slow",      100, false, false },
		{  2, "Checkered Spheres",
		   "Two spheres with procedural checker texture",
		   "Fast",      100, false, true  },
		{  3, "Earth",
		   "Globe with earth texture mapping (requires earthmap.jpg)",
		   "Fast",      100, true,  false },
		{  4, "Perlin Spheres",
		   "Spheres with Perlin noise marble texture",
		   "Fast",      100, false, false },
		{  5, "Colored Quads",
		   "Five colored quad primitives",
		   "Fast",      100, false, true  },
		{  6, "Simple Light",
		   "Perlin spheres with emissive light sources",
		   "Fast",      100, false, false },
		{  7, "Cornell Smoke",
		   "Cornell box with volumetric fog",
		   "Slow",      200, false, false },
		{  8, "Final Scene (very slow!)",
		   "Complex scene from The Next Week",
		   "Very Slow", 500, false, false },
		{  9, "Rough Metal Spheres (GGX)",
		   "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
		   "Medium",    200, false, false },
		{ 10, "Cornell Rough Metal (GGX)",
		   "Cornell box with rough aluminum box and rough gold sphere",
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
