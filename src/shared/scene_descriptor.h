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
    constexpr const char* PrincipledShowcase   = "Principled Showcase";
    constexpr const char* HairFibers           = "Hair Fibers";
    constexpr const char* NormalMappedCornell  = "Normal Mapped Cornell";
    constexpr const char* SubsurfaceSlab       = "Subsurface Slab";
    constexpr const char* DepthOfField         = "Depth of Field";
    constexpr const char* BilinearPatchScene   = "Bilinear Patch";
    // pbrt-v4 light / camera / medium showcase scenes
    constexpr const char* HdriSky              = "HDRI Sky";
    constexpr const char* SpotlightCornell     = "Spotlight Cornell";
    constexpr const char* DistantLightCornell  = "Distant Light Cornell";
    constexpr const char* PointLightCornell    = "Point Light Cornell";
    constexpr const char* GoniometricLight     = "Goniometric Light";
    constexpr const char* ProjectionLight      = "Projection Light";
    constexpr const char* HomogeneousMedium    = "Homogeneous Medium";
    constexpr const char* CloudMedium          = "Cloud Medium";
    constexpr const char* OrthographicCamera   = "Orthographic Camera";
    constexpr const char* SphericalCamera      = "Spherical Camera";
    constexpr const char* MeasuredBrdf         = "Measured BRDF";
    constexpr const char* PortalInfiniteLight  = "Portal Infinite Light";
    constexpr const char* RealisticCamera      = "Realistic Camera";
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
		{ 18, SceneNames::PrincipledShowcase,
		   "Row of spheres from matte plastic to metallic with clearcoat (pbrt-v4 PrincipledBxDF)",
		   "Medium",    200, false, false },
		{ 19, SceneNames::HairFibers,
		   "Sphere cluster with hair/fur fiber scattering (pbrt-v4 HairBxDF)",
		   "Medium",    200, false, false },
		{ 20, SceneNames::NormalMappedCornell,
		   "Cornell box with procedural bump-mapped back wall and normal-mapped sphere (pbrt-v4 NormalMap/BumpMap)",
		   "Medium",    200, false, false },
		{ 21, SceneNames::SubsurfaceSlab,
		   "Cornell box with translucent wax slab and jade sphere using subsurface-like scattering",
		   "Slow",      300, false, false },
		{ 22, SceneNames::DepthOfField,
		   "Row of spheres with defocus blur showing depth-of-field from the thin-lens camera model",
		   "Medium",    200, false, false },
		{ 23, SceneNames::BilinearPatchScene,
		   "Cornell box with curved bilinear patch saddle surface (pbrt-v4 BilinearPatch shape)",
		   "Medium",    200, false, false },
		// pbrt-v4 light / camera / medium showcase scenes
		{ 24, SceneNames::HdriSky,
		   "Open scene lit by a procedural gradient sky (pbrt-v4 ImageInfiniteLight / sky_light)",
		   "Medium",    200, false, false },
		{ 25, SceneNames::SpotlightCornell,
		   "Cornell box lit by a spotlight with smooth penumbra (pbrt-v4 SpotLight)",
		   "Medium",    200, false, true  },
		{ 26, SceneNames::DistantLightCornell,
		   "Cornell box lit by a parallel sun-like distant light (pbrt-v4 DistantLight)",
		   "Medium",    200, false, true  },
		{ 27, SceneNames::PointLightCornell,
		   "Cornell box lit by a single overhead point light with 1/r^2 falloff (pbrt-v4 PointLight)",
		   "Medium",    200, false, true  },
		{ 28, SceneNames::GoniometricLight,
		   "Cornell box lit by a goniometric (IES-profile) point light (pbrt-v4 GoniometricLight)",
		   "Medium",    200, false, true  },
		{ 29, SceneNames::ProjectionLight,
		   "Cornell box with a slide-projector beam casting a checkerboard pattern (pbrt-v4 ProjectionLight)",
		   "Medium",    200, false, true  },
		{ 30, SceneNames::HomogeneousMedium,
		   "Cornell box filled with a homogeneous scattering fog (pbrt-v4 HomogeneousMedium / HenyeyGreenstein)",
		   "Slow",      300, false, false },
		{ 31, SceneNames::CloudMedium,
		   "Open scene with a procedural Perlin-noise cloud volume (pbrt-v4 CloudMedium)",
		   "Slow",      300, false, false },
		{ 32, SceneNames::OrthographicCamera,
		   "Geometric showcase rendered with an orthographic (parallel-projection) camera (pbrt-v4 OrthographicCamera)",
		   "Fast",      100, false, false },
		{ 33, SceneNames::SphericalCamera,
		   "360-degree equirectangular panorama from a spherical camera (pbrt-v4 SphericalCamera)",
		   "Medium",    200, false, false },
		{ 34, SceneNames::MeasuredBrdf,
		   "Sphere cluster with measured BRDF material using tabulated RGL data (pbrt-v4 MeasuredBxDF)",
		   "Medium",    200, false, false },
		{ 35, SceneNames::PortalInfiniteLight,
		   "Room scene with a portal window sampling the sky through a planar quad (pbrt-v4 PortalImageInfiniteLight)",
		   "Slow",      300, false, false },
		{ 36, SceneNames::RealisticCamera,
		   "Spheres rendered through a thin-lens with realistic lens-element bokeh (pbrt-v4 RealisticCamera)",
		   "Medium",    200, false, false },
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
