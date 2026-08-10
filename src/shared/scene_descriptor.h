// scene_descriptor.h -- Canonical scene NAME constants only.
//
// All scene METADATA (description, performance hint, recommended SPP,
// requires_files, gpu_compatible, camera) lives solely in
// src/TheRestOfYourLife/scene_registry.h's SceneDescriptor table, queried
// live by the Qt GUI via scene_metadata.dll / qt_gui/scene_metadata_client.h
// - there is exactly one place that can drift from actual renderer
// behavior, not two. This header used to also carry its own duplicate
// presentational table (SceneDesc/kAllScenes/get_all_scenes/
// find_scene_desc) so the GUI - which can't link scene_registry.h's full
// CPU hittable/material class hierarchy - could read it without the DLL
// bridge; that duplication drifted out of sync once already (scene 1 got
// gpu_compatible=true in the registry without its mirror row here being
// updated, so the GUI kept showing "CPU only" until fixed separately) and
// was removed once scene_metadata.dll grew accessors for every field, not
// just camera/gpu_compatible.
//
// SceneNames stays here (rather than moving into scene_registry.h itself)
// because it's a lightweight, dependency-free header both scene_registry.h
// (CPU) and gpu/optix/scene_builder.cpp (GPU) #include just for these
// string constants, without pulling in anything heavier.
//
// To add a scene:
//   1. Add a name constant in SceneNames below
//   2. Add a builder in src/TheRestOfYourLife/scenes.h (or scenes_book.h/
//      scenes_advanced.h)
//   3. Add the scene's row to scene_registry.h (CPU) using the same
//      SceneNames constant, including its CameraConfig - scene_metadata.dll
//      serves every field of this row to the GUI live, no separate step
//      needed here
//   4. Optionally add a case in gpu/optix/scene_builder.cpp (GPU) and set
//      scene_registry.h's gpu_compatible = true - scene_metadata.dll picks
//      this up automatically too

#pragma once

#ifdef __cplusplus
#include <cstddef>

// -----------------------------------------------------------------------
// Canonical scene name constants
// Use these everywhere (registry, GPU builder, tests) rather than raw
// string literals, so a scene's name can never drift between call sites.
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
    constexpr const char* TriangleMesh         = "Triangle Mesh";
    constexpr const char* StanfordBunny        = "Stanford Bunny";
    constexpr const char* StanfordArmadillo    = "Stanford Armadillo";
    constexpr const char* StanfordHappyBuddha  = "Stanford Happy Buddha";
} // namespace SceneNames

#endif // __cplusplus
