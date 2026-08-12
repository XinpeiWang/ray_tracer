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
    constexpr const char* StanfordLucy         = "Stanford Lucy";
    constexpr const char* StanfordDragon       = "Stanford XYZRGB Dragon";
    constexpr const char* UtahTeapot           = "Utah Teapot";
    constexpr const char* SpotCow              = "Spot the Cow";
    constexpr const char* Suzanne              = "Suzanne";
    constexpr const char* NefertitiBust        = "Nefertiti Bust";
    constexpr const char* Horse                = "Horse";
    constexpr const char* Cheburashka          = "Cheburashka";
    constexpr const char* TrophyRoom           = "Trophy Room";
    constexpr const char* GlassDragon          = "Glass Dragon";
    constexpr const char* Beast                = "Beast";
    constexpr const char* VWBeetle             = "VW Beetle";
    constexpr const char* VWBeetleAlt          = "VW Beetle (Alt)";
    constexpr const char* Bimba                = "Bimba";
    constexpr const char* Cow                  = "Cow";
    constexpr const char* Fandisk              = "Fandisk";
    constexpr const char* Homer                = "Homer";
    constexpr const char* Igea                 = "Igea";
    constexpr const char* MaxPlanck            = "Max Planck";
    constexpr const char* Ogre                 = "Ogre";
    constexpr const char* RockerArm            = "Rocker Arm";
    constexpr const char* CrytekSponza         = "Crytek Sponza";
    constexpr const char* AmazonBistro         = "Amazon Lumberyard Bistro";
    constexpr const char* Rungholt             = "Rungholt";
} // namespace SceneNames

// -----------------------------------------------------------------------
// Canonical scene category constants
// -----------------------------------------------------------------------
// Categories group scenes by WHAT THEY DEMONSTRATE, which is the question a
// user browsing 65 scenes is actually asking - not by which book chapter or
// source file they came from. The Qt GUI turns these into a filter tab bar
// above the scene list (see qt_gui/mainwindow_tabs.cpp).
//
// Same rule as SceneNames: use these constants in the registry, never raw
// string literals, so a category can't drift between call sites. kAll is the
// display order for the tabs and is what tests/unit/scene_registry_tests.cpp
// validates every scene's category against - a typo'd category would
// otherwise silently produce a scene that appears under no tab at all.
namespace SceneCategories {
    constexpr const char* Basics     = "Basics";       // the book scenes
    constexpr const char* Materials  = "Materials";    // BxDF / surface appearance
    constexpr const char* Lights     = "Lights";       // light types and sampling
    constexpr const char* Cameras    = "Cameras";      // projection and lens models
    constexpr const char* Volumes    = "Volumes";      // participating media
    constexpr const char* Geometry   = "Geometry";     // shape primitives
    constexpr const char* Models     = "Models";       // single imported meshes
    constexpr const char* LargeScene = "Large Scenes"; // full textured environments
    // Scenes loaded from .pbrt files found on disk rather than compiled in.
    // Unlike every category above it, this one is populated at runtime and is
    // legitimately empty when the user has no scene collection installed -
    // which is why the registry tests exempt it from "every category has at
    // least one scene".
    constexpr const char* UserScenes = "User Scenes";

    // Display order for the GUI's category tabs. UserScenes is last so the
    // built-in tabs never shift position when a scene folder appears.
    constexpr const char* kAll[] = {
        Basics, Materials, Lights, Cameras, Volumes, Geometry, Models, LargeScene,
        UserScenes
    };
    constexpr std::size_t kAllCount = sizeof(kAll) / sizeof(kAll[0]);
} // namespace SceneCategories

#endif // __cplusplus
