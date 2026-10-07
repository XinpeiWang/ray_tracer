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
// To add a scene (docs/SCENE_SELECTION.md has the full list):
//   1. Put its .pbrt file in pbrt_scenes/
//   2. Add a name constant in SceneNames below (what the scene shows, not how it is stored)
//   3. Add its durable name to src/shared/scene_slugs.h (kBuiltin)
//   4. Add its row to src/TheRestOfYourLife/scene_registry_data.h (build_curated_pbrt_scene_descriptor), using the SceneNames constant
//   5. Add its technique note to qt_gui/scene_technique_notes.h, keyed by the slug
// scene_metadata.dll serves every field of the row to the GUI live. A test fixture or a scene of your own needs none of this: a .pbrt file in
// pbrt_scenes/ is listed on its own (see "# @rt-category" in pbrt_discover.h).

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstring>

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
    constexpr const char* GlassPrismDispersion = "Glass Prism Dispersion";
    constexpr const char* FrostedPrismDispersion = "Frosted Prism Dispersion";
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
    constexpr const char* DielectricMediumShowcase = "Dielectric Medium Showcase";
    constexpr const char* RgbGridMedium        = "RGB Grid Medium";
    constexpr const char* OrthographicCamera   = "Orthographic Camera";
    constexpr const char* SphericalCamera      = "Spherical Camera";
    constexpr const char* MeasuredBrdf         = "Measured BRDF";
    constexpr const char* PortalInfiniteLight  = "Portal Infinite Light";
    constexpr const char* RealisticCamera      = "Realistic Camera";
    // Same classic Cornell box as A1 (CornellBox), rendered with each of the
    // other four camera models - a fixed reference scene makes the actual
    // camera differences (blur, projection, distortion) easier to compare
    // than each D1-D4 scene's own bespoke geometry does.
    constexpr const char* DepthOfFieldCornellBox      = "Depth of Field (Cornell Box)";
    constexpr const char* OrthographicCameraCornellBox = "Orthographic Camera (Cornell Box)";
    constexpr const char* SphericalCameraCornellBox    = "Spherical Camera (Cornell Box)";
    constexpr const char* RealisticCameraCornellBox    = "Realistic Camera (Cornell Box)";
    constexpr const char* CameraMotionBlur             = "Camera Motion Blur (Cornell Box)";
    constexpr const char* TriangleMesh         = "Triangle Mesh";
    constexpr const char* InstancedSpheres     = "Instanced Spheres";
    constexpr const char* CurveFibers          = "Curve Fibers";
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
    constexpr const char* FireplaceRoom        = "Fireplace Room";
    constexpr const char* SanMiguel            = "San Miguel";
    constexpr const char* SibenikCathedral     = "Sibenik Cathedral";
    constexpr const char* BreakfastRoom        = "Breakfast Room";
    constexpr const char* SalleDeBain          = "Salle de Bain";
    constexpr const char* Gallery              = "Gallery";
    constexpr const char* LostEmpire           = "Lost Empire";
    constexpr const char* VokseliaSpawn        = "Vokselia Spawn";
    constexpr const char* PowerPlant           = "Power Plant";
    constexpr const char* ContemporaryBathroomPbrtExample = "Contemporary Bathroom";
    constexpr const char* BarcelonaPavilionPbrtExample     = "Barcelona Pavilion";
    constexpr const char* SssDragonPbrtExample             = "Subsurface Dragon";
    constexpr const char* GaneshaPbrtExample               = "Ganesha";
    constexpr const char* SportsCarPbrtExample             = "Sports Car";
    constexpr const char* ZeroDayPbrtExample                = "Zero Day";
    constexpr const char* CrownPbrtExample                  = "Crown";
    constexpr const char* VillaPbrtExample                  = "Villa";
    constexpr const char* TransparentMachinesPbrtExample    = "Transparent Machines";

    // Curated entries for self-contained pbrt_scenes/*.pbrt example files (see
    // scene_registry.h's pbrt_scene_registry::build_curated_pbrt_scene_
    // descriptor() and its call sites), shown under their real topic tab. A
    // name says what the scene shows, not how it is stored; the one exception
    // is "(pbrt file)", added only where a compiled-in scene already has the
    // plain name (Depth of Field, Cloud Medium, ...) so the two can be told
    // apart. The constant names keep their old "PbrtExample" suffix.
    constexpr const char* MixMaterialPbrtExample              = "Mix Material";
    constexpr const char* LayeredMaterialsPbrtExample         = "Layered Materials";
    constexpr const char* CoatedDiffuseTexturePbrtExample     = "CoatedDiffuse Texture";
    constexpr const char* ConductorRgbEtaKPbrtExample         = "Conductor RGB Eta/K";
    constexpr const char* DiffuseTransmissionTexturePbrtExample = "DiffuseTransmission Texture";
    constexpr const char* HairMaterialPbrtExample             = "Hair Material";
    constexpr const char* NestedCheckerTexturePbrtExample     = "Nested Checker Texture";
    constexpr const char* NamedMaterialAndTexturePbrtExample  = "Named Material & Texture";
    constexpr const char* GlassPresetsPbrtExample              = "Glass Presets";
    constexpr const char* TextureEncodingWrapInvertPbrtExample = "Texture Encoding & Wrap";
    constexpr const char* ProceduralTextureGalleryPbrtExample  = "Procedural Texture Gallery";
    constexpr const char* NestedTexture2LevelPbrtExample       = "Nested Texture: 2 Levels";

    constexpr const char* PunctualLightsPbrtExample           = "Punctual Lights";
    constexpr const char* GoniometricProjectionPbrtExample    = "Goniometric & Projection Lights";
    constexpr const char* BlackbodyLightPbrtExample           = "Blackbody Light";
    constexpr const char* ColorSpaceBlackbodyPbrtExample       = "ColorSpace + Blackbody Light";
    constexpr const char* TexturedTwoSidedLightsPbrtExample   = "Textured Two-Sided Lights";
    constexpr const char* InfiniteLightPbrtExample            = "Infinite Light";
    constexpr const char* DiskCylinderLightPbrtExample        = "Disk & Cylinder Lights";
    constexpr const char* TwoSphereLightsPbrtExample          = "Two Sphere Lights";
    constexpr const char* TriangleFanLightPbrtExample         = "Triangle Fan Light";
    constexpr const char* PortalLightPbrtExample               = "Portal Light";
    constexpr const char* LightPowerParameterPbrtExample       = "Light Power Parameter";
    constexpr const char* ProjectionLightNonSquarePbrtExample  = "Projection Light: Non-Square";
    constexpr const char* SpectralGamutSaturationPbrtExample   = "Spectral Gamut Saturation";

    constexpr const char* DepthOfFieldPbrtExample             = "Depth of Field (pbrt file)";
    constexpr const char* OrthographicCameraPbrtExample       = "Orthographic Camera (pbrt file)";
    constexpr const char* SphericalCameraPbrtExample          = "Spherical Camera (pbrt file)";
    constexpr const char* RealisticCameraPbrtExample          = "Realistic Camera (pbrt file)";

    constexpr const char* CloudMediumPbrtExample              = "Cloud Medium (pbrt file)";
    constexpr const char* CylinderMediumPbrtExample           = "Cylinder Medium";
    constexpr const char* RgbGridMediumPbrtExample            = "RGB Grid Medium (pbrt file)";
    constexpr const char* UniformGridMediumPbrtExample        = "Uniform Grid Medium";
    constexpr const char* NanoVdbMediumPbrtExample             = "NanoVDB Medium";
    constexpr const char* CameraMediumPbrtExample              = "Camera Medium";
    constexpr const char* ThinDielectricMediumPbrtExample      = "Thin Dielectric Medium";
    constexpr const char* RoughDielectricMediumPbrtExample     = "Rough Dielectric Medium";

    constexpr const char* PlymeshUvPbrtExample                = "PLY Mesh UV";
    constexpr const char* PlymeshGeometryPbrtExample          = "PLY Mesh Geometry";
    constexpr const char* CurveTuftPbrtExample                = "Curve Tuft";
    constexpr const char* CurveHairTuftPbrtExample            = "Curve + Hair Tuft";
    constexpr const char* TrianglemeshUvPbrtExample           = "Triangle Mesh UV";
    constexpr const char* PixelFilterBoxPbrtExample           = "Pixel Filter: Box";
    constexpr const char* ObjectMotionBlurPbrtExample         = "Object Motion Blur";
    constexpr const char* DiskCylinderMotionBlurPbrtExample   = "Disk & Cylinder Motion Blur";
    constexpr const char* ReverseOrientationPbrtExample       = "ReverseOrientation";
    constexpr const char* ConeParaboloidGalleryPbrtExample    = "Cone & Paraboloid Gallery";

    constexpr const char* KillerooSimplePbrtExample           = "Killeroo";

    // Education category - each reuses an existing scene's geometry (see
    // that entry's own comment in scene_registry.h), so the name says what
    // Render Options control it demonstrates rather than what's in frame.
    constexpr const char* SamplerComparison    = "Sampler Comparison (Low Sample Count)";
    constexpr const char* SpectralDispersionEducation = "Spectral Rendering: Prism Dispersion";
    constexpr const char* ExposureToneMapping  = "Exposure & Tone Mapping (HDR Sky)";
    constexpr const char* DenoiserComparison   = "GPU Denoiser: Before & After";
    constexpr const char* SppmCausticsEducation = "SPPM: Rough Glass Caustic";
    constexpr const char* BdptMltEducation      = "BDPT / MLT: Bidirectional Light Transport";
    constexpr const char* LightTransportStrategies = "Light Transport Strategies (NEE / BSDF / MIS)";
    constexpr const char* LightSamplerComparison    = "Light Sampler Strategy (Uniform / Power / BVH)";
    constexpr const char* AmbientOcclusionEducation = "Ambient Occlusion (Debug Integrator)";
    constexpr const char* FireflySuppression        = "Firefly Suppression (Regularize / Clamp)";
} // namespace SceneNames

// -----------------------------------------------------------------------
// Canonical scene category constants
// -----------------------------------------------------------------------
// Categories group scenes by WHAT THEY DEMONSTRATE, which is the question a
// user browsing 78 scenes is actually asking - not by which book chapter or
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
    // Curated demos of the Render Options tab's own controls (Sampler,
    // Spectral rendering, Exposure, Tone mapping, OptiX AI denoiser) and its
    // Integrator selector (SPPM/BDPT/MLT) - each entry
    // reuses an existing scene's geometry (same technique B23/F3 already use
    // to share content with another entry) rather than being new content in
    // its own right, so the description/technique-note is the point: which
    // control to try, and why this particular scene shows it clearly.
    constexpr const char* Education  = "Education";
    // Texture SYSTEM demos - encoding/wrap/invert, procedural texture
    // classes, nested texture references - as distinct from Materials'
    // BxDF/material-kind demos, which a texture is only ever bound INTO.
    // Split out once Materials grew past 25 scenes mixing both concerns
    // (a scene about "does imagemap wrap or clamp" and a scene about "what
    // does coateddiffuse look like" answer genuinely different questions,
    // even though both happen to declare a Material). Inserted here, after
    // every other compiled-in category and before CustomScenes (same
    // zero-disruption insertion point Education itself used - see kAll's
    // own comment below), so no other category's id letter moves.
    constexpr const char* Textures   = "Textures";
    // Scenes loaded from .pbrt files found on disk rather than compiled in.
    // Unlike every category above it, this one is populated at runtime and is
    // legitimately empty when the user has no scene collection installed -
    // which is why the registry tests exempt it from "every category has at
    // least one scene". Named CustomScenes rather than the more generic-
    // sounding "UserScenes" this used to be - see git history for the
    // rename - since what actually distinguishes this category is that its
    // scenes are described by a file on disk rather than compiled in, not
    // that a "user" made them (a git-tracked, bundled pbrt_scenes/*.pbrt
    // example is just as "yours" as anything in the other categories).
    constexpr const char* CustomScenes = "Custom Scenes";
    // The scenes this project's own tests render: closed-form furnaces, light-transport probes, one-feature fixtures. They live in pbrt_scenes/ and
    // are worth browsing (each isolates one thing), but they are not demos, so they stay out of the user-facing tabs above. A scene file puts itself
    // here with a "# @rt-category Test Scenes" line in its header. Last in kAll so that no earlier category's id letter moves.
    constexpr const char* Tests = "Test Scenes";

    // Display order for the GUI's category tabs. Education and Textures sit
    // after the other compiled-in categories and before CustomScenes, which
    // comes next so the built-in tabs never shift position when a scene
    // folder appears; Test Scenes, also filled from disk, is last.
    constexpr const char* kAll[] = {
        Basics, Materials, Lights, Cameras, Volumes, Geometry, Models, LargeScene,
        Education, Textures, CustomScenes, Tests
    };
    constexpr std::size_t kAllCount = sizeof(kAll) / sizeof(kAll[0]);

    // A category's id letter (used to build scene ids like "B10", the 10th
    // Materials scene - see scene_registry.h's SceneDescriptor::id) is just
    // 'A' + its position in kAll: A, B, C, ... in display order, no gaps.
    // This used to be a hand-maintained parallel kAllLetters array - real
    // drift, twice: Education's own insertion once shifted CustomScenes
    // from 'I' to 'J', then Textures's insertion shifted it again from 'J'
    // to 'K', and both times at least one comment elsewhere in the codebase
    // citing the old letter was missed and went stale (caught by code
    // review, not by any compiler or test). Deriving the letter directly
    // from kAll's own index removes the second array entirely, so the next
    // category addition is purely a one-line kAll edit - there is no
    // parallel array left to forget to update, and no letter left to go
    // stale in a comment written against yesterday's kAll.
    static_assert(kAllCount <= 26, "kAll has grown past 'Z' - letter_for_category() needs a two-letter or numeric id scheme");

    // Returns the letter for a category, or '\0' if category doesn't match
    // any entry in kAll (a typo'd category, same failure mode kAll's own
    // scene_registry_tests.cpp check exists to catch). strcmp, not pointer
    // identity - correct regardless of whether the caller's `category` and
    // kAll's entries happen to be the same interned string literal.
    inline char letter_for_category(const char* category) {
        for (std::size_t i = 0; i < kAllCount; ++i) {
            if (std::strcmp(kAll[i], category) == 0)
                return static_cast<char>('A' + static_cast<int>(i));
        }
        return '\0';
    }
} // namespace SceneCategories

#endif // __cplusplus
