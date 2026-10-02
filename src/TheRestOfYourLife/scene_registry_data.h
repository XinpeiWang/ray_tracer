#pragma once
// scene_registry_data.h -- the compiled-in scene data table, split out of
// scene_registry.h to keep that file to the wiring functions
// (build_curated_pbrt_scene_descriptor/wire_pbrt_backed_scene/append/paths)
// around it. This file is a single function, get_builtin_scene_registry(),
// whose body is one large SceneDescriptor vector literal - not control
// flow - so the split is a pure textual move with no logic change.
// #include'd directly from scene_registry.h at the point this content used
// to live; not meant to be included standalone (relies on SceneDescriptor,
// SceneNames::/SceneCategories::, kCornellBoxCamera/kPrismCamera, and
// pbrt_scene_registry::build_curated_pbrt_scene_descriptor(), all declared
// earlier in scene_registry.h).

// -----------------------------------------------------------------------
// Scenes loaded from .pbrt files (see append_pbrt_scenes below)
// -----------------------------------------------------------------------

// The scenes compiled into this binary. Everything the full registry holds
// beyond these came from a .pbrt file found on disk at startup.
inline const std::vector<SceneDescriptor>& get_builtin_scene_registry() {
    static const std::vector<SceneDescriptor> registry = {
        // A1 migrated to pbrt-backed (this project's GPU scene-construction-
        // duplication elimination, pilot batch) - see pbrt_scenes/
        // cornell-box-native.pbrt's own header comment. legacy_id 169 (past
        // every real case in gpu/optix/scene_builder.cpp's switch, same
        // convention A6's own migration used, next after A6's 168) since 0
        // no longer has a switch case of its own to reuse. CameraMode::
        // UserControlled explicitly passed (see build_curated_pbrt_scene_
        // descriptor()'s own `mode` parameter comment) - kCornellBoxCamera
        // was UserControlled, and wire_pbrt_backed_scene()'s own default
        // (Fixed) would otherwise silently disable --cam_x/y/z and the
        // GUI's camera controls for this scene.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "A1", 169, SceneNames::CornellBox, SceneCategories::Basics,
            "Classic Cornell box with glass sphere and aluminum box",
            "Medium", "cornell-box-native.pbrt", CameraMode::UserControlled),
        {
            "A2", 1, SceneNames::BouncingSpheres, SceneCategories::Basics,
            "Random spheres with checker ground (In One Weekend final)",
            "Slow", 100, false, true,
            // defocus_angle/focus_dist: the book's own final-render values
            // for this exact scene - a subtle depth-of-field "beauty shot"
            // focused on the 3 hero spheres near the origin, without
            // redesigning the iconic grid composition itself.
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00, CameraMode::Fixed, 0.6, 10.0 },
            build_bouncing_spheres,
            sky_dummy_lights
        },
        // A3 migrated to pbrt-backed - see pbrt_scenes/checkered-spheres.pbrt.
        // legacy_id 185 (next after D8's 184). First scene to use this
        // loader's newly-added real pbrt-v4 "checkerboard" "integer
        // dimension" [3] support (see that .pbrt file's own header comment)
        // - a genuine fix, not an approximation: A3's own checker_texture is
        // now expressible in pbrt exactly. build_checkered_spheres() has no
        // other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "A3", 185, SceneNames::CheckeredSpheres, SceneCategories::Basics,
            "Two spheres with procedural checker texture",
            "Fast", "checkered-spheres.pbrt"),
        // A4 migrated to pbrt-backed - see pbrt_scenes/earth-globe.pbrt.
        // legacy_id 190 (next after F4's 189). Reuses the already-bundled
        // images/earthmap.jpg - no new asset needed. build_earth()/
        // build_earth_lights() have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "A4", 190, SceneNames::Earth, SceneCategories::Basics,
            "Globe with earth texture mapping (requires earthmap.jpg)",
            "Fast", "earth-globe.pbrt", CameraMode::Fixed,
            /*requires_files=*/true),
        // A5/A7 PERMANENTLY stay native - the native noise_texture class
        // (src/TheRestOfYourLife/texture.h) computes a book-specific
        // grayscale formula, 0.5*(1+sin(scale*p.z + 10*Turbulence(p,
        // octaves=7))) (the classic "Ray Tracing: The Next Week" marble
        // recipe) - pbrt-v4 itself never implements this exact formula.
        // This loader's own real pbrt-v4 noise textures (marble_texture/
        // wrinkled_texture/windy_texture/fbm_texture, texture.h) share the
        // identical underlying pbrt-v4-exact noise basis (perlin_noise<T>,
        // src/shared/noise.h) but are structurally different compositions
        // of it: marble_texture uses p.y (not p.z) and FBm (signed octave
        // sum, not Turbulence's abs-value sum) inside the sine, and always
        // outputs a COLORED 9-knot Bezier-spline-mapped result, never the
        // native class's flat grayscale; wrinkled_texture has the right
        // inner Turbulence function but applies no sin() wrap at all. No
        // parameter choice on any of the four bridges this gap - pbrt-v4
        // has no generic "wrap this texture in a scene-specified sin()"
        // composition primitive, so expressing the native formula from a
        // .pbrt file would need an invented, non-standard texture type,
        // defeating the point of migrating to real pbrt-v4 syntax. Both
        // GPU switch cases/native builders stay.
        {
            "A5", 4, SceneNames::PerlinSpheres, SceneCategories::Basics,
            "Spheres with Perlin noise marble texture",
            "Fast", 100, false, true,
            { 20, 13, 2, 3,  0, 0, 0,  0.70, 0.80, 1.00 },
            build_perlin_spheres,
            build_perlin_spheres_lights
        },
        // A6 migrated to pbrt-backed (this project's GPU scene-construction-
        // duplication elimination, pilot batch) - see pbrt_scenes/
        // colored-quads.pbrt's own header comment. legacy_id 168 (past every
        // real case in gpu/optix/scene_builder.cpp's switch, same 100+
        // convention every other curated pbrt scene below uses) since 5 no
        // longer has a switch case of its own to reuse.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "A6", 168, SceneNames::ColoredQuads, SceneCategories::Basics,
            "Five colored quad primitives",
            "Fast", "colored-quads.pbrt"),
        // A7 PERMANENTLY stays native - same noise_texture gap as A5's own
        // comment just above. Despite the SimpleLight name/id (inherited
        // from the book chapter, not its actual content), A7's own
        // description ("Perlin spheres with emissive light sources") and
        // its build_simple_light() builder confirm it uses the identical
        // noise_texture(4) formula on its ground/center spheres.
        {
            "A7", 6, SceneNames::SimpleLight, SceneCategories::Basics,
            "Perlin spheres with emissive light sources",
            "Fast", 100, false, true,
            { 20, 26, 3, 6,  0, 2, 0,  0, 0, 0 },
            build_simple_light,
            no_lights
        },
        // A8 migrated to pbrt-backed - see pbrt_scenes/cornell-smoke.pbrt.
        // legacy_id 191. build_cornell_smoke()/build_cornell_smoke_lights()
        // have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "A8", 191, SceneNames::CornellSmoke, SceneCategories::Basics,
            "Cornell box with volumetric fog",
            "Slow", "cornell-smoke.pbrt", CameraMode::UserControlled),
        {
            "A9", 8, SceneNames::FinalScene, SceneCategories::Basics,
            "Complex scene from The Next Week",
            "Very Slow", 500, false, true,
            // Subtle deep ambient instead of pure black - the box-grid
            // ground and negative space used to render into a stark void.
            { 40, 478, 278, -600,  278, 278, 0,  0.03, 0.025, 0.02 },
            build_final_scene,
            build_final_scene_lights
        },
        // B1/B2 PERMANENTLY stay native - rough_metal has no pbrt-authorable
        // equivalent, confirmed at the BxDF-formula level (src/shared/
        // bxdfs_conductor.h): rough_metal::sample_local() weights purely by
        // the GGX G/G1 shadow-masking ratio times a flat, direction-
        // independent RGB albedo (no Fresnel model at all), while pbrt-v4's
        // real Material "conductor" (ConductorBxDF) weights by FrComplex(wi,
        // wm, eta, k) - a genuine complex-IOR Fresnel that is NEVER angle-
        // independent for any physically valid (k>0) conductor. No (eta, k)
        // choice reproduces a constant-angle albedo, so this is a real,
        // structural mismatch, not a missing parameter - same conclusion
        // this project already reached and skipped B2 for specifically, in
        // commits aa57e0d2 and 19d8bd01 ("rough_metal has no pbrt-
        // authorable equivalent - confirmed by checking both pbrt builders
        // never construct it"). B1 was re-investigated independently (not
        // named in those two commits) and uses the identical rough_metal
        // class, so the identical verdict applies. Both GPU switch cases/
        // native builders stay - deleting them would be a real feature
        // loss (GGX-microfacet-with-flat-tint has no equivalent GPU
        // material slot to fall back to either).
        {
            "B1", 9, SceneNames::RoughMetalSpheres, SceneCategories::Materials,
            "Five GGX spheres roughness 0.05 to 0.8 -- showcases microfacet BRDF",
            "Medium", 200, false, true,
            { 42, 0, 2.7, 17,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_rough_metal_spheres,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-6,6,-4), vec3(12,0,0), vec3(0,0,8), empty_mat));
                return l;
            }
        },
        // B2 PERMANENTLY stays native - same rough_metal gap as B1's own
        // comment just above (aluminum box, gold sphere, both rough_metal).
        {
            "B2", 10, SceneNames::CornellRoughMetal, SceneCategories::Materials,
            "Cornell box with rough aluminum box and rough gold sphere",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0.05, 0.055, 0.07, CameraMode::UserControlled },
            build_cornell_rough_metal,
            build_cornell_box_lights
        },
        // B3 migrated to pbrt-backed (this project's GPU scene-construction-
        // duplication elimination, pilot batch) - see pbrt_scenes/
        // cornell-rough-glass.pbrt's own header comment (also fixes I5/I10,
        // which reuse "the same world as B3" - see their own entries below).
        // legacy_id 171 (next after B4's 170) since 11 no longer has a
        // switch case of its own to reuse (I5/I10's own cases still call
        // build_cornell_rough_glass(scene) directly and are unaffected).
        // CameraMode::UserControlled explicitly passed, same reason as
        // A1's/B4's own migrations.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B3", 171, SceneNames::CornellRoughGlass, SceneCategories::Materials,
            "Cornell box with a GGX rough-dielectric sphere (pbrt-v4 RoughDielectricBxDF)",
            "Medium", "cornell-rough-glass.pbrt", CameraMode::UserControlled),
        // B4 migrated to pbrt-backed (this project's GPU scene-construction-
        // duplication elimination, pilot batch) - see pbrt_scenes/
        // cornell-conductor.pbrt's own header comment. legacy_id 170 (next
        // after A1's 169) since 12 no longer has a switch case of its own to
        // reuse. CameraMode::UserControlled explicitly passed, same reason
        // as A1's own migration.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B4", 170, SceneNames::CornellConductor, SceneCategories::Materials,
            "Cornell box with polished gold sphere and aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)",
            "Medium", "cornell-conductor.pbrt", CameraMode::UserControlled),
        // B5 migrated to pbrt-backed - see pbrt_scenes/cornell-coated-diffuse.pbrt.
        // legacy_id 172 (next after B3's 171). build_cornell_coated_diffuse()
        // itself is NOT deleted - tests/integration/skip_pdf_material_brightness_tests.cpp
        // calls it directly.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B5", 172, SceneNames::CornellCoatedDiffuse, SceneCategories::Materials,
            "Cornell box with blue coated-diffuse sphere and red coated-diffuse box (pbrt-v4 CoatedDiffuseBxDF)",
            "Medium", "cornell-coated-diffuse.pbrt", CameraMode::UserControlled),
        // B6 migrated to pbrt-backed - see pbrt_scenes/cornell-thin-glass.pbrt.
        // legacy_id 173. build_cornell_thin_glass()/build_cornell_thin_glass_lights()
        // have no other consumers - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B6", 173, SceneNames::CornellThinGlass, SceneCategories::Materials,
            "Cornell box with a vertical thin-glass panel, analytic multi-bounce Fresnel (pbrt-v4 ThinDielectricBxDF)",
            "Medium", "cornell-thin-glass.pbrt", CameraMode::UserControlled),
        // B7 migrated to pbrt-backed - see pbrt_scenes/cornell-coated-conductor.pbrt.
        // legacy_id 174. build_cornell_coated_conductor() has no other
        // consumers - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B7", 174, SceneNames::CornellCoatedConductor, SceneCategories::Materials,
            "Cornell box with lacquered-gold sphere and lacquered-copper box (pbrt-v4 CoatedConductorBxDF)",
            "Medium", "cornell-coated-conductor.pbrt", CameraMode::UserControlled),
        // B8 migrated to pbrt-backed - see pbrt_scenes/cornell-wax-slab.pbrt.
        // legacy_id 175. build_cornell_wax_slab() has no other consumers -
        // deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B8", 175, SceneNames::CornellWaxSlab, SceneCategories::Materials,
            "Cornell box with a wax sphere that diffusely reflects and transmits light (pbrt-v4 DiffuseTransmissionBxDF)",
            "Medium", "cornell-wax-slab.pbrt", CameraMode::UserControlled),
        {
            "B9", 17, SceneNames::CornellCrystal, SceneCategories::Materials,
            "Cornell box with a crystal sphere using Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_cornell_crystal,
            build_cornell_box_lights
        },
        {
            "B10", 18, SceneNames::PrincipledShowcase, SceneCategories::Materials,
            "Row of spheres from matte plastic to metallic with clearcoat (pbrt-v4 PrincipledBxDF)",
            "Medium", 200, false, true,
            { 45, 0, 2.7, 17,  0, 1, 0,  0.10, 0.10, 0.12 },
            build_principled_showcase,
            []() {
                hittable_list l;
                auto empty_mat = std::shared_ptr<material>();
                l.add(std::make_shared<quad>(point3(-7,7,-5), vec3(14,0,0), vec3(0,0,10), empty_mat));
                return l;
            }
        },
        // B11 migrated to pbrt-backed - see pbrt_scenes/hair-fibers-scene.pbrt.
        // legacy_id 192. Uses real Material "hair" on Shape "sphere" - the
        // same normal-as-tangent proxy native's own hair_material class
        // already documents using for this exact scene (see that .pbrt
        // file's own header comment) - not an approximation.
        // build_hair_fibers() has no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B11", 192, SceneNames::HairFibers, SceneCategories::Materials,
            "Sphere cluster with hair/fur fiber scattering (pbrt-v4 HairBxDF)",
            "Medium", "hair-fibers-scene.pbrt"),
        {
            "B12", 20, SceneNames::NormalMappedCornell, SceneCategories::Materials,
            "Cornell box with procedural bump-mapped back wall and normal-mapped sphere (pbrt-v4 NormalMap/BumpMap)",
            "Medium", 200, false, true,
            { 40, 278, 278, -800,  278, 278, 278,  0, 0, 0, CameraMode::UserControlled },
            build_normal_mapped_cornell,
            build_cornell_box_lights
        },
        // B13 migrated to pbrt-backed - see pbrt_scenes/subsurface-slab.pbrt.
        // legacy_id 193. build_subsurface_slab() has no other consumer -
        // deleted below (build_cornell_box_lights() stays - many other
        // scenes still use it).
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B13", 193, SceneNames::SubsurfaceSlab, SceneCategories::Materials,
            "Cornell box with translucent wax slab and jade sphere using subsurface-like scattering",
            "Slow", "subsurface-slab.pbrt", CameraMode::UserControlled),
        // D1 migrated to pbrt-backed - see pbrt_scenes/depth-of-field-spheres.pbrt
        // (legacy_id 22 kept). build_depth_of_field() (CPU) has no other
        // consumer - deleted below. build_depth_of_field_gpu() (GPU) is ALSO
        // deleted below, its own case 22 having diverged into a different,
        // simpler scene than CPU's - the new pbrt file reunifies both
        // backends on CPU's original, richer design (see that file's own
        // header comment).
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D1", 22, SceneNames::DepthOfField, SceneCategories::Cameras,
            "Row of spheres with defocus blur showing depth-of-field from the thin-lens camera model",
            "Medium", "depth-of-field-spheres.pbrt"),
        // F1 migrated to pbrt-backed - see pbrt_scenes/bilinear-patch-scene.pbrt.
        // legacy_id 200. build_bilinear_patch_scene()/
        // build_bilinear_patch_lights() have no other consumer - deleted
        // below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F1", 200, SceneNames::BilinearPatchScene, SceneCategories::Geometry,
            "Cornell box with curved bilinear patch saddle surface (pbrt-v4 BilinearPatch shape)",
            "Medium", "bilinear-patch-scene.pbrt", CameraMode::UserControlled),
        // ---- pbrt-v4 light / camera / medium showcase ----
        // C1 migrated to pbrt-backed - see pbrt_scenes/hdri-sky-gradient.pbrt
        // for the full derivation (native's own gradient baked losslessly
        // to a real, git-tracked .exr via write_exr_image(), then loaded
        // back through a real LightSource "infinite" "string filename" -
        // a genuine GPU fidelity improvement, since native GPU never
        // reproduced the gradient at all, only a flat average-tone
        // approximation). legacy_id 203 (next after E12's 202).
        // build_hdri_sky_world()/build_hdri_sky() have no other consumer -
        // deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C1", 203, SceneNames::HdriSky, SceneCategories::Lights,
            "Open scene lit by a real pbrt-v4 image infinite light (the same procedural gradient native always used, now a real baked .exr on both backends)",
            "Medium", "hdri-sky-gradient.pbrt"),
        // C2 migrated to pbrt-backed - see pbrt_scenes/cornell-spotlight.pbrt.
        // legacy_id 176 (next after B8's 175). build_spotlight_cornell()/
        // build_spotlight_punct() have no other consumers - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C2", 176, SceneNames::SpotlightCornell, SceneCategories::Lights,
            "Cornell box lit by a spotlight with smooth penumbra (pbrt-v4 SpotLight)",
            "Medium", "cornell-spotlight.pbrt", CameraMode::UserControlled),
        // C3 migrated to pbrt-backed - see pbrt_scenes/cornell-distant-light.pbrt.
        // legacy_id 177. build_distant_light_cornell()/build_distant_light_punct()
        // have no other consumers - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C3", 177, SceneNames::DistantLightCornell, SceneCategories::Lights,
            "Cornell box lit by a parallel sun-like distant light (pbrt-v4 DistantLight)",
            "Medium", "cornell-distant-light.pbrt", CameraMode::UserControlled),
        // C4 migrated to pbrt-backed - see pbrt_scenes/cornell-point-light.pbrt.
        // legacy_id 178. build_point_light_cornell()/build_point_light_punct()
        // (CPU) are NOT deleted - tests/integration/sppm_first_slice_test.cpp
        // calls both directly.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C4", 178, SceneNames::PointLightCornell, SceneCategories::Lights,
            "Cornell box lit by a single overhead point light with 1/r^2 falloff (pbrt-v4 PointLight)",
            "Medium", "cornell-point-light.pbrt", CameraMode::UserControlled),
        // C5 migrated to pbrt-backed - see pbrt_scenes/cornell-goniometric.pbrt.
        // legacy_id 179 (next after C4's 178). build_goniometric_light_scene()/
        // build_goniometric_punct() have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C5", 179, SceneNames::GoniometricLight, SceneCategories::Lights,
            "Cornell box lit by a goniometric (IES-profile) point light (pbrt-v4 GoniometricLight)",
            "Medium", "cornell-goniometric.pbrt", CameraMode::UserControlled),
        // C6 migrated to pbrt-backed - see pbrt_scenes/cornell-projection.pbrt.
        // legacy_id 180. build_projection_light_scene()/build_projection_punct()
        // have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C6", 180, SceneNames::ProjectionLight, SceneCategories::Lights,
            "Cornell box with a slide-projector beam casting a checkerboard pattern (pbrt-v4 ProjectionLight)",
            "Medium", "cornell-projection.pbrt", CameraMode::UserControlled),
        // E1 migrated to pbrt-backed - see pbrt_scenes/homogeneous-medium.pbrt.
        // legacy_id 197. build_homogeneous_medium_scene()/
        // build_homogeneous_medium_lights() have no other consumer -
        // deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E1", 197, SceneNames::HomogeneousMedium, SceneCategories::Volumes,
            "Cornell box filled with a homogeneous scattering fog (pbrt-v4 HomogeneousMedium / HenyeyGreenstein)",
            "Slow", "homogeneous-medium.pbrt", CameraMode::UserControlled),
        // E2 migrated to pbrt-backed - see pbrt_scenes/cloud-medium-scene.pbrt.
        // legacy_id 198. build_cloud_medium_scene() has no other consumer -
        // deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E2", 198, SceneNames::CloudMedium, SceneCategories::Volumes,
            "Open scene with a procedural Perlin-noise cloud volume (pbrt-v4 CloudMedium)",
            "Slow", "cloud-medium-scene.pbrt"),
        // E3 migrated to pbrt-backed - see
        // pbrt_scenes/dielectric-medium-showcase.pbrt. legacy_id 199.
        // build_dielectric_medium_scene() has no other consumer - deleted
        // below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E3", 199, SceneNames::DielectricMediumShowcase, SceneCategories::Volumes,
            "Three glass spheres containing colored internal fog at varying density - dielectric surface + participating medium combined (pbrt-v4 style)",
            "Medium", "dielectric-medium-showcase.pbrt"),
        {
            "E4", 70, SceneNames::RgbGridMedium, SceneCategories::Volumes,
            "Heterogeneous nebula with an independent per-voxel R/G/B scattering grid (pbrt-v4 RGBGridMedium)",
            "Slow", 300, false, true,
            // Pulled back further than E2's cloud camera - this box is
            // taller (world y:[1,5] vs E2's [1,4]) and the context spheres
            // sit further out (x:+-6) - see build_rgb_grid_medium_scene.
            { 45, 0, 5, 30,  0, 3, 0,  0.5, 0.7, 1.0 },
            build_rgb_grid_medium_scene,
            sky_dummy_lights
        },
        // D2 migrated to pbrt-backed - see pbrt_scenes/ortho-camera-scene.pbrt.
        // legacy_id 186 (next after A3's 185). build_ortho_camera_scene()/
        // build_ortho_sky() have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D2", 186, SceneNames::OrthographicCamera, SceneCategories::Cameras,
            "Geometric showcase rendered with an orthographic (parallel-projection) camera (pbrt-v4 OrthographicCamera)",
            "Fast", "ortho-camera-scene.pbrt"),
        // D3 migrated to pbrt-backed - see pbrt_scenes/spherical-camera-scene.pbrt.
        // legacy_id 195. Camera type already proven pbrt-representable via
        // D7's own earlier migration. build_spherical_camera_scene()/
        // build_spherical_sky() have no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D3", 195, SceneNames::SphericalCamera, SceneCategories::Cameras,
            "360-degree equirectangular panorama from a spherical camera (pbrt-v4 SphericalCamera)",
            "Medium", "spherical-camera-scene.pbrt"),
        // B14 migrated to pbrt-backed - see pbrt_scenes/measured-brdf-
        // showroom.pbrt for the full derivation. A real, disclosed fidelity
        // IMPROVEMENT, not a lateral move: native's own measured_material
        // (scenes_advanced.h) never read its own MeasuredBRDFData at all -
        // scatter() just returned a flat tint, a mislabeled Lambertian on
        // both backends. This loader's real Measured BRDF support (already
        // proven on 3 downloaded pbrt-v4-scenes bundles) is now reachable
        // from a small, self-contained, git-tracked demo too, via a new
        // synthetic pbrt_scenes/synthetic-gold.bsdf baked specifically for
        // this scene (an original, licence-free glossy lobe, not a real
        // gonioreflectometer measurement - see the .pbrt file's own header
        // comment). legacy_id 205 (next after C1's 203/I3's 204).
        // build_measured_brdf_scene()/measured_material have no other
        // consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B14", 205, SceneNames::MeasuredBrdf, SceneCategories::Materials,
            "Sphere cluster with a real, importance-sampled measured BRDF (pbrt-v4 MeasuredBxDF) loaded from a synthetic .bsdf tensor file",
            "Medium", "measured-brdf-showroom.pbrt"),
        {
            "B23", 131, SceneNames::GlassPrismDispersion, SceneCategories::Materials,
            "A real glass prism splitting a parallel white light into a visible chromatic fan (CPU --spectral, GPU --wavefront: real continuous spectral integration; GPU-recursive (--gpu, no --wavefront): a simplified 3-representative-wavelength RGB-channel approximation, same qualitative fan, see shade_material()'s inout_rgb_channel comment, optix_device_helpers.h - see dielectric's dispersive constructor, material_simple.h)",
            "Medium", 200, false, true,
            kPrismCamera,
            build_prism_dispersion,
            no_lights,
            nullptr,
            build_prism_dispersion_punct
        },
        {
            // Same prism/light/screen as B23, frosted glass instead of
            // smooth - see build_prism_dispersion_rough()'s own comment
            // (scenes_materials.h) for why this exercises rough_dielectric's
            // real NEE/MIS path, not just its initial scatter.
            "B24", 136, SceneNames::FrostedPrismDispersion, SceneCategories::Materials,
            "The same glass prism as B23, frosted (rough_dielectric) instead of smooth - same chromatic fan, blurred by the roughness (CPU --spectral, GPU --wavefront with real continuous-wavelength dispersion, and GPU-recursive with the same 3-representative-wavelength approximation as B23)",
            "Medium", 200, false, true,
            kPrismCamera,
            build_prism_dispersion_rough,
            no_lights,
            nullptr,
            build_prism_dispersion_punct
        },
        // C7 migrated to pbrt-backed - see pbrt_scenes/portal-window-room.pbrt.
        // legacy_id 194. build_portal_light_scene()/build_portal_sky() have
        // no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C7", 194, SceneNames::PortalInfiniteLight, SceneCategories::Lights,
            "Room scene with a sky visible through a windowed wall aperture (a flat sky_light behind a geometric hole, NOT the real pbrt-v4 PortalImageInfiniteLight class - see pbrt_scenes/portal-light.pbrt for that)",
            "Slow", "portal-window-room.pbrt", CameraMode::UserControlled),
        // D4 migrated to pbrt-backed - see pbrt_scenes/realistic-camera-scene.pbrt.
        // legacy_id 187. build_realistic_camera_scene() has no other
        // consumer - deleted below. The two real bugs found and fixed
        // during this scene's original authoring (CPU get_ray() dropping
        // RealisticCamera::generate_ray()'s weight; the 5 spheres sitting
        // on the camera's own viewing axis needing an oblique lookfrom to
        // all be visible) live in shared code/this scene's own camera
        // params respectively, both already reflected in the .pbrt file's
        // own LookAt and this loader's existing weight-applying code path -
        // nothing left to carry forward here.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D4", 187, SceneNames::RealisticCamera, SceneCategories::Cameras,
            "Spheres rendered through a thin-lens with realistic lens-element bokeh (pbrt-v4 RealisticCamera)",
            "Medium", "realistic-camera-scene.pbrt"),
        // D5-D8: the exact same classic Cornell box as A1 (build_cornell_box /
        // build_cornell_box_lights, backed by src/shared/cornell_box_data.h),
        // rendered by each of D1-D4's camera models in turn. Keeping the scene
        // fixed and only varying the camera makes the actual differences
        // between the four models (defocus blur, parallel projection, 360
        // panorama, real lens bokeh) directly comparable, which D1-D4's own
        // bespoke per-scene geometry doesn't support. D1-D4 are left
        // unchanged - these are additive, not replacements.
        // D5-D8 migrated to pbrt-backed - see pbrt_scenes/cornell-dof.pbrt/
        // cornell-orthographic.pbrt/cornell-spherical.pbrt/cornell-realistic.pbrt.
        // legacy_ids 181-184 (next after C6's 180) - the old 65-68 are no
        // longer assigned to any scene. All 4 reuse cornell-box-native.pbrt's
        // (A1) exact same world, only the Camera directive differs, matching
        // native's own "same scene, only the camera model changes" design.
        // build_cornell_box()/build_cornell_box_lights() are NOT deleted -
        // A1 (already pbrt-backed, doesn't call them either) aside, D13 and
        // several B23/B24 "same world as X" aliases below still call them
        // directly.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D5", 181, SceneNames::DepthOfFieldCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D6-D8) with defocus blur from the thin-lens perspective camera",
            "Medium", "cornell-dof.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D6", 182, SceneNames::OrthographicCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5/D7/D8) rendered with a parallel-projection orthographic camera",
            "Medium", "cornell-orthographic.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D7", 183, SceneNames::SphericalCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5/D6/D8), toured from its center as a 360-degree equirectangular panorama",
            "Medium", "cornell-spherical.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D8", 184, SceneNames::RealisticCameraCornellBox, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5-D7), rendered through a multi-element lens for realistic bokeh (pbrt-v4 RealisticCamera)",
            "Medium", "cornell-realistic.pbrt"),
        // Same Cornell box world as A1/D5-D8 - only the camera differs
        // (keyframed across the exposure instead of static). closes the
        // "no motion blur anywhere" gap from docs/FEATURE_INVENTORY.md -
        // CPU default path tracer (+SPPM), see camera.h's own
        // camera_is_animated comment, AND both GPU backends (see
        // GpuCameraParams::animated, gpu/optix/optix_types.h).
        // D13 migrated to pbrt-backed - see
        // pbrt_scenes/cornell-camera-motion-blur.pbrt. legacy_id 196. Reuses
        // A1's exact world (cornell-box-native.pbrt's own geometry,
        // transcribed again here since this scene needs its own Camera
        // block) with a real ActiveTransform "StartTime"/"EndTime" animated
        // camera - ordinary already-supported pbrt-v4 camera motion blur, no
        // new loader capability needed. build_cornell_box()/
        // build_cornell_box_lights() are NOT deleted - many other scenes
        // still call them directly.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D13", 196, SceneNames::CameraMotionBlur, SceneCategories::Cameras,
            "The classic Cornell box (same scene as A1/D5-D8), camera trucking sideways (lookat stays fixed, so this is really a small combined translate+rotate) across the exposure for real AnimatedTransform-based motion blur - CPU and GPU (both recursive and wavefront) all interpolate the same two keyframes",
            "Medium", "cornell-camera-motion-blur.pbrt"),
        // F2 migrated to pbrt-backed - see pbrt_scenes/triangle-mesh-scene.pbrt.
        // legacy_id 188 (next after D4's 187). The icosahedron's 12 vertices/
        // 20 faces are fully deterministic (golden-ratio formula, no RNG),
        // transcribed verbatim into the .pbrt file's own Shape "trianglemesh".
        // build_triangle_mesh_scene() has no other consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F2", 188, SceneNames::TriangleMesh, SceneCategories::Geometry,
            "Procedurally-generated icosahedron showcasing real triangle-mesh geometry (watertight Moller-Trumbore intersection)",
            "Fast", "triangle-mesh-scene.pbrt"),
        build_instanced_spheres_descriptor(),
        // F4 migrated to pbrt-backed - see pbrt_scenes/curve-fibers-scene.pbrt.
        // legacy_id 189 (next after F2's 188). All 70 strands' control points
        // are fully deterministic (hash01()-driven, no RNG) and were computed
        // once from native's exact formula, transcribed verbatim - see that
        // .pbrt file's own header comment, including how it relates to the
        // already-bundled pbrt_scenes/curve-tuft.pbrt example (same formula,
        // fewer strands, no palette). build_curve_fibers_scene() has no other
        // consumer - deleted below.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F4", 189, SceneNames::CurveFibers, SceneCategories::Geometry,
            "A windswept tuft of real Bezier curve strands (CurveShape, tapered Cylinder cross-section) - genuine ray-curve intersection on CPU, not the sphere+HairBxDF trick scene B11 uses. GPU renders the same 70 strands tessellated into tapered tubes of bilinear patches (matches pbrt-v4's own GPU curve strategy) rather than an exact curve intersection, so the tube surface reads slightly faceted up close.",
            "Fast", "curve-fibers-scene.pbrt"),
        {
            "G1", 38, SceneNames::StanfordBunny, SceneCategories::Models,
            "Classic Stanford bunny scan (69,451 triangles) in polished bronze, loaded from an external .obj file (requires models/stanford-bunny.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_bunny,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G2", 39, SceneNames::StanfordArmadillo, SceneCategories::Models,
            "Stanford armadillo scan (99,976 triangles) in gunmetal, loaded from an external .obj file (requires models/armadillo.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_armadillo,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G3", 40, SceneNames::StanfordHappyBuddha, SceneCategories::Models,
            "Stanford happy buddha scan (98,601 triangles) in polished gold, loaded from an external .obj file (requires models/happy-buddha.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_happy_buddha,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G4", 41, SceneNames::StanfordLucy, SceneCategories::Models,
            "Stanford Lucy angel figure (99,970 triangles) in bright silver, loaded from an external .obj file (requires models/lucy.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_lucy,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G5", 42, SceneNames::StanfordDragon, SceneCategories::Models,
            "Stanford XYZRGB Dragon (249,882 triangles) in bright silver, loaded from an external .obj file (requires models/xyzrgb_dragon.obj). Camera pulled back/up further than the other mesh scenes' default (0,3,7): the dragon's lunging pose is much wider than tall (~5.4 units wide vs ~3 tall after normalization, similar to scene 43's teapot), and the default statue framing cropped the head and tail.",
            "Very Slow", 150, true, true,
            { 35, 0, 4, 12,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_stanford_dragon,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G6", 43, SceneNames::UtahTeapot, SceneCategories::Models,
            "The classic Utah Teapot (6,320 triangles) in bright silver, loaded from an external .obj file (requires models/teapot.obj)",
            "Medium", 150, true, true,
            // Camera pulled back further than the other mesh scenes (0,3,7)
            // because the teapot's spout+handle make it much wider than it
            // is tall (~9 units wide vs ~3 tall after normalization) -
            // the statue framing crops the spout/handle at this aspect.
            { 35, 0, 6, 20,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_utah_teapot,
            []() {
                // Matches build_utah_teapot()'s light sphere, raised to
                // y=20 - see that function's comment for why (this scene's
                // raised/pulled-back camera brought the standard y=8 light
                // into frame as a blown-out disc).
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,20,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G7", 44, SceneNames::SpotCow, SceneCategories::Models,
            "Keenan Crane's Spot the Cow (5,856 triangles) in bright silver, loaded from an external .obj file (requires models/spot.obj)",
            "Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_spot_cow,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G8", 45, SceneNames::Suzanne, SceneCategories::Models,
            "Blender's Suzanne monkey-head mascot (968 triangles after fan-triangulating its mostly-quad faces) in bright silver, loaded from an external .obj file (requires models/suzanne.obj). Unlike every other mesh scene, Suzanne is a disembodied head with no neck/shoulders/pedestal, so grounding its chin at y=0 (the shared statue convention) puts its face well above the generic eye-level camera - the camera below is raised and pulled in closer to look at roughly the model's own eye height instead.",
            "Fast", 150, true, true,
            { 35, 0, 2.1, 6.5,  0, 1.9, 0,  0.05, 0.05, 0.08 },
            build_suzanne,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G9", 46, SceneNames::NefertitiBust, SceneCategories::Models,
            "Scanned bust of Nefertiti (99,938 triangles) in bright silver, loaded from an external .obj file (requires models/nefertiti.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_nefertiti,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G10", 47, SceneNames::Horse, SceneCategories::Models,
            "Classic geometry-processing test horse head/neck bust (96,966 triangles) in bright silver, loaded from an external .obj file (requires models/horse.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_horse,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G11", 48, SceneNames::Cheburashka, SceneCategories::Models,
            "Beloved cartoon-character bust from Keenan Crane's geometry-processing course (13,334 triangles) in bright silver, loaded from an external .obj file (requires models/cheburashka.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_cheburashka,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G12", 49, SceneNames::TrophyRoom, SceneCategories::Models,
            "Four already-loaded meshes (bunny, teapot, Suzanne, Spot the Cow) lined up in bronze/chrome/gold/gunmetal, the first scene to combine multiple external .obj meshes in one composition (requires models/stanford-bunny.obj, teapot.obj, suzanne.obj, spot.obj)",
            "Very Slow", 200, true, true,
            { 34, 0, 2.3, 14,  0, 0.9, 0,  0.05, 0.05, 0.08 },
            build_trophy_room,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G13", 50, SceneNames::GlassDragon, SceneCategories::Models,
            "Stanford XYZRGB Dragon (249,882 triangles) in clear glass (dielectric, IOR 1.5), loaded from an external .obj file (requires models/xyzrgb_dragon.obj). The dragon's own surface renders persistently noisy at any sample count under EITHER the regular path tracer OR --sppm -- refraction through this deeply concave mesh is a hard case for any unidirectional camera-side estimator (SPPM's photon-density gather only ever helps non-delta/diffuse surfaces, and the dragon is 100% delta-BSDF glass), not a bug. --sppm's real benefit here is a genuine floor caustic from the dragon (CPU only -- GPU SPPM currently supports scene 11 only) that the regular path tracer's NEE can't resolve; a fully clean render of the glass surface itself would need bidirectional path tracing or MLT.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_glass_dragon,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G14", 51, SceneNames::Beast, SceneCategories::Models,
            "Fantasy creature bust (common-3d-test-models) in bronze, loaded from an external .obj file (requires models/beast.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_beast,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G15", 52, SceneNames::VWBeetle, SceneCategories::Models,
            "Classic CAD-style Volkswagen Beetle in bright chrome, loaded from an external .obj file (requires models/beetle.obj). Elongated along Z after normalization, so the camera is pulled back further than the other mesh scenes, same reasoning as scene 43's Utah Teapot.",
            "Medium", 150, true, true,
            { 35, 0, 3, 16,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_beetle,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G17", 54, SceneNames::Bimba, SceneCategories::Models,
            "Smooth abstract bust/statue (AIM@SHAPE repository test model) in gold, loaded from an external .obj file (requires models/bimba.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_bimba,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G18", 55, SceneNames::Cow, SceneCategories::Models,
            "Classic Viewpoint/Alias Cow test model (distinct from scene 44's Spot the Cow) in brass, loaded from an external .obj file (requires models/cow.obj)",
            "Medium", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_cow,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G19", 56, SceneNames::Fandisk, SceneCategories::Models,
            "Classic CAD mechanical-engineering test model with sharp creases, in gunmetal, loaded from an external .obj file (requires models/fandisk.obj). Camera moved to a three-quarter elevated angle rather than the usual eye-level statue framing - this mesh's proportions are shallow along the default view axis, and a face-on shot showed only a smooth, featureless wedge with none of the sharp creases the model is known for.",
            "Medium", 150, true, true,
            { 35, 4, 9, 4,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_fandisk,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G20", 57, SceneNames::Homer, SceneCategories::Models,
            "Homer Simpson bust in gold, loaded from an external .obj file (requires models/homer.obj)",
            "Medium", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_homer,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G21", 58, SceneNames::Igea, SceneCategories::Models,
            "Classical Italian bust (Igea, Roman goddess of health) in bright silver, loaded from an external .obj file (requires models/igea.obj). An earlier camera here (raised and looking steeply down) was meant to compensate for this scan's upward-tilted face, but actually framed the shiny crown of the skull instead of the face - lowered/pulled back closer to the other mesh scenes' eye-level convention, which shows the face (eyes, nose, tilted-up chin) correctly.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 5,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_igea,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G22", 59, SceneNames::MaxPlanck, SceneCategories::Models,
            "Scanned bust of physicist Max Planck in aged bronze, loaded from an external .obj file (requires models/max-planck.obj). This scan's face points toward -Z, so the camera sits on that side (unlike the other mesh scenes' +Z default) to actually see the face instead of the back of the head.",
            "Very Slow", 150, true, true,
            { 35, 0, 3, -7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_max_planck,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G23", 60, SceneNames::Ogre, SceneCategories::Models,
            "Fantasy ogre head in dark olive metal, loaded from an external .obj file (requires models/ogre.obj)",
            "Very Slow", 150, true, true,
            { 35, 0, 3, 7,  0, 1.5, 0,  0.05, 0.05, 0.08 },
            build_ogre,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "G24", 61, SceneNames::RockerArm, SceneCategories::Models,
            "Mechanical engine-part test model in gunmetal, loaded from an external .obj file (requires models/rocker-arm.obj). Elongated along Z after normalization like the Beetle scene (G15), but much smaller overall and taller than that comparison suggested - the camera is pulled back/up further than originally set, which cropped the two boss/lobe cylinders at the top of the part. Now visible, those bosses' flat tops catch a strong mirror-like specular highlight from the overhead light - a legitimate result of a flat, low-roughness surface facing a point-ish light, confirmed by testing (repositioning/brightening the light didn't change it), not a bug.",
            "Slow", 150, true, true,
            { 35, 0, 4, 12,  0, 1.2, 0,  0.05, 0.05, 0.08 },
            build_rocker_arm,
            []() {
                hittable_list l;
                l.add(std::make_shared<sphere>(point3(0,8,0), 2,
                      std::shared_ptr<material>()));
                return l;
            }
        },
        {
            "H1", 62, SceneNames::CrytekSponza, SceneCategories::LargeScene,
            "Crytek Sponza (262K triangles) - the classic architectural global-illumination benchmark scene, with real per-face .mtl materials and image textures (curtains, columns, floor) loaded from models/sponza_textures/, lit by an open sky, loaded from an external .obj file (requires models/sponza.obj). First 'whole environment' mesh scene here rather than a single statue -- see build_sponza()'s own comment for the full design rationale.",
            "Very Slow", 150, true, true,
            { 70, -800, 300, 0,  800, 300, 0,  0, 0, 0 },
            build_sponza,
            build_sponza_lights,
            build_sponza_sky,
            nullptr
        },
        {
            "H2", 63, SceneNames::AmazonBistro, SceneCategories::LargeScene,
            "Amazon Lumberyard Bistro, Exterior (2.84M triangles) - a full outdoor street block (multiple buildings + plaza), with real per-face .mtl materials and image textures (windows, doors, foliage) loaded from models/bistro_textures/, lit by an open sky, loaded from an external .obj file (requires models/bistro_exterior.obj). Second 'whole environment' mesh scene, same design rationale as scene 62 (Crytek Sponza) -- see build_bistro_exterior()'s own comment. Camera nudged 300 units in Z from the original verified-clear-sightline position: a decorative streetlamp post sat directly in the foreground as a fully-black silhouette blocking most of the frame; the shift turns it into a pleasant framing element instead (visible tree/building behind it) rather than eliminating it.",
            "Very Slow", 150, true, true,
            { 60, 1500, 700, 1700,  4000, 700, 2000,  0, 0, 0 },
            build_bistro_exterior,
            build_bistro_exterior_lights,
            build_bistro_exterior_sky,
            nullptr
        },
        {
            "H3", 64, SceneNames::Rungholt, SceneCategories::LargeScene,
            "Rungholt (6.7M triangles) - a giant blocky Minecraft-style town, with real per-face .mtl material colors (no image textures for this one, unlike scenes 62/63's Sponza/Bistro), loaded from an external .obj file (requires models/rungholt.obj). Third 'whole environment' mesh scene, same design rationale as scenes 62-63 -- see build_rungholt()'s own comment (including a real OBJ-loader bug this mesh exposed and fixed: negative/relative face indices).",
            "Very Slow", 150, true, true,
            { 45, 400, 300, 400,  0, 40, 0,  0, 0, 0 },
            build_rungholt,
            build_rungholt_lights,
            build_rungholt_sky,
            nullptr
        },
        {
            "H4", 73, SceneNames::FireplaceRoom, SceneCategories::LargeScene,
            "Fireplace Room - a small, human-scale furnished living room (fireplace, wood floor, framed pictures, a potted plant), with real per-face .mtl materials and image textures loaded from models/fireplace_room_textures/, lit by an open sky through its windows, loaded from an external .obj file (requires models/fireplace_room.obj). Fourth 'whole environment' mesh scene, same design rationale as scenes 62-64 -- see build_fireplace_room()'s own comment. A furnished interior rather than a building/street/town-scale environment.",
            "Slow", 150, true, true,
            { 55, -2.0, 1.6, -1.5,  0, 1.3, 0,  0, 0, 0 },
            build_fireplace_room,
            build_fireplace_room_lights,
            build_fireplace_room_sky,
            nullptr
        },
        {
            "H5", 74, SceneNames::SanMiguel, SceneCategories::LargeScene,
            "San Miguel (9.9M triangles) - a dense Mexican hacienda courtyard/villa, the classic 'hero' benchmark scene with real per-face .mtl materials and image textures (tile, wood, fabric, foliage) loaded from models/san_miguel_textures/, lit by an open sky, loaded from an external .obj file (requires models/san_miguel.obj). Fifth 'whole environment' mesh scene, same design rationale as scenes 62-64/73 -- see build_san_miguel()'s own comment.",
            "Very Slow", 150, true, true,
            { 45, 10, 3, 5,  0, 3, 0,  0, 0, 0 },
            build_san_miguel,
            build_san_miguel_lights,
            build_san_miguel_sky,
            nullptr
        },
        {
            "H6", 75, SceneNames::SibenikCathedral, SceneCategories::LargeScene,
            "Sibenik Cathedral - a Gothic cathedral interior (vaulted nave, stone columns, a rose window, colored stained glass), with real per-face .mtl materials, image textures, and real bump maps loaded from models/sibenik_cathedral_textures/, lit through its open doorway/arches, loaded from an external .obj file (requires models/sibenik_cathedral.obj). Sixth 'whole environment' mesh scene, same design rationale as scenes 62-64/73/74 -- see build_sibenik_cathedral()'s own comment.",
            "Very Slow", 400, true, true,
            { 60, -15, 1.7, 0,  15, 5, 0,  0, 0, 0 },
            build_sibenik_cathedral,
            build_sibenik_cathedral_lights,
            build_sibenik_cathedral_sky,
            nullptr
        },
        {
            "H7", 76, SceneNames::BreakfastRoom, SceneCategories::LargeScene,
            "Breakfast Room - a cozy furnished dining interior with glassware, table settings, and marble/tile textures, with real per-face .mtl materials and image textures loaded from models/breakfast_room_textures/, lit by an open sky through its windows, loaded from an external .obj file (requires models/breakfast_room.obj). Seventh 'whole environment' mesh scene, same design rationale as scenes 62-64/73/74/75 -- see build_breakfast_room()'s own comment.",
            "Very Slow", 300, true, true,
            { 70, -3.0, 1.5, 3.0,  2.5, 1.3, 0,  0, 0, 0 },
            build_breakfast_room,
            build_breakfast_room_lights,
            build_breakfast_room_sky,
            nullptr
        },
        {
            "H8", 77, SceneNames::SalleDeBain, SceneCategories::LargeScene,
            "Salle de Bain - a tiled bathroom interior with a mirror, tub, and a real ceiling light fixture (genuine Ke emission -- exercises the NEE-light path a second time, after Fireplace Room), with real per-face .mtl materials and image textures loaded from models/salle_de_bain_textures/, loaded from an external .obj file (requires models/salle_de_bain.obj). Eighth 'whole environment' mesh scene, same design rationale as scenes 62-64/73-75 -- see build_salle_de_bain()'s own comment.",
            "Slow", 150, true, true,
            { 50, 10, 15, -5,  -10, 12, 5,  0, 0, 0 },
            build_salle_de_bain,
            build_salle_de_bain_lights,
            build_salle_de_bain_sky,
            nullptr
        },
        {
            "H9", 78, SceneNames::Gallery, SceneCategories::LargeScene,
            "Gallery - the Hallwyl Museum picture gallery in Stockholm, an ornate room of framed paintings, chandeliers, and a parquet floor, with a real per-face .mtl material and an image texture loaded from models/gallery_textures/, lit by an open sky, loaded from an external .obj file (requires models/gallery.obj). Ninth 'whole environment' mesh scene, same design rationale as scenes 62-64/73-76 -- see build_gallery()'s own comment.",
            "Very Slow", 300, true, true,
            { 55, 0, 2.2, -5,  0, 2.2, 0,  0, 0, 0 },
            build_gallery,
            build_gallery_lights,
            build_gallery_sky,
            nullptr
        },
        {
            "H10", 79, SceneNames::LostEmpire, SceneCategories::LargeScene,
            "Lost Empire - a large half-buried ancient city exported from a Minecraft world, with temple platforms, staircases, and a lava chamber, with real per-face .mtl materials and an image texture loaded from models/lost_empire_textures/, lit by an open sky, loaded from an external .obj file (requires models/lost_empire.obj). Tenth 'whole environment' mesh scene, and the first at a scale (165 units deep) that suits a long video flythrough -- see build_lost_empire()'s own comment.",
            "Slow", 150, true, true,
            { 55, 0, 60, 100,  0, 10, 0,  0, 0, 0 },
            build_lost_empire,
            build_lost_empire_lights,
            build_lost_empire_sky,
            nullptr
        },
        {
            "H11", 80, SceneNames::VokseliaSpawn, SceneCategories::LargeScene,
            "Vokselia Spawn - a small floating voxel island, exported from the same Minecraft world as Lost Empire from its spawn point, with a real per-face .mtl material and an image texture loaded from models/vokselia_spawn_textures/, lit by an open sky, loaded from an external .obj file (requires models/vokselia_spawn.obj). Eleventh 'whole environment' mesh scene -- see build_vokselia_spawn()'s own comment.",
            "Medium", 100, true, true,
            { 40, 4.5, 0.9, 4.5,  0, 0.25, 0,  0, 0, 0 },
            build_vokselia_spawn,
            build_vokselia_spawn_lights,
            build_vokselia_spawn_sky,
            nullptr
        },
        {
            "H12", 81, SceneNames::PowerPlant, SceneCategories::LargeScene,
            "Power Plant - a complete model of an actual coal-fired power plant (12.76M triangles, 5.98M vertices), the largest scene in this collection by triangle count, with flat per-face .mtl colors (no image textures), lit by an open sky, loaded from an external .obj file (requires models/powerplant.obj). Twelfth 'whole environment' mesh scene, and the first needing a real coordinate rescale rather than raw OBJ units -- see build_power_plant()'s own comment.",
            "Slow", 150, true, true,
            { 40, 130, 85, 130,  -55, 40, -35,  0, 0, 0 },
            build_power_plant,
            build_power_plant_lights,
            build_power_plant_sky,
            nullptr
        },
        // H13/H14: real pbrt-v4-scenes bundles (github.com/mmp/pbrt-v4-scenes),
        // NOT git-tracked (see build_curated_external_pbrt_scene_descriptor()'s
        // own comment, scene_registry.h, for why) - download into
        // pbrt_scenes/contemporary-bathroom/ and pbrt_scenes/barcelona-pavilion/
        // respectively (pbrt_scenes/README.md's "Getting scenes" section) for
        // these to render; both already existed locally as parser-verification
        // fixtures (see the many pbrt_cpu_builder.h/material_pbrt.h/
        // PBRT_SUPPORT.md comments citing them by name) well before either got
        // a curated registry entry of its own. Unlike H1-H12's hand-written
        // OBJ "whole environment" scenes, these load through the real pbrt-v4
        // parser/builder like every other pbrt-backed scene in this registry -
        // camera/lights/materials all come from the file itself, not a
        // hand-transcribed struct.
        //
        // Both (and H19/H20 further below) pass a trailing curated
        // recommended_exposure argument - see SceneDescriptor::
        // recommended_exposure's own comment (scene_registry.h) for why
        // these four specifically needed one: at this engine's neutral
        // exposure=1.0 they rendered as a near-black silhouette rather
        // than the furnished/architectural scene the geometry actually
        // contains.
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H13", 159, SceneNames::ContemporaryBathroomPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a fully furnished bathroom interior with real-world material variety (tile, wood, chrome, glass, fabric) and a blackbody-temperature light fixture, one of the two scenes this project's own pbrt-v4 parser was verified against most heavily during development.",
            "Very Slow", "contemporary-bathroom/contemporary-bathroom.pbrt", 15.0),
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H14", 160, SceneNames::BarcelonaPavilionPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a reconstruction of Mies van der Rohe's Barcelona Pavilion, glass-and-marble architecture surrounded by dense foliage - the other major scene this project's own pbrt-v4 parser was verified against during development (its foliage is what motivated real diffusetransmission texture-binding support). Renders the daytime lighting variant.",
            "Very Slow", "barcelona-pavilion/pavilion-day.pbrt", 15.0),
        // H15-H17: three more real pbrt-v4-scenes bundles, same
        // build_curated_external_pbrt_scene_descriptor() mechanism as H13/H14
        // above - already sitting locally as parser-verification fixtures,
        // now given a proper curated entry. Download into
        // pbrt_scenes/sssdragon/, pbrt_scenes/ganesha/, pbrt_scenes/sportscar/
        // respectively (pbrt_scenes/README.md's "Getting scenes" section) for
        // these to render.
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H15", 161, SceneNames::SssDragonPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a dragon statue rendered as translucent skin-like subsurface scattering material lit by an HDR environment map. Three density variants ship in the same directory (dragon_10/50/250.pbrt, differing only in the subsurface scale parameter, 10 being the most translucent); this entry renders the middle one.",
            "Very Slow", "sssdragon/dragon_50.pbrt"),
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H16", 162, SceneNames::GaneshaPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a detailed statue of Ganesha in a marble-like subsurface scattering material, lit by an HDR environment map - a second, independent subsurface-scattering showcase alongside H15's dragon, on a completely different mesh/material combination.",
            "Very Slow", "ganesha/ganesha.pbrt"),
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H17", 163, SceneNames::SportsCarPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a studio-lit sports car with real-world car-paint (layered coated conductor), glass, chrome, and rubber materials under an HDR sky. Renders the daytime sky-lit variant (sportscar-sky.pbrt); a separate area-lit studio variant also ships in the same directory.",
            "Very Slow", "sportscar/sportscar-sky.pbrt"),
        // H18: a genuinely NEW download (not a pre-existing local fixture
        // like H13-H17 were) - github.com/mmp/pbrt-v4-scenes' "zero-day"
        // scene, a detailed game-level-style interior (~422MB). It's an
        // animated camera fly-through shipped as 9 separate per-frame
        // .pbrt files (frame25/35/52/85/120/180/210/300/380.pbrt, each
        // Include-ing its own geometry/geometry-fNNN.pbrt subset); this
        // entry renders frame180, a representative mid-sequence interior
        // shot, the same "pick the reasonable middle option" call made for
        // H15's three subsurface-dragon density variants.
        //
        // Measured render cost is a real outlier even among this
        // registry's other "Very Slow" entries: a 400x400x32spp CPU render
        // still hadn't finished after 700s (H13-H17 all finish comparable
        // resolutions in under a minute to a few minutes), while a 100x100
        // render at any sample count finishes in ~45s - so the cost is
        // concentrated in per-scanline shading at this camera position
        // (151 ObjectInstance placements plus many layered `coateddiffuse`
        // materials), not scene loading/BVH build. The .pbrt file's own
        // authored "1024 pixelsamples" (which recommended_spp below is
        // honestly pulled from, like every other curated pbrt entry) would
        // make a full render impractically slow - said so explicitly here
        // rather than silently reusing the same "Very Slow" wording H13-H17
        // use for renders that actually finish quickly.
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H18", 164, SceneNames::ZeroDayPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a highly detailed, game-level-style interior scene (an office/atrium space) shot as an animated camera fly-through. This entry renders one representative frame (frame180) of the 9 that ship in the bundle. Note: this is markedly heavier than this project's other Large Scenes - per-scanline shading cost at this camera position is high (heavy instancing, many layered materials), so treat the file's own recommended sample count as a starting point to scale down from, not a target.",
            "Very Slow", "zero-day/frame180.pbrt"),
        // H19-H21: three more genuinely NEW downloads (same
        // build_curated_external_pbrt_scene_descriptor() mechanism, same
        // "download into pbrt_scenes/<name>/" convention as H18). All
        // three author their .pbrt with Integrator "volpath" rather than
        // the default path integrator - like every other scene here,
        // cpu_render_main always runs the default path tracer regardless
        // (a warning says so at render time); this only matters if the
        // scene also declares a participating medium (crown's gem
        // interior looks like the most likely candidate of the three) -
        // worth checking the render for a plausible look, since a real
        // medium under the wrong integrator can still look reasonable
        // without necessarily matching pbrt-v4's own reference render.
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H19", 165, SceneNames::CrownPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a jeweled crown rendered with real glass/gem dispersion and refraction - a classic pbrt showcase scene, and a different material story than any other curated Large Scene here (none of H13-H18 focus on dispersion).",
            "Very Slow", "crown/crown.pbrt", 30.0),
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H20", 166, SceneNames::VillaPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): a furnished villa interior/exterior, in the same 'full building environment' vein as H13's bathroom and H14's pavilion. Renders the daylight lighting variant (villa-daylight.pbrt); a lights-on night variant also ships in the same directory.",
            "Very Slow", "villa/villa-daylight.pbrt", 15.0),
        pbrt_scene_registry::build_curated_external_pbrt_scene_descriptor(
            "H21", 167, SceneNames::TransparentMachinesPbrtExample, SceneCategories::LargeScene,
            "A real pbrt-v4-scenes bundle (not bundled with this repo - see pbrt_scenes/README.md): detailed mechanical objects (gears, casings) rendered in transparent glass-like materials, shot as an animated camera fly-through. Five frames ship in the bundle (frame542/675/812/888/1266.pbrt); this entry renders a representative middle one (frame812).",
            "Very Slow", "transparent-machines/frame812.pbrt"),

        // ---------------------------------------------------------------
        // Education (I1-I6): curated demos of the Render Options tab's own
        // controls (Sampler, Spectral rendering, Exposure, Tone mapping,
        // OptiX AI denoiser) and, as of I5/I6, the Settings tab's
        // Integrator selector (SPPM; BDPT/MLT). Each reuses an existing
        // scene's build functions and CameraConfig verbatim - same
        // technique B23/F3 use to share content with another entry - rather
        // than being new renderer content: the description/technique-note
        // is the point, not the geometry. No entry for OptiX validation
        // mode - it has no visual effect by design (extra device-side
        // checks only), so "which scene shows the difference" doesn't
        // apply; I4's note says so instead. No entry for the 5 remaining
        // debug/reference integrators (RandomWalk/AO/SimplePath/
        // SimpleVolPath/LightPath) either - none has a distinct visual
        // showcase angle the way SPPM's caustics or BDPT/MLT's bidirectional
        // convergence do (AO isn't even a lit render), so their explanation
        // stays in the GUI's own per-integrator description text
        // (qt_gui/mainwindow_style.cpp's integratorDescription()) rather
        // than a dedicated scene here.
        // ---------------------------------------------------------------
        {
            // Same world/lights as A1 (Cornell Box) - only the id, category,
            // description, and recommended_spp differ. 16 spp (vs A1's 100)
            // is deliberately low: at that count, different Sampler choices
            // (Sobol/Z-Sobol/Stratified/Halton/...) leave visibly different
            // clumping in the soft shadow penumbra. CPU-only, matching the
            // Sampler control's own tooltip (no effect on GPU) - so no GPU
            // case is needed here.
            "I1", 132, SceneNames::SamplerComparison, SceneCategories::Education,
            "Cornell box rendered at a deliberately low 16 spp so different Sampler choices (Render Options tab) leave visibly different noise/clumping in the soft shadow.",
            "Fast", 16, false, false,
            kCornellBoxCamera,
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            // Same world/lights/punctual-lights as B23 (Glass Prism
            // Dispersion) - the prism fan is already the clearest possible
            // demonstration of --spectral in this registry, so this entry
            // just re-frames it under Education with a description pointing
            // at the Spectral rendering checkbox instead of duplicating the
            // geometry. CPU-only, matching --spectral's own tooltip.
            "I2", 133, SceneNames::SpectralDispersionEducation, SceneCategories::Education,
            "Same glass prism as B23: white light only fans into a visible spectrum with Spectral rendering (Render Options tab) switched on - off, every wavelength refracts by the same fixed amount.",
            "Medium", 200, false, false,
            kPrismCamera,
            build_prism_dispersion,
            no_lights,
            nullptr,
            build_prism_dispersion_punct
        },
        // I3 migrated to pbrt-backed alongside C1 (same world/sky, just a
        // different curated row for the Education category's own purpose) -
        // reuses the identical pbrt_scenes/hdri-sky-gradient.pbrt file C1's
        // own entry does, exactly the "fix an alias scene for free" this
        // project's migration plan anticipated for same-world I-series
        // entries. legacy_id 204 (next after C1's 203). This was the LAST
        // consumer of build_hdri_sky_world_gpu()/case 134 on GPU - deleted
        // below. build_hdri_sky_world()/build_hdri_sky() themselves stay on
        // CPU (tests/integration/sppm_first_slice_test.cpp still calls them
        // directly, independent of the scene registry).
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "I3", 204, SceneNames::ExposureToneMapping, SceneCategories::Education,
            "Same HDR sky gradient as C1: try raising/lowering Exposure, then compare ACES/Reinhard/None Tone mapping (both on the Render Options tab) against this scene's bright sky vs. shadowed sphere.",
            "Medium", "hdri-sky-gradient.pbrt"),
        [] {
            // Same world/lights as A1 (Cornell Box) - migrated to
            // pbrt-backed alongside A1 (reuses cornell-box-native.pbrt
            // verbatim, see A1's own entry below). recommended_spp is
            // force-overridden to 32 after wiring (vs. the file's own 200,
            // which A1 uses) because this scene's whole point is being
            // genuinely noisy on the GPU recursive backend before
            // denoising - build_curated_pbrt_scene_descriptor() has no spp
            // parameter of its own (spp comes from the file's Sampler
            // directive), so this is the one pbrt-backed Education alias
            // that needs its own spp distinct from the file it reuses.
            // Was gpu/optix/scene_builder.cpp's case 135 (a near-verbatim
            // copy of case 0/A1's own GPU case) - deleted now that this
            // falls through to default: -> build_loaded_pbrt_scene() like
            // every other pbrt-backed scene.
            auto s = pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
                "I4", 135, SceneNames::DenoiserComparison, SceneCategories::Education,
                "Cornell box at a deliberately low 32 spp - render once with the OptiX AI denoiser (Render Options tab, GPU only - both the recursive and wavefront backends have their own denoiser) off, once on, and compare. The neighboring OptiX validation mode checkbox has no visual effect either way - it only adds debugging checks.",
                "Fast", "cornell-box-native.pbrt", CameraMode::UserControlled);
            s.recommended_spp = 32;
            return s;
        }(),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            // Same world/lights as B3 (Cornell Rough Glass) - migrated to
            // pbrt-backed alongside B3 (reuses cornell-rough-glass.pbrt
            // verbatim, see B3's own entry below). No spp override needed:
            // the file's own Sampler already specifies 200, matching this
            // scene's original intended spp exactly. Was
            // gpu/optix/scene_builder.cpp's case 139 - deleted now that
            // this falls through to default: -> build_loaded_pbrt_scene()
            // like every other pbrt-backed scene. SPPM's own GPU capability
            // check (gpu/optix/optix_types.h's
            // sppm_gpu_material_supported()) already whitelists
            // RoughDielectric independent of this registry entry, so
            // --sppm --gpu keeps working here too, not just plain --sppm.
            "I5", 139, SceneNames::SppmCausticsEducation, SceneCategories::Education,
            "Same Cornell box as B3, with a rough-dielectric (frosted glass) sphere: render once with the default Path Tracer, once with SPPM (Integrator dropdown, Render Options tab), and compare how much faster the floor caustic cleans up - SPPM's photon mapping is built for exactly this case.",
            "Medium", "cornell-rough-glass.pbrt", CameraMode::UserControlled),
        [] {
            // Same world/lights as A1 (Cornell Box) - migrated to
            // pbrt-backed alongside A1 (reuses cornell-box-native.pbrt
            // verbatim, see A1's own entry below). recommended_spp is
            // force-overridden to 100 after wiring (vs. the file's own 200)
            // to preserve this scene's original BDPT/MLT-comparison spp -
            // see I4's own entry just above for why the override is needed
            // at all. Was gpu/optix/scene_builder.cpp's case 140 - deleted
            // now that this falls through to default: ->
            // build_loaded_pbrt_scene() like every other pbrt-backed scene.
            // BDPT/MLT themselves still have no GPU implementation at all
            // (CPU only, unconditionally; see main.cpp's own --gpu-ignored
            // warning under either flag) - unaffected by this migration.
            auto s = pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
                "I6", 140, SceneNames::BdptMltEducation, SceneCategories::Education,
                "Same Cornell box as A1: try BDPT or MLT (Integrator dropdown, Render Options tab) instead of the default Path Tracer - both trace light paths from the camera AND the light source and connect them, which can converge differently than the default on scenes with indirect lighting like this one.",
                "Medium", "cornell-box-native.pbrt", CameraMode::UserControlled);
            s.recommended_spp = 100;
            return s;
        }(),
        {
            // Same world/lights as A1 (Cornell Box) - deliberately CPU-only
            // (gpu_compatible=false, matching I1's own precedent) since
            // every alternate integrator this demonstrates is CPU-only:
            // --randomwalk (no NEE/MIS at all - pure uniform-sphere BSDF
            // sampling, pbrt-v4's unbiased reference), --simplepath-no-bsdf
            // (NEE/light-sampling only), --simplepath-no-lights (BSDF
            // importance sampling only), and the default MIS-combined
            // path tracer - all reachable from the same Integrator
            // dropdown (Render Options tab). This box's small ceiling
            // light against mostly-indirect Lambertian bounces is exactly
            // the case where NEE-only and BSDF-only converge very
            // differently (NEE nails the direct light term cheaply, BSDF
            // sampling instead has to get lucky and hit the small light by
            // chance) and MIS combines both - random-walk has neither and
            // is visibly the noisiest of the four at equal spp.
            "I7", 155, SceneNames::LightTransportStrategies, SceneCategories::Education,
            "Same Cornell box as A1: pick RandomWalk, SimplePath (then try its NEE/BSDF sub-checkboxes), or leave the default Path Tracer (Integrator dropdown, Render Options tab) and compare noise at the same low sample count - each includes a different subset of next-event estimation and BSDF importance sampling, and MIS (the default) is what combines both well.",
            "Fast", 32, false, false,
            kCornellBoxCamera,
            build_cornell_box,
            build_cornell_box_lights
        },
        {
            // Purpose-built world (see build_light_sampler_comparison()'s
            // own comment, cornell_box_scene.h, for the full design/power-
            // ratio rationale) - the one Education scene not simply
            // reusing another entry's geometry unchanged, since no
            // existing scene has enough lights of different power to show
            // a light-sampler-strategy difference at all. CPU-only,
            // matching --lightsampler's own "CPU default path tracer
            // only" scope.
            "I8", 156, SceneNames::LightSamplerComparison, SceneCategories::Education,
            "Cornell box with five ceiling lights of deliberately lopsided power (roughly 1:2:6:15:80) instead of one: try Uniform vs. Power vs. BVH (Light sampler, Render Options tab) at a low sample count - Uniform spends a fifth of its next-event-estimation samples on each light regardless of how much it actually contributes, so it stays noisier on the one dominant light than Power/BVH, which weight selection toward it instead.",
            "Fast", 32, false, false,
            kCornellBoxCamera,
            build_light_sampler_comparison,
            build_light_sampler_comparison_lights
        },
        {
            // Same world/lights as A1 (Cornell Box) - CPU-only
            // (gpu_compatible=false), matching --ao's own "CPU only" scope.
            // AOIntegrator skips material color and indirect lighting
            // entirely (pure occlusion visualization), so this renders as
            // flat grayscale with the box/sphere's own contact shadows and
            // crevices as the only visible structure - about as different
            // from this same box's usual lit render as any Integrator
            // switch in this registry produces.
            "I9", 157, SceneNames::AmbientOcclusionEducation, SceneCategories::Education,
            "Same Cornell box as A1: switch to Ambient Occlusion (Integrator dropdown, Render Options tab) - a debug/visualization mode with no material color or indirect light at all, just a grayscale occlusion term from nearby geometry, and compare against the default Path Tracer's full lit render of the identical scene.",
            "Fast", 64, false, false,
            kCornellBoxCamera,
            build_cornell_box,
            build_cornell_box_lights
        },
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            // Same world/lights as B3 (Cornell Rough Glass) - the same
            // hard-caustic scene I5 already reuses for SPPM, and for the
            // identical reason: it's the one CPU scene launcher_args.h's
            // own --sppm help text calls "verified end-to-end", and its
            // frosted-glass floor caustic is specifically the kind of
            // hard specular-then-diffuse path that produces fireflies
            // under plain path tracing - exactly the case --regularize
            // and --maxcomponentvalue each exist to tame, via two
            // different mechanisms (widening the BSDF vs. clamping the
            // sample directly). Migrated to pbrt-backed alongside B3/I5
            // (reuses cornell-rough-glass.pbrt verbatim). No spp override
            // needed: the file's own Sampler already specifies 200,
            // matching this scene's original intended spp exactly. Was
            // gpu/optix/scene_builder.cpp's case 158 - deleted now that
            // this falls through to default: -> build_loaded_pbrt_scene()
            // like every other pbrt-backed scene. Both --regularize and
            // --maxcomponentvalue still work on both GPU backends
            // (recursive exact, wavefront approximate for the clamp - see
            // each flag's own help text), unaffected by this migration.
            "I10", 158, SceneNames::FireflySuppression, SceneCategories::Education,
            "Same rough-glass Cornell box as B3 (and I5's own SPPM demo): render once plain, once with Regularize checked, once with Firefly clamp (--maxcomponentvalue) checked instead (both on the Render Options tab) - the hard caustic through the frosted sphere is exactly the case each is built to tame, via two different mechanisms (blurring the BSDF vs. clamping the sample directly).",
            "Medium", "cornell-rough-glass.pbrt", CameraMode::UserControlled),

        // ---------------------------------------------------------------
        // Curated pbrt_scenes/*.pbrt example scenes, under their real topic
        // tab instead of only the generic "Custom Scenes" bucket every
        // loaded .pbrt file auto-discovers into (see
        // pbrt_scene_registry::build_curated_pbrt_scene_descriptor()'s own
        // comment). Legacy ids 100+ - past every real case in
        // gpu/optix/scene_builder.cpp's switch, same reasoning as
        // append()'s own dynamically-assigned ids below.
        // ---------------------------------------------------------------

        // -- Materials --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B15", 100, SceneNames::MixMaterialPbrtExample, SceneCategories::Materials,
            "Real per-shading-point stochastic pbrt \"mix\" material resolution on all three backends -- a fine-grained speckle of matte red diffuse and a conductor's real specular highlights, not one flat averaged color.",
            "Fast", "mix-material.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B16", 101, SceneNames::LayeredMaterialsPbrtExample, SceneCategories::Materials,
            "Four pbrt material kinds no other bundled example scene touches: thindielectric, coatedconductor, diffusetransmission, and subsurface via a named measured-scattering preset (\"Marble\", no external file needed).",
            "Fast", "layered-materials.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B18", 103, SceneNames::ConductorRgbEtaKPbrtExample, SceneCategories::Materials,
            "Explicit RGB eta/k for pbrt's conductor material, plus named-metal-spectrum resolution for coatedconductor -- real complex-IOR GGX highlights instead of the flat fuzz-mirror/reflectance-only fallback.",
            "Fast", "conductor-rgb-eta-k.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B20", 105, SceneNames::HairMaterialPbrtExample, SceneCategories::Materials,
            "pbrt's Material \"hair\" (Marschner/Chiang fiber scattering) applied to ordinary spheres, matching this project's own native Hair Fibers demo for a fair comparison.",
            "Fast", "hair-material.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B22", 107, SceneNames::NamedMaterialAndTexturePbrtExample, SceneCategories::Materials,
            "pbrt's NamedMaterial referenced directly for a shape (not just as a \"mix\" sub-material), plus a texture-bound material parameter and AreaLightSource's twosided flag.",
            "Fast", "named-material-and-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "B25", 146, SceneNames::GlassPresetsPbrtExample, SceneCategories::Materials,
            "All seven of Material \"dielectric\"'s named glass IOR presets (BK7, BAF10, FK51A, LASF9, F5, F10, F11) as separate spheres, resolved via FindGlassPreset() -- previously exercised only by unit tests, never rendered.",
            "Fast", "glass-presets.pbrt"),

        // -- Textures --
        // Split out of Materials once it grew past 25 scenes mixing two
        // distinct concerns - see SceneCategories::Textures's own comment
        // (scene_descriptor.h). Ids keep their original legacy_id (the
        // stable internal scene_builder.cpp switch key - see
        // SceneDescriptor::legacy_id's comment) even though their letter-id
        // changed from B to J, matching this file's own precedent of
        // reassigning a scene's user-facing id when its category genuinely
        // changes rather than preserving a now-inconsistent one.
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J1", 102, SceneNames::CoatedDiffuseTexturePbrtExample, SceneCategories::Textures,
            "Real texture-bound reflectance for pbrt's CoatedDiffuse material, which previously silently dropped to a flat color on both backends.",
            "Fast", "coateddiffuse-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J2", 104, SceneNames::DiffuseTransmissionTexturePbrtExample, SceneCategories::Textures,
            "Texture-bound reflectance/transmittance for pbrt's DiffuseTransmission material, threaded through both backends.",
            "Fast", "diffusetransmission-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J3", 106, SceneNames::NestedCheckerTexturePbrtExample, SceneCategories::Textures,
            "One level of nested imagemap texture reference inside a pbrt checkerboard/mix texture -- tex1/tex2 bound to a real image instead of only a flat literal color.",
            "Fast", "nested-checker-texture.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J4", 147, SceneNames::TextureEncodingWrapInvertPbrtExample, SceneCategories::Textures,
            "Four quads isolating Texture \"imagemap\"'s \"encoding\"/\"wrap\"/\"invert\" params one at a time -- linear vs. sRGB decode, clamp vs. repeat past [0,1], and inverted channel values, none reachable via any other bundled scene's defaults.",
            "Fast", "texture-encoding-wrap-invert.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J5", 148, SceneNames::ProceduralTextureGalleryPbrtExample, SceneCategories::Textures,
            "Four pbrt-v4 procedural texture classes wired into the CPU builder but never used by any other bundled scene: windy turbulence, wrinkled (Perlin-octave) turbulence, dots, and bilerp corner-blend.",
            "Fast", "procedural-textures-windy-wrinkled-dots-bilerp.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "J6", 149, SceneNames::NestedTexture2LevelPbrtExample, SceneCategories::Textures,
            "A second level of checkerboard-texture nesting (checker of a checker of a real image) -- real on CPU, GPU intentionally approximates the whole nested tree as one flat average colour and warns; compare both to see the documented divergence.",
            "Fast", "nested-texture-2level.pbrt"),

        // -- Lights --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C8", 108, SceneNames::PunctualLightsPbrtExample, SceneCategories::Lights,
            "All five of pbrt-v4's punctual (delta-distribution) light kinds in one scene: point, spot, distant, goniometric, and projection.",
            "Fast", "punctual-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C9", 109, SceneNames::GoniometricProjectionPbrtExample, SceneCategories::Lights,
            "Real image decoding for pbrt's goniometric and projection lights, which previously silently ignored their own filename and fell back to a uniform beam.",
            "Fast", "goniometric-projection.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C10", 110, SceneNames::BlackbodyLightPbrtExample, SceneCategories::Lights,
            "pbrt's \"blackbody L\" colour-temperature area lights -- two identical panels at 2500K and 9000K, so a regression back to flat-white emission would be immediately visible.",
            "Fast", "blackbody-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C11", 111, SceneNames::TexturedTwoSidedLightsPbrtExample, SceneCategories::Lights,
            "A real filename-textured, two-sided AreaLightSource on a non-triangle (sphere/quad) shape, on both backends.",
            "Fast", "textured-twosided-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C12", 112, SceneNames::InfiniteLightPbrtExample, SceneCategories::Lights,
            "pbrt's \"infinite\" constant-colour sky light in open geometry, actually lighting the scene from every direction rather than being blocked by a room.",
            "Fast", "infinite-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C13", 113, SceneNames::DiskCylinderLightPbrtExample, SceneCategories::Lights,
            "Disk and cylinder shapes as real NEE-samplable area lights on both GPU backends, converging as cleanly as CPU's solid-angle sampling instead of noisier hit-only emission.",
            "Fast", "disk-cylinder-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C14", 114, SceneNames::TwoSphereLightsPbrtExample, SceneCategories::Lights,
            "Two sphere area lights in one scene, pinning a GPU light-type-table width bug where every light after the first silently misread its own type.",
            "Fast", "two-sphere-lights.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C15", 115, SceneNames::TriangleFanLightPbrtExample, SceneCategories::Lights,
            "An area light that is NOT a parallelogram -- an irregular 5-triangle fan the quad-merge pass can't rejoin, exercising the GPU's per-triangle light sampling.",
            "Fast", "triangle-fan-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C16", 137, SceneNames::ColorSpaceBlackbodyPbrtExample, SceneCategories::Lights,
            "Identical to the Blackbody Light example except for one added \"ColorSpace rec2020\" directive -- the same 2500K/9000K temperatures resolve to visibly different RGB under Rec.2020's wider primaries than the sRGB default.",
            "Fast", "colorspace-blackbody.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C17", 142, SceneNames::PortalLightPbrtExample, SceneCategories::Lights,
            "The real pbrt-v4 PortalImageInfiniteLight -- an equal-area environment map restricted to a single window quad, so only that opening shows real sky detail. The native \"Portal Infinite Light\" scene (C7) only cuts a geometric hole in a wall behind a flat sky_light; this is the class it doesn't actually build.",
            "Fast", "portal-light.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C18", 143, SceneNames::LightPowerParameterPbrtExample, SceneCategories::Lights,
            "pbrt-v4's \"float power\" parameter on point, spot, and area lights -- three otherwise-identical spheres, each lit only by one light type specifying total flux instead of intensity/radiance directly, so a wrong power-to-intensity conversion shows up as a visibly mismatched brightness.",
            "Fast", "light-power-parameter.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C19", 144, SceneNames::ProjectionLightNonSquarePbrtExample, SceneCategories::Lights,
            "LightSource \"projection\" with a non-square (8x4) slide image, the first bundled scene to exercise a real aspect-ratio mismatch between the image and the light's own field of view -- a correct render shows a clearly wide rectangular footprint, not a squished or stretched one.",
            "Fast", "projection-light-nonsquare.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "C20", 145, SceneNames::SpectralGamutSaturationPbrtExample, SceneCategories::Lights,
            "Three saturated, close-range colored lights on a plain diffuse surface under --spectral rendering, chosen so their overlap sits right at the sRGB gamut boundary -- exactly the condition a per-sample XYZ->RGB gamut-clamp bug used to darken and desaturate incorrectly.",
            "Fast", "spectral-gamut-saturation.pbrt"),

        // -- Cameras --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D9", 116, SceneNames::DepthOfFieldPbrtExample, SceneCategories::Cameras,
            "A perspective camera's thin-lens depth-of-field (lensradius/focaldistance) loaded from a pbrt file, on both backends.",
            "Fast", "depth-of-field.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D10", 117, SceneNames::OrthographicCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's orthographic (parallel-projection) camera loaded from a file -- two same-size spheres at different depths read as equal size, not perspective-foreshortened.",
            "Fast", "orthographic-camera.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D11", 118, SceneNames::SphericalCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's spherical (equal-area) camera loaded from a file, positioned inside an enclosed room so it actually captures every direction at once.",
            "Fast", "spherical-camera.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "D12", 119, SceneNames::RealisticCameraPbrtExample, SceneCategories::Cameras,
            "pbrt's realistic multi-element lens camera loaded from a file, including the lensfile-loading path a compiled-in scene never exercised.",
            "Fast", "realistic-camera.pbrt"),

        // -- Volumes --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E5", 120, SceneNames::CloudMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"cloud\" (Perlin-noise heterogeneous scattering) loaded from a file, on both backends.",
            "Fast", "cloud-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E6", 121, SceneNames::CylinderMediumPbrtExample, SceneCategories::Volumes,
            "A homogeneous fog medium on pbrt's Shape \"cylinder\", now real on both GPU backends instead of silently rendering as ordinary empty geometry.",
            "Fast", "cylinder-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E11", 201, SceneNames::ThinDielectricMediumPbrtExample, SceneCategories::Volumes,
            "Material \"thindielectric\" fused with MediumInterface, now real on both GPU backends via the same DielectricMedium material the smooth-dielectric fusion case uses.",
            "Fast", "thin-dielectric-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E12", 202, SceneNames::RoughDielectricMediumPbrtExample, SceneCategories::Volumes,
            "A frosted (rough) Material \"dielectric\" fused with MediumInterface, now real on both GPU backends via a GGX microfacet DielectricMedium surface with real glossy NEE.",
            "Fast", "rough-dielectric-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E7", 122, SceneNames::RgbGridMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"rgbgrid\" (an RGB voxel grid) rendering as a soft coloured nebula, on both backends.",
            "Fast", "rgbgrid-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E8", 123, SceneNames::UniformGridMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"uniformgrid\" (a single-channel density voxel grid) rendering as a soft glowing blob, on both backends.",
            "Fast", "uniformgrid-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E9", 141, SceneNames::NanoVdbMediumPbrtExample, SceneCategories::Volumes,
            "pbrt's MakeNamedMedium \"nanovdb\" (a real NanoVDB-format sparse density grid read from an external .nvdb file) rendering as a soft fog-volume sphere - CPU only, GPU falls back to flat homogeneous fog.",
            "Fast", "nanovdb-medium.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "E10", 154, SceneNames::CameraMediumPbrtExample, SceneCategories::Volumes,
            "pbrt-v4's camera-medium idiom -- a MediumInterface issued before the Camera directive puts the camera itself inside a fog with no boundary shape at all, unlike every other bundled medium scene. Real on CPU and GPU-recursive; GPU-wavefront support is deferred.",
            "Fast", "camera-medium.pbrt"),

        // -- Geometry --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F5", 124, SceneNames::PlymeshUvPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"plymesh\" real per-vertex UV data, threaded through both backends -- previously GPU-recursive rendered this exact scene solid black.",
            "Fast", "plymesh-uv.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F6", 125, SceneNames::PlymeshGeometryPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"plymesh\" loading a real external .ply file, including fan-triangulation of a non-triangular base face.",
            "Fast", "plymesh-geometry.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F7", 126, SceneNames::CurveTuftPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"curve\" (real cubic-Bezier fiber geometry, tessellated for GPU) compared against this project's own native curve-tuft demo.",
            "Fast", "curve-tuft.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F8", 127, SceneNames::CurveHairTuftPbrtExample, SceneCategories::Geometry,
            "Real curve geometry paired with Material \"hair\" for the first time -- the exact combination that motivated HairBxDF's own fiber-tangent fix.",
            "Fast", "curve-hair-tuft.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F9", 128, SceneNames::TrianglemeshUvPbrtExample, SceneCategories::Geometry,
            "pbrt's Shape \"trianglemesh\" \"point2 uv\" parameter threaded through both backends -- previously GPU-recursive rendered this exact scene solid black.",
            "Fast", "trianglemesh-uv.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F10", 129, SceneNames::PixelFilterBoxPbrtExample, SceneCategories::Geometry,
            "pbrt's PixelFilter directive end-to-end on both backends -- a box filter's harder, more aliased silhouette edges compared to the default Gaussian.",
            "Fast", "pixel-filter-box.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F11", 150, SceneNames::ObjectMotionBlurPbrtExample, SceneCategories::Geometry,
            "Shape \"sphere\" object motion blur via ActiveTransform StartTime/EndTime -- a moving sphere renders as a soft directional streak instead of a crisp or doubled sphere, real on all three backends (CPU, GPU-recursive, GPU-wavefront).",
            "Fast", "object-motion-blur.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F12", 151, SceneNames::DiskCylinderMotionBlurPbrtExample, SceneCategories::Geometry,
            "Shape \"disk\"/\"cylinder\" object motion blur via ActiveTransform -- CPU only, both GPU backends render these shapes frozen at their start pose and warn instead of blurring.",
            "Fast", "disk-cylinder-motion-blur.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F13", 152, SceneNames::ReverseOrientationPbrtExample, SceneCategories::Geometry,
            "The bare ReverseOrientation directive on two identical-winding quads -- one stays dark (normal facing away from the camera), the other is flipped visible by ReverseOrientation, so a mistake shows up as both quads dark or both lit.",
            "Fast", "reverseorientation.pbrt"),
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "F14", 153, SceneNames::ConeParaboloidGalleryPbrtExample, SceneCategories::Geometry,
            "Shape \"cone\"/\"paraboloid\" (this project's own pbrt-v3-compatibility extension) as plain diffuse shapes, an area-light emitter, and a medium boundary in one gallery -- CPU only, GPU drops cone/paraboloid shapes entirely.",
            "Fast", "cone-paraboloid-gallery.pbrt"),

        // -- Models --
        pbrt_scene_registry::build_curated_pbrt_scene_descriptor(
            "G25", 130, SceneNames::KillerooSimplePbrtExample, SceneCategories::Models,
            "The classic pbrt-v4 \"killeroo\" statue example scene, loaded end-to-end from its own .pbrt file rather than a compiled-in scene.",
            "Medium", "killeroo-simple.pbrt"),
    };
    return registry;
}
