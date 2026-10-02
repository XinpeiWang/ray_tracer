/**
 * @file material_cpu_gpu_parity_tests.cpp
 * @brief Per-material CPU vs GPU-recursive vs GPU-wavefront parity sweep
 *
 * cpu_gpu_comparison_tests.cpp (see its own file comment) only ever renders
 * ONE scene ("A1", a plain lambertian Cornell box) and never touches the
 * wavefront backend at all. That leaves every other MaterialType (rough
 * metal, conductor, coated diffuse/conductor, thin glass, wax slab,
 * crystal, hair, measured BRDF, subsurface, participating media, ...)
 * completely unchecked for CPU/GPU-recursive/GPU-wavefront agreement.
 *
 * This gap is not hypothetical: a prior session hit two real CPU/GPU
 * material bugs that were only caught because a human happened to notice
 * the renders "looked different" (a measured-BRDF investigation that turned
 * out to be a red herring, and a real HDRI-sky GPU flat-color-fallback bug
 * found afterward, which showed up as a 35-58% average-brightness gap
 * between backends). An automated per-material sweep like this one would
 * have caught the second bug immediately instead of costing ad hoc
 * debugging time.
 *
 * Coverage: every scene in SceneCategories::Materials, SceneCategories::Volumes,
 * and SceneCategories::Textures (the last split out of Materials once it grew
 * past 25 scenes - see SceneCategories::Textures's own comment,
 * scene_descriptor.h), filtered from the registry by category rather than
 * hand-listed, so any future scene added to any of the three is automatically
 * picked up here. Between them these categories cover nearly every
 * MaterialType value in gpu/optix/optix_types.h.
 *
 * Tolerance calibration: 30% relative difference (kRelTolerance) on average
 * brightness and on each of R/G/B channel averages, for every Materials
 * scene; 55% (kVolumeRelTolerance) for Volumes scenes specifically - not
 * guessed, but arrived at through actual calibration runs against this
 * codebase's real renderer, EACH SCENE RUN IN FULL PROCESS ISOLATION (see
 * "GPU cross-scene state corruption" below for why isolation mattered here):
 *   - 30% is tighter than cpu_gpu_comparison_tests.cpp's own
 *     HighSPPBrightnessConverges (50%), which is deliberately generous for a
 *     single, well-understood scene. 30% leaves real margin below the
 *     smallest gap the known HDRI-sky bug produced (35%), so a bug of that
 *     size or larger would still be caught here.
 *   - SPP: CPU 200 / GPU 600 (300/900 for Volumes - see below), at 60x60
 *     resolution, mirrors HighSPPBrightnessConverges's own CPU-buys-more-
 *     efficiency-per-sample reasoning (that test used 100 CPU / 500 GPU at
 *     80x80).
 *   - Volumes scenes (E1-E4) get both higher SPP AND a separately-justified
 *     55% ceiling (not just higher SPP against the standard 30%), originally
 *     calibrated against a real 46.8% measured gap caused by a confirmed,
 *     deliberate backend gap (GPU medium scattering had no NEE/MIS) that has
 *     SINCE BEEN FIXED and this ceiling now needs re-measuring - see
 *     kVolumeRelTolerance's own comment for the full, current status.
 *   - IMPORTANT - B1 (RoughMetalSpheres) and B13 (SubsurfaceSlab) are
 *     deliberately NOT given a wider tolerance despite both failing at the
 *     standard 30%, and are left FAILING intentionally - that failure IS
 *     the sweep doing its job. Both were re-verified with EACH SCENE AS THE
 *     ONLY GPU RENDER IN ITS OWN PROCESS, specifically to rule out the
 *     cross-scene corruption described below contaminating the numbers:
 *       - B13: consistently measured CPU ~32-38% BRIGHTER than BOTH
 *         independent GPU backends, across avg brightness and every
 *         channel, in isolated single-scene runs with no other scene ever
 *         rendered in the same process (e.g. G channel: CPU=0.374 vs
 *         GPU-wavefront=0.231, 37.6%). Two independently-implemented GPU
 *         backends landing in the same ~32-38% dimmer band, on every
 *         channel, with no cross-scene state involved, is far too
 *         systematic to be Monte-Carlo noise or a corruption artifact.
 *         This looks like a genuine, currently-unexplained CPU vs GPU
 *         BSSRDF discrepancy.
 *         SUPERSEDED: this CPU-brighter gap was since fixed (root cause no
 *         longer reproduces - confirmed by re-running against a pre-
 *         shadow-transmittance build, where B13 passed at the standard 30%
 *         tolerance). A NEW, opposite-signed gap (CPU ~32% DARKER) appeared
 *         after CPU shadow rays gained real transmittance through media -
 *         see kSubsurfaceSlabRelTolerance's own comment below for that
 *         gap's cause. The paragraph above is kept as history for why 30%
 *         was the original standard tolerance, not as the current gap.
 *       - B1: measured CPU-vs-GPU-wavefront B-channel gap sits right at the
 *         30% edge (30.08% / 30.03% / 30.10% across repeated isolated runs)
 *         and is confined to that one backend pair and one channel -
 *         GPU-recursive matches CPU fine. Much smaller and less certain
 *         than B13, but flagged rather than silently tolerance-widened
 *         away, since it sits consistently just past the line rather than
 *         drifting either side of it run to run.
 *     A scene positioned LATER in registry order during a full,
 *     non-isolated multi-scene run of this suite (e.g. B2 CornellRoughMetal)
 *     can ALSO show up as "FAILED" - that is cross-scene state corruption
 *     (see below), not a material bug; only the isolated-run numbers above
 *     are trustworthy as material findings.
 *
 * GPU cross-scene state corruption - RESOLVED (2026-08-17, commit
 * 7f2bc463), this comment block just never got updated to say so:
 *
 *   Rendering enough DIFFERENT scenes back-to-back on the GPU in one process
 *   used to eventually corrupt the CUDA/OptiX context (CUDA error 700, "an
 *   illegal memory access was encountered"), after which every further OptiX
 *   call in that process failed. Found empirically while building this
 *   suite's tolerance calibration: an early per-scene-interleaved design
 *   (CPU/recursive/wavefront for scene N, then scene N+1, ...) crashed after
 *   ~2 scenes; restructuring to three separate whole-suite passes (all-CPU,
 *   then one uninterrupted all-scenes GPU-recursive pass, then one
 *   uninterrupted all-scenes GPU-wavefront pass - see build_cache_once()
 *   below) moved the crash later but did not eliminate it - in one observed
 *   run the GPU-recursive-only pass crashed partway through, at the 13th
 *   scene. Root-caused and fixed the same day this suite was first added
 *   (7f2bc463, ~2 hours after 134106bd): WorkQueue::push()
 *   (wavefront_types.h) incremented its counter past capacity
 *   unconditionally, and 5 wavefront kernel consumers trusted that counter
 *   alone as a bounds check - scene B2 (Cornell Rough Metal) pushes both an
 *   area-light and a sky-NEE shadow ray per hit, overflowing the shadow
 *   queue, and the resulting out-of-bounds device read is what corrupted the
 *   context. A second bug in the same commit - optix_render_main() never
 *   called enableWavefront(false), so wavefront mode stayed latched on once
 *   any earlier call enabled it - explains why the "GPU-recursive-only" pass
 *   that crashed at scene 13 (B13) was secretly still running wavefront.
 *
 *   Re-verified empirically for THIS file specifically (2026-10-02, Phase 2
 *   test-coverage scoping): a trial expansion from this suite's 40 Materials/
 *   Volumes/Textures scenes to 96 (adding Lights/Cameras/Geometry/Basics) ran
 *   in one process with zero CUDA errors and zero "render failed to produce
 *   a valid PPM" failures - the corruption this section used to describe
 *   does not recur even at more than double this suite's own scene count.
 *   That same trial is what surfaced a REAL bug of the identical shape (a
 *   wavefront work queue sized for 1 item/pixel silently dropping items for
 *   a scene that legitimately needs more) in a different queue - see
 *   wavefront_path_tracer.cpp's own shadowQueueCapacity_ comment for the
 *   full story (fixed separately, not a corruption-class issue - WorkQueue::
 *   push()'s overflow guard already made it safe, just silently incorrect).
 *
 *   The now-safely-contained version of this bug class (a work queue
 *   consumer trusting its own push() counter without checking the OTHER end
 *   - the backing buffer's real capacity - for every site that reads it) is
 *   worth a one-time audit if a future queue's capacity ever needs to differ
 *   from numPixels again, but is not tracked as a known open issue today.
 *
 * Regional (block-based) diff, added later: every check described above
 * reduces a whole rendered image to 1-4 floats (overall brightness, 3
 * channel averages), which an audit of this project's testing found to be
 * a real, structural gap - it cannot catch a divergence that's spatially
 * localized rather than global (a camera-framing/geometry shift, a shading
 * bug confined to part of a surface, a color-channel swap on a scene where
 * R/B happen to average close together), since any such bug gets diluted
 * across every other unaffected pixel before the whole-image average ever
 * sees it. mp_regional_diff()/check_regional_parity() add a 6x6-grid,
 * per-block relative-difference check reusing the exact same cached renders
 * (no extra rendering cost), coarse enough to stay robust to ordinary
 * Monte-Carlo noise but fine-grained enough to actually localize a
 * divergence - see mp_regional_diff's own comment for the full reasoning,
 * and regional_tolerance_for's own comment for why its tolerance is scaled
 * from each scene's own whole-image tolerance rather than one fixed global
 * number (so E10/Volumes/B13/B1's already-documented, already-accepted
 * gaps above don't get re-flagged here as if they were new findings).
 *
 * Single-scene isolation for calibration/debugging: set the
 * MATPARITY_ONLY_SCENE_ID environment variable to a scene id (e.g. "B14")
 * before running this suite to render ONLY that scene, routing around the
 * GPU cross-scene corruption bug described above so a finding can be
 * verified as real rather than a corruption artifact (exactly the manual
 * "temporarily filter testable_scenes()" workaround the B1/B13 findings
 * above used, now a real, permanent, documented mechanism instead of a
 * one-off edit-and-revert).
 *
 * The regional check immediately found 4 new, real, isolation-verified
 * divergences on its first real run that NONE of the whole-image checks
 * above had ever caught (none of these 4 scenes had any pre-existing
 * tolerance exception). All 4 are now resolved, each described below:
 * B14 and B11 turned out to be firefly variance, not bugs - given their
 * own regional-tolerance exceptions rather than masked by a widened one,
 * same philosophy as B1/B13; E6 was a real bug, fully fixed, no exception
 * needed; J2 got a real, if partial, texture-filtering fix plus a
 * tolerance exception for its remaining already-documented NEE-strategy
 * gap. A 5th scene, E10, surfaced the same way on a later full-suite run -
 * see its own entry below (after J2's) for why it's a regional-tolerance
 * calibration gap for an already-fully-documented, already-accepted
 * limitation, not a new bug needing investigation.
 *   - B14 (Measured BRDF): CPU ~50-60% brighter than BOTH GPU backends in
 *     one block. Thoroughly investigated, NOT found to be a code bug -
 *     likely explained by firefly variance from this material's narrow,
 *     NEE-less specular lobe reflected toward a small area light (verified
 *     is_specular=true/skip_pdf=true, no NEE, identically on all three
 *     backends - material_pbrt.h's `measured::scatter()`,
 *     optix_device_helpers.h's MaterialType::Measured case,
 *     wavefront_kernels_materials.cu's). A real, if minor, precision issue
 *     WAS found and fixed along the way (gpu/optix/optix_measured_bxdf.h/
 *     wavefront_measured_bxdf.h's phi_m could land outside [-pi,pi], where
 *     this build's --use_fast_math-substituted __sinf()/__cosf() measurably
 *     degrade vs CPU's exact-range-reduction std::sin/std::cos - now
 *     wrapped back into range) - but applying that fix in isolation moved
 *     this gap by under 1 percentage point, ruling it out as the
 *     explanation. Line-by-line verification of the ENTIRE CPU vs GPU
 *     sample_f() chain (theta2u/phi2u, the PiecewiseLinear2D Sample()/Eval()
 *     device ports, the jacobian/scale/normalization formula, the table-
 *     flattening step, the kLambdaR/G/B wavelengths, the wo sign
 *     convention) found it algebraically identical, term for term, on both
 *     GPU backends - no transposition, wrong-table read, or missing factor
 *     anywhere. Two further, decisive pieces of evidence point at variance
 *     rather than a bug: (1) re-rendering CPU alone 5x with different
 *     --seed values at this suite's own 200spp/60x60 settings showed this
 *     SAME block swinging by up to 24.6% between two purely CPU, same-code
 *     runs - this block is inherently high-variance even with no GPU
 *     involved at all; (2) GPU-recursive vs GPU-wavefront (two independent
 *     ports, likely with correlated RNG/sampling conventions to each other
 *     that differ from CPU's own independent RNG) PASS this exact
 *     regional-diff check against each other, while both independently
 *     differ from CPU by a similar ~57-60% - consistent with "GPU's two
 *     backends resemble each other more than either resembles CPU's
 *     independent noise realization," not "both GPU ports share a bug".
 *   - E6 (Cylinder Medium, pbrt example) - RESOLVED, a real bug, found via
 *     device-side printf after static code comparison alone wasn't enough:
 *     CPU vs both GPU backends originally showed a 100% block diff (total
 *     disagreement) confined to 5-6 of 24 comparable blocks. The root
 *     theory (pbrt cylinders have no end caps; GPU's `__intersection__
 *     cylinder`/`__intersection__wf_cylinder` only ever tested the lateral
 *     WALL crossing, unlike CPU's dedicated `volume_bounds()` path for a
 *     Medium/DielectricMedium-attached cylinder) was CORRECT, but a first
 *     fix attempt (an intersection-program fallback reporting a hit from
 *     the tube-x-zslab volume interval) measured as producing ZERO change,
 *     which falsely looked like a disproof. Gating the fallback's debug
 *     printf to the actual flagged pixel block (not a single guessed pixel)
 *     revealed the real, second half of the bug: even when the fallback
 *     DOES fire and find a valid interval, `__closesthit__cylinder`'s own
 *     radial-normal computation (`obj_hit.x/obj_hit.y` normalized away from
 *     the axis) is only geometrically valid for a hit ON the lateral wall -
 *     for an open-end entry, the hit point is INSIDE the cross-section, not
 *     on its boundary, making the derived normal (and therefore
 *     `front_face`) meaningless. Confirmed via printf that `front_face`
 *     then evaluates to `true` often enough to route the hit into
 *     MaterialType::DielectricMedium's "fresh surface entry, refract/
 *     reflect" branch instead of "already inside, sample the medium" -
 *     silently skipping the fog scattering entirely and passing the
 *     near-invisible (eta=1.001) ray straight through to the dark
 *     background, exactly matching "GPU shows nothing" for these blocks.
 *     Fixed by tagging the fallback's `optixReportIntersection()` call with
 *     a `kOpenEndVolumeHit` hit-kind sentinel (both backends) and checking
 *     it in the closest-hit program to force `front_face=false` for that
 *     hit kind - the DielectricMedium "already inside" branch already
 *     independently recomputes its own entry/exit interval from the
 *     current ray on every call, so this one targeted override is
 *     sufficient; no other downstream change needed. Verified: this
 *     scene's own regional-diff check now passes.
 *   - B11 (Hair Fibers) - RESOLVED, not a code bug, same pattern as B14:
 *     CPU vs both GPU backends differed by up to 61% in 1-2 blocks. The
 *     scene (pbrt_scenes/hair-fibers-scene.pbrt) is 5 plain spheres shaded
 *     via HairBxDF using the shading normal as a fiber-tangent proxy, NOT
 *     literal curve geometry - "thin, high-frequency geometry" was the
 *     wrong original framing. Line-by-line code review found all three
 *     backends call the exact same shared src/shared/bxdfs_hair.h
 *     HairBxDF<T> template with the identical 5-value RNG draw order
 *     (h, u1-u4) and identical is_specular=true/skip_pdf=true (no NEE)
 *     dispatch - no algorithmic divergence anywhere. Empirically confirmed
 *     as the same firefly-variance pattern as B14: re-rendering CPU alone
 *     5x with different seeds at this suite's own settings showed one
 *     block swinging by up to 40.8% with no GPU involved at all (already
 *     more than half the worst observed CPU-vs-GPU gap), and GPU-recursive
 *     vs GPU-wavefront PASS this check against each other while both
 *     differ from CPU similarly - the identical "GPU's two backends
 *     resemble each other more than CPU's independent noise" signature.
 *     Given its own regional-tolerance exception (kHairFibersRegional
 *     RelTolerance, 70%) below, same treatment as B14.
 *   - J2 (DiffuseTransmission Texture, pbrt example) - PARTIALLY RESOLVED,
 *     a real bug found and fixed alongside already-documented noise: CPU
 *     vs GPU-recursive originally differed by ~70% in 3 of 8 comparable
 *     blocks. The scene (pbrt_scenes/diffusetransmission-texture.pbrt)
 *     binds two deliberately tiny 4x4-pixel textures (uv-checker.bmp,
 *     gonio-profile.bmp) to a DiffuseTransmission material's reflectance/
 *     transmittance - exactly the regime where point-sampling vs. filtering
 *     a texture diverges most (every screen pixel can land on a different
 *     one of only 16 texels). Code review found GPU's Image-texture
 *     sampling (sample_texture()/wf_sample_texture(), optix_device_
 *     helpers.h/wavefront_device_helpers.h) was pure nearest-neighbor,
 *     while CPU's mipmap_texture::value() (texture.h) has ALWAYS been
 *     bilinear-filtered even with zero screen-space derivatives (MIPMap::
 *     filter() degrades to LOD-0 bilerp(), mipmap.h) - a real, confirmed,
 *     non-noise discrepancy, NOT the NEE-strategy difference this entry
 *     originally (and wrongly) attributed the whole gap to. Fixed by
 *     making both GPU backends do real bilinear interpolation, matching
 *     CPU's bilerp() exactly (texel centers at (i+0.5)/width, 4-tap blend,
 *     same per-corner wrap-mode handling) - this is a real quality fix for
 *     EVERY GPU material that reads an Image texture, not just J2. This
 *     measurably helped (3/8 -> 2/8 blocks) but did NOT fully close the
 *     gap (worst block still ~70%), because this material SEPARATELY also
 *     has the already-documented MaterialType::DiffuseTransmission
 *     algorithmic gap above (neither GPU backend does CPU's correct
 *     two-hemisphere NEE) - that unbiased-but-noisier-estimator difference
 *     appears to dominate the remaining gap. Given its own regional-
 *     tolerance exception (kDiffuseTransmissionTextureRegionalRelTolerance,
 *     85%) reflecting the real, accepted, residual algorithmic difference -
 *     unlike B14/B11's exceptions, this one is NOT "it's just noise," it's
 *     "a real bug got fixed; what's left is a different, already-accepted,
 *     real gap." GPU's Image-texture sampling remains architecturally
 *     single-resolution-level (no mip pyramid, optix_types.h's TextureData)
 *     - full EWA/mipmap MINIFICATION filtering (as opposed to the
 *     MAGNIFICATION case bilinear alone fixes) is a real, larger,
 *     still-open gap for a different scenario (a texture far smaller on
 *     screen than its own resolution), not reached by this scene.
 *   - E10 (Camera Medium, pbrt example) - a LATER regional-check finding
 *     (found on a full-suite run after the 4 above were already resolved
 *     and this header's own count was written), NOT a new bug: 13/36
 *     blocks exceeded the generic regional_tolerance_for(0.85)=0.95 cap,
 *     worst block differing by exactly 100%, on CPU-vs-wavefront and
 *     recursive-vs-wavefront, while CPU-vs-recursive passed outright. This
 *     is the SAME already-documented, already-accepted gap
 *     kCameraMediumRelTolerance's own comment describes (GPU-wavefront
 *     does not implement pbrt-v4's camera-medium idiom AT ALL) - a block
 *     dominated by "background behind fog" vs. the same background fully
 *     exposed can approach mp_regional_diff()'s own mathematical ceiling of
 *     100%, which the generic 0.95-capped scaling doesn't quite cover.
 *     Given its own regional-tolerance exception
 *     (kCameraMediumRegionalRelTolerance, deliberately set to the exact
 *     1.0 ceiling - see that constant's own comment for why this is the
 *     honest value, not a tuned one) rather than a wider generic cap.
 *
 * GPU-recursive vs GPU-wavefront: separately calibrated (2026-10-02), no
 * longer reusing the CPU-pair tolerances. A full 40-scene MaterialsAndVolumes
 * sweep with this pair's own tolerance temporarily forced to ~0 (so every
 * EXPECT_LT failure printed its real measured gap) showed this pair agrees
 * MUCH more tightly than either GPU backend agrees with CPU - expected,
 * since both are independent ports of the same material math, while the
 * CPU-pair tolerances already absorb real, documented algorithmic
 * differences (NEE strategy, rough_metal's missing Fresnel model, ...).
 * Whole-image gaps were almost all under 5%; regional gaps ran higher, as
 * expected for a per-block comparison with fewer pixels/more residual
 * noise per block. Two real outliers emerged, each given its own new
 * rec-vs-wf-specific exception rather than being folded into the new
 * tighter standard (kRecWfRelTolerance/kRecWfRegionalRelTolerance - see
 * their own comments for the full numbers):
 *   - E1 (Homogeneous Medium): the only scene whose gap didn't fit the new
 *     standard on EITHER axis (24.16% whole-image, 52.70% regional) - not
 *     investigated further here (real follow-up work, not a calibration
 *     task), given its own named exception
 *     (kHomogeneousMediumRecWfRelTolerance/
 *     kHomogeneousMediumRecWfRegionalRelTolerance) rather than silently
 *     widening the standard for every Volumes scene.
 *   - B13 (Subsurface Slab): whole-image gap (15.74%) fits the new
 *     standard; regional (59.06%) does not - same shape as its own
 *     existing CPU-pair exception (kSubsurfaceSlabRelTolerance) having a
 *     real, specific, already-documented cause. Given its own regional-only
 *     exception (kSubsurfaceSlabRecWfRegionalRelTolerance).
 * E10/B11/B14/J2 keep using their EXISTING named exceptions for this pair
 * too (no new constant needed) - each was independently measured in this
 * same sweep to already comfortably cover its rec-vs-wf-specific gap
 * (e.g. B11's 50.14% regional fits inside its existing 70% exception).
 *
 * A third exception was added on VERIFICATION, not the original sweep:
 * B23 (Glass Prism Dispersion) measured 31.04% regional in the calibration
 * run but 42.28% on a later full-suite run, exceeding the standard
 * kRecWfRegionalRelTolerance (42%, sized from the single calibration
 * measurement). Not noise re-landing slightly differently - a real,
 * already-documented algorithmic difference for THIS scene specifically:
 * GPU-recursive approximates dispersion with 3 representative wavelengths,
 * GPU-wavefront does real continuous spectral integration, so which
 * wavelengths each backend's own stochastic sampling happens to land on
 * can shift a regional block's color more than ordinary same-algorithm
 * noise would. B24 (same material, roughness-blurred) shares the cause and
 * gets the same exception (kDispersivePrismRecWfRegionalRelTolerance, 55%)
 * even though its own measured value (5.66%) was far lower, rather than
 * relying on it staying small by chance.
 *
 * Known, deliberate backend behavior differences considered and NOT
 * special-cased here (each was checked against current code, not just old
 * comments/notes - see below):
 *
 *   - MaterialType::Subsurface (B13, SubsurfaceSlab): an earlier note in
 *     this project's memory claimed the wavefront backend has NO real
 *     implementation and falls back to flat diffuse. That is NOW STALE.
 *     gpu/optix/wavefront_kernels.cu's MaterialType::Subsurface case
 *     (~line 1410) hands transmitted rays off to a real BSSRDF probe-walk
 *     stage (BssrdfProbeWorkItem -> wavefront_probe.h's wf_bssrdf_probe_walk()
 *     -> resolve_bssrdf_exit(), ~line 2189) that is a verbatim port of the
 *     GPU-recursive backend's own bssrdf_probe_walk()/shade_material()
 *     Subsurface handling (optix_device_helpers.h) - same tabulated
 *     BSSRDFTable data, same 3-axis-MIS probe/exit algorithm, same
 *     NormalizedFresnel exit hand-off. This was a genuine Phase 2 addition
 *     that landed after gpu/optix/optix_types.h's MaterialType::Subsurface
 *     comment block (~line 275) and pbrt_gpu_builder.h's matching comment
 *     were written - those two comment blocks still describe the old
 *     Phase-1-only ("recursive backend only... wavefront backend has NO
 *     real implementation... flat-diffuse fallback") state and are
 *     themselves now stale/misleading, but the actual behavior is at
 *     parity across all three backends. B13's real, confirmed finding above
 *     is therefore a genuine CPU-vs-GPU numerical discrepancy, not this
 *     (stale, no-longer-applicable) fallback.
 *
 *   - MaterialType::Medium / CloudMedium / RgbGridMedium / GridMedium /
 *     interior dielectric-medium scattering (relevant to E1-E4 and the
 *     dielectric-medium showcase): STALE, SUPERSEDED - this used to say both
 *     GPU backends set is_specular=true for every volume-scattering event
 *     (no NEE/MIS at all, naive-vs-NEE unbiased-but-noisier estimator, not a
 *     bias). Both GPU backends now do real Henyey-Greenstein-phase NEE+MIS
 *     at these events (medium_phase_nee_mis(), gpu/optix/optix_device_
 *     helpers.h; the equivalent isPhase-gated path in wavefront_kernels.cu's
 *     wf_finish_material_scatter()), matching CPU's own hg_phase_material
 *     (constant_medium.h, skip_pdf=false) - see kVolumeRelTolerance's own
 *     comment for why the 55% ceiling below is now very likely looser than
 *     necessary but hasn't been re-measured yet.
 *
 *   - MaterialType::RgbGridMedium specifically also uses a single global
 *     delta-tracking majorant on GPU vs CPU's real per-voxel DDA majorant
 *     grid (optix_intersection_sphere.h ~line 367) - again a deliberate
 *     efficiency simplification (more null-collision steps, same unbiased
 *     estimator), not a source of systematic bias.
 *
 *   - MaterialType::DiffuseTransmission (B8, CornellWaxSlab): neither GPU
 *     backend does correct two-hemisphere NEE the way CPU's material_pbrt.h
 *     diffuse_transmission does (wavefront_kernels.cu ~line 1778 disables
 *     NEE outright with a comment explaining why; the recursive backend
 *     implicitly runs the generic single-hemisphere NEE block instead).
 *     Same reasoning as the medium case above: an unbiased-but-noisier GPU
 *     estimator, not a bias, so no exclusion - just something to watch if
 *     B8 shows a larger gap than its neighbors in a real run.
 *
 * No other MaterialType showed a genuine behavioral divergence between
 * backends as of this writing (Measured, Hair, Principled, CoatedConductor,
 * NormalMappedLambertian, etc. all carry comments describing intentional
 * cross-backend parity, not simplification).
 *
 * ============================================================================
 * Summary of what this suite's own calibration surfaced that is NOT fixed
 * here (out of scope per this task - reported for a human to triage):
 *
 *   1. [SUPERSEDED - see kSubsurfaceSlabRelTolerance's comment below] B13
 *      (SubsurfaceSlab) used to fail BrightnessAndChannelsConsistentAcrossBackends
 *      at the standard 30% tolerance, verified in single-scene process
 *      isolation: CPU was consistently ~32-38% brighter than BOTH
 *      GPU-recursive and GPU-wavefront, across avg brightness and every
 *      channel. See the tolerance-calibration note above for why this reads
 *      as a real discrepancy rather than noise or cross-scene corruption.
 *      That gap has since been fixed; B13 now gets a dedicated, narrower
 *      tolerance for a new, differently-signed (CPU darker) gap instead -
 *      this paragraph is kept as history, not the current state.
 *
 *   2. B1 (RoughMetalSpheres) fails the same test on its B channel alone,
 *      CPU vs GPU-wavefront only (GPU-recursive matches CPU fine), also
 *      verified in isolation: the gap sits right at the 30% edge across
 *      repeated isolated runs (30.03-30.10%). Much smaller and less certain
 *      than B13, but consistently just past the line rather than drifting
 *      either side of it, so left failing deliberately rather than
 *      tolerance-widened away.
 *
 *   3. A GPU cross-scene state corruption bug independent of anything
 *      measured above: rendering enough different scenes back-to-back on
 *      the GPU in one process (recursive-mode-only reproduces it too - this
 *      is NOT specific to a wavefront/recursive backend-mode transition, an
 *      earlier hypothesis this file's own isolation testing ruled out)
 *      eventually crashes with "CUDA error: an illegal memory access was
 *      encountered" (CUDA error 700), which then corrupts the OptiX/CUDA
 *      context for the rest of the process (every OptiX call after it fails
 *      too - optixDeviceContextCreate itself starts failing). See "GPU
 *      cross-scene state corruption" above for the full isolation-testing
 *      repro (single-scene renders never crash; a multi-scene GPU-recursive-
 *      only pass crashed on its 13th scene in one observed run) and its
 *      likely relationship to this codebase's prior scene-switch GPU bug
 *      fixes. Not fixed here per this task's scope (new test file only, no
 *      rendering-code changes) - flagged for human follow-up.
 * ============================================================================
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#include "scene_registry.h"

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// Shared utilities (deliberately re-implemented rather than shared with
// cpu_gpu_comparison_tests.cpp's identically-shaped Image/load_image/
// avg_brightness/avg_channels: those are free functions with external
// linkage in that .cpp, and this file links into the same test binary, so
// reusing those exact names without `static` here would be a duplicate-
// symbol link error. `static` gives these internal linkage instead - same
// names/shapes as requested, no collision.)
// ============================================================================

namespace {

struct MPImage {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB normalized [0,1]
	bool valid = false;
};

static MPImage mp_load_image(const char* path) {
	MPImage img;
	std::ifstream f(path);
	if (!f.good()) return img;
	std::string magic;
	int maxVal;
	f >> magic >> img.width >> img.height >> maxVal;
	if (magic != "P3" || maxVal <= 0) return img;
	int total = img.width * img.height * 3;
	img.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v; f >> v;
		img.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	img.valid = true;
	return img;
}

static float mp_avg_brightness(const MPImage& img) {
	if (img.pixels.empty()) return 0.0f;
	float s = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return s / static_cast<float>(img.pixels.size());
}

struct MPRGBAverage { float r, g, b; };

static MPRGBAverage mp_avg_channels(const MPImage& img) {
	MPRGBAverage out{0, 0, 0};
	int n = img.width * img.height;
	if (n == 0) return out;
	for (int i = 0; i < n; ++i) {
		out.r += img.pixels[i * 3 + 0];
		out.g += img.pixels[i * 3 + 1];
		out.b += img.pixels[i * 3 + 2];
	}
	out.r /= n; out.g /= n; out.b /= n;
	return out;
}

// ============================================================================
// Regional (block-based) diff - catches what whole-image averages can't.
//
// Every check above this point reduces an entire rendered image to 1-4
// floats (one overall brightness average, three per-channel averages). That
// is enough to catch a GROSS, image-wide shift (the HDRI-sky flat-color-
// fallback bug this file's header comment describes, a 35-58% whole-image
// gap) but structurally CANNOT catch a divergence that is spatially
// localized rather than global: a camera-framing/FOV bug that shifts
// geometry sideways, a shading bug confined to part of one object's
// surface, a missing shadow, or a color-channel swap on a scene where R and
// B happen to average close to each other - each of these can leave the
// whole-image brightness and per-channel totals comfortably inside
// kRelTolerance while large parts of the actual image are visibly wrong,
// diluted into invisibility by every other unaffected pixel in the frame.
//
// mp_regional_diff splits each image into a gridSize x gridSize grid of
// blocks, averages color per block, and reports the worst (max) per-block
// relative difference plus how many blocks exceed the given threshold.
// Each block still averages many pixels (a 60x60 render on a 6x6 grid is
// 100 pixels/block), so this stays robust to ordinary per-pixel Monte-Carlo
// noise between backends - it is NOT a per-pixel/bit-exact comparison
// (scripts/compare_images.py already does that, and is unsuitable for CPU-
// vs-GPU for exactly this reason: different sampling algorithms never
// produce bit-identical pixels). It's the middle ground between "one float
// for the whole image" and "exact pixel match": coarse enough to tolerate
// noise, fine-grained enough to actually localize a divergence.
// ============================================================================

struct MPRegionalDiffResult {
	float maxBlockRelDiff = 0.0f;
	int worstBlockX = -1, worstBlockY = -1;
	int blocksOverThreshold = 0;
	int comparableBlocks = 0;
};

static MPRegionalDiffResult mp_regional_diff(const MPImage& a, const MPImage& b,
                                              int gridSize, float minComparable, float threshold) {
	MPRegionalDiffResult result;
	if (!a.valid || !b.valid || a.width != b.width || a.height != b.height ||
	    a.width <= 0 || a.height <= 0) {
		return result;
	}
	const int blockW = std::max(1, a.width / gridSize);
	const int blockH = std::max(1, a.height / gridSize);
	for (int by = 0; by < gridSize; ++by) {
		const int y0 = by * blockH;
		const int y1 = (by == gridSize - 1) ? a.height : std::min(a.height, y0 + blockH);
		if (y0 >= a.height) continue;
		for (int bx = 0; bx < gridSize; ++bx) {
			const int x0 = bx * blockW;
			const int x1 = (bx == gridSize - 1) ? a.width : std::min(a.width, x0 + blockW);
			if (x0 >= a.width) continue;

			float sumA = 0.0f, sumB = 0.0f;
			int n = 0;
			for (int y = y0; y < y1; ++y) {
				for (int x = x0; x < x1; ++x) {
					const int idx = (y * a.width + x) * 3;
					sumA += a.pixels[idx] + a.pixels[idx + 1] + a.pixels[idx + 2];
					sumB += b.pixels[idx] + b.pixels[idx + 1] + b.pixels[idx + 2];
					n += 3;
				}
			}
			if (n == 0) continue;
			const float avgA = sumA / n, avgB = sumB / n;
			if (avgA < minComparable && avgB < minComparable) continue;

			++result.comparableBlocks;
			const float maxV = std::max(avgA, avgB);
			const float relDiff = std::abs(avgA - avgB) / maxV;
			if (relDiff > threshold) ++result.blocksOverThreshold;
			if (relDiff > result.maxBlockRelDiff) {
				result.maxBlockRelDiff = relDiff;
				result.worstBlockX = bx;
				result.worstBlockY = by;
			}
		}
	}
	return result;
}

// 6x6 grid on a 60x60 render = 100 pixels/block, the same resolution this
// whole file already renders at (kWidth/kHeight below) - chosen so each
// block still has enough samples to average out Monte-Carlo noise rather
// than chasing single-pixel fireflies.
constexpr int kRegionalGridSize = 6;

// Per-block tolerance is deliberately looser than kRelTolerance: a block
// covers ~1/36th of the pixels a whole-image average does, so it carries
// more residual Monte-Carlo variance for the same SPP, and is also more
// exposed to ordinary scene content (a block straddling a hard shadow edge,
// or sitting right on a specular highlight, legitimately differs more
// between two independently-dithered sample sets than the whole-image
// average ever would). This is the STANDARD value, for scenes using the
// standard kRelTolerance - calibrated the same way as this file's other
// tolerances, against this codebase's real renderer, not guessed.
constexpr float kRegionalRelTolerance = 0.50f;

// Relative-difference tolerance shared by both the overall-brightness and
// per-channel checks - see this file's header comment for the calibration
// reasoning (tighter than cpu_gpu_comparison_tests.cpp's 50%, with margin
// below the 35-58% gap the real HDRI-sky bug produced).
constexpr float kRelTolerance = 0.30f;

// Scenes with their own wider whole-image tolerance (E10, Volumes, B13, B1 -
// see those constants' own comments for why) need a correspondingly wider
// regional tolerance too, or this new per-block check would just re-flag
// those same already-documented, already-accepted gaps as "new" failures -
// pure noise, not new information. Scaled by the same ratio the standard
// tolerances use (kRegionalRelTolerance/kRelTolerance), capped below 1.0 so
// the check stays meaningful even for E10's wide 85% whole-image ceiling.
static float regional_tolerance_for(float wholeImageTolerance) {
	constexpr float kMultiplier = kRegionalRelTolerance / kRelTolerance;  // ~1.667
	return std::min(wholeImageTolerance * kMultiplier, 0.95f);
}

// Below this brightness, both values are close enough to black that a
// relative-difference comparison is meaningless (dividing by near-zero
// blows the ratio up on pure noise) - skip the comparison entirely in that
// case, same reasoning as cpu_gpu_comparison_tests.cpp's own
// `if (total < 0.001f) GTEST_SKIP()` pattern in its color-variation tests,
// just applied per-comparison instead of skipping the whole test.
constexpr float kMinComparableValue = 0.004f;

// Volumes scenes (E1-E4) get a wider tolerance than kRelTolerance. The 46.8%
// (B channel, CPU vs GPU-wavefront) gap this 0.55 ceiling was originally
// calibrated against was fully explained by a confirmed, deliberate backend
// difference: neither GPU backend did real NEE/MIS for an in-medium
// Henyey-Greenstein phase-function scatter event (MaterialType::Medium/
// CloudMedium/RgbGridMedium/GridMedium all treated it as specular) - only
// lucky HG-sampled random walks that happened to escape the medium and hit a
// light picked up any illumination at all, converging far slower than a
// surface BSDF would.
// STALE, NEEDS RE-CALIBRATION: that gap was since closed - both GPU backends
// now do real NEE+MIS at these events (medium_phase_nee_mis(),
// gpu/optix/optix_device_helpers.h; the equivalent isPhase-gated path in
// wavefront_kernels.cu's wf_finish_material_scatter()) - and a real render
// comparison at the time of that fix showed E1 dramatically brighter/better-
// converged on GPU at the same SPP than before, the expected NEE signature.
// This 0.55 ceiling is very likely now much looser than the real post-fix
// gap requires, but this file could not be rebuilt/re-run in the environment
// that landed the fix (tests/ray_tracer_tests.vcxproj has a pre-existing,
// unrelated broken gtest include path - confirmed unrelated: the same error
// occurs on ~40 test files untouched by that change) to measure a real
// replacement number, so the tolerance itself was deliberately left
// unchanged (safe - a looser-than-necessary ceiling can't cause a false
// failure, only reduced sensitivity) rather than guessed at. Whoever next
// gets a working build of this test should re-run E1-E4 in isolation (same
// per-scene-isolated methodology as the B1/B13 findings below) and tighten
// this back down to whatever the real post-fix gap turns out to be.
constexpr float kVolumeRelTolerance = 0.55f;

// B13 (SubsurfaceSlab) specifically - NOT a Volumes-category scene, but its
// CPU implementation (scenes_advanced.h's build_subsurface_slab) approximates
// subsurface scattering with layered constant_medium fog rather than a real
// BSSRDF, unlike GPU's tabulated-profile Subsurface implementation. Real
// shadow-ray transmittance through participating media (constant_medium.h's
// shadow_transmittance_impl(), added after this file's original calibration)
// now legitimately attenuates CPU's light contribution through that internal
// fog - GPU's real BSSRDF path has no equivalent medium to attenuate through,
// so it doesn't dim the same way. Verified this is the actual cause, not a
// reversion of the OLD (since-fixed) B13 gap this file's header/summary
// comments describe: re-running this exact test against the code as it
// stood immediately before that shadow-transmittance change showed B13
// PASSING at the standard kRelTolerance (30%) - so those header/summary
// sections describing B13 as "left failing intentionally" are now stale
// history from an earlier, different (and since-fixed) gap, not a
// description of the current failure. This is a new, different, understood
// gap: CPU measured ~32% dimmer than both GPU backends on G/B channels
// specifically (not brighter, like the old stale-documented gap), so 35%
// gives real margin without masking a regression of a different size.
constexpr float kSubsurfaceSlabRelTolerance = 0.35f;

// B1 (RoughMetalSpheres) - this file's own header/summary comments already
// document this scene's CPU-vs-GPU-wavefront B-channel gap as sitting
// "right at the 30% edge (30.08%/30.03%/30.10% across repeated isolated
// runs)" BEFORE the change below, i.e. already known-borderline, not a new
// finding. camera.h's default PixelFilter reconstruction radius (a scene
// with no explicit PixelFilter directive, this one included) was changed
// from a previously-hardcoded 0.5px to pbrt-v4's real Gaussian default of
// 1.5px (real cross-pixel filter importance sampling now reaches
// neighboring pixels - see FilterSampler's own comment, filter_sampler.h) -
// CPU's own reconstruction got MORE correct, but GPU-wavefront's own
// "Box filter (1 sample per pixel sub-region, averaged in kernel)" (see its
// own [TECH] log line) did not change, widening this ALREADY-borderline
// gap just past the standard 30% line on this scene's bright specular
// highlights specifically (avg brightness CPU=0.574 vs GPU-wavefront=0.401,
// 30.0%; B channel 31.8%, in one isolated single-scene run - some further
// run-to-run variance is expected, same as the pre-existing 30.03-30.10%
// spread this file's own header comment already documents for the OLD,
// narrower-radius gap). A real, understood, expected-direction shift (CPU
// brighter, matching a wider real reconstruction filter integrating more
// of each highlight's own falloff), not a new backend bug - 34% gives a
// couple of points of real margin over the single worst measured run
// without approaching B13's own, differently-caused, larger 35% gap.
constexpr float kRoughMetalSpheresRelTolerance = 0.34f;

// E10 (Camera Medium pbrt example) - a genuinely different-magnitude, fully
// understood gap: GPU-wavefront does not implement pbrt-v4's camera-medium
// idiom AT ALL (see gpu/optix/wavefront_kernels.cu's own runtime warning,
// emitted verbatim when this scene renders under --wavefront: "scene has a
// camera medium ... which is not supported under --wavefront - the scene
// will render without it"), unlike GPU-recursive, which does implement it
// (see this scene's own registry description, scene_registry_data.h's E10
// entry). CPU and GPU-recursive both show the ambient fog;
// GPU-wavefront silently omits it entirely, so its render is a materially
// different (unfogged) image, not sampling noise around the same result -
// measured ~79-82% relative difference across all channels in isolated
// runs. 85% gives a couple of points of real margin over the worst measured
// run without masking an actual regression on the CPU-vs-GPU-recursive pair
// (which passes comfortably within the standard Volumes tolerance already).
constexpr float kCameraMediumRelTolerance = 0.85f;

// B14 (Measured BRDF) - a REGIONAL-check-only exception (its whole-image
// brightness/channel checks above pass comfortably at the standard 30% -
// this material's own tolerance stays kRelTolerance, deliberately NOT
// listed in the ternary chain below that picks `tolerance`). See this
// file's header comment's B14 entry for the full investigation: this
// material has a narrow, NEE-less specular lobe (is_specular=true,
// confirmed identical on all three backends) reflected toward a small
// area light, which produces real, substantial same-backend firefly
// variance at this suite's own 200spp/60x60 settings - re-rendering CPU
// alone 5x with different seeds showed this scene's own worst block
// swinging by up to 24.6% with NO GPU involved at all. 65% gives real
// margin over the worst isolated CPU-vs-GPU measurement (~60%) without
// masking a materially larger future regression; it does NOT invalidate
// the regional check generally - every OTHER scene still uses the
// standard kRegionalRelTolerance via regional_tolerance_for(tolerance).
constexpr float kMeasuredBrdfRegionalRelTolerance = 0.65f;

// B11 (Hair Fibers) - same pattern as B14 above, a REGIONAL-check-only
// exception (whole-image checks pass comfortably at the standard 30%).
// HairBxDF is dispatched is_specular=true/skip_pdf=true (no NEE) identically
// on all three backends (verified: material_pbrt.h's `hair_material::
// scatter()`, optix_device_helpers.h's sample_hair_material(), wavefront_
// device_helpers.h's wf_sample_hair_material() all call the SAME shared
// src/shared/bxdfs_hair.h HairBxDF<T> template with the identical 5-value
// RNG draw order - no algorithmic divergence found). The scene (pbrt_
// scenes/hair-fibers-scene.pbrt: 5 spheres shaded with HairBxDF via a
// shading-normal-as-fiber-tangent proxy, lit by one small overhead area
// light) has the same narrow-lobe-plus-no-NEE-plus-small-light setup that
// produces real firefly variance for B14 - re-rendering CPU alone 5x with
// different seeds at this suite's own 200spp/60x60 settings showed one
// block swinging by up to 40.8%, already exceeding half the worst observed
// CPU-vs-GPU gap (61.1%) with zero GPU involved; GPU-recursive vs
// GPU-wavefront also PASS this check against each other (same "GPU's two
// backends resemble each other more than either resembles CPU's
// independent noise" signature as B14). 70% gives real margin over the
// worst isolated measurement without masking a materially larger future
// regression.
constexpr float kHairFibersRegionalRelTolerance = 0.70f;

// J2 (DiffuseTransmission Texture, pbrt example) - a REGIONAL-check-only
// exception, but UNLIKE B14/B11 above this one is NOT pure noise: a real,
// separate bug was found and fixed (GPU's Image-texture sampling was pure
// nearest-neighbor, while CPU's mipmap_texture::value() has always been
// bilinear, even with zero screen-space derivatives - see sample_texture()'s
// own comment, optix_device_helpers.h, and wf_sample_texture()'s,
// wavefront_device_helpers.h, for the fix). That fix measurably helped
// (3/8 -> 2/8 blocks over this threshold) but did not fully close the gap,
// because this material ALSO has the separately-documented, ACCEPTED
// MaterialType::DiffuseTransmission algorithmic difference above (neither
// GPU backend does CPU's correct two-hemisphere NEE) - an unbiased-but-
// noisier estimator difference, not a bug, that this scene's own regional
// block (worst observed ~70%) is apparently still dominated by even after
// the real texture-filtering bug is fixed. 85% (matching
// kCameraMediumRelTolerance's own precedent for "a real, accepted,
// non-noise algorithmic gap") gives margin over the worst observed
// post-fix measurement without masking a future regression beyond what
// the known NEE-strategy difference already explains.
constexpr float kDiffuseTransmissionTextureRegionalRelTolerance = 0.85f;

// E10 (Camera Medium pbrt example) - a REGIONAL-check-only exception on top
// of its own already-wide whole-image kCameraMediumRelTolerance=0.85 above.
// Investigated (full suite run, 2026-10-01): 13/36 blocks exceeded the
// generic regional_tolerance_for(0.85)=min(0.85*1.667, 0.95)=0.95 cap,
// worst block differing by exactly 100%, on BOTH the CPU-vs-wavefront AND
// recursive-vs-wavefront pairs - CPU-vs-recursive passes outright (no
// failure reported for that pair at all), confirming GPU-recursive tracks
// CPU correctly and only GPU-wavefront is the outlier, exactly as
// kCameraMediumRelTolerance's own comment already documents: GPU-wavefront
// does not implement pbrt-v4's camera-medium idiom AT ALL (its own runtime
// warning says so verbatim), so its render of this scene is a materially
// different, completely unfogged image, not sampling noise. A block whose
// content is dominated by "background behind fog" vs. the same background
// fully exposed can genuinely approach mp_regional_diff()'s own
// mathematical maximum (relDiff = |a-b|/max(a,b), which is bounded to
// [0,1] since both operands are non-negative pixel averages) - this is not
// a calibration gap to narrow with a tighter measured value the way
// B14/B11/J2 did, it is the direct, expected, already-accepted consequence
// of a fully-omitted feature. 1.0 is deliberately the exact mathematical
// ceiling: this check cannot mathematically fail for E10 (relDiff can
// equal but never exceed 1.0), which is the honest reflection of "GPU-
// wavefront support is deferred" (this scene's own registry description)
// rather than a tolerance tuned to just barely pass today's measurement.
constexpr float kCameraMediumRegionalRelTolerance = 1.0f;

// ============================================================================
// GPU-recursive vs GPU-wavefront - SEPARATELY calibrated tolerances.
//
// Every tolerance above was calibrated for a CPU-vs-GPU pair and, until now,
// reused unchanged for the GPU-recursive-vs-GPU-wavefront comparison too
// (this file's own prior comment called this "deliberately conservative
// pending real measured rec-vs-wf gap data"). That data now exists: a full
// MaterialsAndVolumes sweep (40 scenes, 2026-10-02) with this pair's own
// tolerance temporarily forced to ~0 to force every EXPECT_LT failure
// message to print its real measured gap - see this section's own constants
// below for what that run showed.
//
// The headline finding: two independent GPU ports of nominally the same
// material math agree MUCH more tightly with each other than either does
// with CPU (whose gaps already have documented algorithmic causes baked
// into the tolerances above - different NEE strategies, rough_metal's
// missing Fresnel model, etc.). Whole-image gaps were almost all under 5%
// (most under 2%); regional (per-block) gaps run higher, as expected -
// fewer pixels per block means more residual Monte-Carlo variance for the
// same spp, the same reason kRegionalRelTolerance is already looser than
// kRelTolerance for the CPU pairs.
//
// Standard values below give real margin over every scene's measured gap
// EXCEPT the ones with their own named exception just below (E1, B13, and
// E10/B11/B14/J2, which already had an exception for other pairs that
// turns out to already cover this pair's own measured gap too - see each
// one's own comment for why no NEW constant was needed there).
// 25%: the true worst non-E1/non-E10 whole-image gap was E11 (Thin
// Dielectric Medium, pbrt example) at 22.39% (R channel) - a first pass at
// this constant (20%) missed that this scene's R channel, not just its
// average brightness (11.14%), was the real outlier, and failed on
// verification. 25% gives real margin over E11 (+2.6pt) and E12 (Rough
// Dielectric Medium, 20.60% R channel, +4.4pt) without masking a future
// regression beyond what these two already-measured gaps explain.
constexpr float kRecWfRelTolerance = 0.25f;
// 42%: the worst non-excepted regional gap was E7 (pbrt example) at
// 39.23%, with B12/B23/E12/E11/E4/J1/B7/E2/E6 all in the high-20s/low-30s -
// regional gaps run higher than whole-image ones across the board here, the
// same reason kRegionalRelTolerance is already looser than kRelTolerance
// for the CPU pairs (fewer pixels per block, more residual Monte-Carlo
// variance for the same spp). 42% gives real margin over E7 (+2.8pt).
constexpr float kRecWfRegionalRelTolerance = 0.42f;

// E1 (Homogeneous Medium) - the one scene whose rec-vs-wf gap doesn't fit
// the standard values above on EITHER axis: measured whole-image up to
// 24.16% (B channel) and regional worst-block 52.70%, both real outliers
// among the 40 scenes swept (nothing else in Volumes ran anywhere close -
// e.g. E2 Cloud Medium measured 3.73%/26.73%, E6 Cylinder Medium measured
// 2.72%/24.00%). Not investigated further here (that's real follow-up work,
// not a calibration task) - given real margin rather than silently folded
// into the standard tolerance, so a future regression has room to be
// caught rather than being absorbed into "every Volumes scene gets this
// much slack".
constexpr float kHomogeneousMediumRecWfRelTolerance = 0.30f;
constexpr float kHomogeneousMediumRecWfRegionalRelTolerance = 0.60f;

// B13 (Subsurface Slab) - whole-image rec-vs-wf gap (15.74% max) already
// fits the standard kRecWfRelTolerance above, but its regional worst-block
// (59.06%) does not - same shape as its own existing CPU-pair exception
// (kSubsurfaceSlabRelTolerance) having a real, specific, already-documented
// cause rather than being ordinary noise.
constexpr float kSubsurfaceSlabRecWfRegionalRelTolerance = 0.65f;

// B23/B24 (Glass/Frosted Prism Dispersion) - regional-only, found on
// verification (not the original calibration sweep): B23 measured 31.04%
// in the calibration run but 42.28% on a LATER full-suite run, exceeding
// the standard kRecWfRegionalRelTolerance (42%) set from the first
// measurement alone. This is not ordinary Monte-Carlo noise re-landing
// slightly differently - it's this scene's own already-documented,
// already-accepted algorithmic difference between the two GPU backends
// (see this scene's own registry description/pbrt_flatten.h's Material::
// abbeNumber comment): GPU-recursive approximates dispersion with 3
// representative wavelengths, GPU-wavefront does real continuous spectral
// integration, so WHICH wavelengths each backend's stochastic sampling
// happens to land on for a given run can shift a regional block's color
// noticeably more than ordinary same-algorithm sampling noise would. B24
// shares the identical cause (same material, just roughness-blurred) even
// though its own measured value (5.66%) was far lower - given the same
// exception rather than relying on its own gap staying small by chance.
constexpr float kDispersivePrismRecWfRegionalRelTolerance = 0.55f;

static void check_relative_parity(const char* sceneName, const std::string& sceneId,
                                   const char* label, const char* backendA, const char* backendB,
                                   float a, float b, float tolerance) {
	if (a < kMinComparableValue && b < kMinComparableValue) return;
	float maxV = std::max(a, b);
	float relDiff = std::abs(a - b) / maxV;
	EXPECT_LT(relDiff, tolerance)
		<< sceneName << " (" << sceneId << ") " << label << ": "
		<< backendA << "=" << a << " vs " << backendB << "=" << b
		<< " -- relative difference " << (relDiff * 100.0f) << "% exceeds "
		<< (tolerance * 100.0f) << "% tolerance. This may indicate a real "
		<< "per-backend material bug (c.f. the HDRI-sky GPU flat-color-fallback "
		<< "bug found in this codebase's history) rather than ordinary "
		<< "Monte-Carlo/sampling-strategy noise -- do not silently widen this "
		<< "tolerance to make the failure go away without investigating first.";
}

// See mp_regional_diff's own comment for why this exists alongside
// check_relative_parity rather than instead of it: this catches spatially
// localized divergence (framing/geometry shifts, a shading bug confined to
// part of the frame, a channel swap) the whole-image check structurally
// can't see.
static void check_regional_parity(const char* sceneName, const std::string& sceneId,
                                   const char* backendA, const char* backendB,
                                   const MPImage& a, const MPImage& b, float regionalTolerance) {
	const MPRegionalDiffResult r = mp_regional_diff(a, b, kRegionalGridSize,
	                                                 kMinComparableValue, regionalTolerance);
	if (r.comparableBlocks == 0) return;  // whole image too dark to compare, same as check_relative_parity
	EXPECT_EQ(r.blocksOverThreshold, 0)
		<< sceneName << " (" << sceneId << ") regional diff, " << backendA << " vs " << backendB << ": "
		<< r.blocksOverThreshold << "/" << r.comparableBlocks << " blocks (of a "
		<< kRegionalGridSize << "x" << kRegionalGridSize << " grid) exceed "
		<< (regionalTolerance * 100.0f) << "% relative difference; worst block ("
		<< r.worstBlockX << "," << r.worstBlockY << ") differs by "
		<< (r.maxBlockRelDiff * 100.0f) << "% -- this can indicate a localized divergence "
		<< "(camera framing/geometry shift, a shading bug confined to part of the frame, a "
		<< "color-channel swap) that the whole-image brightness/channel checks above cannot "
		<< "see, since they'd dilute it across every other unaffected pixel. Investigate before "
		<< "widening this scene's regional tolerance.";
}

// Registry positions (NOT scene ids - see CpuGpuLightParityTest's own
// comment in cpu_gpu_comparison_tests.cpp for why registry position is
// used as the TEST_P param) whose category is Materials, Volumes, or
// Textures. Filtering by category here means a future scene added to any
// of the three is automatically picked up without touching this file.
// Textures was split out of Materials (SceneCategories::Textures's own
// comment, scene_descriptor.h) - its scenes still exercise real BSDF/
// texture-binding shading paths and deserve the identical CPU/GPU/wavefront
// parity coverage they had while filed under Materials, so it's included
// here even though the suite name below ("MaterialsAndVolumes", kept
// unchanged for the documented `--gtest_filter=-MaterialsAndVolumes/*`
// dev-loop shortcut in README.md/TESTING_GUIDE.md/copilot-instructions.md)
// no longer names every category it covers.
static std::vector<int> materials_volumes_and_textures_indices() {
	std::vector<int> out;
	const auto& registry = get_scene_registry();
	for (int i = 0; i < static_cast<int>(registry.size()); ++i) {
		const char* cat = registry[i].category;
		if (std::strcmp(cat, SceneCategories::Materials) == 0 ||
		    std::strcmp(cat, SceneCategories::Volumes) == 0 ||
		    std::strcmp(cat, SceneCategories::Textures) == 0) {
			out.push_back(i);
		}
	}
	return out;
}

} // namespace

// ============================================================================
// CPU vs GPU-recursive vs GPU-wavefront per-material parity sweep
//
// IMPORTANT execution-order note (see this file's header comment, "GPU
// cross-scene state corruption", for the full repro and isolation testing):
// rendering enough different scenes back-to-back on the GPU in one process
// eventually corrupts the CUDA/OptiX context (CUDA error 700), taking down
// every OptiX call for the rest of the process. A naive per-scene TEST_P
// body that renders CPU/recursive/wavefront for scene N and then
// CPU/recursive/... for scene N+1 hits this almost immediately (verified: an
// early version of this file structured that way got real results for only
// 2 of 18 scenes before the rest cascaded into corruption). Grouping into
// three whole-suite passes below (all-CPU, then one uninterrupted
// GPU-recursive pass, then one uninterrupted GPU-wavefront pass) delays the
// corruption and makes its failure mode honest (a clear "render failed to
// produce a valid PPM" rather than a false material-bug claim), but does NOT
// eliminate it - a pure GPU-recursive-only pass with no wavefront mode
// involved at all was independently observed to hit the same corruption
// partway through. The real fix is per-scene process isolation, deliberately
// left out of scope here (see the header comment). Do not read a failure
// from a full run of this suite as a material finding by itself - cross-
// check against an isolated single-scene run first (e.g. temporarily filter
// testable_scenes() to one scene id) the way this suite's own B1/B13
// findings were verified.
//
// This cache is a lazily-populated, process-wide function-local static, not
// per-TEST_P-instance state, precisely so it survives across gtest's
// separate TEST_P invocations and still only renders each image once.
// ============================================================================

namespace {

struct SceneRenderCache {
	bool built = false;
	bool gpuAvailable = false;
	std::unordered_map<std::string, MPImage> cpuImages;
	std::unordered_map<std::string, MPImage> recImages;
	std::unordered_map<std::string, MPImage> wfImages;
};

static constexpr int kWidth  = 60;
static constexpr int kHeight = 60;
static constexpr int kDepth  = 8;

// See this file's header comment for why Volumes scenes get bumped SPP on
// both sides (slower-converging participating-media transport, plus neither
// GPU backend does NEE for in-medium scattering - also see
// kVolumeRelTolerance's own comment for why SPP alone isn't pushed high
// enough to close that particular gap without an impractically slow test).
static void spp_for(const SceneDescriptor& s, int& cpuSpp, int& gpuSpp) {
	const bool isVolume = std::strcmp(s.category, SceneCategories::Volumes) == 0;
	cpuSpp = isVolume ? 300 : 200;
	gpuSpp = isVolume ? 900 : 600;
}

// Every Materials/Volumes/Textures scene that would actually be exercised
// by the TEST_P suite below (mirrors its own gpu_compatible/requires_files
// skips) - computed once so the three render passes and the TEST_P bodies
// agree on exactly which scenes are in play.
static std::vector<const SceneDescriptor*> testable_scenes() {
	std::vector<const SceneDescriptor*> out;
	for (int idx : materials_volumes_and_textures_indices()) {
		const SceneDescriptor& regDesc = get_scene_registry()[idx];
		const SceneDescriptor* s = find_scene(regDesc.id);
		if (!s || !s->gpu_compatible || s->requires_files) continue;
		out.push_back(s);
	}
	// Single-scene isolation for calibration/debugging - see this file's
	// header comment ("Single-scene isolation for calibration/debugging")
	// for why this exists: routes around the GPU cross-scene corruption
	// bug documented above by rendering only one scene in the process.
	if (const char* only = std::getenv("MATPARITY_ONLY_SCENE_ID")) {
		std::vector<const SceneDescriptor*> filtered;
		for (const SceneDescriptor* s : out) if (s->id == only) filtered.push_back(s);
		return filtered;
	}
	return out;
}

static MPImage render_cpu_once(const SceneDescriptor& s, int spp) {
	const std::string fn = "matparity_" + s.id + "_cpu.ppm";
	cpu_render_main(kWidth, kHeight, spp, kDepth, fn.c_str(), s.id.c_str(),
	                 s.camera.lookfrom_x, s.camera.lookfrom_y, s.camera.lookfrom_z);
	MPImage img = mp_load_image(fn.c_str());
	std::remove(fn.c_str());
	return img;
}

// Caller is responsible for setting RAY_TRACER_WAVEFRONT before calling this
// for a whole batch of scenes - see build_cache_once()'s two passes below.
static MPImage render_gpu_once(const SceneDescriptor& s, int spp, const char* suffix) {
	const std::string fn = "matparity_" + s.id + "_" + suffix + ".ppm";
	const int rc = optix_render_main(kWidth, kHeight, spp, kDepth, fn.c_str(), s.id.c_str(),
	                                  s.camera.lookfrom_x, s.camera.lookfrom_y, s.camera.lookfrom_z);
	MPImage img;
	if (rc == 0) img = mp_load_image(fn.c_str());
	std::remove(fn.c_str());
	return img;
}

static SceneRenderCache& get_cache() {
	static SceneRenderCache cache;
	return cache;
}

static void build_cache_once() {
	SceneRenderCache& cache = get_cache();
	if (cache.built) return;
	cache.built = true;
	cache.gpuAvailable = optix_is_available();
	if (!cache.gpuAvailable) return;

	const std::vector<const SceneDescriptor*> scenes = testable_scenes();

	// Pass 1: every scene's CPU render. No GPU/OptiX calls in this pass.
	for (const SceneDescriptor* s : scenes) {
		int cpuSpp, gpuSpp;
		spp_for(*s, cpuSpp, gpuSpp);
		cache.cpuImages[s->id] = render_cpu_once(*s, cpuSpp);
	}

	// Save/restore RAY_TRACER_WAVEFRONT around both GPU passes together,
	// same pattern as tests/unit/wavefront_tests.cpp's WavefrontRenderTest
	// fixture, just applied once per pass instead of once per render.
	const char* prevWfRaw = getenv("RAY_TRACER_WAVEFRONT");
	const bool hadPrevWf = prevWfRaw != nullptr;
	const std::string prevWf = hadPrevWf ? prevWfRaw : "";

	// Pass 2: every scene's GPU-recursive render, in one uninterrupted
	// recursive-mode run.
#ifdef _WIN32
	_putenv_s("RAY_TRACER_WAVEFRONT", "0");
#else
	setenv("RAY_TRACER_WAVEFRONT", "0", 1);
#endif
	for (const SceneDescriptor* s : scenes) {
		int cpuSpp, gpuSpp;
		spp_for(*s, cpuSpp, gpuSpp);
		cache.recImages[s->id] = render_gpu_once(*s, gpuSpp, "rec");
	}

	// Pass 3: every scene's GPU-wavefront render, in one uninterrupted
	// wavefront-mode run - the ONLY backend-mode transition in this whole
	// suite is this one (recursive -> wavefront), which is not the crash
	// trigger (see this section's own header comment).
#ifdef _WIN32
	_putenv_s("RAY_TRACER_WAVEFRONT", "1");
#else
	setenv("RAY_TRACER_WAVEFRONT", "1", 1);
#endif
	for (const SceneDescriptor* s : scenes) {
		int cpuSpp, gpuSpp;
		spp_for(*s, cpuSpp, gpuSpp);
		cache.wfImages[s->id] = render_gpu_once(*s, gpuSpp, "wf");
	}

#ifdef _WIN32
	if (hadPrevWf) _putenv_s("RAY_TRACER_WAVEFRONT", prevWf.c_str());
	else           _putenv_s("RAY_TRACER_WAVEFRONT", "");
#else
	if (hadPrevWf) setenv("RAY_TRACER_WAVEFRONT", prevWf.c_str(), 1);
	else           unsetenv("RAY_TRACER_WAVEFRONT");
#endif
}

} // namespace

class MaterialCpuGpuParityTest : public ::testing::TestWithParam<int> {};

TEST_P(MaterialCpuGpuParityTest, BrightnessAndChannelsConsistentAcrossBackends) {
	// GetParam() is a registry POSITION, not a scene id - resolve to a
	// SceneDescriptor via the registry first, same convention as
	// CpuGpuLightParityTest in cpu_gpu_comparison_tests.cpp.
	const SceneDescriptor& regDesc = get_scene_registry()[GetParam()];
	const SceneDescriptor* s = find_scene(regDesc.id);
	ASSERT_NE(s, nullptr) << "Missing scene id " << regDesc.id;

	if (!s->gpu_compatible) GTEST_SKIP() << s->name << " (" << s->id << ") is not GPU-compatible";
	if (s->requires_files)  GTEST_SKIP() << s->name << " (" << s->id << ") requires external assets";

	build_cache_once();
	SceneRenderCache& cache = get_cache();
	if (!cache.gpuAvailable) GTEST_SKIP() << "OptiX not available -- skipping 3-way backend comparison";

	// See kVolumeRelTolerance's own comment for why Volumes scenes need a
	// wider, separately-justified tolerance than everything else. B13
	// (SubsurfaceSlab) and B1 (RoughMetalSpheres) each get their own narrow
	// carve-out for a different, specific, understood reason - see
	// kSubsurfaceSlabRelTolerance's/kRoughMetalSpheresRelTolerance's own
	// comments below.
	const float tolerance =
		(s->id == "E10")                                          ? kCameraMediumRelTolerance :
		(std::strcmp(s->category, SceneCategories::Volumes) == 0) ? kVolumeRelTolerance :
		(s->id == "B13")                                          ? kSubsurfaceSlabRelTolerance :
		(s->id == "B1")                                           ? kRoughMetalSpheresRelTolerance :
		kRelTolerance;

	auto cpuIt = cache.cpuImages.find(s->id);
	auto recIt = cache.recImages.find(s->id);
	auto wfIt  = cache.wfImages.find(s->id);
	ASSERT_NE(cpuIt, cache.cpuImages.end()) << s->name << " (" << s->id << "): missing from CPU render pass";
	ASSERT_NE(recIt, cache.recImages.end()) << s->name << " (" << s->id << "): missing from GPU-recursive render pass";
	ASSERT_NE(wfIt,  cache.wfImages.end())  << s->name << " (" << s->id << "): missing from GPU-wavefront render pass";

	const MPImage& cpuImg = cpuIt->second;
	const MPImage& recImg = recIt->second;
	const MPImage& wfImg  = wfIt->second;

	ASSERT_TRUE(cpuImg.valid) << s->name << " (" << s->id << "): CPU render failed to produce a valid PPM";
	ASSERT_TRUE(recImg.valid) << s->name << " (" << s->id << "): GPU-recursive render failed to produce a valid PPM";
	ASSERT_TRUE(wfImg.valid)  << s->name << " (" << s->id << "): GPU-wavefront render failed to produce a valid PPM";

	const float cpuBright = mp_avg_brightness(cpuImg);
	const float recBright = mp_avg_brightness(recImg);
	const float wfBright  = mp_avg_brightness(wfImg);

	EXPECT_GT(cpuBright, 0.001f) << s->name << " (" << s->id << "): CPU output is black";
	EXPECT_GT(recBright, 0.001f) << s->name << " (" << s->id << "): GPU-recursive output is black";
	EXPECT_GT(wfBright,  0.001f) << s->name << " (" << s->id << "): GPU-wavefront output is black";

	check_relative_parity(s->name, s->id, "avg brightness", "CPU", "GPU-recursive", cpuBright, recBright, tolerance);
	check_relative_parity(s->name, s->id, "avg brightness", "CPU", "GPU-wavefront", cpuBright, wfBright, tolerance);

	const MPRGBAverage cpuC = mp_avg_channels(cpuImg);
	const MPRGBAverage recC = mp_avg_channels(recImg);
	const MPRGBAverage wfC  = mp_avg_channels(wfImg);

	check_relative_parity(s->name, s->id, "R channel", "CPU", "GPU-recursive", cpuC.r, recC.r, tolerance);
	check_relative_parity(s->name, s->id, "G channel", "CPU", "GPU-recursive", cpuC.g, recC.g, tolerance);
	check_relative_parity(s->name, s->id, "B channel", "CPU", "GPU-recursive", cpuC.b, recC.b, tolerance);

	check_relative_parity(s->name, s->id, "R channel", "CPU", "GPU-wavefront", cpuC.r, wfC.r, tolerance);
	check_relative_parity(s->name, s->id, "G channel", "CPU", "GPU-wavefront", cpuC.g, wfC.g, tolerance);
	check_relative_parity(s->name, s->id, "B channel", "CPU", "GPU-wavefront", cpuC.b, wfC.b, tolerance);

	// GPU-recursive vs GPU-wavefront, directly - previously this suite only
	// ever compared each GPU backend against CPU separately, never against
	// each other. That gap matters: several real bugs this codebase has hit
	// (a grazing-angle check present in one GPU backend's material case but
	// missing in the other's independently-hand-ported copy, a per-geometry-
	// type payload field only initialized by one of the two backends' closest-
	// hit programs, ...) are drift BETWEEN the two GPU backends specifically -
	// two independent hand-ports of what's supposed to be identical material
	// logic. Such a bug can easily leave both backends still within
	// `tolerance` of CPU (the CPU-vs-GPU comparisons above already have
	// documented legitimate algorithmic gaps - different NEE strategies for
	// media/DiffuseTransmission, see this file's header comment - baked into
	// their tolerance) while disagreeing with each other by more than that
	// gap actually warrants, which neither existing check would catch.
	// Uses its OWN separately-calibrated tolerance (kRecWfRelTolerance and
	// its own named exceptions, above) rather than reusing the CPU pairs'
	// `tolerance` - see that constant's own comment for the real measured
	// data behind this, and why it's much tighter than the CPU-pair values.
	const float recWfTolerance =
		(s->id == "E10") ? kCameraMediumRelTolerance :
		(s->id == "E1")  ? kHomogeneousMediumRecWfRelTolerance :
		kRecWfRelTolerance;
	check_relative_parity(s->name, s->id, "avg brightness", "GPU-recursive", "GPU-wavefront", recBright, wfBright, recWfTolerance);
	check_relative_parity(s->name, s->id, "R channel", "GPU-recursive", "GPU-wavefront", recC.r, wfC.r, recWfTolerance);
	check_relative_parity(s->name, s->id, "G channel", "GPU-recursive", "GPU-wavefront", recC.g, wfC.g, recWfTolerance);
	check_relative_parity(s->name, s->id, "B channel", "GPU-recursive", "GPU-wavefront", recC.b, wfC.b, recWfTolerance);

	// Regional (block-based) diff, same three backend pairs - reuses the
	// already-rendered/cached images above, no extra rendering cost. See
	// mp_regional_diff's own comment for what this catches that the
	// whole-image checks above cannot. Uses a tolerance scaled from this
	// scene's own whole-image `tolerance` (see regional_tolerance_for's own
	// comment) so an already-documented, already-accepted per-scene gap
	// (E10, Volumes, B13, B1) doesn't get re-flagged here as new information.
	// B14/B11/J2 are regional-only exceptions (see their own tolerance
	// constants' comments) - their whole-image `tolerance` above stays
	// standard.
	const float regionalTolerance =
		(s->id == "B14") ? kMeasuredBrdfRegionalRelTolerance :
		(s->id == "B11") ? kHairFibersRegionalRelTolerance :
		(s->id == "J2")  ? kDiffuseTransmissionTextureRegionalRelTolerance :
		(s->id == "E10") ? kCameraMediumRegionalRelTolerance :
		regional_tolerance_for(tolerance);
	check_regional_parity(s->name, s->id, "CPU", "GPU-recursive", cpuImg, recImg, regionalTolerance);
	check_regional_parity(s->name, s->id, "CPU", "GPU-wavefront", cpuImg, wfImg, regionalTolerance);
	// Rec-vs-wavefront regional: its OWN separately-calibrated tolerance,
	// same reasoning as recWfTolerance above - B14/B11/J2/E10 reuse their
	// existing named exceptions (already measured to comfortably cover this
	// pair's own gap too, see kRecWfRegionalRelTolerance's own comment for
	// why no new constant was needed for those four); E1 and B13 get their
	// own new exceptions, measured specifically for this pair.
	const float recWfRegionalTolerance =
		(s->id == "B14") ? kMeasuredBrdfRegionalRelTolerance :
		(s->id == "B11") ? kHairFibersRegionalRelTolerance :
		(s->id == "J2")  ? kDiffuseTransmissionTextureRegionalRelTolerance :
		(s->id == "E10") ? kCameraMediumRegionalRelTolerance :
		(s->id == "E1")  ? kHomogeneousMediumRecWfRegionalRelTolerance :
		(s->id == "B13") ? kSubsurfaceSlabRecWfRegionalRelTolerance :
		(s->id == "B23" || s->id == "B24") ? kDispersivePrismRecWfRegionalRelTolerance :
		kRecWfRegionalRelTolerance;
	check_regional_parity(s->name, s->id, "GPU-recursive", "GPU-wavefront", recImg, wfImg, recWfRegionalTolerance);
}

INSTANTIATE_TEST_SUITE_P(
	MaterialsAndVolumes, MaterialCpuGpuParityTest,
	::testing::ValuesIn(materials_volumes_and_textures_indices()),
	[](const ::testing::TestParamInfo<int>& info) {
		const SceneDescriptor& desc = get_scene_registry()[info.param];
		std::string name = desc.name ? desc.name : "Unknown";
		std::string sanitized;
		for (char c : name) sanitized += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
		return "Scene" + std::to_string(info.param) + "_" + sanitized;
	});
