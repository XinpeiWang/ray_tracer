#pragma once
// optix_types_restir_media.h -- part 2 of 5 of optix_types.h (included by it, in order; not meant to be included on its own).

// ReSTIR DI (Live Preview only - gpu/optix/wavefront_restir_helpers.h) plain
// data structs. Defined here (host+device safe, like every other struct in
// this file) rather than in wavefront_restir_helpers.h itself, because that
// header also carries __device__-only RIS/reservoir-combine math (wf_rand()-
// based candidate generation, etc.) that must NOT be pulled into the plain
// host .cpp / cross-module extern "C" boundary files (wavefront_launch.h,
// WavefrontPathTracer's own header/.cpp, optix_interface.cpp) - those only
// ever need the STORAGE SHAPE of a reservoir (to allocate/pass a buffer
// pointer), never the device math itself. Mirrors how GpuAliasEntry/
// GpuLightKind above already separate "plain shared data" (this file) from
// "device-only logic that consumes it" (wavefront_device_helpers.h).
struct GpuLightSample {
	int          lightIdx = -1;
	GpuLightKind kind     = GpuLightKind::Quad;
	int          primIdx  = -1;
	float        sampleU = 0.0f, sampleV = 0.0f;
	float3       point  = make_float3(0.0f, 0.0f, 0.0f);
	float3       normal = make_float3(0.0f, 0.0f, 1.0f);
	// The shutter time this sample was drawn at (wf_generate_restir_candidate's
	// own `time` parameter) - needed to re-derive a MOVING Sphere light's
	// correct center (SphereData::center/center1 interpolation) when this
	// sample is later re-evaluated from a different origin than the one it
	// was drawn from (wf_reevaluate_light_geometry, wavefront_restir_helpers.h).
	// Travels WITH the sample specifically so temporal/spatial reuse (which
	// re-evaluates a sample generated at a DIFFERENT pixel/frame, with no
	// shutter time of its own to offer) can't silently fall back to a wrong
	// guess (0.0f) - see that function's own header comment for the bug this
	// fixed. 0.0f (the default) is exactly correct for every non-Sphere kind
	// and every static (center==center1) Sphere, which is why this was easy
	// to not notice was missing.
	float        time = 0.0f;

	CPU_GPU bool valid() const { return lightIdx >= 0; }
};

struct GpuReservoir {
	GpuLightSample sample;
	float weightSum = 0.0f;
	int   M         = 0;
	float W         = 0.0f;
	float pHat      = 0.0f;

	CPU_GPU bool valid() const { return sample.valid() && weightSum > 0.0f; }
	CPU_GPU void clear() { sample = GpuLightSample{}; weightSum = 0.0f; M = 0; W = 0.0f; pHat = 0.0f; }
};

// A previous frame's pinhole camera basis, for GPU-side ReSTIR temporal reuse
// to reproject a current-frame hit point into that frame's screen space -
// exactly the 4 vectors qt_gui/camera_math.h's own CameraBasis/projectToScreen
// already use for the EXISTING host-side accumulation-reprojection feature
// (see that header's own comment on the ray-generation convention these come
// from). A separate, smaller struct rather than reusing the full
// GpuCameraParams (which also carries lens-table pointers/spherical-camera
// fields irrelevant to reprojection) - kept here (not wavefront_restir_helpers.h)
// for the same host+device-safe, math-free reason GpuLightSample/GpuReservoir
// are.
struct GpuReprojectBasis {
	float3 origin;
	float3 lowerLeftCorner;
	float3 horizontal;
	float3 vertical;
};

// Bundles everything wf_finish_material_scatter's ReSTIR temporal-reuse step
// needs beyond the current-frame GpuReservoir* buffer, as ONE extra kernel
// parameter instead of six - see that function's own restirReservoirs
// parameter comment. `history`/`worldPosHistory` are the PREVIOUS render()
// call's post-spatial-reuse reservoirs and per-pixel world positions
// (WavefrontPathTracer's own d_reservoirsHistory_/d_worldPosHistory_);
// `normalOut` is where this call writes the current-frame shading normal at
// each depth==0 pixel (mirrors wf_write_world_pos()'s plain-overwrite
// convention), which the SPATIAL-reuse pass (wavefront_kernels_restir.cu)
// then reads back for its own neighbor-rejection test. `historyValid` is
// false on the very first call after a scene upload (WavefrontPathTracer::
// invalidateRestirHistory()) - deliberately the ONLY explicit invalidation
// signal; a camera CUT (vs. a smooth interactive move) needs no separate
// flag because the per-pixel screen-bounds/disocclusion test below already
// naturally rejects a reprojection that lands off-screen or on a
// wildly-different surface, the same way a hard scene change would.
struct GpuRestirTemporalContext {
	const GpuReservoir* history = nullptr;
	const float4*       worldPosHistory = nullptr;
	float3*             normalOut = nullptr;
	GpuReprojectBasis   prevCamera{};
	bool                historyValid = false;
	int                 imageWidth = 0;
	int                 imageHeight = 0;
};

// ReSTIR for volumetric/participating-media scattering (Live Preview only) -
// see this project's own plan. A depth==0 medium phase-scatter vertex
// (isPhase, wf_finish_material_scatter) already resamples a GpuLightSample
// exactly like a surface DI candidate does (same wf_generate_restir_candidate,
// same restir_reservoir_add - only the target-function proxy differs,
// wf_restir_target_proxy_phase vs wf_restir_target_proxy). GpuVolumeReservoir
// is therefore NOT a new sample-space, just GpuReservoir plus one extra
// identity field: which medium (materials[] index) this reservoir's
// candidate was last validated against - the volumetric analog of a surface
// reservoir's implicit "same geometry" assumption, since two different
// media can have wildly different sigma_t/g/albedo with no partial-credit
// case the way two similarly-oriented surfaces have.
struct GpuVolumeReservoir {
	GpuLightSample sample;
	float weightSum = 0.0f;
	int   M         = 0;
	float W         = 0.0f;
	float pHat      = 0.0f;
	int   mediumMatIdx = -1;

	CPU_GPU bool valid() const { return sample.valid() && weightSum > 0.0f; }
};

// Mirrors GpuRestirTemporalContext exactly, substituting GpuVolumeReservoir
// for GpuReservoir. worldPosHistory is the SAME buffer surface DI/GI/SVGF
// already share (d_worldPosHistory_) - a medium's own BOUNDARY/entry point
// (HitWorkItem::hitPoint, written before any medium-specific interior-point
// logic runs) is exactly as deterministic frame-to-frame as a surface hit,
// so it needs no separate history buffer of its own; only the STOCHASTIC
// interior scatter point (redrawn by a fresh free-flight distance sample
// every frame) would have needed one, and that point is never stored or
// reprojected at all - see this project's own plan for why reprojecting the
// entry point (not the interior point) is what makes this whole feature
// possible with no new reprojection math. No `normalOut` field - a phase
// vertex has no shading normal; medium identity (mediumMatIdx, above) plays
// that role instead, via a separate small per-pixel identity buffer
// (WavefrontPathTracer::d_volumeMatIdx_) rather than a field here, since the
// spatial-reuse pass needs to know a NEIGHBOR pixel's own current-frame
// medium identity, not just a reused reservoir's stored one.
struct GpuVolumeRestirTemporalContext {
	const GpuVolumeReservoir* history         = nullptr;
	const float4*             worldPosHistory = nullptr;
	GpuReprojectBasis         prevCamera{};
	bool                      historyValid    = false;
	int                       imageWidth  = 0;
	int                       imageHeight = 0;
};

// ReSTIR GI's own plain data structs (GpuGiSample/GpuGiReservoir/
// GpuGiOriginContext) live in wavefront_types.h, not here, alongside
// RayWorkItem/ShadowRayWorkItem - unlike GpuLightSample/GpuReservoir above,
// they need kWFNWavelengths-sized spectral storage (a GI candidate's cached
// radiance and stashed throughput must stay in the same spectral domain,
// using the SAME per-path hero wavelengths a bounce inherits from its
// primary ray - RayWorkItem::wavelengths' own "fixed for the whole path"
// comment), and kWFNWavelengths is defined in wavefront_types.h, which
// includes THIS file, not the other way around.

// Material types
enum class MaterialType : int {
	Lambertian = 0,
	Metal = 1,
	Dielectric = 2,
	DiffuseLight = 3,
	RoughDielectric = 4,
	Conductor = 5,           // GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)
	CoatedDiffuse = 6,       // rough dielectric coat over Lambertian (pbrt-v4 CoatedDiffuseBxDF)
	ThinDielectric = 7,      // zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF)
	CoatedConductor = 8,     // rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF)
	DiffuseTransmission = 9,  // diffuse reflection + diffuse transmission (pbrt-v4 DiffuseTransmissionBxDF)
	NormalizedFresnel   = 10, // Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)
	// Homogeneous participating medium (pbrt-v4/RTIOW constant_medium):
	// closed-form free-path (Beer-Lambert) sampling + Henyey-Greenstein
	// phase function, matching src/TheRestOfYourLife/constant_medium.h +
	// src/shared/volume_scattering.h - only usable on sphere geometry
	// (the boundary shape), reuses MaterialData.ior as sigma_t (extinction
	// coefficient) and MaterialData.fuzz as the HG asymmetry g.
	Medium = 11,
	// Marschner/Chiang fiber scattering (pbrt-v4 HairBxDF, src/shared/
	// bxdfs_hair.h) - matches src/TheRestOfYourLife/hair_material.h exactly,
	// including its "shading normal as fiber-tangent proxy" simplification
	// (no literal fiber geometry - src/shared/shapes.h's CurveShape is unused
	// dead code, never wired to any scene). Sphere-only, since that's the
	// only geometry any scene ever applies it to (scene 19's 5 hair spheres).
	// Reuses MaterialData.albedo as sigma_a (RGB absorption), .fuzz as
	// beta_m, .ior as eta, .eta_c.x as beta_n, .eta_c.y as alpha_deg.
	Hair = 12,
	// Sphere simultaneously a dielectric surface AND an internal
	// participating medium - matches src/TheRestOfYourLife/scenes_book.h's
	// build_final_scene() trick of adding the SAME boundary sphere to the
	// CPU world twice (once as a plain dielectric, once wrapped in
	// constant_medium): a ray always hits the dielectric surface first when
	// entering from outside (the medium's sampled hit distance can never be
	// closer than the entry surface), but on the very next bounce - now
	// travelling inside, hitting this sphere's exit surface - it may
	// scatter off the medium before reaching that exit. Handled inline in
	// optix_intersection_sphere.h (needs the same shape-specific near/far
	// re-intersection Medium above already does), not through
	// shade_material(). Reuses MaterialData.ior as the dielectric index of
	// refraction (Dielectric's own field), .albedo as the medium's
	// single-scatter color and .fuzz as its HG asymmetry g (Medium's own
	// fields), and .eta_c.x - otherwise unused outside Conductor/Hair - as
	// sigma_t (extinction coefficient), since this is the only material
	// type that needs a dielectric IOR and a medium's sigma_t/g/albedo at
	// the same time.
	DielectricMedium = 13,
	// Lambertian with a perturbed shading normal, sourced from a tangent-
	// space RGB normal-map texture (MaterialData.textureIdx) - matches
	// src/TheRestOfYourLife/normal_map_materials.h's normal_map_material
	// wrapper exactly, but collapsed into one material instead of CPU's
	// "wrapper delegates to inner material after perturbing rec.normal"
	// pattern: handled inline in optix_intersection_sphere.h (needs the
	// sphere's own tangent/dpdu, computed the same way CPU's sphere.h
	// does), which then calls shade_material() with the perturbed normal
	// and a temporary MaterialType::Lambertian view of this same data -
	// reusing that case's existing NEE/MIS logic verbatim rather than
	// duplicating it. Reuses MaterialData.albedo as the inner Lambertian's
	// flat color (CPU's scene never combines this with an albedo texture,
	// so .textureIdx is unambiguously "the normal map" here, not "the
	// albedo texture" the way it means for a plain Lambertian).
	// Sphere-only, matching CPU's only current use (scene 20's one
	// normal-mapped sphere) - see also this codebase's confirmed-empirically
	// finding that scene 20's OTHER normal-perturbation technique (bump-
	// mapping the back wall/box via bump_map_material + noise_texture) is
	// a no-op on CPU itself (a p-only procedural texture can't produce a
	// nonzero (u,v) finite-difference gradient), so it isn't ported here -
	// the GPU builder renders those surfaces as plain flat Lambertian,
	// which already matches CPU's actual rendered pixels exactly.
	NormalMappedLambertian = 14,
	// Disney/pbrt-v4-style multi-lobe BSDF (diffuse + GGX specular + GGX
	// clearcoat, metallic-blended Fresnel) - matches src/TheRestOfYourLife/
	// principled_material.h exactly by directly instantiating the same
	// CPU_GPU-tagged src/shared/bxdfs_principled.h::PrincipledBxDF<T> struct
	// device-side (T=float), the same pattern already used for
	// MaterialType::Hair's HairBxDF<T> - see sample_principled_material() in
	// optix_device_helpers.h. Like Hair, CPU's principled::scatter() sets
	// skip_pdf=true (its returned weight already folds in the BSDF value,
	// cosine term, and multi-lobe balance-heuristic PDF), so this is handled
	// as a specular-style bounce (no NEE/MIS) exactly like Hair/Metal/
	// Dielectric/Conductor.
	// Field reuse (no new MaterialData fields needed): albedo = base color,
	// ior = ior, fuzz = roughness, eta_c.x = metallic, eta_c.y = clearcoat,
	// eta_c.z = clearcoat_rough (k_c unused - Principled isn't a Conductor).
	// Sphere-only, matching CPU's only current use (scene 18's 7 showcase
	// spheres).
	Principled = 15,
	// Heterogeneous, procedural Perlin-FBm-density participating medium
	// (pbrt-v4 CloudMedium, src/shared/cloud_medium.h) - matches
	// src/TheRestOfYourLife/cloud_medium_hittable.h exactly (delta-tracking/
	// null-collision free-path sampling using the medium's majorant sigma_t,
	// since density varies per point unlike Medium's single fixed sigma_t).
	// Unlike every other MaterialType, this one's real parameters (the
	// world-to-medium affine transform, density/wispiness/frequency, and
	// sigma_a/sigma_s) don't fit MaterialData's spare per-material bytes, so
	// this only stores an INDEX (cloud_medium_extra.cloudMediumIdx) into
	// LaunchParams::cloudMediums, mirroring the RealisticCamera lens-table
	// pattern (GpuLensElement/GpuExitPupilBounds) rather than any other
	// MaterialType's direct field reuse. .medium_albedo/.g are still reused
	// directly (single-scatter color, HG phase asymmetry) since those DO fit
	// and every other Medium-family type already uses those same two names.
	// Sphere-only for the same reason Medium is: only usable on the boundary
	// shape the intersection code re-derives entry/exit roots for.
	CloudMedium = 16,
	// Heterogeneous participating medium with a real per-voxel R/G/B
	// scattering grid (pbrt-v4 RGBGridMedium, src/shared/rgb_grid_medium.h) -
	// matches src/TheRestOfYourLife/rgb_grid_medium_hittable.h's delta
	// tracking, except GPU uses one single GLOBAL majorant for the whole
	// medium (GpuRgbGridMedium::sigma_maj) rather than CPU's real per-voxel
	// DDA majorant grid - a deliberate simplification (see gpu_rgb_grid_
	// trilinear's comment in optix_intersection_sphere.h) to keep GPU delta
	// tracking algorithmically simple. Like CloudMedium, this only stores an
	// INDEX (rgb_grid_medium_extra.rgbGridMediumIdx) into
	// LaunchParams::rgbGridMediums, one level further than CloudMedium even:
	// RGBGridMediumData<T> can't be used directly on device at all (its
	// SampledGrid<T> members use std::vector/std::optional internally), so
	// GpuRgbGridMedium is a from-scratch flat metadata struct (below) whose
	// actual voxel data lives in a SEPARATE shared flat buffer
	// (LaunchParams::rgbGridData), indexed via GpuRgbGridMedium::dataOffset.
	// Sphere-only, same reason as Medium/CloudMedium.
	RgbGridMedium = 17,
	// Real tabulated BSSRDF (pbrt-v4 TabulatedBSSRDF / SubsurfaceMaterial),
	// on BOTH GPU backends. Mirrors src/TheRestOfYourLife/material_pbrt.h's
	// `class subsurface` + camera.h::sample_bssrdf_exit()'s 3-axis-MIS
	// probe/exit-point algorithm exactly, replacing the flat-diffuse
	// fallback pbrt_gpu_builder.h used to emit for pbrt_flatten::
	// MaterialKind::Subsurface. Landed in two phases: the recursive backend
	// first (optix_bssrdf.h / optix_probe_hit.h - see shade_material()'s own
	// comment, optix_device_helpers.h), then the wavefront backend as a real
	// probe-walk stage (BssrdfProbeWorkItem -> wavefront_probe.h's
	// wf_bssrdf_probe_walk() -> resolve_bssrdf_exit() in wavefront_kernels.cu)
	// - both now share the same tabulated BSSRDFTable data and algorithm, at
	// parity with each other and with CPU.
	//
	// Field reuse: .albedo stays `m.color` (the flat-gray fallback color,
	// 0.5/0.5/0.5 by pbrt_flatten.h's own default) - a real texture is never
	// carried by a Subsurface material in this loader, so this slot is only
	// ever read as the CPU-parity fallback color, never as a real albedo.
	// .ior is eta (index of refraction, already assigned generically before
	// this type's own switch case). sigma_a/
	// sigma_s (RGB absorption/scattering coefficients, ALREADY scale-
	// multiplied - see pbrt_flatten::Material::sigma_a/sigma_s's own comment)
	// live in the emission/transmittance union and the k_c-aliased union
	// respectively (see MaterialData's own field comments below) - both
	// otherwise unused by this material kind (no emission, no Conductor k).
	// .textureIdx is repurposed as an index into LaunchParams::bssrdfTables
	// (this material's precomputed BSSRDFTable, built once per unique (g,eta)
	// pair at scene-build time - see pbrt_gpu_builder.h) rather than a real
	// texture index; Subsurface materials never carry a real texture in this
	// loader, so this reuse is unambiguous on the recursive backend, which is
	// the only backend that ever reads it as anything but -1.
	Subsurface = 18,
	// Real tabulated measured-BRDF (pbrt-v4 Material "measured" - a real
	// gonioreflectometer-measured BRDF loaded from a binary .bsdf tensor
	// file, Dupuy & Jakob's format), BOTH GPU backends. Mirrors
	// src/TheRestOfYourLife/material_pbrt.h's `class measured` exactly: a
	// single VNDF-importance-sampled BxDF sample per hit (MeasuredBxDF::
	// sample_f - src/shared/measured_bxdf.h), no Fresnel entry/exit split,
	// no probe walk - same *shape* as MaterialType::Metal/Conductor/Hair/
	// Principled (sample once, get a direction + full skip_pdf-style
	// throughput weight, continue as a non-NEE specular-style bounce). This
	// replaces the flat-diffuse fallback pbrt_gpu_builder.h used to emit for
	// pbrt_flatten::MaterialKind::Measured before this MaterialType existed
	// (see gpu/optix/optix_measured_bxdf.h / wavefront_measured_bxdf.h for
	// the device math, and optix_device_helpers.h's shade_material() /
	// wavefront_kernels.cu's evaluate_materials() for the two backends'
	// dispatch).
	//
	// Field reuse: .textureIdx is repurposed as an index into
	// LaunchParams::measuredTables (this material's flattened 5-sub-table
	// set - see GpuMeasuredTable below), built once per unique resolved
	// .bsdf file path at scene-build time (getOrBuildMeasuredTable(),
	// pbrt_gpu_builder.h) - same "index into a shared LaunchParams array"
	// convention as MaterialType::Subsurface's own textureIdx reuse just
	// above; a measured material never carries a real texture in this
	// loader, so this reuse is unambiguous on both backends. No other
	// MaterialData field is used - the query wavelengths (612/549/465nm,
	// approximating sRGB primaries, matching `class measured`'s kLambdaR/G/B
	// exactly) are fixed constants in the device code, not per-material data.
	Measured = 19,
	// GGX microfacet metal with a flat RGB tint standing in for Fresnel, no
	// complex IOR (pbrt-v4/RTOW rough_metal, src/shared/bxdfs_conductor.h's
	// RoughMetalBxDF) - matches src/TheRestOfYourLife/material_pbrt.h's
	// `class rough_metal` exactly, and is NOT the same model as
	// MaterialType::Metal above (that one is a fuzz-perturbed mirror with no
	// real microfacet distribution at all - CPU's plain `class metal`, a
	// different class). Scenes B1 (RoughMetalSpheres) and B2
	// (CornellRoughMetal) previously called add_metal() for their "rough
	// metal" spheres/box, which silently rendered the wrong model on GPU
	// (confirmed via the CPU builder always using `rough_metal`) - fixed by
	// routing those two scenes through add_rough_metal() instead.
	// Field reuse: .albedo = flat tint, .fuzz (aliased .roughness) = GGX
	// roughness (RoughnessToAlpha'd device-side, same as Conductor/
	// RoughDielectric/CoatedDiffuse/CoatedConductor) - no eta_c/k_c, unlike
	// Conductor, since RoughMetalBxDF has no real Fresnel model.
	RoughMetal = 20,
	// Heterogeneous participating medium with a single-channel per-voxel
	// scalar density grid (pbrt-v4 GridMedium/"uniformgrid",
	// src/shared/sampled_grid.h's GridMediumData<T>) - matches
	// src/TheRestOfYourLife/grid_medium_hittable.h's delta tracking, the
	// single-channel twin of MaterialType::RgbGridMedium just above (same
	// "one global majorant, not CPU's real per-voxel DDA majorant grid"
	// deliberate GPU simplification - see that enumerator's own comment).
	// Only stores an INDEX (grid_medium_extra.gridMediumIdx) into
	// LaunchParams::gridMediums, same "flat metadata struct + separate flat
	// voxel-data buffer (LaunchParams::gridData)" shape as RgbGridMedium,
	// since GridMediumData<T> can't be used directly on device either (same
	// std::vector-backed SampledGrid<T> problem). Sphere-only, same reason
	// as Medium/CloudMedium/RgbGridMedium.
	GridMedium = 21,
	// Stochastic two-material blend (pbrt-v4 MixMaterial) - matches
	// src/TheRestOfYourLife/material_pbrt.h's `mix_material`/`branch_hash01`
	// exactly: at each shading point, deterministically (hashed from the
	// world-space hit point, NOT a fresh random draw - see branch_hash01's
	// own CPU-side comment for why: scatter()/scattering_pdf()/shadow-ray
	// classification must all agree about which sub-material a given
	// scattering event used) picks sub-material A or B and delegates
	// entirely to it - not a real blended BxDF evaluation. This MaterialType
	// itself has NO shading case anywhere; every shape's closest-hit/
	// intersection program and every shadow any-hit program resolves it
	// (mix_extra below -> LaunchParams::materials[chosen index], looping
	// while the result is itself another Mix, capped the same kMaxMixDepth
	// pbrt_gpu_builder.h's build-time resolveMixColor()/recursive
	// makeMaterial() already use) to a REAL MaterialType before any other
	// mat.type branch runs - see resolve_mix_material()/wf_resolve_mix_
	// material() in optix_device_helpers.h/wavefront_kernels.cu. Shape-
	// agnostic (unlike Hair/Subsurface/Medium-family): it never needs its
	// own entry in material_requires_sphere_only_handling()/wf_material_
	// requires_sphere_only_handling(), since resolution always happens
	// before those checks see the (by-then-real) resolved type.
	Mix = 22,
	// pbrt-v4's real "no BSDF" interface material (Material "none"/"" -
	// pbrt_flatten::MaterialKind::Interface) - a shape that bounds a
	// participating medium with literally no surface response of its own.
	// The ray passes straight through completely unperturbed: no Fresnel
	// reflection, no refraction, no critical angle (unlike the earlier
	// approach of routing this through Dielectric with eta forced to
	// 1.001). shade_material()'s new case sets out_is_medium_boundary=true
	// instead of out_is_specular - a distinct signal from "real specular
	// bounce" every closest-hit program packs into flag==4 (not flag==1),
	// so optix_raygen.h can skip both the depth/RR accounting AND the
	// specular_bounce/prev_brdf_pdf MIS-state update for the crossing,
	// exactly like the CPU integrators' equivalent scatter_record::
	// is_medium_boundary branch (camera.h, bdpt.h, sppm_adapter.h). Sphere/
	// disk/cylinder only, matching CPU's own shape support (Triangle/
	// BilinearPatch have no `medium` field - see pbrt_flatten.h's own
	// comment on the trianglemesh-medium gap this doesn't attempt to
	// close). No MaterialData fields needed at all.
	Interface = 23
};

// The single canonical list of MaterialTypes GPU SPPM's camera/photon-pass
// raygens (gpu/optix/sppm_programs.cu) actually implement BSDF sampling for
// - Lambertian/DiffuseLight handled directly, RoughDielectric/Metal/
// Dielectric/Conductor/RoughMetal/DiffuseTransmission via
// sppm_is_delta_material()/sppm_sample_delta_material()'s explicit dispatch,
// Mix via sppm_resolve_mix_material_index() (resolved to one of the other
// entries in this list before any dispatch runs - a Mix instance itself
// never reaches sppm_is_delta_material()/sppm_bsdf_f(), so it needs no case
// of its own in either). Mix's own resolution needs the current SPPM
// iteration index folded into its branch-selection hash (GPU SPPM's camera
// pass traces an unjittered primary ray, unlike the other two backends -
// see sppm_resolve_mix_material_index()'s own comment, sppm_programs.cu,
// for the full rationale and the real-render-verified bug this fixes: a
// variant-less resolve rendered Mix as a frozen per-pixel binary choice for
// the whole render, not the intended converging stochastic blend). Any
// MaterialType NOT in this list falls through
// sppm_programs.cu's own "treat as Lambertian" default, silently reading
// that material's (possibly unset) albedo union slot - which is exactly the
// class of bug gpu/optix/optix_interface.cpp's sppm_gpu_unsupported_reason()
// exists to reject a scene for BEFORE it can happen, rather than let it
// render silently wrong.
//
// CoatedDiffuse/CoatedConductor/Hair/Subsurface/Measured are deliberately
// NOT in this list despite CPU and both other GPU backends supporting all
// five - see sppm_is_delta_material()'s own comment (sppm_programs.cu) for
// why each is out of scope: CoatedDiffuse/CoatedConductor were investigated
// and real support was built (the shared CoatedDiffuseBxDF/
// CoatedConductorBxDF::f()/sample_local(), src/shared/bxdfs_layered.h) but
// pulled back out after real-render verification showed it severely too
// dark - a real architecture mismatch (SPPM's single-NEE-sample-per-
// visible-point design vs. this material's largely-specular-coat
// reflectance), not a simple bug. Hair only ever renders on tessellated-
// curve (bilinear-patch) geometry and Subsurface needs a whole separate
// BSSRDF probe-walk raygen pass, both permanently rejected by
// sppm_gpu_unsupported_reason()'s own geometry checks regardless of
// material-dispatch support; Measured's tabulated-BRDF tables aren't wired
// into SPPMLaunchParams at all.
//
// Defined here, next to MaterialType's own definition, and used by both
// sppm_gpu_unsupported_reason() (host-side rejection check,
// optix_interface.cpp) and README/comment cross-references, SPECIFICALLY so
// there is exactly one place to update when sppm_programs.cu's own
// dispatch gains a new case - previously optix_interface.cpp carried its
// own separately-hand-maintained copy of this set with nothing forcing it
// to stay in sync with the device code it was describing.
inline bool sppm_gpu_material_supported(MaterialType t) {
	switch (t) {
	case MaterialType::Lambertian:
	case MaterialType::DiffuseLight:
	case MaterialType::RoughDielectric:
	case MaterialType::Metal:
	case MaterialType::Dielectric:
	case MaterialType::Conductor:
	case MaterialType::RoughMetal:
	case MaterialType::DiffuseTransmission:
	case MaterialType::Interface:
	case MaterialType::Mix:
		return true;
	default:
		return false;
	}
}

// GPU flat-array mirror of one PiecewiseLinear2D<Dimension> instance (see
// src/shared/piecewise_linear_2d.h's private members m_nx/m_ny/m_ps/m_pst/
// m_pv/m_data/m_mcdf/m_ccdf, exposed read-only via the const accessors added
// there for exactly this purpose - see that file's own comment). Used by
// MaterialType::Measured's device math (gpu/optix/optix_measured_bxdf.h /
// wavefront_measured_bxdf.h). One instance per PiecewiseLinear2D sub-table
// (ndf/sigma/vndf/luminance/spectra - see GpuMeasuredTable below); `dim`
// records which of the three Dimension values (0, 2, or 3) this particular
// sub-table was built at, since a MeasuredBRDFData mixes all three (unlike
// GpuBssrdfTable, which only ever describes one shape).
struct GpuPL2DTable {
	int nx, ny;                  // XSize()/YSize()
	int dim;                     // Dimension: 0 (ndf/sigma), 2 (vndf/luminance), 3 (spectra)
	int param_res[3];            // m_ps[i] per axis; unused axes (i >= dim) are 1
	int param_stride[3];         // m_pst[i] per axis; unused axes are 0
	// Offset into LaunchParams::measuredParamValues, param_res[i] floats
	// starting here, per axis; -1 for unused axes (i >= dim).
	int param_value_offset[3];
	int data_offset;             // offset into ::measuredData, size = slices*nx*ny
	// offset into ::measuredMcdf/::measuredCcdf, or -1 for a table with no
	// CDF (build_cdf=false - ndf/sigma/spectra; only vndf/luminance have one).
	int mcdf_offset;
	int ccdf_offset;
};

// One measured-BRDF material's full table set - mirrors MeasuredBRDFData
// (src/shared/measured_bxdf.h) exactly: ndf/sigma unconditional
// (Dimension=0, Eval()-only), vndf/luminance conditioned on (phi_i,theta_i)
// (Dimension=2, real CDFs, Sample()-only - this material samples exactly
// once per hit, see MaterialType::Measured's own comment), spectra
// conditioned on (phi_i,theta_i,lambda) (Dimension=3, Eval()-only). Built
// once per unique resolved .bsdf file path (pbrt_gpu_builder.h's
// getOrBuildMeasuredTable(), deduped the same way getOrBuildBssrdfTable()
// dedupes by (g,eta) - the sportscar scene's ilm_l3_37_metallic_spec.bsdf is
// referenced by 4 materials).
struct GpuMeasuredTable {
	GpuPL2DTable ndf, sigma, vndf, luminance, spectra;
	int isotropic;   // bool as int - see MeasuredBRDFData::isotropic
};

// Device-uploaded mirror of src/shared/bssrdf.h's BSSRDFTable, for the
// recursive GPU backend's MaterialType::Subsurface probe walk (see
// shade_material()'s own comment and gpu/optix/optix_bssrdf.h). Built ONCE
// per unique (g,eta) pair at scene-build time by calling the existing CPU
// ComputeBeamDiffusionBSSRDF() and uploading the resulting arrays - not a
// device-side computation (see pbrt_gpu_builder.h). Multiple Subsurface
// materials commonly share one table (same g/eta, different sigma_a/sigma_s
// - e.g. this codebase's own dragon_10/50/250 "scale" scenes, which all use
// eta=1.5, g=0 and differ only in scale), so tables are deduplicated and
// referenced by index (MaterialData::textureIdx, repurposed - see
// MaterialType::Subsurface's own comment) rather than embedded per-material,
// mirroring the CloudMedium/RgbGridMedium "index into a LaunchParams array"
// convention rather than GoniometricLightGPU/ProjectionLightGPU's inline-
// array convention, since a table (100 rho x 64 radius by default) is much
// larger than those lights' tiny images.
//
// rho_eff (pbrt-v4's per-row effective albedo, used by SubsurfaceFromDiffuse
// for CPU-only reflectance+mfp inversion - not needed here, see
// pbrt_flatten.h/material_pbrt.h) is NOT uploaded: it is exactly
// profile_cdf[row*n_radius + (n_radius-1)] (integrate_catmull_rom's own
// returned total equals the CDF's own last entry - see src/shared/
// sampling_2d.h), so gpu_bssrdf_pdf_sr() (optix_bssrdf.h) recovers it from
// profile_cdf directly instead of uploading a fourth per-table array.
struct GpuBssrdfTable {
	int n_rho;
	int n_radius;
	int rho_offset;      // into LaunchParams::bssrdfRhoSamples
	int radius_offset;   // into LaunchParams::bssrdfRadiusSamples
	// profile and profile_cdf share this same offset and n_rho*n_radius
	// layout (row i = rho_samples[i], column j = radius_samples[j]) into
	// their own separate flat buffers (LaunchParams::bssrdfProfile /
	// bssrdfProfileCdf) - one offset serves both since every table's rows
	// are built and stored in lockstep (see pbrt_gpu_builder.h).
	int profile_offset;
};

// Heterogeneous per-voxel R/G/B scattering grid metadata (GPU-only flat
// analog of RGBGridMediumData<T> - see MaterialType::RgbGridMedium's
// comment for why that struct can't be used directly on device). One
// instance per RgbGridMedium material, indexed via MaterialData::
// rgb_grid_medium_extra.rgbGridMediumIdx. Plain POD, no CPU_GPU tagging
// needed (its own fields are read directly by ordinary device code in
// optix_intersection_sphere.h/wavefront_kernels.cu, same as SphereData/
// QuadData already are).
struct GpuRgbGridMedium {
	float bounds_min[3], bounds_max[3];  // world-space AABB
	float mat[9];                         // row-major 3x3 world->medium transform
	float translate[3];
	int   nx, ny, nz;                     // grid resolution
	// Element offset into LaunchParams::rgbGridData - the R channel block
	// starts here, G at +nx*ny*nz, B at +2*(nx*ny*nz).
	int   dataOffset;
	float sigma_scale;  // overall density multiplier (matches CPU's sigma_scale)
	float sigma_maj;    // precomputed global majorant = sigma_scale * max
	                     // voxel value across all three channels, with a
	                     // small safety margin - see the GPU scene builder.
	float phase_g;       // Henyey-Greenstein asymmetry
	// Real per-voxel "rgb Le" emission (pbrt-v4 RGBGridMedium::LeGrid, CPU's
	// RGBGridMediumData::Le_grids) - GPU-side counterpart of dataOffset
	// above, but a SEPARATE offset into the same flat LaunchParams::
	// rgbGridData buffer (R block at leDataOffset, G at +nx*ny*nz, B at
	// +2*(nx*ny*nz)), since the sigma_s block above already occupies
	// [dataOffset, dataOffset+3*nx*ny*nz). -1 when this medium has no valid
	// "rgb Le" array (matches CPU's Le_grids.has_value()==false) - device
	// code must check this before reading rgbGridData at leDataOffset.
	int   leDataOffset;
	// Real per-voxel "rgb sigma_a" absorption (RGBGridMediumData::sigma_a_grids): a SEPARATE block in the same rgbGridData buffer (R at
	// saDataOffset, G at +nx*ny*nz, B at +2*(nx*ny*nz)). -1 when the scene gave no "rgb sigma_a" - then, exactly as on the CPU and in
	// pbrt, sigma_a is the constant sigma_scale * 1 in every channel (NOT zero: an RGB grid that only gives sigma_s is strongly absorbing).
	// The grid used to be scattering-only on the GPU (absorption silently dropped); both backends now do per-channel spectral tracking with
	// sigma_a and sigma_s (heterogeneous_tracking_step, src/shared/volume_scattering.h), with sigma_maj bounding max_c(sigma_a + sigma_s).
	int   saDataOffset;
	// Matches CPU's RGBGridMediumData::Le_scale / pbrt-v4 "Lescale" - applied
	// at sample time, NOT baked into the stored grid values (mirrors
	// dataOffset's own sigma_scale being a separate multiplier for the same
	// reason). 0 when absent (is_emissive()'s own CPU-side `Le_scale > 0`
	// gate), so a stray leDataOffset>=0 with a zero scale still safely
	// contributes no emission even without checking leDataOffset first.
	//
	// IMPORTANT - not radiometrically weighted like CPU: CPU's
	// RGBGridMediumData::sample_point() only attributes emission to the
	// sigma_a/sigma_t fraction of a real collision (rgb_grid_medium_
	// hittable.h) - the "this collision was absorption, not scattering"
	// probability. This struct has no sigma_a grid at all (sigma_scale/
	// sigma_maj above are scattering-only - a real, pre-existing GPU
	// simplification), so that fraction is architecturally always 0 here;
	// applying CPU's exact formula would make every rgbgrid Le request a
	// silent no-op. Each backend's own RgbGridMedium closest-hit case
	// instead emits the FULL per-voxel Le at every accepted scatter
	// collision (weight 1, not sigma_a/sigma_t) as the only choice that
	// makes the feature visible given this constraint - a real, deliberate
	// tradeoff, not an oversight. Consequence: a scene combining real
	// "rgb sigma_s" (heavy multi-scattering) with "rgb Le" on the SAME
	// medium can render substantially - not just slightly - brighter on GPU
	// than the equivalent --cpu render, since each internal scatter bounce
	// re-adds the same full, unattenuated Le rather than CPU's small
	// absorption-weighted fraction. Giving this struct a real sigma_a grid
	// (closing that gap for exact CPU/GPU parity) is a larger, separate lift
	// deliberately deferred, not attempted here.
	float Le_scale;
};

// Single-channel twin of GpuRgbGridMedium above - see MaterialType::
// GridMedium's own comment.
struct GpuGridMedium {
	float bounds_min[3], bounds_max[3];  // world-space AABB
	float mat[9];                         // row-major 3x3 world->medium transform
	float translate[3];
	int   nx, ny, nz;                     // grid resolution
	// Element offset into LaunchParams::gridData - one flat block of
	// nx*ny*nz density values, no R/G/B split (unlike GpuRgbGridMedium's
	// dataOffset).
	int   dataOffset;
	float sigma_scale;  // sigma_a+sigma_s collapsed to one scalar (matches
	                     // GridMediumData<T>::sigma_a/sigma_s's own CPU-side
	                     // luminance() collapse - see pbrt_gpu_builder.h)
	float sigma_maj;    // precomputed global majorant = sigma_scale * max
	                     // density value, with a small safety margin - see
	                     // the GPU scene builder.
	float phase_g;       // Henyey-Greenstein asymmetry
};

// Texture "imagemap"'s own "string wrap" (pbrt-v4) - matches CPU's
// MipWrapMode (src/shared/mipmap.h) exactly, same enumerator order/values,
// so pbrt_gpu_builder.h's resolution of Material::textureWrap ("repeat"/
// "clamp"/"black" - already parsed by pbrt_flatten.h) can mirror
// imageMapOptionsFor()'s (pbrt_cpu_builder.h) identical if/else chain
// verbatim. Clamp=0 is TextureData::wrapMode's zero-init default (every
// pre-existing `TextureData tex{}` call site, and image-texture-table
// entries never built through the GPU imagemap-with-options path below -
// e.g. roughness/transmittance/displacement/checker/mix), matching this
// codebase's pre-existing hard-clamp GPU behavior exactly for every one of
// those slots, unchanged by this addition.
enum class GpuWrapMode : int { Clamp = 0, Repeat = 1, Black = 2 };

// Texture kinds - see TextureData below. Matches CPU texture classes in
// src/TheRestOfYourLife/texture.h: image_texture, noise_texture,
// checker_texture, uv_checker_texture, fbm_texture, marble_texture, and
// mix_texture. mipmap_texture still has no GPU equivalent.
enum class TextureKind : int {
	Image = 0,
	Noise = 1,
	Checker = 2,
	// pbrt-v4's UV-tiled 2D "checkerboard" texture class - deliberately a
	// SEPARATE kind from Checker above rather than a variant of it: Checker
	// is this project's own original 3D world-space checker (parity of
	// floor(p.xyz * scale), keyed on the hit point), while pbrt's
	// checkerboard tiles by (u,v) * (uscale,vscale) - a materially
	// different result for the same "scale" the two shouldn't share a code
	// path over (see texture.h's uv_checker_texture, the CPU counterpart).
	UVChecker = 3,
	// pbrt-v4 FBmTexture (fractional Brownian motion) - matches CPU's
	// fbm_texture exactly (fbm_simple<T>, noise.h).
	FBm = 4,
	// pbrt-v4 MarbleTexture - matches CPU's marble_texture exactly (FBm-
	// perturbed sine wave through the same 9-knot RGB Bezier spline).
	Marble = 5,
	// pbrt-v4 SpectrumMixTexture, flat-literal tex1/tex2 only (see
	// pbrt_flatten.h's Material::hasMixReflectance comment) - matches CPU's
	// new mix_texture exactly.
	Mix = 6,
	// pbrt-v4 WindyTexture - matches CPU's windy_texture exactly
	// (two FBm calls combined - see that class's own comment). Real
	// pbrt-v4 WindyTexture takes no scene-overridable params, so this
	// kind needs no new TextureData fields at all.
	Windy = 7,
	// pbrt-v4 WrinkledTexture - matches CPU's wrinkled_texture exactly
	// (raw Turbulence, not FBm - see that class's own comment). Reuses
	// omega/octaves below, same meaning as the FBm kind's.
	Wrinkled = 8,
	// pbrt-v4 DotsTexture - matches CPU's dots_texture exactly (a
	// per-UV-cell polka-dot test via perlin_noise at z=0.5, blending
	// "inside"/"outside"). Reuses color1/color2 (inside/outside) and
	// tex1ImageIdx/tex2ImageIdx (one-level-nested bare imagemap for
	// either slot) below, same convention as Mix/UVChecker.
	Dots = 9,
	// pbrt-v4 BilerpTexture - matches CPU's bilerp_texture exactly (plain
	// bilinear blend of 4 corner colours by (u,v), flat-literal corners
	// only - no nested-imagemap support, matching how rarely real scenes
	// bind anything but a flat colour to a bilerp corner). Reuses
	// color1/color2 for v00/v01; the other two corners (v10/v11) are
	// packed into uScale/vScale/omega and marbleScale/marbleVariation/
	// mixAmount below (all unused by Bilerp) rather than adding two more
	// float3 fields that would grow every TextureData entry - see those
	// fields' own comments.
	Bilerp = 10
};

// One entry per texture, indexed by MaterialData::textureIdx. Image
// textures point into a single shared flat 8-bit RGB pixel buffer
// (LaunchParams::texturePixels) via byte offset - same "flat device
// buffer, index in device code" convention as every other array in this
// codebase (see LaunchParams below). Matches src/TheRestOfYourLife/
// rtw_stb_image.h's own byte layout exactly (3 bytes/pixel, row-major).
// Exact sRGB -> linear for one 8-bit texel channel (pbrt-v4's SRGB8ToLinear, util/color.h). An Image
// TextureData with `srgb` set keeps the file's own sRGB bytes and every lookup decodes them before
// filtering, as pbrt does; the previous 8-bit LINEAR copy (decoded with pow 2.2, then requantized)
// crushed dark texels to zero and banded the darks.
inline __host__ __device__ float srgb8_to_linear(unsigned char b) {
	const float c = static_cast<float>(b) * (1.0f / 255.0f);
	return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

// One texel's RGB as linear floats, decoding the sRGB bytes when the texture asks for it.
inline __host__ __device__ float3 texel_rgb(const unsigned char* px, bool srgb) {
	if (srgb) return make_float3(srgb8_to_linear(px[0]), srgb8_to_linear(px[1]), srgb8_to_linear(px[2]));
	const float k = 1.0f / 255.0f;
	return make_float3(px[0] * k, px[1] * k, px[2] * k);
}
