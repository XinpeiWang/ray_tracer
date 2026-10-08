#pragma once
// optix_device_shade.h -- part 4 of 5 of optix_device_helpers.h (included by it, in order; not meant to be included on its own).

// Evaluates material scattering for every MaterialType except Medium and
// Hair, which are sphere-only and stay in optix_intersection_sphere.h's own
// closest-hit program (they need shape-specific re-intersection/geometry
// data - a medium's exit distance, a hair fiber's curve parameterization -
// that no other primitive type has). Identical scattering/NEE/MIS logic
// regardless of which primitive was actually hit; only `normal`,
// `hit_point`, `front_face`, and `uv` differ per caller, and those are
// passed in rather than recomputed here. Leaves out_scattered false (every
// other output untouched) for DiffuseLight/default, exactly matching each
// call site's prior inline default: case - `emission` is the sole in/out
// parameter (NEE contributions from Lambertian/CoatedDiffuse/
// NormalizedFresnel get added directly into whatever the caller already
// initialized it to, i.e. mat.emission). `uv` is only meaningful for
// Lambertian materials with textureIdx >= 0 (scene 8's Earth/noise
// spheres) - every other caller/material can pass (0,0).
// `matIdx` (this material's own index into params.materials[]) is only used
// by MaterialType::Subsurface's probe walk (identifying "same material"
// candidates - see bssrdf_probe_walk()'s own comment); every other case
// ignores it. `out_bssrdf_exit`/`out_bssrdf_exit_pos` are only ever set true
// by that same case, signalling to the caller (each geometry type's own
// closest-hit program) that the NEXT ray's origin must be this explicit
// world-space position - found via an off-ray probe walk - rather than the
// usual `hit_point + t_hit*ray_dir` reconstruction every other material
// (including the Medium family's own t-along-the-original-ray override)
// relies on. See optix_raygen.h's flag==3 handling.
__device__ __forceinline__ void shade_material(
	const MaterialData& mat,
	int matIdx,
	const float3& normal,
	const float3& ray_dir,
	const float3& hit_point,
	bool front_face,
	float uv_u, float uv_v,
	// World-space (unnormalized ok) surface tangent, i.e. the shape's own
	// dp/du at this point - real per-shape analytic/UV-derived value from
	// every caller (see each intersection file's own computation), used
	// ONLY by the 4 anisotropy-capable material kinds below to build a
	// UV-aligned shading frame via BuildDpduTangentFrame (matches CPU's
	// ShadingFrame::from_dpdu - see MaterialData::roughnessV's own comment
	// on why GPU previously used an arbitrary, non-UV-aligned frame
	// instead). Every other material kind ignores this parameter.
	const float3& dpdu,
	// Integrator "bool regularize" (pbrt-v4 default false) - already the
	// caller's "params.camera.regularize && anyNonSpecularBounces-so-far"
	// AND, matching CPU camera.h's `regularize && any_nonspecular` and
	// GPU-wavefront's `regularize && (bool)h.any_nonspecular` (see
	// wavefront_kernels.cu's own wf_glossy_alpha comment) exactly - only
	// the alpha-WIDENING at the 4 rough-material call sites below reads
	// this; anyNonSpecularBounces itself is tracked unconditionally in
	// optix_raygen.h's bounce loop regardless of this flag's value, per
	// pbrt-v4's own real semantics (only the widening-at-use-time gates).
	bool do_regularize,
	unsigned int& seed,
	float3& out_attenuation,
	float3& out_scattered_dir,
	bool& out_scattered,
	bool& out_is_specular,
	bool& out_is_medium_boundary,
	float& out_brdf_pdf_override,
	float3& emission,
	bool& out_bssrdf_exit,
	float3& out_bssrdf_exit_pos,
	float& out_eta,
	// pbrt-v4 dispersion (Cauchy formula): which of R/G/B this PATH is
	// confined to for its remaining lifetime, or kRgbChannelUnset (3) if no
	// dispersive event has happened yet. IN: whatever a PRIOR bounce's
	// dispersive hit already chose (persists via payload register p24 -
	// see optix_raygen.h's own rgbChannel comment). OUT: unchanged unless
	// THIS hit is the material's own FIRST dispersive event (a
	// MaterialType::Dielectric with mat.dispersive_extra.cauchy_A > 0 - see
	// that field's own comment, optix_types.h), in which case a channel is
	// picked here (once) and threaded back out. This is a deliberately
	// SIMPLER technique than CPU/wavefront's real 4-hero-wavelength
	// SampledWavelengths<4> Monte Carlo spectral integration (src/shared/
	// sampled_spectrum.h): rather than a continuous importance-sampled
	// wavelength + a full CIE-XYZ uplift at path end (which would need a
	// new device-constant-memory upload of the CIE tables into THIS
	// backend's own separate OptiX module/pipeline - recursive and
	// wavefront don't share device memory, see wavefront_kernels.cu's own
	// "separate module" precedent), this stochastically confines the WHOLE
	// path to exactly one of 3 fixed representative wavelengths (one per
	// RGB channel - see kRgbChannelWavelengthNm below) with a compensating
	// 3x reweight (optix_raygen.h), a coarser but far cheaper approximation
	// - no new device-memory uploads, one new payload register instead of
	// eight. Covers both MaterialType::Dielectric and MaterialType::
	// RoughDielectric (matching CPU dispersion's own scope) - RoughDielectric's
	// real NEE/MIS (rd_bxdf.pdf()/f() calls, this same switch) also threads
	// the same resolved dispersive ior through, for a consistent result.
	unsigned int& inout_rgb_channel
) {
	float3 nee_shadow_rgb = make_float3(1.0f, 1.0f, 1.0f);  // per-channel transmittance of the last NEE shadow ray through chromatic media (trace_shadow_ray_stochastic)
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	// True only for MaterialType::Interface - a real "nothing happened
	// here" signal, distinct from is_specular (a genuine delta/specular
	// BSDF event). See MaterialType::Interface's own comment (optix_types.h).
	bool is_medium_boundary = false;
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);
	// pbrt-v4 etaScale term for this event - see PathTracingPayload::eta's
	// own comment. 1.0f (a no-op) unless a case below sets it on a genuine
	// transmission through a dielectric interface.
	float eta = 1.0f;

	switch (mat.type) {
		case MaterialType::Lambertian: {
			// Multiple Importance Sampling (MIS) for diffuse surfaces

			// Sample BRDF (cosine-weighted hemisphere) for indirect lighting
			scattered_dir = normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = normal;
			}
			scattered_dir = normalize(scattered_dir);
			attenuation = (mat.textureIdx >= 0)
				? sample_texture(mat.textureIdx, uv_u, uv_v, hit_point) * mat.emissionScale
				: mat.albedo;
			scattered = true;

			// Add direct lighting via explicit light sampling (Next Event Estimation)
			{
				float3 to_light, sampled_light_emission; float max_dist, light_pdf;
				if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
					// Check if light is visible (shadow ray)
					bool visible = trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb);

					if (visible) {
						// Evaluate BRDF PDF for this direction
						float brdf_pdf = cosine_pdf(to_light, normal);

						// MIS weight using power heuristic
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

						// Emission of the light that was actually sampled -
						// looked up by the same helper that chose it, so the
						// two can't disagree about which primitive it was.
						const float3 light_emission = sampled_light_emission;

						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						// Uses `attenuation` (already texture-sampled above if
						// mat.textureIdx >= 0), NOT mat.albedo directly - for a
						// textured Lambertian (e.g. scenes 38-40's checkered
						// ground), mat.albedo is just a (1,1,1) white
						// placeholder (see line 611's own comment), so using it
						// here made every NEE-lit textured surface's direct
						// lighting term use full white reflectance regardless
						// of its actual (possibly much darker) texture color -
						// confirmed via a real, sample-count-independent ~1.5x
						// CPU/GPU brightness mismatch on scene 40's checkered
						// ground (a hard bias, not noise: identical ratio at
						// 100 spp and 2000 spp) that CPU didn't have (its own
						// NEE path already samples the checker texture, see
						// camera.h's `srec.attenuation`). Indirect/BSDF-sampled
						// bounces were never affected - they already read
						// `attenuation`, not mat.albedo.
						float cos_theta = fmaxf(0.0f, dot(to_light, normal));
						float3 brdf = attenuation / 3.14159265358979323846f;  // Lambertian BRDF
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						// Add to emission (raygen will apply throughput)
						emission = emission + direct_light * (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
					}
				}
			}

			// Direct lighting from punctual (point/spot/distant) lights -
			// deterministic delta lights, evaluated separately from the
			// area-light alias table above (see optix_device_helpers.h).
			// Same fix as the NEE term above: pass the already texture-
			// sampled `attenuation`, not the (1,1,1)-placeholder mat.albedo,
			// so a textured Lambertian under a punctual light isn't
			// incorrectly lit as if fully white.
			add_punctual_lights_lambertian(hit_point, normal, attenuation, emission);

			// Direct lighting from the sky (infinite light) - mirrors CPU's
			// camera.h Strategy A-2 (NEE toward the sky, MIS-weighted against
			// the BSDF-sampled bounce), which the GPU miss program
			// (optix_miss.h) never had: it only adds the background color when
			// a bounced ray happens to escape the scene entirely, with no
			// importance sampling toward it - fine for open scenes, but
			// starves interiors with small apertures (confirmed: Sibenik
			// Cathedral rendered 2.85x darker than CPU at matched settings
			// before this fix). sample_sky_nee() (optix_sky_light.h)
			// dispatches to real HDR importance sampling when the scene's
			// infinite light carries an image (params.camera.skyDist.height >
			// 0), falling back to this exact uniform-sphere sample otherwise -
			// same shape as sky_light::sample_Le()'s own two modes. Skipped
			// outright when backgroundColor is black (every scene without a
			// sky) - free, since NEE toward a zero-radiance light contributes
			// nothing.
			{
				const float3& skyColor = params.camera.backgroundColor;
				if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
					float3 sky_dir, sky_Le_val; float pdf_sky;
					sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
					float  cos_sky = dot(sky_dir, normal);
					// pdf_sky > 0.0f is REQUIRED, not just an optimization: a
					// portal-light NEE sample (sample_sky_nee() -> gpu_portal_
					// sample_Li()) returns pdf_sky == 0.0f with sky_Le_val ==
					// (0,0,0) whenever the portal window subtends zero area
					// from hit_point (a routine outcome for most points not
					// facing the window, not a rare edge case) - without this
					// check, `.../pdf_sky` below divides 0.0f by 0.0f, a real
					// NaN, not just a wasted no-op. Every other sky-NEE block
					// in this switch (and wf_finish_material_scatter's own
					// identical block, wavefront_kernels.cu) needs the exact
					// same guard - mirrors CPU's own
					// `if (portal->sample_li(...) && pdf_portal > 0.0)` gate
					// (src/TheRestOfYourLife/camera.h) and medium_phase_nee_
					// mis()'s own pdf_sky>0.0f check just above in this file.
					if (cos_sky > 0.0f && pdf_sky > 0.0f) {
						if (trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
							float brdf_pdf_sky = cosine_pdf(sky_dir, normal);
							float mis_weight    = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
							float3 brdf = attenuation / 3.14159265358979323846f;
							emission = emission + mis_weight * brdf * sky_Le_val * cos_sky / pdf_sky
								* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
						}
					}
				}
			}

			break;
		}

		case MaterialType::Metal: {
			// Same sampling as the shared MetalBxDF (CPU) and the wavefront backend: perturb the unit
			// reflection by fuzz times a point ON the unit sphere, and ABSORB the ray when that points
			// below the surface. This used to substitute the pure reflection instead, so a fuzzed metal
			// never lost the energy the others lose to those rays - a furnace plane of "conductor
			// reflectance 0.9 roughness 1" (which the loader builds as this fuzzed mirror) rendered
			// 0.90 here against 0.645 on CPU and wavefront, and Salle de Bain's metals came out ~3% bright.
			float3 reflected = normalize(reflect(normalize(ray_dir), normal));
			float3 fuzzed = reflected + mat.fuzz * random_unit_vector(seed);
			if (dot(fuzzed, normal) <= 0.0f) { scattered = false; break; }
			scattered_dir = normalize(fuzzed);
			attenuation = mat.albedo;
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::Measured: {
			// Real tabulated measured BRDF (pbrt-v4 Material "measured"): one VNDF-importance-sampled BxDF sample continues the path with
			// the weight f * |cos| / pdf, and, like the glossy Conductor below, the vertex also takes direct-light samples (area, punctual,
			// sky) with MIS against that sampling density, using the BxDF's own f() and pdf() (gpu_measured_f_pdf). It used to be a
			// specular-style bounce with no light sampling, which is unbiased but very noisy under a small light, and which never lit the
			// material from a point/spot/distant light at all (a delta light cannot be hit by BSDF sampling).
			if (mat.textureIdx < 0 || (unsigned int)mat.textureIdx >= params.numMeasuredTables) { scattered = false; break; }
			const GpuMeasuredTable& m_tab = params.measuredTables[mat.textureIdx];
			ShadingFrame<float> m_frame = ShadingFrame<float>::from_normal(normal.x, normal.y, normal.z);
			const float3 m_wo_world = normalize(-ray_dir);
			float m_wox, m_woy, m_woz;
			m_frame.to_local(m_wo_world.x, m_wo_world.y, m_wo_world.z, m_wox, m_woy, m_woz);
			// sRGB-primary approximations - the same fixed query wavelengths as material_pbrt.h's `measured` (kLambdaR/G/B)
			const float m_lambda[3] = { 612.0f, 549.0f, 465.0f };

			float m_wix, m_wiy, m_wiz, m_wr, m_wg, m_wb, m_spdf;
			const bool m_ok = gpu_measured_sample_f(m_tab, m_tab.isotropic != 0,
				params.measuredParamValues, params.measuredData, params.measuredMcdf, params.measuredCcdf,
				m_wox, m_woy, m_woz, random_float(seed), random_float(seed), m_lambda,
				m_wix, m_wiy, m_wiz, m_wr, m_wg, m_wb, m_spdf);
			// A rejected sample (grazing wo, zero pdf, a reflection below the horizon) must not skip this vertex's NEE (pbrt takes the
			// direct-light sample whether or not the sample that follows succeeds): it carries zero weight and the path ends after the NEE.
			attenuation = m_ok ? make_float3(m_wr, m_wg, m_wb) : make_float3(0.0f, 0.0f, 0.0f);
			if (m_ok) {
				float m_dx, m_dy, m_dz;
				m_frame.to_world(m_wix, m_wiy, m_wiz, m_dx, m_dy, m_dz);
				scattered_dir = normalize(make_float3(m_dx, m_dy, m_dz));
			} else {
				scattered_dir = normal;
			}
			scattered = true;
			is_specular = false;
			brdf_pdf_override = m_ok ? m_spdf : -1.0f;

			{
				float3 to_light, sampled_light_emission; float max_dist, light_pdf;
				if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
					float llx, lly, llz;
					m_frame.to_local(to_light.x, to_light.y, to_light.z, llx, lly, llz);
					float fr, fg, fb, brdf_pdf;
					if (llz > 0.0f && gpu_measured_f_pdf(m_tab, m_tab.isotropic != 0,
							params.measuredParamValues, params.measuredData, params.measuredMcdf, params.measuredCcdf,
							m_wox, m_woy, m_woz, llx, lly, llz, m_lambda, fr, fg, fb, brdf_pdf)
						&& trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb)) {
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
						emission = emission + mis_weight * make_float3(fr, fg, fb) * sampled_light_emission * llz / light_pdf
							* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
					}
				}
			}

			for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
				float3 wi_p, Li_p; float t_max_p;
				if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
				float plx, ply, plz;
				m_frame.to_local(wi_p.x, wi_p.y, wi_p.z, plx, ply, plz);
				if (plz <= 0.0f) continue;
				float fr, fg, fb, brdf_pdf;
				if (!gpu_measured_f_pdf(m_tab, m_tab.isotropic != 0,
						params.measuredParamValues, params.measuredData, params.measuredMcdf, params.measuredCcdf,
						m_wox, m_woy, m_woz, plx, ply, plz, m_lambda, fr, fg, fb, brdf_pdf)) continue;
				if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb)) {
					emission = emission + make_float3(fr, fg, fb) * Li_p * plz * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb);
				}
			}

			{
				const float3& skyColor = params.camera.backgroundColor;
				if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
					float3 sky_dir, sky_Le_val; float pdf_sky;
					sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
					float skx, sky_y, skz;
					m_frame.to_local(sky_dir.x, sky_dir.y, sky_dir.z, skx, sky_y, skz);
					float fr, fg, fb, brdf_pdf_sky;
					if (skz > 0.0f && pdf_sky > 0.0f
						&& gpu_measured_f_pdf(m_tab, m_tab.isotropic != 0,
							params.measuredParamValues, params.measuredData, params.measuredMcdf, params.measuredCcdf,
							m_wox, m_woy, m_woz, skx, sky_y, skz, m_lambda, fr, fg, fb, brdf_pdf_sky)
						&& trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
						float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
						emission = emission + mis_weight * make_float3(fr, fg, fb) * sky_Le_val * skz / pdf_sky
							* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
					}
				}
			}
			break;
		}

		case MaterialType::Dielectric: {
			// pbrt-v4 dispersion (Cauchy formula) - see shade_material()'s
			// own inout_rgb_channel parameter comment for the full
			// rationale/scope (smooth Dielectric only this round). Picks a
			// channel ONCE per path, lazily, at the FIRST dispersive hit -
			// every later dispersive hit along the SAME path (e.g. entering
			// then exiting the same glass object, or a second glass object)
			// reuses that already-chosen channel instead of re-rolling,
			// matching CPU/wavefront's own "one hero wavelength persists
			// for the whole path" convention. mat.dispersive_extra.cauchy_A
			// > 0.0f is the established sentinel (see that field's own
			// comment, optix_types.h) - 0 (the default) means "not
			// dispersive, use the flat mat.ior below".
			float dielectric_ior = mat.ior;
			if (mat.dispersive_extra.cauchy_A > 0.0f) {
				if (inout_rgb_channel == kRgbChannelUnset) {
					inout_rgb_channel = static_cast<unsigned int>(random_float(seed) * 3.0f);
					if (inout_rgb_channel > 2u) inout_rgb_channel = 2u;  // 3.0f*u can hit exactly 3.0f
				}
				dielectric_ior = CauchyEta(kRgbChannelWavelengthNm[inout_rgb_channel],
				                           mat.dispersive_extra.cauchy_A, mat.dispersive_extra.cauchy_B);
			}
			scattered_dir = dielectric_scatter(ray_dir, normal, front_face, dielectric_ior, seed);
			// `normal` here always satisfies dot(ray_dir, normal) < 0 (flipped
			// to face the incoming ray - see the front_face/final_normal
			// convention at each closesthit's call site). A reflected
			// direction bounces back to the ray_dir side (dot > 0); a
			// refracted direction continues through to the far side
			// (dot < 0, same sign as ray_dir's own). Tf tints transmission
			// only, matching real colored glass (a mirror-like reflection
			// off the surface doesn't pick up the pane's tint) - see
			// add_dielectric()'s transmission_filter comment.
			bool is_transmission = dot(scattered_dir, normal) < 0.0f;
			attenuation = is_transmission ? mat.transmission_filter : make_float3(1.0f, 1.0f, 1.0f);
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			// pbrt-v4 etaScale: eta = ri = front_face ? 1/ior : ior (same
			// formula dielectric_scatter() used internally, recomputed here
			// rather than threading it out of that function - see
			// PathTracingPayload::eta's own comment), only on a genuine
			// transmission event (matches CPU material_simple.h's
			// `res.eta = ri` only in the refract branch, `T(1)` otherwise).
			// Uses dielectric_ior (the dispersive-resolved value when
			// applicable), not the flat mat.ior, so RR's etaScale correction
			// stays consistent with the direction actually sampled above.
			if (is_transmission) eta = front_face ? (1.0f / dielectric_ior) : dielectric_ior;
			break;
		}

		case MaterialType::Interface: {
			// Real pass-through - exact same direction as the incoming ray,
			// no Fresnel/refraction math at all. See MaterialType::
			// Interface's own comment (optix_types.h).
			scattered_dir = ray_dir;
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			scattered = true;
			is_medium_boundary = true;
			break;
		}

		case MaterialType::Subsurface: {
			// Real tabulated BSSRDF (recursive backend only, Phase 1 - see
			// this MaterialType's own comment in optix_types.h and
			// bssrdf_probe_walk()'s comment above for the full algorithm).
			// Entry interface: the exact same smooth DielectricBxDF sample
			// as MaterialType::Dielectric above (matches pbrt-v4's own
			// SubsurfaceMaterial::GetBxDF / this codebase's CPU `class
			// subsurface`, material_pbrt.h - a plain DielectricBxDF(eta),
			// no roughness, no Tf tint).
			//
			// `emission` on entry is already zero (the caller reads it via
			// material_emission(), which guards the mat.emission union-slot
			// read behind mat.type == DiffuseLight - Subsurface has no real
			// emission field, that slot is bssrdf_sigma_a here, see
			// optix_types.h's MaterialType::Subsurface comment - so no reset
			// needed here anymore).
			scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
			bool is_transmission = dot(scattered_dir, normal) < 0.0f;
			// pbrt-v4 etaScale, entry interface only - matches CPU camera.h's
			// `entry_eta` (captured once here, before the exit-surface's own
			// NormalizedFresnel shading below, which does NOT independently
			// contribute another etaScale factor).
			if (is_transmission) eta = front_face ? (1.0f / mat.ior) : mat.ior;
			if (!is_transmission) {
				// Specular reflection off the entry interface - identical
				// to Dielectric's own reflection case.
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
				scattered   = true;
				is_specular = true;
				break;
			}

			// Transmission: attempt the probe/exit-point search. On
			// failure the path terminates with zero contribution, matching
			// camera.h's `if (!sample_bssrdf_exit(...)) break;`.
			float3 exit_pos, exit_normal, Sp;
			float pdf, sample_prob;
			if (!bssrdf_probe_walk(mat, matIdx, hit_point, normal, seed,
									exit_pos, exit_normal, Sp, pdf, sample_prob)) {
				scattered = false;
				break;
			}
			const float3 sss_weight = Sp / (sample_prob * pdf);

			// Shade the exit point as pbrt-v4's Sw (NormalizedFresnel(eta))
			// via shade_normalized_fresnel() - the same standalone function
			// MaterialType::NormalizedFresnel's own case calls, NOT a
			// nested shade_material() call: OptiX's module compiler
			// statically rejects any self-recursive call graph (see that
			// function's own comment for the exact compile error), even
			// though the actual call depth here would have been provably
			// bounded to one extra level. This still exactly mirrors
			// camera.h's own hand-off design (rec.mat swapped to
			// ss->get_exit_bsdf(), then that material's own scatter()
			// called once) - just with the shared logic factored into its
			// own function instead of reached by re-entering the switch.
			// shade_normalized_fresnel() always scatters (no failure case).
			float3 exit_atten, exit_dir;
			float exit_pdf_override;
			shade_normalized_fresnel(mat.ior, exit_normal, exit_pos, seed,
				exit_atten, exit_dir, exit_pdf_override, emission);

			// emission was just accumulated (in place) by that call's
			// own NEE terms, evaluated at exit_pos with the OLD (pre-this-
			// bounce) throughput - but those terms didn't know about
			// sss_weight, so scale the just-added contribution by it,
			// leaving whatever emission already held (mat.emission, always
			// zero for Subsurface) untouched. Mirrors camera.h's own
			// ordering: sample_bssrdf_exit() multiplies `beta` by
			// entry_attenuation*Sp*inv BEFORE the exit bounce's own NEE
			// (inside ray_color's main loop) ever runs, so that NEE already
			// sees the weighted throughput - here, where entry+exit shading
			// happen in one call, the weighting has to be applied after
			// the fact instead.
			emission = sss_weight * emission;

			attenuation    = sss_weight * exit_atten;
			scattered_dir  = exit_dir;
			scattered      = true;
			is_specular    = false;  // NormalizedFresnel participates in MIS
			brdf_pdf_override = exit_pdf_override;

			// The next ray must originate at exit_pos (found off to the
			// side via the probe walk), not along this hit's own ray - see
			// shade_material()'s own comment on out_bssrdf_exit/
			// out_bssrdf_exit_pos and optix_raygen.h's flag==3 handling.
			bssrdf_exit     = true;
			bssrdf_exit_pos = exit_pos;
			break;
		}

		case MaterialType::ThinDielectric: {
			// Zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF) -- see
			// thin_dielectric_scatter()'s own comment (above) for the full
			// derivation - factored out so MaterialType::DielectricMedium's
			// own thin-fused entry/exit boundary can reuse it exactly.
			attenuation   = make_float3(1.0f, 1.0f, 1.0f);
			scattered_dir = thin_dielectric_scatter(ray_dir, normal, mat.ior, seed);
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::CoatedConductor: {
			// Dielectric coat over a GGX conductor (pbrt-v4 CoatedConductorBxDF) - see layered_scatter_and_nee().
			// mat.fuzz/mat.roughnessV are the COAT's roughness (mat.roughnessV<0 means isotropic - see
			// MaterialData::roughnessV); the conductor has its own (mat.condRoughness/condRoughnessV, negative =
			// the coat's). As in pbrt's CoatedConductorMaterial::GetBxDF the conductor's complex IOR is relative to
			// the coat: eta and k are both divided by the coat's IOR.
			float cc_alpha_x = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;
			float cc_alpha_y = ResolveAnisotropicAlphaV(mat.roughnessV, mat.fuzz, mat.remapRoughness);
			float cc_cond_x, cc_cond_y;
			ResolveCoatedConductorBaseAlpha(mat.condRoughness, mat.condRoughnessV, mat.remapRoughness,
			                                cc_alpha_x, cc_alpha_y, cc_cond_x, cc_cond_y);
			if (do_regularize) {
				cc_alpha_x = RegularizeAlpha(cc_alpha_x);
				cc_alpha_y = RegularizeAlpha(cc_alpha_y);
				cc_cond_x  = RegularizeAlpha(cc_cond_x);
				cc_cond_y  = RegularizeAlpha(cc_cond_y);
			}
			const float cc_inv_ior = 1.0f / mat.ior;
			CoatedConductorBxDF<float> cc_bxdf{ mat.eta_c.x * cc_inv_ior, mat.eta_c.y * cc_inv_ior, mat.eta_c.z * cc_inv_ior,
			                                    mat.k_c.x * cc_inv_ior, mat.k_c.y * cc_inv_ior, mat.k_c.z * cc_inv_ior,
			                                    mat.ior, cc_alpha_x, cc_alpha_y, mat.layerThickness, 0.0f, 0.0f, 10, 1,
			                                    cc_cond_x, cc_cond_y };
			if (!layered_scatter_and_nee(cc_bxdf, normal, ray_dir, hit_point, dpdu, seed,
			                             attenuation, scattered_dir, is_specular, brdf_pdf_override, emission)) {
				scattered = false;
				break;
			}
			scattered = true;
			break;
		}

		case MaterialType::RoughDielectric: {
			// GGX microfacet BSDF (pbrt-v4 RoughDielectricBxDF) - factored
			// into rough_dielectric_scatter_and_nee() (above) so MaterialType::
			// DielectricMedium's own fused rough case can call the exact
			// same logic - see that function's own header comment for the
			// full derivation this case used to carry inline. flatRoughness
			// is mat.fuzz here (this type's own real roughness field, unlike
			// DielectricMedium's borrowed slot); allowDispersion is true
			// (this type's own dispersive_extra is a real, intentional
			// field, unlike DielectricMedium's aliased one).
			if (!rough_dielectric_scatter_and_nee(
					mat, mat.fuzz, /*allowDispersion=*/true,
					normal, ray_dir, hit_point, front_face, uv_u, uv_v, dpdu,
					do_regularize, seed, inout_rgb_channel,
					attenuation, scattered_dir, is_specular, brdf_pdf_override, emission, eta)) {
				scattered = false;
				break;
			}
			scattered = true;
			break;
		}

		case MaterialType::Conductor: {
			// GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF) -- sphere version
			// mat.roughnessV<0 means "isotropic" - see MaterialData::
			// roughnessV's own comment (optix_types.h).
			float c_alpha_x = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;
			float c_alpha_y = ResolveAnisotropicAlphaV(mat.roughnessV, mat.fuzz, mat.remapRoughness);
			if (do_regularize) {
				c_alpha_x = RegularizeAlpha(c_alpha_x);
				c_alpha_y = RegularizeAlpha(c_alpha_y);
			}
			float3 cn = normal;
			float3 ctan, cbitan;
			BuildDpduTangentFrame(cn.x, cn.y, cn.z, dpdu.x, dpdu.y, dpdu.z, ctan.x, ctan.y, ctan.z, cbitan.x, cbitan.y, cbitan.z);
			float3 cwi = normalize(-ray_dir);
			float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
			if (cwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> c_dist(c_alpha_x, c_alpha_y);
			float cwm_x, cwm_y, cwm_z;
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, random_float(seed), random_float(seed), cwm_x, cwm_y, cwm_z);
			float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
			float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
			float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
			// A rejected continuation sample (a reflection that lands below the horizon) must not skip this vertex's NEE - pbrt takes the
			// direct-light sample at every vertex whether or not the sample that follows succeeds. It carries zero weight and the path
			// ends after the NEE below; the effectively-smooth lobe has no NEE and keeps the old early exit.
			const bool c_rej = (cwo_z <= 0.0f);
			if (c_rej && c_dist.EffectivelySmooth()) { scattered = false; break; }
			float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
			float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
			float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
			float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
			attenuation = c_rej ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight);
			scattered_dir = c_rej ? cn : normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
			scattered     = true;

			// Real NEE/MIS for glossy (non-EffectivelySmooth) conductors,
			// mirroring the Lambertian NEE block above but evaluated in the
			// local (tangent) frame via the shared ConductorBxDF<float>'s
			// real f()/pdf() (src/shared/bxdfs_conductor.h) - the same
			// CPU_GPU template already verified on CPU for #222. The
			// existing VNDF-sampled `attenuation`/`scattered_dir` above are
			// left untouched (self-normalizing G/G1 weight, mathematically
			// equivalent to f()*cos/pdf() - see conductor::scatter() in
			// material_pbrt.h for the identity this relies on); this block
			// only adds direct-light contributions and switches off the
			// specular MIS-skip so future bounces off area/sky lights get
			// properly weighted.
			if (!c_dist.EffectivelySmooth()) {
				is_specular = false;
				ConductorBxDF<float> c_bxdf{ mat.eta_c.x, mat.eta_c.y, mat.eta_c.z,
											  mat.k_c.x, mat.k_c.y, mat.k_c.z,
											  c_alpha_x, c_alpha_y };
				brdf_pdf_override = c_rej ? -1.0f : c_bxdf.pdf(cwi_x, cwi_y, cwi_z, cwo_x, cwo_y, cwo_z);

				{
					float3 to_light, sampled_light_emission; float max_dist, light_pdf;
					if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
						float llx = dot(to_light, ctan), lly = dot(to_light, cbitan), llz = dot(to_light, cn);
						if (llz > 0.0f && trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb)) {
							float fr, fg, fb;
							c_bxdf.f(cwi_x, cwi_y, cwi_z, llx, lly, llz, fr, fg, fb);
							float brdf_pdf = c_bxdf.pdf(cwi_x, cwi_y, cwi_z, llx, lly, llz);
							float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
							emission = emission + mis_weight * make_float3(fr, fg, fb) * sampled_light_emission * llz / light_pdf
								* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
						}
					}
				}

				for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
					float3 wi_p, Li_p; float t_max_p;
					if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
					float plx = dot(wi_p, ctan), ply = dot(wi_p, cbitan), plz = dot(wi_p, cn);
					if (plz <= 0.0f) continue;
					if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb)) {
						float fr, fg, fb;
						c_bxdf.f(cwi_x, cwi_y, cwi_z, plx, ply, plz, fr, fg, fb);
						emission = emission + make_float3(fr, fg, fb) * Li_p * plz * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb);
					}
				}

				{
					const float3& skyColor = params.camera.backgroundColor;
					if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
						float3 sky_dir, sky_Le_val; float pdf_sky;
						sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
						float skx = dot(sky_dir, ctan), sky_y = dot(sky_dir, cbitan), skz = dot(sky_dir, cn);
						if (skz > 0.0f && pdf_sky > 0.0f && trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
							float fr, fg, fb;
							c_bxdf.f(cwi_x, cwi_y, cwi_z, skx, sky_y, skz, fr, fg, fb);
							float brdf_pdf_sky = c_bxdf.pdf(cwi_x, cwi_y, cwi_z, skx, sky_y, skz);
							float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
							emission = emission + mis_weight * make_float3(fr, fg, fb) * sky_Le_val * skz / pdf_sky
								* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
						}
					}
				}
			} else {
				is_specular = true;
			}
			break;
		}

		case MaterialType::RoughMetal: {
			// GGX VNDF + flat-tint reflectance, no complex Fresnel (pbrt-v4/
			// RTOW rough_metal) -- sphere version. Same shape as
			// MaterialType::Conductor above, minus FrConductorRGB (RoughMetalBxDF
			// has no real Fresnel model - see src/shared/bxdfs_conductor.h's own
			// header comment and material_pbrt.h's `rough_metal`, which this
			// mirrors exactly). NOT the same material as MaterialType::Metal
			// (a fuzz-perturbed mirror, a different model entirely - CPU's
			// plain `metal` class).
			float rm_alpha = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;  // pbrt-v4 remaproughness (see MaterialData::remapRoughness)
			if (do_regularize) rm_alpha = RegularizeAlpha(rm_alpha);
			float3 rmn = normal;
			float3 rmup = (fabsf(rmn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 rmtan   = normalize(cross(rmup, rmn));
			float3 rmbitan = cross(rmn, rmtan);
			float3 rmwi = normalize(-ray_dir);
			float rmwi_x = dot(rmwi, rmtan), rmwi_y = dot(rmwi, rmbitan), rmwi_z = dot(rmwi, rmn);
			if (rmwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> rm_dist(rm_alpha, rm_alpha);
			float rmwm_x, rmwm_y, rmwm_z;
			rm_dist.Sample_wm(rmwi_x, rmwi_y, rmwi_z, random_float(seed), random_float(seed), rmwm_x, rmwm_y, rmwm_z);
			float rm_dot = rmwi_x*rmwm_x + rmwi_y*rmwm_y + rmwi_z*rmwm_z;
			float rmwo_x = 2.0f*rm_dot*rmwm_x - rmwi_x;
			float rmwo_y = 2.0f*rm_dot*rmwm_y - rmwi_y;
			float rmwo_z = 2.0f*rm_dot*rmwm_z - rmwi_z;
			// A rejected continuation sample (a reflection that lands below the horizon) must not skip this vertex's NEE - pbrt takes the
			// direct-light sample at every vertex whether or not the sample that follows succeeds. It carries zero weight and the path
			// ends after the NEE below; the effectively-smooth lobe has no NEE and keeps the old early exit.
			const bool rm_rej = (rmwo_z <= 0.0f);
			if (rm_rej && rm_dist.EffectivelySmooth()) { scattered = false; break; }
			float rm_G1_wi  = rm_dist.G1(rmwi_x, rmwi_y, rmwi_z);
			float rm_G_wowi = rm_dist.G(rmwo_x, rmwo_y, rmwo_z, rmwi_x, rmwi_y, rmwi_z);
			float rm_weight = (rm_G1_wi > 1e-8f) ? rm_G_wowi / rm_G1_wi : 0.0f;
			attenuation = rm_rej ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(mat.albedo.x * rm_weight, mat.albedo.y * rm_weight, mat.albedo.z * rm_weight);
			scattered_dir = rm_rej ? rmn : normalize(rmwo_x*rmtan + rmwo_y*rmbitan + rmwo_z*rmn);
			scattered     = true;

			// Real NEE/MIS for glossy (non-EffectivelySmooth) rough metal -
			// see MaterialType::Conductor's identical-shape block above for
			// the full rationale.
			if (!rm_dist.EffectivelySmooth()) {
				is_specular = false;
				RoughMetalBxDF<float> rm_bxdf{ mat.albedo.x, mat.albedo.y, mat.albedo.z, rm_alpha, rm_alpha };
				brdf_pdf_override = rm_rej ? -1.0f : rm_bxdf.pdf(rmwi_x, rmwi_y, rmwi_z, rmwo_x, rmwo_y, rmwo_z);

				{
					float3 to_light, sampled_light_emission; float max_dist, light_pdf;
					if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
						float llx = dot(to_light, rmtan), lly = dot(to_light, rmbitan), llz = dot(to_light, rmn);
						if (llz > 0.0f && trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb)) {
							float fr, fg, fb;
							rm_bxdf.f(rmwi_x, rmwi_y, rmwi_z, llx, lly, llz, fr, fg, fb);
							float brdf_pdf = rm_bxdf.pdf(rmwi_x, rmwi_y, rmwi_z, llx, lly, llz);
							float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
							emission = emission + mis_weight * make_float3(fr, fg, fb) * sampled_light_emission * llz / light_pdf
								* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
						}
					}
				}

				for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
					float3 wi_p, Li_p; float t_max_p;
					if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
					float plx = dot(wi_p, rmtan), ply = dot(wi_p, rmbitan), plz = dot(wi_p, rmn);
					if (plz <= 0.0f) continue;
					if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb)) {
						float fr, fg, fb;
						rm_bxdf.f(rmwi_x, rmwi_y, rmwi_z, plx, ply, plz, fr, fg, fb);
						emission = emission + make_float3(fr, fg, fb) * Li_p * plz * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb);
					}
				}

				{
					const float3& skyColor = params.camera.backgroundColor;
					if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
						float3 sky_dir, sky_Le_val; float pdf_sky;
						sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
						float skx = dot(sky_dir, rmtan), sky_y = dot(sky_dir, rmbitan), skz = dot(sky_dir, rmn);
						if (skz > 0.0f && pdf_sky > 0.0f && trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
							float fr, fg, fb;
							rm_bxdf.f(rmwi_x, rmwi_y, rmwi_z, skx, sky_y, skz, fr, fg, fb);
							float brdf_pdf_sky = rm_bxdf.pdf(rmwi_x, rmwi_y, rmwi_z, skx, sky_y, skz);
							float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
							emission = emission + mis_weight * make_float3(fr, fg, fb) * sky_Le_val * skz / pdf_sky
								* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
						}
					}
				}
			} else {
				is_specular = true;
			}
			break;
		}

		case MaterialType::CoatedDiffuse: {
			// Dielectric coat over a Lambertian base (pbrt-v4 CoatedDiffuseBxDF) - see layered_scatter_and_nee().
			// "reflectance" bound to a real Texture (pbrt's own ganesha/barcelona-pavilion "texture reflectance" -
			// see pbrt_flatten::Material::textureFilename's own comment) instead of mat.albedo's flat colour when
			// textureIdx>=0 - scaled by mat.emissionScale, reused here for CoatedDiffuse's own "scale"-wrapped-
			// imagemap case (see that field's own comment in optix_types.h). mat.roughnessV<0 means "isotropic" -
			// see MaterialData::roughnessV's own comment (optix_types.h).
			const float3 cd_albedo = (mat.textureIdx >= 0)
				? sample_texture(mat.textureIdx, uv_u, uv_v, hit_point) * mat.emissionScale
				: mat.albedo;
			float cd_alpha_x = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;
			float cd_alpha_y = ResolveAnisotropicAlphaV(mat.roughnessV, mat.fuzz, mat.remapRoughness);
			if (do_regularize) {
				cd_alpha_x = RegularizeAlpha(cd_alpha_x);
				cd_alpha_y = RegularizeAlpha(cd_alpha_y);
			}
			CoatedDiffuseBxDF<float> cd_bxdf{ cd_albedo.x, cd_albedo.y, cd_albedo.z, mat.ior, cd_alpha_x, cd_alpha_y };
			if (!layered_scatter_and_nee(cd_bxdf, normal, ray_dir, hit_point, dpdu, seed,
			                             attenuation, scattered_dir, is_specular, brdf_pdf_override, emission)) {
				scattered = false;
				break;
			}
			scattered = true;
			break;
		}

		case MaterialType::DiffuseTransmission: {
			// pbrt-v4 DiffuseTransmissionBxDF -- sphere version
			// albedo = reflectance R (same hemisphere), emission = transmittance T (reused field)
			// Real per-point value when texture-bound (barcelona-pavilion's
			// foliage - see pbrt_flatten::Material::textureFilename/
			// transmittanceTextureFilename's own comments), else the flat
			// mat.albedo/mat.emission fallback - uv_u/uv_v/hit_point are
			// already in scope in this function (same pattern CoatedDiffuse's
			// own case a few switch-arms away uses).
			float3 R = (mat.textureIdx >= 0)
				? sample_texture(mat.textureIdx, uv_u, uv_v, hit_point) * mat.emissionScale
				: mat.albedo;
			float3 T_col = (mat.transmittanceTextureIdx >= 0)
				? sample_texture(mat.transmittanceTextureIdx, uv_u, uv_v, hit_point) * mat.transmittanceScale
				: mat.emission;
			// `emission` (the OUT parameter, not T_col above) is already
			// zero on entry - the caller reads it via material_emission(),
			// which guards the mat.emission union-slot read behind
			// mat.type == DiffuseLight (see that function's own comment).
			// This material has no real emission field either: that union
			// slot is T (transmittance), which is exactly T_col just read
			// above.
			float pr = fmaxf(R.x, fmaxf(R.y, R.z));
			float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
			if (pr + pt <= 0.0f) { scattered = false; break; }

			// pbrt-v4 samples reflection with probability pr / (pr + pt) (else transmission) from a cosine lobe, so the path weight
			// f * cos / pdf = R * (cos / pi) / (pr / (pr + pt) * cos / pi) = R / p_lobe. The weight used to be R alone: a 0.2 / 0.6 surface read a
			// quarter of its reflection and a third of its transmission, on both GPU backends.
			const float p_refl = pr / (pr + pt);
			const bool dt_reflect = random_float(seed) < p_refl;
			if (dt_reflect) {
				// Diffuse reflection: cosine-weighted same hemisphere
				scattered_dir = normalize(normal + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = normal;
				attenuation   = R / p_refl;
			} else {
				// Diffuse transmission: cosine-weighted opposite hemisphere
				float3 neg_n  = -normal;
				scattered_dir = normalize(neg_n + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = neg_n;
				attenuation   = T_col / (1.0f - p_refl);
			}
			scattered   = true;

			// Direct-light sampling on BOTH sides of the surface, with MIS - pbrt-v4's DiffuseTransmissionBxDF is R/pi for a light on wo's side and T/pi
			// for one on the far side. The BSDF is chosen between the two cosine lobes with probabilities p_refl and 1 - p_refl, so the density of a
			// direction wi is p_lobe(side of wi) * |cos| / pi. This used to be a BSDF-only estimator (flagged is_specular, no light sampling at all):
			// unbiased for an area light a bounce can hit, but a point, spot or distant light can never be hit, so a diffuse-transmission surface lit
			// only by one rendered black on both GPU backends.
			is_specular = false;
			{
				const float kInvPi = 0.31830988618379067f;
				const float cos_s = fabsf(dot(scattered_dir, normal));
				brdf_pdf_override = (dt_reflect ? p_refl : 1.0f - p_refl) * cos_s * kInvPi;
				float3 nee_shadow_rgb_dt = make_float3(1.0f, 1.0f, 1.0f);

				{
					float3 to_light, sampled_light_emission; float max_dist, light_pdf;
					if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
						const float llz = dot(to_light, normal);
						if (llz != 0.0f && trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb_dt)) {
							const float3 f_l = ((llz > 0.0f) ? R : T_col) * kInvPi;
							const float brdf_pdf = ((llz > 0.0f) ? p_refl : 1.0f - p_refl) * fabsf(llz) * kInvPi;
							const float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
							emission = emission + mis_weight * f_l * sampled_light_emission * fabsf(llz) / light_pdf
								* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb_dt);
						}
					}
				}

				for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
					float3 wi_p, Li_p; float t_max_p;
					if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
					const float plz = dot(wi_p, normal);
					if (plz == 0.0f) continue;
					if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb_dt)) {
						const float3 f_p = ((plz > 0.0f) ? R : T_col) * kInvPi;
						emission = emission + f_p * Li_p * fabsf(plz) * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb_dt);
					}
				}

				{
					const float3& skyColor = params.camera.backgroundColor;
					if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
						float3 sky_dir, sky_Le_val; float pdf_sky;
						sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
						const float skz = dot(sky_dir, normal);
						if (skz != 0.0f && pdf_sky > 0.0f && trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb_dt)) {
							const float3 f_s = ((skz > 0.0f) ? R : T_col) * kInvPi;
							const float brdf_pdf_sky = ((skz > 0.0f) ? p_refl : 1.0f - p_refl) * fabsf(skz) * kInvPi;
							const float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
							emission = emission + mis_weight * f_s * sky_Le_val * fabsf(skz) / pdf_sky
								* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb_dt);
						}
					}
				}
			}
			break;
		}

		case MaterialType::NormalizedFresnel: {
			// pbrt-v4 NormalizedFresnelBxDF - see shade_normalized_fresnel()
			// above (factored out so MaterialType::Subsurface's exit point
			// can shade with the exact same code without a self-recursive
			// call into shade_material() itself, which OptiX's module
			// compiler statically rejects - see that function's own
			// comment). Diffuse BSDF: participates in MIS (is_specular=false).
			float3 nf_atten, nf_dir;
			float nf_pdf_override;
			shade_normalized_fresnel(mat.ior, normal, hit_point, seed,
				nf_atten, nf_dir, nf_pdf_override, emission);
			attenuation = nf_atten;
			scattered_dir = nf_dir;
			brdf_pdf_override = nf_pdf_override;
			scattered = true;
			is_specular = false;
			break;
		}
		case MaterialType::DiffuseLight: {
			// Emissive material - no scattering
			scattered = false;
			break;
		}

		// Everything else either (a) is one of the 8 MaterialTypes never
		// meant to reach this switch - each special-cased earlier, before
		// shade_material() is even called, and documented as such on its own
		// MaterialType enumerator (optix_types.h): Medium/DielectricMedium/
		// CloudMedium/RgbGridMedium/GridMedium (participating media, handled
		// inline in optix_intersection_sphere.h's shape-specific near/far
		// re-intersection), Hair (HairBxDF sampled directly, see
		// sample_principled_material()'s sibling dispatch above this
		// function), Principled (same sibling dispatch, PrincipledBxDF),
		// NormalMappedLambertian (perturbs the normal in
		// optix_intersection_sphere.h then re-enters THIS switch's own
		// Lambertian case with a temporary MaterialType::Lambertian view) -
		// or (b) a genuinely new MaterialType nobody wired into this switch
		// yet. Either way it's a real bug, and the compiler can't catch it
		// for us: nvcc's device-code frontend (confirmed empirically against
		// this project's CUDA 13.2 toolchain, including every --diag-warn
		// flag it exposes) does not implement switch/enum exhaustiveness
		// diagnostics the way MSVC's C4062 or clang's -Wswitch do, so a
		// missing case here compiles silently clean. Trap loudly instead of
		// absorbing the ray, so the gap surfaces the moment a real render
		// exercises it rather than being found reactively later (this
		// project's own history, more than once). Not gated behind NDEBUG -
		// nvcc's own invocation for this file never defines it either way
		// (see build_optix.targets), so the guard is simply always on.
		default: {
			printf("[SHADE-MATERIAL] unhandled MaterialType %d\n", (int)mat.type);
			__trap();
		}
	}

	out_attenuation = attenuation;
	out_scattered_dir = scattered_dir;
	out_scattered = scattered;
	out_is_specular = is_specular;
	out_is_medium_boundary = is_medium_boundary;
	out_brdf_pdf_override = brdf_pdf_override;
	out_bssrdf_exit = bssrdf_exit;
	out_bssrdf_exit_pos = bssrdf_exit_pos;
	out_eta = eta;
}
