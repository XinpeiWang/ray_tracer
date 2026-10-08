#pragma once
// optix_device_scatter_helpers.h -- part 3 of 5 of optix_device_helpers.h (included by it, in order; not meant to be included on its own).

// pbrt-v4 NormalizedFresnelBxDF (MaterialType::NormalizedFresnel's own
// logic, factored out to a standalone function so MaterialType::
// Subsurface's probe-walk exit point (bssrdf_probe_walk() below /
// shade_material()'s own Subsurface case) can shade with it too WITHOUT
// calling shade_material() itself a second time - OptiX's module compiler
// statically rejects any self-recursive call graph outright ("COMPILE
// ERROR: Malformed input... Found call graph recursion involving
// shade_material") even when the actual runtime call depth is providably
// bounded to one extra level (a different mat.type reaching a different
// switch case) - the module simply fails to compile at OptiX pipeline
// creation time, not a soft warning. This function is called from BOTH
// shade_material()'s own NormalizedFresnel case (unchanged behavior) and
// its Subsurface case (the new caller), so there is exactly one
// implementation of pbrt-v4's Sw exit BSDF, just no longer reached via a
// recursive shade_material() call.
//
// Always scatters (NormalizedFresnel has no failure/absorption case), so
// there is no out_scattered - unlike shade_material() itself.
__device__ __forceinline__ void shade_normalized_fresnel(
	float eta,
	const float3& normal,
	const float3& hit_point,
	unsigned int& seed,
	float3& out_attenuation,
	float3& out_scattered_dir,
	float& out_brdf_pdf_override,
	float3& emission)
{
	float3 nee_shadow_rgb = make_float3(1.0f, 1.0f, 1.0f);  // per-channel transmittance of the last NEE shadow ray through chromatic media (trace_shadow_ray_stochastic)
	float nf_eta = eta;
	float inv_eta = 1.0f / nf_eta;
	float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
	if (nf_c <= 0.0f) nf_c = 1e-6f;

	float3 scattered_dir = normalize(normal + random_unit_vector(seed));
	if (near_zero(scattered_dir)) scattered_dir = normal;

	float cos_wi = fmaxf(dot(scattered_dir, normal), 1e-6f);
	float fr     = FrDielectric(cos_wi, nf_eta);
	float weight = (1.0f - fr) / nf_c;
	float3 attenuation = make_float3(weight, weight, weight);
	// p12: correct BSDF PDF for MIS on next bounce: (1-Fr)*cos/(c*pi)
	float brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265358979323846f);

	{
		float3 to_light, light_emission; float max_dist, light_pdf;
		if (sample_nee_light(hit_point, seed, to_light, light_emission, max_dist, light_pdf, optixGetRayTime())) {
			bool visible = trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb);
			if (visible) {
				float cos_to_light = fmaxf(dot(to_light, normal), 0.0f);
				if (cos_to_light > 0.0f) {
					float fr_l  = FrDielectric(cos_to_light, nf_eta);
					float brdf_val = (1.0f - fr_l) / (nf_c * 3.14159265358979323846f);
					float brdf_pdf_l = brdf_val * cos_to_light;
					float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf_l);

					float3 direct_light = mis_weight * brdf_val * light_emission * cos_to_light / light_pdf;
					emission = emission + direct_light * (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
				}
			}
		}
	}

	{
		const float3& skyColor = params.camera.backgroundColor;
		if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
			// sample_sky_nee() (optix_sky_light.h) - see shade_material()'s
			// own Lambertian sky-NEE block for the full comment.
			float3 sky_dir, sky_Le_val; float pdf_sky;
			sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
			float  cos_sky = dot(sky_dir, normal);
			if (cos_sky > 0.0f && pdf_sky > 0.0f) {
				if (trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
					float fr_sky       = FrDielectric(cos_sky, nf_eta);
					float brdf_val_sky = (1.0f - fr_sky) / (nf_c * 3.14159265358979323846f);
					float brdf_pdf_sky = brdf_val_sky * cos_sky;
					float mis_weight    = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
					emission = emission + mis_weight * brdf_val_sky * sky_Le_val * cos_sky / pdf_sky
						* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
				}
			}
		}
	}

	out_attenuation = attenuation;
	out_scattered_dir = scattered_dir;
	out_brdf_pdf_override = brdf_pdf_override;
}

// MaterialType::Subsurface's probe/exit-point search - the GPU port of
// src/TheRestOfYourLife/camera.h::sample_bssrdf_exit(), replicating its
// exact 3-axis MIS algorithm (see that function's own extensive comment for
// the full derivation): pick an RGB channel uniformly for importance-
// sampling the radius only, sample r/r_max from that channel's profile,
// pick one of 3 orthonormal probe axes (shading normal, weight 0.5; two
// tangents, weight 0.25 each), build a probe segment (disc-sampled point +-
// half_len along the chosen axis), walk the segment via repeated
// trace_probe_ray() calls collecting same-material candidates via
// unweighted reservoir sampling (Algorithm R - pbrt-v4's GPU
// __raygen__randomHit does the identical bounded loop of sequential
// optixTrace() calls, see this codebase's own research notes on that
// function), then compute Sp (evaluated at the full 3D entry-to-exit
// distance) and the combined (one-sample MIS, balance heuristic) pdf summed
// over all 3 axes exactly as camera.h does.
//
// `matIdx` is this material's own index into params.materials[] (shade_
// material() doesn't otherwise know it - see its own new parameter) - the
// GPU equivalent of camera.h's `rec.mat.get()` pointer-identity check
// against a probe candidate's own material.
//
// bounded to kMaxProbeSteps segment-walk iterations (matches pbrt-v4's GPU
// __raygen__randomHit's own 100-iteration cap) so a pathological segment
// (e.g. many thin, closely-stacked same-material faces) cannot turn into an
// unbounded device loop.
__device__ __forceinline__ bool bssrdf_probe_walk(
	const MaterialData& mat, int matIdx,
	const float3& p0, const float3& axis0,
	unsigned int& seed,
	float3& out_exit_pos, float3& out_exit_normal,
	float3& out_Sp, float& out_pdf, float& out_sample_prob)
{
	constexpr int kMaxProbeSteps = 100;
	const float kPi = 3.14159265358979323846f;

	const GpuBssrdfTable& table = params.bssrdfTables[mat.textureIdx];
	const float sigma_a[3] = { mat.bssrdf_sigma_a.x, mat.bssrdf_sigma_a.y, mat.bssrdf_sigma_a.z };
	const float sigma_s[3] = { mat.bssrdf_sigma_s.x, mat.bssrdf_sigma_s.y, mat.bssrdf_sigma_s.z };
	float sigma_t[3], rho[3];
	for (int c = 0; c < 3; ++c) {
		sigma_t[c] = sigma_a[c] + sigma_s[c];
		rho[c] = (sigma_t[c] > 0.0f) ? (sigma_s[c] / sigma_t[c]) : 0.0f;
	}

	const float3 axis = normalize(axis0);
	const float3 t1 = (fabsf(axis.x) > 0.9f) ? normalize(cross(make_float3(0, 1, 0), axis))
											  : normalize(cross(make_float3(1, 0, 0), axis));
	const float3 t2 = cross(axis, t1);

	const int channel = min(2, (int)(random_float(seed) * 3.0f));
	const float r = gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], random_float(seed));
	if (r < 0.0f) return false;
	const float r_max = gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], 0.999f);
	if (r_max <= 0.0f || r >= r_max) return false;

	const float phi = 2.0f * kPi * random_float(seed);
	const float half_len = sqrtf(fmaxf(0.0f, r_max * r_max - r * r));

	const float u_axis = random_float(seed);
	float3 probe_axis, basis_a, basis_b;
	if (u_axis < 0.5f)       { probe_axis = axis; basis_a = t1;   basis_b = t2; }
	else if (u_axis < 0.75f) { probe_axis = t1;   basis_a = t2;   basis_b = axis; }
	else                     { probe_axis = t2;   basis_a = axis; basis_b = t1; }

	const float3 p_target = p0 + r * (cosf(phi) * basis_a + sinf(phi) * basis_b);
	const float3 p_start  = p_target - half_len * probe_axis;
	const float3 p_end    = p_target + half_len * probe_axis;

	float3 seg_dir = p_end - p_start;
	float seg_len = length(seg_dir);
	if (seg_len < 1e-10f) return false;
	seg_dir = seg_dir / seg_len;

	float3 chosen_pos = make_float3(0.0f, 0.0f, 0.0f);
	float3 chosen_normal = make_float3(0.0f, 0.0f, 0.0f);
	int candidate_count = 0;
	float3 base = p_start;
	float remaining = seg_len;
	for (int iter = 0; iter < kMaxProbeSteps && remaining > 1e-6f; ++iter) {
		ProbeHit hit = trace_probe_ray(base, seg_dir, remaining);
		if (!hit.found) break;
		const float t_local = length(hit.position - base);
		if (hit.materialIdx == matIdx) {
			++candidate_count;
			if (random_float(seed) < 1.0f / (float)candidate_count) {
				chosen_pos = hit.position;
				chosen_normal = hit.normal;
			}
		}
		const float step = t_local + 1e-4f;
		base = base + step * seg_dir;
		remaining -= step;
	}
	if (candidate_count == 0) return false;

	const float sample_prob = 1.0f / (float)candidate_count;

	const float3 d = chosen_pos - p0;
	const float dist = length(d);
	const float3 Sp = make_float3(
		gpu_bssrdf_sr(table, sigma_t[0], rho[0], dist),
		gpu_bssrdf_sr(table, sigma_t[1], rho[1], dist),
		gpu_bssrdf_sr(table, sigma_t[2], rho[2], dist));

	const float3 exit_n = normalize(chosen_normal);
	const float d_n  = dot(d, axis);
	const float d_t1 = dot(d, t1);
	const float d_t2 = dot(d, t2);
	const float r_proj_axis = sqrtf(fmaxf(0.0f, d_t1 * d_t1 + d_t2 * d_t2));
	const float r_proj_t1   = sqrtf(fmaxf(0.0f, d_t2 * d_t2 + d_n  * d_n));
	const float r_proj_t2   = sqrtf(fmaxf(0.0f, d_n  * d_n  + d_t1 * d_t1));
	const float cos_axis = fabsf(dot(exit_n, axis));
	const float cos_t1   = fabsf(dot(exit_n, t1));
	const float cos_t2   = fabsf(dot(exit_n, t2));
	constexpr float kAxisProb = 0.5f, kTangentProb = 0.25f;
	float pdf = 0.0f;
	for (int c = 0; c < 3; ++c) {
		pdf += kAxisProb   * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_axis) * cos_axis
			 + kTangentProb * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t1)   * cos_t1
			 + kTangentProb * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t2)   * cos_t2;
	}
	pdf /= 3.0f;
	if (pdf <= 0.0f) return false;

	out_exit_pos = chosen_pos;
	out_exit_normal = exit_n;
	out_Sp = Sp;
	out_pdf = pdf;
	out_sample_prob = sample_prob;
	return true;
}

// GGX microfacet rough-dielectric scatter + real glossy NEE/MIS (pbrt-v4
// RoughDielectricBxDF), factored out of shade_material()'s own
// MaterialType::RoughDielectric case (below) so MaterialType::
// DielectricMedium's entry/exit boundary (optix_intersection_sphere.h/
// optix_intersection_disk_cylinder.h) can call the EXACT same logic when
// fused with a rough dielectric surface, instead of duplicating it a
// second time inside this one file (duplication across the recursive/
// wavefront BACKEND boundary is this codebase's own established
// convention - see wavefront_device_helpers.h's own header comment - but
// duplicating within a single backend's own file has no such precedent or
// justification).
//
// `flatRoughness`: the "mat.fuzz"-equivalent flat (pre-remap) isotropic
// roughness value - a plain parameter rather than always reading mat.fuzz
// directly, because DielectricMedium's own copy of that same union slot
// (fuzz/roughness/g/beta_m, optix_types.h) is already taken by `g` (the
// medium's Henyey-Greenstein asymmetry); DielectricMedium's caller passes
// mat.dielectric_medium_extra.roughness instead (see that field's own
// comment), while plain RoughDielectric's caller passes mat.fuzz, same as
// before this factoring. `mat.roughnessV`/`mat.remapRoughness`/
// `mat.textureIdx` need no such indirection - genuinely unused by
// DielectricMedium otherwise (see optix_types.h), so both callers read
// them directly off `mat`.
//
// `allowDispersion`: false for the DielectricMedium caller - critically,
// NOT just "no dispersive scene currently fuses a medium", a genuine
// safety requirement: mat.dispersive_extra and mat.dielectric_medium_extra
// are THE SAME UNION SLOT (optix_types.h), so reading mat.dispersive_extra.
// cauchy_A for a DielectricMedium material would reinterpret its own
// sigma_t/roughness/surfaceKind bits as a bogus Cauchy coefficient -
// almost always a small positive float, which the dispersion gate
// (`cauchy_A > 0.0f`) would misread as "this material IS dispersive",
// triggering a spurious per-path RGB-channel lock and a nonsense IOR.
// Plain RoughDielectric's caller passes true (its own dispersive_extra is
// a real, intentional field for that type).
//
// Returns false only in the rare grazing-angle degenerate case (the
// sampled microfacet reflection/TIR-fallback direction ends up below the
// local hemisphere, wo_z<=0) - the caller must then treat this hit as "no
// scatter at all" (matching shade_material()'s own `scattered=false`
// handling for this same condition, the one place the original case could
// early-exit via `break` before reaching this function's own call site).
// Returns true otherwise, having written scattered_dir/attenuation/
// is_specular/brdf_pdf_override, added any NEE contribution into
// `emission`, and set `eta` on a genuine transmission (left untouched -
// already defaulted to 1.0f by every caller - otherwise).
__device__ __forceinline__ bool rough_dielectric_scatter_and_nee(
	const MaterialData& mat, float flatRoughness, bool allowDispersion,
	const float3& normal, const float3& ray_dir, const float3& hit_point,
	bool front_face, float uv_u, float uv_v, const float3& dpdu,
	bool do_regularize, unsigned int& seed, unsigned int& inout_rgb_channel,
	float3& attenuation, float3& scattered_dir, bool& is_specular,
	float& brdf_pdf_override, float3& emission, float& eta)
{
	float3 nee_shadow_rgb = make_float3(1.0f, 1.0f, 1.0f);  // per-channel transmittance of the last NEE shadow ray through chromatic media (trace_shadow_ray_stochastic)
	// RoughnessToAlpha (sqrt), unless pbrt-v4 "remaproughness" is false (see
	// MaterialData::remapRoughness) - then flatRoughness/mat.roughnessV
	// already ARE the alpha values. mat.textureIdx>=0 means "roughness" was
	// texture-bound - sample the image's red/x channel as the scalar
	// isotropic roughness at THIS hit instead of the flat value, matching
	// CPU's rough_dielectric::true_alpha() (material_pbrt.h) exactly,
	// including its isotropic-only scope (no separate uroughness/
	// vroughness texture support).
	float rd_alpha_x, rd_alpha_y;
	if (mat.textureIdx >= 0) {
		const float rd_rough = sample_texture(mat.textureIdx, uv_u, uv_v, hit_point).x;
		rd_alpha_x = rd_alpha_y = mat.remapRoughness ? sqrtf(rd_rough) : rd_rough;
	} else {
		rd_alpha_x = mat.remapRoughness ? sqrtf(flatRoughness) : flatRoughness;
		rd_alpha_y = ResolveAnisotropicAlphaV(mat.roughnessV, flatRoughness, mat.remapRoughness);
	}
	if (do_regularize) {
		rd_alpha_x = RegularizeAlpha(rd_alpha_x);
		rd_alpha_y = RegularizeAlpha(rd_alpha_y);
	}
	// pbrt-v4 dispersion (Cauchy formula) - see this function's own header
	// comment for why allowDispersion must be false for the DielectricMedium
	// caller specifically, not just "no scene currently combines these".
	float rd_ior = mat.ior;
	if (allowDispersion && mat.dispersive_extra.cauchy_A > 0.0f) {
		if (inout_rgb_channel == kRgbChannelUnset) {
			inout_rgb_channel = static_cast<unsigned int>(random_float(seed) * 3.0f);
			if (inout_rgb_channel > 2u) inout_rgb_channel = 2u;  // 3.0f*u can hit exactly 3.0f
		}
		rd_ior = CauchyEta(kRgbChannelWavelengthNm[inout_rgb_channel],
		                   mat.dispersive_extra.cauchy_A, mat.dispersive_extra.cauchy_B);
	}
	float rd_ri = front_face ? (1.0f / rd_ior) : rd_ior;

	// Local shading frame (n = +Z)
	float3 n = normal;
	float3 tan, bitan;
	BuildDpduTangentFrame(n.x, n.y, n.z, dpdu.x, dpdu.y, dpdu.z, tan.x, tan.y, tan.z, bitan.x, bitan.y, bitan.z);

	float3 wi_w = normalize(-ray_dir);
	float wi_x = dot(wi_w, tan), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
	// Track whether wi got sign-flipped below so any OTHER direction
	// queried against this same local wi (NEE light directions, see
	// below) can be put through the identical flip - f()/pdf() (see
	// RoughDielectricBxDF in src/shared/bxdfs_conductor.h) branch on
	// wo_z's sign relative to THIS wi, so a queried direction must be
	// expressed in the same mirrored coordinate system, not just
	// dotted against the raw (tan,bitan,n) basis.
	bool rd_flip = (wi_z < 0.0f);
	if (rd_flip) { wi_z=-wi_z; wi_x=-wi_x; wi_y=-wi_y; }

	TrowbridgeReitz<float> rd_dist(rd_alpha_x, rd_alpha_y);
	float wm_x, wm_y, wm_z;
	rd_dist.Sample_wm(wi_x, wi_y, wi_z,
					  random_float(seed), random_float(seed),
					  wm_x, wm_y, wm_z);

	float cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
	float F = FrDielectric(cos_i, 1.0f / rd_ri);

	// A rejected continuation sample (a reflection that lands below the horizon, a refraction that lands on the wrong
	// side) must not skip this vertex's NEE: pbrt takes the direct-light sample at every vertex whether or not the
	// BSDF sample that follows succeeds. Returning false here used to drop NEE too, losing the rejection probability
	// of the interior side of a rough glass object - 15-40% of the glass pixels at roughness 0.25-0.6.
	bool rd_rej = false;
	float3 wo_local;
	if (random_float(seed) < F) {
		// Reflect
		float wo_x = 2.0f*cos_i*wm_x - wi_x;
		float wo_y = 2.0f*cos_i*wm_y - wi_y;
		float wo_z = 2.0f*cos_i*wm_z - wi_z;
		if (wo_z <= 0.0f) rd_rej = true;
		wo_local = make_float3(wo_x, wo_y, wo_z);
	} else {
		// Refract
		if (wm_z < 0.0f) { wm_x=-wm_x; wm_y=-wm_y; wm_z=-wm_z; }
		float sin2_t = rd_ri*rd_ri * (1.0f - cos_i*cos_i);
		if (sin2_t >= 1.0f) {
			// TIR: reflect
			float wo_x = 2.0f*cos_i*wm_x - wi_x;
			float wo_y = 2.0f*cos_i*wm_y - wi_y;
			float wo_z = 2.0f*cos_i*wm_z - wi_z;
			if (wo_z <= 0.0f) rd_rej = true;
			wo_local = make_float3(wo_x, wo_y, wo_z);
		} else {
			// Transmitted direction (pbrt-v4 Refract in local frame):
			//   wo = -eta*wi + (eta*dot(wi,wm) - cos_t)*wm
			// wo_z < 0: ray crosses through the surface boundary
			float cos_t = sqrtf(1.0f - sin2_t);
			float wo_x = rd_ri*(-wi_x) + (rd_ri*cos_i - cos_t)*wm_x;
			float wo_y = rd_ri*(-wi_y) + (rd_ri*cos_i - cos_t)*wm_y;
			float wo_z = -(rd_ri*wi_z  - (rd_ri*cos_i - cos_t)*wm_z);
			// pbrt-v4 rejects a refracted sample that ends up on the same side as wo (DielectricBxDF::Sample_f: SameHemisphere(wo, wi)), which a strongly tilted
			// microfacet produces at grazing incidence. Keeping it added ~7% to the albedo leaving glass at 75 degrees.
			if (wo_z >= 0.0f) rd_rej = true;
			wo_local = make_float3(wo_x, wo_y, wo_z);
			// pbrt-v4 etaScale - a genuine transmission (not the TIR
			// fallback-to-reflect branch above), eta = rd_ri exactly
			// like plain Dielectric's own eta.
			if (!rd_rej) eta = rd_ri;
		}
	}
	// Dispersive glass and the effectively-smooth lobe keep the old behaviour (no NEE on a rejected sample): the dispersive
	// channel is only fixed once a sample is accepted, and a smooth lobe has no NEE at all.
	if (rd_rej && ((allowDispersion && mat.dispersive_extra.cauchy_A > 0.0f) || rd_dist.EffectivelySmooth())) return false;
	scattered_dir = rd_rej ? n : normalize(wo_local.x*tan + wo_local.y*bitan + wo_local.z*n);
	// pbrt-v4 RoughDielectricBxDF: BSDF weight with VNDF sampling = G(wo,wi)/G1(wi)
	// (the D, cos, and pdf terms cancel; only shadow-masking ratio remains)
	if (rd_rej) {
		attenuation = make_float3(0.0f, 0.0f, 0.0f);  // the path ends after this vertex's NEE below
	} else {
		float wo_x = wo_local.x, wo_y = wo_local.y, wo_z = fabsf(wo_local.z);
		float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z);
		float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
		float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
		attenuation = make_float3(w, w, w);
	}

	// Real NEE/MIS for glossy (non-EffectivelySmooth) rough glass - NEE here
	// can reach lights on EITHER side of the interface (reflection when the
	// local light direction's z is positive, transmission/"seen through the
	// glass" when negative) - trace_shadow_ray() nudges its origin along the
	// shadow ray's OWN direction (not the surface normal), so a
	// transmission-side shadow ray correctly steps through the interface
	// instead of immediately self-intersecting it.
	if (!rd_dist.EffectivelySmooth()) {
		is_specular = false;
		// rd_ior, not mat.ior: RoughDielectricBxDF::f()/pdf() (src/shared/
		// bxdfs_conductor.h) take eta as an explicit call parameter and
		// never read this struct's own .ior member - so this value is
		// currently inert either way - but using the dispersion-resolved
		// one here avoids leaving a flat, undispersed IOR sitting in a
		// field literally named for the thing this whole case just
		// computed a dispersed value for.
		RoughDielectricBxDF<float> rd_bxdf{ rd_ior, rd_alpha_x, rd_alpha_y };
		// wo_local was already derived (above) from the flip-adjusted
		// wi_x/wi_y/wi_z - it's already in the same mirrored frame as
		// wi, same as the attenuation-weight G2/G1 computation just
		// above uses it unflipped. Re-flipping it here would double-
		// flip it relative to wi, breaking RoughDielectricBxDF::pdf()'s
		// documented "wo expressed in the same flipped frame as wi"
		// contract (src/shared/bxdfs_conductor.h) - matches
		// wavefront_kernels.cu's identical computation, which passes
		// wo_local's components through unmodified.
		brdf_pdf_override = rd_rej ? -1.0f : rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, wo_local.x, wo_local.y, wo_local.z);

		{
			float3 to_light, sampled_light_emission; float max_dist, light_pdf;
			if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
				float llx = dot(to_light, tan), lly = dot(to_light, bitan), llz = dot(to_light, n);
				if (rd_flip) { llx=-llx; lly=-lly; llz=-llz; }
				if (llz != 0.0f && trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb)) {
					float fval = rd_bxdf.f(wi_x, wi_y, wi_z, rd_ri, llx, lly, llz);
					float brdf_pdf = rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, llx, lly, llz);
					float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
					emission = emission + mis_weight * make_float3(fval, fval, fval) * sampled_light_emission * fabsf(llz) / light_pdf
						* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
				}
			}
		}

		for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
			float3 wi_p, Li_p; float t_max_p;
			if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
			float plx = dot(wi_p, tan), ply = dot(wi_p, bitan), plz = dot(wi_p, n);
			if (rd_flip) { plx=-plx; ply=-ply; plz=-plz; }
			if (plz == 0.0f) continue;
			if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb)) {
				float fval = rd_bxdf.f(wi_x, wi_y, wi_z, rd_ri, plx, ply, plz);
				emission = emission + make_float3(fval, fval, fval) * Li_p * fabsf(plz) * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb);
			}
		}

		{
			const float3& skyColor = params.camera.backgroundColor;
			if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
				float3 sky_dir, sky_Le_val; float pdf_sky;
				sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
				float skx = dot(sky_dir, tan), sky_y = dot(sky_dir, bitan), skz = dot(sky_dir, n);
				if (rd_flip) { skx=-skx; sky_y=-sky_y; skz=-skz; }
				if (skz != 0.0f && pdf_sky > 0.0f && trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
					float fval = rd_bxdf.f(wi_x, wi_y, wi_z, rd_ri, skx, sky_y, skz);
					float brdf_pdf_sky = rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, skx, sky_y, skz);
					float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
					emission = emission + mis_weight * make_float3(fval, fval, fval) * sky_Le_val * fabsf(skz) / pdf_sky
						* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
				}
			}
		}
	} else {
		is_specular = true;
	}
	return true;
}

// Scatter + next-event estimation for a layered (coated) BSDF, shared by MaterialType::CoatedDiffuse and
// CoatedConductor - pbrt-v4's LayeredBxDF, ported in src/shared/bxdfs_layered.h. pbrt integrates it with three
// separate pieces, and so does this:
//   - Sample_f (a random walk): the continuation direction and an unbiased weight f*cos/pdf (attenuation) -
//     BSDFSample::pdfIsProportional, the walk has no usable density of its own;
//   - PDF(): an approximate density used only for MIS weights (brdf_pdf_override, and the pdf paired with each
//     light sample below);
//   - f(): a stochastic BSDF value for the light samples.
// pbrt decides NEE from the BSDF's flags, not from the sample: a smooth coat over a rough base still gets NEE
// (f() has no delta part), while the coat's mirror reflection, if that is the sample drawn, is a specular bounce
// (is_specular, no MIS). Only a smooth coat over a smooth conductor - a delta lobe overall - skips NEE.
// A failed walk (Russian roulette, a refraction lost to total internal reflection, ...) still takes the light
// samples, then ends the path with zero weight: pbrt samples direct light at every vertex whether or not the
// BSDF sample that follows succeeds. Returns false only when there is nothing to do at all (a delta lobe whose
// sample failed, or a ray below the horizon).
template<typename Bx>
__device__ __forceinline__ bool layered_scatter_and_nee(
	const Bx& bx, const float3& normal, const float3& ray_dir, const float3& hit_point, const float3& dpdu,
	unsigned int& seed, float3& attenuation, float3& scattered_dir, bool& is_specular,
	float& brdf_pdf_override, float3& emission)
{
	float3 nee_shadow_rgb = make_float3(1.0f, 1.0f, 1.0f);  // per-channel transmittance of the last NEE shadow ray through chromatic media (trace_shadow_ray_stochastic)
	float3 n = normal;
	float3 tan, bit;
	BuildDpduTangentFrame(n.x, n.y, n.z, dpdu.x, dpdu.y, dpdu.z, tan.x, tan.y, tan.z, bit.x, bit.y, bit.z);
	const float3 wi_w = normalize(-ray_dir);
	const float wi_x = dot(wi_w, tan), wi_y = dot(wi_w, bit), wi_z = dot(wi_w, n);
	if (wi_z <= 0.0f) return false;

	uint64_t ws0, ws1; random_seed64_pair(seed, ws0, ws1);
	const BxDFSampleResult<float> smp = bx.sample_local(wi_x, wi_y, wi_z, ws0, ws1);
	const bool nee_enabled = !bx.is_delta();
	if (!smp.valid) {
		if (!nee_enabled) return false;
		attenuation       = make_float3(0.0f, 0.0f, 0.0f);   // the path ends after this vertex's NEE below
		scattered_dir     = n;
		is_specular       = false;
		brdf_pdf_override = -1.0f;
	} else {
		attenuation   = make_float3(smp.r, smp.g, smp.b);
		scattered_dir = normalize(smp.wo_x*tan + smp.wo_y*bit + smp.wo_z*n);
		is_specular   = smp.is_specular || !nee_enabled;
		brdf_pdf_override = is_specular ? -1.0f : bx.pdf(wi_x, wi_y, wi_z, smp.wo_x, smp.wo_y, smp.wo_z);
	}
	if (!nee_enabled) return true;

	{
		float3 to_light, sampled_light_emission; float max_dist, light_pdf;
		if (sample_nee_light(hit_point, seed, to_light, sampled_light_emission, max_dist, light_pdf, optixGetRayTime())) {
			const float llx = dot(to_light, tan), lly = dot(to_light, bit), llz = dot(to_light, n);
			if (llz > 0.0f && trace_shadow_ray_stochastic(hit_point, to_light, max_dist, seed, nee_shadow_rgb)) {
				uint64_t ns0, ns1; random_seed64_pair(seed, ns0, ns1);
				float fr, fg, fb;
				bx.f(wi_x, wi_y, wi_z, llx, lly, llz, ns0, ns1, fr, fg, fb);
				const float brdf_pdf = bx.pdf(wi_x, wi_y, wi_z, llx, lly, llz);
				const float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);
				emission = emission + mis_weight * make_float3(fr, fg, fb) * sampled_light_emission * llz / light_pdf
					* (camera_medium_shadow_trans(max_dist) * nee_shadow_rgb);
			}
		}
	}

	for (unsigned int pi = 0; pi < params.numPunctualLights; ++pi) {
		float3 wi_p, Li_p; float t_max_p;
		if (!eval_punctual_light(params.punctualLights[pi], hit_point, wi_p, Li_p, t_max_p)) continue;
		const float plx = dot(wi_p, tan), ply = dot(wi_p, bit), plz = dot(wi_p, n);
		if (plz <= 0.0f) continue;
		if (trace_shadow_ray_stochastic(hit_point, wi_p, t_max_p, seed, nee_shadow_rgb)) {
			uint64_t ns0, ns1; random_seed64_pair(seed, ns0, ns1);
			float fr, fg, fb;
			bx.f(wi_x, wi_y, wi_z, plx, ply, plz, ns0, ns1, fr, fg, fb);
			emission = emission + make_float3(fr, fg, fb) * Li_p * plz * (camera_medium_shadow_trans(t_max_p) * nee_shadow_rgb);
		}
	}

	{
		const float3& skyColor = params.camera.backgroundColor;
		if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
			float3 sky_dir, sky_Le_val; float pdf_sky;
			sample_sky_nee(seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
			const float skx = dot(sky_dir, tan), sky_y = dot(sky_dir, bit), skz = dot(sky_dir, n);
			if (skz > 0.0f && pdf_sky > 0.0f && trace_shadow_ray_stochastic(hit_point, sky_dir, 1e30f, seed, nee_shadow_rgb)) {
				uint64_t ns0, ns1; random_seed64_pair(seed, ns0, ns1);
				float fr, fg, fb;
				bx.f(wi_x, wi_y, wi_z, skx, sky_y, skz, ns0, ns1, fr, fg, fb);
				const float brdf_pdf_sky = bx.pdf(wi_x, wi_y, wi_z, skx, sky_y, skz);
				const float mis_weight = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
				emission = emission + mis_weight * make_float3(fr, fg, fb) * sky_Le_val * skz / pdf_sky
					* (camera_medium_shadow_trans(1e30f) * nee_shadow_rgb);
			}
		}
	}
	return true;
}
