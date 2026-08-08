// optix_intersection_bilinear_patch.h -- Bilinear patch intersection + closest-hit programs
// Included by optix_programs.cu
//
// A genuinely curved (non-planar in general) ruled surface through 4 corners
// p00,p10,p01,p11 - NOT a flat quad. Ray intersection ports
// src/shared/bilinear_patch.h's blp_intersect() (Ramsey et al. 2004 / pbrt-v4
// IntersectBilinearPatch): solve a quadratic in u for where the ray crosses
// the patch's u-isolines, then for each valid root find the closest-point-
// between-two-lines (ray vs. u-isoline) to get v and t.

extern "C" __global__ void __intersection__bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];

	const float3 ro = optixGetWorldRayOrigin();
	const float3 rd = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 p00 = patch.p00;
	const float3 p10 = patch.p10;
	const float3 p01 = patch.p01;
	const float3 p11 = patch.p11;

	// Quadratic coefficients for the u iso-line distance (blp_intersect)
	const float3 e0 = p10 - p00;
	const float3 e1 = p01 - p11;
	const float3 e2 = p00 - ro;
	const float3 e3 = p10 - ro;

	const float a = dot(cross(e0, e1), rd);
	const float c = dot(cross(e2, rd), p01 - p00);
	const float b = dot(cross(e3, rd), p11 - p10) - (a + c);

	float u1, u2;
	if (a == 0.0f) {
		if (b == 0.0f) return;  // no solution
		u1 = u2 = -c / b;
	} else {
		const double disc = (double)b * (double)b - 4.0 * (double)a * (double)c;
		if (disc < 0.0) return;
		const double sqrt_disc = sqrt(disc);
		const double q = (b < 0.0f) ? -0.5 * ((double)b - sqrt_disc) : -0.5 * ((double)b + sqrt_disc);
		u1 = (float)(q / (double)a);
		u2 = (q != 0.0) ? (float)((double)c / q) : 0.0f;
		if (u1 > u2) { const float tmp = u1; u1 = u2; u2 = tmp; }
	}

	// eps = gamma(30) * (max|ro| + max|rd| + max|p00|+max|p10|+max|p01|+max|p11|)
	const float eps = 4e-6f *
		(fmaxf(fmaxf(fabsf(ro.x), fabsf(ro.y)), fabsf(ro.z)) +
		 fmaxf(fmaxf(fabsf(rd.x), fabsf(rd.y)), fabsf(rd.z)) +
		 fmaxf(fmaxf(fabsf(p00.x), fabsf(p00.y)), fabsf(p00.z)) +
		 fmaxf(fmaxf(fabsf(p10.x), fabsf(p10.y)), fabsf(p10.z)) +
		 fmaxf(fmaxf(fabsf(p01.x), fabsf(p01.y)), fabsf(p01.z)) +
		 fmaxf(fmaxf(fabsf(p11.x), fabsf(p11.y)), fabsf(p11.z)));

	float t_hit = ray_tmax;
	float u_hit = -1.0f, v_hit = -1.0f;
	bool found = false;

	for (int i = 0; i < 2; ++i) {
		const float u_cand = (i == 0) ? u1 : u2;
		if (i == 1 && u2 == u1) continue;
		if (u_cand < 0.0f || u_cand > 1.0f) continue;

		const float3 uo   = lerp(p00, p10, u_cand);
		const float3 p11u = lerp(p01, p11, u_cand);
		const float3 ud   = p11u - uo;

		const float3 delta = uo - ro;
		const float3 perp  = cross(rd, ud);
		const float p2 = dot(perp, perp);
		if (p2 < 1e-20f) continue;

		// det3(a,b,c) == dot(a, cross(b,c)) (scalar triple product)
		const float v_num = dot(delta, cross(rd, perp));
		const float t_num = dot(delta, cross(ud, perp));

		const float t_cand = t_num / p2;
		const float v_cand = v_num / p2;

		if (t_num > p2 * eps && v_cand >= 0.0f && v_cand <= 1.0f &&
			t_cand < t_hit && t_cand >= ray_tmin && t_cand <= ray_tmax) {
			t_hit = t_cand;
			u_hit = u_cand;
			v_hit = v_cand;
			found = true;
		}
	}

	if (!found) return;

	optixReportIntersection(
		t_hit, 0,
		__float_as_int(u_hit),  // attribute 0
		__float_as_int(v_hit),  // attribute 1
		0, 0                    // attributes 2-3 unused
	);
}

//==============================================================================
// Bilinear Patch Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];
	const int matIdx = patch.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// dpdu/dpdv at (u,v) - see bilinear_patch.h's blp_point
	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);

	const float3 geom_normal = normalize(cross(dpdu, dpdv));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 final_normal = front_face ? geom_normal : -geom_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering (same as sphere/quad)
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing

	switch (mat.type) {
		case MaterialType::Lambertian: {
			// Multiple Importance Sampling (MIS) for diffuse surfaces
			scattered_dir = final_normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = final_normal;
			}
			scattered_dir = normalize(scattered_dir);
			attenuation = mat.albedo;
			scattered = true;

			// Add direct lighting via explicit light sampling (Next Event Estimation)
			if (params.numLights > 0) {
				int light_idx;
				float selection_pdf;
				if (params.aliasTable) {
					int slot = int(random_float(seed) * float(params.numLights));
					if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
					const GpuAliasEntry& entry = params.aliasTable[slot];
					light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
					selection_pdf = params.aliasTable[light_idx].pdf;
				} else {
					light_idx = int(random_float(seed) * float(params.numLights));
					if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
					selection_pdf = 1.0f / float(params.numLights);
				}

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere = params.isLightSphere[light_idx];

				float3 to_light;
				float geom_pdf = 0.0f;
				float max_dist = 0.0f;

				if (is_sphere) {
					const SphereData& light_sphere = params.spheres[prim_idx];
					to_light = sample_sphere_light(light_sphere, hit_point, seed, geom_pdf);
					float3 to_center = light_sphere.center - hit_point;
					max_dist = length(to_center);
				} else {
					const QuadData& light_quad = params.quads[prim_idx];
					to_light = sample_quad_light(light_quad, hit_point, seed, geom_pdf, max_dist);
				}

				float light_pdf = selection_pdf * geom_pdf;

				if (light_pdf > 1e-6f) {
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);

					if (visible) {
						float brdf_pdf = cosine_pdf(to_light, final_normal);
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

						float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
						if (is_sphere) {
							const MaterialData& light_mat = params.materials[params.spheres[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						} else {
							const MaterialData& light_mat = params.materials[params.quads[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						}

						float cos_theta = fmaxf(0.0f, dot(to_light, final_normal));
						float3 brdf = mat.albedo / 3.14159265358979323846f;
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						emission = emission + direct_light;
					}
				}
			}

			add_punctual_lights_lambertian(hit_point, final_normal, mat.albedo, emission);

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), final_normal);
			scattered_dir = normalize(reflected) + mat.fuzz * random_in_unit_sphere(seed);
			if (dot(scattered_dir, final_normal) <= 0.0f)
				scattered_dir = reflected;
			attenuation = mat.albedo;
			scattered = true;
			is_specular = true;
			break;
		}

		case MaterialType::Dielectric: {
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			float ri = front_face ? (1.0f / mat.ior) : mat.ior;
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fminf(dot(-unit_direction, final_normal), 1.0f);
			float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0f;

			if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
				scattered_dir = reflect(unit_direction, final_normal);
			} else {
				scattered_dir = refract(unit_direction, final_normal, ri);
			}
			scattered = true;
			is_specular = true;
			break;
		}

		case MaterialType::ThinDielectric: {
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fabsf(dot(unit_direction, final_normal));
			float R = FrDielectric(cos_theta, mat.ior);
			if (R < 1.0f) {
				float T = 1.0f - R;
				R += T * T * R / (1.0f - R * R);
			}
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			if (random_float(seed) < R) {
				scattered_dir = reflect(unit_direction, final_normal);
			} else {
				scattered_dir = unit_direction;
			}
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::CoatedConductor: {
			float cc_alpha = sqrtf(mat.fuzz);
			float3 cc_n   = final_normal;
			float3 cc_up  = (fabsf(cc_n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cc_tan  = normalize(cross(cc_up, cc_n));
			float3 cc_bit  = cross(cc_n, cc_tan);

			float3 cc_wi_w = normalize(-ray_dir);
			float cc_wi_x = dot(cc_wi_w, cc_tan);
			float cc_wi_y = dot(cc_wi_w, cc_bit);
			float cc_wi_z = dot(cc_wi_w, cc_n);
			if (cc_wi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> cc_dist(cc_alpha, cc_alpha);

			float cwm_x, cwm_y, cwm_z;
			cc_dist.Sample_wm(cc_wi_x, cc_wi_y, cc_wi_z,
							  random_float(seed), random_float(seed),
							  cwm_x, cwm_y, cwm_z);
			float cc_cos_i = cc_wi_x*cwm_x + cc_wi_y*cwm_y + cc_wi_z*cwm_z;
			float F_in     = FrDielectric(cc_cos_i, mat.ior);

			float3 cc_wo;
			if (random_float(seed) < F_in) {
				float wo_x = 2.0f*cc_cos_i*cwm_x - cc_wi_x;
				float wo_y = 2.0f*cc_cos_i*cwm_y - cc_wi_y;
				float wo_z = 2.0f*cc_cos_i*cwm_z - cc_wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }
				float G1 = cc_dist.G1(cc_wi_x, cc_wi_y, cc_wi_z);
				float G  = cc_dist.G(wo_x, wo_y, wo_z, cc_wi_x, cc_wi_y, cc_wi_z);
				float w  = (G1 > 1e-8f) ? G / G1 : 0.0f;
				float fv = F_in * w;
				attenuation = make_float3(fv, fv, fv);
				cc_wo = make_float3(wo_x, wo_y, wo_z);
			} else {
				float w_x = 2.0f*cc_cos_i*cwm_x - cc_wi_x;
				float w_y = 2.0f*cc_cos_i*cwm_y - cc_wi_y;
				float w_z = 2.0f*cc_cos_i*cwm_z - cc_wi_z;
				if (w_z > 0.0f) w_z = -w_z;
				if (w_z == 0.0f) { scattered = false; break; }

				float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

				float bwm_x, bwm_y, bwm_z;
				cc_dist.Sample_wm(fw_x, fw_y, fw_z,
								  random_float(seed), random_float(seed),
								  bwm_x, bwm_y, bwm_z);
				float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
				if (cos_c <= 0.0f) { scattered = false; break; }

				float rwo_x = 2.0f*cos_c*bwm_x - fw_x;
				float rwo_y = 2.0f*cos_c*bwm_y - fw_y;
				float rwo_z = 2.0f*cos_c*bwm_z - fw_z;
				if (rwo_z <= 0.0f) { scattered = false; break; }

				float G1_c = cc_dist.G1(fw_x, fw_y, fw_z);
				float G_c  = cc_dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
				float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

				float F_r = FrComplex(cos_c, mat.eta_c.x, mat.k_c.x) * wt_c;
				float F_g = FrComplex(cos_c, mat.eta_c.y, mat.k_c.y) * wt_c;
				float F_b = FrComplex(cos_c, mat.eta_c.z, mat.k_c.z) * wt_c;

				float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);
				float T_out = 1.0f - F_out;
				float T_in  = 1.0f - F_in;

				attenuation = make_float3(F_r * T_in * T_out,
										 F_g * T_in * T_out,
										 F_b * T_in * T_out);
				cc_wo = make_float3(rwo_x, rwo_y, rwo_z);
			}
			scattered_dir = normalize(cc_wo.x*cc_tan + cc_wo.y*cc_bit + cc_wo.z*cc_n);
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::RoughDielectric: {
			float rd_alpha = mat.fuzz;
			rd_alpha = sqrtf(rd_alpha);
			float rd_ri    = front_face ? (1.0f / mat.ior) : mat.ior;

			float3 n = final_normal;
			float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 tan   = normalize(cross(up_v, n));
			float3 bitan = cross(n, tan);

			float3 wi_w = normalize(-ray_dir);
			float wi_x = dot(wi_w, tan), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
			if (wi_z < 0.0f) { wi_z=-wi_z; wi_x=-wi_x; wi_y=-wi_y; }

			TrowbridgeReitz<float> rd_dist(rd_alpha, rd_alpha);
			float wm_x, wm_y, wm_z;
			rd_dist.Sample_wm(wi_x, wi_y, wi_z,
							  random_float(seed), random_float(seed),
							  wm_x, wm_y, wm_z);

			float cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			float F = FrDielectric(cos_i, 1.0f / rd_ri);

			float3 wo_local;
			if (random_float(seed) < F) {
				float wo_x = 2.0f*cos_i*wm_x - wi_x;
				float wo_y = 2.0f*cos_i*wm_y - wi_y;
				float wo_z = 2.0f*cos_i*wm_z - wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }
				wo_local = make_float3(wo_x, wo_y, wo_z);
			} else {
				if (wm_z < 0.0f) { wm_x=-wm_x; wm_y=-wm_y; wm_z=-wm_z; }
				float sin2_t = rd_ri*rd_ri * (1.0f - cos_i*cos_i);
				if (sin2_t >= 1.0f) {
					float wo_x = 2.0f*cos_i*wm_x - wi_x;
					float wo_y = 2.0f*cos_i*wm_y - wi_y;
					float wo_z = 2.0f*cos_i*wm_z - wi_z;
					if (wo_z <= 0.0f) { scattered = false; break; }
					wo_local = make_float3(wo_x, wo_y, wo_z);
				} else {
					float cos_t = sqrtf(1.0f - sin2_t);
					float wo_x = rd_ri*(-wi_x) + (rd_ri*cos_i - cos_t)*wm_x;
					float wo_y = rd_ri*(-wi_y) + (rd_ri*cos_i - cos_t)*wm_y;
					float wo_z = -(rd_ri*wi_z  - (rd_ri*cos_i - cos_t)*wm_z);
					wo_local = make_float3(wo_x, wo_y, wo_z);
				}
			}

			scattered_dir = normalize(wo_local.x*tan + wo_local.y*bitan + wo_local.z*n);
			{
				float wo_x = wo_local.x, wo_y = wo_local.y, wo_z = fabsf(wo_local.z);
				float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z);
				float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
				float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
				attenuation = make_float3(w, w, w);
			}
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::Conductor: {
			float c_alpha = sqrtf(mat.fuzz);
			float3 cn = final_normal;
			float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 ctan  = normalize(cross(cup, cn));
			float3 cbitan = cross(cn, ctan);

			float3 cwi = normalize(-ray_dir);
			float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
			if (cwi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
			float cwm_x, cwm_y, cwm_z;
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z,
							 random_float(seed), random_float(seed),
							 cwm_x, cwm_y, cwm_z);

			float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
			float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
			float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
			if (cwo_z <= 0.0f) { scattered = false; break; }

			float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
			float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
			float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;

			float3 c_F = FrConductorRGB(c_dot,
									   mat.eta_c.x, mat.eta_c.y, mat.eta_c.z,
									   mat.k_c.x,   mat.k_c.y,   mat.k_c.z);
			attenuation = make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight);

			scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::CoatedDiffuse: {
			float cd_alpha = sqrtf(mat.fuzz);
			float3 cdn = final_normal;
			float3 cdup = (fabsf(cdn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cdtan  = normalize(cross(cdup, cdn));
			float3 cdbit  = cross(cdn, cdtan);

			float3 cdwi = normalize(-ray_dir);
			float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbit), cdwi_z = dot(cdwi, cdn);
			if (cdwi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> cd_dist(cd_alpha, cd_alpha);
			float cdwm_x, cdwm_y, cdwm_z;
			cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z,
							  random_float(seed), random_float(seed),
							  cdwm_x, cdwm_y, cdwm_z);

			float cd_cosi = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
			float F_in    = FrDielectric(cd_cosi, mat.ior);

			if (random_float(seed) < F_in) {
				float cwo_x = 2.0f*cd_cosi*cdwm_x - cdwi_x;
				float cwo_y = 2.0f*cd_cosi*cdwm_y - cdwi_y;
				float cwo_z = 2.0f*cd_cosi*cdwm_z - cdwi_z;
				if (cwo_z <= 0.0f) { scattered = false; break; }
				float G1 = cd_dist.G1(cdwi_x, cdwi_y, cdwi_z);
				float G  = cd_dist.G(cwo_x, cwo_y, cwo_z, cdwi_x, cdwi_y, cdwi_z);
				float wt = (G1 > 1e-8f) ? G / G1 : 0.0f;
				float fw = F_in * wt;
				attenuation   = make_float3(fw, fw, fw);
				scattered_dir = normalize(cwo_x*cdtan + cwo_y*cdbit + cwo_z*cdn);
				scattered     = true;
				is_specular   = true;
			} else {
				float3 diff_dir = cdn + random_in_unit_sphere(seed);
				if (dot(diff_dir, diff_dir) < 1e-12f) diff_dir = cdn;
				diff_dir = normalize(diff_dir);
				float cos_out = fabsf(dot(diff_dir, cdn));
				float F_out   = FrDielectric(cos_out, 1.0f / mat.ior);
				float T       = (1.0f - F_in) * (1.0f - F_out);
				attenuation   = make_float3(mat.albedo.x*T, mat.albedo.y*T, mat.albedo.z*T);
				scattered_dir = diff_dir;
				scattered     = true;
				is_specular   = false;

				// NEE: direct light with the coat-weighted diffuse BSDF -
				// this branch sets is_specular=false but was missing the
				// light-sampling block every other is_specular=false
				// material has (Lambertian above), so it could only get
				// illuminated by BSDF-sampled paths that randomly hit the
				// light - no importance sampling toward it at all, making
				// it render far too dark. T_in (F_in) is fixed for this
				// scattering event; F_out is re-evaluated at the
				// light-sampled direction.
				if (params.numLights > 0) {
					int light_idx;
					float selection_pdf;
					if (params.aliasTable) {
						int slot = int(random_float(seed) * float(params.numLights));
						if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
						const GpuAliasEntry& entry = params.aliasTable[slot];
						light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
						selection_pdf = params.aliasTable[light_idx].pdf;
					} else {
						light_idx = int(random_float(seed) * float(params.numLights));
						if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
						selection_pdf = 1.0f / float(params.numLights);
					}

					int prim_idx = params.lightIndices[light_idx];
					bool is_sphere_light = params.isLightSphere[light_idx];

					float3 to_light;
					float geom_pdf = 0.0f;
					float max_dist = 0.0f;

					if (is_sphere_light) {
						const SphereData& light_sphere = params.spheres[prim_idx];
						to_light = sample_sphere_light(light_sphere, hit_point, seed, geom_pdf);
						float3 to_center = light_sphere.center - hit_point;
						max_dist = length(to_center);
					} else {
						const QuadData& light_quad = params.quads[prim_idx];
						to_light = sample_quad_light(light_quad, hit_point, seed, geom_pdf, max_dist);
					}

					float light_pdf = selection_pdf * geom_pdf;
					if (light_pdf > 1e-6f) {
						bool visible = trace_shadow_ray(hit_point, to_light, max_dist);
						if (visible) {
							float cos_to_light = fmaxf(dot(to_light, cdn), 0.0f);
							if (cos_to_light > 0.0f) {
								float F_out_l = FrDielectric(cos_to_light, 1.0f / mat.ior);
								float T_l     = (1.0f - F_in) * (1.0f - F_out_l);
								float3 brdf_val = make_float3(mat.albedo.x*T_l, mat.albedo.y*T_l, mat.albedo.z*T_l)
												  / 3.14159265358979323846f;
								float brdf_pdf   = cosine_pdf(to_light, cdn);
								float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

								float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
								if (is_sphere_light) {
									const MaterialData& light_mat = params.materials[params.spheres[prim_idx].materialIdx];
									light_emission = light_mat.emission;
								} else {
									const MaterialData& light_mat = params.materials[params.quads[prim_idx].materialIdx];
									light_emission = light_mat.emission;
								}

								float3 direct_light = mis_weight * brdf_val * light_emission * cos_to_light / light_pdf;
								emission = emission + direct_light;
							}
						}
					}
				}
			}
			break;
		}

		case MaterialType::DiffuseTransmission: {
			float3 R = mat.albedo;
			float3 T_col = mat.emission;
			float pr = fmaxf(R.x, fmaxf(R.y, R.z));
			float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
			if (pr + pt <= 0.0f) { scattered = false; break; }

			if (random_float(seed) < pr / (pr + pt)) {
				scattered_dir = normalize(final_normal + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = final_normal;
				attenuation   = R;
			} else {
				float3 neg_n  = -final_normal;
				scattered_dir = normalize(neg_n + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = neg_n;
				attenuation   = T_col;
			}
			scattered   = true;
			is_specular = false;
			break;
		}

		case MaterialType::NormalizedFresnel: {
			float nf_eta = mat.ior;
			float inv_eta = 1.0f / nf_eta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;

			scattered_dir = normalize(final_normal + random_unit_vector(seed));
			if (near_zero(scattered_dir)) scattered_dir = final_normal;

			float cos_wi = fmaxf(dot(scattered_dir, final_normal), 1e-6f);
			float fr     = FrDielectric(cos_wi, nf_eta);
			float weight = (1.0f - fr) / nf_c;
			attenuation  = make_float3(weight, weight, weight);
			scattered    = true;
			is_specular  = false;
			brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265358979323846f);

			if (params.numLights > 0) {
				int light_idx;
				float selection_pdf;
				if (params.aliasTable) {
					int slot = int(random_float(seed) * float(params.numLights));
					if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
					const GpuAliasEntry& entry = params.aliasTable[slot];
					light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
					selection_pdf = params.aliasTable[light_idx].pdf;
				} else {
					light_idx = int(random_float(seed) * float(params.numLights));
					if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
					selection_pdf = 1.0f / float(params.numLights);
				}

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere_light = params.isLightSphere[light_idx];

				float3 to_light;
				float geom_pdf = 0.0f;
				float max_dist = 0.0f;

				if (is_sphere_light) {
					const SphereData& light_sphere = params.spheres[prim_idx];
					to_light = sample_sphere_light(light_sphere, hit_point, seed, geom_pdf);
					float3 to_center = light_sphere.center - hit_point;
					max_dist = length(to_center);
				} else {
					const QuadData& light_quad = params.quads[prim_idx];
					to_light = sample_quad_light(light_quad, hit_point, seed, geom_pdf, max_dist);
				}

				float light_pdf = selection_pdf * geom_pdf;
				if (light_pdf > 1e-6f) {
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);
					if (visible) {
						float cos_to_light = fmaxf(dot(to_light, final_normal), 0.0f);
						if (cos_to_light > 0.0f) {
							float fr_l     = FrDielectric(cos_to_light, nf_eta);
							float brdf_val = (1.0f - fr_l) / (nf_c * 3.14159265358979323846f);
							float brdf_pdf_l = brdf_val * cos_to_light;
							float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf_l);

							float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
							if (is_sphere_light) {
								const MaterialData& light_mat = params.materials[params.spheres[prim_idx].materialIdx];
								light_emission = light_mat.emission;
							} else {
								const MaterialData& light_mat = params.materials[params.quads[prim_idx].materialIdx];
								light_emission = light_mat.emission;
							}

							float3 direct_light = mis_weight * brdf_val * light_emission * cos_to_light / light_pdf;
							emission = emission + direct_light;
						}
					}
				}
			}
			break;
		}

		case MaterialType::DiffuseLight: {
			// Emissive material - no scattering. Bilinear patches are never
			// added to lightIndices by scene_builder.cpp (no scene emits light
			// from one), so the "hit_light" NEE-pdf branch below is dead for
			// this geometry type - unlike sphere/quad, there's no 3-way
			// isLightSphere-style tag to look a bilinear-patch light up by, so
			// that branch isn't ported here.
			scattered = false;
			break;
		}

		default: {
			// Unknown material - absorb
			scattered = false;
			break;
		}
	}

	// Pack updated payload back into registers (see optix_intersection_quad.h
	// for the p0-p12 layout - identical here)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// See the DiffuseLight case comment above - not reachable in practice
		// (no scene registers a bilinear patch as a light), pdf=0 is safe.
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}
