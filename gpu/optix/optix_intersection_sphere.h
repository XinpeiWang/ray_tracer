// optix_intersection_sphere.h -- Sphere intersection + closest-hit programs
// Included by optix_programs.cu

extern "C" __global__ void __intersection__sphere() {
	// Get primitive index from OptiX
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch sphere data from device array
	const SphereData& sphere = params.spheres[primIdx];

	// Get ray parameters
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	// Sphere intersection
	const float3 oc = ray_orig - sphere.center;
	const float a = dot(ray_dir, ray_dir);
	const float half_b = dot(oc, ray_dir);
	const float c = dot(oc, oc) - sphere.radius * sphere.radius;
	const float discriminant = half_b * half_b - a * c;

	if (discriminant < 0.0f) return;  // No hit

	const float sqrtd = sqrtf(discriminant);

	// Find nearest root in valid range
	float root = (-half_b - sqrtd) / a;
	if (root < ray_tmin || root > ray_tmax) {
		root = (-half_b + sqrtd) / a;
		if (root < ray_tmin || root > ray_tmax)
			return;  // No valid hit
	}

	// Report intersection
	optixReportIntersection(
		root,                      // t value
		0,                         // hit kind
		__float_as_int(sphere.center.x),  // attribute 0 (sphere center for normal calc)
		__float_as_int(sphere.center.y),  // attribute 1
		__float_as_int(sphere.center.z),  // attribute 2
		__float_as_int(sphere.radius)     // attribute 3
	);
}

//==============================================================================
// Sphere Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__sphere() {
	// Get primitive index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch sphere data from device array
	const SphereData& sphere = params.spheres[primIdx];
	const int matIdx = sphere.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	// Reconstruct sphere data from attributes
	const float3 sphere_center = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);
	const float sphere_radius = __int_as_float(optixGetAttribute_3());

	// Get hit point
	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Compute normal
	float3 outward_normal = (hit_point - sphere_center) / sphere_radius;
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing
	bool is_medium = false;    // MaterialType::Medium: t_hit below must use medium_t_hit,
	float medium_t_hit = 0.0f; // not optixGetRayTmax() (which is just the sphere's entry surface)

	switch (mat.type) {
		case MaterialType::Lambertian: {
			// Multiple Importance Sampling (MIS) for diffuse surfaces

			// Sample BRDF (cosine-weighted hemisphere) for indirect lighting
			scattered_dir = normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = normal;
			}
			scattered_dir = normalize(scattered_dir);
			attenuation = mat.albedo;
			scattered = true;

			// Add direct lighting via explicit light sampling (Next Event Estimation)
			if (params.numLights > 0) {
				// Power-weighted light selection via alias table (pbrt-v4 PowerLightSampler pattern)
				int light_idx;
				float selection_pdf;
				if (params.aliasTable) {
					// O(1) alias-table sample: pick slot, accept or redirect
					int slot = int(random_float(seed) * float(params.numLights));
					if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
					const GpuAliasEntry& entry = params.aliasTable[slot];
					light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
					selection_pdf = params.aliasTable[light_idx].pdf;
				} else {
					// Fallback: uniform selection
					light_idx = int(random_float(seed) * float(params.numLights));
					if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
					selection_pdf = 1.0f / float(params.numLights);
				}

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere = params.isLightSphere[light_idx];

				// Sample direction toward light
				float3 to_light;
				float geom_pdf = 0.0f;  // PDF of the sampled direction (geometry term)
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

				// Combined PDF = selection_pdf * geometric_pdf
				float light_pdf = selection_pdf * geom_pdf;

				if (light_pdf > 1e-6f) {
					// Check if light is visible (shadow ray)
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);

					if (visible) {
						// Evaluate BRDF PDF for this direction
						float brdf_pdf = cosine_pdf(to_light, normal);

						// MIS weight using power heuristic
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

						// Get light emission
						float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
						if (is_sphere) {
							const MaterialData& light_mat = params.materials[params.spheres[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						} else {
							const MaterialData& light_mat = params.materials[params.quads[prim_idx].materialIdx];
							light_emission = light_mat.emission;
						}

						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						float cos_theta = fmaxf(0.0f, dot(to_light, normal));
						float3 brdf = mat.albedo / 3.14159265358979323846f;  // Lambertian BRDF
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						// Add to emission (raygen will apply throughput)
						emission = emission + direct_light;
					}
				}
			}

			// Direct lighting from punctual (point/spot/distant) lights -
			// deterministic delta lights, evaluated separately from the
			// area-light alias table above (see optix_device_helpers.h).
			add_punctual_lights_lambertian(hit_point, normal, mat.albedo, emission);

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), normal);
			scattered_dir = normalize(reflected) + mat.fuzz * random_in_unit_sphere(seed);
			// Clamp to hemisphere: if fuzz pushes below surface, use pure reflection
			if (dot(scattered_dir, normal) <= 0.0f)
				scattered_dir = reflected;
			attenuation = mat.albedo;
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::Dielectric: {
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			float ri = front_face ? (1.0f / mat.ior) : mat.ior;
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
			float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0f;

			// FrDielectric expects eta_t/eta_i; ri = eta_i/eta_t, so pass 1/ri
			if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
				scattered_dir = reflect(unit_direction, normal);
			} else {
				scattered_dir = refract(unit_direction, normal, ri);
			}
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::ThinDielectric: {
			// Zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF) -- sphere version
			// Transmission goes straight through (no bending); reflection is specular.
			// Multiple internal bounces folded analytically: R_eff = R + T^2*R/(1-R^2)
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fabsf(dot(unit_direction, normal));
			float R = FrDielectric(cos_theta, mat.ior);
			if (R < 1.0f) {
				float T = 1.0f - R;
				R += T * T * R / (1.0f - R * R);
			}
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			if (random_float(seed) < R) {
				scattered_dir = reflect(unit_direction, normal);
			} else {
				scattered_dir = unit_direction;  // straight through
			}
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::CoatedConductor: {
			// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF) -- sphere version
			// coat: ior=mat.ior, roughness=mat.fuzz; conductor: eta_c, k_c per RGB
			float cc_alpha = sqrtf(mat.fuzz);
			float3 cc_n   = normal;
			float3 cc_up  = (fabsf(cc_n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cc_tan  = normalize(cross(cc_up, cc_n));
			float3 cc_bit  = cross(cc_n, cc_tan);

			float3 cc_wi_w = normalize(-ray_dir);
			float cc_wi_x = dot(cc_wi_w, cc_tan);
			float cc_wi_y = dot(cc_wi_w, cc_bit);
			float cc_wi_z = dot(cc_wi_w, cc_n);
			if (cc_wi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> cc_dist(cc_alpha, cc_alpha);

			// Coat top interface: GGX VNDF + FrDielectric
			float cwm_x, cwm_y, cwm_z;
			cc_dist.Sample_wm(cc_wi_x, cc_wi_y, cc_wi_z,
							  random_float(seed), random_float(seed),
							  cwm_x, cwm_y, cwm_z);
			float cc_cos_i = cc_wi_x*cwm_x + cc_wi_y*cwm_y + cc_wi_z*cwm_z;
			float F_in     = FrDielectric(cc_cos_i, mat.ior);

			float3 cc_wo;
			if (random_float(seed) < F_in) {
				// Path A: coat specular reflection
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
					// Path B: transmit into layer -> conductor bounce -> exit coat
					// Step 1: transmitted direction inside layer.
					// The coat microfacet cwm was already sampled above; the reflected direction
					// off cwm is (2*cc_cos_i*cwm - cc_wi). Inside the slab the ray goes downward,
					// so we flip the z component.  (Matches CPU CoatedConductorBxDF step 1.)
					float w_x = 2.0f*cc_cos_i*cwm_x - cc_wi_x;
					float w_y = 2.0f*cc_cos_i*cwm_y - cc_wi_y;
					float w_z = 2.0f*cc_cos_i*cwm_z - cc_wi_z;
					if (w_z > 0.0f) w_z = -w_z;   // ensure pointing downward into layer
					if (w_z == 0.0f) { scattered = false; break; }

					// Step 2: flip to conductor frame -- "incoming from above" (fw_z > 0)
					float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

					// Sample conductor microfacet from the correct transmitted direction
					float bwm_x, bwm_y, bwm_z;
					cc_dist.Sample_wm(fw_x, fw_y, fw_z,
									  random_float(seed), random_float(seed),
									  bwm_x, bwm_y, bwm_z);
					float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
					if (cos_c <= 0.0f) { scattered = false; break; }

					// Conductor reflection direction (upward, rwo_z > 0 = exits toward coat top)
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

					// Step 3: exit through coat top � Fresnel at exit angle (rwo_z = cos of exit)
					float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);  // inside->outside
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
			// GGX microfacet BSDF (pbrt-v4 RoughDielectricBxDF)
			// fuzz field stores GGX roughness; ior = index of refraction
			float rd_alpha = mat.fuzz;
			rd_alpha = sqrtf(rd_alpha);  // RoughnessToAlpha: alpha = sqrt(roughness)
			float rd_ri    = front_face ? (1.0f / mat.ior) : mat.ior;

			// Local shading frame (n = +Z)
			float3 n = normal;
			float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 tan  = normalize(cross(up_v, n));
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
				// Reflect
				float wo_x = 2.0f*cos_i*wm_x - wi_x;
				float wo_y = 2.0f*cos_i*wm_y - wi_y;
				float wo_z = 2.0f*cos_i*wm_z - wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }
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
					if (wo_z <= 0.0f) { scattered = false; break; }
					wo_local = make_float3(wo_x, wo_y, wo_z);
				} else {
					// Transmitted direction (pbrt-v4 Refract in local frame):
					//   wo = -eta*wi + (eta*dot(wi,wm) - cos_t)*wm
					// wo_z < 0: ray crosses through the surface boundary
					float cos_t = sqrtf(1.0f - sin2_t);
					float wo_x = rd_ri*(-wi_x) + (rd_ri*cos_i - cos_t)*wm_x;
					float wo_y = rd_ri*(-wi_y) + (rd_ri*cos_i - cos_t)*wm_y;
					float wo_z = -(rd_ri*wi_z  - (rd_ri*cos_i - cos_t)*wm_z);
					wo_local = make_float3(wo_x, wo_y, wo_z);
				}
			}
			scattered_dir = normalize(wo_local.x*tan + wo_local.y*bitan + wo_local.z*n);
			// pbrt-v4 RoughDielectricBxDF: BSDF weight with VNDF sampling = G(wo,wi)/G1(wi)
			// (the D, cos, and pdf terms cancel; only shadow-masking ratio remains)
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
			// GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF) -- sphere version
			float c_alpha = sqrtf(mat.fuzz);
			float3 cn = normal;
			float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 ctan   = normalize(cross(cup, cn));
			float3 cbitan = cross(cn, ctan);
			float3 cwi = normalize(-ray_dir);
			float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
			if (cwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
			float cwm_x, cwm_y, cwm_z;
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, random_float(seed), random_float(seed), cwm_x, cwm_y, cwm_z);
			float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
			float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
			float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
			if (cwo_z <= 0.0f) { scattered = false; break; }
			float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
			float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
			float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
			float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
			attenuation = make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight);
			scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::CoatedDiffuse: {
			// Rough dielectric coat over Lambertian base (pbrt-v4 CoatedDiffuseBxDF) -- sphere version
			float cd_alpha = sqrtf(mat.fuzz);
			float3 cdn  = normal;
			float3 cdup = (fabsf(cdn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cdtan = normalize(cross(cdup, cdn));
			float3 cdbit = cross(cdn, cdtan);
			float3 cdwi  = normalize(-ray_dir);
			float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbit), cdwi_z = dot(cdwi, cdn);
			if (cdwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> cd_dist(cd_alpha, cd_alpha);
			float cdwm_x, cdwm_y, cdwm_z;
			cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z, random_float(seed), random_float(seed), cdwm_x, cdwm_y, cdwm_z);
			float cd_cosi = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
			float F_in = FrDielectric(cd_cosi, mat.ior);
			if (random_float(seed) < F_in) {
				float cwo_x2 = 2.0f*cd_cosi*cdwm_x - cdwi_x;
				float cwo_y2 = 2.0f*cd_cosi*cdwm_y - cdwi_y;
				float cwo_z2 = 2.0f*cd_cosi*cdwm_z - cdwi_z;
				if (cwo_z2 <= 0.0f) { scattered = false; break; }
				float G1 = cd_dist.G1(cdwi_x, cdwi_y, cdwi_z);
				float G  = cd_dist.G(cwo_x2, cwo_y2, cwo_z2, cdwi_x, cdwi_y, cdwi_z);
				float wt = (G1 > 1e-8f) ? G / G1 : 0.0f;
				float fw = F_in * wt;
				attenuation   = make_float3(fw, fw, fw);
				scattered_dir = normalize(cwo_x2*cdtan + cwo_y2*cdbit + cwo_z2*cdn);
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
			}
			break;
		}

		case MaterialType::DiffuseTransmission: {
			// pbrt-v4 DiffuseTransmissionBxDF -- sphere version
			// albedo = reflectance R (same hemisphere), emission = transmittance T (reused field)
			float3 R = mat.albedo;
			float3 T_col = mat.emission;  // transmittance packed into emission field
			float pr = fmaxf(R.x, fmaxf(R.y, R.z));
			float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
			if (pr + pt <= 0.0f) { scattered = false; break; }

			if (random_float(seed) < pr / (pr + pt)) {
				// Diffuse reflection: cosine-weighted same hemisphere
				scattered_dir = normalize(normal + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = normal;
				attenuation   = R;
			} else {
				// Diffuse transmission: cosine-weighted opposite hemisphere
				float3 neg_n  = -normal;
				scattered_dir = normalize(neg_n + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = neg_n;
				attenuation   = T_col;
			}
			scattered   = true;
			is_specular = false;
			break;
		}

		case MaterialType::NormalizedFresnel: {
			// pbrt-v4 NormalizedFresnelBxDF -- sphere version
			// f(wi) = (1 - FrDielectric(cos_wi, eta)) / (c * pi)
			// where c = 1 - 2*FresnelMoment1(1/eta)
			// Diffuse BSDF: participates in MIS (is_specular=false).
			// attenuation = BRDF weight for BRDF-sampled direction.
			// NEE: direct light with BSDF-evaluated brdf factor.
			float nf_eta = mat.ior;
			float inv_eta = 1.0f / nf_eta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;

			scattered_dir = normalize(normal + random_unit_vector(seed));
			if (near_zero(scattered_dir)) scattered_dir = normal;

			float cos_wi = fmaxf(dot(scattered_dir, normal), 1e-6f);
			float fr     = FrDielectric(cos_wi, nf_eta);
			float weight = (1.0f - fr) / nf_c;
			attenuation  = make_float3(weight, weight, weight);
			scattered    = true;
			is_specular  = false;
			// p12: correct BSDF PDF for MIS on next bounce: (1-Fr)*cos/(c*pi)
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
						float cos_to_light = fmaxf(dot(to_light, normal), 0.0f);
						if (cos_to_light > 0.0f) {
							// BSDF value at the light direction
							float fr_l  = FrDielectric(cos_to_light, nf_eta);
							float brdf_val = (1.0f - fr_l) / (nf_c * 3.14159265358979323846f);
							float brdf_pdf_l = brdf_val * cos_to_light;  // cosine_pdf * brdf * pi cancel
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

		case MaterialType::Medium: {
			// Homogeneous participating medium - see MaterialType::Medium's
			// comment in optix_types.h. Recompute both sphere-intersection
			// roots (t here is only the near/entry root optixGetRayTmax()
			// already reported) to find the exit distance, then sample a
			// free-path distance via Beer-Lambert inversion; if it lands
			// before the exit, scatter via the HG phase function, else pass
			// straight through and continue from the exit point.
			float3 unit_dir = normalize(ray_dir);
			float3 oc2 = ray_orig - sphere_center;
			float half_b2 = dot(oc2, unit_dir);
			float c2 = dot(oc2, oc2) - sphere_radius * sphere_radius;
			float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
			float sq2 = sqrtf(disc2);
			float t_near = fmaxf(0.0f, -half_b2 - sq2);
			float t_far  = -half_b2 + sq2;
			float dist_inside = fmaxf(0.0f, t_far - t_near);

			float sigma_t = mat.ior;
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t) : 1e30f;

			if (free_path < dist_inside) {
				medium_t_hit = t_near + free_path;
				scattered_dir = sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
				attenuation = mat.albedo;
			} else {
				medium_t_hit = t_far;
				scattered_dir = unit_dir;  // straight through, no interaction
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
			}
			scattered    = true;
			is_specular  = true;  // no NEE/MIS for volume scattering (not yet implemented)
			is_medium    = true;
			break;
		}

		case MaterialType::Hair: {
			// Marschner/Chiang fiber scattering - see sample_hair_material's
			// comment in optix_device_helpers.h. Matches hair_material.h's
			// skip_pdf=true: no NEE/MIS, res.r/g/b already divides by the
			// sample pdf (same convention as this codebase's other
			// is_specular=true cases).
			scattered   = sample_hair_material(ray_dir, normal, mat, seed, scattered_dir, attenuation);
			is_specular = true;
			break;
		}

		case MaterialType::DiffuseLight: {
			// Emissive material - no scattering
			scattered = false;
			break;
		}

		default: {
			// Unknown material - absorb
			scattered = false;
			break;
		}
	}

	// Pack updated payload back into registers

	// p3-p5: emission from this surface hit
	// p6-p8: scatter direction (if scattered)
	// p9: updated seed
	// p10: scattered flag (0=absorbed, 1=scattered, 2=hit_light)
	// p11: hit distance 't'
	// p12: brdf_pdf of scattered dir (flag==1) OR light NEE pdf of incoming dir (flag==2) for MIS

	// Always set emission (for all material types - non-lights have emission=0)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		// Return surface attenuation ONLY (raygen will multiply with throughput)
		// Medium: the scatter/exit point is not the sphere's entry surface
		// (optixGetRayTmax()) - use the distance computed in the Medium
		// case above instead.
		float t_hit = is_medium ? medium_t_hit : optixGetRayTmax();  // Hit distance
		// p12: BRDF PDF for MIS on the next bounce.
		// Specular (Metal/Dielectric): 0 = no MIS (pbrt-v4 specularBounce pattern).
		// Lambertian: cosine_pdf of the sampled direction.
		// NormalizedFresnel: (1-Fr)*cos/(c*pi) stored in brdf_pdf_override.
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));  // Scatter direction
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// p12: NEE PDF for the incoming ray direction reaching this sphere light.
		// This is the solid-angle PDF used by the NEE sampler, enabling MIS in raygen.
		float light_pdf_for_incoming = 0.0f;
		if (params.aliasTable && params.numLights > 0) {
			// Find selection PMF for this sphere in the alias table
			int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && params.isLightSphere[li]) {
					sel_pdf = params.aliasTable[li].pdf;
					break;
				}
			}
			// Solid-angle PDF: 1 / (2*pi*(1 - cos_theta_max))
			float3 to_center = sphere_center - ray_orig;
			float dist_sq = dot(to_center, to_center);
			if (dist_sq > sphere_radius * sphere_radius) {
				float cos_theta_max = sqrtf(1.0f - sphere_radius * sphere_radius / dist_sq);
				float solid_angle = 2.0f * 3.14159265f * (1.0f - cos_theta_max);
				light_pdf_for_incoming = sel_pdf / solid_angle;
			}
		}
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Quad Intersection Program
//==============================================================================

