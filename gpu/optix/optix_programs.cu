// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"    // Shared exact Fresnel (CPU+GPU)
#include "../../src/shared/microfacet.h" // GGX TrowbridgeReitz (CPU+GPU)

// Launch parameters (constant across all threads)
extern "C" { __constant__ LaunchParams params; }

//==============================================================================
// Utility functions
//==============================================================================

// Random number generator (PCG)
__device__ __forceinline__ unsigned int pcg_hash(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ float random_float(unsigned int& seed) {
	seed = pcg_hash(seed);
	return float(seed) / 4294967296.0f;
}

__device__ __forceinline__ float3 random_float3(unsigned int& seed) {
	return make_float3(random_float(seed), random_float(seed), random_float(seed));
}

__device__ __forceinline__ float3 random_in_unit_sphere(unsigned int& seed) {
	while (true) {
		float3 p = 2.0f * random_float3(seed) - make_float3(1.0f, 1.0f, 1.0f);
		if (dot(p, p) < 1.0f) return p;
	}
}

__device__ __forceinline__ float3 random_unit_vector(unsigned int& seed) {
	return normalize(random_in_unit_sphere(seed));
}

__device__ __forceinline__ float3 random_on_hemisphere(const float3& normal, unsigned int& seed) {
	float3 on_unit_sphere = random_unit_vector(seed);
	if (dot(on_unit_sphere, normal) > 0.0f)
		return on_unit_sphere;
	else
		return -on_unit_sphere;
}

__device__ __forceinline__ bool near_zero(const float3& v) {
	const float s = 1e-8f;
	return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

// reflect/refract from shared CPU/GPU header (pbrt-v4 pattern)
#include "../../src/shared/math_utils.h"
__device__ __forceinline__ float3 reflect(const float3& v, const float3& n) { return cpu_gpu_reflect(v, n); }
__device__ __forceinline__ float3 refract(const float3& uv, const float3& n, float e) { return cpu_gpu_refract<float3,float>(uv, n, e); }

// Schlick's approximation removed â€” FrDielectric<float> from shared/fresnel.h is used instead.

//==============================================================================
// Multiple Importance Sampling (MIS) Helpers
//==============================================================================

// MIS power heuristic (beta=2) -- delegates to shared PowerHeuristic (pbrt-v4 pattern)
__device__ __forceinline__ float mis_power_heuristic(float pdf_a, float pdf_b) {
	return PowerHeuristic(pdf_a, pdf_b);
}

// Cosine-weighted hemisphere sampling PDF
__device__ __forceinline__ float cosine_pdf(const float3& direction, const float3& normal) {
	float cosine = dot(normalize(direction), normal);
	return fmaxf(0.0f, cosine / 3.14159265358979323846f);
}

// Sample a random point on a sphere light
__device__ __forceinline__ float3 sample_sphere_light(
	const SphereData& sphere,
	const float3& origin,
	unsigned int& seed,
	float& pdf
) {
	// Direction from origin to sphere center
	float3 to_center = sphere.center - origin;
	float dist_sq = dot(to_center, to_center);

	// Avoid division by zero
	if (dist_sq < 1e-6f) {
		pdf = 0.0f;
		return make_float3(0.0f, 0.0f, 1.0f);
	}

	// Compute solid angle PDF
	float cos_theta_max = sqrtf(1.0f - sphere.radius * sphere.radius / dist_sq);
	float solid_angle = 2.0f * 3.14159265358979323846f * (1.0f - cos_theta_max);
	pdf = 1.0f / solid_angle;

	// Build ONB around direction to sphere
	float3 w = normalize(to_center);
	float3 a = (fabsf(w.x) > 0.9f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
	float3 v = normalize(cross(w, a));
	float3 u = cross(w, v);

	// Sample direction within cone
	float z = 1.0f + random_float(seed) * (cos_theta_max - 1.0f);
	float phi = 2.0f * 3.14159265358979323846f * random_float(seed);
	float r = sqrtf(1.0f - z * z);

	float3 direction = r * cosf(phi) * u + r * sinf(phi) * v + z * w;
	return normalize(direction);
}

// Sample a random point on a quad light
__device__ __forceinline__ float3 sample_quad_light(
	const QuadData& quad,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist
) {
	// Random point on quad surface
	float a = random_float(seed);
	float b = random_float(seed);
	float3 point = quad.Q + a * quad.u + b * quad.v;

	// Direction to sampled point
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	float3 direction = to_light / out_dist;

	// Area-based PDF converted to solid angle
	float area = length(quad.w);  // w = u x v, so |w| = area
	float cosine = fabsf(dot(direction, quad.normal));

	if (cosine < 1e-6f || area < 1e-6f) {
		pdf = 0.0f;
		return direction;
	}

	pdf = dist_sq / (cosine * area);
	return direction;
}

// Evaluate quad light PDF for a given direction
__device__ __forceinline__ float quad_light_pdf(
	const QuadData& quad,
	const float3& origin,
	const float3& direction
) {
	// Intersect ray with quad plane
	float denom = dot(direction, quad.normal);
	if (fabsf(denom) < 1e-6f) return 0.0f;

	float t = (quad.D - dot(quad.normal, origin)) / denom;
	if (t < 0.001f) return 0.0f;

	// Check if hit point is inside quad
	float3 hit_point = origin + t * direction;
	float3 p = hit_point - quad.Q;

	// Solve for (alpha, beta) such that p = alpha*u + beta*v
	float3 n = quad.w;  // u x v
	float n_len_sq = dot(n, n);
	if (n_len_sq < 1e-6f) return 0.0f;

	float alpha = dot(cross(p, quad.v), n) / n_len_sq;
	float beta = dot(cross(quad.u, p), n) / n_len_sq;

	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
		return 0.0f;  // Outside quad
	}

	// Compute PDF
	float dist_sq = t * t * dot(direction, direction);
	float cosine = fabsf(dot(direction, quad.normal));
	float area = sqrtf(n_len_sq);

	return dist_sq / (cosine * area);
}

// Evaluate sphere light PDF for a given direction
__device__ __forceinline__ float sphere_light_pdf(
	const SphereData& sphere,
	const float3& origin,
	const float3& direction
) {
	// Check if direction intersects sphere (simplified - just use solid angle)
	float3 to_center = sphere.center - origin;
	float dist_sq = dot(to_center, to_center);

	if (dist_sq < 1e-6f) return 0.0f;

	float cos_theta_max = sqrtf(1.0f - sphere.radius * sphere.radius / dist_sq);
	float solid_angle = 2.0f * 3.14159265358979323846f * (1.0f - cos_theta_max);

	return 1.0f / solid_angle;
}

// Trace a shadow ray to test visibility
// Returns true if path to light is unoccluded (false if occluded)
__device__ __forceinline__ bool trace_shadow_ray(
	const float3& origin,
	const float3& direction,
	float max_distance
) {
	// Pack shadow payload (single bool: occluded)
	unsigned int occluded = 1;  // Default to occluded (will be set to 0 if miss)

	// Trace shadow ray with occlusion testing
	optixTrace(
		params.traversable,           // Acceleration structure
		origin,                        // Ray origin
		direction,                     // Ray direction
		0.001f,                        // tmin (avoid self-intersection)
		max_distance,                  // tmax
		0.0f,                          // rayTime
		OptixVisibilityMask(255),      // Visibility mask
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,  // Flags
		RAY_TYPE_SHADOW,               // SBT offset (shadow ray type)
		RAY_TYPE_COUNT,                // SBT stride (number of ray types)
		RAY_TYPE_SHADOW,               // Miss SBT index
		occluded                       // Payload (single unsigned int)
	);

	// Return true if NOT occluded (path is clear)
	return (occluded == 0);
}



//==============================================================================
// Sphere Intersection Program
//==============================================================================

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
				float bwm_x, bwm_y, bwm_z;
				cc_dist.Sample_wm(cc_wi_x, cc_wi_y, cc_wi_z,
								  random_float(seed), random_float(seed),
								  bwm_x, bwm_y, bwm_z);
				float cos_c = cc_wi_x*bwm_x + cc_wi_y*bwm_y + cc_wi_z*bwm_z;
				float wo_x  = 2.0f*cos_c*bwm_x - cc_wi_x;
				float wo_y  = 2.0f*cos_c*bwm_y - cc_wi_y;
				float wo_z  = 2.0f*cos_c*bwm_z - cc_wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }

				float G1_c = cc_dist.G1(cc_wi_x, cc_wi_y, cc_wi_z);
				float G_c  = cc_dist.G(wo_x, wo_y, wo_z, cc_wi_x, cc_wi_y, cc_wi_z);
				float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

				float F_r = FrComplex(cos_c, mat.eta_c.x, mat.k_c.x) * wt_c;
				float F_g = FrComplex(cos_c, mat.eta_c.y, mat.k_c.y) * wt_c;
				float F_b = FrComplex(cos_c, mat.eta_c.z, mat.k_c.z) * wt_c;

				float F_out = FrDielectric(fabsf(wo_z), 1.0f / mat.ior);
				float T_out = 1.0f - F_out;
				float T_in  = 1.0f - F_in;

				attenuation = make_float3(F_r * T_in * T_out,
										 F_g * T_in * T_out,
										 F_b * T_in * T_out);
				cc_wo = make_float3(wo_x, wo_y, wo_z);
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
		float t_hit = optixGetRayTmax();  // Hit distance
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

extern "C" __global__ void __intersection__quad() {
	// Get primitive index from OptiX (this is relative to the build input)
	// Since we have 2 build inputs (spheres first, quads second), we need the quad index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch quad data from device array
	const QuadData& quad = params.quads[primIdx];

	// Get ray parameters
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	// Quad intersection (plane intersection + bounds check)
	const float denom = dot(quad.normal, ray_dir);

	// Parallel to plane?
	if (fabsf(denom) < 1e-8f) return;

	// Compute t
	const float t = (quad.D - dot(quad.normal, ray_orig)) / denom;
	if (t < ray_tmin || t > ray_tmax) return;

	// Compute hit point in plane
	const float3 intersection = ray_orig + t * ray_dir;
	const float3 planar_vec = intersection - quad.Q;

	// Check if hit is within quad bounds using barycentric coordinates
	const float w_dot_w = dot(quad.w, quad.w);
	const float alpha = dot(quad.w, cross(planar_vec, quad.v)) / w_dot_w;
	const float beta = dot(quad.w, cross(quad.u, planar_vec)) / w_dot_w;

	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
		return;  // Outside quad

	// Report intersection
	optixReportIntersection(
		t,                      // t value
		0,                      // hit kind
		__float_as_int(quad.normal.x),  // attribute 0 (normal)
		__float_as_int(quad.normal.y),  // attribute 1
		__float_as_int(quad.normal.z),  // attribute 2
		0                                // attribute 3 (unused, but required for consistency)
	);
}

//==============================================================================
// Quad Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__quad() {
	// Get primitive index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch quad data from device array
	const QuadData& quad = params.quads[primIdx];
	const int matIdx = quad.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	// Reconstruct normal from attributes
	const float3 normal = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);

	// Get hit point
	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Determine front face
	const bool front_face = dot(ray_dir, normal) < 0.0f;
	const float3 final_normal = front_face ? normal : -normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering (same as sphere)
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing

	switch (mat.type) {
		case MaterialType::Lambertian: {
			// Multiple Importance Sampling (MIS) for diffuse surfaces

			// Sample BRDF (cosine-weighted hemisphere) for indirect lighting
			scattered_dir = final_normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = final_normal;
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

				// Sample direction toward light
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

				// Combined PDF = selection_pdf * geometric_pdf
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

						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						float cos_theta = fmaxf(0.0f, dot(to_light, final_normal));
						float3 brdf = mat.albedo / 3.14159265358979323846f;  // Lambertian BRDF
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						emission = emission + direct_light;
					}
				}
			}

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), final_normal);
			scattered_dir = normalize(reflected) + mat.fuzz * random_in_unit_sphere(seed);
			// Clamp to hemisphere: if fuzz pushes below surface, use pure reflection
			if (dot(scattered_dir, final_normal) <= 0.0f)
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
			float cos_theta = fminf(dot(-unit_direction, final_normal), 1.0f);
			float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0f;

			// FrDielectric expects eta_t/eta_i; ri = eta_i/eta_t, so pass 1/ri
			if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
				scattered_dir = reflect(unit_direction, final_normal);
			} else {
				scattered_dir = refract(unit_direction, final_normal, ri);
			}
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::ThinDielectric: {
			// Zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF) -- quad version
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
				scattered_dir = unit_direction;  // straight through
			}
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::CoatedConductor: {
			// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF) -- quad version
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
				float bwm_x, bwm_y, bwm_z;
				cc_dist.Sample_wm(cc_wi_x, cc_wi_y, cc_wi_z,
								  random_float(seed), random_float(seed),
								  bwm_x, bwm_y, bwm_z);
				float cos_c = cc_wi_x*bwm_x + cc_wi_y*bwm_y + cc_wi_z*bwm_z;
				float wo_x  = 2.0f*cos_c*bwm_x - cc_wi_x;
				float wo_y  = 2.0f*cos_c*bwm_y - cc_wi_y;
				float wo_z  = 2.0f*cos_c*bwm_z - cc_wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }

				float G1_c = cc_dist.G1(cc_wi_x, cc_wi_y, cc_wi_z);
				float G_c  = cc_dist.G(wo_x, wo_y, wo_z, cc_wi_x, cc_wi_y, cc_wi_z);
				float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

				float F_r = FrComplex(cos_c, mat.eta_c.x, mat.k_c.x) * wt_c;
				float F_g = FrComplex(cos_c, mat.eta_c.y, mat.k_c.y) * wt_c;
				float F_b = FrComplex(cos_c, mat.eta_c.z, mat.k_c.z) * wt_c;

				float F_out = FrDielectric(fabsf(wo_z), 1.0f / mat.ior);
				float T_out = 1.0f - F_out;
				float T_in  = 1.0f - F_in;

				attenuation = make_float3(F_r * T_in * T_out,
										 F_g * T_in * T_out,
										 F_b * T_in * T_out);
				cc_wo = make_float3(wo_x, wo_y, wo_z);
			}
			scattered_dir = normalize(cc_wo.x*cc_tan + cc_wo.y*cc_bit + cc_wo.z*cc_n);
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::RoughDielectric: {
			// GGX microfacet BSDF (pbrt-v4 RoughDielectricBxDF) â€” quad version
			float rd_alpha = mat.fuzz;
			rd_alpha = sqrtf(rd_alpha);  // RoughnessToAlpha: alpha = sqrt(roughness)
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
					// GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF) -- quad version
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

					// VNDF weight: F * G(wo,wi) / G1(wi)  (pbrt-v4 ConductorBxDF::Sample_f)
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
					// Rough dielectric coat over Lambertian base (pbrt-v4 CoatedDiffuseBxDF) -- quad version
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
					}
					break;
				}

				case MaterialType::DiffuseTransmission: {
				// pbrt-v4 DiffuseTransmissionBxDF -- quad version
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
				// pbrt-v4 NormalizedFresnelBxDF -- quad version
				// f(wi) = (1 - FrDielectric(cos_wi, eta)) / (c * pi)
				// Diffuse BSDF: participates in MIS (is_specular=false).
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
				// p12: correct BSDF PDF for MIS on next bounce: (1-Fr)*cos/(c*pi)
				brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265358979323846f);

				// NEE direct lighting
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
					// Emissive material - no scattering
					// Emission already set from mat.emission above
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
			// p0-p2: surface attenuation (BRDF albedo - raygen multiplies with throughput)
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
		float t_hit = optixGetRayTmax();  // Hit distance
		// p12: BRDF PDF for MIS on the next bounce.
		// Specular (Metal/Dielectric): 0 = no MIS (pbrt-v4 specularBounce pattern).
		// Lambertian: cosine_pdf of the sampled direction.
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

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
		// p12: NEE PDF for the incoming ray direction reaching this quad light.
		float light_pdf_for_incoming = 0.0f;
		if (params.aliasTable && params.numLights > 0) {
			int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && !params.isLightSphere[li]) {
					sel_pdf = params.aliasTable[li].pdf;
					break;
				}
			}
			// Solid-angle PDF for quad: dist^2 / (cos * area)
			float3 to_light = hit_point - ray_orig;
			float dist_sq = dot(to_light, to_light);
			float area = length(quad.w);  // |u x v|
			float cos_theta = fabsf(dot(normalize(ray_dir), quad.normal));
			if (cos_theta > 1e-6f && area > 1e-6f && dist_sq > 1e-6f)
				light_pdf_for_incoming = sel_pdf * dist_sq / (cos_theta * area);
		}
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Shadow Any-Hit Programs
//==============================================================================

// Shadow any-hit for spheres
// For opaque geometry, any hit means occlusion - terminate immediately
extern "C" __global__ void __anyhit__shadow_sphere() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = params.spheres[primIdx];
	const MaterialData& mat = params.materials[sphere.materialIdx];

	// IMPORTANT: When hitting light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);  // NOT occluded - light is visible
		optixTerminateRay();
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

// Shadow any-hit for quads
// For opaque geometry, any hit means occlusion - terminate immediately
extern "C" __global__ void __anyhit__shadow_quad() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = params.quads[primIdx];
	const MaterialData& mat = params.materials[quad.materialIdx];

	// IMPORTANT: When hitting a light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);  // NOT occluded - light is visible
		optixTerminateRay();
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

//==============================================================================
// Miss Program
//==============================================================================

extern "C" __global__ void __miss__ms() {
	// Cornell Box uses BLACK background (no sky light)
	const float3 color = make_float3(0.0f, 0.0f, 0.0f);

	// Unpack attenuation from payload
	float3 attenuation = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Black background - no emission
	float3 emission = make_float3(0.0f, 0.0f, 0.0f);

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);
	optixSetPayload_10(0);  // absorbed (terminate path with no emission)
	optixSetPayload_12(0);  // no brdf_pdf (path terminated)
}

//==============================================================================
// Shadow Miss Program
//==============================================================================

extern "C" __global__ void __miss__shadow() {
	// Shadow ray missed all geometry - path is clear (not occluded)
	optixSetPayload_0(0);  // occluded = false
}

//==============================================================================
// Ray Generation Program (will implement with path tracing loop)
//==============================================================================

extern "C" __global__ void __raygen__rg() {
	// Get pixel coordinates
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	const unsigned int px = idx.x;
	const unsigned int py = idx.y;

	if (px >= params.width || py >= params.height) return;

	// Initialize random seed from pixel + frame
	unsigned int seed = (py * params.width + px) + params.frameNumber * 719393;

	//Accumulate samples
	float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

	for (unsigned int s = 0; s < params.samplesPerPixel; ++s) {
		// Halton low-discrepancy pixel offset (pbrt-v4 HaltonSampler pattern)
		// base-2 for x, base-3 for y. Pixel coords (px,py) are mixed into the
		// sample index for per-pixel decorrelation â€” adjacent pixels use different
		// sub-sequences, avoiding a structured grid artifact across the image.
		// Bounce RNG (seed) keeps using PCG32 for scatter/light directions.
		float u = (float(px) + halton2(s, px, py)) / float(params.width - 1);
		float v = (float(params.height - 1 - py) + halton3(s, px, py)) / float(params.height - 1);  // Flip Y

		// Generate camera ray
		float3 ray_origin = params.camera.origin;
		float3 ray_direction = normalize(
			params.camera.lower_left_corner +
			u * params.camera.horizontal +
			v * params.camera.vertical -
			ray_origin
		);

		// Path tracing loop
		float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
		float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
		float  prev_brdf_pdf = 0.0f;  // BRDF PDF of the ray that arrived at this bounce (0 = primary)

		for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
			// Initialize payload
			PathTracingPayload payload;
			payload.attenuation = throughput;
			payload.emission = make_float3(0.0f, 0.0f, 0.0f);
			payload.seed = seed;
			payload.depth = depth;
			payload.scattered = false;

			// Trace ray - pack 13 payload registers
			unsigned int p0 = __float_as_uint(payload.attenuation.x);
			unsigned int p1 = __float_as_uint(payload.attenuation.y);
			unsigned int p2 = __float_as_uint(payload.attenuation.z);
			unsigned int p3 = 0;  // emission (will be set by hit/miss)
			unsigned int p4 = 0;
			unsigned int p5 = 0;
			unsigned int p6 = 0;  // scatter direction
			unsigned int p7 = 0;
			unsigned int p8 = 0;
			unsigned int p9 = payload.seed;
			unsigned int p10 = 0;  // scattered flag
			unsigned int p11 = 0;  // hit distance 't'
			unsigned int p12 = 0;  // brdf_pdf of scattered direction (for MIS on next bounce)

			optixTrace(
				params.traversable,     // Acceleration structure
				ray_origin,             // Ray origin
				ray_direction,          // Ray direction
				0.001f,                 // tmin
				1e16f,                  // tmax
				0.0f,                   // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				RAY_TYPE_RADIANCE,      // SBT offset
				RAY_TYPE_COUNT,         // SBT stride
				RAY_TYPE_RADIANCE,      // missSBTIndex
				p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
			);

			// Unpack payload (13 registers)
			payload.attenuation.x = __uint_as_float(p0);
			payload.attenuation.y = __uint_as_float(p1);
			payload.attenuation.z = __uint_as_float(p2);
			payload.emission.x = __uint_as_float(p3);  // Emission from this hit
			payload.emission.y = __uint_as_float(p4);
			payload.emission.z = __uint_as_float(p5);
			payload.scatterDir.x = __uint_as_float(p6);
			payload.scatterDir.y = __uint_as_float(p7);
			payload.scatterDir.z = __uint_as_float(p8);
			payload.seed = p9;
			unsigned int flag = p10;
			float t_hit = __uint_as_float(p11);
			float scatter_brdf_pdf = __uint_as_float(p12);  // BRDF PDF of the new scatter direction

			// Decode flag: 0=absorbed, 1=scattered, 2=hit_light
			if (flag == 2) {
				// Hit an emissive surface via a BRDF-sampled bounce.
				// Apply MIS weight (pbrt-v4 PathIntegrator pattern):
				//   w_b = PowerHeuristic(p_b, p_l)  where:
				//     p_b = prev_brdf_pdf  (BRDF PDF of the direction that arrived here)
				//     p_l = p12            (NEE selection*geometry PDF for this light+direction)
				// Special cases: depth==0 (primary ray) or prev_brdf_pdf==0 (specular bounce)
				//   -> add full Le, no MIS.
				float3 Le = payload.emission;
				if (depth > 0 && prev_brdf_pdf > 0.0f &&
					(Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f)) {
					float p_l = scatter_brdf_pdf;  // hit program writes light NEE pdf into p12 for flag==2
					if (p_l > 0.0f) {
						float w_b = mis_power_heuristic(prev_brdf_pdf, p_l);
						radiance = radiance + throughput * w_b * Le;
					} else {
						radiance = radiance + throughput * Le;
					}
				} else {
					radiance = radiance + throughput * Le;
				}
				break;
			} else if (flag == 1) {
				// Scattered - compute scatter origin and update for next bounce

				// Add NEE direct-light emission from this surface hit (already MIS-weighted inside hit program)
				radiance = radiance + throughput * payload.emission;

				float3 hit_point = ray_origin + t_hit * ray_direction;
				float3 scatter_origin = hit_point + 0.01f * normalize(payload.scatterDir);

				// Multiply throughput by surface BRDF (attenuation from hit program)
				throughput = throughput * payload.attenuation;

				// Russian Roulette (pbrt-v4 PathIntegrator pattern)
				// Start after depth > 1 so the primary ray and first bounce always survive.
				// q = max(0, 1 - MaxComponent(throughput)); terminate if rand < q, else reweight.
				if (depth > 1) {
					float rr_max = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
					if (rr_max < 1.0f) {
						float q = fmaxf(0.0f, 1.0f - rr_max);
						if (random_float(seed) < q) break;   // terminate path
						throughput = throughput / (1.0f - q); // unbiased reweight
					}
				}

				// Carry BRDF PDF of the new scatter direction for MIS at the next bounce
				prev_brdf_pdf = scatter_brdf_pdf;

				ray_origin = scatter_origin;
				ray_direction = normalize(payload.scatterDir);  // MUST normalize!
				seed = payload.seed;
			} else {
				// Absorbed â€” add any surface emission (e.g. background hit) then stop
				radiance = radiance + throughput * payload.emission;
				break;
			}
		}  // end depth loop

		pixel_color = pixel_color + radiance;
	}  // end sample loop

	// Average samples
	pixel_color = pixel_color / float(params.samplesPerPixel);

	// Write to framebuffer
	const unsigned int idx_flat = py * params.width + px;
	params.framebuffer[idx_flat] = pixel_color;
}
