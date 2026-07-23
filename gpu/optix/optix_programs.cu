// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"   // Shared exact Fresnel (CPU+GPU)

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

__device__ __forceinline__ float3 reflect(const float3& v, const float3& n) {
	return v - 2.0f * dot(v, n) * n;
}

__device__ __forceinline__ float3 refract(const float3& uv, const float3& n, float etai_over_etat) {
	float cos_theta = fminf(dot(-uv, n), 1.0f);
	float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	float3 r_out_parallel = -sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

// Schlick's approximation removed — FrDielectric<float> from shared/fresnel.h is used instead.

//==============================================================================
// Multiple Importance Sampling (MIS) Helpers
//==============================================================================

// MIS power heuristic (beta = 2)
__device__ __forceinline__ float mis_power_heuristic(float pdf_a, float pdf_b) {
	float a2 = pdf_a * pdf_a;
	float b2 = pdf_b * pdf_b;
	return a2 / (a2 + b2);
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
	float& pdf
) {
	// Random point on quad surface
	float a = random_float(seed);
	float b = random_float(seed);
	float3 point = quad.Q + a * quad.u + b * quad.v;

	// Direction to sampled point
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	float3 direction = to_light / sqrtf(dist_sq);

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
				// Pick a random light
				int light_idx = int(random_float(seed) * float(params.numLights));
				if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere = params.isLightSphere[light_idx];

				// Sample direction toward light
				float3 to_light;
				float light_pdf = 0.0f;
				float max_dist = 0.0f;

				if (is_sphere) {
					const SphereData& light_sphere = params.spheres[prim_idx];
					to_light = sample_sphere_light(light_sphere, hit_point, seed, light_pdf);
					// Calculate distance to sphere center
					float3 to_center = light_sphere.center - hit_point;
					max_dist = length(to_center);
				} else {
					const QuadData& light_quad = params.quads[prim_idx];
					to_light = sample_quad_light(light_quad, hit_point, seed, light_pdf);
					// Calculate distance to sampled point
					float a = random_float(seed);
					float b = random_float(seed);
					float3 light_point = light_quad.Q + a * light_quad.u + b * light_quad.v;
					max_dist = length(light_point - hit_point);
				}

				// Adjust PDF for multiple lights (we picked one uniformly)
				light_pdf *= float(params.numLights);

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

						// Compute direct lighting contribution
						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						float cos_theta = fmaxf(0.0f, dot(to_light, normal));
						float3 brdf = mat.albedo / 3.14159265358979323846f;  // Lambertian BRDF
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						// Add to emission (will be weighted by throughput in raygen loop)
						// NOTE: Do NOT multiply by attenuation_in here - raygen will apply throughput
						emission = emission + direct_light;
					}
				}
			}

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), normal);
			scattered_dir = reflected + mat.fuzz * random_in_unit_sphere(seed);
			attenuation = mat.albedo;
			scattered = (dot(scattered_dir, normal) > 0.0f);
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
	// For ALL hits, we return emission in p3-p5 (CPU adds emission at every bounce)
	// p0-p2: new attenuation (throughput for next bounce)
	// p3-p5: emission from this surface hit
	// p6-p8: scatter direction (if scattered)
	// p9: updated seed
	// p10: scattered flag (0=absorbed, 1=scattered, 2=hit_light)

	// Always set emission (for all material types - non-lights have emission=0)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		// Return surface attenuation ONLY (raygen will multiply with throughput)
		float t_hit = optixGetRayTmax();  // Hit distance

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));  // Scatter direction
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));  // Hit distance
	} else if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_10(2);  // hit_light
	} else {
		optixSetPayload_10(0);  // absorbed
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
				// Pick a random light
				int light_idx = int(random_float(seed) * float(params.numLights));
				if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;

				int prim_idx = params.lightIndices[light_idx];
				bool is_sphere = params.isLightSphere[light_idx];

				// Sample direction toward light
				float3 to_light;
				float light_pdf = 0.0f;
				float max_dist = 0.0f;

				if (is_sphere) {
					const SphereData& light_sphere = params.spheres[prim_idx];
					to_light = sample_sphere_light(light_sphere, hit_point, seed, light_pdf);
					// Calculate distance to sphere center
					float3 to_center = light_sphere.center - hit_point;
					max_dist = length(to_center);
				} else {
					const QuadData& light_quad = params.quads[prim_idx];
					to_light = sample_quad_light(light_quad, hit_point, seed, light_pdf);
					// Calculate distance to sampled point
					float a = random_float(seed);
					float b = random_float(seed);
					float3 light_point = light_quad.Q + a * light_quad.u + b * light_quad.v;
					max_dist = length(light_point - hit_point);
				}

				// Adjust PDF for multiple lights (we picked one uniformly)
				light_pdf *= float(params.numLights);

				if (light_pdf > 1e-6f) {
					// Check if light is visible (shadow ray)
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);

					if (visible) {
						// Evaluate BRDF PDF for this direction
						float brdf_pdf = cosine_pdf(to_light, final_normal);

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

						// Compute direct lighting contribution
						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						float cos_theta = fmaxf(0.0f, dot(to_light, final_normal));
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
			float3 reflected = reflect(normalize(ray_dir), final_normal);
			scattered_dir = reflected + mat.fuzz * random_in_unit_sphere(seed);
			attenuation = mat.albedo;
			scattered = (dot(scattered_dir, final_normal) > 0.0f);
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
	// For ALL hits, we return emission in p3-p5 (CPU adds emission at every bounce)
	// p0-p2: surface attenuation (BRDF albedo - raygen multiplies with throughput)
	// p3-p5: emission from this surface hit
	// p6-p8: scatter direction (if scattered)
	// p9: updated seed
	// p10: scattered flag (0=absorbed, 1=scattered, 2=hit_light)

	// Always set emission (for all material types - non-lights have emission=0)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		// Return surface attenuation ONLY (raygen will multiply with throughput)
		float t_hit = optixGetRayTmax();  // Hit distance

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));  // Scatter direction
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));  // Hit distance
	} else if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_10(2);  // hit_light
	} else {
		optixSetPayload_10(0);  // absorbed
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

	// For other materials, treat as opaque (occludes light)
	// TODO: Add alpha-testing for transparent materials (e.g., glass)
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

	// For other materials, treat as opaque (occludes light)
	// TODO: Add alpha-testing for transparent materials (e.g., glass)
	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

//==============================================================================
// Miss Program (will implement with sky background)
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
		// Random offset within pixel
		float u = (float(px) + random_float(seed)) / float(params.width - 1);
		float v = (float(params.height - 1 - py) + random_float(seed)) / float(params.height - 1);  // Flip Y

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

		for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
			// Initialize payload
			PathTracingPayload payload;
			payload.attenuation = throughput;
			payload.emission = make_float3(0.0f, 0.0f, 0.0f);
			payload.seed = seed;
			payload.depth = depth;
			payload.scattered = false;

			// Trace ray - pack 12 payload registers
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
				p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11
			);

			// Unpack payload (12 registers)
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
			float t_hit = __uint_as_float(p11);  // Hit distance

			// Add emission from this hit (weighted by throughput)
			radiance = radiance + throughput * payload.emission;

			// Decode flag: 0=absorbed, 1=scattered, 2=hit_light
			if (flag == 2) {
				// Hit a light - terminate path
				break;
			} else if (flag == 1) {
				// Scattered - compute scatter origin and update for next bounce
				float3 hit_point = ray_origin + t_hit * ray_direction;
				float3 scatter_origin = hit_point + 0.01f * normalize(payload.scatterDir);

				// Multiply throughput by surface BRDF (attenuation from hit program)
				throughput = throughput * payload.attenuation;
				ray_origin = scatter_origin;
				ray_direction = normalize(payload.scatterDir);  // MUST normalize!
				seed = payload.seed;
			} else {
				// Absorbed
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
