// optix_device_helpers.h -- Shared device utilities for OptiX programs
// Included by optix_programs.cu

// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"    // Shared exact Fresnel (CPU+GPU)
#include "../../src/shared/microfacet.h" // GGX TrowbridgeReitz (CPU+GPU)
#include "../../src/shared/bxdfs.h"      // HairBxDF<T> (CPU+GPU) - see MaterialType::Hair
#include "../../src/shared/noise.h"      // Perlin turbulence (CPU+GPU) - see sample_texture()
#include "../../src/shared/normal_map.h" // apply_normal_map (CPU+GPU) - see MaterialType::NormalMappedLambertian

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

// Rejection-sampled point in the unit disk (z=0). Mirrors src/TheRestOfYourLife/
// vec3.h's random_in_unit_disk() - used by the thin-lens depth-of-field camera.
__device__ __forceinline__ float3 random_in_unit_disk(unsigned int& seed) {
	while (true) {
		float3 p = make_float3(2.0f * random_float(seed) - 1.0f, 2.0f * random_float(seed) - 1.0f, 0.0f);
		if (dot(p, p) < 1.0f) return p;
	}
}

// Henyey-Greenstein phase function importance sampling. `wo` is the
// direction of travel (incoming ray direction, forward); returns a new
// travel direction sampled relative to it - g>0 biases toward continuing
// forward (near wo), g<0 biases backward, g=0 is isotropic. Mirrors
// src/shared/volume_scattering.h's HenyeyGreensteinPhaseFunction (itself a
// port of pbrt-v4's HGPhaseFunction::Sample_p).
__device__ __forceinline__ float3 sample_henyey_greenstein(const float3& wo, float g, unsigned int& seed) {
	float u1 = random_float(seed), u2 = random_float(seed);
	float cos_theta;
	if (fabsf(g) < 1e-3f) {
		cos_theta = 1.0f - 2.0f * u1;
	} else {
		float sqr = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
		cos_theta = -(1.0f + g * g - sqr * sqr) / (2.0f * g);
	}
	float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
	float phi = 2.0f * 3.14159265358979323846f * u2;
	float3 t1 = (fabsf(wo.x) > 0.9f) ? normalize(cross(make_float3(0, 1, 0), wo))
									  : normalize(cross(make_float3(1, 0, 0), wo));
	float3 t2 = cross(wo, t1);
	return normalize(sin_theta * cosf(phi) * t1 + sin_theta * sinf(phi) * t2 + cos_theta * wo);
}

// MaterialType::Hair: Marschner/Chiang fiber scattering (src/shared/
// bxdfs_hair.h's HairBxDF<T>), using the shading normal as a fiber-tangent
// proxy - matches src/TheRestOfYourLife/hair_material.h::scatter() exactly
// (same "no literal fiber geometry" simplification, see MaterialType::Hair's
// comment in optix_types.h). Returns false if the sample should be rejected
// (mirrors hair_material.h's `if (!res.valid) return false;`).
__device__ __forceinline__ bool sample_hair_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	HairBxDF<float> bxdf(
		random_float(seed) * 2.0f - 1.0f,  // h in [-1,1], sampled per-scatter like the CPU
		mat.ior,                            // eta
		mat.albedo.x, mat.albedo.y, mat.albedo.z,  // sigma_a (RGB absorption)
		mat.fuzz,                           // beta_m
		mat.eta_c.x,                        // beta_n
		mat.eta_c.y);                       // alpha_deg

	float3 unit_dir = normalize(ray_dir);
	float u1 = random_float(seed), u2 = random_float(seed);
	float u3 = random_float(seed), u4 = random_float(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3, u4);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
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

// Schlick's approximation removed — FrDielectric<float> from shared/fresnel.h is used instead.

// Smooth dielectric reflect-or-refract, shared by MaterialType::Dielectric
// (shade_material's case below) and MaterialType::DielectricMedium
// (optix_intersection_sphere.h's inline handling, which needs this same
// surface interaction at both the entry AND exit surface, alongside its own
// medium free-path sampling that doesn't fit shade_material's generic
// per-material switch). `front_face` selects the eta ratio direction; `ior`
// is the material's index of refraction on the denser side.
__device__ __forceinline__ float3 dielectric_scatter(const float3& ray_dir, const float3& normal,
		bool front_face, float ior, unsigned int& seed) {
	float ri = front_face ? (1.0f / ior) : ior;
	float3 unit_direction = normalize(ray_dir);
	float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	bool cannot_refract = ri * sin_theta > 1.0f;

	// FrDielectric expects eta_t/eta_i; ri = eta_i/eta_t, so pass 1/ri
	if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
		return reflect(unit_direction, normal);
	} else {
		return refract(unit_direction, normal, ri);
	}
}

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

// Evaluate one punctual (point/spot/distant) light at shading point p:
// direction toward the light (wi), incident radiance (Li), and shadow-ray
// max distance (t_max). Returns false if the light contributes nothing here
// (e.g. outside a spot's cone) so the caller can skip the shadow ray.
// Equal-area sphere->square mapping, used by the goniometric light's image
// lookup. Direct copy of src/shared/sampling_patched.h's EqualAreaSphereToSquare
// (CPU_GPU-tagged there too, but that header pulls in ~1400 lines of mostly
// CPU-only sampling code not needed here - duplicated locally instead,
// matching this file's existing pattern of small self-contained device
// helpers rather than pulling in unrelated shared headers).
__device__ __forceinline__ void dev_equal_area_sphere_to_square(
	double wx, double wy, double wz, double& u, double& v
) {
	double x = fabs(wx), y = fabs(wy), z = fabs(wz);
	double r = sqrt(fmax(0.0, 1.0 - z));
	double a = (x > y) ? x : y;
	double b = (x > y) ? y : x;
	b = (a == 0.0) ? 0.0 : b / a;

	const double t1 =  0.406758566246788489601959989e-5;
	const double t2 =  0.636226545274016134946890922156;
	const double t3 =  0.61572017898280213493197203466e-2;
	const double t4 = -0.247333733281268944196501420480;
	const double t5 =  0.881770664775316294736387951347e-1;
	const double t6 =  0.419038818029165735901852432784e-1;
	const double t7 = -0.251390972343483509333252996350e-1;
	double phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))));
	if (x < y) phi = 1.0 - phi;

	double vv = phi * r;
	double uu = r - vv;
	if (wz < 0.0) { double tmp = uu; uu = 1.0 - vv; vv = 1.0 - tmp; }
	uu = copysign(uu, wx);
	vv = copysign(vv, wy);
	u = 0.5*(uu + 1.0);
	v = 0.5*(vv + 1.0);
}

// Mirrors src/TheRestOfYourLife/punctual_light_objects.h's PunctualLiSample /
// sample_direct() on the CPU - pdf is always 1 for these delta lights, so
// callers add the contribution directly with no MIS weight or pdf division
// (see camera.h's punct_lights NEE block for the reference formula).
__device__ __forceinline__ bool eval_punctual_light(
	const PunctualLightGPU& light,
	const float3& p,
	float3& wi,
	float3& Li,
	float& t_max
) {
	float Lr = 0.0f, Lg = 0.0f, Lb = 0.0f, wx = 0.0f, wy = 0.0f, wz = 0.0f;
	switch (light.kind) {
		case PunctualLightKind::Point: {
			light.point.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.point.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.point.pos_x - p.x, dy = light.point.pos_y - p.y, dz = light.point.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Spot: {
			light.spot.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.spot.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.spot.pos_x - p.x, dy = light.spot.pos_y - p.y, dz = light.spot.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Distant: {
			light.distant.sample_wi(wx, wy, wz);
			light.distant.eval_Li(Lr, Lg, Lb);
			t_max = 1e30f;  // no finite geometric distance for a directional light
			break;
		}
		case PunctualLightKind::Goniometric: {
			const GoniometricLightGPU& g = light.gonio;
			float dx = g.pos_x - p.x, dy = g.pos_y - p.y, dz = g.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			// Direction from light toward shading point, rotated into light
			// space (mirrors GoniometricLight::eval_I(-wi) called from sample_li).
			float lx = g.world_to_light[0]*(-wx) + g.world_to_light[1]*(-wy) + g.world_to_light[2]*(-wz);
			float ly = g.world_to_light[3]*(-wx) + g.world_to_light[4]*(-wy) + g.world_to_light[5]*(-wz);
			float lz = g.world_to_light[6]*(-wx) + g.world_to_light[7]*(-wy) + g.world_to_light[8]*(-wz);
			double u, v;
			dev_equal_area_sphere_to_square((double)lx, (double)ly, (double)lz, u, v);
			int iu = (int)(u * g.nu); iu = iu < 0 ? 0 : (iu >= g.nu ? g.nu - 1 : iu);
			int iv = (int)(v * g.nv); iv = iv < 0 ? 0 : (iv >= g.nv ? g.nv - 1 : iv);
			float gonio = g.image[iv * g.nu + iu];
			float weight = g.scale * gonio / r2;
			Lr = g.ir * weight; Lg = g.ig * weight; Lb = g.ib * weight;
			break;
		}
		case PunctualLightKind::Projection: {
			const ProjectionLightGPU& pr = light.proj;
			float dx = pr.pos_x - p.x, dy = pr.pos_y - p.y, dz = pr.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			// Direction from light toward shading point, rotated into light space.
			float lx = pr.world_to_light[0]*(-wx) + pr.world_to_light[1]*(-wy) + pr.world_to_light[2]*(-wz);
			float ly = pr.world_to_light[3]*(-wx) + pr.world_to_light[4]*(-wy) + pr.world_to_light[5]*(-wz);
			float lz = pr.world_to_light[6]*(-wx) + pr.world_to_light[7]*(-wy) + pr.world_to_light[8]*(-wz);
			if (lz < pr.hither) return false;
			// screenFromLight reduces to this for make_perspective()'s matrix
			// shape (see ProjectionLightGPU's comment in optix_types.h).
			float sx = pr.inv_tan * lx / lz;
			float sy = pr.inv_tan * ly / lz;
			if (sx < pr.sb_xmin || sx > pr.sb_xmax || sy < pr.sb_ymin || sy > pr.sb_ymax) return false;
			float u = (sx - pr.sb_xmin) / (pr.sb_xmax - pr.sb_xmin);
			float v = (sy - pr.sb_ymin) / (pr.sb_ymax - pr.sb_ymin);
			int iu = (int)(u * pr.nx); iu = iu < 0 ? 0 : (iu >= pr.nx ? pr.nx - 1 : iu);
			int iv = (int)(v * pr.ny); iv = iv < 0 ? 0 : (iv >= pr.ny ? pr.ny - 1 : iv);
			int idx = (iv * pr.nx + iu) * 3;
			float rC = fmaxf(0.0f, pr.image_rgb[idx + 0]);
			float gC = fmaxf(0.0f, pr.image_rgb[idx + 1]);
			float bC = fmaxf(0.0f, pr.image_rgb[idx + 2]);
			float inv_r2 = 1.0f / r2;
			Lr = pr.scale * rC * inv_r2; Lg = pr.scale * gC * inv_r2; Lb = pr.scale * bC * inv_r2;
			break;
		}
		default:
			return false;
	}
	if (Lr <= 0.0f && Lg <= 0.0f && Lb <= 0.0f) return false;
	wi = make_float3(wx, wy, wz);
	Li = make_float3(Lr, Lg, Lb);
	return true;
}

// Add every punctual light's direct contribution at a Lambertian hit to
// `emission` (accumulated in-place). `normal` must be the shading normal
// (front-facing). Call from the same place area-light NEE happens.
__device__ __forceinline__ void add_punctual_lights_lambertian(
	const float3& hit_point,
	const float3& normal,
	const float3& albedo,
	float3& emission
) {
	for (unsigned int i = 0; i < params.numPunctualLights; ++i) {
		float3 wi, Li; float t_max;
		if (!eval_punctual_light(params.punctualLights[i], hit_point, wi, Li, t_max)) continue;
		float cos_theta = dot(wi, normal);
		if (cos_theta <= 0.0f) continue;
		if (trace_shadow_ray(hit_point, wi, t_max)) {
			float3 brdf = albedo / 3.14159265358979323846f;
			emission = emission + brdf * Li * cos_theta;
		}
	}
}

// Samples a texture by index (see TextureData in optix_types.h), matching
// CPU's texture::value(u, v, p) dispatch (src/TheRestOfYourLife/texture.h)
// for the two kinds ported so far - only called when
// MaterialData::textureIdx >= 0 (Lambertian only for now, see
// shade_material() below). u/v are used for Image; p (world-space hit
// point) is used for Noise - CPU's own base-class interface takes all
// three regardless of which the concrete texture subclass actually needs,
// same here.
__device__ __forceinline__ float3 sample_texture(int textureIdx, float u, float v, const float3& p) {
	const TextureData& tex = params.textures[textureIdx];
	if (tex.kind == TextureKind::Image) {
		// Matches image_texture::value() (texture.h:74-88) exactly: clamp
		// uv to [0,1], flip v (stored image rows are top-to-bottom, v=0 is
		// the bottom of the [0,1] texture-coordinate convention), clamp the
		// resulting integer pixel index to the image bounds (rtw_image::
		// pixel_data()'s own clamp), nearest-neighbor, 8-bit -> [0,1] float.
		// A failed image load (width/height <= 0) matches CPU's own solid-
		// cyan debugging fallback (texture.h:76) exactly.
		if (tex.width <= 0 || tex.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uc = fminf(fmaxf(u, 0.0f), 1.0f);
		const float vc = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
		const int i = min(static_cast<int>(uc * tex.width), tex.width - 1);
		const int j = min(static_cast<int>(vc * tex.height), tex.height - 1);
		const unsigned char* px = params.texturePixels + tex.pixelOffset + (j * tex.width + i) * 3;
		constexpr float kColorScale = 1.0f / 255.0f;
		return make_float3(px[0] * kColorScale, px[1] * kColorScale, px[2] * kColorScale);
	} else if (tex.kind == TextureKind::Checker) {
		// Matches checker_texture::value() (texture.h:53-61) exactly: floor
		// each world-space coordinate scaled by 1/scale, sum the three
		// integers, and pick a color by parity - equality-to-zero on `%2`
		// is sign-agnostic, so this is correct for negative coordinates too
		// (just like the CPU version, which relies on the same C++ rule).
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else {
		// Matches noise_texture::value() (texture.h:127-129) exactly:
		// color(.5,.5,.5) * (1 + sin(scale*p.z + 10*turb(p,7))), where
		// turb(p,7) is perlin::turb's own default (depth=7, omega=0.5,
		// perlin.h:37) delegating to turbulence_simple<T> (noise.h:252).
		const float turb = turbulence_simple<float>(p.x, p.y, p.z, 0.5f, 7);
		const float s = 0.5f * (1.0f + sinf(tex.noiseScale * p.z + 10.0f * turb));
		return make_float3(s, s, s);
	}
}

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
__device__ __forceinline__ void shade_material(
	const MaterialData& mat,
	const float3& normal,
	const float3& ray_dir,
	const float3& hit_point,
	bool front_face,
	float uv_u, float uv_v,
	unsigned int& seed,
	float3& out_attenuation,
	float3& out_scattered_dir,
	bool& out_scattered,
	bool& out_is_specular,
	float& out_brdf_pdf_override,
	float3& emission
) {
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
			attenuation = (mat.textureIdx >= 0) ? sample_texture(mat.textureIdx, uv_u, uv_v, hit_point) : mat.albedo;
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
			scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
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

				// NEE: direct light with the coat-weighted diffuse BSDF -
				// this branch sets is_specular=false (it's a real diffuse
				// lobe under the coat) but was missing the light-sampling
				// block every other is_specular=false material has
				// (Lambertian above, NormalizedFresnel below), so it could
				// only ever get illuminated by BSDF-sampled paths that
				// randomly happened to hit the light - no importance
				// sampling toward it at all. That made it render far too
				// dark, especially for a small/hard-to-hit area light.
				// T_in (F_in) is fixed for this scattering event (from the
				// same coat sample drawn above); F_out is re-evaluated at
				// the light-sampled direction, mirroring how the BSDF
				// sample's own F_out is evaluated at diff_dir above.
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

	out_attenuation = attenuation;
	out_scattered_dir = scattered_dir;
	out_scattered = scattered;
	out_is_specular = is_specular;
	out_brdf_pdf_override = brdf_pdf_override;
}

// Realistic multi-element lens camera (pbrt-v4 RealisticCamera, src/shared/
// cameras.h). Host-side precompute (focus-adjusted lens table + exit-pupil
// bounds table) happens once in scene_builder.cpp by directly constructing a
// RealisticCamera<float> and reading its accessors - this device function
// only ports the per-ray hot path: film-plane mapping, exit-pupil sampling,
// and the per-element Snell's-law trace (Woop et al. watertight lens
// intersection, mirrors cameras.h's trace_lenses_from_film/sample_exit_pupil/
// generate_ray exactly, including the eta_i/eta_t Refract convention fix).
// u,v are in [0,1]^2 with v already flipped by the raygen caller (matching
// the other CameraKinds' "lower-left-origin" viewport convention) - unlike
// those, RealisticCamera's film coordinates increase top-to-bottom matching
// raw raster row order (see raster_to_film's comment in cameras.h), so v is
// un-flipped back here first.
// Returns false (weight left at 0) if the ray is fully vignetted.
__device__ __forceinline__ bool sample_realistic_camera_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& out_origin, float3& out_direction, float& out_weight
) {
	out_weight = 0.0f;
	if (cam.numLensElements <= 0 || cam.numExitPupilBounds <= 0) return false;

	float v_raw = 1.0f - v;  // undo raygen's lower-left-origin flip
	float pfx = -((2.0f*u - 1.0f) * cam.film_half_x);  // pbrt-v4 negates x
	float pfy =   (2.0f*v_raw - 1.0f) * cam.film_half_y;

	// sample_exit_pupil
	float rFilm = sqrtf(pfx*pfx + pfy*pfy);
	float film_diag = 2.0f * sqrtf(cam.film_half_x*cam.film_half_x + cam.film_half_y*cam.film_half_y);
	int sz = cam.numExitPupilBounds;
	int rIndex = (int)(rFilm / (film_diag*0.5f) * (float)sz);
	if (rIndex >= sz) rIndex = sz - 1;
	if (rIndex < 0) rIndex = 0;
	const GpuExitPupilBounds& b = cam.exitPupilBounds[rIndex];
	if (b.degenerate != 0) return false;

	float area = (b.xMax - b.xMin) * (b.yMax - b.yMin);
	if (area <= 0.0f) return false;
	float ppdf = 1.0f / area;

	float u0 = random_float(seed), u1 = random_float(seed);
	float lx = b.xMin + u0*(b.xMax - b.xMin);
	float ly = b.yMin + u1*(b.yMax - b.yMin);

	float sinTheta = (rFilm > 0.0f) ? pfy/rFilm : 0.0f;
	float cosTheta0 = (rFilm > 0.0f) ? pfx/rFilm : 1.0f;
	float ppx = cosTheta0*lx - sinTheta*ly;
	float ppy = sinTheta*lx + cosTheta0*ly;
	float ppz = cam.lens_rear_z;

	float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
	float rLen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);

	// trace_lenses_from_film: camera space (film z=0, +z toward scene) ->
	// lens space (z flipped): loz=-oz, ldz=-dz.
	float lox = pfx, loy = pfy, loz = 0.0f;
	float ldx = rdx, ldy = rdy, ldz = -rdz;
	float elementZ = 0.0f;

	for (int i = cam.numLensElements - 1; i >= 0; --i) {
		const GpuLensElement& el = cam.lensElements[i];
		elementZ -= el.thickness;
		bool isStop = (el.curvatureRadius == 0.0f);
		float t, nx = 0.0f, ny = 0.0f, nz = 0.0f;

		if (isStop) {
			if (ldz == 0.0f) return false;
			t = (elementZ - loz) / ldz;
			if (t < 0.0f) return false;
		} else {
			// intersect_spherical
			float zCenter = elementZ + el.curvatureRadius;
			float cox = lox, coy = loy, coz = loz - zCenter;
			float A = ldx*ldx + ldy*ldy + ldz*ldz;
			float B = 2.0f*(ldx*cox + ldy*coy + ldz*coz);
			float C = cox*cox + coy*coy + coz*coz - el.curvatureRadius*el.curvatureRadius;
			float disc = B*B - 4.0f*A*C;
			if (disc < 0.0f) return false;
			float sq = sqrtf(disc);
			float q = (B < 0.0f) ? -0.5f*(B - sq) : -0.5f*(B + sq);
			float t0 = q / A;
			float t1 = C / q;
			if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
			bool useCloserT = (ldz > 0.0f) != (el.curvatureRadius < 0.0f);
			t = useCloserT ? fminf(t0, t1) : fmaxf(t0, t1);
			if (t < 0.0f) return false;
			float hx0 = lox+t*ldx, hy0 = loy+t*ldy, hz0 = loz+t*ldz;
			nx = hx0; ny = hy0; nz = hz0 - zCenter;
			float nlen = sqrtf(nx*nx+ny*ny+nz*nz);
			if (nlen == 0.0f) return false;
			nx/=nlen; ny/=nlen; nz/=nlen;
			if (ldx*nx+ldy*ny+ldz*nz > 0.0f) { nx=-nx; ny=-ny; nz=-nz; }
		}

		float hx = lox+t*ldx, hy = loy+t*ldy, hz = loz+t*ldz;
		if (hx*hx + hy*hy > el.apertureRadius*el.apertureRadius) return false;
		lox = hx; loy = hy; loz = hz;

		if (!isStop) {
			float eta_i = (el.eta == 0.0f) ? 1.0f : el.eta;
			float eta_t = (i > 0 && cam.lensElements[i-1].eta != 0.0f) ? cam.lensElements[i-1].eta : 1.0f;
			float len = sqrtf(ldx*ldx+ldy*ldy+ldz*ldz);
			float dxn = ldx/len, dyn = ldy/len, dzn = ldz/len;
			float eta = eta_i/eta_t;  // matches cameras.h Refract's eta=eta_i/eta_t contract
			float cosI = -(dxn*nx+dyn*ny+dzn*nz);
			float sin2T = eta*eta * fmaxf(0.0f, 1.0f - cosI*cosI);
			if (sin2T >= 1.0f) return false;
			float cosT = sqrtf(1.0f - sin2T);
			ldx = eta*dxn + (eta*cosI - cosT)*nx;
			ldy = eta*dyn + (eta*cosI - cosT)*ny;
			ldz = eta*dzn + (eta*cosI - cosT)*nz;
		}
	}

	float lensOutOx = lox, lensOutOy = loy, lensOutOz = -loz;
	float lensOutDx = ldx, lensOutDy = ldy, lensOutDz = -ldz;

	float cosThetaW = (rLen > 0.0f) ? fabsf(rdz/rLen) : 0.0f;
	float lrz = cam.lens_rear_z;
	if (lrz <= 0.0f) return false;
	float w = (cosThetaW*cosThetaW*cosThetaW*cosThetaW) / (ppdf * lrz * lrz);

	// Camera space -> world space via su(right)/sv(up)/sw(forward)/origin.
	out_origin = cam.origin + lensOutOx*cam.su + lensOutOy*cam.sv + lensOutOz*cam.sw;
	out_direction = normalize(lensOutDx*cam.su + lensOutDy*cam.sv + lensOutDz*cam.sw);
	out_weight = w;
	return true;
}

// Generate a primary camera ray for raster coordinates (u,v) in [0,1]^2,
// dispatching on params.camera.kind. Mirrors the CPU camera models in
// src/shared/cameras.h (OrthographicCamera/SphericalCamera/RealisticCamera::
// generate_ray) and src/TheRestOfYourLife/camera.h's book-style
// defocus_angle/focus_dist thin-lens DOF, which the Perspective case folds
// in via camera.defocus_disk_u/v (both zero = DOF disabled, matching how
// scene_builder.cpp always zero-initializes GpuCameraParams). `weight`
// applies to Realistic only (cos^4(theta)/pdf vignetting term, 1.0 for every
// other CameraKind - fold into the caller's initial path throughput).
__device__ __forceinline__ void generate_primary_ray(
	float u, float v, unsigned int& seed, float3& origin, float3& direction, float& weight
) {
	const GpuCameraParams& cam = params.camera;
	weight = 1.0f;
	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			// pbrt-v4 SphericalCamera::GenerateRay (EquiRectangular mapping):
			// theta in [0,pi], phi in [0,2pi], then swap(dir.y, dir.z) - see
			// src/shared/cameras.h for the reference this mirrors.
			float theta = 3.14159265358979323846f * v;
			float phi   = 2.0f * 3.14159265358979323846f * u;
			float sin_t = sinf(theta), cos_t = cosf(theta);
			float lx = sin_t * cosf(phi);
			float ly = cos_t;             // swapped with lz below (pbrt-v4 convention)
			float lz = sin_t * sinf(phi); // swapped with ly above
			origin = cam.origin;
			direction = normalize(lx * cam.su + ly * cam.sv + lz * cam.sw);
			break;
		}
		case CameraKind::Realistic: {
			if (!sample_realistic_camera_ray(cam, u, v, seed, origin, direction, weight)) {
				origin = cam.origin;
				direction = cam.sw;  // arbitrary valid direction; weight=0 zeroes its contribution
				weight = 0.0f;
			}
			break;
		}
		default: { // Perspective, optionally thin-lens DOF
			float3 pixel_sample = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			bool hasDOF = (cam.defocus_disk_u.x != 0.0f || cam.defocus_disk_u.y != 0.0f || cam.defocus_disk_u.z != 0.0f ||
						   cam.defocus_disk_v.x != 0.0f || cam.defocus_disk_v.y != 0.0f || cam.defocus_disk_v.z != 0.0f);
			if (hasDOF) {
				float3 p = random_in_unit_disk(seed);
				origin = cam.origin + p.x * cam.defocus_disk_u + p.y * cam.defocus_disk_v;
			} else {
				origin = cam.origin;
			}
			direction = normalize(pixel_sample - origin);
			break;
		}
	}
}

//==============================================================================
// Sphere Intersection Program
//==============================================================================

