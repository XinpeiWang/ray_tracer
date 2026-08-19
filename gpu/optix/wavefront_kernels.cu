// wavefront_kernels.cu
// CUDA compute kernels for the wavefront GPU path tracer.
//
// These kernels run between OptiX trace calls and handle the parts that
// don't need ray traversal:
//   1. generate_camera_rays  - fill the initial RayQueue for a given sample
//   2. evaluate_materials    - shade hit results, write shadow + next-ray items
//   3. accumulate_miss       - add background radiance for escaped rays
//   4. accumulate_shadow     - add pending Ld contributions that passed shadow test
//   5. reset_queue_counter   - zero an atomic counter before a new phase
//
// The evaluate_materials kernel intentionally mirrors the material evaluation
// logic in optix_programs.cu (closesthit) but operates on the pre-filled
// HitQueue instead of within an OptiX program.  This removes the recursion
// and allows all material evaluations to proceed in one wide CUDA kernel.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_types.h"
#include "optix_types.h"
#include "spectral_device.h"
#include "sampled_spectrum.h"
#include "spectrum_types.h"
#include "color_utils.h"
#include "optix_math_helpers.h"
#include "fresnel.h"
#include "microfacet.h"
#include "math_utils.h"
#include "bxdfs.h"  // HairBxDF<T>/PrincipledBxDF<T> - see MaterialType::Hair/Principled
#include "../../src/shared/noise.h"       // turbulence_simple - see wf_sample_texture()
#include "../../src/shared/normal_map.h"  // apply_normal_map - see MaterialType::NormalMappedLambertian
#include "../../src/shared/bilinear_patch.h"  // blp_sample - see wf_sample_bilinear_patch_light()
#include "../../src/shared/shading_frame.h"   // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured

// ============================================================================
// Device helpers (shared with optix_programs.cu logic)
// ============================================================================

// GPU-only cloud density: matches optix_intersection_sphere.h's
// gpu_cloud_density exactly (see that copy's comment for the full
// reasoning - calling CloudMedium::compute_density()'s member function
// directly stalled mid-computation on the recursive backend; this hand-
// duplicated 5-octave-FBm-only version, without CPU's wispiness/dnoise
// perturbation, is proven correct and fast on both GPU backends). Own
// definition here since this is a separate translation unit from
// optix_programs.cu.
__device__ __forceinline__ float gpu_cloud_density(const CloudMedium<float>& cloud,
													 float mx, float my, float mz) {
	float ppx = cloud.frequency * mx;
	float ppy = cloud.frequency * my;
	float ppz = cloud.frequency * mz;
	float d = 0.0f;
	float omega = 0.5f, lambda = 1.0f;
	for (int oct = 0; oct < 5; ++oct) {
		d += omega * perlin_noise<float>(lambda * ppx, lambda * ppy, lambda * ppz);
		omega *= 0.5f;
		lambda *= 1.99f;
	}
	d = fminf(1.0f, fmaxf(0.0f, (1.0f - my) * 4.5f * cloud.density * d));
	float extra = 2.0f * fmaxf(0.0f, 0.5f - my);
	return fminf(1.0f, fmaxf(0.0f, d + extra));
}

// Matches optix_intersection_sphere.h's gpu_rgb_grid_at/gpu_rgb_grid_
// trilinear exactly - see that copy's comment for why this is a from-
// scratch reimplementation rather than a call into RGBGridMediumData<T>/
// SampledGrid<T> (std::vector-backed, host-only). Own definition here since
// this is a separate translation unit from optix_programs.cu.
__device__ __forceinline__ float gpu_rgb_grid_at(const float* d, int nx, int ny, int nz,
												   int x, int y, int z) {
	x = x < 0 ? 0 : (x >= nx ? nx-1 : x);
	y = y < 0 ? 0 : (y >= ny ? ny-1 : y);
	z = z < 0 ? 0 : (z >= nz ? nz-1 : z);
	return d[x + nx * (y + ny * z)];
}

__device__ __forceinline__ float gpu_rgb_grid_trilinear(const float* d, int nx, int ny, int nz,
														  float px, float py, float pz) {
	float gx = px*nx - 0.5f, gy = py*ny - 0.5f, gz = pz*nz - 0.5f;
	int ix0 = (int)floorf(gx), iy0 = (int)floorf(gy), iz0 = (int)floorf(gz);
	float fx = gx-ix0, fy = gy-iy0, fz = gz-iz0;
	float c000 = gpu_rgb_grid_at(d,nx,ny,nz, ix0,   iy0,   iz0);
	float c100 = gpu_rgb_grid_at(d,nx,ny,nz, ix0+1, iy0,   iz0);
	float c010 = gpu_rgb_grid_at(d,nx,ny,nz, ix0,   iy0+1, iz0);
	float c110 = gpu_rgb_grid_at(d,nx,ny,nz, ix0+1, iy0+1, iz0);
	float c001 = gpu_rgb_grid_at(d,nx,ny,nz, ix0,   iy0,   iz0+1);
	float c101 = gpu_rgb_grid_at(d,nx,ny,nz, ix0+1, iy0,   iz0+1);
	float c011 = gpu_rgb_grid_at(d,nx,ny,nz, ix0,   iy0+1, iz0+1);
	float c111 = gpu_rgb_grid_at(d,nx,ny,nz, ix0+1, iy0+1, iz0+1);
	float c00 = c000 + fx*(c100-c000);
	float c10 = c010 + fx*(c110-c010);
	float c01 = c001 + fx*(c101-c001);
	float c11 = c011 + fx*(c111-c011);
	float c0 = c00 + fy*(c10-c00);
	float c1 = c01 + fy*(c11-c01);
	return c0 + fz*(c1-c0);
}

__device__ __forceinline__ unsigned int wf_pcg(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ float wf_rand(unsigned int& seed) {
	seed = wf_pcg(seed);
	return float(seed) / 4294967296.0f;
}

__device__ __forceinline__ float3 wf_rand3(unsigned int& seed) {
	return make_float3(wf_rand(seed), wf_rand(seed), wf_rand(seed));
}

// Builds two independent 64-bit seeds (mirrors optix_device_helpers.h's
// random_seed64_pair()) for the layered_detail::PCG32 RNG that
// CoatedDiffuseBxDF::f()/CoatedConductorBxDF::f() (src/shared/bxdfs_layered.h)
// run their internal random walk with - used when NEE needs a fresh
// stochastic f() evaluation toward a light direction.
__device__ __forceinline__ void wf_random_seed64_pair(unsigned int& seed, uint64_t& s0, uint64_t& s1) {
	unsigned int a = (seed = wf_pcg(seed));
	unsigned int b = (seed = wf_pcg(seed));
	unsigned int c = (seed = wf_pcg(seed));
	unsigned int d = (seed = wf_pcg(seed));
	s0 = (uint64_t(a) << 32) | uint64_t(b);
	s1 = (uint64_t(c) << 32) | uint64_t(d);
}

__device__ __forceinline__ float3 wf_rand_unit(unsigned int& seed) {
	while (true) {
		float3 p = 2.0f * wf_rand3(seed) - make_float3(1.0f, 1.0f, 1.0f);
		float  l = dot(p, p);
		if (l > 1e-8f && l < 1.0f) return p / sqrtf(l);
	}
}

__device__ __forceinline__ bool wf_near_zero(const float3& v) {
	const float s = 1e-8f;
	return fabsf(v.x) < s && fabsf(v.y) < s && fabsf(v.z) < s;
}

// Henyey-Greenstein phase function direction sample. Duplicated from
// optix_device_helpers.h's sample_henyey_greenstein (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path.
__device__ __forceinline__ float3 wf_sample_henyey_greenstein(const float3& wo, float g, unsigned int& seed) {
	float u1 = wf_rand(seed), u2 = wf_rand(seed);
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

// Henyey-Greenstein phase function VALUE p(cos_theta, g) - duplicated from
// optix_device_helpers.h's hg_phase_value (with the wf_ prefix), matching
// this file's existing pattern of not sharing device helpers with the
// recursive path. Used by MaterialType::DielectricMedium's medium-interior
// phase-scatter case below to do real NEE+MIS at that scatter event - see
// that call site's own comment for why (closing B13's CPU-vs-GPU brightness
// gap; this is the same quantity as both the phase "BSDF" f and its own
// pdf, since a normalized phase function is its own perfect importance
// sampler).
__device__ __forceinline__ float wf_hg_phase_value(float cos_theta, float g) {
	float gc = fminf(0.99f, fmaxf(-0.99f, g));
	const float inv4pi = 1.0f / (4.0f * 3.14159265358979323846f);
	float denom = 1.0f + gc * gc + 2.0f * gc * cos_theta;
	return inv4pi * (1.0f - gc * gc) / (denom * sqrtf(fmaxf(1e-12f, denom)));
}

// MaterialType::Hair: Marschner/Chiang fiber scattering, duplicated from
// optix_device_helpers.h's sample_hair_material (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path.
__device__ __forceinline__ bool wf_sample_hair_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	HairBxDF<float> bxdf(
		wf_rand(seed) * 2.0f - 1.0f,
		mat.ior,
		mat.albedo.x, mat.albedo.y, mat.albedo.z,
		mat.fuzz,
		mat.eta_c.x,
		mat.eta_c.y);

	float3 unit_dir = normalize(ray_dir);
	float u1 = wf_rand(seed), u2 = wf_rand(seed);
	float u3 = wf_rand(seed), u4 = wf_rand(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3, u4);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// Samples a texture by index - duplicated from optix_device_helpers.h's
// sample_texture (with the wf_ prefix), matching this file's existing
// pattern of not sharing device helpers with the recursive path. Only
// called when MaterialData::textureIdx >= 0 (Lambertian/NormalMappedLambertian).
__device__ __forceinline__ float3 wf_sample_texture(
	const TextureData* textures, const unsigned char* texturePixels,
	int textureIdx, float u, float v, const float3& p)
{
	const TextureData& tex = textures[textureIdx];
	if (tex.kind == TextureKind::Image) {
		if (tex.width <= 0 || tex.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uc = fminf(fmaxf(u, 0.0f), 1.0f);
		const float vc = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
		const int i = min(static_cast<int>(uc * tex.width), tex.width - 1);
		const int j = min(static_cast<int>(vc * tex.height), tex.height - 1);
		const unsigned char* px = texturePixels + tex.pixelOffset + (j * tex.width + i) * 3;
		constexpr float kColorScale = 1.0f / 255.0f;
		return make_float3(px[0] * kColorScale, px[1] * kColorScale, px[2] * kColorScale);
	} else if (tex.kind == TextureKind::Checker) {
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else {
		const float turb = turbulence_simple<float>(p.x, p.y, p.z, 0.5f, 7);
		const float s = 0.5f * (1.0f + sinf(tex.noiseScale * p.z + 10.0f * turb));
		return make_float3(s, s, s);
	}
}

// MaterialType::Principled: Disney/pbrt-v4-style multi-lobe BSDF, duplicated
// from optix_device_helpers.h's sample_principled_material (with the wf_
// prefix) rather than shared, matching this file's existing pattern of not
// sharing device helpers with the recursive path. Field reuse: albedo=base
// color, ior=ior, fuzz=roughness, eta_c.x=metallic, eta_c.y=clearcoat,
// eta_c.z=clearcoat_rough - see MaterialType::Principled's comment in
// optix_types.h.
__device__ __forceinline__ bool wf_sample_principled_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	PrincipledBxDF<float> bxdf{
		mat.albedo.x, mat.albedo.y, mat.albedo.z,
		mat.eta_c.x,   // metallic
		mat.fuzz,      // roughness
		mat.ior,
		mat.eta_c.y,   // clearcoat
		mat.eta_c.z }; // clearcoat_rough

	float3 unit_dir = normalize(ray_dir);
	float u1 = wf_rand(seed), u2 = wf_rand(seed), u3 = wf_rand(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// MaterialType::Measured: real tabulated measured-BRDF (wavefront-native
// device math + wf_sample_measured_material(), duplicated from
// optix_measured_bxdf.h with the wf_ prefix - see that new header's own
// extensive comment). Needs wf_rand() (defined above) and ShadingFrame<T>
// (included at the top of this file), so it lives here rather than
// wavefront_probe.h - unlike BSSRDF, this material never needs an OptiX
// trace, so it does not belong in that probe-walk-specific module.
#include "wavefront_measured_bxdf.h"

// Device-side real importance-sampled HDR sky for the wavefront backend
// (mirrors optix_sky_light.h - see wavefront_sky_light.h's own header
// comment). Needs wf_rand()/wf_rand_unit() (defined above). The actual math
// (shared with the recursive backend) lives in gpu_sky_light_shared.h,
// included first.
#include "gpu_sky_light_shared.h"
#include "wavefront_sky_light.h"

// reflect/refract wrappers
__device__ __forceinline__ float3 wf_reflect(const float3& v, const float3& n) {
	return cpu_gpu_reflect(v, n);
}
__device__ __forceinline__ float3 wf_refract(const float3& v, const float3& n, float e) {
	return cpu_gpu_refract<float3, float>(v, n, e);
}

// Smooth dielectric reflect-or-refract, duplicated from optix_device_helpers.h's
// dielectric_scatter (with the wf_ prefix) - shared by MaterialType::Dielectric's
// existing inline handling (kept as-is, not routed through this) and
// MaterialType::DielectricMedium's entry-surface case below, which needs
// this exact surface interaction alongside its own medium free-path
// sampling that doesn't fit a single self-contained case block the way
// Dielectric's does.
__device__ __forceinline__ float3 wf_dielectric_scatter(
	const float3& ray_dir, const float3& normal, bool front_face, float ior, unsigned int& seed)
{
	float ri = front_face ? (1.0f / ior) : ior;
	float3 unit_direction = normalize(ray_dir);
	float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
	bool cannot_refract = ri * sin_theta > 1.0f;
	if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > wf_rand(seed)) {
		return wf_reflect(unit_direction, normal);
	} else {
		return wf_refract(unit_direction, normal, ri);
	}
}

// pbrt-v4 NormalizedFresnelBxDF direction sample - factored out of
// MaterialType::NormalizedFresnel's own switch case below so
// resolve_bssrdf_exit() (MaterialType::Subsurface's BSSRDF exit point,
// Phase 2) can reuse the identical formula without duplicating it. NEE for
// this BSDF lives in wf_finish_material_scatter() instead (keyed off
// MaterialType::NormalizedFresnel there too) - this function is only the
// BSDF-sample half (direction + weight + pdf), matching how the switch
// case below already separated the two concerns before this factoring.
__device__ __forceinline__ void wf_sample_normalized_fresnel(
	float eta, const float3& normal, unsigned int& seed,
	float3& out_dir, float& out_weight, float& out_brdf_pdf)
{
	float inv_eta = 1.0f / eta;
	float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
	if (nf_c <= 0.0f) nf_c = 1e-6f;
	float3 dir = normalize(normal + wf_rand_unit(seed));
	if (wf_near_zero(dir)) dir = normal;
	float cos_wi = fmaxf(dot(dir, normal), 1e-6f);
	float fr = FrDielectric(cos_wi, eta);
	out_dir        = dir;
	out_weight     = (1.0f - fr) / nf_c;
	out_brdf_pdf   = (1.0f - fr) * cos_wi / (nf_c * 3.14159265f);
}

// MIS power heuristic (balance if both 0)
__device__ __forceinline__ float wf_mis(float a, float b) {
	float a2 = a * a, b2 = b * b;
	return (a2 + b2 < 1e-10f) ? 1.0f : a2 / (a2 + b2);
}

// XYZ -> LINEAR sRGB (matrix only, no gamma/OETF curve).
//
// sampled_spectrum.h's XYZToSRGB() is a full linear-XYZ-to-display-sRGB
// conversion, gamma-encoding included by design (see its GammaEncodingApplied
// unit test) - correct for a caller that writes gamma-encoded pixels
// directly. This codebase's actual output path doesn't: both GPU renderers
// write into a shared linear-RGB framebuffer, and optix_interface.cpp
// applies gamma exactly once, itself, when converting that framebuffer to
// 8-bit PPM output (see its "Write pixels with gamma correction" step) -
// correct for the recursive path, whose device code (optix_intersection_
// {sphere,quad}.h) writes genuinely linear RGB. Using XYZToSRGB() here
// applied gamma a second time on top of that shared step, compounding into
// severe overexposure for anything bright enough to matter (barely visible
// for area lights' modest emission values, blown fully to white for point/
// spot lights' much larger radiance) - this stays in the same linear space
// the rest of the pipeline expects.
// No negative clamp here deliberately: this runs once per PARTIAL contribution
// (one bounce's emission, one light's NEE term, ...), and each partial XYZ->RGB
// step can legitimately dip slightly negative for saturated colors even though
// every individual spectral sample is non-negative (a standard artifact of the
// sRGB gamut being narrower than the visible-color gamut some spectra round-trip
// through). Clamping per-contribution before the caller's atomicAdd sums them
// into the framebuffer is a one-directional bias - each clamp can only push the
// running total up, never down - and it compounds with every extra bounce/light,
// which is why this was far more visible for point lights (evaluated at every
// diffuse bounce) than area lights. The single already-correct clamp on the
// fully summed per-pixel value lives downstream in optix_interface.cpp, right
// before the gamma curve; let it do the clamping.
__device__ __forceinline__ void wf_xyz_to_linear_rgb(float X, float Y, float Z,
													   float& r, float& g, float& b) {
	r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
	g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
	b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
}

// Sample a point on a sphere light; returns direction, sets geom_pdf, and
// sets maxDist to the distance to the ACTUAL sampled point on the sphere's
// surface (via ray-sphere intersection along the sampled direction) - not
// the distance to the sphere's center. Matches wf_sample_quad_light's
// contract (maxDist = distance to the sampled surface point), needed so the
// caller's shadow ray stops at the light's surface instead of continuing
// past it toward the center, where it would self-intersect the light
// sphere and register as occluded on every sample. This bug was never
// caught before because no scene had ever registered a genuinely emissive
// sphere as a light in wavefront mode until the spherical-camera scene.
__device__ float3 wf_sample_sphere_light(const SphereData& sph, const float3& hit,
										  unsigned int& seed, float& pdf, float& maxDist) {
	float3 to_c = sph.center - hit;
	float dist  = length(to_c);
	float r     = sph.radius;
	float3 dir;
	if (dist <= r) {
		pdf = 1.0f / (4.0f * 3.14159265f * r * r);
		dir = normalize(wf_rand_unit(seed));
	} else {
		float cos_max = sqrtf(fmaxf(0.0f, 1.0f - (r * r) / (dist * dist)));
		float phi     = 2.0f * 3.14159265f * wf_rand(seed);
		float cos_t   = 1.0f - wf_rand(seed) * (1.0f - cos_max);
		float sin_t   = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
		float3 w      = normalize(to_c);
		float3 u, v;
		if (fabsf(w.x) > 0.9f) u = normalize(cross(make_float3(0,1,0), w));
		else                    u = normalize(cross(make_float3(1,0,0), w));
		v = cross(w, u);
		dir = normalize(sin_t * cosf(phi) * u + sin_t * sinf(phi) * v + cos_t * w);
		float solid = 2.0f * 3.14159265f * (1.0f - cos_max);
		pdf = (solid > 1e-10f) ? 1.0f / solid : 1.0f;
	}
	// Ray-sphere intersection along `dir` from `hit` to find the true
	// surface distance (dir is constructed to be guaranteed to hit the
	// sphere, so the discriminant is never negative in practice). Take the
	// near root normally; when `hit` is inside the sphere (dist<=r above)
	// the near root is behind the origin (negative), so fall back to the
	// far root in that case.
	float3 oc = hit - sph.center;
	float b = dot(oc, dir);
	float c = dot(oc, oc) - r * r;
	float disc = fmaxf(0.0f, b * b - c);
	float sq = sqrtf(disc);
	float tNear = -b - sq;
	maxDist = (tNear > 1e-6f) ? tNear : fmaxf(0.0f, -b + sq);
	return dir;
}

// Sample a point on a quad light.
__device__ float3 wf_sample_quad_light(const QuadData& q, const float3& hit,
										unsigned int& seed, float& geom_pdf, float& maxDist) {
	float s = wf_rand(seed), t = wf_rand(seed);
	float3 p = q.Q + s * q.u + t * q.v;
	float3 dir = p - hit;
	maxDist     = length(dir);
	dir         = normalize(dir);
	float area  = length(cross(q.u, q.v));
	float cos_l = fabsf(dot(q.normal, -dir));
	geom_pdf = (cos_l > 1e-6f && maxDist > 1e-6f) ? (maxDist * maxDist) / (cos_l * area) : 0.0f;
	return dir;
}

// Sample a point on a triangle light. Same shape of answer/logic as
// sample_triangle_light() in optix_device_helpers.h (the recursive path's
// own copy - duplicated here for the same cross-module reason every other
// wf_sample_*_light function in this file is: wavefront_kernels.cu and the
// recursive path's optix_programs.cu are compiled as separate OptiX
// modules). Was entirely missing before - the NEE light-selection dispatch
// below only ever distinguished Sphere from "everything else is Quad",
// so a genuinely selected GpuLightKind::Triangle light read Quad data at
// a triangle's index instead - into an empty quads[] array for any
// OBJ/.mtl mesh scene with real Ke emission and zero actual quads, a wild
// out-of-bounds read (CUDA error 700, illegal memory access). Never
// caught before because no scene had ever registered a genuinely emissive
// OBJ triangle as a light in wavefront mode until Fireplace Room's real
// Ke materials exercised it - the exact same "first real use surfaces a
// latent gap" story as wf_sample_sphere_light's own comment above.
// GpuLightKind::BilinearPatch had this identical gap - fixed below in
// wf_sample_bilinear_patch_light(), same shape of fix, still unexercised
// by any real scene (no scene combines wavefront mode with a bilinear-patch
// light yet) but no longer a live landmine either.
__device__ float3 wf_sample_triangle_light(const TriangleData& tri, const float3& hit,
											unsigned int& seed, float& geom_pdf, float& maxDist) {
	float a = wf_rand(seed), b = wf_rand(seed);
	if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
	const float3 e1 = tri.p1 - tri.p0;
	const float3 e2 = tri.p2 - tri.p0;
	const float3 point = tri.p0 + a * e1 + b * e2;

	float3 dir = point - hit;
	maxDist = length(dir);
	if (maxDist < 1e-6f) { geom_pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	dir = dir / maxDist;

	const float3 n_unnorm = cross(e1, e2);
	const float twice_area = length(n_unnorm);
	if (twice_area < 1e-12f) { geom_pdf = 0.0f; return dir; }
	const float3 normal = n_unnorm / twice_area;
	const float area = 0.5f * twice_area;

	const float cosine = fabsf(dot(dir, normal));
	if (cosine < 1e-6f) { geom_pdf = 0.0f; return dir; }

	geom_pdf = (maxDist * maxDist) / (cosine * area);
	return dir;
}

// Sample a point on a bilinear-patch light. Thin wrapper around
// src/shared/bilinear_patch.h's blp_sample() (CPU_GPU-tagged, safe to call
// directly here - the recursive-backend member-call stall documented for
// CloudMedium::compute_density() was specific to that mega-kernel's
// closesthit and a member function; blp_sample is a free function and the
// wavefront backend was never affected by that stall in the first place).
// Same shape of answer as optix_device_helpers.h's own
// sample_bilinear_patch_light() (the recursive path's copy) - duplicated
// here for the same cross-module reason every other wf_sample_*_light
// function in this file is.
__device__ float3 wf_sample_bilinear_patch_light(const BilinearPatchData& bp, const float3& hit,
												   unsigned int& seed, float& geom_pdf, float& maxDist) {
	const float p00[3] = {bp.p00.x, bp.p00.y, bp.p00.z};
	const float p10[3] = {bp.p10.x, bp.p10.y, bp.p10.z};
	const float p01[3] = {bp.p01.x, bp.p01.y, bp.p01.z};
	const float p11[3] = {bp.p11.x, bp.p11.y, bp.p11.z};
	const float u2[2] = {wf_rand(seed), wf_rand(seed)};
	float outP[3], outN[3], areaPdf = 0.0f;
	blp_sample(p00, p10, p01, p11, u2, outP, outN, &areaPdf);

	const float3 point  = make_float3(outP[0], outP[1], outP[2]);
	const float3 normal = make_float3(outN[0], outN[1], outN[2]);
	float3 to_light = point - hit;
	float dist_sq = dot(to_light, to_light);
	maxDist = sqrtf(dist_sq);
	if (maxDist < 1e-6f || areaPdf <= 0.0f) { geom_pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	float3 dir = to_light / maxDist;

	const float cosine = fabsf(dot(dir, normal));
	if (cosine < 1e-6f) { geom_pdf = 0.0f; return dir; }

	geom_pdf = areaPdf * dist_sq / cosine;
	return dir;
}

// Equal-area sphere->square mapping for the goniometric light's image
// lookup. Duplicated from optix_device_helpers.h's dev_equal_area_sphere_to_square
// (itself a local copy of src/shared/sampling_patched.h's EqualAreaSphereToSquare -
// see that comment for why it's not shared via #include).
__device__ __forceinline__ void wf_equal_area_sphere_to_square(
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

// Forward direction of the mapping above, for the Spherical camera's
// EqualArea raygen case below. Duplicated from optix_device_helpers.h's
// dev_equal_area_square_to_sphere / dev_wrap_equal_area_square (same
// no-cross-file-sharing reason as wf_equal_area_sphere_to_square above).
__device__ __forceinline__ void wf_equal_area_square_to_sphere(
	double u, double v, double& wx, double& wy, double& wz
) {
	double uu = 2.0*u - 1.0, vv = 2.0*v - 1.0;
	double up = fabs(uu), vp = fabs(vv);
	double signed_dist = 1.0 - (up + vp);
	double d = fabs(signed_dist);
	double r = 1.0 - d;
	double phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (3.14159265358979323846 / 4.0);
	wz = copysign(1.0 - r*r, signed_dist);
	double cos_phi = cos(phi);
	double sin_phi = sin(phi);
	double xy_r = r * sqrt(fmax(0.0, 2.0 - r*r));
	wx = copysign(cos_phi * xy_r, uu);
	wy = copysign(sin_phi * xy_r, vv);
}

__device__ __forceinline__ void wf_wrap_equal_area_square(double& u, double& v) {
	if (u < 0.0) { u = -u; v = 1.0 - v; }
	else if (u > 1.0) { u = 2.0 - u; v = 1.0 - v; }
	if (v < 0.0) { u = 1.0 - u; v = -v; }
	else if (v > 1.0) { u = 1.0 - u; v = 2.0 - v; }
}

// Evaluate one punctual (point/spot/distant/goniometric/projection) light at
// shading point p: same dispatch as optix_device_helpers.h's
// eval_punctual_light(), duplicated here (with the wf_ prefix) rather than
// shared, matching this file's existing pattern of not sharing NEE helpers
// with the recursive path (this kernel has no access to the recursive
// path's __constant__ params global, and the two strategies' NEE
// application differs - RGB float3 here vs this file's spectral radiance).
// PDF is always 1 (delta light) - see punctual_lights.h.
__device__ __forceinline__ bool wf_eval_punctual_light(
	const PunctualLightGPU& light, const float3& p,
	float3& wi, float3& Li, float& t_max)
{
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
			t_max = 1e30f;
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
			float lx = g.world_to_light[0]*(-wx) + g.world_to_light[1]*(-wy) + g.world_to_light[2]*(-wz);
			float ly = g.world_to_light[3]*(-wx) + g.world_to_light[4]*(-wy) + g.world_to_light[5]*(-wz);
			float lz = g.world_to_light[6]*(-wx) + g.world_to_light[7]*(-wy) + g.world_to_light[8]*(-wz);
			double u, v;
			wf_equal_area_sphere_to_square((double)lx, (double)ly, (double)lz, u, v);
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
			float lx = pr.world_to_light[0]*(-wx) + pr.world_to_light[1]*(-wy) + pr.world_to_light[2]*(-wz);
			float ly = pr.world_to_light[3]*(-wx) + pr.world_to_light[4]*(-wy) + pr.world_to_light[5]*(-wz);
			float lz = pr.world_to_light[6]*(-wx) + pr.world_to_light[7]*(-wy) + pr.world_to_light[8]*(-wz);
			if (lz < pr.hither) return false;
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

// ============================================================================
// Kernel 1 — generate_camera_rays
//   Fills rayQueue with one primary ray per pixel for sample index `sampleIdx`.
// ============================================================================

// Realistic multi-element lens camera, wavefront duplicate of
// optix_device_helpers.h's sample_realistic_camera_ray (see that function's
// doc comment for the full algorithm derivation). generate_camera_rays
// below now flips v the same way optix_raygen.h does (py=0/top row -> v=1),
// so - exactly like the recursive path - that flip must be undone here to
// get back to the raw top-to-bottom raster convention
// RealisticCamera::raster_to_film expects.
__device__ __forceinline__ bool wf_sample_realistic_camera_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& out_origin, float3& out_direction, float& out_weight
) {
	out_weight = 0.0f;
	if (cam.numLensElements <= 0 || cam.numExitPupilBounds <= 0) return false;

	float v_raw = 1.0f - v;  // undo generate_camera_rays' lower-left-origin flip
	float pfx = -((2.0f*u - 1.0f) * cam.film_half_x);  // pbrt-v4 negates x
	float pfy =   (2.0f*v_raw - 1.0f) * cam.film_half_y;

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

	float u0 = wf_rand(seed), u1 = wf_rand(seed);
	float lx = b.xMin + u0*(b.xMax - b.xMin);
	float ly = b.yMin + u1*(b.yMax - b.yMin);

	float sinTheta = (rFilm > 0.0f) ? pfy/rFilm : 0.0f;
	float cosTheta0 = (rFilm > 0.0f) ? pfx/rFilm : 1.0f;
	float ppx = cosTheta0*lx - sinTheta*ly;
	float ppy = sinTheta*lx + cosTheta0*ly;
	float ppz = cam.lens_rear_z;

	float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
	float rLen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);

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
			float eta = eta_i/eta_t;
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

	out_origin = cam.origin + lensOutOx*cam.su + lensOutOy*cam.sv + lensOutOz*cam.sw;
	out_direction = normalize(lensOutDx*cam.su + lensOutDy*cam.sv + lensOutDz*cam.sw);
	out_weight = w;
	return true;
}

// Generate a primary camera ray, dispatching on camera.kind. Duplicated from
// optix_device_helpers.h's generate_primary_ray (with the wf_ prefix)
// rather than shared, matching this file's existing pattern of not sharing
// device helpers with the recursive path (this kernel has no access to the
// recursive path's __constant__ params global). `weight` applies to
// Realistic only (1.0 for every other CameraKind).
__device__ __forceinline__ void wf_generate_primary_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& origin, float3& direction, float& weight
) {
	weight = 1.0f;
	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			// Both mappings finish with a swap(dir.y, dir.z) folded directly
			// into which raw component feeds ly vs lz below - see
			// optix_device_helpers.h's generate_primary_ray for the same
			// pattern (including why v_sph below undoes optix_raygen.h's
			// shared Y-flip) and src/shared/cameras.h for the CPU reference.
			const float v_sph = 1.0f - v;
			float lx, ly, lz;
			if (cam.sphericalMapping == 1) {  // EqualArea
				double ud = (double)u, vd = (double)v_sph;
				wf_wrap_equal_area_square(ud, vd);
				double ewx, ewy, ewz;
				wf_equal_area_square_to_sphere(ud, vd, ewx, ewy, ewz);
				lx = (float)ewx;
				ly = (float)ewz;  // swap(wy,wz): final y = raw z
				lz = (float)ewy;  // swap(wy,wz): final z = raw y
			} else {  // EquiRectangular
				float theta = 3.14159265358979323846f * v_sph;
				float phi   = 2.0f * 3.14159265358979323846f * u;
				float sin_t = sinf(theta), cos_t = cosf(theta);
				lx = sin_t * cosf(phi);
				ly = cos_t;
				lz = sin_t * sinf(phi);
			}
			origin = cam.origin;
			direction = normalize(lx * cam.su + ly * cam.sv + lz * cam.sw);
			break;
		}
		case CameraKind::Realistic: {
			if (!wf_sample_realistic_camera_ray(cam, u, v, seed, origin, direction, weight)) {
				origin = cam.origin;
				direction = cam.sw;
				weight = 0.0f;
			}
			break;
		}
		default: { // Perspective, optionally thin-lens DOF
			float3 pixel_sample = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			bool hasDOF = (cam.defocus_disk_u.x != 0.0f || cam.defocus_disk_u.y != 0.0f || cam.defocus_disk_u.z != 0.0f ||
						   cam.defocus_disk_v.x != 0.0f || cam.defocus_disk_v.y != 0.0f || cam.defocus_disk_v.z != 0.0f);
			if (hasDOF) {
				float rx = 2.0f * wf_rand(seed) - 1.0f, ry = 2.0f * wf_rand(seed) - 1.0f;
				while (rx*rx + ry*ry >= 1.0f) { rx = 2.0f * wf_rand(seed) - 1.0f; ry = 2.0f * wf_rand(seed) - 1.0f; }
				origin = cam.origin + rx * cam.defocus_disk_u + ry * cam.defocus_disk_v;
			} else {
				origin = cam.origin;
			}
			direction = normalize(pixel_sample - origin);
			break;
		}
	}
}

extern "C" __global__ void generate_camera_rays(
	WorkQueue<RayWorkItem> rayQueue,
	unsigned int width,
	unsigned int height,
	GpuCameraParams camera,
	unsigned int sampleIdx,
	unsigned int frameNumber
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= (int)width || py >= (int)height) return;

	int pixelIdx = py * (int)width + px;
	unsigned int seed = wf_pcg(wf_pcg(pixelIdx + sampleIdx * width * height) ^ frameNumber);

	float u = (float(px) + wf_rand(seed)) / float(width  - 1);
	// Flip Y to match optix_raygen.h's lower-left-origin viewport convention
	// (py=0/top row -> v=1, matching how lower_left_corner+u*horizontal+
	// v*vertical is constructed for Perspective/Orthographic, and how
	// Spherical's theta=pi*v expects v=0 at the bottom). Without this flip
	// every wavefront-mode render using those camera kinds came out
	// vertically mirrored relative to the recursive path.
	float v = (float(height - 1 - py) + wf_rand(seed)) / float(height - 1);
	// NOTE for future CameraKind additions: same lower-left-origin `v` as
	// optix_raygen.h's __raygen__rg (see that function's matching comment) -
	// a camera whose reference model assumes raw raster order (v=0 at the
	// top row) needs to locally undo this flip before using it, the way
	// wf_generate_primary_ray's Spherical/Realistic cases already do.

	RayWorkItem item;
	float cam_weight;
	wf_generate_primary_ray(camera, u, v, seed, item.origin, item.direction, cam_weight);
	// Sample hero wavelengths for spectral rendering (pbrt-v4: SampledWavelengths::SampleVisible)
	float lambda_u = wf_rand(seed);
	SampledWavelengths<kWFNWavelengths> swl = SampledWavelengths<kWFNWavelengths>::SampleVisible(lambda_u);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		item.throughput[i]     = cam_weight;
		item.radiance[i]       = 0.0f;
		item.wavelengths[i]    = swl.lambda[i];
		item.wavelength_pdfs[i] = swl.pdf[i];
	}
	item.seed       = seed;
	item.pixelIndex = pixelIdx;
	item.depth      = 0;
	item.specular_bounce = 1;  // primary ray: always allow emissive hit
	item.brdf_pdf   = 0.0f;    // primary ray: no MIS on an escaped camera ray
	item.tMin       = 0.001f;
	item.tMax       = 1e30f;

	rayQueue.push(item);
}

// ============================================================================
// Kernel 2 — evaluate_materials
//   Processes HitWorkItems.  For each hit:
//     - Evaluates the BSDF (same logic as closesthit in optix_programs.cu)
//     - Pushes a ShadowRayWorkItem for direct-light NEE
//     - Pushes a RayWorkItem into nextRayQueue for the next bounce
//     - Emissive surfaces write directly to the framebuffer
// ============================================================================

// wf_finish_material_scatter -- NEE (area/sky/punctual lights) + next-ray-
// push tail, factored out of evaluate_materials's own per-hit switch tail
// (previously inline there) so resolve_bssrdf_exit() below (MaterialType::
// Subsurface's BSSRDF exit point, Phase 2) can reuse the EXACT same
// NormalizedFresnel-shaped NEE/bounce logic that MaterialType::
// NormalizedFresnel's own switch case in evaluate_materials already
// exercises for a genuine NormalizedFresnel hit, instead of duplicating it -
// mirrors the recursive backend's own shade_normalized_fresnel() factoring
// (optix_device_helpers.h) for the identical reason, just without that
// backend's OptiX self-recursive-call-graph constraint (this is a plain
// CUDA device function, callable from as many __global__ kernels in this
// translation unit as need it - no module compiler involved).
//
// Callers must already know `scattered == true` - evaluate_materials keeps
// its own `if (!scattered) { addToFramebuffer(...); return; }` check before
// calling this (a material can fail to scatter; this function no longer
// handles that case at all, since resolve_bssrdf_exit()'s own "did the
// probe walk find an exit point" failure is a DIFFERENT check, handled
// entirely before this function is ever called - see that kernel).
//
// `matType`/`nfEta` together select the bsdf_color/bsdf_val formula exactly
// as the NEE blocks below always did: MaterialType::Lambertian reuses
// `attenuation` directly (already the diffuse albedo, direction-independent);
// MaterialType::NormalizedFresnel (nfEta = mat.ior for a genuine hit, or the
// Subsurface material's own eta for a BSSRDF exit point - see
// resolve_bssrdf_exit()) recomputes the achromatic Fresnel-weighted formula;
// MaterialType::Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor (glossy
// - EffectivelySmooth() surfaces stay is_specular=true and never reach here)
// recompute a real per-direction, per-channel f() via the shared CPU_GPU BxDF
// templates (src/shared/bxdfs_conductor.h, bxdfs_layered.h) using `matIdx` to
// look up the full MaterialData (fuzz/ior/eta_c/k_c/albedo - `nfEta` alone
// isn't enough for these); RoughDielectric reuses the `nfEta` slot to carry
// its own precomputed eta (front_face ? 1/ior : ior) instead of mat.ior
// directly, mirroring how the two uses are already mutually exclusive.
// `phaseWo`/`matIdx`-driven materials also reuse phaseWo as `wi_world` (see
// its own comment below) to reconstruct the local shading frame; anything
// else falls back to the flat Lambertian default (every remaining non-
// specular material either caller ever reaches this with is one of the
// above - is_specular materials skip the whole NEE block, matching the
// original inline code's own `if (!is_specular)` gate). RoughDielectric's NEE
// here only covers the reflection hemisphere (cos_l > 0, same gate every
// other material gets) - unlike the recursive backend's two-sided glass NEE,
// extending the transmission side would mean threading a flip flag through
// every one of this shared function's 3 NEE blocks; skipped as a deliberate,
// documented scope reduction (still correct/unbiased, just misses the rare
// "light seen straight through frosted glass" contribution).
__device__ __forceinline__ void wf_finish_material_scatter(
	MaterialType matType, float nfEta, int matIdx,
	const float3& normal, const float3& hit_point,
	unsigned int& seed,
	const SampledSpectrum<kWFNWavelengths>& throughput,
	const SampledSpectrum<kWFNWavelengths>& radiance,
	const SampledWavelengths<kWFNWavelengths>& swl,
	const SampledSpectrum<kWFNWavelengths>& attenuation,
	const float3& scattered_dir, bool is_specular, float brdf_pdf_override,
	// Only meaningful when matType == MaterialType::DielectricMedium AND
	// is_specular == false - the one sub-case (medium-interior phase-function
	// scatter, see evaluate_materials()'s DielectricMedium case) that reaches
	// this function's NEE block with that material type at all (its other two
	// sub-cases, the entry/exit dielectric-surface refractions, are genuinely
	// specular and never clear the `if (!is_specular)` gate below). phaseWo is
	// the direction back toward where the ray came from (-incoming ray dir,
	// matching CPU hg_phase_pdf's own `wo`); phaseG is the material's HG
	// asymmetry (mat.fuzz). ALSO reused (same "-incoming ray dir" meaning) as
	// `wi_world` by the Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor
	// glossy branches above to rebuild their local shading frame - harmless/
	// unused for every other material type.
	const float3& phaseWo, float phaseG,
	int pixelIndex, int depth,
	const SphereData* spheres, const QuadData* quads,
	const TriangleData* triangles, const BilinearPatchData* bilinearPatches,
	const MaterialData* materials,
	const int* lightIndices, const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable, unsigned int numLights,
	const PunctualLightGPU* punctualLights, unsigned int numPunctualLights,
	float3 skyColor, float shadow_eps, const GpuSkyDistribution& skyDist,
	WorkQueue<ShadowRayWorkItem>& shadowQueue,
	WorkQueue<RayWorkItem>& nextRayQueue,
	float3* framebuffer)
{
	using SS = SampledSpectrum<kWFNWavelengths>;

	// See phaseWo/phaseG's own parameter comment above - matType alone
	// unambiguously identifies the medium-interior phase-scatter sub-case
	// here, since DielectricMedium's other two (specular) sub-cases never
	// reach this NEE block at all.
	const bool isPhase = (matType == MaterialType::DielectricMedium);

	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	// Real per-direction, per-channel f() for the 4 glossy (non-
	// EffectivelySmooth) BxDF-templated materials, via the same CPU_GPU
	// structs already verified on CPU (#222) and the recursive backend
	// (#229) - reconstructs the local shading frame from `normal` alone
	// (deterministic, matches every per-material frame-construction formula
	// in evaluate_materials()'s own switch exactly) and `phaseWo` as wi_world
	// (see that parameter's own comment). Returns false - meaning "no
	// contribution, not an error" - for a matType this function doesn't
	// handle, or when the queried direction falls in a hemisphere the
	// material can't reach from wi (grazing/back-facing wi, or wo below the
	// coat/dielectric's own reflection hemisphere - see RoughDielectric's
	// wo_z<=0 check and this function's own header comment on why
	// transmission-side NEE is out of scope here).
	//
	// Also returns outPdf: the BSDF pdf AT THIS QUERIED DIRECTION (not the
	// separately-tracked brdf_pdf_override, which is the pdf at the BSDF-
	// sampled CONTINUATION direction computed once in evaluate_materials() -
	// a different quantity). MIS needs the pdf at the direction actually
	// being weighted; mirrors optix_device_helpers.h's NEE blocks, which
	// compute a fresh {c,rm,rd,cd,cc}_bxdf.pdf(wi,...,ll/sk...) per light/sky
	// sample rather than reusing the continuation-direction pdf. For
	// Conductor/RoughMetal this is the BxDF's own real pdf() (a thin wrapper
	// over ggx_vndf_reflection_pdf); RoughDielectric uses its own real
	// pdf(); CoatedDiffuse/CoatedConductor have no closed-form pdf for their
	// unbounded-depth random walk, so - matching optix_device_helpers.h's
	// documented choice exactly - they reuse the coat's top-surface GGX
	// VNDF pdf as a cheap shape-matched proxy (any valid pdf keeps MIS
	// unbiased; this only affects variance, not correctness).
	// Shared per-shading-point setup for evalGlossyF below, computed ONCE
	// instead of on each of the (up to) 3 calls it gets per shading point
	// (area-light NEE, sky NEE, punctual-light NEE) - normal/phaseWo/
	// matType/matIdx are all invariant across those calls within a single
	// wf_finish_material_scatter() invocation, so re-deriving the local
	// frame, wi, and material lookup on every call redid identical work up
	// to 3x for no reason. glossy_valid folds both of evalGlossyF's old
	// early-outs (wrong matType, wi_z<=0) into one check the lambda itself
	// no longer needs to redo.
	bool glossy_isType = (matType == MaterialType::Conductor || matType == MaterialType::RoughDielectric ||
		matType == MaterialType::CoatedDiffuse || matType == MaterialType::CoatedConductor ||
		matType == MaterialType::RoughMetal);
	float3 glossy_tan = make_float3(0.0f,0.0f,0.0f), glossy_bit = make_float3(0.0f,0.0f,0.0f);
	float glossy_wi_x = 0.0f, glossy_wi_y = 0.0f, glossy_wi_z = 0.0f, glossy_alpha = 0.0f;
	bool glossy_valid = false;
	if (glossy_isType) {
		float3 up = (fabsf(normal.x) > 0.9f) ? make_float3(0.0f,1.0f,0.0f) : make_float3(1.0f,0.0f,0.0f);
		glossy_tan = normalize(cross(up, normal));
		glossy_bit = cross(normal, glossy_tan);
		glossy_wi_x = dot(phaseWo, glossy_tan);
		glossy_wi_y = dot(phaseWo, glossy_bit);
		glossy_wi_z = dot(phaseWo, normal);
		if (glossy_wi_z > 0.0f) {
			glossy_valid = true;
			glossy_alpha = sqrtf(materials[matIdx].fuzz);
		}
	}

	auto evalGlossyF = [&](const float3& queryDir, float3& outF, float& outPdf) -> bool {
		if (!glossy_valid) return false;
		float wo_x = dot(queryDir, glossy_tan), wo_y = dot(queryDir, glossy_bit), wo_z = dot(queryDir, normal);
		// RoughDielectric reaches both hemispheres (wo_z<0 = transmission,
		// "seen through the glass" - RoughDielectricBxDF::f()/pdf() already
		// handle either sign, matching optix_device_helpers.h's identical
		// `llz != 0.0f` gate); every other glossy type here is reflection-
		// only, unchanged.
		if (matType == MaterialType::RoughDielectric) {
			if (wo_z == 0.0f) return false;
		} else if (wo_z <= 0.0f) {
			return false;
		}
		const MaterialData& fm = materials[matIdx];
		float fr = 0.0f, fg = 0.0f, fb = 0.0f;
		if (matType == MaterialType::Conductor) {
			ConductorBxDF<float> bx{ fm.eta_c.x, fm.eta_c.y, fm.eta_c.z, fm.k_c.x, fm.k_c.y, fm.k_c.z, glossy_alpha, glossy_alpha };
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, fr, fg, fb);
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
		} else if (matType == MaterialType::RoughMetal) {
			RoughMetalBxDF<float> bx{ fm.albedo.x, fm.albedo.y, fm.albedo.z, glossy_alpha, glossy_alpha };
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, fr, fg, fb);
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
		} else if (matType == MaterialType::RoughDielectric) {
			RoughDielectricBxDF<float> bx{ fm.ior, glossy_alpha, glossy_alpha };
			float v = bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, nfEta, wo_x, wo_y, wo_z);
			fr = fg = fb = v;
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, nfEta, wo_x, wo_y, wo_z);
		} else if (matType == MaterialType::CoatedDiffuse) {
			CoatedDiffuseBxDF<float> bx{ fm.albedo.x, fm.albedo.y, fm.albedo.z, fm.ior, glossy_alpha, glossy_alpha };
			uint64_t s0, s1; wf_random_seed64_pair(seed, s0, s1);
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, s0, s1, fr, fg, fb);
			outPdf = ggx_vndf_reflection_pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, glossy_alpha, glossy_alpha);
		} else {
			CoatedConductorBxDF<float> bx{ fm.eta_c.x, fm.eta_c.y, fm.eta_c.z, fm.k_c.x, fm.k_c.y, fm.k_c.z, fm.ior, glossy_alpha, glossy_alpha };
			uint64_t s0, s1; wf_random_seed64_pair(seed, s0, s1);
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, s0, s1, fr, fg, fb);
			outPdf = ggx_vndf_reflection_pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, glossy_alpha, glossy_alpha);
		}
		outF = make_float3(fr, fg, fb);
		return true;
	};

	// Uplift an unbounded-positive RGB f() value (can exceed 1, unlike a
	// plain reflectance) to spectral - same technique as evaluate_materials()'s
	// own unboundedSpectrum lambda, deliberately WITHOUT the D65 illuminant
	// factor liftEmission below applies (that factor belongs to light sources,
	// not BRDF values).
	auto liftUnboundedRGB = [&](float3 rgb) -> SS {
		float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(rgb.x / sc, rgb.y / sc, rgb.z / sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]);
		return s;
	};

	// -------------------------------------------------------------------------
	// NEE: direct-light shadow ray (non-specular materials only)
	// -------------------------------------------------------------------------
	if (!is_specular) {
	if (numLights > 0 && aliasTable) {
		// Pick a light via alias table
		int   slot = int(wf_rand(seed) * float(numLights));
		if (slot >= (int)numLights) slot = (int)numLights - 1;
		const GpuAliasEntry& entry = aliasTable[slot];
		int light_idx = (wf_rand(seed) < entry.q) ? slot : entry.alias;
		float selection_pdf = aliasTable[light_idx].pdf;

		int            prim_idx  = lightIndices[light_idx];
		GpuLightKind   kind      = lightKinds[light_idx];

		float  geom_pdf = 0.0f, max_dist = 0.0f;
		float3 to_light;
		SS light_emission_spec(0.f);

		// A light's RGB colour is an ILLUMINANT (pbrt-v4 RGBIlluminantSpectrum:
		// scale * rsp(lambda) * D65(lambda)), not a bare RGBUnboundedSpectrum
		// (scale * rsp(lambda)) -- without the D65 factor, a grey light
		// uplifts to a flat/equal-energy spectrum (chromaticity (0.333,
		// 0.333)) instead of D65-white (0.3127,0.3290), which then
		// reconstructs as a non-neutral RGB through wf_xyz_to_linear_rgb's
		// D65-targeted matrix (R inflated ~20%, G/B suppressed ~5-11%) -- see
		// dev_sample_d65()'s own comment in spectral_device.h for the full
		// derivation.
		auto liftEmission = [&](float3 le) -> SS {
			float m = le.x > le.y ? (le.x > le.z ? le.x : le.z)
								  : (le.y > le.z ? le.y : le.z);
			float sc = 2.f * m;
			if (sc <= 0.f) return SS(0.f);
			float c0, c1, c2;
			dev_srgb_to_coeffs(le.x/sc, le.y/sc, le.z/sc, c0, c1, c2);
			RGBSigmoidPolynomial poly(c0, c1, c2);
			SS s(0.f);
			for (int i = 0; i < kWFNWavelengths; ++i)
				s[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
			return s;
		};

		if (kind == GpuLightKind::Sphere) {
			const SphereData& s = spheres[prim_idx];
			to_light = wf_sample_sphere_light(s, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[s.materialIdx].emission);
		} else if (kind == GpuLightKind::Triangle) {
			const TriangleData& tri = triangles[prim_idx];
			to_light = wf_sample_triangle_light(tri, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[tri.materialIdx].emission);
		} else if (kind == GpuLightKind::BilinearPatch) {
			const BilinearPatchData& bp = bilinearPatches[prim_idx];
			to_light = wf_sample_bilinear_patch_light(bp, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[bp.materialIdx].emission);
		} else {
			const QuadData& q = quads[prim_idx];
			to_light = wf_sample_quad_light(q, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[q.materialIdx].emission);
		}

		float light_pdf = selection_pdf * geom_pdf;
		// RoughDielectric NEE can reach lights on EITHER side of the
		// interface (reflection when raw_cos>0, transmission/"seen through
		// the glass" when raw_cos<0) - see evalGlossyF's own comment and
		// optix_device_helpers.h's identical two-sided RoughDielectric NEE,
		// which this now matches. Every other material stays reflection-
		// only (raw_cos>0 required), same as before.
		float raw_cos = dot(to_light, normal);
		if (light_pdf > 1e-6f && (isPhase || matType == MaterialType::RoughDielectric || raw_cos > 0.0f)) {
			// A phase function has no hemisphere/cosine restriction (it's
			// normalized over the full sphere, unlike a surface BRDF) - the
			// `cos_l` slot is set to 1 so the shared `bsdf_val * cos_l`
			// formula below reduces to the phase value alone. RoughDielectric
			// uses the absolute cosine (matches optix_device_helpers.h's
			// fabsf(llz)): the sign only selects reflection vs. transmission,
			// already handled inside evalGlossyF/RoughDielectricBxDF, not a
			// zero-below-the-hemisphere cutoff like the other glossy types.
			float cos_l = isPhase ? 1.0f
				: (matType == MaterialType::RoughDielectric) ? fabsf(raw_cos)
				: fmaxf(raw_cos, 0.0f);
			float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
			// Lambertian's BRDF is albedo/pi, direction-independent - `attenuation`
			// already equals albedoSpectrum(mat.albedo) from the caller, so reuse
			// it here instead of leaving the NEE contribution achromatic.
			// NormalizedFresnel's bsdf_val below is already a complete,
			// achromatic BRDF value (no color/texture involved), so it needs
			// no equivalent multiply. Same reuse-attenuation-directly reasoning
			// applies to the phase-scatter case: `attenuation` there already
			// equals the medium's single-scatter albedo (mat.albedo).
			SS bsdf_color(1.f);
			// Set inside the glossy else-branch below (evalGlossyF's outPdf,
			// the BSDF pdf at to_light) - 0 for every non-glossy material,
			// declared here (not nested inside that branch) so it's visible
			// where brdf_pdf_l is computed just below the if/else chain.
			float glossyPdf = 0.0f;
			if (matType == MaterialType::Lambertian) {
				bsdf_color = attenuation;
			} else if (matType == MaterialType::NormalizedFresnel) {
				float inv_eta = 1.0f / nfEta;
				float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
				if (nf_c <= 0.0f) nf_c = 1e-6f;
				float fr_l = FrDielectric(cos_l, nfEta);
				bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
			} else if (isPhase) {
				bsdf_val = wf_hg_phase_value(dot(phaseWo, to_light), phaseG);
				bsdf_color = attenuation;
			} else {
				// Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor
				// (glossy) - see evalGlossyF's own comment. bsdf_val=1 folds
				// the whole per-channel f() into bsdf_color instead of
				// splitting it into a scalar shape * color like the
				// Lambertian/NormalizedFresnel cases above (those have an
				// achromatic BRDF shape; this one doesn't). evalGlossyF
				// itself returns false both for "not one of my 4 types" (in
				// which case leave bsdf_val/bsdf_color at their Lambertian-
				// shaped defaults, unchanged from before this else - matches
				// e.g. NormalMappedLambertian's existing behavior, which
				// reaches here as matType==NormalMappedLambertian, not
				// Lambertian, and has relied on that same fallback since
				// before this NEE extension existed) and "wrong hemisphere
				// for one of my 4 types" (this light sits behind the glass'
				// reflection side) - only the second case should zero the
				// contribution, so explicitly re-check matType here rather
				// than trusting evalGlossyF's return value alone.
				float3 fRgb;
				if (evalGlossyF(to_light, fRgb, glossyPdf)) {
					bsdf_val = 1.0f;
					bsdf_color = liftUnboundedRGB(fRgb);
				} else if (glossy_isType) {
					bsdf_val = 0.0f;
				}
			}
			// glossyPdf (the BSDF pdf evaluated AT to_light, from evalGlossyF
			// above) takes priority for the 5 glossy types - brdf_pdf_override
			// is the pdf at the unrelated BSDF-sampled continuation direction
			// and must not be reused here (see evalGlossyF's own header
			// comment). glossyPdf is 0 (falls through to the non-glossy
			// branches) for every other material type, matching prior
			// behavior exactly.
			float brdf_pdf_l = (glossyPdf > 0.0f) ? glossyPdf
				: (brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(light_pdf, brdf_pdf_l);

			// Spectral direct-light contribution
			SS Ld = (mis_w * bsdf_val * cos_l / light_pdf) * throughput * bsdf_color * light_emission_spec;

			// 0.01, not the original 0.001: a scene with many densely-packed
			// custom primitives (spheres) reproducibly crashed the shadow
			// OptiX launch with an illegal memory access - the 0.001 offset
			// left the shadow ray's origin too close to its own emitting
			// surface (and, at high primitive density, to a neighboring
			// primitive's surface too) for this driver's any-hit traversal
			// over custom AABBs to handle; confirmed via bisection on
			// synthetic sphere-field pbrt scenes (compute-sanitizer memcheck/
			// initcheck found nothing, ruling out a plain buffer overrun).
			// Offsetting along BOTH the shading normal AND the shadow ray's
			// own direction, not just the normal alone. The normal-only
			// offset is load-bearing on its own (see above): a pure
			// direction-only offset - matching optix_device_helpers.h's
			// trace_shadow_ray() - was tried first and broke
			// NormalMappedLambertian (scene 20 rendered 99% black), since a
			// shading normal bent away from the true surface no longer
			// guarantees "away from this primitive" the way the true
			// geometric normal does, letting the ray re-enter its own
			// (unperturbed) sphere at a grazing angle. But normal-only,
			// even at a much larger shadow_eps, turned out far weaker than
			// direction-only at escaping self-intersection on dense,
			// closely-packed real geometry: Sibenik Cathedral's stone
			// tracery false-occluded most sky-NEE shadow rays at
			// shadow_eps=0.01, and bumping shadow_eps up to 2.0 with a
			// normal-only offset barely helped (GPU stayed under 50% of
			// CPU's brightness at matched settings) - adding the direction
			// component back in (while keeping the normal component, so
			// NormalMappedLambertian stays fixed) closed the gap to ~73%
			// at shadow_eps=0.5, confirmed by direct experiment. 2.0 with
			// the combined offset overshot badly (GPU 1.5x *brighter* than
			// CPU - real light leaks from skipping past legitimate
			// occluders), which is why this is a per-scene-tunable value
			// (GpuCameraParams::shadowRayEpsilon), not a blanket increase.
			// isPhase (medium-interior scatter point): skip the normal-based
			// offset entirely - `normal` here is the sphere's own entry-surface
			// normal, not a meaningful direction at this interior point, and
			// direction-only is exactly what optix_device_helpers.h's own
			// trace_shadow_ray() already uses for this same phase-scatter NEE
			// on the recursive backend (see that call site's comment).
			// The normal-offset nudges toward whichever side to_light is
			// actually on (copysignf(shadow_eps, raw_cos), not always
			// +shadow_eps): for every material except RoughDielectric's new
			// transmission-side case raw_cos>0 always holds here (the gate
			// above still requires it), so this is exactly +shadow_eps as
			// before - only RoughDielectric's transmission side (raw_cos<0)
			// newly nudges the origin to the far side of the surface instead
			// of back into the same hemisphere the ray isn't going toward.
			ShadowRayWorkItem shadow;
			shadow.origin    = hit_point + (isPhase ? make_float3(0.0f, 0.0f, 0.0f) : copysignf(shadow_eps, raw_cos) * normal)
				+ shadow_eps * normalize(to_light);
			shadow.direction = to_light;
			shadow.tMax      = max_dist - 0.002f;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				shadow.Ld[i]             = Ld[i];
				shadow.wavelengths[i]    = swl.lambda[i];
				shadow.wavelength_pdfs[i] = swl.pdf[i];
			}
			shadow.pixelIndex = pixelIndex;
			shadowQueue.push(shadow);
		}
	}
	// -------------------------------------------------------------------------
	// NEE: sky (infinite) light. Mirrors CPU's camera.h Strategy A-2 and the
	// recursive backend's own sky-NEE block (optix_device_helpers.h) - see
	// that comment for the full story (GPU used to only pick up sky light via
	// a lucky BSDF-sampled escape, no NEE, which under-lit small-aperture
	// interiors like Sibenik Cathedral). skyColor is the same constant color
	// the miss path already uses (camera.backgroundColor - see
	// optix_miss.h/accumulate_miss's own use of it), defaulting to black for
	// every scene without a sky, which makes this a free no-op there.
	// -------------------------------------------------------------------------
	if (!is_specular && (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f)) {
		// wf_sample_sky_nee() (wavefront_sky_light.h) dispatches to real HDR
		// importance sampling when the scene's infinite light carries an
		// image (skyDist.height > 0), falling back to this exact uniform-
		// sphere sample + flat skyColor otherwise - see optix_device_
		// helpers.h's own Lambertian sky-NEE block for the recursive
		// backend's identical dispatch, and optix_sky_light.h for the full
		// algorithm comment.
		float3 sky_dir, sky_Le_val; float pdf_sky;
		wf_sample_sky_nee(skyDist, seed, skyColor, sky_dir, pdf_sky, sky_Le_val);
		// See the area-light block above for the RoughDielectric two-sided
		// rationale (matches optix_device_helpers.h's sky-NEE block).
		float  raw_cos = dot(sky_dir, normal);
		float  cos_l   = isPhase ? 1.0f
			: (matType == MaterialType::RoughDielectric) ? fabsf(raw_cos)
			: raw_cos;
		if (isPhase || matType == MaterialType::RoughDielectric || cos_l > 0.0f) {
			float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
			// See the area-light block above: attenuation == albedoSpectrum(mat.albedo)
			// for Lambertian (direction-independent BRDF, safe to reuse here);
			// NormalizedFresnel's bsdf_val is already a complete achromatic value.
			SS bsdf_color(1.f);
			// See the area-light block above for why this is declared here
			// rather than nested inside the glossy else-branch.
			float glossyPdf = 0.0f;
			if (matType == MaterialType::Lambertian) {
				bsdf_color = attenuation;
			} else if (matType == MaterialType::NormalizedFresnel) {
				float inv_eta = 1.0f / nfEta;
				float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
				if (nf_c <= 0.0f) nf_c = 1e-6f;
				float fr_l = FrDielectric(cos_l, nfEta);
				bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
			} else if (isPhase) {
				bsdf_val = wf_hg_phase_value(dot(phaseWo, sky_dir), phaseG);
				bsdf_color = attenuation;
			} else {
				// See the area-light block's identical else-branch for the
				// full rationale (evalGlossyF/glossy_isType split).
				float3 fRgb;
				if (evalGlossyF(sky_dir, fRgb, glossyPdf)) {
					bsdf_val = 1.0f;
					bsdf_color = liftUnboundedRGB(fRgb);
				} else if (glossy_isType) {
					bsdf_val = 0.0f;
				}
			}
			// See the area-light block above: glossyPdf (pdf at sky_dir) takes
			// priority over brdf_pdf_override (pdf at the unrelated
			// continuation direction) for the 5 glossy types.
			float brdf_pdf_sky = (glossyPdf > 0.0f) ? glossyPdf
				: (brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(pdf_sky, brdf_pdf_sky);

			// Uplift RGB sky_Le_val (the real per-direction radiance for an
			// image sky, or the flat skyColor otherwise - see
			// wf_sample_sky_nee()) to spectrum (same pattern as liftEmission
			// above, including the D65 illuminant factor -- see that
			// lambda's own comment).
			float m = sky_Le_val.x > sky_Le_val.y ? (sky_Le_val.x > sky_Le_val.z ? sky_Le_val.x : sky_Le_val.z)
			                                       : (sky_Le_val.y > sky_Le_val.z ? sky_Le_val.y : sky_Le_val.z);
			float sc = 2.f * m;
			SS sky_spec(0.f);
			if (sc > 0.f) {
				float c0, c1, c2;
				dev_srgb_to_coeffs(sky_Le_val.x / sc, sky_Le_val.y / sc, sky_Le_val.z / sc, c0, c1, c2);
				RGBSigmoidPolynomial poly(c0, c1, c2);
				for (int i = 0; i < kWFNWavelengths; ++i)
					sky_spec[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
			}

			SS Ld = (mis_w * bsdf_val * cos_l / pdf_sky) * throughput * bsdf_color * sky_spec;

			// See the area-light block above for why this is a combined
			// normal+direction offset (not the normal alone), why isPhase
			// skips the normal term entirely, and why the normal term uses
			// copysignf(shadow_eps, raw_cos) rather than always +shadow_eps.
			ShadowRayWorkItem shadow;
			shadow.origin    = hit_point + (isPhase ? make_float3(0.0f, 0.0f, 0.0f) : copysignf(shadow_eps, raw_cos) * normal)
				+ shadow_eps * normalize(sky_dir);
			shadow.direction = sky_dir;
			shadow.tMax      = 1e30f;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				shadow.Ld[i]              = Ld[i];
				shadow.wavelengths[i]     = swl.lambda[i];
				shadow.wavelength_pdfs[i] = swl.pdf[i];
			}
			shadow.pixelIndex = pixelIndex;
			shadowQueue.push(shadow);
		}
	}

	// -------------------------------------------------------------------------
	// NEE: punctual (point/spot/distant) delta lights. Unlike the area-light
	// block above (one stochastic pick via alias table), every contributing
	// punctual light gets its own shadow ray - matches recursive path
	// (optix_device_helpers.h add_punctual_lights_lambertian) and the CPU
	// reference (camera.h punct_lights loop): pdf=1 by construction, so no
	// MIS weight or pdf division, just beta * BRDF * cos_theta * Li per light.
	// -------------------------------------------------------------------------
	for (unsigned int pli = 0; pli < numPunctualLights; ++pli) {
		float3 wi, Li; float t_max;
		if (!wf_eval_punctual_light(punctualLights[pli], hit_point, wi, Li, t_max)) continue;
		// See the area-light block above for the RoughDielectric two-sided
		// rationale (matches optix_device_helpers.h's punctual-light NEE,
		// which only excludes exact grazing (plz==0), not either sign).
		float raw_cos = dot(wi, normal);
		if (matType != MaterialType::RoughDielectric && raw_cos <= 0.0f) continue;
		if (matType == MaterialType::RoughDielectric && raw_cos == 0.0f) continue;
		float cos_l = (matType == MaterialType::RoughDielectric) ? fabsf(raw_cos) : raw_cos;

		float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
		// See the area-light block above: attenuation == albedoSpectrum(mat.albedo)
		// for Lambertian (direction-independent BRDF, safe to reuse here);
		// NormalizedFresnel's bsdf_val is already a complete achromatic value.
		SS bsdf_color(1.f);
		if (matType == MaterialType::Lambertian) {
			bsdf_color = attenuation;
		} else if (matType == MaterialType::NormalizedFresnel) {
			float inv_eta = 1.0f / nfEta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;
			float fr_l = FrDielectric(cos_l, nfEta);
			bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
		} else {
			// See the area-light block's identical else-branch for the full
			// rationale (evalGlossyF/glossy_isType split). No MIS weight for
			// punctual (delta) lights, same as every other material here -
			// the pdf-at-wi output isn't needed here, unlike the area/sky
			// NEE blocks above.
			float3 fRgb; float unusedPdf;
			if (evalGlossyF(wi, fRgb, unusedPdf)) {
				bsdf_val = 1.0f;
				bsdf_color = liftUnboundedRGB(fRgb);
			} else if (glossy_isType) {
				bsdf_val = 0.0f;
			}
		}

		// Uplift RGB Li to spectrum (same pattern as liftEmission above,
		// including the D65 illuminant factor -- see that lambda's own
		// comment)
		float m = Li.x > Li.y ? (Li.x > Li.z ? Li.x : Li.z) : (Li.y > Li.z ? Li.y : Li.z);
		float sc = 2.f * m;
		SS Li_spec(0.f);
		if (sc > 0.f) {
			float c0, c1, c2;
			dev_srgb_to_coeffs(Li.x / sc, Li.y / sc, Li.z / sc, c0, c1, c2);
			RGBSigmoidPolynomial poly(c0, c1, c2);
			for (int i = 0; i < kWFNWavelengths; ++i)
				Li_spec[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
		}

		SS Ld = (bsdf_val * cos_l) * throughput * bsdf_color * Li_spec;

		// See the area-light block above for why this is a combined
		// normal+direction offset, not the normal alone, and why the normal
		// term uses copysignf(shadow_eps, raw_cos) rather than always
		// +shadow_eps (RoughDielectric's transmission side).
		ShadowRayWorkItem shadow;
		shadow.origin    = hit_point + copysignf(shadow_eps, raw_cos) * normal + shadow_eps * normalize(wi);
		shadow.direction = wi;
		shadow.tMax      = t_max - 0.002f;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			shadow.Ld[i]              = Ld[i];
			shadow.wavelengths[i]     = swl.lambda[i];
			shadow.wavelength_pdfs[i] = swl.pdf[i];
		}
		shadow.pixelIndex = pixelIndex;
		shadowQueue.push(shadow);
	}

	// Flush accumulated prior-bounce radiance (once, regardless of whether
	// any area/punctual lights actually contributed above).
	if ((bool)radiance) addToFramebuffer(pixelIndex, radiance);
	} // if (!is_specular) - NEE (area + punctual lights)

	// -------------------------------------------------------------------------
	// Bounce: push next ray
	// -------------------------------------------------------------------------
	SS new_throughput = throughput * attenuation;

	// Russian roulette after depth 3
	if (depth >= 3) {
		float p = new_throughput.MaxComponentValue();
		if (wf_rand(seed) >= p) {
			if (is_specular) addToFramebuffer(pixelIndex, radiance);
			return;
		}
		new_throughput = new_throughput / p;
	}

	RayWorkItem next;
	next.origin     = hit_point + 0.001f * scattered_dir;
	next.direction  = normalize(scattered_dir);
	next.seed            = seed;
	next.pixelIndex      = pixelIndex;
	next.depth           = depth + 1;
	next.specular_bounce = is_specular ? 1 : 0;
	// BRDF PDF of this new direction, for MIS if this ray escapes the scene
	// on its next bounce (see RayWorkItem::brdf_pdf's own comment) - mirrors
	// optix_intersection_sphere.h's brdf_pdf_out exactly (0 for specular,
	// else brdf_pdf_override if the material set one, else the cosine-
	// weighted hemisphere pdf every non-specular case here samples from).
	next.brdf_pdf = is_specular ? 0.0f
		: (brdf_pdf_override > 0.0f ? brdf_pdf_override
			: fmaxf(dot(next.direction, normal), 0.0f) / 3.14159265f);
	next.tMin       = 0.001f;
	next.tMax       = 1e30f;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		next.throughput[i]      = new_throughput[i];
		next.radiance[i]        = is_specular ? radiance[i] : 0.0f;
		next.wavelengths[i]     = swl.lambda[i];
		next.wavelength_pdfs[i] = swl.pdf[i];
	}
	nextRayQueue.push(next);
}

extern "C" __global__ void evaluate_materials(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	// BSSRDF probe-request queue (MaterialType::Subsurface, Phase 2) - see
	// this file's own Subsurface case and BssrdfProbeWorkItem's comment
	// (wavefront_types.h).
	WorkQueue<BssrdfProbeWorkItem> bssrdfProbeQueue,
	float3*                      framebuffer,
	// Scene data
	const SphereData*   spheres,
	const QuadData*     quads,
	const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches,
	const MaterialData* materials,
	const int*          lightIndices,
	const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	const TextureData*  textures,
	const unsigned char* texturePixels,
	int maxDepth,
	const CloudMedium<float>* cloudMediums,
	unsigned int numCloudMediums,
	const GpuRgbGridMedium* rgbGridMediums,
	const float* rgbGridData,
	// Real tabulated measured-BRDF tables (MaterialType::Measured) - see
	// optix_types.h's GpuMeasuredTable comment and wavefront_measured_bxdf.h.
	// Plain extra kernel parameters, same shape as cloudMediums/
	// rgbGridMediums above (this material needs no OptiX trace, so it does
	// not go through the wf_params/lp raygen-launch-params path the BSSRDF
	// probe walk uses).
	const GpuMeasuredTable* measuredTables,
	unsigned int numMeasuredTables,
	const float* measuredParamValues,
	const float* measuredData,
	const float* measuredMcdf,
	const float* measuredCcdf,
	float3 skyColor,
	float shadowRayEpsilon,
	// Real importance-sampled HDR sky (LightSource "infinite" with an image) -
	// see optix_types.h's GpuSkyDistribution comment. Passed by value, same
	// established pattern as GpuCameraParams itself already crossing this
	// exact kernel-launch boundary (generate_camera_rays). height<=0 (the
	// default when the scene has no image sky) makes every sky-NEE/miss call
	// site below fall back to skyColor + uniform-sphere sampling, unchanged.
	GpuSkyDistribution skyDist
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;
	// <= 0 (zero-init default) means "use the standard 0.01f" - mirrors
	// optix_device_helpers.h's trace_shadow_ray() exactly, see
	// GpuCameraParams::shadowRayEpsilon's own comment.
	const float shadow_eps = (shadowRayEpsilon > 0.0f) ? shadowRayEpsilon : 0.01f;

	// Reconstruct spectral path state from work item
	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SS throughput(h.throughput);
	SS radiance(h.radiance);
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = h.wavelengths[i];
		swl.pdf[i]    = h.wavelength_pdfs[i];
	}

	// Helper: convert SampledSpectrum to sRGB and atomicAdd to framebuffer
	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	// -------------------------------------------------------------------------
	// Emissive: add emission term, path terminates (no scatter).
	// pbrt-v4 alignment: only add emissive if depth==0 or specular_bounce==1
	// to avoid double-counting with NEE shadow rays at non-specular bounces.
	// -------------------------------------------------------------------------
	if (mat.type == MaterialType::DiffuseLight) {
		if (h.specular_bounce || h.depth == 0) {
			float cos_face = dot(-h.rayDir, normal);
			if (cos_face > 0.0f) {
				// Uplift RGB emission to spectrum via device sRGB table
				// (deferred: emissionSpectrum helper is declared below; use inline here)
				// A real map_Ke texture (Gallery's painted-canvas glow) is
				// sampled at the hit UV instead of the flat mat.emission -
				// see add_diffuse_light()'s textureIdx comment and the
				// recursive backend's identical optix_intersection_triangle.h
				// change.
				float3 le = (mat.textureIdx >= 0)
					? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point)
					: mat.emission;
				float m_le = le.x > le.y ? (le.x > le.z ? le.x : le.z)
										  : (le.y > le.z ? le.y : le.z);
				float sc = 2.f * m_le;
				if (sc > 0.f) {
					float c0, c1, c2;
					dev_srgb_to_coeffs(le.x/sc, le.y/sc, le.z/sc, c0, c1, c2);
					RGBSigmoidPolynomial emitPoly(c0, c1, c2);
					SS emitS(0.f);
					// D65 illuminant factor -- see liftEmission's own comment
					// above (this is the same "light RGB is an illuminant,
					// not a bare unbounded spectrum" uplift, just for a
					// directly-hit emissive surface instead of an NEE sample).
					for (int i = 0; i < kWFNWavelengths; ++i)
						emitS[i] = sc * emitPoly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
					radiance = radiance + throughput * emitS;
				}
			}
		}
		addToFramebuffer(h.pixelIndex, radiance);
		return; // path ends at emissive surface
	}

	if (h.depth >= maxDepth) {
		// Max depth reached — just accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	// -------------------------------------------------------------------------
	// Scatter — evaluate BSDF (mirrors optix_programs.cu closesthit logic)
	// -------------------------------------------------------------------------
	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);   // spectral BSDF * cos / pdf
	bool   scattered        = false;
	bool   is_specular      = false;
	float  brdf_pdf_override = -1.0f;
	// Only set by MaterialType::DielectricMedium's medium-interior
	// phase-scatter sub-case (see that case below) - passed through to
	// wf_finish_material_scatter's NEE block for real phase-function NEE+MIS.
	// Unused (harmless zero) for every other material type.
	float3 phaseWo = make_float3(0.0f, 0.0f, 0.0f);
	// wf_finish_material_scatter's `nfEta` slot - mat.ior by default
	// (correct for MaterialType::NormalizedFresnel); MaterialType::
	// RoughDielectric's glossy branch overrides this to its own precomputed
	// eta (front_face ? 1/ior : ior), which is NOT the same value for a
	// back-face hit - see that case's own comment.
	float matEta = mat.ior;
	float  phaseG  = 0.0f;

	// Helper: uplift RGB albedo to SampledSpectrum via device sigmoid polynomial
	auto albedoSpectrum = [&](float3 rgb) -> SS {
		float c0, c1, c2;
		float r = rgb.x < 0.f ? 0.f : (rgb.x > 1.f ? 1.f : rgb.x);
		float g = rgb.y < 0.f ? 0.f : (rgb.y > 1.f ? 1.f : rgb.y);
		float b = rgb.z < 0.f ? 0.f : (rgb.z > 1.f ? 1.f : rgb.z);
		dev_srgb_to_coeffs(r, g, b, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = poly(swl.lambda[i]);
		return s;
	};
	auto emissionSpectrum = [&](float3 /*rgb*/) -> SS { return SS(0.f); };  // placeholder, liftEmission used inline
	(void)emissionSpectrum;

	// Uplift an unbounded-positive RGB weight (can exceed 1, unlike a plain
	// reflectance) to spectral: same normalize-by-2*max/uplift/rescale
	// technique as the NEE block's liftEmission lambda below - needed for
	// MaterialType::Hair, whose HairBxDF::sample() result already divides by
	// the sample pdf (a proper importance-sampling weight, not a [0,1]
	// albedo), so naively clamping it into albedoSpectrum's range would bias
	// bright/peaky fiber lobes toward black.
	auto unboundedSpectrum = [&](float3 rgb) -> SS {
		// Clamp negative components before normalizing - dev_srgb_to_coeffs
		// expects each channel in [0,1] once divided by sc; a slightly
		// negative component (numerically possible at extreme absorption or
		// near-grazing angles in HairBxDF's Marschner model, the only caller
		// of this lambda) would otherwise reach it unclamped, unlike
		// albedoSpectrum above which always clamps first.
		float rx = rgb.x < 0.f ? 0.f : rgb.x;
		float ry = rgb.y < 0.f ? 0.f : rgb.y;
		float rz = rgb.z < 0.f ? 0.f : rgb.z;
		float m = rx > ry ? (rx > rz ? rx : rz) : (ry > rz ? ry : rz);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(rx/sc, ry/sc, rz/sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]);
		return s;
	};

	switch (mat.type) {
	case MaterialType::Lambertian: {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		const float3 lambertianColor = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point)
			: mat.albedo;
		attenuation = albedoSpectrum(lambertianColor);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Subsurface: {
		// Real tabulated BSSRDF (wavefront backend, Phase 2 - Phase 1
		// shipped this for the recursive backend only, see optix_device_
		// helpers.h's shade_material() Subsurface case, which this mirrors).
		//
		// Entry interface: the exact same smooth dielectric sample as
		// MaterialType::Dielectric, via the already-shared wf_dielectric_
		// scatter() helper (also used by DielectricMedium's own entry
		// surface above) - matches pbrt-v4's SubsurfaceMaterial::GetBxDF /
		// this codebase's CPU `class subsurface` (material_pbrt.h), and the
		// recursive backend's identical dielectric_scatter() call.
		//
		// front_face is hardcoded true: by this whole design's construction
		// (mirroring pbrt-v4's own SubsurfaceMaterial and the recursive
		// backend's Phase 1), a Subsurface material's entry interface is
		// only ever reached from a genuine outside hit - the probe walk's
		// own exit point resumes as a MaterialType::NormalizedFresnel bounce
		// instead of a second Subsurface entry (see resolve_bssrdf_exit()
		// below), so this switch case is never reached "from inside".
		// HitWorkItem::frontFace itself isn't usable here regardless - it's
		// only populated for sphere hits (__closesthit__wf_sphere), and the
		// SSS dragon scenes this targets are triangle meshes.
		float3 sdir = wf_dielectric_scatter(h.rayDir, normal, /*front_face=*/true, mat.ior, seed);
		bool is_transmission = dot(sdir, normal) < 0.0f;
		if (!is_transmission) {
			// Specular reflection off the entry interface - identical to
			// Dielectric's own reflection case; falls through to the shared
			// tail below as an ordinary specular bounce (no NEE, no probe
			// walk needed).
			scattered_dir = sdir;
			attenuation   = SS(1.f);
			scattered     = true;
			is_specular   = true;
			break;
		}

		// Transmission: hand off to the dedicated BSSRDF probe-walk stage
		// (wavefront_probe.h's __raygen__wf_probe + this file's own
		// resolve_bssrdf_exit(), wired into WavefrontPathTracer::render()'s
		// per-bounce loop) instead of scattering inline - the probe walk
		// needs real optixTrace calls, unavailable from this plain CUDA
		// kernel. Carries everything needed to resume the path once (or if)
		// an exit point is found - see BssrdfProbeWorkItem's own comment.
		// This thread's own work ends here: nothing is pushed to
		// nextRayQueue/shadowQueue/framebuffer now, that happens later (this
		// bounce, before the shadow pass - see the per-bounce loop) via
		// resolve_bssrdf_exit() once the probe walk resolves.
		{
			BssrdfProbeWorkItem probeItem;
			probeItem.p0     = hit_point;
			probeItem.axis0  = normal;
			probeItem.matIdx = h.materialIdx;
			probeItem.seed   = seed;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				probeItem.throughput[i]      = throughput[i];
				probeItem.radiance[i]        = radiance[i];
				probeItem.wavelengths[i]     = swl.lambda[i];
				probeItem.wavelength_pdfs[i] = swl.pdf[i];
			}
			probeItem.pixelIndex = h.pixelIndex;
			probeItem.depth      = h.depth;
			bssrdfProbeQueue.push(probeItem);
		}
		return;
	}
	case MaterialType::Metal: {
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
		break;
	}
	case MaterialType::Measured: {
		// Real tabulated measured-BRDF - see wf_sample_measured_material's
		// comment above. Matches material_pbrt.h `class measured`'s
		// skip_pdf=true: no NEE/MIS (is_specular=true), and the result needs
		// unboundedSpectrum (not albedoSpectrum) since sample_f()'s returned
		// fr/fg/fb is already the material's full throughput weight, not a
		// [0,1] albedo - same convention as MaterialType::Hair/Principled
		// above (their BxDF::sample() results are likewise already-weighted,
		// potentially-unbounded values).
		float3 sdir, atten;
		if (wf_sample_measured_material(h.rayDir, normal, mat,
				measuredTables, numMeasuredTables,
				measuredParamValues, measuredData, measuredMcdf, measuredCcdf,
				seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	case MaterialType::Dielectric: {
		float eta = dot(h.rayDir, normal) < 0.0f ? (1.0f / mat.ior) : mat.ior;
		float3 unit_dir = normalize(h.rayDir);
		float  cos_t = fminf(dot(-unit_dir, normal), 1.0f);
		float  sin_t = sqrtf(1.0f - cos_t * cos_t);
		bool   cannot_refract = eta * sin_t > 1.0f;
		float  r0 = (1.0f - mat.ior) / (1.0f + mat.ior);
		r0 = r0 * r0;
		float schlick = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);
		bool is_transmission;
		if (cannot_refract || schlick > wf_rand(seed)) {
			scattered_dir = wf_reflect(unit_dir, normal);
			is_transmission = false;
		} else {
			scattered_dir = wf_refract(unit_dir, normal, eta);
			is_transmission = true;
		}
		// Tf tints transmission only, matching real colored glass - see
		// add_dielectric()'s transmission_filter comment (recursive
		// backend's optix_device_helpers.h identical case has the same
		// reasoning).
		attenuation = is_transmission ? albedoSpectrum(mat.transmission_filter) : SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::RoughDielectric: {
		// Ported from optix_device_helpers.h's recursive-path case (see its
		// own comments) - this one previously diverged in three ways that
		// together made every rough-dielectric surface render pure black:
		//  1. wi_z<=0 terminated the path instead of flipping wi into the
		//     upper hemisphere - a ray exiting the glass from inside
		//     legitimately lands with wi_z<0 in the local frame, and killing
		//     it there means light that entered the sphere almost never
		//     found its way back out.
		//  2. Fresnel used mat.ior directly with no front/back-face
		//     adjustment, instead of pbrt-v4's front_face ? 1/ior : ior
		//     convention (see FrDielectric's own eta contract).
		//  3. attenuation read mat.albedo, which optix_types.h's own
		//     MaterialData comment documents as unused/undefined for
		//     RoughDielectric (a union slot shared with Hair/Medium/
		//     DiffuseTransmission/Principled's own fields) - for a material
		//     built via add_rough_dielectric() (scene_builder.cpp), which
		//     never writes albedo, that slot is always zero, so every
		//     scatter event was multiplied by black regardless of what
		//     light actually reached the surface. The correct weight is the
		//     VNDF-sampling shadow-masking ratio G(wo,wi)/G1(wi) - the D,
		//     cos, and pdf terms cancel under VNDF importance sampling,
		//     matching pbrt-v4's RoughDielectricBxDF exactly.
		float rd_alpha = sqrtf(mat.fuzz);
		bool rd_front_face = h.frontFace != 0;
		float rd_ri = rd_front_face ? (1.0f / mat.ior) : mat.ior;
		float3 n = normal;
		float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 tan_v  = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		float3 wi_w = normalize(-h.rayDir);
		float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
		if (wi_z < 0.0f) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }
		TrowbridgeReitz<float> rd_dist(rd_alpha, rd_alpha);
		float wm_x, wm_y, wm_z;
		rd_dist.Sample_wm(wi_x, wi_y, wi_z, wf_rand(seed), wf_rand(seed), wm_x, wm_y, wm_z);
		float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float Fr = FrDielectric(rd_dot, 1.0f / rd_ri);
		float wo_x, wo_y, wo_z;
		if (wf_rand(seed) < Fr) {
			// Reflect
			wo_x = 2.0f*rd_dot*wm_x - wi_x;
			wo_y = 2.0f*rd_dot*wm_y - wi_y;
			wo_z = 2.0f*rd_dot*wm_z - wi_z;
			if (wo_z <= 0.0f) { scattered = false; break; }
			scattered_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		} else {
			// Refract
			float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
			float eta = rd_front_face ? (1.0f / mat.ior) : mat.ior;
			float3 refracted = wf_refract(normalize(h.rayDir), wm_world, eta);
			wo_x = dot(refracted, tan_v); wo_y = dot(refracted, bitan); wo_z = dot(refracted, n);
			scattered_dir = refracted;
		}
		{
			float wo_z_abs = fabsf(wo_z);
			float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z_abs);
			float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
			float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
			attenuation = SS(w);
		}
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) rough glass, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment - reflection-hemisphere-only, a documented scope
		// reduction vs. the recursive backend's two-sided glass NEE).
		// matEta is overridden to rd_ri (front_face ? 1/ior : ior) since
		// wf_finish_material_scatter's `nfEta` slot otherwise carries plain
		// mat.ior (correct for NormalizedFresnel, wrong for a back-face
		// RoughDielectric hit). phaseWo carries the UNFLIPPED wi_w (matches
		// evalGlossyF's own, equally unflipped, frame reconstruction -
		// the defensive wi_z<0 flip a few lines up is a rare-edge-case
		// safety net for the BSDF-sampled continuation only, not something
		// evalGlossyF needs to replicate: a genuinely non-front-facing wi
		// there just yields zero NEE contribution instead of the wrong
		// answer). pdf uses the REAL RoughDielectricBxDF::pdf() (not the
		// GGX-reflection-shape proxy Conductor/CoatedDiffuse/CoatedConductor
		// use below) since it already correctly spans both lobes - needed
		// because the BSDF-sampled continuation direction can legitimately
		// be a refraction even when NEE itself only covers reflection.
		if (!rd_dist.EffectivelySmooth()) {
			is_specular = false;
			matEta = rd_ri;
			RoughDielectricBxDF<float> rd_bxdf{ mat.ior, rd_alpha, rd_alpha };
			brdf_pdf_override = rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, wo_x, wo_y, wo_z);
			phaseWo = wi_w;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::Conductor: {
		float c_alpha = sqrtf(mat.fuzz);
		float3 cn = normal;
		float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 ctan   = normalize(cross(cup, cn));
		float3 cbitan = cross(cn, ctan);
		float3 cwi = -normalize(h.rayDir);
		float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
		if (cwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
		float cwm_x, cwm_y, cwm_z;
		c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, wf_rand(seed), wf_rand(seed), cwm_x, cwm_y, cwm_z);
		float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
		float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
		float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
		float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
		if (cwo_z <= 0.0f) { scattered = false; break; }
		float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
		float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
		float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
		float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
		// Use average Fresnel weight as scalar (conductor is specular, color from albedo)
		attenuation = albedoSpectrum(make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight));
		scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) conductors, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment). brdf_pdf_override uses the GGX-reflection VNDF pdf
		// (ggx_vndf_reflection_pdf, src/shared/bxdfs_conductor.h) at the
		// sampled direction - Conductor is reflection-only to begin with,
		// so this is the exact (not proxy) pdf, unlike CoatedDiffuse/
		// CoatedConductor below. phaseWo carries cwi (already `-ray_dir`,
		// matching evalGlossyF's `wi_world` convention).
		if (!c_dist.EffectivelySmooth()) {
			is_specular = false;
			brdf_pdf_override = ggx_vndf_reflection_pdf(cwi_x, cwi_y, cwi_z, cwo_x, cwo_y, cwo_z, c_alpha, c_alpha);
			phaseWo = cwi;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::RoughMetal: {
		// GGX VNDF + flat-tint reflectance, no complex Fresnel (pbrt-v4/RTOW
		// rough_metal) - same shape as MaterialType::Conductor above, minus
		// FrConductorRGB (RoughMetalBxDF has no real Fresnel model). NOT the
		// same material as MaterialType::Metal (fuzz-perturbed mirror, a
		// different model - CPU's plain `metal` class).
		float rm_alpha = sqrtf(mat.fuzz);
		float3 rmn = normal;
		float3 rmup = (fabsf(rmn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 rmtan   = normalize(cross(rmup, rmn));
		float3 rmbitan = cross(rmn, rmtan);
		float3 rmwi = -normalize(h.rayDir);
		float rmwi_x = dot(rmwi, rmtan), rmwi_y = dot(rmwi, rmbitan), rmwi_z = dot(rmwi, rmn);
		if (rmwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> rm_dist(rm_alpha, rm_alpha);
		float rmwm_x, rmwm_y, rmwm_z;
		rm_dist.Sample_wm(rmwi_x, rmwi_y, rmwi_z, wf_rand(seed), wf_rand(seed), rmwm_x, rmwm_y, rmwm_z);
		float rm_dot = rmwi_x*rmwm_x + rmwi_y*rmwm_y + rmwi_z*rmwm_z;
		float rmwo_x = 2.0f*rm_dot*rmwm_x - rmwi_x;
		float rmwo_y = 2.0f*rm_dot*rmwm_y - rmwi_y;
		float rmwo_z = 2.0f*rm_dot*rmwm_z - rmwi_z;
		if (rmwo_z <= 0.0f) { scattered = false; break; }
		float rm_G1_wi  = rm_dist.G1(rmwi_x, rmwi_y, rmwi_z);
		float rm_G_wowi = rm_dist.G(rmwo_x, rmwo_y, rmwo_z, rmwi_x, rmwi_y, rmwi_z);
		float rm_weight = (rm_G1_wi > 1e-8f) ? rm_G_wowi / rm_G1_wi : 0.0f;
		attenuation = albedoSpectrum(make_float3(mat.albedo.x * rm_weight, mat.albedo.y * rm_weight, mat.albedo.z * rm_weight));
		scattered_dir = normalize(rmwo_x*rmtan + rmwo_y*rmbitan + rmwo_z*rmn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) rough metal, via
		// wf_finish_material_scatter's shared evalGlossyF - see
		// MaterialType::Conductor's identical-shape block above.
		if (!rm_dist.EffectivelySmooth()) {
			is_specular = false;
			brdf_pdf_override = ggx_vndf_reflection_pdf(rmwi_x, rmwi_y, rmwi_z, rmwo_x, rmwo_y, rmwo_z, rm_alpha, rm_alpha);
			phaseWo = rmwi;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::CoatedDiffuse: {
		float cd_alpha = sqrtf(mat.fuzz);
		float3 cdn = normal;
		float3 cdup = (fabsf(cdn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 cdtan   = normalize(cross(cdup, cdn));
		float3 cdbitan = cross(cdn, cdtan);
		float3 cdwi = -normalize(h.rayDir);
		float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbitan), cdwi_z = dot(cdwi, cdn);
		// Grazing/back-facing incoming ray: no valid local frame to sample
		// against - matches optix_device_helpers.h's shade_material()
		// CoatedDiffuse case (`if (cdwi_z <= 0.0f) { scattered = false; }`),
		// missing here let a grazing-angle ray fall through to Sample_wm()
		// with a degenerate/negative-z local direction instead of
		// terminating the path like the recursive backend does.
		if (cdwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> cd_dist(cd_alpha, cd_alpha);
		float cdwm_x, cdwm_y, cdwm_z;
		cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z, wf_rand(seed), wf_rand(seed), cdwm_x, cdwm_y, cdwm_z);
		float cd_dot = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
		float Fr = FrDielectric(fmaxf(cd_dot, 0.0f), mat.ior);
		if (wf_rand(seed) < Fr) {
			float cdwo_x = 2.0f*cd_dot*cdwm_x - cdwi_x;
			float cdwo_y = 2.0f*cd_dot*cdwm_y - cdwi_y;
			float cdwo_z = 2.0f*cd_dot*cdwm_z - cdwi_z;
			scattered_dir = normalize(cdwo_x*cdtan + cdwo_y*cdbitan + cdwo_z*cdn);
			attenuation   = SS(1.f);
		} else {
			// Multi-bounce escape through the coat, matching
			// optix_device_helpers.h's shade_material() CoatedDiffuse case
			// (see its own comment for why this replaced a single Lambertian
			// bounce + one exit attempt: giving up immediately on a failed
			// exit test discards that sample's energy entirely, rendering
			// far too dark, rather than retrying like the real layered
			// material's random walk does). is_specular is decided once,
			// uniformly, after this if/else (see below) rather than per-
			// branch now that glossy coats get real NEE via CoatedDiffuseBxDF
			// ::f() - pbrt-v4's LayeredBxDF has no closed-form f(wo,wi)
			// either, but the shared BxDF template's own stochastic estimator
			// (src/shared/bxdfs_layered.h, already verified on CPU for #228
			// and the recursive backend for #229) gives us one anyway.
			constexpr int kMaxCoatBounces = 8;
			float3 beta = make_float3(1.0f - Fr, 1.0f - Fr, 1.0f - Fr);
			float3 diff_dir = cdn;
			bool escaped = false;
			for (int cb = 0; cb < kMaxCoatBounces; ++cb) {
				diff_dir = cdn + wf_rand_unit(seed);
				if (wf_near_zero(diff_dir)) diff_dir = cdn;
				diff_dir = normalize(diff_dir);
				beta.x *= mat.albedo.x; beta.y *= mat.albedo.y; beta.z *= mat.albedo.z;

				float dw_x = dot(diff_dir, cdtan), dw_y = dot(diff_dir, cdbitan), dw_z = dot(diff_dir, cdn);
				float dwm_x, dwm_y, dwm_z;
				cd_dist.Sample_wm(dw_x, dw_y, dw_z, wf_rand(seed), wf_rand(seed), dwm_x, dwm_y, dwm_z);
				float cos_out = dw_x*dwm_x + dw_y*dwm_y + dw_z*dwm_z;
				float F_out = FrDielectric(cos_out, 1.0f / mat.ior);
				if (wf_rand(seed) < F_out) continue;  // TIR: bounce again
				beta.x *= (1.0f - F_out); beta.y *= (1.0f - F_out); beta.z *= (1.0f - F_out);
				escaped = true;
				break;
			}
			if (!escaped) { scattered = false; break; }
			attenuation   = albedoSpectrum(beta);
			scattered_dir = diff_dir;
		}
		scattered = (dot(scattered_dir, normal) > 0.0f);

		// Real NEE/MIS for glossy (non-EffectivelySmooth) coats, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment and MaterialType::Conductor's identical-shape block
		// above). brdf_pdf_override uses the coat's own top-surface GGX-
		// reflection VNDF pdf as a proxy for the walk's true (unknown, no
		// closed form) density - matches CPU's coated_diffuse::scatter()
		// (material_pbrt.h), which uses the same ggx_reflection_pdf proxy
		// for its own srec.pdf_ptr. phaseWo carries cdwi (already
		// `-ray_dir`, matching evalGlossyF's `wi_world` convention).
		if (scattered && !cd_dist.EffectivelySmooth()) {
			is_specular = false;
			float swo_x = dot(scattered_dir, cdtan), swo_y = dot(scattered_dir, cdbitan), swo_z = dot(scattered_dir, cdn);
			brdf_pdf_override = (swo_z > 0.0f)
				? ggx_vndf_reflection_pdf(cdwi_x, cdwi_y, cdwi_z, swo_x, swo_y, swo_z, cd_alpha, cd_alpha)
				: 0.0f;
			phaseWo = cdwi;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::ThinDielectric: {
		float3 V = -normalize(h.rayDir);
		float cos_i = fabsf(dot(V, normal));
		float Fr = FrDielectric(cos_i, mat.ior);
		// Account for double transmission (glass slab): T^2
		float T2 = (1.0f - Fr) * (1.0f - Fr);
		if (wf_rand(seed) < Fr / (Fr + T2)) {
			scattered_dir = wf_reflect(-V, normal);
		} else {
			scattered_dir = normalize(h.rayDir); // straight through
		}
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CoatedConductor: {
		// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF).
		// Matches optix_intersection_sphere.h's recursive-path handling: path B
		// (transmit into the coat, bounce off the conductor, exit back through
		// the coat) must weight the conductor's Fresnel by T_in*T_out (how
		// much light actually gets through the dielectric coat both ways) -
		// the previous version here resampled a microfacet from the ORIGINAL
		// viewing direction and skipped both the refraction-into-the-coat
		// step and the T_in/T_out weighting entirely, making the conductor
		// visible at full strength as if the coat weren't there.
		float cc_alpha = sqrtf(mat.fuzz);
		float3 ccn = normal;
		float3 ccup = (fabsf(ccn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 cctan   = normalize(cross(ccup, ccn));
		float3 ccbitan = cross(ccn, cctan);
		float3 ccwi = -normalize(h.rayDir);
		float ccwi_x = dot(ccwi, cctan), ccwi_y = dot(ccwi, ccbitan), ccwi_z = dot(ccwi, ccn);
		if (ccwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> cc_dist(cc_alpha, cc_alpha);
		float ccwm_x, ccwm_y, ccwm_z;
		cc_dist.Sample_wm(ccwi_x, ccwi_y, ccwi_z, wf_rand(seed), wf_rand(seed), ccwm_x, ccwm_y, ccwm_z);
		float cc_dot = ccwi_x*ccwm_x + ccwi_y*ccwm_y + ccwi_z*ccwm_z;
		float F_in = FrDielectric(fmaxf(cc_dot, 0.0f), mat.ior);
		float3 ccwo;
		if (wf_rand(seed) < F_in) {
			// Path A: coat specular reflection
			float wo_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float wo_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float wo_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (wo_z <= 0.0f) { scattered = false; break; }
			float G1 = cc_dist.G1(ccwi_x, ccwi_y, ccwi_z);
			float G  = cc_dist.G(wo_x, wo_y, wo_z, ccwi_x, ccwi_y, ccwi_z);
			float w  = (G1 > 1e-8f) ? G / G1 : 0.0f;
			float fv = F_in * w;
			attenuation = SS(fv);
			ccwo = make_float3(wo_x, wo_y, wo_z);
		} else {
			// Path B: transmit into layer -> conductor bounce -> exit coat
			float w_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float w_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float w_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (w_z > 0.0f) w_z = -w_z;   // ensure pointing downward into layer
			if (w_z == 0.0f) { scattered = false; break; }

			// Flip to conductor frame: "incoming from above" (fw_z > 0)
			float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

			float bwm_x, bwm_y, bwm_z;
			cc_dist.Sample_wm(fw_x, fw_y, fw_z, wf_rand(seed), wf_rand(seed), bwm_x, bwm_y, bwm_z);
			float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
			if (cos_c <= 0.0f) { scattered = false; break; }

			float rwo_x = 2.0f*cos_c*bwm_x - fw_x;
			float rwo_y = 2.0f*cos_c*bwm_y - fw_y;
			float rwo_z = 2.0f*cos_c*bwm_z - fw_z;
			if (rwo_z <= 0.0f) { scattered = false; break; }

			float G1_c = cc_dist.G1(fw_x, fw_y, fw_z);
			float G_c  = cc_dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
			float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

			float3 c_F  = FrConductorRGB(cos_c, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);

			float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);  // inside -> outside
			float T_out = 1.0f - F_out;
			float T_in  = 1.0f - F_in;

			attenuation = albedoSpectrum(make_float3(
				c_F.x * wt_c * T_in * T_out,
				c_F.y * wt_c * T_in * T_out,
				c_F.z * wt_c * T_in * T_out));
			ccwo = make_float3(rwo_x, rwo_y, rwo_z);
		}
		scattered_dir = normalize(ccwo.x*cctan + ccwo.y*ccbitan + ccwo.z*ccn);
		scattered   = (dot(scattered_dir, normal) > 0.0f);

		// Real NEE/MIS for glossy (non-EffectivelySmooth) coats - see
		// MaterialType::CoatedDiffuse's identical-shape block above for the
		// full rationale (shared CoatedConductorBxDF<float>::f() stochastic
		// estimator, GGX top-surface VNDF pdf as the MIS proxy).
		if (scattered && !cc_dist.EffectivelySmooth()) {
			is_specular = false;
			float swo_x = dot(scattered_dir, cctan), swo_y = dot(scattered_dir, ccbitan), swo_z = dot(scattered_dir, ccn);
			brdf_pdf_override = (swo_z > 0.0f)
				? ggx_vndf_reflection_pdf(ccwi_x, ccwi_y, ccwi_z, swo_x, swo_y, swo_z, cc_alpha, cc_alpha)
				: 0.0f;
			phaseWo = ccwi;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::DiffuseTransmission: {
		// pbrt-v4 DiffuseTransmissionBxDF - matches optix_intersection_sphere.h's
		// recursive-path handling exactly: albedo = reflectance R (same
		// hemisphere), emission = transmittance T (reused field, not
		// derived as 1-R), max-component probability weighting.
		//
		// is_specular = true (this material's BSDF is genuinely non-specular,
		// but the flag here means "skip the caller's generic NEE block", see
		// below): recursive's own shade_material() case for this material has
		// no explicit light-sampling code at all, so setting is_specular=false
		// here (matching a naive reading of "diffuse = non-specular") made
		// this branch the ONLY one of the two backends' NEE blocks that ran
		// for DiffuseTransmission - and it ran with the caller's hardcoded
		// Lambertian-shaped 1/pi white BRDF default, which is wrong on two
		// counts: it ignores mat.albedo's actual color entirely, and it can't
		// tell the reflective lobe (BRDF = R/pi) from the transmissive one
		// (BRDF = T/pi, and only for a light on the FAR side of the surface,
		// which the block's `dot(to_light, normal) > 0` gate would wrongly
		// reject anyway). CPU's own diffuse_transmission material (see
		// material_pbrt.h) DOES support real two-hemisphere NEE via its own
		// cosine_pdf(±normal)/skip_pdf=false machinery - a real BRDF-aware fix
		// here would port that, not just disable NEE - but until that lands on
		// both GPU backends, disabling it here removes the wrong, colour-blind
		// contribution and matches what recursive already (implicitly) ships:
		// zero explicit NEE, all illumination via the BSDF-sampled bounce.
		float3 R = mat.albedo;
		float3 T_col = mat.emission;
		float pr = fmaxf(R.x, fmaxf(R.y, R.z));
		float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
		if (pr + pt <= 0.0f) { scattered = false; break; }
		if (wf_rand(seed) < pr / (pr + pt)) {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(R);
		} else {
			float3 neg_n = -normal;
			scattered_dir = normalize(neg_n + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = neg_n;
			attenuation = albedoSpectrum(T_col);
		}
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::NormalizedFresnel: {
		float weight;
		wf_sample_normalized_fresnel(mat.ior, normal, seed, scattered_dir, weight, brdf_pdf_override);
		attenuation = SS(weight);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Medium: {
		// Homogeneous participating medium - mirrors optix_intersection_sphere.h's
		// closesthit Medium case. Geometry (entry/exit roots) was already
		// recomputed in __closesthit__wf_sphere and handed over via h.t (near)
		// and h.mediumTFar (far); here we sample the free-path distance and
		// either scatter inside via the HG phase function or pass straight
		// through to the exit point, matching the recursive path's
		// is_specular=true (no NEE/MIS for volume scattering, not yet implemented).
		float t_near = h.t;
		float t_far  = h.mediumTFar;
		float dist_inside = fmaxf(0.0f, t_far - t_near);
		float sigma_t = mat.ior;
		float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;
		float3 unit_dir = normalize(h.rayDir);
		if (free_path < dist_inside) {
			float medium_t = t_near + free_path;
			hit_point     = h.rayOrigin + medium_t * unit_dir;
			scattered_dir = wf_sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
			attenuation   = albedoSpectrum(mat.albedo);
		} else {
			hit_point     = h.rayOrigin + t_far * unit_dir;
			scattered_dir = unit_dir;  // straight through, no interaction
			attenuation   = SS(1.f);
		}
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CloudMedium: {
		// Heterogeneous, procedural Perlin-noise cloud - mirrors
		// optix_intersection_sphere.h's closesthit CloudMedium case exactly
		// (see that comment for the full reasoning). Unlike Medium above,
		// this does NOT use h.t/h.mediumTFar (the sphere's own near/far
		// roots, which __closesthit__wf_sphere only recomputes for
		// MaterialType::Medium/DielectricMedium - see that function's
		// needsNearFar) - CloudMedium's own world-space AABB, tested fresh
		// against the true ray via CloudMedium<float>::sample_ray(), is what
		// actually bounds it; the trigger sphere's own geometry is
		// irrelevant beyond having gotten the ray into this branch at all.
		const CloudMedium<float>& cloud = cloudMediums[(int)mat.cloud_medium_extra.cloudMediumIdx];
		float3 unit_dir = normalize(h.rayDir);
		float ray_o3[3] = { h.rayOrigin.x, h.rayOrigin.y, h.rayOrigin.z };
		float ray_d3[3] = { unit_dir.x, unit_dir.y, unit_dir.z };

		auto maj_it = cloud.sample_ray(ray_o3, ray_d3, 1e30f);
		float segMin, segMax, sigma_maj;
		bool has_seg = maj_it.next(segMin, segMax, sigma_maj);

		bool did_scatter = false;
		float medium_t = 0.0f;
		if (has_seg && sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			// Cap matches optix_intersection_sphere.h's CloudMedium branch.
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float3 p = h.rayOrigin + tt * unit_dir;
				float mx, my, mz;
				cloud.world_to_medium_pt(p.x, p.y, p.z, mx, my, mz);
				float d = gpu_cloud_density(cloud, mx, my, mz);
				float sigma_s_local = d * cloud.sigma_s;
				if (wf_rand(seed) < sigma_s_local / sigma_maj) {
					did_scatter = true;
					medium_t      = tt;
					scattered_dir = wf_sample_henyey_greenstein(unit_dir, mat.fuzz, seed);
					attenuation   = albedoSpectrum(mat.albedo);
				}
			}
			if (!did_scatter) medium_t = segMax;
		} else {
			medium_t = h.t;  // missed the medium's own AABB - pass straight through
		}
		if (!did_scatter) {
			scattered_dir = unit_dir;
			attenuation   = SS(1.f);
		}
		hit_point   = h.rayOrigin + medium_t * unit_dir;
		scattered   = true;
		is_specular = true;
		break;
	}

	case MaterialType::RgbGridMedium: {
		// Heterogeneous per-voxel R/G/B scattering grid - mirrors
		// optix_intersection_sphere.h's closesthit RgbGridMedium case exactly
		// (see that comment for the full reasoning, including why this uses
		// one single GLOBAL majorant rather than CPU's real per-voxel DDA
		// majorant grid).
		const GpuRgbGridMedium& grid = rgbGridMediums[(int)mat.rgb_grid_medium_extra.rgbGridMediumIdx];
		float3 unit_dir = normalize(h.rayDir);

		float mox = grid.mat[0]*h.rayOrigin.x + grid.mat[1]*h.rayOrigin.y + grid.mat[2]*h.rayOrigin.z + grid.translate[0];
		float moy = grid.mat[3]*h.rayOrigin.x + grid.mat[4]*h.rayOrigin.y + grid.mat[5]*h.rayOrigin.z + grid.translate[1];
		float moz = grid.mat[6]*h.rayOrigin.x + grid.mat[7]*h.rayOrigin.y + grid.mat[8]*h.rayOrigin.z + grid.translate[2];
		float mdx = grid.mat[0]*unit_dir.x + grid.mat[1]*unit_dir.y + grid.mat[2]*unit_dir.z;
		float mdy = grid.mat[3]*unit_dir.x + grid.mat[4]*unit_dir.y + grid.mat[5]*unit_dir.z;
		float mdz = grid.mat[6]*unit_dir.x + grid.mat[7]*unit_dir.y + grid.mat[8]*unit_dir.z;

		float segMin = 0.0f, segMax = 1e30f;
		bool has_seg = true;
		{
			float invd, s0, s1;
			invd = (mdx != 0.0f) ? 1.0f/mdx : 1e30f;
			s0 = (0.0f - mox)*invd; s1 = (1.0f - mox)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdy != 0.0f) ? 1.0f/mdy : 1e30f;
			s0 = (0.0f - moy)*invd; s1 = (1.0f - moy)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdz != 0.0f) ? 1.0f/mdz : 1e30f;
			s0 = (0.0f - moz)*invd; s1 = (1.0f - moz)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;
		}

		bool did_scatter = false;
		float medium_t = 0.0f;
		if (has_seg && grid.sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			const int voxelCount = grid.nx * grid.ny * grid.nz;
			const float* rData = rgbGridData + grid.dataOffset;
			const float* gData = rData + voxelCount;
			const float* bData = gData + voxelCount;
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / grid.sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
				float dr = gpu_rgb_grid_trilinear(rData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float dg = gpu_rgb_grid_trilinear(gData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float db = gpu_rgb_grid_trilinear(bData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float sr = dr * grid.sigma_scale, sg = dg * grid.sigma_scale, sb = db * grid.sigma_scale;
				float sigma_t_local = fmaxf(sr, fmaxf(sg, sb));
				if (wf_rand(seed) < sigma_t_local / grid.sigma_maj) {
					did_scatter = true;
					medium_t      = tt;
					scattered_dir = wf_sample_henyey_greenstein(unit_dir, grid.phase_g, seed);
					float maxc = fmaxf(sr, fmaxf(sg, fmaxf(sb, 1e-6f)));
					attenuation = albedoSpectrum(make_float3(sr/maxc, sg/maxc, sb/maxc));
				}
			}
			if (!did_scatter) medium_t = segMax;
		} else {
			medium_t = h.t;
		}
		if (!did_scatter) {
			scattered_dir = unit_dir;
			attenuation   = SS(1.f);
		}
		hit_point   = h.rayOrigin + medium_t * unit_dir;
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::Hair: {
		// Marschner/Chiang fiber scattering - see wf_sample_hair_material's
		// comment above. Matches hair_material.h's skip_pdf=true: no NEE/MIS
		// (is_specular=true), and the result needs unboundedSpectrum (not
		// albedoSpectrum) since it's already divided by the sample pdf.
		float3 sdir, atten;
		if (wf_sample_hair_material(h.rayDir, normal, mat, seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	case MaterialType::DielectricMedium: {
		// Combined dielectric surface + internal medium - mirrors
		// optix_intersection_sphere.h's closesthit case exactly. On entry
		// (h.frontFace) the direct dielectric surface always wins the bounce
		// (the medium's sampled hit distance can never be closer than the
		// entry surface), so just refract/reflect normally. On the exit
		// surface (frontFace false), __closesthit__wf_sphere already
		// recomputed the near/far roots into h.t/h.mediumTFar (same
		// mechanism as MaterialType::Medium above) - sample a free path
		// through that interior segment and either scatter via the HG phase
		// function or fall through to a normal exit refraction/reflection
		// at the far surface.
		if (h.frontFace) {
			attenuation   = SS(1.f);
			scattered_dir = wf_dielectric_scatter(h.rayDir, normal, true, mat.ior, seed);
			is_specular   = true;  // genuinely specular (Dirac-delta) dielectric bounce
		} else {
			float t_near = h.t;
			float t_far  = h.mediumTFar;
			float dist_inside = fmaxf(0.0f, t_far - t_near);
			float sigma_t = mat.eta_c.x;  // dielectric_medium_extra.sigma_t
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;
			float3 unit_dir = normalize(h.rayDir);
			if (free_path < dist_inside) {
				float medium_t = t_near + free_path;
				hit_point     = h.rayOrigin + medium_t * unit_dir;
				float g       = mat.fuzz;
				scattered_dir = wf_sample_henyey_greenstein(unit_dir, g, seed);
				attenuation   = albedoSpectrum(mat.albedo);
				// Real NEE+MIS at the phase-function scatter event, matching
				// CPU's hg_phase_material (skip_pdf=false) - see
				// optix_intersection_sphere.h's identical fix for the full
				// root-cause derivation (this closes B13/SubsurfaceSlab's
				// ~32-38% CPU-brighter gap). Unlike the recursive backend,
				// this path's shadow rays are queued (wf_finish_material_
				// scatter, below) rather than traced inline, so all this case
				// needs to do is hand off phaseWo/phaseG and flip is_specular.
				phaseWo = -unit_dir;
				phaseG  = g;
				brdf_pdf_override = wf_hg_phase_value(dot(phaseWo, scattered_dir), g);
				is_specular = false;
			} else {
				hit_point     = h.rayOrigin + t_far * unit_dir;
				attenuation   = SS(1.f);
				scattered_dir = wf_dielectric_scatter(h.rayDir, normal, false, mat.ior, seed);
				is_specular   = true;  // genuinely specular exit refraction/reflection
			}
		}
		scattered   = true;
		break;
	}
	case MaterialType::NormalMappedLambertian: {
		// Lambertian with a perturbed shading normal from a tangent-space
		// RGB normal-map texture - mirrors optix_intersection_sphere.h's
		// (spheres) / optix_intersection_triangle.h's (triangles) closesthit
		// cases. dpdu comes from h.objNormal, whose meaning depends on which
		// geometry was hit - see HitWorkItem::objNormal's own comment.
		float3 dpdu;
		if (h.geomType == 3) {
			// Triangle: h.objNormal already IS the real, world-space,
			// UV-derived tangent (see HitWorkItem::objNormal's own comment
			// and __closesthit__wf_triangle) - use directly. Previously
			// this branch didn't exist, so every triangle fell through to
			// the sphere-only cross-product path below; since objNormal is
			// never populated for triangle hits (stays zero), that always
			// degenerated to the (1,0,0) fallback regardless of the
			// triangle's real surface orientation - a silently wrong (not
			// crashing) shading tangent for every normal-mapped triangle.
			dpdu = h.objNormal;
		} else {
			const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
			const float3 t = cross(world_up, h.objNormal);
			const float tlen = length(t);
			dpdu = (tlen > 1e-6f) ? (t / tlen) : make_float3(1.0f, 0.0f, 0.0f);
		}

		// Decode the tangent-space normal from the map texture: 2*RGB-1,
		// normalize (fallback (0,0,1) i.e. "no perturbation" if degenerate) -
		// matches CPU's normal_map_material::apply() exactly.
		const float3 packed = wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point);
		float ns_x = 2.0f * packed.x - 1.0f;
		float ns_y = 2.0f * packed.y - 1.0f;
		float ns_z = 2.0f * packed.z - 1.0f;
		const float ns_len = sqrtf(ns_x * ns_x + ns_y * ns_y + ns_z * ns_z);
		if (ns_len > 1e-8f) { ns_x /= ns_len; ns_y /= ns_len; ns_z /= ns_len; }
		else                { ns_x = 0.0f; ns_y = 0.0f; ns_z = 1.0f; }

		float out_nx, out_ny, out_nz;
		apply_normal_map(ns_x, ns_y, ns_z, normal.x, normal.y, normal.z,
			dpdu.x, dpdu.y, dpdu.z, out_nx, out_ny, out_nz);
		normal = make_float3(out_nx, out_ny, out_nz);  // perturbed shading normal;
		                                                // NEE below reads this same
		                                                // outer-scope `normal`.

		// Shade as plain Lambertian (mat.albedo) using the perturbed normal -
		// reuses the Lambertian case's logic rather than duplicating it.
		// textureIdx means "normal map" for this material type, not "albedo
		// texture" (matches recursive - CPU never combines the two).
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Principled: {
		// Disney/pbrt-v4-style multi-lobe BSDF - see wf_sample_principled_material's
		// comment above. Matches principled_material.h's skip_pdf=true: no
		// NEE/MIS (is_specular=true), and the result needs unboundedSpectrum
		// (not albedoSpectrum) since it's already divided by the sample pdf -
		// same convention as MaterialType::Hair above.
		float3 sdir, atten;
		if (wf_sample_principled_material(h.rayDir, normal, mat, seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	// Every MaterialType except DiffuseLight (handled via the early-exit
	// above, before this switch) has a real case above - this default: is
	// only ever reached by a genuinely new MaterialType nobody wired into
	// this backend yet, exactly the class of gap this project's own history
	// has had to find and fix reactively more than once (see e.g.
	// "Wavefront gaps" in prior commits). The obvious fix - drop the
	// catch-all and let the compiler's missing-enum-case-in-switch warning
	// catch it at compile time - does not actually work here: nvcc's
	// device-code frontend (confirmed empirically against this project's
	// CUDA 13.2 toolchain, including every --diag-warn flag it exposes)
	// does not implement that diagnostic the way MSVC's C4062 or clang's
	// -Wswitch do, so a missing case compiles silently clean either way.
	// Trap loudly instead of absorbing the ray, so the gap surfaces the
	// moment a real render exercises it. Not gated behind NDEBUG - nvcc's
	// own invocation for this file never defines it either way (see
	// build_optix.targets), so the guard is simply always on.
	default: {
		printf("[EVAL-MATERIALS] unhandled MaterialType %d\n", (int)mat.type);
		__trap();
	}
	}

	if (!scattered) {
		// Path absorbed — accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	wf_finish_material_scatter(mat.type, matEta, h.materialIdx, normal, hit_point, seed,
		throughput, radiance, swl, attenuation, scattered_dir, is_specular, brdf_pdf_override,
		phaseWo, phaseG,
		h.pixelIndex, h.depth,
		spheres, quads, triangles, bilinearPatches, materials,
		lightIndices, lightKinds, aliasTable, numLights,
		punctualLights, numPunctualLights,
		skyColor, shadow_eps, skyDist,
		shadowQueue, nextRayQueue, framebuffer);
}

// ============================================================================
// evaluate_materials_simple
//
// Twin of evaluate_materials() above, scoped to just MaterialType::Lambertian
// and MaterialType::Metal - the two cheap material types routed into
// simpleHitQueue at push time (see wavefront_programs.cu's
// __raygen__wf_intersect). Everything shared with evaluate_materials() below
// (the spectral-state preamble, the maxDepth cutoff, the Lambertian/Metal
// case bodies themselves, the wf_finish_material_scatter tail call) is copied
// verbatim, not reimplemented - this kernel exists ONLY to replace the big
// switch with a 2-way if/else, so the compiler doesn't have to reserve
// registers for the switch's expensive arms (CoatedDiffuse/CoatedConductor/
// Principled/RoughDielectric/...) for threads that will never take them,
// improving occupancy for this launch. No DiffuseLight early-exit and no
// default:/__trap() guard are needed here - the routing at push time
// guarantees only Lambertian/Metal hits ever reach this queue, so unlike
// evaluate_materials() there is no "genuinely new MaterialType" case to
// defend against structurally (a routing bug would show up immediately as a
// visibly wrong render, not a silent gap, since it's a two-way branch over a
// closed set of exactly two types this kernel is defined to handle).
//
// Fewer scene-data parameters than evaluate_materials(): no bssrdfProbeQueue
// (Subsurface-only), no cloud/RGB-grid medium buffers or measured-BRDF
// tables (none of those material types can reach this queue).
// ============================================================================
extern "C" __global__ void evaluate_materials_simple(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      framebuffer,
	// Scene data
	const SphereData*   spheres,
	const QuadData*     quads,
	const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches,
	const MaterialData* materials,
	const int*          lightIndices,
	const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	const TextureData*  textures,
	const unsigned char* texturePixels,
	int maxDepth,
	float3 skyColor,
	float shadowRayEpsilon,
	GpuSkyDistribution skyDist
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;
	const float shadow_eps = (shadowRayEpsilon > 0.0f) ? shadowRayEpsilon : 0.01f;

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SS throughput(h.throughput);
	SS radiance(h.radiance);
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = h.wavelengths[i];
		swl.pdf[i]    = h.wavelength_pdfs[i];
	}

	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	if (h.depth >= maxDepth) {
		// Max depth reached — just accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);
	bool   scattered        = false;
	bool   is_specular      = false;
	float  brdf_pdf_override = -1.0f;
	// Never set by Lambertian/Metal - passed through to wf_finish_material_
	// scatter as harmless zero, matching evaluate_materials()'s own default
	// for every material type that doesn't use them.
	float3 phaseWo = make_float3(0.0f, 0.0f, 0.0f);
	float matEta = mat.ior;
	float  phaseG  = 0.0f;

	auto albedoSpectrum = [&](float3 rgb) -> SS {
		float c0, c1, c2;
		float r = rgb.x < 0.f ? 0.f : (rgb.x > 1.f ? 1.f : rgb.x);
		float g = rgb.y < 0.f ? 0.f : (rgb.y > 1.f ? 1.f : rgb.y);
		float b = rgb.z < 0.f ? 0.f : (rgb.z > 1.f ? 1.f : rgb.z);
		dev_srgb_to_coeffs(r, g, b, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = poly(swl.lambda[i]);
		return s;
	};

	if (mat.type == MaterialType::Lambertian) {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		const float3 lambertianColor = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point)
			: mat.albedo;
		attenuation = albedoSpectrum(lambertianColor);
		scattered   = true;
		is_specular = false;
	} else { // MaterialType::Metal
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
	}

	if (!scattered) {
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

	wf_finish_material_scatter(mat.type, matEta, h.materialIdx, normal, hit_point, seed,
		throughput, radiance, swl, attenuation, scattered_dir, is_specular, brdf_pdf_override,
		phaseWo, phaseG,
		h.pixelIndex, h.depth,
		spheres, quads, triangles, bilinearPatches, materials,
		lightIndices, lightKinds, aliasTable, numLights,
		punctualLights, numPunctualLights,
		skyColor, shadow_eps, skyDist,
		shadowQueue, nextRayQueue, framebuffer);
}

// Uplift a flat RGB color to a spectral sample at the given hero
// wavelengths. Standalone copy of the `liftEmission` lambda used inside
// evaluate_materials below (that one captures its `swl` by reference from
// its own hit context; accumulate_miss needs the same math against a
// MissWorkItem's own wavelengths instead).
//
// isIlluminant: true for a light-source colour (background/sky radiance in
// accumulate_miss below), which needs the D65 illuminant factor -- see
// liftEmission's own comment above for why. false (default) for an
// already-computed reflectance-like weight (e.g. resolve_bssrdf_exit's Sp
// spatial term), which must NOT get an extra illuminant multiply -- it's
// not a colour being lit, it's already a throughput.
__device__ __forceinline__ SampledSpectrum<kWFNWavelengths> wf_lift_rgb_to_spectrum(
	float3 rgb, const SampledWavelengths<kWFNWavelengths>& swl, bool isIlluminant = false
) {
	using SS = SampledSpectrum<kWFNWavelengths>;
	float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
	float sc = 2.f * m;
	if (sc <= 0.f) return SS(0.f);
	float c0, c1, c2;
	dev_srgb_to_coeffs(rgb.x / sc, rgb.y / sc, rgb.z / sc, c0, c1, c2);
	RGBSigmoidPolynomial poly(c0, c1, c2);
	SS s(0.f);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		s[i] = sc * poly(swl.lambda[i]);
		if (isIlluminant) s[i] *= dev_sample_d65(swl.lambda[i]);
	}
	return s;
}

// ============================================================================
// Kernel 2b — resolve_bssrdf_exit
//   Consumes BssrdfExitWorkItem (the result of __raygen__wf_probe's BSSRDF
//   probe walk, wavefront_probe.h) - MaterialType::Subsurface, Phase 2.
//
//   found==0 (probe walk failed to find a valid exit point): flush this
//   path's still-pending `radiance` (deferred specular-chain emission) if
//   nonzero and stop - mirrors evaluate_materials's own `if (!scattered)`
//   handling, and BssrdfExitWorkItem's own comment on why this can't be
//   silently dropped.
//
//   found==1: resume the path as an ordinary MaterialType::
//   NormalizedFresnel(ior=materials[matIdx].ior) non-specular bounce at
//   exitPos/exitNormal, weighted by Sp/(sampleProb*pdf) - mirrors camera.h::
//   sample_bssrdf_exit()'s hand-off and the recursive backend's identical
//   shade_material() Subsurface-case hand-off (optix_device_helpers.h).
//
//   Folds sss_weight into `throughput` BEFORE calling wf_finish_material_
//   scatter() - rather than scaling emission after the fact the way the
//   recursive backend's own deferred-outer-multiply payload design does:
//   wavefront's NEE math already multiplies every Ld term by `throughput`
//   directly (see wf_finish_material_scatter()'s own Ld formulas), so pre-
//   weighting throughput here makes that existing math do the right thing
//   automatically, both for this bounce's own NEE and for the
//   `new_throughput = throughput * attenuation` the next bounce inherits -
//   no separate post-hoc emission scale needed. `radiance` (unrelated prior-
//   bounce emission, not part of this BSSRDF event) is passed through
//   unweighted, exactly as wf_finish_material_scatter() already treats it.
// ============================================================================

extern "C" __global__ void resolve_bssrdf_exit(
	WorkQueue<BssrdfExitWorkItem> exitQueue,
	int                           numExit,
	WorkQueue<RayWorkItem>        nextRayQueue,
	WorkQueue<ShadowRayWorkItem>  shadowQueue,
	float3*                       framebuffer,
	const SphereData*   spheres,
	const QuadData*     quads,
	const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches,
	const MaterialData* materials,
	const int*          lightIndices,
	const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	float3 skyColor,
	float shadowRayEpsilon,
	GpuSkyDistribution skyDist
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Same defensive capacity guard as accumulate_shadow's own version of
	// this comment (this file, above) and __raygen__wf_shadow's (wavefront_
	// programs.cu): numExit is a host-read counter value, which
	// WorkQueue::push() can inflate past exitQueue.capacity without that
	// many items actually being written.
	if (idx >= numExit || idx >= exitQueue.capacity) return;

	const BssrdfExitWorkItem& item = exitQueue.items[idx];

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = item.wavelengths[i];
		swl.pdf[i]    = item.wavelength_pdfs[i];
	}
	SS throughput(item.throughput);
	SS radiance(item.radiance);
	unsigned int seed = item.seed;

	// <= 0 (zero-init default) means "use the standard 0.01f" - mirrors
	// evaluate_materials's own identical default.
	const float shadow_eps = (shadowRayEpsilon > 0.0f) ? shadowRayEpsilon : 0.01f;

	if (!item.found) {
		// Probe walk failed to find a valid exit point - path terminates,
		// same as evaluate_materials's own `if (!scattered)` handling.
		if ((bool)radiance) {
			auto xyz = SampledSpectrumToXYZ(radiance, swl, d_cie_x, d_cie_y, d_cie_z,
											kDevCIEMin, kDevCIENSamples);
			float r, g, b;
			wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
			atomicAdd(&framebuffer[item.pixelIndex].x, r);
			atomicAdd(&framebuffer[item.pixelIndex].y, g);
			atomicAdd(&framebuffer[item.pixelIndex].z, b);
		}
		return;
	}

	const MaterialData& mat = materials[item.matIdx];
	const float eta = mat.ior;

	// sss_weight = Sp / (sampleProb * pdf) - see bssrdf_probe_walk()'s own
	// comment (optix_device_helpers.h) for the full derivation. Sp is an
	// unbounded-positive RGB value (a BSSRDF profile value, not a [0,1]
	// albedo), so it needs the same normalize-by-2*max uplift technique
	// evaluate_materials's own unboundedSpectrum lambda uses for
	// MaterialType::Hair/Principled results - wf_lift_rgb_to_spectrum()
	// above already IS that same technique in standalone form.
	SS Sp_spec = wf_lift_rgb_to_spectrum(item.Sp, swl);
	float denom = item.sampleProb * item.pdf;
	SS sss_weight = (denom > 1e-12f) ? (Sp_spec / denom) : SS(0.f);

	SS weightedThroughput = throughput * sss_weight;

	float3 scattered_dir;
	float weight, brdf_pdf_override;
	wf_sample_normalized_fresnel(eta, item.exitNormal, seed, scattered_dir, weight, brdf_pdf_override);
	SS attenuation = SS(weight);

	wf_finish_material_scatter(MaterialType::NormalizedFresnel, eta, /*matIdx=*/-1,
		item.exitNormal, item.exitPos, seed,
		weightedThroughput, radiance, swl, attenuation, scattered_dir,
		/*is_specular=*/false, brdf_pdf_override,
		/*phaseWo=*/make_float3(0.0f, 0.0f, 0.0f), /*phaseG=*/0.0f,
		item.pixelIndex, item.depth,
		spheres, quads, triangles, bilinearPatches, materials,
		lightIndices, lightKinds, aliasTable, numLights,
		punctualLights, numPunctualLights,
		skyColor, shadow_eps, skyDist,
		shadowQueue, nextRayQueue, framebuffer);
}

// ============================================================================
// Kernel 3 — accumulate_miss
//   For rays that escaped the scene, add the flat background color (black
//   by default - see GpuCameraParams::backgroundColor in optix_types.h for
//   why a flat color, not an environment map, matches what the CPU
//   renderer actually does for every scene).
// ============================================================================

extern "C" __global__ void accumulate_miss(
	WorkQueue<MissWorkItem> missQueue,
	int                     numMiss,
	float3*                 framebuffer,
	float3                  backgroundColor,
	// Real importance-sampled HDR sky - see evaluate_materials's own
	// GpuSkyDistribution parameter comment.
	GpuSkyDistribution      skyDist
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numMiss) return;

	const MissWorkItem& m = missQueue.items[idx];
	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = m.wavelengths[i];
		swl.pdf[i]    = m.wavelength_pdfs[i];
	}
	SS throughput(m.throughput);
	// Real per-direction radiance for an image sky (wf_sky_radiance(),
	// wavefront_sky_light.h - uses m.rayDir, previously carried but unused -
	// see MissWorkItem::rayDir's own comment), or the flat backgroundColor
	// otherwise. MIS weight against the sky-NEE strategy (evaluate_
	// materials's own sky-NEE block), mirroring optix_miss.h's recursive-
	// backend equivalent exactly - see that comment for the full reasoning.
	// m.brdf_pdf == 0 for the primary ray or a specular bounce (RayWorkItem::
	// brdf_pdf's own sentinel) -> full weight; otherwise discounted by the
	// balance heuristic against the sky strategy's own pdf at this direction
	// (wf_sky_pdf_for_mis() - the real importance-sampled pdf for an image
	// sky, or the uniform-sphere 1/4pi constant otherwise), so a genuinely-
	// escaped ray doesn't double-count against the NEE sample already taken
	// at the previous hit. `backgroundColor` still gates "does this scene
	// have a sky at all" below, matching prior behavior exactly.
	float3 color = wf_sky_radiance(skyDist, m.rayDir, backgroundColor);
	float3 weightedBackground = color;
	if (m.brdf_pdf > 0.0f && (backgroundColor.x > 0.0f || backgroundColor.y > 0.0f || backgroundColor.z > 0.0f)) {
		const float pdf_sky = wf_sky_pdf_for_mis(skyDist, m.rayDir);
		float w_b = wf_mis(m.brdf_pdf, pdf_sky);
		weightedBackground = w_b * color;
	}
	// m.radiance is already fully weighted (accumulated by evaluate_materials
	// as radiance = radiance + throughput * emission at each light hit along
	// a deferred-flush specular chain - see MissWorkItem::radiance's own
	// comment) - add it directly, not multiplied by the CURRENT throughput
	// again, the same way evaluate_materials's own flush sites do.
	SS L = throughput * wf_lift_rgb_to_spectrum(weightedBackground, swl, /*isIlluminant=*/true) + SS(m.radiance);
	if (!(bool)L) return;

	auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z, kDevCIEMin, kDevCIENSamples);
	float r, g, b;
	wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
	atomicAdd(&framebuffer[m.pixelIndex].x, r);
	atomicAdd(&framebuffer[m.pixelIndex].y, g);
	atomicAdd(&framebuffer[m.pixelIndex].z, b);
}

// ============================================================================
// Kernel 4 — accumulate_shadow
//   Runs after the OptiX shadow-trace pass.  ShadowRayWorkItems that were NOT
//   occluded have their `occluded` flag cleared by the shadow miss program;
//   we accumulate their Ld into the framebuffer.
//   (The `occluded` flag is stored in a separate bool array passed alongside
//    the shadow queue items — see wavefront_path_tracer.cpp.)
// ============================================================================

extern "C" __global__ void accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	int                          numShadow,
	const bool*                  occluded,   // per-item occlusion result
	float3*                      framebuffer
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Must also guard against shadowQueue.capacity, not just the host-
	// supplied numShadow: numShadow is *shadowCounter read back on the host
	// (WavefrontPathTracer::render(), readQueueSize()), and WorkQueue::push()
	// (wavefront_types.h) keeps incrementing that counter even once the
	// backing d_shadowItems_/d_occluded_ buffers (sized to shadowQueue.
	// capacity = queueCapacity_) are full - it only stops writing items[]
	// past capacity. A bounce that legitimately queues more shadow rays than
	// capacity (confirmed: scene B2/"Cornell Rough Metal" combines area-
	// light NEE with a non-zero-backgroundColor sky-NEE push per hit - see
	// __raygen__wf_shadow's own version of this comment, wavefront_
	// programs.cu) would otherwise read occluded[idx]/shadowQueue.items[idx]
	// past their allocations here too.
	if (idx >= numShadow || idx >= shadowQueue.capacity) return;

	if (!occluded[idx]) {
		const ShadowRayWorkItem& s = shadowQueue.items[idx];
		// Reconstruct spectral wavelengths for XYZ conversion
		using SS  = SampledSpectrum<kWFNWavelengths>;
		using SWL = SampledWavelengths<kWFNWavelengths>;
		SS Ld(s.Ld);
		SWL swl;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			swl.lambda[i] = s.wavelengths[i];
			swl.pdf[i]    = s.wavelength_pdfs[i];
		}
		auto xyz = SampledSpectrumToXYZ(Ld, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		atomicAdd(&framebuffer[s.pixelIndex].x, r);
		atomicAdd(&framebuffer[s.pixelIndex].y, g);
		atomicAdd(&framebuffer[s.pixelIndex].z, b);
	}
}

// ============================================================================
// Kernel 5 — reset_queue_counter  (single-thread helper)
// ============================================================================

extern "C" __global__ void reset_queue_counter(int* counter) {
	*counter = 0;
}

// ============================================================================
// Kernel 6 — normalize_framebuffer
//   Divides accumulated radiance by samplesPerPixel for the final image.
// ============================================================================

extern "C" __global__ void normalize_framebuffer(
	float3*      framebuffer,
	unsigned int numPixels,
	float        invSPP
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	framebuffer[idx] = framebuffer[idx] * invSPP;
}
