#pragma once
// optix_device_basics.h -- part 1 of 5 of optix_device_helpers.h (included by it, in order; not meant to be included on its own).

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

// Recursive backend's simplified 3-representative-wavelength dispersion
// scheme - see shade_material()'s own inout_rgb_channel parameter comment
// for the full rationale versus CPU/wavefront's real continuous
// SampledWavelengths<4> spectral integration. kRgbChannelUnset (3) means
// "no dispersive event yet, this path stays full RGB" - payload register
// p24 (optix_raygen.h) is initialized to this and only ever changes to
// 0/1/2 the first time the path hits a dispersive MaterialType::Dielectric.
// kRgbChannelWavelengthNm are the sRGB primaries' own commonly-cited
// dominant wavelengths (not importance-sampled - a fixed, representative
// value per channel), used as CauchyEta()'s lambda_nm input. Deliberately
// the SAME values as this codebase's own pre-existing "3 fixed
// representative wavelengths" convention (src/TheRestOfYourLife/
// material_pbrt.h's `measured` class kLambdaR/G/B, mirrored on GPU at
// gpu/optix/optix_measured_bxdf.h's own `lambda[3]` local) - a code-review
// pass on this feature's own first version found it had hand-picked a
// near-duplicate, off-by-1nm table (611/464 vs the established 612/465)
// instead of matching the one already in use for the identical purpose.
static constexpr unsigned int kRgbChannelUnset = 3u;
__device__ __constant__ float kRgbChannelWavelengthNm[3] = { 612.0f, 549.0f, 465.0f };

// Pixel reconstruction filter weight for a sample at sub-pixel offset
// (ox, oy) in [-0.5, 0.5] - device-side port of src/shared/filter.h's
// PixelFilterDispatch::evaluate(), same 5 shapes, evaluated at a fixed
// radius=0.5. This is now ONLY the fallback for a launch that never went
// through OptiXRenderer::render() (GpuCameraParams::filterSampler == nullptr):
// the normal path draws each sample's sub-pixel position from the scene's
// real, possibly multi-pixel-wide filter (FilterSampler, same table as CPU's
// camera.h) and weights it by FilterSample::weight - see that field's own
// comment. kind: 0=gaussian 1=box 2=triangle 3=mitchell 4=sinc
// (GpuCameraParams::filterKind). Shared by both GPU backends (optix_raygen.h,
// wavefront_kernels_camera.cu's generate_camera_rays).
__device__ __forceinline__ float gpu_filter_evaluate(
		int kind, float B, float C, float sigma, float tau, float ox, float oy) {
	const float radius = 0.5f;
	if (kind == 1) return 1.0f;  // box: uniform weight
	if (kind == 2) {  // triangle (tent): max(0, radius-|x|) * max(0, radius-|y|)
		float tx = fmaxf(0.0f, radius - fabsf(ox));
		float ty = fmaxf(0.0f, radius - fabsf(oy));
		return tx * ty;
	}
	if (kind == 3) {  // mitchell (pbrt-v4 Mitchell1D, separable)
		auto mitchell1d = [B, C](float x) -> float {
			x = fabsf(x);
			if (x <= 1.0f)
				return ((12.0f - 9.0f*B - 6.0f*C) * x*x*x
					  + (-18.0f + 12.0f*B + 6.0f*C) * x*x
					  + (6.0f - 2.0f*B)) * (1.0f / 6.0f);
			else if (x <= 2.0f)
				return ((-B - 6.0f*C) * x*x*x
					  + (6.0f*B + 30.0f*C) * x*x
					  + (-12.0f*B - 48.0f*C) * x
					  + (8.0f*B + 24.0f*C)) * (1.0f / 6.0f);
			return 0.0f;
		};
		return mitchell1d(2.0f*ox/radius) * mitchell1d(2.0f*oy/radius);
	}
	if (tau <= 0.0f) tau = 3.0f;  // GpuCameraParams has no in-class defaults
	                              // (see its own filterKind comment) - guard
	                              // the zero-init value here instead.
	if (kind == 4) {  // sinc (windowed Lanczos) - Sinc/WindowedSinc from
		// src/shared/scalar_math.h (already CPU_GPU-tagged, pulled in
		// transitively via noise.h above).
		return WindowedSinc(ox, radius, tau) * WindowedSinc(oy, radius, tau);
	}
	// gaussian (kind == 0, or anything unrecognized - matches CPU's own
	// fallback default). sigma<=0 (the zero-init default) would make
	// gauss1d(x) = exp(0)-exp(0) = 0 for every x, zeroing every sample's
	// weight - guard it to pbrt-v4's own real default instead.
	if (sigma <= 0.0f) sigma = 0.5f;
	{
		const float expVal = expf(-sigma*sigma*radius*radius);
		auto gauss1d = [sigma, expVal](float x) -> float {
			float v = expf(-sigma*sigma*x*x) - expVal;
			return (v > 0.0f) ? v : 0.0f;
		};
		return gauss1d(ox) * gauss1d(oy);
	}
}

// Film "cropwindow"/"pixelbounds" (pbrt-v4) - device-side gate matching
// CPU's own camera.h `in_crop` skip-the-sampling-loop check. cam.cropX1<=0
// (GpuCameraParams::cropX0's own comment) means no crop was requested -
// every pixel is in bounds, exactly the pre-cropwindow-support behavior.
// Shared by both GPU backends (optix_raygen.h, wavefront_kernels.cu's
// generate_camera_rays), same "shared helper" convention as
// gpu_filter_evaluate() just above.
__device__ __forceinline__ bool gpu_in_crop(const GpuCameraParams& cam, int px, int py) {
	if (cam.cropX1 <= 0) return true;
	return px >= cam.cropX0 && px < cam.cropX1 && py >= cam.cropY0 && py < cam.cropY1;
}

__device__ __forceinline__ float3 random_float3(unsigned int& seed) {
	return make_float3(random_float(seed), random_float(seed), random_float(seed));
}

// Builds two independent 64-bit seeds (each from two chained pcg_hash draws,
// advancing the caller's 32-bit `seed` state) for the layered_detail::PCG32
// RNG that CoatedDiffuseBxDF::f()/CoatedConductorBxDF::f() (src/shared/
// bxdfs_layered.h) run their internal random walk with - used when NEE needs
// a fresh stochastic f() evaluation toward a light direction.
__device__ __forceinline__ void random_seed64_pair(unsigned int& seed, uint64_t& s0, uint64_t& s1) {
	unsigned int a = (seed = pcg_hash(seed));
	unsigned int b = (seed = pcg_hash(seed));
	unsigned int c = (seed = pcg_hash(seed));
	unsigned int d = (seed = pcg_hash(seed));
	s0 = (uint64_t(a) << 32) | uint64_t(b);
	s1 = (uint64_t(c) << 32) | uint64_t(d);
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

// Henyey-Greenstein phase function VALUE p(cos_theta, g) - the same quantity
// as both the phase "BSDF" f and its own pdf (a properly normalized phase
// function is its own perfect importance sampler, exactly like
// src/TheRestOfYourLife/constant_medium.h's hg_phase_pdf::value() ==
// hg_phase_material::scattering_pdf(), both of which just call
// HenyeyGreensteinPhaseFunction<double>::p()). Hand-duplicated as a free
// function rather than calling src/shared/volume_scattering.h's
// HenyeyGreensteinPhaseFunction<T>::p() member function directly, matching
// sample_henyey_greenstein()'s own reason just above (this backend's
// recursive mega-kernel has previously stalled on CPU_GPU-tagged struct
// member-function calls - see this project's own notes on that issue; a
// hand-duplicated free function sidesteps it entirely, the same fix already
// applied there).
//
// Used by MaterialType::DielectricMedium's medium-interior phase-scatter
// case (optix_intersection_sphere.h) to do real NEE+MIS at that scatter
// event - see that call site's own comment for why (closing B13's
// CPU-vs-GPU brightness gap).
__device__ __forceinline__ float hg_phase_value(float cos_theta, float g) {
	float gc = fminf(0.99f, fmaxf(-0.99f, g));
	const float inv4pi = 1.0f / (4.0f * 3.14159265358979323846f);
	float denom = 1.0f + gc * gc + 2.0f * gc * cos_theta;
	return inv4pi * (1.0f - gc * gc) / (denom * sqrtf(fmaxf(1e-12f, denom)));
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

// MaterialType::Principled: Disney/pbrt-v4-style multi-lobe BSDF (src/shared/
// bxdfs_principled.h's PrincipledBxDF<T>) - matches src/TheRestOfYourLife/
// principled_material.h::scatter() exactly (same "instantiate the shared
// CPU_GPU BxDF struct directly" pattern as sample_hair_material() above).
// Returns false if the sample should be rejected (mirrors principled's
// `if (!res.valid) return false;`). Field reuse: albedo=base color, ior=ior,
// fuzz=roughness, eta_c.x=metallic, eta_c.y=clearcoat, eta_c.z=clearcoat_rough
// - see MaterialType::Principled's comment in optix_types.h.
__device__ __forceinline__ bool sample_principled_material(
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
	float u1 = random_float(seed), u2 = random_float(seed), u3 = random_float(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// Device-side real tabulated-measured-BRDF evaluation (MaterialType::
// Measured, both GPU backends - this is the recursive backend's copy) -
// needs `params` (declared above) for the flat table arrays and
// random_float() (defined above) for the Measured case's own
// per-sample randoms, so this include must stay below both.
#include "optix_measured_bxdf.h"

// Device-side real importance-sampled HDR sky (LightSource "infinite" with
// an image) - needs `params` and random_float()/random_unit_vector() (all
// defined above), so this include must stay below them too. See that file's
// own header comment. The actual math (shared with the wavefront backend)
// lives in gpu_sky_light_shared.h, included first - it needs nothing above
// it (no OptiX intrinsics, no `params`), so its own position here is only
// "before optix_sky_light.h, which calls into it" not a hard requirement.
#include "gpu_sky_light_shared.h"
#include "gpu_portal_light_shared.h"
// Bounding-cone light BVH traversal (stochastic descent + bit-trail PMF
// replay) - same shared-explicit-parameter split as the two lines above.
// Needs nothing but LightBVHNode/GpuLightBvhSample (already visible via
// optix_types.h's own light_bvh_node.h include above), so it could sit
// anywhere before optix_device_helpers_lighting.h; grouped here with its
// two siblings instead.
#include "light_bvh_traversal_shared.h"
#include "optix_sky_light.h"

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

// Zero-thickness glass slab reflect-or-straight-through (pbrt-v4
// ThinDielectricBxDF), shared by MaterialType::ThinDielectric (shade_
// material's case below) and MaterialType::DielectricMedium's own entry/
// exit boundary when fused with a thin surface (optix_intersection_
// sphere.h/optix_intersection_disk_cylinder.h, mat.dielectric_medium_
// extra.isThin - see pbrt_gpu_builder.h's mediumMaterialIndex() for how a
// shape's own Material "thindielectric" sets that flag) - same drop-in
// relationship to dielectric_scatter() just above. No bending (unlike
// smooth refraction) and no actual interior path length, so unlike rough
// dielectric this needed no new per-call-site setup (tangent frame,
// texture, NEE) to fuse - it's exactly as cheap here as the smooth case.
// Multiple internal bounces folded analytically: R_eff = R + T^2*R/(1-R^2).
__device__ __forceinline__ float3 thin_dielectric_scatter(const float3& ray_dir, const float3& normal,
		float ior, unsigned int& seed) {
	float3 unit_direction = normalize(ray_dir);
	float cos_theta = fabsf(dot(unit_direction, normal));
	float R = FrDielectric(cos_theta, ior);
	if (R < 1.0f) {
		float T = 1.0f - R;
		R += T * T * R / (1.0f - R * R);
	}
	if (random_float(seed) < R) {
		return reflect(unit_direction, normal);
	} else {
		return unit_direction;  // straight through
	}
}

//==============================================================================
// Multiple Importance Sampling (MIS) Helpers
//==============================================================================

// MIS power heuristic (beta=2) -- delegates to shared PowerHeuristic (pbrt-v4 pattern)
__device__ __forceinline__ float mis_power_heuristic(float pdf_a, float pdf_b) {
	return PowerHeuristic(pdf_a, pdf_b);
}

// Packs shade_material()'s boolean out-params into the single outgoing
// payload flag every closest-hit program sends back via optixSetPayload_10:
// 1 = ordinary scattered bounce, 3 = scattered w/ explicit origin override
// (Subsurface probe exit), 4 = interface pass-through (MaterialType::
// Interface - real medium-boundary, no BSDF), with bit 3 (value 8) OR'd in
// when the bounce was specular (pbrt-v4 specularBounce). Called identically
// from all 6 closest-hit programs across the 5 optix_intersection_*.h files -
// kept as one shared function so a future flag value only needs editing here.
// The is_specular bit rides along in this same register rather than being
// re-derived from brdf_pdf_out==0.0f in optix_raygen.h (see that file's own
// bounce_is_specular comment) - a real boolean, not a proxy that a
// legitimately non-specular but numerically-underflowed-to-zero pdf could
// misclassify. Base values (1/3/4) stay < 8, so masking with `& 7` recovers
// the original flag unchanged for every existing flag==N comparison.
__device__ __forceinline__ unsigned int pack_scatter_flag(bool bssrdf_exit, bool is_medium_boundary, bool is_specular) {
	return (bssrdf_exit ? 3 : (is_medium_boundary ? 4 : 1)) | (is_specular ? 8 : 0);
}

// The path's last real vertex (optix_raygen.h's mis_origin, payload p25-p27) - where the BSDF sample that reached an
// emitter was taken. Not the ray origin: a free medium-boundary crossing moves that. An emitter's MIS pdf must be
// evaluated from here to match the NEE sample taken at the vertex (CPU's prev_surface_p).
__device__ __forceinline__ float3 mis_origin_from_payload() {
	return make_float3(__uint_as_float(optixGetPayload_25()), __uint_as_float(optixGetPayload_26()),
					   __uint_as_float(optixGetPayload_27()));
}

// Integrator "bool regularize" gate for THIS bounce, read fresh by each
// closest-hit program right before its shade_material() call - same
// "called identically from all closest-hit programs across the 5
// optix_intersection_*.h files, kept as one shared function" shape as
// pack_scatter_flag() just above. params.camera.regularize is a per-launch
// constant; optixGetPayload_23() carries anyNonSpecularBounces-so-far in
// from optix_raygen.h's bounce loop (an INPUT register, never written by
// any closest-hit program - see that file's own p23 comment).
__device__ __forceinline__ bool current_do_regularize() {
	return params.camera.regularize != 0 && optixGetPayload_23() != 0u;
}

// Cosine-weighted hemisphere sampling PDF
__device__ __forceinline__ float cosine_pdf(const float3& direction, const float3& normal) {
	float cosine = dot(normalize(direction), normal);
	return fmaxf(0.0f, cosine / 3.14159265358979323846f);
}

// Light sampling, NEE, and medium/shadow helpers - see that file's own
// header comment for why it's split out.
#include "optix_device_helpers_lighting.h"
