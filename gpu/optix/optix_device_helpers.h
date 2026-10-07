// optix_device_helpers.h -- Shared device utilities for OptiX programs
// Included by optix_programs.cu

// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"    // Shared exact Fresnel (CPU+GPU)
#include "../../src/shared/volume_scattering.h" // HomogeneousMediumData/sample_homogeneous_event (CPU+GPU) - per-channel media
#include "../../src/shared/ray_hash.h"  // pbrt-v4 stochastic alpha test hash (CPU+GPU)
#include "../../src/shared/microfacet.h" // GGX TrowbridgeReitz (CPU+GPU)
#include "../../src/shared/bxdfs.h"      // HairBxDF<T> (CPU+GPU) - see MaterialType::Hair
#include "../../src/shared/noise.h"      // Perlin turbulence (CPU+GPU) - see sample_texture()
#include "../../src/shared/normal_map.h" // apply_normal_map (CPU+GPU) - see MaterialType::NormalMappedLambertian
#include "gpu_bump_map.h"               // grayscale "texture displacement" bump - see optix_intersection_triangle.h
#include "../../src/shared/bilinear_patch.h" // blp_sample/blp_pdf_wi (CPU+GPU) - see GpuLightKind::BilinearPatch
#include "../../src/shared/shading_frame.h"  // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured
#include "camera_motion_blur_device.h"        // gpu_camera_anim_rotation/apply (shared with wavefront_kernels.cu)

// Launch parameters (constant across all threads)
extern "C" { __constant__ LaunchParams params; }

// Device-side tabulated-BSSRDF evaluation (MaterialType::Subsurface,
// recursive backend only) - needs `params` above for the flat table arrays,
// so this include must stay below it.
#include "optix_bssrdf.h"

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
	// Shared by the Image case below and by UVChecker/Mix's own
	// tex1ImageIdx/tex2ImageIdx (a one-level-nested bare imagemap bound to
	// tex1/tex2 instead of a flat literal - see TextureData's own comment)
	// - factored out so both call sites share one copy of the pixel-lookup
	// math instead of duplicating it. wide-clamp uv to [-1024,1024] (a
	// safety rail against a pathological UV, NOT [0,1] - see wide_clamp()'s
	// own comment), flip v (stored image rows are top-to-bottom, v=0 is the
	// bottom of the [0,1] texture-coordinate convention), then bilinearly
	// interpolate the 4 texels around (u,v), each wrapped independently per
	// t.wrapMode (GpuWrapMode's own comment) - Repeat/Black/Clamp. A failed
	// image load (width/height <= 0) matches CPU's own solid-cyan debugging
	// fallback (texture.h) exactly.
	//
	// BILINEAR, not nearest-neighbor: this now matches CPU's own
	// mipmap_texture::value() (texture.h) exactly - its "zero derivatives"
	// path (used whenever no real screen-space UV footprint is available,
	// e.g. every non-primary-ray hit) is NOT a nearest-neighbor lookup, it's
	// MIPMap::filter()'s own LOD-0 `bilerp()` (mipmap.h) - a real 4-tap
	// bilinear blend with texel CENTERS at (i+0.5)/width, hence the `-0.5f`
	// before floor() below (mirrors bilerp()'s own `s*w - 0.5`). A prior
	// version of this function did pure nearest-neighbor and claimed (in
	// this now-corrected comment) that this matched CPU's value() - it did
	// not; CPU has never been nearest-neighbor here, even without ray
	// differentials. Confirmed as a real, non-noise, texture-filtering
	// discrepancy (not just the already-known MaterialType::
	// DiffuseTransmission NEE-strategy noise) by a CPU/GPU parity sweep's
	// regional-diff check on scene J2 (a 4x4-pixel texture, where
	// nearest-vs-bilinear snapping is maximally visible) - see tests/
	// integration/material_cpu_gpu_parity_tests.cpp's own J2 comment. This
	// still does NOT implement full EWA/mipmap minification filtering (no
	// GPU mip pyramid exists - TextureData holds one resolution level only,
	// a real, larger, and still-open architectural gap for a texture
	// MINIFIED below screen resolution) - bilinear only fixes MAGNIFICATION
	// (texture resolution below screen resolution, J2's exact case and the
	// common one for small/debug textures), which needs no mip chain at all.
	auto sampleImage = [&](const TextureData& t) -> float3 {
		if (t.width <= 0 || t.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uw = fminf(fmaxf(u, -1024.0f), 1024.0f);
		const float vw = fminf(fmaxf(1.0f - v, -1024.0f), 1024.0f);
		// double, not float, for the multiply - matches CPU's own bilerp()
		// (src/shared/mipmap.h), which promotes to double for this exact
		// computation: a wide-tiled UV (now possible here too, via the
		// +-1024 clamp above) times a large texture width can exceed
		// float's exact-integer range (2^24), losing sub-texel precision
		// right before floor() picks the pixel index.
		const double x = (double)uw * t.width - 0.5;
		const double y = (double)vw * t.height - 0.5;
		const int x0 = static_cast<int>(floor(x)), x1 = x0 + 1;
		const int y0 = static_cast<int>(floor(y)), y1 = y0 + 1;
		const float fx = (float)(x - x0), fy = (float)(y - y0);

		// One corner's wrapped texel fetch - No `default:` case in the
		// switch, deliberately - this is what lets the compiler
		// (-Wswitch/MSVC's equivalent) flag a future 4th GpuWrapMode
		// enumerator left unhandled here, in BOTH this copy and
		// wavefront_device_helpers.h's own duplicate, instead of both
		// silently falling through to Clamp behavior with no diagnostic
		// anywhere.
		auto wrapTexel = [&](int xi, int yi) -> float3 {
			switch (t.wrapMode) {
			case GpuWrapMode::Repeat:
				xi = ((xi % t.width) + t.width) % t.width;
				yi = ((yi % t.height) + t.height) % t.height;
				break;
			case GpuWrapMode::Black:
				if (xi < 0 || xi >= t.width || yi < 0 || yi >= t.height) return make_float3(0.0f, 0.0f, 0.0f);
				break;
			case GpuWrapMode::Clamp:
				xi = min(max(xi, 0), t.width - 1);
				yi = min(max(yi, 0), t.height - 1);
				break;
			}
			const unsigned char* px = params.texturePixels + t.pixelOffset + (yi * t.width + xi) * 3;
			return texel_rgb(px, t.srgb);
		};

		const float3 c00 = wrapTexel(x0, y0);
		const float3 c10 = wrapTexel(x1, y0);
		const float3 c01 = wrapTexel(x0, y1);
		const float3 c11 = wrapTexel(x1, y1);
		return (1.0f - fy) * ((1.0f - fx) * c00 + fx * c10)
		     +         fy  * ((1.0f - fx) * c01 + fx * c11);
	};
	if (tex.kind == TextureKind::Image) {
		return sampleImage(tex);
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
	} else if (tex.kind == TextureKind::UVChecker) {
		// Matches uv_checker_texture::value() (texture.h) exactly: parity of
		// floor(u*uscale)+floor(v*vscale) - pbrt-v4's own UV-tiled
		// checkerboard convention, deliberately NOT the world-space Checker
		// case above (see TextureKind::UVChecker's own comment). tex1ImageIdx/
		// tex2ImageIdx (-1 by default) let either cell colour instead be a
		// one-level-nested bare imagemap - see TextureData's own comment.
		const int ui = static_cast<int>(floorf(u * tex.uScale));
		const int vi = static_cast<int>(floorf(v * tex.vScale));
		const bool is_even = ((ui + vi) % 2) == 0;
		// Only the winning cell's slot is sampled - is_even already fully
		// determines which of tex1/tex2 is used, so evaluating BOTH
		// (including a global-memory image fetch when either is nested)
		// would be pure waste on this per-ray hot path.
		return is_even
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::FBm) {
		// Matches fbm_texture::value() (texture.h) exactly: fbm_simple(p,
		// omega, octaves) mapped from [-~1,~1] to a clamped [0,1] greyscale.
		const float v = fbm_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
		float t = 0.5f + 0.5f * v;
		t = fminf(fmaxf(t, 0.0f), 1.0f);
		return make_float3(t, t, t);
	} else if (tex.kind == TextureKind::Marble) {
		return sample_marble_texture(tex, p);
	} else if (tex.kind == TextureKind::Windy) {
		return sample_windy_texture(p);
	} else if (tex.kind == TextureKind::Wrinkled) {
		return sample_wrinkled_texture(tex, p);
	} else if (tex.kind == TextureKind::Dots) {
		// Same one-level-nested-imagemap convention as UVChecker/Mix above -
		// tex1ImageIdx/tex2ImageIdx (-1 by default) let inside/outside each
		// independently be a bare imagemap instead of the flat color1/color2.
		const bool inside = is_inside_dot(u, v);
		return inside
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::Bilerp) {
		return sample_bilerp_texture(tex, u, v);
	} else if (tex.kind == TextureKind::Mix) {
		// Matches mix_texture::value() (texture.h) exactly: lerp, no
		// footprint/UV dependence. tex1ImageIdx/tex2ImageIdx - see
		// UVChecker's own identical comment just above. amountImageIdx (-1
		// by default) lets "amount" itself be a one-level-nested bare
		// imagemap instead of the flat tex.mixAmount scalar - only its .x
		// channel is read, matching mix_texture's own amount_tex convention.
		const float3 c1 = (tex.tex1ImageIdx >= 0) ? sampleImage(params.textures[tex.tex1ImageIdx]) : tex.color1;
		const float3 c2 = (tex.tex2ImageIdx >= 0) ? sampleImage(params.textures[tex.tex2ImageIdx]) : tex.color2;
		const float amt = (tex.amountImageIdx >= 0) ? sampleImage(params.textures[tex.amountImageIdx]).x : tex.mixAmount;
		return (1.0f - amt) * c1 + amt * c2;
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

// Emission the caller's local `emission` variable should be initialized to
// BEFORE calling shade_material() - see shade_material()'s own comment on
// that in/out convention (NEE contributions get added on top of whatever
// this returns). Guards the mat.emission union-slot read behind
// mat.type == DiffuseLight, so a material that reuses that slot for
// something else entirely (DiffuseTransmission's transmittance,
// Subsurface's sigma_a, ...) never depends on shade_material()'s own switch
// remembering to reset it back to zero - previously TWO separate cases each
// carried a near-identical "don't forget to zero this" comment doing that
// reset by hand; a third material reusing this slot in the future would
// silently need a fourth. One-sided per front_face UNLESS mat.twoSided
// (matches CPU's diffuse_light::emitted()/is_two_sided()). textureIdx>=0
// samples a real texture at the given UV - map_Ke on triangles (see
// add_diffuse_light()'s textureIdx comment) or a pbrt AreaLightSource
// "filename" image (any geometry the caller passes real UV for) - scaled by
// emissionScale (a no-op 1.0 multiply when the light wasn't built with a
// "scale" param). Any future geometry type gets both for free by going
// through this accessor instead of reading mat.emission raw.
__device__ __forceinline__ float3 material_emission(
		const MaterialData& mat, bool front_face,
		float uv_u = 0.0f, float uv_v = 0.0f,
		float3 hit_point = make_float3(0.0f, 0.0f, 0.0f)) {
	if (mat.type != MaterialType::DiffuseLight || (!mat.twoSided && !front_face))
		return make_float3(0.0f, 0.0f, 0.0f);
	if (mat.textureIdx >= 0) {
		const float3 texel = sample_texture(mat.textureIdx, uv_u, uv_v, hit_point);
		return make_float3(texel.x * mat.emissionScale, texel.y * mat.emissionScale, texel.z * mat.emissionScale);
	}
	return mat.emission;
}

// Alpha-cutout test (OBJ/.mtl map_d): returns true if the hit should be
// kept, false if it should be treated as fully transparent - a caller at an
// any-hit call site (radiance or shadow) should then call
// optixIgnoreIntersection() so the ray continues past this point as if the
// geometry weren't there. A no-op (always true) for alphaMaskTexIdx < 0,
// i.e. the overwhelming majority of materials. pbrt-v4's stochastic alpha test
// (cpu/primitive.cpp:57-71): a hit is kept with probability `alpha`, decided by a
// hash of the ray itself (ray_hash.h), so shadow rays and radiance rays make
// independent, state-free choices and a partial-alpha material is partly
// transparent instead of snapping to a threshold. Must be called from an any-hit
// program: it reads the current ray.
__device__ __forceinline__ bool passes_alpha_cutout(int alphaMaskTexIdx, float u, float v, const float3& p) {
	if (alphaMaskTexIdx < 0) return true;
	const float a = sample_texture(alphaMaskTexIdx, u, v, p).x;
	const float3 o = optixGetWorldRayOrigin();
	const float3 d = optixGetWorldRayDirection();
	return ray_hash::alphaPasses(a, o.x, o.y, o.z, d.x, d.y, d.z);
}

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
			if (random_float(seed) < p_refl) {
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
			// Not a delta BSDF, but flagged like one: no light sampling is done at this vertex (it has no hemisphere-aware NEE here) so the next
			// emitter hit must be added in full, not MIS-weighted against a light sample that never happened - same as the wavefront backend.
			is_specular = true;
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

// True for the 7 MaterialTypes that need sphere/box-specific handling -
// Medium/DielectricMedium/CloudMedium/RgbGridMedium/GridMedium (volume
// ray-marching against a sphere's own two intersection roots, or a box's
// slab test - see optix_intersection_sphere.h's shape-specific branches)
// and Hair/Principled (sampled directly via a sibling dispatch alongside
// shade_material(), never through it - see shade_material()'s own default:
// comment above). NormalMappedLambertian is deliberately NOT included here:
// unlike these 6, it has a real, working implementation on triangles too
// (optix_intersection_triangle.h's own tangent-frame branch), just not on
// quads or bilinear patches - callers decide separately whether to also
// reject that type.
//
// A primitive type that does not implement one of these 6 has no
// physically-sensible fallback to construct (there is no "inside" to a
// flat quad/triangle/bilinear-patch to ray-march a medium through), so
// scene authors assigning one of these types to an unsupported primitive
// is a real authoring bug, not a recoverable render state. Callers should
// trap on this BEFORE ever reaching shade_material(), with a message
// identifying the actual type/primitive mismatch - shade_material()'s own
// default: trap would still catch it, but with a generic "unhandled
// MaterialType" message that doesn't reveal it was actually a valid type
// used on the wrong geometry.
// Device-side port of CPU's branch_hash01() (src/TheRestOfYourLife/
// material_pbrt.h) - deterministic [0,1) value from a world-space point, NOT
// a fresh random_float()/seed draw. Mix resolution needs this determinism:
// a single scattering event's radiance closest-hit, its shadow any-hit, and
// (on re-intersection) any later ray touching the same point must all agree
// on which sub-material a Mix resolved to - a per-call RNG draw would let
// them silently disagree, exactly the bug branch_hash01's own CPU comment
// describes for scatter()/scattering_pdf()/is_shadow_transmissive().
__device__ __forceinline__ float mix_branch_hash01(const float3& p) {
	float h = sinf(p.x * 127.1f + p.y * 311.7f + p.z * 74.7f) * 43758.5453f;
	return h - floorf(h);
}

// Resolves a (possibly Mix) MaterialData to a real, non-Mix MaterialData by
// hashing `hit_point` and iteratively picking sub-material A or B -
// LOOPING, not recursing (a Mix's own sub-material can itself be another
// Mix - pbrt-v4 allows mix-of-mix, matches pbrt_gpu_builder.h's build-time
// resolveMixColor()/makeMaterial() recursion), so this never adds to
// shade_material()'s call graph. Must run before ANY other mat.type branch
// in a shape's closest-hit/intersection program or shadow any-hit program
// (see MaterialType::Mix's own comment) - in particular before
// material_requires_sphere_only_handling() below, so a Mix resolving to a
// sphere-only type on the wrong geometry still traps correctly, and before
// shade_material() itself, since Hair/Subsurface/Medium-family are
// dispatched via sibling paths this function never touches.
//
// `outMatIdx` receives the resolved material's own index into
// params.materials (updated even when `mat` was never Mix, so callers can
// use it unconditionally) - needed by callers that separately look up
// per-material auxiliary data by index (e.g. Subsurface's bssrdfTables,
// Measured's measuredTables) after resolution.
__device__ __forceinline__ MaterialData resolve_mix_material(MaterialData mat, int matIdx,
															  const float3& hit_point, int& outMatIdx) {
	constexpr int kMaxMixDepth = 8;
	for (int depth = 0; mat.type == MaterialType::Mix && depth < kMaxMixDepth; ++depth) {
		const float w = mat.mix_extra.mixWeight;
		const float h = mix_branch_hash01(hit_point);
		const int subIdx = (h >= w) ? static_cast<int>(mat.mix_extra.mixMaterialAIdx)
									 : static_cast<int>(mat.mix_extra.mixMaterialBIdx);
		matIdx = subIdx;
		mat = params.materials[subIdx];
	}
	outMatIdx = matIdx;
	return mat;
}

__device__ __forceinline__ bool material_requires_sphere_only_handling(MaterialType type) {
	switch (type) {
		case MaterialType::Medium:
		case MaterialType::DielectricMedium:
		case MaterialType::CloudMedium:
		case MaterialType::RgbGridMedium:
		case MaterialType::GridMedium:
		case MaterialType::Hair:
		case MaterialType::Principled:
			return true;
		default:
			return false;
	}
}

// Whether shade_material()'s dpdu parameter is ever actually read for this
// material type - NormalMappedLambertian (its normal-map tangent basis), the
// 4 anisotropy-capable kinds (their UV-aligned shading frame, see
// BuildDpduTangentFrame's own comment, microfacet.h), and DielectricMedium
// (rough_dielectric_scatter_and_nee()'s own tangent frame, needed whenever
// THIS material's own surfaceKind turns out to be rough - checked per-hit
// inside that function, not here, since this dispatch only has the
// MaterialType to go on, not the fused sub-flavor; a smooth/thin
// DielectricMedium pays a small, unconditional dpdu-computation cost it
// doesn't actually use, the same trade-off material_requires_sphere_only_
// handling()'s own switch-based dispatch style already accepts elsewhere
// for simplicity over per-instance precision). Every OTHER material kind
// (Lambertian, Metal, Dielectric, DiffuseLight, RoughMetal, the rest of the
// Medium family, etc.) never touches dpdu at all, so each intersection
// file's own dpdu computation is gated on this - rather than paying for
// trig/solve/transform work on every hit regardless of whether the
// material can ever use the result.
__device__ __forceinline__ bool material_needs_dpdu(MaterialType type) {
	switch (type) {
		case MaterialType::NormalMappedLambertian:
		case MaterialType::Conductor:
		case MaterialType::RoughDielectric:
		case MaterialType::CoatedDiffuse:
		case MaterialType::CoatedConductor:
		case MaterialType::DielectricMedium:
			return true;
		default:
			return false;
	}
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

	if (cam.animated) {
		// Real per-ray shutter-time camera motion blur - mirrors
		// src/TheRestOfYourLife/camera.h's camera_is_animated branch of
		// get_ray(): build a LOCAL-space ray exactly as the static
		// Perspective case below does, then place it in world space via a
		// per-sample-time-interpolated camera-to-world transform instead
		// of a single static basis. Perspective-only, like CPU's own
		// camera_is_animated (mutually exclusive with an alt camera model
		// there too) - scene_builder.cpp never sets `animated` alongside a
		// non-Perspective `kind`.
		float3 local_pixel_sample = cam.localLowerLeftCorner + u * cam.localHorizontal + v * cam.localVertical;
		float3 local_origin = make_float3(0.0f, 0.0f, 0.0f);
		bool hasDOF = (cam.localDefocusDiskU.x != 0.0f || cam.localDefocusDiskU.y != 0.0f || cam.localDefocusDiskU.z != 0.0f ||
					   cam.localDefocusDiskV.x != 0.0f || cam.localDefocusDiskV.y != 0.0f || cam.localDefocusDiskV.z != 0.0f);
		if (hasDOF) {
			float3 p = random_in_unit_disk(seed);
			local_origin = p.x * cam.localDefocusDiskU + p.y * cam.localDefocusDiskV;
		}
		float3 local_direction = local_pixel_sample - local_origin;
		// Shutter-time interpolation fraction, uniform on [0,1]:
		// AnimatedTransform::Interpolate always normalizes by (endTime -
		// startTime) before use, so for a time drawn uniformly across the
		// shutter window, dt is uniform on [0,1] regardless of the
		// window's actual numeric bounds - GpuCameraParams doesn't need to
		// store shutterOpen/Close at all.
		float dt = random_float(seed);
		// Slerp + quaternion-to-matrix built ONCE, then applied to both
		// origin and direction (gpu_camera_anim_apply, camera_motion_blur_
		// device.h) - the matrix is identical for both, only the vector and
		// isPoint differ.
		GpuAnimRotMat rot = gpu_camera_anim_rotation(cam.animR0, cam.animR1, dt);
		origin = gpu_camera_anim_apply(rot, local_origin, true, cam.animT0, cam.animT1, dt);
		// local_direction is not unit length (it's pixel_sample - lens
		// origin, same as the static Perspective case just below) and the
		// rotation preserves length, so this needs an explicit normalize -
		// every other camera branch in this function returns a unit
		// direction too.
		direction = normalize(gpu_camera_anim_apply(rot, local_direction, false, cam.animT0, cam.animT1, dt));
		return;
	}

	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			// pbrt-v4 SphericalCamera::GenerateRay - see src/shared/cameras.h
			// for the reference this mirrors, both mappings finish with a
			// swap(dir.y, dir.z) folded directly into which raw component
			// feeds ly vs lz below (rather than an actual runtime swap).
			// `v` here already carries the Y-flip optix_raygen.h applies for
			// every CameraKind (v=1 at the top framebuffer row), matching
			// Perspective/Orthographic's `vertical` basis vector, which
			// points world-up - so v=1 (top row) adds the full +up vector,
			// correctly landing "up" at the top of the frame for those two.
			// CPU's SphericalCamera::generate_ray, by contrast, uses raw
			// pFilm_y/res_y with NO such flip (v=0 at its own top row) - its
			// theta=v*pi formula relies on THAT convention to put dir.y=+1
			// (up) at the top row. Feeding it this shared, already-flipped
			// `v` directly would put "down" at the top of the frame instead
			// - undo the flip locally so both mappings match CPU exactly.
			const float v_sph = 1.0f - v;
			float lx, ly, lz;
			if (cam.sphericalMapping == 1) {  // EqualArea
				double ud = (double)u, vd = (double)v_sph;
				dev_wrap_equal_area_square(ud, vd);
				double ewx, ewy, ewz;
				dev_equal_area_square_to_sphere(ud, vd, ewx, ewy, ewz);
				lx = (float)ewx;
				ly = (float)ewz;  // swap(wy,wz): final y = raw z
				lz = (float)ewy;  // swap(wy,wz): final z = raw y
			} else {  // EquiRectangular: theta in [0,pi], phi in [0,2pi]
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

// Denoiser guide-layer AOV payload packing (albedo/normal, p16-p21) - see
// PathTracingPayload::albedo/normal's own comment (optix_types.h) and
// raygen's own comment for why every closest-hit/miss program packs these
// in every branch (scattered/DiffuseLight-hit/absorbed/miss) - raygen is
// the one that decides whether to accumulate them, not the hit/miss
// program. Shared by all 4 closest-hit programs (sphere/quad/triangle/
// bilinear-patch) and the miss program instead of each hand-rolling the
// same 6 optixSetPayload_* calls. Gated on params.albedoBuffer (set only
// when this render will actually be denoised - see OptiXRenderer::render()'s
// own alloc site) so the common non-denoised path doesn't pay 6 extra
// register writes on every ray for a feature it isn't using; raygen's own
// accumulation is gated the same way (see its depth==0 check), and the
// payload registers left unwritten here are never read when albedoBuffer
// is null, since raygen only accumulates/writes through that same guard.
__device__ __forceinline__ void pack_aov_payload(float3 albedo, float3 normal) {
	if (!params.albedoBuffer) return;
	optixSetPayload_16(__float_as_uint(albedo.x));
	optixSetPayload_17(__float_as_uint(albedo.y));
	optixSetPayload_18(__float_as_uint(albedo.z));
	optixSetPayload_19(__float_as_uint(normal.x));
	optixSetPayload_20(__float_as_uint(normal.y));
	optixSetPayload_21(__float_as_uint(normal.z));
}

//==============================================================================
// Sphere Intersection Program
//==============================================================================

