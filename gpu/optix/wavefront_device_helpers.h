#pragma once
// wavefront_device_helpers.h -- shared __device__ helper functions for the
// wavefront GPU path tracer's CUDA kernels, split out of wavefront_kernels.cu
// (which had grown to 4707 lines, 1451 of them - the RNG, texture, light-
// sampling, and camera-ray helpers below - shared by every kernel in that
// file). All __device__ __forceinline__, so including this header into
// multiple separately-compiled .cu translation units is the same one-
// definition-rule-safe pattern as any other inline C++ header; no new
// build_optix.targets/vcxproj wiring is needed for a header (unlike adding
// a new .cu file, which needs updating the WavefrontObjSource item group,
// the DeviceLinkWavefront dlink inputs, both configs' AdditionalDependencies,
// and the CleanOptixPrograms cleanup list) - build_optix.targets' own
// CudaHeaderDeps already globs every *.h in this directory, so editing this
// file alone correctly invalidates and recompiles every kernel .cu that
// includes it.

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
#include "camera_motion_blur_device.h"  // gpu_camera_anim_rotation/apply (shared with optix_device_helpers.h)
#include "bxdfs.h"  // HairBxDF<T>/PrincipledBxDF<T> - see MaterialType::Hair/Principled
#include "../../src/shared/noise.h"       // turbulence_simple - see wf_sample_texture()
#include "../../src/shared/normal_map.h"  // apply_normal_map - see MaterialType::NormalMappedLambertian
#include "../../src/shared/bilinear_patch.h"  // blp_sample - see wf_sample_bilinear_patch_light()
#include "../../src/shared/shading_frame.h"   // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured
#include "../../src/shared/sampling_helpers.h"  // SampleUniformDiskConcentric - see wf_sample_disk_light()

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

// Pixel reconstruction filter weight - matches optix_device_helpers.h's
// identical gpu_filter_evaluate() exactly (own definition here since this
// is a separate translation unit, same "duplicate rather than share across
// a .cpp/.h boundary" convention gpu_cloud_density above already uses).
// See that copy's own comment (or src/shared/filter.h's PixelFilterDispatch,
// the CPU reference both port) for the full explanation.
__device__ __forceinline__ float gpu_filter_evaluate(
		int kind, float B, float C, float sigma, float tau, float ox, float oy) {
	const float radius = 0.5f;
	if (kind == 1) return 1.0f;  // box
	if (kind == 2) {  // triangle (tent)
		float tx = fmaxf(0.0f, radius - fabsf(ox));
		float ty = fmaxf(0.0f, radius - fabsf(oy));
		return tx * ty;
	}
	if (kind == 3) {  // mitchell
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
	                              // (see optix_types.h's filterKind comment) -
	                              // guard the zero-init value here instead.
	if (kind == 4) {  // sinc
		return WindowedSinc(ox, radius, tau) * WindowedSinc(oy, radius, tau);
	}
	// gaussian (kind == 0, or anything unrecognized). sigma<=0 (zero-init
	// default) would zero every sample's weight - guard to pbrt-v4's own
	// real default instead.
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

// Film "cropwindow"/"pixelbounds" (pbrt-v4) - matches optix_device_helpers.h's
// identical gpu_in_crop() exactly (own definition here, same "duplicate
// rather than share across a .cpp/.h boundary" convention gpu_filter_evaluate
// just above already uses). cam.cropX1<=0 (GpuCameraParams::cropX0's own
// comment) means no crop was requested - every pixel is in bounds.
__device__ __forceinline__ bool gpu_in_crop(const GpuCameraParams& cam, int px, int py) {
	if (cam.cropX1 <= 0) return true;
	return px >= cam.cropX0 && px < cam.cropX1 && py >= cam.cropY0 && py < cam.cropY1;
}

// Denoiser guide-layer AOV accumulation - primary-ray hits only (depth==0
// checked by the caller), matching optix_intersection_sphere.h's own
// "mat.albedo is the safe always-valid fallback" choice: unlike the
// recursive backend, evaluate_materials()/_simple()/_dielectric() don't have
// easy access to the eventual scatter attenuation at this point (computed
// later, per material case, in spectral form) - mat.albedo is a plain RGB
// field available uniformly for every material kind, a simplification that
// trades a little guide-layer precision for not needing a spectral-to-RGB
// conversion of the scatter weight here. Shared by all three hit-evaluation
// kernels (identical accumulation, only the caller's queue/material type
// differs) - accumulate_miss() below has its own AOV write instead, since a
// miss has no MaterialData/HitWorkItem::normal to read from.
__device__ __forceinline__ void wf_accumulate_aov(float3* albedoBuffer, float3* normalBuffer,
													int pixelIndex, float3 albedo, float3 normal) {
	atomicAdd(&albedoBuffer[pixelIndex].x, albedo.x);
	atomicAdd(&albedoBuffer[pixelIndex].y, albedo.y);
	atomicAdd(&albedoBuffer[pixelIndex].z, albedo.z);
	atomicAdd(&normalBuffer[pixelIndex].x, normal.x);
	atomicAdd(&normalBuffer[pixelIndex].y, normal.y);
	atomicAdd(&normalBuffer[pixelIndex].z, normal.z);
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

// GGX alpha for one of the 5 pbrt-v4 rough/layered material types (Conductor/
// RoughMetal/RoughDielectric/CoatedDiffuse/CoatedConductor - the same 5
// classes material_pbrt.h gates behind do_regularize), optionally widened by
// path regularization: once a path has taken a non-specular bounce, widen
// every subsequent GGX lobe it hits to cut fireflies from near-specular-but-
// not-quite surfaces catching a hard-to-sample light path late in a bounce
// chain. RegularizeAlpha() (src/shared/microfacet.h) is the single CPU_GPU
// source of truth for the widening formula, shared with TrowbridgeReitz::
// Regularize() and material_simple.h's CPU-side regularize_alpha() - this is
// just the "derive alpha from mat.fuzz, then conditionally widen" pairing
// every one of this file's glossy call sites needs, hoisted into one place
// (mirrors the project's own prior "Hoist repeated isGlossyType check"
// precedent in this same function) so the BSDF-sampling side (each glossy
// switch-arm below) and the NEE side (wf_finish_material_scatter's
// glossy_alpha, which callers now pass in rather than re-deriving) always
// agree - unlike CPU, which only ever regularizes the scatter() side
// (scattering_pdf() takes no do_regularize parameter, see that class's own
// comment), disagreeing here would desync the BSDF-sampled continuation's
// true pdf from what evalGlossyF/brdf_pdf_override reports for it, a real
// MIS-weight mismatch.
__device__ __forceinline__ float wf_glossy_alpha(const MaterialData& mat, bool do_regularize) {
	// mat.remapRoughness (pbrt-v4 "remaproughness", default true): when
	// false, mat.fuzz already IS the GGX alpha (see MaterialData::
	// remapRoughness's own comment) - skipping sqrtf() here matches
	// material_pbrt.h's CPU-side roughness_or_alpha().
	float a = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;
	// do_regularize is the caller's already-AND'd "Integrator bool
	// regularize (a per-launch constant) && h.any_nonspecular (per-path
	// history)" - this file has no accessible __constant__ params global
	// (a deliberate wavefront-architecture choice, unlike the recursive
	// backend - see evaluate_materials()'s own `regularize` parameter for
	// where the per-launch half of this AND comes from), so the gating
	// happens at each call site instead of in here.
	return do_regularize ? RegularizeAlpha(a) : a;
}

// Second (v/bitangent-axis) GGX alpha, mirroring wf_glossy_alpha() above but
// via the shared ResolveAnisotropicAlphaV() (microfacet.h) - see
// MaterialData::roughnessV's own comment (optix_types.h) for the
// <0-means-isotropic sentinel and which 4 material kinds
// (RoughDielectric/Conductor/CoatedDiffuse/CoatedConductor) call this;
// RoughMetal keeps calling wf_glossy_alpha() alone for both axes, same as
// CPU's rough_metal has no anisotropic variant either.
__device__ __forceinline__ float wf_glossy_alpha_v(const MaterialData& mat, bool do_regularize) {
	float a = ResolveAnisotropicAlphaV(mat.roughnessV, mat.fuzz, mat.remapRoughness);
	// See wf_glossy_alpha()'s own comment just above - do_regularize is
	// already the caller's "regularize && any_nonspecular" AND.
	return do_regularize ? RegularizeAlpha(a) : a;
}

// Duplicated from optix_device_helpers.h's material_requires_sphere_only_
// handling() (with the wf_ prefix), matching this file's existing pattern of
// not sharing device helpers with the recursive path. See that function's
// own comment for why these types have no physically-sensible fallback on a
// flat/curved-but-non-spherical primitive.
__device__ __forceinline__ bool wf_material_requires_sphere_only_handling(MaterialType type) {
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

// Whether `type` has real shading support on disk/cylinder geometry
// (evaluate_materials()'s geomType 4/5) in this shared kernel - the
// positive inverse of "should trap", kept as one named predicate instead
// of letting evaluate_materials()'s own trap condition accumulate a new
// `&& !exemptionN` term for every (shape, material) combination that goes
// from trapped to real support (Hair first, now Medium-on-cylinder) - see
// wf_material_requires_sphere_only_handling()'s own switch list above for
// why each of these types is trapped by DEFAULT; this function is where
// that default gets overridden per shape, in one place, rather than at
// each call site.
__device__ __forceinline__ bool wf_material_supported_on_disk_cylinder_geom(MaterialType type, int geomType) {
	if (type == MaterialType::Hair) return true;           // real support on disk (4) and cylinder (5) alike
	if (type == MaterialType::Medium) return geomType == 5; // real support on cylinder only - see pbrt_gpu_builder.h's cylinder loop
	return !wf_material_requires_sphere_only_handling(type) && type != MaterialType::NormalMappedLambertian;
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

// Henyey-Greenstein phase-function scatter setup, shared by every
// medium-interior scatter case in evaluate_materials()'s own switch
// (Medium/CloudMedium/RgbGridMedium/GridMedium/DielectricMedium) - this is
// intra-file reuse within one already-compiled module, NOT the cross-module
// wavefront/recursive duplication convention every other wf_ helper in this
// file follows for a real compilation-boundary reason. Samples the outgoing
// scatter direction and sets phaseWo/phaseG/brdf_pdf_override for
// wf_finish_material_scatter()'s own isPhase-gated NEE block to consume
// afterward. Callers still set `attenuation`/`is_specular=false` themselves
// - those differ per medium type (flat albedo vs. per-voxel RGB) and aren't
// part of what every call site shares.
__device__ __forceinline__ float3 wf_sample_phase_scatter(
	const float3& unit_dir, float g, unsigned int& seed,
	float3& phaseWo, float& phaseG, float& brdf_pdf_override)
{
	// wf_sample_henyey_greenstein's own `wo` parameter wants the OUTGOING
	// direction (toward where the ray came from), not the forward travel
	// direction - see this function's own callers for the history of the
	// bug this negation fixes.
	phaseWo = -unit_dir;
	phaseG  = g;
	const float3 scattered_dir = wf_sample_henyey_greenstein(phaseWo, phaseG, seed);
	brdf_pdf_override = wf_hg_phase_value(dot(phaseWo, scattered_dir), phaseG);
	return scattered_dir;
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

// De Casteljau cubic Bezier evaluation - duplicated from optix_device_
// helpers.h's marble_cubic_bezier4 (with the wf_ prefix), matching this
// file's existing pattern of not sharing device helpers with the recursive
// path.
__device__ __forceinline__ float3 wf_marble_cubic_bezier4(
		const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t) {
	const float s = 1.0f - t;
	const float w0 = s*s*s, w1 = 3.0f*s*s*t, w2 = 3.0f*s*t*t, w3 = t*t*t;
	return make_float3(
		w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x,
		w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y,
		w0*p0.z + w1*p1.z + w2*p2.z + w3*p3.z);
}

// Matches marble_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_marble_texture (with the wf_ prefix),
// same no-shared-device-helpers convention as wf_marble_cubic_bezier4 above.
__device__ __forceinline__ float3 wf_sample_marble_texture(const TextureData& tex, const float3& p) {
	const float px = p.x * tex.marbleScale, py = p.y * tex.marbleScale, pz = p.z * tex.marbleScale;
	const float fbm_val = fbm_simple<float>(px, py, pz, tex.omega, tex.octaves);
	const float marble = py + tex.marbleVariation * fbm_val;
	float t = 0.5f + 0.5f * sinf(marble);

	constexpr int kN = 9;
	const float3 knots[kN] = {
		make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f),
		make_float3(.50f,.50f,.50f), make_float3(.60f,.59f,.58f), make_float3(.58f,.58f,.60f),
		make_float3(.58f,.58f,.60f), make_float3(.20f,.20f,.33f), make_float3(.58f,.58f,.60f)
	};
	constexpr int nSeg = kN - 3;
	int first = static_cast<int>(t * nSeg);
	if (first >= nSeg) first = nSeg - 1;
	const float lt = t * nSeg - first;

	float3 rgb = wf_marble_cubic_bezier4(knots[first], knots[first+1], knots[first+2], knots[first+3], lt);
	return make_float3(fminf(rgb.x * 1.5f, 1.0f), fminf(rgb.y * 1.5f, 1.0f), fminf(rgb.z * 1.5f, 1.0f));
}

// Matches windy_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_windy_texture (with the wf_ prefix), same
// no-shared-device-helpers convention as wf_marble_cubic_bezier4 above.
__device__ __forceinline__ float3 wf_sample_windy_texture(const float3& p) {
	const float windStrength = fbm_simple<float>(0.1f*p.x, 0.1f*p.y, 0.1f*p.z, 0.5f, 3);
	const float waveHeight   = fbm_simple<float>(p.x, p.y, p.z, 0.5f, 6);
	const float v = fabsf(windStrength) * waveHeight;
	float t = 0.5f + 0.5f * v;
	t = fminf(fmaxf(t, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches wrinkled_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_wrinkled_texture (with the wf_ prefix).
__device__ __forceinline__ float3 wf_sample_wrinkled_texture(const TextureData& tex, const float3& p) {
	const float v = turbulence_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
	const float t = fminf(fmaxf(v, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches dots_texture::is_inside_dot() (texture.h) exactly - duplicated
// from optix_device_helpers.h's is_inside_dot (with the wf_ prefix).
__device__ __forceinline__ bool wf_is_inside_dot(float s, float t) {
	const float sCell = floorf(s + 0.5f);
	const float tCell = floorf(t + 0.5f);
	if (perlin_noise<float>(sCell + 0.5f, tCell + 0.5f, 0.5f) <= 0.0f) return false;
	constexpr float radius = 0.35f;
	constexpr float maxShift = 0.5f - radius;
	const float sCenter = sCell + maxShift * perlin_noise<float>(sCell + 1.5f, tCell + 2.8f, 0.5f);
	const float tCenter = tCell + maxShift * perlin_noise<float>(sCell + 4.5f, tCell + 9.8f, 0.5f);
	const float ds = s - sCenter, dt = t - tCenter;
	return ds*ds + dt*dt < radius*radius;
}

// Matches bilerp_texture::value() (texture.h) exactly - duplicated from
// optix_device_helpers.h's sample_bilerp_texture (with the wf_ prefix).
// color1/color2 carry v00/v01; the other two corners are packed into
// uScale/vScale/omega and marbleScale/marbleVariation/mixAmount (see
// TextureKind::Bilerp's own comment, optix_types.h).
__device__ __forceinline__ float3 wf_sample_bilerp_texture(const TextureData& tex, float u, float v) {
	const float3& v00 = tex.color1; const float3& v01 = tex.color2;
	const float3 v10 = make_float3(tex.uScale, tex.vScale, tex.omega);
	const float3 v11 = make_float3(tex.marbleScale, tex.marbleVariation, tex.mixAmount);
	const float a = (1.0f-u)*(1.0f-v), b = u*(1.0f-v), c = (1.0f-u)*v, d = u*v;
	return make_float3(
		a*v00.x + b*v10.x + c*v01.x + d*v11.x,
		a*v00.y + b*v10.y + c*v01.y + d*v11.y,
		a*v00.z + b*v10.z + c*v01.z + d*v11.z);
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
	// Shared by the Image case below and by UVChecker/Mix's own
	// tex1ImageIdx/tex2ImageIdx (a one-level-nested bare imagemap - see
	// TextureData's own comment and optix_device_helpers.h's identical
	// sampleImage lambda) - factored out so both call sites in THIS
	// duplicated copy share one instance too. See optix_device_helpers.h's
	// own sampleImage comment for the full wide-clamp/wrap-mode rationale -
	// mirrored here verbatim (same cross-module duplication reason every
	// other wf_ helper in this file is duplicated rather than shared).
	auto sampleImage = [&](const TextureData& t) -> float3 {
		if (t.width <= 0 || t.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uw = fminf(fmaxf(u, -1024.0f), 1024.0f);
		const float vw = fminf(fmaxf(1.0f - v, -1024.0f), 1024.0f);
		// double, not float - see optix_device_helpers.h's identical
		// sampleImage comment for why (matches CPU's own bilerp() promotion,
		// mipmap.h - guards against float's 2^24 exact-integer limit for a
		// wide-tiled UV times a large texture width).
		int i = static_cast<int>(floor((double)uw * t.width));
		int j = static_cast<int>(floor((double)vw * t.height));
		// No `default:` case, deliberately - see optix_device_helpers.h's
		// identical comment for why (lets the compiler flag a future
		// unhandled GpuWrapMode enumerator here too).
		switch (t.wrapMode) {
		case GpuWrapMode::Repeat:
			i = ((i % t.width) + t.width) % t.width;
			j = ((j % t.height) + t.height) % t.height;
			break;
		case GpuWrapMode::Black:
			if (i < 0 || i >= t.width || j < 0 || j >= t.height) return make_float3(0.0f, 0.0f, 0.0f);
			break;
		case GpuWrapMode::Clamp:
			i = min(max(i, 0), t.width - 1);
			j = min(max(j, 0), t.height - 1);
			break;
		}
		const unsigned char* px = texturePixels + t.pixelOffset + (j * t.width + i) * 3;
		constexpr float kColorScale = 1.0f / 255.0f;
		return make_float3(px[0] * kColorScale, px[1] * kColorScale, px[2] * kColorScale);
	};
	if (tex.kind == TextureKind::Image) {
		return sampleImage(tex);
	} else if (tex.kind == TextureKind::Checker) {
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else if (tex.kind == TextureKind::UVChecker) {
		// Matches optix_device_helpers.h's sample_texture() UVChecker branch
		// (and uv_checker_texture::value(), texture.h) exactly, including
		// tex1ImageIdx/tex2ImageIdx's one-level-nested bare imagemap support.
		const int ui = static_cast<int>(floorf(u * tex.uScale));
		const int vi = static_cast<int>(floorf(v * tex.vScale));
		const bool is_even = ((ui + vi) % 2) == 0;
		// Only the winning cell's slot is sampled - see
		// optix_device_helpers.h's identical UVChecker branch comment.
		return is_even
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::FBm) {
		// Matches optix_device_helpers.h's sample_texture() FBm branch (and
		// fbm_texture::value(), texture.h) exactly.
		const float v = fbm_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
		float t = 0.5f + 0.5f * v;
		t = fminf(fmaxf(t, 0.0f), 1.0f);
		return make_float3(t, t, t);
	} else if (tex.kind == TextureKind::Marble) {
		return wf_sample_marble_texture(tex, p);
	} else if (tex.kind == TextureKind::Windy) {
		return wf_sample_windy_texture(p);
	} else if (tex.kind == TextureKind::Wrinkled) {
		return wf_sample_wrinkled_texture(tex, p);
	} else if (tex.kind == TextureKind::Dots) {
		const bool inside = wf_is_inside_dot(u, v);
		return inside
			? ((tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1)
			: ((tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2);
	} else if (tex.kind == TextureKind::Bilerp) {
		return wf_sample_bilerp_texture(tex, u, v);
	} else if (tex.kind == TextureKind::Mix) {
		// Matches optix_device_helpers.h's sample_texture() Mix branch (and
		// mix_texture::value(), texture.h) exactly, including
		// amountImageIdx's one-level-nested bare imagemap support for
		// "amount" itself.
		const float3 c1 = (tex.tex1ImageIdx >= 0) ? sampleImage(textures[tex.tex1ImageIdx]) : tex.color1;
		const float3 c2 = (tex.tex2ImageIdx >= 0) ? sampleImage(textures[tex.tex2ImageIdx]) : tex.color2;
		const float amt = (tex.amountImageIdx >= 0) ? sampleImage(textures[tex.amountImageIdx]).x : tex.mixAmount;
		return (1.0f - amt) * c1 + amt * c2;
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
#include "gpu_portal_light_shared.h"
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
	// Shared matrix-only core (sampled_spectrum.h) - deliberately NOT
	// XYZToLinearRGB(), which clamps negatives; see this function's own
	// comment above for why that clamp must not happen here.
	XYZToLinearRGBMatrix(X, Y, Z, r, g, b);
}

// Forward-declared: defined further down alongside wf_dc_apply_vector/
// wf_dc_apply_normal_from_w2o (this function's own header comment there),
// needed here by wf_sample_sphere_light's ClippedSphere UV branch below.
__device__ __forceinline__ float3 wf_dc_apply_point(const float m[12], const float3& p);

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
										  unsigned int& seed, float& pdf, float& maxDist,
										  float& out_u, float& out_v, float3& out_normal,
										  // Object (per-primitive sphere) motion blur shutter
										  // time (RayWorkItem::time's own comment) - a moving
										  // emissive sphere's NEE sample must target the SAME
										  // interpolated position the caller's own shadow ray
										  // (traced at this same time) will test occlusion
										  // against, or the two disagree on where the light
										  // actually is. 0.0f for a static sphere (center1==
										  // center) is a provable no-op, same convention as
										  // optix_intersection_sphere.h's own ray_time lerp.
										  float time) {
	const float3 center = make_float3(
		sph.center.x + time * (sph.center1.x - sph.center.x),
		sph.center.y + time * (sph.center1.y - sph.center.y),
		sph.center.z + time * (sph.center1.z - sph.center.z));
	float3 to_c = center - hit;
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
	float3 oc = hit - center;
	float b = dot(oc, dir);
	float c = dot(oc, oc) - r * r;
	float disc = fmaxf(0.0f, b * b - c);
	float sq = sqrtf(disc);
	float tNear = -b - sq;
	maxDist = (tNear > 1e-6f) ? tNear : fmaxf(0.0f, -b + sq);

	// UV/normal at the recovered surface point - same theta/phi convention
	// as __closesthit__sphere's direct-hit formula (optix_intersection_
	// sphere.h), see optix_device_helpers.h's sample_sphere_light() for the
	// identical derivation.
	const float3 point = hit + maxDist * dir;
	const float3 local = (point - center) / r;
	out_normal = local;

	// UV convention must match the DIRECT-hit closest-hit program for this
	// same shapeKind, or a "filename"-textured light samples a different
	// texel via NEE than a camera ray hitting it directly - a ClippedSphere's
	// direct hit uses pbrt-v4's Z-pole convention (__closesthit__wf_sphere's
	// own comment), not this function's plain-sphere Y-pole one below, since
	// zMin/zMax/phiMax are themselves Z-pole-defined. `local`/sph.center/
	// sph.radius above stay the full-sphere-cone approximation (this
	// function's own established, accepted geometric/pdf simplification) -
	// only the UV derivation is shapeKind-aware, via the real object-space
	// affine, purely to keep the (u,v) direct-hit-consistent.
	if (sph.shapeKind == GpuMediumShapeKind::ClippedSphere) {
		const float3 objPt = wf_dc_apply_point(sph.w2o, point);
		const float rl = sph.radiusLocal;
		const float cosTheta = fminf(1.0f, fmaxf(-1.0f, (rl > 0.0f) ? (objPt.z / rl) : 0.0f));
		const float theta = acosf(cosTheta);
		float phi = atan2f(objPt.y, objPt.x);
		if (phi < 0.0f) phi += 2.0f * 3.14159265358979323846f;
		// thetaZMin/thetaZMax are host-precomputed (SphereData's own comment).
		out_u = (sph.phiMax > 1e-8f) ? (phi / sph.phiMax) : 0.0f;
		out_v = (sph.thetaZMax > sph.thetaZMin)
			? (theta - sph.thetaZMin) / (sph.thetaZMax - sph.thetaZMin) : 0.0f;
	} else {
		const float sphere_theta = acosf(fmaxf(-1.0f, fminf(1.0f, -local.y)));
		const float sphere_phi = atan2f(-local.z, local.x) + 3.14159265358979323846f;
		out_u = sphere_phi / (2.0f * 3.14159265358979323846f);
		out_v = sphere_theta / 3.14159265358979323846f;
	}

	return dir;
}

// Sample a point on a quad light.
__device__ float3 wf_sample_quad_light(const QuadData& q, const float3& hit,
										unsigned int& seed, float& geom_pdf, float& maxDist,
										float& out_u, float& out_v) {
	float s = wf_rand(seed), t = wf_rand(seed);
	out_u = s;
	out_v = t;
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
											unsigned int& seed, float& geom_pdf, float& maxDist,
											float& out_u, float& out_v, float3& out_normal) {
	float a = wf_rand(seed), b = wf_rand(seed);
	if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
	const float3 e1 = tri.p1 - tri.p0;
	const float3 e2 = tri.p2 - tri.p0;
	const float3 point = tri.p0 + a * e1 + b * e2;

	// Same b0/b1/b2 = (1-a-b, a, b) convention __closesthit__wf_triangle
	// uses (real barycentric UV) - see optix_device_helpers.h's
	// sample_triangle_light() for the identical derivation.
	out_u = out_v = 0.0f;
	if (tri.hasUVs) {
		const float b0 = 1.0f - a - b;
		out_u = b0 * tri.uv0.x + a * tri.uv1.x + b * tri.uv2.x;
		out_v = b0 * tri.uv0.y + a * tri.uv1.y + b * tri.uv2.y;
	}

	float3 dir = point - hit;
	maxDist = length(dir);
	if (maxDist < 1e-6f) { geom_pdf = 0.0f; out_normal = make_float3(0.0f, 0.0f, 1.0f); return make_float3(0.0f, 0.0f, 1.0f); }
	dir = dir / maxDist;

	const float3 n_unnorm = cross(e1, e2);
	const float twice_area = length(n_unnorm);
	if (twice_area < 1e-12f) { geom_pdf = 0.0f; out_normal = dir; return dir; }
	const float3 normal = n_unnorm / twice_area;
	out_normal = normal;
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
												   unsigned int& seed, float& geom_pdf, float& maxDist,
												   float& out_u, float& out_v, float3& out_normal) {
	const float p00[3] = {bp.p00.x, bp.p00.y, bp.p00.z};
	const float p10[3] = {bp.p10.x, bp.p10.y, bp.p10.z};
	const float p01[3] = {bp.p01.x, bp.p01.y, bp.p01.z};
	const float p11[3] = {bp.p11.x, bp.p11.y, bp.p11.z};
	const float u2[2] = {wf_rand(seed), wf_rand(seed)};
	float outP[3], outN[3], areaPdf = 0.0f, su = 0.0f, sv = 0.0f;
	blp_sample(p00, p10, p01, p11, u2, outP, outN, &areaPdf, &su, &sv);
	out_u = su;
	out_v = sv;

	const float3 point  = make_float3(outP[0], outP[1], outP[2]);
	const float3 normal = make_float3(outN[0], outN[1], outN[2]);
	out_normal = normal;
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

// wf_dc_apply_point/wf_dc_apply_vector/wf_dc_apply_normal_from_w2o -
// wavefront-native duplicates of optix_disk_cylinder_helpers.h's identically
// named recursive-backend functions (same cross-module reason every other
// wf_ helper in this file is duplicated, not shared via #include - see
// optix_disk_cylinder_helpers.h's own header comment).
__device__ __forceinline__ float3 wf_dc_apply_point(const float m[12], const float3& p) {
	return make_float3(
		m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
		m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
		m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}
__device__ __forceinline__ float3 wf_dc_apply_vector(const float m[12], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2]  * v.z,
		m[4] * v.x + m[5] * v.y + m[6]  * v.z,
		m[8] * v.x + m[9] * v.y + m[10] * v.z);
}
__device__ __forceinline__ float3 wf_dc_apply_normal_from_w2o(const float w2o[12], const float3& n) {
	return make_float3(
		w2o[0] * n.x + w2o[4] * n.y + w2o[8]  * n.z,
		w2o[1] * n.x + w2o[5] * n.y + w2o[9]  * n.z,
		w2o[2] * n.x + w2o[6] * n.y + w2o[10] * n.z);
}
__device__ __forceinline__ float wf_dc_representative_scale(const float o2w[12]) {
	return length(make_float3(o2w[0], o2w[4], o2w[8]));
}
__device__ __forceinline__ float wf_dc_area_disk(const DiskData& disk) {
	const float scale = wf_dc_representative_scale(disk.o2w);
	const float objArea = disk.phiMax * 0.5f * (disk.radius * disk.radius - disk.innerRadius * disk.innerRadius);
	return objArea * scale * scale;
}
__device__ __forceinline__ float wf_dc_area_cylinder(const CylinderData& cyl) {
	const float scale = wf_dc_representative_scale(cyl.o2w);
	const float objArea = (cyl.zMax - cyl.zMin) * cyl.radius * cyl.phiMax;
	return objArea * scale * scale;
}

// Sample a point on a disk light. Object-space uniform-area concentric
// sample (SampleUniformDiskConcentric, src/shared/sampling_helpers.h -
// CPU_GPU-tagged, safe to call directly here, same reasoning as blp_sample's
// own comment above) transformed to world space via the disk's own o2w -
// same math as optix_disk_cylinder_helpers.h's dc_sample_disk()/
// sample_disk_light(), duplicated here for the same cross-module reason
// every other wf_sample_*_light function in this file is.
__device__ float3 wf_sample_disk_light(const DiskData& disk, const float3& hit,
										unsigned int& seed, float& geom_pdf, float& maxDist,
										float& out_u, float& out_v, float3& out_normal) {
	float dx, dy;
	SampleUniformDiskConcentric<float>(wf_rand(seed), wf_rand(seed), dx, dy);
	const float3 obj_point = make_float3(dx * disk.radius, dy * disk.radius, disk.height);
	const float3 point = wf_dc_apply_point(disk.o2w, obj_point);
	const float3 normal = normalize(wf_dc_apply_normal_from_w2o(disk.w2o, make_float3(0.0f, 0.0f, 1.0f)));
	out_normal = normal;
	const float area = wf_dc_area_disk(disk);
	const float area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;

	// Real UV (phi/phiMax, radial fraction) - same formula __closesthit__wf_disk
	// recomputes for a direct hit. obj_point is already the object-space
	// sampled point (before the o2w transform above), so no re-derivation
	// via w2o is needed here, unlike the closest-hit path which only has the
	// world-space hit point to start from.
	{
		float uv_phi = atan2f(dy, dx);
		if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
		const float uv_dist = sqrtf(dx * dx + dy * dy) * disk.radius;
		out_u = uv_phi / disk.phiMax;
		out_v = (disk.radius > disk.innerRadius)
			? 1.0f - (uv_dist - disk.innerRadius) / (disk.radius - disk.innerRadius)
			: 0.0f;
	}

	float3 dir = point - hit;
	maxDist = length(dir);
	if (maxDist < 1e-6f || area_pdf <= 0.0f) { geom_pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	dir = dir / maxDist;

	const float cosine = fabsf(dot(dir, normal));
	if (cosine < 1e-6f) { geom_pdf = 0.0f; return dir; }
	geom_pdf = area_pdf * maxDist * maxDist / cosine;
	return dir;
}

// Sample a point on a cylinder light. Uniform Z along the axis, uniform phi
// within the sweep, same math as optix_disk_cylinder_helpers.h's
// dc_sample_cylinder()/sample_cylinder_light().
__device__ float3 wf_sample_cylinder_light(const CylinderData& cyl, const float3& hit,
											unsigned int& seed, float& geom_pdf, float& maxDist,
											float& out_u, float& out_v, float3& out_normal) {
	const float z = cyl.zMin + wf_rand(seed) * (cyl.zMax - cyl.zMin);
	const float phi = wf_rand(seed) * cyl.phiMax;
	const float3 obj_normal = make_float3(cosf(phi), sinf(phi), 0.0f);
	const float3 obj_point = make_float3(cyl.radius * obj_normal.x, cyl.radius * obj_normal.y, z);
	const float3 point = wf_dc_apply_point(cyl.o2w, obj_point);
	const float3 normal = normalize(wf_dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	out_normal = normal;
	const float area = wf_dc_area_cylinder(cyl);
	const float area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;

	// Real UV (phi/phiMax, z-fraction) - phi was already drawn directly above
	// (not recovered from obj_point), matching __closesthit__wf_cylinder's
	// direct-hit formula.
	out_u = phi / cyl.phiMax;
	out_v = (cyl.zMax > cyl.zMin) ? (z - cyl.zMin) / (cyl.zMax - cyl.zMin) : 0.0f;

	float3 dir = point - hit;
	maxDist = length(dir);
	if (maxDist < 1e-6f || area_pdf <= 0.0f) { geom_pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	dir = dir / maxDist;

	const float cosine = fabsf(dot(dir, normal));
	if (cosine < 1e-6f) { geom_pdf = 0.0f; return dir; }
	geom_pdf = area_pdf * maxDist * maxDist / cosine;
	return dir;
}

// Equal-area sphere->square mapping for the goniometric light's image
// lookup. Duplicated from optix_device_helpers.h's dev_equal_area_sphere_to_square
// (itself a local copy of src/shared/sampling_extra.h's EqualAreaSphereToSquare -
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

	if (cam.animated) {
		// Real per-ray shutter-time camera motion blur - see
		// optix_device_helpers.h's generate_primary_ray for the derivation
		// this mirrors.
		float3 local_pixel_sample = cam.localLowerLeftCorner + u * cam.localHorizontal + v * cam.localVertical;
		float3 local_origin = make_float3(0.0f, 0.0f, 0.0f);
		bool hasDOF = (cam.localDefocusDiskU.x != 0.0f || cam.localDefocusDiskU.y != 0.0f || cam.localDefocusDiskU.z != 0.0f ||
					   cam.localDefocusDiskV.x != 0.0f || cam.localDefocusDiskV.y != 0.0f || cam.localDefocusDiskV.z != 0.0f);
		if (hasDOF) {
			float rx = 2.0f * wf_rand(seed) - 1.0f, ry = 2.0f * wf_rand(seed) - 1.0f;
			while (rx*rx + ry*ry >= 1.0f) { rx = 2.0f * wf_rand(seed) - 1.0f; ry = 2.0f * wf_rand(seed) - 1.0f; }
			local_origin = rx * cam.localDefocusDiskU + ry * cam.localDefocusDiskV;
		}
		float3 local_direction = local_pixel_sample - local_origin;
		float dt = wf_rand(seed);
		// Slerp + quaternion-to-matrix built ONCE (gpu_camera_anim_rotation,
		// camera_motion_blur_device.h - shared with optix_device_helpers.h,
		// since neither it nor gpu_camera_anim_apply touch this kernel's
		// own state), then applied to both origin and direction.
		GpuAnimRotMat rot = gpu_camera_anim_rotation(cam.animR0, cam.animR1, dt);
		origin = gpu_camera_anim_apply(rot, local_origin, true, cam.animT0, cam.animT1, dt);
		// local_direction is not unit length and rotation preserves length,
		// so this needs an explicit normalize - every other camera branch
		// in this function returns a unit direction too.
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

