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

// ============================================================================
// Device helpers (shared with optix_programs.cu logic)
// ============================================================================

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

// reflect/refract wrappers
__device__ __forceinline__ float3 wf_reflect(const float3& v, const float3& n) {
	return cpu_gpu_reflect(v, n);
}
__device__ __forceinline__ float3 wf_refract(const float3& v, const float3& n, float e) {
	return cpu_gpu_refract<float3, float>(v, n, e);
}

// MIS power heuristic (balance if both 0)
__device__ __forceinline__ float wf_mis(float a, float b) {
	float a2 = a * a, b2 = b * b;
	return (a2 + b2 < 1e-10f) ? 1.0f : a2 / (a2 + b2);
}

// Sample a point on a sphere light; returns direction and sets geom_pdf.
__device__ float3 wf_sample_sphere_light(const SphereData& sph, const float3& hit,
										  unsigned int& seed, float& pdf) {
	float3 to_c = sph.center - hit;
	float dist  = length(to_c);
	float r     = sph.radius;
	if (dist <= r) { pdf = 1.0f / (4.0f * 3.14159265f * r * r); return normalize(wf_rand_unit(seed)); }
	float cos_max = sqrtf(fmaxf(0.0f, 1.0f - (r * r) / (dist * dist)));
	float phi     = 2.0f * 3.14159265f * wf_rand(seed);
	float cos_t   = 1.0f - wf_rand(seed) * (1.0f - cos_max);
	float sin_t   = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
	float3 w      = normalize(to_c);
	float3 u, v;
	if (fabsf(w.x) > 0.9f) u = normalize(cross(make_float3(0,1,0), w));
	else                    u = normalize(cross(make_float3(1,0,0), w));
	v = cross(w, u);
	float3 dir = normalize(sin_t * cosf(phi) * u + sin_t * sinf(phi) * v + cos_t * w);
	float solid = 2.0f * 3.14159265f * (1.0f - cos_max);
	pdf = (solid > 1e-10f) ? 1.0f / solid : 1.0f;
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

// Evaluate one punctual (point/spot/distant) light at shading point p: same
// dispatch as optix_device_helpers.h's eval_punctual_light(), duplicated here
// (with the wf_ prefix) rather than shared, matching this file's existing
// pattern of not sharing NEE helpers with the recursive path (this kernel
// has no access to the recursive path's __constant__ params global, and the
// two strategies' NEE application differs - RGB float3 here vs this file's
// spectral radiance). PDF is always 1 (delta light) - see punctual_lights.h.
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

extern "C" __global__ void generate_camera_rays(
	WorkQueue<RayWorkItem> rayQueue,
	unsigned int width,
	unsigned int height,
	float3       camOrigin,
	float3       lowerLeft,
	float3       horizontal,
	float3       vertical,
	unsigned int sampleIdx,
	unsigned int frameNumber
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= (int)width || py >= (int)height) return;

	int pixelIdx = py * (int)width + px;
	unsigned int seed = wf_pcg(wf_pcg(pixelIdx + sampleIdx * width * height) ^ frameNumber);

	float u = (float(px) + wf_rand(seed)) / float(width  - 1);
	float v = (float(py) + wf_rand(seed)) / float(height - 1);

	RayWorkItem item;
	item.origin     = camOrigin;
	item.direction  = normalize(lowerLeft + u * horizontal + v * vertical - camOrigin);
	// Sample hero wavelengths for spectral rendering (pbrt-v4: SampledWavelengths::SampleVisible)
	float lambda_u = wf_rand(seed);
	SampledWavelengths<kWFNWavelengths> swl = SampledWavelengths<kWFNWavelengths>::SampleVisible(lambda_u);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		item.throughput[i]     = 1.0f;
		item.radiance[i]       = 0.0f;
		item.wavelengths[i]    = swl.lambda[i];
		item.wavelength_pdfs[i] = swl.pdf[i];
	}
	item.seed       = seed;
	item.pixelIndex = pixelIdx;
	item.depth      = 0;
	item.specular_bounce = 1;  // primary ray: always allow emissive hit
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

extern "C" __global__ void evaluate_materials(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      framebuffer,
	// Scene data
	const SphereData*   spheres,
	const QuadData*     quads,
	const MaterialData* materials,
	const int*          lightIndices,
	const bool*         isLightSphere,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	int maxDepth
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;

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
		XYZToSRGB(xyz.x, xyz.y, xyz.z, r, g, b);
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
				float3 le = mat.emission;
				float m_le = le.x > le.y ? (le.x > le.z ? le.x : le.z)
										  : (le.y > le.z ? le.y : le.z);
				float sc = 2.f * m_le;
				if (sc > 0.f) {
					float c0, c1, c2;
					dev_srgb_to_coeffs(le.x/sc, le.y/sc, le.z/sc, c0, c1, c2);
					RGBSigmoidPolynomial emitPoly(c0, c1, c2);
					SS emitS(0.f);
					for (int i = 0; i < kWFNWavelengths; ++i)
						emitS[i] = sc * emitPoly(swl.lambda[i]);
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

	switch (mat.type) {
	case MaterialType::Lambertian: {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Metal: {
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
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
		if (cannot_refract || schlick > wf_rand(seed))
			scattered_dir = wf_reflect(unit_dir, normal);
		else
			scattered_dir = wf_refract(unit_dir, normal, eta);
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::RoughDielectric: {
		float rd_alpha = sqrtf(mat.fuzz);
		float3 n = normal;
		float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 tan_v  = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		float3 wi_w = -normalize(h.rayDir);
		float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
		if (wi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> rd_dist(rd_alpha, rd_alpha);
		float wm_x, wm_y, wm_z;
		rd_dist.Sample_wm(wi_x, wi_y, wi_z, wf_rand(seed), wf_rand(seed), wm_x, wm_y, wm_z);
		float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float Fr = FrDielectric(rd_dot, mat.ior);
		if (wf_rand(seed) < Fr) {
			// Reflect
			float wo_x = 2.0f*rd_dot*wm_x - wi_x;
			float wo_y = 2.0f*rd_dot*wm_y - wi_y;
			float wo_z = 2.0f*rd_dot*wm_z - wi_z;
			scattered_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		} else {
			// Refract
			float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
			float eta = dot(h.rayDir, normal) < 0.0f ? (1.0f / mat.ior) : mat.ior;
			scattered_dir = wf_refract(normalize(h.rayDir), wm_world, eta);
		}
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = true;
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
		is_specular = true;
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
			is_specular   = true;
		} else {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(mat.albedo);
			is_specular = false;
		}
		scattered = (dot(scattered_dir, normal) > 0.0f);
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
		float Fr_coat = FrDielectric(fmaxf(cc_dot, 0.0f), mat.ior);
		if (wf_rand(seed) < Fr_coat) {
			float ccwo_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float ccwo_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float ccwo_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			scattered_dir = normalize(ccwo_x*cctan + ccwo_y*ccbitan + ccwo_z*ccn);
			attenuation   = SS(1.f);
		} else {
			// Conductor layer: resample microfacet normal
			float ccwm2_x, ccwm2_y, ccwm2_z;
			cc_dist.Sample_wm(ccwi_x, ccwi_y, ccwi_z, wf_rand(seed), wf_rand(seed), ccwm2_x, ccwm2_y, ccwm2_z);
			float cc_dot2 = ccwi_x*ccwm2_x + ccwi_y*ccwm2_y + ccwi_z*ccwm2_z;
			float ccwo2_x = 2.0f*cc_dot2*ccwm2_x - ccwi_x;
			float ccwo2_y = 2.0f*cc_dot2*ccwm2_y - ccwi_y;
			float ccwo2_z = 2.0f*cc_dot2*ccwm2_z - ccwi_z;
			if (ccwo2_z <= 0.0f) { scattered = false; break; }
			float cc_G1  = cc_dist.G1(ccwi_x, ccwi_y, ccwi_z);
			float cc_G   = cc_dist.G(ccwo2_x, ccwo2_y, ccwo2_z, ccwi_x, ccwi_y, ccwi_z);
			float cc_w   = (cc_G1 > 1e-8f) ? cc_G / cc_G1 : 0.0f;
			float3 c_F   = FrConductorRGB(cc_dot2, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
			attenuation   = albedoSpectrum(make_float3(c_F.x * cc_w, c_F.y * cc_w, c_F.z * cc_w));
			scattered_dir = normalize(ccwo2_x*cctan + ccwo2_y*ccbitan + ccwo2_z*ccn);
		}
		scattered   = (dot(scattered_dir, normal) > 0.0f);
		is_specular = true;
		break;
	}
	case MaterialType::DiffuseTransmission: {
		float3 R = mat.albedo;
		float3 T_col = make_float3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z);
		float prob_r = (R.x + R.y + R.z) / 3.0f;
		if (wf_rand(seed) < prob_r) {
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
		is_specular = false;
		break;
	}
	case MaterialType::NormalizedFresnel: {
		float nf_eta = mat.ior;
		float inv_eta = 1.0f / nf_eta;
		float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
		if (nf_c <= 0.0f) nf_c = 1e-6f;
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		float cos_wi = fmaxf(dot(scattered_dir, normal), 1e-6f);
		float fr = FrDielectric(cos_wi, nf_eta);
		float weight = (1.0f - fr) / nf_c;
		attenuation = SS(weight);
		scattered   = true;
		is_specular = false;
		brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265f);
		break;
	}
	default:
		scattered = false;
		break;
	}

	if (!scattered) {
		// Path absorbed — accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance);
		return;
	}

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

		int   prim_idx        = lightIndices[light_idx];
		bool  is_sphere_light = isLightSphere[light_idx];

		float  geom_pdf = 0.0f, max_dist = 0.0f;
		float3 to_light;
		SS light_emission_spec(0.f);

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
				s[i] = sc * poly(swl.lambda[i]);
			return s;
		};

		if (is_sphere_light) {
			const SphereData& s = spheres[prim_idx];
			to_light = wf_sample_sphere_light(s, hit_point, seed, geom_pdf);
			max_dist = length(s.center - hit_point);
			light_emission_spec = liftEmission(materials[s.materialIdx].emission);
		} else {
			const QuadData& q = quads[prim_idx];
			to_light = wf_sample_quad_light(q, hit_point, seed, geom_pdf, max_dist);
			light_emission_spec = liftEmission(materials[q.materialIdx].emission);
		}

		float light_pdf = selection_pdf * geom_pdf;
		if (light_pdf > 1e-6f && dot(to_light, normal) > 0.0f) {
			float cos_l = fmaxf(dot(to_light, normal), 0.0f);
			float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
			if (mat.type == MaterialType::NormalizedFresnel) {
				float nf_eta = mat.ior;
				float inv_eta = 1.0f / nf_eta;
				float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
				if (nf_c <= 0.0f) nf_c = 1e-6f;
				float fr_l = FrDielectric(cos_l, nf_eta);
				bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
			}
			float brdf_pdf_l = (brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(light_pdf, brdf_pdf_l);

			// Spectral direct-light contribution
			SS Ld = (mis_w * bsdf_val * cos_l / light_pdf) * throughput * light_emission_spec;

			ShadowRayWorkItem shadow;
			shadow.origin    = hit_point + 0.001f * normal;
			shadow.direction = to_light;
			shadow.tMax      = max_dist - 0.002f;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				shadow.Ld[i]             = Ld[i];
				shadow.wavelengths[i]    = swl.lambda[i];
				shadow.wavelength_pdfs[i] = swl.pdf[i];
			}
			shadow.pixelIndex = h.pixelIndex;
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
		float cos_l = dot(wi, normal);
		if (cos_l <= 0.0f) continue;

		float bsdf_val = 1.0f / 3.14159265f; // Lambertian default
		if (mat.type == MaterialType::NormalizedFresnel) {
			float nf_eta = mat.ior;
			float inv_eta = 1.0f / nf_eta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;
			float fr_l = FrDielectric(cos_l, nf_eta);
			bsdf_val = (1.0f - fr_l) / (nf_c * 3.14159265f);
		}

		// Uplift RGB Li to spectrum (same pattern as liftEmission above)
		float m = Li.x > Li.y ? (Li.x > Li.z ? Li.x : Li.z) : (Li.y > Li.z ? Li.y : Li.z);
		float sc = 2.f * m;
		SS Li_spec(0.f);
		if (sc > 0.f) {
			float c0, c1, c2;
			dev_srgb_to_coeffs(Li.x / sc, Li.y / sc, Li.z / sc, c0, c1, c2);
			RGBSigmoidPolynomial poly(c0, c1, c2);
			for (int i = 0; i < kWFNWavelengths; ++i)
				Li_spec[i] = sc * poly(swl.lambda[i]);
		}

		SS Ld = (bsdf_val * cos_l) * throughput * Li_spec;

		ShadowRayWorkItem shadow;
		shadow.origin    = hit_point + 0.001f * normal;
		shadow.direction = wi;
		shadow.tMax      = t_max - 0.002f;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			shadow.Ld[i]              = Ld[i];
			shadow.wavelengths[i]     = swl.lambda[i];
			shadow.wavelength_pdfs[i] = swl.pdf[i];
		}
		shadow.pixelIndex = h.pixelIndex;
		shadowQueue.push(shadow);
	}

	// Flush accumulated prior-bounce radiance (once, regardless of whether
	// any area/punctual lights actually contributed above).
	if ((bool)radiance) addToFramebuffer(h.pixelIndex, radiance);
	} // if (!is_specular) - NEE (area + punctual lights)

	// -------------------------------------------------------------------------
	// Bounce: push next ray
	// -------------------------------------------------------------------------
	SS new_throughput = throughput * attenuation;

	// Russian roulette after depth 3
	if (h.depth >= 3) {
		float p = new_throughput.MaxComponentValue();
		if (wf_rand(seed) >= p) {
			if (is_specular) addToFramebuffer(h.pixelIndex, radiance);
			return;
		}
		new_throughput = new_throughput / p;
	}

	RayWorkItem next;
	next.origin     = hit_point + 0.001f * scattered_dir;
	next.direction  = normalize(scattered_dir);
	next.seed            = seed;
	next.pixelIndex      = h.pixelIndex;
	next.depth           = h.depth + 1;
	next.specular_bounce = is_specular ? 1 : 0;
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

// ============================================================================
// Kernel 3 — accumulate_miss
//   For rays that escaped the scene, add background color (black by default;
//   extend here for an environment map).
// ============================================================================

extern "C" __global__ void accumulate_miss(
	WorkQueue<MissWorkItem> missQueue,
	int                     numMiss,
	float3*                 framebuffer
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numMiss) return;

	const MissWorkItem& m = missQueue.items[idx];
	// Cornell box uses a black background (closed scene) — no contribution.
	// If throughput * background were non-zero, convert via XYZ and accumulate.
	// For extensibility: background is black, so nothing to do here.
	(void)m;
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
	if (idx >= numShadow) return;

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
		XYZToSRGB(xyz.x, xyz.y, xyz.z, r, g, b);
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
