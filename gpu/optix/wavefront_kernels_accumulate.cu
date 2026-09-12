// wavefront_kernels_accumulate.cu
// CUDA compute kernels: accumulate_miss, accumulate_shadow,
// reset_queue_counter, normalize_framebuffer, normalize_aov_buffers
// (Kernels 3-6 of the original wavefront_kernels.cu) - framebuffer/queue
// bookkeeping kernels grouped together since each is small. Split out for
// the same independent-recompilation reason as the other
// wavefront_kernels_*.cu files - see wavefront_device_helpers.h's own
// header comment for where the two shared per-kernel-family helpers now
// live.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

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
	GpuSkyDistribution      skyDist,
	GpuPortalLight          portalLight,
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h).
	float                   maxComponentValue,
	// Denoiser guide-layer AOVs - see evaluate_materials()'s own comment.
	float3*                 albedoBuffer,
	float3*                 normalBuffer,
	float4*                 worldPosBuffer
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
	float3 color = wf_sky_radiance(skyDist, portalLight, m.rayDir, backgroundColor, m.rayOrigin);

	// Denoiser guide-layer AOV, primary-ray misses only (depth==0) - mirrors
	// optix_miss.h's pack_aov_payload(color, -rayDir) for the recursive
	// backend exactly (background color as "albedo", the incoming ray's
	// reverse direction as a placeholder "normal" for a surface that isn't
	// there). Unconditional on whether L below ends up non-empty - the AOV
	// write is independent of the radiance contribution.
	// Live Preview reprojection guide buffer - a miss has no real surface
	// point, so this is explicitly marked invalid (w=0) rather than left to
	// the render()-call-start memset (see wf_write_world_pos()'s own
	// comment on the validity flag) - defensively correct even if that
	// memset is ever removed/changed.
	if (worldPosBuffer && m.depth == 0) {
		worldPosBuffer[m.pixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}

	if (albedoBuffer && m.depth == 0) {
		atomicAdd(&albedoBuffer[m.pixelIndex].x, color.x);
		atomicAdd(&albedoBuffer[m.pixelIndex].y, color.y);
		atomicAdd(&albedoBuffer[m.pixelIndex].z, color.z);
		atomicAdd(&normalBuffer[m.pixelIndex].x, -m.rayDir.x);
		atomicAdd(&normalBuffer[m.pixelIndex].y, -m.rayDir.y);
		atomicAdd(&normalBuffer[m.pixelIndex].z, -m.rayDir.z);
	}

	float3 weightedBackground = color;
	if (m.brdf_pdf > 0.0f && (backgroundColor.x > 0.0f || backgroundColor.y > 0.0f || backgroundColor.z > 0.0f)) {
		const float pdf_sky = wf_sky_pdf_for_mis(skyDist, portalLight, m.rayDir, m.rayOrigin);
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
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h) for
	// why this is a per-contribution clamp, not a true per-sample one.
	{
		const float m2 = fmaxf(r, fmaxf(g, b));
		if (maxComponentValue > 0.0f && m2 > maxComponentValue) {
			const float s = maxComponentValue / m2;
			r *= s; g *= s; b *= s;
		}
	}
	atomicAdd(&framebuffer[m.pixelIndex].x, r);
	atomicAdd(&framebuffer[m.pixelIndex].y, g);
	atomicAdd(&framebuffer[m.pixelIndex].z, b);
}

// ============================================================================
// Kernel 4 — accumulate_shadow
//   Runs after the OptiX shadow-trace pass.  ShadowRayWorkItems that were NOT
//   occluded have their `occluded` flag cleared by the shadow miss program;
//   we accumulate their Ld, scaled by whatever transmittance survived any
//   participating media the ray crossed, into the framebuffer.
//   (The transmittance value is stored in a separate float array passed
//    alongside the shadow queue items — see wavefront_path_tracer.cpp. <= 0.0f
//    means fully occluded, same as the old plain `bool occluded` did; values
//    in between mean the ray crossed one or more media - see
//    WfShadowPayload::transmittance's own comment, wavefront_common.h.)
// ============================================================================

extern "C" __global__ void accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	int                          numShadow,
	const float*                 transmittance,   // per-item transmittance result
	float3*                      framebuffer,
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h).
	float                        maxComponentValue,
	// ReSTIR GI (Live Preview only, gpu/optix/wavefront_restir_gi_math.h) -
	// an item with isGiCandidate set (ShadowRayWorkItem's own comment) adds
	// its Ld into giCandidateOut[s.pixelIndex].radiance instead of
	// `framebuffer`, UNCLAMPED (see this parameter's own null-default call
	// site comment in wavefront_launch.h/.cu for why no maxComponentValue
	// clamp applies here). The candidate's OTHER fields (x1Point/x1Normal/
	// x0Point/pdfAtX0) are already written synchronously by
	// wf_finish_material_scatter itself (giCandidateOut's own parameter
	// comment, wavefront_device_helpers.h) - this kernel only ever adds into
	// `.radiance`. nullptr (every existing call site) makes every item
	// behave exactly as before - isGiCandidate is never set without a real
	// buffer to write to. No default here (unlike wf_launch_accumulate_
	// shadow's own host-side wrapper) - a __global__ kernel launch lists
	// every argument explicitly at its one real call site anyway, so there
	// is no ambiguity to resolve.
	GpuGiSample*                 giCandidateOut
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Must also guard against shadowQueue.capacity, not just the host-
	// supplied numShadow: numShadow is *shadowCounter read back on the host
	// (WavefrontPathTracer::render(), readQueueSize()), and WorkQueue::push()
	// (wavefront_types.h) keeps incrementing that counter even once the
	// backing d_shadowItems_/d_transmittance_ buffers (sized to shadowQueue.
	// capacity = queueCapacity_) are full - it only stops writing items[]
	// past capacity. A bounce that legitimately queues more shadow rays than
	// capacity (confirmed: scene B2/"Cornell Rough Metal" combines area-
	// light NEE with a non-zero-backgroundColor sky-NEE push per hit - see
	// __raygen__wf_shadow's own version of this comment, wavefront_
	// programs.cu) would otherwise read transmittance[idx]/shadowQueue.
	// items[idx] past their allocations here too.
	if (idx >= numShadow || idx >= shadowQueue.capacity) return;

	const float tr = transmittance[idx];
	if (tr > 0.0f) {
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
		// Attenuate by whatever survived any participating media crossed -
		// see WfShadowPayload::transmittance's own comment. 1.0f (the common
		// case: no medium in the way) is a no-op multiply.
		r *= tr; g *= tr; b *= tr;

		// ReSTIR GI (Live Preview only) - redirect into the GI candidate
		// buffer instead of the real framebuffer, UNCLAMPED: this value is an
		// internal cache the GI finalize pass (wavefront_kernels_restir.cu)
		// re-derives a fresh, already-clamped framebuffer contribution FROM -
		// clamping twice here would double-attenuate a legitimately bright
		// indirect bounce before finalize ever sees it. See ShadowRayWorkItem::
		// isGiCandidate's own comment (wavefront_types.h) for who sets this.
		if (s.isGiCandidate && giCandidateOut != nullptr) {
			atomicAdd(&giCandidateOut[s.pixelIndex].radiance.x, r);
			atomicAdd(&giCandidateOut[s.pixelIndex].radiance.y, g);
			atomicAdd(&giCandidateOut[s.pixelIndex].radiance.z, b);
			return;
		}

		// "float maxcomponentvalue" firefly clamp - see
		// GpuCameraParams::maxComponentValue's own comment (optix_types.h)
		// for why this is a per-contribution clamp, not a true
		// per-sample one.
		const float m = fmaxf(r, fmaxf(g, b));
		if (maxComponentValue > 0.0f && m > maxComponentValue) {
			const float s2 = maxComponentValue / m;
			r *= s2; g *= s2; b *= s2;
		}
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
//   Divides accumulated radiance by this pixel's own filter-weight sum
//   (pbrt-v4 film reconstruction formula: rgbSum / weightSum) - see
//   generate_camera_rays's own filter_w comment. Replaces the old flat
//   1/samplesPerPixel box-average division; every sample already carries
//   its own filter weight folded into its throughput, so this is just the
//   matching per-pixel divisor.
// ============================================================================

extern "C" __global__ void normalize_framebuffer(
	float3*      framebuffer,
	unsigned int numPixels,
	const float* weightBuffer
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	float w = weightBuffer[idx];
	framebuffer[idx] = (w > 0.0f) ? (framebuffer[idx] * (1.0f / w)) : make_float3(0.0f, 0.0f, 0.0f);
}

// Denoiser guide-layer AOV normalization (--denoise only, null buffers
// otherwise - see evaluate_materials()'s own accumulation comment). Plain
// arithmetic mean over samplesPerPixel, NOT filter-weighted like
// normalize_framebuffer() above - matches the recursive backend's own
// `albedo_sum / samplesPerPixel` (optix_raygen.h), which also averages
// unweighted by the reconstruction filter.
extern "C" __global__ void normalize_aov_buffers(
	float3*      albedoBuffer,
	float3*      normalBuffer,
	unsigned int numPixels,
	unsigned int samplesPerPixel
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	if (samplesPerPixel == 0) {
		albedoBuffer[idx] = make_float3(0.0f, 0.0f, 0.0f);
		normalBuffer[idx] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}
	float inv = 1.0f / float(samplesPerPixel);
	albedoBuffer[idx] = albedoBuffer[idx] * inv;
	normalBuffer[idx] = normalBuffer[idx] * inv;
}
