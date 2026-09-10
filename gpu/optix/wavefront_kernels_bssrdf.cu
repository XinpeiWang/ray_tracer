// wavefront_kernels_bssrdf.cu
// CUDA compute kernel: resolve_bssrdf_exit (Kernel 2b of the original
// wavefront_kernels.cu) - consumes the BSSRDF probe walk's exit point and
// resumes the path as a NormalizedFresnel bounce. Split out for the same
// independent-recompilation reason as the other wavefront_kernels_*.cu
// files - see wavefront_device_helpers.h's own header comment for where
// wf_finish_material_scatter/wf_lift_rgb_to_spectrum now live.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

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
	const DiskData*     disks,
	const CylinderData* cylinders,
	const MaterialData* materials,
	const int*          lightIndices,
	const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	// Needed only for a textured (pbrt AreaLightSource "filename") NEE
	// target - see wf_finish_material_scatter()'s own comment.
	const TextureData* textures,
	const unsigned char* texturePixels,
	float3 skyColor,
	float shadowRayEpsilon,
	GpuSkyDistribution skyDist,
	GpuPortalLight portalLight,
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h).
	float maxComponentValue,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. A BSSRDF exit can
	// land at depth==1 for a pixel whose primary hit was GI-eligible
	// (BssrdfExitWorkItem's own depth-propagation comment - a BSSRDF entry
	// hit does NOT itself become a GI candidate, since it's dispatched here
	// as MaterialType::NormalizedFresnel/matIdx=-1 below, never Lambertian -
	// but its depth==1 NEE still needs to reach giCandidateOut when the hit
	// that led here originated from a Lambertian x0). nullptr for batch/
	// offline rendering, same opt-in pattern as every other ReSTIR parameter.
	GpuGiOriginContext* giOriginContext,
	GpuGiSample* giCandidateOut
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
			auto xyz = SampledSpectrumToXYZ(radiance * item.filterWeight, swl, d_cie_x, d_cie_y, d_cie_z,
											kDevCIEMin, kDevCIENSamples);
			float r, g, b;
			wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
			// "float maxcomponentvalue" firefly clamp - see
			// GpuCameraParams::maxComponentValue's own comment (optix_types.h)
			// and evaluate_materials's own addToFramebuffer comment for why
			// this is a per-contribution clamp, not a true per-sample one.
			const float m = fmaxf(r, fmaxf(g, b));
			if (maxComponentValue > 0.0f && m > maxComponentValue) {
				const float s = maxComponentValue / m;
				r *= s; g *= s; b *= s;
			}
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
		// do_regularize is inert here regardless of value: is_specular=false
		// below already forces next.any_nonspecular=1 unconditionally, and
		// NormalizedFresnel never reaches glossy_isType (see BssrdfProbeWorkItem
		// any_nonspecular's own comment, wavefront_types.h, for why this item
		// doesn't carry a real one through).
		/*do_regularize=*/false,
		/*glossyAlpha=*/0.0f, /*glossyAlphaV=*/0.0f, // NormalizedFresnel never reaches glossy_isType
		// item.etaScale, unchanged - the entry interface's transmission
		// already folded its eta^2 in (see BssrdfProbeWorkItem::etaScale's
		// own comment); this exit-surface NormalizedFresnel shading doesn't
		// add another factor, matching CPU camera.h's entry_eta convention.
		item.etaScale,
		item.filterWeight,
		maxComponentValue,
		item.exitNormal, item.exitPos,
		/*dpdu=*/make_float3(0.0f, 0.0f, 0.0f),  // never reaches glossy_isType - see glossyAlpha comment above
		seed,
		weightedThroughput, radiance, swl, attenuation, scattered_dir,
		/*is_specular=*/false, brdf_pdf_override,
		/*phaseWo=*/make_float3(0.0f, 0.0f, 0.0f), /*phaseG=*/0.0f,
		item.pixelIndex, item.depth,
		spheres, quads, triangles, bilinearPatches, disks, cylinders, materials,
		lightIndices, lightKinds, aliasTable, numLights,
		punctualLights, numPunctualLights,
		skyColor, shadow_eps, skyDist, portalLight,
		shadowQueue, nextRayQueue, framebuffer,
		textures, texturePixels, /*uv_u=*/0.0f, /*uv_v=*/0.0f, item.time,
		// ReSTIR DI stays excluded here exactly as before (nullptr/default
		// context - see wf_finish_material_scatter's own restirReservoirs
		// parameter comment for why: a BSSRDF exit is never itself a
		// depth==0 primary hit's own resampling target). Explicit rather
		// than omitted because C++ default arguments can only be dropped
		// from the END of a call's argument list, and giOriginContext/
		// giCandidateOut below need to be supplied.
		/*restirReservoirs=*/nullptr, /*restirCtx=*/GpuRestirTemporalContext{},
		giOriginContext, giCandidateOut);
}
