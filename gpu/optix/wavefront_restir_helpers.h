#pragma once
// wavefront_restir_helpers.h -- ReSTIR DI (Bitterli, Wyman, Pharr, Shirley,
// Lefohn, Salvi, "Spatiotemporal Reservoir Resampling for Real-Time Ray
// Tracing with Dynamic Direct Lighting", SIGGRAPH 2020) GPU-native reservoir
// primitives for the wavefront path tracer's real-time Live Preview NEE path.
//
// Mirrors src/shared/restir.h's RIS/reservoir math (RestirCandidate<T>,
// Reservoir<T>, ris_fill, reservoir_ucw - see that header for the reference
// formulas and its own test coverage) but is NOT a template instantiation of
// it: a stored candidate here must carry the actual resolved light sample
// (world point/normal/UV, not just an opaque index) so it can be re-shaded
// from a DIFFERENT pixel's shading point later, for temporal/spatial reuse.
// float throughout (not restir.h's double), matching every other wavefront/
// device numeric type.
//
// Included by wavefront_device_helpers.h AFTER the wf_sample_*_light/wf_dc_*
// helpers (wf_reevaluate_light_geometry below calls them) and after
// src/shared/bilinear_patch.h (already included earlier in that file, for
// blp_point() - a shared, non-OptiX-module-specific header, unlike the
// wf_dc_*/dc_* split, so no wavefront-native duplicate of it is needed here).
//
// restir.h's temporal_update()/spatial_merge() explicitly do NOT implement
// the reweighting needed for unbiased reuse (that header's own TODO comment:
// "for full unbiasedness, re-evaluate p_hat for each neighbor's sample at
// this pixel's context and call reservoir_ucw() with the corrected value").
// restir_reservoir_combine() below IS that missing piece: every reuse (this
// file's callers, wavefront_kernels_restir.cu) re-evaluates the OTHER
// reservoir's stored sample's target function fresh, at the CURRENT pixel's
// own context, before folding it in - never reusing a stale p_hat/W computed
// at a different shading point. Because GpuLightSample stores the sample's
// exact world-space point (not a reparameterized direction or a UV to
// re-intersect), re-evaluating the target function at a new origin already
// yields the fully-corrected solid-angle pdf/cosine/distance for that origin
// with no separate multiplicative Jacobian required - see
// wf_reevaluate_light_geometry's own comment. wf_restir_jacobian() further
// down is a real (not decorative) geometric-ratio guard used by spatial reuse
// to reject neighbor samples whose configuration is too different between the
// two shading points to reuse with acceptable variance (see its own comment
// for why this is a robustness guard, not a second correction on top of the
// re-evaluation above - applying both would double-count the same geometry).

#include "optix_types.h"
#include "optix_math_helpers.h"   // cross()/dot()/length()/normalize()

// Candidates resampled per primary-hit pixel per frame (Bitterli 2020's M) -
// tunable; 8 balances RIS's noise reduction against the extra alias-table
// draws/texture lookups this costs per pixel every frame.
constexpr int kRestirCandidateCount = 8;

// Temporal reuse's M-clamp (restir.h's own max_M concept) - caps how many
// candidates' worth of history a reservoir can claim to represent, so a
// long-lived reservoir carried across many static frames doesn't drown out
// fresh candidates the instant the scene/camera starts changing again.
constexpr int kRestirTemporalMaxM = 20;

// ===========================================================================
// GpuLightSample / GpuReservoir - defined in optix_types.h, not here (see
// that header's own comment on why: this file also carries __device__-only
// RIS math that must not leak into the plain host/extern-"C" boundary files
// that only need the reservoir's storage shape). optix_types.h is already
// included above.
// ===========================================================================
// Core RIS math - mirrors src/shared/restir.h's ris_add/reservoir_ucw exactly
// (same formulas, ported to GpuReservoir/float). Pure float in/out, no
// CUDA-only types, so it is callable and unit-testable from plain host C++ as
// well as device code.
// ===========================================================================

// Streams one candidate into `r` via weighted reservoir sampling (Bitterli
// eq. 5's streaming RIS update). `risWeight` is target_pdf/source_pdf for
// this candidate; `candidateM` is how many underlying samples it represents
// (1 for a freshly-drawn candidate, an existing reservoir's own M when
// combining reservoirs - see restir_reservoir_combine); `candidatePHat` is
// the candidate's own target-function value, stored on acceptance so the
// caller can later call restir_finalize() without re-evaluating it again.
// `rand01` MUST be a fresh uniform random in [0,1) per call. Returns true iff
// `candidate` became (or stayed) the reservoir's selected sample.
CPU_GPU inline bool restir_reservoir_add(GpuReservoir& r, const GpuLightSample& candidate,
										  float risWeight, int candidateM, float candidatePHat,
										  float rand01) {
	r.weightSum += risWeight;
	r.M += candidateM;
	if (r.weightSum <= 0.0f || risWeight <= 0.0f) return false;
	if (rand01 * r.weightSum < risWeight) {
		r.sample = candidate;
		r.pHat = candidatePHat;
		return true;
	}
	return false;
}

// Unbiased contribution weight - Bitterli eq. 6: W = w_sum / (M * pHat).
// Mirrors restir.h's reservoir_ucw exactly (0 when M*pHat is non-positive).
CPU_GPU inline float restir_reservoir_ucw(float weightSum, int M, float pHat) {
	float denom = float(M) * pHat;
	return (denom > 0.0f) ? (weightSum / denom) : 0.0f;
}

CPU_GPU inline void restir_finalize(GpuReservoir& r) {
	r.W = restir_reservoir_ucw(r.weightSum, r.M, r.pHat);
}

// Combines an already-formed reservoir `other` into `dst`, both understood to
// describe the SAME pixel's context (`dst` is that pixel's own running
// reservoir; `other` is a temporal or spatial neighbor's reservoir being
// reused there). `otherPHatAtDstContext` MUST be `other.sample`'s target
// function freshly evaluated at `dst`'s own shading point (never `other`'s
// stored `pHat`, which was evaluated at `other`'s original pixel) - this is
// exactly restir.h's documented missing piece for full unbiasedness. Treats
// `other` as one weighted candidate of weight `otherPHatAtDstContext *
// other.W * other.M`, carrying `other.M` samples - the standard reservoir-
// combine identity (Bitterli Algorithm 4): reusing a whole reservoir's UCW
// as a single RIS candidate weight is valid because W is itself an unbiased
// estimator of 1/pHat integrated over that reservoir's own M candidates.
CPU_GPU inline bool restir_reservoir_combine(GpuReservoir& dst, const GpuReservoir& other,
											  float otherPHatAtDstContext, float rand01) {
	if (!other.valid() || other.M <= 0) return false;
	float w = otherPHatAtDstContext * other.W * float(other.M);
	return restir_reservoir_add(dst, other.sample, w, other.M, otherPHatAtDstContext, rand01);
}

// ===========================================================================
// Shape-aware re-evaluation - given an ALREADY-KNOWN light sample, recompute
// direction/distance/solid-angle pdf from an arbitrary NEW query origin,
// without redrawing and without a search/re-intersection: `s.point` is
// already an exact point on the light (established at generation time), so
// the direction from any origin to it trivially reaches that same point -
// unlike optix_disk_cylinder_helpers.h's dc_pdf_disk/dc_pdf_cylinder, which
// must re-intersect a BSDF-sampled ray because the recursive backend's MIS
// path only ever has a direction, not an already-known point. This is also
// why no separate Jacobian multiply is needed here for correctness: cosine
// and distance are recomputed fresh at the new origin, which already IS the
// full geometric correction ReSTIR's unbiased reuse requires (restir.h's own
// TODO wording: "re-evaluate p_hat ... at this pixel's context").
//
// Returns false (out_geom_pdf left at 0) if the sample is degenerate from
// this origin (grazing angle, coincident points, or - Sphere only - the
// sampled direction falls outside the new origin's sampling cone).
// ===========================================================================
__device__ __forceinline__ bool wf_reevaluate_light_geometry(
		const GpuLightSample& s, const float3& origin,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
		float3& out_dir, float& out_dist, float& out_geom_pdf) {
	float3 toSample = s.point - origin;
	out_dist = length(toSample);
	if (out_dist < 1e-6f) {
		out_dir = make_float3(0.0f, 0.0f, 1.0f);
		out_geom_pdf = 0.0f;
		return false;
	}
	out_dir = toSample / out_dist;

	if (s.kind == GpuLightKind::Sphere) {
		// Sphere's cone/inside pdf is already a solid-angle density (see
		// wf_sample_sphere_light) - no separate cosine/area conversion.
		// Motion blur is intentionally ignored here (time=0, sph.center) -
		// Live Preview's real-time-only scope never exercises a moving
		// emissive sphere with camera-shutter motion blur at the same time,
		// so this is an accepted simplification specific to ReSTIR reuse,
		// not a general-purpose sphere-light limitation.
		const SphereData& sph = spheres[s.primIdx];
		const float3 center = sph.center;
		const float3 toC = center - origin;
		const float distC = length(toC);
		const float r = sph.radius;
		if (distC <= r) {
			out_geom_pdf = 1.0f / (4.0f * 3.14159265f * r * r);
			return true;
		}
		const float cosMax = sqrtf(fmaxf(0.0f, 1.0f - (r * r) / (distC * distC)));
		const float3 w = toC / distC;
		const float cosTheta = dot(out_dir, w);
		if (cosTheta < cosMax) { out_geom_pdf = 0.0f; return false; }
		const float solid = 2.0f * 3.14159265f * (1.0f - cosMax);
		out_geom_pdf = (solid > 1e-10f) ? (1.0f / solid) : 0.0f;
		return out_geom_pdf > 0.0f;
	}

	float area_pdf = 0.0f;
	if (s.kind == GpuLightKind::Quad) {
		const QuadData& q = quads[s.primIdx];
		const float area = length(cross(q.u, q.v));
		area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;
	} else if (s.kind == GpuLightKind::Triangle) {
		const TriangleData& tri = triangles[s.primIdx];
		const float3 e1 = tri.p1 - tri.p0, e2 = tri.p2 - tri.p0;
		const float twiceArea = length(cross(e1, e2));
		area_pdf = (twiceArea > 1e-12f) ? (2.0f / twiceArea) : 0.0f;
	} else if (s.kind == GpuLightKind::Disk) {
		const float area = wf_dc_area_disk(disks[s.primIdx]);
		area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;
	} else if (s.kind == GpuLightKind::Cylinder) {
		const float area = wf_dc_area_cylinder(cylinders[s.primIdx]);
		area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;
	} else {
		// BilinearPatch: area pdf is the LOCAL Jacobian |dpdu x dpdv| at the
		// sample's exact (u,v) - not constant across the patch (this
		// header's own file comment / wf_sample_bilinear_patch_light's
		// comment) - reconstructed directly from the stored (sampleU,
		// sampleV) rather than re-intersecting to find it (blp_pdf_wi's
		// approach), for the same "point already known" reason as every
		// other shape here.
		const BilinearPatchData& bp = bilinearPatches[s.primIdx];
		const float p00[3] = { bp.p00.x, bp.p00.y, bp.p00.z };
		const float p10[3] = { bp.p10.x, bp.p10.y, bp.p10.z };
		const float p01[3] = { bp.p01.x, bp.p01.y, bp.p01.z };
		const float p11[3] = { bp.p11.x, bp.p11.y, bp.p11.z };
		float hp[3], dpdu[3], dpdv[3];
		blp_point(p00, p10, p01, p11, s.sampleU, s.sampleV, hp, dpdu, dpdv);
		float cr[3];
		blp_detail::cross(dpdu, dpdv, cr);
		const float dA = blp_detail::length(cr);
		area_pdf = (dA > 1e-12f) ? (1.0f / dA) : 0.0f;
	}

	const float cosine = fabsf(dot(out_dir, s.normal));
	if (cosine < 1e-6f || area_pdf <= 0.0f) { out_geom_pdf = 0.0f; return false; }
	out_geom_pdf = area_pdf * out_dist * out_dist / cosine;
	return true;
}

// Geometric-ratio robustness guard for spatial reuse (Bitterli 2020 eq. 11's
// cos/dist^2 ratio between the two shading points' view of the same sampled
// light point) - NOT an additional multiplicative correction on top of
// wf_reevaluate_light_geometry's fresh cosine/distance recompute (applying
// both would double the same geometric term; see this file's header comment
// for why the re-evaluation alone is already the full unbiased correction
// this codebase's restir.h identifies as missing). Used only to REJECT a
// spatial neighbor whose light-sample geometry differs too drastically
// between the current and neighbor shading points (e.g. the sample is
// steeply grazing from one point but not the other), the same variance-
// control role normal/depth neighbor rejection already plays - a large ratio
// signals the neighbor's sample is a poor, high-variance fit for the current
// pixel, not that it is biased to include.
__device__ __forceinline__ float wf_restir_jacobian(const float3& currentDir, float currentDist,
													  const float3& neighborDir, float neighborDist,
													  const float3& lightNormal) {
	const float cosCurrent = fabsf(dot(currentDir, lightNormal));
	const float cosNeighbor = fabsf(dot(neighborDir, lightNormal));
	if (cosNeighbor < 1e-6f || neighborDist < 1e-6f) return 0.0f;
	const float numerator = cosCurrent * neighborDist * neighborDist;
	const float denominator = cosNeighbor * currentDist * currentDist;
	return (denominator > 1e-12f) ? (numerator / denominator) : 0.0f;
}

// ===========================================================================
// RIS candidate generation - the geometry/raw-emission half of the existing
// single-draw NEE dispatch (wf_finish_material_scatter's own alias-table-
// draw + per-shape-sample + twoSided-gate block), factored out here so RIS
// can call it M times without duplicating that per-shape switch. Kept
// deliberately free of the spectral uplift (liftEmission/geoAndGate's D65
// polynomial fit) that block also does - that only needs to run once, for
// the reservoir's FINAL winning sample, not for all M candidates - so this
// returns a raw (pre-uplift) RGB emission, already twoSided-gated to black
// when the sampled point faces away on a one-sided light. Returns false only
// when there is nothing to sample (numLights==0 or no alias table).
// ===========================================================================
__device__ __forceinline__ int wf_light_material_index(
		GpuLightKind kind, int primIdx,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders) {
	if (kind == GpuLightKind::Sphere) return spheres[primIdx].materialIdx;
	if (kind == GpuLightKind::Triangle) return triangles[primIdx].materialIdx;
	if (kind == GpuLightKind::BilinearPatch) return bilinearPatches[primIdx].materialIdx;
	if (kind == GpuLightKind::Disk) return disks[primIdx].materialIdx;
	if (kind == GpuLightKind::Cylinder) return cylinders[primIdx].materialIdx;
	return quads[primIdx].materialIdx;
}

// Raw (pre-spectral-uplift) emission at an already-known light sample, gated
// to black by the material's twoSided flag exactly like the original inline
// NEE dispatch's geoAndGate lambda - shared by wf_generate_restir_candidate
// (below) and wf_finish_material_scatter's own RIS final-shading step (the
// winning candidate needs this looked up exactly once more, since only the
// SELECTED sample's actual radiance ever reaches the framebuffer - see this
// file's header comment on M-candidate resampling not needing the spectral
// uplift for every candidate).
__device__ __forceinline__ float3 wf_light_raw_emission(
		const GpuLightSample& s, const float3& dirFromQuery,
		const MaterialData* materials,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
		const TextureData* textures, const unsigned char* texturePixels) {
	const int matIdx = wf_light_material_index(s.kind, s.primIdx, spheres, quads, triangles, bilinearPatches, disks, cylinders);
	const MaterialData& lm = materials[matIdx];
	float3 raw = (lm.textureIdx >= 0)
		? wf_sample_texture(textures, texturePixels, lm.textureIdx, s.sampleU, s.sampleV, s.point)
		: lm.emission;
	if (lm.textureIdx >= 0) { raw.x *= lm.emissionScale; raw.y *= lm.emissionScale; raw.z *= lm.emissionScale; }
	if (!lm.twoSided && dot(dirFromQuery, s.normal) >= 0.0f) raw = make_float3(0.0f, 0.0f, 0.0f);
	return raw;
}

__device__ __forceinline__ bool wf_generate_restir_candidate(
		const float3& hit, unsigned int& seed, float time,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
		const MaterialData* materials, const int* lightIndices, const GpuLightKind* lightKinds,
		const GpuAliasEntry* aliasTable, unsigned int numLights,
		const TextureData* textures, const unsigned char* texturePixels,
		GpuLightSample& out_sample, float3& out_dir, float& out_maxDist,
		float& out_lightPdf, float3& out_rawEmission) {
	if (numLights == 0 || !aliasTable) return false;

	int slot = int(wf_rand(seed) * float(numLights));
	if (slot >= (int)numLights) slot = (int)numLights - 1;
	const GpuAliasEntry& entry = aliasTable[slot];
	int light_idx = (wf_rand(seed) < entry.q) ? slot : entry.alias;
	const float selection_pdf = aliasTable[light_idx].pdf;

	const int prim_idx = lightIndices[light_idx];
	const GpuLightKind kind = lightKinds[light_idx];

	float geom_pdf = 0.0f;
	float3 sampleNormal = make_float3(0.0f, 0.0f, 1.0f);
	float su = 0.0f, sv = 0.0f;

	if (kind == GpuLightKind::Sphere) {
		const SphereData& s = spheres[prim_idx];
		out_dir = wf_sample_sphere_light(s, hit, seed, geom_pdf, out_maxDist, su, sv, sampleNormal, time);
	} else if (kind == GpuLightKind::Triangle) {
		const TriangleData& tri = triangles[prim_idx];
		out_dir = wf_sample_triangle_light(tri, hit, seed, geom_pdf, out_maxDist, su, sv, sampleNormal);
	} else if (kind == GpuLightKind::BilinearPatch) {
		const BilinearPatchData& bp = bilinearPatches[prim_idx];
		out_dir = wf_sample_bilinear_patch_light(bp, hit, seed, geom_pdf, out_maxDist, su, sv, sampleNormal);
	} else if (kind == GpuLightKind::Disk) {
		const DiskData& d = disks[prim_idx];
		out_dir = wf_sample_disk_light(d, hit, seed, geom_pdf, out_maxDist, su, sv, sampleNormal);
	} else if (kind == GpuLightKind::Cylinder) {
		const CylinderData& c = cylinders[prim_idx];
		out_dir = wf_sample_cylinder_light(c, hit, seed, geom_pdf, out_maxDist, su, sv, sampleNormal);
	} else {
		const QuadData& q = quads[prim_idx];
		out_dir = wf_sample_quad_light(q, hit, seed, geom_pdf, out_maxDist, su, sv);
		sampleNormal = q.normal;
	}

	out_sample.lightIdx = light_idx;
	out_sample.kind = kind;
	out_sample.primIdx = prim_idx;
	out_sample.sampleU = su;
	out_sample.sampleV = sv;
	out_sample.point = hit + out_dir * out_maxDist;
	out_sample.normal = sampleNormal;

	out_lightPdf = selection_pdf * geom_pdf;
	out_rawEmission = wf_light_raw_emission(out_sample, out_dir, materials, spheres, quads, triangles,
											 bilinearPatches, disks, cylinders, textures, texturePixels);
	return true;
}
