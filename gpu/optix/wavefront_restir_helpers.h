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
// Pure RIS/reservoir-combine math (restir_reservoir_add/ucw/combine/finalize,
// wf_restir_target_proxy, wf_restir_jacobian) - split out into its own
// header purely so it can be unit-tested from a plain host build without
// this file's own wf_sample_*_light/wf_rand dependency on being included
// from within wavefront_device_helpers.h - see that header's own comment.
#include "wavefront_restir_math.h"

// Candidates resampled per primary-hit pixel per frame (Bitterli 2020's M) -
// tunable; 8 balances RIS's noise reduction against the extra alias-table
// draws/texture lookups this costs per pixel every frame.
constexpr int kRestirCandidateCount = 8;

// Temporal reuse's M-clamp (restir.h's own max_M concept) - caps how many
// candidates' worth of history a reservoir can claim to represent, so a
// long-lived reservoir carried across many static frames doesn't drown out
// fresh candidates the instant the scene/camera starts changing again.
constexpr int kRestirTemporalMaxM = 20;

// Spatial reuse's own M-clamp. restir.h's own spatial_merge() reference
// comment suggests 500 for this same role, but that reference assumes a
// full unbiased visibility-reuse treatment (re-tracing/accounting for
// occlusion during resampling itself, not just at final shading); this
// codebase's own restir_reservoir_combine() re-evaluates a neighbor's target
// function UNSHADOWED (the actual traced shadow ray only happens once, for
// the FINAL winning sample - wf_finish_material_scatter's own comment).
// Matching temporal reuse's own cap (20) keeps a single spatial combine from
// injecting disproportionate confidence relative to what a fresh RIS draw
// earns. (An earlier version of this comment attributed a since-fixed
// whole-image-mean blowup to this cap being too permissive at 500 - that
// diagnosis didn't hold up: dropping it to 20 alone made no measurable
// difference, confirmed by an identical reproduced frame before and after.
// The real cause was restir_reservoir_combine's own caller code clamping the
// COMBINED M after already using the uncapped value in the weight formula -
// see wf_restir_temporal_combine's and this file's neighbor-clamp's own
// comments - now fixed at the clamp site itself, independent of this
// constant's value.)
constexpr int kRestirSpatialMaxM = kRestirTemporalMaxM;

// Spatial reuse's neighbor sampling - kRestirSpatialNeighbors candidate
// pixels drawn uniformly from a disk of this pixel radius (Bitterli 2020's
// own "small screen-space radius" recommendation; RTXDI's public reference
// implementation uses a similar single-digit neighbor count and a
// few-dozen-pixel radius).
constexpr int kRestirSpatialNeighbors = 4;
constexpr float kRestirSpatialRadiusPixels = 20.0f;

// Spatial reuse's neighbor-rejection threshold - reject a neighbor whose
// shading normal has drifted more than ~25 degrees from this pixel's own
// (cos(25 deg) ~= 0.9), the same normal-similarity heuristic real-time
// denoisers/ReSTIR implementations use to avoid blending across a geometric
// edge (two different surfaces that happen to land near each other on
// screen).
constexpr float kRestirSpatialNormalCosThreshold = 0.9f;

// ===========================================================================
// GpuLightSample / GpuReservoir - defined in optix_types.h, not here (see
// that header's own comment on why: this file also carries __device__-only
// RIS math that must not leak into the plain host/extern-"C" boundary files
// that only need the reservoir's storage shape). optix_types.h is already
// included above. The core RIS math (restir_reservoir_add/ucw/combine/
// finalize) itself now lives in wavefront_restir_math.h (included above) -
// see that header's own comment for why.
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
//
// `time` is the shutter time to interpolate a MOVING sphere's center by
// (SphereData::center/center1, matching wf_sample_sphere_light's own
// convention) - only ever exact for the immediate post-RIS-selection
// re-evaluation (wf_finish_material_scatter), which re-evaluates at the SAME
// hit_point/time the candidate was just generated from. Temporal reuse
// (wf_restir_temporal_combine) and spatial reuse (wavefront_kernels_restir.cu)
// re-evaluate a sample from a DIFFERENT pixel/frame with no shutter time of
// its own available (no per-pixel time buffer exists), so both pass 0.0f -
// a known, narrower limitation than a blanket "never happens" claim: a
// moving emissive sphere's cross-frame/cross-pixel reuse can still use the
// wrong (static) center, but a fresh per-frame RIS selection - which runs
// unconditionally every frame even before any reuse history exists - no
// longer does.
// ===========================================================================
__device__ __forceinline__ bool wf_reevaluate_light_geometry(
		const GpuLightSample& s, const float3& origin, float time,
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
		// Time-interpolated center, matching wf_sample_sphere_light's own
		// convention exactly - see this function's own header comment on
		// which callers can and can't supply the real shutter time.
		const SphereData& sph = spheres[s.primIdx];
		const float3 center = make_float3(
			sph.center.x + time * (sph.center1.x - sph.center.x),
			sph.center.y + time * (sph.center1.y - sph.center.y),
			sph.center.z + time * (sph.center1.z - sph.center.z));
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
		// Degenerate near-zero solid angle: fall back to pdf=1.0f, matching
		// wf_sample_sphere_light's OWN fallback exactly (not 0.0f/failure) -
		// these two formulas must agree, since a candidate generated with
		// the generation-time fallback is later re-evaluated right here for
		// its actual shading contribution; disagreeing fallbacks meant a
		// candidate could win RIS selection (using pdf=1.0f, generation
		// time) and then have its contribution silently dropped moments
		// later (using pdf=0.0f/failure, this function, previously) for the
		// exact same degenerate geometric configuration.
		out_geom_pdf = (solid > 1e-10f) ? (1.0f / solid) : 1.0f;
		return true;
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

// wf_restir_jacobian() itself now lives in wavefront_restir_math.h (included
// above) - see that header's own comment.

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
	// Back-face gate FIRST - a real early-out, not just a post-hoc zeroing:
	// skips the texture fetch below entirely for every back-facing sample of
	// a one-sided light, instead of paying for it and discarding the result.
	if (!lm.twoSided && dot(dirFromQuery, s.normal) >= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
	float3 raw = (lm.textureIdx >= 0)
		? wf_sample_texture(textures, texturePixels, lm.textureIdx, s.sampleU, s.sampleV, s.point)
		: lm.emission;
	if (lm.textureIdx >= 0) { raw.x *= lm.emissionScale; raw.y *= lm.emissionScale; raw.z *= lm.emissionScale; }
	return raw;
}

// Bounding-cone light BVH (spatial+power selection) for wavefront's own
// ReSTIR DI candidate generation / classic NEE - a hand-duplicated twin of
// gpu_light_bvh_sample_index() (optix_device_helpers_lighting.h), reading
// the tree via explicit parameters rather than a __constant__ params global.
// evaluate_materials/evaluate_materials_simple (wavefront_kernels_materials*.
// cu) are plain cudaLaunchKernel-style compute kernels with their own
// explicit parameter lists, not OptiX raygen launches - they never read
// wf_params (see either kernel's own "not go through the wf_params/lp
// raygen-launch-params path" comment), so the tree has to arrive as a real
// kernel parameter, threaded all the way from WavefrontPathTracer::
// setLightBvh()/render() down through wf_finish_material_scatter() and
// wf_generate_restir_candidate() below.
//
// This is GPU-recursive's own light BVH tree, reused as-is (OptiXRenderer::
// buildScene() builds it once; WavefrontPathTracer::setLightBvh() just
// points at the same device buffers - no separate build/upload) - but unlike
// GPU-recursive, wavefront's separately-compiled, shallower kernels do NOT
// exhibit the NVCC device-execution divergence that makes GPU-recursive's
// own gpu_light_bvh_sample_index() unusable (see that function's own KNOWN
// UNRESOLVED BUG comment for the full investigation and why this backend is
// confirmed clean: instrumented counters showed ~99% success against a real
// 23-node/12-light tree, the same failure signature GPU-recursive hits 100%
// of the time on).
__device__ __forceinline__ GpuLightBvhSample wf_light_bvh_sample_index(
	float px, float py, float pz, float u,
	const LightBVHNode* lightBvhNodes, int lightBvhNodeCount, unsigned int numLights,
	float allBMinX, float allBMinY, float allBMinZ,
	float allBMaxX, float allBMaxY, float allBMaxZ)
{
	if (lightBvhNodeCount <= 0) return GpuLightBvhSample{-1, 0.f};
	int nodeIndex = 0;
	float pmf = 1.f;
	u = fminf(u, 1.f - 1e-7f);
	while (true) {
		if (nodeIndex < 0 || nodeIndex >= lightBvhNodeCount) {
			return GpuLightBvhSample{-1, 0.f};
		}
		const LightBVHNode& node = lightBvhNodes[nodeIndex];
		if (!node.isLeaf) {
			const int c1Index = (int)node.childOrLightIndex;
			if (c1Index <= nodeIndex + 1 || c1Index >= lightBvhNodeCount) {
				return GpuLightBvhSample{-1, 0.f};
			}
			const LightBVHNode& c0 = lightBvhNodes[nodeIndex + 1];
			const LightBVHNode& c1 = lightBvhNodes[node.childOrLightIndex];
			float ci0 = c0.lightBounds.Importance(px,py,pz, 0.f,0.f,0.f,
				allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
			float ci1 = c1.lightBounds.Importance(px,py,pz, 0.f,0.f,0.f,
				allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
			if (ci0 == 0.f && ci1 == 0.f) return GpuLightBvhSample{-1, 0.f};
			float sum = ci0 + ci1;
			float nodePMF; int child;
			if (u < ci0 / sum) { child = 0; nodePMF = ci0 / sum; u = u / nodePMF; }
			else { child = 1; nodePMF = ci1 / sum; u = (u - ci0/sum) / nodePMF; }
			u = fminf(u, 1.f - 1e-7f);
			pmf *= nodePMF;
			nodeIndex = (child == 0) ? (nodeIndex + 1) : (int)node.childOrLightIndex;
		} else {
			if (node.childOrLightIndex >= numLights) {
				return GpuLightBvhSample{-1, 0.f};
			}
			return GpuLightBvhSample{(int)node.childOrLightIndex, pmf};
		}
	}
}

// wf_light_bvh_pmf: wavefront's twin of gpu_light_bvh_pmf() (optix_device_
// helpers_lighting.h - see that function's own comment for the full
// rationale). Replays the bit-trail for `lightIndex` to recompute its
// selection PMF at THIS shading point (position-dependent, unlike the alias
// table's fixed pdf) - needed whenever a light-BVH-selected candidate's
// winning sample is re-used from somewhere other than the draw that produced
// it (ReSTIR temporal/spatial reuse, or a BSDF-sampled light hit's own MIS
// weight), since the pmf that draw actually used isn't otherwise recoverable
// without redoing the stochastic descent. Returns 0 if no light BVH was
// built, or if every ancestor's combined importance was zero (can't happen
// for a real bit-trail from a light actually in the tree, but matches
// gpu_light_bvh_pmf()'s own defensive return).
__device__ __forceinline__ float wf_light_bvh_pmf(
	float px, float py, float pz, int lightIndex, unsigned int numLights,
	const LightBVHNode* lightBvhNodes, const unsigned int* lightBvhBitTrail, int lightBvhNodeCount,
	float allBMinX, float allBMinY, float allBMinZ,
	float allBMaxX, float allBMaxY, float allBMaxZ)
{
	if (lightBvhNodeCount <= 0 || !lightBvhBitTrail) return 0.f;
	if (lightIndex < 0 || (unsigned int)lightIndex >= numLights) return 0.f;
	unsigned int bitTrail = lightBvhBitTrail[lightIndex];
	float pmf = 1.f;
	int nodeIndex = 0;
	while (true) {
		if (nodeIndex < 0 || nodeIndex >= lightBvhNodeCount) return 0.f;
		const LightBVHNode& node = lightBvhNodes[nodeIndex];
		if (node.isLeaf) return pmf;
		const int c1Index = (int)node.childOrLightIndex;
		if (c1Index <= nodeIndex + 1 || c1Index >= lightBvhNodeCount) return 0.f;
		const LightBVHNode& c0 = lightBvhNodes[nodeIndex + 1];
		const LightBVHNode& c1 = lightBvhNodes[node.childOrLightIndex];
		float ci0 = c0.lightBounds.Importance(px,py,pz, 0.f,0.f,0.f,
			allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
		float ci1 = c1.lightBounds.Importance(px,py,pz, 0.f,0.f,0.f,
			allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
		float sum = ci0 + ci1;
		if (sum == 0.f) return 0.f;
		int branch = (int)(bitTrail & 1u);
		pmf *= (branch == 0 ? ci0 : ci1) / sum;
		nodeIndex = (branch == 0) ? (nodeIndex + 1) : (int)node.childOrLightIndex;
		bitTrail >>= 1;
	}
}

__device__ __forceinline__ bool wf_generate_restir_candidate(
		const float3& hit, unsigned int& seed, float time,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
		const MaterialData* materials, const int* lightIndices, const GpuLightKind* lightKinds,
		const GpuAliasEntry* aliasTable, unsigned int numLights,
		const TextureData* textures, const unsigned char* texturePixels,
		GpuLightSample& out_sample, float3& out_dir, float& out_maxDist,
		float& out_lightPdf, float3& out_rawEmission,
		// See wf_light_bvh_sample_index()'s own comment.
		// lightBvh.nodeCount<=0 (the default) means "no light BVH built" -
		// falls straight through to the alias table below.
		WfLightBvhContext lightBvh = {}) {
	if (numLights == 0) return false;

	int light_idx;
	float selection_pdf;
	// Light BVH first (real spatial+power selection, position-dependent) -
	// see wf_light_bvh_sample_index()'s own comment. Falls back to the alias
	// table for any scene that didn't build a light BVH. A per-draw rejection
	// (importance genuinely zero at this hit point for every light the BVH
	// currently considers) returns false with NO alias-table fallback for
	// that same draw - see this function's callers (wf_finish_material_
	// scatter's RIS loop uses `continue`, not `break`, specifically because
	// of this: a BVH scene's occasional reject is a per-draw event, not a
	// "no lights in the scene" one).
	if (lightBvh.nodeCount > 0) {
		GpuLightBvhSample s = wf_light_bvh_sample_index(hit.x, hit.y, hit.z, wf_rand(seed),
			lightBvh.nodes, lightBvh.nodeCount, numLights,
			lightBvh.allBMinX, lightBvh.allBMinY, lightBvh.allBMinZ,
			lightBvh.allBMaxX, lightBvh.allBMaxY, lightBvh.allBMaxZ);
		if (s.lightIndex < 0) return false;
		light_idx = s.lightIndex;
		selection_pdf = s.pmf;
	} else if (aliasTable) {
		int slot = int(wf_rand(seed) * float(numLights));
		if (slot >= (int)numLights) slot = (int)numLights - 1;
		const GpuAliasEntry& entry = aliasTable[slot];
		light_idx = (wf_rand(seed) < entry.q) ? slot : entry.alias;
		selection_pdf = aliasTable[light_idx].pdf;
	} else {
		return false;
	}

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
	// Skip the emission lookup (a real texture fetch for a textured light)
	// entirely for a degenerate draw - the caller's own `candPdf <= 1e-6f`
	// check (wf_finish_material_scatter's RIS loop - see its own comment on
	// why 1e-6f, matching the classic NEE path's floor exactly, not a looser
	// value) would discard this candidate anyway, so there is nothing to
	// gain from fetching its emission first. Ordinary, not rare: a grazing-
	// angle or degenerate-sphere-cone draw happens routinely, and this
	// function runs kRestirCandidateCount times per pixel every frame.
	out_rawEmission = (out_lightPdf > 1e-6f)
		? wf_light_raw_emission(out_sample, out_dir, materials, spheres, quads, triangles,
								 bilinearPatches, disks, cylinders, textures, texturePixels)
		: make_float3(0.0f, 0.0f, 0.0f);
	return true;
}

// wf_restir_target_proxy() itself now lives in wavefront_restir_math.h
// (included above) - see that header's own comment.

// Device-only port of qt_gui/camera_math.h's projectToScreen() (see that
// function's own derivation comment - same algorithm, float/float3 instead
// of double/Vec3, since that header isn't CPU_GPU-tagged and duplicating a
// handful of vector ops is simpler than making a Qt-adjacent host header
// device-safe). Used by ReSTIR temporal reuse to find where a current-frame
// hit point would have landed in the PREVIOUS frame's camera.
struct WfScreenProjection {
	float s = 0.0f, t = 0.0f;
	bool inFront = false;
};

__device__ __forceinline__ WfScreenProjection wf_project_to_screen(const float3& worldPoint,
																	 const GpuReprojectBasis& basis) {
	const float3 toPoint = worldPoint - basis.origin;
	const float3 wScaled = basis.origin - basis.lowerLeftCorner
						  - basis.horizontal * 0.5f - basis.vertical * 0.5f;
	const float3 vCrossW = cross(basis.vertical, wScaled);
	const float denom = dot(basis.horizontal, vCrossW);
	if (fabsf(denom) < 1e-18f) return WfScreenProjection{0.0f, 0.0f, false};
	const float a = dot(toPoint, vCrossW) / denom;
	const float b = dot(basis.horizontal, cross(toPoint, wScaled)) / denom;
	const float c = dot(basis.horizontal, cross(basis.vertical, toPoint)) / denom;
	if (c >= 0.0f) return WfScreenProjection{0.0f, 0.0f, false};
	WfScreenProjection out;
	out.s = 0.5f - a / c;
	out.t = 0.5f - b / c;
	out.inFront = true;
	return out;
}

// Reprojects `currentPoint` into the previous frame's camera and returns the
// flat pixel index of a valid, non-disoccluded history entry there, or -1 if
// reprojection fails for any reason (behind the camera, off-screen, that
// pixel never wrote a valid hit last frame, or disoccluded). Payload-agnostic
// (only ever depends on x0's own position and the camera, never on which
// ReSTIR technique - DI or GI - is consuming the result), factored out of
// wf_restir_temporal_combine below purely so GI's own temporal reuse
// (restir_gi_finalize, wavefront_kernels_restir.cu) can share this exact,
// already-tested reprojection/disocclusion test instead of forking a
// byte-for-byte duplicate of it - the correctness-sensitive M-cap-before-
// combine bug this project already hit once (this file's own header comment)
// lived exactly in code shaped like this, so keeping one copy matters.
//
// Disocclusion test: compares the reprojected history pixel's stored world
// position against `currentPoint`, with a self-scaling epsilon (a fraction of
// that surface's own distance from the PREVIOUS frame's camera - matches the
// "same relative error tolerance regardless of depth" reasoning real-time
// reprojection techniques generally use) rather than a fixed world-space
// constant, which would be too loose close up and too tight far away.
__device__ __forceinline__ int wf_restir_reproject_prev_pixel(
		const float3& currentPoint, const GpuReprojectBasis& prevCamera,
		const float4* worldPosHistory, int imageWidth, int imageHeight) {
	if (!worldPosHistory || imageWidth <= 0 || imageHeight <= 0) return -1;

	WfScreenProjection proj = wf_project_to_screen(currentPoint, prevCamera);
	if (!proj.inFront || proj.s < 0.0f || proj.s >= 1.0f || proj.t < 0.0f || proj.t >= 1.0f) return -1;

	const int px = (int)(proj.s * (float)imageWidth);
	// t is bottom-to-top (camera_math.h's own ScreenProjection comment) - flip
	// to a top-to-bottom pixel row, matching every other screen buffer here.
	const int py = (int)((1.0f - proj.t) * (float)imageHeight);
	if (px < 0 || px >= imageWidth || py < 0 || py >= imageHeight) return -1;
	const int prevPixel = py * imageWidth + px;

	const float4 prevWorldPos = worldPosHistory[prevPixel];
	if (prevWorldPos.w == 0.0f) return -1;  // previous frame never wrote a valid hit there (miss, or specular)

	const float3 prevPoint = make_float3(prevWorldPos.x, prevWorldPos.y, prevWorldPos.z);
	const float3 delta = currentPoint - prevPoint;
	const float distSq = dot(delta, delta);
	const float3 prevCamToPoint = prevPoint - prevCamera.origin;
	const float prevCamDist = sqrtf(fmaxf(dot(prevCamToPoint, prevCamToPoint), 0.0f));
	const float eps = fmaxf(prevCamDist * 0.01f, 1e-4f);
	if (distSq > eps * eps) return -1;  // disoccluded - a genuinely different surface reprojected here

	return prevPixel;
}

// Checkerboard temporal upsampling's "hold" building block, shared by
// svgf_temporal_integrate (wavefront_kernels_svgf.cu) and restir_gi_finalize
// (wavefront_kernels_restir.cu) - both need the exact same sequence (check a
// hit exists this frame, reproject it into the previous frame via the SAME
// helper just above, fetch that pixel's own history-buffer entry) to decide
// whether a checkerboard-inactive pixel has something to carry forward
// unchanged. Factored out here rather than hand-duplicated per kernel (as an
// earlier version of this code did) for the same reason
// wf_restir_reproject_prev_pixel itself was factored out - see that
// function's own header comment: a fix to this sequence (e.g. a stricter
// disocclusion test) previously needed two call sites updated in lockstep
// with no compiler enforcement they stayed identical.
//
// Templated on the history element type (GpuSvgfState, GpuGiReservoir - both
// plain PODs) since the reprojection/lookup logic is identical regardless of
// payload; only what the CALLER does with the returned entry differs (SVGF
// still applies its own temporal blend when isActive, GI only ever calls
// this from its already-inactive branch). Returns false (leaving `outHeld`
// untouched) if there's nothing to hold - a miss, no reprojection target, or
// disocclusion - the same three cases wf_restir_reproject_prev_pixel itself
// already collapses into a single "no reuse" signal.
template <typename T>
__device__ __forceinline__ bool wf_checkerboard_try_hold(
		const float4& currentHitWorldPos, const GpuReprojectBasis& prevCamera,
		const float4* worldPosHistory, int imageWidth, int imageHeight,
		const T* history, T& outHeld) {
	if (currentHitWorldPos.w == 0.0f) return false;
	const float3 hitPoint = make_float3(currentHitWorldPos.x, currentHitWorldPos.y, currentHitWorldPos.z);
	const int prevPixel = wf_restir_reproject_prev_pixel(hitPoint, prevCamera, worldPosHistory, imageWidth, imageHeight);
	if (prevPixel < 0) return false;
	outHeld = history[prevPixel];
	return true;
}

// ReSTIR temporal reuse: reprojects `hitPoint` into the previous frame's
// camera (ctx.prevCamera), and - if that lands on-screen, on a
// non-disoccluded surface, and ctx.historyValid - combines ctx.history's
// reservoir at that reprojected pixel into `current` via
// restir_reservoir_combine(), clamping the resulting M to kRestirTemporalMaxM
// (restir.h's own max_M concept: bounds how much a long-lived reservoir can
// outweigh fresh candidates once the scene/camera starts changing again).
// `current` must already hold this frame's freshly-generated (not yet
// finalized) reservoir - restir_finalize() is NOT called here, since the
// caller (wf_finish_material_scatter) still needs to fold in more candidates/
// call restir_finalize() itself afterward.
//
// Disocclusion test: compares the reprojected history pixel's stored world
// position against hitPoint, with a self-scaling epsilon (a fraction of this
// surface's own distance from the CURRENT camera - matches the "same relative
// error tolerance regardless of depth" reasoning real-time reprojection
// techniques generally use) rather than a fixed world-space constant, which
// would be too loose close up and too tight far away. No separate explicit
// "hard reset" signal is needed (see GpuRestirTemporalContext's own comment):
// a camera cut fails this same test almost everywhere, naturally falling back
// to fresh candidates only.
__device__ __forceinline__ void wf_restir_temporal_combine(
		GpuReservoir& current, const float3& hitPoint, const float3& normal,
		const GpuRestirTemporalContext& ctx, unsigned int& seed,
		const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
		const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
		const MaterialData* materials, const TextureData* textures, const unsigned char* texturePixels) {
	if (!ctx.historyValid || !ctx.history || !ctx.worldPosHistory || ctx.imageWidth <= 0 || ctx.imageHeight <= 0)
		return;

	const int prevPixel = wf_restir_reproject_prev_pixel(hitPoint, ctx.prevCamera, ctx.worldPosHistory,
														   ctx.imageWidth, ctx.imageHeight);
	if (prevPixel < 0) return;

	GpuReservoir prev = ctx.history[prevPixel];
	if (!prev.valid()) return;
	// Clamp the INCOMING reservoir's M before folding it in, not the summed
	// current.M afterward (as this code used to). restir_reservoir_combine's
	// candidate weight is `pHat * other.W * other.M` - if M is truncated only
	// after that weight (and current.weightSum) already used the untruncated
	// value, restir_finalize's W = weightSum / (M * pHat) divides by a SMALLER
	// M than weightSum was built from, inflating W. That inflated W is exactly
	// what gets stored as this frame's own W and fed forward as `other.W` into
	// EVERY subsequent frame's combine, compounding the inflation frame over
	// frame instead of merely bounding history staleness (confirmed via a
	// direct instrumented A/B: disabling temporal reuse alone dropped this
	// scene's converged image mean from ~44x classic NEE's own mean back to
	// parity with it). Clamping M here, before it ever reaches the weight
	// formula, keeps weightSum and M mutually consistent - the standard ReSTIR
	// M-cap remains an intentional, bounded, accepted bias (per Bitterli 2020),
	// not an unbounded runaway.
	if (prev.M > kRestirTemporalMaxM) prev.M = kRestirTemporalMaxM;

	// Re-evaluate the history sample's geometry AND target function fresh, at
	// THIS pixel's own hitPoint/normal - restir.h's documented missing piece
	// for unbiased reuse (this file's own header comment).
	float3 dirToSample; float dist; float geomPdf;
	// time=0.0f: no per-pixel shutter-time buffer exists to recover the
	// PREVIOUS frame's actual draw time here - see wf_reevaluate_light_
	// geometry's own header comment on this narrower, still-open limitation
	// (a moving emissive sphere's re-evaluation across frames can still use
	// the wrong center, unlike the same-frame immediate re-evaluation case).
	if (!wf_reevaluate_light_geometry(prev.sample, hitPoint, 0.0f, spheres, quads, triangles,
									   bilinearPatches, disks, cylinders, dirToSample, dist, geomPdf) ||
		geomPdf <= 0.0f)
		return;

	const float3 rawEmission = wf_light_raw_emission(prev.sample, dirToSample, materials, spheres, quads, triangles,
													  bilinearPatches, disks, cylinders, textures, texturePixels);
	const float pHatAtCurrent = wf_restir_target_proxy(rawEmission, dirToSample, normal);
	if (pHatAtCurrent <= 0.0f) return;

	restir_reservoir_combine(current, prev, pHatAtCurrent, wf_rand(seed));
}
