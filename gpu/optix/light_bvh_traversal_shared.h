// light_bvh_traversal_shared.h -- pbrt-v4 bounding-cone light BVH traversal
// (stochastic descent + bit-trail PMF replay), shared between the GPU-
// recursive backend (optix_device_helpers_lighting.h's own gpu_light_bvh_
// sample_index()/gpu_light_bvh_pmf(), which read params.lightBvh* into these
// calls) and the GPU-wavefront backend (wavefront_restir_helpers.h's
// wf_generate_restir_candidate()/wf_finish_material_scatter(), which already
// carry the same data as explicit WfLightBvhContext fields). Same split
// rationale as gpu_portal_light_shared.h/gpu_sky_light_shared.h (see either
// file's own header comment): every function here takes its data as plain
// explicit parameters, no `params`/`wf_params` global read, so both backends
// include and call it directly - one copy of the traversal logic instead of
// two hand-kept-in-sync ones that could silently drift apart.
//
// GpuLightBvhSample (the return type below) lives in src/shared/light_bvh_
// node.h - see that struct's own comment. A plain by-value struct,
// deliberately NOT a `float&`/`int&` reference-output parameter: this
// codebase's own memory of a prior GPU-recursive-backend miscompile
// (CloudMedium::compute_density()'s dnoise() helper, see gpu_cloud_density()'s
// own history) found reference-output device functions unreliable in this
// exact NVCC/OptiX toolchain when not force-inlined - by-value struct returns
// sidestep that class of bug entirely.
//
// This header does NOT fix or reproduce GPU-recursive's own KNOWN UNRESOLVED
// BUG (see optix_device_helpers_lighting.h's own comment on gpu_light_bvh_
// sample_index() for the full write-up - a real NVCC floating-point
// divergence in CompactLightBounds::Importance()'s own call chain, specific
// to that backend's one-thread-per-pixel megakernel). Moving the traversal
// here instead of hand-duplicating it doesn't change that: GPU-recursive
// still forces params.lightBvhNodeCount to 0 (optix_renderer_render.cpp), so
// neither function below is ever actually called with real data on that
// backend today - only wavefront calls these live in production. What this
// removes is the DUPLICATION risk: once Importance() is eventually fixed for
// GPU-recursive (see that comment's own "next concrete step"), there is only
// one traversal to have gotten right, not two to keep in sync by hand.

#pragma once

// light_bvh_sample_index: returns the selected light's index (or -1 if no
// light BVH was built for this scene, or every light's importance at this
// point is zero) and its selection PMF - a drop-in replacement for the alias
// table's `selection_pdf` at every call site, since `light_pdf = pmf *
// geom_pdf` is the same formula either way.
//
// Bounds+monotonicity guards below - see optix_device_helpers_lighting.h's
// own header comment (gpu_light_bvh_sample_index's) for the crash-diagnosis
// history these exist to guard against: a bounds check alone isn't enough to
// prevent a corrupted-but-in-range index from turning the `while(true)` loop
// below into an infinite loop (a GPU hang/driver TDR) - the forward-progress
// check (`c1Index <= nodeIndex + 1`) is what a well-formed flattened BVH
// always guarantees (see BVHLightSampler2::buildBVH()'s own `nodeIndex + 1
// == i0` assertion, src/shared/bvh_light_sampler2.h) and a genuine right-
// child index can never violate.
__device__ __forceinline__ GpuLightBvhSample light_bvh_sample_index(
	float px, float py, float pz, float nx, float ny, float nz, float u,
	const LightBVHNode* nodes, int nodeCount, unsigned int numLights,
	float allBMinX, float allBMinY, float allBMinZ,
	float allBMaxX, float allBMaxY, float allBMaxZ)
{
	if (nodeCount <= 0) return GpuLightBvhSample{-1, 0.f};
	int nodeIndex = 0;
	float pmf = 1.f;
	u = fminf(u, 1.f - 1e-7f);
	while (true) {
		if (nodeIndex < 0 || nodeIndex >= nodeCount) {
			return GpuLightBvhSample{-1, 0.f};
		}
		const LightBVHNode& node = nodes[nodeIndex];
		if (!node.isLeaf) {
			const int c1Index = (int)node.childOrLightIndex;
			if (c1Index <= nodeIndex + 1 || c1Index >= nodeCount) {
				return GpuLightBvhSample{-1, 0.f};
			}
			const LightBVHNode& c0 = nodes[nodeIndex + 1];
			const LightBVHNode& c1 = nodes[node.childOrLightIndex];
			float ci0 = c0.lightBounds.Importance(px,py,pz, nx,ny,nz,
				allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
			float ci1 = c1.lightBounds.Importance(px,py,pz, nx,ny,nz,
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
			// Leaf's own bounds guard - childOrLightIndex here is a LIGHT
			// index (into lightIndices/lightKinds), a completely different
			// range than node indices.
			if (node.childOrLightIndex >= numLights) {
				return GpuLightBvhSample{-1, 0.f};
			}
			return GpuLightBvhSample{(int)node.childOrLightIndex, pmf};
		}
	}
}

// light_bvh_pmf: replays the bit-trail for `lightIndex` to recompute its
// selection PMF at THIS shading point (position-dependent, unlike the alias
// table's fixed pdf) - needed at a BSDF-sampled light hit to MIS-weight
// against whatever NEE would have picked from the point the BSDF sample was
// actually taken (GPU-recursive's 6 shape closest-hit files' own
// DiffuseLight branches), or whenever a light-BVH-selected candidate's
// winning sample is re-used from somewhere other than the draw that produced
// it (wavefront's ReSTIR temporal/spatial reuse). Returns 0 if no light BVH
// was built, or if every ancestor's combined importance was zero (can't
// happen for a real bit-trail from a light actually in the tree, but matches
// this function's own defensive return everywhere else).
__device__ __forceinline__ float light_bvh_pmf(
	float px, float py, float pz, float nx, float ny, float nz,
	int lightIndex, unsigned int numLights,
	const LightBVHNode* nodes, const unsigned int* bitTrail, int nodeCount,
	float allBMinX, float allBMinY, float allBMinZ,
	float allBMaxX, float allBMaxY, float allBMaxZ)
{
	if (nodeCount <= 0 || !bitTrail) return 0.f;
	if (lightIndex < 0 || (unsigned int)lightIndex >= numLights) return 0.f;
	unsigned int trail = bitTrail[lightIndex];
	float pmf = 1.f;
	int nodeIndex = 0;
	while (true) {
		if (nodeIndex < 0 || nodeIndex >= nodeCount) return 0.f;
		const LightBVHNode& node = nodes[nodeIndex];
		if (node.isLeaf) return pmf;
		const int c1Index = (int)node.childOrLightIndex;
		if (c1Index <= nodeIndex + 1 || c1Index >= nodeCount) return 0.f;
		const LightBVHNode& c0 = nodes[nodeIndex + 1];
		const LightBVHNode& c1 = nodes[node.childOrLightIndex];
		float ci0 = c0.lightBounds.Importance(px,py,pz, nx,ny,nz,
			allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
		float ci1 = c1.lightBounds.Importance(px,py,pz, nx,ny,nz,
			allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
		float sum = ci0 + ci1;
		if (sum == 0.f) return 0.f;
		int branch = (int)(trail & 1u);
		pmf *= (branch == 0 ? ci0 : ci1) / sum;
		nodeIndex = (branch == 0) ? (nodeIndex + 1) : (int)node.childOrLightIndex;
		trail >>= 1;
	}
}
