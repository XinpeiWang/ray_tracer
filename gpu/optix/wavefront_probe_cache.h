// wavefront_probe_cache.h -- Probe cache update rays: PHASE 1 (intersection
// only) of the world-space irradiance probe cache (Live Preview only). See
// this project's own plan for the full design.
//
// Included by wavefront_programs.cu AFTER wavefront_common.h (needs
// wf_instance_base()/packPointer()/unpackPointer() AND wavefront_probe.h's
// own wf_trace_probe_ray()/WfProbePayload, both already pulled in by
// wavefront_common.h's own #include "wavefront_probe.h" at its end).
//
// Deliberately traces the intersection query ONLY - no NEE, no light
// sampling, no shading. wf_generate_restir_candidate()/wf_sample_*_light()
// (the full, all-light-kind NEE machinery a probe-update ray wants to reuse
// rather than re-derive) live in wavefront_device_helpers.h's translation
// unit, which this file's own translation unit (wavefront_programs.cu,
// compiled as a separate OptiX module) cannot reach - the same "duplicate
// wf_pcg/wf_rand, don't share across the module boundary" constraint every
// other wf_-prefixed helper in this codebase already works around, except
// here the thing on the other side of the boundary (full NEE across 6 light
// shape kinds) is too large to duplicate reasonably. Instead: this raygen
// traces ONE optixTrace() per queued ray (reusing wavefront_probe.h's own
// probeSBT_ hit groups - no new hit/miss programs, no new SBT) and pushes
// the raw hit result into a SECOND queue; probe_cache_shade
// (wavefront_kernels_restir.cu, same translation unit as evaluate_materials
// and therefore able to call wf_generate_restir_candidate() directly) does
// the actual NEE shading from there. Mirrors the shadow pass's own
// raygen-traces / separate-kernel-shades split.

// ---------------------------------------------------------------------------
// __raygen__wf_probe_cache -- one thread per queued ProbeCacheRayWorkItem.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_probe_cache() {
	const unsigned int idx = optixGetLaunchIndex().x;
	const WorkQueue<ProbeCacheRayWorkItem>& rq = wf_params.probeCacheRayQueue;
	// Same capacity guard as every other wavefront raygen - see
	// __raygen__wf_shadow's own comment (wavefront_raygen.h) for why this
	// must check BOTH *rq.counter and rq.capacity.
	if ((int)idx >= *rq.counter || (int)idx >= rq.capacity) return;

	const ProbeCacheRayWorkItem& item = rq.items[idx];

	// Reuses wavefront_probe.h's own WfProbePayload/wf_trace_probe_ray() -
	// the exact same closest-hit query the BSSRDF probe walk already uses,
	// just one single trace instead of a bounded walk loop, and against an
	// arbitrary probe-ray direction instead of a probe-segment axis. time=0
	// (probe rays aren't tied to any one path's own per-sample shutter
	// instant - they're persistent, cross-frame, cross-sample updates; a
	// fixed time is an acceptable simplification for a cache this coarse
	// already, see this project's own plan for the analogous participating-
	// media deferral).
	WfProbePayload hit = wf_trace_probe_ray(item.origin, item.direction, 1e30f, 0.0f);

	ProbeCacheHitWorkItem out;
	out.batchSlot   = item.batchSlot;
	out.hit         = hit.found;
	out.hitPoint    = hit.position;
	out.hitNormal   = hit.normal;
	out.materialIdx = hit.materialIdx;
	out.direction   = item.direction;
	out.seed        = item.seed;
	// Matches probe_cache_shade's own miss-sentinel convention (large but
	// finite, NOT 1e30f - see that function's own comment for why a raw
	// 1e30f would corrupt the mean-distance leak test at lookup time).
	out.hitDist     = hit.found ? length(hit.position - item.origin) : 1.0e4f;
	wf_params.probeCacheHitQueue.push(out);
}
