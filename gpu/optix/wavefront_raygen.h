// wavefront_raygen.h -- Intersect-phase and shadow-phase raygen programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __raygen__wf_intersect
//   Launch dimensions: (numRaysInQueue, 1, 1).
//   Each thread handles one RayWorkItem from rayQueue.
// ============================================================================
extern "C" __global__ void __raygen__wf_intersect() {
	const unsigned int rayIdx = optixGetLaunchIndex().x;
	// Bounds check (launch size is numRays, set by host). Must also guard
	// against rq.capacity, not just the live *rq.counter: WorkQueue::push()
	// (wavefront_types.h) increments the counter unconditionally even once
	// the backing array is full - it only skips the actual item write past
	// capacity (returns -1, "should not happen if capacity >= numPixels").
	// A queue that legitimately overflows (see __raygen__wf_shadow's own
	// version of this comment for a confirmed real case) would otherwise
	// leave *rq.counter reporting more live entries than were ever written,
	// so trusting it alone here reads rq.items[] past its cudaMalloc'd end -
	// the exact "illegal memory access" this guard prevents.
	const WorkQueue<RayWorkItem>& rq = wf_params.rayQueue;
	if ((int)rayIdx >= *rq.counter || (int)rayIdx >= rq.capacity) return;

	const RayWorkItem& ray = rq.items[rayIdx];

	WfHitPayload payload;
	payload.hit = false;

	unsigned int p0, p1;
	packPointer(&payload, p0, p1);

	optixTrace(
		wf_params.traversable,
		ray.origin,
		ray.direction,
		ray.tMin,
		ray.tMax,
		0.0f,                             // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,                                // SBT offset (radiance)
		RAY_TYPE_COUNT,                   // SBT stride - see WavefrontPathTracer::
										   // buildSBT()'s pushTriple comment for why
										   // this must match the shared IAS's baked
										   // instance.sbtOffset stride, not a smaller
										   // per-type record count
		0,                                // miss SBT index (radiance miss)
		p0, p1
	);

	if (payload.hit) {
		HitWorkItem h;
		h.hitPoint    = payload.hitPoint;
		h.normal      = payload.normal;
		h.t           = payload.t;
		h.materialIdx = payload.materialIdx;
		h.geomType    = payload.geomType;
		h.mediumTFar  = payload.mediumTFar;
		h.frontFace   = payload.frontFace;
		h.objDpdu     = payload.objDpdu;
		h.uv_u        = payload.uv_u;
		h.uv_v        = payload.uv_v;
		h.rayOrigin   = ray.origin;
		h.rayDir      = ray.direction;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			h.throughput[i]      = ray.throughput[i];
			h.radiance[i]        = ray.radiance[i];
			h.wavelengths[i]     = ray.wavelengths[i];
			h.wavelength_pdfs[i] = ray.wavelength_pdfs[i];
		}
		h.seed        = ray.seed;
		h.pixelIndex  = ray.pixelIndex;
		h.depth       = ray.depth;
		h.specular_bounce = ray.specular_bounce;
		h.any_nonspecular = ray.any_nonspecular;
		h.etaScale        = ray.etaScale;
		h.filterWeight    = ray.filterWeight;
		h.brdf_pdf        = ray.brdf_pdf;
		// Resolve Mix HERE, once, before routing - not in each consumer
		// kernel. h.materialIdx is overwritten with the RESOLVED index, so
		// evaluate_materials()/_simple()/_dielectric() (wavefront_kernels.cu)
		// need no Mix-awareness of their own: they read materials[h.materialIdx]
		// and see the real sub-material directly, exactly as if it had been
		// assigned to this hit's primitive to begin with. See MaterialType::
		// Mix's own comment (optix_types.h).
		{
			int resolvedIdx = h.materialIdx;
			wf_resolve_mix_material(wf_params.materials[h.materialIdx], h.materialIdx, h.hitPoint, resolvedIdx);
			h.materialIdx = resolvedIdx;
		}
		// Route cheap materials (no texture/layered-BxDF work) into their
		// own queue so evaluate_materials_simple() can process them without
		// the big switch's register pressure - see WavefrontQueues::
		// simpleHitQueue's comment (wavefront_types.h).
		const MaterialType mt = wf_params.materials[h.materialIdx].type;
		if (mt == MaterialType::Lambertian || mt == MaterialType::Metal) {
			wf_params.simpleHitQueue.push(h);
		} else if (mt == MaterialType::Dielectric || mt == MaterialType::RoughDielectric) {
			wf_params.dielectricHitQueue.push(h);
		} else {
			wf_params.hitQueue.push(h);
		}
	} else {
		MissWorkItem m;
		// MissWorkItem is a one-shot terminal type (no further bounces), so
		// the pure reconstruction weight (ray.filterWeight - see RayWorkItem::
		// filterWeight's own comment) is folded in here at creation rather
		// than carried as its own field, matching ShadowRayWorkItem's Ld's
		// identical treatment (__raygen__wf_shadow's own NEE-push sites) and
		// wf_finish_material_scatter()'s "flush" sites.
		for (int i = 0; i < kWFNWavelengths; ++i) {
			m.throughput[i]      = ray.throughput[i] * ray.filterWeight;
			m.radiance[i]        = ray.radiance[i] * ray.filterWeight;
			m.wavelengths[i]     = ray.wavelengths[i];
			m.wavelength_pdfs[i] = ray.wavelength_pdfs[i];
		}
		m.rayDir      = ray.direction;
		m.rayOrigin   = ray.origin;
		m.pixelIndex  = ray.pixelIndex;
		m.brdf_pdf    = ray.brdf_pdf;
		m.depth       = ray.depth;
		wf_params.missQueue.push(m);
	}
}

// ============================================================================
// Shadow pass programs
// ============================================================================

// Shadow launch params: reuse wf_params but a separate optixLaunch with a
// shadow pipeline.  We store occluded[] as a bool* in wf_params.framebuffer
// during the shadow pass launch (the host casts it; the bool array has exactly
// numShadow entries allocated separately).

// __raygen__wf_shadow: one thread per shadow ray.
extern "C" __global__ void __raygen__wf_shadow() {
	const unsigned int idx = optixGetLaunchIndex().x;
	const WorkQueue<ShadowRayWorkItem>& sq = wf_params.shadowQueue;
	// Must also guard against sq.capacity, not just the live *sq.counter:
	// WorkQueue::push() (wavefront_types.h) increments the counter
	// unconditionally even once the backing array (d_shadowItems_, sized to
	// queueCapacity_ = width*height) is full - it only skips the actual item
	// write past capacity (returns -1). The shadow queue is genuinely NOT
	// bounded by 1 push per hit: evaluate_materials's wf_finish_material_
	// scatter() (wavefront_kernels.cu) can push once for area-light NEE,
	// once more for sky NEE, and once per punctual light for the SAME hit,
	// so a scene combining several of those (confirmed: scene B2/"Cornell
	// Rough Metal", which sets a non-zero GpuCameraParams::backgroundColor -
	// see scene_builder.cpp case 10 - so its own sky-NEE branch fires
	// alongside its area light) can legitimately push more items in one
	// bounce than queueCapacity_ holds. Before this fix, *sq.counter (e.g.
	// 3934) exceeding sq.capacity (e.g. 3600) meant this raygen launched
	// with more threads than the buffer has slots for, and every idx in
	// [capacity, counter) read sq.items[idx]/occluded[idx] past their
	// cudaMalloc'd end - confirmed via compute-sanitizer as an out-of-bounds
	// __global__ read landing just past d_shadowItems_'s allocation. That
	// stray read (and the matching one in accumulate_shadow, wavefront_
	// kernels.cu) is what corrupted the CUDA/OptiX context: once enough
	// prior scene switches left the right garbage in the allocator's
	// adjacent memory, the read landed on unmapped/foreign memory instead of
	// harmless padding, raising CUDA error 700 for the rest of the process.
	// Excess pushes beyond capacity are still silently dropped (same
	// WorkQueue::push() contract as before this fix) - this only stops the
	// out-of-bounds READ of those dropped slots, it does not change which
	// shadow rays get traced.
	if ((int)idx >= *sq.counter || (int)idx >= sq.capacity) return;

	const ShadowRayWorkItem& s = sq.items[idx];

	WfShadowPayload sp;
	sp.occluded = false;

	unsigned int p0, p1;
	packPointer(&sp, p0, p1);

	optixTrace(
		wf_params.traversable,
		s.origin,
		s.direction,
		0.001f,
		s.tMax,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		1,                  // SBT offset (shadow) -- still a valid slot within
							// each type's now-RAY_TYPE_COUNT-record block; see
							// __raygen__wf_intersect's own SBT stride comment
		RAY_TYPE_COUNT,     // SBT stride
		0,                  // miss SBT index -- shadowSBT_ has its OWN dedicated
							// missRecordBase with exactly ONE record (see
							// buildSBT()'s shadowSBT_.missRecordCount = 1), unlike
							// a combined [radiance, shadow] miss array; index 1 was
							// out-of-bounds. Silently "worked" on small scenes by
							// reading adjacent heap bytes past the 1-record cudaMalloc
							// (undefined behavior, not correctness) until confirmed via
							// OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, which flags it
							// as MISS_SBT_OUT_OF_BOUNDS on every scene, and turns it into
							// a hard device fault whose exact address depends on
							// allocation layout -- which is what made scene 8's larger
							// memory footprint crash outright instead of "working".
		p0, p1
	);

	// Write result into the bool array (reused from framebuffer pointer during shadow pass)
	bool* occluded = (bool*)wf_params.framebuffer;
	occluded[idx]  = sp.occluded;
}
