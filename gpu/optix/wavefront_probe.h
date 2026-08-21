// wavefront_probe.h -- BSSRDF probe walk for the wavefront GPU backend
// (MaterialType::Subsurface, Phase 2). Included by wavefront_programs.cu
// AFTER wf_params/wf_instance_base()/wf_prim_base()/packPointer()/
// unpackPointer() are declared (needs all of them).
//
// Mirrors gpu/optix/optix_bssrdf.h + optix_probe_hit.h + optix_device_
// helpers.h's trace_probe_ray()/bssrdf_probe_walk() (the recursive
// backend's Phase 1 BSSRDF, see that phase's own commit message for the
// full algorithm derivation) - but reads `wf_params` (WavefrontLaunchParams)
// instead of the recursive path's `params` (LaunchParams) global, since
// wavefront_programs.cu and optix_programs.cu are compiled as two separate
// OptiX modules, each with its own __constant__ launch-params symbol (see
// this file's own wf_gpu_bssrdf_sr() etc. below, and every other wf_-
// prefixed device-helper duplicate already in wavefront_kernels.cu/
// wavefront_programs.cu for the same reason). NOT a #include of
// optix_bssrdf.h/optix_probe_hit.h themselves - those hardcode the
// identifier `params`, which would either collide with or fail to resolve
// against this module's `wf_params`.
//
// __raygen__wf_probe is launched once per bounce depth, only when the prior
// evaluate_materials pass queued at least one MaterialType::Subsurface
// transmission (WavefrontPathTracer::render()'s per-bounce loop) - one
// thread per queued BssrdfProbeWorkItem, mirroring pbrt-v4's own
// SampleSubsurface -> IntersectOneRandom staging (src/pbrt/wavefront/
// subsurface.cpp, src/pbrt/gpu/optix/optix.cu's __raygen__randomHit) and,
// more directly, this codebase's own already-committed recursive-backend
// bssrdf_probe_walk() (optix_device_helpers.h), just running the identical
// bounded-loop-of-sequential-optixTrace-calls algorithm from a dedicated
// raygen program over a whole queue in parallel instead of inline from a
// hit program.
//
// Reuses this module's own existing intersect pipeline/module (wfModule_/
// intersectPipeline_ - see WavefrontPathTracer::createProgramGroups()/
// linkPipeline()) rather than a new OptiX module: this file's programs are
// simply MORE program groups linked into that same pipeline, selected via a
// separate SBT (WavefrontPathTracer::buildSBT()'s probeSBT_) at launch time
// - no new module/pipeline/build-system plumbing needed, matching how the
// existing shadow pass already reuses wfModule_ with its own SBT.

// PCG RNG - duplicated from wavefront_kernels.cu's wf_pcg()/wf_rand() (same
// algorithm, same name is safe: wavefront_kernels.cu and this file/
// wavefront_programs.cu are compiled as completely separate translation
// units/OptiX modules, so there is no link-time collision - matches this
// file's own established duplicate-not-share convention for every other
// device helper).
__device__ __forceinline__ unsigned int wf_pcg(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ float wf_rand(unsigned int& seed) {
	seed = wf_pcg(seed);
	return float(seed) / 4294967296.0f;
}

// ---------------------------------------------------------------------------
// wf_gpu_bssrdf_sr()/pdf_sr()/sample_sr() -- wavefront-native duplicates of
// optix_bssrdf.h's identically-named functions (see that file's own
// extensive comment for the algorithm - a lambda-free float port of
// src/shared/bssrdf.h's TabulatedBSSRDF::sr()/pdf_sr()/sample_sr() and the
// Catmull-Rom spline machinery they depend on). Only difference from that
// file: reads wf_params.bssrdfRhoSamples/bssrdfRadiusSamples/bssrdfProfile/
// bssrdfProfileCdf instead of params.*.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int wf_gpu_find_interval_le(const float* arr, int n, float x) {
	int size = n - 2, first = 1;
	while (size > 0) {
		int half = size >> 1;
		int middle = first + half;
		if (arr[middle] <= x) { first = middle + 1; size -= half + 1; }
		else                  { size = half; }
	}
	int r = first - 1;
	if (r < 0)     r = 0;
	if (r > n - 2) r = n - 2;
	return r;
}

__device__ __forceinline__ bool wf_gpu_catmull_rom_weights(const float* nodes, int n, float x,
															 int* offset, float weights[4]) {
	if (!(x >= nodes[0] && x <= nodes[n - 1])) return false;
	const int idx = wf_gpu_find_interval_le(nodes, n, x);
	*offset = idx - 1;
	const float x0 = nodes[idx], x1 = nodes[idx + 1];
	const float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
	weights[1] =  2.0f * t3 - 3.0f * t2 + 1.0f;
	weights[2] = -2.0f * t3 + 3.0f * t2;
	if (idx > 0) {
		const float w0 = (t3 - 2.0f * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
		weights[0] = -w0;
		weights[2] += w0;
	} else {
		const float w0 = t3 - 2.0f * t2 + t;
		weights[0] = 0.0f;
		weights[1] -= w0;
		weights[2] += w0;
	}
	if (idx + 2 < n) {
		const float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
		weights[1] -= w3;
		weights[3] = w3;
	} else {
		const float w3 = t3 - t2;
		weights[1] -= w3;
		weights[2] += w3;
		weights[3] = 0.0f;
	}
	return true;
}

__device__ __forceinline__ float wf_gpu_bssrdf_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float r) {
	const float kPi = 3.14159265358979323846f;
	const float r_opt = r * sigma_t;
	const float* rho_samples    = wf_params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = wf_params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = wf_params.bssrdfProfile + tab.profile_offset;

	int rho_off, rad_off;
	float rho_w[4], rad_w[4];
	if (!wf_gpu_catmull_rom_weights(rho_samples, tab.n_rho, rho, &rho_off, rho_w)) return 0.0f;
	if (!wf_gpu_catmull_rom_weights(radius_samples, tab.n_radius, r_opt, &rad_off, rad_w)) return 0.0f;

	float val = 0.0f;
	for (int j = 0; j < 4; ++j) {
		if (rho_w[j] == 0.0f) continue;
		const int ri = rho_off + j;
		if (ri < 0 || ri >= tab.n_rho) continue;
		for (int k = 0; k < 4; ++k) {
			if (rad_w[k] == 0.0f) continue;
			const int rj = rad_off + k;
			if (rj < 0 || rj >= tab.n_radius) continue;
			val += rho_w[j] * rad_w[k] * profile[ri * tab.n_radius + rj];
		}
	}
	if (r_opt != 0.0f) val /= (2.0f * kPi * r_opt);
	val = fmaxf(0.0f, val);
	return val * sigma_t * sigma_t;
}

__device__ __forceinline__ float wf_gpu_bssrdf_pdf_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float r) {
	const float kPi = 3.14159265358979323846f;
	const float r_opt = r * sigma_t;
	const float* rho_samples    = wf_params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = wf_params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = wf_params.bssrdfProfile + tab.profile_offset;
	const float* profile_cdf    = wf_params.bssrdfProfileCdf + tab.profile_offset;

	int rho_off, rad_off;
	float rho_w[4], rad_w[4];
	if (!wf_gpu_catmull_rom_weights(rho_samples, tab.n_rho, rho, &rho_off, rho_w)) return 0.0f;
	if (!wf_gpu_catmull_rom_weights(radius_samples, tab.n_radius, r_opt, &rad_off, rad_w)) return 0.0f;

	float val = 0.0f, rho_eff_interp = 0.0f;
	for (int j = 0; j < 4; ++j) {
		if (rho_w[j] == 0.0f) continue;
		const int ri = rho_off + j;
		if (ri < 0 || ri >= tab.n_rho) continue;
		rho_eff_interp += rho_w[j] * profile_cdf[ri * tab.n_radius + (tab.n_radius - 1)];
		for (int k = 0; k < 4; ++k) {
			if (rad_w[k] == 0.0f) continue;
			const int rj = rad_off + k;
			if (rj < 0 || rj >= tab.n_radius) continue;
			val += rho_w[j] * rad_w[k] * profile[ri * tab.n_radius + rj];
		}
	}
	if (r_opt != 0.0f) val /= (2.0f * kPi * r_opt);
	if (rho_eff_interp <= 0.0f) return 0.0f;
	return fmaxf(0.0f, val * sigma_t * sigma_t / rho_eff_interp);
}

__device__ __forceinline__ float wf_gpu_cr2d_interp(const float* arr, int col, int offset,
													  const float weights[4], int n1, int n2) {
	float v = 0.0f;
	for (int k = 0; k < 4; ++k) {
		const int ri = offset + k;
		if (weights[k] != 0.0f && ri >= 0 && ri < n1)
			v += arr[ri * n2 + col] * weights[k];
	}
	return v;
}

__device__ __forceinline__ void wf_gpu_hermite_cdf_eval(float t, float f0, float d0, float f1, float d1,
														  float u, float& Fhat_minus_u, float& fhat) {
	const float t2 = t * t, t3 = t2 * t, t4 = t3 * t;
	const float Fhat = f0 * t + 0.5f * d0 * t2
		+ ((-2.0f * d0 - d1) * (1.0f / 3.0f) + f1 - f0) * t3
		+ (0.25f * (d0 + d1) + 0.5f * (f0 - f1)) * t4;
	fhat = f0 + d0 * t
		+ (-2.0f * d0 - d1 + 3.0f * (f1 - f0)) * t2
		+ (d0 + d1 + 2.0f * (f0 - f1)) * t3;
	Fhat_minus_u = Fhat - u;
}

__device__ __forceinline__ float wf_gpu_catmullrom_newton_bisection(float f0, float d0, float f1, float d1, float u) {
	float x0 = 0.0f, x1 = 1.0f;
	float Fmu0, fp0, Fmu1, fp1;
	wf_gpu_hermite_cdf_eval(x0, f0, d0, f1, d1, u, Fmu0, fp0);
	wf_gpu_hermite_cdf_eval(x1, f0, d0, f1, d1, u, Fmu1, fp1);
	if (fabsf(Fmu0) < 1e-6f) return x0;
	if (fabsf(Fmu1) < 1e-6f) return x1;
	const bool neg_start = (Fmu0 < 0.0f);
	float xMid = x0 + (x1 - x0) * (-Fmu0) / (Fmu1 - Fmu0);
	for (int iter = 0; iter < 64; ++iter) {
		if (!(x0 < xMid && xMid < x1)) xMid = 0.5f * (x0 + x1);
		float Fmid, dfMid;
		wf_gpu_hermite_cdf_eval(xMid, f0, d0, f1, d1, u, Fmid, dfMid);
		if (fabsf(Fmid) < 1e-6f || x1 - x0 < 1e-6f) break;
		if ((Fmid < 0.0f) == neg_start) x0 = xMid; else x1 = xMid;
		xMid -= Fmid / dfMid;
	}
	return xMid;
}

__device__ __forceinline__ float wf_gpu_sample_catmull_rom_2d(
	const float* nodes1, int n1, const float* nodes2, int n2,
	const float* values, const float* cdf, float rho, float u)
{
	int offset;
	float weights[4];
	if (!wf_gpu_catmull_rom_weights(nodes1, n1, rho, &offset, weights)) return 0.0f;

	const float maximum = wf_gpu_cr2d_interp(cdf, n2 - 1, offset, weights, n1, n2);
	if (maximum <= 0.0f) return 0.0f;
	u *= maximum;

	int size = n2 - 2, first = 1;
	while (size > 0) {
		const int half = size >> 1;
		const int middle = first + half;
		if (wf_gpu_cr2d_interp(cdf, middle, offset, weights, n1, n2) <= u) { first = middle + 1; size -= half + 1; }
		else                                                                { size = half; }
	}
	int idx = first - 1;
	if (idx < 0)      idx = 0;
	if (idx > n2 - 2) idx = n2 - 2;

	const float f0 = wf_gpu_cr2d_interp(values, idx, offset, weights, n1, n2);
	const float f1 = wf_gpu_cr2d_interp(values, idx + 1, offset, weights, n1, n2);
	const float x0 = nodes2[idx], x1 = nodes2[idx + 1];
	const float w = x1 - x0;
	const float d0 = (idx > 0)
		? w * (f1 - wf_gpu_cr2d_interp(values, idx - 1, offset, weights, n1, n2)) / (x1 - nodes2[idx - 1])
		: (f1 - f0);
	const float d1 = (idx + 2 < n2)
		? w * (wf_gpu_cr2d_interp(values, idx + 2, offset, weights, n1, n2) - f0) / (nodes2[idx + 2] - x0)
		: (f1 - f0);
	const float uu = (u - wf_gpu_cr2d_interp(cdf, idx, offset, weights, n1, n2)) / w;

	const float t = wf_gpu_catmullrom_newton_bisection(f0, d0, f1, d1, uu);
	return x0 + w * t;
}

__device__ __forceinline__ float wf_gpu_bssrdf_sample_sr(const GpuBssrdfTable& tab, float sigma_t, float rho, float u) {
	if (sigma_t <= 0.0f) return -1.0f;
	const float* rho_samples    = wf_params.bssrdfRhoSamples + tab.rho_offset;
	const float* radius_samples = wf_params.bssrdfRadiusSamples + tab.radius_offset;
	const float* profile        = wf_params.bssrdfProfile + tab.profile_offset;
	const float* profile_cdf    = wf_params.bssrdfProfileCdf + tab.profile_offset;
	const float r_opt = wf_gpu_sample_catmull_rom_2d(rho_samples, tab.n_rho, radius_samples, tab.n_radius,
													   profile, profile_cdf, rho, u);
	return r_opt / sigma_t;
}

// ---------------------------------------------------------------------------
// Probe hit groups (RAY_TYPE analog for the wavefront backend) - wavefront-
// native duplicates of optix_probe_hit.h's __closesthit__probe_*/__miss__
// probe, adapted to this module's packed-pointer payload convention
// (pipelineCompileOptions_.numPayloadValues == 2, fixed for the whole
// wfModule_ - see WfHitPayload/WfShadowPayload above for the same pattern)
// instead of the recursive backend's raw 7-register payload (that module
// uses numPayloadValues == 4/13 elsewhere; this one doesn't, so the same
// convention can't be reused verbatim). Deliberately skips alpha-cutout,
// same documented simplification as optix_probe_hit.h's own comment.
// ---------------------------------------------------------------------------

struct WfProbePayload {
	bool   found;
	float3 position;
	float3 normal;
	int    materialIdx;
};

extern "C" __global__ void __closesthit__wf_probe_sphere() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const int instBase = wf_instance_base();
	const SphereData& sphere = wf_params.spheres[wf_prim_base(instBase) + optixGetPrimitiveIndex()];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	float3 obj_hit = hit_point;
	if (instBase >= 0) obj_hit = optixTransformPointFromWorldToObjectSpace(hit_point);
	const float3 obj_normal = normalize(obj_hit - sphere.center);
	float3 outward_normal = obj_normal;
	if (instBase >= 0) outward_normal = normalize(optixTransformNormalFromObjectToWorldSpace(outward_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = sphere.materialIdx;
}

extern "C" __global__ void __closesthit__wf_probe_quad() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const int quadIdx = optixGetPrimitiveIndex();
	const QuadData& quad = wf_params.quads[quadIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	const bool front_face = dot(ray_dir, quad.normal) < 0.0f;
	const float3 normal = front_face ? quad.normal : -quad.normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = quad.materialIdx;
}

extern "C" __global__ void __closesthit__wf_probe_bilinear_patch() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = wf_params.bilinearPatches[primIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);
	const float3 geom_normal = normalize(cross(dpdu, dpdv));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 normal = front_face ? geom_normal : -geom_normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = patch.materialIdx;
}

extern "C" __global__ void __closesthit__wf_probe_triangle() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const int instBase = wf_instance_base();
	const int primIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());
	const TriangleData& tri = wf_params.triangles[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	float3 geom_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	if (instBase >= 0)
		geom_normal = normalize(optixTransformNormalFromObjectToWorldSpace(geom_normal));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 normal = front_face ? geom_normal : -geom_normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = tri.materialIdx;
}

// Disk/Cylinder (Phase 4c) - see wavefront_programs.cu's __intersection__
// wf_disk/wf_cylinder for the object<->world transform helpers these reuse
// (wf_dc_apply_point/wf_dc_apply_normal_from_w2o), defined earlier in the
// same translation unit (this file is #included into wavefront_programs.cu
// after them - see this file's own header comment).
extern "C" __global__ void __closesthit__wf_probe_disk() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = wf_params.disks[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	const float3 obj_normal = make_float3(0.0f, 0.0f, 1.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(disk.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = disk.materialIdx;
}

extern "C" __global__ void __closesthit__wf_probe_cylinder() {
	WfProbePayload* payload = (WfProbePayload*)unpackPointer(optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = wf_params.cylinders[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t        = optixGetRayTmax();
	const float3 hit_point = ray_orig + t * ray_dir;

	const float3 obj_hit = wf_dc_apply_point(cyl.w2o, hit_point);
	const float obj_hit_len = sqrtf(obj_hit.x * obj_hit.x + obj_hit.y * obj_hit.y);
	const float3 obj_normal = (obj_hit_len > 1e-8f)
		? make_float3(obj_hit.x / obj_hit_len, obj_hit.y / obj_hit_len, 0.0f)
		: make_float3(1.0f, 0.0f, 0.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->found       = true;
	payload->position    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = cyl.materialIdx;
}

// No-op: leaves payload->found == false, exactly as wf_trace_probe_ray()
// initialised it (matches optix_probe_hit.h's __miss__probe sentinel
// convention).
extern "C" __global__ void __miss__wf_probe() {}

// ---------------------------------------------------------------------------
// wf_trace_probe_ray() -- one closest-hit probe trace, modeled directly on
// __raygen__wf_shadow's own optixTrace() call above (same SBT-offset/
// stride/miss-index shape, just against probeSBT_'s own hit records instead
// of the shadow ones - see WavefrontPathTracer::buildSBT()'s probeSBT_,
// which mirrors intersectSBT_'s own per-present-type, stride-RAY_TYPE_COUNT
// layout exactly so the shared IAS instance sbtOffset math stays correct -
// see that method's own pushTriple comment). Unlike the recursive backend's
// trace_probe_ray() (optix_device_helpers.h), no self-intersection epsilon nudge is
// needed here either, for the identical reason: wf_bssrdf_probe_walk()
// below already advances `base` past each hit by t+1e-4 before the next
// call.
// ---------------------------------------------------------------------------
__device__ __forceinline__ WfProbePayload wf_trace_probe_ray(
	const float3& origin, const float3& direction, float max_distance)
{
	WfProbePayload payload;
	payload.found = false;

	unsigned int p0, p1;
	packPointer(&payload, p0, p1);

	optixTrace(
		wf_params.traversable,
		origin,
		direction,
		1e-5f,                 // tmin
		max_distance,          // tmax
		0.0f,                  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,   // real closest-hit, not occlusion-only
		0,                     // SBT offset - probeSBT_'s own dedicated hit records
		RAY_TYPE_COUNT,        // SBT stride - mirrors intersectSBT_'s own per-type layout
		0,                     // miss SBT index - probeSBT_ has exactly one miss record
		p0, p1
	);

	return payload;
}

// ---------------------------------------------------------------------------
// wf_bssrdf_probe_walk() -- wavefront-native duplicate of the recursive
// backend's bssrdf_probe_walk() (optix_device_helpers.h - see that
// function's own extensive comment for the full 3-axis-MIS algorithm
// derivation, replicated here verbatim, just reading wf_params instead of
// params and calling wf_trace_probe_ray()/wf_gpu_bssrdf_*() instead of
// their recursive-backend twins).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool wf_bssrdf_probe_walk(
	int matIdx,
	const float3& p0, const float3& axis0,
	unsigned int& seed,
	float3& out_exit_pos, float3& out_exit_normal,
	float3& out_Sp, float& out_pdf, float& out_sample_prob)
{
	constexpr int kMaxProbeSteps = 100;
	const float kPi = 3.14159265358979323846f;

	const MaterialData& mat = wf_params.materials[matIdx];
	const GpuBssrdfTable& table = wf_params.bssrdfTables[mat.textureIdx];
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

	const int channel = min(2, (int)(wf_rand(seed) * 3.0f));
	const float r = wf_gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], wf_rand(seed));
	if (r < 0.0f) return false;
	const float r_max = wf_gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], 0.999f);
	if (r_max <= 0.0f || r >= r_max) return false;

	const float phi = 2.0f * kPi * wf_rand(seed);
	const float half_len = sqrtf(fmaxf(0.0f, r_max * r_max - r * r));

	const float u_axis = wf_rand(seed);
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
		WfProbePayload hit = wf_trace_probe_ray(base, seg_dir, remaining);
		if (!hit.found) break;
		const float t_local = length(hit.position - base);
		if (hit.materialIdx == matIdx) {
			++candidate_count;
			if (wf_rand(seed) < 1.0f / (float)candidate_count) {
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
		wf_gpu_bssrdf_sr(table, sigma_t[0], rho[0], dist),
		wf_gpu_bssrdf_sr(table, sigma_t[1], rho[1], dist),
		wf_gpu_bssrdf_sr(table, sigma_t[2], rho[2], dist));

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
		pdf += kAxisProb   * wf_gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_axis) * cos_axis
			 + kTangentProb * wf_gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t1)   * cos_t1
			 + kTangentProb * wf_gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t2)   * cos_t2;
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

// ---------------------------------------------------------------------------
// __raygen__wf_probe -- one thread per queued BssrdfProbeWorkItem.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_probe() {
	const unsigned int idx = optixGetLaunchIndex().x;
	const WorkQueue<BssrdfProbeWorkItem>& pq = wf_params.bssrdfProbeQueue;
	// Must also guard against pq.capacity, not just the live *pq.counter -
	// see __raygen__wf_shadow's own version of this comment
	// (wavefront_programs.cu) for why: WorkQueue::push() keeps incrementing
	// the counter past a full queue even though it stops writing items[], so
	// a queue that overflows would otherwise read pq.items[] past its
	// cudaMalloc'd end here too.
	if ((int)idx >= *pq.counter || (int)idx >= pq.capacity) return;

	const BssrdfProbeWorkItem& item = pq.items[idx];
	unsigned int seed = item.seed;

	BssrdfExitWorkItem exitItem;
	exitItem.found      = 0;
	exitItem.exitPos    = make_float3(0.0f, 0.0f, 0.0f);
	exitItem.exitNormal = make_float3(0.0f, 0.0f, 0.0f);
	exitItem.Sp         = make_float3(0.0f, 0.0f, 0.0f);
	exitItem.pdf        = 0.0f;
	exitItem.sampleProb = 0.0f;
	exitItem.matIdx     = item.matIdx;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		exitItem.throughput[i]      = item.throughput[i];
		exitItem.radiance[i]        = item.radiance[i];
		exitItem.wavelengths[i]     = item.wavelengths[i];
		exitItem.wavelength_pdfs[i] = item.wavelength_pdfs[i];
	}
	exitItem.pixelIndex = item.pixelIndex;
	exitItem.depth      = item.depth;
	exitItem.any_nonspecular = item.any_nonspecular;

	float3 exitPos, exitNormal, Sp;
	float pdf, sampleProb;
	bool found = wf_bssrdf_probe_walk(item.matIdx, item.p0, item.axis0, seed,
									   exitPos, exitNormal, Sp, pdf, sampleProb);
	exitItem.seed = seed;
	if (found) {
		exitItem.found      = 1;
		exitItem.exitPos    = exitPos;
		exitItem.exitNormal = exitNormal;
		exitItem.Sp         = Sp;
		exitItem.pdf        = pdf;
		exitItem.sampleProb = sampleProb;
	}
	wf_params.bssrdfExitQueue.push(exitItem);
}
