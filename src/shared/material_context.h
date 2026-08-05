#pragma once
// ---------------------------------------------------------------------------
// material_context.h -- Shared CPU/GPU material evaluation context
//
// Mirrors pbrt-v4's MaterialEvalContext:
//   A small flat struct carrying the per-hit surface data that materials need
//   to evaluate textures and build BxDFs.  Keeps material interfaces
//   decoupled from the CPU-only hit_record and from OptiX payload structs.
//
// Usage (CPU):
//   MaterialContext ctx = MaterialContext::from_hit(rec, r_in);
//   auto bxdf = mat.get_bxdf(ctx);
//
// Usage (GPU):
//   MaterialContext ctx;
//   ctx.u = ...; ctx.v = ...; // fill from OptiX attributes
//   auto bxdf = gpu_get_bxdf(mat_id, ctx);
//
// Design rules (same as bxdfs.h / shading_frame.h):
//   - Plain data, no virtual functions, no heap allocation
//   - GPU-friendly: all scalar fields, no std:: types
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

template<typename T>
struct MaterialContext {
	// Texture coordinates
	T u, v;

	// Hit point in world space
	T px, py, pz;

	// Outward shading normal (unit vector, world space)
	T nx, ny, nz;

	// Surface tangent along U (world space) -- for normal/bump mapping
	// pbrt-v4: ctx.shading.dpdu
	T dpdu_x, dpdu_y, dpdu_z;

	// Outgoing direction (away from surface, world space)
	// = -ray_direction at hit point
	T wo_x, wo_y, wo_z;

	// True if the ray hit the front face (enters from outside)
	bool front_face;

	// Hit time (for motion blur)
	T time;

#ifndef __CUDACC__
	// CPU-only convenience constructor from RTOW hit_record + ray.
	// Included only on CPU to avoid pulling <ray.h> into GPU code.
	template<typename HitRecord, typename Ray>
	static MaterialContext from_hit(const HitRecord& rec, const Ray& r_in) {
		MaterialContext ctx;
		ctx.u          = static_cast<T>(rec.u);
		ctx.v          = static_cast<T>(rec.v);
		ctx.px         = static_cast<T>(rec.p.x());
		ctx.py         = static_cast<T>(rec.p.y());
		ctx.pz         = static_cast<T>(rec.p.z());
		ctx.nx         = static_cast<T>(rec.normal.x());
		ctx.ny         = static_cast<T>(rec.normal.y());
		ctx.nz         = static_cast<T>(rec.normal.z());
		ctx.dpdu_x     = static_cast<T>(rec.dpdu.x());
		ctx.dpdu_y     = static_cast<T>(rec.dpdu.y());
		ctx.dpdu_z     = static_cast<T>(rec.dpdu.z());
		auto wo = -r_in.direction();
		T    wo_len = static_cast<T>(wo.length());
		if (wo_len > T(1e-8)) wo_len = T(1) / wo_len; else wo_len = T(0);
		ctx.wo_x       = static_cast<T>(wo.x()) * wo_len;
		ctx.wo_y       = static_cast<T>(wo.y()) * wo_len;
		ctx.wo_z       = static_cast<T>(wo.z()) * wo_len;
		ctx.front_face = rec.front_face;
		ctx.time       = static_cast<T>(r_in.time());
		return ctx;
	}
#endif
};
