#pragma once
// optix_disk_cylinder_helpers.h -- Disk/Cylinder object<->world transform
// helpers, plus real NEE area/sample/pdf support (GpuLightKind::Disk/
// Cylinder). Included EARLY by optix_programs.cu (before optix_device_
// helpers.h), unlike optix_intersection_disk_cylinder.h's closest-hit/
// intersection programs (included late, after optix_device_helpers.h) -
// optix_device_helpers.h's sample_disk_light()/sample_cylinder_light() need
// dc_apply_point() etc, so those transform helpers (and dc_cylinder_check(),
// needed by both the intersection program and this file's own dc_pdf_
// cylinder()) live here instead of in optix_intersection_disk_cylinder.h,
// which now just includes this file rather than defining its own copies.
//
// Mirrors src/shared/shapes.h's DiskShape<T>/CylinderShape<T>::area()/
// sample()/sample_from()/pdf_from() (CPU's real NEE support for these
// shapes), hand-ported to bool/out-param CUDA device functions rather than
// instantiating those templates on device - same reason as src/shared/
// bilinear_patch.h's own blp_* free functions (that header's comment: the
// std::optional-returning ShapeHit<T>-based intersect()/pdf_from() cannot be
// parsed by nvcc's device frontend). SampleUniformDiskConcentric
// (src/shared/sampling_helpers.h) IS already CPU_GPU-safe (a plain
// template, no std::optional), so the disk's own uniform-area point sample
// reuses it directly rather than re-deriving the concentric mapping by hand.
//
// Approximate under non-uniform (anisotropic) scale, same accepted
// simplification as every other GPU area-light kind and as CPU's own
// disk_cylinder_hittable.h - the world-space area is estimated from a single
// representative scale factor (the length of o2w's transformed local-X
// basis vector) rather than an exact per-direction Jacobian.

#include "optix_types.h"
#include "optix_math_helpers.h"                  // cross()/dot()/length()/normalize()
#include "../../src/shared/sampling_helpers.h"   // SampleUniformDiskConcentric

// Row-major 3x4 affine transform, dropping the implicit [0,0,0,1] bottom row
// - same convention as SceneData::InstancePlacementGPU::transform.
__device__ __forceinline__ float3 dc_apply_point(const float m[12], const float3& p) {
	return make_float3(
		m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
		m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
		m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

// A direction: the translation column is deliberately not applied. NOT
// renormalised on purpose (see disk_cylinder_hittable.h's own apply_vector)
// - that's what lets t survive unmodified between object and world space.
__device__ __forceinline__ float3 dc_apply_vector(const float m[12], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2]  * v.z,
		m[4] * v.x + m[5] * v.y + m[6]  * v.z,
		m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

// Object -> world for a normal is the TRANSPOSE of world -> object (same
// reasoning as disk_cylinder_hittable.h's apply_normal), so this reads w2o
// column-wise rather than taking o2w row-wise.
__device__ __forceinline__ float3 dc_apply_normal_from_w2o(const float w2o[12], const float3& n) {
	return make_float3(
		w2o[0] * n.x + w2o[4] * n.y + w2o[8]  * n.z,
		w2o[1] * n.x + w2o[5] * n.y + w2o[9]  * n.z,
		w2o[2] * n.x + w2o[6] * n.y + w2o[10] * n.z);
}

// One candidate root: in range, inside the object-space Z clip, and inside
// the phi sweep. Mirrors CylinderShape<T>::intersect's check() lambda
// (src/shared/shapes.h) exactly, ported to float/CUDA math rather than
// instantiating that template device-side - see this project's own precedent
// for why (optix_intersection_sphere.h's gpu_cloud_density comment).
__device__ __forceinline__ bool dc_cylinder_check(const CylinderData& cyl, const float3& ro, const float3& rd,
													float t, float ray_tmin, float ray_tmax) {
	if (t < ray_tmin || t > ray_tmax) return false;
	const float hz = ro.z + t * rd.z;
	if (hz < cyl.zMin || hz > cyl.zMax) return false;
	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	return phi <= cyl.phiMax;
}

// World-space representative scale factor from an object->world transform's
// linear part - the length of the transformed local-X unit vector. Exact
// for a similarity transform (uniform scale, any rotation/translation);
// an approximation under anisotropic scale, same accepted simplification
// this whole file's header comment describes.
__device__ __forceinline__ float dc_representative_scale(const float o2w[12]) {
	const float3 x_axis = make_float3(o2w[0], o2w[4], o2w[8]);
	return length(x_axis);
}

// World-space area, mirroring DiskShape<T>::area() (src/shared/shapes.h)
// scaled by the square of the representative scale (area is a squared-length
// quantity).
__device__ __forceinline__ float dc_area_disk(const DiskData& disk) {
	const float scale = dc_representative_scale(disk.o2w);
	const float objArea = disk.phiMax * 0.5f * (disk.radius * disk.radius - disk.innerRadius * disk.innerRadius);
	return objArea * scale * scale;
}

// Mirrors CylinderShape<T>::area() = (zMax-zMin)*radius*phiMax.
__device__ __forceinline__ float dc_area_cylinder(const CylinderData& cyl) {
	const float scale = dc_representative_scale(cyl.o2w);
	const float objArea = (cyl.zMax - cyl.zMin) * cyl.radius * cyl.phiMax;
	return objArea * scale * scale;
}

// Uniform-area sample in WORLD space. Mirrors DiskShape<T>::sample(): draws
// a concentric-mapped point on the FULL disk of radius `disk.radius`, not
// clipped to the annulus/phiMax sweep - same CPU-parity limitation
// DiskShape<T>::sample() itself already has (the returned pdf_area() = 1/
// area() accounts for the annulus/sweep in aggregate, even though any single
// sampled point isn't individually clipped to it); ported as-is rather than
// fixed, since fixing it would diverge from CPU's own real NEE behavior.
// Takes u0/u1 directly (not a seed) - mirrors blp_sample()'s own signature
// (src/shared/bilinear_patch.h). This header is included BEFORE optix_
// device_helpers.h (which is where random_float() lives), by design - see
// this file's own header comment - so the caller (sample_disk_light(),
// optix_device_helpers.h) draws the randoms and passes them in.
__device__ __forceinline__ void dc_sample_disk(const DiskData& disk, float u0, float u1,
												float3& out_point, float3& out_normal, float& out_area_pdf) {
	float dx, dy;
	SampleUniformDiskConcentric<float>(u0, u1, dx, dy);
	const float3 obj_point = make_float3(dx * disk.radius, dy * disk.radius, disk.height);
	out_point = dc_apply_point(disk.o2w, obj_point);
	out_normal = normalize(dc_apply_normal_from_w2o(disk.w2o, make_float3(0.0f, 0.0f, 1.0f)));
	const float area = dc_area_disk(disk);
	out_area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;
}

// Uniform-area sample in WORLD space. Mirrors CylinderShape<T>::sample():
// uniform Z along the axis, uniform phi within the sweep.
__device__ __forceinline__ void dc_sample_cylinder(const CylinderData& cyl, float u0, float u1,
													float3& out_point, float3& out_normal, float& out_area_pdf) {
	const float z = cyl.zMin + u0 * (cyl.zMax - cyl.zMin);
	const float phi = u1 * cyl.phiMax;
	const float3 obj_normal = make_float3(cosf(phi), sinf(phi), 0.0f);
	const float3 obj_point = make_float3(cyl.radius * obj_normal.x, cyl.radius * obj_normal.y, z);
	out_point = dc_apply_point(cyl.o2w, obj_point);
	out_normal = normalize(dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const float area = dc_area_cylinder(cyl);
	out_area_pdf = (area > 1e-12f) ? (1.0f / area) : 0.0f;
}

// Solid-angle PDF of NEE having sampled the point a BSDF-sampled ray just
// hit, for MIS (recursive backend only - see wavefront's simpler emission-
// weighting scheme, wavefront_kernels.cu). Re-intersects `wi` (from `ctx`)
// against the disk in OBJECT space (same transform-the-ray-by-hand technique
// __intersection__disk uses), converts the found hit's area-domain pdf
// (1/area) to solid angle via the standard dist^2/cos Jacobian. Returns 0 if
// the ray misses the disk (can happen for a non-uniformly-scaled placement,
// where world-space BSDF sampling and this object-space re-intersection can
// disagree at the margins - same order of approximation as this whole
// file's area estimate).
__device__ __forceinline__ float dc_pdf_disk(const DiskData& disk, const float3& ctx,
											  const float3& wi_world) {
	const float3 ro = dc_apply_point(disk.w2o, ctx);
	const float3 rd = dc_apply_vector(disk.w2o, wi_world);
	if (rd.z == 0.0f) return 0.0f;
	const float t = (disk.height - ro.z) / rd.z;
	if (t <= 1e-4f) return 0.0f;
	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	const float dist2 = hx * hx + hy * hy;
	if (dist2 > disk.radius * disk.radius || dist2 < disk.innerRadius * disk.innerRadius) return 0.0f;
	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	if (phi > disk.phiMax) return 0.0f;

	// World-space hit distance and cosine, not the object-space t: the ray
	// direction transform (dc_apply_vector) isn't renormalised, so t itself
	// survives unmodified between spaces (this file's header comment), but
	// wi_world is assumed pre-normalised by the caller, matching every other
	// *_light_pdf-style computation in this codebase.
	const float world_dist2 = t * t;
	const float3 world_normal = normalize(dc_apply_normal_from_w2o(disk.w2o, make_float3(0.0f, 0.0f, 1.0f)));
	const float cosine = fabsf(dot(wi_world, world_normal));
	if (cosine < 1e-6f) return 0.0f;
	const float area = dc_area_disk(disk);
	if (area < 1e-12f) return 0.0f;
	return world_dist2 / (cosine * area);
}

// Same shape as dc_pdf_disk, cylinder side - reuses dc_cylinder_check() for
// the object-space root validity test, matching __intersection__cylinder's
// own two-root logic.
__device__ __forceinline__ float dc_pdf_cylinder(const CylinderData& cyl, const float3& ctx,
												   const float3& wi_world) {
	const float3 ro = dc_apply_point(cyl.w2o, ctx);
	const float3 rd = dc_apply_vector(cyl.w2o, wi_world);
	const double da = (double)rd.x, db = (double)rd.y;
	const double oa = (double)ro.x, ob = (double)ro.y;
	const double a = da * da + db * db;
	const double b = 2.0 * (oa * da + ob * db);
	const double c = oa * oa + ob * ob - (double)cyl.radius * (double)cyl.radius;
	if (a == 0.0) return 0.0f;
	const double f = b / (2.0 * a);
	const double vx = oa - f * da, vy = ob - f * db;
	const double len_v = sqrt(vx * vx + vy * vy);
	const double discrim = 4.0 * a * ((double)cyl.radius + len_v) * ((double)cyl.radius - len_v);
	if (discrim < 0.0) return 0.0f;
	const double sqrt_disc = sqrt(discrim);
	const double q = (b < 0.0) ? -0.5 * (b - sqrt_disc) : -0.5 * (b + sqrt_disc);
	float t0 = (float)(q / a), t1 = (float)(c / q);
	if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
	const float ray_tmin = 1e-4f, ray_tmax = 1e30f;
	if (t0 > ray_tmax || t1 < ray_tmin) return 0.0f;

	float t = t0;
	bool hit = dc_cylinder_check(cyl, ro, rd, t0, ray_tmin, ray_tmax);
	if (!hit) { t = t1; hit = dc_cylinder_check(cyl, ro, rd, t1, ray_tmin, ray_tmax); }
	if (!hit) return 0.0f;

	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	const float3 obj_normal = make_float3(hx / cyl.radius, hy / cyl.radius, 0.0f);
	const float3 world_normal = normalize(dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const float cosine = fabsf(dot(wi_world, world_normal));
	if (cosine < 1e-6f) return 0.0f;
	const float area = dc_area_cylinder(cyl);
	if (area < 1e-12f) return 0.0f;
	return (t * t) / (cosine * area);
}
