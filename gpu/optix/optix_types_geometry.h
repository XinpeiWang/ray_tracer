#pragma once
// optix_types_geometry.h -- part 1 of 5 of optix_types.h (included by it, in order; not meant to be included on its own).

// Ray payload for path tracing
struct PathTracingPayload {
	float3 attenuation;   // Color filter (throughput)
	float3 emission;      // Emitted light
	float3 scatterOrigin; // Next ray origin
	float3 scatterDir;    // Next ray direction
	unsigned int seed;    // Random number seed
	int depth;            // Current bounce depth
	bool scattered;       // True if ray scattered (not absorbed)
	// Denoiser guide-layer AOVs (recursive backend only) - only meaningful
	// on the primary ray (depth==0); every closest-hit/miss program packs
	// them unconditionally (p16-p21, see optix_raygen.h's own comment)
	// regardless of depth, since a hit program has no way to know which
	// bounce it's shading - raygen is the one that decides whether to
	// accumulate them, exactly mirroring how ray-differential dudx/dvdx
	// already only matters (and is only consumed) on depth==0.
	float3 albedo;
	float3 normal;
	// pbrt-v4's per-event refraction ratio (eta_i/eta_t, front_face ?
	// 1/ior : ior) for this bounce's scatter event - 1.0f (a no-op) unless
	// this hit was a genuine transmission through a dielectric interface.
	// optix_raygen.h accumulates eta_scale *= eta*eta across every bounce
	// (pbrt-v4 PathIntegrator: `if (bs->IsTransmission()) etaScale *=
	// Sqr(bs->eta);`) and folds it into the Russian Roulette throughput
	// test, matching CPU's camera.h eta_scale exactly - without it, RR
	// terminates transmission-heavy (glass) paths too aggressively. Packed
	// into payload register p22 (see optix_raygen.h): unlike p16-p21,
	// which every closest-hit/miss program packs unconditionally, only the
	// programs/branches that produce a real transmission event ever call
	// optixSetPayload_22 - everywhere else the register keeps whatever
	// value optix_raygen.h initialized it to before the trace call (1.0f),
	// the same "closest-hit only writes, unset means input survives"
	// convention p12 already relies on for __miss__ms.
	float eta;
};

// Shadow ray payload (minimal - just occlusion result)
struct ShadowPayload {
	bool occluded;  // True if ray hit anything (path is blocked)
};

// Which analytic shape a SphereData entry's near/far intersection math uses.
// Sphere (0) is the default - every pre-existing SphereData{} zero-init (the
// overwhelming majority of call sites) lands here with no changes needed, and
// is the quadratic ray/sphere test __intersection__sphere/__closesthit__sphere
// (optix_intersection_sphere.h) have always used. Box (1) is a standard
// ray/AABB slab test instead, for the few scenes whose medium boundary is
// actually an axis-aligned box rather than a sphere - e.g. scene 21's wax
// slab (see build_subsurface_slab_gpu's comment in scene_builder.cpp), which
// used to be approximated as a sphere sized to roughly match its footprint
// (bulging through the box's side walls while falling short of its floor/
// ceiling). Fixing this geometric mismatch was originally hypothesized to be
// the dominant cause of scene 21's ~32-38% CPU/GPU brightness gap, but a
// direct isolated-render A/B measurement (sphere vs box, GPU-recursive only,
// same seeds) showed the shape change moves average brightness by well under
// 1% (0.24428 -> 0.24566) - nowhere near enough to explain the ~30% gap. The
// gap's real dominant cause remains unexplained; this fix is still correct
// and worth keeping (it makes the rendered geometry actually match CPU's box
// instead of a bulging/undershooting sphere), it just isn't the fix for the
// brightness discrepancy itself. Deliberately additive: every
// field below is reused as-is regardless of shapeKind (materialIdx, the GAS/
// SBT/instancing/shadow-ray machinery, ...) - only the geometric test
// changes. center/center1/radius are unused (left zero) when shapeKind==Box;
// boxMin/boxMax are unused (left zero) when shapeKind==Sphere.
// ClippedSphere (2): pbrt-v4's real "float zmin"/"float zmax"/"float phimax"
// on Shape "sphere" (caps/wedges/hemispheres, e.g. a domed skylight cutout) -
// previously silently dropped on GPU (rendered as the full unclipped
// sphere), matching CPU's own now-fixed gap (see pbrt_flatten::Sphere::
// clipped's own comment for the history). Unlike Sphere/Box above, a
// ClippedSphere is NOT rotation-invariant, so it carries its own object<->
// world affine transform (SphereData::o2w/w2o below) exactly the way
// Disk/Cylinder already do (see DiskData's own comment) - boxMin/boxMax are
// unused (left zero) for this kind; the real hit-testing object-space
// geometry lives in radiusLocal/zMin/zMax/phiMax instead, and motion blur is
// not supported in combination with clipping (CPU doesn't support it either
// - a clipped sphere is never also a moving one in this loader's own
// pbrt_flatten.h). center/center1/radius ARE populated for this kind too (a
// real world-space center + baked "largest axis" full-sphere radius, same
// as every other sphere) but ONLY for the pre-existing full-sphere-cone NEE
// machinery (sample_area_light_by_kind, DiffuseLight solid-angle pdf) to
// reuse unmodified when this sphere is also an AreaLightSource - the exact
// same accepted CPU approximation documented in sphere_clipped_hittable.h's
// own header comment (noise, not bias: some sampled directions land on the
// clipped-away region and contribute nothing, since real hit-testing still
// only consults radiusLocal/o2w/w2o and correctly rejects them).
enum class GpuMediumShapeKind : int { Sphere = 0, Box = 1, ClippedSphere = 2 };

// Sphere geometry data (custom primitive)
struct SphereData {
	float3 center;   // position at ray-time t=0 (the only position, for a static sphere)
	// Position at ray-time t=1, for motion blur (RTIOW-style linear "bounce"
	// interpolation, matching src/TheRestOfYourLife/sphere.h's moving-sphere
	// constructor: current_center = lerp(center, center1, ray.time())).
	// Left zero-initialized (garbage) for every static sphere in every scene
	// that doesn't use motion blur - always safe, because the intersection
	// program only ever multiplies it by a ray-time that is provably 0.0f
	// for those scenes (see optix_raygen.h), and lerp(a, b, 0) == a exactly
	// regardless of b. Only scenes that actually build moving spheres (with
	// center1 != center) need to also enable GpuCameraParams::motionBlurEnabled
	// and give their sphere GAS build 2 motion keys - see
	// OptiXRenderer::buildScene()'s sceneHasMotion_ detection. Unused (left
	// zero) for shapeKind==Box - no scene combines motion blur with a box
	// medium boundary, and center1==center==0 trivially keeps
	// sceneHasMotion_ detection from false-triggering on a box entry.
	float3 center1;
	float radius;
	int materialIdx;
	// See GpuMediumShapeKind's own comment above. Zero-init default (every
	// pre-existing `SphereData s{}` call site) is GpuMediumShapeKind::Sphere.
	GpuMediumShapeKind shapeKind;
	// Box (shapeKind==Box) world-space (equivalently object-space - every
	// scene using this is non-instanced, see GpuMediumShapeKind's comment)
	// min/max corners, mirroring the CPU box() representation exactly
	// (src/TheRestOfYourLife/quad.h) - an axis-aligned box from corner `a`
	// to corner `b`. Unused (left zero) for shapeKind==Sphere.
	float3 boxMin;
	float3 boxMax;

	// ClippedSphere (shapeKind==ClippedSphere) only - object-space partial-
	// sphere parameters, matching pbrt_flatten::Sphere::radiusLocal/zMin/
	// zMax/phiMaxDeg (radians here, converted host-side) exactly, plus the
	// object<->world affine transform (row-major 3x4, same convention as
	// DiskData::o2w/w2o - see that struct's own comment for the full
	// rationale) computed once host-side from pbrt_flatten::Sphere::xform.
	// Unused (left zero) for shapeKind==Sphere/Box. (center/center1/radius
	// above ARE used for ClippedSphere too, but only by NEE sampling, not
	// hit-testing - see GpuMediumShapeKind::ClippedSphere's own comment.)
	float radiusLocal;
	float zMin, zMax;
	float phiMax;
	float o2w[12];
	float w2o[12];
	// thetaZMin/thetaZMax: acos(clamp(zMax/radiusLocal))/acos(clamp(zMin/
	// radiusLocal)) - the v-coordinate normalization bounds SphereShape<T>::
	// intersect() (src/shared/shapes.h) derives from zMin/zMax/radiusLocal
	// every call. Depends only on those three host-constant fields, so it's
	// computed once here (pbrt_gpu_builder.h) instead of by every closest-hit
	// program/NEE sample against this primitive - see __closesthit__sphere's
	// own v-coordinate comment for why re-deriving it per-hit was flagged as
	// wasted work.
	float thetaZMin, thetaZMax;
};

// Quad geometry data (custom primitive)
struct QuadData {
	float3 Q;  // Corner point
	float3 u;  // First edge vector
	float3 v;  // Second edge vector
	float3 normal;  // Precomputed normal
	float D;        // Plane constant
	float3 w;       // Cross product u x v
	int materialIdx;
};

// Bilinear patch geometry data (custom primitive) - a genuinely curved
// (non-planar in general) ruled surface through 4 corners, NOT a flat quad.
// p(u,v) = lerp(u, lerp(v,p00,p01), lerp(v,p10,p11)); p00=(u=0,v=0),
// p10=(u=1,v=0), p01=(u=0,v=1), p11=(u=1,v=1) - matches pbrt-v4's
// BilinearPatch / src/shared/bilinear_patch.h convention. See
// optix_intersection_bilinear_patch.h for the ray-intersection algorithm
// (Ramsey et al. 2004, ported from bilinear_patch.h's blp_intersect).
struct BilinearPatchData {
	float3 p00, p10, p01, p11;
	int materialIdx;
};

// Disk/Cylinder geometry data (custom primitives) - Shape "disk"/"cylinder"
// from a loaded .pbrt scene (see src/shared/pbrt_flatten.h's Disk/Cylinder
// and src/TheRestOfYourLife/disk_cylinder_hittable.h, the CPU backend these
// mirror). Unlike every other shape here, these are NOT baked to world
// space - a disk/cylinder is not rotation-invariant the way a sphere is, so
// baking would mean either re-deriving Sphere's own "warn under anisotropic
// scale" approximation or getting rotation silently wrong. Instead each
// carries its own object<->world affine transform (o2w/w2o, row-major 3x4 -
// same convention as SceneData::InstancePlacementGPU::transform), computed
// once host-side (pbrt_gpu_builder.h, via pbrt_scene::Matrix4::
// inverseAffine()) from the flat scene's raw double xform[16]. The device
// intersection/closest-hit programs (gpu/optix/optix_intersection_disk_
// cylinder.h) apply these manually - carrying the RAY into object space,
// exactly mirroring the CPU hittable's own technique - rather than through
// OptiX's per-GAS-instance transform, since neither shape is instanced
// (no ObjectInstance support for them yet, matching the CPU loader's own
// current scope): giving each one its own GAS+IAS-instance just to reach
// OptiX's instance-transform machinery would be heavier than applying the
// same 3x4 by hand in the two device programs that need it.
struct DiskData {
	float radius;       // outer radius
	float innerRadius;  // 0 for a solid disk
	float height;       // object-space z of the disk's plane
	float phiMax;       // azimuthal sweep, RADIANS (converted from pbrt's degrees host-side)
	int materialIdx;
	float o2w[12];  // object -> world
	float w2o[12];  // world -> object
};

struct CylinderData {
	float radius;
	float zMin, zMax;   // object-space Z extent (axis is object-space Z)
	float phiMax;       // azimuthal sweep, RADIANS
	int materialIdx;
	float o2w[12];
	float w2o[12];
};

// Triangle geometry data (native OptiX triangle, see optix_renderer.cpp's
// buildAccelerationStructure). Shading normal is per-vertex-interpolated
// (barycentric, via optixGetTriangleBarycentrics()) when the source mesh
// had "vn" data - n0/n1/n2 - and falls back to the flat geometric normal
// cross(e1,e2) when hasNormals is false (e.g. scene 37's procedural
// icosahedron, or any mesh scene whose source .obj has no vn lines - most
// of them; Suzanne, scene 45, was the first to actually exercise this
// path). Mirrors src/TheRestOfYourLife/triangle.h's CPU
// has_normals()-gated interpolation exactly.
// uv0/uv1/uv2: per-vertex texture coordinates ("vt" data), barycentric-
// interpolated the same way as n0/n1/n2 when hasUVs is set (see
// optix_intersection_triangle.h) - feeds MaterialData::textureIdx image
// sampling for meshes with a real map_Kd texture (pbrt_scenes/environment-*.pbrt
// reading a models/*.obj through Shape "plymesh"). Meshes
// with no "vt" data (or whose material has no textureIdx) leave hasUVs
// false and uv0-2 unused, matching hasNormals' same opt-in pattern.
struct TriangleData {
	float3 p0, p1, p2;
	float3 n0, n1, n2;
	float2 uv0, uv1, uv2;
	int materialIdx;
	bool hasNormals;
	bool hasUVs;
};

// Which array a sampled area light lives in, and therefore how to sample it.
//
// Explicitly : int, and the width is load-bearing. The host uploads one of
// these per light and the device reads the same buffer back through a
// pointer; when those two widths disagreed (this was a bool* over an int
// buffer) only light 0 landed on its own entry - lights 1..3 read the upper,
// always zero, bytes of light 0's value and so all looked like the zero kind.
// A sphere light misread that way is then looked up in params.quads, which in
// a scene with no quads at all is an out-of-bounds read: an illegal memory
// access that kills the launch outright, not a subtle shading difference.
// optix_renderer.cpp static_asserts the two widths against each other at the
// upload site; wavefront_types.h and sppm_types.h carry their own copies of
// the pointer and must stay this type too.
//
// Quad is deliberately 0 so the historical false==quad / true==sphere
// encoding is preserved for anything still reasoning in those terms.
enum class GpuLightKind : int {
	Quad = 0,
	Sphere = 1,
	// A single emissive triangle. Most pbrt area lights arrive as a pair of
	// triangles that pbrt_quadify.h rejoins into one parallelogram and which
	// therefore become Quad; this kind is for the ones that will not merge -
	// an odd triangle, a fan, anything non-parallelogram - which used to be
	// emitted as geometry that glows when hit but that next-event estimation
	// could not aim at, leaving the GPU image darker and noisier than the CPU
	// one for no reason the picture explained.
	Triangle = 2,
	// Shape "bilinearmesh" carrying an AreaLightSource - real published pbrt
	// scenes use this for non-planar/non-rectangular light panels (e.g.
	// sportscar-area-lights.pbrt's studio softboxes), which pbrt_quadify.h
	// cannot fold into a Quad because they are not necessarily coplanar.
	// Same "used to glow but not be aimable" gap Triangle closed, for the one
	// shape kind that gap didn't cover.
	BilinearPatch = 3,
	// Shape "disk"/"cylinder" carrying an AreaLightSource - same "used to
	// glow but not be aimable" gap Triangle/BilinearPatch closed, for the
	// last two shape kinds that gap didn't cover. Sampling/pdf go through
	// optix_disk_cylinder_helpers.h's dc_sample_disk/dc_pdf_disk (real
	// device-safe ports of src/shared/shapes.h's DiskShape<T>/pdf_from, not
	// direct template instantiation - see that header's own comment for why).
	Disk = 4,
	Cylinder = 5,
};
// Order in which the pbrt GPU builder registers lights, shape kind by shape kind, each kind in ascending primitive
// index. gpu_light_sort_key() reproduces that order, so the builder's final sort is a no-op in practice, and a
// light can be found from the primitive a ray hit by binary search (gpu_find_light) instead of a scan.
CPU_GPU inline int gpu_light_kind_rank(GpuLightKind k) {
	switch (k) {
		case GpuLightKind::Sphere:        return 0;
		case GpuLightKind::BilinearPatch: return 1;
		case GpuLightKind::Disk:          return 2;
		case GpuLightKind::Cylinder:      return 3;
		case GpuLightKind::Quad:          return 4;
		default:                          return 5;   // Triangle
	}
}
CPU_GPU inline long long gpu_light_sort_key(GpuLightKind k, int primIdx) {
	return ((long long)gpu_light_kind_rank(k) << 32) | (long long)(unsigned int)primIdx;
}
// Index of the light for (kind, primIdx) in lightIndices/lightKinds, which must be sorted by gpu_light_sort_key;
// -1 when that primitive is not a sampled light.
CPU_GPU inline int gpu_find_light(const int* lightIndices, const GpuLightKind* lightKinds, unsigned int numLights,
								  GpuLightKind kind, int primIdx) {
	const long long key = gpu_light_sort_key(kind, primIdx);
	int lo = 0, hi = (int)numLights - 1;
	while (lo <= hi) {
		const int mid = lo + (hi - lo) / 2;
		const long long k = gpu_light_sort_key(lightKinds[mid], lightIndices[mid]);
		if (k == key) return mid;
		if (k < key) lo = mid + 1; else hi = mid - 1;
	}
	return -1;
}
