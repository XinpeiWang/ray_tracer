// wavefront_types.h
// Queue types and work items for the wavefront GPU path tracer.
#pragma once

// When compiled by nvcc or with OptiX, pull in the real headers.
// When included from a plain C++ host-only context (e.g. unit tests without
// the CUDA toolkit on the include path), provide minimal stubs so the POD
// struct layouts can be inspected without the GPU toolchain.
#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_types.h"
#else
// Minimal float3 stub for host-only compilation
#ifndef __VECTOR_TYPES_H__
struct float3 { float x, y, z; };
#endif
#endif

// ============================================================================
// Work item types
// ============================================================================

// Number of hero wavelengths per ray (matches pbrt-v4 NSpectrumSamples = 4)
static constexpr int kWFNWavelengths = 4;

// A primary or bounce ray pending intersection.
struct RayWorkItem {
	float3       origin;
	float3       direction;
	// Spectral path weight (SampledSpectrum<4>): starts at {1,1,1,1}
	float        throughput[kWFNWavelengths];
	// Accumulated spectral radiance (SampledSpectrum<4>) for this path
	float        radiance[kWFNWavelengths];
	// Hero wavelengths (nm) sampled at camera ray generation
	float        wavelengths[kWFNWavelengths];
	// Sampling PDFs for each hero wavelength
	float        wavelength_pdfs[kWFNWavelengths];
	unsigned int seed;             // PCG state
	int          pixelIndex;       // flat pixel index (y*width + x)
	int          depth;            // current bounce (0 = primary)
	int          specular_bounce;  // 1 if this ray was spawned by a specular event
	// 1 once any PRIOR bounce along this path was non-specular (0 for the
	// primary ray - see wf_generate_primary_ray). Read by evaluate_materials()/
	// evaluate_materials_dielectric() as the do_regularize flag for the 5
	// GGX-distributed materials (Conductor/RoughMetal/RoughDielectric/
	// CoatedDiffuse/CoatedConductor - see wf_regularize_alpha's own comment,
	// wavefront_kernels.cu), then updated (never cleared) in
	// wf_finish_material_scatter's next-ray tail from THIS bounce's own
	// is_specular - mirrors camera.h's any_nonspecular local exactly (see
	// that file's ray_color() loop), just carried across bounce iterations
	// via this field instead of a stack-resident C++ local, since each
	// bounce here is a separate kernel launch.
	int          any_nonspecular;
	// pbrt-v4 etaScale: product of eta^2 over every transmission event so
	// far along this path (PathIntegrator: `if (bs->IsTransmission())
	// etaScale *= Sqr(bs->eta);`), read into the Russian Roulette throughput
	// test in wf_finish_material_scatter - matches CPU's camera.h eta_scale
	// and the recursive backend's PathTracingPayload::eta/optix_raygen.h
	// eta_scale exactly. 1.0f for the primary ray (see wf_generate_primary_
	// ray) - same "carried across bounce iterations via this field instead
	// of a stack-resident C++ local" reasoning as any_nonspecular above,
	// since each bounce is a separate kernel launch here.
	float        etaScale;
	// Pixel reconstruction filter weight for the sample this path belongs
	// to (gpu_filter_evaluate() in wavefront_kernels.cu) - a pure
	// reconstruction/blending weight, unrelated to how much light this path
	// carries. Deliberately carried as its OWN field, separate from
	// throughput: an earlier version of this feature folded it into
	// throughput/cam_weight the way the recursive backend's cam_weight
	// already works, which was a real, confirmed bug - Gaussian's own
	// evaluate() is tiny in absolute magnitude (well under 1 even at dead
	// center), so contaminating throughput with it made Russian Roulette
	// see a near-zero beta from the first eligible bounce onward and kill
	// the overwhelming majority of paths almost immediately (unbiased in
	// expectation via RR's 1/(1-q) reweight, but a variance explosion that
	// made any scene needing several bounces to converge render close to
	// black at ordinary sample counts). Every atomicAdd into framebuffer
	// (evaluate_materials's several branches, accumulate_miss,
	// accumulate_shadow, resolve_bssrdf_exit) multiplies its own
	// contribution by this field directly, exactly mirroring CPU camera.h's
	// `weighted_color += w * sample;` - the filter weight touches only the
	// FINAL contribution, never a transport decision.
	float        filterWeight;
	// BRDF PDF of this ray's own scatter direction, for MIS if it escapes the
	// scene on the next bounce (accumulate_miss's own comment) - mirrors the
	// recursive backend's prev_brdf_pdf (optix_raygen.h/optix_miss.h). 0 for
	// the primary ray or a specular bounce (same sentinel meaning as
	// specular_bounce above - "no MIS, full weight"), cosine_pdf(direction,
	// normal) otherwise.
	float        brdf_pdf;
	float        tMin;             // ray t_min (normally 0.001)
	float        tMax;             // ray t_max (normally 1e30)
};

// Result of a successful intersection, produced by OptiX closesthit.
struct HitWorkItem {
	// Surface geometry
	float3 hitPoint;
	float3 normal;             // shading normal (outward)
	float  t;                  // ray parameter at hit

	// Material info
	int    materialIdx;
	int    geomType;           // 0 = sphere, 1 = quad, 2 = bilinear patch, 3 = triangle, 4 = disk, 5 = cylinder (read in the NormalMappedLambertian case only - see evaluate_materials())

	// MaterialType::Medium and MaterialType::DielectricMedium (exit surface,
	// i.e. front-facing false) only: `t` above holds the entry (near) root of
	// the medium sphere's boundary; this holds the exit (far) root. Both are
	// recomputed in closesthit relative to the CURRENT ray origin (handles a
	// ray that already starts inside the sphere, e.g. continuing after a
	// prior in-medium scatter event) - see __closesthit__wf_sphere. Unused
	// (0) for every other material type, and for DielectricMedium's entry
	// surface (frontFace true - see that field below).
	float  mediumTFar;

	// True if the ray hit the front (outward-facing) side of the surface,
	// i.e. the same thing `front_face` means in optix_intersection_sphere.h.
	// Lost once `normal` below is flipped to always oppose the incident ray
	// (see __closesthit__wf_sphere), so it has to travel separately -
	// MaterialType::DielectricMedium needs it to pick between its entry
	// (refract/reflect the dielectric surface) and exit (re-enter the
	// internal medium, or exit-refract) behaviour. Meaningless (left 0) for
	// every material that doesn't branch on it.
	int    frontFace;

	// Dual-purpose carrier for a real surface tangent (dpdu) - named objDpdu
	// (not objNormal, its name before this field stopped ever holding a
	// normal for any geomType) since neither this struct nor WfHitPayload
	// otherwise carries per-vertex position/UV data to recompute one later.
	// Used by MaterialType::NormalMappedLambertian
	// (its normal-map basis) and by the 4 anisotropy-capable material kinds
	// (Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor - their
	// UV-aligned shading frame, see BuildDpduTangentFrame's own comment,
	// microfacet.h). Every geomType now populates this with a real,
	// ready-to-use WORLD-space dpdu at intersection time - no per-consumer
	// derivation needed, unlike this field's earlier sphere-only "raw
	// object-space normal, derive an approximate cross-product tangent
	// later" convention:
	//   - Sphere: real analytic dpdu (theta/phi parametric derivative,
	//     matching CPU's sphere.h exactly - see __closesthit__wf_sphere's
	//     own comment). Previously this held the raw outward normal instead,
	//     with dpdu only approximated (cross(world_up, normal)) by whichever
	//     consumer read it.
	//   - Quad: q.u (the quad's own edge vector) - already exactly CPU's
	//     quad.h dpdu, no derivation needed.
	//   - Triangle: the UV-gradient-solved real dpdu (see
	//     __closesthit__wf_triangle).
	//   - Disk/Cylinder: real analytic phi-tangent dpdu (see each
	//     __closesthit__wf_*'s own comment, matching CPU's
	//     disk_cylinder_hittable.h). Previously left zero entirely.
	//   - Bilinear patch: the patch's own real dpdu, left UNNORMALIZED
	//     (unlike every other geomType above) - the normalize() is deferred
	//     to evaluate_materials()'s own Hair case, the only reader that
	//     currently exists for this geomType, so the (much more common)
	//     non-Hair bilinear-patch hit doesn't pay for it.
	float3 objDpdu;

	// Surface texture coordinates. Sphere: standard spherical (theta,phi)
	// mapping, computed in __closesthit__wf_sphere from the object-space
	// normal (a local, not this struct's own objDpdu field), matching
	// CPU's get_sphere_uv() and optix_intersection_sphere.h's sphere_uv_u/v.
	// Triangle: barycentric-interpolated from TriangleData::uv0/1/2 when
	// tri.hasUVs (matches optix_intersection_triangle.h), else 0. Quad: no
	// per-quad UV exists on this codebase's GPU side (matches the recursive
	// path, which never textures a quad either), always 0. Only meaningful
	// for MaterialType::Lambertian/NormalMappedLambertian with
	// MaterialData::textureIdx >= 0.
	float  uv_u, uv_v;

	// Incident ray (needed to evaluate material)
	float3 rayOrigin;
	float3 rayDir;
	// Spectral path state (SampledSpectrum<4>)
	float  throughput[kWFNWavelengths];
	float  radiance[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	unsigned int seed;
	int    pixelIndex;
	int    depth;
	int    specular_bounce;    // 1 if this path arrived via a specular event
	// Carried from RayWorkItem::any_nonspecular (see its own comment) - the
	// do_regularize flag for this hit's material evaluation.
	int    any_nonspecular;
	// Carried from RayWorkItem::etaScale (see its own comment) - the
	// accumulated-so-far value evaluate_materials()/evaluate_materials_
	// dielectric() fold this hit's own transmission eta (if any) into
	// before passing to wf_finish_material_scatter.
	float  etaScale;
	// Carried from RayWorkItem::filterWeight (see its own comment) -
	// multiplied into every radiance contribution this hit produces.
	float  filterWeight;
	// Carried from RayWorkItem::brdf_pdf (see its own comment) - the BRDF
	// pdf of the ray that arrived at THIS hit. Not read by the DiffuseLight
	// early-exit above (that gate is specular_bounce/depth==0 only, no real
	// MIS blend - see evaluate_materials()'s own comment), only added here
	// so MaterialType::Interface's pass-through case can propagate it
	// unchanged into the NEXT RayWorkItem it pushes directly (bypassing
	// wf_finish_material_scatter's own recompute) - without this, a path
	// that crosses an interface boundary and then escapes to the sky on
	// its very next segment would lose its real MIS pdf and get treated as
	// a specular bounce (full weight, no MIS) instead.
	float  brdf_pdf;
};

#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
// Copies the RayWorkItem fields that a genuine pass-through (no real scatter
// event - e.g. MaterialType::Interface) carries UNCHANGED from the hit that
// produced it, rather than recomputing them the way a real scatter event
// does (wf_finish_material_scatter(), wavefront_kernels.cu, sets these same
// 7 field names to NEW values instead - depth+1, a freshly sampled
// specular_bounce/brdf_pdf, etc. - so it does not call this). Callers still
// set origin/direction/seed/throughput/radiance/wavelengths/wavelength_pdfs/
// tMin/tMax themselves; this only covers the fields a pass-through leaves
// untouched, so a future field added to both structs only needs updating
// here instead of at every hand-rolled pass-through call site.
__device__ __forceinline__ void wf_carry_ray_state(RayWorkItem& next, const HitWorkItem& h) {
	next.pixelIndex      = h.pixelIndex;
	next.depth           = h.depth;
	next.specular_bounce = h.specular_bounce;
	next.any_nonspecular = h.any_nonspecular;
	next.etaScale        = h.etaScale;
	next.filterWeight    = h.filterWeight;
	next.brdf_pdf        = h.brdf_pdf;
}
#endif

// A shadow ray: if it reaches tMax unoccluded, Ld is added to the framebuffer.
struct ShadowRayWorkItem {
	float3 origin;
	float3 direction;
	float  tMax;               // distance to light surface (use light.t - eps)
	// Spectral direct-light contribution (throughput * Le * BSDF / pdf)
	float  Ld[kWFNWavelengths];
	// Wavelengths needed for XYZ conversion at accumulation
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	int    pixelIndex;
};

// A miss result: the ray escaped, accumulate background/environment.
struct MissWorkItem {
	float  throughput[kWFNWavelengths];
	// Radiance already accumulated along this path before it missed (e.g. a
	// specular-bounce chain's emission from an earlier light hit - see
	// RayWorkItem::radiance's own comment). Non-specular bounces flush their
	// radiance at every hit (evaluate_materials's NEE block), but a specular
	// chain defers the flush until the path terminates, so a miss has to
	// carry it here or it's silently lost - accumulate_miss adds this on top
	// of the flat background term, not instead of it.
	float  radiance[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	float3 rayDir;             // for environment-map lookups (currently unused)
	int    pixelIndex;
	// Carried from RayWorkItem::brdf_pdf (see its own comment) - accumulate_miss
	// needs this to MIS-weight the sky's background contribution against the
	// sky-NEE strategy (evaluate_materials's own sky-NEE block), mirroring
	// optix_miss.h's recursive-backend equivalent.
	float  brdf_pdf;
};

// A MaterialType::Subsurface entry hit that transmitted through the entry
// dielectric interface (evaluate_materials's own Subsurface case,
// wavefront_kernels.cu) and needs a BSSRDF probe walk to find its exit
// point. Pushed instead of scattering directly, because the probe walk
// needs real optixTrace calls (optix_probe_hit.h's payload/hit-group
// pattern via wf_trace_probe_ray(), wavefront_probe.h) which are
// unavailable from evaluate_materials - a plain CUDA kernel, not an OptiX
// program. Consumed by __raygen__wf_probe (wavefront_probe.h), one thread
// per queued item, mirroring pbrt-v4's own SampleSubsurface/
// GetBSSRDFAndProbeRayWorkItem -> IntersectOneRandom staging.
//
// Carries the probe's own entry point/axis (p0/axis0 - see
// bssrdf_probe_walk()'s comment in optix_device_helpers.h for what these
// mean, this is the exact same 3-axis-MIS algorithm) plus everything
// needed to resume the path once (or if) an exit point is found - the same
// per-path state RayWorkItem itself carries, since a probe request is a
// path in flight just like any other queued ray.
struct BssrdfProbeWorkItem {
	float3 p0;      // probe segment center (Subsurface entry hit point)
	float3 axis0;   // probe axis seed (entry shading normal)
	// This material's own index into materials[] - identifies "same
	// material" candidates during the walk (bssrdf_probe_walk()'s own
	// matIdx parameter) and is looked up again to read
	// bssrdf_sigma_a/bssrdf_sigma_s/ior/textureIdx (-> bssrdfTables).
	int    matIdx;
	unsigned int seed;
	float  throughput[kWFNWavelengths];
	float  radiance[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	int    pixelIndex;
	int    depth;
	// Deliberately does NOT carry any_nonspecular (unlike RayWorkItem/
	// HitWorkItem): resolve_bssrdf_exit() always resumes the exit bounce as
	// MaterialType::NormalizedFresnel with is_specular=false, which
	// unconditionally sets the resumed path's own any_nonspecular to 1
	// regardless of what came before, and NormalizedFresnel is never one of
	// the 5 regularized GGX types either - so a carried-through flag here
	// would read correctly but influence nothing observable.
	//
	// UNLIKE any_nonspecular, DOES carry etaScale: the entry interface this
	// probe request follows was a genuine transmission event (see
	// evaluate_materials()'s Subsurface case), so its eta^2 contribution
	// must survive through the probe walk to the eventual exit bounce's RR
	// test - matches CPU camera.h's `entry_eta`, captured before the
	// exit-surface scatter overwrites srec.
	float  etaScale;
	// Also carried, same reasoning as etaScale - the eventual exit bounce's
	// own radiance contribution still needs this path's filter weight (see
	// RayWorkItem::filterWeight's own comment).
	float  filterWeight;
};

// Result of one BSSRDF probe walk (__raygen__wf_probe, wavefront_probe.h) -
// always pushed, whether or not a valid exit point was found (see `found`),
// so resolve_bssrdf_exit() (wavefront_kernels.cu) can flush this path's
// still-pending `radiance` (deferred specular-chain emission, same meaning
// as RayWorkItem::radiance) even on failure - otherwise it would be
// silently dropped, the same class of bug MissWorkItem::radiance's own
// comment warns about for a different case. On success, resolves as an
// ordinary MaterialType::NormalizedFresnel(ior=materials[matIdx].ior)
// non-specular bounce at exitPos/exitNormal, weighted by Sp/(sampleProb*pdf) -
// mirrors camera.h::sample_bssrdf_exit()'s hand-off and the recursive
// backend's identical shade_material() Subsurface-case hand-off exactly.
struct BssrdfExitWorkItem {
	int    found;        // 0/1 - see this struct's own comment
	float3 exitPos;
	float3 exitNormal;
	float3 Sp;            // BSSRDF profile value at the found exit distance (unbounded positive RGB, not a [0,1] albedo)
	float  pdf;            // combined 3-axis MIS pdf (bssrdf_probe_walk()'s out_pdf)
	float  sampleProb;     // 1/candidate_count from the walk's reservoir sampling
	int    matIdx;
	unsigned int seed;
	float  throughput[kWFNWavelengths];
	float  radiance[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	int    pixelIndex;
	int    depth;
	// See BssrdfProbeWorkItem's own comment - any_nonspecular deliberately
	// omitted here too, for the same reason.
	//
	// etaScale IS carried, same as BssrdfProbeWorkItem - see that struct's
	// own comment. resolve_bssrdf_exit() passes it through to wf_finish_
	// material_scatter() unchanged (the exit-surface NormalizedFresnel
	// shading doesn't add its own etaScale factor).
	float  etaScale;
	// filterWeight IS also carried, same as BssrdfProbeWorkItem - see
	// RayWorkItem::filterWeight's own comment.
	float  filterWeight;
};

// ============================================================================
// WorkQueue<T> — device-side append-only queue with an atomic size counter
// ============================================================================

template<typename T>
struct WorkQueue {
	T*            items;       // device pointer to item array
	int*          counter;     // device pointer to atomic counter (1 int)
	int           capacity;    // max items (= width * height)

	#ifdef __CUDACC__
	// Append an item from device code; returns the slot index or -1 if full.
	__device__ __forceinline__ int push(const T& item) {
		int slot = atomicAdd(counter, 1);
		if (slot < capacity) {
			items[slot] = item;
			return slot;
		}
		return -1;  // overflow (should not happen if capacity >= numPixels)
	}

	// Number of items currently in the queue (only meaningful after device sync).
	__host__ __device__ __forceinline__ int size() const {
#ifdef __CUDA_ARCH__
		return atomicAdd(counter, 0);
#else
		int val = 0;
		cudaMemcpy(&val, counter, sizeof(int), cudaMemcpyDeviceToHost);
		return val;
#endif
	}
#endif // __CUDACC__
};

// ============================================================================
// WavefrontQueues — all queues for one bounce iteration, plus the framebuffer
// ============================================================================

struct WavefrontQueues {
	WorkQueue<RayWorkItem>    rayQueue;       // rays pending intersection
	WorkQueue<RayWorkItem>    nextRayQueue;   // rays for the next bounce
	WorkQueue<HitWorkItem>    hitQueue;       // intersection results (complex materials)
	// Hits on cheap materials (Lambertian/Metal - no texture/layered-BxDF work),
	// routed here at push time (see wavefront_programs.cu) instead of hitQueue
	// so evaluate_materials_simple() can process them with a 2-way if/else
	// instead of the big switch, avoiding register pressure from the switch's
	// expensive arms (CoatedDiffuse/Principled/RoughDielectric/...) for rays
	// that don't need it. Same HitWorkItem layout as hitQueue - reused as-is
	// rather than a trimmed struct, since the memory win is marginal and a
	// second struct would mean duplicated fill-out/test code.
	WorkQueue<HitWorkItem>    simpleHitQueue;
	// Hits on MaterialType::Dielectric/RoughDielectric, routed here at push
	// time (see wavefront_programs.cu) for the same register-pressure reason
	// as simpleHitQueue - a 3rd tier alongside simpleHitQueue/hitQueue so
	// evaluate_materials_dielectric() only ever compiles the dielectric arms,
	// not the CoatedDiffuse/Principled/... arms of the big switch. Same
	// HitWorkItem layout, reused as-is.
	WorkQueue<HitWorkItem>    dielectricHitQueue;
	WorkQueue<MissWorkItem>   missQueue;      // escaped rays
	WorkQueue<ShadowRayWorkItem> shadowQueue; // shadow rays pending occlusion test
};

// ============================================================================
// WavefrontLaunchParams — constant data passed to OptiX wavefront programs
// (requires optix_types.h for SphereData/QuadData/MaterialData)
// ============================================================================

#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
struct WavefrontLaunchParams {
	// Queues
	WorkQueue<RayWorkItem>       rayQueue;
	WorkQueue<HitWorkItem>       hitQueue;
	// See WavefrontQueues::simpleHitQueue above - same purpose, mirrored here
	// since this is the struct actually passed to the OptiX launch params.
	WorkQueue<HitWorkItem>       simpleHitQueue;
	// See WavefrontQueues::dielectricHitQueue above - same purpose, mirrored
	// here since this is the struct actually passed to the OptiX launch params.
	WorkQueue<HitWorkItem>       dielectricHitQueue;
	WorkQueue<MissWorkItem>      missQueue;

	// Shadow ray queue is traced separately via a second optixLaunch.
	WorkQueue<ShadowRayWorkItem> shadowQueue;

	// BSSRDF probe walk (MaterialType::Subsurface, wavefront backend Phase
	// 2 - see BssrdfProbeWorkItem/BssrdfExitWorkItem's own comments). Both
	// traced via a THIRD optixLaunch (reusing the same intersect pipeline/
	// module, a dedicated raygen + hit groups - see wavefront_probe.h and
	// WavefrontPathTracer::buildSBT()'s probeSBT_), between the intersect
	// and shadow launches each bounce.
	WorkQueue<BssrdfProbeWorkItem> bssrdfProbeQueue;
	WorkQueue<BssrdfExitWorkItem>  bssrdfExitQueue;

	// Output
	float3*      framebuffer;
	unsigned int width;
	unsigned int height;

	// Scene
	OptixTraversableHandle traversable;

	// Geometry arrays
	SphereData*  spheres;
	unsigned int numSpheres;
	QuadData*    quads;
	unsigned int numQuads;
	BilinearPatchData* bilinearPatches;
	unsigned int numBilinearPatches;
	DiskData*    disks;
	unsigned int numDisks;
	CylinderData* cylinders;
	unsigned int numCylinders;
	TriangleData* triangles;
	unsigned int numTriangles;

	// Materials
	MaterialData*  materials;
	unsigned int   numMaterials;

	// Texture metadata + shared pixel buffer (OptiXRenderer's own
	// d_textures_/d_texturePixels_, already uploaded once at buildScene()
	// time for the recursive path - see WavefrontPathTracer::setTextures()).
	// Needed here (unlike the shading-kernel texture args wf_sample_texture()
	// already takes in wavefront_kernels.cu) so wavefront_programs.cu's
	// OptiX trace-time any-hit programs can sample an alpha-cutout mask
	// (MaterialData::alphaMaskTexIdx) before closest-hit is even decided -
	// see __anyhit__wf_triangle. Null/null (the default) is a valid "no
	// textures in this scene" state, same as the recursive path's
	// LaunchParams::textures/texturePixels.
	TextureData*   textures;
	unsigned char* texturePixels;

	// Heterogeneous cloud media (MaterialType::CloudMedium), same shared
	// device buffer OptiXRenderer::buildScene() already uploads for the
	// recursive path - see wavefront_kernels.cu's CloudMedium case.
	CloudMedium<float>* cloudMediums;
	unsigned int         numCloudMediums;

	// Heterogeneous RGB grid media (MaterialType::RgbGridMedium), same
	// shared device buffers OptiXRenderer::buildScene() already uploads for
	// the recursive path - see wavefront_kernels.cu's RgbGridMedium case.
	GpuRgbGridMedium* rgbGridMediums;
	unsigned int      numRgbGridMediums;
	float*            rgbGridData;
	unsigned int      rgbGridDataCount;

	// Heterogeneous single-channel grid media (MaterialType::GridMedium),
	// same shared device buffers OptiXRenderer::buildScene() already
	// uploads for the recursive path - see wavefront_kernels.cu's
	// GridMedium case.
	GpuGridMedium* gridMediums;
	unsigned int   numGridMediums;
	float*         gridData;
	unsigned int   gridDataCount;

	// Tabulated BSSRDF profile tables (MaterialType::Subsurface), same
	// device buffers OptiXRenderer already uploads once for the recursive
	// path (see optix_types.h's GpuBssrdfTable / LaunchParams::bssrdfTables
	// comments) - read here by wavefront_probe.h's wf_gpu_bssrdf_sr()/
	// wf_gpu_bssrdf_pdf_sr()/wf_gpu_bssrdf_sample_sr() (wavefront-native
	// duplicates of gpu/optix/optix_bssrdf.h's math, reading through
	// `wf_params` instead of the recursive path's `params` - separate OptiX
	// modules can't share a __constant__ launch-params symbol, matching
	// every other wf_-prefixed device-helper duplicate in this codebase).
	GpuBssrdfTable* bssrdfTables;
	unsigned int    numBssrdfTables;
	float* bssrdfRhoSamples;
	float* bssrdfRadiusSamples;
	float* bssrdfProfile;
	float* bssrdfProfileCdf;

	// Light sampling (alias table, same layout as RecursivePathTracer)
	int*         lightIndices;
	const GpuLightKind* lightKinds;   // see optix_types.h - width is load-bearing
	// Per-instance primitive base, same table and same -1 sentinel as
	// LaunchParams::instancePrimBase in optix_types.h. Null when the scene has
	// no object instances, which is every built-in scene.
	const int* instancePrimBase;
	GpuAliasEntry* aliasTable;
	unsigned int   numLights;

	// Rendering params
	unsigned int samplesPerPixel;
	unsigned int maxDepth;
	unsigned int frameNumber;  // seed offset
};
#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
