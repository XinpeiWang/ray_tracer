#pragma once
// pbrt_gpu_builder.h -- turns a loaded .pbrt scene into GPU SceneData.
//
// The GPU counterpart of src/TheRestOfYourLife/pbrt_cpu_builder.h, consuming
// the same pbrt_flatten::FlatScene so the two backends cannot disagree about
// what the file said - only about how they render it.
//
// AREA LIGHTS ARE THE WHOLE DIFFICULTY
// ------------------------------------
// pbrt has no quad shape - a light is a `trianglemesh` with an AreaLightSource
// attached - while the GPU samples an area light as one of the shapes named by
// GpuLightKind. Handed over naively, every pbrt light becomes geometry that
// glows when hit but that next-event estimation cannot aim at. That is not a
// slightly noisier image; it is a black one.
//
// So emissive geometry takes a different route here from everything else.
// pbrt_quadify.h first rejoins triangle PAIRS into parallelograms, which is
// what the overwhelming majority of pbrt area lights are and which the quad
// sampler handles well. What will not merge - an odd triangle, a fan, anything
// genuinely non-parallelogram - is registered as GpuLightKind::Triangle and
// sampled per triangle instead. That second path is newer: those lights used
// to be emitted as glowing geometry and left out of the light list entirely,
// which cost real brightness and noise on GPU with nothing on screen to
// explain it.

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "scene_builder.h"
#include "optix_math_helpers.h"   // cross(), dot(), length() - used below
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_quadify.h"
// ComputeBeamDiffusionBSSRDF/BSSRDFTable - the SAME CPU-side table builder
// src/TheRestOfYourLife/material_pbrt.h's `class subsurface` already uses,
// called here once per unique (g,eta) pair to build the table this builder
// then uploads for the recursive GPU backend's probe walk - see
// getOrBuildBssrdfTable() below and optix_types.h's GpuBssrdfTable comment.
// Not a device computation: this runs host-side at scene-build time, same
// cost class as building a BVH or alias table.
#include "../../src/shared/bssrdf.h"
// GetMeasuredBRDFDataCached()/MeasuredBRDFData/PiecewiseLinear2D<N> - the
// SAME process-wide cache src/TheRestOfYourLife/material_pbrt.h's `class
// measured` already uses (reused here rather than reparsing a multi-
// megabyte .bsdf file a second time), called once per unique resolved
// filename to get the already-built warp tables this builder then flattens
// and uploads for BOTH GPU backends - see getOrBuildMeasuredTable() below
// and optix_types.h's GpuPL2DTable/GpuMeasuredTable comments. CPU-only
// (std::ifstream, std::mutex - see that header's own comment), but this
// file (pbrt_gpu_builder.h) is itself only ever included from host-compiled
// .cpp files (scene_builder.cpp and tests/unit/*.cpp - never a .cu), so
// that is not a problem here.
#include "../../src/shared/measured_bxdf_loader.h"

namespace pbrt_gpu {

struct BuildStats {
	std::size_t triangles = 0;
	std::size_t spheres = 0;
	std::size_t bilinearPatches = 0;
	std::size_t quadLights = 0;
	std::size_t emissiveTrianglesSampledIndividually = 0;
	// Placements the builder prepared; optix_renderer.cpp's buildScene()
	// turns each one into its own IAS entry over a per-definition GAS.
	std::size_t instancePlacements = 0;
	// scene.infiniteLight's flat colour for missed rays, or (0,0,0) if the
	// scene has none - GPU approximation of a real environment light, same
	// shape as the hand-written HDRI scenes' own GPU port (see
	// pbrt_flatten.h's InfiniteLight comment for why full image-based
	// importance sampling stays CPU-only). Currently L*scale only; once the
	// image resolver lands this becomes the decoded image's average colour
	// for the filename case.
	float3 backgroundColor = make_float3(0.0f, 0.0f, 0.0f);
};

namespace detail {

inline float3 f3(const double *v) {
	return make_float3(static_cast<float>(v[0]), static_cast<float>(v[1]),
					   static_cast<float>(v[2]));
}

// Mirrors pbrt_cpu_builder.h's reflectanceToConductorK() - see its comment
// for why (a reflectance-only conductor, eta=1 solved for k via the
// normal-incidence Schlick relation). Kept in sync by hand since one is
// CPU-only (double, color) and the other GPU-only (float, float3); a shared
// header for six lines was judged not worth the indirection.
inline float3 reflectanceToConductorK(const float3 &r) {
	const auto k1 = [](float x) {
		x = x < 0.0f ? 0.0f : (x > 0.9999f ? 0.9999f : x);
		return 2.0f * sqrtf(x) / sqrtf(fmaxf(1e-4f, 1.0f - x));
	};
	return make_float3(k1(r.x), k1(r.y), k1(r.z));
}

// Builds (or looks up, if an earlier material already asked for the same
// (g,eta) pair - e.g. this codebase's own dragon_10/50/250 "scale" scenes,
// which all share eta=1.5/g=0 and differ only in sigma_a/sigma_s) a
// BSSRDFTable via the existing CPU ComputeBeamDiffusionBSSRDF(), uploads its
// arrays into `out`'s flat buffers, and returns the new GpuBssrdfTable's
// index (for MaterialData::textureIdx - see MaterialType::Subsurface's
// comment). n_rho=100/n_radius=64 matches src/TheRestOfYourLife/
// material_pbrt.h's `class subsurface` constructor exactly, so CPU and GPU
// build tables at the same resolution.
inline int getOrBuildBssrdfTable(double g, double eta, SceneData &out,
								 std::map<std::pair<double,double>, int> &cache) {
	const auto key = std::make_pair(g, eta);
	const auto it = cache.find(key);
	if (it != cache.end()) return it->second;

	BSSRDFTable table(100, 64);
	ComputeBeamDiffusionBSSRDF(g, eta, &table);

	GpuBssrdfTable gt{};
	gt.n_rho = table.n_rho;
	gt.n_radius = table.n_radius;
	gt.rho_offset = static_cast<int>(out.bssrdfRhoSamples.size());
	gt.radius_offset = static_cast<int>(out.bssrdfRadiusSamples.size());
	gt.profile_offset = static_cast<int>(out.bssrdfProfile.size());

	for (double v : table.rho_samples) out.bssrdfRhoSamples.push_back(static_cast<float>(v));
	for (double v : table.radius_samples) out.bssrdfRadiusSamples.push_back(static_cast<float>(v));
	for (double v : table.profile) out.bssrdfProfile.push_back(static_cast<float>(v));
	for (double v : table.profile_cdf) out.bssrdfProfileCdf.push_back(static_cast<float>(v));

	const int idx = static_cast<int>(out.bssrdfTables.size());
	out.bssrdfTables.push_back(gt);
	cache.emplace(key, idx);
	return idx;
}

// Flattens one already-built PiecewiseLinear2D<Dimension> instance (its CDF
// construction, if any, already ran CPU-side when `t` was built - see
// piecewise_linear_2d.h's own header comment; this is purely a read-and-copy
// step) into `out`'s shared measuredParamValues/measuredData/measuredMcdf/
// measuredCcdf buffers, and returns the resulting GpuPL2DTable describing
// the slices it just appended. Template on Dimension (0, 2, or 3 - matches
// MeasuredBRDFData's own ndf/sigma vs. vndf/luminance vs. spectra split)
// purely to accept PiecewiseLinear2D<N>'s own compile-time-Dimension type;
// GpuPL2DTable::dim below carries the SAME value at runtime, for the device
// code (which reads a GpuMeasuredTable's five sub-tables generically, not
// through a template - see optix_measured_bxdf.h's own comment on why).
template <size_t Dimension>
inline GpuPL2DTable flattenMeasuredSubTable(const PiecewiseLinear2D<Dimension>& t, SceneData& out) {
	GpuPL2DTable g{};
	g.nx = t.XSize();
	g.ny = t.YSize();
	g.dim = static_cast<int>(Dimension);

	const uint32_t* res = t.ParamRes();
	const uint32_t* stride = t.ParamStride();
	for (size_t i = 0; i < 3; ++i) {
		if (i < Dimension) {
			const std::vector<float>& axis = t.ParamValues(i);
			g.param_res[i] = static_cast<int>(res[i]);
			g.param_stride[i] = static_cast<int>(stride[i]);
			g.param_value_offset[i] = static_cast<int>(out.measuredParamValues.size());
			out.measuredParamValues.insert(out.measuredParamValues.end(), axis.begin(), axis.end());
		} else {
			g.param_res[i] = 1;
			g.param_stride[i] = 0;
			g.param_value_offset[i] = -1;
		}
	}

	g.data_offset = static_cast<int>(out.measuredData.size());
	const std::vector<float>& data = t.Data();
	out.measuredData.insert(out.measuredData.end(), data.begin(), data.end());

	const std::vector<float>& mcdf = t.Mcdf();
	if (!mcdf.empty()) {
		g.mcdf_offset = static_cast<int>(out.measuredMcdf.size());
		out.measuredMcdf.insert(out.measuredMcdf.end(), mcdf.begin(), mcdf.end());
	} else {
		g.mcdf_offset = -1;
	}
	const std::vector<float>& ccdf = t.Ccdf();
	if (!ccdf.empty()) {
		g.ccdf_offset = static_cast<int>(out.measuredCcdf.size());
		out.measuredCcdf.insert(out.measuredCcdf.end(), ccdf.begin(), ccdf.end());
	} else {
		g.ccdf_offset = -1;
	}
	return g;
}

// Builds (or looks up, if an earlier material already asked for the same
// resolved .bsdf path - e.g. the sportscar scene's ilm_l3_37_metallic_spec.
// bsdf, referenced by 4 materials) a GpuMeasuredTable by loading the file
// through the SAME CPU-side cache src/TheRestOfYourLife/material_pbrt.h's
// `class measured` already uses (measured_bxdf_io::GetMeasuredBRDFDataCached
// - reuses the already-parsed MeasuredBRDFData rather than reparsing several
// megabytes of binary a second time), flattening its 5 PiecewiseLinear2D
// sub-tables into `out`'s shared buffers, and returning the new
// GpuMeasuredTable's index (for MaterialData::textureIdx - see
// MaterialType::Measured's comment). Returns -1 if the file failed to load
// (should not normally happen here: pbrt_load.h already load-tested
// resolvedPath during CPU scene loading and would have cleared
// Material::measuredFilename on failure - see that field's own comment -
// but a failed GetMeasuredBRDFDataCached() call is still handled explicitly
// rather than assumed impossible).
inline int getOrBuildMeasuredTable(const std::string& resolvedPath, SceneData& out,
								   std::map<std::string, int>& cache) {
	const auto it = cache.find(resolvedPath);
	if (it != cache.end()) return it->second;

	std::string error;
	std::shared_ptr<const MeasuredBRDFData> data =
		measured_bxdf_io::GetMeasuredBRDFDataCached(resolvedPath, error);
	if (!data) {
		cache.emplace(resolvedPath, -1);
		return -1;
	}

	GpuMeasuredTable mt{};
	mt.isotropic = data->isotropic ? 1 : 0;
	mt.ndf       = flattenMeasuredSubTable<0>(data->ndf, out);
	mt.sigma     = flattenMeasuredSubTable<0>(data->sigma, out);
	mt.vndf      = flattenMeasuredSubTable<2>(data->vndf, out);
	mt.luminance = flattenMeasuredSubTable<2>(data->luminance, out);
	mt.spectra   = flattenMeasuredSubTable<3>(data->spectra, out);

	const int idx = static_cast<int>(out.measuredTables.size());
	out.measuredTables.push_back(mt);
	cache.emplace(resolvedPath, idx);
	return idx;
}

// Mirrors pbrt_cpu_builder.h's makeMaterial() decision for decision, including
// emission winning over the declared material - in pbrt an AreaLightSource
// attaches to the shape, and the surface is an emitter regardless of what else
// it said it was. The two builders disagreeing here would mean the same file
// renders as two different scenes depending on the backend.
//
// `out`/`bssrdfTableCache` are only touched by the Subsurface case below
// (building/deduping this material's BSSRDFTable); every other material kind
// ignores them.
inline MaterialData makeMaterial(const pbrt_flatten::Material &m,
								 const pbrt_flatten::Emission *emission,
								 SceneData &out,
								 std::map<std::pair<double,double>, int> &bssrdfTableCache,
								 std::map<std::string, int> &measuredTableCache) {
	MaterialData d = {};
	d.textureIdx = -1;

	if (emission) {
		d.type = MaterialType::DiffuseLight;
		d.emission = make_float3(
			static_cast<float>(emission->L[0] * emission->scale),
			static_cast<float>(emission->L[1] * emission->scale),
			static_cast<float>(emission->L[2] * emission->scale));
		return d;
	}

	d.albedo = make_float3(static_cast<float>(m.color[0]),
						   static_cast<float>(m.color[1]),
						   static_cast<float>(m.color[2]));
	d.roughness = static_cast<float>(m.roughness);
	d.ior = static_cast<float>(m.ior);

	switch (m.kind) {
	case pbrt_flatten::MaterialKind::Conductor:
		// Metal rather than MaterialType::Conductor on purpose: Conductor is
		// described by a complex IOR (eta_c/k_c) that a pbrt scene only
		// supplies as named spectra we do not parse. Metal takes the albedo
		// and roughness we actually have, and matches what the CPU builder
		// does with the same material.
		d.type = MaterialType::Metal;
		break;
	case pbrt_flatten::MaterialKind::Dielectric:
		d.type = MaterialType::Dielectric;
		// d.albedo was just set to m.color above (same union slot as
		// Dielectric's own transmission_filter - see optix_types.h's
		// comment) for every material kind generically; a pbrt dielectric's
		// "color" isn't the OBJ/.mtl "Tf" tint feature that field means for
		// Dielectric specifically, so reset it to the neutral/no-op value
        // here rather than accidentally tinting every pbrt-loaded glass
        // material by whatever m.color happened to default to.
		d.transmission_filter = make_float3(1.0f, 1.0f, 1.0f);
		break;
	case pbrt_flatten::MaterialKind::ThinDielectric:
		d.type = MaterialType::ThinDielectric;
		break;
	case pbrt_flatten::MaterialKind::CoatedDiffuse:
		d.type = MaterialType::CoatedDiffuse;
		break;
	case pbrt_flatten::MaterialKind::DiffuseTransmission:
		d.type = MaterialType::DiffuseTransmission;
		// Was d.albedo (the reflectance channel, already assigned above) -
		// silently making transmittance identical to reflectance regardless
		// of what the scene's own "transmittance" parameter said, before
		// pbrt_flatten.h had anywhere to keep that value separately.
		d.transmittance = make_float3(static_cast<float>(m.transmittance[0]),
									  static_cast<float>(m.transmittance[1]),
									  static_cast<float>(m.transmittance[2]));
		break;
	case pbrt_flatten::MaterialKind::CoatedConductor:
		// Same reflectance-only approximation pbrt_cpu_builder.h uses (see
		// its reflectanceToConductorK() comment) - eta=1, k solved from the
		// albedo already read above as a normal-incidence reflectance.
		d.type = MaterialType::CoatedConductor;
		d.eta_c = make_float3(1.0f, 1.0f, 1.0f);
		d.k_c = reflectanceToConductorK(d.albedo);
		break;
	case pbrt_flatten::MaterialKind::Subsurface:
		// Real tabulated BSSRDF, RECURSIVE GPU BACKEND ONLY (Phase 1 - see
		// optix_types.h's MaterialType::Subsurface comment for the full
		// field-reuse layout, and shade_material()'s own comment,
		// optix_device_helpers.h, for the probe-walk algorithm). `d.albedo`
		// is deliberately left as `m.color` (already assigned above,
		// generically, before this switch) rather than repurposed for
		// sigma_a - that is what lets the wavefront backend's own explicit
		// Subsurface fallback (wavefront_kernels.cu) keep rendering exactly
		// the same flat gray it always has, unaware this real GPU
		// implementation exists at all.
		d.type = MaterialType::Subsurface;
		d.bssrdf_sigma_a = make_float3(static_cast<float>(m.sigma_a[0]),
										static_cast<float>(m.sigma_a[1]),
										static_cast<float>(m.sigma_a[2]));
		d.bssrdf_sigma_s = make_float3(static_cast<float>(m.sigma_s[0]),
										static_cast<float>(m.sigma_s[1]),
										static_cast<float>(m.sigma_s[2]));
		// d.ior already holds eta (m.ior), assigned generically above -
		// same field NormalizedFresnel(ior=eta) reads at the found exit
		// point (shade_material()'s probe-walk success path).
		d.textureIdx = getOrBuildBssrdfTable(m.g, m.ior, out, bssrdfTableCache);
		break;
	case pbrt_flatten::MaterialKind::Diffuse:
	case pbrt_flatten::MaterialKind::Unsupported:
		d.type = MaterialType::Lambertian;
		break;
	// Real tabulated measured-BRDF support, BOTH GPU backends now (see
	// MaterialType::Measured's own comment in optix_types.h and
	// optix_measured_bxdf.h/wavefront_measured_bxdf.h for the device math).
	// m.measuredFilename is already a resolved, load-tested absolute path by
	// this point (pbrt_load.h ran during CPU scene loading, before this
	// builder ever sees the FlatScene - see that field's own comment) or
	// empty if resolution/loading failed - either way, falling back to
	// Lambertian here matches pbrt_cpu_builder.h's own `class measured`
	// fallback (`loaded()` false) exactly, so the two backends cannot
	// disagree about which materials got real data and which fell back.
	case pbrt_flatten::MaterialKind::Measured: {
		const int tableIdx = m.measuredFilename.empty()
			? -1
			: getOrBuildMeasuredTable(m.measuredFilename, out, measuredTableCache);
		if (tableIdx >= 0) {
			d.type = MaterialType::Measured;
			d.textureIdx = tableIdx;
		} else {
			d.type = MaterialType::Lambertian;
		}
		break;
	}
	}
	return d;
}

} // namespace detail

// Fills `out` with the scene's geometry, materials and light list. Returns the
// counts; `out` is cleared first.
inline BuildStats build(const pbrt_flatten::FlatScene &scene, SceneData &out) {
	using namespace detail;
	BuildStats stats;

	out.spheres.clear();
	out.quads.clear();
	out.triangles.clear();
	out.bilinearPatches.clear();
	out.materials.clear();
	out.lightIndices.clear();
	out.lightKinds.clear();

	// One MaterialData per distinct (material, emission) pair, exactly as the
	// CPU builder caches them - a mesh with a thousand faces sharing one
	// material must not produce a thousand identical GPU materials.
	std::map<std::pair<int, int>, int> cache;
	// Separate dedup cache for BSSRDFTable-by-(g,eta) - see
	// getOrBuildBssrdfTable()'s own comment. Only Subsurface materials touch
	// this; stays empty for every scene without one.
	std::map<std::pair<double, double>, int> bssrdfTableCache;
	// Separate dedup cache for GpuMeasuredTable-by-resolved-.bsdf-path - see
	// getOrBuildMeasuredTable()'s own comment. Only Measured materials touch
	// this; stays empty for every scene without one.
	std::map<std::string, int> measuredTableCache;
	const auto materialIndex = [&](int mi, int ai) {
		const auto key = std::make_pair(mi, ai);
		const auto it = cache.find(key);
		if (it != cache.end()) return it->second;

		const pbrt_flatten::Emission *em =
			(ai >= 0 && static_cast<std::size_t>(ai) < scene.areaLights.size())
				? &scene.areaLights[static_cast<std::size_t>(ai)]
				: nullptr;
		static const pbrt_flatten::Material kDefault{};
		const pbrt_flatten::Material &m =
			(mi >= 0 && static_cast<std::size_t>(mi) < scene.materials.size())
				? scene.materials[static_cast<std::size_t>(mi)]
				: kDefault;

		const int idx = static_cast<int>(out.materials.size());
		out.materials.push_back(makeMaterial(m, em, out, bssrdfTableCache, measuredTableCache));
		cache.emplace(key, idx);
		return idx;
	};

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : scene.spheres) {
		SphereData sd = {};
		sd.center = f3(s.center);
		sd.center1 = sd.center;          // static; see SphereData's comment
		sd.radius = static_cast<float>(s.radius);
		sd.materialIdx = materialIndex(s.material, s.areaLight);
		if (s.areaLight >= 0) {
			out.lightIndices.push_back(static_cast<int>(out.spheres.size()));
			out.lightKinds.push_back(GpuLightKind::Sphere);
		}
		out.spheres.push_back(sd);
	}
	stats.spheres = out.spheres.size();

	// ---- bilinear patches -------------------------------------------------
	// Shape "bilinearmesh" - unlike triangles, never routed through
	// pbrt_quadify.h: a bilinear patch is not necessarily planar, so folding
	// two of them into one parallelogram the way triangle pairs are would be
	// wrong in general (see pbrt_flatten.h's BilinearPatch comment).
	for (const pbrt_flatten::BilinearPatch &p : scene.bilinearPatches) {
		BilinearPatchData bd = {};
		bd.p00 = f3(p.p[0]);
		bd.p10 = f3(p.p[1]);
		bd.p01 = f3(p.p[2]);
		bd.p11 = f3(p.p[3]);
		bd.materialIdx = materialIndex(p.material, p.areaLight);
		if (p.areaLight >= 0) {
			out.lightIndices.push_back(static_cast<int>(out.bilinearPatches.size()));
			out.lightKinds.push_back(GpuLightKind::BilinearPatch);
		}
		out.bilinearPatches.push_back(bd);
	}
	stats.bilinearPatches = out.bilinearPatches.size();

	// ---- lights recovered as quads, then everything else as triangles ----
	const pbrt_quadify::Result merged = pbrt_quadify::quadify(scene.triangles);

	for (const pbrt_quadify::Quad &q : merged.quads) {
		QuadData qd = {};
		qd.Q = f3(q.Q);
		qd.u = f3(q.u);
		qd.v = f3(q.v);
		// w = u x v DIRECTLY, matching every other GPU quad builder in
		// scene_builder.cpp (grep quad.w = quad_cross / lc there) - NOT the
		// n/dot(n,n) reciprocal that the CPU-side RTIOW quad.h barycentric
		// trick uses, which is a different convention for a different purpose.
		//
		// gpu/optix/optix_device_helpers.h's sample_quad_light() reads
		// `area = length(quad.w)` on the documented assumption "w = u x v, so
		// |w| = area". Handing it n/dot(n,n) instead gives |w| = 1/area, which
		// silently inverts that assumption: the light's NEE pdf comes out
		// scaled by area^2 (~1.86e8 for this scene's ~13650-unit light quad),
		// and dividing radiance by a pdf that far too large is indistinguishable
		// from no light at all once quantized to 8 bits. Confirmed by dumping
		// the raw pre-tonemap framebuffer: values were finite, positive, and
		// real (not NaN, not exactly zero) but capped at ~4.7e-7 - light WAS
		// reaching every surface, just at a hundred-millionth of its true
		// magnitude, which 8-bit output cannot represent as anything but black.
		const float3 n = cross(qd.u, qd.v);
		qd.w = n;
		const float len = sqrtf(dot(n, n));
		qd.normal = make_float3(n.x / len, n.y / len, n.z / len);
		qd.D = dot(qd.normal, qd.Q);
		qd.materialIdx = materialIndex(q.material, q.areaLight);
		if (q.areaLight >= 0) {
			out.lightIndices.push_back(static_cast<int>(out.quads.size()));
			out.lightKinds.push_back(GpuLightKind::Quad);
		}
		out.quads.push_back(qd);
	}
	stats.quadLights = out.quads.size();

	for (const pbrt_flatten::Triangle &t : merged.leftover) {
		TriangleData td = {};
		td.p0 = f3(&t.v[0]);
		td.p1 = f3(&t.v[3]);
		td.p2 = f3(&t.v[6]);
		if (t.hasNormals) {
			td.n0 = f3(&t.n[0]);
			td.n1 = f3(&t.n[3]);
			td.n2 = f3(&t.n[6]);
		}
		td.hasNormals = t.hasNormals;
		td.hasUVs = false;               // flatten does not carry UVs yet
		td.materialIdx = materialIndex(t.material, t.areaLight);
		// An emissive triangle that would not fold into a parallelogram is
		// registered as a light in its own right. It used to be emitted as
		// geometry and nothing else - it glowed when a ray happened to hit it,
		// but next-event estimation could not aim at it, so the GPU image came
		// out darker and noisier than the CPU one with nothing on screen to
		// explain why. GpuLightKind::Triangle and sample_triangle_light() are
		// the two halves of the fix.
		if (t.areaLight >= 0) {
			out.lightIndices.push_back(static_cast<int>(out.triangles.size()));
			out.lightKinds.push_back(GpuLightKind::Triangle);
		}
		out.triangles.push_back(td);
	}
	stats.triangles = out.triangles.size();
	// Kept as a stat because it still says something real - how many lights
	// needed the per-triangle path rather than the cheaper merged-quad one -
	// but it no longer means "these will not be sampled". The caller's warning
	// went away with the gap it described.
	stats.emissiveTrianglesSampledIndividually = pbrt_quadify::unmergedEmissiveCount(merged);

	// ---- instanced geometry ----------------------------------------------
	// Object space, kept apart from the world-space list above. Emissive
	// shapes are not here: flatten() already baked those per placement into
	// scene.triangles, because a light must be enumerable to be sampled.
	out.instanceGroups.clear();
	out.instancePlacements.clear();
	out.instanceTriangles.clear();
	out.instanceSpheres.clear();

	std::vector<int> groupIndexMap(scene.groups.size(), -1);
	for (std::size_t g = 0; g < scene.groups.size(); ++g) {
		const pbrt_flatten::InstanceGroup &grp = scene.groups[g];
		if (grp.triangles.empty() && grp.spheres.empty()) continue;

		SceneData::InstanceGroupGPU gpuGroup;
		gpuGroup.triangleBase = static_cast<int>(out.instanceTriangles.size());
		for (const pbrt_flatten::Triangle &t : grp.triangles) {
			TriangleData td = {};
			td.p0 = f3(&t.v[0]);
			td.p1 = f3(&t.v[3]);
			td.p2 = f3(&t.v[6]);
			if (t.hasNormals) {
				td.n0 = f3(&t.n[0]);
				td.n1 = f3(&t.n[3]);
				td.n2 = f3(&t.n[6]);
			}
			td.hasNormals = t.hasNormals;
			td.hasUVs = false;
			td.materialIdx = materialIndex(t.material, t.areaLight);
			out.instanceTriangles.push_back(td);
		}
		gpuGroup.triangleCount =
			static_cast<int>(out.instanceTriangles.size()) - gpuGroup.triangleBase;

		// Spheres are custom AABB primitives, so they cannot share the GAS the
		// triangles above get; the renderer gives this group a second one. They
		// stay in the definition's object space like the triangles, which is
		// what lets a placement with a non-uniform scale render as the ellipsoid
		// the scene asked for - a baked world-space sphere could only ever be
		// round. Never emissive: flatten() bakes those per placement instead,
		// because a light has to be enumerable to be sampled.
		gpuGroup.sphereBase = static_cast<int>(out.instanceSpheres.size());
		for (const pbrt_flatten::Sphere &s : grp.spheres) {
			SphereData sd = {};
			sd.center = f3(s.center);
			sd.center1 = sd.center;          // static; see SphereData's comment
			sd.radius = static_cast<float>(s.radius);
			sd.materialIdx = materialIndex(s.material, s.areaLight);
			out.instanceSpheres.push_back(sd);
		}
		gpuGroup.sphereCount =
			static_cast<int>(out.instanceSpheres.size()) - gpuGroup.sphereBase;

		groupIndexMap[g] = static_cast<int>(out.instanceGroups.size());
		out.instanceGroups.push_back(gpuGroup);
	}

	for (const pbrt_flatten::Instance &inst : scene.instances) {
		if (inst.group < 0 ||
			static_cast<std::size_t>(inst.group) >= groupIndexMap.size()) continue;
		const int mapped = groupIndexMap[static_cast<std::size_t>(inst.group)];
		if (mapped < 0) continue;

		SceneData::InstancePlacementGPU p;
		p.group = mapped;
		// FlatScene stores a row-major 4x4; OptiX wants the top three rows as
		// a 3x4, which is the same memory order with the last row dropped.
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 4; ++col)
				p.transform[row * 4 + col] =
					static_cast<float>(inst.xform[row * 4 + col]);
		out.instancePlacements.push_back(p);
	}
	stats.instancePlacements = out.instancePlacements.size();

	// ---- infinite/sky light, flat-colour GPU approximation ----------------
	if (scene.infiniteLight.present) {
		const auto &sky = scene.infiniteLight;
		if (sky.imageWidth > 0 && sky.imageHeight > 0 && !sky.imagePixels.empty()) {
			// Mean of every decoded pixel - a crude approximation of the real
			// environment map's overall brightness/tint (no directional
			// detail at all, unlike the CPU path's real importance-sampled
			// image - see pbrt_flatten.h's InfiniteLight comment for why
			// that stays CPU-only), but far closer than treating an
			// environment-lit scene as a flat colour the scene never named.
			double r = 0.0, g = 0.0, b = 0.0;
			const std::size_t n = static_cast<std::size_t>(sky.imageWidth) * sky.imageHeight;
			for (std::size_t i = 0; i < n; ++i) {
				r += sky.imagePixels[i * 3 + 0];
				g += sky.imagePixels[i * 3 + 1];
				b += sky.imagePixels[i * 3 + 2];
			}
			stats.backgroundColor = make_float3(
				static_cast<float>(r / n * sky.scale),
				static_cast<float>(g / n * sky.scale),
				static_cast<float>(b / n * sky.scale));
		} else {
			stats.backgroundColor = make_float3(
				static_cast<float>(sky.L[0] * sky.scale),
				static_cast<float>(sky.L[1] * sky.scale),
				static_cast<float>(sky.L[2] * sky.scale));
		}
	}

	return stats;
}

} // namespace pbrt_gpu
