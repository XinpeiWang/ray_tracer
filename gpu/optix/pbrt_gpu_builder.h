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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "scene_builder.h"
#include "optix_math_helpers.h"   // cross(), dot(), length() - used below
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_quadify.h"
#include "../../src/shared/curve_tessellate.h"
// pbrt_scene::Matrix4::inverseAffine() - used below (disks/cylinders loop) to
// precompute each primitive's w2o from its flattened o2w, host-side, once,
// the same "invert once at scene-build time, never on device" split
// src/TheRestOfYourLife/disk_cylinder_hittable.h's CPU backend also uses.
#include "../../src/shared/pbrt_scene.h"
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
// PiecewiseConstant1D/PiecewiseConstant2D - the SAME distribution
// src/TheRestOfYourLife/sky_light.h builds CPU-side for its real
// importance-sampled HDR environment light. Built here (once per scene, from
// the scene's already-decoded infinite-light image) purely to flatten and
// upload - see the "infinite/sky light" block at the end of build() below.
#include "../../src/shared/piecewise_dist.h"
// stbi_loadf - loading a Diffuse material's imagemap-bound reflectance
// texture (Material::textureFilename) into scene.texturePixels, the exact
// same shared flat pixel buffer + TextureData table scene_builder.cpp's own
// load_image_texture_gpu() (OBJ/MTL map_Kd's GPU path) already uses. That
// function itself is not reusable here: it lives in scene_builder.cpp's file-
// local anonymous namespace, defined AFTER this header is #include'd there,
// so getOrBuildPbrtImageTexture() below duplicates its handful of lines
// rather than depending on link order. Declarations only, like
// scene_builder.cpp's own include - the implementation (STB_IMAGE_
// IMPLEMENTATION) is linked once from src/external/stb_image_impl.cpp.
#include "../../src/external/stb_image.h"

namespace pbrt_gpu {

struct BuildStats {
	std::size_t triangles = 0;
	std::size_t spheres = 0;
	std::size_t bilinearPatches = 0;
	// Shape "disk"/"cylinder" - supported on both the recursive (Phase 4b)
	// and wavefront (Phase 4c) backends; see optix_types.h's DiskData/
	// CylinderData comment.
	std::size_t disks = 0;
	std::size_t cylinders = 0;
	// Shape "curve" - each strand's segments are diced into bilinear-patch
	// quads (see pbrt_gpu_builder.h's own curve loop comment), so this counts
	// source curve segments, not the resulting patch count (already folded
	// into `bilinearPatches` above).
	std::size_t curveSegments = 0;
	std::size_t quadLights = 0;
	std::size_t emissiveTrianglesSampledIndividually = 0;
	// Placements the builder prepared; optix_renderer.cpp's buildScene()
	// turns each one into its own IAS entry over a per-definition GAS.
	std::size_t instancePlacements = 0;
	// scene.infiniteLight's mean colour (L*scale for a constant-colour sky,
	// or the decoded image's unweighted average colour*scale for an image
	// sky), or (0,0,0) if the scene has none. Still used as the "does this
	// scene have a sky at all" gate every NEE/miss call site checks, and as
	// the flat-colour + uniform-sphere fallback for a constant-colour sky -
	// but for an image sky, the REAL per-direction importance-sampled
	// machinery (SceneData::skyImagePixels/skyMarginalCdf/etc., built and
	// flattened just below this comment's own call site - see build()'s
	// "infinite/sky light" block) now does the actual per-direction Le()/
	// pdf_Li() work on both GPU backends (gpu/optix/optix_sky_light.h /
	// wavefront_sky_light.h), mirroring src/TheRestOfYourLife/sky_light.h
	// exactly instead of approximating it with a flat wash.
	float3 backgroundColor = make_float3(0.0f, 0.0f, 0.0f);
};

namespace detail {

inline float3 f3(const double *v) {
	return make_float3(static_cast<float>(v[0]), static_cast<float>(v[1]),
					   static_cast<float>(v[2]));
}

inline float2 f2(const double *v) {
	return make_float2(static_cast<float>(v[0]), static_cast<float>(v[1]));
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

// Loads a Diffuse material's imagemap-bound reflectance texture (Material::
// textureFilename - already a resolved, existence-tested absolute path by
// this point, pbrt_load.h having run during CPU scene loading before this
// builder ever sees the FlatScene, same convention as measuredFilename
// above) into `out`'s shared texture table, returning its MaterialData::
// textureIdx. Mirrors scene_builder.cpp's load_image_texture_gpu() exactly
// (same search behaviour, same float-to-byte conversion), plus a filename
// cache so a texture shared by many materials is decoded once. Returns -1
// only if stb_image itself fails to decode an existence-tested file (a
// corrupt-but-present image) - the caller falls back to Lambertian's flat
// `albedo` colour, matching pbrt_cpu_builder.h's own mipmap_texture(mip_==
// nullptr) degradation for the same case.
inline int getOrBuildPbrtImageTexture(const std::string& resolvedPath, SceneData& out,
									  std::map<std::string, int>& cache) {
	const auto it = cache.find(resolvedPath);
	if (it != cache.end()) return it->second;

	int width = 0, height = 0, channels = 0;
	float* fdata = stbi_loadf(resolvedPath.c_str(), &width, &height, &channels, 3);
	if (!fdata) {
		cache.emplace(resolvedPath, -1);
		return -1;
	}

	TextureData tex{};
	tex.kind = TextureKind::Image;
	tex.noiseScale = 0.0f;
	tex.pixelOffset = static_cast<int>(out.texturePixels.size());
	tex.width = width;
	tex.height = height;
	const std::size_t total = static_cast<std::size_t>(width) * height * 3;
	out.texturePixels.resize(out.texturePixels.size() + total);
	unsigned char* dst = out.texturePixels.data() + tex.pixelOffset;
	for (std::size_t i = 0; i < total; ++i) {
		const float v = fdata[i];
		dst[i] = (v <= 0.0f) ? 0 : (v >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * v));
	}
	stbi_image_free(fdata);

	const int idx = static_cast<int>(out.textures.size());
	out.textures.push_back(tex);
	cache.emplace(resolvedPath, idx);
	return idx;
}

// A representative average color for an already-built Image texture (see
// getOrBuildPbrtImageTexture() above) - used ONLY as a light-selection
// power estimate for a textured DiffuseLight (see that emission branch's
// own comment in makeMaterial()), never for actual per-sample radiance
// (which always goes through the real per-pixel lookup at render time).
// Strided rather than a full per-pixel walk - a few thousand samples is
// plenty for an importance-sampling weight, and this runs once per unique
// texture at scene build, not per frame, but a multi-megapixel image
// (a real risk for pbrt scenes) shouldn't become an O(width*height) scene-
// build-time cost for a value this approximate.
inline float3 averageTextureColor(int textureIdx, const SceneData& out) {
	if (textureIdx < 0 || textureIdx >= static_cast<int>(out.textures.size()))
		return make_float3(0.5f, 0.5f, 0.5f);
	const TextureData& tex = out.textures[textureIdx];
	const std::size_t pixelCount = static_cast<std::size_t>(tex.width) * tex.height;
	if (pixelCount == 0) return make_float3(0.5f, 0.5f, 0.5f);

	constexpr std::size_t kMaxSamples = 4096;
	const std::size_t stride = (pixelCount > kMaxSamples) ? (pixelCount / kMaxSamples) : 1;
	const unsigned char* base = out.texturePixels.data() + tex.pixelOffset;
	double sumR = 0.0, sumG = 0.0, sumB = 0.0;
	std::size_t sampled = 0;
	for (std::size_t p = 0; p < pixelCount; p += stride) {
		const unsigned char* px = base + p * 3;
		sumR += px[0]; sumG += px[1]; sumB += px[2];
		++sampled;
	}
	const double norm = 1.0 / (255.0 * static_cast<double>(sampled ? sampled : 1));
	return make_float3(static_cast<float>(sumR * norm),
						static_cast<float>(sumG * norm),
						static_cast<float>(sumB * norm));
}

// Loads a Shape "alpha" cutout mask (Material::alphaTextureFilename) into
// `out`'s shared texture table. A SEPARATE function/cache from
// getOrBuildPbrtImageTexture() above rather than a reused call, because that
// one goes through stbi_loadf() - which, for an ordinary 8-bit/LDR source
// image, silently applies stb_image's default gamma-2.2 decode (its
// "LDR-to-HDR" conversion, meant for genuine colour data going from sRGB to
// linear). An alpha/opacity mask is a linear coverage fraction, not a
// display colour, so that decode would systematically bias every cutout
// threshold comparison (e.g. an authored 0.6 alpha, byte 153/255, would
// decode to pow(0.6, 2.2) =~ 0.32 and silently flip which side of the 0.5
// cutout threshold - triangle.h's kAlphaCutoutThreshold - it falls on).
// stbi_load() (the plain 8-bit loader, no float conversion, no gamma of any
// kind) sidesteps this entirely; the byte/255 divide below is the exact
// linear reconstruction pbrt's own alpha-cutout convention expects.
inline int getOrBuildPbrtAlphaMaskTexture(const std::string& resolvedPath, SceneData& out,
										   std::map<std::string, int>& cache) {
	const auto it = cache.find(resolvedPath);
	if (it != cache.end()) return it->second;

	int width = 0, height = 0, channels = 0;
	unsigned char* bdata = stbi_load(resolvedPath.c_str(), &width, &height, &channels, 3);
	if (!bdata) {
		cache.emplace(resolvedPath, -1);
		return -1;
	}

	TextureData tex{};
	tex.kind = TextureKind::Image;
	tex.noiseScale = 0.0f;
	tex.pixelOffset = static_cast<int>(out.texturePixels.size());
	tex.width = width;
	tex.height = height;
	const std::size_t total = static_cast<std::size_t>(width) * height * 3;
	out.texturePixels.resize(out.texturePixels.size() + total);
	std::memcpy(out.texturePixels.data() + tex.pixelOffset, bdata, total);
	stbi_image_free(bdata);

	const int idx = static_cast<int>(out.textures.size());
	out.textures.push_back(tex);
	cache.emplace(resolvedPath, idx);
	return idx;
}

// Duplicates scene_builder.cpp's own is_grayscale_texture_gpu() exactly
// (same 8x8-grid pixel-content sample, same threshold) for the same reason
// getOrBuildPbrtImageTexture() above duplicates load_image_texture_gpu(): a
// link-order constraint, not a design choice - that function lives in
// scene_builder.cpp's own file-local anonymous namespace, defined AFTER
// this header is #include'd there.
inline bool isPbrtTextureGrayscale(const SceneData& scene, int texIdx) {
	if (texIdx < 0 || texIdx >= static_cast<int>(scene.textures.size())) return true;
	const TextureData& tex = scene.textures[texIdx];
	if (tex.width <= 0 || tex.height <= 0) return true;
	constexpr int kGrid = 8;
	int max_diff = 0;
	for (int sy = 0; sy < kGrid; ++sy) {
		int y = (sy * tex.height) / kGrid;
		for (int sx = 0; sx < kGrid; ++sx) {
			int x = (sx * tex.width) / kGrid;
			std::size_t idx = static_cast<std::size_t>(tex.pixelOffset) + (static_cast<std::size_t>(y) * tex.width + x) * 3;
			int r = scene.texturePixels[idx], g = scene.texturePixels[idx + 1], b = scene.texturePixels[idx + 2];
			max_diff = std::max({max_diff, std::abs(r - g), std::abs(g - b), std::abs(r - b)});
		}
	}
	return max_diff <= 10;
}

// Resolves a material's own effective flat colour for use as one side of a
// Mix blend: `m.color` directly for every ordinary material kind, or - when
// `m` is ITSELF a nested Mix (pbrt-v4 allows mix-of-mix) - the recursive
// weighted average of ITS own two sub-materials' resolved colours, instead
// of `m.color`, which a Mix-kind pbrt_flatten::Material never populates
// (stays at the struct's generic {0.5,0.5,0.5} default - see that struct's
// own comment). Reading `m.color` directly for a nested Mix would silently
// blend in that placeholder grey instead of the nested mix's real colour, a
// visible, silent CPU/GPU divergence for any scene nesting mix materials
// (CPU's own makeMaterial() already recurses through real material objects
// for exactly this case). Same depth cap and reasoning as CPU's
// kMaxMixDepth - guards a cyclic/self-referential "materials" list a
// malformed scene could produce, not a case this loader's own corpus has
// ever needed.
inline float3 resolveMixColor(const pbrt_flatten::Material &m,
							  const std::vector<pbrt_flatten::Material> &allMaterials,
							  int depth) {
	constexpr int kMaxMixDepth = 8;
	if (m.kind != pbrt_flatten::MaterialKind::Mix || depth >= kMaxMixDepth)
		return make_float3(static_cast<float>(m.color[0]),
						   static_cast<float>(m.color[1]),
						   static_cast<float>(m.color[2]));
	if (m.mixMaterialA < 0 || static_cast<std::size_t>(m.mixMaterialA) >= allMaterials.size()
		|| m.mixMaterialB < 0 || static_cast<std::size_t>(m.mixMaterialB) >= allMaterials.size())
		return make_float3(static_cast<float>(m.color[0]),
						   static_cast<float>(m.color[1]),
						   static_cast<float>(m.color[2]));
	const float3 ca = resolveMixColor(allMaterials[static_cast<std::size_t>(m.mixMaterialA)], allMaterials, depth + 1);
	const float3 cb = resolveMixColor(allMaterials[static_cast<std::size_t>(m.mixMaterialB)], allMaterials, depth + 1);
	const float w = static_cast<float>(m.mixWeight);
	return make_float3(ca.x * (1.0f - w) + cb.x * w,
					   ca.y * (1.0f - w) + cb.y * w,
					   ca.z * (1.0f - w) + cb.z * w);
}

// Mirrors pbrt_cpu_builder.h's makeMaterial() decision for decision, including
// emission winning over the declared material - in pbrt an AreaLightSource
// attaches to the shape, and the surface is an emitter regardless of what else
// it said it was. The two builders disagreeing here would mean the same file
// renders as two different scenes depending on the backend.
//
// `out`/`bssrdfTableCache` are only touched by the Subsurface case below
// (building/deduping this material's BSSRDFTable); `allMaterials`/
// `resolveMaterialIndex`/`mixDepth` only by Mix (recursively resolving its
// two sub-materials to real out.materials indices - see that case's own
// comment); every other material kind ignores all four.
inline MaterialData makeMaterial(const pbrt_flatten::Material &m,
								 const pbrt_flatten::Emission *emission,
								 SceneData &out,
								 std::map<std::pair<double,double>, int> &bssrdfTableCache,
								 std::map<std::string, int> &measuredTableCache,
								 std::map<std::string, int> &imageTextureCache,
								 std::map<std::string, int> &alphaMaskTextureCache,
								 const std::vector<pbrt_flatten::Material> &allMaterials = {},
								 const std::function<int(int,int,int)> &resolveMaterialIndex = {},
								 int mixDepth = 0) {
	MaterialData d = {};
	d.textureIdx = -1;

	// Shape "alpha" cutout mask (Material::alphaTextureFilename - see that
	// field's own comment). Computed before the emission early-return below
	// since it's orthogonal to material kind/emission - CPU's own
	// alphaMaskFor() (pbrt_cpu_builder.h) likewise applies regardless of
	// whether the owning triangle turns out emissive, so a leaf-shaped area
	// light stays consistent between backends. A SEPARATE decode/cache
	// (getOrBuildPbrtAlphaMaskTexture(), its own alphaMaskTextureCache/
	// out.textures entries - not imageTextureCache below) because alpha
	// masks must NOT get the gamma decode reflectance imagemaps need - see
	// that function's own comment.
	if (!m.alphaTextureFilename.empty())
		d.alphaMaskTexIdx = getOrBuildPbrtAlphaMaskTexture(m.alphaTextureFilename, out, alphaMaskTextureCache);

	if (emission) {
		d.type = MaterialType::DiffuseLight;
		d.twoSided = emission->twoSided;
		// A "filename" area light wins over "L" entirely (matches CPU's
		// pbrt_cpu_builder.h and pbrt-v4's own DiffuseAreaLight) - see
		// Emission::filename's own comment. "scale" is applied at
		// emission-lookup time via emissionScale instead of baked in here,
		// since the image data itself is shared/cached across materials.
		if (!emission->filename.empty()) {
			d.textureIdx = getOrBuildPbrtImageTexture(emission->filename, out, imageTextureCache);
			d.emissionScale = static_cast<float>(emission->scale);
			// d.emission itself is never READ for a textured light's actual
			// per-sample radiance (material_emission()/sample_area_light_by_
			// kind()'s NEE path both go through textureIdx/emissionScale
			// instead - see each one's own comment), but optix_renderer.cpp's
			// power-weighted light alias table reads raw MaterialData::
			// emission unconditionally to estimate every light's selection
			// weight, with no textureIdx awareness of its own - leaving this
			// at its zero default would make a textured light's true (often
			// substantial) contribution to NEE severely under-sampled,
			// without changing its final per-sample radiance at all (a
			// pure-variance bug, confirmed by a much darker room than an
			// equally-bright flat-color light in the same scene).
			d.emission = averageTextureColor(d.textureIdx, out) * d.emissionScale;
		}
		if (d.textureIdx < 0) {
			// Either no filename was given, or the image failed to decode -
			// same flat-L fallback pbrt_cpu_builder.h's mipmap_texture(mip_==
			// nullptr) degradation uses for the same case.
			d.emission = make_float3(
				static_cast<float>(emission->L[0] * emission->scale),
				static_cast<float>(emission->L[1] * emission->scale),
				static_cast<float>(emission->L[2] * emission->scale));
		}
		return d;
	}

	d.albedo = make_float3(static_cast<float>(m.color[0]),
						   static_cast<float>(m.color[1]),
						   static_cast<float>(m.color[2]));
	d.roughness = static_cast<float>(m.roughness);
	d.remapRoughness = m.remapRoughness;
	d.ior = static_cast<float>(m.ior);

	switch (m.kind) {
	case pbrt_flatten::MaterialKind::Conductor:
		// A recognized named conductor spectrum ("metal-Ag-eta"/"metal-Ag-k"
		// etc. - see pbrt_flatten.h's conductorElementFromSpectrumName())
		// gets the real GGX + complex-Fresnel MaterialType::Conductor,
		// matching pbrt_cpu_builder.h's identical branch and this codebase's
		// native B5/B7 scenes. Anything else (explicit RGB k, or an
		// unrecognized/non-metal named spectrum) keeps the pre-existing
		// Metal (fuzz-mirror) approximation - a pbrt scene only supplies a
		// complex IOR as a named spectrum, which is what this case can't
		// resolve on its own.
		if (m.hasConductorPreset) {
			d.type = MaterialType::Conductor;
			d.eta_c = make_float3(static_cast<float>(m.conductorEta[0]),
								   static_cast<float>(m.conductorEta[1]),
								   static_cast<float>(m.conductorEta[2]));
			d.k_c = make_float3(static_cast<float>(m.conductorK[0]),
								 static_cast<float>(m.conductorK[1]),
								 static_cast<float>(m.conductorK[2]));
		} else {
			d.type = MaterialType::Metal;
		}
		break;
	case pbrt_flatten::MaterialKind::Dielectric:
		// A nonzero "roughness"/"uroughness"/"vroughness" (m.roughness,
		// already copied into d.roughness above generically) means the scene
		// asked for a GGX microfacet dielectric (pbrt-v4 DielectricBxDF's
		// rough path) - matches pbrt_cpu_builder.h's identical branch. This
		// codebase already has a real GPU model for that
		// (MaterialType::RoughDielectric), it just wasn't wired up here.
		// RoughDielectric doesn't read d.albedo/transmission_filter at all
		// (see its own field-reuse comment in optix_types.h), so no reset
		// needed on this branch the way the smooth path needs below.
		if (m.roughness > 0.0) {
			d.type = MaterialType::RoughDielectric;
			break;
		}
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
		// Real tabulated BSSRDF, on BOTH GPU backends (see optix_types.h's
		// MaterialType::Subsurface comment for the full field-reuse layout
		// and backend history, shade_material()'s own comment in
		// optix_device_helpers.h for the recursive-backend probe-walk
		// algorithm, and wavefront_probe.h/wavefront_kernels.cu for the
		// wavefront backend's equivalent). `d.albedo` is deliberately left
		// as `m.color` (already assigned above, generically, before this
		// switch) rather than repurposed for sigma_a - a real Subsurface
		// material never carries a texture in this loader, so this slot is
		// only ever read as the CPU-parity fallback color.
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
	case pbrt_flatten::MaterialKind::Hair:
		// MaterialType::Hair is already fully wired for shading on both GPU
		// backends (sample_hair_material(), optix_device_helpers.h/
		// wavefront_kernels.cu) via 4 REUSED fields, not dedicated ones - see
		// that function's own comment for the exact mapping this mirrors:
		// d.ior=eta, d.albedo=sigma_a (RGB absorption - overrides the m.color
		// generic default assigned above, the same way Dielectric's own
		// branch resets its reused slot), d.fuzz=beta_m,
		// d.eta_c.x/y=beta_n/alpha_deg (eta_c.z unused).
		d.type = MaterialType::Hair;
		d.albedo = make_float3(static_cast<float>(m.sigma_a[0]),
								static_cast<float>(m.sigma_a[1]),
								static_cast<float>(m.sigma_a[2]));
		d.fuzz = static_cast<float>(m.betaM);
		d.eta_c = make_float3(static_cast<float>(m.betaN), static_cast<float>(m.alphaDeg), 0.0f);
		break;
	case pbrt_flatten::MaterialKind::Diffuse:
		// m.textureFilename mirrors measuredFilename: already a resolved,
		// existence-tested absolute path by this point (pbrt_load.h's own
		// pass, see Material::textureFilename's comment) or empty. A -1 from
		// getOrBuildPbrtImageTexture (corrupt-but-present file) leaves
		// d.textureIdx at its -1 default, so sample_texture()
		// (optix_device_helpers.h) falls through to solid_color-equivalent
		// behaviour reading `d.albedo` - same degrade-to-flat-colour the
		// Diffuse/Unsupported fallback below already gives every other
		// material without a texture.
		d.type = MaterialType::Lambertian;
		if (!m.textureFilename.empty())
			d.textureIdx = getOrBuildPbrtImageTexture(m.textureFilename, out, imageTextureCache);
		// m.hasCheckerReflectance (Material::hasCheckerReflectance's own
		// comment) - a procedural pbrt-v4 checkerboard, appended directly to
		// out.textures as a TextureKind::UVChecker entry (no cache/dedup:
		// each pbrt Texture declaration is already deduped 1:1 with the
		// Material referencing it by materialCache in build(), so this
		// runs at most once per distinct checkerboard material).
		else if (m.hasCheckerReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::UVChecker;
			tex.color1 = make_float3(static_cast<float>(m.checkerColor1[0]),
									 static_cast<float>(m.checkerColor1[1]),
									 static_cast<float>(m.checkerColor1[2]));
			tex.color2 = make_float3(static_cast<float>(m.checkerColor2[0]),
									 static_cast<float>(m.checkerColor2[1]),
									 static_cast<float>(m.checkerColor2[2]));
			tex.uScale = static_cast<float>(m.checkerUScale);
			tex.vScale = static_cast<float>(m.checkerVScale);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		// m.hasFbmReflectance/hasMarbleReflectance/hasMixReflectance
		// (Material's own comments) - same procedural-not-file,
		// append-one-TextureData pattern as hasCheckerReflectance above.
		else if (m.hasFbmReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::FBm;
			tex.omega = static_cast<float>(m.fbmRoughness);
			tex.octaves = m.fbmOctaves;
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasMarbleReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Marble;
			tex.omega = static_cast<float>(m.marbleRoughness);
			tex.octaves = m.marbleOctaves;
			tex.marbleScale = static_cast<float>(m.marbleScale);
			tex.marbleVariation = static_cast<float>(m.marbleVariation);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasMixReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Mix;
			tex.color1 = make_float3(static_cast<float>(m.mixColor1[0]),
									 static_cast<float>(m.mixColor1[1]),
									 static_cast<float>(m.mixColor1[2]));
			tex.color2 = make_float3(static_cast<float>(m.mixColor2[0]),
									 static_cast<float>(m.mixColor2[1]),
									 static_cast<float>(m.mixColor2[2]));
			tex.mixAmount = static_cast<float>(m.mixAmount);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		break;
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
	// Real stochastic two-material blend on GPU (pbrt-v4 MixMaterial), on
	// BOTH backends - see MaterialType::Mix's own comment (optix_types.h)
	// and resolve_mix_material()/wf_resolve_mix_material() (optix_device_
	// helpers.h/wavefront_kernels.cu) for the device-side per-shading-point
	// resolution this MaterialData only supplies indices/weight for; this
	// replaces the flat-Lambertian-averaged-colour approximation GPU used
	// to render every Mix as. `resolveMaterialIndex` is materialIndexDepth()
	// itself (build()'s own comment) - calling back into it, mutually
	// recursive through this function, gets each sub-material's REAL
	// out.materials index (not merely its flat colour the way the old
	// fallback did), so a Mix of e.g. Metal+Diffuse now renders as a real
	// per-point stochastic pick on GPU, matching CPU's mix_material exactly
	// instead of one averaged flat colour. Depth-capped (mirrors
	// resolveMixColor()'s own pre-existing kMaxMixDepth) and falls back to
	// that same flat-colour-average Lambertian beyond the cap or when the
	// sub-materials are not resolvable/no callback was supplied - guards a
	// cyclic/self-referential "materials" list a malformed scene could
	// produce (should not happen otherwise: flatten() already downgraded an
	// unresolvable mix to Unsupported before this MaterialKind is ever
	// reached - see there).
	case pbrt_flatten::MaterialKind::Mix: {
		constexpr int kMaxMixDepth = 8;
		const bool resolvable =
			m.mixMaterialA >= 0 && static_cast<std::size_t>(m.mixMaterialA) < allMaterials.size()
			&& m.mixMaterialB >= 0 && static_cast<std::size_t>(m.mixMaterialB) < allMaterials.size();
		if (resolvable && mixDepth < kMaxMixDepth && resolveMaterialIndex) {
			const int idxA = resolveMaterialIndex(m.mixMaterialA, -1, mixDepth + 1);
			const int idxB = resolveMaterialIndex(m.mixMaterialB, -1, mixDepth + 1);
			d.type = MaterialType::Mix;
			d.mix_extra.mixMaterialAIdx = static_cast<float>(idxA);
			d.mix_extra.mixMaterialBIdx = static_cast<float>(idxB);
			d.mix_extra.mixWeight = static_cast<float>(m.mixWeight);
		} else {
			d.type = MaterialType::Lambertian;
			if (resolvable) {
				const float3 ca = resolveMixColor(allMaterials[static_cast<std::size_t>(m.mixMaterialA)], allMaterials, 1);
				const float3 cb = resolveMixColor(allMaterials[static_cast<std::size_t>(m.mixMaterialB)], allMaterials, 1);
				const float w = static_cast<float>(m.mixWeight);
				d.albedo = make_float3(ca.x * (1.0f - w) + cb.x * w,
				                       ca.y * (1.0f - w) + cb.y * w,
				                       ca.z * (1.0f - w) + cb.z * w);
			}
		}
		break;
	}
	}

	// Material "texture displacement" (bump mapping - Material::
	// displacementTextureFilename's own comment). Mirrors scene_builder.cpp's
	// own OBJ/MTL map_Bump dispatch exactly, including its scope: only a
	// Lambertian material with no existing diffuse texture can host
	// NormalMappedLambertian (MaterialData has one shared textureIdx slot
	// per material - see add_normal_mapped_lambertian()'s own comment for
	// why), and only the RGB tangent-space normal-map case is wired at all -
	// GPU has no device-side scalar bump/height perturbation path today
	// (the grayscale case is a real, pre-existing, documented gap shared
	// with OBJ/MTL - see is_grayscale_texture_gpu()'s own comment - not
	// something pbrt-specific wiring alone can close).
	if (!m.displacementTextureFilename.empty() && d.type == MaterialType::Lambertian && d.textureIdx < 0) {
		const int dispTexIdx = getOrBuildPbrtImageTexture(m.displacementTextureFilename, out, imageTextureCache);
		if (dispTexIdx >= 0 && !isPbrtTextureGrayscale(out, dispTexIdx)) {
			d.type = MaterialType::NormalMappedLambertian;
			d.textureIdx = dispTexIdx;
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
	out.punctualLights.clear();

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
	// Separate dedup cache for GPU image textures-by-resolved-path - see
	// getOrBuildPbrtImageTexture()'s own comment. Only Diffuse materials with
	// an imagemap-bound reflectance touch this; stays empty otherwise.
	std::map<std::string, int> imageTextureCache;
	// Separate dedup cache for Shape "alpha" cutout masks - see
	// getOrBuildPbrtAlphaMaskTexture()'s own comment on why this can't share
	// imageTextureCache above even when the same file is used for both (its
	// decode must skip the gamma correction imageTextureCache's applies).
	std::map<std::string, int> alphaMaskTextureCache;
	// Two lambdas sharing the same `cache`/`out` state: materialIndexDepth is
	// the real, depth-threaded implementation (needed so MaterialKind::Mix's
	// own case in makeMaterial() below can recursively resolve its two
	// sub-materials to REAL out.materials indices - not just a flat colour,
	// see that case's own comment - while still bounding a cyclic/self-
	// referential "materials" list a malformed scene could produce, same
	// kMaxMixDepth guard as resolveMixColor()'s own pre-existing depth cap).
	// materialIndex is the ordinary 2-arg entry point every non-Mix call
	// site below already uses, unchanged, always starting at depth 0.
	// std::function (not an ordinary lambda) because it must be passed BY
	// REFERENCE into makeMaterial() so Mix can call back into it - an
	// ordinary `const auto` lambda can't appear in its own not-yet-deduced
	// type this way.
	std::function<int(int,int,int)> materialIndexDepth;
	materialIndexDepth = [&](int mi, int ai, int depth) -> int {
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

		// makeMaterial() is evaluated BEFORE `idx` is computed (not the other
		// way around, despite every other cache-then-push site in this file
		// looking that way) - a Mix material's own construction recursively
		// calls back into materialIndexDepth for its two sub-materials,
		// pushing THEM into out.materials as a side effect DURING this call.
		// Computing idx = out.materials.size() up front (the naive order)
		// would capture the size BEFORE those nested pushes, then push this
		// material at a LATER index once they've already grown the vector -
		// caching the wrong index for it. Every other material kind has no
		// such side effect, so this reordering is a no-op for them.
		MaterialData built = makeMaterial(m, em, out, bssrdfTableCache, measuredTableCache, imageTextureCache, alphaMaskTextureCache, scene.materials, materialIndexDepth, depth);
		const int idx = static_cast<int>(out.materials.size());
		out.materials.push_back(built);
		cache.emplace(key, idx);
		return idx;
	};
	const auto materialIndex = [&](int mi, int ai) {
		return materialIndexDepth(mi, ai, 0);
	};

	// MediumInterface "insideMedium" "" on a sphere. GPU's MaterialType::
	// Medium is sphere-only (see optix_types.h's comment on that enumerator)
	// AND takes over the sphere's material slot entirely, unlike the CPU
	// builder's constant_medium, which wraps a separately-added boundary
	// hittable around the sphere's own real surface material (pbrt_cpu_
	// builder.h's identical sphere loop) - so a scene pairing MediumInterface
	// with a dielectric surface (fog inside glass) loses the glass shell on
	// GPU and renders as a plain fog sphere instead. Documented, scoped
	// simplification (docs/PBRT_SUPPORT.md), not an oversight: layering a
	// second material onto one sphere would need a real combined material
	// slot (this codebase already has one for the dielectric+fog case
	// specifically - MaterialType::DielectricMedium, gpu/optix/scene_
	// builder.cpp's add_dielectric_medium() - but resolving here whether the
	// shape's own Material directive was "dielectric" to pick between the
	// two is more than this phase's scope). Same luminance/albedo-tint
	// collapse as pbrt_cpu_builder.h's identical derivation - see its
	// comment for why a scalar sigma_a/sigma_s plus a chromatic tint is
	// what MaterialData::medium_albedo/g/sigma_t actually store.
	std::map<int, int> mediumCache;
	const auto mediumMaterialIndex = [&](int medIdx) {
		const auto it = mediumCache.find(medIdx);
		if (it != mediumCache.end()) return it->second;
		const pbrt_flatten::Medium &md = scene.media[static_cast<std::size_t>(medIdx)];
		const auto luminance = [](const double c[3]) {
			return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
		};
		const double sig_a = luminance(md.sigma_a);
		const double sig_s = luminance(md.sigma_s);
		const float3 albedo = (sig_s > 1e-9)
			? make_float3(static_cast<float>(md.sigma_s[0] / sig_s),
						  static_cast<float>(md.sigma_s[1] / sig_s),
						  static_cast<float>(md.sigma_s[2] / sig_s))
			: make_float3(1.0f, 1.0f, 1.0f);

		// cloud/rgbgrid: real heterogeneous media, matching this codebase's
		// own E2/E4 showcase scenes' GPU construction exactly (see
		// gpu/optix/scene_builder.cpp's build_cloud_medium_scene_gpu/
		// build_rgb_grid_medium_scene_gpu - not called directly since those
		// live in a separate .cpp/translation unit; this mirrors their
		// short field-population logic instead, same "each loader
		// duplicates its own small helpers rather than sharing across a
		// .cpp/.h boundary" convention this file already uses for
		// getOrBuildPbrtImageTexture() etc). Same sphere-only, "replaces
		// the sphere's whole material slot" limitation the pre-existing
		// homogeneous MaterialType::Medium case above already has (see the
		// sphere loop just below) - not a new constraint these two add.
		if (md.type == "cloud") {
			float toMediumMatF[9], toMediumTranslateF[3];
			for (int i = 0; i < 9; ++i) toMediumMatF[i] = static_cast<float>(md.toMediumMat[i]);
			for (int i = 0; i < 3; ++i) toMediumTranslateF[i] = static_cast<float>(md.toMediumTranslate[i]);
			const CloudMedium<float> cloud = CloudMedium<float>::make(
				static_cast<float>(md.p0[0]), static_cast<float>(md.p0[1]), static_cast<float>(md.p0[2]),
				static_cast<float>(md.p1[0]), static_cast<float>(md.p1[1]), static_cast<float>(md.p1[2]),
				toMediumMatF, toMediumTranslateF,
				0.0f, static_cast<float>(sig_s), static_cast<float>(md.g),
				static_cast<float>(md.density), static_cast<float>(md.wispiness), static_cast<float>(md.frequency));
			const int cloudIdx = static_cast<int>(out.cloudMediums.size());
			out.cloudMediums.push_back(cloud);
			MaterialData d = {};
			d.type = MaterialType::CloudMedium;
			d.medium_albedo = make_float3(1.0f, 1.0f, 1.0f);
			d.g = static_cast<float>(md.g);
			d.cloud_medium_extra.cloudMediumIdx = static_cast<float>(cloudIdx);
			const int idx = static_cast<int>(out.materials.size());
			out.materials.push_back(d);
			mediumCache.emplace(medIdx, idx);
			return idx;
		}
		if (md.type == "rgbgrid") {
			// Scattering channels only - GpuRgbGridMedium/add_rgb_grid_
			// medium's own single (r,g,b) triple shape (see optix_types.h's
			// GpuRgbGridMedium - one dataOffset, not two) has no separate
			// absorption-grid slot the way CPU's RGBGridMediumData (real
			// sigma_a_grids alongside sigma_s_grids) does - a real,
			// GPU-specific simplification beyond CPU's fuller support, not
			// something this pbrt-loader path is introducing on its own.
			const std::size_t voxels = static_cast<std::size_t>(md.nx)
				* static_cast<std::size_t>(md.ny) * static_cast<std::size_t>(md.nz);
			const bool hasScattering = md.sigma_s_r.size() == voxels
				&& md.sigma_s_g.size() == voxels && md.sigma_s_b.size() == voxels;
			GpuRgbGridMedium meta{};
			for (int i = 0; i < 3; ++i) {
				meta.bounds_min[i] = static_cast<float>(md.worldMin[i]);
				meta.bounds_max[i] = static_cast<float>(md.worldMax[i]);
				meta.translate[i] = static_cast<float>(md.toMediumTranslate[i]);
			}
			for (int i = 0; i < 9; ++i) meta.mat[i] = static_cast<float>(md.toMediumMat[i]);
			meta.nx = md.nx; meta.ny = md.ny; meta.nz = md.nz;
			meta.sigma_scale = 1.0f;
			meta.phase_g = static_cast<float>(md.g);
			std::vector<float> rf, gf, bf;
			float max_density = 0.0f;
			if (hasScattering) {
				rf.assign(md.sigma_s_r.begin(), md.sigma_s_r.end());
				gf.assign(md.sigma_s_g.begin(), md.sigma_s_g.end());
				bf.assign(md.sigma_s_b.begin(), md.sigma_s_b.end());
				for (float v : rf) max_density = std::fmax(max_density, v);
				for (float v : gf) max_density = std::fmax(max_density, v);
				for (float v : bf) max_density = std::fmax(max_density, v);
			} else {
				// No scattering data (an absorption-only "rgbgrid" scene,
				// or a size mismatch already warned about at flatten() time)
				// - degrade to an empty, effectively invisible grid rather
				// than reading past the end of an empty vector.
				rf.assign(voxels, 0.0f); gf.assign(voxels, 0.0f); bf.assign(voxels, 0.0f);
			}
			meta.sigma_maj = max_density * meta.sigma_scale * 1.01f;   // small safety margin, matches scene_builder.cpp's own convention
			meta.dataOffset = static_cast<int>(out.rgbGridData.size());
			out.rgbGridData.insert(out.rgbGridData.end(), rf.begin(), rf.end());
			out.rgbGridData.insert(out.rgbGridData.end(), gf.begin(), gf.end());
			out.rgbGridData.insert(out.rgbGridData.end(), bf.begin(), bf.end());
			const int gridIdx = static_cast<int>(out.rgbGridMediums.size());
			out.rgbGridMediums.push_back(meta);
			MaterialData d = {};
			d.type = MaterialType::RgbGridMedium;
			d.rgb_grid_medium_extra.rgbGridMediumIdx = static_cast<float>(gridIdx);
			const int idx = static_cast<int>(out.materials.size());
			out.materials.push_back(d);
			mediumCache.emplace(medIdx, idx);
			return idx;
		}
		if (md.type == "uniformgrid") {
			// Single-channel twin of the rgbgrid branch just above - see
			// that one's own comment. sigma_a is dropped the same way it is
			// for cloud (pure scattering) - flatten() already warned about
			// this. sig_s (luminance(md.sigma_s), already computed above for
			// the cloud branch) is the per-voxel scattering multiplier,
			// matching CPU's pbrt_cpu_builder.h uniformgrid branch exactly.
			const std::size_t voxels = static_cast<std::size_t>(md.nx)
				* static_cast<std::size_t>(md.ny) * static_cast<std::size_t>(md.nz);
			const bool hasDensity = md.gridDensity.size() == voxels;
			GpuGridMedium meta{};
			for (int i = 0; i < 3; ++i) {
				meta.bounds_min[i] = static_cast<float>(md.worldMin[i]);
				meta.bounds_max[i] = static_cast<float>(md.worldMax[i]);
				meta.translate[i] = static_cast<float>(md.toMediumTranslate[i]);
			}
			for (int i = 0; i < 9; ++i) meta.mat[i] = static_cast<float>(md.toMediumMat[i]);
			meta.nx = md.nx; meta.ny = md.ny; meta.nz = md.nz;
			meta.sigma_scale = static_cast<float>(sig_s);
			meta.phase_g = static_cast<float>(md.g);
			std::vector<float> df;
			float max_density = 0.0f;
			if (hasDensity) {
				df.assign(md.gridDensity.begin(), md.gridDensity.end());
				for (float v : df) max_density = std::fmax(max_density, v);
			} else {
				// No density data (missing/wrong-length "float density" -
				// flatten() already warned) - degrade to an empty, invisible
				// grid rather than reading past the end of an empty vector.
				df.assign(voxels, 0.0f);
			}
			meta.sigma_maj = max_density * meta.sigma_scale * 1.01f;   // small safety margin, matches rgbgrid's own convention
			meta.dataOffset = static_cast<int>(out.gridData.size());
			out.gridData.insert(out.gridData.end(), df.begin(), df.end());
			const int gridIdx = static_cast<int>(out.gridMediums.size());
			out.gridMediums.push_back(meta);
			MaterialData d = {};
			d.type = MaterialType::GridMedium;
			d.medium_albedo = make_float3(1.0f, 1.0f, 1.0f);
			d.grid_medium_extra.gridMediumIdx = static_cast<float>(gridIdx);
			const int idx = static_cast<int>(out.materials.size());
			out.materials.push_back(d);
			mediumCache.emplace(medIdx, idx);
			return idx;
		}

		MaterialData d = {};
		d.type = MaterialType::Medium;
		d.medium_albedo = albedo;
		d.g = static_cast<float>(md.g);
		d.sigma_t = static_cast<float>(sig_a + sig_s);
		const int idx = static_cast<int>(out.materials.size());
		out.materials.push_back(d);
		mediumCache.emplace(medIdx, idx);
		return idx;
	};

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : scene.spheres) {
		SphereData sd = {};
		sd.center = f3(s.center);
		sd.center1 = sd.center;          // static; see SphereData's comment
		sd.radius = static_cast<float>(s.radius);
		sd.materialIdx = (s.medium >= 0 && static_cast<std::size_t>(s.medium) < scene.media.size())
			? mediumMaterialIndex(s.medium)
			: materialIndex(s.material, s.areaLight);
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

	// ---- curves -------------------------------------------------------------
	// Shape "curve" - neither GPU backend has a native curve-intersection
	// program (see src/shared/curve_tessellate.h's own comment - pbrt-v4's
	// GPU path makes the identical choice, for the identical reason), so each
	// already-per-segment-split Curve (pbrt_flatten::Curve, see its own
	// comment) is diced into a tapered tube of bilinear-patch quads via the
	// same curve_tessellate::tessellate() call build_curve_fibers_scene_gpu()
	// (scene_builder.cpp) already uses for its native (non-pbrt) curve demo,
	// with that function's own density constants (n_length=10, n_radial=8) as
	// a reasonable default. Curve `type` (flat/cylinder/ribbon) is NOT
	// distinguished here - the tessellator only emits a round tube regardless
	// - a real, documented CPU/GPU visual divergence (see docs/
	// PBRT_SUPPORT.md), not an oversight; pbrt-v4's own GPU dicing has the
	// identical divergence for the identical reason.
	for (const pbrt_flatten::Curve &cv : scene.curves) {
		const int matIdx = materialIndex(cv.material, cv.areaLight);
		const int n_length = 10, n_radial = 8;
		std::vector<curve_tessellate::Quad> quads;
		for (int seg = 0; seg < cv.nSegments; ++seg) {
			const std::size_t base = static_cast<std::size_t>(seg) * 4;
			float cp[4][3];
			for (int i = 0; i < 4; ++i) {
				const std::size_t idx = (base + static_cast<std::size_t>(i)) * 3;
				cp[i][0] = static_cast<float>(cv.cp[idx]);
				cp[i][1] = static_cast<float>(cv.cp[idx + 1]);
				cp[i][2] = static_cast<float>(cv.cp[idx + 2]);
			}
			const double t0 = static_cast<double>(seg) / cv.nSegments;
			const double t1 = static_cast<double>(seg + 1) / cv.nSegments;
			const float segW0 = static_cast<float>(cv.width0 + (cv.width1 - cv.width0) * t0);
			const float segW1 = static_cast<float>(cv.width0 + (cv.width1 - cv.width0) * t1);
			quads.clear();
			curve_tessellate::tessellate(cp, 0.0f, 1.0f, segW0, segW1, n_length, n_radial, quads);
			for (const curve_tessellate::Quad &q : quads) {
				BilinearPatchData bd = {};
				bd.p00 = make_float3(q.p00[0], q.p00[1], q.p00[2]);
				bd.p10 = make_float3(q.p10[0], q.p10[1], q.p10[2]);
				bd.p01 = make_float3(q.p01[0], q.p01[1], q.p01[2]);
				bd.p11 = make_float3(q.p11[0], q.p11[1], q.p11[2]);
				bd.materialIdx = matIdx;
				out.bilinearPatches.push_back(bd);
			}
		}
		stats.curveSegments += static_cast<std::size_t>(cv.nSegments);
	}

	// ---- disks / cylinders -------------------------------------------------
	// Shape "disk"/"cylinder" - supported on both the recursive (Phase 4b)
	// and wavefront (Phase 4c) GPU backends; see optix_types.h's DiskData/
	// CylinderData comment for why these carry their own transform rather
	// than being baked to world space the way Sphere is. Not registered as
	// NEE-samplable lights yet even when areaLight >= 0 (no
	// GpuLightKind::Disk/Cylinder exists) - an emissive one still emits when
	// directly hit, just without explicit light sampling, same "hit but not
	// aimed-at" tier documented in docs/PBRT_SUPPORT.md. The medium field
	// (d.medium/c.medium) is intentionally unread here, UNLIKE the sphere
	// loop above (which resolves s.medium via mediumMaterialIndex()) - a
	// real, known GPU-only gap (a MediumInterface around a disk/cylinder is
	// silently dropped on GPU), not yet fixed here.
	const auto flattenTransform = [](const double xform[16], float out12[12]) {
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 4; ++col)
				out12[row * 4 + col] = static_cast<float>(xform[row * 4 + col]);
	};
	const double kDiskCylDegToRad = 3.14159265358979323846 / 180.0;

	for (const pbrt_flatten::Disk &d : scene.disks) {
		pbrt_scene::Matrix4 o2w;
		for (int i = 0; i < 16; ++i) o2w.m[i] = d.xform[i];
		pbrt_scene::Matrix4 w2o;
		if (!o2w.inverseAffine(w2o)) continue;  // singular - drop, matching disk_hittable's own valid_ guard

		DiskData dd = {};
		dd.radius = static_cast<float>(d.radius);
		dd.innerRadius = static_cast<float>(d.innerRadius);
		dd.height = static_cast<float>(d.height);
		dd.phiMax = static_cast<float>(d.phiMaxDeg * kDiskCylDegToRad);
		dd.materialIdx = materialIndex(d.material, d.areaLight);
		flattenTransform(o2w.m, dd.o2w);
		flattenTransform(w2o.m, dd.w2o);
		out.disks.push_back(dd);
	}
	stats.disks = out.disks.size();

	for (const pbrt_flatten::Cylinder &c : scene.cylinders) {
		pbrt_scene::Matrix4 o2w;
		for (int i = 0; i < 16; ++i) o2w.m[i] = c.xform[i];
		pbrt_scene::Matrix4 w2o;
		if (!o2w.inverseAffine(w2o)) continue;

		CylinderData cd = {};
		cd.radius = static_cast<float>(c.radius);
		cd.zMin = static_cast<float>(c.zMin);
		cd.zMax = static_cast<float>(c.zMax);
		cd.phiMax = static_cast<float>(c.phiMaxDeg * kDiskCylDegToRad);
		cd.materialIdx = materialIndex(c.material, c.areaLight);
		flattenTransform(o2w.m, cd.o2w);
		flattenTransform(w2o.m, cd.w2o);
		out.cylinders.push_back(cd);
	}
	stats.cylinders = out.cylinders.size();

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
		if (t.hasUVs) {
			td.uv0 = f2(&t.uv[0]);
			td.uv1 = f2(&t.uv[2]);
			td.uv2 = f2(&t.uv[4]);
		}
		td.hasUVs = t.hasUVs;
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
			if (t.hasUVs) {
				td.uv0 = f2(&t.uv[0]);
				td.uv1 = f2(&t.uv[2]);
				td.uv2 = f2(&t.uv[4]);
			}
			td.hasUVs = t.hasUVs;
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

			// Real importance-sampled HDR sky (GpuSkyDistribution - see
			// optix_types.h's own comment): the SAME PiecewiseConstant2D
			// construction src/TheRestOfYourLife/sky_light.h's private
			// constructor builds (Rec.709 luminance * sin(theta) weights,
			// sin(theta) being the equirectangular solid-angle Jacobian
			// correction - see that file's own header comment), from the
			// SAME already-decoded image `stats.backgroundColor` just used
			// above for its mean-colour fallback. That mean colour stays as
			// computed either way - it still serves both as the "does this
			// scene have a sky at all" gate every NEE/miss call site already
			// checks (backgroundColor != 0) and as the flat-colour fallback
			// for the rare degenerate case where the distribution itself
			// ends up empty (e.g. an all-black image).
			std::vector<double> weights(static_cast<std::size_t>(sky.imageWidth) * sky.imageHeight);
			constexpr double kPi = 3.14159265358979323846;
			for (int v = 0; v < sky.imageHeight; ++v) {
				const double theta = (v + 0.5) / sky.imageHeight * kPi;
				const double sin_theta = std::sin(theta);
				for (int u = 0; u < sky.imageWidth; ++u) {
					const float* px = &sky.imagePixels[(static_cast<std::size_t>(v) * sky.imageWidth + u) * 3];
					const double lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
					weights[static_cast<std::size_t>(v) * sky.imageWidth + u] = (lum > 0.0 ? lum : 0.0) * sin_theta;
				}
			}
			const PiecewiseConstant2D dist(weights, sky.imageWidth, sky.imageHeight);
			if (!dist.empty()) {
				const PiecewiseConstant1D& marg = dist.Marginal();
				out.skyMarginalFuncInt = static_cast<float>(marg.integral());
				out.skyMarginalCdf.reserve(marg.Cdf().size());
				for (double d : marg.Cdf()) out.skyMarginalCdf.push_back(static_cast<float>(d));
				out.skyMarginalFunc.reserve(marg.Func().size());
				for (double d : marg.Func()) out.skyMarginalFunc.push_back(static_cast<float>(d));

				const int nv = dist.NV();
				out.skyConditionalCdf.reserve(static_cast<std::size_t>(nv) * (sky.imageWidth + 1));
				out.skyConditionalFunc.reserve(static_cast<std::size_t>(nv) * sky.imageWidth);
				out.skyConditionalFuncInt.reserve(nv);
				for (int r = 0; r < nv; ++r) {
					const PiecewiseConstant1D& cond = dist.Conditional(r);
					for (double d : cond.Cdf()) out.skyConditionalCdf.push_back(static_cast<float>(d));
					for (double d : cond.Func()) out.skyConditionalFunc.push_back(static_cast<float>(d));
					out.skyConditionalFuncInt.push_back(static_cast<float>(cond.integral()));
				}

				out.skyImagePixels = sky.imagePixels; // already row-major float RGB, linear
				out.skyWidth = sky.imageWidth;
				out.skyHeight = sky.imageHeight;
				out.skyScale = static_cast<float>(sky.scale);
			}
		} else {
			stats.backgroundColor = make_float3(
				static_cast<float>(sky.L[0] * sky.scale),
				static_cast<float>(sky.L[1] * sky.scale),
				static_cast<float>(sky.L[2] * sky.scale));
		}
	}

	// ---- punctual (delta) lights -------------------------------------------
	// LightSource point/spot/distant/goniometric/projection - one
	// PunctualLightGPU per parsed pbrt_flatten::PunctualLight. Mirrors
	// scene_builder.cpp's own build_spotlight_cornell_gpu() and its four
	// siblings (scenes 25-29) field for field - same PunctualLightKind
	// tags, same struct layout - just filled from parsed values instead of
	// hardcoded ones. optix_renderer.cpp uploads whatever ends up in
	// SceneData::punctualLights generically (it does not know or care
	// whether a scene is one of the hand-built C2-C6 showcases or a loaded
	// .pbrt file), so nothing past this point needs touching for either GPU
	// backend to render these.
	constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
	for (const pbrt_flatten::PunctualLight &pl : scene.punctualLights) {
		PunctualLightGPU light{};
		switch (pl.kind) {
		case pbrt_flatten::PunctualLightKind::Point:
			light.kind = PunctualLightKind::Point;
			light.point.pos_x = static_cast<float>(pl.pos[0]);
			light.point.pos_y = static_cast<float>(pl.pos[1]);
			light.point.pos_z = static_cast<float>(pl.pos[2]);
			light.point.ir = static_cast<float>(pl.intensity[0]);
			light.point.ig = static_cast<float>(pl.intensity[1]);
			light.point.ib = static_cast<float>(pl.intensity[2]);
			light.point.scale = static_cast<float>(pl.scale);
			break;
		case pbrt_flatten::PunctualLightKind::Spot:
			light.kind = PunctualLightKind::Spot;
			light.spot.pos_x = static_cast<float>(pl.pos[0]);
			light.spot.pos_y = static_cast<float>(pl.pos[1]);
			light.spot.pos_z = static_cast<float>(pl.pos[2]);
			light.spot.dir_x = static_cast<float>(pl.dir[0]);
			light.spot.dir_y = static_cast<float>(pl.dir[1]);
			light.spot.dir_z = static_cast<float>(pl.dir[2]);
			light.spot.ir = static_cast<float>(pl.intensity[0]);
			light.spot.ig = static_cast<float>(pl.intensity[1]);
			light.spot.ib = static_cast<float>(pl.intensity[2]);
			light.spot.scale = static_cast<float>(pl.scale);
			light.spot.cos_falloff_start =
				static_cast<float>(std::cos(pl.falloffStartAngleDeg * kDegToRad));
			light.spot.cos_falloff_end =
				static_cast<float>(std::cos(pl.coneAngleDeg * kDegToRad));
			break;
		case pbrt_flatten::PunctualLightKind::Distant:
			light.kind = PunctualLightKind::Distant;
			light.distant.dir_x = static_cast<float>(pl.dir[0]);
			light.distant.dir_y = static_cast<float>(pl.dir[1]);
			light.distant.dir_z = static_cast<float>(pl.dir[2]);
			light.distant.ir = static_cast<float>(pl.intensity[0]);
			light.distant.ig = static_cast<float>(pl.intensity[1]);
			light.distant.ib = static_cast<float>(pl.intensity[2]);
			light.distant.scale = static_cast<float>(pl.scale);
			light.distant.scene_radius = static_cast<float>(pl.sceneRadius);
			break;
		case pbrt_flatten::PunctualLightKind::Goniometric: {
			light.kind = PunctualLightKind::Goniometric;
			GoniometricLightGPU &g = light.gonio;
			g.pos_x = static_cast<float>(pl.pos[0]);
			g.pos_y = static_cast<float>(pl.pos[1]);
			g.pos_z = static_cast<float>(pl.pos[2]);
			for (int c = 0; c < 9; ++c) g.world_to_light[c] = static_cast<float>(pl.worldToLight[c]);
			g.ir = static_cast<float>(pl.intensity[0]);
			g.ig = static_cast<float>(pl.intensity[1]);
			g.ib = static_cast<float>(pl.intensity[2]);
			g.scale = static_cast<float>(pl.scale);
			// Uniform (isotropic) image - see pbrt_flatten::PunctualLight::
			// hadImageFilename's comment for why no real IES profile is ever
			// decoded. Matches pbrt_cpu_builder.h's own kUniformImage.
			g.nu = 4; g.nv = 4;
			for (int i = 0; i < g.nu * g.nv; ++i) g.image[i] = 1.0f;
			break;
		}
		case pbrt_flatten::PunctualLightKind::Projection: {
			light.kind = PunctualLightKind::Projection;
			ProjectionLightGPU &pr = light.proj;
			pr.pos_x = static_cast<float>(pl.pos[0]);
			pr.pos_y = static_cast<float>(pl.pos[1]);
			pr.pos_z = static_cast<float>(pl.pos[2]);
			for (int c = 0; c < 9; ++c) pr.world_to_light[c] = static_cast<float>(pl.worldToLight[c]);
			pr.scale = static_cast<float>(pl.scale);
			pr.hither = 1e-3f;
			// Uniform white slide - see pbrt_flatten::PunctualLight::
			// hadImageFilename's comment for why no real projected image is
			// ever decoded. Matches pbrt_cpu_builder.h's own kUniformSlide
			// (2x2, aspect 1 so sb_xmin/xmax below match
			// ProjectionLight<T>::make's own aspect>=1 branch exactly).
			pr.nx = 2; pr.ny = 2;
			pr.sb_xmin = -1.0f; pr.sb_xmax = 1.0f;
			pr.sb_ymin = -1.0f; pr.sb_ymax = 1.0f;
			pr.inv_tan = 1.0f / static_cast<float>(std::tan(pl.fovDeg * kDegToRad * 0.5));
			for (int i = 0; i < pr.nx * pr.ny * 3; ++i) pr.image_rgb[i] = 1.0f;
			break;
		}
		}
		out.punctualLights.push_back(light);
	}

	return stats;
}

} // namespace pbrt_gpu
