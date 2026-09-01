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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
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
#include "../../src/shared/portal_image_infinite_light.h"
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
//
// gamma/wrap/invert: Texture "imagemap"'s own "string encoding"/"string
// wrap"/"bool invert" (pbrt-v4), resolved by pbrt_flatten.h into Material::
// textureGamma/textureWrap/textureInvert for the primary "reflectance" slot
// ONLY (Diffuse/CoatedDiffuse/DiffuseTransmission) - same scope as CPU's own
// imageMapOptionsFor() (pbrt_cpu_builder.h), matching this loader's
// established "close the reflectance slot first" precedent (see that
// function's own comment). Every OTHER call site below (roughness/
// transmittance/displacement/checker/mix tex1/tex2/amount) still calls this
// with the defaults below, so their behavior is completely unchanged by this
// parameter's addition. gamma/invert are baked into the decoded pixel bytes
// here (matching CPU's rtw_image::load()/mipmap_texture::build_from() -
// applied once at load time, not re-applied per sample); wrap is stored on
// the built TextureData for sampleImage() (optix_device_helpers.h) to read
// per texel lookup, matching CPU's own gamma/invert-at-load vs.
// wrap-at-sample split exactly.
// Mirrors src/shared/mipmap.h's own kDefaultImagemapGamma constant (not
// shared via #include - pulling in the full MipMap class/128-entry EWA LUT
// for one float here isn't worth the extra include-graph surface in a TU
// that never uses the rest of that header, matching how this file already
// hand-duplicates a handful of other CPU-side constants rather than
// including their whole home header).
constexpr float kGpuImagemapDefaultGamma = 2.2f;

inline int getOrBuildPbrtImageTexture(const std::string& resolvedPath, SceneData& out,
									  std::map<std::string, int>& cache,
									  float gamma = kGpuImagemapDefaultGamma,
									  GpuWrapMode wrap = GpuWrapMode::Clamp,
									  bool invert = false) {
	// Always a composite key - `cache` is a fresh std::map local to one
	// buildPbrtMaterials() call and never read/written anywhere but this
	// function, so there's no external reader relying on a plain-path key
	// for the default-options case; always including gamma/wrap/invert here
	// costs a few extra bytes per key and nothing else, while guaranteeing a
	// file shared with different options across materials always gets its
	// own, separately-decoded (gamma/invert-baked) entry.
	//
	// gamma is encoded via its exact IEEE-754 bit pattern, not
	// std::to_string(gamma) (fixed 6 decimal digits) - two distinct-but-close
	// requested gamma values could otherwise format identically and
	// collide, silently sharing pixel bytes baked for the wrong exponent.
	std::uint32_t gammaBits;
	std::memcpy(&gammaBits, &gamma, sizeof(gammaBits));
	const std::string cacheKey = resolvedPath + "|g" + std::to_string(gammaBits) + "|w" +
		std::to_string(static_cast<int>(wrap)) + "|i" + (invert ? "1" : "0");
	const auto it = cache.find(cacheKey);
	if (it != cache.end()) return it->second;

	// stbi_ldr_to_hdr_gamma is process-global mutable state (see rtw_stb_
	// image.h's identical CPU-side use for the full rationale) - mutated for
	// the duration of this one stbi_loadf() call, then restored. Real GPU
	// scene building is single-threaded today (no concurrent caller of this
	// function exists anywhere in this codebase), but that used to be an
	// unenforced assumption rather than a guarantee - this mutex actually
	// closes the gap for the realistic future case (a batch/preview mode
	// building more than one GPU scene concurrently within one process): it
	// doesn't extend to CPU's own identical mutation in rtw_stb_image.h (a
	// genuinely cross-module race would need a mutex shared with that file
	// too, out of scope for a GPU-side fix), but no such CPU/GPU-concurrent
	// scene-building path exists in this codebase either.
	static std::mutex gammaMutex;
	int width = 0, height = 0, channels = 0;
	float* fdata = nullptr;
	{
		std::lock_guard<std::mutex> lock(gammaMutex);
		stbi_ldr_to_hdr_gamma(gamma);
		fdata = stbi_loadf(resolvedPath.c_str(), &width, &height, &channels, 3);
		stbi_ldr_to_hdr_gamma(kGpuImagemapDefaultGamma);
	}
	if (!fdata) {
		cache.emplace(cacheKey, -1);
		return -1;
	}

	TextureData tex{};
	tex.kind = TextureKind::Image;
	tex.noiseScale = 0.0f;
	tex.pixelOffset = static_cast<int>(out.texturePixels.size());
	tex.width = width;
	tex.height = height;
	tex.wrapMode = wrap;
	const std::size_t total = static_cast<std::size_t>(width) * height * 3;
	out.texturePixels.resize(out.texturePixels.size() + total);
	unsigned char* dst = out.texturePixels.data() + tex.pixelOffset;
	for (std::size_t i = 0; i < total; ++i) {
		const float v = fdata[i];
		unsigned char q = (v <= 0.0f) ? 0 : (v >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * v));
		// Matches CPU's own quantize-then-invert order exactly
		// (mipmap_texture::build_from(), texture.h): CPU reads the ALREADY-
		// quantized byte back via rtw_image::pixel_data(), reconstructs a
		// [0,1] value via /255 (not /256 - see build_from()'s own `scale`),
		// and only then applies invert (1-c, clamped at 0) - inverting the
		// raw pre-quantization float here instead (as an earlier version of
		// this code did) diverges from CPU by 1 LSB whenever the decoded
		// value lands exactly on a multiple of 1/256.
		if (invert) {
			const float reconstructed = q / 255.0f;
			const float inverted = fmaxf(0.0f, 1.0f - reconstructed);
			q = (inverted <= 0.0f) ? 0 : (inverted >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * inverted));
		}
		dst[i] = q;
	}
	stbi_image_free(fdata);

	const int idx = static_cast<int>(out.textures.size());
	out.textures.push_back(tex);
	cache.emplace(cacheKey, idx);
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
// Shared by resolveMixColor() below and makeMaterial()'s own MaterialKind::
// Mix case - a single file-scope constant instead of two independently-
// declared local ones (they must stay in lockstep: makeMaterial()'s real
// GPU-side Mix resolution and this flat-colour fallback need the identical
// cap so a scene doesn't get a different effective mix-of-mix depth
// depending on which of the two paths handles a given nesting level).
inline constexpr int kMaxMixDepth = 8;

inline float3 resolveMixColor(const pbrt_flatten::Material &m,
							  const std::vector<pbrt_flatten::Material> &allMaterials,
							  int depth) {
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
			// Real independent u/v roughness (matches pbrt_cpu_builder.h's
			// identical Conductor branch) - d.roughness (set generically
			// from m.roughness above) is overridden to the u-axis value
			// specifically, since m.roughness and m.roughness_u only
			// coincide when a scene doesn't set "uroughness"/"roughness"
			// to different values (see pbrt_flatten.h's own fallback-chain
			// comment). d.roughnessV is always the real m.roughness_v value,
			// unconditionally - NOT gated on "does it differ from
			// roughness_u": m.roughness_v is a real, always-present double
			// (never an ambiguous "unset" state), and MaterialData::
			// roughnessV's own sentinel is negative (-1.0f), which no real
			// roughness value can ever equal - so a scene that authors a
			// legitimate roughness_v of exactly 0.0 (e.g. only "uroughness"
			// given) stores a real, unambiguous 0.0f here instead of
			// colliding with the isotropic sentinel.
			d.roughness  = static_cast<float>(m.roughness_u);
			d.roughnessV = static_cast<float>(m.roughness_v);
		} else {
			d.type = MaterialType::Metal;
		}
		break;
	// Pass-through "interface" material (pbrt-v4's Material "none"/"" -
	// see MaterialKind::Interface's own comment). flatten() already forced
	// m.ior to 1.001 for this kind, and d.ior is already assigned
	// generically from m.ior above, so this shares Dielectric's exact
	// build path with no special-casing needed here.
	// Real, dedicated pass-through material (MaterialType::Interface) - see
	// its own comment (optix_types.h). No Fresnel/refraction math, no
	// MaterialData fields needed, so this is its own case, not a Dielectric
	// fallthrough.
	case pbrt_flatten::MaterialKind::Interface:
		d.type = MaterialType::Interface;
		break;
	case pbrt_flatten::MaterialKind::Dielectric:
		// m.roughnessTextureFilename (Material::roughnessTextureFilename
		// own comment, pbrt_flatten.h) - checked FIRST, same priority
		// pbrt_cpu_builder.h's identical branch gives it. d.textureIdx is
		// otherwise unused/free for RoughDielectric (unlike Conductor/
		// CoatedDiffuse/DiffuseTransmission above, which reuse it for their
		// own texture-bound reflectance), so reusing it here for "sample
		// this texture for roughness instead of d.roughness" doesn't
		// collide with anything. A -1 from getOrBuildPbrtImageTexture
		// (corrupt-but-present file) falls through to the flat-roughness
		// check below instead of forcing RoughDielectric with an unusable
		// texture index - m.roughness_u/m.roughness_v are both still 0.0
		// here (flatten() never sets them when a texture was bound), so
		// that degrades to smooth Dielectric, same "falls back to a
		// constant/default value" convention every other texture-decode
		// failure in this codebase gets.
		if (!m.roughnessTextureFilename.empty()) {
			const int roughTexIdx = getOrBuildPbrtImageTexture(m.roughnessTextureFilename, out, imageTextureCache,
				static_cast<float>(m.roughnessTextureOptions.gamma),
				static_cast<GpuWrapMode>(m.roughnessTextureOptions.wrapIndex),
				m.roughnessTextureOptions.invert);
			if (roughTexIdx >= 0) {
				d.type = MaterialType::RoughDielectric;
				d.textureIdx = roughTexIdx;
				break;
			}
		}
		// A nonzero "roughness"/"uroughness"/"vroughness" means the scene
		// asked for a GGX microfacet dielectric (pbrt-v4 DielectricBxDF's
		// rough path) - matches pbrt_cpu_builder.h's identical
		// `m.roughness_u > 0.0 || m.roughness_v > 0.0` gate exactly (not
		// just m.roughness, which can miss a scene that sets only
		// "vroughness" to a nonzero value with no "roughness"/"uroughness"
		// - see pbrt_flatten.h's own fallback-chain comment on why m.roughness
		// alone still happens to catch the common cases but not every one).
		// This codebase already has a real GPU model for that
		// (MaterialType::RoughDielectric), it just wasn't wired up here.
		// RoughDielectric doesn't read d.albedo/transmission_filter at all
		// (see its own field-reuse comment in optix_types.h), so no reset
		// needed on this branch the way the smooth path needs below.
		if (m.roughness_u > 0.0 || m.roughness_v > 0.0) {
			d.type = MaterialType::RoughDielectric;
			// Real independent u/v roughness - see MaterialType::Conductor's
			// identical-shape override above for the full rationale.
			d.roughness  = static_cast<float>(m.roughness_u);
			d.roughnessV = static_cast<float>(m.roughness_v);
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
		// m.textureFilename/m.textureScale (Material::textureFilename's own
		// comment) - same resolved-by-pbrt_load.h convention, and the same
		// getOrBuildPbrtImageTexture() call, the Diffuse case below already
		// uses. d.emissionScale (reused, not a dedicated field - see its own
		// comment in optix_types.h) carries a "scale"-class wrapper's
		// multiplier (barcelona-pavilion's own dominant pattern); stays at
		// its 1.0 no-op default for a bare imagemap (ganesha's pattern).
		// A -1 from getOrBuildPbrtImageTexture (corrupt-but-present file)
		// leaves d.textureIdx at its -1 default, falling back to `d.albedo`
		// (already assigned generically above) - same degrade-to-flat-
		// colour every other texture-bound material kind gets.
		d.type = MaterialType::CoatedDiffuse;
		if (!m.textureFilename.empty()) {
			d.textureIdx = getOrBuildPbrtImageTexture(m.textureFilename, out, imageTextureCache,
				static_cast<float>(m.textureGamma), static_cast<GpuWrapMode>(m.textureWrapIndex), m.textureInvert);
			d.emissionScale = static_cast<float>(m.textureScale);
		}
		// m.hasCheckerReflectance/hasFbmReflectance/hasMarbleReflectance/
		// hasMixReflectance (Material's own comments) - same procedural-not-
		// file, append-one-TextureData pattern as the Diffuse case below,
		// now also resolved for CoatedDiffuse (previously Diffuse-only).
		// sample_texture() (optix_device_helpers.h) dispatches purely on
		// TextureData::kind, not on MaterialType, so pointing d.textureIdx
		// at a procedural entry works identically for CoatedDiffuse's own
		// "reflectance" read as it already does for Lambertian's.
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
			if (!m.checkerTex1Filename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.checkerTex1Filename, out, imageTextureCache);
			if (!m.checkerTex2Filename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.checkerTex2Filename, out, imageTextureCache);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
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
			if (!m.mixTex1Filename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.mixTex1Filename, out, imageTextureCache);
			if (!m.mixTex2Filename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.mixTex2Filename, out, imageTextureCache);
			// m.mixAmountTextureFilename (Material::mixAmountTextureFilename's
			// own comment) - a real per-point spatially-varying blend when
			// "amount" itself nested a bare imagemap; tex.mixAmount (above)
			// stays at its resolved flat value and is unused by
			// sample_texture()/wf_sample_texture() when amountImageIdx >= 0.
			if (!m.mixAmountTextureFilename.empty())
				tex.amountImageIdx = getOrBuildPbrtImageTexture(m.mixAmountTextureFilename, out, imageTextureCache);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasWindyReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Windy;
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasWrinkledReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Wrinkled;
			tex.omega = static_cast<float>(m.wrinkledRoughness);
			tex.octaves = m.wrinkledOctaves;
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasDotsReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Dots;
			tex.color1 = make_float3(static_cast<float>(m.dotsInsideColor[0]),
									 static_cast<float>(m.dotsInsideColor[1]),
									 static_cast<float>(m.dotsInsideColor[2]));
			tex.color2 = make_float3(static_cast<float>(m.dotsOutsideColor[0]),
									 static_cast<float>(m.dotsOutsideColor[1]),
									 static_cast<float>(m.dotsOutsideColor[2]));
			if (!m.dotsInsideTexFilename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.dotsInsideTexFilename, out, imageTextureCache);
			if (!m.dotsOutsideTexFilename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.dotsOutsideTexFilename, out, imageTextureCache);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasBilerpReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Bilerp;
			tex.color1 = make_float3(static_cast<float>(m.bilerpV00[0]),
									 static_cast<float>(m.bilerpV00[1]),
									 static_cast<float>(m.bilerpV00[2]));
			tex.color2 = make_float3(static_cast<float>(m.bilerpV01[0]),
									 static_cast<float>(m.bilerpV01[1]),
									 static_cast<float>(m.bilerpV01[2]));
			tex.bilerpV10 = make_float3(static_cast<float>(m.bilerpV10[0]),
										static_cast<float>(m.bilerpV10[1]),
										static_cast<float>(m.bilerpV10[2]));
			tex.bilerpV11 = make_float3(static_cast<float>(m.bilerpV11[0]),
										static_cast<float>(m.bilerpV11[1]),
										static_cast<float>(m.bilerpV11[2]));
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		// Real independent u/v coat roughness - see MaterialType::Conductor's
		// identical-shape override above for the full rationale.
		d.roughness  = static_cast<float>(m.roughness_u);
		d.roughnessV = static_cast<float>(m.roughness_v);
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
		// m.textureFilename/m.transmittanceTextureFilename (own comments in
		// pbrt_flatten.h) - barcelona-pavilion's foliage binds both
		// "reflectance" and "transmittance" to the SAME bare imagemap, each
		// optionally further wrapped in its own independent "scale" texture
		// (m.textureScale/m.transmittanceTextureScale's own comments).
		// d.textureIdx covers reflectance (falls back to d.albedo, already
		// assigned above), d.emissionScale reused for its scale (same
		// reuse Lambertian/CoatedDiffuse's own reflectance already uses);
		// d.transmittanceTextureIdx covers transmittance (falls back to
		// d.transmittance just above), d.transmittanceScale (a dedicated
		// field, not reused - see its own comment in optix_types.h) covers
		// its independent scale.
		if (!m.textureFilename.empty()) {
			d.textureIdx = getOrBuildPbrtImageTexture(m.textureFilename, out, imageTextureCache,
				static_cast<float>(m.textureGamma), static_cast<GpuWrapMode>(m.textureWrapIndex), m.textureInvert);
			d.emissionScale = static_cast<float>(m.textureScale);
		}
		if (!m.transmittanceTextureFilename.empty()) {
			d.transmittanceTextureIdx = getOrBuildPbrtImageTexture(m.transmittanceTextureFilename, out, imageTextureCache,
				static_cast<float>(m.transmittanceTextureOptions.gamma),
				static_cast<GpuWrapMode>(m.transmittanceTextureOptions.wrapIndex),
				m.transmittanceTextureOptions.invert);
			d.transmittanceScale = static_cast<float>(m.transmittanceTextureScale);
		}
		break;
	case pbrt_flatten::MaterialKind::CoatedConductor:
		// A recognized named conductor spectrum or an explicit "rgb eta"/
		// "rgb k" (m.hasConductorPreset - see flatten()'s own Conductor-OR-
		// CoatedConductor branch, and pbrt_cpu_builder.h's identical
		// branch) gets the real complex IOR; otherwise the same
		// reflectance-only approximation as before (see
		// reflectanceToConductorK()'s own comment) - eta=1, k solved from
		// the albedo already read above as a normal-incidence reflectance.
		d.type = MaterialType::CoatedConductor;
		if (m.hasConductorPreset) {
			d.eta_c = make_float3(static_cast<float>(m.conductorEta[0]),
								   static_cast<float>(m.conductorEta[1]),
								   static_cast<float>(m.conductorEta[2]));
			d.k_c = make_float3(static_cast<float>(m.conductorK[0]),
								 static_cast<float>(m.conductorK[1]),
								 static_cast<float>(m.conductorK[2]));
		} else {
			d.eta_c = make_float3(1.0f, 1.0f, 1.0f);
			d.k_c = reflectanceToConductorK(d.albedo);
		}
		// Real independent u/v coat roughness - see MaterialType::Conductor's
		// identical-shape override above for the full rationale.
		d.roughness  = static_cast<float>(m.roughness_u);
		d.roughnessV = static_cast<float>(m.roughness_v);
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
		// material without a texture. d.emissionScale (reused, not a
		// dedicated field - see its own comment in optix_types.h and
		// CoatedDiffuse's identical-shape case below) carries a "scale"-
		// class wrapper's multiplier (barcelona-pavilion's own dominant
		// pattern); stays at its 1.0 no-op default for a bare imagemap.
		d.type = MaterialType::Lambertian;
		if (!m.textureFilename.empty()) {
			d.textureIdx = getOrBuildPbrtImageTexture(m.textureFilename, out, imageTextureCache,
				static_cast<float>(m.textureGamma), static_cast<GpuWrapMode>(m.textureWrapIndex), m.textureInvert);
			d.emissionScale = static_cast<float>(m.textureScale);
		}
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
			// One-level-nested bare imagemap tex1/tex2 (Material::
			// checkerTex1Filename/checkerTex2Filename's own comment) -
			// tex1ImageIdx/tex2ImageIdx stay -1 (color1/color2 used
			// directly) when that slot was a flat literal instead.
			if (!m.checkerTex1Filename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.checkerTex1Filename, out, imageTextureCache);
			if (!m.checkerTex2Filename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.checkerTex2Filename, out, imageTextureCache);
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
			// Same one-level-nested-imagemap support as UVChecker above
			// (Material::mixTex1Filename/mixTex2Filename's own comment).
			if (!m.mixTex1Filename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.mixTex1Filename, out, imageTextureCache);
			if (!m.mixTex2Filename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.mixTex2Filename, out, imageTextureCache);
			// m.mixAmountTextureFilename (Material::mixAmountTextureFilename's
			// own comment) - a real per-point spatially-varying blend when
			// "amount" itself nested a bare imagemap; tex.mixAmount (above)
			// stays at its resolved flat value and is unused by
			// sample_texture()/wf_sample_texture() when amountImageIdx >= 0.
			if (!m.mixAmountTextureFilename.empty())
				tex.amountImageIdx = getOrBuildPbrtImageTexture(m.mixAmountTextureFilename, out, imageTextureCache);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasWindyReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Windy;
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasWrinkledReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Wrinkled;
			tex.omega = static_cast<float>(m.wrinkledRoughness);
			tex.octaves = m.wrinkledOctaves;
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasDotsReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Dots;
			tex.color1 = make_float3(static_cast<float>(m.dotsInsideColor[0]),
									 static_cast<float>(m.dotsInsideColor[1]),
									 static_cast<float>(m.dotsInsideColor[2]));
			tex.color2 = make_float3(static_cast<float>(m.dotsOutsideColor[0]),
									 static_cast<float>(m.dotsOutsideColor[1]),
									 static_cast<float>(m.dotsOutsideColor[2]));
			if (!m.dotsInsideTexFilename.empty())
				tex.tex1ImageIdx = getOrBuildPbrtImageTexture(m.dotsInsideTexFilename, out, imageTextureCache);
			if (!m.dotsOutsideTexFilename.empty())
				tex.tex2ImageIdx = getOrBuildPbrtImageTexture(m.dotsOutsideTexFilename, out, imageTextureCache);
			d.textureIdx = static_cast<int>(out.textures.size());
			out.textures.push_back(tex);
		}
		else if (m.hasBilerpReflectance) {
			TextureData tex{};
			tex.kind = TextureKind::Bilerp;
			tex.color1 = make_float3(static_cast<float>(m.bilerpV00[0]),
									 static_cast<float>(m.bilerpV00[1]),
									 static_cast<float>(m.bilerpV00[2]));
			tex.color2 = make_float3(static_cast<float>(m.bilerpV01[0]),
									 static_cast<float>(m.bilerpV01[1]),
									 static_cast<float>(m.bilerpV01[2]));
			tex.bilerpV10 = make_float3(static_cast<float>(m.bilerpV10[0]),
										static_cast<float>(m.bilerpV10[1]),
										static_cast<float>(m.bilerpV10[2]));
			tex.bilerpV11 = make_float3(static_cast<float>(m.bilerpV11[0]),
										static_cast<float>(m.bilerpV11[1]),
										static_cast<float>(m.bilerpV11[2]));
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
			// Falls back to the same flat-colour-average resolveMixColor()
			// already computes - passing `m` itself (not its sub-materials)
			// at depth 0 reproduces the exact same two-call computation this
			// branch used to duplicate inline: resolveMixColor() checks
			// m.kind==Mix and the depth/resolvability conditions itself
			// (identical to `resolvable` above), then recurses into
			// mixMaterialA/B at depth+1 - matching the old code's own two
			// resolveMixColor(..., 1) calls exactly. Also correctly handles
			// the unresolvable case with no separate guard needed:
			// resolveMixColor() falls back to m.color, the same default
			// d.albedo already holds from the generic assignment above.
			d.type = MaterialType::Lambertian;
			d.albedo = resolveMixColor(m, allMaterials, 0);
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
			// Real per-voxel "rgb Le"/"float Lescale" (pbrt-v4 RGBGridMedium::
			// LeGrid/LeScale) - previously silently dropped entirely on GPU
			// (both backends), unlike CPU's real RGBGridMediumData::Le_grids
			// support (rgb_grid_medium_hittable.h). Same is_emissive() gate
			// as CPU (RGBGridMediumData::is_emissive(): Le_grids present AND
			// Le_scale > 0) - md.Le_scale defaults to 1.0 when "Lescale" was
			// omitted (pbrt_flatten.h's own comment on that default), so a
			// scene with only "rgb Le" and no "Lescale" still glows here too.
			meta.leDataOffset = -1;
			meta.Le_scale = static_cast<float>(md.Le_scale);
			const bool hasLe = md.Le_r.size() == voxels && md.Le_g.size() == voxels
				&& md.Le_b.size() == voxels;
			if (hasLe && meta.Le_scale > 0.0f) {
				// Inserted directly from md.Le_r/g/b (double) into
				// out.rgbGridData (float) - unlike the sigma_s block above,
				// which needs its own rf/gf/bf float copies to compute
				// max_density from afterward, ler/leg/leb here had no second
				// use, so the intermediate copy is skipped; insert() narrows
				// double->float per element the same way the copy would have.
				meta.leDataOffset = static_cast<int>(out.rgbGridData.size());
				out.rgbGridData.insert(out.rgbGridData.end(), md.Le_r.begin(), md.Le_r.end());
				out.rgbGridData.insert(out.rgbGridData.end(), md.Le_g.begin(), md.Le_g.end());
				out.rgbGridData.insert(out.rgbGridData.end(), md.Le_b.begin(), md.Le_b.end());
			}
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
		// MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4) - same
		// sigma_a/sigma_t collision-probability weighting as CPU's
		// constant_medium constructor (src/TheRestOfYourLife/constant_
		// medium.h) applies to the identical raw md.Le, so the two backends
		// agree on the exact weighted value, not just which color to use.
		const float leWeight = (d.sigma_t > 1e-9f) ? static_cast<float>(sig_a) / d.sigma_t : 0.0f;
		d.medium_emission = f3(md.Le) * leWeight;
		const int idx = static_cast<int>(out.materials.size());
		out.materials.push_back(d);
		mediumCache.emplace(medIdx, idx);
		return idx;
	};

	// Shared by clipped spheres (below) and disk/cylinder (further down,
	// which originally declared these) - a generic "flatten a row-major 4x4
	// affine to the 3x4 device convention" helper and the degrees->radians
	// constant every pbrt-v4 shape's own "phimax"-style angle param needs
	// converted once, host-side, before reaching the GPU.
	const auto flattenTransform = [](const double xform[16], float out12[12]) {
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 4; ++col)
				out12[row * 4 + col] = static_cast<float>(xform[row * 4 + col]);
	};
	const double kDiskCylDegToRad = 3.14159265358979323846 / 180.0;

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : scene.spheres) {
		if (s.clipped) {
			// Real zmin/zmax/phimax clipping (pbrt_flatten::Sphere::clipped's
			// own comment on the history of this gap) - a clipped sphere
			// carries its own object<->world transform (like Disk/Cylinder)
			// rather than being baked to world-space center/radius the way
			// every OTHER sphere in this loop is, since clipping breaks
			// rotation-invariance. medium/motion-blur are not combined with
			// clipping (pbrt_flatten::Sphere::cpuMediumUnsupported's own
			// comment on the medium side; motion blur was never threaded
			// into the clipped path on either backend).
			pbrt_scene::Matrix4 o2w;
			for (int i = 0; i < 16; ++i) o2w.m[i] = s.xform[i];
			pbrt_scene::Matrix4 w2o;
			if (!o2w.inverseAffine(w2o)) continue;  // singular - drop, matching sphere_clipped_hittable's own valid_ guard

			SphereData sd = {};
			sd.shapeKind = GpuMediumShapeKind::ClippedSphere;
			sd.materialIdx = materialIndex(s.material, s.areaLight);
			sd.radiusLocal = static_cast<float>(s.radiusLocal);
			sd.zMin = static_cast<float>(s.zMin);
			sd.zMax = static_cast<float>(s.zMax);
			sd.phiMax = static_cast<float>(s.phiMaxDeg * kDiskCylDegToRad);
			flattenTransform(o2w.m, sd.o2w);
			flattenTransform(w2o.m, sd.w2o);
			// v-coordinate normalization bounds - see SphereData::thetaZMin/
			// thetaZMax's own comment for why this is precomputed once here
			// rather than by every closest-hit/NEE-sample call site.
			{
				const double r = s.radiusLocal;
				const double thZMin = std::clamp(r > 0.0 ? s.zMin / r : -1.0, -1.0, 1.0);
				const double thZMax = std::clamp(r > 0.0 ? s.zMax / r : 1.0, -1.0, 1.0);
				sd.thetaZMin = static_cast<float>(std::acos(thZMax));
				sd.thetaZMax = static_cast<float>(std::acos(thZMin));
			}
			// center/center1/radius are otherwise unused for this shapeKind
			// (see GpuMediumShapeKind::ClippedSphere's own comment) - populated
			// here anyway, from the same world-space center + baked "largest
			// axis" radius pbrt_flatten.h already computes for every sphere
			// regardless of clipping, purely so the EXISTING full-sphere NEE
			// machinery (sample_area_light_by_kind, __closesthit__sphere's
			// DiffuseLight solid-angle-pdf branch) has a real cone to sample
			// when this sphere is also an AreaLightSource, without any new
			// clip-aware sampling code. Sampling/weighting over the full
			// (unclipped) cone rather than the true visible cap is the exact
			// same accepted approximation CPU's sphere_clipped_hittable.h
			// already documents for this identical narrow case (extra NEE
			// noise from proposed directions landing on the clipped-away
			// region, not bias - real hit-testing above still only uses
			// radiusLocal/o2w/w2o and correctly refuses those directions).
			sd.center = f3(s.center);
			sd.center1 = sd.center;
			sd.radius = static_cast<float>(s.radius);
			if (s.areaLight >= 0) {
				out.lightIndices.push_back(static_cast<int>(out.spheres.size()));
				out.lightKinds.push_back(GpuLightKind::Sphere);
			}
			out.spheres.push_back(sd);
			continue;
		}
		SphereData sd = {};
		sd.center = f3(s.center);
		// Object motion blur - pbrt_flatten::Sphere::center1 already carries
		// the real end-time centre (defaults to == center for a static
		// sphere), computed once with the same real-inequality gate this
		// backend's own motion detection (sceneHasMotion_) uses. See
		// SphereData's own comment for the rest of the pipeline (native
		// OptiX 2-key motion GAS, per-ray time via optixGetRayTime()) -
		// already built and working, this was the last hardcoded value
		// standing in its way.
		sd.center1 = f3(s.center1);
		sd.radius = static_cast<float>(s.radius);
		// MediumInterface wins the material slot entirely when both it and
		// AreaLightSource are set on the same shape (a valid pbrt idiom - an
		// emitting shell with fog inside) - this GPU builder has no combined
		// material slot for "real dielectric/light surface AND a medium"
		// (see mediumMaterialIndex()'s own comment), so the emissive
		// material this sphere would otherwise have built is discarded. Only
		// register as an NEE-samplable light when that emissive material
		// actually survives - otherwise NEE would aim at a real geometric
		// sphere whose resolved MaterialData has zero emission (a wasted,
		// contribution-free sample every time it's selected), and the alias
		// table's own power-weighting would keep it alive at a tiny
		// geometry-only floor forever. CPU has no such gap (it keeps the
		// shape's own emissive material AND layers a separate medium-wrapped
		// hittable on top - see pbrt_cpu_builder.h's addMediumIfPresent), so
		// this is a real, accepted GPU-only divergence for this specific
		// combination, not a bug this loop can fix without a combined
		// material slot (out of scope here - same "more than this phase's
		// scope" reasoning mediumMaterialIndex()'s own comment already gives
		// for the dielectric+fog case).
		const bool sphereHasMedium = s.medium >= 0 && static_cast<std::size_t>(s.medium) < scene.media.size();
		sd.materialIdx = sphereHasMedium
			? mediumMaterialIndex(s.medium)
			: materialIndex(s.material, s.areaLight);
		if (s.areaLight >= 0 && !sphereHasMedium) {
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
	// than being baked to world space the way Sphere is. Now registered as
	// real NEE-samplable lights (GpuLightKind::Disk/Cylinder) when
	// areaLight >= 0, same as every other shape kind - see optix_types.h's
	// GpuLightKind::Disk/Cylinder comment and optix_disk_cylinder_helpers.h
	// for the sampling/pdf machinery this needed. Cylinder's medium field
	// (c.medium) now also resolves via mediumMaterialIndex() (below), same as
	// the sphere loop above - see that call site's own comment for the exact
	// scope (homogeneous MaterialType::Medium only gets real near/far support
	// on cylinder; cloud/rgbgrid/uniformgrid stay correctly trapped). Disk's
	// medium field (d.medium) is intentionally still unread - see
	// docs/PBRT_SUPPORT.md: a zero-thickness plane has no "inside" volume for
	// a homogeneous medium's entry/exit pair, so wrapping one in
	// MediumInterface is structurally not meaningful and is not planned.
	// flattenTransform/kDiskCylDegToRad are declared earlier now, shared
	// with the clipped-sphere loop above (see that loop's own comment).

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
		if (d.areaLight >= 0) {
			out.lightIndices.push_back(static_cast<int>(out.disks.size()));
			out.lightKinds.push_back(GpuLightKind::Disk);
		}
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
		// MediumInterface on a cylinder - unlike disk (structurally not
		// meaningful, see this loop's own comment above), a cylinder's
		// intersection is already a two-root quadratic (like sphere's), so
		// device shading can recompute real entry/exit roots the same way
		// sphere's own Medium case does - see __closesthit__cylinder's own
		// comment (optix_intersection_disk_cylinder.h) and __closesthit__wf_
		// cylinder's (wavefront_programs.cu). mediumMaterialIndex() itself is
		// shape-agnostic (also handles cloud/rgbgrid/uniformgrid media, not
		// just homogeneous) - only the homogeneous MaterialType::Medium case
		// gets real near/far support on cylinder in this pass, so a
		// cloud/rgbgrid/uniformgrid MediumInterface on a cylinder still
		// correctly traps (material_requires_sphere_only_handling()) rather
		// than silently misrendering, exactly like every other still-
		// sphere-only type does on any non-sphere shape.
		// Same "medium wins the material slot, so don't register a now-non-
		// emissive shape as an NEE light" fix as the sphere loop above - see
		// its own comment for the full reasoning (CPU has no such gap; this
		// is an accepted GPU-only divergence for MediumInterface + real
		// emission on the same shape, not fixable here without a combined
		// material slot).
		const bool cylinderHasMedium = c.medium >= 0 && static_cast<std::size_t>(c.medium) < scene.media.size();
		cd.materialIdx = cylinderHasMedium
			? mediumMaterialIndex(c.medium)
			: materialIndex(c.material, c.areaLight);
		flattenTransform(o2w.m, cd.o2w);
		flattenTransform(w2o.m, cd.w2o);
		if (c.areaLight >= 0 && !cylinderHasMedium) {
			out.lightIndices.push_back(static_cast<int>(out.cylinders.size()));
			out.lightKinds.push_back(GpuLightKind::Cylinder);
		}
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
			// Object motion blur - same fix as the top-level sphere loop above
			// (see its own comment): pbrt_flatten.h already bakes a real,
			// per-ray-time-independent center1 for THIS instance definition's
			// own spheres too (its ShapeWork loop runs uniformly over
			// out.groups[g].spheres, not just top-level out.spheres), so
			// hardcoding center1=center here silently dropped motion blur for
			// any sphere declared inside ObjectBegin/ObjectEnd - a real
			// CPU/GPU divergence, since pbrt_cpu_builder.h's single shared
			// emitGeometry lambda already handles this case correctly.
			sd.center1 = f3(s.center1);
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
		const bool hasImage = sky.imageWidth > 0 && sky.imageHeight > 0 && !sky.imagePixels.empty();

		// Mean of every decoded pixel, scaled - shared by the portal and
		// plain-sky branches below since both need it: the plain-sky branch
		// uses it as a real approximate flat-colour fallback (no directional
		// detail at all, unlike the CPU path's real importance-sampled image
		// - see pbrt_flatten.h's InfiniteLight comment for why that stays
		// CPU-only), while the portal branch uses it PURELY to satisfy the
		// "any infinite light present" gate every NEE/miss call site on both
		// GPU backends checks via "skyColor/backgroundColor != 0" BEFORE
		// even looking at whether a real distribution (sky OR portal) is
		// present (see e.g. optix_device_helpers.h's Lambertian sky-NEE
		// block) - gpu_portal_Le() never actually reads this value as
		// radiance, so any nonzero placeholder works, but a leftover zero
		// here would make the portal light completely unreachable despite
		// portalWidth/Height being set. Floored to a small nonzero epsilon
		// rather than left at whatever the raw mean computes to, for exactly
		// that reason: a portal (or plain sky) image whose overall content
		// happens to average out near-black (a real, unremarkable
		// composition - e.g. a mostly-dark photo/exr with one bright window)
		// must not silently zero out this gate and disable the entire light
		// with no diagnostic.
		const auto meanPixelColor = [&]() -> float3 {
			double r = 0.0, g = 0.0, b = 0.0;
			const std::size_t n = static_cast<std::size_t>(sky.imageWidth) * sky.imageHeight;
			for (std::size_t i = 0; i < n; ++i) {
				r += sky.imagePixels[i * 3 + 0];
				g += sky.imagePixels[i * 3 + 1];
				b += sky.imagePixels[i * 3 + 2];
			}
			constexpr float kMinGateBrightness = 1e-4f;
			return make_float3(
				std::max(kMinGateBrightness, static_cast<float>(r / n * sky.scale)),
				std::max(kMinGateBrightness, static_cast<float>(g / n * sky.scale)),
				std::max(kMinGateBrightness, static_cast<float>(b / n * sky.scale)));
		};

		if (sky.hasPortal) {
			// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's
			// own comment (optix_types.h). Mirrors src/TheRestOfYourLife/
			// pbrt_cpu_builder.h's own portal-construction block exactly,
			// including its "fails closed" behavior: hasPortal but no valid
			// image means no light at all (not even the flat-colour fallback
			// the plain-sky branch below would take) - matches CPU's own
			// BuildResult::portal staying null with no out.sky fallback either.
			if (hasImage) {
				std::array<pil_detail::Vec3<double>, 4> corners;
				for (int i = 0; i < 4; ++i) {
					corners[i] = pil_detail::Vec3<double>(
						sky.portal[i*3+0], sky.portal[i*3+1], sky.portal[i*3+2]);
				}
				// Real host-side construction (equal-area rectification + SAT
				// build) - see PortalImageInfiniteLightData::rectified()/
				// distribution()'s own comment (portal_image_infinite_light.h)
				// for why this is a real instance of the exact class CPU uses,
				// not a from-scratch GPU reimplementation of the same math.
				PortalImageInfiniteLightData<double> portalData(
					sky.imagePixels.data(), sky.imageWidth, sky.imageHeight, sky.scale, corners);

				out.portalRectifiedImage = portalData.rectified();
				const Array2D<float>& func = portalData.distribution().func();
				out.portalDistFunc = func.data();
				out.portalSatSum = portalData.distribution().sat().sum().data();
				out.portalWidth = portalData.width();
				out.portalHeight = portalData.height();
				out.portalScale = static_cast<float>(portalData.scale());

				// Portal frame - mirrors PortalImageInfiniteLightData's own ctor
				// (FrameT::FromXY(p03, p01), p01=normalize(portal[1]-portal[0]),
				// p03=normalize(portal[3]-portal[0])) in float rather than
				// double, recomputed here rather than exposing the CPU class's
				// Frame<double> member directly (this codebase's GPU structs are
				// float3-based throughout, see optix_types.h). Frame::FromXY's
				// own algorithm (vec3_frame.h): nx=normalize(vx),
				// nz=normalize(cross(nx,vy)), ny=cross(nz,nx).
				auto toF3 = [&](int i) {
					return make_float3(static_cast<float>(sky.portal[i*3+0]),
										static_cast<float>(sky.portal[i*3+1]),
										static_cast<float>(sky.portal[i*3+2]));
				};
				const float3 p0f = toF3(0), p1f = toF3(1), p2f = toF3(2), p3f = toF3(3);
				const float3 p01 = normalize(p1f - p0f);
				const float3 p03 = normalize(p3f - p0f);
				const float3 nx = normalize(p03);
				const float3 nz = normalize(cross(nx, p01));
				const float3 ny = cross(nz, nx);
				out.portalFrameX = nx; out.portalFrameY = ny; out.portalFrameZ = nz;
				out.portalP0 = p0f; out.portalP2 = p2f;

				stats.backgroundColor = meanPixelColor();
			}
		} else if (hasImage) {
			stats.backgroundColor = meanPixelColor();

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
			// pl.filename is only ever non-empty after pbrt_load.h's post-
			// flatten pass confirmed the file exists - see
			// pbrt_cpu_builder.h's identical decode for the CPU side and its
			// own comment on the square-image requirement / greyscale
			// collapse. Downsampled (nearest-neighbor) to kGonioImageMaxDim
			// when larger, rather than falling back to uniform, so a real
			// profile still shows on GPU even if this build's fixed inline
			// cap can't hold it at full resolution.
			bool usedRealProfile = false;
			if (!pl.filename.empty()) {
				int width = 0, height = 0, channels = 0;
				float *fdata = stbi_loadf(pl.filename.c_str(), &width, &height, &channels, 3);
				if (fdata && width > 0 && width == height) {
					const int n = std::min(width, kGonioImageMaxDim);
					g.nu = n; g.nv = n;
					for (int v = 0; v < n; ++v) {
						const int sv = v * height / n;
						for (int u = 0; u < n; ++u) {
							const int su = u * width / n;
							const float *px = fdata + (static_cast<std::size_t>(sv) * width + su) * 3;
							g.image[v * n + u] = (px[0] + px[1] + px[2]) / 3.0f;
						}
					}
					usedRealProfile = true;
				}
				if (fdata) stbi_image_free(fdata);
			}
			if (!usedRealProfile) {
				// Uniform (isotropic) fallback - matches
				// pbrt_cpu_builder.h's own kUniformImage.
				g.nu = 4; g.nv = 4;
				for (int i = 0; i < g.nu * g.nv; ++i) g.image[i] = 1.0f;
			}
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
			// pl.filename is only ever non-empty after pbrt_load.h's post-
			// flatten pass confirmed the file exists. Downsampled (nearest-
			// neighbor) to kProjImageMaxDim when larger - see the identical
			// reasoning on the Goniometric case just above.
			bool usedRealSlide = false;
			if (!pl.filename.empty()) {
				int width = 0, height = 0, channels = 0;
				float *fdata = stbi_loadf(pl.filename.c_str(), &width, &height, &channels, 3);
				if (fdata && width > 0 && height > 0) {
					// Cap the LARGER dimension to kProjImageMaxDim and scale
					// the other one proportionally, not each dimension
					// independently - independent per-axis caps would distort
					// a non-square image's aspect ratio whenever only one
					// axis exceeds the cap (e.g. a 1920x1080 photo would
					// become 64x64, aspect 1.0 instead of ~1.78), which then
					// corrupts the screen-bounds aspect computed below and
					// visibly squishes the projected content relative to
					// CPU's own uncapped (always-correct-aspect) version.
					const float fitScale = std::min(1.0f,
						static_cast<float>(kProjImageMaxDim) / static_cast<float>(std::max(width, height)));
					const int nx = std::max(1, static_cast<int>(width * fitScale + 0.5f));
					const int ny = std::max(1, static_cast<int>(height * fitScale + 0.5f));
					pr.nx = nx; pr.ny = ny;
					for (int v = 0; v < ny; ++v) {
						const int sv = v * height / ny;
						for (int u = 0; u < nx; ++u) {
							const int su = u * width / nx;
							const float *px = fdata + (static_cast<std::size_t>(sv) * width + su) * 3;
							const int i = (v * nx + u) * 3;
							pr.image_rgb[i + 0] = px[0];
							pr.image_rgb[i + 1] = px[1];
							pr.image_rgb[i + 2] = px[2];
						}
					}
					usedRealSlide = true;
				}
				if (fdata) stbi_image_free(fdata);
			}
			if (!usedRealSlide) {
				// Uniform white 2x2 slide - matches pbrt_cpu_builder.h's own
				// kUniformSlide (2x2, aspect 1 so sb_xmin/xmax below match
				// ProjectionLight<T>::make's own aspect>=1 branch exactly).
				pr.nx = 2; pr.ny = 2;
				for (int i = 0; i < pr.nx * pr.ny * 3; ++i) pr.image_rgb[i] = 1.0f;
			}
			// Screen bounds depend on the FINAL nx/ny's aspect ratio (matches
			// ProjectionLight<T>::make's own aspect-derived bounds), so this
			// has to run after nx/ny are settled above, real or fallback.
			{
				const float aspect = static_cast<float>(pr.nx) / static_cast<float>(pr.ny);
				if (aspect >= 1.0f) {
					pr.sb_xmin = -aspect; pr.sb_xmax = aspect;
					pr.sb_ymin = -1.0f;   pr.sb_ymax = 1.0f;
				} else {
					pr.sb_xmin = -1.0f;         pr.sb_xmax = 1.0f;
					pr.sb_ymin = -1.0f / aspect; pr.sb_ymax = 1.0f / aspect;
				}
			}
			pr.inv_tan = 1.0f / static_cast<float>(std::tan(pl.fovDeg * kDegToRad * 0.5));
			break;
		}
		}
		out.punctualLights.push_back(light);
	}

	return stats;
}

} // namespace pbrt_gpu
