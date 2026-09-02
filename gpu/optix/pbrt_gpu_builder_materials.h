// pbrt_gpu_builder_materials.h -- texture/material resolution helpers
// (namespace pbrt_gpu::detail) for the pbrt-v4 GPU scene builder.
//
// Split out of pbrt_gpu_builder.h, which #includes this file directly
// (textually, still inside its own `namespace pbrt_gpu { ... }` block -
// this file opens/closes ONLY the inner `namespace detail`, and must never
// be included from outside that outer namespace). makeMaterial() alone -
// the real work of resolving a pbrt_flatten::Material into a GPU
// MaterialData, including every texture-slot/BSSRDF-preset/measured-BxDF
// special case - is close to half this file on its own; the rest are its
// own small supporting helpers (resolveMixColor, reflectanceToConductorK,
// texture-index resolution). build()'s own shape/light-building loops
// stayed in pbrt_gpu_builder.h - see that file's own header comment.


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

// Resolves whichever procedural (non-file) reflectance texture kind `m` has
// set - checkerboard/FBm/marble/mix/windy/wrinkled/dots/bilerp, in the same
// priority order as makeMaterial()'s own imagemap-then-procedural chain -
// into a new TextureData entry appended to `out.textures`, pointing
// `d.textureIdx` at it. Shared between the CoatedDiffuse and Diffuse/
// Lambertian cases in makeMaterial() below, which previously hand-duplicated
// this entire chain twice (same "no cache/dedup needed" rationale as
// getOrBuildPbrtImageTexture()'s own call sites - each pbrt Texture
// declaration is already deduped 1:1 with the Material referencing it by
// materialCache in build()). Returns true if a procedural kind matched
// (d.textureIdx was set); false leaves `d` untouched so the caller's own
// imagemap-filename/flat-color fallback still applies.
inline bool resolveProceduralReflectanceTexture(const pbrt_flatten::Material &m, MaterialData &d,
												 SceneData &out, std::map<std::string, int> &imageTextureCache) {
	if (m.hasCheckerReflectance) {
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
		return true;
	}
	if (m.hasFbmReflectance) {
		TextureData tex{};
		tex.kind = TextureKind::FBm;
		tex.omega = static_cast<float>(m.fbmRoughness);
		tex.octaves = m.fbmOctaves;
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	if (m.hasMarbleReflectance) {
		TextureData tex{};
		tex.kind = TextureKind::Marble;
		tex.omega = static_cast<float>(m.marbleRoughness);
		tex.octaves = m.marbleOctaves;
		tex.marbleScale = static_cast<float>(m.marbleScale);
		tex.marbleVariation = static_cast<float>(m.marbleVariation);
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	if (m.hasMixReflectance) {
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
		if (!m.mixAmountTextureFilename.empty())
			tex.amountImageIdx = getOrBuildPbrtImageTexture(m.mixAmountTextureFilename, out, imageTextureCache);
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	if (m.hasWindyReflectance) {
		TextureData tex{};
		tex.kind = TextureKind::Windy;
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	if (m.hasWrinkledReflectance) {
		TextureData tex{};
		tex.kind = TextureKind::Wrinkled;
		tex.omega = static_cast<float>(m.wrinkledRoughness);
		tex.octaves = m.wrinkledOctaves;
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	if (m.hasDotsReflectance) {
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
		return true;
	}
	if (m.hasBilerpReflectance) {
		TextureData tex{};
		tex.kind = TextureKind::Bilerp;
		tex.color1 = make_float3(static_cast<float>(m.bilerpV00[0]),
								 static_cast<float>(m.bilerpV00[1]),
								 static_cast<float>(m.bilerpV00[2]));
		tex.color2 = make_float3(static_cast<float>(m.bilerpV01[0]),
								 static_cast<float>(m.bilerpV01[1]),
								 static_cast<float>(m.bilerpV01[2]));
		// v10/v11 packed into otherwise-unused fields rather than two more
		// float3s - see TextureKind::Bilerp's own comment (optix_types.h).
		tex.uScale = static_cast<float>(m.bilerpV10[0]);
		tex.vScale = static_cast<float>(m.bilerpV10[1]);
		tex.omega  = static_cast<float>(m.bilerpV10[2]);
		tex.marbleScale     = static_cast<float>(m.bilerpV11[0]);
		tex.marbleVariation = static_cast<float>(m.bilerpV11[1]);
		tex.mixAmount       = static_cast<float>(m.bilerpV11[2]);
		d.textureIdx = static_cast<int>(out.textures.size());
		out.textures.push_back(tex);
		return true;
	}
	return false;
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
		// hasMixReflectance/hasWindyReflectance/hasWrinkledReflectance/
		// hasDotsReflectance/hasBilerpReflectance (Material's own comments) -
		// same procedural-not-file, append-one-TextureData pattern as the
		// Diffuse case below, now also resolved for CoatedDiffuse (previously
		// Diffuse-only). sample_texture() (optix_device_helpers.h) dispatches
		// purely on TextureData::kind, not on MaterialType, so pointing
		// d.textureIdx at a procedural entry works identically for
		// CoatedDiffuse's own "reflectance" read as it already does for
		// Lambertian's - shared via resolveProceduralReflectanceTexture()
		// above rather than duplicated here (matches the Diffuse case's own
		// call to the same helper).
		else resolveProceduralReflectanceTexture(m, d, out, imageTextureCache);
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
		// m.hasCheckerReflectance/hasFbmReflectance/hasMarbleReflectance/
		// hasMixReflectance/hasWindyReflectance/hasWrinkledReflectance/
		// hasDotsReflectance/hasBilerpReflectance (Material's own comments) -
		// procedural pbrt-v4 textures, appended directly to out.textures (no
		// cache/dedup: each pbrt Texture declaration is already deduped 1:1
		// with the Material referencing it by materialCache in build(), so
		// this runs at most once per distinct procedural material) - shared
		// via resolveProceduralReflectanceTexture() above rather than
		// duplicated here (matches the CoatedDiffuse case's own call to the
		// same helper).
		else resolveProceduralReflectanceTexture(m, d, out, imageTextureCache);
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
