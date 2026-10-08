#pragma once
// optix_types_material.h -- part 3 of 5 of optix_types.h (included by it, in order; not meant to be included on its own).

struct TextureData {
	TextureKind kind;
	int pixelOffset;   // Image: byte offset into texturePixels. Unused otherwise.
	int width;         // Image: pixel width. Unused otherwise.
	int height;        // Image: pixel height. Unused otherwise.
	// Image (and UVChecker/Mix's own tex1ImageIdx/tex2ImageIdx - see their
	// own comment below, both of which resolve to another Image-kind
	// TextureData and share sampleImage()'s one lookup path) only - see
	// GpuWrapMode's own comment above for what this does and why every
	// OTHER Image entry (checker/mix/roughness/transmittance/displacement)
	// stays at the zero-init Clamp default regardless of what the scene's
	// own imagemap "wrap" declares.
	GpuWrapMode wrapMode = GpuWrapMode::Clamp;
	// Image only: texturePixels holds the file's own sRGB-encoded bytes, decoded per texel at lookup
	// (texel_rgb above). False (the zero-init default) for every linear texture: normal maps, alpha
	// masks, inverted images, HDR-derived data, native scenes.
	bool srgb = false;
	float noiseScale;  // Noise: scale param. Checker: 1/scale (checker_texture's own inv_scale). Unused otherwise.
	float3 color1;     // Checker/UVChecker: "even"/tex1 cell color. Mix: tex1. Unused otherwise.
	float3 color2;     // Checker/UVChecker: "odd"/tex2 cell color. Mix: tex2. Unused otherwise.
	// UVChecker: u-axis tile frequency (pbrt-v4 "uscale"). Bilerp: v10.x
	// (the 3rd/4th corner colours are packed into these otherwise-unused
	// fields instead of two new float3s - see TextureKind::Bilerp's own
	// comment; deliberately not a union, since CUDA/MSVC/GCC/Clang all
	// agree on plain sequential layout for these but an anonymous
	// struct-in-union spanning them would be non-standard). Unused by
	// every other kind.
	float uScale;
	float vScale;      // UVChecker: v-axis tile frequency (pbrt-v4 "vscale"). Bilerp: v10.y. Unused otherwise.
	float omega;       // FBm/Marble: persistence/roughness param. Bilerp: v10.z. Unused otherwise.
	int   octaves;     // FBm/Marble: octave count. Unused otherwise.
	float marbleScale;     // Marble: spatial frequency multiplier (pbrt-v4 "scale"). Bilerp: v11.x. Unused otherwise.
	float marbleVariation; // Marble: FBm displacement amplitude (pbrt-v4 "variation"). Bilerp: v11.y. Unused otherwise.
	float mixAmount;   // Mix: blend weight, 0->color1 1->color2 (pbrt-v4 "amount"). Bilerp: v11.z. Unused otherwise.
	// UVChecker/Mix only (NOT the world-space Checker kind, which has no
	// loader path that ever sets these): index into LaunchParams::textures
	// for a ONE-LEVEL-nested bare imagemap Texture bound to tex1/tex2 instead of a
	// flat literal (pbrt_flatten::Material::checkerTex1Filename/
	// checkerTex2Filename/mixTex1Filename/mixTex2Filename's own comments),
	// or -1 (the default, via `TextureData tex{};`'s aggregate-init - no
	// existing call site needed updating) to use color1/color2 directly.
	// Not itself recursive - a texture kind other than Image at this index
	// is never produced by the loader (see flatten()'s own one-level scope
	// cut), so no cycle/depth guarding is needed on GPU.
	int tex1ImageIdx = -1;
	int tex2ImageIdx = -1;
	// Mix only: index into LaunchParams::textures for a ONE-LEVEL-nested
	// bare imagemap Texture bound to "amount" instead of the flat mixAmount
	// scalar above (pbrt_flatten::Material::mixAmountTextureFilename's own
	// comment), or -1 (the default) to use mixAmount directly. Same
	// one-level, non-recursive scope as tex1ImageIdx/tex2ImageIdx.
	int amountImageIdx = -1;
};

// Material data (packed for SBT).
//
// This struct is reused across all 16 MaterialTypes (see the field-reuse
// notes already on each MaterialType enumerator above), which used to mean
// every field's *meaning* depended entirely on which MaterialType was set
// on the same MaterialData instance - e.g. plain `.fuzz` meant Metal
// roughness, RoughDielectric/Conductor/CoatedDiffuse/CoatedConductor/
// Principled roughness, Medium/DielectricMedium's Henyey-Greenstein
// asymmetry g, or Hair's beta_m, entirely by convention/comment.
//
// Each field below is now an anonymous union of same-size alternatives,
// one name per material family that owns that storage - purely a set of
// alternate *names* for the exact same bytes (standard C++11 anonymous
// unions), not a layout change: sizeof/alignment/positional-brace-init
// order are all unchanged (the first-listed name in each union is what a
// positional aggregate-init like `{ MaterialType::Metal, albedo, fuzz,
// 0.0f, ... }` still binds to), and every pre-existing field name (.fuzz,
// .ior, .eta_c, ...) remains valid and reads/writes the identical bytes -
// this is why no reading code (shade_material(), wavefront_kernels.cu's
// parallel switch, optix_intersection_sphere.h's special-cased types) had
// to change: they still compile and run byte-for-byte identically. Only
// scene_builder.cpp's add_<type>() material factories (the sole writers,
// after the Phase 1 refactor that funneled all ~190 material-creation call
// sites through them) were updated to use the clearer names.
struct MaterialData {
	MaterialType type;

	union {
		float3 albedo;         // Lambertian/Metal/RoughDielectric(unused)/CoatedDiffuse: diffuse/base color
		float3 sigma_a;        // Hair: RGB absorption coefficient
		float3 medium_albedo;  // Medium/DielectricMedium: single-scatter color
		float3 reflectance;    // DiffuseTransmission: R (reflected diffuse color)
		float3 base_color;     // Principled: base color
		// Dielectric: transmission_filter (OBJ/.mtl "Tf") - multiplies the
		// refracted/transmitted contribution only, leaving reflection
		// clear/white. Defaults to (1,1,1) (no-op) via add_dielectric()'s
		// default param - see mesh.h's CPU dielectric class for the full
		// rationale (a flat per-surface tint, not volumetric absorption).
		float3 transmission_filter;
	};

	union {
		float fuzz;       // Legacy/generic name - still what most reading code uses
		float roughness;  // Metal/RoughDielectric/Conductor/CoatedDiffuse/CoatedConductor/Principled:
		                  //   authored roughness (RoughnessToAlpha conversion done in shading code)
		float g;          // Medium/DielectricMedium: Henyey-Greenstein phase asymmetry
		float beta_m;     // Hair: longitudinal roughness
	};

	union {
		float ior;      // Dielectric/RoughDielectric/ThinDielectric/CoatedDiffuse(coat)/
		                //   CoatedConductor(coat)/NormalizedFresnel/DielectricMedium/Principled: index of refraction
		float sigma_t;  // Medium: extinction coefficient
		float eta;      // Hair: fiber index of refraction
	};

	union {
		float3 emission;       // DiffuseLight: emitted radiance
		float3 transmittance;  // DiffuseTransmission: T (transmitted diffuse color)
		// Subsurface: RGB absorption coefficient (pbrt-v4 sigma_a, already
		// scale-multiplied - see pbrt_flatten::Material::sigma_a's comment).
		// Named bssrdf_sigma_a (not sigma_a - that name is already taken by
		// the OTHER union above, Hair's own RGB absorption field) so both can
		// coexist; shares this slot rather than `albedo` specifically so that
		// `albedo` can stay the flat-gray fallback color the wavefront
		// backend's Subsurface fallback case reads - see MaterialType::
		// Subsurface's own comment in optix_types.h.
		float3 bssrdf_sigma_a;
		// Medium (homogeneous only - see pbrt_flatten::Medium::Le's own
		// comment): MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4),
		// already weighted by sigma_a/sigma_t at build time (pbrt_gpu_
		// builder.h's mediumMaterialIndex()), matching CPU's hg_phase_
		// material::emitted() exactly - see each backend's own Medium
		// closest-hit case (optix_intersection_sphere.h/optix_intersection_
		// disk_cylinder.h/wavefront_kernels.cu) for where this is added,
		// unconditionally, only on a genuine phase-scatter collision (never
		// on the straight-through no-interaction sub-case). Zero (the
		// aggregate-init default) for every Medium built without a nonzero
		// "Le" and for every other material kind that reuses this slot.
		float3 medium_emission;
	};

	// Conductor complex IOR real part is this union's own, unambiguous
	// purpose; Hair/DielectricMedium/Principled each borrow the same 12
	// bytes for their own unrelated per-component data (see each nested
	// struct below - matches the field-reuse already on MaterialType's own
	// enumerator comments exactly).
	union {
		float3 eta_c;  // Conductor/CoatedConductor: real part η per R/G/B channel (pbrt-v4 ConductorBxDF)
		struct { float beta_n, alpha_deg, _hair_pad; } hair_extra;               // Hair
		// `surfaceKind` (0.0f=smooth/default, 1.0f=thin, 2.0f=rough): which
		// BSDF model the entry/exit surface uses instead of always the
		// default smooth Dielectric refraction - see pbrt_gpu_builder.h's
		// mediumMaterialIndex() for how a shape's own Material "dielectric"
		// (rough, via roughness_u/v or a roughness texture) / "thindielectric"
		// sets this. A plain float, not an enum, matching this union's own
		// existing "flag/index stored as a float" idiom (cloud_medium_extra.
		// cloudMediumIdx's own comment just below) so this struct's layout
		// stays uniform. `roughness`: the fused material's own flat/u-axis
		// GGX roughness (pre-remap, same convention as the top-level
		// `roughness` field RoughDielectric uses - stored HERE instead,
		// since DielectricMedium's own copy of that same union slot
		// (fuzz/roughness/g/beta_m, below) is already taken by `g`, the
		// medium's Henyey-Greenstein asymmetry) - meaningful only when
		// surfaceKind==2 (rough); the v-axis roughness reuses the top-level
		// `roughnessV` field directly (genuinely unused by DielectricMedium
		// otherwise, same -1-or-real-value convention RoughDielectric's own
		// reader already has), and `remapRoughness`/`textureIdx` (both
		// top-level, also otherwise unused here) are reused the same way.
		struct { float sigma_t, roughness, surfaceKind; } dielectric_medium_extra; // DielectricMedium
		struct { float metallic, clearcoat, clearcoat_rough; } principled_params; // Principled
		// CloudMedium: index into LaunchParams::cloudMediums (stored as a
		// float so it lives in this union without changing MaterialData's
		// layout - see MaterialType::CloudMedium's comment for why an index
		// rather than direct field reuse).
		struct { float cloudMediumIdx, _cloud_medium_pad1, _cloud_medium_pad2; } cloud_medium_extra;
		// RgbGridMedium: index into LaunchParams::rgbGridMediums - same
		// index-not-direct-field-reuse reasoning as cloud_medium_extra above.
		struct { float rgbGridMediumIdx, _rgb_grid_medium_pad1, _rgb_grid_medium_pad2; } rgb_grid_medium_extra;
		// GridMedium: index into LaunchParams::gridMediums - same
		// index-not-direct-field-reuse reasoning as cloud_medium_extra above.
		struct { float gridMediumIdx, _grid_medium_pad1, _grid_medium_pad2; } grid_medium_extra;
		// Mix: both sub-materials' indices into LaunchParams::materials
		// (stored as floats, same "index in an otherwise-unused union slot"
		// convention as cloudMediumIdx/rgbGridMediumIdx/gridMediumIdx above -
		// cast to int on read) plus the blend weight (pbrt-v4 "amount":
		// probability of B winning at any given shading point, matching CPU's
		// mix_material/pbrt_flatten::Material::mixWeight convention exactly).
		struct { float mixMaterialAIdx, mixMaterialBIdx, mixWeight; } mix_extra;
		// Dielectric/RoughDielectric: two-term Cauchy dispersion coefficients
		// (n(lambda) = cauchy_A + cauchy_B/lambda^2 - see src/shared/fresnel.h's
		// CauchyEta()/CauchyCoefficientsFromAbbe()), matching CPU's already-
		// shipped dielectric::make_dispersive()/rough_dielectric::
		// make_dispersive(). cauchy_A > 0.0f means "this material is
		// dispersive" - a real Cauchy A coefficient is always ~1.4-1.9, so 0
		// (the union's zero-initialized default) is a safe sentinel for "not
		// dispersive, use the flat .ior above", no separate bool needed. Note
		// this is a different shape than every other slot in this union
		// (eta_c/hair_extra/.../mix_extra above): those are selected purely
		// by MaterialType, with no secondary in-union value test - this is
		// the first one where MaterialType alone (Dielectric/RoughDielectric)
		// isn't enough and a runtime float comparison also has to fire. If a
		// future Dielectric/RoughDielectric variant ever needs cauchy_A <= 0
		// to be meaningful, this sentinel needs to become a real tag instead.
		struct { float cauchy_A, cauchy_B, _dispersive_pad; } dispersive_extra;
	};

	union {
		float3 k_c;         // Conductor/CoatedConductor only: imaginary part k per R/G/B channel
		// Subsurface: RGB scattering coefficient (pbrt-v4 sigma_s, already
		// scale-multiplied). k_c has no meaning for Subsurface (not a
		// Conductor), so this slot is free to reuse - same alternate-name
		// pattern as every other union in this struct.
		float3 bssrdf_sigma_s;
	};

	// Index into LaunchParams::textures, or -1 to use `albedo` directly
	// (every material before this field existed omitted it in brace-init,
	// which C++ aggregate-init already defaults to -1 via this default
	// member initializer - not 0 - so no existing call site needed
	// updating). Lambertian: albedo texture. NormalMappedLambertian reuses
	// this as the normal-map texture index instead (see that type's own
	// comment) - unambiguous since CPU never combines that material with a
	// real albedo texture too. Subsurface (recursive backend only) reuses it
	// again, as an index into LaunchParams::bssrdfTables - see
	// MaterialType::Subsurface's own comment for why this reuse is safe
	// (never a real texture for this material kind) and why the wavefront
	// backend's own Subsurface fallback case must NOT read this field as a
	// texture index. RoughDielectric reuses it again, as its own
	// texture-bound "roughness" (pbrt-v4 "texture roughness" on a
	// Dielectric) - >= 0 means sample this texture's red/x channel as the
	// scalar isotropic roughness at each hit instead of reading d.roughness
	// (see each GPU shading kernel's own RoughDielectric case), unambiguous
	// the same way NormalMappedLambertian's/Subsurface's reuse above is
	// (RoughDielectric never has a real albedo texture to conflict with).
	int textureIdx = -1;

	// Index into LaunchParams::textures for an opacity/alpha-cutout mask
	// (map_d in OBJ/.mtl), or -1 for none (the overwhelming majority of
	// materials) - see optix_intersection_triangle.h's __anyhit__triangle
	// and optix_anyhit_shadow.h's __anyhit__shadow_triangle. Independent of
	// textureIdx above (a material can have both a diffuse texture and a
	// separate alpha mask), and applies only to triangles today - no
	// sphere/quad/bilinear-patch scene sets it. Same default-member-
	// initializer trick as textureIdx: every brace-init call site gets -1
	// automatically, no existing call site needs updating.
	int alphaMaskTexIdx = -1;

	// pbrt "texture displacement" bump map with a grayscale height image: index into
	// LaunchParams::textures (-1 = none) and the factor of the "scale" texture wrapping it. The triangle
	// closest-hit programs perturb the shading normal with it (gpu_bump_map.h) for any non-emissive
	// material; other shapes ignore it. Same default-member-initializer trick as alphaMaskTexIdx.
	int   bumpTexIdx = -1;
	float bumpScale  = 1.0f;

	// DiffuseTransmission only: index into LaunchParams::textures for a
	// texture-bound "transmittance", or -1 to use the `transmittance` union
	// field directly - independent of textureIdx above, which this same
	// material kind reuses for its own texture-bound "reflectance"
	// (barcelona-pavilion's foliage binds both to the SAME bare imagemap in
	// practice, so they resolve to the same cached texture index there, but
	// the fields are independent for a scene that binds them differently).
	// Same default-member-initializer trick as textureIdx/alphaMaskTexIdx:
	// every brace-init call site gets -1 automatically, no existing call
	// site needs updating.
	int transmittanceTextureIdx = -1;

	// DiffuseTransmission only, paired with transmittanceTextureIdx above
	// the same way emissionScale (below) pairs with textureIdx for that
	// same material's own reflectance - a "scale"-class wrapper's
	// multiplier applied to the sampled transmittance texel (pbrt_flatten::
	// Material::transmittanceTextureScale's own comment). A separate field
	// from emissionScale rather than reused, since a DiffuseTransmission's
	// reflectance and transmittance can each be wrapped in independently-
	// valued "scale" textures. Defaults to a no-op multiply.
	float transmittanceScale = 1.0f;

	// DiffuseLight only (matches CPU's diffuse_light::is_two_sided()): when
	// true, material_emission()/its wavefront equivalent emit from both
	// faces instead of gating on front_face. false (the default) preserves
	// every pre-existing light's one-sided behavior - no call site needed
	// updating.
	bool twoSided = false;

	// DiffuseLight, Lambertian, or CoatedDiffuse, and only meaningful when
	// textureIdx >= 0: a flat scalar multiplier applied to the sampled texel
	// at lookup time (mirrors CPU's scaled_texture wrapper - see that
	// class's own comment). A flat-color DiffuseLight already bakes "scale"
	// into `emission` directly at build time and never reads this field.
	// Lambertian and CoatedDiffuse each reuse this same field (not a
	// dedicated one) for their own "reflectance" bound to a "scale"-class
	// Texture wrapping an imagemap (barcelona-pavilion's own dominant
	// pattern - see pbrt_flatten::Material::textureScale's own comment) -
	// same "a per-material scalar multiplier over a sampled texture"
	// concept, just read by a different material kind's own shading code.
	// Defaults to a no-op multiply so no existing call site needed updating.
	float emissionScale = 1.0f;

	// pbrt-v4's "remaproughness" (default true): when true, `roughness`
	// above is a perceptually-remapped authored value that still needs
	// RoughnessToAlpha (sqrt) applied on-device before use as real GGX
	// alpha (Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor/
	// RoughMetal); when false, `roughness` already IS the alpha value.
	// Defaults to true (the pre-existing unconditional-sqrt behavior), so
	// no existing call site needed updating. Applies identically to
	// roughnessV below (pbrt-v4 remaps both axes together, never one
	// without the other).
	bool remapRoughness = true;

	// Second GGX alpha axis (the "v"/bitangent-direction roughness),
	// consumed ONLY by the 4 material kinds whose pbrt-v4 BxDF is
	// genuinely anisotropy-capable: RoughDielectric/Conductor/
	// CoatedDiffuse/CoatedConductor (mirrors pbrt_cpu_builder.h's
	// identical roughness_u/roughness_v dispatch - RoughMetal/Metal have
	// no anisotropic variant on CPU either, so they don't read this
	// field). Read by all 3 GPU backends (optix_device_helpers.h,
	// wavefront_kernels.cu), but GPU SPPM (sppm_programs.cu) only
	// implements 2 of the 4 kinds (Conductor/RoughDielectric - CoatedDiffuse/
	// CoatedConductor have no SPPM material case at all, unrelated to
	// anisotropy). Defaults to -1.0f, meaning "no real v-roughness was set -
	// use `roughness` above for both axes" (isotropic); every reading
	// site treats <0 this way rather than a literal zero-roughness
	// v-axis, so every pre-existing brace-init call site (which never
	// mentions this field) keeps its old isotropic behavior unchanged.
	// A NEGATIVE sentinel (not 0.0f) is deliberate: a real, authored
	// v-roughness of exactly 0.0 is a legitimate value (e.g. a scene that
	// sets only "uroughness", leaving "vroughness" to fall back to 0 -
	// meaning "near-mirror in v, rough in u") and real roughness is never
	// negative, so -1.0f can't collide with any value a scene could
	// actually author - unlike a 0.0f sentinel, which would be bitwise
	// indistinguishable from that legitimate zero and silently collapse
	// the material to isotropic instead. Same authored-vs-alpha
	// convention as `roughness` (remapRoughness above applies to both).
	//
	// The local tangent/bitangent frame these 4 material kinds build is
	// now UV/dpdu-aligned on the recursive and wavefront backends, matching
	// CPU's ShadingFrame::from_dpdu exactly - built via BuildDpduTangentFrame()
	// (src/shared/microfacet.h), which projects each shape's real per-shape
	// dpdu (analytic for sphere/disk/cylinder, UV-gradient-solved for
	// triangle, the shape's own edge/patch tangent for quad/bilinear-patch -
	// see each intersection file's own dpdu comment) onto the tangent plane
	// via Gram-Schmidt, falling back to the older arbitrary (non-aligned)
	// frame - BuildArbitraryTangentFrame(), a continuous/branchless
	// construction that replaced an even earlier hard-discontinuous one -
	// only at genuine parametrization poles (e.g. a sphere's dpdu vanishing
	// at theta=0/pi) where dpdu itself is degenerate. So a nonzero
	// roughnessV now produces both the correct anisotropic GGX response
	// AND the correct highlight orientation, matching CPU's render of the
	// same material almost everywhere (RoughMetal is the one exception -
	// it has no anisotropic variant on CPU either, so it's excluded from
	// this and keeps its own separate, isotropic-only frame construction).
	// GPU SPPM (sppm_programs.cu) was deliberately NOT converted - it would
	// need dpdu threaded through its own separate camera/photon-pass
	// intersection code and its own per-pixel persisted state, a materially
	// bigger lift for a backend that already only implements 2 of these 4
	// material kinds - so it still uses BuildArbitraryTangentFrame() and
	// will still show CPU-orientation-mismatched anisotropic highlights.
	float roughnessV = -1.0f;

	// CoatedConductor only. `roughness`/`roughnessV` above are the COAT's; the base conductor has its own GGX
	// roughness (pbrt: conductor.uroughness/vroughness - same authored-vs-alpha convention, remapRoughness applies
	// to it too). Negative (the default) means none was given and the coat's roughness applies to both interfaces,
	// the older single-roughness spelling - NOT a smooth conductor, which would be an authored 0.
	float condRoughness  = -1.0f;
	float condRoughnessV = -1.0f;
	// CoatedConductor only: pbrt's layer "thickness" (Beer-Lambert attenuation through the coat, unit extinction).
	float layerThickness = 0.01f;

	// Medium / DielectricMedium only: the PER-CHANNEL coefficients of a homogeneous medium whose extinction differs between colour
	// channels (sigma_t = chromaSigmaA + chromaSigmaS). All zero (the default) for a grey medium, which keeps using the scalar
	// extinction (`ior`/`sigma_t`) and the tint albedo above exactly as before - medium_is_chromatic() is the one test. chromaLe is
	// MakeNamedMedium's RAW "rgb Le" (not the sigma_a/sigma_t-weighted `medium_emission` above): the per-channel event sampler
	// weights it per collision (see sample_homogeneous_event in src/shared/volume_scattering.h, which CPU's constant_medium uses too).
	float3 chromaSigmaA = {0.0f, 0.0f, 0.0f};
	float3 chromaSigmaS = {0.0f, 0.0f, 0.0f};
	float3 chromaLe     = {0.0f, 0.0f, 0.0f};
	// Wavefront (spectral) twin of chromaSigmaA/S: the coefficients (c0, c1, c2) of the sigmoid polynomial of each coefficient's
	// unbounded RGB->spectrum uplift, value(lambda) = scale * s(c0*lambda^2 + c1*lambda + c2), baked on the host so the shadow any-hit
	// program (which has no uplift tables) can evaluate sigma(lambda) per wavelength. scale == 0 means that coefficient is zero.
	float3 chromaCoefA  = {0.0f, 0.0f, 0.0f};
	float3 chromaCoefS  = {0.0f, 0.0f, 0.0f};
	float  chromaScaleA = 0.0f;
	float  chromaScaleS = 0.0f;
};

// Punctual (delta) light kinds - point/spot/distant. These are evaluated
// deterministically at every hit (pdf=0, no MIS, not part of lightIndices/
// aliasTable/lightKinds at all) rather than stochastically picked like
// the area lights, mirroring how src/TheRestOfYourLife/punctual_light_objects.h
// handles them on the CPU.
enum class PunctualLightKind : int {
	Point = 0,
	Spot = 1,
	Distant = 2,
	Goniometric = 3,
	Projection = 4
};

// Max image dimensions for goniometric/projection lights, stored inline in
// PunctualLightGPU (see below) rather than as a separately-allocated device
// buffer: light counts are tiny (typically 1-3 per scene) and PunctualLightGPU
// itself already lives behind a device POINTER sized to numPunctualLights at
// runtime (LaunchParams::punctualLights, not a fixed __constant__ array), so
// a fixed-size inline array per element avoids a second device-buffer-
// management path (alloc/upload/free, extra SBT plumbing) without risking any
// __constant__-memory budget. Raised from this codebase's original 32 (sized
// only for its own hand-built showcase scenes' synthetic 16x8/8x8 patterns)
// to 64 once real decoded profile/slide images could exceed that - pbrt-v4's
// own `imgtool makeequiarea` typically emits goniometric profiles in this
// range, and a real projection slide photo can be much larger still, so an
// oversized real image is downsampled (nearest-neighbor) to fit this cap at
// build time (pbrt_gpu_builder.h) rather than silently cropped or falling
// back to a flat approximation - see that file's own comment.
static constexpr int kGonioImageMaxDim = 64;
static constexpr int kProjImageMaxDim  = 64;
