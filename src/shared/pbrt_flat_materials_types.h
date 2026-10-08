#pragma once
// pbrt_flat_materials_types.h -- the flattened Material (and its procedural-texture and spectrum-name helpers) pbrt_flatten.h produces. Plain data; part of
// pbrt_flatten.h, which includes it.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"
#include "conductor_data.h"
#include "glass_data.h"
#include "spectrum_types.h"   // BlackbodySpectrum - see resolveEmissionColor()'s own comment
#include "spectral_math.h"    // SpectrumToXYZ/InnerProduct/GetCIE_Y - ditto
#include "rgb_colorspace.h"   // RGBColorSpaceFromName() - ditto

namespace pbrt_flatten {

// Texture "imagemap"'s own "string encoding"/"string wrap"/"bool invert"
// (pbrt-v4), resolved once per bound imagemap - see Material::textureGamma's
// own comment (below) for what each field means and defaults to. Originally
// only threaded for the primary reflectance-equivalent slot (Material::
// textureFilename, as 3 loose fields); this struct exists so the
// transmittance/roughness slots below can carry the identical resolution
// without duplicating those 3 fields a second and third time - see
// resolveTextureDecodeOptions() (flatten_detail, below) for how it's built.
struct TextureDecodeOptions {
	double gamma = kDefaultTextureGamma;
	std::string wrap = "repeat";
	int wrapIndex = 1;   // Repeat - see Material::textureWrapIndex's own comment
	bool invert = false;
};

// A second level of checkerboard/mix nesting: tex1/tex2 (or mix's amount)
// binds to ANOTHER "checkerboard"/"mix" Texture instead of a flat literal
// or a bare imagemap - e.g. a checker-of-checkers, or a mix blending two
// checker patterns. Deliberately capped at exactly this one extra level:
// THIS struct's own tex1/tex2/amount slots may only be a flat literal or a
// bare imagemap (the same one-level scope Material::checkerTex1Filename's
// own comment documents), never a further nested procedural texture -
// unbounded recursion would need real cycle/depth guarding this loader
// has never needed before, for a feature real pbrt-v4 scenes essentially
// never use past 2 levels. `kind` ("checkerboard" or "mix") is empty when
// this slot isn't a nested-procedural at all (the common case) - checked
// before color1/color2/etc, which otherwise carry their own always-valid
// defaults and can't distinguish "unset" from "genuinely default".
// checkerboard's uscale/vscale and mix's amount/amountFilename fields
// simply go unused for the other kind.
struct NestedProceduralTexture {
	std::string kind;
	double color1[3] = {1.0, 1.0, 1.0};
	double color2[3] = {0.0, 0.0, 0.0};
	std::string tex1Filename;
	std::string tex2Filename;
	double uscale = 1.0, vscale = 1.0;   // checkerboard only
	double amount = 0.5;                  // mix only
	std::string amountFilename;           // mix only
};

// GPU's flat-colour stand-in for a resolved NestedProceduralTexture (GPU has
// no representation for a nested procedural entry - see Material::
// checkerTex1Nested's own comment). A plain 50/50 average is exactly right
// for "checkerboard" (its two cells cover equal area by construction), but
// would silently ignore "mix"'s own amount - weight by it instead, unless
// amount itself is texture-bound (amountFilename set), where there's no
// single per-point value available at flatten() time to weight by, so 50/50
// is the best available approximation, same as it is for checkerboard.
inline void nestedProceduralAverageColor(const NestedProceduralTexture &n, double out[3]) {
	const double w = (n.kind == "mix" && n.amountFilename.empty()) ? n.amount : 0.5;
	for (int i = 0; i < 3; ++i)
		out[i] = (1.0 - w) * n.color1[i] + w * n.color2[i];
}

// Same idea as nestedProceduralAverageColor() just above, collapsed to a
// single scalar (plain channel average of that same approximate colour) -
// used when a NestedProceduralTexture stands in for a top-level "mix"
// texture's own "amount" slot (Material::mixAmountNested's own comment):
// GPU has no per-point representation for a nested-procedural amount any
// more than it does for a nested-procedural tex1/tex2 (TextureData's
// amountImageIdx is image-only, same as tex1ImageIdx/tex2ImageIdx), so this
// feeds Material::mixAmount - the SAME flat fallback GPU already reads
// whenever amountImageIdx isn't set - with a representative value instead
// of leaving it at the meaningless struct default.
inline double nestedProceduralAverageScalar(const NestedProceduralTexture &n) {
	double c[3];
	nestedProceduralAverageColor(n, c);
	return (c[0] + c[1] + c[2]) / 3.0;
}

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	std::string pbrtType;              // as written, for diagnostics
	double color[3] = {0.5, 0.5, 0.5}; // reflectance / albedo
	double roughness = 0.0;
	// Independent GGX roughness along the surface's own tangent (u) and
	// bitangent (v) directions - pbrt-v4's real anisotropic spelling
	// ("uroughness"/"vroughness"). Populated alongside `roughness` above
	// (which stays exactly as before, an isotropic fallback derived the
	// same way it always has, for callers that haven't been updated to
	// read these two instead - see Round 6 Phase 3). Each independently
	// falls back to plain "roughness" when its own name isn't present, so
	// an ordinary isotropic scene (only "roughness" set) yields
	// roughness_u == roughness_v == roughness, and Round 6 Phase 3's own
	// downstream wiring (pbrt_cpu_builder.h/pbrt_gpu_builder.h) degrades
	// to the exact isotropic construction it used before whenever the two
	// happen to be equal.
	double roughness_u = 0.0;
	double roughness_v = 0.0;
	// pbrt-v4's "remaproughness" (default true): when true, roughness/
	// roughness_u/roughness_v above are perceptually-remapped authored
	// values that still need RoughnessToAlpha applied before use as real
	// GGX alpha; when false, they already ARE the alpha value. See
	// material_pbrt.h's roughness_or_alpha() for where this gets applied.
	bool remapRoughness = true;
	// Also reused for Hair's own "eta" (fiber IOR) - flatten()'s Hair branch
	// overrides this field's default to pbrt-v4's own Hair-specific 1.55
	// (HairMaterial::Create) rather than this field's general 1.5.
	double ior = 1.5;
	// Principled only (this loader's own non-standard material - see
	// MaterialKind::Principled's own comment): 0 = pure dielectric/plastic,
	// 1 = pure metal, blended in between - matches `principled` (CPU,
	// principled_material.h) and MaterialData::principled_params.metallic
	// (GPU) exactly, same default (0.0) as that CPU class's own convenience
	// constructor.
	double metallic = 0.0;
	// Principled only: clearcoat layer weight (0 = none) and its own
	// independent specular roughness - matches `principled`'s clearcoat_tex/
	// clearcoat_rough and MaterialData::principled_params.clearcoat/
	// clearcoat_rough exactly, same defaults (0.0/0.1) as that CPU class's
	// own convenience constructor. Deliberately separate fields from
	// roughness/roughness_u/roughness_v above (CoatedDiffuse/CoatedConductor's
	// own coat-roughness slots) - Principled's clearcoat is a SEPARATE third
	// layer on top of its own base metallic/roughness response, not a
	// substitute for either.
	double clearcoat = 0.0;
	double clearcoatRoughness = 0.1;
	// Dielectric only (smooth or rough, via m.roughness/m.roughness_u/
	// m.roughness_v above) - an ARTIST-FACING ABBE NUMBER, not real pbrt-v4
	// syntax at all (pbrt-v4's own genuine spectral dispersion is a full
	// "spectrum eta" per-wavelength curve, which this loader doesn't
	// integrate). 0.0 (the default) means "not dispersive" - a real Abbe
	// number is always positive (common glasses run roughly 20-90), so this
	// needs no separate bool, same "zero means off" convention
	// dispersive_extra.cauchy_A > 0.0f already uses on the GPU side
	// (optix_types.h). Reachable only under --spectral (per-wavelength
	// hero-wavelength tracing) - see dispersive_material's own comment,
	// material_base.h, for why a dispersive material renders identically to
	// a flat-IOR one under the default RGB path. Thin wrapper around this
	// codebase's own already-implemented, already-working
	// dielectric::make_dispersive()/rough_dielectric::make_dispersive()
	// (CPU) and add_dispersive_dielectric()/add_dispersive_rough_dielectric()'s
	// own Cauchy-coefficient derivation (GPU, CauchyCoefficientsFromAbbe() -
	// src/shared/fresnel.h) - not a new rendering feature, purely new pbrt-
	// file reachability for an existing one.
	double abbeNumber = 0.0;
	// Dielectric only - "tf", a non-standard "rgb" parameter: the OBJ/.mtl
	// "Tf" transmission filter (stained glass, tinted windows). Multiplies the
	// transmitted contribution only, leaving reflection clear - what the native
	// dielectric(ior, Tf) / MaterialData::transmission_filter always did; the
	// environment scenes migrated from .mtl assets (pbrt_scenes/environment-*.pbrt)
	// are the only users. White (the default) is a no-op, and a rough or
	// dispersive dielectric ignores it, as the native glass did.
	double transmissionFilter[3] = {1.0, 1.0, 1.0};
	// DiffuseTransmission only: the light that passes through rather than
	// reflects. pbrt-v4's own default (0.25) is closer to that material's
	// intent than reusing `color`'s 0.5 default would be - a
	// diffusetransmission with neither parameter set should look like a
	// frosted panel passing about a quarter of the light each way, not a
	// mirror-symmetric reflectance/transmittance split at 0.5/0.5.
	double transmittance[3] = {0.25, 0.25, 0.25};

	// Subsurface only: absorption/scattering coefficients, ALREADY multiplied
	// by the material's "scale" parameter (pbrt-v4 GetBSSRDF: sig_a = scale *
	// sigma_a, sig_s = scale * sigma_s - see flatten()'s subsurface branch).
	// Defaults are pbrt-v4's own "nothing specified" preset
	// (SubsurfaceMaterial::Create's case 4, materials.cpp), which happens to
	// equal the "Wholemilk" named preset.
	//
	// ALSO reused for Hair's own absorption coefficient (see flatten()'s Hair
	// branch) - the struct-level default above is meaningless for Hair
	// (always overwritten there, never left at this Subsurface-shaped
	// default): a genuine Hair material always resolves sigma_a[] to one of
	// a literal "sigma_a", a computed eumelanin/pheomelanin concentration, or
	// pbrt-v4's own default-brown fallback.
	double sigma_a[3] = {0.0011, 0.0024, 0.014};
	double sigma_s[3] = {2.55, 3.21, 3.77};
	double g = 0.0;   // Henyey-Greenstein asymmetry (subsurface only)

	// Hair only - longitudinal/azimuthal roughness and cuticle scale-tilt
	// angle (degrees). Defaults match pbrt-v4's own HairMaterial::Create
	// exactly (materials.cpp), which happen to already match hair_material.h's
	// own pre-existing constructor defaults.
	double betaM = 0.3, betaN = 0.3, alphaDeg = 2.0;

	// Measured only: the "filename" parameter naming the .bsdf tensor file,
	// exactly AS WRITTEN in the scene ("bsdfs/foo.bsdf" - relative to the
	// scene file's own directory, same convention as Shape "plymesh"'s
	// filename or LightSource "infinite"'s). This header stays filesystem-
	// free by design (see the file comment), so it cannot resolve or read
	// the file itself - pbrt_load.h::loadFile() does both AFTER flatten()
	// returns, exactly as it already does for the infinite light's image,
	// and OVERWRITES this field with the resolved path on success so
	// pbrt_cpu_builder.h's `class measured` never needs to know the scene's
	// directory. Empty means "no filename given" (or, after pbrt_load.h's
	// pass, "could not be resolved/loaded") - either way the material falls
	// back to a diffuse approximation.
	std::string measuredFilename;

	// Diffuse, CoatedDiffuse, or DiffuseTransmission's own "reflectance": an
	// "imagemap" Texture bound to it, naming the image file exactly AS
	// WRITTEN in the scene - same "stays filesystem-free, resolved later by
	// pbrt_load.h" convention as measuredFilename above (see that field's
	// own comment). Empty means either no texture was bound, the bound
	// texture wasn't an imagemap (or a "scale" wrapping one - see
	// textureScale below, Diffuse/CoatedDiffuse only), or (after
	// pbrt_load.h's pass) the file could not be found - any of which falls
	// back to `color` as a flat reflectance, same as today. Only
	// "reflectance" is handled (not every texture-bindable parameter on
	// every material kind): pbrt's own ganesha scene (a CoatedDiffuse statue
	// whose reflectance is an imagemap), barcelona-pavilion's CoatedDiffuse
	// AND plain-Diffuse surfaces (mostly reflectance bound to a "scale"
	// texture wrapping an imagemap - see the warning loop below) and
	// barcelona-pavilion's own foliage (DiffuseTransmission, "texture
	// reflectance"/"texture transmittance" both bound to the SAME bare
	// imagemap - see transmittanceTextureFilename below) are the motivating
	// cases, and scoping to reflectance/transmittance on these three kinds
	// keeps this addition bounded rather than building a general
	// procedural-texture pipeline in one pass - checkerboard/fbm/marble/mix
	// (hasCheckerReflectance etc. below) and the "scale" unwrap (see
	// textureScale below) both stay Diffuse/CoatedDiffuse-only, since no
	// bundled scene needs DiffuseTransmission's own reflectance bound to
	// either.
	std::string textureFilename;

	// A "scale"-class Texture's own "float scale" when textureFilename came
	// from unwrapping one (pbrt's own real syntax for this: a named "scale"
	// Texture whose "texture tex" names the real imagemap - barcelona-
	// pavilion's own dominant pattern for reflectance, already unwrapped the
	// identical way for "displacement" below, see displacementScale's own
	// comment). Diffuse/CoatedDiffuse/DiffuseTransmission all apply this to
	// their own reflectance now (scaled_texture on CPU, MaterialData::
	// emissionScale reused on GPU - see transmittanceTextureScale below for
	// DiffuseTransmission's OWN transmittance, a separate field since the
	// two channels can be bound to independently-scaled textures). 1.0 (a
	// no-op multiply) when textureFilename came from a bare imagemap with
	// no wrapping "scale", or when textureFilename is empty.
	double textureScale = 1.0;

	// Texture "imagemap"'s own "string encoding"/"string wrap"/"bool
	// invert" (pbrt-v4), resolved for textureFilename above as 3 loose
	// fields (kept exactly as-is - CPU's imageMapOptionsFor()/GPU's own call
	// sites already read these 3 by name) - transmittanceTextureFilename/
	// roughnessTextureFilename below carry the identical resolution via
	// their own TextureDecodeOptions field instead (transmittanceTextureOptions/
	// roughnessTextureOptions). alphaTextureFilename/displacementTextureFilename
	// still don't: alpha is a coverage MASK, not colour, so "encoding"
	// (gamma) is not meaningful there by this codebase's own established
	// design (see gpu/optix/pbrt_gpu_builder.h's getOrBuildPbrtAlphaMaskTexture()
	// own comment on why alpha masks deliberately skip the gamma decode a
	// reflectance imagemap needs); displacement goes through a materially
	// different CPU pipeline (rtw_image/image_texture, not mipmap_texture/
	// MipMapOptions) with no wrap-mode concept at all today. Both remaining
	// gaps still warn via warnIfImagemapOptionsIgnored() below rather than
	// silently dropping the request. textureGamma: resolved from
	// "encoding" to an actual gamma exponent - "linear" -> 1.0 (no decode,
	// pbrt-v4's own real intent for a roughness/normal/displacement map
	// bound this way), "gamma <value>" -> that value, "sRGB" or absent ->
	// 2.2 (this codebase's own long-standing default for every 8-bit
	// texture - an approximation of pbrt-v4's real sRGB curve, not its
	// exact piecewise-linear-toe formula, matching this loader's existing
	// "close enough, not bit-exact" precedent for other approximated
	// features). See src/TheRestOfYourLife/rtw_stb_image.h's rtw_image
	// constructor for what this value actually does downstream.
	double textureGamma = kDefaultTextureGamma;
	// "repeat" (pbrt-v4's own real default)/"clamp"/"black" - unlike
	// mipmap.h's own MipMapOptions::wrap field default (Clamp, kept
	// unchanged there deliberately so native/non-pbrt scenes are
	// unaffected - see that field's own comment), THIS field's default is
	// pbrt-v4's real one, applied explicitly per pbrt-loaded texture.
	std::string textureWrap = "repeat";
	// The same resolved value as textureWrap above, as an ordinal
	// (Clamp=0/Repeat=1/Black=2) instead of a string - matches both CPU's
	// MipWrapMode (src/shared/mipmap.h) and GPU's GpuWrapMode (gpu/optix/
	// optix_types.h) enumerator values exactly by construction (both were
	// defined to mirror this ordering), so either backend's own builder can
	// just `static_cast` this into its own enum type instead of
	// re-implementing the same wrap-string validation/fallback logic this
	// struct's own resolution code (below, in flatten()) already owns.
	// textureWrap itself is kept alongside this (not replaced) since it's
	// still useful as a human-readable value for anything that wants one.
	int textureWrapIndex = 1;  // Repeat, matching textureWrap's own default
	bool textureInvert = false;

	// DiffuseTransmission only: an "imagemap" Texture bound to
	// "transmittance", same "raw as written, resolved later by
	// pbrt_load.h" convention as textureFilename above - bare imagemap,
	// optionally further wrapped in a "scale" texture (transmittanceTexture
	// Scale below) - same one-level unwrap textureScale's own comment
	// documents for reflectance, applied independently here since
	// barcelona-pavilion's foliage happens to bind "reflectance" and
	// "transmittance" to the identical bare-imagemap texture in practice,
	// but a scene binding them to two DIFFERENTLY-scaled textures must
	// still resolve each correctly. No procedural (checkerboard/fbm/marble/
	// mix) support - no bundled scene needs it.
	std::string transmittanceTextureFilename;

	// transmittanceTextureFilename's own "scale", same shape as
	// textureScale above but independent - a DiffuseTransmission material's
	// reflectance and transmittance can each be wrapped in their own,
	// differently-valued "scale" Texture. 1.0 (a no-op multiply) when
	// transmittanceTextureFilename came from a bare imagemap with no
	// wrapping "scale", or when transmittanceTextureFilename is empty.
	double transmittanceTextureScale = 1.0;

	// This slot's own "encoding"/"wrap"/"invert" - see TextureDecodeOptions'
	// own comment. Default-constructed (2.2/"repeat"/no-invert) when
	// transmittanceTextureFilename is empty or the scene gave none of the
	// three.
	TextureDecodeOptions transmittanceTextureOptions;

	// Dielectric only: an "imagemap" Texture bound to "roughness" (e.g. a
	// scratched/frosted-glass mask), same "raw as written, resolved later
	// by pbrt_load.h" convention as textureFilename above - bare imagemap
	// only, no "scale"-wrap or procedural (checkerboard/fbm/marble/mix)
	// support, matching transmittanceTextureFilename's identical scope
	// narrowing. No bundled scene needs this (added for texture-parity
	// with Diffuse/CoatedDiffuse/DiffuseTransmission's own reflectance/
	// transmittance texture-binding, not a specific scene's requirement) -
	// the image's own red/x channel becomes the scalar roughness at each
	// hit; see rough_dielectric::true_alpha()'s own comment (material_pbrt.h)
	// for why this is isotropic-only, sampled per-hit rather than once.
	std::string roughnessTextureFilename;

	// This slot's own "encoding"/"wrap"/"invert" - see
	// transmittanceTextureOptions' own comment just above.
	TextureDecodeOptions roughnessTextureOptions;

	// A Diffuse material's "reflectance" bound to a "checkerboard" Texture
	// instead of an "imagemap" one (e.g. named-material-and-texture.pbrt's
	// "floor-check": Texture "floor-check" "spectrum" "checkerboard"
	// "float uscale" [8] "float vscale" [8] - no tex1/tex2 given, so
	// pbrt-v4's own defaults apply). Unlike textureFilename, this can't be
	// represented as a plain filename (there is no file - it's two flat
	// colours procedurally tiled by UV), hence the separate fields below
	// rather than overloading textureFilename's meaning. hasCheckerReflectance
	// is the "is this meaningful" flag, since an all-default checkerboard's
	// fields are otherwise indistinguishable from "unset".
	//
	// tex1/tex2 each independently support up to TWO levels of nesting: a
	// flat float/rgb literal (checkerColor1/2 below), a reference to a bare
	// "imagemap" Texture (checkerTex1Filename/checkerTex2Filename below), or
	// a reference to ANOTHER "checkerboard"/"mix" Texture
	// (checkerTex1Nested/checkerTex2Nested below, CPU-only - see that
	// struct's own comment for why exactly two levels and no GPU support).
	// Exactly one of the three (checkerColorN / checkerTexNFilename /
	// checkerTexNNested) is meaningful per slot - see flatten()'s own
	// checkerboard-resolution code for which.
	bool hasCheckerReflectance = false;
	double checkerColor1[3] = {1.0, 1.0, 1.0};  // pbrt-v4 tex1 default: white
	double checkerColor2[3] = {0.0, 0.0, 0.0};  // pbrt-v4 tex2 default: black
	std::string checkerTex1Filename;  // set instead of checkerColor1 when tex1 nests a bare imagemap
	std::string checkerTex2Filename;  // set instead of checkerColor2 when tex2 nests a bare imagemap
	NestedProceduralTexture checkerTex1Nested;  // kind non-empty when tex1 nests a further checkerboard/mix
	NestedProceduralTexture checkerTex2Nested;
	double checkerUScale = 1.0;
	double checkerVScale = 1.0;
	// pbrt-v4's real "checkerboard" texture also supports "integer
	// dimension" [3] - a 3D WORLD-SPACE checker (CheckerboardTexture::
	// Evaluate's dimension==3 branch), keyed on a texture-space point
	// instead of (u,v) - checkerUScale/checkerVScale above are meaningless
	// for this variant (real pbrt-v4 ignores them too). This is the exact
	// same pattern this project's OWN original (pre-pbrt) checker_texture
	// (src/TheRestOfYourLife/texture.h) already implements - see that
	// class's own header comment. checkerWorldToTexture is the inverse of
	// the CTM active when the Texture directive was declared
	// (TextureDecl::xform's own comment) - a plain "Scale s s s" before the
	// Texture directive reproduces this project's own native checker_
	// texture(scale, ...) constructor exactly, since its inv_scale=1/scale
	// is exactly what Scale(s)'s own inverse applies to a world point.
	// Row-major affine 4x4, identity when dimension isn't 3 (checkerIs3D
	// stays false so callers never need to consult this field).
	bool checkerIs3D = false;
	double checkerWorldToTexture[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

	// A Diffuse material's "reflectance" bound to an "fbm" Texture (pbrt-v4
	// FBmTexture - fractional Brownian motion noise, e.g. a cloudy/mottled
	// pattern). Same flat-literal-only scope as checkerboard above - "fbm"
	// takes no tex1/tex2 to begin with (it IS the pattern, not a blend of
	// two others), so there's no nested-texture case to fall through on.
	// Param names match pbrt-v4 exactly: "octaves" (int) and "roughness"
	// (float, internally called omega) - see fbm_texture (src/TheRestOfYourLife/
	// texture.h), the existing CPU class this resolves to (already used by
	// non-pbrt scenes; this just gives the pbrt loader a way to reach it).
	bool hasFbmReflectance = false;
	int fbmOctaves = 8;
	double fbmRoughness = 0.5;

	// A Diffuse material's "reflectance" bound to a "marble" Texture
	// (pbrt-v4 MarbleTexture - FBm-perturbed sine wave through a marble
	// colour spline). Resolves to the existing marble_texture CPU class
	// (src/TheRestOfYourLife/texture.h). Param names match pbrt-v4 exactly.
	bool hasMarbleReflectance = false;
	int marbleOctaves = 8;
	double marbleRoughness = 0.5;
	double marbleScale = 1.0;
	double marbleVariation = 0.2;

	// A Diffuse material's "reflectance" bound to a "mix" Texture (pbrt-v4
	// SpectrumMixTexture - lerp between two colours by "amount"). tex1/tex2
	// support the SAME up-to-two-level nesting as checkerboard's own
	// tex1/tex2 above (mixTex1Filename/mixTex2Filename for a bare imagemap;
	// mixTex1Nested/mixTex2Nested for a further checkerboard/mix, CPU-only -
	// see NestedProceduralTexture's own comment). "amount" ITSELF bound to a
	// texture (e.g. driven by an fbm pattern for a dirt/wear mask, pbrt-v4's
	// most common real use of "mix" - barcelona-pavilion's own
	// materials.pbrt has a commented-out "float amount" override on several
	// Mix declarations, hinting the original scene author considered
	// exactly this) is ALSO supported now, at the SAME up-to-two-level
	// nesting as tex1/tex2 (mixAmountTextureFilename below for a bare
	// imagemap; mixAmountNested for a further checkerboard/mix, CPU-only -
	// see NestedProceduralTexture's own comment) - a per-point scalar blend
	// mask driven by a further procedural pattern (e.g. a checker-driven
	// mix, or a mix-of-mixes weight) is a real pbrt-v4 capability this
	// loader now reaches too. A nested amount that resolves to neither
	// level still falls through to the generic "not supported" warning.
	// Defaults match pbrt-v4's SpectrumMixTexture exactly (tex1 black, tex2
	// white, amount 0.5).
	bool hasMixReflectance = false;
	double mixColor1[3] = {0.0, 0.0, 0.0};
	double mixColor2[3] = {1.0, 1.0, 1.0};
	std::string mixTex1Filename;  // set instead of mixColor1 when tex1 nests a bare imagemap
	std::string mixTex2Filename;  // set instead of mixColor2 when tex2 nests a bare imagemap
	NestedProceduralTexture mixTex1Nested;  // kind non-empty when tex1 nests a further checkerboard/mix
	NestedProceduralTexture mixTex2Nested;
	double mixAmount = 0.5;
	std::string mixAmountTextureFilename;  // set instead of mixAmount when "amount" nests a bare imagemap
	NestedProceduralTexture mixAmountNested;  // kind non-empty when "amount" nests a further checkerboard/mix

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "windy" Texture
	// (pbrt-v4 WindyTexture - two FBm calls combined for a windswept-grass
	// pattern). Parameterless in real pbrt-v4 (no scene-overridable params
	// at all), so this is just the "is this bound" flag - see
	// windy_texture's own comment (texture.h) for the formula.
	bool hasWindyReflectance = false;

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "wrinkled" Texture
	// (pbrt-v4 WrinkledTexture - raw Turbulence, not FBm). Same "octaves"/
	// "roughness" param names/defaults as fbm above.
	bool hasWrinkledReflectance = false;
	int wrinkledOctaves = 8;
	double wrinkledRoughness = 0.5;

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "dots" Texture
	// (pbrt-v4 DotsTexture - a per-UV-cell polka-dot pattern blending
	// "inside"/"outside"). Same one-level-nested-bare-imagemap support for
	// inside/outside as checkerboard's own tex1/tex2 (dotsInsideTexFilename/
	// dotsOutsideTexFilename below). Defaults match pbrt-v4's DotsTexture
	// exactly (inside white, outside black).
	bool hasDotsReflectance = false;
	double dotsInsideColor[3] = {1.0, 1.0, 1.0};
	double dotsOutsideColor[3] = {0.0, 0.0, 0.0};
	std::string dotsInsideTexFilename;   // set instead of dotsInsideColor when "inside" nests a bare imagemap
	std::string dotsOutsideTexFilename;  // set instead of dotsOutsideColor when "outside" nests a bare imagemap

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "bilerp" Texture
	// (pbrt-v4 BilerpTexture - plain bilinear blend of 4 corner colours by
	// (u,v)). Flat-literal corners only - no nested-imagemap support,
	// matching how rarely a real scene binds anything but a flat colour to
	// a bilerp corner (no bundled scene needs more). Defaults match
	// pbrt-v4's BilerpTexture exactly (v00/v10 black, v01/v11 white).
	bool hasBilerpReflectance = false;
	double bilerpV00[3] = {0.0, 0.0, 0.0};
	double bilerpV01[3] = {1.0, 1.0, 1.0};
	double bilerpV10[3] = {0.0, 0.0, 0.0};
	double bilerpV11[3] = {1.0, 1.0, 1.0};

	// A pbrt Shape's own "alpha" parameter (bound to a "float"/"imagemap"
	// Texture - e.g. barcelona-pavilion's foliage, "Shape \"plymesh\"
	// \"texture alpha\" [ \"leaf_alpha\" ]"), NOT a Material directive
	// parameter - pbrt's alpha-cutout mask is authored per-Shape, one leaf
	// mesh at a time, reusing the same colour texture's alpha/luminance
	// channel. Stored here anyway (rather than as a new field on Triangle/
	// FlatScene) because both this loader's Material-building convention
	// (measuredFilename/textureFilename, both "raw as written, resolved
	// later by pbrt_load.h") and the GPU backend's actual layout
	// (MaterialData::alphaMaskTexIdx is per-material, not per-triangle) are
	// already per-material - flatten() writes this onto whichever Material
	// the owning Shape resolved to (out.materials[shape.materialIndex]) at
	// the point the Shape is processed. Every scene in this loader's own
	// corpus that uses "texture alpha" gives each alpha-masked Shape its own
	// unnamed Material declared immediately before it (never a NamedMaterial
	// shared by shapes with different alpha masks), so this 1:1 shape<->
	// material correspondence holds in practice; a scene that violated it
	// would have the last such Shape's alpha win for every shape sharing
	// that material index - flatten() warns when this happens (see its own
	// Shape "alpha" handling), but does not prevent it.
	std::string alphaTextureFilename;

	// A Material's own "texture displacement" parameter (pbrt-v4 bump
	// mapping, e.g. barcelona-pavilion's "pavet" material: MakeNamedMaterial
	// ... "texture displacement" ["pavet-bump"]) - a "float" texture, NOT
	// gated on MaterialKind::Diffuse the way textureFilename's reflectance
	// binding is, since real scenes bind displacement on coateddiffuse,
	// dielectric, and other kinds too. Raw as written, resolved later by
	// pbrt_load.h - same convention as textureFilename/measuredFilename/
	// alphaTextureFilename above. Mirrors OBJ/MTL's own map_Bump handling
	// (mesh.h) once resolved - see pbrt_cpu_builder.h/pbrt_gpu_builder.h's
	// own comments on how this field is consumed.
	std::string displacementTextureFilename;
	// A scale factor from a "scale"-class texture wrapping the actual
	// imagemap (e.g. barcelona-pavilion's water material: "texture
	// displacement" ["water-bump"], where "water-bump" is itself
	// "float" "scale" { "float scale" [0.005], "texture tex" ["water-bump-base"] }).
	// 1.0 (no-op) when displacement resolves directly to an imagemap with no
	// wrapping scale texture.
	double displacementScale = 1.0;

	// Mix only: indices into FlatScene::materials of the two blended
	// sub-materials - pbrt-v4's own "string materials" parameter names them
	// (an array of exactly two named-material references, resolved against
	// every MakeNamedMaterial in the scene, not just ones declared earlier -
	// see flatten()'s own mix-resolution code), and mixWeight is pbrt-v4's
	// "amount" (default 0.5): the probability of material B winning at any
	// given shading point, matching mix_material's own "weight" parameter
	// exactly (0 = pure A, 1 = pure B). -1 means "could not be resolved" -
	// fewer than two names given, or a name that does not match any
	// MakeNamedMaterial - in which case `kind` is downgraded to Unsupported
	// by flatten() itself (see there), so pbrt_cpu_builder.h never has to
	// check these fields are valid before indexing with them.
	int mixMaterialA = -1;
	int mixMaterialB = -1;
	double mixWeight = 0.5;

	// Conductor only: real complex IOR, resolved when "spectrum eta"/
	// "spectrum k" named one of this codebase's known metals (see
	// conductorElementFromSpectrumName()/src/shared/conductor_data.h's
	// FindConductorPreset() below) - pbrt's conductors are normally
	// described this way ("metal-Ag-eta"/"metal-Ag-k"), not via plain
	// floats/RGB, which flatten() used to just fail to parse silently.
	// hasConductorPreset stays false (and the builders keep falling back to
	// the existing metal/fuzz-mirror approximation, exactly as before this
	// field existed) for an explicit RGB k, an unrecognized named spectrum,
	// or no eta/k given at all.
	bool hasConductorPreset = false;
	double conductorEta[3] = {0.0, 0.0, 0.0};
	double conductorK[3] = {0.0, 0.0, 0.0};

	// CoatedConductor only. roughness_u/roughness_v/ior above are the COAT's (pbrt: interface.uroughness/
	// vroughness/roughness and interface.eta). The base conductor has its own roughness (pbrt:
	// conductor.uroughness/vroughness/roughness, default 0): -1 means none was given and the coat's applies to
	// both interfaces (the legacy single "roughness" spelling). coatThickness is pbrt's "thickness".
	double conductorRoughness_u = -1.0;
	double conductorRoughness_v = -1.0;
	double coatThickness = 0.01;
};

// Extracts "Ag" from pbrt-v4's "metal-Ag-eta"/"metal-Ag-k" named-spectrum
// convention (the only shape this loader's bundled scene corpus uses for
// conductor eta/k - see conductor_data.h's own table for which elements are
// recognized once extracted). Returns "" for anything that doesn't match
// the "metal-<elem>-eta"/"metal-<elem>-k" shape at all (an explicit RGB
// value, or an unrecognized/non-metal named spectrum).
inline std::string conductorElementFromSpectrumName(const std::string &name) {
	const std::string prefix = "metal-";
	if (name.rfind(prefix, 0) != 0) return "";
	for (const char *suffix : {"-eta", "-k"}) {
		const std::string suf(suffix);
		if (name.size() > prefix.size() + suf.size() &&
			name.compare(name.size() - suf.size(), suf.size(), suf) == 0) {
			return name.substr(prefix.size(), name.size() - prefix.size() - suf.size());
		}
	}
	return "";
}

// Extracts "BK7" from pbrt-v4's "glass-BK7" named-spectrum convention (the
// only shape a dielectric's "spectrum eta" uses - see glass_data.h's own
// comment for the full 7-name list pbrt-v4 actually recognizes). Returns ""
// for anything that doesn't start with "glass-" at all (an explicit float
// eta, or a non-glass named spectrum).
inline std::string glassElementFromSpectrumName(const std::string &name) {
	const std::string prefix = "glass-";
	if (name.rfind(prefix, 0) != 0) return "";
	return name.substr(prefix.size());
}

// Resolves an emission-colour parameter (pbrt-v4's "L" on AreaLightSource/
// distant/infinite, "I" on point/spot/goniometric), handling both the
// ordinary flat "rgb"/"float" case (delegates to ParamList::getVec3 exactly
// as before - unaffected) and a "blackbody" temperature in Kelvin, which
// this loader previously either warned-and-ignored (AreaLightSource) or
// silently dropped with no warning at all (every punctual light kind) -
// getVec3 requires >=3 numbers, and a "blackbody L" param has exactly 1, so
// it always fell through to the flat default colour regardless of the
// requested temperature (see docs/PBRT_SUPPORT.md's own note on this, and
// barcelona-pavilion's/contemporary-bathroom's real "blackbody L" area
// lights - the motivating case: every one of them rendered as flat white at
// whatever "float scale" said, with zero hue difference between a 2500K and
// a 6500K light).
//
// Converts via this codebase's own already-ported pbrt-v4 spectral pipeline
// (BlackbodySpectrum -> SpectrumToXYZ -> RGBColorSpace::sRGB(), spectrum_
// types.h/spectral_math.h/rgb_colorspace.h - no new spectral machinery
// needed) exactly matching pbrt-v4's own light-construction code (e.g.
// DiffuseAreaLight::Create, lights.cpp): the blackbody spectrum is
// normalized so its own photometric integral (InnerProduct against the CIE
// Y curve - matches pbrt-v4's SpectrumToPhotometric exactly, NOT the same
// as SpectrumToXYZ's Y, which additionally divides by CIE_Y_integral) comes
// out to ~1 nit, BEFORE any separate "float scale" the light also specifies
// - that scale is applied afterward by the existing `L * scale` multiply
// every consumer (CPU/GPU builder) already does downstream, unchanged, so
// this function's return value slots into that exact same pattern.
// Converts a single Kelvin temperature to sRGB via this codebase's own
// already-ported pbrt-v4 spectral pipeline - the same normalize-to-~1-nit-
// photometric-integral technique resolveEmissionColor() (just below) uses
// for a scene's "blackbody L"/"I" param, factored out here so a per-voxel
// caller (pbrt_cpu_builder.h's nanovdb "temperaturename" grid bake, the
// motivating case - see Medium::nanovdbTemperatureGridName's own comment)
// doesn't need a fake single-number ParamList to reuse it.
inline pbrt_scene::Vec3 blackbodyKelvinToRGB(float T,
                                              const RGBColorSpace &colorSpace = RGBColorSpace::sRGB()) {
	if (T <= 0.0f) return pbrt_scene::Vec3{0.0, 0.0, 0.0};
	const BlackbodySpectrum bb(T);
	const XYZ xyz = SpectrumToXYZ(bb);
	const float photometric = InnerProduct(GetCIE_Y(), bb);
	const float norm = (photometric > 0.0f) ? (1.0f / photometric) : 0.0f;
	float r, g, b;
	colorSpace.FromXYZ(xyz.X * norm, xyz.Y * norm, xyz.Z * norm, r, g, b);
	// Small negative components are possible near the edge of the sRGB
	// gamut even for a physically real source - clamp rather than let a
	// negative emission subtract light, matching every other colour path in
	// this loader's own convention of clamping at the edges.
	return pbrt_scene::Vec3{ std::fmax(0.0, static_cast<double>(r)),
							  std::fmax(0.0, static_cast<double>(g)),
							  std::fmax(0.0, static_cast<double>(b)) };
}

inline pbrt_scene::Vec3 resolveEmissionColor(const pbrt_scene::ParamList &params,
                                              const char *name, pbrt_scene::Vec3 def,
                                              const RGBColorSpace &colorSpace = RGBColorSpace::sRGB()) {
	const pbrt_scene::Param *p = params.find(name);
	if (p && p->type == "blackbody" && !p->numbers.empty()) {
		return blackbodyKelvinToRGB(static_cast<float>(p->numbers[0]), colorSpace);
	}
	return params.getVec3(name, def);
}


}  // namespace pbrt_flatten
