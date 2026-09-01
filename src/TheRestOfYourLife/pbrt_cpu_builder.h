#pragma once
// pbrt_cpu_builder.h -- builds CPU hittables from a flattened pbrt scene.
//
// This is the first file in the pbrt chain that knows about renderer types.
// pbrt_scene.h (text -> description), ply_mesh.h (mesh bytes -> arrays) and
// pbrt_flatten.h (description -> world-space geometry) are all deliberately
// free of both Qt and hittable/material, so the MSVC test binary can reach
// them. Everything renderer-specific lives here and in the eventual GPU
// counterpart, which consume the same FlatScene.

#include <cmath>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "../shared/pbrt_flatten.h"
#include "../shared/portal_image_infinite_light.h"   // PortalImageInfiniteLightData<double> - LightSource "infinite" "point3 portal[4]"
#include "../external/tinyexr.h"   // LoadEXR() - see decodePunctualLightImageFile()

#include "bvh.h"
#include "constant_medium.h"
#include "curve_shape_hittable.h"
#include "disk_cylinder_hittable.h"
#include "cone_paraboloid_hittable.h"
#include "grid_medium_hittable.h"
#include "hair_material.h"
#include "hittable_list.h"
#include "material.h"
#include "rtw_stb_image.h"     // stbi_load() - see alphaMaskFor()'s own comment
#include "scenes_advanced.h"   // bilinear_patch_hittable
#include "sphere_clipped_hittable.h"
#include "sky_light.h"
#include "sphere.h"
#include "triangle.h"
#include "transform_instance.h"

namespace pbrt_cpu {

namespace detail {

// pbrt-v4 lets a conductor be given directly as a reflectance colour instead
// of measured eta/k spectra (its own ConductorMaterial does exactly this
// conversion when "reflectance" is bound rather than "eta"/"k"): an eta of 1
// paired with k solved from the normal-incidence Schlick reflectance
// r = ((eta-1)^2+k^2) / ((eta+1)^2+k^2), which at eta=1 reduces to
// k = 2*sqrt(r) / sqrt(max(eps, 1-r)). CoatedConductor's own eta/k
// sub-parameters ("conductor.eta"/"conductor.k") are a real pbrt-v4 syntax
// this parser does not read yet (see pbrt_flatten.h's generic, unprefixed
// "reflectance"/"k" lookup), so this conversion is the same approximation
// tier plain Conductor already accepts here rather than a new one - a
// coated metal rendered from its base colour instead of exact spectra,
// clearly better than the flat Lambertian this used to fall back to.
inline color reflectanceToConductorK(const color& r) {
	const auto k1 = [](double x) {
		x = x < 0.0 ? 0.0 : (x > 0.9999 ? 0.9999 : x);
		return 2.0 * std::sqrt(x) / std::sqrt(std::fmax(1e-4, 1.0 - x));
	};
	return color(k1(r.x()), k1(r.y()), k1(r.z()));
}

// A checkerboard/mix texture's own tex1/tex2 slot: a bare imagemap filename
// (one-level-nested texture reference already resolved by pbrt_flatten.h -
// see Material::checkerTex1Filename/mixTex1Filename's own comments) when
// present, otherwise the flat literal RGB colour flatten() resolved into the
// paired colour array. Shared by hasCheckerReflectance's and
// hasMixReflectance's own tex1/tex2 slots, on both the Diffuse and (since
// CoatedDiffuse gained the same procedural-texture support) CoatedDiffuse
// cases below - four call sites for what used to be identical inline logic.
inline shared_ptr<texture> checkerOrMixSlot(const std::string &filename, const double color3[3]) {
	return filename.empty()
		? std::static_pointer_cast<texture>(std::make_shared<solid_color>(color(color3[0], color3[1], color3[2])))
		: std::static_pointer_cast<texture>(std::make_shared<mipmap_texture>(filename.c_str()));
}

// Builds the MipMapOptions for m.textureFilename specifically - see
// Material::textureGamma's own comment (pbrt_flatten.h) for why this slot
// keeps its own 3 loose fields instead of a TextureDecodeOptions like
// transmittance/roughness now use (toMipMapOptions() just below). Every
// OTHER mipmap_texture construction in this file that has no corresponding
// pbrt_flatten::Material options field at all (mix-amount/checker/emission/
// etc.) still deliberately keeps using the default-constructed
// MipMapOptions{} (gamma 2.2, Clamp, no invert) - unchanged.
inline MipMapOptions imageMapOptionsFor(const pbrt_flatten::Material &m) {
	MipMapOptions opts;
	opts.gamma = m.textureGamma;
	opts.invert = m.textureInvert;
	// m.textureWrapIndex is the single, already-validated resolution of
	// m.textureWrap (Material::textureWrapIndex's own comment, pbrt_flatten.h)
	// - MipWrapMode's own enumerator values (Clamp=0/Repeat=1/Black=2) match
	// it by construction, so this just recovers the enum rather than
	// re-parsing the string a second time.
	opts.wrap = static_cast<MipWrapMode>(m.textureWrapIndex);
	return opts;
}

// Same conversion as imageMapOptionsFor() above, for a slot that carries its
// own pbrt_flatten::TextureDecodeOptions (transmittance/roughness) instead
// of 3 loose Material fields - see that struct's own comment.
inline MipMapOptions toMipMapOptions(const pbrt_flatten::TextureDecodeOptions &o) {
	MipMapOptions opts;
	opts.gamma = static_cast<float>(o.gamma);
	opts.invert = o.invert;
	opts.wrap = static_cast<MipWrapMode>(o.wrapIndex);
	return opts;
}

// pbrt's material names already match ours (see pbrt_flatten.h), so this is
// construction rather than interpretation. An Unsupported material becomes
// diffuse - flatten() has already warned about it by name, so failing here
// would only turn a documented approximation into a refusal to open the file.
// `allMaterials`/`depth` are only touched by the Mix case below (resolving
// its two named sub-materials, which needs the full parallel-to-out.materials
// list flatten() built - see pbrt_flatten::Material::mixMaterialA/B's own
// comment on why those are indices into that list rather than something
// self-contained); every other material kind ignores them, so a default
// empty vector is fine for every existing call site.
//
// `forCurve` is read only by the Hair case (see hair_material's own
// tangentIsDpdu parameter comment) - every other kind ignores it. Threaded
// through the Mix case's own recursive calls so a Mix containing a Hair
// sub-material still gets the right tangent behavior when applied to real
// curve geometry.
inline std::shared_ptr<material> makeMaterial(const pbrt_flatten::Material &m,
											  const pbrt_flatten::Emission *emission,
											  const std::vector<pbrt_flatten::Material> &allMaterials = {},
											  int depth = 0,
											  bool forCurve = false) {
	// Emission wins: in pbrt an AreaLightSource attaches to the shape, and its
	// material describes what the surface does with light arriving at it. Our
	// diffuse_light is the emissive case, so an emissive shape becomes one
	// regardless of the material it also declared.
	if (emission) {
		// A "filename" area light wins over "L" entirely (matches pbrt-v4's
		// own DiffuseAreaLight - see Emission::filename's own comment),
		// still honoring "scale" via scaled_texture since a filename-backed
		// light never touches pbrt_flatten::Material::color the way a
		// flat-L light implicitly could. point_sample=true matches pbrt-v4's
		// own plain (non-EWA) image emission lookup - see diffuse_light's
		// own comment on that flag.
		if (!emission->filename.empty()) {
			shared_ptr<texture> tex = std::make_shared<mipmap_texture>(emission->filename.c_str());
			if (emission->scale != 1.0)
				tex = std::make_shared<scaled_texture>(tex, emission->scale);
			return std::make_shared<diffuse_light>(tex, emission->twoSided, /*point_sample=*/true);
		}
		const color L(emission->L[0] * emission->scale,
					  emission->L[1] * emission->scale,
					  emission->L[2] * emission->scale);
		return std::make_shared<diffuse_light>(L, emission->twoSided);
	}

	const color albedo(m.color[0], m.color[1], m.color[2]);
	switch (m.kind) {
	case pbrt_flatten::MaterialKind::Conductor:
		// A recognized named conductor spectrum ("metal-Ag-eta"/"metal-Ag-k"
		// etc. - see pbrt_flatten.h's conductorElementFromSpectrumName() and
		// src/shared/conductor_data.h's own table) gets the real GGX +
		// complex-Fresnel model (`conductor`, matching this codebase's
		// native scenes' own B5/B7 - #229/#230's real-NEE work applies here
		// too). Anything else (explicit RGB k, or an unrecognized/non-metal
		// named spectrum) keeps the pre-existing approximation: our `metal`
		// takes a flat albedo and a fuzz, so roughness maps onto fuzz
		// directly.
		if (m.hasConductorPreset)
			return std::make_shared<conductor>(
				m.conductorEta[0], m.conductorEta[1], m.conductorEta[2],
				m.conductorK[0], m.conductorK[1], m.conductorK[2],
				m.roughness_u, m.roughness_v, m.remapRoughness);
		return std::make_shared<metal>(albedo, m.roughness);
	// Pass-through "interface" material (pbrt-v4's Material "none"/"" -
	// see MaterialKind::Interface's own comment and interface_material's
	// own comment, material_simple.h). A real dedicated class, not routed
	// through Dielectric - no Fresnel/refraction math at all, so no
	// critical angle, and it carries a genuine "nothing happened here"
	// signal (scatter_record::is_medium_boundary) the integrators use to
	// preserve MIS state and skip the bounce budget across the crossing.
	case pbrt_flatten::MaterialKind::Interface:
		return std::make_shared<interface_material>();
	case pbrt_flatten::MaterialKind::Dielectric:
		// m.roughnessTextureFilename (Material::roughnessTextureFilename
		// own comment) - checked FIRST since a texture-bound roughness
		// takes priority over m.roughness_u/m.roughness_v below (which
		// stay at their 0.0 default when the scene bound a texture
		// instead of a flat number - see flatten()'s own resolution).
		if (!m.roughnessTextureFilename.empty())
			return std::make_shared<rough_dielectric>(m.ior,
				std::make_shared<mipmap_texture>(m.roughnessTextureFilename.c_str(),
					toMipMapOptions(m.roughnessTextureOptions)), m.remapRoughness);
		// A nonzero "roughness"/"uroughness"/"vroughness" (m.roughness_u/
		// m.roughness_v - see flatten()'s own fallback-chain comment) means
		// the scene asked for a GGX microfacet dielectric (pbrt-v4
		// DielectricBxDF's rough path), not a perfect-specular one - this
		// codebase already has a real model for that (rough_dielectric,
		// RoughDielectricBxDF), it just wasn't wired up here, so any scene
		// with a rough glass/window silently rendered as a perfect
		// mirror-and-refract surface instead. Round 6 Phase 3: independent
		// u/v roughness (anisotropic GGX) instead of the single collapsed
		// value.
		if (m.roughness_u > 0.0 || m.roughness_v > 0.0)
			return std::make_shared<rough_dielectric>(m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		return std::make_shared<dielectric>(m.ior);
	case pbrt_flatten::MaterialKind::ThinDielectric:
		// A zero-thickness slab (thin_dielectric, material_pbrt.h) is NOT the
		// same BxDF as a solid dielectric volume: the transmitted ray exits
		// un-refracted on the same side it entered, and R/T are the
		// closed-form internal-bounce sums (ThinDielectricBxDF), not Snell's
		// law - this used to fall through to plain `dielectric`, refracting a
		// window pane/soap bubble as if it had real thickness and an interior
		// (visibly wrong bending, and no gpu_thin_dielectric_material parity
		// with the GPU backend's MaterialType::ThinDielectric either).
		return std::make_shared<thin_dielectric>(m.ior);
	case pbrt_flatten::MaterialKind::CoatedDiffuse:
		// m.textureFilename (Material::textureFilename's own comment) - same
		// resolved-by-pbrt_load.h convention the Diffuse case below already
		// relies on. m.textureScale (default 1.0, a no-op) wraps a
		// scale-class Texture's own multiplier when reflectance was bound to
		// one wrapping an imagemap (barcelona-pavilion's dominant pattern) -
		// scaled_texture has a real value_diff() override forwarding to the
		// inner mipmap_texture's own EWA filtering (see that class's own
		// header comment), so a scale-wrapped coateddiffuse texture keeps
		// real mip-level filtering under minification, not just its
		// original AreaLightSource caller's point-sampled use.
		if (!m.textureFilename.empty()) {
			shared_ptr<texture> tex = std::make_shared<mipmap_texture>(m.textureFilename.c_str(), imageMapOptionsFor(m));
			if (m.textureScale != 1.0)
				tex = std::make_shared<scaled_texture>(tex, m.textureScale);
			return std::make_shared<coated_diffuse>(tex, m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		// m.hasCheckerReflectance/hasFbmReflectance/hasMarbleReflectance/
		// hasMixReflectance (Material's own comments) - same procedural-not-
		// file pattern as the Diffuse case below, now also resolved for
		// CoatedDiffuse (previously Diffuse-only): the identical resolved
		// fields, just plugged into coated_diffuse's own texture-taking
		// constructor instead of lambertian's.
		if (m.hasCheckerReflectance) {
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.checkerTex1Filename, m.checkerColor1);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.checkerTex2Filename, m.checkerColor2);
			return std::make_shared<coated_diffuse>(
				std::make_shared<uv_checker_texture>(m.checkerUScale, m.checkerVScale, tex1, tex2),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		if (m.hasFbmReflectance)
			return std::make_shared<coated_diffuse>(
				std::make_shared<fbm_texture>(1.0, m.fbmOctaves, m.fbmRoughness),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		if (m.hasMarbleReflectance)
			return std::make_shared<coated_diffuse>(
				std::make_shared<marble_texture>(m.marbleScale, m.marbleOctaves, m.marbleRoughness, m.marbleVariation),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		if (m.hasMixReflectance) {
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.mixTex1Filename, m.mixColor1);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.mixTex2Filename, m.mixColor2);
			// m.mixAmountTextureFilename (Material::mixAmountTextureFilename's
			// own comment) - a real per-point spatially-varying blend when
			// "amount" itself nested a bare imagemap, via mix_texture's own
			// texture-taking amount constructor; the flat-scalar constructor
			// otherwise.
			shared_ptr<texture> mix = m.mixAmountTextureFilename.empty()
				? std::make_shared<mix_texture>(tex1, tex2, m.mixAmount)
				: std::make_shared<mix_texture>(tex1, tex2,
					std::make_shared<mipmap_texture>(m.mixAmountTextureFilename.c_str()));
			return std::make_shared<coated_diffuse>(mix, m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		if (m.hasWindyReflectance)
			return std::make_shared<coated_diffuse>(
				std::make_shared<windy_texture>(), m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		if (m.hasWrinkledReflectance)
			return std::make_shared<coated_diffuse>(
				std::make_shared<wrinkled_texture>(m.wrinkledOctaves, m.wrinkledRoughness),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		if (m.hasDotsReflectance) {
			shared_ptr<texture> insideTex = checkerOrMixSlot(m.dotsInsideTexFilename, m.dotsInsideColor);
			shared_ptr<texture> outsideTex = checkerOrMixSlot(m.dotsOutsideTexFilename, m.dotsOutsideColor);
			return std::make_shared<coated_diffuse>(
				std::make_shared<dots_texture>(insideTex, outsideTex),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		if (m.hasBilerpReflectance) {
			const color v00(m.bilerpV00[0], m.bilerpV00[1], m.bilerpV00[2]);
			const color v01(m.bilerpV01[0], m.bilerpV01[1], m.bilerpV01[2]);
			const color v10(m.bilerpV10[0], m.bilerpV10[1], m.bilerpV10[2]);
			const color v11(m.bilerpV11[0], m.bilerpV11[1], m.bilerpV11[2]);
			return std::make_shared<coated_diffuse>(
				std::make_shared<bilerp_texture>(v00, v01, v10, v11),
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		return std::make_shared<coated_diffuse>(albedo, m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
	case pbrt_flatten::MaterialKind::CoatedConductor: {
		// A recognized named conductor spectrum or an explicit "rgb eta"/
		// "rgb k" (m.hasConductorPreset - see flatten()'s own Conductor-OR-
		// CoatedConductor branch) gets the real complex-IOR model, same as
		// the Conductor case above; "nothing given"/an unrecognized case
		// keeps the pre-existing reflectanceToConductorK() approximation.
		if (m.hasConductorPreset)
			return std::make_shared<coated_conductor>(
				m.conductorEta[0], m.conductorEta[1], m.conductorEta[2],
				m.conductorK[0], m.conductorK[1], m.conductorK[2],
				m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		const color k = reflectanceToConductorK(albedo);
		return std::make_shared<coated_conductor>(
			1.0, 1.0, 1.0, k.x(), k.y(), k.z(), m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
	}
	case pbrt_flatten::MaterialKind::DiffuseTransmission: {
		// m.textureFilename/m.transmittanceTextureFilename (own comments in
		// pbrt_flatten.h) - barcelona-pavilion's foliage binds both
		// "reflectance" and "transmittance" to the SAME bare imagemap; each
		// optionally further wrapped in its own independent "scale" texture
		// (m.textureScale/m.transmittanceTextureScale's own comments),
		// same scaled_texture-wrap pattern as CoatedDiffuse's identical-
		// shape case above.
		const color transmittance(m.transmittance[0], m.transmittance[1], m.transmittance[2]);
		if (m.textureFilename.empty() && m.transmittanceTextureFilename.empty())
			return std::make_shared<diffuse_transmission>(albedo, transmittance);
		shared_ptr<texture> rTex = m.textureFilename.empty()
			? nullptr : std::make_shared<mipmap_texture>(m.textureFilename.c_str(), imageMapOptionsFor(m));
		if (rTex && m.textureScale != 1.0)
			rTex = std::make_shared<scaled_texture>(rTex, m.textureScale);
		shared_ptr<texture> tTex = m.transmittanceTextureFilename.empty()
			? nullptr : std::make_shared<mipmap_texture>(m.transmittanceTextureFilename.c_str(),
				toMipMapOptions(m.transmittanceTextureOptions));
		if (tTex && m.transmittanceTextureScale != 1.0)
			tTex = std::make_shared<scaled_texture>(tTex, m.transmittanceTextureScale);
		return std::make_shared<diffuse_transmission>(albedo, transmittance, rTex, tTex);
	}
	case pbrt_flatten::MaterialKind::Subsurface:
		return std::make_shared<subsurface>(m.ior, m.sigma_a, m.sigma_s, m.g);
	case pbrt_flatten::MaterialKind::Hair:
		return std::make_shared<hair_material>(
			m.sigma_a[0], m.sigma_a[1], m.sigma_a[2],
			m.betaM, m.betaN, m.alphaDeg, m.ior, forCurve);
	case pbrt_flatten::MaterialKind::Measured: {
		// m.measuredFilename is empty unless pbrt_load.h's post-flatten pass
		// both resolved AND successfully load-tested it (see pbrt_flatten.h's
		// Material::measuredFilename comment) - so an empty filename here
		// means "already warned about, fall back to diffuse", same as
		// Unsupported below. A non-empty filename means `measured`'s own
		// constructor is doing a cache hit, not a fresh multi-megabyte parse.
		if (m.measuredFilename.empty())
			break;
		auto mat = std::make_shared<measured>(m.measuredFilename);
		// Guards against the theoretically-possible case of the file having
		// become unreadable between pbrt_load.h's validation pass and here
		// (both happen back-to-back during scene loading, so this is belt-
		// and-suspenders, not an expected path) - a `measured` that failed
		// to load can only ever return false from scatter(), which would
		// render the surface pure black rather than the documented
		// diffuse-approximation fallback every other unsupported/failed
		// material gets.
		if (!mat->loaded())
			break;
		return mat;
	}
	case pbrt_flatten::MaterialKind::Mix: {
		// Real recursive resolution, not a documented approximation - see
		// MaterialKind::Mix's own comment (pbrt_flatten.h) for why this is
		// backed by the existing, generic `class mix_material`
		// (material_pbrt.h) rather than a new one. flatten() already
		// downgraded an unresolvable mix to Unsupported (see there), so
		// mixMaterialA/B are valid indices into allMaterials whenever this
		// case is reached with a non-empty allMaterials - the emptiness/
		// depth checks below only guard the pathological cases (a stray
		// direct call with the defaulted empty vector, or a cyclic/self-
		// referential "materials" list a malformed scene could produce,
		// neither of which this loader's own corpus has ever needed).
		constexpr int kMaxMixDepth = 8;
		if (depth >= kMaxMixDepth
			|| m.mixMaterialA < 0 || static_cast<std::size_t>(m.mixMaterialA) >= allMaterials.size()
			|| m.mixMaterialB < 0 || static_cast<std::size_t>(m.mixMaterialB) >= allMaterials.size())
			break;
		// Sub-materials of a mix are never emissive on their own in pbrt-v4
		// (AreaLightSource attaches to the SHAPE, handled generically by the
		// `emission` check at the top of this function - reached before this
		// switch runs at all when the shape is emissive, so this recursive
		// call never needs to pass one through).
		auto matA = makeMaterial(allMaterials[static_cast<std::size_t>(m.mixMaterialA)],
								 nullptr, allMaterials, depth + 1, forCurve);
		auto matB = makeMaterial(allMaterials[static_cast<std::size_t>(m.mixMaterialB)],
								 nullptr, allMaterials, depth + 1, forCurve);
		return std::make_shared<mix_material>(matA, matB, m.mixWeight);
	}
	case pbrt_flatten::MaterialKind::Diffuse:
		// m.textureFilename is only ever non-empty after pbrt_load.h's post-
		// flatten pass confirmed the file exists (Material::textureFilename's
		// own comment) - mirrors mesh.h's load_obj_mtl() map_Kd path: decode
		// once, hand the pixels straight to a mipmap_texture-backed
		// lambertian instead of the flat-colour one below. A corrupt-but-
		// present file (mip_ stays null) degrades to mipmap_texture's own
		// cyan debug colour rather than a silent flat-colour fallback - rare
		// enough (pbrt_load.h already validated the file opens) not to be
		// worth a second probe-and-fallback dance here. m.textureScale
		// (default 1.0, a no-op) wraps a "scale"-class Texture's own
		// multiplier when reflectance was bound to one wrapping an imagemap
		// (barcelona-pavilion's own dominant pattern, same as CoatedDiffuse's
		// identical-shape case below).
		if (!m.textureFilename.empty()) {
			shared_ptr<texture> tex = std::make_shared<mipmap_texture>(m.textureFilename.c_str(), imageMapOptionsFor(m));
			if (m.textureScale != 1.0)
				tex = std::make_shared<scaled_texture>(tex, m.textureScale);
			return std::make_shared<lambertian>(tex);
		}
		// m.hasCheckerReflectance (Material::hasCheckerReflectance's own
		// comment) - a procedural pbrt-v4 checkerboard, not an image file,
		// so uv_checker_texture (texture.h) is built directly from the
		// resolved colours/scales rather than decoded from disk. tex1/tex2
		// each independently use uv_checker_texture's own polymorphic
		// constructor when checkerTex1Filename/checkerTex2Filename named a
		// one-level-nested bare imagemap instead of a flat literal (see that
		// field's own comment) - a mipmap_texture per nested slot instead of
		// the flat solid_color the plain literal case still uses.
		if (m.hasCheckerReflectance) {
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.checkerTex1Filename, m.checkerColor1);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.checkerTex2Filename, m.checkerColor2);
			return std::make_shared<lambertian>(std::make_shared<uv_checker_texture>(
				m.checkerUScale, m.checkerVScale, tex1, tex2));
		}
		// m.hasFbmReflectance/hasMarbleReflectance/hasMixReflectance
		// (Material's own comments) - same procedural-not-file pattern as
		// hasCheckerReflectance above, resolving to the existing fbm_
		// texture/marble_texture CPU classes (texture.h) or the new
		// mix_texture. No separate world-space "scale" param exists for
		// pbrt-v4's real FBmTexture (only octaves/roughness), so 1.0 (no
		// extra scaling beyond the world-space point itself) is passed for
		// fbm_texture's own scale argument - matches pbrt-v4 semantics.
		if (m.hasFbmReflectance)
			return std::make_shared<lambertian>(std::make_shared<fbm_texture>(
				1.0, m.fbmOctaves, m.fbmRoughness));
		if (m.hasMarbleReflectance)
			return std::make_shared<lambertian>(std::make_shared<marble_texture>(
				m.marbleScale, m.marbleOctaves, m.marbleRoughness, m.marbleVariation));
		if (m.hasMixReflectance) {
			// Same one-level-nested-imagemap support as checkerboard above,
			// via mix_texture's own new polymorphic constructor.
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.mixTex1Filename, m.mixColor1);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.mixTex2Filename, m.mixColor2);
			// m.mixAmountTextureFilename (Material::mixAmountTextureFilename's
			// own comment) - a real per-point spatially-varying blend when
			// "amount" itself nested a bare imagemap.
			shared_ptr<texture> mix = m.mixAmountTextureFilename.empty()
				? std::make_shared<mix_texture>(tex1, tex2, m.mixAmount)
				: std::make_shared<mix_texture>(tex1, tex2,
					std::make_shared<mipmap_texture>(m.mixAmountTextureFilename.c_str()));
			return std::make_shared<lambertian>(mix);
		}
		if (m.hasWindyReflectance)
			return std::make_shared<lambertian>(std::make_shared<windy_texture>());
		if (m.hasWrinkledReflectance)
			return std::make_shared<lambertian>(std::make_shared<wrinkled_texture>(
				m.wrinkledOctaves, m.wrinkledRoughness));
		if (m.hasDotsReflectance) {
			shared_ptr<texture> insideTex = checkerOrMixSlot(m.dotsInsideTexFilename, m.dotsInsideColor);
			shared_ptr<texture> outsideTex = checkerOrMixSlot(m.dotsOutsideTexFilename, m.dotsOutsideColor);
			return std::make_shared<lambertian>(std::make_shared<dots_texture>(insideTex, outsideTex));
		}
		if (m.hasBilerpReflectance) {
			const color v00(m.bilerpV00[0], m.bilerpV00[1], m.bilerpV00[2]);
			const color v01(m.bilerpV01[0], m.bilerpV01[1], m.bilerpV01[2]);
			const color v10(m.bilerpV10[0], m.bilerpV10[1], m.bilerpV10[2]);
			const color v11(m.bilerpV11[0], m.bilerpV11[1], m.bilerpV11[2]);
			return std::make_shared<lambertian>(std::make_shared<bilerp_texture>(v00, v01, v10, v11));
		}
		break;
	case pbrt_flatten::MaterialKind::Unsupported:
		break;
	}
	return std::make_shared<lambertian>(albedo);
}

// Key for restoring vertex sharing. FlatScene stores each triangle's three
// vertices explicitly, which is convenient to test but triples the vertex
// count on a real mesh where most vertices are shared by six faces. The
// positions came from transforming the same source vertex, so equal vertices
// are bitwise equal and an exact-match dedup recovers the original count -
// worth doing when the target is scenes with millions of triangles.
// The shading normal is part of the key, not just the position. Two faces can
// legitimately meet at the same point with different normals - that is exactly
// how a crease is expressed - and merging them into one vertex would smooth
// the edge away. Deduping on position alone is only correct when there are no
// shading normals at all, which is no longer the case.
struct VertexKey {
	double x, y, z;
	double nx, ny, nz;
	// UV joins the dedup key for the same reason normals do: a real mesh can
	// have a UV seam at a position it shares with a differently-textured
	// neighbor (matching a hard-normal edge's own reason to NOT merge those
	// vertices), so two otherwise-identical positions with different UV must
	// stay distinct vertices too.
	double u, v;
	bool operator<(const VertexKey &o) const {
		if (x != o.x) return x < o.x;
		if (y != o.y) return y < o.y;
		if (z != o.z) return z < o.z;
		if (nx != o.nx) return nx < o.nx;
		if (ny != o.ny) return ny < o.ny;
		if (nz != o.nz) return nz < o.nz;
		if (u != o.u) return u < o.u;
		return v < o.v;
	}
};

} // namespace detail

struct BuildResult {
	std::shared_ptr<hittable_list> world;
	std::shared_ptr<hittable_list> lights;   // emissive shapes, for NEE
	std::shared_ptr<sky_light> sky;          // null if the scene has no infinite light
	// LightSource "infinite" "point3 portal[4]" (pbrt-v4's windowed infinite
	// light - visible only through a finite rectangular window instead of
	// the whole sphere) - null unless the scene's infinite light declared a
	// real portal[4]. Mutually exclusive with `sky` above (see the
	// sky-construction block below): a portal light needs no separate `sky`
	// fallback, since PortalImageInfiniteLightData::eval_Le()/sample_li()
	// already return zero/false outside the window on their own - callers
	// (camera.h) branch on `portal` first, falling through to `sky` only
	// when this is null.
	std::shared_ptr<PortalImageInfiniteLightData<double>> portal;
	// LightSource point/spot/distant/goniometric/projection - null if the
	// scene has none (matches camera_t::punct_lights' own "nullptr = none"
	// convention, which this feeds directly - see scene_registry.h's
	// build_punct wiring). Never empty-but-non-null: left null unless
	// scene.punctualLights actually held something, same as `sky` above.
	std::shared_ptr<punctual_light_list> punctLights;
	// pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own
	// comment, pbrt_flatten.h) - null unless the scene declared a
	// MediumInterface before its Camera directive AND flatten() resolved
	// it (homogeneous only, not combined with a real per-shape medium -
	// see that field's own comment for the two scope cuts). Feeds
	// camera_t::camera_medium (camera.h) directly, same "null = none, no
	// existing scene's behavior changes" convention as `sky`/`portal` above.
	std::shared_ptr<ambient_medium> cameraMedium;
	std::size_t triangleCount = 0;
	std::size_t sphereCount = 0;
	std::size_t diskCount = 0;
	std::size_t cylinderCount = 0;
	std::size_t coneCount = 0;
	std::size_t paraboloidCount = 0;
	std::size_t bilinearPatchCount = 0;
	std::size_t curveCount = 0;
	std::size_t uniqueVertexCount = 0;
	// Instance placements actually added to the world. Each shares one
	// BVH with every other placement of the same definition.
	std::size_t instanceCount = 0;
};

// Decodes a resolved (existing) image file for a goniometric/projection
// punctual light's "filename" into an rtw_image, EXR-aware - rtw_image's own
// load() only understands stb_image's formats (PNG/JPG/BMP/HDR/...), not EXR
// (see its raw-pixel constructor's own comment), so a real IES-derived
// equal-area profile (commonly distributed as EXR, matching pbrt-v4's own
// `imgtool makeequiarea` output) needs tinyexr decoded here first. Mirrors
// pbrt_load.h's decodeInfiniteLightImage() dispatch exactly (extension-only,
// not content-sniffed), just returning an rtw_image instead of a raw
// std::vector<float> since this file's own callers want pixel_data()/
// float_pixel_data() access, not a buffer to hand off further. Returns an
// empty (width()==0) rtw_image on any decode failure - callers already
// treat that as "no real image" identically to a missing file.
inline rtw_image decodePunctualLightImageFile(const std::string &resolvedPath) {
	if (resolvedPath.size() >= 4 &&
		resolvedPath.compare(resolvedPath.size() - 4, 4, ".exr") == 0) {
		float *rgba = nullptr;
		int w = 0, h = 0;
		const char *err = nullptr;
		const int rc = LoadEXR(&rgba, &w, &h, resolvedPath.c_str(), &err);
		if (err) FreeEXRErrorMessage(err);
		if (rc != TINYEXR_SUCCESS || !rgba || w <= 0 || h <= 0) {
			if (rgba) free(rgba);
			return rtw_image();
		}
		std::vector<float> rgb(static_cast<std::size_t>(w) * h * 3);
		for (int i = 0; i < w * h; ++i) {
			rgb[i * 3 + 0] = rgba[i * 4 + 0];
			rgb[i * 3 + 1] = rgba[i * 4 + 1];
			rgb[i * 3 + 2] = rgba[i * 4 + 2];
		}
		free(rgba);
		return rtw_image(w, h, rgb.data());
	}
	return rtw_image(resolvedPath.c_str());
}

// Sphere/Disk/Cylinder each carry their object-to-world transform as a flat
// double[16] (pbrt_flatten.h) rather than pbrt_scene::Matrix4 directly, since
// that header is deliberately kept free of renderer types (see its own file
// comment) - this converts back at the one point each of the 3 shape kinds
// below needs a real Matrix4 to hand to its *_hittable constructor.
inline pbrt_scene::Matrix4 toMatrix4(const double (&m)[16]) {
	pbrt_scene::Matrix4 xform;
	for (int i = 0; i < 16; ++i) xform.m[i] = m[i];
	return xform;
}

// Turns flattened geometry into a BVH-accelerated world plus the light list
// the integrator samples. Materials are created once per (material, emission)
// pair rather than per primitive - a million-triangle mesh with one material
// should hold one material object, not a million.
inline BuildResult build(const pbrt_flatten::FlatScene &scene) {
	using namespace detail;
	BuildResult out;
	out.world = std::make_shared<hittable_list>();
	out.lights = std::make_shared<hittable_list>();


	// forCurve: see makeMaterial's own comment - only affects the Hair case,
	// picked by the one caller (the curve loop below) that actually has real
	// curve geometry under the resolved material.
	const auto materialFor = [&scene](int materialIndex, int areaLightIndex,
									   bool forCurve = false)
			-> std::shared_ptr<material> {
		const pbrt_flatten::Emission *em =
			(areaLightIndex >= 0 && static_cast<std::size_t>(areaLightIndex) < scene.areaLights.size())
				? &scene.areaLights[static_cast<std::size_t>(areaLightIndex)]
				: nullptr;
		static const pbrt_flatten::Material kDefault{};
		const pbrt_flatten::Material &m =
			(materialIndex >= 0 && static_cast<std::size_t>(materialIndex) < scene.materials.size())
				? scene.materials[static_cast<std::size_t>(materialIndex)]
				: kDefault;
		std::shared_ptr<material> base = makeMaterial(m, em, scene.materials, /*depth=*/0, forCurve);

		// Material "texture displacement" (bump mapping - Material::
		// displacementTextureFilename's own comment). Wraps whatever base
		// material was just built, mirroring mesh.h's own OBJ/MTL map_Bump
		// dispatch exactly: skipped for emissive materials (perturbing an
		// emitter's normal has no meaningful effect), classified grayscale-
		// vs-RGB by real pixel content (is_grayscale_image(), not filename)
		// since a real scene's displacement image could in principle be
		// either, even though every bundled pbrt scene's own "*bump*.png"
		// naming is grayscale in practice.
		if (base && !em && !m.displacementTextureFilename.empty()) {
			rtw_image disp_probe(m.displacementTextureFilename.c_str());
			if (disp_probe.height() > 0) {
				const bool grayscale = is_grayscale_image(disp_probe);
				auto disp_tex = std::make_shared<image_texture>(std::move(disp_probe));
				base = grayscale
					? std::static_pointer_cast<material>(
						  std::make_shared<bump_map_material>(disp_tex, base, m.displacementScale))
					: std::static_pointer_cast<material>(
						  std::make_shared<normal_map_material>(disp_tex, base));
			}
		}
		return base;
	};

	// One material instance per distinct (material, emission, forCurve) triple
	// - forCurve is part of the key (not just an argument materialFor reads)
	// so a materialIndex shared between a curve and a non-curve shape (e.g.
	// via NamedMaterial reuse - unusual but valid pbrt) gets two distinct
	// hair_material instances with the right tangent behavior each, instead
	// of whichever shape asks first silently winning for both.
	std::map<std::tuple<int, int, bool>, std::shared_ptr<material>> materialCache;
	const auto cachedMaterial = [&](int mi, int ai, bool forCurve = false) {
		const auto key = std::make_tuple(mi, ai, forCurve);
		auto it = materialCache.find(key);
		if (it != materialCache.end()) return it->second;
		auto made = materialFor(mi, ai, forCurve);
		materialCache.emplace(key, made);
		return made;
	};

	// A pbrt Shape "alpha" cutout mask (Material::alphaTextureFilename - see
	// that field's own comment: attached to the Shape's own resolved
	// material, one entry per unique materialIndex). image_texture rather
	// than mipmap_texture: an alpha-cutout test only ever needs a single
	// point sample (triangle::hit()'s alpha test), never mip filtering.
	// nullptr (the default) for every material with no alpha texture,
	// matching triangle's own zero-cost default.
	//
	// Deliberately NOT rtw_image's own load() (which calls stbi_loadf() -
	// see OBJ/MTL's map_d handling in mesh.h for that same pattern): for an
	// 8-bit/LDR source image, stbi_loadf silently applies stb_image's
	// default gamma-2.2 decode (its "LDR-to-HDR" conversion, meant for
	// colour data going sRGB -> linear). An alpha/opacity mask is a linear
	// coverage fraction, not a display colour, so that decode would
	// systematically bias the cutout threshold (e.g. an authored 0.6 alpha,
	// byte 153/255, decodes to pow(0.6, 2.2) =~ 0.32 and silently flips
	// which side of triangle.h's kAlphaCutoutThreshold it falls on).
	// stbi_load() (the plain 8-bit loader - no float conversion, no gamma of
	// any kind) plus a manual byte/255 divide is the exact linear
	// reconstruction pbrt's own alpha-cutout convention expects; the result
	// is fed into rtw_image's raw-pixel constructor (already used elsewhere
	// for pre-decoded HDR data) rather than rtw_image::load().
	std::map<int, std::shared_ptr<texture>> alphaMaskCache;
	const auto alphaMaskFor = [&](int mi) -> std::shared_ptr<texture> {
		if (mi < 0 || static_cast<std::size_t>(mi) >= scene.materials.size()) return nullptr;
		const auto it = alphaMaskCache.find(mi);
		if (it != alphaMaskCache.end()) return it->second;
		std::shared_ptr<texture> mask;
		const std::string &fn = scene.materials[static_cast<std::size_t>(mi)].alphaTextureFilename;
		if (!fn.empty()) {
			int w = 0, h = 0, channels = 0;
			unsigned char *bdata = stbi_load(fn.c_str(), &w, &h, &channels, 3);
			if (bdata) {
				std::vector<float> pixels(static_cast<std::size_t>(w) * h * 3);
				for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = bdata[i] / 255.0f;
				stbi_image_free(bdata);
				mask = std::make_shared<image_texture>(rtw_image(w, h, pixels.data()));
			}
		}
		alphaMaskCache.emplace(mi, mask);
		return mask;
	};

	// RGB-to-scalar collapse (Rec.709 weights) for a homogeneous medium's
	// sigma_a/sigma_s - used both by emitGeometry()'s own per-shape medium
	// handling below (captured by reference into that lambda) and by the
	// camera-medium block further down (build()'s own top-level scope, well
	// after emitGeometry() has returned) - declared here, before both, so
	// there's exactly one definition rather than two independently-
	// maintained copies of the same formula.
	const auto luminance = [](const double c[3]) {
		return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
	};

	// Emitting geometry is now done more than once - for the scene itself, and
	// again for each instance definition, whose geometry stays in object space
	// and is placed by a transform rather than baked. Everything below is what
	// it always was; only its inputs and outputs became parameters.
	const auto emitGeometry = [&](const std::vector<pbrt_flatten::Triangle> &tris,
								  const std::vector<pbrt_flatten::Sphere> &sphs,
								  const std::vector<pbrt_flatten::Disk> &disks,
								  const std::vector<pbrt_flatten::Cylinder> &cylinders,
								  const std::vector<pbrt_flatten::Cone> &cones,
								  const std::vector<pbrt_flatten::Paraboloid> &paraboloids,
								  const std::vector<pbrt_flatten::BilinearPatch> &patches,
								  const std::vector<pbrt_flatten::Curve> &curveDecls,
								  hittable_list &world, hittable_list &lights) {
	// ---- triangles -------------------------------------------------------
	if (!tris.empty()) {
		auto mesh = std::make_shared<triangle_mesh_data>();
		std::map<VertexKey, int> seen;

		// A mesh either has a normal for every vertex or for none: `triangle`
		// gates interpolation on has_normals(), which is all-or-nothing, so a
		// partially filled list would index past the end. Same story for UV -
		// `has_uvs()` (triangle.h) is the same all-or-nothing gate.
		bool anyNormals = false;
		bool anyUVs = false;
		for (const pbrt_flatten::Triangle &t : tris) {
			if (t.hasNormals) anyNormals = true;
			if (t.hasUVs) anyUVs = true;
		}

		const auto vertexIndex = [&](const double *p, const double *n, const double *uv) {
			const VertexKey k{p[0], p[1], p[2],
							  n ? n[0] : 0.0, n ? n[1] : 0.0, n ? n[2] : 0.0,
							  uv ? uv[0] : 0.0, uv ? uv[1] : 0.0};
			auto it = seen.find(k);
			if (it != seen.end()) return it->second;
			const int idx = static_cast<int>(mesh->positions.size());
			mesh->positions.push_back(point3(p[0], p[1], p[2]));
			if (anyNormals) mesh->normals.push_back(vec3(n[0], n[1], n[2]));
			// A triangle from a source that never threads UV (loopsubdiv/
			// plymesh - see pbrt_flatten::Triangle::hasUVs's own comment) has
			// no meaningful "geometric" UV to fall back to the way a face
			// normal does - (0,0) is an arbitrary but harmless filler, same
			// as GPU's own pre-this-fix "no data" default.
			if (anyUVs) { mesh->uvs.push_back(uv ? uv[0] : 0.0); mesh->uvs.push_back(uv ? uv[1] : 0.0); }
			seen.emplace(k, idx);
			return idx;
		};

		// Indices first, so the mesh is complete before any triangle refers to
		// it - triangle's constructor reads the positions immediately to
		// precompute its normal and area.
		std::vector<std::pair<int, int>> perTriangleMaterial;
		perTriangleMaterial.reserve(tris.size());
		for (const pbrt_flatten::Triangle &t : tris) {
			// When any mesh in the scene has shading normals, a face without
			// its own still needs one per vertex or the two arrays fall out of
			// step. Its geometric normal is the honest answer - it renders
			// exactly as it would have with no normals at all.
			double gn[3] = {0, 0, 1};
			if (anyNormals && !t.hasNormals) {
				const double e1[3] = {t.v[3] - t.v[0], t.v[4] - t.v[1], t.v[5] - t.v[2]};
				const double e2[3] = {t.v[6] - t.v[0], t.v[7] - t.v[1], t.v[8] - t.v[2]};
				gn[0] = e1[1] * e2[2] - e1[2] * e2[1];
				gn[1] = e1[2] * e2[0] - e1[0] * e2[2];
				gn[2] = e1[0] * e2[1] - e1[1] * e2[0];
				const double len = std::sqrt(gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2]);
				if (len > 0) { gn[0] /= len; gn[1] /= len; gn[2] /= len; }
			}
			const double *n0 = t.hasNormals ? &t.n[0] : gn;
			const double *n1 = t.hasNormals ? &t.n[3] : gn;
			const double *n2 = t.hasNormals ? &t.n[6] : gn;
			const double *uv0 = t.hasUVs ? &t.uv[0] : nullptr;
			const double *uv1 = t.hasUVs ? &t.uv[2] : nullptr;
			const double *uv2 = t.hasUVs ? &t.uv[4] : nullptr;

			mesh->indices.push_back(vertexIndex(&t.v[0], n0, uv0));
			mesh->indices.push_back(vertexIndex(&t.v[3], n1, uv1));
			mesh->indices.push_back(vertexIndex(&t.v[6], n2, uv2));
			perTriangleMaterial.emplace_back(t.material, t.areaLight);
		}
		out.uniqueVertexCount += mesh->positions.size();

		for (std::size_t i = 0; i < perTriangleMaterial.size(); ++i) {
			auto mat = cachedMaterial(perTriangleMaterial[i].first,
									  perTriangleMaterial[i].second);
			auto tri = std::make_shared<triangle>(mesh, static_cast<int>(i), mat,
												   alphaMaskFor(perTriangleMaterial[i].first));
			world.add(tri);
			if (perTriangleMaterial[i].second >= 0) lights.add(tri);
		}
		out.triangleCount += perTriangleMaterial.size();
	}

	// MediumInterface "insideMedium" "" - layer a participating medium INSIDE
	// a shape already added to world above, exactly the pattern this
	// codebase's own hand-built scenes use (e.g. scenes_advanced.h's
	// build_dielectric_medium_scene: a real surface material - glass, or
	// here whatever the shape's own Material directive resolved to - with
	// fog/smoke boxed inside it). constant_medium's constructor wants a
	// scalar sigma_a/sigma_s plus a chromatic albedo tint, not pbrt's own
	// per-channel RGB pair (see pbrt_flatten::Medium's own comment) -
	// `luminance` (captured from build()'s own top-level scope - see its
	// declaration above emitGeometry() for why) collapses each to a scalar,
	// and the scattering channel ratio survives as the albedo tint. Shared
	// across every shape kind below (sphere, disk, cylinder) that carries a
	// `medium` field.
	const auto addMediumIfPresent = [&](const std::shared_ptr<hittable> &shape, int mediumIndex) {
		if (mediumIndex < 0 || static_cast<std::size_t>(mediumIndex) >= scene.media.size()) return;
		const pbrt_flatten::Medium &md = scene.media[static_cast<std::size_t>(mediumIndex)];

		// cloud/rgbgrid: real heterogeneous media (src/shared/cloud_medium.h,
		// src/shared/rgb_grid_medium.h), wrapped in the SAME CPU hittables
		// this codebase's own E2/E4 showcase scenes use - see
		// pbrt_flatten::Medium's own comment for why only these two of
		// pbrt-v4's several non-homogeneous types are wired here. Both
		// wrap `shape` the same way constant_medium does below (visible
		// bounds come from the shape's own bounding_box(); world-space
		// AABB / world<->medium transform were already resolved at
		// flatten() time).
		if (md.type == "cloud") {
			// sigma_a is always forced to 0 below (pure scattering) even if
			// the scene gave a nonzero one - see cloud_medium_hittable.h's
			// own comment for why (matches constant_medium's identical
			// convention); flatten() already warned about this - see
			// pbrt_flatten.h's own cloud-parsing block.
			const auto cloud = CloudMedium<double>::make(
				md.p0[0], md.p0[1], md.p0[2], md.p1[0], md.p1[1], md.p1[2],
				md.toMediumMat, md.toMediumTranslate,
				/*sigma_a=*/0.0, luminance(md.sigma_s), md.g,
				md.density, md.wispiness, md.frequency);
			const point3 world_min(md.worldMin[0], md.worldMin[1], md.worldMin[2]);
			const point3 world_max(md.worldMax[0], md.worldMax[1], md.worldMax[2]);
			world.add(std::make_shared<cloud_medium_hittable>(cloud, color(1,1,1), world_min, world_max));
			return;
		}
		if (md.type == "rgbgrid") {
			const Bounds3<double> bounds(md.p0[0], md.p0[1], md.p0[2], md.p1[0], md.p1[1], md.p1[2]);
			// Real per-voxel "rgb Le"/"float Lescale" (pbrt-v4's own
			// RGBGridMedium::LeGrid/LeScale) - see pbrt_flatten::Medium::
			// Le_r's own comment for the full derivation. md.Le_scale is
			// passed through UNBAKED (matches md.Le_r/g/b staying raw too) -
			// RGBGridMediumData::build()/sample_point() are already the
			// consumers that apply it at sample time.
			const auto grid = RGBGridMediumData<double>::build(
				md.sigma_a_r, md.sigma_a_g, md.sigma_a_b,
				md.sigma_s_r, md.sigma_s_g, md.sigma_s_b,
				md.Le_r, md.Le_g, md.Le_b,
				md.nx, md.ny, md.nz, bounds, /*sigma_scale=*/1.0, md.Le_scale, md.g);
			const point3 world_min(md.worldMin[0], md.worldMin[1], md.worldMin[2]);
			const point3 world_max(md.worldMax[0], md.worldMax[1], md.worldMax[2]);
			world.add(std::make_shared<rgb_grid_medium_hittable>(
				grid, md.g, world_min, world_max, md.toMediumMat, md.toMediumTranslate));
			return;
		}
		if (md.type == "uniformgrid") {
			// gridDensity is empty when flatten() couldn't find a valid "float
			// density" array (missing, or wrong length - see pbrt_flatten.h's
			// own uniformgrid-parsing block, which already warned) -
			// GridMediumData<T>'s constructor has no empty-vector safety net
			// the way RGBGridMediumData::build()'s make_grid() sentinel does,
			// so skip adding a hittable entirely rather than risk constructing
			// a mismatched-size SampledGrid; an invisible medium is the
			// correct, safe degradation for "no density data given" anyway.
			if (md.gridDensity.empty()) return;
			// sigma_a is always forced to 0 below (pure scattering), same
			// convention/reason as cloud and rgbgrid above.
			const Bounds3<double> bounds(md.p0[0], md.p0[1], md.p0[2], md.p1[0], md.p1[1], md.p1[2]);
			const GridMediumData<double> grid(
				md.gridDensity, md.nx, md.ny, md.nz, bounds,
				/*sa=*/0.0, luminance(md.sigma_s), md.g);
			const point3 world_min(md.worldMin[0], md.worldMin[1], md.worldMin[2]);
			const point3 world_max(md.worldMax[0], md.worldMax[1], md.worldMax[2]);
			world.add(std::make_shared<grid_medium_hittable>(
				grid, color(1,1,1), md.g, world_min, world_max, md.toMediumMat, md.toMediumTranslate));
			return;
		}

		const double sig_a = luminance(md.sigma_a);
		const double sig_s = luminance(md.sigma_s);
		const color albedo = (sig_s > 1e-9)
			? color(md.sigma_s[0] / sig_s, md.sigma_s[1] / sig_s, md.sigma_s[2] / sig_s)
			: color(1, 1, 1);
		// "rgb Le"/"float Lescale" (pbrt_flatten::Medium::Le's own comment) -
		// passed RAW (not pre-weighted by sigma_a/sigma_t) - constant_medium's
		// own constructor already computes sigma_t for ss_albedo just above
		// this same value, so it also weights Le by sigma_a/sigma_t there
		// (the fraction of collisions that would have been absorption rather
		// than scattering, pbrt-v4's own VolPathIntegrator collision-
		// estimator - see hg_phase_material::emitted()'s own comment) rather
		// than this call site deriving an independent, redundant copy of
		// sigma_t just to pre-weight it here.
		const color Le(md.Le[0], md.Le[1], md.Le[2]);
		world.add(std::make_shared<constant_medium>(shape, sig_a, sig_s, albedo, md.g, Le));
	};

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : sphs) {
		auto mat = cachedMaterial(s.material, s.areaLight);
		std::shared_ptr<hittable> sp;
		if (s.clipped) {
			// Real zmin/zmax/phimax clipping - see pbrt_flatten::Sphere's
			// own comment for why this needs the real object-to-world
			// transform (sphere_clipped_hittable.h), unlike the plain
			// baked center/radius path below.
			sp = std::make_shared<sphere_clipped_hittable>(
				s.radiusLocal, s.zMin, s.zMax, degrees_to_radians(s.phiMaxDeg), toMatrix4(s.xform), mat);
		} else if (s.center1[0] != s.center[0] || s.center1[1] != s.center[1] ||
				   s.center1[2] != s.center[2]) {
			// Object motion blur (pbrt_flatten::Sphere::center1's own
			// comment) - the existing two-centre moving constructor
			// (sphere.h) already stores center as a ray and interpolates
			// via center.at(r.time()) in hit(); ray.time() is sampled once
			// per camera ray in camera.h and threaded through every bounce
			// already, so no other CPU change is needed for this to work.
			sp = std::make_shared<sphere>(point3(s.center[0], s.center[1], s.center[2]),
										   point3(s.center1[0], s.center1[1], s.center1[2]),
										   s.radius, mat);
		} else {
			sp = std::make_shared<sphere>(point3(s.center[0], s.center[1], s.center[2]),
										   s.radius, mat);
		}
		world.add(sp);
		if (s.areaLight >= 0) lights.add(sp);
		// cpuMediumUnsupported (pbrt_flatten::Sphere's own comment): an open
		// shell can't bound a participating medium correctly, on GPU now
		// either (its own ClippedSphere branch drops the medium too) - not a
		// CPU-only limitation anymore. flatten() already warned; here we
		// just honor it by not wrapping this specific hittable in
		// constant_medium.
		if (!s.cpuMediumUnsupported) addMediumIfPresent(sp, s.medium);
	}
	out.sphereCount += sphs.size();

	// ---- disks / cylinders -------------------------------------------------
	// Shape "disk"/"cylinder" - unlike Sphere, these keep their CTM unbaked
	// (see pbrt_flatten::Disk/Cylinder's own comment for why) and apply it at
	// intersection time via disk_hittable/cylinder_hittable, the same
	// ray-into-object-space technique transform_instance.h already uses for
	// object instancing.
	for (const pbrt_flatten::Disk &d : disks) {
		auto mat = cachedMaterial(d.material, d.areaLight);
		auto disk = std::make_shared<disk_hittable>(
			d.radius, d.innerRadius, d.height, degrees_to_radians(d.phiMaxDeg), toMatrix4(d.xform), mat);
		world.add(disk);
		if (d.areaLight >= 0) lights.add(disk);
		addMediumIfPresent(disk, d.medium);
	}
	out.diskCount += disks.size();

	for (const pbrt_flatten::Cylinder &c : cylinders) {
		auto mat = cachedMaterial(c.material, c.areaLight);
		auto cyl = std::make_shared<cylinder_hittable>(
			c.radius, c.zMin, c.zMax, degrees_to_radians(c.phiMaxDeg), toMatrix4(c.xform), mat);
		world.add(cyl);
		if (c.areaLight >= 0) lights.add(cyl);
		addMediumIfPresent(cyl, c.medium);
	}
	out.cylinderCount += cylinders.size();

	// ---- cones / paraboloids -----------------------------------------------
	// Shape "cone"/"paraboloid" - same unbaked-CTM technique as disk/cylinder
	// above. v1 scope (see pbrt_flatten::Cone/Paraboloid's own comment):
	// geometry-only, so no lights.add()/addMediumIfPresent() call - neither
	// struct carries an areaLight or medium field at all (flatten() already
	// warns and drops both at parse time if the scene requested them).
	for (const pbrt_flatten::Cone &cn : cones) {
		auto mat = cachedMaterial(cn.material, -1);
		auto cone = std::make_shared<cone_hittable>(
			cn.radius, cn.height, degrees_to_radians(cn.phiMaxDeg), toMatrix4(cn.xform), mat);
		world.add(cone);
	}
	out.coneCount += cones.size();

	for (const pbrt_flatten::Paraboloid &pb : paraboloids) {
		auto mat = cachedMaterial(pb.material, -1);
		auto para = std::make_shared<paraboloid_hittable>(
			pb.radius, pb.zMin, pb.zMax, degrees_to_radians(pb.phiMaxDeg), toMatrix4(pb.xform), mat);
		world.add(para);
	}
	out.paraboloidCount += paraboloids.size();

	// ---- bilinear patches -------------------------------------------------
	// Shape "bilinearmesh" - see pbrt_flatten.h's BilinearPatch comment for
	// why only the single-patch form reaches here. bilinear_patch_hittable
	// (scenes_advanced.h) now overrides pdf_value()/random() the same way
	// quad does, so an emissive one is NEE-samplable, not just hittable.
	for (const pbrt_flatten::BilinearPatch &bp : patches) {
		auto mat = cachedMaterial(bp.material, bp.areaLight);
		auto patch = std::make_shared<bilinear_patch_hittable>(
			point3(bp.p[0][0], bp.p[0][1], bp.p[0][2]),
			point3(bp.p[1][0], bp.p[1][1], bp.p[1][2]),
			point3(bp.p[2][0], bp.p[2][1], bp.p[2][2]),
			point3(bp.p[3][0], bp.p[3][1], bp.p[3][2]),
			mat);
		world.add(patch);
		if (bp.areaLight >= 0) lights.add(patch);
	}
	out.bilinearPatchCount += patches.size();

	// ---- curves ------------------------------------------------------------
	// Shape "curve" - see pbrt_flatten::Curve's own comment for scope (cubic
	// Bezier, "bezier" basis only, already split into independent per-segment
	// 4-control-point Bezier curves by flatten()). One CurveShape<double> per
	// segment, wrapped in the existing curve_shape_hittable. width0/width1 are
	// re-lerped per segment (matching pbrt-v4's own Curve::Create,
	// shapes.cpp:894-895 exactly: Lerp(seg/nSegments, width0,width1)) so a
	// multi-segment strand tapers smoothly across its whole length rather than
	// each segment re-tapering its own full width0->width1 range.
	for (const pbrt_flatten::Curve &cd : curveDecls) {
		// forCurve=true: see hair_material's own tangentIsDpdu comment - real
		// curve geometry has a genuine fiber tangent (dpdu) available, unlike
		// every other shape here, which only offers the shading normal as a
		// proxy.
		auto mat = cachedMaterial(cd.material, cd.areaLight, /*forCurve=*/true);
		const CurveType type = (cd.curveType == "cylinder") ? CurveType::Cylinder
			: (cd.curveType == "ribbon") ? CurveType::Ribbon : CurveType::Flat;
		for (int seg = 0; seg < cd.nSegments; ++seg) {
			double cpx[4], cpy[4], cpz[4];
			for (int i = 0; i < 4; ++i) {
				const std::size_t idx = (static_cast<std::size_t>(seg) * 4 + i) * 3;
				cpx[i] = cd.cp[idx]; cpy[i] = cd.cp[idx + 1]; cpz[i] = cd.cp[idx + 2];
			}
			const double t0 = static_cast<double>(seg) / cd.nSegments;
			const double t1 = static_cast<double>(seg + 1) / cd.nSegments;
			const double segW0 = cd.width0 + (cd.width1 - cd.width0) * t0;
			const double segW1 = cd.width0 + (cd.width1 - cd.width0) * t1;
			CurveShape<double> curve = (type == CurveType::Ribbon)
				? CurveShape<double>::make_ribbon(cpx, cpy, cpz, 0.0, 1.0, segW0, segW1,
					cd.n[static_cast<std::size_t>(seg) * 3], cd.n[static_cast<std::size_t>(seg) * 3 + 1],
					cd.n[static_cast<std::size_t>(seg) * 3 + 2],
					cd.n[static_cast<std::size_t>(seg + 1) * 3], cd.n[static_cast<std::size_t>(seg + 1) * 3 + 1],
					cd.n[static_cast<std::size_t>(seg + 1) * 3 + 2])
				: CurveShape<double>::make(cpx, cpy, cpz, 0.0, 1.0, segW0, segW1, type);
			auto ch = std::make_shared<curve_shape_hittable>(curve, mat);
			world.add(ch);
			if (cd.areaLight >= 0) lights.add(ch);
		}
	}
	out.curveCount += curveDecls.size();
	};


	emitGeometry(scene.triangles, scene.spheres, scene.disks, scene.cylinders,
				 scene.cones, scene.paraboloids,
				 scene.bilinearPatches, scene.curves, *out.world, *out.lights);

	// ---- instances -------------------------------------------------------
	// Each definition is built once, into its own BVH, and then placed by a
	// transform per instance. That BVH is shared by every placement - which is
	// the entire point, and the reason this cannot simply bake vertices.
	//
	// No light list is passed: flatten() has already moved any emissive shapes
	// out of the group and baked them per placement into world space, because
	// a light has to be enumerable to be sampled. Passing one here would be
	// harmless but misleading, so it gets a scratch list that stays empty.
	// Built once per DEFINITION, before any placement looks at them. Building
	// inside the instance loop instead would produce one BVH per placement,
	// which is baking with extra steps.
	std::vector<std::shared_ptr<hittable>> groupBVHs(scene.groups.size());
	for (std::size_t g = 0; g < scene.groups.size(); ++g) {
		const pbrt_flatten::InstanceGroup &grp = scene.groups[g];
		if (grp.triangles.empty() && grp.spheres.empty()) continue;

		auto geometry = std::make_shared<hittable_list>();
		hittable_list unusedLights;
		// No InstanceGroup::bilinearPatches/disks/cylinders/cones/paraboloids/
		// curves - object-space bilinear patches, disks, cylinders, cones,
		// paraboloids and curves inside an instance definition are all out of
		// scope (see flatten()'s null-bilinearPatches/disks/cylinders/cones/
		// paraboloids/curves comments on why), so these are always empty.
		static const std::vector<pbrt_flatten::BilinearPatch> kNoBilinearPatches;
		static const std::vector<pbrt_flatten::Disk> kNoDisks;
		static const std::vector<pbrt_flatten::Cylinder> kNoCylinders;
		static const std::vector<pbrt_flatten::Cone> kNoCones;
		static const std::vector<pbrt_flatten::Paraboloid> kNoParaboloids;
		static const std::vector<pbrt_flatten::Curve> kNoCurves;
		emitGeometry(grp.triangles, grp.spheres, kNoDisks, kNoCylinders,
					 kNoCones, kNoParaboloids,
					 kNoBilinearPatches, kNoCurves, *geometry, unusedLights);
		if (!geometry->objects.empty())
			groupBVHs[g] = std::make_shared<bvh_node>(*geometry);
	}

	for (const pbrt_flatten::Instance &inst : scene.instances) {
		if (inst.group < 0 ||
			static_cast<std::size_t>(inst.group) >= groupBVHs.size()) continue;
		const std::shared_ptr<hittable> &shared =
			groupBVHs[static_cast<std::size_t>(inst.group)];
		if (!shared) continue;

		pbrt_scene::Matrix4 m;
		for (int k = 0; k < 16; ++k) m.m[k] = inst.xform[k];
		out.world->add(std::make_shared<transform_instance>(shared, m));
		++out.instanceCount;
	}

	// A flat list would make every ray test every primitive; these scenes are
	// the reason the BVH exists.
	if (!out.world->objects.empty()) {
		auto accelerated = std::make_shared<hittable_list>();
		accelerated->add(std::make_shared<bvh_node>(*out.world));
		out.world = accelerated;
	}

	// ---- infinite/sky light ------------------------------------------------
	// Image-based when pbrt_load::loadFile() successfully decoded one
	// (imageWidth/imageHeight > 0 - see FlatScene::InfiniteLight's comment on
	// why the decode happens there and not here or in flatten()). Falls back
	// to the scene's constant L otherwise - either it never named an image,
	// or naming one failed to resolve/decode (a warning was already recorded
	// for that case).
	if (scene.infiniteLight.present && scene.infiniteLight.hasPortal) {
		if (scene.infiniteLight.imageWidth > 0 && scene.infiniteLight.imageHeight > 0) {
			// Windowed/portal infinite light (pbrt-v4 "point3 portal[4]") - a
			// DIFFERENT image format (equal-area octahedral, not the
			// equirectangular map plain sky_light/InfiniteLight expects - see
			// PortalImageInfiniteLightData's own comment), so this does NOT
			// also build a `sky` from the same pixels the way the plain
			// image-based branch below does - see BuildResult::portal's own
			// comment for why no separate `sky` fallback is needed either.
			// PortalImageInfiniteLightData's own Vec3T (src/shared/vec3_frame.h's
			// global Vec3<T> template) - NOT this codebase's own `vec3` class
			// (a different, unrelated type; see that header's own comment on
			// why it's kept separate from this project's vec3/point3/color).
			std::array<Vec3<double>, 4> corners;
			for (int i = 0; i < 4; ++i)
				corners[i] = Vec3<double>(scene.infiniteLight.portal[i*3+0],
				                          scene.infiniteLight.portal[i*3+1],
				                          scene.infiniteLight.portal[i*3+2]);
			out.portal = std::make_shared<PortalImageInfiniteLightData<double>>(
				scene.infiniteLight.imagePixels.data(),
				scene.infiniteLight.imageWidth, scene.infiniteLight.imageHeight,
				scene.infiniteLight.scale, corners);
		}
		// else: a portal window was declared but its image never named, or
		// named-and-failed to decode (pbrt_load.h already recorded a
		// warning for that). Deliberately builds NEITHER out.portal NOR
		// out.sky here - falling back to the plain-sky branch below would
		// silently turn a windowed light into an unwindowed, full-strength
		// sky flooding the whole scene with light from every direction
		// (InfiniteLight defaults L to white, scale to 1.0), the opposite
		// of what a portal author asked for. Failing closed (no sky light
		// at all) matches this codebase's established "fail black rather
		// than fail wrong" convention for unsupported portal combinations
		// (see the SPPM/BDPT portal-skip comments in cpu_interface.cpp and
		// cpu_interface_bdpt.cpp).
	} else if (scene.infiniteLight.present) {
		if (scene.infiniteLight.imageWidth > 0 && scene.infiniteLight.imageHeight > 0) {
			out.sky = std::make_shared<sky_light>(
				scene.infiniteLight.imageWidth, scene.infiniteLight.imageHeight,
				scene.infiniteLight.imagePixels.data(), scene.infiniteLight.scale);
		} else {
			out.sky = std::make_shared<sky_light>(color(
				scene.infiniteLight.L[0] * scene.infiniteLight.scale,
				scene.infiniteLight.L[1] * scene.infiniteLight.scale,
				scene.infiniteLight.L[2] * scene.infiniteLight.scale));
		}
	}

	// ---- camera medium ------------------------------------------------------
	// pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own
	// comment) - already resolved by flatten() to a valid homogeneous-only,
	// no-per-shape-medium-conflict index, or -1 if none/unsupported (both
	// scope cuts already warned about there) - this is just the same
	// Medium-struct-to-runtime-object construction addMediumIfPresent()
	// below does for a per-shape medium, minus the boundary shape.
	if (scene.cameraMediumIndex >= 0 &&
		static_cast<std::size_t>(scene.cameraMediumIndex) < scene.media.size()) {
		const pbrt_flatten::Medium &m = scene.media[static_cast<std::size_t>(scene.cameraMediumIndex)];
		// Collapsed to a scalar extinction + chromatic albedo tint via the
		// SAME `luminance` addMediumIfPresent() (above) uses for a per-shape
		// medium's identical homogeneous branch - luminance-weighted (not a
		// flat per-channel average), tint derived from sigma_s alone (the
		// scattering-only single-scattering-albedo direction), Le passed
		// RAW since ambient_medium's own constructor (mirroring
		// constant_medium's) already does the sigma_a/sigma_t weighting.
		const double sig_a = luminance(m.sigma_a);
		const double sig_s = luminance(m.sigma_s);
		const color tint = (sig_s > 1e-9)
			? color(m.sigma_s[0] / sig_s, m.sigma_s[1] / sig_s, m.sigma_s[2] / sig_s)
			: color(1, 1, 1);
		out.cameraMedium = std::make_shared<ambient_medium>(
			sig_a, sig_s, tint, m.g, color(m.Le[0], m.Le[1], m.Le[2]));
	}

	// ---- punctual (delta) lights -------------------------------------------
	// LightSource point/spot/distant/goniometric/projection - see
	// pbrt_flatten::PunctualLight's own comment for why this is a bridging
	// job onto punctual_light_objects.h's existing constructors, already
	// proven by this codebase's own C2-C6 showcase scenes, rather than new
	// rendering math.
	if (!scene.punctualLights.empty()) {
		out.punctLights = std::make_shared<punctual_light_list>();
		for (const pbrt_flatten::PunctualLight &pl : scene.punctualLights) {
			switch (pl.kind) {
			case pbrt_flatten::PunctualLightKind::Point:
				out.punctLights->add_point(
					point3(pl.pos[0], pl.pos[1], pl.pos[2]),
					color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
					pl.scale);
				break;
			case pbrt_flatten::PunctualLightKind::Spot:
				out.punctLights->add_spot(
					point3(pl.pos[0], pl.pos[1], pl.pos[2]),
					vec3(pl.dir[0], pl.dir[1], pl.dir[2]),
					color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
					pl.coneAngleDeg, pl.falloffStartAngleDeg, pl.scale);
				break;
			case pbrt_flatten::PunctualLightKind::Distant:
				out.punctLights->add_distant(
					vec3(pl.dir[0], pl.dir[1], pl.dir[2]),
					color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
					pl.sceneRadius, pl.scale);
				break;
			case pbrt_flatten::PunctualLightKind::Goniometric: {
				double id[9];
				for (int c = 0; c < 9; ++c) id[c] = pl.worldToLight[c];
				// pl.filename is only ever non-empty after pbrt_load.h's
				// post-flatten pass confirmed the file exists (PunctualLight::
				// filename's own comment). GoniometricLight<T>'s own equal-
				// area mapping requires a SQUARE image (see its make()'s own
				// comment) - matches pbrt-v4's own GoniometricLight::Create,
				// which ErrorExits on a non-square image; softened here to a
				// silent fallback (same "rare enough, not worth a second
				// probe-and-fallback dance" precedent as the Diffuse
				// imagemap-texture case above) rather than aborting the load.
				// pbrt-v4 collapses a multi-channel image down to one
				// greyscale channel before use (lights.cpp's own
				// GoniometricLight::Create) - a plain per-pixel RGB average
				// approximates that collapse without needing this loader's
				// own luminance-weight table.
				bool usedRealProfile = false;
				if (!pl.filename.empty()) {
					rtw_image img = decodePunctualLightImageFile(pl.filename);
					if (img.width() > 0 && img.width() == img.height()) {
						const int n = img.width();
						std::vector<double> image(static_cast<std::size_t>(n) * n);
						for (int v = 0; v < n; ++v) {
							for (int u = 0; u < n; ++u) {
								const float *px = img.float_pixel_data(u, v);
								image[static_cast<std::size_t>(v) * n + u] =
									(px[0] + px[1] + px[2]) / 3.0;
							}
						}
						out.punctLights->add_gonio(
							point3(pl.pos[0], pl.pos[1], pl.pos[2]),
							color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
							pl.scale, id, image, n, n);
						usedRealProfile = true;
					}
				}
				if (!usedRealProfile) {
					// Uniform (isotropic) fallback - same shape as
					// GoniometricLight<T>::make_isotropic(), just built
					// explicitly here so the real worldToLight rotation
					// (rather than that helper's hardcoded identity) still
					// carries through for a scene that rotated the light.
					static const std::vector<double> kUniformImage(4 * 4, 1.0);
					out.punctLights->add_gonio(
						point3(pl.pos[0], pl.pos[1], pl.pos[2]),
						color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
						pl.scale, id, kUniformImage, 4, 4);
				}
				break;
			}
			case pbrt_flatten::PunctualLightKind::Projection: {
				double wtl[9];
				for (int c = 0; c < 9; ++c) wtl[c] = pl.worldToLight[c];
				// pl.filename is only ever non-empty after pbrt_load.h's
				// post-flatten pass confirmed the file exists.
				bool usedRealSlide = false;
				if (!pl.filename.empty()) {
					rtw_image img = decodePunctualLightImageFile(pl.filename);
					if (img.width() > 0 && img.height() > 0) {
						const int nx = img.width(), ny = img.height();
						std::vector<double> image(static_cast<std::size_t>(nx) * ny * 3);
						for (int v = 0; v < ny; ++v) {
							for (int u = 0; u < nx; ++u) {
								const float *px = img.float_pixel_data(u, v);
								const std::size_t i = (static_cast<std::size_t>(v) * nx + u) * 3;
								image[i + 0] = px[0];
								image[i + 1] = px[1];
								image[i + 2] = px[2];
							}
						}
						out.punctLights->add_projection(
							point3(pl.pos[0], pl.pos[1], pl.pos[2]),
							pl.scale, wtl, pl.fovDeg, image, nx, ny);
						usedRealSlide = true;
					}
				}
				if (!usedRealSlide) {
					// Uniform white 2x2 slide - reproduces a plain cone-
					// shaped beam of the requested fov/scale/aim, matching
					// ProjectionLight<T>::make_uniform()'s own fallback.
					static const std::vector<double> kUniformSlide(2 * 2 * 3, 1.0);
					out.punctLights->add_projection(
						point3(pl.pos[0], pl.pos[1], pl.pos[2]),
						pl.scale, wtl, pl.fovDeg, kUniformSlide, 2, 2);
				}
				break;
			}
			}
		}
	}

	return out;
}

} // namespace pbrt_cpu
