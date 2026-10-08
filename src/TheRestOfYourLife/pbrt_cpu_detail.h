#pragma once
// pbrt_cpu_detail.h -- the helpers pbrt_cpu_builder.h builds on (textures, materials, mediums, image decoding). Part of pbrt_cpu_builder.h, which includes it.

#include <cmath>
#include <cstdint>    // std::int64_t - see addMediumIfPresent()'s nanovdb voxel-count overflow guard
#include <iostream>   // std::cerr - see addMediumIfPresent()'s nanovdb read-failure diagnostic, this file's one deliberate exception to its own no-console-output convention
#include <map>
#include <mutex>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../shared/pbrt_flatten.h"
#include "../shared/portal_image_infinite_light.h"   // PortalImageInfiniteLightData<double> - LightSource "infinite" "point3 portal[4]"
#include "../external/tinyexr.h"   // LoadEXR() - see decodePunctualLightImageFile()
#include "../external/nanovdb/NanoVDB.h"   // MakeNamedMedium "nanovdb" - see addMediumIfPresent()'s own nanovdb branch
#include "../external/nanovdb/io/IO.h"     // nanovdb::io::readGrid()

#include "bvh.h"
#include "bvh_aggregate_hittable.h"
#include "kd_tree_hittable.h"
#include "constant_medium.h"
#include "curve_shape_hittable.h"
#include "disk_cylinder_hittable.h"
#include "cone_paraboloid_hittable.h"
#include "grid_medium_hittable.h"
#include "hair_material.h"
#include "hittable_list.h"
#include "material.h"
#include "mesh_mtl.h"        // is_grayscale_image() - see the displacement-map bump-vs-normal dispatch
#include "principled_material.h"
#include "../shared/srgb_decode.h"
#include "rtw_stb_image.h"     // stbi_load() - see alphaMaskFor()'s own comment
#include "scenes_advanced.h"   // bilinear_patch_hittable
#include "sphere_clipped_hittable.h"
#include "sky_light.h"
#include "sphere.h"
#include "triangle.h"
#include "transform_instance.h"
#include "animated_transform_instance.h"


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

// One decoded mipmap per (file, options), shared by every material that names it.
// A scene that gives each material of one .mtl/OBJ its own Material directive (the
// environment-*.pbrt scenes: Lost Empire's 45 materials all read the same three
// 8192x8192 atlases) would otherwise decode and keep a private copy per material -
// tens of gigabytes. Weak references, so the pixels still go away with the last
// material that uses them (a later scene load does not inherit a stale gigabyte).
inline std::shared_ptr<mipmap_texture> sharedMipmapTexture(const char *filename,
														   MipMapOptions opts = MipMapOptions{}) {
	static std::mutex mutex;
	static std::map<std::string, std::weak_ptr<mipmap_texture>> cache;
	std::string key = filename;
	key += '|' + std::to_string(static_cast<int>(opts.filter)) + '|' + std::to_string(opts.max_anisotropy) +
		   '|' + std::to_string(static_cast<int>(opts.wrap)) + '|' + std::to_string(opts.invert ? 1 : 0) +
		   '|' + std::to_string(opts.gamma);
	std::lock_guard<std::mutex> lock(mutex);
	if (auto hit = cache[key].lock()) return hit;
	auto fresh = std::make_shared<mipmap_texture>(filename, opts);
	cache[key] = fresh;
	return fresh;
}

// Forward declaration - checkerOrMixSlot (full comment below, at its actual
// definition) and buildNestedProceduralTexture() are mutually referential (a
// nested checkerboard/mix's own tex1/tex2 go back through checkerOrMixSlot).
inline shared_ptr<texture> checkerOrMixSlot(const std::string &filename, const double color3[3],
											  const pbrt_flatten::NestedProceduralTexture *nested = nullptr);

// Builds the real CPU texture object for a resolved second-level nested
// checkerboard/mix (Material::checkerTex1Nested's own comment,
// pbrt_flatten.h) - this level's own tex1/tex2 are already capped to flat
// literal/bare imagemap only (NestedProceduralTexture's own comment), so
// this is NOT itself recursive beyond the one call each way.
inline shared_ptr<texture> buildNestedProceduralTexture(const pbrt_flatten::NestedProceduralTexture &n) {
	shared_ptr<texture> tex1 = checkerOrMixSlot(n.tex1Filename, n.color1);
	shared_ptr<texture> tex2 = checkerOrMixSlot(n.tex2Filename, n.color2);
	if (n.kind == "checkerboard")
		return std::make_shared<uv_checker_texture>(n.uscale, n.vscale, tex1, tex2);
	// "mix"
	if (!n.amountFilename.empty())
		return std::make_shared<mix_texture>(
			tex1, tex2, std::static_pointer_cast<texture>(sharedMipmapTexture(n.amountFilename.c_str())));
	return std::make_shared<mix_texture>(tex1, tex2, n.amount);
}

// A checkerboard/mix texture's own tex1/tex2 slot: a further nested
// checkerboard/mix (Material::checkerTex1Nested's own comment) when
// `nested` names one, else a bare imagemap filename (one-level-nested
// texture reference already resolved by pbrt_flatten.h - see
// Material::checkerTex1Filename/mixTex1Filename's own comments) when
// present, otherwise the flat literal RGB colour flatten() resolved into the
// paired colour array. Shared by hasCheckerReflectance's and
// hasMixReflectance's own tex1/tex2 slots, on both the Diffuse and (since
// CoatedDiffuse gained the same procedural-texture support) CoatedDiffuse
// cases below - four call sites for what used to be identical inline logic.
inline shared_ptr<texture> checkerOrMixSlot(const std::string &filename, const double color3[3],
											  const pbrt_flatten::NestedProceduralTexture *nested) {
	if (nested && !nested->kind.empty())
		return buildNestedProceduralTexture(*nested);
	return filename.empty()
		? std::static_pointer_cast<texture>(std::make_shared<solid_color>(color(color3[0], color3[1], color3[2])))
		: std::static_pointer_cast<texture>(sharedMipmapTexture(filename.c_str()));
}

// A resolved "checkerboard" Texture's own top-level pattern class -
// Material::checkerIs3D's own comment: pbrt-v4's real "integer dimension"
// [3] variant is this project's OWN original world-space checker_texture
// (texture.h), keyed on a transformed hit point; the default 2D variant
// stays the existing UV-tiled uv_checker_texture. Shared by the Diffuse and
// CoatedDiffuse hasCheckerReflectance cases below so the branch exists once.
inline shared_ptr<texture> checkerPatternTexture(const pbrt_flatten::Material &m,
												  shared_ptr<texture> tex1, shared_ptr<texture> tex2) {
	if (m.checkerIs3D)
		return std::make_shared<checker_texture>(m.checkerWorldToTexture, tex1, tex2);
	return std::make_shared<uv_checker_texture>(m.checkerUScale, m.checkerVScale, tex1, tex2);
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
	// static_cast, matching the identical GPU-side cast already used for
	// this same double(pbrt)->float(MipMapOptions) field 3x in
	// gpu/optix/pbrt_gpu_builder_materials.h.
	opts.gamma = static_cast<float>(m.textureGamma);
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
			shared_ptr<texture> tex = sharedMipmapTexture(emission->filename.c_str());
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
		{
			// "rgb reflectance" (or nothing given): pbrt-v4's own conversion to eta = 1 and k solved from the reflectance, run through
			// the same GGX + complex-Fresnel conductor. The old flat-albedo `metal` (roughness as a mirror fuzz) never had a glossy
			// lobe, so a point/spot light could not make a highlight on it at all (K43 rendered no highlight; Metal and pbrt do).
			const color k = reflectanceToConductorK(albedo);
			return std::make_shared<conductor>(1.0, 1.0, 1.0, k.x(), k.y(), k.z(),
				m.roughness_u, m.roughness_v, m.remapRoughness);
		}
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
				sharedMipmapTexture(m.roughnessTextureFilename.c_str(),
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
		// "abbenumber" (Material::abbeNumber's own comment - not real
		// pbrt-v4 syntax) - only wired for the flat-roughness/smooth paths
		// just below, matching the only two shapes this codebase's own
		// dielectric::make_dispersive()/rough_dielectric::make_dispersive()
		// support (isotropic roughness only, no anisotropic-dispersive or
		// texture-roughness-dispersive combination - no bundled scene needs
		// either, not worth the extra surface for this one param).
		if (m.roughness_u > 0.0 || m.roughness_v > 0.0) {
			if (m.abbeNumber > 0.0)
				return rough_dielectric::make_dispersive(m.ior, m.abbeNumber, m.roughness_u);
			return std::make_shared<rough_dielectric>(m.ior, m.roughness_u, m.roughness_v, m.remapRoughness);
		}
		if (m.abbeNumber > 0.0)
			return dielectric::make_dispersive(m.ior, m.abbeNumber);
		if (m.transmissionFilter[0] != 1.0 || m.transmissionFilter[1] != 1.0 || m.transmissionFilter[2] != 1.0)
			return std::make_shared<dielectric>(m.ior, color(m.transmissionFilter[0], m.transmissionFilter[1], m.transmissionFilter[2]));
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
			shared_ptr<texture> tex = sharedMipmapTexture(m.textureFilename.c_str(), imageMapOptionsFor(m));
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
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.checkerTex1Filename, m.checkerColor1, &m.checkerTex1Nested);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.checkerTex2Filename, m.checkerColor2, &m.checkerTex2Nested);
			return std::make_shared<coated_diffuse>(
				checkerPatternTexture(m, tex1, tex2),
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
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.mixTex1Filename, m.mixColor1, &m.mixTex1Nested);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.mixTex2Filename, m.mixColor2, &m.mixTex2Nested);
			// m.mixAmountTextureFilename/mixAmountNested (Material::
			// mixAmountNested's own comment) - a real per-point spatially-
			// varying blend when "amount" itself nested a bare imagemap or a
			// further checkerboard/mix, via mix_texture's own texture-taking
			// amount constructor (reusing checkerOrMixSlot exactly like
			// tex1/tex2 just above - amount's own "flat" fallback color3
			// argument is never actually read here, since this branch only
			// runs when one of the two texture-valued cases is set); the
			// flat-scalar constructor otherwise.
			const bool amountIsTexture = !m.mixAmountTextureFilename.empty() || !m.mixAmountNested.kind.empty();
			shared_ptr<texture> mix = !amountIsTexture
				? std::make_shared<mix_texture>(tex1, tex2, m.mixAmount)
				: std::make_shared<mix_texture>(tex1, tex2,
					checkerOrMixSlot(m.mixAmountTextureFilename, m.mixColor1, &m.mixAmountNested));
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
		// m.conductorRoughness_u/v < 0: no conductor roughness was given, the coat's applies to both.
		const double cru = m.conductorRoughness_u >= 0.0 ? m.conductorRoughness_u : m.roughness_u;
		const double crv = m.conductorRoughness_v >= 0.0 ? m.conductorRoughness_v : m.roughness_v;
		if (m.hasConductorPreset)
			return std::make_shared<coated_conductor>(
				m.conductorEta[0], m.conductorEta[1], m.conductorEta[2],
				m.conductorK[0], m.conductorK[1], m.conductorK[2],
				m.ior, m.roughness_u, m.roughness_v, cru, crv, m.remapRoughness, m.coatThickness);
		const color k = reflectanceToConductorK(albedo);
		return std::make_shared<coated_conductor>(
			1.0, 1.0, 1.0, k.x(), k.y(), k.z(),
			m.ior, m.roughness_u, m.roughness_v, cru, crv, m.remapRoughness, m.coatThickness);
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
			? nullptr : sharedMipmapTexture(m.textureFilename.c_str(), imageMapOptionsFor(m));
		if (rTex && m.textureScale != 1.0)
			rTex = std::make_shared<scaled_texture>(rTex, m.textureScale);
		shared_ptr<texture> tTex = m.transmittanceTextureFilename.empty()
			? nullptr : sharedMipmapTexture(m.transmittanceTextureFilename.c_str(),
				toMipMapOptions(m.transmittanceTextureOptions));
		if (tTex && m.transmittanceTextureScale != 1.0)
			tTex = std::make_shared<scaled_texture>(tTex, m.transmittanceTextureScale);
		return std::make_shared<diffuse_transmission>(albedo, transmittance, rTex, tTex);
	}
	case pbrt_flatten::MaterialKind::Subsurface:
		return std::make_shared<subsurface>(m.ior, m.sigma_a, m.sigma_s, m.g);
	case pbrt_flatten::MaterialKind::Principled:
		// Not real pbrt-v4 (see MaterialKind::Principled's own comment,
		// pbrt_flatten.h) - m.color/m.roughness/m.ior are already generically
		// parsed above from "reflectance"/"roughness"/"eta" like every other
		// kind; only metallic/clearcoat/clearcoatRoughness are Principled-
		// specific fields.
		return std::make_shared<principled>(
			color(m.color[0], m.color[1], m.color[2]),
			m.metallic, m.roughness, m.ior, m.clearcoat, m.clearcoatRoughness);
	case pbrt_flatten::MaterialKind::NormalizedFresnel:
		// Not real pbrt-v4 either (see MaterialKind::NormalizedFresnel's own
		// comment, pbrt_flatten.h) - single parameter, m.ior, already
		// generically parsed above from "eta" like every other kind.
		return std::make_shared<normalized_fresnel>(m.ior);
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
			shared_ptr<texture> tex = sharedMipmapTexture(m.textureFilename.c_str(), imageMapOptionsFor(m));
			if (m.textureScale != 1.0)
				tex = std::make_shared<scaled_texture>(tex, m.textureScale);
			return std::make_shared<lambertian>(tex);
		}
		// m.hasCheckerReflectance (Material::hasCheckerReflectance's own
		// comment) - a procedural pbrt-v4 checkerboard, not an image file, so
		// checkerPatternTexture() (this file, above) builds either the
		// default 2D uv_checker_texture or - "integer dimension" [3],
		// Material::checkerIs3D - this project's own original 3D world-space
		// checker_texture (texture.h) directly, rather than decoding
		// anything from disk. tex1/tex2 each independently use their own
		// polymorphic slot when checkerTex1Filename/checkerTex2Filename
		// named a one-level-nested bare imagemap instead of a flat literal
		// (see that field's own comment) - a mipmap_texture per nested slot
		// instead of the flat solid_color the plain literal case still uses.
		if (m.hasCheckerReflectance) {
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.checkerTex1Filename, m.checkerColor1, &m.checkerTex1Nested);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.checkerTex2Filename, m.checkerColor2, &m.checkerTex2Nested);
			return std::make_shared<lambertian>(checkerPatternTexture(m, tex1, tex2));
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
			// Same up-to-two-level nesting as checkerboard above, via
			// mix_texture's own polymorphic constructor - see the
			// CoatedDiffuse case's identical block (above in this same
			// switch's other branch) for the full comment.
			shared_ptr<texture> tex1 = checkerOrMixSlot(m.mixTex1Filename, m.mixColor1, &m.mixTex1Nested);
			shared_ptr<texture> tex2 = checkerOrMixSlot(m.mixTex2Filename, m.mixColor2, &m.mixTex2Nested);
			const bool amountIsTexture = !m.mixAmountTextureFilename.empty() || !m.mixAmountNested.kind.empty();
			shared_ptr<texture> mix = !amountIsTexture
				? std::make_shared<mix_texture>(tex1, tex2, m.mixAmount)
				: std::make_shared<mix_texture>(tex1, tex2,
					checkerOrMixSlot(m.mixAmountTextureFilename, m.mixColor1, &m.mixAmountNested));
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
	bool operator==(const VertexKey &o) const {
		return x == o.x && y == o.y && z == o.z && nx == o.nx && ny == o.ny && nz == o.nz && u == o.u && v == o.v;
	}
};

// Hash for the vertex weld: a hash table instead of the std::map the weld used (a tree node per unique
// vertex - ~100 bytes of overhead on a 64-byte key - and O(log n) compares of eight doubles, on meshes of
// ten million vertices). std::hash<double> hashes -0.0 and +0.0 alike, matching operator==.
struct VertexKeyHash {
	std::size_t operator()(const VertexKey &k) const {
		std::size_t h = 1469598103934665603ull;
		for (const double d : {k.x, k.y, k.z, k.nx, k.ny, k.nz, k.u, k.v})
			h = (h ^ std::hash<double>{}(d)) * 1099511628211ull;
		return h;
	}
};

} // namespace detail

} // namespace pbrt_cpu
