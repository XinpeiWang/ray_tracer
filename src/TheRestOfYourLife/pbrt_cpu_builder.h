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
#include <vector>

#include "../shared/pbrt_flatten.h"

#include "bvh.h"
#include "hittable_list.h"
#include "material.h"
#include "scenes_advanced.h"   // bilinear_patch_hittable
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
inline std::shared_ptr<material> makeMaterial(const pbrt_flatten::Material &m,
											  const pbrt_flatten::Emission *emission,
											  const std::vector<pbrt_flatten::Material> &allMaterials = {},
											  int depth = 0) {
	// Emission wins: in pbrt an AreaLightSource attaches to the shape, and its
	// material describes what the surface does with light arriving at it. Our
	// diffuse_light is the emissive case, so an emissive shape becomes one
	// regardless of the material it also declared.
	if (emission) {
		const color L(emission->L[0] * emission->scale,
					  emission->L[1] * emission->scale,
					  emission->L[2] * emission->scale);
		return std::make_shared<diffuse_light>(L);
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
				m.roughness);
		return std::make_shared<metal>(albedo, m.roughness);
	case pbrt_flatten::MaterialKind::Dielectric:
		// A nonzero "roughness"/"uroughness"/"vroughness" (m.roughness - see
		// flatten()'s own fallback-chain comment) means the scene asked for a
		// GGX microfacet dielectric (pbrt-v4 DielectricBxDF's rough path), not
		// a perfect-specular one - this codebase already has a real model for
		// that (rough_dielectric, RoughDielectricBxDF), it just wasn't wired
		// up here, so any scene with a rough glass/window silently rendered
		// as a perfect mirror-and-refract surface instead.
		if (m.roughness > 0.0)
			return std::make_shared<rough_dielectric>(m.ior, m.roughness);
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
		return std::make_shared<coated_diffuse>(albedo, m.ior, m.roughness);
	case pbrt_flatten::MaterialKind::CoatedConductor: {
		const color k = reflectanceToConductorK(albedo);
		return std::make_shared<coated_conductor>(
			1.0, 1.0, 1.0, k.x(), k.y(), k.z(), m.ior, m.roughness);
	}
	case pbrt_flatten::MaterialKind::DiffuseTransmission:
		return std::make_shared<diffuse_transmission>(
			albedo, color(m.transmittance[0], m.transmittance[1], m.transmittance[2]));
	case pbrt_flatten::MaterialKind::Subsurface:
		return std::make_shared<subsurface>(m.ior, m.sigma_a, m.sigma_s, m.g);
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
								 nullptr, allMaterials, depth + 1);
		auto matB = makeMaterial(allMaterials[static_cast<std::size_t>(m.mixMaterialB)],
								 nullptr, allMaterials, depth + 1);
		return std::make_shared<mix_material>(matA, matB, m.mixWeight);
	}
	case pbrt_flatten::MaterialKind::Diffuse:
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
	bool operator<(const VertexKey &o) const {
		if (x != o.x) return x < o.x;
		if (y != o.y) return y < o.y;
		if (z != o.z) return z < o.z;
		if (nx != o.nx) return nx < o.nx;
		if (ny != o.ny) return ny < o.ny;
		return nz < o.nz;
	}
};

} // namespace detail

struct BuildResult {
	std::shared_ptr<hittable_list> world;
	std::shared_ptr<hittable_list> lights;   // emissive shapes, for NEE
	std::shared_ptr<sky_light> sky;          // null if the scene has no infinite light
	// LightSource point/spot/distant/goniometric/projection - null if the
	// scene has none (matches camera_t::punct_lights' own "nullptr = none"
	// convention, which this feeds directly - see scene_registry.h's
	// build_punct wiring). Never empty-but-non-null: left null unless
	// scene.punctualLights actually held something, same as `sky` above.
	std::shared_ptr<punctual_light_list> punctLights;
	std::size_t triangleCount = 0;
	std::size_t sphereCount = 0;
	std::size_t bilinearPatchCount = 0;
	std::size_t uniqueVertexCount = 0;
	// Instance placements actually added to the world. Each shares one
	// BVH with every other placement of the same definition.
	std::size_t instanceCount = 0;
};

// Turns flattened geometry into a BVH-accelerated world plus the light list
// the integrator samples. Materials are created once per (material, emission)
// pair rather than per primitive - a million-triangle mesh with one material
// should hold one material object, not a million.
inline BuildResult build(const pbrt_flatten::FlatScene &scene) {
	using namespace detail;
	BuildResult out;
	out.world = std::make_shared<hittable_list>();
	out.lights = std::make_shared<hittable_list>();


	const auto materialFor = [&scene](int materialIndex, int areaLightIndex)
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
		return makeMaterial(m, em, scene.materials);
	};

	// One material instance per distinct (material, emission) pair.
	std::map<std::pair<int, int>, std::shared_ptr<material>> materialCache;
	const auto cachedMaterial = [&](int mi, int ai) {
		const auto key = std::make_pair(mi, ai);
		auto it = materialCache.find(key);
		if (it != materialCache.end()) return it->second;
		auto made = materialFor(mi, ai);
		materialCache.emplace(key, made);
		return made;
	};

	// Emitting geometry is now done more than once - for the scene itself, and
	// again for each instance definition, whose geometry stays in object space
	// and is placed by a transform rather than baked. Everything below is what
	// it always was; only its inputs and outputs became parameters.
	const auto emitGeometry = [&](const std::vector<pbrt_flatten::Triangle> &tris,
								  const std::vector<pbrt_flatten::Sphere> &sphs,
								  const std::vector<pbrt_flatten::BilinearPatch> &patches,
								  hittable_list &world, hittable_list &lights) {
	// ---- triangles -------------------------------------------------------
	if (!tris.empty()) {
		auto mesh = std::make_shared<triangle_mesh_data>();
		std::map<VertexKey, int> seen;

		// A mesh either has a normal for every vertex or for none: `triangle`
		// gates interpolation on has_normals(), which is all-or-nothing, so a
		// partially filled list would index past the end.
		bool anyNormals = false;
		for (const pbrt_flatten::Triangle &t : tris)
			if (t.hasNormals) { anyNormals = true; break; }

		const auto vertexIndex = [&](const double *p, const double *n) {
			const VertexKey k{p[0], p[1], p[2],
							  n ? n[0] : 0.0, n ? n[1] : 0.0, n ? n[2] : 0.0};
			auto it = seen.find(k);
			if (it != seen.end()) return it->second;
			const int idx = static_cast<int>(mesh->positions.size());
			mesh->positions.push_back(point3(p[0], p[1], p[2]));
			if (anyNormals) mesh->normals.push_back(vec3(n[0], n[1], n[2]));
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

			mesh->indices.push_back(vertexIndex(&t.v[0], n0));
			mesh->indices.push_back(vertexIndex(&t.v[3], n1));
			mesh->indices.push_back(vertexIndex(&t.v[6], n2));
			perTriangleMaterial.emplace_back(t.material, t.areaLight);
		}
		out.uniqueVertexCount += mesh->positions.size();

		for (std::size_t i = 0; i < perTriangleMaterial.size(); ++i) {
			auto mat = cachedMaterial(perTriangleMaterial[i].first,
									  perTriangleMaterial[i].second);
			auto tri = std::make_shared<triangle>(mesh, static_cast<int>(i), mat);
			world.add(tri);
			if (perTriangleMaterial[i].second >= 0) lights.add(tri);
		}
		out.triangleCount += perTriangleMaterial.size();
	}

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : sphs) {
		auto mat = cachedMaterial(s.material, s.areaLight);
		auto sp = std::make_shared<sphere>(point3(s.center[0], s.center[1], s.center[2]),
										   s.radius, mat);
		world.add(sp);
		if (s.areaLight >= 0) lights.add(sp);
	}
	out.sphereCount += sphs.size();

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
	};


	emitGeometry(scene.triangles, scene.spheres, scene.bilinearPatches, *out.world, *out.lights);

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
		// No InstanceGroup::bilinearPatches - object-space bilinear patches
		// inside an instance definition are out of scope (see flatten()'s
		// null-bilinearPatches comment on why), so this is always empty.
		static const std::vector<pbrt_flatten::BilinearPatch> kNoBilinearPatches;
		emitGeometry(grp.triangles, grp.spheres, kNoBilinearPatches, *geometry, unusedLights);
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
	if (scene.infiniteLight.present) {
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
				// No real IES-style profile is ever decoded (see
				// PunctualLight::hadImageFilename's own comment) - a uniform
				// 4x4 all-ones image is the same "isotropic" fallback
				// GoniometricLight<T>::make_isotropic() already uses, just
				// built explicitly here so the real worldToLight rotation
				// (rather than that helper's hardcoded identity) still
				// carries through for a scene that rotated the light even
				// though the image itself is flat.
				static const std::vector<double> kUniformImage(4 * 4, 1.0);
				double id[9];
				for (int c = 0; c < 9; ++c) id[c] = pl.worldToLight[c];
				out.punctLights->add_gonio(
					point3(pl.pos[0], pl.pos[1], pl.pos[2]),
					color(pl.intensity[0], pl.intensity[1], pl.intensity[2]),
					pl.scale, id, kUniformImage, 4, 4);
				break;
			}
			case pbrt_flatten::PunctualLightKind::Projection: {
				// No real slide image is ever decoded (see
				// PunctualLight::hadImageFilename's own comment) - a uniform
				// white 2x2 slide reproduces a plain cone-shaped beam of the
				// requested fov/scale/aim, the same synthetic-pattern tier
				// this loader's own C6 showcase scene already uses.
				static const std::vector<double> kUniformSlide(2 * 2 * 3, 1.0);
				double wtl[9];
				for (int c = 0; c < 9; ++c) wtl[c] = pl.worldToLight[c];
				out.punctLights->add_projection(
					point3(pl.pos[0], pl.pos[1], pl.pos[2]),
					pl.scale, wtl, pl.fovDeg, kUniformSlide, 2, 2);
				break;
			}
			}
		}
	}

	return out;
}

} // namespace pbrt_cpu
