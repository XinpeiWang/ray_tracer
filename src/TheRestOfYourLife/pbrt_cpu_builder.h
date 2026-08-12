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
#include "sphere.h"
#include "triangle.h"

namespace pbrt_cpu {

namespace detail {

// pbrt's material names already match ours (see pbrt_flatten.h), so this is
// construction rather than interpretation. An Unsupported material becomes
// diffuse - flatten() has already warned about it by name, so failing here
// would only turn a documented approximation into a refusal to open the file.
inline std::shared_ptr<material> makeMaterial(const pbrt_flatten::Material &m,
											  const pbrt_flatten::Emission *emission) {
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
		// pbrt describes conductors spectrally; our metal takes an albedo and
		// a fuzz, so roughness maps onto fuzz directly.
		return std::make_shared<metal>(albedo, m.roughness);
	case pbrt_flatten::MaterialKind::Dielectric:
	case pbrt_flatten::MaterialKind::ThinDielectric:
		return std::make_shared<dielectric>(m.ior);
	case pbrt_flatten::MaterialKind::Diffuse:
	case pbrt_flatten::MaterialKind::CoatedDiffuse:
	case pbrt_flatten::MaterialKind::CoatedConductor:
	case pbrt_flatten::MaterialKind::DiffuseTransmission:
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
	std::size_t triangleCount = 0;
	std::size_t sphereCount = 0;
	std::size_t uniqueVertexCount = 0;
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
		return makeMaterial(m, em);
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

	// ---- triangles -------------------------------------------------------
	if (!scene.triangles.empty()) {
		auto mesh = std::make_shared<triangle_mesh_data>();
		std::map<VertexKey, int> seen;

		// A mesh either has a normal for every vertex or for none: `triangle`
		// gates interpolation on has_normals(), which is all-or-nothing, so a
		// partially filled list would index past the end.
		bool anyNormals = false;
		for (const pbrt_flatten::Triangle &t : scene.triangles)
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
		perTriangleMaterial.reserve(scene.triangles.size());
		for (const pbrt_flatten::Triangle &t : scene.triangles) {
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
		out.uniqueVertexCount = mesh->positions.size();

		for (std::size_t i = 0; i < perTriangleMaterial.size(); ++i) {
			auto mat = cachedMaterial(perTriangleMaterial[i].first,
									  perTriangleMaterial[i].second);
			auto tri = std::make_shared<triangle>(mesh, static_cast<int>(i), mat);
			out.world->add(tri);
			if (perTriangleMaterial[i].second >= 0) out.lights->add(tri);
		}
		out.triangleCount = perTriangleMaterial.size();
	}

	// ---- spheres ---------------------------------------------------------
	for (const pbrt_flatten::Sphere &s : scene.spheres) {
		auto mat = cachedMaterial(s.material, s.areaLight);
		auto sp = std::make_shared<sphere>(point3(s.center[0], s.center[1], s.center[2]),
										   s.radius, mat);
		out.world->add(sp);
		if (s.areaLight >= 0) out.lights->add(sp);
	}
	out.sphereCount = scene.spheres.size();

	// A flat list would make every ray test every primitive; these scenes are
	// the reason the BVH exists.
	if (!out.world->objects.empty()) {
		auto accelerated = std::make_shared<hittable_list>();
		accelerated->add(std::make_shared<bvh_node>(*out.world));
		out.world = accelerated;
	}
	return out;
}

} // namespace pbrt_cpu
