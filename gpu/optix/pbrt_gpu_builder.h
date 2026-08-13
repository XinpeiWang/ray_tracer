#pragma once
// pbrt_gpu_builder.h -- turns a loaded .pbrt scene into GPU SceneData.
//
// The GPU counterpart of src/TheRestOfYourLife/pbrt_cpu_builder.h, consuming
// the same pbrt_flatten::FlatScene so the two backends cannot disagree about
// what the file said - only about how they render it.
//
// AREA LIGHTS ARE THE WHOLE DIFFICULTY
// ------------------------------------
// The GPU samples area lights as spheres or quads and nothing else: the light
// list carries one is-it-a-sphere flag per entry (optix_renderer.cpp). pbrt
// has no quad shape - a light is a `trianglemesh` with an AreaLightSource
// attached - so handed over naively, every pbrt light becomes geometry that
// glows when hit but that next-event estimation cannot aim at. That is not a
// slightly noisier image; it is a black one.
//
// pbrt_quadify.h rejoins the triangle pairs into parallelograms, which is why
// the emissive geometry here goes through it and the rest does not. Emissive
// triangles that will not merge are kept as geometry and counted, so the
// caller can say plainly that some lights will not be sampled rather than
// leaving the user to wonder why a render is dark.

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "scene_builder.h"
#include "optix_math_helpers.h"   // cross(), dot(), length() - used below
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_quadify.h"

namespace pbrt_gpu {

struct BuildStats {
	std::size_t triangles = 0;
	std::size_t spheres = 0;
	std::size_t quadLights = 0;
	std::size_t unsampledEmissiveTriangles = 0;
	// Placements the builder prepared. The renderer turns these into IAS
	// entries; until it does, they are prepared but not drawn.
	std::size_t instancePlacements = 0;
	std::size_t unsupportedInstancedSpheres = 0;
	std::size_t unhandledInstances = 0;
};

namespace detail {

inline float3 f3(const double *v) {
	return make_float3(static_cast<float>(v[0]), static_cast<float>(v[1]),
					   static_cast<float>(v[2]));
}

// Mirrors pbrt_cpu_builder.h's makeMaterial() decision for decision, including
// emission winning over the declared material - in pbrt an AreaLightSource
// attaches to the shape, and the surface is an emitter regardless of what else
// it said it was. The two builders disagreeing here would mean the same file
// renders as two different scenes depending on the backend.
inline MaterialData makeMaterial(const pbrt_flatten::Material &m,
								 const pbrt_flatten::Emission *emission) {
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
		break;
	case pbrt_flatten::MaterialKind::ThinDielectric:
		d.type = MaterialType::ThinDielectric;
		break;
	case pbrt_flatten::MaterialKind::CoatedDiffuse:
		d.type = MaterialType::CoatedDiffuse;
		break;
	case pbrt_flatten::MaterialKind::DiffuseTransmission:
		d.type = MaterialType::DiffuseTransmission;
		d.transmittance = d.albedo;
		break;
	case pbrt_flatten::MaterialKind::CoatedConductor:
	case pbrt_flatten::MaterialKind::Diffuse:
	case pbrt_flatten::MaterialKind::Unsupported:
		d.type = MaterialType::Lambertian;
		break;
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
	out.materials.clear();
	out.lightIndices.clear();
	out.isLightSphere.clear();

	// One MaterialData per distinct (material, emission) pair, exactly as the
	// CPU builder caches them - a mesh with a thousand faces sharing one
	// material must not produce a thousand identical GPU materials.
	std::map<std::pair<int, int>, int> cache;
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
		out.materials.push_back(makeMaterial(m, em));
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
			out.isLightSphere.push_back(true);
		}
		out.spheres.push_back(sd);
	}
	stats.spheres = out.spheres.size();

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
			out.isLightSphere.push_back(false);
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
		out.triangles.push_back(td);
	}
	stats.triangles = out.triangles.size();
	stats.unsampledEmissiveTriangles = pbrt_quadify::unmergedEmissiveCount(merged);

	// ---- instanced geometry ----------------------------------------------
	// Object space, kept apart from the world-space list above. Emissive
	// shapes are not here: flatten() already baked those per placement into
	// scene.triangles, because a light must be enumerable to be sampled.
	out.instanceGroups.clear();
	out.instancePlacements.clear();
	out.instanceTriangles.clear();

	std::vector<int> groupIndexMap(scene.groups.size(), -1);
	for (std::size_t g = 0; g < scene.groups.size(); ++g) {
		const pbrt_flatten::InstanceGroup &grp = scene.groups[g];
		if (grp.triangles.empty()) continue;      // spheres: see the note below

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
		groupIndexMap[g] = static_cast<int>(out.instanceGroups.size());
		out.instanceGroups.push_back(gpuGroup);

		if (!grp.spheres.empty()) {
			// Spheres are custom AABB primitives sharing a GAS with quads, so
			// instancing them needs a second per-group GAS and its own SBT
			// region. Triangles cover the published scenes; this is reported
			// rather than silently dropped.
			stats.unsupportedInstancedSpheres += grp.spheres.size();
		}
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
	// Prepared but not yet drawn: the renderer does not build the per-group
	// GASes or the IAS entries yet, so these placements are data only.
	stats.unhandledInstances = out.instancePlacements.size();

	return stats;
}

} // namespace pbrt_gpu
