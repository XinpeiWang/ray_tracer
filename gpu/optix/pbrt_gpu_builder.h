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
	std::size_t bilinearPatches = 0;
	std::size_t quadLights = 0;
	std::size_t emissiveTrianglesSampledIndividually = 0;
	// Placements the builder prepared; optix_renderer.cpp's buildScene()
	// turns each one into its own IAS entry over a per-definition GAS.
	std::size_t instancePlacements = 0;
	// scene.infiniteLight's flat colour for missed rays, or (0,0,0) if the
	// scene has none - GPU approximation of a real environment light, same
	// shape as the hand-written HDRI scenes' own GPU port (see
	// pbrt_flatten.h's InfiniteLight comment for why full image-based
	// importance sampling stays CPU-only). Currently L*scale only; once the
	// image resolver lands this becomes the decoded image's average colour
	// for the filename case.
	float3 backgroundColor = make_float3(0.0f, 0.0f, 0.0f);
};

namespace detail {

inline float3 f3(const double *v) {
	return make_float3(static_cast<float>(v[0]), static_cast<float>(v[1]),
					   static_cast<float>(v[2]));
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
		// No GPU BSSRDF (out of scope - see src/TheRestOfYourLife/
		// material_pbrt.h's `class subsurface` and camera.h::
		// sample_bssrdf_exit(), the CPU-only implementation). Falls back to
		// flat diffuse, exactly as it did before this MaterialKind existed
		// (when "subsurface" mapped to Unsupported, which also fell back to
		// Lambertian right here) - CPU gaining real support for this kind
		// does not change GPU's rendered output at all, only that the
		// shared "not supported" warning in pbrt_flatten.h's flatten() no
		// longer fires for it, since it is genuinely supported on CPU now.
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
	out.bilinearPatches.clear();
	out.materials.clear();
	out.lightIndices.clear();
	out.lightKinds.clear();

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
		td.hasUVs = false;               // flatten does not carry UVs yet
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
			td.hasUVs = false;
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
		} else {
			stats.backgroundColor = make_float3(
				static_cast<float>(sky.L[0] * sky.scale),
				static_cast<float>(sky.L[1] * sky.scale),
				static_cast<float>(sky.L[2] * sky.scale));
		}
	}

	return stats;
}

} // namespace pbrt_gpu
