#pragma once
// pbrt_cpu_builder.h -- builds CPU hittables from a flattened pbrt scene.
//
// This is the first file in the pbrt chain that knows about renderer types.
// pbrt_scene.h (text -> description), ply_mesh.h (mesh bytes -> arrays) and
// pbrt_flatten.h (description -> world-space geometry) are all deliberately
// free of both Qt and hittable/material, so the MSVC test binary can reach
// them. Everything renderer-specific lives here and in the eventual GPU
// counterpart, which consume the same FlatScene.

#include "pbrt_cpu_stages.h"

namespace pbrt_cpu {

namespace scene_builder_impl {
using namespace detail;

// Turns flattened geometry into a BVH-accelerated world plus the light list the integrator samples. Materials are created once per (material,
// emission) pair rather than per primitive - a million-triangle mesh with one material should hold one material object, not a million.
// run() calls the stages below in order.
class CpuSceneBuilder {
public:
	explicit CpuSceneBuilder(const pbrt_flatten::FlatScene &s) : scene(s) {}

	BuildResult run() {
		out.world = std::make_shared<hittable_list>();
		out.lights = std::make_shared<hittable_list>();

		emitGeometry(scene.triangles, scene.spheres, scene.disks, scene.cylinders,
					 scene.cones, scene.paraboloids,
					 scene.bilinearPatches, scene.curves, *out.world, *out.lights,
					 !scene.animatedTriangleMeshes.empty());
		emitInstances();
		emitAnimatedMeshes();
		emitAnimatedBilinearPatches();
		emitAnimatedCurves();
		accelerate();

		// ---- infinite/sky light ------------------------------------------------
		// Image-based when pbrt_load::loadFile() successfully decoded one
		// (imageWidth/imageHeight > 0 - see FlatScene::InfiniteLight's comment on
		// why the decode happens there and not here or in flatten()). Falls back
		// to the scene's constant L otherwise - either it never named an image,
		// or naming one failed to resolve/decode (a warning was already recorded
		// for that case). Extracted to buildSkyOrPortal() above.
		buildSkyOrPortal(scene, out);

		// ---- camera medium ------------------------------------------------------
		// pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own
		// comment) - already resolved by flatten() to a valid homogeneous-only,
		// no-per-shape-medium-conflict index, or -1 if none/unsupported (both
		// scope cuts already warned about there) - this is just the same
		// Medium-struct-to-runtime-object construction addMediumIfPresent()
		// above does for a per-shape medium, minus the boundary shape. Extracted
		// to buildCameraMedium() above.
		buildCameraMedium(scene, out, luminanceOf);

		// ---- punctual (delta) lights -------------------------------------------
		// LightSource point/spot/distant/goniometric/projection - see
		// pbrt_flatten::PunctualLight's own comment for why this is a bridging
		// job onto punctual_light_objects.h's existing constructors, already
		// proven by this codebase's own C2-C6 showcase scenes, rather than new
		// rendering math. Extracted to buildPunctualLights() above.
		buildPunctualLights(scene, out);

		return std::move(out);
	}

private:
	const pbrt_flatten::FlatScene &scene;
	BuildResult out;

	// Displacement images decoded once per file (bump vs normal map decided by pixel
	// content, once), shared by every material that names the same file - see
	// sharedMipmapTexture()'s comment for why per-material copies are not affordable.
	struct DispEntry { std::shared_ptr<texture> tex; bool grayscale = false; };
	std::map<std::string, DispEntry> dispCache;

	// One material instance per distinct (material, emission, forCurve) triple
	// - forCurve is part of the key (not just an argument materialFor reads)
	// so a materialIndex shared between a curve and a non-curve shape (e.g.
	// via NamedMaterial reuse - unusual but valid pbrt) gets two distinct
	// hair_material instances with the right tangent behavior each, instead
	// of whichever shape asks first silently winning for both.
	std::map<std::tuple<int, int, bool>, std::shared_ptr<material>> materialCache;

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
	// Decoded masks by file as well: each is width*height*3 floats (8192x8192 is
	// ~800 MB), and many materials of one asset name the same mask.
	std::map<std::string, std::shared_ptr<texture>> alphaMaskByFile;

	// addMediumIfPresent()'s own nanovdb branch (below) reads the .nvdb file
	// from disk and bakes its active region into a dense array EVERY time
	// it's called - unlike cloud/rgbgrid/uniformgrid (procedural or already
	// in-memory on `md`), a real file read plus an O(voxel count) bake is
	// expensive enough to be worth caching. Keyed on mediumIndex (matching
	// materialCache/alphaMaskCache's own per-index keying just above) rather
	// than the shape, since the built hittable doesn't depend on which
	// shape triggered it - see addMediumIfPresent()'s own comment on why
	// cloud/rgbgrid/uniformgrid/nanovdb all add an independent world-space
	// hittable rather than wrapping `shape` (only the homogeneous fallback
	// does that). A scene where N shapes share one MediumInterface
	// (a normal pbrt pattern) now reads+bakes the file once, not N times.
	std::map<int, std::shared_ptr<hittable>> nanovdbMediumCache;

	// forCurve: see makeMaterial's own comment - only affects the Hair case, picked by the one caller (the curve loop) that has real curve geometry.
	std::shared_ptr<material> materialFor(int materialIndex, int areaLightIndex, bool forCurve) {
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
			auto disp = dispCache.find(m.displacementTextureFilename);
			if (disp == dispCache.end()) {
				rtw_image disp_probe(m.displacementTextureFilename.c_str());
				DispEntry entry;
				if (disp_probe.height() > 0) {
					entry.grayscale = is_grayscale_image(disp_probe);
					// Bilinear + Repeat, like pbrt-v4's displacement/normal-map lookups: the nearest, clamped image_texture gave blocky
					// bump gradients and ignored tiled UVs. Texels are interpolated as FLOATS: a byte copy of the decoded values
					// (what this used to build) crushes the darks of an sRGB height map and turns a smooth ramp into stairs, so a bump
					// map's finite-difference slope came out wrong (bump-mapped-plane read 1-5% off the GPU, which decodes per texel).
					const auto texelsOf = [](const rtw_image &img) {
						std::vector<float> f(static_cast<std::size_t>(img.width()) * img.height() * 3);
						for (int y = 0; y < img.height(); ++y)
							for (int x = 0; x < img.width(); ++x) {
								const float *px = img.float_pixel_data(x, y);
								float *o = &f[(static_cast<std::size_t>(y) * img.width() + x) * 3];
								o[0] = px[0]; o[1] = px[1]; o[2] = px[2];
							}
						return f;
					};
					if (entry.grayscale) {
						// A scalar height map is a float imagemap: pbrt decodes a PNG as sRGB unless told otherwise (the default here).
						entry.tex = std::make_shared<bilinear_wrap_texture>(disp_probe.width(), disp_probe.height(),
							texelsOf(disp_probe), MipWrapMode::Repeat);
					} else {
						// A tangent-space normal map is read LINEAR (pbrt scene.cpp: Image::Read(filename, ..., ColorEncoding::Linear)); the
						// default image load above is sRGB-decoded, which tilts every texel (a flat 128,128,255 map would read as (-.57,-.57,1)).
						const rtw_image linear_probe(m.displacementTextureFilename.c_str(), 1.0f);
						if (linear_probe.height() > 0)
							entry.tex = std::make_shared<bilinear_wrap_texture>(linear_probe.width(), linear_probe.height(),
								texelsOf(linear_probe), MipWrapMode::Repeat);
					}
				}
				disp = dispCache.emplace(m.displacementTextureFilename, std::move(entry)).first;
			}
			if (disp->second.tex) {
				const bool grayscale = disp->second.grayscale;
				auto disp_tex = disp->second.tex;
				base = grayscale
					? std::static_pointer_cast<material>(
						  std::make_shared<bump_map_material>(disp_tex, base, m.displacementScale))
					: std::static_pointer_cast<material>(
						  std::make_shared<normal_map_material>(disp_tex, base));
			}
		}
		return base;
	}

	std::shared_ptr<material> cachedMaterial(int mi, int ai, bool forCurve = false) {
		const auto key = std::make_tuple(mi, ai, forCurve);
		auto it = materialCache.find(key);
		if (it != materialCache.end()) return it->second;
		auto made = materialFor(mi, ai, forCurve);
		materialCache.emplace(key, made);
		return made;
	}

	std::shared_ptr<texture> alphaMaskFor(int mi) {
		if (mi < 0 || static_cast<std::size_t>(mi) >= scene.materials.size()) return nullptr;
		const auto it = alphaMaskCache.find(mi);
		if (it != alphaMaskCache.end()) return it->second;
		std::shared_ptr<texture> mask;
		const std::string &fn = scene.materials[static_cast<std::size_t>(mi)].alphaTextureFilename;
		if (!fn.empty()) {
			const auto byFile = alphaMaskByFile.find(fn);
			if (byFile != alphaMaskByFile.end()) {
				mask = byFile->second;
			} else {
				int w = 0, h = 0, channels = 0;
				unsigned char *bdata = stbi_load(fn.c_str(), &w, &h, &channels, 3);
				if (bdata) {
					// Bilinear + Repeat, like pbrt-v4's alpha lookup (primitive.cpp:57-71 reads
					// the texture at mip level 0 with the default wrap). pbrt reads a float
					// imagemap from an 8-bit image as the mean of its channels, sRGB-decoded by
					// default (textures.cpp:436, mipmap.cpp:396-405), so a mid-grey mask of byte
					// 153 is 0.32, not 0.6. 3 bytes/pixel instead of the 12 the float copy needed.
					std::vector<unsigned char> bytes =
						srgb_decode::alphaMaskFromRgb8(bdata, static_cast<std::size_t>(w) * h);
					stbi_image_free(bdata);
					mask = std::make_shared<bilinear_wrap_texture>(w, h, std::move(bytes), MipWrapMode::Repeat);
				}
				alphaMaskByFile.emplace(fn, mask);
			}
		}
		alphaMaskCache.emplace(mi, mask);
		return mask;
	}

	// MediumInterface "insideMedium" - layers a participating medium inside a shape already added to `world`, or adds a world-space medium hittable
	// (see the comments in the body for which kinds wrap the shape and which do not).
	void addMediumIfPresent(hittable_list &world, const std::shared_ptr<hittable> &shape, int mediumIndex) {
		if (mediumIndex < 0 || static_cast<std::size_t>(mediumIndex) >= scene.media.size()) return;
		const pbrt_flatten::Medium &md = scene.media[static_cast<std::size_t>(mediumIndex)];
		out.hasShapeMedia = true;

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
				/*sigma_a=*/0.0, luminanceOf(md.sigma_s), md.g,
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
			auto gridMedium = std::make_shared<rgb_grid_medium_hittable>(
				grid, md.g, world_min, world_max, md.toMediumMat, md.toMediumTranslate);
			world.add(gridMedium);
			// camera::ray_color() samples a grid itself, per channel (rgb_grid_medium_hittable::sample_event); every other integrator
			// keeps the max-channel delta tracking of hit().
			out.chromaticMedia.push_back(gridMedium);
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
			// Not const: moved into grid_medium_hittable below (its own
			// constructor comment) rather than deep-copied a second time.
			GridMediumData<double> grid(
				md.gridDensity, md.nx, md.ny, md.nz, bounds,
				/*sa=*/0.0, luminanceOf(md.sigma_s), md.g);
			const point3 world_min(md.worldMin[0], md.worldMin[1], md.worldMin[2]);
			const point3 world_max(md.worldMax[0], md.worldMax[1], md.worldMax[2]);
			world.add(std::make_shared<grid_medium_hittable>(
				std::move(grid), color(1,1,1), md.g, world_min, world_max, md.toMediumMat, md.toMediumTranslate));
			return;
		}
		if (md.type == "nanovdb") {
			// Cache keyed on mediumIndex (nanovdbMediumCache's own comment,
			// defined above alongside materialCache/alphaMaskCache) - the
			// file read + bake below is genuinely expensive, and the built
			// hittable doesn't depend on `shape`, so a second shape sharing
			// this medium reuses the first shape's already-built hittable
			// instead of re-reading/re-baking the identical file.
			const auto cacheIt = nanovdbMediumCache.find(mediumIndex);
			if (cacheIt != nanovdbMediumCache.end()) {
				if (cacheIt->second) world.add(cacheIt->second);
				return;
			}
			// Real NanoVDB file support (Medium::nanovdbFilename's own
			// comment) - bakes the grid's active index region into a DENSE
			// flat array reusing GridMediumData<double>/grid_medium_
			// hittable.h completely unchanged, same as uniformgrid just
			// above. Unlike every other medium type here, the world-space
			// bounds/transform can't be resolved until the .nvdb file is
			// actually read (the grid's own extent isn't known from the
			// scene text alone) - so this block computes md.worldMin/
			// worldMax/toMediumMat/toMediumTranslate's local equivalents
			// itself, from md.nanovdbXform (the raw, unbaked scene CTM),
			// instead of reading them pre-computed off `md` the way cloud/
			// rgbgrid/uniformgrid do. Wrapped in an immediately-invoked
			// lambda returning the built hittable (or nullptr) rather than
			// calling world.add()/return directly, so every exit path also
			// populates nanovdbMediumCache above.
			const std::shared_ptr<hittable> nanovdbBuilt = bakeNanovdbMedium(md);
			nanovdbMediumCache.emplace(mediumIndex, nanovdbBuilt);
			if (nanovdbBuilt) world.add(nanovdbBuilt);
			return;
		}

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
		// Per-channel sigma_a/sigma_s go in as written; constant_medium keeps the scalar model (the luminance of sigma_t and a tint from
		// sigma_s - what this site used to build) for hit() and the other integrators, and samples the real per-channel one in the
		// default path tracer when the extinction differs between channels.
		auto medium = std::make_shared<constant_medium>(shape,
			color(md.sigma_a[0], md.sigma_a[1], md.sigma_a[2]), color(md.sigma_s[0], md.sigma_s[1], md.sigma_s[2]), md.g, Le);
		world.add(medium);
		if (medium->chromatic()) out.chromaticMedia.push_back(medium);
	}

	// Triangles: one shared mesh (vertices welded) plus one `triangle` per face.
	// trisMayHaveGpuOnlyFallback: see emitGeometry().
	void emitTriangles(const std::vector<pbrt_flatten::Triangle> &tris, hittable_list &world, hittable_list &lights,
					   bool trisMayHaveGpuOnlyFallback) {
		// Triangle::gpuOnlyStaticFallback entries (a StartTime-pose duplicate of
		// a mesh that's also in scene.animatedTriangleMeshes - see that field's
		// own comment) exist purely so GPU's own separate builder, which has no
		// concept of that list, doesn't lose the shape entirely. CPU gets its
		// real motion blur from animatedTriangleMeshes instead (built by this
		// same function's own animated-mesh call site below in
		// pbrt_cpu_builder.h's caller), so building a second, static `triangle`
		// hittable here too would double-render it - skipped inline below via
		// `trisMayHaveGpuOnlyFallback` (true only for the one call site that
		// could ever see a flagged entry) rather than a separate filtering pass
		// + pointer vector, which would otherwise be unconditional overhead paid
		// by every scene with mesh geometry, animated or not (a code-review pass
		// on this feature's own commit caught that cost).
		if (!tris.empty()) {
			auto mesh = std::make_shared<triangle_mesh_data>();
			std::unordered_map<VertexKey, int, detail::VertexKeyHash> seen;
			seen.reserve(tris.size() * 3 / 2 + 16);   // ~1.5 unique vertices per triangle on a typical mesh

			// A mesh either has a normal for every vertex or for none: `triangle`
			// gates interpolation on has_normals(), which is all-or-nothing, so a
			// partially filled list would index past the end. Same story for UV -
			// `has_uvs()` (triangle.h) is the same all-or-nothing gate.
			bool anyNormals = false;
			bool anyUVs = false;
			for (const pbrt_flatten::Triangle &t : tris) {
				if (trisMayHaveGpuOnlyFallback && t.gpuOnlyStaticFallback) continue;
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
				if (trisMayHaveGpuOnlyFallback && t.gpuOnlyStaticFallback) continue;
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
	}

	void emitSpheres(const std::vector<pbrt_flatten::Sphere> &sphs, hittable_list &world, hittable_list &lights) {
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
			if (!s.cpuMediumUnsupported) addMediumIfPresent(world, sp, s.medium);
		}
		out.sphereCount += sphs.size();
	}

	// Emitting geometry is now done more than once - for the scene itself, and
	// again for each instance definition, whose geometry stays in object space
	// and is placed by a transform rather than baked. Everything below is what
	// it always was; only its inputs and outputs became parameters.
	void emitGeometry(const std::vector<pbrt_flatten::Triangle> &tris,
					  const std::vector<pbrt_flatten::Sphere> &sphs,
					  const std::vector<pbrt_flatten::Disk> &disks,
					  const std::vector<pbrt_flatten::Cylinder> &cylinders,
					  const std::vector<pbrt_flatten::Cone> &cones,
					  const std::vector<pbrt_flatten::Paraboloid> &paraboloids,
					  const std::vector<pbrt_flatten::BilinearPatch> &patches,
					  const std::vector<pbrt_flatten::Curve> &curveDecls,
					  hittable_list &world, hittable_list &lights,
								  // True only for the main scene.triangles call site,
								  // and only when the scene actually has at least one
								  // animated mesh (Triangle::gpuOnlyStaticFallback's
								  // own comment) - both other call sites (ObjectInstance
								  // group definitions, the animated-mesh block itself)
								  // structurally never contain a flagged entry, so
								  // skipping the filtering pass entirely for them (the
								  // default here) is a correctness no-op, not just an
								  // optimization; see this lambda's own gpuOnlyStaticFallback
								  // handling just below for why a real per-scene "any
								  // flagged?" scan+allocation is otherwise unconditional
								  // overhead paid by every scene with mesh geometry,
								  // animated or not (a code-review pass on this feature's
								  // own commit caught that cost).
					  bool trisMayHaveGpuOnlyFallback = false) {
		const auto cached = [this](int mi, int ai, bool forCurve = false) { return cachedMaterial(mi, ai, forCurve); };
		const auto addMedium = [this, &world](const std::shared_ptr<hittable> &shape, int mediumIndex) {
			addMediumIfPresent(world, shape, mediumIndex);
		};

		emitTriangles(tris, world, lights, trisMayHaveGpuOnlyFallback);

		// ---- spheres ---------------------------------------------------------
		emitSpheres(sphs, world, lights);

		// ---- disks / cylinders -------------------------------------------------
		// Shape "disk"/"cylinder" - unlike Sphere, these keep their CTM unbaked
		// (see pbrt_flatten::Disk/Cylinder's own comment for why) and apply it at
		// intersection time via disk_hittable/cylinder_hittable, the same
		// ray-into-object-space technique transform_instance.h already uses for
		// object instancing. Extracted to buildDisksAndCylinders() above.
		buildDisksAndCylinders(disks, cylinders, world, lights,
								out.diskCount, out.cylinderCount,
								cached, addMedium);

		// ---- cones / paraboloids -----------------------------------------------
		// Shape "cone"/"paraboloid" - same unbaked-CTM technique as disk/cylinder
		// above. Real AreaLightSource/MediumInterface support now (see
		// pbrt_flatten::Cone/Paraboloid's own comment) - lights.add()/
		// addMediumIfPresent() the identical way disk/cylinder already are.
		// Extracted to buildConesAndParaboloids() above.
		buildConesAndParaboloids(cones, paraboloids, world, lights,
								  out.coneCount, out.paraboloidCount,
								  cached, addMedium);

		// ---- bilinear patches -------------------------------------------------
		// Shape "bilinearmesh" - see pbrt_flatten.h's BilinearPatch comment for
		// why only the single-patch form reaches here. bilinear_patch_hittable
		// (scenes_advanced.h) now overrides pdf_value()/random() the same way
		// quad does, so an emissive one is NEE-samplable, not just hittable.
		// Extracted to buildBilinearPatches() above.
		buildBilinearPatches(patches, world, lights, out.bilinearPatchCount, cached);

		// ---- curves ------------------------------------------------------------
		// Shape "curve" - see pbrt_flatten::Curve's own comment for scope (degree
		// 2/3 Bezier and cubic B-spline all convert down to cubic Bezier before
		// reaching here, already split into independent per-segment 4-control-
		// point Bezier curves by flatten()). One CurveShape<double> per
		// segment, wrapped in the existing curve_shape_hittable. width0/width1 are
		// re-lerped per segment (matching pbrt-v4's own Curve::Create,
		// shapes.cpp:894-895 exactly: Lerp(seg/nSegments, width0,width1)) so a
		// multi-segment strand tapers smoothly across its whole length rather than
		// each segment re-tapering its own full width0->width1 range. Extracted
		// to buildCurves() above.
		buildCurves(curveDecls, world, lights, out.curveCount, cached);
	}

	void emitInstances() {
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
	}

	void emitAnimatedMeshes() {
		// ---- animated meshes --------------------------------------------------
		// Real object motion blur (trianglemesh/plymesh/loopsubdiv) - see
		// pbrt_flatten::AnimatedTriangleMesh's own comment. Each entry's
		// triangles are already OBJECT space (not baked to world - that's the
		// entire reason this list exists separately from scene.triangles), so
		// they go through the exact same emitGeometry() reused for
		// ObjectInstance's own object-space geometry above, into a fresh scratch
		// hittable_list, then wrapped in animated_transform_instance (the
		// MotionState-based sibling of transform_instance used just above,
		// carrying a per-ray-time-resolved transform instead of one static one).
		// No light list is passed for the same reason the ObjectInstance loop
		// above passes a scratch one - pbrt_flatten.h already excludes an
		// emissive mesh from ever populating this list at all (falls back to a
		// static, StartTime-only bake instead, warned there), so passing a real
		// one here would never receive anything, and a scratch list keeps that
		// invariant visible rather than implying this path DOES enumerate
		// lights when it deliberately never does.
		if (!scene.animatedTriangleMeshes.empty()) {
			static const std::vector<pbrt_flatten::Sphere> kNoSpheres;
			static const std::vector<pbrt_flatten::Disk> kNoDisks;
			static const std::vector<pbrt_flatten::Cylinder> kNoCylinders;
			static const std::vector<pbrt_flatten::Cone> kNoCones;
			static const std::vector<pbrt_flatten::Paraboloid> kNoParaboloids;
			static const std::vector<pbrt_flatten::BilinearPatch> kNoBilinearPatches;
			static const std::vector<pbrt_flatten::Curve> kNoCurves;
			for (const pbrt_flatten::AnimatedTriangleMesh &atm : scene.animatedTriangleMeshes) {
				if (atm.triangles.empty()) continue;
				auto geometry = std::make_shared<hittable_list>();
				hittable_list unusedLights;
				emitGeometry(atm.triangles, kNoSpheres, kNoDisks, kNoCylinders,
							 kNoCones, kNoParaboloids, kNoBilinearPatches, kNoCurves,
							 *geometry, unusedLights);
				if (geometry->objects.empty()) continue;
				auto objectBVH = std::make_shared<bvh_node>(*geometry);
				pbrt_scene::Matrix4 o2w, o2wEnd;
				for (int k = 0; k < 16; ++k) {
					o2w.m[k]    = atm.xform[k];
					o2wEnd.m[k] = atm.xformEnd[k];
				}
				out.world->add(std::make_shared<animated_transform_instance>(objectBVH, o2w, o2wEnd));
			}
		}
	}

	void emitAnimatedBilinearPatches() {
		// ---- animated bilinear patches -----------------------------------------
		// Real object motion blur (Shape "bilinearmesh") - see pbrt_flatten::
		// AnimatedBilinearPatch's own comment. Each entry's corners are already
		// OBJECT space, so - exactly like the animated-mesh block above - it's
		// fed back through this SAME emitGeometry() (as a synthetic, one-entry
		// BilinearPatch list; emitGeometry has no opinion on whether `.p` holds
		// object- or world-space points) into a scratch hittable_list, then
		// wrapped in animated_transform_instance. Never emissive (AnimatedBilinearPatch's
		// own comment - flatten() excludes an emissive patch from this list
		// entirely), so a scratch (never-populated) light list, same reasoning
		// as the animated-mesh block above.
		if (!scene.animatedBilinearPatches.empty()) {
			static const std::vector<pbrt_flatten::Triangle> kNoTriangles;
			static const std::vector<pbrt_flatten::Sphere> kNoSpheres;
			static const std::vector<pbrt_flatten::Disk> kNoDisks;
			static const std::vector<pbrt_flatten::Cylinder> kNoCylinders;
			static const std::vector<pbrt_flatten::Cone> kNoCones;
			static const std::vector<pbrt_flatten::Paraboloid> kNoParaboloids;
			static const std::vector<pbrt_flatten::Curve> kNoCurves;
			for (const pbrt_flatten::AnimatedBilinearPatch &abp : scene.animatedBilinearPatches) {
				pbrt_flatten::BilinearPatch bp;
				for (int i = 0; i < 4; ++i) {
					bp.p[i][0] = abp.p[i][0]; bp.p[i][1] = abp.p[i][1]; bp.p[i][2] = abp.p[i][2];
				}
				bp.material = abp.material;
				const std::vector<pbrt_flatten::BilinearPatch> oneBp{bp};
				auto geometry = std::make_shared<hittable_list>();
				hittable_list unusedLights;
				emitGeometry(kNoTriangles, kNoSpheres, kNoDisks, kNoCylinders,
							 kNoCones, kNoParaboloids, oneBp, kNoCurves,
							 *geometry, unusedLights);
				if (geometry->objects.empty()) continue;
				pbrt_scene::Matrix4 o2w, o2wEnd;
				for (int k = 0; k < 16; ++k) {
					o2w.m[k]    = abp.xform[k];
					o2wEnd.m[k] = abp.xformEnd[k];
				}
				out.world->add(std::make_shared<animated_transform_instance>(
					geometry->objects[0], o2w, o2wEnd));
			}
		}
	}

	void emitAnimatedCurves() {
		// ---- animated curves ----------------------------------------------------
		// Real object motion blur (Shape "curve") - see pbrt_flatten::
		// AnimatedCurve's own comment. Each entry's control points are already
		// OBJECT space; unlike a single bilinear patch, a curve can have several
		// segments, so - exactly like the animated-mesh block above - they're
		// fed back through this SAME emitGeometry() (as a synthetic, one-entry
		// Curve list) into a scratch hittable_list, wrapped in ONE bvh_node
		// (matching the animated-mesh block's own reasoning: several segments
		// sharing one xform/xformEnd pair, cheaper as one small tree than one
		// animated_transform_instance per segment), then wrapped in
		// animated_transform_instance. Never emissive or ribbon-type
		// (AnimatedCurve's own comment - flatten() excludes both from this
		// list), so a scratch light list, same reasoning as the animated-mesh
		// block above.
		if (!scene.animatedCurves.empty()) {
			static const std::vector<pbrt_flatten::Triangle> kNoTriangles;
			static const std::vector<pbrt_flatten::Sphere> kNoSpheres;
			static const std::vector<pbrt_flatten::Disk> kNoDisks;
			static const std::vector<pbrt_flatten::Cylinder> kNoCylinders;
			static const std::vector<pbrt_flatten::Cone> kNoCones;
			static const std::vector<pbrt_flatten::Paraboloid> kNoParaboloids;
			static const std::vector<pbrt_flatten::BilinearPatch> kNoBilinearPatches;
			for (const pbrt_flatten::AnimatedCurve &ac : scene.animatedCurves) {
				pbrt_flatten::Curve c;
				c.cp = ac.cp;
				c.nSegments = ac.nSegments;
				c.width0 = ac.width0;
				c.width1 = ac.width1;
				c.curveType = ac.curveType;
				c.material = ac.material;
				const std::vector<pbrt_flatten::Curve> oneCurve{std::move(c)};
				auto geometry = std::make_shared<hittable_list>();
				hittable_list unusedLights;
				emitGeometry(kNoTriangles, kNoSpheres, kNoDisks, kNoCylinders,
							 kNoCones, kNoParaboloids, kNoBilinearPatches, oneCurve,
							 *geometry, unusedLights);
				if (geometry->objects.empty()) continue;
				auto objectBVH = std::make_shared<bvh_node>(*geometry);
				pbrt_scene::Matrix4 o2w, o2wEnd;
				for (int k = 0; k < 16; ++k) {
					o2w.m[k]    = ac.xform[k];
					o2wEnd.m[k] = ac.xformEnd[k];
				}
				out.world->add(std::make_shared<animated_transform_instance>(objectBVH, o2w, o2wEnd));
			}
		}
	}

	void accelerate() {
		// A flat list would make every ray test every primitive; these scenes are
		// the reason the BVH/kd-tree exists. scene.acceleratorType/
		// acceleratorSplitMethod are already fully resolved by flatten() (falls
		// back to "bvh"/"sah" for anything unrecognized, or combined with
		// object motion blur - see FlatScene::acceleratorType's own comment) -
		// kd_tree_hittable.h's KdTree<double,...> wrapper for "kdtree"; for
		// "bvh", bvh_node (real SAH, this project's pre-existing default) for
		// "sah", bvh_aggregate_hittable.h's BvhTree<double,...> wrapper for an
		// explicit "middle"/"equal"/"hlbvh". All produce the same converged
		// image over the same primitives - this only changes build strategy/
		// acceleration structure, not rendering behavior.
		if (!out.world->objects.empty()) {
			auto accelerated = std::make_shared<hittable_list>();
			if (scene.acceleratorType == "kdtree") {
				accelerated->add(std::make_shared<kd_tree_hittable>(*out.world, scene.acceleratorKdParams));
			} else if (scene.acceleratorSplitMethod == "middle") {
				accelerated->add(std::make_shared<bvh_aggregate_hittable>(
					*out.world, BvhSplitMethod::Middle, scene.acceleratorMaxNodePrims));
			} else if (scene.acceleratorSplitMethod == "equal") {
				accelerated->add(std::make_shared<bvh_aggregate_hittable>(
					*out.world, BvhSplitMethod::EqualCounts, scene.acceleratorMaxNodePrims));
			} else if (scene.acceleratorSplitMethod == "hlbvh") {
				accelerated->add(std::make_shared<bvh_aggregate_hittable>(
					*out.world, BvhSplitMethod::HLBVH, scene.acceleratorMaxNodePrims));
			} else {
				accelerated->add(std::make_shared<bvh_node>(*out.world));
			}
			out.world = accelerated;
		}
	}
};

} // namespace scene_builder_impl

// Turns flattened geometry into a BVH-accelerated world plus the light list the integrator samples (see scene_builder_impl::CpuSceneBuilder).
inline BuildResult build(const pbrt_flatten::FlatScene &scene) {
	return scene_builder_impl::CpuSceneBuilder(scene).run();
}

} // namespace pbrt_cpu
