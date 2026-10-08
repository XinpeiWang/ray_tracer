#pragma once
// pbrt_cpu_stages.h -- BuildResult and the stage helpers (media, materials, lights, camera medium) CpuSceneBuilder calls. Part of pbrt_cpu_builder.h, which includes it.

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


#include "pbrt_cpu_detail.h"

namespace pbrt_cpu {


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
	// The per-shape homogeneous media whose extinction differs between colour channels (constant_medium::chromatic()). camera::ray_color()
	// samples these itself against the nearest surface (constant_medium::sample_event()); the same objects are in the world, where their
	// hit() reports nothing while that integrator runs. Empty for every grey-medium scene.
	std::vector<std::shared_ptr<event_medium>> chromaticMedia;
	// True when any shape carries a participating medium (grey or not). SPPM, which has no volume model, says so for these scenes.
	bool hasShapeMedia = false;
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

// ---------------------------------------------------------------------------
// The following 7 helpers are sections extracted out of build()/emitGeometry()
// below verbatim (a pure refactor, no behavior change). build() has no class
// to hold shared state, so each takes an explicit small parameter list: the
// cachedMaterial/addMediumIfPresent lambdas it needs (as template parameters,
// so a real closure type is passed by reference rather than type-erased
// through std::function), the world/lights hittable_list it emits into, and
// the BuildResult count field(s) it accumulates into. Every one of these
// still routes materials/media through the SAME cachedMaterial/
// addMediumIfPresent closures its caller passes in (emitGeometry's own
// materialCache/alphaMaskCache/nanovdbMediumCache, shared across triangles/
// instances/animated-* too) - never a private cache of its own.

// Shape "disk"/"cylinder" - see emitGeometry's own comment (pbrt_cpu_builder.h)
// for why these keep their CTM unbaked and apply it at intersection time via
// disk_hittable/cylinder_hittable.
template <typename CachedMaterialFn, typename AddMediumFn>
inline void buildDisksAndCylinders(const std::vector<pbrt_flatten::Disk> &disks,
									const std::vector<pbrt_flatten::Cylinder> &cylinders,
									hittable_list &world, hittable_list &lights,
									std::size_t &diskCount, std::size_t &cylinderCount,
									const CachedMaterialFn &cachedMaterial,
									const AddMediumFn &addMediumIfPresent) {
	for (const pbrt_flatten::Disk &d : disks) {
		auto mat = cachedMaterial(d.material, d.areaLight);
		auto disk = std::make_shared<disk_hittable>(
			d.radius, d.innerRadius, d.height, degrees_to_radians(d.phiMaxDeg),
			toMatrix4(d.xform), toMatrix4(d.xformEnd), mat);
		world.add(disk);
		if (d.areaLight >= 0) lights.add(disk);
		addMediumIfPresent(disk, d.medium);
	}
	diskCount += disks.size();

	for (const pbrt_flatten::Cylinder &c : cylinders) {
		auto mat = cachedMaterial(c.material, c.areaLight);
		auto cyl = std::make_shared<cylinder_hittable>(
			c.radius, c.zMin, c.zMax, degrees_to_radians(c.phiMaxDeg),
			toMatrix4(c.xform), toMatrix4(c.xformEnd), mat);
		world.add(cyl);
		if (c.areaLight >= 0) lights.add(cyl);
		addMediumIfPresent(cyl, c.medium);
	}
	cylinderCount += cylinders.size();
}

// Shape "cone"/"paraboloid" - same unbaked-CTM technique as disk/cylinder
// above.
template <typename CachedMaterialFn, typename AddMediumFn>
inline void buildConesAndParaboloids(const std::vector<pbrt_flatten::Cone> &cones,
									  const std::vector<pbrt_flatten::Paraboloid> &paraboloids,
									  hittable_list &world, hittable_list &lights,
									  std::size_t &coneCount, std::size_t &paraboloidCount,
									  const CachedMaterialFn &cachedMaterial,
									  const AddMediumFn &addMediumIfPresent) {
	for (const pbrt_flatten::Cone &cn : cones) {
		auto mat = cachedMaterial(cn.material, cn.areaLight);
		auto cone = std::make_shared<cone_hittable>(
			cn.radius, cn.height, degrees_to_radians(cn.phiMaxDeg), toMatrix4(cn.xform), mat);
		world.add(cone);
		if (cn.areaLight >= 0) lights.add(cone);
		addMediumIfPresent(cone, cn.medium);
	}
	coneCount += cones.size();

	for (const pbrt_flatten::Paraboloid &pb : paraboloids) {
		auto mat = cachedMaterial(pb.material, pb.areaLight);
		auto para = std::make_shared<paraboloid_hittable>(
			pb.radius, pb.zMin, pb.zMax, degrees_to_radians(pb.phiMaxDeg), toMatrix4(pb.xform), mat);
		world.add(para);
		if (pb.areaLight >= 0) lights.add(para);
		addMediumIfPresent(para, pb.medium);
	}
	paraboloidCount += paraboloids.size();
}

// Shape "bilinearmesh" - see pbrt_flatten.h's BilinearPatch comment for why
// only the single-patch form reaches here. No medium support for this shape
// kind (unlike disk/cylinder/cone/paraboloid above), so no addMediumIfPresent
// parameter.
template <typename CachedMaterialFn>
inline void buildBilinearPatches(const std::vector<pbrt_flatten::BilinearPatch> &patches,
								  hittable_list &world, hittable_list &lights,
								  std::size_t &bilinearPatchCount,
								  const CachedMaterialFn &cachedMaterial) {
	for (const pbrt_flatten::BilinearPatch &bp : patches) {
		// gpuOnlyStaticFallback's own comment (emitGeometry, below): this
		// entry is a StartTime-pose duplicate of a patch CPU already renders
		// for real via scene.animatedBilinearPatches - skip it here or it
		// would double-render on CPU.
		if (bp.gpuOnlyStaticFallback) continue;
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
	bilinearPatchCount += patches.size();
}

// Shape "curve" - see pbrt_flatten::Curve's own comment for scope (degree 2/3
// Bezier and cubic B-spline all convert down to cubic Bezier before reaching
// here). One CurveShape<double> per segment, wrapped in curve_shape_hittable.
template <typename CachedMaterialFn>
inline void buildCurves(const std::vector<pbrt_flatten::Curve> &curveDecls,
						 hittable_list &world, hittable_list &lights,
						 std::size_t &curveCount,
						 const CachedMaterialFn &cachedMaterial) {
	for (const pbrt_flatten::Curve &cd : curveDecls) {
		// gpuOnlyStaticFallback's own comment (emitGeometry, below): this
		// entry is a StartTime-pose duplicate of a curve CPU already renders
		// for real via scene.animatedCurves - skip it here or it would
		// double-render on CPU.
		if (cd.gpuOnlyStaticFallback) continue;
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
	curveCount += curveDecls.size();
}

// ---- infinite/sky light -----------------------------------------------
// Image-based when pbrt_load::loadFile() successfully decoded one
// (imageWidth/imageHeight > 0 - see FlatScene::InfiniteLight's comment on why
// the decode happens there and not here or in flatten()). Falls back to the
// scene's constant L otherwise - either it never named an image, or naming
// one failed to resolve/decode (a warning was already recorded for that
// case).
inline void buildSkyOrPortal(const pbrt_flatten::FlatScene &scene, BuildResult &out) {
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
}

// ---- camera medium --------------------------------------------------------
// pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own comment) -
// already resolved by flatten() to a valid homogeneous-only,
// no-per-shape-medium-conflict index, or -1 if none/unsupported (both scope
// cuts already warned about there) - this is just the same Medium-struct-to-
// runtime-object construction addMediumIfPresent() (emitGeometry, below) does
// for a per-shape medium, minus the boundary shape. `luminance` is passed in
// (rather than duplicated here) so both call sites keep sharing the exact
// same RGB-to-scalar collapse formula.
template <typename LuminanceFn>
inline void buildCameraMedium(const pbrt_flatten::FlatScene &scene, BuildResult &out,
							   const LuminanceFn &luminance) {
	if (scene.cameraMediumIndex >= 0 &&
		static_cast<std::size_t>(scene.cameraMediumIndex) < scene.media.size()) {
		const pbrt_flatten::Medium &m = scene.media[static_cast<std::size_t>(scene.cameraMediumIndex)];
		// Collapsed to a scalar extinction + chromatic albedo tint via the
		// SAME `luminance` addMediumIfPresent() uses for a per-shape medium's
		// identical homogeneous branch - luminance-weighted (not a flat
		// per-channel average), tint derived from sigma_s alone (the
		// scattering-only single-scattering-albedo direction), Le passed RAW
		// since ambient_medium's own constructor (mirroring constant_medium's)
		// already does the sigma_a/sigma_t weighting.
		// Per-channel coefficients go to ambient_medium as they are: it keeps the scalar-extinction model for the integrators that use
		// it and samples the real per-channel one in camera::ray_color() (see constant_medium's RGB constructor). `luminance` is no
		// longer needed here.
		(void)luminance;
		out.cameraMedium = std::make_shared<ambient_medium>(
			color(m.sigma_a[0], m.sigma_a[1], m.sigma_a[2]), color(m.sigma_s[0], m.sigma_s[1], m.sigma_s[2]),
			m.g, color(m.Le[0], m.Le[1], m.Le[2]));
	}
}

// ---- punctual (delta) lights -----------------------------------------------
// LightSource point/spot/distant/goniometric/projection - see
// pbrt_flatten::PunctualLight's own comment for why this is a bridging job
// onto punctual_light_objects.h's existing constructors, already proven by
// this codebase's own C2-C6 showcase scenes, rather than new rendering math.
inline void buildPunctualLights(const pbrt_flatten::FlatScene &scene, BuildResult &out) {
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
}

// RGB-to-scalar collapse (Rec.709 weights) for a homogeneous medium's sigma_a/sigma_s - used by the per-shape medium handling and by the
// camera-medium block, so there is exactly one definition of the formula.
inline double luminanceOf(const double c[3]) {
	return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

// Reads the NanoVDB file a medium names and bakes its active region into a dense grid hittable; nullptr (with a message on stderr) if it cannot.
// The file read plus an O(voxel count) bake is expensive, which is why CpuSceneBuilder caches the result per medium index.
inline std::shared_ptr<hittable> bakeNanovdbMedium(const pbrt_flatten::Medium &md) {
	if (md.nanovdbFilename.empty()) return nullptr;

	std::vector<double> density;
	// Kelvin per voxel, same layout/resolution as density - filled
	// below only when md.nanovdbTemperatureGridName is non-empty
	// (see that field's own comment); empty means "no emission".
	std::vector<double> temperature;
	int nx = 0, ny = 0, nz = 0;
	double corner00[3] = {0,0,0}, cornerX[3] = {0,0,0}, cornerY[3] = {0,0,0}, cornerZ[3] = {0,0,0};
	bool ok = false;
	try {
		auto handle = nanovdb::io::readGrid(md.nanovdbFilename, md.nanovdbGridName);
		const auto* grid = handle.grid<float>();
		if (!grid) {
			// The named grid exists in the file but isn't a plain
			// float build (Vec3f/Mask/Fp4/Fp8/Fp16/FpN/etc - this
			// loader only reads float grids, disclosed in
			// docs/PBRT_SUPPORT.md). Unlike a corrupt/unreadable
			// file, readGrid() doesn't throw for this case, so
			// without an explicit message here the medium would
			// silently vanish with nothing in the log to explain
			// why - the exact gap the catch block below's own
			// "this file's one deliberate exception" comment claims
			// is already closed.
			std::cerr << "[pbrt_cpu_builder] nanovdb medium: grid \""
					  << md.nanovdbGridName << "\" in \"" << md.nanovdbFilename
					  << "\" is not a plain float grid (only float grids are "
						 "supported); the medium will render as empty (invisible)\n";
		} else {
			const auto bbox = grid->indexBBox();
			const auto bmin = bbox.min();
			const auto bmax = bbox.max();
			// bmin/bmax components are int32_t (nanovdb::Coord) read
			// straight from the file with no min<=max validation -
			// a corrupt/adversarial file can claim any individually
			// representable int32 pair, including a degenerate one
			// (e.g. bmin > bmax). Subtracting in int32 itself (the
			// prior version of this code did `bmax[0]-bmin[0]` before
			// ever casting) is signed-overflow UB for such a pair,
			// which could wrap to a small value that slips past the
			// kMaxVoxelsPerAxis cap below meant to reject exactly
			// this input - promoting to int64_t before subtracting
			// closes that off; the true difference of two int32
			// values always fits in int64_t, so this is exact, not
			// just "less wrong".
			const std::int64_t nx64 = static_cast<std::int64_t>(bmax[0]) - static_cast<std::int64_t>(bmin[0]) + 1;
			const std::int64_t ny64 = static_cast<std::int64_t>(bmax[1]) - static_cast<std::int64_t>(bmin[1]) + 1;
			const std::int64_t nz64 = static_cast<std::int64_t>(bmax[2]) - static_cast<std::int64_t>(bmin[2]) + 1;
			// Sanity cap, checked BEFORE any multiplication and
			// entirely in int64_t (so it can't itself overflow): a
			// real .nvdb grid's active bounding box is at most a few
			// thousand voxels per axis even for large production
			// assets, but a corrupt or maliciously crafted file
			// could in principle claim an extreme (still
			// individually int32-representable) bbox - nx*ny*nz
			// would then overflow even a 64-bit size_t computation,
			// silently under-allocating `density` below while the
			// bake loop still iterates the TRUE (huge) nx/ny/nz - a
			// real heap buffer overflow, not just a slow/huge
			// render. Capping each axis independently, before any
			// multiplication happens, closes that off entirely
			// (512^3 already safely fits in size_t with enormous
			// headroom) while still being far more generous than
			// any real or bundled test asset needs.
			constexpr std::int64_t kMaxVoxelsPerAxis = 512;
			if (nx64 > 0 && ny64 > 0 && nz64 > 0 &&
				nx64 <= kMaxVoxelsPerAxis && ny64 <= kMaxVoxelsPerAxis && nz64 <= kMaxVoxelsPerAxis) {
				nx = static_cast<int>(nx64);
				ny = static_cast<int>(ny64);
				nz = static_cast<int>(nz64);
				const auto& tree = grid->tree();
				// A cached accessor, not tree.getValue() directly -
				// this bake loop's access pattern is spatially
				// coherent (x fastest, matching the grid's own
				// internal locality), so a real accessor's cached
				// traversal state pays off here.
				auto acc = grid->getAccessor();
				density.resize(static_cast<std::size_t>(nx) * ny * nz);
				for (int z = 0; z < nz; ++z)
					for (int y = 0; y < ny; ++y)
						for (int x = 0; x < nx; ++x) {
							const nanovdb::Coord ijk(bmin[0] + x, bmin[1] + y, bmin[2] + z);
							density[(static_cast<std::size_t>(z) * ny + y) * nx + x] =
								static_cast<double>(acc.getValue(ijk));
						}

				// Real blackbody emission (Medium::
				// nanovdbTemperatureGridName's own comment): a
				// SECOND named grid in the same file, sampled at the
				// SAME index-space voxel coordinates as density
				// above (bmin[]+x/y/z) rather than re-deriving its
				// own active bbox - a real .nvdb fire/smoke asset's
				// temperature and density grids share the same
				// active region in practice, and nanovdb's own
				// accessor already degrades gracefully (returns the
				// grid's background value, typically 0) for any
				// index outside whatever active region the
				// temperature grid actually has, so a mismatched
				// bbox just means "no emission at the mismatched
				// voxels" rather than a crash or garbage read.
				if (!md.nanovdbTemperatureGridName.empty()) {
					try {
						auto tHandle = nanovdb::io::readGrid(md.nanovdbFilename, md.nanovdbTemperatureGridName);
						const auto* tGrid = tHandle.grid<float>();
						if (!tGrid) {
							std::cerr << "[pbrt_cpu_builder] nanovdb medium: temperature grid \""
									  << md.nanovdbTemperatureGridName << "\" in \"" << md.nanovdbFilename
									  << "\" is not a plain float grid (only float grids are "
										 "supported); blackbody emission is dropped\n";
						} else {
							auto tAcc = tGrid->getAccessor();
							temperature.resize(static_cast<std::size_t>(nx) * ny * nz);
							for (int z = 0; z < nz; ++z)
								for (int y = 0; y < ny; ++y)
									for (int x = 0; x < nx; ++x) {
										const nanovdb::Coord ijk(bmin[0] + x, bmin[1] + y, bmin[2] + z);
										temperature[(static_cast<std::size_t>(z) * ny + y) * nx + x] =
											static_cast<double>(tAcc.getValue(ijk));
									}
						}
					} catch (...) {
						std::cerr << "[pbrt_cpu_builder] nanovdb medium: failed to read temperature "
									 "grid \"" << md.nanovdbTemperatureGridName << "\" from \""
								  << md.nanovdbFilename << "\" (corrupt file or wrong gridname); "
									 "blackbody emission is dropped\n";
						temperature.clear();
					}
				}

				// Reconstruct the composed (index-[0,1]-space ->
				// world) affine map by sampling its origin + 3
				// unit-basis corners, rather than hand-deriving/
				// multiplying NanoVDB's own Map matrix convention
				// together with md.nanovdbXform - 4 points fully
				// determine any affine map, and grid->indexToWorld()
				// already folds in the grid's own baked voxel-size/
				// origin transform correctly regardless of what
				// convention it uses internally.
				pbrt_scene::Matrix4 sceneXform;
				for (int i = 0; i < 16; ++i) sceneXform.m[i] = md.nanovdbXform[i];
				auto mapCorner = [&](double u, double v, double w, double out[3]) {
					const nanovdb::Vec3d idx(bmin[0] + u * nx, bmin[1] + v * ny, bmin[2] + w * nz);
					const nanovdb::Vec3d native = grid->indexToWorld(idx);
					pbrt_flatten::flatten_detail::transformPoint(sceneXform, native[0], native[1], native[2], out);
				};
				mapCorner(0, 0, 0, corner00);
				mapCorner(1, 0, 0, cornerX);
				mapCorner(0, 1, 0, cornerY);
				mapCorner(0, 0, 1, cornerZ);
				ok = true;
			} else {
				// Either a degenerate bbox (bmin > bmax on some axis
				// - possible for a corrupt/adversarial file even
				// after the int64_t-safe subtraction above, which
				// only guarantees the VALUE is exact, not positive)
				// or a real, oversized active region past the
				// kMaxVoxelsPerAxis cap. Same "explain every nanovdb
				// degradation" rationale as the grid<float>() check
				// above and the catch block below.
				std::cerr << "[pbrt_cpu_builder] nanovdb medium: grid \""
						  << md.nanovdbGridName << "\" in \"" << md.nanovdbFilename
						  << "\" has an active bounding box of " << nx64 << "x" << ny64
						  << "x" << nz64 << " voxels, which is degenerate or exceeds the "
						  << kMaxVoxelsPerAxis << "-voxels-per-axis cap this loader "
						  << "enforces; the medium will render as empty (invisible)\n";
			}
		}
	} catch (...) {
		// Corrupt file, wrong/missing grid name, or any other NanoVDB
		// read failure - degrade the same way an empty/wrong-length
		// uniformgrid "density" array does just above: no hittable
		// added, an invisible medium, rather than propagating the
		// exception. pbrt_load.h already confirmed the file EXISTS
		// (path resolution); this catches everything path resolution
		// can't - a truncated/non-NanoVDB file, or a gridname pbrt_
		// flatten.h had no way to check without opening it.
		//
		// This file otherwise has NO console-output convention at
		// all (every other silent-degradation branch above/below
		// was already warned about earlier, at flatten()/pbrt_load.h
		// time, where this codebase's real warning infrastructure
		// lives) - this is a deliberate, disclosed exception: unlike
		// every sibling case, THIS failure mode genuinely cannot be
		// detected any earlier than here (opening and parsing the
		// file is the only way to know the gridname exists or the
		// bytes are valid NanoVDB), so without this line a scene
		// author who typos "gridname" or ships a corrupt .nvdb gets
		// zero indication anywhere why their medium vanished.
		std::cerr << "[pbrt_cpu_builder] nanovdb medium: failed to read grid \""
				  << md.nanovdbGridName << "\" from \"" << md.nanovdbFilename
				  << "\" (corrupt file, wrong gridname, or an unsupported "
					 "NanoVDB grid type - only plain float grids are read); "
					 "the medium will render as empty (invisible)\n";
		ok = false;
	}
	if (!ok || density.empty()) return nullptr;

	// world = worldFromMedium * (u,v,w) + corner00 - worldFromMedium's
	// columns are the 3 sampled basis differences, matching
	// pbrt_scene::Matrix4's own row-major layout exactly.
	pbrt_scene::Matrix4 worldFromMedium;
	for (int r = 0; r < 3; ++r) {
		worldFromMedium.m[r * 4 + 0] = cornerX[r] - corner00[r];
		worldFromMedium.m[r * 4 + 1] = cornerY[r] - corner00[r];
		worldFromMedium.m[r * 4 + 2] = cornerZ[r] - corner00[r];
		worldFromMedium.m[r * 4 + 3] = corner00[r];
	}
	pbrt_scene::Matrix4 mediumFromWorld;
	if (!worldFromMedium.inverseAffine(mediumFromWorld)) return nullptr;

	// World-space AABB of the unit cube [0,1]^3 (this medium's own
	// index-space bounds) under worldFromMedium, and the world<-
	// >medium matrix split - same shared helpers pbrt_flatten.h's
	// own cloud/rgbgrid/uniformgrid AABB/transform block uses
	// (pbrt_flatten::flatten_detail::aabbOfTransformedBox/
	// splitAffine), rather than each re-deriving the "transform 8
	// corners, take axis-aligned min/max" and "slice an affine
	// Matrix4 into mat9+translate3" logic by hand a second time.
	double worldMin[3], worldMax[3];
	const double unitLo[3] = {0.0, 0.0, 0.0}, unitHi[3] = {1.0, 1.0, 1.0};
	pbrt_flatten::flatten_detail::aabbOfTransformedBox(worldFromMedium, unitLo, unitHi, worldMin, worldMax);

	double toMediumMat[9], toMediumTranslate[3];
	pbrt_flatten::flatten_detail::splitAffine(mediumFromWorld, toMediumMat, toMediumTranslate);

	// sigma_a is forced to 0 (pure scattering) UNLESS a real
	// temperature grid was just baked above - same convention/
	// reason as uniformgrid above otherwise (flatten() already
	// warned if the scene gave a nonzero sigma_a with no
	// "temperaturename"); see Medium::nanovdbTemperatureGridName's
	// own comment for why blackbody emission needs a real sigma_a
	// to be anything other than a physical no-op.
	const bool hasEmission = !temperature.empty();
	const Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	// std::move: `density` is a disposable local (unlike
	// uniformgrid's own md.gridDensity above, a persistent member
	// it can't move out of) - for a large baked grid this avoids a
	// copy here. Not const: `grid` itself is moved into
	// grid_medium_hittable below (that constructor's own comment)
	// rather than deep-copied a second time - for a grid near the
	// 512-voxel-per-axis cap (up to ~1GB of doubles) that second
	// copy was a real, avoidable allocation spike.
	GridMediumData<double> grid(
		std::move(density), nx, ny, nz, bounds,
		/*sa=*/hasEmission ? luminanceOf(md.sigma_a) : 0.0, luminanceOf(md.sigma_s), md.g);
	if (hasEmission) {
		// Per-voxel Kelvin -> RGB (pbrt_flatten::blackbodyKelvinToRGB,
		// the same spectral pipeline resolveEmissionColor() uses for
		// a scene's flat "blackbody L"/"I"), de-interleaved into
		// three flat channel arrays matching GridMediumData::
		// set_emission()'s own expected shape (mirrors
		// RGBGridMediumData<T>::build()'s le_r/le_g/le_b split).
		//
		// blackbodyKelvinToRGB() is NOT cheap - it runs a real
		// spectral integration (BlackbodySpectrum -> SpectrumToXYZ,
		// ~1900 Blackbody()/FastExp() evaluations per call, see that
		// function's own comment) - and a real fire/smoke grid can
		// have tens of millions of voxels (up to the 512-per-axis
		// cap above), most sharing very similar temperatures (or, in
		// a sparse grid's inactive region, the exact same
		// background value). Quantizing to 10K buckets before
		// converting - well below anything visibly distinguishable
		// as a colour shift - and memoizing per bucket turns what
		// would otherwise be one full spectral bake per voxel into
		// at most a few hundred, independent of grid size.
		std::vector<double> le_r(temperature.size()), le_g(temperature.size()), le_b(temperature.size());
		std::map<int, pbrt_scene::Vec3> kelvinBucketCache;
		constexpr float kKelvinBucketSize = 10.0f;
		for (std::size_t i = 0; i < temperature.size(); ++i) {
			const float t = static_cast<float>(temperature[i]);
			const int bucket = static_cast<int>(std::lround(t / kKelvinBucketSize));
			auto [it, inserted] = kelvinBucketCache.try_emplace(bucket);
			if (inserted) {
				it->second = pbrt_flatten::blackbodyKelvinToRGB(bucket * kKelvinBucketSize);
			}
			le_r[i] = it->second.x; le_g[i] = it->second.y; le_b[i] = it->second.z;
		}
		grid.set_emission(std::move(le_r), std::move(le_g), std::move(le_b), md.nanovdbLeScale);
	}
	// Peak-memory note: `temperature` is no longer needed once le_r/
	// le_g/le_b above are built from it - freeing it here (rather
	// than leaving it to fall out of scope alongside density/grid
	// much later) cuts the transient peak from ~5x a single grid
	// array's size down to ~4x (density/grid + le_r + le_g + le_b)
	// for a temperaturename medium near the 512-voxel-per-axis cap.
	temperature.clear();
	temperature.shrink_to_fit();
	const point3 world_min(worldMin[0], worldMin[1], worldMin[2]);
	const point3 world_max(worldMax[0], worldMax[1], worldMax[2]);
	return std::make_shared<grid_medium_hittable>(
		std::move(grid), color(1,1,1), md.g, world_min, world_max, toMediumMat, toMediumTranslate);
}


} // namespace pbrt_cpu
