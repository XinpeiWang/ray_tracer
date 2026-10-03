#pragma once
// scenes_advanced.h -- Advanced technique showcase scenes (scenes 18-37 +
// Curve Fibers). Included by scenes.h umbrella header.
//
// Split into two halves: this file holds the hand-authored demo scenes,
// each built to exercise one specific technique (principled BSDF, hair
// shading, subsurface, DoF, bilinear patches, media, cameras, measured
// BRDFs, portal lights, procedural mesh geometry...). The imported third-party
// mesh gallery (scenes_mesh_gallery.h) is gone: its scenes are pbrt-backed now.

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "triangle.h"
#include "bvh.h"
#include "material.h"
#include "constant_medium.h"
#include "cloud_medium_hittable.h"
#include "rgb_grid_medium_hittable.h"
#include "curve_shape_hittable.h"
#include "../shared/rgb_nebula_generator.h"
#include "hair_material.h"
#include "principled_material.h"
#include "normal_map_materials.h"
#include "sky_light.h"
#include "punctual_light_objects.h"
#include "../shared/bilinear_patch.h"
#include "../shared/cameras.h"
#include "../shared/cornell_box_data.h"
#include "../shared/portal_image_infinite_light.h"
#include <memory>

// build_principled_showcase() deleted - B10 migrated to pbrt-backed, see
// pbrt_scenes/principled-showcase.pbrt and scene_registry_data.h's own
// entry (that file's own header comment documents the new, non-standard
// Material "principled" kind added to the pbrt loader for this migration).
// No other consumer (gpu/metal/metal_poc_scenes_b.mm's own scene is an
// independent hand-ported Metal implementation that never called this
// function).

// build_hair_fibers() deleted - B11 migrated to pbrt-backed, see
// pbrt_scenes/hair-fibers-scene.pbrt and scene_registry_data.h's own entry.
// No other consumer.

// build_normal_mapped_cornell() deleted - B12 migrated to pbrt-backed, see
// pbrt_scenes/normal-mapped-cornell.pbrt and scene_registry_data.h's own
// entry (that file's own header comment has the full derivation, including
// the two baked-PNG approximations for its procedural bump/normal sources).
// No other consumer (gpu/metal/metal_poc_scenes_b.mm's own scene is an
// independent hand-ported Metal implementation that never called this
// function).

// build_subsurface_slab() deleted - B13 migrated to pbrt-backed, see
// pbrt_scenes/subsurface-slab.pbrt and scene_registry_data.h's own entry.
// No other consumer.

// build_depth_of_field() deleted - D1 migrated to pbrt-backed, see
// pbrt_scenes/depth-of-field-spheres.pbrt and scene_registry_data.h's own
// entry. No other consumer (gpu/metal/metal_poc_scenes_d.mm's own
// buildDepthOfField() is an independent hand-ported Metal implementation
// that never called this function).

//==============================================================================================
// Scene 23: Bilinear Patch
// Cornell box containing a curved bilinear patch saddle surface demonstrating
// pbrt-v4 BilinearPatch intersection (non-planar quadrilateral).
// We wrap BilinearPatchShape<double> in a custom hittable adapter.
//==============================================================================================

// Inline hittable wrapper for BilinearPatchShape<double>
class bilinear_patch_hittable : public hittable {
public:
	bilinear_patch_hittable(
		point3 p00, point3 p10, point3 p01, point3 p11,
		shared_ptr<material> mat)
		: mat(mat)
	{
		shape.p00x = p00.x(); shape.p00y = p00.y(); shape.p00z = p00.z();
		shape.p10x = p10.x(); shape.p10y = p10.y(); shape.p10z = p10.z();
		shape.p01x = p01.x(); shape.p01y = p01.y(); shape.p01z = p01.z();
		shape.p11x = p11.x(); shape.p11y = p11.y(); shape.p11z = p11.z();

		// Compute bounding box from the 4 corners (conservative)
		double xmin = std::min({p00.x(),p10.x(),p01.x(),p11.x()});
		double xmax = std::max({p00.x(),p10.x(),p01.x(),p11.x()});
		double ymin = std::min({p00.y(),p10.y(),p01.y(),p11.y()});
		double ymax = std::max({p00.y(),p10.y(),p01.y(),p11.y()});
		double zmin = std::min({p00.z(),p10.z(),p01.z(),p11.z()});
		double zmax = std::max({p00.z(),p10.z(),p01.z(),p11.z()});
		bbox = aabb(interval(xmin-0.01,xmax+0.01), interval(ymin-0.01,ymax+0.01), interval(zmin-0.01,zmax+0.01));
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		double ro[3] = { r.origin().x(), r.origin().y(), r.origin().z() };
		double rd[3] = { r.direction().x(), r.direction().y(), r.direction().z() };
		auto h = shape.intersect(ro[0],ro[1],ro[2], rd[0],rd[1],rd[2],
			ray_t.min, ray_t.max);
		if (!h) return false;
		rec.t = h->t;
		rec.p = r.at(rec.t);
		vec3 outward_normal(h->nx, h->ny, h->nz);
		rec.set_face_normal(r, outward_normal);
		rec.u = h->u; rec.v = h->v;
		rec.mat = mat;
		// dpdu: approximate tangent along u direction
		float fro[3]={(float)ro[0],(float)ro[1],(float)ro[2]};
		float frd[3]={(float)rd[0],(float)rd[1],(float)rd[2]};
		float sq00[3]={(float)shape.p00x,(float)shape.p00y,(float)shape.p00z};
		float sq10[3]={(float)shape.p10x,(float)shape.p10y,(float)shape.p10z};
		float sq01[3]={(float)shape.p01x,(float)shape.p01y,(float)shape.p01z};
		float sq11[3]={(float)shape.p11x,(float)shape.p11y,(float)shape.p11z};
		float out_p[3], out_dpdu[3], out_dpdv[3];
		blp_point(sq00,sq10,sq01,sq11,(float)h->u,(float)h->v, out_p,out_dpdu,out_dpdv);
		rec.dpdu = vec3(out_dpdu[0], out_dpdu[1], out_dpdu[2]);
		return true;
	}

	aabb bounding_box() const override { return bbox; }

	// Solid-angle PDF for a direction from origin toward this patch - the
	// hittable NEE hooks quad.h overrides for the same reason (see its own
	// pdf_value/random pair). Backed by blp_pdf_wi (src/shared/bilinear_patch.h),
	// which is CPU_GPU-tagged and shared with the GPU closest-hit's own MIS
	// lookup, so the two backends cannot disagree about a light's pdf.
	double pdf_value(const point3& origin, const vec3& direction) const override {
		const float o[3] = {(float)origin.x(), (float)origin.y(), (float)origin.z()};
		const float d[3] = {(float)direction.x(), (float)direction.y(), (float)direction.z()};
		const float p00[3] = {(float)shape.p00x, (float)shape.p00y, (float)shape.p00z};
		const float p10[3] = {(float)shape.p10x, (float)shape.p10y, (float)shape.p10z};
		const float p01[3] = {(float)shape.p01x, (float)shape.p01y, (float)shape.p01z};
		const float p11[3] = {(float)shape.p11x, (float)shape.p11y, (float)shape.p11z};
		return static_cast<double>(blp_pdf_wi(p00, p10, p01, p11, o, d));
	}

	// Uniform-area sample, returned as the vector from origin to the sampled
	// point (unnormalized) - matching quad.h::random()'s own contract, which
	// every caller of a light's random() already expects.
	vec3 random(const point3& origin) const override {
		const float u2[2] = {(float)random_double(), (float)random_double()};
		const float p00[3] = {(float)shape.p00x, (float)shape.p00y, (float)shape.p00z};
		const float p10[3] = {(float)shape.p10x, (float)shape.p10y, (float)shape.p10z};
		const float p01[3] = {(float)shape.p01x, (float)shape.p01y, (float)shape.p01z};
		const float p11[3] = {(float)shape.p11x, (float)shape.p11y, (float)shape.p11z};
		float outP[3], outN[3], outPdf = 0.f;
		blp_sample(p00, p10, p01, p11, u2, outP, outN, &outPdf);
		return vec3(outP[0] - origin.x(), outP[1] - origin.y(), outP[2] - origin.z());
	}

private:
	BilinearPatchShape<double> shape;
	shared_ptr<material> mat;
	aabb bbox;
};

// build_bilinear_patch_scene() deleted - F1 migrated to pbrt-backed, see
// pbrt_scenes/bilinear-patch-scene.pbrt and scene_registry_data.h's own
// entry. No other consumer. bilinear_patch_hittable itself (above) is NOT
// deleted - the pbrt loader's own pbrt_cpu_builder.h builds real
// Shape "bilinearmesh" geometry with it directly, and unit tests exercise
// it too.

// ============================================================================
// Scene 24: HDRI Sky
// Open scene lit by a procedural gradient sky_light (pbrt-v4 ImageInfiniteLight).
// build_sky() lambda returns the sky; world has no emissive geometry.
// ============================================================================
inline hittable_list build_hdri_sky_world() {
	hittable_list world;
	// Ground and a few spheres to catch sky light
	auto ground = make_shared<lambertian>(color(0.4, 0.4, 0.4));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground));
	world.add(make_shared<sphere>(point3(-3, 1, 0), 1,
		make_shared<lambertian>(color(0.7, 0.3, 0.2))));
	world.add(make_shared<sphere>(point3(0, 1, 0), 1,
		make_shared<metal>(color(0.8, 0.8, 0.9), 0.05)));
	world.add(make_shared<sphere>(point3(3, 1, 0), 1,
		make_shared<dielectric>(1.5)));
	return world;
}

inline std::shared_ptr<sky_light> build_hdri_sky() {
	// Procedural gradient: blue sky above, warm horizon at equator
	// Build a 64x32 synthetic HDR image
	const int W = 64, H = 32;
	std::vector<float> hdr(W * H * 3);
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			float t = (float)y / (H - 1); // 0=top, 1=bottom
			// Sky: gradient from deep blue (top) -> pale blue (midpoint) -> warm orange (bottom)
			float r = 0.1f + 0.9f * t * t;
			float g = 0.3f + 0.4f * (1.f - std::abs(t - 0.5f) * 2.f);
			float b = 0.8f * (1.f - t * t);
			int idx = (y * W + x) * 3;
			hdr[idx]   = r;
			hdr[idx+1] = g;
			hdr[idx+2] = b;
		}
	}
	// Actually use the gradient built above via sky_light's raw-pixel-data
	// constructor (this used to build the image then discard it, returning
	// a flat solid_color sky instead - description promised a "procedural
	// gradient sky" but the render was a single flat blue tint).
	return std::make_shared<sky_light>(W, H, hdr.data(), 1.0);
}

// ============================================================================
// Helper: build a minimal Cornell box (walls only, no light geometry)
// Used by punctual-light Cornell scenes
// ============================================================================
inline hittable_list cornell_walls_no_light() {
	using namespace cornell_box_data;
	hittable_list world;
	// The 5 standard walls (green/red/ceiling/floor/back), no light quad -
	// scenes 25-29 are lit by a punctual light instead. Shares
	// cornell_box_data::kQuads[0..4] with GPU's build_punctual_light_walls()
	// so the two can't drift apart, same pattern as scene 0.
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		auto mat = make_shared<lambertian>(color(q.color.r, q.color.g, q.color.b));
		world.add(make_shared<quad>(
			point3(q.Q.x, q.Q.y, q.Q.z),
			vec3(q.u.x, q.u.y, q.u.z),
			vec3(q.v.x, q.v.y, q.v.z),
			mat));
	}
	// Two spheres
	world.add(make_shared<sphere>(point3(190,90,190), 90, make_shared<lambertian>(color(.73,.73,.73))));
	world.add(make_shared<sphere>(point3(370,120,380), 120, make_shared<metal>(color(0.8,0.8,0.9),0.1)));
	return world;
}

// build_spotlight_cornell()/build_spotlight_punct() (former scene 25 / C2)
// and build_distant_light_cornell()/build_distant_light_punct() (former
// scene 26 / C3) all deleted - migrated to pbrt-backed, see pbrt_scenes/
// cornell-spotlight.pbrt/cornell-distant-light.pbrt and scene_registry_data.h's
// own entries. Neither had any other consumer (unlike build_point_light_cornell()/
// build_point_light_punct() just below, which tests/integration/
// sppm_first_slice_test.cpp calls directly, so C4 keeps its CPU functions).

// ============================================================================
// Scene 27: Point Light Cornell
// Cornell box lit by a single overhead point light with 1/r^2 falloff
// ============================================================================
inline hittable_list build_point_light_cornell() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_point_light_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// Intensity scale matches build_spotlight_punct()'s calibrated 600000.0
	// (same PointLightData::eval_Li = intensity/r^2 formula, similar height
	// above the floor) - the previous 5000000.0 was ~8x too bright, blowing
	// the room to near-white.
	pl->add_point(
		point3(278, 540, 278),          // overhead center
		color(1.0, 0.98, 0.90),         // warm white
		600000.0                        // intensity
	);
	return pl;
}

// build_goniometric_light_scene()/build_goniometric_punct() (former scene 28
// / C5) and build_projection_light_scene()/build_projection_punct() (former
// scene 29 / C6) both deleted - migrated to pbrt-backed, see pbrt_scenes/
// cornell-goniometric.pbrt/cornell-projection.pbrt and scene_registry_data.h's
// own entries. Neither had any other consumer.

// build_homogeneous_medium_scene() deleted - E1 migrated to pbrt-backed,
// see pbrt_scenes/homogeneous-medium.pbrt and scene_registry_data.h's own
// entry. No other consumer.

// build_cloud_medium_scene() deleted - E2 migrated to pbrt-backed, see
// pbrt_scenes/cloud-medium-scene.pbrt and scene_registry_data.h's own
// entry. No other consumer.

// build_dielectric_medium_scene() deleted - E3 migrated to pbrt-backed, see
// pbrt_scenes/dielectric-medium-showcase.pbrt and scene_registry_data.h's
// own entry. No other consumer.

// build_rgb_grid_medium_scene() deleted - E4 migrated to pbrt-backed, see
// pbrt_scenes/rgb-grid-nebula.pbrt and scene_registry_data.h's own entry
// (the pbrt file's own header comment has the full derivation of its baked
// "rgb sigma_s" array from generate_nebula_channel()). No other consumer
// (gpu/metal/metal_poc_scenes_e.mm's own scene is an independent
// hand-ported Metal implementation that never called this function).

// build_ortho_camera_scene()/build_ortho_sky() deleted - D2 migrated to
// pbrt-backed, see pbrt_scenes/ortho-camera-scene.pbrt and
// scene_registry_data.h's own entry. Neither had any other consumer.

// build_spherical_camera_scene()/build_spherical_sky() deleted - D3
// migrated to pbrt-backed, see pbrt_scenes/spherical-camera-scene.pbrt and
// scene_registry_data.h's own entry. Neither had any other consumer.

// measured_material/build_measured_brdf_scene() (former B14) deleted - B14
// migrated to pbrt-backed, see pbrt_scenes/measured-brdf-showroom.pbrt and
// scene_registry_data.h's own entry for the full derivation (a real
// fidelity improvement: measured_material's own scatter() never read its
// MeasuredBRDFData member at all, byte-for-byte a mislabeled Lambertian -
// the pbrt-backed version uses this project's REAL, working, importance-
// sampled Measured BRDF support instead). No other consumer.

// build_portal_light_scene()/build_portal_sky() deleted - C7 migrated to
// pbrt-backed, see pbrt_scenes/portal-window-room.pbrt and
// scene_registry_data.h's own entry. Neither had any other consumer.

// build_realistic_camera_scene() deleted - D4 migrated to pbrt-backed, see
// pbrt_scenes/realistic-camera-scene.pbrt and scene_registry_data.h's own
// entry. No other consumer.

// build_triangle_mesh_scene() deleted - F2 migrated to pbrt-backed, see
// pbrt_scenes/triangle-mesh-scene.pbrt and scene_registry_data.h's own
// entry. No other consumer.

//==============================================================================================
// Scene F4: Curve Fibers
// A windswept tuft of real Bezier curve strands (CurveShape<double>, Cylinder
// type, root-to-tip tapered width) rooted in a Fibonacci-disk arrangement -
// genuine curved geometry with a true ray-curve intersection test, unlike
// scene B11 (Hair Fibers), which applies HairBxDF shading to ordinary spheres
// rather than curving the geometry itself (see build_hair_fibers()'s
// comment). Colored with the same five hair-tone palette as B11 so the two
// scenes read as companions: B11 shows the shading model, this one shows the
// actual fiber shape a real strand traces.
//==============================================================================================
// build_curve_fibers_scene() deleted - F4 migrated to pbrt-backed, see
// pbrt_scenes/curve-fibers-scene.pbrt and scene_registry_data.h's own
// entry. No other consumer.


// scenes_mesh_gallery.h (the imported third-party mesh gallery, G1-G24 and the
// H1-H12 environment scenes) deleted - every scene in it is pbrt-backed now, see
// pbrt_scenes/mesh-*.pbrt and pbrt_scenes/environment-*.pbrt.
