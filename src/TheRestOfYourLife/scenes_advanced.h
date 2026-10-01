#pragma once
// scenes_advanced.h -- Advanced technique showcase scenes (scenes 18-37 +
// Curve Fibers). Included by scenes.h umbrella header.
//
// Split into two halves: this file holds the hand-authored demo scenes,
// each built to exercise one specific technique (principled BSDF, hair
// shading, subsurface, DoF, bilinear patches, media, cameras, measured
// BRDFs, portal lights, procedural mesh geometry...). scenes_mesh_gallery.h
// (#include'd at the bottom of this file) holds the imported third-party
// mesh gallery (Stanford models onward) -- a much more rigid, repetitive
// template of "load an external asset, put it under a light" that doesn't
// share this half's per-scene variety.

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

//==============================================================================================
// Scene 18: Principled Showcase
// A row of 7 spheres demonstrating the principled BSDF parameter space:
//   matte diffuse -> plastic -> semi-metallic -> fully metallic -> clearcoated metal
//==============================================================================================
inline hittable_list build_principled_showcase() {
	hittable_list world;

	// Ground plane (subtle dark checker)
	auto checker = make_shared<checker_texture>(0.5, color(0.1, 0.1, 0.12), color(0.2, 0.2, 0.22));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));

	// Row of 7 spheres: metallic 0->1, roughness varies, clearcoat on last two.
	// Spacing 2.0 (radius 1.0 each) so neighbors don't overlap/fuse -- each
	// material needs to read as its own distinct sphere.
	//  0: pure matte diffuse (red)
	world.add(make_shared<sphere>(point3(-6, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.1, 0.1), 0.0, 0.9, 1.5, 0.0)));
	//  1: plastic, low roughness (blue)
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.2, 0.8), 0.0, 0.2, 1.5, 0.0)));
	//  2: plastic, clearcoated (green)
	world.add(make_shared<sphere>(point3(-2, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.7, 0.2), 0.0, 0.3, 1.5, 1.0, 0.05)));
	//  3: semi-metallic (gold-tinted)
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.2), 0.5, 0.3, 1.5, 0.0)));
	//  4: near-metallic, rough (copper-ish)
	world.add(make_shared<sphere>(point3(2, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.45, 0.2), 0.8, 0.4, 1.5, 0.0)));
	//  5: fully metallic, smooth (silver)
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.9, 0.9), 1.0, 0.05, 1.5, 0.0)));
	//  6: fully metallic, clearcoated (lacquered gold)
	world.add(make_shared<sphere>(point3(6, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.1), 1.0, 0.1, 1.5, 1.0, 0.08)));

	// Overhead area light -- without a real light source the clearcoat/
	// metallic spheres show no specular highlight, defeating the point of
	// the demo (matches build_rough_metal_spheres()'s own light style).
	world.add(make_shared<quad>(point3(-7, 7, -5), vec3(14, 0, 0), vec3(0, 0, 10),
		make_shared<diffuse_light>(color(6, 6, 6))));

	return world;
}

// build_hair_fibers() deleted - B11 migrated to pbrt-backed, see
// pbrt_scenes/hair-fibers-scene.pbrt and scene_registry_data.h's own entry.
// No other consumer.

//==============================================================================================
// Scene 20: Normal Mapped Cornell Box
// Cornell box where the back wall has a bump-mapped wavy displacement and the
// sphere has a procedural normal-map perturbation, showing pbrt-v4 NormalMap/BumpMap.
//==============================================================================================
inline hittable_list build_normal_mapped_cornell() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls (plain)
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Back wall: bump-mapped wavy surface (uses marble Perlin noise as displacement)
	auto marble_tex = make_shared<noise_texture>(4.0);
	auto white_base = make_shared<lambertian>(color(.73, .73, .73));
	auto bumped_back = make_shared<bump_map_material>(marble_tex, white_base, 0.08, 0.002);
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), bumped_back));

	// Sphere: normal-map perturbed lambertian (checker pattern as normal source)
	// The checker drives subtle low-frequency normal variation over the sphere surface
	auto norm_tex = make_shared<checker_texture>(8.0, color(0.5, 0.5, 1.0), color(0.8, 0.8, 1.0));
	auto blue_base = make_shared<lambertian>(color(0.2, 0.3, 0.8));
	auto normal_sphere_mat = make_shared<normal_map_material>(norm_tex, blue_base);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, normal_sphere_mat));

	// Rotated box: bump-mapped white
	auto bumped_box = make_shared<bump_map_material>(marble_tex, white_base, 0.05, 0.002);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), bumped_box);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

// build_subsurface_slab() deleted - B13 migrated to pbrt-backed, see
// pbrt_scenes/subsurface-slab.pbrt and scene_registry_data.h's own entry.
// No other consumer.

//==============================================================================================
// Scene 22: Depth of Field
// An open-air scene with spheres at varying depths, rendered with defocus blur.
// The camera is focused at distance 3.5 with a wide aperture to exaggerate DOF.
// Camera setup (defocus_angle, focus_dist) is applied in cpu_interface via CameraConfig.
//==============================================================================================
inline hittable_list build_depth_of_field() {
	hittable_list world;

	// Checker ground
	auto checker = make_shared<checker_texture>(0.5, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));

	// Background sphere (out of focus far)
	world.add(make_shared<sphere>(point3(-4, 1, -3), 1.0,
		make_shared<lambertian>(color(0.4, 0.2, 0.1))));
	// Near sphere (out of focus near)
	world.add(make_shared<sphere>(point3(-1.5, 0.5, 1.5), 0.5,
		make_shared<lambertian>(color(0.7, 0.3, 0.3))));
	// In-focus center sphere (glass)
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0,
		make_shared<dielectric>(1.5)));
	// In-focus metal sphere
	world.add(make_shared<sphere>(point3(2, 1, 0), 1.0,
		make_shared<metal>(color(0.7, 0.6, 0.5), 0.05)));
	// Far sphere (out of focus)
	world.add(make_shared<sphere>(point3(4, 1, -2), 1.0,
		make_shared<lambertian>(color(0.1, 0.2, 0.6))));
	// Small near spheres
	for (int i = -3; i <= 3; ++i) {
		world.add(make_shared<sphere>(point3(i * 1.2, 0.2, 2.5 + i * 0.3), 0.2,
			make_shared<lambertian>(color(random_double(0.3,0.9), random_double(0.3,0.9), random_double(0.3,0.9)))));
	}

	return world;
}

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

// ============================================================================
// Scene E4: RGB Grid Medium ("nebula")
// A heterogeneous medium with an independent per-voxel R/G/B scattering
// grid (pbrt-v4 RGBGridMedium / src/shared/rgb_grid_medium.h), wired up for
// the first time - previously only unit-tested, never used by any scene
// (the same "built but unwired" state CloudMedium was in before scene E2).
// Unlike E2's CloudMedium (procedural, evaluated analytically per point at
// render time), this stores real per-voxel data in a grid baked once at
// scene-build time via generate_nebula_channel() (src/shared/
// rgb_nebula_generator.h) - the SAME generator GPU's build_rgb_grid_medium_
// scene_gpu() calls, so both backends render identical voxel data. Each
// channel uses a different frequency-space offset so R/G/B genuinely
// decorrelate into visible color variation, not just a uniformly-tinted
// cloud.
// ============================================================================
inline hittable_list build_rgb_grid_medium_scene() {
	hittable_list world;
	// Ground
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000,
								 make_shared<lambertian>(color(0.4, 0.5, 0.3))));

	const int nx = 24, ny = 24, nz = 24;
	std::vector<double> sa_zero(static_cast<size_t>(nx)*ny*nz, 0.0);  // sigma_a=0: pure
	                                                                  // scattering (see
	                                                                  // rgb_grid_medium_hittable.h)
	std::vector<double> ss_r, ss_g, ss_b;
	generate_nebula_channel<double>(nx, ny, nz, 3.0, 0.0,  0.0,  0.0,  ss_r);
	generate_nebula_channel<double>(nx, ny, nz, 3.0, 5.2,  1.7,  3.3,  ss_g);
	generate_nebula_channel<double>(nx, ny, nz, 3.0, 11.4, 8.8,  2.1,  ss_b);

	Bounds3<double> unit_cube(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	auto grid = RGBGridMediumData<double>::build(
		sa_zero, sa_zero, sa_zero,   // sigma_a: zero everywhere (must be non-empty
		                             // to actually get 0 - see build()'s own doc:
		                             // an OMITTED grid defaults to 1, not 0)
		ss_r, ss_g, ss_b,            // sigma_s: the nebula's actual per-channel density
		{}, {}, {},                  // Le: no emission
		nx, ny, nz,
		unit_cube,
		4.0,   // sigma_scale: overall density multiplier, tuned so the ~8-unit
		       // box reads as a real volumetric nebula rather than a faint haze
		0.0,   // Le_scale: unused (no emission)
		0.2,   // phase_g: slight forward scattering, matches E1/E2's fog
		16     // maj_res: majorant grid resolution (DDA acceleration)
	);

	point3 world_min(-4, 1, -4), world_max(4, 5, 4);
	double sx = 1.0 / (world_max.x() - world_min.x());
	double sy = 1.0 / (world_max.y() - world_min.y());
	double sz = 1.0 / (world_max.z() - world_min.z());
	double world_to_medium_mat[9] = { sx,0,0,  0,sy,0,  0,0,sz };
	double world_to_medium_translate[3] = {
		-world_min.x()*sx, -world_min.y()*sy, -world_min.z()*sz
	};
	world.add(make_shared<rgb_grid_medium_hittable>(
		grid, 0.2, world_min, world_max,
		world_to_medium_mat, world_to_medium_translate));

	// Context spheres, same idea as E2's - clearly outside the nebula's own
	// x:[-4,4] extent so they read as separate objects, not buried in it.
	world.add(make_shared<sphere>(point3(-6, 0.5, 4), 0.5,
								 make_shared<lambertian>(color(0.9, 0.3, 0.2))));
	world.add(make_shared<sphere>(point3(6, 0.5, 4), 0.5,
								 make_shared<metal>(color(0.8,0.8,0.9), 0.05)));
	return world;
}

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


// ============================================================================
// Imported third-party mesh gallery (Stanford models onward) -- see this
// file's own header comment above for why it's split out.
// ============================================================================
#include "scenes_mesh_gallery.h"
