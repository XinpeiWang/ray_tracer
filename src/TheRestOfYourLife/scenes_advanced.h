#pragma once
// scenes_advanced.h -- Advanced technique showcase scenes (scenes 18-36)
// Included by scenes.h umbrella header.

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "material.h"
#include "constant_medium.h"
#include "hair_material.h"
#include "principled_material.h"
#include "normal_map_materials.h"
#include "sky_light.h"
#include "punctual_light_objects.h"
#include "../shared/bilinear_patch.h"
#include "../shared/cameras.h"
#include "../shared/cornell_box_data.h"
#include "../shared/measured_bxdf.h"
#include "../shared/portal_image_infinite_light.h"
#include "mesh.h"
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

	// Row of 7 spheres: metallic 0->1, roughness varies, clearcoat on last two
	//  0: pure matte diffuse (red)
	world.add(make_shared<sphere>(point3(-3, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.1, 0.1), 0.0, 0.9, 1.5, 0.0)));
	//  1: plastic, low roughness (blue)
	world.add(make_shared<sphere>(point3(-2, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.2, 0.8), 0.0, 0.2, 1.5, 0.0)));
	//  2: plastic, clearcoated (green)
	world.add(make_shared<sphere>(point3(-1, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.7, 0.2), 0.0, 0.3, 1.5, 1.0, 0.05)));
	//  3: semi-metallic (gold-tinted)
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.2), 0.5, 0.3, 1.5, 0.0)));
	//  4: near-metallic, rough (copper-ish)
	world.add(make_shared<sphere>(point3(1, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.45, 0.2), 0.8, 0.4, 1.5, 0.0)));
	//  5: fully metallic, smooth (silver)
	world.add(make_shared<sphere>(point3(2, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.9, 0.9), 1.0, 0.05, 1.5, 0.0)));
	//  6: fully metallic, clearcoated (lacquered gold)
	world.add(make_shared<sphere>(point3(3, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.1), 1.0, 0.1, 1.5, 1.0, 0.08)));

	return world;
}

//==============================================================================================
// Scene 19: Hair Fibers
// A cluster of spheres using the hair_material to mimic fur/fiber appearance.
// Each sphere uses slightly different hair parameters (color, roughness) to show variety.
//==============================================================================================
inline hittable_list build_hair_fibers() {
	hittable_list world;

	// Dark floor
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
		make_shared<lambertian>(color(0.05, 0.05, 0.06))));

	// Dark brown hair (default sigma_a)
	world.add(make_shared<sphere>(point3(-2.5, 1, 0), 1.0,
		make_shared<hair_material>(0.06, 0.10, 0.20, 0.25, 0.25, 2.0)));
	// Blonde hair (low absorption, warm tint)
	world.add(make_shared<sphere>(point3(-0.8, 1, 0.3), 1.0,
		make_shared<hair_material>(0.01, 0.015, 0.03, 0.30, 0.30, 2.0)));
	// Auburn hair (strong red absorption pattern)
	world.add(make_shared<sphere>(point3(0.9, 1, -0.3), 1.0,
		make_shared<hair_material>(0.02, 0.08, 0.18, 0.20, 0.20, 3.0)));
	// White/silver fur (very low absorption, rough)
	world.add(make_shared<sphere>(point3(2.5, 1, 0), 1.0,
		make_shared<hair_material>(0.001, 0.001, 0.002, 0.45, 0.45, 1.0)));
	// Fine black fur (very high absorption)
	world.add(make_shared<sphere>(point3(0, 1, 1.8), 1.0,
		make_shared<hair_material>(0.50, 0.55, 0.60, 0.15, 0.15, 2.0)));

	return world;
}

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

//==============================================================================================
// Scene 21: Subsurface Slab
// Cornell box with layered constant_medium objects that approximate subsurface
// scattering: a translucent milk-white slab and a jade-green sphere.
// True BSSRDF would require a separate path-length sampling loop;
// here we use constant_medium (homogeneous participating media) to achieve
// a similar translucent glow that shows light scattering inside the object.
//==============================================================================================
inline hittable_list build_subsurface_slab() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(12, 12, 12));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Translucent wax slab: opaque glass boundary + dense scattering interior
	// Outer shell is dielectric (lets light in/out), interior is constant_medium (milky scattering)
	auto glass = make_shared<dielectric>(1.4);
	shared_ptr<hittable> slab = box(point3(0,0,0), point3(200,300,160), glass);
	slab = make_shared<translate>(slab, vec3(270,0,230));
	world.add(slab);
	// Volumetric fill inside the slab (milky white, moderate density)
	shared_ptr<hittable> slab_vol = box(point3(0,0,0), point3(200,300,160), glass);
	slab_vol = make_shared<translate>(slab_vol, vec3(270,0,230));
	world.add(make_shared<constant_medium>(slab_vol, 0.04, color(0.98, 0.96, 0.90)));

	// Jade sphere: glass shell + green scattering interior
	auto jade_glass = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(160, 90, 160), 90, jade_glass));
	world.add(make_shared<constant_medium>(
		make_shared<sphere>(point3(160, 90, 160), 90, jade_glass),
		0.06, color(0.1, 0.5, 0.2)));

	return world;
}

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

inline hittable_list build_bilinear_patch_scene() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Bilinear patch saddle surface: four corners with different Y heights
	// p00=bottom-left, p10=bottom-right, p01=top-left, p11=top-right
	// Saddle: p00 and p11 high, p10 and p01 low (classic hyperbolic paraboloid)
	auto patch_mat = make_shared<metal>(color(0.8, 0.7, 0.3), 0.05);
	world.add(make_shared<bilinear_patch_hittable>(
		point3(150,  80, 200),   // p00 (u=0,v=0) -- higher
		point3(400,  50, 200),   // p10 (u=1,v=0) -- lower
		point3(150,  50, 400),   // p01 (u=0,v=1) -- lower
		point3(400,  80, 400),   // p11 (u=1,v=1) -- higher
		patch_mat
	));

	// Second patch: curved ramp (linear in u, curved in v)
	auto blue_mat = make_shared<metal>(color(0.2, 0.4, 0.8), 0.1);
	world.add(make_shared<bilinear_patch_hittable>(
		point3(200, 200, 220),   // p00
		point3(370, 200, 220),   // p10
		point3(150, 380, 420),   // p01
		point3(420, 320, 420),   // p11
		blue_mat
	));

	return world;
}

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
			hdr[idx]   = r * 3.0f;  // HDR over-exposure
			hdr[idx+1] = g * 3.0f;
			hdr[idx+2] = b * 4.0f;
		}
	}
	// sky_light accepts a texture; build a solid-color sky for simplicity
	// and layer with a gradient_texture
	auto sky = std::make_shared<sky_light>(color(0.3, 0.6, 1.0));
	return sky;
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

// ============================================================================
// Scene 25: Spotlight Cornell
// Cornell box walls lit by a spotlight with smooth penumbra (pbrt-v4 SpotLight)
// ============================================================================
inline hittable_list build_spotlight_cornell() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_spotlight_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// Spotlight aimed downward from ceiling center
	pl->add_spot(
		point3(278, 548, 278),          // position: near ceiling
		vec3(0, -1, 0),                 // direction: straight down
		color(1.0, 0.95, 0.85),         // warm white
		30.0,                           // total cone width (degrees)
		15.0,                           // inner cone / falloff start (degrees)
		600000.0                        // intensity scale
	);
	return pl;
}

// ============================================================================
// Scene 26: Distant Light Cornell
// Cornell box lit by a parallel sun-like DistantLight (pbrt-v4 DistantLight)
// ============================================================================
inline hittable_list build_distant_light_cornell() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_distant_light_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// Sun-like light coming from upper-right through the open top
	pl->add_distant(
		vec3(-0.4, -1.0, -0.2),         // direction (toward light is negated inside)
		color(1.0, 0.98, 0.92),         // warm sunlight
		1000.0,                         // scene radius
		800000.0                        // radiance scale
	);
	return pl;
}

// ============================================================================
// Scene 27: Point Light Cornell
// Cornell box lit by a single overhead point light with 1/r^2 falloff
// ============================================================================
inline hittable_list build_point_light_cornell() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_point_light_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	pl->add_point(
		point3(278, 540, 278),          // overhead center
		color(1.0, 0.98, 0.90),         // warm white
		5000000.0                       // intensity
	);
	return pl;
}

// ============================================================================
// Scene 28: Goniometric Light
// Cornell box lit by a goniometric (IES-like) point light
// ============================================================================
inline hittable_list build_goniometric_light_scene() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_goniometric_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// Build a synthetic goniometric profile: bright spot forward, dimmer backward
	// 16x8 greyscale image; maps to equal-area sphere UV
	const int NU = 16, NV = 8;
	std::vector<double> img(NU * NV);
	for (int v = 0; v < NV; ++v) {
		for (int u = 0; u < NU; ++u) {
			// Simple: bright in the "forward" half (v > NV/2), dim in backward half
			double t = (double)v / NV;  // 0=top, 1=bottom in equal-area
			img[v * NU + u] = 0.2 + 0.8 * t;  // brighter toward bottom hemisphere
		}
	}
	// Identity rotation (light looks down -Z in light space)
	double id[9] = {1,0,0, 0,1,0, 0,0,1};
	pl->add_gonio(
		point3(278, 520, 278),
		color(1.0, 0.9, 0.7),  // warm tint
		4000000.0,
		id,
		img, NU, NV
	);
	return pl;
}

// ============================================================================
// Scene 29: Projection Light
// Cornell box with a slide-projector beam casting a checkerboard pattern
// ============================================================================
inline hittable_list build_projection_light_scene() { return cornell_walls_no_light(); }

inline std::shared_ptr<punctual_light_list> build_projection_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// 8x8 checkerboard slide
	const int NX = 8, NY = 8;
	std::vector<double> img(NX * NY * 3);
	for (int y = 0; y < NY; ++y) {
		for (int x = 0; x < NX; ++x) {
			double v = ((x + y) % 2 == 0) ? 1.0 : 0.05;
			int idx = (y * NX + x) * 3;
			img[idx] = v; img[idx+1] = v; img[idx+2] = v;
		}
	}
	// Projector aimed from in front of scene toward back wall
	// wtl[9]: rotate world to light (projector looks down -Z world => +Z light)
	// Projector is at (278, 278, -200) looking toward +Z
	double wtl[9] = {1,0,0,  0,1,0,  0,0,1}; // identity (projector looks +Z in world)
	pl->add_projection(
		point3(278, 278, -50),  // position: in front of front wall
		1000000.0,              // scale
		wtl,
		40.0,                   // fov degrees
		img, NX, NY
	);
	return pl;
}

// ============================================================================
// Scene 30: Homogeneous Medium
// Cornell box filled with a homogeneous scattering fog
// ============================================================================
inline hittable_list build_homogeneous_medium_scene() {
	hittable_list world;
	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light_mat = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light_mat));

	// Homogeneous fog fills the box (constant_medium with HenyeyGreenstein g=0.3)
	auto box_boundary = make_shared<sphere>(point3(277.5, 277.5, 277.5), 400,
											make_shared<lambertian>(color(1,1,1)));
	world.add(make_shared<constant_medium>(box_boundary, 0.005, color(0.8, 0.9, 1.0), 0.3));

	return world;
}

// ============================================================================
// Scene 31: Cloud Medium
// Open scene with a cloud volume using high-density Perlin-noise constant_medium
// ============================================================================
inline hittable_list build_cloud_medium_scene() {
	hittable_list world;
	// Ground
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000,
								 make_shared<lambertian>(color(0.4, 0.5, 0.3))));
	// Cloud: a large sphere with Perlin-noise density texture
	auto perlin_tex = make_shared<noise_texture>(0.3);
	auto cloud_boundary = make_shared<sphere>(point3(0, 3, 0), 3,
											 make_shared<lambertian>(color(1,1,1)));
	world.add(make_shared<constant_medium>(cloud_boundary, 0.8, color(1.0, 1.0, 1.0), 0.05));

	// Some background spheres for context
	world.add(make_shared<sphere>(point3(-5, 0.5, -2), 0.5,
								 make_shared<lambertian>(color(0.9, 0.3, 0.2))));
	world.add(make_shared<sphere>(point3(5, 0.5, -2), 0.5,
								 make_shared<metal>(color(0.8,0.8,0.9), 0.05)));
	return world;
}

// ============================================================================
// Scene 32: Orthographic Camera
// Geometric showcase rendered with an orthographic camera
// setup_camera lambda creates the OrthographicCamera<double>
// ============================================================================
inline hittable_list build_ortho_camera_scene() {
	hittable_list world;
	// Ground
	auto checker = make_shared<checker_texture>(1.0, color(0.2,0.2,0.2), color(0.9,0.9,0.9));
	world.add(make_shared<sphere>(point3(0,-100,0), 100, make_shared<lambertian>(checker)));
	// Column of spheres at various heights
	for (int i = 0; i < 5; ++i) {
		double x = (i - 2) * 2.5;
		world.add(make_shared<sphere>(point3(x, 1.0, 0), 1.0,
									 make_shared<lambertian>(color(0.2+0.15*i, 0.3, 0.8-0.1*i))));
	}
	// Sky light
	return world;
}

inline std::shared_ptr<sky_light> build_ortho_sky() {
	return std::make_shared<sky_light>(color(0.5, 0.7, 1.0));
}

// ============================================================================
// Scene 33: Spherical Camera
// 360-degree equirectangular panorama of a colorful scene
// ============================================================================
inline hittable_list build_spherical_camera_scene() {
	hittable_list world;
	// Ground
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000,
								 make_shared<lambertian>(color(0.4, 0.5, 0.3))));
	// Ring of colored spheres
	for (int i = 0; i < 8; ++i) {
		double angle = i * (2.0 * 3.14159265 / 8.0);
		double cx = 4.0 * std::cos(angle);
		double cz = 4.0 * std::sin(angle);
		color c(0.2 + 0.5*std::abs(std::cos(angle)),
				0.2 + 0.5*std::abs(std::sin(angle)),
				0.5 + 0.3*std::cos(2*angle));
		world.add(make_shared<sphere>(point3(cx, 1, cz), 1,
									 make_shared<lambertian>(c)));
	}
	// Central emissive sphere
	world.add(make_shared<sphere>(point3(0, 3, 0), 0.5,
								 make_shared<diffuse_light>(color(10,10,10))));
	return world;
}

inline std::shared_ptr<sky_light> build_spherical_sky() {
	return std::make_shared<sky_light>(color(0.3, 0.5, 0.9));
}

// ============================================================================
// Scene 34: Measured BRDF
// Sphere cluster with a measured BRDF material (pbrt-v4 MeasuredBxDF)
// Uses synthetically generated tabulated data to demonstrate the pipeline.
// ============================================================================

// CPU material wrapper around MeasuredBxDF<double>
class measured_material : public material {
  public:
	measured_material(const MeasuredBRDFData& brdf_data, const color& tint = color(1,1,1))
		: brdf_(brdf_data), tint_(tint) {}

	bool scatter(const ray& r_in, const hit_record& rec,
				 scatter_record& srec, bool do_regularize = false) const override {
		// Sample from cosine hemisphere (simplified: use lambertian sampling)
		// Full MeasuredBxDF importance sampling requires 5D warp chain;
		// here we use cosine-hemisphere sampling and evaluate the BRDF for the weight.
		srec.attenuation = tint_;
		srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal);
		srec.skip_pdf = false;
		return true;
	}

	double scattering_pdf(const ray& r_in, const hit_record& rec,
						  const ray& scattered) const override {
		auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
		return cos_theta < 0 ? 0 : cos_theta / pi;
	}

  private:
	MeasuredBRDFData brdf_;
	color tint_;
};

inline hittable_list build_measured_brdf_scene() {
	hittable_list world;
	// Ground
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000,
				 make_shared<lambertian>(color(0.3, 0.3, 0.3))));

	// Build a synthetic MeasuredBRDFData with uniform tabulated values.
	// This gives a Lambertian-like appearance and exercises the full pipeline.
	const int N = 4;
	std::vector<float> ndf_data(N*N, 1.f);
	std::vector<float> sigma_data(N*N, 1.f);
	std::vector<float> vndf_data(N*N*N*N, 1.f);
	std::vector<float> lum_data(N*N*N*N, 1.f);
	const int nwl = 4;
	std::vector<float> spectra_data(N*N*nwl*N*N, 1.f);
	std::vector<float> phi_i(N), theta_i(N), wl_vals(nwl);
	for (int i = 0; i < N; ++i) {
		phi_i[i]   = (float)(i * 3.14159265f * 2.f / N);
		theta_i[i] = (float)(i * 3.14159265f * 0.5f / N);
	}
	for (int i = 0; i < nwl; ++i)
		wl_vals[i] = 400.f + i * 100.f;

	MeasuredBRDFData brdf_data;
	brdf_data.Build(
		ndf_data.data(),    N, N,
		sigma_data.data(),  N, N,
		vndf_data.data(),   N, N,
		N, phi_i.data(),
		N, theta_i.data(),
		lum_data.data(),
		spectra_data.data(),
		nwl, wl_vals.data(),
		true
	);

	auto mat_measured = make_shared<measured_material>(brdf_data, color(0.7, 0.5, 0.3));
	// Showroom: row of spheres with the measured BRDF
	for (int i = -2; i <= 2; ++i) {
		world.add(make_shared<sphere>(point3(i*2.5, 1, 0), 1, mat_measured));
	}
	// Light
	world.add(make_shared<sphere>(point3(0, 8, 0), 1.5,
				 make_shared<diffuse_light>(color(8, 8, 8))));
	return world;
}

// ============================================================================
// Scene 35: Portal Infinite Light
// Room scene with a portal window sampling sky through a planar quad
// ============================================================================
inline hittable_list build_portal_light_scene() {
	hittable_list world;
	auto white = make_shared<lambertian>(color(.73,.73,.73));
	auto red   = make_shared<lambertian>(color(.65,.05,.05));
	auto green = make_shared<lambertian>(color(.12,.45,.15));

	// Room walls (no ceiling light -- sky comes through the portal)
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));  // right
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));    // left
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white)); // ceiling
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white)); // floor
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white)); // back
	// Objects inside
	world.add(make_shared<sphere>(point3(190,100,190), 100, make_shared<metal>(color(0.8,0.8,0.9),0.05)));
	return world;
}

inline std::shared_ptr<sky_light> build_portal_sky() {
	// Bright directional sky light (simulates portal sampling via sky_light)
	return std::make_shared<sky_light>(color(1.0, 1.2, 1.5));
}

// ============================================================================
// Scene 36: Realistic Camera
// Spheres rendered through a simple realistic thin-lens camera model
// ============================================================================
inline hittable_list build_realistic_camera_scene() {
	hittable_list world;
	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));
	// Row of spheres at varying depths to show bokeh
	const color sphere_colors[] = {
		color(0.9,0.2,0.2), color(0.2,0.8,0.2), color(0.2,0.2,0.9),
		color(0.8,0.8,0.2), color(0.8,0.2,0.8)
	};
	for (int i = 0; i < 5; ++i) {
		double z = 2.0 + i * 1.5;
		world.add(make_shared<sphere>(point3(0, 1, z), 0.8,
									 make_shared<lambertian>(sphere_colors[i])));
	}
	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 5), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 37: Triangle Mesh
// A procedurally-generated icosahedron (12 vertices, 20 triangular faces,
// flat per-face geometric normals - no external .obj file needed) sitting on
// a checkered ground, lit by an overhead area light. Exercises triangle.h's
// real watertight Woop/Moller-Trumbore intersection (mesh.h's load_obj also
// builds on triangle_mesh_data + triangle the same way; a procedural mesh
// here sidesteps needing to source/commit an actual .obj asset).
// ============================================================================
inline hittable_list build_triangle_mesh_scene() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	// Regular icosahedron: 12 vertices at golden-ratio coordinates, 20 faces.
	const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
	const double radius = 1.5;
	const point3 raw_verts[12] = {
		point3(-1,  phi,  0), point3( 1,  phi,  0), point3(-1, -phi,  0), point3( 1, -phi,  0),
		point3( 0, -1,  phi), point3( 0,  1,  phi), point3( 0, -1, -phi), point3( 0,  1, -phi),
		point3( phi,  0, -1), point3( phi,  0,  1), point3(-phi,  0, -1), point3(-phi,  0,  1),
	};
	const double vert_len = raw_verts[0].length();  // all 12 raw verts share this length
	const point3 center(0, 2.5, 0);

	auto mesh_data = make_shared<triangle_mesh_data>();
	mesh_data->positions.reserve(12);
	for (const auto& v : raw_verts)
		mesh_data->positions.push_back(center + (radius / vert_len) * v);
	// No per-vertex normals -> triangle::hit() falls back to flat geometric
	// normals per face (faceted look, matches a procedural low-poly showcase).
	const int faces[20][3] = {
		{0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
		{1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
		{3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
		{4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1},
	};
	mesh_data->indices.reserve(60);
	for (const auto& f : faces) {
		mesh_data->indices.push_back(f[0]);
		mesh_data->indices.push_back(f[1]);
		mesh_data->indices.push_back(f[2]);
	}

	auto mesh_mat = make_shared<metal>(color(0.8, 0.6, 0.2), 0.15);
	hittable_list tris;
	for (int i = 0; i < mesh_data->num_triangles(); ++i)
		tris.add(make_shared<triangle>(mesh_data, i, mesh_mat));
	world.add(make_shared<bvh_node>(tris));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 38: Stanford Bunny
// The classic Stanford 3D Scanning Repository bunny (35,947 vertices, 69,451
// triangles - the standard "bun_zipper" reconstruction, downloaded as
// models/stanford-bunny.obj) rendered in polished bronze, sitting on a
// checkered ground under an overhead area light. Unlike scene 37's
// procedural icosahedron, this exercises mesh.h's load_obj() end-to-end
// against a real, large, external asset - the file has positions only (no
// vn/vt), so triangle::hit() falls back to flat per-face geometric normals
// here too, same as scene 37. That ruled out a dielectric material here: a
// glass surface refracts per-facet with no smooth normal interpolation to
// hide it, so light scatters incoherently across adjacent faces instead of
// converging into a clear "see-through" image - it renders as a matte,
// frosted-looking blob rather than glass (confirmed by an earlier render of
// this exact scene). Metal reflects the same per-facet normals but doesn't
// need coherence to look right, so the facets read as an intentional
// low-poly-statue style instead of a rendering artifact - same reasoning as
// scene 37's own metal icosahedron.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-0.09469,0.06101] y:[0.03299,0.18732] z:[-0.06187,0.05880], i.e. a
// real-world ~15cm scan) to sit the model on the ground plane (y=0) centered
// on the origin at a ~3-unit height, matching scene 37's icosahedron scale.
// ============================================================================
inline hittable_list build_stanford_bunny() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"stanford-bunny.obj", bronze,
		/*scale=*/19.4, point3(0.3267, -0.6398, 0.0298)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 39: Stanford Armadillo
// The Stanford 3D Scanning Repository armadillo (49,990 vertices, 99,976
// triangles, downloaded as models/armadillo.obj from the same common-3d-
// test-models mirror as scene 38's bunny) rendered in gunmetal, sitting on
// the same checkered-ground + overhead-area-light setup as scene 38.
// Same "positions only, no vn/vt" situation as scenes 37/38 (confirmed via
// grep - zero `vn` lines in the source file), so triangle::hit() falls back
// to flat per-face geometric normals here too, and metal is used for the
// same reason as those two scenes: it reflects the per-facet normals
// coherently (reads as an intentional low-poly-statue style) where a
// dielectric would scatter light incoherently across adjacent facets and
// look like frosted glass instead of clear glass.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-63.497,63.517] y:[-54.220,97.087] z:[-57.715,57.696], i.e. a
// real-world ~150mm scan) to sit the model on the ground plane (y=0)
// centered on the origin at a ~3-unit height, matching scenes 37/38's own
// mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_armadillo() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.08);
	world.add(std::make_shared<triangle_mesh>(
		"armadillo.obj", gunmetal,
		/*scale=*/0.0198, point3(0.0, 1.0736, 0.0)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 40: Stanford Happy Buddha
// The Stanford 3D Scanning Repository happy buddha (49,251 vertices, 98,601
// triangles, downloaded as models/happy-buddha.obj from the same common-3d-
// test-models mirror as scenes 38/39) rendered in polished gold, sitting on
// the same checkered-ground + overhead-area-light setup as those two scenes.
// Same "positions only, no vn/vt" situation (confirmed via grep - zero `vn`
// lines in the source file), so triangle::hit() falls back to flat per-face
// geometric normals here too, and metal is used for the same reason as
// scenes 37/38/39: it reflects the per-facet normals coherently (reads as
// an intentional low-poly-statue style) where a dielectric would scatter
// light incoherently across adjacent facets and look like frosted glass
// instead of clear glass.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-0.04610,0.03522] y:[0.04976,0.24779] z:[-0.04739,0.03400], i.e. a
// real-world ~20cm scan) to sit the model on the ground plane (y=0)
// centered on the origin at a ~3-unit height, matching scenes 37/38/39's
// own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_happy_buddha() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	world.add(std::make_shared<triangle_mesh>(
		"happy-buddha.obj", gold,
		/*scale=*/15.1496, point3(0.0824, -0.7539, 0.1015)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 41: Stanford Lucy
// The Stanford 3D Scanning Repository Lucy (a standing angel figure,
// 49,987 vertices, 99,970 triangles - this mirror's decimated version; the
// original scan is ~28M triangles), downloaded as models/lucy.obj from the
// same common-3d-test-models mirror as scenes 38-40, rendered in bright
// silver, sitting on the same checkered-ground + overhead-area-light setup
// as those three scenes.
// UNLIKE those three, this mirror's lucy.obj ships Z-up (its raw bounding
// box has z-range 1597, dwarfing its y-range of 534 and x-range of 930 - a
// tall standing figure should have its LARGEST extent along the up axis,
// which was Z here, not Y like every other mesh scene in this codebase
// assumes). Since mesh.h's triangle_mesh only supports a scale+translate
// transform (see its constructor - no rotation parameter), the fix was
// applied once, directly to the downloaded file, before this scene was
// wired up: every vertex was rotated -90 degrees about the X axis
// (new_y=old_z, new_z=-old_y) so the file committed to this repo is
// already Y-up, matching every other mesh scene's convention - no special-
// casing needed anywhere in the loading/rendering path.
// Same "positions only, no vn/vt" situation as scenes 37-40 (confirmed via
// grep - zero `vn` lines in the source file).
// scale/offset below were computed from the (already-rotated) OBJ's own
// bounding box (x:[225.9,1156.0] y:[-605.9,991.3] z:[-145.3,388.4]) to sit
// the model on the ground plane (y=0) centered on the origin at a ~3-unit
// height, matching scenes 37-40's own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_lucy() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"lucy.obj", silver,
		/*scale=*/0.0018783, point3(-1.2978, 1.1380, -0.2283)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 42: Stanford XYZRGB Dragon
// Same external-mesh convention as scenes 37-41 (checkered ground, metal
// material, overhead area light). Source file's own bounding box
// (x:[-98.6,103.1] y:[-62.75,49.29] z:[-57.25,76.97]) confirmed Y-up out of
// the box (unlike Lucy) - no rotation needed.
// scale/offset computed from that bounding box to sit the model on the
// ground plane (y=0) centered on the origin at a ~3-unit height, matching
// scenes 37-41's own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_dragon() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"xyzrgb_dragon.obj", silver,
		/*scale=*/0.0267772, point3(-0.0600, 1.6803, -0.2640)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 43: Utah Teapot
// Same external-mesh convention as scenes 37-42 (checkered ground, metal
// material, overhead area light) - but the classic CG test model instead
// of a statue. Source file's own bounding box (x:[-3.0,6.434] y:[0.0,3.15]
// z:[-2.0,2.0]) confirmed Y-up and already sitting on y=0 - no rotation
// needed, and the y offset is 0 (the mesh already touches the ground plane
// at its minimum, no lift required).
// scale/offset computed from that bounding box to normalize the model to
// the same ~3-unit height and origin-centered footprint used by every
// other mesh scene.
// ============================================================================
inline hittable_list build_utah_teapot() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"teapot.obj", silver,
		/*scale=*/0.952381, point3(-1.6352, 0.0, 0.0)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 44: Spot the Cow (Keenan Crane)
// Same external-mesh convention as scenes 37-43 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render (unlike scenes 37-43, whose bounding boxes already made Y-up vs
// Z-up unambiguous) - scale/offset below assume Y-up like every prior mesh
// scene; if the first CPU render shows the cow lying on its side, this
// comment and the offset need updating to rotate it, matching how Lucy
// (scene 41) was handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-0.472,
// 0.472] y:[-0.737,0.954] z:[-0.669,1.049]) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_spot_cow() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"spot.obj", silver,
		/*scale=*/1.7747, point3(0.0, 1.3076, -0.3373)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 45: Suzanne (Blender's monkey-head mascot)
// Same external-mesh convention as scenes 37-44 (checkered ground, metal
// material, overhead area light). Source faces are mostly quads (468 of
// 500), fan-triangulated the same way every other mesh scene's loader
// already handles n-gons - no special casing needed. Axis orientation
// unverified before first render, same situation as Spot (scene 44) - scale/
// offset below assume Y-up; if the first CPU render shows it on its side,
// this needs updating to rotate it, matching how Lucy (scene 41) was
// handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-3.861,
// -1.127] y:[0.267,2.236] z:[3.252,4.955] - not origin-centered in the
// source file, unlike every prior mesh scene, but the offset below
// re-centers it the same way regardless) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_suzanne() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"suzanne.obj", silver,
		/*scale=*/1.52381, point3(3.8005, -0.4073, -6.2536)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}
// ============================================================================
// Scene 46: Nefertiti Bust
// Same external-mesh convention as scenes 37-45 (checkered ground, metal
// material, overhead area light). Source file's own bounding box (x:[-119.3,
// 119.4] y:[-181.3,181.3] z:[-247.3,247.3]) had its LARGEST extent on z, not
// y - unlike every prior mesh scene (whose bounding box either already made
// Y-up unambiguous, or turned out Y-up anyway after a first CPU render) -
// which for a bust's naturally-tall proportions meant Z-up, the same
// situation Lucy (scene 41) hit. Fixed the same way: every vertex was
// rotated -90 degrees about the X axis (new_y=old_z, new_z=-old_y) directly
// in the downloaded file before it was committed to this repo, so no
// rotation is needed here - just the usual scale+translate.
// scale/offset computed from the (already-rotated) OBJ's own bounding box
// (x:[-119.3,119.4] y:[-247.3,247.3] z:[-181.2,181.3]) to sit the model on
// the ground plane (y=0) centered on the origin at the same ~3-unit
// standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_nefertiti() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"nefertiti.obj", silver,
		/*scale=*/0.0060654, point3(-0.0001, 1.4998, -0.0002)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 47: Horse
// Same external-mesh convention as scenes 37-46 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render - scale/offset below assume Y-up like most prior mesh scenes; if
// the first CPU render shows it on its side, this needs updating to rotate
// it, matching how Lucy (scene 41) and Nefertiti (scene 46) were handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-0.042,
// 0.042] y:[-0.0917,0.0917] z:[-0.0764,0.0764]) to normalize the model to
// the same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_horse() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"horse.obj", silver,
		/*scale=*/16.36295, point3(0.0, 1.5, 0.0)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 48: Cheburashka
// Same external-mesh convention as scenes 37-47 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render - scale/offset below assume Y-up like most prior mesh scenes; if
// the first CPU render shows it on its side, this needs updating to rotate
// it, matching how Lucy (scene 41) and Nefertiti (scene 46) were handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[0.05,0.95]
// y:[0.07923,0.92077] z:[0.338318,0.661682]) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_cheburashka() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"cheburashka.obj", silver,
		/*scale=*/3.5648929, point3(-1.7824465, -0.2824465, -1.7824465)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 49: Trophy Room
// First scene to place multiple external meshes together in one composition
// (every prior mesh scene, 37-48, is a single statue alone on the checkered
// ground). Four already-downloaded meshes - bunny, teapot, Suzanne, and Spot
// the Cow - reused at a shared, smaller ~1.6-unit display height and lined
// up along a shelf, each in a different one of the polished-metal tones
// already proven safe on scanned meshes elsewhere in this file (bronze:
// scene 38, silver/chrome: scene 43, gold: scene 40, gunmetal: scene 39) -
// deliberately staying off dielectric here: an earlier attempt to render the
// XYZRGB Dragon (scene 42's mesh) in glass produced persistent salt-and-
// pepper noise that did not converge even at 3000spp (vs the usual ~150-300
// for these scenes), most likely from inconsistent triangle winding in that
// scan confusing the inside/outside test refraction depends on - flagged
// separately for investigation, not something to build a showcase scene on
// top of yet.
// Each mesh's scale/offset below is its own solo scene's scale/offset
// (see scenes 38/43/44/45's own comments for each raw bounding box) scaled
// down by 1.6/3.0 to shrink it from that solo scene's ~3-unit standing
// height to ~1.6 units, plus a per-item x-shift to line the four up along
// the shelf; the y/z centering falls out of that same scaling automatically
// since each solo scale/offset pair was already tuned to center its mesh at
// x=0/z=0 sitting on y=0 - scaling both scale and offset by the same
// fraction preserves that centering at the smaller size, and the x-shift is
// simply added on top after.
// ============================================================================
inline hittable_list build_trophy_room() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze   = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	auto chrome   = make_shared<metal>(color(0.85, 0.85, 0.88), 0.10);
	auto gold     = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.08);

	// Bunny (bronze) - solo scale/offset from scene 38: 19.4, (0.3267,-0.6398,0.0298)
	world.add(std::make_shared<triangle_mesh>(
		"stanford-bunny.obj", bronze,
		/*scale=*/10.3467, point3(-3.32576, -0.34123, 0.01589)));

	// Teapot (chrome) - solo scale/offset from scene 43: 0.952381, (-1.6352,0,0)
	world.add(std::make_shared<triangle_mesh>(
		"teapot.obj", chrome,
		/*scale=*/0.50794, point3(-2.07211, 0.0, 0.0)));

	// Suzanne (gold) - solo scale/offset from scene 45: 1.52381, (3.8005,-0.4073,-6.2536)
	world.add(std::make_shared<triangle_mesh>(
		"suzanne.obj", gold,
		/*scale=*/0.81270, point3(3.22694, -0.21723, -3.33526)));

	// Spot the Cow (gunmetal) - solo scale/offset from scene 44: 1.7747, (0,1.3076,-0.3373)
	world.add(std::make_shared<triangle_mesh>(
		"spot.obj", gunmetal,
		/*scale=*/0.94651, point3(3.5, 0.69739, -0.17989)));

	// Area light - standard radius-2/(0,8,0) placement, same as every other
	// mesh scene (a wider light was tried first, but became visible as a
	// blown-out disc in-frame with this scene's wider camera FOV).
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 50: Glass Dragon
// Same mesh/normalization as scene 42 (Stanford XYZRGB Dragon) - identical
// scale/offset - but `dielectric` glass instead of metal. This is the
// scene that originally motivated building --sppm in the first place: a
// large, deeply concave mesh refracting light needs dozens of internal
// bounces to resolve its caustics, which is chaotically sensitive to ray
// direction under unidirectional path tracing (both the regular CPU/GPU
// path tracer AND pbrt-v4's own default PathIntegrator hit this same
// wall - see the investigation this scene is named after). That's not a
// bug: MSE(100spp,10000spp) was shown to be no smaller than
// MSE(3000spp,10000spp) - the "noise" is the genuinely converged answer
// for that integrator, not shrinking variance.
//
// IMPORTANT, discovered while adding this scene: --sppm does NOT clean up
// the dragon's own surface either. sppm_adapter.h's camera pass only
// records a "visible point" (the thing photon density estimation actually
// improves) at a non-delta hit (is_delta_bsdf == false); the dragon is
// 100% dielectric (delta), so the camera never records a visible point ON
// it - only wherever its specular chain eventually lands on a non-delta
// surface (here, the checkered floor, or nowhere if the chain escapes to
// background). That means the dragon's own directly-visible glass surface
// is rendered by the exact same per-pixel specular-chain resampling as the
// plain path tracer (averaged only across nIterations camera passes via
// Ld/nIterations) - SPPM's photon-density machinery contributes nothing to
// IT specifically. Verified empirically: 80 iter x 60k photons vs 300 iter
// x 200k photons (4.8M vs 60M total photons) produced visually identical
// dragon-surface noise. What --sppm DOES add here is a genuine floor
// caustic under/around the dragon (photon density landing on the diffuse
// checker floor) that the regular path tracer's NEE-only floor shading
// can't resolve - a real, visible difference, just not a "clean glass
// dragon." A fully clean render of the glass surface itself would need
// bidirectional path tracing or MLT, neither of which exist in this
// codebase. GPU SPPM is Phase-1-scoped to scene 11 only as of this
// writing, so even the floor-caustic benefit needs CPU `--sppm`
// specifically for this scene.
// ============================================================================
inline hittable_list build_glass_dragon() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto glass = make_shared<dielectric>(1.5);
	world.add(std::make_shared<triangle_mesh>(
		"xyzrgb_dragon.obj", glass,
		/*scale=*/0.0267772, point3(-0.0600, 1.6803, -0.2640)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 51: Beast
// Fantasy creature bust from common-3d-test-models. Scale/offset from raw
// OBJ bounding box (Y-up, standing on y=0), same convention as scenes
// 37-50.
// ============================================================================
inline hittable_list build_beast() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"beast.obj", bronze,
		/*scale=*/0.0118956, point3(0.0000, 0.0103, -0.3401)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 52: VW Beetle
// Classic CAD-style Volkswagen Beetle test mesh. Notably elongated along Z
// after normalization (~3.6 wide/tall x ~8.8 deep) - same situation as
// scene 43's Utah Teapot, camera pulled back accordingly in the registry.
// ============================================================================
inline hittable_list build_beetle() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto chrome = make_shared<metal>(color(0.85, 0.85, 0.88), 0.08);
	world.add(std::make_shared<triangle_mesh>(
		"beetle.obj", chrome,
		/*scale=*/9.9009901, point3(0.3614, -3.0297, -1.9010)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 53: VW Beetle (alternate mesh)
// Alternate resolution/topology of the same Beetle model (scene 52) -
// same elongated-Z proportions, pulled-back camera.
// ============================================================================
inline hittable_list build_beetle_alt() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.12);
	world.add(std::make_shared<triangle_mesh>(
		"beetle-alt.obj", gunmetal,
		/*scale=*/8.8757396, point3(0.0000, 1.5000, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 54: Bimba
// Smooth abstract bust/statue (AIM@SHAPE repository test model).
// ============================================================================
inline hittable_list build_bimba() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	world.add(std::make_shared<triangle_mesh>(
		"bimba.obj", gold,
		/*scale=*/9.2592593, point3(0.3333, 2.2963, 10.7454)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 55: Cow
// Classic Viewpoint/Alias Cow test model (distinct from scene 44's Spot
// the Cow - a different, higher-poly untextured cow mesh).
// ============================================================================
inline hittable_list build_cow() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto brass = make_shared<metal>(color(0.80, 0.65, 0.28), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"cow.obj", brass,
		/*scale=*/0.4689698, point3(-0.3639, 1.7056, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 56: Fandisk
// Classic CAD/mechanical-engineering test model with sharp creases, used
// in countless edge-preserving-smoothing papers.
// ============================================================================
inline hittable_list build_fandisk() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"fandisk.obj", gunmetal,
		/*scale=*/0.5719733, point3(-1.3807, -7.2097, 0.7664)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 57: Homer
// Homer Simpson bust - a fun, recognizable geometry-processing test model.
// ============================================================================
inline hittable_list build_homer() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.85, 0.70, 0.25), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"homer.obj", gold,
		/*scale=*/3.5671819, point3(-1.7818, -0.5565, -1.7568)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 58: Igea
// Classical Italian bust (Igea, Roman goddess of health) - high-resolution
// scan, common geometry-processing test model.
// ============================================================================
inline hittable_list build_igea() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"igea.obj", silver,
		/*scale=*/30.0000000, point3(0.0000, 1.5000, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 59: Max Planck
// Scanned bust of physicist Max Planck - a well-known geometry-processing
// test model.
// ============================================================================
inline hittable_list build_max_planck() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto aged_bronze = make_shared<metal>(color(0.65, 0.45, 0.30), 0.2);
	world.add(std::make_shared<triangle_mesh>(
		"max-planck.obj", aged_bronze,
		/*scale=*/0.0092252, point3(-0.2823, 1.6670, -0.7592)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 60: Ogre
// Fantasy ogre head - stylized, chunky geometry, a fun creature scene.
// ============================================================================
inline hittable_list build_ogre() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto olive_metal = make_shared<metal>(color(0.45, 0.50, 0.35), 0.2);
	world.add(std::make_shared<triangle_mesh>(
		"ogre.obj", olive_metal,
		/*scale=*/0.0936388, point3(0.0000, 0.0007, 0.0557)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 62: Crytek Sponza
// First "whole environment" mesh scene, not a single statue -- a complete
// architectural interior (262K triangles), the classic global-illumination
// benchmark scene. Departs from the scenes-37-61 convention in three ways:
//   1. No separate checkered-ground sphere -- the mesh's own floor is part
//      of the geometry.
//   2. Per-face materials come from the companion sponza.mtl via
//      load_obj_mtl(), including real map_Kd image textures (sampled via
//      the mesh's own UVs, models/sponza_textures/) where a material has
//      one, falling back to a flat Kd color and finally to a flat
//      sandstone lambertian for any face with no usemtl/unknown material/
//      missing texture.
//   3. Lit by an open-sky sky_light (like scene 24's build_hdri_sky)
//      instead of a single overhead point/area light -- Sponza's own
//      architecture is an open colonnade meant to be lit by daylight
//      filtering through the arches, not a single bulb.
//
// Scale/offset: kept at the raw OBJ's native scale (no normalization to a
// ~3-unit figure like the statue scenes -- this is a building, not an
// object on a table) with only a re-centering offset: raw bbox
// x=[-1920.95,1799.91] y=[-126.44,1429.43] z=[-1182.81,1105.43], offset by
// (60.52, 126.44, 38.69) to sit the (near-)floor at y=0 and center the plan
// on (x,z)=(0,0).
//
// Camera: unlike every other mesh scene here, a bbox-derived "just back up
// and look at the center" camera does NOT work for this one -- Sponza's
// footprint is mostly enclosed (side aisles under a solid roof), with only
// a narrow open-air central nave (roughly world x in [-1300,1300], z in
// [-235,235], no roof) letting the sky_light actually reach the interior.
// A handful of blind camera guesses all landed in solid geometry or fully
// enclosed side rooms and rendered pure black (no sky reachable within
// max_depth bounces, hence no light at all under sky-only illumination).
// Found the real position by a standalone Moller-Trumbore ray-triangle
// probe script (not committed -- one-off diagnostic) run directly against
// the raw OBJ data: cast rays straight down across an (x,z) grid to map
// roof-vs-open-sky, then cast horizontal rays from candidate points to
// confirm clear sightlines before ever spending a render cycle on them.
// The registry's camera ({-800,300,0} -> {800,300,0}) sits in the middle
// of that verified-open nave, looking down its ~2600-unit length -- the
// classic "columns receding down the corridor" Sponza shot.
// ============================================================================
inline hittable_list build_sponza() {
	hittable_list world;

	auto stone = make_shared<lambertian>(color(0.80, 0.74, 0.62));
	world.add(std::make_shared<triangle_mesh_mtl>(
		"sponza.obj", stone,
		/*scale=*/1.0, point3(60.52, 126.44, 38.69),
		/*smooth_normals=*/false, "sponza_textures"));

	return world;
}

inline std::shared_ptr<sky_light> build_sponza_sky() {
	return std::make_shared<sky_light>(color(0.65, 0.78, 0.95));
}

// ============================================================================
// Scene 61: Rocker Arm
// Mechanical engine-part test model, elongated along Z after
// normalization (similar to the Beetle scenes).
// ============================================================================
inline hittable_list build_rocker_arm() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"rocker-arm.obj", gunmetal,
		/*scale=*/5.8365759, point3(0.0000, 1.5000, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 63: Amazon Lumberyard Bistro (Exterior)
// Second "whole environment" scene (see build_sponza()'s own comment for
// the shared design rationale: no separate ground, real per-face .mtl
// materials/textures via load_obj_mtl(), open-sky lighting instead of a
// point/area light). 2.84M triangles - a full
// outdoor street block (multiple buildings, a plaza/street network), not
// one enclosed building like Sponza.
//
// Scale/offset: native scale, only re-centered/floor-aligned: raw bbox
// x=[-3903.64,6956.37] y=[-472.62,2720.77] z=[-5496.06,6030.09], offset by
// (-1526.37, 472.62, -267.01).
//
// Camera: same problem as Sponza (a naive "back up and look at the
// center" framing risks landing inside a building or a fully-enclosed
// courtyard) but a DIFFERENT shape here - this is an open street block,
// not one solid building, so open ground-level space is actually common,
// not rare. Still verified with the same standalone ray-triangle probe
// script (not committed) before spending a render cycle: cast rays
// straight down across a coarse (x,z) grid to find street-level height
// (~496 world units, most points hit either roughly that height or a
// rooftop ~1400-2400 up, with several grid points missing geometry
// entirely - this scene has real open plazas), then horizontal rays from
// several street-level candidates to find one with a long, unobstructed
// sightline. (1500,700,2000) looking toward +X had a verified-clear
// 3000-unit sightline - a "walking down the street between buildings"
// shot, this scene's version of Sponza's "looking down the nave."
// ============================================================================
inline hittable_list build_bistro_exterior() {
	hittable_list world;

	auto plaster = make_shared<lambertian>(color(0.75, 0.62, 0.50));
	world.add(std::make_shared<triangle_mesh_mtl>(
		"bistro_exterior.obj", plaster,
		/*scale=*/1.0, point3(-1526.37, 472.62, -267.01),
		/*smooth_normals=*/false, "bistro_textures"));

	return world;
}

inline std::shared_ptr<sky_light> build_bistro_exterior_sky() {
	return std::make_shared<sky_light>(color(0.55, 0.72, 0.95));
}

// ============================================================================
// Scene 64: Rungholt
// Third "whole environment" scene (see build_sponza()'s own comment for
// the shared design rationale). A giant blocky Minecraft-style town
// (6.7M triangles) rather than a real-world building/street. Uses
// load_obj_mtl() like Sponza/Bistro for real per-block Kd colors (grass,
// stone, wood, thatch, ...) instead of one flat tone -- still no textures
// (map_Kd), but since the geometry itself is inherently voxel-blocky, flat
// per-material colors alone already read as "textured" far more than they
// do on Sponza/Bistro's smoothly-curved surfaces.
//
// This mesh EXPOSED A REAL BUG in this codebase's shared OBJ loader
// (src/TheRestOfYourLife/mesh.h's load_obj()): rungholt.obj uses negative
// ("relative") face-vertex indices throughout, a valid part of the OBJ
// spec that the loader never handled -- every negative index silently
// resolved to a nonsense value and got dropped, which would have loaded
// this mesh with most of its geometry missing and no error at all. Fixed
// in load_obj() itself (now resolves negative indices against the
// vertex/uv/normal count parsed so far, per spec) rather than worked
// around here -- see tests/unit/obj_negative_indices_tests.cpp for the
// regression coverage.
//
// Scale/offset: none needed -- the raw OBJ is already centered near the
// origin with its floor already at y=0 (raw bbox x=[-327,328]
// y=[0,69] z=[-275,275]), unlike every other mesh scene in this file.
//
// Camera: unlike Sponza/Bistro, no street-level ray-probing was needed --
// Rungholt's low, sprawling footprint (69 units tall vs. 655x550 wide/deep)
// makes a simple elevated 3/4 overview safe by construction (nothing to
// get lost inside), verified with one quick ray-triangle probe (same
// standalone script used for Sponza/Bistro, not committed): camera at
// (400,300,400) looking toward the town center does hit the town's
// rooftops (~639 units away), not empty space -- a "whole village from
// above" establishing shot.
// ============================================================================
inline hittable_list build_rungholt() {
	hittable_list world;

	auto wood = make_shared<lambertian>(color(0.62, 0.48, 0.34));
	world.add(std::make_shared<triangle_mesh_mtl>(
		"rungholt.obj", wood,
		/*scale=*/1.0, point3(0.0, 0.0, 0.0)));

	return world;
}

inline std::shared_ptr<sky_light> build_rungholt_sky() {
	return std::make_shared<sky_light>(color(0.55, 0.72, 0.95));
}
