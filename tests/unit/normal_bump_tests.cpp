// normal_bump_tests.cpp
// pbrt-v4-style unit tests for normal/bump mapping math and CPU material wrappers.

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

#include "rtweekend.h"
#include "normal_map.h"
#include "material.h"
#include "normal_map_materials.h"

// ---------------------------------------------------------------------------
// apply_normal_map tests
// ---------------------------------------------------------------------------

TEST(NormalMap, FlatMapLeavesNormalUnchanged) {
	// tangent-space (0,0,1) -> should map to geometric normal
	double out_nx, out_ny, out_nz;
	apply_normal_map(0.0, 0.0, 1.0,
					 0.0, 1.0, 0.0,   // geometric normal = world Y
					 1.0, 0.0, 0.0,   // dpdu = world X
					 out_nx, out_ny, out_nz);
	EXPECT_NEAR(out_nx, 0.0, 1e-9);
	EXPECT_NEAR(out_ny, 1.0, 1e-9);
	EXPECT_NEAR(out_nz, 0.0, 1e-9);
}

TEST(NormalMap, PerturbedNormalOnSameHemisphere) {
	// tilt toward +X in tangent space
	double ts_x = 0.7, ts_y = 0.0, ts_z = 0.7;
	double len = std::sqrt(ts_x*ts_x + ts_z*ts_z);
	ts_x /= len; ts_z /= len;

	double out_nx, out_ny, out_nz;
	apply_normal_map(ts_x, ts_y, ts_z,
					 0.0, 0.0, 1.0,   // geometric normal = world Z
					 1.0, 0.0, 0.0,   // dpdu = world X
					 out_nx, out_ny, out_nz);

	// dot with geometric normal must be positive
	double dot = out_nz;
	EXPECT_GT(dot, 0.0);

	double length = std::sqrt(out_nx*out_nx + out_ny*out_ny + out_nz*out_nz);
	EXPECT_NEAR(length, 1.0, 1e-9);
}

TEST(NormalMap, OutputIsUnitVector) {
	struct Case { double nx,ny,nz, dx,dy,dz, tx,ty,tz; };
	Case cases[] = {
		{0,1,0,  1,0,0,  0,0,1},
		{0,0,1,  1,0,0,  0,0,1},
		{1,0,0,  0,1,0,  0,0,1},
	};
	for (auto& c : cases) {
		double tl = std::sqrt(c.tx*c.tx+c.ty*c.ty+c.tz*c.tz);
		double out_nx, out_ny, out_nz;
		apply_normal_map(c.tx/tl, c.ty/tl, c.tz/tl,
						 c.nx, c.ny, c.nz,
						 c.dx, c.dy, c.dz,
						 out_nx, out_ny, out_nz);
		double len = std::sqrt(out_nx*out_nx + out_ny*out_ny + out_nz*out_nz);
		EXPECT_NEAR(len, 1.0, 1e-9);
	}
}

// ---------------------------------------------------------------------------
// apply_bump_map tests
// ---------------------------------------------------------------------------

TEST(BumpMap, ZeroDisplacementLeavesNormalUnchanged) {
	double out_nx, out_ny, out_nz;
	apply_bump_map(0.0, 0.0, 0.0,
				   0.001, 0.001,
				   0.0, 1.0, 0.0,   // normal = world Y
				   1.0, 0.0, 0.0,   // dpdu
				   0.0, 0.0, 1.0,   // dpdv
				   out_nx, out_ny, out_nz);
	EXPECT_NEAR(out_nx, 0.0, 1e-9);
	EXPECT_NEAR(out_ny, 1.0, 1e-9);
	EXPECT_NEAR(out_nz, 0.0, 1e-9);
}

TEST(BumpMap, BumpedNormalOnSameHemisphere) {
	double out_nx, out_ny, out_nz;
	apply_bump_map(0.0, 0.1, 0.0,   // gradient along U
				   0.001, 0.001,
				   0.0, 0.0, 1.0,   // normal = world Z
				   1.0, 0.0, 0.0,   // dpdu
				   0.0, 1.0, 0.0,   // dpdv
				   out_nx, out_ny, out_nz);

	double dot = out_nz;  // dot with geometric normal (0,0,1)
	EXPECT_GT(dot, 0.0);
	double len = std::sqrt(out_nx*out_nx + out_ny*out_ny + out_nz*out_nz);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Helper: build a minimal hit_record
// ---------------------------------------------------------------------------
static hit_record make_hit(vec3 normal, vec3 dpdu,
							double u = 0.5, double v = 0.5) {
	hit_record rec;
	rec.p          = point3(0,0,0);
	rec.normal     = normal;
	rec.dpdu       = dpdu;
	rec.u          = u;
	rec.v          = v;
	rec.t          = 1.0;
	rec.front_face = true;
	return rec;
}

// ---------------------------------------------------------------------------
// normal_map_material wrapper tests
// ---------------------------------------------------------------------------

TEST(NormalMapMaterial, FlatMapNormalUnchanged) {
	// (0.5, 0.5, 1.0) decodes to tangent-space (0,0,1) -> no perturbation
	auto flat_tex = std::make_shared<solid_color>(color(0.5, 0.5, 1.0));
	auto inner    = std::make_shared<lambertian>(color(0.8, 0.6, 0.2));
	normal_map_material nm(flat_tex, inner);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	hit_record mod = nm.apply(rec);

	EXPECT_NEAR(mod.normal.x(), 0.0, 1e-6);
	EXPECT_NEAR(mod.normal.y(), 1.0, 1e-6);
	EXPECT_NEAR(mod.normal.z(), 0.0, 1e-6);
}

TEST(NormalMapMaterial, PerturbedNormalOnSameHemisphere) {
	// tilt ~45 deg toward +X: tangent-space (1,0,1)/sqrt2
	double v = static_cast<float>((std::sqrt(2.0)/2.0 + 1.0) / 2.0);
	auto tex   = std::make_shared<solid_color>(color(v, 0.5f, v));
	auto inner = std::make_shared<lambertian>(color(1,1,1));
	normal_map_material nm(tex, inner);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	hit_record mod = nm.apply(rec);

	// perturbed normal must still face same hemisphere as geometric normal
	EXPECT_GT(dot(mod.normal, rec.normal), 0.0);
}

TEST(NormalMapMaterial, ScatterSucceeds) {
	auto flat_tex = std::make_shared<solid_color>(color(0.5, 0.5, 1.0));
	auto inner    = std::make_shared<lambertian>(color(0.8, 0.8, 0.8));
	normal_map_material nm(flat_tex, inner);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));
	scatter_record srec;
	EXPECT_TRUE(nm.scatter(r_in, rec, srec));
}

TEST(NormalMapMaterial, DpduPerpendicularToNewNormal) {
	// pbrt-v4 alignment: after apply(), dpdu must be perpendicular to the new normal
	auto flat_tex = std::make_shared<solid_color>(color(0.5, 0.5, 1.0));
	auto inner    = std::make_shared<lambertian>(color(1,1,1));
	normal_map_material nm(flat_tex, inner);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	hit_record mod = nm.apply(rec);

	double d = dot(mod.dpdu, mod.normal);
	EXPECT_NEAR(d, 0.0, 1e-6);  // dpdu ⊥ shading normal
}



TEST(BumpMapMaterial, ZeroDisplacementNormalUnchanged) {
	auto zero = std::make_shared<solid_color>(color(0,0,0));
	auto inner = std::make_shared<lambertian>(color(0.5,0.5,0.5));
	bump_map_material bm(zero, inner, 1.0, 0.001);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	hit_record mod = bm.apply(rec);

	EXPECT_NEAR(mod.normal.x(), 0.0, 1e-6);
	EXPECT_NEAR(mod.normal.y(), 1.0, 1e-6);
	EXPECT_NEAR(mod.normal.z(), 0.0, 1e-6);
}

TEST(BumpMapMaterial, BumpedNormalOnSameHemisphere) {
	auto grey  = std::make_shared<solid_color>(color(0.5, 0.5, 0.5));
	auto inner = std::make_shared<lambertian>(color(0.5,0.5,0.5));
	bump_map_material bm(grey, inner, 5.0, 0.001);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	hit_record mod = bm.apply(rec);

	EXPECT_GT(dot(mod.normal, rec.normal), 0.0);
}

TEST(BumpMapMaterial, ScatterSucceeds) {
	auto tex   = std::make_shared<solid_color>(color(0.3,0.3,0.3));
	auto inner = std::make_shared<lambertian>(color(0.8,0.8,0.8));
	bump_map_material bm(tex, inner, 1.0, 0.001);

	hit_record rec = make_hit(vec3(0,1,0), vec3(1,0,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));
	scatter_record srec;
	EXPECT_TRUE(bm.scatter(r_in, rec, srec));
}
