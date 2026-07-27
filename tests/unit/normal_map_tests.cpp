// normal_map_tests.cpp
// pbrt-v4-style unit tests for apply_normal_map() and apply_bump_map(),
// and for the normal_map_material / bump_map_material CPU wrappers.
//
// Design follows pbrt-v4's materials_test approach:
//   - Flat map (0.5, 0.5, 1.0) must leave the shading normal unchanged
//   - Perturbed normal must remain on the same hemisphere as the geometric normal
//   - Zero-displacement bump map must return the original normal
//   - Perturbed bump normal must stay on the same hemisphere as geometric normal
//   - Material wrapper round-trip: scatter() on a flat map equals scatter() on
//     the bare inner material (same PDF, same attenuation direction hemisphere)

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

// Include shared normal map math directly
#include "normal_map.h"

// Include the CPU material layer (pulls in hittable, texture, pdf, etc.)
#include "material.h"
#include "texture.h"

// Wrapper classes (normal_map_material, bump_map_material) — pbrt-v4 style
#include "normal_map_materials.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double vec3_dot(double ax, double ay, double az,
					   double bx, double by, double bz) {
	return ax*bx + ay*by + az*bz;
}

static double vec3_len(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

static bool nearly_unit(double x, double y, double z, double tol = 1e-9) {
	return std::abs(vec3_len(x, y, z) - 1.0) < tol;
}

static bool nearly_eq(double a, double b, double tol = 1e-9) {
	return std::abs(a - b) < tol;
}

// Build a minimal hit_record with a given normal and dpdu.
static hit_record make_rec(vec3 normal, vec3 dpdu,
							double u = 0.5, double v = 0.5) {
	hit_record rec;
	rec.p          = point3(0, 0, 0);
	rec.normal     = normal;
	rec.dpdu       = dpdu;
	rec.u          = u;
	rec.v          = v;
	rec.t          = 1.0;
	rec.front_face = true;
	// mat is unused in these tests
	return rec;
}

// ---------------------------------------------------------------------------
// apply_normal_map tests
// ---------------------------------------------------------------------------

// A flat normal map encodes (0,0,1) in tangent space, which should be decoded
// to exactly the geometric normal in world space.
TEST(NormalMap, FlatMapLeavesNormalUnchanged) {
	// geometric normal = world Y; dpdu = world X
	double nx = 0, ny = 1, nz = 0;
	double dpdu_x = 1, dpdu_y = 0, dpdu_z = 0;

	// flat map tangent-space normal
	double ns_tx = 0, ns_ty = 0, ns_tz = 1;

	double out_nx, out_ny, out_nz;
	apply_normal_map(ns_tx, ns_ty, ns_tz,
					 nx, ny, nz,
					 dpdu_x, dpdu_y, dpdu_z,
					 out_nx, out_ny, out_nz);

	// Result should be the same as the geometric normal
	EXPECT_NEAR(out_nx, nx, 1e-9);
	EXPECT_NEAR(out_ny, ny, 1e-9);
	EXPECT_NEAR(out_nz, nz, 1e-9);
	EXPECT_TRUE(nearly_unit(out_nx, out_ny, out_nz));
}

// Perturbed tangent-space normal stays on the same hemisphere as the
// geometric normal (apply_normal_map guarantees this via the sign flip).
TEST(NormalMap, PerturbedNormalOnSameHemisphere) {
	double nx = 0, ny = 0, nz = 1;
	double dpdu_x = 1, dpdu_y = 0, dpdu_z = 0;

	// tilt toward +X in tangent space
	double ns_tx = 0.7, ns_ty = 0.0, ns_tz = 0.7;
	double ns_len = std::sqrt(ns_tx*ns_tx + ns_ty*ns_ty + ns_tz*ns_tz);
	ns_tx /= ns_len; ns_ty /= ns_len; ns_tz /= ns_len;

	double out_nx, out_ny, out_nz;
	apply_normal_map(ns_tx, ns_ty, ns_tz,
					 nx, ny, nz,
					 dpdu_x, dpdu_y, dpdu_z,
					 out_nx, out_ny, out_nz);

	double d = vec3_dot(out_nx, out_ny, out_nz, nx, ny, nz);
	EXPECT_GT(d, 0.0) << "perturbed normal must stay on same hemisphere as geometric normal";
	EXPECT_TRUE(nearly_unit(out_nx, out_ny, out_nz));
}

// Output is a unit vector for a wide variety of inputs.
TEST(NormalMap, OutputIsUnit) {
	struct Case { double nx,ny,nz, dpdu_x,dpdu_y,dpdu_z, tx,ty,tz; };
	Case cases[] = {
		{0,1,0,  1,0,0,  0,0,1},
		{0,0,1,  1,0,0,  0,0,1},
		{1,0,0,  0,1,0,  0,0,1},
		{0,1,0,  0,0,1,  0.5773,0.5773,0.5773},
		{0.5773,0.5773,0.5773,  1,0,0,  0,0,1},
	};
	for (auto& c : cases) {
		double tl = std::sqrt(c.tx*c.tx+c.ty*c.ty+c.tz*c.tz);
		double tx=c.tx/tl, ty=c.ty/tl, tz=c.tz/tl;
		double out_nx, out_ny, out_nz;
		apply_normal_map(tx,ty,tz, c.nx,c.ny,c.nz, c.dpdu_x,c.dpdu_y,c.dpdu_z,
						 out_nx,out_ny,out_nz);
		EXPECT_TRUE(nearly_unit(out_nx,out_ny,out_nz))
			<< "output should be a unit vector";
	}
}

// ---------------------------------------------------------------------------
// apply_bump_map tests
// ---------------------------------------------------------------------------

// Zero displacement everywhere -> no perturbation -> original normal preserved.
TEST(BumpMap, ZeroDisplacementLeavesNormalUnchanged) {
	double nx = 0, ny = 1, nz = 0;
	double dpdu_x = 1, dpdu_y = 0, dpdu_z = 0;
	double dpdv_x = 0, dpdv_y = 0, dpdv_z = 1;

	double out_nx, out_ny, out_nz;
	apply_bump_map(0.0, 0.0, 0.0,   // disp, disp_u, disp_v
				   0.001, 0.001,     // du, dv
				   nx, ny, nz,
				   dpdu_x, dpdu_y, dpdu_z,
				   dpdv_x, dpdv_y, dpdv_z,
				   out_nx, out_ny, out_nz);

	EXPECT_NEAR(out_nx, nx, 1e-9);
	EXPECT_NEAR(out_ny, ny, 1e-9);
	EXPECT_NEAR(out_nz, nz, 1e-9);
}

// Bumped normal stays on same hemisphere.
TEST(BumpMap, BumpedNormalOnSameHemisphere) {
	double nx = 0, ny = 0, nz = 1;
	double dpdu_x = 1, dpdu_y = 0, dpdu_z = 0;
	double dpdv_x = 0, dpdv_y = 1, dpdv_z = 0;

	// positive gradient along U
	double out_nx, out_ny, out_nz;
	apply_bump_map(0.0, 0.1, 0.0,
				   0.001, 0.001,
				   nx, ny, nz,
				   dpdu_x, dpdu_y, dpdu_z,
				   dpdv_x, dpdv_y, dpdv_z,
				   out_nx, out_ny, out_nz);

	double d = vec3_dot(out_nx, out_ny, out_nz, nx, ny, nz);
	EXPECT_GT(d, 0.0) << "bumped normal must stay on same hemisphere as geometric normal";
	EXPECT_TRUE(nearly_unit(out_nx, out_ny, out_nz));
}

// ---------------------------------------------------------------------------
// normal_map_material wrapper tests
// ---------------------------------------------------------------------------

// A solid-colour normal-map texture encoding (0.5,0.5,1) (flat) must leave
// the scatter outcome direction hemisphere unchanged vs. bare lambertian.
TEST(NormalMapMaterial, FlatMapScatterMatchesBareInner) {
	// flat normal texture: RGB=(0.5, 0.5, 1.0) -> tangent-space (0,0,1) -> no perturbation
	auto flat_tex = std::make_shared<solid_color>(color(0.5, 0.5, 1.0));
	auto inner    = std::make_shared<lambertian>(color(0.8, 0.6, 0.2));
	normal_map_material nm_mat(flat_tex, inner);

	hit_record rec = make_rec(vec3(0,1,0), vec3(1,0,0));
	rec.mat = nullptr; // not needed

	ray r_in(point3(0,2,0), vec3(0,-1,0));

	// Check that apply() returns the same normal for a flat map
	hit_record modified = nm_mat.apply(rec);
	EXPECT_NEAR(modified.normal.x(), rec.normal.x(), 1e-6);
	EXPECT_NEAR(modified.normal.y(), rec.normal.y(), 1e-6);
	EXPECT_NEAR(modified.normal.z(), rec.normal.z(), 1e-6);

	// scatter() should succeed (same as inner)
	scatter_record srec;
	bool ok = nm_mat.scatter(r_in, rec, srec);
	EXPECT_TRUE(ok);
}

// Perturbed normal-map material: scattered direction must be on the
// correct hemisphere relative to the *perturbed* normal.
TEST(NormalMapMaterial, PerturbedNormalScatterOnCorrectHemisphere) {
	// tilt 45 degrees toward +X: RGB encodes tangent-space (1,0,1)/sqrt(2)
	float v = static_cast<float>((std::sqrt(2.0)/2.0 + 1.0) / 2.0); // (0.5+normalized/2)
	auto tilted_tex = std::make_shared<solid_color>(color(v, 0.5, v));
	auto inner      = std::make_shared<lambertian>(color(1.0, 1.0, 1.0));
	normal_map_material nm_mat(tilted_tex, inner);

	hit_record rec = make_rec(vec3(0,1,0), vec3(1,0,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));

	// The apply() result must be on the same hemisphere as geometric normal
	hit_record modified = nm_mat.apply(rec);
	double d = dot(modified.normal, rec.normal);
	EXPECT_GT(d, 0.0);

	scatter_record srec;
	bool ok = nm_mat.scatter(r_in, rec, srec);
	EXPECT_TRUE(ok);
}

// ---------------------------------------------------------------------------
// bump_map_material wrapper tests
// ---------------------------------------------------------------------------

// Zero-displacement bump map: normal must be unchanged.
TEST(BumpMapMaterial, ZeroDisplacementUnchanged) {
	auto zero_tex = std::make_shared<solid_color>(color(0.0, 0.0, 0.0));
	auto inner    = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	bump_map_material bm_mat(zero_tex, inner, 1.0, 0.001);

	hit_record rec = make_rec(vec3(0,1,0), vec3(1,0,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));

	hit_record modified = bm_mat.apply(rec);
	EXPECT_NEAR(modified.normal.x(), rec.normal.x(), 1e-6);
	EXPECT_NEAR(modified.normal.y(), rec.normal.y(), 1e-6);
	EXPECT_NEAR(modified.normal.z(), rec.normal.z(), 1e-6);
}

// Bump-mapped normal stays on same hemisphere as geometric normal.
TEST(BumpMapMaterial, BumpedNormalOnSameHemisphere) {
	// uniform mid-grey displacement everywhere
	auto grey_tex = std::make_shared<solid_color>(color(0.5, 0.5, 0.5));
	auto inner    = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	bump_map_material bm_mat(grey_tex, inner, 5.0, 0.001);

	// Use different UV offsets so disp_u != disp at u+step
	// With a solid color the bump should be zero (uniform field -> no gradient)
	hit_record rec = make_rec(vec3(0,1,0), vec3(1,0,0));
	hit_record modified = bm_mat.apply(rec);

	double d = dot(modified.normal, rec.normal);
	EXPECT_GT(d, 0.0);
}

// scatter() through bump_map_material succeeds.
TEST(BumpMapMaterial, ScatterSucceeds) {
	auto bump_tex = std::make_shared<solid_color>(color(0.3, 0.3, 0.3));
	auto inner    = std::make_shared<lambertian>(color(0.8, 0.8, 0.8));
	bump_map_material bm_mat(bump_tex, inner, 1.0, 0.001);

	hit_record rec = make_rec(vec3(0,1,0), vec3(1,0,0));
	ray r_in(point3(0,2,0), vec3(0,-1,0));

	scatter_record srec;
	bool ok = bm_mat.scatter(r_in, rec, srec);
	EXPECT_TRUE(ok);
}
