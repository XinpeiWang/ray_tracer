// mix_material_tests.cpp
// Unit tests for mix_material (pbrt-v4 MixMaterial stochastic blend).
//
// Validates:
//   - w=0 always delegates to mat_a
//   - w=1 always delegates to mat_b
//   - w=0.5 splits ~50/50 over many samples
//   - emitted() linearly blends between both materials
//   - Texture-weight constructor works
//   - Attenuation is always valid color (>=0)
//   - Nested mix materials work

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable.h"
#include "material.h"
#include "texture.h"

#include <cmath>
#include <memory>

using namespace std;

static hit_record make_hit() {
	hit_record rec;
	rec.p          = point3(0, 0, 0);
	rec.normal     = vec3(0, 0, 1);
	rec.front_face = true;
	rec.t          = 1.0;
	rec.u = 0.5; rec.v = 0.5;
	return rec;
}

// ---------------------------------------------------------------------------
// w=0: should always act like mat_a (lambertian red)
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, WeightZeroAlwaysA) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));
	auto mat_b = make_shared<lambertian>(color(0.0, 0.0, 1.0));
	auto mix   = make_shared<mix_material>(mat_a, mat_b, 0.0);

	hit_record rec = make_hit();
	int red_count = 0;
	for (int i = 0; i < 200; ++i) {
		scatter_record srec;
		ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
		if (mix->scatter(r_in, rec, srec)) {
			if (srec.attenuation.x() > 0.5)
				++red_count;
		}
	}
	// All samples from mat_a (red), so all should have r > 0.5
	EXPECT_EQ(red_count, 200);
}

// ---------------------------------------------------------------------------
// w=1: should always act like mat_b (lambertian blue)
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, WeightOneAlwaysB) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));
	auto mat_b = make_shared<lambertian>(color(0.0, 0.0, 1.0));
	auto mix   = make_shared<mix_material>(mat_a, mat_b, 1.0);

	hit_record rec = make_hit();
	int blue_count = 0;
	for (int i = 0; i < 200; ++i) {
		scatter_record srec;
		ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
		if (mix->scatter(r_in, rec, srec)) {
			if (srec.attenuation.z() > 0.5)
				++blue_count;
		}
	}
	EXPECT_EQ(blue_count, 200);
}

// ---------------------------------------------------------------------------
// w=0.5: should split ~50/50 over a spread of shading points.
//
// mix_material::scatter()'s branch choice is deterministic given rec.p (see
// branch_hash01()'s comment in material_pbrt.h) - scattering_pdf() and
// is_shadow_transmissive() re-derive the SAME hash for the same point so all
// three calls about one scattering event agree, instead of each
// independently re-rolling random_double() and silently disagreeing about
// which sub-material a given event actually used. That means repeating
// scatter() on the exact same hit_record (as this test used to do) always
// returns the same branch now - by design, not a regression - so the ~50/50
// split has to come from varying rec.p across iterations instead, mirroring
// SharedMixMaterialTest.WeightHalfReturnsBothMaterials's own spread-of-points
// approach for the GPU-side MixMaterial<float>, which already models this
// same deterministic-per-point contract.
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, HalfWeightSplits) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));  // red
	auto mat_b = make_shared<lambertian>(color(0.0, 0.0, 1.0));  // blue
	auto mix   = make_shared<mix_material>(mat_a, mat_b, 0.5);

	int a_count = 0, total = 0;
	for (int i = 0; i < 2000; ++i) {
		hit_record rec = make_hit();
		rec.p = point3(i * 0.137, i * 0.251, i * 0.373);
		scatter_record srec;
		ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
		if (mix->scatter(r_in, rec, srec)) {
			++total;
			if (srec.attenuation.x() > 0.5) ++a_count;
		}
	}
	ASSERT_GT(total, 1500);
	double fraction = static_cast<double>(a_count) / total;
	EXPECT_NEAR(fraction, 0.5, 0.05);
}

// ---------------------------------------------------------------------------
// The same shading point must always resolve to the same sub-material -
// this is the actual property fix #12 introduced (scatter()/scattering_pdf()/
// is_shadow_transmissive() no longer independently re-roll and disagree).
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, ChoiceIsDeterministicPerPoint) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));  // red
	auto mat_b = make_shared<lambertian>(color(0.0, 0.0, 1.0));  // blue
	auto mix   = make_shared<mix_material>(mat_a, mat_b, 0.5);

	hit_record rec = make_hit();
	rec.p = point3(1.234, 5.678, -9.012);
	ray r_in(point3(0, 0, -1), vec3(0, 0, 1));

	scatter_record first_srec;
	ASSERT_TRUE(mix->scatter(r_in, rec, first_srec));
	bool first_is_a = first_srec.attenuation.x() > 0.5;

	for (int i = 0; i < 20; ++i) {
		scatter_record srec;
		ASSERT_TRUE(mix->scatter(r_in, rec, srec));
		EXPECT_EQ(srec.attenuation.x() > 0.5, first_is_a);
	}
}

// ---------------------------------------------------------------------------
// Attenuation components are always non-negative
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, AttenuationNonNegative) {
	auto mat_a = make_shared<lambertian>(color(0.8, 0.2, 0.3));
	auto mat_b = make_shared<metal>(color(0.9, 0.9, 0.9), 0.1);
	auto mix   = make_shared<mix_material>(mat_a, mat_b, 0.5);

	hit_record rec = make_hit();
	for (int i = 0; i < 500; ++i) {
		scatter_record srec;
		ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
		if (mix->scatter(r_in, rec, srec)) {
			EXPECT_GE(srec.attenuation.x(), 0.0);
			EXPECT_GE(srec.attenuation.y(), 0.0);
			EXPECT_GE(srec.attenuation.z(), 0.0);
		}
	}
}

// ---------------------------------------------------------------------------
// emitted(): blends linearly between two emitting materials
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, EmittedBlends) {
	auto diffuse_light_a = make_shared<diffuse_light>(color(4.0, 0.0, 0.0));
	auto diffuse_light_b = make_shared<diffuse_light>(color(0.0, 0.0, 4.0));
	auto mix   = make_shared<mix_material>(diffuse_light_a, diffuse_light_b, 0.5);

	hit_record rec = make_hit();
	ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
	color e = mix->emitted(r_in, rec, 0.5, 0.5, rec.p);

	// Expected: (1-0.5)*[4,0,0] + 0.5*[0,0,4] = [2,0,2]
	EXPECT_NEAR(e.x(), 2.0, 1e-9);
	EXPECT_NEAR(e.y(), 0.0, 1e-9);
	EXPECT_NEAR(e.z(), 2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Texture-weight constructor: vary weight per uv
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, TextureWeightConstructor) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));
	auto mat_b = make_shared<lambertian>(color(0.0, 0.0, 1.0));

	// Checker texture: alternates 0 and 1 => effectively a 50/50 mix over domain
	auto wt = make_shared<solid_color>(color(0.5, 0.5, 0.5));
	auto mix = make_shared<mix_material>(mat_a, mat_b, wt);

	EXPECT_EQ(mix->get_weight(), wt);
	EXPECT_EQ(mix->get_mat_a(), mat_a);
	EXPECT_EQ(mix->get_mat_b(), mat_b);
}

// ---------------------------------------------------------------------------
// Nested mix_material: mix(mix(A,B,0.5), C, 0.0) == mix(A,B,0.5)
// ---------------------------------------------------------------------------
TEST(MixMaterialTest, NestedMixWorks) {
	auto mat_a = make_shared<lambertian>(color(1.0, 0.0, 0.0));
	auto mat_b = make_shared<lambertian>(color(0.0, 1.0, 0.0));
	auto mat_c = make_shared<lambertian>(color(0.0, 0.0, 1.0));

	auto inner_mix = make_shared<mix_material>(mat_a, mat_b, 0.5);
	auto outer_mix = make_shared<mix_material>(inner_mix, mat_c, 0.0); // always inner

	hit_record rec = make_hit();
	int c_count = 0;
	for (int i = 0; i < 500; ++i) {
		scatter_record srec;
		ray r_in(point3(0, 0, -1), vec3(0, 0, 1));
		if (outer_mix->scatter(r_in, rec, srec)) {
			// mat_c has z=1; none of mat_a or mat_b do
			if (srec.attenuation.z() > 0.5) ++c_count;
		}
	}
	// w=0 means never mat_c
	EXPECT_EQ(c_count, 0);
}
