/**
 * @file transform_instance_tests.cpp
 * @brief Unit tests for placing shared geometry under an affine transform
 *
 * These are ray-casting tests rather than data tests, because the thing that
 * can be wrong here is not what is stored but where a ray finds it. A
 * transform applied in the wrong direction, or an inverse computed wrongly,
 * still produces perfectly well-formed geometry - just somewhere else.
 */

#include <gtest/gtest.h>

#include "rtweekend.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "transform_instance.h"

#include <cmath>
#include <memory>

namespace {

// A unit sphere at the origin - the shared geometry every test below places.
std::shared_ptr<hittable> unitSphere() {
	auto mat = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	return std::make_shared<sphere>(point3(0, 0, 0), 1.0, mat);
}

pbrt_scene::Matrix4 translation(double x, double y, double z) {
	pbrt_scene::Matrix4 m;
	m.m[3] = x; m.m[7] = y; m.m[11] = z;
	return m;
}

pbrt_scene::Matrix4 scaling(double x, double y, double z) {
	pbrt_scene::Matrix4 m;
	m.m[0] = x; m.m[5] = y; m.m[10] = z;
	return m;
}

// Fires along +x from far away and reports where it first hits, or -inf.
double hitX(const hittable &h, double y = 0.0, double z = 0.0) {
	const ray r(point3(-1000, y, z), vec3(1, 0, 0));
	hit_record rec;
	if (!h.hit(r, interval(0.001, infinity), rec)) return -std::numeric_limits<double>::infinity();
	return rec.p.x();
}

} // namespace

TEST(TransformInstanceTest, AnIdentityTransformChangesNothing) {
	const transform_instance t(unitSphere(), pbrt_scene::Matrix4::identity());
	EXPECT_NEAR(hitX(t), -1.0, 1e-9);
}

TEST(TransformInstanceTest, TranslationMovesTheGeometryNotTheOppositeWay) {
	// The direction that is easy to get backwards: the RAY goes through the
	// inverse, so the object appears to move by the forward transform. An
	// inverted sign here puts everything at -100 instead of +100 and still
	// renders a perfectly good picture of the wrong scene.
	const transform_instance t(unitSphere(), translation(100, 0, 0));
	EXPECT_NEAR(hitX(t), 99.0, 1e-9);
}

TEST(TransformInstanceTest, ScaleChangesTheApparentSize) {
	const transform_instance t(unitSphere(), scaling(3, 3, 3));
	EXPECT_NEAR(hitX(t), -3.0, 1e-9);
}

TEST(TransformInstanceTest, TheRayParameterIsMeasuredInWorldSpace) {
	// The reason the ray direction is transformed without renormalising. Under
	// a scale, a normalised direction would make rec.t mean something
	// different inside the instance than outside, so the caller's interval and
	// the returned distance would disagree with the rest of the scene.
	const transform_instance t(unitSphere(), scaling(3, 3, 3));
	const ray r(point3(-1000, 0, 0), vec3(1, 0, 0));
	hit_record rec;
	ASSERT_TRUE(t.hit(r, interval(0.001, infinity), rec));
	EXPECT_NEAR(rec.t, 997.0, 1e-6)
		<< "t is not a world-space distance, so ray intervals will misbehave";
}

TEST(TransformInstanceTest, ANonUniformScaleKeepsNormalsOnTheSurface) {
	// The inverse-transpose case. Squash the sphere along y and hit it
	// off-axis: pushing the normal through the forward matrix would tilt it
	// off the ellipsoid, which shows up as shading that looks subtly lit from
	// the wrong direction and nothing more obvious than that.
	const transform_instance t(unitSphere(), scaling(1, 0.25, 1));

	// Fire downward at the top of the squashed sphere, slightly off centre.
	const ray r(point3(0.5, 100, 0), vec3(0, -1, 0));
	hit_record rec;
	ASSERT_TRUE(t.hit(r, interval(0.001, infinity), rec));

	EXPECT_NEAR(rec.normal.length(), 1.0, 1e-9) << "normal is not unit length";

	// For the ellipsoid x^2 + (y/0.25)^2 + z^2 = 1 the outward normal at a
	// point is proportional to (x, y/0.0625, z). Checking the DIRECTION
	// against the surface's own gradient rather than a number I worked out by
	// hand, so the test cannot agree with my arithmetic error.
	const vec3 expected = unit_vector(vec3(rec.p.x(), rec.p.y() / 0.0625, rec.p.z()));
	const double alignment = dot(unit_vector(rec.normal), expected);
	EXPECT_NEAR(std::fabs(alignment), 1.0, 1e-6)
		<< "the normal is not perpendicular to the squashed surface";
}

TEST(TransformInstanceTest, ComposedTransformsApplyInTheRightOrder) {
	// Scale then translate: the sphere should end up radius 2 centred at 10,
	// not radius 2 centred at 20 (which is what applying the scale to the
	// translation would give).
	pbrt_scene::Matrix4 m = translation(10, 0, 0) * scaling(2, 2, 2);
	const transform_instance t(unitSphere(), m);
	EXPECT_NEAR(hitX(t), 8.0, 1e-9);
}

TEST(TransformInstanceTest, TheBoundingBoxContainsTheTransformedGeometry) {
	// Bounds must come from transforming the corners, not from transforming
	// the box - they differ under rotation, and a box that clips its own
	// contents makes the BVH drop hits at odd angles.
	const transform_instance t(unitSphere(), translation(100, 0, 0));
	const aabb b = t.bounding_box();
	EXPECT_LE(b.x.min, 99.0 + 1e-9);
	EXPECT_GE(b.x.max, 101.0 - 1e-9);
}

TEST(TransformInstanceTest, ASingularTransformIsEmptyRatherThanInfinite) {
	// A zero scale cannot be inverted. Returning no hits is a legible symptom;
	// dividing by zero produces infinities that surface much later as geometry
	// mysteriously missing from somewhere else entirely.
	const transform_instance t(unitSphere(), scaling(1, 0, 1));
	const ray r(point3(-1000, 0, 0), vec3(1, 0, 0));
	hit_record rec;
	EXPECT_FALSE(t.hit(r, interval(0.001, infinity), rec));
}

TEST(TransformInstanceTest, TheSameGeometryCanBePlacedTwiceWithoutCopying) {
	// The property the whole feature exists for: two placements, one shared
	// object. If the geometry were copied, this would still pass - so the
	// assertion is on use_count, which only stays low if it is genuinely shared.
	auto shared = unitSphere();
	const long before = shared.use_count();
	const transform_instance a(shared, translation(10, 0, 0));
	const transform_instance b(shared, translation(20, 0, 0));

	EXPECT_NEAR(hitX(a), 9.0, 1e-9);
	EXPECT_NEAR(hitX(b), 19.0, 1e-9);
	EXPECT_EQ(shared.use_count(), before + 2)
		<< "the geometry was copied rather than shared";
}
