// ---------------------------------------------------------------------------
// interval_vec_tests.cpp
// Unit tests for interval_vec.h (Vector3fi, Point3fi, OffsetRayOrigin,
// SpawnRay, SpawnRayTo) and the ShapeHit<T> error fields in shapes.h
//
// Mirrors pbrt-v4 util/vecmath.h and ray.h (Apache-2.0).
//
// Tests:
//  Vector3fi
//    1.  Default constructor gives zero midpoints
//    2.  Scalar constructor: midpoints match, zero error
//    3.  Value+error constructor: error stored correctly
//    4.  IsExact: true for exact, false when error > 0
//    5.  Arithmetic: interval addition propagates error outward
//    6.  Negation: flips sign of midpoint, error unchanged
//
//  Point3fi
//    7.  Default constructor gives zero midpoints
//    8.  Scalar constructor: midpoints match, zero error
//    9.  Value+error constructor: ex/ey/ez = half-width
//   10.  FromValueAndError: interval contains value ± error
//   11.  Point + Vector3fi (interval addition)
//   12.  Point - Point returns Vector3fi
//
//  OffsetRayOrigin / SpawnRay
//   13.  Exact hit on XY plane: offset nudges along +Z when w points up
//   14.  Exact hit on XY plane: offset nudges along -Z when w points down
//   15.  Large error: offset grows with pi.Error()
//   16.  Zero-error hit: offset is exactly zero (rounded result = midpoint)
//   17.  SpawnRay: origin on correct side, direction unchanged
//   18.  SpawnRayTo: origin offset toward target point
//   19.  SpawnRayTo (two interval endpoints): both offset outward
//
//  ShapeHit error fields
//   20.  Sphere intersector fills ex/ey/ez > 0 for non-trivial hit
//   21.  Triangle intersector fills ex/ey/ez > 0 for non-trivial hit
//   22.  ShapeHit::ToPoint3fi reconstructs hit point within error bounds
//
//  SurfaceInteraction wiring (interval-backed spawn_ray_origin)
//   23.  set_error() stores pex/pey/pez; to_point3fi() reflects them
//   24.  spawn_ray_origin with error set uses OffsetRayOrigin (not eps-bias)
//   25.  spawn_ray_origin without error falls back to eps-bias
//   26.  Full pipeline: sphere ShapeHit -> set_error -> spawn_ray_origin
//        produces offset consistent with OffsetRayOrigin
//   27.  Full pipeline: triangle ShapeHit -> set_error -> spawn_ray_origin
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "../../src/shared/interval_vec.h"
#include "../../src/shared/shapes.h"
#include "../../src/shared/surface_interaction.h"
#include <cmath>
#include <limits>

// Tolerance used for midpoint comparisons
static constexpr double kEps = 1e-12;

// ===========================================================================
// Vector3fi tests
// ===========================================================================

TEST(Vector3fiTest, DefaultConstructorZero) {
	Vector3fi v;
	EXPECT_NEAR(v.mx(), 0.0, kEps);
	EXPECT_NEAR(v.my(), 0.0, kEps);
	EXPECT_NEAR(v.mz(), 0.0, kEps);
	EXPECT_NEAR(v.ex(), 0.0, kEps);
	EXPECT_NEAR(v.ey(), 0.0, kEps);
	EXPECT_NEAR(v.ez(), 0.0, kEps);
}

TEST(Vector3fiTest, ScalarConstructorExact) {
	Vector3fi v(1.0, 2.0, 3.0);
	EXPECT_NEAR(v.mx(), 1.0, kEps);
	EXPECT_NEAR(v.my(), 2.0, kEps);
	EXPECT_NEAR(v.mz(), 3.0, kEps);
	EXPECT_TRUE(v.IsExact());
}

TEST(Vector3fiTest, ValueErrorConstructor) {
	Vector3fi v(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
	EXPECT_NEAR(v.mx(), 1.0, 1e-6);
	EXPECT_NEAR(v.my(), 2.0, 1e-6);
	EXPECT_NEAR(v.mz(), 3.0, 1e-6);
	// Half-width should be approximately equal to the input error
	EXPECT_NEAR(v.ex(), 0.1, 1e-6);
	EXPECT_NEAR(v.ey(), 0.2, 1e-6);
	EXPECT_NEAR(v.ez(), 0.3, 1e-6);
	EXPECT_FALSE(v.IsExact());
}

TEST(Vector3fiTest, IsExact) {
	Vector3fi exact(1.0, 2.0, 3.0);
	EXPECT_TRUE(exact.IsExact());

	Vector3fi inexact(1.0, 2.0, 3.0, 1e-5, 0.0, 0.0);
	EXPECT_FALSE(inexact.IsExact());
}

TEST(Vector3fiTest, AdditionPropagatesError) {
	Vector3fi a(1.0, 2.0, 3.0, 0.1, 0.1, 0.1);
	Vector3fi b(4.0, 5.0, 6.0, 0.2, 0.2, 0.2);
	Vector3fi c = a + b;
	// Midpoints should sum
	EXPECT_NEAR(c.mx(), 5.0, 1e-6);
	EXPECT_NEAR(c.my(), 7.0, 1e-6);
	EXPECT_NEAR(c.mz(), 9.0, 1e-6);
	// Error should be at least as large as sum of input errors
	EXPECT_GE(c.ex(), 0.3 - 1e-9);
	EXPECT_GE(c.ey(), 0.3 - 1e-9);
	EXPECT_GE(c.ez(), 0.3 - 1e-9);
}

TEST(Vector3fiTest, Negation) {
	Vector3fi v(1.0, -2.0, 3.0, 0.1, 0.2, 0.3);
	Vector3fi neg = -v;
	EXPECT_NEAR(neg.mx(), -1.0, 1e-6);
	EXPECT_NEAR(neg.my(),  2.0, 1e-6);
	EXPECT_NEAR(neg.mz(), -3.0, 1e-6);
	// Error magnitude preserved
	EXPECT_NEAR(neg.ex(), v.ex(), 1e-9);
	EXPECT_NEAR(neg.ey(), v.ey(), 1e-9);
	EXPECT_NEAR(neg.ez(), v.ez(), 1e-9);
}

// ===========================================================================
// Point3fi tests
// ===========================================================================

TEST(Point3fiTest, DefaultConstructorZero) {
	Point3fi p;
	EXPECT_NEAR(p.mx(), 0.0, kEps);
	EXPECT_NEAR(p.my(), 0.0, kEps);
	EXPECT_NEAR(p.mz(), 0.0, kEps);
	EXPECT_TRUE(p.IsExact());
}

TEST(Point3fiTest, ScalarConstructorExact) {
	Point3fi p(3.0, 4.0, 5.0);
	EXPECT_NEAR(p.mx(), 3.0, kEps);
	EXPECT_NEAR(p.my(), 4.0, kEps);
	EXPECT_NEAR(p.mz(), 5.0, kEps);
	EXPECT_TRUE(p.IsExact());
}

TEST(Point3fiTest, ValueErrorConstructor) {
	Point3fi p(1.0, 2.0, 3.0, 0.01, 0.02, 0.03);
	EXPECT_NEAR(p.mx(), 1.0, 1e-6);
	EXPECT_NEAR(p.ex(), 0.01, 1e-6);
	EXPECT_NEAR(p.ey(), 0.02, 1e-6);
	EXPECT_NEAR(p.ez(), 0.03, 1e-6);
	EXPECT_FALSE(p.IsExact());
}

TEST(Point3fiTest, FromValueAndErrorContainsInterval) {
	// The interval should bracket [v-err, v+err]
	Point3fi p(1.0, 0.0, 0.0, 0.5, 0.0, 0.0);
	EXPECT_LE(p.x.LowerBound(), 0.5 + 1e-9);
	EXPECT_GE(p.x.UpperBound(), 1.5 - 1e-9);
}

TEST(Point3fiTest, AddVector) {
	Point3fi p(1.0, 2.0, 3.0);
	Vector3fi v(10.0, 20.0, 30.0);
	Point3fi q = p + v;
	EXPECT_NEAR(q.mx(), 11.0, kEps);
	EXPECT_NEAR(q.my(), 22.0, kEps);
	EXPECT_NEAR(q.mz(), 33.0, kEps);
}

TEST(Point3fiTest, SubtractPoint) {
	Point3fi a(4.0, 5.0, 6.0);
	Point3fi b(1.0, 2.0, 3.0);
	Vector3fi diff = a - b;
	EXPECT_NEAR(diff.mx(), 3.0, kEps);
	EXPECT_NEAR(diff.my(), 3.0, kEps);
	EXPECT_NEAR(diff.mz(), 3.0, kEps);
}

// ===========================================================================
// OffsetRayOrigin / SpawnRay tests
// ===========================================================================

TEST(OffsetRayOriginTest, ExactHitUpwardNormal) {
	// Hit at z=0 plane, normal pointing up (+Z), outgoing direction also up
	// Offset should push origin above z=0
	Point3fi pi(0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6);
	double ox, oy, oz;
	OffsetRayOrigin(pi, 0.0, 0.0, 1.0,   // normal = +Z
						0.0, 0.0, 1.0,   // w = +Z (same side)
						ox, oy, oz);
	EXPECT_GE(oz, 0.0); // origin should be on or above z=0
}

TEST(OffsetRayOriginTest, ExactHitDownwardDirection) {
	// Hit at z=0, normal up (+Z), outgoing direction downward → offset flips
	Point3fi pi(0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6);
	double ox, oy, oz;
	OffsetRayOrigin(pi, 0.0, 0.0, 1.0,   // normal = +Z
						0.0, 0.0, -1.0,  // w = -Z (opposite side)
						ox, oy, oz);
	EXPECT_LE(oz, 0.0); // origin should be on or below z=0
}

TEST(OffsetRayOriginTest, LargerErrorMeansLargerOffset) {
	// More error → larger offset distance
	auto offset_z = [](double err) {
		Point3fi pi(0.0, 0.0, 0.0, err, err, err);
		double ox, oy, oz;
		OffsetRayOrigin(pi, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, ox, oy, oz);
		return oz;
	};
	double small_off = offset_z(1e-7);
	double large_off = offset_z(1e-3);
	EXPECT_GT(large_off, small_off);
}

TEST(OffsetRayOriginTest, ZeroErrorHitExactLocation) {
	// Zero error → no adjustment needed beyond rounding (offset = 0)
	Point3fi pi(2.0, 3.0, 4.0); // exact point, no error
	double ox, oy, oz;
	OffsetRayOrigin(pi, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, ox, oy, oz);
	// With zero error, d=0 and offset=0, so result should equal midpoint exactly
	EXPECT_NEAR(ox, 2.0, kEps);
	EXPECT_NEAR(oy, 3.0, kEps);
	EXPECT_NEAR(oz, 4.0, kEps);
}

TEST(SpawnRayTest, OriginOnCorrectSide) {
	// Spawn a reflection from a hit on the XZ plane (normal = +Y)
	Point3fi pi(1.0, 0.0, 1.0, 1e-5, 1e-5, 1e-5);
	double dx = 0.0, dy = 1.0, dz = 0.0; // direction: straight up
	double ox, oy, oz;
	SpawnRay(pi, 0.0, 1.0, 0.0, dx, dy, dz, ox, oy, oz);
	// Origin should be above the hit plane (y >= 0)
	EXPECT_GE(oy, 0.0);
}

TEST(SpawnRayToTest, OriginOffsetedTowardTarget) {
	// Shadow ray from hit at origin to target at (5,0,0)
	Point3fi pi(0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5);
	double nx = 0.0, ny = 1.0, nz = 0.0; // normal = +Y
	double tx = 5.0, ty = 1.0, tz = 0.0; // target above hit plane
	double ox, oy, oz;
	SpawnRayTo(pi, nx, ny, nz, tx, ty, tz, ox, oy, oz);
	// Direction from origin to target has positive y component;
	// dot(dir, n) > 0 so we offset along +n (positive y)
	EXPECT_GE(oy, 0.0);
}

TEST(SpawnRayToTest, TwoIntervalEndpoints) {
	// Bidirectional shadow ray — mirrors pbrt-v4 SpawnRayTo(pFrom,nFrom,pTo,nTo).
	// Step 1: pf = OffsetRayOrigin(pFrom, nFrom, pTo - pFrom)
	// Step 2: pt = OffsetRayOrigin(pTo,   nTo,   pf  - pTo)    (uses pf, not -dir)
	Point3fi pFrom(0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5);
	Point3fi pTo  (5.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5);
	// Both normals pointing up (+Y)
	double nfx = 0.0, nfy = 1.0, nfz = 0.0;
	double ntx = 0.0, nty = 1.0, ntz = 0.0;
	double ofx, ofy, ofz, otx, oty, otz;
	SpawnRayTo(pFrom, nfx, nfy, nfz, pTo, ntx, nty, ntz,
			   ofx, ofy, ofz, otx, oty, otz);
	// Both offset origins should lie on the +Y side (dot(w,n) > 0 for both)
	EXPECT_GE(ofy, 0.0);
	EXPECT_GE(oty, 0.0);
	// pFrom offset should be close to (0, 0, 0)
	EXPECT_NEAR(ofx, 0.0, 1e-3);
	// pTo offset should be close to (5, 0, 0)
	EXPECT_NEAR(otx, 5.0, 1e-3);
}

// ===========================================================================
// ShapeHit error field tests (sphere and triangle integrators)
// ===========================================================================

TEST(ShapeHitErrorTest, SphereIntersectorFillsError) {
	// Unit sphere at origin — ray along +X from outside.
	// Hit point is at (1, 0, 0), so only ex > 0; ey and ez are zero
	// because those components of the hit point are zero.
	SphereShape<double> sphere = SphereShape<double>::make(0.0, 0.0, 0.0, 1.0);
	auto hit = sphere.intersect(2.0, 0.0, 0.0,  // ray origin
								-1.0, 0.0, 0.0, // ray direction
								0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	// Hit is at (1,0,0): ex > 0, ey == ez == 0 (correct for gamma(5)*|component|)
	EXPECT_GT(hit->ex, 0.0);
	EXPECT_GE(hit->ey, 0.0);  // zero is correct: hit.y == 0
	EXPECT_GE(hit->ez, 0.0);  // zero is correct: hit.z == 0

	// Verify with a diagonal ray so all three components are non-zero
	double inv = 1.0 / std::sqrt(3.0);
	auto hit2 = sphere.intersect(5.0*inv, 5.0*inv, 5.0*inv,
								 -inv, -inv, -inv,
								 0.0, 1e30);
	ASSERT_TRUE(hit2.has_value());
	EXPECT_GT(hit2->ex, 0.0);
	EXPECT_GT(hit2->ey, 0.0);
	EXPECT_GT(hit2->ez, 0.0);
}

TEST(ShapeHitErrorTest, TriangleIntersectorFillsError) {
	// Axis-aligned triangle in the XY plane
	TriangleShape<double> tri;
	tri.p0x =  0.0; tri.p0y = 0.0; tri.p0z = 0.0;
	tri.p1x =  1.0; tri.p1y = 0.0; tri.p1z = 0.0;
	tri.p2x =  0.0; tri.p2y = 1.0; tri.p2z = 0.0;
	tri.has_shading_normals = false;
	auto hit = tri.intersect(0.25, 0.25, 2.0,   // ray origin above centroid
							 0.0,  0.0, -1.0,   // ray direction downward
							 0.0,  1e30);
	ASSERT_TRUE(hit.has_value());
	// Error bounds should be non-negative (they may be zero for very small
	// barycentrics, but must not be negative)
	EXPECT_GE(hit->ex, 0.0);
	EXPECT_GE(hit->ey, 0.0);
	EXPECT_GE(hit->ez, 0.0);
	// For a hit on a triangle with unit-scale coordinates, error should be
	// at least on the order of gamma(7) * coordinate
	double g7 = (7.0 * std::numeric_limits<double>::epsilon() * 0.5)
			  / (1.0 - 7.0 * std::numeric_limits<double>::epsilon() * 0.5);
	EXPECT_GT(hit->ex + hit->ey + hit->ez, g7 * 0.1);
}

TEST(ShapeHitErrorTest, ToPoint3fiReconstructsHit) {
	// Sphere hit: reconstruct point via ToPoint3fi and verify it lies within
	// the interval error bounds
	SphereShape<double> sphere = SphereShape<double>::make(0.0, 0.0, 0.0, 1.0);
	double ox = 2.0, oy = 0.0, oz = 0.0;
	double dx = -1.0, dy = 0.0, dz = 0.0;
	auto hit = sphere.intersect(ox, oy, oz, dx, dy, dz, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	Point3fi pi = hit->ToPoint3fi(ox, oy, oz, dx, dy, dz);
	// The hit is at x=1, y=0, z=0; midpoint should be close
	EXPECT_NEAR(pi.mx(), 1.0, 1e-6);
	EXPECT_NEAR(pi.my(), 0.0, 1e-6);
	EXPECT_NEAR(pi.mz(), 0.0, 1e-6);
	// The interval should span the actual hit point
	EXPECT_LE(pi.x.LowerBound(), 1.0 + 1e-9);
	EXPECT_GE(pi.x.UpperBound(), 1.0 - 1e-9);
}

// ===========================================================================
// SurfaceInteraction wiring tests
// ===========================================================================

TEST(SurfaceInteractionOffsetTest, SetErrorStoresFields) {
	SurfaceInteraction<double> si;
	si.px = 1.0; si.py = 2.0; si.pz = 3.0;
	si.nx = 0.0; si.ny = 1.0; si.nz = 0.0;
	si.set_error(0.01, 0.02, 0.03);
	EXPECT_DOUBLE_EQ(si.pex, 0.01);
	EXPECT_DOUBLE_EQ(si.pey, 0.02);
	EXPECT_DOUBLE_EQ(si.pez, 0.03);
	// to_point3fi() should reflect stored error
	Point3fi pi = si.to_point3fi();
	EXPECT_NEAR(pi.mx(), 1.0, kEps);
	EXPECT_NEAR(pi.my(), 2.0, kEps);
	EXPECT_NEAR(pi.mz(), 3.0, kEps);
	EXPECT_NEAR(pi.ex(), 0.01, 1e-9);
	EXPECT_NEAR(pi.ey(), 0.02, 1e-9);
	EXPECT_NEAR(pi.ez(), 0.03, 1e-9);
}

TEST(SurfaceInteractionOffsetTest, SpawnRayOriginWithError_UsesOffsetRayOrigin) {
	// Hit at (1,0,0) on a sphere: normal = +X, outgoing = +X
	// set_error with gamma(5)*|p| = gamma(5)*1 for x component
	SurfaceInteraction<double> si;
	si.px = 1.0; si.py = 0.0; si.pz = 0.0;
	si.nx = 1.0; si.ny = 0.0; si.nz = 0.0;

	double g5 = (5.0 * std::numeric_limits<double>::epsilon() * 0.5)
			  / (1.0 - 5.0 * std::numeric_limits<double>::epsilon() * 0.5);
	si.set_error(g5 * 1.0, 0.0, 0.0);  // matches sphere gamma(5)*|x|

	double ox, oy, oz;
	si.spawn_ray_origin(ox, oy, oz, 1.0, 0.0, 0.0);  // outgoing = +X

	// Direct OffsetRayOrigin call on the same Point3fi should give same result
	double ref_ox, ref_oy, ref_oz;
	OffsetRayOrigin(si.to_point3fi(),
					1.0, 0.0, 0.0,  // normal
					1.0, 0.0, 0.0,  // direction
					ref_ox, ref_oy, ref_oz);

	EXPECT_DOUBLE_EQ(ox, ref_ox);
	EXPECT_DOUBLE_EQ(oy, ref_oy);
	EXPECT_DOUBLE_EQ(oz, ref_oz);
	// And the origin should be slightly beyond the surface (>= 1.0)
	EXPECT_GE(ox, 1.0);
}

TEST(SurfaceInteractionOffsetTest, SpawnRayOriginWithoutError_FallsBackToEps) {
	// No set_error call -> pex=pey=pez=0 -> use eps-bias
	SurfaceInteraction<double> si;
	si.px = 0.0; si.py = 0.0; si.pz = 0.0;
	si.nx = 0.0; si.ny = 1.0; si.nz = 0.0;
	// pex/pey/pez remain zero (default)

	double ox, oy, oz;
	double eps = 1e-4;
	si.spawn_ray_origin(ox, oy, oz, 0.0, 1.0, 0.0, eps);  // outgoing = +Y

	// Eps-bias: origin = p + eps * n
	EXPECT_NEAR(ox, 0.0,  kEps);
	EXPECT_NEAR(oy, eps,  kEps);
	EXPECT_NEAR(oz, 0.0,  kEps);
}

TEST(SurfaceInteractionOffsetTest, FullPipeline_SphereShapeHitToSpawnRay) {
	// Full pipeline: sphere intersection -> set_error -> spawn_ray_origin
	SphereShape<double> sphere = SphereShape<double>::make(0.0, 0.0, 0.0, 1.0);
	double rox = 3.0, roy = 0.0, roz = 0.0;
	double rdx = -1.0, rdy = 0.0, rdz = 0.0;
	auto hit = sphere.intersect(rox, roy, roz, rdx, rdy, rdz, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());

	// Build SurfaceInteraction from ShapeHit
	SurfaceInteraction<double> si;
	si.px = rox + hit->t * rdx;
	si.py = roy + hit->t * rdy;
	si.pz = roz + hit->t * rdz;
	si.nx = hit->nx; si.ny = hit->ny; si.nz = hit->nz;
	si.set_error(hit->ex, hit->ey, hit->ez);  // wire shape error bounds

	// Spawn a reflected ray (outgoing = +X, same side as normal)
	double ox, oy, oz;
	si.spawn_ray_origin(ox, oy, oz, 1.0, 0.0, 0.0);

	// The spawned origin must be outside the sphere (ox >= 1.0)
	double dist_from_center = std::sqrt(ox*ox + oy*oy + oz*oz);
	EXPECT_GE(dist_from_center, 1.0 - 1e-9);

	// It must be strictly closer than the eps-bias would produce
	// (mathematically derived offset << 1e-4 for unit sphere)
	double eps_offset = std::abs(ox - 1.0);
	EXPECT_LT(eps_offset, 1e-4);
}

TEST(SurfaceInteractionOffsetTest, FullPipeline_TriangleShapeHitToSpawnRay) {
	// Full pipeline: triangle intersection -> set_error -> spawn_ray_origin
	TriangleShape<double> tri;
	tri.p0x = 0.0; tri.p0y = 0.0; tri.p0z = 0.0;
	tri.p1x = 1.0; tri.p1y = 0.0; tri.p1z = 0.0;
	tri.p2x = 0.0; tri.p2y = 1.0; tri.p2z = 0.0;
	tri.has_shading_normals = false;

	double rox = 0.25, roy = 0.25, roz = 3.0;
	double rdx = 0.0,  rdy = 0.0,  rdz = -1.0;
	auto hit = tri.intersect(rox, roy, roz, rdx, rdy, rdz, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());

	SurfaceInteraction<double> si;
	si.px = rox + hit->t * rdx;
	si.py = roy + hit->t * rdy;
	si.pz = roz + hit->t * rdz;
	si.nx = hit->nx; si.ny = hit->ny; si.nz = hit->nz;
	si.set_error(hit->ex, hit->ey, hit->ez);

	// Spawn toward +Z (reflection side)
	double ox, oy, oz;
	si.spawn_ray_origin(ox, oy, oz, 0.0, 0.0, 1.0);

	// Triangle is in z=0 plane, normal = +Z; origin should be above z=0
	EXPECT_GE(oz, 0.0);
	// Offset should be tiny (gamma-derived, not eps=1e-4)
	EXPECT_LT(std::abs(oz), 1e-4);
}

