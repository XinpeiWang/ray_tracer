// surface_interaction_tests.cpp -- unit tests for src/shared/surface_interaction.h
// Validates SurfaceInteraction<T> and BSDF<BxDF,T> against pbrt-v4 behavior.

#include <gtest/gtest.h>
#include <cmath>

#include "../../src/shared/surface_interaction.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float normalize3(float& x, float& y, float& z) {
	float len = std::sqrt(x*x + y*y + z*z);
	if (len > 0.f) { x /= len; y /= len; z /= len; }
	return len;
}

static float dot3(float ax, float ay, float az,
				  float bx, float by, float bz) {
	return ax*bx + ay*by + az*bz;
}

// ---------------------------------------------------------------------------
// MockBxDF: simple Lambertian-like BxDF in local space for testing BSDF wrapper.
// sample() returns cosine-hemisphere wi in local space.
// eval() returns albedo / pi * |cos(theta_wi)|.
// pdf() returns |cos(theta_wi)| / pi.
// ---------------------------------------------------------------------------
template <typename T>
struct MockDiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;

	// Local-space sample: cosine hemisphere around +Z
	BxDFSampleResult<T> sample(T /*wo_lx*/, T /*wo_ly*/, T wo_lz,
							   T u0, T u1, T /*u2*/) const {
		BxDFSampleResult<T> res{};
		if (wo_lz <= T(0)) { res.valid = false; return res; }

		// Cosine-weighted hemisphere sample in local frame
		T r = std::sqrt(u0);
		T phi = T(2) * T(M_PI) * u1;
		T wi_lx = r * std::cos(phi);
		T wi_ly = r * std::sin(phi);
		T wi_lz = std::sqrt(std::max(T(0), T(1) - u0));

		res.wo_x = wi_lx; res.wo_y = wi_ly; res.wo_z = wi_lz;
		res.r = albedo_r; res.g = albedo_g; res.b = albedo_b;
		res.is_specular = false; res.is_transmission = false;
		res.eta = T(1); res.valid = true;
		return res;
	}

	// eval: albedo / pi
	void eval(T /*wo_lx*/, T /*wo_ly*/, T wo_lz,
			  T wi_lx, T wi_ly, T wi_lz,
			  T& out_r, T& out_g, T& out_b) const {
		T cos_wi = wi_lz;
		if (wo_lz <= T(0) || cos_wi <= T(0)) {
			out_r = out_g = out_b = T(0); return;
		}
		T inv_pi = T(1) / T(M_PI);
		out_r = albedo_r * inv_pi;
		out_g = albedo_g * inv_pi;
		out_b = albedo_b * inv_pi;
	}

	// pdf: cos(theta_wi) / pi
	T pdf(T /*wo_lx*/, T /*wo_ly*/, T wo_lz,
		  T /*wi_lx*/, T /*wi_ly*/, T wi_lz) const {
		if (wo_lz <= T(0) || wi_lz <= T(0)) return T(0);
		return wi_lz / T(M_PI);
	}
};

// ---------------------------------------------------------------------------
// SurfaceInteraction construction
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, DefaultConstructorZeroInit) {
	SurfaceInteraction<float> si;
	EXPECT_FLOAT_EQ(si.px, 0.f);
	EXPECT_FLOAT_EQ(si.ny, 1.f);  // default normal points up
	EXPECT_FLOAT_EQ(si.t,  0.f);
}

TEST(SurfaceInteraction, ConstructorSetsGeometry) {
	// Hit at (1,2,3), normal (0,1,0), uv (0.5, 0.25), t=1.5
	SurfaceInteraction<float> si(
		1.f, 2.f, 3.f,       // p
		0.f, 1.f, 0.f,       // n
		0.5f, 0.25f,         // uv
		1.5f,                // t
		0.f, 0.f, -1.f,      // wo (toward camera)
		1.f, 0.f, 0.f,       // dpdu
		0.f, 0.f, 1.f        // dpdv
	);

	EXPECT_FLOAT_EQ(si.px, 1.f);
	EXPECT_FLOAT_EQ(si.py, 2.f);
	EXPECT_FLOAT_EQ(si.pz, 3.f);
	EXPECT_FLOAT_EQ(si.nx, 0.f);
	EXPECT_FLOAT_EQ(si.ny, 1.f);
	EXPECT_FLOAT_EQ(si.nz, 0.f);
	EXPECT_FLOAT_EQ(si.u,  0.5f);
	EXPECT_FLOAT_EQ(si.v,  0.25f);
	EXPECT_FLOAT_EQ(si.t,  1.5f);
}

TEST(SurfaceInteraction, ShadingNormalInitEqualsGeometric) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f,
		0.f, 1.f, 0.f,
		0.f, 0.f, 0.f,
		0.f, 0.f, -1.f,
		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f
	);
	EXPECT_FLOAT_EQ(si.ns_x, si.nx);
	EXPECT_FLOAT_EQ(si.ns_y, si.ny);
	EXPECT_FLOAT_EQ(si.ns_z, si.nz);
}

TEST(SurfaceInteraction, FlipNormalNegatesGeomAndShading) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f,
		0.f, 1.f, 0.f,
		0.f, 0.f, 0.f,
		0.f, 0.f, -1.f,
		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		/*flip_normal=*/true
	);
	EXPECT_FLOAT_EQ(si.ny,   -1.f);
	EXPECT_FLOAT_EQ(si.ns_y, -1.f);
}

// ---------------------------------------------------------------------------
// set_shading_geometry
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, SetShadingGeometryUpdatesNs) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	// Set a tilted shading normal
	float ns_x = 0.1f, ns_y = 0.9f, ns_z = 0.f;
	normalize3(ns_x, ns_y, ns_z);
	si.set_shading_geometry(ns_x, ns_y, ns_z, 1.f, 0.f, 0.f, true);

	EXPECT_NEAR(si.ns_x, ns_x, 1e-5f);
	EXPECT_NEAR(si.ns_y, ns_y, 1e-5f);
	EXPECT_NEAR(si.ns_z, ns_z, 1e-5f);
}

TEST(SurfaceInteraction, SetShadingGeometryOrientationAuthoritative) {
	// geo normal = (0,1,0), set shading normal to (0,-1,0) with authoritative=true
	// -> geo normal should be flipped to point same hemisphere as shading
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	si.set_shading_geometry(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, /*authoritative=*/true);
	// geo normal (0,1,0) dot shading_ns (0,-1,0) < 0 -> flip geo to (0,-1,0)
	EXPECT_LT(si.ny, 0.f);
}

TEST(SurfaceInteraction, SetShadingGeometryNonAuthoritative) {
	// geo normal = (0,1,0), set shading normal to (0,-1,0) with authoritative=false
	// -> shading normal should be flipped to match geo
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	si.set_shading_geometry(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, /*authoritative=*/false);
	// geo normal stays (0,1,0); shading_ns flipped to (0,1,0)
	EXPECT_GT(si.ns_y, 0.f);
	EXPECT_GT(si.ny,   0.f);
}

// ---------------------------------------------------------------------------
// to_material_context
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, ToMaterialContextCopiesFields) {
	SurfaceInteraction<float> si(
		1.f, 2.f, 3.f, 0.f, 1.f, 0.f, 0.5f, 0.25f, 1.5f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);
	si.dudx = 0.1f; si.dvdx = 0.2f;

	auto ctx = si.to_material_context();
	EXPECT_FLOAT_EQ(ctx.px, 1.f);
	EXPECT_FLOAT_EQ(ctx.py, 2.f);
	EXPECT_FLOAT_EQ(ctx.pz, 3.f);
	EXPECT_FLOAT_EQ(ctx.nx, si.ns_x);
	EXPECT_FLOAT_EQ(ctx.ny, si.ns_y);
	EXPECT_FLOAT_EQ(ctx.u,  0.5f);
	EXPECT_FLOAT_EQ(ctx.v,  0.25f);
	EXPECT_FLOAT_EQ(ctx.dudx, 0.1f);
	EXPECT_FLOAT_EQ(ctx.dvdx, 0.2f);
}

// ---------------------------------------------------------------------------
// compute_differentials
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, ComputeDifferentialsNoDiff) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	si.compute_differentials(false,
		0,0,0, 0,0,0, 0,0,0, 0,0,0);

	EXPECT_FLOAT_EQ(si.dpdx_x, 0.f);
	EXPECT_FLOAT_EQ(si.dpdy_y, 0.f);
	EXPECT_FLOAT_EQ(si.dudx,   0.f);
	EXPECT_FLOAT_EQ(si.dvdy,   0.f);
}

TEST(SurfaceInteraction, ComputeDifferentialsAxisAlignedRays) {
	// Hit at (0,0,0) on xz-plane (normal = (0,1,0))
	// dpdu = (1,0,0), dpdv = (0,0,1)
	// x-ray offset dx along +X, y-ray offset dy along +Z
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f,       // p
		0.f, 1.f, 0.f,       // n
		0.f, 0.f,            // uv
		0.f,                 // t
		0.f, -1.f, 0.f,      // wo (from above)
		1.f, 0.f, 0.f,       // dpdu = (1,0,0)
		0.f, 0.f, 1.f        // dpdv = (0,0,1)
	);

	// x-differential ray: offset by (delta, 0, 0), pointing straight down
	float dx = 0.01f;
	// y-differential ray: offset by (0, 0, delta), pointing straight down
	float dz = 0.01f;

	si.compute_differentials(true,
		dx, 0.f, 0.f,  0.f, -1.f, 0.f,   // rx: origin (dx,0,0), dir (0,-1,0)
		0.f, 0.f, dz,  0.f, -1.f, 0.f);  // ry: origin (0,0,dz), dir (0,-1,0)

	// dpdx = px_hit - p = (dx,0,0) - (0,0,0) = (dx,0,0)
	EXPECT_NEAR(si.dpdx_x, dx,  1e-5f);
	EXPECT_NEAR(si.dpdx_y, 0.f, 1e-5f);
	EXPECT_NEAR(si.dpdx_z, 0.f, 1e-5f);

	// dpdy = (0,0,dz)
	EXPECT_NEAR(si.dpdy_x, 0.f, 1e-5f);
	EXPECT_NEAR(si.dpdy_z, dz,  1e-5f);

	// dudx = (dpdu · dpdx) / |dpdu|^2 = dx / 1 = dx
	EXPECT_NEAR(si.dudx, dx,  1e-5f);
	EXPECT_NEAR(si.dvdx, 0.f, 1e-5f);
	EXPECT_NEAR(si.dudy, 0.f, 1e-5f);
	EXPECT_NEAR(si.dvdy, dz,  1e-5f);
}

// ---------------------------------------------------------------------------
// spawn_ray_origin
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, SpawnRayOriginOffsetAlongNormal) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	float ox, oy, oz;
	si.spawn_ray_origin(ox, oy, oz, 1.f, 1e-4f);

	EXPECT_NEAR(ox, 0.f,   1e-8f);
	EXPECT_NEAR(oy, 1e-4f, 1e-8f);
	EXPECT_NEAR(oz, 0.f,   1e-8f);
}

TEST(SurfaceInteraction, SpawnRayOriginTransmissionSideNegative) {
	SurfaceInteraction<float> si(
		0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	float ox, oy, oz;
	si.spawn_ray_origin(ox, oy, oz, -1.f, 1e-4f);

	// Offset below the surface: y = 1 - 1e-4
	EXPECT_NEAR(oy, 1.f - 1e-4f, 1e-7f);
}

// ---------------------------------------------------------------------------
// geo_dot / shading_dot
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, GeoDotCorrect) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	EXPECT_NEAR(si.geo_dot(0.f, 1.f, 0.f), 1.f, 1e-6f);
	EXPECT_NEAR(si.geo_dot(1.f, 0.f, 0.f), 0.f, 1e-6f);
	EXPECT_NEAR(si.geo_dot(0.f, -1.f, 0.f), -1.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// BSDF construction
// ---------------------------------------------------------------------------
TEST(BSDF, ConstructFromSI) {
	SurfaceInteraction<float> si(
		0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f
	);

	MockDiffuseBxDF<float> bxdf{0.8f, 0.6f, 0.4f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, si);

	// Shading frame z should match shading normal (0,1,0)
	EXPECT_NEAR(bsdf.frame.nx, 0.f, 1e-5f);
	EXPECT_NEAR(bsdf.frame.ny, 1.f, 1e-5f);
	EXPECT_NEAR(bsdf.frame.nz, 0.f, 1e-5f);
}

TEST(BSDF, DirectConstructFromNormal) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);
	EXPECT_NEAR(bsdf.frame.ny, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// BSDF::render_to_local / local_to_render roundtrip
// Mirrors pbrt-v4 BSDF::RenderToLocal / LocalToRender
// ---------------------------------------------------------------------------
TEST(BSDF, RenderToLocalRoundTrip) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// Normal is (0,1,0), so "up" in render space == z-axis in local space
	float lx, ly, lz;
	bsdf.render_to_local(0.f, 1.f, 0.f, lx, ly, lz);
	EXPECT_NEAR(lz, 1.f, 1e-5f);  // local z = normal direction

	// Roundtrip: local_to_render should recover (0,1,0)
	float wx, wy, wz;
	bsdf.local_to_render(lx, ly, lz, wx, wy, wz);
	EXPECT_NEAR(wx, 0.f, 1e-5f);
	EXPECT_NEAR(wy, 1.f, 1e-5f);
	EXPECT_NEAR(wz, 0.f, 1e-5f);
}

TEST(BSDF, RenderToLocalArbitraryDirection) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// Direction (1,1,0) normalized = (1/sqrt2, 1/sqrt2, 0)
	float wx = 1.f/std::sqrt(2.f), wy = 1.f/std::sqrt(2.f), wz = 0.f;
	float lx, ly, lz;
	bsdf.render_to_local(wx, wy, wz, lx, ly, lz);

	// local z component = dot((wx,wy,wz), n=(0,1,0)) = 1/sqrt2
	EXPECT_NEAR(lz, 1.f/std::sqrt(2.f), 1e-5f);

	// Roundtrip
	float rx, ry, rz;
	bsdf.local_to_render(lx, ly, lz, rx, ry, rz);
	EXPECT_NEAR(rx, wx, 1e-5f);
	EXPECT_NEAR(ry, wy, 1e-5f);
	EXPECT_NEAR(rz, wz, 1e-5f);
}

// ---------------------------------------------------------------------------
// BSDF::f -- evaluate BRDF in render space
// ---------------------------------------------------------------------------
TEST(BSDF, FEvalAboveSurfaceNonZero) {
	MockDiffuseBxDF<float> bxdf{0.8f, 0.5f, 0.3f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// wo and wi both in upper hemisphere (y > 0)
	float r, g, b;
	bsdf.f(0.f, 1.f, 0.f,   // wo = (0,1,0) -> local z = 1
		   0.f, 1.f, 0.f,   // wi = (0,1,0)
		   r, g, b);

	EXPECT_GT(r, 0.f);
	EXPECT_GT(g, 0.f);
	EXPECT_GT(b, 0.f);
	// Lambertian: f = albedo / pi
	EXPECT_NEAR(r, 0.8f / float(M_PI), 1e-5f);
}

TEST(BSDF, FEvalBelowSurfaceZero) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// wo below surface (y < 0) -> local z < 0 -> should return 0
	float r, g, b;
	bsdf.f(0.f, -1.f, 0.f,   // wo below surface
		   0.f,  1.f, 0.f,
		   r, g, b);

	EXPECT_FLOAT_EQ(r, 0.f);
	EXPECT_FLOAT_EQ(g, 0.f);
	EXPECT_FLOAT_EQ(b, 0.f);
}

TEST(BSDF, FEvalAlbedoScalesResult) {
	MockDiffuseBxDF<float> bxdf1{1.f, 1.f, 1.f};
	MockDiffuseBxDF<float> bxdf2{0.5f, 0.5f, 0.5f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf1(bxdf1, 0.f, 1.f, 0.f);
	BSDF<MockDiffuseBxDF<float>, float> bsdf2(bxdf2, 0.f, 1.f, 0.f);

	float r1, g1, b1, r2, g2, b2;
	bsdf1.f(0.f, 1.f, 0.f, 0.f, 1.f, 0.f, r1, g1, b1);
	bsdf2.f(0.f, 1.f, 0.f, 0.f, 1.f, 0.f, r2, g2, b2);

	EXPECT_NEAR(r1 / r2, 2.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// BSDF::sample_f -- sample scattering direction
// ---------------------------------------------------------------------------
TEST(BSDF, SampleFReturnsValidResult) {
	MockDiffuseBxDF<float> bxdf{0.8f, 0.6f, 0.4f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	auto bs = bsdf.sample_f(0.f, 1.f, 0.f,   // wo = up
							0.5f, 0.3f, 0.7f);

	EXPECT_TRUE(bs.valid);
	EXPECT_GT(bs.r, 0.f);
}

TEST(BSDF, SampleFWiInUpperHemisphere) {
	// Normal = (0,1,0), sample should be in upper hemisphere (wy > 0)
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// Test several samples
	float us[4][3] = {{0.1f,0.2f,0.3f},{0.5f,0.5f,0.5f},{0.9f,0.1f,0.5f},{0.3f,0.7f,0.8f}};
	for (auto& u : us) {
		auto bs = bsdf.sample_f(0.f, 1.f, 0.f, u[0], u[1], u[2]);
		EXPECT_TRUE(bs.valid);
		// wi in render space should have positive y (upper hemisphere)
		// wo_x/y/z in BxDFSampleResult = sampled wi in render space
		EXPECT_GT(bs.wo_y, -1e-5f);  // must be in upper hemisphere
	}
}

TEST(BSDF, SampleFWoBelowSurfaceInvalid) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// wo below surface -> local wo_z = dot((0,-1,0),(0,1,0)) = -1 < 0 -> invalid
	auto bs = bsdf.sample_f(0.f, -1.f, 0.f, 0.5f, 0.5f, 0.5f);
	EXPECT_FALSE(bs.valid);
}

// ---------------------------------------------------------------------------
// BSDF::pdf
// ---------------------------------------------------------------------------
TEST(BSDF, PDFPositiveForUpperHemisphere) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	// Both wo and wi in upper hemisphere
	float p = bsdf.pdf(0.f, 1.f, 0.f,  // wo
					   0.f, 1.f, 0.f); // wi
	EXPECT_GT(p, 0.f);
	// For Lambertian: pdf = cos(theta_wi) / pi; here theta_wi = 0 -> cos=1
	EXPECT_NEAR(p, 1.f / float(M_PI), 1e-5f);
}

TEST(BSDF, PDFZeroForBelowSurface) {
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	float p = bsdf.pdf(0.f, -1.f, 0.f,  // wo below surface
					   0.f,  1.f, 0.f);
	EXPECT_FLOAT_EQ(p, 0.f);
}

TEST(BSDF, PDFConsistentWithSampleF) {
	// PDF of sampling at exact wi should match pdf() evaluated at same wi
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, 0.f, 1.f, 0.f);

	float wo_x = 0.f, wo_y = 1.f, wo_z = 0.f;
	auto bs = bsdf.sample_f(wo_x, wo_y, wo_z, 0.5f, 0.5f, 0.5f);
	ASSERT_TRUE(bs.valid);

	// Evaluate pdf at the sampled direction
	float p = bsdf.pdf(wo_x, wo_y, wo_z,
					   bs.wo_x, bs.wo_y, bs.wo_z);
	// For cosine hemisphere: pdf = cos(theta_wi) / pi
	// cos(theta_wi) = dot(wi, normal=(0,1,0)) = bs.wo_y
	float expected_pdf = bs.wo_y / float(M_PI);
	EXPECT_NEAR(p, expected_pdf, 1e-4f);
}

// ---------------------------------------------------------------------------
// BSDF with tilted normal -- verify transform is correct
// ---------------------------------------------------------------------------
TEST(BSDF, TiltedNormalFrameConsistent) {
	// Normal at 45 degrees: (1/sqrt2, 1/sqrt2, 0)
	float nx = 1.f/std::sqrt(2.f), ny = 1.f/std::sqrt(2.f), nz = 0.f;
	MockDiffuseBxDF<float> bxdf{1.f, 1.f, 1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, nx, ny, nz);

	// Direction along normal should map to local z = 1
	float lx, ly, lz;
	bsdf.render_to_local(nx, ny, nz, lx, ly, lz);
	EXPECT_NEAR(lz, 1.f, 1e-5f);

	// Sample should produce valid wi
	auto bs = bsdf.sample_f(nx, ny, nz, 0.5f, 0.5f, 0.5f);
	EXPECT_TRUE(bs.valid);

	// wi should be in the upper hemisphere of the tilted normal
	float wi_dot_n = dot3(bs.wo_x, bs.wo_y, bs.wo_z, nx, ny, nz);
	EXPECT_GT(wi_dot_n, -1e-4f);
}

// ---------------------------------------------------------------------------
// Double precision
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, DoubleConstructor) {
	SurfaceInteraction<double> si(
		1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.5, 0.25, 1.5,
		0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
	);
	EXPECT_DOUBLE_EQ(si.px, 1.0);
	EXPECT_DOUBLE_EQ(si.ny, 1.0);
	EXPECT_DOUBLE_EQ(si.t,  1.5);
}

TEST(BSDF, DoublePrecisionSampleValid) {
	MockDiffuseBxDF<double> bxdf{0.8, 0.6, 0.4};
	BSDF<MockDiffuseBxDF<double>, double> bsdf(bxdf, 0.0, 1.0, 0.0);

	auto bs = bsdf.sample_f(0.0, 1.0, 0.0, 0.5, 0.3, 0.7);
	EXPECT_TRUE(bs.valid);
	EXPECT_GT(bs.r, 0.0);
}

// ---------------------------------------------------------------------------
// ShadingFrame::from_xz -- pbrt-v4 Frame::FromXZ alignment tests
// ---------------------------------------------------------------------------
TEST(ShadingFrame, FromXZOrthonormal) {
	// Tangent along X, normal along Z (standard basis)
	ShadingFrame<float> f = ShadingFrame<float>::from_xz(1.f,0.f,0.f, 0.f,0.f,1.f);
	// x-axis = tangent
	EXPECT_NEAR(f.tx, 1.f, 1e-6f); EXPECT_NEAR(f.ty, 0.f, 1e-6f); EXPECT_NEAR(f.tz, 0.f, 1e-6f);
	// z-axis = normal
	EXPECT_NEAR(f.nx, 0.f, 1e-6f); EXPECT_NEAR(f.ny, 0.f, 1e-6f); EXPECT_NEAR(f.nz, 1.f, 1e-6f);
	// bitangent = cross(z, x) = (0,0,1) x (1,0,0) = (0,1,0)
	EXPECT_NEAR(f.bx, 0.f, 1e-6f); EXPECT_NEAR(f.by, 1.f, 1e-6f); EXPECT_NEAR(f.bz, 0.f, 1e-6f);
}

TEST(ShadingFrame, FromXZPreservesNormal) {
	// Arbitrary normal (0,1,0), tangent (1,0,0)
	ShadingFrame<float> f = ShadingFrame<float>::from_xz(1.f,0.f,0.f, 0.f,1.f,0.f);
	// Normal component of normal direction should be 1
	float lx, ly, lz;
	f.to_local(0.f,1.f,0.f, lx,ly,lz);
	EXPECT_NEAR(lz, 1.f, 1e-6f);
}

TEST(ShadingFrame, FromXZRoundTrip) {
	// Arbitrary orthonormal pair: normal = normalize(1,1,1), tangent = normalize(1,-1,0)
	float nx=1.f,ny=1.f,nz=1.f, tx=1.f,ty=-1.f,tz=0.f;
	float nlen = std::sqrt(3.f), tlen = std::sqrt(2.f);
	nx/=nlen; ny/=nlen; nz/=nlen;
	tx/=tlen; ty/=tlen;
	ShadingFrame<float> f = ShadingFrame<float>::from_xz(tx,ty,tz, nx,ny,nz);
	// Round-trip: to_local then to_world
	float wx=0.3f,wy=0.5f,wz=0.8f;
	float lx,ly,lz, ox,oy,oz;
	f.to_local(wx,wy,wz, lx,ly,lz);
	f.to_world(lx,ly,lz, ox,oy,oz);
	EXPECT_NEAR(ox, wx, 1e-5f); EXPECT_NEAR(oy, wy, 1e-5f); EXPECT_NEAR(oz, wz, 1e-5f);
}

// ---------------------------------------------------------------------------
// BSDF -- tangent-aligned frame (from_xz path via SI constructor)
// ---------------------------------------------------------------------------
TEST(BSDF, TangentAlignedFrameXAxisMapsToDpdu) {
	// SI with normal = (0,1,0), dpdu = (1,0,0) -> shading tangent along X
	SurfaceInteraction<float> si(
		0.f,0.f,0.f,  0.f,1.f,0.f,  0.f,0.f,  0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f,  0.f,0.f,1.f
	);
	MockDiffuseBxDF<float> bxdf{1.f,1.f,1.f};
	BSDF<MockDiffuseBxDF<float>, float> bsdf(bxdf, si);

	// dpdu direction should map to local x (lz=0, lx=1)
	float lx,ly,lz;
	bsdf.render_to_local(1.f,0.f,0.f, lx,ly,lz);
	EXPECT_NEAR(lx, 1.f, 1e-5f);
	EXPECT_NEAR(lz, 0.f, 1e-5f);

	// normal direction should map to local z=1
	bsdf.render_to_local(0.f,1.f,0.f, lx,ly,lz);
	EXPECT_NEAR(lz, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// spawn_ray_origin -- direction-based overload (pbrt-v4 OffsetRayOrigin(w))
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, SpawnRayOriginDirectionFrontSide) {
	SurfaceInteraction<float> si(
		0.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f, 0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f, 0.f,0.f,1.f
	);
	float ox,oy,oz;
	// Direction pointing away from surface (same side as normal)
	si.spawn_ray_origin(ox,oy,oz, 0.f,1.f,0.f);
	EXPECT_NEAR(ox, 0.f, 1e-6f);
	EXPECT_GT(oy, 0.f);   // offset along +n
	EXPECT_NEAR(oz, 0.f, 1e-6f);
}

TEST(SurfaceInteraction, SpawnRayOriginDirectionBackSide) {
	SurfaceInteraction<float> si(
		0.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f, 0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f, 0.f,0.f,1.f
	);
	float ox,oy,oz;
	// Direction pointing into the surface (opposite to normal) -> transmission
	si.spawn_ray_origin(ox,oy,oz, 0.f,-1.f,0.f);
	EXPECT_NEAR(ox, 0.f, 1e-6f);
	EXPECT_LT(oy, 0.f);   // offset along -n
	EXPECT_NEAR(oz, 0.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// set_shading_geometry -- shading_dpdv stored, length^2 clamp
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, SetShadingGeometryStoresDpdv) {
	SurfaceInteraction<float> si(
		0.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f, 0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f, 0.f,0.f,1.f
	);
	si.set_shading_geometry(0.f,1.f,0.f, 1.f,0.f,0.f, 0.f,0.f,1.f);
	EXPECT_NEAR(si.shading_dpdv_x, 0.f, 1e-6f);
	EXPECT_NEAR(si.shading_dpdv_y, 0.f, 1e-6f);
	EXPECT_NEAR(si.shading_dpdv_z, 1.f, 1e-6f);
}

TEST(SurfaceInteraction, SetShadingGeometryLargeDerivativesAreClamped) {
	SurfaceInteraction<float> si(
		0.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f, 0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f, 0.f,0.f,1.f
	);
	// Derivatives with length >> 1e8 (length^2 > 1e16) should be scaled down
	float big = 1e9f;
	si.set_shading_geometry(0.f,1.f,0.f, big,0.f,0.f, 0.f,big,0.f);
	float len2u = si.shading_dpdu_x*si.shading_dpdu_x + si.shading_dpdu_y*si.shading_dpdu_y + si.shading_dpdu_z*si.shading_dpdu_z;
	float len2v = si.shading_dpdv_x*si.shading_dpdv_x + si.shading_dpdv_y*si.shading_dpdv_y + si.shading_dpdv_z*si.shading_dpdv_z;
	EXPECT_LE(len2u, 1e16f);
	EXPECT_LE(len2v, 1e16f);
}

// ---------------------------------------------------------------------------
// compute_differentials -- degenerate (det=0) returns zero derivatives
// ---------------------------------------------------------------------------
TEST(SurfaceInteraction, ComputeDifferentialsDegenerateSurfaceZeroDerivs) {
	// dpdu = dpdv = (1,0,0) => det = 0 => should return all zeros
	SurfaceInteraction<float> si(
		0.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f, 0.f,
		0.f,-1.f,0.f, 1.f,0.f,0.f, 1.f,0.f,0.f  // dpdu == dpdv
	);
	si.compute_differentials(true,
		0.01f,0.f,0.f, 0.f,-1.f,0.f,
		0.f,0.f,0.01f, 0.f,-1.f,0.f);
	EXPECT_FLOAT_EQ(si.dudx, 0.f);
	EXPECT_FLOAT_EQ(si.dvdx, 0.f);
	EXPECT_FLOAT_EQ(si.dudy, 0.f);
	EXPECT_FLOAT_EQ(si.dvdy, 0.f);
}
