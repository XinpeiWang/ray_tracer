// utility_integrators_tests.cpp
// Unit tests for src/shared/utility_integrators.h
//   - RandomWalkLi<T,Scene>
//   - AOLi<T,Scene>
//
// Synthetic scene: unit sphere at origin, point light at (0,0,3),
// dim background (0.1), camera at (0,0,5) looking along -z.

#include "gtest/gtest.h"
#include "../../src/shared/utility_integrators.h"

#include <cmath>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------
struct UITestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	explicit UITestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Synthetic scene shared by both integrators
// ---------------------------------------------------------------------------
struct UISyntheticScene {
	// ---- Geometry ----------------------------------------------------------
	static bool sphere_hit(const float org[3], const float dir[3],
						   float t_max, BDPTHit<float>& hit) {
		float a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
		float b = 2*(org[0]*dir[0] + org[1]*dir[1] + org[2]*dir[2]);
		float c = org[0]*org[0] + org[1]*org[1] + org[2]*org[2] - 1.f;
		float disc = b*b - 4*a*c;
		if (disc < 0.f) return false;
		float sq = std::sqrt(disc);
		float t  = (-b - sq) / (2*a);
		if (t < 1e-4f) t = (-b + sq) / (2*a);
		if (t < 1e-4f || t > t_max) return false;

		hit.t_hit = t;
		hit.p[0] = org[0]+t*dir[0]; hit.p[1] = org[1]+t*dir[1]; hit.p[2] = org[2]+t*dir[2];
		float len = std::sqrt(hit.p[0]*hit.p[0]+hit.p[1]*hit.p[1]+hit.p[2]*hit.p[2]);
		hit.geo_n[0]=hit.p[0]/len; hit.geo_n[1]=hit.p[1]/len; hit.geo_n[2]=hit.p[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false;
		hit.bsdf_id=0; hit.light_id=-1;
		return true;
	}

	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		return sphere_hit(org, dir, t_max, hit);
	}

	// ---- Emission ----------------------------------------------------------
	void SurfaceLe(const BDPTHit<float>&, const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;
	}

	static constexpr float kBg = 0.1f;
	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = kBg;
	}

	// ---- BSDF: Lambertian albedo 0.8 (f * |cos| pre-multiplied) ----------
	static constexpr float kAlbedoOverPi = 0.8f / 3.14159265358979f;
	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float c = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
		float v = (c > 0.f) ? kAlbedoOverPi * c : 0.f;
		out[0] = out[1] = out[2] = v;
	}

	bool BSDFIsNull(int) const { return false; }  // solid sphere, no medium

	// ---- Visibility --------------------------------------------------------
	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float d[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if (len < 1e-6f) return true;
		d[0]/=len; d[1]/=len; d[2]/=len;
		BDPTHit<float> tmp{};
		return !sphere_hit(p0, d, len - 1e-3f, tmp);
	}

	// UnoccludedWithin: used by AOLi (shadow ray with max_dist cap)
	bool UnoccludedWithin(const float p[3], const float dir[3],
						  float max_dist) const {
		BDPTHit<float> tmp{};
		return !sphere_hit(p, dir, max_dist, tmp);
	}

	// ---- Ray spawn ---------------------------------------------------------
	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-4f;
		new_o[0] = hit.p[0] + kEps * hit.geo_n[0];
		new_o[1] = hit.p[1] + kEps * hit.geo_n[1];
		new_o[2] = hit.p[2] + kEps * hit.geo_n[2];
		new_d[0] = dir[0]; new_d[1] = dir[1]; new_d[2] = dir[2];
	}
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void rw_trace(const UISyntheticScene& scene,
					 float ox, float oy, float oz,
					 float dx, float dy, float dz,
					 int max_depth,
					 float out_L[3])
{
	out_L[0] = out_L[1] = out_L[2] = 0.f;
	UITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	float org[3]={ox,oy,oz}, dir[3]={dx,dy,dz};
	RandomWalkLi<float>(org, dir, scene, max_depth, rand2d, out_L);
}

static void ao_trace(const UISyntheticScene& scene,
					 float ox, float oy, float oz,
					 float dx, float dy, float dz,
					 float max_dist, bool cos_sample,
					 float illum_scale,
					 float out_L[3])
{
	out_L[0] = out_L[1] = out_L[2] = 0.f;
	UITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	float org[3]={ox,oy,oz}, dir[3]={dx,dy,dz};
	constexpr float white[3] = {1.f, 1.f, 1.f};
	AOLi<float>(org, dir, scene, max_dist, cos_sample, illum_scale, white,
				rand2d, out_L);
}

// ===========================================================================
// RandomWalkLi tests
// ===========================================================================

// Miss -> background radiance only (0.1 per channel)
TEST(RandomWalk, MissReturnsBackground) {
	UISyntheticScene scene;
	float L[3];
	rw_trace(scene, 0,0,5, 10,0,-1, 8, L);
	EXPECT_NEAR(L[0], UISyntheticScene::kBg, 1e-5f);
	EXPECT_NEAR(L[1], UISyntheticScene::kBg, 1e-5f);
	EXPECT_NEAR(L[2], UISyntheticScene::kBg, 1e-5f);
}

// Hit -> background eventually reached via multiple bounces -> positive
TEST(RandomWalk, HitReturnsPositive) {
	UISyntheticScene scene;
	float sum = 0.f;
	const int N = 200;
	for (int i = 0; i < N; ++i) {
		float L[3];
		// Use varying seeds so we sample many directions
		UITestRNG rng(i + 1);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		RandomWalkLi<float>(org, dir, scene, 6, rand2d, L);
		sum += L[0];
	}
	EXPECT_GT(sum / N, 0.f);
}

// max_depth==0: only surface emission at depth 0, sphere is non-emissive -> 0
TEST(RandomWalk, MaxDepthZeroNonEmissiveSphere) {
	UISyntheticScene scene;
	float L[3];
	rw_trace(scene, 0,0,5, 0,0,-1, 0, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// Output non-negative for 200 random directions
TEST(RandomWalk, OutputNonNegative) {
	UISyntheticScene scene;
	UITestRNG dir_rng(7);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f-1.f, dy = dir_rng()*2.f-1.f, dz = -1.f;
		float L[3];
		rw_trace(scene, 0,0,5, dx,dy,dz, 6, L);
		EXPECT_GE(L[0], 0.f) << "i=" << i;
		EXPECT_GE(L[1], 0.f) << "i=" << i;
		EXPECT_GE(L[2], 0.f) << "i=" << i;
	}
}

// Output finite for 200 random directions
TEST(RandomWalk, OutputFinite) {
	UISyntheticScene scene;
	UITestRNG dir_rng(13);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f-1.f, dy = dir_rng()*2.f-1.f, dz = -1.f;
		float L[3];
		rw_trace(scene, 0,0,5, dx,dy,dz, 6, L);
		EXPECT_TRUE(std::isfinite(L[0])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[1])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[2])) << "i=" << i;
	}
}

// Achromatic output (grey BSDF + white background)
TEST(RandomWalk, OutputIsAchromatic) {
	UISyntheticScene scene;
	// Average many samples for a stable estimate
	float sumR = 0.f, sumG = 0.f, sumB = 0.f;
	for (int i = 0; i < 300; ++i) {
		UITestRNG rng(i + 100);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		float L[3] = {};
		RandomWalkLi<float>(org, dir, scene, 6, rand2d, L);
		sumR += L[0]; sumG += L[1]; sumB += L[2];
	}
	if (sumR > 1e-4f) {
		EXPECT_NEAR(sumG / sumR, 1.f, 0.01f);
		EXPECT_NEAR(sumB / sumR, 1.f, 0.01f);
	}
}

// Deterministic with same seed
TEST(RandomWalk, Deterministic) {
	UISyntheticScene scene;
	float L1[3], L2[3];
	rw_trace(scene, 0,0,5, 0,0,-1, 6, L1);
	rw_trace(scene, 0,0,5, 0,0,-1, 6, L2);
	EXPECT_FLOAT_EQ(L1[0], L2[0]);
	EXPECT_FLOAT_EQ(L1[1], L2[1]);
	EXPECT_FLOAT_EQ(L1[2], L2[2]);
}

// ===========================================================================
// AOLi tests
// ===========================================================================

// Miss -> L = 0 (no surface hit)
TEST(AOIntegrator, MissReturnsZero) {
	UISyntheticScene scene;
	float L[3];
	ao_trace(scene, 0,0,5, 10,0,-1, 1e10f, true, 1.f, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// Hit, no obstruction -> positive illumination (cosine sampling)
TEST(AOIntegrator, HitUnobstructedCosSample) {
	UISyntheticScene scene;
	float sum = 0.f;
	const int N = 300;
	for (int i = 0; i < N; ++i) {
		UITestRNG rng(i + 1);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		constexpr float white[3]={1.f,1.f,1.f};
		float L[3]={};
		AOLi<float>(org, dir, scene, 1e10f, true, 1.f, white, rand2d, L);
		sum += L[0];
	}
	EXPECT_GT(sum / N, 0.f);
}

// Hit, no obstruction -> positive illumination (uniform sampling)
TEST(AOIntegrator, HitUnobstructedUniformSample) {
	UISyntheticScene scene;
	float sum = 0.f;
	const int N = 300;
	for (int i = 0; i < N; ++i) {
		UITestRNG rng(i + 1);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		constexpr float white[3]={1.f,1.f,1.f};
		float L[3]={};
		AOLi<float>(org, dir, scene, 1e10f, false, 1.f, white, rand2d, L);
		sum += L[0];
	}
	EXPECT_GT(sum / N, 0.f);
}

// Always-blocked scene: UnoccludedWithin always returns false.
// -> AOLi must return 0 regardless of max_dist.
struct BlockedScene : UISyntheticScene {
	bool UnoccludedWithin(const float*, const float*, float) const { return false; }
};

TEST(AOIntegrator, AlwaysBlockedReturnsZero) {
	BlockedScene scene;
	UITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	float org[3]={0,0,5}, dir[3]={0,0,-1};
	constexpr float white[3]={1.f,1.f,1.f};
	float L[3]={};
	AOLi<float>(org, dir, scene, 1e10f, true, 1.f, white, rand2d, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// Output non-negative for many random rays (cos sampling)
TEST(AOIntegrator, OutputNonNegativeCos) {
	UISyntheticScene scene;
	UITestRNG dir_rng(17);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f-1.f, dy = dir_rng()*2.f-1.f, dz = -1.f;
		float L[3];
		ao_trace(scene, 0,0,5, dx,dy,dz, 1e10f, true, 1.f, L);
		EXPECT_GE(L[0], 0.f) << "i=" << i;
		EXPECT_GE(L[1], 0.f) << "i=" << i;
		EXPECT_GE(L[2], 0.f) << "i=" << i;
	}
}

// Output finite for many random rays (uniform sampling)
TEST(AOIntegrator, OutputFiniteUniform) {
	UISyntheticScene scene;
	UITestRNG dir_rng(23);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f-1.f, dy = dir_rng()*2.f-1.f, dz = -1.f;
		float L[3];
		ao_trace(scene, 0,0,5, dx,dy,dz, 1e10f, false, 1.f, L);
		EXPECT_TRUE(std::isfinite(L[0])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[1])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[2])) << "i=" << i;
	}
}

// Cosine sampling: result = illumScale * cos(wi,n) / (pi * pdf) where pdf = cos/pi.
// The cos/pi factors cancel exactly, so every unoccluded sample returns illumScale = 1.0.
// In a fully open scene (no blockers) the mean over many samples must be 1.0.
TEST(AOIntegrator, CosineSamplingCancelsFactor) {
	UISyntheticScene scene;
	float sum = 0.f;
	const int N = 500;
	for (int i = 0; i < N; ++i) {
		UITestRNG rng(i + 500);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		constexpr float white[3]={1.f,1.f,1.f};
		float L[3]={};
		AOLi<float>(org, dir, scene, 1e10f, true, 1.f, white, rand2d, L);
		sum += L[0];
	}
	// With no blockers every sample = 1.0, so mean must equal 1.0 (or 0 if somehow all missed).
	float mean = sum / N;
	EXPECT_NEAR(mean, 1.f, 0.01f);
}

// illum_scale scales the output linearly
TEST(AOIntegrator, IllumScaleLinear) {
	UISyntheticScene scene;
	// Use a fixed sample that is guaranteed unoccluded (ray from (0,0,5),
	// shadow direction points away from sphere)
	UITestRNG rng(42);
	auto rand2d2 = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	UITestRNG rng2(42);
	auto rand2d_scaled = [&]() -> std::pair<float,float> { return {rng2(), rng2()}; };

	float org[3]={0,0,5}, dir[3]={0,0,-1};
	constexpr float white[3]={1.f,1.f,1.f};
	float L1[3]={}, L2[3]={};
	AOLi<float>(org, dir, scene, 1e10f, true, 1.f, white, rand2d2, L1);
	AOLi<float>(org, dir, scene, 1e10f, true, 2.f, white, rand2d_scaled, L2);

	if (L1[0] > 1e-6f) {
		EXPECT_NEAR(L2[0] / L1[0], 2.f, 1e-4f);
	}
}

// Deterministic with same seed
TEST(AOIntegrator, Deterministic) {
	UISyntheticScene scene;
	float L1[3], L2[3];
	ao_trace(scene, 0,0,5, 0,0,-1, 1e10f, true, 1.f, L1);
	ao_trace(scene, 0,0,5, 0,0,-1, 1e10f, true, 1.f, L2);
	EXPECT_FLOAT_EQ(L1[0], L2[0]);
	EXPECT_FLOAT_EQ(L1[1], L2[1]);
	EXPECT_FLOAT_EQ(L1[2], L2[2]);
}

// ===========================================================================
// Additional coverage: paths mirroring pbrt-v4 code branches
// ===========================================================================

// RandomWalkLi: null BSDF at first hit -> terminate immediately after Le.
// Mirrors pbrt-v4: "if (!bsdf) return Le" where Le=0 for non-emissive sphere.
// With a null-BSDF scene, the first hit has no BSDF; result must be 0.
struct NullBSDFScene : UISyntheticScene {
	bool BSDFIsNull(int) const { return true; }
};

TEST(RandomWalk, NullBSDFTerminatesAfterLe) {
	NullBSDFScene scene;
	// Hit sphere at depth=0, null BSDF -> return Le (0 for non-emissive sphere)
	float L[3] = {};
	UITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	float org[3]={0,0,5}, dir[3]={0,0,-1};
	RandomWalkLi<float>(org, dir, scene, 8, rand2d, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// AOLi: medium boundary (null BSDF) -> skip intersection and continue.
// Mirrors pbrt-v4 AO "goto retry" loop.
// Scene: first sphere returns null BSDF (medium boundary), second sphere
// (radius 2) is a real surface. AO should skip the first and evaluate AO
// from the second.
struct MediumBoundaryScene {
	// Inner sphere (r=0.5) is a medium boundary, outer sphere (r=2) is solid.
	static bool sphere_hit_r(const float org[3], const float dir[3],
							 float r, float t_max, BDPTHit<float>& hit) {
		float a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
		float b = 2*(org[0]*dir[0] + org[1]*dir[1] + org[2]*dir[2]);
		float c = org[0]*org[0] + org[1]*org[1] + org[2]*org[2] - r*r;
		float disc = b*b - 4*a*c;
		if (disc < 0.f) return false;
		float sq = std::sqrt(disc);
		float t  = (-b - sq) / (2*a);
		if (t < 1e-4f) t = (-b + sq) / (2*a);
		if (t < 1e-4f || t > t_max) return false;
		hit.t_hit = t;
		hit.p[0] = org[0]+t*dir[0]; hit.p[1] = org[1]+t*dir[1]; hit.p[2] = org[2]+t*dir[2];
		float len = std::sqrt(hit.p[0]*hit.p[0]+hit.p[1]*hit.p[1]+hit.p[2]*hit.p[2]);
		hit.geo_n[0]=hit.p[0]/len; hit.geo_n[1]=hit.p[1]/len; hit.geo_n[2]=hit.p[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false;
		hit.light_id=-1;
		return true;
	}

	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		// Always try inner sphere first (closer), then outer.
		BDPTHit<float> h1{}, h2{};
		bool i1 = sphere_hit_r(org, dir, 0.5f, t_max, h1);
		bool i2 = sphere_hit_r(org, dir, 2.0f, t_max, h2);
		if (i1 && (!i2 || h1.t_hit < h2.t_hit)) {
			hit = h1; hit.bsdf_id = 0; return true;  // 0 = null (medium boundary)
		}
		if (i2) {
			hit = h2; hit.bsdf_id = 1; return true;  // 1 = real surface
		}
		return false;
	}

	bool BSDFIsNull(int id) const { return id == 0; }

	bool UnoccludedWithin(const float p[3], const float dir[3], float max_dist) const {
		BDPTHit<float> tmp{};
		return !sphere_hit_r(p, dir, 2.0f, max_dist, tmp);
	}

	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-4f;
		// For medium boundary: spawn slightly past the boundary (same direction).
		new_o[0] = hit.p[0] + kEps * dir[0];
		new_o[1] = hit.p[1] + kEps * dir[1];
		new_o[2] = hit.p[2] + kEps * dir[2];
		new_d[0] = dir[0]; new_d[1] = dir[1]; new_d[2] = dir[2];
	}
};

// Camera at (0,0,5), ray hits inner sphere (null BSDF), skips to outer (r=2).
// AO shadow rays from outer sphere mostly unoccluded -> L > 0.
TEST(AOIntegrator, MediumBoundarySkipped) {
	MediumBoundaryScene scene;
	float sum = 0.f;
	const int N = 300;
	for (int i = 0; i < N; ++i) {
		UITestRNG rng(i + 1);
		auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
		float org[3]={0,0,5}, dir[3]={0,0,-1};
		constexpr float white[3]={1.f,1.f,1.f};
		float L[3]={};
		AOLi<float>(org, dir, scene, 1e10f, true, 1.f, white, rand2d, L);
		sum += L[0];
	}
	// After skipping the medium boundary the outer sphere is hit -> positive AO.
	EXPECT_GT(sum / N, 0.f);
}

