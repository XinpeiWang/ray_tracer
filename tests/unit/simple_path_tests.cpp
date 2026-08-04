// simple_path_tests.cpp -- unit tests for src/shared/simple_path.h
// Validates SimplePathLi<T,Scene> against expected pbrt-v4 behaviour.
//
// Uses the same synthetic unit-sphere scene as mlt_tests / bdpt_tests:
//   - Lambertian sphere of radius 1 centred at origin (albedo 0.8)
//   - Point light at (0, 3, 0)
//   - Camera at (0, 0, 5) looking towards -z

#include "gtest/gtest.h"
#include "../../src/shared/simple_path.h"

#include <cmath>
#include <random>

// ---------------------------------------------------------------------------
// Minimal RNG helper
// ---------------------------------------------------------------------------
struct SPTestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	explicit SPTestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Synthetic scene: unit sphere + point light above
// ---------------------------------------------------------------------------
struct SPSyntheticScene {
	SPTestRNG rng;

	// ---- Geometry ----------------------------------------------------------
	static bool sphere_intersect(const float org[3], const float dir[3],
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
		hit.geo_n[0] = hit.p[0]/len; hit.geo_n[1] = hit.p[1]/len; hit.geo_n[2] = hit.p[2]/len;
		hit.shading_n[0] = hit.geo_n[0]; hit.shading_n[1] = hit.geo_n[1]; hit.shading_n[2] = hit.geo_n[2];
		hit.wo[0] = -dir[0]; hit.wo[1] = -dir[1]; hit.wo[2] = -dir[2];
		hit.uv[0] = hit.uv[1] = 0.f;
		hit.area_Le[0] = hit.area_Le[1] = hit.area_Le[2] = 0.f;
		hit.is_medium_boundary = false;
		hit.is_delta_bsdf      = false;
		hit.bsdf_id  = 0;
		hit.light_id = -1;
		return true;
	}

	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		return sphere_intersect(org, dir, t_max, hit);
	}

	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float d[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if (len < 1e-6f) return true;
		d[0] /= len; d[1] /= len; d[2] /= len;
		BDPTHit<float> tmp{};
		return !sphere_intersect(p0, d, len - 1e-3f, tmp);
	}

	// ---- Lights ------------------------------------------------------------
	// Light is placed between camera (0,0,5) and sphere surface (0,0,1) so
	// that the lit face has cos_i = dot(wi, n) = 1 (always above horizon).
	static constexpr float kLightPos[3] = {0.f, 0.f, 3.f};

	void SurfaceLe(const BDPTHit<float>&, const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;  // sphere is not emissive
	}

	// Small constant background emission so that paths that escape the sphere
	// can accumulate energy even without NEE (tests the InfiniteLightLe path).
	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.1f;
	}

	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		ls.p_light[0] = kLightPos[0]; ls.p_light[1] = kLightPos[1]; ls.p_light[2] = kLightPos[2];
		ls.n_light[0] = ls.n_light[1] = ls.n_light[2] = 0.f;
		float d[3] = {kLightPos[0]-ref_p[0], kLightPos[1]-ref_p[1], kLightPos[2]-ref_p[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0] = d[0]/len; ls.wi[1] = d[1]/len; ls.wi[2] = d[2]/len;
		ls.L[0] = ls.L[1] = ls.L[2] = 1.f / (len*len);
		ls.pdf = 1.f;
		ls.is_delta = true;
		ls.light_id = 0;
		return true;
	}

	// ---- BSDF: Lambertian (albedo 0.8) ------------------------------------
	static constexpr float kAlbedoOverPi = 0.8f / 3.14159265358979f;

	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float cos_i = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
		if (cos_i <= 0.f) { out[0] = out[1] = out[2] = 0.f; return; }
		out[0] = out[1] = out[2] = kAlbedoOverPi;
	}

	bool BSDFSampleF(int, const float*, const float n[3],
					 float u1, float u2,
					 float new_dir[3], float f_val[3],
					 float& pdf, bool& is_specular) const {
		// Cosine-hemisphere sampling
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(std::max(0.f, 1.f - u2));
		float phi   = 2.f * 3.14159265358979f * u1;
		float lx = sin_t*std::cos(phi), ly = sin_t*std::sin(phi), lz = cos_t;

		// Build ONB
		float tx, ty, tz;
		if (std::fabs(n[0]) > 0.9f) { tx=0.f; ty=1.f; tz=0.f; }
		else                         { tx=1.f; ty=0.f; tz=0.f; }
		float cx=ty*n[2]-tz*n[1], cy=tz*n[0]-tx*n[2], cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl; ty=cy/cl; tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty, by=n[2]*tx-n[0]*tz, bz=n[0]*ty-n[1]*tx;

		new_dir[0] = lx*tx + ly*bx + lz*n[0];
		new_dir[1] = lx*ty + ly*by + lz*n[1];
		new_dir[2] = lx*tz + ly*bz + lz*n[2];

		f_val[0] = f_val[1] = f_val[2] = kAlbedoOverPi;
		pdf         = cos_t / 3.14159265358979f;
		is_specular = false;
		return pdf > 0.f;
	}

	bool BSDFIsReflectiveAndTransmissive(int) const { return false; }

	// ---- Ray spawning -------------------------------------------------------
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
// Helper: shoot one camera ray and return L
// ---------------------------------------------------------------------------
static void trace(const SPSyntheticScene& scene,
				  float ox, float oy, float oz,
				  float dx, float dy, float dz,
				  int max_depth, bool sample_lights, bool sample_bsdf,
				  float L[3])
{
	L[0] = L[1] = L[2] = 0.f;
	float org[3] = {ox, oy, oz};
	float dir[3] = {dx, dy, dz};

	auto rand2d = [&]() -> std::pair<float,float> {
		return {scene.rng(), scene.rng()};
	};
	auto rand1d = [&]() -> float { return scene.rng(); };

	SimplePathLi<float>(org, dir, scene, max_depth,
						sample_lights, sample_bsdf,
						rand2d, rand1d, L);
}

// ============================================================================
// Tests
// ============================================================================

// ---------------------------------------------------------------------------
// A ray that misses the sphere returns the background radiance (0.1 per channel
// from InfiniteLightLe), not zero.
// ---------------------------------------------------------------------------
TEST(SimplePath, MissingRayReturnsBackground) {
	SPSyntheticScene scene;
	// Ray going far to the side — guaranteed to miss the unit sphere
	float L[3];
	trace(scene, 0,0,5, 10,0,-1, 8, true, true, L);
	// With sample_lights=true the specular_bounce flag starts true, so the
	// infinite-light contribution IS added (mirrors pbrt-v4 behaviour).
	EXPECT_NEAR(L[0], 0.1f, 1e-5f);
	EXPECT_NEAR(L[1], 0.1f, 1e-5f);
	EXPECT_NEAR(L[2], 0.1f, 1e-5f);
}

// ---------------------------------------------------------------------------
// A ray that hits the sphere returns positive radiance (scene is illuminated).
// ---------------------------------------------------------------------------
TEST(SimplePath, HittingRayReturnsPositive) {
	SPSyntheticScene scene;
	float L[3];
	// Ray from camera at (0,0,5) towards origin (hits sphere front face)
	trace(scene, 0,0,5, 0,0,-1, 8, true, true, L);
	bool any_positive = L[0] > 0.f || L[1] > 0.f || L[2] > 0.f;
	EXPECT_TRUE(any_positive);
}

// ---------------------------------------------------------------------------
// With max_depth=0 the path terminates immediately (no emission on sphere),
// so the result is zero.
// ---------------------------------------------------------------------------
TEST(SimplePath, MaxDepthZeroReturnsZero) {
	SPSyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 0,0,-1, 0, true, true, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Output is always non-negative (no fireflies / negative radiance).
// ---------------------------------------------------------------------------
TEST(SimplePath, OutputNonNegative) {
	SPSyntheticScene scene;
	for (int i = 0; i < 200; ++i) {
		float dx = scene.rng() * 2.f - 1.f;
		float dy = scene.rng() * 2.f - 1.f;
		float dz = -1.f;
		float L[3];
		trace(scene, 0,0,5, dx,dy,dz, 8, true, true, L);
		EXPECT_GE(L[0], 0.f) << "i=" << i;
		EXPECT_GE(L[1], 0.f) << "i=" << i;
		EXPECT_GE(L[2], 0.f) << "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// Output is always finite (no Inf/NaN).
// ---------------------------------------------------------------------------
TEST(SimplePath, OutputFinite) {
	SPSyntheticScene scene;
	for (int i = 0; i < 200; ++i) {
		float dx = scene.rng() * 2.f - 1.f;
		float dy = scene.rng() * 2.f - 1.f;
		float dz = -1.f;
		float L[3];
		trace(scene, 0,0,5, dx,dy,dz, 8, true, true, L);
		EXPECT_TRUE(std::isfinite(L[0])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[1])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[2])) << "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// sample_lights=false, sample_bsdf=true: paths that scatter out of the sphere
// pick up the background emission (InfiniteLightLe returns 0.1). Average over
// many paths should be positive.
// ---------------------------------------------------------------------------
TEST(SimplePath, NoLightSamplingBackgroundContribution) {
	SPSyntheticScene scene;
	const int N = 500;
	float sum = 0.f;
	for (int i = 0; i < N; ++i) {
		float L[3];
		trace(scene, 0,0,5, 0,0,-1, 8, /*sample_lights*/false, /*sample_bsdf*/true, L);
		sum += L[0];
	}
	// Paths that escape the sphere accumulate the background (0.1); mean > 0.
	EXPECT_GT(sum / N, 0.f);
}

// ---------------------------------------------------------------------------
// sample_bsdf=false (uniform hemisphere fallback) + NEE enabled: the point
// light at (0,0,3) is in the upper hemisphere of the hit point so NEE should
// contribute. Average over many paths must be positive.
// ---------------------------------------------------------------------------
TEST(SimplePath, UniformSamplingFallbackPositive) {
	SPSyntheticScene scene;
	const int N = 300;
	float sum = 0.f;
	for (int i = 0; i < N; ++i) {
		float L[3];
		trace(scene, 0,0,5, 0,0,-1, 8, /*sample_lights*/true, /*sample_bsdf*/false, L);
		sum += L[0];
	}
	EXPECT_GT(sum / N, 0.f);
}

// ---------------------------------------------------------------------------
// RGB channels are equal (Lambertian white BSDF + white light = achromatic).
// ---------------------------------------------------------------------------
TEST(SimplePath, OutputIsAchromatic) {
	SPSyntheticScene scene;
	const int N = 500;
	float sumR = 0.f, sumG = 0.f, sumB = 0.f;
	for (int i = 0; i < N; ++i) {
		float L[3];
		trace(scene, 0,0,5, 0,0,-1, 8, true, true, L);
		sumR += L[0]; sumG += L[1]; sumB += L[2];
	}
	float meanR = sumR / N, meanG = sumG / N, meanB = sumB / N;
	EXPECT_NEAR(meanR, meanG, 0.05f) << "R/G not equal";
	EXPECT_NEAR(meanG, meanB, 0.05f) << "G/B not equal";
}

// ---------------------------------------------------------------------------
// specular_bounce flag: with sample_lights=true, a depth-1 path should still
// add emission (emission is always added on specular bounces, and bounce 0 is
// always specular). Non-emissive sphere → zero direct emission, but positive
// from NEE to light.
// ---------------------------------------------------------------------------
TEST(SimplePath, DepthOneSampleLightsPositive) {
	SPSyntheticScene scene;
	float L[3];
	// max_depth=1: one surface hit, perform NEE, then terminate
	trace(scene, 0,0,5, 0,0,-1, 1, true, true, L);
	bool any_positive = L[0] > 0.f || L[1] > 0.f || L[2] > 0.f;
	EXPECT_TRUE(any_positive);
}

// ---------------------------------------------------------------------------
// Accumulation: calling SimplePathLi twice and summing gives 2× the result
// of a single call with the same RNG seed (determinism check).
// ---------------------------------------------------------------------------
TEST(SimplePath, DeterministicWithSameSeed) {
	float L1[3] = {}, L2[3] = {};

	{
		SPSyntheticScene scene;   // same seed 42
		float org[3] = {0,0,5}, dir[3] = {0,0,-1};
		auto r2 = [&]{ return std::make_pair(scene.rng(), scene.rng()); };
		auto r1 = [&]{ return scene.rng(); };
		SimplePathLi<float>(org, dir, scene, 8, true, true, r2, r1, L1);
	}
	{
		SPSyntheticScene scene;   // same seed 42
		float org[3] = {0,0,5}, dir[3] = {0,0,-1};
		auto r2 = [&]{ return std::make_pair(scene.rng(), scene.rng()); };
		auto r1 = [&]{ return scene.rng(); };
		SimplePathLi<float>(org, dir, scene, 8, true, true, r2, r1, L2);
	}

	EXPECT_FLOAT_EQ(L1[0], L2[0]);
	EXPECT_FLOAT_EQ(L1[1], L2[1]);
	EXPECT_FLOAT_EQ(L1[2], L2[2]);
}
