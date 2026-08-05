// path_integrator_tests.cpp -- unit tests for src/shared/path_integrator.h
//
// Tests PathLi<T,Scene> and PathSampleLd<T,Scene> against expected behaviour
// derived from pbrt-v4's PathIntegrator.
//
// Scene setup:
//   - Lambertian unit sphere centred at origin (albedo 0.8)
//   - Point light at (0, 0, 3)  -- in front of the sphere (same side as camera)
//   - Camera at (0, 0, 5) looking along -z
//   - Constant background emission 0.1 per channel

#include "gtest/gtest.h"
#include "../../src/shared/path_integrator.h"

#include <cmath>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// Minimal RNG
// ---------------------------------------------------------------------------
struct PITestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{ 0.f, 1.f };
	explicit PITestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Synthetic scene
// ---------------------------------------------------------------------------
struct PISyntheticScene {
	PITestRNG rng;

	// ---- Geometry -----------------------------------------------------------
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
		d[0]/=len; d[1]/=len; d[2]/=len;
		BDPTHit<float> tmp{};
		return !sphere_intersect(p0, d, len - 1e-3f, tmp);
	}

	// ---- Lights -------------------------------------------------------------
	// Point light at (0, 0, 3) -- in the +z hemisphere of the front face (0,0,1)
	static constexpr float kLightPos[3] = {0.f, 0.f, 3.f};
	static constexpr float kBg          = 0.1f;

	void SurfaceLe(const BDPTHit<float>&, const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;
	}

	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = kBg;
	}

	// p_l for the background: single infinite light, PMF=1, PDF_Li = 1/(4*pi).
	float InfiniteLightPdf(const float*, const float*, const float*) const {
		return 1.f / (4.f * 3.14159265358979f);
	}

	// No area lights in this scene.
	float AreaLightPdf(const BDPTHit<float>&, const float*, const float*, const float*) const {
		return 0.f;
	}

	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		ls.p_light[0] = kLightPos[0]; ls.p_light[1] = kLightPos[1]; ls.p_light[2] = kLightPos[2];
		ls.n_light[0] = ls.n_light[1] = ls.n_light[2] = 0.f;
		float d[3] = {kLightPos[0]-ref_p[0], kLightPos[1]-ref_p[1], kLightPos[2]-ref_p[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0] = d[0]/len; ls.wi[1] = d[1]/len; ls.wi[2] = d[2]/len;
		ls.L[0] = ls.L[1] = ls.L[2] = 1.f / (len*len);
		ls.pdf      = 1.f;   // only one light, selection p=1, point light p_sample=1
		ls.is_delta = true;  // point light -> delta -> no MIS weighting
		ls.light_id = 0;
		return true;
	}

	// ---- BSDF: Lambertian (albedo 0.8) -------------------------------------
	static constexpr float kAlbedoOverPi = 0.8f / 3.14159265358979f;
	static constexpr float kPi           = 3.14159265358979f;

	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float cos_i = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
		if (cos_i <= 0.f) { out[0] = out[1] = out[2] = 0.f; return; }
		// f * |cos| = (albedo/pi) * cos_i  (already multiplied)
		out[0] = out[1] = out[2] = kAlbedoOverPi * cos_i;
	}

	// BSDFPdf: cosine-hemisphere PDF = cos_i / pi
	float BSDFPdf(int, const float*, const float wi[3],
				  const float n[3]) const {
		float cos_i = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
		return (cos_i > 0.f) ? cos_i / kPi : 0.f;
	}

	bool BSDFSampleF(int, const float*, const float n[3],
					 float u1, float u2,
					 float new_dir[3], float f_val[3],
					 float& pdf, bool& is_specular,
					 bool& is_transmission,
					 bool& pdf_is_proportional, float& eta) const {
		// Cosine-hemisphere sampling
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(std::max(0.f, 1.f - u2));
		float phi   = 2.f * kPi * u1;
		float lx = sin_t*std::cos(phi), ly = sin_t*std::sin(phi), lz = cos_t;

		// ONB
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

		// f_val already includes |cos|
		f_val[0] = f_val[1] = f_val[2] = kAlbedoOverPi * cos_t;
		pdf                  = cos_t / kPi;
		is_specular          = false;
		is_transmission      = false;  // purely reflective Lambertian
		pdf_is_proportional  = false;
		eta                  = 1.f;
		return pdf > 0.f;
	}

	bool BSDFIsReflectiveAndTransmissive(int) const { return false; }
	bool BSDFIsNonSpecular(int) const { return true; }  // Lambertian is diffuse
	void BSDFRegularize(int) const {}  // no-op stub; regularize=false by default

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
static void trace(const PISyntheticScene& scene,
				  float ox, float oy, float oz,
				  float dx, float dy, float dz,
				  int max_depth, float rr_threshold,
				  float out_L[3])
{
	out_L[0] = out_L[1] = out_L[2] = 0.f;
	PITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> {
		return { rng(), rng() };
	};
	auto rand1d = [&]() -> float { return rng(); };
	float org[3] = { ox, oy, oz };
	float dir[3] = { dx, dy, dz };
	PathLi<float>(org, dir, scene, max_depth, rr_threshold, rand2d, rand1d, out_L);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A ray missing the sphere returns background radiance (0.1 per channel).
TEST(PathIntegrator, MissReturnsBackground) {
	PISyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 10,0,-1, 8, 1.f, L);
	// depth==0, specular_bounce==false: still adds Le unweighted at depth==0
	EXPECT_NEAR(L[0], PISyntheticScene::kBg, 1e-5f);
	EXPECT_NEAR(L[1], PISyntheticScene::kBg, 1e-5f);
	EXPECT_NEAR(L[2], PISyntheticScene::kBg, 1e-5f);
}

// A ray hitting the sphere towards the lit face must return positive radiance.
TEST(PathIntegrator, HitReturnsPositive) {
	PISyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 0,0,-1, 8, 1.f, L);
	EXPECT_GT(L[0], 0.f);
	EXPECT_GT(L[1], 0.f);
	EXPECT_GT(L[2], 0.f);
}

// max_depth == 0: only emission at depth==0 contributes.
// Sphere is non-emissive => L should be zero.
TEST(PathIntegrator, MaxDepthZeroNoEmission) {
	PISyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 0,0,-1, 0, 1.f, L);
	EXPECT_NEAR(L[0], 0.f, 1e-6f);
	EXPECT_NEAR(L[1], 0.f, 1e-6f);
	EXPECT_NEAR(L[2], 0.f, 1e-6f);
}

// Output must be non-negative for 200 random rays.
TEST(PathIntegrator, OutputNonNegative) {
	PISyntheticScene scene;
	PITestRNG dir_rng(7);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f - 1.f;
		float dy = dir_rng()*2.f - 1.f;
		float dz = -1.f;
		float L[3];
		trace(scene, 0,0,5, dx,dy,dz, 8, 1.f, L);
		EXPECT_GE(L[0], 0.f) << "i=" << i;
		EXPECT_GE(L[1], 0.f) << "i=" << i;
		EXPECT_GE(L[2], 0.f) << "i=" << i;
	}
}

// Output must be finite for 200 random rays.
TEST(PathIntegrator, OutputFinite) {
	PISyntheticScene scene;
	PITestRNG dir_rng(13);
	for (int i = 0; i < 200; ++i) {
		float dx = dir_rng()*2.f - 1.f;
		float dy = dir_rng()*2.f - 1.f;
		float dz = -1.f;
		float L[3];
		trace(scene, 0,0,5, dx,dy,dz, 8, 1.f, L);
		EXPECT_TRUE(std::isfinite(L[0])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[1])) << "i=" << i;
		EXPECT_TRUE(std::isfinite(L[2])) << "i=" << i;
	}
}

// PathLi must produce higher average radiance than SimplePathLi on the same
// scene because it uses MIS for NEE (lower variance, correct weights).
// We just verify the mean is positive and within a reasonable range.
TEST(PathIntegrator, MeanRadiancePositiveAndReasonable) {
	PISyntheticScene scene;
	const int N = 300;
	float sum = 0.f;
	for (int i = 0; i < N; ++i) {
		float L[3];
		trace(scene, 0,0,5, 0,0,-1, 8, 1.f, L);
		sum += L[0];
	}
	float mean = sum / N;
	EXPECT_GT(mean, 0.f);
	EXPECT_LT(mean, 10.f);  // sanity upper bound
}

// Output is achromatic (equal R/G/B) because the BSDF is grey and the
// light is white.
TEST(PathIntegrator, OutputIsAchromatic) {
	PISyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 0,0,-1, 8, 1.f, L);
	if (L[0] > 1e-6f) {
		EXPECT_NEAR(L[1]/L[0], 1.f, 1e-5f);
		EXPECT_NEAR(L[2]/L[0], 1.f, 1e-5f);
	}
}

// PathSampleLd alone: direct contribution for a hitting ray must be positive.
TEST(PathIntegrator, SampleLdPositive) {
	PISyntheticScene scene;
	// Shoot a ray to get a surface hit
	float org[3]={0,0,5}, dir[3]={0,0,-1};
	BDPTHit<float> hit{};
	ASSERT_TRUE(scene.Intersect(org, dir, 1e30f, hit));

	float wo[3] = { -dir[0], -dir[1], -dir[2] };
	PITestRNG rng(42);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	auto rand1d = [&]() -> float { return rng(); };

	float Ld[3];
	PathSampleLd<float>(hit, wo, scene, rand2d, rand1d, Ld);
	EXPECT_GT(Ld[0], 0.f);
	EXPECT_GT(Ld[1], 0.f);
	EXPECT_GT(Ld[2], 0.f);
}

// Russian roulette: with rr_threshold=0 (disabled), long paths must still
// produce finite results.
TEST(PathIntegrator, RussianRouletteDisabledFinite) {
	PISyntheticScene scene;
	float L[3];
	trace(scene, 0,0,5, 0,0,-1, 20, 0.f, L);  // rr_threshold=0 disables RR
	EXPECT_TRUE(std::isfinite(L[0]));
	EXPECT_TRUE(std::isfinite(L[1]));
	EXPECT_TRUE(std::isfinite(L[2]));
}

// Deterministic: same seed -> same result.
TEST(PathIntegrator, DeterministicWithSameSeed) {
	PISyntheticScene scene;
	float L1[3], L2[3];
	trace(scene, 0,0,5, 0,0,-1, 8, 1.f, L1);
	trace(scene, 0,0,5, 0,0,-1, 8, 1.f, L2);
	EXPECT_FLOAT_EQ(L1[0], L2[0]);
	EXPECT_FLOAT_EQ(L1[1], L2[1]);
	EXPECT_FLOAT_EQ(L1[2], L2[2]);
}

// MIS weight check: with a delta light (point light), the NEE weight is 1
// (no BSDF PDF blending), so the direct contribution should exactly equal
// f * L / pdf  =  kAlbedoOverPi * cos_i * L / 1.0
TEST(PathIntegrator, DeltaLightNoMISWeight) {
	PISyntheticScene scene;
	// Hit the sphere front face at (0, 0, 1)
	float org[3]={0,0,5}, dir[3]={0,0,-1};
	BDPTHit<float> hit{};
	ASSERT_TRUE(scene.Intersect(org, dir, 1e30f, hit));

	float wo[3] = { -dir[0], -dir[1], -dir[2] };

	// Build the expected NEE result manually:
	// wi points from (0,0,1) to kLightPos=(0,0,3), so wi=(0,0,1)
	// f*|cos| = kAlbedoOverPi * dot(wi, n) = kAlbedoOverPi * 1.0
	// L = 1 / dist^2 = 1/4
	// pdf = 1, w_l = 1 (delta light)
	// Ld = f*|cos| * L / pdf = kAlbedoOverPi * 0.25

	float expected = PISyntheticScene::kAlbedoOverPi * 0.25f;

	PITestRNG rng(0);
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	auto rand1d = [&]() -> float { return rng(); };
	float Ld[3];
	PathSampleLd<float>(hit, wo, scene, rand2d, rand1d, Ld);

	EXPECT_NEAR(Ld[0], expected, 1e-5f);
	EXPECT_NEAR(Ld[1], expected, 1e-5f);
	EXPECT_NEAR(Ld[2], expected, 1e-5f);
}
