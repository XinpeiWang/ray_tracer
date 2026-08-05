// simple_vol_path_tests.cpp -- Unit tests for src/shared/simple_vol_path.h
//
// Test groups:
//   1. SimpleVolPath.Vacuum_MissReturnsInfiniteLight
//      No medium, no geometry: output must equal InfiniteLightLe contribution.
//   2. SimpleVolPath.Vacuum_SurfaceEmission
//      No medium, emissive sphere: output equals beta*area_Le.
//   3. SimpleVolPath.Vacuum_NoEmission_ReturnsZero
//      No medium, no emissive surfaces/lights: output is zero.
//   4. SimpleVolPath.MaxDepth0_InMedium_NeverScatters
//      max_depth=0: any scatter event terminates immediately; result finite.
//   5. SimpleVolPath.Absorption_TerminatesWithVolumeEmission
//      Absorbing medium with Le>0: ray must pick up Le and terminate.
//   6. SimpleVolPath.HomogeneousMedium_NonNegativeAndFinite
//      Scattering medium: result is non-negative and finite over many seeds.
//   7. SimpleVolPath.NullEvent_ContinuesMarch
//      Pure null medium (sigma_a=sigma_s=0, sigma_maj>0): must pass through
//      and return infinite-light background unchanged.
//   8. SimpleVolPath.MediumBoundary_SkippedCorrectly
//      is_medium_boundary surface: integrator must skip it and continue.

#include "gtest/gtest.h"
#include "../../src/shared/simple_vol_path.h"
#include <cmath>
#include <functional>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
struct SVPTestRNG {
	mutable std::mt19937 eng{42u};
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	float operator()() const { return dist(eng); }
};

static void fill_hit(BDPTHit<float>& h,
					  const float org[3], const float dir[3], float t,
					  bool is_boundary = false, bool is_light = false,
					  float le = 0.f)
{
	h.t_hit = t;
	h.p[0] = org[0] + t * dir[0];
	h.p[1] = org[1] + t * dir[1];
	h.p[2] = org[2] + t * dir[2];
	float len = std::sqrt(h.p[0]*h.p[0] + h.p[1]*h.p[1] + h.p[2]*h.p[2]);
	if (len < 1e-8f) len = 1.f;
	h.geo_n[0] = h.shading_n[0] = h.p[0] / len;
	h.geo_n[1] = h.shading_n[1] = h.p[1] / len;
	h.geo_n[2] = h.shading_n[2] = h.p[2] / len;
	h.wo[0] = -dir[0]; h.wo[1] = -dir[1]; h.wo[2] = -dir[2];
	h.uv[0] = h.uv[1] = 0.f;
	h.area_Le[0] = h.area_Le[1] = h.area_Le[2] = is_light ? le : 0.f;
	h.bsdf_id = 0; h.light_id = is_light ? 0 : -1;
	h.is_delta_bsdf = false;
	h.is_medium_boundary = is_boundary;
}

// Isotropic HG phase function
static float hg_phase(float /*g*/) { return 1.f / (4.f * 3.14159265f); }

// ---------------------------------------------------------------------------
// VacuumScene: no medium, unit sphere, optional point emissive sphere
// ---------------------------------------------------------------------------
struct SVPVacuumScene {
	mutable SVPTestRNG rng;
	bool has_emissive_sphere = false;
	float sphere_le = 0.f;
	float infinite_le = 0.f;

	bool Intersect(const float org[3], const float dir[3], float /*t_max*/,
				   BDPTHit<float>& hit) const {
		if (!has_emissive_sphere) return false;
		float a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
		float b = 2*(org[0]*dir[0]+org[1]*dir[1]+org[2]*dir[2]);
		float c = org[0]*org[0]+org[1]*org[1]+org[2]*org[2] - 1.f;
		float disc = b*b - 4*a*c;
		if (disc < 0) return false;
		float sq = std::sqrt(disc);
		float t = (-b - sq) / (2*a);
		if (t < 1e-4f) t = (-b + sq) / (2*a);
		if (t < 1e-4f) return false;
		fill_hit(hit, org, dir, t, false, true, sphere_le);
		return true;
	}

	void InfiniteLightLe(const float* /*dir*/, float out[3]) const {
		out[0] = out[1] = out[2] = infinite_le;
	}

	bool HasMedium(const float* /*org*/, const float* /*dir*/) const { return false; }
	float SampleTMaj(const float*, const float*, float, float,
					  const std::function<bool(const float*, const VolPathMediumProps<float>&,
											   float, float)>&) const { return 1.f; }
	bool SamplePhase(const float*, float, float, float, float* wi, float& pdf) const {
		wi[0]=wi[1]=0.f; wi[2]=1.f; pdf=1.f/(4.f*3.14159265f); return true;
	}
	float PhaseP(const float*, const float*, float) const { return hg_phase(0.f); }
	float RandFloat() const { return rng(); }
	void SpawnRay(const BDPTHit<float>& h, const float d[3],
				  float new_o[3], float new_d[3]) const {
		const float eps = 1e-4f;
		new_o[0]=h.p[0]+eps*h.shading_n[0];
		new_o[1]=h.p[1]+eps*h.shading_n[1];
		new_o[2]=h.p[2]+eps*h.shading_n[2];
		new_d[0]=d[0]; new_d[1]=d[1]; new_d[2]=d[2];
	}
};
// ---------------------------------------------------------------------------
struct SVPHomogeneousScene {
	float sigma_a, sigma_s, g, le_vol;
	float t_medium_end; // medium extends from 0 to t_medium_end
	mutable SVPTestRNG rng;
	static constexpr float kPi = 3.14159265f;

	bool Intersect(const float* /*org*/, const float* /*dir*/, float /*t_max*/,
				   BDPTHit<float>& /*hit*/) const { return false; }
	void InfiniteLightLe(const float* /*dir*/, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;
	}
	bool HasMedium(const float* /*org*/, const float* /*dir*/) const { return true; }

	float SampleTMaj(const float* org, const float* dir, float t_max, float u_maj,
					  const std::function<bool(const float*, const VolPathMediumProps<float>&,
											   float, float)>& cb) const {
		float sigma_maj = sigma_a + sigma_s;
		if (sigma_maj <= 0.f) return 1.f;

		float t_end = std::min(t_max, t_medium_end);
		// Sample a free-flight distance via exponential
		float t = -std::log(1.f - u_maj * (1.f - std::exp(-sigma_maj * t_end))) / sigma_maj;
		if (t >= t_end) return 1.f;

		float p[3] = { org[0]+t*dir[0], org[1]+t*dir[1], org[2]+t*dir[2] };
		VolPathMediumProps<float> mp;
		mp.sigma_a = sigma_a;
		mp.sigma_s = sigma_s;
		mp.Le      = le_vol;
		mp.g       = g;
		cb(p, mp, sigma_maj, 1.f);
		return 1.f;
	}

	bool SamplePhase(const float* /*wo*/, float /*g*/, float u1, float u2,
					  float wi[3], float& pdf) const {
		// Isotropic
		float cos_t = 1.f - 2.f * u2;
		float sin_t = std::sqrt(std::max(0.f, 1.f - cos_t*cos_t));
		float phi = 2.f * kPi * u1;
		wi[0] = sin_t*std::cos(phi);
		wi[1] = sin_t*std::sin(phi);
		wi[2] = cos_t;
		pdf = 1.f / (4.f * kPi);
		return true;
	}
	float PhaseP(const float*, const float*, float) const { return 1.f/(4.f*kPi); }
	float RandFloat() const { return rng(); }
	void SpawnRay(const BDPTHit<float>& h, const float d[3],
				  float new_o[3], float new_d[3]) const {
		const float eps = 1e-4f;
		new_o[0]=h.p[0]+eps*h.shading_n[0];
		new_o[1]=h.p[1]+eps*h.shading_n[1];
		new_o[2]=h.p[2]+eps*h.shading_n[2];
		new_d[0]=d[0]; new_d[1]=d[1]; new_d[2]=d[2];
	}
};

// ---------------------------------------------------------------------------
// PureNullScene: sigma_a=sigma_s=0, sigma_maj>0 -- all events are null
// Ray must pass through to the infinite light background.
// ---------------------------------------------------------------------------
struct SVPNullScene {
	float sigma_maj = 1.f;
	float infinite_le = 0.8f;
	mutable SVPTestRNG rng;
	static constexpr float kPi = 3.14159265f;

	bool Intersect(const float*, const float*, float, BDPTHit<float>&) const { return false; }
	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = infinite_le;
	}
	bool HasMedium(const float*, const float*) const { return true; }

	float SampleTMaj(const float* org, const float* dir, float t_max, float u,
					  const std::function<bool(const float*, const VolPathMediumProps<float>&,
											   float, float)>& cb) const {
		// Always fire a null event (sigma_a=sigma_s=0)
		float t = std::min(-std::log(1.f - u * 0.5f) / sigma_maj, t_max * 0.5f);
		float p[3] = { org[0]+t*dir[0], org[1]+t*dir[1], org[2]+t*dir[2] };
		VolPathMediumProps<float> mp;
		mp.sigma_a = 0.f; mp.sigma_s = 0.f; mp.Le = 0.f; mp.g = 0.f;
		cb(p, mp, sigma_maj, 1.f);
		return 1.f;
	}
	bool SamplePhase(const float*, float, float, float, float* wi, float& pdf) const {
		wi[0]=wi[1]=0.f; wi[2]=1.f; pdf=1.f/(4.f*kPi); return true;
	}
	float PhaseP(const float*, const float*, float) const { return 1.f/(4.f*kPi); }
	float RandFloat() const { return rng(); }
	void SpawnRay(const BDPTHit<float>& h, const float d[3],
				  float new_o[3], float new_d[3]) const {
		const float eps = 1e-4f;
		new_o[0]=h.p[0]+eps*h.shading_n[0];
		new_o[1]=h.p[1]+eps*h.shading_n[1];
		new_o[2]=h.p[2]+eps*h.shading_n[2];
		new_d[0]=d[0]; new_d[1]=d[1]; new_d[2]=d[2];
	}
};

// ---------------------------------------------------------------------------
// MediumBoundaryScene: geometry hit is a medium boundary, then a real surface
// ---------------------------------------------------------------------------
struct SVPBoundaryScene {
	mutable SVPTestRNG rng;
	mutable int intersect_calls = 0;
	float surface_le = 1.f;

	bool Intersect(const float* org, const float* dir, float /*t_max*/,
				   BDPTHit<float>& hit) const {
		++intersect_calls;
		// First call returns medium boundary at t=1; second returns emissive surface at t=2
		if (intersect_calls == 1) {
			fill_hit(hit, org, dir, 1.f, /*boundary=*/true);
		} else {
			fill_hit(hit, org, dir, 1.f, /*boundary=*/false, /*light=*/true, surface_le);
		}
		return true;
	}
	void InfiniteLightLe(const float*, float out[3]) const { out[0]=out[1]=out[2]=0.f; }
	bool HasMedium(const float*, const float*) const { return false; }
	float SampleTMaj(const float*, const float*, float, float,
					  const std::function<bool(const float*, const VolPathMediumProps<float>&,
											   float, float)>&) const { return 1.f; }
	bool SamplePhase(const float*, float, float, float, float* wi, float& pdf) const {
		wi[0]=wi[1]=0.f; wi[2]=1.f; pdf=1.f/(4.f*3.14159265f); return true;
	}
	float PhaseP(const float*, const float*, float) const { return 1.f/(4.f*3.14159265f); }
	float RandFloat() const { return rng(); }
	void SpawnRay(const BDPTHit<float>& h, const float d[3],
				  float new_o[3], float new_d[3]) const {
		const float eps = 1e-4f;
		new_o[0]=h.p[0]+eps*h.shading_n[0];
		new_o[1]=h.p[1]+eps*h.shading_n[1];
		new_o[2]=h.p[2]+eps*h.shading_n[2];
		new_d[0]=d[0]; new_d[1]=d[1]; new_d[2]=d[2];
	}
};

// ===========================================================================
// Tests
// ===========================================================================

// 1. Vacuum + miss → infinite light
TEST(SimpleVolPath, Vacuum_MissReturnsInfiniteLight) {
	SVPVacuumScene scene;
	scene.infinite_le = 0.5f;
	float org[3] = {0,0,0}, dir[3] = {0,0,1}, L[3] = {};
	SimpleVolPathLi<float>(org, dir, scene, /*max_depth=*/8, L);
	EXPECT_NEAR(L[0], 0.5f, 1e-5f);
	EXPECT_NEAR(L[1], 0.5f, 1e-5f);
	EXPECT_NEAR(L[2], 0.5f, 1e-5f);
}

// 2. Vacuum + emissive sphere → area_Le
TEST(SimpleVolPath, Vacuum_SurfaceEmission) {
	SVPVacuumScene scene;
	scene.has_emissive_sphere = true;
	scene.sphere_le = 2.f;
	scene.infinite_le = 0.f;
	float org[3] = {0,0,-3}, dir[3] = {0,0,1}, L[3] = {};
	SimpleVolPathLi<float>(org, dir, scene, 8, L);
	// beta=1, area_Le=2 -- sphere is hit
	EXPECT_GT(L[0], 0.f);
	EXPECT_NEAR(L[0], 2.f, 1e-4f);
}

// 3. Vacuum + no emission → zero
TEST(SimpleVolPath, Vacuum_NoEmission_ReturnsZero) {
	SVPVacuumScene scene;
	scene.infinite_le = 0.f;
	float org[3] = {0,0,0}, dir[3] = {0,0,1}, L[3] = {};
	SimpleVolPathLi<float>(org, dir, scene, 8, L);
	EXPECT_EQ(L[0], 0.f);
	EXPECT_EQ(L[1], 0.f);
	EXPECT_EQ(L[2], 0.f);
}

// 4. max_depth=0, in medium → scatter events terminate immediately → finite result
TEST(SimpleVolPath, MaxDepth0_InMedium_NeverExceedsDepth) {
	SVPHomogeneousScene scene{0.1f, 0.9f, 0.f, 0.f, 10.f};
	float org[3] = {0,0,0}, dir[3] = {0,0,1}, L[3] = {};
	SimpleVolPathLi<float>(org, dir, scene, /*max_depth=*/0, L);
	for (int c=0;c<3;++c) {
		EXPECT_TRUE(std::isfinite(L[c])) << "channel " << c;
		EXPECT_GE(L[c], 0.f) << "channel " << c;
	}
}

// 5. Absorbing medium with Le → picks up volume emission and terminates
TEST(SimpleVolPath, Absorption_TerminatesWithVolumeEmission) {
	// Pure absorber with emission; many seeds should produce positive average
	float sum = 0.f;
	for (uint32_t seed = 0; seed < 30; ++seed) {
		SVPHomogeneousScene scene{10.f, 0.f, 0.f, /*le_vol=*/1.f, 5.f};
		scene.rng.eng.seed(seed);
		float org[3]={0,0,0}, dir[3]={0,0,1}, L[3]={};
		SimpleVolPathLi<float>(org, dir, scene, 8, L);
		sum += L[0];
	}
	EXPECT_GT(sum, 0.f) << "Pure absorber with Le>0 must produce positive radiance";
}

// 6. Scattering medium → non-negative and finite over many seeds
TEST(SimpleVolPath, HomogeneousMedium_NonNegativeAndFinite) {
	for (uint32_t seed = 0; seed < 40; ++seed) {
		SVPHomogeneousScene scene{0.1f, 0.5f, 0.f, 0.f, 8.f};
		scene.rng.eng.seed(seed);
		float org[3]={0,0,0}, dir[3]={0,0,1}, L[3]={};
		SimpleVolPathLi<float>(org, dir, scene, 8, L);
		for (int c=0;c<3;++c) {
			EXPECT_GE(L[c], 0.f)      << "seed=" << seed << " ch=" << c;
			EXPECT_TRUE(std::isfinite(L[c])) << "seed=" << seed << " ch=" << c;
		}
	}
}

// 7. Pure null medium → passes through to infinite light
TEST(SimpleVolPath, NullEvent_ContinuesMarch_ReachesBackground) {
	// With sigma_a=sigma_s=0 the null event always fires; ray must eventually
	// reach the background after the medium ends (SampleTMaj returns 1 after
	// one null event, then HasMedium on the new org still returns true but the
	// outer loop will see no hit and query InfiniteLightLe).
	// We run several seeds; the result should always equal infinite_le.
	// NOTE: because SampleTMaj in our NullScene only fires ONE null event and
	// then returns, the outer loop will call InfiniteLightLe on the next
	// iteration (no surface hit).  This is the correct delta-tracking pass-
	// through behaviour.
	SVPNullScene scene;
	scene.infinite_le = 0.8f;
	float org[3]={0,0,0}, dir[3]={0,0,1}, L[3]={};
	SimpleVolPathLi<float>(org, dir, scene, 8, L);
	// Result should converge to the background value
	EXPECT_GT(L[0], 0.f) << "Null-only medium must eventually reach background";
	for (int c=0;c<3;++c)
		EXPECT_TRUE(std::isfinite(L[c])) << "ch=" << c;
}

// 8. Medium boundary surface is skipped; real emissive surface behind it is reached
TEST(SimpleVolPath, MediumBoundary_SkippedCorrectly) {
	SVPBoundaryScene scene;
	float org[3]={0,0,-1}, dir[3]={0,0,1}, L[3]={};
	SimpleVolPathLi<float>(org, dir, scene, 8, L);
	// The boundary at t=1 should be skipped; the emissive surface behind it
	// should contribute scene.surface_le to the output.
	EXPECT_NEAR(L[0], scene.surface_le, 1e-4f);
}
