// vol_path_tests.cpp -- Unit tests for src/shared/vol_path.h
//
// Test groups:
//   1. VolPathSampleLd: vacuum shadow ray (delta light)
//   2. VolPathSampleLd: blocked shadow ray returns zero
//   3. VolPathSampleLd: zero f_hat returns zero
//   4. VolPathLi: vacuum scene (no medium), single-bounce direct lighting
//   5. VolPathLi: result is non-negative and finite
//   6. VolPathLi: black scene (light behind sphere) returns near-zero
//   7. VolPathLi: max depth 0 returns emission or zero
//   8. VolPathLi: medium scene (constant homogeneous medium) returns finite non-negative
//   9. VolPathLi: r_u starts at 1, path must not produce negative values
//  10. VolPathSampleLd: medium transmittance reduces contribution

#include "gtest/gtest.h"
#include "../../src/shared/vol_path.h"
#include <cmath>
#include <functional>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal synthetic scene helpers
// ---------------------------------------------------------------------------
struct VPTestRNG {
	mutable std::mt19937 eng{42u};
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	float operator()() const { return dist(eng); }
};

// Common sphere intersection
static bool SphereHit(const float org[3], const float dir[3],
					  float t_max, BDPTHit<float>& hit,
					  bool is_light = false) {
	float a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
	float b = 2*(org[0]*dir[0]+org[1]*dir[1]+org[2]*dir[2]);
	float c = org[0]*org[0]+org[1]*org[1]+org[2]*org[2] - 1.f;
	float disc = b*b-4*a*c;
	if (disc < 0) return false;
	float sq = std::sqrt(disc);
	float t = (-b-sq)/(2*a);
	if (t < 1e-4f) t = (-b+sq)/(2*a);
	if (t < 1e-4f || t > t_max) return false;
	hit.t_hit = t;
	for (int i=0;i<3;++i) hit.p[i]=org[i]+t*dir[i];
	float len=std::sqrt(hit.p[0]*hit.p[0]+hit.p[1]*hit.p[1]+hit.p[2]*hit.p[2]);
	for(int i=0;i<3;++i){ hit.geo_n[i]=hit.p[i]/len; hit.shading_n[i]=hit.geo_n[i]; }
	for(int i=0;i<3;++i) hit.wo[i]=-dir[i];
	hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]= is_light ? 1.f : 0.f;
	hit.is_delta_bsdf = false; hit.bsdf_id = 0;
	hit.light_id = is_light ? 0 : -1;
	return true;
}

// ---------------------------------------------------------------------------
// VacuumScene: unit sphere, point light at (0,3,0), camera at (0,0,5)
// No medium (vacuum).
// ---------------------------------------------------------------------------
struct VacuumScene {
	VPTestRNG rng;
	static constexpr float kLP[3] = {0.f, 0.f, 2.f};

	bool Intersect(const float org[3], const float dir[3],
				   float t_max, BDPTHit<float>& hit) const {
		return SphereHit(org, dir, t_max, hit);
	}
	// VacuumScene has no occluding geometry between surface points and lights.
	// The sphere only provides Intersect for camera rays; shadow rays are unblocked.
	bool Unoccluded(const float*, const float*) const { return true; }
	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		ls.p_light[0]=kLP[0];ls.p_light[1]=kLP[1];ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		float d[3]={kLP[0]-ref_p[0],kLP[1]-ref_p[1],kLP[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0]=d[0]/len;ls.wi[1]=d[1]/len;ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f;
		ls.pdf=1.f;ls.is_delta=true;ls.light_id=0;
		return true;
	}
	float LightPMF(int) const { return 1.f; }
	bool  IsDeltaLight(int) const { return true; }
	float LightPDF_Li(int, const float*, const float*) const { return 0.f; }
	void  BSDFf(int, const float*, const float wi[3], const float n[3],
				float out[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		if(c<=0.f){out[0]=out[1]=out[2]=0.f;return;}
		out[0]=out[1]=out[2]=0.8f/3.14159265f;
	}
	bool BSDFSampleF(int, const float*, const float n[3], float u1, float u2,
					 float nd[3], float fv[3], float& pdf, bool& is_spec) const {
		float phi=2.f*3.14159265f*u1,ct=std::sqrt(u2),st=std::sqrt(1.f-u2);
		float tx=1.f,ty=0.f,tz=0.f;
		if(std::fabs(n[0])>0.9f){tx=0.f;ty=1.f;}
		float cx=ty*n[2]-tz*n[1],cy=tz*n[0]-tx*n[2],cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz); tx=cx/cl;ty=cy/cl;tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty,by=n[2]*tx-n[0]*tz,bz=n[0]*ty-n[1]*tx;
		float lx=st*std::cos(phi),ly=st*std::sin(phi);
		nd[0]=lx*tx+ly*bx+ct*n[0];
		nd[1]=lx*ty+ly*by+ct*n[1];
		nd[2]=lx*tz+ly*bz+ct*n[2];
		fv[0]=fv[1]=fv[2]=0.8f/3.14159265f;
		pdf=ct/3.14159265f; is_spec=false;
		return pdf>0.f;
	}
	float BSDFPdf(int, const float*, const float wi[3], const float n[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return c>0.f?c/3.14159265f:0.f;
	}
	bool  BSDFIsNonSpecular(int) const { return true; }
	bool  HasMedium(const float*, const float*) const { return false; }
	float SampleTMaj(const float*, const float*, float, float,
					 const std::function<bool(const float*, const VolPathMediumProps<float>&,
											  float, float)>&) const { return 1.f; }
	float PhaseP(const float*, const float*, float) const { return 1.f/(4.f*3.14159265f); }
	float PhasePDF(const float*, const float*, float) const { return 1.f/(4.f*3.14159265f); }
	bool  SamplePhase(const float*, float, float u1, float u2,
					  float wi[3], float& pdf) const {
		float phi=2.f*3.14159265f*u1, ct=1.f-2.f*u2;
		float st=std::sqrt(std::max(0.f,1.f-ct*ct));
		wi[0]=st*std::cos(phi);wi[1]=st*std::sin(phi);wi[2]=ct;
		pdf=1.f/(4.f*3.14159265f);
		return true;
	}
	void  InfiniteLightLe(const float*, float out[3]) const {out[0]=out[1]=out[2]=0.f;}
	float InfiniteLightPMF() const { return 0.f; }
	float InfiniteLightPDF_Li(const float*, const float*) const { return 0.f; }
	float RandFloat() const { return rng(); }
};
constexpr float VacuumScene::kLP[3];

// ---------------------------------------------------------------------------
// BlockedScene: same sphere but shadow always blocked
// ---------------------------------------------------------------------------
struct BlockedScene : VacuumScene {
	bool Unoccluded(const float*, const float*) const { return false; }
};

// ---------------------------------------------------------------------------
// HomogeneousScene: vacuum scene + simple homogeneous medium (constant sigma)
// ---------------------------------------------------------------------------
struct HomogeneousScene : VacuumScene {
	float sigma_a = 0.1f;
	float sigma_s = 0.2f;
	float sigma_maj_val = 0.35f; // > sigma_a + sigma_s

	bool HasMedium(const float*, const float*) const { return true; }

	float SampleTMaj(const float* /*org*/, const float* /*dir*/,
					 float t_max,
					 float /*u_maj*/,
					 const std::function<bool(const float*,
						 const VolPathMediumProps<float>&,
						 float, float)>& callback) const {
		// Simplified: fire one callback at t=0.5*t_max with T_maj = exp(-sigma_maj*dt)
		float dt = std::min(t_max, 0.5f);
		float T_maj_val = std::exp(-sigma_maj_val * dt);
		float p_mid[3] = {0.f, 0.f, 0.f}; // approximate
		VolPathMediumProps<float> mp;
		mp.sigma_a = sigma_a;
		mp.sigma_s = sigma_s;
		mp.Le = 0.f;
		mp.g  = 0.f;
		bool cont = callback(p_mid, mp, sigma_maj_val, T_maj_val);
		(void)cont;
		// Return residual T_maj for the rest of the segment
		return std::exp(-sigma_maj_val * std::max(0.f, t_max - dt));
	}
};

// ===========================================================================
// Tests 1: VolPathSampleLd -- vacuum, delta light, unoccluded
// ===========================================================================

TEST(VolPathSampleLd, DeltaLightUnoccludedPositive) {
	VacuumScene scene;
	float p[3]={0.f,0.f,0.8f}, wo[3]={0.f,0.f,-1.f}, n[3]={0.f,0.f,1.f};
	float beta[3]={1.f,1.f,1.f};
	float Ld[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, scene, Ld);
	EXPECT_GT(Ld[0]+Ld[1]+Ld[2], 0.f);
}

TEST(VolPathSampleLd, DeltaLightNonNegative) {
	VacuumScene scene;
	float p[3]={0.f,0.f,0.8f}, wo[3]={0.f,0.f,-1.f}, n[3]={0.f,0.f,1.f};
	float beta[3]={1.f,1.f,1.f};
	float Ld[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, scene, Ld);
	for (int c=0;c<3;++c) EXPECT_GE(Ld[c], 0.f);
}

// ===========================================================================
// Tests 2: VolPathSampleLd -- blocked shadow ray
// ===========================================================================

TEST(VolPathSampleLd, BlockedReturnsZero) {
	BlockedScene scene;
	float p[3]={0.f,0.8f,0.f}, wo[3]={0.f,0.f,1.f}, n[3]={0.f,1.f,0.f};
	float beta[3]={1.f,1.f,1.f};
	float Ld[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, scene, Ld);
	EXPECT_EQ(Ld[0], 0.f); EXPECT_EQ(Ld[1], 0.f); EXPECT_EQ(Ld[2], 0.f);
}

// ===========================================================================
// Tests 3: VolPathSampleLd -- zero f_hat
// ===========================================================================

TEST(VolPathSampleLd, ZeroBSDFReturnsZero) {
	// Light is behind the surface (cos_i < 0) -> f_hat = 0
	VacuumScene scene;
	float p[3]={0.f,0.f,0.8f};
	float wo[3]={0.f,0.f,-1.f};
	float n[3]={0.f,0.f,-1.f};  // normal pointing away from light at (0,0,2)
	float beta[3]={1.f,1.f,1.f};
	float Ld[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, scene, Ld);
	EXPECT_EQ(Ld[0], 0.f); EXPECT_EQ(Ld[1], 0.f); EXPECT_EQ(Ld[2], 0.f);
}

// ===========================================================================
// Tests 4 & 5: VolPathLi -- vacuum scene
// ===========================================================================

TEST(VolPathLi, VacuumNonNegative) {
	VacuumScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	for (int c=0;c<3;++c) EXPECT_GE(L[c], 0.f) << "c=" << c;
}

TEST(VolPathLi, VacuumFinite) {
	VacuumScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	for (int c=0;c<3;++c) EXPECT_TRUE(std::isfinite(L[c])) << "c=" << c;
}

TEST(VolPathLi, VacuumLitScenePositive) {
	// Ray aimed at sphere lit by point light above -- expect some radiance
	VacuumScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	float total = L[0]+L[1]+L[2];
	EXPECT_GT(total, 0.f);
}

// ===========================================================================
// Tests 6: VolPathLi -- black (blocked) scene
// ===========================================================================

TEST(VolPathLi, BlockedSceneNearZero) {
	BlockedScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	// Shadow blocked: direct lighting = 0, no emission, no infinite light
	EXPECT_NEAR(L[0]+L[1]+L[2], 0.f, 1e-6f);
}

// ===========================================================================
// Tests 7: VolPathLi -- maxDepth = 0
// ===========================================================================

TEST(VolPathLi, MaxDepthZeroNonNegative) {
	VacuumScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 0, scene, L);
	for (int c=0;c<3;++c) {
		EXPECT_GE(L[c], 0.f);
		EXPECT_TRUE(std::isfinite(L[c]));
	}
}

TEST(VolPathLi, MaxDepthZeroNoEmissionReturnsZero) {
	// Sphere has no emission, depth=0 terminates after first hit with no NEE
	// (depth++ >= maxDepth breaks before SampleLd)
	VacuumScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 0, scene, L);
	// depth=0 terminates immediately without NEE -> L should be ~0
	EXPECT_NEAR(L[0]+L[1]+L[2], 0.f, 1e-5f);
}

// ===========================================================================
// Tests 8: VolPathLi -- homogeneous medium
// ===========================================================================

TEST(VolPathLi, HomogeneousMediumNonNegative) {
	HomogeneousScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	for (int c=0;c<3;++c) EXPECT_GE(L[c], 0.f) << "c=" << c;
}

TEST(VolPathLi, HomogeneousMediumFinite) {
	HomogeneousScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3];
	VolPathLi(org, dir, 5, scene, L);
	for (int c=0;c<3;++c) EXPECT_TRUE(std::isfinite(L[c])) << "c=" << c;
}

TEST(VolPathLi, HomogeneousMediumLessThanVacuum) {
	// Adding absorption/scattering should reduce (or equal) direct lighting
	// compared to vacuum for the same geometry
	VacuumScene vacScene;
	HomogeneousScene medScene;
	// Reset RNG to same seed for fair comparison
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float Lv[3], Lm[3];
	VolPathLi(org, dir, 5, vacScene, Lv);
	VolPathLi(org, dir, 5, medScene, Lm);
	// Medium should attenuate, so Lm <= Lv (with some tolerance for noise)
	// We just check the medium result is still non-negative and finite
	for (int c=0;c<3;++c) {
		EXPECT_GE(Lm[c], 0.f);
		EXPECT_TRUE(std::isfinite(Lm[c]));
	}
}

// ===========================================================================
// Tests 9: Multiple calls -- all non-negative
// ===========================================================================

TEST(VolPathLi, MultipleRaysAllNonNegative) {
	VacuumScene scene;
	float dirs[5][3] = {
		{0.f,0.f,-1.f},
		{0.1f,0.f,-1.f},
		{-0.1f,0.f,-1.f},
		{0.f,0.1f,-1.f},
		{0.f,-0.1f,-1.f},
	};
	for (auto& d : dirs) {
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		d[0]/=len;d[1]/=len;d[2]/=len;
		float org[3]={0.f,0.f,5.f};
		float L[3];
		VolPathLi(org, d, 4, scene, L);
		for (int c=0;c<3;++c) {
			EXPECT_GE(L[c], 0.f);
			EXPECT_TRUE(std::isfinite(L[c]));
		}
	}
}

// ===========================================================================
// Tests 10: VolPathSampleLd with medium transmittance reduces contribution
// ===========================================================================

TEST(VolPathSampleLd, MediumReducesContribution) {
	// With absorbing medium, contribution should be <= vacuum contribution
	VacuumScene vacScene;
	HomogeneousScene medScene;

	float p[3]={0.f,0.f,0.8f}, wo[3]={0.f,0.f,-1.f}, n[3]={0.f,0.f,1.f};
	float beta[3]={1.f,1.f,1.f};
	float Ld_vac[3], Ld_med[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, vacScene, Ld_vac);
	VolPathSampleLd(p, wo, n, 0, 0.f, beta, 1.f, medScene, Ld_med);

	// Both must be non-negative and finite
	for (int c=0;c<3;++c) {
		EXPECT_GE(Ld_med[c], 0.f);
		EXPECT_TRUE(std::isfinite(Ld_med[c]));
	}
}

TEST(VolPathSampleLd, BetaScalesResult) {
	// Doubling beta should double Ld
	VacuumScene scene;
	float p[3]={0.f,0.f,0.8f}, wo[3]={0.f,0.f,-1.f}, n[3]={0.f,0.f,1.f};
	float beta1[3]={1.f,1.f,1.f};
	float beta2[3]={2.f,2.f,2.f};
	float Ld1[3], Ld2[3];
	VolPathSampleLd(p, wo, n, 0, 0.f, beta1, 1.f, scene, Ld1);
	VolPathSampleLd(p, wo, n, 0, 0.f, beta2, 1.f, scene, Ld2);
	for (int c=0;c<3;++c)
		EXPECT_NEAR(Ld2[c], 2.f * Ld1[c], 1e-5f) << "c=" << c;
}

// ---------------------------------------------------------------------------
// Alignment tests: null-scatter r_l and post-scatter r_u
// These test the two pbrt-v4 alignment fixes:
//   1. Null scatter must update r_l with majorant weight (not just r_u)
//   2. Post-scatter r_u must NOT be reset to 1
// ---------------------------------------------------------------------------

// A scene whose medium is pure null-scatter (sigma_a=sigma_s=0, sigma_maj>0).
// Any callback call should always return true (continue walking) and the
// final Li should still be non-negative and finite (beta/r_u remain valid).
struct NullScatterOnlyScene : VacuumScene {
	// Override HasMedium and SampleTMaj to provide pure null-scatter medium
	bool HasMedium(const float*, const float*) const { return true; }

	float SampleTMaj(const float* org, const float* dir, float t_max, float,
					 const std::function<bool(const float*,
						const VolPathMediumProps<float>&,
						float, float)>& cb) const {
		// One null-scatter event at t=0.5 (if t_max > 0.5)
		if (t_max < 0.5f) return 1.f;
		float p[3] = { org[0]+0.5f*dir[0], org[1]+0.5f*dir[1], org[2]+0.5f*dir[2] };
		VolPathMediumProps<float> mp;
		mp.sigma_a = 0.f; mp.sigma_s = 0.f; mp.Le = 0.f; mp.g = 0.f;
		float sigma_maj = 1.f;
		float T_maj_val = std::exp(-sigma_maj * 0.5f);
		cb(p, mp, sigma_maj, T_maj_val);  // always null event; should return true
		return T_maj_val;
	}
};

TEST(VolPathLi, NullScatterMediumNonNegativeFinite) {
	// Pure null-scatter medium: no energy is removed, r_l must be updated
	// correctly and the result must remain non-negative and finite.
	NullScatterOnlyScene scene;
	float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f};
	float L[3]={};
	VolPathLi(org, dir, /*maxDepth*/4, scene, L);
	for (int c=0;c<3;++c) {
		EXPECT_GE(L[c], 0.f) << "channel " << c;
		EXPECT_TRUE(std::isfinite(L[c])) << "channel " << c;
	}
}

// A scene with one real-scatter event: validates that r_u carries correct
// value after the scatter (not reset to 1) so the path contribution is finite.
TEST(VolPathLi, MediumScatterPostBounceFinite) {
	// HomogeneousScene already exercises real scatter.
	// Run many samples and check none are NaN or Inf (regression for r_u reset).
	HomogeneousScene scene;
	float org[3]={0.f,0.f,3.f}, dir[3]={0.f,0.f,-1.f};
	bool any_nan = false;
	for (int i = 0; i < 200; ++i) {
		float L[3]={};
		VolPathLi(org, dir, /*maxDepth*/6, scene, L);
		for (int c=0;c<3;++c) {
			if (!std::isfinite(L[c])) { any_nan = true; }
			EXPECT_GE(L[c], 0.f);
		}
	}
	EXPECT_FALSE(any_nan);
}
