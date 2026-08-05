// path_regularization_tests.cpp
//
// Tests for path regularization in PathLi<T> (path_integrator.h) and
// VolPathLi<T> (vol_path.h).
//
// pbrt-v4 reference: integrators.cpp lines 711-714 (PathIntegrator) and
//                    integrators.cpp lines 1151-1153 (VolPathIntegrator).
//
// The key invariant:
//   When regularize=true, BSDFRegularize(bsdf_id) is called if and only if
//   any_non_specular_bounce has been set before the current surface hit.
//   On the FIRST bounce (no prior non-specular scatter), it must NOT be called.
//   On subsequent bounces after a diffuse/glossy hit, it MUST be called.

#include "gtest/gtest.h"
#include "../../src/shared/path_integrator.h"
#include "../../src/shared/vol_path.h"

#include <cmath>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// Minimal RNG
// ---------------------------------------------------------------------------
struct PRTestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{ 0.f, 1.f };
	explicit PRTestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Synthetic scene with regularize tracking
//
// Geometry: unit sphere at origin (bsdf_id=0).
// BSDF: Lambertian (non-specular) with albedo 0.5.
// BSDFRegularize increments a mutable counter so tests can assert it was
// (or wasn't) called.
// ---------------------------------------------------------------------------
struct PRScene {
	mutable int  regularize_calls = 0;  // counts BSDFRegularize invocations
	mutable bool is_specular_bsdf = false; // toggle to test specular paths

	PRTestRNG rng;

	static constexpr float kPi = 3.14159265358979f;

	// ---- Geometry --------------------------------------------------------
	static bool sphere_hit(const float org[3], const float dir[3],
							float t_max, BDPTHit<float>& hit) {
		float a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
		float b = 2.f*(org[0]*dir[0] + org[1]*dir[1] + org[2]*dir[2]);
		float c = org[0]*org[0] + org[1]*org[1] + org[2]*org[2] - 1.f;
		float disc = b*b - 4.f*a*c;
		if (disc < 0.f) return false;
		float sq = std::sqrt(disc);
		float t  = (-b - sq) / (2.f*a);
		if (t < 1e-4f) t = (-b + sq) / (2.f*a);
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
		return sphere_hit(org, dir, t_max, hit);
	}

	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float d[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if (len < 1e-6f) return true;
		d[0]/=len; d[1]/=len; d[2]/=len;
		BDPTHit<float> tmp{};
		return !sphere_hit(p0, d, len - 1e-3f, tmp);
	}

	// ---- Lights (single dim point light, no occlusion test) --------------
	void SurfaceLe(const BDPTHit<float>&, const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;
	}
	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.05f;
	}
	float InfiniteLightPdf(const float*, const float*, const float*) const {
		return 1.f / (4.f * kPi);
	}
	float AreaLightPdf(const BDPTHit<float>&, const float*, const float*, const float*) const {
		return 0.f;
	}
	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		static constexpr float kLP[3] = {0.f, 0.f, 3.f};
		float d[3] = {kLP[0]-ref_p[0], kLP[1]-ref_p[1], kLP[2]-ref_p[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.p_light[0]=kLP[0]; ls.p_light[1]=kLP[1]; ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		ls.wi[0]=d[0]/len; ls.wi[1]=d[1]/len; ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f; ls.is_delta=true; ls.light_id=0;
		return true;
	}

	// ---- BSDF (Lambertian by default; specular mirror when is_specular_bsdf) --
	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		if (is_specular_bsdf) { out[0]=out[1]=out[2]=0.f; return; }
		float cos_i = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		out[0]=out[1]=out[2]=(cos_i>0.f)?(0.5f/kPi)*cos_i:0.f;
	}
	float BSDFPdf(int, const float*, const float wi[3],
				  const float n[3]) const {
		if (is_specular_bsdf) return 0.f;
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return (c > 0.f) ? c/kPi : 0.f;
	}
	bool BSDFSampleF(int, const float*, const float n[3],
					 float u1, float u2,
					 float new_dir[3], float f_val[3],
					 float& pdf, bool& is_specular,
					 bool& is_transmission, bool& pdf_is_proportional,
					 float& eta) const {
		if (is_specular_bsdf) {
			// Mirror reflection: wi = 2*(wo·n)*n - wo  (perfect specular)
			// wo is implicit from context; for testing, just bounce back along n
			new_dir[0]=n[0]; new_dir[1]=n[1]; new_dir[2]=n[2];
			f_val[0]=f_val[1]=f_val[2]=1.f;
			pdf=1.f; is_specular=true; is_transmission=false;
			pdf_is_proportional=false; eta=1.f;
			return true;
		}
		// Cosine-hemisphere
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(std::max(0.f, 1.f - u2));
		float phi   = 2.f * kPi * u1;
		float lx=sin_t*std::cos(phi), ly=sin_t*std::sin(phi), lz=cos_t;
		// ONB
		float tx, ty, tz;
		if (std::fabs(n[0]) > 0.9f) { tx=0.f; ty=1.f; tz=0.f; }
		else                         { tx=1.f; ty=0.f; tz=0.f; }
		float cx=ty*n[2]-tz*n[1], cy=tz*n[0]-tx*n[2], cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl; ty=cy/cl; tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty, by=n[2]*tx-n[0]*tz, bz=n[0]*ty-n[1]*tx;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		f_val[0]=f_val[1]=f_val[2]=(0.5f/kPi)*cos_t;
		pdf=cos_t/kPi;
		is_specular=false; is_transmission=false;
		pdf_is_proportional=false; eta=1.f;
		return pdf > 0.f;
	}

	bool BSDFIsReflectiveAndTransmissive(int) const { return false; }
	bool BSDFIsNonSpecular(int) const { return !is_specular_bsdf; }

	// BSDFRegularize: counts invocations
	void BSDFRegularize(int) const { ++regularize_calls; }

	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-4f;
		new_o[0]=hit.p[0]+kEps*hit.geo_n[0];
		new_o[1]=hit.p[1]+kEps*hit.geo_n[1];
		new_o[2]=hit.p[2]+kEps*hit.geo_n[2];
		new_d[0]=dir[0]; new_d[1]=dir[1]; new_d[2]=dir[2];
	}
};

// ---------------------------------------------------------------------------
// Always-hit scene: every ray hits an infinite Lambertian floor at y = -1.
// This guarantees multi-bounce paths for depth-dependent call-count tests.
// ---------------------------------------------------------------------------
struct PRAlwaysHitScene {
	mutable int regularize_calls = 0;
	PRTestRNG rng;
	static constexpr float kPi = 3.14159265358979f;

	// Always hit at t=1; normal is set to the incoming wo so BSDF sampling
	// stays in the upper hemisphere and the next ray always has a valid hit.
	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		constexpr float kT = 1.f;
		if (kT > t_max) return false;
		hit.t_hit = kT;
		hit.p[0]=org[0]+kT*dir[0]; hit.p[1]=org[1]+kT*dir[1]; hit.p[2]=org[2]+kT*dir[2];
		// Normal always faces the incoming ray so BSDF cos-hemisphere scatters away from it
		float len = std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
		hit.geo_n[0]=-dir[0]/len; hit.geo_n[1]=-dir[1]/len; hit.geo_n[2]=-dir[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		hit.wo[0]=-dir[0]/len; hit.wo[1]=-dir[1]/len; hit.wo[2]=-dir[2]/len;
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false;
		hit.bsdf_id=0; hit.light_id=-1;
		return true;
	}
	bool Unoccluded(const float*, const float*) const { return true; }

	void SurfaceLe(const BDPTHit<float>&, const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.f;
	}
	void InfiniteLightLe(const float*, float out[3]) const {
		out[0] = out[1] = out[2] = 0.05f;
	}
	float InfiniteLightPdf(const float*, const float*, const float*) const {
		return 1.f / (4.f * kPi);
	}
	float AreaLightPdf(const BDPTHit<float>&, const float*, const float*, const float*) const {
		return 0.f;
	}
	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		// Point light above
		static constexpr float kLP[3] = {0.f, 2.f, 0.f};
		float d[3] = {kLP[0]-ref_p[0], kLP[1]-ref_p[1], kLP[2]-ref_p[2]};
		float len  = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.p_light[0]=kLP[0]; ls.p_light[1]=kLP[1]; ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		ls.wi[0]=d[0]/len; ls.wi[1]=d[1]/len; ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f; ls.is_delta=true; ls.light_id=0;
		return true;
	}

	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		out[0]=out[1]=out[2]=(c>0.f)?(0.5f/kPi)*c:0.f;
	}
	float BSDFPdf(int, const float*, const float wi[3],
				  const float n[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return (c > 0.f) ? c/kPi : 0.f;
	}
	bool BSDFSampleF(int, const float*, const float n[3],
					 float u1, float u2,
					 float new_dir[3], float f_val[3],
					 float& pdf, bool& is_specular,
					 bool& is_transmission, bool& pdf_is_proportional,
					 float& eta) const {
		// Cosine-hemisphere sampling aligned to n
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(std::max(0.f, 1.f - u2));
		float phi   = 2.f * kPi * u1;
		float lx=sin_t*std::cos(phi), ly=sin_t*std::sin(phi), lz=cos_t;
		// ONB from n
		float tx, ty, tz;
		if (std::fabs(n[0]) > 0.9f) { tx=0.f; ty=1.f; tz=0.f; }
		else                         { tx=1.f; ty=0.f; tz=0.f; }
		float cx=ty*n[2]-tz*n[1], cy=tz*n[0]-tx*n[2], cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl; ty=cy/cl; tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty, by=n[2]*tx-n[0]*tz, bz=n[0]*ty-n[1]*tx;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		f_val[0]=f_val[1]=f_val[2]=(0.5f/kPi)*cos_t;
		pdf=cos_t/kPi;
		is_specular=false; is_transmission=false;
		pdf_is_proportional=false; eta=1.f;
		return pdf > 0.f;
	}
	bool BSDFIsReflectiveAndTransmissive(int) const { return false; }
	bool BSDFIsNonSpecular(int) const { return true; }
	void BSDFRegularize(int) const { ++regularize_calls; }

	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-3f;
		new_o[0]=hit.p[0]+kEps*hit.geo_n[0];
		new_o[1]=hit.p[1]+kEps*hit.geo_n[1];
		new_o[2]=hit.p[2]+kEps*hit.geo_n[2];
		new_d[0]=dir[0]; new_d[1]=dir[1]; new_d[2]=dir[2];
	}
};

// Helper: trace one ray with PathLi
static float trace_regularize(PRScene& scene,
							   bool regularize,
							   int max_depth = 4,
							   uint32_t seed = 42)
{
	PRTestRNG rng(seed);
	auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
	auto rand1d = [&]() { return rng(); };
	float org[3] = {0.f, 0.f, 5.f};
	float dir[3] = {0.f, 0.f, -1.f};
	float L[3]   = {0.f, 0.f, 0.f};
	PathLi<float>(org, dir, scene, max_depth, 1.f, rand2d, rand1d, L, regularize);
	return (L[0] + L[1] + L[2]) / 3.f;
}

// ===========================================================================
// PathRegularize — BSDFRegularize call-count invariants
// ===========================================================================

// regularize=false: BSDFRegularize must never be called regardless of depth
TEST(PathRegularize, Disabled_NeverCallsRegularize) {
	PRScene scene;
	trace_regularize(scene, /*regularize=*/false, /*max_depth=*/6);
	EXPECT_EQ(scene.regularize_calls, 0);
}

// regularize=true, max_depth=1: only one surface hit, any_non_specular_bounce
// is still false when we arrive (it's updated *after* BSDF sampling at depth 0),
// so BSDFRegularize should NOT be called on the very first bounce.
TEST(PathRegularize, Enabled_NoCallOnFirstBounce) {
	PRScene scene;
	trace_regularize(scene, /*regularize=*/true, /*max_depth=*/1);
	// Only one hit — no prior non-specular bounce yet
	EXPECT_EQ(scene.regularize_calls, 0);
}

// regularize=true, depth >= 2 with a Lambertian BSDF + guaranteed floor hit:
// any_non_specular_bounce is true by depth 1 so BSDFRegularize fires on all
// subsequent hits. With max_depth=4 and a floor that every ray hits,
// at least one regularize call is guaranteed.
TEST(PathRegularize, Enabled_CallsAfterFirstDiffuseBounce) {
	PRAlwaysHitScene scene;
	PRTestRNG rng(42);
	auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
	auto rand1d = [&]() { return rng(); };
	float org[3]={0.f, 1.f, 0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
	PathLi<float>(org, dir, scene, /*max_depth=*/4, /*rr=*/1.f,
				  rand2d, rand1d, L, /*regularize=*/true);
	EXPECT_GT(scene.regularize_calls, 0)
		<< "Expected BSDFRegularize to be called at least once after first diffuse bounce";
}

// More depth => same or more regularize calls (floor scene: every bounce hits)
TEST(PathRegularize, Enabled_MoreDepthMoreCalls) {
	auto run = [](int max_depth) {
		PRAlwaysHitScene scene;
		PRTestRNG rng(42);
		auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
		auto rand1d = [&]() { return rng(); };
		float org[3]={0.f,1.f,0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
		PathLi<float>(org, dir, scene, max_depth, 1.f, rand2d, rand1d, L, true);
		return scene.regularize_calls;
	};
	EXPECT_LE(run(2), run(5));
}


// BSDFRegularize should never be called even with regularize=true.
TEST(PathRegularize, SpecularBSDF_NoRegularizeEvenIfEnabled) {
	PRScene scene;
	scene.is_specular_bsdf = true;
	trace_regularize(scene, /*regularize=*/true, /*max_depth=*/4);
	EXPECT_EQ(scene.regularize_calls, 0);
}

// ===========================================================================
// PathRegularize — Output invariants
// ===========================================================================

// regularize=false should produce the same result as regularize=true when
// the mock BSDF's Regularize is a no-op.  Since our BSDFRegularize only
// increments a counter (does not change BSDF behavior in this mock), the
// radiance should be identical for the same seed.
TEST(PathRegularize, SameOutputWithMockRegularize) {
	PRScene scene_off, scene_on;
	float L_off = trace_regularize(scene_off, false, 4, 77);
	float L_on  = trace_regularize(scene_on,  true,  4, 77);
	// Values should be equal because BSDFRegularize is a counter-only no-op
	EXPECT_FLOAT_EQ(L_off, L_on);
}

// Non-negative radiance with regularize enabled
TEST(PathRegularize, NonNegativeRadiance_Enabled) {
	for (uint32_t seed = 0; seed < 20; ++seed) {
		PRScene scene;
		float L = trace_regularize(scene, true, 4, seed);
		EXPECT_GE(L, 0.f) << "seed=" << seed;
	}
}

// Non-negative radiance without regularize
TEST(PathRegularize, NonNegativeRadiance_Disabled) {
	for (uint32_t seed = 0; seed < 20; ++seed) {
		PRScene scene;
		float L = trace_regularize(scene, false, 4, seed);
		EXPECT_GE(L, 0.f) << "seed=" << seed;
	}
}

// ===========================================================================
// PathRegularize — Default parameter backward-compat
// ===========================================================================

// Calling PathLi without the regularize argument should compile and behave
// identically to passing regularize=false.
TEST(PathRegularize, DefaultParam_EquivalentToFalse) {
	PRScene scene_default, scene_false;
	PRTestRNG rng_a(99), rng_b(99);

	auto make_path = [](PRTestRNG& rng) {
		auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
		auto rand1d = [&]() { return rng(); };
		float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
		return std::make_tuple(rand2d, rand1d, org, dir, L);
	};
	(void)make_path; // suppress unused warning

	{
		PRTestRNG rng(99);
		auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
		auto rand1d = [&]() { return rng(); };
		float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
		PathLi<float>(org, dir, scene_default, 4, 1.f, rand2d, rand1d, L);
		EXPECT_EQ(scene_default.regularize_calls, 0);
	}
	{
		PRTestRNG rng(99);
		auto rand2d = [&]() { return std::make_pair(rng(), rng()); };
		auto rand1d = [&]() { return rng(); };
		float org[3]={0.f,0.f,5.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
		PathLi<float>(org, dir, scene_false, 4, 1.f, rand2d, rand1d, L, false);
		EXPECT_EQ(scene_false.regularize_calls, 0);
	}
}

// ===========================================================================
// VolPathRegularize — VolPathLi regularization (mirrors pbrt-v4 VolPathIntegrator)
// pbrt-v4 reference: integrators.cpp lines 1151-1153
// ===========================================================================

// Minimal scene satisfying VolPathLi's scene concept.
// Reuses PRAlwaysHitScene geometry (always hits at t=1) and adds the
// medium/phase/RNG methods that VolPathLi requires.
struct VRScene {
	mutable int  regularize_calls = 0;
	mutable PRTestRNG rng_;
	explicit VRScene(uint32_t seed = 42) : rng_(seed) {}
	static constexpr float kPi = 3.14159265358979f;

	// --- Geometry (same always-hit geometry as PRAlwaysHitScene) ---
	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		constexpr float kT = 1.f;
		if (kT > t_max) return false;
		hit.t_hit = kT;
		hit.p[0]=org[0]+kT*dir[0]; hit.p[1]=org[1]+kT*dir[1]; hit.p[2]=org[2]+kT*dir[2];
		float len = std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
		hit.geo_n[0]=-dir[0]/len; hit.geo_n[1]=-dir[1]/len; hit.geo_n[2]=-dir[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		hit.wo[0]=-dir[0]/len; hit.wo[1]=-dir[1]/len; hit.wo[2]=-dir[2]/len;
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false;
		hit.bsdf_id=0; hit.light_id=-1;
		return true;
	}
	bool Unoccluded(const float*, const float*) const { return true; }

	// --- Lights ---
	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		static constexpr float kLP[3]={0.f,2.f,0.f};
		float d[3]={kLP[0]-ref_p[0],kLP[1]-ref_p[1],kLP[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.p_light[0]=kLP[0];ls.p_light[1]=kLP[1];ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		ls.wi[0]=d[0]/len;ls.wi[1]=d[1]/len;ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f;ls.is_delta=true;ls.light_id=0;
		return true;
	}
	float LightPMF(int) const { return 1.f; }
	bool  IsDeltaLight(int) const { return true; }
	float LightPDF_Li(int, const float*, const float*) const { return 0.f; }
	void  InfiniteLightLe(const float*, float out[3]) const { out[0]=out[1]=out[2]=0.05f; }
	float InfiniteLightPMF() const { return 0.f; }
	float InfiniteLightPDF_Li(const float*, const float*) const { return 0.f; }

	// --- BSDF (Lambertian, non-specular) ---
	void BSDFf(int, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		out[0]=out[1]=out[2]=(c>0.f)?0.5f/kPi*c:0.f;
	}
	float BSDFPdf(int, const float*, const float wi[3], const float n[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return c>0.f?c/kPi:0.f;
	}
	bool BSDFSampleF(int, const float*, const float n[3],
				 float u1, float u2,
				 float nd[3], float fv[3], float& pdf,
				 bool& is_spec, bool& is_trans) const {
		float phi=2.f*kPi*u1, ct=std::sqrt(u2), st=std::sqrt(std::max(0.f,1.f-u2));
		float tx=1.f,ty=0.f,tz=0.f;
		if(std::fabs(n[0])>0.9f){tx=0.f;ty=1.f;}
		float cx=ty*n[2]-tz*n[1],cy=tz*n[0]-tx*n[2],cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);tx=cx/cl;ty=cy/cl;tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty,by=n[2]*tx-n[0]*tz,bz=n[0]*ty-n[1]*tx;
		float lx=st*std::cos(phi),ly=st*std::sin(phi);
		nd[0]=lx*tx+ly*bx+ct*n[0]; nd[1]=lx*ty+ly*by+ct*n[1]; nd[2]=lx*tz+ly*bz+ct*n[2];
		fv[0]=fv[1]=fv[2]=0.5f/kPi;
		pdf=ct/kPi; is_spec=false; is_trans=false;
		return pdf>0.f;
	}
	bool BSDFIsNonSpecular(int) const { return true; }
	void BSDFRegularize(int) const { ++regularize_calls; }

	// --- Medium (vacuum: no medium) ---
	bool HasMedium(const float*, const float*) const { return false; }
	float SampleTMaj(const float*, const float*, float, float,
					  const std::function<bool(const float*,
						  const VolPathMediumProps<float>&, float, float)>&) const {
		return 1.f;
	}
	float PhaseP(const float*, const float*, float) const { return 1.f/(4.f*kPi); }
	float PhasePDF(const float*, const float*, float) const { return 1.f/(4.f*kPi); }
	bool  SamplePhase(const float*, float, float u1, float u2,
					  float wi[3], float& pdf_out) const {
		float phi=2.f*kPi*u1, ct=1.f-2.f*u2, st=std::sqrt(std::max(0.f,1.f-ct*ct));
		wi[0]=st*std::cos(phi); wi[1]=st*std::sin(phi); wi[2]=ct;
		pdf_out=1.f/(4.f*kPi);
		return true;
	}
	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps=1e-3f;
		new_o[0]=hit.p[0]+kEps*hit.geo_n[0];
		new_o[1]=hit.p[1]+kEps*hit.geo_n[1];
		new_o[2]=hit.p[2]+kEps*hit.geo_n[2];
		new_d[0]=dir[0]; new_d[1]=dir[1]; new_d[2]=dir[2];
	}
	float RandFloat() const { return rng_(); }

	// --- BSSRDF (no subsurface scattering in this scene) ---
	bool HasBSSRDF(const BDPTHit<float>&) const { return false; }
	bool SampleBSSRDFProbe(const BDPTHit<float>&, float, float,
							BSSRDFProbeSegment<float>&) const { return false; }
	bool SampleBSSRDF(const BDPTHit<float>&, const BDPTHit<float>&,
					   float, BSSRDFSample<float>&) const { return false; }
};

// VolPathLi: disabled mode must never call BSDFRegularize
TEST(VolPathRegularize, Disabled_NeverCallsRegularize) {
	VRScene scene;
	float org[3]={0.f,1.f,0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/6, scene, L, /*regularize=*/false);
	EXPECT_EQ(scene.regularize_calls, 0);
}

// VolPathLi: regularize=true, maxDepth=1 — only one surface hit,
// any_non_specular_bounce is still false, so BSDFRegularize must NOT fire.
TEST(VolPathRegularize, Enabled_NoCallOnFirstBounce) {
	VRScene scene;
	float org[3]={0.f,1.f,0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/1, scene, L, /*regularize=*/true);
	EXPECT_EQ(scene.regularize_calls, 0);
}

// VolPathLi: guaranteed multi-bounce — BSDFRegularize must fire at least once.
TEST(VolPathRegularize, Enabled_CallsAfterFirstDiffuseBounce) {
	VRScene scene;
	float org[3]={0.f,1.f,0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L, /*regularize=*/true);
	EXPECT_GT(scene.regularize_calls, 0)
		<< "Expected BSDFRegularize to fire after first diffuse bounce";
}

// VolPathLi: default regularize parameter must equal false.
TEST(VolPathRegularize, DefaultParam_EquivalentToFalse) {
	VRScene scene;
	float org[3]={0.f,1.f,0.f}, dir[3]={0.f,-1.f,0.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L);
	EXPECT_EQ(scene.regularize_calls, 0);
}
