// bssrdf_vol_path_tests.cpp
//
// Unit tests for the BSSRDF probe-segment sampling branch in VolPathLi<T>.
//
// pbrt-v4 reference: VolPathIntegrator::Li lines 1187-1255
//   (cpu/integrators.cpp)
//
// Key invariants tested:
//   1. HasBSSRDF=false  -> BSSRDF branch never executes, output matches vanilla path.
//   2. HasBSSRDF=true, transmissive scatter -> BSSRDF branch fires; beta != 0.
//   3. SampleBSSRDFProbe returns false -> path terminates cleanly (no crash, L >= 0).
//   4. Regularize fires at exit point when regularize=true and HasBSSRDF fires.
//   5. Non-negative radiance invariant with BSSRDF enabled.

#include "gtest/gtest.h"
#include "../../src/shared/vol_path.h"

#include <cmath>
#include <functional>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// Minimal RNG
// ---------------------------------------------------------------------------
struct BSSTestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	explicit BSSTestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// SlabScene
//
// Geometry:
//   Entry surface: an infinite plane at z = 0 (normal = +z), hit from above.
//   Exit  surface: an infinite plane at z = -0.1 (normal = +z).
//   Rays from above (org z > 0, dir z < 0) hit the entry plane.
//   A Lambertian BSDF with a specular transmission flag controlled by
//   `has_bssrdf` so we can toggle the subsurface branch.
//
// BSSRDF:
//   SampleBSSRDFProbe builds a probe segment from p0=(hit.p) to p1 shifted
//   by (0,0,-0.2), i.e. straight downward.  The exit plane is at z=-0.1
//   so Intersect will find it.
//
// Notes:
//   - The exit-plane hit has is_medium_boundary=false, bsdf_id=1.
//   - BSDFSampleF for bsdf_id=0 returns is_specular=true, is_transmission=true
//     when has_bssrdf, so the BSSRDF branch fires (condition: is_transmission && HasBSSRDF).
// ---------------------------------------------------------------------------
struct SlabScene {
	mutable int   regularize_calls = 0;
	mutable bool  bssrdf_probe_called = false;
	mutable bool  bssrdf_sample_called = false;
	bool has_bssrdf        = true;   // enable/disable BSSRDF branch
	bool probe_fails       = false;  // make SampleBSSRDFProbe return false
	mutable BSSTestRNG rng_;
	static constexpr float kPi = 3.14159265358979f;

	// -----------------------------------------------------------------------
	// Geometry: ray hits entry plane (z=0) or exit plane (z=-0.1)
	// -----------------------------------------------------------------------
	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		// Try entry plane z=0 first
		if (std::fabs(dir[2]) > 1e-8f) {
			float t_entry = -org[2] / dir[2];
			if (t_entry > 1e-3f && t_entry < t_max) {
				fill_hit(hit, org, dir, t_entry, 0.f, 0 /*bsdf_id entry*/);
				return true;
			}
			// Try exit plane z=-0.1
			float t_exit = (-0.1f - org[2]) / dir[2];
			if (t_exit > 1e-3f && t_exit < t_max) {
				fill_hit(hit, org, dir, t_exit, -0.1f, 1 /*bsdf_id exit*/);
				return true;
			}
		}
		return false;
	}
	bool Unoccluded(const float*, const float*) const { return true; }

	// -----------------------------------------------------------------------
	// Lights: single point light above
	// -----------------------------------------------------------------------
	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		static constexpr float kLP[3] = {0.f, 0.f, 2.f};
		float d[3] = {kLP[0]-ref_p[0], kLP[1]-ref_p[1], kLP[2]-ref_p[2]};
		float len = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.p_light[0]=kLP[0]; ls.p_light[1]=kLP[1]; ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		ls.wi[0]=d[0]/len; ls.wi[1]=d[1]/len; ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f; ls.is_delta=true; ls.light_id=0;
		return true;
	}
	float LightPMF(int) const { return 1.f; }
	bool  IsDeltaLight(int) const { return true; }
	float LightPDF_Li(int, const float*, const float*) const { return 0.f; }
	void  InfiniteLightLe(const float*, float out[3]) const {
		out[0]=out[1]=out[2]=0.05f;
	}
	float InfiniteLightPMF() const { return 0.f; }
	float InfiniteLightPDF_Li(const float*, const float*) const { return 0.f; }

	// -----------------------------------------------------------------------
	// BSDF
	//   bsdf_id=0 (entry): specular transmission when has_bssrdf (to trigger branch)
	//                       Lambertian otherwise.
	//   bsdf_id=1 (exit):  Lambertian always.
	// -----------------------------------------------------------------------
	void BSDFf(int id, const float*, const float wi[3],
			   const float n[3], float out[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		out[0]=out[1]=out[2]=(c>0.f)?0.5f/kPi*c:0.f;
		(void)id;
	}
	float BSDFPdf(int, const float*, const float wi[3], const float n[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return c>0.f?c/kPi:0.f;
	}
	bool BSDFSampleF(int id, const float*, const float n[3],
					  float u1, float u2,
					  float nd[3], float fv[3], float& pdf,
					  bool& is_spec, bool& is_trans) const {
		if (id == 0 && has_bssrdf) {
			// Specular transmission straight through: direction = -n
			nd[0]=-n[0]; nd[1]=-n[1]; nd[2]=-n[2];
			fv[0]=fv[1]=fv[2]=1.f;
			pdf=1.f; is_spec=true; is_trans=true;
			return true;
		}
		// Cosine-hemisphere Lambertian
		float phi=2.f*kPi*u1, ct=std::sqrt(u2), st=std::sqrt(std::max(0.f,1.f-u2));
		float tx=1.f,ty=0.f,tz=0.f;
		if(std::fabs(n[0])>0.9f){tx=0.f;ty=1.f;}
		float cx=ty*n[2]-tz*n[1],cy=tz*n[0]-tx*n[2],cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz); tx=cx/cl;ty=cy/cl;tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty,by=n[2]*tx-n[0]*tz,bz=n[0]*ty-n[1]*tx;
		float lx=st*std::cos(phi),ly=st*std::sin(phi);
		nd[0]=lx*tx+ly*bx+ct*n[0]; nd[1]=lx*ty+ly*by+ct*n[1]; nd[2]=lx*tz+ly*bz+ct*n[2];
		fv[0]=fv[1]=fv[2]=0.5f/kPi;
		pdf=ct/kPi; is_spec=false; is_trans=false;
		return pdf>0.f;
	}
	bool BSDFIsNonSpecular(int id) const {
		if (id == 0 && has_bssrdf) return false; // specular transmission
		return true;
	}
	void BSDFRegularize(int) const { ++regularize_calls; }

	// -----------------------------------------------------------------------
	// Medium: vacuum
	// -----------------------------------------------------------------------
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

	// -----------------------------------------------------------------------
	// BSSRDF
	// -----------------------------------------------------------------------
	bool HasBSSRDF(const BDPTHit<float>&) const { return has_bssrdf; }

	bool SampleBSSRDFProbe(const BDPTHit<float>& hit,
							float /*u1*/, float /*u2*/,
							BSSRDFProbeSegment<float>& seg) const {
		bssrdf_probe_called = true;
		if (probe_fails) return false;
		// p0: just below the entry surface
		seg.p0[0]=hit.p[0]; seg.p0[1]=hit.p[1]; seg.p0[2]=hit.p[2]-1e-3f;
		// p1: 0.2 units below entry, so exit plane at z=-0.1 is within segment
		seg.p1[0]=hit.p[0]; seg.p1[1]=hit.p[1]; seg.p1[2]=hit.p[2]-0.2f;
		return true;
	}

	bool SampleBSSRDF(const BDPTHit<float>& /*entry*/,
					   const BDPTHit<float>& exit_hit,
					   float sample_prob,
					   BSSRDFSample<float>& out) const {
		bssrdf_sample_called = true;
		// Sp = 0.5 (diffuse-like profile value), all channels equal
		out.Sp[0]=out.Sp[1]=out.Sp[2]=0.5f;
		// pdf = uniform over sample_prob (= 1/N candidates = 1 here)
		out.pdf = (sample_prob > 0.f) ? sample_prob : 1.f;
		// wo at exit: pointing upward (+z)
		out.wo[0]=0.f; out.wo[1]=0.f; out.wo[2]=1.f;
		out.exit_bsdf_id = 1; // exit surface uses Lambertian bsdf_id=1
		return true;
	}

	// -----------------------------------------------------------------------
	// SpawnRay
	// -----------------------------------------------------------------------
	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-3f;
		new_o[0]=hit.p[0]+kEps*hit.geo_n[0];
		new_o[1]=hit.p[1]+kEps*hit.geo_n[1];
		new_o[2]=hit.p[2]+kEps*hit.geo_n[2];
		new_d[0]=dir[0]; new_d[1]=dir[1]; new_d[2]=dir[2];
	}

	float RandFloat() const { return rng_(); }

private:
	void fill_hit(BDPTHit<float>& hit,
				  const float org[3], const float dir[3],
				  float t, float plane_z, int bsdf_id) const {
		hit.t_hit = t;
		hit.p[0]=org[0]+t*dir[0]; hit.p[1]=org[1]+t*dir[1]; hit.p[2]=plane_z;
		// Normal always points up (+z)
		hit.geo_n[0]=0.f; hit.geo_n[1]=0.f; hit.geo_n[2]=1.f;
		hit.shading_n[0]=0.f; hit.shading_n[1]=0.f; hit.shading_n[2]=1.f;
		hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false;
		hit.bsdf_id=bsdf_id; hit.light_id=-1;
	}
};

// ===========================================================================
// Test 1: BSSRDF disabled (has_bssrdf=false)
//   The BSSRDF branch must never execute — probe and sample callbacks are
//   never called, and radiance is non-negative.
// ===========================================================================
TEST(BSSRDFVolPath, Disabled_NoBSSRDFCallbacks) {
	SlabScene scene;
	scene.has_bssrdf = false;
	float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L);
	EXPECT_FALSE(scene.bssrdf_probe_called);
	EXPECT_FALSE(scene.bssrdf_sample_called);
	for (int c = 0; c < 3; ++c)
		EXPECT_GE(L[c], 0.f) << "channel " << c;
}

// ===========================================================================
// Test 2: BSSRDF enabled — probe and sample are called, beta updates fire
//   The BSSRDF branch must execute: bssrdf_probe_called and
//   bssrdf_sample_called both become true.  Radiance must be non-negative.
// ===========================================================================
TEST(BSSRDFVolPath, Enabled_ProbeAndSampleCalled) {
	SlabScene scene;
	scene.has_bssrdf = true;
	float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/6, scene, L);
	EXPECT_TRUE(scene.bssrdf_probe_called)
		<< "SampleBSSRDFProbe must be called when HasBSSRDF=true";
	EXPECT_TRUE(scene.bssrdf_sample_called)
		<< "SampleBSSRDF must be called after a successful probe walk";
	for (int c = 0; c < 3; ++c)
		EXPECT_GE(L[c], 0.f) << "channel " << c;
}

// ===========================================================================
// Test 3: SampleBSSRDFProbe returns false -> path terminates cleanly
//   No crash, L is non-negative.
// ===========================================================================
TEST(BSSRDFVolPath, ProbeFails_PathTerminatesGracefully) {
	SlabScene scene;
	scene.has_bssrdf = true;
	scene.probe_fails = true;
	float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
	ASSERT_NO_FATAL_FAILURE(
		VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L));
	for (int c = 0; c < 3; ++c)
		EXPECT_GE(L[c], 0.f) << "channel " << c;
}

// ===========================================================================
// Test 4: Regularize fires at exit point when regularize=true
//   (mirrors pbrt-v4 lines 1234-1239: anyNonSpecularBounces=true, Sw.Regularize())
// ===========================================================================
TEST(BSSRDFVolPath, Regularize_FiresAtExitPoint) {
	SlabScene scene;
	scene.has_bssrdf = true;
	float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L, /*regularize=*/true);
	EXPECT_GT(scene.regularize_calls, 0)
		<< "BSDFRegularize must be called at the BSSRDF exit point when regularize=true";
}

// ===========================================================================
// Test 5: Non-negative radiance invariant across multiple seeds
// ===========================================================================
TEST(BSSRDFVolPath, NonNegativeRadiance_MultipleSeedsEnabled) {
	for (uint32_t seed = 0; seed < 20; ++seed) {
		SlabScene scene;
		scene.rng_ = BSSTestRNG(seed);
		scene.has_bssrdf = true;
		float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
		VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L);
		for (int c = 0; c < 3; ++c)
			EXPECT_GE(L[c], 0.f) << "seed=" << seed << " channel=" << c;
	}
}

// ===========================================================================
// Test 6: Mirror reflection (is_specular=true, is_transmission=false) does NOT
//         enter the BSSRDF branch even when HasBSSRDF returns true.
//         Mirrors pbrt-v4 guard: if (bssrdf && bs->IsTransmission())
// ===========================================================================

// Minimal scene: BSDF always does mirror reflection (is_trans=false) but
// HasBSSRDF returns true.  The BSSRDF probe must never be called.
struct MirrorScene {
	mutable bool bssrdf_probe_called = false;
	mutable BSSTestRNG rng_;
	static constexpr float kPi = 3.14159265358979f;

	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		// Single plane at z=0
		if (std::fabs(dir[2]) > 1e-8f) {
			float t = -org[2] / dir[2];
			if (t > 1e-3f && t < t_max) {
				hit.p[0]=org[0]+t*dir[0]; hit.p[1]=org[1]+t*dir[1]; hit.p[2]=0.f;
				hit.geo_n[0]=hit.shading_n[0]=0.f;
				hit.geo_n[1]=hit.shading_n[1]=0.f;
				hit.geo_n[2]=hit.shading_n[2]=1.f;
				hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
				hit.uv[0]=hit.uv[1]=0.f;
				hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
				hit.bsdf_id=0; hit.light_id=-1; hit.t_hit=t;
				hit.is_medium_boundary=false; hit.is_delta_bsdf=true;
				return true;
			}
		}
		return false;
	}
	bool Unoccluded(const float*, const float*) const { return true; }

	bool SampleLight(float, const float*, BDPTLightSample<float>&) const { return false; }
	float LightPMF(int) const { return 0.f; }
	bool  IsDeltaLight(int) const { return false; }
	float LightPDF_Li(int, const float*, const float*) const { return 0.f; }
	void  InfiniteLightLe(const float*, float out[3]) const { out[0]=out[1]=out[2]=0.f; }
	float InfiniteLightPMF() const { return 0.f; }
	float InfiniteLightPDF_Li(const float*, const float*) const { return 0.f; }

	void BSDFf(int, const float*, const float wi[3], const float n[3], float out[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		out[0]=out[1]=out[2]=(c>0.f)?1.f:0.f;
	}
	float BSDFPdf(int, const float*, const float*, const float*) const { return 1.f; }
	// Mirror reflection: is_spec=true, is_trans=false
	bool BSDFSampleF(int, const float* wo, const float n[3],
					  float, float,
					  float nd[3], float fv[3], float& pdf,
					  bool& is_spec, bool& is_trans) const {
		// Reflect wo about n: r = wo - 2*(wo·n)*n
		float dot = wo[0]*n[0]+wo[1]*n[1]+wo[2]*n[2];
		nd[0]=wo[0]-2.f*dot*n[0];
		nd[1]=wo[1]-2.f*dot*n[1];
		nd[2]=wo[2]-2.f*dot*n[2];
		fv[0]=fv[1]=fv[2]=1.f;
		pdf=1.f; is_spec=true; is_trans=false;
		return true;
	}
	bool BSDFIsNonSpecular(int) const { return false; }
	void BSDFRegularize(int) const {}

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

	// HasBSSRDF returns true but BSDFSampleF never sets is_trans=true
	bool HasBSSRDF(const BDPTHit<float>&) const { return true; }
	bool SampleBSSRDFProbe(const BDPTHit<float>&, float, float,
						   BSSRDFProbeSegment<float>&) const {
		bssrdf_probe_called = true;
		return false; // should never be reached
	}
	bool SampleBSSRDF(const BDPTHit<float>&, const BDPTHit<float>&, float,
					  BSSRDFSample<float>&) const { return false; }
	void SpawnRay(const BDPTHit<float>& h, const float d[3],
				  float out_o[3], float out_d[3]) const {
		out_o[0]=h.p[0]+1e-4f*h.shading_n[0];
		out_o[1]=h.p[1]+1e-4f*h.shading_n[1];
		out_o[2]=h.p[2]+1e-4f*h.shading_n[2];
		out_d[0]=d[0]; out_d[1]=d[1]; out_d[2]=d[2];
	}
	float RandFloat() const { return rng_(); }
};

TEST(BSSRDFVolPath, MirrorReflection_DoesNotEnterBSSRDF) {
	MirrorScene scene;
	float org[3]={0.f,0.f,1.f}, dir[3]={0.f,0.f,-1.f}, L[3]={};
	VolPathLi<float>(org, dir, /*maxDepth=*/4, scene, L);
	EXPECT_FALSE(scene.bssrdf_probe_called)
		<< "BSSRDF probe must NOT be called when BSDF scatter is specular reflection (is_transmission=false)";
}
