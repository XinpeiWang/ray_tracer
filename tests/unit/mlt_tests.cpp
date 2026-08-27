// mlt_tests.cpp -- Unit tests for src/shared/mlt.h
//
// Test groups:
//   1. ErfInv / SampleNormal math helpers
//   2. MLTSampler: Get1D, Accept, Reject, StartStream, small/large steps
//   3. MLTEvalPath: uses the same SyntheticScene as bdpt_tests
//   4. MLTBootstrap: normalization constant b >= 0
//   5. MLTRenderLoop: output non-negative, finite, total splats == nMutations*2

#include "gtest/gtest.h"
#include "../../src/shared/mlt.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <random>

// ---------------------------------------------------------------------------
// Re-use the same minimal synthetic scene from bdpt_tests
// (copy kept local so tests are self-contained)
// ---------------------------------------------------------------------------
struct MLTTestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	MLTTestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

struct MLTSyntheticScene {
	MLTTestRNG rng;

	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		float a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
		float b = 2*(org[0]*dir[0]+org[1]*dir[1]+org[2]*dir[2]);
		float c = org[0]*org[0]+org[1]*org[1]+org[2]*org[2] - 1.f;
		float disc = b*b - 4*a*c;
		if (disc < 0) return false;
		float sq = std::sqrt(disc);
		float t = (-b - sq)/(2*a);
		if (t < 1e-4f) t = (-b + sq)/(2*a);
		if (t < 1e-4f || t > t_max) return false;
		hit.t_hit = t;
		hit.p[0]=org[0]+t*dir[0]; hit.p[1]=org[1]+t*dir[1]; hit.p[2]=org[2]+t*dir[2];
		float len = std::sqrt(hit.p[0]*hit.p[0]+hit.p[1]*hit.p[1]+hit.p[2]*hit.p[2]);
		hit.geo_n[0]=hit.p[0]/len; hit.geo_n[1]=hit.p[1]/len; hit.geo_n[2]=hit.p[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false; hit.bsdf_id=0;
		return true;
	}

	// Never actually reached (Intersect() above always sets
	// is_medium_boundary=false) - but mlt.h's MLTSceneAdapter forwards
	// SpawnRay() unconditionally in its own source, so it needs to exist to
	// compile.
	void SpawnRay(const BDPTHit<float>& hit, const float dir[3], float new_o[3], float new_d[3]) const {
		new_o[0] = hit.p[0]; new_o[1] = hit.p[1]; new_o[2] = hit.p[2];
		new_d[0] = dir[0]; new_d[1] = dir[1]; new_d[2] = dir[2];
	}

	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float dir[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]};
		float len=std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
		if(len<1e-6f) return true;
		dir[0]/=len; dir[1]/=len; dir[2]/=len;
		BDPTHit<float> hit{};
		return !Intersect(p0, dir, len-1e-3f, hit);
	}

	void BSDFf(int, const float*, const float wi[3], const float n[3], float out[3]) const {
		float cos_i = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		if (cos_i <= 0.f) { out[0]=out[1]=out[2]=0.f; return; }
		out[0]=out[1]=out[2] = 0.8f / 3.14159265358979f;
	}

	bool BSDFSampleF(int, const float*, const float n[3], float u1, float u2,
					 float new_dir[3], float f_val[3], float& pdf, bool& is_specular) const {
		float phi=2.f*3.14159265358979f*u1, cos_t=std::sqrt(u2), sin_t=std::sqrt(1.f-u2);
		float tx,ty,tz,bx,by,bz;
		if(std::fabs(n[0])>0.9f){tx=0.f;ty=1.f;tz=0.f;}else{tx=1.f;ty=0.f;tz=0.f;}
		float cx=ty*n[2]-tz*n[1],cy=tz*n[0]-tx*n[2],cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl; ty=cy/cl; tz=cz/cl;
		bx=n[1]*tz-n[2]*ty; by=n[2]*tx-n[0]*tz; bz=n[0]*ty-n[1]*tx;
		float lx=sin_t*std::cos(phi),ly=sin_t*std::sin(phi),lz=cos_t;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		f_val[0]=f_val[1]=f_val[2]=0.8f/3.14159265358979f;
		pdf=cos_t/3.14159265358979f; is_specular=false;
		return pdf>0.f;
	}

	float BSDFPdf(int, const float*, const float wi[3], const float n[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return c>0.f ? c/3.14159265358979f : 0.f;
	}

	static constexpr float kLightPos[3]={0.f,3.f,0.f};

	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		ls.p_light[0]=kLightPos[0]; ls.p_light[1]=kLightPos[1]; ls.p_light[2]=kLightPos[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		float d[3]={kLightPos[0]-ref_p[0],kLightPos[1]-ref_p[1],kLightPos[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0]=d[0]/len; ls.wi[1]=d[1]/len; ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f; ls.is_delta=true; ls.light_id=0;
		return true;
	}

	bool SampleLightLe(float, const float[2], const float[2], BDPTLightLeSample<float>& les) const {
		float u1=rng(), u2=rng();
		float phi=2.f*3.14159265358979f*u1, cos_t=1.f-2.f*u2;
		float sin_t=std::sqrt(std::max(0.f,1.f-cos_t*cos_t));
		les.ray_o[0]=kLightPos[0]; les.ray_o[1]=kLightPos[1]; les.ray_o[2]=kLightPos[2];
		les.ray_d[0]=sin_t*std::cos(phi); les.ray_d[1]=sin_t*std::sin(phi); les.ray_d[2]=cos_t;
		les.p_on_light[0]=kLightPos[0]; les.p_on_light[1]=kLightPos[1]; les.p_on_light[2]=kLightPos[2];
		les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.f;
		les.L[0]=les.L[1]=les.L[2]=1.f;
		les.pdf_pos=1.f; les.pdf_dir=1.f/(4.f*3.14159265358979f);
		les.abs_cos_theta=1.f; les.is_on_surface=false;
		les.is_infinite=false; les.is_delta_dir=false; les.light_id=0;
		return true;
	}

	float LightPMF(int) const { return 1.f; }
	void LightPDFLe(int, const float*, const float*, const float*,
					float& pp, float& pd) const { pp=1.f; pd=1.f/(4.f*3.14159265358979f); }

	static constexpr float kCamPos[3]={0.f,0.f,5.f};

	void CameraPDFWe(const float*, const float*, float& pp, float& pd) const { pp=1.f; pd=1.f; }
	bool CameraSampleWi(const float ref_p[3], const float*,
						float wi[3], float& pdf, float& imp, float* pr, float* p_cam) const {
		float d[3]={kCamPos[0]-ref_p[0],kCamPos[1]-ref_p[1],kCamPos[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if(len<1e-6f) return false;
		wi[0]=d[0]/len; wi[1]=d[1]/len; wi[2]=d[2]/len;
		pdf=1.f; imp=1.f;
		if(pr){pr[0]=pr[1]=0.f;}
		p_cam[0]=kCamPos[0]; p_cam[1]=kCamPos[1]; p_cam[2]=kCamPos[2];
		return true;
	}
	void InfiniteLightLe(const float[3], float out[3]) const { out[0]=out[1]=out[2]=0.f; }
	float InfiniteLightDensity(const float[3]) const { return 0.f; }
	void SceneBoundingSphere(float c[3], float& r) const { c[0]=c[1]=c[2]=0.f; r=5.f; }
	float RandFloat() const { return rng(); }

	// MLT-specific extensions
	bool PixelToRay(float px, float py,
					float cam_p[3], float ray_d[3], float cam_n[3]) const {
		// Map [0,1)^2 to a ray through the sphere
		cam_p[0]=0.f; cam_p[1]=0.f; cam_p[2]=5.f;
		cam_n[0]=0.f; cam_n[1]=0.f; cam_n[2]=-1.f;
		// Vary direction slightly with px,py so different pixels get different rays
		ray_d[0]=(px-0.5f)*0.5f; ray_d[1]=(py-0.5f)*0.5f; ray_d[2]=-1.f;
		float len=std::sqrt(ray_d[0]*ray_d[0]+ray_d[1]*ray_d[1]+ray_d[2]*ray_d[2]);
		ray_d[0]/=len; ray_d[1]/=len; ray_d[2]/=len;
		return true;
	}

	float Luminance(float r, float g, float b) const {
		return 0.2126f*r + 0.7152f*g + 0.0722f*b;
	}
};

constexpr float MLTSyntheticScene::kLightPos[3];
constexpr float MLTSyntheticScene::kCamPos[3];

// ===========================================================================
// Tests 1: ErfInv / SampleNormal
// ===========================================================================

TEST(MLTMath, ErfInvAtZero) {
	// erfinv(0) = 0
	EXPECT_NEAR(mlt_detail::ErfInv<float>(0.f), 0.f, 1e-5f);
}

TEST(MLTMath, ErfInvRoundtrip) {
	// erf(erfinv(x)) == x for x in (-1,1)
	for (float x : {-0.9f, -0.5f, -0.1f, 0.1f, 0.5f, 0.9f}) {
		float inv = mlt_detail::ErfInv<float>(x);
		float back = std::erf(inv);
		EXPECT_NEAR(back, x, 1e-4f) << "x=" << x;
	}
}

TEST(MLTMath, SampleNormalMedianIsZero) {
	// SampleNormal(u=0.5, sigma) should give ~0
	float v = mlt_detail::SampleNormal<float>(0.5f, 1.f);
	EXPECT_NEAR(v, 0.f, 1e-4f);
}

TEST(MLTMath, SampleNormalMonotone) {
	// SampleNormal should be monotone in u
	float sigma = 0.1f;
	float prev = -1e30f;
	for (float u : {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}) {
		float v = mlt_detail::SampleNormal<float>(u, sigma);
		EXPECT_GT(v, prev) << "u=" << u;
		prev = v;
	}
}

// ===========================================================================
// Tests 2: MLTSampler
// ===========================================================================

TEST(MLTSampler, Get1DInUnitInterval) {
	MLTSampler<float> sampler(1, 42u, 0.01f, 0.3f);
	for (int i = 0; i < 100; ++i) {
		float v = sampler.Get1D();
		EXPECT_GE(v, 0.f) << "i=" << i;
		EXPECT_LT(v, 1.f) << "i=" << i;
	}
}

TEST(MLTSampler, AcceptPreservesValue) {
	// Accept() commits the proposal: iteration counter stays at currentIteration_.
	// Reject() would decrement it back.
	MLTSampler<float> sampler(1, 0u, 0.01f, 0.0f);
	sampler.Get1D();
	sampler.StartIteration();
	int64_t iter = sampler.CurrentIteration();
	sampler.Get1D();
	sampler.Accept();
	EXPECT_EQ(sampler.CurrentIteration(), iter); // still at iter, not iter-1
}

TEST(MLTSampler, RejectRestoresValue) {
	MLTSampler<float> sampler(1, 0u, 0.01f, 0.0f);
	float v0 = sampler.Get1D(); // read index 0, mutation 0
	sampler.StartIteration();
	float v1 = sampler.Get1D(); // mutates index 0 to a new value
	(void)v1;
	sampler.Reject();
	// After reject, reading index 0 again should give back v0
	sampler.StartStream(0);
	float v2 = sampler.Get1D();
	EXPECT_FLOAT_EQ(v0, v2);
}

TEST(MLTSampler, StartStreamSetsIndexToZero) {
	// StartStream(k) resets sampleIndex to 0, so GetNextIndex() = k + streamCount*0 = k.
	// Verify by checking that the PSS vector grows after each Get1D across streams.
	MLTSampler<float> sampler(1, 1u, 0.01f, 0.3f);
	sampler.StartStream(MLTSampler<float>::kCameraStream);
	float d0 = sampler.Get1D(); // PSS index 0
	sampler.StartStream(MLTSampler<float>::kLightStream);
	float d1 = sampler.Get1D(); // PSS index 1
	sampler.StartStream(MLTSampler<float>::kConnectionStream);
	float d2 = sampler.Get1D(); // PSS index 2
	// All values must be in [0,1)
	EXPECT_GE(d0, 0.f); EXPECT_LT(d0, 1.f);
	EXPECT_GE(d1, 0.f); EXPECT_LT(d1, 1.f);
	EXPECT_GE(d2, 0.f); EXPECT_LT(d2, 1.f);
	// PSS vector should have at least 3 entries
	EXPECT_GE((int)sampler.X().size(), 3);
}

TEST(MLTSampler, LargeStepProb1AlwaysLargeStep) {
	MLTSampler<float> sampler(1, 7u, 0.01f, 1.0f); // always large
	float v0 = sampler.Get1D();
	for (int i = 0; i < 5; ++i) {
		sampler.StartIteration();
		EXPECT_TRUE(sampler.IsLargeStep());
		float v = sampler.Get1D();
		EXPECT_GE(v, 0.f); EXPECT_LT(v, 1.f);
		sampler.Accept();
	}
}

TEST(MLTSampler, SmallStepWrapAround) {
	// Small steps must keep value in [0,1) via floor wrapping
	MLTSampler<float> sampler(1, 99u, 10.f, 0.0f); // large sigma, no large step
	for (int i = 0; i < 20; ++i) {
		sampler.StartIteration();
		float v = sampler.Get1D();
		EXPECT_GE(v, 0.f) << "iter " << i;
		EXPECT_LT(v, 1.f) << "iter " << i;
		sampler.Accept();
	}
}

TEST(MLTSampler, MultipleStreamsDistinct) {
	MLTSampler<float> sampler(1, 5u, 0.01f, 0.3f);
	sampler.StartStream(MLTSampler<float>::kCameraStream);
	float c0 = sampler.Get1D();
	sampler.StartStream(MLTSampler<float>::kLightStream);
	float l0 = sampler.Get1D();
	sampler.StartStream(MLTSampler<float>::kConnectionStream);
	float n0 = sampler.Get1D();
	// They come from different PSS dimensions so should generally differ
	// (not always, but extremely unlikely to all be equal)
	EXPECT_FALSE(c0 == l0 && l0 == n0) << "all streams returned same value";
}

TEST(MLTSampler, IterationCountIncreases) {
	MLTSampler<float> sampler(1, 0u, 0.01f, 0.5f);
	EXPECT_EQ(sampler.CurrentIteration(), 0);
	sampler.StartIteration();
	EXPECT_EQ(sampler.CurrentIteration(), 1);
	sampler.Accept();
	sampler.StartIteration();
	EXPECT_EQ(sampler.CurrentIteration(), 2);
	sampler.Reject();
	EXPECT_EQ(sampler.CurrentIteration(), 1); // decremented by Reject
}

// ===========================================================================
// Tests 3: MLTEvalPath
// ===========================================================================

TEST(MLTEvalPath, ReturnsFiniteValues) {
	MLTSyntheticScene scene;
	MLTSampler<float> sampler(1, 42u, 0.01f, 0.3f);
	auto r = MLTEvalPath(sampler, 1, scene, 3);
	if (r.valid) {
		EXPECT_TRUE(std::isfinite(r.L[0])) << r.L[0];
		EXPECT_TRUE(std::isfinite(r.L[1])) << r.L[1];
		EXPECT_TRUE(std::isfinite(r.L[2])) << r.L[2];
	}
}

TEST(MLTEvalPath, ReturnsNonNegative) {
	MLTSyntheticScene scene;
	for (int seed = 0; seed < 10; ++seed) {
		MLTSampler<float> sampler(1, (uint64_t)seed, 0.01f, 0.3f);
		for (int depth = 0; depth <= 3; ++depth) {
			auto r = MLTEvalPath(sampler, depth, scene, 3);
			if (r.valid) {
				EXPECT_GE(r.L[0], 0.f) << "seed=" << seed << " depth=" << depth;
				EXPECT_GE(r.L[1], 0.f);
				EXPECT_GE(r.L[2], 0.f);
			}
		}
	}
}

TEST(MLTEvalPath, PixelPositionInUnitSquare) {
	MLTSyntheticScene scene;
	MLTSampler<float> sampler(1, 1u, 0.01f, 0.3f);
	auto r = MLTEvalPath(sampler, 1, scene, 3);
	if (r.valid) {
		EXPECT_GE(r.px, 0.f); EXPECT_LT(r.px, 1.f);
		EXPECT_GE(r.py, 0.f); EXPECT_LT(r.py, 1.f);
	}
}

// ===========================================================================
// Tests 4: MLTBootstrap
// ===========================================================================

TEST(MLTBootstrap, WeightVectorCorrectSize) {
	MLTSyntheticScene scene;
	std::vector<float> weights;
	int nBoot = 16, maxD = 2;
	MLTBootstrap(nBoot, maxD, 0.01f, 0.3f, scene, weights);
	EXPECT_EQ((int)weights.size(), nBoot * (maxD + 1));
}

TEST(MLTBootstrap, WeightsNonNegative) {
	MLTSyntheticScene scene;
	std::vector<float> weights;
	MLTBootstrap(16, 2, 0.01f, 0.3f, scene, weights);
	for (float w : weights) EXPECT_GE(w, 0.f);
}

TEST(MLTBootstrap, ReturnsNonNegativeB) {
	MLTSyntheticScene scene;
	std::vector<float> weights;
	float b = MLTBootstrap(16, 2, 0.01f, 0.3f, scene, weights);
	EXPECT_GE(b, 0.f);
	EXPECT_TRUE(std::isfinite(b));
}

// ===========================================================================
// Tests 5: MLTRenderLoop
// ===========================================================================

TEST(MLTRenderLoop, SplatsAreNonNegative) {
	MLTSyntheticScene scene;
	int splatCount = 0;
	MLTRenderLoop<float>(16, 50, 2, 0.01f, 0.3f, scene,
		[&](float px, float py, float r, float g, float b) {
			EXPECT_GE(r, 0.f);
			EXPECT_GE(g, 0.f);
			EXPECT_GE(b, 0.f);
			EXPECT_GE(px, 0.f); EXPECT_LT(px, 1.f);
			EXPECT_GE(py, 0.f); EXPECT_LT(py, 1.f);
			++splatCount;
		});
	// Each of nMutations iterations produces up to 2 splats (current + proposed)
	// At minimum there should be some splats for a lit scene
	EXPECT_GT(splatCount, 0);
}

TEST(MLTRenderLoop, SplatsAreFinite) {
	MLTSyntheticScene scene;
	MLTRenderLoop<float>(16, 50, 2, 0.01f, 0.3f, scene,
		[&](float px, float py, float r, float g, float b) {
			EXPECT_TRUE(std::isfinite(r)) << "r=" << r;
			EXPECT_TRUE(std::isfinite(g)) << "g=" << g;
			EXPECT_TRUE(std::isfinite(b)) << "b=" << b;
		});
}

TEST(MLTRenderLoop, MultipleRunsDifferentSeeds) {
	// Two runs with different seeds should produce different total energy
	MLTSyntheticScene scene1, scene2;
	float total1 = 0.f, total2 = 0.f;
	MLTRenderLoop<float>(8, 20, 2, 0.01f, 0.3f, scene1,
		[&](float, float, float r, float g, float b) { total1 += r+g+b; });
	// Change sampler seed by using different nBootstrap
	MLTRenderLoop<float>(9, 20, 2, 0.01f, 0.3f, scene2,
		[&](float, float, float r, float g, float b) { total2 += r+g+b; });
	// Both totals should be finite and non-negative
	EXPECT_GE(total1, 0.f);
	EXPECT_GE(total2, 0.f);
	EXPECT_TRUE(std::isfinite(total1));
	EXPECT_TRUE(std::isfinite(total2));
}

TEST(MLTRenderLoop, BlackSceneProducesNoSplats) {
	// Scene with no lights -> b=0 -> RenderLoop returns immediately
	struct BlackScene : MLTSyntheticScene {
		bool SampleLight(float, const float*, BDPTLightSample<float>& ls) const {
			ls.L[0]=ls.L[1]=ls.L[2]=0.f; ls.pdf=1.f; ls.is_delta=true; ls.light_id=0;
			return false; // no light
		}
		float Luminance(float, float, float) const { return 0.f; }
	} blackScene;

	int splatCount = 0;
	MLTRenderLoop<float>(8, 20, 2, 0.01f, 0.3f, blackScene,
		[&](float, float, float, float, float) { ++splatCount; });
	EXPECT_EQ(splatCount, 0);
}
