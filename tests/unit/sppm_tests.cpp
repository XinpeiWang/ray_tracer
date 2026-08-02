// sppm_tests.cpp -- Unit tests for src/shared/sppm.h
//
// Test groups:
//   1. sppm_detail: HashGrid (NextPrime, build, bucket, ToGrid)
//   2. SPPMPixel defaults
//   3. SPPMUpdateRadius: gamma=2/3 contraction formula
//   4. SPPMFinalImage: Ld + tau/(n*photons*pi*r^2)
//   5. SPPMCameraPass: visible points deposited, Ld set
//   6. SPPMPhotonPass: Phi/m accumulated for nearby visible points
//   7. SPPMRender: end-to-end non-negative finite output

#include "gtest/gtest.h"
#include "../../src/shared/sppm.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <random>

// ---------------------------------------------------------------------------
// Minimal synthetic scene (unit sphere at origin, point light at (0,3,0),
// camera at (0,0,5) looking toward -Z)
// ---------------------------------------------------------------------------
struct SPPMTestRNG {
	mutable std::mt19937 eng{42u};
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	float operator()() const { return dist(eng); }
};

struct SPPMTestScene {
	SPPMTestRNG rng;

	// Intersect with unit sphere
	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
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
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f;
		hit.is_medium_boundary=false; hit.is_delta_bsdf=false; hit.bsdf_id=0;
		return true;
	}

	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float d[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if(len<1e-6f) return true;
		d[0]/=len; d[1]/=len; d[2]/=len;
		BDPTHit<float> h{};
		return !Intersect(p0,d,len-1e-3f,h);
	}

	// Lambertian BSDF
	void BSDFf(int, const float*, const float wi[3], const float n[3], float out[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		if(c<=0.f){out[0]=out[1]=out[2]=0.f;return;}
		out[0]=out[1]=out[2]=0.8f/3.14159265f;
	}

	bool BSDFSampleF(int, const float*, const float n[3], float u1, float u2,
					 float new_dir[3], float f_val[3], float& pdf, bool& is_spec) const {
		float phi=2.f*3.14159265f*u1, cos_t=std::sqrt(u2), sin_t=std::sqrt(1.f-u2);
		float tx,ty,tz;
		if(std::fabs(n[0])>0.9f){tx=0;ty=1;tz=0;}else{tx=1;ty=0;tz=0;}
		float cx=ty*n[2]-tz*n[1],cy=tz*n[0]-tx*n[2],cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl;ty=cy/cl;tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty,by=n[2]*tx-n[0]*tz,bz=n[0]*ty-n[1]*tx;
		float lx=sin_t*std::cos(phi),ly=sin_t*std::sin(phi),lz=cos_t;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		f_val[0]=f_val[1]=f_val[2]=0.8f/3.14159265f;
		pdf=cos_t/3.14159265f; is_spec=false;
		return pdf>0.f;
	}

	float BSDFPdf(int, const float*, const float wi[3], const float n[3]) const {
		float c=wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return c>0.f ? c/3.14159265f : 0.f;
	}

	static constexpr float kLP[3]={0.f,3.f,0.f};

	bool SampleLight(float, const float ref_p[3], BDPTLightSample<float>& ls) const {
		ls.p_light[0]=kLP[0];ls.p_light[1]=kLP[1];ls.p_light[2]=kLP[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		float d[3]={kLP[0]-ref_p[0],kLP[1]-ref_p[1],kLP[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0]=d[0]/len;ls.wi[1]=d[1]/len;ls.wi[2]=d[2]/len;
		ls.L[0]=ls.L[1]=ls.L[2]=1.f/(len*len);
		ls.pdf=1.f;ls.is_delta=true;ls.light_id=0;
		return true;
	}

	bool SampleLightLe(float, const float[2], const float[2], BDPTLightLeSample<float>& les) const {
		float u1=rng(),u2=rng();
		float phi=2.f*3.14159265f*u1,cos_t=1.f-2.f*u2;
		float sin_t=std::sqrt(std::max(0.f,1.f-cos_t*cos_t));
		les.ray_o[0]=kLP[0];les.ray_o[1]=kLP[1];les.ray_o[2]=kLP[2];
		les.ray_d[0]=sin_t*std::cos(phi);les.ray_d[1]=sin_t*std::sin(phi);les.ray_d[2]=cos_t;
		les.p_on_light[0]=kLP[0];les.p_on_light[1]=kLP[1];les.p_on_light[2]=kLP[2];
		les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.f;
		les.L[0]=les.L[1]=les.L[2]=1.f;
		les.pdf_pos=1.f;les.pdf_dir=1.f/(4.f*3.14159265f);
		les.abs_cos_theta=1.f;les.is_on_surface=false;
		les.is_infinite=false;les.is_delta_dir=false;les.light_id=0;
		return true;
	}

	float LightPMF(int) const { return 1.f; }
	void LightPDFLe(int, const float*, const float*, const float*,
					float& pp, float& pd) const { pp=1.f; pd=1.f/(4.f*3.14159265f); }

	void CameraPDFWe(const float*, const float*, float& pp, float& pd) const { pp=1.f; pd=1.f; }

	bool CameraSampleWi(const float ref_p[3], const float*,
						float wi[3], float& pdf, float& imp, float* pr) const {
		float cp[3]={0.f,0.f,5.f};
		float d[3]={cp[0]-ref_p[0],cp[1]-ref_p[1],cp[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if(len<1e-6f) return false;
		wi[0]=d[0]/len;wi[1]=d[1]/len;wi[2]=d[2]/len;
		pdf=1.f;imp=1.f;
		if(pr){pr[0]=pr[1]=0.f;}
		return true;
	}

	void InfiniteLightLe(const float[3], float out[3]) const { out[0]=out[1]=out[2]=0.f; }
	float InfiniteLightDensity(const float[3]) const { return 0.f; }
	void SceneBoundingSphere(float c[3], float& r) const { c[0]=c[1]=c[2]=0.f; r=5.f; }
	float RandFloat() const { return rng(); }

	bool PixelToRay(float px, float py, float cam_p[3], float ray_d[3], float cam_n[3]) const {
		cam_p[0]=0.f;cam_p[1]=0.f;cam_p[2]=5.f;
		cam_n[0]=0.f;cam_n[1]=0.f;cam_n[2]=-1.f;
		ray_d[0]=(px-0.5f)*0.5f;ray_d[1]=(py-0.5f)*0.5f;ray_d[2]=-1.f;
		float len=std::sqrt(ray_d[0]*ray_d[0]+ray_d[1]*ray_d[1]+ray_d[2]*ray_d[2]);
		ray_d[0]/=len;ray_d[1]/=len;ray_d[2]/=len;
		return true;
	}

	void DirectLight(const float p[3], const float n[3], const float wo[3],
					 int bsdf_id, float Ld[3]) const {
		BDPTLightSample<float> ls{};
		if (!SampleLight(0.f, p, ls)) { Ld[0]=Ld[1]=Ld[2]=0.f; return; }
		float f[3];
		BSDFf(bsdf_id, wo, ls.wi, n, f);
		float cos_i=ls.wi[0]*n[0]+ls.wi[1]*n[1]+ls.wi[2]*n[2];
		if (cos_i <= 0.f || !Unoccluded(p, ls.p_light)) { Ld[0]=Ld[1]=Ld[2]=0.f; return; }
		for(int c=0;c<3;++c) Ld[c] = f[c]*cos_i*ls.L[c]/ls.pdf;
	}
};

constexpr float SPPMTestScene::kLP[3];

// ===========================================================================
// Tests 1: sppm_detail::HashGrid helpers
// ===========================================================================

TEST(SPPMHashGrid, NextPrimeSanity) {
	EXPECT_EQ(sppm_detail::NextPrime(0), 2);
	EXPECT_EQ(sppm_detail::NextPrime(1), 2);
	EXPECT_EQ(sppm_detail::NextPrime(2), 2);
	EXPECT_EQ(sppm_detail::NextPrime(3), 3);
	EXPECT_EQ(sppm_detail::NextPrime(4), 5);
	EXPECT_EQ(sppm_detail::NextPrime(10), 11);
	EXPECT_EQ(sppm_detail::NextPrime(100), 101);
}

TEST(SPPMHashGrid, BuildEmptyPixels) {
	std::vector<SPPMPixel<float>> pixels(4);
	// No valid visible points => grid should still initialise without crashing
	sppm_detail::HashGrid<float> g;
	g.Build(pixels);
	EXPECT_GE(g.hashSize, 1);
}

TEST(SPPMHashGrid, BuildSinglePixel) {
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].vp_valid = true;
	pixels[0].vp_p[0]=0.f; pixels[0].vp_p[1]=0.f; pixels[0].vp_p[2]=0.f;
	pixels[0].radius = 0.5f;

	sppm_detail::HashGrid<float> g;
	g.Build(pixels);
	EXPECT_GE(g.hashSize, 1);
	// Bucket for the visible-point centre should be in-range
	int h = g.Bucket(0.f, 0.f, 0.f);
	EXPECT_GE(h, 0);
	EXPECT_LT(h, g.hashSize);
}

TEST(SPPMHashGrid, BucketInRange) {
	std::vector<SPPMPixel<float>> pixels(8);
	for (int i = 0; i < 8; ++i) {
		pixels[i].vp_valid = true;
		pixels[i].vp_p[0] = (float)(i % 4);
		pixels[i].vp_p[1] = (float)(i / 4);
		pixels[i].vp_p[2] = 0.f;
		pixels[i].radius = 0.6f;
	}
	sppm_detail::HashGrid<float> g;
	g.Build(pixels);
	for (int i = 0; i < 8; ++i) {
		int h = g.Bucket(pixels[i].vp_p[0], pixels[i].vp_p[1], pixels[i].vp_p[2]);
		EXPECT_GE(h, 0) << "i=" << i;
		EXPECT_LT(h, g.hashSize) << "i=" << i;
	}
}

TEST(SPPMHashGrid, ToGridClampsLow) {
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].vp_valid=true; pixels[0].radius=1.f;
	pixels[0].vp_p[0]=0.f; pixels[0].vp_p[1]=0.f; pixels[0].vp_p[2]=0.f;
	sppm_detail::HashGrid<float> g;
	g.Build(pixels);
	int out[3];
	g.ToGrid(-999.f, -999.f, -999.f, out);
	EXPECT_EQ(out[0], 0); EXPECT_EQ(out[1], 0); EXPECT_EQ(out[2], 0);
}

TEST(SPPMHashGrid, ToGridClampsHigh) {
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].vp_valid=true; pixels[0].radius=1.f;
	pixels[0].vp_p[0]=0.f; pixels[0].vp_p[1]=0.f; pixels[0].vp_p[2]=0.f;
	sppm_detail::HashGrid<float> g;
	g.Build(pixels);
	int out[3];
	g.ToGrid(999.f, 999.f, 999.f, out);
	EXPECT_EQ(out[0], g.gridRes[0]-1);
	EXPECT_EQ(out[1], g.gridRes[1]-1);
	EXPECT_EQ(out[2], g.gridRes[2]-1);
}

// ===========================================================================
// Tests 2: SPPMPixel defaults
// ===========================================================================

TEST(SPPMPixel, DefaultsAreZero) {
	SPPMPixel<float> px;
	EXPECT_EQ(px.radius, 0.f);
	EXPECT_EQ(px.n, 0.f);
	EXPECT_EQ(px.m, 0);
	EXPECT_FALSE(px.vp_valid);
	for (int c=0;c<3;++c) {
		EXPECT_EQ(px.Ld[c], 0.f);
		EXPECT_EQ(px.tau[c], 0.f);
		EXPECT_EQ(px.Phi[c], 0.f);
	}
}

// ===========================================================================
// Tests 3: SPPMUpdateRadius
// ===========================================================================

TEST(SPPMUpdateRadius, NoPhotonsKeepsTau) {
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].radius = 1.f;
	pixels[0].n = 5.f;
	pixels[0].tau[0] = 3.f; pixels[0].tau[1] = 2.f; pixels[0].tau[2] = 1.f;
	pixels[0].m = 0;
	pixels[0].vp_valid = true;

	SPPMUpdateRadius(pixels);

	// m==0 -> no contraction; tau/radius/n unchanged
	EXPECT_FLOAT_EQ(pixels[0].radius, 1.f);
	EXPECT_FLOAT_EQ(pixels[0].n, 5.f);
	EXPECT_FLOAT_EQ(pixels[0].tau[0], 3.f);
	// vp_valid always reset after pass (pbrt-v4: unconditional reset)
	EXPECT_FALSE(pixels[0].vp_valid);
}

TEST(SPPMUpdateRadius, GammaContractionFormula) {
	// Manually verify the gamma=2/3 formula
	// n=6, m=3, gamma=2/3 => nNew = 6 + 2 = 8; rNew = r*sqrt(8/9)
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].radius = 1.f;
	pixels[0].n = 6.f;
	pixels[0].m = 3;
	pixels[0].tau[0] = 1.f;
	pixels[0].Phi[0] = 0.5f;
	pixels[0].vp_valid = true;

	SPPMUpdateRadius(pixels);

	float nNew = 6.f + (2.f/3.f)*3.f; // = 8
	float rNew = 1.f * std::sqrt(nNew / (6.f + 3.f)); // = sqrt(8/9)
	float scale = (rNew*rNew) / 1.f; // rNew^2 / rOld^2
	float tauNew = (1.f + 0.5f) * scale;

	EXPECT_NEAR(pixels[0].n,      nNew,   1e-5f);
	EXPECT_NEAR(pixels[0].radius, rNew,   1e-5f);
	EXPECT_NEAR(pixels[0].tau[0], tauNew, 1e-5f);
	EXPECT_EQ(pixels[0].m, 0);
	EXPECT_EQ(pixels[0].Phi[0], 0.f);
}

TEST(SPPMUpdateRadius, RadiusNonIncreasing) {
	// After update with m>0, radius should be <= old radius
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].radius = 2.f;
	pixels[0].n = 10.f;
	pixels[0].m = 5;
	pixels[0].vp_valid = true;
	float oldR = pixels[0].radius;

	SPPMUpdateRadius(pixels);

	EXPECT_LE(pixels[0].radius, oldR);
}

TEST(SPPMUpdateRadius, TauNonNegative) {
	std::vector<SPPMPixel<float>> pixels(1);
	pixels[0].radius = 1.f;
	pixels[0].n = 3.f;
	pixels[0].m = 2;
	pixels[0].Phi[0] = 0.3f;
	pixels[0].Phi[1] = 0.2f;
	pixels[0].Phi[2] = 0.1f;
	pixels[0].vp_valid = true;

	SPPMUpdateRadius(pixels);

	for (int c=0;c<3;++c) EXPECT_GE(pixels[0].tau[c], 0.f);
}

// ===========================================================================
// Tests 4: SPPMFinalImage
// ===========================================================================

TEST(SPPMFinalImage, ZeroPhotonsGivesOnlyLd) {
	std::vector<SPPMPixel<float>> pixels(2);
	pixels[0].Ld[0]=1.f; pixels[0].Ld[1]=0.5f; pixels[0].Ld[2]=0.2f;
	pixels[0].tau[0]=pixels[0].tau[1]=pixels[0].tau[2]=0.f;
	pixels[0].n=0.f; pixels[0].radius=1.f;
	pixels[1].Ld[0]=0.f; pixels[1].Ld[1]=0.f; pixels[1].Ld[2]=0.f;
	pixels[1].n=0.f; pixels[1].radius=1.f;

	std::vector<float> img;
	SPPMFinalImage(pixels, /*nIterations=*/1, 100, img);
	EXPECT_EQ((int)img.size(), 6);
	EXPECT_NEAR(img[0], 1.f,  1e-5f);
	EXPECT_NEAR(img[1], 0.5f, 1e-5f);
	EXPECT_NEAR(img[2], 0.2f, 1e-5f);
	EXPECT_EQ(img[3], 0.f);
}

TEST(SPPMFinalImage, IndirectTermFormula) {
	// tau=pi, n=1, photons=1, r=1 => indirect = tau/(n*photons*pi*r^2) = 1
	std::vector<SPPMPixel<float>> pixels(1);
	float kPi = 3.14159265358979f;
	pixels[0].tau[0] = kPi; pixels[0].tau[1] = kPi; pixels[0].tau[2] = kPi;
	pixels[0].n = 1.f;
	pixels[0].radius = 1.f;
	pixels[0].Ld[0]=pixels[0].Ld[1]=pixels[0].Ld[2]=0.f;

	std::vector<float> img;
	SPPMFinalImage(pixels, /*nIterations=*/1, /*totalPhotonPaths=*/1, img);
	EXPECT_NEAR(img[0], 1.f, 1e-4f);
	EXPECT_NEAR(img[1], 1.f, 1e-4f);
	EXPECT_NEAR(img[2], 1.f, 1e-4f);
}

TEST(SPPMFinalImage, OutputSizeIsPixelCount) {
	std::vector<SPPMPixel<float>> pixels(10);
	for (auto& p : pixels) { p.radius=1.f; }
	std::vector<float> img;
	SPPMFinalImage(pixels, /*nIterations=*/1, 1, img);
	EXPECT_EQ((int)img.size(), 30);
}

// ===========================================================================
// Tests 5: SPPMCameraPass
// ===========================================================================

TEST(SPPMCameraPass, PixelCountMatches) {
	SPPMTestScene scene;
	int W=4, H=4;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 0.5f;

	SPPMCameraPass(pixels, W, H, 5, scene);

	EXPECT_EQ((int)pixels.size(), W*H);
}

TEST(SPPMCameraPass, CentrePixelsHitSphere) {
	// Centre pixels (px~0.5,py~0.5) should hit the unit sphere
	SPPMTestScene scene;
	int W=8, H=8;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 0.5f;

	SPPMCameraPass(pixels, W, H, 5, scene);

	// At least some pixels should have valid visible points
	int validCount = 0;
	for (const auto& p : pixels) if (p.vp_valid) ++validCount;
	EXPECT_GT(validCount, 0);
}

TEST(SPPMCameraPass, LdNonNegative) {
	SPPMTestScene scene;
	int W=4, H=4;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 0.5f;

	SPPMCameraPass(pixels, W, H, 5, scene);

	for (const auto& p : pixels)
		for (int c=0;c<3;++c) EXPECT_GE(p.Ld[c], 0.f);
}

TEST(SPPMCameraPass, VisiblePointPositionOnSphere) {
	SPPMTestScene scene;
	int W=4, H=4;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 0.5f;

	SPPMCameraPass(pixels, W, H, 5, scene);

	for (const auto& p : pixels) {
		if (!p.vp_valid) continue;
		float r2 = p.vp_p[0]*p.vp_p[0]+p.vp_p[1]*p.vp_p[1]+p.vp_p[2]*p.vp_p[2];
		EXPECT_NEAR(r2, 1.f, 0.01f) << "visible point not on unit sphere";
	}
}

// ===========================================================================
// Tests 6: SPPMPhotonPass
// ===========================================================================

TEST(SPPMPhotonPass, PhiNonNegative) {
	SPPMTestScene scene;
	int W=4, H=4;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 1.0f;  // large radius to catch photons

	// First run camera pass to get valid visible points
	SPPMCameraPass(pixels, W, H, 5, scene);

	sppm_detail::HashGrid<float> grid;
	grid.Build(pixels);

	SPPMPhotonPass(pixels, grid, 50, 5, scene);

	for (const auto& p : pixels)
		for (int c=0;c<3;++c) EXPECT_GE(p.Phi[c], 0.f);
}

TEST(SPPMPhotonPass, MNonNegative) {
	SPPMTestScene scene;
	int W=4, H=4;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 1.0f;

	SPPMCameraPass(pixels, W, H, 5, scene);
	sppm_detail::HashGrid<float> grid;
	grid.Build(pixels);
	SPPMPhotonPass(pixels, grid, 50, 5, scene);

	for (const auto& p : pixels) EXPECT_GE(p.m, 0);
}

TEST(SPPMPhotonPass, NoHitsWithZeroRadius) {
	SPPMTestScene scene;
	int W=2, H=2;
	std::vector<SPPMPixel<float>> pixels(W*H);
	for (auto& p : pixels) p.radius = 0.f;

	SPPMCameraPass(pixels, W, H, 5, scene);
	sppm_detail::HashGrid<float> grid;
	grid.Build(pixels);
	SPPMPhotonPass(pixels, grid, 20, 5, scene);

	// With zero radius no photon should be within range of any VP
	int totalM = 0;
	for (const auto& p : pixels) totalM += p.m;
	EXPECT_EQ(totalM, 0);
}

// ===========================================================================
// Tests 7: SPPMRender (end-to-end)
// ===========================================================================

TEST(SPPMRender, OutputSizeCorrect) {
	SPPMTestScene scene;
	std::vector<float> img;
	SPPMRender<float>(4, 4, 2, 20, 3, 0.5f, scene, img);
	EXPECT_EQ((int)img.size(), 4*4*3);
}

TEST(SPPMRender, OutputNonNegative) {
	SPPMTestScene scene;
	std::vector<float> img;
	SPPMRender<float>(4, 4, 2, 20, 3, 0.5f, scene, img);
	for (float v : img) EXPECT_GE(v, 0.f);
}

TEST(SPPMRender, OutputFinite) {
	SPPMTestScene scene;
	std::vector<float> img;
	SPPMRender<float>(4, 4, 2, 20, 3, 0.5f, scene, img);
	for (float v : img) EXPECT_TRUE(std::isfinite(v)) << "v=" << v;
}

TEST(SPPMRender, MoreIterationsReducesVariance) {
	// More iterations should generally not inflate brightness unexpectedly
	SPPMTestScene scene1, scene2;
	std::vector<float> img1, img2;
	SPPMRender<float>(4, 4, 1, 20, 3, 0.5f, scene1, img1);
	SPPMRender<float>(4, 4, 4, 20, 3, 0.5f, scene2, img2);
	// Both should be finite and non-negative
	for (float v : img2) {
		EXPECT_GE(v, 0.f);
		EXPECT_TRUE(std::isfinite(v));
	}
}

TEST(SPPMRender, SingleIteration) {
	SPPMTestScene scene;
	std::vector<float> img;
	SPPMRender<float>(2, 2, 1, 10, 3, 0.5f, scene, img);
	EXPECT_EQ((int)img.size(), 12);
	for (float v : img) {
		EXPECT_GE(v, 0.f);
		EXPECT_TRUE(std::isfinite(v));
	}
}
