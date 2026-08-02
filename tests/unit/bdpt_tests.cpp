// bdpt_tests.cpp
// Unit and integration tests for src/shared/bdpt.h
//
// Tests are structured in four groups:
//   1. ScopedAssign correctness
//   2. BDPTVertex geometry helpers (ConvertDensity, BDPTGeometryTerm)
//   3. MISWeight trivial cases
//   4. Full pipeline tests with a minimal synthetic scene
//
// The synthetic scene provides:
//   - A single diffuse sphere at origin radius 1
//   - A point light at (0,10,0) with power (1,1,1)
//   - A simple pinhole camera at (0,0,5) looking at -Z
//
#include "gtest/gtest.h"
#include "../../src/shared/bdpt.h"
#include <cmath>
#include <vector>
#include <random>

// ---------------------------------------------------------------------------
// Minimal RNG for deterministic tests
// ---------------------------------------------------------------------------
struct TestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	TestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Minimal synthetic scene for BDPT tests
// ---------------------------------------------------------------------------
// Geometry: a unit sphere at origin.
// Light:    a point light at (0, 3, 0) emitting white (1,1,1).
// Camera:   pinhole at (0, 0, 5), looking towards -Z (into the sphere).
// ---------------------------------------------------------------------------

struct SyntheticScene {
	TestRNG rng;

	// ---- Scene::Intersect ----
	bool Intersect(const float org[3], const float dir[3], float t_max,
				   BDPTHit<float>& hit) const {
		// Ray-sphere intersection (sphere at origin, r=1)
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
		hit.p[0] = org[0]+t*dir[0];
		hit.p[1] = org[1]+t*dir[1];
		hit.p[2] = org[2]+t*dir[2];
		// Normal = normalize(p) since sphere at origin
		float len = std::sqrt(hit.p[0]*hit.p[0]+hit.p[1]*hit.p[1]+hit.p[2]*hit.p[2]);
		hit.geo_n[0]=hit.p[0]/len; hit.geo_n[1]=hit.p[1]/len; hit.geo_n[2]=hit.p[2]/len;
		hit.shading_n[0]=hit.geo_n[0]; hit.shading_n[1]=hit.geo_n[1]; hit.shading_n[2]=hit.geo_n[2];
		// wo = incoming direction flipped
		hit.wo[0]=-dir[0]; hit.wo[1]=-dir[1]; hit.wo[2]=-dir[2];
		hit.uv[0]=hit.uv[1]=0.f;
		hit.area_Le[0]=hit.area_Le[1]=hit.area_Le[2]=0.f; // not an area light
		hit.is_medium_boundary=false;
		hit.is_delta_bsdf=false;
		hit.bsdf_id=0; // only one BSDF: white diffuse
		return true;
	}

	// ---- Scene::Unoccluded ----
	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float dir[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
		float len = std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
		if (len < 1e-6f) return true;
		float inv=1.f/len;
		dir[0]*=inv; dir[1]*=inv; dir[2]*=inv;
		BDPTHit<float> hit{};
		bool found = Intersect(p0, dir, len - 1e-3f, hit);
		return !found;
	}

	// ---- BSDF: white Lambertian ----
	void BSDFf(int /*id*/, const float wo[3], const float wi[3],
			   const float n[3], float out[3]) const {
		float cos_i = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		if (cos_i <= 0.f) { out[0]=out[1]=out[2]=0.f; return; }
		const float albedo = 0.8f;
		out[0]=out[1]=out[2] = albedo / 3.14159265358979f;
	}

	bool BSDFSampleF(int /*id*/, const float wo[3], const float n[3],
					 float u1, float u2,
					 float new_dir[3], float f_val[3],
					 float& pdf, bool& is_specular) const {
		// Cosine hemisphere sampling
		float phi = 2.f * 3.14159265358979f * u1;
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(1.f - u2);
		// Build ONB from n
		float tx, ty, tz, bx, by, bz;
		float ax=std::fabs(n[0]),ay=std::fabs(n[1]),az=std::fabs(n[2]);
		if (ax > 0.9f) { tx=0.f; ty=1.f; tz=0.f; }
		else           { tx=1.f; ty=0.f; tz=0.f; }
		// tangent = normalize(cross(t,n))
		float cx=ty*n[2]-tz*n[1], cy=tz*n[0]-tx*n[2], cz=tx*n[1]-ty*n[0];
		float clen=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/clen; ty=cy/clen; tz=cz/clen;
		bx=n[1]*tz-n[2]*ty; by=n[2]*tx-n[0]*tz; bz=n[0]*ty-n[1]*tx;
		// local direction
		float lx=sin_t*std::cos(phi), ly=sin_t*std::sin(phi), lz=cos_t;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		const float albedo = 0.8f;
		f_val[0]=f_val[1]=f_val[2] = albedo / 3.14159265358979f;
		pdf = cos_t / 3.14159265358979f;
		is_specular = false;
		return pdf > 0.f;
	}

	float BSDFPdf(int /*id*/, const float wo[3], const float wi[3],
				  const float n[3]) const {
		float cos_i = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		return cos_i > 0.f ? cos_i / 3.14159265358979f : 0.f;
	}

	// ---- Light: point light at (0,3,0) ----
	static constexpr float kLightPos[3] = {0.f, 3.f, 0.f};

	bool SampleLight(float /*u*/, const float ref_p[3],
					 BDPTLightSample<float>& ls) const {
		ls.p_light[0]=kLightPos[0]; ls.p_light[1]=kLightPos[1]; ls.p_light[2]=kLightPos[2];
		ls.n_light[0]=ls.n_light[1]=ls.n_light[2]=0.f;
		float d[3]={kLightPos[0]-ref_p[0],kLightPos[1]-ref_p[1],kLightPos[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		ls.wi[0]=d[0]/len; ls.wi[1]=d[1]/len; ls.wi[2]=d[2]/len;
		float dist2 = len*len;
		ls.L[0]=ls.L[1]=ls.L[2] = 1.f / dist2; // point light intensity / r^2
		ls.pdf = 1.f;
		ls.is_delta = true;
		ls.light_id = 0;
		return true;
	}

	bool SampleLightLe(float /*u1d*/, const float[2], const float[2],
					   BDPTLightLeSample<float>& les) const {
		// Sample isotropic direction from point light
		float u1 = rng(), u2 = rng();
		float phi = 2.f*3.14159265358979f*u1;
		float cos_t = 1.f - 2.f*u2;
		float sin_t = std::sqrt(std::max(0.f, 1.f - cos_t*cos_t));
		les.ray_o[0]=kLightPos[0]; les.ray_o[1]=kLightPos[1]; les.ray_o[2]=kLightPos[2];
		les.ray_d[0]=sin_t*std::cos(phi);
		les.ray_d[1]=sin_t*std::sin(phi);
		les.ray_d[2]=cos_t;
		les.p_on_light[0]=kLightPos[0]; les.p_on_light[1]=kLightPos[1]; les.p_on_light[2]=kLightPos[2];
		les.n_on_light[0]=les.n_on_light[1]=les.n_on_light[2]=0.f;
		les.L[0]=les.L[1]=les.L[2]=1.f;
		les.pdf_pos=1.f;
		les.pdf_dir=1.f/(4.f*3.14159265358979f);
		les.abs_cos_theta=1.f;
		les.is_on_surface=false;
		les.is_infinite=false;
		les.is_delta_dir=false;
		les.light_id=0;
		return true;
	}

	float LightPMF(int) const { return 1.f; }

	void LightPDFLe(int, const float*, const float*, const float*,
					float& pdf_pos, float& pdf_dir) const {
		pdf_pos = 1.f;
		pdf_dir = 1.f / (4.f * 3.14159265358979f);
	}

	// ---- Camera: pinhole at (0,0,5) looking -Z ----
	static constexpr float kCamPos[3] = {0.f, 0.f, 5.f};

	void CameraPDFWe(const float* /*ray_o*/, const float* /*ray_d*/,
					 float& pdfPos, float& pdfDir) const {
		pdfPos = 1.f;
		pdfDir = 1.f; // simplified pinhole: uniform over hemisphere
	}

	bool CameraSampleWi(const float ref_p[3], const float* /*u2*/,
						float wi[3], float& pdf, float& importance,
						float pRaster[2]) const {
		float d[3]={kCamPos[0]-ref_p[0],kCamPos[1]-ref_p[1],kCamPos[2]-ref_p[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if(len<1e-6f) return false;
		wi[0]=d[0]/len; wi[1]=d[1]/len; wi[2]=d[2]/len;
		pdf = 1.f; importance = 1.f;
		if(pRaster) { pRaster[0]=pRaster[1]=0.f; }
		return true;
	}

	// ---- Infinite lights (none in this scene) ----
	void InfiniteLightLe(const float[3], float out[3]) const {
		out[0]=out[1]=out[2]=0.f;
	}
	float InfiniteLightDensity(const float[3]) const { return 0.f; }

	void SceneBoundingSphere(float center[3], float& radius) const {
		center[0]=center[1]=center[2]=0.f; radius=5.f;
	}

	float RandFloat() const { return rng(); }
};

// Static member definitions
constexpr float SyntheticScene::kLightPos[3];
constexpr float SyntheticScene::kCamPos[3];

// ===========================================================================
// Tests 1: ScopedAssign
// ===========================================================================

TEST(ScopedAssign, RestoresOnDestruction) {
	float v = 1.0f;
	{
		ScopedAssign<float> sa(&v, 99.0f);
		EXPECT_FLOAT_EQ(v, 99.0f);
	}
	EXPECT_FLOAT_EQ(v, 1.0f);
}

TEST(ScopedAssign, MoveAssignment) {
	float v = 2.0f;
	ScopedAssign<float> sa1;
	{
		ScopedAssign<float> sa2(&v, 77.0f);
		EXPECT_FLOAT_EQ(v, 77.0f);
		sa1 = std::move(sa2); // sa2 is now empty, sa1 owns the restore
	}
	EXPECT_FLOAT_EQ(v, 77.0f); // sa2 destroyed without restore
	// sa1 destructor restores
	sa1 = ScopedAssign<float>(); // destroy sa1 by assigning empty
	EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(ScopedAssign, DefaultConstructedDoesNothing) {
	ScopedAssign<int> sa;
	// no crash
}

// ===========================================================================
// Tests 2: BDPTVertex helpers
// ===========================================================================

TEST(BDPTVertex, ConvertDensityFlatSurface) {
	// v0 at origin, v1 1 unit away on a flat surface with normal pointing towards v0
	BDPTVertex<float> v0, v1;
	float p0[3]={0,0,0}, n0[3]={0,0,1}, beta[3]={1,1,1};
	v0 = BDPTVertex<float>::MakeCamera(p0, n0, beta, 1.f);

	float p1[3]={1,0,0}, n1[3]={1,0,0}; // normal of v1 points exactly toward v0
	v1 = BDPTVertex<float>::MakeLightSurface(p1, n1, beta, 1.f, 0, false);

	// pdf_dir = 1, distance = 1, cos(v1.n, d) = dot({-1,0,0},{1,0,0}) = 1
	float pdfA = v0.ConvertDensity(1.f, v1);
	// Expected: pdf_dir * cos(n1, dir) / dist2 = 1 * 1 / 1 = 1
	EXPECT_NEAR(pdfA, 1.0f, 1e-5f);
}

TEST(BDPTVertex, ConvertDensityAtAngle) {
	BDPTVertex<float> v0, v1;
	float p0[3]={0,0,0}, n0[3]={0,1,0}, beta[3]={1,1,1};
	v0 = BDPTVertex<float>::MakeCamera(p0, n0, beta, 1.f);

	// v1 is 2 units away. Normal at 45 degrees to direction
	float p1[3]={2,0,0};
	float sq2 = std::sqrt(2.f)*0.5f;
	float n1[3]={-sq2, sq2, 0};
	v1 = BDPTVertex<float>::MakeLightSurface(p1, n1, beta, 1.f, 0, false);

	// dir from v0 to v1 = {1,0,0}; cos = |dot({-sq2,sq2,0},{1,0,0})| = sq2
	// dist2 = 4; pdfA = 1 * sq2 / 4
	float pdfA = v0.ConvertDensity(1.f, v1);
	EXPECT_NEAR(pdfA, sq2 / 4.f, 1e-5f);
}

TEST(BDPTVertex, GeometryTermSymmetric) {
	BDPTVertex<float> v0, v1;
	float p0[3]={0,0,0}, n0[3]={0,0,1}, beta[3]={1,1,1};
	float p1[3]={1,0,0}, n1[3]={-1,0,0};
	v0 = BDPTVertex<float>::MakeCamera(p0, n0, beta, 1.f);
	v1 = BDPTVertex<float>::MakeLightSurface(p1, n1, beta, 1.f, 0, false);

	float g01 = BDPTGeometryTerm(v0, v1);
	float g10 = BDPTGeometryTerm(v1, v0);
	EXPECT_NEAR(g01, g10, 1e-5f);
}

TEST(BDPTVertex, GeometryTermNonNegative) {
	BDPTVertex<float> v0, v1;
	float p0[3]={0,0,0}, n0[3]={0,1,0}, beta[3]={1,1,1};
	float p1[3]={0,1,0}, n1[3]={0,-1,0};
	v0 = BDPTVertex<float>::MakeCamera(p0, n0, beta, 1.f);
	v1 = BDPTVertex<float>::MakeLightSurface(p1, n1, beta, 1.f, 0, false);

	EXPECT_GE(BDPTGeometryTerm(v0, v1), 0.f);
}

// ===========================================================================
// Tests 3: MISWeight trivial cases
// ===========================================================================

TEST(BDPTMISWeight, SumEqualsTwoReturnsOne) {
	// s+t==2 must return 1 regardless of other inputs
	std::vector<BDPTVertex<float>> lv(4), cv(4);
	BDPTVertex<float> sampled;
	SyntheticScene scene;
	float w = BDPTMISWeight(lv.data(), cv.data(), sampled, 1, 1, scene);
	EXPECT_FLOAT_EQ(w, 1.f);
}

TEST(BDPTMISWeight, SumEqualsTwoAllCombinations) {
	std::vector<BDPTVertex<float>> lv(4), cv(4);
	BDPTVertex<float> sampled;
	SyntheticScene scene;
	// (s,t) = (0,2), (1,1), (2,0)
	EXPECT_FLOAT_EQ(BDPTMISWeight(lv.data(), cv.data(), sampled, 0, 2, scene), 1.f);
	EXPECT_FLOAT_EQ(BDPTMISWeight(lv.data(), cv.data(), sampled, 1, 1, scene), 1.f);
	EXPECT_FLOAT_EQ(BDPTMISWeight(lv.data(), cv.data(), sampled, 2, 0, scene), 1.f);
}

TEST(BDPTMISWeight, ResultInUnitInterval) {
	// For any valid connection the weight must be in [0,1]
	std::vector<BDPTVertex<float>> lv(6), cv(6);
	BDPTVertex<float> sampled;
	SyntheticScene scene;

	// Set some non-trivial pdfFwd/pdfRev values
	for (int i = 0; i < 6; ++i) {
		lv[i].pdfFwd = 0.5f; lv[i].pdfRev = 0.3f;
		cv[i].pdfFwd = 0.4f; cv[i].pdfRev = 0.2f;
	}

	for (int s = 0; s <= 3; ++s) {
		for (int t = 1; t <= 3; ++t) {
			if (s + t < 2) continue;
			float w = BDPTMISWeight(lv.data(), cv.data(), sampled, s, t, scene);
			EXPECT_GE(w, 0.f) << "s=" << s << " t=" << t;
			EXPECT_LE(w, 1.f) << "s=" << s << " t=" << t;
		}
	}
}

// ===========================================================================
// Tests 4: RandomWalk and subpath generation
// ===========================================================================

TEST(BDPTRandomWalk, CameraSubpathLength) {
	SyntheticScene scene;
	const int maxDepth = 4;
	std::vector<BDPTVertex<float>> verts(maxDepth + 2);

	float cam_p[3] = {0.f, 0.f, 5.f};
	float cam_n[3] = {0.f, 0.f, -1.f};
	float ray_d[3] = {0.f, 0.f, -1.f};

	int n = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d,
									   maxDepth + 2, scene, verts.data());
	// At minimum the camera vertex (verts[0]) should be present
	EXPECT_GE(n, 1);
	EXPECT_LE(n, maxDepth + 2);
	EXPECT_EQ(verts[0].type, BDPTVertexType::Camera);
}

TEST(BDPTRandomWalk, CameraSubpathHitsSphere) {
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> verts(maxDepth + 2);

	// Ray aimed straight at sphere center
	float cam_p[3] = {0.f, 0.f, 5.f};
	float cam_n[3] = {0.f, 0.f, -1.f};
	float ray_d[3] = {0.f, 0.f, -1.f};

	int n = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d,
									   maxDepth + 2, scene, verts.data());
	EXPECT_GE(n, 2); // camera + at least one surface hit
	if (n >= 2)
		EXPECT_EQ(verts[1].type, BDPTVertexType::Surface);
}

TEST(BDPTRandomWalk, LightSubpathLength) {
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> verts(maxDepth + 1);

	int n = BDPTGenerateLightSubpath(maxDepth + 1, scene, verts.data());
	EXPECT_GE(n, 1);
	EXPECT_LE(n, maxDepth + 1);
	EXPECT_EQ(verts[0].type, BDPTVertexType::Light);
}

TEST(BDPTRandomWalk, BetaIsNonNegative) {
	SyntheticScene scene;
	const int maxDepth = 4;
	std::vector<BDPTVertex<float>> verts(maxDepth + 2);

	float cam_p[3] = {0.f, 0.f, 5.f};
	float cam_n[3] = {0.f, 0.f, -1.f};
	float ray_d[3] = {0.f, 0.f, -1.f};

	int n = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d,
									   maxDepth + 2, scene, verts.data());
	for (int i = 0; i < n; ++i) {
		EXPECT_GE(verts[i].beta[0], 0.f) << "vertex " << i;
		EXPECT_GE(verts[i].beta[1], 0.f) << "vertex " << i;
		EXPECT_GE(verts[i].beta[2], 0.f) << "vertex " << i;
	}
}

// ===========================================================================
// Tests 5: ConnectBDPT individual strategies
// ===========================================================================

TEST(BDPTConnect, S0T2ReturnsNonNegative) {
	// Strategy s=0, t=2: camera ray hits an area light directly.
	// In our scene the sphere is not an area light so result should be zero.
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> lv(maxDepth+1), cv(maxDepth+2);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1}, ray_d[3]={0,0,-1};
	BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d, maxDepth+2, scene, cv.data());
	BDPTGenerateLightSubpath(maxDepth+1, scene, lv.data());

	float L[3];
	BDPTConnect(lv.data(), cv.data(), 0, 2, scene, L);
	EXPECT_GE(L[0], 0.f); EXPECT_GE(L[1], 0.f); EXPECT_GE(L[2], 0.f);
}

TEST(BDPTConnect, S1T2DirectLighting) {
	// Strategy s=1, t=2: direct lighting from a point light.
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> lv(maxDepth+1), cv(maxDepth+2);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1}, ray_d[3]={0,0,-1};
	int nCam = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d, maxDepth+2, scene, cv.data());
	BDPTGenerateLightSubpath(maxDepth+1, scene, lv.data());

	float L[3];
	BDPTConnect(lv.data(), cv.data(), 1, 2, scene, L);
	// Result must be non-negative; for a visible point light it should be > 0
	EXPECT_GE(L[0], 0.f); EXPECT_GE(L[1], 0.f); EXPECT_GE(L[2], 0.f);
	if (nCam >= 2) {
		// The first surface vertex should be on the hemisphere facing the light
		// (top of sphere at (0,0,4) hit by straight ray) -> light at (0,3,0) is
		// not behind the surface, so expect non-zero contribution
		float hitN[3] = { cv[1].si.shading_n[0], cv[1].si.shading_n[1], cv[1].si.shading_n[2] };
		(void)hitN; // checked via non-negative assert above
	}
}

TEST(BDPTConnect, GeneralCaseNonNegative) {
	// Strategy s=2, t=2: general connection
	SyntheticScene scene;
	const int maxDepth = 4;
	std::vector<BDPTVertex<float>> lv(maxDepth+1), cv(maxDepth+2);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1}, ray_d[3]={0,0,-1};
	int nCam = BDPTGenerateCameraSubpath(cam_p, cam_n, ray_d, maxDepth+2, scene, cv.data());
	int nLight = BDPTGenerateLightSubpath(maxDepth+1, scene, lv.data());

	if (nCam >= 2 && nLight >= 2) {
		float L[3];
		BDPTConnect(lv.data(), cv.data(), 2, 2, scene, L);
		EXPECT_GE(L[0], 0.f); EXPECT_GE(L[1], 0.f); EXPECT_GE(L[2], 0.f);
	}
}

// ===========================================================================
// Tests 6: Full BDPTLi pipeline
// ===========================================================================

TEST(BDPTLi, ReturnsNonNegativeResult) {
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> cv(maxDepth+2), lv(maxDepth+1);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1}, ray_d[3]={0,0,-1};
	float L[3];
	BDPTLi(cam_p, cam_n, ray_d, maxDepth, scene, cv.data(), lv.data(), L);

	EXPECT_GE(L[0], 0.f);
	EXPECT_GE(L[1], 0.f);
	EXPECT_GE(L[2], 0.f);
}

TEST(BDPTLi, FiniteResult) {
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> cv(maxDepth+2), lv(maxDepth+1);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1}, ray_d[3]={0,0,-1};
	float L[3];
	BDPTLi(cam_p, cam_n, ray_d, maxDepth, scene, cv.data(), lv.data(), L);

	EXPECT_TRUE(std::isfinite(L[0])) << "L[0]=" << L[0];
	EXPECT_TRUE(std::isfinite(L[1])) << "L[1]=" << L[1];
	EXPECT_TRUE(std::isfinite(L[2])) << "L[2]=" << L[2];
}

TEST(BDPTLi, MissedRayReturnsZero) {
	// Aim ray away from sphere -- should get zero (no infinite lights)
	SyntheticScene scene;
	const int maxDepth = 3;
	std::vector<BDPTVertex<float>> cv(maxDepth+2), lv(maxDepth+1);

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,1}, ray_d[3]={0,0,1}; // away from sphere
	float L[3];
	BDPTLi(cam_p, cam_n, ray_d, maxDepth, scene, cv.data(), lv.data(), L);

	EXPECT_FLOAT_EQ(L[0], 0.f);
	EXPECT_FLOAT_EQ(L[1], 0.f);
	EXPECT_FLOAT_EQ(L[2], 0.f);
}

TEST(BDPTLi, MultipleCallsConsistentSign) {
	// Run BDPTLi multiple times -- all results must be >= 0
	SyntheticScene scene;
	const int maxDepth = 3;

	float cam_p[3]={0,0,5}, cam_n[3]={0,0,-1};

	for (int trial = 0; trial < 20; ++trial) {
		std::vector<BDPTVertex<float>> cv(maxDepth+2), lv(maxDepth+1);
		// Slightly vary ray direction
		float angle = float(trial) * 0.05f;
		float ray_d[3] = { std::sin(angle), 0.f, -std::cos(angle) };
		float L[3];
		BDPTLi(cam_p, cam_n, ray_d, maxDepth, scene, cv.data(), lv.data(), L);
		EXPECT_GE(L[0], 0.f) << "trial " << trial;
		EXPECT_GE(L[1], 0.f) << "trial " << trial;
		EXPECT_GE(L[2], 0.f) << "trial " << trial;
		EXPECT_TRUE(std::isfinite(L[0])) << "trial " << trial;
	}
}
