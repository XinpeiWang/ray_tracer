// light_path_tests.cpp
// Unit tests for src/shared/light_path.h  --  LightPathTrace<T,Scene,Film>
//
// Synthetic scene:
//   - Unit sphere (radius 1) centred at origin, Lambertian albedo 0.8
//   - Point / area light at (0, 0, 3)
//   - Camera at (0, 0, 5) looking along -z
//   - Film: 1x1 accumulation buffer (single pixel)
//
// Test strategy: because LightPathTrace *splats* into the film (it does not
// return a value), correctness is verified by inspecting the film buffer
// after one or many traces.

#include "gtest/gtest.h"
#include "../../src/shared/light_path.h"

#include <cmath>
#include <random>
#include <array>

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------
struct LPTestRNG {
	mutable std::mt19937 eng;
	mutable std::uniform_real_distribution<float> dist{0.f, 1.f};
	explicit LPTestRNG(uint32_t seed = 42) : eng(seed) {}
	float operator()() const { return dist(eng); }
};

// ---------------------------------------------------------------------------
// Minimal accumulation film
// ---------------------------------------------------------------------------
struct LPTestFilm {
	float pixels[4] = {};      // [0..2] = RGB sum, [3] = splat count
	void Splat(float /*px*/, float /*py*/, const float L[3]) {
		pixels[0] += L[0];
		pixels[1] += L[1];
		pixels[2] += L[2];
		pixels[3] += 1.f;
	}
	void Reset() { pixels[0]=pixels[1]=pixels[2]=pixels[3]=0.f; }
};

// ---------------------------------------------------------------------------
// Synthetic scene
// ---------------------------------------------------------------------------
struct LPSyntheticScene {

	// ---- Geometry ----------------------------------------------------------
	static bool sphere_hit(const float org[3], const float dir[3],
						   float t_max, BDPTHit<float>& hit) {
		float a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
		float b = 2*(org[0]*dir[0]+org[1]*dir[1]+org[2]*dir[2]);
		float c = org[0]*org[0]+org[1]*org[1]+org[2]*org[2]-1.f;
		float disc = b*b-4*a*c;
		if (disc < 0.f) return false;
		float sq = std::sqrt(disc);
		float t  = (-b-sq)/(2*a);
		if (t < 1e-4f) t = (-b+sq)/(2*a);
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

	bool Unoccluded(const float p0[3], const float p1[3]) const {
		float d[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]};
		float len=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if (len < 1e-6f) return true;
		d[0]/=len; d[1]/=len; d[2]/=len;
		BDPTHit<float> tmp{};
		return !sphere_hit(p0, d, len-1e-3f, tmp);
	}

	void SpawnRay(const BDPTHit<float>& hit, const float dir[3],
				  float new_o[3], float new_d[3]) const {
		constexpr float kEps = 1e-4f;
		new_o[0]=hit.p[0]+kEps*hit.geo_n[0];
		new_o[1]=hit.p[1]+kEps*hit.geo_n[1];
		new_o[2]=hit.p[2]+kEps*hit.geo_n[2];
		new_d[0]=dir[0]; new_d[1]=dir[1]; new_d[2]=dir[2];
	}

	bool BSDFIsNull(int) const { return false; }

	// ---- BSDF (Lambertian albedo 0.8, importance = radiance for Lambertian) -
	static constexpr float kAlbedoOverPi = 0.8f / 3.14159265358979f;

	void BSDFfImportance(int, const float*, const float wi[3],
						 const float n[3], float out[3]) const {
		float c = wi[0]*n[0]+wi[1]*n[1]+wi[2]*n[2];
		float v = (c > 0.f) ? kAlbedoOverPi * c : 0.f;
		out[0] = out[1] = out[2] = v;
	}

	bool BSDFSampleFImportance(int, const float*, const float n[3],
							   float u1, float u2,
							   float new_dir[3], float f_val[3],
							   float& pdf) const {
		// Cosine-hemisphere sampling (Lambertian is reciprocal)
		float cos_t = std::sqrt(u2);
		float sin_t = std::sqrt(std::max(0.f, 1.f-u2));
		float phi   = 2.f*3.14159265358979f*u1;
		float lx = sin_t*std::cos(phi), ly = sin_t*std::sin(phi), lz = cos_t;
		// Build ONB
		float tx, ty, tz;
		if (std::fabs(n[0]) > 0.9f) { tx=0.f; ty=1.f; tz=0.f; }
		else                        { tx=1.f; ty=0.f; tz=0.f; }
		float cx=ty*n[2]-tz*n[1], cy=tz*n[0]-tx*n[2], cz=tx*n[1]-ty*n[0];
		float cl=std::sqrt(cx*cx+cy*cy+cz*cz);
		tx=cx/cl; ty=cy/cl; tz=cz/cl;
		float bx=n[1]*tz-n[2]*ty, by=n[2]*tx-n[0]*tz, bz=n[0]*ty-n[1]*tx;
		new_dir[0]=lx*tx+ly*bx+lz*n[0];
		new_dir[1]=lx*ty+ly*by+lz*n[1];
		new_dir[2]=lx*tz+ly*bz+lz*n[2];
		f_val[0]=f_val[1]=f_val[2]=kAlbedoOverPi;
		pdf = cos_t / 3.14159265358979f;
		return pdf > 0.f;
	}

	// ---- Light: point light at (0,0,3), isotropic, power 1 -----------------
	// has_surface = false (point light)
	static constexpr float kLightPos[3] = {0.f, 0.f, 3.f};
	static constexpr float kLightPower  = 1.f;

	bool SampleLightEmission(float /*u_light*/,
							 float u0a, float u0b,
							 float u1a, float u1b,
							 LightEmissionSample<float>& les) const {
		// Point light: position is fixed, uniform sphere direction
		les.ray_o[0] = kLightPos[0];
		les.ray_o[1] = kLightPos[1];
		les.ray_o[2] = kLightPos[2];

		// Sample uniform sphere direction for emission
		float cos_theta = 1.f - 2.f*u1a;
		float sin_theta = std::sqrt(std::max(0.f, 1.f-cos_theta*cos_theta));
		float phi       = 2.f*3.14159265358979f*u1b;
		les.ray_d[0] = sin_theta*std::cos(phi);
		les.ray_d[1] = sin_theta*std::sin(phi);
		les.ray_d[2] = cos_theta;

		les.Le[0] = les.Le[1] = les.Le[2] = kLightPower / (4.f*3.14159265358979f);
		les.pdf_pos       = 1.f;    // delta distribution for point light
		les.pdf_dir       = 1.f / (4.f*3.14159265358979f);
		les.p_light       = 1.f;    // only one light
		les.abs_cos_theta = 1.f;    // point light: no surface normal
		les.has_surface   = false;
		les.light_id      = 0;
		(void)u0a; (void)u0b;
		return true;
	}

	// Point light has no surface -> SampleCameraConnection never called.
	bool SampleCameraConnection(const BDPTHit<float>& hit,
								float /*u1*/, float /*u2*/,
								CameraConnection<float>& cc) const {
		// Camera at (0,0,5); simple pinhole model: connect hit point to camera.
		static constexpr float kCamPos[3] = {0.f, 0.f, 5.f};

		float d[3] = {kCamPos[0]-hit.p[0], kCamPos[1]-hit.p[1], kCamPos[2]-hit.p[2]};
		float dist = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		if (dist < 1e-6f) return false;

		cc.wi[0]=d[0]/dist; cc.wi[1]=d[1]/dist; cc.wi[2]=d[2]/dist;
		cc.p_lens[0]=kCamPos[0]; cc.p_lens[1]=kCamPos[1]; cc.p_lens[2]=kCamPos[2];

		// Simple orthographic pixel: map hit.p.xy to raster (clamp to [0,1])
		cc.p_raster[0] = 0.5f + hit.p[0];
		cc.p_raster[1] = 0.5f + hit.p[1];

		// Wi: camera response -- the camera sensor faces along -z, so the
		// relevant cosine is dot(wi, camera_forward) where camera_forward = (0,0,-1).
		// A hit point in front of the camera (z < 5) has wi[2] > 0 (pointing toward
		// the camera at z=5), so cos_cam = wi[2] (positive for visible points).
		float cos_cam = std::max(0.f, cc.wi[2]);  // wi points toward camera (+z)
		cc.Wi[0] = cc.Wi[1] = cc.Wi[2] = cos_cam / (dist*dist + 1e-8f);
		cc.pdf = 1.f;
		return cc.Wi[0] > 0.f;
	}

	float LightPdfLi(int /*light_id*/, const float* /*p_lens*/,
					 const float* /*neg_wi*/) const {
		// Point light: PDF_Li = 0 (delta distribution; area-light path skipped)
		return 0.f;
	}
};

// ---------------------------------------------------------------------------
// Scene where BSDFIsNull always returns true (medium boundary)
// ---------------------------------------------------------------------------
struct NullBSDFLPScene : LPSyntheticScene {
	bool BSDFIsNull(int) const { return true; }
};

// ---------------------------------------------------------------------------
// Scene where SampleCameraConnection always returns false
// ---------------------------------------------------------------------------
struct NoConnectScene : LPSyntheticScene {
	bool SampleCameraConnection(const BDPTHit<float>&, float, float,
								CameraConnection<float>&) const {
		return false;
	}
};

// ---------------------------------------------------------------------------
// Scene where Unoccluded always returns false (everything blocked)
// ---------------------------------------------------------------------------
struct FullyBlockedScene : LPSyntheticScene {
	bool Unoccluded(const float*, const float*) const { return false; }
};

// ---------------------------------------------------------------------------
// Helper: run N light paths and return total splat count
// ---------------------------------------------------------------------------
template <typename SceneT>
static LPTestFilm run_traces(const SceneT& scene, int n, int max_depth,
							 uint32_t seed = 42) {
	LPTestFilm film;
	LPTestRNG rng(seed);
	auto rand1d = [&]() -> float { return rng(); };
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	for (int i = 0; i < n; ++i)
		LightPathTrace<float>(scene, film, max_depth, rand1d, rand2d);
	return film;
}

// ===========================================================================
// Tests
// ===========================================================================

// Light emission with no scene geometry -> no splat (ray misses sphere)
// since the point light is inside the sphere, *some* emission directions
// will hit the sphere, but most random directions will miss.
// We verify no NaN/Inf in the film after many traces.
TEST(LightPath, NoNanAfterManyTraces) {
	LPSyntheticScene scene;
	auto film = run_traces(scene, 500, 4);
	EXPECT_TRUE(std::isfinite(film.pixels[0]));
	EXPECT_TRUE(std::isfinite(film.pixels[1]));
	EXPECT_TRUE(std::isfinite(film.pixels[2]));
}

// Film accumulates non-negative values only
TEST(LightPath, FilmNonNegative) {
	LPSyntheticScene scene;
	auto film = run_traces(scene, 500, 4);
	EXPECT_GE(film.pixels[0], 0.f);
	EXPECT_GE(film.pixels[1], 0.f);
	EXPECT_GE(film.pixels[2], 0.f);
}

// At least some paths hit the sphere and connect to camera -> splat count > 0
// after enough samples (light at (0,0,3), sphere at origin radius 1).
TEST(LightPath, SomeSplatsOccur) {
	LPSyntheticScene scene;
	auto film = run_traces(scene, 1000, 4);
	EXPECT_GT(film.pixels[3], 0.f);
}

// Achromatic scene (grey BSDF, grey light) -> R == G == B splat totals
TEST(LightPath, OutputIsAchromatic) {
	LPSyntheticScene scene;
	auto film = run_traces(scene, 2000, 4, 7);
	if (film.pixels[0] > 1e-6f) {
		EXPECT_NEAR(film.pixels[1] / film.pixels[0], 1.f, 0.01f);
		EXPECT_NEAR(film.pixels[2] / film.pixels[0], 1.f, 0.01f);
	}
}

// Deterministic: same seed => identical film
TEST(LightPath, Deterministic) {
	LPSyntheticScene scene;
	auto f1 = run_traces(scene, 200, 4, 42);
	auto f2 = run_traces(scene, 200, 4, 42);
	EXPECT_FLOAT_EQ(f1.pixels[0], f2.pixels[0]);
	EXPECT_FLOAT_EQ(f1.pixels[1], f2.pixels[1]);
	EXPECT_FLOAT_EQ(f1.pixels[2], f2.pixels[2]);
}

// max_depth == 0: path is terminated at first real hit -> no BSDF bounces,
// but the camera-connection at depth 0 is still attempted.
// With max_depth=0 the "if (depth++ == 0) break" fires immediately after
// the first camera connection attempt.  Result must still be non-negative.
TEST(LightPath, MaxDepthZeroNonNegative) {
	LPSyntheticScene scene;
	auto film = run_traces(scene, 500, 0);
	EXPECT_GE(film.pixels[0], 0.f);
	EXPECT_GE(film.pixels[1], 0.f);
	EXPECT_GE(film.pixels[2], 0.f);
}

// Null BSDF scene: every surface hit is skipped (medium boundary).
// A ray that keeps spawning through a null BSDF will eventually miss and
// produce zero splat; film should stay at zero.
TEST(LightPath, NullBSDFNoSplat) {
	NullBSDFLPScene scene;
	LPTestFilm film;
	LPTestRNG rng(42);
	auto rand1d = [&]() -> float { return rng(); };
	auto rand2d = [&]() -> std::pair<float,float> { return {rng(), rng()}; };
	for (int i = 0; i < 300; ++i)
		LightPathTrace<float>(scene, film, 4, rand1d, rand2d);
	// No real BSDF -> camera connection always hits the null path -> 0 splats
	EXPECT_FLOAT_EQ(film.pixels[3], 0.f);
}

// No camera connection -> film stays empty
TEST(LightPath, NoConnectionNoSplat) {
	NoConnectScene scene;
	auto film = run_traces(scene, 300, 4);
	EXPECT_FLOAT_EQ(film.pixels[0], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[1], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[2], 0.f);
}

// Fully blocked scene -> no unoccluded paths -> film stays empty
TEST(LightPath, FullyBlockedNoSplat) {
	FullyBlockedScene scene;
	auto film = run_traces(scene, 300, 4);
	EXPECT_FLOAT_EQ(film.pixels[0], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[1], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[2], 0.f);
}

// Higher max_depth accumulates >= lower max_depth (more paths can contribute)
TEST(LightPath, HigherDepthAccumulatesMore) {
	LPSyntheticScene scene;
	auto f1 = run_traces(scene, 2000, 1, 13);
	auto f4 = run_traces(scene, 2000, 4, 13);
	// With more bounces the total accumulated energy must be >= single-bounce
	EXPECT_GE(f4.pixels[0] + f4.pixels[1] + f4.pixels[2],
			  f1.pixels[0] + f1.pixels[1] + f1.pixels[2] - 1e-5f);
}

// Degenerate emission (Le = 0) -> no splat
struct ZeroLeScene : LPSyntheticScene {
	bool SampleLightEmission(float u_l, float a, float b, float c, float d,
							 LightEmissionSample<float>& les) const {
		bool ok = LPSyntheticScene::SampleLightEmission(u_l,a,b,c,d,les);
		les.Le[0]=les.Le[1]=les.Le[2]=0.f;  // force zero emission
		return ok;
	}
};

TEST(LightPath, ZeroLeNoSplat) {
	ZeroLeScene scene;
	auto film = run_traces(scene, 300, 4);
	EXPECT_FLOAT_EQ(film.pixels[0], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[1], 0.f);
	EXPECT_FLOAT_EQ(film.pixels[2], 0.f);
}
