// cameras_tests.cpp — unit tests for src/shared/cameras.h
// Validates OrthographicCamera, PerspectiveCamera, SphericalCamera
// against expected pbrt-v4 ray generation behavior.

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/cameras_new.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float ray_len(const CameraRayResult<float>& r) {
	float dx = r.direction.x, dy = r.direction.y, dz = r.direction.z;
	return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Build a default identity camera_to_world (camera == world)
static Mat4<float> identity() { return Mat4<float>{}; }

// Build a standard screen window for given resolution
static void sw(int rx, int ry, float& xmin, float& xmax, float& ymin, float& ymax) {
	compute_screen_window<float>(rx, ry, xmin, xmax, ymin, ymax);
}

// ---------------------------------------------------------------------------
// Mat4 tests
// ---------------------------------------------------------------------------
TEST(Cameras, Mat4Identity) {
	Mat4<float> m;
	auto p = m.transform_point(1.f, 2.f, 3.f);
	EXPECT_NEAR(p.x, 1.f, 1e-5f);
	EXPECT_NEAR(p.y, 2.f, 1e-5f);
	EXPECT_NEAR(p.z, 3.f, 1e-5f);
}

TEST(Cameras, Mat4Inverse) {
	Mat4<float> m = make_translate<float>(3.f, -1.f, 2.f);
	Mat4<float> inv = m.inverse();
	Mat4<float> prod = m * inv;
	// Product should be identity
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			EXPECT_NEAR(prod.m[i][j], (i==j) ? 1.f : 0.f, 1e-4f) << "i=" << i << " j=" << j;
}

TEST(Cameras, Mat4TranslatePoint) {
	Mat4<float> m = make_translate<float>(1.f, 2.f, 3.f);
	auto p = m.transform_point(0.f, 0.f, 0.f);
	EXPECT_NEAR(p.x, 1.f, 1e-5f);
	EXPECT_NEAR(p.y, 2.f, 1e-5f);
	EXPECT_NEAR(p.z, 3.f, 1e-5f);
}

TEST(Cameras, Mat4TranslateVec) {
	// Vectors are not translated
	Mat4<float> m = make_translate<float>(1.f, 2.f, 3.f);
	auto v = m.transform_vec(1.f, 0.f, 0.f);
	EXPECT_NEAR(v.x, 1.f, 1e-5f);
	EXPECT_NEAR(v.y, 0.f, 1e-5f);
	EXPECT_NEAR(v.z, 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Screen window helper
// ---------------------------------------------------------------------------
TEST(Cameras, ScreenWindowSquare) {
	float xmin, xmax, ymin, ymax;
	sw(512, 512, xmin, xmax, ymin, ymax);
	EXPECT_NEAR(xmin, -1.f, 1e-5f);
	EXPECT_NEAR(xmax,  1.f, 1e-5f);
	EXPECT_NEAR(ymin, -1.f, 1e-5f);
	EXPECT_NEAR(ymax,  1.f, 1e-5f);
}

TEST(Cameras, ScreenWindowWide) {
	float xmin, xmax, ymin, ymax;
	sw(800, 400, xmin, xmax, ymin, ymax);
	EXPECT_NEAR(xmin, -2.f, 1e-5f);
	EXPECT_NEAR(xmax,  2.f, 1e-5f);
	EXPECT_NEAR(ymin, -1.f, 1e-5f);
	EXPECT_NEAR(ymax,  1.f, 1e-5f);
}

TEST(Cameras, ScreenWindowTall) {
	float xmin, xmax, ymin, ymax;
	sw(400, 800, xmin, xmax, ymin, ymax);
	EXPECT_NEAR(xmin, -1.f,   1e-5f);
	EXPECT_NEAR(xmax,  1.f,   1e-5f);
	EXPECT_NEAR(ymin, -2.f,   1e-5f);
	EXPECT_NEAR(ymax,  2.f,   1e-5f);
}

// ---------------------------------------------------------------------------
// OrthographicCamera
// ---------------------------------------------------------------------------
static OrthographicCamera<float> make_ortho(int rx=512, int ry=512) {
	float xmin, xmax, ymin, ymax;
	compute_screen_window<float>(rx, ry, xmin, xmax, ymin, ymax);
	return OrthographicCamera<float>(xmin, xmax, ymin, ymax, rx, ry);
}

TEST(Cameras, OrthoRayIsUnitLength) {
	auto cam = make_ortho();
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f, 0.f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray_len(ray), 1.f, 1e-5f);
}

TEST(Cameras, OrthoRayDirectionIsForward) {
	// Orthographic rays always point along +Z in camera (= world with identity)
	auto cam = make_ortho();
	CameraSample<float> s{100.f, 200.f, 0.5f, 0.5f, 0.f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray.direction.x, 0.f, 1e-5f);
	EXPECT_NEAR(ray.direction.y, 0.f, 1e-5f);
	EXPECT_NEAR(ray.direction.z, 1.f, 1e-5f);
}

TEST(Cameras, OrthoRayOriginVariesWithPixel) {
	auto cam = make_ortho();
	CameraSample<float> s1{0.f,   0.f,   0.5f, 0.5f, 0.f};
	CameraSample<float> s2{511.f, 511.f, 0.5f, 0.5f, 0.f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	// Origins differ but directions are identical
	EXPECT_NE(r1.origin.x, r2.origin.x);
	EXPECT_NEAR(r1.direction.z - r2.direction.z, 0.f, 1e-5f);
}

TEST(Cameras, OrthoRayWeight) {
	auto cam = make_ortho();
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	EXPECT_NEAR(cam.generate_ray(s).weight, 1.f, 1e-6f);
}

TEST(Cameras, OrthoRayTransformedOrigin) {
	// Camera translated by (0,0,5) in world
	Mat4<float> cam2world = make_translate<float>(0.f, 0.f, 5.f);
	float xmin=-1.f, xmax=1.f, ymin=-1.f, ymax=1.f;
	OrthographicCamera<float> cam(xmin, xmax, ymin, ymax, 512, 512, cam2world);
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f, 0.f};
	auto ray = cam.generate_ray(s);
	// z-origin should be shifted by 5
	EXPECT_GT(ray.origin.z, 4.f);
}

TEST(Cameras, OrthoNoDOFWithZeroRadius) {
	// Two different lens samples should produce same ray when no DOF
	auto cam = make_ortho();
	CameraSample<float> s1{256.f, 256.f, 0.1f, 0.1f, 0.f};
	CameraSample<float> s2{256.f, 256.f, 0.9f, 0.9f, 0.f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	EXPECT_NEAR(r1.origin.x, r2.origin.x, 1e-5f);
	EXPECT_NEAR(r1.direction.x, r2.direction.x, 1e-5f);
}

TEST(Cameras, OrthoDOFChangesOrigin) {
	float xmin=-1.f, xmax=1.f, ymin=-1.f, ymax=1.f;
	OrthographicCamera<float> cam(xmin, xmax, ymin, ymax, 512, 512,
								   Mat4<float>{}, 0.1f, 5.f);
	CameraSample<float> s1{256.f, 256.f, 0.1f, 0.1f};
	CameraSample<float> s2{256.f, 256.f, 0.9f, 0.9f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	// Different lens samples -> different origins
	EXPECT_GT(std::abs(r1.origin.x - r2.origin.x) + std::abs(r1.origin.y - r2.origin.y), 0.001f);
}

// ---------------------------------------------------------------------------
// PerspectiveCamera
// ---------------------------------------------------------------------------
static PerspectiveCamera<float> make_persp(float fov=90.f, int rx=512, int ry=512) {
	float xmin, xmax, ymin, ymax;
	compute_screen_window<float>(rx, ry, xmin, xmax, ymin, ymax);
	return PerspectiveCamera<float>(fov, xmin, xmax, ymin, ymax, rx, ry);
}

TEST(Cameras, PerspRayIsUnitLength) {
	auto cam = make_persp();
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	EXPECT_NEAR(ray_len(cam.generate_ray(s)), 1.f, 1e-5f);
}

TEST(Cameras, PerspCenterRayIsForward) {
	// Center pixel of a square image should produce a ray along +Z (identity cam)
	auto cam = make_persp(90.f, 512, 512);
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray.direction.x, 0.f, 1e-3f);
	EXPECT_NEAR(ray.direction.y, 0.f, 1e-3f);
	EXPECT_GT(ray.direction.z, 0.99f);
}

TEST(Cameras, PerspOriginAtCameraOrigin) {
	// Pinhole origin is always (0,0,0) in camera space → world with identity
	auto cam = make_persp();
	CameraSample<float> s{100.f, 200.f, 0.5f, 0.5f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray.origin.x, 0.f, 1e-5f);
	EXPECT_NEAR(ray.origin.y, 0.f, 1e-5f);
	EXPECT_NEAR(ray.origin.z, 0.f, 1e-5f);
}

TEST(Cameras, PerspRayWeight) {
	auto cam = make_persp();
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	EXPECT_NEAR(cam.generate_ray(s).weight, 1.f, 1e-6f);
}

TEST(Cameras, PerspNarrowerFOVMoreForward) {
	// 30-degree FOV ray at edge should be closer to forward than 90-degree
	float xmin, xmax, ymin, ymax;
	compute_screen_window<float>(512, 512, xmin, xmax, ymin, ymax);
	PerspectiveCamera<float> cam90(90.f, xmin, xmax, ymin, ymax, 512, 512);
	PerspectiveCamera<float> cam30(30.f, xmin, xmax, ymin, ymax, 512, 512);
	CameraSample<float> s{512.f, 256.f, 0.5f, 0.5f}; // right edge
	auto r90 = cam90.generate_ray(s);
	auto r30 = cam30.generate_ray(s);
	// 30-degree ray has less lateral deviation → direction.z closer to 1
	EXPECT_GT(r30.direction.z, r90.direction.z);
}

TEST(Cameras, PerspDifferentPixelsDifferentDirections) {
	auto cam = make_persp();
	CameraSample<float> s1{0.f,   0.f,   0.5f, 0.5f};
	CameraSample<float> s2{511.f, 511.f, 0.5f, 0.5f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	EXPECT_GT(std::abs(r1.direction.x - r2.direction.x) +
			  std::abs(r1.direction.y - r2.direction.y), 0.01f);
}

TEST(Cameras, PerspNoDOFLensSampleDoesNotMatter) {
	auto cam = make_persp(); // no DOF
	CameraSample<float> s1{256.f, 256.f, 0.0f, 0.0f};
	CameraSample<float> s2{256.f, 256.f, 0.9f, 0.9f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	EXPECT_NEAR(r1.direction.x, r2.direction.x, 1e-5f);
	EXPECT_NEAR(r1.direction.z, r2.direction.z, 1e-5f);
}

TEST(Cameras, PerspDOFChangesOriginAndDirection) {
	float xmin, xmax, ymin, ymax;
	compute_screen_window<float>(512, 512, xmin, xmax, ymin, ymax);
	PerspectiveCamera<float> cam(90.f, xmin, xmax, ymin, ymax, 512, 512,
								  Mat4<float>{}, 0.5f, 10.f);
	CameraSample<float> s1{256.f, 256.f, 0.1f, 0.1f};
	CameraSample<float> s2{256.f, 256.f, 0.9f, 0.9f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	EXPECT_GT(std::abs(r1.origin.x - r2.origin.x) + std::abs(r1.origin.y - r2.origin.y), 0.01f);
	// Both rays still unit length
	EXPECT_NEAR(ray_len(r1), 1.f, 1e-5f);
	EXPECT_NEAR(ray_len(r2), 1.f, 1e-5f);
}

TEST(Cameras, PerspTranslatedCamera) {
	Mat4<float> cam2world = make_translate<float>(1.f, 2.f, 3.f);
	float xmin=-1.f, xmax=1.f, ymin=-1.f, ymax=1.f;
	PerspectiveCamera<float> cam(90.f, xmin, xmax, ymin, ymax, 512, 512, cam2world);
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray.origin.x, 1.f, 1e-4f);
	EXPECT_NEAR(ray.origin.y, 2.f, 1e-4f);
	EXPECT_NEAR(ray.origin.z, 3.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// SphericalCamera — EquiRectangular
// ---------------------------------------------------------------------------
TEST(Cameras, SphericalEquiRectRayIsUnitLength) {
	SphericalCamera<float> cam(512, 256, SphericalCamera<float>::EquiRectangular);
	CameraSample<float> s{100.f, 100.f, 0.5f, 0.5f};
	EXPECT_NEAR(ray_len(cam.generate_ray(s)), 1.f, 1e-5f);
}

TEST(Cameras, SphericalEquiRectCenterRay) {
	// u=0.5, v=0.5 -> theta=pi/2, phi=pi -> direction (-1,0,0) in camera
	// After swap(y,z): still (-1,0,0)
	SphericalCamera<float> cam(512, 256, SphericalCamera<float>::EquiRectangular);
	// pFilm = (256, 128) -> u=0.5, v=0.5
	CameraSample<float> s{256.f, 128.f, 0.5f, 0.5f};
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray_len(ray), 1.f, 1e-5f);
	// Direction should be a valid unit vector
	EXPECT_LT(std::abs(ray_len(ray) - 1.f), 1e-4f);
}

TEST(Cameras, SphericalEquiRectTopRay) {
	// v~=0 -> theta~=0 -> direction points toward +Z (north pole = wz=1, but swapped -> wy=1)
	SphericalCamera<float> cam(512, 256, SphericalCamera<float>::EquiRectangular);
	CameraSample<float> s{0.f, 0.001f, 0.5f, 0.5f}; // very top
	auto ray = cam.generate_ray(s);
	EXPECT_NEAR(ray_len(ray), 1.f, 1e-4f);
}

TEST(Cameras, SphericalEquiRectDifferentPixelsDifferentDirs) {
	SphericalCamera<float> cam(512, 256, SphericalCamera<float>::EquiRectangular);
	CameraSample<float> s1{0.f,   0.f,   0.5f, 0.5f};
	CameraSample<float> s2{256.f, 128.f, 0.5f, 0.5f};
	auto r1 = cam.generate_ray(s1);
	auto r2 = cam.generate_ray(s2);
	EXPECT_GT(std::abs(r1.direction.x - r2.direction.x) +
			  std::abs(r1.direction.y - r2.direction.y) +
			  std::abs(r1.direction.z - r2.direction.z), 0.01f);
}

// ---------------------------------------------------------------------------
// SphericalCamera — EqualArea
// ---------------------------------------------------------------------------
TEST(Cameras, SphericalEqualAreaRayIsUnitLength) {
	SphericalCamera<float> cam(512, 512, SphericalCamera<float>::EqualArea);
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	EXPECT_NEAR(ray_len(cam.generate_ray(s)), 1.f, 1e-5f);
}

TEST(Cameras, SphericalEqualAreaMultiplePixelsUnitLength) {
	SphericalCamera<float> cam(512, 512, SphericalCamera<float>::EqualArea);
	float test_pixels[][2] = {{0.f,0.f},{511.f,0.f},{0.f,511.f},{511.f,511.f},{256.f,256.f}};
	for (auto& p : test_pixels) {
		CameraSample<float> s{p[0], p[1], 0.5f, 0.5f};
		EXPECT_NEAR(ray_len(cam.generate_ray(s)), 1.f, 1e-5f) << "px=" << p[0] << " py=" << p[1];
	}
}

// ---------------------------------------------------------------------------
// LookAt helper
// ---------------------------------------------------------------------------
TEST(Cameras, LookAtForwardIsZ) {
	// eye at origin, looking toward +Z, up is +Y
	auto m = make_look_at<float>(0.f,0.f,0.f,  0.f,0.f,1.f,  0.f,1.f,0.f);
	// Forward direction (0,0,1) should stay (0,0,1) in world
	auto d = m.transform_vec(0.f, 0.f, 1.f);
	EXPECT_NEAR(d.x, 0.f, 1e-5f);
	EXPECT_NEAR(d.y, 0.f, 1e-5f);
	EXPECT_NEAR(d.z, 1.f, 1e-5f);
}

TEST(Cameras, LookAtOriginTranslated) {
	// Camera at (1,2,3)
	auto m = make_look_at<float>(1.f,2.f,3.f,  1.f,2.f,4.f,  0.f,1.f,0.f);
	auto p = m.transform_point(0.f, 0.f, 0.f);
	EXPECT_NEAR(p.x, 1.f, 1e-5f);
	EXPECT_NEAR(p.y, 2.f, 1e-5f);
	EXPECT_NEAR(p.z, 3.f, 1e-5f);
}

TEST(Cameras, PerspWithLookAtRayOriginAtEye) {
	auto cam2world = make_look_at<float>(0.f, 0.f, -5.f,  // eye behind scene
										  0.f, 0.f,  0.f,  // look at origin
										  0.f, 1.f,  0.f); // up
	float xmin=-1.f, xmax=1.f, ymin=-1.f, ymax=1.f;
	PerspectiveCamera<float> cam(90.f, xmin, xmax, ymin, ymax, 512, 512, cam2world);
	CameraSample<float> s{256.f, 256.f, 0.5f, 0.5f};
	auto ray = cam.generate_ray(s);
	// Origin should be at (0,0,-5) in world
	EXPECT_NEAR(ray.origin.x, 0.f, 1e-4f);
	EXPECT_NEAR(ray.origin.y, 0.f, 1e-4f);
	EXPECT_NEAR(ray.origin.z, -5.f, 1e-4f);
	// Center ray should point toward +Z (toward origin)
	EXPECT_NEAR(ray.direction.x, 0.f, 1e-3f);
	EXPECT_NEAR(ray.direction.y, 0.f, 1e-3f);
	EXPECT_GT(ray.direction.z, 0.99f);
}
