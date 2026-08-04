// cameras.h — pbrt-v4 camera models (simplified, header-only, CPU, templated on T)
// Mirrors pbrt-v4 src/pbrt/cameras.h / cameras.cpp
//
// Ported camera models:
//   1. OrthographicCamera  — parallel projection with optional thin-lens DOF
//   2. PerspectiveCamera   — perspective (FOV) with optional thin-lens DOF
//   3. SphericalCamera     — full 360° via equirectangular or equal-area mapping
//
// Design notes:
//   - Templated on scalar T (float or double).
//   - Camera-to-world transform: plain 4x4 column-major matrix (row = destination axis).
//     We provide a minimal Mat4<T> + Vec3<T> inline so the header is self-contained.
//   - CameraSample: pixel film position + lens sample (both in continuous coords).
//   - GenerateRay() returns the ray in world space, exactly as pbrt-v4.
//   - DOF thin-lens: lensRadius > 0 enables concentric-disk sampling + focus plane.
//   - No motion blur (static camera_to_world), no participating medium pointer.
//   - RealisticCamera deferred (requires lens element data).

#pragma once
#include <cmath>
#include <algorithm>
#include "sampling.h"           // SampleUniformDiskConcentric, EqualAreaSquareToSphere, WrapEqualAreaSquare

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Minimal 3-component vector used internally
// ---------------------------------------------------------------------------
template <typename T>
struct CamVec3 {
	T x, y, z;
	CamVec3() : x(T(0)), y(T(0)), z(T(0)) {}
	CamVec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
	CamVec3 operator+(const CamVec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
	CamVec3 operator-(const CamVec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
	CamVec3 operator*(T s) const { return {x*s, y*s, z*s}; }
	T dot(const CamVec3& o) const { return x*o.x + y*o.y + z*o.z; }
	T length_sq() const { return x*x + y*y + z*z; }
	T length() const { return std::sqrt(length_sq()); }
	CamVec3 normalized() const { T l = length(); return {x/l, y/l, z/l}; }
};

// ---------------------------------------------------------------------------
// Minimal Ray
// ---------------------------------------------------------------------------
template <typename T>
struct CameraRayResult {
	CamVec3<T> origin;
	CamVec3<T> direction;  // unit length
	T          time = T(0);
	T          weight = T(1); // spectral weight (always 1 for these cameras)
};

// ---------------------------------------------------------------------------
// CameraSample — mirrors pbrt-v4 CameraSample
//   pFilm: continuous pixel position (x,y) in [0,resolution)
//   pLens: 2D uniform sample in [0,1)^2 for DOF
//   time:  shutter time in [0,1)
// ---------------------------------------------------------------------------
template <typename T>
struct CameraSample {
	T pFilm_x, pFilm_y; // pixel position
	T pLens_u, pLens_v; // lens sample for DOF
	T time = T(0);
};

// ---------------------------------------------------------------------------
// Mat4x4: column-major 4x4 transform matrix
//   m[col][row]  — same convention as OpenGL / pbrt-v4 Transform
// ---------------------------------------------------------------------------
template <typename T>
struct Mat4 {
	T m[4][4]; // m[col][row]

	Mat4() {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				m[i][j] = (i == j) ? T(1) : T(0);
	}

	// Build from 16 row-major values (natural C array notation)
	static Mat4 from_rows(
		T r0c0, T r0c1, T r0c2, T r0c3,
		T r1c0, T r1c1, T r1c2, T r1c3,
		T r2c0, T r2c1, T r2c2, T r2c3,
		T r3c0, T r3c1, T r3c2, T r3c3) {
		Mat4 out;
		out.m[0][0]=r0c0; out.m[1][0]=r0c1; out.m[2][0]=r0c2; out.m[3][0]=r0c3;
		out.m[0][1]=r1c0; out.m[1][1]=r1c1; out.m[2][1]=r1c2; out.m[3][1]=r1c3;
		out.m[0][2]=r2c0; out.m[1][2]=r2c1; out.m[2][2]=r2c2; out.m[3][2]=r2c3;
		out.m[0][3]=r3c0; out.m[1][3]=r3c1; out.m[2][3]=r3c2; out.m[3][3]=r3c3;
		return out;
	}

	// Transform a point (w=1)
	CamVec3<T> transform_point(T px, T py, T pz) const {
		T w = m[0][3]*px + m[1][3]*py + m[2][3]*pz + m[3][3];
		T x = m[0][0]*px + m[1][0]*py + m[2][0]*pz + m[3][0];
		T y = m[0][1]*px + m[1][1]*py + m[2][1]*pz + m[3][1];
		T z = m[0][2]*px + m[1][2]*py + m[2][2]*pz + m[3][2];
		if (w != T(1)) { x/=w; y/=w; z/=w; }
		return {x, y, z};
	}

	// Transform a vector (w=0 — no translation)
	CamVec3<T> transform_vec(T vx, T vy, T vz) const {
		return {
			m[0][0]*vx + m[1][0]*vy + m[2][0]*vz,
			m[0][1]*vx + m[1][1]*vy + m[2][1]*vz,
			m[0][2]*vx + m[1][2]*vy + m[2][2]*vz
		};
	}

	// Matrix multiply
	Mat4 operator*(const Mat4& o) const {
		Mat4 r;
		for (int i = 0; i < 4; ++i)      // col of result
			for (int j = 0; j < 4; ++j) { // row of result
				r.m[i][j] = T(0);
				for (int k = 0; k < 4; ++k)
					r.m[i][j] += m[k][j] * o.m[i][k];
			}
		return r;
	}

	// 4x4 inverse (Gauss-Jordan)
	Mat4 inverse() const {
		T a[4][8];
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) a[i][j] = m[j][i]; // column-major -> row-major
			for (int j = 0; j < 4; ++j) a[i][j+4] = (i==j) ? T(1) : T(0);
		}
		for (int col = 0; col < 4; ++col) {
			int pivot = col;
			T best = std::abs(a[col][col]);
			for (int r = col+1; r < 4; ++r)
				if (std::abs(a[r][col]) > best) { best = std::abs(a[r][col]); pivot = r; }
			if (pivot != col) for (int j = 0; j < 8; ++j) std::swap(a[col][j], a[pivot][j]);
			T inv = T(1) / a[col][col];
			for (int j = 0; j < 8; ++j) a[col][j] *= inv;
			for (int r = 0; r < 4; ++r) {
				if (r == col) continue;
				T f = a[r][col];
				for (int j = 0; j < 8; ++j) a[r][j] -= f * a[col][j];
			}
		}
		Mat4 inv;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				inv.m[j][i] = a[i][j+4]; // back to column-major
		return inv;
	}
};

// ---------------------------------------------------------------------------
// Build standard camera matrices (mirrors pbrt-v4 util/transform.h)
// ---------------------------------------------------------------------------

// Perspective projection: maps camera frustum to [-1,1]^2 NDC
// pbrt-v4: Perspective(fov, near, far) — we only need it to build cameraFromRaster
template <typename T>
Mat4<T> make_perspective(T fov_deg, T near_clip, T far_clip) {
	T inv_tan = T(1) / std::tan(T(M_PI / 180.0) * fov_deg / T(2));
	T f = far_clip, n = near_clip;
	// pbrt-v4 Perspective matrix (z maps to [0,1] depth)
	return Mat4<T>::from_rows(
		inv_tan, T(0),    T(0),              T(0),
		T(0),    inv_tan, T(0),              T(0),
		T(0),    T(0),    f/(f-n),           -f*n/(f-n),
		T(0),    T(0),    T(1),              T(0)
	);
}

// Orthographic projection
template <typename T>
Mat4<T> make_orthographic(T near_clip, T far_clip) {
	T scale = T(1) / (far_clip - near_clip);
	return Mat4<T>::from_rows(
		T(1), T(0), T(0),  T(0),
		T(0), T(1), T(0),  T(0),
		T(0), T(0), scale, -near_clip * scale,
		T(0), T(0), T(0),  T(1)
	);
}

// Scale matrix
template <typename T>
Mat4<T> make_scale(T sx, T sy, T sz) {
	return Mat4<T>::from_rows(
		sx,   T(0), T(0), T(0),
		T(0), sy,   T(0), T(0),
		T(0), T(0), sz,   T(0),
		T(0), T(0), T(0), T(1)
	);
}

// Translation matrix
template <typename T>
Mat4<T> make_translate(T tx, T ty, T tz) {
	return Mat4<T>::from_rows(
		T(1), T(0), T(0), tx,
		T(0), T(1), T(0), ty,
		T(0), T(0), T(1), tz,
		T(0), T(0), T(0), T(1)
	);
}

// ---------------------------------------------------------------------------
// ProjectiveCameraBase — shared raster<->camera transform infrastructure
// Mirrors pbrt-v4 ProjectiveCamera.
//
// screenWindow: NDC screen extents, e.g. [-aspect,aspect] x [-1,1]
// resolution_x, resolution_y: full image resolution in pixels
// ---------------------------------------------------------------------------
template <typename T>
struct ProjectiveCameraBase {
	Mat4<T> screen_from_camera;  // projective transform (camera -> screen NDC)
	Mat4<T> camera_from_raster;  // raster pixel -> camera space point
	Mat4<T> camera_to_world;     // camera space -> world space
	T lens_radius   = T(0);
	T focal_dist    = T(1e6);
	int res_x, res_y;

	// screen_window: {xmin, xmax, ymin, ymax} in screen (NDC) space
	void init(const Mat4<T>& screen_from_cam,
			  T sw_xmin, T sw_xmax, T sw_ymin, T sw_ymax,
			  int rx, int ry,
			  const Mat4<T>& cam_to_world,
			  T lens_r, T focal_d) {
		screen_from_camera = screen_from_cam;
		camera_to_world    = cam_to_world;
		lens_radius        = lens_r;
		focal_dist         = focal_d;
		res_x = rx; res_y = ry;

		// Mirrors pbrt-v4 ProjectiveCamera constructor:
		//   NDCFromScreen = Scale(1/(sw.xmax-sw.xmin), 1/(sw.ymax-sw.ymin), 1)
		//                 * Translate(-sw.xmin, -sw.ymax, 0)
		//   rasterFromNDC = Scale(res_x, -res_y, 1)
		//   rasterFromScreen = rasterFromNDC * NDCFromScreen
		//   cameraFromRaster = Inverse(screenFromCamera) * screenFromRaster
		T sx = T(1) / (sw_xmax - sw_xmin);
		T sy = T(1) / (sw_ymax - sw_ymin);
		Mat4<T> ndc_from_screen = make_scale<T>(sx, sy, T(1)) *
								   make_translate<T>(-sw_xmin, -sw_ymax, T(0));
		Mat4<T> raster_from_ndc = make_scale<T>(T(rx), -T(ry), T(1));
		Mat4<T> raster_from_screen = raster_from_ndc * ndc_from_screen;
		Mat4<T> screen_from_raster = raster_from_screen.inverse();
		Mat4<T> camera_from_screen = screen_from_camera.inverse();
		camera_from_raster = camera_from_screen * screen_from_raster;
	}

	// Transform world-space ray direction + origin to world using camera_to_world
	CameraRayResult<T> to_world(CamVec3<T> o_cam, CamVec3<T> d_cam, T time) const {
		CamVec3<T> o_world = camera_to_world.transform_point(o_cam.x, o_cam.y, o_cam.z);
		CamVec3<T> d_world = camera_to_world.transform_vec(d_cam.x, d_cam.y, d_cam.z);
		return {o_world, d_world.normalized(), time, T(1)};
	}
};

// ---------------------------------------------------------------------------
// 1. OrthographicCamera
//    Mirrors pbrt-v4 OrthographicCamera::GenerateRay()
// ---------------------------------------------------------------------------
template <typename T>
struct OrthographicCamera : ProjectiveCameraBase<T> {
	CamVec3<T> dx_camera, dy_camera; // per-pixel offset in camera space

	// screen_window: {xmin, xmax, ymin, ymax}
	OrthographicCamera(T sw_xmin, T sw_xmax, T sw_ymin, T sw_ymax,
					   int res_x, int res_y,
					   const Mat4<T>& camera_to_world = Mat4<T>{},
					   T lens_radius = T(0),
					   T focal_dist  = T(1e6)) {
		this->init(make_orthographic<T>(T(0), T(1)),
				   sw_xmin, sw_xmax, sw_ymin, sw_ymax,
				   res_x, res_y, camera_to_world, lens_radius, focal_dist);
		// Compute per-pixel differentials in camera space
		dx_camera = this->camera_from_raster.transform_vec(T(1), T(0), T(0));
		dy_camera = this->camera_from_raster.transform_vec(T(0), T(1), T(0));
	}

	// Generate primary ray — mirrors pbrt-v4 OrthographicCamera::GenerateRay()
	CameraRayResult<T> generate_ray(const CameraSample<T>& sample) const {
		// Raster -> camera
		CamVec3<T> p_cam = this->camera_from_raster.transform_point(
			sample.pFilm_x, sample.pFilm_y, T(0));

		CamVec3<T> o = p_cam;
		CamVec3<T> d{T(0), T(0), T(1)}; // +Z forward in camera space

		// Thin-lens depth of field
		if (this->lens_radius > T(0)) {
			T dx, dy;
			SampleUniformDiskConcentric(sample.pLens_u, sample.pLens_v, dx, dy);
			T lx = this->lens_radius * dx;
			T ly = this->lens_radius * dy;

			T ft = this->focal_dist / d.z; // d.z == 1
			CamVec3<T> p_focus{o.x + ft*d.x, o.y + ft*d.y, o.z + ft*d.z};

			o = {lx, ly, T(0)};
			d = (p_focus - o).normalized();
		}

		return this->to_world(o, d, sample.time);
	}
};

// ---------------------------------------------------------------------------
// 2. PerspectiveCamera
//    Mirrors pbrt-v4 PerspectiveCamera::GenerateRay()
// ---------------------------------------------------------------------------
template <typename T>
struct PerspectiveCamera : ProjectiveCameraBase<T> {
	CamVec3<T> dx_camera, dy_camera; // per-pixel direction differentials

	// fov_deg: horizontal field of view in degrees
	// screen_window: {xmin, xmax, ymin, ymax} (computed from aspect ratio)
	PerspectiveCamera(T fov_deg,
					  T sw_xmin, T sw_xmax, T sw_ymin, T sw_ymax,
					  int res_x, int res_y,
					  const Mat4<T>& camera_to_world = Mat4<T>{},
					  T lens_radius = T(0),
					  T focal_dist  = T(1e6)) {
		this->init(make_perspective<T>(fov_deg, T(1e-2), T(1000)),
				   sw_xmin, sw_xmax, sw_ymin, sw_ymax,
				   res_x, res_y, camera_to_world, lens_radius, focal_dist);

		// dxCamera = cameraFromRaster(1,0,0) - cameraFromRaster(0,0,0)
		CamVec3<T> p00 = this->camera_from_raster.transform_point(T(0), T(0), T(0));
		CamVec3<T> p10 = this->camera_from_raster.transform_point(T(1), T(0), T(0));
		CamVec3<T> p01 = this->camera_from_raster.transform_point(T(0), T(1), T(0));
		dx_camera = p10 - p00;
		dy_camera = p01 - p00;
	}

	// Generate primary ray — mirrors pbrt-v4 PerspectiveCamera::GenerateRay()
	CameraRayResult<T> generate_ray(const CameraSample<T>& sample) const {
		// Raster -> camera point on near plane
		CamVec3<T> p_cam = this->camera_from_raster.transform_point(
			sample.pFilm_x, sample.pFilm_y, T(0));

		CamVec3<T> o{T(0), T(0), T(0)};
		// Direction: normalize the camera-space point (pinhole at origin)
		CamVec3<T> d = p_cam.normalized();

		// Thin-lens depth of field
		if (this->lens_radius > T(0)) {
			T dx, dy;
			SampleUniformDiskConcentric(sample.pLens_u, sample.pLens_v, dx, dy);
			T lx = this->lens_radius * dx;
			T ly = this->lens_radius * dy;

			T ft = this->focal_dist / d.z;
			CamVec3<T> p_focus{o.x + ft*d.x, o.y + ft*d.y, o.z + ft*d.z};

			o = {lx, ly, T(0)};
			d = (p_focus - o).normalized();
		}

		return this->to_world(o, d, sample.time);
	}
};

// ---------------------------------------------------------------------------
// 3. SphericalCamera
//    Mirrors pbrt-v4 SphericalCamera::GenerateRay()
//    Supports EquiRectangular and EqualArea mappings.
// ---------------------------------------------------------------------------
template <typename T>
struct SphericalCamera {
	enum Mapping { EquiRectangular, EqualArea };

	Mat4<T> camera_to_world;
	int res_x, res_y;
	Mapping mapping;

	SphericalCamera(int rx, int ry,
					Mapping m = EquiRectangular,
					const Mat4<T>& cam_to_world = Mat4<T>{})
		: camera_to_world(cam_to_world), res_x(rx), res_y(ry), mapping(m) {}

	// Generate ray — mirrors pbrt-v4 SphericalCamera::GenerateRay()
	CameraRayResult<T> generate_ray(const CameraSample<T>& sample) const {
		T u = sample.pFilm_x / T(res_x);
		T v = sample.pFilm_y / T(res_y);

		T wx, wy, wz;
		if (mapping == EquiRectangular) {
			// theta in [0, pi], phi in [0, 2pi]
			double theta = M_PI * (double)v;
			double phi   = 2.0 * M_PI * (double)u;
			double sin_t = std::sin(theta);
			double cos_t = std::cos(theta);
			wx = T(sin_t * std::cos(phi));
			wy = T(sin_t * std::sin(phi));
			wz = T(cos_t);
		} else {
			// Equal-area square-to-sphere
			double ud = (double)u, vd = (double)v;
			WrapEqualAreaSquare(ud, vd);
			double ewx, ewy, ewz;
			EqualAreaSquareToSphere(ud, vd, ewx, ewy, ewz);
			wx = T(ewx); wy = T(ewy); wz = T(ewz);
		}

		// pbrt-v4: swap(dir.y, dir.z) after computing spherical direction
		std::swap(wy, wz);

		CamVec3<T> o{T(0), T(0), T(0)};
		CamVec3<T> d{wx, wy, wz}; // already unit length from mapping

		// Transform to world
		CamVec3<T> o_w = camera_to_world.transform_point(o.x, o.y, o.z);
		CamVec3<T> d_w = camera_to_world.transform_vec(d.x, d.y, d.z);
		return {o_w, d_w.normalized(), sample.time, T(1)};
	}
};

// ---------------------------------------------------------------------------
// Helper: build a standard screen window from image resolution + aspect
// Mirrors pbrt-v4 PerspectiveCamera::Create / OrthographicCamera::Create
// Returns {xmin, xmax, ymin, ymax}
// ---------------------------------------------------------------------------
template <typename T>
void compute_screen_window(int res_x, int res_y,
						   T& xmin, T& xmax, T& ymin, T& ymax) {
	T frame = T(res_x) / T(res_y);
	if (frame > T(1)) {
		xmin = -frame; xmax = frame;
		ymin = T(-1);  ymax = T(1);
	} else {
		xmin = T(-1);  xmax = T(1);
		ymin = -T(1)/frame; ymax = T(1)/frame;
	}
}

// ===========================================================================
// RealisticCamera<T>
//
// Port of pbrt-v4 RealisticCamera (cameras.h / cameras.cpp).
//
// Models a real camera lens system as a sequence of spherical refractive
// elements and aperture stops.  Rays are traced from the film plane through
// all elements (TraceLensesFromFilm) and, if they exit the front element,
// are transformed to world space and weighted by the vignetting factor
//   w = cos^4(theta) / (pdf * LensRearZ^2)
// (pbrt-v4 GenerateRay lines 944-950).
//
// Lens data format (same as pbrt-v4 lensParameters):
//   groups of 4 floats per element: curvatureRadius_mm, thickness_mm, eta,
//   apertureDiameter_mm.  curvatureRadius == 0 marks the aperture stop.
//
// Usage:
//   std::vector<T> lens = {/* 4-tuples */};
//   RealisticCamera<T> cam(camera_to_world, film_x_mm, film_y_mm,
//                          focus_distance_mm, aperture_diameter_mm, lens);
//   CameraRayResult<T> r = cam.generate_ray(sample);
//   if (r.weight > 0) { /* use r.origin, r.direction */ }
//
// pbrt-v4 reference: cameras.h lines 465-625, cameras.cpp lines 694-952.
// ===========================================================================

namespace realistic_detail {

// Solve a*t^2 + b*t + c = 0; returns false if no real roots.
template<typename T>
inline bool Quadratic(T a, T b, T c, T* t0, T* t1) {
    double da=(double)a, db=(double)b, dc=(double)c;
    double disc = db*db - 4.0*da*dc;
    if (disc < 0) return false;
    double sq = std::sqrt(disc);
    double q = (db < 0) ? -0.5*(db - sq) : -0.5*(db + sq);
    *t0 = (T)(q / da);
    *t1 = (T)(dc / q);
    if (*t0 > *t1) std::swap(*t0, *t1);
    return true;
}

// Snell's law refraction. n must face the incoming ray (dot(d,n)<0).
// eta = eta_i / eta_t.  Returns false on total internal reflection.
template<typename T>
inline bool Refract(T dx, T dy, T dz,
                    T nx, T ny, T nz,
                    T eta, T& ox, T& oy, T& oz) {
    T cosI = -(dx*nx + dy*ny + dz*nz);
    T sin2T = eta*eta * std::max(T(0), T(1) - cosI*cosI);
    if (sin2T >= T(1)) return false;
    T cosT = std::sqrt(T(1) - sin2T);
    ox = eta*dx + (eta*cosI - cosT)*nx;
    oy = eta*dy + (eta*cosI - cosT)*ny;
    oz = eta*dz + (eta*cosI - cosT)*nz;
    return true;
}

// Van der Corput (base-2) radical inverse.
inline double RI2(uint64_t i) {
    i = (i << 32)|(i >> 32);
    i = ((i&0x0000FFFF0000FFFFull)<<16)|((i&0xFFFF0000FFFF0000ull)>>16);
    i = ((i&0x00FF00FF00FF00FFull)<< 8)|((i&0xFF00FF00FF00FF00ull)>> 8);
    i = ((i&0x0F0F0F0F0F0F0F0Full)<< 4)|((i&0xF0F0F0F0F0F0F0F0ull)>> 4);
    i = ((i&0x3333333333333333ull)<< 2)|((i&0xCCCCCCCCCCCCCCCCull)>> 2);
    i = ((i&0x5555555555555555ull)<< 1)|((i&0xAAAAAAAAAAAAAAAAull)>> 1);
    double inv = 1.0 / (double)(1ull<<32);
    return (double)(i >> 32) * inv + (double)(i & 0xFFFFFFFFull) * (inv * inv);
}
// Base-3 radical inverse.
inline double RI3(uint64_t n) {
    double r=0, f=1.0/3.0;
    while (n>0){r+=(n%3)*f; n/=3; f/=3.0;}
    return r;
}
} // namespace realistic_detail

// ---------------------------------------------------------------------------
// RealisticCamera<T>
// ---------------------------------------------------------------------------
template<typename T>
struct RealisticCamera {
    struct LensElement {
        T curvatureRadius; // metres
        T thickness;       // metres
        T eta;             // IOR on image side; 0 = aperture stop
        T apertureRadius;  // metres
    };
    struct Bounds2 {
        T xMin=T(0), xMax=T(0), yMin=T(0), yMax=T(0);
        bool degenerate=true;
        T area() const { return (xMax-xMin)*(yMax-yMin); }
    };

    // nSamples_pupil: samples per exit-pupil annulus slab (pbrt-v4 uses 1M; 1024 is fine for testing)
    RealisticCamera(Mat4<T> camera_to_world,
                    T film_x_mm, T film_y_mm,
                    T focus_distance,
                    T aperture_diameter_mm,
                    const std::vector<T>& lens_params,
                    int nSamples_pupil = 1024)
        : camera_to_world_(camera_to_world)
    {
        film_half_x_ = film_x_mm / T(1000);
        film_half_y_ = film_y_mm / T(1000);

        for (size_t i=0; i+3<lens_params.size(); i+=4) {
            T cr  = lens_params[i]   / T(1000);
            T th  = lens_params[i+1] / T(1000);
            T eta = lens_params[i+2];
            T apd = lens_params[i+3] / T(1000);
            if (cr == T(0)) {
                T req = aperture_diameter_mm / T(1000);
                if (req < apd) apd = req;
            }
            elements_.push_back({cr, th, eta, apd/T(2)});
        }

        // Adjust rear element thickness to focus at the desired distance.
        T orig_rear = elements_.back().thickness;
        elements_.back().thickness = focus_thick_lens(focus_distance, orig_rear);

        // Precompute exit pupil bounds.
        int nSlabs = 64;
        exit_pupil_bounds_.resize(nSlabs);
        for (int i=0; i<nSlabs; ++i) {
            T r0 = T(i)     / T(nSlabs) * film_diagonal() / T(2);
            T r1 = T(i+1)   / T(nSlabs) * film_diagonal() / T(2);
            exit_pupil_bounds_[i] = bound_exit_pupil(r0, r1, nSamples_pupil);
        }
    }

    // generate_ray: produce a world-space ray. Returns weight=0 if vignetted.
    CameraRayResult<T> generate_ray(const CameraSample<T>& sample) const {
        CameraRayResult<T> result; result.weight = T(0);

        // pbrt-v4 negates x to match film orientation
        T pfx = -sample.pFilm_x;
        T pfy =  sample.pFilm_y;

        T ppx, ppy, ppz, ppdf;
        if (!sample_exit_pupil(pfx, pfy, sample.pLens_u, sample.pLens_v,
                               ppx, ppy, ppz, ppdf))
            return result;

        // rFilm direction in camera space (from film to exit pupil)
        T rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz - T(0);
        T rLen = std::sqrt(rdx*rdx + rdy*rdy + rdz*rdz);

        T out_ox, out_oy, out_oz, out_dx, out_dy, out_dz;
        T w = trace_lenses_from_film(pfx, pfy, T(0),
                                      rdx, rdy, rdz,
                                      out_ox, out_oy, out_oz,
                                      out_dx, out_dy, out_dz);
        if (w == T(0)) return result;

        // cos^4(theta) weighting where theta is angle to optical axis in lens space
        // cosTheta = |rdz| / |rFilm.d|  (pbrt-v4: Normalize(rFilm.d).z in lens space)
        T cosTheta = (rLen > T(0)) ? std::abs(rdz / rLen) : T(0);
        T lrz = lens_rear_z();
        if (lrz <= T(0)) return result;
        w *= (cosTheta*cosTheta*cosTheta*cosTheta) / (ppdf * lrz * lrz);

        // Transform out ray (already in camera space with z=+forward) to world space.
        CamVec3<T> wo = camera_to_world_.transform_point(out_ox, out_oy, out_oz);
        CamVec3<T> wd = camera_to_world_.transform_vec(out_dx, out_dy, out_dz);
        wd = wd.normalized();

        result.origin    = wo;
        result.direction = wd;
        result.time      = sample.time;
        result.weight    = w;
        return result;
    }

    T lens_rear_z()  const { return elements_.back().thickness; }
    T lens_front_z() const { T s=T(0); for (const auto& e:elements_) s+=e.thickness; return s; }
    T rear_element_radius() const { return elements_.back().apertureRadius; }
    int num_elements() const { return (int)elements_.size(); }

private:
    // TraceLensesFromFilm: mirrors pbrt-v4.
    // Camera space: film at z=0, optical axis +z toward scene.
    // Lens space (internal): same x,y; z is flipped (loz = -cam_z).
    // Returns 0 if vignetted, 1 if passed.
    // out_* are in camera space on exit.
    T trace_lenses_from_film(T ox, T oy, T oz,
                              T dx, T dy, T dz,
                              T& out_ox, T& out_oy, T& out_oz,
                              T& out_dx, T& out_dy, T& out_dz) const {
        // pbrt-v4: rLens.o = (rCamera.o.x, rCamera.o.y, -rCamera.o.z)
        //          rLens.d = (rCamera.d.x, rCamera.d.y, -rCamera.d.z)
        T lox = ox, loy = oy, loz = -oz;
        T ldx = dx, ldy = dy, ldz = -dz;
        T elementZ = T(0);

        for (int i=(int)elements_.size()-1; i>=0; --i) {
            const LensElement& el = elements_[i];
            elementZ -= el.thickness;
            bool isStop = (el.curvatureRadius == T(0));
            T t, nx=T(0), ny=T(0), nz=T(0);

            if (isStop) {
                if (ldz == T(0)) return T(0);
                t = (elementZ - loz) / ldz;
                if (t < T(0)) return T(0);
            } else {
                if (!intersect_spherical(el.curvatureRadius,
                                          elementZ + el.curvatureRadius,
                                          lox, loy, loz, ldx, ldy, ldz,
                                          t, nx, ny, nz))
                    return T(0);
            }
            T hx=lox+t*ldx, hy=loy+t*ldy, hz=loz+t*ldz;
            if (hx*hx+hy*hy > el.apertureRadius*el.apertureRadius) return T(0);
            lox=hx; loy=hy; loz=hz;

            if (!isStop) {
                T eta_i = (el.eta == T(0)) ? T(1) : el.eta;
                T eta_t = (i>0 && elements_[i-1].eta != T(0)) ? elements_[i-1].eta : T(1);
                T len = std::sqrt(ldx*ldx+ldy*ldy+ldz*ldz);
                if (!realistic_detail::Refract(ldx/len, ldy/len, ldz/len,
                                               nx, ny, nz,
                                               eta_t/eta_i,
                                               ldx, ldy, ldz))
                    return T(0);
            }
        }
        // pbrt-v4: rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
        //                     Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), ...)
        out_ox=lox; out_oy=loy; out_oz=-loz;
        out_dx=ldx; out_dy=ldy; out_dz=-ldz;
        return T(1);
    }

    // TraceLensesFromScene: scene->film direction. Used for cardinal points.
    bool trace_lenses_from_scene(T ox, T oy, T oz,
                                  T dx, T dy, T dz,
                                  T& out_ox, T& out_oy, T& out_oz,
                                  T& out_dx, T& out_dy, T& out_dz) const {
        // pbrt-v4 uses Scale(1,1,-1): loz=-oz, ldz=-dz
        T lox=ox, loy=oy, loz=-oz;
        T ldx=dx, ldy=dy, ldz=-dz;
        T elementZ = -lens_front_z();

        for (int i=0; i<(int)elements_.size(); ++i) {
            const LensElement& el = elements_[i];
            bool isStop = (el.curvatureRadius == T(0));
            T t, nx=T(0), ny=T(0), nz=T(0);

            if (isStop) {
                if (ldz == T(0)) return false;
                t = (elementZ - loz) / ldz;
                if (t < T(0)) return false;
            } else {
                if (!intersect_spherical(el.curvatureRadius,
                                          elementZ + el.curvatureRadius,
                                          lox, loy, loz, ldx, ldy, ldz,
                                          t, nx, ny, nz))
                    return false;
            }
            T hx=lox+t*ldx, hy=loy+t*ldy, hz=loz+t*ldz;
            if (hx*hx+hy*hy > el.apertureRadius*el.apertureRadius) return false;
            lox=hx; loy=hy; loz=hz;

            if (!isStop) {
                T eta_i = (i==0 || elements_[i-1].eta==T(0)) ? T(1) : elements_[i-1].eta;
                T eta_t = (el.eta==T(0)) ? T(1) : el.eta;
                T len=std::sqrt(ldx*ldx+ldy*ldy+ldz*ldz);
                T rx, ry, rz;
                if (!realistic_detail::Refract(ldx/len, ldy/len, ldz/len,
                                               nx, ny, nz,
                                               eta_i/eta_t, rx, ry, rz))
                    return false;
                ldx=rx; ldy=ry; ldz=rz;
            }
            elementZ += el.thickness;
        }
        out_ox=lox; out_oy=loy; out_oz=-loz;
        out_dx=ldx; out_dy=ldy; out_dz=-ldz;
        return true;
    }

    // Mirrors pbrt-v4 IntersectSphericalElement.
    static bool intersect_spherical(T radius, T zCenter,
                                     T ox, T oy, T oz,
                                     T dx, T dy, T dz,
                                     T& t, T& nx, T& ny, T& nz) {
        T cox=ox, coy=oy, coz=oz-zCenter;
        T A=dx*dx+dy*dy+dz*dz;
        T B=T(2)*(dx*cox+dy*coy+dz*coz);
        T C=cox*cox+coy*coy+coz*coz-radius*radius;
        T t0, t1;
        if (!realistic_detail::Quadratic(A,B,C,&t0,&t1)) return false;
        bool useCloserT = (dz>T(0)) ^ (radius<T(0));
        t = useCloserT ? std::min(t0,t1) : std::max(t0,t1);
        if (t < T(0)) return false;
        T hx=ox+t*dx, hy=oy+t*dy, hz=oz+t*dz;
        nx=hx; ny=hy; nz=hz-zCenter;
        T nlen=std::sqrt(nx*nx+ny*ny+nz*nz);
        if (nlen == T(0)) return false;
        nx/=nlen; ny/=nlen; nz/=nlen;
        // FaceForward: normal must oppose incident direction
        if (dx*nx+dy*ny+dz*nz > T(0)){nx=-nx; ny=-ny; nz=-nz;}
        return true;
    }

    // Mirrors pbrt-v4 ComputeCardinalPoints.
    // rIn/rOut are paraxial rays traced through the lens; both in camera space.
    // Camera space: +z toward scene, film at z=0.
    static void compute_cardinal_points(
        T rin_ox, T /*rin_oy*/, T /*rin_oz*/,
        T /*rin_dx*/, T /*rin_dy*/, T /*rin_dz*/,
        T rout_ox, T /*rout_oy*/, T rout_oz,
        T rout_dx, T /*rout_dy*/, T rout_dz,
        T& pz, T& fz) {
        // pbrt-v4: tf = -rOut.o.x / rOut.d.x; *fz = -rOut(tf).z
        //          tp = (rIn.o.x - rOut.o.x) / rOut.d.x; *pz = -rOut(tp).z
        if (std::abs(rout_dx) < T(1e-12)) { pz=fz=T(0); return; }
        T tf = -rout_ox / rout_dx;
        fz = -(rout_oz + tf * rout_dz);
        T tp = (rin_ox - rout_ox) / rout_dx;
        pz = -(rout_oz + tp * rout_dz);
    }

    // Mirrors pbrt-v4 FocusThickLens.  Returns adjusted rear element thickness.
    // orig_rear: the original rear element thickness before any focus shift.
    T focus_thick_lens(T focus_distance, T orig_rear) {
        T x = T(0.001) * film_diagonal();

        // --- Scene-side cardinal points ---
        // pbrt-v4: rScene = Ray( (x,0, LensFrontZ+1), (0,0,-1) )
        // In our camera space (z=+toward scene): origin z = lensFrontZ+1, dir z = -1.
        T front_z = T(0); for (const auto& e:elements_) front_z += e.thickness;
        T pz0=T(0), fz0=T(0);
        {
            T ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz;
            bool ok = trace_lenses_from_scene(
                x, T(0), front_z + T(0.001),
                T(0), T(0), T(-1),
                ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz);
            if (ok)
                compute_cardinal_points(
                    x, T(0), front_z+T(0.001), T(0), T(0), T(-1),
                    ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz,
                    pz0, fz0);
        }

        // --- Film-side cardinal points ---
        // pbrt-v4: rFilm = Ray( (x, 0, LensRearZ-1), (0,0,1) )
        // LensRearZ in pbrt-v4 is the last element's thickness (>0), a positive z.
        // We use orig_rear as that distance.
        T pz1=T(0), fz1=T(0);
        {
            T rs_ox, rs_oy, rs_oz, rs_dx, rs_dy, rs_dz;
            T w = trace_lenses_from_film(
                x, T(0), orig_rear - T(0.001),
                T(0), T(0), T(1),
                rs_ox, rs_oy, rs_oz, rs_dx, rs_dy, rs_dz);
            if (w != T(0))
                compute_cardinal_points(
                    x, T(0), orig_rear-T(0.001), T(0), T(0), T(1),
                    rs_ox, rs_oy, rs_oz, rs_dx, rs_dy, rs_dz,
                    pz1, fz1);
        }

        // pbrt-v4 FocusThickLens formula
        T f = fz0 - pz0;
        if (std::abs(f) < T(1e-9)) return orig_rear;
        T z = -focus_distance;
        T c = (pz1-z-pz0) * (pz1-z-T(4)*f-pz0);
        if (c <= T(0)) return orig_rear;
        T delta = (pz1-z+pz0 - std::sqrt(c)) / T(2);
        T result = orig_rear + delta;
        return (result > T(1e-6)) ? result : orig_rear;
    }

    // Mirrors pbrt-v4 BoundExitPupil.
    Bounds2 bound_exit_pupil(T r0, T r1, int nSamples) const {
        Bounds2 pupil;
        T rearRadius = rear_element_radius();
        T projMin = -T(1.5)*rearRadius, projMax = T(1.5)*rearRadius;

        for (int i=0; i<nSamples; ++i) {
            T filmX = r0 + (r1-r0) * T((i+0.5)/nSamples);
            T u0 = T(realistic_detail::RI2((uint64_t)i));
            T u1 = T(realistic_detail::RI3((uint64_t)i));
            T prx = projMin + u0*(projMax-projMin);
            T pry = projMin + u1*(projMax-projMin);

            bool inside = !pupil.degenerate &&
                          prx>=pupil.xMin && prx<=pupil.xMax &&
                          pry>=pupil.yMin && pry<=pupil.yMax;
            if (!inside) {
                T rdx=prx-filmX, rdy=pry, rdz=lens_rear_z();
                T ox2, oy2, oz2, dx2, dy2, dz2;
                if (trace_lenses_from_film(filmX, T(0), T(0),
                                            rdx, rdy, rdz,
                                            ox2, oy2, oz2, dx2, dy2, dz2) != T(0)) {
                    if (pupil.degenerate) {
                        pupil.xMin=pupil.xMax=prx;
                        pupil.yMin=pupil.yMax=pry;
                        pupil.degenerate=false;
                    } else {
                        pupil.xMin=std::min(pupil.xMin,prx);
                        pupil.xMax=std::max(pupil.xMax,prx);
                        pupil.yMin=std::min(pupil.yMin,pry);
                        pupil.yMax=std::max(pupil.yMax,pry);
                    }
                }
            }
        }
        if (!pupil.degenerate) {
            // pbrt-v4: Expand(bounds, 2*Length(projRearBounds.Diagonal())/sqrt(nSamples))
            // Diagonal of the projection square = sqrt(2)*(projMax-projMin).
            T expand = T(2)*std::sqrt(T(2))*(projMax-projMin)/std::sqrt(T(nSamples));
            pupil.xMin-=expand; pupil.xMax+=expand;
            pupil.yMin-=expand; pupil.yMax+=expand;
        }
        return pupil;
    }

    // Mirrors pbrt-v4 SampleExitPupil.
    bool sample_exit_pupil(T pfx, T pfy, T u0, T u1,
                            T& ppx, T& ppy, T& ppz, T& pdf) const {
        T rFilm = std::sqrt(pfx*pfx + pfy*pfy);
        int sz = (int)exit_pupil_bounds_.size();
        int rIndex = (int)(rFilm / (film_diagonal()/T(2)) * sz);
        if (rIndex >= sz) rIndex = sz-1;
        const Bounds2& b = exit_pupil_bounds_[rIndex];
        if (b.degenerate) return false;

        T lx = b.xMin + u0*(b.xMax-b.xMin);
        T ly = b.yMin + u1*(b.yMax-b.yMin);
        T a  = b.area();
        if (a <= T(0)) return false;
        pdf = T(1) / a;

        T sinTheta = (rFilm > T(0)) ? pfy/rFilm : T(0);
        T cosTheta = (rFilm > T(0)) ? pfx/rFilm : T(1);
        ppx = cosTheta*lx - sinTheta*ly;
        ppy = sinTheta*lx + cosTheta*ly;
        ppz = lens_rear_z();
        return true;
    }

    T film_diagonal() const {
        return T(2)*std::sqrt(film_half_x_*film_half_x_+film_half_y_*film_half_y_);
    }

    Mat4<T>            camera_to_world_;
    T                  film_half_x_, film_half_y_;
    std::vector<LensElement> elements_;
    std::vector<Bounds2>     exit_pupil_bounds_;
};

// ---------------------------------------------------------------------------
// Helper: build a look-at camera_to_world matrix
// eye, at, up are world-space 3-vectors
// ---------------------------------------------------------------------------
template <typename T>
Mat4<T> make_look_at(T ex, T ey, T ez,
					  T ax, T ay, T az,
					  T ux, T uy, T uz) {
	CamVec3<T> forward{ax-ex, ay-ey, az-ez};
	forward = forward.normalized();
	CamVec3<T> up{ux, uy, uz};
	// right = up x forward  (mirrors pbrt-v4 LookAt: Cross(Normalize(up), dir))
	CamVec3<T> right{
		up.y*forward.z - up.z*forward.y,
		up.z*forward.x - up.x*forward.z,
		up.x*forward.y - up.y*forward.x
	};
	right = right.normalized();
	// recompute up = forward x right  (mirrors pbrt-v4: Cross(dir, right))
	CamVec3<T> newup{
		forward.y*right.z - forward.z*right.y,
		forward.z*right.x - forward.x*right.z,
		forward.x*right.y - forward.y*right.x
	};
	// Camera space: +X = right, +Y = up, +Z = forward
	// Column-major: columns are the axes of camera space in world space + translation
	return Mat4<T>::from_rows(
		right.x,   right.y,   right.z,   ex,
		newup.x,   newup.y,   newup.z,   ey,
		forward.x, forward.y, forward.z, ez,
		T(0),      T(0),      T(0),      T(1)
	);
}
