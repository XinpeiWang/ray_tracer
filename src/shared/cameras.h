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
//
// RealisticCamera (a full physical lens simulator, ~500 lines) lives in
// realistic_camera.h, #included at this file's own end so existing
// "#include cameras.h" callers keep getting it unchanged.

#pragma once
#include <cmath>
#include <algorithm>
#include "sampling.h"           // SampleUniformDiskConcentric, EqualAreaSquareToSphere, WrapEqualAreaSquare

// ---------------------------------------------------------------------------
// Film "cropwindow"/"pixelbounds" (pbrt-v4) - NDC-fraction bounds resolved
// to concrete PIXEL bounds [x0,x1) x [y0,y1) at the render's actual
// resolution. Shared home for logic both CPU and GPU need: this header is
// already included by both src/TheRestOfYourLife/camera.h and
// gpu/optix/scene_builder.cpp for their own camera-model needs.
//
// gpu/optix/scene_builder.cpp calls this directly. src/TheRestOfYourLife/
// camera.h's own initialize() still performs the identical arithmetic
// inline (split across its own crop_x0/x1/y0/y1 member fields and a
// separate "-1 = unset" sentinel convention tied to its class's lazy-
// initialization design) rather than calling this - left as its own,
// pre-existing, already-tested implementation rather than risked for a
// pure duplication cleanup; if the two ever need to be unified, this is the
// natural target to migrate camera.h's own copy onto.
//
// Returns wasDegenerate=true when the resolved rectangle collapsed to
// empty (x1<=x0 or y1<=y0) - a valid NDC-fraction rectangle from
// pbrt_flatten.h can still round to an empty pixel range at a small enough
// actual render resolution - and DOES fall back to the full frame in that
// case, but leaves it to the caller to decide whether/how to warn about it
// (CPU and GPU want different message text/prefixes).
// ---------------------------------------------------------------------------
struct CropPixelBounds { int x0, x1, y0, y1; bool wasDegenerate; };

inline CropPixelBounds resolve_crop_pixel_bounds(double fracX0, double fracX1,
                                                  double fracY0, double fracY1,
                                                  int width, int height) {
	int x0 = static_cast<int>(std::lround(fracX0 * width));
	int x1 = static_cast<int>(std::lround(fracX1 * width));
	int y0 = static_cast<int>(std::lround(fracY0 * height));
	int y1 = static_cast<int>(std::lround(fracY1 * height));
	x0 = std::clamp(x0, 0, width);
	x1 = std::clamp(x1, 0, width);
	y0 = std::clamp(y0, 0, height);
	y1 = std::clamp(y1, 0, height);
	const bool degenerate = (x1 <= x0 || y1 <= y0);
	if (degenerate) { x0 = 0; x1 = width; y0 = 0; y1 = height; }
	return CropPixelBounds{x0, x1, y0, y1, degenerate};
}

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

// The RealisticCamera<T> lens simulator lives in realistic_camera.h,
// #included here so every existing "#include cameras.h" caller keeps
// getting it unchanged -- see that file's own header comment.
#include "realistic_camera.h"
