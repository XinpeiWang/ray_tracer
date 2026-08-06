#pragma once
// ---------------------------------------------------------------------------
// interval_vec.h -- Interval-backed 3D vector and point types
//
// Mirrors pbrt-v4 Vector3fi / Point3fi (util/vecmath.h) and
// OffsetRayOrigin / SpawnRay / SpawnRayTo (ray.h), adapted for
// double-precision CPU/GPU use.
//
// Components:
//   Vector3fi   -- Vector3<Interval>: carries floating-point error bounds
//   Point3fi    -- Point3<Interval>:  carries floating-point error bounds
//   OffsetRayOrigin -- robustly offset a hit point along the surface normal
//   SpawnRay        -- create an offset ray from a Point3fi hit
//   SpawnRayTo      -- shadow-ray helpers
//
// Design rules:
//   - Uses Interval from interval.h (double precision, outward rounding)
//   - Uses NextFloatUp/Down from float_bits.h for final nudge
//   - CPU-only; no CUDA specifics
//   - Header-only, no .cpp needed
//
// References:
//   pbrt-v4 src/pbrt/util/vecmath.h  (Apache-2.0)
//   pbrt-v4 src/pbrt/ray.h           (Apache-2.0)
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "interval.h"
#include "float_bits.h"   // NextFloatUp, NextFloatDown
#include <cmath>
#include <algorithm>

// ===========================================================================
// Vector3fi -- 3D vector with per-component interval error bounds
//
// Mirrors pbrt-v4 Vector3fi (util/vecmath.h).
// Stores an interval [mid-err, mid+err] per component.
// ===========================================================================

struct Vector3fi {
	Interval x, y, z;

	// Default: zero vector with no error
	CPU_GPU Vector3fi() : x(0.0), y(0.0), z(0.0) {}

	// Exact scalar values (zero error)
	CPU_GPU Vector3fi(double vx, double vy, double vz)
		: x(vx), y(vy), z(vz) {}

	// From intervals directly
	CPU_GPU Vector3fi(Interval ix, Interval iy, Interval iz)
		: x(ix), y(iy), z(iz) {}

	// From exact float vector (zero error)
	template<typename T>
	CPU_GPU explicit Vector3fi(T vx, T vy, T vz)
		: x(double(vx)), y(double(vy)), z(double(vz)) {}

	// From value + per-component error vector (pbrt-v4: FromValueAndError)
	CPU_GPU Vector3fi(double vx, double vy, double vz,
					  double ex, double ey, double ez)
		: x(Interval::FromValueAndError(vx, ex)),
		  y(Interval::FromValueAndError(vy, ey)),
		  z(Interval::FromValueAndError(vz, ez)) {}

	// Midpoint as plain double triple
	CPU_GPU double mx() const { return x.Midpoint(); }
	CPU_GPU double my() const { return y.Midpoint(); }
	CPU_GPU double mz() const { return z.Midpoint(); }

	// Per-component half-width = error estimate (mirrors pbrt-v4 .Error())
	CPU_GPU double ex() const { return x.Width() * 0.5; }
	CPU_GPU double ey() const { return y.Width() * 0.5; }
	CPU_GPU double ez() const { return z.Width() * 0.5; }

	// True when all intervals are point-exact
	CPU_GPU bool IsExact() const {
		return x.Width() == 0.0 && y.Width() == 0.0 && z.Width() == 0.0;
	}

	// Arithmetic (interval × interval)
	CPU_GPU Vector3fi operator+(const Vector3fi& v) const {
		return {x + v.x, y + v.y, z + v.z};
	}
	CPU_GPU Vector3fi operator-(const Vector3fi& v) const {
		return {x - v.x, y - v.y, z - v.z};
	}
	CPU_GPU Vector3fi operator*(double s) const {
		return {x * Interval(s), y * Interval(s), z * Interval(s)};
	}
	CPU_GPU Vector3fi operator-() const { return {-x, -y, -z}; }
};

CPU_GPU Vector3fi operator*(double s, const Vector3fi& v) { return v * s; }

// ===========================================================================
// Point3fi -- 3D point with per-component interval error bounds
//
// Mirrors pbrt-v4 Point3fi (util/vecmath.h).
// ===========================================================================

struct Point3fi {
	Interval x, y, z;

	// Default: origin with no error
	CPU_GPU Point3fi() : x(0.0), y(0.0), z(0.0) {}

	// Exact scalar values (zero error)
	CPU_GPU Point3fi(double px, double py, double pz)
		: x(px), y(py), z(pz) {}

	// From intervals directly
	CPU_GPU Point3fi(Interval ix, Interval iy, Interval iz)
		: x(ix), y(iy), z(iz) {}

	// From exact point + per-component absolute error (pbrt-v4 convention)
	CPU_GPU Point3fi(double px, double py, double pz,
					 double ex, double ey, double ez)
		: x(Interval::FromValueAndError(px, ex)),
		  y(Interval::FromValueAndError(py, ey)),
		  z(Interval::FromValueAndError(pz, ez)) {}

	// Midpoint as plain double triple
	CPU_GPU double mx() const { return x.Midpoint(); }
	CPU_GPU double my() const { return y.Midpoint(); }
	CPU_GPU double mz() const { return z.Midpoint(); }

	// Per-component half-width = error estimate (mirrors pbrt-v4 .Error())
	CPU_GPU double ex() const { return x.Width() * 0.5; }
	CPU_GPU double ey() const { return y.Width() * 0.5; }
	CPU_GPU double ez() const { return z.Width() * 0.5; }

	// True when all intervals are point-exact
	CPU_GPU bool IsExact() const {
		return x.Width() == 0.0 && y.Width() == 0.0 && z.Width() == 0.0;
	}

	// Point + vector
	CPU_GPU Point3fi operator+(const Vector3fi& v) const {
		return {x + v.x, y + v.y, z + v.z};
	}
	CPU_GPU Point3fi& operator+=(const Vector3fi& v) {
		x += v.x; y += v.y; z += v.z; return *this;
	}
	CPU_GPU Point3fi operator-(const Vector3fi& v) const {
		return {x - v.x, y - v.y, z - v.z};
	}
	CPU_GPU Point3fi& operator-=(const Vector3fi& v) {
		x -= v.x; y -= v.y; z -= v.z; return *this;
	}
	// Point - Point -> Vector3fi
	CPU_GPU Vector3fi operator-(const Point3fi& p) const {
		return {x - p.x, y - p.y, z - p.z};
	}
	CPU_GPU Point3fi operator-() const { return {-x, -y, -z}; }
};

// ===========================================================================
// OffsetRayOrigin
//
// Mirrors pbrt-v4 OffsetRayOrigin(Point3fi pi, Normal3f n, Vector3f w) in
// ray.h.  Given a surface hit point with error bounds and an outward normal,
// computes a new origin that is safely on the correct side of the surface.
//
// Algorithm (pbrt-v4):
//   d      = Dot(Abs(n), error)           -- worst-case offset distance
//   offset = d * n                        -- push along normal
//   if Dot(w, n) < 0: offset = -offset   -- flip to correct side
//   po     = p_mid + offset
//   round each coordinate away from surface (NextFloatUp/Down)
// ===========================================================================

// ===========================================================================
// Internal helper in a named namespace to avoid ADL conflicts with
// shapes_detail::dot3 / normalize3 when both headers are included together.
namespace interval_vec_detail {
CPU_GPU inline double dot3(double ax, double ay, double az,
							double bx, double by, double bz) {
	return ax*bx + ay*by + az*bz;
}
} // namespace interval_vec_detail

/// Robust ray-origin offset from an interval-backed hit point.
/// n = surface normal (unit, outward), w = outgoing direction.
/// Returns the offset origin as a plain (ox, oy, oz) triple.
CPU_GPU void OffsetRayOrigin(const Point3fi& pi,
								 double nx, double ny, double nz,
								 double wx, double wy, double wz,
								 double& ox, double& oy, double& oz)
{
	// d = Dot(|n|, error)
	double anx = std::abs(nx), any = std::abs(ny), anz = std::abs(nz);
	double d = anx * pi.ex() + any * pi.ey() + anz * pi.ez();

	// offset = d * n  (or -d*n if w and n point in opposite directions)
	double offx = d * nx, offy = d * ny, offz = d * nz;
	if (interval_vec_detail::dot3(wx, wy, wz, nx, ny, nz) < 0.0) {
		offx = -offx; offy = -offy; offz = -offz;
	}

	// Start from midpoint, add offset
	double px = pi.mx() + offx;
	double py = pi.my() + offy;
	double pz = pi.mz() + offz;

	// Round away from surface (pbrt-v4: NextFloatUp if offset > 0, else Down)
	ox = (offx > 0.0) ? NextFloatUp(px)   : (offx < 0.0) ? NextFloatDown(px) : px;
	oy = (offy > 0.0) ? NextFloatUp(py)   : (offy < 0.0) ? NextFloatDown(py) : py;
	oz = (offz > 0.0) ? NextFloatUp(pz)   : (offz < 0.0) ? NextFloatDown(pz) : pz;
}

/// Spawn a secondary ray from a surface hit (shadow / BSDF bounce).
/// Fills ray origin (ox,oy,oz) using robust interval-based offset.
/// The caller supplies direction (dx, dy, dz).
CPU_GPU void SpawnRay(const Point3fi& pi,
							  double nx, double ny, double nz,
							  double dx, double dy, double dz,
							  double& ox, double& oy, double& oz)
{
	OffsetRayOrigin(pi, nx, ny, nz, dx, dy, dz, ox, oy, oz);
}

/// Shadow ray toward a point: spawn from pi toward target (tx,ty,tz).
CPU_GPU void SpawnRayTo(const Point3fi& pi,
								double nx, double ny, double nz,
								double tx, double ty, double tz,
								double& ox, double& oy, double& oz)
{
	double dx = tx - pi.mx(), dy = ty - pi.my(), dz = tz - pi.mz();
	OffsetRayOrigin(pi, nx, ny, nz, dx, dy, dz, ox, oy, oz);
}

/// Shadow ray between two interval-backed endpoints.
/// Mirrors pbrt-v4 SpawnRayTo(Point3fi pFrom, Normal3f nFrom, Float time,
///                            Point3fi pTo,   Normal3f nTo).
/// Fills both offset endpoints: (ofx,ofy,ofz) for pFrom-side,
///                              (otx,oty,otz) for pTo-side.
CPU_GPU void SpawnRayTo(const Point3fi& pFrom,
								double nfx, double nfy, double nfz,
								const Point3fi& pTo,
								double ntx, double nty, double ntz,
								double& ofx, double& ofy, double& ofz,
								double& otx, double& oty, double& otz)
{
	// Step 1: offset pFrom toward pTo (pbrt-v4: pf = OffsetRayOrigin(pFrom, nFrom, pTo-pFrom))
	double dx = pTo.mx() - pFrom.mx();
	double dy = pTo.my() - pFrom.my();
	double dz = pTo.mz() - pFrom.mz();
	OffsetRayOrigin(pFrom, nfx, nfy, nfz, dx, dy, dz, ofx, ofy, ofz);

	// Step 2: offset pTo toward already-offset pf  (pbrt-v4: pt = OffsetRayOrigin(pTo, nTo, pf-pTo))
	// Using pf (ofx,ofy,ofz) as the direction source matches pbrt-v4 exactly.
	double dx2 = ofx - pTo.mx();
	double dy2 = ofy - pTo.my();
	double dz2 = ofz - pTo.mz();
	OffsetRayOrigin(pTo, ntx, nty, ntz, dx2, dy2, dz2, otx, oty, otz);
}
