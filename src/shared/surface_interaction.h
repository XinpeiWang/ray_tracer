#pragma once
// ---------------------------------------------------------------------------
// surface_interaction.h -- pbrt-v4 SurfaceInteraction + BSDF (header-only, CPU)
// Mirrors pbrt-v4 src/pbrt/interaction.h + bsdf.h
//
// Provides:
//   SurfaceInteraction<T>  -- full hit-point record bridging shapes -> materials -> lights
//   BSDF<BxDF,T>           -- shading-frame wrapper around a concrete BxDF
//
// Design notes:
//   - Templated on scalar T (float or double).
//   - No dependency on pbrt TaggedPointer, Medium, or Camera types.
//   - SurfaceInteraction bridges local ShapeHit<T> -> MaterialEvalContext<T> -> BSDF.
//   - BSDF wraps ShadingFrame<T> + BxDF, providing render-space f/sample_f/pdf.
//   - compute_differentials() mirrors pbrt-v4 ComputeDifferentials():
//       plane-intersection method using ray differentials, fallback to zero.
//   - spawn_ray() offsets the ray origin by a small epsilon along the normal to
//       avoid self-intersection (mirrors pbrt-v4 OffsetRayOrigin concept).
//
// pbrt-v4 references:
//   interaction.h  -- Interaction, SurfaceInteraction
//   interaction.cpp -- ComputeDifferentials, Le, SpawnRay
//   bsdf.h         -- BSDF::f, Sample_f, PDF, RenderToLocal, LocalToRender
// ---------------------------------------------------------------------------

#include "shading_frame.h"
#include "materials.h"   // MaterialEvalContext<T>
#include "interval_vec.h" // Point3fi, OffsetRayOrigin

#include <cmath>
#include <optional>

// ---------------------------------------------------------------------------
// SurfaceInteraction<T>
//
// Carries all state at a ray-surface intersection point.
// Mirrors pbrt-v4 SurfaceInteraction (interaction.h).
//
// Member naming follows pbrt-v4 conventions:
//   p          -- hit point (world space)
//   n          -- geometric normal (unit, world space)
//   ns         -- shading normal (unit, world space; may differ due to bump/normal map)
//   wo         -- outgoing direction (toward eye, world space)
//   uv         -- surface parameterisation at hit
//   dpdu/dpdv  -- partial derivatives of p w.r.t. u and v (world space)
//   dndu/dndv  -- partial derivatives of n w.r.t. u and v (world space)
//   shading.dpdu -- shading frame tangent (used to build BSDF frame)
//   dpdx/dpdy  -- screen-space partial derivatives (computed by compute_differentials)
//   dudx/dvdx/dudy/dvdy -- UV screen-space derivatives (texture filtering)
//   t          -- ray parameter at hit
//   time       -- ray time
//   face_index -- for meshes with per-face materials
// ---------------------------------------------------------------------------
template <typename T>
struct SurfaceInteraction {
	// -----------------------------------------------------------------------
	// Hit geometry (world space)
	// -----------------------------------------------------------------------
	T px = T(0), py = T(0), pz = T(0);   // hit point
	T nx = T(0), ny = T(1), nz = T(0);   // geometric normal (unit)
	T ns_x = T(0), ns_y = T(1), ns_z = T(0); // shading normal (unit)
	T wo_x = T(0), wo_y = T(0), wo_z = T(1); // outgoing direction (toward eye)
	T u = T(0), v = T(0);                // surface UV

	// Partial derivatives of position w.r.t. u and v (world space)
	T dpdu_x = T(1), dpdu_y = T(0), dpdu_z = T(0);
	T dpdv_x = T(0), dpdv_y = T(1), dpdv_z = T(0);

	// Partial derivatives of normal w.r.t. u and v (world space)
	T dndu_x = T(0), dndu_y = T(0), dndu_z = T(0);
	T dndv_x = T(0), dndv_y = T(0), dndv_z = T(0);

	// Shading frame tangent and bitangent (may differ from dpdu/dpdv if normal-mapped)
	// pbrt-v4 stores shading.dpdu / shading.dpdv
	T shading_dpdu_x = T(1), shading_dpdu_y = T(0), shading_dpdu_z = T(0);
	T shading_dpdv_x = T(0), shading_dpdv_y = T(1), shading_dpdv_z = T(0);

	// Screen-space differentials (computed by compute_differentials)
	T dpdx_x = T(0), dpdx_y = T(0), dpdx_z = T(0);
	T dpdy_x = T(0), dpdy_y = T(0), dpdy_z = T(0);
	T dudx = T(0), dvdx = T(0), dudy = T(0), dvdy = T(0);

	// Ray parameter, time, and mesh face index
	T t    = T(0);
	T time = T(0);
	int face_index = 0;

	// Per-component absolute position-error bounds (pbrt-v4 pError / Point3fi).
	// Set by shapes via set_error(); zero means fall back to eps-bias in spawn_ray_origin.
	T pex = T(0), pey = T(0), pez = T(0);

	// Set position error from a ShapeHit (or any computed gamma-bound triple).
	void set_error(T ex, T ey, T ez) { pex = ex; pey = ey; pez = ez; }

	// Build a Point3fi from the stored hit point + error (for OffsetRayOrigin).
	Point3fi to_point3fi() const {
		return Point3fi(double(px), double(py), double(pz),
						double(pex), double(pey), double(pez));
	}

	// -----------------------------------------------------------------------
	// Default constructor
	// -----------------------------------------------------------------------
	SurfaceInteraction() = default;

	// -----------------------------------------------------------------------
	// Construct from shape hit data.
	// Mirrors pbrt-v4 SurfaceInteraction(pi, uv, wo, dpdu, dpdv, dndu, dndv, time, flip).
	//
	// Parameters:
	//   hit_p*     -- world-space hit point
	//   hit_n*     -- world-space geometric normal (unit)
	//   hit_u/v    -- surface UV
	//   hit_t      -- ray parameter
	//   wo_*       -- outgoing direction (toward eye, world space)
	//   dpdu_*     -- dp/du (world space, used for shading frame + differentials)
	//   dpdv_*     -- dp/dv (world space)
	//   flip_normal -- if true, negate geometric and shading normal (pbrt-v4 flipNormal)
	// -----------------------------------------------------------------------
	SurfaceInteraction(T hit_px, T hit_py, T hit_pz,
					   T hit_nx, T hit_ny, T hit_nz,
					   T hit_u,  T hit_v,
					   T hit_t,
					   T wo_x_, T wo_y_, T wo_z_,
					   T dpdu_x_, T dpdu_y_, T dpdu_z_,
					   T dpdv_x_, T dpdv_y_, T dpdv_z_,
					   bool flip_normal = false)
		: px(hit_px), py(hit_py), pz(hit_pz)
		, nx(hit_nx), ny(hit_ny), nz(hit_nz)
		, ns_x(hit_nx), ns_y(hit_ny), ns_z(hit_nz)
		, wo_x(wo_x_), wo_y(wo_y_), wo_z(wo_z_)
		, u(hit_u), v(hit_v)
		, dpdu_x(dpdu_x_), dpdu_y(dpdu_y_), dpdu_z(dpdu_z_)
		, dpdv_x(dpdv_x_), dpdv_y(dpdv_y_), dpdv_z(dpdv_z_)
		, shading_dpdu_x(dpdu_x_), shading_dpdu_y(dpdu_y_), shading_dpdu_z(dpdu_z_)
		, shading_dpdv_x(dpdv_x_), shading_dpdv_y(dpdv_y_), shading_dpdv_z(dpdv_z_)
		, t(hit_t)
	{
		if (flip_normal) {
			nx = -nx; ny = -ny; nz = -nz;
			ns_x = -ns_x; ns_y = -ns_y; ns_z = -ns_z;
		}
	}

	// -----------------------------------------------------------------------
	// set_shading_geometry -- override shading normal and tangent.
	// Mirrors pbrt-v4 SurfaceInteraction::SetShadingGeometry().
	//
	// If orientation_authoritative: adjust geometric normal to match shading normal.
	// Otherwise: adjust shading normal to match geometric normal hemisphere.
	// -----------------------------------------------------------------------
	// Convenience overload: update shading normal + tangent only; dpdv stays unchanged.
	// Mirrors pbrt-v4 call sites that supply (ns, dpdus, dpdvs, dndus, dndvs, authoritativeFlag)
	// but don't need to update dpdv explicitly.
	void set_shading_geometry(T new_ns_x, T new_ns_y, T new_ns_z,
							  T new_dpdu_x_, T new_dpdu_y_, T new_dpdu_z_,
							  bool orientation_authoritative) {
		set_shading_geometry(new_ns_x, new_ns_y, new_ns_z,
							 new_dpdu_x_, new_dpdu_y_, new_dpdu_z_,
							 shading_dpdv_x, shading_dpdv_y, shading_dpdv_z,
							 orientation_authoritative);
	}

	void set_shading_geometry(T new_ns_x, T new_ns_y, T new_ns_z,
							  T new_dpdu_x_, T new_dpdu_y_, T new_dpdu_z_,
							  T new_dpdv_x_, T new_dpdv_y_, T new_dpdv_z_,
							  bool orientation_authoritative = true) {
		// Normalize incoming shading normal (mirrors pbrt-v4 shading.n = ns)
		T len = std::sqrt(new_ns_x*new_ns_x + new_ns_y*new_ns_y + new_ns_z*new_ns_z);
		if (len > T(0)) { new_ns_x /= len; new_ns_y /= len; new_ns_z /= len; }

		if (orientation_authoritative) {
			// pbrt-v4: n = FaceForward(n, shading.n) when orientation is authoritative
			T dot = nx*new_ns_x + ny*new_ns_y + nz*new_ns_z;
			if (dot < T(0)) { nx = -nx; ny = -ny; nz = -nz; }
		} else {
			// pbrt-v4: shading.n = FaceForward(shading.n, n)
			T dot = new_ns_x*nx + new_ns_y*ny + new_ns_z*nz;
			if (dot < T(0)) { new_ns_x = -new_ns_x; new_ns_y = -new_ns_y; new_ns_z = -new_ns_z; }
		}

		ns_x = new_ns_x; ns_y = new_ns_y; ns_z = new_ns_z;

		// pbrt-v4: clamp shading dpdu/dpdv if length^2 > 1e16 (avoid overflow)
		T len2u = new_dpdu_x_*new_dpdu_x_ + new_dpdu_y_*new_dpdu_y_ + new_dpdu_z_*new_dpdu_z_;
		T len2v = new_dpdv_x_*new_dpdv_x_ + new_dpdv_y_*new_dpdv_y_ + new_dpdv_z_*new_dpdv_z_;
		while (len2u > T(1e16) || len2v > T(1e16)) {
			new_dpdu_x_ *= T(1e-8); new_dpdu_y_ *= T(1e-8); new_dpdu_z_ *= T(1e-8);
			new_dpdv_x_ *= T(1e-8); new_dpdv_y_ *= T(1e-8); new_dpdv_z_ *= T(1e-8);
			len2u = new_dpdu_x_*new_dpdu_x_ + new_dpdu_y_*new_dpdu_y_ + new_dpdu_z_*new_dpdu_z_;
			len2v = new_dpdv_x_*new_dpdv_x_ + new_dpdv_y_*new_dpdv_y_ + new_dpdv_z_*new_dpdv_z_;
		}
		shading_dpdu_x = new_dpdu_x_; shading_dpdu_y = new_dpdu_y_; shading_dpdu_z = new_dpdu_z_;
		shading_dpdv_x = new_dpdv_x_; shading_dpdv_y = new_dpdv_y_; shading_dpdv_z = new_dpdv_z_;
	}

	// -----------------------------------------------------------------------
	// to_material_context -- build a MaterialEvalContext from this interaction.
	// This is the bridge from SurfaceInteraction -> Material::get_bxdf().
	// -----------------------------------------------------------------------
	MaterialEvalContext<T> to_material_context() const {
		MaterialEvalContext<T> ctx;
		ctx.px = px; ctx.py = py; ctx.pz = pz;
		ctx.nx = ns_x; ctx.ny = ns_y; ctx.nz = ns_z;  // shading normal
		ctx.dpdu_x = shading_dpdu_x; ctx.dpdu_y = shading_dpdu_y; ctx.dpdu_z = shading_dpdu_z;
		ctx.wo_x = wo_x; ctx.wo_y = wo_y; ctx.wo_z = wo_z;
		ctx.u = u; ctx.v = v;
		ctx.dudx = dudx; ctx.dudy = dudy; ctx.dvdx = dvdx; ctx.dvdy = dvdy;
		ctx.dpdx_x = dpdx_x; ctx.dpdx_y = dpdx_y; ctx.dpdx_z = dpdx_z;
		ctx.dpdy_x = dpdy_x; ctx.dpdy_y = dpdy_y; ctx.dpdy_z = dpdy_z;
		ctx.faceIndex = face_index;
		return ctx;
	}

	// -----------------------------------------------------------------------
	// compute_differentials -- estimate screen-space (dpdx, dpdy) and UV
	// derivatives (dudx, dvdx, dudy, dvdy) from ray differentials.
	//
	// Mirrors pbrt-v4 SurfaceInteraction::ComputeDifferentials().
	//
	// Ray differential inputs (world space):
	//   rx_o*, rx_d*  -- origin and direction of x-offset ray
	//   ry_o*, ry_d*  -- origin and direction of y-offset ray
	//   has_differentials -- if false, zeroes all derivatives
	// -----------------------------------------------------------------------
	void compute_differentials(bool has_differentials,
							   T rx_ox, T rx_oy, T rx_oz,
							   T rx_dx, T rx_dy, T rx_dz,
							   T ry_ox, T ry_oy, T ry_oz,
							   T ry_dx, T ry_dy, T ry_dz) {
		// If no differentials, zero everything
		if (!has_differentials ||
			(nx*rx_dx + ny*rx_dy + nz*rx_dz == T(0)) ||
			(nx*ry_dx + ny*ry_dy + nz*ry_dz == T(0))) {
			dpdx_x = dpdx_y = dpdx_z = T(0);
			dpdy_x = dpdy_y = dpdy_z = T(0);
			dudx = dvdx = dudy = dvdy = T(0);
			return;
		}

		// Plane at hit: dot(n, p) = d
		T d = -(nx*px + ny*py + nz*pz);

		// Intersect x-differential ray with the tangent plane
		T denom_x = nx*rx_dx + ny*rx_dy + nz*rx_dz;
		T tx_ = -(nx*rx_ox + ny*rx_oy + nz*rx_oz + d) / denom_x;
		T pxx = rx_ox + tx_ * rx_dx;
		T pxy = rx_oy + tx_ * rx_dy;
		T pxz = rx_oz + tx_ * rx_dz;

		// Intersect y-differential ray
		T denom_y = nx*ry_dx + ny*ry_dy + nz*ry_dz;
		T ty_ = -(nx*ry_ox + ny*ry_oy + nz*ry_oz + d) / denom_y;
		T pyx = ry_ox + ty_ * ry_dx;
		T pyy = ry_oy + ty_ * ry_dy;
		T pyz = ry_oz + ty_ * ry_dz;

		// dpdx = px - p,  dpdy = py - p
		dpdx_x = pxx - px; dpdx_y = pxy - py; dpdx_z = pxz - pz;
		dpdy_x = pyx - px; dpdy_y = pyy - py; dpdy_z = pyz - pz;

		// Estimate dudx, dvdx, dudy, dvdy via least squares:
		//   A = [[dpdu · dpdu, dpdu · dpdv],
		//        [dpdu · dpdv, dpdv · dpdv]]
		//   solve A * [du, dv]^T = [dpdu · dpdxy, dpdv · dpdxy] for x and y
		//
		// pbrt-v4 uses DifferenceOfProducts (FMA-based) for the determinant.
		// We use the same formula; invDet is zeroed if non-finite (degenerate).
		T ata00 = dpdu_x*dpdu_x + dpdu_y*dpdu_y + dpdu_z*dpdu_z;
		T ata01 = dpdu_x*dpdv_x + dpdu_y*dpdv_y + dpdu_z*dpdv_z;
		T ata11 = dpdv_x*dpdv_x + dpdv_y*dpdv_y + dpdv_z*dpdv_z;
		T det = ata00*ata11 - ata01*ata01;
		T inv_det = T(1) / det;
		if (!std::isfinite(inv_det)) inv_det = T(0);  // degenerate surface

		// For x
		T atb0x = dpdu_x*dpdx_x + dpdu_y*dpdx_y + dpdu_z*dpdx_z;
		T atb1x = dpdv_x*dpdx_x + dpdv_y*dpdx_y + dpdv_z*dpdx_z;
		dudx = (ata11*atb0x - ata01*atb1x) * inv_det;
		dvdx = (ata00*atb1x - ata01*atb0x) * inv_det;

		// For y
		T atb0y = dpdu_x*dpdy_x + dpdu_y*dpdy_y + dpdu_z*dpdy_z;
		T atb1y = dpdv_x*dpdy_x + dpdv_y*dpdy_y + dpdv_z*dpdy_z;
		dudy = (ata11*atb0y - ata01*atb1y) * inv_det;
		dvdy = (ata00*atb1y - ata01*atb0y) * inv_det;

		// Clamp to reasonable values; zero out any non-finite results.
		// Mirrors pbrt-v4: IsFinite(x) ? Clamp(x, -1e8, 1e8) : 0
		T lim = T(1e8);
		auto safe_clamp = [lim](T x) -> T {
			return std::isfinite(x) ? (x < -lim ? -lim : (x > lim ? lim : x)) : T(0);
		};
		dudx = safe_clamp(dudx);
		dvdx = safe_clamp(dvdx);
		dudy = safe_clamp(dudy);
		dvdy = safe_clamp(dvdy);
	}

	// -----------------------------------------------------------------------
	// spawn_ray_origin -- compute origin of a spawned ray offset along the
	// geometric normal to avoid self-intersection.
	// Mirrors pbrt-v4 Interaction::OffsetRayOrigin(Vector3f w).
	//
	// When position error bounds (pex/pey/pez) have been set by the shape
	// intersector, uses the pbrt-v4 interval-backed OffsetRayOrigin algorithm
	// for a mathematically tight, geometry-derived offset.
	// Falls back to a fixed eps-bias when no error bounds are available
	// (e.g., procedural geometry or legacy call sites).
	void spawn_ray_origin(T& ox, T& oy, T& oz,
						  T wx, T wy, T wz,
						  T eps = T(1e-4)) const {
		if (pex != T(0) || pey != T(0) || pez != T(0)) {
			// pbrt-v4 path: interval-backed OffsetRayOrigin (always double precision)
			double dox, doy, doz;
			OffsetRayOrigin(to_point3fi(),
							double(nx), double(ny), double(nz),
							double(wx), double(wy), double(wz),
							dox, doy, doz);
			ox = T(dox); oy = T(doy); oz = T(doz);
		} else {
			// Legacy fallback: fixed eps-bias along normal
			T sign = (nx*wx + ny*wy + nz*wz >= T(0)) ? T(1) : T(-1);
			ox = px + sign * eps * nx;
			oy = py + sign * eps * ny;
			oz = pz + sign * eps * nz;
		}
	}

	// Overload: explicit sign (+1 front-side, -1 back-side).
	// Kept for backward compatibility with existing call sites.
	// Uses OffsetRayOrigin when errors are set (direction = sign * n).
	void spawn_ray_origin(T& ox, T& oy, T& oz,
						  T sign_along_normal,
						  T eps = T(1e-4)) const {
		if (pex != T(0) || pey != T(0) || pez != T(0)) {
			// Direction is sign_along_normal * n
			double dox, doy, doz;
			OffsetRayOrigin(to_point3fi(),
							double(nx), double(ny), double(nz),
							double(sign_along_normal) * double(nx),
							double(sign_along_normal) * double(ny),
							double(sign_along_normal) * double(nz),
							dox, doy, doz);
			ox = T(dox); oy = T(doy); oz = T(doz);
		} else {
			ox = px + sign_along_normal * eps * nx;
			oy = py + sign_along_normal * eps * ny;
			oz = pz + sign_along_normal * eps * nz;
		}
	}

	// -----------------------------------------------------------------------
	// geometric_normal_dot -- dot product of geometric normal with a direction
	// -----------------------------------------------------------------------
	T geo_dot(T wx, T wy, T wz) const {
		return nx*wx + ny*wy + nz*wz;
	}

	// -----------------------------------------------------------------------
	// shading_normal_dot -- dot product of shading normal with a direction
	// -----------------------------------------------------------------------
	T shading_dot(T wx, T wy, T wz) const {
		return ns_x*wx + ns_y*wy + ns_z*wz;
	}
};

// ---------------------------------------------------------------------------
// BSDF<BxDF, T>
//
// Shading-frame wrapper around a concrete BxDF.
// Mirrors pbrt-v4 BSDF (bsdf.h).
//
// All directions passed to f/sample_f/pdf are in render (world) space.
// The BSDF transforms them to/from local shading space internally.
//
// Template parameters:
//   BxDF -- any type with:
//     BxDFSampleResult<T> sample(T wo_lx,lo_ly,wo_lz, T u0,u1,u2) const
//     T eval(T wo_lx,..., T wi_lx,...) const  (returns R+G+B weight)
//     T pdf(T wo_lx,..., T wi_lx,...) const
//   T    -- scalar type (float or double)
// ---------------------------------------------------------------------------
template <typename BxDF, typename T = float>
struct BSDF {
	BxDF           bxdf;
	ShadingFrame<T> frame;   // shading frame (z = shading normal)

	// -----------------------------------------------------------------------
	// Construct from a BxDF and a SurfaceInteraction.
	// Builds the shading frame from the interaction's shading normal + tangent.
	// Mirrors pbrt-v4: BSDF(dpdus, ns, bxdf) -> shadingFrame = Frame::FromXZ(dpdus, ns)
	// -----------------------------------------------------------------------
	BSDF() = default;

	BSDF(const BxDF& bxdf_, const SurfaceInteraction<T>& si)
		: bxdf(bxdf_)
	{
		// Build frame from shading tangent (shading_dpdu) + shading normal (ns).
		// pbrt-v4: shadingFrame = Frame::FromXZ(Normalize(dpdus), Vector3f(ns))
		// This pins the X-axis to the surface tangent so texture derivatives
		// computed by compute_differentials() align correctly with the frame.
		T tx = si.shading_dpdu_x, ty = si.shading_dpdu_y, tz = si.shading_dpdu_z;
		T tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
		if (tlen > T(0)) { tx /= tlen; ty /= tlen; tz /= tlen; }
		frame = ShadingFrame<T>::from_xz(tx, ty, tz, si.ns_x, si.ns_y, si.ns_z);
	}

	// Direct construction from BxDF + shading normal
	BSDF(const BxDF& bxdf_, T ns_x, T ns_y, T ns_z)
		: bxdf(bxdf_), frame(ShadingFrame<T>::from_normal(ns_x, ns_y, ns_z)) {}

	// -----------------------------------------------------------------------
	// render_to_local -- transform render-space direction to shading-local space
	// Mirrors pbrt-v4 BSDF::RenderToLocal()
	// -----------------------------------------------------------------------
	void render_to_local(T wx, T wy, T wz, T& lx, T& ly, T& lz) const {
		frame.to_local(wx, wy, wz, lx, ly, lz);
	}

	// -----------------------------------------------------------------------
	// local_to_render -- transform shading-local direction to render space
	// Mirrors pbrt-v4 BSDF::LocalToRender()
	// -----------------------------------------------------------------------
	void local_to_render(T lx, T ly, T lz, T& wx, T& wy, T& wz) const {
		frame.to_world(lx, ly, lz, wx, wy, wz);
	}

	// -----------------------------------------------------------------------
	// f -- evaluate BSDF for given render-space wo and wi.
	// Returns attenuation (r,g,b). Returns {0,0,0} if wo is in the tangent plane.
	// Mirrors pbrt-v4 BSDF::f(woRender, wiRender).
	// -----------------------------------------------------------------------
	void f(T wo_wx, T wo_wy, T wo_wz,
		   T wi_wx, T wi_wy, T wi_wz,
		   T& out_r, T& out_g, T& out_b) const {
		T wo_lx, wo_ly, wo_lz;
		render_to_local(wo_wx, wo_wy, wo_wz, wo_lx, wo_ly, wo_lz);
		if (wo_lz == T(0)) { out_r = out_g = out_b = T(0); return; }

		T wi_lx, wi_ly, wi_lz;
		render_to_local(wi_wx, wi_wy, wi_wz, wi_lx, wi_ly, wi_lz);

		// Call BxDF eval -- our BxDFs return a single scalar (Lambertian) or
		// three via sample(); we use sample with u=0,0,0 as fallback if needed.
		// For proper eval, BxDF must provide eval(wo_l, wi_l) -> (r,g,b).
		bxdf.eval(wo_lx, wo_ly, wo_lz, wi_lx, wi_ly, wi_lz, out_r, out_g, out_b);
	}

	// -----------------------------------------------------------------------
	// sample_f -- sample BSDF for given render-space wo.
	// Returns BxDFSampleResult with wi in render space.
	// Mirrors pbrt-v4 BSDF::Sample_f(woRender, u, u2, mode, sampleFlags).
	//
	// u0,u1,u2: three uniform random numbers in [0,1].
	// -----------------------------------------------------------------------
	auto sample_f(T wo_wx, T wo_wy, T wo_wz,
				  T u0, T u1, T u2) const {
		T wo_lx, wo_ly, wo_lz;
		render_to_local(wo_wx, wo_wy, wo_wz, wo_lx, wo_ly, wo_lz);

		// Sample the BxDF in local space
		auto bs = bxdf.sample(wo_lx, wo_ly, wo_lz, u0, u1, u2);

		if (!bs.valid || bs.wo_z == T(0)) {
			bs.valid = false;
			return bs;
		}

		// Transform sampled wi back to render space
		T wi_wx, wi_wy, wi_wz;
		local_to_render(bs.wo_x, bs.wo_y, bs.wo_z, wi_wx, wi_wy, wi_wz);
		bs.wo_x = wi_wx; bs.wo_y = wi_wy; bs.wo_z = wi_wz;
		return bs;
	}

	// -----------------------------------------------------------------------
	// pdf -- evaluate sampling PDF for render-space wo and wi.
	// Mirrors pbrt-v4 BSDF::PDF(woRender, wiRender).
	// -----------------------------------------------------------------------
	T pdf(T wo_wx, T wo_wy, T wo_wz,
		  T wi_wx, T wi_wy, T wi_wz) const {
		T wo_lx, wo_ly, wo_lz;
		render_to_local(wo_wx, wo_wy, wo_wz, wo_lx, wo_ly, wo_lz);
		if (wo_lz == T(0)) return T(0);

		T wi_lx, wi_ly, wi_lz;
		render_to_local(wi_wx, wi_wy, wi_wz, wi_lx, wi_ly, wi_lz);

		return bxdf.pdf(wo_lx, wo_ly, wo_lz, wi_lx, wi_ly, wi_lz);
	}
};
