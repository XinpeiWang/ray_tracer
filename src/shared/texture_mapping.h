#pragma once
// ---------------------------------------------------------------------------
// texture_mapping.h -- Texture coordinate mapping (pbrt-v4 textures.h port)
//
// pbrt-v4 reference: src/pbrt/textures.h
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides:
//   TextureEvalContext    -- surface point + differentials for texture eval
//   TexCoord2D            -- (s,t) + partial derivatives
//   TexCoord3D            -- 3D point + differentials
//   TextureTransform      -- lightweight 4x4 affine transform (point + vector apply)
//   UVMapping             -- scale/offset UV mapping (pbrt-v4 UVMapping)
//   SphericalMapping      -- spherical environment projection (pbrt-v4 SphericalMapping)
//   CylindricalMapping    -- cylindrical projection (pbrt-v4 CylindricalMapping)
//   PlanarMapping         -- planar 2-axis projection (pbrt-v4 PlanarMapping)
//   TextureMapping2D      -- variant over the 4 2D mappings
//   PointTransformMapping -- 3D identity/affine mapping (pbrt-v4 PointTransformMapping)
//   TextureMapping3D      -- variant holding PointTransformMapping
//
// Design notes:
//   - Header-only; no allocator, no TaggedPointer, no pbrt infrastructure.
//   - TextureMapping2D / TextureMapping3D use std::variant for dispatch.
//   - TextureTransform wraps SquareMatrix<4> from square_matrix.h.
//   - Floating-point type is float throughout (matching pbrt-v4 Float=float).
//   - SphericalTheta/Phi use acos/atan2; constants from scalar_math.h.
// ---------------------------------------------------------------------------

#include "square_matrix.h"   // SquareMatrix<4>, Inverse

#include <cmath>
#include <variant>
#include <optional>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

// Pi constants local to this header (avoid dependency on scalar_math_detail namespace)
namespace tx_consts {
static constexpr double kPi     = 3.14159265358979323846;
static constexpr double kInvPi  = 0.31830988618379067154;
static constexpr double kInv2Pi = 0.15915494309189533577;
template<typename T> CPU_GPU T Sqr(T v) { return v * v; }
} // namespace tx_consts

// ---------------------------------------------------------------------------
// Lightweight geometry types local to this header
// (float-precision; mirrors pbrt-v4 Point3f / Vector3f / Point2f / Normal3f)
// ---------------------------------------------------------------------------
struct TxPoint3f  { float x, y, z; };
struct TxVector3f { float x, y, z; };
struct TxNormal3f { float x, y, z; };
struct TxPoint2f  { float x, y; };

namespace tx_detail {

CPU_GPU float dot(TxVector3f a, TxVector3f b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
CPU_GPU float length(TxVector3f v) {
	return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
CPU_GPU TxVector3f normalize(TxVector3f v) {
	float len = length(v);
	return { v.x/len, v.y/len, v.z/len };
}

// SphericalTheta: angle from +z axis (0 at north pole, pi at south pole)
// Mirrors pbrt-v4 SphericalTheta(Vector3f v)
CPU_GPU float SphericalTheta(TxVector3f v) {
	return std::acos(std::fmax(-1.f, std::fmin(1.f, v.z)));
}

// SphericalPhi: azimuthal angle in [0, 2pi)
// Mirrors pbrt-v4 SphericalPhi(Vector3f v)
CPU_GPU float SphericalPhi(TxVector3f v) {
	float p = std::atan2(v.y, v.x);
	return (p < 0.f) ? p + 2.f * static_cast<float>(tx_consts::kPi) : p;
}

} // namespace tx_detail

// ---------------------------------------------------------------------------
// TextureTransform
//
// Lightweight wrapper over SquareMatrix<4> that applies a 4x4 affine transform
// to points and vectors, matching pbrt-v4 Transform::operator()(Point3f/Vector3f).
// Stores forward matrix and pre-computed inverse.
// ---------------------------------------------------------------------------
struct TextureTransform {
	SquareMatrix<4> m;     // forward
	SquareMatrix<4> mInv;  // inverse

	// Identity transform
	TextureTransform() : m(), mInv() {}

	// Construct from explicit forward and inverse matrices
	TextureTransform(const SquareMatrix<4>& fwd, const SquareMatrix<4>& inv)
		: m(fwd), mInv(inv) {}

	// Construct from forward matrix; inverse computed automatically
	explicit TextureTransform(const SquareMatrix<4>& fwd) : m(fwd) {
		auto inv = Inverse(fwd);
		mInv = inv.value_or(SquareMatrix<4>());
	}

	// Apply to point (w=1 homogeneous divide)
	CPU_GPU TxPoint3f operator()(TxPoint3f p) const {
		float x = (float)(m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3]);
		float y = (float)(m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3]);
		float z = (float)(m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3]);
		float w = (float)(m[3][0]*p.x + m[3][1]*p.y + m[3][2]*p.z + m[3][3]);
		if (w == 1.f) return {x, y, z};
		float inv_w = 1.f / w;
		return {x*inv_w, y*inv_w, z*inv_w};
	}

	// Apply to vector (w=0, no translation)
	CPU_GPU TxVector3f operator()(TxVector3f v) const {
		float x = (float)(m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z);
		float y = (float)(m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z);
		float z = (float)(m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z);
		return {x, y, z};
	}

	// Factory: scale
	static TextureTransform Scale(float sx, float sy, float sz) {
		double d[4][4] = {
			{sx,  0,  0, 0},
			{ 0, sy,  0, 0},
			{ 0,  0, sz, 0},
			{ 0,  0,  0, 1}
		};
		double di[4][4] = {
			{1.0/sx,      0,      0, 0},
			{     0, 1.0/sy,      0, 0},
			{     0,      0, 1.0/sz, 0},
			{     0,      0,      0, 1}
		};
		return TextureTransform(SquareMatrix<4>(d), SquareMatrix<4>(di));
	}

	// Factory: translation
	static TextureTransform Translate(float tx, float ty, float tz) {
		double d[4][4] = {
			{1, 0, 0, tx},
			{0, 1, 0, ty},
			{0, 0, 1, tz},
			{0, 0, 0,  1}
		};
		double di[4][4] = {
			{1, 0, 0, -tx},
			{0, 1, 0, -ty},
			{0, 0, 1, -tz},
			{0, 0, 0,   1}
		};
		return TextureTransform(SquareMatrix<4>(d), SquareMatrix<4>(di));
	}

	bool IsIdentity() const { return m.IsIdentity(); }
};

// ---------------------------------------------------------------------------
// TextureEvalContext
//
// Bundles the surface-point data needed to evaluate a texture mapping.
// Mirrors pbrt-v4 TextureEvalContext (textures.h).
// ---------------------------------------------------------------------------
struct TextureEvalContext {
	TxPoint3f  p   = {0,0,0};      // surface point in render space
	TxVector3f dpdx = {0,0,0};     // dp/dx screen-space differential
	TxVector3f dpdy = {0,0,0};     // dp/dy screen-space differential
	TxNormal3f n   = {0,0,1};      // shading normal
	TxPoint2f  uv  = {0,0};        // (u,v) surface parameters
	float dudx = 0, dudy = 0;
	float dvdx = 0, dvdy = 0;
	int faceIndex = 0;

	TextureEvalContext() = default;

	TextureEvalContext(TxPoint3f p, TxVector3f dpdx, TxVector3f dpdy,
					   TxNormal3f n, TxPoint2f uv,
					   float dudx, float dudy, float dvdx, float dvdy,
					   int faceIndex = 0)
		: p(p), dpdx(dpdx), dpdy(dpdy), n(n), uv(uv),
		  dudx(dudx), dudy(dudy), dvdx(dvdx), dvdy(dvdy),
		  faceIndex(faceIndex) {}
};

// ---------------------------------------------------------------------------
// TexCoord2D
//
// 2D texture coordinates plus partial derivatives for filtering.
// Mirrors pbrt-v4 TexCoord2D.
// ---------------------------------------------------------------------------
struct TexCoord2D {
	TxPoint2f st = {0,0};
	float dsdx = 0, dsdy = 0;
	float dtdx = 0, dtdy = 0;
};

// ---------------------------------------------------------------------------
// TexCoord3D
//
// 3D texture coordinate (for solid textures) plus differentials.
// Mirrors pbrt-v4 TexCoord3D.
// ---------------------------------------------------------------------------
struct TexCoord3D {
	TxPoint3f  p    = {0,0,0};
	TxVector3f dpdx = {0,0,0};
	TxVector3f dpdy = {0,0,0};
};

// ---------------------------------------------------------------------------
// UVMapping
//
// Standard UV mapping with per-axis scale (su, sv) and offset (du, dv).
// Mirrors pbrt-v4 UVMapping.
// ---------------------------------------------------------------------------
class UVMapping {
public:
	// su,sv: scale; du,dv: offset (all default to identity mapping)
	explicit UVMapping(float su = 1.f, float sv = 1.f,
					   float du = 0.f, float dv = 0.f)
		: su_(su), sv_(sv), du_(du), dv_(dv) {}

	CPU_GPU TexCoord2D Map(const TextureEvalContext& ctx) const {
		// Compute texture differentials for 2D (u,v) mapping
		// pbrt-v4: dsdx = su*ctx.dudx, dsdy = su*ctx.dudy, etc.
		float dsdx = su_ * ctx.dudx, dsdy = su_ * ctx.dudy;
		float dtdx = sv_ * ctx.dvdx, dtdy = sv_ * ctx.dvdy;
		TxPoint2f st{su_ * ctx.uv.x + du_, sv_ * ctx.uv.y + dv_};
		return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
	}

	float su() const { return su_; }
	float sv() const { return sv_; }
	float du() const { return du_; }
	float dv() const { return dv_; }

private:
	float su_, sv_, du_, dv_;
};

// ---------------------------------------------------------------------------
// SphericalMapping
//
// Projects a surface point onto a unit sphere centered at the origin of the
// texture space, returning (phi/(2pi), theta/pi) as (s,t).
// Mirrors pbrt-v4 SphericalMapping.
// ---------------------------------------------------------------------------
class SphericalMapping {
public:
	explicit SphericalMapping(const TextureTransform& textureFromRender)
		: textureFromRender_(textureFromRender) {}

	CPU_GPU TexCoord2D Map(const TextureEvalContext& ctx) const {
		using namespace tx_detail;
		TxPoint3f pt = textureFromRender_(ctx.p);

		// Compute partial derivatives for the spherical (s,t) parameters.
		// pbrt-v4: dsdp = (-y, x, 0) / (2*pi*(x^2+y^2))
		//          dtdp = 1/(pi*(x^2+y^2+z^2)) * (xz/r, yz/r, -r)  where r=sqrt(x^2+y^2)
		float x2y2     = tx_consts::Sqr(pt.x) + tx_consts::Sqr(pt.y);
		float sqrtx2y2 = std::sqrt(x2y2);
		float inv2pix2y2 = (x2y2 > 0.f) ? (1.f / (2.f * (float)tx_consts::kPi * x2y2)) : 0.f;
		TxVector3f dsdp = { -pt.y * inv2pix2y2,  pt.x * inv2pix2y2, 0.f };

		float denom = (float)tx_consts::kPi * (x2y2 + tx_consts::Sqr(pt.z));
		float inv_denom = (denom > 0.f) ? (1.f / denom) : 0.f;
		float rInv = (sqrtx2y2 > 0.f) ? (1.f / sqrtx2y2) : 0.f;
		TxVector3f dtdp = {
			 pt.x * pt.z * rInv * inv_denom,
			 pt.y * pt.z * rInv * inv_denom,
			-sqrtx2y2 * inv_denom
		};

		// Transform screen-space differentials into texture space
		TxVector3f dpdx = textureFromRender_(ctx.dpdx);
		TxVector3f dpdy = textureFromRender_(ctx.dpdy);
		float dsdx = dot(dsdp, dpdx), dsdy = dot(dsdp, dpdy);
		float dtdx = dot(dtdp, dpdx), dtdy = dot(dtdp, dpdy);

		// Spherical (s,t): s = theta/pi,  t = phi/(2pi)
		// pbrt-v4: st(SphericalTheta(vec) * InvPi, SphericalPhi(vec) * Inv2Pi)
		TxVector3f vec = normalize({pt.x, pt.y, pt.z});
		TxPoint2f st{
			SphericalTheta(vec) * (float)tx_consts::kInvPi,
			SphericalPhi(vec)   * (float)tx_consts::kInv2Pi
		};
		return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
	}

	const TextureTransform& GetTransform() const { return textureFromRender_; }

private:
	TextureTransform textureFromRender_;
};

// ---------------------------------------------------------------------------
// CylindricalMapping
//
// Projects onto a cylinder aligned with the z-axis.
// s = (pi + atan2(y,x)) / (2*pi),  t = z.
// Mirrors pbrt-v4 CylindricalMapping.
// ---------------------------------------------------------------------------
class CylindricalMapping {
public:
	explicit CylindricalMapping(const TextureTransform& textureFromRender)
		: textureFromRender_(textureFromRender) {}

	CPU_GPU TexCoord2D Map(const TextureEvalContext& ctx) const {
		using namespace tx_detail;
		TxPoint3f pt = textureFromRender_(ctx.p);

		// Partial derivatives: ds/dp = (-y, x, 0)/(2*pi*(x^2+y^2)),  dt/dp = (0,0,1)
		float x2y2 = tx_consts::Sqr(pt.x) + tx_consts::Sqr(pt.y);
		float inv2pix2y2 = (x2y2 > 0.f) ? (1.f / (2.f * (float)tx_consts::kPi * x2y2)) : 0.f;
		TxVector3f dsdp = { -pt.y * inv2pix2y2, pt.x * inv2pix2y2, 0.f };
		TxVector3f dtdp = { 0.f, 0.f, 1.f };

		TxVector3f dpdx = textureFromRender_(ctx.dpdx);
		TxVector3f dpdy = textureFromRender_(ctx.dpdy);
		float dsdx = dot(dsdp, dpdx), dsdy = dot(dsdp, dpdy);
		float dtdx = dot(dtdp, dpdx), dtdy = dot(dtdp, dpdy);

		TxPoint2f st{
			((float)tx_consts::kPi + std::atan2(pt.y, pt.x)) * (float)tx_consts::kInv2Pi,
			pt.z
		};
		return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
	}

	const TextureTransform& GetTransform() const { return textureFromRender_; }

private:
	TextureTransform textureFromRender_;
};

// ---------------------------------------------------------------------------
// PlanarMapping
//
// Projects a point onto a plane via two arbitrary axes vs, vt with offsets ds, dt.
// s = ds + dot(vs, p),  t = dt + dot(vt, p).
// Mirrors pbrt-v4 PlanarMapping.
// ---------------------------------------------------------------------------
class PlanarMapping {
public:
	PlanarMapping(const TextureTransform& textureFromRender,
				  TxVector3f vs, TxVector3f vt,
				  float ds = 0.f, float dt = 0.f)
		: textureFromRender_(textureFromRender),
		  vs_(vs), vt_(vt), ds_(ds), dt_(dt) {}

	CPU_GPU TexCoord2D Map(const TextureEvalContext& ctx) const {
		using namespace tx_detail;
		// pbrt-v4: Vector3f vec(textureFromRender(ctx.p)) -- transform point once
		TxPoint3f tp = textureFromRender_(ctx.p);
		TxVector3f vec{ tp.x, tp.y, tp.z };
		TxVector3f dpdx = textureFromRender_(ctx.dpdx);
		TxVector3f dpdy = textureFromRender_(ctx.dpdy);
		float dsdx = dot(vs_, dpdx), dsdy = dot(vs_, dpdy);
		float dtdx = dot(vt_, dpdx), dtdy = dot(vt_, dpdy);
		TxPoint2f st{ ds_ + dot(vs_, vec), dt_ + dot(vt_, vec) };
		return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
	}

	TxVector3f vs() const { return vs_; }
	TxVector3f vt() const { return vt_; }
	float ds() const { return ds_; }
	float dt() const { return dt_; }

private:
	TextureTransform textureFromRender_;
	TxVector3f vs_, vt_;
	float ds_, dt_;
};

// ---------------------------------------------------------------------------
// TextureMapping2D
//
// Variant-based union over the four 2D mapping strategies.
// Mirrors pbrt-v4 TextureMapping2D (replaces TaggedPointer with std::variant).
// ---------------------------------------------------------------------------
class TextureMapping2D {
public:
	using VariantType = std::variant<UVMapping, SphericalMapping,
									 CylindricalMapping, PlanarMapping>;

	/* implicit */ TextureMapping2D(UVMapping m)          : v_(std::move(m)) {}
	/* implicit */ TextureMapping2D(SphericalMapping m)   : v_(std::move(m)) {}
	/* implicit */ TextureMapping2D(CylindricalMapping m) : v_(std::move(m)) {}
	/* implicit */ TextureMapping2D(PlanarMapping m)      : v_(std::move(m)) {}

	CPU_GPU TexCoord2D Map(const TextureEvalContext& ctx) const {
		return std::visit([&](const auto& mapping) {
			return mapping.Map(ctx);
		}, v_);
	}

	const VariantType& AsVariant() const { return v_; }

private:
	VariantType v_;
};

// ---------------------------------------------------------------------------
// PointTransformMapping
//
// 3D texture mapping that transforms a render-space point into texture space.
// Mirrors pbrt-v4 PointTransformMapping.
// ---------------------------------------------------------------------------
class PointTransformMapping {
public:
	explicit PointTransformMapping(const TextureTransform& textureFromRender)
		: textureFromRender_(textureFromRender) {}

	CPU_GPU TexCoord3D Map(const TextureEvalContext& ctx) const {
		return TexCoord3D{
			textureFromRender_(ctx.p),
			textureFromRender_(ctx.dpdx),
			textureFromRender_(ctx.dpdy)
		};
	}

	const TextureTransform& GetTransform() const { return textureFromRender_; }

private:
	TextureTransform textureFromRender_;
};

// ---------------------------------------------------------------------------
// TextureMapping3D
//
// Variant-based 3D mapping (currently only PointTransformMapping).
// Mirrors pbrt-v4 TextureMapping3D.
// ---------------------------------------------------------------------------
class TextureMapping3D {
public:
	using VariantType = std::variant<PointTransformMapping>;

	/* implicit */ TextureMapping3D(PointTransformMapping m) : v_(std::move(m)) {}

	CPU_GPU TexCoord3D Map(const TextureEvalContext& ctx) const {
		return std::visit([&](const auto& mapping) {
			return mapping.Map(ctx);
		}, v_);
	}

	const VariantType& AsVariant() const { return v_; }

private:
	VariantType v_;
};
