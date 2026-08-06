#pragma once
// ---------------------------------------------------------------------------
// bilinear_patch.h -- Bilinear patch intersection and sampling
//
// Ports pbrt-v4 shapes.h / shapes.cpp BilinearPatch (Apache-2.0) as a
// self-contained, header-only C++ implementation with no pbrt dependencies.
//
// Public API:
//
//   struct BlpHit { float u, v, t; };
//
//   // Intersect a ray with a bilinear patch defined by four corners.
//   // p00/p10/p01/p11 are the four corners (u=0/1, v=0/1).
//   // Returns BlpHit if 0 < t < t_max, otherwise nullopt.
//   std::optional<BlpHit>
//   blp_intersect(const float* ro, const float* rd, float t_max,
//                 const float* p00, const float* p10,
//                 const float* p01, const float* p11);
//
//   // Point on patch at (u,v).  out_p[3], out_dpdu[3], out_dpdv[3].
//   void blp_point(const float* p00, const float* p10,
//                  const float* p01, const float* p11,
//                  float u, float v,
//                  float* out_p, float* out_dpdu, float* out_dpdv);
//
//   // Unit outward normal at (u,v). out_n[3].
//   void blp_normal(const float* p00, const float* p10,
//                   const float* p01, const float* p11,
//                   float u, float v, float* out_n);
//
//   // Approximate surface area (3x3 grid integration, pbrt-v4 formula).
//   float blp_area(const float* p00, const float* p10,
//                  const float* p01, const float* p11);
//
//   // Test whether four corners form an axis-aligned rectangle.
//   bool blp_is_rectangle(const float* p00, const float* p10,
//                         const float* p01, const float* p11);
//
//   // Uniform area sample.  u[2] in [0,1)^2 -> out_p[3], out_n[3], out_pdf.
//   void blp_sample(const float* p00, const float* p10,
//                   const float* p01, const float* p11,
//                   const float* u2, float* out_p, float* out_n, float* out_pdf);
//
//   // Solid-angle PDF for direction wi from reference point ref_p[3].
//   float blp_pdf_wi(const float* p00, const float* p10,
//                    const float* p01, const float* p11,
//                    const float* ref_p,
//                    const float* wi);
//
// Adaptations from pbrt-v4:
//   - No pbrt Allocator / TriangleMesh / Ray / Transform dependencies
//   - Plain float[3] for points / vectors / normals
//   - std::optional<BlpHit> replaces pstd::optional<BilinearIntersection>
//   - Quadratic solver inlined
//   - 3x3 grid area approximation (pbrt-v4 uses the same grid size = 3)
//
// Reference: pbrt-v4 src/pbrt/shapes.h / shapes.cpp (Apache-2.0)
//            Ramsey et al., "Ray Bilinear Patch Intersections", 2004
// ---------------------------------------------------------------------------

#include <array>
#include <cmath>
#include <optional>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace blp_detail {

inline float dot(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline void cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

inline void sub(const float* a, const float* b, float* out) {
    out[0]=a[0]-b[0]; out[1]=a[1]-b[1]; out[2]=a[2]-b[2];
}

inline void lerp3(const float* a, const float* b, float t, float* out) {
    out[0]=(1-t)*a[0]+t*b[0];
    out[1]=(1-t)*a[1]+t*b[1];
    out[2]=(1-t)*a[2]+t*b[2];
}

inline float length(const float* a) {
    return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

inline float length2(const float* a) {
    return a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
}

inline float abs_max(const float* a) {
    return std::max({std::abs(a[0]), std::abs(a[1]), std::abs(a[2])});
}

// 3x3 matrix determinant  (column-major: col0,col1,col2 each float[3])
inline float det3(const float* c0, const float* c1, const float* c2) {
    return c0[0]*(c1[1]*c2[2]-c1[2]*c2[1])
          -c1[0]*(c0[1]*c2[2]-c0[2]*c2[1])
          +c2[0]*(c0[1]*c1[2]-c0[2]*c1[1]);
}

// Solve quadratic a*t^2 + b*t + c = 0, return false if no real roots
inline bool quadratic(float a, float b, float c, float* t0, float* t1) {
    // Degenerate: linear equation
    if (a == 0.f) {
        if (b == 0.f) return false;
        *t0 = *t1 = -c / b;
        return true;
    }
    double disc = (double)b*(double)b - 4.0*(double)a*(double)c;
    if (disc < 0) return false;
    double sqrt_disc = std::sqrt(disc);
    double q = (b < 0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
    *t0 = (float)(q / a);
    *t1 = (q != 0) ? (float)(c / q) : 0.f;
    if (*t0 > *t1) { float tmp=*t0; *t0=*t1; *t1=tmp; }
    return true;
}

} // namespace blp_detail

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

struct BlpHit {
    float u, v, t;
};

// ---------------------------------------------------------------------------
// blp_intersect
//
// pbrt-v4: IntersectBilinearPatch() in shapes.h
//
// Ray: origin ro[3], direction rd[3].
// Patch corners: p00, p10, p01, p11 (u=0/1, v=0/1).
// Returns BlpHit{u,v,t} if hit in (eps, t_max), else nullopt.
// ---------------------------------------------------------------------------
inline std::optional<BlpHit>
blp_intersect(const float* ro, const float* rd, float t_max,
              const float* p00, const float* p10,
              const float* p01, const float* p11)
{
    using namespace blp_detail;

    // Find quadratic coefficients for distance from ray to u iso-lines
    // a = dot(cross(p10-p00, p01-p11), rd)
    // c = dot(cross(p00-ro, rd), p01-p00)
    // b = dot(cross(p10-ro, rd), p11-p10) - (a+c)
    float e0[3], e1[3], e2[3], e3[3];
    sub(p10, p00, e0);  // p10-p00
    sub(p01, p11, e1);  // p01-p11
    sub(p00, ro,  e2);  // p00-ro
    sub(p10, ro,  e3);  // p10-ro

    float cr0[3], cr1[3], cr2[3];
    cross(e0, e1, cr0);
    float a = dot(cr0, rd);

    float dp[3]; sub(p01, p00, dp);
    cross(e2, rd, cr1);
    float c = dot(cr1, dp);

    float dp2[3]; sub(p11, p10, dp2);
    cross(e3, rd, cr2);
    float b = dot(cr2, dp2) - (a + c);

    float u1, u2;
    if (!quadratic(a, b, c, &u1, &u2))
        return std::nullopt;

    // eps = gamma(30) * (max|ro| + max|rd| + max|p00|+max|p10|+max|p01|+max|p11|)
    float eps = 4e-6f * (abs_max(ro) + abs_max(rd) +
                         abs_max(p00) + abs_max(p10) +
                         abs_max(p01) + abs_max(p11));

    float t_hit = t_max;
    float u_hit = -1.f, v_hit = -1.f;

    auto try_u = [&](float u_cand) {
        if (u_cand < 0.f || u_cand > 1.f) return;
        // uo = lerp(u, p00, p10),  ud = lerp(u, p01, p11) - uo
        float uo[3], p11u[3], ud[3];
        lerp3(p00, p10, u_cand, uo);
        lerp3(p01, p11, u_cand, p11u);  // lerp of p01..p11 at u_cand
        sub(p11u, uo, ud);

        float delta[3]; sub(uo, ro, delta);
        float perp[3];  cross(rd, ud, perp);
        float p2 = length2(perp);

        float v_num = det3(delta, rd,  perp);
        float t_num = det3(delta, ud,  perp);

        float t_cand = t_num / p2;
        float v_cand = v_num / p2;

        if (t_num > p2 * eps && v_cand >= 0.f && v_cand <= 1.f &&
            t_cand < t_hit) {
            t_hit = t_cand;
            u_hit = u_cand;
            v_hit = v_cand;
        }
    };

    try_u(u1);
    if (u2 != u1) try_u(u2);

    if (t_hit >= t_max) return std::nullopt;
    return BlpHit{u_hit, v_hit, t_hit};
}

// ---------------------------------------------------------------------------
// blp_point
//
// Evaluate patch point and partial derivatives at (u,v).
// pbrt-v4: bilinear interpolation used throughout Sample / Intersect.
// ---------------------------------------------------------------------------
inline void blp_point(const float* p00, const float* p10,
                      const float* p01, const float* p11,
                      float u, float v,
                      float* out_p, float* out_dpdu, float* out_dpdv)
{
    using namespace blp_detail;
    // pu0 = lerp(v, p00, p01),  pu1 = lerp(v, p10, p11)
    float pu0[3], pu1[3];
    lerp3(p00, p01, v, pu0);
    lerp3(p10, p11, v, pu1);

    if (out_p)    lerp3(pu0, pu1, u, out_p);
    if (out_dpdu) sub(pu1, pu0, out_dpdu);    // dpdu = pu1 - pu0

    // dpdv = lerp(u, p01, p11) - lerp(u, p00, p10)
    if (out_dpdv) {
        float a[3], b2[3];
        lerp3(p01, p11, u, a);
        lerp3(p00, p10, u, b2);
        sub(a, b2, out_dpdv);
    }
}

// ---------------------------------------------------------------------------
// blp_normal
//
// Unit outward normal at (u,v): normalize(cross(dpdu, dpdv)).
// pbrt-v4: Normal3f(Normalize(Cross(dpdu, dpdv)))
// ---------------------------------------------------------------------------
inline void blp_normal(const float* p00, const float* p10,
                       const float* p01, const float* p11,
                       float u, float v, float* out_n)
{
    using namespace blp_detail;
    float dpdu[3], dpdv[3];
    blp_point(p00, p10, p01, p11, u, v, nullptr, dpdu, dpdv);
    cross(dpdu, dpdv, out_n);
    float len = length(out_n);
    if (len > 0.f) { out_n[0]/=len; out_n[1]/=len; out_n[2]/=len; }
}

// ---------------------------------------------------------------------------
// blp_is_rectangle
//
// pbrt-v4: IsRectangle() in shapes.h.
// Checks coplanarity and equal diagonal distances from centroid.
// ---------------------------------------------------------------------------
inline bool blp_is_rectangle(const float* p00, const float* p10,
                              const float* p01, const float* p11)
{
    using namespace blp_detail;
    // Degenerate edge check
    float e[3];
    sub(p10, p00, e); if (length2(e)==0) return false;
    sub(p11, p01, e); if (length2(e)==0) return false;
    sub(p01, p00, e); if (length2(e)==0) return false;
    sub(p11, p10, e); if (length2(e)==0) return false;

    // Check coplanarity: n = norm(cross(p10-p00, p01-p00))
    float e0[3], e1[3], n[3];
    sub(p10, p00, e0); sub(p01, p00, e1);
    cross(e0, e1, n);
    float nlen = length(n); if (nlen == 0) return false;
    n[0]/=nlen; n[1]/=nlen; n[2]/=nlen;

    float d[3]; sub(p11, p00, d);
    float dlen = length(d); if (dlen == 0) return false;
    d[0]/=dlen; d[1]/=dlen; d[2]/=dlen;
    if (std::abs(dot(d, n)) > 1e-5f) return false;

    // Equal diagonal distances from centroid
    float cx = (p00[0]+p01[0]+p10[0]+p11[0])*0.25f;
    float cy = (p00[1]+p01[1]+p10[1]+p11[1])*0.25f;
    float cz = (p00[2]+p01[2]+p10[2]+p11[2])*0.25f;
    float c[3] = {cx, cy, cz};
    float d2[4];
    auto dist2 = [&](const float* p) {
        float dx=p[0]-c[0], dy=p[1]-c[1], dz=p[2]-c[2];
        return dx*dx+dy*dy+dz*dz;
    };
    d2[0]=dist2(p00); d2[1]=dist2(p10);
    d2[2]=dist2(p01); d2[3]=dist2(p11);
    for (int i=1;i<4;++i)
        if (std::abs(d2[i]-d2[0]) / (d2[0]+1e-7f) > 1e-4f) return false;

    // Perpendicular diagonals check (rectangle)
    float diag0[3], diag1[3];
    sub(p11, p00, diag0); sub(p01, p10, diag1);
    float len0=length(diag0), len1=length(diag1);
    if (len0==0||len1==0) return false;
    float cosA = std::abs(dot(diag0,diag1)/(len0*len1));
    return cosA < 1e-4f;
}

// ---------------------------------------------------------------------------
// blp_area
//
// Approximate surface area using a 3x3 grid of quads (diagonal cross product).
// pbrt-v4: constexpr int na = 3 grid, exact same formula:
//   area += 0.5 * |cross(p[i+1][j+1]-p[i][j], p[i+1][j]-p[i][j+1])|
// For a rectangle: Distance(p00,p01)*Distance(p00,p10).
// ---------------------------------------------------------------------------
inline float blp_area(const float* p00, const float* p10,
                      const float* p01, const float* p11)
{
    if (blp_is_rectangle(p00, p10, p01, p11)) {
        using namespace blp_detail;
        float e0[3], e1[3];
        sub(p01, p00, e0); sub(p10, p00, e1);
        return length(e0) * length(e1);
    }
    constexpr int na = 3;
    // Build (na+1)x(na+1) grid of points
    float pts[na+1][na+1][3];
    for (int i = 0; i <= na; ++i) {
        float u = float(i) / float(na);
        for (int j = 0; j <= na; ++j) {
            float v = float(j) / float(na);
            blp_point(p00, p10, p01, p11, u, v, pts[i][j], nullptr, nullptr);
        }
    }
    float area = 0.f;
    using namespace blp_detail;
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < na; ++j) {
            // pbrt-v4 formula: 0.5 * |cross(p[i+1][j+1]-p[i][j], p[i+1][j]-p[i][j+1])|
            float d0[3], d1[3], n2[3];
            sub(pts[i+1][j+1], pts[i][j],   d0);
            sub(pts[i+1][j],   pts[i][j+1], d1);
            cross(d0, d1, n2);
            area += 0.5f * length(n2);
        }
    }
    return area;
}

// ---------------------------------------------------------------------------
// blp_sample
//
// Uniform area sample of the patch.
// pbrt-v4: BilinearPatch::Sample(Point2f u)
//
// For a general (non-rectangle) patch uses corner-weight bilinear sampling
// (pbrt-v4 SampleBilinear / BilinearPDF approximation with corner areas).
// For a rectangle uses uniform uv sampling.
//
// u2[2] - uniform sample in [0,1)^2
// out_p[3] - sampled point
// out_n[3] - unit normal at sampled point
// out_pdf   - PDF with respect to solid angle (= 1/area for area sampling)
// ---------------------------------------------------------------------------
inline void blp_sample(const float* p00, const float* p10,
                       const float* p01, const float* p11,
                       const float* u2,
                       float* out_p, float* out_n, float* out_pdf)
{
    using namespace blp_detail;

    // Corner differential areas (weights for bilinear sampling)
    // w[0]=|cross(p10-p00, p01-p00)|, etc.  pbrt-v4 identical.
    auto corner_area = [&](const float* a, const float* b, const float* c) {
        float ab[3], ac[3], cr[3];
        sub(b, a, ab); sub(c, a, ac);
        cross(ab, ac, cr);
        return length(cr);
    };
    float w[4] = {
        corner_area(p00, p10, p01),
        corner_area(p10, p00, p11),
        corner_area(p01, p00, p11),
        corner_area(p11, p10, p01)
    };

    float su = u2[0], sv = u2[1];
    float pdf = 1.f;

    bool is_rect = blp_is_rectangle(p00, p10, p01, p11);
    if (!is_rect) {
        // Bilinear sample with corner weights (pbrt-v4 SampleBilinear)
        // Marginal in u: w0_u = (w[0]+w[2])/2, w1_u = (w[1]+w[3])/2
        float w0u = 0.5f*(w[0]+w[2]);
        float w1u = 0.5f*(w[1]+w[3]);
        float wsum = w0u + w1u;
        if (wsum < 1e-10f) { su = u2[0]; sv = u2[1]; }
        else {
            // Sample u: CDF is linear segment
            if (w0u == w1u) {
                su = u2[0];
            } else {
                // Solve (w0u + (w1u-w0u)*u)*u/wsum = xi -> quadratic
                float disc = w0u*w0u + (w1u*w1u - w0u*w0u)*u2[0];
                if (disc < 0) disc = 0;
                su = (std::sqrt(disc) - w0u) / (w1u - w0u);
                su = std::max(0.f, std::min(1.f, su));
            }
            // PDF for u
            float pdf_u = (w0u + (w1u-w0u)*su) / (0.5f*wsum);

            // Given su, conditional weights for v
            float w0v = (1-su)*w[0] + su*w[1];
            float w1v = (1-su)*w[2] + su*w[3];
            float wvsum = w0v + w1v;
            if (wvsum < 1e-10f) { sv = u2[1]; }
            else {
                if (w0v == w1v) {
                    sv = u2[1];
                } else {
                    float disc2 = w0v*w0v + (w1v*w1v - w0v*w0v)*u2[1];
                    if (disc2 < 0) disc2 = 0;
                    sv = (std::sqrt(disc2) - w0v) / (w1v - w0v);
                    sv = std::max(0.f, std::min(1.f, sv));
                }
                float pdf_v = (w0v + (w1v-w0v)*sv) / (0.5f*wvsum);
                pdf = pdf_u * pdf_v;
            }
        }
    }

    float dpdu[3], dpdv[3];
    blp_point(p00, p10, p01, p11, su, sv, out_p, dpdu, dpdv);
    blp_normal(p00, p10, p01, p11, su, sv, out_n);

    float cr[3]; cross(dpdu, dpdv, cr);
    float dA = length(cr);
    if (dA > 0.f)
        *out_pdf = pdf / dA;
    else
        *out_pdf = 0.f;
}

// ---------------------------------------------------------------------------
// blp_pdf_wi
//
// Area-sampling solid-angle PDF from reference point ref_p toward wi.
// pbrt-v4: BilinearPatch::PDF(ShapeSampleContext, wi) — area path.
//
// = pdf_area(hit_point) * dist^2 / |cos_theta|
// ---------------------------------------------------------------------------
inline float blp_pdf_wi(const float* p00, const float* p10,
                        const float* p01, const float* p11,
                        const float* ref_p,
                        const float* wi)
{
    using namespace blp_detail;
    // Intersect the ray from ref_p along wi with the patch
    auto hit = blp_intersect(ref_p, wi, 1e20f, p00, p10, p01, p11);
    if (!hit) return 0.f;

    // Area PDF at hit point = 1 / |dpdu x dpdv|  (uniform area sampling)
    float hp[3], dpdu[3], dpdv[3];
    blp_point(p00, p10, p01, p11, hit->u, hit->v, hp, dpdu, dpdv);
    float cr[3]; cross(dpdu, dpdv, cr);
    float dA = length(cr);
    if (dA == 0.f) return 0.f;

    float area_pdf = 1.f / dA;

    // normal at hit
    float n[3] = {cr[0]/dA, cr[1]/dA, cr[2]/dA};
    float cos_theta = std::abs(dot(n, wi));
    if (cos_theta == 0.f) return 0.f;

    // dist^2 = t^2 * |wi|^2
    float wi_len2 = length2(wi);
    float dist2 = hit->t * hit->t * wi_len2;

    return area_pdf * dist2 / cos_theta;
}

// ===========================================================================
// BilinearPatchShape<T>
// ===========================================================================
//
// Template struct wrapping the bilinear patch with the standard shape API
// used by SphereShape, DiskShape, CylinderShape, TriangleShape.
//
// Stores four corners p00, p10, p01, p11 as flat scalars (world space).
// Convention (matches pbrt-v4 BilinearPatch):
//   p(u,v) = lerp(u, lerp(v, p00, p01), lerp(v, p10, p11))
//   dpdu   = lerp(v, p10, p11) - lerp(v, p00, p01)
//   dpdv   = lerp(u, p01, p11) - lerp(u, p00, p10)
//
// Exposes:
//   intersect(rox,roy,roz, rdx,rdy,rdz, t_min, t_max) -> optional<ShapeHit<T>>
//   area()                     -> T      (3x3 grid quadrature, pbrt-v4)
//   pdf_area()                 -> T      (1/area)
//   sample(u0,u1)              -> ShapeSample<T>  (area-uniform)
//   sample_from(ctx,u0,u1)     -> ShapeSample<T>  (solid-angle from point)
//   pdf_from(ctx,wi_dx,dy,dz)  -> T               (solid-angle PDF)
//
// Reference: pbrt-v4 src/pbrt/shapes.h  BilinearPatch class
//            Ramsey et al., "Ray Bilinear Patch Intersections", 2004
// ===========================================================================

// Always include shapes.h so ShapeHit<T>, ShapeSample<T>, and SamplingContext<T>
// have a single canonical definition (avoids ODR violations when bilinear_patch.h
// and shapes.h are both compiled into the same translation unit or linked together).
#include "shapes.h"

#ifndef BLP_CPU_GPU
#  if defined(__CUDACC__)
#    define BLP_CPU_GPU __host__ __device__ __forceinline__
#  else
#    define BLP_CPU_GPU inline
#  endif
#endif

template<typename T>
struct BilinearPatchShape {
    // Four corners in world space (u=0/1 left/right, v=0/1 bottom/top)
    T p00x, p00y, p00z;
    T p10x, p10y, p10z;
    T p01x, p01y, p01z;
    T p11x, p11y, p11z;

    CPU_GPU static BilinearPatchShape make(
        T p00x, T p00y, T p00z,
        T p10x, T p10y, T p10z,
        T p01x, T p01y, T p01z,
        T p11x, T p11y, T p11z)
    {
        return BilinearPatchShape{
            p00x,p00y,p00z, p10x,p10y,p10z,
            p01x,p01y,p01z, p11x,p11y,p11z};
    }

    // -----------------------------------------------------------------------
    // Bilinear interpolation helpers
    // -----------------------------------------------------------------------
    CPU_GPU void lerp_point(T u, T v,
                             T& px, T& py, T& pz) const {
        // p(u,v) = lerp(u, lerp(v,p00,p01), lerp(v,p10,p11))
        T ax = p00x + v*(p01x-p00x), ay = p00y + v*(p01y-p00y), az = p00z + v*(p01z-p00z);
        T bx = p10x + v*(p11x-p10x), by = p10y + v*(p11y-p10y), bz = p10z + v*(p11z-p10z);
        px = ax + u*(bx-ax); py = ay + u*(by-ay); pz = az + u*(bz-az);
    }

    CPU_GPU void dpdu_dpdv(T u, T v,
                            T& dux, T& duy, T& duz,
                            T& dvx, T& dvy, T& dvz) const {
        // dpdu = lerp(v,p10,p11) - lerp(v,p00,p01)
        T ax = p00x + v*(p01x-p00x), ay = p00y + v*(p01y-p00y), az = p00z + v*(p01z-p00z);
        T bx = p10x + v*(p11x-p10x), by = p10y + v*(p11y-p10y), bz = p10z + v*(p11z-p10z);
        dux = bx-ax; duy = by-ay; duz = bz-az;
        // dpdv = lerp(u,p01,p11) - lerp(u,p00,p10)
        T cx = p00x + u*(p10x-p00x), cy = p00y + u*(p10y-p00y), cz = p00z + u*(p10z-p00z);
        T dx = p01x + u*(p11x-p01x), dy = p01y + u*(p11y-p01y), dz = p01z + u*(p11z-p01z);
        dvx = dx-cx; dvy = dy-cy; dvz = dz-cz;
    }

    CPU_GPU void outward_normal(T u, T v,
                                 T& nx, T& ny, T& nz) const {
        T dux,duy,duz,dvx,dvy,dvz;
        dpdu_dpdv(u,v,dux,duy,duz,dvx,dvy,dvz);
        // n = dpdu x dpdv, normalised
        T cx = duy*dvz - duz*dvy;
        T cy = duz*dvx - dux*dvz;
        T cz = dux*dvy - duy*dvx;
        T len = std::sqrt(cx*cx+cy*cy+cz*cz);
        if (len > T(0)) { nx=cx/len; ny=cy/len; nz=cz/len; }
        else            { nx=T(0);   ny=T(0);   nz=T(1);   }
    }

    // -----------------------------------------------------------------------
    // Area -- 3x3 triangulated grid (matches pbrt-v4 BilinearPatch area)
    // Reference: pbrt-v4 shapes.cpp BilinearPatch constructor
    //   na=3 grid; area += 0.5 * |cross(p[i+1][j+1]-p[i][j], p[i+1][j]-p[i][j+1])|
    // -----------------------------------------------------------------------
    CPU_GPU T area() const {
        // Build (na+1)x(na+1) = 4x4 grid of surface points
        constexpr int na = 3;
        T px[na+1][na+1], py[na+1][na+1], pz_[na+1][na+1];
        for (int i = 0; i <= na; ++i) {
            T u = T(i) / T(na);
            for (int j = 0; j <= na; ++j) {
                T v = T(j) / T(na);
                lerp_point(u, v, px[i][j], py[i][j], pz_[i][j]);
            }
        }
        T A = T(0);
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                // diagonal vectors of the quad
                T d1x = px[i+1][j+1]-px[i][j],   d1y = py[i+1][j+1]-py[i][j],   d1z = pz_[i+1][j+1]-pz_[i][j];
                T d2x = px[i+1][j]  -px[i][j+1], d2y = py[i+1][j]  -py[i][j+1], d2z = pz_[i+1][j]  -pz_[i][j+1];
                T cx = d1y*d2z - d1z*d2y;
                T cy = d1z*d2x - d1x*d2z;
                T cz = d1x*d2y - d1y*d2x;
                A += T(0.5) * std::sqrt(cx*cx + cy*cy + cz*cz);
            }
        }
        return A;
    }

    CPU_GPU T pdf_area() const { return T(1) / area(); }

    // -----------------------------------------------------------------------
    // Intersection
    // Delegates to the proven blp_intersect low-level function and wraps
    // the result in the standard ShapeHit<T> API.
    // Reference: pbrt-v4 BilinearPatch::BasicIntersect / Ramsey et al. 2004
    // -----------------------------------------------------------------------
    CPU_GPU std::optional<ShapeHit<T>>
    intersect(T rox, T roy, T roz,
              T rdx, T rdy, T rdz,
              T t_min, T t_max) const
    {
        float ro[3] = {(float)rox,(float)roy,(float)roz};
        float rd[3] = {(float)rdx,(float)rdy,(float)rdz};
        float sq00[3]={(float)p00x,(float)p00y,(float)p00z};
        float sq10[3]={(float)p10x,(float)p10y,(float)p10z};
        float sq01[3]={(float)p01x,(float)p01y,(float)p01z};
        float sq11[3]={(float)p11x,(float)p11y,(float)p11z};
        auto h = blp_intersect(ro, rd, (float)t_max, sq00, sq10, sq01, sq11);
        if (!h || (T)h->t < t_min || (T)h->t > t_max) return {};
        float u = h->u, v = h->v;
        // Compute normal at (u,v)
        T nnx,nny,nnz;
        outward_normal((T)u,(T)v, nnx,nny,nnz);
        // Orient normal toward ray origin
        if (nnx*rdx+nny*rdy+nnz*rdz > T(0)) { nnx=-nnx; nny=-nny; nnz=-nnz; }
        return ShapeHit<T>{(T)h->t, nnx,nny,nnz, (T)u,(T)v};
    }

    // -----------------------------------------------------------------------
    // Area-uniform surface sample
    // Reference: pbrt-v4 BilinearPatch::Sample(Point2f u)
    // -----------------------------------------------------------------------
    CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
        // Simple uniform (u,v) -- correct for rectangles, approximate otherwise
        T px,py,pz;
        lerp_point(u0, u1, px,py,pz);
        T nx,ny,nz;
        outward_normal(u0, u1, nx,ny,nz);
        return ShapeSample<T>{px,py,pz, nx,ny,nz, u0,u1, pdf_area()};
    }

    // -----------------------------------------------------------------------
    // Solid-angle sample from a shading point (area sample + Jacobian)
    // Reference: pbrt-v4 BilinearPatch::Sample(ShapeSampleContext, Point2f)
    // -----------------------------------------------------------------------
    CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
                                        T u0, T u1) const {
        ShapeSample<T> ss = sample(u0, u1);
        T wix = ss.px-ctx.px, wiy = ss.py-ctx.py, wiz = ss.pz-ctx.pz;
        T dist2 = wix*wix+wiy*wiy+wiz*wiz;
        if (dist2 == T(0)) { ss.pdf = T(0); return ss; }
        T inv_d = T(1)/std::sqrt(dist2);
        T cos_t = std::abs(ss.nx*(-wix*inv_d)+ss.ny*(-wiy*inv_d)+ss.nz*(-wiz*inv_d));
        if (cos_t == T(0)) { ss.pdf = T(0); return ss; }
        ss.pdf = pdf_area() * dist2 / cos_t;
        return ss;
    }

    // -----------------------------------------------------------------------
    // Solid-angle PDF from a shading point (intersect + Jacobian)
    // Reference: pbrt-v4 BilinearPatch::PDF(ShapeSampleContext, Vector3f)
    // -----------------------------------------------------------------------
    CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
                        T wi_dx, T wi_dy, T wi_dz) const {
        T wi_len = std::sqrt(wi_dx*wi_dx+wi_dy*wi_dy+wi_dz*wi_dz);
        if (wi_len == T(0)) return T(0);
        T wix=wi_dx/wi_len, wiy=wi_dy/wi_len, wiz=wi_dz/wi_len;
        auto hit = intersect(ctx.px,ctx.py,ctx.pz, wix,wiy,wiz,
                             T(1e-4), std::numeric_limits<T>::max());
        if (!hit) return T(0);
        T dist2 = hit->t * hit->t;
        T cos_t = std::abs(hit->nx*(-wix)+hit->ny*(-wiy)+hit->nz*(-wiz));
        if (cos_t == T(0)) return T(0);
        T pdf = pdf_area() * dist2 / cos_t;
        return std::isfinite(pdf) ? pdf : T(0);
    }
};
