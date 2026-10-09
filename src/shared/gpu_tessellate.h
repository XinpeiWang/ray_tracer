#pragma once
// gpu_tessellate.h -- turns the shapes a GPU backend has no intersection routine for into triangles, once, before it converts the scene.
//
// Both GPU backends read the same FlatScene (src/shared/pbrt_flatten.h) and neither can ray-trace every shape in it: Metal has only triangles, spheres, disks and
// cylinders, OptiX adds bilinear patches (and dices curves into patches itself) but has no cone or paraboloid. This is the step that rewrites those shapes as ordinary
// triangles appended to the scene, so area lights, materials, instancing and the scene bounds all see them like any other mesh. Which shapes are rewritten is the
// backend's TessellationCaps: Metal asks for all four kinds, OptiX only for cones and paraboloids (its native patches and curve dicing stay). Moved here from
// gpu/metal/metal_poc_pbrt_loader.mm (tessellateUnsupportedShapes) with the numbers unchanged, so a seeded Metal render is byte-identical.
//
// The tessellation is a close approximation, not exact geometry: smooth analytic vertex normals for the quadrics and patches, flat-shaded 8-sided tubes for curves
// (the same dicing density as src/shared/curve_tessellate.h gives OptiX). A shape with Material "interface" only bounds a medium and is skipped (transparent).
//
// Standard library only, so the CPU unit tests can reach it.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "curve_tessellate.h"
#include "pbrt_flatten.h"

namespace gpu_tessellate {

// Which kinds of shape to rewrite as triangles.
struct TessellationCaps {
	bool bilinearPatches = false;
	bool curves = false;
	bool cones = false;
	bool paraboloids = false;
	static TessellationCaps all() { return {true, true, true, true}; }                  // Metal: only triangles and the three analytic shapes it has
	static TessellationCaps conesAndParaboloids() { return {false, false, true, true}; }   // OptiX: its patches and curve dicing are native
};

// Triangle index (in scene.triangles after the call) -> the fibre direction of the curve tube it came from, which a hair shader wants.
using FiberTangents = std::unordered_map<int, std::array<float, 3>>;

namespace tessellate_detail {

constexpr double kPi = 3.14159265358979323846;

inline void tessellateGrid(std::vector<pbrt_flatten::Triangle>& out, int nu, int nv, int material, int areaLight,
                    const std::function<void(double, double, double*, double*)>& eval) {
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const double u0 = (double)i / nu, u1 = (double)(i + 1) / nu;
            const double v0 = (double)j / nv, v1 = (double)(j + 1) / nv;
            double p[4][3], n[4][3];
            const double us[4] = {u0, u1, u0, u1}, vs[4] = {v0, v0, v1, v1};   // p00 p10 p01 p11
            for (int k = 0; k < 4; ++k) eval(us[k], vs[k], p[k], n[k]);
            const int tris[2][3] = {{0, 1, 3}, {0, 3, 2}};
            for (const auto& t : tris) {
                pbrt_flatten::Triangle tri;
                for (int c = 0; c < 3; ++c) {
                    for (int a = 0; a < 3; ++a) { tri.v[c * 3 + a] = p[t[c]][a]; tri.n[c * 3 + a] = n[t[c]][a]; }
                    tri.uv[c * 2 + 0] = us[t[c]]; tri.uv[c * 2 + 1] = vs[t[c]];
                }
                tri.hasNormals = true;
                tri.hasUVs = true;
                tri.material = material;
                tri.areaLight = areaLight;
                out.push_back(tri);
            }
        }
    }
}

inline void normalize3(double* v) {
    const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-300) { v[0] /= l; v[1] /= l; v[2] /= l; }
}

// Returns how many triangles were added (0 when the scene has none of these shapes).
}  // namespace tessellate_detail

// Rewrites the shapes `caps` names as triangles appended to scene.triangles (and removes them from the scene). Returns how many triangles were added (0 when the
// scene has none of those shapes). `fiberTangent`, if given, receives the fibre direction of each curve triangle.
inline size_t tessellateForBackend(pbrt_flatten::FlatScene& scene, const TessellationCaps& caps, FiberTangents* fiberTangent) {
    std::vector<pbrt_flatten::Triangle> out;
    std::vector<std::pair<size_t, std::array<float, 3>>> outTangents;   // (index into `out`, fibre direction)
    // A shape with `Material "interface"` only bounds a participating medium: skip it (transparent)
    // rather than turning it into an opaque gray mesh - same rule the sphere/disk/cylinder loaders use.
    auto isInterface = [&scene](int m) {
        return m >= 0 && m < (int)scene.materials.size() &&
               scene.materials[m].kind == pbrt_flatten::MaterialKind::Interface;
    };

    // Bilinear patch: p(u,v) = (1-u)(1-v)p00 + u(1-v)p10 + (1-u)v p01 + uv p11; normal = dpdu x dpdv.
    for (const pbrt_flatten::BilinearPatch& bp : scene.bilinearPatches) {
        if (!caps.bilinearPatches) break;
        if (isInterface(bp.material)) continue;
        const double (*P)[3] = bp.p;
        tessellate_detail::tessellateGrid(out, 16, 16, bp.material, bp.areaLight, [&](double u, double v, double* p, double* n) {
            double dpdu[3], dpdv[3];
            for (int a = 0; a < 3; ++a) {
                p[a] = (1 - u) * (1 - v) * P[0][a] + u * (1 - v) * P[1][a] + (1 - u) * v * P[2][a] + u * v * P[3][a];
                dpdu[a] = (1 - v) * (P[1][a] - P[0][a]) + v * (P[3][a] - P[2][a]);
                dpdv[a] = (1 - u) * (P[2][a] - P[0][a]) + u * (P[3][a] - P[1][a]);
            }
            n[0] = dpdu[1] * dpdv[2] - dpdu[2] * dpdv[1];
            n[1] = dpdu[2] * dpdv[0] - dpdu[0] * dpdv[2];
            n[2] = dpdu[0] * dpdv[1] - dpdu[1] * dpdv[0];
            tessellate_detail::normalize3(n);
        });
    }

    // Cone (open, no base cap): object space z in [0,height], radius R*(1 - z/height).
    for (const pbrt_flatten::Cone& c : scene.cones) {
        if (!caps.cones) break;
        if (isInterface(c.material)) continue;
        pbrt_scene::Matrix4 xf;
        for (int i = 0; i < 16; ++i) xf.m[i] = c.xform[i];
        const double phiMax = c.phiMaxDeg * tessellate_detail::kPi / 180.0;
        tessellate_detail::tessellateGrid(out, 48, 4, c.material, c.areaLight, [&](double u, double v, double* p, double* n) {
            const double ph = u * phiMax, rad = c.radius * (1.0 - v);
            const double po[3] = {rad * std::cos(ph), rad * std::sin(ph), v * c.height};
            pbrt_flatten::flatten_detail::transformPoint(xf, po[0], po[1], po[2], p);
            pbrt_flatten::flatten_detail::transformNormal(xf, c.height * std::cos(ph), c.height * std::sin(ph), c.radius, n);
            tessellate_detail::normalize3(n);
        });
    }

    // Paraboloid: z in [zMin,zMax], radius R*sqrt(z/zMax); outward normal is the gradient of
    // x^2 + y^2 - (R^2/zMax) z.
    for (const pbrt_flatten::Paraboloid& pa : scene.paraboloids) {
        if (!caps.paraboloids) break;
        if (isInterface(pa.material)) continue;
        pbrt_scene::Matrix4 xf;
        for (int i = 0; i < 16; ++i) xf.m[i] = pa.xform[i];
        const double phiMax = pa.phiMaxDeg * tessellate_detail::kPi / 180.0;
        tessellate_detail::tessellateGrid(out, 48, 24, pa.material, pa.areaLight, [&](double u, double v, double* p, double* n) {
            const double ph = u * phiMax;
            const double z = pa.zMin + v * (pa.zMax - pa.zMin);
            const double rad = pa.radius * std::sqrt(std::max(z, 0.0) / pa.zMax);
            const double x = rad * std::cos(ph), y = rad * std::sin(ph);
            pbrt_flatten::flatten_detail::transformPoint(xf, x, y, z, p);
            pbrt_flatten::flatten_detail::transformNormal(xf, 2.0 * x, 2.0 * y, -pa.radius * pa.radius / pa.zMax, n);
            tessellate_detail::normalize3(n);
        });
    }

    // Curve: every Bezier segment becomes a tapered 8-sided tube (the same dicing density OptiX
    // uses); width interpolates across the WHOLE curve, as in pbrt-v4. Flat-shaded.
    for (const pbrt_flatten::Curve& cv : scene.curves) {
        if (!caps.curves) break;
        if (isInterface(cv.material)) continue;
        std::vector<curve_tessellate::Quad> quads;
        for (int seg = 0; seg < cv.nSegments; ++seg) {
            float cp[4][3];
            for (int i = 0; i < 4; ++i)
                for (int a = 0; a < 3; ++a) cp[i][a] = (float)cv.cp[((size_t)seg * 4 + i) * 3 + a];
            const double t0 = (double)seg / cv.nSegments, t1 = (double)(seg + 1) / cv.nSegments;
            quads.clear();
            curve_tessellate::tessellate(cp, 0.0f, 1.0f, (float)(cv.width0 + (cv.width1 - cv.width0) * t0),
                                         (float)(cv.width0 + (cv.width1 - cv.width0) * t1), 10, 8, quads);
            for (const curve_tessellate::Quad& q : quads) {
                const float* c4[4] = {q.p00, q.p10, q.p01, q.p11};
                const int tris[2][3] = {{0, 1, 3}, {0, 3, 2}};
                for (const auto& t : tris) {
                    pbrt_flatten::Triangle tri;
                    for (int c = 0; c < 3; ++c)
                        for (int a = 0; a < 3; ++a) tri.v[c * 3 + a] = c4[t[c]][a];
                    tri.material = cv.material;
                    tri.areaLight = cv.areaLight;
                    // Fibre direction = along the tube (ring i -> ring i+1), for the hair shader.
                    {
                        float tx = q.p10[0] - q.p00[0], ty = q.p10[1] - q.p00[1], tz = q.p10[2] - q.p00[2];
                        const float tl = std::sqrt(tx * tx + ty * ty + tz * tz);
                        if (tl > 1e-20f) outTangents.push_back({out.size(), std::array<float, 3>{tx / tl, ty / tl, tz / tl}});
                    }
                    out.push_back(tri);
                }
            }
        }
    }

    const size_t baseIndex = scene.triangles.size();
    if (fiberTangent)
        for (const auto& tp : outTangents) (*fiberTangent)[(int)(baseIndex + tp.first)] = tp.second;
    scene.triangles.insert(scene.triangles.end(), out.begin(), out.end());
    if (caps.bilinearPatches) scene.bilinearPatches.clear();
    if (caps.cones) scene.cones.clear();
    if (caps.paraboloids) scene.paraboloids.clear();
    if (caps.curves) scene.curves.clear();
    return out.size();
}


}  // namespace gpu_tessellate
