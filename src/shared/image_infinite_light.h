#pragma once
// ---------------------------------------------------------------------------
// image_infinite_light.h -- ImageInfiniteLightData<T>
//
// Self-contained CPU/GPU port of pbrt-v4 ImageInfiniteLight
// (src/pbrt/lights.h / lights.cpp).
//
// Aligned with pbrt-v4 (lights.cpp 1007-1104):
//   1. compensated_distribution: lum_i - avg, clamped >= 0
//      (pbrt-v4 ImageInfiniteLight ctor lines 1033-1039)
//   2. renderFromLight rotation matrix for render->light direction transform
//   3. WrapEqualAreaSquare boundary wrap (pbrt-v4 WrapMode::OctahedralSphere)
//   4. allow_incomplete flag on sample_wi/pdf_wi
//      (pbrt-v4 allowIncompletePDF selects compensatedDistribution)
//
// pbrt-v4 reference:
//   lights.h  ImageInfiniteLight / SampleLi / PDF_Li
//   lights.cpp ImageInfiniteLight ctor / Phi
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "piecewise_dist.h"
#include "sampling.h"

namespace iil_detail {

CPU_GPU double luminance(double r, double g, double b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

CPU_GPU void apply_matrix(const double* m,
                                  double  x,  double  y,  double  z,
                                  double& ox, double& oy, double& oz) {
    ox = m[0]*x + m[3]*y + m[6]*z;
    oy = m[1]*x + m[4]*y + m[7]*z;
    oz = m[2]*x + m[5]*y + m[8]*z;
}

// Bilinear lookup with WrapEqualAreaSquare boundary wrap.
// Matches pbrt-v4 WrapMode::OctahedralSphere.
inline void bilinear_lookup(const std::vector<double>& img,
                             int width, int height,
                             double u, double v,
                             double& r, double& g, double& b) {
    double pu = u * width  - 0.5;
    double pv = v * height - 0.5;
    int x0 = (int)std::floor(pu), y0 = (int)std::floor(pv);
    double tx = pu - x0, ty = pv - y0;
    auto tap = [&](int xi, int yi, int c) -> double {
        double tu = (xi + 0.5) / width, tv = (yi + 0.5) / height;
        WrapEqualAreaSquare(tu, tv);
        int wx = std::max(0, std::min((int)std::floor(tu * width  - 0.5 + 0.5), width  - 1));
        int wy = std::max(0, std::min((int)std::floor(tv * height - 0.5 + 0.5), height - 1));
        return img[(wy * width + wx) * 3 + c];
    };
    r=(1-tx)*(1-ty)*tap(x0,y0,0)+tx*(1-ty)*tap(x0+1,y0,0)+(1-tx)*ty*tap(x0,y0+1,0)+tx*ty*tap(x0+1,y0+1,0);
    g=(1-tx)*(1-ty)*tap(x0,y0,1)+tx*(1-ty)*tap(x0+1,y0,1)+(1-tx)*ty*tap(x0,y0+1,1)+tx*ty*tap(x0+1,y0+1,1);
    b=(1-tx)*(1-ty)*tap(x0,y0,2)+tx*(1-ty)*tap(x0+1,y0,2)+(1-tx)*ty*tap(x0,y0+1,2)+tx*ty*tap(x0+1,y0+1,2);
}

} // namespace iil_detail

template <typename T>
struct ImageInfiniteLightData {
    // rgb: T* row-major R,G,B, width*height*3 elements
    // sc:  radiance scale factor
    // rot: optional 9-element col-major 3x3 rotation, render->light space
    ImageInfiniteLightData(const T* rgb, int w, int h, T sc = T(1),
                           const double* rot = nullptr)
        : width(w), height(h), scale(sc), has_rotation(rot != nullptr)
    {
        if (rot) for (int i = 0; i < 9; ++i) rotation[i] = rot[i];
        else {
            rotation[0]=1; rotation[1]=0; rotation[2]=0;
            rotation[3]=0; rotation[4]=1; rotation[5]=0;
            rotation[6]=0; rotation[7]=0; rotation[8]=1;
        }
        int n = w * h;
        image.resize(n * 3);
        for (int i = 0; i < n * 3; ++i) image[i] = (double)rgb[i];
        std::vector<double> lum(n);
        for (int i = 0; i < n; ++i) {
            double r=image[i*3], g=image[i*3+1], b=image[i*3+2];
            lum[i] = std::max(0.0, iil_detail::luminance(r,g,b));
        }
        // Primary distribution (pbrt-v4: distribution)
        distribution = PiecewiseConstant2D(lum, w, h);
        // Compensated distribution: lum_i - average, clamped >= 0.
        // Exactly mirrors pbrt-v4 ImageInfiniteLight ctor (lights.cpp 1033-1039).
        std::vector<double> lum_c(lum);
        double avg = std::accumulate(lum_c.begin(), lum_c.end(), 0.0) / n;
        for (double& v : lum_c) v = std::max(v - avg, 0.0);
        if (std::all_of(lum_c.begin(),lum_c.end(),[](double v){return v==0.0;}))
            std::fill(lum_c.begin(), lum_c.end(), 1.0);
        compensated_distribution = PiecewiseConstant2D(lum_c, w, h);
        total_lum = 0.0;
        for (double l : lum) total_lum += l;
        total_lum *= (4.0 * 3.14159265358979323846) / double(n);
    }

    // eval_Le: luminance radiance for render-space direction.
    // Mirrors pbrt-v4 ImageInfiniteLight::Le
    CPU_GPU T eval_Le(T wx, T wy, T wz) const {
        double lx=(double)wx, ly=(double)wy, lz=(double)wz;
        if (has_rotation) iil_detail::apply_matrix(rotation,lx,ly,lz,lx,ly,lz);
        double u,v; EqualAreaSphereToSquare(lx,ly,lz,u,v);
        double r,g,b; iil_detail::bilinear_lookup(image,width,height,u,v,r,g,b);
        return (T)(scale * iil_detail::luminance(r,g,b));
    }

    CPU_GPU void eval_Le_rgb(T wx, T wy, T wz, T& or2, T& og, T& ob) const {
        double lx=(double)wx, ly=(double)wy, lz=(double)wz;
        if (has_rotation) iil_detail::apply_matrix(rotation,lx,ly,lz,lx,ly,lz);
        double u,v; EqualAreaSphereToSquare(lx,ly,lz,u,v);
        double r,g,b; iil_detail::bilinear_lookup(image,width,height,u,v,r,g,b);
        or2=(T)(scale*r); og=(T)(scale*g); ob=(T)(scale*b);
    }

    // sample_wi: importance-sample a render-space direction.
    // allow_incomplete=false: primary distribution  (pbrt-v4 !allowIncompletePDF)
    // allow_incomplete=true:  compensated distrib.  (pbrt-v4  allowIncompletePDF)
    // pdf = map_pdf / (4*pi) -- same as pbrt-v4 SampleLi
    void sample_wi(double ru, double rv,
                   double& wx, double& wy, double& wz,
                   double& pdf,
                   bool allow_incomplete = false) const {
        double map_pdf = 0.0;
        auto uv = allow_incomplete
            ? compensated_distribution.sample(ru,rv,&map_pdf)
            : distribution.sample(ru,rv,&map_pdf);
        if (map_pdf == 0.0) { wx=wy=wz=0.0; pdf=0.0; return; }
        double lx,ly,lz;
        EqualAreaSquareToSphere(uv.first, uv.second, lx, ly, lz);
        // transpose of rotation (render->light) gives light->render
        if (has_rotation) {
            wx=rotation[0]*lx+rotation[1]*ly+rotation[2]*lz;
            wy=rotation[3]*lx+rotation[4]*ly+rotation[5]*lz;
            wz=rotation[6]*lx+rotation[7]*ly+rotation[8]*lz;
        } else { wx=lx; wy=ly; wz=lz; }
        pdf = map_pdf / (4.0 * 3.14159265358979323846);
    }

    // pdf_wi: solid-angle PDF. Mirrors pbrt-v4 ImageInfiniteLight::PDF_Li
    CPU_GPU T pdf_wi(T wx, T wy, T wz, bool allow_incomplete = false) const {
        double lx=(double)wx, ly=(double)wy, lz=(double)wz;
        if (has_rotation) iil_detail::apply_matrix(rotation,lx,ly,lz,lx,ly,lz);
        double u,v; EqualAreaSphereToSquare(lx,ly,lz,u,v);
        double mp = allow_incomplete
            ? compensated_distribution.pdf(u,v)
            : distribution.pdf(u,v);
        return (T)(mp / (4.0 * 3.14159265358979323846));
    }

    // power(): scale * sum(lum) * 4pi/N.
    // pbrt-v4 Phi = 4*pi^2 * r^2 * scale * sum(L) / N.
    // pi*r^2 omitted (scene-dependent). Multiply by (pi*r^2) for full power.
    CPU_GPU T power() const { return (T)(scale * total_lum); }

    int  width=0, height=0;
    T    scale=T(1);
    bool has_rotation=false;

private:
    std::vector<double>  image;
    PiecewiseConstant2D  distribution;
    PiecewiseConstant2D  compensated_distribution;
    double               total_lum=0.0;
    double               rotation[9]={1,0,0,0,1,0,0,0,1};
};
