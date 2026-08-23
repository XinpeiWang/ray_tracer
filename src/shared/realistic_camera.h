#pragma once
// realistic_camera.h -- pbrt-v4 RealisticCamera port (header-only, CPU,
// templated on T). Split out of cameras.h, which #includes this file at its
// own end -- see that file's header comment for the simpler Ortho/Persp/
// Spherical camera models this one is a physically-simulated sibling to
// (and for CamVec3<T>/Mat4<T>/CameraRayResult<T>/CameraSample<T>, which this
// file builds on and gets via the #include below).
//
// Models a real camera lens system as a sequence of spherical refractive
// elements and aperture stops. Rays are traced from the film plane through
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

#include "cameras.h"

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
    T film_half_x() const { return film_half_x_; }
    T film_half_y() const { return film_half_y_; }

    // Convert a raster (pixel) sample coordinate to the physical film-plane
    // coordinate (metres) that generate_ray()'s CameraSample expects (see its
    // pfx/pfy comment - unlike OrthographicCamera/SphericalCamera::generate_ray,
    // this one does NOT do its own raster normalization, matching pbrt-v4's
    // RealisticCamera::GenerateRay, which normalizes pFilm by film resolution
    // and lerps into physicalExtent itself before calling this). raster_x/y
    // are in [0, image_width]/[0, image_height]; returns physical film coords
    // in [-film_half_x_, film_half_x_] / [-film_half_y_, film_half_y_].
    void raster_to_film(T raster_x, T raster_y, int image_width, int image_height,
                        T& film_x, T& film_y) const {
        T nx = raster_x / T(image_width);
        T ny = raster_y / T(image_height);
        film_x = (T(2)*nx - T(1)) * film_half_x_;
        film_y = (T(2)*ny - T(1)) * film_half_y_;
    }

    // -----------------------------------------------------------------------
    // GPU-port accessors: expose the (already focus-adjusted) lens table,
    // exit-pupil bounds table, and camera-to-world basis so a caller (e.g.
    // gpu/optix/scene_builder.cpp) can upload them as flat device buffers
    // without re-running FocusThickLens/BoundExitPupil on the GPU - both are
    // one-time, host-only precomputes, the same cost class as building a BVH.
    // -----------------------------------------------------------------------
    T lens_curvature_radius(int i) const { return elements_[i].curvatureRadius; }
    T lens_thickness(int i)        const { return elements_[i].thickness; }
    T lens_eta(int i)              const { return elements_[i].eta; }
    T lens_aperture_radius(int i)  const { return elements_[i].apertureRadius; }

    int num_exit_pupil_bounds() const { return (int)exit_pupil_bounds_.size(); }
    T    exit_pupil_xmin(int i)      const { return exit_pupil_bounds_[i].xMin; }
    T    exit_pupil_xmax(int i)      const { return exit_pupil_bounds_[i].xMax; }
    T    exit_pupil_ymin(int i)      const { return exit_pupil_bounds_[i].yMin; }
    T    exit_pupil_ymax(int i)      const { return exit_pupil_bounds_[i].yMax; }
    bool exit_pupil_degenerate(int i) const { return exit_pupil_bounds_[i].degenerate; }

    CamVec3<T> world_origin()  const { return camera_to_world_.transform_point(T(0), T(0), T(0)); }
    CamVec3<T> world_right()   const { return camera_to_world_.transform_vec(T(1), T(0), T(0)); }
    CamVec3<T> world_up()      const { return camera_to_world_.transform_vec(T(0), T(1), T(0)); }
    CamVec3<T> world_forward() const { return camera_to_world_.transform_vec(T(0), T(0), T(1)); }

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
                // realistic_detail::Refract's own doc comment says "eta = eta_i /
                // eta_t" - trace_lenses_from_scene (below) passes eta_i/eta_t
                // correctly using its own (direction-appropriately-swapped)
                // eta_i/eta_t locals; this call previously passed the inverted
                // eta_t/eta_i (apparently transcribed from pbrt-v4's own global
                // Refract(), which uses the opposite eta_t/eta_i convention -
                // this codebase's local Refract helper does not), causing every
                // refraction traced from the film to bend the wrong way.
                if (!realistic_detail::Refract(ldx/len, ldy/len, ldz/len,
                                               nx, ny, nz,
                                               eta_i/eta_t,
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
        // The "+1"/"-1" below are 1 METRE offsets (this class's internal unit,
        // like lens_rear_z()/lens_front_z()) placing the paraxial reference ray
        // far enough from the lens system to approximate a ray from infinity -
        // this previously used T(0.001) (1mm) here, apparently copy-pasted from
        // the unrelated mm->m conversion constant used elsewhere in this class
        // (e.g. `x` just above, or film_half_x_'s constructor). At 1mm the
        // "far" reference point sits at or inside the lens stack itself (this
        // lens's rear_z is only ~4mm), producing a degenerate/wrong cardinal-
        // point calculation - the root cause of scene D4 (RealisticCamera)
        // rendering almost entirely black except a small central disc: the
        // resulting lens_rear_z was wrong regardless of focus_distance or the
        // scene's own lens data, collapsing all but the innermost exit-pupil
        // bounds to zero area (confirmed empirically: sweeping focus_distance
        // and the seed rear-thickness both left lens_rear_z and the
        // non-degenerate exit-pupil-bounds count unchanged).
        T front_z = T(0); for (const auto& e:elements_) front_z += e.thickness;
        T pz0=T(0), fz0=T(0);
        {
            T ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz;
            bool ok = trace_lenses_from_scene(
                x, T(0), front_z + T(1),
                T(0), T(0), T(-1),
                ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz);
            if (ok)
                compute_cardinal_points(
                    x, T(0), front_z+T(1), T(0), T(0), T(-1),
                    ro_ox, ro_oy, ro_oz, ro_dx, ro_dy, ro_dz,
                    pz0, fz0);
        }

        // --- Film-side cardinal points ---
        // pbrt-v4: rFilm = Ray( (x, 0, LensRearZ-1), (0,0,1) )
        // LensRearZ in pbrt-v4 is the last element's thickness (>0), a positive z.
        // We use orig_rear as that distance. See the scene-side comment above
        // for why this is a 1-metre offset, not 1mm.
        T pz1=T(0), fz1=T(0);
        {
            T rs_ox, rs_oy, rs_oz, rs_dx, rs_dy, rs_dz;
            T w = trace_lenses_from_film(
                x, T(0), orig_rear - T(1),
                T(0), T(0), T(1),
                rs_ox, rs_oy, rs_oz, rs_dx, rs_dy, rs_dz);
            if (w != T(0))
                compute_cardinal_points(
                    x, T(0), orig_rear-T(1), T(0), T(0), T(1),
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
