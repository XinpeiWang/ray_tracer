#ifndef PDF_H
#define PDF_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable_list.h"
#include "onb.h"
#include "../shared/shading_frame.h"
#include "../shared/bxdfs_conductor.h"


class pdf {
  public:
    virtual ~pdf() {}

    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};


class sphere_pdf : public pdf {
  public:
    sphere_pdf() {}

    double value(const vec3& direction) const override {
        return 1/ (4 * pi);
    }

    vec3 generate() const override {
        return random_unit_vector();
    }
};


class cosine_pdf : public pdf {
  public:
    cosine_pdf(const vec3& w) : uvw(w) {}

    double value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return std::fmax(0, cosine_theta/pi);
    }

    vec3 generate() const override {
        return uvw.transform(random_cosine_direction());
    }

  private:
    onb uvw;
};


// The sampling density of pbrt-v4's DiffuseTransmissionBxDF: a cosine lobe about the normal chosen with probability p_reflect and a cosine lobe about
// -normal with probability p_transmit (pr / (pr + pt) and pt / (pr + pt)). generate() makes that choice itself and value() is the density of the whole
// mixture at any queried direction - what MIS and the path weight need. A pdf of just the lobe a coin flip in scatter() happened to commit to is not:
// the weight used to carry the lobe probability twice (once in scattering_pdf(), once through the committed lobe) and a surface reflecting 0.2 and
// transmitting 0.6 read a quarter of its reflection.
class two_sided_cosine_pdf : public pdf {
  public:
    two_sided_cosine_pdf(const vec3& normal, double p_reflect, double p_transmit)
      : uvw(normal), p_reflect(p_reflect), p_transmit(p_transmit) {}

    double value(const vec3& direction) const override {
        const double c = dot(unit_vector(direction), uvw.w());
        return c > 0.0 ? p_reflect * c / pi : p_transmit * (-c) / pi;
    }

    vec3 generate() const override {
        const vec3 d = uvw.transform(random_cosine_direction());
        const double u = random_double() * (p_reflect + p_transmit);
        // d is cosine-distributed about +normal; mirrored across the surface plane it is cosine-distributed about -normal.
        return u < p_reflect ? d : d - 2.0 * dot(d, uvw.w()) * uvw.w();
    }

  private:
    onb uvw;
    double p_reflect, p_transmit;
};


class hittable_pdf : public pdf {
  public:
    hittable_pdf(const hittable& objects, const point3& origin)
      : objects(objects), origin(origin)
    {}

    double value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    vec3 generate() const override {
        return objects.random(origin);
    }

  private:
    const hittable& objects;
    point3 origin;
};


// ggx_reflection_pdf -- VNDF-based sampling density for a GGX rough-
// conductor reflection lobe (RoughMetalBxDF / ConductorBxDF share the
// identical geometric sampling density -- Fresnel affects f(), not pdf(),
// see ggx_vndf_reflection_pdf in bxdfs_conductor.h). Used as srec.pdf_ptr
// for real NEE/MIS on rough_metal/conductor: generate() draws the same
// VNDF-sampled direction sample_local() would, value() evaluates the
// sampling density at an ARBITRARY queried direction (e.g. a shadow ray
// toward a light).
class ggx_reflection_pdf : public pdf {
  public:
    ggx_reflection_pdf(const vec3& normal_, const vec3& wo_world_,
                        double alpha_x_, double alpha_y_)
        : frame(ShadingFrame<double>::from_normal(normal_.x(), normal_.y(), normal_.z())),
          alpha_x(alpha_x_), alpha_y(alpha_y_) {
        frame.to_local(wo_world_.x(), wo_world_.y(), wo_world_.z(), wi_x, wi_y, wi_z);
    }

    double value(const vec3& direction) const override {
        vec3 d = unit_vector(direction);
        double lx, ly, lz;
        frame.to_local(d.x(), d.y(), d.z(), lx, ly, lz);
        return ggx_vndf_reflection_pdf(wi_x, wi_y, wi_z, lx, ly, lz, alpha_x, alpha_y);
    }

    vec3 generate() const override {
        TrowbridgeReitz<double> dist(alpha_x, alpha_y);
        double wm_x, wm_y, wm_z;
        dist.Sample_wm(wi_x, wi_y, wi_z, random_double(), random_double(), wm_x, wm_y, wm_z);
        double dot_wi_wm = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
        double lo_x = 2*dot_wi_wm*wm_x - wi_x;
        double lo_y = 2*dot_wi_wm*wm_y - wi_y;
        double lo_z = 2*dot_wi_wm*wm_z - wi_z;
        double wo_x, wo_y, wo_z;
        frame.to_world(lo_x, lo_y, lo_z, wo_x, wo_y, wo_z);
        return unit_vector(vec3(wo_x, wo_y, wo_z));
    }

  private:
    ShadingFrame<double> frame;
    double wi_x, wi_y, wi_z;
    double alpha_x, alpha_y;
};


// ggx_dielectric_pdf -- VNDF-based sampling density for rough dielectric
// reflection + transmission (both lobes, including the discrete Fresnel
// branch choice), used as srec.pdf_ptr for rough_dielectric's real
// NEE/MIS. generate() reuses RoughDielectricBxDF::sample_local() directly
// (same VNDF + Fresnel-branch + TIR-fallback logic, not re-derived here);
// value() evaluates RoughDielectricBxDF::pdf() at an arbitrary queried
// direction (e.g. a shadow ray toward a light). `eta` matches
// sample_local()'s own convention (eta_i/eta_t, resolved once by the
// caller for entering vs exiting -- see rough_dielectric::scatter()).
class ggx_dielectric_pdf : public pdf {
  public:
    ggx_dielectric_pdf(const vec3& normal_, const vec3& wo_world_,
                        double eta_, double alpha_x_, double alpha_y_)
        : frame(ShadingFrame<double>::from_normal(normal_.x(), normal_.y(), normal_.z())),
          eta(eta_), bxdf{1.5, alpha_x_, alpha_y_} {
        frame.to_local(wo_world_.x(), wo_world_.y(), wo_world_.z(), wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }
    }

    double value(const vec3& direction) const override {
        vec3 d = unit_vector(direction);
        double lx, ly, lz;
        frame.to_local(d.x(), d.y(), d.z(), lx, ly, lz);
        return bxdf.pdf(wi_x, wi_y, wi_z, eta, lx, ly, lz);
    }

    vec3 generate() const override {
        auto res = bxdf.sample_local(wi_x, wi_y, wi_z, eta,
                                      random_double(), random_double(), random_double());
        if (!res.valid) {
            // sample_local() rejects rare grazing-reflect edge cases (see
            // its own comment) -- return a tangent direction (wo_z=0 in
            // local frame), which value() above correctly evaluates to
            // exactly 0 (both f() and pdf() early-return 0 when wo_z==0).
            // camera.h's NEE/Strategy-B both check the density <= 0 before
            // using a direction, so this is a safe, zero-contribution
            // placeholder.
            double tx, ty, tz;
            frame.to_world(1.0, 0.0, 0.0, tx, ty, tz);
            return vec3(tx, ty, tz);
        }
        double wo_x, wo_y, wo_z;
        frame.to_world(res.wo_x, res.wo_y, res.wo_z, wo_x, wo_y, wo_z);
        return unit_vector(vec3(wo_x, wo_y, wo_z));
    }

  private:
    ShadingFrame<double> frame;
    double wi_x, wi_y, wi_z;
    double eta;
    RoughDielectricBxDF<double> bxdf;
};


class mixture_pdf : public pdf {
  public:
    mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }

    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
    }

    vec3 generate() const override {
        if (random_double() < 0.5)
            return p[0]->generate();
        else
            return p[1]->generate();
    }

  private:
    shared_ptr<pdf> p[2];
};


#endif
