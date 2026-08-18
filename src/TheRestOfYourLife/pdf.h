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
