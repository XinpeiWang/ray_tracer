#ifndef HITTABLE_H
#define HITTABLE_H
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

#include "rtweekend.h"
#include "aabb.h"


class material;


class hit_record {
  public:
    point3 p;
    vec3 normal;
    vec3 dpdu;      // surface tangent along U (world space) -- for normal/bump mapping
    vec3 dpdv;      // surface tangent along V (world space) -- for texture-differential filtering
    shared_ptr<material> mat;
    double t;
    double u;
    double v;
    bool front_face;

    // Texture-lookup footprint (pbrt-v4 SurfaceInteraction's dudx/dvdx/
    // dudy/dvdy) -- set only for the primary camera ray hit (see camera.h's
    // ray_color()), left at their zero defaults for every bounce/shadow/NEE
    // ray, which is also mipmap.h's own correct "no footprint info, fall
    // back to plain bilinear" case (see texture.h's value_diff()).
    bool has_differentials = false;
    double dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


// AreaLightSample -- result of hittable::sample_area(): a uniformly-sampled
// point on a shape's own surface, independent of any reference/viewing
// point. This is the complement of pdf_value()/random() below, which sample
// a *direction toward* the shape from a given reference point (used for
// NEE); sample_area() is needed for light *emission* (e.g. photon tracing),
// which needs to pick a start point on the light before picking a direction
// leaving it.
struct AreaLightSample {
    point3 p;         // sampled point, world space
    vec3   n;         // outward geometric normal at p (unit length)
    double u, v;       // surface parameterization at p (shape's own UV convention)
    double pdf_pos;    // probability density over area, i.e. 1 / surface area
};


class hittable {
  public:
    virtual ~hittable() = default;

    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

    virtual aabb bounding_box() const = 0;

    virtual double pdf_value(const point3& origin, const vec3& direction) const {
        return 0.0;
    }

    virtual vec3 random(const point3& origin) const {
        return vec3(1,0,0);
    }

    // Uniform-area sample of this shape's own surface (see AreaLightSample
    // above). Default: not an area-samplable shape (mirrors the
    // pdf_value/random default-return pattern). Overridden by quad/sphere.
    virtual bool sample_area(double u1, double u2, AreaLightSample& out) const {
        return false;
    }
};


class translate : public hittable {
  public:
    translate(shared_ptr<hittable> object, const vec3& offset)
      : object(object), offset(offset)
    {
        bbox = object->bounding_box() + offset;
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Move the ray backwards by the offset
        ray offset_r(r.origin() - offset, r.direction(), r.time());

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!object->hit(offset_r, ray_t, rec))
            return false;

        // Move the intersection point forwards by the offset
        rec.p += offset;

        return true;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    shared_ptr<hittable> object;
    vec3 offset;
    aabb bbox;
  public:
    // Accessors for serialization
    shared_ptr<hittable> get_object() const { return object; }
    vec3 get_offset() const { return offset; }
};


class rotate_y : public hittable {
  public:
    rotate_y(shared_ptr<hittable> object, double angle) : object(object) {
        auto radians = degrees_to_radians(angle);
        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);
        this->angle = radians;
        bbox = object->bounding_box();

        point3 min( infinity,  infinity,  infinity);
        point3 max(-infinity, -infinity, -infinity);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i*bbox.x.max + (1-i)*bbox.x.min;
                    auto y = j*bbox.y.max + (1-j)*bbox.y.min;
                    auto z = k*bbox.z.max + (1-k)*bbox.z.min;

                    auto newx =  cos_theta*x + sin_theta*z;
                    auto newz = -sin_theta*x + cos_theta*z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fmin(min[c], tester[c]);
                        max[c] = std::fmax(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {

        // Transform the ray from world space to object space.

        auto origin = point3(
            (cos_theta * r.origin().x()) - (sin_theta * r.origin().z()),
            r.origin().y(),
            (sin_theta * r.origin().x()) + (cos_theta * r.origin().z())
        );

        auto direction = vec3(
            (cos_theta * r.direction().x()) - (sin_theta * r.direction().z()),
            r.direction().y(),
            (sin_theta * r.direction().x()) + (cos_theta * r.direction().z())
        );

        ray rotated_r(origin, direction, r.time());

        // Determine whether an intersection exists in object space (and if so, where).

        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.

        rec.p = point3(
            (cos_theta * rec.p.x()) + (sin_theta * rec.p.z()),
            rec.p.y(),
            (-sin_theta * rec.p.x()) + (cos_theta * rec.p.z())
        );

        rec.normal = vec3(
            (cos_theta * rec.normal.x()) + (sin_theta * rec.normal.z()),
            rec.normal.y(),
            (-sin_theta * rec.normal.x()) + (cos_theta * rec.normal.z())
        );

        // dpdu/dpdv are tangent VECTORS (not points), so they rotate the
        // same way normal does (no translation component) -- a pre-existing
        // gap (dpdu was never rotated here before texture-differential
        // filtering existed to care), worth fixing now since rotate_y wraps
        // textured meshes (cow/horse/trophy_cow) whose normal/bump maps and
        // (as of this change) EWA texture filtering both depend on a
        // correctly-oriented world-space tangent frame.
        rec.dpdu = vec3(
            (cos_theta * rec.dpdu.x()) + (sin_theta * rec.dpdu.z()),
            rec.dpdu.y(),
            (-sin_theta * rec.dpdu.x()) + (cos_theta * rec.dpdu.z())
        );
        rec.dpdv = vec3(
            (cos_theta * rec.dpdv.x()) + (sin_theta * rec.dpdv.z()),
            rec.dpdv.y(),
            (-sin_theta * rec.dpdv.x()) + (cos_theta * rec.dpdv.z())
        );

        return true;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    shared_ptr<hittable> object;
    double sin_theta;
    double cos_theta;
    aabb bbox;
    double angle = 0.0;
  public:
    // Accessors for serialization
    shared_ptr<hittable> get_object() const { return object; }
    double get_angle() const { return angle; }
};


#endif
