#ifndef SPHERE_H
#define SPHERE_H
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

#include "hittable.h"
#include "onb.h"


class sphere : public hittable {
  public:
    // Stationary Sphere
    sphere(const point3& static_center, double radius, shared_ptr<material> mat)
      : center(static_center, vec3(0,0,0)), radius(std::fmax(0,radius)), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(static_center - rvec, static_center + rvec);
    }

    // Moving Sphere
    sphere(const point3& center1, const point3& center2, double radius,
           shared_ptr<material> mat)
      : center(center1, center2 - center1), radius(std::fmax(0,radius)), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(box1, box2);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        point3 current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;
        // dpdu = tangent along U for sphere UV mapping (phi direction)
        // ∂p/∂u ∝ (-sin(phi), 0, cos(phi)) in world space ∝ (-pz, 0, px) normalised.
        // For the equirectangular parameterization used by get_sphere_uv:
        //   p = (sin(theta)*cos(phi), -cos(theta), -sin(theta)*sin(phi)) * radius  [shifted]
        // A stable tangent: use the cross product of world Y and outward normal.
        {
            vec3 n = outward_normal;
            vec3 world_up(0, 1, 0);
            vec3 tangent = cross(world_up, n);
            double tlen = tangent.length();
            if (tlen > 1e-6)
                rec.dpdu = tangent / tlen;
            else
                rec.dpdu = vec3(1, 0, 0); // fallback at poles
        }

        return true;
    }

    aabb bounding_box() const override { return bbox; }

    // Accessors for serialization
    point3 center_at_time0() const { return center.at(0); }
    double get_radius() const { return radius; }
    shared_ptr<material> get_material() const { return mat; }
    // For transforms we may need to move points; provide setter
    void set_center(const point3 &p) { center = ray(p, vec3(0,0,0)); bbox = aabb(p - vec3(radius,radius,radius), p + vec3(radius,radius,radius)); }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        // This method only works for stationary spheres.

        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001, infinity), rec))
            return 0;

        auto dist_squared = (center.at(0) - origin).length_squared();
        auto sin2ThetaMax = radius * radius / dist_squared;
        auto cosThetaMax  = std::sqrt(1.0 - sin2ThetaMax > 0.0 ? 1.0 - sin2ThetaMax : 0.0);

        // pbrt-v4: use Taylor expansion for small solid angle to avoid
        // catastrophic cancellation in (1 - cosThetaMax) when sphere is
        // far away (sin2ThetaMax < sin²(1.5°) ≈ 0.00068523)
        double oneMinusCosThetaMax;
        if (sin2ThetaMax < 0.00068523)
            oneMinusCosThetaMax = sin2ThetaMax / 2.0;
        else
            oneMinusCosThetaMax = 1.0 - cosThetaMax;

        auto solid_angle = 2 * pi * oneMinusCosThetaMax;
        return 1.0 / solid_angle;
    }

    vec3 random(const point3& origin) const override {
        vec3 direction = center.at(0) - origin;
        auto distance_squared = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared));
    }

    // sample_area: uniform sample of this sphere's own surface, independent
    // of any reference point (needed for light emission / photon tracing, as
    // opposed to random()'s solid-angle sample toward a viewer). Only
    // correct for stationary spheres (uses center.at(0)), matching
    // pdf_value()'s own documented limitation above.
    bool sample_area(double u1, double u2, AreaLightSample& out) const override {
        double z = 1.0 - 2.0*u1;
        double r = std::sqrt(std::max(0.0, 1.0 - z*z));
        double phi = 2.0*pi*u2;
        vec3 n_local(r*std::cos(phi), r*std::sin(phi), z);   // unit sphere point
        out.n = n_local;
        out.p = center.at(0) + radius * n_local;
        get_sphere_uv(n_local, out.u, out.v);
        out.pdf_pos = (radius > 0.0) ? 1.0 / (4.0 * pi * radius * radius) : 0.0;
        return true;
    }

  private:
    ray center;
    double radius;
    shared_ptr<material> mat;
    aabb bbox;

    static void get_sphere_uv(const point3& p, double& u, double& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = std::acos(-p.y());
        auto phi = std::atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }

    static vec3 random_to_sphere(double radius, double distance_squared) {
        auto r1 = random_double();
        auto r2 = random_double();
        auto z = 1 + r2*(std::sqrt(1-radius*radius/distance_squared) - 1);

        auto phi = 2*pi*r1;
        auto x = std::cos(phi) * std::sqrt(1-z*z);
        auto y = std::sin(phi) * std::sqrt(1-z*z);

        return vec3(x, y, z);
    }
};


#endif
