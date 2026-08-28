#pragma once
// sphere_clipped_hittable.h -- pbrt-v4 Shape "sphere" with real zmin/zmax/
// phimax clipping (caps/wedges/hemispheres), CPU backend.
//
// A full (unclipped) sphere is rotation-invariant, so pbrt_flatten.h bakes
// it straight to a world-space center+radius (see pbrt_flatten::Sphere's
// own comment) and the plain `sphere` class (sphere.h) renders it. Once
// zmin/zmax/phimax clip away part of the sphere, it is no longer
// orientation-independent, so this class instead carries the real
// object-to-world transform and intersects the ray in OBJECT space against
// the shared, already-tested SphereShape<T> clipping math (src/shared/
// shapes.h) - exactly disk_hittable/cylinder_hittable's own technique (see
// disk_cylinder_hittable.h's own header comment for why: the ray moves, the
// geometry doesn't). Non-uniform scale is therefore handled EXACTLY for
// intersection here, unlike the unclipped-sphere baking path's "warn and
// use the largest axis" shortcut (pbrt_flatten.h's own sphere-parsing
// comment) - carrying the real transform removes the need for that
// approximation entirely, the same way it already does for disk/cylinder.
//
// Solid-angle NEE sampling (random()/pdf_value(), forwarded to
// SphereShape<T>::sample_from()/pdf_from()) samples/weights over the FULL
// (unclipped) sphere's subtended cone - a pre-existing property of that
// shared template (see its own "Solid-angle sample"/"Solid-angle PDF"
// sections), not something this class adds. In practice this only matters
// when a clipped sphere is ALSO declared as an AreaLightSource - a narrow,
// rare combination - and its effect is extra sampling noise (some proposed
// light directions land on the clipped-away, invisible part of the sphere
// and contribute nothing), not bias: the real shading-point intersection
// above still correctly refuses to hit the clipped-away region. This
// mirrors this codebase's own accepted "approximate under uniform-scale"
// caveat for disk_hittable/cylinder_hittable's NEE sampling - documented,
// not solved, since fixing it needs restricting the sampled cone to the
// visible cap, a separate, self-contained piece of work.

#include "hittable.h"
#include "material.h"
#include "../shared/pbrt_scene.h"
#include "../shared/shapes.h"

#include "affine_transform_apply.h"

#include <algorithm>
#include <cmath>
#include <memory>


class sphere_clipped_hittable : public hittable {
  public:
	sphere_clipped_hittable(double radius, double z_min, double z_max, double phi_max,
							 const pbrt_scene::Matrix4& object_to_world, shared_ptr<material> mat)
		: shape_(SphereShape<double>::make_clipped(0.0, 0.0, 0.0, radius, z_min, z_max, phi_max)),
		  mat_(mat), o2w_(object_to_world)
	{
		valid_ = o2w_.inverseAffine(w2o_);
		if (!valid_) return;
		bbox_ = affine_transform::transformed_bbox(o2w_,
			-radius, radius, -radius, radius, z_min, z_max);
		// thetaZMin_/thetaZMax_ depend only on z_min/z_max/radius - all fixed
		// for the object's lifetime - so they're computed once here instead
		// of with 2 acos() calls on every hit()/sample_area() call.
		thetaZMin_ = std::acos(std::clamp(z_max / radius, -1.0, 1.0));
		thetaZMax_ = std::acos(std::clamp(z_min / radius, -1.0, 1.0));
		// World-space area (needed for sample_area()'s pdf_pos) estimated
		// from a single representative scale factor of the object->world
		// transform - exact under a similarity transform (rotation/
		// translation/uniform scale), approximate under anisotropic scale,
		// the same accepted simplification this codebase's GPU disk/
		// cylinder area lights already use for the identical reason (see
		// docs/PBRT_SUPPORT.md's disk/cylinder entry). Read directly as the
		// linear part's column 0 (o2w_ applied to the unit X axis) rather
		// than going through the general apply_vector() path - the zero Y/Z
		// components make that path's extra work pure overhead here.
		const vec3 ex(o2w_.m[0], o2w_.m[4], o2w_.m[8]);
		areaScale2_ = std::fmax(1e-18, ex.length_squared());
		// pdf_pos is also object-lifetime-invariant (same inputs as
		// thetaZMin_/thetaZMax_ above plus areaScale2_) - cached for the
		// same reason, since sample_area() is a per-emitter-sample hot path
		// under --bdpt/--sppm.
		const double objectArea = phi_max * radius * (z_max - z_min);
		const double worldArea = objectArea * areaScale2_;
		pdfPos_ = (worldArea > 0.0) ? 1.0 / worldArea : 0.0;
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace affine_transform;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see disk_cylinder_hittable.h

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		const double hz = ro.z() + h->t * rd.z();
		rec.p = apply_point(o2w_, point3(hx, hy, hz));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Sphere::Intersect: dpdu tangent to increasing phi, dpdv
		// tangent to increasing theta (both re-derived here since ShapeHit
		// only carries the final normal/uv, not the intermediate angles).
		// zRadius = radius*sin(theta), so -radius*sin(theta) collapses to -zRadius.
		const double radius = shape_.r;
		const double zRadius = std::sqrt(hx * hx + hy * hy);
		vec3 dpdu_obj, dpdv_obj;
		if (zRadius > 1e-9 * radius) {
			const double cosPhi = hx / zRadius, sinPhi = hy / zRadius;
			dpdu_obj = vec3(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
			dpdv_obj = (thetaZMax_ - thetaZMin_) * vec3(hz * cosPhi, hz * sinPhi, -zRadius);
		} else {
			// At a pole (zRadius ~ 0): phi is undefined, so dpdu's direction
			// is a genuine coordinate singularity (same as longitude lines
			// converging at Earth's poles) - correct degenerate behavior,
			// not a bug, matching sphere.h's own identical pole handling.
			// Only dpdu's DIRECTION needs an arbitrary but valid substitute
			// (magnitude correctly ~0, phi_max * zRadius) - dpdv does NOT
			// vanish at a pole (a line of longitude has a well-defined
			// direction and nonzero length-rate right at the pole, same as
			// sphere.h's own pole fallback keeps its dpdv coefficient at
			// the constant radius*pi with no vanishing factor), so dpdv's
			// magnitude here is the same (thetaZMax_-thetaZMin_)*radius the
			// general-case formula above always produces, NOT scaled by
			// the (here, ~0) zRadius.
			const vec3 n_obj(h->nx, h->ny, h->nz);   // == (hx,hy,hz)/radius, already computed by intersect()
			vec3 tangent = cross(vec3(0, 1, 0), n_obj);
			const double tlen = tangent.length();
			const vec3 dir = (tlen > 1e-6) ? (tangent / tlen) : vec3(1, 0, 0);
			dpdu_obj = dir * (shape_.phi_max * zRadius);
			dpdv_obj = cross(n_obj, dir) * (radius * (thetaZMax_ - thetaZMin_));
		}
		rec.dpdu = apply_vector(o2w_, dpdu_obj);
		rec.dpdv = apply_vector(o2w_, dpdv_obj);

		rec.set_face_normal(r, worldNormal(vec3(h->nx, h->ny, h->nz)));
		rec.mat = mat_;
		return true;
	}

	aabb bounding_box() const override { return bbox_; }

	// Solid-angle NEE sampling - see file header comment for the clipping
	// caveat (and disk_cylinder_hittable.h's own header comment for the
	// analogous uniform-scale one, which applies here too).
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w_, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const vec3 dir_obj = apply_vector(w2o_, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	// Area-uniform sample of this shape's own (clipped) surface, independent
	// of any reference point - needed for BDPT/SPPM's emitter list, which
	// gates entirely on this returning true (bdpt_adapter.h/sppm_adapter.h
	// skip any object where it doesn't - a clipped sphere with no override
	// here would silently emit no light at all under --bdpt/--sppm, unlike
	// the plain `sphere` class it replaces for the clipped case, which does
	// implement this). Unlike random()/pdf_value() above (NEE, which samples
	// the full sphere's cone and is documented as noise-not-bias for that
	// reason - a shadow ray landing off the visible cap simply misses), a
	// sample from here is used DIRECTLY as an emission point without ever
	// being re-intersected, so sampling outside the visible cap would be a
	// real bias, not just noise. Archimedes' hat-box theorem makes this
	// exact and rejection-free: for a sphere, uniform-z + uniform-phi IS
	// uniform-area (the sphere's surface-area element is r*dz*dphi,
	// constant in z), so sampling z uniformly in [z_min,z_max] and phi
	// uniformly in [0,phi_max) lands correctly on the visible cap only.
	bool sample_area(double u1, double u2, AreaLightSample& out) const override {
		if (!valid_) return false;
		const double radius = shape_.r;
		const double z = shape_.z_min + u1 * (shape_.z_max - shape_.z_min);
		const double phi = u2 * shape_.phi_max;
		const double rho = std::sqrt(std::max(0.0, radius * radius - z * z));
		const point3 p_obj(rho * std::cos(phi), rho * std::sin(phi), z);
		out.p = affine_transform::apply_point(o2w_, p_obj);
		out.n = worldNormal(vec3(p_obj.x() / radius, p_obj.y() / radius, p_obj.z() / radius),
							 vec3(0, 0, 1));
		const double theta = std::acos(std::clamp(z / radius, -1.0, 1.0));
		out.u = phi / shape_.phi_max;
		out.v = (thetaZMax_ > thetaZMin_) ? (theta - thetaZMin_) / (thetaZMax_ - thetaZMin_) : 0.0;
		out.pdf_pos = pdfPos_;
		return true;
	}

	shared_ptr<material> get_material() const { return mat_; }

  private:
	// Transforms an object-space unit normal to world space and re-
	// normalises (a general affine transform doesn't preserve length) -
	// shared by hit() and sample_area(), which otherwise each repeated this
	// identical 3-statement "transform, measure, divide-or-fallback"
	// pattern. `fallback` differs per caller: hit() keeps the original
	// (unnormalised, i.e. zero-vector-on-degenerate) behavior by defaulting
	// to (0,0,0), while sample_area() substitutes a real unit vector.
	vec3 worldNormal(const vec3& n_obj, const vec3& fallback = vec3(0, 0, 0)) const {
		const vec3 n = affine_transform::apply_normal(w2o_, n_obj);
		const double len = n.length();
		return (len > 0.0) ? (n / len) : fallback;
	}

	SphereShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
	double thetaZMin_ = 0.0, thetaZMax_ = 0.0;
	double areaScale2_ = 1.0;
	double pdfPos_ = 0.0;
};
