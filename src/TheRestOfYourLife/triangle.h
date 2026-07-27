#ifndef TRIANGLE_H
#define TRIANGLE_H
//==============================================================================================
// triangle.h -- Triangle primitive with pbrt-v4-aligned intersection and NEE sampling.
//
// Intersection: watertight Woop/Möller-Trumbore transform (pbrt-v4 IntersectTriangle,
//               shapes.cpp:168-268). Works for both front- and back-facing triangles.
//
// Sampling: spherical-triangle solid-angle sampling (Arvo 1995), ported from
//           pbrt-v4 SampleSphericalTriangle (util/sampling.cpp:28-110).
//           pdf_value = 1/solidAngle  (constant, no area->SA conversion needed).
//
// Normal/UV interpolation: per-vertex normals and UVs interpolated via barycentric coords.
//==============================================================================================

#include "hittable.h"
#include "../shared/sampling.h"
#include <array>
#include <memory>


// ---------------------------------------------------------------------------
// triangle_mesh_data
// Shared vertex/normal/UV data for a group of triangles (avoids per-triangle copies).
// Mirrors pbrt-v4 TriangleMesh ownership model.
// ---------------------------------------------------------------------------
struct triangle_mesh_data {
	std::vector<point3>   positions;   // per-vertex positions
	std::vector<vec3>     normals;     // per-vertex normals (may be empty)
	std::vector<double>   uvs;         // per-vertex UV pairs: [u0,v0, u1,v1, ...]  (may be empty)
	std::vector<int>      indices;     // triangle index triples: [i0,i1,i2, ...]

	int num_triangles() const { return (int)(indices.size() / 3); }
	bool has_normals()  const { return !normals.empty(); }
	bool has_uvs()      const { return !uvs.empty(); }
};


// ---------------------------------------------------------------------------
// triangle
// A single triangle referencing shared mesh data.
// ---------------------------------------------------------------------------
class triangle : public hittable {
  public:
	triangle(std::shared_ptr<triangle_mesh_data> mesh,
			 int tri_index,
			 std::shared_ptr<material> mat)
		: mesh(mesh), tri_idx(tri_index), mat(mat)
	{
		const int* idx = &mesh->indices[3 * tri_idx];
		const point3& p0 = mesh->positions[idx[0]];
		const point3& p1 = mesh->positions[idx[1]];
		const point3& p2 = mesh->positions[idx[2]];

		// Precompute geometry normal and area
		vec3 e1 = p1 - p0, e2 = p2 - p0;
		geom_normal = unit_vector(cross(e1, e2));
		area = cross(e1, e2).length() * 0.5;

		// Build bounding box
		auto mn = point3(
			std::min({p0.x(), p1.x(), p2.x()}),
			std::min({p0.y(), p1.y(), p2.y()}),
			std::min({p0.z(), p1.z(), p2.z()}));
		auto mx = point3(
			std::max({p0.x(), p1.x(), p2.x()}),
			std::max({p0.y(), p1.y(), p2.y()}),
			std::max({p0.z(), p1.z(), p2.z()}));
		// Pad degenerate axes by 1e-4 to avoid zero-thickness AABB
		const double pad = 1e-4;
		bbox = aabb(
			point3(mn.x()-pad, mn.y()-pad, mn.z()-pad),
			point3(mx.x()+pad, mx.y()+pad, mx.z()+pad));
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		const int* idx = &mesh->indices[3 * tri_idx];
		const point3& p0 = mesh->positions[idx[0]];
		const point3& p1 = mesh->positions[idx[1]];
		const point3& p2 = mesh->positions[idx[2]];

		// ------------------------------------------------------------------
		// Watertight Woop intersection test (pbrt-v4 IntersectTriangle)
		// ------------------------------------------------------------------
		// Translate vertices to ray origin
		double p0tx = p0.x() - r.origin().x();
		double p0ty = p0.y() - r.origin().y();
		double p0tz = p0.z() - r.origin().z();
		double p1tx = p1.x() - r.origin().x();
		double p1ty = p1.y() - r.origin().y();
		double p1tz = p1.z() - r.origin().z();
		double p2tx = p2.x() - r.origin().x();
		double p2ty = p2.y() - r.origin().y();
		double p2tz = p2.z() - r.origin().z();

		// Permute: kz = largest abs component of direction
		double dx = r.direction().x(), dy = r.direction().y(), dz = r.direction().z();
		double ax = std::abs(dx), ay = std::abs(dy), az = std::abs(dz);
		int kz, kx, ky;
		if (ax >= ay && ax >= az) { kz = 0; kx = 1; ky = 2; }
		else if (ay >= ax && ay >= az) { kz = 1; kx = 2; ky = 0; }
		else { kz = 2; kx = 0; ky = 1; }

		auto perm = [&](double x, double y, double z, int k) -> double {
			return k == 0 ? x : (k == 1 ? y : z);
		};
		double Dx = perm(dx,dy,dz,kx), Dy = perm(dx,dy,dz,ky), Dz = perm(dx,dy,dz,kz);
		double p0x = perm(p0tx,p0ty,p0tz,kx), p0y = perm(p0tx,p0ty,p0tz,ky), p0z = perm(p0tx,p0ty,p0tz,kz);
		double p1x = perm(p1tx,p1ty,p1tz,kx), p1y = perm(p1tx,p1ty,p1tz,ky), p1z = perm(p1tx,p1ty,p1tz,kz);
		double p2x = perm(p2tx,p2ty,p2tz,kx), p2y = perm(p2tx,p2ty,p2tz,ky), p2z = perm(p2tx,p2ty,p2tz,kz);

		// Shear
		double Sx = -Dx / Dz, Sy = -Dy / Dz, Sz = 1.0 / Dz;
		p0x += Sx * p0z; p0y += Sy * p0z;
		p1x += Sx * p1z; p1y += Sy * p1z;
		p2x += Sx * p2z; p2y += Sy * p2z;

		// Edge function coefficients
		double e0 = p1x*p2y - p1y*p2x;
		double e1 = p2x*p0y - p2y*p0x;
		double e2 = p0x*p1y - p0y*p1x;

		// Determinant and sign test
		if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
			return false;
		double det = e0 + e1 + e2;
		if (det == 0.0) return false;

		// Compute t
		p0z *= Sz; p1z *= Sz; p2z *= Sz;
		double tScaled = e0*p0z + e1*p1z + e2*p2z;
		if (det < 0 && (tScaled >= 0 || tScaled < ray_t.max * det)) return false;
		if (det > 0 && (tScaled <= 0 || tScaled > ray_t.max * det)) return false;

		double invDet = 1.0 / det;
		double b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
		double t  = tScaled * invDet;

		if (!ray_t.contains(t)) return false;

		// ------------------------------------------------------------------
		// Fill hit_record
		// ------------------------------------------------------------------
		rec.t = t;
		rec.p = r.at(t);

		// Interpolated normal
		vec3 shading_n;
		if (mesh->has_normals()) {
			shading_n = unit_vector(
				b0 * mesh->normals[idx[0]] +
				b1 * mesh->normals[idx[1]] +
				b2 * mesh->normals[idx[2]]);
		} else {
			shading_n = geom_normal;
		}
		rec.set_face_normal(r, shading_n);

		// UV
		if (mesh->has_uvs()) {
			double u0 = mesh->uvs[2*idx[0]], v0 = mesh->uvs[2*idx[0]+1];
			double u1 = mesh->uvs[2*idx[1]], v1 = mesh->uvs[2*idx[1]+1];
			double u2 = mesh->uvs[2*idx[2]], v2 = mesh->uvs[2*idx[2]+1];
			rec.u = b0*u0 + b1*u1 + b2*u2;
			rec.v = b0*v0 + b1*v1 + b2*v2;
		} else {
			rec.u = b1;  // barycentric fallback
			rec.v = b2;
		}

		// dpdu: approximate tangent along u axis (used by normal maps)
		{
			vec3 e1v = p1 - p0, e2v = p2 - p0;
			rec.dpdu = e1v.length_squared() > 1e-14 ? unit_vector(e1v) : vec3(1,0,0);
		}

		rec.mat = mat;
		return true;
	}

	aabb bounding_box() const override { return bbox; }

	// pdf_value: solid-angle PDF for sampling this triangle as a light
	double pdf_value(const point3& origin, const vec3& direction) const override {
		const int* idx = &mesh->indices[3 * tri_idx];
		const point3& p0 = mesh->positions[idx[0]];
		const point3& p1 = mesh->positions[idx[1]];
		const point3& p2 = mesh->positions[idx[2]];

		// Quick check: does the direction actually hit?
		hit_record rec;
		if (!this->hit(ray(origin, direction), interval(1e-4, infinity), rec))
			return 0.0;

		double sa = SphericalTriangleSolidAngle(
			p0.x(),p0.y(),p0.z(),
			p1.x(),p1.y(),p1.z(),
			p2.x(),p2.y(),p2.z(),
			origin.x(),origin.y(),origin.z());

		if (sa > 1e-10) return 1.0 / sa;

		// Fallback: area-based PDF
		double dist2 = (rec.t * rec.t) * direction.length_squared();
		double cosine = std::fabs(dot(direction, rec.normal) / direction.length());
		return (cosine > 1e-10) ? dist2 / (cosine * area) : 0.0;
	}

	// random: sample a direction toward this triangle using solid-angle sampling
	vec3 random(const point3& origin) const override {
		const int* idx = &mesh->indices[3 * tri_idx];
		const point3& p0 = mesh->positions[idx[0]];
		const point3& p1 = mesh->positions[idx[1]];
		const point3& p2 = mesh->positions[idx[2]];

		double b0, b1, b2, pdf;
		SampleSphericalTriangle(
			p0.x(),p0.y(),p0.z(),
			p1.x(),p1.y(),p1.z(),
			p2.x(),p2.y(),p2.z(),
			origin.x(),origin.y(),origin.z(),
			random_double(), random_double(),
			&b0, &b1, &b2, &pdf);

		// Sampled point on triangle
		point3 pt = b0*p0 + b1*p1 + b2*p2;
		return pt - origin;
	}

	// Accessors
	double get_area() const { return area; }
	vec3   get_geom_normal() const { return geom_normal; }
	std::shared_ptr<material> get_material() const { return mat; }

  private:
	std::shared_ptr<triangle_mesh_data> mesh;
	int tri_idx;
	std::shared_ptr<material> mat;
	vec3  geom_normal;
	double area;
	aabb   bbox;
};


#endif
