#ifndef MESH_H
#define MESH_H
//==============================================================================================
// mesh.h -- OBJ/PLY mesh loader and triangle_mesh hittable.
//
// Loads a Wavefront OBJ file (positions, normals, UVs, face indices) into a
// triangle_mesh_data struct, then creates a BVH of triangle hittables.
//
// Design mirrors pbrt-v4 TriangleMesh + BVHAggregate:
//   - One triangle_mesh_data owns the shared vertex arrays
//   - Each triangle references it via shared_ptr (zero-copy)
//   - A bvh_node wraps all triangles for O(log N) ray traversal
//
// Usage:
//   auto mesh = load_obj("bunny.obj", mat);
//   world.add(mesh);
//==============================================================================================

#include "triangle.h"
#include "bvh.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>


// ---------------------------------------------------------------------------
// load_obj
// Minimal Wavefront OBJ loader.
// Supports:
//   v  x y z           (positions)
//   vn nx ny nz        (normals)
//   vt u v             (texture coordinates)
//   f  v[/vt[/vn]] ... (triangulated faces, >=3 vertices via fan triangulation)
//   o / g / # / mtllib / usemtl (ignored)
//
// Returns a shared_ptr<bvh_node> wrapping all triangles, or throws on error.
// ---------------------------------------------------------------------------
inline std::shared_ptr<hittable> load_obj(
		const std::string& filepath,
		std::shared_ptr<material> mat,
		// Optional transform: scale and translate applied to positions
		double scale = 1.0,
		point3 offset = point3(0,0,0),
		// When true AND the file itself has no "vn" data, generate smooth
		// per-vertex normals (area-weighted average of adjacent face
		// normals) instead of leaving the mesh flat-faceted. Defaults to
		// false so every existing flat-shaded mesh scene (37-49) keeps its
		// current, already-verified faceted look - this is opt-in, not a
		// silent behavior change.
		//
		// Investigated as a possible fix for a dielectric (glass) render of
		// xyzrgb_dragon.obj (250K flat-faceted triangles, deeply concave)
		// producing salt-and-pepper noise that didn't visibly change between
		// 100spp and 10000spp. Smooth normals alone did NOT resolve that -
		// a follow-up pixel-wise MSE comparison across sample counts showed
		// MSE(100spp, 10000spp) was no larger than MSE(3000spp, 10000spp),
		// which rules out ordinary unconverged noise entirely: the image is
		// converging, just to a genuinely chaotic-looking result, since
		// refraction through a deeply concave solid needing dozens of
		// internal bounces is extremely sensitive to sub-pixel ray origin
		// (a well-documented hard case for unidirectional path tracing,
		// requiring bidirectional/photon-mapping style integrators to
		// resolve cleanly - not something either this renderer or pbrt-v4's
		// own default PathIntegrator has). Kept as a real, independently
		// useful capability (smooth-shaded meshes without authored vn data)
		// even though it wasn't the fix for that specific investigation.
		bool smooth_normals = false)
{
	// Hunt for the file the same way rtw_stb_image.h's rtw_image does for
	// earthmap.jpg: try filepath as given first, then models/<filepath>
	// climbing up to 5 parent directories, so a single copy of the asset at
	// the repo root's models/ dir loads correctly regardless of which
	// deploy directory (x64/Release, bin/Release, RayTracer_Package, ...)
	// the renderer's current working directory happens to be.
	std::ifstream file(filepath);
	if (!file.is_open()) {
		static const char* kSearchPrefixes[] = {
			"models/", "../models/", "../../models/",
			"../../../models/", "../../../../models/", "../../../../../models/"
		};
		for (const char* prefix : kSearchPrefixes) {
			file.clear();
			file.open(prefix + filepath);
			if (file.is_open()) break;
		}
	}
	if (!file.is_open())
		throw std::runtime_error("load_obj: cannot open file: " + filepath);

	// Raw attribute lists (1-indexed in OBJ, we'll convert to 0-indexed)
	std::vector<point3> raw_pos;
	std::vector<vec3>   raw_norm;
	std::vector<double> raw_u, raw_v;

	// Each face vertex is a triple (pos_idx, uv_idx, norm_idx), -1 = absent
	struct FaceVertex { int p, t, n; };
	std::vector<std::array<FaceVertex,3>> faces;  // triangulated

	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;

		if (tok == "v") {
			double x, y, z;
			ss >> x >> y >> z;
			raw_pos.push_back(point3(x*scale + offset.x(),
									 y*scale + offset.y(),
									 z*scale + offset.z()));
		} else if (tok == "vn") {
			double nx, ny, nz;
			ss >> nx >> ny >> nz;
			raw_norm.push_back(unit_vector(vec3(nx, ny, nz)));
		} else if (tok == "vt") {
			double u, v;
			ss >> u >> v;
			raw_u.push_back(u);
			raw_v.push_back(v);
		} else if (tok == "f") {
			// Parse all face vertices, then fan-triangulate
			std::vector<FaceVertex> fverts;
			std::string fv_str;
			while (ss >> fv_str) {
				FaceVertex fv{ -1, -1, -1 };
				// Possible formats: p   p/t   p//n   p/t/n
				// sscanf is simpler than splitting for these patterns
				int pi, ti, ni;
				if (sscanf(fv_str.c_str(), "%d/%d/%d", &pi, &ti, &ni) == 3) {
					fv.p = pi - 1; fv.t = ti - 1; fv.n = ni - 1;
				} else if (sscanf(fv_str.c_str(), "%d//%d", &pi, &ni) == 2) {
					fv.p = pi - 1; fv.n = ni - 1;
				} else if (sscanf(fv_str.c_str(), "%d/%d", &pi, &ti) == 2) {
					fv.p = pi - 1; fv.t = ti - 1;
				} else if (sscanf(fv_str.c_str(), "%d", &pi) == 1) {
					fv.p = pi - 1;
				}
				if (fv.p >= 0) fverts.push_back(fv);
			}
			// Fan triangulate: v0, v1, v2; v0, v2, v3; ...
			for (size_t i = 1; i + 1 < fverts.size(); ++i) {
				faces.push_back({ fverts[0], fverts[i], fverts[i+1] });
			}
		}
		// All other tokens (o, g, mtllib, usemtl, s) are silently ignored
	}

	if (raw_pos.empty() || faces.empty())
		throw std::runtime_error("load_obj: no geometry in file: " + filepath);

	// ------------------------------------------------------------------
	// Optionally generate smooth per-vertex normals (area-weighted average
	// of adjacent face normals -- each face's contribution is the raw,
	// unnormalized cross product, so larger triangles naturally pull the
	// average more, matching the standard mesh-processing convention). Only
	// engages when the file itself has no vn data; if it does, that data is
	// authoritative and used as-is below.
	// ------------------------------------------------------------------
	std::vector<vec3> generated_norm;
	if (smooth_normals && raw_norm.empty()) {
		generated_norm.assign(raw_pos.size(), vec3(0,0,0));
		for (auto& tri : faces) {
			int i0 = tri[0].p, i1 = tri[1].p, i2 = tri[2].p;
			vec3 face_normal = cross(raw_pos[i1] - raw_pos[i0], raw_pos[i2] - raw_pos[i0]);
			generated_norm[i0] += face_normal;
			generated_norm[i1] += face_normal;
			generated_norm[i2] += face_normal;
		}
		for (auto& n : generated_norm)
			if (n.length_squared() > 1e-20) n = unit_vector(n);
	}

	// ------------------------------------------------------------------
	// Build triangle_mesh_data from face soup
	// Each face vertex gets its own entry in the flat arrays to keep
	// the indexing simple (allows different normals/UVs per face corner).
	// ------------------------------------------------------------------
	auto mesh_data = std::make_shared<triangle_mesh_data>();
	bool have_norms = !raw_norm.empty() || !generated_norm.empty();
	bool have_uvs   = !raw_u.empty();

	mesh_data->positions.reserve(faces.size() * 3);
	if (have_norms) mesh_data->normals.reserve(faces.size() * 3);
	if (have_uvs)   mesh_data->uvs.reserve(faces.size() * 6);
	mesh_data->indices.reserve(faces.size() * 3);

	int v_idx = 0;
	for (auto& tri : faces) {
		for (int c = 0; c < 3; ++c) {
			const FaceVertex& fv = tri[c];
			if (fv.p < 0 || fv.p >= (int)raw_pos.size())
				throw std::runtime_error("load_obj: invalid position index in " + filepath);
			mesh_data->positions.push_back(raw_pos[fv.p]);

			if (!generated_norm.empty())
				mesh_data->normals.push_back(generated_norm[fv.p]);
			else if (have_norms && fv.n >= 0 && fv.n < (int)raw_norm.size())
				mesh_data->normals.push_back(raw_norm[fv.n]);
			else if (have_norms)
				mesh_data->normals.push_back(vec3(0,1,0)); // fallback

			if (have_uvs && fv.t >= 0 && fv.t < (int)raw_u.size()) {
				mesh_data->uvs.push_back(raw_u[fv.t]);
				mesh_data->uvs.push_back(raw_v[fv.t]);
			} else if (have_uvs) {
				mesh_data->uvs.push_back(0.0);
				mesh_data->uvs.push_back(0.0);
			}

			mesh_data->indices.push_back(v_idx++);
		}
	}

	// ------------------------------------------------------------------
	// Build one triangle hittable per face, then wrap in a BVH
	// ------------------------------------------------------------------
	hittable_list tris;
	int n = mesh_data->num_triangles();
	for (int i = 0; i < n; ++i)
		tris.add(std::make_shared<triangle>(mesh_data, i, mat));

	return std::make_shared<bvh_node>(tris);
}


// ---------------------------------------------------------------------------
// triangle_mesh
// Convenience class: wraps load_obj result as a named hittable so scenes
// can store a typed pointer.  Delegates hit() and bounding_box() to the BVH.
// ---------------------------------------------------------------------------
class triangle_mesh : public hittable {
  public:
	triangle_mesh(const std::string& filepath,
				  std::shared_ptr<material> mat,
				  double scale = 1.0,
				  point3 offset = point3(0,0,0),
				  bool smooth_normals = false)
	{
		bvh = load_obj(filepath, mat, scale, offset, smooth_normals);
		bbox = bvh->bounding_box();
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		return bvh->hit(r, ray_t, rec);
	}

	aabb bounding_box() const override { return bbox; }

  private:
	std::shared_ptr<hittable> bvh;
	aabb bbox;
};


#endif
