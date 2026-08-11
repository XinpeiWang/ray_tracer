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
#include "material_simple.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <stdexcept>
#include <cstdio>
#include <utility>


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
			// OBJ indices may be negative ("relative"): -1 refers to the
			// most-recently-defined v/vt/vn, -2 the one before that, etc.,
			// resolved against however many have been parsed so far in the
			// file (not the eventual total) - e.g. rungholt.obj (a McGuire
			// Computer Graphics Archive scene) uses this convention
			// throughout. A positive index resolves the same way regardless
			// (1-based from the start of the file).
			auto resolve_index = [](int raw, size_t count_so_far) -> int {
				return raw > 0 ? raw - 1 : static_cast<int>(count_so_far) + raw;
			};
			while (ss >> fv_str) {
				FaceVertex fv{ -1, -1, -1 };
				// Possible formats: p   p/t   p//n   p/t/n
				// sscanf is simpler than splitting for these patterns
				int pi, ti, ni;
				if (sscanf(fv_str.c_str(), "%d/%d/%d", &pi, &ti, &ni) == 3) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.t = resolve_index(ti, raw_u.size());
					fv.n = resolve_index(ni, raw_norm.size());
				} else if (sscanf(fv_str.c_str(), "%d//%d", &pi, &ni) == 2) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.n = resolve_index(ni, raw_norm.size());
				} else if (sscanf(fv_str.c_str(), "%d/%d", &pi, &ti) == 2) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.t = resolve_index(ti, raw_u.size());
				} else if (sscanf(fv_str.c_str(), "%d", &pi) == 1) {
					fv.p = resolve_index(pi, raw_pos.size());
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


// ---------------------------------------------------------------------------
// parse_mtl
// Minimal Wavefront .mtl parser: maps material name -> diffuse (Kd) color.
// Only Kd is read; textures (map_Kd) and other physical parameters are
// ignored, consistent with this renderer's no-texture mesh convention.
// Returns an empty map (never throws) if the file can't be found, so
// callers can fall back to a flat material.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, color> parse_mtl(const std::string& filepath) {
	std::unordered_map<std::string, color> result;

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
	if (!file.is_open()) return result;

	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if (tok == "Kd" && !current.empty()) {
			double r, g, b;
			ss >> r >> g >> b;
			result[current] = color(r, g, b);
		}
	}
	return result;
}


// ---------------------------------------------------------------------------
// parse_mtl_textures
// Companion to parse_mtl(): maps material name -> its map_Kd (diffuse
// texture) path, exactly as written in the .mtl (backslashes, "..", etc.
// un-normalized -- see resolve_mtl_texture_path() for that). Only materials
// with a map_Kd line appear in the result. Kept separate from parse_mtl()
// rather than folded into its return value so parse_mtl()'s existing
// color-only signature (and its test coverage) stays unchanged.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, std::string> parse_mtl_textures(const std::string& filepath) {
	std::unordered_map<std::string, std::string> result;

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
	if (!file.is_open()) return result;

	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if (tok == "map_Kd" && !current.empty()) {
			// Take the rest of the line (minus surrounding whitespace) as
			// the filename rather than a single ss >> token: real texture
			// filenames in these archives can contain literal spaces (e.g.
			// Bistro's "Metal_ RollDoor_01/Metal_ RollDoor_01_diff.png"),
			// which a single >> would silently truncate at. Real-world
			// .mtl "map_Kd" lines can also carry options (-o, -s, -bm, ...)
			// before the filename; none of the three scenes this was built
			// for (Sponza/Bistro/Rungholt) use them, so this doesn't handle
			// that general case.
			std::string path;
			std::getline(ss, path);
			size_t start = path.find_first_not_of(" \t");
			if (start != std::string::npos) {
				size_t end = path.find_last_not_of(" \t\r");
				result[current] = path.substr(start, end - start + 1);
			}
		}
	}
	return result;
}


// ---------------------------------------------------------------------------
// resolve_mtl_texture_path
// Rewrites a .mtl-relative texture path (as read by parse_mtl_textures, e.g.
// "textures\foo.png" or "..\BuildingTextures\bar.png") into a path under
// texture_dir. Backslashes are normalized to forward slashes and any
// leading "../"/"./" segments are stripped rather than resolved literally --
// callers relocate the actual texture files under texture_dir instead of
// mirroring the original archive's exact directory nesting relative to the
// .mtl file, so only the meaningful tail (e.g. "BuildingTextures/bar.png")
// matters.
// ---------------------------------------------------------------------------
inline std::string resolve_mtl_texture_path(const std::string& relative_path, const std::string& texture_dir) {
	std::string p = relative_path;
	for (auto& c : p) if (c == '\\') c = '/';

	size_t pos = 0;
	while (true) {
		if (p.compare(pos, 3, "../") == 0) pos += 3;
		else if (p.compare(pos, 2, "./") == 0) pos += 2;
		else break;
	}
	return texture_dir + "/" + p.substr(pos);
}


// ---------------------------------------------------------------------------
// load_obj_mtl
// Like load_obj(), but assigns each triangle its own material by tracking
// the OBJ's mtllib/usemtl directives and looking up each name's Kd color in
// the companion .mtl file (via parse_mtl above), instead of applying one
// flat material to the whole mesh. Falls back to fallback_mat for faces
// with no usemtl, a name absent from the .mtl (or an unreadable/missing
// .mtl entirely), so a bad/absent .mtl degrades to load_obj()'s old flat
// look rather than failing the load.
//
// texture_dir: when non-empty, materials with a map_Kd entry get a real
// image_texture-backed lambertian (sampled via this mesh's own "vt" UVs)
// instead of a flat Kd color, resolved via resolve_mtl_texture_path()
// against found_prefix + texture_dir -- a path *relative to the models/
// folder itself* (e.g. "sponza_textures", not "models/sponza_textures"),
// so it inherits the same climb-to-find-models/ prefix that successfully
// located filepath, regardless of the render's current working directory
// (RayTracer_Package/, bin/Release/, a test binary's own cwd, ...). Left
// empty (the default), this behaves exactly like the Kd-only version --
// Rungholt (whose .mtl carries no real map_Kd textures worth fetching)
// keeps calling load_obj_mtl() without this arg.
//
// A standalone duplicate of load_obj()'s parse loop rather than a shared
// helper: load_obj() is used by ~50 existing single-material mesh scenes
// and must not change behavior, and the two loaders' inner loops diverge
// enough (face material tracking, per-face material assignment instead of
// one shared_ptr for the whole mesh) that factoring out the overlap would
// cost more clarity than it saves.
// ---------------------------------------------------------------------------
inline std::shared_ptr<hittable> load_obj_mtl(
		const std::string& filepath,
		std::shared_ptr<material> fallback_mat,
		double scale = 1.0,
		point3 offset = point3(0,0,0),
		bool smooth_normals = false,
		const std::string& texture_dir = "")
{
	// Tracks which search prefix (if any) actually located filepath, so
	// texture_dir (a path relative to models/, e.g. "sponza_textures") can
	// be resolved with the same number of ".." climbs the .obj itself
	// needed - texture_dir alone doesn't know how deep under models/ the
	// current working directory actually is (RayTracer_Package/, bin/
	// Release/, a test binary's own cwd, ...).
	std::string found_prefix;
	std::ifstream file(filepath);
	if (!file.is_open()) {
		static const char* kSearchPrefixes[] = {
			"models/", "../models/", "../../models/",
			"../../../models/", "../../../../models/", "../../../../../models/"
		};
		for (const char* prefix : kSearchPrefixes) {
			file.clear();
			file.open(prefix + filepath);
			if (file.is_open()) { found_prefix = prefix; break; }
		}
	}
	if (!file.is_open())
		throw std::runtime_error("load_obj_mtl: cannot open file: " + filepath);

	std::vector<point3> raw_pos;
	std::vector<vec3>   raw_norm;
	std::vector<double> raw_u, raw_v;

	struct FaceVertex { int p, t, n; };
	struct Face { std::array<FaceVertex,3> v; std::string mtl; };
	std::vector<Face> faces;

	std::string mtllib_name;
	std::string current_mtl;

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
		} else if (tok == "mtllib") {
			ss >> mtllib_name;
		} else if (tok == "usemtl") {
			ss >> current_mtl;
		} else if (tok == "f") {
			std::vector<FaceVertex> fverts;
			std::string fv_str;
			auto resolve_index = [](int raw, size_t count_so_far) -> int {
				return raw > 0 ? raw - 1 : static_cast<int>(count_so_far) + raw;
			};
			while (ss >> fv_str) {
				FaceVertex fv{ -1, -1, -1 };
				int pi, ti, ni;
				if (sscanf(fv_str.c_str(), "%d/%d/%d", &pi, &ti, &ni) == 3) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.t = resolve_index(ti, raw_u.size());
					fv.n = resolve_index(ni, raw_norm.size());
				} else if (sscanf(fv_str.c_str(), "%d//%d", &pi, &ni) == 2) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.n = resolve_index(ni, raw_norm.size());
				} else if (sscanf(fv_str.c_str(), "%d/%d", &pi, &ti) == 2) {
					fv.p = resolve_index(pi, raw_pos.size());
					fv.t = resolve_index(ti, raw_u.size());
				} else if (sscanf(fv_str.c_str(), "%d", &pi) == 1) {
					fv.p = resolve_index(pi, raw_pos.size());
				}
				if (fv.p >= 0) fverts.push_back(fv);
			}
			for (size_t i = 1; i + 1 < fverts.size(); ++i) {
				faces.push_back({ { fverts[0], fverts[i], fverts[i+1] }, current_mtl });
			}
		}
		// All other tokens (o, g, #, s) are silently ignored, same as load_obj()
	}

	if (raw_pos.empty() || faces.empty())
		throw std::runtime_error("load_obj_mtl: no geometry in file: " + filepath);

	// Locate the companion .mtl: prefer the file's own mtllib directive,
	// falling back to "<same name as the .obj>.mtl" if that's missing,
	// unreadable, or defines no Kd colors at all.
	std::unordered_map<std::string, color> mtl_colors;
	std::unordered_map<std::string, std::string> mtl_textures;
	std::string mtl_path_used = mtllib_name;
	if (!mtllib_name.empty())
		mtl_colors = parse_mtl(mtllib_name);
	if (mtl_colors.empty()) {
		auto dot = filepath.find_last_of('.');
		mtl_path_used = (dot == std::string::npos ? filepath : filepath.substr(0, dot)) + ".mtl";
		mtl_colors = parse_mtl(mtl_path_used);
	}
	if (!texture_dir.empty() && !mtl_path_used.empty())
		mtl_textures = parse_mtl_textures(mtl_path_used);

	std::vector<vec3> generated_norm;
	if (smooth_normals && raw_norm.empty()) {
		generated_norm.assign(raw_pos.size(), vec3(0,0,0));
		for (auto& f : faces) {
			int i0 = f.v[0].p, i1 = f.v[1].p, i2 = f.v[2].p;
			vec3 face_normal = cross(raw_pos[i1] - raw_pos[i0], raw_pos[i2] - raw_pos[i0]);
			generated_norm[i0] += face_normal;
			generated_norm[i1] += face_normal;
			generated_norm[i2] += face_normal;
		}
		for (auto& n : generated_norm)
			if (n.length_squared() > 1e-20) n = unit_vector(n);
	}

	auto mesh_data = std::make_shared<triangle_mesh_data>();
	bool have_norms = !raw_norm.empty() || !generated_norm.empty();
	bool have_uvs   = !raw_u.empty();

	mesh_data->positions.reserve(faces.size() * 3);
	if (have_norms) mesh_data->normals.reserve(faces.size() * 3);
	if (have_uvs)   mesh_data->uvs.reserve(faces.size() * 6);
	mesh_data->indices.reserve(faces.size() * 3);

	int v_idx = 0;
	for (auto& f : faces) {
		for (int c = 0; c < 3; ++c) {
			const FaceVertex& fv = f.v[c];
			if (fv.p < 0 || fv.p >= (int)raw_pos.size())
				throw std::runtime_error("load_obj_mtl: invalid position index in " + filepath);
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
	// Build one triangle hittable per face, resolving each face's .mtl
	// name to a cached material (one shared_ptr per unique name, shared
	// across every face using it): a real image_texture-backed lambertian
	// when texture_dir is set and the material's map_Kd image loads
	// successfully, else a flat lambertian(Kd), else fallback_mat when the
	// name is empty/unknown or has neither.
	// ------------------------------------------------------------------
	std::unordered_map<std::string, std::shared_ptr<material>> mat_cache;
	hittable_list tris;
	int n = mesh_data->num_triangles();
	for (int i = 0; i < n; ++i) {
		const std::string& name = faces[i].mtl;
		std::shared_ptr<material> tri_mat = fallback_mat;
		if (!name.empty()) {
			auto cached = mat_cache.find(name);
			if (cached != mat_cache.end()) {
				tri_mat = cached->second;
			} else {
				std::shared_ptr<material> resolved;
				auto tex_it = mtl_textures.find(name);
				if (!texture_dir.empty() && tex_it != mtl_textures.end()) {
					std::string img_path = resolve_mtl_texture_path(tex_it->second, found_prefix + texture_dir);
					// Decode once: probe the load here, then move the
					// already-decoded pixels into image_texture instead of
					// having its own constructor decode the same file
					// again (image_texture(rtw_image&&), texture.h).
					rtw_image probe(img_path.c_str());
					if (probe.height() > 0)
						resolved = std::make_shared<lambertian>(std::make_shared<image_texture>(std::move(probe)));
				}
				if (!resolved) {
					auto color_it = mtl_colors.find(name);
					if (color_it != mtl_colors.end())
						resolved = std::make_shared<lambertian>(color_it->second);
				}
				tri_mat = resolved ? resolved : fallback_mat;
				mat_cache[name] = tri_mat;
			}
		}
		tris.add(std::make_shared<triangle>(mesh_data, i, tri_mat));
	}

	return std::make_shared<bvh_node>(tris);
}


// ---------------------------------------------------------------------------
// triangle_mesh_mtl
// Like triangle_mesh, but backed by load_obj_mtl() for real per-face .mtl
// colors (and, when texture_dir is given, real map_Kd image textures)
// instead of one flat material for the whole mesh.
// ---------------------------------------------------------------------------
class triangle_mesh_mtl : public hittable {
  public:
	triangle_mesh_mtl(const std::string& filepath,
					   std::shared_ptr<material> fallback_mat,
					   double scale = 1.0,
					   point3 offset = point3(0,0,0),
					   bool smooth_normals = false,
					   const std::string& texture_dir = "")
	{
		bvh = load_obj_mtl(filepath, fallback_mat, scale, offset, smooth_normals, texture_dir);
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
