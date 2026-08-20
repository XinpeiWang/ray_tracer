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
#include "normal_map_materials.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "../shared/mtl_parse.h"


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
// parse_mtl / mtl_specular_params / parse_mtl_specular / parse_mtl_textures /
// parse_mtl_ke_textures / parse_mtl_emission / parse_mtl_bump_textures /
// parse_mtl_alpha_textures / resolve_mtl_texture_path / mtl_has_real_
// transmission_filter / phong_to_roughness / is_grayscale_image
//
// Thin `color`-typed wrappers over src/shared/mtl_parse.h's renderer-
// agnostic parser (RGB triples, not `color`) - the actual .mtl text-parsing
// logic used to be independently hand-duplicated here and in gpu/optix/
// scene_builder.cpp (two copies of 7 near-identical functions each, one per
// .mtl property). Names/signatures/behavior are unchanged from before this
// consolidation, so load_obj_mtl() below (and every test exercising these
// functions by name) needs no changes.
// ---------------------------------------------------------------------------

inline color to_color(const mtl_parse::RGB& c) { return color(c.r, c.g, c.b); }

inline std::unordered_map<std::string, color> parse_mtl(const std::string& filepath) {
	std::unordered_map<std::string, color> result;
	for (const auto& [name, c] : mtl_parse::parse_kd(filepath)) result[name] = to_color(c);
	return result;
}

struct mtl_specular_params {
	color  ks = color(0, 0, 0);
	double ns = 0.0;
	int    illum = -1;
	double ni = -1.0;
	color  tf = color(1, 1, 1);
};

inline std::unordered_map<std::string, mtl_specular_params> parse_mtl_specular(const std::string& filepath) {
	std::unordered_map<std::string, mtl_specular_params> result;
	for (const auto& [name, sp] : mtl_parse::parse_specular(filepath)) {
		mtl_specular_params out;
		out.ks = to_color(sp.ks);
		out.ns = sp.ns;
		out.illum = sp.illum;
		out.ni = sp.ni;
		out.tf = to_color(sp.tf);
		result[name] = out;
	}
	return result;
}

// True when a material's Tf is meaningfully different from the .mtl spec's
// own neutral default (1,1,1) - see mtl_specular_params::tf's comment.
inline bool mtl_has_real_transmission_filter(const color& tf) {
	return mtl_parse::has_real_transmission_filter(mtl_parse::RGB{tf.x(), tf.y(), tf.z()});
}

// Standard Phong-exponent-to-roughness conversion.
inline double phong_to_roughness(double ns) { return mtl_parse::phong_to_roughness(ns); }

inline std::unordered_map<std::string, std::string> parse_mtl_textures(const std::string& filepath) {
	return mtl_parse::parse_map_kd(filepath);
}

inline std::unordered_map<std::string, std::string> parse_mtl_ke_textures(const std::string& filepath) {
	return mtl_parse::parse_map_ke(filepath);
}

inline std::unordered_map<std::string, color> parse_mtl_emission(const std::string& filepath) {
	std::unordered_map<std::string, color> result;
	for (const auto& [name, c] : mtl_parse::parse_ke(filepath)) result[name] = to_color(c);
	return result;
}

inline std::unordered_map<std::string, std::string> parse_mtl_bump_textures(const std::string& filepath) {
	return mtl_parse::parse_map_bump(filepath);
}

// Distinguishes a genuine grayscale height/displacement map from a
// tangent-space RGB normal map by sampling an 8x8 grid of pixels - see
// src/shared/mtl_parse.h's own file comment for why this stays CPU-side
// rather than moving into that shared header (it inspects `rtw_image`,
// a CPU-only representation; GPU's mirror, is_grayscale_texture_gpu() in
// scene_builder.cpp, does the same algorithm against its own device pixel
// buffer instead).
inline bool is_grayscale_image(const rtw_image& img) {
	int w = img.width(), h = img.height();
	if (w <= 0 || h <= 0) return true; // degenerate load; harmless default
	constexpr int kGrid = 8;
	int max_diff = 0;
	for (int sy = 0; sy < kGrid; ++sy) {
		int y = (sy * h) / kGrid;
		for (int sx = 0; sx < kGrid; ++sx) {
			int x = (sx * w) / kGrid;
			const unsigned char* p = img.pixel_data(x, y);
			int r = p[0], g = p[1], b = p[2];
			max_diff = std::max({max_diff, std::abs(r - g), std::abs(g - b), std::abs(r - b)});
		}
	}
	return max_diff <= 10;
}

inline std::unordered_map<std::string, std::string> parse_mtl_alpha_textures(const std::string& filepath) {
	return mtl_parse::parse_map_d(filepath);
}

inline std::string resolve_mtl_texture_path(const std::string& relative_path, const std::string& texture_dir) {
	return mtl_parse::resolve_texture_path(relative_path, texture_dir);
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
//
// out_lights: when non-null, every triangle whose .mtl material has a real
// Ke gets a diffuse_light(Ke) instead of its Kd/map_Kd material, and is also
// appended here -- the same "emissive geometry doubles as an NEE light"
// pattern pbrt_cpu_builder.h already uses for .pbrt area lights. Left null
// (the default) for callers that don't need it; empty for a mesh whose .mtl
// has no real Ke data regardless (Sponza/Bistro/Rungholt today).
// ---------------------------------------------------------------------------
inline std::shared_ptr<hittable> load_obj_mtl(
		const std::string& filepath,
		std::shared_ptr<material> fallback_mat,
		double scale = 1.0,
		point3 offset = point3(0,0,0),
		bool smooth_normals = false,
		const std::string& texture_dir = "",
		hittable_list* out_lights = nullptr)
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
	std::unordered_map<std::string, color> mtl_emission;
	std::unordered_map<std::string, mtl_specular_params> mtl_specular;
	std::unordered_map<std::string, std::string> mtl_bump_textures;
	std::unordered_map<std::string, std::string> mtl_ke_textures;
	std::string mtl_path_used = mtllib_name;
	if (!mtllib_name.empty())
		mtl_colors = parse_mtl(mtllib_name);
	if (mtl_colors.empty()) {
		auto dot = filepath.find_last_of('.');
		mtl_path_used = (dot == std::string::npos ? filepath : filepath.substr(0, dot)) + ".mtl";
		mtl_colors = parse_mtl(mtl_path_used);
	}
	std::unordered_map<std::string, std::string> mtl_alpha_textures;
	if (!texture_dir.empty() && !mtl_path_used.empty()) {
		mtl_textures = parse_mtl_textures(mtl_path_used);
		mtl_bump_textures = parse_mtl_bump_textures(mtl_path_used);
		mtl_alpha_textures = parse_mtl_alpha_textures(mtl_path_used);
	}
	if (out_lights && !mtl_path_used.empty()) {
		mtl_emission = parse_mtl_emission(mtl_path_used);
		if (!texture_dir.empty())
			mtl_ke_textures = parse_mtl_ke_textures(mtl_path_used);
	}
	if (!mtl_path_used.empty())
		mtl_specular = parse_mtl_specular(mtl_path_used);

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
	// across every face using it): a diffuse_light(Ke) when the material
	// is genuinely emissive (out_lights requested and a non-degenerate Ke
	// line present); else, by illum, a dielectric (illum 7 -- "reflection
	// and refraction on") or a metal(Kd, roughness-from-Ns) (illum 2 --
	// "highlight on" -- with a real, non-boilerplate Ks); else a real
	// image_texture-backed lambertian when texture_dir is set and the
	// material's map_Kd image loads successfully, else a flat lambertian
	// (Kd), else fallback_mat when the name is empty/unknown or has none
	// of the above -- then, whatever that base material is, a map_Bump
	// texture (when present and texture_dir is set) wraps it in either
	// bump_map_material or normal_map_material, chosen by the loaded
	// image's actual pixel content via is_grayscale_image() (see that
	// function's own comment for why the .mtl keyword alone can't say
	// which one a given map_Bump reference really is). Independently, a
	// map_d texture (when present and texture_dir is set) becomes an
	// alpha-cutout mask passed to each face's triangle constructor - see
	// triangle::hit()'s own alpha test.
	// ------------------------------------------------------------------
	constexpr double kMeaningfulKsComponent = 0.02;
	std::unordered_map<std::string, std::shared_ptr<material>> mat_cache;
	std::unordered_map<std::string, std::shared_ptr<texture>> alpha_mask_cache;
	std::unordered_set<std::string> emissive_mats;
	// Subset of emissive_mats that should also be registered as explicit NEE
	// lights (added to out_lights below). Deliberately excludes map_Ke
	// textured-emissive materials: Gallery's own single material covers its
	// entire ~1M-triangle mesh (walls, floor, columns - not just the
	// painting), and most of that texture's surface is black/near-zero: a
	// scene-build-time attempt to register every one of those triangles in
	// `lights` made hittable_pdf's per-sample light-list scan effectively
	// O(n) over ~1M candidates and never finished a render. Skipping NEE
	// registration for this case is not a correctness compromise - camera.h's
	// own emission-handling code (search "Emission -- full Le on camera/
	// specular hits") already gives an emitter that ISN'T in `lights` a MIS
	// weight of 1 whenever a BSDF-sampled/camera ray hits it directly (light
	// pdf evaluates to 0 for an unregistered object, and the power heuristic
	// gives full weight to the only strategy that could have produced the
	// sample) - unbiased, just not explicitly importance-sampled. Since a
	// painting is normally seen because the camera is pointed at it, not
	// because another surface needs it as a light source, this trades away
	// only a variance-reduction technique this scene doesn't need, not
	// correctness. A real emissive material with a small, bounded triangle
	// count (Fireplace Room/Salle de Bain's actual light fixtures) has no
	// such performance cliff and keeps full NEE registration below.
	std::unordered_set<std::string> nee_light_mats;
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
				auto ke_it = mtl_emission.find(name);
				if (ke_it != mtl_emission.end()) {
					resolved = std::make_shared<diffuse_light>(ke_it->second);
					emissive_mats.insert(name);
					nee_light_mats.insert(name);
				}
				if (!resolved) {
					auto ke_tex_it = mtl_ke_textures.find(name);
					if (!texture_dir.empty() && ke_tex_it != mtl_ke_textures.end()) {
						// map_Ke with no scalar Ke line (Gallery's own case) -
						// same decode-once-then-move pattern as the map_Kd
						// textured-lambertian branch below, but into a
						// texture-backed diffuse_light instead: diffuse_light
						// already supports a shared_ptr<texture> emission
						// source (see material_simple.h), so this needed no
						// material-class changes, only the parsing/dispatch
						// wiring here. Deliberately NOT added to
						// nee_light_mats - see that set's own comment.
						std::string ke_img_path = resolve_mtl_texture_path(ke_tex_it->second, found_prefix + texture_dir);
						rtw_image probe(ke_img_path.c_str());
						if (probe.height() > 0) {
							resolved = std::make_shared<diffuse_light>(std::make_shared<mipmap_texture>(std::move(probe)));
							emissive_mats.insert(name);
						}
					}
				}
				if (!resolved) {
					auto spec_it = mtl_specular.find(name);
					if (spec_it != mtl_specular.end()) {
						const mtl_specular_params& sp = spec_it->second;
						if (sp.illum == 7) {
							// Tf tints the transmitted contribution only (see
							// dielectric's own Tf-tint constructor comment) -
							// a no-op (white) for every illum-7 material that
							// doesn't set a real Tf, e.g. Bistro's
							// MASTER_Glass_Clean.
							resolved = std::make_shared<dielectric>(sp.ni > 0.0 ? sp.ni : 1.5, sp.tf);
						} else if ((sp.illum == 4 || sp.illum == 6) && mtl_has_real_transmission_filter(sp.tf) &&
								   mtl_textures.find(name) == mtl_textures.end()) {
							// illum 4/6 ("transparency, glass on") only when a real,
							// non-boilerplate Tf backs it up (see
							// mtl_has_real_transmission_filter's comment) - some
							// exporters (Breakfast Room's Blender export) write
							// illum 4 as a blanket default on every material
							// regardless of type, which would otherwise turn its
							// Chrome/Ceramic/matte-black-paint materials into glass
							// too. Also requires no map_Kd texture: a textured
							// material (e.g. Artwork, Kd 0,0,0 like real glass but
							// meant to show its picture) is clearly not glass either.
							// Tf tints the transmitted contribution (Sibenik's
							// colored stained-glass windows) - see dielectric's
							// own Tf-tint constructor comment for why this is a
							// flat per-surface tint rather than the volumetric
							// MaterialType::DielectricMedium/constant_medium
							// path this codebase already has for thick glass.
							resolved = std::make_shared<dielectric>(sp.ni > 0.0 ? sp.ni : 1.5, sp.tf);
						} else if ((sp.illum == 2 || sp.illum == 3) && std::max({sp.ks.x(), sp.ks.y(), sp.ks.z()}) > kMeaningfulKsComponent) {
							// illum 3 ("diffuse+specular, reflection on") gets the
							// same glossy-metal treatment as illum 2 - Salle de
							// Bain's "Mirror" material (Kd 0,0,0, Ks 0.99, illum 3)
							// would otherwise render as a plain black diffuse
							// rectangle instead of anything mirror-like.
							auto color_it = mtl_colors.find(name);
							color kd = (color_it != mtl_colors.end()) ? color_it->second : color(1, 1, 1);
							resolved = std::make_shared<metal>(kd, phong_to_roughness(sp.ns));
						}
					}
				}
				if (!resolved) {
					auto tex_it = mtl_textures.find(name);
					if (!texture_dir.empty() && tex_it != mtl_textures.end()) {
						std::string img_path = resolve_mtl_texture_path(tex_it->second, found_prefix + texture_dir);
						// Decode once: probe the load here, then move the
						// already-decoded pixels into image_texture instead of
						// having its own constructor decode the same file
						// again (image_texture(rtw_image&&), texture.h).
						rtw_image probe(img_path.c_str());
						if (probe.height() > 0)
							resolved = std::make_shared<lambertian>(std::make_shared<mipmap_texture>(std::move(probe)));
					}
				}
				if (!resolved) {
					auto color_it = mtl_colors.find(name);
					if (color_it != mtl_colors.end())
						resolved = std::make_shared<lambertian>(color_it->second);
				}
				// map_Bump wraps whatever base material was just resolved
				// above (lambertian, metal, dielectric, ...) -- skipped for
				// emissive materials, where perturbing the emitted-light
				// normal has no meaningful effect.
				if (resolved && !texture_dir.empty() && !emissive_mats.count(name)) {
					auto bump_it = mtl_bump_textures.find(name);
					if (bump_it != mtl_bump_textures.end()) {
						std::string bump_path = resolve_mtl_texture_path(bump_it->second, found_prefix + texture_dir);
						rtw_image bump_probe(bump_path.c_str());
						if (bump_probe.height() > 0) {
							bool grayscale = is_grayscale_image(bump_probe);
							auto bump_tex = std::make_shared<image_texture>(std::move(bump_probe));
							resolved = grayscale
								? std::static_pointer_cast<material>(std::make_shared<bump_map_material>(bump_tex, resolved))
								: std::static_pointer_cast<material>(std::make_shared<normal_map_material>(bump_tex, resolved));
						}
					}
				}
				tri_mat = resolved ? resolved : fallback_mat;
				mat_cache[name] = tri_mat;

				// map_d alpha-cutout mask: independent of which branch
				// above produced tri_mat (unlike map_Bump, this doesn't
				// wrap the material -- see triangle::hit()'s own alpha
				// test). Resolved and cached once per unique name here,
				// same as tri_mat itself, so a mask shared by thousands of
				// foliage triangles is decoded once, not per-triangle.
				if (!texture_dir.empty()) {
					auto alpha_it = mtl_alpha_textures.find(name);
					if (alpha_it != mtl_alpha_textures.end()) {
						std::string alpha_path = resolve_mtl_texture_path(alpha_it->second, found_prefix + texture_dir);
						rtw_image alpha_probe(alpha_path.c_str());
						if (alpha_probe.height() > 0)
							alpha_mask_cache[name] = std::make_shared<image_texture>(std::move(alpha_probe));
					}
				}
			}
		}
		std::shared_ptr<texture> tri_alpha;
		if (!name.empty()) {
			auto a_it = alpha_mask_cache.find(name);
			if (a_it != alpha_mask_cache.end()) tri_alpha = a_it->second;
		}
		auto tri = std::make_shared<triangle>(mesh_data, i, tri_mat, tri_alpha);
		tris.add(tri);
		if (out_lights && !name.empty() && nee_light_mats.count(name))
			out_lights->add(tri);
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
		bvh = load_obj_mtl(filepath, fallback_mat, scale, offset, smooth_normals, texture_dir, &emissive_tris);
		bbox = bvh->bounding_box();
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		return bvh->hit(r, ray_t, rec);
	}

	aabb bounding_box() const override { return bbox; }

	// Triangles whose .mtl material has a real Ke, for NEE sampling as area
	// lights (see build_sponza()/build_bistro_exterior()/build_rungholt()'s
	// callers in scenes_advanced.h). Empty for any mesh whose .mtl has no
	// non-degenerate Ke data, which is every large scene as of this writing.
	const hittable_list& lights() const { return emissive_tris; }

  private:
	std::shared_ptr<hittable> bvh;
	aabb bbox;
	hittable_list emissive_tris;
};


#endif
