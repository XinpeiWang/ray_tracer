#pragma once
// ---------------------------------------------------------------------------
// mtl_parse.h -- Wavefront .mtl parser, shared between the CPU OBJ loader
// (src/TheRestOfYourLife/mesh.h's load_obj_mtl()) and the GPU OBJ loader
// (gpu/optix/scene_builder.cpp's load_obj_triangles_mtl_gpu()).
//
// Both loaders used to carry their own independently hand-ported copy of
// this exact parsing logic (7 near-identical functions each, one per .mtl
// property), which meant every fix or format quirk found in one had to be
// separately remembered and re-applied to the other - the same
// "two-implementations-of-one-thing" risk this codebase has hit repeatedly
// elsewhere (GPU-recursive vs GPU-wavefront backend drift being the other
// major instance of it). This header is renderer-agnostic on purpose (plain
// structs/std::string/std::unordered_map, no `color`/`float3`/rtw_image
// dependency) precisely so both loaders can call it directly instead of
// each maintaining their own copy - the same reasoning src/shared/
// pbrt_scene.h/pbrt_flatten.h already established for the pbrt loader.
//
// NOT shared: the actual OBJ face-parsing loops (load_obj_mtl()/
// load_obj_triangles_mtl_gpu() themselves) or grayscale-vs-normal-map pixel
// detection (is_grayscale_image() / is_grayscale_texture_gpu()) - those
// genuinely operate on different per-backend data representations (this
// codebase's `vec3`/`hittable` triangle mesh vs `SceneData`'s flat device
// buffers; `rtw_image` vs an already-uploaded device pixel buffer) and
// aren't the "same text-parsing logic duplicated" problem this header
// solves.
// ---------------------------------------------------------------------------

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace mtl_parse {

// Plain RGB triple - callers convert to their own renderer's color/float3
// type at the call site (this header has no renderer-type dependency).
struct RGB {
	double r = 0.0, g = 0.0, b = 0.0;
};

// Classic Blinn-Phong specular parameters (Ks, Ns, illum, Ni, Tf), read
// together in one pass since dispatching a material to lambertian/metal/
// dielectric needs all of them jointly. illum/ni default to -1 ("not
// specified in the file") rather than 0/1.5, so callers can distinguish
// "this .mtl declares illum 0" from "this .mtl says nothing about illum at
// all". tf defaults to (1,1,1) ("no filtering") per the .mtl spec's own
// default.
struct SpecularParams {
	RGB    ks;
	double ns = 0.0;
	int    illum = -1;
	double ni = -1.0;
	RGB    tf{1.0, 1.0, 1.0};
};

// True when a material's Tf is meaningfully different from the .mtl spec's
// own neutral default (1,1,1) - see SpecularParams::tf's comment. A handful
// of real glass materials (e.g. Sibenik's stained-glass windows) write a
// genuinely non-neutral Tf, used as the "this material really is glass"
// signal rather than illum alone (illum 4 is also written, as an unrelated
// exporter default, on every material in some Blender-exported .mtl files
// regardless of type).
inline bool has_real_transmission_filter(const RGB& tf) {
	constexpr double kEpsilon = 0.05;
	return std::fabs(tf.r - 1.0) > kEpsilon ||
	       std::fabs(tf.g - 1.0) > kEpsilon ||
	       std::fabs(tf.b - 1.0) > kEpsilon;
}

// Standard Phong-exponent-to-roughness conversion, mapping a classic OBJ/
// .mtl Ns (specular exponent, typically 1-1000) onto a physically-based
// roughness in (0, 1]. At Ns=0 (the boilerplate default most exporters
// write when they don't otherwise populate it) this returns 1.0 (fully
// rough), which every metal material constructor in this codebase clamps
// to anyway.
inline double phong_to_roughness(double ns) {
	return std::sqrt(2.0 / (ns + 2.0));
}

// Opens filepath directly, or under any of a fixed set of models/-relative
// search prefixes - matches how OBJ scene assets are actually laid out on
// disk relative to wherever the renderer's current working directory
// happens to be (RayTracer_Package/, bin/Release/, a test binary's own cwd,
// ...). out_found_prefix (optional) receives which prefix actually worked,
// empty if the direct path opened without one - callers that also need to
// resolve texture paths relative to the same models/ climb (see
// resolve_texture_path()) pass this through.
inline std::ifstream open_mtl_file(const std::string& filepath, std::string* out_found_prefix = nullptr) {
	if (out_found_prefix) out_found_prefix->clear();
	std::ifstream file(filepath);
	if (file.is_open()) return file;
	static const char* kSearchPrefixes[] = {
		"models/", "../models/", "../../models/",
		"../../../models/", "../../../../models/", "../../../../../models/"
	};
	for (const char* prefix : kSearchPrefixes) {
		file.clear();
		file.open(prefix + filepath);
		if (file.is_open()) {
			if (out_found_prefix) *out_found_prefix = prefix;
			break;
		}
	}
	return file;
}

// Real texture filenames can contain literal spaces (e.g. Bistro's own
// "Metal_ RollDoor_01/Metal_ RollDoor_01_diff.png"), so this takes the rest
// of the line rather than a single `>>` token. Real-world map_Kd/map_Ke
// lines can also carry leading options (-o, -s, -bm, -mm, ...) before the
// filename (e.g. Gallery's own .mtl: "map_Kd -bm 0.7 gallery.jpg") - skips
// any leading "-option <numeric args...>" groups (detected by "does this
// token parse fully as a number", not a hardcoded arg count per option)
// before taking the remainder as the filename. Returns empty if the line
// has nothing left after option-skipping.
inline std::string extract_texture_path(std::istringstream& ss) {
	std::string path;
	std::getline(ss, path);
	size_t pos = path.find_first_not_of(" \t");
	while (pos != std::string::npos && path[pos] == '-') {
		size_t tokEnd = path.find_first_of(" \t", pos);
		pos = (tokEnd == std::string::npos) ? std::string::npos
		                                     : path.find_first_not_of(" \t", tokEnd);
		while (pos != std::string::npos) {
			size_t argEnd = path.find_first_of(" \t", pos);
			std::string argTok = path.substr(pos, argEnd == std::string::npos ? std::string::npos : argEnd - pos);
			char* endp = nullptr;
			std::strtod(argTok.c_str(), &endp);
			if (endp == argTok.c_str() || *endp != '\0') break;  // not purely numeric
			pos = (argEnd == std::string::npos) ? std::string::npos
			                                     : path.find_first_not_of(" \t", argEnd);
		}
	}
	if (pos == std::string::npos) return std::string();
	size_t end = path.find_last_not_of(" \t\r");
	return path.substr(pos, end - pos + 1);
}

// Bare filename lines (map_Bump/map_d/...) - no leading-option skip, just
// trims surrounding whitespace.
inline std::string extract_plain_path(std::istringstream& ss) {
	std::string path;
	std::getline(ss, path);
	size_t start = path.find_first_not_of(" \t");
	if (start == std::string::npos) return std::string();
	size_t end = path.find_last_not_of(" \t\r");
	return path.substr(start, end - start + 1);
}

// material name -> Kd (diffuse) color.
inline std::unordered_map<std::string, RGB> parse_kd(const std::string& filepath) {
	std::unordered_map<std::string, RGB> result;
	std::ifstream file = open_mtl_file(filepath);
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
			RGB c;
			ss >> c.r >> c.g >> c.b;
			result[current] = c;
		}
	}
	return result;
}

// material name -> its classic Blinn-Phong specular parameters.
inline std::unordered_map<std::string, SpecularParams> parse_specular(const std::string& filepath) {
	std::unordered_map<std::string, SpecularParams> result;
	std::ifstream file = open_mtl_file(filepath);
	if (!file.is_open()) return result;
	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if (tok == "Ks" && !current.empty()) {
			RGB c; ss >> c.r >> c.g >> c.b; result[current].ks = c;
		} else if (tok == "Ns" && !current.empty()) {
			ss >> result[current].ns;
		} else if (tok == "illum" && !current.empty()) {
			ss >> result[current].illum;
		} else if (tok == "Ni" && !current.empty()) {
			ss >> result[current].ni;
		} else if (tok == "Tf" && !current.empty()) {
			RGB c; ss >> c.r >> c.g >> c.b; result[current].tf = c;
		}
	}
	return result;
}

// material name -> map_Kd (diffuse texture) path, exactly as written in the
// .mtl (backslashes, "..", etc. un-normalized - see resolve_texture_path()
// for that). Only materials with a map_Kd line appear in the result.
inline std::unordered_map<std::string, std::string> parse_map_kd(const std::string& filepath) {
	std::unordered_map<std::string, std::string> result;
	std::ifstream file = open_mtl_file(filepath);
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
			std::string p = extract_texture_path(ss);
			if (!p.empty()) result[current] = p;
		}
	}
	return result;
}

// material name -> map_Ke (emissive texture) path. Distinct from parse_ke()
// below, which only reads the scalar "Ke r g b" line - a material can have
// a real map_Ke with no scalar Ke at all (Gallery's own .mtl: "map_Ke -bm
// 0.3 gallery.jpg", no plain "Ke" line anywhere), which parse_ke() alone
// would miss entirely.
inline std::unordered_map<std::string, std::string> parse_map_ke(const std::string& filepath) {
	std::unordered_map<std::string, std::string> result;
	std::ifstream file = open_mtl_file(filepath);
	if (!file.is_open()) return result;
	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if (tok == "map_Ke" && !current.empty()) {
			std::string p = extract_texture_path(ss);
			if (!p.empty()) result[current] = p;
		}
	}
	return result;
}

// material name -> Ke (emission) color. An explicit "Ke 0 0 0" (the
// boilerplate default many exporters always write) is treated the same as
// no Ke line at all - only materials with a non-degenerate Ke appear in the
// result, so callers can use "is this name present" directly as "is this
// material a light", without re-deriving it from a returned all-black color.
inline std::unordered_map<std::string, RGB> parse_ke(const std::string& filepath) {
	std::unordered_map<std::string, RGB> result;
	std::ifstream file = open_mtl_file(filepath);
	if (!file.is_open()) return result;
	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if (tok == "Ke" && !current.empty()) {
			RGB c;
			ss >> c.r >> c.g >> c.b;
			if (c.r > 1e-6 || c.g > 1e-6 || c.b > 1e-6)
				result[current] = c;
		}
	}
	return result;
}

// material name -> map_Bump (bump/normal) texture path. Matches "map_Bump"
// (the OBJ/.mtl spec's own casing), the all-lowercase "map_bump" actually
// used throughout this renderer's own Sponza/Bistro .mtl files, and the
// bare "bump" alias some exporters use.
//
// The OBJ/.mtl spec's map_Bump statement is genuinely ambiguous in the
// wild: it's used for both a real scalar height/displacement map AND a
// tangent-space RGB normal map, with nothing in the .mtl text itself to
// tell them apart - only the material name -> path mapping is resolved
// here; deciding which kind a given texture actually is requires
// inspecting the loaded image's own pixel content (each backend does this
// separately - see this header's own file comment for why).
inline std::unordered_map<std::string, std::string> parse_map_bump(const std::string& filepath) {
	std::unordered_map<std::string, std::string> result;
	std::ifstream file = open_mtl_file(filepath);
	if (!file.is_open()) return result;
	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if ((tok == "map_Bump" || tok == "map_bump" || tok == "bump") && !current.empty()) {
			std::string p = extract_plain_path(ss);
			if (!p.empty()) result[current] = p;
		}
	}
	return result;
}

// material name -> map_d (alpha-cutout mask) texture path. Matches "map_d"
// and, per the OBJ/.mtl spec's own alternate spelling, "map_D".
inline std::unordered_map<std::string, std::string> parse_map_d(const std::string& filepath) {
	std::unordered_map<std::string, std::string> result;
	std::ifstream file = open_mtl_file(filepath);
	if (!file.is_open()) return result;
	std::string line, current;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		std::string tok;
		ss >> tok;
		if (tok == "newmtl") {
			ss >> current;
		} else if ((tok == "map_d" || tok == "map_D") && !current.empty()) {
			std::string p = extract_plain_path(ss);
			if (!p.empty()) result[current] = p;
		}
	}
	return result;
}

// Rewrites a .mtl-relative texture path (e.g. "textures\foo.png" or
// "..\BuildingTextures\bar.png") into a path under texture_dir. Backslashes
// are normalized to forward slashes and any leading "../"/"./" segments are
// stripped rather than resolved literally - callers relocate the actual
// texture files under texture_dir instead of mirroring the original
// archive's exact directory nesting relative to the .mtl file, so only the
// meaningful tail (e.g. "BuildingTextures/bar.png") matters.
inline std::string resolve_texture_path(const std::string& relative_path, const std::string& texture_dir) {
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

} // namespace mtl_parse
