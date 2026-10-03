#pragma once
// ply_mesh.h -- reader for the Stanford PLY mesh format.
//
// pbrt scenes reference geometry as `Shape "plymesh" "string filename"
// [ "geometry/mesh.ply" ]`, and mesh.h reads only .obj, so this is the missing
// half of loading a real pbrt scene. Large scenes cannot inline millions of
// vertices as text, so in practice every substantial pbrt scene depends on it.
//
// SCOPE
// -----
// ASCII and both binary byte orders. Vertex positions, plus normals and
// texture coordinates when present; faces of any arity, triangulated as a fan.
// That covers essentially every PLY produced by an exporter.
//
// OBJ SUPPORT (non-standard extension)
// -----------------------------------
// loadFile() also reads Wavefront .obj (chosen by file extension), so a scene can
// write `Shape "plymesh" "string filename" [ "../models/bunny.obj" ]` and use a
// mesh asset as-is instead of first converting every multi-megabyte .obj to PLY.
// See parseObj() for what is and is not read (positions, texture coordinates,
// normals, faces of any arity as a fan; NOT .mtl materials).
//
// Deliberately handles elements and properties it does NOT care about, rather
// than assuming a layout. A PLY may carry per-vertex colour, confidence,
// material indices, or whole extra elements, in any order. Skipping them
// requires knowing each property's width, and getting that wrong does not
// produce a clean failure - it desynchronises the byte stream and yields a mesh
// of plausible-looking garbage. That is the single most likely bug in a reader
// like this, so the size table below is the part worth checking.
//
// TESTABILITY
// -----------
// parse() takes the file's BYTES, not a path, so the tests can build PLY files
// in memory and cover truncation and malformed headers without touching the
// filesystem. loadFile() is the thin convenience wrapper. Same Qt-free,
// pure-function shape as pbrt_scene.h beside it.

#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gzip_inflate.h"

namespace ply_mesh {

struct Mesh {
	std::vector<float> positions;   // 3 per vertex
	std::vector<float> normals;     // 3 per vertex; empty when the file had none
	std::vector<float> uvs;         // 2 per vertex; empty when the file had none
	std::vector<int> indices;       // 3 per triangle

	std::size_t vertexCount() const { return positions.size() / 3; }
	std::size_t triangleCount() const { return indices.size() / 3; }
};

struct LoadResult {
	bool ok = false;
	std::string error;
	Mesh mesh;
};

namespace detail {

enum class Format { Ascii, BinaryLittleEndian, BinaryBigEndian };

enum class ScalarType {
	Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64, Invalid
};

inline ScalarType scalarFromName(const std::string &n) {
	if (n == "char"   || n == "int8")    return ScalarType::Int8;
	if (n == "uchar"  || n == "uint8")   return ScalarType::UInt8;
	if (n == "short"  || n == "int16")   return ScalarType::Int16;
	if (n == "ushort" || n == "uint16")  return ScalarType::UInt16;
	if (n == "int"    || n == "int32")   return ScalarType::Int32;
	if (n == "uint"   || n == "uint32")  return ScalarType::UInt32;
	if (n == "float"  || n == "float32") return ScalarType::Float32;
	if (n == "double" || n == "float64") return ScalarType::Float64;
	return ScalarType::Invalid;
}

inline int scalarSize(ScalarType t) {
	switch (t) {
	case ScalarType::Int8:
	case ScalarType::UInt8:    return 1;
	case ScalarType::Int16:
	case ScalarType::UInt16:   return 2;
	case ScalarType::Int32:
	case ScalarType::UInt32:
	case ScalarType::Float32:  return 4;
	case ScalarType::Float64:  return 8;
	case ScalarType::Invalid:  break;
	}
	return 0;
}

struct Property {
	std::string name;
	bool isList = false;
	ScalarType countType = ScalarType::Invalid;   // list only
	ScalarType valueType = ScalarType::Invalid;
};

struct Element {
	std::string name;
	std::size_t count = 0;
	std::vector<Property> properties;
};

// Reads one scalar from `data`, advancing `off`. Returns false on truncation,
// which is what makes a short file an error rather than a crash.
inline bool readBinaryScalar(const std::string &data, std::size_t &off,
							 ScalarType type, Format fmt, double &out) {
	const int size = scalarSize(type);
	if (size == 0 || off + static_cast<std::size_t>(size) > data.size()) return false;

	unsigned char buf[8];
	std::memcpy(buf, data.data() + off, static_cast<std::size_t>(size));
	off += static_cast<std::size_t>(size);

	// The host is little-endian everywhere this builds; a big-endian file is
	// byte-reversed into host order rather than assumed away.
	if (fmt == Format::BinaryBigEndian) {
		for (int i = 0; i < size / 2; ++i) {
			const unsigned char t = buf[i];
			buf[i] = buf[size - 1 - i];
			buf[size - 1 - i] = t;
		}
	}

	switch (type) {
	case ScalarType::Int8:    { std::int8_t v;   std::memcpy(&v, buf, 1); out = v; break; }
	case ScalarType::UInt8:   { std::uint8_t v;  std::memcpy(&v, buf, 1); out = v; break; }
	case ScalarType::Int16:   { std::int16_t v;  std::memcpy(&v, buf, 2); out = v; break; }
	case ScalarType::UInt16:  { std::uint16_t v; std::memcpy(&v, buf, 2); out = v; break; }
	case ScalarType::Int32:   { std::int32_t v;  std::memcpy(&v, buf, 4); out = v; break; }
	case ScalarType::UInt32:  { std::uint32_t v; std::memcpy(&v, buf, 4); out = v; break; }
	case ScalarType::Float32: { float v;         std::memcpy(&v, buf, 4); out = v; break; }
	case ScalarType::Float64: { double v;        std::memcpy(&v, buf, 8); out = v; break; }
	case ScalarType::Invalid: return false;
	}
	return true;
}

// Which slot, if any, a vertex property feeds. Exporters disagree on the
// texture-coordinate names, so the common spellings are all accepted.
enum class VertexSlot { None, X, Y, Z, Nx, Ny, Nz, U, V };

inline VertexSlot vertexSlotFor(const std::string &n) {
	if (n == "x")  return VertexSlot::X;
	if (n == "y")  return VertexSlot::Y;
	if (n == "z")  return VertexSlot::Z;
	if (n == "nx") return VertexSlot::Nx;
	if (n == "ny") return VertexSlot::Ny;
	if (n == "nz") return VertexSlot::Nz;
	if (n == "u" || n == "s" || n == "texture_u" || n == "texture_s") return VertexSlot::U;
	if (n == "v" || n == "t" || n == "texture_v" || n == "texture_t") return VertexSlot::V;
	return VertexSlot::None;
}

inline bool isFaceIndexProperty(const std::string &n) {
	return n == "vertex_indices" || n == "vertex_index";
}

} // namespace detail

inline LoadResult parse(const std::string &data) {
	using namespace detail;
	LoadResult r;

	// ---- header ----------------------------------------------------------
	const std::size_t headerEnd = data.find("end_header");
	if (data.compare(0, 3, "ply") != 0) {
		r.error = "not a PLY file (missing 'ply' magic)";
		return r;
	}
	if (headerEnd == std::string::npos) {
		r.error = "malformed PLY: no 'end_header'";
		return r;
	}

	// Data starts after end_header's line terminator, which may be \n or \r\n.
	std::size_t dataStart = data.find('\n', headerEnd);
	if (dataStart == std::string::npos) dataStart = data.size(); else ++dataStart;

	std::istringstream header(data.substr(0, headerEnd));
	Format fmt = Format::Ascii;
	bool sawFormat = false;
	std::vector<Element> elements;

	std::string line;
	while (std::getline(header, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		std::istringstream ls(line);
		std::string kw;
		if (!(ls >> kw)) continue;

		if (kw == "ply" || kw == "comment" || kw == "obj_info") continue;

		if (kw == "format") {
			std::string f;
			ls >> f;
			if (f == "ascii") fmt = Format::Ascii;
			else if (f == "binary_little_endian") fmt = Format::BinaryLittleEndian;
			else if (f == "binary_big_endian") fmt = Format::BinaryBigEndian;
			else { r.error = "unsupported PLY format '" + f + "'"; return r; }
			sawFormat = true;
			continue;
		}
		if (kw == "element") {
			Element e;
			ls >> e.name >> e.count;
			elements.push_back(e);
			continue;
		}
		if (kw == "property") {
			if (elements.empty()) { r.error = "malformed PLY: property before any element"; return r; }
			Property p;
			std::string t;
			ls >> t;
			if (t == "list") {
				std::string ct, vt;
				ls >> ct >> vt >> p.name;
				p.isList = true;
				p.countType = scalarFromName(ct);
				p.valueType = scalarFromName(vt);
				if (p.countType == ScalarType::Invalid || p.valueType == ScalarType::Invalid) {
					r.error = "malformed PLY: unknown list property type in '" + line + "'";
					return r;
				}
			} else {
				ls >> p.name;
				p.valueType = scalarFromName(t);
				if (p.valueType == ScalarType::Invalid) {
					r.error = "malformed PLY: unknown property type '" + t + "'";
					return r;
				}
			}
			elements.back().properties.push_back(p);
			continue;
		}
	}

	if (!sawFormat) { r.error = "malformed PLY: no 'format' line"; return r; }

	// ---- body ------------------------------------------------------------
	// Elements are read in declaration order. Ones we do not recognise still
	// have to be consumed exactly, or everything after them decodes as noise.
	std::size_t off = dataStart;
	std::istringstream ascii(fmt == Format::Ascii ? data.substr(dataStart) : std::string());

	const auto readScalar = [&](ScalarType t, double &out) -> bool {
		if (fmt == Format::Ascii) return static_cast<bool>(ascii >> out);
		return readBinaryScalar(data, off, t, fmt, out);
	};

	bool haveNormals = false, haveUVs = false;

	for (const Element &e : elements) {
		const bool isVertex = (e.name == "vertex");
		const bool isFace   = (e.name == "face");

		for (std::size_t i = 0; i < e.count; ++i) {
			float px = 0, py = 0, pz = 0, nx = 0, ny = 0, nz = 0, u = 0, v = 0;

			for (const Property &p : e.properties) {
				if (p.isList) {
					double n = 0.0;
					if (!readScalar(p.countType, n))
						{ r.error = "truncated PLY: ran out of data in element '" + e.name + "'"; return r; }
					const int count = static_cast<int>(n);
					if (count < 0)
						{ r.error = "malformed PLY: negative list length in element '" + e.name + "'"; return r; }

					std::vector<int> poly;
					poly.reserve(static_cast<std::size_t>(count));
					for (int k = 0; k < count; ++k) {
						double val = 0.0;
						if (!readScalar(p.valueType, val))
							{ r.error = "truncated PLY: ran out of data in element '" + e.name + "'"; return r; }
						poly.push_back(static_cast<int>(val));
					}

					if (isFace && isFaceIndexProperty(p.name)) {
						// Fan-triangulate. Exporters emit quads and n-gons even
						// when the scene is nominally triangular.
						for (std::size_t k = 2; k < poly.size(); ++k) {
							r.mesh.indices.push_back(poly[0]);
							r.mesh.indices.push_back(poly[k - 1]);
							r.mesh.indices.push_back(poly[k]);
						}
					}
					continue;
				}

				double val = 0.0;
				if (!readScalar(p.valueType, val))
					{ r.error = "truncated PLY: ran out of data in element '" + e.name + "'"; return r; }
				if (!isVertex) continue;

				switch (vertexSlotFor(p.name)) {
				case VertexSlot::X:  px = static_cast<float>(val); break;
				case VertexSlot::Y:  py = static_cast<float>(val); break;
				case VertexSlot::Z:  pz = static_cast<float>(val); break;
				case VertexSlot::Nx: nx = static_cast<float>(val); haveNormals = true; break;
				case VertexSlot::Ny: ny = static_cast<float>(val); break;
				case VertexSlot::Nz: nz = static_cast<float>(val); break;
				case VertexSlot::U:  u  = static_cast<float>(val); haveUVs = true; break;
				case VertexSlot::V:  v  = static_cast<float>(val); break;
				case VertexSlot::None: break;   // colour, confidence, ... ignored
				}
			}

			if (isVertex) {
				r.mesh.positions.push_back(px);
				r.mesh.positions.push_back(py);
				r.mesh.positions.push_back(pz);
				if (haveNormals) {
					r.mesh.normals.push_back(nx);
					r.mesh.normals.push_back(ny);
					r.mesh.normals.push_back(nz);
				}
				if (haveUVs) {
					r.mesh.uvs.push_back(u);
					r.mesh.uvs.push_back(v);
				}
			}
		}
	}

	if (r.mesh.positions.empty()) { r.error = "PLY contains no vertices"; return r; }

	// An out-of-range index would index past the vertex array during
	// triangulation downstream - a crash or silent garbage, well away from the
	// actual cause. Reject it here, where the file can be named.
	const int vertexCount = static_cast<int>(r.mesh.vertexCount());
	for (int idx : r.mesh.indices) {
		if (idx < 0 || idx >= vertexCount) {
			r.error = "malformed PLY: face index " + std::to_string(idx)
					  + " is outside the " + std::to_string(vertexCount) + " vertices";
			return r;
		}
	}

	r.ok = true;
	return r;
}

// Wavefront OBJ reader - `v`, `vt`, `vn` and `f` records only.
//
//  - Faces of any arity are fan-triangulated, exactly like the PLY path.
//  - Face vertex references may be `v`, `v/vt`, `v//vn` or `v/vt/vn`, and any
//    index may be negative (relative to the end of the list so far).
//  - `vn` IS read when faces reference it, so a smooth-shaded export keeps its
//    shading (an OBJ with no `vn`, like the Stanford scans, shades with flat
//    per-face geometric normals - what a plymesh with no normals gets, and what
//    the native mesh scenes this exists for do).
//  - UVs and normals are per-vertex in a Mesh but per-corner in an OBJ, so when
//    the file carries `vt`/`vn` data a position is duplicated for every distinct
//    (position, uv, normal) triple it is used with; with neither (or a mesh that
//    never references them) positions are used 1:1 and Mesh.uvs/Mesh.normals
//    stay empty, so plain meshes pay nothing.
//  - Everything else (`o`, `g`, `s`, `usemtl`, `mtllib`, comments) is ignored:
//    one mesh, one material, set by the scene file's own Material directive.
// Out-of-range references fail the load with the line named, like the PLY path.
inline LoadResult parseObj(const std::string &data) {
	LoadResult r;
	std::vector<float> pos;     // 3 per `v`
	std::vector<float> uv;      // 2 per `vt`
	std::vector<float> nrm;     // 3 per `vn`

	// A cheap pre-scan: does any face corner actually reference a vt / a vn?
	// (`v/vt` and `v/vt/vn` reference a vt; `v//vn` and `v/vt/vn` reference a vn.)
	bool needUv = false, needN = false;
	{
		std::size_t i = 0;
		const std::size_t n = data.size();
		while (i < n && !(needUv && needN)) {
			std::size_t e = data.find('\n', i);
			if (e == std::string::npos) e = n;
			if (e - i > 2 && data[i] == 'f' && (data[i + 1] == ' ' || data[i + 1] == '\t')) {
				std::size_t k = i + 1;
				while (k < e) {
					while (k < e && (data[k] == ' ' || data[k] == '\t' || data[k] == '\r')) ++k;
					const std::size_t tokStart = k;
					while (k < e && data[k] != ' ' && data[k] != '\t' && data[k] != '\r') ++k;
					// Searches ONLY inside this token [tokStart,k): a bare
					// data.find('/', tokStart) scans to the end of the whole file
					// when the file has no slashes at all (the common position-only
					// OBJ), once per face token - quadratic, ~24 s for a 10 MB mesh.
					auto slashIn = [&](std::size_t from) {
						for (std::size_t q = from; q < k; ++q)
							if (data[q] == '/') return q;
						return k;
					};
					const std::size_t s1 = slashIn(tokStart);
					if (s1 >= k) continue;
					auto isNum = [&](std::size_t q) {
						return q < k && ((data[q] >= '0' && data[q] <= '9') || data[q] == '-');
					};
					if (isNum(s1 + 1)) needUv = true;
					const std::size_t s2 = slashIn(s1 + 1);
					if (s2 < k && isNum(s2 + 1)) needN = true;
				}
			}
			i = e + 1;
		}
	}
	const bool expand = needUv || needN;

	struct Corner { int v, t, n; };
	std::vector<Corner> corners;
	struct Key {
		int v, t, n;
		bool operator==(const Key &o) const { return v == o.v && t == o.t && n == o.n; }
	};
	struct KeyHash {
		std::size_t operator()(const Key &k) const {
			std::size_t h = static_cast<std::size_t>(k.v) * 0x9E3779B97F4A7C15ull;
			h ^= static_cast<std::size_t>(k.t + 1) * 0xC2B2AE3D27D4EB4Full + (h << 6) + (h >> 2);
			h ^= static_cast<std::size_t>(k.n + 1) * 0x165667B19E3779F9ull + (h << 6) + (h >> 2);
			return h;
		}
	};
	std::unordered_map<Key, int, KeyHash> corner;

	std::size_t lineNo = 0, i = 0;
	const std::size_t n = data.size();
	auto parseIndex = [](const char *&c, const char *end, int count, int &out, bool &present) {
		// Parses an optionally-signed integer index and resolves negatives
		// against `count`; `present` is false when the field is empty (`v//vn`).
		present = false;
		const char *q = c;
		bool neg = false;
		if (q < end && (*q == '-' || *q == '+')) { neg = (*q == '-'); ++q; }
		if (q >= end || *q < '0' || *q > '9') return;
		long v = 0;
		while (q < end && *q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); ++q; }
		c = q;
		present = true;
		out = neg ? static_cast<int>(count - v) : static_cast<int>(v - 1);
	};

	while (i < n) {
		std::size_t e = data.find('\n', i);
		if (e == std::string::npos) e = n;
		++lineNo;
		const char *c = data.data() + i;
		const char *end = data.data() + e;
		i = e + 1;
		while (c < end && (*c == ' ' || *c == '\t')) ++c;
		if (c >= end) continue;
		if (c[0] == 'v' && c + 1 < end && (c[1] == ' ' || c[1] == '\t')) {
			char *next = nullptr;
			float x = std::strtof(c + 1, &next);
			float y = std::strtof(next, &next);
			float z = std::strtof(next, &next);
			pos.push_back(x); pos.push_back(y); pos.push_back(z);
		} else if (c[0] == 'v' && c + 2 < end && c[1] == 't' && (c[2] == ' ' || c[2] == '\t')) {
			char *next = nullptr;
			float u = std::strtof(c + 2, &next);
			float v = std::strtof(next, &next);
			uv.push_back(u); uv.push_back(v);
		} else if (c[0] == 'v' && c + 2 < end && c[1] == 'n' && (c[2] == ' ' || c[2] == '\t')) {
			char *next = nullptr;
			float x = std::strtof(c + 2, &next);
			float y = std::strtof(next, &next);
			float z = std::strtof(next, &next);
			nrm.push_back(x); nrm.push_back(y); nrm.push_back(z);
		} else if (c[0] == 'f' && c + 1 < end && (c[1] == ' ' || c[1] == '\t')) {
			std::vector<int> face;
			const char *q = c + 1;
			while (q < end) {
				while (q < end && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
				if (q >= end) break;
				int vi = 0, ti = -1, ni = -1;
				bool havePos = false, haveUv = false, haveN = false;
				parseIndex(q, end, static_cast<int>(pos.size() / 3), vi, havePos);
				if (q < end && *q == '/') {
					++q;
					parseIndex(q, end, static_cast<int>(uv.size() / 2), ti, haveUv);
					if (q < end && *q == '/') {
						++q;
						parseIndex(q, end, static_cast<int>(nrm.size() / 3), ni, haveN);
					}
				}
				// skip any trailing junk on this token
				while (q < end && *q != ' ' && *q != '\t' && *q != '\r') ++q;
				const std::string at = "OBJ line " + std::to_string(lineNo) + ": ";
				if (!havePos) {
					r.error = at + "face vertex has no position index";
					return r;
				}
				if (vi < 0 || static_cast<std::size_t>(vi) >= pos.size() / 3) {
					r.error = at + "face index " + std::to_string(vi + 1) + " is outside the " +
							  std::to_string(pos.size() / 3) + " vertices declared so far";
					return r;
				}
				int useUv = -1, useN = -1;
				if (needUv && haveUv) {
					if (ti < 0 || static_cast<std::size_t>(ti) >= uv.size() / 2) {
						r.error = at + "texture index " + std::to_string(ti + 1) + " is outside the " +
								  std::to_string(uv.size() / 2) + " texture coordinates declared so far";
						return r;
					}
					useUv = ti;
				}
				if (needN && haveN) {
					if (ni < 0 || static_cast<std::size_t>(ni) >= nrm.size() / 3) {
						r.error = at + "normal index " + std::to_string(ni + 1) + " is outside the " +
								  std::to_string(nrm.size() / 3) + " normals declared so far";
						return r;
					}
					useN = ni;
				}
				if (!expand) {
					face.push_back(vi);
				} else {
					const Key key{vi, useUv, useN};
					auto it = corner.find(key);
					if (it == corner.end()) {
						it = corner.emplace(key, static_cast<int>(corners.size())).first;
						corners.push_back({vi, useUv, useN});
					}
					face.push_back(it->second);
				}
			}
			for (std::size_t k = 2; k < face.size(); ++k) {
				r.mesh.indices.push_back(face[0]);
				r.mesh.indices.push_back(face[k - 1]);
				r.mesh.indices.push_back(face[k]);
			}
		}
	}

	if (pos.empty()) { r.error = "OBJ has no vertices"; return r; }
	if (expand) {
		r.mesh.positions.reserve(corners.size() * 3);
		if (needUv) r.mesh.uvs.reserve(corners.size() * 2);
		if (needN) r.mesh.normals.reserve(corners.size() * 3);
		for (const Corner &cv : corners) {
			r.mesh.positions.push_back(pos[cv.v * 3 + 0]);
			r.mesh.positions.push_back(pos[cv.v * 3 + 1]);
			r.mesh.positions.push_back(pos[cv.v * 3 + 2]);
			if (needUv) {
				r.mesh.uvs.push_back(cv.t >= 0 ? uv[cv.t * 2 + 0] : 0.0f);
				r.mesh.uvs.push_back(cv.t >= 0 ? uv[cv.t * 2 + 1] : 0.0f);
			}
			if (needN) {
				// A corner with no normal (mixed file) gets +Z rather than garbage.
				r.mesh.normals.push_back(cv.n >= 0 ? nrm[cv.n * 3 + 0] : 0.0f);
				r.mesh.normals.push_back(cv.n >= 0 ? nrm[cv.n * 3 + 1] : 0.0f);
				r.mesh.normals.push_back(cv.n >= 0 ? nrm[cv.n * 3 + 2] : 1.0f);
			}
		}
	} else {
		r.mesh.positions = std::move(pos);
	}
	r.ok = true;
	return r;
}

// Convenience wrapper. Binary mode matters: on Windows a text-mode read would
// eat 0x0D bytes inside binary vertex data.
inline LoadResult loadFile(const std::string &path) {
	LoadResult r;
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		r.error = "cannot open PLY file: " + path;
		return r;
	}
	std::ostringstream ss;
	ss << in.rdbuf();
	std::string bytes = ss.str();

	// Detected from the content, not the extension. Most published pbrt
	// geometry is gzipped, and a scene is free to name a compressed file .ply
	// - sniffing the magic bytes is right in both directions, where trusting
	// the name is wrong in both.
	if (gzip::looksGzipped(bytes)) {
		std::string inflated, error;
		if (!gzip::inflate(bytes, inflated, error)) {
			r.error = path + ": " + error;
			return r;
		}
		bytes.swap(inflated);
	}

	// .obj goes through parseObj() (see the OBJ SUPPORT note at the top of this
	// file); everything else is PLY. By extension, case-insensitively: unlike a
	// gzip wrapper, an OBJ has no magic bytes to sniff.
	const std::size_t dot = path.find_last_of('.');
	if (dot != std::string::npos) {
		std::string ext = path.substr(dot);
		for (char &ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
		if (ext == ".obj") {
			LoadResult objParsed = parseObj(bytes);
			if (!objParsed.ok) objParsed.error = path + ": " + objParsed.error;
			return objParsed;
		}
	}

	LoadResult parsed = parse(bytes);
	if (!parsed.ok) parsed.error = path + ": " + parsed.error;
	return parsed;
}

} // namespace ply_mesh
