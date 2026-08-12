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
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
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

	LoadResult parsed = parse(bytes);
	if (!parsed.ok) parsed.error = path + ": " + parsed.error;
	return parsed;
}

} // namespace ply_mesh
