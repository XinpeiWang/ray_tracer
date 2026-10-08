#pragma once
// scene_document.h -- the document model behind the GUI's Scene Builder: a small, editable description of a scene (a camera, shapes with
// materials, lights) that is written out as a pbrt-v4 scene the renderer already loads, and read back for editing.
//
// Header-only and dependency-free (standard library only), so the Qt GUI, the unit tests and the command line all use the same code.
//
// What a builder file is. toPbrt() writes an ordinary .pbrt file whose first lines are comments, one of them a single-line JSON copy of the whole
// document (`# @rt-builder-doc {...}`). Any renderer reads the file as pbrt (the comment is ignored); the Scene Builder reads the JSON back, so the
// scene stays editable. A .pbrt file without that line is not a builder scene (fromPbrt() says so) and is not parsed.
//
// Conventions of the builder's world (so a user never has to think about pbrt's):
//   * +Y is up, units are free (the starter scene is a few units across).
//   * Rotation is three angles in degrees about the world X, Y and Z axes, applied in the order X, then Y, then Z.
//   * A Sphere is centred at its position. Box, Quad, Cylinder, Cone and the generated shapes (Pyramid, Wedge, Stairs, Torus, Capsule, Dome, Tube; see
//     scene_shapes.h) are centred on their position too (their bounding box centre).
//   * A Quad is a flat rectangle in the XZ plane (size.x by size.z) whose front side faces +Y; a Disk faces +Y; a Cylinder and a Cone stand along +Y.
//     An emissive object radiates from its front side (a Box, a Sphere and the generated shapes outward; a Quad or Disk towards +Y, so a ceiling light is rotated 180 degrees about X).
//   * Colours are linear RGB in 0..1 (an emission colour is multiplied by its strength).
//
// pbrt-v4 reference for every directive written here: https://pbrt.org/fileformat-v4

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "scene_shapes.h"

namespace scene_doc {

struct Float3 {
	double x = 0, y = 0, z = 0;
};

struct Rgb {
	double r = 0.8, g = 0.8, b = 0.8;
};

enum class MaterialKind { Diffuse, Conductor, Dielectric, CoatedDiffuse, DiffuseTransmission };

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	Rgb color{0.8, 0.8, 0.8};          // diffuse / coated reflectance, conductor reflectance; unused by Dielectric
	bool checker = false;              // Diffuse only: a checkerboard of `color` and `color2`
	Rgb color2{0.2, 0.2, 0.2};
	double checkerCount = 8.0;         // checks across the object (its width), for Quad / Disk / Sphere
	double roughness = 0.0;            // Conductor, Dielectric, CoatedDiffuse: 0 = smooth
	double ior = 1.5;                  // Dielectric and CoatedDiffuse
	Rgb transmittance{0.5, 0.5, 0.5};  // DiffuseTransmission (its reflectance is `color`)
	std::string imageFile;             // Diffuse / CoatedDiffuse: a picture used as the surface colour (replaces `color` and the checker)
};

enum class ShapeKind { Sphere, Box, Quad, Disk, Cylinder, Cone, Mesh, Pyramid, Wedge, Stairs, Torus, Capsule, Dome, Tube };

// Every shape, in the order the GUI lists them.
inline const std::vector<ShapeKind>& allShapeKinds() {
	static const std::vector<ShapeKind> all = {ShapeKind::Sphere,  ShapeKind::Box,   ShapeKind::Quad,    ShapeKind::Disk,  ShapeKind::Cylinder,
	                                           ShapeKind::Cone,    ShapeKind::Pyramid, ShapeKind::Wedge, ShapeKind::Stairs, ShapeKind::Torus,
	                                           ShapeKind::Capsule, ShapeKind::Dome,  ShapeKind::Tube,    ShapeKind::Mesh};
	return all;
}
// A shape that is written as a generated triangle mesh (scene_shapes.h) rather than one of pbrt's own shapes.
inline bool isGeneratedShape(ShapeKind k) {
	return k == ShapeKind::Pyramid || k == ShapeKind::Wedge || k == ShapeKind::Stairs || k == ShapeKind::Torus || k == ShapeKind::Capsule ||
	       k == ShapeKind::Dome || k == ShapeKind::Tube;
}

struct Object {
	std::string name = "Object";
	ShapeKind shape = ShapeKind::Sphere;
	Float3 position{};
	Float3 rotation{};                   // degrees, about world X, Y, Z (applied in that order)
	double radius = 1.0;               // Sphere, Disk, Cylinder, Cone, Capsule, Dome; Torus: the ring; Tube: outside
	double radius2 = 0.25;             // Torus: the tube's radius; Tube: the hole's radius
	double height = 1.0;               // Cylinder, Cone, Pyramid, Capsule, Tube
	Float3 size{1.0, 1.0, 1.0};          // Box, Wedge, Stairs (x, y, z); Quad (x and z); Pyramid (x and z, the base)
	int steps = 5;                     // Stairs
	double meshScale = 1.0;            // Mesh
	std::string meshFile;              // Mesh: a .ply file
	Material material;
	bool emissive = false;             // an area light
	Rgb emission{1.0, 1.0, 1.0};
	double emissionStrength = 10.0;    // radiance = emission * strength
	bool twoSided = false;             // emits from both sides
};

enum class LightKind { Point, Spot, Distant, Infinite };

struct Light {
	std::string name = "Light";
	LightKind kind = LightKind::Point;
	Float3 position{0.0, 5.0, 0.0};      // Point, Spot (and Distant: where it shines from)
	Float3 target{0.0, 0.0, 0.0};        // Spot, Distant: what it points at
	Rgb color{1.0, 1.0, 1.0};
	double intensity = 50.0;           // Point/Spot: radiant intensity; Distant/Infinite: radiance
	double coneAngle = 30.0;           // Spot, degrees
	double coneDelta = 5.0;            // Spot, degrees of soft edge
	std::string imageFile;             // Infinite: an equirectangular image (empty: a constant colour)
};

struct Camera {
	Float3 position{0.0, 2.5, 8.0};
	Float3 target{0.0, 1.0, 0.0};
	Float3 up{0.0, 1.0, 0.0};
	double fov = 40.0;                 // vertical field of view, degrees
	double lensRadius = 0.0;           // 0 = a pinhole; > 0 blurs what is not at focusDistance
	double focusDistance = 8.0;
};

struct RenderSettings {
	int width = 800;
	int height = 600;
	int samples = 64;
	int maxDepth = 8;
};

struct Document {
	std::string title = "Untitled scene";
	Camera camera;
	RenderSettings render;
	std::vector<Object> objects;
	std::vector<Light> lights;
};

struct Problem {
	enum class Severity { Warning, Error } severity = Severity::Warning;
	std::string message;
};

// ---------------------------------------------------------------------------------------------------------------------------------
// Names <-> enums (also what the JSON stores)
// ---------------------------------------------------------------------------------------------------------------------------------
inline const char* toString(MaterialKind k) {
	switch (k) {
		case MaterialKind::Diffuse: return "diffuse";
		case MaterialKind::Conductor: return "conductor";
		case MaterialKind::Dielectric: return "dielectric";
		case MaterialKind::CoatedDiffuse: return "coateddiffuse";
		case MaterialKind::DiffuseTransmission: return "diffusetransmission";
	}
	return "diffuse";
}
inline bool materialKindFromString(const std::string& s, MaterialKind& out) {
	for (MaterialKind k : {MaterialKind::Diffuse, MaterialKind::Conductor, MaterialKind::Dielectric, MaterialKind::CoatedDiffuse,
	                       MaterialKind::DiffuseTransmission})
		if (s == toString(k)) { out = k; return true; }
	return false;
}
inline const char* toString(ShapeKind k) {
	switch (k) {
		case ShapeKind::Sphere: return "sphere";
		case ShapeKind::Box: return "box";
		case ShapeKind::Quad: return "quad";
		case ShapeKind::Disk: return "disk";
		case ShapeKind::Cylinder: return "cylinder";
		case ShapeKind::Cone: return "cone";
		case ShapeKind::Mesh: return "mesh";
		case ShapeKind::Pyramid: return "pyramid";
		case ShapeKind::Wedge: return "wedge";
		case ShapeKind::Stairs: return "stairs";
		case ShapeKind::Torus: return "torus";
		case ShapeKind::Capsule: return "capsule";
		case ShapeKind::Dome: return "dome";
		case ShapeKind::Tube: return "tube";
	}
	return "sphere";
}
inline bool shapeKindFromString(const std::string& s, ShapeKind& out) {
	for (ShapeKind k : allShapeKinds())
		if (s == toString(k)) { out = k; return true; }
	return false;
}
inline const char* toString(LightKind k) {
	switch (k) {
		case LightKind::Point: return "point";
		case LightKind::Spot: return "spot";
		case LightKind::Distant: return "distant";
		case LightKind::Infinite: return "infinite";
	}
	return "point";
}
inline bool lightKindFromString(const std::string& s, LightKind& out) {
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite})
		if (s == toString(k)) { out = k; return true; }
	return false;
}

namespace detail {

// ---- number and string formatting that never depends on the process locale -------------------------------------------------------
inline std::string num(double v) {
	if (!std::isfinite(v)) v = 0.0;
	std::ostringstream os;
	os.imbue(std::locale::classic());
	os << std::setprecision(9) << v;
	return os.str();
}
// The shortest of 15, 16 and 17 significant digits that reads back as exactly v: the JSON copy of the document (and so every undo snapshot) must not round.
inline std::string numExact(double v) {
	if (!std::isfinite(v)) v = 0.0;
	std::string text;
	for (int digits : {15, 16, 17}) {
		std::ostringstream os;
		os.imbue(std::locale::classic());
		os << std::setprecision(digits) << v;
		text = os.str();
		std::istringstream is(text);
		is.imbue(std::locale::classic());
		double back = 0.0;
		if ((is >> back) && back == v) break;
	}
	return text;
}
inline std::string vec(const Float3& v) { return num(v.x) + " " + num(v.y) + " " + num(v.z); }
inline std::string quoted(const std::string& s) {
	std::string out = "\"";
	for (char c : s) {
		if (c == '"' || c == '\\') out += '\\';
		if (c == '\n' || c == '\r') c = ' ';
		out += c;
	}
	return out + "\"";
}
// Text for a '#' comment line: a newline in a name must not end the comment and start a directive.
inline std::string commentText(std::string s) {
	for (char& c : s)
		if (c == '\n' || c == '\r') c = ' ';
	// The word that marks the line holding the document (see fromPbrt) must appear only on that line, so a name that contains it is written with a space.
	for (size_t at = s.find("@rt-builder-doc"); at != std::string::npos; at = s.find("@rt-builder-doc", at + 1)) s[at + 3] = ' ';
	return s;
}
// A path as pbrt wants it: forward slashes, so a file written on Windows still reads on macOS.
inline std::string pbrtPath(std::string p) {
	std::replace(p.begin(), p.end(), '\\', '/');
	return p;
}
inline Rgb scaled(const Rgb& c, double s) { return Rgb{c.r * s, c.g * s, c.b * s}; }
inline std::string rgb(const Rgb& c) { return num(c.r) + " " + num(c.g) + " " + num(c.b); }

// ---- a tiny JSON value, writer and parser (enough for the document; no external dependency) -------------------------------------
struct Json {
	enum class Type { Null, Bool, Num, Str, Arr, Obj } type = Type::Null;
	bool b = false;
	double n = 0.0;
	std::string s;
	std::vector<Json> a;
	std::vector<std::pair<std::string, Json>> o;

	static Json number(double v) { Json j; j.type = Type::Num; j.n = v; return j; }
	static Json boolean(bool v) { Json j; j.type = Type::Bool; j.b = v; return j; }
	static Json string(const std::string& v) { Json j; j.type = Type::Str; j.s = v; return j; }
	static Json array() { Json j; j.type = Type::Arr; return j; }
	static Json object() { Json j; j.type = Type::Obj; return j; }
	Json& set(const std::string& k, Json v) { o.emplace_back(k, std::move(v)); return *this; }
	Json& push(Json v) { a.push_back(std::move(v)); return *this; }
	const Json* find(const std::string& k) const {
		for (const auto& kv : o) if (kv.first == k) return &kv.second;
		return nullptr;
	}
};

inline void dump(const Json& j, std::string& out) {
	switch (j.type) {
		case Json::Type::Null: out += "null"; break;
		case Json::Type::Bool: out += j.b ? "true" : "false"; break;
		case Json::Type::Num: out += numExact(j.n); break;
		case Json::Type::Str: {
			out += '"';
			for (unsigned char c : j.s) {
				switch (c) {
					case '"': out += "\\\""; break;
					case '\\': out += "\\\\"; break;
					case '\n': out += "\\n"; break;
					case '\r': out += "\\r"; break;
					case '\t': out += "\\t"; break;
					default:
						if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
						else out += static_cast<char>(c);
				}
			}
			out += '"';
			break;
		}
		case Json::Type::Arr: {
			out += '[';
			for (size_t i = 0; i < j.a.size(); ++i) { if (i) out += ','; dump(j.a[i], out); }
			out += ']';
			break;
		}
		case Json::Type::Obj: {
			out += '{';
			for (size_t i = 0; i < j.o.size(); ++i) {
				if (i) out += ',';
				dump(Json::string(j.o[i].first), out);
				out += ':';
				dump(j.o[i].second, out);
			}
			out += '}';
			break;
		}
	}
}

class JsonParser {
  public:
	explicit JsonParser(const std::string& t) : t_(t) {}
	bool parse(Json& out, std::string& err) {
		skip();
		if (!value(out, 0)) { err = err_.empty() ? "invalid JSON" : err_; return false; }
		skip();
		if (p_ != t_.size()) { err = "unexpected text after the JSON value"; return false; }
		return true;
	}

  private:
	const std::string& t_;
	size_t p_ = 0;
	std::string err_;

	void skip() { while (p_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[p_]))) ++p_; }
	bool fail(const char* m) { if (err_.empty()) err_ = m; return false; }
	bool lit(const char* w) {
		size_t n = std::char_traits<char>::length(w);
		if (t_.compare(p_, n, w) != 0) return false;
		p_ += n;
		return true;
	}
	bool value(Json& out, int depth) {
		if (depth > 64) return fail("JSON nested too deeply");
		if (p_ >= t_.size()) return fail("unexpected end of JSON");
		char c = t_[p_];
		if (c == '{') {
			++p_; out = Json::object(); skip();
			if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
			for (;;) {
				skip();
				Json key;
				if (p_ >= t_.size() || t_[p_] != '"' || !str(key)) return fail("expected a string key");
				skip();
				if (p_ >= t_.size() || t_[p_] != ':') return fail("expected ':'");
				++p_; skip();
				Json v;
				if (!value(v, depth + 1)) return false;
				out.o.emplace_back(key.s, std::move(v));
				skip();
				if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
				if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
				return fail("expected ',' or '}'");
			}
		}
		if (c == '[') {
			++p_; out = Json::array(); skip();
			if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
			for (;;) {
				skip();
				Json v;
				if (!value(v, depth + 1)) return false;
				out.a.push_back(std::move(v));
				skip();
				if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
				if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
				return fail("expected ',' or ']'");
			}
		}
		if (c == '"') return str(out);
		if (lit("true")) { out = Json::boolean(true); return true; }
		if (lit("false")) { out = Json::boolean(false); return true; }
		if (lit("null")) { out = Json(); return true; }
		// number
		size_t q = p_;
		while (q < t_.size() && (std::isdigit(static_cast<unsigned char>(t_[q])) || t_[q] == '-' || t_[q] == '+' || t_[q] == '.' ||
		                         t_[q] == 'e' || t_[q] == 'E')) ++q;
		if (q == p_) return fail("unexpected character in JSON");
		std::istringstream is(t_.substr(p_, q - p_));
		is.imbue(std::locale::classic());
		double v = 0.0;
		if (!(is >> v)) return fail("bad number in JSON");
		p_ = q;
		out = Json::number(v);
		return true;
	}
	bool str(Json& out) {
		++p_;  // opening quote
		std::string s;
		while (p_ < t_.size() && t_[p_] != '"') {
			char c = t_[p_++];
			if (c == '\\') {
				if (p_ >= t_.size()) return fail("bad escape in JSON string");
				char e = t_[p_++];
				switch (e) {
					case '"': s += '"'; break;
					case '\\': s += '\\'; break;
					case '/': s += '/'; break;
					case 'n': s += '\n'; break;
					case 'r': s += '\r'; break;
					case 't': s += '\t'; break;
					case 'b': s += '\b'; break;
					case 'f': s += '\f'; break;
					case 'u': {
						if (p_ + 4 > t_.size()) return fail("bad \\u escape in JSON string");
						unsigned code = static_cast<unsigned>(std::strtoul(t_.substr(p_, 4).c_str(), nullptr, 16));
						p_ += 4;
						// the writer only emits \u00XX for control characters; anything else is kept as a UTF-8 sequence
						if (code < 0x80) s += static_cast<char>(code);
						else if (code < 0x800) { s += static_cast<char>(0xC0 | (code >> 6)); s += static_cast<char>(0x80 | (code & 0x3F)); }
						else { s += static_cast<char>(0xE0 | (code >> 12)); s += static_cast<char>(0x80 | ((code >> 6) & 0x3F)); s += static_cast<char>(0x80 | (code & 0x3F)); }
						break;
					}
					default: return fail("bad escape in JSON string");
				}
			} else {
				s += c;
			}
		}
		if (p_ >= t_.size()) return fail("unterminated JSON string");
		++p_;  // closing quote
		out = Json::string(s);
		return true;
	}
};

// ---- Json <-> document -------------------------------------------------------------------------------------------------------------
inline Json toJson(const Float3& v) { Json j = Json::array(); j.push(Json::number(v.x)).push(Json::number(v.y)).push(Json::number(v.z)); return j; }
inline Json toJson(const Rgb& c) { Json j = Json::array(); j.push(Json::number(c.r)).push(Json::number(c.g)).push(Json::number(c.b)); return j; }

inline bool readNum(const Json& o, const char* k, double& out) {
	const Json* j = o.find(k);
	if (!j) return true;  // a missing field keeps its default
	if (j->type != Json::Type::Num) return false;
	out = j->n;
	return true;
}
inline bool readInt(const Json& o, const char* k, int& out) {
	double d = out;
	if (!readNum(o, k, d)) return false;
	out = static_cast<int>(std::lround(d));
	return true;
}
inline bool readBool(const Json& o, const char* k, bool& out) {
	const Json* j = o.find(k);
	if (!j) return true;
	if (j->type != Json::Type::Bool) return false;
	out = j->b;
	return true;
}
inline bool readStr(const Json& o, const char* k, std::string& out) {
	const Json* j = o.find(k);
	if (!j) return true;
	if (j->type != Json::Type::Str) return false;
	out = j->s;
	return true;
}
inline bool readTriple(const Json& o, const char* k, double& a, double& b, double& c) {
	const Json* j = o.find(k);
	if (!j) return true;
	if (j->type != Json::Type::Arr || j->a.size() != 3) return false;
	for (const Json& e : j->a) if (e.type != Json::Type::Num) return false;
	a = j->a[0].n; b = j->a[1].n; c = j->a[2].n;
	return true;
}
inline bool readVec(const Json& o, const char* k, Float3& v) { return readTriple(o, k, v.x, v.y, v.z); }
inline bool readRgb(const Json& o, const char* k, Rgb& c) { return readTriple(o, k, c.r, c.g, c.b); }

}  // namespace detail

// ---------------------------------------------------------------------------------------------------------------------------------
// Document <-> JSON
// ---------------------------------------------------------------------------------------------------------------------------------
inline std::string toJson(const Document& d) {
	using detail::Json;
	Json root = Json::object();
	root.set("version", Json::number(1));
	root.set("title", Json::string(d.title));

	Json cam = Json::object();
	cam.set("position", detail::toJson(d.camera.position)).set("target", detail::toJson(d.camera.target)).set("up", detail::toJson(d.camera.up));
	cam.set("fov", Json::number(d.camera.fov)).set("lensRadius", Json::number(d.camera.lensRadius)).set("focusDistance", Json::number(d.camera.focusDistance));
	root.set("camera", std::move(cam));

	Json rs = Json::object();
	rs.set("width", Json::number(d.render.width)).set("height", Json::number(d.render.height));
	rs.set("samples", Json::number(d.render.samples)).set("maxDepth", Json::number(d.render.maxDepth));
	root.set("render", std::move(rs));

	Json objs = Json::array();
	for (const Object& o : d.objects) {
		Json j = Json::object();
		j.set("name", Json::string(o.name)).set("shape", Json::string(toString(o.shape)));
		j.set("position", detail::toJson(o.position)).set("rotation", detail::toJson(o.rotation));
		j.set("radius", Json::number(o.radius)).set("height", Json::number(o.height)).set("size", detail::toJson(o.size));
		j.set("radius2", Json::number(o.radius2)).set("steps", Json::number(o.steps));
		j.set("meshScale", Json::number(o.meshScale)).set("meshFile", Json::string(o.meshFile));
		Json m = Json::object();
		m.set("kind", Json::string(toString(o.material.kind))).set("color", detail::toJson(o.material.color));
		m.set("checker", Json::boolean(o.material.checker)).set("color2", detail::toJson(o.material.color2));
		m.set("checkerCount", Json::number(o.material.checkerCount)).set("roughness", Json::number(o.material.roughness));
		m.set("ior", Json::number(o.material.ior)).set("transmittance", detail::toJson(o.material.transmittance));
		m.set("imageFile", Json::string(o.material.imageFile));
		j.set("material", std::move(m));
		j.set("emissive", Json::boolean(o.emissive)).set("emission", detail::toJson(o.emission));
		j.set("emissionStrength", Json::number(o.emissionStrength)).set("twoSided", Json::boolean(o.twoSided));
		objs.push(std::move(j));
	}
	root.set("objects", std::move(objs));

	Json lights = Json::array();
	for (const Light& l : d.lights) {
		Json j = Json::object();
		j.set("name", Json::string(l.name)).set("kind", Json::string(toString(l.kind)));
		j.set("position", detail::toJson(l.position)).set("target", detail::toJson(l.target)).set("color", detail::toJson(l.color));
		j.set("intensity", Json::number(l.intensity)).set("coneAngle", Json::number(l.coneAngle)).set("coneDelta", Json::number(l.coneDelta));
		j.set("imageFile", Json::string(l.imageFile));
		lights.push(std::move(j));
	}
	root.set("lights", std::move(lights));

	std::string out;
	detail::dump(root, out);
	return out;
}

inline bool fromJson(const std::string& text, Document& out, std::string& err) {
	using detail::Json;
	Json root;
	if (!detail::JsonParser(text).parse(root, err)) return false;
	if (root.type != Json::Type::Obj) { err = "the scene description is not a JSON object"; return false; }
	double version = 1;
	detail::readNum(root, "version", version);
	if (version > 1) { err = "this scene was saved by a newer version of the Scene Builder"; return false; }

	Document d;
	bool ok = detail::readStr(root, "title", d.title);
	if (const Json* cam = root.find("camera")) {
		if (cam->type != Json::Type::Obj) ok = false;
		else {
			ok = ok && detail::readVec(*cam, "position", d.camera.position) && detail::readVec(*cam, "target", d.camera.target) &&
			     detail::readVec(*cam, "up", d.camera.up) && detail::readNum(*cam, "fov", d.camera.fov) &&
			     detail::readNum(*cam, "lensRadius", d.camera.lensRadius) && detail::readNum(*cam, "focusDistance", d.camera.focusDistance);
		}
	}
	if (const Json* rs = root.find("render")) {
		if (rs->type != Json::Type::Obj) ok = false;
		else {
			ok = ok && detail::readInt(*rs, "width", d.render.width) && detail::readInt(*rs, "height", d.render.height) &&
			     detail::readInt(*rs, "samples", d.render.samples) && detail::readInt(*rs, "maxDepth", d.render.maxDepth);
		}
	}
	if (const Json* objs = root.find("objects")) {
		if (objs->type != Json::Type::Arr) ok = false;
		else for (const Json& j : objs->a) {
			Object o;
			if (j.type != Json::Type::Obj) { ok = false; break; }
			std::string shape = toString(o.shape);
			ok = ok && detail::readStr(j, "name", o.name) && detail::readStr(j, "shape", shape) && shapeKindFromString(shape, o.shape) &&
			     detail::readVec(j, "position", o.position) && detail::readVec(j, "rotation", o.rotation) && detail::readNum(j, "radius", o.radius) &&
			     detail::readNum(j, "height", o.height) && detail::readVec(j, "size", o.size) && detail::readNum(j, "meshScale", o.meshScale) && detail::readNum(j, "radius2", o.radius2) && detail::readInt(j, "steps", o.steps) &&
			     detail::readStr(j, "meshFile", o.meshFile) && detail::readBool(j, "emissive", o.emissive) && detail::readRgb(j, "emission", o.emission) &&
			     detail::readNum(j, "emissionStrength", o.emissionStrength) && detail::readBool(j, "twoSided", o.twoSided);
			if (const Json* m = j.find("material")) {
				if (m->type != Json::Type::Obj) ok = false;
				else {
					std::string kind = toString(o.material.kind);
					ok = ok && detail::readStr(*m, "kind", kind) && materialKindFromString(kind, o.material.kind) &&
					     detail::readRgb(*m, "color", o.material.color) && detail::readBool(*m, "checker", o.material.checker) &&
					     detail::readRgb(*m, "color2", o.material.color2) && detail::readNum(*m, "checkerCount", o.material.checkerCount) &&
					     detail::readNum(*m, "roughness", o.material.roughness) && detail::readNum(*m, "ior", o.material.ior) &&
					     detail::readRgb(*m, "transmittance", o.material.transmittance) && detail::readStr(*m, "imageFile", o.material.imageFile);
				}
			}
			if (!ok) break;
			d.objects.push_back(o);
		}
	}
	if (const Json* ls = root.find("lights")) {
		if (ls->type != Json::Type::Arr) ok = false;
		else for (const Json& j : ls->a) {
			Light l;
			if (j.type != Json::Type::Obj) { ok = false; break; }
			std::string kind = toString(l.kind);
			ok = ok && detail::readStr(j, "name", l.name) && detail::readStr(j, "kind", kind) && lightKindFromString(kind, l.kind) &&
			     detail::readVec(j, "position", l.position) && detail::readVec(j, "target", l.target) && detail::readRgb(j, "color", l.color) &&
			     detail::readNum(j, "intensity", l.intensity) && detail::readNum(j, "coneAngle", l.coneAngle) &&
			     detail::readNum(j, "coneDelta", l.coneDelta) && detail::readStr(j, "imageFile", l.imageFile);
			if (!ok) break;
			d.lights.push_back(l);
		}
	}
	if (!ok) { err = "the scene description has a field of the wrong type or an unknown name"; return false; }
	out = std::move(d);
	return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Checks, shown to the user before rendering
// ---------------------------------------------------------------------------------------------------------------------------------
namespace detail {
// An absolute path to a file that is not there. (A relative path is relative to wherever the .pbrt file ends up, which validate() cannot know;
// the Scene Builder's file dialogs only ever store absolute ones.)
inline bool absoluteFileMissing(const std::string& path) {
	if (path.empty()) return false;
	std::error_code ec;
	const std::filesystem::path p(path);
	return p.is_absolute() && !std::filesystem::exists(p, ec);
}
}  // namespace detail

inline std::vector<Problem> validate(const Document& d) {
	std::vector<Problem> out;
	auto err = [&](const std::string& m) { out.push_back({Problem::Severity::Error, m}); };
	auto warn = [&](const std::string& m) { out.push_back({Problem::Severity::Warning, m}); };

	const Float3 dc{d.camera.target.x - d.camera.position.x, d.camera.target.y - d.camera.position.y, d.camera.target.z - d.camera.position.z};
	const double len = std::sqrt(dc.x * dc.x + dc.y * dc.y + dc.z * dc.z);
	if (len < 1e-6) err("The camera is at its target: move one of them.");
	else {
		const double upLen = std::sqrt(d.camera.up.x * d.camera.up.x + d.camera.up.y * d.camera.up.y + d.camera.up.z * d.camera.up.z);
		if (upLen < 1e-9) err("The camera's up direction is zero.");
		else {
			const double cx = dc.y * d.camera.up.z - dc.z * d.camera.up.y, cy = dc.z * d.camera.up.x - dc.x * d.camera.up.z,
			             cz = dc.x * d.camera.up.y - dc.y * d.camera.up.x;
			if (std::sqrt(cx * cx + cy * cy + cz * cz) < 1e-6 * len * upLen) err("The camera looks straight along its up direction: tilt it a little.");
		}
	}
	if (!(d.camera.fov >= 1.0 && d.camera.fov <= 170.0)) err("The field of view must be between 1 and 170 degrees.");
	if (d.camera.lensRadius < 0.0) err("The lens radius cannot be negative.");
	if (d.camera.lensRadius > 0.0 && !(d.camera.focusDistance > 0.0)) err("A camera with depth of field needs a focus distance above zero.");
	if (d.render.width < 1 || d.render.height < 1 || d.render.width > 16384 || d.render.height > 16384) err("The image size must be between 1 and 16384 pixels each way.");
	if (d.render.samples < 1) err("At least one sample per pixel is needed.");
	if (d.render.maxDepth < 1 || d.render.maxDepth > 100) err("The maximum depth must be between 1 and 100.");

	bool anyLight = !d.lights.empty();
	for (const Object& o : d.objects) {
		const std::string who = "'" + o.name + "': ";
		if (o.emissive) anyLight = true;
		switch (o.shape) {
			case ShapeKind::Sphere: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Disk: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Cylinder:
			case ShapeKind::Cone:
				if (!(o.radius > 0.0) || !(o.height > 0.0)) err(who + "the radius and height must be above zero.");
				break;
			case ShapeKind::Box: if (!(o.size.x > 0.0 && o.size.y > 0.0 && o.size.z > 0.0)) err(who + "every box size must be above zero."); break;
			case ShapeKind::Wedge:
			case ShapeKind::Stairs:
				if (!(o.size.x > 0.0 && o.size.y > 0.0 && o.size.z > 0.0)) err(who + "every size must be above zero.");
				if (o.shape == ShapeKind::Stairs && (o.steps < 1 || o.steps > 100)) err(who + "the number of steps must be between 1 and 100.");
				break;
			case ShapeKind::Pyramid:
				if (!(o.size.x > 0.0 && o.size.z > 0.0 && o.height > 0.0)) err(who + "the pyramid's base and height must be above zero.");
				break;
			case ShapeKind::Dome: if (!(o.radius > 0.0)) err(who + "the radius must be above zero."); break;
			case ShapeKind::Capsule:
				if (!(o.radius > 0.0) || !(o.height > 0.0)) err(who + "the radius and height must be above zero.");
				else if (o.height < 2 * o.radius) warn(who + "the capsule is shorter than its two rounded ends, so it is as tall as they need (twice the radius).");
				break;
			case ShapeKind::Torus:
				if (!(o.radius > 0.0) || !(o.radius2 > 0.0)) err(who + "the ring and tube radii must be above zero.");
				else if (o.radius2 >= o.radius) err(who + "the tube must be thinner than the ring's radius, or the ring folds into itself.");
				break;
			case ShapeKind::Tube:
				if (!(o.radius > 0.0) || !(o.height > 0.0) || !(o.radius2 > 0.0)) err(who + "the radii and height must be above zero.");
				else if (o.radius2 >= o.radius) err(who + "the hole must be narrower than the outside.");
				break;
			case ShapeKind::Quad: if (!(o.size.x > 0.0 && o.size.z > 0.0)) err(who + "the quad's width and depth must be above zero."); break;
			case ShapeKind::Mesh:
				if (o.meshFile.empty()) err(who + "choose a mesh file (.ply or .obj).");
				if (!(o.meshScale > 0.0)) err(who + "the mesh scale must be above zero.");
				break;
		}
		if (o.emissive && o.emissionStrength <= 0.0) warn(who + "it is an area light with no strength, so it gives no light.");
		if (!o.material.imageFile.empty()) {
			if (o.material.kind != MaterialKind::Diffuse && o.material.kind != MaterialKind::CoatedDiffuse)
				warn(who + "a picture only applies to diffuse and glossy paint materials.");
			else if (o.material.checker) warn(who + "the picture replaces the checker pattern.");
			if (detail::absoluteFileMissing(o.material.imageFile))
				warn(who + "the picture file was not found (" + o.material.imageFile + "), so it will render without it. Choose it again.");
		}
		if (o.shape == ShapeKind::Mesh && detail::absoluteFileMissing(o.meshFile))
			warn(who + "the mesh file was not found (" + o.meshFile + "). Choose it again.");
		if (o.material.checker && o.material.kind != MaterialKind::Diffuse) warn(who + "the checker pattern only applies to diffuse materials.");
		if (o.material.checker && o.shape == ShapeKind::Box) warn(who + "the checker pattern is not applied to boxes.");
		if ((o.material.kind == MaterialKind::Dielectric || o.material.kind == MaterialKind::CoatedDiffuse) && !(o.material.ior >= 1.0 && o.material.ior <= 3.0))
			warn(who + "an index of refraction outside 1 to 3 is unusual.");
		if (o.material.roughness < 0.0 || o.material.roughness > 1.0) err(who + "the roughness must be between 0 and 1.");
		if (o.material.kind == MaterialKind::Diffuse || o.material.kind == MaterialKind::Conductor || o.material.kind == MaterialKind::CoatedDiffuse ||
		    o.material.kind == MaterialKind::DiffuseTransmission) {
			auto bad = [](const Rgb& c) { return c.r < 0 || c.g < 0 || c.b < 0 || c.r > 1 || c.g > 1 || c.b > 1; };
			if (bad(o.material.color)) warn(who + "a reflectance colour outside 0 to 1 gives more light out than in.");
		}
	}
	for (const Light& l : d.lights) {
		const std::string who = "'" + l.name + "': ";
		if (l.intensity < 0.0) err(who + "the light strength cannot be negative.");
		if (l.kind == LightKind::Infinite && detail::absoluteFileMissing(l.imageFile)) warn(who + "the sky image was not found (" + l.imageFile + "). Choose it again.");
		if (l.kind == LightKind::Spot && !(l.coneAngle > 0.0 && l.coneAngle < 90.0)) err(who + "the spot cone angle must be between 0 and 90 degrees.");
		if (l.kind == LightKind::Spot && (l.coneDelta < 0.0 || l.coneDelta > l.coneAngle)) err(who + "the spot soft edge must be between 0 and the cone angle.");
		if ((l.kind == LightKind::Spot || l.kind == LightKind::Distant) && l.position.x == l.target.x && l.position.y == l.target.y && l.position.z == l.target.z)
			err(who + "the light is at the point it aims at.");
	}
	if (!anyLight) warn("The scene has no light, so the picture will be black. Add a light or make an object emissive.");
	return out;
}

inline bool hasErrors(const std::vector<Problem>& ps) {
	return std::any_of(ps.begin(), ps.end(), [](const Problem& p) { return p.severity == Problem::Severity::Error; });
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Document -> pbrt-v4 text
// ---------------------------------------------------------------------------------------------------------------------------------
namespace detail {

// `textureName`, when not empty, is a Texture already declared: it replaces the flat reflectance of a diffuse or coated material.
inline void writeMaterial(std::ostringstream& os, const Material& m, const std::string& ind, const std::string& textureName = std::string()) {
	const auto reflectance = [&]() {
		return textureName.empty() ? "\"rgb reflectance\" [ " + rgb(m.color) + " ]" : "\"texture reflectance\" \"" + textureName + "\"";
	};
	switch (m.kind) {
		case MaterialKind::Diffuse:
			os << ind << "Material \"diffuse\" " << reflectance() << "\n";
			break;
		case MaterialKind::Conductor:
			os << ind << "Material \"conductor\" \"rgb reflectance\" [ " << rgb(m.color) << " ] \"float roughness\" [ " << num(m.roughness) << " ]\n";
			break;
		case MaterialKind::Dielectric:
			os << ind << "Material \"dielectric\" \"float eta\" [ " << num(m.ior) << " ]";
			if (m.roughness > 0.0) os << " \"float roughness\" [ " << num(m.roughness) << " ]";
			os << "\n";
			break;
		case MaterialKind::CoatedDiffuse:
			os << ind << "Material \"coateddiffuse\" " << reflectance() << " \"float eta\" [ " << num(m.ior) << " ] \"float roughness\" [ "
			   << num(m.roughness) << " ]\n";
			break;
		case MaterialKind::DiffuseTransmission:
			os << ind << "Material \"diffusetransmission\" \"rgb reflectance\" [ " << rgb(m.color) << " ] \"rgb transmittance\" [ " << rgb(m.transmittance)
			   << " ]\n";
			break;
	}
}

// Box faces (outward winding), 4 vertices each with their own uv so a texture can be applied per face.
inline void writeBox(std::ostringstream& os, const Float3& s, const std::string& ind) {
	const double hx = s.x / 2, hy = s.y / 2, hz = s.z / 2;
	struct Face { double v[4][3]; };
	const Face faces[6] = {
		{{{ hx, -hy, -hz}, { hx,  hy, -hz}, { hx,  hy,  hz}, { hx, -hy,  hz}}},   // +X
		{{{-hx, -hy,  hz}, {-hx,  hy,  hz}, {-hx,  hy, -hz}, {-hx, -hy, -hz}}},   // -X
		{{{-hx,  hy, -hz}, {-hx,  hy,  hz}, { hx,  hy,  hz}, { hx,  hy, -hz}}},   // +Y
		{{{-hx, -hy,  hz}, {-hx, -hy, -hz}, { hx, -hy, -hz}, { hx, -hy,  hz}}},   // -Y
		{{{-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz}}},   // +Z
		{{{ hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx,  hy, -hz}, { hx,  hy, -hz}}},   // -Z
	};
	os << ind << "Shape \"trianglemesh\" \"integer indices\" [";
	for (int f = 0; f < 6; ++f) os << " " << f * 4 << " " << f * 4 + 1 << " " << f * 4 + 2 << " " << f * 4 << " " << f * 4 + 2 << " " << f * 4 + 3;
	os << " ]\n" << ind << "  \"point3 P\" [";
	for (const Face& f : faces) for (const auto& v : f.v) os << " " << num(v[0]) << " " << num(v[1]) << " " << num(v[2]);
	os << " ]\n" << ind << "  \"point2 uv\" [";
	for (int f = 0; f < 6; ++f) os << " 0 0 1 0 1 1 0 1";
	os << " ]\n";
}

inline ShapeMesh meshOfShape(const Object& o) {
	switch (o.shape) {
		case ShapeKind::Pyramid: return pyramidMesh(o.size.x, o.height, o.size.z);
		case ShapeKind::Wedge: return wedgeMesh(o.size.x, o.size.y, o.size.z);
		case ShapeKind::Stairs: return stairsMesh(o.size.x, o.size.y, o.size.z, o.steps);
		case ShapeKind::Torus: return torusMesh(o.radius, o.radius2);
		case ShapeKind::Capsule: return capsuleMesh(o.radius, o.height);
		case ShapeKind::Dome: return domeMesh(o.radius);
		case ShapeKind::Tube: return tubeMesh(o.radius, o.radius2, o.height);
		default: break;
	}
	return ShapeMesh();
}

inline void writeGenerated(std::ostringstream& os, const ShapeMesh& m, const std::string& ind) {
	os << ind << "Shape \"trianglemesh\" \"integer indices\" [";
	for (std::size_t i = 0; i < m.indices.size(); ++i) os << (i % 12 == 0 ? "\n" + ind + "    " : " ") << m.indices[i];
	os << " ]\n" << ind << "  \"point3 P\" [";
	for (std::size_t i = 0; i < m.P.size(); ++i) os << (i % 12 == 0 ? "\n" + ind + "    " : " ") << num(m.P[i]);
	os << " ]\n" << ind << "  \"normal N\" [";
	for (std::size_t i = 0; i < m.N.size(); ++i) os << (i % 12 == 0 ? "\n" + ind + "    " : " ") << num(m.N[i]);
	os << " ]\n" << ind << "  \"point2 uv\" [";
	for (std::size_t i = 0; i < m.UV.size(); ++i) os << (i % 12 == 0 ? "\n" + ind + "    " : " ") << num(m.UV[i]);
	os << " ]\n";
}

// `pictured`: the surface is coloured by a picture. A quad's texture coordinates then run so the picture is upright and not mirrored when the quad is
// stood up facing +Z (rotation X = 90) and seen from +Z, i.e. its top is the quad's -Z edge. (Without a picture they keep the checker's orientation.)
inline void writeShape(std::ostringstream& os, const Object& o, const std::string& ind, bool pictured = false) {
	switch (o.shape) {
		case ShapeKind::Sphere:
			os << ind << "Shape \"sphere\" \"float radius\" [ " << num(o.radius) << " ]\n";
			break;
		case ShapeKind::Box:
			writeBox(os, o.size, ind);
			break;
		case ShapeKind::Quad: {
			const double hx = o.size.x / 2, hz = o.size.z / 2;
			// Winding chosen so the front (and the emitting side) faces +Y.
			os << ind << "Shape \"trianglemesh\" \"integer indices\" [ 0 2 1 0 3 2 ]\n"
			   << ind << "  \"point3 P\" [ " << num(-hx) << " 0 " << num(-hz) << "  " << num(hx) << " 0 " << num(-hz) << "  " << num(hx) << " 0 " << num(hz)
			   << "  " << num(-hx) << " 0 " << num(hz) << " ]\n"
			   << ind << "  \"point2 uv\" [ " << (pictured ? "0 1 1 1 1 0 0 0" : "0 0 1 0 1 1 0 1") << " ]\n";
			break;
		}
		case ShapeKind::Disk:
			os << ind << "Rotate -90 1 0 0\n" << ind << "Shape \"disk\" \"float radius\" [ " << num(o.radius) << " ]\n";
			break;
		case ShapeKind::Cylinder:
			os << ind << "Rotate -90 1 0 0\n"
			   << ind << "Shape \"cylinder\" \"float radius\" [ " << num(o.radius) << " ] \"float zmin\" [ " << num(-o.height / 2) << " ] \"float zmax\" [ "
			   << num(o.height / 2) << " ]\n";
			break;
		case ShapeKind::Cone:
			os << ind << "Rotate -90 1 0 0\n" << ind << "Translate 0 0 " << num(-o.height / 2) << "\n"
			   << ind << "Shape \"cone\" \"float radius\" [ " << num(o.radius) << " ] \"float height\" [ " << num(o.height) << " ]\n";
			break;
		case ShapeKind::Pyramid:
		case ShapeKind::Wedge:
		case ShapeKind::Stairs:
		case ShapeKind::Torus:
		case ShapeKind::Capsule:
		case ShapeKind::Dome:
		case ShapeKind::Tube:
			writeGenerated(os, meshOfShape(o), ind);
			break;
		case ShapeKind::Mesh:
			os << ind << "Scale " << num(o.meshScale) << " " << num(o.meshScale) << " " << num(o.meshScale) << "\n"
			   << ind << "Shape \"plymesh\" \"string filename\" [ " << quoted(pbrtPath(o.meshFile)) << " ]\n";
			break;
	}
}

}  // namespace detail

// The triangle mesh of one of the generated shapes (isGeneratedShape), in the object's own space; the GUI's previews draw it too.
inline ShapeMesh generatedMesh(const Object& o) { return detail::meshOfShape(o); }

inline std::string toPbrt(const Document& d) {
	std::ostringstream os;
	os.imbue(std::locale::classic());
	os << "# " << detail::commentText(d.title) << "\n";
	os << "# Written by the ray_tracer Scene Builder. It is an ordinary pbrt-v4 scene; the line below lets the Scene Builder open it again for editing.\n";
	os << "# @rt-builder-doc " << toJson(d) << "\n\n";

	const Camera& c = d.camera;
	os << "LookAt " << detail::vec(c.position) << "  " << detail::vec(c.target) << "  " << detail::vec(c.up) << "\n";
	os << "Camera \"perspective\" \"float fov\" [ " << detail::num(c.fov) << " ]";
	if (c.lensRadius > 0.0) os << " \"float lensradius\" [ " << detail::num(c.lensRadius) << " ] \"float focaldistance\" [ " << detail::num(c.focusDistance) << " ]";
	os << "\n";
	os << "Film \"rgb\" \"integer xresolution\" [ " << d.render.width << " ] \"integer yresolution\" [ " << d.render.height << " ]\n";
	os << "Sampler \"sobol\" \"integer pixelsamples\" [ " << d.render.samples << " ]\n";
	os << "Integrator \"volpath\" \"integer maxdepth\" [ " << d.render.maxDepth << " ]\n\n";
	os << "WorldBegin\n\n";

	for (const Light& l : d.lights) {
		os << "# " << detail::commentText(l.name) << "\n";
		const Rgb col = detail::scaled(l.color, l.intensity);
		switch (l.kind) {
			case LightKind::Point:
				os << "LightSource \"point\" \"point3 from\" [ " << detail::vec(l.position) << " ] \"rgb I\" [ " << detail::rgb(col) << " ]\n";
				break;
			case LightKind::Spot:
				os << "LightSource \"spot\" \"point3 from\" [ " << detail::vec(l.position) << " ] \"point3 to\" [ " << detail::vec(l.target)
				   << " ] \"float coneangle\" [ " << detail::num(l.coneAngle) << " ] \"float conedeltaangle\" [ " << detail::num(l.coneDelta)
				   << " ] \"rgb I\" [ " << detail::rgb(col) << " ]\n";
				break;
			case LightKind::Distant:
				os << "LightSource \"distant\" \"point3 from\" [ " << detail::vec(l.position) << " ] \"point3 to\" [ " << detail::vec(l.target)
				   << " ] \"rgb L\" [ " << detail::rgb(col) << " ]\n";
				break;
			case LightKind::Infinite:
				if (!l.imageFile.empty())
					os << "LightSource \"infinite\" \"string filename\" [ " << detail::quoted(detail::pbrtPath(l.imageFile)) << " ] \"float scale\" [ "
					   << detail::num(l.intensity) << " ]\n";
				else
					os << "LightSource \"infinite\" \"rgb L\" [ " << detail::rgb(col) << " ]\n";
				break;
		}
	}
	if (!d.lights.empty()) os << "\n";

	int textureIndex = 0;
	for (const Object& o : d.objects) {
		os << "# " << detail::commentText(o.name) << "\n";
		std::string textureName;
		const bool pictured = !o.material.imageFile.empty() && (o.material.kind == MaterialKind::Diffuse || o.material.kind == MaterialKind::CoatedDiffuse);
		if (pictured) {
			textureName = "picture-" + std::to_string(++textureIndex);
			os << "Texture \"" << textureName << "\" \"spectrum\" \"imagemap\" \"string filename\" [ " << detail::quoted(detail::pbrtPath(o.material.imageFile))
			   << " ]\n";
		} else if (o.material.checker && o.material.kind == MaterialKind::Diffuse && o.shape != ShapeKind::Box) {
			textureName = "checks-" + std::to_string(++textureIndex);
			os << "Texture \"" << textureName << "\" \"spectrum\" \"checkerboard\" \"float uscale\" [ " << detail::num(o.material.checkerCount)
			   << " ] \"float vscale\" [ " << detail::num(o.material.checkerCount) << " ] \"rgb tex1\" [ " << detail::rgb(o.material.color)
			   << " ] \"rgb tex2\" [ " << detail::rgb(o.material.color2) << " ]\n";
		}
		os << "AttributeBegin\n";
		const std::string ind = "  ";
		os << ind << "Translate " << detail::vec(o.position) << "\n";
		if (o.rotation.z != 0.0) os << ind << "Rotate " << detail::num(o.rotation.z) << " 0 0 1\n";
		if (o.rotation.y != 0.0) os << ind << "Rotate " << detail::num(o.rotation.y) << " 0 1 0\n";
		if (o.rotation.x != 0.0) os << ind << "Rotate " << detail::num(o.rotation.x) << " 1 0 0\n";
		detail::writeMaterial(os, o.material, ind, textureName);
		if (o.emissive) {
			os << ind << "AreaLightSource \"diffuse\" \"rgb L\" [ " << detail::rgb(detail::scaled(o.emission, o.emissionStrength)) << " ]";
			if (o.twoSided) os << " \"bool twosided\" true";
			os << "\n";
		}
		detail::writeShape(os, o, ind, pictured);
		os << "AttributeEnd\n\n";
	}
	return os.str();
}

// Reads the document back out of text written by toPbrt(). A .pbrt file without the builder line is not a builder scene.
inline bool fromPbrt(const std::string& text, Document& out, std::string& err) {
	// The marker starts a line (toPbrt() never lets a name put it anywhere else; see detail::commentText).
	const std::string marker = "# @rt-builder-doc ";
	size_t at = text.compare(0, marker.size(), marker) == 0 ? 0 : text.find("\n" + marker);
	if (at != std::string::npos && at != 0) ++at;
	if (at == std::string::npos) {
		err = "this file was not made by the Scene Builder (it has no builder data), so it can be rendered but not edited here";
		return false;
	}
	size_t begin = at + marker.size();
	size_t end = text.find_first_of("\r\n", begin);
	if (end == std::string::npos) end = text.size();
	return fromJson(text.substr(begin, end - begin), out, err);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Starting points
// ---------------------------------------------------------------------------------------------------------------------------------
inline Object makeObject(ShapeKind shape, const std::string& name) {
	Object o;
	o.shape = shape;
	o.name = name;
	switch (shape) {
		case ShapeKind::Sphere: o.radius = 1.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Box: o.size = {1.5, 1.5, 1.5}; o.position = {0, 0.75, 0}; break;
		case ShapeKind::Quad: o.size = {6, 1, 6}; o.position = {0, 0, 0}; break;
		case ShapeKind::Disk: o.radius = 1.5; o.position = {0, 0, 0}; break;
		case ShapeKind::Cylinder: o.radius = 0.6; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Cone: o.radius = 0.8; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Mesh: o.meshScale = 1.0; break;
		case ShapeKind::Pyramid: o.size = {1.6, 1.6, 1.6}; o.height = 1.8; o.position = {0, 0.9, 0}; break;
		case ShapeKind::Wedge: o.size = {2.0, 1.0, 2.0}; o.position = {0, 0.5, 0}; break;
		case ShapeKind::Stairs: o.size = {2.0, 1.5, 2.5}; o.steps = 5; o.position = {0, 0.75, 0}; break;
		case ShapeKind::Torus: o.radius = 1.0; o.radius2 = 0.3; o.position = {0, 0.3, 0}; break;
		case ShapeKind::Capsule: o.radius = 0.5; o.height = 2.0; o.position = {0, 1.0, 0}; break;
		case ShapeKind::Dome: o.radius = 1.2; o.position = {0, 0.6, 0}; break;
		case ShapeKind::Tube: o.radius = 0.7; o.radius2 = 0.5; o.height = 1.6; o.position = {0, 0.8, 0}; break;
	}
	return o;
}

// A light panel: a quad that faces down, to hang above a scene.
inline Object makeAreaLightPanel(const std::string& name) {
	Object o = makeObject(ShapeKind::Quad, name);
	o.size = {2.0, 1.0, 2.0};
	o.position = {0, 5.0, 0};
	o.rotation = {180.0, 0.0, 0.0};
	o.emissive = true;
	o.emission = {1.0, 1.0, 1.0};
	o.emissionStrength = 12.0;
	o.material.color = {0.0, 0.0, 0.0};
	return o;
}

// What "New scene" opens: a checkered floor, a glass sphere, a gold sphere, a red box, a panel light and a faint sky: it renders to something nice at once.
inline Document makeStarterScene() {
	Document d;
	d.title = "My scene";

	Object floorObj = makeObject(ShapeKind::Quad, "Floor");
	floorObj.size = {14, 1, 14};
	floorObj.material.checker = true;
	floorObj.material.color = {0.82, 0.82, 0.82};
	floorObj.material.color2 = {0.25, 0.27, 0.30};
	floorObj.material.checkerCount = 10;
	d.objects.push_back(floorObj);

	Object glass = makeObject(ShapeKind::Sphere, "Glass ball");
	glass.position = {-1.3, 1.0, 0.4};
	glass.material.kind = MaterialKind::Dielectric;
	d.objects.push_back(glass);

	Object gold = makeObject(ShapeKind::Sphere, "Gold ball");
	gold.radius = 0.8;
	gold.position = {1.4, 0.8, -0.3};
	gold.material.kind = MaterialKind::Conductor;
	gold.material.color = {0.95, 0.72, 0.30};
	gold.material.roughness = 0.12;
	d.objects.push_back(gold);

	Object box = makeObject(ShapeKind::Box, "Red box");
	box.size = {1.0, 1.0, 1.0};
	box.position = {0.1, 0.5, -1.8};
	box.rotation = {0, 28, 0};
	box.material.color = {0.75, 0.12, 0.10};
	d.objects.push_back(box);

	d.objects.push_back(makeAreaLightPanel("Ceiling light"));

	Light sky;
	sky.name = "Sky";
	sky.kind = LightKind::Infinite;
	sky.color = {0.55, 0.70, 1.0};
	sky.intensity = 0.35;
	d.lights.push_back(sky);
	return d;
}

}  // namespace scene_doc
