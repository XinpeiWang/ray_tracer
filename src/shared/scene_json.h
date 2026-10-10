#pragma once
// scene_json.h -- number formatting, a tiny JSON reader/writer and the document <-> JSON conversion (the copy of the whole document a builder file carries).
//
// Part of the Scene Builder's document (scene_document.h includes all of it); header-only, standard library only.

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

#include "scene_model.h"

namespace scene_doc {

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
		if (!o.group.empty()) j.set("group", Json::string(o.group));
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
		if (l.physicalSky) {   // (older files and ordinary lights do not have these)
			j.set("physicalSky", Json::boolean(true)).set("sunElevation", Json::number(l.sky.sunElevation)).set("sunAzimuth", Json::number(l.sky.sunAzimuth));
			j.set("turbidity", Json::number(l.sky.turbidity)).set("groundAlbedo", Json::number(l.sky.groundAlbedo));
		}
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
			     detail::readNum(j, "emissionStrength", o.emissionStrength) && detail::readBool(j, "twoSided", o.twoSided) && detail::readStr(j, "group", o.group);
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
			     detail::readNum(j, "coneDelta", l.coneDelta) && detail::readStr(j, "imageFile", l.imageFile) && detail::readBool(j, "physicalSky", l.physicalSky) &&
			     detail::readNum(j, "sunElevation", l.sky.sunElevation) && detail::readNum(j, "sunAzimuth", l.sky.sunAzimuth) &&
			     detail::readNum(j, "turbidity", l.sky.turbidity) && detail::readNum(j, "groundAlbedo", l.sky.groundAlbedo);
			if (!ok) break;
			d.lights.push_back(l);
		}
	}
	if (!ok) { err = "the scene description has a field of the wrong type or an unknown name"; return false; }
	out = std::move(d);
	return true;
}

}  // namespace scene_doc
