#pragma once
// pbrt_scene.h -- parser for a subset of the pbrt-v4 scene description format.
//
// WHY PBRT'S FORMAT RATHER THAN OUR OWN
// -------------------------------------
// Every one of the 78 built-in scenes is a C++ builder in scene_registry.h
// plus a hand-written case in gpu/optix/scene_builder.cpp, with the GPU camera
// block copied from the CPU CameraConfig by hand (the comments there say so).
// Adding a scene means editing two files and rebuilding.
//
// Inventing a text format would fix that but load nobody else's scenes. pbrt's
// format costs more to parse and buys an ecosystem: Benedikt Bitterli's
// rendering resources - the scenes GLSL-PathTracer and most hobby renderers
// draw from - are published in pbrt-v4 form directly, as are the 28 scenes in
// mmp/pbrt-v4-scenes.
//
// It also happens to fit this renderer unusually well. Our MaterialType enum
// already IS pbrt-v4's material set under pbrt's own names - conductor,
// coateddiffuse, coatedconductor, thindielectric, diffusetransmission, hair,
// measured, mix - so `Material "coateddiffuse"` maps across without a
// translation layer.
//
// SCOPE, AND WHAT HAPPENS TO THE REST
// -----------------------------------
// This is a SUBSET. Unrecognised directives are recorded as a warning naming
// the directive and line, then skipped - they do not fail the load. That is
// deliberate: a real pbrt scene will always contain something we do not
// implement yet (subsurface, curves, instancing, media), and a parser that
// refuses the whole file over one unknown directive is useless for adopting
// scenes incrementally. A scene that renders with three materials approximated
// is worth more than one that will not open.
//
// Parsing only. This produces a description and nothing else: no hittables, no
// CUDA, no textures loaded from disk. That keeps it compilable by the MSVC
// test binary (the Qt GUI is MinGW-only) and lets both backends consume one
// description instead of each parsing their own - the same reasoning behind
// render_output_parser.h, camera_math.h and palette_data.h.

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace pbrt_scene {

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------
// Row-major 4x4. pbrt's CTM is composed right-to-left as directives are read,
// so Translate then Rotate means "translate, then rotate in the translated
// frame" - i.e. new = current * delta, not delta * current. Getting that
// backwards silently mirrors or misplaces every transformed shape.
struct Matrix4 {
	double m[16] = {1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1};

	static Matrix4 identity() { return Matrix4{}; }

	Matrix4 operator*(const Matrix4 &o) const {
		Matrix4 r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j) {
				double s = 0.0;
				for (int k = 0; k < 4; ++k) s += m[i * 4 + k] * o.m[k * 4 + j];
				r.m[i * 4 + j] = s;
			}
		return r;
	}

	// The inverse of an affine transform, which is what a renderer needs to
	// carry a ray INTO an instance's object space. Returns false when the
	// matrix is singular - a scale of zero on some axis - rather than
	// producing infinities that would surface later as missing geometry.
	//
	// Affine only, and deliberately: every transform pbrt can build here has
	// (0,0,0,1) as its bottom row, so inverting the 3x3 and negating the
	// translated origin is both exact and much cheaper than a general 4x4
	// inverse.  [A t]^-1 = [A^-1  -A^-1 t]
	//                      [0 1]   [0        1     ]
	bool inverseAffine(Matrix4 &out) const {
		const double a = m[0], b = m[1], c = m[2];
		const double d = m[4], e = m[5], f = m[6];
		const double g = m[8], h = m[9], i = m[10];

		const double A =  (e * i - f * h);
		const double B = -(d * i - f * g);
		const double C =  (d * h - e * g);
		const double det = a * A + b * B + c * C;
		if (!(det > 1e-300 || det < -1e-300)) return false;
		const double inv = 1.0 / det;

		// Adjugate divided by the determinant, written out transposed.
		const double n0 = A * inv,                    n1 = -(b * i - c * h) * inv, n2 =  (b * f - c * e) * inv;
		const double n4 = B * inv,                    n5 =  (a * i - c * g) * inv, n6 = -(a * f - c * d) * inv;
		const double n8 = C * inv,                    n9 = -(a * h - b * g) * inv, n10 = (a * e - b * d) * inv;

		const double tx = m[3], ty = m[7], tz = m[11];
		out.m[0] = n0; out.m[1] = n1; out.m[2]  = n2;
		out.m[4] = n4; out.m[5] = n5; out.m[6]  = n6;
		out.m[8] = n8; out.m[9] = n9; out.m[10] = n10;
		out.m[3]  = -(n0 * tx + n1 * ty + n2  * tz);
		out.m[7]  = -(n4 * tx + n5 * ty + n6  * tz);
		out.m[11] = -(n8 * tx + n9 * ty + n10 * tz);
		out.m[12] = 0; out.m[13] = 0; out.m[14] = 0; out.m[15] = 1;
		return true;
	}

	static Matrix4 translate(double x, double y, double z) {
		Matrix4 r;
		r.m[3] = x; r.m[7] = y; r.m[11] = z;
		return r;
	}

	static Matrix4 scale(double x, double y, double z) {
		Matrix4 r;
		r.m[0] = x; r.m[5] = y; r.m[10] = z;
		return r;
	}

	// Angle in DEGREES about an arbitrary axis, matching pbrt's Rotate.
	static Matrix4 rotate(double degrees, double ax, double ay, double az) {
		const double len = std::sqrt(ax * ax + ay * ay + az * az);
		Matrix4 r;
		if (len == 0.0) return r;   // degenerate axis: no-op rather than NaN
		ax /= len; ay /= len; az /= len;
		const double rad = degrees * 3.14159265358979323846 / 180.0;
		const double s = std::sin(rad), c = std::cos(rad);
		r.m[0]  = ax * ax + (1 - ax * ax) * c;
		r.m[1]  = ax * ay * (1 - c) - az * s;
		r.m[2]  = ax * az * (1 - c) + ay * s;
		r.m[4]  = ax * ay * (1 - c) + az * s;
		r.m[5]  = ay * ay + (1 - ay * ay) * c;
		r.m[6]  = ay * az * (1 - c) - ax * s;
		r.m[8]  = ax * az * (1 - c) - ay * s;
		r.m[9]  = ay * az * (1 - c) + ax * s;
		r.m[10] = az * az + (1 - az * az) * c;
		return r;
	}

	// pbrt's LookAt yields WORLD-TO-CAMERA. The camera-to-world basis is built
	// first and then inverted by transposing the rotation and negating the
	// translated origin - valid because the basis is orthonormal.
	static Matrix4 lookAt(double ex, double ey, double ez,
						  double lx, double ly, double lz,
						  double ux, double uy, double uz) {
		double dx = lx - ex, dy = ly - ey, dz = lz - ez;
		double dlen = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (dlen == 0.0) return Matrix4::identity();
		dx /= dlen; dy /= dlen; dz /= dlen;

		// right = normalize(cross(normalize(up), dir))
		double ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
		if (ulen == 0.0) return Matrix4::identity();
		ux /= ulen; uy /= ulen; uz /= ulen;
		double rx = uy * dz - uz * dy;
		double ry = uz * dx - ux * dz;
		double rz = ux * dy - uy * dx;
		const double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
		if (rlen == 0.0) return Matrix4::identity();   // up parallel to dir
		rx /= rlen; ry /= rlen; rz /= rlen;

		// newUp = cross(dir, right)
		const double nux = dy * rz - dz * ry;
		const double nuy = dz * rx - dx * rz;
		const double nuz = dx * ry - dy * rx;

		Matrix4 w2c;
		w2c.m[0] = rx;  w2c.m[1] = ry;  w2c.m[2] = rz;
		w2c.m[4] = nux; w2c.m[5] = nuy; w2c.m[6] = nuz;
		w2c.m[8] = dx;  w2c.m[9] = dy;  w2c.m[10] = dz;
		w2c.m[3]  = -(rx * ex + ry * ey + rz * ez);
		w2c.m[7]  = -(nux * ex + nuy * ey + nuz * ez);
		w2c.m[11] = -(dx * ex + dy * ey + dz * ez);
		return w2c;
	}
};

struct Vec3 {
	double x = 0.0, y = 0.0, z = 0.0;
};

// ---------------------------------------------------------------------------
// Typed parameter lists:  "float radius" [ 0.25 ]
// ---------------------------------------------------------------------------
struct Param {
	std::string type;    // float, integer, point3, rgb, string, bool, texture, ...
	std::string name;
	std::vector<double> numbers;
	std::vector<std::string> strings;
};

struct ParamList {
	std::vector<Param> items;

	const Param *find(const std::string &name) const {
		for (const Param &p : items) if (p.name == name) return &p;
		return nullptr;
	}
	double getFloat(const std::string &name, double def) const {
		const Param *p = find(name);
		return (p && !p->numbers.empty()) ? p->numbers[0] : def;
	}
	int getInt(const std::string &name, int def) const {
		const Param *p = find(name);
		return (p && !p->numbers.empty()) ? static_cast<int>(p->numbers[0]) : def;
	}
	std::string getString(const std::string &name, const std::string &def) const {
		const Param *p = find(name);
		return (p && !p->strings.empty()) ? p->strings[0] : def;
	}
	bool getBool(const std::string &name, bool def) const {
		const Param *p = find(name);
		if (!p || p->strings.empty()) return def;
		return p->strings[0] == "true";
	}
	Vec3 getVec3(const std::string &name, Vec3 def) const {
		const Param *p = find(name);
		if (!p || p->numbers.size() < 3) return def;
		return Vec3{p->numbers[0], p->numbers[1], p->numbers[2]};
	}
};

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------
struct MaterialDecl {
	std::string name;    // empty for an unnamed `Material` directive
	std::string type;    // "diffuse", "conductor", "coateddiffuse", ...
	ParamList params;
};

struct TextureDecl {
	std::string name;
	std::string dataType;   // "float" or "spectrum"
	std::string cls;        // "imagemap", "checkerboard", "scale", ...
	ParamList params;
};

struct LightDecl {
	std::string type;    // "infinite", "distant", "point", "spot" / area: "diffuse"
	ParamList params;
	Matrix4 xform;
	// The working color space (GraphicsState::colorSpaceName's own comment)
	// in effect when this light was declared - captured here, not read from
	// a single scene-wide setting, so a ColorSpace directive scoped to one
	// AttributeBegin/End block affects only the lights declared inside it.
	std::string colorSpaceName = "srgb";
};

// MakeNamedMedium - a participating medium (fog/smoke), named so a later
// MediumInterface can refer back to it. Only "homogeneous" (pbrt-v4's
// constant-density medium) is recognized; anything else falls through to
// Scene::media as a decl whose params the CPU/GPU builders will not
// recognize (same "not implemented, skipped" posture as an unhandled Shape).
struct MediumDecl {
	std::string name;
	std::string type = "homogeneous";
	ParamList params;
	// The CTM active at MakeNamedMedium declaration time - pbrt-v4's own
	// "renderFromMedium" (BasicSceneBuilder::MakeNamedMedium,
	// pbrt-v4/src/pbrt/scene.cpp), needed for a heterogeneous medium
	// (cloud/rgbgrid) whose density lives in its own medium-space box, not
	// world space. Same capture-at-declaration convention as ShapeDecl::
	// xform/LightDecl::xform just below - identity if the medium was
	// declared with no transform active.
	Matrix4 xform;
};

struct ShapeDecl {
	std::string type;    // "sphere", "trianglemesh", "plymesh", "disk", ...
	ParamList params;
	Matrix4 xform;
	int materialIndex = -1;    // index into Scene::materials, -1 = pbrt's default
	int areaLightIndex = -1;   // index into Scene::areaLights, -1 = not emissive
	// index into Scene::media, -1 = vacuum. Only the medium a ray enters by
	// crossing INTO this shape is tracked - "outsideMedium" (ambient fog a
	// camera ray already starts inside, before hitting anything) would need
	// a scene-wide default medium threaded through every ray's origin, a
	// materially bigger feature than "one more per-shape index" and not
	// needed by any scene this loader has actually seen.
	int insideMedium = -1;
	bool reverseOrientation = false;
};

struct Warning {
	int line = 0;
	std::string file;      // empty for the top-level scene
	std::string message;
};

// Supplies the contents of an Include'd file. Returning false means "not
// found", which is an error rather than a warning: a scene whose geometry
// lives in an included file loads as an empty scene otherwise, and an empty
// render is a far more confusing symptom than a refusal naming the path.
//
// A callback rather than direct file I/O so the parser stays a pure function -
// the tests exercise nesting, cycles and missing files from an in-memory map,
// and the caller decides what "a path" means (working directory, scene
// directory, an archive).
using FileResolver = std::function<bool(const std::string &path, std::string &outContents)>;

// A named collection of shapes that exists once and is placed many times.
struct ObjectDecl {
	std::string name;
	std::vector<ShapeDecl> shapes;
};

// One placement of an ObjectDecl.
struct InstanceDecl {
	std::string name;
	Matrix4 xform;      // the CTM where ObjectInstance appeared
	int line = 0;       // so an instance of an undefined object can be located
};

struct Scene {
	// Pre-world state.
	std::string cameraType = "perspective";
	ParamList cameraParams;
	Matrix4 worldToCamera;      // the CTM at WorldBegin
	// Whether a Camera directive was actually seen, as opposed to cameraType/
	// cameraParams/worldToCamera just sitting at their defaults. Distinguishes
	// a real top-level scene from a fragment (a MakeNamedMaterial/geometry
	// library meant only to be Include'd) that happens to parse cleanly -
	// see pbrt_discover.h's scanDirectory(), the only reader of this field.
	bool cameraDeclared = false;
	int xResolution = 1280;
	int yResolution = 720;
	std::string filmFilename;
	std::string integrator = "volpath";
	// Sampler directive's type name (e.g. "sobol", "zsobol", "halton") -
	// like `integrator` above, purely informational: nothing in this parser
	// or the builders it feeds selects a sampler implementation from this
	// string. "sobol" is both pbrt-v4's actual default AND what this
	// project's own CPU render loop already hardcoded before SamplerKind
	// existed (camera.h), so a scene with no Sampler directive at all
	// reports the same value a render already produces.
	std::string samplerType = "sobol";
	int samplesPerPixel = 16;
	int maxDepth = 5;
	// PixelFilter directive's type name (pbrt-v4's real default is
	// "gaussian" - confirmed against pbrt-v4/src/pbrt/scene.cpp's own
	// SceneEntity default, NOT "box"/"triangle"/"mitchell") plus its raw
	// params (radius/xradius/yradius/b/c/sigma/tau, kind-dependent) -
	// unlike samplerType/integrator above, this ONE is actually applied
	// (see camera.h's own filter dispatch), not just advisory: swapping
	// which of this project's existing filter.h classes gets used doesn't
	// change which C++ code path executes the way selecting a different
	// Sampler/Integrator implementation would, so there's no reason to
	// gate it behind a separate CLI opt-in the way those stay gated.
	std::string filterType = "gaussian";
	ParamList filterParams;

	std::vector<MaterialDecl> materials;
	std::vector<TextureDecl> textures;
	std::vector<LightDecl> lights;       // LightSource
	std::vector<LightDecl> areaLights;   // AreaLightSource
	std::vector<MediumDecl> media;       // MakeNamedMedium
	std::vector<ShapeDecl> shapes;

	// An instance definition: shapes recorded between ObjectBegin and
	// ObjectEnd, which are NOT part of the scene until an ObjectInstance
	// places them. Their xform is the CTM where each shape was declared,
	// matching pbrt - the instance's own transform is applied on top, so a
	// definition opened under a transform has that transform applied twice.
	// Odd, but it is what published scenes are written against.
	std::vector<ObjectDecl> objects;
	std::vector<InstanceDecl> instances;

	// Directives recognised as pbrt but not implemented here. Never fatal -
	// see the header comment.
	std::vector<Warning> warnings;

	double cameraFov() const { return cameraParams.getFloat("fov", 90.0); }
};

struct ParseResult {
	bool ok = false;
	std::string error;   // includes the line number
	int line = 0;
	Scene scene;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
namespace detail {

struct Token {
	std::string text;
	int line = 0;
	bool quoted = false;
	int file = 0;      // index into TokenStream::files; 0 = the top-level scene
};

inline void tokenizeInto(const std::string &src, int fileIndex, std::vector<Token> &out) {
	int line = 1;
	std::size_t i = 0;
	while (i < src.size()) {
		const char c = src[i];
		if (c == '\n') { ++line; ++i; continue; }
		if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
		if (c == '#') {                       // comment to end of line
			while (i < src.size() && src[i] != '\n') ++i;
			continue;
		}
		if (c == '[' || c == ']') {
			out.push_back({std::string(1, c), line, false, fileIndex});
			++i;
			continue;
		}
		if (c == '"') {
			const int start = line;
			std::string text;
			++i;
			while (i < src.size() && src[i] != '"') {
				if (src[i] == '\n') ++line;
				text += src[i++];
			}
			if (i < src.size()) ++i;
			out.push_back({text, start, true, fileIndex});
			continue;
		}
		std::string word;
		while (i < src.size() && !std::isspace(static_cast<unsigned char>(src[i]))
			   && src[i] != '"' && src[i] != '[' && src[i] != ']' && src[i] != '#') {
			word += src[i++];
		}
		out.push_back({word, line, false, fileIndex});
	}
}

// strtod, not stod: no exceptions, and the end pointer proves the WHOLE token
// was a number so "1.5x" is rejected rather than silently read as 1.5.
inline bool toDouble(const std::string &s, double &out) {
	if (s.empty()) return false;
	char *end = nullptr;
	const double v = std::strtod(s.c_str(), &end);
	if (end != s.c_str() + s.size()) return false;
	out = v;
	return true;
}

// Everything the parser sees, after includes have been spliced in.
struct TokenStream {
	std::vector<Token> tokens;
	std::vector<std::string> files{""};   // index 0 is the top-level scene
};

// Splices Include/Import contents into the token stream at the point they
// appear, which is exactly pbrt's textual semantics - the included file
// inherits the current graphics state and can leave it modified.
//
// Depth and cycle guards are not paranoia: a file that includes itself, or a
// pair that include each other, otherwise recurses until the stack dies, and
// scene archives really do contain leftover self-references.
inline bool expandIncludes(const std::string &text, int fileIndex,
						   TokenStream &out, const FileResolver &resolver,
						   std::vector<std::string> &openStack,
						   std::string &error) {
	if (openStack.size() > 32) {
		error = "Include nesting deeper than 32 files - probably a cycle";
		return false;
	}

	std::vector<Token> local;
	tokenizeInto(text, fileIndex, local);

	for (std::size_t i = 0; i < local.size(); ++i) {
		const bool isInclude = !local[i].quoted
							   && (local[i].text == "Include" || local[i].text == "Import");
		if (!isInclude) { out.tokens.push_back(local[i]); continue; }

		if (i + 1 >= local.size() || !local[i + 1].quoted) {
			error = "line " + std::to_string(local[i].line) + ": "
					+ local[i].text + " needs a quoted filename";
			return false;
		}
		const std::string path = local[i + 1].text;
		++i;   // consume the filename

		// No resolver: the caller is parsing text in isolation and there is
		// nothing to read from. Drop the directive rather than failing - this
		// is the deliberate choice of a caller who passed no resolver, not a
		// broken scene.
		if (!resolver) continue;

		for (const std::string &open : openStack) {
			if (open == path) {
				error = "Include cycle: '" + path + "' includes itself";
				return false;
			}
		}

		std::string contents;
		if (!resolver(path, contents)) {
			error = "cannot open included file '" + path + "'";
			return false;
		}

		out.files.push_back(path);
		const int childIndex = static_cast<int>(out.files.size()) - 1;
		openStack.push_back(path);
		if (!expandIncludes(contents, childIndex, out, resolver, openStack, error)) return false;
		openStack.pop_back();
	}
	return true;
}

// What counts as a directive, and how we resynchronise after skipping one.
//
// The rule is structural, not a fixed list of names: in pbrt every directive
// is an unquoted, capitalised identifier, while every parameter is a quoted
// "type name", a number, or a bracket. Matching against a hardcoded list was
// tried first and is subtly wrong - a directive from a newer pbrt than this
// parser knows about would not be in the list, so it would be reported as
// malformed input and fail the whole file. That is precisely the case
// warn-and-skip exists to survive.
inline bool looksLikeDirective(const Token &t) {
	return !t.quoted && !t.text.empty()
		   && std::isupper(static_cast<unsigned char>(t.text[0]));
}

struct GraphicsState {
	Matrix4 ctm;
	int materialIndex = -1;
	int areaLightIndex = -1;
	int insideMedium = -1;      // set by MediumInterface, inherited by Shape
	bool reverseOrientation = false;
	// ColorSpace directive's chosen working color space name - scoped by
	// AttributeBegin/AttributeEnd exactly like every other field here
	// (matches real pbrt-v4's own GraphicsState.colorSpace). Consumed
	// downstream via pbrt_flatten.h's resolveEmissionColor()/
	// RGBColorSpaceFromName() (rgb_colorspace.h) for converting a
	// "blackbody" emission temperature to RGB - the one place this loader
	// ever needs a working color space at all, since every ordinary "rgb"/
	// "color" parameter is just read as raw numbers with no color-space-
	// aware conversion. Always one of "srgb" (default), "dci-p3",
	// "rec2020", "aces2065-1" - the ColorSpace directive handler validates
	// and warns rather than storing an unrecognized name here, so nothing
	// downstream needs to re-validate it. Captured onto each LightDecl at
	// declaration time (LightDecl::colorSpaceName, alongside xform) the
	// same way the CTM is, so two lights under different ColorSpace scopes
	// in the same file each resolve correctly.
	std::string colorSpaceName = "srgb";
};

class Parser {
public:
	explicit Parser(TokenStream stream) : stream_(std::move(stream)), t_(stream_.tokens) {}

	ParseResult run() {
		while (pos_ < t_.size()) {
			const Token &tok = t_[pos_];
			if (!looksLikeDirective(tok)) {
				// Stray token where a directive was expected. Nearly always a
				// malformed parameter list, and continuing would misattribute
				// everything after it, so this one is fatal.
				return fail(tok.line, "expected a directive, found '" + tok.text + "'");
			}
			if (!dispatch()) return result_;
		}
		result_.ok = true;
		result_.scene = s_;
		return result_;
	}

private:
	TokenStream stream_;
	const std::vector<Token> &t_;
	std::size_t pos_ = 0;
	Scene s_;
	ParseResult result_;
	GraphicsState gs_;
	std::vector<GraphicsState> stack_;
	// Index into Scene::objects while between ObjectBegin and ObjectEnd, or
	// -1. Shapes declared while this is set belong to the definition rather
	// than to the scene, which is the whole difference between defining an
	// object and drawing one.
	int recordingObject_ = -1;
	bool inWorld_ = false;
	// CoordinateSystem/CoordSysTransform: named snapshots of the CTM, saved
	// and recalled by name - pure parser bookkeeping (mirrors pbrt-v4's own
	// GraphicsState.namedCoordinateSystems), never read by anything past
	// this parser since only the resolved gs_.ctm ever reaches Scene. A
	// plain append + linear scan by name, matching this class's other
	// "declare by name, recall later" bookkeeping (s_.objects/ObjectBegin,
	// s_.materials/NamedMaterial, s_.media/MediumInterface just below) -
	// a redefinition simply appends again and the scan below always finds
	// the LAST match, same "later definition wins" rule those use. Never
	// more than a handful of entries in a real scene, so this trades
	// nothing for staying consistent with the established idiom here.
	std::vector<std::pair<std::string, Matrix4>> namedCoordinateSystems_;

	int curLine() const {
		if (pos_ < t_.size()) return t_[pos_].line;
		return t_.empty() ? 0 : t_.back().line;
	}

	// Errors and warnings name the file they came from. Without it, "line 12"
	// in a scene that includes six geometry files is close to useless.
	std::string fileOf(std::size_t at) const {
		if (at >= t_.size()) return std::string();
		const int idx = t_[at].file;
		if (idx <= 0 || static_cast<std::size_t>(idx) >= stream_.files.size()) return std::string();
		return stream_.files[static_cast<std::size_t>(idx)];
	}

	ParseResult fail(int line, const std::string &msg) {
		result_.ok = false;
		result_.line = line;
		const std::string f = fileOf(pos_ ? pos_ - 1 : 0);
		result_.error = (f.empty() ? std::string() : f + " ") + "line "
						+ std::to_string(line) + ": " + msg;
		return result_;
	}

	void warn(int line, const std::string &msg) {
		s_.warnings.push_back({line, fileOf(pos_ ? pos_ - 1 : 0), msg});
	}

	bool readNumbers(int count, double *out, const std::string &directive) {
		for (int i = 0; i < count; ++i) {
			if (pos_ >= t_.size() || !toDouble(t_[pos_].text, out[i])) {
				result_ = fail(curLine(), directive + " needs " + std::to_string(count) + " numbers");
				return false;
			}
			++pos_;
		}
		return true;
	}

	// A parameter list runs until the next directive or end of file. Each entry
	// is a quoted "type name" followed by one value or a bracketed list.
	ParamList readParams() {
		ParamList list;
		while (pos_ < t_.size() && !looksLikeDirective(t_[pos_])) {
			// A parameter list holds only quoted "type name" entries. Anything
			// else here is malformed, so stop and let the main loop report it
			// against the offending token rather than silently swallowing it.
			if (!t_[pos_].quoted) break;

			const std::string decl = t_[pos_].text;
			++pos_;
			Param p;
			const std::size_t sp = decl.find(' ');
			if (sp == std::string::npos) {
				// A bare quoted string with no type, e.g. the filename after
				// Include. Callers that want it read it before calling here.
				p.type = "string";
				p.name = decl;
			} else {
				p.type = decl.substr(0, sp);
				p.name = decl.substr(sp + 1);
				while (!p.name.empty() && p.name.front() == ' ') p.name.erase(p.name.begin());
			}

			const bool bracketed = (pos_ < t_.size() && t_[pos_].text == "[" && !t_[pos_].quoted);
			if (bracketed) ++pos_;
			for (;;) {
				if (pos_ >= t_.size()) break;
				if (bracketed && t_[pos_].text == "]" && !t_[pos_].quoted) { ++pos_; break; }
				if (!bracketed && looksLikeDirective(t_[pos_])) break;
				double d = 0.0;
				if (!t_[pos_].quoted && toDouble(t_[pos_].text, d)) {
					p.numbers.push_back(d);
				} else if (t_[pos_].quoted) {
					p.strings.push_back(t_[pos_].text);
				} else if (t_[pos_].text == "true" || t_[pos_].text == "false") {
					p.strings.push_back(t_[pos_].text);
				} else {
					break;   // next parameter's "type name", or something odd
				}
				++pos_;
				if (!bracketed) break;   // unbracketed means exactly one value
			}
			list.items.push_back(p);
		}
		return list;
	}

	bool dispatch() {
		const std::string d = t_[pos_].text;
		const int line = t_[pos_].line;
		++pos_;

		if (d == "Identity")     { gs_.ctm = Matrix4::identity(); return true; }
		if (d == "Translate")    { double v[3]; if (!readNumbers(3, v, d)) return false;
								   gs_.ctm = gs_.ctm * Matrix4::translate(v[0], v[1], v[2]); return true; }
		if (d == "Scale")        { double v[3]; if (!readNumbers(3, v, d)) return false;
								   gs_.ctm = gs_.ctm * Matrix4::scale(v[0], v[1], v[2]); return true; }
		if (d == "Rotate")       { double v[4]; if (!readNumbers(4, v, d)) return false;
								   gs_.ctm = gs_.ctm * Matrix4::rotate(v[0], v[1], v[2], v[3]); return true; }
		if (d == "LookAt")       { double v[9]; if (!readNumbers(9, v, d)) return false;
								   gs_.ctm = gs_.ctm * Matrix4::lookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
								   return true; }
		if (d == "Transform" || d == "ConcatTransform") {
			// Optionally bracketed; pbrt writes these column-major.
			const bool br = (pos_ < t_.size() && t_[pos_].text == "[");
			if (br) ++pos_;
			double v[16];
			if (!readNumbers(16, v, d)) return false;
			if (br && pos_ < t_.size() && t_[pos_].text == "]") ++pos_;
			Matrix4 m;
			for (int c = 0; c < 4; ++c)
				for (int r = 0; r < 4; ++r) m.m[r * 4 + c] = v[c * 4 + r];
			gs_.ctm = (d == "Transform") ? m : gs_.ctm * m;
			return true;
		}

		if (d == "WorldBegin") {
			s_.worldToCamera = gs_.ctm;
			// pbrt resets the CTM (and the world-local pieces of graphics
			// state - current material/area light/medium/orientation, which
			// only ever make sense inside the world block) to their defaults
			// here. colorSpaceName does NOT reset: ColorSpace is a legitimate
			// pre-WorldBegin "options block" directive (declared once, near
			// the top of the file, same as Sampler/Integrator/Film) and is
			// expected to carry into the world block, not be wiped by it -
			// this codebase's own colorspace-blackbody.pbrt demo scene
			// relies on exactly this.
			const std::string colorSpaceName = gs_.colorSpaceName;
			gs_ = GraphicsState{};
			gs_.colorSpaceName = colorSpaceName;
			// pbrt-v4 builtin: "world" always names the CTM in effect right
			// after WorldBegin (i.e. identity) - a scene can CoordSysTransform
			// back to it later without ever having declared its own
			// CoordinateSystem "world".
			namedCoordinateSystems_.push_back({"world", gs_.ctm});
			inWorld_ = true;
			return true;
		}
		if (d == "WorldEnd") return true;   // pbrt-v4 dropped it; tolerate v3 files

		// TransformBegin/End were removed in pbrt-v4 but appear in v3 scenes,
		// where they push only the CTM. Treating them as full attribute scopes
		// is close enough for loading and avoids rejecting the file.
		if (d == "AttributeBegin" || d == "TransformBegin") {
			stack_.push_back(gs_);
			return true;
		}
		if (d == "ObjectBegin") {
			std::string name;
			if (pos_ < t_.size() && t_[pos_].quoted) { name = t_[pos_].text; ++pos_; }
			// Nesting is rejected rather than flattened. pbrt refuses it too,
			// and a definition that silently absorbed another one would place
			// the inner object's geometry at every outer instance - geometry
			// appearing where the scene never put it.
			if (recordingObject_ >= 0)
				return (result_ = fail(line, "ObjectBegin inside an instance definition")), false;
			for (const ObjectDecl &o : s_.objects) {
				if (o.name == name) {
					warn(line, "object '" + name + "' is redefined; "
						 "the later definition replaces the earlier one");
					break;
				}
			}
			stack_.push_back(gs_);
			s_.objects.push_back(ObjectDecl{name, {}});
			recordingObject_ = static_cast<int>(s_.objects.size()) - 1;
			return true;
		}
		if (d == "ObjectEnd") {
			if (recordingObject_ < 0)
				return (result_ = fail(line, "ObjectEnd without a matching ObjectBegin")), false;
			recordingObject_ = -1;
			if (stack_.empty())
				return (result_ = fail(line, "ObjectEnd without a matching Begin")), false;
			gs_ = stack_.back();
			stack_.pop_back();
			return true;
		}
		if (d == "AttributeEnd" || d == "TransformEnd") {
			if (stack_.empty())
				return (result_ = fail(line, d + " without a matching Begin")), false;
			gs_ = stack_.back();
			stack_.pop_back();
			return true;
		}
		if (d == "ObjectInstance") {
			std::string name;
			if (pos_ < t_.size() && t_[pos_].quoted) { name = t_[pos_].text; ++pos_; }
			if (recordingObject_ >= 0)
				return (result_ = fail(line, "ObjectInstance inside an instance definition")), false;
			InstanceDecl inst;
			inst.name = name;
			inst.xform = gs_.ctm;
			inst.line = line;
			s_.instances.push_back(inst);
			return true;
		}
		if (d == "ReverseOrientation") { gs_.reverseOrientation = !gs_.reverseOrientation; return true; }

		if (d == "Camera") {
			s_.cameraDeclared = true;
			if (pos_ < t_.size() && t_[pos_].quoted) { s_.cameraType = t_[pos_].text; ++pos_; }
			s_.cameraParams = readParams();
			// pbrt-v4 builtin: "camera" always names the CTM in effect at the
			// Camera statement - the standard idiom for positioning a light
			// relative to the camera via a later CoordSysTransform "camera".
			namedCoordinateSystems_.push_back({"camera", gs_.ctm});
			return true;
		}
		if (d == "Film") {
			if (pos_ < t_.size() && t_[pos_].quoted) ++pos_;   // "rgb", "gbuffer"
			const ParamList p = readParams();
			s_.xResolution = p.getInt("xresolution", s_.xResolution);
			s_.yResolution = p.getInt("yresolution", s_.yResolution);
			s_.filmFilename = p.getString("filename", s_.filmFilename);
			return true;
		}
		if (d == "Sampler") {
			if (pos_ < t_.size() && t_[pos_].quoted) { s_.samplerType = t_[pos_].text; ++pos_; }
			s_.samplesPerPixel = readParams().getInt("pixelsamples", s_.samplesPerPixel);
			return true;
		}
		if (d == "PixelFilter") {
			if (pos_ < t_.size() && t_[pos_].quoted) { s_.filterType = t_[pos_].text; ++pos_; }
			s_.filterParams = readParams();
			return true;
		}
		if (d == "Integrator") {
			if (pos_ < t_.size() && t_[pos_].quoted) { s_.integrator = t_[pos_].text; ++pos_; }
			s_.maxDepth = readParams().getInt("maxdepth", s_.maxDepth);
			return true;
		}

		if (d == "Material" || d == "MakeNamedMaterial") {
			MaterialDecl m;
			if (pos_ < t_.size() && t_[pos_].quoted) {
				if (d == "Material") m.type = t_[pos_].text; else m.name = t_[pos_].text;
				++pos_;
			}
			m.params = readParams();
			if (d == "MakeNamedMaterial") m.type = m.params.getString("type", "diffuse");
			s_.materials.push_back(m);
			gs_.materialIndex = static_cast<int>(s_.materials.size()) - 1;
			return true;
		}
		if (d == "NamedMaterial") {
			if (pos_ >= t_.size() || !t_[pos_].quoted)
				return (result_ = fail(line, "NamedMaterial needs a quoted name")), false;
			const std::string want = t_[pos_].text;
			++pos_;
			int found = -1;
			for (std::size_t i = 0; i < s_.materials.size(); ++i)
				if (s_.materials[i].name == want) found = static_cast<int>(i);
			if (found < 0)
				return (result_ = fail(line, "NamedMaterial '" + want + "' was never declared")), false;
			gs_.materialIndex = found;
			return true;
		}
		if (d == "Texture") {
			TextureDecl tx;
			if (pos_ < t_.size() && t_[pos_].quoted) { tx.name = t_[pos_].text; ++pos_; }
			if (pos_ < t_.size() && t_[pos_].quoted) { tx.dataType = t_[pos_].text; ++pos_; }
			if (pos_ < t_.size() && t_[pos_].quoted) { tx.cls = t_[pos_].text; ++pos_; }
			tx.params = readParams();
			s_.textures.push_back(tx);
			return true;
		}
		if (d == "MakeNamedMedium") {
			MediumDecl md;
			if (pos_ < t_.size() && t_[pos_].quoted) { md.name = t_[pos_].text; ++pos_; }
			md.params = readParams();
			md.type = md.params.getString("type", "homogeneous");
			md.xform = gs_.ctm;   // see MediumDecl::xform's own comment
			s_.media.push_back(md);
			return true;
		}
		if (d == "MediumInterface") {
			// Two consecutive quoted names - "inside" "outside" - not a
			// "type name" param list. An empty string names vacuum (pbrt's
			// own convention); only "inside" is tracked (see ShapeDecl::
			// insideMedium's comment on why "outside" is out of scope).
			std::string insideName;
			if (pos_ < t_.size() && t_[pos_].quoted) { insideName = t_[pos_].text; ++pos_; }
			if (pos_ < t_.size() && t_[pos_].quoted) { ++pos_; }   // outside name, unused
			int found = -1;
			for (std::size_t i = 0; i < s_.media.size(); ++i)
				if (s_.media[i].name == insideName) found = static_cast<int>(i);
			if (!insideName.empty() && found < 0)
				warn(line, "MediumInterface names '" + insideName +
					 "', which was never declared by MakeNamedMedium; treated as vacuum");
			gs_.insideMedium = found;
			return true;
		}
		if (d == "LightSource" || d == "AreaLightSource") {
			LightDecl l;
			if (pos_ < t_.size() && t_[pos_].quoted) { l.type = t_[pos_].text; ++pos_; }
			l.params = readParams();
			l.xform = gs_.ctm;
			l.colorSpaceName = gs_.colorSpaceName;
			if (d == "LightSource") {
				s_.lights.push_back(l);
			} else {
				s_.areaLights.push_back(l);
				gs_.areaLightIndex = static_cast<int>(s_.areaLights.size()) - 1;
			}
			return true;
		}
		if (d == "Shape") {
			ShapeDecl sh;
			if (pos_ < t_.size() && t_[pos_].quoted) { sh.type = t_[pos_].text; ++pos_; }
			sh.params = readParams();
			sh.xform = gs_.ctm;
			sh.materialIndex = gs_.materialIndex;
			sh.areaLightIndex = gs_.areaLightIndex;
			sh.insideMedium = gs_.insideMedium;
			sh.reverseOrientation = gs_.reverseOrientation;
			if (recordingObject_ >= 0)
				s_.objects[static_cast<std::size_t>(recordingObject_)].shapes.push_back(sh);
			else
				s_.shapes.push_back(sh);
			return true;
		}

		if (d == "ColorSpace") {
			std::string name;
			if (pos_ < t_.size() && t_[pos_].quoted) { name = t_[pos_].text; ++pos_; }
			// Validated here, eagerly, rather than deferred to a downstream
			// FromName-style helper (unlike samplerType/filterType above) so
			// the warning gets this directive's real line number - matching
			// this dispatch()'s own MediumInterface precedent just above,
			// which validates its name inline for the same reason. Keep this
			// name list in sync by hand with rgb_colorspace.h's
			// RGBColorSpaceFromName() - pbrt_scene.h is parsing-only (see
			// this file's own header comment) and deliberately does not
			// depend on that header's colorimetry math, so the two lists
			// can't just call one shared function.
			if (name == "srgb" || name == "dci-p3" || name == "rec2020" || name == "aces2065-1") {
				gs_.colorSpaceName = name;
			} else {
				warn(line, "ColorSpace \"" + name + "\": unrecognized name, using sRGB");
			}
			return true;
		}
		if (d == "CoordinateSystem") {
			std::string name;
			if (pos_ < t_.size() && t_[pos_].quoted) { name = t_[pos_].text; ++pos_; }
			namedCoordinateSystems_.push_back({name, gs_.ctm});
			return true;
		}
		if (d == "CoordSysTransform") {
			std::string name;
			if (pos_ < t_.size() && t_[pos_].quoted) { name = t_[pos_].text; ++pos_; }
			int found = -1;
			for (std::size_t i = 0; i < namedCoordinateSystems_.size(); ++i)
				if (namedCoordinateSystems_[i].first == name) found = static_cast<int>(i);
			if (found >= 0) {
				gs_.ctm = namedCoordinateSystems_[static_cast<std::size_t>(found)].second;
			} else {
				warn(line, "CoordSysTransform: unknown coordinate system \"" + name + "\"");
			}
			return true;
		}

		// Recognised pbrt, not implemented. Skip its arguments and carry on -
		// see the header comment on why this is not an error.
		warn(line, d + ": not supported, skipped");
		while (pos_ < t_.size() && !looksLikeDirective(t_[pos_])) ++pos_;
		return true;
	}
};

} // namespace detail

// `resolver` supplies Include'd files. Omit it to parse a self-contained
// scene; see FileResolver for why a callback rather than direct file I/O.
inline ParseResult parse(const std::string &text, const FileResolver &resolver = {}) {
	detail::TokenStream stream;
	std::vector<std::string> openStack;
	std::string error;
	if (!detail::expandIncludes(text, 0, stream, resolver, openStack, error)) {
		ParseResult r;
		r.ok = false;
		r.error = error;
		return r;
	}
	return detail::Parser(std::move(stream)).run();
}

} // namespace pbrt_scene
