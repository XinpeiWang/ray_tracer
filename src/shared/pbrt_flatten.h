#pragma once
// pbrt_flatten.h -- turns a parsed pbrt scene into world-space geometry.
//
// pbrt describes geometry in local coordinates under a current transformation
// matrix. Neither backend can consume that directly: the CPU side has only
// `translate` and `rotate_y` hittables (hittable.h) - no general 4x4 - and the
// GPU side builds flat AABB primitive arrays. So the CTM is BAKED into vertex
// positions here, once, instead of being applied per ray at render time.
//
// That is the cheaper answer as well as the simpler one. A transform wrapper
// costs a matrix multiply on every ray-object test forever; baking costs one
// multiply per vertex at load.
//
// This sits between pbrt_scene.h (text -> description) and the backends
// (geometry -> renderer), and like both of its neighbours it is Qt-free and
// free of renderer types, so the MSVC test binary can reach it.
//
// WHAT IT APPROXIMATES, AND SAYS SO
// ---------------------------------
// Baking works exactly for triangles: any affine transform maps a triangle to
// a triangle. It does NOT work for spheres. A non-uniform scale turns a sphere
// into an ellipsoid, which this cannot represent, so such a sphere is emitted
// with its largest axis and a warning. Silently emitting a round sphere where
// the scene wanted a squashed one is the kind of difference nobody spots
// against a reference image.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <utility>
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"
#include "conductor_data.h"
#include "glass_data.h"
#include "spectrum_types.h"   // BlackbodySpectrum - see resolveEmissionColor()'s own comment
#include "spectral_math.h"    // SpectrumToXYZ/InnerProduct/GetCIE_Y - ditto
#include "rgb_colorspace.h"   // RGBColorSpaceFromName() - ditto
#include "bssrdf.h"           // BSSRDFTable, ComputeBeamDiffusionBSSRDF, SubsurfaceFromDiffuse - a subsurface material given as reflectance + mfp

// Refinement is exponential: every level multiplies the triangle count by
// four, so a scene asking for 8 turns a 10k-triangle cage into 650 million.
// Clamping is the difference between a slow render and an exhausted machine.
inline constexpr int kMaxSubdivLevels = 4;

#include "pbrt_flat_shapes_types.h"
#include "pbrt_flat_materials_types.h"
#include "pbrt_flat_scene_types.h"

namespace pbrt_flatten {

// Named flatten_detail, not the more generic "detail" - a bare "detail"
// here previously collided with compensated_float.h's own unrelated
// namespace detail (ambiguous unqualified lookup, MSVC error C2872) in any
// translation unit that both instantiates InnerProduct() (compensated_float.h,
// via e.g. square_matrix.h) and does `using namespace pbrt_flatten;` (this
// header's own established convention for callers, see flatten()'s local
// using-directive below) - exactly tests/unit/pbrt_flatten_tests.cpp's shape.
namespace flatten_detail {

// The beam-diffusion table SubsurfaceFromDiffuse() inverts, one per (g, eta)
// pair (building it costs ~40 ms, a scene has a handful of subsurface
// materials at most, usually sharing one pair).
inline const BSSRDFTable &beamDiffusionTable(double g, double eta) {
	static std::map<std::pair<double, double>, BSSRDFTable> cache;
	auto it = cache.find({g, eta});
	if (it == cache.end()) {
		BSSRDFTable table(100, 64);
		ComputeBeamDiffusionBSSRDF(g, eta, &table);
		it = cache.emplace(std::make_pair(g, eta), std::move(table)).first;
	}
	return it->second;
}

// Resolves a Texture by name against the scene's declared list - pbrt's own
// "last declaration wins" rule (a later `Texture "name" ...` shadows an
// earlier one with the same name), so this deliberately doesn't `break` on
// the first match. Centralizes what used to be 8 independent, hand-rolled
// copies of this exact scan scattered through flatten()'s material loop.
inline const pbrt_scene::TextureDecl *findTexture(const pbrt_scene::Scene &scene,
													const std::string &name) {
	const pbrt_scene::TextureDecl *tex = nullptr;
	for (const pbrt_scene::TextureDecl &t : scene.textures)
		if (t.name == name) tex = &t;
	return tex;
}

// Resolves an "imagemap" Texture's own "string encoding" param to an actual
// gamma exponent - see Material::textureGamma's own comment for the mapping
// rationale (linear->1.0, gamma <value>->that value, sRGB/absent->2.2, this
// codebase's existing default). pbrt-v4 also accepts a literal filename
// (e.g. "encoding" "srgb8.icc") for a real ICC profile - this loader has no
// ICC support at all, so any value not recognized as "linear"/"sRGB"/
// starting with "gamma " falls back to the 2.2 default rather than
// misinterpreting it - `warn` is called in that case (and when a "gamma "
// value fails to parse, or parses to something not usable as a decode
// exponent: non-finite or <= 0) so the fallback is never silent, matching
// this codebase's "approximate, but never silently wrong" precedent for
// other unrecognized-string params. The empty/"sRGB"/"srgb" case is the
// ordinary "no encoding requested, or explicitly the default" case and
// does NOT warn - it is not a fallback from something the scene asked for.
inline double resolveTextureGamma(const pbrt_scene::TextureDecl &imgTex,
								   const std::function<void(const std::string&)> &warn) {
	const std::string encoding = imgTex.params.getString("encoding", "");
	if (encoding.empty() || encoding == "sRGB" || encoding == "srgb") return kDefaultTextureGamma;
	if (encoding == "linear") return 1.0;
	if (encoding.rfind("gamma ", 0) == 0) {
		double value = 0.0;
		bool parsed = true;
		try { value = std::stod(encoding.substr(6)); } catch (...) { parsed = false; }
		if (parsed && std::isfinite(value) && value > 0.0) return value;
		warn("Texture \"imagemap\" \"string encoding\" \"" + encoding +
			 "\" is not a usable gamma value; using the default gamma 2.2 instead");
		return kDefaultTextureGamma;
	}
	warn("Texture \"imagemap\" \"string encoding\" \"" + encoding +
		 "\" is not recognized (expected \"linear\", \"sRGB\", or \"gamma <value>\"); "
		 "using the default gamma 2.2 instead");
	return kDefaultTextureGamma;
}

// Resolves an "imagemap" Texture's "string wrap" param against the 3 real
// values this loader understands - guaranteed to always return "repeat"/
// "clamp"/"black" by construction, matching resolveTextureGamma's own
// "approximate, but never silently wrong" precedent. An unrecognized value
// (typo, wrong case, e.g. "Clamp") falls back to "repeat" (pbrt-v4's real
// default) WITH a warning, rather than silently resolving to the opposite of
// what may have been intended.
inline std::string resolveTextureWrap(const pbrt_scene::TextureDecl &imgTex,
									   const std::function<void(const std::string&)> &warn) {
	const std::string wrapStr = imgTex.params.getString("wrap", "repeat");
	if (wrapStr == "repeat" || wrapStr == "clamp" || wrapStr == "black") return wrapStr;
	warn("Texture \"imagemap\" \"string wrap\" \"" + wrapStr +
		 "\" is not recognized (expected \"repeat\", \"clamp\", or \"black\"); "
		 "using \"repeat\" instead");
	return "repeat";
}

// Resolves an "imagemap" Texture's full "encoding"/"wrap"/"invert" set into
// one TextureDecodeOptions - the single per-slot resolution point every
// slot that carries this options struct (currently textureFilename's own 3
// loose fields, plus transmittanceTextureOptions/roughnessTextureOptions)
// funnels through, so the gamma/wrap-validation logic above lives in exactly
// one place regardless of how many slots end up using it.
inline TextureDecodeOptions resolveTextureDecodeOptions(const pbrt_scene::TextureDecl &imgTex,
														  const std::function<void(const std::string&)> &warn) {
	TextureDecodeOptions opts;
	opts.gamma = resolveTextureGamma(imgTex, warn);
	opts.wrap = resolveTextureWrap(imgTex, warn);
	// See Material::textureWrapIndex's own comment - kept in sync with
	// `wrap` above, the single resolution point for both.
	opts.wrapIndex = (opts.wrap == "clamp") ? 0 : (opts.wrap == "black") ? 2 : 1;
	opts.invert = imgTex.params.getBool("invert", false);
	return opts;
}

// The 2 remaining non-primary texture-filename slots (alpha/displacement)
// still never read "encoding"/"wrap"/"invert" - alpha is a coverage MASK,
// not colour, so "encoding" (gamma) is not meaningful there by this
// codebase's own established design (see gpu/optix/pbrt_gpu_builder.h's
// getOrBuildPbrtAlphaMaskTexture() own comment); displacement goes through a
// materially different CPU pipeline (rtw_image/image_texture, not
// mipmap_texture/MipMapOptions) with no wrap-mode concept at all today -
// both a deliberate, documented scope cut (see docs/PBRT_SUPPORT.md's own
// note on this), NOT the "every non-primary slot" cut this used to be
// (transmittance/roughness now resolve for real via
// resolveTextureDecodeOptions() above). That cut is otherwise silent: a
// scene author binding one of those params to alpha/displacement's imagemap
// gets it dropped with no diagnostic anywhere (CPU or GPU). Called at each
// of those 2 resolution sites right after a bare/scale-wrapped imagemap is
// found, so the warning names the slot it was ignored for.
inline void warnIfImagemapOptionsIgnored(const pbrt_scene::TextureDecl &imgTex,
										  const char *slotName,
										  const std::function<void(const std::string&)> &warn) {
	if (imgTex.params.find("encoding") || imgTex.params.find("wrap") || imgTex.params.find("invert")) {
		warn(std::string("Texture \"imagemap\" bound to \"") + slotName +
			 "\" declares \"encoding\"/\"wrap\"/\"invert\", but this loader only "
			 "honors those on a material's reflectance/transmittance/roughness "
			 "textures - the request is ignored for this slot");
	}
}

// One shape to emit, where to put it, and under which transform. Routing the
// single shape loop through this is what lets the same code serve three
// callers - scene geometry, an instance definition's object-space geometry,
// and an instanced emitter baked into world space - without three copies of
// the triangulation, subdivision and PLY handling.
struct ShapeWork {
	const pbrt_scene::ShapeDecl *shape = nullptr;
	pbrt_scene::Matrix4 xform;
	// EndTime counterpart of xform above (ShapeDecl::xformEnd, composed with
	// any instance placement transform the same way xform itself is) - see
	// Sphere::center1's own comment for what consumes this. Defaults to
	// identity like Matrix4 itself; every call site below sets it to mirror
	// xform's own construction exactly, so a shape whose ShapeDecl was never
	// authored inside an ActiveTransform pair keeps xformEnd == xform (no
	// motion), same "real inequality, not directive presence" convention as
	// Scene::cameraIsAnimated().
	pbrt_scene::Matrix4 xformEnd;
	std::vector<Triangle> *triangles = nullptr;
	std::vector<Sphere> *spheres = nullptr;
	std::vector<BilinearPatch> *bilinearPatches = nullptr;
	// Left null for an instance DEFINITION's own (object-space) shape list,
	// same as bilinearPatches above - Disk/Cylinder inside an ObjectBegin/End
	// block fall through to the generic unsupported-shape warning rather than
	// silently being dropped with no explanation, matching that precedent
	// exactly rather than inventing a new one.
	std::vector<Disk> *disks = nullptr;
	std::vector<Cylinder> *cylinders = nullptr;
	// Same "left null inside an ObjectBegin/End instance definition falls
	// through to the generic unsupported-shape warning" precedent as disks/
	// cylinders above.
	std::vector<Curve> *curves = nullptr;
	// Appended at the END, not alongside disks/cylinders above, so every
	// existing positional-brace-init call site (this struct is built with
	// `{&shape, xform, xformEnd, &triangles, ...}`-style positional args,
	// same convention/hazard as SceneDescriptor - see that struct's own
	// comment, scene_registry.h) keeps binding its trailing `&out.curves`
	// argument to `curves` above, not silently shifting onto one of these -
	// left at their nullptr default (correctly: cone/paraboloid/hyperboloid
	// are never emissive-baked-into-an-instance in this loader, matching
	// bilinearPatches' own "left null for an instance definition" precedent).
	std::vector<Cone> *cones = nullptr;
	std::vector<Paraboloid> *paraboloids = nullptr;
	// True only for a plain top-level scene shape (the work.push_back() call
	// site iterating scene.shapes directly) - false for both an ObjectInstance
	// DEFINITION's own object-space shape (triangles points at
	// out.groups[g].triangles, not out.triangles) and an instanced EMISSIVE
	// shape baked into world space (triangles points at out.triangles too,
	// same as the top-level case, but this flag is explicitly false there).
	// Real object motion blur (AnimatedTriangleMesh, mesh-building branch
	// below) is gated on this instead of on `triangles == &out.triangles`
	// pointer identity - that pointer alone doesn't distinguish the top-level
	// and instanced-emissive cases (both target the same list), so a
	// gating check built on it depended on a SEPARATE, non-obvious invariant
	// (every instanced-emissive ShapeWork also having areaLightIndex >= 0) to
	// stay correct; a code-review pass on this feature's own commit flagged
	// that coupling as fragile enough to warrant this explicit field instead,
	// once it was clear (from ShapeWork's own "fields appended at the end
	// default for every other call site" convention, see curves' own comment
	// above) that doing so only requires updating the ONE call site that
	// needs `true`, not all three.
	bool isTopLevelScene = false;
};

// Row-major 4x4 multiply: `a` applied after `b`, i.e. the result maps a point
// through b and then through a.
inline pbrt_scene::Matrix4 compose(const pbrt_scene::Matrix4 &a,
								   const pbrt_scene::Matrix4 &b) {
	pbrt_scene::Matrix4 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			double s = 0;
			for (int k = 0; k < 4; ++k) s += a.m[i * 4 + k] * b.m[k * 4 + j];
			r.m[i * 4 + j] = s;
		}
	return r;
}

inline void transformPoint(const pbrt_scene::Matrix4 &m,
						   double x, double y, double z, double *out) {
	out[0] = m.m[0] * x + m.m[1] * y + m.m[2]  * z + m.m[3];
	out[1] = m.m[4] * x + m.m[5] * y + m.m[6]  * z + m.m[7];
	out[2] = m.m[8] * x + m.m[9] * y + m.m[10] * z + m.m[11];
}

// Splits a row-major affine Matrix4 into a flat 3x3 double[9] (rotation/
// scale block) plus a flat double[3] translation - the shape every grid
// medium type's toMediumMat/toMediumTranslate pair (Medium's own comment)
// and grid_medium_hittable/rgb_grid_medium_hittable's own mat_/translate_
// members want. Shared so a caller doesn't hand-copy the same m[i*4+j]/
// m[i*4+3] indexing convention a second time - see this codebase's own
// "reuse" code-review finding on the nanovdb medium block, which
// originally re-derived this by hand instead of calling here.
inline void splitAffine(const pbrt_scene::Matrix4 &m, double mat9[9], double translate3[3]) {
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			mat9[i * 3 + j] = m.m[i * 4 + j];
	translate3[0] = m.m[3];
	translate3[1] = m.m[7];
	translate3[2] = m.m[11];
}

// World-space AABB of a local-space box [lo,hi] under an arbitrary affine
// transform `worldFromLocal`: transforms all 8 corners and takes the
// axis-aligned min/max, so a rotated/non-uniformly-scaled placement still
// gets a correct (if conservatively larger) world bound rather than a
// naive diagonal scale. Shared for the same reason as splitAffine above -
// every grid medium type needs exactly this (cloud/rgbgrid/uniformgrid
// call it with lo=p0/hi=p1/worldFromLocal=the medium's own CTM; nanovdb
// calls it with lo=(0,0,0)/hi=(1,1,1)/worldFromLocal=its own reconstructed
// worldFromMedium, since its "box" is the unit cube in its own baked
// coordinate space, not a p0/p1 pair).
inline void aabbOfTransformedBox(const pbrt_scene::Matrix4 &worldFromLocal,
								  const double lo[3], const double hi[3],
								  double outMin[3], double outMax[3]) {
	outMin[0] = outMin[1] = outMin[2] = 1e300;
	outMax[0] = outMax[1] = outMax[2] = -1e300;
	for (int c = 0; c < 8; ++c) {
		const double cx = (c & 1) ? hi[0] : lo[0];
		const double cy = (c & 2) ? hi[1] : lo[1];
		const double cz = (c & 4) ? hi[2] : lo[2];
		double w[3];
		transformPoint(worldFromLocal, cx, cy, cz, w);
		for (int a = 0; a < 3; ++a) {
			outMin[a] = std::fmin(outMin[a], w[a]);
			outMax[a] = std::fmax(outMax[a], w[a]);
		}
	}
}

// Normals do NOT transform by the matrix that transforms points. Under a
// non-uniform scale, transforming a normal directly tilts it off the surface -
// squash a sphere and its normals stop being perpendicular to it. The correct
// transform is the inverse transpose of the upper-left 3x3, which is what this
// computes (by adjugate, then transposed by reading it column-wise).
//
// Doing it properly rather than assuming rigid transforms costs twenty lines
// and buys correctness on every scene that scales an axis - which real ones do.
inline void transformNormal(const pbrt_scene::Matrix4 &m,
							double x, double y, double z, double *out) {
	const double a = m.m[0], b = m.m[1], c = m.m[2];
	const double d = m.m[4], e = m.m[5], f = m.m[6];
	const double g = m.m[8], h = m.m[9], i = m.m[10];

	// Cofactors of the 3x3. The adjugate is their transpose, and the inverse
	// is adjugate/det - but we then want the transpose of that inverse, so the
	// two transposes cancel and the cofactor matrix is used directly.
	const double c00 = e * i - f * h, c01 = f * g - d * i, c02 = d * h - e * g;
	const double c10 = c * h - b * i, c11 = a * i - c * g, c12 = b * g - a * h;
	const double c20 = b * f - c * e, c21 = c * d - a * f, c22 = a * e - b * d;

	const double det = a * c00 + b * c01 + c * c02;
	if (std::fabs(det) < 1e-18) {           // degenerate: leave it alone
		out[0] = x; out[1] = y; out[2] = z;
		return;
	}

	double nx = c00 * x + c01 * y + c02 * z;
	double ny = c10 * x + c11 * y + c12 * z;
	double nz = c20 * x + c21 * y + c22 * z;

	// The cofactor matrix alone is inverse-transpose scaled by det (the two
	// transposes noted above cancel the adjugate's transpose, not the 1/det
	// factor) - for det>0 that uniform positive scale vanishes under the
	// normalize below and this was silently correct, but for det<0 (a
	// mirroring transform - e.g. pbrt's own "Scale -1 1 1", which shows up in
	// real scenes for cheaply flipping an asset) the omitted sign flip left
	// every transformed normal pointing exactly backwards.
	if (det < 0.0) { nx = -nx; ny = -ny; nz = -nz; }

	const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
	if (len > 0) { nx /= len; ny /= len; nz /= len; }
	out[0] = nx; out[1] = ny; out[2] = nz;
}

// Shape "curve"'s "degree 2"/"basis \"bspline\"" support - both reduce
// EXACTLY (not approximately) to the same cubic-Bezier-per-segment
// representation the rest of this loader already builds for the default
// degree-3/"bezier" case (CurveShape's own intersection/subdivision math,
// src/shared/shapes.h, is hard-coded to exactly 4 control points per
// segment throughout) - so both conversions live entirely here, at
// flatten() time, with zero changes needed anywhere downstream on either
// backend (both already consume the same pbrt_flatten::Curve::cp layout).
//
// Quadratic (degree 2) Bezier -> cubic Bezier: the standard, EXACT degree-
// elevation formula (raising a Bezier curve's degree by 1 never changes the
// curve itself, only its control-point count) - Q0=P0, Q3=P2, and the two
// interior points split the P0->P1/P1->P2 legs 1/3 of the way in.
inline void curveDegreeElevateQuadratic(const double p0[3], const double p1[3],
                                         const double p2[3], double out4x3[12]) {
	for (int k = 0; k < 3; ++k) {
		out4x3[0 + k] = p0[k];
		out4x3[3 + k] = p0[k] + (2.0 / 3.0) * (p1[k] - p0[k]);
		out4x3[6 + k] = p2[k] + (2.0 / 3.0) * (p1[k] - p2[k]);
		out4x3[9 + k] = p2[k];
	}
}

// Uniform cubic B-spline -> Bezier, one segment: the standard, EXACT
// change-of-basis matrix (a textbook result - see e.g. Foley/van Dam, or
// pbrt-v4's own BlossomCubicBezier specialized to uniform knot spacing) for
// the Bezier control points spanned by 4 consecutive B-spline control
// points p0..p3.
inline void curveBsplineSegmentToBezierCubic(const double p0[3], const double p1[3],
                                              const double p2[3], const double p3[3],
                                              double out4x3[12]) {
	for (int k = 0; k < 3; ++k) {
		out4x3[0 + k] = (p0[k] + 4.0 * p1[k] + p2[k]) / 6.0;
		out4x3[3 + k] = (4.0 * p1[k] + 2.0 * p2[k]) / 6.0;
		out4x3[6 + k] = (2.0 * p1[k] + 4.0 * p2[k]) / 6.0;
		out4x3[9 + k] = (p1[k] + 4.0 * p2[k] + p3[k]) / 6.0;
	}
}

// True when this transform's upper-left 3x3 has a negative determinant (a
// mirroring transform - e.g. pbrt's own "Scale -1 1 1") - pbrt-v4's own
// "transformSwapsHandedness" test, needed alongside a shape's own
// ReverseOrientation flag to decide whether its normal ends up flipped
// (pbrt-v4: reverseOrientation ^ transformSwapsHandedness). Reuses the
// existing, more numerically stable SquareMatrix<3>/Determinant() (square_matrix.h,
// already transitively included here via rgb_colorspace.h - a real port of
// pbrt-v4's own SquareMatrix<N>/Determinant, computed via DifferenceOfProducts
// to avoid the catastrophic cancellation a plain a*b-c*d subtraction risks)
// rather than a third hand-rolled cofactor expansion - transformNormal below
// already has its own inline copy of this exact math, so this at least
// avoids adding a THIRD independent implementation of the same determinant.
// Same fabs(det)<1e-18 degenerate threshold transformNormal uses for its own
// det<0 mirroring check (see that function's own comment) - a near-singular
// transform's determinant SIGN is numerical noise, not a real answer, so
// this returns false (no flip) rather than letting floating-point rounding
// decide whether a shape's normal flips.
inline bool matrixSwapsHandedness(const pbrt_scene::Matrix4 &m) {
	const double flat[9] = {m.m[0], m.m[1], m.m[2],
							 m.m[4], m.m[5], m.m[6],
							 m.m[8], m.m[9], m.m[10]};
	const SquareMatrix<3> m3(flat, 9);
	const double det = Determinant(m3);
	if (std::fabs(det) < 1e-18) return false;
	return det < 0.0;
}

// True world -> light ROTATION (not the inverse-TRANSPOSE transformNormal
// computes - a light's "aim" needs the plain inverse of its light -> world
// rotation, the same sense pbrt-v4's own ApplyInverse(w) uses). Used only by
// Goniometric/Projection, whose image lookup is defined in light space: a
// world-space direction is rotated back into that space to index the
// profile/slide image (see PunctualLight::worldToLight's own comment).
//
// General 3x3 inverse via cofactors/adjugate/determinant - the same cofactor
// terms transformNormal already computes (c00..c22), just assembled as
// adjugate/det (= cofactor^T/det) here instead of being used directly as the
// inverse-transpose transformNormal wants. Degenerate (zero-determinant)
// input - a scene that somehow projected its light's CTM flat - falls back
// to identity rather than dividing by zero, matching transformNormal's own
// "leave it alone" degenerate case.
inline void worldToLightRotation(const pbrt_scene::Matrix4 &m, double *out) {
	const double a = m.m[0], b = m.m[1], c = m.m[2];
	const double d = m.m[4], e = m.m[5], f = m.m[6];
	const double g = m.m[8], h = m.m[9], i = m.m[10];

	const double c00 = e * i - f * h, c01 = f * g - d * i, c02 = d * h - e * g;
	const double c10 = c * h - b * i, c11 = a * i - c * g, c12 = b * g - a * h;
	const double c20 = b * f - c * e, c21 = c * d - a * f, c22 = a * e - b * d;

	const double det = a * c00 + b * c01 + c * c02;
	if (std::fabs(det) < 1e-18) {
		out[0] = 1; out[1] = 0; out[2] = 0;
		out[3] = 0; out[4] = 1; out[5] = 0;
		out[6] = 0; out[7] = 0; out[8] = 1;
		return;
	}

	// inverse = adjugate/det, adjugate = cofactor^T - so inverse row r is
	// cofactor COLUMN r, scaled by 1/det.
	const double invDet = 1.0 / det;
	out[0] = c00 * invDet; out[1] = c10 * invDet; out[2] = c20 * invDet;
	out[3] = c01 * invDet; out[4] = c11 * invDet; out[5] = c21 * invDet;
	out[6] = c02 * invDet; out[7] = c12 * invDet; out[8] = c22 * invDet;
}

// Normalizes a 3-vector in place; a degenerate (near-zero) input is left as
// the harmless default direction +Z rather than producing NaNs downstream -
// a scene whose "from"/"to" happen to coincide is a malformed light, not a
// reason to poison every ray direction computed from it.
inline void normalizeOrDefault(double *v, double defX, double defY, double defZ) {
	const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 1e-12) {
		v[0] /= len; v[1] /= len; v[2] /= len;
	} else {
		v[0] = defX; v[1] = defY; v[2] = defZ;
	}
}

// The three basis vectors' lengths are the scale along each axis. Comparing
// them is how a non-uniform scale is detected without decomposing the matrix
// properly - enough to know a sphere cannot survive it.
inline void axisScales(const pbrt_scene::Matrix4 &m, double *out) {
	for (int c = 0; c < 3; ++c) {
		const double a = m.m[0 + c], b = m.m[4 + c], d = m.m[8 + c];
		out[c] = std::sqrt(a * a + b * b + d * d);
	}
}

// Sphere/Disk/Cylinder/InstancePlacement each carry their object-to-world
// (or, for InstancePlacement, instance-placement) transform as a flat
// double[16] rather than pbrt_scene::Matrix4 directly, since this header is
// deliberately kept free of renderer types (see this file's own top
// comment) while pbrt_scene::Matrix4 is itself a shared, non-renderer type
// - this collapses the 4 independent copies of that flattening loop into
// one.
inline void fromMatrix4(const pbrt_scene::Matrix4 &m, double (&out)[16]) {
	for (int i = 0; i < 16; ++i) out[i] = m.m[i];
}

inline MaterialKind materialKindFor(const std::string &type) {
	if (type == "diffuse")             return MaterialKind::Diffuse;
	if (type == "conductor")           return MaterialKind::Conductor;
	if (type == "dielectric")          return MaterialKind::Dielectric;
	if (type == "thindielectric")      return MaterialKind::ThinDielectric;
	if (type == "coateddiffuse")       return MaterialKind::CoatedDiffuse;
	if (type == "coatedconductor")     return MaterialKind::CoatedConductor;
	if (type == "diffusetransmission") return MaterialKind::DiffuseTransmission;
	if (type == "subsurface")          return MaterialKind::Subsurface;
	if (type == "measured")            return MaterialKind::Measured;
	if (type == "mix")                 return MaterialKind::Mix;
	if (type == "hair")                return MaterialKind::Hair;
	// pbrt-v4's parser rewrites "none" and "" to "interface" (parser.cpp), so the canonical
	// spelling is "interface" itself - all three are the same no-BSDF boundary material.
	if (type == "interface" || type == "none" || type.empty()) return MaterialKind::Interface;
	// Not real pbrt-v4 - this loader's own extension, see MaterialKind::
	// Principled's own comment above for why it exists anyway.
	if (type == "principled")          return MaterialKind::Principled;
	// Ditto - see MaterialKind::NormalizedFresnel's own comment.
	if (type == "normalizedfresnel")   return MaterialKind::NormalizedFresnel;
	return MaterialKind::Unsupported;
}

// A subset of pbrt-v4's own named "measured scattering coefficient" table
// (media.cpp, GetMediumScatteringProperties - from Jensen/Marschner/Levoy/
// Hanrahan, "A Practical Model for Subsurface Light Transport", SIGGRAPH
// 2001). sigmaPrimeS is the REDUCED scattering coefficient - pbrt-v4 forces
// g=0 whenever a named preset is used, at which point sigma_s' == sigma_s,
// so it can be assigned straight to Material::sigma_s. Only the first
// (best-known, most commonly used) dozen entries are carried here; pbrt-v4's
// own table has several dozen more (mostly foods/drinks from a 2006 dilution
// study) that no scene in this loader's corpus names.
struct SubsurfacePreset {
	const char *name;
	double sigmaPrimeS[3];
	double sigmaA[3];
};

inline const SubsurfacePreset *subsurfacePresetFor(const std::string &name) {
	static const SubsurfacePreset kPresets[] = {
		{"Apple",      {2.29, 2.39, 1.97}, {0.0030, 0.0034, 0.046}},
		{"Chicken1",   {0.15, 0.21, 0.38}, {0.015,  0.077,  0.19}},
		{"Chicken2",   {0.19, 0.25, 0.32}, {0.018,  0.088,  0.20}},
		{"Cream",      {7.38, 5.47, 3.15}, {0.0002, 0.0028, 0.0163}},
		{"Ketchup",    {0.18, 0.07, 0.03}, {0.061,  0.97,   1.45}},
		{"Marble",     {2.19, 2.62, 3.00}, {0.0021, 0.0041, 0.0071}},
		{"Potato",     {0.68, 0.70, 0.55}, {0.0024, 0.0090, 0.12}},
		{"Skimmilk",   {0.70, 1.22, 1.90}, {0.0014, 0.0025, 0.0142}},
		{"Skin1",      {0.74, 0.88, 1.01}, {0.032,  0.17,   0.48}},
		{"Skin2",      {1.09, 1.59, 1.79}, {0.013,  0.070,  0.145}},
		{"Spectralon", {11.6, 20.4, 14.9}, {0.00,   0.00,   0.00}},
		{"Wholemilk",  {2.55, 3.21, 3.77}, {0.0011, 0.0024, 0.014}},
	};
	for (const SubsurfacePreset &p : kPresets)
		if (name == p.name) return &p;
	return nullptr;
}

// Recovers the eye point and viewing direction from pbrt's WORLD-TO-CAMERA
// matrix by inverting it. The rotation part is orthonormal (LookAt builds it
// from normalised, mutually perpendicular axes), so the inverse is the
// transpose and the eye is -R^T * t. Doing a general 4x4 inverse here would be
// both slower and less numerically pleasant.
//
// pbrt's camera looks down +z with +y up, which is where the row picks below
// come from: R^T * (0,0,1) is R's third row, R^T * (0,1,0) is its second.
//
// `lookAtDistance` (Scene::cameraLookAtDistance(End), positive when the
// original LookAt survived untouched to WorldBegin - see that field's own
// comment) restores the REAL "lookat" the scene file declared. Without it,
// only the forward DIRECTION survives the matrix round-trip, not how far
// along it the scene's actual subject was - lookat would land exactly 1
// unit from lookfrom regardless of whether the file's own LookAt put its
// target 1 unit away or 1000. That placeholder is fine for anything that
// only needs the camera's aim (rendering itself never depends on lookat
// specifically), but it silently breaks any consumer that treats lookat as
// the subject's real location - focusDistanceFor()'s depth-of-field
// fallback, and every orbit-family launcher/camera_path.h path (orbit/
// spiral/showcase/figure8), which pivot the camera around lookat at a
// radius derived from this same distance.
inline Camera cameraFromWorldToCamera(const pbrt_scene::Matrix4 &w2c, double lookAtDistance = -1.0) {
	Camera c;
	const double *m = w2c.m;

	// eye = -R^T * t
	const double tx = m[3], ty = m[7], tz = m[11];
	c.lookfrom[0] = -(m[0] * tx + m[4] * ty + m[8]  * tz);
	c.lookfrom[1] = -(m[1] * tx + m[5] * ty + m[9]  * tz);
	c.lookfrom[2] = -(m[2] * tx + m[6] * ty + m[10] * tz);

	const double fwd[3] = {m[8], m[9], m[10]};
	c.up[0] = m[4]; c.up[1] = m[5]; c.up[2] = m[6];
	const double dist = (lookAtDistance > 0.0) ? lookAtDistance : 1.0;
	for (int i = 0; i < 3; ++i) c.lookat[i] = c.lookfrom[i] + fwd[i] * dist;
	return c;
}

} // namespace flatten_detail

#include "pbrt_flatten_materials.h"
#include "pbrt_flatten_scene_parts.h"
#include "pbrt_flatten_shapes.h"

namespace flatten_detail {

// PixelFilter, regularize, accelerator and the remaining scene-wide settings.
inline void flattenSettings(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// PixelFilter - see PixelFilter's own struct comment for radius, and
	// why "gaussian" is the right fallback for an absent/unrecognized kind.
	out.filter.kind = scene.filterType.empty() ? "gaussian" : scene.filterType;
	out.filter.B = scene.filterParams.getFloat("B", out.filter.B);
	out.filter.C = scene.filterParams.getFloat("C", out.filter.C);
	out.filter.sigma = scene.filterParams.getFloat("sigma", out.filter.sigma);
	out.filter.tau = scene.filterParams.getFloat("tau", out.filter.tau);
	// pbrt-v4's real per-kind default radius (PixelFilter::radius's own
	// comment) - resolved AFTER out.filter.kind above, since the default
	// itself depends on which kind was requested. "xradius"/"yradius" (an
	// explicit scene override) both fall back to whichever of the two was
	// actually given, defaulting to this kind's own real default when
	// neither is present.
	{
		const double kindDefault =
			(out.filter.kind == "box") ? 0.5
			: (out.filter.kind == "mitchell") ? 2.0
			: (out.filter.kind == "sinc") ? 4.0
			: (out.filter.kind == "triangle") ? 2.0
			: 1.5;  // "gaussian", or any unrecognized kind
		out.filter.radius = scene.filterParams.getFloat("xradius",
			scene.filterParams.getFloat("yradius", kindDefault));
	}

	out.regularize = scene.regularize;

	// Whether the scene has any object motion blur that bvh_aggregate_
	// hittable.h's BvhTree wrapper AND kd_tree_hittable.h's KdTree wrapper
	// both cannot correctly render: neither wrapper's intersect() has a
	// ray-time channel (see each file's own top comment), so a moving
	// sphere (center1 != center, baked by an ActiveTransform "EndTime"
	// pair - see Sphere::center1's own comment)/disk/cylinder/mesh would
	// silently render frozen at time 0 through either. Computed once, used
	// below to gate both the non-"sah" splitmethod fallback and the
	// "kdtree" accelerator-type fallback identically.
	bool hasAcceleratorIncompatibleMotion = false;
	{
		for (const Sphere &s : out.spheres) {
			if (s.center1[0] != s.center[0] || s.center1[1] != s.center[1] ||
				s.center1[2] != s.center[2]) { hasAcceleratorIncompatibleMotion = true; break; }
		}
		// Disk/Cylinder gained the same per-ray-time-channel motion blur as
		// Sphere above (see Disk::xformEnd's own comment) - the same "no
		// ray-time channel" gap applies to them too, so this check has to
		// cover both, not just Sphere.
		auto xformArrayDiffers = [](const double (&a)[16], const double (&b)[16]) {
			pbrt_scene::Matrix4 ma, mb;
			for (int i = 0; i < 16; ++i) { ma.m[i] = a[i]; mb.m[i] = b[i]; }
			return ma.differsFrom(mb);
		};
		if (!hasAcceleratorIncompatibleMotion) {
			for (const Disk &d : out.disks) {
				if (xformArrayDiffers(d.xform, d.xformEnd)) { hasAcceleratorIncompatibleMotion = true; break; }
			}
		}
		if (!hasAcceleratorIncompatibleMotion) {
			for (const Cylinder &c : out.cylinders) {
				if (xformArrayDiffers(c.xform, c.xformEnd)) { hasAcceleratorIncompatibleMotion = true; break; }
			}
		}
		// Mesh motion blur (trianglemesh/plymesh/loopsubdiv - see
		// AnimatedTriangleMesh's own comment) is the same "hit() resolves a
		// per-ray-time transform internally" shape as Sphere/Disk/Cylinder
		// above, via animated_transform_instance.h's own MotionState::
		// resolve() - the same non-SAH-BVH gap applies. (Every entry in
		// this list is animated by construction - pbrt_cpu_builder.h only
		// ever populates it when xform genuinely differs from xformEnd - so
		// presence alone is enough, no per-entry xformArrayDiffers scan
		// needed.)
		if (!hasAcceleratorIncompatibleMotion && !out.animatedTriangleMeshes.empty())
			hasAcceleratorIncompatibleMotion = true;
		// Bilinear-patch/curve motion blur (AnimatedBilinearPatch/
		// AnimatedCurve's own comments) - the identical gap, via the same
		// animated_transform_instance.h wrapper mesh motion blur uses.
		if (!hasAcceleratorIncompatibleMotion &&
			(!out.animatedBilinearPatches.empty() || !out.animatedCurves.empty()))
			hasAcceleratorIncompatibleMotion = true;
	}

	{
		std::string at = scene.acceleratorType.empty() ? "bvh" : scene.acceleratorType;
		if (at != "bvh" && at != "kdtree") {
			warn("Accelerator \"" + at + "\" is not supported (only \"bvh\"/"
				 "\"kdtree\" are) - falling back to \"bvh\"");
			at = "bvh";
		}
		if (at == "kdtree" && hasAcceleratorIncompatibleMotion) {
			warn("Accelerator \"kdtree\" is not supported together with "
				 "object motion blur (kd_tree_hittable.h's KdTree wrapper "
				 "has no ray-time channel) - falling back to \"bvh\"");
			at = "bvh";
		}
		out.acceleratorType = at;
	}
	{
		std::string sm = scene.acceleratorSplitMethod.empty()
			? "sah" : scene.acceleratorSplitMethod;
		if (sm != "sah" && sm != "middle" && sm != "equal" && sm != "hlbvh") {
			warn("Accelerator \"bvh\" \"string splitmethod\" \"" + sm +
				 "\" is not recognized (expected \"sah\"/\"middle\"/\"equal\"/"
				 "\"hlbvh\") - falling back to \"sah\"");
			sm = "sah";
		}
		// A non-"sah" split method routes through bvh_aggregate_hittable.h's
		// BvhTree<double,...> wrapper (pbrt_cpu_builder.h) instead of this
		// project's own pre-existing bvh_node - see
		// hasAcceleratorIncompatibleMotion's own comment just above for why
		// that wrapper falls back to "sah" (bvh_node, which DOES carry ray
		// time correctly) on a scene with object motion blur, rather than
		// silently freezing every moving object at time 0. Gated on
		// `out.acceleratorType == "bvh"` (already resolved just above) so a
		// scene that also asked for "kdtree" - where splitmethod is unread
		// by pbrt_cpu_builder.h either way - doesn't get a SECOND, redundant
		// motion-blur warning on top of the "kdtree"-specific one just
		// issued.
		if (sm != "sah" && hasAcceleratorIncompatibleMotion && out.acceleratorType == "bvh") {
			warn("Accelerator \"bvh\" \"string splitmethod\" \"" + sm +
				 "\" is not supported together with object motion blur "
				 "(this loader's non-SAH BVH build has no ray-time "
				 "channel) - falling back to \"sah\"");
			sm = "sah";
		}
		out.acceleratorSplitMethod = sm;
	}
	out.acceleratorMaxNodePrims = scene.acceleratorMaxNodePrims;
	out.acceleratorKdParams = scene.acceleratorKdParams;
	out.maxComponentValue = scene.maxComponentValue;

	// Film "float[4] cropwindow" / "integer[4] pixelbounds" -> a single
	// NDC-fraction rectangle. pbrt-v4's own rule: start from the full
	// frame, then intersect with cropwindow (defaults to {0,1,0,1}, a
	// no-op when absent) and with pixelbounds/xResolution/yResolution if
	// given - both may apply together, each independently narrowing the
	// region. pixelbounds is converted to a fraction using the SCENE's
	// own declared resolution (the only resolution known at this layer;
	// see FlatScene::cropX0's own comment on why fractions, not pixels,
	// are what's stored).
	{
		// cropwindow is the starting rectangle, not an intersection against
		// some other range - clamping each bound to [0,1] independently
		// (rather than max/min-ing against a running x0/x1/y0/y1 that
		// starts at exactly {0,1,0,1}, which would be a no-op restating
		// the same clamp) is all that's needed here. pixelbounds below IS
		// a real intersection, against this already-narrowed rectangle.
		double x0 = std::clamp(std::min(scene.cropWindow[0], scene.cropWindow[1]), 0.0, 1.0);
		double x1 = std::clamp(std::max(scene.cropWindow[0], scene.cropWindow[1]), 0.0, 1.0);
		double y0 = std::clamp(std::min(scene.cropWindow[2], scene.cropWindow[3]), 0.0, 1.0);
		double y1 = std::clamp(std::max(scene.cropWindow[2], scene.cropWindow[3]), 0.0, 1.0);

		if (scene.hasPixelBounds && scene.xResolution > 0 && scene.yResolution > 0) {
			const double pbx0 = std::clamp(std::min(scene.pixelBounds[0], scene.pixelBounds[1]) /
											static_cast<double>(scene.xResolution), 0.0, 1.0);
			const double pbx1 = std::clamp(std::max(scene.pixelBounds[0], scene.pixelBounds[1]) /
											static_cast<double>(scene.xResolution), 0.0, 1.0);
			const double pby0 = std::clamp(std::min(scene.pixelBounds[2], scene.pixelBounds[3]) /
											static_cast<double>(scene.yResolution), 0.0, 1.0);
			const double pby1 = std::clamp(std::max(scene.pixelBounds[2], scene.pixelBounds[3]) /
											static_cast<double>(scene.yResolution), 0.0, 1.0);
			x0 = std::max(x0, pbx0); x1 = std::min(x1, pbx1);
			y0 = std::max(y0, pby0); y1 = std::min(y1, pby1);
		}

		if (x1 <= x0 || y1 <= y0) {
			warn("Film \"cropwindow\"/\"pixelbounds\" resolve to an empty pixel "
				 "range; rendering the full frame instead");
			x0 = 0.0; x1 = 1.0;
			y0 = 0.0; y1 = 1.0;
		}

		out.cropX0 = x0; out.cropX1 = x1;
		out.cropY0 = y0; out.cropY1 = y1;
	}
}

} // namespace flatten_detail

inline FlatScene flatten(const pbrt_scene::Scene &scene,
						 const MeshResolver &meshes = {}) {
	using namespace flatten_detail;
	FlatScene out;
	out.warnings = scene.warnings;   // carry the parser's own warnings through

	flattenMaterials(scene, out);
	flattenMedia(scene, out);
	flattenLights(scene, out);
	flattenAreaLights(scene, out);
	flattenCamera(scene, out);
	flattenShapes(scene, out, meshes);
	resolveAfterShapes(scene, out);
	flattenSettings(scene, out);
	return out;
}

} // namespace pbrt_flatten
