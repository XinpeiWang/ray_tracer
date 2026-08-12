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

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"

// Refinement is exponential: every level multiplies the triangle count by
// four, so a scene asking for 8 turns a 10k-triangle cage into 650 million.
// Clamping is the difference between a slow render and an exhausted machine.
inline constexpr int kMaxSubdivLevels = 4;

namespace pbrt_flatten {

struct Triangle {
	double v[9];              // three vertices, world space, xyz each
	int material = -1;        // index into Scene::materials, -1 = pbrt default
	int areaLight = -1;       // index into Scene::areaLights, -1 = not emissive
};

struct Sphere {
	double center[3] = {0, 0, 0};
	double radius = 1.0;
	int material = -1;
	int areaLight = -1;
};

// pbrt's material set and ours are the same set under the same names - the
// MaterialType enum in gpu/optix/optix_types.h cites pbrt's BxDFs by name - so
// this is a rename, not a translation. Anything genuinely absent maps to
// Unsupported and is reported rather than quietly substituted, because a
// subsurface material silently rendered as diffuse looks plausible and wrong.
enum class MaterialKind {
	Diffuse,
	Conductor,
	Dielectric,
	ThinDielectric,
	CoatedDiffuse,
	CoatedConductor,
	DiffuseTransmission,
	Unsupported,
};

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	std::string pbrtType;              // as written, for diagnostics
	double color[3] = {0.5, 0.5, 0.5}; // reflectance / albedo
	double roughness = 0.0;
	double ior = 1.5;
};

struct Emission {
	double L[3] = {1.0, 1.0, 1.0};
	double scale = 1.0;
};

// Our camera is described the way camera.h wants it - an eye point, a target
// and a vertical field of view - rather than as pbrt's world-to-camera matrix.
struct Camera {
	double lookfrom[3] = {0, 0, 0};
	double lookat[3] = {0, 0, 1};
	double up[3] = {0, 1, 0};
	double vfov = 90.0;          // degrees, VERTICAL - see the note in flatten()
	double aperture = 0.0;
	// pbrt's own default, and it is a sentinel meaning "effectively at
	// infinity", not a measurement. See focusDistanceFor() before using it.
	double focusDistance = 1e6;
};

// The focus distance to actually give a camera, which is NOT camera.focusDistance.
//
// pbrt only uses focal distance to place the plane of sharp focus, so its
// "no depth of field" default of 1e6 is harmless there. Our camera also uses
// focus_dist to size the viewport (see camera.h's initialize()), which makes
// the primary ray's direction vector grow in proportion. Ray parameters are
// then measured in units of that vector, so the fixed t_min of 0.001 used for
// self-intersection stops rejecting hits within 0.001 world units and starts
// rejecting hits within a THOUSAND of them - silently deleting near geometry
// while distant geometry renders normally.
//
// That is not a hypothetical: it rendered a metal sphere in the bundled
// example scene as a perfectly black disc with a hard edge, which reads like
// a broken material and is not one. A test pins it.
//
// With no aperture there is no plane of focus to honour, so the distance to
// the subject is both harmless and the sane choice. With an aperture the
// scene meant something by it, but a value at the sentinel still cannot be
// used literally.
inline double focusDistanceFor(const Camera &c) {
	double toSubject = 0.0;
	for (int i = 0; i < 3; ++i) {
		const double d = c.lookat[i] - c.lookfrom[i];
		toSubject += d * d;
	}
	toSubject = std::sqrt(toSubject);
	if (toSubject <= 0.0) toSubject = 10.0;

	if (c.aperture <= 0.0) return toSubject;
	return (c.focusDistance > 0.0 && c.focusDistance < 1e5) ? c.focusDistance
														   : toSubject;
}

struct FlatScene {
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<Material> materials;    // parallel to Scene::materials
	std::vector<Emission> areaLights;   // parallel to Scene::areaLights
	Camera camera;
	std::vector<pbrt_scene::Warning> warnings;

	bool empty() const { return triangles.empty() && spheres.empty(); }
};

// Supplies a PLY mesh's positions and indices for `Shape "plymesh"`. Same
// callback shape, and for the same reason, as pbrt_scene::FileResolver: keeps
// this a pure function and lets the caller decide how a path resolves.
// Returning false means the mesh could not be read.
using MeshResolver = std::function<bool(const std::string &path,
										std::vector<float> &positions,
										std::vector<int> &indices)>;

namespace detail {

inline void transformPoint(const pbrt_scene::Matrix4 &m,
						   double x, double y, double z, double *out) {
	out[0] = m.m[0] * x + m.m[1] * y + m.m[2]  * z + m.m[3];
	out[1] = m.m[4] * x + m.m[5] * y + m.m[6]  * z + m.m[7];
	out[2] = m.m[8] * x + m.m[9] * y + m.m[10] * z + m.m[11];
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

inline MaterialKind materialKindFor(const std::string &type) {
	if (type == "diffuse")             return MaterialKind::Diffuse;
	if (type == "conductor")           return MaterialKind::Conductor;
	if (type == "dielectric")          return MaterialKind::Dielectric;
	if (type == "thindielectric")      return MaterialKind::ThinDielectric;
	if (type == "coateddiffuse")       return MaterialKind::CoatedDiffuse;
	if (type == "coatedconductor")     return MaterialKind::CoatedConductor;
	if (type == "diffusetransmission") return MaterialKind::DiffuseTransmission;
	return MaterialKind::Unsupported;   // subsurface, measured, mix, hair, ...
}

// Recovers the eye point and viewing direction from pbrt's WORLD-TO-CAMERA
// matrix by inverting it. The rotation part is orthonormal (LookAt builds it
// from normalised, mutually perpendicular axes), so the inverse is the
// transpose and the eye is -R^T * t. Doing a general 4x4 inverse here would be
// both slower and less numerically pleasant.
//
// pbrt's camera looks down +z with +y up, which is where the row picks below
// come from: R^T * (0,0,1) is R's third row, R^T * (0,1,0) is its second.
inline Camera cameraFromWorldToCamera(const pbrt_scene::Matrix4 &w2c) {
	Camera c;
	const double *m = w2c.m;

	// eye = -R^T * t
	const double tx = m[3], ty = m[7], tz = m[11];
	c.lookfrom[0] = -(m[0] * tx + m[4] * ty + m[8]  * tz);
	c.lookfrom[1] = -(m[1] * tx + m[5] * ty + m[9]  * tz);
	c.lookfrom[2] = -(m[2] * tx + m[6] * ty + m[10] * tz);

	const double fwd[3] = {m[8], m[9], m[10]};
	c.up[0] = m[4]; c.up[1] = m[5]; c.up[2] = m[6];
	for (int i = 0; i < 3; ++i) c.lookat[i] = c.lookfrom[i] + fwd[i];
	return c;
}

} // namespace detail

inline FlatScene flatten(const pbrt_scene::Scene &scene,
						 const MeshResolver &meshes = {}) {
	using namespace detail;
	FlatScene out;
	out.warnings = scene.warnings;   // carry the parser's own warnings through

	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- materials -------------------------------------------------------
	for (const pbrt_scene::MaterialDecl &md : scene.materials) {
		Material m;
		m.pbrtType = md.type;
		m.kind = materialKindFor(md.type);
		if (m.kind == MaterialKind::Unsupported) {
			warn("material type '" + md.type + "' is not supported; "
				 "it will fall back to a diffuse approximation");
		}
		// pbrt spells the base colour differently per material: reflectance for
		// diffuse, but conductors are described by their complex IOR and use
		// k/eta instead. Take whichever is present, defaulting to mid grey.
		const pbrt_scene::Vec3 def{m.color[0], m.color[1], m.color[2]};
		pbrt_scene::Vec3 c = md.params.getVec3("reflectance", def);
		if (!md.params.find("reflectance")) c = md.params.getVec3("k", c);
		m.color[0] = c.x; m.color[1] = c.y; m.color[2] = c.z;

		m.roughness = md.params.getFloat("roughness", 0.0);
		// "eta" is pbrt's name for index of refraction on dielectrics.
		m.ior = md.params.getFloat("eta", md.params.getFloat("ior", 1.5));
		out.materials.push_back(m);
	}

	// ---- area lights -----------------------------------------------------
	for (const pbrt_scene::LightDecl &ld : scene.areaLights) {
		Emission e;
		const pbrt_scene::Vec3 L = ld.params.getVec3("L", pbrt_scene::Vec3{1, 1, 1});
		e.L[0] = L.x; e.L[1] = L.y; e.L[2] = L.z;
		e.scale = ld.params.getFloat("scale", 1.0);
		// "blackbody L" declares a colour TEMPERATURE under the parameter name
		// L, so this has to test the parameter's type, not look for a
		// parameter called "blackbody" - the latter never matches and the
		// warning never fires.
		const pbrt_scene::Param *Lp = ld.params.find("L");
		if (Lp && Lp->type == "blackbody") {
			// Converting a temperature to radiance properly needs a spectral
			// pipeline; say so rather than quietly emitting the raw number.
			warn("an area light is given as a blackbody temperature, which is "
				 "approximated rather than converted spectrally");
		}
		out.areaLights.push_back(e);
	}

	// ---- camera ----------------------------------------------------------
	out.camera = cameraFromWorldToCamera(scene.worldToCamera);
	{
		const double fov = scene.cameraFov();
		// pbrt's fov applies to the NARROWER image axis. Ours is always
		// vertical, so on a landscape frame they agree and on a portrait one
		// they do not - taking pbrt's number as vertical unconditionally
		// silently mis-frames every portrait scene.
		if (scene.xResolution >= scene.yResolution) {
			out.camera.vfov = fov;
		} else {
			const double aspect = (scene.yResolution > 0)
								  ? static_cast<double>(scene.xResolution) / scene.yResolution
								  : 1.0;
			const double halfRad = fov * 0.5 * 3.14159265358979323846 / 180.0;
			const double tanV = (aspect > 0.0) ? std::tan(halfRad) / aspect : std::tan(halfRad);
			out.camera.vfov = 2.0 * std::atan(tanV) * 180.0 / 3.14159265358979323846;
		}
		out.camera.aperture = scene.cameraParams.getFloat("lensradius", 0.0) * 2.0;
		out.camera.focusDistance = scene.cameraParams.getFloat("focaldistance",
															   out.camera.focusDistance);
	}

	for (const pbrt_scene::ShapeDecl &shape : scene.shapes) {
		if (shape.type == "sphere") {
			const double r = shape.params.getFloat("radius", 1.0);
			Sphere s;
			transformPoint(shape.xform, 0.0, 0.0, 0.0, s.center);

			double sc[3];
			axisScales(shape.xform, sc);
			const double lo = std::fmin(sc[0], std::fmin(sc[1], sc[2]));
			const double hi = std::fmax(sc[0], std::fmax(sc[1], sc[2]));
			if (hi - lo > 1e-6 * std::fmax(1.0, hi)) {
				warn("a sphere carries a non-uniform scale and would be an "
					 "ellipsoid; emitted with its largest radius instead");
			}
			s.radius = r * hi;
			s.material = shape.materialIndex;
			s.areaLight = shape.areaLightIndex;
			out.spheres.push_back(s);
			continue;
		}

		if (shape.type == "trianglemesh" || shape.type == "plymesh"
				|| shape.type == "loopsubdiv") {
			std::vector<double> P;
			std::vector<int> indices;

			if (shape.type == "loopsubdiv") {
				// A subdivision surface is a control cage plus a refinement
				// rule, so it arrives looking exactly like a trianglemesh and
				// renders as a faceted lump if treated as one. loop_subdivide.h
				// is already in this project - itself a port of pbrt's own
				// loopsubdiv - so this is a refinement step, not a new feature.
				//
				// Refining in object space, before the CTM is applied, is safe
				// as well as convenient: Loop limit positions are affine
				// combinations of the control points, so subdividing and then
				// transforming gives the same surface as the reverse.
				const pbrt_scene::Param *pp = shape.params.find("P");
				const pbrt_scene::Param *pi = shape.params.find("indices");
				if (!pp || !pi) {
					warn("a loopsubdiv is missing its P or indices parameter; skipped");
					continue;
				}

				std::vector<std::array<double, 3>> cage(pp->numbers.size() / 3);
				for (std::size_t v = 0; v < cage.size(); ++v)
					cage[v] = {pp->numbers[v * 3], pp->numbers[v * 3 + 1],
							   pp->numbers[v * 3 + 2]};
				std::vector<int> cageIdx;
				cageIdx.reserve(pi->numbers.size());
				for (double d : pi->numbers) cageIdx.push_back(static_cast<int>(d));

				// Each level quadruples the triangle count, so an unbounded
				// value is a way to run out of memory rather than a way to get
				// a smoother surface. pbrt's own default is 3.
				int levels = shape.params.getInt("levels", 3);
				if (levels > kMaxSubdivLevels) {
					warn("loopsubdiv asks for " + std::to_string(levels) +
						 " levels; clamped to " + std::to_string(kMaxSubdivLevels) +
						 " (each level quadruples the triangle count)");
					levels = kMaxSubdivLevels;
				}
				if (levels < 0) levels = 0;

				const LoopSubdivResult<double> refined =
					loop_subdivide<double>(cage, cageIdx, levels);
				P.resize(refined.positions.size() * 3);
				for (std::size_t v = 0; v < refined.positions.size(); ++v) {
					P[v * 3]     = refined.positions[v][0];
					P[v * 3 + 1] = refined.positions[v][1];
					P[v * 3 + 2] = refined.positions[v][2];
				}
				indices = refined.indices;
			} else if (shape.type == "trianglemesh") {
				const pbrt_scene::Param *pp = shape.params.find("P");
				const pbrt_scene::Param *pi = shape.params.find("indices");
				if (!pp || !pi) {
					warn("a trianglemesh is missing its P or indices parameter; skipped");
					continue;
				}
				P = pp->numbers;
				indices.reserve(pi->numbers.size());
				for (double d : pi->numbers) indices.push_back(static_cast<int>(d));
			} else {
				const std::string file = shape.params.getString("filename", "");
				if (file.empty()) { warn("a plymesh has no filename; skipped"); continue; }
				if (!meshes) {
					warn("plymesh '" + file + "' skipped: no mesh resolver supplied");
					continue;
				}
				std::vector<float> pos;
				if (!meshes(file, pos, indices)) {
					warn("plymesh '" + file + "' could not be read; skipped");
					continue;
				}
				P.assign(pos.begin(), pos.end());
			}

			const std::size_t vertexCount = P.size() / 3;
			if (indices.size() % 3 != 0)
				warn("a mesh has an index count that is not a multiple of 3; "
					 "the trailing indices are ignored");

			// Transform once per vertex rather than once per index: a shared
			// vertex is referenced by several triangles, and transforming it
			// repeatedly is both slower and a source of tiny inconsistencies
			// between the same point on adjacent faces.
			std::vector<double> world(vertexCount * 3);
			for (std::size_t v = 0; v < vertexCount; ++v)
				transformPoint(shape.xform, P[v * 3], P[v * 3 + 1], P[v * 3 + 2],
							   &world[v * 3]);

			bool reportedRange = false;
			for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
				const int a = indices[i], b = indices[i + 1], c = indices[i + 2];
				if (a < 0 || b < 0 || c < 0
					|| static_cast<std::size_t>(a) >= vertexCount
					|| static_cast<std::size_t>(b) >= vertexCount
					|| static_cast<std::size_t>(c) >= vertexCount) {
					if (!reportedRange) {
						warn("a mesh has face indices outside its vertex list; "
							 "those faces are dropped");
						reportedRange = true;
					}
					continue;
				}
				Triangle t;
				for (int k = 0; k < 3; ++k) {
					t.v[0 + k] = world[static_cast<std::size_t>(a) * 3 + k];
					t.v[3 + k] = world[static_cast<std::size_t>(b) * 3 + k];
					t.v[6 + k] = world[static_cast<std::size_t>(c) * 3 + k];
				}
				t.material = shape.materialIndex;
				t.areaLight = shape.areaLightIndex;
				out.triangles.push_back(t);
			}
			continue;
		}

		// disk, cylinder, curve, bilinearmesh, ... Recognised as geometry we
		// cannot build, which is worth saying: the scene will render with a
		// hole in it rather than looking subtly wrong.
		warn("shape '" + shape.type + "' is not supported; skipped");
	}

	return out;
}

} // namespace pbrt_flatten
