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
#include <map>
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"
#include "conductor_data.h"

// Refinement is exponential: every level multiplies the triangle count by
// four, so a scene asking for 8 turns a 10k-triangle cage into 650 million.
// Clamping is the difference between a slow render and an exhausted machine.
inline constexpr int kMaxSubdivLevels = 4;

namespace pbrt_flatten {

struct Triangle {
	double v[9];              // three vertices, world space, xyz each
	// Per-vertex shading normals, world space, same layout as v. Only
	// meaningful when hasNormals - a mesh with none falls back to the flat
	// geometric normal, which is correct for genuinely faceted geometry and
	// wrong-looking for anything curved. A subdivision surface without these
	// renders as the polygon soup it was refined from.
	double n[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	bool hasNormals = false;
	int material = -1;        // index into Scene::materials, -1 = pbrt default
	int areaLight = -1;       // index into Scene::areaLights, -1 = not emissive
};

struct Sphere {
	double center[3] = {0, 0, 0};
	double radius = 1.0;
	int material = -1;
	int areaLight = -1;
	// index into FlatScene::media, -1 = no participating medium. Sphere-only
	// (not triangles/bilinear patches) because the GPU's own MaterialType::
	// Medium is sphere-only - see gpu/optix/optix_types.h's comment on that
	// enumerator - and this loader keeps both backends able to render every
	// medium-bearing shape it accepts, rather than accepting shapes GPU
	// would just drop.
	int medium = -1;
};

// Shape "disk" / "cylinder" - unlike Sphere (rotation-invariant, so baked
// straight to a world-space center+radius) these are NOT rotation-invariant,
// so baking them the sphere's way would need the same "warn on anisotropic
// scale" approximation sphere already accepts, except wrong far more often -
// an arbitrary rotation changes which way a disk faces or a cylinder's axis
// points, not just its size. So the object-space parameters (pbrt-v4's own
// convention: a disk in the z=height plane on the z-axis, a cylinder along
// the z-axis) are kept as-is, and `xform` (the CTM at the point this shape
// was declared, row-major - same convention as Instance::xform below) is
// carried through unbaked for pbrt_cpu_builder.h/pbrt_gpu_builder.h to apply
// at intersection time instead, exactly the technique transform_instance.h
// already uses for object instancing.
struct Disk {
	double radius = 1.0, innerRadius = 0.0, height = 0.0;
	// Degrees, matching PunctualLight::coneAngleDeg's own precedent just
	// above - pbrt scene-file angle params are stored as-written and
	// converted to radians at the CPU/GPU builders' point of use, not here.
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	int areaLight = -1;
	int medium = -1;
};

struct Cylinder {
	double radius = 1.0, zMin = -1.0, zMax = 1.0;
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	int areaLight = -1;
	int medium = -1;
};

// MakeNamedMedium "homogeneous" - a constant-density participating medium.
// pbrt-v4's real HomogeneousMedium is per-channel RGB sigma_a/sigma_s; both
// builders' actual medium primitive (src/TheRestOfYourLife/constant_medium.h,
// gpu/optix/optix_types.h's MaterialType::Medium) instead takes a scalar
// extinction plus a chromatic albedo tint - see pbrt_cpu_builder.h's and
// pbrt_gpu_builder.h's own comments at their point of use for how this
// struct's per-channel values are collapsed into that shape.
struct Medium {
	double sigma_a[3] = {1.0, 1.0, 1.0};
	double sigma_s[3] = {1.0, 1.0, 1.0};
	double g = 0.0;    // Henyey-Greenstein asymmetry
};

// Shape "bilinearmesh" - a single bilinear patch (4 corner points, not
// necessarily planar or a parallelogram - that generality is the point of
// the shape existing separately from a quad/trianglemesh at all). Only the
// single-patch form (`"point3 P"`, 4 points) is built; pbrt-v4's multi-patch
// form (`"integer indices"`, N patches sharing one vertex pool) still falls
// through to flatten()'s generic "shape not supported" warning - no scene
// this loader has actually seen uses it. World-space corners, laid out
// (u,v) = (0,0),(1,0),(0,1),(1,1), matching src/shared/bilinear_patch.h's
// own p00/p10/p01/p11 convention (see pbrt_cpu_builder.h/pbrt_gpu_builder.h,
// the two consumers of this layout).
struct BilinearPatch {
	double p[4][3] = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};
	int material = -1;
	int areaLight = -1;
};

// pbrt's material set and ours are the same set under the same names - the
// MaterialType enum in gpu/optix/optix_types.h cites pbrt's BxDFs by name - so
// this is a rename, not a translation. Anything genuinely absent maps to
// Unsupported and is reported rather than quietly substituted, because a
// subsurface material silently rendered as diffuse looks plausible and wrong.
//
// Subsurface IS a real, CPU-supported kind (src/TheRestOfYourLife/
// material_pbrt.h's `class subsurface` + camera.h's BSSRDF probe/exit-point
// branch) - it earns its own enumerator rather than staying Unsupported. GPU
// ALSO has a real BSSRDF implementation now (a tabulated-BSSRDF probe-walk,
// added after this comment was originally written - present on both the
// GPU-recursive and GPU-wavefront backends; see gpu/optix/
// pbrt_gpu_builder.h's Subsurface case and docs/PBRT_SUPPORT.md for the
// current per-backend support matrix), so both backends render this for
// real rather than falling back to flat diffuse.
//
// Measured is the same story: a real, CPU-supported kind (src/
// TheRestOfYourLife/material_pbrt.h's `class measured`, backed by src/
// shared/measured_bxdf.h's ported pbrt-v4 MeasuredBxDF + src/shared/
// measured_bxdf_loader.h's .bsdf tensor-file reader) rather than the flat
// diffuse fallback every other Unsupported material still gets. GPU also
// flattens and uploads the same tensor tables now (see gpu/optix/
// pbrt_gpu_builder.h's Measured case) - both backends fall back to flat
// diffuse only on the shared "filename didn't resolve/load" gate, not as a
// standing GPU limitation. See docs/PBRT_SUPPORT.md for the current
// per-backend support matrix.
enum class MaterialKind {
	Diffuse,
	Conductor,
	Dielectric,
	ThinDielectric,
	CoatedDiffuse,
	CoatedConductor,
	DiffuseTransmission,
	Subsurface,
	Measured,
	// A stochastic blend of two OTHER materials (pbrt-v4 MixMaterial),
	// referenced by name via the "materials" parameter - see Material::
	// mixMaterialA/mixMaterialB/mixWeight below and flatten()'s own mix-
	// resolution code for how those names become indices. CPU-only: this
	// codebase's own src/TheRestOfYourLife/material_pbrt.h `class
	// mix_material` (added earlier for SPPM support) is a real, generic
	// two-material blend - not SPPM-specific - so this backs a pbrt "mix"
	// material directly, real support rather than a documented
	// approximation. GPU has no equivalent (mix_material's own header
	// comment already says so, for the same class) and keeps rendering it
	// as flat diffuse - see gpu/optix/pbrt_gpu_builder.h's own comment on
	// its Mix case, which exists only to preserve that fallback, now blended
	// by colour rather than plain grey, now that this enumerator is no
	// longer an alias for Unsupported.
	Mix,
	Unsupported,
};

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	std::string pbrtType;              // as written, for diagnostics
	double color[3] = {0.5, 0.5, 0.5}; // reflectance / albedo
	double roughness = 0.0;
	double ior = 1.5;
	// DiffuseTransmission only: the light that passes through rather than
	// reflects. pbrt-v4's own default (0.25) is closer to that material's
	// intent than reusing `color`'s 0.5 default would be - a
	// diffusetransmission with neither parameter set should look like a
	// frosted panel passing about a quarter of the light each way, not a
	// mirror-symmetric reflectance/transmittance split at 0.5/0.5.
	double transmittance[3] = {0.25, 0.25, 0.25};

	// Subsurface only: absorption/scattering coefficients, ALREADY multiplied
	// by the material's "scale" parameter (pbrt-v4 GetBSSRDF: sig_a = scale *
	// sigma_a, sig_s = scale * sigma_s - see flatten()'s subsurface branch).
	// Defaults are pbrt-v4's own "nothing specified" preset
	// (SubsurfaceMaterial::Create's case 4, materials.cpp), which happens to
	// equal the "Wholemilk" named preset.
	double sigma_a[3] = {0.0011, 0.0024, 0.014};
	double sigma_s[3] = {2.55, 3.21, 3.77};
	double g = 0.0;   // Henyey-Greenstein asymmetry (subsurface only)

	// Measured only: the "filename" parameter naming the .bsdf tensor file,
	// exactly AS WRITTEN in the scene ("bsdfs/foo.bsdf" - relative to the
	// scene file's own directory, same convention as Shape "plymesh"'s
	// filename or LightSource "infinite"'s). This header stays filesystem-
	// free by design (see the file comment), so it cannot resolve or read
	// the file itself - pbrt_load.h::loadFile() does both AFTER flatten()
	// returns, exactly as it already does for the infinite light's image,
	// and OVERWRITES this field with the resolved path on success so
	// pbrt_cpu_builder.h's `class measured` never needs to know the scene's
	// directory. Empty means "no filename given" (or, after pbrt_load.h's
	// pass, "could not be resolved/loaded") - either way the material falls
	// back to a diffuse approximation.
	std::string measuredFilename;

	// Diffuse only: an "imagemap" Texture bound to "reflectance", naming the
	// image file exactly AS WRITTEN in the scene - same "stays filesystem-
	// free, resolved later by pbrt_load.h" convention as measuredFilename
	// above (see that field's own comment). Empty means either no texture was
	// bound, the bound texture wasn't an imagemap, or (after pbrt_load.h's
	// pass) the file could not be found - any of which falls back to `color`
	// as a flat reflectance, same as today. Only "reflectance" is handled
	// (not every texture-bindable parameter on every material kind): pbrt's
	// ganesha is the motivating case (a Diffuse statue whose reflectance is
	// an imagemap - see the warning loop below), and scoping to that one
	// parameter keeps this addition bounded rather than building a general
	// procedural-texture pipeline in one pass.
	std::string textureFilename;

	// A pbrt Shape's own "alpha" parameter (bound to a "float"/"imagemap"
	// Texture - e.g. barcelona-pavilion's foliage, "Shape \"plymesh\"
	// \"texture alpha\" [ \"leaf_alpha\" ]"), NOT a Material directive
	// parameter - pbrt's alpha-cutout mask is authored per-Shape, one leaf
	// mesh at a time, reusing the same colour texture's alpha/luminance
	// channel. Stored here anyway (rather than as a new field on Triangle/
	// FlatScene) because both this loader's Material-building convention
	// (measuredFilename/textureFilename, both "raw as written, resolved
	// later by pbrt_load.h") and the GPU backend's actual layout
	// (MaterialData::alphaMaskTexIdx is per-material, not per-triangle) are
	// already per-material - flatten() writes this onto whichever Material
	// the owning Shape resolved to (out.materials[shape.materialIndex]) at
	// the point the Shape is processed. Every scene in this loader's own
	// corpus that uses "texture alpha" gives each alpha-masked Shape its own
	// unnamed Material declared immediately before it (never a NamedMaterial
	// shared by shapes with different alpha masks), so this 1:1 shape<->
	// material correspondence holds in practice; a scene that violated it
	// would have the last such Shape's alpha win for every shape sharing
	// that material index - not enforced, but not a case this loader's own
	// bundled scenes exercise either.
	std::string alphaTextureFilename;

	// Mix only: indices into FlatScene::materials of the two blended
	// sub-materials - pbrt-v4's own "string materials" parameter names them
	// (an array of exactly two named-material references, resolved against
	// every MakeNamedMaterial in the scene, not just ones declared earlier -
	// see flatten()'s own mix-resolution code), and mixWeight is pbrt-v4's
	// "amount" (default 0.5): the probability of material B winning at any
	// given shading point, matching mix_material's own "weight" parameter
	// exactly (0 = pure A, 1 = pure B). -1 means "could not be resolved" -
	// fewer than two names given, or a name that does not match any
	// MakeNamedMaterial - in which case `kind` is downgraded to Unsupported
	// by flatten() itself (see there), so pbrt_cpu_builder.h never has to
	// check these fields are valid before indexing with them.
	int mixMaterialA = -1;
	int mixMaterialB = -1;
	double mixWeight = 0.5;

	// Conductor only: real complex IOR, resolved when "spectrum eta"/
	// "spectrum k" named one of this codebase's known metals (see
	// conductorElementFromSpectrumName()/src/shared/conductor_data.h's
	// FindConductorPreset() below) - pbrt's conductors are normally
	// described this way ("metal-Ag-eta"/"metal-Ag-k"), not via plain
	// floats/RGB, which flatten() used to just fail to parse silently.
	// hasConductorPreset stays false (and the builders keep falling back to
	// the existing metal/fuzz-mirror approximation, exactly as before this
	// field existed) for an explicit RGB k, an unrecognized named spectrum,
	// or no eta/k given at all.
	bool hasConductorPreset = false;
	double conductorEta[3] = {0.0, 0.0, 0.0};
	double conductorK[3] = {0.0, 0.0, 0.0};
};

// Extracts "Ag" from pbrt-v4's "metal-Ag-eta"/"metal-Ag-k" named-spectrum
// convention (the only shape this loader's bundled scene corpus uses for
// conductor eta/k - see conductor_data.h's own table for which elements are
// recognized once extracted). Returns "" for anything that doesn't match
// the "metal-<elem>-eta"/"metal-<elem>-k" shape at all (an explicit RGB
// value, or an unrecognized/non-metal named spectrum).
inline std::string conductorElementFromSpectrumName(const std::string &name) {
	const std::string prefix = "metal-";
	if (name.rfind(prefix, 0) != 0) return "";
	for (const char *suffix : {"-eta", "-k"}) {
		const std::string suf(suffix);
		if (name.size() > prefix.size() + suf.size() &&
			name.compare(name.size() - suf.size(), suf.size(), suf) == 0) {
			return name.substr(prefix.size(), name.size() - prefix.size() - suf.size());
		}
	}
	return "";
}

struct Emission {
	double L[3] = {1.0, 1.0, 1.0};
	double scale = 1.0;
};

// LightSource "infinite" - a scene's environment/sky light. This is the one
// non-area light kind worth carrying through here, because it is usually a
// scene's main illumination (see flatten()'s own comment on why dropping it
// silently is worse than most warnings). All 5 punctual kinds (distant,
// point, spot, goniometric, projection) are also supported - see
// PunctualLight below - not dropped; see docs/PBRT_SUPPORT.md for the full
// per-light-kind CPU/GPU support matrix.
struct InfiniteLight {
	bool present = false;
	double L[3] = {1.0, 1.0, 1.0};   // used as-is when imageWidth/imageHeight are 0
	double scale = 1.0;
	std::string imageFile;           // as named by the scene; empty = constant colour only
	// Decoded pixel data, filled in by pbrt_load::loadFile() AFTER flatten()
	// returns - this header stays filesystem-free by design (see the file
	// comment), so it cannot itself resolve or decode imageFile. Row-major,
	// 3 floats/pixel, linear. imageWidth/imageHeight are 0 until (and unless)
	// that happens, which is also how a caller tells "decode did not run yet
	// or failed" apart from "this scene has no image, only a constant L".
	std::vector<float> imagePixels;
	int imageWidth = 0;
	int imageHeight = 0;
	// The CTM at the LightSource directive, world -> light space. An
	// environment map's sun/horizon faces the wrong way if this is dropped -
	// not black, but visibly wrong, which is easy to miss without a scene
	// that actually has a directional feature to check against.
	pbrt_scene::Matrix4 xform;
};

// pbrt-v4's five punctual (delta-distribution) LightSource kinds - "point",
// "spot", "distant", "goniometric" and "projection". Unlike "infinite" there
// can genuinely be several of these in one scene (a room lit by three
// spotlights, say), so they collect into FlatScene::punctualLights rather
// than a single optional field the way InfiniteLight does.
//
// Rendering support for every one of these already exists on both backends -
// src/TheRestOfYourLife/punctual_light_objects.h (CPU) and
// gpu/optix/optix_types.h's PunctualLightGPU/PunctualLightKind (GPU), proven
// by this codebase's own hand-built showcase scenes C2-C6
// (scenes_advanced.h). This struct only has to carry each type's pbrt
// parameters far enough for pbrt_cpu_builder.h/pbrt_gpu_builder.h to feed
// those existing constructors - it is a parsing/bridging job, not new
// rendering math.
enum class PunctualLightKind {
	Point,
	Spot,
	Distant,
	Goniometric,
	Projection,
};

struct PunctualLight {
	PunctualLightKind kind = PunctualLightKind::Point;

	// Point/Spot/Goniometric/Projection: world-space position - pbrt's
	// "from" point (default the origin) run through the LightSource
	// directive's CTM, the same treatment InfiniteLight::xform documents.
	// Goniometric/Projection have no "from"/"to" of their own in pbrt-v4 (see
	// worldToLight below for how they instead use the CTM's rotation), so
	// this is simply the CTM applied to the origin for those two kinds.
	double pos[3] = {0, 0, 0};

	// Spot: world-space unit direction the cone points toward - pbrt's "to"
	// minus "from" (defaults (0,0,1) and (0,0,0)), both CTM-transformed,
	// then renormalized.
	// Distant: world-space unit direction FROM ANY POINT TOWARD THE LIGHT -
	// exactly the `wi` this loader's punctual_light_list/PunctualLightGPU
	// already expect (DistantLightData<T>::sample_wi returns this field
	// verbatim, and camera.h's NEE block casts its shadow ray straight down
	// it - see flatten()'s own comment at the distant-light parsing site for
	// the from/to sign derivation). NOT the direction sunlight travels,
	// which is this vector's negation.
	double dir[3] = {0, 0, 1};

	// Point/Spot/Goniometric: pbrt's "I" (peak intensity, RGB, candela).
	// Distant: pbrt's "L" (radiance, RGB). Unused for Projection - pbrt-v4's
	// ProjectionLight has no "I"/"L" of its own; the projected image supplies
	// colour directly (see the image-file comment on `fovDeg` below).
	double intensity[3] = {1, 1, 1};

	// pbrt's "scale" parameter, read the same way for all five kinds.
	double scale = 1.0;

	// Spot only, in degrees. pbrt's "coneangle" is the OUTER edge (falloff
	// reaches zero here); the INNER edge (full intensity inside) is
	// coneangle - conedeltaangle, matching pbrt-v4 SpotLight::Create exactly.
	double coneAngleDeg = 30.0;
	double falloffStartAngleDeg = 25.0;

	// Distant only. No pbrt parameter feeds this - pbrt-v4 itself only uses
	// a scene's bounding radius to place DistantLight's virtual "reference
	// point at infinity" for bidirectional techniques, and this loader's
	// punctual lights are visited deterministically every NEE step rather
	// than power-sampled (see camera.h's punct_lights block) - so it does
	// not affect any image this loader actually renders. Kept at the same
	// 1000.0 default scenes_advanced.h's own hand-built distant-light scene
	// (C3, build_distant_light_punct()) uses, for parity if that ever
	// changes.
	double sceneRadius = 1000.0;

	// Goniometric/Projection only: world -> light rotation, row-major 3x3,
	// recovered from the LightSource directive's CTM by inverting its
	// upper-left 3x3 (see detail::worldToLightRotation()) - both kinds have
	// no "from"/"to" of their own in pbrt-v4, so a scene aims either one
	// purely by rotating the CTM before the LightSource directive (e.g.
	// `Rotate` then `LightSource "projection" ...`). Identity when the CTM
	// is a pure translation, which covers every scene this loader's own
	// corpus (C5/C6) uses either kind in.
	double worldToLight[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

	// Projection only, degrees, pbrt's "fov".
	double fovDeg = 90.0;

	// Both image-based kinds (Goniometric/Projection) name a "filename" in
	// real pbrt-v4 scenes - an IES-style directional profile for
	// goniometric, the projected slide image for projection - neither of
	// which this loader decodes. This is not a new gap: this codebase's own
	// hand-built showcase scenes for both kinds (scenes_advanced.h's
	// build_goniometric_punct()/build_projection_punct(), registry ids
	// C5/C6) use a synthetic procedural pattern rather than a real image
	// file too, so falling back to a uniform pattern here (see
	// pbrt_cpu_builder.h/pbrt_gpu_builder.h's construction of these) matches
	// the precedent this codebase already set for these exact light kinds
	// rather than opening a new one. Only recorded so flatten() can warn
	// when a scene actually names a file - the loss of the real projected
	// picture/profile shape is then a legible, documented approximation
	// rather than a silently plain light.
	bool hadImageFilename = false;
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

	// Non-perspective cameras. "perspective" (the default) uses only the
	// fields above, exactly as before. Anything else additionally carries
	// its own parameters, read from the Camera directive's own parameter
	// list - lookfrom/lookat/up above still apply to all of them (they come
	// from the world-to-camera matrix, which every pbrt camera type shares).
	std::string type = "perspective";                  // as pbrt spells it
	std::string sphericalMapping = "equirectangular";   // spherical/environment only
	std::string lensFile;                               // realistic only: path, relative to the scene file
	double filmDiagonalMM = 35.0;      // realistic only - pbrt-v4's own default
	double apertureDiameterMM = 1.0;   // realistic only - pbrt-v4's own default

	// Orthographic only: pbrt's optional explicit screen-window override
	// (xmin, xmax, ymin, ymax, in world units - orthographic has no fov to
	// derive a scale from any other way). Without it, an orthographic
	// camera falls back to a screen window sized for a roughly 1-unit-across
	// scene, which is pbrt's own default too - a real scene authored at a
	// larger scale is expected to give this explicitly, the same as it
	// would have to for real pbrt.
	bool hasScreenWindow = false;
	double screenWindow[4] = {-1.0, 1.0, -1.0, 1.0};  // xmin, xmax, ymin, ymax
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

// The defocus_angle to actually give camera.h's CameraConfig, which is NOT
// c.aperture. c.aperture is set (a few dozen lines below, where "lensradius"
// is read) to pbrt's lensradius*2 - a world-space lens DIAMETER - while
// camera.h's defocus_angle is a full-angle measurement in DEGREES
// (defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle/2)),
// camera.h's initialize()). Passing the world-space diameter straight into a
// degrees field (as callers used to) isn't a unit conversion away from
// correct, it's simply the wrong quantity - e.g. pbrt's "lensradius 0.1" at a
// focus distance of 10 world units should barely blur the image, but read as
// "0.2 degrees" the defocus disk is spuriously enormous or vanishingly small
// depending on the scene's actual scale, essentially unrelated to what the
// scene file asked for. Solves defocus_radius = lens_radius for defocus_angle
// given the same focus_dist this camera will actually be built with.
inline double defocusAngleDegreesFor(const Camera &c, double focus_dist) {
	if (c.aperture <= 0.0 || focus_dist <= 0.0) return 0.0;
	const double lens_radius = c.aperture * 0.5;
	return 2.0 * (std::atan(lens_radius / focus_dist) * 180.0 / 3.14159265358979323846);
}

// Geometry that exists once and is drawn many times, in OBJECT space - the one
// place in this header where the CTM is deliberately not baked, because baking
// it is exactly what instancing exists to avoid.
struct InstanceGroup {
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
};

struct Instance {
	int group = -1;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};   // object -> world, row major
};

struct FlatScene {
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<Disk> disks;
	std::vector<Cylinder> cylinders;
	std::vector<BilinearPatch> bilinearPatches;
	std::vector<Material> materials;    // parallel to Scene::materials
	std::vector<Emission> areaLights;   // parallel to Scene::areaLights
	std::vector<Medium> media;          // parallel to Scene::media
	InfiniteLight infiniteLight;        // present=false if the scene has none
	std::vector<PunctualLight> punctualLights;   // LightSource point/spot/distant/goniometric/projection

	// Instanced geometry. `groups` hold object-space shapes; `instances` place
	// them. A backend that ignores these renders a scene missing everything
	// that was instanced, so both builders must handle them.
	std::vector<InstanceGroup> groups;
	std::vector<Instance> instances;

	Camera camera;
	std::vector<pbrt_scene::Warning> warnings;

	bool empty() const {
		return triangles.empty() && spheres.empty() && instances.empty();
	}
};

// Supplies a PLY mesh's positions and indices for `Shape "plymesh"`. Same
// callback shape, and for the same reason, as pbrt_scene::FileResolver: keeps
// this a pure function and lets the caller decide how a path resolves.
// Returning false means the mesh could not be read.
using MeshResolver = std::function<bool(const std::string &path,
										std::vector<float> &positions,
										std::vector<int> &indices)>;

namespace detail {

// One shape to emit, where to put it, and under which transform. Routing the
// single shape loop through this is what lets the same code serve three
// callers - scene geometry, an instance definition's object-space geometry,
// and an instanced emitter baked into world space - without three copies of
// the triangulation, subdivision and PLY handling.
struct ShapeWork {
	const pbrt_scene::ShapeDecl *shape = nullptr;
	pbrt_scene::Matrix4 xform;
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
	return MaterialKind::Unsupported;   // hair, ...
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
	// Named materials, by name -> index in scene.materials. Built BEFORE the
	// main loop below (rather than incrementally during it) because a "mix"
	// material's "materials" parameter can name one declared LATER in the
	// file - pbrt itself resolves MixMaterial's sub-materials against the
	// full scene, not just what came before the mix directive - and because
	// flatten() mirrors scene.materials into out.materials 1:1 (same order,
	// no skipped entries), an index found here is already the right index
	// into out.materials too.
	std::map<std::string, int> namedMaterialIndex;
	for (std::size_t i = 0; i < scene.materials.size(); ++i)
		if (!scene.materials[i].name.empty())
			namedMaterialIndex[scene.materials[i].name] = static_cast<int>(i);

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

		// A parameter bound to a Texture is silently worth a warning: the
		// material still renders, but with a constant colour where the scene
		// asked for a pattern. pbrt's ganesha is the case that showed this -
		// its statue's reflectance is an imagemap, and without a word about
		// it the render just looks like a shading bug. "reflectance" bound to
		// an "imagemap" texture on a Diffuse material is the one case
		// actually wired up below (Material::textureFilename's own comment,
		// pbrt_cpu_builder.h's MaterialKind::Diffuse case) - gated on `kind`
		// so a coateddiffuse/conductor/etc. with the same binding still warns
		// instead of silently keeping its texture reference unused; every
		// other texture binding (other parameters, other texture classes like
		// checkerboard/scale/mix) also still just warns.
		for (const pbrt_scene::Param &p : md.params.items) {
			if (p.type != "texture") continue;
			if (m.kind == MaterialKind::Diffuse && p.name == "reflectance" && !p.strings.empty()) {
				const pbrt_scene::TextureDecl *tex = nullptr;
				for (const pbrt_scene::TextureDecl &t : scene.textures)
					if (t.name == p.strings[0]) tex = &t;
				if (tex && tex->cls == "imagemap") {
					const std::string filename = tex->params.getString("filename", "");
					if (!filename.empty()) {
						m.textureFilename = filename;
						continue;   // resolved to an image, not a "not supported" warning
					}
				}
			}
			warn("material '" + md.type + "' binds its '" + p.name +
				 "' to a texture, which is not supported; a constant colour is "
				 "used instead");
		}

		// "roughness" is pbrt's isotropic spelling; a scene that instead gives
		// separate "uroughness"/"vroughness" (its anisotropic spelling) has no
		// isotropic value to find, and without this fallback chain silently
		// renders as a perfect mirror (roughness 0) regardless of what either
		// one said - a much larger visual difference than the isotropic
		// approximation this settles for (using whichever of the two is
		// present; a scene with both must average them by hand, but that is
		// rare in practice and this format has no field for true anisotropy).
		m.roughness = md.params.getFloat("roughness",
							md.params.getFloat("uroughness",
								md.params.getFloat("vroughness", 0.0)));
		// "eta" is pbrt's name for index of refraction on dielectrics.
		m.ior = md.params.getFloat("eta", md.params.getFloat("ior", 1.5));

		// Conductor only: pbrt describes a conductor's complex IOR via
		// "spectrum eta"/"spectrum k" bound to a NAMED spectrum
		// ("metal-Ag-eta"/"metal-Ag-k", etc.), not the plain floats/RGB
		// getFloat()/getVec3() above can read - resolve the common named-
		// metal case against this codebase's own RGB-approximated conductor
		// table instead of always falling back to the metal/fuzz-mirror
		// approximation (getString() only inspects the param's `strings`
		// vector, so this is safe to call even when "eta"/"k" turn out to be
		// an explicit RGB value instead - it just won't find one there).
		if (m.kind == MaterialKind::Conductor) {
			std::string elem = conductorElementFromSpectrumName(md.params.getString("eta", ""));
			if (elem.empty()) elem = conductorElementFromSpectrumName(md.params.getString("k", ""));
			if (const ConductorPreset* preset = elem.empty() ? nullptr : FindConductorPreset(elem.c_str())) {
				m.hasConductorPreset = true;
				m.conductorEta[0] = preset->eta_r; m.conductorEta[1] = preset->eta_g; m.conductorEta[2] = preset->eta_b;
				m.conductorK[0]   = preset->k_r;   m.conductorK[1]   = preset->k_g;   m.conductorK[2]   = preset->k_b;
			}
		}

		// DiffuseTransmission only, but harmless to read unconditionally: no
		// other material kind has a "transmittance" parameter to collide with.
		const pbrt_scene::Vec3 defT{m.transmittance[0], m.transmittance[1], m.transmittance[2]};
		const pbrt_scene::Vec3 t = md.params.getVec3("transmittance", defT);
		m.transmittance[0] = t.x; m.transmittance[1] = t.y; m.transmittance[2] = t.z;

		// Subsurface: resolve sigma_a/sigma_s the way pbrt-v4's own
		// SubsurfaceMaterial::Create does (materials.cpp) - "4 mutually
		// exclusive ways to specify the subsurface properties", of which this
		// supports the first three (named preset, explicit sigma_a+sigma_s,
		// and "nothing specified" defaults). The fourth (reflectance+mfp,
		// inverted through the diffusion table via SubsurfaceFromDiffuse)
		// would need a BSSRDFTable built here at PARSE time just to invert
		// one number, for a form no scene in this loader's corpus uses; it
		// warns and falls back to the default coefficients instead.
		if (m.kind == MaterialKind::Subsurface) {
			const double scale = md.params.getFloat("scale", 1.0);
			double g = md.params.getFloat("g", 0.0);
			double sigA[3] = {m.sigma_a[0], m.sigma_a[1], m.sigma_a[2]};
			double sigS[3] = {m.sigma_s[0], m.sigma_s[1], m.sigma_s[2]};

			const std::string name = md.params.getString("name", "");
			const bool hasSigmaA = md.params.find("sigma_a") != nullptr;
			const bool hasSigmaS = md.params.find("sigma_s") != nullptr;

			if (!name.empty()) {
				if (const SubsurfacePreset *preset = subsurfacePresetFor(name)) {
					for (int c = 0; c < 3; ++c) {
						sigA[c] = preset->sigmaA[c];
						sigS[c] = preset->sigmaPrimeS[c];
					}
					// pbrt-v4: "Enforce g=0 (the database specifies reduced
					// scattering coefficients)".
					if (g != 0.0)
						warn("material 'subsurface' ignores \"g\" when \"name\" "
							 "selects a measured scattering preset (matches pbrt-v4)");
					g = 0.0;
				} else {
					warn("material 'subsurface' names unknown scattering preset '" +
						 name + "'; using the default coefficients instead");
				}
			} else if (hasSigmaA && hasSigmaS) {
				const pbrt_scene::Vec3 a =
					md.params.getVec3("sigma_a", pbrt_scene::Vec3{sigA[0], sigA[1], sigA[2]});
				const pbrt_scene::Vec3 s =
					md.params.getVec3("sigma_s", pbrt_scene::Vec3{sigS[0], sigS[1], sigS[2]});
				sigA[0] = a.x; sigA[1] = a.y; sigA[2] = a.z;
				sigS[0] = s.x; sigS[1] = s.y; sigS[2] = s.z;
			} else if (hasSigmaA != hasSigmaS) {
				warn("material 'subsurface' gives only one of \"sigma_a\"/\"sigma_s\"; "
					 "both are required together, so the default coefficients are used instead");
			} else if (md.params.find("reflectance")) {
				warn("material 'subsurface' gives \"reflectance\" without \"sigma_a\"/"
					 "\"sigma_s\"; this loader does not invert reflectance+mfp into "
					 "scattering coefficients, so the default coefficients are used instead");
			}
			// else: nothing specified at all -- m.sigma_a/sigma_s's own
			// defaults (already pbrt-v4's "nothing specified" preset) stand.

			for (int c = 0; c < 3; ++c) {
				m.sigma_a[c] = sigA[c] * scale;
				m.sigma_s[c] = sigS[c] * scale;
			}
			m.g = g;

			// pbrt-v4's SubsurfaceMaterial defaults eta to 1.33 (skin/water-
			// like), not the 1.5 (glass-like) default the generic "eta"/"ior"
			// read above already applied for every material kind - redo it
			// here with the right default when the scene gave neither.
			if (!md.params.find("eta") && !md.params.find("ior"))
				m.ior = 1.33;
		}

		// Measured: just the as-written filename (see Material::
		// measuredFilename's own comment for why resolution/loading happens
		// later, in pbrt_load.h). pbrt-v4's own MeasuredMaterial has no
		// other parameters worth reading - reflectance/roughness/eta above
		// don't apply to a tabulated BRDF.
		if (m.kind == MaterialKind::Measured) {
			m.measuredFilename = md.params.getString("filename", "");
			if (m.measuredFilename.empty()) {
				warn("material 'measured' has no \"filename\"; "
					 "it will fall back to a diffuse approximation");
			}
		}

		// Mix: resolve "materials" (two names) against namedMaterialIndex,
		// built above. A "mix" that cannot be resolved is downgraded to
		// Unsupported here rather than left half-populated - the generic
		// diffuse fallback below the main loop already exists and is exactly
		// what an unresolvable mix should do, so reusing it (instead of a
		// separate broken-mix code path in pbrt_cpu_builder.h) means that
		// file never has to check mixMaterialA/B are valid before indexing
		// with them (see those fields' own comment).
		if (m.kind == MaterialKind::Mix) {
			const pbrt_scene::Param *mats = md.params.find("materials");
			if (!mats || mats->strings.size() < 2) {
				warn("material 'mix' needs two names in \"materials\"; "
					 "it will fall back to a diffuse approximation");
				m.kind = MaterialKind::Unsupported;
			} else {
				const auto itA = namedMaterialIndex.find(mats->strings[0]);
				const auto itB = namedMaterialIndex.find(mats->strings[1]);
				if (itA == namedMaterialIndex.end() || itB == namedMaterialIndex.end()) {
					warn("material 'mix' names an unknown material in \"materials\"; "
						 "it will fall back to a diffuse approximation");
					m.kind = MaterialKind::Unsupported;
				} else {
					m.mixMaterialA = itA->second;
					m.mixMaterialB = itB->second;
					// pbrt-v4's own default; "amount" may instead be bound to
					// a texture, which the generic texture-binding warning
					// above (the `for (const pbrt_scene::Param &p : ...)`
					// loop near the top of this material's handling) already
					// reports - this just reads the constant fallback either
					// way, same as every other texture-eligible parameter in
					// this function.
					m.mixWeight = md.params.getFloat("amount", 0.5);
				}
			}
		}

		out.materials.push_back(m);
	}

	// ---- media -------------------------------------------------------------
	// MakeNamedMedium, mirrored 1:1 into out.media (same convention as
	// materials above) - ShapeDecl::insideMedium already resolved names to
	// indices during parsing (see pbrt_scene.h's MediumInterface dispatch),
	// so this is a straight per-channel copy, not a second name lookup.
	for (const pbrt_scene::MediumDecl &md : scene.media) {
		Medium medium;
		if (md.type != "homogeneous") {
			warn("medium type '" + md.type + "' is not supported; "
				 "treated as homogeneous with its given sigma_a/sigma_s");
		}
		const pbrt_scene::Vec3 defA{medium.sigma_a[0], medium.sigma_a[1], medium.sigma_a[2]};
		const pbrt_scene::Vec3 defS{medium.sigma_s[0], medium.sigma_s[1], medium.sigma_s[2]};
		pbrt_scene::Vec3 sa = md.params.getVec3("sigma_a", defA);
		pbrt_scene::Vec3 ss = md.params.getVec3("sigma_s", defS);
		const double scale = md.params.getFloat("scale", 1.0);
		medium.sigma_a[0] = sa.x * scale; medium.sigma_a[1] = sa.y * scale; medium.sigma_a[2] = sa.z * scale;
		medium.sigma_s[0] = ss.x * scale; medium.sigma_s[1] = ss.y * scale; medium.sigma_s[2] = ss.z * scale;
		medium.g = md.params.getFloat("g", 0.0);
		out.media.push_back(medium);
	}

	// ---- lights that are not area lights ---------------------------------
	// "infinite" (the environment/sky light) is carried through below - it is
	// usually a scene's main illumination, so dropping it silently produces a
	// render that looks exactly like a shading bug and sends you hunting
	// through the BSDF code instead. point/spot/distant/goniometric/
	// projection are carried through too, into FlatScene::punctualLights -
	// see PunctualLight's own comment for why this is a bridging job rather
	// than new rendering math (this codebase's own C2-C6 showcase scenes
	// already exercise every one of these five on both backends). Anything
	// else genuinely unknown still gets the warn-and-drop this whole block
	// used to give every non-infinite kind.
	//
	// Only the LAST "infinite" LightSource in the scene wins if there is more
	// than one - real pbrt scenes have at most one (this codebase has never
	// seen otherwise among the scenes it loads), and picking the last one
	// matches how pbrt's own graphics-state model would apply them (each
	// later directive's effect is visible, not merged). Punctual lights have
	// no such restriction - a scene can genuinely want three spotlights - so
	// every one of those is kept, not just the last.
	for (const pbrt_scene::LightDecl &ld : scene.lights) {
		if (ld.type == "infinite") {
			out.infiniteLight.present = true;
			const pbrt_scene::Vec3 L = ld.params.getVec3("L", pbrt_scene::Vec3{1, 1, 1});
			out.infiniteLight.L[0] = L.x; out.infiniteLight.L[1] = L.y; out.infiniteLight.L[2] = L.z;
			out.infiniteLight.scale = ld.params.getFloat("scale", 1.0);
			out.infiniteLight.imageFile = ld.params.getString("filename", "");
			out.infiniteLight.xform = ld.xform;
			continue;
		}

		if (ld.type == "point") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Point;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			transformPoint(ld.xform, from.x, from.y, from.z, pl.pos);
			const pbrt_scene::Vec3 I = ld.params.getVec3("I", pbrt_scene::Vec3{1, 1, 1});
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "spot") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Spot;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			const pbrt_scene::Vec3 to = ld.params.getVec3("to", pbrt_scene::Vec3{0, 0, 1});
			double worldFrom[3], worldTo[3];
			transformPoint(ld.xform, from.x, from.y, from.z, worldFrom);
			transformPoint(ld.xform, to.x, to.y, to.z, worldTo);
			for (int c = 0; c < 3; ++c) pl.pos[c] = worldFrom[c];
			// The cone axis - the direction the spot is AIMED, from `from`
			// toward `to` - both already CTM-transformed, so subtracting
			// cancels the translation and leaves exactly the CTM's rotation
			// applied to pbrt's own local (to - from) (matches pbrt-v4
			// SpotLight::Create's w = Normalize(to - from) exactly, just
			// composed with the CTM by transforming the endpoints instead of
			// building pbrt's own dirToZ rotation matrix - unnecessary here
			// since SpotLightData only ever needs the resulting axis, not a
			// full light-space frame).
			for (int c = 0; c < 3; ++c) pl.dir[c] = worldTo[c] - worldFrom[c];
			normalizeOrDefault(pl.dir, 0, -1, 0);
			const pbrt_scene::Vec3 I = ld.params.getVec3("I", pbrt_scene::Vec3{1, 1, 1});
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			pl.coneAngleDeg = ld.params.getFloat("coneangle", 30.0);
			// pbrt-v4's own SpotLight::Create passes coneangle - conedeltaangle
			// through UNCLAMPED (see SpotLight's cosFalloffStart/cosFalloffEnd
			// construction) - a conedeltaangle larger than coneangle yields a
			// negative falloffStartAngleDeg, which is not malformed: cos() is
			// even, so cos(-10deg) == cos(10deg), meaning a negative start angle
			// still produces a real, meaningful full-intensity core (just one
			// that extends slightly past the geometric cone axis) rather than
			// collapsing to cosFalloffStart=1.0 (no core at all), which is what
			// clamping to 0 here used to produce.
			const double delta = ld.params.getFloat("conedeltaangle", 5.0);
			pl.falloffStartAngleDeg = pl.coneAngleDeg - delta;
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "distant") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Distant;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			const pbrt_scene::Vec3 to = ld.params.getVec3("to", pbrt_scene::Vec3{0, 0, 1});
			double worldFrom[3], worldTo[3];
			transformPoint(ld.xform, from.x, from.y, from.z, worldFrom);
			transformPoint(ld.xform, to.x, to.y, to.z, worldTo);
			// dir must be `wi` - the direction FROM ANY SHADING POINT TOWARD
			// THE LIGHT (see PunctualLight::dir's own comment) - i.e. the
			// OPPOSITE of the direction sunlight travels, which runs from
			// `from` toward `to`. pbrt-v4 DistantLight::Create builds this
			// same vector (w = Normalize(from - to)) for exactly this reason
			// - its own SampleLi later returns renderFromLight(local +Z),
			// and its local +Z axis is constructed to already equal that
			// (from - to) direction, so this is the same result reached
			// without needing pbrt's intermediate rotation-to-Z matrix.
			for (int c = 0; c < 3; ++c) pl.dir[c] = worldFrom[c] - worldTo[c];
			normalizeOrDefault(pl.dir, 0, 1, 0);
			const pbrt_scene::Vec3 L = ld.params.getVec3("L", pbrt_scene::Vec3{1, 1, 1});
			pl.intensity[0] = L.x; pl.intensity[1] = L.y; pl.intensity[2] = L.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "goniometric") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Goniometric;
			transformPoint(ld.xform, 0.0, 0.0, 0.0, pl.pos);
			worldToLightRotation(ld.xform, pl.worldToLight);
			const pbrt_scene::Vec3 I = ld.params.getVec3("I", pbrt_scene::Vec3{1, 1, 1});
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			const std::string file = ld.params.getString("filename", "");
			if (!file.empty()) {
				pl.hadImageFilename = true;
				warn("light 'goniometric' names \"filename\" ('" + file +
					 "'), whose IES-style directional profile is not decoded; "
					 "the light is emitted as a plain isotropic point light instead "
					 "(matches this loader's own C5 showcase scene, which uses a "
					 "synthetic profile rather than a real image file too)");
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "projection") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Projection;
			transformPoint(ld.xform, 0.0, 0.0, 0.0, pl.pos);
			worldToLightRotation(ld.xform, pl.worldToLight);
			pl.scale = ld.params.getFloat("scale", 1.0);
			pl.fovDeg = ld.params.getFloat("fov", 90.0);
			const std::string file = ld.params.getString("filename", "");
			// pbrt-v4 requires "filename" (a projection light has no other
			// way to describe what it projects) - a scene that omits it is
			// itself malformed, and a scene that gives it names an image
			// this loader still cannot decode (see hadImageFilename's own
			// comment). Either way the light is emitted as a uniform white
			// beam of the requested shape/scale/aim, which is the same
			// synthetic-pattern approximation this loader's own C6 showcase
			// scene already uses; only the wording differs by which case it is.
			if (!file.empty()) {
				pl.hadImageFilename = true;
				warn("light 'projection' names \"filename\" ('" + file +
					 "'), whose projected image is not decoded; the light is "
					 "emitted as a uniform white beam instead (matches this "
					 "loader's own C6 showcase scene, which uses a synthetic "
					 "pattern rather than a real image file too)");
			} else {
				warn("light 'projection' has no \"filename\" (pbrt-v4 requires "
					 "one); emitted as a uniform white beam of the requested "
					 "shape/scale/aim instead of failing outright");
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		warn("light source '" + ld.type + "' is not supported and was dropped; "
			 "the scene will be darker than intended");
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
		// pbrt-v4 spells this "focaldistance" for perspective/orthographic but
		// "focusdistance" for realistic - reading both means a scene does not
		// silently keep the 1e6 sentinel just because it used the other
		// camera type's spelling.
		out.camera.focusDistance = scene.cameraParams.getFloat("focaldistance",
									   scene.cameraParams.getFloat("focusdistance",
										   out.camera.focusDistance));

		// Non-perspective cameras. lookfrom/lookat/up above already came from
		// the world-to-camera matrix, which every pbrt camera type shares -
		// only the type-specific parameters need reading here.
		out.camera.type = scene.cameraType;
		if (const pbrt_scene::Param *sw = scene.cameraParams.find("screenwindow")) {
			if (sw->numbers.size() >= 4) {
				out.camera.hasScreenWindow = true;
				out.camera.screenWindow[0] = sw->numbers[0];
				out.camera.screenWindow[1] = sw->numbers[1];
				out.camera.screenWindow[2] = sw->numbers[2];
				out.camera.screenWindow[3] = sw->numbers[3];
			} else {
				warn("a camera's \"screenwindow\" needs 4 numbers (xmin xmax ymin "
					 "ymax); ignored");
			}
		}
		if (out.camera.type == "spherical" || out.camera.type == "environment") {
			out.camera.sphericalMapping =
				scene.cameraParams.getString("mapping", out.camera.sphericalMapping);
		} else if (out.camera.type == "realistic") {
			out.camera.lensFile = scene.cameraParams.getString("lensfile", "");
			out.camera.apertureDiameterMM =
				scene.cameraParams.getFloat("aperturediameter", out.camera.apertureDiameterMM);
			out.camera.filmDiagonalMM =
				scene.cameraParams.getFloat("filmdiag", out.camera.filmDiagonalMM);
			if (out.camera.lensFile.empty()) {
				warn("a realistic camera has no \"lensfile\"; rendering with a "
					 "perspective camera instead");
				out.camera.type = "perspective";
			}
		}
	}

	// ---- decide what goes where before emitting anything ------------------
	//
	// Three destinations:
	//   * plain scene shapes            -> world space, as before
	//   * an object's non-emissive shapes -> its group, in OBJECT space
	//   * an object's EMISSIVE shapes   -> world space, once per instance
	//
	// That last one is the interesting decision. Instancing saves memory
	// because geometry is shared during traversal, but lights are not
	// traversed - they are enumerated into a flat list and sampled from a
	// distribution, so every copy of an emitter needs its own entry whatever
	// we do. Instancing them would save nothing and would force the MIS path
	// to recover which instance a ray struck. Baking them costs one entry per
	// copy, which for the handful of emitters an object has is nothing.
	//
	// pbrt declines this case outright ("Area lights not supported with object
	// instancing"). Baking is cheap enough that we do not have to.
	std::vector<detail::ShapeWork> work;

	for (const pbrt_scene::ShapeDecl &shape : scene.shapes)
		work.push_back({&shape, shape.xform, &out.triangles, &out.spheres, &out.bilinearPatches,
						&out.disks, &out.cylinders});

	// Sized up front so the pointers taken below stay valid as work is added.
	out.groups.resize(scene.objects.size());
	for (std::size_t g = 0; g < scene.objects.size(); ++g) {
		out.groups[g].name = scene.objects[g].name;
		for (const pbrt_scene::ShapeDecl &shape : scene.objects[g].shapes) {
			if (shape.areaLightIndex >= 0) continue;    // emissive: baked below
			// bilinearPatches left null: an instanced (non-emissive)
			// bilinearmesh has no scene in this loader's own corpus and no
			// InstanceGroup storage for it - the bilinearmesh branch below
			// treats a null output the same way it treats any other
			// unsupported case, rather than silently dropping the shape
			// with no explanation.
			work.push_back({&shape, shape.xform,
							&out.groups[g].triangles, &out.groups[g].spheres});
		}
	}

	for (const pbrt_scene::InstanceDecl &inst : scene.instances) {
		int group = -1;
		// Last match wins, matching pbrt's own name-rebinding semantics and
		// the parser's own warning when ObjectBegin redefines a name ("the
		// later definition replaces the earlier one" - pbrt_scene.h). A
		// redefined object still leaves BOTH definitions in scene.objects
		// (the parser doesn't erase the old one, only warns), so stopping at
		// the first match here silently placed the earlier, supposedly-
		// superseded definition instead - the exact opposite of what the
		// warning told the user would happen. No `break`: keep scanning so
		// the last (most recent) definition's index is what survives.
		for (std::size_t g = 0; g < scene.objects.size(); ++g)
			if (scene.objects[g].name == inst.name) group = static_cast<int>(g);
		if (group < 0) {
			warn("ObjectInstance names '" + inst.name +
				 "', which was never defined; the instance is skipped");
			continue;
		}

		Instance placed;
		placed.group = group;
		for (int i = 0; i < 16; ++i) placed.xform[i] = inst.xform.m[i];
		out.instances.push_back(placed);

		// Emissive shapes in the definition, baked into world space for this
		// placement so they can be sampled like any other light.
		for (const pbrt_scene::ShapeDecl &shape :
				 scene.objects[static_cast<std::size_t>(group)].shapes) {
			if (shape.areaLightIndex < 0) continue;
			work.push_back({&shape, detail::compose(inst.xform, shape.xform),
							&out.triangles, &out.spheres, &out.bilinearPatches,
							&out.disks, &out.cylinders});
		}
	}

	for (const detail::ShapeWork &w : work) {
		const pbrt_scene::ShapeDecl &shape = *w.shape;
		const pbrt_scene::Matrix4 &xform = w.xform;
		if (shape.type == "sphere") {
			const double r = shape.params.getFloat("radius", 1.0);
			Sphere s;
			transformPoint(xform, 0.0, 0.0, 0.0, s.center);

			double sc[3];
			axisScales(xform, sc);
			const double lo = std::fmin(sc[0], std::fmin(sc[1], sc[2]));
			const double hi = std::fmax(sc[0], std::fmax(sc[1], sc[2]));
			if (hi - lo > 1e-6 * std::fmax(1.0, hi)) {
				warn("a sphere carries a non-uniform scale and would be an "
					 "ellipsoid; emitted with its largest radius instead");
			}
			s.radius = r * hi;
			s.material = shape.materialIndex;
			s.areaLight = shape.areaLightIndex;
			s.medium = shape.insideMedium;
			w.spheres->push_back(s);
			continue;
		}

		if (shape.type == "disk" && w.disks) {
			Disk d;
			d.height = shape.params.getFloat("height", 0.0);
			d.radius = shape.params.getFloat("radius", 1.0);
			d.innerRadius = shape.params.getFloat("innerradius", 0.0);
			d.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
			for (int i = 0; i < 16; ++i) d.xform[i] = xform.m[i];
			d.material = shape.materialIndex;
			d.areaLight = shape.areaLightIndex;
			d.medium = shape.insideMedium;
			w.disks->push_back(d);
			continue;
		}

		if (shape.type == "cylinder" && w.cylinders) {
			Cylinder c;
			c.radius = shape.params.getFloat("radius", 1.0);
			c.zMin = shape.params.getFloat("zmin", -1.0);
			c.zMax = shape.params.getFloat("zmax", 1.0);
			c.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
			for (int i = 0; i < 16; ++i) c.xform[i] = xform.m[i];
			c.material = shape.materialIndex;
			c.areaLight = shape.areaLightIndex;
			c.medium = shape.insideMedium;
			w.cylinders->push_back(c);
			continue;
		}

		if (shape.type == "bilinearmesh") {
			// Single-patch form only (see BilinearPatch's own comment) - a
			// bare "point3 P" with exactly 4 points, no "integer indices".
			// Anything else (the multi-patch form, or an instanced/object-
			// space one - see the null-bilinearPatches comment above) falls
			// through to the generic "shape not supported" warning below,
			// same as every other shape kind this file does not build.
			const pbrt_scene::Param *P = shape.params.find("P");
			if (w.bilinearPatches && P && P->numbers.size() == 12) {
				BilinearPatch bp;
				for (int i = 0; i < 4; ++i)
					transformPoint(xform, P->numbers[i * 3 + 0], P->numbers[i * 3 + 1],
								   P->numbers[i * 3 + 2], bp.p[i]);
				bp.material = shape.materialIndex;
				bp.areaLight = shape.areaLightIndex;
				w.bilinearPatches->push_back(bp);
				continue;
			}
		}

		if (shape.type == "trianglemesh" || shape.type == "plymesh"
				|| shape.type == "loopsubdiv") {
			// "texture alpha" - an alpha-cutout mask authored per-Shape (see
			// Material::alphaTextureFilename's own comment for why it lands
			// on the Shape's resolved material rather than a new per-
			// triangle field). Only the texture-bound form is handled here;
			// a literal "float alpha" constant has no bundled scene using it
			// and is left for the generic "not supported" path below like
			// any other unhandled shape parameter.
			if (shape.materialIndex >= 0 &&
					static_cast<std::size_t>(shape.materialIndex) < out.materials.size()) {
				if (const pbrt_scene::Param *pa = shape.params.find("alpha");
						pa && pa->type == "texture" && !pa->strings.empty()) {
					const pbrt_scene::TextureDecl *tex = nullptr;
					for (const pbrt_scene::TextureDecl &t : scene.textures)
						if (t.name == pa->strings[0]) tex = &t;
					if (tex && tex->cls == "imagemap") {
						const std::string filename = tex->params.getString("filename", "");
						if (!filename.empty())
							out.materials[static_cast<std::size_t>(shape.materialIndex)]
								.alphaTextureFilename = filename;
					}
				}
			}

			std::vector<double> P;
			std::vector<int> indices;
			std::vector<double> N;    // per-vertex, object space; empty = none

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

				// The limit-surface normals are the entire reason to subdivide
				// rather than just tessellate. Discarding them renders the
				// refined mesh as visible facets - smoother facets than the
				// control cage, but facets.
				N.resize(refined.normals.size() * 3);
				for (std::size_t v = 0; v < refined.normals.size(); ++v) {
					N[v * 3]     = refined.normals[v][0];
					N[v * 3 + 1] = refined.normals[v][1];
					N[v * 3 + 2] = refined.normals[v][2];
				}
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
				// pbrt's own name for per-vertex shading normals.
				if (const pbrt_scene::Param *pn = shape.params.find("N")) N = pn->numbers;
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
				transformPoint(xform, P[v * 3], P[v * 3 + 1], P[v * 3 + 2],
							   &world[v * 3]);

			// A normal list that does not cover every vertex cannot be indexed
			// safely by the face indices, so it is refused wholesale rather
			// than used for part of the mesh - half-smooth shading is a worse
			// artefact than none, and much harder to recognise.
			std::vector<double> worldN;
			if (!N.empty() && N.size() / 3 >= vertexCount) {
				worldN.resize(vertexCount * 3);
				for (std::size_t v = 0; v < vertexCount; ++v)
					transformNormal(xform, N[v * 3], N[v * 3 + 1], N[v * 3 + 2],
									&worldN[v * 3]);
			} else if (!N.empty()) {
				warn("a mesh supplied fewer normals than vertices; "
					 "they are ignored and it will render flat-shaded");
			}

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
				if (!worldN.empty()) {
					for (int k = 0; k < 3; ++k) {
						t.n[0 + k] = worldN[static_cast<std::size_t>(a) * 3 + k];
						t.n[3 + k] = worldN[static_cast<std::size_t>(b) * 3 + k];
						t.n[6 + k] = worldN[static_cast<std::size_t>(c) * 3 + k];
					}
					t.hasNormals = true;
				}
				t.material = shape.materialIndex;
				t.areaLight = shape.areaLightIndex;
				w.triangles->push_back(t);
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
