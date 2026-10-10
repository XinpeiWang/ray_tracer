#pragma once
// pbrt_flat_scene_types.h -- the rest of what pbrt_flatten.h produces: lights, pixel filter, camera, instances and the FlatScene that holds it all. Plain data;
// part of pbrt_flatten.h, which includes it.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"
#include "conductor_data.h"
#include "glass_data.h"
#include "spectrum_types.h"   // BlackbodySpectrum - see resolveEmissionColor()'s own comment
#include "spectral_math.h"    // SpectrumToXYZ/InnerProduct/GetCIE_Y - ditto
#include "rgb_colorspace.h"   // RGBColorSpaceFromName() - ditto

namespace pbrt_flatten {

struct Emission {
	double L[3] = {1.0, 1.0, 1.0};
	double scale = 1.0;
	// Round 6 Phase 4: pbrt-v4's real AreaLightSource "diffuse" also accepts
	// a "filename" parameter for spatially-varying emission (an image
	// mapped onto the shape instead of a flat L) - when set, this wins over
	// L entirely (matches pbrt-v4's own DiffuseAreaLight, which ignores L
	// once an image is given). Empty (default) means "use L", this
	// struct's pre-existing behavior.
	std::string filename;
	// "twosided" - parsed nowhere before this (see docs/PBRT_SUPPORT.md and
	// named-material-and-texture.pbrt's own comments flagging this gap) -
	// every area light emitted only from its geometric front face
	// regardless of what the scene asked for. false (default) preserves
	// that exact pre-existing one-sided behavior.
	bool twoSided = false;

	// pbrt-v4's "power" (total emitted radiometric power Phi, watts) - an
	// alternative to specifying L directly. hasPower distinguishes "power
	// was given" from the real default of 0.0. Unlike the punctual lights'
	// own "power" (resolved immediately at parse time, into `scale`), an
	// area light's Phi = L * pi * area * (twoSided ? 2 : 1) needs the
	// attached shape's surface area - and AreaLightSource is declared
	// BEFORE the Shape it attaches to in pbrt syntax, so area isn't known
	// yet here. `power` is carried through as-is and resolved into `scale`
	// by flatten()'s own post-pass, once every shape (and hence every area
	// light's total attached area) has been built - see that pass's own
	// comment for the formula.
	bool hasPower = false;
	double power = 0.0;
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

	// pbrt-v4's windowed/portal infinite light ("point3 portal[4]"): the
	// environment map is only visible through this finite rectangular
	// window instead of the whole sphere - see PortalImageInfiniteLightData
	// (src/shared/portal_image_infinite_light.h) for the sampling math
	// this feeds. hasPortal distinguishes a real portal[4] param from the
	// default-zeroed array (all 12 numbers 0.0 is not a valid rectangle,
	// but an explicit boolean is clearer than relying on that never
	// colliding with a real scene). Already transformed into world/render
	// space via `xform` at parse time (transformPoint(), matching every
	// other point-valued light param in this file) - the class itself
	// applies no further transform (see its own constructor comment).
	// Ordering matches pbrt-v4/this port's own convention: portal[0] is
	// the origin corner, portal[1]=portal[0]+right, portal[3]=portal[0]+up,
	// portal[2] the diagonal opposite corner.
	bool hasPortal = false;
	double portal[12] = {0.0};
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
	// upper-left 3x3 (see flatten_detail::worldToLightRotation()) - both kinds have
	// no "from"/"to" of their own in pbrt-v4, so a scene aims either one
	// purely by rotating the CTM before the LightSource directive (e.g.
	// `Rotate` then `LightSource "projection" ...`). Identity when the CTM
	// is a pure translation, which covers every scene this loader's own
	// corpus (C5/C6) uses either kind in.
	double worldToLight[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

	// Projection only, degrees, pbrt's "fov".
	double fovDeg = 90.0;

	// Both image-based kinds (Goniometric/Projection) name a "filename" in
	// real pbrt-v4 scenes - an IES-derived equal-area profile image for
	// goniometric (matching pbrt-v4's own convention: it reads this through
	// its generic Image::Read(), not a raw .ies text parser - see
	// docs/PBRT_SUPPORT.md), the projected slide image for projection. As
	// given by the scene, NOT yet resolved to an existing file - mirrors
	// Material::textureFilename/Emission::filename's own "resolved by
	// pbrt_load.h post-flatten, decoded by pbrt_cpu_builder.h/
	// pbrt_gpu_builder.h" convention (this header stays filesystem-free by
	// design - see the file comment), rather than InfiniteLight's own
	// decode-into-pixels-here-in-this-struct convention, since both builders
	// already had a direct-from-resolved-path image decode utility to reuse
	// (mipmap_texture/getOrBuildPbrtImageTexture) that InfiniteLight's
	// sky_light raw-buffer constructor didn't.
	std::string filename;

	// True iff the scene named a "filename" at all, independent of whether
	// it was later found/decoded - lets a caller (or a test) distinguish "no
	// file was named" from "a file was named" without inspecting `filename`
	// itself, same shape as Material::alphaTextureFilename's own empty-means-
	// absent convention would give for free if this were the only signal
	// needed, but pbrt_load.h clears `filename` back to empty on a resolve
	// failure (see its own comment) so this bool is the only way to still
	// tell "never named" apart from "named but not found" after that point.
	bool hadImageFilename = false;
};

// pbrt-v4's PixelFilter, resolved from Scene::filterType/filterParams into
// the exact numeric params each of this project's own filter.h classes
// needs. `kind` is left as pbrt's own name string (not an enum) so this
// header doesn't need to depend on filter.h/know its class names - the
// consumer (camera.h) owns the name->class mapping, matching how
// Scene::samplerType is handled elsewhere in this loader. Unrecognized or
// absent kind means "gaussian", pbrt-v4's real default (confirmed against
// pbrt-v4/src/pbrt/scene.cpp - NOT "box"/"triangle"/"mitchell", easy
// defaults to assume wrong). Only B/C/sigma/tau are threaded through, not
// a radius: this codebase's own filter.h classes (and camera.h's own
// per-pixel-only sampling loop, which never gathers samples from a
// neighboring pixel) are built around a fixed 0.5-pixel footprint - see
// camera.h's own comment on why pbrt-v4's real filter radii (e.g.
// Mitchell's default 2) aren't supported. This still applies the actual
// requested filter's SHAPE (its falloff curve), just clamped to the one
// footprint every filter here already assumes.
struct PixelFilter {
	std::string kind = "gaussian";
	double B = 1.0 / 3.0, C = 1.0 / 3.0;   // mitchell
	double sigma = 0.5;                     // gaussian
	double tau = 3.0;                       // sinc (LanczosSinc)
	// "float xradius"/"float yradius" - pbrt-v4's real per-kind default
	// (NOT a single shared default - see flatten()'s own resolution site):
	// box=0.5, gaussian=1.5, mitchell=2.0, sinc=4.0, triangle=2.0. This
	// project's filter classes (src/shared/filter.h) only carry one scalar
	// radius for both axes (no existing asymmetric-radius support to plug
	// an independent yradius into), so xradius/yradius are read as ONE
	// value - whichever was actually given (matching real scenes, which
	// essentially always set both to the same number).
	double radius = 1.5;
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

	// pbrt-v4's real camera-motion-blur idiom: ActiveTransform "StartTime"/
	// "EndTime" around two LookAt/Transform blocks before Camera/WorldBegin
	// (see pbrt_scene::Scene::cameraIsAnimated()'s own comment) - true only
	// when the scene actually authored two DIFFERENT keyframes, not merely
	// declared ActiveTransform. lookfrom1/lookat1 are the end-time keyframe
	// (extracted from Scene::worldToCameraEnd the same way lookfrom/lookat
	// above come from worldToCamera); there is deliberately no separate
	// `up1` - src/TheRestOfYourLife/camera.h's own CameraConfig has no such
	// field either (its own comment: "roll during the exposure isn't
	// supported by this simplified two-keyframe setup"), so the single
	// `up` above is reused for both keyframes, matching that existing,
	// already-tested CPU design exactly rather than inventing a richer one
	// this round doesn't need.
	bool isAnimated = false;
	double lookfrom1[3] = {0, 0, -1};
	double lookat1[3] = {0, 0, -2};

	// pbrt-v4's real Camera "float shutteropen"/"float shutterclose"
	// parameters (defaults match pbrt-v4's own: 0.0/1.0). CPU's own
	// CameraConfig (camera.h) uses this SAME pair as BOTH the shutter's
	// random-sampling window AND the two keyframes' own AnimatedTransform
	// start/end times (its own build_cam_to_world() call passes
	// shutter_open/shutter_close directly as those times) - there is no
	// independent notion of "keyframe time" distinct from "shutter window"
	// in this codebase's existing camera implementation, so
	// TransformTimes' own distinct value (when it differs from
	// shutteropen/shutterclose) has no effect - see the warning this round
	// adds in flatten() for that case.
	double shutterOpen = 0.0;
	double shutterClose = 1.0;
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

// The primitives one top-level `Shape` of the scene file produced: index ranges [begin, end) into FlatScene::triangles / spheres / disks / cylinders. Written by
// flattenShapes() in file order, so index i is the i-th Shape that made any of those four kinds; the Live Preview's object picking and moving use it. Shapes that were
// turned into other kinds (curves, cones, patches, instances) have no range, and primitives the loader adds later (a tessellation) lie beyond every range.
struct ShapeRange {
	std::string type;                  // the pbrt shape name: "sphere", "trianglemesh", "plymesh", ...
	int material = -1;                 // index into FlatScene::materials of the first primitive it made
	int group = -1;                    // ShapeDecl::group: the AttributeBegin/End block the Shape sat in, -1 outside any
	std::size_t triBegin = 0, triEnd = 0, sphereBegin = 0, sphereEnd = 0, diskBegin = 0, diskEnd = 0, cylinderBegin = 0, cylinderEnd = 0;
};

struct FlatScene {
	std::vector<ShapeRange> shapeRanges;
	std::vector<Triangle> triangles;
	// See AnimatedTriangleMesh's own comment - real object motion blur for
	// trianglemesh/plymesh/loopsubdiv, CPU only, populated only when a
	// shape's xformEnd genuinely differs from xform.
	std::vector<AnimatedTriangleMesh> animatedTriangleMeshes;
	std::vector<Sphere> spheres;
	std::vector<Disk> disks;
	std::vector<Cylinder> cylinders;
	std::vector<Cone> cones;
	std::vector<Paraboloid> paraboloids;
	std::vector<BilinearPatch> bilinearPatches;
	// See AnimatedBilinearPatch's own comment - real object motion blur for
	// bilinearmesh, CPU only, populated only when a shape's xformEnd
	// genuinely differs from xform (and it's not emissive).
	std::vector<AnimatedBilinearPatch> animatedBilinearPatches;
	std::vector<Curve> curves;
	// See AnimatedCurve's own comment - real object motion blur for curve,
	// CPU only, populated only when a shape's xformEnd genuinely differs
	// from xform (and it's not emissive or ribbon-type).
	std::vector<AnimatedCurve> animatedCurves;
	std::vector<Material> materials;    // parallel to Scene::materials
	std::vector<Emission> areaLights;   // parallel to Scene::areaLights
	std::vector<Medium> media;          // parallel to Scene::media
	// pbrt-v4's own "camera medium" - see Scene::cameraMediumIndex's own
	// comment (pbrt_scene.h) for what this requests. -1 (the default)
	// means none. Resolved by flatten()'s own post-pass (after every shape
	// is built, so it can check for - and warn about - a scene that ALSO
	// uses real per-shape media, a combination this loader doesn't model
	// yet): homogeneous only (media[cameraMediumIndex].type must be
	// "homogeneous" - matching this loader's own "close the homogeneous
	// case first" precedent for other pbrt-v4 medium features). Consumed on
	// CPU (src/TheRestOfYourLife/camera.h's ray_color() - see camera::
	// camera_medium's own comment) and on GPU-recursive
	// (gpu/optix/scene_builder.cpp resolves it into out_camera_extra,
	// consumed by optix_raygen.h) - ray_color_spectral()/BDPT/MLT/SPPM on
	// CPU and the GPU-wavefront backend still don't consume this field.
	int cameraMediumIndex = -1;
	InfiniteLight infiniteLight;        // present=false if the scene has none
	std::vector<PunctualLight> punctualLights;   // LightSource point/spot/distant/goniometric/projection

	// Instanced geometry. `groups` hold object-space shapes; `instances` place
	// them. A backend that ignores these renders a scene missing everything
	// that was instanced, so both builders must handle them.
	std::vector<InstanceGroup> groups;
	std::vector<Instance> instances;

	Camera camera;
	PixelFilter filter;
	// Integrator "bool regularize" - see pbrt_scene::Scene::regularize's
	// own comment. Applied unconditionally from the scene's own
	// declaration (matching PixelFilter's precedent, not maxDepth/
	// samplerType's "advisory only, CLI wins" one) since this is a
	// genuine scene-authored behavior toggle, not a perf knob a user
	// would want to casually override between a preview and a final
	// render.
	bool regularize = false;
	// Accelerator "bvh"/"kdtree" - see pbrt_scene::Scene::acceleratorType's
	// own comment. Applied unconditionally like PixelFilter/regularize
	// above (not CLI-overridable): which acceleration structure/build
	// strategy runs isn't something a user would want to casually override
	// per-render, and all of them produce the same converged image, so
	// there's no correctness reason to gate any of this behind a flag.
	// "bvh" (the default, both here and in real pbrt-v4) keeps using
	// bvh_node/bvh_aggregate_hittable.h as resolved by
	// acceleratorSplitMethod below; "kdtree" routes through
	// kd_tree_hittable.h's KdTree<double,...> instead - see flatten()'s own
	// motion-blur fallback (both this and a non-"sah" splitmethod share the
	// identical "no ray-time channel" limitation, so both fall back to
	// "bvh"/"sah" together on a scene with object motion blur).
	std::string acceleratorType = "bvh";
	// "string splitmethod"/"integer maxnodeprims" - only consulted when
	// acceleratorType is "bvh" (kdtree has its own, separate param set
	// below). "sah" keeps using bvh_node - the CPU builder's own pre-
	// existing, already-real SAH BVH - unchanged; only an explicit
	// "middle"/"equal"/"hlbvh" routes through BvhTree<double,...> instead
	// (bvh_aggregate_hittable.h).
	std::string acceleratorSplitMethod = "sah";
	// NOTE: BvhTree<T,Prim>::build() (src/shared/bvh_aggregate.h, untouched
	// by this loader) only actually consults max_prims_in_node for "sah" and
	// "hlbvh" - "middle"/"equal"'s own build_recursive() branches split down
	// to exactly 1 primitive per leaf every time regardless of this value
	// (no max_prims_ check in either branch). Real, pre-existing behavior of
	// that already-tested class, not something this loader's wiring
	// introduces or could easily change - passed through honestly rather
	// than silently clamped/ignored at this layer.
	int acceleratorMaxNodePrims = 4;
	// Accelerator "kdtree"'s own params - see
	// pbrt_scene::Scene::acceleratorKdParams's own comment for the real
	// pbrt-v4 defaults this mirrors. Only consulted when acceleratorType is
	// "kdtree".
	KdTreeAccelParams acceleratorKdParams;
	// Film "float[4] cropwindow" / "integer[4] pixelbounds", resolved to a
	// single NDC-fraction rectangle [cropX0,cropX1) x [cropY0,cropY1) in
	// [0,1] - see flatten()'s own computation for the exact rule. Kept as
	// fractions rather than resolved to pixel indices here because
	// xResolution/yResolution are only advisory in this codebase (like
	// maxDepth/samplerType - a CLI width/height arg wins, see
	// scene_registry.h): a fraction stays correct however the actual
	// render resolution ends up differing from the scene's own declared
	// one, where absolute pixel indices resolved against the WRONG
	// resolution would not. Applied unconditionally like PixelFilter/
	// regularize above (not CLI-overridable, matching that same
	// "genuine scene-authored behavior" precedent); no directive at all
	// resolves to the full frame {0, 1, 0, 1}.
	double cropX0 = 0.0, cropX1 = 1.0, cropY0 = 0.0, cropY1 = 1.0;
	// Film "float maxcomponentvalue" - pbrt-v4's own real default
	// (effectively unbounded), a straight pass-through of pbrt_scene::
	// Scene::maxComponentValue - see that field's own comment. Applied
	// unconditionally, same "genuine scene-authored behavior" shape as
	// cropX0 above, not maxDepth/samplerType's CLI-overridable one.
	double maxComponentValue = 1e9;
	std::vector<pbrt_scene::Warning> warnings;
	// Mesh files (plymesh / .obj) a Shape named but the resolver could not read.
	// Those shapes are skipped with a warning; this list is what lets the loader
	// tell "a few meshes are missing" from "the scene is empty because every
	// mesh is missing" (pbrt_load::loadFile turns the latter into an error).
	std::vector<std::string> missingFiles;

	bool empty() const {
		return triangles.empty() && spheres.empty() && instances.empty();
	}

	// True if ANY shape survived flattening, of any kind. empty() above only
	// looks at triangles/spheres/instances, and an instance of a group whose
	// shapes were all skipped still counts there.
	bool hasAnyGeometry() const {
		if (!triangles.empty() || !animatedTriangleMeshes.empty() || !spheres.empty() || !disks.empty() ||
			!cylinders.empty() || !cones.empty() || !paraboloids.empty() || !bilinearPatches.empty() ||
			!animatedBilinearPatches.empty() || !curves.empty() || !animatedCurves.empty())
			return true;
		for (const Instance &inst : instances) {
			if (inst.group < 0 || inst.group >= static_cast<int>(groups.size())) continue;
			const InstanceGroup &g = groups[inst.group];
			if (!g.triangles.empty() || !g.spheres.empty()) return true;
		}
		return false;
	}
};

// Supplies a PLY mesh's positions, indices and (optionally) per-vertex UV for
// `Shape "plymesh"`. Same callback shape, and for the same reason, as
// pbrt_scene::FileResolver: keeps this a pure function and lets the caller
// decide how a path resolves. Returning false means the mesh could not be
// read. `uvs` is filled 2-per-vertex when the PLY file carries "u"/"v" (or
// "s"/"t") vertex properties (see ply_mesh.h's own vertexSlotFor()), left
// empty otherwise - mirroring how a `Shape "trianglemesh"` with no `"uv"`
// parameter leaves this loader's own UV vector empty.
// `normals` is filled 3-per-vertex when the file carries per-vertex normals (PLY
// "nx"/"ny"/"nz" properties, or OBJ `vn` records referenced by its faces) and
// left empty otherwise, which keeps flat per-face shading - same convention as a
// `Shape "trianglemesh"` with no "N" parameter.
using MeshResolver = std::function<bool(const std::string &path,
										std::vector<float> &positions,
										std::vector<int> &indices,
										std::vector<float> &uvs,
										std::vector<float> &normals)>;


}  // namespace pbrt_flatten
