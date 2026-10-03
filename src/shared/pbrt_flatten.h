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
#include <string>
#include <vector>

#include "pbrt_scene.h"
#include "loop_subdivide.h"
#include "conductor_data.h"
#include "glass_data.h"
#include "spectrum_types.h"   // BlackbodySpectrum - see resolveEmissionColor()'s own comment
#include "spectral_math.h"    // SpectrumToXYZ/InnerProduct/GetCIE_Y - ditto
#include "rgb_colorspace.h"   // RGBColorSpaceFromName() - ditto

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
	// Per-vertex texture coordinates, (u,v) each - pbrt's own "point2 uv"
	// trianglemesh parameter, only when actually given (see flatten()'s own
	// trianglemesh branch for why pbrt-v4's real "no uv given" default is
	// deliberately NOT synthesized here). loopsubdiv/plymesh don't thread
	// UV through this loader at all yet either (a separate, smaller gap,
	// tracked in docs/PBRT_SUPPORT.md) - hasUVs is false for both cases,
	// and both builders' own barycentric fallback (matching CPU triangle.h's
	// pre-existing `rec.u=b1,rec.v=b2`) covers all of them uniformly.
	double uv[6] = {0, 0, 0, 0, 0, 0};
	bool hasUVs = false;
	int material = -1;        // index into Scene::materials, -1 = pbrt default
	int areaLight = -1;       // index into Scene::areaLights, -1 = not emissive
	// True only for a StartTime-pose duplicate of a mesh that's ALSO in
	// FlatScene::animatedTriangleMeshes (real CPU motion blur - see that
	// struct's own comment) - gpu/optix/pbrt_gpu_builder.h has no concept
	// of that separate list, so this duplicate exists purely to keep GPU
	// rendering the shape at all (static at its StartTime pose, same
	// "GPU (both backends) renders static at the StartTime position,
	// warned" convention Disk/Cylinder's own motion blur already
	// established), rather than the shape silently vanishing from GPU
	// renders because its geometry moved to a list GPU never reads.
	// pbrt_cpu_builder.h's own emitGeometry() skips any triangle with this
	// set - CPU gets its real motion blur from the animatedTriangleMeshes
	// entry instead, so building an extra static `triangle` hittable here
	// too would double-render it.
	bool gpuOnlyStaticFallback = false;
};

// A trianglemesh/plymesh/loopsubdiv shape under real object motion blur
// (pbrt-v4 ActiveTransform "StartTime"/"EndTime" around one of these Shape
// types) - CPU only. Unlike Triangle above (baked once to WORLD space at
// flatten() time, the overwhelmingly common static case), every triangle
// here is OBJECT space (v[9]/n[9] straight from the shape's own "P"/"N",
// no CTM applied) - real per-ray-time motion needs the geometry to stay in
// its own frame so a wrapper (src/TheRestOfYourLife/
// animated_transform_instance.h) can carry the ray into it at each ray's
// own resolved time (mirrors Disk/Cylinder's own MotionState-based
// technique, motion_state.h - see that file's own comment for why: a mesh
// bakes to world-space vertices at load time in the static case, so real
// per-ray motion needs the SAME "keep geometry, move the ray" trick
// transform_instance.h already uses for pbrt's ObjectInstance, just with a
// second (EndTime) transform interpolated per ray instead of one static
// one).
//
// Only ever populated when a shape's own xformEnd genuinely differs from
// xform (real inequality, not ActiveTransform directive presence - same
// convention as every other animated-shape field in this file); the
// overwhelming common static case keeps using the plain Triangle list
// above completely unchanged, at zero overhead. An EMISSIVE animated mesh
// is deliberately excluded from this path (falls back to a static,
// StartTime-only bake instead, warned) - NEE sampling needs enumerable
// world-space geometry (see the comment on the ObjectInstance-definition
// case just above InstanceGroup for the same rule already established
// there), which an object-space-transform-wrapped hittable can't give it;
// see pbrt_cpu_builder.h's own animated-mesh dispatch for the exact
// warning. curve/bilinearmesh are structurally identical candidates for
// this same treatment, not done this round.
struct AnimatedTriangleMesh {
	std::vector<Triangle> triangles;  // OBJECT space (not world space - see above)
	double xform[16]    = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	double xformEnd[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct Sphere {
	double center[3] = {0, 0, 0};
	// Object-motion-blur end-time centre (pbrt-v4 ActiveTransform "EndTime"
	// on this Shape - see pbrt_scene.h's ShapeDecl::xformEnd comment).
	// Defaults to `center` (real-inequality checked below, at the sole
	// call site that bakes this field, so every shape that never sat
	// inside an ActiveTransform pair renders exactly as before this field
	// existed - a stationary sphere). Only baked for a full (non-clipped)
	// sphere: rotation-invariant like `center`/`radius` above, so a plain
	// translated-endpoint lerp (matching src/TheRestOfYourLife/sphere.h's
	// existing two-centre moving constructor and GPU's own SphereData::
	// center1, both already real, working infrastructure - just never
	// wired to a pbrt-parsed shape until this field existed) is exact, not
	// an approximation. A clipped sphere's real end-time xform would need
	// a materially bigger "carry two full object-to-world transforms,
	// slerp/lerp the whole matrix" feature (mirroring GpuMediumShapeKind::
	// ClippedSphere's own unbaked-xform path) - out of scope for this
	// round, so a clipped sphere's `clipped` flag stays authoritative and
	// this field is left at `center` (no motion) whenever clipped is true.
	double center1[3] = {0, 0, 0};
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
	// Partial-sphere clipping (pbrt-v4's real "float zmin"/"float zmax"/
	// "float phimax" on Shape "sphere" - caps/wedges/hemispheres, e.g. a
	// domed skylight cutout). Unlike a full sphere (rotation-invariant, so
	// baked straight to center/radius above), a CLIPPED sphere is
	// orientation-dependent, so CPU renders it with the real object-to-world
	// transform instead - radiusLocal/zMin/zMax/xform below are in OBJECT
	// space (pbrt-v4's own convention, matching Disk/Cylinder's own xform
	// fields just below) and are only populated/read when clipped is true.
	// GPU also renders a clipped sphere with its real transform now (not
	// baked to center/radius) - gpu/optix/pbrt_gpu_builder.h's ClippedSphere
	// branch carries radiusLocal/zMin/zMax/xform (below) through to
	// SphereData::radiusLocal/zMin/zMax/o2w/w2o, and both GPU backends
	// intersect it in object space via a custom software program, following
	// Disk/Cylinder's own precedent (GPU sphere intersection was already a
	// custom program, not OptiX's hardware primitive, despite this comment's
	// own earlier claim otherwise). center/radius above stay populated for
	// every sphere regardless of clipping - for a clipped one they're used
	// only by the pre-existing full-sphere-cone NEE approximation (see
	// GpuMediumShapeKind::ClippedSphere's comment, optix_types.h), never for
	// hit-testing.
	bool clipped = false;
	double radiusLocal = 1.0;
	double zMin = -1.0, zMax = 1.0;   // object-space, matches radiusLocal's units
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	// True only when clipped AND medium>=0: an open shell (a hole where
	// zmin/zmax/phimax cut it away) can't bound a participating medium
	// correctly - CPU's clipped-sphere hittable (sphere_clipped_hittable.h)
	// can't handle constant_medium's watertight-boundary requirement, and
	// (now that GPU also renders a clipped sphere as a real open shell
	// rather than the old always-full-closed-sphere approximation) neither
	// can GPU - see gpu/optix/pbrt_gpu_builder.h's ClippedSphere branch,
	// which resolves this sphere's material via materialIndex(), never
	// mediumMaterialIndex(), for exactly this reason. `medium` above is
	// still left fully populated regardless (an unclipped sphere has no
	// problem bounding a medium on either backend) - this flag exists
	// purely so pbrt_cpu_builder.h can skip wrapping ITS OWN hittable in
	// constant_medium without disturbing the shared `medium` field GPU
	// reads independently.
	bool cpuMediumUnsupported = false;
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
	// Object-motion-blur end-time transform (pbrt-v4 ActiveTransform
	// "EndTime" around this Shape) - see Sphere::center1's own comment for
	// the general idiom. Unlike Sphere (which bakes a single lerp-able
	// world-space point), a Disk keeps its FULL transform unbaked already
	// (xform above), so the natural motion representation is a SECOND full
	// transform rather than a second point: both CPU (disk_cylinder_
	// hittable.h) and GPU resolve the real object<->world affine at each
	// ray's own time via AnimatedTransform's TRS decomposition (src/shared/
	// animated_transform.h - translation lerp, rotation slerp, scale lerp),
	// not a naive per-element matrix lerp, which would visibly shear a
	// rotating disk. Defaults to `xform` (no motion) whenever this shape
	// never sat inside an ActiveTransform pair - real-inequality checked at
	// this field's sole bake site (pbrt_flatten.h's disk-building branch),
	// same `Matrix4::differsFrom()` convention as Sphere::center1.
	double xformEnd[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	int areaLight = -1;
	int medium = -1;
};

struct Cylinder {
	double radius = 1.0, zMin = -1.0, zMax = 1.0;
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	// See Disk::xformEnd's own comment - identical idiom.
	double xformEnd[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	int areaLight = -1;
	int medium = -1;
};

// Shape "cone" / "paraboloid" - 2 of pbrt-v4's 3 remaining quadric Shape
// kinds this loader had no counterpart for at all (previously fell to the
// generic "shape not supported" warning, nothing rendered in their place -
// see flatten()'s own trailing warning for that fallback). Same unbaked-
// object-space-plus-CTM technique as Disk/Cylinder above, for the identical
// reason (not rotation-invariant). Now also carry a real AreaLightSource
// (areaLight below - registered as a real, NEE-samplable light, matching
// Disk/Cylinder's own identical field exactly) and a real MediumInterface
// (medium below - a real participating-medium boundary, same
// addMediumIfPresent() path Sphere/Disk/Cylinder already use), CPU only -
// GPU support is a separate, not-yet-attempted follow-up (both backends
// warn and drop this geometry entirely already, same as every other CPU-
// only shape kind in this codebase; that pre-existing "shape unsupported on
// GPU" warning covers the area-light/medium case too, since there's no
// shape at all on GPU to attach either to). Shape "hyperboloid" (the 3rd
// quadric) is deliberately NOT covered here - pbrt-v4's real hyperboloid is
// a TWISTED ruled surface (ah/ch quadric coefficients derived from two
// arbitrary 3D points, not a plain surface of revolution the way cone/
// paraboloid/cylinder are), meaningfully harder to get right than these two
// and the rarest of the three in practice - left as a future addition
// rather than risking a subtly-wrong implementation this round.
struct Cone {
	double radius = 1.0, height = 1.0;
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	// AreaLightSource/MediumInterface - see this shape's own top comment
	// (just above Cone) for the real, non-geometry-only support these two
	// fields now carry (matching Disk/Cylinder's own identical fields
	// exactly), CPU only.
	int areaLight = -1;
	int medium = -1;
};

struct Paraboloid {
	double radius = 1.0, zMin = 0.0, zMax = 1.0;
	double phiMaxDeg = 360.0;
	double xform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int material = -1;
	// See Cone::areaLight/medium's own comment - identical.
	int areaLight = -1;
	int medium = -1;
};

// "Is this RGB triple effectively nonzero" - the same per-channel epsilon
// threshold used at 4 sites across this file and gpu/optix/scene_builder.cpp
// (2 pre-existing sigma_a-dropped warnings for cloud/uniformgrid, Medium::
// Le's own "dropped for non-homogeneous" warning below, and the GPU-side
// "scene requests Le but GPU can't honor it" warning) - named once here,
// at namespace (not flatten_detail) scope, so scene_builder.cpp can call it
// too without reaching into flatten()'s own internals.
inline bool isNonzeroRGB(const double v[3]) {
	constexpr double kEps = 1e-9;
	return v[0] > kEps || v[1] > kEps || v[2] > kEps;
}

// Perceptual (Rec. 709) luminance of an RGB triple - the same weights
// power_light_sampler.h documents as its own stand-in for a full spectral
// SpectrumToPhotometric conversion. Real pbrt-v4 normalizes a light's
// "power" parameter by exactly this quantity (computed from the light's
// own colour/spectrum) before scaling to the requested total power - see
// PointLight::Create/SpotLight::Create/DiffuseAreaLight::Create - so a
// power-driven light's brightness doesn't also depend on how bright its L/I
// happened to be typed in as. Falls back to 1.0 (unweighted) for a literal
// black input, which has no ratio worth preserving.
inline double relativeLuminance(const double c[3]) {
	const double lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
	return (lum > 1e-9) ? lum : 1.0;
}

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

	// MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4) - a
	// self-emitting medium (fire/plasma/glowing fog). Lescale is baked into
	// Le at flatten time - UNLIKE textureScale/Emission::scale elsewhere in
	// this file, which stay separate fields because they have multiple
	// independent downstream consumers with different needs (CPU's
	// scaled_texture wrapper, GPU's own emissionScale field). Le has
	// exactly one consumer (pbrt_cpu_builder.h's addMediumIfPresent, which
	// immediately multiplies it by sigma_a/sigma_t and discards the
	// unscaled value) with nothing GPU-side reading it yet, so there is
	// currently no real downstream need for the unscaled value - baking it
	// in here is a genuine simplification for THIS field's actual usage,
	// not an application of the scale-texture precedent. Revisit (keep
	// Lescale as a separate field) if a future consumer - e.g. a GPU
	// implementation, see scene_builder.cpp's own warning comment - turns
	// out to need the unscaled Le. "homogeneous" ONLY this round (see the
	// flatten() loop's own warning for cloud/rgbgrid/uniformgrid) - those
	// types' own real
	// pbrt-v4 emission is a genuinely separate per-voxel feature (their own
	// "Le"/"LeScale" GRIDS, not a flat colour), a materially bigger lift
	// deferred to a later round, matching this loader's own "close the
	// homogeneous case first" precedent for other medium params. CPU-only:
	// see hg_phase_material::emitted() (constant_medium.h) for how this
	// flows into the render; GPU (both backends) doesn't read this field at
	// all yet - scene_builder.cpp warns once if it's nonzero.
	double Le[3] = {0.0, 0.0, 0.0};

	// "homogeneous" (default, uses sigma_a/sigma_s/g above only), "cloud",
	// "rgbgrid", "uniformgrid", or "nanovdb" - a real, genuinely different
	// medium type each. cloud/rgbgrid/uniformgrid have a real implementation
	// on both backends (src/shared/cloud_medium.h, src/shared/
	// rgb_grid_medium.h, src/shared/sampled_grid.h's GridMediumData);
	// nanovdb is CPU-only (see nanovdbFilename's own comment) - GPU falls
	// back to homogeneous with a warning, same as any other type it doesn't
	// recognize.
	std::string type = "homogeneous";

	// World-space AABB the medium actually occupies - the medium-space box
	// [p0,p1] below, transformed by the CTM captured at MakeNamedMedium
	// declaration time (see pbrt_scene::MediumDecl::xform's own comment).
	// Used for GPU's trigger-sphere-only dispatch (cloud/rgbgrid/uniformgrid
	// only - see pbrt_gpu_builder.h's own comment) and CPU's hittable
	// bounding_box().
	double worldMin[3] = {0.0, 0.0, 0.0};
	double worldMax[3] = {1.0, 1.0, 1.0};

	// World -> medium-space affine transform (row-major 3x3 + translate) -
	// the inverse of the CTM captured at declaration time. cloud/rgbgrid/
	// uniformgrid only; matches CloudMedium<T>::mat/translate and
	// rgb_grid_medium_hittable's own world_to_medium_mat/translate
	// convention exactly (no rearrangement needed at the call site).
	double toMediumMat[9] = {1,0,0, 0,1,0, 0,0,1};
	double toMediumTranslate[3] = {0.0, 0.0, 0.0};

	// pbrt-v4's own "point3 p0"/"p1" - the medium's bounds IN medium space
	// (default unit cube, matching pbrt-v4's real default for cloud,
	// rgbgrid, AND uniformgrid alike). cloud/rgbgrid/uniformgrid only.
	double p0[3] = {0.0, 0.0, 0.0};
	double p1[3] = {1.0, 1.0, 1.0};

	// "cloud" only - pbrt-v4's real per-param defaults (media.cpp's
	// CloudMedium::Create): density 1, wispiness 1, frequency 5. NOT the
	// same defaults this codebase's own hand-built E2 showcase scene
	// happens to use (frequency 4) - that scene picked its own look,
	// this is pbrt-v4's actual spec default for a scene that omits the
	// param entirely.
	double density = 1.0;
	double wispiness = 1.0;
	double frequency = 5.0;

	// "rgbgrid"/"uniformgrid" only, shared between them - grid resolution.
	int nx = 1, ny = 1, nz = 1;

	// "rgbgrid" only - de-interleaved flat per-channel voxel arrays, each
	// either empty (that channel group absent, matching pbrt-v4's own
	// "channel omitted -> defaults to 1 for sigma_a/s, no emission"
	// convention - see RGBGridMediumData<T>::build()'s own comment) or
	// length nx*ny*nz.
	std::vector<double> sigma_a_r, sigma_a_g, sigma_a_b;
	std::vector<double> sigma_s_r, sigma_s_g, sigma_s_b;

	// "rgbgrid" only - pbrt-v4's own real per-voxel emission: a flat "Le"
	// array (one RGB triple per voxel, same shape/convention as sigma_a/
	// sigma_s above) plus a scalar "Lescale" multiplier applied at SAMPLE
	// time by RGBGridMediumData<T>::sample_point(), not baked in here (see
	// that function's own `Le_out[c] = Le_scale * le;` - this struct passes
	// Le_scale through unbaked to match). Deliberately a SEPARATE field
	// from this struct's own `Le[3]`/homogeneous-only scalar emission
	// above - the top-level "Le"/"Lescale" params this file's own flatten()
	// loop reads are only ever a flat colour for `homogeneous`; for
	// `rgbgrid` those same two param NAMES mean this per-voxel array
	// instead (pbrt-v4's own directive reuse), so reading them through the
	// generic getVec3()/resolveEmissionColor() path used for homogeneous
	// would silently misinterpret a per-voxel array as a single flat
	// colour (getVec3 only requires numbers.size()>=3, not ==3, so it
	// would happily read voxel 0's own RGB and ignore the rest) - flatten()
	// skips that generic read entirely for rgbgrid and parses "Le" as an
	// array here instead, the same way it already does for "sigma_a"/
	// "sigma_s". Empty (all three channels) means no emission, matching
	// RGBGridMediumData::build()'s own "empty vector = channel absent"
	// convention. `cloud`/`uniformgrid` still don't support any emission
	// at all (real pbrt-v4 support exists for `rgbgrid` specifically here;
	// the other two would need their own, differently-shaped per-voxel Le
	// grid a future round could add the identical way).
	std::vector<double> Le_r, Le_g, Le_b;
	double Le_scale = 0.0;

	// "uniformgrid" only - pbrt-v4's own "float density" (a REQUIRED flat
	// scalar array, length nx*ny*nz, one per-voxel density multiplier -
	// NOT the same field as this struct's own scalar `density` above, which
	// is "cloud"-only and means something entirely different). sigma_a/
	// sigma_s above (the same generic RGB-triple fields homogeneous/cloud/
	// rgbgrid already share) supply the base coefficients GridMediumData<T>
	// scales per-voxel by this - see media.cpp's GridMedium::Create: a
	// single Spectrum sigma_a/sigma_s each, not per-voxel like rgbgrid's
	// own "rgb sigma_a"/"rgb sigma_s".
	std::vector<double> gridDensity;

	// "nanovdb" only - pbrt-v4's own real MakeNamedMedium "nanovdb"
	// "string filename" (REQUIRED - real pbrt-v4's NanoVDBMedium::Create
	// has no default and errors without one, matching this loader's own
	// "empty means absent, warn and skip" convention for every other
	// filename-typed field) and "string gridname" (default "density" -
	// the name of the grid INSIDE the .nvdb file to read; a .nvdb can
	// contain multiple named grids, matching pbrt-v4's own default
	// exactly). AS WRITTEN here (relative to the scene file's own
	// directory) - pbrt_load.h::loadFile() resolves it to a real
	// filesystem path afterward, exactly like Material::textureFilename's
	// own comment documents for every other filename field this loader
	// carries unbaked through flatten(); the actual NanoVDB file read (and
	// its densification into a GridMediumData<double>-compatible flat
	// array) happens later still, in pbrt_cpu_builder.h - this header
	// stays free of both filesystem access and the vendored NanoVDB
	// reader itself (src/external/nanovdb/), matching the same "flatten()
	// captures the request, the builder does the heavy/backend-specific
	// work" split image textures already use (see pbrt_cpu_builder.h's
	// own stbi_load()/LoadEXR() call sites) - keeping pbrt_flatten.h (and
	// everything that transitively includes it, including the GPU
	// builder) free of an unnecessary NanoVDB dependency, not just an
	// arbitrary layering preference.
	//
	// CPU-only, real per-voxel density, reusing GridMediumData<double>/
	// grid_medium_hittable.h completely unchanged - pbrt_cpu_builder.h
	// bakes the (typically sparse) NanoVDB grid's active region into a
	// DENSE flat array at load time (matching how "uniformgrid" already
	// works), rather than keeping it natively sparse; a real, disclosed
	// scope cut, not a NanoVDB limitation. GPU has no NanoVDB support at
	// all this round - scene_builder.cpp warns explicitly, matching the
	// Cone/Paraboloid GPU-unsupported precedent, rather than silently
	// falling back to homogeneous the way an UNRECOGNIZED medium type
	// would (this type IS recognized here; only GPU can't build it).
	//
	// Real pbrt-v4 also supports a "string temperaturename" grid for real
	// blackbody emission (glowing fire/smoke) - IMPLEMENTED: see
	// nanovdbTemperatureGridName's own comment below.
	std::string nanovdbFilename;
	std::string nanovdbGridName = "density";
	// "string temperaturename" - a second named grid in the same .nvdb file
	// (pbrt_cpu_builder.h's nanovdb path reads it the same way as
	// nanovdbGridName's own density grid), whose per-voxel values are real
	// Kelvin temperatures converted to RGB emission via
	// blackbodyKelvinToRGB() (this file, used by resolveEmissionColor() too)
	// and attached to GridMediumData<double> as a real per-voxel Le_grids
	// (GridMediumData::set_emission(), mirroring RGBGridMediumData<T>'s own
	// Le_grids/Le_scale). Empty (the default) means no emission - matches
	// this loader's existing "absent grid = no glow" convention for
	// rgbgrid's own "Le" array. A structural note: this loader's volumetric
	// emission model everywhere (constant_medium.h/rgb_grid_medium_
	// hittable.h) weights emission by sigma_a/sigma_t at the scatter event,
	// so a nonzero temperature grid is only physically meaningful alongside
	// a REAL (nonzero) sigma_a - unlike every OTHER nanovdb medium (which
	// stays pure-scattering, sigma_a forced to 0; see the warning just below
	// this struct), pbrt_cpu_builder.h lets sigma_a through unforced
	// specifically when this field is non-empty.
	std::string nanovdbTemperatureGridName;
	// "float Lescale" - scalar multiplier on the per-voxel blackbody-
	// converted RGB emission, applied at GridMediumData::sample_emission()
	// time (same "keep the scale un-baked, apply at sample time" convention
	// as RGBGridMediumData::Le_scale/rgbgrid's own "Lescale" - see
	// Medium::Le_r's own comment). Has no effect unless
	// nanovdbTemperatureGridName is non-empty. Note this is NOT the same
	// knob as pbrt-v4's real NanoVDB "temperaturescale"/"temperaturecutoff"
	// (which rescale the raw grid value into Kelvin before blackbody
	// conversion) - this loader assumes the named grid's values are already
	// real Kelvin temperatures and only lets a scene dim/brighten the
	// resulting colour afterward; a disclosed scope simplification, not an
	// oversight.
	double nanovdbLeScale = 1.0;
	// The CTM captured at MakeNamedMedium declaration time (pbrt_scene::
	// MediumDecl::xform), UNBAKED - mirrors this loader's own "object-space-
	// plus-unbaked-CTM" technique used for disk/cylinder/cone/paraboloid
	// (see e.g. Cone::xform's own comment). Every OTHER medium type instead
	// gets `toMediumMat`/`toMediumTranslate` above (the ALREADY-inverted,
	// already-composed-with-p0/p1 transform) because flatten() knows their
	// full medium-space box (p0/p1) up front; nanovdb's own "box" is the
	// grid's own index bounding box, which isn't known until the .nvdb file
	// is actually read - so composing the final world<->medium transform
	// has to wait for pbrt_cpu_builder.h, which needs the raw scene CTM
	// (this field) to do it, not the other fields above (left at their
	// harmless unit-cube/identity defaults for this type).
	double nanovdbXform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
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
	// True only for a StartTime-pose duplicate of a patch that's ALSO in
	// FlatScene::animatedBilinearPatches (real CPU motion blur - see that
	// struct's own comment) - mirrors Triangle::gpuOnlyStaticFallback's own
	// identical purpose and comment exactly: GPU has no concept of the
	// separate animated list, so this duplicate exists purely to keep GPU
	// rendering the shape at all (static at its StartTime pose), while
	// pbrt_cpu_builder.h skips building a CPU hittable for a flagged entry
	// (CPU gets its real motion blur from the animatedBilinearPatches entry
	// instead).
	bool gpuOnlyStaticFallback = false;
};

// A bilinearmesh shape under real object motion blur (pbrt-v4
// ActiveTransform "StartTime"/"EndTime" around Shape "bilinearmesh") - CPU
// only. Mirrors AnimatedTriangleMesh's own design exactly (see that struct's
// own comment for the full rationale) - a single patch's 4 corners are
// small enough that, unlike a whole mesh, there's no benefit to a separate
// "flattened object-space list"; the OBJECT-space corners here (unlike
// BilinearPatch::p above, baked to WORLD space at flatten() time) are handed
// straight to bilinear_patch_hittable (which has no transform concept of
// its own - it just uses whatever 4 points it's given), then the whole
// hittable is wrapped in animated_transform_instance.h for real per-ray-time
// motion. Only ever populated when a shape's own xformEnd genuinely differs
// from xform (same "real inequality" convention as every other animated-
// shape field), and never for an emissive patch (NEE needs enumerable
// world-space geometry - same rule AnimatedTriangleMesh's own comment
// documents) - both excluded cases fall back to the existing static,
// StartTime-only bake instead, warned.
struct AnimatedBilinearPatch {
	double p[4][3] = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};  // OBJECT space
	int material = -1;
	double xform[16]    = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	double xformEnd[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

// Shape "curve" - a cubic Bezier hair/fiber strand (src/shared/shapes.h's
// CurveShape<T>, CPU builder) / a tessellated tube of bilinear patches (GPU
// builder, since neither GPU backend has a native curve-intersection program -
// see src/shared/curve_tessellate.h's own comment, and pbrt-v4 itself makes
// the identical choice on its own GPU path). CurveShape itself is always
// exactly cubic Bezier (cp[4][3] per segment), so every "integer degree"/
// "string basis" combination is converted down to that representation before
// reaching it: degree 3 + "bezier" needs no conversion; degree 2 + "bezier"
// is exactly degree-elevated (curveDegreeElevateQuadratic(), below); basis
// "bspline" (degree 3 only - pbrt-v4 has no quadratic B-spline curve either)
// is exactly converted via curveBsplineSegmentToBezierCubic(), below -
// mirroring pbrt-v4's own shapes.cpp ElevateQuadraticBezierToCubic/
// CubicBSplineToBezier. Any other degree/basis combination (degree 2 +
// "bspline", or a degree outside {2,3}) falls through to flatten()'s generic
// "shape not supported" warning - no scene this loader has actually seen
// needs it.
//
// World-space control points, one segment's worth (4 points = 12 doubles)
// per contiguous block - mirrors pbrt-v4's own Curve::Create segment-
// splitting loop exactly: an N-point "point3 P" array with (N-1) a multiple
// of 3 becomes nSegments = (N-1)/3 independent cubic segments, each sharing
// its first control point with the previous segment's last (cpOffset += 3
// per iteration). Each segment becomes its own CurveShape/tessellated tube
// with its own-local uMin=0/uMax=1 - no uMin/uMax sub-range splitting is
// needed for this. "splitdepth" (pbrt-v4's separate, BVH-bounds-tightening
// recursive sub-splitting) is deliberately not implemented: pbrt-v4 itself
// forces splitdepth=0 whenever GPU rendering is active (shapes.cpp: "since
// we dice curves on the GPU we really don't want to have them split here"),
// so omitting it matches pbrt-v4's own GPU-mode behavior, not a corner cut.
struct Curve {
	std::vector<double> cp;   // nSegments * 4 * 3 doubles, world space
	int nSegments = 0;
	// pbrt-v4's own default: "float width" (1.0) sets both, "width0"/
	// "width1" each independently override it.
	double width0 = 1.0, width1 = 1.0;
	std::string curveType = "flat";   // "flat" | "cylinder" | "ribbon"
	// Ribbon only - one shading normal per segment ENDPOINT (nSegments+1
	// total, world space, unit length, transformed via transformNormal not
	// transformPoint) - see flatten()'s own ribbon-normal validation, which
	// mirrors pbrt-v4's Curve::Create (shapes.cpp:824-841) exactly.
	std::vector<double> n;    // (nSegments+1)*3 doubles, ribbon only
	int material = -1;
	int areaLight = -1;
	// True only for a StartTime-pose duplicate of a curve that's ALSO in
	// FlatScene::animatedCurves (real CPU motion blur) - mirrors
	// Triangle::gpuOnlyStaticFallback/BilinearPatch::gpuOnlyStaticFallback's
	// own identical purpose exactly (see either's own comment).
	bool gpuOnlyStaticFallback = false;
};

// A curve shape under real object motion blur (pbrt-v4 ActiveTransform
// "StartTime"/"EndTime" around Shape "curve") - CPU only. Mirrors
// AnimatedTriangleMesh's own design (see that struct's own comment for the
// full rationale): `cp` here is OBJECT space (the same already-degree/basis-
// converted cubic Bezier control points Curve::cp above holds, just before
// the final transformPoint-to-world-space loop - flatten()'s own curve
// branch already computes this as `objCp` regardless of degree/basis, so no
// new conversion math is needed here, only skipping the bake). All of a
// curve declaration's segments share ONE xform/xformEnd pair (unlike a
// mesh's many triangles, still cheap enough to wrap as a single group - see
// pbrt_cpu_builder.h's own animated-curve build site), so this holds every
// segment's object-space control points together, not one struct per
// segment. Only ever populated when a shape's own xformEnd genuinely
// differs from xform, and never for an emissive OR ribbon-type curve
// (emissive: NEE needs enumerable world-space geometry, same rule as
// AnimatedTriangleMesh's own comment; ribbon: its per-segment-endpoint
// shading normals - Curve::n above - would need their own per-ray-time
// transform too, a real but narrower feature scoped out here) - both
// excluded cases fall back to the existing static, StartTime-only bake
// instead, warned.
struct AnimatedCurve {
	std::vector<double> cp;   // nSegments * 4 * 3 doubles, OBJECT space
	int nSegments = 0;
	double width0 = 1.0, width1 = 1.0;
	std::string curveType = "flat";   // never "ribbon" - see this struct's own comment
	int material = -1;
	double xform[16]    = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	double xformEnd[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
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
	// resolution code for how those names become indices. Real on all three
	// backends: CPU's src/TheRestOfYourLife/material_pbrt.h `class
	// mix_material` (added earlier for SPPM support, a real generic
	// two-material blend, not SPPM-specific) backs it directly; both GPU
	// backends resolve it at each shading point to a real sub-material
	// (deterministically, hashed from the hit point - not a per-ray random
	// draw, matching mix_material's own scatter()/scattering_pdf()/shadow-
	// ray-classification agreement requirement) via MaterialType::Mix - see
	// that enumerator's own comment (gpu/optix/optix_types.h) and gpu/optix/
	// pbrt_gpu_builder.h's Mix case, which only falls back to a flat-colour
	// average (its pre-existing, now-secondary behavior) for a
	// pathologically deep/cyclic mix-of-mix chain.
	Mix,
	// Marschner/Chiang fiber scattering (src/shared/bxdfs_hair.h's
	// HairBxDF<T>) - real on both CPU (src/TheRestOfYourLife/
	// hair_material.h) and GPU (MaterialType::Hair, already fully wired for
	// shading on both GPU backends - this loader only needed to populate its
	// existing fields). See flatten()'s own Hair branch for the
	// sigma_a/reflectance/eumelanin-pheomelanin priority order (matches
	// pbrt-v4's HairMaterial::Create exactly) and Material::betaM/betaN/
	// alphaDeg's own comments for the remaining parameters.
	Hair,
	// pbrt-v4's real "interface material" idiom (`Material "none"` or
	// `Material ""`) - a shape that bounds a participating medium with no
	// BSDF response of its own; the ray passes straight through, only the
	// medium changes. Built on both backends as a real, dedicated pass-
	// through material (CPU's interface_material, material_simple.h;
	// GPU's MaterialType::Interface) - no Fresnel/refraction math, no
	// critical angle, and a real "nothing happened here" signal every
	// integrator (default path tracer, BDPT/MLT, SPPM, both backends)
	// uses to skip the crossing entirely rather than treating it as a
	// specular bounce.
	Interface,
	// This loader's own non-standard extension - NOT a real pbrt-v4 material
	// type (pbrt-v4 has no "principled"/Disney-style material at all; its
	// closest real equivalents are CoatedDiffuse (metallic=0) and
	// CoatedConductor (metallic=1), which this loader already fully
	// supports as their own, separate, real material kinds). Exists purely
	// to expose this codebase's own already-implemented, already-working
	// PrincipledBxDF<T> (src/shared/bxdfs_principled.h; CPU: src/
	// TheRestOfYourLife/principled_material.h's `principled` class; GPU:
	// MaterialType::Principled/MaterialData::principled_params) from a
	// .pbrt file, for scenes whose whole point IS demonstrating that
	// specific, unified base_color/metallic/roughness/clearcoat parameter
	// space in one material rather than pbrt-v4's own split
	// dielectric-coat-over-diffuse-or-conductor modeling. "reflectance"/
	// "roughness"/"eta" reuse the exact same generic parsing every other
	// kind already shares (Material::color/roughness/ior above) - only
	// "metallic"/"clearcoat"/"clearcoatroughness" are new, Principled-only
	// fields (see their own comments below).
	Principled,
	// This loader's own second non-standard extension (see MaterialKind::
	// Principled's own comment for the first) - NOT a standalone real
	// pbrt-v4 material either. pbrt-v4's real NormalizedFresnelBxDF is an
	// internal implementation detail of its own "subsurface" material (the
	// Fresnel-weighted diffuse BRDF shaded at a BSSRDF probe's exit point,
	// already real and working here too - MaterialKind::Subsurface above),
	// never exposed as its own user-facing top-level material in real
	// pbrt-v4. Exists purely to expose this codebase's own already-
	// implemented, already-working standalone version of that same BRDF in
	// isolation (CPU: src/TheRestOfYourLife/material_pbrt.h's
	// `normalized_fresnel` class; GPU: MaterialType::NormalizedFresnel,
	// shared by both GPU backends' real Subsurface exit-point shading too)
	// for a scene demonstrating the Fresnel-weighted-diffuse term on its
	// own, with no actual subsurface light transport underneath. Single
	// parameter: "eta" - reuses the exact same generic Material::ior
	// parsing every other kind already shares; no new fields at all.
	NormalizedFresnel,
	Unsupported,
};

// This codebase's own long-standing default gamma-decode exponent for
// every 8-bit texture (matches src/shared/mipmap.h's own
// kDefaultImagemapGamma / src/TheRestOfYourLife/rtw_stb_image.h's own
// kDefaultGamma conceptually - not referenced directly, since this header
// deliberately stays free of those renderer-layer headers, see the top-of-
// file comment on pbrt_flatten.h/pbrt_scene.h being testable without a
// filesystem or renderer types). Named once here so Material::textureGamma
// and resolveTextureGamma() (flatten_detail, below) share one literal
// instead of 3 independent copies of `2.2`.
constexpr double kDefaultTextureGamma = 2.2;

// Texture "imagemap"'s own "string encoding"/"string wrap"/"bool invert"
// (pbrt-v4), resolved once per bound imagemap - see Material::textureGamma's
// own comment (below) for what each field means and defaults to. Originally
// only threaded for the primary reflectance-equivalent slot (Material::
// textureFilename, as 3 loose fields); this struct exists so the
// transmittance/roughness slots below can carry the identical resolution
// without duplicating those 3 fields a second and third time - see
// resolveTextureDecodeOptions() (flatten_detail, below) for how it's built.
struct TextureDecodeOptions {
	double gamma = kDefaultTextureGamma;
	std::string wrap = "repeat";
	int wrapIndex = 1;   // Repeat - see Material::textureWrapIndex's own comment
	bool invert = false;
};

// A second level of checkerboard/mix nesting: tex1/tex2 (or mix's amount)
// binds to ANOTHER "checkerboard"/"mix" Texture instead of a flat literal
// or a bare imagemap - e.g. a checker-of-checkers, or a mix blending two
// checker patterns. Deliberately capped at exactly this one extra level:
// THIS struct's own tex1/tex2/amount slots may only be a flat literal or a
// bare imagemap (the same one-level scope Material::checkerTex1Filename's
// own comment documents), never a further nested procedural texture -
// unbounded recursion would need real cycle/depth guarding this loader
// has never needed before, for a feature real pbrt-v4 scenes essentially
// never use past 2 levels. `kind` ("checkerboard" or "mix") is empty when
// this slot isn't a nested-procedural at all (the common case) - checked
// before color1/color2/etc, which otherwise carry their own always-valid
// defaults and can't distinguish "unset" from "genuinely default".
// checkerboard's uscale/vscale and mix's amount/amountFilename fields
// simply go unused for the other kind.
struct NestedProceduralTexture {
	std::string kind;
	double color1[3] = {1.0, 1.0, 1.0};
	double color2[3] = {0.0, 0.0, 0.0};
	std::string tex1Filename;
	std::string tex2Filename;
	double uscale = 1.0, vscale = 1.0;   // checkerboard only
	double amount = 0.5;                  // mix only
	std::string amountFilename;           // mix only
};

// GPU's flat-colour stand-in for a resolved NestedProceduralTexture (GPU has
// no representation for a nested procedural entry - see Material::
// checkerTex1Nested's own comment). A plain 50/50 average is exactly right
// for "checkerboard" (its two cells cover equal area by construction), but
// would silently ignore "mix"'s own amount - weight by it instead, unless
// amount itself is texture-bound (amountFilename set), where there's no
// single per-point value available at flatten() time to weight by, so 50/50
// is the best available approximation, same as it is for checkerboard.
inline void nestedProceduralAverageColor(const NestedProceduralTexture &n, double out[3]) {
	const double w = (n.kind == "mix" && n.amountFilename.empty()) ? n.amount : 0.5;
	for (int i = 0; i < 3; ++i)
		out[i] = (1.0 - w) * n.color1[i] + w * n.color2[i];
}

// Same idea as nestedProceduralAverageColor() just above, collapsed to a
// single scalar (plain channel average of that same approximate colour) -
// used when a NestedProceduralTexture stands in for a top-level "mix"
// texture's own "amount" slot (Material::mixAmountNested's own comment):
// GPU has no per-point representation for a nested-procedural amount any
// more than it does for a nested-procedural tex1/tex2 (TextureData's
// amountImageIdx is image-only, same as tex1ImageIdx/tex2ImageIdx), so this
// feeds Material::mixAmount - the SAME flat fallback GPU already reads
// whenever amountImageIdx isn't set - with a representative value instead
// of leaving it at the meaningless struct default.
inline double nestedProceduralAverageScalar(const NestedProceduralTexture &n) {
	double c[3];
	nestedProceduralAverageColor(n, c);
	return (c[0] + c[1] + c[2]) / 3.0;
}

struct Material {
	MaterialKind kind = MaterialKind::Diffuse;
	std::string pbrtType;              // as written, for diagnostics
	double color[3] = {0.5, 0.5, 0.5}; // reflectance / albedo
	double roughness = 0.0;
	// Independent GGX roughness along the surface's own tangent (u) and
	// bitangent (v) directions - pbrt-v4's real anisotropic spelling
	// ("uroughness"/"vroughness"). Populated alongside `roughness` above
	// (which stays exactly as before, an isotropic fallback derived the
	// same way it always has, for callers that haven't been updated to
	// read these two instead - see Round 6 Phase 3). Each independently
	// falls back to plain "roughness" when its own name isn't present, so
	// an ordinary isotropic scene (only "roughness" set) yields
	// roughness_u == roughness_v == roughness, and Round 6 Phase 3's own
	// downstream wiring (pbrt_cpu_builder.h/pbrt_gpu_builder.h) degrades
	// to the exact isotropic construction it used before whenever the two
	// happen to be equal.
	double roughness_u = 0.0;
	double roughness_v = 0.0;
	// pbrt-v4's "remaproughness" (default true): when true, roughness/
	// roughness_u/roughness_v above are perceptually-remapped authored
	// values that still need RoughnessToAlpha applied before use as real
	// GGX alpha; when false, they already ARE the alpha value. See
	// material_pbrt.h's roughness_or_alpha() for where this gets applied.
	bool remapRoughness = true;
	// Also reused for Hair's own "eta" (fiber IOR) - flatten()'s Hair branch
	// overrides this field's default to pbrt-v4's own Hair-specific 1.55
	// (HairMaterial::Create) rather than this field's general 1.5.
	double ior = 1.5;
	// Principled only (this loader's own non-standard material - see
	// MaterialKind::Principled's own comment): 0 = pure dielectric/plastic,
	// 1 = pure metal, blended in between - matches `principled` (CPU,
	// principled_material.h) and MaterialData::principled_params.metallic
	// (GPU) exactly, same default (0.0) as that CPU class's own convenience
	// constructor.
	double metallic = 0.0;
	// Principled only: clearcoat layer weight (0 = none) and its own
	// independent specular roughness - matches `principled`'s clearcoat_tex/
	// clearcoat_rough and MaterialData::principled_params.clearcoat/
	// clearcoat_rough exactly, same defaults (0.0/0.1) as that CPU class's
	// own convenience constructor. Deliberately separate fields from
	// roughness/roughness_u/roughness_v above (CoatedDiffuse/CoatedConductor's
	// own coat-roughness slots) - Principled's clearcoat is a SEPARATE third
	// layer on top of its own base metallic/roughness response, not a
	// substitute for either.
	double clearcoat = 0.0;
	double clearcoatRoughness = 0.1;
	// Dielectric only (smooth or rough, via m.roughness/m.roughness_u/
	// m.roughness_v above) - an ARTIST-FACING ABBE NUMBER, not real pbrt-v4
	// syntax at all (pbrt-v4's own genuine spectral dispersion is a full
	// "spectrum eta" per-wavelength curve, which this loader doesn't
	// integrate). 0.0 (the default) means "not dispersive" - a real Abbe
	// number is always positive (common glasses run roughly 20-90), so this
	// needs no separate bool, same "zero means off" convention
	// dispersive_extra.cauchy_A > 0.0f already uses on the GPU side
	// (optix_types.h). Reachable only under --spectral (per-wavelength
	// hero-wavelength tracing) - see dispersive_material's own comment,
	// material_base.h, for why a dispersive material renders identically to
	// a flat-IOR one under the default RGB path. Thin wrapper around this
	// codebase's own already-implemented, already-working
	// dielectric::make_dispersive()/rough_dielectric::make_dispersive()
	// (CPU) and add_dispersive_dielectric()/add_dispersive_rough_dielectric()'s
	// own Cauchy-coefficient derivation (GPU, CauchyCoefficientsFromAbbe() -
	// src/shared/fresnel.h) - not a new rendering feature, purely new pbrt-
	// file reachability for an existing one.
	double abbeNumber = 0.0;
	// Dielectric only - "tf", a non-standard "rgb" parameter: the OBJ/.mtl
	// "Tf" transmission filter (stained glass, tinted windows). Multiplies the
	// transmitted contribution only, leaving reflection clear - what the native
	// dielectric(ior, Tf) / MaterialData::transmission_filter always did; the
	// environment scenes migrated from .mtl assets (pbrt_scenes/environment-*.pbrt)
	// are the only users. White (the default) is a no-op, and a rough or
	// dispersive dielectric ignores it, as the native glass did.
	double transmissionFilter[3] = {1.0, 1.0, 1.0};
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
	//
	// ALSO reused for Hair's own absorption coefficient (see flatten()'s Hair
	// branch) - the struct-level default above is meaningless for Hair
	// (always overwritten there, never left at this Subsurface-shaped
	// default): a genuine Hair material always resolves sigma_a[] to one of
	// a literal "sigma_a", a computed eumelanin/pheomelanin concentration, or
	// pbrt-v4's own default-brown fallback.
	double sigma_a[3] = {0.0011, 0.0024, 0.014};
	double sigma_s[3] = {2.55, 3.21, 3.77};
	double g = 0.0;   // Henyey-Greenstein asymmetry (subsurface only)

	// Hair only - longitudinal/azimuthal roughness and cuticle scale-tilt
	// angle (degrees). Defaults match pbrt-v4's own HairMaterial::Create
	// exactly (materials.cpp), which happen to already match hair_material.h's
	// own pre-existing constructor defaults.
	double betaM = 0.3, betaN = 0.3, alphaDeg = 2.0;

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

	// Diffuse, CoatedDiffuse, or DiffuseTransmission's own "reflectance": an
	// "imagemap" Texture bound to it, naming the image file exactly AS
	// WRITTEN in the scene - same "stays filesystem-free, resolved later by
	// pbrt_load.h" convention as measuredFilename above (see that field's
	// own comment). Empty means either no texture was bound, the bound
	// texture wasn't an imagemap (or a "scale" wrapping one - see
	// textureScale below, Diffuse/CoatedDiffuse only), or (after
	// pbrt_load.h's pass) the file could not be found - any of which falls
	// back to `color` as a flat reflectance, same as today. Only
	// "reflectance" is handled (not every texture-bindable parameter on
	// every material kind): pbrt's own ganesha scene (a CoatedDiffuse statue
	// whose reflectance is an imagemap), barcelona-pavilion's CoatedDiffuse
	// AND plain-Diffuse surfaces (mostly reflectance bound to a "scale"
	// texture wrapping an imagemap - see the warning loop below) and
	// barcelona-pavilion's own foliage (DiffuseTransmission, "texture
	// reflectance"/"texture transmittance" both bound to the SAME bare
	// imagemap - see transmittanceTextureFilename below) are the motivating
	// cases, and scoping to reflectance/transmittance on these three kinds
	// keeps this addition bounded rather than building a general
	// procedural-texture pipeline in one pass - checkerboard/fbm/marble/mix
	// (hasCheckerReflectance etc. below) and the "scale" unwrap (see
	// textureScale below) both stay Diffuse/CoatedDiffuse-only, since no
	// bundled scene needs DiffuseTransmission's own reflectance bound to
	// either.
	std::string textureFilename;

	// A "scale"-class Texture's own "float scale" when textureFilename came
	// from unwrapping one (pbrt's own real syntax for this: a named "scale"
	// Texture whose "texture tex" names the real imagemap - barcelona-
	// pavilion's own dominant pattern for reflectance, already unwrapped the
	// identical way for "displacement" below, see displacementScale's own
	// comment). Diffuse/CoatedDiffuse/DiffuseTransmission all apply this to
	// their own reflectance now (scaled_texture on CPU, MaterialData::
	// emissionScale reused on GPU - see transmittanceTextureScale below for
	// DiffuseTransmission's OWN transmittance, a separate field since the
	// two channels can be bound to independently-scaled textures). 1.0 (a
	// no-op multiply) when textureFilename came from a bare imagemap with
	// no wrapping "scale", or when textureFilename is empty.
	double textureScale = 1.0;

	// Texture "imagemap"'s own "string encoding"/"string wrap"/"bool
	// invert" (pbrt-v4), resolved for textureFilename above as 3 loose
	// fields (kept exactly as-is - CPU's imageMapOptionsFor()/GPU's own call
	// sites already read these 3 by name) - transmittanceTextureFilename/
	// roughnessTextureFilename below carry the identical resolution via
	// their own TextureDecodeOptions field instead (transmittanceTextureOptions/
	// roughnessTextureOptions). alphaTextureFilename/displacementTextureFilename
	// still don't: alpha is a coverage MASK, not colour, so "encoding"
	// (gamma) is not meaningful there by this codebase's own established
	// design (see gpu/optix/pbrt_gpu_builder.h's getOrBuildPbrtAlphaMaskTexture()
	// own comment on why alpha masks deliberately skip the gamma decode a
	// reflectance imagemap needs); displacement goes through a materially
	// different CPU pipeline (rtw_image/image_texture, not mipmap_texture/
	// MipMapOptions) with no wrap-mode concept at all today. Both remaining
	// gaps still warn via warnIfImagemapOptionsIgnored() below rather than
	// silently dropping the request. textureGamma: resolved from
	// "encoding" to an actual gamma exponent - "linear" -> 1.0 (no decode,
	// pbrt-v4's own real intent for a roughness/normal/displacement map
	// bound this way), "gamma <value>" -> that value, "sRGB" or absent ->
	// 2.2 (this codebase's own long-standing default for every 8-bit
	// texture - an approximation of pbrt-v4's real sRGB curve, not its
	// exact piecewise-linear-toe formula, matching this loader's existing
	// "close enough, not bit-exact" precedent for other approximated
	// features). See src/TheRestOfYourLife/rtw_stb_image.h's rtw_image
	// constructor for what this value actually does downstream.
	double textureGamma = kDefaultTextureGamma;
	// "repeat" (pbrt-v4's own real default)/"clamp"/"black" - unlike
	// mipmap.h's own MipMapOptions::wrap field default (Clamp, kept
	// unchanged there deliberately so native/non-pbrt scenes are
	// unaffected - see that field's own comment), THIS field's default is
	// pbrt-v4's real one, applied explicitly per pbrt-loaded texture.
	std::string textureWrap = "repeat";
	// The same resolved value as textureWrap above, as an ordinal
	// (Clamp=0/Repeat=1/Black=2) instead of a string - matches both CPU's
	// MipWrapMode (src/shared/mipmap.h) and GPU's GpuWrapMode (gpu/optix/
	// optix_types.h) enumerator values exactly by construction (both were
	// defined to mirror this ordering), so either backend's own builder can
	// just `static_cast` this into its own enum type instead of
	// re-implementing the same wrap-string validation/fallback logic this
	// struct's own resolution code (below, in flatten()) already owns.
	// textureWrap itself is kept alongside this (not replaced) since it's
	// still useful as a human-readable value for anything that wants one.
	int textureWrapIndex = 1;  // Repeat, matching textureWrap's own default
	bool textureInvert = false;

	// DiffuseTransmission only: an "imagemap" Texture bound to
	// "transmittance", same "raw as written, resolved later by
	// pbrt_load.h" convention as textureFilename above - bare imagemap,
	// optionally further wrapped in a "scale" texture (transmittanceTexture
	// Scale below) - same one-level unwrap textureScale's own comment
	// documents for reflectance, applied independently here since
	// barcelona-pavilion's foliage happens to bind "reflectance" and
	// "transmittance" to the identical bare-imagemap texture in practice,
	// but a scene binding them to two DIFFERENTLY-scaled textures must
	// still resolve each correctly. No procedural (checkerboard/fbm/marble/
	// mix) support - no bundled scene needs it.
	std::string transmittanceTextureFilename;

	// transmittanceTextureFilename's own "scale", same shape as
	// textureScale above but independent - a DiffuseTransmission material's
	// reflectance and transmittance can each be wrapped in their own,
	// differently-valued "scale" Texture. 1.0 (a no-op multiply) when
	// transmittanceTextureFilename came from a bare imagemap with no
	// wrapping "scale", or when transmittanceTextureFilename is empty.
	double transmittanceTextureScale = 1.0;

	// This slot's own "encoding"/"wrap"/"invert" - see TextureDecodeOptions'
	// own comment. Default-constructed (2.2/"repeat"/no-invert) when
	// transmittanceTextureFilename is empty or the scene gave none of the
	// three.
	TextureDecodeOptions transmittanceTextureOptions;

	// Dielectric only: an "imagemap" Texture bound to "roughness" (e.g. a
	// scratched/frosted-glass mask), same "raw as written, resolved later
	// by pbrt_load.h" convention as textureFilename above - bare imagemap
	// only, no "scale"-wrap or procedural (checkerboard/fbm/marble/mix)
	// support, matching transmittanceTextureFilename's identical scope
	// narrowing. No bundled scene needs this (added for texture-parity
	// with Diffuse/CoatedDiffuse/DiffuseTransmission's own reflectance/
	// transmittance texture-binding, not a specific scene's requirement) -
	// the image's own red/x channel becomes the scalar roughness at each
	// hit; see rough_dielectric::true_alpha()'s own comment (material_pbrt.h)
	// for why this is isotropic-only, sampled per-hit rather than once.
	std::string roughnessTextureFilename;

	// This slot's own "encoding"/"wrap"/"invert" - see
	// transmittanceTextureOptions' own comment just above.
	TextureDecodeOptions roughnessTextureOptions;

	// A Diffuse material's "reflectance" bound to a "checkerboard" Texture
	// instead of an "imagemap" one (e.g. named-material-and-texture.pbrt's
	// "floor-check": Texture "floor-check" "spectrum" "checkerboard"
	// "float uscale" [8] "float vscale" [8] - no tex1/tex2 given, so
	// pbrt-v4's own defaults apply). Unlike textureFilename, this can't be
	// represented as a plain filename (there is no file - it's two flat
	// colours procedurally tiled by UV), hence the separate fields below
	// rather than overloading textureFilename's meaning. hasCheckerReflectance
	// is the "is this meaningful" flag, since an all-default checkerboard's
	// fields are otherwise indistinguishable from "unset".
	//
	// tex1/tex2 each independently support up to TWO levels of nesting: a
	// flat float/rgb literal (checkerColor1/2 below), a reference to a bare
	// "imagemap" Texture (checkerTex1Filename/checkerTex2Filename below), or
	// a reference to ANOTHER "checkerboard"/"mix" Texture
	// (checkerTex1Nested/checkerTex2Nested below, CPU-only - see that
	// struct's own comment for why exactly two levels and no GPU support).
	// Exactly one of the three (checkerColorN / checkerTexNFilename /
	// checkerTexNNested) is meaningful per slot - see flatten()'s own
	// checkerboard-resolution code for which.
	bool hasCheckerReflectance = false;
	double checkerColor1[3] = {1.0, 1.0, 1.0};  // pbrt-v4 tex1 default: white
	double checkerColor2[3] = {0.0, 0.0, 0.0};  // pbrt-v4 tex2 default: black
	std::string checkerTex1Filename;  // set instead of checkerColor1 when tex1 nests a bare imagemap
	std::string checkerTex2Filename;  // set instead of checkerColor2 when tex2 nests a bare imagemap
	NestedProceduralTexture checkerTex1Nested;  // kind non-empty when tex1 nests a further checkerboard/mix
	NestedProceduralTexture checkerTex2Nested;
	double checkerUScale = 1.0;
	double checkerVScale = 1.0;
	// pbrt-v4's real "checkerboard" texture also supports "integer
	// dimension" [3] - a 3D WORLD-SPACE checker (CheckerboardTexture::
	// Evaluate's dimension==3 branch), keyed on a texture-space point
	// instead of (u,v) - checkerUScale/checkerVScale above are meaningless
	// for this variant (real pbrt-v4 ignores them too). This is the exact
	// same pattern this project's OWN original (pre-pbrt) checker_texture
	// (src/TheRestOfYourLife/texture.h) already implements - see that
	// class's own header comment. checkerWorldToTexture is the inverse of
	// the CTM active when the Texture directive was declared
	// (TextureDecl::xform's own comment) - a plain "Scale s s s" before the
	// Texture directive reproduces this project's own native checker_
	// texture(scale, ...) constructor exactly, since its inv_scale=1/scale
	// is exactly what Scale(s)'s own inverse applies to a world point.
	// Row-major affine 4x4, identity when dimension isn't 3 (checkerIs3D
	// stays false so callers never need to consult this field).
	bool checkerIs3D = false;
	double checkerWorldToTexture[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

	// A Diffuse material's "reflectance" bound to an "fbm" Texture (pbrt-v4
	// FBmTexture - fractional Brownian motion noise, e.g. a cloudy/mottled
	// pattern). Same flat-literal-only scope as checkerboard above - "fbm"
	// takes no tex1/tex2 to begin with (it IS the pattern, not a blend of
	// two others), so there's no nested-texture case to fall through on.
	// Param names match pbrt-v4 exactly: "octaves" (int) and "roughness"
	// (float, internally called omega) - see fbm_texture (src/TheRestOfYourLife/
	// texture.h), the existing CPU class this resolves to (already used by
	// non-pbrt scenes; this just gives the pbrt loader a way to reach it).
	bool hasFbmReflectance = false;
	int fbmOctaves = 8;
	double fbmRoughness = 0.5;

	// A Diffuse material's "reflectance" bound to a "marble" Texture
	// (pbrt-v4 MarbleTexture - FBm-perturbed sine wave through a marble
	// colour spline). Resolves to the existing marble_texture CPU class
	// (src/TheRestOfYourLife/texture.h). Param names match pbrt-v4 exactly.
	bool hasMarbleReflectance = false;
	int marbleOctaves = 8;
	double marbleRoughness = 0.5;
	double marbleScale = 1.0;
	double marbleVariation = 0.2;

	// A Diffuse material's "reflectance" bound to a "mix" Texture (pbrt-v4
	// SpectrumMixTexture - lerp between two colours by "amount"). tex1/tex2
	// support the SAME up-to-two-level nesting as checkerboard's own
	// tex1/tex2 above (mixTex1Filename/mixTex2Filename for a bare imagemap;
	// mixTex1Nested/mixTex2Nested for a further checkerboard/mix, CPU-only -
	// see NestedProceduralTexture's own comment). "amount" ITSELF bound to a
	// texture (e.g. driven by an fbm pattern for a dirt/wear mask, pbrt-v4's
	// most common real use of "mix" - barcelona-pavilion's own
	// materials.pbrt has a commented-out "float amount" override on several
	// Mix declarations, hinting the original scene author considered
	// exactly this) is ALSO supported now, at the SAME up-to-two-level
	// nesting as tex1/tex2 (mixAmountTextureFilename below for a bare
	// imagemap; mixAmountNested for a further checkerboard/mix, CPU-only -
	// see NestedProceduralTexture's own comment) - a per-point scalar blend
	// mask driven by a further procedural pattern (e.g. a checker-driven
	// mix, or a mix-of-mixes weight) is a real pbrt-v4 capability this
	// loader now reaches too. A nested amount that resolves to neither
	// level still falls through to the generic "not supported" warning.
	// Defaults match pbrt-v4's SpectrumMixTexture exactly (tex1 black, tex2
	// white, amount 0.5).
	bool hasMixReflectance = false;
	double mixColor1[3] = {0.0, 0.0, 0.0};
	double mixColor2[3] = {1.0, 1.0, 1.0};
	std::string mixTex1Filename;  // set instead of mixColor1 when tex1 nests a bare imagemap
	std::string mixTex2Filename;  // set instead of mixColor2 when tex2 nests a bare imagemap
	NestedProceduralTexture mixTex1Nested;  // kind non-empty when tex1 nests a further checkerboard/mix
	NestedProceduralTexture mixTex2Nested;
	double mixAmount = 0.5;
	std::string mixAmountTextureFilename;  // set instead of mixAmount when "amount" nests a bare imagemap
	NestedProceduralTexture mixAmountNested;  // kind non-empty when "amount" nests a further checkerboard/mix

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "windy" Texture
	// (pbrt-v4 WindyTexture - two FBm calls combined for a windswept-grass
	// pattern). Parameterless in real pbrt-v4 (no scene-overridable params
	// at all), so this is just the "is this bound" flag - see
	// windy_texture's own comment (texture.h) for the formula.
	bool hasWindyReflectance = false;

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "wrinkled" Texture
	// (pbrt-v4 WrinkledTexture - raw Turbulence, not FBm). Same "octaves"/
	// "roughness" param names/defaults as fbm above.
	bool hasWrinkledReflectance = false;
	int wrinkledOctaves = 8;
	double wrinkledRoughness = 0.5;

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "dots" Texture
	// (pbrt-v4 DotsTexture - a per-UV-cell polka-dot pattern blending
	// "inside"/"outside"). Same one-level-nested-bare-imagemap support for
	// inside/outside as checkerboard's own tex1/tex2 (dotsInsideTexFilename/
	// dotsOutsideTexFilename below). Defaults match pbrt-v4's DotsTexture
	// exactly (inside white, outside black).
	bool hasDotsReflectance = false;
	double dotsInsideColor[3] = {1.0, 1.0, 1.0};
	double dotsOutsideColor[3] = {0.0, 0.0, 0.0};
	std::string dotsInsideTexFilename;   // set instead of dotsInsideColor when "inside" nests a bare imagemap
	std::string dotsOutsideTexFilename;  // set instead of dotsOutsideColor when "outside" nests a bare imagemap

	// A Diffuse/CoatedDiffuse "reflectance" bound to a "bilerp" Texture
	// (pbrt-v4 BilerpTexture - plain bilinear blend of 4 corner colours by
	// (u,v)). Flat-literal corners only - no nested-imagemap support,
	// matching how rarely a real scene binds anything but a flat colour to
	// a bilerp corner (no bundled scene needs more). Defaults match
	// pbrt-v4's BilerpTexture exactly (v00/v10 black, v01/v11 white).
	bool hasBilerpReflectance = false;
	double bilerpV00[3] = {0.0, 0.0, 0.0};
	double bilerpV01[3] = {1.0, 1.0, 1.0};
	double bilerpV10[3] = {0.0, 0.0, 0.0};
	double bilerpV11[3] = {1.0, 1.0, 1.0};

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
	// that material index - flatten() warns when this happens (see its own
	// Shape "alpha" handling), but does not prevent it.
	std::string alphaTextureFilename;

	// A Material's own "texture displacement" parameter (pbrt-v4 bump
	// mapping, e.g. barcelona-pavilion's "pavet" material: MakeNamedMaterial
	// ... "texture displacement" ["pavet-bump"]) - a "float" texture, NOT
	// gated on MaterialKind::Diffuse the way textureFilename's reflectance
	// binding is, since real scenes bind displacement on coateddiffuse,
	// dielectric, and other kinds too. Raw as written, resolved later by
	// pbrt_load.h - same convention as textureFilename/measuredFilename/
	// alphaTextureFilename above. Mirrors OBJ/MTL's own map_Bump handling
	// (mesh.h) once resolved - see pbrt_cpu_builder.h/pbrt_gpu_builder.h's
	// own comments on how this field is consumed.
	std::string displacementTextureFilename;
	// A scale factor from a "scale"-class texture wrapping the actual
	// imagemap (e.g. barcelona-pavilion's water material: "texture
	// displacement" ["water-bump"], where "water-bump" is itself
	// "float" "scale" { "float scale" [0.005], "texture tex" ["water-bump-base"] }).
	// 1.0 (no-op) when displacement resolves directly to an imagemap with no
	// wrapping scale texture.
	double displacementScale = 1.0;

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

// Extracts "BK7" from pbrt-v4's "glass-BK7" named-spectrum convention (the
// only shape a dielectric's "spectrum eta" uses - see glass_data.h's own
// comment for the full 7-name list pbrt-v4 actually recognizes). Returns ""
// for anything that doesn't start with "glass-" at all (an explicit float
// eta, or a non-glass named spectrum).
inline std::string glassElementFromSpectrumName(const std::string &name) {
	const std::string prefix = "glass-";
	if (name.rfind(prefix, 0) != 0) return "";
	return name.substr(prefix.size());
}

// Resolves an emission-colour parameter (pbrt-v4's "L" on AreaLightSource/
// distant/infinite, "I" on point/spot/goniometric), handling both the
// ordinary flat "rgb"/"float" case (delegates to ParamList::getVec3 exactly
// as before - unaffected) and a "blackbody" temperature in Kelvin, which
// this loader previously either warned-and-ignored (AreaLightSource) or
// silently dropped with no warning at all (every punctual light kind) -
// getVec3 requires >=3 numbers, and a "blackbody L" param has exactly 1, so
// it always fell through to the flat default colour regardless of the
// requested temperature (see docs/PBRT_SUPPORT.md's own note on this, and
// barcelona-pavilion's/contemporary-bathroom's real "blackbody L" area
// lights - the motivating case: every one of them rendered as flat white at
// whatever "float scale" said, with zero hue difference between a 2500K and
// a 6500K light).
//
// Converts via this codebase's own already-ported pbrt-v4 spectral pipeline
// (BlackbodySpectrum -> SpectrumToXYZ -> RGBColorSpace::sRGB(), spectrum_
// types.h/spectral_math.h/rgb_colorspace.h - no new spectral machinery
// needed) exactly matching pbrt-v4's own light-construction code (e.g.
// DiffuseAreaLight::Create, lights.cpp): the blackbody spectrum is
// normalized so its own photometric integral (InnerProduct against the CIE
// Y curve - matches pbrt-v4's SpectrumToPhotometric exactly, NOT the same
// as SpectrumToXYZ's Y, which additionally divides by CIE_Y_integral) comes
// out to ~1 nit, BEFORE any separate "float scale" the light also specifies
// - that scale is applied afterward by the existing `L * scale` multiply
// every consumer (CPU/GPU builder) already does downstream, unchanged, so
// this function's return value slots into that exact same pattern.
// Converts a single Kelvin temperature to sRGB via this codebase's own
// already-ported pbrt-v4 spectral pipeline - the same normalize-to-~1-nit-
// photometric-integral technique resolveEmissionColor() (just below) uses
// for a scene's "blackbody L"/"I" param, factored out here so a per-voxel
// caller (pbrt_cpu_builder.h's nanovdb "temperaturename" grid bake, the
// motivating case - see Medium::nanovdbTemperatureGridName's own comment)
// doesn't need a fake single-number ParamList to reuse it.
inline pbrt_scene::Vec3 blackbodyKelvinToRGB(float T,
                                              const RGBColorSpace &colorSpace = RGBColorSpace::sRGB()) {
	if (T <= 0.0f) return pbrt_scene::Vec3{0.0, 0.0, 0.0};
	const BlackbodySpectrum bb(T);
	const XYZ xyz = SpectrumToXYZ(bb);
	const float photometric = InnerProduct(GetCIE_Y(), bb);
	const float norm = (photometric > 0.0f) ? (1.0f / photometric) : 0.0f;
	float r, g, b;
	colorSpace.FromXYZ(xyz.X * norm, xyz.Y * norm, xyz.Z * norm, r, g, b);
	// Small negative components are possible near the edge of the sRGB
	// gamut even for a physically real source - clamp rather than let a
	// negative emission subtract light, matching every other colour path in
	// this loader's own convention of clamping at the edges.
	return pbrt_scene::Vec3{ std::fmax(0.0, static_cast<double>(r)),
							  std::fmax(0.0, static_cast<double>(g)),
							  std::fmax(0.0, static_cast<double>(b)) };
}

inline pbrt_scene::Vec3 resolveEmissionColor(const pbrt_scene::ParamList &params,
                                              const char *name, pbrt_scene::Vec3 def,
                                              const RGBColorSpace &colorSpace = RGBColorSpace::sRGB()) {
	const pbrt_scene::Param *p = params.find(name);
	if (p && p->type == "blackbody" && !p->numbers.empty()) {
		return blackbodyKelvinToRGB(static_cast<float>(p->numbers[0]), colorSpace);
	}
	return params.getVec3(name, def);
}

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

struct FlatScene {
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

	bool empty() const {
		return triangles.empty() && spheres.empty() && instances.empty();
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

// Named flatten_detail, not the more generic "detail" - a bare "detail"
// here previously collided with compensated_float.h's own unrelated
// namespace detail (ambiguous unqualified lookup, MSVC error C2872) in any
// translation unit that both instantiates InnerProduct() (compensated_float.h,
// via e.g. square_matrix.h) and does `using namespace pbrt_flatten;` (this
// header's own established convention for callers, see flatten()'s local
// using-directive below) - exactly tests/unit/pbrt_flatten_tests.cpp's shape.
namespace flatten_detail {

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
	if (type == "none" || type.empty()) return MaterialKind::Interface;
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
