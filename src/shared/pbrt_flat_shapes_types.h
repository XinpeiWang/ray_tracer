#pragma once
// pbrt_flat_shapes_types.h -- the world-space shapes pbrt_flatten.h produces (Triangle, Sphere, Disk, Cylinder, Cone, Paraboloid, BilinearPatch, Curve and their animated
// forms) and the Medium. Plain data; part of pbrt_flatten.h, which includes it.

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


}  // namespace pbrt_flatten
