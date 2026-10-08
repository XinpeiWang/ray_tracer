#pragma once
// optix_types_lights_camera.h -- part 4 of 5 of optix_types.h (included by it, in order; not meant to be included on its own).

// GPU-side goniometric (IES-profile) point light. Mirrors the *evaluation*
// half of src/shared/goniometric_light.h's GoniometricLight<T> (sample_li +
// eval_I) - the CPU-only sample_le/pdf_le (light-tracing/BDPT) are not
// needed here since GPU rendering is NEE-only. image[] holds the same
// row-major nu*nv equal-area-square greyscale data the CPU struct owns as a
// std::vector<double>; the CPU is the source of truth for the pattern, this
// just holds a device-uploadable copy of it.
struct GoniometricLightGPU {
	float pos_x, pos_y, pos_z;
	float world_to_light[9];  // row-major 3x3 world->light rotation
	float ir, ig, ib;         // base intensity (candela)
	float scale;
	int nu, nv;                // nu*nv must be <= kGonioImageMaxDim^2
	float image[kGonioImageMaxDim * kGonioImageMaxDim];  // [v*nu+u], greyscale
};

// GPU-side projection (slide-projector) light. Mirrors the evaluation half
// of src/shared/projection_light.h's ProjectionLight<T> (sample_li +
// eval_I_rgb). `inv_tan` replaces the CPU struct's full 4x4 screenFromLight/
// lightFromScreen perspective matrices: make_perspective()'s matrix reduces
// exactly to screen_x = inv_tan*lx/lz, screen_y = inv_tan*ly/lz for a point
// (lz is already the homogeneous w after transform, per its [0,0,1,0] bottom
// row), so storing the one scalar 1/tan(fov/2) is sufficient and avoids
// needing Mat4/matrix-multiply plumbing on the device.
struct ProjectionLightGPU {
	float pos_x, pos_y, pos_z;
	float world_to_light[9];  // row-major 3x3 world->light rotation
	float scale;
	float hither;              // near-plane cutoff (pbrt-v4 default 1e-3)
	int nx, ny;                 // nx*ny must be <= kProjImageMaxDim^2
	float sb_xmin, sb_xmax, sb_ymin, sb_ymax;  // screen bounds
	float inv_tan;              // 1 / tan(fov_deg/2 in radians)
	float image_rgb[kProjImageMaxDim * kProjImageMaxDim * 3];  // [(v*nx+u)*3+c]
};

// A single punctual light, tagged by kind. Only the member matching `kind`
// is meaningful; the others are left default-constructed. Kept as inline
// structs rather than a union since light counts are tiny (typically 1-3
// per scene) and the member types are already CPU_GPU-tagged (or, for the
// two image-based kinds, trivially-copyable) PODs - no union/variant
// plumbing needed for this scale.
struct PunctualLightGPU {
	PunctualLightKind kind;
	PointLightData<float> point;
	SpotLightData<float> spot;
	DistantLightData<float> distant;
	GoniometricLightGPU gonio;
	ProjectionLightGPU proj;
};

// Alias table entry for power-weighted light sampling (pbrt-v4 PowerLightSampler pattern)
// Stored in GPU memory; sampled in O(1) by the device code.
struct GpuAliasEntry {
	float  q;      // Acceptance probability in [0,1]
	int    alias;  // Fallback index if rejected
	float  pdf;    // Probability mass for this entry (= power_i / total_power)
};

// Which ray-generation formula GpuCameraParams describes. Mirrors the CPU
// camera models this GPU camera type supports (src/shared/cameras.h's
// OrthographicCamera/PerspectiveCamera/SphericalCamera/RealisticCamera, plus
// the book-style default camera's defocus_angle/focus_dist thin-lens DOF
// extension).
enum class CameraKind : int {
	Perspective = 0,   // pinhole, optionally with thin-lens DOF (defocus_disk_u/v)
	Orthographic = 1,  // parallel projection, constant ray direction `w`
	Spherical = 2,     // 360-degree equirectangular panorama from a point
	Realistic = 3      // multi-element lens (pbrt-v4 RealisticCamera) - see GpuLensElement
};

// One spherical (or planar, for the aperture stop) lens surface - mirrors
// src/shared/cameras.h's RealisticCamera<T>::LensElement, already in metres
// and with the rear element's thickness already focus-adjusted (both done
// host-side by directly reusing RealisticCamera<float>'s own constructor -
// see scene_builder.cpp's case 36 - so device code never needs to run
// FocusThickLens/ComputeCardinalPoints itself).
struct GpuLensElement {
	float curvatureRadius;  // 0 = aperture stop
	float thickness;
	float eta;               // 0 = no interface (air on both sides don't apply eta)
	float apertureRadius;
};

// One radial exit-pupil bounding box slab - mirrors RealisticCamera<T>::Bounds2,
// precomputed host-side (RealisticCamera<float>::bound_exit_pupil, run once at
// scene-build time, same cost class as building a BVH or alias table).
struct GpuExitPupilBounds {
	float xMin, xMax, yMin, yMax;
	int   degenerate;  // 1 = no valid exit-pupil sample in this radial slab
};

// GPU flat mirror of sky_light's REAL (image + PiecewiseConstant2D)
// importance-sampling machinery (src/TheRestOfYourLife/sky_light.h +
// src/shared/piecewise_dist.h), built once per scene CPU-side (from the SAME
// decoded infinite-light image the flat-colour mean-brightness approximation
// already used - see pbrt_gpu_builder.h) and uploaded as flat device
// buffers, mirroring the measured-BRDF GPU work's "flatten CPU-built tables,
// upload flat buffers, port only the read-path math to device" pattern (see
// optix_measured_bxdf.h's own comment) - just simpler, since a scene has at
// most ONE infinite light (no per-material indexing/dedup needed the way
// GpuMeasuredTable/GpuBssrdfTable require).
//
// height <= 0 (the zero-init default) means "this scene has no real HDR
// image" (a constant-colour sky, or no infinite light at all) - every call
// site (gpu/optix/optix_sky_light.h / wavefront_sky_light.h and their
// callers) falls back to the existing flat-colour + uniform-sphere path in
// that case, unchanged from before this struct existed.
//
// Embedded directly in GpuCameraParams (not a separate LaunchParams array,
// unlike GpuMeasuredTable/GpuBssrdfTable) so it rides along on the SAME
// pass-through mechanism backgroundColor/shadowRayEpsilon already use to
// reach both GPU backends: recursive reads it via params.camera.skyDist,
// wavefront's kernels receive it as an explicit by-value parameter (same
// established pattern as GpuCameraParams itself already being passed by
// value into generate_camera_rays) extracted from the `camera` argument
// WavefrontPathTracer::render() already receives - see that function's own
// call sites. The raw device pointers below are only valid AFTER
// OptiXRenderer::buildScene() has uploaded them (mirrors CameraKind::
// Realistic's lensElements/exitPupilBounds pointers just below, which follow
// the identical "logical camera params built early, raw uploaded-table
// pointers patched in fresh inside render()" lifecycle).
//
// Deliberately NO in-class member initializers (unlike some other structs in
// this codebase): LaunchParams's own `__constant__ params` global (optix_
// device_helpers.h) requires every type it contains support trivial, non-
// dynamic initialization - nvcc rejects in-class initializers there with
// "dynamic initialization is not supported for a __constant__ variable".
// Every host-side use already zero-initializes the containing struct
// explicitly (`LaunchParams params = {};` / GpuCameraParams's own "always
// zero-initialized" convention - see its own comment), so height/scale/every
// pointer here are reliably 0/0.0f/nullptr by construction anyway.
struct GpuSkyDistribution {
	int width, height;              // image dimensions (nu, nv)
	float scale;                    // sky_light::scale (brightness multiplier)
	const float* imagePixels;       // width*height*3 floats, row-major RGB, linear
	const float* marginalCdf;       // height+1 floats
	const float* marginalFunc;      // height floats
	float marginalFuncInt;
	// Row r's slice starts at r*(width+1) (cdf) / r*width (func) - regular
	// stride since every row has the same width, unlike GpuPL2DTable's
	// per-axis offsets (which vary because different sub-tables have
	// different axis counts).
	const float* conditionalCdf;    // height*(width+1) floats
	const float* conditionalFunc;   // height*width floats
	const float* conditionalFuncInt; // height floats, one per row
};

// pbrt-v4 "portal" (windowed) infinite light - GPU-side flat-buffer port of
// src/shared/portal_image_infinite_light.h's PortalImageInfiniteLightData<T>.
// Mutually exclusive with GpuSkyDistribution above (matches CPU: a scene's
// single infinite light is either a plain image/flat-colour sky OR a portal
// one, never both - src/TheRestOfYourLife/pbrt_cpu_builder.h's own
// BuildResult::portal/sky comment) - height<=0 here (the zero-init default)
// means "this scene has no portal light", same convention as
// GpuSkyDistribution::height. When BOTH are height<=0, every call site falls
// back to the existing flat-colour + uniform-sphere path unchanged.
//
// Unlike GpuSkyDistribution, this struct's queries (eval_Le/sample_li/
// pdf_li - gpu_portal_light_shared.h) depend on the SHADING POINT, not just
// a direction: the portal quad's visible angular window changes with
// viewpoint (ImageBounds() in the CPU class). Every caller (sample_sky_nee/
// sky_radiance/sky_pdf_for_mis and their wf_ twins) therefore takes an
// additional shading-point parameter now, threaded through from whichever
// hit/miss point is already in scope at each call site.
//
// The three flat buffers below are NOT rebuilt on GPU - pbrt_gpu_builder.h
// constructs a real, host-side PortalImageInfiniteLightData<double> (the
// exact same class CPU uses) and uploads its already-computed rectified
// image / distribution values / summed-area-table prefix sums verbatim, so
// the rectified image's content (including the CPU class's own equal-area-
// vs-equirectangular quirk, see that class's ctor comment) matches CPU
// bit-for-bit rather than risking a second, independently-reimplemented
// equal-area rectification pass here.
//
// Deliberately NO in-class member initializers - same __constant__-global
// constraint as GpuSkyDistribution above (see that struct's own comment).
struct GpuPortalLight {
	int width, height;   // rectified image dimensions; height<=0 = "no portal light"
	float scale;
	// Orthonormal portal-quad frame (PortalImageInfiniteLightData::
	// portalFrame_'s FromXY(p03,p01) result) - built once, host-side, in
	// pbrt_gpu_builder.h; ImageFromRender()/RenderFromImage_uv() (gpu_
	// portal_light_shared.h) project through this frame every query.
	float3 frameX, frameY, frameZ;
	// portal[0] and portal[2] (diagonally opposite corners) - the only two
	// of the CPU class's 4 stored corners ImageBounds() actually reads at
	// query time; portal[1]/[3] only ever fed the frame construction above,
	// already baked into frameX/Y/Z, so they need no GPU-side copy.
	float3 p0, p2;
	const float*  rectifiedImage;  // width*height*3 floats, row-major RGB
	const float*  distFunc;        // width*height floats, raw per-cell distribution values
	const double* satSum;          // width*height doubles, summed-area-table prefix sums
};

// GPU camera parameters. `origin`/`lower_left_corner`/`horizontal`/`vertical`
// describe the perspective/orthographic viewport (same meaning as before this
// struct existed); the remaining fields are only meaningful for their
// respective CameraKind and are zeroed otherwise (scene_builder.cpp always
// zero-initializes this struct, so "zero defocus disk" reliably means
// "DOF disabled" for Perspective).
struct GpuCameraParams {
	CameraKind kind;
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 w;               // Orthographic: constant unit ray direction
	float3 defocus_disk_u;  // Perspective DOF: disk basis vector (zero = disabled)
	float3 defocus_disk_v;
	float3 su, sv, sw;      // Spherical: world-space camera basis (right, up, forward);
	                        // also reused by Realistic for its camera-to-world rotation
	                        // (su=right, sv=up, sw=forward) - `origin` above doubles as
	                        // its camera-to-world translation.
	int    sphericalMapping; // Spherical only: 0 = equirectangular (default, matches every
	                        // zero-initialized scene before this field existed), 1 = equal-
	                        // area (pbrt-v4's concentric-octahedral square-to-sphere map,
	                        // src/shared/sampling_sphere.h's EqualAreaSquareToSphere on CPU).

	// Realistic (CameraKind::Realistic): fixed scalars + device buffers for the
	// host-precomputed (focus-adjusted) lens table and exit-pupil bounds table -
	// see GpuLensElement/GpuExitPupilBounds. Null/zero for every other CameraKind.
	float film_half_x, film_half_y;  // physical film half-extents, metres
	float lens_rear_z;               // distance from film to rear lens element, metres
	int   numLensElements;
	int   numExitPupilBounds;
	GpuLensElement*     lensElements;
	GpuExitPupilBounds* exitPupilBounds;

	// Flat constant-color background for missed rays (default black =
	// existing behavior for every scene that doesn't set it). Piggybacked
	// onto this struct (rather than a new render()/LaunchParams field of
	// its own) since it's already threaded through the exact same call
	// chain this struct is. See optix_miss.h for why a flat color - not
	// full image-based env lighting - matches what the CPU renderer
	// actually does for every scene (scenes_advanced.h's build_hdri_sky()/
	// build_portal_sky() both return a solid-color sky_light; the
	// importance-sampled-image machinery in image_infinite_light.h is
	// unused dead code on the CPU side too, never wired to any scene).
	float3 backgroundColor;

	// Shadow-ray self-intersection offset override (world units), along the
	// shadow ray's own direction - see trace_shadow_ray()'s (optix_device_
	// helpers.h) own comment for why this offset exists at all. <= 0 (the
	// default, zero-init-safe for every scene that doesn't set it) means
	// "use the standard 0.01f". Nothing sets it any more: the native Sibenik
	// scene raised it to 0.5f because its GPU render was ~24% of the CPU's,
	// but that gap came from the CPU's nearest-neighbour bump lookup
	// brightening the walls, not from false shadow occlusion - with the bump
	// removed the GPU at 0.01f is within ~7% of the CPU, and 0.5f just leaks
	// light (~160%). Kept as a per-launch override hook.
	float shadowRayEpsilon;

	// Real importance-sampled HDR sky (LightSource "infinite" with an image) -
	// see GpuSkyDistribution's own comment. height<=0 (default) means "use
	// backgroundColor + uniform-sphere sampling", exactly as before this
	// field existed - every scene with a constant-colour sky, or no infinite
	// light at all, is completely unaffected.
	GpuSkyDistribution skyDist;

	// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's own
	// comment. Mutually exclusive with skyDist above (matches CPU); height<=0
	// (default) means "no portal light", same convention as skyDist.height.
	GpuPortalLight portalLight;

	// Pixel reconstruction filter (pbrt-v4 PixelFilter directive) - the GPU
	// twin of src/shared/filter.h's PixelFilterDispatch (CPU); gpu_filter_
	// evaluate() (optix_device_helpers.h) is the device-side port of its
	// evaluate(). filterKind 0 (the zero-init default, matching every scene
	// that never set one, hand-built or pbrt-loaded) = Gaussian, pbrt-v4's
	// own real default - see that enum's own values just below. NO in-class
	// member initializers here (unlike the rest of this struct's own
	// convention elsewhere) - `LaunchParams params` is declared `__constant__`
	// (optix_device_helpers.h), which nvcc requires to be static/zero-
	// initializable; a non-trivial default constructor (which member
	// initializers would introduce) breaks that with "dynamic initialization
	// is not supported for a __constant__ variable". filterSigma/filterTau
	// default to 0.0f via plain zero-init instead, which would be a
	// degenerate Gaussian - gpu_filter_evaluate() defends against exactly
	// this (sigma<=0 -> 0.5f, tau<=0 -> 3.0f) rather than relying on this
	// struct's own initialization.
	//
	// Reconstruction filter reach: filterRadius is the filter's real support
	// radius in pixels (<=0, the zero-init default, means "pbrt-v4's own
	// default for this kind": gaussian 1.5, box 0.5, triangle 2, mitchell 2,
	// sinc 4 - the same defaults CPU's PixelFilterDispatch/camera::filter_
	// radius use, so a native scene with no explicit PixelFilter matches
	// CPU's Gaussian-1.5 with no scene-side wiring). filterSampler is a
	// device pointer to the SAME FilterSampler<float,32> table CPU builds
	// (src/shared/filter_sampler.h, pbrt-v4's tabulated importance sampling),
	// built from those parameters and uploaded by OptiXRenderer::render() -
	// like lensElements/skyDist it can't be known at scene-build time. Each
	// camera sample CDF-inverts its (Halton) [0,1)^2 draw through it into a
	// sub-pixel position that can land OUTSIDE the pixel for a filter wider
	// than one pixel (reaching the neighbouring pixels' area, exactly like
	// CPU's camera.h get_ray() offset), weighted by FilterSample::weight.
	// nullptr (a caller that never went through render()) keeps the old
	// within-pixel, gpu_filter_evaluate()-weighted fallback.
	int   filterKind;   // 0=gaussian 1=box 2=triangle 3=mitchell 4=sinc
	float filterB, filterC, filterSigma, filterTau;
	float filterRadius;
	const FilterSampler<float, 32>* filterSampler;

	// pbrt-v4 Integrator "bool regularize" - defaults false via zero-init
	// (matching pbrt-v4's own real default), same "no in-class initializer"
	// convention as filterKind above and for the identical reason
	// (__constant__ zero-initialization). Gates whether h.any_nonspecular
	// (wavefront_types.h) actually widens a rough BSDF's GGX alpha
	// (RegularizeAlpha(), src/shared/microfacet.h) - any_nonspecular's own
	// path-history TRACKING stays unconditional either way (matches
	// pbrt-v4's anyNonSpecularBounces, which is also tracked regardless of
	// the regularize flag's value - only the widening at USE time is
	// gated), so this is read only at wf_glossy_alpha()/RegularizeAlpha()
	// call sites, never at the any_nonspecular propagation site itself.
	int regularize;

	// Film "cropwindow"/"pixelbounds" (pbrt-v4), resolved to PIXEL bounds
	// [cropX0,cropX1) x [cropY0,cropY1) at scene-build time (scene_builder.cpp,
	// where the render's real width/height are already known - unlike CPU's
	// camera class, which resolves the same NDC-fraction FlatScene::cropX0..Y1
	// fields lazily in its own initialize(), GPU has no equivalent lazy
	// resolution step, so this struct always carries the final, concrete
	// pixel bounds, never the raw fractions). cropX1<=0 (the zero-init
	// default - same "no in-class initializer" __constant__ constraint as
	// filterKind/regularize above) means "no crop requested" - gpu_in_crop()
	// (optix_device_helpers.h) treats that as "every pixel is in bounds",
	// exactly matching every scene (hand-built or pbrt-loaded-without-a-
	// cropwindow) that never sets these fields at all.
	int cropX0, cropX1, cropY0, cropY1;

	// --seed (CLI/GUI) - an explicit request for reproducible renders. Set
	// to -1 (NOT the zero-init default - explicitly assigned right after
	// construction in optix_interface.cpp, same "no in-class initializer"
	// __constant__ constraint as every other field in this struct) when no
	// --seed was requested, since unlike regularize/cropX1 above, 0 is
	// itself a perfectly valid real seed value and can't double as "unset"
	// the way their own zero-init defaults do. Recursive backend: read
	// directly into LaunchParams::frameNumber (optix_renderer_render.cpp).
	// Wavefront backend: read into WavefrontPathTracer::frameNumber_ at
	// the start of render() (wavefront_path_tracer.cpp), overriding
	// whatever that self-incrementing counter was left at by a PRIOR
	// render() call on the same long-lived WavefrontPathTracer instance -
	// otherwise a second render in the same process would silently use a
	// different, no-longer-user-chosen seed than the first.
	int userSeed;

	// "float maxcomponentvalue" (pbrt-v4 Film parameter) / --maxcomponentvalue
	// (CLI) - CPU's own per-sample firefly clamp (camera::max_component_value,
	// camera.h's own comment) is now real on GPU too. 1e9f (pbrt-v4's real
	// "effectively unbounded" default, matching camera::max_component_value's
	// own C++ default exactly - NOT the zero-init default, which would clamp
	// every sample to black - explicitly assigned in optix_interface.cpp
	// after GpuCameraParams cameraExtra{}, same "no in-class initializer"
	// __constant__ constraint as every other field here) means "no clamp
	// requested", so a render that never touches this behaves exactly as
	// before this field existed. Recursive backend (optix_raygen.h): a real,
	// exact clamp on the WHOLE sample's radiance right before it's
	// filter-weighted into pixel_color, byte-for-byte matching CPU's own
	// clamp_sensor_rgb() call site and pbrt-v4's true per-sample semantics -
	// this backend accumulates one sample's full radiance in a single local
	// before ever touching the framebuffer, so a true per-sample clamp is a
	// direct, exact port. Wavefront backend (wavefront_kernels.cu): only an
	// APPROXIMATION of the same feature, not a true port - this backend's
	// framebuffer is a single running total shared across every sample AND
	// bounce of the whole render, fed by 7 separate atomicAdd call sites
	// (NEE shadow hits, direct emission, escaped/miss rays, BSSRDF exit,
	// deferred shadow-ray accumulation) with no single point where "this
	// one sample's total radiance" is ever known as one value - clamping the
	// true per-sample TOTAL would need deferring every one of those adds
	// into a per-ray accumulator until definitive path termination, a
	// materially bigger change than this round's scope (the same class of
	// tradeoff this codebase already discloses for GPU's area-light texture
	// filtering - see docs/PBRT_SUPPORT.md). Instead, each of those 7 sites
	// clamps its own individual contribution independently before adding -
	// real firefly suppression, but NOT pbrt-v4's exact semantics (a sample
	// whose total exceeds the threshold via several individually-under-
	// threshold contributions isn't caught this way). Disclosed as "Approx"
	// in docs/PBRT_SUPPORT.md, not silently presented as CPU/recursive parity.
	//
	// Every clamp call site (both backends) additionally guards on
	// `maxComponentValue > 0.0f` before comparing, not just `m >
	// maxComponentValue` - a test-only or native-scene `GpuCameraParams`
	// built without knowing about this field (this struct has no in-class
	// initializer) leaves it as uninitialized garbage, and a negative
	// garbage value would otherwise clamp with a negative scale factor,
	// producing negative/NaN pixel output (a real regression a first
	// version of this feature caused in 3 disk/cylinder GPU render tests,
	// found and fixed via the test suite before this shipped). Matches
	// this same struct's own "<=0 means not requested" convention already
	// used by cropX1/cameraMediumSigmaT above - this field's own C++
	// default just happens to be the wrong number (0.0f, not 1e9f) for
	// this same trick to fall out for free, so the guard is explicit here
	// instead.
	float maxComponentValue;

	// Camera motion blur (pbrt-v4 real per-ray AnimatedTransform-based
	// shutter-time interpolation - mirrors src/TheRestOfYourLife/camera.h's
	// camera_is_animated path). 0 (zero-init default, matching every scene
	// before this feature existed) = static camera, every field below
	// unused. When nonzero (Perspective only - see generate_primary_ray()'s
	// own comment), a per-sample shutter fraction dt in [0,1] interpolates
	// animT0/R0 -> animT1/R1 (the two camera-to-world keyframes, already
	// decomposed into translate+rotate ONCE, host-side, by the real
	// double-precision AnimatedTransform class - see
	// build_gpu_animated_camera_params(), scene_builder.cpp) and applies
	// the result to a ray built from the LOCAL (canonical-axis, world-
	// placement-independent) viewport geometry below, exactly mirroring
	// camera.h's local_pixel00_loc/local_pixel_delta_u/v/local_defocus_disk_u/v.
	// No scale keyframes: camera-to-world matrices here are always built
	// from an orthonormal lookat basis (unit-length u/v/w via normalize()/
	// cross()), so unlike the general AnimatedTransform there is never a
	// scale term to interpolate.
	int    animated;
	float3 animT0, animT1;
	float4 animR0, animR1;  // quaternion (x,y,z,w), shortest-arc-aligned
	float3 localLowerLeftCorner, localHorizontal, localVertical;
	float3 localDefocusDiskU, localDefocusDiskV;  // zero = no DOF

	// Object (per-primitive sphere) motion blur - the SINGLE canonical flag
	// for both GPU backends, not a per-backend copy: the recursive backend's
	// own raygen (optix_raygen.h) reads it as `params.camera.motionBlurEnabled`
	// (LaunchParams embeds this struct as `camera`), and wavefront kernels
	// read it directly off this struct, which is already threaded as an
	// explicit parameter through every kernel that needs camera state (see
	// e.g. GpuPortalLight's own comment on this file's piggyback convention) -
	// wavefront kernels are plain CUDA kernels, never given a `LaunchParams`
	// global, so they have no other way to reach it. An earlier version of
	// this field had a second, independent LaunchParams::motionBlurEnabled
	// twin that both backends set separately from the same sceneHasMotion_
	// source at two different call sites in OptiXRenderer::render() - a real
	// divergence risk (a future edit to one assignment, forgetting the
	// other, would let the two backends silently disagree). Consolidated
	// into this one field instead: set once, from OptiXRenderer::
	// sceneHasMotion_ auto-detection (SphereData::center1 != center for at
	// least one sphere), and both backends read the exact same value off the
	// exact same struct - structurally incapable of drifting apart. 0
	// (zero-init default, matching every scene before this feature existed)
	// = static spheres, generate_camera_rays() always samples ray.time = 0
	// and optix_raygen.h always samples ray_time = 0.0f.
	int motionBlurEnabled;

	// pbrt-v4's own "camera medium" (unbounded ambient fog/haze the camera
	// itself starts inside - MediumInterface declared before the Camera
	// directive; see FlatScene::cameraMediumIndex's own comment, pbrt_
	// flatten.h, and CPU's ambient_medium, src/TheRestOfYourLife/constant_
	// medium.h, for the full feature writeup, including why this is an
	// explicit per-bounce step rather than one more scene-BVH entry).
	// cameraMediumSigmaT<=0 (the zero-init default, same "no in-class
	// initializer" __constant__ constraint as every other optional field on
	// this struct) means "no camera medium" - every scene that doesn't
	// request this is completely unaffected. Scalar sigma_t (not per-RGB-
	// channel) matches CPU's own camera-medium construction exactly - CPU's
	// pbrt_cpu_builder.h collapses to a scalar sigma_a/sigma_s via
	// `collapse_homogeneous_medium()` before ever reaching ambient_medium's
	// constructor, so CPU's own per-channel HomogeneousMediumData ends up
	// with all three channels identical too; the real per-channel color
	// lives entirely in cameraMediumAlbedo. cameraMediumEmission is already
	// weighted by sigma_a/sigma_t (MakeNamedMedium's own "rgb Le"), matching
	// CPU's hg_phase_material::emitted() exactly - not re-weighted per
	// launch. Living on GpuCameraParams (not a bare LaunchParams field)
	// means the wavefront backend gets this for free through its existing
	// GpuCameraParams argument - see motionBlurEnabled's own comment just
	// above for why that's the established pattern for scene state either
	// backend's kernels need.
	float  cameraMediumSigmaT;
	float3 cameraMediumAlbedo;
	float  cameraMediumG;
	float3 cameraMediumEmission;
	// Wavefront only: index into LaunchParams::materials of the synthetic Medium material standing for the camera
	// medium (SceneData::cameraMediumMaterialIdx); meaningful only while cameraMediumSigmaT > 0.
	int    cameraMediumMaterialIdx;
	// A camera medium whose extinction differs between colour channels (CameraMediumGpu::chromatic): the raw per-channel coefficients,
	// sampled with sample_homogeneous_event like a per-shape chromatic medium (MaterialData::chromaSigmaA). All zero otherwise - the
	// scalar fields above then apply unchanged. cameraMediumSigmaT stays set either way (it gates "there is a camera medium").
	float3 cameraMediumSigmaA;
	float3 cameraMediumSigmaS;
	float3 cameraMediumLeRaw;
};
