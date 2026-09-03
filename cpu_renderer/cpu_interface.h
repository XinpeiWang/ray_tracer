/// @file cpu_interface.h
/// @brief CPU Renderer C Interface
/// @details This header defines the C-linkage interface for the CPU-based ray tracer.
/// The CPU renderer uses:
///   - Multithreaded C++ path tracing
///   - Importance sampling (PDF-based lighting, NEE + MIS)
///   - Any scene registered in src/TheRestOfYourLife/scene_registry.h (123
///     built-in scenes at last count), selected by scene_id
///
/// Camera System:
///   - Camera position (lookfrom) is configurable via cam_x, cam_y, cam_z
///   - Camera target (lookat), up vector, and field of view all come from
///     the selected scene's own CameraConfig (scene_registry.h) - there is
///     no single fixed camera target across scenes
///
/// This interface allows the launcher (launcher/main.cpp) to call the CPU
/// renderer as an in-process library function with C linkage (no name
/// mangling).

#ifndef CPU_INTERFACE_H
#define CPU_INTERFACE_H

#include "../src/shared/render_options.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Render the requested scene using CPU path tracing with importance sampling
/// @details Any scene registered in src/TheRestOfYourLife/scene_registry.h,
/// selected by scene_id (a category letter + number, e.g. "A1" for Cornell
/// Box). Scene geometry/materials/lights are built by scene_registry.h's
/// build_world()/build_lights() closures for the selected entry.
///
/// Camera configuration:
///   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position (or the
///     scene's own default, see force_camera_override below)
///   - lookat/vup/vfov: come from the selected scene's own CameraConfig
///     (scene_registry.h) - not fixed across scenes
/// 
/// @param width         Image width in pixels (height = width for square aspect)
/// @param height        Image height in pixels
/// @param spp           Samples per pixel (higher = less noise, slower render)
/// @param max_depth     Maximum ray bounce depth (higher = more realistic lighting)
/// @param output_path   Output PPM file path (e.g., "C:/path/to/image.ppm")
/// @param scene_id      Scene selector, category letter + number within
///                      category (e.g. "A1"=Cornell Box, "A2"=Bouncing
///                      Spheres - see src/TheRestOfYourLife/scene_registry.h)
/// @param cam_x         Camera position X coordinate (default: 278)
/// @param cam_y         Camera position Y coordinate (default: 278)
/// @param cam_z         Camera position Z coordinate (default: -800)
/// @param force_camera_override
///                      1 = always use cam_x/y/z, even for scenes whose
///                      CameraConfig.mode isn't UserControlled (which
///                      otherwise silently ignore cam_x/y/z and use the
///                      scene's own fixed lookfrom - see cpu_interface.cpp).
///                      main.cpp passes 1 unconditionally for BOTH
///                      single-image and video-mode renders (see its own
///                      comment on why) - the 0 default here only matters
///                      to direct callers that bypass main.cpp entirely
///                      (e.g. unit tests).
/// @param options       Render-behavior flags (exposure/sampler/spectral/
///                      tonemap) bundled into one struct rather than
///                      individual trailing parameters - see
///                      src/shared/render_options.h's own field-by-field
///                      comments for exact semantics/defaults/fallback
///                      behavior. A default-constructed RenderOptions
///                      reproduces this project's pre-existing hardcoded
///                      behavior exactly (sobol sampler, flat RGB, ACES
///                      tone mapping, no denoise).
/// @return 0 on success, non-zero error code on failure
int cpu_render_main(
    int width,
    int height,
    int spp,
    int max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0,
    const RenderOptions& options = {}
);

/// @brief Render a scene using Stochastic Progressive Photon Mapping (SPPM)
/// instead of the default unidirectional path tracer.
/// @details A separate entry point (not a mode flag on cpu_render_main)
/// deliberately - SPPM is a fundamentally different render loop (iterative
/// camera-pass + photon-pass + progressive-radius-contraction, not a
/// one-shot per-pixel Monte Carlo sample loop), so keeping it as its own
/// function leaves cpu_render_main's existing ABI/behavior completely
/// untouched. Intended for scenes with hard-to-converge specular/caustic
/// transport (e.g. glass spheres) that the path tracer either can't
/// converge on in reasonable time or renders very noisily - see
/// src/TheRestOfYourLife/sppm_adapter.h's file comment for the underlying
/// investigation. Works against the same scene_registry build_world()/
/// build_lights() closures as cpu_render_main - any scene id is technically
/// valid, but this is currently only verified end-to-end against scene 11
/// (Cornell Rough Glass); other scenes may render correctly but are
/// unverified.
///
/// Supports lambertian (non-specular) materials for indirect/caustic light
/// transport, the 8 delta-BSDF material classes (metal, dielectric,
/// rough_metal, rough_dielectric, conductor, coated_diffuse,
/// thin_dielectric, coated_conductor), and diffuse_transmission/
/// normalized_fresnel/mix_material (each individually handled by the BSDF
/// bridge - see src/TheRestOfYourLife/bsdf_bridge.h's sppm_bsdf_f() comment
/// for exactly how).
///
/// @param width          Image width in pixels (height = width for square aspect)
/// @param height         Image height in pixels
/// @param iterations     Number of SPPM iterations (more = less noise, slower)
/// @param photons        Photons shot per iteration
/// @param max_depth      Maximum ray/photon path depth
/// @param output_path    Output PPM file path
/// @param scene_id       Scene selector (see scene_registry.h)
/// @param cam_x/y/z      Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_sppm(
    int width,
    int height,
    int iterations,
    int photons,
    int max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using Bidirectional Path Tracing (BDPT).
/// @details Implemented in cpu_renderer/cpu_interface_bdpt.cpp -- a
/// SEPARATE translation unit from cpu_interface.cpp (which hosts
/// cpu_render_main/cpu_render_main_sppm above), not for architectural
/// symmetry but to avoid a genuine ODR conflict: cpu_interface.cpp already
/// includes sppm_adapter.h, which includes
/// src/TheRestOfYourLife/power_light_sampler.h (a global `class AliasTable`),
/// while BDPT/MLT need src/shared/mlt.h, which includes
/// src/shared/reservoir_sampler.h (a DIFFERENT global `class AliasTable`).
/// Both are legitimate, independently-developed ports of the same pbrt-v4
/// concept for different subsystems; neither can be renamed away for free,
/// so they simply can never appear in the same translation unit. See
/// src/TheRestOfYourLife/bdpt_adapter.h's own file comment for the full
/// story (also the reason SPPMSceneAdapter's BSDF bridge was extracted into
/// bsdf_bridge.h -- reused by both this file's adapter and SPPM's without
/// re-pulling power_light_sampler.h).
///
/// Uses BDPTSceneAdapter (src/TheRestOfYourLife/bdpt_adapter.h) against the
/// same scene_registry build_world()/build_lights() closures
/// cpu_render_main()/cpu_render_main_sppm() use. CPU-only: bidirectional
/// path tracing on the GPU (OptiX) is out of scope for this integration,
/// comparable in scope to the GPU SPPM work and left for a future,
/// dedicated project -- see launcher_args.h's --bdpt/--gpu handling.
///
/// Area, point/spot/distant, AND sky/infinite lights are all supported via
/// one unified power-weighted light distribution -- see bdpt_adapter.h's own
/// "Scope (v2)" comment. Goniometric and projection lights still have no
/// light-subpath-emission (SampleLe) implementation, so a scene whose ONLY
/// light is one of those still renders black. Verified end-to-end against
/// scene A1 (Cornell Box) only; other scenes are unverified.
///
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel (one full BDPTLi() estimate per sample)
/// @param bdpt_max_depth  Maximum BDPT path depth (camera+light vertices combined)
/// @param output_path     Output PPM file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_bdpt(
    int width,
    int height,
    int spp,
    int bdpt_max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using Metropolis Light Transport (MLT).
/// @details Same translation-unit split rationale, same BDPTSceneAdapter
/// reuse (MLT is built directly on top of BDPT's subpath machinery -- see
/// src/shared/mlt.h's own file comment), and the same v1 scope (area lights
/// only, scene A1 the only end-to-end-verified scene) as
/// cpu_render_main_bdpt() above -- see that function's doc comment and
/// bdpt_adapter.h's file comment for the full reasoning, not repeated here.
///
/// Multiple independent Markov chains run in parallel across worker threads
/// (see src/TheRestOfYourLife/bdpt_adapter.h's mlt_render_with_adapter() and
/// src/shared/mlt.h's chainSeed parameter) -- mlt.h's own MLTRenderLoop() is
/// otherwise a single-chain driver.
///
/// @param width/height     Image dimensions
/// @param mlt_bootstrap    Bootstrap samples per depth (mlt.h's nBootstrap)
/// @param mlt_mutations    Total Metropolis mutations across ALL chains combined
/// @param mlt_max_depth    Maximum BDPT path depth used by each MLT sample
/// @param output_path      Output PPM file path
/// @param scene_id         Scene selector (see scene_registry.h)
/// @param cam_x/y/z        Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_mlt(
    int width,
    int height,
    int mlt_bootstrap,
    long long mlt_mutations,
    int mlt_max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using RandomWalkIntegrator -- pbrt-v4's unbiased
/// reference path tracer (uniform-sphere sampling at every bounce, no NEE,
/// no MIS, no importance sampling at all). Slow to converge and noisy by
/// design -- a correctness reference for other integrators, not a
/// production render mode. See src/shared/utility_integrators.h's own file
/// comment. CPU only. Works on any scene (no light-sampling dependency,
/// unlike --bdpt/--mlt/--lightpath -- an emissive surface only contributes
/// when a bounce happens to hit it directly). A bounce that escapes the
/// scene entirely shows the real cam.sky environment map when the scene
/// defines one (matching what every other integrator's own miss handling
/// shows), not a forced flat background - punctual (point/spot/distant)
/// lights remain invisible here regardless, since a uniform-sphere bounce
/// can never hit a light with no real geometry.
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel
/// @param max_depth       Maximum bounce depth
/// @param output_path     Output PPM/EXR file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_randomwalk(
    int width,
    int height,
    int spp,
    int max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using AOIntegrator -- ambient occlusion only (a
/// single hemisphere shadow ray per camera-ray hit, no indirect lighting,
/// no material colors). A debugging/visualization tool (occlusion mask),
/// not a lit render. See src/shared/utility_integrators.h's own file
/// comment. CPU only.
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel
/// @param max_dist        Occlusion test distance (pbrt-v4 "maxdistance";
///                        a very large value, e.g. 1e10, means unbounded)
/// @param cos_sample      1 = cosine-hemisphere sampling, 0 = uniform-hemisphere
/// @param illum_scale     Flat multiplier on the occlusion color
/// @param illum_r/g/b     Occlusion color (pbrt-v4 "illuminant"; default white)
/// @param output_path     Output PPM/EXR file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_ao(
    int width,
    int height,
    int spp,
    double max_dist,
    int cos_sample,
    double illum_scale,
    double illum_r,
    double illum_g,
    double illum_b,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using SimplePathIntegrator -- pbrt-v4's canonical
/// reference path tracer with optional NEE and optional BSDF importance
/// sampling. See src/shared/simple_path.h's own file comment. CPU only.
/// NEE samples area + point/spot/distant + sky lights (same unified
/// distribution as --bdpt/--mlt -- see bdpt_adapter.h's own "Scope"
/// comment; goniometric/projection lights still aren't wired in); with
/// sample_lights=0 this restriction is moot (no light sampling happens at
/// all).
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel
/// @param max_depth       Maximum path length
/// @param sample_lights   1 = perform NEE at each diffuse vertex (pbrt-v4 default)
/// @param sample_bsdf     1 = BSDF-importance-sample the next direction,
///                        0 = uniform hemisphere (pbrt-v4 default: 1)
/// @param output_path     Output PPM/EXR file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_simplepath(
    int width,
    int height,
    int spp,
    int max_depth,
    int sample_lights,
    int sample_bsdf,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using SimpleVolPathIntegrator -- pbrt-v4's
/// simplest volumetric path tracer (pure delta tracking, no NEE, no
/// surface BSDFs). See src/shared/simple_vol_path.h's own file comment.
/// CPU only.
///
/// This integration is reachable but medium-FREE: BDPTSceneAdapter's
/// HasMedium() always returns false (no participating-media wiring in this
/// adapter -- see bdpt_adapter.h's own comment on that method), so on
/// today's ordinary solid-geometry scenes this behaves exactly like
/// pbrt-v4's own SimpleVolPathIntegrator on a medium-free scene: it
/// terminates at the first surface hit, adding only that surface's own
/// area emission (SimpleVolPathIntegrator has no surface BSDF support at
/// all, matching upstream) -- so most scenes render mostly black except
/// where camera rays land directly on a light, or (if the scene defines
/// one) escape to the real cam.sky environment map on a miss - matching
/// every other integrator's own miss handling, not a forced flat
/// background. Full participating-media support (constant_medium.h/
/// cloud_medium.h/rgb_grid_medium.h) is future work.
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel
/// @param max_depth       Maximum scattering-event depth (medium scenes only)
/// @param output_path     Output PPM/EXR file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_simplevolpath(
    int width,
    int height,
    int spp,
    int max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// @brief Render a scene using LightPathIntegrator -- a pure light-tracer:
/// every sample starts at a light and walks forward, splatting a
/// camera-connection contribution into the film at each vertex (the
/// opposite direction of every other integrator this project has, which
/// all start at the camera). See src/shared/light_path.h's own file
/// comment. CPU only. Area lights only (SampleLightEmission only samples
/// this adapter's diffuse_light emitters -- unlike SampleLight/SampleLightLe,
/// which --bdpt/--mlt/--simplepath now also draw from point/spot/distant/
/// sky lights, this method was not extended -- see bdpt_adapter.h's own
/// "Scope" comment and AreaEmitterCount()).
///
/// Uses BDPTSceneAdapter's importance-transport BSDF hooks
/// (BSDFfImportance/BSDFSampleFImportance), which do not apply the eta^2
/// non-symmetric-scattering correction real refraction needs -- exact for
/// every purely-reflective material this codebase's BSDF bridge supports,
/// an approximation through the dielectric family. See bdpt_adapter.h's
/// own comment on those two methods.
/// @param width/height    Image dimensions in pixels
/// @param spp             Samples per pixel (light paths traced per pixel,
///                        matching pbrt-v4's own LightPathIntegrator
///                        convention -- total paths traced = spp*width*height)
/// @param max_depth       Maximum light-path length
/// @param output_path     Output PPM/EXR file path
/// @param scene_id        Scene selector (see scene_registry.h)
/// @param cam_x/y/z       Camera position (see cpu_render_main's own doc)
/// @param force_camera_override  Same semantics as cpu_render_main's own parameter
/// @return 0 on success, non-zero error code on failure
int cpu_render_main_lightpath(
    int width,
    int height,
    int spp,
    int max_depth,
    const char* output_path,
    const char* scene_id,
    double cam_x,
    double cam_y,
    double cam_z,
    int force_camera_override = 0
);

/// Scene metadata C API -- lets the GUI query the registry without C++ headers
/// @return total number of registered scenes
int cpu_scene_count();

/// @param index  position in registry (0..cpu_scene_count()-1)
/// @return scene id (category letter + number, e.g. "B10"), or "" if index
/// out of range. This is the only bridge from "position in registry" to
/// "id" - every function below takes an id, not a position, so a caller
/// that only knows a position (e.g. enumerating the whole registry) starts
/// here.
const char* cpu_scene_id(int index);

/// @param index  position in registry
/// @return scene name string, or "" if out of range
const char* cpu_scene_name(int index);

/// @param index  position in registry
/// @return short description string
const char* cpu_scene_description(int index);

/// @param index  position in registry
/// @return performance hint: "Fast", "Medium", "Slow", "Very Slow"
const char* cpu_scene_performance(int index);

/// @param index  position in registry
/// @return recommended samples-per-pixel
int cpu_scene_recommended_spp(int index);

/// @param index  position in registry
/// @return 1 if scene requires external files (earthmap.jpg etc), else 0
int cpu_scene_requires_files(int index);

/// @param index  position in registry
/// @return 1 if GPU compatible, else 0
int cpu_scene_gpu_compatible(int index);

/// Looks up scene_id (not index - a registry id, unlike the functions above)
/// and returns its recommended camera: the CameraConfig's lookfrom and
/// lookat points, verbatim. Used by main.cpp's video mode to scale AND
/// phase-align the built-in camera-path animations (orbit/linear/figure8/
/// spiral - see launcher/camera_path.h's get_camera_position()) to each
/// scene's actual coordinate scale and default viewpoint, instead of every
/// scene's video using the same Cornell-Box-scale orbit (radius 800 around
/// (278,278,278), starting at a fixed angle unrelated to any scene's actual
/// default view) regardless of scene. Any of the six out-params may be null
/// if that value isn't needed.
/// @return 1 on success, 0 if scene_id isn't found (out-params left untouched)
int cpu_scene_recommended_camera(const char* scene_id,
	double* lookfrom_x, double* lookfrom_y, double* lookfrom_z,
	double* lookat_x, double* lookat_y, double* lookat_z);

/// Looks up scene_id (not index, like cpu_scene_recommended_camera above -
/// unlike cpu_scene_gpu_compatible above, which is index-based) and returns
/// whether gpu/optix/scene_builder.cpp has a case for it.
/// @return 1 if GPU compatible, 0 if not (including if scene_id isn't found)
int cpu_scene_gpu_compatible_by_id(const char* scene_id);

/// Whether this scene's CameraConfig.animated is set (real AnimatedTransform
/// camera motion blur - see camera.h's camera_is_animated field comment).
/// launcher/main.cpp uses this to reject --video combined with an
/// animated-camera scene at argument-parsing time.
/// @return 1 if animated, 0 if not (including if scene_id isn't found)
int cpu_scene_camera_is_animated_by_id(const char* scene_id);

/// The rest of this file's index-based accessors (cpu_scene_name/
/// description/performance/recommended_spp/requires_files) are also
/// available by id - lets a caller with just a scene_id (the GUI's scene
/// combo box stores ids, not registry positions) look a single scene up
/// directly instead of first resolving id -> index via cpu_scene_id().
/// Same "" / 0 / default-on-not-found behavior as their index-based
/// counterparts.
const char* cpu_scene_name_by_id(const char* scene_id);
/// The scene's category ("Basics", "Materials", ...) - one of the
/// SceneCategories:: constants in src/shared/scene_descriptor.h. The Qt GUI
/// groups its scene list by this.
const char* cpu_scene_category_by_id(const char* scene_id);
const char* cpu_scene_description_by_id(const char* scene_id);
/// For a scene loaded from a .pbrt file, the path it was loaded from; "" for
/// every built-in scene. This exists so the GPU scene builder can find the
/// same file the CPU registry found. The alternative - re-scanning the scene
/// directory from the GPU side - would be a second implementation of the
/// search order and free to disagree about which file is scene 65.
const char* cpu_scene_pbrt_path_by_id(const char* scene_id);
/// The OLD flat 0..N int id (SceneDescriptor::legacy_id - see its comment
/// in scene_registry.h) for a scene, or -1 if scene_id isn't found. This
/// exists ONLY so gpu/optix/scene_builder.cpp's build_scene() can translate
/// the new string id into what its large switch(scene_id) still expects,
/// without that file needing to include scene_registry.h's full CPU
/// hittable/material class hierarchy just for this one lookup. Do not use
/// this for anything else - it is not a second public id.
int cpu_scene_legacy_id_by_id(const char* scene_id);
const char* cpu_scene_performance_by_id(const char* scene_id);
int cpu_scene_recommended_spp_by_id(const char* scene_id);
int cpu_scene_requires_files_by_id(const char* scene_id);
/// For a scene loaded from a .pbrt file that itself declares a Sampler/
/// Integrator/light sampler directive (SceneDescriptor::recommended_*,
/// see scene_registry.h) - "" for every hand-built scene, and for a
/// loaded scene whose file doesn't declare that particular directive.
/// NOT auto-applied anywhere - cpu_render_main() only warns to stderr
/// when the actual render settings differ (cpu_interface.cpp's own
/// comment on that warning block); this lets the GUI show the same
/// mismatch as a visible hint instead of a buried console line.
const char* cpu_scene_recommended_integrator_by_id(const char* scene_id);
const char* cpu_scene_recommended_sampler_by_id(const char* scene_id);
const char* cpu_scene_recommended_light_sampler_by_id(const char* scene_id);

#ifdef __cplusplus
}
#endif

#endif // CPU_INTERFACE_H
