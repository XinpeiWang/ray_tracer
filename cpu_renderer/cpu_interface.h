/// @file cpu_interface.h
/// @brief CPU Renderer C Interface
/// @details This header defines the C-linkage interface for the CPU-based ray tracer.
/// The CPU renderer uses:
///   - Multithreaded C++ path tracing
///   - Importance sampling (PDF-based lighting)
///   - The shared Cornell box scene definition
///
/// Camera System:
///   - Camera position (lookfrom) is configurable via cam_x, cam_y, cam_z
///   - Camera target (lookat) is fixed at Cornell box center: (278, 278, 278)
///   - View up vector (vup) is fixed at (0, 1, 0)
///   - Vertical field of view (vfov) is fixed at 40 degrees
///
/// This interface allows the launcher (main.cpp) to call the CPU renderer
/// as an in-process library function with C linkage (no name mangling).

#ifndef CPU_INTERFACE_H
#define CPU_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Render the Cornell box scene using CPU path tracing with importance sampling
/// @details The Cornell box scene is predefined in src/TheRestOfYourLife/cornell_box_scene.h
/// and includes:
///   - Cornell box walls (red, green, white)
///   - Ceiling light source
///   - Glass sphere
///   - Rotated box
/// 
/// Camera configuration:
///   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position
///   - lookat:   (278, 278, 278) - fixed at Cornell box center
///   - vup:      (0, 1, 0) - fixed up direction
///   - vfov:     40 degrees - fixed field of view
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
    int force_camera_override = 0
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
/// Only supports lambertian (non-specular) materials for indirect/caustic
/// light transport plus the 8 delta-BSDF material classes (metal,
/// dielectric, rough_metal, rough_dielectric, conductor, coated_diffuse,
/// thin_dielectric, coated_conductor) - diffuse_transmission/
/// normalized_fresnel/mix_material are not yet supported by the BSDF
/// bridge (sppm_adapter.h's sppm_bsdf_f() comment explains why) and will
/// render those surfaces as black/incorrect.
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
/// Area lights ONLY (no punctual/sky-light NEE) -- see bdpt_adapter.h's own
/// "Scope (v1)" comment for the full reasoning. Verified end-to-end against
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

#ifdef __cplusplus
}
#endif

#endif // CPU_INTERFACE_H
