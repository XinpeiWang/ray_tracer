// ============================================================================
// CPU Renderer Implementation
// ============================================================================
// This file implements the CPU-based ray tracer entry point.
//
// Renderer features:
//   - Multithreaded C++ path tracing
//   - Importance sampling using PDFs (probability density functions, NEE + MIS)
//   - Any scene registered in scene_registry.h, selected via scene_id
//   - Configurable camera position
//
// Camera behavior:
//   - Position (lookfrom) is set by caller via (cam_x, cam_y, cam_z)
//   - Target (lookat), up vector, and field of view come from the selected
//     scene's own CameraConfig (scene_registry.h) unless overridden
//
// Output handling:
//   - camera::render() writes directly to the caller's requested output
//     path (camera::output_path, set below) - no default-location detour
// ============================================================================

#include "cpu_interface.h"
#include "../src/TheRestOfYourLife/rtweekend.h"
#include "../src/TheRestOfYourLife/camera.h"
#include "../src/TheRestOfYourLife/scene_registry.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
#include "../src/TheRestOfYourLife/power_light_sampler.h"
#include "../src/TheRestOfYourLife/bvh_light_sampler.h"
#include "../src/TheRestOfYourLife/sppm_adapter.h"
#include "../src/TheRestOfYourLife/error_codes.h"
#include "../src/TheRestOfYourLife/bvh.h"
#include "../src/TheRestOfYourLife/bvh_aggregate_hittable.h"
#include "../src/TheRestOfYourLife/triangle.h"
#include "../src/TheRestOfYourLife/disk_cylinder_hittable.h"
#include "../src/TheRestOfYourLife/sphere_clipped_hittable.h"
#include "../src/TheRestOfYourLife/constant_medium.h"
#include "../src/TheRestOfYourLife/mesh.h"
#include "../src/TheRestOfYourLife/transform_instance.h"
#include "../src/shared/exr_writer.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <cctype>
#include <typeinfo>

// ============================================================================
// --spectral material-scan walker
// ============================================================================
// Recursively verifies every material reachable from a scene's `world` tree
// is one camera.h's ray_color_spectral() actually knows how to uplift to
// spectral (lambertian/metal/dielectric/rough_dielectric/conductor/
// diffuse_light - see that function's own comment for why exactly these
// six). Run once, before rendering starts, only when --spectral is passed -
// see cpu_render_main()'s own call site below.
//
// Fails closed on unrecognized STRUCTURE too, not just unrecognized
// materials: a hittable wrapper type this walker doesn't know how to
// recurse into (the final `else` branch below) is itself reported as an
// error, rather than silently skipped. A silent skip would turn "the
// walker has a gap" into a silently-wrong render - exactly the failure
// mode --spectral must never have. Any new hittable wrapper type added to
// this codebase in the future needs a case added here before --spectral
// can be used with a scene containing it.
//
// Checks a single material against --spectral's supported set, recursing
// into mix_material's own sub-materials (mat_a/mat_b) instead of rejecting
// it outright - a mix_material carries no color/BSDF of its own to check,
// only a stochastic choice between two others, so what actually matters is
// whether THOSE are supported. Without this, mix_material::as_dispersive()'s
// forwarding to a dispersive dielectric/rough_dielectric sub-material (see
// that override's own comment, material_pbrt.h) was correct but dead code:
// any scene using mix_material at all failed this scan before ever reaching
// ray_color_spectral(), regardless of what was mixed inside it or whether
// dispersion was involved at all.
//
// This is deliberately a SEPARATE recursive case from mix_material's own
// as_subsurface()/as_dispersive() overrides (material_pbrt.h), not a shared
// walker with those - they resolve to whichever ONE of mat_a/mat_b a given
// ray's branch_hash01() pick already committed to, while this has to check
// BOTH unconditionally: --spectral must reject a scene up front if EITHER
// possible pick would be unsupported, not just whichever one a probe query
// happens to land on. If a third place ever needs to walk a mix_material's
// reachable sub-materials, that's the point to weigh a shared traversal
// utility against these two differently-shaped needs - not before.
static bool spectral_scan_material(const material* mat, std::string& error_out, int depth = 0) {
	if (!mat) return true;

	if (dynamic_cast<const lambertian*>(mat))       return true;
	if (dynamic_cast<const metal*>(mat))            return true;
	if (dynamic_cast<const dielectric*>(mat))       return true;
	if (dynamic_cast<const rough_dielectric*>(mat)) return true;
	if (dynamic_cast<const conductor*>(mat))        return true;
	if (dynamic_cast<const diffuse_light*>(mat))    return true;

	if (const auto* mix = dynamic_cast<const mix_material*>(mat)) {
		// Depth-limited, not unbounded: a .pbrt "mix" material can name
		// another "mix" material as one of its own two sub-materials, so a
		// long chain of named mix materials each referencing the next would
		// otherwise recurse once per level with no cap. 64 is far beyond any
		// real scene (the deepest this codebase's own tests or curated
		// examples go is 2 - see mix_material_tests.cpp/
		// sppm_adapter_bsdf_tests.cpp's own inner_mix/outer_mix cases) - this
		// only ever fires on a pathological, hand-authored .pbrt file, and
		// failing closed here (same "unsupported thing found" reporting as
		// every other case in this function) beats a stack overflow crashing
		// the whole render before a single pixel traces.
		constexpr int kMaxMixMaterialDepth = 64;
		if (depth >= kMaxMixMaterialDepth) {
			error_out = "a mix_material chain nested more than " +
				std::to_string(kMaxMixMaterialDepth) + " levels deep";
			return false;
		}
		return spectral_scan_material(mix->get_mat_a().get(), error_out, depth + 1)
			&& spectral_scan_material(mix->get_mat_b().get(), error_out, depth + 1);
	}

	error_out = typeid(*mat).name();
	return false;
}

// Returns true (and leaves error_out untouched) if every reachable
// material/structure is recognized and supported; otherwise returns false
// with error_out set to a human-readable description of the first
// unsupported thing found.
static bool spectral_scan_hittable(const hittable* h, std::string& error_out) {
	if (!h) return true;

	if (const auto* list = dynamic_cast<const hittable_list*>(h)) {
		for (const auto& obj : list->objects)
			if (!spectral_scan_hittable(obj.get(), error_out)) return false;
		return true;
	}
	if (const auto* node = dynamic_cast<const bvh_node*>(h)) {
		return spectral_scan_hittable(node->get_left().get(), error_out)
			&& spectral_scan_hittable(node->get_right().get(), error_out);
	}
	if (const auto* leaf = dynamic_cast<const bvh_leaf*>(h)) {
		for (const auto& p : leaf->get_prims())
			if (!spectral_scan_hittable(p.get(), error_out)) return false;
		return true;
	}
	if (const auto* agg = dynamic_cast<const bvh_aggregate_hittable*>(h)) {
		for (const auto& p : agg->get_prims())
			if (!spectral_scan_hittable(p.get(), error_out)) return false;
		return true;
	}
	if (const auto* t = dynamic_cast<const translate*>(h))
		return spectral_scan_hittable(t->get_object().get(), error_out);
	if (const auto* r = dynamic_cast<const rotate_y*>(h))
		return spectral_scan_hittable(r->get_object().get(), error_out);
	if (const auto* m = dynamic_cast<const triangle_mesh*>(h))
		return spectral_scan_hittable(m->get_object().get(), error_out);
	if (const auto* mm = dynamic_cast<const triangle_mesh_mtl*>(h))
		return spectral_scan_hittable(mm->get_object().get(), error_out);
	if (const auto* ti = dynamic_cast<const transform_instance*>(h))
		return spectral_scan_hittable(ti->get_object().get(), error_out);

	if (dynamic_cast<const constant_medium*>(h)) {
		error_out = "constant_medium (volumetric fog/participating media - not "
			"one of --spectral's supported materials)";
		return false;
	}

	// Leaf primitive - resolve its material and check it against the
	// whitelist. Unlike GPU SPPM's equivalent check (optix_types.h), there's
	// no flat MaterialType enum here to switch on - each primitive type owns
	// its own material via a get_material() accessor (sphere.h/quad.h/
	// triangle.h/disk_cylinder_hittable.h), mirroring translate/rotate_y's
	// own get_object() pattern above.
	shared_ptr<material> mat;
	if (const auto* s = dynamic_cast<const sphere*>(h)) mat = s->get_material();
	else if (const auto* q = dynamic_cast<const quad*>(h)) mat = q->get_material();
	else if (const auto* tr = dynamic_cast<const triangle*>(h)) mat = tr->get_material();
	else if (const auto* d = dynamic_cast<const disk_hittable*>(h)) mat = d->get_material();
	else if (const auto* c = dynamic_cast<const cylinder_hittable*>(h)) mat = c->get_material();
	else if (const auto* sc = dynamic_cast<const sphere_clipped_hittable*>(h)) mat = sc->get_material();
	else {
		error_out = std::string("an unrecognized hittable wrapper/primitive type (") +
			typeid(*h).name() + ") - add support to the --spectral material-scan "
			"walker (cpu_interface.cpp, spectral_scan_hittable()) before using "
			"--spectral with a scene containing it";
		return false;
	}

	if (!mat) return true;  // no material on this primitive - nothing to check

	return spectral_scan_material(mat.get(), error_out);
}

// ============================================================================
// applyCameraConfig - shared CameraConfig -> camera wiring
// ============================================================================
// Used identically by cpu_render_main() and cpu_render_main_sppm() below -
// the two call sites became byte-for-byte duplicates once camera motion
// blur was added, so this factors the logic out to one place.
//
// IMPORTANT: force_camera_override is NOT a "this is a --video frame"
// signal - launcher/main.cpp passes force_camera_override=1 for EVERY
// single-frame CLI render too (see that file's own comment above its
// single-frame call sites: cam_x/y/z is always pre-populated, either from
// an explicit CLI arg or from the scene's own recommended camera via
// cpu_scene_recommended_camera(), and force_camera_override=1 just means
// "use whichever of those is now in cam_x/y/z"). So an animated
// (cc.animated) camera ALWAYS uses its own registered keyframes, ignoring
// cam_x/y/z/force_camera_override AND cc.mode entirely (unlike the
// non-animated branch below, where force_camera_override/UserControlled
// mode both let the caller substitute cam_x/y/z) - a moving camera has no
// single "current position" for either an override or a UserControlled-
// style UI preset to mean. Letting force_camera_override win here would
// silently disable motion blur (cam.camera_is_animated=false) on every
// ordinary, non-video render of an animated scene, not just fix --video
// mode - force_camera_override=1 on its own can't distinguish a genuine
// --video-frame conflict from the ordinary single-frame case where it's
// always set and never conflicts with anything. main.cpp separately
// rejects --video combined with an animated-camera scene outright, before
// any rendering starts (see that file's own comment) - the real conflict
// case, so no per-render warning is needed here.
static void applyCameraConfig(camera& cam, const CameraConfig& cc,
                               double cam_x, double cam_y, double cam_z,
                               bool force_camera_override) {
	cam.vfov          = cc.vfov;
	cam.background    = color(cc.bg_r, cc.bg_g, cc.bg_b);
	cam.defocus_angle = cc.defocus_angle;  // 0 = no DOF blur
	cam.focus_dist    = cc.focus_dist;
	cam.camera_is_animated = cc.animated;

	if (cc.animated) {
		cam.lookfrom      = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
		cam.lookfrom1     = point3(cc.lookfrom_t1_x, cc.lookfrom_t1_y, cc.lookfrom_t1_z);
		cam.lookat1       = point3(cc.lookat_t1_x, cc.lookat_t1_y, cc.lookat_t1_z);
		cam.shutter_open  = cc.shutter_open;
		cam.shutter_close = cc.shutter_close;
	} else if (force_camera_override || cc.mode == CameraMode::UserControlled) {
		// Let caller override lookfrom (camera presets in UI, or the CLI's
		// force_camera_override=1 for a single-frame render / --video frame).
		cam.lookfrom = point3(cam_x, cam_y, cam_z);
	} else {
		cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
	}
	cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
}

// ============================================================================
// cpu_render_main - CPU Render Entry Point
// ============================================================================
// C-linkage function that can be called from the launcher (launcher/main.cpp)
// Builds the requested scene, configures camera, renders, and copies output
// ============================================================================

extern "C" int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path,
								 const char* scene_id, double cam_x, double cam_y, double cam_z,
								 int force_camera_override, const RenderOptions& options) {
	try {
		// ====================================================================
		// Parameter Validation
		// ====================================================================

		// Validate dimensions
		if (width <= 0 || height <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_DIMENSIONS).to_string() << std::endl;
			return ERR_INVALID_DIMENSIONS;
		}

		// Validate sample count
		if (spp <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_SAMPLE_COUNT).to_string() << std::endl;
			return ERR_INVALID_SAMPLE_COUNT;
		}

		// Validate max depth
		if (max_depth <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_MAX_DEPTH).to_string() << std::endl;
			return ERR_INVALID_MAX_DEPTH;
		}

		// Validate output path - streaming a null const char* below (the
		// start-of-render log line) and later calls that treat it as a
		// filesystem path are both undefined behavior on a null pointer.
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << std::endl;
			return ERR_OUTPUT_PATH_INVALID;
		}

		// Validate scene ID against registry
		const SceneDescriptor* scene_desc = find_scene(scene_id);
		if (!scene_desc) {
			std::cerr << ErrorInfo(ERR_INVALID_SCENE_ID).to_string() << " (received: " << scene_id << ")" << std::endl;
			return ERR_INVALID_SCENE_ID;
		}

		// Log render start with all parameters
		std::clog << "[cpu_interface] cpu_render_main start: " << width << "x" << height
				  << " spp=" << spp << " scene_id=" << scene_id << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		// A loaded .pbrt scene's own Integrator directive is otherwise
		// silently ignored - this render always uses whatever max_depth the
		// caller passed in, regardless of what the scene file itself asked
		// for. Not auto-applied (see SceneDescriptor::recommended_max_depth's
		// own comment for why), but at minimum made visible rather than a
		// silent divergence between what the scene requested and what it got.
		// recommended_integrator != "volpath" is checked too - "volpath" is
		// both pbrt's own passthrough default AND what a scene with no
		// Integrator directive at all reports, so warning on that value
		// would fire for every ordinary scene, not just ones that actually
		// asked for something this function isn't running.
		if (scene_desc->recommended_max_depth > 0 && scene_desc->recommended_max_depth != max_depth) {
			std::cerr << "Warning: scene '" << scene_id << "' requests Integrator maxdepth="
					  << scene_desc->recommended_max_depth << " but this render is using max_depth="
					  << max_depth << " - the scene's own request has no effect here.\n";
		}
		if (!scene_desc->recommended_integrator.empty() && scene_desc->recommended_integrator != "volpath") {
			std::cerr << "Warning: scene '" << scene_id << "' requests Integrator \""
					  << scene_desc->recommended_integrator
					  << "\" but cpu_render_main always runs the default path tracer - "
						 "pass --bdpt/--mlt/--sppm explicitly if that's what the scene wants.\n";
		}
		// Same "advisory, not auto-applied" shape as maxdepth/integrator above
		// - "sobol" is skipped for the same reason "volpath" is above (both
		// the loader's own passthrough default AND what a scene with no
		// Sampler directive reports), and unlike maxdepth/exposure there's
		// no positional CLI argument for --sampler to compare against here
		// (main.cpp already resolved the string to a SamplerKind before this
		// call), so this only fires for an outright empty/unset --sampler -
		// i.e. the render is about to use Sobol regardless of what the scene
		// itself asked for.
		if (options.sampler == nullptr || options.sampler[0] == '\0') {
			if (!scene_desc->recommended_sampler.empty() && scene_desc->recommended_sampler != "sobol") {
				std::cerr << "Warning: scene '" << scene_id << "' requests Sampler \""
						  << scene_desc->recommended_sampler
						  << "\" but no --sampler was passed, so this render uses sobol - "
							 "pass --sampler " << scene_desc->recommended_sampler
						  << " explicitly if that's what the scene wants.\n";
			}
		}

		// ====================================================================
		// Scene Construction
		// ====================================================================
		// Build the selected scene using the centralized scene library

		// Build world and lights via registry -- no switch needed
		std::cout << "[cpu_interface] Building scene " << scene_id << " (" << scene_desc->name << ")..." << std::endl;
		hittable_list world      = scene_desc->build_world();
		hittable_list lights_raw = scene_desc->build_lights();

		// pbrt-v4 Integrator "string lightsampler" - advisory only, same
		// "CLI always decides, scene's own request only feeds a mismatch
		// warning" shape as maxdepth/samplerType above (a light sampler's
		// choice affects convergence/variance, not the converged image -
		// see pbrt_scene::Scene::lightSamplerType's own comment). "bvh" is
		// both pbrt-v4's own real default AND this project's own prior
		// hardcoded choice, so an empty --lightsampler reports/behaves the
		// same as before this option existed.
		const bool has_explicit_lightsampler = options.lightsampler != nullptr && options.lightsampler[0] != '\0';
		if (!has_explicit_lightsampler) {
			if (!scene_desc->recommended_light_sampler.empty() && scene_desc->recommended_light_sampler != "bvh") {
				std::cerr << "Warning: scene '" << scene_id << "' requests Integrator lightsampler \""
						  << scene_desc->recommended_light_sampler
						  << "\" but no --lightsampler was passed, so this render uses bvh - "
							 "pass --lightsampler " << scene_desc->recommended_light_sampler
						  << " explicitly if that's what the scene wants.\n";
			}
		}
		const std::string light_sampler_choice = has_explicit_lightsampler ? options.lightsampler : "bvh";

		// For Cornell box scenes use explicitly-weighted light sampling
		// (pbrt-v4 Â§12.6's bounding-cone importance sampler for "bvh", or
		// the matching hand-tuned power-weighted variant for "power" - see
		// bvh_light_sampler.h/power_light_sampler.h); for all others (and
		// for "uniform" even on Cornell scenes) wrap lights_raw directly
		// (equal weights = same as old hittable_list).
		// Scene 7 (Cornell Smoke) used to be included in the Cornell-box
		// special case too, but its light is a different size/color than
		// build_cornell_box_power_lights() assumes and it has no glass
		// sphere - the general path below (scene_desc->build_lights() ->
		// build_cornell_smoke_lights(), a single correctly-sized light) is
		// both correct and, with only one light in the list, equivalent to
		// power weighting anyway.
		// "A1"/"B2" are scene 0 (Cornell Box) / scene 10 (Cornell Rough
		// Metal) under the old flat numbering - see scene_registry.h's
		// SceneDescriptor::id. Held as a `hittable*` (cam.render() below
		// takes `const hittable&`, and each choice is a different concrete
		// type) rather than a stack object, since which type gets
		// constructed is now a runtime choice, not compile-time.
		// cpu_render_main_sppm() below has its own, structurally simpler copy
		// of this same "A1"/"B2" check (it can't express a --lightsampler
		// choice at all - SPPM's own light object is hard-typed to
		// bvh_light_sampler, so it never had a reason to grow past a plain
		// 2-way branch). If --lightsampler support is ever added to SPPM,
		// that copy needs the same is_cornell_box_scene/power/bvh treatment
		// this one has - grep this file for "A1") == 0" to find it.
		std::unique_ptr<hittable> lights_ptr;
		const bool is_cornell_box_scene = std::strcmp(scene_id, "A1") == 0 || std::strcmp(scene_id, "B2") == 0;
		if (light_sampler_choice == "uniform") {
			lights_ptr = std::make_unique<hittable_list>(lights_raw);
		} else if (light_sampler_choice == "power") {
			lights_ptr = is_cornell_box_scene
				? std::make_unique<power_light_list>(build_cornell_box_power_lights())
				: std::make_unique<power_light_list>(lights_raw);
		} else {
			lights_ptr = is_cornell_box_scene
				? std::make_unique<bvh_light_sampler>(build_cornell_box_bvh_lights())
				: std::make_unique<bvh_light_sampler>(lights_raw);
		}

			// Validate that scene was built successfully
			if (world.objects.size() == 0) {
				std::cerr << ErrorInfo(ERR_CPU_SCENE_EMPTY).to_string() << std::endl;
				return ERR_CPU_SCENE_EMPTY;
			}

			// --spectral only supports a bounded material set - see
			// spectral_scan_hittable()'s own comment. Scanned here (after the
			// world is built, before any pixel is traced) so an unsupported
			// scene fails loudly and immediately rather than silently
			// rendering with the wrong color model.
			if (options.spectral) {
				std::string bad;
				for (const auto& obj : world.objects) {
					if (!spectral_scan_hittable(obj.get(), bad)) break;
				}
				if (!bad.empty()) {
					std::cerr << ErrorInfo(ERR_CPU_MATERIAL_INVALID).to_string()
							  << " -- --spectral does not support scene " << scene_id
							  << ": uses " << bad << ". Render without --spectral instead.\n";
					return ERR_CPU_MATERIAL_INVALID;
				}
			}

		// ====================================================================
		// Camera Configuration
		// ====================================================================
		// Set up camera with scene-specific settings

		camera cam;
		cam.aspect_ratio      = double(width) / double(height);
		cam.image_width       = width;
		// image_height is normally computed by camera::initialize() (see its
		// comment at camera.h's image_height declaration), but setup_camera()
		// below runs before initialize() - scenes 32/33's alt-camera lambdas
		// read cam.image_height directly (compute_screen_window,
		// SphericalCamera's constructor), so it must already be valid here.
		// Mirrors camera::initialize()'s exact formula so the value
		// initialize() computes later is identical (harmless recompute).
		cam.image_height      = int(cam.image_width / cam.aspect_ratio);
		cam.image_height      = (cam.image_height < 1) ? 1 : cam.image_height;
		cam.samples_per_pixel = spp;
		cam.max_depth         = max_depth;
		cam.exposure          = options.exposure;
		cam.spectral          = options.spectral;
		// Auto-detected from the caller's requested extension, matching how
		// launcher/main.cpp already auto-triggers PNG conversion off the
		// output extension rather than a separate flag - see camera.h's own
		// exr_output field comment for what this actually changes in
		// render().
		cam.exr_output = is_exr_output_path(output_path);
		// render() now takes this path as a parameter (see its own comment)
		// and writes straight to it, instead of always writing to Desktop
		// and relying on this function to guess that location and copy the
		// file out afterward - the copy-out step below is gone accordingly.
		// sampler==nullptr (every existing caller that predates this param)
		// or an unrecognized name both fall back to Sobol - see
		// sampler_kind_from_name()'s own comment.
		if (options.sampler != nullptr) {
			SamplerKind kind;
			if (sampler_kind_from_name(options.sampler, kind)) {
				cam.sampler_kind = kind;
			} else if (options.sampler[0] != '\0') {
				std::cerr << "Warning: unrecognized --sampler \"" << options.sampler
						  << "\", using sobol. Valid: sobol, zsobol, paddedsobol, "
							 "stratified, pmj02bn, halton.\n";
			}
		}
		// tonemap==nullptr (every existing caller that predates this param)
		// or an unrecognized name both fall back to ACES - see
		// tone_map_mode_from_name()'s own comment (src/shared/tone_map.h).
		if (options.tonemap != nullptr) {
			ToneMapMode mode;
			if (tone_map_mode_from_name(options.tonemap, mode)) {
				cam.tone_map = mode;
			} else if (options.tonemap[0] != '\0') {
				std::cerr << "Warning: unrecognized --tonemap \"" << options.tonemap
						  << "\", using aces. Valid: aces, reinhard, none.\n";
			}
		}
		cam.vup               = vec3(0, 1, 0);  // Up direction is +Y

		// Apply camera config from registry
		const CameraConfig& cc = scene_desc->camera;
		applyCameraConfig(cam, cc, cam_x, cam_y, cam_z, force_camera_override);
		std::cout << "[cpu_interface] Camera: vfov=" << cc.vfov
				  << " lookfrom=(" << cam.lookfrom.x() << "," << cam.lookfrom.y() << "," << cam.lookfrom.z() << ")"
				  << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;

		// Apply optional sky light and punctual lights from scene descriptor
		if (scene_desc->build_sky)
			cam.sky = scene_desc->build_sky();
		if (scene_desc->build_portal)
			cam.portal = scene_desc->build_portal();
		// cam.sky/cam.portal are documented as mutually exclusive (see
		// BuildResult::portal's own comment) - pbrt_cpu_builder.h enforces
		// that for pbrt-loaded scenes, but nothing stops a future hand-
		// authored SceneDescriptor (scene_registry.h) from setting both
		// build_sky and build_portal by mistake. camera.h's "if (portal)
		// else if (sky)" branches would then silently drop the sky light
		// with no warning, so check for it here where both are populated.
		if (cam.sky && cam.portal) {
			std::cerr << "Warning: scene '" << scene_id << "' has both a plain sky light and a "
			             "portal light configured - only the portal will be used, the sky light "
			             "is being dropped (this scene's descriptor should only set one).\n";
			cam.sky.reset();
		}
		if (scene_desc->build_punct)
			cam.punct_lights = scene_desc->build_punct();
		// camera::camera_medium is ray_color() (default path tracer) only -
		// ray_color_spectral() (the function --spectral actually dispatches
		// to) never reads it - same "warn rather than silently drop"
		// precedent as the --sppm/--bdpt/--mlt/GPU cases.
		if (scene_desc->build_camera_medium) {
			if (options.spectral) {
				if (scene_desc->build_camera_medium()) {
					std::cerr << "Warning: scene '" << scene_id << "' has a camera medium (MediumInterface "
								 "declared before the Camera directive), which is not supported under "
								 "--spectral - the scene will render without it; use the default path "
								 "tracer instead if the ambient fog matters for this render.\n";
				}
			} else {
				cam.camera_medium = scene_desc->build_camera_medium();
			}
		}
		// Apply optional alternate camera model from scene descriptor
		if (scene_desc->setup_camera)
			scene_desc->setup_camera(cam);

		// ====================================================================
		// Output Path Handling
		// ====================================================================
		// Ensure the output directory exists
		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		// ====================================================================
		// Technique Summary
		// ====================================================================
		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : Iterative path tracer  (pbrt-v4 PathIntegrator::Li style)" << std::endl;
		std::cout << "[TECH] Sampler        : Stratified grid + Halton LDS (base-2/3, per-pixel decorrelated) + Sobol-Owen per bounce" << std::endl;
		std::cout << "[TECH] Reconstruction : Mitchell-Netravali filter  B=1/3  C=1/3  radius=0.5 px" << std::endl;
		std::cout << "[TECH] Acceleration   : SAH BVH  |  12 buckets  |  max 4 prims/leaf  |  C_trav=1  C_isect=2" << std::endl;
		// Reflects the actual light_sampler_choice made above, not a fixed
		// string - before Integrator "string lightsampler" existed this was
		// always bvh_light_sampler, so a hardcoded line was accurate; now
		// --lightsampler uniform/power select real, differently-behaved
		// samplers and the summary needs to say which one actually ran.
		if (light_sampler_choice == "uniform") {
			std::cout << "[TECH] Light sampling : Uniform (flat equal-weight, no importance sampling)" << std::endl;
		} else if (light_sampler_choice == "power") {
			std::cout << "[TECH] Light sampling : Power-weighted alias table  phi = Le * area" << std::endl;
		} else {
			std::cout << "[TECH] Light sampling : Bounding-cone BVH (pbrt-v4 sec. 12.6)  phi = area * Le * pi" << std::endl;
		}
		std::cout << "[TECH] MIS            : Power heuristic  beta=2  (BSDF sample + NEE light sample)" << std::endl;
		std::cout << "[TECH] Path termination: Russian Roulette per-bounce, etaScale-aware" << std::endl;
		std::cout << "[TECH] Firefly guard  : NaN / Inf samples clamped to 0" << std::endl;
		std::cout << "[TECH] Tone mapping   : " << tone_map_mode_display_name(cam.tone_map) << " --> sRGB OETF (IEC 61966-2-1)" << std::endl;
		std::cout << "[TECH] Threading      : " << std::thread::hardware_concurrency() << " logical cores (auto idle-adjusted on Windows)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		// ====================================================================
		// Render Execution
		// ====================================================================
		// output_path is passed straight through to render() (see its own
		// comment) - no Desktop detour, no copy-out step needed afterward.

		std::cout << "[cpu_interface] Starting render..." << std::endl;
		if (!cam.render(world, *lights_ptr, output_path)) {
			// render() already logged specifically what it tried and why -
			// see its own comment on the requested-path/cwd/TEMP fallback
			// chain (and the EXR-encode failure case) it just exhausted.
			std::cerr << "[cpu_interface] " << ErrorInfo(ERR_FILE_WRITE_FAILED).to_string() << std::endl;
			return ERR_FILE_WRITE_FAILED;
		}

		std::clog << "[cpu_interface] CPU render complete: " << output_path << std::endl;
		return SUCCESS; // Success code 0

	} catch (const std::bad_alloc& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
				  << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
				  << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

// ============================================================================
// cpu_render_main_sppm - SPPM Render Entry Point
// ============================================================================
// Separate from cpu_render_main() deliberately - see cpu_interface.h's doc
// comment on this function for why. Builds the scene via the same
// scene_registry closures cpu_render_main() uses, but renders through
// SPPMSceneAdapter (src/TheRestOfYourLife/sppm_adapter.h) instead of
// camera::render(), and writes the PPM directly to output_path (no
// Desktop-write-then-copy dance, since - unlike camera::render() - this
// path never writes anywhere else first).
// ============================================================================

extern "C" int cpu_render_main_sppm(int width, int height, int iterations, int photons, int max_depth,
									  const char* output_path, const char* scene_id, double cam_x, double cam_y,
									  double cam_z, int force_camera_override) {
	try {
		if (width <= 0 || height <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_DIMENSIONS).to_string() << std::endl;
			return ERR_INVALID_DIMENSIONS;
		}
		if (iterations <= 0 || photons <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_SAMPLE_COUNT).to_string() << std::endl;
			return ERR_INVALID_SAMPLE_COUNT;
		}
		if (max_depth <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_MAX_DEPTH).to_string() << std::endl;
			return ERR_INVALID_MAX_DEPTH;
		}
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << std::endl;
			return ERR_OUTPUT_PATH_INVALID;
		}

		const SceneDescriptor* scene_desc = find_scene(scene_id);
		if (!scene_desc) {
			std::cerr << ErrorInfo(ERR_INVALID_SCENE_ID).to_string() << " (received: " << scene_id << ")" << std::endl;
			return ERR_INVALID_SCENE_ID;
		}

		std::clog << "[cpu_interface] cpu_render_main_sppm start: " << width << "x" << height
				   << " iterations=" << iterations << " photons=" << photons << " scene_id=" << scene_id
				   << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		std::cout << "[cpu_interface] Building scene " << scene_id << " (" << scene_desc->name << ") for SPPM..." << std::endl;
		hittable_list world      = scene_desc->build_world();
		hittable_list lights_raw = scene_desc->build_lights();

		// SPPM has no --lightsampler of its own (see main.cpp's "has no
		// effect under ... --sppm ..." warning) - `lights` stays hard-typed
		// to bvh_light_sampler, so this "A1"/"B2" check stays a plain 2-way
		// branch rather than growing the uniform/power/bvh x cornell-or-not
		// matrix cpu_render_main() above has. If --lightsampler support is
		// ever added here too, mirror that function's own treatment of this
		// same check.
		bvh_light_sampler lights;
		if (std::strcmp(scene_id, "A1") == 0 || std::strcmp(scene_id, "B2") == 0) {
			lights = build_cornell_box_bvh_lights();
		} else {
			lights = bvh_light_sampler(lights_raw);
		}

		if (world.objects.size() == 0) {
			std::cerr << ErrorInfo(ERR_CPU_SCENE_EMPTY).to_string() << std::endl;
			return ERR_CPU_SCENE_EMPTY;
		}

		camera cam;
		cam.aspect_ratio = double(width) / double(height);
		cam.image_width  = width;
		cam.image_height = int(cam.image_width / cam.aspect_ratio);
		cam.image_height = (cam.image_height < 1) ? 1 : cam.image_height;
		cam.vup          = vec3(0, 1, 0);

		const CameraConfig& cc = scene_desc->camera;
		// Camera motion blur - see applyCameraConfig()'s own comment (same
		// CameraConfig fields, same reasoning, shared with cpu_render_main).
		applyCameraConfig(cam, cc, cam_x, cam_y, cam_z, force_camera_override);
		std::cout << "[cpu_interface] Camera: vfov=" << cc.vfov
				   << " lookfrom=(" << cam.lookfrom.x() << "," << cam.lookfrom.y() << "," << cam.lookfrom.z() << ")"
				   << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;
		// Alternate camera models (ortho/spherical/realistic) are honored
		// transparently via camera::get_ray() inside SPPMSceneAdapter::
		// PixelToRay(). Both punctual (delta) lights AND sky/infinite lights
		// are supported (SPPMSceneAdapter::DirectLight() queries
		// cam.punct_lights; sppm_camera_pass_with_sky() queries cam.sky on a
		// camera-ray miss - see sppm_adapter.h) - must be wired up here or a
		// scene like scene 27 "Point Light Cornell" or scene 24 "HDRI Sky"
		// renders pure black. cam.portal is deliberately NOT wired here:
		// sppm_adapter.h's own sky handling only ever calls a direction-only
		// Le() lookup (matching sky_light's own API), never a position-
		// dependent one - a portal light's window visibility genuinely
		// needs the query point, which SPPM's own miss-handling doesn't
		// carry - so a portal-lit scene rendered with --sppm still renders
		// pure black (no different from before this feature existed) rather
		// than silently doing something wrong.
		if (scene_desc->build_sky)
			cam.sky = scene_desc->build_sky();
		if (scene_desc->build_portal) {
			std::cerr << "Warning: scene '" << scene_id << "' has a portal (windowed) infinite "
						 "light, which is not supported under --sppm - it will not contribute "
						 "any light (rendering black through the window); use the default path "
						 "tracer instead if the portal light matters for this render.\n";
		}
		if (scene_desc->build_punct)
			cam.punct_lights = scene_desc->build_punct();
		// camera::camera_medium is ray_color() (default path tracer) only -
		// see that field's own comment (camera.h) - deliberately NOT set
		// here, same "warn rather than silently drop" precedent as the
		// portal-light case just above.
		if (scene_desc->build_camera_medium && scene_desc->build_camera_medium()) {
			std::cerr << "Warning: scene '" << scene_id << "' has a camera medium (MediumInterface "
						 "declared before the Camera directive), which is not supported under "
						 "--sppm - the scene will render without it; use the default path tracer "
						 "instead if the ambient fog matters for this render.\n";
		}
		if (scene_desc->setup_camera)
			scene_desc->setup_camera(cam);

		// Film "cropwindow"/"pixelbounds" (cam.crop_x0/x1/y0/y1, just set by
		// setup_camera() above) is now honored under --sppm too - see the
		// sppm_render_with_adapter() call below, which reads
		// cam.crop_x0/x1/y0/y1 after initialize() resolves them.
		cam.initialize();

		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : Stochastic Progressive Photon Mapping (pbrt-v4 SPPMIntegrator style)" << std::endl;
		std::cout << "[TECH] Photon lookup  : Spatial hash grid, progressive radius contraction (gamma=2/3)" << std::endl;
		std::cout << "[TECH] BSDF coverage  : lambertian + 8 delta materials only (see cpu_render_main_sppm's doc comment)" << std::endl;
		std::cout << "[TECH] Threading      : single-threaded (see sppm_adapter.h's BeginIteration() comment)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		std::cout << "[cpu_interface] Starting SPPM render (" << iterations << " iterations x " << photons << " photons)..." << std::endl;
		SPPMSceneAdapter adapter(world, lights, cam);
		std::vector<double> out_rgb;
		sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
								   iterations, photons, max_depth,
								   /*initialRadius=*/10.0, out_rgb,
								   cam.crop_x0, cam.crop_x1, cam.crop_y0, cam.crop_y1);

		if (is_exr_output_path(output_path)) {
			std::string exr_error;
			if (!sppm_write_exr(output_path, cam.image_width, cam.image_height, out_rgb, exr_error)) {
				std::cerr << "[cpu_interface] Failed to write EXR '" << output_path << "': " << exr_error << std::endl;
				return ERR_FILE_WRITE_FAILED;
			}
		} else {
			sppm_write_ppm(output_path, cam.image_width, cam.image_height, out_rgb);
		}

		std::clog << "[cpu_interface] SPPM render complete: " << output_path << std::endl;
		return SUCCESS;

	} catch (const std::bad_alloc& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
				   << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
				   << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[cpu_interface] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

// ============================================================================
// Scene metadata C API -- read from the registry, safe to call from C / Qt
// ============================================================================

extern "C" int cpu_scene_count() {
	return scene_count();
}

extern "C" const char* cpu_scene_id(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].id.c_str();
}

extern "C" const char* cpu_scene_name(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].name;
}

extern "C" const char* cpu_scene_description(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].description;
}

extern "C" const char* cpu_scene_performance(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return "";
	return reg[index].performance;
}

extern "C" int cpu_scene_recommended_spp(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 100;
	return reg[index].recommended_spp;
}

extern "C" int cpu_scene_requires_files(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 0;
	return reg[index].requires_files ? 1 : 0;
}

extern "C" int cpu_scene_gpu_compatible(int index) {
	const auto& reg = get_scene_registry();
	if (index < 0 || index >= (int)reg.size()) return 0;
	return reg[index].gpu_compatible ? 1 : 0;
}

extern "C" int cpu_scene_recommended_camera(const char* scene_id,
	double* lookfrom_x, double* lookfrom_y, double* lookfrom_z,
	double* lookat_x, double* lookat_y, double* lookat_z) {
	const SceneDescriptor* s = find_scene(scene_id);
	if (!s) return 0;
	const CameraConfig& cc = s->camera;
	// Every scene, including CameraMode::UserControlled ones (the
	// Cornell-box family), has a documented default lookfrom in the
	// registry (e.g. Cornell Box's (278,278,-800) front view - see main.cpp's
	// header comment) - it's a perfectly good video camera-path starting
	// point even though single-image rendering lets the user freely move
	// away from it, so there's no need to special-case UserControlled here.
	if (lookfrom_x) *lookfrom_x = cc.lookfrom_x;
	if (lookfrom_y) *lookfrom_y = cc.lookfrom_y;
	if (lookfrom_z) *lookfrom_z = cc.lookfrom_z;
	if (lookat_x) *lookat_x = cc.lookat_x;
	if (lookat_y) *lookat_y = cc.lookat_y;
	if (lookat_z) *lookat_z = cc.lookat_z;
	return 1;
}

extern "C" int cpu_scene_gpu_compatible_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	if (!s) return 0;
	return s->gpu_compatible ? 1 : 0;
}

// Lets launcher/main.cpp reject --video combined with an animated-camera
// scene (e.g. D13) at argument-parsing time, before any rendering starts -
// see main.cpp's own call site comment for why (the scene's own keyframed
// motion and --video's per-frame flythrough path can't be meaningfully
// composed).
extern "C" int cpu_scene_camera_is_animated_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	if (!s) return 0;
	return s->camera.animated ? 1 : 0;
}

extern "C" const char* cpu_scene_name_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->name : "";
}

extern "C" const char* cpu_scene_category_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->category : "";
}

extern "C" const char* cpu_scene_pbrt_path_by_id(const char* scene_id) {
	const auto& byId = pbrt_scene_registry::paths();
	const auto it = byId.find(scene_id);
	return (it == byId.end()) ? "" : it->second.c_str();
}

extern "C" int cpu_scene_legacy_id_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->legacy_id : -1;
}

extern "C" const char* cpu_scene_description_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->description : "";
}

extern "C" const char* cpu_scene_performance_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->performance : "";
}

extern "C" int cpu_scene_recommended_spp_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return s ? s->recommended_spp : 100;
}

extern "C" int cpu_scene_requires_files_by_id(const char* scene_id) {
	const SceneDescriptor* s = find_scene(scene_id);
	return (s && s->requires_files) ? 1 : 0;
}
