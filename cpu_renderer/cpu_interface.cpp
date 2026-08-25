// ============================================================================
// CPU Renderer Implementation
// ============================================================================
// This file implements the CPU-based ray tracer entry point.
// 
// Renderer features:
//   - Multithreaded C++ path tracing
//   - Importance sampling using PDFs (probability density functions)
//   - Cornell box scene with glass sphere and rotated box
//   - Configurable camera position
//
// Camera behavior:
//   - Position (lookfrom) is set by caller via (cam_x, cam_y, cam_z)
//   - Target (lookat) is fixed at Cornell box center: (278, 278, 278)
//   - This ensures camera always points toward the center of the scene
//
// Output handling:
//   - Camera class writes to OneDrive/Desktop by default
//   - This interface copies the result to the requested output path
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
#include "../src/TheRestOfYourLife/triangle.h"
#include "../src/TheRestOfYourLife/disk_cylinder_hittable.h"
#include "../src/TheRestOfYourLife/constant_medium.h"
#include "../src/TheRestOfYourLife/mesh.h"
#include "../src/TheRestOfYourLife/transform_instance.h"
#include "../src/shared/exr_writer.h"
#include <iostream>
#include <fstream>
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
// dispersion was involved at all. Recurses arbitrarily deep so a
// mix_material nested inside another mix_material is checked too.
static bool spectral_scan_material(const material* mat, std::string& error_out) {
	if (!mat) return true;

	if (dynamic_cast<const lambertian*>(mat))       return true;
	if (dynamic_cast<const metal*>(mat))            return true;
	if (dynamic_cast<const dielectric*>(mat))       return true;
	if (dynamic_cast<const rough_dielectric*>(mat)) return true;
	if (dynamic_cast<const conductor*>(mat))        return true;
	if (dynamic_cast<const diffuse_light*>(mat))    return true;

	if (const auto* mix = dynamic_cast<const mix_material*>(mat)) {
		return spectral_scan_material(mix->get_mat_a().get(), error_out)
			&& spectral_scan_material(mix->get_mat_b().get(), error_out);
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
// cpu_render_main - CPU Render Entry Point
// ============================================================================
// C-linkage function that can be called from the launcher (main.cpp)
// Builds the Cornell box scene, configures camera, renders, and copies output
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

		// For Cornell box scenes use explicitly-weighted BVH light sampling
		// (pbrt-v4 Â§12.6's bounding-cone importance sampler, replacing the
		// flat power_light_list this used to build - same power weights,
		// but now picked with spatial/directional locality instead of a
		// single global PMF; see bvh_light_sampler.h); for all others wrap
		// uniformly (equal weights = same as old hittable_list).
		// Scene 7 (Cornell Smoke) used to be included here too, but its light
		// is a different size/color than build_cornell_box_power_lights()
		// assumes and it has no glass sphere - the uniform-weight path below
		// (scene_desc->build_lights() -> build_cornell_smoke_lights(), a
		// single correctly-sized light) is both correct and, with only one
		// light in the list, equivalent to power weighting anyway.
		// "A1"/"B2" are scene 0 (Cornell Box) / scene 10 (Cornell Rough Metal)
		// under the old flat numbering - see scene_registry.h's SceneDescriptor::id.
		bvh_light_sampler lights;
		if (std::strcmp(scene_id, "A1") == 0 || std::strcmp(scene_id, "B2") == 0) {
			lights = build_cornell_box_bvh_lights();
		} else {
			lights = bvh_light_sampler(lights_raw);
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
		cam.vfov          = cc.vfov;
		cam.background    = color(cc.bg_r, cc.bg_g, cc.bg_b);
		cam.defocus_angle = cc.defocus_angle;  // 0 = no DOF blur
		cam.focus_dist    = cc.focus_dist;
		if (cc.mode == CameraMode::UserControlled || force_camera_override) {
			// Let caller override lookfrom (camera presets in UI, or a
			// video-mode frame's animated per-frame position - the latter
			// must be honored regardless of CameraMode, since a "video" that
			// silently ignores its own animated camera and stays frozen
			// defeats the point of video mode)
			cam.lookfrom = point3(cam_x, cam_y, cam_z);
		} else {
			cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
		}
		cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
		std::cout << "[cpu_interface] Camera: vfov=" << cc.vfov
				  << " lookfrom=(" << cam.lookfrom.x() << "," << cam.lookfrom.y() << "," << cam.lookfrom.z() << ")"
				  << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;

		// Apply optional sky light and punctual lights from scene descriptor
		if (scene_desc->build_sky)
			cam.sky = scene_desc->build_sky();
		if (scene_desc->build_punct)
			cam.punct_lights = scene_desc->build_punct();
		// Apply optional alternate camera model from scene descriptor
		if (scene_desc->setup_camera)
			scene_desc->setup_camera(cam);

		// ====================================================================
		// Output Path Handling
		// ====================================================================
		// Save original requested path for later file copy

		std::string orig_path = std::string(output_path);

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
		std::cout << "[TECH] Light sampling : Bounding-cone BVH (pbrt-v4 sec. 12.6)  phi = area * Le * pi" << std::endl;
		std::cout << "[TECH] MIS            : Power heuristic  beta=2  (BSDF sample + NEE light sample)" << std::endl;
		std::cout << "[TECH] Path termination: Russian Roulette per-bounce, etaScale-aware" << std::endl;
		std::cout << "[TECH] Firefly guard  : NaN / Inf samples clamped to 0" << std::endl;
		std::cout << "[TECH] Tone mapping   : " << tone_map_mode_display_name(cam.tone_map) << " --> sRGB OETF (IEC 61966-2-1)" << std::endl;
		std::cout << "[TECH] Threading      : " << std::thread::hardware_concurrency() << " logical cores (auto idle-adjusted on Windows)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		// ====================================================================
		// Render Execution
		// ====================================================================
		// Camera class handles multithreaded rendering and writes to a default
		// location (OneDrive/Desktop). We'll copy the file afterward.

		std::cout << "[cpu_interface] Starting render..." << std::endl;
		cam.render(world, lights);

		// ====================================================================
		// Output File Copy
		// ====================================================================
		// The camera wrote to its default Desktop location; copy to requested path
		// This is necessary because the camera class has hardcoded output logic

		// Determine where the camera actually wrote the file
		// Priority: OneDrive Desktop > regular Desktop > current directory
		// Same filename render() itself computed (camera.h's exr_output-
		// gated `filename` local) - must match exactly or this copy finds
		// nothing.
		const char* written_filename = cam.exr_output ? "image.exr" : "image.ppm";
		std::string actual_output;
		if (const char* od = std::getenv("OneDrive")) {
			actual_output = std::string(od) + "\\Desktop\\" + written_filename;
		} else if (const char* up = std::getenv("USERPROFILE")) {
			if (std::filesystem::exists(std::filesystem::path(std::string(up) + "\\OneDrive"))) {
				actual_output = std::string(up) + "\\OneDrive\\Desktop\\" + written_filename;
			} else {
				actual_output = std::string(up) + "\\Desktop\\" + written_filename;
			}
		} else {
			actual_output = written_filename;
		}

		// Copy the file to the requested location if different
		if (actual_output != output_path) {
			try {
				std::filesystem::copy_file(actual_output, output_path, std::filesystem::copy_options::overwrite_existing);
				std::clog << "[cpu_interface] Copied " << actual_output << " to " << output_path << std::endl;
			} catch (const std::exception& e) {
				std::cerr << "[cpu_interface] " << ErrorInfo(ERR_FILE_COPY_FAILED).to_string() 
						  << " - " << e.what() << std::endl;
				return ERR_FILE_COPY_FAILED;
			}
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
		cam.vfov          = cc.vfov;
		cam.background    = color(cc.bg_r, cc.bg_g, cc.bg_b);
		cam.defocus_angle = cc.defocus_angle;
		cam.focus_dist    = cc.focus_dist;
		if (cc.mode == CameraMode::UserControlled || force_camera_override) {
			cam.lookfrom = point3(cam_x, cam_y, cam_z);
		} else {
			cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
		}
		cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
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
		// renders pure black.
		if (scene_desc->build_sky)
			cam.sky = scene_desc->build_sky();
		if (scene_desc->build_punct)
			cam.punct_lights = scene_desc->build_punct();
		if (scene_desc->setup_camera)
			scene_desc->setup_camera(cam);
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
								   /*initialRadius=*/10.0, out_rgb);

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
