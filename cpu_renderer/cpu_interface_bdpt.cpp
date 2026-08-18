// ============================================================================
// cpu_interface_bdpt.cpp -- BDPT / MLT Render Entry Points
// ============================================================================
// The extern "C" entry points cpu_interface.h declares for --bdpt/--mlt.
// Builds the scene via scene_registry.h -- the SAME closures
// cpu_render_main()/cpu_render_main_sppm() (cpu_interface.cpp) use -- then
// runs bdpt_render_core()/mlt_render_core() (below) to actually render via
// BDPTSceneAdapter + bdpt_render_with_adapter()/mlt_render_with_adapter().
//
// A separate translation unit from cpu_interface.cpp specifically to avoid
// an AliasTable ODR collision: scene_registry.h (needed here, for
// find_scene()) pulls in src/TheRestOfYourLife/power_light_sampler.h, while
// src/shared/mlt.h (needed by BDPT/MLT's own rendering code below) pulls in
// src/shared/reservoir_sampler.h - both used to define their own,
// differently-implemented global `class AliasTable`, an ODR violation if
// both headers were ever included in the same translation unit. That
// collision no longer exists (power_light_sampler.h now includes
// reservoir_sampler.h's AliasTable directly instead of hand-porting its own
// copy - see power_light_sampler.h's own comment), which is also why the
// scene-building and rendering halves of this file no longer need to be two
// separate .cpp files joined by a narrow bridge header (this file used to be
// split as cpu_interface_bdpt.cpp + bdpt_render_core.cpp +
// src/TheRestOfYourLife/bdpt_render_bridge.h purely to keep scene_registry.h
// and bdpt_adapter.h/mlt.h from ever coexisting in one TU).
// ============================================================================

#include "cpu_interface.h"
#include "../src/TheRestOfYourLife/rtweekend.h"
#include "../src/TheRestOfYourLife/camera.h"
#include "../src/TheRestOfYourLife/scene_registry.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
#include "../src/TheRestOfYourLife/bdpt_adapter.h"
#include "../src/TheRestOfYourLife/error_codes.h"
#include <iostream>
#include <filesystem>
#include <cstring>
#include <thread>

namespace {

// Shared by both entry points below -- builds world + configures cam
// exactly like cpu_render_main_sppm() does, minus the sky/punctual-light
// wiring (BDPT/MLT are area-lights-only in this v1 -- see bdpt_adapter.h's
// own "Scope (v1)" comment). Returns nullptr on success, or an ErrorInfo
// error code (via out_err) on failure.
const SceneDescriptor* build_scene_for_bdpt(const char* scene_id, int width, int height,
                                             double cam_x, double cam_y, double cam_z,
                                             int force_camera_override,
                                             hittable_list& out_world, camera& out_cam,
                                             int& out_err) {
	out_err = SUCCESS;
	const SceneDescriptor* scene_desc = find_scene(scene_id);
	if (!scene_desc) {
		std::cerr << ErrorInfo(ERR_INVALID_SCENE_ID).to_string() << " (received: " << scene_id << ")" << std::endl;
		out_err = ERR_INVALID_SCENE_ID;
		return nullptr;
	}

	out_world = scene_desc->build_world();
	if (out_world.objects.size() == 0) {
		std::cerr << ErrorInfo(ERR_CPU_SCENE_EMPTY).to_string() << std::endl;
		out_err = ERR_CPU_SCENE_EMPTY;
		return nullptr;
	}

	out_cam.aspect_ratio = double(width) / double(height);
	out_cam.image_width  = width;
	out_cam.image_height = int(out_cam.image_width / out_cam.aspect_ratio);
	out_cam.image_height = (out_cam.image_height < 1) ? 1 : out_cam.image_height;
	out_cam.vup           = vec3(0, 1, 0);

	const CameraConfig& cc = scene_desc->camera;
	out_cam.vfov          = cc.vfov;
	out_cam.background    = color(cc.bg_r, cc.bg_g, cc.bg_b);
	out_cam.defocus_angle = cc.defocus_angle;
	out_cam.focus_dist    = cc.focus_dist;
	if (cc.mode == CameraMode::UserControlled || force_camera_override) {
		out_cam.lookfrom = point3(cam_x, cam_y, cam_z);
	} else {
		out_cam.lookfrom = point3(cc.lookfrom_x, cc.lookfrom_y, cc.lookfrom_z);
	}
	out_cam.lookat = point3(cc.lookat_x, cc.lookat_y, cc.lookat_z);
	std::cout << "[cpu_interface_bdpt] Camera: vfov=" << cc.vfov
	           << " lookfrom=(" << out_cam.lookfrom.x() << "," << out_cam.lookfrom.y() << "," << out_cam.lookfrom.z() << ")"
	           << " lookat=(" << cc.lookat_x << "," << cc.lookat_y << "," << cc.lookat_z << ")" << std::endl;
	// Alternate camera models (ortho/spherical/realistic) are honored
	// transparently via camera::get_ray() inside BDPTSceneAdapter::
	// PixelToRay() -- but BDPTSceneAdapter::CameraPDFWe() itself assumes the
	// default perspective vfov/focus_dist model (see bdpt_adapter.h's own
	// "Scope (v1)" comment), so alt-camera scenes are unverified under
	// --bdpt/--mlt even though they won't crash. Sky/punctual lights are
	// v1-out-of-scope (area lights only -- same comment), so build_sky()/
	// build_punct() are deliberately NOT wired into out_cam here, unlike
	// cpu_render_main_sppm.
	if (scene_desc->setup_camera)
		scene_desc->setup_camera(out_cam);

	return scene_desc;
}

// bdpt_render_core/mlt_render_core -- given an already-built world/camera
// (`cam` fully configured but NOT yet initialize()'d, matching
// cpu_render_main_sppm()'s own convention), run BDPT/MLT via
// BDPTSceneAdapter + bdpt_render_with_adapter()/mlt_render_with_adapter()
// and write the result.
int bdpt_render_core(const hittable_list& world, camera& cam,
                      int spp, int bdpt_max_depth,
                      const std::string& output_path) {
	try {
		cam.initialize();

		std::cout << "[TECH] -- Render Technique Summary --------------------------" << std::endl;
		std::cout << "[TECH] Integrator     : Bidirectional Path Tracing (pbrt-v4 BDPTIntegrator style)" << std::endl;
		std::cout << "[TECH] MIS            : Balanced multi-strategy weight over all (s,t) connections" << std::endl;
		std::cout << "[TECH] Light coverage : area lights only (v1 -- see bdpt_adapter.h's Scope comment)" << std::endl;
		std::cout << "[TECH] BSDF coverage  : lambertian/normalized_fresnel/diffuse_transmission (full) + 8 delta materials (resampled)" << std::endl;
		std::cout << "[TECH] Threading      : " << std::thread::hardware_concurrency() << " logical cores, row-parallel" << std::endl;
		std::cout << "[TECH] -------------------------------------------------------" << std::endl;

		BDPTSceneAdapter adapter(world, cam);
		if (adapter.EmitterCount() == 0) {
			std::cerr << "[bdpt_render_core] WARNING: this scene has no area-light emitters "
			             "(diffuse_light shapes) - BDPT only samples area lights (v1, see "
			             "bdpt_adapter.h's Scope comment), so this render will be entirely "
			             "black even though it will report success. If this scene's only "
			             "lighting is punctual (point/spot/distant) or sky/infinite, that is "
			             "not yet supported by --bdpt/--mlt; use the default path tracer or "
			             "--sppm instead." << std::endl;
		}
		std::vector<double> out_rgb;
		bdpt_render_with_adapter(adapter, cam.image_width, cam.image_height, spp, bdpt_max_depth, out_rgb);

		bdpt_write_ppm(output_path, cam.image_width, cam.image_height, out_rgb);
		return SUCCESS;

	} catch (const std::bad_alloc& e) {
		std::cerr << "[bdpt_render_core] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[bdpt_render_core] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[bdpt_render_core] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

int mlt_render_core(const hittable_list& world, camera& cam,
                     int mlt_bootstrap, int64_t mlt_mutations, int mlt_max_depth,
                     const std::string& output_path) {
	try {
		cam.initialize();

		// pbrt-v4's own MLT defaults (Render.cpp / mlt.pbrt): sigma=0.01,
		// largeStepProbability=0.3 -- not exposed as CLI flags (see
		// launcher_args.h's --mlt-* flag list) since these tune the Markov
		// chain's mixing behavior, not something a typical render needs to
		// retune per scene the way iteration/sample counts do.
		constexpr double kSigma = 0.01;
		constexpr double kLargeStepProb = 0.3;

		std::cout << "[TECH] -- Render Technique Summary --------------------------" << std::endl;
		std::cout << "[TECH] Integrator     : Metropolis Light Transport (pbrt-v4 MLTIntegrator style)" << std::endl;
		std::cout << "[TECH] Sampler        : Primary-sample-space Markov chain (bootstrap + small/large steps)" << std::endl;
		std::cout << "[TECH] sigma=" << kSigma << "  largeStepProb=" << kLargeStepProb << std::endl;
		std::cout << "[TECH] Light coverage : area lights only (v1 -- see bdpt_adapter.h's Scope comment)" << std::endl;
		std::cout << "[TECH] Threading      : " << std::thread::hardware_concurrency()
		           << " independent Markov chains (see mlt_render_with_adapter())" << std::endl;
		std::cout << "[TECH] -------------------------------------------------------" << std::endl;

		BDPTSceneAdapter adapter(world, cam);
		if (adapter.EmitterCount() == 0) {
			std::cerr << "[mlt_render_core] WARNING: this scene has no area-light emitters "
			             "(diffuse_light shapes) - MLT only samples area lights (v1, see "
			             "bdpt_adapter.h's Scope comment), so this render will be entirely "
			             "black even though it will report success. If this scene's only "
			             "lighting is punctual (point/spot/distant) or sky/infinite, that is "
			             "not yet supported by --bdpt/--mlt; use the default path tracer or "
			             "--sppm instead." << std::endl;
		}
		std::vector<double> out_rgb;
		mlt_render_with_adapter(adapter, cam.image_width, cam.image_height,
		                         mlt_bootstrap, mlt_mutations, mlt_max_depth,
		                         kSigma, kLargeStepProb, out_rgb);

		bdpt_write_ppm(output_path, cam.image_width, cam.image_height, out_rgb);
		return SUCCESS;

	} catch (const std::bad_alloc& e) {
		std::cerr << "[mlt_render_core] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[mlt_render_core] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[mlt_render_core] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

} // namespace

extern "C" int cpu_render_main_bdpt(int width, int height, int spp, int bdpt_max_depth,
                                     const char* output_path, const char* scene_id,
                                     double cam_x, double cam_y, double cam_z,
                                     int force_camera_override) {
	try {
		if (width <= 0 || height <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_DIMENSIONS).to_string() << std::endl;
			return ERR_INVALID_DIMENSIONS;
		}
		if (spp <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_SAMPLE_COUNT).to_string() << std::endl;
			return ERR_INVALID_SAMPLE_COUNT;
		}
		if (bdpt_max_depth <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_MAX_DEPTH).to_string() << std::endl;
			return ERR_INVALID_MAX_DEPTH;
		}
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << std::endl;
			return ERR_OUTPUT_PATH_INVALID;
		}

		std::clog << "[cpu_interface_bdpt] cpu_render_main_bdpt start: " << width << "x" << height
		           << " spp=" << spp << " bdpt_max_depth=" << bdpt_max_depth << " scene_id=" << scene_id
		           << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		hittable_list world;
		camera cam;
		int err = SUCCESS;
		const SceneDescriptor* scene_desc = build_scene_for_bdpt(
			scene_id, width, height, cam_x, cam_y, cam_z, force_camera_override, world, cam, err);
		if (!scene_desc) return err;
		std::cout << "[cpu_interface_bdpt] Built scene " << scene_id << " (" << scene_desc->name << ") for BDPT" << std::endl;

		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		std::cout << "[cpu_interface_bdpt] Starting BDPT render (" << spp << " spp, max depth " << bdpt_max_depth << ")..." << std::endl;
		int render_result = bdpt_render_core(world, cam, spp, bdpt_max_depth, std::string(output_path));

		if (render_result == SUCCESS) {
			std::clog << "[cpu_interface_bdpt] BDPT render complete: " << output_path << std::endl;
		}
		return render_result;

	} catch (const std::bad_alloc& e) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}

extern "C" int cpu_render_main_mlt(int width, int height, int mlt_bootstrap, long long mlt_mutations,
                                    int mlt_max_depth, const char* output_path, const char* scene_id,
                                    double cam_x, double cam_y, double cam_z,
                                    int force_camera_override) {
	try {
		if (width <= 0 || height <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_DIMENSIONS).to_string() << std::endl;
			return ERR_INVALID_DIMENSIONS;
		}
		if (mlt_bootstrap <= 0 || mlt_mutations <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_SAMPLE_COUNT).to_string() << std::endl;
			return ERR_INVALID_SAMPLE_COUNT;
		}
		if (mlt_max_depth <= 0) {
			std::cerr << ErrorInfo(ERR_INVALID_MAX_DEPTH).to_string() << std::endl;
			return ERR_INVALID_MAX_DEPTH;
		}
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << std::endl;
			return ERR_OUTPUT_PATH_INVALID;
		}

		std::clog << "[cpu_interface_bdpt] cpu_render_main_mlt start: " << width << "x" << height
		           << " bootstrap=" << mlt_bootstrap << " mutations=" << mlt_mutations
		           << " mlt_max_depth=" << mlt_max_depth << " scene_id=" << scene_id
		           << " camera=(" << cam_x << "," << cam_y << "," << cam_z << ") out=" << output_path << std::endl;

		hittable_list world;
		camera cam;
		int err = SUCCESS;
		const SceneDescriptor* scene_desc = build_scene_for_bdpt(
			scene_id, width, height, cam_x, cam_y, cam_z, force_camera_override, world, cam, err);
		if (!scene_desc) return err;
		std::cout << "[cpu_interface_bdpt] Built scene " << scene_id << " (" << scene_desc->name << ") for MLT" << std::endl;

		std::filesystem::path out_fs_path(output_path);
		if (!out_fs_path.parent_path().empty() && !std::filesystem::exists(out_fs_path.parent_path())) {
			std::filesystem::create_directories(out_fs_path.parent_path());
		}

		std::cout << "[cpu_interface_bdpt] Starting MLT render (" << mlt_bootstrap << " bootstrap x "
		           << mlt_mutations << " mutations, max depth " << mlt_max_depth << ")..." << std::endl;
		int render_result = mlt_render_core(world, cam, mlt_bootstrap, (int64_t)mlt_mutations, mlt_max_depth,
		                                     std::string(output_path));

		if (render_result == SUCCESS) {
			std::clog << "[cpu_interface_bdpt] MLT render complete: " << output_path << std::endl;
		}
		return render_result;

	} catch (const std::bad_alloc& e) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_CPU_MEMORY_ALLOCATION).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_MEMORY_ALLOCATION;
	} catch (const std::exception& e) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_CPU_RENDER_FAILED).to_string()
		           << " - " << e.what() << std::endl;
		return ERR_CPU_RENDER_FAILED;
	} catch (...) {
		std::cerr << "[cpu_interface_bdpt] " << ErrorInfo(ERR_UNKNOWN).to_string() << std::endl;
		return ERR_UNKNOWN;
	}
}
