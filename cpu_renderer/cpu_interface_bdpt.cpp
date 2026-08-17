// ============================================================================
// cpu_interface_bdpt.cpp -- BDPT / MLT Render Entry Points (scene-building half)
// ============================================================================
// The extern "C" entry points cpu_interface.h declares for --bdpt/--mlt.
// Builds the scene via scene_registry.h -- the SAME closures
// cpu_render_main()/cpu_render_main_sppm() (cpu_interface.cpp) use -- and
// hands the built world/camera to bdpt_render_core()/mlt_render_core()
// (cpu_renderer/bdpt_render_core.cpp) to actually render.
//
// Deliberately does NOT include bdpt_adapter.h or src/shared/mlt.h directly
// -- see src/TheRestOfYourLife/bdpt_render_bridge.h's own file comment for
// why: scene_registry.h (needed here) and mlt.h (needed by the actual BDPT/
// MLT render loops) each transitively pull in a DIFFERENT, same-named
// global `class AliasTable` (power_light_sampler.h vs. reservoir_sampler.h)
// that can never coexist in one translation unit. This file only ever
// builds a scene and calls across bdpt_render_bridge.h's narrow interface;
// bdpt_render_core.cpp is the other half, and never includes scene_registry.h.
// ============================================================================

#include "cpu_interface.h"
#include "../src/TheRestOfYourLife/rtweekend.h"
#include "../src/TheRestOfYourLife/camera.h"
#include "../src/TheRestOfYourLife/scene_registry.h"
#include "../src/TheRestOfYourLife/hittable_list.h"
#include "../src/TheRestOfYourLife/bdpt_render_bridge.h"
#include "../src/TheRestOfYourLife/error_codes.h"
#include <iostream>
#include <filesystem>
#include <cstring>

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
