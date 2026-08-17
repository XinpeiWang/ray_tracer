// ============================================================================
// bdpt_render_core.cpp -- BDPT / MLT Render Entry Points (rendering half)
// ============================================================================
// Implements bdpt_render_core()/mlt_render_core(), declared in
// src/TheRestOfYourLife/bdpt_render_bridge.h -- the actual BDPTSceneAdapter
// + bdpt_render_with_adapter()/mlt_render_with_adapter() calls, given an
// already-built world/camera from cpu_interface_bdpt.cpp (the OTHER half of
// this split; see bdpt_render_bridge.h's own file comment for exactly why
// this needed to be two translation units at all -- an AliasTable ODR
// collision between scene_registry.h and src/shared/mlt.h that neither side
// can resolve alone).
//
// Deliberately does NOT include scene_registry.h (or anything that pulls in
// src/TheRestOfYourLife/power_light_sampler.h) -- only bdpt_render_bridge.h
// (hittable_list.h + camera.h, confirmed collision-free) and bdpt_adapter.h
// (which pulls src/shared/mlt.h -> src/shared/reservoir_sampler.h).
// ============================================================================

#include "../src/TheRestOfYourLife/bdpt_render_bridge.h"
#include "../src/TheRestOfYourLife/bdpt_adapter.h"
#include "../src/TheRestOfYourLife/error_codes.h"

#include <iostream>
#include <thread>

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
