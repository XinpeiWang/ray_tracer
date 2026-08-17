#pragma once
//==============================================================================================
// bdpt_render_bridge.h -- narrow interface between the scene-building half
// and the BDPT/MLT-rendering half of the --bdpt/--mlt CLI integration,
// split across two .cpp files specifically to avoid a THIRD-level AliasTable
// ODR collision beyond the one bdpt_adapter.h's own file comment already
// documents.
//
// The collision, in full: cpu_renderer/cpu_interface.h's cpu_render_main_bdpt()/
// cpu_render_main_mlt() need TWO things that individually already avoid the
// power_light_sampler.h vs. reservoir_sampler.h AliasTable collision, but
// NOT when combined in one translation unit:
//   1. scene_registry.h (to build a scene by id via find_scene()) -- which
//      includes cornell_box_scene.h, which includes
//      src/TheRestOfYourLife/power_light_sampler.h (a global `class AliasTable`).
//   2. bdpt_adapter.h (to actually run BDPT/MLT) -- which includes
//      src/shared/mlt.h, which includes src/shared/reservoir_sampler.h (a
//      DIFFERENT global `class AliasTable`).
// bdpt_adapter.h's own file comment already explains why sppm_adapter.h's
// Layer 1 was extracted into bsdf_bridge.h so bdpt_adapter.h itself doesn't
// need power_light_sampler.h -- but scene_registry.h needs it regardless
// (for build_cornell_box_power_lights(), unrelated to BDPT/MLT at all), so
// splitting bdpt_adapter.h away from sppm_adapter.h wasn't sufficient on its
// own once scene construction is also required in the same place.
//
// The fix: this header is the ONLY thing shared across that boundary.
// hittable_list.h and camera.h (its only two dependencies) are confirmed to
// pull in neither AliasTable, so passing an already-built hittable_list +
// camera across this boundary lets the scene-building side
// (cpu_interface_bdpt.cpp: scene_registry.h, no bdpt_adapter.h/mlt.h) and
// the rendering side (bdpt_render_core.cpp: bdpt_adapter.h/mlt.h, no
// scene_registry.h) each live in their own translation unit, neither of
// which ever sees both AliasTables at once.
//==============================================================================================

#include "hittable_list.h"
#include "camera.h"

#include <string>
#include <cstdint>

// Implemented in cpu_renderer/bdpt_render_core.cpp. `cam` must already be
// fully configured (lookfrom/lookat/vfov/background/etc, any alt-camera-model
// setup_camera() already applied) but NOT yet have had initialize() called --
// these functions call it themselves, matching cpu_render_main_sppm()'s own
// convention of calling cam.initialize() right before handing off to the
// adapter.
int bdpt_render_core(const hittable_list& world, camera& cam,
                      int spp, int bdpt_max_depth,
                      const std::string& output_path);

int mlt_render_core(const hittable_list& world, camera& cam,
                     int mlt_bootstrap, int64_t mlt_mutations, int mlt_max_depth,
                     const std::string& output_path);
