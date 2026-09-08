#pragma once
// accelerator_override.h -- carries --accelerator/--splitmethod's CLI value
// from cpu_render_main() (cpu_renderer/cpu_interface.cpp, where RenderOptions
// is parsed) to pbrt_scene_registry::wire_pbrt_backed_scene()'s lazy ensure()
// lambda (src/TheRestOfYourLife/scene_registry.h). That lambda is created
// once, for every pbrt-backed SceneDescriptor, at process-startup registry
// construction time - long before any CLI argument has been parsed - but is
// only actually INVOKED lazily, on first access to a scene's build_world()/
// build_lights()/etc, which happens well after cpu_render_main() has parsed
// options and called set() below.
//
// A process-global (rather than a build_world() parameter) because
// build_world is a bare, zero-argument std::function referenced positionally
// by ~100 native scene builder functions (scene_registry_data.h) that have
// no Accelerator-directive concept to override in the first place - see
// SceneDescriptor::is_pbrt_backed's own comment for how those scenes are
// told this flag has no effect for them instead.
//
// Deliberately dumb: just a value carrier, no validation/fallback logic of
// its own. pbrt_load::loadFile()'s own `override` parameter applies this
// onto the freshly-parsed pbrt_scene::Scene BEFORE pbrt_flatten::flatten()
// runs, so a CLI override is resolved by the exact same, already-tested
// validation/fallback/motion-blur-compatibility logic flatten() already
// applies to the scene's own Accelerator directive (pbrt_flatten.h, the
// acceleratorType/acceleratorSplitMethod resolution block) - not a second,
// duplicated resolution path.
//
// Only ever written once per process, by cpu_render_main(), before any
// scene lookup/build happens - not a multi-render-per-process hazard in
// today's one-render-per-process CLI/launcher architecture. reset() exists
// so unit tests (and any future persistent-process caller) can restore
// defaults between uses.

#include <string>

namespace accelerator_override {

struct Override {
	bool has_type = false;
	std::string type;           // "bvh" or "kdtree"
	bool has_split_method = false;
	std::string split_method;   // "sah"/"middle"/"equal"/"hlbvh"
};

inline Override &state() {
	static Override o;
	return o;
}

inline void set(const Override &o) { state() = o; }
inline void reset() { state() = Override(); }

} // namespace accelerator_override
