#pragma once
// accelerator_override.h -- carries --accelerator/--splitmethod's CLI value
// from launcher/main.cpp (where LaunchArgs is parsed - set once, right after
// scene_id is unpacked, before any render entry point is dispatched to) to
// pbrt_scene_registry::wire_pbrt_backed_scene()'s lazy ensure() lambda
// (src/TheRestOfYourLife/scene_registry.h). That lambda is created once, for
// every pbrt-backed SceneDescriptor, at process-startup registry
// construction time - long before any CLI argument has been parsed - but is
// only actually INVOKED lazily, on first access to a scene's build_world()/
// build_lights()/etc, which happens well after main.cpp has parsed argv and
// called set() below. Not routed through RenderOptions (unlike --sampler/
// --lightsampler) because BDPT/MLT/SPPM/the debug integrators' own
// cpu_render_main_*() entry points take no RenderOptions parameter at all -
// see render_options.h's own comment on this.
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
// type/split_method use an empty string as "no override" (never a valid
// value - real ones are "bvh"/"kdtree" and "sah"/"middle"/"equal"/"hlbvh"),
// the same sentinel convention already used by every other link in this
// feature's own data path (launcher_args.h's std::string accelerator/
// splitmethod, render_options.h's const char* sampler/lightsampler) -
// deliberately NOT a separate has_type/has_split_method bool pair, which
// would just be redundant state a caller could set out of sync with the
// string.
//
// Only ever written once per process, by cpu_render_main(), before any
// scene lookup/build happens - not a multi-render-per-process hazard in
// today's one-render-per-process CLI/launcher architecture. reset() exists
// so unit tests (and any future persistent-process caller) can restore
// defaults between uses - see accelerator_override_tests.cpp, which calls
// it in TearDown() so one test's override can never leak into another's
// pbrt-backed scene build (wire_pbrt_backed_scene's ensure() lambda caches
// its BuildResult forever after the first build, so a leaked override isn't
// just wrong for one test - it silently persists for the rest of the
// process unless reset() runs).
//
// was_set() backs a debug-only ordering assertion (see scene_registry.h's
// ensure() lambda): cpu_render_main() calls set()
// unconditionally, with or without --accelerator/--splitmethod actually
// passed, specifically so a future edit that inserts an earlier scene touch
// (e.g. a preview render, an extra validation pass) ahead of that call has
// something to trip over instead of silently caching a wrong default -
// this couldn't be told apart from "no override was requested" using
// type/split_method alone, since both are legitimately empty in that case
// too.

#include <string>

namespace accelerator_override {

struct Override {
	std::string type;           // "bvh"/"kdtree", or "" for no override
	std::string split_method;   // "sah"/"middle"/"equal"/"hlbvh", or "" for no override
};

inline Override &state() {
	static Override o;
	return o;
}

inline bool &set_flag() {
	static bool flag = false;
	return flag;
}

inline void set(const Override &o) { state() = o; set_flag() = true; }
inline void reset() { state() = Override(); set_flag() = false; }

// Whether set() has run yet in this process - see this header's own
// was_set()/mark_set() comment above for why this exists as a debug-only
// ordering check distinct from whether an override value is actually
// non-empty.
inline bool was_set() { return set_flag(); }

} // namespace accelerator_override
