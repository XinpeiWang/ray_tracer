#pragma once
// light_sampler_resolution.h -- resolves --lightsampler's CLI value plus a
// scene's own recommendation into what to actually construct and what (if
// anything) to print, for cpu_interface.cpp's cpu_render_main(). Pulled out
// into its own pure, header-only function (rather than left inline) so it's
// unit-testable without a full render - see
// tests/unit/light_sampler_resolution_tests.cpp.
//
// pbrt-v4 Integrator "string lightsampler" is advisory only, same "CLI
// always decides, scene's own request only feeds a mismatch warning" shape
// as maxdepth/samplerType (a light sampler's choice affects
// convergence/variance, not the converged image - see
// pbrt_scene::Scene::lightSamplerType's own comment). "bvh" is both
// pbrt-v4's own real default AND this project's own prior hardcoded choice,
// so an empty --lightsampler reports/behaves the same as before this
// resolution existed.
//
// "auto" is the one explicit value that does NOT mean "use this literal
// implementation" - it means "use whatever the scene's own Integrator
// lightsampler parameter requested, whatever that turns out to be" (bvh if
// it made no request, or made one this project doesn't implement).
// Deliberately opt-in rather than the default for a bare
// --lightsampler-less invocation - an explicit choice (or none at all,
// "bvh") still always wins otherwise, unchanged from before "auto" existed
// - "auto" is just what lets a CLI user ask for the scene's own request
// instead, the same convenience the GUI's "Apply recommended settings"
// button already gives GUI users for this exact setting.

#include <string>

namespace light_sampler_resolution {

struct Result {
	// What to actually construct: "uniform", "power", or "bvh" - matches
	// cpu_interface.cpp's own light_sampler_choice if/else-if/else exactly,
	// so any value other than "uniform"/"power" here means "bvh".
	std::string choice;
	// Non-empty when the CLI-less path should warn that the scene wanted
	// something it isn't getting.
	std::string mismatch_warning;
	// Non-empty when --lightsampler auto actually picked something other
	// than bvh, worth confirming.
	std::string auto_info;
};

// `scene_recommendation` is unvalidated text taken verbatim from a .pbrt
// file's own Integrator "string lightsampler" parameter (pbrt_scene.h's own
// parsing has no allowlist) - it can be a real pbrt-v4 option this project
// doesn't implement (e.g. "exhaustive"), a typo, or different case than
// expected. Only "uniform"/"power" are real, actionable alternatives to the
// bvh default; anything else (including an unimplemented request) resolves
// to "bvh" the same as an empty recommendation, and neither message below
// ever claims a request is honored or actionable when it actually isn't -
// there's nothing useful to say about a recommendation this project can't
// honor either way, so both messages simply stay silent for one.
inline Result resolve(const std::string& scene_id, const std::string& scene_recommendation,
					   bool has_explicit_cli_value, const std::string& cli_value) {
	Result result;
	const bool wants_auto = has_explicit_cli_value && cli_value == "auto";
	const bool scene_recommends_implemented =
		scene_recommendation == "uniform" || scene_recommendation == "power";

	if (!has_explicit_cli_value) {
		result.choice = "bvh";
		if (scene_recommends_implemented) {
			result.mismatch_warning =
				"Warning: scene '" + scene_id + "' requests Integrator lightsampler \"" + scene_recommendation +
				"\" but no --lightsampler was passed, so this render uses bvh - "
				"pass --lightsampler " + scene_recommendation + " (or --lightsampler auto) explicitly if that's what the scene wants.\n";
		}
	} else if (wants_auto) {
		result.choice = scene_recommends_implemented ? scene_recommendation : "bvh";
		if (scene_recommends_implemented) {
			result.auto_info =
				"[cpu_interface] --lightsampler auto: scene '" + scene_id + "' requested Integrator lightsampler \"" +
				scene_recommendation + "\", using it.\n";
		}
	} else {
		result.choice = cli_value;
	}
	return result;
}

} // namespace light_sampler_resolution
