#pragma once
// backend_capabilities.h -- which render options each renderer really honours, in ONE table.
//
// The launcher warns about a flag the chosen renderer ignores ("--sampler has no effect under --gpu ..."), and the GUI greys out the matching control. Both
// used to carry their own copy of the rules: the launcher a list of near-identical `if (flag && (use_gpu || use_bdpt || ...))` blocks, the GUI one long
// hand-written condition per control (with a comment per control about which backend reads what). The two disagreed in places - the Metal renderer reads
// the adaptive-sampling options, but both the launcher and the GUI treated every GPU as not reading them. Now both ask supports().
//
// "Renderer" is what actually runs, not just "GPU or CPU": the CPU's default path tracer reads more options than BDPT, MLT, SPPM and the debug
// integrators do, and the two GPU backends differ from each other. docs/BACKEND_SUPPORT.md shows the same facts as a table for people.
//
// Plain std-only header: usable from the launcher, the GUI and the unit tests.

namespace backend_caps {

enum class Renderer {
	CpuDefault,          // the CPU path tracer
	CpuOtherIntegrator,  // BDPT, MLT, SPPM, random walk, AO, simple path, simple volume path, light path
	GpuOptix,            // Windows, recursive or wavefront
	GpuMetal,            // macOS
};

enum class Option {
	Sampler,           // --sampler
	LightSampler,      // --lightsampler
	Spectral,          // --spectral
	AdaptiveSampling,  // --adaptive, --adaptive-threshold
	TimeLimit,         // --time-limit
	Regularize,        // --regularize
	Denoise,           // --denoise, --denoise-blend
	DofOverride,       // --aperture, --focus-distance
	MaxComponentValue, // --maxcomponentvalue
	Crop,              // --crop
	Seed,              // --seed
	Exposure,          // --exposure
	Tonemap,           // --tonemap
	Accelerator,       // --accelerator, --splitmethod
	OptixValidate,     // --optix-validate
};

// The command-line flag (or flags) an option is set with, for messages.
inline const char* flagName(Option o) {
	switch (o) {
		case Option::Sampler: return "--sampler";
		case Option::LightSampler: return "--lightsampler";
		case Option::Spectral: return "--spectral";
		case Option::AdaptiveSampling: return "--adaptive";
		case Option::TimeLimit: return "--time-limit";
		case Option::Regularize: return "--regularize";
		case Option::Denoise: return "--denoise";
		case Option::DofOverride: return "--aperture/--focus-distance";
		case Option::MaxComponentValue: return "--maxcomponentvalue";
		case Option::Crop: return "--crop";
		case Option::Seed: return "--seed";
		case Option::Exposure: return "--exposure";
		case Option::Tonemap: return "--tonemap";
		case Option::Accelerator: return "--accelerator/--splitmethod";
		case Option::OptixValidate: return "--optix-validate";
	}
	return "";
}

// Whether `renderer` reads `option`. An option it does not read is accepted and ignored (the launcher says so; the GUI shows the control disabled).
inline bool supports(Renderer renderer, Option option) {
	const bool cpu = renderer == Renderer::CpuDefault || renderer == Renderer::CpuOtherIntegrator;
	const bool gpu = renderer == Renderer::GpuOptix || renderer == Renderer::GpuMetal;
	const bool defaultPathTracer = renderer != Renderer::CpuOtherIntegrator;   // every GPU renderer is a plain path tracer too
	switch (option) {
		case Option::Sampler:
		case Option::LightSampler:
		case Option::Spectral:
		case Option::TimeLimit:
			return renderer == Renderer::CpuDefault;
		case Option::AdaptiveSampling:
			return renderer == Renderer::CpuDefault || renderer == Renderer::GpuMetal;
		case Option::Regularize:
			return renderer == Renderer::CpuDefault || renderer == Renderer::GpuOptix;
		case Option::Denoise:   // OptiX's AI denoiser, or Open Image Denoise (CPU default path tracer, Metal) when its library is installed
			return renderer == Renderer::CpuDefault || gpu;
		case Option::DofOverride:   // the CPU has its own depth-of-field support; of the GPUs only OptiX takes the override
			return cpu || renderer == Renderer::GpuOptix;
		case Option::MaxComponentValue:
		case Option::Crop:
		case Option::Seed:
		case Option::Exposure:
		case Option::Tonemap:
			return defaultPathTracer;
		case Option::Accelerator:   // scene construction, shared by every CPU integrator; a GPU always builds its own BVH
			return cpu;
		case Option::OptixValidate:
			return renderer == Renderer::GpuOptix;
	}
	return false;
}

// What to say when a requested option is not read, e.g. "--sampler has no effect under the Metal renderer - ignoring."
inline const char* rendererName(Renderer r) {
	switch (r) {
		case Renderer::CpuDefault: return "the CPU path tracer";
		case Renderer::CpuOtherIntegrator: return "--bdpt/--mlt/--sppm/--randomwalk/--ao/--simplepath/--simplevolpath/--lightpath";
		case Renderer::GpuOptix: return "the OptiX renderer";
		case Renderer::GpuMetal: return "the Metal renderer";
	}
	return "";
}

}  // namespace backend_caps
