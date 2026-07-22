#include "path_tracing_strategy.h"
#include "recursive_path_tracer.h"
#include "wavefront_path_tracer.h"
#include "renderer_config.h"
#include <iostream>
#include <memory>

namespace optix_renderer {

std::unique_ptr<PathTracingStrategy> createPathTracingStrategy(
	PathTracingMode mode,
	bool enableFallback
) {
	std::unique_ptr<PathTracingStrategy> strategy;

	// Try to create the requested mode
	switch (mode) {
		case PathTracingMode::RECURSIVE:
			std::cout << "[Factory] Creating RecursivePathTracer strategy\n";
			strategy = std::make_unique<RecursivePathTracer>();
			break;

		case PathTracingMode::WAVEFRONT:
			std::cout << "[Factory] Creating WavefrontPathTracer strategy\n";
			strategy = std::make_unique<WavefrontPathTracer>();
			// Wavefront not yet implemented - fall back
			if (enableFallback) {
				std::cout << "[Factory] Wavefront not implemented, falling back to RECURSIVE\n";
				strategy = std::make_unique<RecursivePathTracer>();
			} else {
				std::cerr << "[Factory] WAVEFRONT mode not implemented and fallback disabled\n";
				strategy = nullptr;
			}
			break;

		default:
			std::cerr << "[Factory] Unknown PathTracingMode\n";
			if (enableFallback) {
				std::cout << "[Factory] Defaulting to RECURSIVE mode\n";
				strategy = std::make_unique<RecursivePathTracer>();
			}
			break;
	}

	return strategy;
}

const char* pathTracingModeToString(PathTracingMode mode) {
	switch (mode) {
		case PathTracingMode::RECURSIVE: return "RECURSIVE";
		case PathTracingMode::WAVEFRONT: return "WAVEFRONT";
		default: return "UNKNOWN";
	}
}

} // namespace optix_renderer
