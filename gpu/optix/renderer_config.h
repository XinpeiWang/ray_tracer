#pragma once

// Suppress MSVC warnings about getenv being "unsafe"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996) // 'getenv': This function or variable may be unsafe
#endif

#include <cstdlib>
#include <string>
#include <iostream>

namespace optix_renderer {

/**
 * @brief Configuration for renderer behavior and mode selection
 */
struct RendererConfig {
	// Path tracing mode
	enum class Mode {
		RECURSIVE,   // Traditional recursive path tracing
		WAVEFRONT,   // Queue-based wavefront path tracing (future)
		AUTO         // Auto-select based on scene complexity
	};

	// Fallback policy when preferred mode fails
	enum class FallbackPolicy {
		NONE,           // Fail if preferred mode unavailable
		TO_RECURSIVE,   // Always fall back to recursive
		TO_FASTEST      // Fall back to fastest available mode
	};

	// Configuration members
	Mode mode = Mode::RECURSIVE;  // Default to recursive (current working implementation)
	FallbackPolicy fallback = FallbackPolicy::TO_RECURSIVE;
	bool verbose = false;  // Enable detailed logging
	bool validate = true;  // Validate mode availability before use

	/**
	 * @brief Load configuration from environment variables
	 * 
	 * Environment variables:
	 * - RAY_TRACER_MODE: "recursive", "wavefront", or "auto"
	 * - RAY_TRACER_FALLBACK: "none", "recursive", or "fastest"
	 * - RAY_TRACER_VERBOSE: "1" or "0"
	 */
	static RendererConfig fromEnvironment() {
		RendererConfig config;

		// Check mode
		const char* mode_env = std::getenv("RAY_TRACER_MODE");
		if (mode_env) {
			std::string mode_str(mode_env);
			if (mode_str == "recursive") {
				config.mode = Mode::RECURSIVE;
			} else if (mode_str == "wavefront") {
				config.mode = Mode::WAVEFRONT;
			} else if (mode_str == "auto") {
				config.mode = Mode::AUTO;
			} else {
				std::cerr << "[Config] Warning: Unknown RAY_TRACER_MODE='" 
						  << mode_str << "', using default (recursive)\n";
			}
		}

		// Check fallback policy
		const char* fallback_env = std::getenv("RAY_TRACER_FALLBACK");
		if (fallback_env) {
			std::string fb_str(fallback_env);
			if (fb_str == "none") {
				config.fallback = FallbackPolicy::NONE;
			} else if (fb_str == "recursive") {
				config.fallback = FallbackPolicy::TO_RECURSIVE;
			} else if (fb_str == "fastest") {
				config.fallback = FallbackPolicy::TO_FASTEST;
			}
		}

		// Check verbose
		const char* verbose_env = std::getenv("RAY_TRACER_VERBOSE");
		if (verbose_env && std::string(verbose_env) == "1") {
			config.verbose = true;
		}

		return config;
	}

	/**
	 * @brief Create default configuration
	 */
	static RendererConfig defaultConfig() {
		RendererConfig config;
		config.mode = Mode::RECURSIVE;
		config.fallback = FallbackPolicy::TO_RECURSIVE;
		config.verbose = false;
		config.validate = true;
		return config;
	}

	/**
	 * @brief Log configuration to stdout
	 */
	void log() const {
		if (!verbose) return;

		std::cout << "[Config] Renderer Configuration:\n";
		std::cout << "  Mode: " << modeToString(mode) << "\n";
		std::cout << "  Fallback: " << fallbackToString(fallback) << "\n";
		std::cout << "  Verbose: " << (verbose ? "enabled" : "disabled") << "\n";
		std::cout << "  Validate: " << (validate ? "enabled" : "disabled") << "\n";
	}

	/**
	 * @brief Convert mode to string
	 */
	static const char* modeToString(Mode m) {
		switch (m) {
			case Mode::RECURSIVE: return "RECURSIVE";
			case Mode::WAVEFRONT: return "WAVEFRONT";
			case Mode::AUTO: return "AUTO";
			default: return "UNKNOWN";
		}
	}

	/**
	 * @brief Convert fallback policy to string
	 */
	static const char* fallbackToString(FallbackPolicy fb) {
		switch (fb) {
			case FallbackPolicy::NONE: return "NONE";
			case FallbackPolicy::TO_RECURSIVE: return "TO_RECURSIVE";
			case FallbackPolicy::TO_FASTEST: return "TO_FASTEST";
			default: return "UNKNOWN";
		}
	}
};

/**
 * @brief Global configuration instance
 * 
 * This can be set once at program startup and used throughout.
 * Default: RECURSIVE mode with TO_RECURSIVE fallback.
 */
extern RendererConfig g_rendererConfig;

/**
 * @brief Initialize global configuration from environment
 * 
 * Call this once at program startup, before creating renderers.
 */
inline void initializeRendererConfig() {
	g_rendererConfig = RendererConfig::fromEnvironment();
	g_rendererConfig.log();
}

} // namespace optix_renderer

#ifdef _MSC_VER
#pragma warning(pop)
#endif
