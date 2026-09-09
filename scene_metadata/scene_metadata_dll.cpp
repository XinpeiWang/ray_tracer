// scene_metadata_dll.cpp -- Thin DLL boundary exposing cpu_renderer's
// scene-metadata C API (cpu_interface.h) to toolchains that can't link
// cpu_renderer.lib directly - specifically the MinGW-built Qt GUI (see
// qt_gui/RayTracerGUI.pro), which is ABI-incompatible with this MSVC-built
// static lib. x64 Windows has a single calling convention regardless of
// compiler, so a MinGW binary CAN call these extern "C" exports via
// LoadLibrary/GetProcAddress even though it can't link the .lib directly.
//
// Every export here is a direct passthrough to cpu_interface.h - no logic
// lives in this file. Add a new export here (and to
// qt_gui/scene_metadata_client.h/.cpp) whenever the GUI needs another piece
// of scene_registry.h's data instead of duplicating it locally.
//
// Note: this links the whole cpu_renderer.lib, not just these two
// functions - cpu_interface.cpp is one translation unit containing both
// this metadata lookup AND cpu_render_main (the full CPU path tracer), so
// the linker pulls in all of it, including every scene builder's
// std::function entry in scene_registry.h's table. The resulting DLL
// (~400KB) is small enough that this hasn't mattered in practice; splitting
// cpu_interface.cpp would be the fix if it ever does.
#include "../cpu_renderer/cpu_interface.h"

// Every export below wraps its cpu_interface.h call in try/catch: this file
// is the actual cross-toolchain boundary (an extern "C" function called via
// LoadLibrary/GetProcAddress from a differently-compiled caller - see this
// file's own header comment above), and cpu_interface.cpp's find_scene()
// constructs a std::string from the caller-supplied scene_id internally. A
// std::bad_alloc (or any other exception) thrown there and left to unwind
// across this boundary is undefined behavior - the caller's stack isn't
// compiled to unwind frames built by this DLL's compiler, and vice versa.
// In practice this is astronomically unlikely (a single small string
// allocation), but it's a one-line-per-function fix once identified, so
// there's no reason to leave it as a latent gap. Fallback values match
// cpu_interface.cpp's own existing "scene not found" convention for each
// field exactly (empty string for const char*, not nullptr - callers on the
// GUI side never null-check these), so a caller can't tell the difference
// between "scene_id not found" and "an exception was thrown looking it up".

// __declspec(dllexport) is MSVC/MinGW-only; every other platform (this file
// builds as scene_metadata.dylib/.so via the new root CMakeLists.txt, loaded
// through qt_gui/scene_metadata_client.cpp's dlopen()/dlsym() branch there)
// exports symbols by default at ELF/Mach-O visibility "default" instead -
// __attribute__((visibility("default"))) makes that explicit rather than
// relying on the compiler's default (which can be flipped to hidden by a
// -fvisibility=hidden build flag elsewhere in the project).
#if defined(_WIN32)
#define SCENE_METADATA_API extern "C" __declspec(dllexport)
#else
#define SCENE_METADATA_API extern "C" __attribute__((visibility("default")))
#endif

SCENE_METADATA_API int scene_metadata_gpu_compatible(const char* scene_id) {
	try {
		return cpu_scene_gpu_compatible_by_id(scene_id);
	} catch (...) {
		return 0;
	}
}

SCENE_METADATA_API int scene_metadata_recommended_camera(const char* scene_id,
	double* lookfrom_x, double* lookfrom_y, double* lookfrom_z,
	double* lookat_x, double* lookat_y, double* lookat_z) {
	try {
		return cpu_scene_recommended_camera(scene_id, lookfrom_x, lookfrom_y, lookfrom_z,
		                                     lookat_x, lookat_y, lookat_z);
	} catch (...) {
		return 0;
	}
}

// Everything below serves the presentational fields (name/description/
// performance/recommended_spp/requires_files) that used to live in the
// GUI's own duplicated src/shared/scene_descriptor.h table - see that
// header's comment for why it was removed. scene_ids are category letter +
// number now (e.g. "B10"), not contiguous ints - a caller enumerating the
// whole registry starts at scene_metadata_id_at_index(0..count()-1) to get
// each scene's id, then looks up its other fields by that id below.
SCENE_METADATA_API int scene_metadata_count() {
	try {
		return cpu_scene_count();
	} catch (...) {
		return 0;
	}
}

SCENE_METADATA_API const char* scene_metadata_id_at_index(int index) {
	try {
		return cpu_scene_id(index);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_name(const char* scene_id) {
	try {
		return cpu_scene_name_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_category(const char* scene_id) {
	try {
		return cpu_scene_category_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_description(const char* scene_id) {
	try {
		return cpu_scene_description_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_performance(const char* scene_id) {
	try {
		return cpu_scene_performance_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API int scene_metadata_recommended_spp(const char* scene_id) {
	try {
		return cpu_scene_recommended_spp_by_id(scene_id);
	} catch (...) {
		return 100;
	}
}

SCENE_METADATA_API int scene_metadata_requires_files(const char* scene_id) {
	try {
		return cpu_scene_requires_files_by_id(scene_id);
	} catch (...) {
		return 0;
	}
}

SCENE_METADATA_API const char* scene_metadata_recommended_integrator(const char* scene_id) {
	try {
		return cpu_scene_recommended_integrator_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_recommended_sampler(const char* scene_id) {
	try {
		return cpu_scene_recommended_sampler_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API const char* scene_metadata_recommended_light_sampler(const char* scene_id) {
	try {
		return cpu_scene_recommended_light_sampler_by_id(scene_id);
	} catch (...) {
		return "";
	}
}

SCENE_METADATA_API double scene_metadata_recommended_exposure(const char* scene_id) {
	try {
		return cpu_scene_recommended_exposure_by_id(scene_id);
	} catch (...) {
		return 1.0;
	}
}

// Bundles every field above, plus the same camera fields
// scene_metadata_recommended_camera() returns, into one call - for callers
// that want more than a couple of fields at once. See SceneMetadataSnapshot's
// own comment (cpu_interface.h) for why this exists alongside, not instead
// of, every single-field export above.
SCENE_METADATA_API int scene_metadata_snapshot(const char* scene_id, SceneMetadataSnapshot* out) {
	try {
		return cpu_scene_metadata_snapshot(scene_id, out);
	} catch (...) {
		return 0;
	}
}
