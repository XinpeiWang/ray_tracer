// OptiX Interface Implementation
// C API wrapper matching gpu/cuda/gpu_interface.h signature

#include "optix_interface.h"
#include "optix_renderer.h"
#include "scene_builder.h"
#include "../../src/TheRestOfYourLife/error_codes.h"
#include "../../src/shared/tone_map.h"
#include "../../src/shared/exr_writer.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>

// Global renderer instance
static std::unique_ptr<OptiXRenderer> g_renderer;

// Which scene_id is currently uploaded to the GPU (materials/geometry/BVH/SBT
// all live in g_renderer's device memory, keyed only by scene_id - see the
// skip-reupload check below). "" = nothing uploaded yet this process (never
// a valid id - every real id has a category letter and a number).
static std::string g_uploaded_scene_id;

extern "C" bool optix_is_available() {
	return OptiXRenderer::isAvailable();
}

extern "C" bool optix_get_diagnostics(OptixDiagnostics* out) {
	if (!out) return false;
	return OptiXRenderer::getDiagnostics(*out);
}

extern "C" int optix_render_main(
	int image_width,
	int image_height,
	int samples_per_pixel,
	int max_depth,
	const char* output_path,
	const char* scene_id,
	double cam_x,
	double cam_y,
	double cam_z,
	int force_camera_override,
	const RenderOptions& options
) {
	try {
		// std::string(output_path) below (and the ofstream open further down)
		// are both undefined behavior on a null pointer - same guard as
		// cpu_render_main's (cpu_interface.cpp).
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << "\n";
			return ERR_OUTPUT_PATH_INVALID;
		}

		// options.tonemap==nullptr (every existing caller that predates this
		// param) or an unrecognized name both fall back to ACES - see
		// tone_map_mode_from_name()'s own comment (src/shared/tone_map.h).
		// Parsed once here and used by both PPM-writing call sites below
		// (the single shared output path for the recursive and wavefront
		// backends - see their own comment).
		ToneMapMode tone_map_mode = ToneMapMode::ACES;
		if (options.tonemap != nullptr) {
			if (!tone_map_mode_from_name(options.tonemap, tone_map_mode) && options.tonemap[0] != '\0') {
				std::cerr << "Warning: unrecognized --tonemap \"" << options.tonemap
						  << "\", using aces. Valid: aces, reinhard, none.\n";
				tone_map_mode = ToneMapMode::ACES;
			}
		}

		// Initialize renderer on first call
		if (!g_renderer) {
			std::cout << "[OptiX] Initializing renderer...\n";
			g_renderer = std::make_unique<OptiXRenderer>();
			if (!g_renderer->initialize()) {
				std::cerr << "[OptiX] Failed to initialize renderer\n";
				return ERR_GPU_DEVICE_INIT_FAILED;
			}
		}

		// Build scene with specified scene_id
		std::cout << "[OptiX] Building scene " << scene_id << "...\n";
		std::cout << "[OptiX] Camera position: (" << cam_x << ", " << cam_y << ", " << cam_z << ")\n";

		SceneData scene;
		float camera_params[12];  // origin(3) + lower_left(3) + horizontal(3) + vertical(3)
		GpuCameraParams cameraExtra{};  // zero-init: kind=Perspective, DOF/spherical fields all zero

		if (!build_scene(scene_id, image_width, image_height, scene, camera_params, cam_x, cam_y, cam_z, &cameraExtra, force_camera_override != 0)) {
			// build_scene() only ever returns false for an unrecognized/
			// unimplemented scene_id (its default: case) - the other
			// return-false path (a null camera_params buffer) is
			// unreachable in practice, since every caller here passes a
			// valid on-stack array. ERR_GPU_UNSUPPORTED_SCENE gives the
			// user the actionable "switch to CPU mode" message
			// (error_handler.h); the old ERR_GPU_SCENE_BUILD_FAILED here
			// was misleading - it reads as a genuine build/geometry
			// failure, not "this scene was never ported to GPU."
			std::cerr << "[OptiX] Scene not supported on GPU\n";
			return ERR_GPU_UNSUPPORTED_SCENE;
		}

		// Scenes that don't use a non-default camera model leave cameraExtra
		// untouched by build_scene() (still zero-init'd: kind=Perspective,
		// defocus disk zero) - fill it from the plain 12-float camera_params
		// in that case, matching every scene's prior behavior. Scenes that
		// DID request a non-default model (Orthographic/Spherical, or
		// Perspective with DOF) set kind and/or a nonzero defocus disk
		// themselves in build_scene(), so this check reliably tells the two
		// cases apart.
		bool defocusDiskZero = cameraExtra.defocus_disk_u.x == 0.0f && cameraExtra.defocus_disk_u.y == 0.0f &&
								cameraExtra.defocus_disk_u.z == 0.0f;
		if (cameraExtra.kind == CameraKind::Perspective && defocusDiskZero) {
			cameraExtra.origin = make_float3(camera_params[0], camera_params[1], camera_params[2]);
			cameraExtra.lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			cameraExtra.horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			cameraExtra.vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
		}

		// buildScene() only touches geometry/material/light device memory -
		// it never sees camera state (that's render()'s cameraExtra param,
		// computed fresh above on every call regardless) - so re-uploading
		// and rebuilding the BVH/SBT is only actually necessary when the
		// scene_id changes from the last call in this process. This is what
		// makes video rendering (same scene, moving camera, many frames in
		// one process) expensive: skip it when nothing but the camera moved.
		if (scene_id != g_uploaded_scene_id) {
			// Instanced geometry travels separately - see setInstanceData().
			// Called inside this same cache-skip guard as buildScene() itself,
			// so instance data is part of what "this scene is already
			// uploaded" means: a scene switch that reuses geometry must not
			// keep a PREVIOUS scene's placements around.
			g_renderer->setInstanceData(scene.instanceTriangles, scene.instanceSpheres,
										scene.instanceGroups, scene.instancePlacements);
			if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials,
										 scene.lightIndices, scene.lightKinds,
										 scene.punctualLights, scene.bilinearPatches,
										 scene.triangles, scene.disks, scene.cylinders,
										 scene.lensElements,
										 scene.exitPupilBounds, scene.textures,
										 scene.texturePixels, scene.cloudMediums,
										 scene.rgbGridMediums, scene.rgbGridData,
										 scene.gridMediums, scene.gridData,
									 scene.bssrdfTables, scene.bssrdfRhoSamples,
									 scene.bssrdfRadiusSamples, scene.bssrdfProfile,
									 scene.bssrdfProfileCdf,
										 scene.measuredTables, scene.measuredParamValues,
										 scene.measuredData, scene.measuredMcdf,
										 scene.measuredCcdf,
										 scene.skyImagePixels, scene.skyMarginalCdf,
										 scene.skyMarginalFunc, scene.skyMarginalFuncInt,
										 scene.skyConditionalCdf, scene.skyConditionalFunc,
										 scene.skyConditionalFuncInt,
										 scene.skyWidth, scene.skyHeight, scene.skyScale)) {
				std::cerr << "[OptiX] Failed to upload scene to GPU\n";
				return ERR_GPU_MEMORY_COPY_FAILED;
			}
			g_uploaded_scene_id = scene_id;
		} else {
			std::cout << "[OptiX] Reusing already-uploaded scene " << scene_id << " (skipping GPU rebuild)\n";
		}

		// Enable/disable wavefront mode based on env var RAY_TRACER_WAVEFRONT=1.
		// g_renderer is a process-lifetime singleton (see g_uploaded_scene_id's
		// own comment above), so OptiXRenderer::useWavefront_ is a member that
		// persists across calls just like the device buffers scene switches
		// already have to reset - reading the env var and calling
		// enableWavefront(true, ...) ONLY when it's "1", with no else branch,
		// left useWavefront_ latched true forever after the first call that
		// requested it: a LATER call with the env var unset/"0" (every caller
		// that believes it is asking for plain recursive-mode rendering, e.g.
		// a "GPU-recursive-only" pass explicitly meant to exclude wavefront
		// entirely) never called enableWavefront(false, ...) to say so, so
		// OptiXRenderer::render() kept routing to wavefrontTracer_->render()
		// regardless. This silently ran scenes/backends never exercised
		// together in wavefront mode, which is likely the real explanation
		// behind this codebase's own "GPU-recursive-only pass still hits the
		// CUDA-700 corruption" finding (material_cpu_gpu_parity_tests.cpp's
		// file header comment) - those "recursive-only" renders may actually
		// have been running under wavefront the whole time, once any earlier
		// call in the same process (e.g. wavefront_tests.cpp) had set the env
		// var to "1" once. Call enableWavefront() unconditionally on both
		// branches so this render's own request is what decides the mode,
		// every time, not whatever the last caller that asked for wavefront
		// left behind.
#pragma warning(suppress: 4996)
		const char* wfEnv = std::getenv("RAY_TRACER_WAVEFRONT");
		if (wfEnv && std::string(wfEnv) == "1") {
			// Derive PTX path: same directory as output_path, or executable directory
			std::string ptxPath;
			std::string outStr(output_path);
			size_t sep = outStr.rfind('\\');
			if (sep == std::string::npos) sep = outStr.rfind('/');
			if (sep != std::string::npos)
				ptxPath = outStr.substr(0, sep + 1) + "wavefront_programs.ptx";
			else
				ptxPath = "wavefront_programs.ptx";
			std::cout << "[OptiX] Wavefront mode enabled (PTX: " << ptxPath << ")\n";
			g_renderer->enableWavefront(true, ptxPath);
		} else {
			g_renderer->enableWavefront(false);
		}

		// g_renderer is a process-lifetime singleton (see enableWavefront()'s
		// call above and g_uploaded_scene_id's own comment) - called
		// unconditionally, same reasoning as enableWavefront(): this render's
		// own `options.denoise` must be what decides the mode, not whatever
		// an earlier call in this process happened to request.
		g_renderer->enableDenoise(options.denoise);

		// enableDenoise(true) has no effect under wavefront mode (render()
		// delegates to wavefrontTracer_->render() and returns before ever
		// reaching the denoise step - see OptiXRenderer::render()'s own
		// comment and enableDenoise()'s doc comment for why). Without this,
		// a user combining --wavefront --denoise gets a normal, undenoised
		// render with nothing telling them why.
		if (options.denoise && wfEnv && std::string(wfEnv) == "1") {
			std::cerr << "[OptiX] Warning: --denoise has no effect under --wavefront "
						 "(wavefront backend does not support denoising) - rendering without it.\n";
		}

		// Allocate float framebuffer
		size_t pixelCount = image_width * image_height;
		std::vector<float> framebuffer(pixelCount * 3);

		// Technique summary
		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : OptiX mega-kernel path tracer  (raygen + closest-hit + miss programs)" << std::endl;
		std::cout << "[TECH] Sampler        : PCG hash-based pseudo-random (per-pixel seed, per-bounce offset)" << std::endl;
		std::cout << "[TECH] Reconstruction : Box filter (1 sample per pixel sub-region, averaged in kernel)" << std::endl;
		std::cout << "[TECH] Acceleration   : OptiX BVH  |  custom AABB primitives  (spheres + quads)" << std::endl;
		std::cout << "[TECH] Light sampling : Power-weighted alias table (Vose method)  phi = area * Le * pi" << std::endl;
		std::cout << "[TECH] MIS            : Power heuristic  beta=2  (BSDF sample + NEE light sample)" << std::endl;
		std::cout << "[TECH] Path termination: fixed max_depth=" << max_depth << "  (no Russian Roulette on GPU)" << std::endl;
		std::cout << "[TECH] Tone mapping   : " << tone_map_mode_display_name(tone_map_mode) << " + sRGB OETF  (matches CPU's write_color())" << std::endl;
		std::cout << "[TECH] Device         : CUDA / OptiX 7+  (NVIDIA GPU)" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		// Render
		std::cout << "[OptiX] Rendering...\n";

		if (!g_renderer->render(
			image_width,
			image_height,
			samples_per_pixel,
			max_depth,
			cameraExtra,
			framebuffer.data()
		)) {
			std::cerr << "[OptiX] Render failed\n";
			return ERR_GPU_RENDER_FAILED;
		}

		// Auto-detected from the caller's requested extension, matching
		// cpu_interface.cpp's own exr_output detection - see camera.h's
		// exr_output field comment for the rationale (extension, not a
		// separate flag, mirrors how PNG conversion already auto-triggers
		// off the extension in launcher/main.cpp).
		const bool exrOutput = is_exr_output_path(output_path);

		if (exrOutput) {
			// Linear, pre-tonemap radiance straight off the GPU (`framebuffer`
			// above) - no display tone-map applied, matching camera.h's own
			// exr_output path and EXR's whole reason for existing (preserving
			// HDR range for downstream compositing, not baking in a display
			// transform). exposure IS applied here though (a linear multiply,
			// not a tonemap) so it stays in parity with CPU's exr_output path
			// (camera.h applies `pixel_color * exposure` unconditionally,
			// before branching on exr_output) - and the same NaN/Inf firefly
			// guard the PPM loop below and CPU's per-sample guard both apply,
			// so a stray firefly doesn't land unfiltered in the saved EXR.
			// Unlike CPU's convoluted temp-file-then-copy architecture
			// (camera.h render()), this backend already writes straight to
			// output_path, so no fallback-path juggling is needed.
			for (size_t i = 0; i < pixelCount * 3; ++i) {
				float &v = framebuffer[i];
				if (!std::isfinite(v)) v = 0.0f;
				v = static_cast<float>(v * options.exposure);
			}

			std::string exrError;
			if (!write_exr_image(output_path, framebuffer.data(), image_width, image_height, exrError)) {
				std::cerr << "[OptiX] Failed to write EXR '" << output_path << "': " << exrError << "\n";
				return ERR_FILE_WRITE_FAILED;
			}
			std::cout << "[OptiX] Wrote EXR: " << output_path << "\n";

			// Albedo/normal AOV export - only possible when denoise was
			// requested, since that is what allocates/fills d_albedoAov_/
			// d_normalAov_ in the first place (ensureAovBuffers(), called
			// from render()'s own denoise step - see readAovBuffers()'s own
			// comment). Silently skipped (not an error) otherwise: AOVs are
			// a bonus on top of the beauty EXR this function already wrote
			// successfully, not something --exr alone promises. Wrapped in
			// its own try/catch so a transient failure here (readAovBuffers()
			// throws via CUDA_CHECK on any device-side error) is reported and
			// skipped rather than escalating into ERR_GPU_EXCEPTION for a
			// render whose actual requested output (the beauty EXR above)
			// already succeeded.
			if (options.denoise && !(wfEnv && std::string(wfEnv) == "1")) {
				try {
					std::vector<float> albedo, normal;
					if (g_renderer->readAovBuffers(static_cast<unsigned int>(image_width),
							static_cast<unsigned int>(image_height), albedo, normal)) {
						const std::filesystem::path base(output_path);
						const std::filesystem::path albedoPath =
							base.parent_path() / (base.stem().string() + "_albedo.exr");
						const std::filesystem::path normalPath =
							base.parent_path() / (base.stem().string() + "_normal.exr");
						std::string aErr, nErr;
						if (write_exr_image(albedoPath.string(), albedo.data(), image_width, image_height, aErr))
							std::cout << "[OptiX] Wrote AOV: " << albedoPath.string() << "\n";
						else
							std::cerr << "[OptiX] Failed to write albedo AOV '" << albedoPath.string() << "': " << aErr << "\n";
						if (write_exr_image(normalPath.string(), normal.data(), image_width, image_height, nErr))
							std::cout << "[OptiX] Wrote AOV: " << normalPath.string() << "\n";
						else
							std::cerr << "[OptiX] Failed to write normal AOV '" << normalPath.string() << "': " << nErr << "\n";
					}
				} catch (const std::exception &e) {
					std::cerr << "[OptiX] AOV export failed (beauty EXR above is still valid): " << e.what() << "\n";
				}
			}

			return 0;  // Success
		}

		// Write to PPM file
		std::ofstream outFile(output_path, std::ios::binary);
		if (!outFile) {
			std::cerr << "[OptiX] Failed to open output file: " << output_path << "\n";
			return ERR_FILE_WRITE_FAILED;
		}

		// PPM header
		outFile << "P3\n" << image_width << " " << image_height << "\n255\n";

		// Write pixels with the same ACES filmic tone map + sRGB OETF CPU's
		// write_color() (src/TheRestOfYourLife/color.h) uses - previously
		// this was a plain sqrt gamma-2.0 approximation, the exact thing
		// tone_map.h's own header comment says CPU "replaced" at some point
		// without GPU's output path being updated to match, which showed up
		// as a real, visible color/contrast mismatch between the two
		// renderers (ACES's filmic S-curve shifts shadow/midtone hue and
		// softens contrast compared to a flat gamma curve).
		for (size_t i = 0; i < pixelCount; ++i) {
			double r = framebuffer[i * 3 + 0];
			double g = framebuffer[i * 3 + 1];
			double b = framebuffer[i * 3 + 2];

			// NaN/Inf guard (firefly guard), matching write_color() exactly.
			if (!std::isfinite(r)) r = 0.0;
			if (!std::isfinite(g)) g = 0.0;
			if (!std::isfinite(b)) b = 0.0;

			// Flat exposure multiply on linear color, right before tone-
			// mapping - see optix_render_main's own exposure parameter
			// comment. 1.0 (default) is a no-op. Covers both the recursive
			// and wavefront backends (this loop is the single shared output
			// path for both, per this function's own file comment above).
			r *= options.exposure;
			g *= options.exposure;
			b *= options.exposure;

			r = linear_to_srgb(apply_tone_map(r, tone_map_mode));
			g = linear_to_srgb(apply_tone_map(g, tone_map_mode));
			b = linear_to_srgb(apply_tone_map(b, tone_map_mode));

			// Clamp and convert to byte (matches write_color()'s
			// interval(0.000, 0.999) clamp-then-*256 convention exactly).
			// fmin/fmax, not std::min/std::max: this TU pulls in <windows.h>
			// transitively (via optix.h/cuda headers) without NOMINMAX, and
			// std::min/std::max collide with its min/max macros here.
			int ir = static_cast<int>(256.0 * std::fmin(std::fmax(r, 0.0), 0.999));
			int ig = static_cast<int>(256.0 * std::fmin(std::fmax(g, 0.0), 0.999));
			int ib = static_cast<int>(256.0 * std::fmin(std::fmax(b, 0.0), 0.999));

			outFile << ir << " " << ig << " " << ib << "\n";
		}

		outFile.close();

		std::cout << "[OptiX] Render complete! Output: " << output_path << "\n";
		return 0;  // Success

	} catch (const std::exception& e) {
		std::cerr << "[OptiX] Exception: " << e.what() << "\n";
		return ERR_GPU_EXCEPTION;
	} catch (...) {
		std::cerr << "[OptiX] Unknown error\n";
		return ERR_GPU_UNKNOWN_ERROR;
	}
}

// Human-readable name for the MaterialTypes this function's error messages
// can name -- deliberately NOT exhaustive (optix_types.h's MaterialType has
// ~20 enumerators); anything past this list falls back to printing the raw
// int, which is enough to look up in optix_types.h when it happens.
static const char* sppm_gpu_material_type_name(MaterialType t) {
	switch (t) {
	case MaterialType::Lambertian:          return "Lambertian";
	case MaterialType::Metal:               return "Metal";
	case MaterialType::Dielectric:          return "Dielectric";
	case MaterialType::DiffuseLight:        return "DiffuseLight";
	case MaterialType::RoughDielectric:     return "RoughDielectric";
	case MaterialType::Conductor:           return "Conductor";
	case MaterialType::CoatedDiffuse:       return "CoatedDiffuse";
	case MaterialType::ThinDielectric:      return "ThinDielectric";
	case MaterialType::CoatedConductor:     return "CoatedConductor";
	case MaterialType::DiffuseTransmission: return "DiffuseTransmission";
	case MaterialType::NormalizedFresnel:   return "NormalizedFresnel";
	case MaterialType::Medium:              return "Medium";
	case MaterialType::Hair:                return "Hair";
	case MaterialType::Interface:           return "Interface";
	default:                                return "<other>";
	}
}

// GPU SPPM scene-capability check -- inspects the ALREADY-BUILT scene (not
// just its id string) so this stays correct automatically as new scenes are
// added to the registry, rather than needing a hand-maintained id allowlist.
//
// This replaces Phase 1's original `if (scene_id != "B3") reject` guard,
// which was overcautious in one respect and exactly right in another:
//   - The hash-grid/world-bounds machinery in sppm_path_tracer.cpp was
//     ALREADY scene-agnostic even under the old guard -- it computes
//     gridMin/gridMax/gridRes from a host-side readback of each iteration's
//     actual camera-pass visible points (see its own comment, "mirrors CPU
//     HashGrid::Build()'s first phase"), never a fixed B3-shaped box. So
//     that part of the restriction really was just leftover Phase 1
//     over-caution, not a real blocker -- nothing here needed to change.
//   - What WAS genuinely B3-specific: sppm_programs.cu's OptiX device
//     programs only ever implemented BSDF sampling for the exact material
//     set B3 happens to use (Lambertian walls/box, a DiffuseLight ceiling
//     quad, and a RoughDielectric glass sphere) -- any other MaterialType
//     silently fell through a "treat as Lambertian" fallback, reading that
//     material's (possibly unset, e.g. RoughDielectric's own albedo slot)
//     union field as if it were a diffuse color. Metal/Dielectric/Conductor
//     dispatch was added to sppm_programs.cu (see its own
//     sppm_is_delta_material()/sppm_sample_delta_material() comments) to
//     close that gap for the mainstream surface-BSDF cases; this function's
//     job is to keep rejecting -- with an honest, specific reason -- the
//     scenes that still hit a genuine gap: mesh/instanced/bilinear-patch
//     geometry (SPPMPathTracer::buildSBT() only has hit-group records for
//     spheres/quads), punctual or sky lighting (neither raygen samples
//     them), non-quad/sphere light shapes (the raygens' light-kind branch is
//     binary: Sphere or "else assume Quad"), and any remaining MaterialType
//     this port still has no BSDF for (CoatedDiffuse, Hair, Subsurface, ...).
//
// Returns "" if the scene is fully supported by GPU SPPM as it stands today,
// or a human-readable reason (naming the specific blocker) otherwise.
static std::string sppm_gpu_unsupported_reason(const SceneData& scene) {
	if (!scene.triangles.empty() || !scene.instanceTriangles.empty() ||
	    !scene.instanceSpheres.empty() || !scene.instanceGroups.empty()) {
		return "uses triangle-mesh and/or instanced geometry -- GPU SPPM's hash-grid "
		       "SBT (SPPMPathTracer::buildSBT()) only has hit-group records for spheres "
		       "and quads";
	}
	if (!scene.bilinearPatches.empty()) {
		return "uses bilinear-patch geometry, which GPU SPPM's OptiX programs have no "
		       "intersection/hit-group support for (spheres and quads only)";
	}
	if (!scene.disks.empty() || !scene.cylinders.empty()) {
		return "uses disk/cylinder geometry (Phase 4b, recursive backend only) -- GPU "
		       "SPPM's OptiX programs have no intersection/hit-group support for either "
		       "shape (spheres and quads only)";
	}
	if (!scene.punctualLights.empty()) {
		return "uses punctual (point/spot) lights -- GPU SPPM's camera/photon-pass "
		       "raygens only sample the scene's area-light alias table";
	}
	if (scene.skyWidth > 0 || scene.skyHeight > 0) {
		return "uses a sky/infinite light -- GPU SPPM's raygens have no sky-sampling "
		       "branch yet (matches this port's camera-pass header comment)";
	}
	for (GpuLightKind k : scene.lightKinds) {
		if (k != GpuLightKind::Quad && k != GpuLightKind::Sphere) {
			return "has an area light shape (e.g. a lone emissive triangle) GPU SPPM's "
			       "raygens don't dispatch on -- they only distinguish Sphere from "
			       "\"else assume Quad\"";
		}
	}
	for (size_t i = 0; i < scene.materials.size(); ++i) {
		const MaterialType t = scene.materials[i].type;
		// The single canonical "does GPU SPPM support this MaterialType"
		// list - see optix_types.h's sppm_gpu_material_supported() comment
		// for why this used to be a second, separately hand-maintained copy
		// of the material set sppm_programs.cu's own dispatch implements.
		if (!sppm_gpu_material_supported(t)) {
			return std::string("uses MaterialType::") + sppm_gpu_material_type_name(t) +
			       " (material index " + std::to_string(i) + ") -- sppm_programs.cu's "
			       "camera/photon passes don't implement a BSDF sampler for it yet";
		}
	}
	return "";
}

// GPU SPPM entry point (sub-phase 1e) -- mirrors optix_render_main()'s own
// scene-build/camera-setup preamble and ACES-tonemap PPM write, but calls
// OptiXRenderer::renderSPPM() (the real multi-iteration loop, sub-phase 1d)
// instead of render(). Scope is now determined dynamically, by inspecting
// the built scene's actual materials/geometry/lights (see
// sppm_gpu_unsupported_reason() above) rather than a hardcoded "only B3" id
// check -- the scene must still be BUILT first (build_scene() below) before
// that inspection is possible, so the capability check happens after the
// build, not before it like the old string-compare guard did.
extern "C" int optix_render_main_sppm(
	int image_width,
	int image_height,
	int iterations,
	int photons,
	int max_depth,
	const char* output_path,
	const char* scene_id,
	double cam_x,
	double cam_y,
	double cam_z,
	int force_camera_override
) {
	try {
		// Same null-output_path guard as optix_render_main() above.
		if (!output_path) {
			std::cerr << ErrorInfo(ERR_OUTPUT_PATH_INVALID).to_string() << "\n";
			return ERR_OUTPUT_PATH_INVALID;
		}

		std::cout << "[OptiX] Building scene " << scene_id << " (SPPM)...\n";
		std::cout << "[OptiX] Camera position: (" << cam_x << ", " << cam_y << ", " << cam_z << ")\n";

		// build_scene() is pure host-side scene construction (parses the
		// scene description, loads any mesh/.obj geometry from disk, fills
		// `scene`/`camera_params` in host memory) - it makes no OptiX/CUDA
		// calls and does not need g_renderer to exist yet, so it and the
		// capability check below both run BEFORE renderer initialization.
		// This used to run after g_renderer->initialize(), which meant an
		// unsupported scene (e.g. a mesh-heavy one rejected below for using
		// triangle geometry) paid for full GPU context/pipeline creation
		// AND a potentially large mesh load from disk before finding out it
		// was never going to be accepted - now that cost is only paid once
		// the scene is already known to be supported.
		SceneData scene;
		float camera_params[12];
		GpuCameraParams cameraExtra{};

		if (!build_scene(scene_id, image_width, image_height, scene, camera_params, cam_x, cam_y, cam_z, &cameraExtra, force_camera_override != 0)) {
			std::cerr << "[OptiX] Scene not supported on GPU\n";
			return ERR_GPU_UNSUPPORTED_SCENE;
		}

		// Dynamic capability check (see sppm_gpu_unsupported_reason()'s own
		// comment for the full "what changed from Phase 1's blunt B3-only
		// guard, and why" story) -- must happen after build_scene() (not
		// before it), since it inspects the scene's actual material/
		// geometry/light data, not just its id - but still before renderer
		// initialization, per this function's own reordering comment above.
		{
			const std::string reason = sppm_gpu_unsupported_reason(scene);
			if (!reason.empty()) {
				std::cerr << "[OptiX] GPU SPPM does not support scene " << scene_id << ": "
				          << reason << ". Use CPU SPPM (--sppm without --gpu) instead.\n";
				return ERR_GPU_UNSUPPORTED_SCENE;
			}
		}

		if (!g_renderer) {
			std::cout << "[OptiX] Initializing renderer...\n";
			g_renderer = std::make_unique<OptiXRenderer>();
			if (!g_renderer->initialize()) {
				std::cerr << "[OptiX] Failed to initialize renderer\n";
				return ERR_GPU_DEVICE_INIT_FAILED;
			}
		}

		bool defocusDiskZero = cameraExtra.defocus_disk_u.x == 0.0f && cameraExtra.defocus_disk_u.y == 0.0f &&
		                       cameraExtra.defocus_disk_u.z == 0.0f;
		if (cameraExtra.kind == CameraKind::Perspective && defocusDiskZero) {
			cameraExtra.origin = make_float3(camera_params[0], camera_params[1], camera_params[2]);
			cameraExtra.lower_left_corner = make_float3(camera_params[3], camera_params[4], camera_params[5]);
			cameraExtra.horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
			cameraExtra.vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);
		}

		if (scene_id != g_uploaded_scene_id) {
			// Instanced geometry travels separately - see setInstanceData().
			// Called inside this same cache-skip guard as buildScene() itself,
			// so instance data is part of what "this scene is already
			// uploaded" means: a scene switch that reuses geometry must not
			// keep a PREVIOUS scene's placements around.
			g_renderer->setInstanceData(scene.instanceTriangles, scene.instanceSpheres,
										scene.instanceGroups, scene.instancePlacements);
			if (!g_renderer->buildScene(scene.spheres, scene.quads, scene.materials,
			                             scene.lightIndices, scene.lightKinds,
			                             scene.punctualLights, scene.bilinearPatches,
			                             scene.triangles, scene.disks, scene.cylinders,
			                             scene.lensElements,
			                             scene.exitPupilBounds, scene.textures,
			                             scene.texturePixels, scene.cloudMediums,
										 scene.rgbGridMediums, scene.rgbGridData,
										 scene.gridMediums, scene.gridData,
									 scene.bssrdfTables, scene.bssrdfRhoSamples,
									 scene.bssrdfRadiusSamples, scene.bssrdfProfile,
									 scene.bssrdfProfileCdf,
										 scene.measuredTables, scene.measuredParamValues,
										 scene.measuredData, scene.measuredMcdf,
										 scene.measuredCcdf,
										 scene.skyImagePixels, scene.skyMarginalCdf,
										 scene.skyMarginalFunc, scene.skyMarginalFuncInt,
										 scene.skyConditionalCdf, scene.skyConditionalFunc,
										 scene.skyConditionalFuncInt,
										 scene.skyWidth, scene.skyHeight, scene.skyScale)) {
				std::cerr << "[OptiX] Failed to upload scene to GPU\n";
				return ERR_GPU_MEMORY_COPY_FAILED;
			}
			g_uploaded_scene_id = scene_id;
		} else {
			std::cout << "[OptiX] Reusing already-uploaded scene " << scene_id << " (skipping GPU rebuild)\n";
		}

		std::cout << "[TECH] ── Render Technique Summary ──────────────────────────" << std::endl;
		std::cout << "[TECH] Integrator     : GPU SPPM (Stochastic Progressive Photon Mapping, Phase 1)" << std::endl;
		std::cout << "[TECH] Iterations     : " << iterations << "  |  Photons/iteration: " << photons << std::endl;
		std::cout << "[TECH] Materials      : Lambertian, Metal, Dielectric, RoughDielectric (GGX), Conductor (GGX)  |  Lights: area only" << std::endl;
		std::cout << "[TECH] Hash grid      : atomic-head-swap linked list over a fixed device node pool" << std::endl;
		std::cout << "[TECH] Tone mapping   : ACES filmic (Narkowicz) + sRGB OETF  (matches CPU's write_color())" << std::endl;
		std::cout << "[TECH] ─────────────────────────────────────────────────────" << std::endl;

		size_t pixelCount = static_cast<size_t>(image_width) * image_height;
		std::vector<float> framebuffer(pixelCount * 3);

		std::string outStr(output_path);
		size_t sep = outStr.rfind('\\');
		if (sep == std::string::npos) sep = outStr.rfind('/');
		std::string ptxPath = (sep != std::string::npos)
			? outStr.substr(0, sep + 1) + "sppm_programs.ptx"
			: "sppm_programs.ptx";

		std::cout << "[OptiX] Rendering (SPPM)...\n";
		// initialRadius: fixed at Phase 1 (5.0, matching CPU SPPM's own
		// scene-11 default -- see cpu_render_main_sppm's call site) rather
		// than exposed as a CLI parameter yet; --sppm's existing CLI surface
		// (main.cpp/launcher_args.h) has no radius flag for the CPU path
		// either, so GPU staying consistent with that is the faithful port,
		// not a gap introduced here.
		if (!g_renderer->renderSPPM(
				static_cast<unsigned int>(image_width), static_cast<unsigned int>(image_height),
				iterations, photons, static_cast<unsigned int>(max_depth), 5.0f,
				cameraExtra, framebuffer.data(), ptxPath)) {
			std::cerr << "[OptiX] SPPM render failed\n";
			return ERR_GPU_RENDER_FAILED;
		}

		// NaN/Inf firefly guard, applied before either output path (matches
		// the PPM loop below and optix_render_main's own EXR/PPM guards).
		for (size_t i = 0; i < pixelCount * 3; ++i) {
			if (!std::isfinite(framebuffer[i])) framebuffer[i] = 0.0f;
		}

		if (is_exr_output_path(output_path)) {
			std::string exrError;
			if (!write_exr_image(output_path, framebuffer.data(), image_width, image_height, exrError)) {
				std::cerr << "[OptiX] Failed to write EXR '" << output_path << "': " << exrError << "\n";
				return ERR_FILE_WRITE_FAILED;
			}
			std::cout << "[OptiX] SPPM render complete! Output: " << output_path << "\n";
			return 0;
		}

		std::ofstream outFile(output_path, std::ios::binary);
		if (!outFile) {
			std::cerr << "[OptiX] Failed to open output file: " << output_path << "\n";
			return ERR_FILE_WRITE_FAILED;
		}
		outFile << "P3\n" << image_width << " " << image_height << "\n255\n";
		for (size_t i = 0; i < pixelCount; ++i) {
			double r = framebuffer[i * 3 + 0];
			double g = framebuffer[i * 3 + 1];
			double b = framebuffer[i * 3 + 2];
			r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
			g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
			b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
			int ir = static_cast<int>(256.0 * std::fmin(std::fmax(r, 0.0), 0.999));
			int ig = static_cast<int>(256.0 * std::fmin(std::fmax(g, 0.0), 0.999));
			int ib = static_cast<int>(256.0 * std::fmin(std::fmax(b, 0.0), 0.999));
			outFile << ir << " " << ig << " " << ib << "\n";
		}
		outFile.close();

		std::cout << "[OptiX] SPPM render complete! Output: " << output_path << "\n";
		return 0;

	} catch (const std::exception& e) {
		std::cerr << "[OptiX] Exception: " << e.what() << "\n";
		return ERR_GPU_EXCEPTION;
	} catch (...) {
		std::cerr << "[OptiX] Unknown error\n";
		return ERR_GPU_UNKNOWN_ERROR;
	}
}

extern "C" int gpu_scene_light_count(const char* scene_id, int image_width, int image_height) {
	try {
		SceneData scene;
		float camera_params[12];
		GpuCameraParams cameraExtra{};
		if (!build_scene(scene_id, image_width, image_height, scene, camera_params,
						  278.0, 278.0, -800.0, &cameraExtra)) {
			return -1;
		}
		return static_cast<int>(scene.lightIndices.size());
	} catch (...) {
		return -1;
	}
}

extern "C" int optix_validation_issue_count() {
	if (!g_renderer) return 0;
	return static_cast<int>(g_renderer->loggedIssues().size());
}

extern "C" const char* optix_validation_issue(int index) {
	if (!g_renderer) return "";
	const auto& issues = g_renderer->loggedIssues();
	if (index < 0 || static_cast<size_t>(index) >= issues.size()) return "";
	return issues[index].c_str();
}

extern "C" void optix_clear_validation_issues() {
	if (g_renderer) g_renderer->clearLoggedIssues();
}

extern "C" bool optix_validation_enabled() {
	return g_renderer && g_renderer->validationEnabled();
}
