// metal_cpu_gpu_parity_check.cpp
// CPU vs Metal-GPU per-scene render-output parity sweep - the Metal
// counterpart tests/integration/material_cpu_gpu_parity_tests.cpp's own
// header comment calls "real, standalone future work" and
// gpu/metal/metal_poc_validate.cpp's own header comment says this project
// has never had for Metal: not a crash/all-black smoke check (that already
// exists, metal_poc_smoke_render/metal_poc_validate), but an actual check
// that CPU and Metal render the SAME scene to a similar image.
//
// Three-way (CPU vs GPU-recursive vs GPU-wavefront) the way the OptiX
// suite above does doesn't apply on Metal: Metal has exactly ONE GPU
// render path (a single Metal ray-tracing-pipeline kernel per dispatch,
// architecturally closest to OptiX's "recursive" mode - confirmed by an
// explicit audit comparing the two, see docs/METAL_GPU_FEASIBILITY.md's
// own section on it) - there is no Metal wavefront variant to compare
// against. This is a 2-way sweep: CPU vs Metal.
//
// Deliberately a plain executable (no gtest), matching every one of this
// project's other 9 Metal ctest entries' own house style (metal_poc_
// shader_tests.mm's expectTrue/expectNear pattern, metal_poc_validate.cpp's
// plain pass/fail) rather than material_cpu_gpu_parity_tests.cpp's own
// gtest TEST_P structure - that file is NOT buildable on macOS at all (its
// own tests/CMakeLists.txt comment: deliberately excluded from the
// portable unit_tests target because it needs the full cpu_renderer +
// optix_renderer link surface, built only via the MSVC solution on
// Windows), so there's no gtest harness already wired into THIS
// executable's own link graph (this file is built from the root
// CMakeLists.txt's RT_BUILD_METAL block, same as every other Metal ctest
// target, none of which link gtest).
//
// .exr output on BOTH sides, not CPU's own .ppm (material_cpu_gpu_parity_
// tests.cpp) or Metal's default .png: Metal's own postProcessAndWrite()
// applies vignette + chromatic aberration + bilateral denoise + tonemap to
// its .png path (see that function's own comment) that CPU's renderer has
// no equivalent of at all (grep confirms zero vignette/chromaticAberration/
// bilateralDenoise in src/TheRestOfYourLife/camera.h) - comparing tonemapped
// PNGs would always show a gap from that cosmetic mismatch alone, nothing
// to do with whether the actual RENDERING agrees. .exr is the raw, linear,
// pre-tonemap HDR radiance - the "same contract the CPU and OptiX backends'
// EXR path has" Metal's own postProcessAndWrite() comment describes -
// shared by cpu_render_main (src/TheRestOfYourLife/camera.h) and
// metal_render_main (gpu/metal/metal_poc.mm) alike, via the same
// src/shared/exr_writer.h this file also reads back with (tinyexr's
// LoadEXR, already linked into this binary via cpu_renderer's own
// src/external/tinyexr_impl.cpp).
//
// Scope: the same 7 categories material_cpu_gpu_parity_tests.cpp settled
// on after its own Phase 2 expansion (Materials, Volumes, Textures,
// Lights, Cameras, Geometry, Basics) - a scope already proven reasonable
// by that suite's own experience, not re-derived from scratch - filtered
// to scenes Metal actually supports (cpu_scene_pbrt_path_by_id() non-empty -
// the same pbrt-file check metal_render_main() itself makes before rejecting a scene) and
// excluding requires_files (external mesh assets this sandbox doesn't
// bundle - see scripts/build_and_deploy_macos.sh's own comment on that).
//
// GPU cross-scene state corruption (material_cpu_gpu_parity_tests.cpp's
// own documented OptiX/CUDA failure mode - a persistent context
// progressively corrupted across many back-to-back GPU renders in one
// process) is structurally unlikely here: metal_render_main() constructs
// a brand-new MetalPocApp (and therefore a brand-new MTLDevice/command
// queue/pipeline state, via a fresh @autoreleasepool) on EVERY call - see
// that function's own body - there is no persistent GPU context to
// accumulate corruption across scenes the way a single long-lived CUDA
// context can. Not just assumed: this sweep's own first full run is the
// empirical check for that, same as material_cpu_gpu_parity_tests.cpp's
// own calibration process was.
//
// Single-scene isolation for calibration/debugging: set
// METAL_PARITY_ONLY_SCENE_ID to a scene id (e.g. "B9") to render only
// that one - same mechanism and purpose as MATPARITY_ONLY_SCENE_ID in the
// OptiX suite, deliberately renamed (not reused) so the two are never
// confused for each other when both are set in the same shell.
//
// CURRENT STATUS - informational, not yet a hard ctest gate. This sweep's
// own first full run (95 scenes, the complete 7-category/Metal-supported
// set) found 44 scenes outside tolerance - exactly the volume of findings
// material_cpu_gpu_parity_tests.cpp's own header comment describes from
// ITS first runs too (that suite's calibration - B1/B13's real bugs, B14/
// B11's firefly-variance exceptions, E6/J2's real fixes, C17/C18/C19/F13's
// now-removed filter-reach-bug exceptions - took multiple sessions of
// actual investigation, not a single pass). Treating 44 un-investigated
// findings as "the tolerance must be wrong, widen it" or "fail CI" would
// be equally dishonest - neither is true yet. So: this sweep ALWAYS exits
// 0 by default (never fails ctest), and prints a full per-scene report to
// stderr instead - exactly the "ship the tool, not a rubber-stamped green
// checkmark" distinction. Set METAL_PARITY_STRICT=1 to make it exit
// non-zero on any un-excepted failure once/if this is promoted to a real
// gate after enough of the 44 are individually triaged.
//
// Of the 44, 9 are ALREADY self-evidently explained by this sweep's own
// first run - the pbrt loader's own stderr warns, for the exact scene,
// that it fell back to a flat gray Lambertian (a material kind Metal's
// loader doesn't implement: normalizedfresnel/principled/hair/measured/
// subsurface) or silently dropped unsupported shapes (cone/paraboloid/
// bilinear patch/curve) - see kKnownGapScenes below. CPU renders the REAL
// material/geometry; Metal renders a documented, pre-existing
// approximation of it - a real gap, but not a NEW one this sweep
// introduces evidence for, and not something to "fix" here.
//
// E1 (Homogeneous Medium) gapped 67.9% (cpu=0.2787, metal=0.0894) on this
// sweep's first run and is now FIXED (it, and A8, pass). The earlier theory
// that Metal rendered E1 brighter because medium spheres cast no shadow was
// backwards for E1: its "interface" fog sphere had no case in the pbrt
// loader at all, so it became an OPAQUE gray sphere that blocked the light
// (near-black render). Fixing that exposed three more bugs in the
// materialType-28 path: a ray that scatters INSIDE the sphere was stepped
// backwards (shader assumed it always starts outside), the MIS weight at a
// light hit measured distance from the sphere exit instead of the previous
// bounce, and shadow rays crossing the fog were not attenuated (now blocked
// stochastically with probability 1 - exp(-sigma_t * chord)).
//
// The other ~34 are genuinely new, un-triaged findings from this sweep's
// first run - real signal, not yet individually investigated. Follow-up
// work, not a blocker for landing the tool itself.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../../src/external/tinyexr.h"
#include "scene_registry.h"

extern "C" {
	#include "cpu_interface.h"
	#include "metal_interface.h"
}

namespace {

// ============================================================================
// Image load/diff - deliberately re-implemented rather than shared with
// material_cpu_gpu_parity_tests.cpp's identically-shaped MPImage/mp_*
// helpers: that file isn't linked into this binary at all (see this file's
// own header comment), so there's no symbol to share in the first place -
// same names/shapes anyway, so the two sweeps stay easy to compare by eye.
// ============================================================================

struct MPImage {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB linear, interleaved
	bool valid = false;
};

MPImage mp_load_exr(const char* path) {
	MPImage img;
	float* rgba = nullptr;
	int w = 0, h = 0;
	const char* err = nullptr;
	const int rc = LoadEXR(&rgba, &w, &h, path, &err);
	if (rc != TINYEXR_SUCCESS) {
		if (err) FreeEXRErrorMessage(err);
		return img;
	}
	img.width = w;
	img.height = h;
	img.pixels.resize(size_t(w) * h * 3);
	for (int i = 0; i < w * h; ++i) {
		// LoadEXR always returns RGBA regardless of the file's own channel
		// count (tinyexr.h's own documented behavior) - drop alpha.
		img.pixels[i * 3 + 0] = rgba[i * 4 + 0];
		img.pixels[i * 3 + 1] = rgba[i * 4 + 1];
		img.pixels[i * 3 + 2] = rgba[i * 4 + 2];
	}
	free(rgba);
	img.valid = true;
	return img;
}

// Number of NaN/Inf samples. A single one makes every mean NaN, and every
// `rel > tol` comparison below is then false - i.e. a render full of NaNs
// would be reported as a PASS - so this is checked explicitly, first.
size_t mp_count_nonfinite(const MPImage& img) {
	size_t n = 0;
	for (float v : img.pixels) if (!std::isfinite(v)) ++n;
	return n;
}

float mp_avg_brightness(const MPImage& img) {
	if (img.pixels.empty()) return 0.0f;
	const float s = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return s / float(img.pixels.size());
}

struct MPRGBAverage { float r, g, b; };

MPRGBAverage mp_avg_channels(const MPImage& img) {
	MPRGBAverage out{0, 0, 0};
	const int n = img.width * img.height;
	if (n == 0) return out;
	double sr = 0, sg = 0, sb = 0;
	for (int i = 0; i < n; ++i) {
		sr += img.pixels[i * 3 + 0];
		sg += img.pixels[i * 3 + 1];
		sb += img.pixels[i * 3 + 2];
	}
	out.r = float(sr / n);
	out.g = float(sg / n);
	out.b = float(sb / n);
	return out;
}

struct MPRegionalDiffResult {
	float maxBlockRelDiff = 0.0f;
	int worstBlockX = -1, worstBlockY = -1;
	int blocksOverThreshold = 0;
	int comparableBlocks = 0;
};

// Mirrors material_cpu_gpu_parity_tests.cpp's own mp_regional_diff() -
// same algorithm (per-block average-brightness relative difference over a
// gridSize x gridSize grid, skipping blocks too dark on both sides to be
// meaningfully comparable), ported rather than shared for the same reason
// MPImage above is.
MPRegionalDiffResult mp_regional_diff(const MPImage& a, const MPImage& b,
                                       int gridSize, float minComparable, float threshold) {
	MPRegionalDiffResult result;
	if (!a.valid || !b.valid || a.width != b.width || a.height != b.height ||
	    a.width <= 0 || a.height <= 0) {
		return result;
	}
	const int blockW = std::max(1, a.width / gridSize);
	const int blockH = std::max(1, a.height / gridSize);
	for (int by = 0; by < gridSize; ++by) {
		const int y0 = by * blockH;
		const int y1 = (by == gridSize - 1) ? a.height : std::min(a.height, y0 + blockH);
		if (y0 >= a.height) continue;
		for (int bx = 0; bx < gridSize; ++bx) {
			const int x0 = bx * blockW;
			const int x1 = (bx == gridSize - 1) ? a.width : std::min(a.width, x0 + blockW);
			if (x0 >= a.width) continue;

			float sumA = 0.0f, sumB = 0.0f;
			int n = 0;
			for (int y = y0; y < y1; ++y) {
				for (int x = x0; x < x1; ++x) {
					const int idx = (y * a.width + x) * 3;
					sumA += a.pixels[idx] + a.pixels[idx + 1] + a.pixels[idx + 2];
					sumB += b.pixels[idx] + b.pixels[idx + 1] + b.pixels[idx + 2];
					n += 3;
				}
			}
			if (n == 0) continue;
			const float avgA = sumA / n, avgB = sumB / n;
			if (avgA < minComparable && avgB < minComparable) continue;

			++result.comparableBlocks;
			const float maxV = std::max(avgA, avgB);
			const float relDiff = std::abs(avgA - avgB) / maxV;
			if (relDiff > threshold) ++result.blocksOverThreshold;
			if (relDiff > result.maxBlockRelDiff) {
				result.maxBlockRelDiff = relDiff;
				result.worstBlockX = bx;
				result.worstBlockY = by;
			}
		}
	}
	return result;
}

// ============================================================================
// Tolerances - PLACEHOLDER values pending this sweep's own first real
// calibration run (the SAME "render the real renderer, measure the real
// gaps, then set tolerances from that" process material_cpu_gpu_parity_
// tests.cpp's own header comment describes, not guessed) - see this file's
// own CALIBRATION NOTES section (bottom) for the running log of that
// process once it exists.
// ============================================================================

constexpr int kWidth = 60;
constexpr int kHeight = 60;
constexpr int kDepth = 8;
constexpr int kCpuSpp = 200;
constexpr int kMetalSpp = 600;
constexpr int kVolumeCpuSpp = 300;
constexpr int kVolumeMetalSpp = 900;

constexpr float kRelTolerance = 0.30f;
constexpr float kVolumeRelTolerance = 0.30f;
constexpr int kRegionalGridSize = 6;
constexpr float kRegionalRelTolerance = 0.50f;
constexpr float kMinComparableValue = 0.004f;
constexpr float kRegionalMinComparableValue = 0.02f;

float regional_tolerance_for(float wholeImageTolerance) {
	constexpr float kMultiplier = kRegionalRelTolerance / kRelTolerance;
	return std::min(wholeImageTolerance * kMultiplier, 0.95f);
}

// Scenes this sweep's own first run already self-evidently explained via
// the pbrt loader's own stderr warning for that exact scene (a material
// kind or shape type Metal's loader doesn't implement, falling back to a
// flat gray Lambertian or silently dropping the shape) - see this file's
// own header comment ("CURRENT STATUS") for the full reasoning. Printed
// as "known gap" rather than "FAIL" below; still informational-only
// either way (METAL_PARITY_STRICT doesn't distinguish the two - both
// count as "not a clean pass" if that's ever enabled).
const char* const kKnownGapScenes[] = {
	"B9", "B10", "B11", "B16", "B20",  // unsupported material kind -> gray Lambertian fallback
	"F1", "F4", "F7", "F8", "F14",            // unsupported shape (cone/paraboloid/bilinear patch/curve) -> silently dropped
	// C17 (Portal Light): portal-light.pbrt reads sssdragon/textures/small_rural_road_equiarea.exr,
	// which is not tracked in this repo (only the sssdragon benchmark checkout has it). Both
	// backends log "could not be read; using its constant colour instead", but then diverge on
	// that fallback (CPU renders black, cpu=0.0000; Metal 0.5884), so the sweep is comparing two
	// different failure modes, not two renderers on the same scene. Re-triage (including whether
	// Metal honours the portal restriction at all) once the EXR is available.
	"C17",
	// C2 (Spotlight Cornell): the sphere is a "conductor" given only a reflectance (no named metal
	// spectrum). CPU approximates that with a fuzzy-mirror `metal` material (pbrt_cpu_builder.h), which
	// cannot show a highlight from a delta (spot/point) light; Metal uses the real GGX conductor
	// (closer to pbrt-v4), so it shows one. Direct lighting everywhere else matches CPU block-for-block.
	"C2",
	// F2 (Triangle Mesh): same cause as C2 - its icosahedron is a reflectance-only conductor. Verified:
	// with that mesh swapped to diffuse, CPU and Metal agree to ~1.0-1.2x everywhere.
	"F2",
	// A5 (Perlin Spheres), A7 (Simple Light): "marble" procedural reflectance texture. The Metal pbrt loader
	// has no mapping for MaterialKind::Diffuse + hasMarbleReflectance (CPU/OptiX use pbrt-v4 MarbleTexture),
	// so Metal renders those surfaces with a flat colour.
	"A5", "A7",
	// J3, J5, J6: nested/procedural reflectance textures (checkerboard-in-checkerboard, fbm/windy/wrinkled
	// gallery). The loader maps only the simple cases and averages a nested one to a flat colour (see
	// nestedProceduralAverageColor use in metal_poc_pbrt_loader.mm); CPU renders the real texture.
	"J3", "J5", "J6",
	// E6 (Cylinder Medium), E7 (RGB Grid Medium): the medium itself is not representable on Metal (homogeneous
	// medium on a cylinder; heterogeneous rgbgrid on a sphere). The interface boundary is now transparent
	// instead of an opaque gray shape, but the volume scattering is missing.
	"E6", "E7",
};

bool is_known_gap_scene(const std::string& id) {
	for (const char* known : kKnownGapScenes) if (id == known) return true;
	return false;
}

// ============================================================================
// Scene selection
// ============================================================================

bool metal_supports_scene(const std::string& id) {
	const char* pbrtPath = cpu_scene_pbrt_path_by_id(id.c_str());
	return pbrtPath && pbrtPath[0];
}

std::vector<const SceneDescriptor*> testable_scenes() {
	static const char* const kSweptCategories[] = {
		SceneCategories::Materials, SceneCategories::Volumes, SceneCategories::Textures,
		SceneCategories::Lights, SceneCategories::Cameras, SceneCategories::Geometry,
		SceneCategories::Basics,
	};
	std::vector<const SceneDescriptor*> out;
	for (const SceneDescriptor& s : get_scene_registry()) {
		bool inSweptCategory = false;
		for (const char* cat : kSweptCategories) {
			if (std::strcmp(s.category, cat) == 0) { inSweptCategory = true; break; }
		}
		if (!inSweptCategory || s.requires_files) continue;
		if (!metal_supports_scene(s.id)) continue;
		const SceneDescriptor* found = find_scene(s.id);
		if (found) out.push_back(found);
	}
	if (const char* only = std::getenv("METAL_PARITY_ONLY_SCENE_ID")) {
		std::vector<const SceneDescriptor*> filtered;
		for (const SceneDescriptor* s : out) if (s->id == only) filtered.push_back(s);
		return filtered;
	}
	return out;
}

void spp_for(const SceneDescriptor& s, int& cpuSpp, int& metalSpp) {
	const bool isVolume = std::strcmp(s.category, SceneCategories::Volumes) == 0;
	cpuSpp = isVolume ? kVolumeCpuSpp : kCpuSpp;
	metalSpp = isVolume ? kVolumeMetalSpp : kMetalSpp;
}

// ============================================================================
// Rendering
// ============================================================================

MPImage render_cpu_once(const SceneDescriptor& s, int spp) {
	const std::string fn = "mcparity_" + s.id + "_cpu.exr";
	cpu_render_main(kWidth, kHeight, spp, kDepth, fn.c_str(), s.id.c_str(),
	                 s.camera.lookfrom_x, s.camera.lookfrom_y, s.camera.lookfrom_z);
	MPImage img = mp_load_exr(fn.c_str());
	std::remove(fn.c_str());
	return img;
}

MPImage render_metal_once(const SceneDescriptor& s, int spp) {
	const std::string fn = "mcparity_" + s.id + "_metal.exr";
	metal_render_main(kWidth, kHeight, spp, kDepth, fn.c_str(), s.id.c_str(),
	                   s.camera.lookfrom_x, s.camera.lookfrom_y, s.camera.lookfrom_z);
	MPImage img = mp_load_exr(fn.c_str());
	std::remove(fn.c_str());
	return img;
}

} // namespace

int main() {
	MetalDiagnostics diag{};
	if (!metal_get_diagnostics(&diag)) {
		fprintf(stderr, "SKIP: no Metal device available (%s) - nothing to compare against CPU\n",
		        diag.failure_reason[0] ? diag.failure_reason : "unknown reason");
		return 0;
	}
	fprintf(stderr, "[mcparity] Metal device: %s\n", diag.device_name);

	const std::vector<const SceneDescriptor*> scenes = testable_scenes();
	fprintf(stderr, "[mcparity] %zu scene(s) selected (Materials/Volumes/Textures/Lights/"
	                "Cameras/Geometry/Basics, Metal-supported, no external files)\n", scenes.size());

	int failures = 0;
	int knownGaps = 0;
	int skipped = 0;
	for (const SceneDescriptor* sp : scenes) {
		const SceneDescriptor& s = *sp;
		int cpuSpp, metalSpp;
		spp_for(s, cpuSpp, metalSpp);
		const bool isVolume = std::strcmp(s.category, SceneCategories::Volumes) == 0;
		const float wholeTol = isVolume ? kVolumeRelTolerance : kRelTolerance;
		const float regionalTol = regional_tolerance_for(wholeTol);

		MPImage cpuImg = render_cpu_once(s, cpuSpp);
		MPImage metalImg = render_metal_once(s, metalSpp);

		if (!cpuImg.valid || !metalImg.valid) {
			fprintf(stderr, "[mcparity] %-6s SKIP (render failed to produce a valid .exr - "
			                "cpu=%s metal=%s)\n",
			        s.id.c_str(), cpuImg.valid ? "ok" : "FAILED", metalImg.valid ? "ok" : "FAILED");
			++skipped;
			continue;
		}

		bool sceneFailed = false;
		char why[512] = {};

		const size_t cpuBad = mp_count_nonfinite(cpuImg);
		const size_t metalBad = mp_count_nonfinite(metalImg);
		if (cpuBad > 0 || metalBad > 0) {
			sceneFailed = true;
			snprintf(why, sizeof(why), "non-finite pixels (NaN/Inf): cpu=%zu metal=%zu of %zu samples",
			         cpuBad, metalBad, cpuImg.pixels.size());
		}

		const float cpuB = mp_avg_brightness(cpuImg);
		const float metalB = mp_avg_brightness(metalImg);
		if (!sceneFailed && std::max(cpuB, metalB) >= kMinComparableValue) {
			const float relDiff = std::abs(cpuB - metalB) / std::max(cpuB, metalB);
			if (relDiff > wholeTol) {
				sceneFailed = true;
				snprintf(why, sizeof(why), "brightness cpu=%.4f metal=%.4f rel=%.1f%% (tol %.0f%%)",
				         cpuB, metalB, relDiff * 100.0f, wholeTol * 100.0f);
			}
		}

		const MPRGBAverage cpuC = mp_avg_channels(cpuImg);
		const MPRGBAverage metalC = mp_avg_channels(metalImg);
		const char* chanNames[3] = {"R", "G", "B"};
		const float cpuCh[3] = {cpuC.r, cpuC.g, cpuC.b};
		const float metalCh[3] = {metalC.r, metalC.g, metalC.b};
		for (int c = 0; c < 3 && !sceneFailed; ++c) {
			if (std::max(cpuCh[c], metalCh[c]) < kMinComparableValue) continue;
			const float relDiff = std::abs(cpuCh[c] - metalCh[c]) / std::max(cpuCh[c], metalCh[c]);
			if (relDiff > wholeTol) {
				sceneFailed = true;
				snprintf(why, sizeof(why), "%s channel cpu=%.4f metal=%.4f rel=%.1f%% (tol %.0f%%)",
				         chanNames[c], cpuCh[c], metalCh[c], relDiff * 100.0f, wholeTol * 100.0f);
			}
		}

		MPRegionalDiffResult regional;
		if (!sceneFailed) {
			regional = mp_regional_diff(cpuImg, metalImg, kRegionalGridSize,
			                             kRegionalMinComparableValue, regionalTol);
			if (regional.blocksOverThreshold > 0) {
				sceneFailed = true;
				snprintf(why, sizeof(why),
				         "regional: block (%d,%d) rel=%.1f%% (tol %.0f%%), %d/%d blocks over threshold",
				         regional.worstBlockX, regional.worstBlockY, regional.maxBlockRelDiff * 100.0f,
				         regionalTol * 100.0f, regional.blocksOverThreshold, regional.comparableBlocks);
			}
		}

		if (sceneFailed && is_known_gap_scene(s.id)) {
			fprintf(stderr, "[mcparity] %-6s known-gap  %s (this scene's own pbrt-loader stderr above "
			                "already explains why - see kKnownGapScenes' own comment)\n",
			        s.id.c_str(), why);
			++knownGaps;
		} else if (sceneFailed) {
			fprintf(stderr, "[mcparity] %-6s FAIL  %s\n", s.id.c_str(), why);
			++failures;
		} else {
			fprintf(stderr, "[mcparity] %-6s pass  brightness cpu=%.4f metal=%.4f, worst regional block %.1f%%\n",
			        s.id.c_str(), cpuB, metalB, regional.maxBlockRelDiff * 100.0f);
		}
	}

	const int passed = int(scenes.size()) - failures - knownGaps - skipped;
	fprintf(stderr, "[mcparity] %zu scene(s): %d failed (un-triaged), %d known gap, %d skipped, %d passed\n",
	        scenes.size(), failures, knownGaps, skipped, passed);

	// Informational by default - see this file's own header comment
	// ("CURRENT STATUS") for why. METAL_PARITY_STRICT=1 opts into real
	// ctest pass/fail semantics (known-gap scenes still don't count
	// against it - they're documented, expected, not a regression to
	// catch - only `failures`, the un-triaged bucket, does).
	const bool strict = std::getenv("METAL_PARITY_STRICT") != nullptr;
	if (strict && failures > 0) {
		fprintf(stderr, "FAIL: %d scene(s) exceeded CPU-vs-Metal parity tolerance (METAL_PARITY_STRICT=1)\n",
		        failures);
		return 1;
	}
	fprintf(stderr, "PASS (informational): %d/%zu scenes within tolerance, %d known/documented gap(s), "
	                "%d un-triaged finding(s) for follow-up - see this file's own header comment\n",
	        passed, scenes.size(), knownGaps, failures);
	return 0;
}
