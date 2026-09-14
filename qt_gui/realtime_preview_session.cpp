#include "realtime_preview_session.h"
#include "camera_math.h"
#include "../src/shared/tone_map.h"
#include "cross_abi_library.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace {

// bool(const char* scene_id, int w, int h, int spp, int max_depth,
//      double camX, double camY, double camZ,
//      bool has_custom_lookat, double lookX, double lookY, double lookZ,
//      bool denoise, double denoiseBlend,
//      float* out_world_pos, float* out_camera_basis,
//      float* out_rgb, bool enable_svgf, bool enable_restir_gi,
//      float max_component_value, const void* svgf_tuning,
//      bool enable_restir_di, bool enable_probe_cache, bool enable_path_guiding,
//      bool enable_temporal_upscale, int temporal_upscale_factor,
//      unsigned int temporal_jitter_base_index)
// Must stay byte-for-byte in sync with gpu/optix/optix_interface.h's
// rt_realtime_render_frame() declaration and realtime_renderer_dll.cpp's own
// export signature - see this file's own header comment on why there's no
// shared header/versioning across this boundary. New parameters are always
// appended at the end, never inserted in the middle - enable_restir_di,
// enable_probe_cache, enable_path_guiding, and now the 3 temporal-upscale
// params are all appended last for exactly this reason, even though
// enable_path_guiding logically pairs with enable_probe_cache (it
// hard-depends on it) rather than sitting at the end.
typedef bool (*RenderFrameFn)(const char*, int, int, int, int, double, double, double,
							   bool, double, double, double, bool, double, float*, float*, float*, bool,
							   bool, float, const void*, bool, bool, bool,
							   bool, int, unsigned int, bool,
							   bool, float*,
							   double, double);

// const char*(void) - see gpu/optix/optix_interface.h's rt_realtime_get_last_error()
// own comment. Same hand-duplication convention as RenderFrameFn above.
typedef const char* (*GetLastErrorFn)();

// Mirrors gpu/optix/svgf_tuning_params.h's SvgfTuningParams field-for-field -
// see that header's own comment on why this boundary hand-duplicates types
// rather than sharing a header. `const void*` in the typedef above (rather
// than `const SvgfTuningParams*`) avoids exposing this qt_gui-local type name
// through the function-pointer type itself; reinterpret_cast<const void*>(&x)
// at the one real call site (renderLoop(), below) is enough - the ACTUAL
// receiving side (optix_interface.cpp) casts it back via the real,
// canonical-layout struct from gpu/optix/svgf_tuning_params.h.
struct SvgfTuningParams {
	float temporalAlpha = 0.2f;
	float maxHistoryLength = 32.0f;
	float varianceBootstrapFrames = 4.0f;
	int   varianceBootstrapRadius = 3;
	float sigmaNormal = 128.0f;
	float sigmaDepth = 1.0f;
	float sigmaLuminance = 4.0f;
	int   atrousRadius = 2;
	float minAlbedo = 0.02f;
	int   atrousPasses = 4;
};

struct DllHandle {
	void* module = nullptr;
	RenderFrameFn renderFrameFn = nullptr;
	GetLastErrorFn getLastErrorFn = nullptr;
};

#ifdef Q_OS_WIN
constexpr const char* kLibraryFileName = "realtime_renderer.dll";
#elif defined(Q_OS_MAC)
constexpr const char* kLibraryFileName = "realtime_renderer.dylib";
#else
constexpr const char* kLibraryFileName = "realtime_renderer.so";
#endif

// Same LoadLibrary/GetProcAddress-across-the-MinGW/MSVC-ABI-boundary
// approach as scene_metadata_client.cpp - see that file's own header
// comment for why a DLL boundary is needed at all here. std::call_once
// since RealtimePreviewWorker::start()/renderLoop() all run on the worker
// thread, not the GUI thread that (re)creates RealtimePreviewSession.
DllHandle& handle() {
	static DllHandle h;
	static std::once_flag loadOnce;
	// [] not [&h]: h has static storage duration, so the lambda can already
	// reference it directly without capturing it - see
	// scene_metadata_client.cpp's identical fix for the full explanation
	// (MSVC's C3495 vs. MinGW silently accepting the same code).
	std::call_once(loadOnce, []() {
		h.module = cross_abi_library::loadLibrary(QCoreApplication::applicationDirPath(), kLibraryFileName);
		if (!h.module) return;
		h.renderFrameFn = reinterpret_cast<RenderFrameFn>(
			cross_abi_library::lookupSymbol(h.module, "realtime_render_frame"));
		h.getLastErrorFn = reinterpret_cast<GetLastErrorFn>(
			cross_abi_library::lookupSymbol(h.module, "realtime_get_last_error"));
	});
	return h;
}

} // namespace

bool RealtimePreviewSession::isAvailable() {
	return handle().renderFrameFn != nullptr;
}

// ============================================================================
// RealtimePreviewWorker
// ============================================================================

void RealtimePreviewWorker::resetAccumulation() {
	m_accum.assign(static_cast<size_t>(m_width) * m_height * 3, 0.0f);
	m_tmp.assign(static_cast<size_t>(m_width) * m_height * 3, 0.0f);
	// Zeroed (not just resized) for the same reason m_accum/m_tmp are: the
	// w=0 validity flag on every pixel means reprojectAccumulation() (if it
	// somehow ran before a single real frame had populated these - it
	// shouldn't, since start() clears m_cameraDirty right below, but this
	// costs nothing to make true anyway) would correctly treat every pixel
	// as "no old data", not read stale/garbage floats as a real position.
	m_worldPos.assign(static_cast<size_t>(m_width) * m_height * 4, 0.0f);
	m_worldPosPrev.assign(static_cast<size_t>(m_width) * m_height * 4, 0.0f);
	m_cameraBasis.assign(12, 0.0f);
	m_prevCameraBasis.assign(12, 0.0f);
	m_sampleCounts.assign(static_cast<size_t>(m_width) * m_height, 0);
	m_accumScratch.assign(static_cast<size_t>(m_width) * m_height * 3, 0.0f);
	m_sampleCountsScratch.assign(static_cast<size_t>(m_width) * m_height, 0);

	// Temporal upscale's own higher-resolution reconstruction buffers - only
	// allocated (Wh*Hh-sized) while the feature is ACTUALLY going to be used
	// this session, cleared to empty otherwise so a disabled/inactive session
	// carries no extra host memory cost (see this project's own plan's Risks
	// section on the ~40-200MB cost at 2x/4x). Gated on useTemporalUpscale()
	// (m_temporalUpscaleEnabled && !effectiveShowLatest()), NOT the raw
	// m_temporalUpscaleEnabled flag alone - using the raw flag here used to
	// let this size m_displayImage at Wh x Hh even when SVGF/show-latest was
	// also on (making useTemporalUpscale() false), while renderLoop()'s own
	// tonemap step only ever wrote the smaller [0,m_width)x[0,m_height)
	// region in that case - leaving the rest of the QImage as uninitialized
	// garbage (a real, confirmed bug this project's own code review caught).
	// Every caller that can change either half of useTemporalUpscale()'s own
	// condition (setSvgf(), setDenoise(), setTemporalUpscale()) already calls
	// resetAccumulation() whenever the EFFECTIVE state changes, so re-
	// evaluating useTemporalUpscale() here always reflects the same "which
	// mode is really active" decision renderLoop() will make next.
	// m_temporalJitterCounter/m_temporalUpscaleWriteCounter restart at 0 too -
	// a fresh reconstruction buffer has no history to resume a jitter phase
	// against.
	m_temporalJitterCounter = 0;
	m_temporalUpscaleWriteCounter = 0;
	if (useTemporalUpscale()) {
		const size_t wh = static_cast<size_t>(m_width) * m_upscaleFactor;
		const size_t hh = static_cast<size_t>(m_height) * m_upscaleFactor;
		m_accumHi.assign(wh * hh * 3, 0.0f);
		m_worldPosHi.assign(wh * hh * 4, 0.0f);
		m_worldPosHiPrev.assign(wh * hh * 4, 0.0f);
		m_accumHiScratch.assign(wh * hh * 3, 0.0f);
		m_worldPosHiScratch.assign(wh * hh * 4, 0.0f);
		// Neural temporal upscale's own GPU-filled result buffer (Live
		// Preview only) - see this project's own plan. Sized the same
		// wh*hh*3 as m_accumHi, but only when the feature is ACTUALLY going
		// to be used this session (m_neuralUpscale, gated the same way
		// m_accumHi itself already is on useTemporalUpscale()) - a plain
		// std::vector, not a GPU buffer; the GPU-side result crosses the DLL
		// boundary and lands here via renderFrame()'s own out_neural_upscale_buffer
		// parameter (renderLoop(), below).
		if (m_neuralUpscale) {
			m_neuralUpscaleOut.assign(wh * hh * 3, 0.0f);
		} else {
			m_neuralUpscaleOut.clear();
		}
		m_displayImage = QImage(static_cast<int>(wh), static_cast<int>(hh), QImage::Format_RGB888);
	} else {
		m_accumHi.clear();
		m_worldPosHi.clear();
		m_worldPosHiPrev.clear();
		m_accumHiScratch.clear();
		m_worldPosHiScratch.clear();
		m_neuralUpscaleOut.clear();
		m_displayImage = QImage(m_width, m_height, QImage::Format_RGB888);
	}
	m_sampleCount = 0;
}

// Reuses the OLD accumulation across a camera move instead of discarding it:
// for each of THIS frame's pixels (already rendered with the NEW camera,
// giving m_worldPos its real world-space primary-hit points), projects that
// world point into m_prevCameraBasis - the basis that rendered whatever is
// CURRENTLY sitting in m_accum (see the member declarations' own comment) -
// via camera_math.h's projectToScreen(), to find which OLD pixel showed the
// same location on screen before the camera moved. Accepts that old pixel's
// accumulated color and (capped) sample count only if its OWN remembered
// world position (m_worldPosPrev) is close enough to this frame's new one -
// otherwise the old pixel showed a DIFFERENT surface (something the camera
// move just occluded or disoccluded), and reusing its color would paste the
// wrong object's history onto this one. Builds entirely new buffers rather
// than overwriting m_accum/m_sampleCounts in place, since pixel P's new
// value is read from a DIFFERENT pixel Q's old one - an in-place write could
// clobber data a later pixel in the same pass still needs to read.
//
// Everything that doesn't pass (off-screen in the old view, the old pixel
// itself had no data, or disocclusion) is simply left at zero/count-0 -
// renderLoop()'s own running-mean update right after this returns then
// folds in this frame's own new sample as sample #1 for those pixels, same
// as it always has for a pixel with no prior history.
void RealtimePreviewWorker::reprojectAccumulation() {
	const camera_math::CameraBasis oldBasis{
		camera_math::Vec3{m_prevCameraBasis[0], m_prevCameraBasis[1], m_prevCameraBasis[2]},
		camera_math::Vec3{m_prevCameraBasis[3], m_prevCameraBasis[4], m_prevCameraBasis[5]},
		camera_math::Vec3{m_prevCameraBasis[6], m_prevCameraBasis[7], m_prevCameraBasis[8]},
		camera_math::Vec3{m_prevCameraBasis[9], m_prevCameraBasis[10], m_prevCameraBasis[11]}};

	// A long-static view could otherwise accumulate an effective history so
	// large that a REAL subsequent change (the camera moves back to reveal
	// something new right at this pixel) would take just as long to react
	// to via the running mean's own 1/(n+1) weighting - capped so reused
	// history never outweighs new evidence by more than this.
	constexpr uint16_t kMaxHistorySamples = 64;
	// World-space units, not screen pixels - scaled by the camera's CURRENT
	// distance from its pivot rather than a fixed constant, since this
	// codebase's own scene library spans wildly different scales (scene 1's
	// spheres sit within roughly +-15 units of the origin vs. Cornell Box's
	// ~555 - see this file's own kUnitsPerStep-style constants elsewhere for
	// the same scale-mismatch problem). A fixed absolute threshold would be
	// far too loose on a small scene (falsely accepting a disoccluded but
	// nearby-in-world-space DIFFERENT surface as a match - visible ghosting)
	// or far too tight on a large one (rejecting real, valid reprojections
	// as "disoccluded" - defeating the whole feature). distanceFromTarget()
	// is already available here with no new plumbing, and scales naturally
	// with whatever the camera is actually looking at right now. The
	// fraction below is tuned so a Cornell-Box-scale session (radius
	// around 800, e.g. resolve_fixed_lookfrom()'s own (278,278,-800)
	// default) lands close to this constant's ORIGINAL fixed value (1.0).
	constexpr double kDisocclusionEpsilonFraction = 0.00125;
	const double cameraRadius = camera_math::distanceFromTarget(
		camera_math::Vec3{m_camX, m_camY, m_camZ}, camera_math::Vec3{m_lookX, m_lookY, m_lookZ});
	const float kDisocclusionEpsilon = static_cast<float>(cameraRadius * kDisocclusionEpsilonFraction);
	const float kDisocclusionEpsilonSq = kDisocclusionEpsilon * kDisocclusionEpsilon;

	// Reused, persistently-sized scratch buffers (resetAccumulation()) rather
	// than freshly allocated here every call - reprojectAccumulation() runs
	// on every rendered frame for the duration of a drag/held key, not just
	// once per discrete move, so a fresh heap allocation here would be
	// avoidable churn on that interactive hot path.
	std::fill(m_accumScratch.begin(), m_accumScratch.end(), 0.0f);
	std::fill(m_sampleCountsScratch.begin(), m_sampleCountsScratch.end(), uint16_t{0});
	std::vector<float> &newAccum = m_accumScratch;
	std::vector<uint16_t> &newSampleCounts = m_sampleCountsScratch;

	for (int y = 0; y < m_height; ++y) {
		for (int x = 0; x < m_width; ++x) {
			const int pixel = y * m_width + x;
			const size_t wpIdx = static_cast<size_t>(pixel) * 4;
			if (m_worldPos[wpIdx + 3] == 0.0f) continue;  // this frame's own pixel missed - nothing to reproject

			const camera_math::Vec3 worldPoint{m_worldPos[wpIdx], m_worldPos[wpIdx + 1], m_worldPos[wpIdx + 2]};
			const camera_math::ScreenProjection proj = camera_math::projectToScreen(worldPoint, oldBasis);
			if (!proj.inFront || proj.s < 0.0 || proj.s >= 1.0 || proj.t < 0.0 || proj.t >= 1.0) continue;

			const int oldCol = static_cast<int>(proj.s * m_width);
			const int oldRow = static_cast<int>((1.0 - proj.t) * m_height);
			if (oldCol < 0 || oldCol >= m_width || oldRow < 0 || oldRow >= m_height) continue;
			const int oldPixel = oldRow * m_width + oldCol;
			const size_t oldWpIdx = static_cast<size_t>(oldPixel) * 4;
			if (m_worldPosPrev[oldWpIdx + 3] == 0.0f) continue;  // old pixel had no real sample either

			const float dx = m_worldPosPrev[oldWpIdx + 0] - worldPoint.x;
			const float dy = m_worldPosPrev[oldWpIdx + 1] - worldPoint.y;
			const float dz = m_worldPosPrev[oldWpIdx + 2] - worldPoint.z;
			if (dx * dx + dy * dy + dz * dz > kDisocclusionEpsilonSq) continue;  // different surface

			const size_t oldColorIdx = static_cast<size_t>(oldPixel) * 3;
			const size_t newColorIdx = static_cast<size_t>(pixel) * 3;
			newAccum[newColorIdx + 0] = m_accum[oldColorIdx + 0];
			newAccum[newColorIdx + 1] = m_accum[oldColorIdx + 1];
			newAccum[newColorIdx + 2] = m_accum[oldColorIdx + 2];
			newSampleCounts[pixel] = std::min(m_sampleCounts[oldPixel], kMaxHistorySamples);
		}
	}

	// Swap rather than assign: newAccum/newSampleCounts are references to
	// the persistent scratch members, so this exchanges their storage with
	// m_accum/m_sampleCounts's own (no allocation) - next call's std::fill
	// above then zeroes what is now the scratch buffer (the OLD m_accum
	// contents), ready to be built into again.
	std::swap(m_accum, newAccum);
	std::swap(m_sampleCounts, newSampleCounts);
}

// Temporal upscale's own reprojection (see this project's own plan) - same
// world-position + disocclusion-test technique as reprojectAccumulation()
// above, generalized two ways: (1) projects directly into the higher-
// resolution Hi grid's own coordinates (continuous screen-space (s,t) makes
// this free - no separate low-res-then-high-res mapping step needed), and
// (2) since the GPU still only ever produces ONE world position per low-res
// pixel, a whole upscaleFactor x upscaleFactor cell block is reprojected as
// a single rigid translation rather than each sub-cell independently - see
// this project's own plan's Risks section for the accepted silhouette-block
// approximation this implies.
void RealtimePreviewWorker::reprojectAccumulationHi() {
	const camera_math::CameraBasis oldBasis{
		camera_math::Vec3{m_prevCameraBasis[0], m_prevCameraBasis[1], m_prevCameraBasis[2]},
		camera_math::Vec3{m_prevCameraBasis[3], m_prevCameraBasis[4], m_prevCameraBasis[5]},
		camera_math::Vec3{m_prevCameraBasis[6], m_prevCameraBasis[7], m_prevCameraBasis[8]},
		camera_math::Vec3{m_prevCameraBasis[9], m_prevCameraBasis[10], m_prevCameraBasis[11]}};

	// Same disocclusion-epsilon formula as reprojectAccumulation() - see that
	// function's own comment for the full rationale (scaled by the camera's
	// current distance from its pivot, since this codebase's scene library
	// spans wildly different world-space scales).
	constexpr double kDisocclusionEpsilonFraction = 0.00125;
	const double cameraRadius = camera_math::distanceFromTarget(
		camera_math::Vec3{m_camX, m_camY, m_camZ}, camera_math::Vec3{m_lookX, m_lookY, m_lookZ});
	const float kDisocclusionEpsilon = static_cast<float>(cameraRadius * kDisocclusionEpsilonFraction);
	const float kDisocclusionEpsilonSq = kDisocclusionEpsilon * kDisocclusionEpsilon;

	const int Wh = m_width * m_upscaleFactor;
	const int Hh = m_height * m_upscaleFactor;

	std::fill(m_accumHiScratch.begin(), m_accumHiScratch.end(), 0.0f);
	std::fill(m_worldPosHiScratch.begin(), m_worldPosHiScratch.end(), 0.0f);
	std::vector<float> &newAccumHi = m_accumHiScratch;
	std::vector<float> &newWorldPosHi = m_worldPosHiScratch;

	for (int y = 0; y < m_height; ++y) {
		for (int x = 0; x < m_width; ++x) {
			const int pixel = y * m_width + x;
			const size_t wpIdx = static_cast<size_t>(pixel) * 4;
			if (m_worldPos[wpIdx + 3] == 0.0f) continue;  // this frame's own pixel missed

			const camera_math::Vec3 worldPoint{m_worldPos[wpIdx], m_worldPos[wpIdx + 1], m_worldPos[wpIdx + 2]};
			const camera_math::ScreenProjection proj = camera_math::projectToScreen(worldPoint, oldBasis);
			if (!proj.inFront || proj.s < 0.0 || proj.s >= 1.0 || proj.t < 0.0 || proj.t >= 1.0) continue;

			// Straight into HIGH-RES grid coordinates - (s,t) is continuous,
			// so this recovers sub-low-res-pixel precision for the ANCHOR
			// (used for the disocclusion test below) versus snapping to a
			// low-res cell first.
			const int oldColHi = static_cast<int>(proj.s * Wh);
			const int oldRowHi = static_cast<int>((1.0 - proj.t) * Hh);
			if (oldColHi < 0 || oldColHi >= Wh || oldRowHi < 0 || oldRowHi >= Hh) continue;
			const int oldAnchor = oldRowHi * Wh + oldColHi;
			const size_t oldWpIdx = static_cast<size_t>(oldAnchor) * 4;
			if (m_worldPosHiPrev[oldWpIdx + 3] == 0.0f) continue;  // old cell had no data either

			const float dx = m_worldPosHiPrev[oldWpIdx + 0] - worldPoint.x;
			const float dy = m_worldPosHiPrev[oldWpIdx + 1] - worldPoint.y;
			const float dz = m_worldPosHiPrev[oldWpIdx + 2] - worldPoint.z;
			if (dx * dx + dy * dy + dz * dz > kDisocclusionEpsilonSq) continue;  // different surface

			// Block-translate: the whole upscaleFactor x upscaleFactor block
			// this low-res pixel owns is sourced from the OLD grid's block
			// that CONTAINS the anchor - snapped down to a multiple of
			// upscaleFactor first (oldBaseCol/oldBaseRow), not the raw
			// sub-pixel-precise anchor itself. Without this snap, the source
			// window generally straddles two (or four) different old low-res
			// pixels' own blocks - most reprojected pixels land off a block
			// boundary - splicing cells together that were never disocclusion-
			// tested against each other (only the anchor cell is checked
			// above) and producing visible seam/ghosting artifacts on nearly
			// every camera move. Snapping first keeps the whole block sourced
			// from ONE coherent old block whose own anchor already passed the
			// disocclusion test - a coarser translation (up to
			// upscaleFactor-1 cells of positional slack) but a coherent one,
			// matching this function's own "one rigid motion vector per
			// block" design intent instead of undermining it.
			const int oldBaseCol = (oldColHi / m_upscaleFactor) * m_upscaleFactor;
			const int oldBaseRow = (oldRowHi / m_upscaleFactor) * m_upscaleFactor;
			const int newBaseCol = x * m_upscaleFactor;
			const int newBaseRow = y * m_upscaleFactor;
			for (int cy = 0; cy < m_upscaleFactor; ++cy) {
				const int oldRow = oldBaseRow + cy;
				if (oldRow < 0 || oldRow >= Hh) continue;
				for (int cx = 0; cx < m_upscaleFactor; ++cx) {
					const int oldCol = oldBaseCol + cx;
					if (oldCol < 0 || oldCol >= Wh) continue;
					const int newCell = (newBaseRow + cy) * Wh + (newBaseCol + cx);
					const int oldCell = oldRow * Wh + oldCol;
					const size_t newColorIdx = static_cast<size_t>(newCell) * 3;
					const size_t oldColorIdx = static_cast<size_t>(oldCell) * 3;
					newAccumHi[newColorIdx + 0] = m_accumHi[oldColorIdx + 0];
					newAccumHi[newColorIdx + 1] = m_accumHi[oldColorIdx + 1];
					newAccumHi[newColorIdx + 2] = m_accumHi[oldColorIdx + 2];
					const size_t newWpIdx = static_cast<size_t>(newCell) * 4;
					const size_t oldCellWpIdx = static_cast<size_t>(oldCell) * 4;
					newWorldPosHi[newWpIdx + 0] = m_worldPosHi[oldCellWpIdx + 0];
					newWorldPosHi[newWpIdx + 1] = m_worldPosHi[oldCellWpIdx + 1];
					newWorldPosHi[newWpIdx + 2] = m_worldPosHi[oldCellWpIdx + 2];
					newWorldPosHi[newWpIdx + 3] = m_worldPosHi[oldCellWpIdx + 3];
				}
			}
		}
	}

	// Swap rather than assign - see reprojectAccumulation()'s own identical
	// comment on why (exchanges storage with the persistent scratch members,
	// no allocation).
	std::swap(m_accumHi, newAccumHi);
	std::swap(m_worldPosHi, newWorldPosHi);
}

void RealtimePreviewWorker::start(QString sceneId, int width, int height, double camX, double camY, double camZ,
								   double lookX, double lookY, double lookZ,
								   bool denoise, double denoiseBlend, bool denoiseShowLatest, bool svgf,
								   bool restirGi, bool restirDi, bool probeCache, bool pathGuiding, int spp, int maxDepth, double fireflyClamp,
								   bool temporalUpscale, int temporalUpscaleFactor, bool nrc, bool neuralUpscale,
								   bool dofEnabled, double aperture, double focusDistance) {
	m_sceneId = sceneId;
	m_width = width;
	m_height = height;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_lookX = lookX;
	m_lookY = lookY;
	m_lookZ = lookZ;
	m_denoise = denoise;
	m_denoiseBlend = denoiseBlend;
	m_denoiseShowLatest = denoiseShowLatest;
	// Set directly here rather than relying solely on a separate setSvgf()
	// call from the caller: setSvgf() is gated on m_running (its own header
	// comment - toggling it before Start has ever run would otherwise be
	// silently lost, since m_svgf would stay at its default until the NEXT
	// start() call), same reasoning m_denoise/m_denoiseBlend/
	// m_denoiseShowLatest are already threaded through start()'s own
	// parameter list instead of requiring a follow-up setDenoise() call.
	m_svgf = svgf;
	// Same "set directly, bypass the m_running gate" reasoning as m_svgf
	// above - these three are also threaded through start()'s own parameter
	// list (rather than requiring a follow-up setX() call) so a value set
	// before the very first start() isn't silently lost.
	m_restirGi = restirGi;
	m_restirDi = restirDi;
	m_probeCache = probeCache;
	m_pathGuiding = pathGuiding;
	m_temporalUpscaleEnabled = temporalUpscale;
	m_upscaleFactor = temporalUpscaleFactor;
	m_nrc = nrc;
	m_neuralUpscale = neuralUpscale;
	m_dofEnabled = dofEnabled;
	m_aperture = aperture;
	m_focusDistance = focusDistance;
	m_spp = spp;
	m_maxDepth = maxDepth;
	m_fireflyClamp = fireflyClamp;
	m_cameraDirty = false;
	resetAccumulation();
	m_running = true;
	renderLoop(++m_epoch);
}

void RealtimePreviewWorker::stop() {
	m_running = false;
}

void RealtimePreviewWorker::setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ) {
	if (!m_running) return;
	m_camX = camX;
	m_camY = camY;
	m_camZ = camZ;
	m_lookX = lookX;
	m_lookY = lookY;
	m_lookZ = lookZ;
	m_cameraDirty = true;
}

void RealtimePreviewWorker::setDenoise(bool denoise, double denoiseBlend, bool denoiseShowLatest) {
	if (!m_running) return;
	const bool wasEffectivelyShowingLatest = effectiveShowLatest();
	m_denoise = denoise;
	m_denoiseBlend = denoiseBlend;
	m_denoiseShowLatest = denoiseShowLatest;
	const bool willEffectivelyShowLatest = effectiveShowLatest();
	if (wasEffectivelyShowingLatest != willEffectivelyShowLatest) {
		// Unlike a plain denoise-enabled/blend change (see this method's own
		// header comment), flipping the EFFECTIVE show-latest state changes
		// what m_accum structurally holds - a single frame vs. a genuine
		// running-mean average - so resuming the running mean without a
		// reset would give brand-new real samples almost no weight against
		// whatever single frame is already sitting in m_accum, weighted by
		// however large m_sampleCount had already grown. Reset immediately
		// (this already runs on the worker thread, so unlike setCamera()'s
		// deferred m_cameraDirty flag there's no need to wait for the next
		// renderLoop() iteration).
		resetAccumulation();
	}
}

void RealtimePreviewWorker::setSvgf(bool svgf) {
	if (!m_running) return;
	// Same "does the OVERALL effective show-latest state change" reset
	// trigger as setDenoise() above, generalized to include m_svgf as a
	// second, independent way to reach it (see renderLoop()'s own
	// effectiveShowLatest comment) - toggling m_svgf while denoise's own
	// show-latest is already effective (or vice versa) is a no-op change to
	// the OVERALL flag, so no reset is needed in that case either.
	const bool wasEffectivelyShowingLatest = effectiveShowLatest();
	m_svgf = svgf;
	const bool willEffectivelyShowLatest = effectiveShowLatest();
	if (wasEffectivelyShowingLatest != willEffectivelyShowLatest) {
		resetAccumulation();
	}
}

void RealtimePreviewWorker::setRestirGi(bool restirGi) {
	if (!m_running) return;
	// Unlike setDenoise()/setSvgf(), toggling GI doesn't change what m_accum
	// structurally holds (still a running mean of the same rendered image,
	// just with/without one more resampled indirect-lighting technique
	// contributing to each sample) - no reset needed, same reasoning
	// setExposure() uses.
	m_restirGi = restirGi;
}

void RealtimePreviewWorker::setRestirDi(bool restirDi) {
	if (!m_running) return;
	// Same reasoning as setRestirGi() above: toggling which direct-light
	// sampling technique feeds each new sample doesn't change what m_accum
	// structurally holds, so no reset is needed.
	m_restirDi = restirDi;
}

void RealtimePreviewWorker::setProbeCache(bool probeCache) {
	if (!m_running) return;
	// Same reasoning as setRestirGi()/setRestirDi() above: toggling whether
	// depth>=2 diffuse bounces consult the probe cache doesn't change what
	// m_accum structurally holds, so no reset is needed.
	m_probeCache = probeCache;
}

void RealtimePreviewWorker::setPathGuiding(bool pathGuiding) {
	if (!m_running) return;
	// Same reasoning as setProbeCache() above: toggling whether glossy
	// bounces consult the guiding histogram doesn't change what m_accum
	// structurally holds, so no reset is needed.
	m_pathGuiding = pathGuiding;
}

void RealtimePreviewWorker::setNrc(bool nrc) {
	if (!m_running) return;
	// Same reasoning as setProbeCache() above: toggling whether the Neural
	// Radiance Cache is consulted/trained doesn't change what m_accum
	// structurally holds, so no reset is needed.
	m_nrc = nrc;
}

void RealtimePreviewWorker::setTemporalUpscale(bool enabled, int factor) {
	if (!m_running) return;
	// Unlike setProbeCache()/setPathGuiding() above, changing EITHER of
	// these DOES reset accumulation - see this method's own header comment
	// (realtime_preview_session.h) for why: it changes m_displayImage's own
	// size and the Hi reconstruction buffers' shape, not just what a new
	// sample contains.
	if (enabled != m_temporalUpscaleEnabled || factor != m_upscaleFactor) {
		m_temporalUpscaleEnabled = enabled;
		m_upscaleFactor = factor;
		resetAccumulation();
	}
}

void RealtimePreviewWorker::setNeuralUpscale(bool enabled) {
	if (!m_running) return;
	// Same "changes which reconstruction buffer feeds m_displayImage" reset
	// reasoning as setTemporalUpscale() just above.
	if (enabled != m_neuralUpscale) {
		m_neuralUpscale = enabled;
		resetAccumulation();
	}
}

void RealtimePreviewWorker::setDof(bool enabled, double aperture, double focusDistance) {
	if (!m_running) return;
	// Unlike setFireflyClamp() below, this DOES reset accumulation - see
	// this method's own header comment (realtime_preview_session.h): it
	// changes the camera rays themselves (sharp vs. blurred), not a post-
	// process, so mixing pre-/post-change samples in one running mean
	// would look wrong. Only resets when something actually changed, same
	// "no-op if nothing moved" guard setTemporalUpscale()/setNeuralUpscale()
	// use - the aperture/focusDistance values are irrelevant (and ignored
	// downstream) while enabled is false, so a value-only change while
	// disabled must not trigger a reset either.
	const bool changed = (enabled != m_dofEnabled) ||
		(enabled && (aperture != m_aperture || focusDistance != m_focusDistance));
	m_dofEnabled = enabled;
	m_aperture = aperture;
	m_focusDistance = focusDistance;
	if (changed) resetAccumulation();
}

void RealtimePreviewWorker::setSppAndMaxDepth(int spp, int maxDepth) {
	if (!m_running) return;
	m_spp = spp;
	m_maxDepth = maxDepth;
}

void RealtimePreviewWorker::setFireflyClamp(double fireflyClamp) {
	if (!m_running) return;
	m_fireflyClamp = fireflyClamp;
}

void RealtimePreviewWorker::setSvgfTuning(double temporalAlpha, double maxHistoryLength,
										   double varianceBootstrapFrames, int varianceBootstrapRadius,
										   double sigmaNormal, double sigmaDepth, double sigmaLuminance,
										   int atrousRadius, double minAlbedo, int atrousPasses) {
	// NOT gated on m_running - see this method's own header comment (mirrors
	// setExposure()'s reasoning): the caller pushes this BEFORE start() so
	// the first frame already reflects it.
	m_svgfTemporalAlpha = temporalAlpha;
	m_svgfMaxHistoryLength = maxHistoryLength;
	m_svgfVarianceBootstrapFrames = varianceBootstrapFrames;
	m_svgfVarianceBootstrapRadius = varianceBootstrapRadius;
	m_svgfSigmaNormal = sigmaNormal;
	m_svgfSigmaDepth = sigmaDepth;
	m_svgfSigmaLuminance = sigmaLuminance;
	m_svgfAtrousRadius = atrousRadius;
	m_svgfMinAlbedo = minAlbedo;
	m_svgfAtrousPasses = atrousPasses;
}

void RealtimePreviewWorker::setExposure(double exposure) {
	// Unlike setDenoise(), not gated on m_running: this is a pure display
	// multiply with no accumulation-structure side effect (see this method's
	// own header comment for why no reset is needed either), so it's safe -
	// and useful - to accept a value before start() as well as while running,
	// letting the caller push the Settings tab's own initial value up front.
	m_exposure = exposure;
}

void RealtimePreviewWorker::renderLoop(int epoch) {
	// epoch != m_epoch means a stop()+start() cycle already happened since
	// THIS continuation was posted (start() bumps m_epoch) - it belongs to
	// an old, already-superseded chain and must not run at all, let alone
	// repost itself, or it would keep running forever alongside the new
	// chain start() began. See m_epoch's own header comment for the exact
	// race this closes.
	if (!m_running || epoch != m_epoch) return;

	// Captured before clearing: renders the NEXT frame with the already-
	// updated (new) m_camX/etc below, then - once that frame's own world
	// positions are in - reprojectAccumulation() uses THIS flag to decide
	// whether to remap the OLD accumulation into the new view first. No
	// immediate resetAccumulation() here anymore - that used to be this
	// block's whole job before reprojection replaced it.
	const bool cameraJustMoved = m_cameraDirty;
	m_cameraDirty = false;

	// Computed here (before the render call, not after) so the value SENT
	// to the GPU this frame matches the value the CPU side will actually
	// act on once the call returns - see useTemporalUpscale()'s own comment.
	// Sending the raw m_temporalUpscaleEnabled instead (this function used
	// to) meant the GPU's own checkerboardActive gate
	// (wavefront_path_tracer.cpp) could see "upscale on" and defensively
	// disable SVGF's checkerboard optimization even on a frame where the
	// CPU had already fallen back to ordinary SVGF display because
	// effectiveShowLatest() was true - a real, confirmed mismatch (this
	// project's own code review caught it), not just a theoretical one.
	const bool useUpscale = useTemporalUpscale();

	RenderFrameFn renderFrame = handle().renderFrameFn;
	bool ok = false;
	if (!renderFrame) {
		emit statusChanged(QStringLiteral("realtime_renderer.dll not found or missing its export"));
		m_running = false;
	} else {
		// Low-spp samples per loop iteration (GUI-configurable, default 1/8 -
		// see this class's own header comment on why a small per-call cost
		// keeps the loop responsive to stop()/setCamera() between frames, and
		// frameNumber_'s own self-incrementing seed (WavefrontPathTracer, see
		// optix_interface.h's rt_realtime_render_frame() comment) already
		// decorrelates noise across calls, so accumulating many low-spp calls
		// converges the same way one big N-spp call would.
		SvgfTuningParams svgfTuning;
		svgfTuning.temporalAlpha = static_cast<float>(m_svgfTemporalAlpha);
		svgfTuning.maxHistoryLength = static_cast<float>(m_svgfMaxHistoryLength);
		svgfTuning.varianceBootstrapFrames = static_cast<float>(m_svgfVarianceBootstrapFrames);
		svgfTuning.varianceBootstrapRadius = m_svgfVarianceBootstrapRadius;
		svgfTuning.sigmaNormal = static_cast<float>(m_svgfSigmaNormal);
		svgfTuning.sigmaDepth = static_cast<float>(m_svgfSigmaDepth);
		svgfTuning.sigmaLuminance = static_cast<float>(m_svgfSigmaLuminance);
		svgfTuning.atrousRadius = m_svgfAtrousRadius;
		svgfTuning.minAlbedo = static_cast<float>(m_svgfMinAlbedo);
		svgfTuning.atrousPasses = m_svgfAtrousPasses;
		ok = renderFrame(m_sceneId.toUtf8().constData(), m_width, m_height, m_spp, m_maxDepth,
						  m_camX, m_camY, m_camZ,
						  /*has_custom_lookat=*/true, m_lookX, m_lookY, m_lookZ,
						  m_denoise, m_denoiseBlend,
						  m_worldPos.data(), m_cameraBasis.data(),
						  m_tmp.data(), m_svgf, m_restirGi, static_cast<float>(m_fireflyClamp),
						  reinterpret_cast<const void*>(&svgfTuning), m_restirDi, m_probeCache, m_pathGuiding,
						  useUpscale, m_upscaleFactor, m_temporalJitterCounter, m_nrc,
						  m_neuralUpscale, (useUpscale && m_neuralUpscale) ? m_neuralUpscaleOut.data() : nullptr,
						  m_dofEnabled ? m_aperture : -1.0, m_dofEnabled ? m_focusDistance : -1.0);
		if (!ok) {
			QString message = QStringLiteral("Render failed - scene may not be GPU-supported, "
											  "or the wavefront backend is unavailable");
			// Extra detail when the failure was a caught exception (CUDA/OptiX
			// error, etc.) - see rt_realtime_get_last_error()'s own comment.
			// "" for an ordinary false return (already fully described by the
			// generic message above), so no redundant empty parenthetical.
			GetLastErrorFn getLastError = handle().getLastErrorFn;
			if (getLastError) {
				const char* detail = getLastError();
				if (detail && *detail) {
					message += QStringLiteral(" (%1)").arg(QString::fromUtf8(detail));
				}
			}
			emit statusChanged(message);
		}
	}

	if (ok) {
		// m_denoise is part of the first term (not m_denoiseShowLatest
		// alone): without denoise, "show the latest raw single-sample frame
		// instead of accumulating" would mean Live Preview never converges
		// at all - see setDenoise()'s own comment on why toggling either
		// side of that term resets accumulation. m_svgf is unconditionally
		// its own reason to show-latest (no equivalent "without X this would
		// never converge" gate): SVGF's own GPU-side temporal integration
		// (gpu/optix/wavefront_svgf_math.h) already IS the accumulation -
		// m_tmp is already temporally stable by the time it reaches here,
		// and re-blending it into m_accum's own separate running mean would
		// double-integrate the same signal through two different, competing
		// temporal filters - see this project's own SVGF plan for why GPU-
		// side integration REPLACES this CPU-side one for that mode, rather
		// than sitting on top of it.
		const bool showLatest = effectiveShowLatest();
		// useUpscale was already computed above, before the render call -
		// see its own comment there for why (must match what was actually
		// sent to the GPU this frame).

		// Reprojection would be immediately thrown away by the show-latest
		// branch below (which overwrites m_accum wholesale every frame
		// regardless), so skip the work entirely in that mode.
		if (cameraJustMoved) {
			// Neural reconstruction does its own reprojection GPU-side every
			// frame regardless (gpu/optix/wavefront_kernels_upscale.cu's own
			// upscale_infer, via wf_restir_reproject_prev_pixel) - this CPU-
			// side reprojection exists only for the OLD block-splat path's
			// own m_accumHi/m_worldPosHi, which m_neuralUpscale bypasses
			// entirely (see the useUpscale branch below).
			if (useUpscale && !m_neuralUpscale) {
				reprojectAccumulationHi();
			} else if (!useUpscale && !showLatest) {
				reprojectAccumulation();
			}
		}

		if (useUpscale && m_neuralUpscale) {
			// --- Neural temporal upscale ---
			// The GPU already produced the final high-res result
			// (m_neuralUpscaleOut, filled via renderFrame()'s own
			// out_neural_upscale_buffer parameter) - just tonemap it
			// directly. No CPU-side reprojection/block-splat/reconstruction
			// at all in this mode - see this project's own plan for why this
			// REPLACES the old-path block below rather than augmenting it.
			m_sampleCount = 1;
			const int Wh = m_width * m_upscaleFactor;
			const int Hh = m_height * m_upscaleFactor;
			for (int y = 0; y < Hh; ++y) {
				uchar* row = m_displayImage.scanLine(y);
				for (int x = 0; x < Wh; ++x) {
					const int cell = y * Wh + x;
					const size_t idx = static_cast<size_t>(cell) * 3;
					double r = m_neuralUpscaleOut[idx + 0];
					double g = m_neuralUpscaleOut[idx + 1];
					double b = m_neuralUpscaleOut[idx + 2];
					if (!std::isfinite(r)) r = 0.0;
					if (!std::isfinite(g)) g = 0.0;
					if (!std::isfinite(b)) b = 0.0;
					r *= m_exposure; g *= m_exposure; b *= m_exposure;
					r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
					g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
					b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
					row[x * 3 + 0] = static_cast<uchar>(std::fmin(std::fmax(r, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 1] = static_cast<uchar>(std::fmin(std::fmax(g, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 2] = static_cast<uchar>(std::fmin(std::fmax(b, 0.0), 1.0) * 255.0 + 0.5);
				}
			}
			// Still track prev camera/world-pos bookkeeping, matching the old
			// path's own end-of-branch update just below, so toggling neural
			// reconstruction OFF mid-session leaves the old path's own
			// reprojection with a sane, up-to-date starting point next frame
			// rather than a stale one from whenever neural mode was enabled.
			m_worldPosPrev = m_worldPos;
			m_prevCameraBasis = m_cameraBasis;

			const unsigned int neuralPeriod = static_cast<unsigned int>(m_upscaleFactor * m_upscaleFactor);
			m_temporalJitterCounter = (neuralPeriod > 0) ? ((m_temporalJitterCounter + static_cast<unsigned int>(m_spp)) % neuralPeriod) : 0;
			m_temporalUpscaleWriteCounter = (neuralPeriod > 0) ? ((m_temporalUpscaleWriteCounter + 1u) % neuralPeriod) : 0;
		} else if (useUpscale) {
			// --- Temporal upscale: write step + reconstruction ---
			// This call's own place in the write step's OWN sequence
			// (m_temporalUpscaleWriteCounter - see its own comment on the
			// header for why this is a SEPARATE counter from
			// m_temporalJitterCounter) selects ONE sub-cell shared by every
			// low-res pixel this frame - each low-res pixel's raw sample
			// (m_tmp, not the running mean) overwrites that one high-res
			// cell, never blended (see this project's own plan for why a
			// running mean doesn't apply at the per-cell level the way it
			// does for m_accum). NOTE: when Samples/Frame > 1, m_tmp is
			// already an average across several GPU-side jitter phases -
			// the write step below still treats it as if it were a single
			// phase, a known v1 simplification that slightly blurs the
			// splatted cell at Samples/Frame > 1 (recommend Samples/Frame=1
			// for the sharpest reconstruction). This blur is independent of
			// - and much less severe than - full-grid coverage, which is
			// what m_temporalUpscaleWriteCounter's own separate, always-by-1
			// advance guarantees regardless of Samples/Frame.
			const int Wh = m_width * m_upscaleFactor;
			const int Hh = m_height * m_upscaleFactor;
			int subCx = 0, subCy = 0;
			camera_math::temporalUpscaleSubcell(m_temporalUpscaleWriteCounter, m_upscaleFactor, subCx, subCy);
			for (int y = 0; y < m_height; ++y) {
				for (int x = 0; x < m_width; ++x) {
					const int pixel = y * m_width + x;
					const int target = (y * m_upscaleFactor + subCy) * Wh + (x * m_upscaleFactor + subCx);
					const size_t srcColorIdx = static_cast<size_t>(pixel) * 3;
					const size_t dstColorIdx = static_cast<size_t>(target) * 3;
					m_accumHi[dstColorIdx + 0] = m_tmp[srcColorIdx + 0];
					m_accumHi[dstColorIdx + 1] = m_tmp[srcColorIdx + 1];
					m_accumHi[dstColorIdx + 2] = m_tmp[srcColorIdx + 2];
					const size_t srcWpIdx = static_cast<size_t>(pixel) * 4;
					const size_t dstWpIdx = static_cast<size_t>(target) * 4;
					m_worldPosHi[dstWpIdx + 0] = m_worldPos[srcWpIdx + 0];
					m_worldPosHi[dstWpIdx + 1] = m_worldPos[srcWpIdx + 1];
					m_worldPosHi[dstWpIdx + 2] = m_worldPos[srcWpIdx + 2];
					m_worldPosHi[dstWpIdx + 3] = m_worldPos[srcWpIdx + 3];
				}
			}
			// No uniform per-pixel convergence metric applies once each
			// high-res cell converges to a single direct sample rather than
			// a running mean - same "1" showLatest already reports for the
			// analogous "no meaningful running-mean count" case below.
			m_sampleCount = 1;

			// This frame's world-pos/camera-basis (and the just-updated Hi
			// buffers) become "the data backing the display" for whenever
			// the NEXT camera move needs to reproject FROM it - see
			// reprojectAccumulationHi()'s own comment.
			m_worldPosHiPrev = m_worldPosHi;
			m_worldPosPrev = m_worldPos;
			m_prevCameraBasis = m_cameraBasis;

			// Tonemap the Hi buffer into m_displayImage (already sized
			// Wh x Hh, see resetAccumulation()). A cell whose m_worldPosHi
			// validity is still 0 (never splatted into - e.g. right after
			// enabling the feature) falls back to a plain nearest-neighbor
			// upscale of THIS frame's own m_tmp - display-only, never
			// written into m_accumHi itself, so it never contaminates
			// history once a real splat arrives (see this project's own
			// plan's first-frame/cold-cell fallback).
			for (int y = 0; y < Hh; ++y) {
				uchar* row = m_displayImage.scanLine(y);
				const int lowY = std::min(y / m_upscaleFactor, m_height - 1);
				for (int x = 0; x < Wh; ++x) {
					const int cell = y * Wh + x;
					const size_t wpIdx = static_cast<size_t>(cell) * 4;
					const size_t idx = static_cast<size_t>(cell) * 3;
					double r, g, b;
					if (m_worldPosHi[wpIdx + 3] != 0.0f) {
						r = m_accumHi[idx + 0]; g = m_accumHi[idx + 1]; b = m_accumHi[idx + 2];
					} else {
						const int lowX = std::min(x / m_upscaleFactor, m_width - 1);
						const size_t lowIdx = (static_cast<size_t>(lowY) * m_width + lowX) * 3;
						r = m_tmp[lowIdx + 0]; g = m_tmp[lowIdx + 1]; b = m_tmp[lowIdx + 2];
					}
					if (!std::isfinite(r)) r = 0.0;
					if (!std::isfinite(g)) g = 0.0;
					if (!std::isfinite(b)) b = 0.0;
					r *= m_exposure; g *= m_exposure; b *= m_exposure;
					r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
					g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
					b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
					row[x * 3 + 0] = static_cast<uchar>(std::fmin(std::fmax(r, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 1] = static_cast<uchar>(std::fmin(std::fmax(g, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 2] = static_cast<uchar>(std::fmin(std::fmax(b, 0.0), 1.0) * 255.0 + 0.5);
				}
			}

			// Advance this call's own place in the GPU-side jitter sequence
			// by m_spp (matching how many samples-per-call this render just
			// consumed, the same per-call advance frameNumber_ itself
			// already gets), wrapped modulo the sequence's own period.
			const unsigned int period = static_cast<unsigned int>(m_upscaleFactor * m_upscaleFactor);
			m_temporalJitterCounter = (period > 0) ? ((m_temporalJitterCounter + static_cast<unsigned int>(m_spp)) % period) : 0;
			// Advance the write step's OWN counter by exactly 1 - see its
			// declaration's own comment for why this must NOT track m_spp.
			m_temporalUpscaleWriteCounter = (period > 0) ? ((m_temporalUpscaleWriteCounter + 1u) % period) : 0;
		} else {
			int minSampleCount = 0;
			if (showLatest) {
				// Skip accumulation entirely - each already-denoised frame is
				// clean enough on its own that averaging it with older, possibly
				// differently-denoised frames would only add lag, not quality.
				m_accum = m_tmp;
				std::fill(m_sampleCounts.begin(), m_sampleCounts.end(), uint16_t{1});
				minSampleCount = 1;
			} else {
				// Running mean: accum += (sample - accum) / (n+1), n now READ
				// PER PIXEL (m_sampleCounts) rather than one shared scalar -
				// reprojectAccumulation() just above can leave different pixels
				// with wildly different effective history (0 for a freshly
				// disoccluded pixel, carried-forward-and-capped for a
				// successfully reprojected one). Both buffers are linear RGB
				// (rt_realtime_render_frame()'s own contract), so this is still
				// a plain per-channel average - no dividing/multiplying needed
				// beyond this, unlike CPU/GPU's own filter-weighted
				// reconstruction (this preview uses a trivial 1-sample-per-pixel
				// box filter, no splatting).
				constexpr uint16_t kMaxSampleCount = 65535;
				minSampleCount = kMaxSampleCount;
				const int numPixels = m_width * m_height;
				for (int pixel = 0; pixel < numPixels; ++pixel) {
					const int n = m_sampleCounts[pixel];
					const size_t idx = static_cast<size_t>(pixel) * 3;
					for (int c = 0; c < 3; ++c) {
						m_accum[idx + c] += (m_tmp[idx + c] - m_accum[idx + c]) / static_cast<float>(n + 1);
					}
					if (m_sampleCounts[pixel] < kMaxSampleCount) ++m_sampleCounts[pixel];
					minSampleCount = std::min<int>(minSampleCount, m_sampleCounts[pixel]);
				}
			}
			m_sampleCount = minSampleCount;

			// This frame's world-pos/camera-basis become "the data backing
			// m_accum" for whenever the NEXT camera move needs to reproject
			// FROM it - see the member declarations' own comment.
			m_worldPosPrev = m_worldPos;
			m_prevCameraBasis = m_cameraBasis;

			// Tonemap the ACCUMULATED result (ACES + sRGB, matching this
			// project's CPU/GPU display convention exactly - see tone_map.h)
			// into m_displayImage IN PLACE. Tonemapping the per-call noisy
			// sample instead would defeat the whole point of accumulating in
			// linear space first.
			for (int y = 0; y < m_height; ++y) {
				uchar* row = m_displayImage.scanLine(y);
				for (int x = 0; x < m_width; ++x) {
					const size_t idx = (static_cast<size_t>(y) * m_width + x) * 3;
					double r = m_accum[idx + 0], g = m_accum[idx + 1], b = m_accum[idx + 2];
					if (!std::isfinite(r)) r = 0.0;
					if (!std::isfinite(g)) g = 0.0;
					if (!std::isfinite(b)) b = 0.0;
					// Same exposure multiply the batch/CLI path applies right
					// before its own identical ACES+sRGB tonemap (optix_interface.cpp) -
					// see m_exposure's own comment for why Live Preview needs this
					// pulled down further than batch's default for the same scene.
					r *= m_exposure; g *= m_exposure; b *= m_exposure;
					r = linear_to_srgb(apply_tone_map(r, ToneMapMode::ACES));
					g = linear_to_srgb(apply_tone_map(g, ToneMapMode::ACES));
					b = linear_to_srgb(apply_tone_map(b, ToneMapMode::ACES));
					row[x * 3 + 0] = static_cast<uchar>(std::fmin(std::fmax(r, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 1] = static_cast<uchar>(std::fmin(std::fmax(g, 0.0), 1.0) * 255.0 + 0.5);
					row[x * 3 + 2] = static_cast<uchar>(std::fmin(std::fmax(b, 0.0), 1.0) * 255.0 + 0.5);
				}
			}
		}

		// .copy() rather than emitting m_displayImage directly: QImage is
		// implicitly shared, and frameReady() carries this image across a
		// queued cross-thread signal to the GUI thread - without a detach,
		// the NEXT loop iteration's in-place scanLine() writes (above,
		// possibly before the GUI thread has consumed the queued copy)
		// would race with whatever the GUI thread reads from the "same"
		// shared buffer. .copy() gives this emit its own buffer up front,
		// letting m_displayImage keep being mutated in place next iteration
		// with no such hazard.
		//
		// m_sampleCount * m_spp, not m_sampleCount alone: m_sampleCount is a
		// BATCH count (see its own comment) - each batch already IS an
		// m_spp-sample average from a single render call, so the status
		// label needs this multiply to show the true number of samples
		// traced, not the number of calls made. Scaling by the CURRENT
		// m_spp rather than tracking a separate running total keeps this a
		// pure display fix with no new per-pixel state to reproject/cap -
		// the one imprecision this trades away is retroactive: if the user
		// changes Samples/Frame mid-session, batches accumulated at the OLD
		// value get re-reported as if they'd used the new one too. Harmless
		// for a live status number nobody is auditing frame-by-frame.
		emit frameReady(m_displayImage.copy(), m_sampleCount * m_spp);
	}

	if (m_running) QMetaObject::invokeMethod(this, [this, epoch]() { renderLoop(epoch); }, Qt::QueuedConnection);
}

// ============================================================================
// RealtimePreviewSession
// ============================================================================

RealtimePreviewSession::RealtimePreviewSession(QObject *parent) : QObject(parent) {
	m_worker = new RealtimePreviewWorker();
	m_worker->moveToThread(&m_thread);
	// Worker is parent-less and owned by the thread's own finished() cleanup
	// below, not by this QObject - it must not be destroyed from the GUI
	// thread while the worker thread might still be running a queued call
	// against it.
	connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
	connect(m_worker, &RealtimePreviewWorker::frameReady, this, &RealtimePreviewSession::frameReady);
	connect(m_worker, &RealtimePreviewWorker::statusChanged, this, &RealtimePreviewSession::statusChanged);
	m_thread.start();
}

RealtimePreviewSession::~RealtimePreviewSession() {
	stop();
	m_thread.quit();
	m_thread.wait();
}

void RealtimePreviewSession::start(const QString &sceneId, int width, int height, double camX, double camY, double camZ,
									double lookX, double lookY, double lookZ,
									bool denoise, double denoiseBlend, bool denoiseShowLatest, bool svgf,
									bool restirGi, bool restirDi, bool probeCache, bool pathGuiding, int spp, int maxDepth, double fireflyClamp,
									bool temporalUpscale, int temporalUpscaleFactor, bool nrc, bool neuralUpscale,
									bool dofEnabled, double aperture, double focusDistance) {
	QMetaObject::invokeMethod(m_worker, "start", Qt::QueuedConnection,
		Q_ARG(QString, sceneId), Q_ARG(int, width), Q_ARG(int, height),
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ),
		Q_ARG(double, lookX), Q_ARG(double, lookY), Q_ARG(double, lookZ),
		Q_ARG(bool, denoise), Q_ARG(double, denoiseBlend), Q_ARG(bool, denoiseShowLatest), Q_ARG(bool, svgf),
		Q_ARG(bool, restirGi), Q_ARG(bool, restirDi), Q_ARG(bool, probeCache), Q_ARG(bool, pathGuiding),
		Q_ARG(int, spp), Q_ARG(int, maxDepth), Q_ARG(double, fireflyClamp),
		Q_ARG(bool, temporalUpscale), Q_ARG(int, temporalUpscaleFactor), Q_ARG(bool, nrc), Q_ARG(bool, neuralUpscale),
		Q_ARG(bool, dofEnabled), Q_ARG(double, aperture), Q_ARG(double, focusDistance));
}

void RealtimePreviewSession::stop() {
	QMetaObject::invokeMethod(m_worker, "stop", Qt::QueuedConnection);
}

void RealtimePreviewSession::setCamera(double camX, double camY, double camZ, double lookX, double lookY, double lookZ) {
	QMetaObject::invokeMethod(m_worker, "setCamera", Qt::QueuedConnection,
		Q_ARG(double, camX), Q_ARG(double, camY), Q_ARG(double, camZ),
		Q_ARG(double, lookX), Q_ARG(double, lookY), Q_ARG(double, lookZ));
}

void RealtimePreviewSession::setDenoise(bool denoise, double denoiseBlend, bool denoiseShowLatest) {
	QMetaObject::invokeMethod(m_worker, "setDenoise", Qt::QueuedConnection,
		Q_ARG(bool, denoise), Q_ARG(double, denoiseBlend), Q_ARG(bool, denoiseShowLatest));
}

void RealtimePreviewSession::setSvgf(bool svgf) {
	QMetaObject::invokeMethod(m_worker, "setSvgf", Qt::QueuedConnection, Q_ARG(bool, svgf));
}

void RealtimePreviewSession::setExposure(double exposure) {
	QMetaObject::invokeMethod(m_worker, "setExposure", Qt::QueuedConnection, Q_ARG(double, exposure));
}

void RealtimePreviewSession::setRestirGi(bool restirGi) {
	QMetaObject::invokeMethod(m_worker, "setRestirGi", Qt::QueuedConnection, Q_ARG(bool, restirGi));
}

void RealtimePreviewSession::setRestirDi(bool restirDi) {
	QMetaObject::invokeMethod(m_worker, "setRestirDi", Qt::QueuedConnection, Q_ARG(bool, restirDi));
}

void RealtimePreviewSession::setProbeCache(bool probeCache) {
	QMetaObject::invokeMethod(m_worker, "setProbeCache", Qt::QueuedConnection, Q_ARG(bool, probeCache));
}

void RealtimePreviewSession::setPathGuiding(bool pathGuiding) {
	QMetaObject::invokeMethod(m_worker, "setPathGuiding", Qt::QueuedConnection, Q_ARG(bool, pathGuiding));
}

void RealtimePreviewSession::setNrc(bool nrc) {
	QMetaObject::invokeMethod(m_worker, "setNrc", Qt::QueuedConnection, Q_ARG(bool, nrc));
}

void RealtimePreviewSession::setTemporalUpscale(bool enabled, int factor) {
	QMetaObject::invokeMethod(m_worker, "setTemporalUpscale", Qt::QueuedConnection, Q_ARG(bool, enabled), Q_ARG(int, factor));
}

void RealtimePreviewSession::setNeuralUpscale(bool enabled) {
	QMetaObject::invokeMethod(m_worker, "setNeuralUpscale", Qt::QueuedConnection, Q_ARG(bool, enabled));
}

void RealtimePreviewSession::setSppAndMaxDepth(int spp, int maxDepth) {
	QMetaObject::invokeMethod(m_worker, "setSppAndMaxDepth", Qt::QueuedConnection, Q_ARG(int, spp), Q_ARG(int, maxDepth));
}

void RealtimePreviewSession::setFireflyClamp(double fireflyClamp) {
	QMetaObject::invokeMethod(m_worker, "setFireflyClamp", Qt::QueuedConnection, Q_ARG(double, fireflyClamp));
}

void RealtimePreviewSession::setDof(bool enabled, double aperture, double focusDistance) {
	QMetaObject::invokeMethod(m_worker, "setDof", Qt::QueuedConnection,
		Q_ARG(bool, enabled), Q_ARG(double, aperture), Q_ARG(double, focusDistance));
}

void RealtimePreviewSession::setSvgfTuning(double temporalAlpha, double maxHistoryLength,
											double varianceBootstrapFrames, int varianceBootstrapRadius,
											double sigmaNormal, double sigmaDepth, double sigmaLuminance,
											int atrousRadius, double minAlbedo, int atrousPasses) {
	QMetaObject::invokeMethod(m_worker, "setSvgfTuning", Qt::QueuedConnection,
		Q_ARG(double, temporalAlpha), Q_ARG(double, maxHistoryLength), Q_ARG(double, varianceBootstrapFrames),
		Q_ARG(int, varianceBootstrapRadius), Q_ARG(double, sigmaNormal), Q_ARG(double, sigmaDepth),
		Q_ARG(double, sigmaLuminance), Q_ARG(int, atrousRadius), Q_ARG(double, minAlbedo), Q_ARG(int, atrousPasses));
}
