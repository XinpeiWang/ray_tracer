// wavefront_path_tracer.cpp
// WavefrontPathTracer host-side implementation.
//
// Drives the wavefront render loop:
//   For each sample:
//     1. generate_camera_rays kernel
//     For each bounce (until queue is empty):
//       2. optixLaunch(intersectPipeline)  -- fills hitQueue + missQueue
//       3. evaluate_materials kernel       -- fills shadowQueue + nextRayQueue
//       4. accumulate_miss kernel
//       5. optixLaunch(shadowPipeline)     -- fills transmittance[] float array
//       6. accumulate_shadow kernel
//       7. swap ray / nextRay queues
//   normalize_framebuffer kernel (once, after all samples)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_path_tracer.h"
#include "optix_types.h"
#include "../../src/data/cie_data.h"
#include "rgb_to_spectrum_table.h"
#include <optix_stack_size.h>
#include "optix_module_parallel.h"
#include <cuda.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <random>  // World-space irradiance probe cache - see launchProbeCacheUpdate()'s own comment

// Declarations for wavefront_launch.cu's C wrappers (no <<<>>> in .cpp) -
// shared with wavefront_launch.cu itself via wavefront_launch.h, rather than
// hand-typed separately in each file (extern "C" linkage isn't type-checked
// across translation units, so two independently-typed copies could drift
// out of sync silently).
#include "wavefront_launch.h"
#include "wavefront_temporal_upscale_math.h"

namespace optix_renderer {

// ============================================================================
// Constructor / Destructor
// ============================================================================

WavefrontPathTracer::WavefrontPathTracer() = default;

WavefrontPathTracer::~WavefrontPathTracer() {
	cleanup();
}

// ============================================================================
// render — main wavefront render loop
// ============================================================================

// Render-time instrumentation (pbrt-v4 STAT_COUNTER-inspired, see this
// project's own plan for why wavefront is the one backend that gets
// this for free: every field here is already a real, host-visible
// readQueueSize() result computed for launch sizing every bounce -
// summing them into a report is the only new work, no new device-side
// counters). Printed once at the end of render() as a [WF-STATS] block.
struct WavefrontRenderStats {
	long long primaryRays = 0;      // sum of numRays across all bounces
	long long hits = 0;             // regular hitQueue
	long long simpleHits = 0;       // simpleHitQueue (Lambertian/Metal)
	long long dielectricHits = 0;   // dielectricHitQueue
	long long misses = 0;
	long long shadowRays = 0;
	long long probeRays = 0;        // BSSRDF probe walk
	long long probeExits = 0;
	long long bounceIterations = 0; // total inner-loop iterations across all samples
	int samplesCompleted = 0;
};

struct RenderCall {
	int width;
	int height;
	int samples_per_pixel;
	int max_depth;
	const GpuCameraParams& camera;
	float*  framebuffer;
	OptixTraversableHandle gas_handle;
	CUdeviceptr d_materials;
	CUdeviceptr d_spheres;
	CUdeviceptr d_quads;
	CUdeviceptr d_light_indices;
	CUdeviceptr d_lightKinds;
	CUdeviceptr d_alias_table;
	unsigned int num_materials;
	unsigned int num_spheres;
	unsigned int num_quads;
	unsigned int num_lights;
	CUdeviceptr d_punctual_lights;
	unsigned int num_punctual_lights;
	CUdeviceptr d_bilinear_patches;
	unsigned int num_bilinear_patches;
	CUdeviceptr d_triangles;
	unsigned int num_triangles;
	CUdeviceptr d_disks;
	unsigned int num_disks;
	CUdeviceptr d_cylinders;
	unsigned int num_cylinders;
	int numPixels{};
	bool regularize{};
	CUdeviceptr d_fb{};
	float3* d_fbPtr{};
	CUdeviceptr d_weight{};
	float* d_weightPtr{};
	CUdeviceptr d_activePixelMaskPtr{};
	bool needsAovGuideBuffers{};
	bool needsWorldPosHistory{};
	float3* d_albedoAovPtr{};
	float3* d_normalAovPtr{};
	bool checkerboardActive{};
	WavefrontLaunchParams lp{};
	int progressPrintInterval{};
	WavefrontRenderStats stats{};
};

bool WavefrontPathTracer::renderPrepareBuffers(RenderCall &rc) {
	int width = rc.width;
	int height = rc.height;
	const auto &camera = rc.camera;
	unsigned int num_punctual_lights = rc.num_punctual_lights;
	const int numPixels = width * height;
	// See restirImageWidth_/restirImageHeight_'s own header comment.
	restirImageWidth_ = width;
	restirImageHeight_ = height;

	// Integrator "bool regularize" - camera.regularize is stored as int
	// (GpuCameraParams is __constant__-safe, see that struct's own comment),
	// converted once here and reused at both launchEvaluateMaterials*() call
	// sites below rather than re-converting the same fixed-for-this-render
	// value at each site.
	const bool regularize = camera.regularize != 0;

	// --seed (CLI/GUI): camera.userSeed >= 0 is an explicit request for a
	// reproducible render (see GpuCameraParams::userSeed's own comment,
	// optix_types.h). frameNumber_ is this backend's own RNG seed input
	// (wavefront_kernels.cu's wf_pcg(...^ frameNumber)) - unlike the
	// recursive backend's frameNumber, which optix_renderer_render.cpp
	// hardcodes to 0 fresh on every render() call, THIS one is a
	// self-incrementing member (frameNumber_++ below, per sample) on a
	// long-lived WavefrontPathTracer instance, so without this reset a
	// second render() call in the same process would silently inherit
	// whatever value the first render left it at, not the user-chosen
	// seed. -1 (no --seed requested) leaves it exactly as it was -
	// today's pre-existing behavior, whatever that happens to be.
	if (camera.userSeed >= 0) frameNumber_ = static_cast<unsigned int>(camera.userSeed);


	if (!allocateQueues(numPixels, static_cast<int>(num_punctual_lights))) return false;

	// Framebuffer accumulator + per-pixel filter-weight buffer - persisted
	// across calls and only reallocated when numPixels changes (same
	// resolution-keyed skip-reallocation shape as allocateQueues() above),
	// so a tight repeated-call loop at a fixed resolution (e.g. a live
	// preview) isn't paying a cudaMalloc/cudaFree pair every single call.
	// Still zeroed every render() regardless - these accumulate per-call,
	// not across calls.
	if (fbCapacity_ != numPixels) {
		if (d_fb_) { cudaFree(reinterpret_cast<void*>(d_fb_)); d_fb_ = 0; }
		if (d_weight_) { cudaFree(reinterpret_cast<void*>(d_weight_)); d_weight_ = 0; }
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fb_), numPixels * sizeof(float3)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight_), numPixels * sizeof(float)));
		fbCapacity_ = numPixels;
	}
	CUdeviceptr d_fb = d_fb_;
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_fb), 0,
							   numPixels * sizeof(float3), stream_));

	float3* d_fbPtr = reinterpret_cast<float3*>(d_fb);

	// Per-pixel sum of this render's filter weights (pbrt-v4 film
	// reconstruction formula). See generate_camera_rays's own filter_w
	// comment and normalize_framebuffer's own comment.
	CUdeviceptr d_weight = d_weight_;
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_weight), 0,
							   numPixels * sizeof(float), stream_));
	float* d_weightPtr = reinterpret_cast<float*>(d_weight);

	// Live Preview's adaptive sampling - see setActivePixelMask()'s own
	// comment. Only allocated/uploaded when the caller actually supplied a
	// mask THIS call; nullptr is forwarded to launchGenerateCameraRays()
	// otherwise (wf_adaptive_pixel_active()'s own "no mask -> always active"
	// contract), so a feature-off caller (or one that hasn't called
	// setActivePixelMask() yet) pays nothing beyond this one branch. Unlike
	// d_fb_/d_weight_ above, this buffer is not zeroed here - it's fully
	// overwritten by the memcpy below every time it's used at all.
	CUdeviceptr d_activePixelMaskPtr = 0;
	if (activePixelMaskHost_) {
		if (activePixelMaskCapacity_ != numPixels) {
			if (d_activePixelMask_) { cudaFree(reinterpret_cast<void*>(d_activePixelMask_)); d_activePixelMask_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_activePixelMask_), numPixels * sizeof(unsigned char)));
			activePixelMaskCapacity_ = numPixels;
		}
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_activePixelMask_), activePixelMaskHost_,
									numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice, stream_));
		d_activePixelMaskPtr = d_activePixelMask_;
	}

	// Denoiser guide-layer AOV buffers (--denoise, OR SVGF - see this
	// project's own SVGF plan for why SVGF reuses these same two buffers
	// instead of a third near-duplicate albedo/normal source: it needs
	// albedo for its own illumination-albedo demodulation, and this
	// existing normal AOV for its A-trous filter's edge-stopping weight,
	// rather than adding either as a new buffer) - persisted across
	// render() calls via ensureAovBuffers() (same resolution-keyed recreate-
	// on-change pattern as the recursive backend's own copy, optix_renderer_
	// render.cpp), zeroed fresh each render() call since they accumulate
	// (atomicAdd) across every sample - see evaluate_materials()'s own
	// accumulation comment, wavefront_kernels.cu.
	// Computed once and reused at every gate below that depends on the same
	// "does anything currently want this" question, rather than repeating
	// the compound condition verbatim at each site (a future edit to one
	// copy silently missing the others was a real risk here).
	const bool needsAovGuideBuffers = denoiseEnabled_ || svgfEnabled_;
	// neuralUpscaleEnabled_ is included here (not just gated by its own
	// temporalUpscaleJitterEnabled_ check in launchNeuralUpscaleUpdate())
	// because that call reads d_worldPos_/d_worldPosHistory_ and
	// prevRestirCamera_ unconditionally once enabled - without this, a user
	// running Neural Reconstruction with ReSTIR DI/GI and SVGF all off
	// (a reachable, independent GUI combination) would silently disable its
	// own temporal reprojection: d_worldPosHistory_ stays null forever, so
	// wf_restir_reproject_prev_pixel's null-guard always fails and the
	// network never sees real history.
	const bool needsWorldPosHistory = restirEnabled_ || restirGiEnabled_ || svgfEnabled_ || neuralUpscaleEnabled_;

	float3* d_albedoAovPtr = nullptr;
	float3* d_normalAovPtr = nullptr;
	if (needsAovGuideBuffers) {
		ensureAovBuffers((unsigned int)width, (unsigned int)height);
		d_albedoAovPtr = reinterpret_cast<float3*>(denoiserResources_.albedoAov);
		d_normalAovPtr = reinterpret_cast<float3*>(denoiserResources_.normalAov);
		// Cleared below, together with d_worldPos_ (see that block's own
		// comment) via svgf_checkerboard_clear_frame - NOT a plain memset
		// here, since a checkerboard-inactive pixel's albedo/normal must
		// survive this call untouched.
	} else if (denoiserResources_.albedoAov || denoiserResources_.normalAov) {
		// Neither consumer wants these this call - free rather than leave
		// them allocated-but-stale. launchEvaluateMaterials*()'s own kernel-
		// launch wrappers read denoiserResources_.albedoAov/.normalAov
		// directly as member state (this function's own comment above, and
		// wavefront_path_tracer.h's "AOV guide buffers travel via
		// denoiserResources_ member state" comment), NOT through the
		// d_albedoAovPtr/d_normalAovPtr locals just computed - so leaving a
		// non-null pointer here from an earlier, smaller-resolution
		// denoise/SVGF call would have every hit this call processes read
		// the STALE, undersized buffer at the OLD resolution once
		// h.pixelIndex exceeds it (found via compute-sanitizer after
		// LivePreviewSvgfTest's 64x64 renders left a stale 64x64 albedoAov/
		// normalAov allocated, then WavefrontRenderTest's own later 80x80
		// render - denoise and svgf both off - wrote past its end).
		destroyAovBuffers();
	}
	rc.numPixels = numPixels;
	rc.regularize = regularize;
	rc.d_fb = d_fb;
	rc.d_fbPtr = d_fbPtr;
	rc.d_weight = d_weight;
	rc.d_weightPtr = d_weightPtr;
	rc.d_activePixelMaskPtr = d_activePixelMaskPtr;
	rc.needsAovGuideBuffers = needsAovGuideBuffers;
	rc.needsWorldPosHistory = needsWorldPosHistory;
	rc.d_albedoAovPtr = d_albedoAovPtr;
	rc.d_normalAovPtr = d_normalAovPtr;
	return true;
}

void WavefrontPathTracer::renderPrepareRestirSvgf(RenderCall &rc) {
	int width = rc.width;
	int height = rc.height;
	const int numPixels = rc.numPixels;
	const bool needsWorldPosHistory = rc.needsWorldPosHistory;
	// Live Preview reprojection guide buffer (see setWorldPosOutputEnabled()'s
	// own comment) - same resolution-keyed allocate-once/only-realloc-on-
	// change lifecycle as d_fb_/d_weight_ above. Kept around after this call
	// returns (freed only on a resolution mismatch or cleanup()), so a later
	// readWorldPosBuffer() call can copy it back at its own pace - the same
	// "separate consumer of a persisted buffer" shape readAovBuffers() already
	// uses for the denoiser's own guide layers.
	// restirEnabled_ (DI), restirGiEnabled_ (GI), AND svgfEnabled_ all need
	// worldPos populated (temporal reuse's own disocclusion test -
	// GpuRestirTemporalContext's own comment; GI's and SVGF's own temporal
	// reuse both reuse this exact buffer for the same reason instead of
	// keeping a second, redundant "depth 0 hit position" copy - see
	// wavefront_path_tracer.h's own GI/SVGF buffer comments), regardless of
	// whether the CALLER separately asked for readback via
	// worldPosOutputEnabled_ (out_world_pos_buffer != nullptr) - these are
	// independent opt-ins for unrelated consumers of the same underlying
	// per-pixel buffer.
	if (worldPosOutputEnabled_ || needsWorldPosHistory) {
		if (worldPosCapacity_ != numPixels) {
			if (d_worldPos_) { cudaFree(reinterpret_cast<void*>(d_worldPos_)); d_worldPos_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_worldPos_), numPixels * sizeof(float4)));
			worldPosCapacity_ = numPixels;
		}
		// Cleared below, together with albedo/normal - see that call's own
		// comment for why this is no longer a plain memset.
	} else if (d_worldPos_) {
		// Flag turned off after being on - free rather than leave a stale
		// buffer readWorldPosBuffer() could otherwise still report success
		// for.
		cudaFree(reinterpret_cast<void*>(d_worldPos_));
		d_worldPos_ = 0;
		worldPosCapacity_ = 0;
	}

	// ReSTIR DI (Live Preview only) reservoir buffer - see setRestirEnabled()'s
	// own comment. Same resolution-keyed allocate-once/only-realloc-on-change
	// lifecycle as d_worldPos_ just above. Cleared every render() call (not
	// just on realloc) since this single buffer is fully rewritten by every
	// depth==0 non-specular hit this frame - see d_reservoirs_'s own header
	// comment on why a stale leftover entry (a pixel that didn't reach the
	// ReSTIR block this frame, e.g. a miss or a specular hit) is harmless
	// zeroed-out state, not a correctness hazard, but zeroing it anyway keeps
	// a resized/reused allocation from ever exposing a wholly unrelated old
	// frame's reservoir at a pixel this frame never touches.
	if (restirEnabled_) {
		if (reservoirsCapacity_ != numPixels) {
			if (d_reservoirs_) { cudaFree(reinterpret_cast<void*>(d_reservoirs_)); d_reservoirs_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_reservoirs_), numPixels * sizeof(GpuReservoir)));
			reservoirsCapacity_ = numPixels;
		}
		// Cleared below, together with the checkerboard AOV/worldPos clear -
		// see that call's own comment for why (needs checkerboardActive,
		// which isn't known-correct until after this call's own history-
		// invalidation checks have all run).
	} else if (d_reservoirs_) {
		cudaFree(reinterpret_cast<void*>(d_reservoirs_));
		d_reservoirs_ = 0;
		reservoirsCapacity_ = 0;
	}

	// ReSTIR temporal reuse's cross-call history buffers - see these members'
	// own header comment (wavefront_path_tracer.h) for the read-then-
	// overwrite-at-end-of-call lifecycle. Deliberately NOT memset every call
	// like d_reservoirs_/d_worldPos_ above (that would erase the very history
	// this call is about to read) - only allocated/resized here; populated by
	// the end-of-render() copy further down, and content only ever trusted
	// when restirHistoryValid_ is true.
	//
	// Checked by exact width/height, not just numPixels - see
	// restirHistoryWidth_/restirHistoryHeight_'s own header comment on why a
	// same-product different-dimensions resolution change must also
	// invalidate the history, even though it wouldn't trigger a capacity
	// mismatch below.
	if (restirHistoryWidth_ != width || restirHistoryHeight_ != height) {
		restirHistoryValid_ = false;
		restirGiHistoryValid_ = false;
		svgfHistoryValid_ = false;
	}
	if (needsWorldPosHistory) {
		if (worldPosHistoryCapacity_ != numPixels) {
			if (d_worldPosHistory_) { cudaFree(reinterpret_cast<void*>(d_worldPosHistory_)); d_worldPosHistory_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_worldPosHistory_), numPixels * sizeof(float4)));
			worldPosHistoryCapacity_ = numPixels;
			restirHistoryValid_ = false;
			restirGiHistoryValid_ = false;
			svgfHistoryValid_ = false;
		}
	} else {
		if (d_worldPosHistory_) { cudaFree(reinterpret_cast<void*>(d_worldPosHistory_)); d_worldPosHistory_ = 0; }
		worldPosHistoryCapacity_ = 0;
		restirHistoryValid_ = false;
		restirGiHistoryValid_ = false;
		svgfHistoryValid_ = false;
	}
	if (restirEnabled_) {
		if (reservoirsHistoryCapacity_ != numPixels) {
			if (d_reservoirsHistory_) { cudaFree(reinterpret_cast<void*>(d_reservoirsHistory_)); d_reservoirsHistory_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_reservoirsHistory_), numPixels * sizeof(GpuReservoir)));
			reservoirsHistoryCapacity_ = numPixels;
			restirHistoryValid_ = false;  // stale/undefined content at the new size
		}
		if (restirNormalCapacity_ != numPixels) {
			if (d_restirNormal_) { cudaFree(reinterpret_cast<void*>(d_restirNormal_)); d_restirNormal_ = 0; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_restirNormal_), numPixels * sizeof(float3)));
			restirNormalCapacity_ = numPixels;
		}
		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_restirNormal_), 0, numPixels * sizeof(float3), stream_));
	} else {
		if (d_reservoirsHistory_) { cudaFree(reinterpret_cast<void*>(d_reservoirsHistory_)); d_reservoirsHistory_ = 0; }
		reservoirsHistoryCapacity_ = 0;
		if (d_restirNormal_) { cudaFree(reinterpret_cast<void*>(d_restirNormal_)); d_restirNormal_ = 0; }
		restirNormalCapacity_ = 0;
		restirHistoryValid_ = false;
	}

	// ReSTIR for volumetric/participating media (Live Preview only) - see
	// wavefront_path_tracer.h's own d_volumeReservoirs_ header comment for
	// the full "why", and this project's own plan. Same restirEnabled_ gate
	// as DI's own buffers above (no separate UI toggle for this feature).
	// Uses reallocateDeviceBufferIfNeeded<T>()/freeDeviceBuffer() - same
	// helpers the GI/SVGF blocks below already use - instead of hand-rolled
	// free/malloc pairs.
	//
	// Unlike d_reservoirs_/d_restirNormal_ above, the 4 "current frame"
	// buffers here are deliberately NEVER memset every render() call - see
	// that same header comment for why an unconditional per-frame clear
	// would defeat this feature's whole point (holding a reservoir across a
	// medium's own "no scatter this frame" pass-through sub-case). Instead,
	// d_volumeMatIdx_ (0xFF fill - see setmemset semantics: an all-1-bits
	// int32 is -1 in two's complement) and d_volumeReservoirs_ (zero fill,
	// safe because GpuVolumeReservoir::valid()'s own weightSum>0.0f check
	// makes a zeroed reservoir read as invalid) are reset BOTH on a fresh
	// allocation of EITHER (captured explicitly below, not inferred from one
	// alone - the two are expected to always resize together since both are
	// keyed on the same numPixels, but a future change that decoupled them
	// should still get both reset rather than silently relying on that
	// coupling) AND whenever restirHistoryValid_ was ALREADY false when this
	// call started (capturing it before this block's own history-capacity
	// check further down can flip it) - i.e. a scene switch at unchanged
	// resolution (invalidateRestirHistory()) - since a stale GpuLightSample
	// carries the PREVIOUS scene's own lightIdx/primIdx, which would
	// otherwise get dereferenced against the new scene's freshly-uploaded
	// (possibly smaller) light/geometry arrays. The other 2 sticky buffers
	// (phaseWoG/entryPoint) need no such reset: every reader gates on
	// matIdx>=0 first, so resetting matIdx alone makes their own stale bytes
	// unreachable.
	if (restirEnabled_) {
		const bool volumeReservoirsResized = reallocateDeviceBufferIfNeeded<GpuVolumeReservoir>(d_volumeReservoirs_, volumeReservoirsCapacity_, numPixels);
		const bool volumeMatIdxResized = reallocateDeviceBufferIfNeeded<int>(d_volumeMatIdx_, volumeMatIdxCapacity_, numPixels);
		const bool volumeHistoryWasInvalid = !restirHistoryValid_;
		if (volumeReservoirsResized || volumeMatIdxResized || volumeHistoryWasInvalid) {
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_volumeMatIdx_), 0xFF, numPixels * sizeof(int), stream_));
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_volumeReservoirs_), 0, numPixels * sizeof(GpuVolumeReservoir), stream_));
		}
		reallocateDeviceBufferIfNeeded<float4>(d_volumePhaseWoG_, volumePhaseWoGCapacity_, numPixels);
		reallocateDeviceBufferIfNeeded<float4>(d_volumeEntryPoint_, volumeEntryPointCapacity_, numPixels);
		if (reallocateDeviceBufferIfNeeded<GpuVolumeReservoir>(d_volumeReservoirsHistory_, volumeReservoirsHistoryCapacity_, numPixels)) {
			restirHistoryValid_ = false;  // stale/undefined content at the new size
		}
	} else {
		freeDeviceBuffer(d_volumeReservoirs_, volumeReservoirsCapacity_);
		freeDeviceBuffer(d_volumeMatIdx_, volumeMatIdxCapacity_);
		freeDeviceBuffer(d_volumePhaseWoG_, volumePhaseWoGCapacity_);
		freeDeviceBuffer(d_volumeEntryPoint_, volumeEntryPointCapacity_);
		freeDeviceBuffer(d_volumeReservoirsHistory_, volumeReservoirsHistoryCapacity_);
	}

	// ReSTIR GI (Live Preview only) - own buffers, same resolution-keyed
	// allocate-once/only-realloc-on-change lifecycle as DI's own above (see
	// wavefront_path_tracer.h's own GI buffer comments for why these are
	// separate from d_reservoirs_/d_reservoirsHistory_: a different payload,
	// not a different mechanism). d_giOriginContext_/d_giCandidateOut_ are
	// pure per-frame scratch (fully rewritten by depth==0/depth==1's own
	// processing this frame - a stale leftover entry is inert, gated on its
	// own pdfAtX0/never-read-unless-valid, same reasoning as d_reservoirs_'s
	// own "harmless zeroed-out state" comment) but still memset every call so
	// a resized/reused allocation never exposes a wholly unrelated old
	// frame's data at a pixel this frame never touches.
	// Allocation only here - see reallocateDeviceBufferIfNeeded()'s own
	// comment for why this replaces 4 near-identical hand-written blocks.
	// d_giOriginContext_/d_giCandidateOut_ are NOT memset here (unlike DI's
	// analogous buffers) - they need to be cleared once per SAMPLE, not once
	// per render() call (a resolution-triggered realloc still leaves stale
	// garbage otherwise, so the fresh allocation IS memset once, immediately
	// below, to cover that one case) - see the per-sample memset inside the
	// sampleIdx loop for why, and restir_gi_finalize's own header comment
	// (wavefront_kernels_restir.cu) for the staleness bug this fixes.
	// d_giReservoirs_ is never memset at all: restir_gi_finalize
	// unconditionally writes every pixel's outputReservoirs[idx] (either a
	// default-constructed GpuGiReservoir{} or the finalized one) before
	// anything ever reads it, so a memset here would be immediately
	// overwritten - pure wasted bandwidth.
	if (restirGiEnabled_) {
		const bool originContextResized = reallocateDeviceBufferIfNeeded<GpuGiOriginContext>(d_giOriginContext_, giOriginContextCapacity_, numPixels);
		if (originContextResized) {
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_giOriginContext_), 0, numPixels * sizeof(GpuGiOriginContext), stream_));
		}
		const bool candidateOutResized = reallocateDeviceBufferIfNeeded<GpuGiSample>(d_giCandidateOut_, giCandidateOutCapacity_, numPixels);
		if (candidateOutResized) {
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_giCandidateOut_), 0, numPixels * sizeof(GpuGiSample), stream_));
		}
		reallocateDeviceBufferIfNeeded<GpuGiReservoir>(d_giReservoirs_, giReservoirsCapacity_, numPixels);
		if (reallocateDeviceBufferIfNeeded<GpuGiReservoir>(d_giReservoirsHistory_, giReservoirsHistoryCapacity_, numPixels)) {
			restirGiHistoryValid_ = false;  // stale/undefined content at the new size
		}
	} else {
		freeDeviceBuffer(d_giOriginContext_, giOriginContextCapacity_);
		freeDeviceBuffer(d_giCandidateOut_, giCandidateOutCapacity_);
		freeDeviceBuffer(d_giReservoirs_, giReservoirsCapacity_);
		freeDeviceBuffer(d_giReservoirsHistory_, giReservoirsHistoryCapacity_);
		restirGiHistoryValid_ = false;
	}

	// SVGF (Live Preview only) - own buffers, same resolution-keyed
	// allocate-once/only-realloc-on-change lifecycle as DI/GI's own above.
	// d_svgfCurrent_ is NOT memset here - svgf_temporal_integrate
	// unconditionally writes every pixel's own GpuSvgfState each call (same
	// "fully overwritten before ever read" reasoning that already justified
	// removing GI's own now-dead d_giReservoirs_ memset), so a memset here
	// would just be immediately-overwritten wasted bandwidth. The ping-pong
	// scratch buffers are pure within-this-call transients, never read
	// across frames, so they need no memset or history-validity tracking at
	// all - each render() call's own A-trous sequence starts by WRITING
	// pingpong[0] (svgf_prepare_for_filter), never reading it first.
	if (svgfEnabled_) {
		reallocateDeviceBufferIfNeeded<GpuSvgfState>(d_svgfCurrent_, svgfCurrentCapacity_, numPixels);
		if (reallocateDeviceBufferIfNeeded<GpuSvgfState>(d_svgfHistory_, svgfHistoryCapacity_, numPixels)) {
			svgfHistoryValid_ = false;  // stale/undefined content at the new size
		}
		for (int i = 0; i < 2; ++i) {
			reallocateDeviceBufferIfNeeded<float4>(d_svgfPingPong_[i], svgfPingPongCapacity_[i], numPixels);
		}
	} else {
		freeDeviceBuffer(d_svgfCurrent_, svgfCurrentCapacity_);
		freeDeviceBuffer(d_svgfHistory_, svgfHistoryCapacity_);
		for (int i = 0; i < 2; ++i) {
			freeDeviceBuffer(d_svgfPingPong_[i], svgfPingPongCapacity_[i]);
		}
		svgfHistoryValid_ = false;
	}

	// Checkerboard temporal upsampling: reuses SVGF's own temporal
	// reprojection/history as a reconstruction filter for a sparse per-frame
	// sample pattern (alternating which half of pixels get a fresh primary-
	// ray sample each frame - see wf_checkerboard_pixel_active(),
	// wavefront_device_helpers.h), so a heavy scene traces roughly half as
	// many primary rays per Live Preview frame. Gated on BOTH svgfEnabled_
	// AND restirGiHistoryValid_ (not just SVGF's own history) - ReSTIR GI is
	// unconditionally on for every Live Preview call with no way to disable
	// it, and its own restir_gi_finalize() kernel needs a warm, reprojectable
	// history to correctly HOLD a checkerboard-inactive pixel's reservoir
	// (see that kernel's own comment) rather than wiping it to empty every
	// other frame. Both *HistoryValid_ flags already reset themselves on
	// exactly the events (scene switch, resolution change) that would
	// otherwise need special-casing here - piggybacking on that existing
	// invalidation is what keeps the very first frame after such an event
	// (where checkerboarding would otherwise bake a fabricated black/empty
	// half-image into fresh history) safe by construction.
	//
	// Computed HERE, after every *HistoryValid_ reset above (resolution-
	// change checks for DI/GI/SVGF's own history/AOV buffers), not earlier -
	// reading these flags before their own reset-on-resize logic had a
	// chance to run for THIS call would latch a stale `true` on the exact
	// frame a resize invalidates history, skipping the checkerboard-clear
	// kernel's zeroing of freshly-(re)allocated, uninitialized buffer
	// entries for "inactive" pixels (a real, confirmed bug in an earlier
	// version of this code - see the code review that caught it).
	//
	// !temporalUpscaleJitterEnabled_ - Live Preview's own UI should never
	// enable both features together (temporal upscale requires SVGF's
	// showLatest path off), but nothing on the GPU side enforced that until
	// this check: a checkerboard-inactive pixel holds over LAST frame's
	// (stale, out-of-jitter-phase) world position/color rather than
	// sampling fresh, which the temporal-upscale splat step would then
	// happily write into a high-res history cell as if it were this
	// frame's genuine sample - corrupting it in a way the low-res
	// running-mean path (which just tolerates an occasional held-over
	// pixel) does not. See this project's own plan for the full rationale.
	const bool checkerboardActive = svgfEnabled_ && svgfHistoryValid_ && restirGiHistoryValid_ && !temporalUpscaleJitterEnabled_;
	rc.checkerboardActive = checkerboardActive;
}

void WavefrontPathTracer::renderPrepareLaunch(RenderCall &rc) {
	int width = rc.width;
	int height = rc.height;
	int samples_per_pixel = rc.samples_per_pixel;
	int max_depth = rc.max_depth;
	const auto &camera = rc.camera;
	OptixTraversableHandle gas_handle = rc.gas_handle;
	CUdeviceptr d_materials = rc.d_materials;
	CUdeviceptr d_spheres = rc.d_spheres;
	CUdeviceptr d_quads = rc.d_quads;
	CUdeviceptr d_light_indices = rc.d_light_indices;
	CUdeviceptr d_lightKinds = rc.d_lightKinds;
	CUdeviceptr d_alias_table = rc.d_alias_table;
	unsigned int num_materials = rc.num_materials;
	unsigned int num_spheres = rc.num_spheres;
	unsigned int num_quads = rc.num_quads;
	unsigned int num_lights = rc.num_lights;
	CUdeviceptr d_bilinear_patches = rc.d_bilinear_patches;
	unsigned int num_bilinear_patches = rc.num_bilinear_patches;
	CUdeviceptr d_triangles = rc.d_triangles;
	unsigned int num_triangles = rc.num_triangles;
	CUdeviceptr d_disks = rc.d_disks;
	unsigned int num_disks = rc.num_disks;
	CUdeviceptr d_cylinders = rc.d_cylinders;
	unsigned int num_cylinders = rc.num_cylinders;
	const int numPixels = rc.numPixels;
	float3* d_fbPtr = rc.d_fbPtr;
	const bool needsAovGuideBuffers = rc.needsAovGuideBuffers;
	const bool needsWorldPosHistory = rc.needsWorldPosHistory;
	float3* d_albedoAovPtr = rc.d_albedoAovPtr;
	float3* d_normalAovPtr = rc.d_normalAovPtr;
	const bool checkerboardActive = rc.checkerboardActive;
	// Checkerboard temporal upsampling's frame-clear (see this function's own
	// checkerboardActive comment above, and svgf_checkerboard_clear_frame's
	// own comment, wavefront_kernels_svgf.cu, for the full rationale). Only
	// actually NEEDS the generic per-pixel kernel when checkerboardActive is
	// true (some pixels must be skipped); when it's false - denoise-only,
	// ReSTIR-only-without-SVGF, or even an SVGF+GI call before both
	// histories warm up, all of which are common - every pixel clears
	// unconditionally, so the plain, driver-specialized cudaMemsetAsync this
	// replaced is strictly faster for the exact same result and is worth
	// keeping as the fast path rather than paying a kernel launch + per-pixel
	// branch to reproduce it. Each of the 3 pointers is independently
	// nullable either way - only the ones actually allocated above (per
	// needsAovGuideBuffers/needsWorldPosHistory/worldPosOutputEnabled_) are
	// non-null here. frameNumber_ + 1: this runs BEFORE frameNumber_++ below
	// (the sample loop's own generate_camera_rays calls, further down, see
	// each other's launch site), so +1 is what makes this call's own
	// checkerboard parity agree with what THIS SAME render() call's own
	// primary rays will use once the increment has actually happened.
	if (checkerboardActive) {
		wf_launch_svgf_checkerboard_clear_frame(
			d_albedoAovPtr, d_normalAovPtr, reinterpret_cast<float4*>(d_worldPos_),
			width, height, frameNumber_ + 1, checkerboardActive, stream_);
	} else {
		if (d_albedoAovPtr) CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_albedoAovPtr), 0, numPixels * sizeof(float3), stream_));
		if (d_normalAovPtr) CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_normalAovPtr), 0, numPixels * sizeof(float3), stream_));
		if (d_worldPos_) CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_worldPos_), 0, numPixels * sizeof(float4), stream_));
	}

	// ReSTIR DI's own reservoir clear (see restir_clear_reservoirs' own
	// comment, wavefront_kernels_restir.cu, for why this must be
	// checkerboard-aware too - DI is unconditionally on for every Live
	// Preview call, same as GI, and was found to have the identical
	// reservoir-wipe bug GI's own hold branch was added to fix). A real
	// per-struct clear (not a raw cudaMemsetAsync zero-fill) for every
	// pixel this call decides to clear - see that kernel's own comment for
	// why a memset would leave GpuLightSample::lightIdx at 0 instead of its
	// documented -1 "invalid" sentinel.
	if (restirEnabled_) {
		wf_launch_restir_clear_reservoirs(
			reinterpret_cast<GpuReservoir*>(d_reservoirs_),
			width, height, frameNumber_ + 1, checkerboardActive, stream_);
	}

	// Build WavefrontLaunchParams template (queue pointers filled per phase)
	WavefrontLaunchParams lp = {};
	lp.framebuffer   = d_fbPtr;
	lp.width         = (unsigned int)width;
	lp.height        = (unsigned int)height;
	lp.bumpFootprint = gpu_make_bump_footprint(camera, width, height);
	lp.cameraMediumSigmaT       = camera.cameraMediumSigmaT;
	lp.cameraMediumMaterialIdx  = camera.cameraMediumMaterialIdx;
	lp.traversable   = gas_handle;
	lp.spheres       = reinterpret_cast<SphereData*>(d_spheres);
	lp.numSpheres    = num_spheres;
	lp.quads         = reinterpret_cast<QuadData*>(d_quads);
	lp.numQuads      = num_quads;
	lp.bilinearPatches = reinterpret_cast<BilinearPatchData*>(d_bilinear_patches);
	lp.numBilinearPatches = num_bilinear_patches;
	lp.triangles     = reinterpret_cast<TriangleData*>(d_triangles);
	lp.numTriangles  = num_triangles;
	lp.disks         = reinterpret_cast<DiskData*>(d_disks);
	lp.numDisks      = num_disks;
	lp.cylinders     = reinterpret_cast<CylinderData*>(d_cylinders);
	lp.numCylinders  = num_cylinders;
	lp.materials     = reinterpret_cast<MaterialData*>(d_materials);
	lp.numMaterials  = num_materials;
	lp.textures       = reinterpret_cast<TextureData*>(d_textures_);
	lp.texturePixels  = reinterpret_cast<unsigned char*>(d_texturePixels_);
	lp.cloudMediums    = reinterpret_cast<CloudMedium<float>*>(d_cloudMediums_);
	lp.numCloudMediums = numCloudMediums_;
	lp.rgbGridMediums    = reinterpret_cast<GpuRgbGridMedium*>(d_rgbGridMediums_);
	lp.numRgbGridMediums = numRgbGridMediums_;
	lp.rgbGridData        = reinterpret_cast<float*>(d_rgbGridData_);
	lp.rgbGridDataCount   = rgbGridDataCount_;
	// The shadow any-hit (wavefront_anyhit_shadow.h) ratio-tracks uniform-grid media from these. Left unset, its
	// bounds check (idx >= numGridMediums) always failed, so NEE shadow rays inside a "uniformgrid" medium were
	// never attenuated and the medium rendered far too bright, growing with every extra bounce.
	lp.gridMediums        = reinterpret_cast<GpuGridMedium*>(d_gridMediums_);
	lp.numGridMediums     = numGridMediums_;
	lp.gridData           = reinterpret_cast<float*>(d_gridData_);
	lp.gridDataCount      = gridDataCount_;
	lp.lightIndices  = reinterpret_cast<int*>(d_light_indices);
	lp.lightKinds = reinterpret_cast<const GpuLightKind*>(d_lightKinds);
	lp.instancePrimBase = reinterpret_cast<const int*>(d_instancePrimBase_);
	lp.aliasTable    = reinterpret_cast<GpuAliasEntry*>(d_alias_table);
	lp.numLights     = num_lights;
	// BSSRDF probe walk (MaterialType::Subsurface, Phase 2) - same device
	// buffers OptiXRenderer already builds/uploads once for the recursive
	// backend, wired in via setBssrdfTables(). Null/0 (the default) is a
	// valid "no Subsurface materials in this scene" state - wf_params.
	// numBssrdfTables stays 0 and evaluate_materials's Subsurface case is
	// simply never reached.
	lp.bssrdfTables         = reinterpret_cast<GpuBssrdfTable*>(d_bssrdfTables_);
	lp.numBssrdfTables      = numBssrdfTables_;
	lp.bssrdfRhoSamples     = reinterpret_cast<float*>(d_bssrdfRhoSamples_);
	lp.bssrdfRadiusSamples  = reinterpret_cast<float*>(d_bssrdfRadiusSamples_);
	lp.bssrdfProfile        = reinterpret_cast<float*>(d_bssrdfProfile_);
	lp.bssrdfProfileCdf     = reinterpret_cast<float*>(d_bssrdfProfileCdf_);
	lp.samplesPerPixel = (unsigned int)samples_per_pixel;
	lp.maxDepth        = (unsigned int)max_depth;
	lp.frameNumber     = frameNumber_++;

	// Progress print interval: a flat "every 10th sample" (the original
	// throttle) still produced 500 lines on a 5000-spp render - plenty to
	// flood a log panel. Scales the interval with samples_per_pixel instead,
	// capping the whole render at ~50 progress lines, but never coarser than
	// every-10-samples so a short/low-spp render (which finishes in a couple
	// seconds anyway) keeps the same fine-grained updates as before.
	const int progressPrintInterval = std::max(10, samples_per_pixel / 50);
	rc.d_fbPtr = d_fbPtr;
	rc.d_albedoAovPtr = d_albedoAovPtr;
	rc.d_normalAovPtr = d_normalAovPtr;
	rc.lp = lp;
	rc.progressPrintInterval = progressPrintInterval;
}

void WavefrontPathTracer::renderSampleLoop(RenderCall &rc) {
	int width = rc.width;
	int height = rc.height;
	int samples_per_pixel = rc.samples_per_pixel;
	int max_depth = rc.max_depth;
	const auto &camera = rc.camera;
	float*  framebuffer = rc.framebuffer;
	CUdeviceptr d_materials = rc.d_materials;
	CUdeviceptr d_spheres = rc.d_spheres;
	CUdeviceptr d_quads = rc.d_quads;
	CUdeviceptr d_light_indices = rc.d_light_indices;
	CUdeviceptr d_lightKinds = rc.d_lightKinds;
	CUdeviceptr d_alias_table = rc.d_alias_table;
	unsigned int num_materials = rc.num_materials;
	unsigned int num_spheres = rc.num_spheres;
	unsigned int num_quads = rc.num_quads;
	unsigned int num_lights = rc.num_lights;
	CUdeviceptr d_punctual_lights = rc.d_punctual_lights;
	unsigned int num_punctual_lights = rc.num_punctual_lights;
	CUdeviceptr d_bilinear_patches = rc.d_bilinear_patches;
	unsigned int num_bilinear_patches = rc.num_bilinear_patches;
	CUdeviceptr d_triangles = rc.d_triangles;
	unsigned int num_triangles = rc.num_triangles;
	CUdeviceptr d_disks = rc.d_disks;
	unsigned int num_disks = rc.num_disks;
	CUdeviceptr d_cylinders = rc.d_cylinders;
	unsigned int num_cylinders = rc.num_cylinders;
	const int numPixels = rc.numPixels;
	const bool regularize = rc.regularize;
	float3* d_fbPtr = rc.d_fbPtr;
	float* d_weightPtr = rc.d_weightPtr;
	CUdeviceptr d_activePixelMaskPtr = rc.d_activePixelMaskPtr;
	const bool checkerboardActive = rc.checkerboardActive;
	WavefrontLaunchParams lp = rc.lp;
	const int progressPrintInterval = rc.progressPrintInterval;
	WavefrontRenderStats stats = rc.stats;
	// -------------------------------------------------------------------------
	// Outer sample loop
	// -------------------------------------------------------------------------
	for (int sampleIdx = 0; sampleIdx < samples_per_pixel; ++sampleIdx) {

		// ReSTIR GI's per-pixel scratch (see wavefront_kernels_restir.cu's
		// own restir_gi_finalize header comment) is genuinely this SAMPLE's
		// own data, not this whole render() call's - cleared once per
		// sample, here, not once per call (the top-of-render() allocation
		// block above only memsets a FRESH allocation, once). Without this,
		// a pixel whose depth==0 hit is Lambertian on sample K but a
		// different, non-Lambertian material on sample K+1 would leave
		// sample K's stale x0 context sitting in giOriginContext[pixelIndex]
		// for sample K+1 to incorrectly pair with ITS OWN, unrelated x1 -
		// a real bug this project's own code review caught (samples_per_pixel
		// is always 1 for every current caller of ReSTIR GI, so this was
		// dormant, not yet reachable - fixed here regardless, since nothing
		// prevents a future caller from raising it).
		if (restirGiEnabled_) {
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_giOriginContext_), 0, numPixels * sizeof(GpuGiOriginContext), stream_));
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_giCandidateOut_), 0, numPixels * sizeof(GpuGiSample), stream_));
		}

		// Reset ray queue counter, generate primary rays
		resetQueueCounter(reinterpret_cast<int*>(d_rayCounter_));
		launchGenerateCameraRays(width, height, sampleIdx, camera, d_weightPtr, checkerboardActive,
			reinterpret_cast<const unsigned char*>(d_activePixelMaskPtr));
		CUDA_CHECK(cudaStreamSynchronize(stream_));

		// -------------------------------------------------------------------------
		// Inner bounce loop
		// -------------------------------------------------------------------------
		// max_depth bounces, one more iteration for the emission-only trace of the last continuation ray, plus room for
		// free medium-boundary crossings (MaterialType::Interface and the medium pass-throughs): those take an iteration
		// here but not a bounce, and each ray's own depth (checked in the intersect and material kernels) is what enforces
		// the real budget. The queue empties as rays finish, so the extra iterations cost nothing in a scene without media.
		for (int depth = 0; depth < max_depth + 1 + kMaxMediumBoundaryCrossings; ++depth) {

			int numRays = readQueueSize(reinterpret_cast<int*>(d_rayCounter_));
			if (numRays == 0) break;
			stats.primaryRays += numRays;
			++stats.bounceIterations;

			// Reset output queue counters
			resetQueueCounter(reinterpret_cast<int*>(d_hitCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_simpleHitCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_dielectricHitCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_missCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_shadowCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_nextRayCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_probeCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_exitCounter_));

			// ------------------------------------------------------------------
			// Phase 2: OptiX intersect launch
			// ------------------------------------------------------------------
			lp.rayQueue.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
			lp.rayQueue.counter  = reinterpret_cast<int*>(d_rayCounter_);
			lp.rayQueue.capacity = queueCapacity_;

			lp.hitQueue.items    = reinterpret_cast<HitWorkItem*>(d_hitItems_);
			lp.hitQueue.counter  = reinterpret_cast<int*>(d_hitCounter_);
			lp.hitQueue.capacity = queueCapacity_;

			lp.simpleHitQueue.items    = reinterpret_cast<HitWorkItem*>(d_simpleHitItems_);
			lp.simpleHitQueue.counter  = reinterpret_cast<int*>(d_simpleHitCounter_);
			lp.simpleHitQueue.capacity = queueCapacity_;

			lp.dielectricHitQueue.items    = reinterpret_cast<HitWorkItem*>(d_dielectricHitItems_);
			lp.dielectricHitQueue.counter  = reinterpret_cast<int*>(d_dielectricHitCounter_);
			lp.dielectricHitQueue.capacity = queueCapacity_;

			lp.missQueue.items    = reinterpret_cast<MissWorkItem*>(d_missItems_);
			lp.missQueue.counter  = reinterpret_cast<int*>(d_missCounter_);
			lp.missQueue.capacity = queueCapacity_;

			CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &lp,
									   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));

			OPTIX_CHECK(optixLaunch(
				intersectPipeline_, stream_,
				d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
				&intersectSBT_,
				(unsigned int)numRays, 1, 1));

			CUDA_CHECK(cudaStreamSynchronize(stream_));

			int numHits            = readQueueSize(reinterpret_cast<int*>(d_hitCounter_));
			int numSimpleHits      = readQueueSize(reinterpret_cast<int*>(d_simpleHitCounter_));
			int numDielectricHits  = readQueueSize(reinterpret_cast<int*>(d_dielectricHitCounter_));
			int numMiss            = readQueueSize(reinterpret_cast<int*>(d_missCounter_));
			stats.hits += numHits;
			stats.simpleHits += numSimpleHits;
			stats.dielectricHits += numDielectricHits;
			stats.misses += numMiss;

			// ------------------------------------------------------------------
			// Phase 3: Evaluate materials (fills shadowQueue + nextRayQueue)
			// ------------------------------------------------------------------
			launchEvaluateMaterials(
				numHits, max_depth, regularize, camera.maxComponentValue,
				reinterpret_cast<const SphereData*>(d_spheres), num_spheres,
				reinterpret_cast<const QuadData*>(d_quads),     num_quads,
				reinterpret_cast<const TriangleData*>(d_triangles), num_triangles,
				reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches), num_bilinear_patches,
				reinterpret_cast<const DiskData*>(d_disks), num_disks,
				reinterpret_cast<const CylinderData*>(d_cylinders), num_cylinders,
				reinterpret_cast<const MaterialData*>(d_materials), num_materials,
				reinterpret_cast<const int*>(d_light_indices),
				reinterpret_cast<const GpuLightKind*>(d_lightKinds),
				reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
				num_lights,
				reinterpret_cast<const PunctualLightGPU*>(d_punctual_lights),
				num_punctual_lights,
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist, camera.portalLight);

			// ------------------------------------------------------------------
			// Phase 3b: Evaluate simple materials (Lambertian/Metal hits
			// routed to simpleHitQueue at push time - see wavefront_types.h's
			// WavefrontQueues::simpleHitQueue comment). Independent of the
			// launch above - reads its own queue, writes into the same
			// shadowQueue/nextRayQueue/framebuffer via the same atomicAdd-
			// based WorkQueue::push() (safe for concurrent writers) and
			// disjoint per-pixel framebuffer slots (a ray belongs to exactly
			// one queue per bounce). Launched on simpleMaterialStream_ (its
			// own stream, separate from stream_ - see that member's header
			// comment) so this genuinely overlaps launchEvaluateMaterials
			// above on the GPU instead of serializing two kernels with no
			// real dependency on each other - both streams are synced below,
			// before numMiss/numShadow are read.
			// ------------------------------------------------------------------
			launchEvaluateMaterialsSimple(
				numSimpleHits, max_depth,
				reinterpret_cast<const SphereData*>(d_spheres), num_spheres,
				reinterpret_cast<const QuadData*>(d_quads),     num_quads,
				reinterpret_cast<const TriangleData*>(d_triangles), num_triangles,
				reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches), num_bilinear_patches,
				reinterpret_cast<const DiskData*>(d_disks), num_disks,
				reinterpret_cast<const CylinderData*>(d_cylinders), num_cylinders,
				reinterpret_cast<const MaterialData*>(d_materials), num_materials,
				reinterpret_cast<const int*>(d_light_indices),
				reinterpret_cast<const GpuLightKind*>(d_lightKinds),
				reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
				num_lights,
				reinterpret_cast<const PunctualLightGPU*>(d_punctual_lights),
				num_punctual_lights,
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist, camera.portalLight, camera.maxComponentValue);

			// ------------------------------------------------------------------
			// Phase 3c: Evaluate dielectric materials (Dielectric/RoughDielectric
			// hits routed to dielectricHitQueue at push time - see
			// wavefront_types.h's WavefrontQueues::dielectricHitQueue comment).
			// Same independent-queue/own-stream overlap reasoning as Phase 3b
			// above - launched on dielectricMaterialStream_, synced below
			// alongside simpleMaterialStream_ before numMiss/numShadow are read.
			// ------------------------------------------------------------------
			launchEvaluateMaterialsDielectric(
				numDielectricHits, max_depth, regularize, camera.maxComponentValue,
				reinterpret_cast<const SphereData*>(d_spheres), num_spheres,
				reinterpret_cast<const QuadData*>(d_quads),     num_quads,
				reinterpret_cast<const TriangleData*>(d_triangles), num_triangles,
				reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches), num_bilinear_patches,
				reinterpret_cast<const DiskData*>(d_disks), num_disks,
				reinterpret_cast<const CylinderData*>(d_cylinders), num_cylinders,
				reinterpret_cast<const MaterialData*>(d_materials), num_materials,
				reinterpret_cast<const int*>(d_light_indices),
				reinterpret_cast<const GpuLightKind*>(d_lightKinds),
				reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
				num_lights,
				reinterpret_cast<const PunctualLightGPU*>(d_punctual_lights),
				num_punctual_lights,
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist, camera.portalLight);

			// ------------------------------------------------------------------
			// Phase 4: Accumulate miss (escaped rays → background)
			// ------------------------------------------------------------------
			launchAccumulateMiss(numMiss, d_fbPtr, camera.backgroundColor, camera.skyDist, camera.portalLight, camera.maxComponentValue);

			CUDA_CHECK(cudaStreamSynchronize(stream_));
			CUDA_CHECK(cudaStreamSynchronize(simpleMaterialStream_));
			CUDA_CHECK(cudaStreamSynchronize(dielectricMaterialStream_));

			// ------------------------------------------------------------------
			// Phase 4b: BSSRDF probe walk + exit resolve (MaterialType::
			// Subsurface, Phase 2) - mirrors pbrt-v4's own SampleSubsurface
			// dedicated stage, called once per bounce depth right after
			// evaluate_materials (which queued any Subsurface-transmission
			// hits this bounce into bssrdfProbeQueue instead of scattering
			// them directly - see wavefront_types.h's BssrdfProbeWorkItem
			// comment). Skipped entirely (both the extra optixLaunch and the
			// CUDA kernel) when nothing was queued - the overwhelming
			// majority of bounces on the overwhelming majority of scenes,
			// since Subsurface materials are rare. Must run BEFORE the
			// numShadow read below: resolve_bssrdf_exit() (wavefront_kernels.cu)
			// appends its own NEE shadow rays into the SAME shadowQueue
			// evaluate_materials already wrote to, so they need to be
			// counted before the shadow pass launches.
			// ------------------------------------------------------------------
			int numProbe = readQueueSize(reinterpret_cast<int*>(d_probeCounter_));
			stats.probeRays += numProbe;
			if (numProbe > 0) {
				lp.bssrdfProbeQueue.items    = reinterpret_cast<BssrdfProbeWorkItem*>(d_probeItems_);
				lp.bssrdfProbeQueue.counter  = reinterpret_cast<int*>(d_probeCounter_);
				lp.bssrdfProbeQueue.capacity = queueCapacity_;

				lp.bssrdfExitQueue.items    = reinterpret_cast<BssrdfExitWorkItem*>(d_exitItems_);
				lp.bssrdfExitQueue.counter  = reinterpret_cast<int*>(d_exitCounter_);
				lp.bssrdfExitQueue.capacity = queueCapacity_;

				CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &lp,
										   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));

				OPTIX_CHECK(optixLaunch(
					intersectPipeline_, stream_,
					d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
					&probeSBT_,
					(unsigned int)numProbe, 1, 1));

				CUDA_CHECK(cudaStreamSynchronize(stream_));

				int numExit = readQueueSize(reinterpret_cast<int*>(d_exitCounter_));
				stats.probeExits += numExit;
				launchResolveBssrdfExit(
					numExit,
					reinterpret_cast<const MaterialData*>(d_materials), num_materials,
					reinterpret_cast<const SphereData*>(d_spheres), num_spheres,
					reinterpret_cast<const QuadData*>(d_quads), num_quads,
					reinterpret_cast<const TriangleData*>(d_triangles), num_triangles,
					reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches), num_bilinear_patches,
					reinterpret_cast<const DiskData*>(d_disks), num_disks,
					reinterpret_cast<const CylinderData*>(d_cylinders), num_cylinders,
					reinterpret_cast<const int*>(d_light_indices),
					reinterpret_cast<const GpuLightKind*>(d_lightKinds),
					reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
					num_lights,
					reinterpret_cast<const PunctualLightGPU*>(d_punctual_lights),
					num_punctual_lights,
					d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist, camera.portalLight, camera.maxComponentValue);

				CUDA_CHECK(cudaStreamSynchronize(stream_));
			}

			int numShadow = readQueueSize(reinterpret_cast<int*>(d_shadowCounter_));
			stats.shadowRays += numShadow;

			// ------------------------------------------------------------------
			// Phase 5: OptiX shadow launch (determine occlusion)
			// ------------------------------------------------------------------
			if (numShadow > 0) {
				// Store shadow queue pointers in lp for the shadow raygen.
				lp.shadowQueue.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
				lp.shadowQueue.counter  = reinterpret_cast<int*>(d_shadowCounter_);
				lp.shadowQueue.capacity = shadowQueueCapacity_;

				// Temporarily point framebuffer to the transmittance float
				// array so __raygen__wf_shadow can write results there.
				// (The kernel casts (float*)wf_params.framebuffer.)
				WavefrontLaunchParams shadowLP = lp;
				shadowLP.framebuffer = reinterpret_cast<float3*>(d_transmittance_);

				CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &shadowLP,
										   sizeof(WavefrontLaunchParams),
										   cudaMemcpyHostToDevice, stream_));

				OPTIX_CHECK(optixLaunch(
					shadowPipeline_, stream_,
					d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
					&shadowSBT_,
					(unsigned int)numShadow, 1, 1));

				CUDA_CHECK(cudaStreamSynchronize(stream_));

				// ------------------------------------------------------------------
				// Phase 6: Accumulate shadow contributions
				// ------------------------------------------------------------------
				launchAccumulateShadow(numShadow,
									   reinterpret_cast<const float*>(d_transmittance_),
									   d_fbPtr, camera.maxComponentValue);
				CUDA_CHECK(cudaStreamSynchronize(stream_));
			}

			// ReSTIR GI finalize (see wavefront_kernels_restir.cu's own
			// restir_gi_finalize header comment) - runs once per SAMPLE,
			// right after depth==1's own NEE has fully resolved (whether or
			// not any shadow ray was actually queued this frame: a fresh
			// candidate with zero captured radiance is still a valid,
			// zero-weight RIS candidate, and temporal reuse alone may still
			// have a good reservoir to fall back on - so this is NOT nested
			// inside the `if (numShadow > 0)` block above). A no-op
			// (launchGiFinalize's own !restirGiEnabled_ early-out) for every
			// other depth and for batch/offline rendering.
			// No host sync after this launch: everything remaining in this
			// loop iteration (Phase 7's pointer swaps, the next depth
			// iteration's own launches) either runs entirely on the host or
			// is queued on this same stream_, so stream-order alone already
			// sequences it correctly after restir_gi_finalize - a
			// cudaStreamSynchronize here would only block the CPU thread for
			// no correctness benefit, once per sample, every interactive
			// frame GI is enabled for.
			if (depth == 1) {
				launchGiFinalize(reinterpret_cast<const MaterialData*>(d_materials), d_fbPtr, camera.maxComponentValue, d_weightPtr);
			}

			// ------------------------------------------------------------------
			// Phase 7: Swap ray queues for next bounce
			// ------------------------------------------------------------------
			std::swap(d_rayItems_,   d_nextRayItems_);
			std::swap(d_rayCounter_, d_nextRayCounter_);
		}

		// Progress reporting: unlike the recursive backend (one monolithic
		// optixLaunch for the whole image x all samples, no host-visible
		// checkpoint to report from), this loop already synchronizes once per
		// sample - report it in the exact "Scanlines remaining: N" shape
		// qt_gui/render_output_parser.h's parseScanlineProgress() already
		// expects from the CPU backend, so the GUI's progress bar picks this
		// up with no parser/GUI changes at all. `height` fills in for the
		// scanline unit (this backend has no real scanlines), scaled by
		// completed/total samples instead of completed/total rows.
		//
		// Throttled to progressPrintInterval (plus always the last sample,
		// so the bar still reaches 100%) - see that variable's own comment.
		const bool isLastSample = sampleIdx == samples_per_pixel - 1;
		if ((sampleIdx + 1) % progressPrintInterval == 0 || isLastSample) {
			const int completed = (int)((long long)height * (sampleIdx + 1) / samples_per_pixel);
			std::cout << "Scanlines remaining: " << (height - completed) << "\r" << std::flush;
		}
		++stats.samplesCompleted;
	}

	// -------------------------------------------------------------------------
	// Normalize (raw radiance + AOVs)
	// -------------------------------------------------------------------------
	// Deliberately BEFORE the world-pos-history overwrite and DI/GI's own
	// spatial-reuse below: SVGF's own temporal integrate (inside launchSvgf,
	// just below) needs d_worldPosHistory_ to still hold the PREVIOUS
	// render() call's content when it runs, exactly like DI's/GI's own
	// temporal reuse already reads it during the sampleIdx loop above,
	// before this call's own final positions overwrite it. Moving
	// normalization earlier (it has no dependency on world-pos at all -
	// only on d_fb_/d_weight_, already final once the sampleIdx loop above
	// exits) is what makes this ordering possible without SVGF needing a
	// SECOND, separate "pre-history-overwrite" hook of its own.
	launchNormalizeFramebuffer((unsigned int)numPixels, d_weightPtr, d_fbPtr);
	CUDA_CHECK(cudaStreamSynchronize(stream_));
	rc.d_fbPtr = d_fbPtr;
	rc.d_weightPtr = d_weightPtr;
	rc.d_activePixelMaskPtr = d_activePixelMaskPtr;
	rc.lp = lp;
	rc.stats = stats;
}

void WavefrontPathTracer::renderFinish(RenderCall &rc) {
	int width = rc.width;
	int height = rc.height;
	int samples_per_pixel = rc.samples_per_pixel;
	int max_depth = rc.max_depth;
	const auto &camera = rc.camera;
	float*  framebuffer = rc.framebuffer;
	CUdeviceptr d_materials = rc.d_materials;
	CUdeviceptr d_spheres = rc.d_spheres;
	CUdeviceptr d_quads = rc.d_quads;
	CUdeviceptr d_bilinear_patches = rc.d_bilinear_patches;
	CUdeviceptr d_triangles = rc.d_triangles;
	CUdeviceptr d_disks = rc.d_disks;
	CUdeviceptr d_cylinders = rc.d_cylinders;
	const int numPixels = rc.numPixels;
	CUdeviceptr d_fb = rc.d_fb;
	CUdeviceptr d_weight = rc.d_weight;
	float* d_weightPtr = rc.d_weightPtr;
	const bool needsAovGuideBuffers = rc.needsAovGuideBuffers;
	const bool needsWorldPosHistory = rc.needsWorldPosHistory;
	float3* d_albedoAovPtr = rc.d_albedoAovPtr;
	float3* d_normalAovPtr = rc.d_normalAovPtr;
	WavefrontLaunchParams lp = rc.lp;
	WavefrontRenderStats stats = rc.stats;
	// Normalize the accumulated albedo/normal AOV buffers (plain mean over
	// samples_per_pixel - see launchNormalizeAovBuffers()'s own comment) -
	// needed by EITHER the OptiX AI denoiser below or SVGF, so gated on
	// either flag rather than duplicated per consumer.
	if (needsAovGuideBuffers) {
		launchNormalizeAovBuffers((unsigned int)numPixels, d_albedoAovPtr, d_normalAovPtr,
			(unsigned int)samples_per_pixel);
		CUDA_CHECK(cudaStreamSynchronize(stream_));
	}
	// Optional OptiX AI denoiser post-process (--denoise, this backend's own
	// support - see setDenoiseEnabled()'s own comment), on-device, in place.
	// A failure here is logged (inside denoise() itself) and otherwise
	// ignored - the already-valid noisy render is still a correct result,
	// matching the recursive backend's own precedent. Skipped when svgfEnabled_
	// is ALSO true (not just an unreachable GUI combination - the GUI keeps
	// its own two checkboxes mutually exclusive, but nothing below this
	// point enforces that on a caller who sets both backend flags directly,
	// e.g. a test or a future non-GUI entry point) - SVGF treats whatever is
	// in d_fb as this frame's raw noisy sample, so denoising it first would
	// corrupt SVGF's own variance/history statistics rather than compose
	// with it.
	if (denoiseEnabled_ && !svgfEnabled_) {
		denoise(d_fb, (unsigned int)width, (unsigned int)height,
			denoiserResources_.albedoAov, denoiserResources_.normalAov);
	}
	// SVGF spatiotemporal denoiser (see wavefront_kernels_svgf.cu's own
	// header comment) - an alternative to the OptiX AI denoiser above, not
	// layered on top of it (the GUI is expected to present these as
	// mutually-exclusive modes; see the skip condition just above for what
	// backend-level enforcement actually looks like when both flags are set
	// anyway). Filters d_fb in place. MUST run before the world-pos-history
	// overwrite just below - see this block's own opening comment.
	if (svgfEnabled_) {
		launchSvgf(reinterpret_cast<float3*>(d_fb), d_albedoAovPtr, camera.origin, d_weightPtr);
		CUDA_CHECK(cudaStreamSynchronize(stream_));
	}

	// Neural temporal upscale (Live Preview only) - see wavefront_kernels_
	// upscale.cu's own header comment. MUST run before the world-pos-history/
	// prevRestirCamera_ overwrite just below (same reasoning as launchSvgf()'s
	// own placement just above it, and launchNeuralUpscaleUpdate()'s own
	// declaration comment, wavefront_path_tracer.h) - unlike
	// launchNrcTrainingUpdate() further down, which runs AFTER that overwrite
	// and therefore must never need last frame's camera/history. Reads d_fb
	// (this frame's own final low-res linear color, after SVGF/the OptiX AI
	// denoiser above have already had their turn on it, same "whatever's in
	// d_fb now IS the ground truth this frame" treatment SVGF itself gives
	// it).
	launchNeuralUpscaleUpdate(reinterpret_cast<const float3*>(d_fb), width, height, camera);

	// ReSTIR's end-of-call history update. d_reservoirs_/d_worldPos_ (this
	// call's own, fully overwritten by the sampleIdx loop above) are about to
	// be fully overwritten again on the next restirEnabled_ render() call
	// regardless, so there is nothing to preserve in them across calls -
	// only the history buffers (read at the START of the next call) need
	// this call's final content.
	//
	// Spatial reuse runs here (once per render() call, over the whole
	// per-pixel image - see wavefront_kernels_restir.cu's own header comment
	// for why it can't run inline per-hit like temporal reuse does), writing
	// its combined result DIRECTLY into d_reservoirsHistory_ - which becomes
	// the NEXT call's temporal-reuse source, closing the loop between the two
	// reuse passes across frames.
	// world-pos history + camera-basis bookkeeping is SHARED between DI, GI,
	// AND SVGF (all only ever depend on x0's own position and the camera,
	// never on which technique is consuming them - wavefront_path_tracer.h's
	// own GI/SVGF buffer comments) - updated once, whenever ANY is enabled,
	// not duplicated per technique. Deliberately AFTER launchSvgf above -
	// see that block's own comment for why the ordering matters here.
	if (needsWorldPosHistory) {
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_worldPosHistory_),
								   reinterpret_cast<void*>(d_worldPos_),
								   numPixels * sizeof(float4), cudaMemcpyDeviceToDevice, stream_));
		prevRestirCamera_.origin = camera.origin;
		prevRestirCamera_.lowerLeftCorner = camera.lower_left_corner;
		prevRestirCamera_.horizontal = camera.horizontal;
		prevRestirCamera_.vertical = camera.vertical;
		restirHistoryWidth_ = width;
		restirHistoryHeight_ = height;
	}
	if (restirEnabled_) {
		launchRestirSpatialReuse(
			reinterpret_cast<const SphereData*>(d_spheres),
			reinterpret_cast<const QuadData*>(d_quads),
			reinterpret_cast<const TriangleData*>(d_triangles),
			reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches),
			reinterpret_cast<const DiskData*>(d_disks),
			reinterpret_cast<const CylinderData*>(d_cylinders),
			reinterpret_cast<const MaterialData*>(d_materials));
		launchRestirVolumeSpatialReuse(
			reinterpret_cast<const SphereData*>(d_spheres),
			reinterpret_cast<const QuadData*>(d_quads),
			reinterpret_cast<const TriangleData*>(d_triangles),
			reinterpret_cast<const BilinearPatchData*>(d_bilinear_patches),
			reinterpret_cast<const DiskData*>(d_disks),
			reinterpret_cast<const CylinderData*>(d_cylinders),
			reinterpret_cast<const MaterialData*>(d_materials));
		restirHistoryValid_ = true;
	}
	if (restirGiEnabled_) {
		launchGiSpatialReuse();
		restirGiHistoryValid_ = true;
	}

	// World-space irradiance probe cache update (Live Preview only) - see
	// launchProbeCacheUpdate()'s own header comment. `lp` is still exactly
	// as built above (never mutated in place anywhere in this function - the
	// per-bounce shadow/probe-walk launches above all worked on their own
	// local copies), so it already carries this call's own
	// traversable/geometry/materials/lights/textures pointers unchanged.
	launchProbeCacheUpdate(lp, camera.backgroundColor, camera.shadowRayEpsilon);

	// Neural Radiance Cache training pipeline (Live Preview only) - see
	// launchNrcTrainingUpdate()'s own header comment. Same "lp is still
	// exactly as built above" reasoning as launchProbeCacheUpdate() just
	// above.
	launchNrcTrainingUpdate(lp, camera, camera.shadowRayEpsilon);

	// -------------------------------------------------------------------------
	// Copy to host
	// -------------------------------------------------------------------------

	CUDA_CHECK(cudaMemcpy(framebuffer, reinterpret_cast<void*>(d_fb),
						  numPixels * sizeof(float3), cudaMemcpyDeviceToHost));

	// d_fb/d_weight are no longer freed here - see fbCapacity_'s own comment
	// above. Released either inline above (the `if (fbCapacity_ != numPixels)`
	// branch, when a later call asks for a different resolution) or in
	// cleanup(), once this backend is actually torn down - there is no
	// separate "freeFramebuffer()" method.

	std::cout << "[WavefrontPathTracer] Rendered " << width << "x" << height
			  << " (" << samples_per_pixel << " spp, " << max_depth << " bounces)\n";

	// [WF-STATS] summary block (pbrt-v4 STAT_COUNTER-inspired) - wavefront-
	// only, since its per-bounce queue sizes are the one backend with real
	// host-visible counters already computed for launch sizing (CPU/
	// recursive-GPU need separate infra - see src/shared/render_stats.h and
	// launcher/main.cpp's own "[STATS]" block). Gated behind the same
	// RAY_TRACER_STATS env var main.cpp sets for --stats (same same-process
	// env-var pattern as RAY_TRACER_WAVEFRONT above) - this used to print
	// unconditionally; making --stats mean "opt-in on every backend" instead
	// of "wavefront always, everything else never" is worth the one-time
	// behavior change.
#pragma warning(suppress: 4996)
	if (std::getenv("RAY_TRACER_STATS")) {
		long long totalHitQueueHits = stats.hits + stats.simpleHits + stats.dielectricHits;
		double avgBouncesPerSample = stats.samplesCompleted > 0
			? double(stats.bounceIterations) / double(stats.samplesCompleted)
			: 0.0;
		std::cout << "[WF-STATS] ── Wavefront Render Statistics ──────────────────\n";
		std::cout << "[WF-STATS] Samples completed  : " << stats.samplesCompleted << " / " << samples_per_pixel << "\n";
		std::cout << "[WF-STATS] Bounce iterations  : " << stats.bounceIterations
				  << "  (avg " << avgBouncesPerSample << " / sample, max_depth=" << max_depth << ")\n";
		std::cout << "[WF-STATS] Primary+bounce rays: " << stats.primaryRays << "\n";
		std::cout << "[WF-STATS] Hits (regular/simple/dielectric): "
				  << stats.hits << " / " << stats.simpleHits << " / " << stats.dielectricHits
				  << "  (total " << totalHitQueueHits << ")\n";
		std::cout << "[WF-STATS] Misses             : " << stats.misses << "\n";
		std::cout << "[WF-STATS] Shadow rays (NEE)  : " << stats.shadowRays << "\n";
		std::cout << "[WF-STATS] BSSRDF probe rays  : " << stats.probeRays
				  << "  |  exits: " << stats.probeExits << "\n";
		std::cout << "[WF-STATS] ─────────────────────────────────────────────────\n";
	}
	rc.d_fb = d_fb;
	rc.d_weight = d_weight;
	rc.d_weightPtr = d_weightPtr;
	rc.d_albedoAovPtr = d_albedoAovPtr;
	rc.d_normalAovPtr = d_normalAovPtr;
	rc.lp = lp;
	rc.stats = stats;
}

bool WavefrontPathTracer::render(
	int width, int height, int samples_per_pixel, int max_depth,
	const GpuCameraParams& camera,
	float*  framebuffer,          // host-side output
	OptixTraversableHandle gas_handle,
	CUdeviceptr d_materials,
	CUdeviceptr d_spheres,
	CUdeviceptr d_quads,
	CUdeviceptr d_light_indices,
	CUdeviceptr d_lightKinds,
	CUdeviceptr d_alias_table,
	unsigned int num_materials,
	unsigned int num_spheres,
	unsigned int num_quads,
	unsigned int num_lights,
	CUdeviceptr d_punctual_lights,
	unsigned int num_punctual_lights,
	CUdeviceptr d_bilinear_patches,
	unsigned int num_bilinear_patches,
	CUdeviceptr d_triangles,
	unsigned int num_triangles,
	CUdeviceptr d_disks,
	unsigned int num_disks,
	CUdeviceptr d_cylinders,
	unsigned int num_cylinders)
{
	RenderCall rc{width, height, samples_per_pixel, max_depth, camera, framebuffer, gas_handle, d_materials, d_spheres, d_quads, d_light_indices, d_lightKinds, d_alias_table, num_materials, num_spheres, num_quads, num_lights, d_punctual_lights, num_punctual_lights, d_bilinear_patches, num_bilinear_patches, d_triangles, num_triangles, d_disks, num_disks, d_cylinders, num_cylinders};
	if (!renderPrepareBuffers(rc)) return false;
	renderPrepareRestirSvgf(rc);
	renderPrepareLaunch(rc);
	renderSampleLoop(rc);
	renderFinish(rc);
	return true;
}

// ============================================================================
// cleanup
// ============================================================================

void WavefrontPathTracer::destroyProgramGroups() {
	auto destroyPG = [](OptixProgramGroup& pg) {
		if (pg) { optixProgramGroupDestroy(pg); pg = nullptr; }
	};
	destroyPG(raygenIntersectPG_);   destroyPG(missRadiancePG_);
	destroyPG(hitSpherePG_);         destroyPG(hitQuadPG_);        destroyPG(hitBilinearPatchPG_); destroyPG(hitTrianglePG_);
	destroyPG(raygenShadowPG_);      destroyPG(missShadowPG_);
	destroyPG(anyhitShadowSpherePG_); destroyPG(anyhitShadowQuadPG_); destroyPG(anyhitShadowBilinearPatchPG_); destroyPG(anyhitShadowTrianglePG_);
	destroyPG(raygenProbePG_);       destroyPG(missProbePG_);
	destroyPG(hitProbeSpherePG_);    destroyPG(hitProbeQuadPG_);   destroyPG(hitProbeBilinearPatchPG_); destroyPG(hitProbeTrianglePG_);
	destroyPG(raygenProbeCachePG_);
	destroyPG(exceptionPG_);
}

void WavefrontPathTracer::destroySBT() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_intersectRaygenRecord_); freeDev(d_intersectMissRecord_); freeDev(d_intersectHitRecords_);
	freeDev(d_shadowRaygenRecord_);    freeDev(d_shadowMissRecord_);    freeDev(d_shadowHitRecords_);
	freeDev(d_probeRaygenRecord_);     freeDev(d_probeMissRecord_);     freeDev(d_probeHitRecords_);
	freeDev(d_probeCacheRaygenRecord_);  // probeCacheSBT_'s own miss/hit/exception records ARE probeSBT_'s, freed just above
	freeDev(d_intersectExceptionRecord_); freeDev(d_shadowExceptionRecord_); freeDev(d_probeExceptionRecord_);
	intersectSBT_ = {};
	shadowSBT_    = {};
	probeSBT_     = {};
	probeCacheSBT_ = {};
}

bool WavefrontPathTracer::readWorldPosBuffer(unsigned int width, unsigned int height, std::vector<float>& out) const {
	const int numPixels = static_cast<int>(width) * static_cast<int>(height);
	// worldPosOutputEnabled_ is checked explicitly (not just inferred from
	// d_worldPos_'s existence) because d_worldPos_ can now also be allocated
	// purely due to restirEnabled_ (ReSTIR's own internal use of it, see
	// this class's render()'s own allocation comment) without the CALLER
	// ever having asked for readback via setWorldPosOutputEnabled(true) -
	// without this check, a caller that never opted in could still get back
	// real (unrequested) world-position data whenever ReSTIR happens to be
	// enabled, instead of this function's own documented false/failure.
	if (!worldPosOutputEnabled_ || !d_worldPos_ || worldPosCapacity_ != numPixels) return false;
	const size_t count = static_cast<size_t>(numPixels) * 4;
	out.resize(count);
	CUDA_CHECK(cudaMemcpy(out.data(), reinterpret_cast<void*>(d_worldPos_),
						   count * sizeof(float), cudaMemcpyDeviceToHost));
	return true;
}

bool WavefrontPathTracer::readNeuralUpscaleBuffer(unsigned int width, unsigned int height, std::vector<float>& out) const {
	if (!neuralUpscaleEnabled_ || !d_upscaleOutput_) return false;
	const int numHighResPixels = static_cast<int>(width) * static_cast<int>(height) *
		temporalUpscaleFactor_ * temporalUpscaleFactor_;
	if (upscaleOutputCapacity_ != numHighResPixels) return false;
	const size_t count = static_cast<size_t>(numHighResPixels) * 3;
	out.resize(count);
	CUDA_CHECK(cudaMemcpy(out.data(), reinterpret_cast<void*>(d_upscaleOutput_),
						   count * sizeof(float), cudaMemcpyDeviceToHost));
	return true;
}

void WavefrontPathTracer::cleanup() {
	destroySBT();
	destroyProgramGroups();
	freeQueues();
	destroyDenoiser();
	destroyAovBuffers();
	if (d_fb_) { cudaFree(reinterpret_cast<void*>(d_fb_)); d_fb_ = 0; }
	if (d_weight_) { cudaFree(reinterpret_cast<void*>(d_weight_)); d_weight_ = 0; }
	fbCapacity_ = 0;
	if (d_activePixelMask_) { cudaFree(reinterpret_cast<void*>(d_activePixelMask_)); d_activePixelMask_ = 0; }
	activePixelMaskCapacity_ = 0;
	if (d_worldPos_) { cudaFree(reinterpret_cast<void*>(d_worldPos_)); d_worldPos_ = 0; }
	worldPosCapacity_ = 0;

	if (d_reservoirs_) { cudaFree(reinterpret_cast<void*>(d_reservoirs_)); d_reservoirs_ = 0; }
	reservoirsCapacity_ = 0;

	freeDeviceBuffer(d_volumeReservoirs_, volumeReservoirsCapacity_);
	freeDeviceBuffer(d_volumeMatIdx_, volumeMatIdxCapacity_);
	freeDeviceBuffer(d_volumePhaseWoG_, volumePhaseWoGCapacity_);
	freeDeviceBuffer(d_volumeEntryPoint_, volumeEntryPointCapacity_);
	freeDeviceBuffer(d_volumeReservoirsHistory_, volumeReservoirsHistoryCapacity_);

	// World-space irradiance probe cache (Live Preview only) - d_probeGrid_
	// itself is OptiXRenderer-owned (never freed here, see setProbeGrid()'s
	// own comment); everything below is this class's own, lazily allocated
	// on first use by launchProbeCacheUpdate().
	{
		auto freeDev = [](CUdeviceptr& p) {
			if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
		};
		freeDev(d_probeCacheRayItems_);    freeDev(d_probeCacheRayCounter_);
		freeDev(d_probeCacheHitItems_);    freeDev(d_probeCacheHitCounter_);
		freeDev(d_probeCacheRadianceOut_); freeDev(d_probeCacheHitDistOut_);
		freeDev(d_probeCacheDirections_);
		freeDev(d_nrcWeights_); freeDev(d_nrcAdamM_); freeDev(d_nrcAdamV_);
		freeDev(d_nrcGradAccum_); freeDev(d_nrcTrainingRecords_); freeDev(d_nrcValidRecordCounter_);
		freeDev(d_upscaleWeights_); freeDev(d_upscaleAdamM_); freeDev(d_upscaleAdamV_);
		freeDev(d_upscaleGradAccum_); freeDev(d_upscaleValidRecordCounter_);
		freeDev(d_upscaleMotionVectors_); freeDev(d_upscaleOutput_);
		freeDev(d_upscaleCurrent_); freeDev(d_upscaleHistory_); freeDev(d_upscaleForwardCache_);
	}

	if (intersectPipeline_) { optixPipelineDestroy(intersectPipeline_); intersectPipeline_ = nullptr; }
	if (shadowPipeline_)    { optixPipelineDestroy(shadowPipeline_);    shadowPipeline_    = nullptr; }
	if (wfModule_)          { optixModuleDestroy(wfModule_);            wfModule_          = nullptr; }
	if (d_wfLaunchParams_)  { cudaFree(reinterpret_cast<void*>(d_wfLaunchParams_)); d_wfLaunchParams_ = 0; }
	if (simpleMaterialStream_) { cudaStreamDestroy(simpleMaterialStream_); simpleMaterialStream_ = nullptr; }
	if (dielectricMaterialStream_) { cudaStreamDestroy(dielectricMaterialStream_); dielectricMaterialStream_ = nullptr; }
}

} // namespace optix_renderer

// Pull in the sRGB spectral table data so it links into this TU
// (avoids needing rgb_spectrum_table_data.cpp as a separate project source).
// Two OTHER independent copies of this same wiring exist -
// scene_metadata/scene_metadata.vcxproj's own ClCompile (see its comment
// for why cpu_renderer.vcxproj itself can't carry this) and
// tests/CMakeLists.txt's source list - there is no single shared target
// providing this dependency to every consumer once.
#include "../../src/data/rgb_spectrum_table_data.cpp"
