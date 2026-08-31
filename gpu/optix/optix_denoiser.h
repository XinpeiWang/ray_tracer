#pragma once

/// @file optix_denoiser.h
/// @brief Shared OptiX AI denoiser lifecycle (state/scratch/intensity
///        buffers + albedo/normal AOV guide buffers), used identically by
///        both OptiXRenderer (recursive backend) and WavefrontPathTracer
///        (wavefront backend). The two backends are NOT related by
///        inheritance - OptiXRenderer owns a WavefrontPathTracer rather
///        than deriving from a common base - but both end up holding the
///        same context_/stream_ values at runtime (OptiXRenderer::
///        initialize() hands its own context_/stream_ to the wavefront
///        tracer's initialize()), so a free function taking those as
///        explicit parameters is genuinely coupling-free: no shared base
///        class, no back-reference, each caller just owns its own
///        DenoiserResources instance and passes its own context/stream.
///
/// Struct/function shapes were verified against the actual installed
/// OptiX SDK 9.1.0 headers (optix_types.h/optix_host.h), not assumed from
/// pbrt-v4's source (which targets its own, possibly different, SDK
/// version).

#include "optix_types.h"
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>

/// @brief Persisted OptiX denoiser state + its albedo/normal AOV guide
///        buffers. One instance per backend (recursive, wavefront) - each
///        keyed to its own resolution, recreated only on a resolution
///        change (see runDenoiser()/ensureAovBuffers()'s own comments).
struct DenoiserResources {
	OptixDenoiser denoiser = nullptr;
	CUdeviceptr state = 0;
	CUdeviceptr scratch = 0;
	CUdeviceptr intensity = 0;
	size_t stateSizeInBytes = 0;
	size_t scratchSizeInBytes = 0;
	size_t computeIntensitySizeInBytes = 0;
	unsigned int width = 0;
	unsigned int height = 0;

	CUdeviceptr albedoAov = 0;
	CUdeviceptr normalAov = 0;
	unsigned int aovWidth = 0;
	unsigned int aovHeight = 0;
};

/// @brief Free the persisted denoiser and its device buffers (see
///        DenoiserResources::denoiser's own comment). Safe to call when
///        nothing is allocated. Called both from cleanup and from
///        runDenoiser() itself when the requested resolution no longer
///        matches width_/height_.
inline void destroyDenoiserResources(DenoiserResources& r) noexcept {
	if (r.intensity) { cudaFree(reinterpret_cast<void*>(r.intensity)); r.intensity = 0; }
	if (r.scratch)   { cudaFree(reinterpret_cast<void*>(r.scratch));   r.scratch = 0; }
	if (r.state)     { cudaFree(reinterpret_cast<void*>(r.state));     r.state = 0; }
	if (r.denoiser)  { optixDenoiserDestroy(r.denoiser); r.denoiser = nullptr; }
	r.width = 0;
	r.height = 0;
}

/// @brief Free albedoAov/normalAov (see their own comment). Safe to call
///        when nothing is allocated.
inline void destroyAovBuffers(DenoiserResources& r) noexcept {
	if (r.albedoAov) { cudaFree(reinterpret_cast<void*>(r.albedoAov)); r.albedoAov = 0; }
	if (r.normalAov) { cudaFree(reinterpret_cast<void*>(r.normalAov)); r.normalAov = 0; }
	r.aovWidth = 0;
	r.aovHeight = 0;
}

/// @brief (Re)allocate albedoAov/normalAov for the given resolution -
///        recreated only when width/height changed since the last call
///        (same pattern as runDenoiser()'s own recreate-on-resolution-
///        change check) - a no-op on the steady-state per-frame call
///        (e.g. video mode, where every frame shares one resolution).
inline void ensureAovBuffers(DenoiserResources& r, unsigned int width, unsigned int height) {
	if (r.albedoAov && r.normalAov && width == r.aovWidth && height == r.aovHeight) {
		return;
	}
	destroyAovBuffers(r);
	size_t fbSize = static_cast<size_t>(width) * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&r.albedoAov), fbSize));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&r.normalAov), fbSize));
	r.aovWidth = width;
	r.aovHeight = height;
}

/// @brief Run the OptiX AI denoiser on an in-device float3 buffer, in
///        place. The denoiser (and its state/scratch buffers) is
///        persisted in `r` across calls and only (re)created when the
///        resolution changes.
/// @param r        Persisted state, owned by the caller (one per backend).
/// @param context  This backend's OptixDeviceContext (recreated denoiser_
///                 is bound to it via optixDenoiserCreate).
/// @param stream   This backend's CUstream (setup/computeIntensity/invoke
///                 all run on it).
/// @param d_buffer In/out device buffer to denoise, width*height float3.
/// @param d_albedo Device buffer with the same width*height layout as
///                 d_buffer, or 0 to denoise without an albedo guide layer.
/// @param d_normal Same, for the world-space normal guide layer, or 0 to
///                 denoise without a normal guide layer.
/// @param logTag   Short prefix for failure log lines (e.g. "[OptiX]" /
///                 "[Wavefront]") so a denoiser failure's origin is
///                 distinguishable in mixed CLI output.
/// @return false on any failure (logged); non-fatal for the caller, since
///         a failed denoise leaves the buffer's already-valid noisy render
///         data untouched.
inline bool runDenoiser(DenoiserResources& r, OptixDeviceContext context, CUstream stream,
	CUdeviceptr d_buffer, unsigned int width, unsigned int height,
	CUdeviceptr d_albedo, CUdeviceptr d_normal, const char* logTag) {
	// Every OptiX/CUDA call below can fail; unlike a render launch sequence
	// (which throws via OPTIX_CHECK/CUDA_CHECK - a failure there means the
	// render itself is unusable), a denoiser failure should leave the
	// caller's already-valid noisy render intact rather than crashing the
	// whole render() call - hence catching here and returning false instead
	// of using those throwing macros directly.
	auto fail = [&](const char* what, OptixResult res) {
		std::cerr << logTag << " Denoiser " << what << " failed (code " << res << ")\n";
		destroyDenoiserResources(r);  // don't leave a half-built denoiser cached for next call
		return false;
	};

	// r.denoiser (and its buffers) is persisted across calls - only
	// (re)created here when this is the first call, a previous call failed
	// (destroyDenoiserResources() clears r.denoiser), or the requested
	// resolution changed (e.g. GUI window resize, or a different scene
	// rendered at a different width/height in the same process). Fixes a
	// real inefficiency: video mode's per-frame render calls all share one
	// resolution, so re-running denoiser create+setup+3 allocations on
	// every one of hundreds of frames was pure waste - now only the first
	// frame pays that cost.
	if (!r.denoiser || width != r.width || height != r.height) {
		destroyDenoiserResources(r);

		OptixDenoiserOptions options = {};
		options.guideAlbedo = d_albedo ? 1 : 0;
		options.guideNormal = d_normal ? 1 : 0;
		options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

		OptixResult res = optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &r.denoiser);
		if (res != OPTIX_SUCCESS) return fail("create", res);

		OptixDenoiserSizes sizes = {};
		res = optixDenoiserComputeMemoryResources(r.denoiser, width, height, &sizes);
		if (res != OPTIX_SUCCESS) return fail("computeMemoryResources", res);

		// r.scratch is shared by optixDenoiserSetup/Invoke (needs
		// withoutOverlapScratchSizeInBytes - no tiling, full image in one
		// call) AND optixDenoiserComputeIntensity (needs the separately-
		// documented computeIntensitySizeInBytes) below. These are two
		// independently-sized requirements per the SDK header, not
		// guaranteed ordered either way, so allocate the max of both rather
		// than assuming one covers the other.
		r.stateSizeInBytes = sizes.stateSizeInBytes;
		r.scratchSizeInBytes = sizes.withoutOverlapScratchSizeInBytes;
		r.computeIntensitySizeInBytes = sizes.computeIntensitySizeInBytes;
		size_t scratchAllocSize = r.scratchSizeInBytes;
		if (r.computeIntensitySizeInBytes > scratchAllocSize)
			scratchAllocSize = r.computeIntensitySizeInBytes;

		if (cudaMalloc(reinterpret_cast<void**>(&r.state), r.stateSizeInBytes) != cudaSuccess)
			return fail("state alloc", OPTIX_ERROR_INTERNAL_ERROR);
		if (cudaMalloc(reinterpret_cast<void**>(&r.scratch), scratchAllocSize) != cudaSuccess)
			return fail("scratch alloc", OPTIX_ERROR_INTERNAL_ERROR);
		if (cudaMalloc(reinterpret_cast<void**>(&r.intensity), sizeof(float)) != cudaSuccess)
			return fail("intensity alloc", OPTIX_ERROR_INTERNAL_ERROR);

		res = optixDenoiserSetup(r.denoiser, stream, width, height,
			r.state, r.stateSizeInBytes, r.scratch, r.scratchSizeInBytes);
		if (res != OPTIX_SUCCESS) return fail("setup", res);

		r.width = width;
		r.height = height;
	}

	OptixImage2D image = {};
	image.data = d_buffer;
	image.width = width;
	image.height = height;
	image.rowStrideInBytes = width * sizeof(float3);
	image.pixelStrideInBytes = sizeof(float3);
	image.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixResult res = optixDenoiserComputeIntensity(r.denoiser, stream, &image, r.intensity,
		r.scratch, r.computeIntensitySizeInBytes);
	if (res != OPTIX_SUCCESS) return fail("computeIntensity", res);

	OptixDenoiserParams params = {};
	params.hdrIntensity = r.intensity;
	params.blendFactor = 0.0f;  // 100% denoised output

	// Albedo/normal guide layers - left zero-initialised (data pointer 0,
	// the documented "not provided" state) when the caller didn't pass a
	// buffer, matching options.guideAlbedo/guideNormal being 0 in that case
	// above - AOV model kind still denoises without them, just less
	// accurately. `normal` is expected already world-space (no transform
	// needed), matching what OPTIX_DENOISER_MODEL_KIND_AOV expects.
	OptixDenoiserGuideLayer guideLayer = {};
	if (d_albedo) {
		guideLayer.albedo.data = d_albedo;
		guideLayer.albedo.width = width;
		guideLayer.albedo.height = height;
		guideLayer.albedo.rowStrideInBytes = width * sizeof(float3);
		guideLayer.albedo.pixelStrideInBytes = sizeof(float3);
		guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	}
	if (d_normal) {
		guideLayer.normal.data = d_normal;
		guideLayer.normal.width = width;
		guideLayer.normal.height = height;
		guideLayer.normal.rowStrideInBytes = width * sizeof(float3);
		guideLayer.normal.pixelStrideInBytes = sizeof(float3);
		guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	}

	OptixDenoiserLayer layer = {};
	layer.input = image;
	// In-place: input and output may refer to the same buffer per
	// optixDenoiserInvoke's own doc comment (pixel formats already match).
	// Avoids a second width*height*sizeof(float3) allocation.
	layer.output = image;
	layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

	res = optixDenoiserInvoke(r.denoiser, stream, &params,
		r.state, r.stateSizeInBytes,
		&guideLayer, &layer, 1,
		0, 0,
		r.scratch, r.scratchSizeInBytes);
	if (res != OPTIX_SUCCESS) return fail("invoke", res);

	if (cudaStreamSynchronize(stream) != cudaSuccess)
		return fail("sync", OPTIX_ERROR_INTERNAL_ERROR);

	return true;
}
