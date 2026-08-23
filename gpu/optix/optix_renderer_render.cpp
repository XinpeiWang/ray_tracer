/// @file optix_renderer_render.cpp
/// @brief OptiX Renderer Implementation -- render/denoise/cleanup.
/// @details Split out of optix_renderer.cpp (see optix_renderer_init.cpp and
///          optix_renderer_scene.cpp for the other two thirds, and
///          optix_renderer.h for the shared class declaration). This part
///          owns launching the recursive-backend path tracer, AI denoising,
///          AOV readback, resource cleanup, and delegating to the
///          wavefront/SPPM path tracers.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
#include "optix_math_helpers.h"
#include "wavefront_path_tracer.h"
#include "sppm_path_tracer.h"
#include <cuda.h>
#include <iostream>

bool OptiXRenderer::render(
	unsigned int width,
	unsigned int height,
	unsigned int samplesPerPixel,
	unsigned int maxDepth,
	const GpuCameraParams& camera,
	float* outputFramebuffer
) {
	// Camera setup: inject the device pointers for the (host-precomputed,
	// scene-build-time-uploaded) lens/exit-pupil-bounds tables into a local
	// mutable copy. scene_builder.cpp's out_camera_extra can't know these
	// device pointers at scene-build time (they aren't allocated until
	// buildScene() below runs), so this is the one place both the recursive
	// and wavefront strategies get a fully-populated GpuCameraParams from -
	// neither WavefrontPathTracer nor the recursive raygen need any further
	// camera wiring changes for CameraKind::Realistic.
	GpuCameraParams gpuCam = camera;
	if (gpuCam.kind == CameraKind::Realistic) {
		gpuCam.lensElements = reinterpret_cast<GpuLensElement*>(d_lensElements_);
		gpuCam.exitPupilBounds = reinterpret_cast<GpuExitPupilBounds*>(d_exitPupilBounds_);
		gpuCam.numLensElements = static_cast<int>(numLensElements_);
		gpuCam.numExitPupilBounds = static_cast<int>(numExitPupilBounds_);
	}

	// Real importance-sampled HDR sky distribution (LightSource "infinite"
	// with an image - see optix_types.h's GpuSkyDistribution comment) - same
	// "device pointers patched in fresh here, not known at scene-build time"
	// lifecycle as CameraKind::Realistic's lens tables just above.
	// skyHeight_ <= 0 (the default, every scene with a constant-colour sky
	// or no infinite light at all) leaves gpuCam.skyDist zero-init'd, which
	// every call site on both GPU backends already treats as "fall back to
	// backgroundColor + uniform-sphere sampling".
	if (skyHeight_ > 0) {
		gpuCam.skyDist.width = skyWidth_;
		gpuCam.skyDist.height = skyHeight_;
		gpuCam.skyDist.scale = skyScale_;
		gpuCam.skyDist.imagePixels = reinterpret_cast<const float*>(d_skyImagePixels_);
		gpuCam.skyDist.marginalCdf = reinterpret_cast<const float*>(d_skyMarginalCdf_);
		gpuCam.skyDist.marginalFunc = reinterpret_cast<const float*>(d_skyMarginalFunc_);
		gpuCam.skyDist.marginalFuncInt = skyMarginalFuncInt_;
		gpuCam.skyDist.conditionalCdf = reinterpret_cast<const float*>(d_skyConditionalCdf_);
		gpuCam.skyDist.conditionalFunc = reinterpret_cast<const float*>(d_skyConditionalFunc_);
		gpuCam.skyDist.conditionalFuncInt = reinterpret_cast<const float*>(d_skyConditionalFuncInt_);
	}

	// Delegate to WavefrontPathTracer if enabled
	if (useWavefront_ && wavefrontTracer_) {
		// Set per render rather than once at enable time, so a scene switch
		// cannot leave a previous scene's table wired in.
		wavefrontTracer_->setInstancePrimBase(d_instanceBase_);
		wavefrontTracer_->setTextures(d_textures_, d_texturePixels_);
		wavefrontTracer_->setCloudMediums(d_cloudMediums_, numCloudMediums_);
		wavefrontTracer_->setRgbGridMediums(d_rgbGridMediums_, numRgbGridMediums_, d_rgbGridData_, rgbGridDataCount_);
		wavefrontTracer_->setGridMediums(d_gridMediums_, numGridMediums_, d_gridData_, gridDataCount_);
		wavefrontTracer_->setBssrdfTables(d_bssrdfTables_, numBssrdfTables_,
			d_bssrdfRhoSamples_, d_bssrdfRadiusSamples_, d_bssrdfProfile_, d_bssrdfProfileCdf_);
		wavefrontTracer_->setMeasuredTables(d_measuredTables_, numMeasuredTables_,
			d_measuredParamValues_, d_measuredData_, d_measuredMcdf_, d_measuredCcdf_);
		return wavefrontTracer_->render(
			(int)width, (int)height, (int)samplesPerPixel, (int)maxDepth,
			gpuCam,
			outputFramebuffer,
			gasHandle_,
			d_materials_, d_spheres_, d_quads_,
			d_lightIndices_, d_lightKinds_, d_aliasTable_,
			numMaterials_, numSpheres_, numQuads_, numLights_,
			d_punctualLights_, numPunctualLights_,
			d_bilinearPatches_, numBilinearPatches_,
			d_triangles_, numTriangles_,
			d_disks_, numDisks_, d_cylinders_, numCylinders_);
	}

	// Allocate framebuffer on device
	CUdeviceptr d_framebuffer;
	size_t fbSize = width * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_framebuffer), fbSize));

	// Denoiser albedo/normal guide-layer AOV buffers (recursive backend
	// only) - only touched when this render will actually be denoised.
	// Persisted across render() calls via ensureAovBuffers() (same
	// resolution-keyed recreate-on-change pattern as the denoiser's own
	// state/scratch buffers - see denoise()'s comment), instead of a fresh
	// cudaMalloc/cudaFree every call, which was wasted work on video mode's
	// hundreds of same-resolution per-frame render() calls. raygen only
	// writes to these when the LaunchParams pointer is non-null (see
	// optix_raygen.h's own null-check), so leaving them 0 here when not
	// denoising costs nothing extra on the device side.
	CUdeviceptr d_albedo = 0;
	CUdeviceptr d_normal = 0;
	if (denoiseEnabled_) {
		ensureAovBuffers(width, height);
		d_albedo = d_albedoAov_;
		d_normal = d_normalAov_;
	}

	// Setup launch params. Zero-initialised on purpose: LaunchParams is a POD
	// whose fields are assigned one by one below, so any field NOT assigned
	// here would otherwise hold stack garbage. instancePrimBase is read on the
	// device as "null means no instancing", and a garbage pointer there is an
	// out-of-bounds read inside a hit program - among the hardest bugs to see.
	LaunchParams params = {};
	params.framebuffer = reinterpret_cast<float3*>(d_framebuffer);
	params.albedoBuffer = reinterpret_cast<float3*>(d_albedo);  // null unless denoising - see alloc above
	params.normalBuffer = reinterpret_cast<float3*>(d_normal);
	params.width = width;
	params.height = height;
	params.samplesPerPixel = samplesPerPixel;
	params.maxDepth = maxDepth;
	params.frameNumber = 0;  // Could be animated

	// Camera setup
	params.camera = gpuCam;

	// Scene 
	params.traversable = gasHandle_;
	params.materials = reinterpret_cast<MaterialData*>(d_materials_);
	params.numMaterials = numMaterials_;
	params.cloudMediums = reinterpret_cast<CloudMedium<float>*>(d_cloudMediums_);
	params.numCloudMediums = numCloudMediums_;
	params.rgbGridMediums = reinterpret_cast<GpuRgbGridMedium*>(d_rgbGridMediums_);
	params.numRgbGridMediums = numRgbGridMediums_;
	params.rgbGridData = reinterpret_cast<float*>(d_rgbGridData_);
	params.rgbGridDataCount = rgbGridDataCount_;
	params.gridMediums = reinterpret_cast<GpuGridMedium*>(d_gridMediums_);
	params.numGridMediums = numGridMediums_;
	params.gridData = reinterpret_cast<float*>(d_gridData_);
	params.gridDataCount = gridDataCount_;
	params.bssrdfTables = reinterpret_cast<GpuBssrdfTable*>(d_bssrdfTables_);
	params.numBssrdfTables = numBssrdfTables_;
	params.bssrdfRhoSamples = reinterpret_cast<float*>(d_bssrdfRhoSamples_);
	params.bssrdfRadiusSamples = reinterpret_cast<float*>(d_bssrdfRadiusSamples_);
	params.bssrdfProfile = reinterpret_cast<float*>(d_bssrdfProfile_);
	params.bssrdfProfileCdf = reinterpret_cast<float*>(d_bssrdfProfileCdf_);
	params.measuredTables = reinterpret_cast<GpuMeasuredTable*>(d_measuredTables_);
	params.numMeasuredTables = numMeasuredTables_;
	params.measuredParamValues = reinterpret_cast<float*>(d_measuredParamValues_);
	params.measuredData = reinterpret_cast<float*>(d_measuredData_);
	params.measuredMcdf = reinterpret_cast<float*>(d_measuredMcdf_);
	params.measuredCcdf = reinterpret_cast<float*>(d_measuredCcdf_);
	params.textures = reinterpret_cast<TextureData*>(d_textures_);
	params.numTextures = numTextures_;
	params.texturePixels = reinterpret_cast<unsigned char*>(d_texturePixels_);
	params.spheres = reinterpret_cast<SphereData*>(d_spheres_);
	params.numSpheres = numSpheres_;
	params.quads = reinterpret_cast<QuadData*>(d_quads_);
	params.numQuads = numQuads_;
	params.bilinearPatches = reinterpret_cast<BilinearPatchData*>(d_bilinearPatches_);
	params.numBilinearPatches = numBilinearPatches_;
	params.disks = reinterpret_cast<DiskData*>(d_disks_);
	params.numDisks = numDisks_;
	params.cylinders = reinterpret_cast<CylinderData*>(d_cylinders_);
	params.numCylinders = numCylinders_;
	params.triangles = reinterpret_cast<TriangleData*>(d_triangles_);
	params.instancePrimBase = reinterpret_cast<int*>(d_instanceBase_);  // null unless the scene has placements
	params.numTriangles = numTriangles_;

	// Light sampling for MIS
	params.lightIndices = reinterpret_cast<int*>(d_lightIndices_);
	params.numLights = numLights_;
	params.lightKinds = reinterpret_cast<const GpuLightKind*>(d_lightKinds_);
	params.aliasTable = reinterpret_cast<GpuAliasEntry*>(d_aliasTable_);

	// Punctual (delta) lights
	params.punctualLights = reinterpret_cast<PunctualLightGPU*>(d_punctualLights_);
	params.numPunctualLights = numPunctualLights_;

	// Motion blur: only the scene(s) with moving spheres set this - see
	// buildScene()'s sceneHasMotion_ detection and optix_raygen.h's use of it.
	params.motionBlurEnabled = sceneHasMotion_;

	// Upload launch params
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_launchParams_),
		&params,
		sizeof(LaunchParams),
		cudaMemcpyHostToDevice
	));

	// Launch OptiX pipeline
	OPTIX_CHECK(optixLaunch(
		pipeline_,
		stream_,
		d_launchParams_,
		sizeof(LaunchParams),
		&sbt_,
		width,
		height,
		1  // depth
	));

	CUDA_CHECK(cudaStreamSynchronize(stream_));

	// Optional OptiX AI denoiser post-process (recursive backend only - see
	// enableDenoise()'s comment). Runs on-device, in place, before the
	// buffer is downloaded to host memory below. A failure here is logged
	// (inside denoise() itself) and otherwise ignored - the already-valid
	// noisy render is still a correct result.
	if (denoiseEnabled_) {
		denoise(d_framebuffer, width, height, d_albedo, d_normal);
	}

	// Download framebuffer
	CUDA_CHECK(cudaMemcpy(
		outputFramebuffer,
		reinterpret_cast<void*>(d_framebuffer),
		fbSize,
		cudaMemcpyDeviceToHost
	));

	// Cleanup framebuffer (transient). d_albedo/d_normal are NOT freed here -
	// they persist across render() calls, see ensureAovBuffers()'s comment.
	cudaFree(reinterpret_cast<void*>(d_framebuffer));

	std::cout << "[OptiX] Rendered " << width << "x" << height
		<< " @ " << samplesPerPixel << " spp\n";

	return true;
}

// ============================================================================
// denoise -- OptiX AI denoiser post-process (recursive backend only)
// ============================================================================
// See optix_renderer.h's enableDenoise()/denoise() comments for scope.
// Struct shapes and function signatures below were verified against the
// actual installed OptiX SDK 9.1.0 headers (optix_types.h/optix_host.h),
// not assumed from pbrt-v4's source (which targets its own, possibly
// different, SDK version) - see this project's Phase 3 plan for why that
// verification step mattered.
bool OptiXRenderer::denoise(CUdeviceptr d_buffer, unsigned int width, unsigned int height,
	CUdeviceptr d_albedo, CUdeviceptr d_normal) {
	// Every OptiX/CUDA call below can fail; unlike render()'s own launch
	// sequence (which throws via OPTIX_CHECK/CUDA_CHECK - a failure there
	// means the render itself is unusable), a denoiser failure should leave
	// the caller's already-valid noisy render intact rather than crashing
	// the whole render() call - hence catching here and returning false
	// instead of using those throwing macros directly.
	auto fail = [&](const char* what, OptixResult res) {
		std::cerr << "[OptiX] Denoiser " << what << " failed (code " << res << ")\n";
		destroyDenoiser();  // don't leave a half-built denoiser_ cached for next call
		return false;
	};

	// denoiser_ (and its buffers) is persisted across render() calls - see
	// its own header comment - only (re)created here when this is the first
	// call, a previous call failed (destroyDenoiser() clears denoiser_), or
	// the requested resolution changed (e.g. GUI window resize, or a
	// different scene rendered at a different width/height in the same
	// process - see g_uploaded_scene_id's own comment for the equivalent
	// per-scene-switch cache-invalidation pattern in optix_interface.cpp).
	// Fixes a real inefficiency: video mode's per-frame optix_render_main()
	// calls (main.cpp) all share one resolution, so re-running denoiser
	// create+setup+3 allocations on every one of hundreds of frames was
	// pure waste - now only the first frame pays that cost.
	if (!denoiser_ || width != denoiserWidth_ || height != denoiserHeight_) {
		destroyDenoiser();

		OptixDenoiserOptions options = {};
		options.guideAlbedo = d_albedo ? 1 : 0;
		options.guideNormal = d_normal ? 1 : 0;
		options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

		OptixResult res = optixDenoiserCreate(context_, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &denoiser_);
		if (res != OPTIX_SUCCESS) return fail("create", res);

		OptixDenoiserSizes sizes = {};
		res = optixDenoiserComputeMemoryResources(denoiser_, width, height, &sizes);
		if (res != OPTIX_SUCCESS) return fail("computeMemoryResources", res);

		// denoiserScratch_ is shared by optixDenoiserSetup/Invoke (needs
		// withoutOverlapScratchSizeInBytes - no tiling, full image in one
		// call) AND optixDenoiserComputeIntensity (needs the separately-
		// documented computeIntensitySizeInBytes) below. These are two
		// independently-sized requirements per the SDK header, not
		// guaranteed ordered either way, so allocate the max of both rather
		// than assuming one covers the other.
		denoiserStateSizeInBytes_ = sizes.stateSizeInBytes;
		denoiserScratchSizeInBytes_ = sizes.withoutOverlapScratchSizeInBytes;
		denoiserComputeIntensitySizeInBytes_ = sizes.computeIntensitySizeInBytes;
		size_t scratchAllocSize = denoiserScratchSizeInBytes_;
		if (denoiserComputeIntensitySizeInBytes_ > scratchAllocSize)
			scratchAllocSize = denoiserComputeIntensitySizeInBytes_;

		if (cudaMalloc(reinterpret_cast<void**>(&denoiserState_), denoiserStateSizeInBytes_) != cudaSuccess)
			return fail("state alloc", OPTIX_ERROR_INTERNAL_ERROR);
		if (cudaMalloc(reinterpret_cast<void**>(&denoiserScratch_), scratchAllocSize) != cudaSuccess)
			return fail("scratch alloc", OPTIX_ERROR_INTERNAL_ERROR);
		if (cudaMalloc(reinterpret_cast<void**>(&denoiserIntensity_), sizeof(float)) != cudaSuccess)
			return fail("intensity alloc", OPTIX_ERROR_INTERNAL_ERROR);

		res = optixDenoiserSetup(denoiser_, stream_, width, height,
			denoiserState_, denoiserStateSizeInBytes_, denoiserScratch_, denoiserScratchSizeInBytes_);
		if (res != OPTIX_SUCCESS) return fail("setup", res);

		denoiserWidth_ = width;
		denoiserHeight_ = height;
	}

	OptixImage2D image = {};
	image.data = d_buffer;
	image.width = width;
	image.height = height;
	image.rowStrideInBytes = width * sizeof(float3);
	image.pixelStrideInBytes = sizeof(float3);
	image.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixResult res = optixDenoiserComputeIntensity(denoiser_, stream_, &image, denoiserIntensity_,
		denoiserScratch_, denoiserComputeIntensitySizeInBytes_);
	if (res != OPTIX_SUCCESS) return fail("computeIntensity", res);

	OptixDenoiserParams params = {};
	params.hdrIntensity = denoiserIntensity_;
	params.blendFactor = 0.0f;  // 100% denoised output

	// Albedo/normal guide layers (recursive backend's per-pixel AOV buffers
	// - see optix_raygen.h's albedo_sum/normal_sum accumulation and this
	// project's plan for why no atomics are needed to produce them). Left
	// zero-initialised (data pointer 0, the documented "not provided" state)
	// when the caller didn't pass a buffer, matching options.guideAlbedo/
	// guideNormal being 0 in that case above - AOV model kind still denoises
	// without them, just less accurately. `normal` is already world-space
	// (no transform needed - see optix_intersection_sphere.h's own normal
	// computation), matching what OPTIX_DENOISER_MODEL_KIND_AOV expects.
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

	res = optixDenoiserInvoke(denoiser_, stream_, &params,
		denoiserState_, denoiserStateSizeInBytes_,
		&guideLayer, &layer, 1,
		0, 0,
		denoiserScratch_, denoiserScratchSizeInBytes_);
	if (res != OPTIX_SUCCESS) return fail("invoke", res);

	if (cudaStreamSynchronize(stream_) != cudaSuccess)
		return fail("sync", OPTIX_ERROR_INTERNAL_ERROR);

	return true;
}

void OptiXRenderer::destroyDenoiser() noexcept {
	if (denoiserIntensity_) { cudaFree(reinterpret_cast<void*>(denoiserIntensity_)); denoiserIntensity_ = 0; }
	if (denoiserScratch_)   { cudaFree(reinterpret_cast<void*>(denoiserScratch_));   denoiserScratch_ = 0; }
	if (denoiserState_)     { cudaFree(reinterpret_cast<void*>(denoiserState_));     denoiserState_ = 0; }
	if (denoiser_)          { optixDenoiserDestroy(denoiser_); denoiser_ = nullptr; }
	denoiserWidth_ = 0;
	denoiserHeight_ = 0;
}

void OptiXRenderer::ensureAovBuffers(unsigned int width, unsigned int height) {
	if (d_albedoAov_ && d_normalAov_ && width == aovWidth_ && height == aovHeight_) {
		return;  // already sized correctly - video mode's steady-state per-frame call
	}
	destroyAovBuffers();
	size_t fbSize = static_cast<size_t>(width) * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_albedoAov_), fbSize));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_normalAov_), fbSize));
	aovWidth_ = width;
	aovHeight_ = height;
}

void OptiXRenderer::destroyAovBuffers() noexcept {
	if (d_albedoAov_) { cudaFree(reinterpret_cast<void*>(d_albedoAov_)); d_albedoAov_ = 0; }
	if (d_normalAov_) { cudaFree(reinterpret_cast<void*>(d_normalAov_)); d_normalAov_ = 0; }
	aovWidth_ = 0;
	aovHeight_ = 0;
}

bool OptiXRenderer::readAovBuffers(unsigned int width, unsigned int height,
		std::vector<float>& albedoOut, std::vector<float>& normalOut) const {
	if (!d_albedoAov_ || !d_normalAov_ || width != aovWidth_ || height != aovHeight_)
		return false;
	const size_t count = static_cast<size_t>(width) * height * 3;
	albedoOut.resize(count);
	normalOut.resize(count);
	CUDA_CHECK(cudaMemcpy(albedoOut.data(), reinterpret_cast<void*>(d_albedoAov_),
		count * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(normalOut.data(), reinterpret_cast<void*>(d_normalAov_),
		count * sizeof(float), cudaMemcpyDeviceToHost));
	return true;
}

void OptiXRenderer::cleanup() noexcept {
	// Must happen before context_ is destroyed below: wavefrontTracer_ and
	// sppmTracer_ each own their own OptiX program groups/pipelines/module,
	// all created from context_. Without this, their implicit member
	// destructors run after this function returns - i.e. after
	// optixDeviceContextDestroy below - and try to destroy OptiX objects
	// belonging to an already-destroyed context (use-after-free, reliably
	// crashes with an access violation on process exit once that mode
	// actually gets far enough to allocate anything). sppmTracer_ was
	// missing from this list entirely until GPU SPPM sub-phase 1f's own
	// verification test caught it under compute-sanitizer (CUDA_ERROR_
	// INVALID_CONTEXT inside cuCtxDestroy, called from SPPMPathTracer's
	// destructor after context_/cudaContext_ were already torn down) -
	// wavefrontTracer_ alone had never exercised this path in a way that
	// happened to crash visibly before.
	wavefrontTracer_.reset();
	sppmTracer_.reset();

	// Same before-context_-destruction ordering requirement as above -
	// denoiser_ is an OptiX object created from context_.
	destroyDenoiser();
	destroyAovBuffers();

	// Free SBT records
	if (d_raygenRecord_) cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
	if (d_missRecord_) cudaFree(reinterpret_cast<void*>(d_missRecord_));
	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));

	// Free acceleration structures (top-level IAS + the two child GASes +
	// one GAS per instance definition)
	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));
	if (d_gasCustom_) cudaFree(reinterpret_cast<void*>(d_gasCustom_));
	if (d_gasTri_) cudaFree(reinterpret_cast<void*>(d_gasTri_));
	if (d_gasDiskCylinder_) cudaFree(reinterpret_cast<void*>(d_gasDiskCylinder_));
	for (CUdeviceptr p : d_gasGroupTri_) if (p) cudaFree(reinterpret_cast<void*>(p));
	for (CUdeviceptr p : d_gasGroupSphere_) if (p) cudaFree(reinterpret_cast<void*>(p));
	if (d_instanceBase_) cudaFree(reinterpret_cast<void*>(d_instanceBase_));

	// Free scene data
	if (d_materials_) cudaFree(reinterpret_cast<void*>(d_materials_));
	if (d_textures_) cudaFree(reinterpret_cast<void*>(d_textures_));
	if (d_texturePixels_) cudaFree(reinterpret_cast<void*>(d_texturePixels_));
	if (d_spheres_) cudaFree(reinterpret_cast<void*>(d_spheres_));
	if (d_quads_) cudaFree(reinterpret_cast<void*>(d_quads_));
	if (d_bilinearPatches_) cudaFree(reinterpret_cast<void*>(d_bilinearPatches_));
	if (d_disks_) cudaFree(reinterpret_cast<void*>(d_disks_));
	if (d_cylinders_) cudaFree(reinterpret_cast<void*>(d_cylinders_));
	if (d_triangles_) cudaFree(reinterpret_cast<void*>(d_triangles_));
	if (d_lensElements_) cudaFree(reinterpret_cast<void*>(d_lensElements_));
	if (d_exitPupilBounds_) cudaFree(reinterpret_cast<void*>(d_exitPupilBounds_));
	if (d_cloudMediums_) cudaFree(reinterpret_cast<void*>(d_cloudMediums_));
	if (d_rgbGridMediums_) cudaFree(reinterpret_cast<void*>(d_rgbGridMediums_));
	if (d_rgbGridData_) cudaFree(reinterpret_cast<void*>(d_rgbGridData_));
	if (d_bssrdfTables_) cudaFree(reinterpret_cast<void*>(d_bssrdfTables_));
	if (d_bssrdfRhoSamples_) cudaFree(reinterpret_cast<void*>(d_bssrdfRhoSamples_));
	if (d_bssrdfRadiusSamples_) cudaFree(reinterpret_cast<void*>(d_bssrdfRadiusSamples_));
	if (d_bssrdfProfile_) cudaFree(reinterpret_cast<void*>(d_bssrdfProfile_));
	if (d_bssrdfProfileCdf_) cudaFree(reinterpret_cast<void*>(d_bssrdfProfileCdf_));
	if (d_measuredTables_) cudaFree(reinterpret_cast<void*>(d_measuredTables_));
	if (d_measuredParamValues_) cudaFree(reinterpret_cast<void*>(d_measuredParamValues_));
	if (d_measuredData_) cudaFree(reinterpret_cast<void*>(d_measuredData_));
	if (d_measuredMcdf_) cudaFree(reinterpret_cast<void*>(d_measuredMcdf_));
	if (d_measuredCcdf_) cudaFree(reinterpret_cast<void*>(d_measuredCcdf_));
	if (d_skyImagePixels_) cudaFree(reinterpret_cast<void*>(d_skyImagePixels_));
	if (d_skyMarginalCdf_) cudaFree(reinterpret_cast<void*>(d_skyMarginalCdf_));
	if (d_skyMarginalFunc_) cudaFree(reinterpret_cast<void*>(d_skyMarginalFunc_));
	if (d_skyConditionalCdf_) cudaFree(reinterpret_cast<void*>(d_skyConditionalCdf_));
	if (d_skyConditionalFunc_) cudaFree(reinterpret_cast<void*>(d_skyConditionalFunc_));
	if (d_skyConditionalFuncInt_) cudaFree(reinterpret_cast<void*>(d_skyConditionalFuncInt_));
	if (d_lightIndices_) cudaFree(reinterpret_cast<void*>(d_lightIndices_));
	if (d_lightKinds_) cudaFree(reinterpret_cast<void*>(d_lightKinds_));
	if (d_aliasTable_) cudaFree(reinterpret_cast<void*>(d_aliasTable_));
	if (d_punctualLights_) cudaFree(reinterpret_cast<void*>(d_punctualLights_));

	// Free launch params
	if (d_launchParams_) cudaFree(reinterpret_cast<void*>(d_launchParams_));

	// Destroy program groups
	if (raygenPG_) optixProgramGroupDestroy(raygenPG_);
	if (missPG_) optixProgramGroupDestroy(missPG_);
	if (hitgroupSpherePG_) optixProgramGroupDestroy(hitgroupSpherePG_);
	if (hitgroupQuadPG_) optixProgramGroupDestroy(hitgroupQuadPG_);
	if (hitgroupBilinearPatchPG_) optixProgramGroupDestroy(hitgroupBilinearPatchPG_);
	if (hitgroupTrianglePG_) optixProgramGroupDestroy(hitgroupTrianglePG_);
	if (shadowHitgroupSpherePG_) optixProgramGroupDestroy(shadowHitgroupSpherePG_);
	if (shadowHitgroupQuadPG_) optixProgramGroupDestroy(shadowHitgroupQuadPG_);
	if (shadowHitgroupBilinearPatchPG_) optixProgramGroupDestroy(shadowHitgroupBilinearPatchPG_);
	if (shadowHitgroupTrianglePG_) optixProgramGroupDestroy(shadowHitgroupTrianglePG_);
	if (probeMissPG_) optixProgramGroupDestroy(probeMissPG_);
	if (probeHitgroupSpherePG_) optixProgramGroupDestroy(probeHitgroupSpherePG_);
	if (probeHitgroupQuadPG_) optixProgramGroupDestroy(probeHitgroupQuadPG_);
	if (probeHitgroupBilinearPatchPG_) optixProgramGroupDestroy(probeHitgroupBilinearPatchPG_);
	if (probeHitgroupTrianglePG_) optixProgramGroupDestroy(probeHitgroupTrianglePG_);

	// Destroy module and pipeline
	if (module_) optixModuleDestroy(module_);
	if (pipeline_) optixPipelineDestroy(pipeline_);

	// Destroy context
	if (context_) optixDeviceContextDestroy(context_);

	// Destroy CUDA resources
	if (stream_) cudaStreamDestroy(stream_);
	// cudaContext_ is a PRIMARY context (createContext()'s own
	// cuDevicePrimaryCtxRetain() call) -- cuCtxDestroy() is invalid for
	// that kind of handle (the driver rejects it: CUDA_ERROR_INVALID_
	// CONTEXT, "Cannot destroy primary context", caught via
	// compute-sanitizer -- it failed silently here before since the
	// CUresult was never checked and nothing ran afterward to trip over
	// the resulting state). The correct release call is
	// cuDevicePrimaryCtxRelease(device), which decrements the refcount
	// cuDevicePrimaryCtxRetain() incremented instead of trying to destroy
	// the context object outright.
	if (cudaContext_) cuDevicePrimaryCtxRelease(cuDevice_);

	// Destroy wavefront tracer if it was created
	wavefrontTracer_.reset();
}

// ============================================================================
// enableWavefront � create/configure the WavefrontPathTracer on first call
// ============================================================================
void OptiXRenderer::enableWavefront(bool enable, const std::string& ptxPath) {
	useWavefront_ = enable;
	if (!enable) return;

	if (!wavefrontTracer_) {
		wavefrontTracer_ = std::make_unique<optix_renderer::WavefrontPathTracer>();
		if (!ptxPath.empty()) wavefrontTracer_->setPTXPath(ptxPath);

		if (!wavefrontTracer_->initialize(context_, module_, stream_)) {
			std::cerr << "[OptiXRenderer] Failed to initialize WavefrontPathTracer � falling back to recursive\n";
			wavefrontTracer_.reset();
			useWavefront_ = false;
			return;
		}
		if (!wavefrontTracer_->createProgramGroups()) {
			std::cerr << "[OptiXRenderer] WavefrontPathTracer::createProgramGroups failed\n";
			wavefrontTracer_.reset();
			useWavefront_ = false;
			return;
		}
		if (!wavefrontTracer_->linkPipeline(4)) {
			std::cerr << "[OptiXRenderer] WavefrontPathTracer::linkPipeline failed\n";
			wavefrontTracer_.reset();
			useWavefront_ = false;
			return;
		}
		std::cout << "[OptiXRenderer] WavefrontPathTracer ready\n";
	}

	// OUTSIDE the first-time block on purpose: the SBT describes the CURRENT
	// scene's geometry, and this object outlives any one scene (it is created
	// once and reused as scenes are switched). Building it only on creation
	// left a scene-A SBT in place for a scene-B render.
	//
	// That used to be survivable, because the pipeline claimed ALLOW_SINGLE_GAS
	// and OptiX then ignored the per-instance sbtOffsets entirely. With real
	// instancing traversal those offsets are honoured, so a stale SBT sends a
	// hit to whatever record happens to sit at that index - and if none does,
	// to no valid program at all: CUDA reports "invalid program counter" (718)
	// and the device context dies, taking every later test in the process with
	// it. Found exactly that way, as WavefrontRenderTest failing only when run
	// after other scenes had been rendered, never in isolation.
	//
	// SCENE-only counts, not the combined ones: numSpheres_/numTriangles_
	// include the instance definitions' geometry, and feeding those here would
	// make the SBT claim a scene-level record for a type the scene itself has
	// none of - shifting every later record. The instanced pairs are appended
	// separately from the flags, exactly as OptiXRenderer::buildSBT() does it.
	wavefrontTracer_->setInstancedGeometryFlags(!instanceTriangles_.empty(),
												!instanceSpheres_.empty());
	if (!wavefrontTracer_->buildSBT(sceneSphereCount_, numQuads_,
									numBilinearPatches_, sceneTriangleCount_,
									numDisks_, numCylinders_)) {
		std::cerr << "[OptiXRenderer] WavefrontPathTracer::buildSBT failed\n";
		wavefrontTracer_.reset();
		useWavefront_ = false;
		return;
	}
}

// ============================================================================
// renderSPPMTrivial — Phase 1a smoke test (see optix_renderer.h's own doc
// comment). Lazily creates sppmTracer_ the same way enableWavefront() does
// for wavefrontTracer_.
// ============================================================================
bool OptiXRenderer::ensureSPPMTracer(const std::string& ptxPath) {
	if (sppmTracer_) return true;

	sppmTracer_ = std::make_unique<optix_renderer::SPPMPathTracer>();
	if (!ptxPath.empty()) sppmTracer_->setPTXPath(ptxPath);

	if (!sppmTracer_->initialize(context_, stream_)) {
		std::cerr << "[OptiXRenderer] Failed to initialize SPPMPathTracer\n";
		sppmTracer_.reset();
		return false;
	}
	if (!sppmTracer_->createProgramGroups()) {
		std::cerr << "[OptiXRenderer] SPPMPathTracer::createProgramGroups failed\n";
		sppmTracer_.reset();
		return false;
	}
	if (!sppmTracer_->linkPipeline()) {
		std::cerr << "[OptiXRenderer] SPPMPathTracer::linkPipeline failed\n";
		sppmTracer_.reset();
		return false;
	}
	if (!sppmTracer_->buildSBT(numSpheres_, numQuads_)) {
		std::cerr << "[OptiXRenderer] SPPMPathTracer::buildSBT failed\n";
		sppmTracer_.reset();
		return false;
	}
	std::cout << "[OptiXRenderer] SPPMPathTracer ready\n";
	return true;
}

bool OptiXRenderer::renderSPPMTrivial(unsigned int width, unsigned int height,
                                       const GpuCameraParams& camera, float* outputFramebuffer,
                                       unsigned int maxDepth, const std::string& ptxPath) {
	if (!ensureSPPMTracer(ptxPath)) return false;

	return sppmTracer_->renderTrivial(
		static_cast<int>(width), static_cast<int>(height), camera, outputFramebuffer,
		gasHandle_, d_materials_, d_spheres_, d_quads_,
		numMaterials_, numSpheres_, numQuads_,
		d_lightIndices_, d_lightKinds_, d_aliasTable_, numLights_,
		maxDepth);
}

bool OptiXRenderer::renderSPPM(unsigned int width, unsigned int height,
                                int nIterations, int nPhotons, unsigned int maxDepth, float initialRadius,
                                const GpuCameraParams& camera, float* outputFramebuffer,
                                const std::string& ptxPath) {
	if (!ensureSPPMTracer(ptxPath)) return false;

	return sppmTracer_->render(
		static_cast<int>(width), static_cast<int>(height), nIterations, nPhotons,
		static_cast<int>(maxDepth), initialRadius, camera, outputFramebuffer,
		gasHandle_, d_materials_, d_spheres_, d_quads_,
		numMaterials_, numSpheres_, numQuads_,
		d_lightIndices_, d_lightKinds_, d_aliasTable_, numLights_);
}

