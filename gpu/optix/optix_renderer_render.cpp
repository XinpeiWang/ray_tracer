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
#include <cstdlib>

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

	// Object (per-primitive sphere) motion blur - same sceneHasMotion_
	// auto-detection buildScene() already uses for params.motionBlurEnabled
	// (the recursive backend's own flag, set below), just also mirrored onto
	// this shared GpuCameraParams so the wavefront backend's plain-CUDA-
	// kernel programs (which never see a `LaunchParams params` global) can
	// read it too - see GpuCameraParams::motionBlurEnabled's own comment.
	gpuCam.motionBlurEnabled = sceneHasMotion_ ? 1 : 0;

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

	// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's own
	// comment (optix_types.h). Same "device pointers patched in fresh here"
	// lifecycle as skyDist just above; mutually exclusive with it (matches
	// CPU) - portalHeight_ <= 0 leaves gpuCam.portalLight zero-init'd, which
	// every call site on both GPU backends already treats as "fall through
	// to the sky/flat-colour path".
	if (portalHeight_ > 0) {
		gpuCam.portalLight.width = portalWidth_;
		gpuCam.portalLight.height = portalHeight_;
		gpuCam.portalLight.scale = portalScale_;
		gpuCam.portalLight.frameX = portalFrameX_;
		gpuCam.portalLight.frameY = portalFrameY_;
		gpuCam.portalLight.frameZ = portalFrameZ_;
		gpuCam.portalLight.p0 = portalP0_;
		gpuCam.portalLight.p2 = portalP2_;
		gpuCam.portalLight.rectifiedImage = reinterpret_cast<const float*>(d_portalRectifiedImage_);
		gpuCam.portalLight.distFunc = reinterpret_cast<const float*>(d_portalDistFunc_);
		gpuCam.portalLight.satSum = reinterpret_cast<const double*>(d_portalSatSum_);
	}

	// Delegate to WavefrontPathTracer if enabled
	if (useWavefront_ && wavefrontTracer_) {
		// Set per render rather than once at enable time, so a scene switch
		// cannot leave a previous scene's table wired in - same reasoning
		// as this backend's own enableDenoise()'s comment. WavefrontPathTracer
		// has its own, independent denoiser/AOV-buffer implementation (see
		// that class's own setDenoiseEnabled() comment for why it's not
		// shared with this class's denoise()).
		wavefrontTracer_->setDenoiseEnabled(denoiseEnabled_);
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
		d_albedo = denoiserResources_.albedoAov;
		d_normal = denoiserResources_.normalAov;
	}

	// --stats device counters (see optix_types.h's LaunchParams::
	// statsBounceRays/statsShadowRays own comment) - same "null unless
	// requested, transient per-call alloc" shape as d_albedo/d_normal above,
	// except these are cheap enough (8 bytes each) not to bother persisting
	// across calls the way the AOV buffers do. Same RAY_TRACER_STATS env-var
	// pattern main.cpp/wavefront_path_tracer.cpp already use.
#pragma warning(suppress: 4996)
	const bool statsEnabled = (std::getenv("RAY_TRACER_STATS") != nullptr);
	CUdeviceptr d_statsBounceRays = 0;
	CUdeviceptr d_statsShadowRays = 0;
	if (statsEnabled) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_statsBounceRays), sizeof(unsigned long long)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_statsShadowRays), sizeof(unsigned long long)));
		CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_statsBounceRays), 0, sizeof(unsigned long long)));
		CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_statsShadowRays), 0, sizeof(unsigned long long)));
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

	// --stats device counters - null unless statsEnabled (see alloc above).
	params.statsBounceRays = reinterpret_cast<unsigned long long*>(d_statsBounceRays);
	params.statsShadowRays = reinterpret_cast<unsigned long long*>(d_statsShadowRays);

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

	// [REC-STATS] - recursive-backend counterpart to wavefront's own
	// "[WF-STATS]" block (wavefront_path_tracer.cpp), same RAY_TRACER_STATS
	// gate. Only two counters exist yet (bounce rays, shadow rays) - see
	// optix_types.h's LaunchParams::statsBounceRays/statsShadowRays comment
	// for why a per-hit-type breakdown like wavefront's isn't available
	// here without real per-hit-type device counters, a separate follow-up.
	if (statsEnabled) {
		unsigned long long bounceRays = 0, shadowRays = 0;
		CUDA_CHECK(cudaMemcpy(&bounceRays, reinterpret_cast<void*>(d_statsBounceRays),
			sizeof(unsigned long long), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&shadowRays, reinterpret_cast<void*>(d_statsShadowRays),
			sizeof(unsigned long long), cudaMemcpyDeviceToHost));
		const unsigned long long primaryRays = (unsigned long long)width * height * samplesPerPixel;
		std::cout << "[REC-STATS] ── Recursive GPU Render Statistics ───────────\n";
		std::cout << "[REC-STATS] Primary rays            : " << primaryRays << "\n";
		std::cout << "[REC-STATS] Total rays (incl. bounces): " << bounceRays << "\n";
		std::cout << "[REC-STATS] Shadow rays (NEE)        : " << shadowRays << "\n";
		std::cout << "[REC-STATS] ─────────────────────────────────────────────\n";
	}
	if (d_statsBounceRays) cudaFree(reinterpret_cast<void*>(d_statsBounceRays));
	if (d_statsShadowRays) cudaFree(reinterpret_cast<void*>(d_statsShadowRays));

	return true;
}

// ============================================================================
// denoise -- OptiX AI denoiser post-process (recursive backend)
// ============================================================================
// See optix_renderer.h's enableDenoise()/denoise() comments for scope. The
// actual lifecycle lives in optix_denoiser.h's runDenoiser()/
// destroyDenoiserResources()/ensureAovBuffers()/destroyAovBuffers(), shared
// byte-for-byte with WavefrontPathTracer (wavefront_path_tracer.cpp) - these
// are thin wrappers passing this backend's own denoiserResources_/context_/
// stream_.
bool OptiXRenderer::denoise(CUdeviceptr d_buffer, unsigned int width, unsigned int height,
	CUdeviceptr d_albedo, CUdeviceptr d_normal) {
	return runDenoiser(denoiserResources_, context_, stream_, d_buffer, width, height,
		d_albedo, d_normal, "[OptiX]");
}

void OptiXRenderer::destroyDenoiser() noexcept {
	destroyDenoiserResources(denoiserResources_);
}

void OptiXRenderer::ensureAovBuffers(unsigned int width, unsigned int height) {
	::ensureAovBuffers(denoiserResources_, width, height);
}

void OptiXRenderer::destroyAovBuffers() noexcept {
	::destroyAovBuffers(denoiserResources_);
}

bool OptiXRenderer::readAovBuffers(unsigned int width, unsigned int height,
		std::vector<float>& albedoOut, std::vector<float>& normalOut) const {
	if (!denoiserResources_.albedoAov || !denoiserResources_.normalAov ||
		width != denoiserResources_.aovWidth || height != denoiserResources_.aovHeight)
		return false;
	const size_t count = static_cast<size_t>(width) * height * 3;
	albedoOut.resize(count);
	normalOut.resize(count);
	CUDA_CHECK(cudaMemcpy(albedoOut.data(), reinterpret_cast<void*>(denoiserResources_.albedoAov),
		count * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(normalOut.data(), reinterpret_cast<void*>(denoiserResources_.normalAov),
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
	// denoiserResources_.denoiser is an OptiX object created from context_.
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

