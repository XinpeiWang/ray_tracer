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
//       5. optixLaunch(shadowPipeline)     -- fills occluded[] bool array
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
#include <cuda.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>

// Forward declarations for wavefront_launch.cu C wrappers (no <<<>>> in .cpp)
extern "C" void wf_launch_generate_camera_rays(
	WorkQueue<RayWorkItem>, int, int, int,
	GpuCameraParams, unsigned int, float*, cudaStream_t);
extern "C" void wf_launch_evaluate_materials(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>, WorkQueue<BssrdfProbeWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const TriangleData*, unsigned int,
	const BilinearPatchData*, unsigned int,
	const DiskData*, unsigned int,
	const CylinderData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const GpuLightKind*, const GpuAliasEntry*, unsigned int,
	const PunctualLightGPU*, unsigned int,
	const TextureData*, const unsigned char*,
	int,
	const CloudMedium<float>*, unsigned int,
	const GpuRgbGridMedium*, const float*,
	const GpuGridMedium*, const float*,
	const GpuMeasuredTable*, unsigned int,
	const float*, const float*, const float*, const float*,
	float3, float, GpuSkyDistribution,
	cudaStream_t);
extern "C" void wf_launch_evaluate_materials_simple(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const TriangleData*, unsigned int,
	const BilinearPatchData*, unsigned int,
	const DiskData*, unsigned int,
	const CylinderData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const GpuLightKind*, const GpuAliasEntry*, unsigned int,
	const PunctualLightGPU*, unsigned int,
	const TextureData*, const unsigned char*,
	int,
	float3, float, GpuSkyDistribution,
	cudaStream_t);
extern "C" void wf_launch_evaluate_materials_dielectric(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const TriangleData*, unsigned int,
	const BilinearPatchData*, unsigned int,
	const DiskData*, unsigned int,
	const CylinderData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const GpuLightKind*, const GpuAliasEntry*, unsigned int,
	const PunctualLightGPU*, unsigned int,
	const TextureData*, const unsigned char*,
	int,
	float3, float, GpuSkyDistribution,
	cudaStream_t);
extern "C" void wf_launch_accumulate_miss(WorkQueue<MissWorkItem>, int, float3*, float3, GpuSkyDistribution, cudaStream_t);
extern "C" void wf_launch_accumulate_shadow(WorkQueue<ShadowRayWorkItem>, int, const bool*, float3*, cudaStream_t);
extern "C" void wf_launch_resolve_bssrdf_exit(
	WorkQueue<BssrdfExitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const TriangleData*, unsigned int,
	const BilinearPatchData*, unsigned int,
	const DiskData*, unsigned int,
	const CylinderData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const GpuLightKind*, const GpuAliasEntry*, unsigned int,
	const PunctualLightGPU*, unsigned int,
	const TextureData*, const unsigned char*,
	float3, float, GpuSkyDistribution,
	cudaStream_t);
extern "C" void wf_launch_normalize_framebuffer(unsigned int, const float*, float3*, cudaStream_t);
extern "C" void wf_reset_queue_counter(int*, cudaStream_t);
extern "C" void wf_upload_cie_tables(const float*, const float*, const float*, int);
extern "C" void wf_upload_srgb_table(const float*, const float*, int);
extern "C" void wf_upload_d65_table(const float*, int);

namespace optix_renderer {

// ============================================================================
// Constructor / Destructor
// ============================================================================

WavefrontPathTracer::WavefrontPathTracer() = default;

WavefrontPathTracer::~WavefrontPathTracer() {
	cleanup();
}

// ============================================================================
// initialize
// ============================================================================

bool WavefrontPathTracer::initialize(OptixDeviceContext context,
									  OptixModule module,
									  cudaStream_t stream) {
	context_ = context;
	// Not currently read anywhere in this class (createProgramGroups() uses
	// wfModule_ exclusively - see its comment for why cross-module reuse of
	// the recursive path's intersection programs doesn't work). Still stored,
	// since this parameter used to be silently dropped entirely (marked
	// /*ignored*/) which was the first of two bugs that kept
	// WavefrontPathTracer from ever initializing - keeping it wired up here
	// costs nothing and avoids re-introducing a dead/misleading parameter.
	module_  = module;
	stream_  = stream;

	// Own stream for launchEvaluateMaterialsSimple()'s kernel - see its
	// member comment in the header for why this needs to be separate from
	// stream_.
	CUDA_CHECK(cudaStreamCreate(&simpleMaterialStream_));
	// Own stream for launchEvaluateMaterialsDielectric()'s kernel - same
	// reasoning, see its member comment in the header.
	CUDA_CHECK(cudaStreamCreate(&dielectricMaterialStream_));

	// Must be true, not false: the traversable this pipeline traces against
	// is the SAME IAS/GAS OptiXRenderer::buildScene() builds for the
	// recursive backend, and that GAS gets motionOptions.numKeys=2 whenever
	// the scene has a moving sphere (see its sceneHasMotion_ detection) -
	// motion keys apply per accel-structure build, not per pipeline. Tracing
	// a motion-enabled traversable from a pipeline compiled with
	// usesMotionBlur=false is undefined behavior in OptiX; in practice it
	// made every primary ray report a miss on scene 8 (Final Scene, whose
	// moving sphere triggers sceneHasMotion_), rendering it solid black
	// while the recursive backend (usesMotionBlur=true, see its own
	// optix_renderer.cpp comment) rendered the same GAS correctly. Safe for
	// every scene either way, motion or not, by the same reasoning as that
	// comment: non-motion scenes keep a single-key GAS and this pipeline's
	// own optixTrace calls (wavefront_programs.cu) always pass rayTime=0.0f,
	// so optixGetRayTime() is a provable no-op for them.
	pipelineCompileOptions_.usesMotionBlur        = true;
	// SINGLE_LEVEL_INSTANCING, matching OptiXRenderer's own pipeline, because
	// the traversable handed to render() is the top-level IAS that
	// OptiXRenderer::buildScene() builds - never a bare GAS. This said
	// ALLOW_SINGLE_GAS, which happened not to misbehave visibly on this driver
	// but is the wrong contract for an IAS and makes optixGetInstanceId()
	// unusable - and that is exactly what object instancing needs to recover
	// which placement a hit came from.
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions_.numPayloadValues       = 2;  // pointer p0/p1
	pipelineCompileOptions_.numAttributeValues     = 4;
	// This is the actual fix for a CUDA 718 "invalid program counter" crash
	// that only reproduced after a specific combination of earlier GPU tests
	// (RenderIntegrationTest + SppmGpuFirstSliceTest + GPURenderTest +
	// GPUSceneSwitchTest, all four, in that relative order) ran in the same
	// process before the first wavefront launch -- never in isolation, and
	// not fixed by giving the pipelines a hand-picked stack size (tried and
	// failed) or one properly computed via optixUtilAccumulateStackSizes /
	// optixUtilComputeStackSizes (see linkPipeline() below -- also tried,
	// also failed). Adding compute-sanitizer memcheck found no invalid
	// device read/write before the crash, which pointed away from a plain
	// buffer overrun and toward the pipeline's own exception handling being
	// unconfigured (exceptionFlags was OPTIX_EXCEPTION_FLAG_NONE, so an
	// in-flight stack overflow or trace-depth violation had no defined
	// handler to divert to and instead corrupted whatever was live on the
	// device's continuation stack).
	//
	// Enabling STACK_OVERFLOW/TRACE_DEPTH here, plus registering the actual
	// __exception__wf_report exception program below and wiring it into
	// both SBTs' exceptionRecord (see createProgramGroups()/buildSBT()),
	// made the crash disappear completely across repeated full-suite runs --
	// even though the exception program itself was never observed to fire
	// (no "[WF-EXCEPTION]" line in any passing run's output). That means the
	// fix isn't "catch and handle the exception": it's that giving OptiX a
	// defined exception path changes how it manages the pipeline's
	// continuation stack even on the success path, closing whatever gap let
	// the corruption happen with OPTIX_EXCEPTION_FLAG_NONE. The exception
	// program stays registered as a real safety net (and a live diagnostic)
	// rather than being stripped back out now that it isn't reproducing.
	pipelineCompileOptions_.exceptionFlags         =
		OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
		OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
		OPTIX_EXCEPTION_FLAG_USER;
	pipelineCompileOptions_.pipelineLaunchParamsVariableName = "wf_params";

	if (!loadModule()) return false;

	// Upload CIE XYZ matching function tables to device __constant__ memory
	wf_upload_cie_tables(CIE_X, CIE_Y, CIE_Z, kCIENSamples);

	// Upload the CIE Standard Illuminant D65 SPD, resampled onto the same
	// 360-830nm/1nm grid as CIE_X/Y/Z and pre-normalised so that
	// InnerProduct(CIE_Y, D65) == kCIE_Y_integral (106.856895f) -- i.e. so a
	// (1,1,1) RGBIlluminantSpectrum reconstructs to XYZ Y=1 under the same
	// unweighted-CIE_Y_integral convention SampledSpectrumToXYZ uses for
	// every other spectral quantity in this pipeline.
	//
	// Root-cause context (found via tests/integration/
	// material_cpu_gpu_parity_tests.cpp's B1/RoughMetalSpheres finding):
	// every light-source RGB the wavefront kernel uplifts to a spectrum
	// (area lights, punctual lights, sky/background, direct-hit emissive
	// surfaces) was being treated as an RGBUnboundedSpectrum (scale *
	// rsp(lambda), no illuminant) instead of pbrt-v4's RGBIlluminantSpectrum
	// (scale * rsp(lambda) * illuminant(lambda)). Since dev_srgb_to_coeffs's
	// achromatic (r==g==b) branch produces a perfectly FLAT spectrum for any
	// grey light colour, omitting the D65 factor made every grey light in
	// the scene carry an equal-energy-illuminant chromaticity (0.333,0.333)
	// instead of D65's (0.3127,0.3290). wf_xyz_to_linear_rgb's matrix is
	// built for D65 white, so reconstructing an equal-energy spectrum
	// through it yields a non-neutral RGB (R inflated ~20%, G/B suppressed
	// ~5-11%) -- which then multiplies through onto everything that light
	// illuminates. Verified with a standalone deterministic Riemann-sum
	// integration (no Monte Carlo/rendering involved): multiplying by this
	// normalised D65 table reproduces CPU's naive per-channel RGB multiply
	// to within 0.05% for B1's gold-metal-under-grey-light case.
	{
		static float h_d65[kCIENSamples];
		const DenselySampledSpectrum& d65 = GetD65Illuminant();
		float normConst = 0.f;
		for (int i = 0; i < kCIENSamples; ++i) {
			float lambda = static_cast<float>(kCIELambda_min + i);
			float d65v = d65(lambda);
			h_d65[i] = d65v;
			normConst += CIE_Y[i] * d65v;
		}
		const float scale = kCIE_Y_integral / normConst;
		for (int i = 0; i < kCIENSamples; ++i) h_d65[i] *= scale;
		wf_upload_d65_table(h_d65, kCIENSamples);
	}

	// Upload sRGB upsampling table to device memory
	{
		const auto& tbl = RGBToSpectrumTable::sRGB();
		wf_upload_srgb_table(
			sRGBToSpectrumTable_Scale,
			&sRGBToSpectrumTable_Data[0][0][0][0][0],
			RGBToSpectrumTable::kRes);
	}

	// Allocate device launch params buffer
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wfLaunchParams_), sizeof(WavefrontLaunchParams)));

	std::cout << "[WavefrontPathTracer] Initialized\n";
	return true;
}

// ============================================================================
// loadModule — compile wavefront_programs.ptx into an OptiX module
// ============================================================================

bool WavefrontPathTracer::loadModule() {
	// Candidates in preference order, mirroring OptiXRenderer::loadPTX's own
	// search for optix_programs.ptx.
	//
	// Searching rather than trusting one path is not defensive padding: the
	// caller derives ptxPath_ from the OUTPUT IMAGE's directory, and nothing
	// in the build ever copies the PTX there. So the single-path version
	// always missed, printed one line, and fell back to the recursive tracer -
	// which renders a perfectly good image, just not with the renderer that
	// was asked for. Wavefront mode was effectively unreachable for anyone who
	// did not hand-copy the file, and said so only in passing.
	const char *kName = "wavefront_programs.ptx";
	std::string candidates[] = {
		ptxPath_,                          // caller's guess, if any
		std::string("gpu/optix/") + kName, // where the build actually writes it
		std::string("optix_output/") + kName,
		std::string("./") + kName,
	};

	std::string ptxSource;
	std::string loadedFrom;
	for (const std::string &path : candidates) {
		if (path.empty()) continue;
		std::ifstream file(path, std::ios::binary);
		if (!file.is_open()) continue;
		std::ostringstream oss;
		oss << file.rdbuf();
		ptxSource = oss.str();
		loadedFrom = path;
		break;
	}

	if (ptxSource.empty()) {
		// Loud, because the consequence is silent: the renderer carries on in
		// recursive mode and the image looks fine.
		std::cerr << "[WavefrontPathTracer] warning: could not find " << kName
				  << " in any of the searched locations; wavefront mode is NOT "
				     "active and this render will use the recursive tracer.\n";
		return false;
	}
	std::cout << "[WavefrontPathTracer] Loaded module from " << loadedFrom << "\n";

	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

	char   log[4096];
	size_t logSize = sizeof(log);

	OPTIX_CHECK(optixModuleCreate(
		context_,
		&moduleCompileOptions,
		&pipelineCompileOptions_,
		ptxSource.c_str(),
		ptxSource.size(),
		log,
		&logSize,
		&wfModule_
	));

	return true;
}

// ============================================================================
// createProgramGroups
// ============================================================================

bool WavefrontPathTracer::createProgramGroups() {
	char   log[2048];
	size_t logSize;
	OptixProgramGroupOptions pgOptions = {};

	// ----- Intersection pipeline -----

	// Raygen
	OptixProgramGroupDesc rgDesc = {};
	rgDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	rgDesc.raygen.module                 = wfModule_;
	rgDesc.raygen.entryFunctionName      = "__raygen__wf_intersect";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &rgDesc, 1, &pgOptions,
										 log, &logSize, &raygenIntersectPG_));

	// Miss (radiance)
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missDesc.miss.module              = wfModule_;
	missDesc.miss.entryFunctionName   = "__miss__wf_radiance";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &missDesc, 1, &pgOptions,
										 log, &logSize, &missRadiancePG_));

	// Sphere/quad closesthit paired with wavefront-native intersection programs
	// (wavefront_programs.cu __intersection__wf_sphere/__intersection__wf_quad).
	// These used to reuse __intersection__sphere/__intersection__quad from the
	// recursive path's module_ instead of having their own copy, which OptiX
	// rejects: combining programs from two modules compiled with different
	// pipelineCompileOptions.numPayloadValues into one hitgroup fails payload-
	// type resolution ("could not be resolved to a common payloadType"), even
	// though neither intersection program actually touches payload registers.
	// Compiling our own copies under wfModule_'s own pipeline options sidesteps
	// the cross-module mismatch entirely - see wavefront_programs.cu for detail.
	OptixProgramGroupDesc sphereHitDesc = {};
	sphereHitDesc.kind                              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	sphereHitDesc.hitgroup.moduleIS                 = wfModule_;
	sphereHitDesc.hitgroup.entryFunctionNameIS      = "__intersection__wf_sphere";
	sphereHitDesc.hitgroup.moduleCH                 = wfModule_;
	sphereHitDesc.hitgroup.entryFunctionNameCH      = "__closesthit__wf_sphere";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &sphereHitDesc, 1, &pgOptions,
										 log, &logSize, &hitSpherePG_));

	OptixProgramGroupDesc quadHitDesc = {};
	quadHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	quadHitDesc.hitgroup.moduleIS               = wfModule_;
	quadHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_quad";
	quadHitDesc.hitgroup.moduleCH               = wfModule_;
	quadHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_quad";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &quadHitDesc, 1, &pgOptions,
										 log, &logSize, &hitQuadPG_));

	OptixProgramGroupDesc blpHitDesc = {};
	blpHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	blpHitDesc.hitgroup.moduleIS               = wfModule_;
	blpHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_bilinear_patch";
	blpHitDesc.hitgroup.moduleCH               = wfModule_;
	blpHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_bilinear_patch";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &blpHitDesc, 1, &pgOptions,
										 log, &logSize, &hitBilinearPatchPG_));

	// Any-hit added for alpha-cutout (MaterialData::alphaMaskTexIdx) - see
	// __anyhit__wf_triangle's own comment (wavefront_programs.cu). Same hit
	// group/program group as before, no new SBT record type - a no-op for
	// the overwhelming majority of triangles.
	OptixProgramGroupDesc triHitDesc = {};
	triHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	triHitDesc.hitgroup.moduleIS               = wfModule_;
	triHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_triangle";
	triHitDesc.hitgroup.moduleCH               = wfModule_;
	triHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_triangle";
	triHitDesc.hitgroup.moduleAH               = wfModule_;
	triHitDesc.hitgroup.entryFunctionNameAH    = "__anyhit__wf_triangle";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &triHitDesc, 1, &pgOptions,
										 log, &logSize, &hitTrianglePG_));

	// Disk/Cylinder (Phase 4c) - appended last, after triangle, matching
	// this backend's own trailing-region SBT placement (see buildSBT()'s
	// own comment) and the recursive backend's diskCylinderSbtOffset.
	OptixProgramGroupDesc diskHitDesc = {};
	diskHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	diskHitDesc.hitgroup.moduleIS               = wfModule_;
	diskHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_disk";
	diskHitDesc.hitgroup.moduleCH               = wfModule_;
	diskHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_disk";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &diskHitDesc, 1, &pgOptions,
										 log, &logSize, &hitDiskPG_));

	OptixProgramGroupDesc cylinderHitDesc = {};
	cylinderHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	cylinderHitDesc.hitgroup.moduleIS               = wfModule_;
	cylinderHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_cylinder";
	cylinderHitDesc.hitgroup.moduleCH               = wfModule_;
	cylinderHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_cylinder";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &cylinderHitDesc, 1, &pgOptions,
										 log, &logSize, &hitCylinderPG_));

	// ----- BSSRDF probe walk (MaterialType::Subsurface, Phase 2) -----
	// Linked into the SAME intersect pipeline as the raygen/hit groups
	// above (see linkPipeline()'s intersectGroups[] array and
	// wavefront_probe.h's own header comment for why no new module/
	// pipeline is needed here).

	OptixProgramGroupDesc probeRgDesc = {};
	probeRgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	probeRgDesc.raygen.module            = wfModule_;
	probeRgDesc.raygen.entryFunctionName = "__raygen__wf_probe";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeRgDesc, 1, &pgOptions,
										 log, &logSize, &raygenProbePG_));

	OptixProgramGroupDesc probeMissDesc = {};
	probeMissDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
	probeMissDesc.miss.module            = wfModule_;
	probeMissDesc.miss.entryFunctionName = "__miss__wf_probe";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeMissDesc, 1, &pgOptions,
										 log, &logSize, &missProbePG_));

	OptixProgramGroupDesc probeSphereDesc = {};
	probeSphereDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeSphereDesc.hitgroup.moduleIS            = wfModule_;
	probeSphereDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_sphere";
	probeSphereDesc.hitgroup.moduleCH            = wfModule_;
	probeSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_sphere";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeSphereDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeSpherePG_));

	OptixProgramGroupDesc probeQuadDesc = {};
	probeQuadDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeQuadDesc.hitgroup.moduleIS            = wfModule_;
	probeQuadDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_quad";
	probeQuadDesc.hitgroup.moduleCH            = wfModule_;
	probeQuadDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_quad";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeQuadDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeQuadPG_));

	OptixProgramGroupDesc probeBlpDesc = {};
	probeBlpDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeBlpDesc.hitgroup.moduleIS            = wfModule_;
	probeBlpDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_bilinear_patch";
	probeBlpDesc.hitgroup.moduleCH            = wfModule_;
	probeBlpDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_bilinear_patch";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeBlpDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeBilinearPatchPG_));

	// Deliberately no any-hit (alpha-cutout) here - see wavefront_probe.h's
	// own header comment for the same documented simplification
	// optix_probe_hit.h's recursive-backend twin already makes.
	OptixProgramGroupDesc probeTriDesc = {};
	probeTriDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeTriDesc.hitgroup.moduleIS            = wfModule_;
	probeTriDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_triangle";
	probeTriDesc.hitgroup.moduleCH            = wfModule_;
	probeTriDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_triangle";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeTriDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeTrianglePG_));

	OptixProgramGroupDesc probeDiskDesc = {};
	probeDiskDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeDiskDesc.hitgroup.moduleIS            = wfModule_;
	probeDiskDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_disk";
	probeDiskDesc.hitgroup.moduleCH            = wfModule_;
	probeDiskDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_disk";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeDiskDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeDiskPG_));

	OptixProgramGroupDesc probeCylinderDesc = {};
	probeCylinderDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeCylinderDesc.hitgroup.moduleIS            = wfModule_;
	probeCylinderDesc.hitgroup.entryFunctionNameIS = "__intersection__wf_cylinder";
	probeCylinderDesc.hitgroup.moduleCH            = wfModule_;
	probeCylinderDesc.hitgroup.entryFunctionNameCH = "__closesthit__wf_probe_cylinder";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeCylinderDesc, 1, &pgOptions,
										 log, &logSize, &hitProbeCylinderPG_));

	// ----- Shadow pipeline -----

	OptixProgramGroupDesc shadowRGDesc = {};
	shadowRGDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	shadowRGDesc.raygen.module                = wfModule_;
	shadowRGDesc.raygen.entryFunctionName     = "__raygen__wf_shadow";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowRGDesc, 1, &pgOptions,
										 log, &logSize, &raygenShadowPG_));

	OptixProgramGroupDesc shadowMissDesc = {};
	shadowMissDesc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
	shadowMissDesc.miss.module               = wfModule_;
	shadowMissDesc.miss.entryFunctionName    = "__miss__wf_shadow";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowMissDesc, 1, &pgOptions,
										 log, &logSize, &missShadowPG_));

	// Shadow anyhit for sphere
	OptixProgramGroupDesc shadowSphereDesc = {};
	shadowSphereDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowSphereDesc.hitgroup.moduleIS             = wfModule_;
	shadowSphereDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_sphere";
	shadowSphereDesc.hitgroup.moduleAH             = wfModule_;
	shadowSphereDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_sphere";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowSphereDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowSpherePG_));

	// Shadow anyhit for quad
	OptixProgramGroupDesc shadowQuadDesc = {};
	shadowQuadDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowQuadDesc.hitgroup.moduleIS             = wfModule_;
	shadowQuadDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_quad";
	shadowQuadDesc.hitgroup.moduleAH             = wfModule_;
	shadowQuadDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_quad";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowQuadDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowQuadPG_));

	// Shadow anyhit for bilinear patch
	OptixProgramGroupDesc shadowBlpDesc = {};
	shadowBlpDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowBlpDesc.hitgroup.moduleIS             = wfModule_;
	shadowBlpDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_bilinear_patch";
	shadowBlpDesc.hitgroup.moduleAH             = wfModule_;
	shadowBlpDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_bilinear_patch";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowBlpDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowBilinearPatchPG_));

	// Shadow anyhit for triangle
	OptixProgramGroupDesc shadowTriDesc = {};
	shadowTriDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowTriDesc.hitgroup.moduleIS             = wfModule_;
	shadowTriDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_triangle";
	shadowTriDesc.hitgroup.moduleAH             = wfModule_;
	shadowTriDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_triangle";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowTriDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowTrianglePG_));

	// Shadow anyhit for disk/cylinder (Phase 4c)
	OptixProgramGroupDesc shadowDiskDesc = {};
	shadowDiskDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowDiskDesc.hitgroup.moduleIS             = wfModule_;
	shadowDiskDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_disk";
	shadowDiskDesc.hitgroup.moduleAH             = wfModule_;
	shadowDiskDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_disk";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowDiskDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowDiskPG_));

	OptixProgramGroupDesc shadowCylinderDesc = {};
	shadowCylinderDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowCylinderDesc.hitgroup.moduleIS             = wfModule_;
	shadowCylinderDesc.hitgroup.entryFunctionNameIS  = "__intersection__wf_cylinder";
	shadowCylinderDesc.hitgroup.moduleAH             = wfModule_;
	shadowCylinderDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow_cylinder";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowCylinderDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowCylinderPG_));

	// Exception program group -- see this file's pipelineCompileOptions_
	// .exceptionFlags comment in initialize() for why this exists and why
	// it's the real CUDA-718 fix, not just diagnostics.
	OptixProgramGroupDesc excDesc = {};
	excDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
	excDesc.exception.module            = wfModule_;
	excDesc.exception.entryFunctionName = "__exception__wf_report";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &excDesc, 1, &pgOptions,
										 log, &logSize, &exceptionPG_));

	std::cout << "[WavefrontPathTracer] Created 25 program groups\n";
	return true;
}

// ============================================================================
// linkPipeline — build two separate pipelines: intersect + shadow
// ============================================================================

bool WavefrontPathTracer::linkPipeline(unsigned int maxTraceDepth) {
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.maxTraceDepth = maxTraceDepth;

	char   log[2048];
	size_t logSize;

	// Intersect pipeline - also carries the BSSRDF probe-walk program groups
	// (raygenProbePG_ etc.), launched via the SAME pipeline object with a
	// separate probeSBT_ (see wavefront_probe.h's own header comment and
	// buildSBT()'s probeSBT_ for why no separate pipeline is needed).
	OptixProgramGroup intersectGroups[] = {
		raygenIntersectPG_,
		missRadiancePG_,
		hitSpherePG_,
		hitQuadPG_,
		hitBilinearPatchPG_,
		hitDiskPG_,
		hitCylinderPG_,
		hitTrianglePG_,
		raygenProbePG_,
		missProbePG_,
		hitProbeSpherePG_,
		hitProbeQuadPG_,
		hitProbeBilinearPatchPG_,
		hitProbeDiskPG_,
		hitProbeCylinderPG_,
		hitProbeTrianglePG_,
		exceptionPG_
	};
	constexpr size_t kNumIntersectGroups = sizeof(intersectGroups) / sizeof(intersectGroups[0]);
	{
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &linkOptions,
			intersectGroups, (unsigned int)kNumIntersectGroups, log, &logSize, &intersectPipeline_));
		std::cout << "[WavefrontPathTracer] Linked intersect pipeline\n";
	}

	// Shadow pipeline (depth 1 — just any-hit)
	OptixProgramGroup shadowGroups[] = {
		raygenShadowPG_,
		missShadowPG_,
		anyhitShadowSpherePG_,
		anyhitShadowQuadPG_,
		anyhitShadowBilinearPatchPG_,
		anyhitShadowDiskPG_,
		anyhitShadowCylinderPG_,
		anyhitShadowTrianglePG_,
		exceptionPG_
	};
	constexpr size_t kNumShadowGroups = sizeof(shadowGroups) / sizeof(shadowGroups[0]);
	OptixPipelineLinkOptions shadowLinkOptions = {};
	shadowLinkOptions.maxTraceDepth = 1;
	{
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &shadowLinkOptions,
			shadowGroups, (unsigned int)kNumShadowGroups, log, &logSize, &shadowPipeline_));
		std::cout << "[WavefrontPathTracer] Linked shadow pipeline\n";
	}

	// Stack size, computed per pipeline via optixUtilAccumulateStackSizes +
	// optixUtilComputeStackSizes -- the same two calls
	// OptiXRenderer::linkPipeline() uses for the recursive pipeline -- rather
	// than a hand-picked literal.
	//
	// Neither pipeline set a stack size at all originally, which left the
	// depth at its default of 1. That was survivable while the pipeline
	// claimed ALLOW_SINGLE_GAS, since traversal never descended through an
	// instance; once it does, a depth-1 stack sends it somewhere undefined
	// and CUDA reports "invalid program counter" (718), which kills the
	// whole device context. A hand-picked continuationStackSize=4096 with
	// maxTraversableGraphDepth=2 was tried next and did NOT fix the crash
	// (see this file's git history) -- it fixed the graph depth but was
	// still a guessed byte count, not one actually derived from these
	// programs' real CSS/DSS via the OptiX stack-size utilities below.
	auto setComputedStackSize = [&](OptixPipeline p, OptixProgramGroup* groups, size_t n,
	                                 unsigned int maxTraceDepthForPipeline) {
		if (!p) return;
		OptixStackSizes stackSizes = {};
		for (size_t i = 0; i < n; ++i) {
			OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &stackSizes, p));
		}
		uint32_t directCallableStackSizeFromTraversal;
		uint32_t directCallableStackSizeFromState;
		uint32_t continuationStackSize;
		OPTIX_CHECK(optixUtilComputeStackSizes(
			&stackSizes,
			maxTraceDepthForPipeline,
			/*maxCCDepth*/ 0,
			/*maxDCDepth*/ 0,
			&directCallableStackSizeFromTraversal,
			&directCallableStackSizeFromState,
			&continuationStackSize));
		OPTIX_CHECK(optixPipelineSetStackSize(
			p,
			directCallableStackSizeFromTraversal,
			directCallableStackSizeFromState,
			continuationStackSize,
			/*maxTraversableGraphDepth*/ 2));  // IAS -> GAS, single-level instancing
	};
	setComputedStackSize(intersectPipeline_, intersectGroups, kNumIntersectGroups, maxTraceDepth);
	setComputedStackSize(shadowPipeline_, shadowGroups, kNumShadowGroups, 1);

	return true;
}

// ============================================================================
// buildSBT — create two separate SBTs
// ============================================================================

bool WavefrontPathTracer::buildSBT(unsigned int numSpheres, unsigned int numQuads, unsigned int numBilinearPatches, unsigned int numTriangles,
									unsigned int numDisks, unsigned int numCylinders) {
	// haveInstanced* come from setInstancedGeometryFlags(); see its comment.
	const bool haveInstTri = haveInstancedTriangles_;
	const bool haveInstSph = haveInstancedSpheres_;
	numSpheres_ = numSpheres;
	numQuads_   = numQuads;
	numBilinearPatches_ = numBilinearPatches;
	numTriangles_ = numTriangles;
	numDisks_ = numDisks;
	numCylinders_ = numCylinders;

	destroySBT();

	// Matches OptiXRenderer::buildScene()'s conditional build-input inclusion
	// (see its comment): OptiX rejects a zero-primitive custom-primitive build
	// input outright, so buildScene() OMITS empty geometry types from the
	// shared GAS entirely rather than keeping a 0-count placeholder - meaning
	// a type's position among the *present* types (not its fixed geometry-
	// type index) determines its SBT slot. Both SBTs below must apply the
	// exact same [sphere, quad, bilinear patch, triangle] present/absent filter.
	const bool hasSpheres = numSpheres > 0;
	const bool hasQuads   = numQuads > 0;
	const bool hasBlp     = numBilinearPatches > 0;
	const bool hasTri     = numTriangles > 0;
	// Disk/Cylinder (Phase 4c) - appended AFTER the instanced-geometry
	// records in all three SBTs below, matching OptiXRenderer::buildScene()'s
	// own diskCylinderSbtOffset placement (its own dedicated GAS/IAS
	// instance, added last - see that function's comment) so this backend's
	// hit records land at the same absolute SBT position the shared IAS's
	// baked instance.sbtOffset expects.
	const bool hasDisks     = numDisks > 0;
	const bool hasCylinders = numCylinders > 0;

	// Every present type-group below gets RAY_TYPE_COUNT (3) identical
	// records, not 2. This is NOT about the recursive backend's real
	// [radiance, shadow, probe] triple - this backend uses three entirely
	// separate SBTs (Intersect/Probe/Shadow below), each single-purpose, so
	// one real record per type would suffice on its own. It has to be 3
	// anyway: this backend's hit-group records live in the SAME shared IAS
	// (gasHandle_) the recursive backend builds, and OptiX resolves a hit
	// record as `instance.sbtOffset + build_input_index * sbtStride +
	// traceOffset`. The instance.sbtOffset values are baked once, by the
	// recursive backend, using ITS stride (RAY_TYPE_COUNT=3) to account for
	// every type-group that precedes a given GAS. If this backend's own
	// array used a different per-type record count (2, as a prior version
	// of this function did), its cumulative offsets would only agree with
	// the shared baked values when at most one type-group precedes the
	// affected one - a discrepancy of `3*N - 2*N = N` records opens up for
	// N>=2 preceding groups, silently landing on a DIFFERENT type's program
	// group or past the end of the array (confirmed: a scene with a sphere
	// AND a triangle mesh present ahead of a disk/cylinder pair reads two
	// records past where the cylinder's own pair actually starts). Using
	// RAY_TYPE_COUNT here too makes this backend's cumulative offsets
	// numerically IDENTICAL to the shared baked ones for any N, not just
	// N<=1 - the two extra records per type are pure padding (all three
	// slots of a type's block point at the same program group), the price
	// of staying on the shared IAS instead of building a second one.
	const auto pushTriple = [](std::vector<HitGroupRecord>& recs, OptixProgramGroup pg) {
		for (int i = 0; i < RAY_TYPE_COUNT; ++i) {
			recs.emplace_back();
			OPTIX_CHECK(optixSbtRecordPackHeader(pg, &recs.back()));
			recs.back().data = {};
		}
	};

	// ---- Intersect SBT ----
	{
		// Raygen record
		RaygenRecord rg;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenIntersectPG_, &rg));
		rg.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intersectRaygenRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_intersectRaygenRecord_), &rg,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));

		// Miss record (radiance only)
		MissRecord missRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missRadiancePG_, &missRec));
		missRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intersectMissRecord_), sizeof(MissRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_intersectMissRecord_), &missRec,
							  sizeof(MissRecord), cudaMemcpyHostToDevice));

		// Hit records: SBT offset=0, stride=RAY_TYPE_COUNT means each PRESENT
		// type (in [sphere, quad, bilinear patch] order, absent types
		// omitted) needs RAY_TYPE_COUNT identical records - see the
		// hasSpheres/hasQuads/hasBlp comment above for why position among
		// present types, not fixed geometry-type index, determines the slot,
		// and the pushTriple comment above for why RAY_TYPE_COUNT records
		// rather than 1.
		std::vector<HitGroupRecord> hitRecs;
		if (hasSpheres) pushTriple(hitRecs, hitSpherePG_);
		if (hasQuads)   pushTriple(hitRecs, hitQuadPG_);
		if (hasBlp)     pushTriple(hitRecs, hitBilinearPatchPG_);
		if (hasTri)     pushTriple(hitRecs, hitTrianglePG_);
		// Instanced geometry's own records, appended after the scene's
		// packed region in the same order OptiXRenderer::buildSBT() uses -
		// see setInstancedGeometryFlags().
		if (haveInstTri) pushTriple(hitRecs, hitTrianglePG_);
		if (haveInstSph) pushTriple(hitRecs, hitSpherePG_);
		if (hasDisks)     pushTriple(hitRecs, hitDiskPG_);
		if (hasCylinders) pushTriple(hitRecs, hitCylinderPG_);

		size_t sz = hitRecs.size() * sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intersectHitRecords_), sz));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_intersectHitRecords_), hitRecs.data(),
							  sz, cudaMemcpyHostToDevice));

		intersectSBT_.raygenRecord                = d_intersectRaygenRecord_;
		intersectSBT_.missRecordBase              = d_intersectMissRecord_;
		intersectSBT_.missRecordStrideInBytes     = sizeof(MissRecord);
		intersectSBT_.missRecordCount             = 1;
		intersectSBT_.hitgroupRecordBase          = d_intersectHitRecords_;
		intersectSBT_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
		intersectSBT_.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());

		// Exception record -- see the CUDA-718 fix comment in initialize().
		RaygenRecord excRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(exceptionPG_, &excRec));
		excRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intersectExceptionRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_intersectExceptionRecord_), &excRec,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));
		intersectSBT_.exceptionRecord = d_intersectExceptionRecord_;
	}

	// ---- Probe SBT (BSSRDF probe walk, Phase 2) ----
	// Mirrors the Intersect SBT's own layout EXACTLY (same present-type
	// ordering, same stride=RAY_TYPE_COUNT padding, same instanced-geometry
	// append order) so the shared IAS instances' baked sbtOffset values
	// resolve correctly against this SBT too, at the SAME trace-time
	// SBTOffset=0/SBTStride=RAY_TYPE_COUNT wf_trace_probe_ray()
	// (wavefront_probe.h) already passes - see that function's own comment.
	// Uses this same pipeline (intersectPipeline_), just a different
	// SBT/program groups.
	{
		RaygenRecord rg;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenProbePG_, &rg));
		rg.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeRaygenRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeRaygenRecord_), &rg,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));

		MissRecord missRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missProbePG_, &missRec));
		missRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeMissRecord_), sizeof(MissRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeMissRecord_), &missRec,
							  sizeof(MissRecord), cudaMemcpyHostToDevice));

		std::vector<HitGroupRecord> hitRecs;
		if (hasSpheres)   pushTriple(hitRecs, hitProbeSpherePG_);
		if (hasQuads)     pushTriple(hitRecs, hitProbeQuadPG_);
		if (hasBlp)       pushTriple(hitRecs, hitProbeBilinearPatchPG_);
		if (hasTri)       pushTriple(hitRecs, hitProbeTrianglePG_);
		if (haveInstTri)  pushTriple(hitRecs, hitProbeTrianglePG_);
		if (haveInstSph)  pushTriple(hitRecs, hitProbeSpherePG_);
		if (hasDisks)     pushTriple(hitRecs, hitProbeDiskPG_);
		if (hasCylinders) pushTriple(hitRecs, hitProbeCylinderPG_);

		size_t sz = hitRecs.size() * sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeHitRecords_), sz));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeHitRecords_), hitRecs.data(),
							  sz, cudaMemcpyHostToDevice));

		probeSBT_.raygenRecord                = d_probeRaygenRecord_;
		probeSBT_.missRecordBase              = d_probeMissRecord_;
		probeSBT_.missRecordStrideInBytes     = sizeof(MissRecord);
		probeSBT_.missRecordCount             = 1;
		probeSBT_.hitgroupRecordBase          = d_probeHitRecords_;
		probeSBT_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
		probeSBT_.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());

		RaygenRecord probeExcRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(exceptionPG_, &probeExcRec));
		probeExcRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeExceptionRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeExceptionRecord_), &probeExcRec,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));
		probeSBT_.exceptionRecord = d_probeExceptionRecord_;
	}

	// ---- Shadow SBT ----
	{
		RaygenRecord rg;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenShadowPG_, &rg));
		rg.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowRaygenRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shadowRaygenRecord_), &rg,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));

		MissRecord missRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missShadowPG_, &missRec));
		missRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowMissRecord_), sizeof(MissRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shadowMissRecord_), &missRec,
							  sizeof(MissRecord), cudaMemcpyHostToDevice));

		std::vector<HitGroupRecord> hitRecs;
		if (hasSpheres) pushTriple(hitRecs, anyhitShadowSpherePG_);
		if (hasQuads)   pushTriple(hitRecs, anyhitShadowQuadPG_);
		if (hasBlp)     pushTriple(hitRecs, anyhitShadowBilinearPatchPG_);
		if (hasTri)     pushTriple(hitRecs, anyhitShadowTrianglePG_);
		// Same appended records as the intersect SBT above - both are
		// indexed by the same per-instance sbtOffset.
		if (haveInstTri)  pushTriple(hitRecs, anyhitShadowTrianglePG_);
		if (haveInstSph)  pushTriple(hitRecs, anyhitShadowSpherePG_);
		if (hasDisks)     pushTriple(hitRecs, anyhitShadowDiskPG_);
		if (hasCylinders) pushTriple(hitRecs, anyhitShadowCylinderPG_);

		size_t sz = hitRecs.size() * sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowHitRecords_), sz));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shadowHitRecords_), hitRecs.data(),
							  sz, cudaMemcpyHostToDevice));

		shadowSBT_.raygenRecord                = d_shadowRaygenRecord_;
		shadowSBT_.missRecordBase              = d_shadowMissRecord_;
		shadowSBT_.missRecordStrideInBytes     = sizeof(MissRecord);
		shadowSBT_.missRecordCount             = 1;
		shadowSBT_.hitgroupRecordBase          = d_shadowHitRecords_;
		shadowSBT_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
		shadowSBT_.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());

		// Exception record -- see the CUDA-718 fix comment in initialize().
		RaygenRecord excRec;
		OPTIX_CHECK(optixSbtRecordPackHeader(exceptionPG_, &excRec));
		excRec.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowExceptionRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shadowExceptionRecord_), &excRec,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));
		shadowSBT_.exceptionRecord = d_shadowExceptionRecord_;
	}

	std::cout << "[WavefrontPathTracer] Built SBTs (probe SBT included) (spheres=" << numSpheres
			  << " quads=" << numQuads
			  << " bilinearPatches=" << numBilinearPatches
			  << " disks=" << numDisks
			  << " cylinders=" << numCylinders
			  << " triangles=" << numTriangles << ")\n";
	return true;
}

// ============================================================================
// allocateQueues
// ============================================================================

bool WavefrontPathTracer::allocateQueues(int numPixels) {
	if (queueCapacity_ == numPixels) return true;  // already allocated

	freeQueues();
	queueCapacity_ = numPixels;
	size_t rayItemSz    = numPixels * sizeof(RayWorkItem);
	size_t hitItemSz    = numPixels * sizeof(HitWorkItem);
	size_t missItemSz   = numPixels * sizeof(MissWorkItem);
	size_t shadowItemSz = numPixels * sizeof(ShadowRayWorkItem);
	size_t occludedSz   = numPixels * sizeof(bool);
	// Worst case every hit this bounce is a Subsurface transmission -
	// same numPixels capacity class as every other per-bounce queue.
	size_t probeItemSz  = numPixels * sizeof(BssrdfProbeWorkItem);
	size_t exitItemSz   = numPixels * sizeof(BssrdfExitWorkItem);
	size_t counterSz    = sizeof(int);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayItems_),     rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayItems_), rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitItems_),     hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_simpleHitItems_), hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dielectricHitItems_), hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missItems_),    missItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowItems_),  shadowItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_occluded_),     occludedSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeItems_),   probeItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitItems_),    exitItemSz));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_simpleHitCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dielectricHitCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missCounter_),    counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowCounter_),  counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCounter_),   counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitCounter_),    counterSz));

	return true;
}

void WavefrontPathTracer::freeQueues() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_rayItems_);      freeDev(d_nextRayItems_);
	freeDev(d_hitItems_);      freeDev(d_simpleHitItems_);
	freeDev(d_dielectricHitItems_);
	freeDev(d_missItems_);
	freeDev(d_shadowItems_);   freeDev(d_occluded_);
	freeDev(d_probeItems_);    freeDev(d_exitItems_);
	freeDev(d_rayCounter_);    freeDev(d_nextRayCounter_);
	freeDev(d_hitCounter_);    freeDev(d_simpleHitCounter_);
	freeDev(d_dielectricHitCounter_);
	freeDev(d_missCounter_);
	freeDev(d_shadowCounter_);
	freeDev(d_probeCounter_);  freeDev(d_exitCounter_);
	queueCapacity_ = 0;
}

// ============================================================================
// Helper: read queue counter (device -> host)
// ============================================================================

int WavefrontPathTracer::readQueueSize(int* d_counter) {
	int val = 0;
	CUDA_CHECK(cudaMemcpyAsync(&val, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream_));
	CUDA_CHECK(cudaStreamSynchronize(stream_));
	return val;
}

void WavefrontPathTracer::resetQueueCounter(int* d_counter) {
	wf_reset_queue_counter(d_counter, stream_);
}

// ============================================================================
// CUDA kernel launchers
// ============================================================================

void WavefrontPathTracer::launchGenerateCameraRays(
	int width, int height, int sampleIdx,
	const GpuCameraParams& camera, float* d_weightBuffer)
{
	WorkQueue<RayWorkItem> rq;
	rq.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
	rq.counter  = reinterpret_cast<int*>(d_rayCounter_);
	rq.capacity = queueCapacity_;
	wf_launch_generate_camera_rays(rq, width, height, sampleIdx, camera, frameNumber_, d_weightBuffer, stream_);
}

void WavefrontPathTracer::launchEvaluateMaterials(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_hitItems_);
	hq.counter  = reinterpret_cast<int*>(d_hitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = queueCapacity_;

	// BSSRDF probe-request queue (MaterialType::Subsurface, Phase 2) - filled
	// by evaluate_materials's own Subsurface case instead of scattering
	// inline (see wavefront_types.h's BssrdfProbeWorkItem comment).
	WorkQueue<BssrdfProbeWorkItem> pq;
	pq.items    = reinterpret_cast<BssrdfProbeWorkItem*>(d_probeItems_);
	pq.counter  = reinterpret_cast<int*>(d_probeCounter_);
	pq.capacity = queueCapacity_;

	wf_launch_evaluate_materials(hq, numHits, nq, sq, pq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		reinterpret_cast<const CloudMedium<float>*>(d_cloudMediums_), numCloudMediums_,
		reinterpret_cast<const GpuRgbGridMedium*>(d_rgbGridMediums_),
		reinterpret_cast<const float*>(d_rgbGridData_),
		reinterpret_cast<const GpuGridMedium*>(d_gridMediums_),
		reinterpret_cast<const float*>(d_gridData_),
		reinterpret_cast<const GpuMeasuredTable*>(d_measuredTables_), numMeasuredTables_,
		reinterpret_cast<const float*>(d_measuredParamValues_),
		reinterpret_cast<const float*>(d_measuredData_),
		reinterpret_cast<const float*>(d_measuredMcdf_),
		reinterpret_cast<const float*>(d_measuredCcdf_),
		skyColor, shadowRayEpsilon, skyDist,
		stream_);
}

// Twin of launchEvaluateMaterials() above, scoped to simpleHitQueue's
// Lambertian/Metal hits - see wavefront_kernels.cu's evaluate_materials_simple()
// and wavefront_types.h's WavefrontQueues::simpleHitQueue comment.
void WavefrontPathTracer::launchEvaluateMaterialsSimple(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_simpleHitItems_);
	hq.counter  = reinterpret_cast<int*>(d_simpleHitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = queueCapacity_;

	wf_launch_evaluate_materials_simple(hq, numHits, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		skyColor, shadowRayEpsilon, skyDist,
		simpleMaterialStream_);
}

// Twin of launchEvaluateMaterialsSimple() above, scoped to dielectricHitQueue's
// Dielectric/RoughDielectric hits - see wavefront_kernels.cu's
// evaluate_materials_dielectric() and wavefront_types.h's WavefrontQueues::
// dielectricHitQueue comment. No texture params - neither material type reads
// mat.textureIdx.
void WavefrontPathTracer::launchEvaluateMaterialsDielectric(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_dielectricHitItems_);
	hq.counter  = reinterpret_cast<int*>(d_dielectricHitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = queueCapacity_;

	wf_launch_evaluate_materials_dielectric(hq, numHits, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		skyColor, shadowRayEpsilon, skyDist,
		dielectricMaterialStream_);
}

void WavefrontPathTracer::launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor,
												GpuSkyDistribution skyDist) {
	if (numMiss == 0) return;

	WorkQueue<MissWorkItem> mq;
	mq.items    = reinterpret_cast<MissWorkItem*>(d_missItems_);
	mq.counter  = reinterpret_cast<int*>(d_missCounter_);
	mq.capacity = queueCapacity_;

	wf_launch_accumulate_miss(mq, numMiss, d_framebuffer, backgroundColor, skyDist, stream_);
}

void WavefrontPathTracer::launchAccumulateShadow(
	int numShadow, const bool* d_occluded, float3* d_framebuffer)
{
	if (numShadow == 0) return;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = queueCapacity_;

	wf_launch_accumulate_shadow(sq, numShadow, d_occluded, d_framebuffer, stream_);
}

void WavefrontPathTracer::launchResolveBssrdfExit(
	int numExit,
	const MaterialData* d_materials, unsigned int numMaterials,
	const SphereData* d_spheres, unsigned int numSpheres,
	const QuadData* d_quads, unsigned int numQuads,
	const TriangleData* d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData* d_disks, unsigned int numDisks,
	const CylinderData* d_cylinders, unsigned int numCylinders,
	const int* d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable, unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist)
{
	if (numExit == 0) return;

	WorkQueue<BssrdfExitWorkItem> eq;
	eq.items    = reinterpret_cast<BssrdfExitWorkItem*>(d_exitItems_);
	eq.counter  = reinterpret_cast<int*>(d_exitCounter_);
	eq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = queueCapacity_;

	wf_launch_resolve_bssrdf_exit(eq, numExit, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		skyColor, shadowRayEpsilon, skyDist,
		stream_);
}

void WavefrontPathTracer::launchNormalizeFramebuffer(
	unsigned int numPixels, const float* d_weightBuffer, float3* d_framebuffer)
{
	wf_launch_normalize_framebuffer(numPixels, d_weightBuffer, d_framebuffer, stream_);
}

// ============================================================================
// render — main wavefront render loop
// ============================================================================

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
	const int numPixels = width * height;

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
	} stats;

	if (!allocateQueues(numPixels)) return false;

	// Allocate device framebuffer (float3 accumulator, zeroed each frame)
	CUdeviceptr d_fb = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fb), numPixels * sizeof(float3)));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_fb), 0,
							   numPixels * sizeof(float3), stream_));

	float3* d_fbPtr = reinterpret_cast<float3*>(d_fb);

	// Per-pixel sum of this render's filter weights (pbrt-v4 film
	// reconstruction formula) - same transient per-call alloc/free
	// lifecycle as d_fb itself, zeroed each render. See generate_camera_
	// rays's own filter_w comment and normalize_framebuffer's own comment.
	CUdeviceptr d_weight = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), numPixels * sizeof(float)));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_weight), 0,
							   numPixels * sizeof(float), stream_));
	float* d_weightPtr = reinterpret_cast<float*>(d_weight);

	// Build WavefrontLaunchParams template (queue pointers filled per phase)
	WavefrontLaunchParams lp = {};
	lp.framebuffer   = d_fbPtr;
	lp.width         = (unsigned int)width;
	lp.height        = (unsigned int)height;
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

	// -------------------------------------------------------------------------
	// Outer sample loop
	// -------------------------------------------------------------------------
	for (int sampleIdx = 0; sampleIdx < samples_per_pixel; ++sampleIdx) {

		// Reset ray queue counter, generate primary rays
		resetQueueCounter(reinterpret_cast<int*>(d_rayCounter_));
		launchGenerateCameraRays(width, height, sampleIdx, camera, d_weightPtr);
		CUDA_CHECK(cudaStreamSynchronize(stream_));

		// -------------------------------------------------------------------------
		// Inner bounce loop
		// -------------------------------------------------------------------------
		for (int depth = 0; depth < max_depth; ++depth) {

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
				numHits, max_depth,
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
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist);

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
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist);

			// ------------------------------------------------------------------
			// Phase 3c: Evaluate dielectric materials (Dielectric/RoughDielectric
			// hits routed to dielectricHitQueue at push time - see
			// wavefront_types.h's WavefrontQueues::dielectricHitQueue comment).
			// Same independent-queue/own-stream overlap reasoning as Phase 3b
			// above - launched on dielectricMaterialStream_, synced below
			// alongside simpleMaterialStream_ before numMiss/numShadow are read.
			// ------------------------------------------------------------------
			launchEvaluateMaterialsDielectric(
				numDielectricHits, max_depth,
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
				d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist);

			// ------------------------------------------------------------------
			// Phase 4: Accumulate miss (escaped rays → background)
			// ------------------------------------------------------------------
			launchAccumulateMiss(numMiss, d_fbPtr, camera.backgroundColor, camera.skyDist);

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
					d_fbPtr, camera.backgroundColor, camera.shadowRayEpsilon, camera.skyDist);

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
				lp.shadowQueue.capacity = queueCapacity_;

				// Temporarily point framebuffer to the occluded bool array so
				// __raygen__wf_shadow can write results there.
				// (The kernel casts (bool*)wf_params.framebuffer.)
				WavefrontLaunchParams shadowLP = lp;
				shadowLP.framebuffer = reinterpret_cast<float3*>(d_occluded_);

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
									   reinterpret_cast<const bool*>(d_occluded_),
									   d_fbPtr);
				CUDA_CHECK(cudaStreamSynchronize(stream_));
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
	// Normalize and copy to host
	// -------------------------------------------------------------------------
	launchNormalizeFramebuffer((unsigned int)numPixels, d_weightPtr, d_fbPtr);
	CUDA_CHECK(cudaStreamSynchronize(stream_));

	CUDA_CHECK(cudaMemcpy(framebuffer, reinterpret_cast<void*>(d_fb),
						  numPixels * sizeof(float3), cudaMemcpyDeviceToHost));

	cudaFree(reinterpret_cast<void*>(d_fb));
	cudaFree(reinterpret_cast<void*>(d_weight));

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
	destroyPG(exceptionPG_);
}

void WavefrontPathTracer::destroySBT() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_intersectRaygenRecord_); freeDev(d_intersectMissRecord_); freeDev(d_intersectHitRecords_);
	freeDev(d_shadowRaygenRecord_);    freeDev(d_shadowMissRecord_);    freeDev(d_shadowHitRecords_);
	freeDev(d_probeRaygenRecord_);     freeDev(d_probeMissRecord_);     freeDev(d_probeHitRecords_);
	freeDev(d_intersectExceptionRecord_); freeDev(d_shadowExceptionRecord_); freeDev(d_probeExceptionRecord_);
	intersectSBT_ = {};
	shadowSBT_    = {};
	probeSBT_     = {};
}

void WavefrontPathTracer::cleanup() {
	destroySBT();
	destroyProgramGroups();
	freeQueues();

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
#include "../../src/data/rgb_spectrum_table_data.cpp"
