// wavefront_path_tracer_setup.cpp - WavefrontPathTracer: creating the OptiX pipeline (initialize, module, program groups, linking, the shader binding table).
// A pure split of wavefront_path_tracer.cpp; nothing here changed.

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
		// GetNormalizedD65Illuminant() (cie_data.h) does exactly this
		// normalization already - sample it directly instead of
		// re-deriving the same normConst/scale loop a second time in this
		// file. That re-derivation is how the ~20%/~9% R/G/B skew this
		// comment used to describe first shipped: two independent copies
		// of the same computation drifted, and CPU's own --spectral
		// (src/TheRestOfYourLife/camera.h) now depends on this exact
		// function too, so there is only one implementation left to keep
		// correct.
		static float h_d65[kCIENSamples];
		const DenselySampledSpectrum& d65 = GetNormalizedD65Illuminant();
		for (int i = 0; i < kCIENSamples; ++i) {
			float lambda = static_cast<float>(kCIELambda_min + i);
			h_d65[i] = d65(lambda);
		}
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

	OPTIX_CHECK(optixModuleCreateParallel(
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

	// ----- Probe cache update rays (Live Preview only) -----
	// Own raygen only - reuses the SAME hit/miss groups as the BSSRDF probe
	// walk above (missProbePG_/hitProbeSpherePG_/...) via a second SBT
	// (probeCacheSBT_) that shares probeSBT_'s own hit/miss record buffers -
	// see buildSBT()'s own comment. No new hit/miss programs needed: this
	// raygen wants exactly the same "find the closest surface hit, report
	// position/normal/materialIdx" query the probe walk already has.
	OptixProgramGroupDesc probeCacheRgDesc = {};
	probeCacheRgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	probeCacheRgDesc.raygen.module            = wfModule_;
	probeCacheRgDesc.raygen.entryFunctionName = "__raygen__wf_probe_cache";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &probeCacheRgDesc, 1, &pgOptions,
										 log, &logSize, &raygenProbeCachePG_));

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
		raygenProbeCachePG_,
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

	// ---- Probe CACHE SBT (world-space irradiance probe cache, Live Preview
	// only) ----
	// Own raygen record only - every other field is a direct COPY of
	// probeSBT_'s own values, just built above: same hit/miss program groups
	// (missProbePG_/hitProbeSpherePG_/...), same SBT layout, since a probe-
	// cache-update ray wants the EXACT same "find the closest surface hit"
	// query the BSSRDF probe walk already has (see wavefront_probe_cache.h's
	// own header comment for why only the raygen differs - the shading logic
	// that would otherwise live in a closest-hit program lives in a separate
	// plain CUDA kernel, probe_cache_shade, instead).
	{
		RaygenRecord rg;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenProbeCachePG_, &rg));
		rg.data = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheRaygenRecord_), sizeof(RaygenRecord)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_probeCacheRaygenRecord_), &rg,
							  sizeof(RaygenRecord), cudaMemcpyHostToDevice));

		probeCacheSBT_.raygenRecord                = d_probeCacheRaygenRecord_;
		probeCacheSBT_.missRecordBase              = probeSBT_.missRecordBase;
		probeCacheSBT_.missRecordStrideInBytes     = probeSBT_.missRecordStrideInBytes;
		probeCacheSBT_.missRecordCount             = probeSBT_.missRecordCount;
		probeCacheSBT_.hitgroupRecordBase          = probeSBT_.hitgroupRecordBase;
		probeCacheSBT_.hitgroupRecordStrideInBytes = probeSBT_.hitgroupRecordStrideInBytes;
		probeCacheSBT_.hitgroupRecordCount         = probeSBT_.hitgroupRecordCount;
		probeCacheSBT_.exceptionRecord             = probeSBT_.exceptionRecord;
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


}  // namespace optix_renderer
