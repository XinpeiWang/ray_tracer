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
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>

// Forward declarations for wavefront_launch.cu C wrappers (no <<<>>> in .cpp)
extern "C" void wf_launch_generate_camera_rays(
	WorkQueue<RayWorkItem>, int, int, int,
	GpuCameraParams, unsigned int, cudaStream_t);
extern "C" void wf_launch_evaluate_materials(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const GpuLightKind*, const GpuAliasEntry*, unsigned int,
	const PunctualLightGPU*, unsigned int,
	const TextureData*, const unsigned char*,
	int,
	const CloudMedium<float>*, unsigned int,
	cudaStream_t);
extern "C" void wf_launch_accumulate_miss(WorkQueue<MissWorkItem>, int, float3*, float3, cudaStream_t);
extern "C" void wf_launch_accumulate_shadow(WorkQueue<ShadowRayWorkItem>, int, const bool*, float3*, cudaStream_t);
extern "C" void wf_launch_normalize_framebuffer(unsigned int, float, float3*, cudaStream_t);
extern "C" void wf_reset_queue_counter(int*, cudaStream_t);
extern "C" void wf_upload_cie_tables(const float*, const float*, const float*, int);
extern "C" void wf_upload_srgb_table(const float*, const float*, int);

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

	OptixProgramGroupDesc triHitDesc = {};
	triHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	triHitDesc.hitgroup.moduleIS               = wfModule_;
	triHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__wf_triangle";
	triHitDesc.hitgroup.moduleCH               = wfModule_;
	triHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_triangle";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &triHitDesc, 1, &pgOptions,
										 log, &logSize, &hitTrianglePG_));

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

	std::cout << "[WavefrontPathTracer] Created 13 program groups\n";
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

	// Intersect pipeline
	OptixProgramGroup intersectGroups[] = {
		raygenIntersectPG_,
		missRadiancePG_,
		hitSpherePG_,
		hitQuadPG_,
		hitBilinearPatchPG_,
		hitTrianglePG_,
		exceptionPG_
	};
	{
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &linkOptions,
			intersectGroups, 7, log, &logSize, &intersectPipeline_));
		std::cout << "[WavefrontPathTracer] Linked intersect pipeline\n";
	}

	// Shadow pipeline (depth 1 — just any-hit)
	OptixProgramGroup shadowGroups[] = {
		raygenShadowPG_,
		missShadowPG_,
		anyhitShadowSpherePG_,
		anyhitShadowQuadPG_,
		anyhitShadowBilinearPatchPG_,
		anyhitShadowTrianglePG_,
		exceptionPG_
	};
	OptixPipelineLinkOptions shadowLinkOptions = {};
	shadowLinkOptions.maxTraceDepth = 1;
	{
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &shadowLinkOptions,
			shadowGroups, 7, log, &logSize, &shadowPipeline_));
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
	setComputedStackSize(intersectPipeline_, intersectGroups, 7, maxTraceDepth);
	setComputedStackSize(shadowPipeline_, shadowGroups, 7, 1);

	return true;
}

// ============================================================================
// buildSBT — create two separate SBTs
// ============================================================================

bool WavefrontPathTracer::buildSBT(unsigned int numSpheres, unsigned int numQuads, unsigned int numBilinearPatches, unsigned int numTriangles) {
	// haveInstanced* come from setInstancedGeometryFlags(); see its comment.
	const bool haveInstTri = haveInstancedTriangles_;
	const bool haveInstSph = haveInstancedSpheres_;
	numSpheres_ = numSpheres;
	numQuads_   = numQuads;
	numBilinearPatches_ = numBilinearPatches;
	numTriangles_ = numTriangles;

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

		// Hit records: SBT offset=0, stride=2 means each PRESENT type (in
		// [sphere, quad, bilinear patch] order, absent types omitted) needs a
		// (radiance, unused) pair - see the hasSpheres/hasQuads/hasBlp comment
		// above for why position among present types, not fixed geometry-type
		// index, determines the slot.
		std::vector<HitGroupRecord> hitRecs;
		if (hasSpheres) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_, &hitRecs.back())); hitRecs.back().data = {}; // unused slot
		}
		if (hasQuads) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitQuadPG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitQuadPG_, &hitRecs.back())); hitRecs.back().data = {}; // unused slot
		}
		if (hasBlp) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitBilinearPatchPG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitBilinearPatchPG_, &hitRecs.back())); hitRecs.back().data = {}; // unused slot
		}
		if (hasTri) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitTrianglePG_, &hitRecs.back())); hitRecs.back().data = {}; // unused slot
		}
		// Instanced geometry's own pairs, appended after the scene's packed
		// region in the same order OptiXRenderer::buildSBT() uses - see
		// setInstancedGeometryFlags().
		if (haveInstTri) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		if (haveInstSph) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
		}

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
		if (hasSpheres) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		if (hasQuads) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowQuadPG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowQuadPG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		if (hasBlp) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowBilinearPatchPG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowBilinearPatchPG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		if (hasTri) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		// Same appended pairs as the intersect SBT above - both are indexed
		// by the same per-instance sbtOffset.
		if (haveInstTri) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowTrianglePG_, &hitRecs.back())); hitRecs.back().data = {};
		}
		if (haveInstSph) {
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
			hitRecs.emplace_back(); OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs.back())); hitRecs.back().data = {};
		}

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

	std::cout << "[WavefrontPathTracer] Built SBTs (spheres=" << numSpheres
			  << " quads=" << numQuads
			  << " bilinearPatches=" << numBilinearPatches
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
	size_t counterSz    = sizeof(int);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayItems_),     rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayItems_), rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitItems_),     hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missItems_),    missItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowItems_),  shadowItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_occluded_),     occludedSz));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missCounter_),    counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowCounter_),  counterSz));

	return true;
}

void WavefrontPathTracer::freeQueues() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_rayItems_);      freeDev(d_nextRayItems_);
	freeDev(d_hitItems_);      freeDev(d_missItems_);
	freeDev(d_shadowItems_);   freeDev(d_occluded_);
	freeDev(d_rayCounter_);    freeDev(d_nextRayCounter_);
	freeDev(d_hitCounter_);    freeDev(d_missCounter_);
	freeDev(d_shadowCounter_);
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
	const GpuCameraParams& camera)
{
	WorkQueue<RayWorkItem> rq;
	rq.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
	rq.counter  = reinterpret_cast<int*>(d_rayCounter_);
	rq.capacity = queueCapacity_;
	wf_launch_generate_camera_rays(rq, width, height, sampleIdx, camera, frameNumber_, stream_);
}

void WavefrontPathTracer::launchEvaluateMaterials(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer)
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

	wf_launch_evaluate_materials(hq, numHits, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		reinterpret_cast<const CloudMedium<float>*>(d_cloudMediums_), numCloudMediums_,
		stream_);
}

void WavefrontPathTracer::launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor) {
	if (numMiss == 0) return;

	WorkQueue<MissWorkItem> mq;
	mq.items    = reinterpret_cast<MissWorkItem*>(d_missItems_);
	mq.counter  = reinterpret_cast<int*>(d_missCounter_);
	mq.capacity = queueCapacity_;

	wf_launch_accumulate_miss(mq, numMiss, d_framebuffer, backgroundColor, stream_);
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

void WavefrontPathTracer::launchNormalizeFramebuffer(
	unsigned int numPixels, float invSPP, float3* d_framebuffer)
{
	wf_launch_normalize_framebuffer(numPixels, invSPP, d_framebuffer, stream_);
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
	unsigned int num_triangles)
{
	const int numPixels = width * height;

	if (!allocateQueues(numPixels)) return false;

	// Allocate device framebuffer (float3 accumulator, zeroed each frame)
	CUdeviceptr d_fb = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fb), numPixels * sizeof(float3)));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_fb), 0,
							   numPixels * sizeof(float3), stream_));

	float3* d_fbPtr = reinterpret_cast<float3*>(d_fb);

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
	lp.materials     = reinterpret_cast<MaterialData*>(d_materials);
	lp.numMaterials  = num_materials;
	lp.cloudMediums    = reinterpret_cast<CloudMedium<float>*>(d_cloudMediums_);
	lp.numCloudMediums = numCloudMediums_;
	lp.lightIndices  = reinterpret_cast<int*>(d_light_indices);
	lp.lightKinds = reinterpret_cast<const GpuLightKind*>(d_lightKinds);
	lp.instancePrimBase = reinterpret_cast<const int*>(d_instancePrimBase_);
	lp.aliasTable    = reinterpret_cast<GpuAliasEntry*>(d_alias_table);
	lp.numLights     = num_lights;
	lp.samplesPerPixel = (unsigned int)samples_per_pixel;
	lp.maxDepth        = (unsigned int)max_depth;
	lp.frameNumber     = frameNumber_++;

	// -------------------------------------------------------------------------
	// Outer sample loop
	// -------------------------------------------------------------------------
	for (int sampleIdx = 0; sampleIdx < samples_per_pixel; ++sampleIdx) {

		// Reset ray queue counter, generate primary rays
		resetQueueCounter(reinterpret_cast<int*>(d_rayCounter_));
		launchGenerateCameraRays(width, height, sampleIdx, camera);
		CUDA_CHECK(cudaStreamSynchronize(stream_));

		// -------------------------------------------------------------------------
		// Inner bounce loop
		// -------------------------------------------------------------------------
		for (int depth = 0; depth < max_depth; ++depth) {

			int numRays = readQueueSize(reinterpret_cast<int*>(d_rayCounter_));
			if (numRays == 0) break;

			// Reset output queue counters
			resetQueueCounter(reinterpret_cast<int*>(d_hitCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_missCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_shadowCounter_));
			resetQueueCounter(reinterpret_cast<int*>(d_nextRayCounter_));

			// ------------------------------------------------------------------
			// Phase 2: OptiX intersect launch
			// ------------------------------------------------------------------
			lp.rayQueue.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
			lp.rayQueue.counter  = reinterpret_cast<int*>(d_rayCounter_);
			lp.rayQueue.capacity = queueCapacity_;

			lp.hitQueue.items    = reinterpret_cast<HitWorkItem*>(d_hitItems_);
			lp.hitQueue.counter  = reinterpret_cast<int*>(d_hitCounter_);
			lp.hitQueue.capacity = queueCapacity_;

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

			int numHits   = readQueueSize(reinterpret_cast<int*>(d_hitCounter_));
			int numMiss   = readQueueSize(reinterpret_cast<int*>(d_missCounter_));

			// ------------------------------------------------------------------
			// Phase 3: Evaluate materials (fills shadowQueue + nextRayQueue)
			// ------------------------------------------------------------------
			launchEvaluateMaterials(
				numHits, max_depth,
				reinterpret_cast<const SphereData*>(d_spheres), num_spheres,
				reinterpret_cast<const QuadData*>(d_quads),     num_quads,
				reinterpret_cast<const MaterialData*>(d_materials), num_materials,
				reinterpret_cast<const int*>(d_light_indices),
				reinterpret_cast<const GpuLightKind*>(d_lightKinds),
				reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
				num_lights,
				reinterpret_cast<const PunctualLightGPU*>(d_punctual_lights),
				num_punctual_lights,
				d_fbPtr);

			// ------------------------------------------------------------------
			// Phase 4: Accumulate miss (escaped rays → background)
			// ------------------------------------------------------------------
			launchAccumulateMiss(numMiss, d_fbPtr, camera.backgroundColor);

			CUDA_CHECK(cudaStreamSynchronize(stream_));

			int numShadow = readQueueSize(reinterpret_cast<int*>(d_shadowCounter_));

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
	}

	// -------------------------------------------------------------------------
	// Normalize and copy to host
	// -------------------------------------------------------------------------
	launchNormalizeFramebuffer((unsigned int)numPixels,
							   1.0f / float(samples_per_pixel), d_fbPtr);
	CUDA_CHECK(cudaStreamSynchronize(stream_));

	CUDA_CHECK(cudaMemcpy(framebuffer, reinterpret_cast<void*>(d_fb),
						  numPixels * sizeof(float3), cudaMemcpyDeviceToHost));

	cudaFree(reinterpret_cast<void*>(d_fb));

	std::cout << "[WavefrontPathTracer] Rendered " << width << "x" << height
			  << " (" << samples_per_pixel << " spp, " << max_depth << " bounces)\n";
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
	destroyPG(exceptionPG_);
}

void WavefrontPathTracer::destroySBT() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_intersectRaygenRecord_); freeDev(d_intersectMissRecord_); freeDev(d_intersectHitRecords_);
	freeDev(d_shadowRaygenRecord_);    freeDev(d_shadowMissRecord_);    freeDev(d_shadowHitRecords_);
	freeDev(d_intersectExceptionRecord_); freeDev(d_shadowExceptionRecord_);
	intersectSBT_ = {};
	shadowSBT_    = {};
}

void WavefrontPathTracer::cleanup() {
	destroySBT();
	destroyProgramGroups();
	freeQueues();

	if (intersectPipeline_) { optixPipelineDestroy(intersectPipeline_); intersectPipeline_ = nullptr; }
	if (shadowPipeline_)    { optixPipelineDestroy(shadowPipeline_);    shadowPipeline_    = nullptr; }
	if (wfModule_)          { optixModuleDestroy(wfModule_);            wfModule_          = nullptr; }
	if (d_wfLaunchParams_)  { cudaFree(reinterpret_cast<void*>(d_wfLaunchParams_)); d_wfLaunchParams_ = 0; }
}

} // namespace optix_renderer

// Pull in the sRGB spectral table data so it links into this TU
// (avoids needing rgb_spectrum_table_data.cpp as a separate project source).
#include "../../src/data/rgb_spectrum_table_data.cpp"
