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
	float3, float3, float3, float3, unsigned int, cudaStream_t);
extern "C" void wf_launch_evaluate_materials(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, unsigned int,
	const QuadData*, unsigned int,
	const MaterialData*, unsigned int,
	const int*, const bool*, const GpuAliasEntry*, unsigned int,
	int, cudaStream_t);
extern "C" void wf_launch_accumulate_miss(WorkQueue<MissWorkItem>, int, float3*, cudaStream_t);
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
									  OptixModule /*ignored*/,
									  cudaStream_t stream) {
	context_ = context;
	stream_  = stream;

	pipelineCompileOptions_.usesMotionBlur        = false;
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions_.numPayloadValues       = 2;  // pointer p0/p1
	pipelineCompileOptions_.numAttributeValues     = 4;
	pipelineCompileOptions_.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
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
	// ptxPath_ is set by enableWavefront() to point at wavefront_programs.ptx.
	// If somehow empty, fall back to the same directory as the current working directory.
	std::string ptxPath = ptxPath_;
	if (ptxPath.empty()) {
		ptxPath = "wavefront_programs.ptx";
	}

	// Read PTX file
	std::ifstream file(ptxPath, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "[WavefrontPathTracer] Could not open PTX: " << ptxPath << "\n";
		return false;
	}
	std::ostringstream oss;
	oss << file.rdbuf();
	std::string ptxSource = oss.str();

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

	std::cout << "[WavefrontPathTracer] Loaded module from " << ptxPath << "\n";
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

	// Sphere closesthit (re-uses intersection program from optix_programs.ptx — but
	// we need the intersection program in the SAME module as the closesthit.
	// Solution: the wavefront module includes __intersection__sphere/quad stubs
	// that are identical to the recursive ones, or we borrow from the base module_.
	// Here we use module_ (the base RecursivePathTracer module) for intersection
	// programs and wfModule_ for the wavefront closesthit programs.
	// This works because OptiX allows mixing modules in a hitgroup.)
	OptixProgramGroupDesc sphereHitDesc = {};
	sphereHitDesc.kind                              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	sphereHitDesc.hitgroup.moduleIS                 = module_;  // base module has __intersection__sphere
	sphereHitDesc.hitgroup.entryFunctionNameIS      = "__intersection__sphere";
	sphereHitDesc.hitgroup.moduleCH                 = wfModule_;
	sphereHitDesc.hitgroup.entryFunctionNameCH      = "__closesthit__wf_sphere";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &sphereHitDesc, 1, &pgOptions,
										 log, &logSize, &hitSpherePG_));

	OptixProgramGroupDesc quadHitDesc = {};
	quadHitDesc.kind                            = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	quadHitDesc.hitgroup.moduleIS               = module_;
	quadHitDesc.hitgroup.entryFunctionNameIS    = "__intersection__quad";
	quadHitDesc.hitgroup.moduleCH               = wfModule_;
	quadHitDesc.hitgroup.entryFunctionNameCH    = "__closesthit__wf_quad";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &quadHitDesc, 1, &pgOptions,
										 log, &logSize, &hitQuadPG_));

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
	shadowSphereDesc.hitgroup.moduleIS             = module_;
	shadowSphereDesc.hitgroup.entryFunctionNameIS  = "__intersection__sphere";
	shadowSphereDesc.hitgroup.moduleAH             = wfModule_;
	shadowSphereDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowSphereDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowSpherePG_));

	// Shadow anyhit for quad
	OptixProgramGroupDesc shadowQuadDesc = {};
	shadowQuadDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowQuadDesc.hitgroup.moduleIS             = module_;
	shadowQuadDesc.hitgroup.entryFunctionNameIS  = "__intersection__quad";
	shadowQuadDesc.hitgroup.moduleAH             = wfModule_;
	shadowQuadDesc.hitgroup.entryFunctionNameAH  = "__anyhit__wf_shadow";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &shadowQuadDesc, 1, &pgOptions,
										 log, &logSize, &anyhitShadowQuadPG_));

	std::cout << "[WavefrontPathTracer] Created 8 program groups\n";
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
	{
		OptixProgramGroup groups[] = {
			raygenIntersectPG_,
			missRadiancePG_,
			hitSpherePG_,
			hitQuadPG_
		};
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &linkOptions,
			groups, 4, log, &logSize, &intersectPipeline_));
		std::cout << "[WavefrontPathTracer] Linked intersect pipeline\n";
	}

	// Shadow pipeline (depth 1 — just any-hit)
	{
		OptixPipelineLinkOptions shadowLinkOptions = {};
		shadowLinkOptions.maxTraceDepth = 1;

		OptixProgramGroup groups[] = {
			raygenShadowPG_,
			missShadowPG_,
			anyhitShadowSpherePG_,
			anyhitShadowQuadPG_
		};
		logSize = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(
			context_, &pipelineCompileOptions_, &shadowLinkOptions,
			groups, 4, log, &logSize, &shadowPipeline_));
		std::cout << "[WavefrontPathTracer] Linked shadow pipeline\n";
	}

	return true;
}

// ============================================================================
// buildSBT — create two separate SBTs
// ============================================================================

bool WavefrontPathTracer::buildSBT(unsigned int numSpheres, unsigned int numQuads) {
	numSpheres_ = numSpheres;
	numQuads_   = numQuads;

	destroySBT();

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

		// Hit records: [sphere_radiance, quad_radiance]
		// SBT layout: SBT_offset=0, stride=2 means we need:
		//   primitive 0 (spheres): slot 0*2+0=0 (radiance), slot 0*2+1 unused
		//   primitive 1 (quads):   slot 1*2+0=2 (radiance), slot 1*2+1 unused
		// But our raygen uses SBT offset=0 stride=2 miss=0, so:
		//   For sphere build input (instIdx=0): sbt offset = 0*2 + 0 = 0
		//   For quad  build input (instIdx=1): sbt offset = 1*2 + 0 = 2
		// We only need 4 records: [sph_rad, sph_unused, quad_rad, quad_unused]
		std::vector<HitGroupRecord> hitRecs(4);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_,    &hitRecs[0])); hitRecs[0].data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_,    &hitRecs[1])); hitRecs[1].data = {}; // unused slot
		OPTIX_CHECK(optixSbtRecordPackHeader(hitQuadPG_,      &hitRecs[2])); hitRecs[2].data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(hitQuadPG_,      &hitRecs[3])); hitRecs[3].data = {}; // unused slot

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

		std::vector<HitGroupRecord> hitRecs(4);
		OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs[0])); hitRecs[0].data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowSpherePG_, &hitRecs[1])); hitRecs[1].data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowQuadPG_,   &hitRecs[2])); hitRecs[2].data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(anyhitShadowQuadPG_,   &hitRecs[3])); hitRecs[3].data = {};

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
	}

	std::cout << "[WavefrontPathTracer] Built SBTs (spheres=" << numSpheres
			  << " quads=" << numQuads << ")\n";
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
	float3 camOrigin, float3 lowerLeft, float3 horizontal, float3 vertical)
{
	WorkQueue<RayWorkItem> rq;
	rq.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
	rq.counter  = reinterpret_cast<int*>(d_rayCounter_);
	rq.capacity = queueCapacity_;
	wf_launch_generate_camera_rays(rq, width, height, sampleIdx,
		camOrigin, lowerLeft, horizontal, vertical, frameNumber_, stream_);
}

void WavefrontPathTracer::launchEvaluateMaterials(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const bool* d_isLightSphere,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
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
		d_lightIndices, d_isLightSphere, d_aliasTable, numLights,
		maxDepth, stream_);
}

void WavefrontPathTracer::launchAccumulateMiss(int numMiss, float3* d_framebuffer) {
	if (numMiss == 0) return;

	WorkQueue<MissWorkItem> mq;
	mq.items    = reinterpret_cast<MissWorkItem*>(d_missItems_);
	mq.counter  = reinterpret_cast<int*>(d_missCounter_);
	mq.capacity = queueCapacity_;

	wf_launch_accumulate_miss(mq, numMiss, d_framebuffer, stream_);
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
	const float* camera_origin, const float* camera_lower_left,
	const float* camera_horizontal, const float* camera_vertical,
	float*  framebuffer,          // host-side output
	OptixTraversableHandle gas_handle,
	CUdeviceptr d_materials,
	CUdeviceptr d_spheres,
	CUdeviceptr d_quads,
	CUdeviceptr d_light_indices,
	CUdeviceptr d_is_light_sphere,
	CUdeviceptr d_alias_table,
	unsigned int num_materials,
	unsigned int num_spheres,
	unsigned int num_quads,
	unsigned int num_lights)
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
	lp.materials     = reinterpret_cast<MaterialData*>(d_materials);
	lp.numMaterials  = num_materials;
	lp.lightIndices  = reinterpret_cast<int*>(d_light_indices);
	lp.isLightSphere = reinterpret_cast<bool*>(d_is_light_sphere);
	lp.aliasTable    = reinterpret_cast<GpuAliasEntry*>(d_alias_table);
	lp.numLights     = num_lights;
	lp.samplesPerPixel = (unsigned int)samples_per_pixel;
	lp.maxDepth        = (unsigned int)max_depth;
	lp.frameNumber     = frameNumber_++;

	float3 camOrigin  = { camera_origin[0],     camera_origin[1],     camera_origin[2] };
	float3 lowerLeft  = { camera_lower_left[0],  camera_lower_left[1],  camera_lower_left[2] };
	float3 horizontal = { camera_horizontal[0],  camera_horizontal[1],  camera_horizontal[2] };
	float3 vertical   = { camera_vertical[0],    camera_vertical[1],    camera_vertical[2] };

	// -------------------------------------------------------------------------
	// Outer sample loop
	// -------------------------------------------------------------------------
	for (int sampleIdx = 0; sampleIdx < samples_per_pixel; ++sampleIdx) {

		// Reset ray queue counter, generate primary rays
		resetQueueCounter(reinterpret_cast<int*>(d_rayCounter_));
		launchGenerateCameraRays(width, height, sampleIdx, camOrigin, lowerLeft, horizontal, vertical);
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
				reinterpret_cast<const bool*>(d_is_light_sphere),
				reinterpret_cast<const GpuAliasEntry*>(d_alias_table),
				num_lights,
				d_fbPtr);

			// ------------------------------------------------------------------
			// Phase 4: Accumulate miss (escaped rays → background)
			// ------------------------------------------------------------------
			launchAccumulateMiss(numMiss, d_fbPtr);

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
	destroyPG(hitSpherePG_);         destroyPG(hitQuadPG_);
	destroyPG(raygenShadowPG_);      destroyPG(missShadowPG_);
	destroyPG(anyhitShadowSpherePG_); destroyPG(anyhitShadowQuadPG_);
}

void WavefrontPathTracer::destroySBT() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_intersectRaygenRecord_); freeDev(d_intersectMissRecord_); freeDev(d_intersectHitRecords_);
	freeDev(d_shadowRaygenRecord_);    freeDev(d_shadowMissRecord_);    freeDev(d_shadowHitRecords_);
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
