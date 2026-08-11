// sppm_path_tracer.cpp
// SPPMPathTracer host-side implementation -- Phase 1a: pipeline/SBT
// plumbing plus a trivial hit-or-miss camera-pass raygen, proving the
// module/program-group/pipeline/SBT/PTX-JIT machinery works against a real
// uploaded scene before any SPPM math is written. See sppm_path_tracer.h's
// own comment for why this class doesn't implement PathTracingStrategy.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "sppm_path_tracer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace optix_renderer {

SPPMPathTracer::SPPMPathTracer() = default;

SPPMPathTracer::~SPPMPathTracer() {
	cleanup();
}

bool SPPMPathTracer::initialize(OptixDeviceContext context, cudaStream_t stream) {
	context_ = context;
	stream_  = stream;

	pipelineCompileOptions_.usesMotionBlur        = false;
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions_.numPayloadValues       = 2;  // pointer p0/p1, same technique as wavefront
	pipelineCompileOptions_.numAttributeValues     = 4;  // matches wavefront/recursive (sphere center.xyz + radius)
	pipelineCompileOptions_.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions_.pipelineLaunchParamsVariableName = "sppm_params";

	if (!loadModule()) return false;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams_), sizeof(SPPMLaunchParams)));

	std::cout << "[SPPMPathTracer] Initialized\n";
	return true;
}

bool SPPMPathTracer::loadModule() {
	// If the caller supplied an explicit path (setPTXPath()), try it first;
	// otherwise (or if that path doesn't exist) fall back to the same
	// cwd-relative search OptiXRenderer::loadPTX() uses for optix_programs.ptx
	// -- output_path-derived paths (what WavefrontPathTracer's own ptxPath_
	// convention assumes) don't reliably point at a directory that actually
	// has the PTX file (e.g. the default --output writes to a nested
	// "output/" subdirectory the build never copies PTX into), so relying on
	// that alone isn't robust.
	std::vector<std::string> candidates;
	if (!ptxPath_.empty()) candidates.push_back(ptxPath_);
	candidates.push_back("gpu/optix/sppm_programs.ptx");
	candidates.push_back("optix_output/sppm_programs.ptx");
	candidates.push_back("./sppm_programs.ptx");

	std::string ptxSource;
	bool found = false;
	for (const auto& candidate : candidates) {
		std::ifstream file(candidate, std::ios::binary);
		if (!file.is_open()) continue;
		std::ostringstream oss;
		oss << file.rdbuf();
		ptxSource = oss.str();
		std::cout << "[SPPMPathTracer] Loaded PTX from: " << candidate << "\n";
		found = true;
		break;
	}
	if (!found) {
		std::cerr << "[SPPMPathTracer] Could not find sppm_programs.ptx (tried "
		          << candidates.size() << " candidate paths)\n";
		return false;
	}

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
		&sppmModule_
	));

	std::cout << "[SPPMPathTracer] Module created\n";
	return true;
}

bool SPPMPathTracer::createProgramGroups() {
	char   log[2048];
	size_t logSize;
	OptixProgramGroupOptions pgOptions = {};

	OptixProgramGroupDesc rgDesc = {};
	rgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	rgDesc.raygen.module            = sppmModule_;
	rgDesc.raygen.entryFunctionName = "__raygen__sppm_camera_pass";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &rgDesc, 1, &pgOptions,
	                                     log, &logSize, &raygenCameraPG_));

	OptixProgramGroupDesc missDesc = {};
	missDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missDesc.miss.module            = sppmModule_;
	missDesc.miss.entryFunctionName = "__miss__sppm_radiance";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &missDesc, 1, &pgOptions,
	                                     log, &logSize, &missRadiancePG_));

	OptixProgramGroupDesc sphereHitDesc = {};
	sphereHitDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	sphereHitDesc.hitgroup.moduleIS             = sppmModule_;
	sphereHitDesc.hitgroup.entryFunctionNameIS  = "__intersection__sppm_sphere";
	sphereHitDesc.hitgroup.moduleCH             = sppmModule_;
	sphereHitDesc.hitgroup.entryFunctionNameCH  = "__closesthit__sppm_sphere";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &sphereHitDesc, 1, &pgOptions,
	                                     log, &logSize, &hitSpherePG_));

	OptixProgramGroupDesc quadHitDesc = {};
	quadHitDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	quadHitDesc.hitgroup.moduleIS            = sppmModule_;
	quadHitDesc.hitgroup.entryFunctionNameIS = "__intersection__sppm_quad";
	quadHitDesc.hitgroup.moduleCH            = sppmModule_;
	quadHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__sppm_quad";
	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(context_, &quadHitDesc, 1, &pgOptions,
	                                     log, &logSize, &hitQuadPG_));

	std::cout << "[SPPMPathTracer] Created 4 program groups\n";
	return true;
}

bool SPPMPathTracer::linkPipeline() {
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.maxTraceDepth = 1;  // Phase 1a: one optixTrace call, no recursion

	OptixProgramGroup groups[] = {
		raygenCameraPG_,
		missRadiancePG_,
		hitSpherePG_,
		hitQuadPG_
	};

	char   log[2048];
	size_t logSize = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context_, &pipelineCompileOptions_, &linkOptions,
		groups, 4, log, &logSize, &pipeline_));

	std::cout << "[SPPMPathTracer] Linked pipeline\n";
	return true;
}

bool SPPMPathTracer::buildSBT(unsigned int numSpheres, unsigned int numQuads) {
	numSpheres_ = numSpheres;
	numQuads_   = numQuads;

	destroySBT();

	// Same present/absent geometry-type filter as WavefrontPathTracer::buildSBT
	// (see its own comment) -- OptiX rejects a zero-primitive custom-primitive
	// build input, so buildScene() omits empty geometry types from the shared
	// GAS entirely, meaning a type's position among the *present* types (not
	// a fixed geometry-type index) determines its SBT slot. Scene 11 (Phase
	// 1a/1b's only target) has both spheres and quads, but this still needs
	// to handle either being absent correctly for buildSBT() to be a faithful
	// port, not a scene-11-specific special case.
	const bool hasSpheres = numSpheres > 0;
	const bool hasQuads   = numQuads > 0;

	RaygenRecord rg;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenCameraPG_, &rg));
	rg.data = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord_), sizeof(RaygenRecord)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygenRecord_), &rg,
	                      sizeof(RaygenRecord), cudaMemcpyHostToDevice));

	MissRecord missRec;
	OPTIX_CHECK(optixSbtRecordPackHeader(missRadiancePG_, &missRec));
	missRec.data = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord_), sizeof(MissRecord)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_missRecord_), &missRec,
	                      sizeof(MissRecord), cudaMemcpyHostToDevice));

	std::vector<HitGroupRecord> hitRecs;
	if (hasSpheres) {
		hitRecs.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitSpherePG_, &hitRecs.back()));
		hitRecs.back().data = {};
	}
	if (hasQuads) {
		hitRecs.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitQuadPG_, &hitRecs.back()));
		hitRecs.back().data = {};
	}

	size_t sz = hitRecs.size() * sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitRecords_), sz));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitRecords_), hitRecs.data(),
	                      sz, cudaMemcpyHostToDevice));

	sbt_.raygenRecord                = d_raygenRecord_;
	sbt_.missRecordBase              = d_missRecord_;
	sbt_.missRecordStrideInBytes     = sizeof(MissRecord);
	sbt_.missRecordCount             = 1;
	sbt_.hitgroupRecordBase          = d_hitRecords_;
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
	sbt_.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());

	std::cout << "[SPPMPathTracer] Built SBT (spheres=" << numSpheres
	          << " quads=" << numQuads << ")\n";
	return true;
}

bool SPPMPathTracer::renderTrivial(int width, int height, const GpuCameraParams& camera,
                                    float* outputFramebuffer, OptixTraversableHandle gasHandle,
                                    CUdeviceptr d_materials, CUdeviceptr d_spheres, CUdeviceptr d_quads,
                                    unsigned int numMaterials, unsigned int numSpheres, unsigned int numQuads) {
	CUdeviceptr d_framebuffer;
	size_t fbSize = static_cast<size_t>(width) * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_framebuffer), fbSize));

	SPPMLaunchParams params{};
	params.traversable  = gasHandle;
	params.spheres      = reinterpret_cast<SphereData*>(d_spheres);
	params.numSpheres    = numSpheres;
	params.quads        = reinterpret_cast<QuadData*>(d_quads);
	params.numQuads      = numQuads;
	params.materials    = reinterpret_cast<MaterialData*>(d_materials);
	params.numMaterials  = numMaterials;
	params.framebuffer  = reinterpret_cast<float3*>(d_framebuffer);
	params.width        = static_cast<unsigned int>(width);
	params.height       = static_cast<unsigned int>(height);
	params.camera       = camera;

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_launchParams_),
		&params,
		sizeof(SPPMLaunchParams),
		cudaMemcpyHostToDevice
	));

	OPTIX_CHECK(optixLaunch(
		pipeline_,
		stream_,
		d_launchParams_,
		sizeof(SPPMLaunchParams),
		&sbt_,
		static_cast<unsigned int>(width),
		static_cast<unsigned int>(height),
		1
	));

	CUDA_CHECK(cudaStreamSynchronize(stream_));

	CUDA_CHECK(cudaMemcpy(
		outputFramebuffer,
		reinterpret_cast<void*>(d_framebuffer),
		fbSize,
		cudaMemcpyDeviceToHost
	));

	cudaFree(reinterpret_cast<void*>(d_framebuffer));

	std::cout << "[SPPMPathTracer] renderTrivial: " << width << "x" << height << " done\n";
	return true;
}

void SPPMPathTracer::destroyProgramGroups() {
	auto destroy = [](OptixProgramGroup& pg) {
		if (pg) { optixProgramGroupDestroy(pg); pg = nullptr; }
	};
	destroy(raygenCameraPG_);
	destroy(missRadiancePG_);
	destroy(hitSpherePG_);
	destroy(hitQuadPG_);
}

void SPPMPathTracer::destroySBT() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_raygenRecord_);
	freeDev(d_missRecord_);
	freeDev(d_hitRecords_);
	sbt_ = {};
}

void SPPMPathTracer::cleanup() {
	destroySBT();
	destroyProgramGroups();
	if (pipeline_) { optixPipelineDestroy(pipeline_); pipeline_ = nullptr; }
	if (sppmModule_) { optixModuleDestroy(sppmModule_); sppmModule_ = nullptr; }
	if (d_launchParams_) { cudaFree(reinterpret_cast<void*>(d_launchParams_)); d_launchParams_ = 0; }
}

} // namespace optix_renderer
