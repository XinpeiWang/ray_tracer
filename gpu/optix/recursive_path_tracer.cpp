#include "recursive_path_tracer.h"
#include "optix_types.h"
#include <iostream>
#include <cuda_runtime.h>

namespace optix_renderer {

RecursivePathTracer::RecursivePathTracer() {
}

RecursivePathTracer::~RecursivePathTracer() {
	cleanup();
}

bool RecursivePathTracer::initialize(
	OptixDeviceContext context,
	OptixModule module,
	CUstream stream
) {
	context_ = context;
	module_ = module;
	stream_ = stream;

	// Initialize pipeline compile options
	pipelineCompileOptions_.usesMotionBlur = false;
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions_.numPayloadValues = 12;  // PathTracingPayload uses 12 registers
	pipelineCompileOptions_.numAttributeValues = 4;  // Custom intersection attributes
	pipelineCompileOptions_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions_.pipelineLaunchParamsVariableName = "params";

	std::cout << "[RecursivePathTracer] Initialized\n";
	return true;
}

bool RecursivePathTracer::createProgramGroups() {
	// This implementation is extracted from OptixRenderer::createProgramGroups()
	// We keep the same logic but now it's part of the strategy

	char log[2048];
	size_t logSize;
	OptixProgramGroupOptions pgOptions = {};

	// Raygen program
	OptixProgramGroupDesc raygenDesc = {};
	raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenDesc.raygen.module = module_;
	raygenDesc.raygen.entryFunctionName = "__raygen__rg";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &raygenDesc, 1, &pgOptions,
		log, &logSize, &raygenPG_
	));

	// Miss program (radiance)
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missDesc.miss.module = module_;
	missDesc.miss.entryFunctionName = "__miss__ms";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &missDesc, 1, &pgOptions,
		log, &logSize, &missPG_
	));

	// Shadow miss program
	OptixProgramGroupDesc shadowMissDesc = {};
	shadowMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	shadowMissDesc.miss.module = module_;
	shadowMissDesc.miss.entryFunctionName = "__miss__shadow";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &shadowMissDesc, 1, &pgOptions,
		log, &logSize, &shadowMissPG_
	));

	// Sphere hit group (intersection + closest-hit)
	OptixProgramGroupDesc sphereHitDesc = {};
	sphereHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	sphereHitDesc.hitgroup.moduleIS = module_;
	sphereHitDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	sphereHitDesc.hitgroup.moduleCH = module_;
	sphereHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &sphereHitDesc, 1, &pgOptions,
		log, &logSize, &hitgroupSpherePG_
	));

	// Quad hit group (intersection + closest-hit)
	OptixProgramGroupDesc quadHitDesc = {};
	quadHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	quadHitDesc.hitgroup.moduleIS = module_;
	quadHitDesc.hitgroup.entryFunctionNameIS = "__intersection__quad";
	quadHitDesc.hitgroup.moduleCH = module_;
	quadHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__quad";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &quadHitDesc, 1, &pgOptions,
		log, &logSize, &hitgroupQuadPG_
	));

	// Shadow hit group for spheres (any-hit only)
	OptixProgramGroupDesc shadowSphereHitDesc = {};
	shadowSphereHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowSphereHitDesc.hitgroup.moduleIS = module_;
	shadowSphereHitDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	shadowSphereHitDesc.hitgroup.moduleAH = module_;
	shadowSphereHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_sphere";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &shadowSphereHitDesc, 1, &pgOptions,
		log, &logSize, &shadowHitgroupSpherePG_
	));

	// Shadow hit group for quads (any-hit only)
	OptixProgramGroupDesc shadowQuadHitDesc = {};
	shadowQuadHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowQuadHitDesc.hitgroup.moduleIS = module_;
	shadowQuadHitDesc.hitgroup.entryFunctionNameIS = "__intersection__quad";
	shadowQuadHitDesc.hitgroup.moduleAH = module_;
	shadowQuadHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_quad";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_, &shadowQuadHitDesc, 1, &pgOptions,
		log, &logSize, &shadowHitgroupQuadPG_
	));

	std::cout << "[RecursivePathTracer] Created 7 program groups\n";
	return true;
}

bool RecursivePathTracer::linkPipeline(unsigned int maxTraceDepth) {
	// Link all program groups into a pipeline
	OptixProgramGroup programGroups[] = {
		raygenPG_,
		missPG_,
		shadowMissPG_,
		hitgroupSpherePG_,
		hitgroupQuadPG_,
		shadowHitgroupSpherePG_,
		shadowHitgroupQuadPG_
	};

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = maxTraceDepth;

	char log[2048];
	size_t logSize = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context_,
		&pipelineCompileOptions_,
		&pipelineLinkOptions,
		programGroups,
		sizeof(programGroups) / sizeof(programGroups[0]),
		log,
		&logSize,
		&pipeline_
	));

	std::cout << "[RecursivePathTracer] Linked pipeline\n";
	return true;
}

bool RecursivePathTracer::buildSBT(unsigned int numSpheres, unsigned int numQuads) {
	// Implementation continues in next part...
	std::cout << "[RecursivePathTracer] Building SBT... (stub)\n";
	return true;
}

bool RecursivePathTracer::render(
	int width,
	int height,
	int samples_per_pixel,
	int max_depth,
	const float* camera_origin,
	const float* camera_lower_left,
	const float* camera_horizontal,
	const float* camera_vertical,
	float* framebuffer,
	OptixTraversableHandle gas_handle,
	CUdeviceptr d_materials,
	CUdeviceptr d_spheres,
	CUdeviceptr d_quads,
	CUdeviceptr d_light_indices,
	CUdeviceptr d_is_light_sphere,
	unsigned int num_materials,
	unsigned int num_spheres,
	unsigned int num_quads,
	unsigned int num_lights
) {
	std::cout << "[RecursivePathTracer] Render stub called\n";
	return true;
}

void RecursivePathTracer::cleanup() {
	destroyProgramGroups();
	destroySBT();

	if (pipeline_) {
		optixPipelineDestroy(pipeline_);
		pipeline_ = nullptr;
	}

	if (d_launchParams_) {
		cudaFree(reinterpret_cast<void*>(d_launchParams_));
		d_launchParams_ = 0;
	}
}

void RecursivePathTracer::destroyProgramGroups() {
	if (raygenPG_) { optixProgramGroupDestroy(raygenPG_); raygenPG_ = nullptr; }
	if (missPG_) { optixProgramGroupDestroy(missPG_); missPG_ = nullptr; }
	if (shadowMissPG_) { optixProgramGroupDestroy(shadowMissPG_); shadowMissPG_ = nullptr; }
	if (hitgroupSpherePG_) { optixProgramGroupDestroy(hitgroupSpherePG_); hitgroupSpherePG_ = nullptr; }
	if (hitgroupQuadPG_) { optixProgramGroupDestroy(hitgroupQuadPG_); hitgroupQuadPG_ = nullptr; }
	if (shadowHitgroupSpherePG_) { optixProgramGroupDestroy(shadowHitgroupSpherePG_); shadowHitgroupSpherePG_ = nullptr; }
	if (shadowHitgroupQuadPG_) { optixProgramGroupDestroy(shadowHitgroupQuadPG_); shadowHitgroupQuadPG_ = nullptr; }
}

void RecursivePathTracer::destroySBT() {
	if (d_raygenRecord_) {
		cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
		d_raygenRecord_ = 0;
	}
	if (d_missRecord_) {
		cudaFree(reinterpret_cast<void*>(d_missRecord_));
		d_missRecord_ = 0;
	}
	if (d_hitgroupRecords_) {
		cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));
		d_hitgroupRecords_ = 0;
	}
}

} // namespace optix_renderer
