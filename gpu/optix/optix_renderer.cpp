/// @file optix_renderer.cpp
/// @brief OptiX Renderer Implementation
/// @details Host-side OptiX context, pipeline, and rendering orchestration.
///          Manages CUDA/OptiX resources and executes path tracing on GPU.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
#include "optix_math_helpers.h"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>

namespace {
	/// Constants for OptiX configuration
	constexpr unsigned int kDefaultLogLevel = 3;  ///< OptiX log level (0=off, 4=verbose)
	constexpr int kDefaultCudaDevice = 0;         ///< Default CUDA device index
	constexpr size_t kMaxDeviceNameLength = 256;  ///< Max length for device name buffer
}

// ============================================================================
// SBT Record Structures (file scope for use across multiple functions)
// ============================================================================

/// @brief Generic SBT record with aligned header and user data
/// @tparam T User data type for this record
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord = SbtRecord<int>;         ///< Raygen program record
using MissRecord = SbtRecord<int>;           ///< Miss program record  
using HitGroupRecord = SbtRecord<HitGroupData>; ///< Hit group record

// ============================================================================
// Logging and Initialization
// ============================================================================

/// @brief Logging callback for OptiX messages
/// @param level Message severity level
/// @param tag Message category tag
/// @param message The log message
/// @param cbdata User callback data (unused)
static void contextLogCallback(
	unsigned int level,
	const char* tag,
	const char* message,
	void* /*cbdata*/
) {
	fprintf(stderr, "[OptiX][%u][%s]: %s\n", level, tag, message);
}

OptiXRenderer::OptiXRenderer() {
}

OptiXRenderer::~OptiXRenderer() {
	cleanup();
}

bool OptiXRenderer::isAvailable() noexcept {
	// Step 1: Check CUDA device availability
	int deviceCount = 0;
	const cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
	if (cudaErr != cudaSuccess || deviceCount == 0) {
		std::cerr << "[OptiX] No CUDA devices found. Error: "
				  << cudaGetErrorString(cudaErr) << "\n";
		return false;
	}

	// Step 2: Set CUDA device
	if (const cudaError_t setErr = cudaSetDevice(kDefaultCudaDevice); setErr != cudaSuccess) {
		std::cerr << "[OptiX] cudaSetDevice(" << kDefaultCudaDevice << ") failed: "
				  << cudaGetErrorString(setErr) << "\n";
		return false;
	}

	// Step 3: Initialize OptiX library (loads OptiX DLL)
	if (const OptixResult initRes = optixInit(); initRes != OPTIX_SUCCESS) {
		std::cerr << "[OptiX] optixInit failed: "
				  << optixGetErrorString(initRes) << "\n";
		return false;
	}

	// Step 4: Test OptiX context creation
	if (const CUresult cuInitErr = cuInit(0); cuInitErr != CUDA_SUCCESS) {
		std::cerr << "[OptiX] cuInit failed\n";
		return false;
	}

	CUdevice cuDevice;
	if (const CUresult devErr = cuDeviceGet(&cuDevice, kDefaultCudaDevice); devErr != CUDA_SUCCESS) {
		std::cerr << "[OptiX] cuDeviceGet failed\n";
		return false;
	}

	CUcontext cuCtx = nullptr;
	if (const CUresult ctxErr = cuDevicePrimaryCtxRetain(&cuCtx, cuDevice); ctxErr != CUDA_SUCCESS) {
		std::cerr << "[OptiX] cuDevicePrimaryCtxRetain failed\n";
		return false;
	}

	// Attempt to create OptiX device context
	OptixDeviceContext context = nullptr;
	OptixDeviceContextOptions options{};
	options.logCallbackFunction = &contextLogCallback;
	options.logCallbackLevel = kDefaultLogLevel;

	const OptixResult ctxCreateRes = optixDeviceContextCreate(cuCtx, &options, &context);

	// Cleanup temporary resources
	if (context) {
		optixDeviceContextDestroy(context);
	}
	if (cuCtx) {
		cuDevicePrimaryCtxRelease(cuDevice);
	}

	if (ctxCreateRes != OPTIX_SUCCESS) {
		std::cerr << "[OptiX] optixDeviceContextCreate failed: "
				  << optixGetErrorString(ctxCreateRes) << "\n";
		return false;
	}

	std::cout << "[OptiX] OptiX is available and functional!\n";
	return true;
}

bool OptiXRenderer::initialize() {
	if (!createContext()) {
		std::cerr << "Failed to create OptiX context\n";
		return false;
	}

	if (!createModule()) {
		std::cerr << "Failed to create OptiX module\n";
		return false;
	}

	if (!createProgramGroups()) {
		std::cerr << "Failed to create program groups\n";
		return false;
	}

	if (!linkPipeline()) {
		std::cerr << "Failed to link pipeline\n";
		return false;
	}

	// Allocate launch params buffer
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams_), sizeof(LaunchParams)));

	std::cout << "[OptiX] Renderer initialized successfully\n";
	return true;
}

/// @brief Create OptiX context and initialize CUDA resources
/// @return true if context created successfully, false otherwise
bool OptiXRenderer::createContext() {
	// Initialize CUDA Driver API
	CU_CHECK(cuInit(0));

	// Get CUDA device
	CUdevice cuDevice;
	CU_CHECK(cuDeviceGet(&cuDevice, kDefaultCudaDevice));

	// Query and display device name
	std::array<char, kMaxDeviceNameLength> deviceName{};
	cuDeviceGetName(deviceName.data(), static_cast<int>(deviceName.size()), cuDevice);
	std::cout << "[OptiX] Using GPU: " << deviceName.data() << "\n";

	// Retain primary CUDA context (modern approach for CUDA 13.2+)
	CU_CHECK(cuDevicePrimaryCtxRetain(&cudaContext_, cuDevice));

	// Create CUDA stream for asynchronous operations
	CUDA_CHECK(cudaStreamCreate(&stream_));

	// Initialize OptiX function table
	OPTIX_CHECK(optixInit());

	// Create OptiX device context with logging
	OptixDeviceContextOptions options{};
	options.logCallbackFunction = &contextLogCallback;
	options.logCallbackLevel = kDefaultLogLevel;

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext_, &options, &context_));

	return true;
}

/// @brief Load PTX shader and create OptiX module
/// @return true if module created successfully, false otherwise
bool OptiXRenderer::createModule() {
	// Load PTX shader code from file
	const std::string ptx = loadPTX("optix_programs.ptx");
	if (ptx.empty()) {
		std::cerr << "Failed to load PTX\n";
		return false;
	}

	// Module compile options
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

	// Pipeline compile options (store as member for use in linkPipeline)
	pipelineCompileOptions_.usesMotionBlur = false;
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions_.numPayloadValues = 12;  // attenuation(3) + emission(3) + dir(3) + seed(1) + flag(1) + t(1)
	pipelineCompileOptions_.numAttributeValues = 4;  // Sphere: center.xyz + radius (4 attrs)
	pipelineCompileOptions_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions_.pipelineLaunchParamsVariableName = "params";

	// Create module
	char log[2048];
	size_t logSize = sizeof(log);
	OPTIX_CHECK(optixModuleCreate(
		context_,
		&moduleCompileOptions,
		&pipelineCompileOptions_,
		ptx.c_str(),
		ptx.size(),
		log,
		&logSize,
		&module_
	));

	if (logSize > 1) {
		std::cout << "[OptiX] Module creation log:\n" << log << "\n";
	}

	return true;
}

bool OptiXRenderer::createProgramGroups() {
	char log[2048];
	size_t logSize;

	// Raygen program
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc raygenDesc = {};
	raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenDesc.raygen.module = module_;
	raygenDesc.raygen.entryFunctionName = "__raygen__rg";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&raygenDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&raygenPG_
	));

	// Miss program
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missDesc.miss.module = module_;
	missDesc.miss.entryFunctionName = "__miss__ms";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&missDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&missPG_
	));

	// Shadow miss program
	OptixProgramGroupDesc shadowMissDesc = {};
	shadowMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	shadowMissDesc.miss.module = module_;
	shadowMissDesc.miss.entryFunctionName = "__miss__shadow";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowMissDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowMissPG_
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
		context_,
		&sphereHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupSpherePG_
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
		context_,
		&quadHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupQuadPG_
	));

	// Shadow hit group for spheres (any-hit only, no closest-hit)
	OptixProgramGroupDesc shadowSphereHitDesc = {};
	shadowSphereHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowSphereHitDesc.hitgroup.moduleIS = module_;
	shadowSphereHitDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	shadowSphereHitDesc.hitgroup.moduleAH = module_;
	shadowSphereHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_sphere";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowSphereHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupSpherePG_
	));

	// Shadow hit group for quads (any-hit only, no closest-hit)
	OptixProgramGroupDesc shadowQuadHitDesc = {};
	shadowQuadHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowQuadHitDesc.hitgroup.moduleIS = module_;
	shadowQuadHitDesc.hitgroup.entryFunctionNameIS = "__intersection__quad";
	shadowQuadHitDesc.hitgroup.moduleAH = module_;
	shadowQuadHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_quad";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowQuadHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupQuadPG_
	));

	std::cout << "[OptiX] Created program groups: raygen, miss (radiance + shadow), sphere hit (radiance + shadow), quad hit (radiance + shadow)\n";
	return true;
}

bool OptiXRenderer::linkPipeline() {
	// Collect all program groups (radiance + shadow)
	OptixProgramGroup programGroups[] = {
		raygenPG_,
		missPG_,
		shadowMissPG_,
		hitgroupSpherePG_,
		hitgroupQuadPG_,
		shadowHitgroupSpherePG_,
		shadowHitgroupQuadPG_
	};

	// Pipeline link options
	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 2;  // Primary + shadow/indirect

	// Create pipeline
	char log[2048];
	size_t logSize = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context_,
		&pipelineCompileOptions_,  // Use member variable from createModule
		&pipelineLinkOptions,
		programGroups,
		sizeof(programGroups) / sizeof(programGroups[0]),
		log,
		&logSize,
		&pipeline_
	));

	if (logSize > 1) {
		std::cout << "[OptiX] Pipeline creation log:\n" << log << "\n";
	}

	// Set stack sizes
	OptixStackSizes stackSizes = {};
	for (auto pg : programGroups) {
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, pipeline_));
	}

	uint32_t maxTraceDepth = 10;  // Match renderer max depth
	uint32_t maxCCDepth = 0;
	uint32_t maxDCDepth = 0;
	uint32_t directCallableStackSizeFromTraversal;
	uint32_t directCallableStackSizeFromState;
	uint32_t continuationStackSize;

	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stackSizes,
		maxTraceDepth,
		maxCCDepth,
		maxDCDepth,
		&directCallableStackSizeFromTraversal,
		&directCallableStackSizeFromState,
		&continuationStackSize
	));

	OPTIX_CHECK(optixPipelineSetStackSize(
		pipeline_,
		directCallableStackSizeFromTraversal,
		directCallableStackSizeFromState,
		continuationStackSize,
		1  // maxTraversableGraphDepth (must be 1 for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
	));

	std::cout << "[OptiX] Pipeline linked successfully\n";
	return true;
}

bool OptiXRenderer::buildScene(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads,
	const std::vector<MaterialData>& materials,
	const std::vector<int>& lightIndices,
	const std::vector<bool>& isLightSphere
) {
	// Store material data on device
	numMaterials_ = static_cast<unsigned int>(materials.size());
	size_t materialSize = materials.size() * sizeof(MaterialData);

	if (d_materials_) {
		cudaFree(reinterpret_cast<void*>(d_materials_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materials_), materialSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_materials_),
		materials.data(),
		materialSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << materials.size() << " materials to GPU\n";

	// Store sphere data on device
	numSpheres_ = static_cast<unsigned int>(spheres.size());
	size_t sphereSize = spheres.size() * sizeof(SphereData);

	if (d_spheres_) {
		cudaFree(reinterpret_cast<void*>(d_spheres_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_spheres_), sphereSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_spheres_),
		spheres.data(),
		sphereSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << spheres.size() << " spheres to GPU\n";

	// Store quad data on device
	numQuads_ = static_cast<unsigned int>(quads.size());
	size_t quadSize = quads.size() * sizeof(QuadData);

	if (d_quads_) {
		cudaFree(reinterpret_cast<void*>(d_quads_));
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_quads_), quadSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_quads_),
		quads.data(),
		quadSize,
		cudaMemcpyHostToDevice
	));

	std::cout << "[OptiX] Uploaded " << quads.size() << " quads to GPU\n";

	// Store light data on device for MIS
	numLights_ = static_cast<unsigned int>(lightIndices.size());

	if (numLights_ > 0) {
		// Upload light indices
		size_t lightIndexSize = lightIndices.size() * sizeof(int);
		if (d_lightIndices_) {
			cudaFree(reinterpret_cast<void*>(d_lightIndices_));
		}
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lightIndices_), lightIndexSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_lightIndices_),
			lightIndices.data(),
			lightIndexSize,
			cudaMemcpyHostToDevice
		));

		// Upload light type flags (convert bool to int for better GPU alignment)
		std::vector<int> lightFlags(isLightSphere.size());
		for (size_t i = 0; i < isLightSphere.size(); ++i) {
			lightFlags[i] = isLightSphere[i] ? 1 : 0;
		}

		size_t lightFlagSize = lightFlags.size() * sizeof(int);
		if (d_isLightSphere_) {
			cudaFree(reinterpret_cast<void*>(d_isLightSphere_));
		}
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_isLightSphere_), lightFlagSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_isLightSphere_),
			lightFlags.data(),
			lightFlagSize,
			cudaMemcpyHostToDevice
		));

		std::cout << "[OptiX] Uploaded " << numLights_ << " light sources for MIS\n";
	} else {
		// No lights in scene
		if (d_lightIndices_) {
			cudaFree(reinterpret_cast<void*>(d_lightIndices_));
			d_lightIndices_ = 0;
		}
		if (d_isLightSphere_) {
			cudaFree(reinterpret_cast<void*>(d_isLightSphere_));
			d_isLightSphere_ = 0;
		}
		std::cout << "[OptiX] No emissive lights in scene\n";
	}

	// Build acceleration structure for custom primitives
	// We'll use AABB (axis-aligned bounding box) custom primitives

	size_t totalGeoms = spheres.size() + quads.size();
	std::vector<OptixAabb> aabbs;
	aabbs.reserve(totalGeoms);

	// Build AABBs for spheres
	for (const auto& s : spheres) {
		OptixAabb aabb;
		aabb.minX = s.center.x - s.radius;
		aabb.minY = s.center.y - s.radius;
		aabb.minZ = s.center.z - s.radius;
		aabb.maxX = s.center.x + s.radius;
		aabb.maxY = s.center.y + s.radius;
		aabb.maxZ = s.center.z + s.radius;
		aabbs.push_back(aabb);
	}

	// Build AABBs for quads
	for (const auto& q : quads) {
		// Quad corners: Q, Q+u, Q+v, Q+u+v
		float3 corners[4] = {
			q.Q,
			make_float3(q.Q.x + q.u.x, q.Q.y + q.u.y, q.Q.z + q.u.z),
			make_float3(q.Q.x + q.v.x, q.Q.y + q.v.y, q.Q.z + q.v.z),
			make_float3(q.Q.x + q.u.x + q.v.x, q.Q.y + q.u.y + q.v.y, q.Q.z + q.u.z + q.v.z)
		};

		OptixAabb aabb;
		aabb.minX = fminf(fminf(corners[0].x, corners[1].x), fminf(corners[2].x, corners[3].x));
		aabb.minY = fminf(fminf(corners[0].y, corners[1].y), fminf(corners[2].y, corners[3].y));
		aabb.minZ = fminf(fminf(corners[0].z, corners[1].z), fminf(corners[2].z, corners[3].z));
		aabb.maxX = fmaxf(fmaxf(corners[0].x, corners[1].x), fmaxf(corners[2].x, corners[3].x));
		aabb.maxY = fmaxf(fmaxf(corners[0].y, corners[1].y), fmaxf(corners[2].y, corners[3].y));
		aabb.maxZ = fmaxf(fmaxf(corners[0].z, corners[1].z), fmaxf(corners[2].z, corners[3].z));
		aabbs.push_back(aabb);
	}

	// Upload AABBs to device
	CUdeviceptr d_aabb;
	size_t aabbSize = aabbs.size() * sizeof(OptixAabb);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb), aabbSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_aabb),
		aabbs.data(),
		aabbSize,
		cudaMemcpyHostToDevice
	));

	// Build input for sphere geometry
	std::vector<uint32_t> sphere_flags(spheres.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput sphereBuildInput = {};
	sphereBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	sphereBuildInput.customPrimitiveArray.aabbBuffers = &d_aabb;
	sphereBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(spheres.size());
	sphereBuildInput.customPrimitiveArray.flags = sphere_flags.data();
	sphereBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all spheres
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for quad geometry
	CUdeviceptr d_quad_aabb = d_aabb + (spheres.size() * sizeof(OptixAabb));
	std::vector<uint32_t> quad_flags(quads.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput quadBuildInput = {};
	quadBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	quadBuildInput.customPrimitiveArray.aabbBuffers = &d_quad_aabb;
	quadBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(quads.size());
	quadBuildInput.customPrimitiveArray.flags = quad_flags.data();
	quadBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all quads
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Combine build inputs
	OptixBuildInput buildInputs[2] = { sphereBuildInput, quadBuildInput };
	unsigned int numBuildInputs = 0;

	// Only include non-empty geometry types
	if (!spheres.empty() && !quads.empty()) {
		numBuildInputs = 2;  // Both
	} else if (!spheres.empty()) {
		buildInputs[0] = sphereBuildInput;
		numBuildInputs = 1;  // Spheres only
	} else if (!quads.empty()) {
		buildInputs[0] = quadBuildInput;
		numBuildInputs = 1;  // Quads only
	} else {
		// Empty scene
		std::cerr << "[OptiX] Error: Scene contains no geometry" << std::endl;
		return false;
	}

	// Accel build options
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	// Query memory requirements
	OptixAccelBufferSizes gasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context_,
		&accelOptions,
		buildInputs,
		numBuildInputs,
		&gasBufferSizes
	));

	// Allocate temp and output buffers
	CUdeviceptr d_temp;
	CUdeviceptr d_gasOutput;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), gasBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput), gasBufferSizes.outputSizeInBytes));

	// Build acceleration structure
	OPTIX_CHECK(optixAccelBuild(
		context_,
		stream_,
		&accelOptions,
		buildInputs,
		numBuildInputs,
		d_temp,
		gasBufferSizes.tempSizeInBytes,
		d_gasOutput,
		gasBufferSizes.outputSizeInBytes,
		&gasHandle_,
		nullptr,  // No compacted size output (simplify for now)
		0
	));

	CUDA_CHECK(cudaStreamSynchronize(stream_));

	// Free temp buffer
	cudaFree(reinterpret_cast<void*>(d_temp));
	cudaFree(reinterpret_cast<void*>(d_aabb));

	// Store GAS buffer
	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));
	d_gas_ = d_gasOutput;

	std::cout << "[OptiX] Built acceleration structure: "
		<< spheres.size() << " spheres, "
		<< quads.size() << " quads\n";

	// Build Shader Binding Table (SBT)
	if (!buildSBT(spheres, quads)) {
		std::cerr << "Failed to build SBT\n";
		return false;
	}

	return true;
}

bool OptiXRenderer::buildSBT(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads
) {
	// Raygen record
	RaygenRecord raygenRecord;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &raygenRecord));
	raygenRecord.data = 0;  // No data

	if (d_raygenRecord_) cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord_), sizeof(RaygenRecord)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_raygenRecord_),
		&raygenRecord,
		sizeof(RaygenRecord),
		cudaMemcpyHostToDevice
	));

	// Miss records (radiance + shadow)
	std::vector<MissRecord> missRecords(2);

	// Radiance miss
	OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &missRecords[0]));
	missRecords[0].data = 0;

	// Shadow miss
	OPTIX_CHECK(optixSbtRecordPackHeader(shadowMissPG_, &missRecords[1]));
	missRecords[1].data = 0;

	if (d_missRecord_) cudaFree(reinterpret_cast<void*>(d_missRecord_));
	size_t missRecordSize = missRecords.size() * sizeof(MissRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord_), missRecordSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_missRecord_),
		missRecords.data(),
		missRecordSize,
		cudaMemcpyHostToDevice
	));

	// Hit group records - radiance + shadow for each geometry type
	// OptiX SBT layout: index = (build_input_index * RAY_TYPE_COUNT) + ray_type_index
	// With 2 build inputs (sphere=0, quad=1) and 2 ray types (radiance=0, shadow=1):
	// [0]: sphere + radiance (0*2+0=0)
	// [1]: sphere + shadow   (0*2+1=1)
	// [2]: quad + radiance   (1*2+0=2)
	// [3]: quad + shadow     (1*2+1=3)
	std::vector<HitGroupRecord> hitGroupRecords(4);

	// [0] Sphere radiance hit record
	OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG_, &hitGroupRecords[0]));
	hitGroupRecords[0].data = {};

	// [1] Sphere shadow hit record
	OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupSpherePG_, &hitGroupRecords[1]));
	hitGroupRecords[1].data = {};

	// [2] Quad radiance hit record
	OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupQuadPG_, &hitGroupRecords[2]));
	hitGroupRecords[2].data = {};

	// [3] Quad shadow hit record
	OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupQuadPG_, &hitGroupRecords[3]));
	hitGroupRecords[3].data = {};

	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));
	size_t hitRecordSize = hitGroupRecords.size() * sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecords_), hitRecordSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroupRecords_),
		hitGroupRecords.data(),
		hitRecordSize,
		cudaMemcpyHostToDevice
	));

	numHitRecords_ = 4;

	// Configure SBT
	sbt_.raygenRecord = d_raygenRecord_;
	sbt_.missRecordBase = d_missRecord_;
	sbt_.missRecordStrideInBytes = sizeof(MissRecord);
	sbt_.missRecordCount = 2;  // radiance + shadow
	sbt_.hitgroupRecordBase = d_hitgroupRecords_;
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
	sbt_.hitgroupRecordCount = 4;  // 2 geometry types × 2 ray types

	std::cout << "[OptiX] Built SBT: 2 miss records (radiance + shadow), 4 hit records (2 geom types × 2 ray types)\n";
	return true;
}

std::string OptiXRenderer::loadPTX(const char* filename) const {
	// Try loading from build output directory
	std::string paths[] = {
		std::string("gpu/optix/") + filename,
		std::string("optix_output/") + filename,
		std::string("./") + filename
	};

	for (const auto& path : paths) {
		std::ifstream file(path.c_str(), std::ios::binary);
		if (file.good()) {
			std::stringstream buffer;
			buffer << file.rdbuf();
			std::cout << "[OptiX] Loaded PTX from: " << path << "\n";
			return buffer.str();
		}
	}

	std::cerr << "[OptiX] Could not find PTX file: " << filename << "\n";
	return "";
}

bool OptiXRenderer::render(
	unsigned int width,
	unsigned int height,
	unsigned int samplesPerPixel,
	unsigned int maxDepth,
	const float* cameraOrigin,
	const float* cameraLowerLeft,
	const float* cameraHorizontal,
	const float* cameraVertical,
	float* outputFramebuffer
) {
	// Allocate framebuffer on device
	CUdeviceptr d_framebuffer;
	size_t fbSize = width * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_framebuffer), fbSize));

	// Setup launch params
	LaunchParams params;
	params.framebuffer = reinterpret_cast<float3*>(d_framebuffer);
	params.width = width;
	params.height = height;
	params.samplesPerPixel = samplesPerPixel;
	params.maxDepth = maxDepth;
	params.frameNumber = 0;  // Could be animated

	// Camera setup
	params.camera.origin = make_float3(cameraOrigin[0], cameraOrigin[1], cameraOrigin[2]);
	params.camera.lower_left_corner = make_float3(cameraLowerLeft[0], cameraLowerLeft[1], cameraLowerLeft[2]);
	params.camera.horizontal = make_float3(cameraHorizontal[0], cameraHorizontal[1], cameraHorizontal[2]);
	params.camera.vertical = make_float3(cameraVertical[0], cameraVertical[1], cameraVertical[2]);

	// Scene 
	params.traversable = gasHandle_;
	params.materials = reinterpret_cast<MaterialData*>(d_materials_);
	params.numMaterials = numMaterials_;
	params.spheres = reinterpret_cast<SphereData*>(d_spheres_);
	params.numSpheres = numSpheres_;
	params.quads = reinterpret_cast<QuadData*>(d_quads_);
	params.numQuads = numQuads_;

	// Light sampling for MIS
	params.lightIndices = reinterpret_cast<int*>(d_lightIndices_);
	params.numLights = numLights_;
	params.isLightSphere = reinterpret_cast<bool*>(d_isLightSphere_);

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

	// Download framebuffer
	CUDA_CHECK(cudaMemcpy(
		outputFramebuffer,
		reinterpret_cast<void*>(d_framebuffer),
		fbSize,
		cudaMemcpyDeviceToHost
	));

	// Cleanup framebuffer
	cudaFree(reinterpret_cast<void*>(d_framebuffer));

	std::cout << "[OptiX] Rendered " << width << "x" << height
		<< " @ " << samplesPerPixel << " spp\n";

	return true;
}

void OptiXRenderer::cleanup() noexcept {
	// Free SBT records
	if (d_raygenRecord_) cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
	if (d_missRecord_) cudaFree(reinterpret_cast<void*>(d_missRecord_));
	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));

	// Free acceleration structure
	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));

	// Free scene data
	if (d_materials_) cudaFree(reinterpret_cast<void*>(d_materials_));
	if (d_spheres_) cudaFree(reinterpret_cast<void*>(d_spheres_));
	if (d_quads_) cudaFree(reinterpret_cast<void*>(d_quads_));
	if (d_lightIndices_) cudaFree(reinterpret_cast<void*>(d_lightIndices_));
	if (d_isLightSphere_) cudaFree(reinterpret_cast<void*>(d_isLightSphere_));

	// Free launch params
	if (d_launchParams_) cudaFree(reinterpret_cast<void*>(d_launchParams_));

	// Destroy program groups
	if (raygenPG_) optixProgramGroupDestroy(raygenPG_);
	if (missPG_) optixProgramGroupDestroy(missPG_);
	if (hitgroupSpherePG_) optixProgramGroupDestroy(hitgroupSpherePG_);
	if (hitgroupQuadPG_) optixProgramGroupDestroy(hitgroupQuadPG_);

	// Destroy module and pipeline
	if (module_) optixModuleDestroy(module_);
	if (pipeline_) optixPipelineDestroy(pipeline_);

	// Destroy context
	if (context_) optixDeviceContextDestroy(context_);

	// Destroy CUDA resources
	if (stream_) cudaStreamDestroy(stream_);
	if (cudaContext_) cuCtxDestroy(cudaContext_);
}
