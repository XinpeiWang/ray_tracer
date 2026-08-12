/// @file optix_renderer.cpp
/// @brief OptiX Renderer Implementation
/// @details Host-side OptiX context, pipeline, and rendering orchestration.
///          Manages CUDA/OptiX resources and executes path tracing on GPU.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
#include "optix_math_helpers.h"
#include "wavefront_path_tracer.h"
#include "sppm_path_tracer.h"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <cstring>

namespace {
	/// Constants for OptiX configuration
	constexpr unsigned int kDefaultLogLevel = 3;  ///< OptiX log level (0=off, 4=verbose)
	constexpr int kDefaultCudaDevice = 0;         ///< Default CUDA device index
	constexpr size_t kMaxDeviceNameLength = 256;  ///< Max length for device name buffer
}

// ============================================================================
// SBT Record Structures (file scope for use across multiple functions)
// ============================================================================

// SBT record types are defined in optix_types.h

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

	// Get CUDA device (stored in cuDevice_ -- cleanup() needs it to release
	// the primary context correctly, see that member's own comment).
	CU_CHECK(cuDeviceGet(&cuDevice_, kDefaultCudaDevice));

	// Query and display device name
	std::array<char, kMaxDeviceNameLength> deviceName{};
	cuDeviceGetName(deviceName.data(), static_cast<int>(deviceName.size()), cuDevice_);
	std::cout << "[OptiX] Using GPU: " << deviceName.data() << "\n";

	// Retain primary CUDA context (modern approach for CUDA 13.2+)
	CU_CHECK(cuDevicePrimaryCtxRetain(&cudaContext_, cuDevice_));

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

	// Pipeline compile options (store as member for use in linkPipeline).
	// usesMotionBlur is a pipeline-wide enable (needed for
	// optixGetRayTime()/optixTrace()'s time argument to do anything at all)
	// but is safe for every scene, motion or not: scenes without motion keep
	// their GAS single-key and their rays always carry time=0.0f (see
	// optix_raygen.h), so optixGetRayTime() simply always returns 0 for them
	// - a provable no-op, not just an assumption.
	pipelineCompileOptions_.usesMotionBlur = true;
	// Single-level instancing (one IAS wrapping exactly the 2 child GASes
	// built in buildAccelerationStructure - triangles can't share a GAS
	// with the custom AABB primitives, see that function's comment), not
	// ALLOW_SINGLE_GAS anymore.
	pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions_.numPayloadValues = 13;  // attenuation(3) + emission(3) + dir(3) + seed(1) + flag(1) + t(1) + brdf_or_light_pdf(1)
	pipelineCompileOptions_.numAttributeValues = 4;  // Sphere: center.xyz + radius (4 attrs)
	pipelineCompileOptions_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions_.pipelineLaunchParamsVariableName = "params";
	// Triangles use OptiX's built-in hardware-accelerated triangle geometry
	// (OPTIX_BUILD_INPUT_TYPE_TRIANGLES, see build_triangle_mesh_gpu below);
	// spheres/quads/bilinear-patches stay custom AABB primitives. Both flags
	// must be declared together since the GAS mixes both kinds of build
	// input - matches pbrt-v4's own GPU renderer, which also uses built-in
	// triangles alongside custom quadric/bilinear-patch primitives.
	pipelineCompileOptions_.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

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

	// Bilinear patch hit group (intersection + closest-hit)
	OptixProgramGroupDesc bilinearPatchHitDesc = {};
	bilinearPatchHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	bilinearPatchHitDesc.hitgroup.moduleIS = module_;
	bilinearPatchHitDesc.hitgroup.entryFunctionNameIS = "__intersection__bilinear_patch";
	bilinearPatchHitDesc.hitgroup.moduleCH = module_;
	bilinearPatchHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__bilinear_patch";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&bilinearPatchHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupBilinearPatchPG_
	));

	// Triangle hit group (closest-hit only - intersection is OptiX's
	// built-in hardware triangle test, no custom IS program bound).
	OptixProgramGroupDesc triangleHitDesc = {};
	triangleHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	triangleHitDesc.hitgroup.moduleCH = module_;
	triangleHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&triangleHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupTrianglePG_
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

	// Shadow hit group for bilinear patches (any-hit only, no closest-hit)
	OptixProgramGroupDesc shadowBilinearPatchHitDesc = {};
	shadowBilinearPatchHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowBilinearPatchHitDesc.hitgroup.moduleIS = module_;
	shadowBilinearPatchHitDesc.hitgroup.entryFunctionNameIS = "__intersection__bilinear_patch";
	shadowBilinearPatchHitDesc.hitgroup.moduleAH = module_;
	shadowBilinearPatchHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_bilinear_patch";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowBilinearPatchHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupBilinearPatchPG_
	));

	// Shadow hit group for triangles (any-hit only, no closest-hit, no
	// custom IS - built-in hardware triangle test, same as the radiance
	// hit group above).
	OptixProgramGroupDesc shadowTriangleHitDesc = {};
	shadowTriangleHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowTriangleHitDesc.hitgroup.moduleAH = module_;
	shadowTriangleHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_triangle";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowTriangleHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupTrianglePG_
	));

	std::cout << "[OptiX] Created program groups: raygen, miss (radiance + shadow), sphere hit (radiance + shadow), quad hit (radiance + shadow), bilinear patch hit (radiance + shadow), triangle hit (radiance + shadow)\n";
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
		hitgroupBilinearPatchPG_,
		hitgroupTrianglePG_,
		shadowHitgroupSpherePG_,
		shadowHitgroupQuadPG_,
		shadowHitgroupBilinearPatchPG_,
		shadowHitgroupTrianglePG_
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
		2  // maxTraversableGraphDepth: IAS -> GAS is 2 levels (single-level instancing)
	));

	std::cout << "[OptiX] Pipeline linked successfully\n";
	return true;
}

bool OptiXRenderer::buildScene(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads,
	const std::vector<MaterialData>& materials,
	const std::vector<int>& lightIndices,
	const std::vector<bool>& isLightSphere,
	const std::vector<PunctualLightGPU>& punctualLights,
	const std::vector<BilinearPatchData>& bilinearPatches,
	const std::vector<TriangleData>& triangles,
	const std::vector<GpuLensElement>& lensElements,
	const std::vector<GpuExitPupilBounds>& exitPupilBounds,
	const std::vector<TextureData>& textures,
	const std::vector<unsigned char>& texturePixels
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

	// Store texture metadata + shared pixel buffer on device. Both are
	// legitimately empty for most scenes (no textures at all) - guard the
	// malloc/memcpy rather than relying on cudaMalloc(0)'s behavior, same
	// caution already taken for bilinearPatches/triangles below.
	numTextures_ = static_cast<unsigned int>(textures.size());
	size_t textureSize = textures.size() * sizeof(TextureData);

	if (d_textures_) {
		cudaFree(reinterpret_cast<void*>(d_textures_));
		d_textures_ = 0;
	}
	if (!textures.empty()) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_textures_), textureSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_textures_),
			textures.data(),
			textureSize,
			cudaMemcpyHostToDevice
		));
	}

	if (d_texturePixels_) {
		cudaFree(reinterpret_cast<void*>(d_texturePixels_));
		d_texturePixels_ = 0;
	}
	if (!texturePixels.empty()) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturePixels_), texturePixels.size()));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_texturePixels_),
			texturePixels.data(),
			texturePixels.size(),
			cudaMemcpyHostToDevice
		));
	}

	if (!textures.empty())
		std::cout << "[OptiX] Uploaded " << textures.size() << " textures (" << texturePixels.size() << " pixel bytes) to GPU\n";

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

	// Store bilinear patch data on device
	numBilinearPatches_ = static_cast<unsigned int>(bilinearPatches.size());
	size_t bilinearPatchSize = bilinearPatches.size() * sizeof(BilinearPatchData);

	if (d_bilinearPatches_) {
		cudaFree(reinterpret_cast<void*>(d_bilinearPatches_));
		d_bilinearPatches_ = 0;
	}

	if (numBilinearPatches_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bilinearPatches_), bilinearPatchSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_bilinearPatches_),
			bilinearPatches.data(),
			bilinearPatchSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << bilinearPatches.size() << " bilinear patches to GPU\n";

	// Store triangle data on device
	numTriangles_ = static_cast<unsigned int>(triangles.size());
	size_t triangleSize = triangles.size() * sizeof(TriangleData);

	if (d_triangles_) {
		cudaFree(reinterpret_cast<void*>(d_triangles_));
		d_triangles_ = 0;
	}

	if (numTriangles_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles_), triangleSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_triangles_),
			triangles.data(),
			triangleSize,
			cudaMemcpyHostToDevice
		));
	}

	std::cout << "[OptiX] Uploaded " << triangles.size() << " triangles to GPU\n";

	// Store the RealisticCamera's host-precomputed (focus-adjusted) lens
	// table and exit-pupil bounds table on device. Both come from
	// scene_builder.cpp directly instantiating a host-side
	// RealisticCamera<float> - see optix_types.h's GpuLensElement/
	// GpuExitPupilBounds and render()'s camera-pointer-injection comment.
	numLensElements_ = static_cast<unsigned int>(lensElements.size());
	size_t lensElementSize = lensElements.size() * sizeof(GpuLensElement);

	if (d_lensElements_) {
		cudaFree(reinterpret_cast<void*>(d_lensElements_));
		d_lensElements_ = 0;
	}

	if (numLensElements_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lensElements_), lensElementSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_lensElements_),
			lensElements.data(),
			lensElementSize,
			cudaMemcpyHostToDevice
		));
	}

	numExitPupilBounds_ = static_cast<unsigned int>(exitPupilBounds.size());
	size_t exitPupilBoundsSize = exitPupilBounds.size() * sizeof(GpuExitPupilBounds);

	if (d_exitPupilBounds_) {
		cudaFree(reinterpret_cast<void*>(d_exitPupilBounds_));
		d_exitPupilBounds_ = 0;
	}

	if (numExitPupilBounds_ > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitPupilBounds_), exitPupilBoundsSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_exitPupilBounds_),
			exitPupilBounds.data(),
			exitPupilBoundsSize,
			cudaMemcpyHostToDevice
		));
	}

	if (numLensElements_ > 0)
		std::cout << "[OptiX] Uploaded " << lensElements.size() << " lens elements, "
			<< exitPupilBounds.size() << " exit-pupil bounds to GPU\n";

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

		// Build power-weighted alias table (pbrt-v4 PowerLightSampler / Vose method)
		// phi_i = area * luminance(emission) * pi  (matches CPU power_light_sampler.h)
		std::vector<float> powers(numLights_);
		for (unsigned int i = 0; i < numLights_; ++i) {
			int prim_idx = lightIndices[i];
			float3 emission = make_float3(0.f, 0.f, 0.f);
			float area = 1.0f;
			if (isLightSphere[i]) {
				const SphereData& s = spheres[prim_idx];
				const MaterialData& m = materials[s.materialIdx];
				emission = m.emission;
				area = 4.0f * 3.14159265f * s.radius * s.radius;  // surface area of sphere
			} else {
				const QuadData& q = quads[prim_idx];
				const MaterialData& m = materials[q.materialIdx];
				emission = m.emission;
				// area = |u x v|
				float3 cr = make_float3(
					q.u.y*q.v.z - q.u.z*q.v.y,
					q.u.z*q.v.x - q.u.x*q.v.z,
					q.u.x*q.v.y - q.u.y*q.v.x);
				area = sqrtf(cr.x*cr.x + cr.y*cr.y + cr.z*cr.z);
			}
			float lum = 0.2126f*emission.x + 0.7152f*emission.y + 0.0722f*emission.z;
			powers[i] = area * lum * 3.14159265f;  // phi = area * Le * pi
			if (powers[i] <= 0.f) powers[i] = 1e-6f;  // geometry-only target
		}

		// Vose alias method
		float total = 0.f;
		for (float p : powers) total += p;
		std::vector<GpuAliasEntry> table(numLights_);
		for (unsigned int i = 0; i < numLights_; ++i) {
			table[i].pdf = powers[i] / total;
			table[i].q   = powers[i] / total * float(numLights_);
			table[i].alias = (int)i;
		}
		std::vector<int> small_idx, large_idx;
		for (unsigned int i = 0; i < numLights_; ++i) {
			if (table[i].q < 1.f) small_idx.push_back(i);
			else                  large_idx.push_back(i);
		}
		while (!small_idx.empty() && !large_idx.empty()) {
			int s = small_idx.back(); small_idx.pop_back();
			int l = large_idx.back(); large_idx.pop_back();
			table[s].alias = l;
			table[l].q -= (1.f - table[s].q);
			if (table[l].q < 1.f) small_idx.push_back(l);
			else                  large_idx.push_back(l);
		}
		// Residuals: floating-point rounding may leave items in either list; set q=1 (pbrt-v4 pattern)
		while (!large_idx.empty()) { int l = large_idx.back(); large_idx.pop_back(); table[l].q = 1.f; table[l].alias = l; }
		while (!small_idx.empty()) { int s = small_idx.back(); small_idx.pop_back(); table[s].q = 1.f; table[s].alias = s; }

		size_t aliasTableSize = table.size() * sizeof(GpuAliasEntry);
		if (d_aliasTable_) cudaFree(reinterpret_cast<void*>(d_aliasTable_));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aliasTable_), aliasTableSize));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aliasTable_),
			table.data(), aliasTableSize, cudaMemcpyHostToDevice));
		std::cout << "[OptiX] Uploaded alias table (" << numLights_ << " entries) for power-weighted sampling\n";
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
		if (d_aliasTable_) {
			cudaFree(reinterpret_cast<void*>(d_aliasTable_));
			d_aliasTable_ = 0;
		}
		std::cout << "[OptiX] No emissive lights in scene\n";
	}

	// Store punctual (point/spot/distant) lights on device - separate from
	// the area-light arrays above, evaluated deterministically every hit
	// rather than selected via the alias table (see optix_device_helpers.h
	// eval_punctual_light / add_punctual_lights_lambertian).
	numPunctualLights_ = static_cast<unsigned int>(punctualLights.size());
	if (d_punctualLights_) {
		cudaFree(reinterpret_cast<void*>(d_punctualLights_));
		d_punctualLights_ = 0;
	}
	if (numPunctualLights_ > 0) {
		size_t punctualSize = punctualLights.size() * sizeof(PunctualLightGPU);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_punctualLights_), punctualSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_punctualLights_),
			punctualLights.data(),
			punctualSize,
			cudaMemcpyHostToDevice
		));
		std::cout << "[OptiX] Uploaded " << numPunctualLights_ << " punctual (point/spot/distant) lights\n";
	}

	// Build acceleration structure for custom primitives
	// We'll use AABB (axis-aligned bounding box) custom primitives

	if (spheres.empty() && quads.empty() && bilinearPatches.empty() && triangles.empty()) {
		std::cerr << "[OptiX] Error: Scene contains no geometry" << std::endl;
		return false;
	}

	// Triangles are excluded from this count - they don't go through the
	// AABB-based custom-primitive path below (see the comment further down).
	size_t totalAabbGeoms = spheres.size() + quads.size() + bilinearPatches.size();
	std::vector<OptixAabb> aabbs;
	aabbs.reserve(totalAabbGeoms);

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

	// Build AABBs for bilinear patches. The patch's surface (a bilinear/convex
	// combination of the 4 corners for (u,v) in [0,1]^2) always lies within
	// the convex hull of its corners, so a tight min/max of the 4 raw corners
	// is already a valid conservative bound - no epsilon slop needed (matches
	// bilinear_patch_hittable's CPU bounding_box, minus its +-0.01 slop).
	for (const auto& p : bilinearPatches) {
		OptixAabb aabb;
		aabb.minX = fminf(fminf(p.p00.x, p.p10.x), fminf(p.p01.x, p.p11.x));
		aabb.minY = fminf(fminf(p.p00.y, p.p10.y), fminf(p.p01.y, p.p11.y));
		aabb.minZ = fminf(fminf(p.p00.z, p.p10.z), fminf(p.p01.z, p.p11.z));
		aabb.maxX = fmaxf(fmaxf(p.p00.x, p.p10.x), fmaxf(p.p01.x, p.p11.x));
		aabb.maxY = fmaxf(fmaxf(p.p00.y, p.p10.y), fmaxf(p.p01.y, p.p11.y));
		aabb.maxZ = fmaxf(fmaxf(p.p00.z, p.p10.z), fmaxf(p.p01.z, p.p11.z));
		aabbs.push_back(aabb);
	}

	// Triangles are NOT included in this combined AABB buffer - they use
	// OptiX's built-in hardware triangle geometry (OPTIX_BUILD_INPUT_TYPE_
	// TRIANGLES, see the vertex-buffer upload and triBuildInput below),
	// which needs a vertex buffer, not host-computed AABBs (OptiX derives
	// its own bounds internally from the vertex data during the accel
	// build). This matches pbrt-v4's own GPU renderer, which also gives
	// triangles OptiX's native path while keeping quadrics/bilinear-patches
	// as custom AABB primitives like `totalAabbGeoms`'s other three types here.

	// Motion blur: true if any sphere's ray-time-t=1 position differs from
	// its t=0 position (see SphereData::center1's doc comment). Detected
	// here, not passed in as a parameter, so any future scene with moving
	// spheres gets motion support automatically just by setting center1.
	sceneHasMotion_ = false;
	for (const auto& s : spheres) {
		if (s.center1.x != s.center.x || s.center1.y != s.center.y || s.center1.z != s.center.z) {
			sceneHasMotion_ = true;
			break;
		}
	}

	// A second set of AABBs at ray-time t=1, only built when the scene
	// actually has motion. Quad/bilinear-patch/triangle geometry never
	// moves, so their t=1 AABB is identical to their t=0 one - motion keys
	// apply per accel-structure build (shared across every build input in
	// it, since OptiX requires all build inputs in one accelBuild() call to
	// use the same key count), not per build-input, so every build input in
	// a motion-enabled GAS must supply 2 keys even if only spheres move.
	std::vector<OptixAabb> aabbsKey1;
	if (sceneHasMotion_) {
		aabbsKey1.reserve(totalAabbGeoms);
		for (const auto& s : spheres) {
			OptixAabb aabb;
			aabb.minX = s.center1.x - s.radius;
			aabb.minY = s.center1.y - s.radius;
			aabb.minZ = s.center1.z - s.radius;
			aabb.maxX = s.center1.x + s.radius;
			aabb.maxY = s.center1.y + s.radius;
			aabb.maxZ = s.center1.z + s.radius;
			aabbsKey1.push_back(aabb);
		}
		// Static primitives: duplicate the t=0 AABBs already computed above
		// (at the same relative offsets in `aabbs`, right after the spheres).
		aabbsKey1.insert(aabbsKey1.end(), aabbs.begin() + spheres.size(), aabbs.end());
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

	// Key-1 (t=1) buffer - only allocated/uploaded when the scene has
	// motion. Left at 0 otherwise; every use of it below is either gated on
	// sceneHasMotion_ or never dereferenced by OptiX when numKeys<2 (see the
	// per-build-input aabbBuffers arrays), so a "0 + offset" placeholder
	// value in the unused case is inert - CUdeviceptr is an integer handle,
	// not a real pointer, so this arithmetic is well-defined either way.
	CUdeviceptr d_aabbKey1 = 0;
	if (sceneHasMotion_) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabbKey1), aabbSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_aabbKey1),
			aabbsKey1.data(),
			aabbSize,
			cudaMemcpyHostToDevice
		));
	}

	// Build input for sphere geometry. OptiX validation rejects a non-null
	// aabbBuffers when numPrimitives==0 ("numPrimitives is zero, but
	// aabbBuffers is non-null") even though it would never be dereferenced -
	// null out the pointer itself for empty types rather than just leaving
	// numPrimitives at 0.
	CUdeviceptr d_sphere_aabb_keys[2] = { d_aabb, d_aabbKey1 };
	std::vector<uint32_t> sphere_flags(spheres.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput sphereBuildInput = {};
	sphereBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	sphereBuildInput.customPrimitiveArray.aabbBuffers = spheres.empty() ? nullptr : d_sphere_aabb_keys;
	sphereBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(spheres.size());
	sphereBuildInput.customPrimitiveArray.flags = sphere_flags.data();
	sphereBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all spheres
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	sphereBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for quad geometry
	CUdeviceptr d_quad_aabb_keys[2] = {
		d_aabb + (spheres.size() * sizeof(OptixAabb)),
		d_aabbKey1 + (spheres.size() * sizeof(OptixAabb))
	};
	std::vector<uint32_t> quad_flags(quads.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput quadBuildInput = {};
	quadBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	quadBuildInput.customPrimitiveArray.aabbBuffers = quads.empty() ? nullptr : d_quad_aabb_keys;
	quadBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(quads.size());
	quadBuildInput.customPrimitiveArray.flags = quad_flags.data();
	quadBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all quads
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	quadBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for bilinear patch geometry
	CUdeviceptr d_blp_aabb_keys[2] = {
		d_aabb + ((spheres.size() + quads.size()) * sizeof(OptixAabb)),
		d_aabbKey1 + ((spheres.size() + quads.size()) * sizeof(OptixAabb))
	};
	std::vector<uint32_t> blp_flags(bilinearPatches.size(), OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput blpBuildInput = {};
	blpBuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	blpBuildInput.customPrimitiveArray.aabbBuffers = bilinearPatches.empty() ? nullptr : d_blp_aabb_keys;
	blpBuildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(bilinearPatches.size());
	blpBuildInput.customPrimitiveArray.flags = blp_flags.data();
	blpBuildInput.customPrimitiveArray.numSbtRecords = 1;  // Single hit group for all bilinear patches
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
	blpBuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

	// Build input for triangle geometry - OptiX's built-in triangle type,
	// not a custom AABB primitive (see the comment above the AABB loops).
	// Vertex buffer is a flat triangle soup (3 vertices per TriangleData,
	// in order, no index buffer) - primitive i's 3 vertices are exactly
	// triangles[i].p0/p1/p2, so optixGetPrimitiveIndex() in the closest-hit/
	// any-hit programs keeps indexing params.triangles[primIdx] unchanged.
	CUdeviceptr d_triVertices = 0;
	std::vector<float3> triVertsHost;
	if (!triangles.empty()) {
		triVertsHost.reserve(triangles.size() * 3);
		for (const auto& t : triangles) {
			triVertsHost.push_back(t.p0);
			triVertsHost.push_back(t.p1);
			triVertsHost.push_back(t.p2);
		}
		size_t triVertsSize = triVertsHost.size() * sizeof(float3);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triVertices), triVertsSize));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_triVertices),
			triVertsHost.data(),
			triVertsSize,
			cudaMemcpyHostToDevice
		));
	}
	// Triangles get their own GAS (see the "two GASes + one IAS" comment
	// below) built with motionOptions.numKeys=0 always, since triangles
	// never move - so unlike the AABB types above, no duplicate-for-key-1
	// buffer is needed here, just a single vertex buffer.
	CUdeviceptr d_tri_vertex_keys[1] = { d_triVertices };
	// flags[] for a triangle build input is indexed per SBT record (here:
	// just 1, since numSbtRecords=1), NOT per primitive like the AABB
	// custom-primitive arrays above - a single entry is correct.
	std::vector<uint32_t> tri_flags(1, OPTIX_GEOMETRY_FLAG_NONE);
	OptixBuildInput triBuildInput = {};
	triBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triBuildInput.triangleArray.vertexBuffers = triangles.empty() ? nullptr : d_tri_vertex_keys;
	triBuildInput.triangleArray.numVertices = static_cast<unsigned int>(triangles.size() * 3);
	triBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triBuildInput.triangleArray.indexBuffer = 0;  // implicit: every 3 vertices form one triangle
	triBuildInput.triangleArray.flags = tri_flags.data();
	triBuildInput.triangleArray.numSbtRecords = 1;  // Single hit group for all triangles
	triBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	// Two GASes + one IAS: OptiX forbids mixing OPTIX_BUILD_INPUT_TYPE_
	// TRIANGLES and OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES build inputs in
	// a single GAS (confirmed via NVIDIA's own forums - "you cannot mix
	// different primitive types in a single GAS"), so triangles get their
	// own GAS, spheres/quads/bilinear-patches keep sharing the other one
	// (same conditional-inclusion, same relative [sphere, quad, bilinear
	// patch] order as before), and a top-level IAS with 2 static-identity
	// instances combines them into the single traversable
	// trace_shadow_ray()/the radiance loop both already use. Each
	// instance's sbtOffset shifts that whole child GAS's hit-group records
	// to its own region of the flat SBT array buildSBT() builds below -
	// custom-prim types keep the exact same indices they always had
	// (instance 0's sbtOffset=0), triangles land right after them (instance
	// 1's sbtOffset = however many hit records the custom-prim types
	// occupy) - so buildSBT()'s own record layout doesn't change at all,
	// only how the offset into it is supplied (per-instance now, instead of
	// via build-input position within one shared GAS).
	std::vector<OptixBuildInput> customBuildInputVec;
	if (!spheres.empty()) customBuildInputVec.push_back(sphereBuildInput);
	if (!quads.empty()) customBuildInputVec.push_back(quadBuildInput);
	if (!bilinearPatches.empty()) customBuildInputVec.push_back(blpBuildInput);

	OptixAccelBuildOptions customAccelOptions = {};
	customAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	customAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	// numKeys<2 means "no motion" (OptiX treats 0 and 1 identically) - only
	// scenes with moving spheres (sceneHasMotion_) pay for real motion keys.
	customAccelOptions.motionOptions.numKeys = sceneHasMotion_ ? 2 : 0;
	customAccelOptions.motionOptions.timeBegin = 0.0f;
	customAccelOptions.motionOptions.timeEnd = 1.0f;
	customAccelOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;

	// g_renderer (optix_interface.cpp) is a process-lifetime singleton reused
	// across scene switches, not reconstructed per scene - so gasCustomHandle_/
	// gasTriHandle_ are members that can carry a stale value from whichever
	// PREVIOUS scene actually had that geometry type, if the CURRENT scene
	// doesn't. Reset both unconditionally before the conditional build blocks
	// below: without this, e.g. switching from a triangle-mesh scene to one
	// with none would leave gasTriHandle_ pointing at a GAS whose device
	// memory this same function already frees a few lines down (d_gasTri_),
	// and the IAS build after would wire that dangling handle into a live
	// instance - undefined behavior at trace time, not something that fails
	// loudly here.
	gasCustomHandle_ = 0;
	gasTriHandle_ = 0;

	CUdeviceptr d_gasCustomOutput = 0;
	if (!customBuildInputVec.empty()) {
		OptixAccelBufferSizes customBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &customAccelOptions, customBuildInputVec.data(),
			static_cast<unsigned int>(customBuildInputVec.size()), &customBufferSizes
		));

		CUdeviceptr d_customTemp;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_customTemp), customBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasCustomOutput), customBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &customAccelOptions, customBuildInputVec.data(),
			static_cast<unsigned int>(customBuildInputVec.size()),
			d_customTemp, customBufferSizes.tempSizeInBytes,
			d_gasCustomOutput, customBufferSizes.outputSizeInBytes,
			&gasCustomHandle_, nullptr, 0
		));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_customTemp));
	}
	cudaFree(reinterpret_cast<void*>(d_aabb));
	cudaFree(reinterpret_cast<void*>(d_aabbKey1));  // cudaFree(0) is a documented no-op when motion wasn't used

	// Triangle GAS - always static (numKeys=0): triangles never move
	// regardless of sceneHasMotion_ (that flag only tracks moving spheres).
	OptixAccelBuildOptions triAccelOptions = {};
	triAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	triAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	triAccelOptions.motionOptions.numKeys = 0;

	CUdeviceptr d_gasTriOutput = 0;
	if (!triangles.empty()) {
		OptixAccelBufferSizes triBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context_, &triAccelOptions, &triBuildInput, 1, &triBufferSizes
		));

		CUdeviceptr d_triTemp;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triTemp), triBufferSizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasTriOutput), triBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			context_, stream_, &triAccelOptions, &triBuildInput, 1,
			d_triTemp, triBufferSizes.tempSizeInBytes,
			d_gasTriOutput, triBufferSizes.outputSizeInBytes,
			&gasTriHandle_, nullptr, 0
		));
		CUDA_CHECK(cudaStreamSynchronize(stream_));
		cudaFree(reinterpret_cast<void*>(d_triTemp));
	}
	cudaFree(reinterpret_cast<void*>(d_triVertices));  // cudaFree(0) is a no-op when there were no triangles

	if (d_gasCustom_) cudaFree(reinterpret_cast<void*>(d_gasCustom_));
	d_gasCustom_ = d_gasCustomOutput;
	if (d_gasTri_) cudaFree(reinterpret_cast<void*>(d_gasTri_));
	d_gasTri_ = d_gasTriOutput;

	// Top-level IAS: one static-identity instance per non-empty child GAS.
	const int numCustomPrimSbtRecords = 2 * (int)(!spheres.empty() + !quads.empty() + !bilinearPatches.empty());
	static const float kIdentity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
	std::vector<OptixInstance> instances;
	if (gasCustomHandle_) {
		OptixInstance inst{};
		memcpy(inst.transform, kIdentity, sizeof(kIdentity));
		inst.instanceId = 0;
		inst.sbtOffset = 0;
		inst.visibilityMask = 255;
		inst.flags = OPTIX_INSTANCE_FLAG_NONE;
		inst.traversableHandle = gasCustomHandle_;
		instances.push_back(inst);
	}
	if (gasTriHandle_) {
		OptixInstance inst{};
		memcpy(inst.transform, kIdentity, sizeof(kIdentity));
		inst.instanceId = 1;
		inst.sbtOffset = numCustomPrimSbtRecords;
		inst.visibilityMask = 255;
		inst.flags = OPTIX_INSTANCE_FLAG_NONE;
		inst.traversableHandle = gasTriHandle_;
		instances.push_back(inst);
	}

	CUdeviceptr d_instances;
	size_t instancesSize = instances.size() * sizeof(OptixInstance);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instancesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), instancesSize, cudaMemcpyHostToDevice));

	OptixBuildInput iasBuildInput = {};
	iasBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	iasBuildInput.instanceArray.instances = d_instances;
	iasBuildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

	OptixAccelBuildOptions iasAccelOptions = {};
	iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	iasAccelOptions.motionOptions.numKeys = 0;  // Instance transforms are static; motion lives inside the custom-prim GAS

	OptixAccelBufferSizes iasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context_, &iasAccelOptions, &iasBuildInput, 1, &iasBufferSizes
	));

	CUdeviceptr d_iasTemp;
	CUdeviceptr d_iasOutput;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iasTemp), iasBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iasOutput), iasBufferSizes.outputSizeInBytes));

	OPTIX_CHECK(optixAccelBuild(
		context_, stream_, &iasAccelOptions, &iasBuildInput, 1,
		d_iasTemp, iasBufferSizes.tempSizeInBytes,
		d_iasOutput, iasBufferSizes.outputSizeInBytes,
		&gasHandle_, nullptr, 0
	));
	CUDA_CHECK(cudaStreamSynchronize(stream_));
	cudaFree(reinterpret_cast<void*>(d_iasTemp));
	cudaFree(reinterpret_cast<void*>(d_instances));

	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));
	d_gas_ = d_iasOutput;

	std::cout << "[OptiX] Built acceleration structure: "
		<< spheres.size() << " spheres, "
		<< quads.size() << " quads, "
		<< bilinearPatches.size() << " bilinear patches, "
		<< triangles.size() << " triangles\n";

	// Build Shader Binding Table (SBT)
	if (!buildSBT(spheres, quads, bilinearPatches, triangles)) {
		std::cerr << "Failed to build SBT\n";
		return false;
	}

	return true;
}

bool OptiXRenderer::buildSBT(
	const std::vector<SphereData>& spheres,
	const std::vector<QuadData>& quads,
	const std::vector<BilinearPatchData>& bilinearPatches,
	const std::vector<TriangleData>& triangles
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

	// Hit group records - radiance + shadow for each geometry type PRESENT in
	// the scene. OptiX SBT layout: index = (build_input_index * RAY_TYPE_COUNT)
	// + ray_type_index, where build_input_index is the geometry type's
	// POSITION among the non-empty build inputs in buildScene() (empty types
	// are omitted from that array entirely - see its comment) - so this must
	// emit exactly one (radiance, shadow) record pair per present type, in
	// the same relative [sphere, quad, bilinear patch, triangle] order, with
	// no gaps for absent types.
	std::vector<HitGroupRecord> hitGroupRecords;
	if (!spheres.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupSpherePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!quads.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupQuadPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupQuadPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!bilinearPatches.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupBilinearPatchPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupBilinearPatchPG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}
	if (!triangles.empty()) {
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
		hitGroupRecords.emplace_back();
		OPTIX_CHECK(optixSbtRecordPackHeader(shadowHitgroupTrianglePG_, &hitGroupRecords.back()));
		hitGroupRecords.back().data = {};
	}

	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));
	size_t hitRecordSize = hitGroupRecords.size() * sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecords_), hitRecordSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroupRecords_),
		hitGroupRecords.data(),
		hitRecordSize,
		cudaMemcpyHostToDevice
	));

	numHitRecords_ = hitGroupRecords.size();

	// Configure SBT
	sbt_.raygenRecord = d_raygenRecord_;
	sbt_.missRecordBase = d_missRecord_;
	sbt_.missRecordStrideInBytes = sizeof(MissRecord);
	sbt_.missRecordCount = 2;  // radiance + shadow
	sbt_.hitgroupRecordBase = d_hitgroupRecords_;
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
	sbt_.hitgroupRecordCount = static_cast<unsigned int>(hitGroupRecords.size());

	std::cout << "[OptiX] Built SBT: 2 miss records (radiance + shadow), "
		<< hitGroupRecords.size() << " hit records (one radiance+shadow pair per present geometry type)\n";
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

	// Delegate to WavefrontPathTracer if enabled
	if (useWavefront_ && wavefrontTracer_) {
		return wavefrontTracer_->render(
			(int)width, (int)height, (int)samplesPerPixel, (int)maxDepth,
			gpuCam,
			outputFramebuffer,
			gasHandle_,
			d_materials_, d_spheres_, d_quads_,
			d_lightIndices_, d_isLightSphere_, d_aliasTable_,
			numMaterials_, numSpheres_, numQuads_, numLights_,
			d_punctualLights_, numPunctualLights_,
			d_bilinearPatches_, numBilinearPatches_,
			d_triangles_, numTriangles_);
	}

	// Allocate framebuffer on device
	CUdeviceptr d_framebuffer;
	size_t fbSize = width * height * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_framebuffer), fbSize));

	// Setup launch params. Zero-initialised on purpose: LaunchParams is a POD
	// whose fields are assigned one by one below, so any field NOT assigned
	// here would otherwise hold stack garbage. instanceTriBase is read on the
	// device as "null means no instancing", and a garbage pointer there is an
	// out-of-bounds read inside a hit program - among the hardest bugs to see.
	LaunchParams params = {};
	params.framebuffer = reinterpret_cast<float3*>(d_framebuffer);
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
	params.textures = reinterpret_cast<TextureData*>(d_textures_);
	params.numTextures = numTextures_;
	params.texturePixels = reinterpret_cast<unsigned char*>(d_texturePixels_);
	params.spheres = reinterpret_cast<SphereData*>(d_spheres_);
	params.numSpheres = numSpheres_;
	params.quads = reinterpret_cast<QuadData*>(d_quads_);
	params.numQuads = numQuads_;
	params.bilinearPatches = reinterpret_cast<BilinearPatchData*>(d_bilinearPatches_);
	params.numBilinearPatches = numBilinearPatches_;
	params.triangles = reinterpret_cast<TriangleData*>(d_triangles_);
	params.instanceTriBase = nullptr;   // set only by scenes with instances
	params.numTriangles = numTriangles_;

	// Light sampling for MIS
	params.lightIndices = reinterpret_cast<int*>(d_lightIndices_);
	params.numLights = numLights_;
	params.isLightSphere = reinterpret_cast<bool*>(d_isLightSphere_);
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

	// Free SBT records
	if (d_raygenRecord_) cudaFree(reinterpret_cast<void*>(d_raygenRecord_));
	if (d_missRecord_) cudaFree(reinterpret_cast<void*>(d_missRecord_));
	if (d_hitgroupRecords_) cudaFree(reinterpret_cast<void*>(d_hitgroupRecords_));

	// Free acceleration structures (top-level IAS + the two child GASes)
	if (d_gas_) cudaFree(reinterpret_cast<void*>(d_gas_));
	if (d_gasCustom_) cudaFree(reinterpret_cast<void*>(d_gasCustom_));
	if (d_gasTri_) cudaFree(reinterpret_cast<void*>(d_gasTri_));

	// Free scene data
	if (d_materials_) cudaFree(reinterpret_cast<void*>(d_materials_));
	if (d_textures_) cudaFree(reinterpret_cast<void*>(d_textures_));
	if (d_texturePixels_) cudaFree(reinterpret_cast<void*>(d_texturePixels_));
	if (d_spheres_) cudaFree(reinterpret_cast<void*>(d_spheres_));
	if (d_quads_) cudaFree(reinterpret_cast<void*>(d_quads_));
	if (d_bilinearPatches_) cudaFree(reinterpret_cast<void*>(d_bilinearPatches_));
	if (d_triangles_) cudaFree(reinterpret_cast<void*>(d_triangles_));
	if (d_lensElements_) cudaFree(reinterpret_cast<void*>(d_lensElements_));
	if (d_exitPupilBounds_) cudaFree(reinterpret_cast<void*>(d_exitPupilBounds_));
	if (d_lightIndices_) cudaFree(reinterpret_cast<void*>(d_lightIndices_));
	if (d_isLightSphere_) cudaFree(reinterpret_cast<void*>(d_isLightSphere_));
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
		if (!wavefrontTracer_->buildSBT(numSpheres_, numQuads_, numBilinearPatches_, numTriangles_)) {
			std::cerr << "[OptiXRenderer] WavefrontPathTracer::buildSBT failed\n";
			wavefrontTracer_.reset();
			useWavefront_ = false;
			return;
		}
		std::cout << "[OptiXRenderer] WavefrontPathTracer ready\n";
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
		d_lightIndices_, d_isLightSphere_, d_aliasTable_, numLights_,
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
		d_lightIndices_, d_isLightSphere_, d_aliasTable_, numLights_);
}

