/// @file optix_renderer_init.cpp
/// @brief OptiX Renderer Implementation -- context/pipeline setup.
/// @details Split out of optix_renderer.cpp (see optix_renderer_scene.cpp
///          and optix_renderer_render.cpp for the other two thirds, and
///          optix_renderer.h for the shared class declaration). This part
///          owns device availability probing, OptiX context/module/pipeline
///          creation, and PTX loading -- everything a scene build needs
///          already in place. Also the sole owner of
///          <optix_function_table_definition.h>, which must be included in
///          exactly one of these 3 translation units (it defines the global
///          OptiX function table that <optix_stubs.h>'s inline wrappers call
///          through) -- do not add it to the other two files.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"
// wavefront_path_tracer.h/sppm_path_tracer.h: not otherwise needed here, but
// ~OptiXRenderer() (defined in this file) implicitly destroys the
// unique_ptr<WavefrontPathTracer>/unique_ptr<SPPMPathTracer> members, which
// requires both types complete at the point of definition.
#include "wavefront_path_tracer.h"
#include "sppm_path_tracer.h"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <algorithm>     // std::min, for getDiagnostics()'s bounded string copy
#include <cstdlib>       // getenv, for RAY_TRACER_OPTIX_VALIDATION

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
/// @param level Message severity level (1=fatal, 2=error, 3=warning, 4=print)
/// @param tag Message category tag
/// @param message The log message
/// @param cbdata The OptiXRenderer instance passed via
///               OptixDeviceContextOptions::logCallbackData at context
///               creation - null for isAvailable()'s throwaway probe
///               context, which doesn't track issues.
void contextLogCallback(
	unsigned int level,
	const char* tag,
	const char* message,
	void* cbdata
) {
	fprintf(stderr, "[OptiX][%u][%s]: %s\n", level, tag, message);
	if (level <= 3) {
		if (auto* self = static_cast<OptiXRenderer*>(cbdata)) {
			self->loggedIssues_.push_back(
				std::string("[") + tag + "] " + message);
		}
	}
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

bool OptiXRenderer::getDiagnostics(OptixDiagnostics& out) noexcept {
	out = OptixDiagnostics{};
	out.available = false;

	auto fail = [&](const char* reason) {
		// std::string::copy instead of strncpy - avoids MSVC's C4996
		// "unsafe function" flag on strncpy without needing
		// _CRT_SECURE_NO_WARNINGS or a Windows-only strncpy_s branch.
		std::string s(reason);
		size_t n = (std::min)(s.size(), sizeof(out.failure_reason) - 1);
		s.copy(out.failure_reason, n);
		out.failure_reason[n] = '\0';
		return false;
	};

	// Step 1: Check CUDA device availability (mirrors isAvailable() exactly -
	// see that function's own comments for why each step is checked in this
	// order).
	int deviceCount = 0;
	const cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
	if (cudaErr != cudaSuccess || deviceCount == 0) {
		return fail(cudaErr != cudaSuccess ? cudaGetErrorString(cudaErr)
											: "No CUDA devices found");
	}

	if (const cudaError_t setErr = cudaSetDevice(kDefaultCudaDevice); setErr != cudaSuccess) {
		return fail(cudaGetErrorString(setErr));
	}

	// Driver/runtime version and VRAM are available as soon as a device is
	// selected - captured here even though the OptiX-specific steps below
	// haven't run yet, so a driver/runtime mismatch is visible in the report
	// even when OptiX itself later fails to initialize.
	int driverVer = 0, runtimeVer = 0;
	cudaDriverGetVersion(&driverVer);
	cudaRuntimeGetVersion(&runtimeVer);
	out.cuda_driver_version  = driverVer;
	out.cuda_runtime_version = runtimeVer;

	size_t freeBytes = 0, totalBytes = 0;
	if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
		out.vram_free_bytes  = static_cast<unsigned long long>(freeBytes);
		out.vram_total_bytes = static_cast<unsigned long long>(totalBytes);
	}

	if (const OptixResult initRes = optixInit(); initRes != OPTIX_SUCCESS) {
		return fail(optixGetErrorString(initRes));
	}
	out.optix_abi_version = OPTIX_VERSION;

	if (const CUresult cuInitErr = cuInit(0); cuInitErr != CUDA_SUCCESS) {
		return fail("cuInit failed");
	}

	CUdevice cuDevice;
	if (const CUresult devErr = cuDeviceGet(&cuDevice, kDefaultCudaDevice); devErr != CUDA_SUCCESS) {
		return fail("cuDeviceGet failed");
	}

	cuDeviceGetName(out.device_name, static_cast<int>(sizeof(out.device_name)), cuDevice);

	CUcontext cuCtx = nullptr;
	if (const CUresult ctxErr = cuDevicePrimaryCtxRetain(&cuCtx, cuDevice); ctxErr != CUDA_SUCCESS) {
		return fail("cuDevicePrimaryCtxRetain failed");
	}

	OptixDeviceContext context = nullptr;
	OptixDeviceContextOptions options{};
	options.logCallbackFunction = &contextLogCallback;
	options.logCallbackLevel   = kDefaultLogLevel;

	const OptixResult ctxCreateRes = optixDeviceContextCreate(cuCtx, &options, &context);

	if (context) {
		optixDeviceContextDestroy(context);
	}
	if (cuCtx) {
		cuDevicePrimaryCtxRelease(cuDevice);
	}

	if (ctxCreateRes != OPTIX_SUCCESS) {
		return fail(optixGetErrorString(ctxCreateRes));
	}

	out.available = true;
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
	options.logCallbackData = this;
	options.logCallbackLevel = kDefaultLogLevel;

	// Opt-in only: validation mode enables extra device-side checks (e.g.
	// SBT-index bounds) that OptiX otherwise skips for performance, at a
	// real per-launch cost. It caught a real, previously-undetected bug
	// this way - a shadow-ray miss-SBT index that was silently reading
	// adjacent heap memory - deterministically, on every scene, the moment
	// it was tried. See tests/integration/optix_validation_sweep_test.cpp
	// for the opt-in test that exercises this. Fixed for the context's
	// whole lifetime (matches OptiX's own API - there's no way to change
	// an existing context's validation mode after creation).
#pragma warning(suppress: 4996)
	const char* validationEnv = std::getenv("RAY_TRACER_OPTIX_VALIDATION");
	validationEnabled_ = validationEnv && std::string(validationEnv) == "1";
	options.validationMode = validationEnabled_
		? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
		: OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
	if (validationEnabled_) {
		std::cout << "[OptiX] Validation mode enabled (RAY_TRACER_OPTIX_VALIDATION=1)\n";
	}

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
	// Pipeline-wide UPPER BOUND on payload registers (not a per-call exact
	// count - trace_shadow_ray() passes just 1 word and the new probe ray
	// (optix_probe_hit.h) passes 7, both well under this). The radiance ray
	// is the largest user: attenuation(3) + emission(3) + dir(3) + seed(1) +
	// flag(1) + t(1) + brdf_or_light_pdf(1) + explicit_origin(3, flag==3
	// only - MaterialType::Subsurface's probe-walk exit point, see
	// optix_raygen.h) = 16, plus denoiser guide-layer AOVs: albedo(3) +
	// normal(3) = 6 more (p16-p21, see PathTracingPayload::albedo/normal's
	// comment in optix_types.h) = 22, plus eta(1) for pbrt-v4's Russian
	// Roulette etaScale correction (p22, see PathTracingPayload::eta's
	// comment) = 23, plus anyNonSpecularBounces(1) for Integrator "bool
	// regularize" (p23, an INPUT-only register - see optix_raygen.h's own
	// comment on p23, same convention as p12's prev_brdf_pdf for
	// __miss__ms) = 24 total. Comfortably under OptiX's 32-register hard
	// limit.
	pipelineCompileOptions_.numPayloadValues = 24;
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

	// Disk hit group (intersection + closest-hit)
	OptixProgramGroupDesc diskHitDesc = {};
	diskHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	diskHitDesc.hitgroup.moduleIS = module_;
	diskHitDesc.hitgroup.entryFunctionNameIS = "__intersection__disk";
	diskHitDesc.hitgroup.moduleCH = module_;
	diskHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__disk";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&diskHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupDiskPG_
	));

	// Cylinder hit group (intersection + closest-hit)
	OptixProgramGroupDesc cylinderHitDesc = {};
	cylinderHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	cylinderHitDesc.hitgroup.moduleIS = module_;
	cylinderHitDesc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";
	cylinderHitDesc.hitgroup.moduleCH = module_;
	cylinderHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__cylinder";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&cylinderHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitgroupCylinderPG_
	));

	// Triangle hit group (closest-hit + any-hit - intersection is OptiX's
	// built-in hardware triangle test, no custom IS program bound). The
	// any-hit program is new: it's a no-op for the overwhelming majority of
	// triangles (MaterialData::alphaMaskTexIdx < 0), only rejecting a
	// candidate hit via optixIgnoreIntersection() for OBJ/.mtl alpha-cutout
	// materials (map_d) - see __anyhit__triangle's own comment
	// (optix_intersection_triangle.h) for why this doesn't touch the
	// built-in intersection test at all.
	OptixProgramGroupDesc triangleHitDesc = {};
	triangleHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	triangleHitDesc.hitgroup.moduleCH = module_;
	triangleHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
	triangleHitDesc.hitgroup.moduleAH = module_;
	triangleHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__triangle";

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

	// Shadow hit group for disks (any-hit only, no closest-hit)
	OptixProgramGroupDesc shadowDiskHitDesc = {};
	shadowDiskHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowDiskHitDesc.hitgroup.moduleIS = module_;
	shadowDiskHitDesc.hitgroup.entryFunctionNameIS = "__intersection__disk";
	shadowDiskHitDesc.hitgroup.moduleAH = module_;
	shadowDiskHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_disk";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowDiskHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupDiskPG_
	));

	// Shadow hit group for cylinders (any-hit only, no closest-hit)
	OptixProgramGroupDesc shadowCylinderHitDesc = {};
	shadowCylinderHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	shadowCylinderHitDesc.hitgroup.moduleIS = module_;
	shadowCylinderHitDesc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";
	shadowCylinderHitDesc.hitgroup.moduleAH = module_;
	shadowCylinderHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_cylinder";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&shadowCylinderHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&shadowHitgroupCylinderPG_
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

	// RAY_TYPE_PROBE program groups (recursive backend only, Phase 1 BSSRDF
	// - see optix_types.h's RAY_TYPE_PROBE comment and optix_probe_hit.h).
	// One dedicated miss program (a true no-op) plus one closest-hit-only
	// hit group per geometry type, reusing each type's existing __intersection__
	// program (only closest-hit differs per ray type).
	OptixProgramGroupDesc probeMissDesc = {};
	probeMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	probeMissDesc.miss.module = module_;
	probeMissDesc.miss.entryFunctionName = "__miss__probe";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeMissDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeMissPG_
	));

	OptixProgramGroupDesc probeSphereHitDesc = {};
	probeSphereHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeSphereHitDesc.hitgroup.moduleIS = module_;
	probeSphereHitDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	probeSphereHitDesc.hitgroup.moduleCH = module_;
	probeSphereHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_sphere";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeSphereHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupSpherePG_
	));

	OptixProgramGroupDesc probeQuadHitDesc = {};
	probeQuadHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeQuadHitDesc.hitgroup.moduleIS = module_;
	probeQuadHitDesc.hitgroup.entryFunctionNameIS = "__intersection__quad";
	probeQuadHitDesc.hitgroup.moduleCH = module_;
	probeQuadHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_quad";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeQuadHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupQuadPG_
	));

	OptixProgramGroupDesc probeBilinearPatchHitDesc = {};
	probeBilinearPatchHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeBilinearPatchHitDesc.hitgroup.moduleIS = module_;
	probeBilinearPatchHitDesc.hitgroup.entryFunctionNameIS = "__intersection__bilinear_patch";
	probeBilinearPatchHitDesc.hitgroup.moduleCH = module_;
	probeBilinearPatchHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_bilinear_patch";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeBilinearPatchHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupBilinearPatchPG_
	));

	OptixProgramGroupDesc probeDiskHitDesc = {};
	probeDiskHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeDiskHitDesc.hitgroup.moduleIS = module_;
	probeDiskHitDesc.hitgroup.entryFunctionNameIS = "__intersection__disk";
	probeDiskHitDesc.hitgroup.moduleCH = module_;
	probeDiskHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_disk";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeDiskHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupDiskPG_
	));

	OptixProgramGroupDesc probeCylinderHitDesc = {};
	probeCylinderHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeCylinderHitDesc.hitgroup.moduleIS = module_;
	probeCylinderHitDesc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";
	probeCylinderHitDesc.hitgroup.moduleCH = module_;
	probeCylinderHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_cylinder";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeCylinderHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupCylinderPG_
	));

	OptixProgramGroupDesc probeTriangleHitDesc = {};
	probeTriangleHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	probeTriangleHitDesc.hitgroup.moduleCH = module_;
	probeTriangleHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__probe_triangle";

	logSize = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context_,
		&probeTriangleHitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&probeHitgroupTrianglePG_
	));

	std::cout << "[OptiX] Created program groups: raygen, miss (radiance + shadow + probe), sphere hit (radiance + shadow + probe), quad hit (radiance + shadow + probe), bilinear patch hit (radiance + shadow + probe), disk hit (radiance + shadow + probe), cylinder hit (radiance + shadow + probe), triangle hit (radiance + shadow + probe)\n";
	return true;
}

bool OptiXRenderer::linkPipeline() {
	// Collect all program groups (radiance + shadow + probe)
	OptixProgramGroup programGroups[] = {
		raygenPG_,
		missPG_,
		shadowMissPG_,
		probeMissPG_,
		hitgroupSpherePG_,
		hitgroupQuadPG_,
		hitgroupBilinearPatchPG_,
		hitgroupDiskPG_,
		hitgroupCylinderPG_,
		hitgroupTrianglePG_,
		shadowHitgroupSpherePG_,
		shadowHitgroupQuadPG_,
		shadowHitgroupBilinearPatchPG_,
		shadowHitgroupDiskPG_,
		shadowHitgroupCylinderPG_,
		shadowHitgroupTrianglePG_,
		probeHitgroupSpherePG_,
		probeHitgroupQuadPG_,
		probeHitgroupBilinearPatchPG_,
		probeHitgroupDiskPG_,
		probeHitgroupCylinderPG_,
		probeHitgroupTrianglePG_
	};

	// Pipeline link options. maxTraceDepth stays 2 (primary + one nested
	// trace): the probe walk (bssrdf_probe_walk(), optix_device_helpers.h)
	// issues a bounded LOOP of SEQUENTIAL, non-nested trace_probe_ray()
	// calls from within the primary hit's own closest-hit program - each
	// one completes before the next starts, exactly like trace_shadow_ray()
	// already does for NEE - so the deepest simultaneous trace nesting is
	// still 2, not 2+numProbeSteps.
	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 2;  // Primary + shadow/probe/indirect

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

