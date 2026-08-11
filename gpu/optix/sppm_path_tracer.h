// sppm_path_tracer.h
// SPPMPathTracer -- GPU SPPM (Stochastic Progressive Photon Mapping)
// integrator. Does NOT implement PathTracingStrategy: that interface's
// render() has a single-call, single-frame contract with no iteration-
// count/photon-count parameters and no persistent cross-call state, which
// doesn't fit SPPM's iteration loop (camera pass -> hash-grid build ->
// photon pass -> radius contraction, repeated, with tau/n/radius shrinking
// across iterations). Precedent for not forcing every render mode through
// that interface already exists: WavefrontPathTracer is reached via a
// side-channel (OptiXRenderer::enableWavefront()), not via
// createPathTracingStrategy(). SPPMPathTracer follows the same pattern --
// see OptiXRenderer::renderSPPM() (added in sub-phase 1e).
//
// Phase 1a: only the pipeline/SBT plumbing for a trivial camera-pass raygen
// (see sppm_programs.cu). Structurally mirrors WavefrontPathTracer
// (own module/program groups/pipeline/SBT), simplified to one pipeline and
// one ray type since Phase 1a has no shadow rays yet.
#pragma once

#include "sppm_types.h"
#include "optix_types.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

namespace optix_renderer {

class SPPMPathTracer {
public:
	SPPMPathTracer();
	~SPPMPathTracer();

	bool initialize(OptixDeviceContext context, cudaStream_t stream);
	bool createProgramGroups();
	bool linkPipeline();
	bool buildSBT(unsigned int numSpheres, unsigned int numQuads);
	void cleanup();

	void setPTXPath(const std::string& path) { ptxPath_ = path; }

	// Phase 1a: traces one primary ray per pixel and writes white-on-hit /
	// black-on-miss to outputFramebuffer -- proves the pipeline/SBT/PTX-JIT
	// machinery works against the real uploaded scene before any real SPPM
	// math exists. Replaced with the real iteration loop in sub-phase 1d;
	// kept as a distinctly-named method (not render()) so it's obvious this
	// is a placeholder, not the final API.
	bool renderTrivial(int width, int height, const GpuCameraParams& camera,
	                    float* outputFramebuffer, OptixTraversableHandle gasHandle,
	                    CUdeviceptr d_materials, CUdeviceptr d_spheres, CUdeviceptr d_quads,
	                    unsigned int numMaterials, unsigned int numSpheres, unsigned int numQuads);

private:
	bool loadModule();
	void destroyProgramGroups();
	void destroySBT();

	OptixDeviceContext context_ = nullptr;
	cudaStream_t       stream_  = nullptr;
	std::string        ptxPath_;

	OptixModule sppmModule_ = nullptr;
	OptixPipelineCompileOptions pipelineCompileOptions_ = {};

	OptixProgramGroup raygenCameraPG_ = nullptr;
	OptixProgramGroup missRadiancePG_ = nullptr;
	OptixProgramGroup hitSpherePG_    = nullptr;
	OptixProgramGroup hitQuadPG_      = nullptr;

	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_ = {};
	CUdeviceptr d_raygenRecord_   = 0;
	CUdeviceptr d_missRecord_     = 0;
	CUdeviceptr d_hitRecords_     = 0;

	CUdeviceptr d_launchParams_ = 0;

	unsigned int numSpheres_ = 0;
	unsigned int numQuads_   = 0;
};

} // namespace optix_renderer
