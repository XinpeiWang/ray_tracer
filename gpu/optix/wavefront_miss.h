// wavefront_miss.h -- Radiance-miss, shadow-miss, and exception programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __miss__wf_radiance
//   Ray escaped — payload.hit stays false; raygen will add to missQueue.
// ============================================================================
extern "C" __global__ void __miss__wf_radiance() {
	// nothing — hit remains false
}

// __miss__wf_shadow: ray reached tMax without hitting anything — not occluded.
extern "C" __global__ void __miss__wf_shadow() {
	// occluded stays false (its default)
}

// Exception program -- see wavefront_path_tracer.cpp's exceptionFlags
// comment (initialize()) for the full story: registering this program and
// enabling STACK_OVERFLOW/TRACE_DEPTH is the actual fix for a CUDA 718
// "invalid program counter" crash that used to kill the device context
// after a specific sequence of earlier GPU tests. It has never been
// observed to fire in a passing run -- kept registered as a real safety
// net (prints what OptiX thinks went wrong instead of a bare post-hoc
// cudaStreamSynchronize failure) rather than diagnostic scaffolding.
extern "C" __global__ void __exception__wf_report() {
	const int code = optixGetExceptionCode();
	const uint3 idx = optixGetLaunchIndex();
	printf("[WF-EXCEPTION] code=%d launchIdx=(%u,%u,%u)\n", code, idx.x, idx.y, idx.z);
	if (code == OPTIX_EXCEPTION_CODE_STACK_OVERFLOW) {
		printf("[WF-EXCEPTION]   -> STACK_OVERFLOW\n");
	} else if (code == OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED) {
		printf("[WF-EXCEPTION]   -> TRACE_DEPTH_EXCEEDED\n");
	}
}
