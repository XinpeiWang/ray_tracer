// wavefront_kernels_upscale.cu
// Neural temporal upscale pipeline kernels (Live Preview only). See this
// project's own plan for the full design.
//
// Replaces qt_gui/realtime_preview_session.cpp's own CPU-side
// reprojectAccumulationHi()/write-step block-splat with a small, online-
// trained network - mirrors the Neural Radiance Cache's own "dedicated,
// fixed-budget side pipeline run once per render() call" shape
// (wavefront_kernels_nrc.cu), but needs NO extra ray tracing at all: every
// "training record" here is a high-res cell that ALREADY received a real
// low-res sample this frame as part of the ordinary render, so this only
// bounds the BACKWARD-pass cost, not any tracing cost - cheaper per frame
// than NRC's own training.
//
// Three kernel families, run in THIS order every render() call (see
// WavefrontPathTracer::launchNeuralUpscaleUpdate(), wavefront_path_tracer.cpp,
// for the exact call site and why the ordering matters):
//   1. upscale_compute_motion_vectors - trivial, one thread per LOW-RES pixel.
//   2. upscale_train + upscale_apply_gradients - MUST run before #3, on
//      d_neuralUpscaleForwardCache_'s still-last-frame contents (see #3's
//      own comment for why).
//   3. upscale_infer - one thread per HIGH-RES output cell; overwrites
//      d_neuralUpscaleForwardCache_/d_neuralUpscaleHistory_/
//      d_neuralUpscaleOutput_ for THIS frame.
//
// One CUDA thread = one item's full inference/training work, entirely in
// registers/local arrays - same "one thread, one item, full eval in
// registers" pattern every wf_finish_material_scatter()/wf_nrc_forward()
// call site already uses (no shared memory, no warp cooperation). The
// neighbor-gathering shape of upscale_infer mirrors svgf_atrous_pass's own
// "small explicit neighbor loop, independent global-memory reads"
// (wavefront_kernels_svgf.cu) more closely than any NRC kernel - this
// codebase has no cooperative-thread/shared-memory kernel infrastructure at
// all, so a true weight-shared convolution was never on the table (see this
// project's own plan).

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"
#include "wavefront_upscale_mlp.h"
#include "wavefront_nrc_encoding.h"      // wf_nrc_encode_scalar - one-blob phase encoding
#include "wavefront_temporal_upscale_math.h"  // wf_temporal_upscale_subcell - THE single source of truth for "which cell is due"
#include "wavefront_restir_helpers.h"    // wf_restir_reproject_prev_pixel, wf_project_to_screen, GpuReprojectBasis

// ---------------------------------------------------------------------------
// upscale_compute_motion_vectors - one thread per LOW-RES pixel. Screen-
// space delta (this frame's own screen position, which for a rendered
// pixel is just its own (px,py) - minus where this SAME world point
// projected to in the PREVIOUS frame's camera). Used ONLY as an auxiliary
// input FEATURE for the network (magnitude/direction hint) - the actual
// history LOOKUP in upscale_infer reprojects via the proven
// wf_restir_reproject_prev_pixel() machinery directly (world-position +
// disocclusion test), not via this buffer, so a wrong/stale motion vector
// here can only ever degrade quality, never correctness (see this
// project's own plan for why the two are kept separate).
// ---------------------------------------------------------------------------
extern "C" __global__ void upscale_compute_motion_vectors(
	const float4* worldPos, float2* outMotionVectors,
	int width, int height, GpuReprojectBasis prevCamera)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int numPixels = width * height;
	if (idx >= numPixels) return;

	const float4 wp = worldPos[idx];
	if (wp.w == 0.0f) {
		outMotionVectors[idx] = make_float2(0.0f, 0.0f);
		return;
	}
	const int px = idx % width;
	const int py = idx / width;
	// This pixel's OWN current screen position - no projection needed, it's
	// exactly where it was rendered (matches wf_project_to_screen's own
	// s=left-right/t=bottom-to-top convention, wavefront_restir_helpers.h).
	const float curS = (px + 0.5f) / (float)width;
	const float curT = 1.0f - (py + 0.5f) / (float)height;

	const WfScreenProjection prevProj = wf_project_to_screen(make_float3(wp.x, wp.y, wp.z), prevCamera);
	if (!prevProj.inFront) {
		outMotionVectors[idx] = make_float2(0.0f, 0.0f);
		return;
	}
	outMotionVectors[idx] = make_float2(curS - prevProj.s, curT - prevProj.t);
}

// ---------------------------------------------------------------------------
// upscale_train - one thread per training SAMPLE (kUpscaleTrainingRecordsPerFrame
// threads, a fixed/resolution-independent budget - see wavefront_upscale_types.h's
// own comment). Each thread picks a RANDOM low-res pixel (this frame's
// write cycle gives EVERY low-res pixel a genuine fresh ground-truth
// sample, so any of them is a valid training candidate - no extra ray
// tracing needed, unlike NRC), finds the exact high-res cell that pixel's
// fresh sample was due to land in THIS frame (the SAME
// wf_temporal_upscale_subcell() call site generate_camera_rays already
// used to pick this frame's own jitter - see WavefrontPathTracer::
// launchNeuralUpscaleUpdate()'s own comment for why a SEPARATE, always-
// advance-by-1 write counter is used here rather than the ray-jitter
// index), and looks up THAT cell's cache from the LAST time upscale_infer
// ran (still resident in d_neuralUpscaleForwardCache_ - upscale_infer has
// not run yet this frame, see this file's own header comment on why
// ordering matters). Backward accumulates into gradAccum; contributing
// threads are counted via atomicAdd so upscale_apply_gradients can average
// correctly (mirrors nrc_apply_gradients' own numContributingRecords role,
// wavefront_kernels_nrc.cu).
//
// NOTE: when Samples/Frame > 1, lowResFramebuffer is already an average
// across several GPU-side jitter phases (see generate_camera_rays), not
// the single fresh phase this kernel (and upscale_infer) credit to
// dueCx/dueCy - the exact same known v1 simplification the pre-existing
// CPU block-splat path documents at its own write-step call site
// (qt_gui/realtime_preview_session.cpp, RealtimePreviewWorker::
// renderLoop()). There it only blurs a display splat; here it also
// mislabels the TRAINING target for that sub-cell, which the network then
// bakes into its learned blend weights rather than just displaying once -
// recommend Samples/Frame=1 with Neural Reconstruction enabled until this
// is addressed with per-phase buffers.
// ---------------------------------------------------------------------------
extern "C" __global__ void upscale_train(
	const float3* lowResFramebuffer, int width, int height,
	const UpscaleForwardCache* forwardCache, int upscaleFactor, unsigned int dueCx, unsigned int dueCy,
	const float* weights, float* gradAccum, int* validRecordCounter,
	unsigned int frameSeed, int numSamples)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numSamples) return;

	unsigned int seed = wf_pcg(wf_pcg((unsigned int)i) ^ frameSeed);
	const int numPixels = width * height;
	const int lowIdx = (int)(wf_rand(seed) * (float)numPixels);
	const int clampedLowIdx = lowIdx >= numPixels ? numPixels - 1 : lowIdx;
	const int px = clampedLowIdx % width;
	const int py = clampedLowIdx / width;

	const int Wh = width * upscaleFactor;
	const int hx = px * upscaleFactor + (int)dueCx;
	const int hy = py * upscaleFactor + (int)dueCy;
	const int cellIdx = hy * Wh + hx;

	const UpscaleForwardCache& cache = forwardCache[cellIdx];
	if ((cache.flags & kUpscaleCacheFlagValid) == 0) return;  // no prediction was ever cached for this cell yet

	const float3 target = lowResFramebuffer[clampedLowIdx];
	float3 dLossDBlended;
	wf_upscale_loss_gradient(cache.blended, target, dLossDBlended);
	wf_upscale_backward(weights, cache, dLossDBlended, gradAccum);
	atomicAdd(validRecordCounter, 1);
}

// ---------------------------------------------------------------------------
// upscale_apply_gradients -- one thread per weight (kUpscaleNumWeights
// threads), the ONLY kernel that ever writes the live weight buffer - see
// nrc_apply_gradients' own identical comment (wavefront_kernels_nrc.cu) for
// the full "gradient accumulation + single serial apply step" rationale.
// ---------------------------------------------------------------------------
extern "C" __global__ void upscale_apply_gradients(
	float* weights, float* adamM, float* adamV, float* gradAccum,
	int numContributingRecords, int stepCount)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= kUpscaleNumWeights) return;
	if (numContributingRecords > 0) {
		const float grad = gradAccum[i] / (float)numContributingRecords;
		wf_upscale_adam_step(weights[i], adamM[i], adamV[i], grad, stepCount);
	}
	gradAccum[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// upscale_reset_weights -- re-randomizes weights to small values and zeroes
// Adam's moment buffers, called once on first allocation (mirrors
// nrc_reset_weights' own call-once-on-scene-load role, wavefront_kernels_nrc.cu,
// though this feature's own weights are NOT reset on scene change - see
// WavefrontPathTracer::launchNeuralUpscaleUpdate()'s own comment for why a
// blend-weight network's own learned "which candidate to trust" behavior is
// less scene-specific than NRC's own learned radiance field, so there is
// less to gain from forcing a fresh start).
//
// Uses NRC's own (rand-0.5)*0.2 range (exactly (-0.1, 0.1)), NOT the wider
// range this feature's own unit test originally used to reproduce a
// training-convergence failure during design. That failure (see this
// project's own diagnostic history) turned out to be caused by the test's
// arbitrary Uniform(0,2) UNCORRELATED candidate scenario - a harder
// optimization landscape than this network will ever see in production,
// where all 6 candidates are noisy samples of the SAME physical
// neighborhood - not by the weight-init range itself; a scale this small
// converged cleanly once the test scenario was made realistic. A wider
// range (matching what the test originally tried, e.g. (rand-0.5)*1.0) is
// the one confirmed to risk the softmax output layer locking onto an
// early-favored candidate before training can correct it - NRC's own
// direct-regression softplus output has no equivalent winner-take-all
// dynamic, so its own init range isn't a comparable safety precedent on
// its own, it just happens to already be this small.
// ---------------------------------------------------------------------------
extern "C" __global__ void upscale_reset_weights(float* weights, float* adamM, float* adamV, unsigned int seed)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= kUpscaleNumWeights) return;
	unsigned int s = wf_pcg(wf_pcg((unsigned int)i) ^ seed);
	weights[i] = (wf_rand(s) - 0.5f) * 0.2f;
	adamM[i] = 0.0f;
	adamV[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// upscale_infer -- one thread per HIGH-RES output cell (width*height*
// upscaleFactor^2 threads - the first GPU-resident high-res buffer/kernel
// in this codebase, see this project's own plan). Gathers the 6-candidate
// input vector, runs the network, writes this frame's blended color into
// outColor, its own forward-pass cache into outCache (for upscale_train to
// consume up to upscaleFactor^2-1 frames from now), and the reprojected-
// forward color+age into outHistory (next frame's own history input).
// ---------------------------------------------------------------------------
extern "C" __global__ void upscale_infer(
	const float3* lowResFramebuffer, const float4* worldPos, const float4* worldPosHistory,
	const float4* history, const float2* motionVectors,
	int width, int height, int upscaleFactor,
	unsigned int dueCx, unsigned int dueCy,
	GpuReprojectBasis prevCamera, bool historyValid, float3 cameraOrigin,
	const float* weights,
	float3* outColor, float4* outHistory, UpscaleForwardCache* outCache)
{
	const int Wh = width * upscaleFactor;
	const int Hh = height * upscaleFactor;
	const int hx = blockIdx.x * blockDim.x + threadIdx.x;
	const int hy = blockIdx.y * blockDim.y + threadIdx.y;
	if (hx >= Wh || hy >= Hh) return;
	const int cellIdx = hy * Wh + hx;

	const int px = hx / upscaleFactor;
	const int py = hy / upscaleFactor;
	const int ox = hx % upscaleFactor;
	const int oy = hy % upscaleFactor;
	const int lowIdx = py * width + px;

	const int pxN = py > 0 ? py - 1 : 0;
	const int pxS = py < height - 1 ? py + 1 : height - 1;
	const int pxE = px < width - 1 ? px + 1 : width - 1;
	const int pxW = px > 0 ? px - 1 : 0;
	const float3 C = lowResFramebuffer[lowIdx];
	const float3 N = lowResFramebuffer[pxN * width + px];
	const float3 S = lowResFramebuffer[pxS * width + px];
	const float3 E = lowResFramebuffer[py * width + pxE];
	const float3 W = lowResFramebuffer[py * width + pxW];

	// Reproject via the SAME proven world-space disocclusion test DI/GI/SVGF
	// all already share (wf_restir_reproject_prev_pixel, wavefront_restir_
	// helpers.h) at LOW-RES granularity, then translate the result to THIS
	// cell's own high-res coordinates by keeping its sub-cell offset (ox,oy)
	// fixed relative to its (possibly moved) low-res parent - a block-rigid-
	// motion approximation, matching the granularity the PRE-existing CPU
	// reconstruction already accepted (see this project's own plan).
	float3 H = C;
	float historyAge = 1.0f;  // no history at all -> treat as maximally stale
	float confidence = 0.0f;
	const float4 wp = worldPos[lowIdx];
	if (historyValid && wp.w != 0.0f) {
		const int prevLowIdx = wf_restir_reproject_prev_pixel(
			make_float3(wp.x, wp.y, wp.z), prevCamera, worldPosHistory, width, height);
		if (prevLowIdx >= 0) {
			confidence = 1.0f;
			const int prevPx = prevLowIdx % width;
			const int prevPy = prevLowIdx / width;
			const int prevHx = prevPx * upscaleFactor + ox;
			const int prevHy = prevPy * upscaleFactor + oy;
			const float4 histEntry = history[prevHy * Wh + prevHx];
			H = make_float3(histEntry.x, histEntry.y, histEntry.z);
			historyAge = histEntry.w;
		}
	}

	const bool isDueThisFrame = ((unsigned int)ox == dueCx && (unsigned int)oy == dueCy);
	const float period = (float)(upscaleFactor * upscaleFactor);
	const float newAge = isDueThisFrame ? 0.0f : fminf(1.0f, historyAge + 1.0f / period);

	float depth = 0.0f;
	if (wp.w != 0.0f) {
		const float3 delta = make_float3(wp.x - cameraOrigin.x, wp.y - cameraOrigin.y, wp.z - cameraOrigin.z);
		depth = sqrtf(fmaxf(dot(delta, delta), 0.0f));
	}
	const float2 mv = motionVectors[lowIdx];

	// Assemble the input vector - layout MUST match wavefront_upscale_types.h's
	// own kUpscaleInputDim comment exactly (candidates first, in
	// kUpscaleCandidate* order).
	float input[kUpscaleInputDim];
	input[3 * kUpscaleCandidateCenter + 0] = C.x; input[3 * kUpscaleCandidateCenter + 1] = C.y; input[3 * kUpscaleCandidateCenter + 2] = C.z;
	input[3 * kUpscaleCandidateNorth  + 0] = N.x; input[3 * kUpscaleCandidateNorth  + 1] = N.y; input[3 * kUpscaleCandidateNorth  + 2] = N.z;
	input[3 * kUpscaleCandidateSouth  + 0] = S.x; input[3 * kUpscaleCandidateSouth  + 1] = S.y; input[3 * kUpscaleCandidateSouth  + 2] = S.z;
	input[3 * kUpscaleCandidateEast   + 0] = E.x; input[3 * kUpscaleCandidateEast   + 1] = E.y; input[3 * kUpscaleCandidateEast   + 2] = E.z;
	input[3 * kUpscaleCandidateWest   + 0] = W.x; input[3 * kUpscaleCandidateWest   + 1] = W.y; input[3 * kUpscaleCandidateWest   + 2] = W.z;
	input[3 * kUpscaleCandidateHistory + 0] = H.x; input[3 * kUpscaleCandidateHistory + 1] = H.y; input[3 * kUpscaleCandidateHistory + 2] = H.z;
	input[18] = historyAge;
	input[19] = confidence;
	const float phaseX = ((float)ox + 0.5f) / (float)upscaleFactor;
	const float phaseY = ((float)oy + 0.5f) / (float)upscaleFactor;
	wf_nrc_encode_scalar(phaseX, input + 20);
	wf_nrc_encode_scalar(phaseY, input + 24);
	input[28] = mv.x;
	input[29] = mv.y;
	input[30] = depth;

	UpscaleForwardCache cache;
	wf_upscale_forward(weights, input, cache);
	cache.flags = kUpscaleCacheFlagValid;

	outColor[cellIdx] = cache.blended;
	outHistory[cellIdx] = make_float4(cache.blended.x, cache.blended.y, cache.blended.z, newAge);
	outCache[cellIdx] = cache;
}
