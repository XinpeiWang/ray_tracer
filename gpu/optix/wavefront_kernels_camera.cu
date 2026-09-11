// wavefront_kernels_camera.cu
// CUDA compute kernel: primary-ray generation for the wavefront GPU path
// tracer (Kernel 1 of the original wavefront_kernels.cu, split out so an
// edit to one kernel family no longer forces recompiling every other one -
// see wavefront_device_helpers.h's own header comment for why the two
// helpers this split needed live there instead of in a 7th split file).

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"
#include "wavefront_svgf_math.h"  // wf_checkerboard_pixel_active

extern "C" __global__ void generate_camera_rays(
	WorkQueue<RayWorkItem> rayQueue,
	unsigned int width,
	unsigned int height,
	GpuCameraParams camera,
	unsigned int sampleIdx,
	unsigned int frameNumber,
	// Per-pixel sum of this render's filter weights so far - see this
	// kernel's own filter_w comment. Same size/lifetime as framebuffer,
	// zeroed once per render before the sample loop starts.
	float* weightBuffer,
	// Checkerboard temporal upsampling (WavefrontPathTracer::render()'s own
	// comment) - see wf_checkerboard_pixel_active()'s own comment
	// (wavefront_device_helpers.h) for the shared parity check every
	// consumer of "was this pixel sampled this frame" must agree on.
	bool checkerboardActive
) {
	// Film "cropwindow"/"pixelbounds" (pbrt-v4) - the HOST launcher
	// (wf_launch_generate_camera_rays, wavefront_launch.cu) already sizes
	// this kernel's own launch grid to just the crop rectangle (not the
	// full frame) when one is active, offsetting thread index 0 to
	// cropX0/cropY0 rather than pixel (0,0) - unlike the recursive
	// backend's single whole-render launch, THIS kernel launches once per
	// sample-index (many times per render), so a cropped-out pixel's
	// thread genuinely never gets scheduled at all here, not just an
	// early-return - real, not just cosmetic, launch-overhead savings at
	// high sample counts. cropX0/cropY0 default to 0 (gpu_in_crop's own
	// cropX1<=0 "no crop" sentinel) so an unmodified/native scene launches
	// over the full frame exactly as before this feature existed.
	const int cropX0 = (camera.cropX1 > 0) ? camera.cropX0 : 0;
	const int cropY0 = (camera.cropY1 > 0) ? camera.cropY0 : 0;
	int px = cropX0 + blockIdx.x * blockDim.x + threadIdx.x;
	int py = cropY0 + blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= (int)width || py >= (int)height) return;

	// Still needed even with the crop-sized launch grid above: 16x16 block
	// rounding can overshoot past cropX1/cropY1 (the crop rectangle's own
	// far edge) the same way it already overshoots past width/height for
	// an ordinary uncropped render - this is that same, pre-existing
	// "last block is partially out of bounds" case, just at the crop's own
	// boundary instead of the frame's. Simply never enqueue a ray (or touch
	// weightBuffer) for an out-of-crop pixel: unlike the recursive
	// backend's own raygen, there's no explicit black-write needed here -
	// normalize_framebuffer already turns a zero-weight pixel (never
	// incremented by this kernel for any sample) into black (its own
	// `w > 0.0f` guard), the exact same mechanism that already exists for
	// a pathological zero-weight filter parameterization.
	if (!gpu_in_crop(camera, px, py)) return;

	// Checkerboard temporal upsampling: an inactive pixel gets neither a
	// queued ray nor a weightBuffer increment this frame - the exact same
	// "never enqueue, never touch weightBuffer" shape the crop check just
	// above already uses, so normalize_framebuffer's existing `w > 0.0f`
	// guard (this kernel's own crop-check comment) does double duty as the
	// signal every downstream SVGF/GI kernel uses to tell "held over from
	// last frame" apart from "genuinely sampled this frame."
	if (!wf_checkerboard_pixel_active(px, py, frameNumber, checkerboardActive)) return;

	int pixelIdx = py * (int)width + px;
	unsigned int seed = wf_pcg(wf_pcg(pixelIdx + sampleIdx * width * height) ^ frameNumber);

	// Captured once so the same two draws drive both the film position AND
	// the reconstruction filter's sub-pixel offset below (see
	// gpu_filter_evaluate()'s own comment) - matches the recursive
	// backend's identical reuse of its own Halton hx/hy for both purposes.
	float rx = wf_rand(seed);
	float ry = wf_rand(seed);
	float u = (float(px) + rx) / float(width  - 1);
	// Flip Y to match optix_raygen.h's lower-left-origin viewport convention
	// (py=0/top row -> v=1, matching how lower_left_corner+u*horizontal+
	// v*vertical is constructed for Perspective/Orthographic, and how
	// Spherical's theta=pi*v expects v=0 at the bottom). Without this flip
	// every wavefront-mode render using those camera kinds came out
	// vertically mirrored relative to the recursive path.
	float v = (float(height - 1 - py) + ry) / float(height - 1);
	// NOTE for future CameraKind additions: same lower-left-origin `v` as
	// optix_raygen.h's __raygen__rg (see that function's matching comment) -
	// a camera whose reference model assumes raw raster order (v=0 at the
	// top row) needs to locally undo this flip before using it, the way
	// wf_generate_primary_ray's Spherical/Realistic cases already do.

	// Sub-pixel offset in [-0.5, 0.5] for the reconstruction filter - see
	// optix_raygen.h's identical computation for why the offset's sign
	// convention doesn't need to match CPU's own (every filter shape is an
	// even function in each axis).
	float ox = rx - 0.5f;
	float oy = ry - 0.5f;
	float filter_w = gpu_filter_evaluate(camera.filterKind, camera.filterB,
		camera.filterC, camera.filterSigma, camera.filterTau, ox, oy);

	RayWorkItem item;
	float cam_weight;
	wf_generate_primary_ray(camera, u, v, seed, item.origin, item.direction, cam_weight);
	// Sample hero wavelengths for spectral rendering (pbrt-v4: SampledWavelengths::SampleVisible)
	float lambda_u = wf_rand(seed);
	SampledWavelengths<kWFNWavelengths> swl = SampledWavelengths<kWFNWavelengths>::SampleVisible(lambda_u);
	// filter_w folded in here (same slot cam_weight already uses) so it
	// propagates through every downstream radiance contribution for free -
	// filter_w is carried as its OWN field (RayWorkItem::filterWeight), NOT
	// folded into throughput - see that field's own comment for why: it's a
	// pure reconstruction weight, and contaminating throughput with it
	// (Gaussian's own evaluate() is tiny in absolute magnitude) was a real,
	// confirmed bug that made Russian Roulette kill the overwhelming
	// majority of paths almost immediately. weightBuffer accumulates once
	// per (pixel, sample) here regardless of how many downstream kernels
	// this ray's radiance eventually flows through - normalize_framebuffer
	// divides by it instead of a flat 1/samplesPerPixel.
	for (int i = 0; i < kWFNWavelengths; ++i) {
		item.throughput[i]     = cam_weight;
		item.radiance[i]       = 0.0f;
		item.wavelengths[i]    = swl.lambda[i];
		item.wavelength_pdfs[i] = swl.pdf[i];
	}
	item.filterWeight = filter_w;
	atomicAdd(&weightBuffer[pixelIdx], filter_w);
	// Object (per-primitive sphere) motion blur shutter time - one draw per
	// PRIMARY ray, then carried unchanged through every bounce of this same
	// path (see RayWorkItem::time's own comment). Mirrors optix_raygen.h's
	// identical `params.camera.motionBlurEnabled ? random_float(seed) : 0.0f`
	// gate for the recursive backend. Drawn BEFORE item.seed is captured below,
	// not after - seed is a PCG state advanced by reference, so capturing
	// it first would silently discard this draw's advancement and leave
	// the next consumer (evaluate_materials, which resumes from item.seed)
	// replaying the same stream this draw already used.
	item.time = camera.motionBlurEnabled ? wf_rand(seed) : 0.0f;
	item.seed       = seed;
	item.pixelIndex = pixelIdx;
	item.depth      = 0;
	item.specular_bounce = 1;  // primary ray: always allow emissive hit
	item.any_nonspecular = 0; // primary ray: no prior bounce to regularize against
	item.etaScale   = 1.0f;    // primary ray: no transmission yet - see RayWorkItem::etaScale
	item.brdf_pdf   = 0.0f;    // primary ray: no MIS on an escaped camera ray
	item.tMin       = 0.001f;
	item.tMax       = 1e30f;

	rayQueue.push(item);
}
