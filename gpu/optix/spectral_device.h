// spectral_device.h
// Device-side CIE XYZ tables and sRGB upsampling table for spectral rendering in CUDA kernels.
//
// Usage (in exactly ONE .cu translation unit, before including this header):
//   #define SPECTRAL_DEVICE_IMPL
// All other .cu files just include without the define.
//
// The tables are loaded from host by wf_upload_cie_tables() / wf_upload_srgb_table()
// in wavefront_launch.cu.
#pragma once
#ifdef __CUDACC__

static constexpr int kDevCIEMin      = 360;
static constexpr int kDevCIEMax      = 830;
static constexpr int kDevCIENSamples = 471;

// CIE 1931 XYZ matching functions (471 samples, 360-830 nm)
// Defined in exactly ONE translation unit (wavefront_launch.cu via SPECTRAL_DEVICE_IMPL).
// All other .cu files get extern declarations.
#ifdef SPECTRAL_DEVICE_IMPL
__constant__ float d_cie_x[kDevCIENSamples];
__constant__ float d_cie_y[kDevCIENSamples];
__constant__ float d_cie_z[kDevCIENSamples];

// CIE Standard Illuminant D65 SPD, resampled onto the same 360-830nm/1nm
// grid as d_cie_x/y/z (see wf_upload_d65_table() in wavefront_launch.cu and
// its host-side caller in wavefront_path_tracer.cpp), and PRE-NORMALISED so
// that InnerProduct(CIE_Y, D65) == kCIE_Y_integral (106.856895) -- i.e. a
// (1,1,1) RGBIlluminantSpectrum reconstructs to XYZ Y=1 through the same
// unweighted-CIE_Y_integral convention SampledSpectrumToXYZ already uses
// below. Without this multiply (or without this exact normalisation), any
// achromatic light-source RGB uplifted through the flat r==g==b branch of
// dev_srgb_to_coeffs produces an equal-energy-illuminant (chromaticity
// (0.333,0.333)) spectrum instead of a D65-white (chromaticity
// (0.3127,0.3290)) one, and reconstructing THAT through wf_xyz_to_linear_rgb
// below (a matrix built for D65 white) yields a non-neutral RGB: R inflated
// ~20%, G/B suppressed ~5-9%/~9-11% -- see wavefront_path_tracer.cpp's
// upload-site comment for the full derivation and measured numbers (found
// via tests/integration/material_cpu_gpu_parity_tests.cpp's B1 finding).
__constant__ float d_d65[kDevCIENSamples];

// sRGB upsampling table in device constant memory
// Layout: [3][64][64][64][3] floats = 3*64*64*64*3 = 2,359,296 floats = ~9 MB
// This fits in device global memory; we store a pointer to device memory.
__constant__ const float* d_srgb_zNodes;     // [64] scale knots
__constant__ const float* d_srgb_coeffs;     // [3*64*64*64*3] trilinear coefficients
#else
extern __constant__ float d_cie_x[kDevCIENSamples];
extern __constant__ float d_cie_y[kDevCIENSamples];
extern __constant__ float d_cie_z[kDevCIENSamples];
extern __constant__ float d_d65[kDevCIENSamples];
extern __constant__ const float* d_srgb_zNodes;
extern __constant__ const float* d_srgb_coeffs;
#endif

// Sample the pre-normalised device D65 table at the nearest integer
// wavelength -- same lookup convention SampledSpectrumToXYZ (sampled_
// spectrum.h) uses for d_cie_x/y/z, so multiplying this into a light's
// uplifted spectrum before it reaches that same estimator is consistent.
// Returns 0 outside the tabulated 360-830nm range (mirrors that function's
// own out-of-range handling).
__device__ __forceinline__ float dev_sample_d65(float lambda) {
	int idx = (int)(lambda + 0.5f) - kDevCIEMin;
	if (idx < 0 || idx >= kDevCIENSamples) return 0.f;
	return d_d65[idx];
}

// Lightweight device-only sRGB upsampling: same algorithm as RGBToSpectrumTable::operator()
// but reads from __constant__ / global device pointers.
#include "sampled_spectrum.h"  // RGBSigmoidPolynomial lives in color_utils.h, but we forward it

// Forward: RGBSigmoidPolynomial is defined in rgb_to_spectrum_table.h included by the kernel
__device__ __forceinline__
float dev_srgb_lerp(float t, float a, float b) { return a + t * (b - a); }

static constexpr int kSRGBRes = 64;

__device__ __forceinline__
float dev_find_interval(const float* nodes, int n, float z) {
	int lo = 0, hi = n - 1;
	while (lo < hi - 1) {
		int mid = (lo + hi) / 2;
		if (nodes[mid] < z) lo = mid; else hi = mid;
	}
	return (float)lo;
}

// Map clamped sRGB [0,1]^3 to RGBSigmoidPolynomial coefficients using device table.
// Returns {c0, c1, c2} coefficients (caller constructs RGBSigmoidPolynomial).
__device__ __forceinline__
void dev_srgb_to_coeffs(float r, float g, float b, float& c0, float& c1, float& c2) {
	if (r == g && g == b) {
		float v = (r - 0.5f) / sqrtf(r * (1.f - r));
		c0 = 0.f; c1 = 0.f; c2 = v;
		return;
	}
	int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
	float rgb[3] = { r, g, b };
	float z = rgb[maxc];
	float x = rgb[(maxc + 1) % 3] * (kSRGBRes - 1) / z;
	float y = rgb[(maxc + 2) % 3] * (kSRGBRes - 1) / z;

	int xi = min((int)x, kSRGBRes - 2);
	int yi = min((int)y, kSRGBRes - 2);
	float zi_f = dev_find_interval(d_srgb_zNodes, kSRGBRes, z);
	int zi  = (int)zi_f;
	float dx = x - xi, dy = y - yi;
	float dz = (z - d_srgb_zNodes[zi]) / (d_srgb_zNodes[zi+1] - d_srgb_zNodes[zi]);

	// Stride: [3][64][64][64][3]  ->  offset = maxc*(64^3*3) + dz_i*(64^2*3) + dy_i*(64*3) + dx_i*3 + coeff
	auto co = [&](int ddx, int ddy, int ddz, int ci) -> float {
		int idx = maxc * (kSRGBRes*kSRGBRes*kSRGBRes*3)
				+ (zi+ddz) * (kSRGBRes*kSRGBRes*3)
				+ (yi+ddy) * (kSRGBRes*3)
				+ (xi+ddx) * 3
				+ ci;
		return d_srgb_coeffs[idx];
	};

	for (int i = 0; i < 3; ++i) {
		float v = dev_srgb_lerp(dz,
			dev_srgb_lerp(dy, dev_srgb_lerp(dx, co(0,0,0,i), co(1,0,0,i)),
							   dev_srgb_lerp(dx, co(0,1,0,i), co(1,1,0,i))),
			dev_srgb_lerp(dy, dev_srgb_lerp(dx, co(0,0,1,i), co(1,0,1,i)),
							   dev_srgb_lerp(dx, co(0,1,1,i), co(1,1,1,i))));
		if (i == 0) c0 = v;
		else if (i == 1) c1 = v;
		else c2 = v;
	}
}

#endif // __CUDACC__
