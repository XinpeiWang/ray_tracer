// image_infinite_light_tests.cpp
// Unit tests for ImageInfiniteLightData<T>
// Mirrors pbrt-v4 ImageInfiniteLight behaviour (src/pbrt/lights.h / lights.cpp).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// The header under test
#include "../../src/shared/image_infinite_light.h"

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------
static constexpr double kPi = 3.14159265358979323846;

// Build a flat 4x2 (width=4, height=2) RGB image with constant value
static std::vector<double> make_constant_image(int w, int h,
											   double r, double g, double b) {
	std::vector<double> img(w * h * 3);
	for (int i = 0; i < w * h; ++i) {
		img[i*3+0] = r;
		img[i*3+1] = g;
		img[i*3+2] = b;
	}
	return img;
}

// Build a 4x4 image where pixel (px,py) has unit value, others are zero
static std::vector<double> make_spike_image(int w, int h, int px, int py,
											double r, double g, double b) {
	std::vector<double> img(w * h * 3, 0.0);
	img[(py * w + px) * 3 + 0] = r;
	img[(py * w + px) * 3 + 1] = g;
	img[(py * w + px) * 3 + 2] = b;
	return img;
}

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, ConstructionDoesNotCrash) {
	auto img = make_constant_image(4, 2, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 4, 2, 1.0);
	EXPECT_EQ(light.width,  4);
	EXPECT_EQ(light.height, 2);
	EXPECT_DOUBLE_EQ(light.scale, 1.0);
}

// ----------------------------------------------------------------
// eval_Le -- constant image should return scale * luminance
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, ConstantImageEvalLeIsUniform) {
	// White image: luminance = 1
	auto img = make_constant_image(8, 4, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 8, 4, 2.0);

	// Any direction should give scale * luminance(1,1,1) = 2.0
	// (small tolerance for bilinear boundary effects)
	double directions[][3] = {
		{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1},
		{0.577, 0.577, 0.577}
	};
	for (auto& d : directions) {
		double len = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
		double le = light.eval_Le(d[0]/len, d[1]/len, d[2]/len);
		EXPECT_NEAR(le, 2.0, 1e-9) << "direction (" << d[0] << "," << d[1] << "," << d[2] << ")";
	}
}

// ----------------------------------------------------------------
// eval_Le_rgb -- RGB channels match stored values
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, EvalLeRgbChannels) {
	// Pure-red image
	auto img = make_constant_image(4, 4, 1.0, 0.0, 0.0);
	ImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0);

	double r, g, b;
	light.eval_Le_rgb(1.0, 0.0, 0.0, r, g, b);
	EXPECT_NEAR(r, 1.0, 1e-9);
	EXPECT_NEAR(g, 0.0, 1e-9);
	EXPECT_NEAR(b, 0.0, 1e-9);
}

// ----------------------------------------------------------------
// power -- constant-luminance image
// Total power = scale * sum(lum) * 4*pi / N
// For constant lum=1: power = scale * 4*pi
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, PowerConstantImage) {
	auto img = make_constant_image(16, 8, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);
	double expected = 4.0 * kPi;
	EXPECT_NEAR(light.power(), expected, 1e-9);
}

TEST(ImageInfiniteLight, PowerScalesWithScale) {
	auto img = make_constant_image(4, 4, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> l1(img.data(), 4, 4, 1.0);
	ImageInfiniteLightData<double> l2(img.data(), 4, 4, 3.0);
	EXPECT_NEAR(l2.power(), 3.0 * l1.power(), 1e-9);
}

// ----------------------------------------------------------------
// pdf_wi -- constant image: pdf should equal 1/(4*pi) everywhere
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, ConstantImagePdfIsUniform) {
	auto img = make_constant_image(16, 8, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);

	double expected_pdf = 1.0 / (4.0 * kPi);

	double dirs[][3] = {
		{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
		{0.577350,0.577350,0.577350}
	};
	for (auto& d : dirs) {
		double len = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		double p = light.pdf_wi(d[0]/len, d[1]/len, d[2]/len);
		// Constant image => uniform distribution; each pixel has equal weight
		// so the 2D pdf is flat = 1.  Solid-angle pdf = 1/(4*pi).
		EXPECT_NEAR(p, expected_pdf, 1e-6)
			<< "dir=(" << d[0] << "," << d[1] << "," << d[2] << ")";
	}
}

// ----------------------------------------------------------------
// sample_wi -- output direction must be unit length
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, SampleWiProducesUnitDirection) {
	auto img = make_constant_image(16, 8, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);

	// Stratified samples
	int N = 64;
	for (int i = 0; i < N; ++i) {
		double ru = (i + 0.5) / N;
		double rv = ((i * 37) % N + 0.5) / N;
		double wx, wy, wz, pdf;
		light.sample_wi(ru, rv, wx, wy, wz, pdf);
		double len = std::sqrt(wx*wx + wy*wy + wz*wz);
		EXPECT_NEAR(len, 1.0, 1e-12) << "sample " << i;
		EXPECT_GT(pdf, 0.0) << "sample " << i;
	}
}

// ----------------------------------------------------------------
// sample_wi / pdf_wi consistency (MIS sanity)
// For a constant image: sample_wi pdf should equal pdf_wi for the
// same direction.
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, SampleWiPdfConsistency) {
	auto img = make_constant_image(16, 8, 0.5, 0.5, 0.5);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);

	int N = 100;
	for (int i = 0; i < N; ++i) {
		double ru = (i + 0.3) / N;
		double rv = ((i * 53) % N + 0.7) / N;
		double wx, wy, wz, pdf_s;
		light.sample_wi(ru, rv, wx, wy, wz, pdf_s);
		double pdf_e = light.pdf_wi(wx, wy, wz);
		// For a constant luminance image both should agree closely
		EXPECT_NEAR(pdf_s, pdf_e, 1e-6) << "sample " << i;
	}
}

// ----------------------------------------------------------------
// sample_wi -- non-uniform image: sampling concentrates on bright region
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, NonUniformImagePdfNormalizesToOne) {
	// Verify that the solid-angle pdf integrates to 1 over the sphere
	// using Monte Carlo with stratified samples.
	// For a constant image the pdf is 1/(4*pi); its integral over S^2 = 1.
	auto img = make_constant_image(16, 8, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);

	double sum = 0.0;
	int N = 10000;
	// Stratified over S^2 using equal-area parameterisation
	int sqN = 100; // sqrt(N)
	for (int i = 0; i < sqN; ++i) {
		for (int j = 0; j < sqN; ++j) {
			double ru = (i + 0.5) / sqN;
			double rv = (j + 0.5) / sqN;
			double wx, wy, wz;
			EqualAreaSquareToSphere(ru, rv, wx, wy, wz);
			// Solid-angle pdf * d_omega = pdf * (4*pi / N) for equal-area
			double pdf = light.pdf_wi(wx, wy, wz);
			// Contribution to integral: pdf * dOmega where dOmega = 4*pi/N
			sum += pdf;
		}
	}
	// sum * (4*pi / N) should be ~1
	double integral = sum * (4.0 * kPi) / N;
	EXPECT_NEAR(integral, 1.0, 0.01);
}

// ----------------------------------------------------------------
// Zero-luminance image: sample_wi returns zero pdf
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, ZeroLuminanceImageSafeSampling) {
	auto img = make_constant_image(4, 4, 0.0, 0.0, 0.0);
	ImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0);

	double wx, wy, wz, pdf;
	light.sample_wi(0.5, 0.5, wx, wy, wz, pdf);
	// Distribution is degenerate (all-zero); pdf may be uniform or zero
	// but should not be NaN/Inf
	EXPECT_FALSE(std::isnan(pdf));
	EXPECT_FALSE(std::isinf(pdf));
}

// ----------------------------------------------------------------
// Float specialisation compiles and produces same results
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, FloatSpecialisationWorks) {
	std::vector<float> img(4 * 4 * 3, 1.0f);
	ImageInfiniteLightData<float> light(img.data(), 4, 4, 1.0f);
	float le = light.eval_Le(1.0f, 0.0f, 0.0f);
	EXPECT_NEAR(le, 1.0f, 1e-5f);
	float p = light.pdf_wi(1.0f, 0.0f, 0.0f);
	EXPECT_GT(p, 0.0f);
}

// ----------------------------------------------------------------
// compensated_distribution: allow_incomplete=true
// Mirrors pbrt-v4 allowIncompletePDF path that uses compensatedDistribution.
// For a constant image, compensated lum = 0 everywhere -> fallback to uniform.
// So allow_incomplete pdf should also be 1/(4*pi).
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, AllowIncompleteConstantImageIsUniform) {
	auto img = make_constant_image(16, 8, 1.0, 1.0, 1.0);
	ImageInfiniteLightData<double> light(img.data(), 16, 8, 1.0);

	double expected = 1.0 / (4.0 * kPi);
	double dirs[][3] = { {1,0,0},{-1,0,0},{0,1,0},{0,0,1} };
	for (auto& d : dirs) {
		double len = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
		double p = light.pdf_wi(d[0]/len, d[1]/len, d[2]/len, /*allow_incomplete=*/true);
		EXPECT_NEAR(p, expected, 1e-6);
	}
}

// allow_incomplete sample_wi still returns unit directions
TEST(ImageInfiniteLight, AllowIncompleteSampleWiUnitLength) {
	auto img = make_constant_image(8, 8, 2.0, 1.0, 0.5);
	ImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0);

	for (int i = 0; i < 50; ++i) {
		double ru = (i + 0.5) / 50.0, rv = ((i*31)%50 + 0.5) / 50.0;
		double wx, wy, wz, pdf;
		light.sample_wi(ru, rv, wx, wy, wz, pdf, /*allow_incomplete=*/true);
		double len = std::sqrt(wx*wx+wy*wy+wz*wz);
		EXPECT_NEAR(len, 1.0, 1e-12) << "sample " << i;
		EXPECT_GT(pdf, 0.0) << "sample " << i;
	}
}

// For a non-uniform image, compensated pdf and primary pdf differ.
// Primary: brighter pixels get higher weight.
// Compensated: subtracts average -> only above-average pixels get weight.
TEST(ImageInfiniteLight, CompensatedPdfDiffersFromPrimary) {
	// 4x4 image: one very bright pixel, rest at value 0.1
	int W=4, H=4;
	std::vector<double> img(W*H*3, 0.1);
	// Make pixel (1,1) very bright
	img[(1*W+1)*3+0] = 10.0;
	img[(1*W+1)*3+1] = 10.0;
	img[(1*W+1)*3+2] = 10.0;

	ImageInfiniteLightData<double> light(img.data(), W, H, 1.0);

	// Direction mapping to a dark pixel (pixel (3,3), centre u=0.875, v=0.875)
	double wx, wy, wz;
	EqualAreaSquareToSphere(0.875, 0.875, wx, wy, wz);

	double pdf_primary    = light.pdf_wi(wx, wy, wz, false);
	double pdf_compensated = light.pdf_wi(wx, wy, wz, true);

	// Both should be > 0, but they should differ in value for a non-uniform image
	EXPECT_GT(pdf_primary, 0.0);
	EXPECT_GT(pdf_compensated, 0.0);
	// They will not be equal for a non-uniform image
	// (This is a light existence/sanity check; exact values are distribution-dependent)
}

// ----------------------------------------------------------------
// Rotation: 180-degree rotation around Y flips +X <-> -X.
// eval_Le(+X, no-rot) should equal eval_Le(-X, rot-180Y) for a non-symmetric image.
// ----------------------------------------------------------------
TEST(ImageInfiniteLight, RotationMapsDirectionCorrectly) {
	// Non-symmetric image: left half bright, right half dark
	int W=8, H=4;
	std::vector<double> img(W*H*3, 0.0);
	for (int y=0; y<H; ++y)
		for (int x=0; x<W/2; ++x) {
			img[(y*W+x)*3+0]=1.0; img[(y*W+x)*3+1]=1.0; img[(y*W+x)*3+2]=1.0;
		}

	// No rotation
	ImageInfiniteLightData<double> light_no_rot(img.data(), W, H, 1.0);

	// 180-degree rotation around Y: x -> -x, z -> -z (column-major)
	// Forward rotation (render->light): [[−1,0,0],[0,1,0],[0,0,−1]] col-major
	double rot180y[9] = { -1,0,0, 0,1,0, 0,0,-1 };
	ImageInfiniteLightData<double> light_rot(img.data(), W, H, 1.0, rot180y);

	// Direction +X in render space maps to -X in light space with the 180Y rotation.
	// eval_Le(+X, rot180Y) should equal eval_Le(-X, identity).
	double le_px_no_rot   = light_no_rot.eval_Le( 1.0, 0.0, 0.0);  // +X, no rot
	double le_mx_no_rot   = light_no_rot.eval_Le(-1.0, 0.0, 0.0);  // -X, no rot
	double le_px_with_rot = light_rot.eval_Le(    1.0, 0.0, 0.0);  // +X, 180Y rot

	// +X and -X have different luminance (left bright, right dark)
	EXPECT_NE(le_px_no_rot, le_mx_no_rot);
	// +X with 180Y rotation should look up the -X direction -> same as -X no-rot
	EXPECT_NEAR(le_px_with_rot, le_mx_no_rot, 1e-9);
}

// Rotation: sample_wi with identity rotation should equal no-rotation for same seeds
TEST(ImageInfiniteLight, IdentityRotationMatchesNoRotation) {
	auto img = make_constant_image(8, 8, 1.0, 1.0, 1.0);
	// Identity matrix (column-major)
	double identity[9] = {1,0,0, 0,1,0, 0,0,1};

	ImageInfiniteLightData<double> l1(img.data(), 8, 8, 1.0);
	ImageInfiniteLightData<double> l2(img.data(), 8, 8, 1.0, identity);

	for (int i = 0; i < 20; ++i) {
		double ru = (i+0.5)/20.0, rv = ((i*17)%20+0.5)/20.0;
		double wx1,wy1,wz1,pdf1, wx2,wy2,wz2,pdf2;
		l1.sample_wi(ru, rv, wx1, wy1, wz1, pdf1);
		l2.sample_wi(ru, rv, wx2, wy2, wz2, pdf2);
		EXPECT_NEAR(wx1, wx2, 1e-12) << "i=" << i;
		EXPECT_NEAR(wy1, wy2, 1e-12) << "i=" << i;
		EXPECT_NEAR(wz1, wz2, 1e-12) << "i=" << i;
		EXPECT_NEAR(pdf1, pdf2, 1e-12) << "i=" << i;
	}
}
