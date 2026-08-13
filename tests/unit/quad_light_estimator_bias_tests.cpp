/**
 * @file quad_light_estimator_bias_tests.cpp
 * @brief Which quad-light NEE estimator is biased: CPU's or GPU's?
 *
 * The CPU and GPU path tracers disagree by 3-5% on every scene lit by a
 * planar area light, already at max_depth=1 (the direct-lighting term, not
 * anything that accumulates). The two backends sample a quad differently:
 *
 *   CPU  uniform in SOLID ANGLE (Urena et al. 2013), pdf = 1/omega
 *   GPU  uniform in AREA, pdf = dist^2 / (cos_light * area)  -- the standard
 *        area-to-solid-angle Jacobian applied to a constant 1/area density
 *
 * spherical_rectangle_adjudication_tests.cpp already cleared CPU's solid
 * angle math (SphericalRectangleSolidAngle matches direct integration on
 * five cases). That does not clear CPU's ESTIMATOR, and it says nothing
 * about GPU's - a strategy can compute a geometrically correct solid angle
 * and still divide by the wrong measure at sample time, or convert area to
 * solid angle incorrectly.
 *
 * This checks the thing that actually matters: does each estimator's
 * SAMPLE VALUE - not its intermediate solid angle - converge to the true
 * direct-lighting integral? Both formulas here are copied from the real
 * call sites (not re-derived), stripped of everything but the light-only
 * NEE estimator (no MIS weight, no BSDF-sampling strategy) - MIS mixes two
 * strategies together and would launder a biased one behind an unbiased
 * one at high sample counts, hiding exactly the thing this needs to isolate.
 *
 * Ground truth is a dense-grid direct integration of the same "double
 * cosine over distance squared" form factor the solid-angle referee already
 * validated, now also weighted by the SURFACE's cosine term - the actual
 * quantity a Lambertian direct-lighting integral reduces to.
 */

#include <gtest/gtest.h>

#include "optix_math_helpers.h"
#include "sampling_helpers.h"

#include <cmath>
#include <random>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Quad {
	double Qx, Qy, Qz;
	double ux, uy, uz;
	double vx, vy, vz;
};

// Dense-grid reference: integral over the light's surface of
//   cos_light * cos_surface / dist^2  dA
// which is exactly what a Lambertian direct-lighting integral (in solid
// angle measure) reduces to once the (albedo/pi)*Le constant is factored
// out - the same "form factor" quantity the solid-angle referee already
// integrates, extended with the surface-side cosine.
double referenceFormFactor(
	double px, double py, double pz,
	double nx, double ny, double nz,       // surface normal at p
	const Quad &q, int steps = 1400)
{
	const double crossx = q.uy*q.vz - q.uz*q.vy;
	const double crossy = q.uz*q.vx - q.ux*q.vz;
	const double crossz = q.ux*q.vy - q.uy*q.vx;
	const double area = std::sqrt(crossx*crossx + crossy*crossy + crossz*crossz);
	const double lnx = crossx/area, lny = crossy/area, lnz = crossz/area;

	const double dA = area / (double(steps) * double(steps));
	double ff = 0.0;
	for (int i = 0; i < steps; ++i) {
		const double a = (i + 0.5) / steps;
		for (int j = 0; j < steps; ++j) {
			const double b = (j + 0.5) / steps;
			const double sx = q.Qx + a*q.ux + b*q.vx;
			const double sy = q.Qy + a*q.uy + b*q.vy;
			const double sz = q.Qz + a*q.uz + b*q.vz;
			double dx = sx - px, dy = sy - py, dz = sz - pz;
			const double d2 = dx*dx + dy*dy + dz*dz;
			const double d = std::sqrt(d2);
			dx /= d; dy /= d; dz /= d;
			const double cosLight = std::fabs(dx*lnx + dy*lny + dz*lnz);
			const double cosSurf = std::max(0.0, dx*nx + dy*ny + dz*nz);
			ff += (cosLight * cosSurf / d2) * dA;
		}
	}
	return ff;
}

// GPU's sample_quad_light(), transcribed verbatim from
// gpu/optix/optix_device_helpers.h (float precision there; double here only
// because everything else in this file is double - the formula, not the
// width, is under test). Returns the light-sampling-only NEE contribution to
// the form factor: cos_surface / pdf, i.e. what a pure (non-MIS) light
// estimator would divide by.
double gpuStyleFormFactorSample(
	double px, double py, double pz,
	double nx, double ny, double nz,
	const Quad &q, double a, double b)
{
	const double crossx = q.uy*q.vz - q.uz*q.vy;
	const double crossy = q.uz*q.vx - q.ux*q.vz;
	const double crossz = q.ux*q.vy - q.uy*q.vx;
	const double area = std::sqrt(crossx*crossx + crossy*crossy + crossz*crossz);
	const double lnx = crossx/area, lny = crossy/area, lnz = crossz/area;

	const double sx = q.Qx + a*q.ux + b*q.vx;
	const double sy = q.Qy + a*q.uy + b*q.vy;
	const double sz = q.Qz + a*q.uz + b*q.vz;
	double dx = sx - px, dy = sy - py, dz = sz - pz;
	const double d2 = dx*dx + dy*dy + dz*dz;
	const double d = std::sqrt(d2);
	dx /= d; dy /= d; dz /= d;

	const double cosLight = std::fabs(dx*lnx + dy*lny + dz*lnz);
	if (cosLight < 1e-6 || area < 1e-6) return 0.0;
	const double pdf = d2 / (cosLight * area);   // GPU's sample_quad_light()

	const double cosSurf = std::max(0.0, dx*nx + dy*ny + dz*nz);
	return cosSurf / pdf;
}

// CPU's quad::random()/pdf_value(), via the real production functions
// (SampleSphericalRectangle / its own pdf output - src/shared/sampling_helpers.h),
// not a reimplementation.
double cpuStyleFormFactorSample(
	double px, double py, double pz,
	double nx, double ny, double nz,
	const Quad &q, double u0, double u1)
{
	double sx, sy, sz, pdf;
	SampleSphericalRectangle(
		px, py, pz, q.Qx, q.Qy, q.Qz, q.ux, q.uy, q.uz, q.vx, q.vy, q.vz,
		u0, u1, &sx, &sy, &sz, &pdf);
	if (pdf <= 0.0) return 0.0;

	double dx = sx - px, dy = sy - py, dz = sz - pz;
	const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
	dx /= d; dy /= d; dz /= d;
	const double cosSurf = std::max(0.0, dx*nx + dy*ny + dz*nz);
	return cosSurf / pdf;
}

struct Estimate { double mean; double stderr_; };

Estimate monteCarlo(
	double px, double py, double pz, double nx, double ny, double nz,
	const Quad &q, bool gpuStyle, long N)
{
	std::mt19937_64 rng(12345);
	std::uniform_real_distribution<double> uni(0.0, 1.0);
	double sum = 0.0, sumSq = 0.0;
	for (long i = 0; i < N; ++i) {
		const double a = uni(rng), b = uni(rng);
		const double v = gpuStyle
			? gpuStyleFormFactorSample(px,py,pz, nx,ny,nz, q, a, b)
			: cpuStyleFormFactorSample(px,py,pz, nx,ny,nz, q, a, b);
		sum += v;
		sumSq += v*v;
	}
	const double mean = sum / double(N);
	const double variance = sumSq/double(N) - mean*mean;
	const double stderr_ = std::sqrt(std::max(0.0, variance) / double(N));
	return {mean, stderr_};
}

} // namespace

// Directly-below case: closed-form-checkable geometry, light facing straight
// down at a point facing straight up.
TEST(QuadLightEstimatorBiasTest, DirectlyBelow_BothEstimatorsMatchReference) {
	const Quad q{-1,2,-1,  2,0,0,  0,0,2};
	const double px=0, py=0, pz=0, nx=0, ny=1, nz=0;
	const double reference = referenceFormFactor(px,py,pz, nx,ny,nz, q);
	ASSERT_GT(reference, 0.0);

	const long N = 2000000;
	Estimate gpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/true, N);
	Estimate cpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/false, N);

	// 6 sigma bounds it own noise; the two backends' 3-5% gap is roughly
	// 40-60 standard errors at this N if either estimator were biased.
	EXPECT_NEAR(gpu.mean, reference, 6*gpu.stderr_ + 1e-9)
		<< "GPU-style estimator: mean=" << gpu.mean << " reference=" << reference
		<< " ratio=" << gpu.mean/reference << " stderr=" << gpu.stderr_;
	EXPECT_NEAR(cpu.mean, reference, 6*cpu.stderr_ + 1e-9)
		<< "CPU-style estimator: mean=" << cpu.mean << " reference=" << reference
		<< " ratio=" << cpu.mean/reference << " stderr=" << cpu.stderr_;
}

// The actual failing case: Cornell-box ceiling light geometry and a floor
// point directly below it, matching spherical_rectangle_adjudication_tests.cpp's
// own "cornell ceiling light" case and the real scene 0 proportions.
TEST(QuadLightEstimatorBiasTest, CornellCeilingLight_BothEstimatorsMatchReference) {
	const Quad q{213,548.7,227,  130,0,0,  0,0,105};
	const double px=278, py=50, pz=279, nx=0, ny=1, nz=0;
	const double reference = referenceFormFactor(px,py,pz, nx,ny,nz, q);
	ASSERT_GT(reference, 0.0);

	const long N = 2000000;
	Estimate gpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/true, N);
	Estimate cpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/false, N);

	EXPECT_NEAR(gpu.mean, reference, 6*gpu.stderr_ + 1e-9)
		<< "GPU-style estimator: mean=" << gpu.mean << " reference=" << reference
		<< " ratio=" << gpu.mean/reference << " stderr=" << gpu.stderr_;
	EXPECT_NEAR(cpu.mean, reference, 6*cpu.stderr_ + 1e-9)
		<< "CPU-style estimator: mean=" << cpu.mean << " reference=" << reference
		<< " ratio=" << cpu.mean/reference << " stderr=" << cpu.stderr_;
}

// Off-axis: the light is not directly overhead, so cos_light and cos_surface
// vary independently across the quad rather than staying near their
// maximum - the case most likely to expose a Jacobian mistake that a
// straight-down geometry could hide by symmetry.
TEST(QuadLightEstimatorBiasTest, OffAxis_BothEstimatorsMatchReference) {
	const Quad q{213,548.7,227,  130,0,0,  0,0,105};
	const double px=460, py=50, pz=120, nx=0, ny=1, nz=0;
	const double reference = referenceFormFactor(px,py,pz, nx,ny,nz, q);
	ASSERT_GT(reference, 0.0);

	const long N = 2000000;
	Estimate gpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/true, N);
	Estimate cpu = monteCarlo(px,py,pz, nx,ny,nz, q, /*gpuStyle=*/false, N);

	EXPECT_NEAR(gpu.mean, reference, 6*gpu.stderr_ + 1e-9)
		<< "GPU-style estimator: mean=" << gpu.mean << " reference=" << reference
		<< " ratio=" << gpu.mean/reference << " stderr=" << gpu.stderr_;
	EXPECT_NEAR(cpu.mean, reference, 6*cpu.stderr_ + 1e-9)
		<< "CPU-style estimator: mean=" << cpu.mean << " reference=" << reference
		<< " ratio=" << cpu.mean/reference << " stderr=" << cpu.stderr_;
}
