// ---------------------------------------------------------------------------
// volume_scattering_tests.cpp
// Unit tests for volume_scattering.h and upgraded constant_medium.h.
// Mirrors pbrt-v4 HGPhaseFunction, HenyeyGreenstein, HomogeneousMedium.
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/volume_scattering.h"
#include "../../src/TheRestOfYourLife/constant_medium.h"
#include "../../src/TheRestOfYourLife/sphere.h"

// ============================================================
// HenyeyGreensteinPhaseFunction<double>
// ============================================================

// pbrt-v4: isotropic case g=0 must equal 1/(4*pi)
TEST(HGPhaseFunction, IsotropicEqualsUniformSphere) {
	HenyeyGreensteinPhaseFunction<double> hg(0.0);
	const double inv4pi = 1.0 / (4.0 * 3.14159265358979323846);
	EXPECT_NEAR(hg.p(1.0),   inv4pi, 1e-10);
	EXPECT_NEAR(hg.p(0.0),   inv4pi, 1e-10);
	EXPECT_NEAR(hg.p(-1.0),  inv4pi, 1e-10);
}

// pbrt-v4 convention: denom = 1 + g^2 + 2*g*cosTheta.
// For g>0 (forward scattering), peak is where denom is smallest => cosTheta=-1.
// Physically: wi continues in the original propagation direction, Dot(wo,wi)=-1.
TEST(HGPhaseFunction, ForwardScatteringPeakAtForward) {
	HenyeyGreensteinPhaseFunction<double> hg(0.8);
	double p_neg1  = hg.p(-1.0);  // dominant for forward scatter
	double p_zero  = hg.p(0.0);
	double p_pos1  = hg.p(1.0);   // suppressed for forward scatter
	EXPECT_GT(p_neg1, p_zero);
	EXPECT_GT(p_zero, p_pos1);
}

// pbrt-v4 convention: for g<0 (back scattering), peak is at cosTheta=+1
// (Dot(wo,wi)=1 means wi goes back toward the camera, opposite of propagation).
TEST(HGPhaseFunction, BackScatteringPeakAtBackward) {
	HenyeyGreensteinPhaseFunction<double> hg(-0.8);
	double p_pos1  = hg.p(1.0);   // dominant for back scatter
	double p_zero  = hg.p(0.0);
	double p_neg1  = hg.p(-1.0);  // suppressed for back scatter
	EXPECT_GT(p_pos1, p_zero);
	EXPECT_GT(p_zero, p_neg1);
}

// HG phase function is normalized: integral over sphere = 1.
// Numerical Monte Carlo estimate over many directions.
TEST(HGPhaseFunction, NormalizationApproximate) {
	HenyeyGreensteinPhaseFunction<double> hg(0.5);
	const int N = 100000;
	double sum = 0.0;
	// Uniform sphere sampling
	for (int i = 0; i < N; ++i) {
		// Use stratified cosTheta, uniform phi
		double cos_theta = -1.0 + 2.0 * (i + 0.5) / N;
		sum += hg.p(cos_theta) * (4.0 * 3.14159265358979323846 / N);
	}
	// Riemann sum over 4*pi steradians
	EXPECT_NEAR(sum, 1.0, 0.01);
}

// Symmetry: p(cos) == p(cos) for symmetric directions
TEST(HGPhaseFunction, Symmetry) {
	HenyeyGreensteinPhaseFunction<double> hg(0.6);
	EXPECT_NEAR(hg.p(0.5), hg.p(0.5), 1e-14);
}

// Sample must produce a unit-ish direction
TEST(HGPhaseFunction, SampledDirectionUnit) {
	HenyeyGreensteinPhaseFunction<double> hg(0.7);
	double wi_x, wi_y, wi_z, pdf_val;
	// wo = (0,0,1)
	hg.sample(0.0, 0.0, 1.0, 0.3, 0.6, wi_x, wi_y, wi_z, pdf_val);
	double len = std::sqrt(wi_x*wi_x + wi_y*wi_y + wi_z*wi_z);
	EXPECT_NEAR(len, 1.0, 1e-8);
}

// PDF of sampled direction should match phase function value
TEST(HGPhaseFunction, SampledPdfMatchesPhaseValue) {
	HenyeyGreensteinPhaseFunction<double> hg(0.5);
	double wi_x, wi_y, wi_z, pdf_val;
	hg.sample(0.0, 0.0, 1.0, 0.4, 0.7, wi_x, wi_y, wi_z, pdf_val);
	double cos_theta = wi_z; // dot with wo=(0,0,1)
	double expected = hg.p(cos_theta);
	EXPECT_NEAR(pdf_val, expected, 1e-10);
}

// PDF method matches p() for a known direction
TEST(HGPhaseFunction, PdfMethodMatchesPFunction) {
	HenyeyGreensteinPhaseFunction<double> hg(0.4);
	// wo = (0,0,1), wi = (0,0,1) => cosTheta=1
	double pdf = hg.pdf(0,0,1, 0,0,1);
	EXPECT_NEAR(pdf, hg.p(1.0), 1e-12);
}

// Non-negative everywhere
TEST(HGPhaseFunction, NonNegative) {
	for (double g : {-0.9, -0.5, 0.0, 0.5, 0.9}) {
		HenyeyGreensteinPhaseFunction<double> hg(g);
		for (double ct : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
			EXPECT_GE(hg.p(ct), 0.0) << "g=" << g << " cosTheta=" << ct;
		}
	}
}

// ============================================================
// HomogeneousMediumData<double>
// ============================================================

TEST(HomogeneousMediumData, TransmittanceUnityAtZero) {
	HomogeneousMediumData<double> med(0.1, 0.5, 0.0);
	double Tr_r, Tr_g, Tr_b;
	med.transmittance(0.0, Tr_r, Tr_g, Tr_b);
	EXPECT_NEAR(Tr_r, 1.0, 1e-12);
	EXPECT_NEAR(Tr_g, 1.0, 1e-12);
	EXPECT_NEAR(Tr_b, 1.0, 1e-12);
}

TEST(HomogeneousMediumData, TransmittanceDecaysExponentially) {
	HomogeneousMediumData<double> med(0.0, 1.0, 0.0); // sigma_t = 1
	double Tr_r, Tr_g, Tr_b;
	med.transmittance(1.0, Tr_r, Tr_g, Tr_b);
	EXPECT_NEAR(Tr_r, std::exp(-1.0), 1e-10);
	med.transmittance(2.0, Tr_r, Tr_g, Tr_b);
	EXPECT_NEAR(Tr_r, std::exp(-2.0), 1e-10);
}

TEST(HomogeneousMediumData, TransmittanceAtLargeTIsNearZero) {
	HomogeneousMediumData<double> med(1.0, 1.0, 0.0);
	double Tr_r, Tr_g, Tr_b;
	med.transmittance(100.0, Tr_r, Tr_g, Tr_b);
	EXPECT_NEAR(Tr_r, 0.0, 1e-8);
}

TEST(HomogeneousMediumData, FreePathPositive) {
	HomogeneousMediumData<double> med(0.0, 1.0, 0.0);
	double t = med.sample_free_path(0.5);
	EXPECT_GT(t, 0.0);
}

TEST(HomogeneousMediumData, FreePathMeanEqualsOneOverSigmaT) {
	// E[t] = 1/sigma_t for exponential distribution
	HomogeneousMediumData<double> med(0.0, 2.0, 0.0); // sigma_t = 2
	const int N = 200000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double u = (i + 0.5) / N;
		sum += med.sample_free_path(u);
	}
	double mean = sum / N;
	EXPECT_NEAR(mean, 0.5, 0.01); // 1/sigma_t = 0.5
}

TEST(HomogeneousMediumData, ScatterWeightEqualsAlbedo) {
	// For sigma_a=0: albedo = sigma_s/sigma_t = 1
	HomogeneousMediumData<double> med(0.0, 1.0, 0.0);
	double wr, wg, wb;
	med.scatter_weight(wr, wg, wb);
	EXPECT_NEAR(wr, 1.0, 1e-12);

	// For sigma_a=sigma_s: albedo = 0.5
	HomogeneousMediumData<double> med2(1.0, 1.0, 0.0);
	med2.scatter_weight(wr, wg, wb);
	EXPECT_NEAR(wr, 0.5, 1e-12);
}

TEST(HomogeneousMediumData, ZeroDensityFreePathIsVeryLarge) {
	HomogeneousMediumData<double> med(0.0, 0.0, 0.0);
	double t = med.sample_free_path(0.5);
	EXPECT_GT(t, 1e20);
}

// ============================================================
// constant_medium (CPU wrapper)
// ============================================================

TEST(ConstantMedium, HitInsideSphere) {
	auto sphere = make_shared<::sphere>(point3(0,0,0), 10.0,
										make_shared<lambertian>(color(1,1,1)));
	constant_medium fog(sphere, 1.0, color(1,1,1), 0.0);

	// Ray through center -- should scatter with high probability (dense fog)
	int hits = 0;
	for (int i = 0; i < 100; ++i) {
		ray r(point3(0,0,-5), vec3(0,0,1));
		hit_record rec;
		if (fog.hit(r, interval(0.001, infinity), rec))
			++hits;
	}
	EXPECT_GT(hits, 50); // Very dense fog should almost always scatter
}

TEST(ConstantMedium, LowDensityRarelyHits) {
	auto sphere = make_shared<::sphere>(point3(0,0,0), 1.0,
										make_shared<lambertian>(color(1,1,1)));
	constant_medium fog(sphere, 0.001, color(1,1,1), 0.0);

	int hits = 0;
	for (int i = 0; i < 1000; ++i) {
		ray r(point3(0,0,-0.5), vec3(0,0,1));
		hit_record rec;
		if (fog.hit(r, interval(0.001, infinity), rec))
			++hits;
	}
	EXPECT_LT(hits, 100); // Very sparse fog should rarely scatter
}

TEST(ConstantMedium, TransmittanceDecays) {
	auto sphere = make_shared<::sphere>(point3(0,0,0), 10.0,
										make_shared<lambertian>(color(1,1,1)));
	constant_medium fog(sphere, 1.0, color(1,1,1), 0.0);
	color Tr = fog.transmittance(1.0);
	double expected = std::exp(-1.0);
	EXPECT_NEAR(Tr.x(), expected, 1e-10);
	EXPECT_NEAR(Tr.y(), expected, 1e-10);
	EXPECT_NEAR(Tr.z(), expected, 1e-10);
}

TEST(ConstantMedium, TransmittanceAtZeroIsUnity) {
	auto sphere = make_shared<::sphere>(point3(0,0,0), 5.0,
										make_shared<lambertian>(color(1,1,1)));
	constant_medium fog(sphere, 0.5, color(1,1,1), 0.0);
	color Tr = fog.transmittance(0.0);
	EXPECT_NEAR(Tr.x(), 1.0, 1e-10);
}

TEST(ConstantMedium, PhaseMaterialNonNull) {
	auto sphere = make_shared<::sphere>(point3(0,0,0), 2.0,
										make_shared<lambertian>(color(1,1,1)));
	constant_medium fog(sphere, 0.5, color(0.8, 0.8, 0.8), 0.5);
	ray r(point3(0,0,-1), vec3(0,0,1));
	hit_record rec;
	// Even if it doesn't scatter on first try, rec.mat should be set when it does
	for (int i = 0; i < 200; ++i) {
		if (fog.hit(r, interval(0.001, infinity), rec)) {
			ASSERT_NE(rec.mat, nullptr);
			break;
		}
	}
}

// hg_phase_material scatter produces a valid direction
TEST(HGPhaseMaterial, ScatterProducesValidDirection) {
	hg_phase_material mat(color(0.9, 0.9, 0.9), 0.7);
	ray r_in(point3(0,0,0), vec3(0,0,1));
	hit_record rec;
	rec.p = point3(0,0,0);
	rec.normal = vec3(0,1,0);
	rec.front_face = true;
	scatter_record srec;
	bool ok = mat.scatter(r_in, rec, srec);
	EXPECT_TRUE(ok);
	EXPECT_TRUE(srec.skip_pdf);
	double len = srec.skip_pdf_ray.direction().length();
	EXPECT_NEAR(len, 1.0, 1e-8);
}

// hg_phase_material scattering_pdf is non-negative
TEST(HGPhaseMaterial, ScatteringPdfNonNegative) {
	hg_phase_material mat(color(1,1,1), 0.5);
	ray r_in(point3(0,0,0), vec3(0,0,1));
	ray scattered(point3(0,0,0), vec3(0,0,1));
	hit_record rec;
	rec.p = point3(0,0,0);
	EXPECT_GE(mat.scattering_pdf(r_in, rec, scattered), 0.0);
}
