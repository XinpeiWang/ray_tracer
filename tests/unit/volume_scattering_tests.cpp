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

// hg_phase_material scatter produces a valid direction via its pdf_ptr.
// skip_pdf is deliberately false here (see hg_phase_material::scatter()'s
// own comment): unlike a specular/delta BSDF, HG is smooth and benefits
// from NEE, so the integrator samples a direction through srec.pdf_ptr
// (generate()/value()) the same way it does for lambertian/cosine_pdf,
// rather than reading a pre-populated skip_pdf_ray. This test used to
// assert the old skip_pdf=true, skip_pdf_ray-populated contract and went
// stale when that NEE fix landed - skip_pdf_ray is simply never touched
// on this path now, so it correctly reads back as a zero direction.
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
	EXPECT_FALSE(srec.skip_pdf);
	ASSERT_NE(srec.pdf_ptr, nullptr);
	vec3 dir = srec.pdf_ptr->generate();
	double len = dir.length();
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

// ============================================================
// pbrt-v4 parity: HenyeyGreenstein media_test.cpp
// ============================================================

// pbrt-v4: TEST(HenyeyGreenstein, SamplingMatch)
// Phase function is normalized and sampling is exact: p == pdf for sampled wi.
TEST(HGPhaseFunction, SamplingMatch) {
	// Simple LCG-style deterministic sequence
	auto lcg = [](uint32_t &s) -> double {
		s = s * 1664525u + 1013904223u;
		return (s >> 8) * (1.0 / (1u << 24));
	};
	uint32_t seed = 12345u;
	for (double g = -0.75; g <= 0.75; g += 0.25) {
		HenyeyGreensteinPhaseFunction<double> hg(g);
		for (int i = 0; i < 100; ++i) {
			double wox = 0.0, woy = 0.0, woz = 1.0; // fixed wo = +z
			double u1 = lcg(seed), u2 = lcg(seed);
			double wi_x, wi_y, wi_z, pdf_val;
			hg.sample(wox, woy, woz, u1, u2, wi_x, wi_y, wi_z, pdf_val);
			double cos_theta = wox*wi_x + woy*wi_y + woz*wi_z;
			double p_val = hg.p(cos_theta);
			EXPECT_NEAR(pdf_val, p_val, 1e-10) << "g=" << g << " i=" << i;
		}
	}
}

// pbrt-v4: TEST(HenyeyGreenstein, SamplingOrientationForward)
// With g=0.95 almost all sampled wi are in the same hemisphere as wo.
TEST(HGPhaseFunction, SamplingOrientationForward) {
	HenyeyGreensteinPhaseFunction<double> hg(0.95);
	double wox = -1.0, woy = 0.0, woz = 0.0;
	auto lcg = [](uint32_t &s) -> double {
		s = s * 1664525u + 1013904223u;
		return (s >> 8) * (1.0 / (1u << 24));
	};
	uint32_t seed = 99u;
	int nForward = 0, nBackward = 0;
	for (int i = 0; i < 100; ++i) {
		double u1 = lcg(seed), u2 = lcg(seed);
		double wi_x, wi_y, wi_z, pdf_val;
		hg.sample(wox, woy, woz, u1, u2, wi_x, wi_y, wi_z, pdf_val);
		// wi.x > 0 means it continued in the -wo direction (forward scatter)
		if (wi_x > 0) ++nForward; else ++nBackward;
	}
	EXPECT_GE(nForward, 10 * nBackward);
}

// pbrt-v4: TEST(HenyeyGreenstein, SamplingOrientationBackward)
// With g=-0.95 almost all sampled wi are in the back hemisphere.
TEST(HGPhaseFunction, SamplingOrientationBackward) {
	HenyeyGreensteinPhaseFunction<double> hg(-0.95);
	double wox = -1.0, woy = 0.0, woz = 0.0;
	auto lcg = [](uint32_t &s) -> double {
		s = s * 1664525u + 1013904223u;
		return (s >> 8) * (1.0 / (1u << 24));
	};
	uint32_t seed = 77u;
	int nForward = 0, nBackward = 0;
	for (int i = 0; i < 100; ++i) {
		double u1 = lcg(seed), u2 = lcg(seed);
		double wi_x, wi_y, wi_z, pdf_val;
		hg.sample(wox, woy, woz, u1, u2, wi_x, wi_y, wi_z, pdf_val);
		if (wi_x > 0) ++nForward; else ++nBackward;
	}
	EXPECT_GE(nBackward, 10 * nForward);
}

// pbrt-v4: TEST(HenyeyGreenstein, g)
// MC estimate of mean cosine must recover the g parameter.
// In our HG convention p uses cos_theta = Dot(wo,wi); g = -E[Dot(wo,wi)]
// (when g>0, forward-scatter has wi near -wo, so Dot(wo,wi) is near -1).
TEST(HGPhaseFunction, GParameterEstimate) {
	auto lcg = [](uint32_t &s) -> double {
		s = s * 1664525u + 1013904223u;
		return (s >> 8) * (1.0 / (1u << 24));
	};
	uint32_t seed = 55555u;
	for (double g : {-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75}) {
		HenyeyGreensteinPhaseFunction<double> hg(g);
		double wox = 0.0, woy = 0.0, woz = 1.0;
		double sum = 0.0;
		const int N = 8192;
		for (int i = 0; i < N; ++i) {
			double u1 = lcg(seed), u2 = lcg(seed);
			double wi_x, wi_y, wi_z, pdf_val;
			hg.sample(wox, woy, woz, u1, u2, wi_x, wi_y, wi_z, pdf_val);
			double cos_t = wox*wi_x + woy*wi_y + woz*wi_z;
			sum += cos_t; // E[cos_theta] = -g in our convention
		}
		double gEst = -(sum / N); // negate to recover g
		EXPECT_NEAR(g, gEst, 0.05) << "g=" << g;
	}
}

// ============================================================
// sample_homogeneous_event<double>: single-sample MIS over the colour channels
// ============================================================

namespace {
// Mean per-channel path weight of `n` stratified-ish draws of one free flight of length d: what the estimator contributes in
// expectation, split into the collision part (w, e) and the pass-through part.
struct EventMeans { double collide_w[3] = {0, 0, 0}, collide_e[3] = {0, 0, 0}, pass_w[3] = {0, 0, 0}; };

EventMeans measure_events(const HomogeneousMediumData<double>& m, double d, int n) {
	EventMeans out;
	uint32_t s = 20260705u;
	auto u = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) + 0.5) / double(1u << 24); };
	for (int i = 0; i < n; ++i) {
		const HomogeneousEvent<double> ev = sample_homogeneous_event<double>(m, d, u(), u());
		for (int c = 0; c < 3; ++c) {
			if (ev.collided) { out.collide_w[c] += ev.w[c] / n; out.collide_e[c] += ev.e[c] / n; }
			else             { out.pass_w[c] += ev.w[c] / n; }
		}
	}
	return out;
}
}

// A pure absorber thicker in blue than red: collisions carry no scattering weight, and the pass-through weights must average to the
// per-channel Beer-Lambert transmittance (not the luminance-collapsed one the scalar model gives every channel).
TEST(HomogeneousEvent, PureAbsorberPassesPerChannelBeerLambert) {
	HomogeneousMediumData<double> m(0.1, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0);
	const double d = 2.0;
	const EventMeans mean = measure_events(m, d, 400000);
	const double sig[3] = {0.1, 0.4, 0.9};
	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(mean.pass_w[c], std::exp(-sig[c] * d), 0.004) << "channel " << c;
		EXPECT_DOUBLE_EQ(mean.collide_w[c], 0.0) << "an absorber never scatters, channel " << c;
	}
}

// Emission of an absorber: the integral of sigma_a * T over the chord is 1 - T for a pure absorber, per channel.
TEST(HomogeneousEvent, AbsorberEmissionWeightsIntegrateToOneMinusTransmittance) {
	HomogeneousMediumData<double> m(0.1, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0);
	const double d = 2.0;
	const EventMeans mean = measure_events(m, d, 400000);
	const double sig[3] = {0.1, 0.4, 0.9};
	for (int c = 0; c < 3; ++c)
		EXPECT_NEAR(mean.collide_e[c], 1.0 - std::exp(-sig[c] * d), 0.004) << "channel " << c;
}

// A pure scatterer with channel-dependent sigma_s: a ray ends up either scattered or through, and in expectation each channel's
// scattered + transmitted weight is exactly 1 (nothing absorbed).
TEST(HomogeneousEvent, PureScattererConservesEnergyPerChannel) {
	HomogeneousMediumData<double> m(0.0, 0.0, 0.0, 0.3, 0.8, 1.5, 0.0);
	const double d = 1.7;
	const EventMeans mean = measure_events(m, d, 400000);
	const double sig[3] = {0.3, 0.8, 1.5};
	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(mean.pass_w[c], std::exp(-sig[c] * d), 0.004) << "channel " << c;
		EXPECT_NEAR(mean.collide_w[c] + mean.pass_w[c], 1.0, 0.006) << "channel " << c;
	}
}

// A grey medium is the old model exactly: the collision weight is the single-scattering albedo, the pass weight is 1, and the
// collision distance is the usual -log(1-u)/sigma_t.
TEST(HomogeneousEvent, GreyMediumReducesToTheScalarModel) {
	HomogeneousMediumData<double> m(0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.0);
	EXPECT_FALSE(m.is_chromatic());
	for (double ud : {0.05, 0.4, 0.9}) {
		const HomogeneousEvent<double> ev = sample_homogeneous_event<double>(m, 3.0, 0.5, ud);
		const double t = -std::log(1.0 - ud) / 2.0;
		if (t < 3.0) {
			ASSERT_TRUE(ev.collided);
			EXPECT_NEAR(ev.t, t, 1e-12);
			for (int c = 0; c < 3; ++c) EXPECT_NEAR(ev.w[c], 0.75, 1e-12);
		} else {
			EXPECT_FALSE(ev.collided);
			for (int c = 0; c < 3; ++c) EXPECT_NEAR(ev.w[c], 1.0, 1e-12);
		}
	}
}

// An unbounded flight (d >= 1e29) through a medium that is clear in one channel: only that channel survives, and the weights still
// average to the true transmittance (0, 0, 1).
TEST(HomogeneousEvent, UnboundedFlightKeepsTheClearChannel) {
	HomogeneousMediumData<double> m(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0);
	const EventMeans mean = measure_events(m, 1e30, 200000);
	EXPECT_NEAR(mean.pass_w[0], 1.0, 0.01);
	EXPECT_NEAR(mean.pass_w[1], 0.0, 1e-12);
	EXPECT_NEAR(mean.pass_w[2], 0.0, 1e-12);
}

// ============================================================
// heterogeneous_tracking_step / heterogeneous_ratio_step: spectral tracking with a shared scalar majorant
// ============================================================

namespace {
// One flight of length L through a medium with the given constant per-channel coefficients, tracked against `majorant` (> max sigma_t, so
// null events happen): accumulates, per channel, the expected weight of a path that scatters (collide_w), emits (collide_e), and passes (pass_w).
void track_flights(const double sa[3], const double ss[3], double majorant, double L, int n, double collide_w[3], double collide_e[3], double pass_w[3]) {
	uint32_t s = 91827364u;
	auto u = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) + 0.5) / double(1u << 24); };
	for (int c = 0; c < 3; ++c) collide_w[c] = collide_e[c] = pass_w[c] = 0.0;
	for (int i = 0; i < n; ++i) {
		double w[3] = {1.0, 1.0, 1.0}, cw[3], ce[3];
		double t = 0.0;
		bool real = false;
		for (;;) {
			t += -std::log(1.0 - u()) / majorant;
			if (t >= L) break;
			if (heterogeneous_tracking_step<3, double>(sa, ss, majorant, u(), w, cw, ce)) { real = true; break; }
		}
		for (int c = 0; c < 3; ++c) {
			if (real) { collide_w[c] += cw[c] / n; collide_e[c] += ce[c] / n; }
			else      { pass_w[c] += w[c] / n; }
		}
	}
}
}

// A pure absorber thicker in blue: the null-collision weights of the flights that do not end in a real (absorbing) event must average to the
// per-channel Beer-Lambert transmittance, although every flight is tracked against one scalar majorant.
TEST(HeterogeneousTracking, NullWeightsGivePerChannelBeerLambert) {
	const double sa[3] = {0.1, 0.4, 0.9}, ss[3] = {0.0, 0.0, 0.0};
	double cw[3], ce[3], pw[3];
	track_flights(sa, ss, /*majorant=*/1.3, /*L=*/2.0, 300000, cw, ce, pw);
	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(pw[c], std::exp(-sa[c] * 2.0), 0.004) << "channel " << c;
		EXPECT_DOUBLE_EQ(cw[c], 0.0) << "an absorber never scatters, channel " << c;
		EXPECT_NEAR(ce[c], 1.0 - std::exp(-sa[c] * 2.0), 0.004) << "emission weight integrates to 1 - T, channel " << c;
	}
}

// Pure scatterer, different per channel: scattered + passed weight is exactly 1 per channel (nothing absorbed).
TEST(HeterogeneousTracking, ScattererConservesEnergyPerChannel) {
	const double sa[3] = {0.0, 0.0, 0.0}, ss[3] = {0.2, 0.35, 0.5};
	double cw[3], ce[3], pw[3];
	track_flights(sa, ss, 0.9, 2.0, 300000, cw, ce, pw);
	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(pw[c], std::exp(-ss[c] * 2.0), 0.004) << "channel " << c;
		EXPECT_NEAR(cw[c] + pw[c], 1.0, 0.006) << "channel " << c;
	}
}

// The old model (extinction of the brightest channel, albedo sigma_s / max sigma_s) lost energy in the lesser channels; with absorption the
// scattered share is sigma_s / sigma_t per channel, i.e. a mixed medium's scatter + pass weight is the transmittance plus the scattered fraction.
TEST(HeterogeneousTracking, AbsorbingScattererMatchesTheHomogeneousClosedForm) {
	const double sa[3] = {0.05, 0.1, 0.2}, ss[3] = {0.3, 0.2, 0.1};
	double cw[3], ce[3], pw[3];
	track_flights(sa, ss, 0.7, 1.5, 300000, cw, ce, pw);
	for (int c = 0; c < 3; ++c) {
		const double st = sa[c] + ss[c], T = std::exp(-st * 1.5);
		EXPECT_NEAR(pw[c], T, 0.004) << "channel " << c;
		EXPECT_NEAR(cw[c], (ss[c] / st) * (1.0 - T), 0.005) << "scatter share, channel " << c;
		EXPECT_NEAR(ce[c], (sa[c] / st) * (1.0 - T), 0.005) << "absorbed share, channel " << c;
	}
}

// A grey medium reduces to ordinary delta tracking: the null weight is 1 for every channel.
TEST(HeterogeneousTracking, GreyMediumNullWeightIsOne) {
	const double sa[3] = {0.1, 0.1, 0.1}, ss[3] = {0.4, 0.4, 0.4};
	double w[3] = {1.0, 1.0, 1.0}, cw[3], ce[3];
	// u large enough to force a null event: u * majorant >= mean sigma_t (0.5)
	EXPECT_FALSE((heterogeneous_tracking_step<3, double>(sa, ss, 1.0, 0.9, w, cw, ce)));
	for (int c = 0; c < 3; ++c) EXPECT_NEAR(w[c], 1.0, 1e-12);
	const double tr_in[3] = {1.0, 1.0, 1.0};
	double tr[3] = {tr_in[0], tr_in[1], tr_in[2]};
	heterogeneous_ratio_step<3, double>(sa, ss, 1.0, tr);
	for (int c = 0; c < 3; ++c) EXPECT_NEAR(tr[c], 0.5, 1e-12);
}
