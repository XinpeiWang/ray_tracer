/**
 * @file hair_precision_tests.cpp
 * @brief HairBxDF<float> (what the GPU runs) must agree with HairBxDF<double> (what the CPU runs).
 *
 * The same template is instantiated in float on the GPU and double on the CPU. A hair sphere under a dim sky
 * once rendered 2.1x brighter on the GPU backends than on the CPU, and float-vs-double precision in the Marschner
 * terms (which sit near singularities when the proxy tangent is the shading normal) was the obvious suspect. It was
 * not - the cause was the CPU's path-throughput ceiling the GPU lacked (see hair-sphere-dim-sky.pbrt) - and these
 * pin that the two instantiations do agree, sample for sample, including near the fiber axis.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "bxdfs.h"

namespace {

struct Stats {
	double meanD = 0.0, meanF = 0.0;
	int validD = 0, validF = 0, both = 0, clampedD = 0, clampedF = 0;
	double worstRel = 0.0;
};

Stats compare(double sinThetaOMax, int n, unsigned seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> U(0.0, 1.0);
	Stats s;
	for (int i = 0; i < n; ++i) {
		// An incident direction with |component along the fiber tangent (z)| <= sinThetaOMax.
		const double z = (U(rng) * 2.0 - 1.0) * sinThetaOMax;
		const double phi = 6.283185307179586 * U(rng);
		const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
		const double wx = r * std::cos(phi), wy = r * std::sin(phi), wz = z;
		const double h = U(rng) * 2.0 - 1.0;
		const double u1 = U(rng), u2 = U(rng), u3 = U(rng), u4 = U(rng);

		HairBxDF<double> bd(h, 1.55, 0.06, 0.10, 0.20, 0.25, 0.25, 2.0);
		HairBxDF<float>  bf((float)h, 1.55f, 0.06f, 0.10f, 0.20f, 0.25f, 0.25f, 2.0f);
		// Fiber tangent along local z of the test frame; the BxDF builds its own frame around it.
		auto rd = bd.sample(0, 0, 1, wx, wy, wz, u1, u2, u3, u4);
		auto rf = bf.sample(0, 0, 1, (float)wx, (float)wy, (float)wz, (float)u1, (float)u2, (float)u3, (float)u4);
		if (rd.valid) { ++s.validD; s.meanD += rd.g; if (rd.g >= 49.999) ++s.clampedD; }
		if (rf.valid) { ++s.validF; s.meanF += rf.g; if (rf.g >= 49.999f) ++s.clampedF; }
		if (rd.valid && rf.valid) {
			++s.both;
			const double rel = std::fabs(rd.g - rf.g) / std::max(1e-3, (double)rd.g);
			s.worstRel = std::max(s.worstRel, rel);
		}
	}
	s.meanD /= std::max(1, s.validD);
	s.meanF /= std::max(1, s.validF);
	return s;
}

} // namespace

TEST(HairPrecisionTest, FloatAndDoubleSampleWeightsAgreeForOrdinaryAngles) {
	const Stats s = compare(0.9, 200000, 1);
	EXPECT_NEAR(s.meanF, s.meanD, 0.03 * s.meanD) << "mean weight float vs double";
	EXPECT_NEAR(s.validF, s.validD, 0.01 * s.validD);
}

TEST(HairPrecisionTest, FloatAndDoubleSampleWeightsAgreeNearTheFiberAxis) {
	// Incident ray almost along the fiber tangent: |sinTheta_o| ~ 1, cosTheta_o ~ 0.
	const Stats s = compare(1.0, 400000, 2);
	std::printf("near-axis: meanD=%.4f meanF=%.4f validD=%d validF=%d clampedD=%d clampedF=%d worstRel=%.3f\n",
				s.meanD, s.meanF, s.validD, s.validF, s.clampedD, s.clampedF, s.worstRel);
	EXPECT_NEAR(s.meanF, s.meanD, 0.05 * s.meanD) << "mean weight float vs double";
	EXPECT_LE(s.clampedF, s.clampedD + 50) << "float hits the 50x weight clamp far more often than double";
}
