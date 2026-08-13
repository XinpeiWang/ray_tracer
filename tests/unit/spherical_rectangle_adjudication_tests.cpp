/**
 * @file spherical_rectangle_adjudication_tests.cpp
 * @brief Which quad-light estimator is right?
 *
 * The CPU and GPU path tracers disagree by 3-5% on every scene lit by a
 * PLANAR area light, and agree to 0.06% on a scene with no area light at all
 * and to 0.7% on one lit by spheres. The gap is already there at max_depth=1,
 * so it is in the direct-lighting term rather than something that accumulates.
 *
 * The two backends sample a quad light differently, and both pair their
 * sampler with a matching pdf, so each is internally consistent:
 *
 *   CPU  quad::random()/pdf_value() - uniform in SOLID ANGLE over the
 *        rectangle (Urena et al. 2013), pdf = 1/omega
 *   GPU  sample_quad_light()        - uniform over AREA, pdf = d^2/(cos*A)
 *
 * Two internally consistent unbiased estimators of the same integral must
 * converge to the same answer. They do not, so one of them is wrong - and
 * neither backend's output can settle which, since that is exactly what is in
 * dispute. This adjudicates against arithmetic instead: the solid angle a
 * rectangle subtends can be integrated directly over its own surface, which
 * needs nothing from either sampler.
 *
 *      omega = integral over A of cos(theta) / d^2  dA
 *
 * That is elementary, slow, and hard to get wrong - the qualities wanted in a
 * referee. If SphericalRectangleSolidAngle matches it, the CPU's estimator is
 * sound and suspicion moves to the GPU; if it does not, this is the bug, and
 * the sign tells which way the error runs.
 */

#include <gtest/gtest.h>

#include "sampling_helpers.h"

#include <cmath>

namespace {

// Brute-force reference: integrate cos(theta)/d^2 over the rectangle's own
// surface with a dense regular grid (midpoint rule). No sampling, no RNG - the
// same answer every run.
double referenceSolidAngle(
	double px, double py, double pz,
	double qx, double qy, double qz,
	double ux, double uy, double uz,
	double vx, double vy, double vz,
	int steps = 900)
{
	// Rectangle normal and area from its two edge vectors.
	const double nx = uy*vz - uz*vy;
	const double ny = uz*vx - ux*vz;
	const double nz = ux*vy - uy*vx;
	const double nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
	const double area = nlen;                       // |u x v|
	const double nhx = nx/nlen, nhy = ny/nlen, nhz = nz/nlen;

	const double dA = area / (double(steps) * double(steps));
	double omega = 0.0;
	for (int i = 0; i < steps; ++i) {
		const double a = (i + 0.5) / steps;
		for (int j = 0; j < steps; ++j) {
			const double b = (j + 0.5) / steps;
			const double sx = qx + a*ux + b*vx;
			const double sy = qy + a*uy + b*vy;
			const double sz = qz + a*uz + b*vz;
			double dx = sx - px, dy = sy - py, dz = sz - pz;
			const double d2 = dx*dx + dy*dy + dz*dz;
			const double d = std::sqrt(d2);
			dx /= d; dy /= d; dz /= d;
			const double cosTheta = std::fabs(dx*nhx + dy*nhy + dz*nhz);
			omega += cosTheta / d2 * dA;
		}
	}
	return omega;
}

struct Case {
	const char *name;
	double p[3];
	double q[3], u[3], v[3];
};

// A point directly below the centre of an axis-aligned rectangle, plus
// off-axis and oblique views of the same one. The first has a closed form to
// cross-check the referee itself.
const Case kCases[] = {
	{"directly below centre",   {0, 0, 0},   {-1, 2, -1}, {2, 0, 0}, {0, 0, 2}},
	{"off to one side",         {3, 0, 0},   {-1, 2, -1}, {2, 0, 0}, {0, 0, 2}},
	{"oblique and distant",     {5, -3, 4},  {-1, 2, -1}, {2, 0, 0}, {0, 0, 2}},
	{"close to one edge",       {0.9, 0.2, 0}, {-1, 2, -1}, {2, 0, 0}, {0, 0, 2}},
	// Cornell-box proportions: the real case the renders disagree on.
	{"cornell ceiling light",   {278, 50, 279}, {213, 548.7, 227}, {130, 0, 0}, {0, 0, 105}},
};

} // namespace

// Sanity-check the referee before trusting it to judge. A rectangle of
// half-width a and half-height b at distance d directly above a point subtends
// 4*arctan(a*b / (d*sqrt(a^2+b^2+d^2))).
TEST(SphericalRectangleAdjudicationTest, TheRefereeMatchesTheClosedFormItCanBeCheckedAgainst) {
	const double a = 1.0, b = 1.0, d = 2.0;
	const double closedForm =
		4.0 * std::atan(a * b / (d * std::sqrt(a*a + b*b + d*d)));
	const double brute = referenceSolidAngle(0,0,0, -1,2,-1, 2,0,0, 0,0,2);
	EXPECT_NEAR(brute, closedForm, 1e-4)
		<< "the brute-force integrator disagrees with the analytic answer, so "
		   "it cannot be used to judge anything else";
}

TEST(SphericalRectangleAdjudicationTest, SolidAngleMatchesDirectIntegration) {
	for (const Case &c : kCases) {
		const double reference = referenceSolidAngle(
			c.p[0], c.p[1], c.p[2], c.q[0], c.q[1], c.q[2],
			c.u[0], c.u[1], c.u[2], c.v[0], c.v[1], c.v[2]);
		const double actual = SphericalRectangleSolidAngle(
			c.p[0], c.p[1], c.p[2], c.q[0], c.q[1], c.q[2],
			c.u[0], c.u[1], c.u[2], c.v[0], c.v[1], c.v[2]);

		ASSERT_GT(reference, 0.0) << c.name << ": degenerate test case";
		// 0.2% - loose enough for the midpoint rule's own error, far tighter
		// than the 3-5% the two renderers differ by.
		EXPECT_NEAR(actual / reference, 1.0, 2e-3)
			<< c.name << ": SphericalRectangleSolidAngle = " << actual
			<< ", direct integration = " << reference
			<< " (ratio " << actual / reference << "). A solid angle that is "
			   "too small makes 1/omega too large, which makes every quad-lit "
			   "surface too DARK.";
	}
}
