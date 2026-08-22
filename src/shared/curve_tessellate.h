#pragma once
// curve_tessellate.h -- tessellates one Bezier curve strand into a
// ring-based "tube" of quads, approximating a tapered cylindrical curve
// with a strip of bilinear patches.
//
// Why this exists: OptiX has no curve-intersection program in this
// codebase (see curve_shape_hittable.h's own comment - sphere, quad, and
// bilinear patch each have hand-written custom-intersection programs;
// curves have none). pbrt-v4's own GPU backend - the reference this
// entire curve pipeline mirrors (CurveShape's header comment: "Mirrors
// pbrt-v4 Curve") - hit the identical problem and solved it the same way:
// dice each curve into bilinear patches rather than porting the CPU's
// recursive-subdivision intersection test to CUDA (Matt Pharr's "Rendering
// the Moana Island scene on the GPU" describes the real curve-intersection
// algorithm as "a poor fit for the GPU"). This codebase already has a
// complete, tested bilinear-patch GPU pipeline (added for scene F1), so
// tessellating into that shape needs no new OptiX code at all - just this
// geometry-generation helper, called once at scene-build time.
//
// Host-only: this only ever runs once per scene, at scene-build time, not
// a per-ray hot path, so plain float host math (via splines.h, which is
// itself CPU/GPU portable but doesn't need to run on device here) is
// fine. Output is plain float[3] triples rather than any particular
// renderer's vector type, matching splines.h's own convention, so this
// stays usable from any host-side scene builder.

#include "splines.h"
#include <cmath>
#include <vector>

namespace curve_tessellate {

struct Quad {
	// Corner order matches BilinearPatchData's p(u,v) convention
	// (gpu/optix/optix_types.h: p00=(u=0,v=0), p10=(u=1,v=0),
	// p01=(u=0,v=1), p11=(u=1,v=1)) - here u runs along the tube's length
	// and v runs around its circumference.
	float p00[3], p10[3], p01[3], p11[3];
};

// Builds an orthonormal frame (x, y) perpendicular to a (unit-length)
// tangent t, via Gram-Schmidt against an arbitrary up vector. Mirrors the
// same "pick an axis that isn't nearly parallel to t" idiom used
// throughout this codebase (e.g. TheRestOfYourLife/onb.h).
inline void perpendicular_frame(const float t[3], float x[3], float y[3]) {
	float up[3] = { 0.f, 1.f, 0.f };
	if (std::fabs(t[1]) > 0.9f) { up[0] = 1.f; up[1] = 0.f; up[2] = 0.f; }

	// x = normalize(cross(up, t))
	x[0] = up[1]*t[2] - up[2]*t[1];
	x[1] = up[2]*t[0] - up[0]*t[2];
	x[2] = up[0]*t[1] - up[1]*t[0];
	float xlen = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
	x[0] /= xlen; x[1] /= xlen; x[2] /= xlen;

	// y = cross(t, x) - already unit length since t and x are orthonormal
	y[0] = t[1]*x[2] - t[2]*x[1];
	y[1] = t[2]*x[0] - t[0]*x[2];
	y[2] = t[0]*x[1] - t[1]*x[0];
}

// Tessellates a single Bezier curve segment spanning [uMin, uMax] of the
// given control points into a tapered tube of (n_length * n_radial) quads.
// width0/width1 are the tube's diameter at uMin and uMax.
//
// Assumes cp[4] are the exact control points for the [uMin, uMax] segment
// being tessellated (true for every current caller - F4's strands are each
// a single un-split segment with uMin=0/uMax=1) - a sub-interval of a
// longer spline would need CubicBezierControlPoints() reparametrization
// first, which this deliberately doesn't add until a caller needs it.
inline void tessellate(
	const float cp[4][3],
	float uMin, float uMax,
	float width0, float width1,
	int n_length, int n_radial,
	std::vector<Quad>& out)
{
	std::vector<float> ring_x((n_length + 1) * n_radial);
	std::vector<float> ring_y((n_length + 1) * n_radial);
	std::vector<float> ring_z((n_length + 1) * n_radial);

	for (int i = 0; i <= n_length; ++i) {
		float t = static_cast<float>(i) / static_cast<float>(n_length);
		float u = uMin + t * (uMax - uMin);

		float p[3], d[3];
		splines::EvaluateCubicBezierD(cp, u, p, d);
		float dlen = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
		if (dlen > 0.f) { d[0] /= dlen; d[1] /= dlen; d[2] /= dlen; }

		float x[3], y[3];
		perpendicular_frame(d, x, y);

		float width = width0 + t * (width1 - width0);
		float radius = width * 0.5f;

		for (int j = 0; j < n_radial; ++j) {
			float angle = 2.0f * 3.14159265358979323846f
			            * static_cast<float>(j) / static_cast<float>(n_radial);
			float c = std::cos(angle), s = std::sin(angle);
			int idx = i * n_radial + j;
			ring_x[idx] = p[0] + radius * (c * x[0] + s * y[0]);
			ring_y[idx] = p[1] + radius * (c * x[1] + s * y[1]);
			ring_z[idx] = p[2] + radius * (c * x[2] + s * y[2]);
		}
	}

	auto ring_point = [&](int i, int j, float out3[3]) {
		int idx = i * n_radial + j;
		out3[0] = ring_x[idx]; out3[1] = ring_y[idx]; out3[2] = ring_z[idx];
	};

	out.reserve(out.size() + static_cast<size_t>(n_length) * n_radial);
	for (int i = 0; i < n_length; ++i) {
		for (int j = 0; j < n_radial; ++j) {
			int j2 = (j + 1) % n_radial;
			Quad q;
			// p00->p10 (the "u" edge) must be the along-length direction and
			// p00->p01 (the "v" edge) the circumferential one, matching this
			// struct's own comment above and BilinearPatchData's convention -
			// so p10 varies the length-ring index i (not the angle index j)
			// and p01 varies the angle. A same-ring/different-angle pairing
			// here was a real, previously-latent bug: nothing before Round 7
			// Phase 3 ever read dpdu directionally (only cross(dpdu,dpdv) for
			// a face normal, which doesn't care which edge is "u" vs "v" -
			// swapping them just flips the pre-face-forward-correction sign,
			// harmless). Phase 3's Hair-fiber-tangent code was the first
			// consumer to actually need dpdu to mean "along the strand," and
			// with the old i/j assignment it silently got the circumferential
			// direction instead - a ~90-degree-rotated Marschner highlight on
			// every curve+hair render on both GPU backends.
			ring_point(i,     j,  q.p00);
			ring_point(i + 1, j,  q.p10);
			ring_point(i,     j2, q.p01);
			ring_point(i + 1, j2, q.p11);
			out.push_back(q);
		}
	}
}

} // namespace curve_tessellate
