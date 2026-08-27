#pragma once
//==============================================================================================
// bsdf_bridge.h -- BSDF bridge between this codebase's `hittable`/`material`
// hierarchy and the flat, scene-graph-agnostic BSDF interface that
// src/shared/sppm.h, src/shared/bdpt.h, and src/shared/mlt.h all expect
// (f(wo,wi), importance-sample, and -- for bdpt.h/mlt.h only -- a plain
// solid-angle pdf(wo,wi)).
//
// This is Layer 1 of what used to be sppm_adapter.h's own file-local
// "Layer 1 -- BSDF bridge" section, split out into its own header so it can
// be shared by BOTH sppm_adapter.h (SPPMSceneAdapter) and bdpt_adapter.h
// (BDPTSceneAdapter) without either one pulling in the other's dependencies.
//
// Originally motivated by an AliasTable ODR collision: sppm_adapter.h
// includes src/TheRestOfYourLife/power_light_sampler.h, and bdpt.h's
// Metropolis Light Transport counterpart (src/shared/mlt.h) independently
// includes src/shared/reservoir_sampler.h - both used to define their own,
// differently-implemented global `class AliasTable`, which could never both
// be visible in the same translation unit. That collision no longer exists
// (power_light_sampler.h now includes reservoir_sampler.h's AliasTable
// directly - see power_light_sampler.h's own comment), but this split is
// kept anyway: it's independently good layering regardless - the BSDF
// bridge is the part BOTH SPPMSceneAdapter and BDPTSceneAdapter actually
// need, and neither adapter has to pull in the other's full dependency set
// (power_light_sampler.h's power-weighted light list for SPPM, mlt.h's
// bootstrap-weight Markov chain machinery for MLT) just to get it.
//
// sppm_adapter.h now `#include`s this header instead of defining Layer 1
// inline -- every name that used to live there (SPPMShadingContext,
// sppm_is_delta_material, sppm_resolve_material, sppm_reconstruct_hit_record,
// sppm_bsdf_f, sppm_bsdf_sample_f) is unchanged, just relocated, so existing
// callers (tests/unit/sppm_adapter_bsdf_tests.cpp, SPPMSceneAdapter itself)
// need no changes. The one true addition is sppm_bsdf_pdf() at the bottom,
// needed by bdpt_adapter.h's BSDFPdf() (bdpt.h/mlt.h's Scene concept
// requires a real pdf(wo,wi) query that sppm.h's Scene concept never
// needed, since SPPM has no MIS between BSDF-sampling and NEE strategies).
//==============================================================================================

#include "hittable.h"
#include "material.h"
#include "pdf.h"

#include <algorithm>
#include <cmath>

// ===========================================================================
// SPPMShadingContext -- captured per-hit shading data, indexed by an opaque
// integer id by whichever adapter owns the table (SPPMSceneAdapter's
// durable_ctx_/transient_ctx_ split, or BDPTSceneAdapter's per-thread
// growable pool -- see bdpt_adapter.h). Kept under its original SPPM-era
// name (rather than renamed to something adapter-neutral) purely to avoid
// a large, purely-cosmetic rename across every existing caller; it was
// never SPPM-specific in what it stores.
// ===========================================================================
struct SPPMShadingContext {
	point3 p;
	vec3   normal;
	double u = 0.0, v = 0.0;
	shared_ptr<material> mat;
};

// True when this material instance's scatter() sets srec.skip_pdf=true --
// delegates to material::is_delta_bsdf(rec) (material_base.h) rather than a
// per-class dynamic_cast list, since that's no longer a static, per-class
// property: rough_metal/conductor/rough_dielectric/coated_diffuse/
// coated_conductor all branch on their own runtime roughness
// (TrowbridgeReitz::EffectivelySmooth()) between a true delta fast path and
// a real glossy NEE/MIS path (see each class's own is_delta_bsdf()
// override) -- a previous version of this function treated all five as
// unconditionally delta regardless of roughness, which silently starved
// SPPM/BDPT/MLT of NEE and photon deposit at any GLOSSY (non-smooth)
// instance of these materials (e.g. a frosted rough_dielectric sphere),
// even though the default path tracer already rendered them correctly via
// real NEE/MIS. metal/dielectric/thin_dielectric have no roughness
// parameter at all and remain unconditionally delta.
//
// Takes a hit_record (both call sites - sppm_adapter.h's and
// bdpt_adapter.h's own Intersect() - already have one at classification
// time) and calls the per-hit-aware is_delta_bsdf(rec) overload, not the
// bare no-arg one: rough_dielectric with a texture-bound roughness
// (material_pbrt.h) needs a real hit point to know whether THIS specific
// texel is smooth, and the no-arg version can only conservatively answer
// "never delta" for such an instance. Using the bare version here was a
// real correctness bug (not just a missed optimization): SPPM's camera
// pass classifies a hit BEFORE calling scatter(), so a conservative
// "never delta" answer made it permanently stop and record a visible
// point at a hit whose real per-hit roughness was actually near enough to
// zero for scatter_impl() to take the specular branch instead - the ray
// never continued through the glass, a visibly wrong image.
//
// Both SPPM's own algorithm (sppm.h's SPPMCameraPass/SPPMPhotonPass) and
// BDPT/MLT's (bdpt.h's BDPTRandomWalk, via hit.is_delta_bsdf) branch on
// this distinction and only ever call BSDFf/BSDFPdf for non-delta hits,
// treating delta hits via BSDFSampleF resampling alone -- matching
// pbrt-v4's own design for both integrators.
//
// diffuse_transmission/normalized_fresnel are intentionally NOT delta
// (skip_pdf=false) but are also not given full BSDFf support in this v1
// bridge -- see sppm_bsdf_f's comment. mix_material is never passed here
// directly -- the owning adapter's Intersect() resolves it down to a
// concrete sub-material first (see sppm_resolve_material() below), so by
// the time anything in this file classifies or evaluates a material, it's
// always one of the real, concrete classes checked below.
inline bool sppm_is_delta_material(const material* m, const hit_record& rec) {
	return m && m->is_delta_bsdf(rec);
}

// Same null-safe bridge shape as sppm_is_delta_material() above, forwarding
// to material::is_medium_boundary() instead - true only for
// interface_material (material_simple.h), pbrt-v4's real "no BSDF"
// interface material. A vertex classified this way is neither a real delta
// bounce nor a real non-delta one: it should be skipped entirely (no vertex
// recorded, no bounce consumed), the same "is_medium_boundary" concept
// src/shared/bdpt.h's BDPTHit and the templated Scene-concept integrators
// already use, now reachable from real Material-based geometry via
// bdpt_adapter.h/sppm_adapter.h.
inline bool sppm_is_medium_boundary(const material* m) {
	return m && m->is_medium_boundary();
}

// Resolves a mix_material down to a concrete, non-mix sub-material, matching
// mix_material::scatter()'s own stochastic weight draw exactly (same
// weight_tex lookup, same `random_double() >= w` comparison) -- recurses to
// handle a mix nested inside a mix, which pbrt-v4 allows.
//
// Why this needs to happen in Intersect() rather than being solved the way
// diffuse_transmission was (a closed-form f(wo,wi) covering both lobes):
// mix_material can blend two COMPLETELY DIFFERENT BSDFs (e.g. lambertian
// blended with a delta metal), and a delta sub-material has no finite f()
// value away from its exact reflection direction -- there's no single
// closed form that could cover both a diffuse and a delta lobe the way
// diffuse_transmission's two same-shape diffuse lobes allowed. But
// mix_material::scatter() itself doesn't attempt one either: it just draws
// one random number and delegates entirely to whichever sub-material won,
// returning that sub-material's own scatter_record unchanged. An adapter
// can copy that same trick, just earlier: resolve the winning sub-material
// once, at Intersect() time (the one place in the algorithm that runs
// before is_delta_bsdf is needed), and store THAT concrete material in the
// shading-context table instead of the raw mix_material pointer. Every
// later BSDFf/BSDFSampleF/BSDFPdf call against this hit's bsdf_id -- even
// one long after the hit was first recorded -- then sees a single,
// already-resolved, ordinary material, exactly like any other hit. This
// mirrors how the regular path tracer already treats mix_material (one
// fresh stochastic draw per scatter() call); the only difference is WHEN
// that draw happens.
inline shared_ptr<material> sppm_resolve_material(const shared_ptr<material>& mat,
                                                    double u, double v, const point3& p) {
	auto mm = std::dynamic_pointer_cast<mix_material>(mat);
	if (!mm) return mat;
	double w = mm->get_weight()->value(u, v, p).x();
	w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
	return (random_double() >= w) ? sppm_resolve_material(mm->get_mat_a(), u, v, p)
	                               : sppm_resolve_material(mm->get_mat_b(), u, v, p);
}

// Reconstructs a synthetic hit_record from a captured shading context plus
// a freshly-supplied shading normal `n`. Both sppm.h and bdpt.h always
// thread their own `n` parameter through rather than trusting a stored one,
// so shading math here honors that `n` (expected to match ctx.normal in
// non-degenerate cases; ctx.normal is kept mainly for future diagnostic
// use). front_face=true throughout: `n` is already the shading-facing
// normal by construction (the caller -- ultimately the owning adapter's
// Intersect() -- is responsible for always supplying a properly
// outward/ray-facing normal, the same convention hit_record::set_face_normal
// already enforces elsewhere in this codebase).
inline hit_record sppm_reconstruct_hit_record(const SPPMShadingContext& ctx, const double n[3]) {
	hit_record rec;
	rec.p = ctx.p;
	rec.normal = vec3(n[0], n[1], n[2]);
	rec.u = ctx.u;
	rec.v = ctx.v;
	rec.mat = ctx.mat;
	rec.front_face = true;
	rec.t = 0.0;
	return rec;
}

// f(wo,wi) for a non-specular material at a captured shading context.
// out[3] = 0 if wi is below the hemisphere, or the material has no
// evaluable BSDF. Callers must not call this for delta materials
// (sppm_is_delta_material() == true) -- neither SPPM's nor BDPT/MLT's own
// algorithm ever does, since both treat those via sppm_bsdf_sample_f()
// resampling alone.
//
// Bridges via material::scattering_pdf() (this codebase's convention: it
// returns f(wo,wi)*cos(wi), NOT a normalized pdf -- confirmed by camera.h's
// own NEE code, which computes contribution as
// `srec.attenuation * scattering_pdf(...)`) combined with
// material::scatter()'s srec.attenuation.
//
// v1 was provably exact only for `lambertian` (attenuation deterministic
// and direction-independent, so calling scatter() here for attenuation
// alone, discarding the direction it importance-samples, is safe).
// `normalized_fresnel` turns out to satisfy the exact same condition --
// its scatter() always sets attenuation=(1,1,1) regardless of any random
// draw (confirmed by reading material_pbrt.h directly), so it was already
// safe under the generic path below without needing a special case; a
// dedicated test locks this in (see tests/unit/sppm_adapter_bsdf_tests.cpp).
//
// `diffuse_transmission` genuinely needed a special case: its scatter()
// stochastically commits to reflection (attenuation=R) OR transmission
// (attenuation=T), so reusing whichever was drawn for an externally-supplied
// wi that might land in the OTHER lobe's hemisphere is wrong. Handled below
// via the material's own closed-form f(wo,wi) = SameHemisphere(wi) ? R/pi :
// T/pi (pbrt-v4 DiffuseTransmissionBxDF::f) computed directly via its
// reflectance_at(rec)/transmittance_at(rec) accessors -- no scatter() call
// needed at all for this material, sidestepping the stochastic-lobe problem
// entirely rather than working around it. Uses reflectance_at/
// transmittance_at (per-point, texture-aware when the material's own rTex/
// tTex are bound), NOT the flat get_reflectance()/get_transmittance() -
// those two ignore a texture-bound reflectance/transmittance entirely
// (barcelona-pavilion's own foliage binds both), which would have silently
// rendered flat colours under --sppm/--bdpt/--mlt even though the plain
// path tracer (camera.h, which calls scatter()/scattering_attenuation()
// polymorphically) got the real per-point texture value.
//
// `mix_material` never reaches this function as such: the owning adapter's
// Intersect() resolves it down to a concrete sub-material before it's ever
// stored in a shading context (see sppm_resolve_material()'s own comment
// for why that has to happen there rather than being handled with a
// closed-form f() the way diffuse_transmission was) -- ctx.mat here is
// always one of the real, concrete material classes.
inline void sppm_bsdf_f(const SPPMShadingContext& ctx, const double wo[3], const double wi[3],
                         const double n[3], double out[3]) {
	out[0] = out[1] = out[2] = 0.0;
	if (!ctx.mat) return;
	double cos_wi = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];

	if (auto dt = dynamic_cast<const diffuse_transmission*>(ctx.mat.get())) {
		hit_record rec = sppm_reconstruct_hit_record(ctx, n);
		color c = (cos_wi > 0.0) ? dt->reflectance_at(rec) : dt->transmittance_at(rec);
		double inv_pi = 1.0 / pi;
		out[0] = c.x() * inv_pi; out[1] = c.y() * inv_pi; out[2] = c.z() * inv_pi;
		return;
	}

	// rough_dielectric needs its own case for the same reason
	// diffuse_transmission does: it's the one other material here whose real
	// f(wo,wi) is nonzero on BOTH sides of the hemisphere (transmission, not
	// just reflection) - unlike diffuse_transmission's R vs T split though,
	// rough_dielectric's glossy-branch scatter() always sets
	// srec.attenuation to a fixed white (1,1,1) regardless of which lobe its
	// own stochastic reflect-vs-refract draw picks (material_pbrt.h's
	// scatter_impl()), so reusing scattering_pdf()'s already-correct,
	// full-sphere GGX value (RoughDielectricBxDF::f() has no hemisphere
	// restriction of its own) needs no extra scatter() call at all - just
	// the same f_cos/cos_theta unnormalization the generic path below uses,
	// with |cos_theta| instead of the generic path's cos_wi (which is
	// negative, not just small, for a transmitted wi - dividing by the
	// signed value would flip the sign of the whole result). Only reachable
	// at all once this material can be classified non-delta - see
	// material::is_delta_bsdf()'s own comment on why a rough_dielectric
	// instance's glossy branch used to never reach this function.
	if (auto rd = dynamic_cast<const rough_dielectric*>(ctx.mat.get())) {
		double cos_theta = std::fabs(cos_wi);
		if (cos_theta <= 0.0) return;
		hit_record rec = sppm_reconstruct_hit_record(ctx, n);
		ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));
		ray fake_scattered(ctx.p, vec3(wi[0], wi[1], wi[2]));
		double f_cos = rd->scattering_pdf(fake_in, rec, fake_scattered);
		if (f_cos <= 0.0) return;
		double f = f_cos / cos_theta;
		out[0] = out[1] = out[2] = f;
		return;
	}

	// lambertian fast path: f(wo,wi) = albedo/pi is a closed form (cosine_pdf
	// cancels exactly against scattering_pdf()'s own cos_theta/pi - see this
	// function's return statement below for the general form this collapses
	// from), so skip material::scatter() entirely rather than pay for its
	// srec.pdf_ptr = make_shared<cosine_pdf>(...) heap allocation just to
	// read attenuation off the SAME texture get_texture() already exposes
	// directly. This function is called from BDPT's O(maxDepth^2)-per-sample
	// (s,t) connection loop and MLT's per-mutation path evaluation (both via
	// BDPTVertex::f() -> this function), so the allocation this avoids would
	// otherwise happen millions of times over an MLT render. attenuation is
	// deterministic and direction-independent for lambertian (this file's
	// own header comment on v1's original scope already established this is
	// exact, not an approximation) - safe to skip scatter()'s random
	// direction draw entirely, not just its allocation.
	if (auto lam = dynamic_cast<const lambertian*>(ctx.mat.get())) {
		if (cos_wi <= 0.0) return;
		color c = lam->get_texture()->value(ctx.u, ctx.v, ctx.p);
		double inv_pi = 1.0 / pi;
		out[0] = c.x() * inv_pi; out[1] = c.y() * inv_pi; out[2] = c.z() * inv_pi;
		return;
	}

	if (cos_wi <= 0.0) return;

	hit_record rec = sppm_reconstruct_hit_record(ctx, n);
	ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));
	ray fake_scattered(ctx.p, vec3(wi[0], wi[1], wi[2]));

	double f_cos = ctx.mat->scattering_pdf(fake_in, rec, fake_scattered);
	if (f_cos <= 0.0) return;

	scatter_record srec;
	if (!ctx.mat->scatter(fake_in, rec, srec)) return;

	out[0] = srec.attenuation.x() * f_cos / cos_wi;
	out[1] = srec.attenuation.y() * f_cos / cos_wi;
	out[2] = srec.attenuation.z() * f_cos / cos_wi;
}

// Importance-samples a new direction at a captured shading context, shaped
// exactly as sppm.h's SPPMCameraPass/SPPMPhotonPass (and bdpt.h's
// BDPTRandomWalk) expect: all of them unconditionally compute
// `beta *= f_val*cosI/pdf` with no separate delta code path, so for delta
// materials f_val/pdf must be engineered so that product collapses to
// exactly `attenuation` (the physically-correct delta-BSDF limit): pdf=1,
// f_val=attenuation/cosI. Getting the cosI division direction backwards
// here would silently scale energy by cosI^2 instead of leaving it
// untouched -- this is the one formula in the whole bridge most worth a
// dedicated regression test (see tests/unit/sppm_adapter_bsdf_tests.cpp).
//
// NOTE: sppm.h/bdpt.h's BSDFSampleF signature takes external u1,u2
// (implying caller-controlled/reproducible randomness, matching pbrt-v4's
// own convention), but material::scatter() always draws its own randomness
// internally via the global random_double() -- there is no way to thread
// caller-supplied random numbers through the existing material interface
// without changing it. u1,u2 are therefore unused here. Functionally this
// is fine for SPPM (doesn't need deterministic replay to converge
// correctly) and for BDPT (each BDPTLi() call is already a single,
// self-contained stochastic estimate); for MLT specifically this means the
// BSDF-sampling decisions inside a mutation are NOT part of MLTSampler's
// reproducible primary-sample-space vector the way pbrt-v4's own BxDF
// samplers are -- a real, intentional limitation of building on top of
// this codebase's existing material::scatter() interface rather than a
// pbrt-v4-style BxDF::Sample_f(u) interface; worth calling out so it isn't
// later mistaken for an oversight (see bdpt_adapter.h's own file comment).
inline bool sppm_bsdf_sample_f(const SPPMShadingContext& ctx, const double wo[3], const double n[3],
                                double /*u1*/, double /*u2*/,
                                double new_dir[3], double f_val[3], double& pdf, bool& is_specular) {
	if (!ctx.mat) return false;
	hit_record rec = sppm_reconstruct_hit_record(ctx, n);
	ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));

	scatter_record srec;
	if (!ctx.mat->scatter(fake_in, rec, srec)) return false;

	if (srec.skip_pdf) {
		vec3 d = unit_vector(srec.skip_pdf_ray.direction());
		new_dir[0] = d.x(); new_dir[1] = d.y(); new_dir[2] = d.z();
		double cosI = std::fabs(d.x()*n[0] + d.y()*n[1] + d.z()*n[2]);
		cosI = std::max(cosI, 1e-6);
		f_val[0] = srec.attenuation.x() / cosI;
		f_val[1] = srec.attenuation.y() / cosI;
		f_val[2] = srec.attenuation.z() / cosI;
		pdf = 1.0;
		is_specular = true;
		return true;
	}

	// Non-specular: importance-sample via the material's own pdf, then
	// evaluate f through the SAME formula BSDFf uses (sppm_bsdf_f) so the
	// two paths can never drift into inconsistent results.
	vec3 d = srec.pdf_ptr->generate();
	double p;
	if (dynamic_cast<const diffuse_transmission*>(ctx.mat.get())) {
		// srec.pdf_ptr is a plain cosine_pdf over whichever single lobe
		// scatter() happened to draw (reflection or transmission) - its
		// value(d) is only that lobe's OWN cos/pi density, missing the
		// lobe-selection-probability factor (pr/(pr+pt) or pt/(pr+pt)) that
		// the true overall sampling density needs. The material's own
		// scattering_pdf() already computes that full, correctly-weighted
		// density (confirmed by reading material_pbrt.h's
		// diffuse_transmission::scattering_pdf() directly) - use it instead
		// of pdf_ptr->value() for this material specifically.
		p = ctx.mat->scattering_pdf(fake_in, rec, ray(ctx.p, d));
	} else {
		p = srec.pdf_ptr->value(d);
	}
	if (p <= 0.0) return false;
	vec3 dn = unit_vector(d);
	double wi[3] = { dn.x(), dn.y(), dn.z() };
	double f[3];
	sppm_bsdf_f(ctx, wo, wi, n, f);
	new_dir[0] = dn.x(); new_dir[1] = dn.y(); new_dir[2] = dn.z();
	f_val[0] = f[0]; f_val[1] = f[1]; f_val[2] = f[2];
	pdf = p;
	is_specular = false;
	return true;
}

// pdf(wo,wi) for a non-specular material at a captured shading context --
// the one Layer-1 query SPPM never needed (it has no MIS between a
// BSDF-sampled continuation and a separately-sampled NEE direction, so it
// never has to ask "what is the density of THIS SPECIFIC wi, which came
// from somewhere else"), but bdpt.h/mlt.h's Scene::BSDFPdf() does: BDPT's
// reverse-pdf bookkeeping (BDPTRandomWalk, BDPTVertex::PDF) needs the
// sampling density of an externally-supplied direction, not just a freshly
// self-sampled one.
//
// Only ever called for non-delta materials (see sppm_is_delta_material's
// own comment -- delta vertices are IsConnectible()==false and bdpt.h never
// calls BSDFPdf on them; is_specular sampling instead forces pdfFwd/pdfRev
// to exactly 0 at the point BDPTRandomWalk records the bounce). Given that,
// this only has to cover the same v1 material set sppm_bsdf_f/
// sppm_bsdf_sample_f already do:
//   - lambertian, normalized_fresnel: both construct srec.pdf_ptr as a
//     plain cosine_pdf(shading normal) in their own scatter() (confirmed by
//     reading material_simple.h/material_pbrt.h directly) -- cosine_pdf's
//     own value() formula is max(0, cos_theta/pi), reusable here in closed
//     form without needing to call scatter() again (whose random draw would
//     be irrelevant to this material's direction-independent pdf shape
//     anyway).
//   - diffuse_transmission: same closed-form scattering_pdf() call
//     sppm_bsdf_sample_f already uses for this material (already the
//     correctly lobe-selection-weighted density, per that function's own
//     comment) -- reused verbatim rather than duplicated.
//   - any other future non-delta, non-diffuse_transmission material is
//     assumed to follow the same cosine_pdf(shading normal) convention
//     lambertian/normalized_fresnel do; if a future material violates that
//     assumption, its BDPT/MLT MIS weights (not its brightness -- BDPTf/
//     BSDFSampleF stay correct regardless) would be off, matching exactly
//     the "unverified beyond v1's material list" caveat this whole bridge
//     already carries for SPPM.
inline double sppm_bsdf_pdf(const SPPMShadingContext& ctx, const double wo[3], const double wi[3],
                             const double n[3]) {
	if (!ctx.mat) return 0.0;
	if (auto dt = dynamic_cast<const diffuse_transmission*>(ctx.mat.get())) {
		hit_record rec = sppm_reconstruct_hit_record(ctx, n);
		ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));
		ray fake_scattered(ctx.p, vec3(wi[0], wi[1], wi[2]));
		return std::max(0.0, ctx.mat->scattering_pdf(fake_in, rec, fake_scattered));
	}
	double cos_wi = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
	return cos_wi > 0.0 ? cos_wi / pi : 0.0;
}
