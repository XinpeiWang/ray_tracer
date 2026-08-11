#pragma once
//==============================================================================================
// sppm_adapter.h -- Bridges src/shared/sppm.h's duck-typed Scene concept to
// this renderer's real hittable/material/camera types.
//
// src/shared/sppm.h is a complete, unit-tested, pbrt-v4-ported SPPM engine
// (camera pass, photon pass, hash-grid nearest-neighbor lookup, progressive
// radius contraction, final reconstruction) that is header-only and
// deliberately engine-agnostic (no virtual functions, duck-typed Scene
// template parameter, GPU-portable). This file is the CPU-only glue that
// makes it usable against real scenes built from this codebase's `hittable`
// hierarchy and `material` classes -- it belongs in src/TheRestOfYourLife/
// (not src/shared/) for that reason, matching where camera.h and
// power_light_sampler.h already live.
//
// Built up in two layers:
//   1. SPPMShadingContext + free functions (sppm_is_delta_material,
//      sppm_bsdf_f, sppm_bsdf_sample_f) -- the BSDF bridge, independently
//      testable without any scene/world (see tests/unit/sppm_adapter_bsdf_tests.cpp).
//   2. SPPMSceneAdapter -- the full duck-typed Scene implementation, adding
//      Intersect/DirectLight/SampleLightLe/PixelToRay on top of layer 1.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "pdf.h"
#include "onb.h"
#include "quad.h"
#include "sphere.h"
#include "camera.h"
#include "power_light_sampler.h"
#include "color.h"
#include "thread_count.h"     // determine_render_thread_count()
#include "../shared/bdpt.h"   // BDPTHit, BDPTLightLeSample
#include "../shared/sppm.h"   // SPPMPixel, SPPMCameraPass, SPPMPhotonPass, SPPMUpdateRadius, SPPMFinalImage

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>

// ===========================================================================
// Layer 1 -- BSDF bridge
// ===========================================================================

// Captured per-hit shading data, indexed by an opaque integer id.
//
// Why this exists: src/shared/sppm.h's Scene concept methods (BSDFf,
// BSDFSampleF, BSDFPdf) take only (bsdf_id, wo, wi, n) -- no position or
// UV -- yet bsdf_id must remain valid from the camera pass (which records a
// pixel's visible point) through to a LATER photon pass (which evaluates
// BSDFf against that same visible point while shooting unrelated photon
// rays). bsdf_id therefore cannot be a material-type enum; it has to be an
// index into an append-only table of captured contexts, one entry per real
// intersection. SPPMSceneAdapter (layer 2) owns that table; this struct is
// its element type.
struct SPPMShadingContext {
	point3 p;
	vec3   normal;
	double u = 0.0, v = 0.0;
	shared_ptr<material> mat;
};

// True for materials whose scatter() sets srec.skip_pdf=true
// unconditionally: metal, dielectric, rough_metal, rough_dielectric,
// conductor, coated_diffuse, thin_dielectric, coated_conductor (confirmed
// via material_simple.h/material_pbrt.h -- these are the only 8 material
// classes with an unconditional `srec.skip_pdf = true`). SPPM's own
// algorithm (sppm.h's SPPMCameraPass/SPPMPhotonPass) branches on this same
// distinction (hit.is_delta_bsdf) and only ever calls BSDFf/BSDFPdf for
// non-delta hits, treating delta hits via BSDFSampleF resampling alone --
// matching pbrt-v4's own SPPM design.
//
// diffuse_transmission/normalized_fresnel are intentionally NOT delta
// (skip_pdf=false) but are also not given full BSDFf support in this v1
// bridge -- see sppm_bsdf_f's comment. mix_material is never passed here
// directly -- SPPMSceneAdapter::Intersect() resolves it down to a concrete
// sub-material first (see sppm_resolve_material() below), so by the time
// anything in this file classifies or evaluates a material, it's always
// one of the real, concrete classes checked below.
inline bool sppm_is_delta_material(const material* m) {
	return dynamic_cast<const metal*>(m) != nullptr ||
	       dynamic_cast<const dielectric*>(m) != nullptr ||
	       dynamic_cast<const rough_metal*>(m) != nullptr ||
	       dynamic_cast<const rough_dielectric*>(m) != nullptr ||
	       dynamic_cast<const conductor*>(m) != nullptr ||
	       dynamic_cast<const coated_diffuse*>(m) != nullptr ||
	       dynamic_cast<const thin_dielectric*>(m) != nullptr ||
	       dynamic_cast<const coated_conductor*>(m) != nullptr;
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
// returning that sub-material's own scatter_record unchanged. SPPM can copy
// that same trick, just earlier: resolve the winning sub-material once, at
// Intersect() time (the one place in SPPM's algorithm that runs before
// is_delta_bsdf is needed), and store THAT concrete material in the
// shading-context table instead of the raw mix_material pointer. Every
// later BSDFf/BSDFSampleF/DirectLight call against this hit's bsdf_id --
// including a photon-pass query long after the camera pass that created it
// -- then sees a single, already-resolved, ordinary material, exactly like
// any other hit. This mirrors how the regular path tracer already treats
// mix_material (one fresh stochastic draw per scatter() call); the only
// difference is WHEN that draw happens.
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
// a freshly-supplied shading normal `n`. sppm.h always threads its own `n`
// parameter through rather than trusting a stored one, so shading math here
// honors that `n` (expected to match ctx.normal in non-degenerate cases;
// ctx.normal is kept mainly for future diagnostic use). front_face=true
// throughout: `n` is already the shading-facing normal by construction (the
// caller -- ultimately SPPMSceneAdapter::Intersect -- is responsible for
// always supplying a properly outward/ray-facing normal, the same
// convention hit_record::set_face_normal already enforces elsewhere in this
// codebase).
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
// (sppm_is_delta_material() == true) -- SPPM's own algorithm never does,
// since it treats those via sppm_bsdf_sample_f() resampling alone.
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
// dedicated test now locks this in (see tests/unit/sppm_adapter_bsdf_tests.cpp).
//
// `diffuse_transmission` genuinely needed a special case: its scatter()
// stochastically commits to reflection (attenuation=R) OR transmission
// (attenuation=T), so reusing whichever was drawn for an externally-supplied
// wi that might land in the OTHER lobe's hemisphere is wrong. Handled below
// via the material's own closed-form f(wo,wi) = SameHemisphere(wi) ? R/pi :
// T/pi (pbrt-v4 DiffuseTransmissionBxDF::f) computed directly from its
// get_reflectance()/get_transmittance() accessors -- no scatter() call
// needed at all for this material, sidestepping the stochastic-lobe problem
// entirely rather than working around it.
//
// `mix_material` never reaches this function as such: SPPMSceneAdapter::
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
		color c = (cos_wi > 0.0) ? dt->get_reflectance() : dt->get_transmittance();
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
// exactly as sppm.h's SPPMCameraPass/SPPMPhotonPass expect: both
// unconditionally compute `beta *= f_val*cosI/pdf` with no separate delta
// code path, so for delta materials f_val/pdf must be engineered so that
// product collapses to exactly `attenuation` (the physically-correct
// delta-BSDF limit): pdf=1, f_val=attenuation/cosI. Getting the cosI
// division direction backwards here would silently scale energy by cosI^2
// instead of leaving it untouched -- this is the one formula in the whole
// bridge most worth a dedicated regression test (see
// tests/unit/sppm_adapter_bsdf_tests.cpp).
//
// NOTE: sppm.h's BSDFSampleF signature takes external u1,u2 (implying
// caller-controlled/reproducible randomness, matching pbrt-v4's own
// convention), but material::scatter() always draws its own randomness
// internally via the global random_double() -- there is no way to thread
// caller-supplied random numbers through the existing material interface
// without changing it. u1,u2 are therefore unused here. Functionally this
// is fine (SPPM doesn't need deterministic replay to converge correctly),
// but is an intentional limitation worth calling out so it isn't later
// mistaken for an oversight.
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


// ===========================================================================
// Layer 2 -- SPPMSceneAdapter: full duck-typed Scene implementation
// ===========================================================================
//
// Implements the subset of src/shared/sppm.h's documented Scene concept
// that its own algorithm actually calls (traced through
// SPPMCameraPass/SPPMPhotonPass/SPPMRender): Intersect, RandFloat,
// BSDFSampleF, DirectLight, SampleLightLe, BSDFf, PixelToRay. The remaining
// methods in sppm.h's header-comment concept (SampleLight, LightPMF,
// LightPDFLe, CameraPDFWe, SceneBoundingSphere, InfiniteLightLe,
// InfiniteLightDensity, Unoccluded, BSDFPdf) exist only for parity with the
// bdpt.h/mlt.h superset concept those files share sppm.h's structs with --
// not needed here.
class SPPMSceneAdapter {
  public:
	// world:      flat scene geometry (as returned directly by a
	//             scene_registry build_world() closure, e.g.
	//             build_cornell_rough_glass() -- NOT the power_light_list
	//             "lights" object, and not BVH-wrapped; hittable_list::hit()
	//             is a fine linear scan for the Cornell-scale scenes this
	//             first targets).
	// nee_lights: NEE importance-sampling geometry (e.g.
	//             build_cornell_box_lights()'s output, or a power_light_list
	//             built from it) -- used only to bias shadow-ray direction
	//             sampling in DirectLight(), exactly like camera.h's own
	//             ray_color() does. Note this list's objects may carry a
	//             null material (a deliberate RTOW idiom: non-emissive
	//             geometry, like a glass sphere, added purely as an
	//             importance-sampling target) -- DirectLight() never trusts
	//             this list's own material; it always re-queries `world`
	//             for the real hit and its real emission.
	// cam:        camera used only for image_width/image_height/get_ray();
	//             caller must have already called cam.initialize().
	//
	// Sizes durable_ctx_ to exactly nPixels here (see that member's own
	// comment for why a fixed-size, never-reallocated table is what makes
	// the lock-free design below possible) -- cam.image_width/image_height
	// are already valid at this point per the precondition above.
	SPPMSceneAdapter(const hittable_list& world, const hittable& nee_lights, const camera& cam)
		: world_(world), nee_lights_(nee_lights), cam_(cam),
		  durable_ctx_(static_cast<size_t>(cam.image_width) * static_cast<size_t>(cam.image_height))
	{
		std::vector<double> weights;
		for (auto& obj : world_.objects) {
			AreaLightSample probe;
			if (!obj->sample_area(0.5, 0.5, probe)) continue;   // not an area-samplable shape
			shared_ptr<material> m = hittable_material(obj);
			auto dl = std::dynamic_pointer_cast<diffuse_light>(m);
			if (!dl) continue;   // real emitters only -- not every area-samplable shape emits
			double area = (probe.pdf_pos > 0.0) ? 1.0 / probe.pdf_pos : 0.0;
			color e = dl->get_texture()->value(probe.u, probe.v, probe.p);
			double lum = 0.2126*e.x() + 0.7152*e.y() + 0.0722*e.z();
			double power = area * lum * pi;   // pbrt-v4 / power_light_sampler.h convention: phi = area*Le*pi
			emitters_.push_back(obj);
			weights.push_back(std::max(power, 1e-9));
		}
		emitter_alias_ = AliasTable(weights.empty() ? std::vector<double>{1.0} : weights);
	}

	// No-op, kept only so existing call sites (sppm_render_with_adapter(),
	// tests) don't need to change: an earlier version of this adapter used
	// a single shared, growable shading-context table that genuinely needed
	// clearing once per iteration (see git history around commit b8d355d's
	// successors for that design). The lock-free split below (durable_ctx_
	// / transient_ctx_) needs no such reset -- see each member's own
	// comment for why.
	void BeginIteration() const {}

	// Intersect/BSDFSampleF/BSDFf/PromoteToVisiblePoint are called
	// concurrently by camera-pass and photon-pass worker threads (see
	// sppm_camera_pass_with_sky() and sppm_photon_pass_mt() below), all
	// sharing this one adapter instance -- but access NO mutex-protected
	// shared state at all. Every Intersect() call captures its shading
	// context into transient_ctx_, a per-thread (static thread_local) single
	// slot: within one hit's processing (Intersect -> immediately either
	// BSDFSampleF for a delta bounce, or PromoteToVisiblePoint+DirectLight
	// for a visible point), exactly one transient context is ever "live" on
	// a given thread at a time, always consumed by the SAME thread that
	// created it before that thread's next Intersect() call overwrites the
	// slot -- so no cross-thread visibility is ever needed for it, and no
	// lock is either. A visible point that needs to survive until a LATER,
	// possibly different-thread photon-pass query is explicitly copied out
	// into durable_ctx_[pixel_idx] by PromoteToVisiblePoint(), a fixed-size
	// table indexed directly by pixel -- see that member's own comment for
	// why concurrent access to it is also lock-free.
	bool Intersect(const double org[3], const double dir[3], double t_max,
	               BDPTHit<double>& hit) const {
		ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]));
		hit_record rec;
		if (!world_.hit(r, interval(0.001, t_max), rec)) return false;

		// Emission is read from the RAW (possibly mix_material) hit before
		// resolution below: mix_material::emitted() correctly blends both
		// sub-materials' emission deterministically ((1-w)*ea + w*eb, no
		// randomness), which is the right way to handle emission from a mix
		// -- resolving to one stochastically-chosen sub-material first would
		// needlessly add variance to something that already has an exact
		// closed form.
		color Le = rec.mat ? rec.mat->emitted(r, rec, rec.u, rec.v, rec.p) : color(0, 0, 0);

		// Resolve mix_material down to a concrete sub-material before it's
		// captured in the shading-context table -- see sppm_resolve_material()'s
		// comment for why this has to happen here, once, rather than being
		// deferred to whatever later call happens to look at ctx.mat first.
		shared_ptr<material> resolved_mat = rec.mat
			? sppm_resolve_material(rec.mat, rec.u, rec.v, rec.p)
			: rec.mat;

		// Thread-local, no lock: see this method's own leading comment.
		transient_ctx_ = SPPMShadingContext{ rec.p, rec.normal, rec.u, rec.v, resolved_mat };

		for (int c = 0; c < 3; ++c) hit.p[c] = rec.p[c];
		for (int c = 0; c < 3; ++c) hit.geo_n[c] = rec.normal[c];       // no separate geometric normal in this codebase
		for (int c = 0; c < 3; ++c) hit.shading_n[c] = rec.normal[c];
		vec3 wo = unit_vector(-r.direction());
		hit.wo[0] = wo.x(); hit.wo[1] = wo.y(); hit.wo[2] = wo.z();
		hit.uv[0] = rec.u; hit.uv[1] = rec.v;
		hit.area_Le[0] = Le.x(); hit.area_Le[1] = Le.y(); hit.area_Le[2] = Le.z();
		hit.t_hit = rec.t;
		hit.is_medium_boundary = false;   // constant_medium scenes out of scope for v1
		hit.is_delta_bsdf = resolved_mat ? sppm_is_delta_material(resolved_mat.get()) : false;
		hit.bsdf_id = kTransientId;
		hit.light_id = -1;   // unused by SPPM's own algorithm
		return true;
	}

	// Copies the current thread's transient shading context (the one just
	// captured by this thread's own Intersect() call, still live -- see
	// Intersect()'s comment) into durable_ctx_[pixel_idx], returning a
	// durable id (simply pixel_idx itself) that stays valid for the rest of
	// this iteration, including later photon-pass queries from ANY thread.
	// Called by sppm_camera_pass_with_sky() exactly when a hit turns out to
	// be a non-delta visible point -- not part of sppm.h's duck-typed Scene
	// concept (its own SPPMCameraPass has no equivalent hook), which is why
	// this integration needed its own adapter-specific camera-pass loop in
	// the first place (see that function's own doc comment).
	//
	// Lock-free: each pixel_idx is written by exactly one thread (the one
	// currently owning that pixel in the camera pass's row-steal loop) and
	// only ever read during the LATER photon pass, which starts only after
	// every camera-pass worker thread has joined -- a full barrier (see
	// sppm_camera_pass_with_sky()'s thread::join() loop) means every
	// promotion is visible to every photon-pass thread with no race and no
	// lock needed on either side.
	int PromoteToVisiblePoint(int pixel_idx) const {
		durable_ctx_[pixel_idx] = transient_ctx_;
		return pixel_idx;
	}

	double RandFloat() const { return random_double(); }

	bool BSDFSampleF(int id, const double wo[3], const double n[3],
	                  double u1, double u2,
	                  double new_dir[3], double f_val[3], double& pdf, bool& is_specular) const {
		const SPPMShadingContext& ctx = (id == kTransientId) ? transient_ctx_ : durable_ctx_[id];
		return sppm_bsdf_sample_f(ctx, wo, n, u1, u2, new_dir, f_val, pdf, is_specular);
	}

	void BSDFf(int id, const double wo[3], const double wi[3], const double n[3], double out[3]) const {
		const SPPMShadingContext& ctx = (id == kTransientId) ? transient_ctx_ : durable_ctx_[id];
		sppm_bsdf_f(ctx, wo, wi, n, out);
	}

	// Near-identical to camera.h's own ray_color() NEE strategy A-1, minus
	// MIS weighting: SPPM's camera pass calls DirectLight() exactly once at
	// the recorded visible point with no second, BSDF-sampled continuation
	// in the same estimate, so there is no double-counting risk requiring a
	// balance/power heuristic weight the way camera.h's own two-strategy
	// loop needs one.
	// Accumulates BOTH area-light NEE and punctual (delta) light
	// contributions into Ld -- structured as two independent blocks (each
	// with its own early-outs) rather than one early-return chain, since a
	// scene can have punctual lights with NO area lights at all
	// (build_lights() == no_lights for scenes 25-28/etc - "Spotlight
	// Cornell", "Point Light Cornell", ... - see scene_registry.h). An
	// earlier version of this function gated everything behind the
	// area-light NEE's own early returns, which silently produced a
	// completely black render for any such scene under --sppm (nee_lights_
	// empty -> pdf_l<=0 -> early return -> punctual block never reached).
	//
	// Sky (infinite) light support: this function's own NEE-toward-sky
	// block (below) handles the case where a DIFFUSE surface receives
	// skylight. The OTHER half - a specular chain escaping the scene
	// entirely and seeing the sky directly - can't be handled here at all
	// (DirectLight is only ever called at a diffuse hit) and is instead
	// handled by sppm_camera_pass_with_sky(), a separate adapter-specific
	// camera-pass loop that mirrors src/shared/sppm.h's own SPPMCameraPass
	// with a sky-on-miss addition - written as a standalone function rather
	// than modifying sppm.h itself, which this integration has deliberately
	// avoided touching throughout.
	void DirectLight(const double p[3], const double n[3], const double wo[3],
	                  int bsdf_id, double Ld[3]) const {
		Ld[0] = Ld[1] = Ld[2] = 0.0;
		point3 P(p[0], p[1], p[2]);

		// --- Area lights (NEE toward nee_lights_) ---
		{
			hittable_pdf light_pdf(nee_lights_, P);
			vec3 raw_dir = light_pdf.generate();
			double dist = raw_dir.length();
			if (dist >= 1e-9) {
				vec3 dir = raw_dir / dist;
				double pdf_l = light_pdf.value(dir);
				double cos_i = dir.x()*n[0] + dir.y()*n[1] + dir.z()*n[2];
				if (pdf_l > 0.0 && cos_i > 0.0) {
					hit_record light_rec;
					ray shadow_ray(P, dir);
					if (world_.hit(shadow_ray, interval(0.001, infinity), light_rec)) {
						color Le = light_rec.mat
							? light_rec.mat->emitted(shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p)
							: color(0, 0, 0);
						if (Le.x() > 0.0 || Le.y() > 0.0 || Le.z() > 0.0) {
							double wi[3] = { dir.x(), dir.y(), dir.z() };
							double f[3];
							BSDFf(bsdf_id, wo, wi, n, f);
							Ld[0] += f[0] * cos_i * Le.x() / pdf_l;
							Ld[1] += f[1] * cos_i * Le.y() / pdf_l;
							Ld[2] += f[2] * cos_i * Le.z() / pdf_l;
						}
					}
				}
			}
		}

		// --- Punctual (delta) lights: point/spot/distant/goniometric/projection ---
		// Mirrors camera.h's own NEE strategy A-3 exactly: pdf=1 for delta
		// lights, no MIS weight needed (no competing BSDF-sampled strategy
		// can ever land exactly on a delta light's direction).
		if (cam_.punct_lights && !cam_.punct_lights->empty()) {
			cam_.punct_lights->for_each_sample(P, [&](const PunctualLiSample& ps) {
				if (ps.Li.x() <= 0.0 && ps.Li.y() <= 0.0 && ps.Li.z() <= 0.0) return;
				double cos_i = ps.wi.x()*n[0] + ps.wi.y()*n[1] + ps.wi.z()*n[2];
				if (cos_i <= 0.0) return;
				double wi[3] = { ps.wi.x(), ps.wi.y(), ps.wi.z() };
				double f[3];
				BSDFf(bsdf_id, wo, wi, n, f);
				if (f[0] <= 0.0 && f[1] <= 0.0 && f[2] <= 0.0) return;
				hit_record shadow_rec;
				interval shadow_t(0.001, ps.t_max == infinity ? infinity : ps.t_max - 0.001);
				if (!world_.hit(ray(P, ps.wi), shadow_t, shadow_rec)) {
					Ld[0] += f[0] * cos_i * ps.Li.x();
					Ld[1] += f[1] * cos_i * ps.Li.y();
					Ld[2] += f[2] * cos_i * ps.Li.z();
				}
			});
		}

		// --- Sky (infinite) light: NEE from this diffuse surface toward the sky ---
		// No MIS weighting needed here either, for a different reason than
		// the delta-light case above: sppm_camera_pass_with_sky()'s own
		// sky-on-miss handling only ever triggers for a PURE SPECULAR chain
		// that escapes the scene (the loop breaks before ever recording a
		// visible point in that case), while DirectLight() is only ever
		// called at a non-specular (diffuse) hit, which is a DIFFERENT,
		// mutually exclusive path outcome within the same camera-pass trace
		// - there is no competing strategy that could double-count the same
		// sky sample from the same trace the way camera.h's own two-strategy
		// (NEE + separate BSDF-sampled continuation) loop needs MIS for.
		if (cam_.sky) {
			SkyLiSample sky_smp = cam_.sky->sample_Le();
			double pdf_sky = sky_smp.pdf;
			if (pdf_sky > 0.0) {
				vec3 dir = sky_smp.direction;
				double cos_i = dir.x()*n[0] + dir.y()*n[1] + dir.z()*n[2];
				if (cos_i > 0.0) {
					hit_record sky_rec;
					ray sky_shadow(P, dir);
					if (!world_.hit(sky_shadow, interval(0.001, infinity), sky_rec)) {
						double wi[3] = { dir.x(), dir.y(), dir.z() };
						double f[3];
						BSDFf(bsdf_id, wo, wi, n, f);
						color Le_sky = cam_.sky->Le(unit_vector(dir));
						Ld[0] += f[0] * cos_i * Le_sky.x() / pdf_sky;
						Ld[1] += f[1] * cos_i * Le_sky.y() / pdf_sky;
						Ld[2] += f[2] * cos_i * Le_sky.z() / pdf_sky;
					}
				}
			}
		}
	}

	// Emits a photon from a power-weighted, uniform-area-sampled point on a
	// real emitter (dynamic_cast-filtered from `world` at construction --
	// see the constructor's comment on why `nee_lights` can't be used here),
	// with a cosine-weighted direction leaving that point into the outward
	// hemisphere (Phase 1's sample_area() + a fresh onb/random_cosine_direction
	// draw, matching cosine_pdf's own construction in pdf.h).
	bool SampleLightLe(double /*u1*/, const double /*u2a*/[2], const double /*u2b*/[2],
	                    BDPTLightLeSample<double>& les) const {
		if (emitters_.empty()) return false;
		int idx = emitter_alias_.sample(random_double());
		const shared_ptr<hittable>& light = emitters_[idx];

		AreaLightSample as;
		if (!light->sample_area(random_double(), random_double(), as)) return false;

		onb uvw(as.n);
		vec3 dir = uvw.transform(random_cosine_direction());
		double cos_theta = dot(dir, as.n);
		if (cos_theta <= 0.0) return false;   // degenerate onb edge case

		shared_ptr<material> m = hittable_material(light);
		auto dl = std::dynamic_pointer_cast<diffuse_light>(m);
		color Le = dl ? dl->get_texture()->value(as.u, as.v, as.p) : color(0, 0, 0);

		les.ray_o[0] = as.p.x(); les.ray_o[1] = as.p.y(); les.ray_o[2] = as.p.z();
		les.ray_d[0] = dir.x();  les.ray_d[1] = dir.y();  les.ray_d[2] = dir.z();
		les.p_on_light[0] = as.p.x(); les.p_on_light[1] = as.p.y(); les.p_on_light[2] = as.p.z();
		les.n_on_light[0] = as.n.x(); les.n_on_light[1] = as.n.y(); les.n_on_light[2] = as.n.z();
		les.L[0] = Le.x(); les.L[1] = Le.y(); les.L[2] = Le.z();
		les.pdf_pos = as.pdf_pos * emitter_alias_.pmf(idx);   // combined light-choice * position pdf
		les.pdf_dir = cos_theta / pi;                         // cosine_pdf's own value() formula
		les.abs_cos_theta = cos_theta;
		les.is_on_surface = true;
		les.is_infinite = false;
		les.is_delta_dir = false;
		les.light_id = idx;
		return true;
	}

	// px,py in [0,1) -- pixel-center-normalized, matching sppm.h's own
	// SPPMCameraPass call convention ((ix+0.5)/width). cam_ must already
	// have had initialize() called by the owner before any PixelToRay call.
	bool PixelToRay(double px, double py, double cam_p[3], double ray_d[3], double cam_n[3]) const {
		int i = static_cast<int>(px * cam_.image_width);
		int j = static_cast<int>(py * cam_.image_height);
		if (i < 0) i = 0;
		if (i >= cam_.image_width) i = cam_.image_width - 1;
		if (j < 0) j = 0;
		if (j >= cam_.image_height) j = cam_.image_height - 1;

		ray r = cam_.get_ray(i, j, 0, 0, vec3(0.0, 0.0, 0.0));
		cam_p[0] = r.origin().x(); cam_p[1] = r.origin().y(); cam_p[2] = r.origin().z();
		vec3 d = unit_vector(r.direction());
		ray_d[0] = d.x(); ray_d[1] = d.y(); ray_d[2] = d.z();
		// cam_n is accepted by sppm.h's PixelToRay signature but never
		// actually read by SPPMCameraPass's own code (confirmed by tracing
		// its call graph) -- filled with a reasonable "camera forward"
		// value for interface completeness/future reuse, not correctness.
		vec3 fwd = unit_vector(cam_.lookat - cam_.lookfrom);
		cam_n[0] = fwd.x(); cam_n[1] = fwd.y(); cam_n[2] = fwd.z();
		return true;
	}

	// Not part of sppm.h's duck-typed Scene concept (its own SPPMCameraPass
	// has no infinite-light-on-miss hook to call these through) -- used
	// directly by sppm_camera_pass_with_sky() below instead, which is an
	// adapter-specific camera-pass loop, not sppm.h's generic one. See that
	// function's doc comment for the full explanation.
	bool HasSky() const { return static_cast<bool>(cam_.sky); }

	void SkyLe(const double dir[3], double out[3]) const {
		color c = cam_.sky->Le(unit_vector(vec3(dir[0], dir[1], dir[2])));
		out[0] = c.x(); out[1] = c.y(); out[2] = c.z();
	}

  private:
	// bsdf_id sentinel: "look in the calling thread's own transient_ctx_
	// slot" rather than durable_ctx_. Any id >= 0 is a durable_ctx_ index
	// (== a pixel index, always non-negative), so -1 is unambiguous.
	static constexpr int kTransientId = -1;

	const hittable_list& world_;
	const hittable&       nee_lights_;
	const camera&          cam_;
	std::vector<shared_ptr<hittable>> emitters_;
	AliasTable emitter_alias_;

	// Fixed-size (nPixels, allocated once in the constructor, never
	// resized) table of visible-point shading contexts, indexed directly
	// by pixel index. Every write happens through PromoteToVisiblePoint(),
	// called by exactly one thread per pixel (the camera pass's row-steal
	// owner); every read happens during the LATER photon pass, strictly
	// after all camera-pass threads have joined. No concurrent read+write
	// or write+write to the same index is ever possible, so this needs no
	// mutex despite being shared across every worker thread.
	mutable std::vector<SPPMShadingContext> durable_ctx_;

	// Per-thread scratch slot for a single, transient shading context --
	// the result of whichever thread's own Intersect() call is currently
	// "in flight" and not yet consumed. static thread_local (a C++17
	// inline variable, required since this header has no corresponding
	// .cpp to hold an out-of-line definition) rather than a genuine
	// per-instance member: in practice exactly one SPPMSceneAdapter is
	// alive per render, and even if more than one existed, each thread
	// only ever has one hit "in flight" at a time regardless of which
	// adapter it belongs to, so sharing this slot across instances on the
	// same thread is harmless.
	static inline thread_local SPPMShadingContext transient_ctx_{};

	// Only quad/sphere currently expose get_material() (both predate this
	// file); this is the one place that needs to know that concretely,
	// since `hittable` itself has no material accessor.
	static shared_ptr<material> hittable_material(const shared_ptr<hittable>& h) {
		if (auto q = std::dynamic_pointer_cast<quad>(h)) return q->get_material();
		if (auto s = std::dynamic_pointer_cast<sphere>(h)) return s->get_material();
		return nullptr;
	}
};


// ===========================================================================
// sppm_render_with_adapter -- shared SPPM iteration loop
// ===========================================================================
// sppm_camera_pass_with_sky -- adapter-specific camera pass
// ===========================================================================
// A direct copy of src/shared/sppm.h's own SPPMCameraPass() logic, with one
// addition: when a specular chain's ray escapes the scene entirely (a
// miss), and the adapter has a sky light configured, accumulate its Le
// before giving up on that pixel for this iteration. sppm.h's own
// SPPMCameraPass has no infinite-light-on-miss hook at all (confirmed by
// reading its full source) - rather than modify that reference-quality
// generic file to add one (this integration has deliberately avoided
// touching sppm.h throughout), this is a separate, adapter-specific
// function that takes SPPMSceneAdapter directly instead of a generic
// template Scene parameter, so it can call SPPMSceneAdapter::HasSky()/
// SkyLe() - methods that exist only on the adapter, not as part of the
// duck-typed Scene concept sppm.h's own functions are written against.
//
// Without this, any scene relying solely on build_sky() for illumination
// (e.g. scene 24 HdriSky, 32/33's alt-camera showcases, 35 PortalInfiniteLight
// - all use no_lights for area lights and no punctual lights either) would
// render pure black under --sppm, the same class of bug punctual-light
// support (SPPMSceneAdapter::DirectLight()) fixed for delta lights.
//
// Multithreaded over image rows (atomic row-steal, mirroring camera.h's own
// render() worker pattern) -- safe because each worker only ever writes to
// pixels[idx] for the one pixel it's currently processing (no cross-pixel
// contention), and the shading-context state shared across pixels/threads
// (SPPMSceneAdapter's durable_ctx_/transient_ctx_) needs no lock either --
// see Intersect()/PromoteToVisiblePoint()'s own comments for why.
// nthreads is passed in rather than rediscovered here because
// determine_render_thread_count()'s auto-detect path samples system idle
// time over ~200ms -- calling it once per pass per iteration (sppm_render_
// with_adapter() calls this function once per iteration, nIterations times)
// would burn seconds to minutes of pure detection overhead across a real
// multi-iteration render; the caller resolves it once for the whole render.
inline void sppm_camera_pass_with_sky(std::vector<SPPMPixel<double>>& pixels,
                                       int width, int height, int maxDepth,
                                       const SPPMSceneAdapter& scene,
                                       unsigned int nthreads) {
	std::atomic<int> next_row(0);

	auto worker = [&]() {
		while (true) {
			int iy = next_row.fetch_add(1);
			if (iy >= height) break;

			for (int ix = 0; ix < width; ++ix) {
				int idx = iy * width + ix;
				SPPMPixel<double>& pixel = pixels[idx];
				pixel.px = ix;
				pixel.py = iy;
				pixel.vp_valid = false;

				double px = (double(ix) + 0.5) / double(width);
				double py = (double(iy) + 0.5) / double(height);

				double cam_p[3], ray_d[3], cam_n[3];
				if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n))
					continue;

				double org[3] = { cam_p[0], cam_p[1], cam_p[2] };
				double dir[3] = { ray_d[0], ray_d[1], ray_d[2] };
				double beta[3] = { 1.0, 1.0, 1.0 };

				for (int depth = 0; depth < maxDepth; ++depth) {
					BDPTHit<double> hit{};
					if (!scene.Intersect(org, dir, std::numeric_limits<double>::max(), hit)) {
						if (scene.HasSky()) {
							double le[3];
							scene.SkyLe(dir, le);
							pixel.Ld[0] += beta[0] * le[0];
							pixel.Ld[1] += beta[1] * le[1];
							pixel.Ld[2] += beta[2] * le[2];
						}
						break;
					}

					pixel.Ld[0] += beta[0] * hit.area_Le[0];
					pixel.Ld[1] += beta[1] * hit.area_Le[1];
					pixel.Ld[2] += beta[2] * hit.area_Le[2];

					if (hit.is_delta_bsdf) {
						double new_dir[3], f_val[3], pdf;
						bool is_spec;
						double u1 = scene.RandFloat(), u2 = scene.RandFloat();
						if (!scene.BSDFSampleF(hit.bsdf_id, hit.wo, hit.shading_n,
						                       u1, u2, new_dir, f_val, pdf, is_spec))
							break;
						if (pdf <= 0.0) break;
						double cosI = std::fabs(new_dir[0]*hit.shading_n[0] +
						                        new_dir[1]*hit.shading_n[1] +
						                        new_dir[2]*hit.shading_n[2]);
						beta[0] *= f_val[0] * cosI / pdf;
						beta[1] *= f_val[1] * cosI / pdf;
						beta[2] *= f_val[2] * cosI / pdf;
						org[0] = hit.p[0]; org[1] = hit.p[1]; org[2] = hit.p[2];
						dir[0] = new_dir[0]; dir[1] = new_dir[1]; dir[2] = new_dir[2];
					} else {
						// Copy this hit's transient shading context into the
						// durable, pixel-indexed table before it's used any
						// further -- both so it survives to the (possibly
						// different-thread, definitely-later) photon pass,
						// and so this pixel's own DirectLight() call below
						// reads from the same place a photon-pass BSDFf
						// query against vp_bsdf_id later will. See
						// PromoteToVisiblePoint()'s own comment.
						int durable_id = scene.PromoteToVisiblePoint(idx);

						pixel.vp_valid = true;
						pixel.vp_p[0] = hit.p[0]; pixel.vp_p[1] = hit.p[1]; pixel.vp_p[2] = hit.p[2];
						pixel.vp_wo[0] = hit.wo[0]; pixel.vp_wo[1] = hit.wo[1]; pixel.vp_wo[2] = hit.wo[2];
						pixel.vp_n[0] = hit.shading_n[0]; pixel.vp_n[1] = hit.shading_n[1]; pixel.vp_n[2] = hit.shading_n[2];
						pixel.vp_beta[0] = beta[0]; pixel.vp_beta[1] = beta[1]; pixel.vp_beta[2] = beta[2];
						pixel.vp_bsdf_id = durable_id;

						double ld[3];
						scene.DirectLight(hit.p, hit.shading_n, hit.wo, durable_id, ld);
						pixel.Ld[0] += beta[0] * ld[0];
						pixel.Ld[1] += beta[1] * ld[1];
						pixel.Ld[2] += beta[2] * ld[2];
						break;
					}
				}
			}
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();
}

// ===========================================================================
// sppm_photon_pass_mt -- multithreaded photon pass
// ===========================================================================
// A direct copy of src/shared/sppm.h's own SPPMPhotonPass() body (kept
// templated over Scene like the original, since -- unlike the camera pass --
// nothing here needs SPPMSceneAdapter-specific methods), parallelized over
// photon index (atomic photon-steal). This integration has deliberately
// never modified sppm.h itself (see sppm_camera_pass_with_sky()'s own
// comment for why), so this is a separate, adapter-file function rather
// than a multithreaded variant added to the reference header.
//
// Per-photon work is independent EXCEPT for the hash-grid deposit step: two
// photons on two different threads can both land within radius of the same
// pixel's visible point at the same time (that's the whole point of the
// hash grid -- a popular visible point is meant to collect flux from many
// photons), so concurrent unsynchronized `px.Phi[c] += ...; px.m++;` would
// race. pixel_mutexes provides one mutex per pixel so that only genuinely
// concurrent writes to the *same* pixel ever contend; every other field
// read here (vp_valid/vp_p/radius/vp_bsdf_id/vp_wo/vp_n/vp_beta) is set
// once by the camera pass and never mutated during the photon pass, so
// reading them unlocked is safe. scene.Intersect()/BSDFf()/BSDFSampleF()
// need no synchronization of their own either (see SPPMSceneAdapter's
// durable_ctx_/transient_ctx_ comments) and run fully in parallel.
template<typename Scene>
inline void sppm_photon_pass_mt(std::vector<SPPMPixel<double>>& pixels,
                                 const sppm_detail::HashGrid<double>& grid,
                                 int nPhotons, int maxDepth,
                                 const Scene& scene,
                                 std::vector<std::mutex>& pixel_mutexes,
                                 unsigned int nthreads) {
	std::atomic<int> next_photon(0);

	auto worker = [&]() {
		while (true) {
			int ph = next_photon.fetch_add(1);
			if (ph >= nPhotons) break;

			BDPTLightLeSample<double> les{};
			double u1 = scene.RandFloat();
			double u2a[2] = { scene.RandFloat(), scene.RandFloat() };
			double u2b[2] = { scene.RandFloat(), scene.RandFloat() };
			if (!scene.SampleLightLe(u1, u2a, u2b, les)) continue;
			if (les.pdf_pos <= 0.0 || les.pdf_dir <= 0.0) continue;

			double beta[3];
			for (int c = 0; c < 3; ++c)
				beta[c] = les.L[c] * les.abs_cos_theta / (les.pdf_pos * les.pdf_dir);

			double org[3] = { les.ray_o[0], les.ray_o[1], les.ray_o[2] };
			double dir[3] = { les.ray_d[0], les.ray_d[1], les.ray_d[2] };

			for (int depth = 0; depth < maxDepth; ++depth) {
				BDPTHit<double> hit{};
				if (!scene.Intersect(org, dir, std::numeric_limits<double>::max(), hit))
					break;

				if (depth > 0 && !hit.is_delta_bsdf) {
					int h = grid.Bucket(hit.p[0], hit.p[1], hit.p[2]);
					for (int pidx : grid.CellList(h)) {
						SPPMPixel<double>& px = pixels[pidx];
						if (!px.vp_valid) continue;
						double dx = px.vp_p[0] - hit.p[0];
						double dy = px.vp_p[1] - hit.p[1];
						double dz = px.vp_p[2] - hit.p[2];
						if (dx*dx + dy*dy + dz*dz > px.radius * px.radius) continue;

						double wi[3] = { -dir[0], -dir[1], -dir[2] };
						double f[3];
						scene.BSDFf(px.vp_bsdf_id, px.vp_wo, wi, px.vp_n, f);

						std::lock_guard<std::mutex> lg(pixel_mutexes[pidx]);
						for (int c = 0; c < 3; ++c)
							px.Phi[c] += beta[c] * px.vp_beta[c] * f[c];
						px.m++;
					}
				}

				double new_dir[3], f_val[3], pdf;
				bool is_spec;
				double su1 = scene.RandFloat(), su2 = scene.RandFloat();
				if (!scene.BSDFSampleF(hit.bsdf_id, hit.wo, hit.shading_n,
				                       su1, su2, new_dir, f_val, pdf, is_spec))
					break;
				if (pdf <= 0.0) break;

				double cosI = std::fabs(new_dir[0]*hit.shading_n[0] +
				                        new_dir[1]*hit.shading_n[1] +
				                        new_dir[2]*hit.shading_n[2]);
				double beta_new[3];
				for (int c = 0; c < 3; ++c)
					beta_new[c] = beta[c] * f_val[c] * cosI / pdf;

				double betaMax = std::max({beta[0], beta[1], beta[2]});
				double betaNewMax = std::max({beta_new[0], beta_new[1], beta_new[2]});
				double q = 0.0;
				if (betaMax > 0.0)
					q = std::max(0.0, 1.0 - betaNewMax / betaMax);
				if (scene.RandFloat() < q) break;
				double invSurv = 1.0 / (1.0 - q);
				for (int c = 0; c < 3; ++c) beta[c] = beta_new[c] * invSurv;

				org[0] = hit.p[0]; org[1] = hit.p[1]; org[2] = hit.p[2];
				dir[0] = new_dir[0]; dir[1] = new_dir[1]; dir[2] = new_dir[2];
			}
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();
}

// ===========================================================================
// sppm_hash_grid_build_mt -- multithreaded hash-grid build
// ===========================================================================
// A drop-in replacement for calling sppm_detail::HashGrid<double>::Build()
// directly: fills the exact same public fields (hashSize/gridRes/gridMin/
// gridMax/cellSize/grid), via the exact same algorithm (mirrors Build()'s
// body line-for-line), so the result is usable everywhere a HashGrid built
// the normal way would be -- sppm_photon_pass_mt()'s grid parameter is
// concretely typed sppm_detail::HashGrid<double>, not a template, so this
// has to actually produce one rather than some adapter-specific stand-in.
// Written as a free function operating on a HashGrid's public fields from
// outside the class (all of them already public, no encapsulation broken)
// rather than adding a Build_mt() method to sppm.h itself, matching this
// integration's rule throughout of never modifying that reference file.
//
// Only the second half -- inserting each visible point into every grid
// cell it overlaps -- gets parallelized. That's deliberate, verified with
// profiling on scene 11 (60 iterations, 8000 photons): once the camera and
// photon passes were themselves multithreaded, HashGrid::Build() became
// the single largest contributor to render time (roughly 30% of it,
// entirely serial) -- and nearly all of that cost is the insertion loop's
// forward_list::push_front calls (one heap allocation each, for
// potentially several overlapping cells per visible point), not the first
// loop's bounds/max-radius reduction, which is a cheap linear scan over
// simple comparisons and stayed single-threaded here since it wasn't worth
// the added complexity.
//
// Thread-safety: concurrently calling push_front on the SAME
// forward_list<int> from two threads races on that list's head pointer, so
// each hash bucket needs its own lock -- bucket_mutexes provides one per
// bucket. hashSize (and therefore bucket_mutexes' required size) depends
// only on pixels.size(), which is fixed for the whole render, so the
// caller allocates it once, like pixel_mutexes.
inline void sppm_hash_grid_build_mt(sppm_detail::HashGrid<double>& hg,
                                     const std::vector<SPPMPixel<double>>& pixels,
                                     std::vector<std::mutex>& bucket_mutexes,
                                     unsigned int nthreads) {
	using T = double;

	// Phase 1 (single-threaded, cheap): bounds + max radius reduction --
	// identical to HashGrid::Build()'s own first loop.
	T gMin[3] = { std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max() };
	T gMax[3] = { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest() };
	T maxR = T(0);
	for (const auto& px : pixels) {
		if (!px.vp_valid) continue;
		for (int d = 0; d < 3; ++d) {
			gMin[d] = std::min(gMin[d], px.vp_p[d] - px.radius);
			gMax[d] = std::max(gMax[d], px.vp_p[d] + px.radius);
		}
		maxR = std::max(maxR, px.radius);
	}
	for (int d = 0; d < 3; ++d) {
		if (gMin[d] > gMax[d]) { gMin[d] = T(0); gMax[d] = T(1); }
		if (gMax[d] - gMin[d] < T(1e-6)) gMax[d] = gMin[d] + T(1e-6);
		hg.gridMin[d] = gMin[d];
		hg.gridMax[d] = gMax[d];
	}
	if (maxR <= T(0)) maxR = T(1);

	T diag[3] = { hg.gridMax[0]-hg.gridMin[0], hg.gridMax[1]-hg.gridMin[1], hg.gridMax[2]-hg.gridMin[2] };
	T maxDiag = std::max({diag[0], diag[1], diag[2]});
	int baseRes = std::max(1, (int)(maxDiag / maxR));
	for (int d = 0; d < 3; ++d) {
		hg.gridRes[d] = std::max(1, (int)(baseRes * diag[d] / maxDiag));
		hg.cellSize[d] = diag[d] / (T)hg.gridRes[d];
	}

	hg.hashSize = sppm_detail::NextPrime((int)pixels.size());
	hg.grid.assign(hg.hashSize, std::forward_list<int>());

	// Phase 2 (parallel): insert each valid visible point into every grid
	// cell it overlaps, atomic pixel-index-steal across threads, one mutex
	// per destination bucket.
	std::atomic<int> next_pixel(0);
	int nPixels = (int)pixels.size();
	auto worker = [&]() {
		while (true) {
			int i = next_pixel.fetch_add(1);
			if (i >= nPixels) break;
			const auto& px = pixels[i];
			if (!px.vp_valid) continue;
			T r = px.radius;
			int pMin[3], pMax[3];
			hg.ToGrid(px.vp_p[0]-r, px.vp_p[1]-r, px.vp_p[2]-r, pMin);
			hg.ToGrid(px.vp_p[0]+r, px.vp_p[1]+r, px.vp_p[2]+r, pMax);
			for (int z = pMin[2]; z <= pMax[2]; ++z)
			for (int y = pMin[1]; y <= pMax[1]; ++y)
			for (int x = pMin[0]; x <= pMax[0]; ++x) {
				int h = (int)(sppm_detail::HashPoint3i(x,y,z) % (uint32_t)hg.hashSize);
				std::lock_guard<std::mutex> lg(bucket_mutexes[h]);
				hg.grid[h].push_front(i);
			}
		}
	};

	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	for (unsigned int t = 0; t < nthreads; ++t) threads.emplace_back(worker);
	for (auto& th : threads) th.join();
}

// Identical to src/shared/sppm.h's own SPPMRender() body, with three
// additions: (1) scene.BeginIteration() is called once per iteration,
// immediately before that iteration's camera pass -- a no-op today (see its
// own doc comment), kept only so this call site and sppm.h's Scene-concept
// contract don't need to change if a future adapter design needs it again.
// (2) uses sppm_camera_pass_with_sky() above instead of sppm.h's own
// SPPMCameraPass() (sky-light support - see that function's doc comment).
// (3) uses sppm_hash_grid_build_mt()/sppm_photon_pass_mt() instead of
// sppm.h's single-threaded HashGrid::Build()/SPPMPhotonPass() -- the
// camera pass, hash-grid build, and photon pass are all multithreaded now.
// nthreads is resolved once here, not per-pass-per-iteration (see
// sppm_camera_pass_with_sky()'s comment on why), and pixel_mutexes/
// bucket_mutexes are each allocated once and reused across every
// iteration's photon pass / hash-grid build rather than rebuilt each time
// (hashSize, and therefore how many bucket_mutexes are needed, depends
// only on pixel count, which is fixed for the whole render). Kept here
// rather than duplicated at each call site (tests, cpu_interface.cpp) so
// all of these invariants only have to be gotten right in one place.
inline void sppm_render_with_adapter(const SPPMSceneAdapter& scene, int width, int height,
                                      int nIterations, int nPhotons, int maxDepth,
                                      double initialRadius, std::vector<double>& out_rgb) {
	int nPixels = width * height;
	std::vector<SPPMPixel<double>> pixels(nPixels);
	for (int i = 0; i < nPixels; ++i) {
		pixels[i].radius = initialRadius;
		pixels[i].px = i % width;
		pixels[i].py = i / width;
	}

	unsigned int nthreads = determine_render_thread_count();
	std::vector<std::mutex> pixel_mutexes(nPixels);
	std::vector<std::mutex> bucket_mutexes(sppm_detail::NextPrime(nPixels));

	int64_t totalPhotonPaths = 0;
	for (int iter = 0; iter < nIterations; ++iter) {
		scene.BeginIteration();
		sppm_camera_pass_with_sky(pixels, width, height, maxDepth, scene, nthreads);

		sppm_detail::HashGrid<double> hashGrid;
		sppm_hash_grid_build_mt(hashGrid, pixels, bucket_mutexes, nthreads);

		sppm_photon_pass_mt(pixels, hashGrid, nPhotons, maxDepth, scene, pixel_mutexes, nthreads);
		totalPhotonPaths += nPhotons;

		SPPMUpdateRadius(pixels);
	}

	SPPMFinalImage(pixels, nIterations, totalPhotonPaths, out_rgb);
}

// Writes a flat RGB double buffer (as produced by sppm_render_with_adapter's
// out_rgb, or SPPMFinalImage's own out_rgb) to a P3 PPM file, applying the
// same tone mapping / sRGB encoding as the existing path tracer's
// write_color() (color.h) so SPPM output looks consistent with every other
// render this codebase produces.
inline void sppm_write_ppm(const std::string& path, int width, int height,
                            const std::vector<double>& rgb) {
	std::ofstream out(path);
	out << "P3\n" << width << ' ' << height << "\n255\n";
	for (int i = 0; i < width * height; ++i) {
		color c(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
		write_color(out, c);
	}
}
