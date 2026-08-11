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
#include "../shared/bdpt.h"   // BDPTHit, BDPTLightLeSample
#include "../shared/sppm.h"   // SPPMPixel, SPPMCameraPass, SPPMPhotonPass, SPPMUpdateRadius, SPPMFinalImage

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

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
// diffuse_transmission/normalized_fresnel/mix_material are intentionally
// NOT delta (skip_pdf=false) but are also not given full BSDFf support in
// this v1 bridge -- see sppm_bsdf_f's comment.
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
// `mix_material` remains genuinely deferred: it can stochastically delegate
// to two COMPLETELY DIFFERENT underlying BSDFs (e.g. lambertian blended
// with a delta metal), and a delta sub-material has no finite f() value
// away from its exact reflection direction at all -- there's no single
// closed form to special-case the way diffuse_transmission's two same-shape
// diffuse lobes allowed.
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
	SPPMSceneAdapter(const hittable_list& world, const hittable& nee_lights, const camera& cam)
		: world_(world), nee_lights_(nee_lights), cam_(cam)
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

	// Must be called once at the start of each SPPM iteration, before that
	// iteration's camera pass -- clears the shading-context table. Safe to
	// do exactly once per iteration (not per-call) because sppm.h's own
	// SPPMUpdateRadius() unconditionally invalidates every pixel's visible
	// point at the end of each iteration, so no bsdf_id from a prior
	// iteration is ever looked up again. Not thread-safe (ctx_ is a plain
	// append-only vector) -- multithreading the passes is explicitly
	// deferred to a later phase.
	void BeginIteration() const { ctx_.clear(); }

	bool Intersect(const double org[3], const double dir[3], double t_max,
	               BDPTHit<double>& hit) const {
		ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]));
		hit_record rec;
		if (!world_.hit(r, interval(0.001, t_max), rec)) return false;

		ctx_.push_back(SPPMShadingContext{ rec.p, rec.normal, rec.u, rec.v, rec.mat });
		int id = static_cast<int>(ctx_.size()) - 1;

		for (int c = 0; c < 3; ++c) hit.p[c] = rec.p[c];
		for (int c = 0; c < 3; ++c) hit.geo_n[c] = rec.normal[c];       // no separate geometric normal in this codebase
		for (int c = 0; c < 3; ++c) hit.shading_n[c] = rec.normal[c];
		vec3 wo = unit_vector(-r.direction());
		hit.wo[0] = wo.x(); hit.wo[1] = wo.y(); hit.wo[2] = wo.z();
		hit.uv[0] = rec.u; hit.uv[1] = rec.v;
		color Le = rec.mat ? rec.mat->emitted(r, rec, rec.u, rec.v, rec.p) : color(0, 0, 0);
		hit.area_Le[0] = Le.x(); hit.area_Le[1] = Le.y(); hit.area_Le[2] = Le.z();
		hit.t_hit = rec.t;
		hit.is_medium_boundary = false;   // constant_medium scenes out of scope for v1
		hit.is_delta_bsdf = rec.mat ? sppm_is_delta_material(rec.mat.get()) : false;
		hit.bsdf_id = id;
		hit.light_id = -1;   // unused by SPPM's own algorithm
		return true;
	}

	double RandFloat() const { return random_double(); }

	bool BSDFSampleF(int id, const double wo[3], const double n[3],
	                  double u1, double u2,
	                  double new_dir[3], double f_val[3], double& pdf, bool& is_specular) const {
		return sppm_bsdf_sample_f(ctx_[id], wo, n, u1, u2, new_dir, f_val, pdf, is_specular);
	}

	void BSDFf(int id, const double wo[3], const double wi[3], const double n[3], double out[3]) const {
		sppm_bsdf_f(ctx_[id], wo, wi, n, out);
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
	// Sky (infinite) light support is NOT included here: unlike punctual
	// lights, sky contribution also needs to be added when a specular ray
	// chain in the camera pass ESCAPES the scene entirely (a ray miss), but
	// src/shared/sppm.h's own SPPMCameraPass has no infinite-light-on-miss
	// hook at all (confirmed by reading its full source) - adding that
	// would mean modifying sppm.h itself, which this integration has
	// deliberately avoided touching throughout. Scenes relying solely on
	// build_sky() (e.g. scene 24 HdriSky) will still render black under
	// --sppm; still deferred.
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

  private:
	const hittable_list& world_;
	const hittable&       nee_lights_;
	const camera&          cam_;
	std::vector<shared_ptr<hittable>> emitters_;
	AliasTable emitter_alias_;
	mutable std::vector<SPPMShadingContext> ctx_;

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
// Identical to src/shared/sppm.h's own SPPMRender() body, with one addition:
// scene.BeginIteration() is called once per iteration, immediately before
// that iteration's camera pass. sppm.h's SPPMRender() can't do this itself
// since it's generic/duck-typed and has no idea SPPMSceneAdapter needs its
// shading-context table cleared between iterations (see
// SPPMSceneAdapter::BeginIteration()'s own doc comment for why that's safe
// to do once per iteration rather than per-call). Kept here rather than
// duplicated at each call site (tests, cpu_interface.cpp) so the
// BeginIteration timing invariant only has to be gotten right in one place.
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

	int64_t totalPhotonPaths = 0;
	for (int iter = 0; iter < nIterations; ++iter) {
		scene.BeginIteration();
		SPPMCameraPass(pixels, width, height, maxDepth, scene);

		sppm_detail::HashGrid<double> hashGrid;
		hashGrid.Build(pixels);

		SPPMPhotonPass(pixels, hashGrid, nPhotons, maxDepth, scene);
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
