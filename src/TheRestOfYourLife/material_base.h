#pragma once
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.h"
#include "pdf.h"
#include "texture.h"
#include "../shared/fresnel.h"
#include "../shared/conductor_data.h"
#include "../shared/microfacet.h"
#include "../shared/bxdfs.h"
#include "../shared/shading_frame.h"
#include "../shared/material_context.h"


// Defined in material_pbrt.h. Forward-declared here so `material::
// as_subsurface()` can return a pointer to it without material_base.h
// needing to know anything else about it.
class subsurface;


class scatter_record {
  public:
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
    bool skip_pdf = false;
    ray skip_pdf_ray;
    // Defaulted so materials that never cross a refractive boundary (metal,
    // rough_metal, conductor, coated_diffuse, thin_dielectric,
    // coated_conductor) don't have to remember to set these -- previously
    // left uninitialized by every skip_pdf material except dielectric,
    // which meant camera.h's etaScale/Russian-roulette logic
    // (`eta_scale *= srec.eta * srec.eta` gated by `srec.is_transmission`)
    // read garbage stack memory on every bounce off one of those materials.
    // For rough_dielectric specifically this reliably corrupted eta_scale
    // right after the entry bounce, causing the very next (exit) bounce's
    // RR check to compute a near-1.0 kill probability from garbage and
    // terminate the path before it could reach any wall or light --
    // rendering the sphere's refraction-dominated interior almost solid
    // black regardless of sample count (confirmed via debug instrumentation:
    // reflect-branch/silhouette rays, which only ever take one skip_pdf
    // bounce before hitting a non-specular wall, never exercised the
    // corrupted path and rendered correctly).
    double eta = 1.0;             // IOR ratio (1.0 unless refraction; pbrt-v4 bs->eta)
    bool is_transmission = false; // true when a refraction occurred (drives etaScale in integrator)
    // True only for interface_material's scatter() (material_simple.h) -
    // pbrt-v4's real "no BSDF" interface material (Material "none"/"" -
    // bounds a participating medium with no surface response of its own).
    // Distinct from skip_pdf (which this ALSO sets, since there's no pdf to
    // importance-sample against): skip_pdf alone means "a real specular
    // event happened here, don't NEE, but count it and update MIS state as
    // a specular bounce" - that's wrong for a true pass-through, where
    // nothing actually happened. camera.h's ray_color()/ray_color_spectral()
    // check this FIRST, before the generic skip_pdf branch, to preserve
    // specular_bounce/prev_bsdf_pdf/prev_surface_p exactly as they were and
    // skip the bounce-budget/Russian-Roulette accounting too - mirroring
    // pbrt-v4's own SurfaceInteraction::SkipIntersection, and the same
    // "is_medium_boundary" concept src/shared/bdpt.h's BDPTHit and the
    // templated Scene-concept integrators (path_integrator.h, light_path.h)
    // already use, now finally reachable from real Material-based geometry
    // via material::is_medium_boundary() below.
    bool is_medium_boundary = false;
};


// Common interface for a material whose IOR is wavelength-dependent (real
// dispersion, reachable only under --spectral) - implemented by `dielectric`
// (material_simple.h) and `rough_dielectric` (material_pbrt.h), each via
// their own make_dispersive() factory. `material::as_dispersive()` below is
// the ONE hook camera.h's ray_color_spectral() dispatches through for every
// dispersive material, regardless of how many concrete types exist.
//
// This replaces what used to be a separate virtual hook PER dispersive
// concrete type (as_dispersive_dielectric()/as_dispersive_rough_dielectric()) -
// that shape didn't scale: every new dispersive material needed its own
// hook, its own mix_material forwarding override, and its own arm in 3
// separate camera.h dispatch points, with nothing stopping a future
// material from getting the hook and override right while still being
// forgotten at one of those 3 call sites - silently reintroducing the exact
// "loses dispersion through NEE" bug rough_dielectric's own dispersion
// needed fixing once already, just for the next material instead.
//
// A material reachable through as_dispersive() IS, unconditionally,
// dispersive - there is no separate is_dispersive() to re-check afterward;
// the pointer itself is the answer, same "non-null means yes" contract
// as_subsurface() already uses above.
class dispersive_material {
  public:
    virtual ~dispersive_material() = default;

    // Spectral-aware scatter: computes eta from the path's hero wavelength
    // instead of a flat IOR, and reports that resolved eta back through
    // eta_out so a later scattering_pdf_dispersive() call for the SAME
    // bounce can reuse it instead of re-deriving eta from lambda_nm all
    // over again (up to 4 more times, once per NEE strategy/Strategy B -
    // camera.h calls this once per bounce, scattering_pdf_dispersive()
    // potentially several times). do_regularize is accepted for parity with
    // material::scatter() even though a material with no roughness of its
    // own (smooth dielectric) simply ignores it.
    virtual bool scatter_dispersive(const ray& r_in, const hit_record& rec,
                                     scatter_record& srec, float lambda_nm,
                                     bool do_regularize, double& eta_out) const = 0;

    // Spectral-aware scattering_pdf, for a dispersive material whose glossy
    // path reaches real NEE/MIS (rough_dielectric) - takes the SAME eta
    // scatter_dispersive() already resolved for this bounce (see eta_out
    // above), not lambda_nm again, so the two calls can't independently
    // drift and NEE never pays for a second Cauchy-formula evaluation.
    // Default 0 matches material::scattering_pdf()'s own default and is
    // correct for any material whose scatter_dispersive() only ever leaves
    // srec.skip_pdf=true (smooth dielectric: always specular, this is never
    // actually called).
    virtual double scattering_pdf_dispersive(const ray& r_in, const hit_record& rec,
                                              const ray& scattered, double eta) const {
        (void)r_in; (void)rec; (void)scattered; (void)eta;
        return 0.0;
    }
};


class material {
  public:
    virtual ~material() = default;

    virtual color emitted(
        const ray& r_in, const hit_record& rec, double u, double v, const point3& p
    ) const {
        return color(0,0,0);
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                         bool do_regularize = false) const {
        return false;
    }

    virtual double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const {
        return 0;
    }

    // Whether THIS material instance's scatter() sets srec.skip_pdf=true -
    // i.e. behaves as a delta/specular BSDF that SPPM/BDPT/MLT
    // (src/TheRestOfYourLife/bsdf_bridge.h's sppm_is_delta_material(), the
    // only caller) should treat via BSDF-sampling resampling alone, with no
    // NEE and no photon deposit at this vertex. Default false is correct
    // for every material that never sets skip_pdf (lambertian,
    // normalized_fresnel, diffuse_transmission, ...).
    //
    // Materials with a roughness parameter (rough_metal, conductor,
    // rough_dielectric, coated_diffuse, coated_conductor) branch on
    // TrowbridgeReitz::EffectivelySmooth() at RENDER time inside scatter()
    // itself - near-zero roughness takes the delta fast path, anything
    // above the threshold takes a real glossy NEE/MIS path - so their
    // overrides consult that exact same check rather than returning a fixed
    // answer. Materials with no roughness parameter at all (metal,
    // dielectric, thin_dielectric) are unconditionally delta and override
    // this to a fixed `true`.
    virtual bool is_delta_bsdf() const { return false; }

    // Per-hit-aware variant of is_delta_bsdf() above, for the rare material
    // whose real answer varies by SURFACE POINT rather than being a fixed
    // property of the instance - e.g. rough_dielectric with a texture-bound
    // roughness (material_pbrt.h), where a scratch-free/near-mirror texel
    // is genuinely delta at that point even though other texels on the same
    // instance are not, and the no-arg is_delta_bsdf() (with no hit_record
    // to sample the texture from) has no way to know that. Default
    // implementation just forwards to the no-arg version, so every existing
    // override (whose answer never varies by hit point) needs no change.
    // SPPM's and BDPT/MLT's own Intersect() (sppm_adapter.h, bdpt_adapter.h)
    // already resolve a real hit_record before classifying a hit, so they
    // call this overload instead of the bare one for a materially more
    // precise answer - see bsdf_bridge.h's sppm_is_delta_material().
    virtual bool is_delta_bsdf(const hit_record& rec) const { (void)rec; return is_delta_bsdf(); }

    // True only for interface_material (material_simple.h) - pbrt-v4's real
    // "no BSDF" interface material. A THIRD classification alongside
    // is_delta_bsdf()'s true/false: not a real delta surface (no NEE, but a
    // genuine specular event that should count as a bounce and update MIS
    // state) and not a real non-delta surface (real NEE/photon deposit) -
    // this hit should be skipped entirely, as if it never happened. Mirrors
    // is_delta_bsdf()'s own shape exactly; see bsdf_bridge.h's
    // sppm_is_medium_boundary() for the same null-safe bridge pattern.
    virtual bool is_medium_boundary() const { return false; }

    // The attenuation color to pair with scattering_pdf(..., scattered) for
    // THAT specific direction - defaults to srec_attenuation (the value
    // scatter() already stored in scatter_record for the direction IT
    // sampled), which is exactly right for every material whose color never
    // varies by direction (Lambertian, metal, coated_diffuse, ...: the
    // overwhelming majority). camera.h's NEE strategies (light/sky/punctual)
    // evaluate scattering_pdf() at a shadow ray toward a light - a DIFFERENT
    // direction than whatever scatter() itself sampled - and pair it with
    // this call instead of blindly reusing scatter_record::attenuation, so a
    // material like diffuse_transmission (whose color genuinely differs
    // between the reflection and transmission hemispheres: R vs T) can
    // return the color matching the hemisphere actually being evaluated
    // rather than whichever one scatter()'s own stochastic pick landed on.
    // See diffuse_transmission's own override.
    virtual color scattering_attenuation(const hit_record& rec, const ray& scattered,
                                          const color& srec_attenuation) const {
        (void)rec; (void)scattered;
        return srec_attenuation;
    }

    // Whether a SHADOW ray's occlusion test should treat a hit on this
    // material as "nothing there" and keep going, rather than as a blocker.
    // Default false (opaque) is the conservative, correct default for
    // anything that doesn't override it.
    //
    // Mirrors gpu/optix/optix_anyhit_shadow.h's per-type skip list exactly
    // (Dielectric/RoughDielectric/ThinDielectric/DiffuseTransmission/Medium) -
    // GPU has treated transmissive materials this way since that file was
    // written; CPU's shadow-ray tests (shadow_ray.h) did not, because they
    // predate it and were never updated to match. A glass sphere sitting
    // between a shading point and an area light made CPU's simple
    // "did the shadow ray hit ANYTHING" + "is that emissive" test find the
    // glass, see a non-emissive material, and silently drop the sample as
    // occluded - even though light physically passes through glass. See
    // shadow_ray.h's own comment for the discovery (a real, measured
    // CPU/GPU brightness divergence, not a hypothetical).
    virtual bool is_shadow_transmissive(const hit_record& rec) const {
        (void)rec;
        return false;
    }

    // Real attenuation a shadow ray picks up passing through this material,
    // for materials where is_shadow_transmissive() is true. Default 1.0 (no
    // attenuation) is correct for ordinary transmissive surfaces like glass -
    // a shadow ray is a visibility test, not a light-transport simulation,
    // and glass doesn't dim what's behind it enough to matter here. Media
    // (hg_phase_material) override this with real Beer-Lambert/ratio-
    // tracking transmittance instead: unlike glass, a thick fog bank
    // measurably dims a light behind it, and shadow_ray.h's walker used to
    // treat every medium as fully transmissive with no attenuation at all
    // (see hg_phase_material::is_shadow_transmissive()'s own comment for
    // the history - the "unoccluded" simplification was correct as far as
    // it went, but stopped short of actually attenuating).
    // t_max bounds the search to the shadow ray's real target distance (the
    // light) rather than the medium's full geometric extent - a punctual
    // light sitting inside or just past a medium's near boundary should only
    // be attenuated by the medium between the entry point and the light, not
    // by however much further the medium happens to continue past it.
    virtual color shadow_transmittance(const ray& r, double t_max) const {
        (void)r; (void)t_max;
        return color(1, 1, 1);
    }

    // Non-null only for `class subsurface` (material_pbrt.h): a material
    // carrying a real BSSRDF diffusion profile. camera.h's ray_color loop
    // uses this to detect when a specular-transmission scatter event should
    // be redirected into the BSSRDF probe/exit-point search instead of just
    // continuing to trace the refracted ray - see camera.h::
    // sample_bssrdf_exit()'s own comment for why (a material's scatter()
    // has no access to the scene geometry that search needs).
    //
    // Takes `rec` (unused by every override except mix_material's) so a
    // wrapper material whose choice of sub-material varies by hit point
    // (mix_material) can report the SAME sub-material the immediately
    // preceding scatter() call on this same rec committed to, rather than
    // being unable to answer the question at all (the immediate motivation:
    // mix_material previously had no override, so mix(subsurface, X)
    // silently dropped BSSRDF regardless of which branch was picked).
    virtual const subsurface* as_subsurface(const hit_record& rec) const {
        (void)rec;
        return nullptr;
    }

    // Non-null only for a material built via its own dispersive factory
    // (dielectric::make_dispersive()/rough_dielectric::make_dispersive()):
    // one whose IOR is wavelength-dependent. camera.h's ray_color_spectral()
    // uses this - instead of a raw dynamic_cast<const T*>(rec.mat.get()) per
    // concrete type - to decide whether to route a transmission event
    // through dispersive_material::scatter_dispersive() at the path's hero
    // wavelength. See dispersive_material's own comment (above) for why
    // this is a single hook covering every dispersive concrete type rather
    // than one hook per type.
    //
    // Mirrors as_subsurface() above and exists for the identical reason: a
    // wrapper material whose choice of sub-material varies by hit point
    // (mix_material) needs to report the SAME sub-material its own
    // scatter() call on this rec just committed to, or dispersion silently
    // vanishes the moment a dispersive material is mixed with anything -
    // the dynamic_cast approach would see the outer mix_material, not the
    // material it stochastically picked, and fall back to flat-IOR
    // refraction with no error or warning.
    virtual const dispersive_material* as_dispersive(const hit_record& rec) const {
        (void)rec;
        return nullptr;
    }
};

