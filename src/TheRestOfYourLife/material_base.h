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
};

