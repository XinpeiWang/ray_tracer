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
};

