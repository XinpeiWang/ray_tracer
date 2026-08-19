#ifndef SHADOW_RAY_H
#define SHADOW_RAY_H
//==============================================================================================
// shadow_ray.h -- occlusion test that lets transmissive materials through.
//
// A NEE shadow ray answers one question: is anything BLOCKING the light?
// Every NEE site in this codebase used to answer it with a single
// world.hit() plus "is what it found emissive" - which is right for opaque
// geometry, but wrong for glass: the shadow ray would find the glass
// sphere, see that a dielectric material isn't emissive, and conclude the
// light was blocked. Light passes through glass; that conclusion doesn't.
//
// gpu/optix/optix_anyhit_shadow.h has never made this mistake - its any-hit
// programs call optixIgnoreIntersection() for exactly the material types
// listed in material::is_shadow_transmissive()'s comment, letting the ray
// continue past them. This is CPU's equivalent: instead of one hit test,
// walk forward past every transmissive hit until an opaque one is found (or
// the ray escapes, or too many transmissive hits are seen in a row).
//
// Discovered as a real, measured brightness gap, not a hypothetical: a
// glass sphere sitting between many floor/wall points and scene 0's ceiling
// light made CPU's NEE succeed in ~63% of geometrically-valid attempts
// where GPU succeeded in ~77%, on the identical scene - CPU was quietly
// discarding every one of those samples as "occluded" by glass it should
// have seen through, undercounting its dominant light-sampling term by about
// the same margin as the CPU/GPU brightness gap this was chasing.
//==============================================================================================

#include "hittable.h"
#include "material_base.h"

// Finds the first OPAQUE surface a shadow ray reaches, walking straight
// through (not refracting - a shadow ray is a visibility test, not a light
// transport simulation) any material whose is_shadow_transmissive() says so.
//
// Returns true and fills out_hit with that opaque surface's hit_record when
// one is found within t_max. Returns false if the ray escapes to t_max (or
// beyond, for the sky/background case) without finding one - the caller
// decides what "nothing opaque in the way" means for its own light: for an
// area light it means keep walking (this only returns false when there is
// truly nothing left, so the caller should not still be expecting a light
// hit); for the sky or a punctual light it IS the answer ("visible").
//
// t_max defaults to infinity for the common area-light/sky case, where the
// caller identifies the light by testing out_hit itself (emitted() > 0)
// rather than by distance.
//
// out_transmittance, when non-null, accumulates each transmissive hit's
// material::shadow_transmittance() (default 1.0/no attenuation for
// ordinary transmissive surfaces like glass; real Beer-Lambert/ratio-
// tracking attenuation for media - see material_base.h's
// shadow_transmittance() comment). An optional trailing pointer rather
// than a required parameter so every pre-existing call site (SPPM/BDPT
// adapters, the unit tests) keeps compiling unchanged - only camera.h's
// real NEE call sites need it, and should multiply their light
// contribution by it regardless of whether this returns true or false,
// since a light can be dimmed by fog without being fully occluded by it.
inline bool shadow_ray_hit(
    const hittable& world, const ray& r, hit_record& out_hit,
    double t_max = infinity, color* out_transmittance = nullptr
) {
    // Bounds a stack of transmissive objects in a row (several glass panes,
    // say) rather than looping until t_max is exhausted one epsilon-sized
    // step at a time. Past this many, conservatively call it occluded -
    // safe because it only matters for scenes deliberately stacking this
    // many transmissive surfaces, far beyond anything rendered here.
    constexpr int kMaxTransmissiveSkips = 32;

    ray current = r;
    double t_min = 0.001;
    double remaining = t_max;
    if (out_transmittance) *out_transmittance = color(1, 1, 1);

    for (int i = 0; i < kMaxTransmissiveSkips; ++i) {
        hit_record rec;
        if (!world.hit(current, interval(t_min, remaining), rec))
            return false;  // nothing (opaque or otherwise) left before t_max

        if (!rec.mat->is_shadow_transmissive(rec)) {
            out_hit = rec;
            return true;
        }

        if (out_transmittance)
            *out_transmittance = *out_transmittance * rec.mat->shadow_transmittance(current);

        // Transmissive: step past this point and keep going in the SAME
        // direction - matches optixIgnoreIntersection()'s effect exactly,
        // continuing traversal unchanged rather than bending the ray.
        remaining -= rec.t;
        if (remaining <= t_min) return false;
        current = ray(rec.p, r.direction(), r.time());
    }
    return false;  // too many transmissive hits in a row; treat as occluded
}

#endif
