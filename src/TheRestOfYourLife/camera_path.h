#pragma once
// camera_path.h -- camera::ray_color(), ray_color_spectral() and sample_bssrdf_exit(): the CPU path tracing integrators, defined out of class here
// (declarations stay in camera.h). A pure move; included by camera.h after the class.

inline bool camera::sample_bssrdf_exit(hit_record& rec, scatter_record& srec,
    const ray& current_ray, const hittable& world,
    color& beta, double& eta_scale) const
{
    const subsurface* ss = rec.mat->as_subsurface(rec);
    if (!ss) return false;

    const color  entry_attenuation     = srec.attenuation;
    const double entry_eta             = srec.eta;
    const bool   entry_is_transmission = srec.is_transmission;

    const point3 p0   = rec.p;
    const vec3   axis = unit_vector(rec.normal);
    // Any tangent frame around axis works -- the disc is sampled with a
    // uniform phi, so its orientation doesn't matter, only that t1/t2/axis
    // are mutually orthonormal.
    vec3 t1 = unit_vector(std::fabs(axis.x()) > 0.9
                             ? cross(vec3(0, 1, 0), axis)
                             : cross(vec3(1, 0, 0), axis));
    vec3 t2 = cross(axis, t1);

    const TabulatedBSSRDF& bssrdf = ss->get_bssrdf();

    const int channel = std::min(2, static_cast<int>(random_double() * 3.0));
    const double r = bssrdf.sample_sr(channel, random_double());
    if (r < 0.0) return false;
    const double r_max = bssrdf.sample_sr(channel, 0.999);
    if (r_max <= 0.0 || r >= r_max) return false;

    const double phi      = 2.0 * pi * random_double();
    const double half_len = std::sqrt(std::max(0.0, r_max * r_max - r * r));

    // Choose one of 3 mutually-orthonormal probe axes for the actual
    // walk -- matches pbrt-v4 SampleSp's axis-selection probabilities
    // (Frame::FromZ(ns) 0.5, FromX(ns)/FromY(ns) 0.25 each). Whichever
    // axis is NOT the probe direction supplies the other disc basis
    // vector alongside the remaining tangent, so every axis choice still
    // walks a segment through p0's local neighborhood, just oriented
    // differently -- this is what lets the probe still find an exit
    // point when the true nearby surface is edge-on to the shading
    // normal (thin fins/creases), which a normal-only probe would
    // systematically miss.
    const double u_axis = random_double();
    vec3 probe_axis, basis_a, basis_b;
    if (u_axis < 0.5)      { probe_axis = axis; basis_a = t1;   basis_b = t2; }
    else if (u_axis < 0.75) { probe_axis = t1;   basis_a = t2;   basis_b = axis; }
    else                    { probe_axis = t2;   basis_a = axis; basis_b = t1; }

    const point3 p_target = p0 + r * (std::cos(phi) * basis_a + std::sin(phi) * basis_b);
    const point3 p_start  = p_target - half_len * probe_axis;
    const point3 p_end    = p_target + half_len * probe_axis;

    vec3   seg_dir = p_end - p_start;
    double seg_len = seg_dir.length();
    if (seg_len < 1e-10) return false;
    seg_dir = seg_dir / seg_len;

    // Walk the probe segment, collecting every hit on the SAME material
    // (matches pbrt-v4's `si->intr.material == isect.material` check)
    // via unweighted reservoir sampling (Algorithm R: accept candidate i
    // with probability 1/i) so the chosen exit point is uniformly
    // distributed among however many candidates the segment crosses --
    // equivalent to src/shared/reservoir_sampler.h's
    // WeightedReservoirSampler with every weight equal to 1, reimplemented
    // inline here rather than included: that header's own AliasTable
    // collides (same class name, different definition) with
    // power_light_sampler.h's, which camera.h already pulls in
    // transitively, and both would land in the same translation unit.
    const material* target_mat = rec.mat.get();
    hit_record chosen_hit;
    int candidate_count = 0;
    point3 base      = p_start;
    double remaining = seg_len;
    while (remaining > 1e-9) {
        hit_record probe_rec;
        ray probe_ray(base, seg_dir, current_ray.time());
        if (!world.hit(probe_ray, interval(1e-6, remaining), probe_rec)) break;
        if (probe_rec.mat.get() == target_mat) {
            ++candidate_count;
            if (random_double() < 1.0 / candidate_count)
                chosen_hit = probe_rec;
        }
        const double step = probe_rec.t + 1e-4;
        base = base + step * seg_dir;
        remaining -= step;
    }
    if (candidate_count == 0) return false;

    const hit_record& exit_hit = chosen_hit;
    const double sample_prob   = 1.0 / candidate_count;

    // Sp(pi) = Sr(Distance(po, pi)) -- pbrt-v4 bssrdf.h TabulatedBSSRDF::Sp,
    // full 3D distance is correct here.
    const double dist = (exit_hit.p - p0).length();
    const color  Sp(bssrdf.sr(0, dist), bssrdf.sr(1, dist), bssrdf.sr(2, dist));

    // PDF_Sp, however, is NOT PDF_Sr(dist) -- pbrt-v4's PDF_Sp projects
    // (pi - po) onto the plane PERPENDICULAR to each candidate probe
    // axis in turn (the profile is defined as a function of radius on
    // that plane, not of straight-line distance to the found point,
    // which also carries an axial offset of up to +-half_len along
    // whichever axis was actually probed) and weights each by the exit
    // point's normal component along that axis (the Jacobian between
    // the disc parameterization and actual surface area, same role as a
    // cosine term). Using `dist` in place of the projected radius
    // systematically under-estimates the pdf for candidates found far
    // along the probe axis (dist > rProj always, and PDF_Sr is
    // decreasing), which inflates 1/pdf into huge, bright-red firefly
    // weights on curved geometry like the SSS dragon meshes.
    //
    // Regardless of which axis was actually walked above, PDF_Sp always
    // returns the *combined* pdf summed over all 3 axes weighted by
    // their selection probability -- the standard one-sample MIS
    // (balance heuristic) estimator for "one of several sampling
    // techniques was used, but every technique could plausibly have
    // produced this same point." This is what reduces variance on
    // creased geometry: a point found via the t1/t2 probes still gets
    // (correctly) up-weighted by however much the normal-axis strategy
    // *would* have favored it too, and vice versa.
    const vec3   d = exit_hit.p - p0;
    const vec3   exit_n = unit_vector(exit_hit.normal);
    const double d_n  = dot(d, axis);
    const double d_t1 = dot(d, t1);
    const double d_t2 = dot(d, t2);
    const double r_proj_axis = std::sqrt(std::max(0.0, d_t1 * d_t1 + d_t2 * d_t2));
    const double r_proj_t1   = std::sqrt(std::max(0.0, d_t2 * d_t2 + d_n  * d_n));
    const double r_proj_t2   = std::sqrt(std::max(0.0, d_n  * d_n  + d_t1 * d_t1));
    const double cos_axis = std::fabs(dot(exit_n, axis));
    const double cos_t1   = std::fabs(dot(exit_n, t1));
    const double cos_t2   = std::fabs(dot(exit_n, t2));
    constexpr double kAxisProb = 0.5, kTangentProb = 0.25;
    double pdf = 0.0;
    for (int c = 0; c < 3; ++c) {
        pdf += kAxisProb   * bssrdf.pdf_sr(c, r_proj_axis) * cos_axis
             + kTangentProb * bssrdf.pdf_sr(c, r_proj_t1)   * cos_t1
             + kTangentProb * bssrdf.pdf_sr(c, r_proj_t2)   * cos_t2;
    }
    pdf /= 3.0;
    if (pdf <= 0.0) return false;

    const double inv = 1.0 / (sample_prob * pdf);
    constexpr double kMaxPathThroughput = 50.0;  // matches ray_color's own ceiling
    color new_beta = beta * entry_attenuation * Sp * inv;
    new_beta = color(std::min(new_beta.x(), kMaxPathThroughput),
                      std::min(new_beta.y(), kMaxPathThroughput),
                      std::min(new_beta.z(), kMaxPathThroughput));
    if (entry_is_transmission) eta_scale *= entry_eta * entry_eta;
    beta = new_beta;

    rec.p          = exit_hit.p;
    rec.normal     = exit_hit.normal;
    rec.dpdu       = exit_hit.dpdu;
    rec.u          = exit_hit.u;
    rec.v          = exit_hit.v;
    rec.front_face = exit_hit.front_face;
    rec.mat        = ss->get_exit_bsdf();   // Sw = normalized_fresnel(eta)

    return rec.mat->scatter(current_ray, rec, srec, false);
}

template <typename Sampler>
    inline color camera::ray_color(const ray& r, int depth, const hittable& world, const hittable& lights,
    Sampler& sampler)
    const
{
    color  L            = color(0, 0, 0);
    color  beta         = color(1, 1, 1);
    ray    current_ray  = r;
    int    bounces_left = depth;
    double prev_bsdf_pdf      = 0.0;
    double eta_scale          = 1.0;  // pbrt-v4: etaScale = product of Sqr(bs->eta) per transmission
    bool   specular_bounce    = true;
    bool   any_nonspecular    = false;  // pbrt-v4: anyNonSpecularBounces
    point3 prev_surface_p     = r.origin(); // pbrt-v4: prevIntrCtx shading point
    int    medium_boundary_crossings = 0;

    // While this path runs, the chromatic per-shape media are sampled here (below) and hidden from hit() - see
    // constant_medium.h's g_chromatic_media_integrator_managed.
    struct ChromaticMediaScope {
        bool prev;
        explicit ChromaticMediaScope(bool on) : prev(g_chromatic_media_integrator_managed) {
            if (on) g_chromatic_media_integrator_managed = true;
        }
        ~ChromaticMediaScope() { g_chromatic_media_integrator_managed = prev; }
    } chromatic_media_scope(!shape_media.empty());

    // Deterministic per-channel transmittance through the chromatic shape media along a shadow ray, up to parameter t_max: what
    // the NEE strategies below multiply in (the media are invisible to shadow_ray_hit() while the scope above is active).
    auto shape_media_trans = [&](const ray& sr, double t_max) -> color {
        color T(1, 1, 1);
        for (const auto& m : shape_media) T = T * m->transmittance_along(sr, t_max);
        return T;
    };

    // Russian roulette below only fires when a path's throughput has
    // dropped under 1.0 - it terminates/reweights *dim* paths, but does
    // nothing when throughput has grown large. General defensive
    // firefly ceiling for the rare case a BSDF's per-bounce ratio (e.g.
    // hair's fr/pdf importance-sampling ratio - see hair_material.h/
    // bxdfs_hair.h, which already clamps that single-bounce ratio to
    // 50) compounds across several bounces into an extreme value.
    // Confirmed NOT the cause of scene B11's original blown-white
    // look (that traced to the scene's overhead light being miscalibrated
    // for hair's naturally bright peak response - see
    // build_hair_fibers()'s comment - clamping throughput here, even
    // aggressively, made no visible difference until the light itself
    // was recalibrated). Kept as a generous, low-risk safety net; a
    // well-behaved BSDF (Fresnel reflectance <=1, importance-sampled
    // f/pdf integrating to ~1) never approaches it.
    constexpr double kMaxPathThroughput = 50.0;
    auto clamp_throughput = [&](const color& c) {
        return color(std::min(c.x(), kMaxPathThroughput),
                     std::min(c.y(), kMaxPathThroughput),
                     std::min(c.z(), kMaxPathThroughput));
    };
    // A medium-boundary crossing (interface_material) is free - doesn't
    // consume bounces_left or an RR trial (see the branch below) - so
    // nothing else bounds how many a single path can take. A degenerate
    // scene (self-intersecting/near-zero-thickness interface geometry)
    // could otherwise hang the path entirely; kMaxMediumBoundaryCrossings
    // (src/shared/cpu_gpu.h) is the one shared bound every integrator
    // that supports this uses, mirroring shadow_ray.h's own
    // kMaxTransmissiveSkips bound for the identical reason.

    // `depth` vertices scatter (light sampling and a continuation ray), and the iteration after the last of them, with bounces_left == 0,
    // only adds what that final continuation ray sees - the MIS-weighted emission of a light it hits, or of the sky/background it escapes
    // to - before stopping, exactly as pbrt-v4's PathIntegrator does (it adds Le, then tests `depth++ >= maxDepth`, then samples lights).
    // Without that iteration, the light sampling at the last vertex was weighted by its MIS share while the BSDF-sampled share it was
    // weighted against was never traced: a diffuse sphere under a uniform sky read 0.09 at depth 1 where one bounce is exactly 0.5,
    // and every depth-limited render came out darker than pbrt's (1% at depth 8, 3% at depth 4, 10% at depth 2 on a Cornell box).
    while (bounces_left >= 0) {
        // One iteration = one traced ray (the primary ray on the first
        // pass, a bounce continuation after) - see render_stats.h's own
        // comment for why this is gated behind enabled() rather than an
        // unconditional atomic increment.
        if (render_stats::enabled())
            render_stats::bounce_rays().fetch_add(1, std::memory_order_relaxed);

        hit_record rec;
        bool hit_something = world.hit(current_ray, interval(0.001, infinity), rec);

        // Per-channel-extinction shape media: one free-flight event per medium over its chord up to the nearest surface, in
        // order of entry (a collision ends the walk and replaces rec; a pass multiplies beta by its importance weight).
        if (!shape_media.empty()) {
            const event_medium* order_buf[8];
            std::size_t n = 0;
            std::vector<const event_medium*> order_heap;
            const event_medium** order = order_buf;
            if (shape_media.size() > 8) {
                order_heap.resize(shape_media.size());
                order = order_heap.data();
            }
            for (const auto& m : shape_media) order[n++] = m.get();
            if (n > 1) {
                double entry[64];
                std::vector<double> entry_heap;
                double* ent = entry;
                if (n > 64) { entry_heap.resize(n); ent = entry_heap.data(); }
                for (std::size_t i = 0; i < n; ++i) {
                    double t0, t1;
                    ent[i] = order[i]->chord(current_ray, t0, t1) ? t0 : infinity;
                }
                for (std::size_t i = 1; i < n; ++i) {   // insertion sort, n is tiny
                    for (std::size_t j = i; j > 0 && ent[j] < ent[j - 1]; --j) {
                        std::swap(ent[j], ent[j - 1]);
                        std::swap(order[j], order[j - 1]);
                    }
                }
            }
            double t_limit = hit_something ? rec.t : infinity;
            for (std::size_t i = 0; i < n; ++i) {
                double entry_t = 0.0;
                bool collided = false;
                hit_record ev_rec;
                color w(1, 1, 1);
                if (!order[i]->sample_event(current_ray, t_limit, entry_t, collided, ev_rec, w)) continue;
                if (collided) {
                    rec = ev_rec;
                    hit_something = true;
                    break;
                }
                beta = beta * w;
            }
        }

        // pbrt-v4's own "camera medium" (camera::camera_medium's own
        // comment) - an unbounded ambient medium the camera itself
        // starts inside. Tested here, AFTER world.hit() already ran, so
        // its own free-path sample can be correctly clipped to whatever
        // real surface (or infinity, for an escaped ray) is already
        // known to be the nearest thing in front of it - see
        // ambient_medium's own comment (constant_medium.h) for why this
        // can't just be one more entry in world's own BVH/hittable_list.
        // Applies on every bounce, not just the primary ray - once
        // inside a medium that fills all of space, every ray segment of
        // the path is inside it too (this v1 doesn't model a real exit
        // via some other shape's own MediumInterface - see
        // pbrt_flatten.h's own warning for that combination).
        if (camera_medium) {
            const double surface_t = hit_something ? rec.t : infinity;
            // sample_scatter() writes rec directly (safe: every one of
            // its failure paths returns before touching rec, so there's
            // no risk of it leaving rec partially mutated on a miss) and
            // hands back the ray length it already computed internally,
            // so the no-scatter branch doesn't need a second sqrt for
            // the same value.
            double ray_length = 0.0;
            color  pass_weight(1, 1, 1);   // only a per-channel-extinction medium sets this
            if (camera_medium->sample_scatter(current_ray, surface_t, rec, &ray_length, &pass_weight)) {
                hit_something = true;
            } else {
                beta = beta * pass_weight;
            }
            // No scatter: the ray reaches the surface (or escapes) with weight 1. The free-flight sample
            // above already decides scatter-or-pass with probabilities 1 - T and T, so multiplying beta by
            // the transmittance T as well (as this branch used to) attenuated a ray that got through by T
            // a second time: a surface seen through a pure absorber of optical depth 1 rendered at
            // exp(-2) instead of exp(-1). (ray_length is only needed by sample_scatter() itself now.)
            (void)ray_length;
        }

        // Miss -- query sky (HDR env map) or fall back to flat background.
        // Mirrors pbrt-v4: "Incorporate emission from infinite lights for escaped ray"
        if (!hit_something) {
            if (portal) {
                // Windowed/portal infinite light - position-dependent
                // (visibility through the finite window depends on the
                // ray's OWN origin, unlike sky->Le()'s direction-only
                // query), so pass current_ray.origin() as the reference
                // point; eval_Le_rgb()/pdf_li() already return
                // zero/black on their own when this escaped ray's
                // origin can't see the window at all, or the direction
                // falls outside it - no separate visibility check
                // needed here.
                point3 ro = current_ray.origin();
                vec3 rd = unit_vector(current_ray.direction());
                double lr, lg, lb;
                portal->eval_Le_rgb(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(), lr, lg, lb);
                color Le(lr, lg, lb);
                if (specular_bounce) {
                    L += beta * Le;
                } else {
                    double p_l = portal->pdf_li(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z());
                    double w_b = mis_power_heuristic(prev_bsdf_pdf, p_l);
                    L += beta * w_b * Le;
                }
            } else if (sky) {
                color Le = sky->Le(unit_vector(current_ray.direction()));
                if (specular_bounce) {
                    // Camera ray or post-specular: full contribution (no MIS needed)
                    L += beta * Le;
                } else {
                    // MIS: balance BSDF-sample weight against sky PDF at this direction
                    // pbrt-v4: p_l = lightSampler.PMF(prevIntrCtx, light) * light.PDF_Li(...);
                    //          w_b = PowerHeuristic(1, p_b, 1, p_l);
                    double p_l = sky->pdf_Li(unit_vector(current_ray.direction()));
                    double w_b = mis_power_heuristic(prev_bsdf_pdf, p_l);
                    L += beta * w_b * Le;
                }
            } else {
                L += beta * background;
            }
            break;
        }

        // Texture-lookup footprint (EWA/mipmap filtering) -- only for the
        // PRIMARY camera-ray hit (bounces_left == depth, i.e. before any
        // bounce has updated current_ray), since only that ray carries
        // real differentials (see get_ray()'s own comment); every bounce/
        // shadow/NEE ray leaves rec.dudx/dvdx/dudy/dvdy at their zero
        // defaults, which is already texture.h's/mipmap.h's correct "no
        // footprint info, use plain bilinear" fallback. Bridges hit_record's flat fields into
        // a temporary SurfaceInteraction<double> purely to reuse the
        // existing, unmodified compute_differentials() (surface_
        // interaction.h) rather than reimplementing its least-squares
        // solve here.
        if (bounces_left == depth && current_ray.has_differentials()) {
            SurfaceInteraction<double> si(
                rec.p.x(), rec.p.y(), rec.p.z(),
                rec.normal.x(), rec.normal.y(), rec.normal.z(),
                rec.u, rec.v, rec.t,
                0.0, 0.0, 0.0,   // wo unused by compute_differentials()
                rec.dpdu.x(), rec.dpdu.y(), rec.dpdu.z(),
                rec.dpdv.x(), rec.dpdv.y(), rec.dpdv.z());
            si.compute_differentials(true,
                current_ray.rx_origin().x(), current_ray.rx_origin().y(), current_ray.rx_origin().z(),
                current_ray.rx_direction().x(), current_ray.rx_direction().y(), current_ray.rx_direction().z(),
                current_ray.ry_origin().x(), current_ray.ry_origin().y(), current_ray.ry_origin().z(),
                current_ray.ry_direction().x(), current_ray.ry_direction().y(), current_ray.ry_direction().z());
            rec.dudx = si.dudx; rec.dvdx = si.dvdx;
            rec.dudy = si.dudy; rec.dvdy = si.dvdy;
        }

        // Emission -- full Le on camera/specular hits; MIS-weighted otherwise.
        // Mirrors pbrt-v4 "Compute MIS weight for area light".
        color Le = rec.mat->emitted(current_ray, rec, rec.u, rec.v, rec.p);
        if (Le.x() > 0 || Le.y() > 0 || Le.z() > 0) {
            if (specular_bounce) {
                L += beta * Le;
            } else {
                // Use prev_surface_p as the PDF origin -- mirrors pbrt-v4 prevIntrCtx.
                // The light PDF must be evaluated from where the BSDF ray was *spawned*
                // (the previous surface), not from the emitter hit point (rec.p).
                hittable_pdf light_pdf_mis(lights, prev_surface_p);
                double pdf_l = light_pdf_mis.value(current_ray.direction());
                double w_b   = mis_power_heuristic(prev_bsdf_pdf, pdf_l);
                L += beta * w_b * Le;
            }
        }

        // The final, emission-only iteration (see the loop's own comment): the emission above is all this vertex contributes; no light
        // sampling, no scattering.
        if (bounces_left == 0) break;

        // No scatter (pure emitter / absorber).
        // do_regularize = regularize && any_nonspecular so rough materials widen
        // their GGX lobe on a thread-local copy -- mirrors pbrt-v4's own
        // "Possibly regularize the BSDF" gate exactly (regularize is a per-
        // Integrator directive default false; any_nonspecular alone isn't enough).
        scatter_record srec;
        if (!rec.mat->scatter(current_ray, rec, srec, regularize && any_nonspecular))
            break;

        // True pass-through (interface_material - pbrt-v4's real "no
        // BSDF" interface material, SkipIntersection in pbrt-v4 itself):
        // nothing actually scattered here, so specular_bounce/
        // prev_bsdf_pdf/prev_surface_p are left exactly as they were -
        // the NEXT emitter hit's MIS weight should reflect whatever the
        // last REAL vertex was, not this crossing. Doesn't consume
        // bounces_left or an RR trial either, matching pbrt-v4's own
        // free SkipIntersection. Checked before the BSSRDF branch below
        // since interface_material never has a subsurface profile.
        if (srec.is_medium_boundary) {
            beta = clamp_throughput(beta * srec.attenuation);
            current_ray = srec.skip_pdf_ray;
            if (++medium_boundary_crossings > kMaxMediumBoundaryCrossings) break;
            continue;
        }

        // BSSRDF subsurface branch: an entry-interface specular
        // transmission into a material with a real diffusion profile
        // (pbrt_flatten::MaterialKind::Subsurface) is redirected here
        // instead of tracing the refracted ray through the interior,
        // which this renderer's surface-only intersection model has no
        // way to do. sample_bssrdf_exit() walks the same object's own
        // geometry to find a physically-sampled exit point and, on
        // success, reassigns rec/srec to it (exit BSDF =
        // normalized_fresnel, pbrt-v4's NormalizedFresnelBxDF) -- see
        // its own comment for the algorithm and documented
        // simplifications. On failure the path terminates cleanly, same
        // as any other zero-contribution sample.
        bool bssrdf_exit = false;
        if (srec.skip_pdf && srec.is_transmission && rec.mat->as_subsurface(rec)) {
            if (!sample_bssrdf_exit(rec, srec, current_ray, world, beta, eta_scale))
                break;
            bssrdf_exit = true;
        }

        // Specular bounce: no NEE, update beta and advance ray
        if (srec.skip_pdf && !bssrdf_exit) {
            color new_beta = clamp_throughput(beta * srec.attenuation);
            // Update etaScale for transmission bounces BEFORE the RR test
            // below, matching pbrt-v4's actual VolPathIntegrator ordering
            // (etaScale *= Sqr(bs->eta) precedes rrBeta = beta * etaScale
            // there) - this used to run after, so rr_beta was computed
            // from THIS bounce's beta but the PREVIOUS bounce's
            // etaScale, one bounce stale. eta_scale only ever grows on a
            // refraction (Sqr of eta ratio > 1 whichever direction light
            // crosses) specifically to counteract RR's tendency to kill
            // transmission-heavy paths too aggressively (a ray that
            // refracted into a denser medium carries more radiance per
            // pbrt's non-symmetric transport scaling, so RR must not
            // judge it by beta alone) - using the stale, smaller
            // etaScale understated rr_beta right after a refraction,
            // giving RR a higher kill probability than pbrt-v4 intends
            // for exactly the paths this mechanism exists to protect.
            if (srec.is_transmission)
                eta_scale *= srec.eta * srec.eta;
            if (bounces_left < depth) {
                // pbrt-v4: rrBeta = beta * etaScale to avoid killing transmission paths
                color rr_beta = new_beta * eta_scale;
                double rr_max = std::max(rr_beta.x(), std::max(rr_beta.y(), rr_beta.z()));
                if (rr_max < 1.0) {
                    double q = std::max(0.0, 1.0 - rr_max);
                    if (sampler.get() < q) break;
                    new_beta = new_beta / (1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = srec.skip_pdf_ray;
            specular_bounce = true;
            prev_bsdf_pdf   = 0.0;
            --bounces_left;
            continue;
        }

        // Non-specular: NEE shadow ray + BSDF path continuation
        // Mirrors pbrt-v4: L += beta * SampleLd(...)  then SpawnRay(bsdf_sample)
        hittable_pdf light_pdf(lights, rec.p);

        // Camera-medium shadow-ray attenuation (closes the gap
        // ambient_medium's own "v1 scope" comment documents:
        // "no shadow-ray/NEE attenuation through it yet ... a light
        // behind fog through this medium is not dimmed by it"). The
        // camera medium is untraced geometry (see camera_medium's own
        // comment above for why it isn't a hittable), so
        // shadow_ray_hit()'s own transmittance walk never sees it - it
        // has to be applied here, once per NEE strategy, exactly the
        // same Beer-Lambert transmittance_over() the primary/bounce ray
        // already applies each iteration (see the `if (camera_medium)`
        // block right after world.hit() above). `distance` is real
        // world units (already ray_length-scaled, matching that same
        // call's `surface_t * ray_length`) - `infinity` is the correct,
        // exact answer for a light at unbounded distance (sky, or the
        // portal light's own window, both treated as infinitely distant
        // for this purpose): transmittance_over(infinity) is 0 for any
        // channel with real extinction, matching this same medium's own
        // physical model (nothing is visible arbitrarily far through an
        // unbounded absorbing/scattering fog).
        auto camera_medium_trans = [&](double distance) -> color {
            return camera_medium ? camera_medium->transmittance_over(distance) : color(1, 1, 1);
        };

        // Strategy A-1: NEE toward area lights (non-recursive shadow test)
        {
            vec3   light_dir = light_pdf.generate();
            double pdf_l     = light_pdf.value(light_dir);
            if (pdf_l > 0.0) {
                ray    shadow_ray(rec.p, light_dir, current_ray.time());
                double f_pdf = rec.mat->scattering_pdf(current_ray, rec, shadow_ray);
                if (f_pdf > 0.0) {
                    double pdf_b_at_l = srec.mis_pdf().value(light_dir);
                    double w_l        = mis_power_heuristic(pdf_l, pdf_b_at_l);
                    hit_record light_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (shadow_ray_hit(world, shadow_ray, light_rec, infinity, &trans)) {
                        color Le_d = light_rec.mat->emitted(
                            shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p);
                        if (Le_d.x() > 0 || Le_d.y() > 0 || Le_d.z() > 0) {
                            color atten = rec.mat->scattering_attenuation(current_ray, rec, shadow_ray, srec.attenuation);
                            // shadow_ray_hit() restarts its ray at every transmissive surface it walks through (an interface
                            // shell, glass), so light_rec.t is measured from the LAST of those, not from shadow_ray's origin.
                            // The distance to the light's hit point is the real extent of the segment the media see.
                            const double dir_len = shadow_ray.direction().length();
                            const double t_light = (light_rec.p - shadow_ray.origin()).length() / dir_len;
                            color med_trans = camera_medium_trans(t_light * dir_len)
                                * shape_media_trans(shadow_ray, t_light);
                            L += beta * w_l * atten * trans * med_trans * f_pdf * Le_d / pdf_l;
                        }
                    }
                }
            }
        }

        // Strategy A-2: NEE toward sky (pbrt-v4 SampleLd for infinite lights)
        // Uses importance-sampled direction when HDR distribution is available,
        // falling back to uniform sphere for solid-color skies.
        if (portal) {
            // Windowed/portal infinite light - sample_li() needs the
            // shading point (visibility through the finite window is
            // position-dependent, unlike sky->sample_Le()'s pure
            // direction sample) and its own explicit (ru,rv) draw
            // (sample_li() has no internal RNG the way sky_light does).
            double ru = random_double(), rv = random_double();
            double wx, wy, wz, pdf_portal;
            if (portal->sample_li(ru, rv, rec.p.x(), rec.p.y(), rec.p.z(),
                                   wx, wy, wz, pdf_portal) && pdf_portal > 0.0) {
                vec3 portal_dir(wx, wy, wz);
                ray  portal_shadow(rec.p, portal_dir, current_ray.time());
                double f_pdf = rec.mat->scattering_pdf(current_ray, rec, portal_shadow);
                if (f_pdf > 0.0) {
                    double pdf_b_at_portal = srec.mis_pdf().value(portal_dir);
                    double w_portal         = mis_power_heuristic(pdf_portal, pdf_b_at_portal);
                    hit_record portal_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (!shadow_ray_hit(world, portal_shadow, portal_rec, infinity, &trans)) {
                        double lr, lg, lb;
                        portal->eval_Le_rgb(rec.p.x(), rec.p.y(), rec.p.z(), wx, wy, wz, lr, lg, lb);
                        color Le_portal(lr, lg, lb);
                        color atten = rec.mat->scattering_attenuation(current_ray, rec, portal_shadow, srec.attenuation);
                        color med_trans = camera_medium_trans(infinity) * shape_media_trans(portal_shadow, infinity);
                        L += beta * w_portal * atten * trans * med_trans * f_pdf * Le_portal / pdf_portal;
                    }
                }
            }
        } else if (sky) {
            SkyLiSample sky_smp = sky->sample_Le();
            vec3   sky_dir  = sky_smp.direction;
            double pdf_sky  = sky_smp.pdf;
            if (pdf_sky > 0.0) {
                ray    sky_shadow(rec.p, sky_dir, current_ray.time());
                double f_pdf = rec.mat->scattering_pdf(current_ray, rec, sky_shadow);
                if (f_pdf > 0.0) {
                    double pdf_b_at_sky = srec.mis_pdf().value(sky_dir);
                    double w_sky        = mis_power_heuristic(pdf_sky, pdf_b_at_sky);
                    hit_record sky_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (!shadow_ray_hit(world, sky_shadow, sky_rec, infinity, &trans)) {
                        color Le_sky = sky->Le(unit_vector(sky_dir));
                        color atten = rec.mat->scattering_attenuation(current_ray, rec, sky_shadow, srec.attenuation);
                        color med_trans = camera_medium_trans(infinity) * shape_media_trans(sky_shadow, infinity);
                        L += beta * w_sky * atten * trans * med_trans * f_pdf * Le_sky / pdf_sky;
                    }
                }
            }
        }

        // Strategy A-3: NEE toward punctual (delta) lights
        // pbrt-v4: DeltaPosition/DeltaDirection lights bypass MIS -- PDF=1 (delta),
        // contribution = beta * BSDF * Li (no MIS weight needed since PDF=1).
        if (punct_lights && !punct_lights->empty()) {
            punct_lights->for_each_sample(rec.p, [&](const PunctualLiSample& ps) {
                if (ps.Li.x() <= 0 && ps.Li.y() <= 0 && ps.Li.z() <= 0) return;
                ray punct_ray(rec.p, ps.wi, current_ray.time());
                double f_pdf = rec.mat->scattering_pdf(current_ray, rec, punct_ray);
                if (f_pdf <= 0.0) return;
                hit_record shadow_rec;
                double shadow_t_max = (ps.t_max == infinity) ? infinity : (ps.t_max - 0.001);
                color trans;
                if (render_stats::enabled())
                    render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                if (!shadow_ray_hit(world, punct_ray, shadow_rec, shadow_t_max, &trans)) {
                    // delta light: pdf=1, no MIS weight needed
                    color atten = rec.mat->scattering_attenuation(current_ray, rec, punct_ray, srec.attenuation);
                    color med_trans = camera_medium_trans(ps.t_max) * shape_media_trans(punct_ray, shadow_t_max);
                    L += beta * atten * trans * med_trans * f_pdf * ps.Li;
                }
            });
        }

        // Strategy B: BSDF sample becomes next path ray
        if (srec.has_walk) {
            // A layered (coated) BSDF: pbrt continues along the random walk's own sample with its weight
            // f*cos/pdf, and weighs the emitter hit it may find with BSDF::PDF() (mis_pdf), not the walk's
            // density (BSDFSample::pdfIsProportional). The NEE strategies above already ran for this vertex.
            if (!srec.walk_valid) break;
            const ray bsdf_ray = srec.walk_ray;
            color new_beta = clamp_throughput(beta * srec.walk_weight);
            if (bounces_left < depth) {
                color rr_beta = new_beta * eta_scale;
                double rr_max = std::max(rr_beta.x(), std::max(rr_beta.y(), rr_beta.z()));
                if (rr_max < 1.0) {
                    double q = std::max(0.0, 1.0 - rr_max);
                    if (sampler.get() < q) break;
                    new_beta = new_beta / (1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = bsdf_ray;
            prev_bsdf_pdf   = srec.walk_specular ? 0.0 : srec.mis_pdf().value(bsdf_ray.direction());
            prev_surface_p  = rec.p;
            specular_bounce = srec.walk_specular;
            if (!srec.walk_specular) any_nonspecular = true;
            --bounces_left;
        } else {
            vec3   bsdf_dir = srec.pdf_ptr->generate();
            double pdf_b    = srec.pdf_ptr->value(bsdf_dir);
            if (pdf_b <= 0.0) break;

            ray    bsdf_ray(rec.p, bsdf_dir, current_ray.time());
            double f_pdf = rec.mat->scattering_pdf(current_ray, rec, bsdf_ray);
            if (f_pdf <= 0.0) break;

            // Update etaScale for transmission bounces BEFORE the RR test
            // below - same ordering as the specular branch above (see its
            // own comment). pbrt-v4's VolPathIntegrator updates etaScale
            // after every BSDF sample, specular or not.
            //
            // srec.is_transmission here means "this material has real
            // refraction physics with ratio srec.eta" (a fixed, direction-
            // independent per-scatter-event constant a material sets once
            // in scatter()) -- NOT "scatter()'s own single sample happened
            // to transmit", since on this non-specular path the actual
            // bounce direction (bsdf_dir) is resampled fresh from
            // srec.pdf_ptr, independently of whatever scatter() itself
            // may or may not have sampled. Whether THIS SPECIFIC bsdf_dir
            // is a transmission is therefore re-derived geometrically
            // (crossed to the opposite side of the surface from the view
            // direction) rather than trusted from scatter()-time. Every
            // material that never sets is_transmission=true on this path
            // (the overwhelming majority - diffuse_transmission's own R/T
            // lobes are geometric hemisphere flips, not IOR refractions)
            // is completely unaffected, since the `&&` short-circuits.
            if (srec.is_transmission) {
                bool crossed_boundary =
                    dot(bsdf_dir, rec.normal) *
                    dot(-unit_vector(current_ray.direction()), rec.normal) < 0.0;
                if (crossed_boundary)
                    eta_scale *= srec.eta * srec.eta;
            }

            // Russian Roulette after first bounce
            color new_beta = clamp_throughput(beta * rec.mat->scattering_attenuation(current_ray, rec, bsdf_ray, srec.attenuation) * f_pdf / pdf_b);
            if (bounces_left < depth) {
                // pbrt-v4: rrBeta = beta * etaScale
                color rr_beta = new_beta * eta_scale;
                double rr_max = std::max(rr_beta.x(), std::max(rr_beta.y(), rr_beta.z()));
                if (rr_max < 1.0) {
                    double q = std::max(0.0, 1.0 - rr_max);
                    if (sampler.get() < q) break;
                    new_beta = new_beta / (1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = bsdf_ray;
            prev_bsdf_pdf   = pdf_b;
            prev_surface_p  = rec.p;   // pbrt-v4: prevIntrCtx = si->intr
            specular_bounce = false;
            any_nonspecular = true;   // pbrt-v4: anyNonSpecularBounces |= true
            --bounces_left;
        }
    }

    return L;
}

template <typename Sampler>
    inline color camera::ray_color_spectral(const ray& r, int depth, const hittable& world, const hittable& lights,
    Sampler& sampler)
    const
{
    using SS  = SampledSpectrum<4>;
    using SWL = SampledWavelengths<4>;

    SWL swl = SWL::SampleVisible(static_cast<float>(sampler.get()));
    const RGBColorSpace& cs = RGBColorSpace::sRGB();
    const DenselySampledSpectrum* d65 = &GetNormalizedD65Illuminant();
    auto albedo = [&](const color& c) -> SS {
        // Fast path: exact white (1,1,1) is the identity reflectance -
        // by construction the RGBToSpectrumTable lookup below always
        // resolves it to the flat spectrum 1 at every wavelength, so
        // skip the table lookup + sigmoid-polynomial eval entirely.
        // This is the overwhelmingly common case for shadow-ray
        // transmittance (`trans` at this function's NEE call sites):
        // shadow_ray.h initializes it to color(1,1,1) and only touches
        // it when the shadow ray actually crosses a transmissive
        // surface/medium, so most NEE samples hit this path.
        if (c.x() == 1.0 && c.y() == 1.0 && c.z() == 1.0)
            return SS(1.f);
        return RGBAlbedoSpectrum(cs, (float)c.x(), (float)c.y(), (float)c.z()).Sample(swl);
    };
    auto illuminant = [&](const color& c) -> SS {
        return RGBIlluminantSpectrum(cs, (float)c.x(), (float)c.y(), (float)c.z(), d65).Sample(swl);
    };

    SS     L            (0.f);
    SS     beta         (1.f);
    ray    current_ray  = r;
    int    bounces_left = depth;
    double prev_bsdf_pdf      = 0.0;
    double eta_scale          = 1.0;
    bool   specular_bounce    = true;
    bool   any_nonspecular    = false;
    point3 prev_surface_p     = r.origin();
    int    medium_boundary_crossings = 0;

    // See ray_color()'s own kMaxPathThroughput comment - identical
    // ceiling, applied per spectral channel instead of per RGB channel.
    constexpr float kMaxPathThroughput = 50.f;
    auto clamp_throughput = [&](const SS& c) {
        SS out = c;
        for (int i = 0; i < 4; ++i)
            if (out[i] > kMaxPathThroughput) out[i] = kMaxPathThroughput;
        return out;
    };
    // See ray_color()'s own kMaxMediumBoundaryCrossings comment.

    // One extra, emission-only iteration at bounces_left == 0 - see ray_color()'s own comment on this loop.
    while (bounces_left >= 0) {
        if (render_stats::enabled())
            render_stats::bounce_rays().fetch_add(1, std::memory_order_relaxed);

        hit_record rec;

        if (!world.hit(current_ray, interval(0.001, infinity), rec)) {
            if (portal) {
                // See ray_color()'s own portal miss-handling comment -
                // identical logic, spectral-uplifted via illuminant().
                point3 ro = current_ray.origin();
                vec3 rd = unit_vector(current_ray.direction());
                double lr, lg, lb;
                portal->eval_Le_rgb(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(), lr, lg, lb);
                SS Le = illuminant(color(lr, lg, lb));
                if (specular_bounce) {
                    L += beta * Le;
                } else {
                    double p_l = portal->pdf_li(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z());
                    double w_b = mis_power_heuristic(prev_bsdf_pdf, p_l);
                    L += beta * static_cast<float>(w_b) * Le;
                }
            } else if (sky) {
                SS Le = illuminant(sky->Le(unit_vector(current_ray.direction())));
                if (specular_bounce) {
                    L += beta * Le;
                } else {
                    double p_l = sky->pdf_Li(unit_vector(current_ray.direction()));
                    double w_b = mis_power_heuristic(prev_bsdf_pdf, p_l);
                    L += beta * static_cast<float>(w_b) * Le;
                }
            } else {
                L += beta * illuminant(background);
            }
            break;
        }

        // Texture-lookup footprint - identical to ray_color(), no
        // spectral dependence (see that function's own comment).
        if (bounces_left == depth && current_ray.has_differentials()) {
            SurfaceInteraction<double> si(
                rec.p.x(), rec.p.y(), rec.p.z(),
                rec.normal.x(), rec.normal.y(), rec.normal.z(),
                rec.u, rec.v, rec.t,
                0.0, 0.0, 0.0,
                rec.dpdu.x(), rec.dpdu.y(), rec.dpdu.z(),
                rec.dpdv.x(), rec.dpdv.y(), rec.dpdv.z());
            si.compute_differentials(true,
                current_ray.rx_origin().x(), current_ray.rx_origin().y(), current_ray.rx_origin().z(),
                current_ray.rx_direction().x(), current_ray.rx_direction().y(), current_ray.rx_direction().z(),
                current_ray.ry_origin().x(), current_ray.ry_origin().y(), current_ray.ry_origin().z(),
                current_ray.ry_direction().x(), current_ray.ry_direction().y(), current_ray.ry_direction().z());
            rec.dudx = si.dudx; rec.dvdx = si.dvdx;
            rec.dudy = si.dudy; rec.dvdy = si.dvdy;
        }

        color Le_rgb = rec.mat->emitted(current_ray, rec, rec.u, rec.v, rec.p);
        if (Le_rgb.x() > 0 || Le_rgb.y() > 0 || Le_rgb.z() > 0) {
            SS Le = illuminant(Le_rgb);
            if (specular_bounce) {
                L += beta * Le;
            } else {
                hittable_pdf light_pdf_mis(lights, prev_surface_p);
                double pdf_l = light_pdf_mis.value(current_ray.direction());
                double w_b   = mis_power_heuristic(prev_bsdf_pdf, pdf_l);
                L += beta * static_cast<float>(w_b) * Le;
            }
        }

        // The final, emission-only iteration (see the loop's own comment): the emission above is all this vertex contributes; no light
        // sampling, no scattering.
        if (bounces_left == 0) break;

        // Dispersive material (any kind - smooth dielectric, rough
        // dielectric, or any future one): dispatch to
        // scatter_dispersive() instead of the ordinary virtual scatter(),
        // passing the path's hero wavelength so a material built via its
        // own make_dispersive() factory actually refracts differently
        // per wavelength. Every other material (and a non-dispersive
        // dielectric/rough_dielectric) takes the unchanged scatter()
        // path. `disp` stays alive past this point for the NEE/Strategy-B
        // dispatch below - smooth dielectric's dispersive path never
        // reaches NEE (always skip_pdf), but rough_dielectric's does.
        //
        // Uses material::as_dispersive(rec) - the same wrapper-
        // forwarding pattern as as_subsurface() - rather than a raw
        // dynamic_cast, so a dispersive material mixed into a
        // mix_material is still found through the wrapper instead of
        // silently losing dispersion the moment rec.mat is the
        // mix_material rather than the material it stochastically
        // picked. One shared hook and one dispersive_material interface
        // (material_base.h) cover every dispersive concrete type, so
        // adding a future one needs no new dispatch arm here.
        scatter_record srec;
        const dispersive_material* disp = rec.mat->as_dispersive(rec);
        // Only meaningful once scatter_dispersive() below resolves it;
        // reused by scattering_pdf_at() further down instead of every
        // NEE/Strategy-B call re-deriving eta from lambda_nm itself.
        double dispersive_eta = 0.0;
        bool scattered = disp
            ? disp->scatter_dispersive(current_ray, rec, srec, static_cast<float>(swl.lambda[0]),
                                        regularize && any_nonspecular, dispersive_eta)
            : rec.mat->scatter(current_ray, rec, srec, regularize && any_nonspecular);
        if (!scattered)
            break;

        // True pass-through (interface_material) - see ray_color()'s own
        // identical branch for the full rationale. `disp` is always
        // nullptr for interface_material (no dispersion), so this can't
        // interact with the TerminateSecondary() gate below either way.
        if (srec.is_medium_boundary) {
            beta = clamp_throughput(beta * albedo(srec.attenuation));
            current_ray = srec.skip_pdf_ray;
            if (++medium_boundary_crossings > kMaxMediumBoundaryCrossings) break;
            continue;
        }

        // Collapse to the hero wavelength once a REAL dispersive
        // refraction happened - see TerminateSecondary()'s own comment
        // (sampled_spectrum.h) for why the other 3 channels' PDFs must
        // be zeroed after this (their shared pre-refraction direction
        // is no longer valid per-wavelength). Gated on `disp` alone -
        // as_dispersive() only ever returns non-null for an already-
        // dispersive material (dispersive_material's own "non-null
        // means yes" contract), so there's nothing further to re-check.
        //
        // Fires unconditionally on every hit of a dispersive rough
        // dielectric, not just transmission-bound ones: rough_dielectric
        // ::scatter()'s glossy branch always sets srec.is_transmission
        // = true regardless of which lobe eventually gets sampled (see
        // that function's own comment), and RoughDielectricBxDF::f()'s
        // REFLECTION lobe is also Fresnel/eta-dependent, not just the
        // transmission one - so a dispersive rough dielectric's
        // per-wavelength divergence is real the moment this material is
        // hit at all, not only when the sampled bounce happens to cross
        // the boundary.
        if (srec.is_transmission && disp)
            swl.TerminateSecondary();

        // Specular bounce: no NEE, update beta and advance ray.
        if (srec.skip_pdf) {
            SS new_beta = clamp_throughput(beta * albedo(srec.attenuation));
            if (srec.is_transmission)
                eta_scale *= srec.eta * srec.eta;
            if (bounces_left < depth) {
                SS rr_beta = new_beta * static_cast<float>(eta_scale);
                float rr_max = rr_beta.MaxComponentValue();
                if (rr_max < 1.0f) {
                    double q = std::max(0.0, 1.0 - static_cast<double>(rr_max));
                    if (sampler.get() < q) break;
                    new_beta = new_beta / static_cast<float>(1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = srec.skip_pdf_ray;
            specular_bounce = true;
            prev_bsdf_pdf   = 0.0;
            --bounces_left;
            continue;
        }

        // Non-specular: NEE shadow ray + BSDF path continuation
        hittable_pdf light_pdf(lights, rec.p);

        // Every scattering_pdf() call below goes through this instead of
        // calling rec.mat->scattering_pdf(...) directly, so a dispersive
        // material's real NEE/MIS path (currently only rough_dielectric
        // reaches this - smooth dielectric is always skip_pdf and never
        // gets here) evaluates f*cos at the SAME per-wavelength eta
        // scatter_dispersive() already resolved above (dispersive_eta),
        // instead of every NEE strategy/Strategy B independently
        // re-deriving it from the hero wavelength - see
        // dispersive_material::scattering_pdf_dispersive()'s own comment
        // (material_base.h). A no-op for every other material (disp is
        // null), so this changes nothing for the common case.
        auto scattering_pdf_at = [&](const ray& scattered) -> double {
            return disp
                ? disp->scattering_pdf_dispersive(current_ray, rec, scattered, dispersive_eta)
                : rec.mat->scattering_pdf(current_ray, rec, scattered);
        };

        // Strategy A-1: NEE toward area lights
        {
            vec3   light_dir = light_pdf.generate();
            double pdf_l     = light_pdf.value(light_dir);
            if (pdf_l > 0.0) {
                ray    shadow_ray(rec.p, light_dir, current_ray.time());
                double f_pdf = scattering_pdf_at(shadow_ray);
                if (f_pdf > 0.0) {
                    double pdf_b_at_l = srec.mis_pdf().value(light_dir);
                    double w_l        = mis_power_heuristic(pdf_l, pdf_b_at_l);
                    hit_record light_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (shadow_ray_hit(world, shadow_ray, light_rec, infinity, &trans)) {
                        color Le_d_rgb = light_rec.mat->emitted(
                            shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p);
                        if (Le_d_rgb.x() > 0 || Le_d_rgb.y() > 0 || Le_d_rgb.z() > 0) {
                            color atten_rgb = rec.mat->scattering_attenuation(current_ray, rec, shadow_ray, srec.attenuation);
                            float scale = static_cast<float>(w_l * f_pdf / pdf_l);
                            L += beta * scale * albedo(atten_rgb) * albedo(trans) * illuminant(Le_d_rgb);
                        }
                    }
                }
            }
        }

        // Strategy A-2: NEE toward sky
        if (portal) {
            // See ray_color()'s own portal NEE comment - identical
            // logic, spectral-uplifted via albedo()/illuminant().
            double ru = random_double(), rv = random_double();
            double wx, wy, wz, pdf_portal;
            if (portal->sample_li(ru, rv, rec.p.x(), rec.p.y(), rec.p.z(),
                                   wx, wy, wz, pdf_portal) && pdf_portal > 0.0) {
                vec3 portal_dir(wx, wy, wz);
                ray  portal_shadow(rec.p, portal_dir, current_ray.time());
                double f_pdf = scattering_pdf_at(portal_shadow);
                if (f_pdf > 0.0) {
                    double pdf_b_at_portal = srec.mis_pdf().value(portal_dir);
                    double w_portal         = mis_power_heuristic(pdf_portal, pdf_b_at_portal);
                    hit_record portal_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (!shadow_ray_hit(world, portal_shadow, portal_rec, infinity, &trans)) {
                        double lr, lg, lb;
                        portal->eval_Le_rgb(rec.p.x(), rec.p.y(), rec.p.z(), wx, wy, wz, lr, lg, lb);
                        color atten_rgb = rec.mat->scattering_attenuation(current_ray, rec, portal_shadow, srec.attenuation);
                        float scale = static_cast<float>(w_portal * f_pdf / pdf_portal);
                        L += beta * scale * albedo(atten_rgb) * albedo(trans)
                           * illuminant(color(lr, lg, lb));
                    }
                }
            }
        } else if (sky) {
            SkyLiSample sky_smp = sky->sample_Le();
            vec3   sky_dir  = sky_smp.direction;
            double pdf_sky  = sky_smp.pdf;
            if (pdf_sky > 0.0) {
                ray    sky_shadow(rec.p, sky_dir, current_ray.time());
                double f_pdf = scattering_pdf_at(sky_shadow);
                if (f_pdf > 0.0) {
                    double pdf_b_at_sky = srec.mis_pdf().value(sky_dir);
                    double w_sky        = mis_power_heuristic(pdf_sky, pdf_b_at_sky);
                    hit_record sky_rec;
                    color trans;
                    if (render_stats::enabled())
                        render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                    if (!shadow_ray_hit(world, sky_shadow, sky_rec, infinity, &trans)) {
                        color atten_rgb = rec.mat->scattering_attenuation(current_ray, rec, sky_shadow, srec.attenuation);
                        float scale = static_cast<float>(w_sky * f_pdf / pdf_sky);
                        L += beta * scale * albedo(atten_rgb) * albedo(trans)
                           * illuminant(sky->Le(unit_vector(sky_dir)));
                    }
                }
            }
        }

        // Strategy A-3: NEE toward punctual (delta) lights
        if (punct_lights && !punct_lights->empty()) {
            punct_lights->for_each_sample(rec.p, [&](const PunctualLiSample& ps) {
                if (ps.Li.x() <= 0 && ps.Li.y() <= 0 && ps.Li.z() <= 0) return;
                ray punct_ray(rec.p, ps.wi, current_ray.time());
                double f_pdf = scattering_pdf_at(punct_ray);
                if (f_pdf <= 0.0) return;
                hit_record shadow_rec;
                double shadow_t_max = (ps.t_max == infinity) ? infinity : (ps.t_max - 0.001);
                color trans;
                if (render_stats::enabled())
                    render_stats::shadow_rays().fetch_add(1, std::memory_order_relaxed);
                if (!shadow_ray_hit(world, punct_ray, shadow_rec, shadow_t_max, &trans)) {
                    color atten_rgb = rec.mat->scattering_attenuation(current_ray, rec, punct_ray, srec.attenuation);
                    L += beta * static_cast<float>(f_pdf) * albedo(atten_rgb) * albedo(trans) * illuminant(ps.Li);
                }
            });
        }

        // Strategy B: BSDF sample becomes next path ray
        if (srec.has_walk) {
            // Layered (coated) BSDF - see ray_color()'s identical branch.
            if (!srec.walk_valid) break;
            const ray bsdf_ray = srec.walk_ray;
            SS new_beta = clamp_throughput(beta * albedo(srec.walk_weight));
            if (bounces_left < depth) {
                SS rr_beta = new_beta * static_cast<float>(eta_scale);
                float rr_max = rr_beta.MaxComponentValue();
                if (rr_max < 1.0f) {
                    double q = std::max(0.0, 1.0 - static_cast<double>(rr_max));
                    if (sampler.get() < q) break;
                    new_beta = new_beta / static_cast<float>(1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = bsdf_ray;
            prev_bsdf_pdf   = srec.walk_specular ? 0.0 : srec.mis_pdf().value(bsdf_ray.direction());
            prev_surface_p  = rec.p;
            specular_bounce = srec.walk_specular;
            if (!srec.walk_specular) any_nonspecular = true;
            --bounces_left;
        } else {
            vec3   bsdf_dir = srec.pdf_ptr->generate();
            double pdf_b    = srec.pdf_ptr->value(bsdf_dir);
            if (pdf_b <= 0.0) break;

            ray    bsdf_ray(rec.p, bsdf_dir, current_ray.time());
            double f_pdf = scattering_pdf_at(bsdf_ray);
            if (f_pdf <= 0.0) break;

            if (srec.is_transmission) {
                bool crossed_boundary =
                    dot(bsdf_dir, rec.normal) *
                    dot(-unit_vector(current_ray.direction()), rec.normal) < 0.0;
                if (crossed_boundary)
                    eta_scale *= srec.eta * srec.eta;
            }

            SS new_beta = clamp_throughput(
                beta * albedo(rec.mat->scattering_attenuation(current_ray, rec, bsdf_ray, srec.attenuation)) * static_cast<float>(f_pdf / pdf_b));
            if (bounces_left < depth) {
                SS rr_beta = new_beta * static_cast<float>(eta_scale);
                float rr_max = rr_beta.MaxComponentValue();
                if (rr_max < 1.0f) {
                    double q = std::max(0.0, 1.0 - static_cast<double>(rr_max));
                    if (sampler.get() < q) break;
                    new_beta = new_beta / static_cast<float>(1.0 - q);
                }
            }
            beta            = new_beta;
            current_ray     = bsdf_ray;
            prev_bsdf_pdf   = pdf_b;
            prev_surface_p  = rec.p;
            specular_bounce = false;
            any_nonspecular = true;
            --bounces_left;
        }
    }

    // Deliberately NOT converted to RGB here - see this function's own
    // comment above. render()'s pixel loop filter-weight-averages this
    // XYZ triple across every sample in the pixel exactly like it does
    // ray_color()'s RGB, then converts to RGB (and clamps) once, after
    // averaging.
    XYZResult xyz = SampledSpectrumToXYZ<4>(L, swl, CIE_X, CIE_Y, CIE_Z);
    return color(static_cast<double>(xyz.x), static_cast<double>(xyz.y), static_cast<double>(xyz.z));
}
