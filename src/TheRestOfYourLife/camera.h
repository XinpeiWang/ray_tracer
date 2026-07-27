#ifndef CAMERA_H
#define CAMERA_H
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
#include <cmath>
#include "material.h"
#include "sky_light.h"
#include "punctual_light_objects.h"
#include "../shared/path_sampler.h"
#include "../shared/sobol_sampler.h"
#include "../shared/filter.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <thread>
#include <vector>
#include <atomic>
#include <sstream>
#include <mutex>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif


class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;               // Scene background color (used when sky==nullptr)
    shared_ptr<sky_light> sky;               // HDR env map (pbrt-v4 ImageInfiniteLight); nullptr = flat background
    shared_ptr<punctual_light_list> punct_lights; // pbrt-v4 PointLight/SpotLight/DistantLight (delta); nullptr = none

    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    int    image_height = 0;         // Rendered image height (set by initialize())
    point3 center;                   // Camera center (set by initialize())

    void render(const hittable& world, const hittable& lights) {
        initialize();

        // Try writing image to the user's Desktop. If that fails, fall back to
        // the current working directory, then to the system temp directory.
        std::string out_path;
        // Prefer OneDrive Desktop when available (common on Windows). Fall back to
        // %USERPROFILE%\OneDrive\Desktop, then %USERPROFILE%\Desktop, then HOME/Desktop.
        if (const char* od = std::getenv("OneDrive")) {
            out_path = std::string(od) + "\\Desktop\\image.ppm";
        } else if (const char* up = std::getenv("USERPROFILE")) {
            // If OneDrive folder exists under the user profile prefer it, otherwise use Desktop
            std::string od_candidate = std::string(up) + "\\OneDrive\\Desktop\\image.ppm";
            try {
                if (std::filesystem::exists(std::filesystem::path(std::string(up) + "\\OneDrive"))) {
                    out_path = od_candidate;
                } else {
                    out_path = std::string(up) + "\\Desktop\\image.ppm";
                }
            } catch (...) {
                out_path = std::string(up) + "\\Desktop\\image.ppm";
            }
        } else if (const char* home = std::getenv("HOME")) {
            out_path = std::string(home) + "/Desktop/image.ppm";
        } else {
            out_path = "image.ppm";
        }

        std::clog << "Attempting to write image to: " << out_path << std::endl;

        // Try to create parent directory if it doesn't exist (Desktop/ray_tracer)
        try {
            std::filesystem::path p(out_path);
            auto parent = p.parent_path();
            if (!parent.empty() && !std::filesystem::exists(parent)) {
                std::filesystem::create_directories(parent);
                std::clog << "Created directory: " << parent.string() << std::endl;
            }
        } catch (const std::exception& e) {
            std::clog << "Could not create Desktop subdirectory: " << e.what() << std::endl;
        }

        std::ofstream out(out_path, std::ios::out | std::ios::binary);
        if (!out) {
            // Fallback to current directory
            out_path = "image.ppm";
            std::clog << "Desktop write failed, falling back to: " << out_path << std::endl;
            out.open(out_path, std::ios::out | std::ios::binary);
        }

        if (!out) {
            // Fallback to TEMP
            if (const char* tmp = std::getenv("TEMP")) {
                out_path = std::string(tmp) + "\\image.ppm";
            } else if (const char* tmp2 = std::getenv("TMP")) {
                out_path = std::string(tmp2) + "\\image.ppm";
            }
            std::clog << "Attempting temp path: " << out_path << std::endl;
            out.open(out_path, std::ios::out | std::ios::binary);
        }

        if (!out) {
            std::cerr << "Failed to open any output file (tried Desktop subfolder, cwd, TEMP)" << std::endl;
            return;
        }

        std::clog << "Writing image to: " << out_path << std::endl;
        out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        // Multithreaded rendering: each worker renders scanlines into a buffer
        std::vector<std::string> scanlines(image_height);
        std::atomic<int> next_j(image_height - 1);
        std::atomic<int> completed_lines(0);
        std::mutex log_mutex;

        auto determine_thread_count = [&]() -> unsigned int {
            unsigned int hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 4;

            // Respect explicit override via env var RAY_TRACER_THREADS
            if (const char* env = std::getenv("RAY_TRACER_THREADS")) {
                std::string s(env);
                if (s == "auto") {
                    // fallthrough to auto-detect below
                } else {
                    try {
                        int v = std::stoi(s);
                        if (v > 0) return static_cast<unsigned int>(v);
                    } catch (...) {
                        // ignore parse errors
                    }
                }
            }

            // Auto-detect free cores by sampling system idle fraction (Windows only).
#ifdef _WIN32
            FILETIME idle1, kernel1, user1;
            FILETIME idle2, kernel2, user2;
            if (GetSystemTimes(&idle1, &kernel1, &user1)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                if (GetSystemTimes(&idle2, &kernel2, &user2)) {
                    auto toULL = [](const FILETIME &ft) -> unsigned long long {
                        ULARGE_INTEGER ui; ui.LowPart = ft.dwLowDateTime; ui.HighPart = ft.dwHighDateTime; return ui.QuadPart;
                    };
                    unsigned long long idleDiff = toULL(idle2) - toULL(idle1);
                    unsigned long long kernelDiff = toULL(kernel2) - toULL(kernel1);
                    unsigned long long userDiff = toULL(user2) - toULL(user1);
                    unsigned long long total = kernelDiff + userDiff;
                    if (total == 0) total = 1;
                    double busy = double(total - idleDiff) / double(total);
                    // estimate free cores = hw * (1 - busy)
                    int recommend = int(std::round(hw * (1.0 - busy)));
                    if (recommend < 1) recommend = 1;
                    if (recommend > (int)hw) recommend = hw;
                    return static_cast<unsigned int>(recommend);
                }
            }
#endif
            // Fallback: use all logical cores
            return hw;
        };

        unsigned int nthreads = determine_thread_count();
        std::clog << "Using " << nthreads << " threads for rendering" << std::endl;

        auto worker = [&](unsigned int tid) {
            std::ostringstream ss;
            while (true) {
                int j = next_j.fetch_sub(1);
                if (j < 0) break;

                // render scanline j
                ss.str(""); ss.clear();
                for (int i = 0; i < image_width; i++) {
                    color  weighted_color(0,0,0);
                    double weight_sum = 0.0;
                    // Mitchell-Netravali reconstruction filter (pbrt-v4 style).
                    // radius=0.5 keeps the footprint within one pixel so we don't
                    // need a film buffer for cross-pixel splatting (pbrt-v4 uses
                    // radius=2 with a full film; we approximate with within-pixel).
                    static const MitchellFilter filter(0.5, 1.0/3.0, 1.0/3.0);
                    for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                            for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                                // Sample index for Halton: unique per (s_i, s_j) stratum
                                    int sample_idx = s_j * sqrt_spp + s_i;
                                    // Compute sub-pixel offset once; use for both the ray
                                    // origin and the filter weight (pbrt-v4: FilterSample).
                                    vec3 offset = sample_square_stratified(s_i, s_j, sample_idx, i, j);
                                    ray r = get_ray(i, j, s_i, s_j, offset);
                                    SobolSampler ps(sample_idx, i, j);  // pbrt-v4 Sobol+FastOwen
                                color sample = ray_color(r, max_depth, world, lights, ps);
                                // NaN/Inf firefly guard (pbrt-v4 style)
                                if (std::isnan(sample.x()) || std::isnan(sample.y()) || std::isnan(sample.z()) ||
                                    std::isinf(sample.x()) || std::isinf(sample.y()) || std::isinf(sample.z()))
                                    sample = color(0, 0, 0);
                                // Mitchell-Netravali filter weight (pbrt-v4 filterWeight)
                                double w = filter.evaluate(offset.x(), offset.y());
                                weighted_color += w * sample;
                                weight_sum    += w;
                            }
                        }
                    // Normalize by filter weight sum -- mirrors pbrt-v4 film:
                    //   pixel.rgbSum / pixel.weightSum
                    color pixel_color = (weight_sum > 0.0)
                        ? weighted_color / weight_sum
                        : color(0, 0, 0);
                    write_color(ss, pixel_color);
                }

                scanlines[j] = ss.str();
                int done = ++completed_lines;
                if ((done % 10) == 0 || done == image_height) {
                    std::lock_guard<std::mutex> lg(log_mutex);
                    std::clog << "\rScanlines remaining: " << (image_height - done) << ' ' << std::flush;
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        for (unsigned int t = 0; t < nthreads; ++t)
            threads.emplace_back(worker, t);

        for (auto &th : threads) th.join();

        // Write buffered scanlines in order
        for (int j = 0; j < image_height; ++j) {
            out << scanlines[j];
        }

        std::clog << "\rDone.                 \n";
        out.close();
    }

  private:
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    int    sqrt_spp;             // Square root of number of samples per pixel
    double recip_sqrt_spp;       // 1 / sqrt_spp
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

  public:
    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        sqrt_spp = int(std::sqrt(samples_per_pixel));
        pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
        recip_sqrt_spp = 1.0 / sqrt_spp;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j, int s_i, int s_j, int sample_idx = 0, int px = 0, int py = 0) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j for stratified sample square s_i, s_j.
        // sample_idx + pixel coords drive Halton per-pixel decorrelation (pbrt-v4 pattern).
        auto offset = sample_square_stratified(s_i, s_j, sample_idx, px, py);
        return get_ray(i, j, s_i, s_j, offset);
    }

    // Overload accepting a pre-computed sub-pixel offset (avoids double Halton evaluation
    // when the caller already has the offset for filter weight computation).
    ray get_ray(int i, int j, int /*s_i*/, int /*s_j*/, const vec3& offset) const {
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double();

        return ray(ray_origin, ray_direction, ray_time);
    }

    vec3 sample_square_stratified(int s_i, int s_j, int sample_idx = 0,
                                   int pixel_x = 0, int pixel_y = 0) const {
        // Returns the vector to a random point in the square sub-pixel specified by grid
        // indices s_i and s_j, for an idealized unit square pixel [-.5,-.5] to [+.5,+.5].
        //
        // Jitter uses Halton low-discrepancy sequences (pbrt-v4 HaltonSampler pattern):
        //   x <- base-2 radical inverse, per-pixel decorrelated
        //   y <- base-3 radical inverse, per-pixel decorrelated
        // Pixel coordinates are mixed into the index so adjacent pixels use different
        // sub-sequences, avoiding the structured grid artifact from shared Halton offsets.
        unsigned int ui = (unsigned int)sample_idx;
        unsigned int ux = (unsigned int)pixel_x;
        unsigned int uy = (unsigned int)pixel_y;
        auto ox = (double)halton2(ui, ux, uy);
        auto oy = (double)halton3(ui, ux, uy);
        auto px = ((s_i + ox) * recip_sqrt_spp) - 0.5;
        auto py = ((s_j + oy) * recip_sqrt_spp) - 0.5;

        return vec3(px, py, 0);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    vec3 sample_disk(double radius) const {
        // Returns a random point in the unit (radius 0.5) disk centered at the origin.
        return radius * random_in_unit_disk();
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    // Multiple Importance Sampling: Balance heuristic weight calculation
    // Given PDFs from two sampling strategies, returns the weight for the first strategy
    static double mis_balance_heuristic(double pdf_a, double pdf_b) {
        // Balance heuristic: w_a = pdf_a / (pdf_a + pdf_b)
        // Handles edge cases: if pdf_a is 0, weight is 0; if both are 0, avoid NaN
        if (pdf_a <= 0.0) return 0.0;
        if (pdf_b <= 0.0) return 1.0;
        return pdf_a / (pdf_a + pdf_b);
    }

    // Power heuristic beta=2 -- delegates to shared PowerHeuristic (pbrt-v4 pattern)
    static double mis_power_heuristic(double pdf_a, double pdf_b) {
        return PowerHeuristic(pdf_a, pdf_b);
    }

    // ray_color -- iterative path integrator (pbrt-v4 PathIntegrator::Li style)
    //
    // Replaces the previous recursive implementation with a while loop
    // carrying explicit state:
    //   beta           -- path throughput (product of f/pdf along the path)
    //   L              -- accumulated radiance
    //   current_ray    -- the active ray, updated each bounce
    //   bounces_left   -- bounces remaining
    //   prev_bsdf_pdf  -- BSDF PDF of the ray that arrived here (0=camera/specular)
    //                     used to MIS-weight emitter hits (mirrors pbrt-v4 p_b)
    //   specular_bounce -- true after a delta-BxDF bounce; suppresses MIS on emitters
    template <typename Sampler>
    color ray_color(const ray& r, int depth, const hittable& world, const hittable& lights,
                    Sampler& sampler)
    const {
        color  L            = color(0, 0, 0);
        color  beta         = color(1, 1, 1);
        ray    current_ray  = r;
        int    bounces_left = depth;
        double prev_bsdf_pdf      = 0.0;
        double eta_scale          = 1.0;  // pbrt-v4: etaScale = product of Sqr(bs->eta) per transmission
        bool   specular_bounce    = true;
        bool   any_nonspecular    = false;  // pbrt-v4: anyNonSpecularBounces
        point3 prev_surface_p     = r.origin(); // pbrt-v4: prevIntrCtx shading point

        while (bounces_left > 0) {
            hit_record rec;

            // Miss -- query sky (HDR env map) or fall back to flat background.
            // Mirrors pbrt-v4: "Incorporate emission from infinite lights for escaped ray"
            if (!world.hit(current_ray, interval(0.001, infinity), rec)) {
                if (sky) {
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

            // No scatter (pure emitter / absorber).
            // Pass any_nonspecular as do_regularize so rough materials widen their GGX
            // lobe on a thread-local copy -- mirrors pbrt-v4 "Possibly regularize the BSDF".
            scatter_record srec;
            if (!rec.mat->scatter(current_ray, rec, srec, any_nonspecular))
                break;

            // Specular bounce: no NEE, update beta and advance ray
            if (srec.skip_pdf) {
                color new_beta = beta * srec.attenuation;
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
                // Update etaScale for transmission bounces (pbrt-v4: etaScale *= Sqr(bs->eta))
                if (srec.is_transmission)
                    eta_scale *= srec.eta * srec.eta;
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

            // Strategy A-1: NEE toward area lights (non-recursive shadow test)
            {
                vec3   light_dir = light_pdf.generate();
                double pdf_l     = light_pdf.value(light_dir);
                if (pdf_l > 0.0) {
                    ray    shadow_ray(rec.p, light_dir, current_ray.time());
                    double f_pdf = rec.mat->scattering_pdf(current_ray, rec, shadow_ray);
                    if (f_pdf > 0.0) {
                        double pdf_b_at_l = srec.pdf_ptr->value(light_dir);
                        double w_l        = mis_power_heuristic(pdf_l, pdf_b_at_l);
                        hit_record light_rec;
                        if (world.hit(shadow_ray, interval(0.001, infinity), light_rec)) {
                            color Le_d = light_rec.mat->emitted(
                                shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p);
                            if (Le_d.x() > 0 || Le_d.y() > 0 || Le_d.z() > 0)
                                L += beta * w_l * srec.attenuation * f_pdf * Le_d / pdf_l;
                        }
                    }
                }
            }

            // Strategy A-2: NEE toward sky (pbrt-v4 SampleLd for infinite lights)
            // Uses importance-sampled direction when HDR distribution is available,
            // falling back to uniform sphere for solid-color skies.
            if (sky) {
                SkyLiSample sky_smp = sky->sample_Le();
                vec3   sky_dir  = sky_smp.direction;
                double pdf_sky  = sky_smp.pdf;
                if (pdf_sky > 0.0) {
                    ray    sky_shadow(rec.p, sky_dir, current_ray.time());
                    double f_pdf = rec.mat->scattering_pdf(current_ray, rec, sky_shadow);
                    if (f_pdf > 0.0) {
                        double pdf_b_at_sky = srec.pdf_ptr->value(sky_dir);
                        double w_sky        = mis_power_heuristic(pdf_sky, pdf_b_at_sky);
                        hit_record sky_rec;
                        if (!world.hit(sky_shadow, interval(0.001, infinity), sky_rec)) {
                            color Le_sky = sky->Le(unit_vector(sky_dir));
                            L += beta * w_sky * srec.attenuation * f_pdf * Le_sky / pdf_sky;
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
                    double f_pdf = rec.mat->scattering_pdf(current_ray, rec,
                                                           ray(rec.p, ps.wi, current_ray.time()));
                    if (f_pdf <= 0.0) return;
                    hit_record shadow_rec;
                    interval shadow_t(0.001, ps.t_max - 0.001);
                    if (ps.t_max == infinity) shadow_t = interval(0.001, infinity);
                    if (!world.hit(ray(rec.p, ps.wi, current_ray.time()), shadow_t, shadow_rec)) {
                        // delta light: pdf=1, no MIS weight needed
                        L += beta * srec.attenuation * f_pdf * ps.Li;
                    }
                });
            }

            // Strategy B: BSDF sample becomes next path ray
            {
                vec3   bsdf_dir = srec.pdf_ptr->generate();
                double pdf_b    = srec.pdf_ptr->value(bsdf_dir);
                if (pdf_b <= 0.0) break;

                ray    bsdf_ray(rec.p, bsdf_dir, current_ray.time());
                double f_pdf = rec.mat->scattering_pdf(current_ray, rec, bsdf_ray);
                if (f_pdf <= 0.0) break;

                // Russian Roulette after first bounce
                color new_beta = beta * srec.attenuation * f_pdf / pdf_b;
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
};


#endif
