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
#include "material.h"
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
#include <windows.h>
#endif


class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;               // Scene background color

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
                    color pixel_color(0,0,0);
                    for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                            for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                                // Sample index for Halton: unique per (s_i, s_j) stratum
                                    int sample_idx = s_j * sqrt_spp + s_i;
                                    ray r = get_ray(i, j, s_i, s_j, sample_idx, i, j);
                                pixel_color += ray_color(r, max_depth, world, lights);
                            }
                        }
                    write_color(ss, pixel_samples_scale * pixel_color);
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

    // ray_color -- pbrt-v4 PathIntegrator pattern
    //
    // prev_bsdf_pdf: the BSDF PDF that generated ray `r` on the previous
    // bounce (0 for the camera ray).  Used to compute the MIS weight when
    // this ray hits an emissive surface (area light).  Mirrors pbrt-v4's
    // p_b / w_l pattern in PathIntegrator::Li().
    color ray_color(const ray& r, int depth, const hittable& world, const hittable& lights,
                    const color& throughput    = color(1.0, 1.0, 1.0),
                    double       prev_bsdf_pdf = 0.0)
    const {
        // Terminate path at max depth.
        if (depth <= 0)
            return color(0, 0, 0);

        hit_record rec;

        // Miss — return background.
        if (!world.hit(r, interval(0.001, infinity), rec))
            return background;

        // ----------------------------------------------------------------
        // Emission at this surface.
        // pbrt-v4: if we arrived here via a specular bounce (prev_bsdf_pdf==0
        // signals camera ray or specular) we take the full emission.
        // If we arrived via a BSDF sample that was part of a MIS pair, we
        // apply the power-heuristic weight w_b so we don't double-count with
        // the NEE contribution from the previous bounce.
        // ----------------------------------------------------------------
        color Le = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);
        color color_from_emission = color(0, 0, 0);
        if (Le.x() > 0 || Le.y() > 0 || Le.z() > 0) {
            if (prev_bsdf_pdf <= 0.0) {
                // Camera ray or specular: take full emission.
                color_from_emission = Le;
            } else {
                // Non-specular BSDF bounce: apply MIS weight against NEE.
                hittable_pdf light_pdf_for_mis(lights, rec.p);
                double pdf_light = light_pdf_for_mis.value(r.direction());
                double w_b = mis_power_heuristic(prev_bsdf_pdf, pdf_light);
                color_from_emission = w_b * Le;
            }
        }

        scatter_record srec;
        if (!rec.mat->scatter(r, rec, srec))
            return color_from_emission;

        // ----------------------------------------------------------------
        // Specular bounce (delta BxDF) -- no direct light sampling possible.
        // Mirrors pbrt-v4 specularBounce path: single recursive call, no NEE.
        // ----------------------------------------------------------------
        if (srec.skip_pdf) {
            color new_throughput = throughput * srec.attenuation;
            double rr_scale = 1.0;

            // Russian Roulette after bounce > 1 (pbrt-v4 pattern).
            if ((max_depth - depth) > 1) {
                double rr_max = (std::max)(new_throughput.x(),
                                (std::max)(new_throughput.y(), new_throughput.z()));
                if (rr_max < 1.0) {
                    double q = (std::max)(0.0, 1.0 - rr_max);
                    if (random_double() < q) return color_from_emission;
                    rr_scale = 1.0 / (1.0 - q);
                    new_throughput = new_throughput * rr_scale;
                }
            }

            // prev_bsdf_pdf = 0 signals specular to the next bounce
            // (full emission, no MIS weight needed).
            return color_from_emission +
                   srec.attenuation * rr_scale *
                   ray_color(srec.skip_pdf_ray, depth - 1, world, lights,
                             new_throughput, /*prev_bsdf_pdf=*/0.0);
        }

        // ----------------------------------------------------------------
        // Non-specular surface: NEE (direct light) + BSDF continuation.
        // Mirrors pbrt-v4 PathIntegrator::Li():
        //   L += beta * SampleLd(...)          <- NEE, non-recursive
        //   ray = SpawnRay(bsdf_sample)        <- single BSDF recursive call
        // ----------------------------------------------------------------
        hittable_pdf light_pdf(lights, rec.p);

        // --- Strategy A: NEE / direct-light sample (non-recursive) ---
        // Sample a point on a light, test visibility, evaluate f*Le*w/pdf.
        color nee_contribution = color(0, 0, 0);
        {
            vec3 light_dir = light_pdf.generate();
            double pdf_l   = light_pdf.value(light_dir);

            if (pdf_l > 0.0) {
                ray shadow_ray(rec.p, light_dir, r.time());
                double f_pdf = rec.mat->scattering_pdf(r, rec, shadow_ray);

                if (f_pdf > 0.0) {
                    // MIS weight: light sample vs BSDF sample.
                    double pdf_b_at_l = srec.pdf_ptr->value(light_dir);
                    double w_l = mis_power_heuristic(pdf_l, pdf_b_at_l);

                    // Evaluate Le at the light by tracing the shadow ray.
                    // We only want the emitted radiance; if hit returns false
                    // (escaped scene) or no emission, contribution is zero.
                    hit_record light_rec;
                    color Le_direct = color(0, 0, 0);
                    if (world.hit(shadow_ray, interval(0.001, infinity), light_rec)) {
                        // Evaluate emission on the light surface.
                        Le_direct = light_rec.mat->emitted(
                            shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p);
                    }

                    if (Le_direct.x() > 0 || Le_direct.y() > 0 || Le_direct.z() > 0) {
                        nee_contribution =
                            w_l * srec.attenuation * f_pdf * Le_direct / pdf_l;
                    }
                }
            }
        }

        // --- Strategy B: BSDF sample (single recursive call) ---
        // Sample outgoing direction from BSDF, recurse.  The MIS weight for
        // any emitter hit by this ray is applied inside the recursive call
        // via prev_bsdf_pdf.
        color bsdf_contribution = color(0, 0, 0);
        {
            vec3 bsdf_dir = srec.pdf_ptr->generate();
            double pdf_b  = srec.pdf_ptr->value(bsdf_dir);

            if (pdf_b > 0.0) {
                ray bsdf_ray(rec.p, bsdf_dir, r.time());
                double f_pdf = rec.mat->scattering_pdf(r, rec, bsdf_ray);

                if (f_pdf > 0.0) {
                    // Russian Roulette: based on throughput after this bounce.
                    color next_throughput = throughput * srec.attenuation;
                    double rr_scale = 1.0;
                    if ((max_depth - depth) > 1) {
                        double rr_max = (std::max)(next_throughput.x(),
                                        (std::max)(next_throughput.y(), next_throughput.z()));
                        if (rr_max < 1.0) {
                            double q = (std::max)(0.0, 1.0 - rr_max);
                            if (random_double() < q) {
                                // Path terminated by RR — only return emission + NEE.
                                return color_from_emission + nee_contribution;
                            }
                            rr_scale = 1.0 / (1.0 - q);
                            next_throughput = next_throughput * rr_scale;
                        }
                    }

                    color incoming = ray_color(bsdf_ray, depth - 1, world, lights,
                                               next_throughput, /*prev_bsdf_pdf=*/pdf_b);

                    bsdf_contribution =
                        rr_scale * srec.attenuation * f_pdf * incoming / pdf_b;
                }
            }
        }

        return color_from_emission + nee_contribution + bsdf_contribution;
    }
};


#endif
