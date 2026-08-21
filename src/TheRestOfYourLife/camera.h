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
#include "shadow_ray.h"
#include "../shared/path_sampler.h"
#include "../shared/sobol_sampler.h"
#include "../shared/stratified_sampler.h"
#include "../shared/pmj02_sampler.h"
#include "../shared/halton_sampler.h"
#include "../shared/filter.h"
#include "../shared/cameras.h"
#include "../shared/surface_interaction.h"  // compute_differentials() for texture-filtering footprint
#include "../shared/exr_writer.h"
#include "thread_count.h"
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
#include <optional>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif


// Which of this codebase's ported pbrt-v4 sampler classes (src/shared/
// sobol_sampler.h, stratified_sampler.h, pmj02_sampler.h, halton_sampler.h)
// drives ray_color()'s random decisions this render. Sobol (this project's
// long-standing default, matching pbrt-v4's ZSobol-adjacent quality without
// the extra Morton-curve bookkeeping) stays the default so existing renders
// are pixel-identical unless --sampler is passed - see launcher_args.h's
// own --sampler flag and camera.h's render() dispatch switch for where this
// is actually consumed.
enum class SamplerKind {
    Sobol,
    ZSobol,
    PaddedSobol,
    Stratified,
    PMJ02BN,
    Halton,
};

// Parses a pbrt-v4 Sampler directive's/--sampler flag's type name
// ("sobol"/"zsobol"/"paddedsobol"/"stratified"/"pmj02bn"/"halton") into a
// SamplerKind. Unrecognized names (including pbrt-v4 names this project
// doesn't port, like "independent") fall back to Sobol - matching the
// existing Integrator-directive precedent of warning rather than failing
// the render (see SceneDescriptor::recommended_integrator's own comment).
inline bool sampler_kind_from_name(const std::string& name, SamplerKind& out) {
    if (name == "sobol")        { out = SamplerKind::Sobol;       return true; }
    if (name == "zsobol")       { out = SamplerKind::ZSobol;      return true; }
    if (name == "paddedsobol")  { out = SamplerKind::PaddedSobol; return true; }
    if (name == "stratified")   { out = SamplerKind::Stratified;  return true; }
    if (name == "pmj02bn")      { out = SamplerKind::PMJ02BN;     return true; }
    if (name == "halton")       { out = SamplerKind::Halton;      return true; }
    return false;
}

class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    // Flat multiplier on linear color, applied right before tone-mapping
    // (write_color()). Mirrors pbrt-v4's PixelSensor imagingRatio =
    // exposureTime * ISO / 100 (film.cpp) collapsed to a single scalar -
    // see launcher_args.h's --exposure flag. 1.0 (default) is a no-op.
    double exposure          = 1.0;
    // Which sampler drives ray_color()'s random decisions - see SamplerKind's
    // own comment. Default Sobol matches this project's pre-existing,
    // hardcoded behavior exactly.
    SamplerKind sampler_kind = SamplerKind::Sobol;
    // When set, render() writes a linear (pre-tonemap), full-float EXR
    // instead of the tonemapped/quantized PPM it writes by default - see
    // exr_writer.h. Set by cpu_interface.cpp when the caller's requested
    // output path ends in ".exr"; false (PPM) is the pre-existing default
    // and behavior for every other extension is unchanged.
    bool   exr_output = false;
    color  background;               // Scene background color (used when sky==nullptr)
    shared_ptr<sky_light> sky;               // HDR env map (pbrt-v4 ImageInfiniteLight); nullptr = flat background
    shared_ptr<punctual_light_list> punct_lights; // pbrt-v4 PointLight/SpotLight/DistantLight (delta); nullptr = none

    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    // Alternate camera models (pbrt-v4 cameras.h). When set, overrides default perspective.
    shared_ptr<OrthographicCamera<double>> alt_ortho_cam;
    shared_ptr<SphericalCamera<double>>    alt_spherical_cam;
    shared_ptr<RealisticCamera<double>>    alt_realistic_cam;

    int    image_height = 0;         // Rendered image height (set by initialize())
    point3 center;                   // Camera center (set by initialize())

    void render(const hittable& world, const hittable& lights) {
        initialize();

        // Try writing image to the user's Desktop. If that fails, fall back to
        // the current working directory, then to the system temp directory.
        // The filename itself (not just its directory) depends on exr_output -
        // see that field's own comment - so cpu_interface.cpp's copy-out step
        // (which mirrors this exact fallback chain to find what render()
        // actually wrote) can look for the same extension it requested.
        const std::string filename = exr_output ? "image.exr" : "image.ppm";
        std::string out_path;
        // Prefer OneDrive Desktop when available (common on Windows). Fall back to
        // %USERPROFILE%\OneDrive\Desktop, then %USERPROFILE%\Desktop, then HOME/Desktop.
        if (const char* od = std::getenv("OneDrive")) {
            out_path = std::string(od) + "\\Desktop\\" + filename;
        } else if (const char* up = std::getenv("USERPROFILE")) {
            // If OneDrive folder exists under the user profile prefer it, otherwise use Desktop
            std::string od_candidate = std::string(up) + "\\OneDrive\\Desktop\\" + filename;
            try {
                if (std::filesystem::exists(std::filesystem::path(std::string(up) + "\\OneDrive"))) {
                    out_path = od_candidate;
                } else {
                    out_path = std::string(up) + "\\Desktop\\" + filename;
                }
            } catch (...) {
                out_path = std::string(up) + "\\Desktop\\" + filename;
            }
        } else if (const char* home = std::getenv("HOME")) {
            out_path = std::string(home) + "/Desktop/" + filename;
        } else {
            out_path = filename;
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
            out_path = filename;
            std::clog << "Desktop write failed, falling back to: " << out_path << std::endl;
            out.open(out_path, std::ios::out | std::ios::binary);
        }

        if (!out) {
            // Fallback to TEMP
            if (const char* tmp = std::getenv("TEMP")) {
                out_path = std::string(tmp) + "\\" + filename;
            } else if (const char* tmp2 = std::getenv("TMP")) {
                out_path = std::string(tmp2) + "\\" + filename;
            }
            std::clog << "Attempting temp path: " << out_path << std::endl;
            out.open(out_path, std::ios::out | std::ios::binary);
        }

        if (!out) {
            std::cerr << "Failed to open any output file (tried Desktop subfolder, cwd, TEMP)" << std::endl;
            return;
        }

        std::clog << "Writing image to: " << out_path << std::endl;
        // exr_output writes through write_exr_image() below instead - `out`
        // above only exists to prove out_path is writable via the exact same
        // Desktop/cwd/TEMP fallback chain the PPM path already used, so it is
        // opened (and then simply closed unused) rather than duplicating that
        // fallback logic a second time for the EXR case.
        if (!exr_output)
            out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        // Multithreaded rendering: each worker renders scanlines into a buffer
        std::vector<std::string> scanlines(image_height);
        // Linear, pre-tonemap pixel buffer for exr_output - filled alongside
        // (not instead of) `scanlines` below, at zero extra cost when
        // exr_output is false (stays empty; every worker's write is guarded
        // on the same flag). Interleaved RGB, row-major, matching
        // write_exr_image()/SaveEXR's own expected layout exactly.
        std::vector<float> exr_pixels(
            exr_output ? static_cast<size_t>(image_width) * image_height * 3 : 0);
        std::atomic<int> next_j(image_height - 1);
        std::atomic<int> completed_lines(0);
        std::mutex log_mutex;

        // Auto-detection (when RAY_TRACER_THREADS isn't set to an explicit
        // value) samples system idle time over 200ms - see thread_count.h.
        // Video mode calls this once for the whole video and sets
        // RAY_TRACER_THREADS itself so this resolves instantly on every
        // frame instead of re-sampling per frame (main.cpp's video branch).
        unsigned int nthreads = determine_render_thread_count();
        std::clog << "Using " << nthreads << " threads for rendering" << std::endl;

        auto worker = [&](unsigned int tid) {
            std::ostringstream ss;
            // Construct the ONE stateful sampler this render actually asked
            // for (sampler_kind), once per worker thread rather than once
            // per ray - PMJ02BNSampler in particular precomputes a whole
            // per-pixel-tile table in its constructor, and ZSobol/
            // PaddedSobol/Stratified/Halton all carry per-(px,py,dim) state
            // reset via start_pixel_sample() rather than being cheap to
            // rebuild per ray the way SobolSampler is (see its own case
            // below, which still constructs fresh per ray - zero behavior
            // change for this project's pre-existing default). Every ctor
            // here seeds purely from (px, py, dim, a fixed global seed=0),
            // never from `tid`, so which thread renders a given pixel still
            // can't change that pixel's random sequence - same determinism
            // SobolSampler already had.
            std::optional<ZSobolSampler>      zsobol_sampler;
            std::optional<PaddedSobolSampler> padded_sobol_sampler;
            std::optional<StratifiedSampler>  stratified_sampler;
            std::optional<PMJ02BNSampler>     pmj02bn_sampler;
            std::optional<halton_sampler>     halton_smp;
            switch (sampler_kind) {
                case SamplerKind::ZSobol:
                    zsobol_sampler.emplace(samples_per_pixel, image_width, image_height);
                    break;
                case SamplerKind::PaddedSobol:
                    padded_sobol_sampler.emplace(samples_per_pixel);
                    break;
                case SamplerKind::Stratified:
                    stratified_sampler.emplace(sqrt_spp, sqrt_spp);
                    break;
                case SamplerKind::PMJ02BN:
                    pmj02bn_sampler.emplace(samples_per_pixel);
                    break;
                case SamplerKind::Halton:
                    halton_smp.emplace(samples_per_pixel, image_width, image_height);
                    break;
                case SamplerKind::Sobol:
                default:
                    break;  // SobolSampler needs no persistent per-thread state
            }
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
                                    double camera_weight = 1.0;
                                    ray r = get_ray(i, j, s_i, s_j, offset, &camera_weight);
                                    // Dispatch to whichever sampler this render asked for
                                    // (sampler_kind) - see SamplerKind's own comment and the
                                    // per-worker construction above. ray_color() is templated
                                    // and duck-typed on `Sampler&`, so each branch instantiates
                                    // its own copy; kept to a two-line construct+call per case.
                                    color sample;
                                    switch (sampler_kind) {
                                        case SamplerKind::ZSobol: {
                                            zsobol_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color(r, max_depth, world, lights, *zsobol_sampler);
                                            break;
                                        }
                                        case SamplerKind::PaddedSobol: {
                                            padded_sobol_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color(r, max_depth, world, lights, *padded_sobol_sampler);
                                            break;
                                        }
                                        case SamplerKind::Stratified: {
                                            stratified_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color(r, max_depth, world, lights, *stratified_sampler);
                                            break;
                                        }
                                        case SamplerKind::PMJ02BN: {
                                            pmj02bn_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color(r, max_depth, world, lights, *pmj02bn_sampler);
                                            break;
                                        }
                                        case SamplerKind::Halton: {
                                            halton_smp->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color(r, max_depth, world, lights, *halton_smp);
                                            break;
                                        }
                                        case SamplerKind::Sobol:
                                        default: {
                                            SobolSampler ps(sample_idx, i, j);  // pbrt-v4 Sobol+FastOwen
                                            sample = ray_color(r, max_depth, world, lights, ps);
                                            break;
                                        }
                                    }
                                // Apply the camera's exposure weight (pbrt-v4: L *= cameraRay->weight).
                                // Always 1.0 for pinhole/Ortho/Spherical; for RealisticCamera this is
                                // the cos^4(theta)/(pdf*LensRearZ^2) factor that converts the traced
                                // radiance into the correct measured exposure - see get_ray's comment.
                                sample = sample * camera_weight;
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
                    pixel_color = pixel_color * exposure;
                    if (exr_output) {
                        const size_t idx = (static_cast<size_t>(j) * image_width + i) * 3;
                        exr_pixels[idx + 0] = static_cast<float>(pixel_color.x());
                        exr_pixels[idx + 1] = static_cast<float>(pixel_color.y());
                        exr_pixels[idx + 2] = static_cast<float>(pixel_color.z());
                    } else {
                        write_color(ss, pixel_color);
                    }
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

        if (exr_output) {
            // `out` was only opened to validate out_path is writable (see
            // the comment where it was opened above) - close it unused and
            // let write_exr_image() create the real file at that same path.
            out.close();
            std::string exr_error;
            if (write_exr_image(out_path, exr_pixels.data(), image_width, image_height, exr_error)) {
                std::clog << "\rWrote EXR: " << out_path << "\n";
            } else {
                std::cerr << "Failed to write EXR '" << out_path << "': " << exr_error << std::endl;
            }
        } else {
            // Write buffered scanlines in order
            for (int j = 0; j < image_height; ++j) {
                out << scanlines[j];
            }
            out.close();
        }

        std::clog << "\rDone.                 \n";
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
    // out_camera_weight, if non-null, receives the camera's exposure weight for this
    // sample (pbrt-v4 CameraRay::weight - always 1 for Ortho/Spherical, but for
    // RealisticCamera it's cos^4(theta)/(pdf*LensRearZ^2) and MUST be multiplied into
    // the returned radiance, exactly like pbrt-v4's integrators do (L *= cameraRay->weight)
    // - omitting it silently under-exposes every RealisticCamera render.
    ray get_ray(int i, int j, int /*s_i*/, int /*s_j*/, const vec3& offset,
                double* out_camera_weight = nullptr) const {
        if (out_camera_weight) *out_camera_weight = 1.0;
        // If an alternate camera model is set, delegate ray generation to it.
        if (alt_ortho_cam || alt_spherical_cam || alt_realistic_cam) {
            CameraSample<double> cs;
            cs.pFilm_x = i + offset.x() + 0.5;
            cs.pFilm_y = j + offset.y() + 0.5;
            cs.pLens_u = random_double();
            cs.pLens_v = random_double();
            cs.time    = random_double();
            CameraRayResult<double> res;
            if (alt_ortho_cam)     res = alt_ortho_cam->generate_ray(cs);
            else if (alt_spherical_cam) res = alt_spherical_cam->generate_ray(cs);
            else {
                // Unlike Ortho/Spherical, RealisticCamera::generate_ray() expects
                // physical film-plane coordinates (metres), not raster pixel
                // coordinates - convert first (see raster_to_film's comment).
                double film_x, film_y;
                alt_realistic_cam->raster_to_film(cs.pFilm_x, cs.pFilm_y,
                    image_width, image_height, film_x, film_y);
                cs.pFilm_x = film_x;
                cs.pFilm_y = film_y;
                res = alt_realistic_cam->generate_ray(cs);
            }
            if (out_camera_weight) *out_camera_weight = res.weight;
            point3 ro(res.origin.x, res.origin.y, res.origin.z);
            vec3   rd(res.direction.x, res.direction.y, res.direction.z);
            return ray(ro, rd, cs.time);
        }

        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double();

        ray r(ray_origin, ray_direction, ray_time);

        // Ray differentials (pbrt-v4 GenerateRayDifferential): two auxiliary
        // rays offset by one pixel in x/y, reusing the SAME lens/origin
        // sample as the primary ray - used downstream (ray_color()) to
        // estimate a texture lookup's footprint for EWA/mipmap filtering
        // (see ray.h's own comment). Only the primary pixel-sample path
        // generates these; the alt-camera branch above returns early
        // without them (has_differentials() stays false there).
        auto rx_sample = pixel_sample + pixel_delta_u;
        auto ry_sample = pixel_sample + pixel_delta_v;
        r.set_differentials(ray_origin, rx_sample - ray_origin,
                             ray_origin, ry_sample - ray_origin);

        return r;
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

    // sample_bssrdf_exit -- BSSRDF probe/exit-point search for real subsurface
    // scattering (pbrt-v4 SubsurfaceMaterial / TabulatedBSSRDF).
    //
    // Called from ray_color() right after rec.mat->scatter() returns a
    // specular TRANSMISSION into a material whose as_subsurface() is
    // non-null (class `subsurface`, material_pbrt.h). On success this
    // reassigns `rec` in place to the found exit point (with rec.mat swapped
    // to the subsurface material's cached normalized_fresnel exit BSDF --
    // pbrt-v4's Sw) and refills `srec` by calling that exit material's own
    // scatter(), so the caller's ordinary non-specular NEE + BSDF-sample code
    // can continue completely unmodified from there, exactly as pbrt-v4
    // treats the exit bounce as a normal (non-specular) BSDF once found.
    //
    // Mirrors pbrt-v4 VolPathIntegrator::Li's "Account for attenuated
    // subsurface scattering" block (cpu/integrators.cpp ~1187-1255) and this
    // codebase's own already-tested but previously unwired port of the same
    // logic, src/shared/vol_path.h::VolPathLi (see its BSSRDF-branch comment
    // and tests/unit/bssrdf_vol_path_tests.cpp) -- this is that same
    // probe-walk-then-reservoir-sample algorithm, reimplemented against the
    // concrete hittable_list/material types instead of vol_path.h's
    // duck-typed Scene concept, since wiring the whole templated integrator
    // in just for this would mean re-deriving NEE/MIS/RR this file already
    // has working.
    //
    // Probe axis: matches pbrt-v4's SampleSp/PDF_Sp exactly -- one of 3
    // mutually-orthonormal axes anchored at the entry point (shading normal,
    // weight 0.5; two tangent directions, weight 0.25 each) is chosen for
    // the actual probe walk, but PDF_Sp always returns the *combined*
    // (one-sample MIS, balance heuristic) pdf across all 3 axes regardless
    // of which one produced the sample. This is what lets the probe still
    // find a good exit point where the surface is nearly edge-on to the
    // normal (thin fins/creases, e.g. the SSS dragon meshes' horns/spikes) --
    // a normal-only probe systematically misses those, and even when it
    // does hit, weighting by only the normal axis's pdf leaves the estimator
    // high-variance on such geometry (a documented, now-superseded
    // simplification of this function's earlier single-axis version).
    //
    // DOCUMENTED SIMPLIFICATION vs. pbrt-v4's TabulatedBSSRDF::SampleSp:
    //   Channel selection: pbrt-v4 is spectral (hero-wavelength sampling
    //      already randomizes which physical wavelength is "channel 0", so
    //      its SampleSr always reads channel 0). This renderer is plain RGB,
    //      so a channel is instead picked uniformly among R/G/B for the
    //      IMPORTANCE-SAMPLING radius only; Sp itself is evaluated across
    //      all 3 channels at the found exit distance (matching pbrt-v4's
    //      Sp(pi) = Sr(Distance(po,pi))), and the pdf is the average of all
    //      3 channels' combined-axis PDF_Sr at that distance -- an unbiased
    //      MC estimator for the same reason pbrt-v3's original
    //      (non-hero-wavelength) BSSRDF::Sample_S channel pick was.
    bool sample_bssrdf_exit(hit_record& rec, scatter_record& srec,
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

            // No scatter (pure emitter / absorber).
            // Pass any_nonspecular as do_regularize so rough materials widen their GGX
            // lobe on a thread-local copy -- mirrors pbrt-v4 "Possibly regularize the BSDF".
            scatter_record srec;
            if (!rec.mat->scatter(current_ray, rec, srec, any_nonspecular))
                break;

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
                        color trans;
                        if (shadow_ray_hit(world, shadow_ray, light_rec, infinity, &trans)) {
                            color Le_d = light_rec.mat->emitted(
                                shadow_ray, light_rec, light_rec.u, light_rec.v, light_rec.p);
                            if (Le_d.x() > 0 || Le_d.y() > 0 || Le_d.z() > 0) {
                                color atten = rec.mat->scattering_attenuation(rec, shadow_ray, srec.attenuation);
                                L += beta * w_l * atten * trans * f_pdf * Le_d / pdf_l;
                            }
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
                        color trans;
                        if (!shadow_ray_hit(world, sky_shadow, sky_rec, infinity, &trans)) {
                            color Le_sky = sky->Le(unit_vector(sky_dir));
                            color atten = rec.mat->scattering_attenuation(rec, sky_shadow, srec.attenuation);
                            L += beta * w_sky * atten * trans * f_pdf * Le_sky / pdf_sky;
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
                    if (!shadow_ray_hit(world, punct_ray, shadow_rec, shadow_t_max, &trans)) {
                        // delta light: pdf=1, no MIS weight needed
                        color atten = rec.mat->scattering_attenuation(rec, punct_ray, srec.attenuation);
                        L += beta * atten * trans * f_pdf * ps.Li;
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
                color new_beta = clamp_throughput(beta * srec.attenuation * f_pdf / pdf_b);
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
