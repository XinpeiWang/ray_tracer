#pragma once
// camera_render.h -- camera::render(), initialize() and get_ray(): the CPU render loop, set-up and primary-ray generation, defined out of class here
// (declarations stay in camera.h). A pure move; included by camera.h after the class.

inline bool camera::render(const hittable& world, const hittable& lights,
    const std::string& output_path)
{
    initialize();

    // A prior version of this function instead always wrote to the
    // user's Desktop and relied on cpu_interface.cpp to guess that
    // location and copy the file out to the real destination afterward
    // - fragile (the guess could disagree with this function's own
    // fallback chain) and a needless side effect (a stray Desktop file
    // on every render).
    const std::string filename = exr_output ? "image.exr" : "image.ppm";
    std::string out_path = output_path.empty() ? filename : output_path;

    std::clog << "Attempting to write image to: " << out_path << std::endl;

    // Try to create the parent directory if it doesn't exist.
    try {
        std::filesystem::path p(out_path);
        auto parent = p.parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent)) {
            std::filesystem::create_directories(parent);
            std::clog << "Created directory: " << parent.string() << std::endl;
        }
    } catch (const std::exception& e) {
        std::clog << "Could not create output directory: " << e.what() << std::endl;
    }

    std::ofstream out(out_path, std::ios::out | std::ios::binary);
    if (!out) {
        // Fallback to current directory, then TEMP - same last-resort
        // chain this function always had, now reached only if the
        // caller-requested path itself isn't writable (a real error
        // condition), not as this function's own everyday first choice.
        out_path = filename;
        std::clog << "Requested path not writable, falling back to: " << out_path << std::endl;
        out.open(out_path, std::ios::out | std::ios::binary);
    }

    if (!out) {
        // TEMP/TMP are the Windows convention, TMPDIR is POSIX's - '/' as
        // separator works for iostream file I/O on both platforms, so no
        // need to branch on it too.
        const char* tmp = std::getenv("TEMP");
        if (!tmp) tmp = std::getenv("TMP");
        if (!tmp) tmp = std::getenv("TMPDIR");
        if (tmp) {
            out_path = std::string(tmp) + "/" + filename;
        }
        std::clog << "Attempting temp path: " << out_path << std::endl;
        out.open(out_path, std::ios::out | std::ios::binary);
    }

    if (!out) {
        std::cerr << "Failed to open any output file (tried the requested path, cwd, TEMP)" << std::endl;
        return false;
    }

    std::clog << "Writing image to: " << out_path << std::endl;
    // exr_output writes through write_exr_image() below instead - `out`
    // above only exists to prove out_path is writable via the exact same
    // requested-path/cwd/TEMP fallback chain the PPM path already used,
    // so it is opened (and then simply closed unused) rather than
    // duplicating that fallback logic a second time for the EXR case.
    if (!exr_output)
        out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Multithreaded rendering: each worker renders scanlines into a buffer.
    // Empty (0 elements) when exr_output, matching exr_pixels' own
    // conditional sizing below - the PPM-text `scanlines[j] = ss.str()`
    // write further down is skipped in that mode too, so this buffer
    // does zero work either way, not just zero allocation.
    std::vector<std::string> scanlines(exr_output ? 0 : image_height);
    // --time-limit: explicit "does scanlines[j] hold a real (if
    // possibly partial) rendered row yet" flag for the post-join
    // backfill pass below, instead of that pass inferring it from
    // whether scanlines[j] happens to be an empty string - a
    // std::string's emptiness is otherwise a coincidental property of
    // write_color() always emitting non-empty text for a positive
    // image_width, not a deliberate signal. char, not bool, to avoid
    // std::vector<bool>'s bit-packed proxy-reference surprises under
    // concurrent writes from different threads (each writing its own
    // distinct index, so no actual data race, but proxy references are
    // worth avoiding here regardless). Same conditional sizing as
    // `scanlines` above - unused in exr_output mode.
    std::vector<char> row_rendered(exr_output ? 0 : image_height, 0);
    // Linear, pre-tonemap pixel buffer for exr_output - filled alongside
    // (not instead of) `scanlines` above, at zero extra cost when
    // exr_output is false (stays empty; every worker's write is guarded
    // on the same flag). Interleaved RGB, row-major, matching
    // write_exr_image()/SaveEXR's own expected layout exactly.
    std::vector<float> exr_pixels(
        exr_output ? static_cast<size_t>(image_width) * image_height * 3 : 0);
    std::atomic<int> next_j(image_height - 1);
    std::atomic<int> completed_lines(0);
    std::mutex log_mutex;
    // --time-limit: see camera::time_limit_seconds's own comment.
    // Captured once, before any worker starts, so every thread measures
    // elapsed time against the same origin regardless of which one
    // actually reads it first.
    const auto render_start_time = std::chrono::steady_clock::now();
    // Shared by every worker thread's own time-limit checks below AND
    // the post-join backfill pass further down - declared once here
    // (rather than once per worker, or hand-copied again in the
    // backfill pass) so there's exactly one implementation of "how much
    // time has elapsed" and "write black pixels for a column range" to
    // keep correct.
    auto elapsed_seconds = [&]() -> double {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - render_start_time).count();
    };
    auto append_black_columns = [&](std::ostream &out_stream, int from_col) {
        for (int i = from_col; i < image_width; i++) write_color(out_stream, color(0, 0, 0), tone_map);
    };

    // Auto-detection (when RAY_TRACER_THREADS isn't set to an explicit
    // value) samples system idle time over 200ms - see thread_count.h.
    // Video mode calls this once for the whole video and sets
    // RAY_TRACER_THREADS itself so this resolves instantly on every
    // frame instead of re-sampling per frame (main.cpp's video branch).
    unsigned int nthreads = determine_render_thread_count();
    std::clog << "Using " << nthreads << " threads for rendering" << std::endl;

    // --seed: see camera::seed's own comment above, and the private
    // reseed_render_rng() method's own comment (below, in this same
    // class) for why thread_rng() is reseeded PER SCANLINE below (in
    // the worker loop) rather than once here -
    // scanline-to-worker assignment is a work-stealing race, not
    // reproducible run to run, so seeding by worker identity doesn't
    // give a reproducible render the way seeding by scanline index does.
    // Same "explicit value or today's exact literal 0" shape as
    // regularize/max_component_value above, computed once rather than
    // at each sampler-construction call site below.
    const int alt_sampler_seed = (seed >= 0) ? static_cast<int>(seed) : 0;

    auto worker = [&]() {
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
        // here seeds purely from (px, py, dim, a fixed global seed=0
        // unless --seed overrides it via alt_sampler_seed above), never
        // from which thread happens to run this worker, so which
        // thread renders a given pixel still can't change that pixel's
        // random sequence - same determinism SobolSampler already had.
        std::optional<ZSobolSampler>      zsobol_sampler;
        std::optional<PaddedSobolSampler> padded_sobol_sampler;
        std::optional<StratifiedSampler>  stratified_sampler;
        std::optional<PMJ02BNSampler>     pmj02bn_sampler;
        std::optional<halton_sampler>     halton_smp;
        switch (sampler_kind) {
            case SamplerKind::ZSobol:
                zsobol_sampler.emplace(samples_per_pixel, image_width, image_height, alt_sampler_seed);
                break;
            case SamplerKind::PaddedSobol:
                padded_sobol_sampler.emplace(samples_per_pixel, alt_sampler_seed);
                break;
            case SamplerKind::Stratified:
                stratified_sampler.emplace(sqrt_spp, sqrt_spp, true, alt_sampler_seed);
                break;
            case SamplerKind::PMJ02BN:
                pmj02bn_sampler.emplace(samples_per_pixel, alt_sampler_seed);
                break;
            case SamplerKind::Halton:
                halton_smp.emplace(samples_per_pixel, image_width, image_height,
                                    HaltonRandomize::PermuteDigits, static_cast<uint32_t>(alt_sampler_seed));
                break;
            case SamplerKind::Sobol:
            default:
                break;  // SobolSampler needs no persistent per-thread state
        }
        while (true) {
            int j = next_j.fetch_sub(1);
            if (j < 0) break;

            // --time-limit: stop CLAIMING new scanlines once the
            // deadline has passed - see camera::time_limit_seconds's
            // own comment. A second, finer-grained check inside the
            // per-pixel loop below can also abort MID-scanline - a
            // single scanline at very high spp can itself take several
            // seconds, so checking only here (once per whole scanline)
            // would let the render overshoot a short deadline by a
            // whole scanline's duration.
            //
            // No black-fill needed here (unlike the mid-scanline and
            // never-claimed cases below): `j` is already claimed (the
            // fetch_sub above already happened) so no other thread will
            // ever see it either way, and this thread does nothing else
            // with `scanlines[j]` or `completed_lines` before breaking -
            // the post-join backfill pass (this function's own comment
            // on it) already fills ANY row left empty, exactly the state
            // this row is already in. A bare `break` here produces
            // byte-identical final output to filling it inline, with
            // one fewer copy of the black-row-writing logic to keep in
            // sync.
            if (time_limit_seconds > 0.0 && elapsed_seconds() >= time_limit_seconds) break;

            // --seed: reseed thread_rng() fresh for THIS scanline,
            // keyed on the scanline index j rather than which worker
            // happens to be running it - see the private
            // reseed_render_rng() method's own comment (below, in this
            // same class) for why that distinction matters. No-op (and
            // no measurable cost beyond one branch) when --seed wasn't
            // requested.
            if (seed >= 0) reseed_render_rng(seed, j);

            // render scanline j
            ss.str(""); ss.clear();
            // --time-limit: whether this row was cut short mid-render,
            // and how many of its columns actually got a real sample
            // before that happened (image_width = every column, i.e.
            // "not cut short", unless the per-pixel check below fires).
            // Used after the loop to fill the remaining columns black
            // via append_black_columns().
            bool time_limit_hit = false;
            int columns_rendered = image_width;
            for (int i = 0; i < image_width; i++) {
                // --time-limit: the finer-grained, per-pixel half of
                // the scanline-claim check above - lets a long scanline
                // (high spp) abort partway through instead of only ever
                // being checked once per whole row. Columns already
                // rendered (0..i-1) keep their real result;
                // i..image_width-1 are filled black after this loop.
                // Only actually reads the clock every 32nd column - a
                // steady_clock::now() syscall on every single pixel is
                // wasted precision (nobody needs sub-32-pixel deadline
                // granularity) for real per-pixel cost across every
                // thread; a 31-pixel-wide worst-case overshoot on the
                // deadline is immaterial next to a whole scanline's own
                // overshoot bound that this check already accepts.
                if (time_limit_seconds > 0.0 && (i % 32) == 0 && elapsed_seconds() >= time_limit_seconds) {
                    time_limit_hit = true;
                    columns_rendered = i;
                    break;
                }
                color  weighted_color(0,0,0);
                double weight_sum = 0.0;
                // --adaptive: tracks this pixel's running luminance
                // estimate across samples so far - see camera::
                // adaptive_sampling's own comment and
                // pixel_convergence::has_converged(). Unused (and free)
                // when adaptive_sampling is off.
                VarianceEstimator<double> luminance_estimator;
                // Film "cropwindow"/"pixelbounds" (crop_x0/x1/y0/y1,
                // resolved in initialize()): a pixel outside the crop
                // rectangle is left at weight_sum=0, which the existing
                // normalization below already turns into black - so
                // skipping the whole sampling loop here is both the
                // compute-saving and the correctness fix in one place.
                const bool in_crop = i >= crop_x0 && i < crop_x1 && j >= crop_y0 && j < crop_y1;
                // --adaptive: the minimum sample count taken before
                // ever considering an early stop. Capped at a fixed 32
                // rather than always waiting for 2 full stratified rows
                // (2*sqrt_spp) - at low/moderate sqrt_spp the two agree,
                // but for a high-spp render (e.g. sqrt_spp=22 at
                // spp=500) an uncapped 2*sqrt_spp=44-sample floor would
                // force extra work on exactly the large-spp renders
                // adaptive sampling should help the most, for no
                // additional statistical benefit past the point a
                // Welford variance ESTIMATE is already trustworthy.
                const int min_samples_before_check = std::min(2 * sqrt_spp, 32);
                if (in_crop)
                for (int row_idx = 0; row_idx < sqrt_spp; row_idx++) {
                        // Visits stratified rows in row_visit_order(),
                        // NOT raster order 0,1,2,... - see that
                        // function's own comment (adaptive_sampling.h)
                        // for why: a raster-order prefix would confine
                        // an early-stopped pixel to only ever sampling
                        // one contiguous, spatially-lopsided side of
                        // the reconstruction filter's footprint instead
                        // of a representative (if smaller) spread of
                        // it. Has zero effect on a non-adaptive or
                        // never-converging render - see that comment's
                        // own "commutative sum" reasoning.
                        const int s_j = adaptive_row_order[row_idx];
                        if (adaptive_sampling &&
                            luminance_estimator.Count() >= min_samples_before_check &&
                            pixel_convergence::has_converged(luminance_estimator, adaptive_threshold,
                                                              1e-4, exposure))
                            break;
                        for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                            // Sample index for Halton: unique per (s_i, s_j) stratum
                                int sample_idx = s_j * sqrt_spp + s_i;
                                // Draw this stratum's [0,1)^2 sample, then CDF-invert it
                                // through the real requested filter (filterSampler, built
                                // once per worker thread above) - the returned position
                                // can legitimately land outside [-0.5,0.5] for any filter
                                // wider than one pixel, reaching a neighboring pixel's
                                // film-plane area via get_ray()'s own offset-onto-pixel-
                                // index math (pbrt-v4: FilterSample). fs.weight is this
                                // sample's already-correct reconstruction weight
                                // (f(p)/pdf(p) - filter_sampler.h's own comment).
                                vec3 u = sample_unit_square_stratified(s_i, s_j, sample_idx, i, j);
                                const FilterSample<double> fs = filterSampler_->sample(u.x(), u.y());
                                vec3 offset(fs.p_x, fs.p_y, 0);
                                double camera_weight = 1.0;
                                ray r = get_ray(i, j, s_i, s_j, offset, &camera_weight);
                                // Dispatch to whichever sampler this render asked for
                                // (sampler_kind) - see SamplerKind's own comment and the
                                // per-worker construction above. ray_color() is templated
                                // and duck-typed on `Sampler&`, so each branch instantiates
                                // its own copy; kept to a two-line construct+call per case.
                                color sample;
                                // spectral mirrors the switch below verbatim, calling
                                // ray_color_spectral() instead of ray_color() - see that
                                // field's own comment (camera::spectral). Kept as a
                                // separate switch (not a branch inside each case) so the
                                // default RGB path below is byte-for-byte unchanged for
                                // anyone not passing --spectral.
                                if (spectral) {
                                    switch (sampler_kind) {
                                        case SamplerKind::ZSobol: {
                                            zsobol_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color_spectral(r, max_depth, world, lights, *zsobol_sampler);
                                            break;
                                        }
                                        case SamplerKind::PaddedSobol: {
                                            padded_sobol_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color_spectral(r, max_depth, world, lights, *padded_sobol_sampler);
                                            break;
                                        }
                                        case SamplerKind::Stratified: {
                                            stratified_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color_spectral(r, max_depth, world, lights, *stratified_sampler);
                                            break;
                                        }
                                        case SamplerKind::PMJ02BN: {
                                            pmj02bn_sampler->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color_spectral(r, max_depth, world, lights, *pmj02bn_sampler);
                                            break;
                                        }
                                        case SamplerKind::Halton: {
                                            halton_smp->start_pixel_sample(i, j, sample_idx);
                                            sample = ray_color_spectral(r, max_depth, world, lights, *halton_smp);
                                            break;
                                        }
                                        case SamplerKind::Independent: {
                                            IndependentSampler ps(sample_idx, i, j);  // pbrt-v4 IndependentSampler
                                            sample = ray_color_spectral(r, max_depth, world, lights, ps);
                                            break;
                                        }
                                        case SamplerKind::Sobol:
                                        default: {
                                            SobolSampler ps(sample_idx, i, j);  // pbrt-v4 Sobol+FastOwen
                                            sample = ray_color_spectral(r, max_depth, world, lights, ps);
                                            break;
                                        }
                                    }
                                } else {
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
                                    case SamplerKind::Independent: {
                                        IndependentSampler ps(sample_idx, i, j);  // pbrt-v4 IndependentSampler
                                        sample = ray_color(r, max_depth, world, lights, ps);
                                        break;
                                    }
                                    case SamplerKind::Sobol:
                                    default: {
                                        SobolSampler ps(sample_idx, i, j);  // pbrt-v4 Sobol+FastOwen
                                        sample = ray_color(r, max_depth, world, lights, ps);
                                        break;
                                    }
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
                            // Film "float maxcomponentvalue" firefly clamp
                            // (max_component_value's own comment) - default
                            // CPU path tracer only; --spectral accumulates
                            // in CIE XYZ at this exact point in the loop
                            // (the "if (spectral)" pixel_color conversion
                            // below), not RGB, so this per-sample RGB clamp
                            // doesn't apply there.
                            if (!spectral) {
                                clamp_sensor_rgb(sample[0], sample[1], sample[2],
                                                  static_cast<float>(max_component_value));
                            }
                            // --adaptive: feed this sample's luminance into the
                            // pixel's running estimate - see camera::
                            // adaptive_sampling's own comment. In spectral mode
                            // `sample` already holds CIE XYZ (see ray_color_spectral()'s
                            // own comment), whose Y channel IS luminance by
                            // definition - no separate RGB-weighted reduction needed
                            // there, unlike the final per-pixel XYZ->RGB conversion
                            // below.
                            if (adaptive_sampling) {
                                const double lum = spectral ? sample.y()
                                    : pixel_convergence::luminance(sample.x(), sample.y(), sample.z());
                                luminance_estimator.Add(lum);
                            }
                            if (render_stats::enabled())
                                render_stats::primary_rays().fetch_add(1, std::memory_order_relaxed);
                            // Reconstruction filter weight (pbrt-v4 filterWeight) -
                            // fs.weight = f(p)/pdf(p), which for correct filter
                            // importance sampling is (nearly) constant across samples
                            // (filter_sampler.h's own comment) - dividing by weight_sum
                            // below instead of a plain sample count is still a valid,
                            // self-normalized Monte Carlo estimator of the same
                            // Integral(f*L)/Integral(f) target (a standard technique,
                            // not a formula this feature needed to change), and
                            // auto-corrects for the table's own quantization noise.
                            double w = fs.weight;
                            weighted_color += w * sample;
                            weight_sum    += w;
                        }
                    }
                // Normalize by filter weight sum -- mirrors pbrt-v4 film:
                //   pixel.rgbSum / pixel.weightSum
                color pixel_color = (weight_sum > 0.0)
                    ? weighted_color / weight_sum
                    : color(0, 0, 0);
                if (spectral) {
                    // pixel_color currently holds the filter-weighted-
                    // averaged CIE XYZ triple, not RGB - see
                    // ray_color_spectral()'s own comment for why that
                    // reduction is deferred to exactly here, once per
                    // pixel, instead of once per sample. Convert (and
                    // clamp negatives from out-of-gamut XYZ, same as the
                    // RGB path's existing per-sample clamp used to) now.
                    float out_r, out_g, out_b;
                    XYZToLinearRGB(static_cast<float>(pixel_color.x()),
                                   static_cast<float>(pixel_color.y()),
                                   static_cast<float>(pixel_color.z()),
                                   out_r, out_g, out_b);
                    pixel_color = color(static_cast<double>(out_r),
                                         static_cast<double>(out_g),
                                         static_cast<double>(out_b));
                }
                pixel_color = pixel_color * exposure;
                if (exr_output) {
                    const size_t idx = (static_cast<size_t>(j) * image_width + i) * 3;
                    exr_pixels[idx + 0] = static_cast<float>(pixel_color.x());
                    exr_pixels[idx + 1] = static_cast<float>(pixel_color.y());
                    exr_pixels[idx + 2] = static_cast<float>(pixel_color.z());
                } else {
                    write_color(ss, pixel_color, tone_map);
                }
            }

            // --time-limit: the per-pixel loop above only stopped
            // adding real samples at columns_rendered - the remaining
            // columns still need filling (already-zero exr_pixels
            // needs nothing further; the PPM row text in `ss` does)
            // before this partial row is usable.
            if (time_limit_hit && !exr_output) append_black_columns(ss, columns_rendered);
            if (!exr_output) { scanlines[j] = ss.str(); row_rendered[j] = 1; }
            int done = ++completed_lines;
            if ((done % 10) == 0 || done == image_height) {
                std::lock_guard<std::mutex> lg(log_mutex);
                std::clog << "\rScanlines remaining: " << (image_height - done) << ' ' << std::flush;
            }
            if (time_limit_hit) break;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (unsigned int t = 0; t < nthreads; ++t)
        threads.emplace_back(worker);

    for (auto &th : threads) th.join();

    // --time-limit: a row that was NEVER CLAIMED by any worker at all
    // (every thread independently observed the deadline and stopped
    // calling next_j.fetch_sub() before reaching this row's index) is
    // NOT the same case the early-abort/mid-abort branches inside
    // worker() handle - those only fire for a row that WAS claimed.
    // exr_pixels is already zero-initialized so an unclaimed row is
    // already correctly black there, but `scanlines` starts as empty
    // std::strings - an unclaimed row would otherwise stay empty,
    // silently corrupting the PPM row/column structure exactly like
    // the abandoned-mid-render case those branches already guard
    // against. row_rendered[j] (not scanlines[j].empty()) is what
    // actually decides this - see that vector's own comment for why an
    // explicit flag rather than inferring it from string emptiness.
    // Cheap to check unconditionally (a no-op loop when
    // time_limit_seconds wasn't used, since every row is already
    // row_rendered in that case).
    if (!exr_output && time_limit_seconds > 0.0) {
        std::ostringstream blackRow;
        append_black_columns(blackRow, 0);
        const std::string blackRowText = blackRow.str();
        for (int j = 0; j < image_height; ++j) {
            if (!row_rendered[j]) scanlines[j] = blackRowText;
        }
        // --time-limit: completed_lines only counts rows a worker
        // actually claimed (whether fully or partially rendered before
        // an abort) - a row backfilled just above because NO thread
        // ever claimed it never went through that counter, so whenever
        // --time-limit actually truncates a render, the last progress
        // line printed inside worker() shows some nonzero "remaining"
        // count and the render() the user sees never gets a closing
        // "Scanlines remaining: 0" - printed here instead, unconditionally,
        // once every row (real or backfilled) is genuinely accounted for.
        std::clog << "\rScanlines remaining: 0 " << std::flush;
    }

    bool wrote_ok = true;
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
            wrote_ok = false;
        }
    } else {
        // Write buffered scanlines in order
        for (int j = 0; j < image_height; ++j) {
            out << scanlines[j];
        }
        out.close();
    }

    std::clog << "\rDone.                 \n";
    return wrote_ok;
}

inline void camera::initialize()
{
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Reconstruction filter (pbrt-v4 style) - see filterSampler_'s own
    // comment for why this builds once here rather than per pixel/
    // ray/thread.
    {
        const PixelFilterDispatch filterDispatch(filter_kind, filter_radius,
                                                  filter_B, filter_C, filter_sigma, filter_tau);
        filterSampler_.emplace(filterDispatch);
    }

    // Resolve crop_x1/crop_y1's "-1 = unset" sentinel now that
    // image_width/image_height are final - scene_registry.h sets these
    // (when it sets them at all) before initialize() runs, in the same
    // pixel space this function just computed, so no rescaling is
    // needed here.
    if (crop_x1 < 0) crop_x1 = image_width;
    if (crop_y1 < 0) crop_y1 = image_height;
    crop_x0 = std::clamp(crop_x0, 0, image_width);
    crop_x1 = std::clamp(crop_x1, 0, image_width);
    crop_y0 = std::clamp(crop_y0, 0, image_height);
    crop_y1 = std::clamp(crop_y1, 0, image_height);
    if (crop_x1 <= crop_x0 || crop_y1 <= crop_y0) {
        // A crop that was a valid, non-degenerate NDC-fraction rectangle
        // in pbrt_flatten.h (which already warns on ITS OWN degenerate
        // case - see that function's own comment) can still collapse to
        // a degenerate PIXEL range here: std::lround() rounds two
        // distinct fractions to the same pixel index at a small enough
        // actual render resolution (exactly the low-res preview use
        // case this feature exists for - see crop_x0's own comment on
        // why the fraction is resolved against the ACTUAL resolution,
        // not the scene's declared one). Warn here too, or a real crop
        // request silently vanishes with zero diagnostic anywhere.
        std::cerr << "Warning: Film \"cropwindow\"/\"pixelbounds\" resolved to an "
                     "empty pixel range at this render resolution (" << image_width
                  << "x" << image_height << "); rendering the full frame instead.\n";
        crop_x0 = 0; crop_x1 = image_width;
        crop_y0 = 0; crop_y1 = image_height;
    }

    sqrt_spp = int(std::sqrt(samples_per_pixel));
    pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
    recip_sqrt_spp = 1.0 / sqrt_spp;
    adaptive_row_order = pixel_convergence::row_visit_order(sqrt_spp);

    center = lookfrom;

    // Determine viewport dimensions.
    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta/2);
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    double center_shift_u = 0.0, center_shift_v = 0.0;
    if (has_screen_window) {
        // pbrt-v4 Camera "perspective" "float screenwindow" - see this
        // field's own comment. h*focus_dist is exactly the world-space
        // distance one NDC unit already covers in the default (unset)
        // case above (default y always spans [-1,1], giving
        // viewport_height=2*h*focus_dist there) - reused directly here
        // so an EXPLICITLY-given window takes both extents verbatim
        // (matching scene_registry.h's own Orthographic-camera
        // precedent - the user's numbers are used as-is, not re-scaled
        // by aspect), including a genuinely off-center one (xmin/xmax
        // not symmetric around 0) via the center_shift_u/v this
        // function's own caller (compute_viewport_geometry) applies.
        //
        // Applied unconditionally whenever has_screen_window is true,
        // with NO special case for a window that happens to equal
        // [-1,1,-1,1] - a code-review pass found an earlier version of
        // this code silently treated that specific explicit value as
        // if no screenwindow had been given at all (routing it through
        // the aspect-scaled default path instead), which diverges from
        // real pbrt-v4 semantics on a non-square-aspect image: real
        // pbrt-v4's own auto-computed default is aspect-scaled (e.g.
        // [-aspect,aspect,-1,1] for aspect>1), genuinely different from
        // a literal [-1,1,-1,1] - has_screen_window (set only when the
        // scene text actually contained a screenwindow directive, see
        // scene_registry.h) already correctly distinguishes "the user
        // wrote this" from "nothing was written" - there is no reason
        // to re-derive that distinction from the numeric value here.
        viewport_width  = (screen_window[1] - screen_window[0]) * h * focus_dist;
        viewport_height = (screen_window[3] - screen_window[2]) * h * focus_dist;
        center_shift_u  = (screen_window[0] + screen_window[1]) * 0.5 * h * focus_dist;
        center_shift_v  = (screen_window[2] + screen_window[3]) * 0.5 * h * focus_dist;
    }

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    compute_lookat_basis(lookfrom, lookat, vup, u, v, w);

    // Calculate the camera defocus disk radius.
    auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));

    // Calculate pixel00_loc/pixel_delta_u/v/defocus_disk_u/v from the
    // world-space u,v,w,center basis above.
    compute_viewport_geometry(u, v, w, center, viewport_width, viewport_height,
                               focus_dist, defocus_radius, image_width, image_height,
                               pixel00_loc, pixel_delta_u, pixel_delta_v,
                               defocus_disk_u, defocus_disk_v,
                               center_shift_u, center_shift_v);

    if (camera_is_animated) {
        // Same 5 quantities as above, but in local camera space
        // (canonical axes local_u=(1,0,0)/local_v=(0,1,0)/
        // local_w=(0,0,1), local_center=(0,0,0)) instead of baked into
        // world space via the (static) lookfrom/lookat/vup basis above -
        // identical formulas, just with the canonical axes substituted
        // for u/v/w/center. These never change per ray; only the
        // camera-to-world placement applied to them in get_ray() does.
        const vec3 local_u(1,0,0), local_v(0,1,0), local_w(0,0,1);
        const point3 local_center(0,0,0);
        compute_viewport_geometry(local_u, local_v, local_w, local_center,
                                   viewport_width, viewport_height, focus_dist,
                                   defocus_radius, image_width, image_height,
                                   local_pixel00_loc, local_pixel_delta_u, local_pixel_delta_v,
                                   local_defocus_disk_u, local_defocus_disk_v,
                                   center_shift_u, center_shift_v);

        // Two camera-to-world keyframes, built via compute_lookat_basis()
        // (the exact same u,v,w derivation the static path above uses),
        // just evaluated at (lookfrom,lookat) and (lookfrom1,lookat1)
        // respectively. Matrix columns are [u | v | w | origin] - maps
        // a local point/vector (expressed in the local_u/local_v/
        // local_w axes above) into world space.
        auto build_cam_to_world = [&](const point3& from, const point3& at) -> AT_Mat44 {
            vec3 ku, kv, kw;
            compute_lookat_basis(from, at, vup, ku, kv, kw);
            AT_Mat44 m;
            m.m[0][0]=ku.x(); m.m[0][1]=kv.x(); m.m[0][2]=kw.x(); m.m[0][3]=from.x();
            m.m[1][0]=ku.y(); m.m[1][1]=kv.y(); m.m[1][2]=kw.y(); m.m[1][3]=from.y();
            m.m[2][0]=ku.z(); m.m[2][1]=kv.z(); m.m[2][2]=kw.z(); m.m[2][3]=from.z();
            m.m[3][0]=0;      m.m[3][1]=0;      m.m[3][2]=0;      m.m[3][3]=1;
            return m;
        };
        anim_cam_to_world_ = AnimatedTransform(
            build_cam_to_world(lookfrom, lookat), shutter_open,
            build_cam_to_world(lookfrom1, lookat1), shutter_close);

        // get_ray() checks the alt-camera-model pointers first (below),
        // so this transform is only ever used for the default
        // perspective path - but that's no longer a silent drop for an
        // alt camera model: scene_registry.h's own setup_camera() (the
        // real pbrt-file-loading path) builds the SAME two-keyframe
        // AnimatedTransform and attaches it directly to whichever
        // alt_*_cam is constructed (see cameras.h's own
        // anim_camera_to_world comment), so an alt camera model
        // genuinely gets real motion blur too there. NOT every
        // alt-camera construction site does this, though -
        // scene_registry_data.h's own compiled-in demo scenes (D2/D3/D4)
        // build alt_ortho_cam/alt_spherical_cam/alt_realistic_cam
        // directly and have never been taught to set
        // anim_camera_to_world - today none of them also set
        // camera_is_animated, so this is latent, but a warning is
        // cheap insurance against a future scene combining the two and
        // silently rendering frozen at the StartTime pose with nothing
        // in the log to explain why (a code-review pass on this exact
        // round is what caught the original blanket "alt camera +
        // motion blur = unsupported" warning being deleted outright
        // rather than narrowed).
        const bool activeAltCameraMissedAnimation =
            (alt_ortho_cam      && !alt_ortho_cam->anim_camera_to_world.has_value()) ||
            (alt_spherical_cam  && !alt_spherical_cam->anim_camera_to_world.has_value()) ||
            (alt_realistic_cam  && !alt_realistic_cam->anim_camera_to_world.has_value());
        if (activeAltCameraMissedAnimation) {
            std::cerr << "Warning: camera_is_animated is set together with an "
                         "alternate camera model (ortho/spherical/realistic), but "
                         "that specific camera object was never given its own "
                         "anim_camera_to_world - it will render static at its "
                         "StartTime pose, not the requested motion blur.\n";
        }
    }
}

inline ray camera::get_ray(int i, int j, int /*s_i*/, int /*s_j*/, const vec3& offset,
    double* out_camera_weight) const
{
    if (out_camera_weight) *out_camera_weight = 1.0;
    // If an alternate camera model is set, delegate ray generation to it.
    if (alt_ortho_cam || alt_spherical_cam || alt_realistic_cam) {
        CameraSample<double> cs;
        cs.pFilm_x = i + offset.x() + 0.5;
        cs.pFilm_y = j + offset.y() + 0.5;
        cs.pLens_u = random_double();
        cs.pLens_v = random_double();
        // Sampled within [shutter_open, shutter_close], not plain
        // [0,1) - matches the default perspective path's own ray_time
        // formula below exactly, and is a genuine no-op when the
        // shutter stays at its [0,1] default (the vast majority of
        // scenes, animated or not). Needed for a genuinely animated alt
        // camera's own anim_camera_to_world (cameras.h) to sample the
        // correct portion of its keyframe interpolation - AnimatedTransform
        // ::Interpolate() clamps outside [startTime,endTime] rather than
        // erroring, so a non-default shutter window without this fix
        // would silently bias every alt-camera ray toward one endpoint
        // instead of spreading across the real exposure.
        cs.time    = shutter_open + random_double() * (shutter_close - shutter_open);
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

    if (camera_is_animated) {
        // Same pixel-sample/defocus-disk math as the default path below,
        // against the LOCAL (canonical-axis) quantities initialize()
        // built instead of the world-space ones - then a single
        // interpolated camera-to-world transform (sampled at this ray's
        // own time) carries the local origin+direction into world space
        // in one step, matching pbrt-v4's own PerspectiveCamera
        // approach (generate in camera space, transform by a possibly
        // time-varying camera-to-world). No ray differentials on this
        // path (matches the alt-camera branch above's own precedent).
        auto local_pixel_sample = local_pixel00_loc
                                 + ((i + offset.x()) * local_pixel_delta_u)
                                 + ((j + offset.y()) * local_pixel_delta_v);
        point3 local_origin(0,0,0);
        if (defocus_angle > 0) {
            auto p = random_in_unit_disk();
            local_origin = point3(0,0,0) + (p[0] * local_defocus_disk_u) + (p[1] * local_defocus_disk_v);
        }
        auto local_direction = local_pixel_sample - local_origin;
        auto ray_time = shutter_open + random_double() * (shutter_close - shutter_open);

        double lo[3] = { local_origin.x(), local_origin.y(), local_origin.z() };
        double ld[3] = { local_direction.x(), local_direction.y(), local_direction.z() };
        double wo[3], wd[3];
        anim_cam_to_world_.apply_ray(lo, ld, ray_time, wo, wd);

        return ray(point3(wo[0], wo[1], wo[2]), vec3(wd[0], wd[1], wd[2]), ray_time);
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
