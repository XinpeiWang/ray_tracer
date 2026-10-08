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
#include <algorithm>
#include <cmath>
#include "material.h"
#include "../shared/adaptive_sampling.h"  // pixel_convergence::has_converged - camera::adaptive_sampling's own comment
#include "../shared/cpu_gpu.h"  // kMaxMediumBoundaryCrossings
#include "../shared/tone_map.h"
#include "../shared/film.h"     // clamp_sensor_rgb - camera::max_component_value's own comment
#include "sky_light.h"
#include "../shared/portal_image_infinite_light.h"   // PortalImageInfiniteLightData<double>
#include "punctual_light_objects.h"
#include "shadow_ray.h"
#include "constant_medium.h"   // ambient_medium - see camera::camera_medium's own comment
#include "../shared/animated_transform.h"
#include "../shared/path_sampler.h"
#include "../shared/sobol_sampler.h"
#include "../shared/stratified_sampler.h"
#include "../shared/pmj02_sampler.h"
#include "../shared/halton_sampler.h"
#include "../shared/independent_sampler.h"
#include "../shared/filter.h"
#include "../shared/filter_sampler.h"
#include "../shared/cameras.h"
#include "../shared/surface_interaction.h"  // compute_differentials() for texture-filtering footprint
#include "../shared/exr_writer.h"
#include "../shared/render_stats.h"
#include "../shared/oidn_runtime.h"   // --denoise on the CPU
#include "../shared/spectral_math.h"  // SampledSpectrum/SampledWavelengths, RGBAlbedoSpectrum/
                                       // RGBIlluminantSpectrum (via cie_data.h -> spectrum_types.h),
                                       // SampledSpectrumToXYZ/XYZToLinearRGB - see ray_color_spectral()
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
// sobol_sampler.h, stratified_sampler.h, pmj02_sampler.h, halton_sampler.h,
// independent_sampler.h) drives ray_color()'s random decisions this
// render. Sobol (this project's
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
    Independent,
};

// Parses a pbrt-v4 Sampler directive's/--sampler flag's type name
// ("sobol"/"zsobol"/"paddedsobol"/"stratified"/"pmj02bn"/"halton"/
// "independent") into a SamplerKind. Any other unrecognized name falls back
// to Sobol - matching the existing Integrator-directive precedent of
// warning rather than failing the render (see
// SceneDescriptor::recommended_integrator's own comment).
inline bool sampler_kind_from_name(const std::string& name, SamplerKind& out) {
    if (name == "sobol")        { out = SamplerKind::Sobol;       return true; }
    if (name == "zsobol")       { out = SamplerKind::ZSobol;      return true; }
    if (name == "paddedsobol")  { out = SamplerKind::PaddedSobol; return true; }
    if (name == "stratified")   { out = SamplerKind::Stratified;  return true; }
    if (name == "pmj02bn")      { out = SamplerKind::PMJ02BN;     return true; }
    if (name == "halton")       { out = SamplerKind::Halton;      return true; }
    if (name == "independent")  { out = SamplerKind::Independent; return true; }
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
    // Which operator write_color() applies before the sRGB OETF - see
    // launcher_args.h's --tonemap flag and tone_map.h's own ToneMapMode
    // doc comment. ACES (default) matches this project's long-standing
    // default behavior; every existing scene/test is unaffected unless
    // --tonemap is explicitly passed.
    ToneMapMode tone_map     = ToneMapMode::ACES;
    // Which sampler drives ray_color()'s random decisions - see SamplerKind's
    // own comment. Default Sobol matches this project's pre-existing,
    // hardcoded behavior exactly.
    SamplerKind sampler_kind = SamplerKind::Sobol;
    // When true, render() dispatches to ray_color_spectral() instead of
    // ray_color() - see that function's own comment. CPU-only, default
    // path tracer only; opt-in via --spectral (launcher_args.h). Only
    // lambertian/metal/dielectric/rough_dielectric/conductor/diffuse_light
    // scenes are supported - cpu_interface.cpp refuses to render (loudly)
    // before this is ever consulted if the loaded scene uses anything else.
    bool spectral = false;
    // Which pixel reconstruction filter shape ray_color()'s samples are
    // weighted by, and how wide (filter_radius - pbrt-v4's real per-kind
    // default, or an explicit xradius/yradius override; see
    // pbrt_flatten::PixelFilter::radius's own comment) - see
    // filterSampler()'s own comment for how a radius wider than one pixel
    // actually reaches neighboring pixels. Unlike sampler_kind above
    // (CLI-only, never auto-applied from a loaded scene's own Sampler
    // directive - see that field's own comment), this one IS set from a
    // loaded pbrt scene's PixelFilter directive automatically - see
    // scene_registry.h's setup for a loaded pbrt scene. Defaults to
    // pbrt-v4's own real default ("gaussian", radius 1.5), matching what an
    // unset PixelFilter directive means.
    std::string filter_kind  = "gaussian";
    double filter_radius     = 1.5;
    double filter_B          = 1.0 / 3.0;
    double filter_C          = 1.0 / 3.0;
    double filter_sigma      = 0.5;
    double filter_tau        = 3.0;

    // pbrt-v4 Film "float maxcomponentvalue" - per-sample firefly clamp:
    // if the largest of a sample's r/g/b exceeds this, all three are
    // scaled down so the max component lands exactly at the threshold
    // (src/shared/film.h's clamp_sensor_rgb() - a pre-existing, tested,
    // but previously totally unwired helper; nothing constructed this
    // codebase's own Film classes from a real scene value before this
    // field existed). Defaults to pbrt-v4's own real default (effectively
    // unbounded), matching what an unset Film "maxcomponentvalue" means -
    // same "auto-applied from a loaded pbrt scene's own directive" shape
    // as filter_kind above. CPU default path tracer only (ray_color()/
    // ray_color_spectral()), matching filter_kind/regularize's own scope
    // cut - GPU has no equivalent clamp at all yet, warned about
    // separately (gpu/optix/scene_builder.cpp) the same way a real,
    // non-default PixelFilter radius already is.
    double max_component_value = 1e9;

    // pbrt-v4's real Integrator "bool regularize" (defaults false, matching
    // pbrt-v4's own default) - widens a rough BSDF's GGX alpha after the
    // path's first non-specular bounce (see src/shared/microfacet.h's
    // RegularizeAlpha()). Same "auto-applied from a loaded pbrt scene's own
    // directive" shape as filter_kind above (scene_registry.h's setup_camera)
    // - unlike max_depth/samples_per_pixel, this is a genuine scene-authored
    // behavior toggle, not a perf knob meant to be freely CLI-overridden.
    bool regularize = false;

    // pbrt-v4's real Film "cropwindow"/"pixelbounds" - restricts which
    // pixels actually get traced to a rectangle of the frame; same
    // "auto-applied from a loaded pbrt scene's own directive" shape as
    // filter_kind/regularize above. Set in PIXEL coordinates against
    // this camera's own image_width/image_height by scene_registry.h
    // (which converts pbrt_flatten::FlatScene's resolution-independent
    // NDC fractions using the ACTUAL render resolution, not the scene's
    // own declared one - see that conversion's own comment for why).
    // Defaults to the full frame, so a scene/native build with no crop
    // request renders unchanged. A pixel outside [crop_x0,crop_x1) x
    // [crop_y0,crop_y1) is written black rather than shrinking the
    // output image - see render()'s own comment on why this is an
    // approximation of real pbrt-v4 (which writes a smaller file).
    int crop_x0 = 0, crop_x1 = -1, crop_y0 = 0, crop_y1 = -1;

    // --seed (CLI/GUI) - an explicit request for reproducible renders.
    // -1 (default) means "not requested": render() leaves
    // rtweekend.h's thread_rng() on its pre-existing, genuinely
    // non-deterministic hardware-entropy seeding, and every alternate
    // sampler below (ZSobol/PaddedSobol/Stratified/PMJ02BN/Halton) keeps
    // using literal seed 0, exactly as before this field existed - zero
    // behavior change for a render that never sets this. >= 0 makes BOTH
    // of those deterministic: render() reseeds rtweekend.h's thread_rng()
    // once per scanline via the private reseed_render_rng() method below
    // (see its own comment for why per-scanline rather than per-worker
    // is required), and the alternate-sampler constructors below use this
    // value instead of a hardcoded 0. No CLI equivalent existed for this
    // at all before - unlike regularize/max_component_value/crop above,
    // there's no "scene's own directive" to preserve, since pbrt-v4's
    // own "Sampler \"seed\"\" isn't parsed by this project's pbrt loader
    // either (see docs/PBRT_SUPPORT.md).
    long long seed = -1;

    // --adaptive/--adaptive-threshold (CLI/GUI) - stops sampling a pixel
    // once its running luminance estimate has converged well enough,
    // instead of always spending the full samples_per_pixel budget on
    // every pixel regardless of how quickly it converged - see
    // src/shared/adaptive_sampling.h's own comment for the stopping rule.
    // samples_per_pixel becomes a ceiling rather than a fixed count when
    // this is on; off (default) renders exactly sqrt_spp*sqrt_spp samples
    // per pixel like before this field existed. CPU default path tracer
    // only, same scope cut as spectral/sampler_kind above - the stratified
    // sqrt_spp x sqrt_spp sampling loop this relies on (render()'s own
    // comment) doesn't exist in BDPT/MLT/SPPM's own sampling loops.
    bool adaptive_sampling = false;
    // Target relative standard error of a pixel's running luminance mean -
    // see pixel_convergence::has_converged()'s own comment. Cycles' own
    // adaptive_threshold default (0.01) is reused here as a familiar
    // starting point for anyone coming from that renderer.
    double adaptive_threshold = 0.01;

    // --time-limit (CLI/GUI) - stops rendering once this many seconds have
    // elapsed since render() started, instead of always running until
    // every scanline is done. <= 0.0 (default) means "no limit", this
    // project's prior, only-ever behavior. Checked at TWO granularities in
    // render()'s work-stealing loop: once per scanline claim (cheap, catches
    // the common case immediately), and again every 32nd pixel column within
    // whatever scanline is currently in flight (catches a single very-high-
    // spp scanline that could otherwise itself take several seconds,
    // overshooting a short deadline by a whole row's duration - the 32-pixel
    // stride avoids reading the clock on every single pixel, a real per-
    // pixel-per-thread cost for deadline precision nobody needs) - see
    // render()'s own elapsed_seconds()/append_black_columns() comments for
    // the exact mechanics. A row cut short mid-render keeps its
    // already-computed columns and gets the remainder filled black, rather
    // than a plain per-pixel weight_sum-based partial-sample normalization
    // (which the already-computed columns don't need at all - each of them
    // already went through the normal, complete per-pixel accumulation).
    // Under --video, main.cpp's own frame loop treats this as a budget for
    // the WHOLE video, re-deriving each frame's own value as whatever's
    // left of it (not the same full value reused every frame) - see that
    // loop's own comment.
    // CPU default path tracer only, same scope cut as spectral/
    // adaptive_sampling above - camera::render() is this integrator's own
    // scanline loop, not something BDPT/MLT/SPPM's separate render loops
    // (cpu_interface_bdpt.cpp, sppm.h) call into.
    double time_limit_seconds = 0.0;

    // When set, render() writes a linear (pre-tonemap), full-float EXR
    // instead of the tonemapped/quantized PPM it writes by default - see
    // exr_writer.h. Set by cpu_interface.cpp when the caller's requested
    // output path ends in ".exr"; false (PPM) is the pre-existing default
    // and behavior for every other extension is unchanged.
    bool   exr_output = false;
    // --denoise on the CPU: Open Image Denoise (src/shared/oidn_runtime.h, loaded at run time) runs on the finished linear image before it is tone mapped
    // or written. Colour only. If the library is missing the render says so and goes on undenoised. denoise_keep is the share of the original image kept
    // (RenderOptions::denoise_blend: 0 = fully denoised).
    bool   denoise = false;
    float  denoise_keep = 0.0f;
    color  background;               // Scene background color (used when sky==nullptr)
    shared_ptr<sky_light> sky;               // HDR env map (pbrt-v4 ImageInfiniteLight); nullptr = flat background
    // pbrt-v4 windowed/portal infinite light ("point3 portal[4]") - visible
    // only through a finite rectangular window, not the whole sphere.
    // Mutually exclusive with `sky` above (see pbrt_cpu_builder.h's
    // BuildResult::portal comment) - every ray_color()/ray_color_spectral()
    // call site below checks `portal` first, falling through to `sky` only
    // when this is null.
    shared_ptr<PortalImageInfiniteLightData<double>> portal;
    shared_ptr<punctual_light_list> punct_lights; // pbrt-v4 PointLight/SpotLight/DistantLight (delta); nullptr = none
    // pbrt-v4's own "camera medium" (FlatScene::cameraMediumIndex's own
    // comment, pbrt_flatten.h) - an unbounded homogeneous medium the camera
    // itself starts inside, no bounding shape needed. nullptr = none (the
    // pre-existing default for every scene that doesn't request this).
    // ray_color() only (see ambient_medium's own scope comment,
    // constant_medium.h) - ray_color_spectral()/BDPT/MLT/SPPM don't read
    // this field at all yet.
    shared_ptr<ambient_medium> camera_medium;

    // The scene's per-shape homogeneous media whose extinction differs between colour channels (BuildResult::chromaticMedia). ray_color()
    // samples these itself (constant_medium::sample_event()) against the nearest surface, because the single-extinction free-flight
    // that constant_medium::hit() does cannot represent a medium that is thicker in one channel than another. Empty (the default) for
    // every grey-medium scene, and ignored by ray_color_spectral() and the other integrators.
    std::vector<shared_ptr<event_medium>> shape_media;

    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    // pbrt-v4 Camera "perspective" "float screenwindow" (xmin,xmax,ymin,ymax,
    // in NDC units where the default, unset window already matches this
    // class's own vfov/aspect-derived viewport exactly - see initialize()'s
    // own comment). Real pbrt-v4 use: anamorphic/cropped/off-center framing
    // - e.g. xmin=-2,xmax=2,ymin=-1,ymax=1 for a 2:1 anamorphic crop wider
    // than a plain aspect-ratio resize would give. scene_registry.h's alt-
    // camera dispatch (Orthographic/Spherical/Realistic) already reads and
    // honors an explicit screenwindow for Orthographic; this is the
    // identical directive, just for the DEFAULT perspective camera this
    // class itself renders, previously silently discarded (see this
    // field's own set site, scene_registry.h, for the historical bug this
    // closes).
    bool has_screen_window = false;
    double screen_window[4] = {-1.0, 1.0, -1.0, 1.0};  // xmin, xmax, ymin, ymax

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    // Camera motion blur (pbrt-v4 AnimatedTransform, src/shared/
    // animated_transform.h) - when camera_is_animated is set, the camera-
    // to-world transform is keyframed between (lookfrom,lookat) at
    // shutter_open and (lookfrom1,lookat1) at shutter_close, and each ray
    // samples its own time in [shutter_open, shutter_close) and is
    // generated from the interpolated transform at that time - see
    // get_ray()'s own comment. vup is shared by both keyframes (a camera
    // roll during the exposure isn't supported by this simplified
    // two-keyframe setup). False (default) is the pre-existing static
    // camera behavior, unchanged.
    // CAUTION: object motion blur (sphere.h's moving-sphere hit()) hard-
    // assumes ray.time() in [0,1] - both its center.at() extrapolation and
    // its precomputed bounding box are keyed to that exact range. Keep
    // shutter_open/shutter_close at [0,1] (the default) in any scene that
    // also uses a moving sphere, or the sphere's bounding box won't cover
    // the ray's actual sampled time range and it can be placed/culled
    // wrong. No scene combines the two yet (D13 uses [0,1] regardless).
    bool   camera_is_animated = false;
    point3 lookfrom1      = point3(0,0,-1);
    point3 lookat1        = point3(0,0,-2);
    double shutter_open   = 0.0;
    double shutter_close  = 1.0;

    // Alternate camera models (pbrt-v4 cameras.h). When set, overrides default perspective.
    shared_ptr<OrthographicCamera<double>> alt_ortho_cam;
    shared_ptr<SphericalCamera<double>>    alt_spherical_cam;
    shared_ptr<RealisticCamera<double>>    alt_realistic_cam;

    int    image_height = 0;         // Rendered image height (set by initialize())
    point3 center;                   // Camera center (set by initialize())

    // output_path: the caller's real requested output path - matches every
    // other integrator/backend in this codebase (BDPT/MLT/randomwalk in
    // cpu_interface_bdpt.cpp, GPU's optix_render_main), which all take
    // output path as an explicit render-entry parameter rather than a
    // field the caller must remember to set on the object first; a
    // parameter also means a caller that forgets it is a compile error,
    // not a silent CWD-relative image.ppm/image.exr. Empty (the default) is
    // only a defensive fallback for the same "requested path not writable"
    // case that used to be render()'s own everyday first choice.
    //
    // Returns false if no output file could be written anywhere (requested
    // path, cwd, and TEMP all failed, or the EXR encoder itself failed) -
    // the caller (cpu_interface.cpp) turns that into ERR_FILE_WRITE_FAILED
    // instead of reporting SUCCESS with no file actually on disk.
    bool render(const hittable& world, const hittable& lights,
                const std::string& output_path = std::string());

  private:
    // --seed support (see camera::seed's own comment above).
    //
    // Reseeding has to happen PER SCANLINE, not once per worker thread at
    // startup - render()'s worker loop below pulls scanlines from a shared
    // atomic counter (next_j.fetch_sub(1)), a work-stealing scheduler, so
    // WHICH thread ends up rendering scanline j is itself a scheduling
    // race, not reproducible run to run. Seeding by "which worker am I"
    // (a first attempt at this feature, since reverted) therefore doesn't
    // work: the same scanline can land on a different thread - and see a
    // different RNG state - on two runs with an identical --seed, even
    // though each individual thread's OWN sequence was internally
    // deterministic. Keying the reseed on the scanline index j instead - a
    // property of the WORK ITEM, never the worker - fixes this: every
    // pixel in scanline j always sees the exact same RNG state at the
    // start of that scanline's render, regardless of which thread got
    // there. (Within one scanline, the same thread renders every pixel
    // single-threaded in the same fixed order every run, so one reseed per
    // scanline is enough - no per-pixel reseed needed.)
    //
    // Deliberately a private member of camera rather than a free function
    // in the shared, generic-feeling rtweekend.h (this project's own
    // thread_rng()/random_double() home): the invariant above is specific
    // to render()'s own work-stealing scanline scheduler, not a general-
    // purpose utility - a caller in some other loop shape could satisfy
    // this function's signature while violating the assumption it depends
    // on. Scoping it here, next to the one caller (render(), below) whose
    // scheduling shape it was actually derived from, means extending
    // --seed to a different render loop later (BDPT/MLT/SPPM) forces a
    // fresh look at whether the same "key on the work item, not the
    // worker" reasoning even applies there, rather than inviting a
    // silent, wrong reuse of this exact function.
    static void reseed_render_rng(int64_t seed, int64_t stream) {
        // Hash(seed, stream) (src/shared/pbrt_hash.h's MurmurHash64A-backed
        // variadic hasher, reachable here via rtweekend.h's own include of
        // rng.h -> pbrt_hash.h) combines both values into a well-
        // distributed 64-bit sequence index, matching how every other
        // seed-derivation site in this codebase turns a work-item index
        // into an RNG seed (e.g. RNG::SetSequence(uint64_t)'s own
        // single-arg overload calls MixBits() for exactly this purpose).
        thread_rng().SetSequence(Hash(seed, stream), static_cast<uint64_t>(seed));
    }

    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    int    sqrt_spp;             // Square root of number of samples per pixel
    double recip_sqrt_spp;       // 1 / sqrt_spp
    // Which stratified row (s_j) render()'s sample loop visits at each
    // iteration - see pixel_convergence::row_visit_order()'s own comment
    // (adaptive_sampling.h) for why this must NOT be raster order 0,1,2,...
    // when --adaptive can break out of that loop early. Computed once here
    // (not per-pixel - it depends only on sqrt_spp, fixed for the whole
    // render) regardless of whether adaptive_sampling is on, since sorting
    // a handful of ints once per render is free either way.
    std::vector<int> adaptive_row_order;
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    // Reconstruction filter (pbrt-v4 style) - built ONCE in initialize(),
    // not per pixel/ray/thread: FilterSampler's own constructor does real
    // work (tabulates the filter on a 32x32 grid and builds CDFs,
    // filter_sampler.h's own comment), and is read-only afterward, so
    // sharing one instance read-only across worker threads is both correct
    // and the cheapest option - matching pixel00_loc/pixel_delta_u above's
    // own "precompute once in initialize(), never touched again" contract.
    // Shape/radius driven by filter_kind/filter_radius/filter_B/filter_C/
    // filter_sigma/filter_tau - see filter_radius's own comment for why
    // radius is no longer hardcoded to 0.5. std::optional because
    // FilterSampler has no default constructor (it always needs a real
    // Filter to tabulate).
    std::optional<FilterSampler<double, 32>> filterSampler_;

    // Camera motion blur (camera_is_animated only) - the SAME quantities
    // as pixel00_loc/pixel_delta_u/v/defocus_disk_u/v above, but expressed
    // in local camera space (canonical axes, identity camera-to-world)
    // rather than baked into world space - see get_ray()'s own comment for
    // why. anim_cam_to_world_ interpolates the actual world placement per
    // ray, applied to these fixed local quantities.
    point3 local_pixel00_loc;
    vec3   local_pixel_delta_u;
    vec3   local_pixel_delta_v;
    vec3   local_defocus_disk_u;
    vec3   local_defocus_disk_v;
    AnimatedTransform anim_cam_to_world_;

    // Shared by initialize()'s static u/v/w derivation and (camera_is_
    // animated) build_cam_to_world()'s per-keyframe basis below - same
    // right/up/back-from-lookat formula, evaluated against whichever
    // (from,at) pair the caller passes in, so the two can't drift apart
    // from independent copy-paste edits.
    static void compute_lookat_basis(const point3& from, const point3& at,
                                      const vec3& vup,
                                      vec3& out_u, vec3& out_v, vec3& out_w) {
        out_w = unit_vector(from - at);
        out_u = unit_vector(cross(vup, out_w));
        out_v = cross(out_w, out_u);
    }

    // Shared by initialize()'s world-space and (camera_is_animated) local-
    // space viewport setups below - same formula, evaluated against
    // whichever (u,v,w,center) basis the caller passes in, so the two
    // can't drift apart from independent copy-paste edits.
    static void compute_viewport_geometry(
            const vec3& u, const vec3& v, const vec3& w, const point3& center,
            double viewport_width, double viewport_height, double focus_dist,
            double defocus_radius, int image_width, int image_height,
            point3& out_pixel00_loc, vec3& out_pixel_delta_u, vec3& out_pixel_delta_v,
            vec3& out_defocus_disk_u, vec3& out_defocus_disk_v,
            // World-space shift of the viewport's own center away from the
            // look direction, along u (screen +x) and v (screen +y) - zero
            // for the overwhelming common case (a centered viewport). Real
            // use: an off-center "float screenwindow" (initialize()'s own
            // comment) - e.g. xmin=0,xmax=2 (instead of the symmetric
            // xmin=-1,xmax=1 default) shifts the frame one full half-width
            // to the right rather than just widening it.
            double center_shift_u = 0.0, double center_shift_v = 0.0) {
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;
        out_pixel_delta_u = viewport_u / image_width;
        out_pixel_delta_v = viewport_v / image_height;
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2
                                    + center_shift_u * u + center_shift_v * v;
        out_pixel00_loc = viewport_upper_left + 0.5 * (out_pixel_delta_u + out_pixel_delta_v);
        out_defocus_disk_u = u * defocus_radius;
        out_defocus_disk_v = v * defocus_radius;
    }

  public:
    void initialize();

    ray get_ray(int i, int j, int s_i, int s_j, int sample_idx = 0, int px = 0, int py = 0) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j for stratified sample square s_i, s_j.
        // sample_idx + pixel coords drive Halton per-pixel decorrelation (pbrt-v4 pattern).
        // Filter-importance-warped the same way the render loop's own call
        // site is (sample_unit_square_stratified/filterSampler_'s own
        // comment) - requires initialize() to have run first (filterSampler_
        // is only set there), same precondition get_ray() already has for
        // pixel00_loc/pixel_delta_u/v.
        vec3 u = sample_unit_square_stratified(s_i, s_j, sample_idx, px, py);
        const FilterSample<double> fs = filterSampler_->sample(u.x(), u.y());
        vec3 offset(fs.p_x, fs.p_y, 0);
        return get_ray(i, j, s_i, s_j, offset);
    }

    // A camera ray for pixel (i, j) whose film position is importance-sampled through the scene's reconstruction filter (the same FilterSampler render()
    // uses): u0, u1 are uniform [0,1) and `filter_weight` receives that sample's filter weight (pbrt-v4 CameraSample::filterWeight; the pixel value is
    // sum(weight * L) / sum(weight)). For the integrators that do not run render()'s own loop (BDPT, the debug integrators), so their camera-path estimates
    // are filtered like the path tracer's instead of box-sampled. Requires initialize() to have run.
    ray get_filtered_ray(int i, int j, double u0, double u1, double& filter_weight) const {
        const FilterSample<double> fs = filterSampler_->sample(u0, u1);
        filter_weight = fs.weight;
        return get_ray(i, j, 0, 0, vec3(fs.p_x, fs.p_y, 0));
    }

    // Overload accepting a pre-computed sub-pixel offset (avoids double Halton evaluation
    // when the caller already has the offset for filter weight computation).
    // out_camera_weight, if non-null, receives the camera's exposure weight for this
    // sample (pbrt-v4 CameraRay::weight - always 1 for Ortho/Spherical, but for
    // RealisticCamera it's cos^4(theta)/(pdf*LensRearZ^2) and MUST be multiplied into
    // the returned radiance, exactly like pbrt-v4's integrators do (L *= cameraRay->weight)
    // - omitting it silently under-exposes every RealisticCamera render.
    ray get_ray(int i, int j, int /*s_i*/, int /*s_j*/, const vec3& offset,
                double* out_camera_weight = nullptr) const;

    // Stratified [0,1)^2 draw for sub-pixel grid stratum (s_i, s_j) - feeds
    // FilterSampler::sample() (filter_sampler.h) at both ray_color() and
    // ray_color_spectral()'s own call sites, which CDF-inverts it into the
    // real requested filter's [-filter_radius,+filter_radius]^2 support
    // (weighted appropriately) - replacing this project's own former
    // "always exactly [-0.5,0.5], regardless of what radius a scene
    // requested" mapping (see camera::filter_radius's own comment for why
    // that was a real, previously-undisclosed gap: get_ray()'s own
    // pixel_sample computation already adds this offset straight onto the
    // pixel index, so a wider offset already reaches a neighboring pixel's
    // film-plane area correctly - filter importance sampling just needed
    // to be allowed to draw one).
    //
    // Jitter uses Halton low-discrepancy sequences (pbrt-v4 HaltonSampler pattern):
    //   x <- base-2 radical inverse, per-pixel decorrelated
    //   y <- base-3 radical inverse, per-pixel decorrelated
    // Pixel coordinates are mixed into the index so adjacent pixels use different
    // sub-sequences, avoiding the structured grid artifact from shared Halton offsets.
    vec3 sample_unit_square_stratified(int s_i, int s_j, int sample_idx = 0,
                                        int pixel_x = 0, int pixel_y = 0) const {
        unsigned int ui = (unsigned int)sample_idx;
        unsigned int ux = (unsigned int)pixel_x;
        unsigned int uy = (unsigned int)pixel_y;
        auto ox = (double)halton2(ui, ux, uy);
        auto oy = (double)halton3(ui, ux, uy);
        auto u1 = (s_i + ox) * recip_sqrt_spp;
        auto u2 = (s_j + oy) * recip_sqrt_spp;

        return vec3(u1, u2, 0);
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
                             color& beta, double& eta_scale) const;

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
    //
    // ray_color_spectral() below is a deliberate, hand-duplicated structural
    // mirror of this function's NEE/MIS/Russian-Roulette control flow (see
    // its own doc comment for why a shared templated core wasn't used - the
    // same isolate-the-new-path-from-the-proven-path tradeoff this codebase
    // already makes in bxdfs_layered.h/bdpt_adapter.h/mlt.h). A correctness
    // fix here (NEE weighting, RR thresholds, MIS math) needs the SAME fix
    // manually re-applied there, or the two integrators silently diverge -
    // there is no compiler warning for a missed one.
    template <typename Sampler>
    color ray_color(const ray& r, int depth, const hittable& world, const hittable& lights,
                    Sampler& sampler)
    const;

    // ray_color_spectral -- spectral variant of ray_color() (see that
    // function's own comment for the control flow this mirrors, bounce for
    // bounce - and for the warning that a NEE/MIS/RR fix there needs to be
    // manually re-applied here too). Samples 4 hero wavelengths once per path
    // (SampledWavelengths<4>::SampleVisible) and carries a real
    // SampledSpectrum<4> beta/L through every bounce - not just a one-time
    // reduction at the end - since the whole point of --spectral is that
    // wavelength sampling actually participates in the walk, giving a
    // different noise pattern than RGB while converging to the same
    // expected color on these non-dispersive materials. Reduces to CIE XYZ
    // exactly once at the end via SampledSpectrumToXYZ (sampled_spectrum.h)
    // - the same technique already proven correct on the GPU wavefront
    // backend (gpu/optix/wavefront_kernels.cu) - but returns that XYZ
    // triple, NOT RGB: XYZ is additive and non-negative by construction (the
    // CIE curves and every L this function can produce are both
    // non-negative), so it is safe for render()'s pixel loop to
    // filter-weight-average many samples' XYZ together the same way it
    // already averages ray_color()'s RGB. The XYZ->RGB matrix multiply
    // (XYZToLinearRGB) has negative coefficients, and a single narrow
    // hero-wavelength sample routinely lands outside the sRGB gamut - doing
    // that conversion (and its negative-clamp) per SAMPLE here would clip
    // away part of each sample's color before it had a chance to combine
    // with the others, a real, systematic desaturation/brightness bias on
    // any near-gamut-boundary color. render() now does that conversion
    // exactly once per PIXEL, after averaging - see its own comment at the
    // spectral post-processing step. Still deliberately not routed through
    // PixelSensor/SpectralFilm (pixel_sensor.h/film.h): PixelSensor::
    // ToSensorRGB's output is scaled by roughly kCIE_Y_integral (~107x)
    // relative to SampledSpectrumToXYZ's for the same input (it has no
    // /kCIE_Y_integral term this codebase's every other XYZ-producing
    // function applies) - real pbrt-v4 may fold that factor into its own
    // Film::Create()-computed imagingRatio rather than into ToSensorRGB
    // itself, so this is NOT confirmed to be a bug in ToSensorRGB, just a
    // real, unresolved discrepancy that makes wiring it in with this port's
    // simplified imagingRatio=1 default (PixelSensor::CreateDefault())
    // risky to do without deeper pbrt-v4 source verification than this
    // round did - left untouched rather than guessed at. SpectralFilm's
    // per-bucket spectral-image storage also has no consumer here (this
    // renderer only ever emits an RGB image), so there's nothing this fix
    // would gain from either class over reusing SampledSpectrumToXYZ/
    // XYZToLinearRGB, which are already proven correct by the GPU wavefront
    // parity tests.
    //
    // Every RGB attenuation/reflectance value (srec.attenuation, NEE
    // `atten`/`trans`) is uplifted via the BOUNDED technique
    // (RGBAlbedoSpectrum) - all 5 materials --spectral supports produce a
    // [0,1]-bounded reflectance/Fresnel value, never HDR (>1) output, so
    // RGBUnboundedSpectrum (GPU wavefront's own Hair-only case) is never
    // needed here. Every emission value (rec.mat->emitted(), sky/background
    // Le, punctual light Li) is uplifted via the ILLUMINANT technique
    // (RGBIlluminantSpectrum), always through GetNormalizedD65Illuminant()
    // (cie_data.h) - NEVER the raw GetD65Illuminant() table, which would
    // silently reproduce a real bug the GPU wavefront backend already
    // shipped and fixed (missing D65 normalization -> inflated/desaturated
    // colors, only caught by a parity test - see that function's own
    // comment in cie_data.h).
    //
    // No BSSRDF branch: `subsurface` is not in --spectral's material
    // whitelist, enforced at scene-load time (cpu_interface.cpp) before
    // this function is ever called, so srec.is_transmission &&
    // rec.mat->as_subsurface(rec) never fires here - the specular branch
    // below is unconditional, unlike ray_color()'s own bssrdf_exit-gated
    // version.
    template <typename Sampler>
    color ray_color_spectral(const ray& r, int depth, const hittable& world, const hittable& lights,
                    Sampler& sampler)
    const;
};


#include "camera_render.h"
#include "camera_path.h"

#endif
