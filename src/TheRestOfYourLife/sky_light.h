#ifndef SKY_LIGHT_H
#define SKY_LIGHT_H
//==============================================================================
// sky_light -- HDR equirectangular environment / infinite area light
//              with importance sampling (pbrt-v4 ImageInfiniteLight)
//
// Sampling strategy
// -----------------
// At construction, pixel luminances are extracted from the HDR image and a
// 2D piecewise-constant distribution is built over (u,v) image space.
// This mirrors pbrt-v4 ImageInfiniteLight::distribution (s14.2.4).
//
// The (u,v) -> direction mapping is equirectangular:
//   phi   = u * 2*pi            (azimuth,  [0, 2*pi])
//   theta = v * pi              (polar,    [0,   pi])
//   dir   = (sin(theta)*cos(phi), cos(theta), -sin(theta)*sin(phi))
//           (theta=0 -> +y up pole, theta=pi -> -y down pole)
//
// The Jacobian between image-space PDF and solid-angle PDF is:
//   pdf_omega = pdf_image / (2 * pi^2 * sin(theta))
// which follows from dOmega = sin(theta) dTheta dPhi
//                           = sin(theta) * pi * 2*pi * dv * du.
//
// Fallback
// --------
// Constant-color and texture skies use uniform sphere sampling
// (PDF = 1/(4*pi)).  All constructors are backward-compatible.
//==============================================================================

#include "rtweekend.h"
#include "texture.h"
#include "../shared/piecewise_dist.h"

// Result of a sky importance sample (direction + its solid-angle PDF)
struct SkyLiSample {
    vec3   direction; // unit world-space direction toward sky
    double pdf;       // solid-angle PDF at that direction
};
class sky_light {
  public:
    // Constant-color sky -- uniform sphere sampling (PDF = 1/(4*pi))
    explicit sky_light(const color& c) : env_tex(make_shared<solid_color>(c)), scale(1.0), has_dist(false) {}
    // Pre-built texture -- no importance sampling (no raw pixel data available)
    explicit sky_light(shared_ptr<texture> tex) : env_tex(tex), scale(1.0), has_dist(false) {}
    // HDR latlong file -- builds luminance-weighted importance sampling distribution.
    // Mirrors pbrt-v4 ImageInfiniteLight constructor building PiecewiseConstant2D.
    explicit sky_light(const char* hdr_filename, double brightness = 1.0)
        : sky_light(rtw_image(hdr_filename), brightness) {}

    // Same as the filename constructor, but from pixel data already decoded
    // by the caller (e.g. the pbrt loader's EXR path - this class has no
    // format support of its own beyond whatever rtw_image/stb_image already
    // read). Delegates to the rtw_image&& constructor below so the same
    // decoded image builds both the importance-sampling distribution and
    // env_tex, rather than decoding twice the way the filename constructor
    // above effectively used to (once here, again inside hdr_image_texture).
    sky_light(int w, int h, const float* pixels, double brightness = 1.0)
        : sky_light(rtw_image(w, h, pixels), brightness) {}

  private:
    // Shared by both HDR constructors above - takes ownership of an already-
    // decoded image (from a file or a raw buffer, rtw_image does not care
    // which) and builds env_tex and the importance-sampling distribution
    // from that ONE decode.
    explicit sky_light(rtw_image&& img, double brightness)
        : scale(brightness), has_dist(false) {
        int W = img.width(), H = img.height();
        if (W > 0 && H > 0) {
            std::vector<double> weights(W * H);
            for (int v = 0; v < H; ++v) {
                double theta = (v + 0.5) / H * pi;
                double sin_theta = std::sin(theta);
                for (int u = 0; u < W; ++u) {
                    const float* px = img.float_pixel_data(u, v);
                    if (!px) { weights[v * W + u] = 0.0; continue; }
                    // Rec.709 luminance * sin(theta) -- mirrors pbrt-v4 s14.2.4
                    double lum = 0.2126*px[0] + 0.7152*px[1] + 0.0722*px[2];
                    weights[v * W + u] = (lum > 0.0 ? lum : 0.0) * sin_theta;
                }
            }
            dist = PiecewiseConstant2D(weights, W, H);
            has_dist = !dist.empty();
            img_w = W; img_h = H;
        }
        // img moved-from below is still a valid (if now-empty on failure)
        // rtw_image - hdr_image_texture's value() already handles
        // height()<=0 with a cyan debug fallback, so no separate check here.
        env_tex = make_shared<hdr_image_texture>(std::move(img));
    }

  public:
    // Radiance arriving from world direction dir (unit vector expected).
    color Le(const vec3& dir) const {
        auto [u, v] = dir_to_uv(dir);
        return scale * env_tex->value(u, v, point3(0,0,0));
    }
    // Sample direction + solid-angle PDF. Uses importance sampling for HDR,
    // uniform sphere fallback otherwise. Mirrors pbrt-v4 ImageInfiniteLight::SampleLi.
    SkyLiSample sample_Le() const {
        if (!has_dist) { vec3 d = random_unit_vector(); return { d, 1.0/(4.0*pi) }; }
        double ru = random_double(), rv = random_double();
        double pdf_img;
        auto [us, vs] = dist.sample(ru, rv, &pdf_img);
        double phi = us * 2.0 * pi;
        double theta = vs * pi;
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        // Guard: Jacobian singularity at poles
        if (sin_theta < 1e-10) { vec3 d = random_unit_vector(); return { d, 1.0/(4.0*pi) }; }
        vec3 dir(sin_theta * std::cos(phi), cos_theta, -sin_theta * std::sin(phi));
        // Jacobian: pdf_image -> solid-angle PDF
        double pdf_omega = pdf_img / (2.0 * pi * pi * sin_theta);
        return { unit_vector(dir), pdf_omega };
    }
    // Solid-angle PDF for a given direction (MIS weight in miss case + NEE).
    // Mirrors pbrt-v4 ImageInfiniteLight::PDF_Li.
    double pdf_Li(const vec3& dir) const {
        if (!has_dist) return 1.0/(4.0*pi);
        auto [u, v] = dir_to_uv(dir);
        double sin_theta = std::sin(v * pi);
        if (sin_theta < 1e-10) return 0.0;
        return dist.pdf(u, v) / (2.0 * pi * pi * sin_theta);
    }
    // Uniform sphere PDF -- backward-compatible for non-HDR skies
    double pdf_Li() const { return 1.0/(4.0*pi); }
    // Uniform sphere sample -- backward-compatible
    vec3   sample_Li() const { return random_unit_vector(); }
    bool   has_importance_sampling() const { return has_dist; }
  private:
    shared_ptr<texture>  env_tex;
    double               scale = 1.0;
    bool                 has_dist = false;
    int                  img_w = 0, img_h = 0;
    PiecewiseConstant2D  dist;
    // Inverse of the forward mapping documented at the top of this file
    // (dir = (sin(theta)*cos(phi), cos(theta), -sin(theta)*sin(phi)), theta =
    // v*pi, phi = u*2*pi): theta = acos(dir.y), phi = atan2(-dir.z, dir.x)
    // wrapped into [0, 2*pi). Also mirrored bit-for-bit in
    // gpu/optix/gpu_sky_light_shared.h's gpu_sky_dir_to_uv() - keep both in
    // sync if this ever changes again.
    static std::pair<double,double> dir_to_uv(const vec3& d) {
        double cy = d.y();
        if (cy > 1.0) cy = 1.0; if (cy < -1.0) cy = -1.0;
        double theta = std::acos(cy);
        double phi = std::atan2(-d.z(), d.x());
        if (phi < 0.0) phi += 2.0*pi;
        return { phi/(2.0*pi), theta/pi };
    }
};
#endif
