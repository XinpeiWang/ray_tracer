#ifndef SKY_LIGHT_H
#define SKY_LIGHT_H
#include "rtweekend.h"
#include "texture.h"
#include "../shared/piecewise_dist.h"
struct SkyLiSample { vec3 direction; double pdf; };
class sky_light {
  public:
    explicit sky_light(const color& c) : env_tex(make_shared<solid_color>(c)), scale(1.0), has_dist(false) {}
    explicit sky_light(shared_ptr<texture> tex) : env_tex(tex), scale(1.0), has_dist(false) {}
    explicit sky_light(const char* hdr_filename, double brightness = 1.0)
        : env_tex(make_shared<hdr_image_texture>(hdr_filename)), scale(brightness), has_dist(false) {
        rtw_image img(hdr_filename);
        int W = img.width(), H = img.height();
        if (W > 0 && H > 0) {
            std::vector<double> weights(W * H);
            for (int v = 0; v < H; ++v) {
                double theta = (v + 0.5) / H * pi;
                double sin_theta = std::sin(theta);
                for (int u = 0; u < W; ++u) {
                    const float* px = img.float_pixel_data(u, v);
                    if (!px) { weights[v * W + u] = 0.0; continue; }
                    double lum = 0.2126*px[0] + 0.7152*px[1] + 0.0722*px[2];
                    weights[v * W + u] = (lum > 0.0 ? lum : 0.0) * sin_theta;
                }
            }
            dist = PiecewiseConstant2D(weights, W, H);
            has_dist = !dist.empty();
            img_w = W; img_h = H;
        }
    }
    color Le(const vec3& dir) const {
        auto [u, v] = dir_to_uv(dir);
        return scale * env_tex->value(u, v, point3(0,0,0));
    }
    SkyLiSample sample_Le() const {
        if (!has_dist) { vec3 d = random_unit_vector(); return { d, 1.0/(4.0*pi) }; }
        double ru = random_double(), rv = random_double();
        double pdf_img;
        auto [us, vs] = dist.sample(ru, rv, &pdf_img);
        double phi = us * 2.0 * pi;
        double theta = vs * pi;
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        if (sin_theta < 1e-10) { vec3 d = random_unit_vector(); return { d, 1.0/(4.0*pi) }; }
        vec3 dir(sin_theta * std::cos(phi), cos_theta, -sin_theta * std::sin(phi));
        double pdf_omega = pdf_img / (2.0 * pi * pi * sin_theta);
        return { unit_vector(dir), pdf_omega };
    }
    double pdf_Li(const vec3& dir) const {
        if (!has_dist) return 1.0/(4.0*pi);
        auto [u, v] = dir_to_uv(dir);
        double sin_theta = std::sin(v * pi);
        if (sin_theta < 1e-10) return 0.0;
        return dist.pdf(u, v) / (2.0 * pi * pi * sin_theta);
    }
    double pdf_Li() const { return 1.0/(4.0*pi); }
    vec3   sample_Li() const { return random_unit_vector(); }
    bool   has_importance_sampling() const { return has_dist; }
  private:
    shared_ptr<texture>  env_tex;
    double               scale = 1.0;
    bool                 has_dist = false;
    int                  img_w = 0, img_h = 0;
    PiecewiseConstant2D  dist;
    static std::pair<double,double> dir_to_uv(const vec3& d) {
        double ct = -d.y();
        if (ct > 1.0) ct = 1.0; if (ct < -1.0) ct = -1.0;
        double theta = std::acos(ct);
        double phi = std::atan2(-d.z(), d.x()) + pi;
        return { phi/(2.0*pi), theta/pi };
    }
};
#endif
