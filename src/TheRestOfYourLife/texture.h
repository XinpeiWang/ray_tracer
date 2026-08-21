#ifndef TEXTURE_H
#define TEXTURE_H
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

#include "perlin.h"
#include "rtw_stb_image.h"
#include "../shared/mipmap.h"


class texture {
  public:
    virtual ~texture() = default;

    virtual color value(double u, double v, const point3& p) const = 0;

    // Differential-aware lookup: dudx/dvdx/dudy/dvdy are the UV screen-space
    // derivatives from compute_differentials() (src/shared/surface_
    // interaction.h), i.e. this hit's texture-lookup footprint - used by
    // mipmap_texture to do EWA filtering instead of a single point sample.
    // Default just forwards to value() and ignores the footprint, so every
    // texture that has no notion of filtering (solid_color, checker_texture,
    // noise/fbm/marble, hdr_image_texture) needs no changes at all.
    virtual color value_diff(double u, double v, const point3& p,
                              double /*dudx*/, double /*dvdx*/,
                              double /*dudy*/, double /*dvdy*/) const {
        return value(u, v, p);
    }
};


class solid_color : public texture {
  public:
    solid_color(const color& albedo) : albedo(albedo) {}

    solid_color(double red, double green, double blue) : solid_color(color(red,green,blue)) {}

    color value(double u, double v, const point3& p) const override {
        return albedo;
    }

    // Accessor for serializer
    color color_value() const { return albedo; }

  private:
    color albedo;
};


class checker_texture : public texture {
  public:
    checker_texture(double scale, shared_ptr<texture> even, shared_ptr<texture> odd)
      : inv_scale(1.0 / scale), even(even), odd(odd) {}

    checker_texture(double scale, const color& c1, const color& c2)
      : checker_texture(scale, make_shared<solid_color>(c1), make_shared<solid_color>(c2)) {}

    color value(double u, double v, const point3& p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    double inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
};


// pbrt-v4's "checkerboard" texture class: a 2D, UV-tiled checker (parity of
// floor(u*uscale)+floor(v*vscale)), NOT the 3D world-space checker
// checker_texture above already implements (that one is this project's own
// original Shirley-book pattern, keyed on world position - a materially
// different, scale/position-dependent result, not a drop-in match for
// pbrt's UV-based convention). Kept as a separate class rather than a
// checker_texture variant so neither one's semantics have to compromise.
class uv_checker_texture : public texture {
  public:
    uv_checker_texture(double uscale, double vscale,
                        shared_ptr<texture> tex1, shared_ptr<texture> tex2)
      : uscale(uscale), vscale(vscale), tex1(tex1), tex2(tex2) {}

    uv_checker_texture(double uscale, double vscale, const color& c1, const color& c2)
      : uv_checker_texture(uscale, vscale, make_shared<solid_color>(c1), make_shared<solid_color>(c2)) {}

    color value(double u, double v, const point3& p) const override {
        const int ui = int(std::floor(u * uscale));
        const int vi = int(std::floor(v * vscale));
        const bool isEven = (ui + vi) % 2 == 0;
        return isEven ? tex1->value(u, v, p) : tex2->value(u, v, p);
    }

  private:
    double uscale, vscale;
    shared_ptr<texture> tex1, tex2;
};


class image_texture : public texture {
  public:
    image_texture(const char* filename) : image(filename) {}

    // Takes ownership of an already-loaded rtw_image instead of decoding
    // the file a second time -- for callers that need to check whether a
    // load actually succeeded (image.height() > 0) before committing to an
    // image_texture, e.g. mesh.h's load_obj_mtl() probing a map_Kd texture
    // and falling back to a flat Kd color on failure instead of this
    // class's own cyan-debug fallback.
    explicit image_texture(rtw_image&& loaded) : image(std::move(loaded)) {}

    color value(double u, double v, const point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return color(0,1,1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0,1).clamp(u);
        v = 1.0 - interval(0,1).clamp(v);  // Flip V to image coordinates

        auto i = int(u * image.width());
        auto j = int(v * image.height());
        auto pixel = image.pixel_data(i,j);

        auto color_scale = 1.0 / 255.0;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    rtw_image image;
};


// HDR-preserving image texture -- reads raw float pixels so values > 1 are retained.
// Use this for environment maps / sky lights loaded from .hdr files.
// Mirrors pbrt-v4: image data is kept in linear floating-point throughout.
class hdr_image_texture : public texture {
  public:
    hdr_image_texture(const char* filename) : image(filename) {}

    // Mirrors image_texture(rtw_image&&) just above - takes ownership of an
    // already-decoded image instead of decoding a file a second time. The
    // pbrt loader's EXR path is the caller: it has to decode the image once
    // anyway to build sky_light's importance-sampling distribution, and this
    // lets that same decode become the texture sky_light::Le() reads from,
    // rather than re-reading (and, for EXR, re-decoding) the file here too.
    explicit hdr_image_texture(rtw_image&& loaded) : image(std::move(loaded)) {}

    color value(double u, double v, const point3& p) const override {
        if (image.height() <= 0) return color(0, 1, 1); // cyan debug fallback

        u = interval(0,1).clamp(u);
        v = 1.0 - interval(0,1).clamp(v); // flip V: image origin is top-left

        int i = int(u * image.width());
        int j = int(v * image.height());

        const float* pixel = image.float_pixel_data(i, j);
        if (!pixel) return color(0, 1, 1); // no float data

        // Return raw HDR values -- no clamping, no divide-by-255
        return color(pixel[0], pixel[1], pixel[2]);
    }

  private:
    rtw_image image;
};


class noise_texture : public texture {
  public:
    noise_texture(double scale) : scale(scale) {}

    color value(double u, double v, const point3& p) const override {
        return color(.5, .5, .5) * (1 + std::sin(scale * p.z() + 10 * noise.turb(p, 7)));
    }

  private:
    perlin noise;
    double scale;
};


// ---------------------------------------------------------------------------
// fbm_texture
// Fractional Brownian motion texture using pbrt-v4 FBm.
// Maps the signed FBm value [-1,1] to a greyscale color via (1+v)/2.
// pbrt-v4 reference: FBmTexture (textures.h / textures.cpp)
// ---------------------------------------------------------------------------
class fbm_texture : public texture {
  public:
    // scale:    spatial frequency multiplier for the input point
    // octaves:  number of octaves of noise
    // omega:    persistence (roughness), 0 < omega < 1; pbrt-v4 default = 0.5
    fbm_texture(double scale, int octaves = 8, double omega = 0.5)
        : scale(scale), octaves(octaves), omega(omega) {}

    color value(double /*u*/, double /*v*/, const point3& p) const override {
        // pbrt-v4 FBmTexture: Evaluate = FBm(c.p, c.dpdx, c.dpdy, omega, octaves)
        // We use the simple (non-antialiased) variant with no footprint info.
        double v = fbm_simple<double>(p.x()*scale, p.y()*scale, p.z()*scale,
                                      omega, octaves);
        // Map [-~1, ~1] to [0,1] greyscale
        double t = 0.5 + 0.5 * v;
        t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        return color(t, t, t);
    }

  private:
    double scale;
    int    octaves;
    double omega;
};


// ---------------------------------------------------------------------------
// marble_texture
// Procedural marble using pbrt-v4 MarbleTexture formula:
//   marble = scale*p.y + variation * FBm(scale*p, ...)
//   t = 0.5 + 0.5 * sin(marble)
//   rgb = CubicBezier spline over pbrt-v4 marble colour knots * 1.5
//
// pbrt-v4 reference: MarbleTexture::Evaluate (textures.cpp lines 480-504)
// ---------------------------------------------------------------------------
class marble_texture : public texture {
  public:
    // scale:     spatial frequency
    // octaves:   FBm octaves
    // omega:     persistence
    // variation: amplitude of FBm displacement (pbrt-v4 default = 5)
    marble_texture(double scale = 1.0, int octaves = 8,
                   double omega = 0.5, double variation = 5.0)
        : scale(scale), octaves(octaves), omega(omega), variation(variation) {}

    color value(double /*u*/, double /*v*/, const point3& p) const override {
        // pbrt-v4: c.p *= scale; marble = c.p.y + variation * FBm(c.p, ...)
        double px = p.x() * scale, py = p.y() * scale, pz = p.z() * scale;
        double fbm_val = fbm_simple<double>(px, py, pz, omega, octaves);
        double marble = py + variation * fbm_val;
        double t = 0.5 + 0.5 * std::sin(marble);

        // pbrt-v4 marble colour spline (9 RGB control points)
        // EvaluateCubicBezier over segments; 1.5 * result
        static const double knots[9][3] = {
            {.58,.58,.60}, {.58,.58,.60}, {.58,.58,.60},
            {.50,.50,.50}, {.60,.59,.58}, {.58,.58,.60},
            {.58,.58,.60}, {.20,.20,.33}, {.58,.58,.60}
        };
        int n_seg = 6; // nSeg = sizeof(colors)/sizeof(colors[0]) - 3 = 9 - 3 = 6
        int first = (int)(t * n_seg);
        if (first >= n_seg) first = n_seg - 1;
        double lt = t * n_seg - first;

        // Cubic Bezier over 4 control points starting at knots[first]
        color rgb = cubic_bezier4(knots[first], knots[first+1],
                                  knots[first+2], knots[first+3], lt);
        // pbrt-v4: rgb *= 1.5f
        return color(
            std::min(rgb.x() * 1.5, 1.0),
            std::min(rgb.y() * 1.5, 1.0),
            std::min(rgb.z() * 1.5, 1.0));
    }

  private:
    double scale;
    int    octaves;
    double omega;
    double variation;

    // De Casteljau cubic Bezier evaluation at t in [0,1]
    static color cubic_bezier4(const double p0[3], const double p1[3],
                                const double p2[3], const double p3[3], double t) {
        double s = 1.0 - t;
        double r = s*s*s*p0[0] + 3*s*s*t*p1[0] + 3*s*t*t*p2[0] + t*t*t*p3[0];
        double g = s*s*s*p0[1] + 3*s*s*t*p1[1] + 3*s*t*t*p2[1] + t*t*t*p3[1];
        double b = s*s*s*p0[2] + 3*s*s*t*p1[2] + 3*s*t*t*p2[2] + t*t*t*p3[2];
        return color(r, g, b);
    }
};


// ---------------------------------------------------------------------------
// mix_texture
// Linear interpolation between two flat colours by a flat amount.
// pbrt-v4 reference: SpectrumMixTexture (textures.h/.cpp) - the general
// class also supports both tex1/tex2/amount bound to nested textures; this
// port is scoped to flat-literal children only (see pbrt_flatten.h's
// hasMixReflectance comment for why), matching checker_texture/
// uv_checker_texture's own established scope.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// scaled_texture -- flat scalar multiplier over another texture.
//
// AreaLightSource "diffuse"'s "scale" parameter still applies when
// "filename" is given (pbrt-v4's real DiffuseAreaLight scales the
// image-sampled radiance the same way it scales a flat "L"), but a
// filename-backed light builds a mipmap_texture directly rather than going
// through pbrt_flatten::Material::color the way flat-L lights do, so there
// is no existing place to fold the multiply in - this wraps any inner
// texture with one. No value_diff() override: its only caller
// (diffuse_light) always uses point-sampled emission for this wrapper (see
// diffuse_light's own point_sample comment), so the base class's
// value_diff()->value() default already does the right thing and stays
// reachable, unlike a dedicated override would be.
// ---------------------------------------------------------------------------
class scaled_texture : public texture {
  public:
    scaled_texture(shared_ptr<texture> inner, double scale) : inner(inner), scale(scale) {}

    color value(double u, double v, const point3& p) const override {
        if (scale == 0.0) return color(0,0,0);   // skip the inner lookup entirely -- "scale" 0 disables a light
        return scale * inner->value(u, v, p);
    }

  private:
    shared_ptr<texture> inner;
    double scale;
};


class mix_texture : public texture {
  public:
    mix_texture(const color& tex1, const color& tex2, double amount)
        : tex1(tex1), tex2(tex2), amount(amount) {}

    color value(double /*u*/, double /*v*/, const point3& /*p*/) const override {
        return (1.0 - amount) * tex1 + amount * tex2;
    }

  private:
    color tex1, tex2;
    double amount;
};


// ---------------------------------------------------------------------------
// mipmap_texture -- pbrt-v4-style MipMap-filtered image texture (§10.4)
//
// Builds a mip pyramid from the loaded image at construction time.
// value() uses trilinear filtering (no UV derivatives available via the
// basic texture interface); call value_ewa() with explicit derivatives for
// full anisotropic quality.
// ---------------------------------------------------------------------------
class mipmap_texture : public texture {
  public:
    mipmap_texture(const char* filename,
                   MipMapOptions opts = MipMapOptions{})
    {
        rtw_image img(filename);
        build_from(std::move(img), opts);
    }

    // Takes ownership of an already-loaded rtw_image instead of decoding the
    // file a second time - mirrors image_texture(rtw_image&&)'s own
    // rationale above (e.g. mesh.h's load_obj_mtl() probing a map_Kd texture
    // before committing to it).
    explicit mipmap_texture(rtw_image&& loaded, MipMapOptions opts = MipMapOptions{}) {
        build_from(std::move(loaded), opts);
    }

    // Basic interface: trilinear with zero derivatives (LOD 0)
    color value(double u, double v, const point3& /*p*/) const override {
        if (!mip_) return color(0,1,1);  // cyan debug
        u = std::max(0.0, std::min(1.0, u));
        v = 1.0 - std::max(0.0, std::min(1.0, v));  // flip V
        return mip_->filter((float)u, (float)v, 0,0, 0,0);
    }

    // Differential-aware lookup (texture base class) - routes into the EWA
    // filter below using this hit's real UV screen-space footprint. mipmap's
    // own filter() already degrades to plain bilinear-at-LOD-0 when the
    // derivatives are exactly zero (the no-differentials case - see
    // hit_record's own comment), so no separate fallback branch is needed
    // here.
    color value_diff(double u, double v, const point3& /*p*/,
                      double dudx, double dvdx, double dudy, double dvdy) const override {
        return value_ewa(u, v, dudx, dvdx, dudy, dvdy);
    }

    // Full EWA interface: caller provides UV screen-space derivatives
    color value_ewa(double u, double v,
                    double ds_dx, double dt_dx,
                    double ds_dy, double dt_dy) const {
        if (!mip_) return color(0,1,1);
        u = std::max(0.0, std::min(1.0, u));
        v = 1.0 - std::max(0.0, std::min(1.0, v));
        // v is flipped above (v_mip = 1-v), so d(v_mip)/dx = -dv/dx: negate
        // dt_dx/dt_dy to match, or the EWA ellipse's cross term (mipmap.h's
        // ewa(), built from ds*dt sums) gets the wrong sign for any sheared/
        // anisotropic footprint - exactly the grazing-angle case this filter
        // exists for. ds_dx/ds_dy (the u derivatives) are unaffected since u
        // isn't flipped.
        return mip_->filter((float)u,    (float)v,
                            (float)ds_dx,(float)-dt_dx,
                            (float)ds_dy,(float)-dt_dy);
    }

    // LOD-explicit lookup (e.g. supplied by ray differentials)
    color value_lod(double u, double v, double lod) const {
        if (!mip_) return color(0,1,1);
        u = std::max(0.0, std::min(1.0, u));
        v = 1.0 - std::max(0.0, std::min(1.0, v));
        return mip_->filter_lod((float)u, (float)v, (float)lod);
    }

    int mip_levels() const { return mip_ ? mip_->levels() : 0; }

  private:
    void build_from(rtw_image&& img, MipMapOptions opts) {
        if (img.height() <= 0) return;  // load failed — mip_ stays nullptr

        int w = img.width(), h = img.height();
        std::vector<color> pixels(w * h);
        const double scale = 1.0 / 255.0;
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                const unsigned char* p = img.pixel_data(i, j);
                pixels[j * w + i] = color(p[0]*scale, p[1]*scale, p[2]*scale);
            }
        mip_ = std::make_shared<mipmap>(pixels, w, h, opts);
    }

    std::shared_ptr<mipmap> mip_;
};


#endif
