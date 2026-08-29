#ifndef RTW_STB_IMAGE_H
#define RTW_STB_IMAGE_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif


#define STBI_FAILURE_USERMSG
#include "../external/stb_image.h"

#ifndef STBI_FREE
#define STBI_FREE(p) free(p)
#endif

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>


class rtw_image {
  public:
    rtw_image() {}

    // Owns two heap buffers (bdata/fdata) freed in ~rtw_image() -- copying
    // would shallow-copy those pointers and double-free on destruction, so
    // copy is disabled outright (a compile error is safer than the latent
    // UB the implicit, deprecated copy constructor would otherwise allow).
    // Move is cheap and safe (just transfers ownership), added so callers
    // can probe-then-reuse a load without decoding the same image twice --
    // see texture.h's image_texture(rtw_image&&) constructor.
    rtw_image(const rtw_image&) = delete;
    rtw_image& operator=(const rtw_image&) = delete;

    rtw_image(rtw_image&& other) noexcept
        : fdata(other.fdata), bdata(other.bdata),
          image_width(other.image_width), image_height(other.image_height),
          bytes_per_scanline(other.bytes_per_scanline)
    {
        other.fdata = nullptr;
        other.bdata = nullptr;
        other.image_width = 0;
        other.image_height = 0;
        other.bytes_per_scanline = 0;
    }

    rtw_image& operator=(rtw_image&& other) noexcept {
        if (this != &other) {
            delete[] bdata;
            STBI_FREE(fdata);
            fdata = other.fdata;
            bdata = other.bdata;
            image_width = other.image_width;
            image_height = other.image_height;
            bytes_per_scanline = other.bytes_per_scanline;
            other.fdata = nullptr;
            other.bdata = nullptr;
            other.image_width = 0;
            other.image_height = 0;
            other.bytes_per_scanline = 0;
        }
        return *this;
    }

    // gamma: the power-law decode exponent applied to an LDR (8-bit) source
    // file's raw bytes before this class ever sees them - stb_image's own
    // stbi_loadf() applies `pow(byte/255, gamma)` internally whenever the
    // file isn't already HDR (see stbi__ldr_to_hdr in stb_image.h); its own
    // process-global default is 2.2, not 1.0 (a fixed gamma-2.2 decode is
    // this codebase's own long-standing default behavior for every 8-bit
    // texture, close to but not identical to pbrt-v4's real sRGB default -
    // see pbrt_flatten::Material::textureGamma's own comment for how a
    // scene's Texture "imagemap" "string encoding" maps onto this
    // parameter). A source file that's ALREADY float/HDR (EXR via a
    // separate decode path, or .hdr/.pic via stb_image's own real HDR
    // reader) is unaffected regardless of this value - the gamma decode
    // stb_image applies here only ever triggers for genuinely 8-bit
    // sources. Mutates stb_image's process-global gamma state for the
    // duration of this one load() call, then restores it - safe only
    // because texture loading happens during single-threaded scene
    // construction (before camera::render()'s worker threads start), never
    // concurrently with another load() call.
    rtw_image(const char* image_filename, float gamma = 2.2f) : gamma_(gamma) {
        // Loads image data from the specified file. If the RTW_IMAGES environment variable is
        // defined, looks only in that directory for the image file. If the image was not found,
        // searches for the specified image file first from the current directory, then in the
        // images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
        // parent, on so on, for six levels up. If the image was not loaded successfully,
        // width() and height() will return 0.

        auto filename = std::string(image_filename);
        auto imagedir = getenv("RTW_IMAGES");

        // Hunt for the image file in some likely locations.
        if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
        if (load(filename)) return;
        if (load("images/" + filename)) return;
        if (load("../images/" + filename)) return;
        if (load("../../images/" + filename)) return;
        if (load("../../../images/" + filename)) return;
        if (load("../../../../images/" + filename)) return;
        if (load("../../../../../images/" + filename)) return;
        if (load("../../../../../../images/" + filename)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    // Takes decoded RGB float pixel data (row-major, 3 floats/pixel, [0,1]
    // linear) that came from somewhere other than stb_image - e.g. an EXR
    // decode, which this class has no format support for itself. Copies into
    // an STBI_FREE-compatible buffer (malloc, not new[]) so the destructor's
    // existing STBI_FREE(fdata) stays correct regardless of which
    // constructor built this image.
    rtw_image(int w, int h, const float* pixels) {
        if (w <= 0 || h <= 0 || !pixels) return;
        // Plain malloc, not STBI_MALLOC: that macro only exists inside
        // stb_image.h's own STB_IMAGE_IMPLEMENTATION translation unit (see
        // stb_image_impl.cpp), invisible here - but STBI_FREE(p) below (the
        // destructor, and the fallback #define a few lines up in this file)
        // is plain free(p) regardless, so a plain malloc pairs with it exactly.
        const std::size_t count = static_cast<std::size_t>(w) * h * bytes_per_pixel;
        fdata = static_cast<float*>(std::malloc(count * sizeof(float)));
        if (!fdata) return;
        std::memcpy(fdata, pixels, count * sizeof(float));
        image_width = w;
        image_height = h;
        bytes_per_scanline = image_width * bytes_per_pixel;
        convert_to_bytes();
    }

    ~rtw_image() {
        delete[] bdata;
        STBI_FREE(fdata);
    }

    bool load(const std::string& filename) {
        // Loads the image data from the given file name, gamma-decoded per
        // gamma_ (2.2 by default, this codebase's own long-standing
        // behavior - see the constructor's own comment; NOT actually
        // linear/gamma=1 despite what this comment used to claim). Returns
        // true if the load succeeded. The resulting data buffer contains
        // the three [0.0, 1.0] floating-point values for the first pixel
        // (red, then green, then blue). Pixels are contiguous, going left
        // to right for the width of the image, followed by the next row
        // below, for the full height of the image.

        auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
        // stbi_ldr_to_hdr_gamma is process-global mutable state (see the
        // constructor's own comment on why mutating it here is safe) -
        // reset back to stb_image's own real default (2.2) immediately
        // after this one load, not left at whatever gamma_ this particular
        // rtw_image happened to want, so an UNRELATED later load that
        // constructs its rtw_image the old way (relying on the default
        // parameter) still gets stb_image's genuine default rather than
        // silently inheriting this call's override.
        stbi_ldr_to_hdr_gamma(gamma_);
        fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        stbi_ldr_to_hdr_gamma(2.2f);
        if (fdata == nullptr) return false;

        bytes_per_scanline = image_width * bytes_per_pixel;
        convert_to_bytes();
        return true;
    }

    int width()  const { return (fdata == nullptr) ? 0 : image_width; }
    int height() const { return (fdata == nullptr) ? 0 : image_height; }

    const unsigned char* pixel_data(int x, int y) const {
        // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
        // data, returns magenta.
        static unsigned char magenta[] = { 255, 0, 255 };
        if (bdata == nullptr) return magenta;

        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return bdata + y*bytes_per_scanline + x*bytes_per_pixel;
    }

    // Return the raw float RGB triplet at (x,y) -- preserves HDR values > 1.
    // Returns nullptr if the image was not loaded.
    const float* float_pixel_data(int x, int y) const {
        if (fdata == nullptr) return nullptr;

        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return fdata + (y * image_width + x) * bytes_per_pixel;
    }

  private:
    const int      bytes_per_pixel = 3;
    // gamma_'s only job is being remembered from the constructor's default
    // argument until load() runs (it's needed there, not here) - see
    // load()'s own comment for what it actually does.
    const float    gamma_ = 2.2f;
    float         *fdata = nullptr;         // Linear floating point pixel data
    unsigned char *bdata = nullptr;         // Linear 8-bit pixel data
    int            image_width = 0;         // Loaded image width
    int            image_height = 0;        // Loaded image height
    int            bytes_per_scanline = 0;

    static int clamp(int x, int low, int high) {
        // Return the value clamped to the range [low, high).
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }

    static unsigned char float_to_byte(float value) {
        if (value <= 0.0)
            return 0;
        if (1.0 <= value)
            return 255;
        return static_cast<unsigned char>(256.0 * value);
    }

    void convert_to_bytes() {
        // Convert the linear floating point pixel data to bytes, storing the resulting byte
        // data in the `bdata` member.

        int total_bytes = image_width * image_height * bytes_per_pixel;
        bdata = new unsigned char[total_bytes];

        // Iterate through all pixel components, converting from [0.0, 1.0] float values to
        // unsigned [0, 255] byte values.

        auto *bptr = bdata;
        auto *fptr = fdata;
        for (auto i=0; i < total_bytes; i++, fptr++, bptr++)
            *bptr = float_to_byte(*fptr);
    }
};


// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif


#endif

