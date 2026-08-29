#ifndef MIPMAP_H
#define MIPMAP_H
//==============================================================================
// mipmap.h -- MipMap with Point / Bilinear / Trilinear / EWA filtering
//
// pbrt-v4 reference: src/pbrt/util/mipmap.h
//                    src/pbrt/util/mipmap.cpp
//
// Algorithm summary (pbrt-v4 §10.4):
//   Construction: box-filter each dimension by 2 to produce a pyramid.
//                 Level 0 = full resolution; level k = res / 2^k.
//   Texel():      nearest-neighbour with wrap (Clamp or Repeat).
//   Bilerp():     bilinear interpolation within a single level.
//   Filter():     dispatch to Point / Bilinear / Trilinear / EWA based on mode.
//   EWA():        elliptically weighted average using a 128-entry Gaussian LUT.
//                 Fits an ellipse to the two UV derivative vectors, clamps the
//                 anisotropy ratio to maxAnisotropy, then integrates the Gaussian
//                 over all texels inside the ellipse.
//
// Input: row-major RGB pixel data in [0,1] (float or from rtw_image/stb).
// Output: color (vec3) samples for use in texture evaluation.
//==============================================================================

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

// color (= vec3) is defined via rtweekend.h -> color.h -> vec3.h.
// Callers must include rtweekend.h before this header, or include it here.
#include "../TheRestOfYourLife/rtweekend.h"

// ---------------------------------------------------------------------------
// Wrap mode
// ---------------------------------------------------------------------------
// pbrt-v4's real wrap modes for Texture "imagemap" (WrapMode in pbrt-v4's
// own image.h): Repeat (its own real default), Clamp, and Black (texel()
// below returns color(0,0,0) for any out-of-bounds lookup). This struct's
// own `wrap` field default stays Clamp - not Repeat - deliberately: it's
// this codebase's own C++-level default for every caller that doesn't
// specify one, including native (non-pbrt) scenes already built against
// it, and changing THIS default would silently change their rendered
// output too. Loaded pbrt scenes get pbrt-v4's real Repeat default
// explicitly at the loader level instead - see pbrt_scene::Scene::
// TextureDecl's own "wrap" parsing and pbrt_flatten.h's Material::
// textureWrap, which resolves to "repeat" absent an explicit request and
// is threaded into a fresh MipMapOptions per pbrt-loaded texture, never
// relying on this struct's own default.
enum class MipWrapMode { Clamp, Repeat, Black };

// ---------------------------------------------------------------------------
// Filter mode  (matches pbrt-v4 FilterFunction)
// ---------------------------------------------------------------------------
enum class MipFilter { Point, Bilinear, Trilinear, EWA };

// This codebase's own long-standing default gamma-decode exponent applied
// to every 8-bit texture (stb_image's own process-global default too - see
// rtw_stb_image.h's rtw_image constructor). Named once here and reused by
// both MipMapOptions::gamma's own default below and rtw_image's own
// constructor default/reset value, rather than the same literal `2.2f`
// living independently in both files.
constexpr float kDefaultImagemapGamma = 2.2f;

// ---------------------------------------------------------------------------
// MipMap options
// ---------------------------------------------------------------------------
struct MipMapOptions {
	MipFilter   filter         = MipFilter::EWA;
	float       max_anisotropy = 8.0f;
	MipWrapMode wrap           = MipWrapMode::Clamp;
	// Texture "imagemap" "bool invert" (pbrt-v4) - not really a mip-
	// filtering option, but threaded through this same options bag since
	// it's already carried end-to-end from the loader into
	// mipmap_texture's constructor; applied once per texel at load time
	// (mipmap_texture::build_from(), src/TheRestOfYourLife/texture.h), not
	// here in mipmap.h itself.
	bool        invert         = false;
	// Texture "imagemap" "string encoding" (pbrt-v4), resolved to a gamma
	// exponent - see rtw_stb_image.h's rtw_image constructor for what this
	// actually does. kDefaultImagemapGamma (not touched by this option
	// unless a scene explicitly asks for something else) matches the
	// mipmap_texture constructor's own file-loading rtw_image call passing
	// no override.
	float       gamma          = kDefaultImagemapGamma;
};

// ---------------------------------------------------------------------------
// EWA Gaussian look-up table -- verbatim from pbrt-v4 mipmap.cpp
// 128 entries: LUT[i] = exp(-2 * (i/127)^2) - exp(-2)
// (i.e. a Gaussian evaluated at r^2 where r is normalised to the ellipse edge)
// ---------------------------------------------------------------------------
namespace mipmap_detail {

static constexpr int MIPFilterLUTSize = 128;

static const float MIPFilterLUT[MIPFilterLUTSize] = {
	0.864664733f, 0.849040031f, 0.83365953f,  0.818519294f, 0.80361563f,
	0.788944781f, 0.774503231f, 0.760287285f, 0.746293485f, 0.732518315f,
	0.718958378f, 0.705610275f, 0.692470789f, 0.679536581f, 0.666804492f,
	0.654271305f, 0.641933978f, 0.629789352f, 0.617834508f, 0.606066525f,
	0.594482362f, 0.583079159f, 0.571854174f, 0.560804546f, 0.549927592f,
	0.539220572f, 0.528680861f, 0.518305838f, 0.50809288f,  0.498039544f,
	0.488143265f, 0.478401601f, 0.468812168f, 0.45937258f,  0.450080454f,
	0.440933526f, 0.431929469f, 0.423066139f, 0.414341331f, 0.405752778f,
	0.397298455f, 0.388976216f, 0.380784035f, 0.372719884f, 0.364781618f,
	0.356967449f, 0.34927541f,  0.341703475f, 0.334249914f, 0.32691282f,
	0.319690347f, 0.312580705f, 0.305582166f, 0.298692942f, 0.291911423f,
	0.285235822f, 0.278664529f, 0.272195935f, 0.265828371f, 0.259560347f,
	0.253390193f, 0.247316495f, 0.241337672f, 0.235452279f, 0.229658857f,
	0.223955944f, 0.21834214f,  0.212816045f, 0.207376286f, 0.202021524f,
	0.196750447f, 0.191561714f, 0.186454013f, 0.181426153f, 0.176476851f,
	0.171604887f, 0.166809067f, 0.162088141f, 0.157441005f, 0.152866468f,
	0.148363426f, 0.143930718f, 0.139567271f, 0.135272011f, 0.131043866f,
	0.126881793f, 0.122784719f, 0.11875169f,  0.114781633f, 0.11087364f,
	0.107026696f, 0.103239879f, 0.0995122194f,0.0958427936f,0.0922307223f,
	0.0886750817f,0.0851749927f,0.0817295909f,0.0783380121f,0.0749994367f,
	0.0717130303f,0.0684779733f,0.0652934611f,0.0621587038f,0.0590728968f,
	0.0560353249f,0.0530452281f,0.0501018465f,0.0472044498f,0.0443523228f,
	0.0415447652f,0.0387810767f,0.0360605568f,0.0333825648f,0.0307464004f,
	0.0281514227f,0.0255970061f,0.0230824798f,0.0206072628f,0.0181707144f,
	0.0157722086f,0.013411209f, 0.0110870898f,0.0087992847f,0.0065472275f,
	0.00433036685f,0.0021481365f,0.f
};

// Safe log2 for float
inline float log2f_safe(float x) {
	if (x <= 0.0f) return -1e30f;
	return std::log2(x);
}

// Lerp between two colors
inline color lerp_color(float t, const color& a, const color& b) {
	return (1.0f - t) * a + t * b;
}

} // namespace mipmap_detail


// ---------------------------------------------------------------------------
// mipmap class
// ---------------------------------------------------------------------------
class mipmap {
public:
	// -----------------------------------------------------------------------
	// Construct from a flat row-major RGB pixel buffer.
	//   pixels : width*height colours, row-major (pixel [x,y] at pixels[y*w+x])
	//   width, height : dimensions of level 0
	//   opts  : filter mode, anisotropy cap, wrap mode
	// -----------------------------------------------------------------------
	mipmap(const std::vector<color>& pixels, int width, int height,
		   MipMapOptions opts = MipMapOptions{})
		: width0_(width), height0_(height), opts_(opts)
	{
		assert(width > 0 && height > 0);
		assert((int)pixels.size() == width * height);

		// Build level 0
		pyramid_.push_back(pixels);
		level_widths_.push_back(width);
		level_heights_.push_back(height);

		// Build pyramid with box filter (pbrt-v4: Image::GeneratePyramid)
		int w = width, h = height;
		while (w > 1 || h > 1) {
			int nw = std::max(1, w / 2);
			int nh = std::max(1, h / 2);
			std::vector<color> level(nw * nh);
			for (int y = 0; y < nh; ++y) {
				for (int x = 0; x < nw; ++x) {
					// Average 2x2 block from previous level
					int prev = (int)pyramid_.size() - 1;
					color c = texel_raw(prev, 2*x,   2*y)
							+ texel_raw(prev, 2*x+1, 2*y)
							+ texel_raw(prev, 2*x,   2*y+1)
							+ texel_raw(prev, 2*x+1, 2*y+1);
					level[y * nw + x] = c * 0.25;
				}
			}
			pyramid_.push_back(std::move(level));
			level_widths_.push_back(nw);
			level_heights_.push_back(nh);
			w = nw; h = nh;
		}
	}

	int levels()      const { return (int)pyramid_.size(); }
	int width(int lv) const { return level_widths_[lv]; }
	int height(int lv) const { return level_heights_[lv]; }

	// -----------------------------------------------------------------------
	// filter() -- main entry point, matches pbrt-v4 MIPMap::Filter<RGB>()
	//
	//   st       : UV coordinates in [0,1]
	//   dst0,dst1: partial derivatives d(st)/dx and d(st)/dy
	//              (pass {0,0},{0,0} to use trilinear with no anisotropy)
	// -----------------------------------------------------------------------
	color filter(float s, float t,
				 float ds0, float dt0,   // d(st)/dx
				 float ds1, float dt1) const // d(st)/dy
	{
		using namespace mipmap_detail;

		if (opts_.filter != MipFilter::EWA) {
			// Non-EWA: compute isotropic LOD from max derivative footprint
			float width = 2.0f * std::max({std::abs(ds0), std::abs(dt0),
										   std::abs(ds1), std::abs(dt1)});
			int nLevels = levels();
			float level = nLevels - 1 + log2f_safe(std::max(width, 1e-8f));
			if (level >= nLevels - 1)
				return texel(nLevels - 1, 0, 0);  // coarsest level
			int iLevel = std::max(0, (int)std::floor(level));

			if (opts_.filter == MipFilter::Point) {
				float lw = (float)level_widths_[iLevel];
				float lh = (float)level_heights_[iLevel];
				int si = (int)std::round(s * lw - 0.5f);
				int ti = (int)std::round(t * lh - 0.5f);
				return texel(iLevel, si, ti);
			} else if (opts_.filter == MipFilter::Bilinear) {
				return bilerp(iLevel, s, t);
			} else { // Trilinear
				if (iLevel == 0)
					return bilerp(0, s, t);
				float frac = level - iLevel;
				return lerp_color(frac, bilerp(iLevel, s, t),
										bilerp(iLevel + 1, s, t));
			}
		}

		// ---- EWA ----
		// Ensure dst0 is the longer vector (pbrt-v4)
		float len0sq = ds0*ds0 + dt0*dt0;
		float len1sq = ds1*ds1 + dt1*dt1;
		if (len0sq < len1sq) {
			std::swap(ds0, ds1); std::swap(dt0, dt1);
			std::swap(len0sq, len1sq);
		}
		float longer  = std::sqrt(len0sq);
		float shorter = std::sqrt(len1sq);

		// Clamp anisotropy ratio
		if (shorter * opts_.max_anisotropy < longer && shorter > 0.0f) {
			float scale = longer / (shorter * opts_.max_anisotropy);
			ds1 *= scale; dt1 *= scale;
			shorter *= scale;
		}
		if (shorter == 0.0f)
			return bilerp(0, s, t);

		// Choose LOD and blend two EWA evaluations
		float lod = std::max(0.0f, levels() - 1 + log2f_safe(shorter));
		int ilod  = (int)std::floor(lod);
		float frac = lod - ilod;
		return mipmap_detail::lerp_color(frac,
			ewa(ilod,     s, t, ds0, dt0, ds1, dt1),
			ewa(ilod + 1, s, t, ds0, dt0, ds1, dt1));
	}

	// Convenience: trilinear lookup with scalar LOD (0 = full res, 1 = half, …)
	color filter_lod(float s, float t, float lod) const {
		int nLevels = levels();
		lod = std::max(0.0f, std::min(lod, (float)(nLevels - 1)));
		int iLevel = (int)std::floor(lod);
		float frac = lod - iLevel;
		if (iLevel >= nLevels - 1)
			return texel(nLevels - 1, 0, 0);
		if (frac < 1e-6f)
			return bilerp(iLevel, s, t);
		return mipmap_detail::lerp_color(frac, bilerp(iLevel, s, t),
											   bilerp(iLevel + 1, s, t));
	}

private:
	// -----------------------------------------------------------------------
	// texel_raw -- unclamped access for pyramid construction
	// -----------------------------------------------------------------------
	color texel_raw(int level, int x, int y) const {
		int w = level_widths_[level];
		int h = level_heights_[level];
		x = std::max(0, std::min(x, w - 1));
		y = std::max(0, std::min(y, h - 1));
		return pyramid_[level][y * w + x];
	}

	// -----------------------------------------------------------------------
	// texel() -- with wrap mode (pbrt-v4 MIPMap::Texel)
	// -----------------------------------------------------------------------
	color texel(int level, int x, int y) const {
		int w = level_widths_[level];
		int h = level_heights_[level];
		if (opts_.wrap == MipWrapMode::Repeat) {
			x = ((x % w) + w) % w;
			y = ((y % h) + h) % h;
		} else if (opts_.wrap == MipWrapMode::Black) {
			if (x < 0 || x >= w || y < 0 || y >= h) return color(0, 0, 0);
		} else { // Clamp
			x = std::max(0, std::min(x, w - 1));
			y = std::max(0, std::min(y, h - 1));
		}
		return pyramid_[level][y * w + x];
	}

	// -----------------------------------------------------------------------
	// bilerp() -- bilinear interpolation (pbrt-v4 MIPMap::Bilerp)
	// -----------------------------------------------------------------------
	color bilerp(int level, float s, float t) const {
		if (level >= levels()) level = levels() - 1;
		// s*w/t*h computed in double, not float: with a wide UV tiling range
		// (Texture "imagemap"'s "wrap" "repeat" - see texture.h's
		// wide_clamp(), which allows s/t up to ~1024) a large texture (tens
		// of thousands of texels wide) times a large s/t can exceed float's
		// exact-integer range (2^24), losing sub-texel precision right
		// before the floor() that picks x0/y0 - a double keeps this exact
		// for any realistic texture size and tiling range.
		double w = (double)level_widths_[level];
		double h = (double)level_heights_[level];

		double x = (double)s * w - 0.5;
		double y = (double)t * h - 0.5;
		int x0 = (int)std::floor(x), x1 = x0 + 1;
		int y0 = (int)std::floor(y), y1 = y0 + 1;
		float fx = (float)(x - x0), fy = (float)(y - y0);

		color c00 = texel(level, x0, y0);
		color c10 = texel(level, x1, y0);
		color c01 = texel(level, x0, y1);
		color c11 = texel(level, x1, y1);

		return (1-fy)*((1-fx)*c00 + fx*c10)
			 +    fy *((1-fx)*c01 + fx*c11);
	}

	// -----------------------------------------------------------------------
	// ewa() -- elliptically weighted average (pbrt-v4 MIPMap::EWA)
	// -----------------------------------------------------------------------
	color ewa(int level, float s, float t,
			  float ds0, float dt0, float ds1, float dt1) const
	{
		using namespace mipmap_detail;

		if (level >= levels()) return texel(levels() - 1, 0, 0);

		float lw = (float)level_widths_[level];
		float lh = (float)level_heights_[level];

		// Scale UV to texel coordinates
		s = s * lw - 0.5f;
		t = t * lh - 0.5f;
		ds0 *= lw; dt0 *= lh;
		ds1 *= lw; dt1 *= lh;

		// Compute ellipse coefficients A,B,C (pbrt-v4 §10.4.4)
		float A = dt0*dt0 + dt1*dt1 + 1;
		float B = -2*(ds0*dt0 + ds1*dt1);
		float C = ds0*ds0 + ds1*ds1 + 1;
		float invF = 1.0f / (A*C - B*B*0.25f);
		A *= invF; B *= invF; C *= invF;

		// Ellipse bounding box
		float det    = -B*B + 4*A*C;
		float invDet = 1.0f / det;
		float uSqrt  = std::sqrt(std::max(0.0f, det * C));
		float vSqrt  = std::sqrt(std::max(0.0f, A * det));

		int s0 = (int)std::ceil (s - 2*invDet*uSqrt);
		int s1 = (int)std::floor(s + 2*invDet*uSqrt);
		int t0 = (int)std::ceil (t - 2*invDet*vSqrt);
		int t1 = (int)std::floor(t + 2*invDet*vSqrt);

		// Accumulate Gaussian-weighted texels inside ellipse
		color sum(0,0,0);
		float sum_wts = 0.0f;
		for (int it = t0; it <= t1; ++it) {
			float tt = it - t;
			for (int is = s0; is <= s1; ++is) {
				float ss = is - s;
				float r2 = A*ss*ss + B*ss*tt + C*tt*tt;
				if (r2 < 1.0f) {
					int idx = std::min((int)(r2 * MIPFilterLUTSize),
									   MIPFilterLUTSize - 1);
					float wt = MIPFilterLUT[idx];
					sum     = sum + wt * texel(level, is, it);
					sum_wts += wt;
				}
			}
		}
		if (sum_wts == 0.0f) return bilerp(level, s / lw + 0.5f/lw,
												  t / lh + 0.5f/lh);
		return sum / sum_wts;
	}

	// -----------------------------------------------------------------------
	// Data
	// -----------------------------------------------------------------------
	int width0_, height0_;
	MipMapOptions opts_;
	std::vector<std::vector<color>> pyramid_;    // pyramid_[level][y*w + x]
	std::vector<int> level_widths_;
	std::vector<int> level_heights_;
};


#endif // MIPMAP_H
