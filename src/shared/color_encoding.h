#pragma once
// ---------------------------------------------------------------------------
// color_encoding.h -- Color space encoding / decoding (pbrt-v4 color.h port)
//
// pbrt-v4 reference: src/pbrt/util/color.h + color.cpp
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides:
//   LinearColorEncoding  -- identity encoding (uint8 / float passthrough)
//   sRGBColorEncoding    -- IEC 61966-2-1 sRGB OETF / EOTF
//   GammaColorEncoding   -- arbitrary gamma with forward/inverse LUTs
//   ColorEncoding        -- std::variant dispatcher over the three above
//
//   Free functions (mirrors pbrt-v4 color.h):
//     LinearToSRGB(float)       -- linear float -> sRGB float  [0,1]
//     LinearToSRGB8(float)      -- linear float -> uint8 with dither
//     SRGBToLinear(float)       -- sRGB float -> linear float
//     SRGB8ToLinear(uint8_t)    -- sRGB uint8 -> linear float via LUT
//
// Design notes:
//   - Header-only; no allocator, no TaggedPointer, no pbrt infrastructure.
//   - ColorEncoding uses std::variant for dispatch (replaces TaggedPointer).
//   - SRGBToLinearLUT[256] is the exact table from pbrt-v4 color.cpp.
//   - LinearToSRGB uses the minimax polynomial from pbrt-v4 (enoki-derived).
//   - std::span<> used for batch encode/decode (C++20 compatible, C++17 with
//     a minimal fallback span provided below).
//   - float throughout (matching pbrt-v4 Float=float build).
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>
#include <variant>
#include <algorithm>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

// ---------------------------------------------------------------------------
// Minimal span shim (avoids requiring C++20 or including heavy headers)
// Only supports contiguous data pointer + size, matching pstd::span usage.
// ---------------------------------------------------------------------------
namespace ce_detail {

template<typename T>
struct Span {
	T*     data_ = nullptr;
	size_t size_ = 0;
	Span() = default;
	Span(T* d, size_t n) : data_(d), size_(n) {}
	T&     operator[](size_t i) const { return data_[i]; }
	size_t size()               const { return size_; }
	T*     begin()              const { return data_; }
	T*     end()                const { return data_ + size_; }
};

// ---------------------------------------------------------------------------
// SRGBToLinearLUT[256]
// Exact table from pbrt-v4 src/pbrt/util/color.cpp.
// Maps an 8-bit sRGB value (index 0..255) to linear-light float.
// ---------------------------------------------------------------------------
static constexpr float SRGBToLinearLUT[256] = {
	0.0000000000f, 0.0003035270f, 0.0006070540f, 0.0009105810f,
	0.0012141080f, 0.0015176350f, 0.0018211619f, 0.0021246888f,
	0.0024282159f, 0.0027317430f, 0.0030352699f, 0.0033465356f,
	0.0036765069f, 0.0040247170f, 0.0043914421f, 0.0047769533f,
	0.0051815170f, 0.0056053917f, 0.0060488326f, 0.0065120910f,
	0.0069954102f, 0.0074990317f, 0.0080231922f, 0.0085681248f,
	0.0091340570f, 0.0097212177f, 0.0103298230f, 0.0109600937f,
	0.0116122449f, 0.0122864870f, 0.0129830306f, 0.0137020806f,
	0.0144438436f, 0.0152085144f, 0.0159962922f, 0.0168073755f,
	0.0176419523f, 0.0185002182f, 0.0193823613f, 0.0202885624f,
	0.0212190095f, 0.0221738834f, 0.0231533647f, 0.0241576303f,
	0.0251868572f, 0.0262412224f, 0.0273208916f, 0.0284260381f,
	0.0295568332f, 0.0307134409f, 0.0318960287f, 0.0331047624f,
	0.0343398079f, 0.0356013142f, 0.0368894450f, 0.0382043645f,
	0.0395462364f, 0.0409151986f, 0.0423114114f, 0.0437350273f,
	0.0451862030f, 0.0466650836f, 0.0481718220f, 0.0497065634f,
	0.0512694679f, 0.0528606549f, 0.0544802807f, 0.0561284944f,
	0.0578054339f, 0.0595112406f, 0.0612460710f, 0.0630100295f,
	0.0648032799f, 0.0666259527f, 0.0684781820f, 0.0703601092f,
	0.0722718611f, 0.0742135793f, 0.0761853904f, 0.0781874284f,
	0.0802198276f, 0.0822827145f, 0.0843762159f, 0.0865004659f,
	0.0886556059f, 0.0908417329f, 0.0930589810f, 0.0953074843f,
	0.0975873619f, 0.0998987406f, 0.1022417471f, 0.1046164930f,
	0.1070231125f, 0.1094617173f, 0.1119324341f, 0.1144353822f,
	0.1169706732f, 0.1195384338f, 0.1221387982f, 0.1247718409f,
	0.1274376959f, 0.1301364899f, 0.1328683347f, 0.1356333494f,
	0.1384316236f, 0.1412633061f, 0.1441284865f, 0.1470272839f,
	0.1499598026f, 0.1529261619f, 0.1559264660f, 0.1589608639f,
	0.1620294005f, 0.1651322246f, 0.1682693958f, 0.1714410931f,
	0.1746473908f, 0.1778884083f, 0.1811642349f, 0.1844749898f,
	0.1878207624f, 0.1912016720f, 0.1946178079f, 0.1980693042f,
	0.2015562356f, 0.2050787061f, 0.2086368501f, 0.2122307271f,
	0.2158605307f, 0.2195262313f, 0.2232279778f, 0.2269658893f,
	0.2307400703f, 0.2345506549f, 0.2383976579f, 0.2422811985f,
	0.2462013960f, 0.2501583695f, 0.2541521788f, 0.2581829131f,
	0.2622507215f, 0.2663556635f, 0.2704978585f, 0.2746773660f,
	0.2788943350f, 0.2831487954f, 0.2874408960f, 0.2917706966f,
	0.2961383164f, 0.3005438447f, 0.3049873710f, 0.3094689548f,
	0.3139887452f, 0.3185468316f, 0.3231432438f, 0.3277781308f,
	0.3324515820f, 0.3371636569f, 0.3419144452f, 0.3467040956f,
	0.3515326977f, 0.3564002514f, 0.3613068759f, 0.3662526906f,
	0.3712377846f, 0.3762622178f, 0.3813261092f, 0.3864295185f,
	0.3915725648f, 0.3967553079f, 0.4019778669f, 0.4072403014f,
	0.4125427008f, 0.4178851545f, 0.4232677519f, 0.4286905527f,
	0.4341537058f, 0.4396572411f, 0.4452012479f, 0.4507858455f,
	0.4564110637f, 0.4620770514f, 0.4677838385f, 0.4735315442f,
	0.4793202281f, 0.4851499796f, 0.4910208881f, 0.4969330430f,
	0.5028865933f, 0.5088814497f, 0.5149177909f, 0.5209956765f,
	0.5271152258f, 0.5332764983f, 0.5394796133f, 0.5457245708f,
	0.5520114899f, 0.5583404899f, 0.5647116303f, 0.5711249113f,
	0.5775805116f, 0.5840784907f, 0.5906189084f, 0.5972018838f,
	0.6038274169f, 0.6104956269f, 0.6172066331f, 0.6239604354f,
	0.6307572126f, 0.6375969648f, 0.6444797516f, 0.6514056921f,
	0.6583748460f, 0.6653873324f, 0.6724432111f, 0.6795425415f,
	0.6866854429f, 0.6938719153f, 0.7011020184f, 0.7083759308f,
	0.7156936526f, 0.7230552435f, 0.7304608822f, 0.7379105687f,
	0.7454043627f, 0.7529423237f, 0.7605246305f, 0.7681512833f,
	0.7758223414f, 0.7835379243f, 0.7912980318f, 0.7991028428f,
	0.8069523573f, 0.8148466945f, 0.8227858543f, 0.8307699561f,
	0.8387991190f, 0.8468732834f, 0.8549926877f, 0.8631572723f,
	0.8713672161f, 0.8796223402f, 0.8879231811f, 0.8962693810f,
	0.9046613574f, 0.9130986929f, 0.9215820432f, 0.9301108718f,
	0.9386858940f, 0.9473065734f, 0.9559735060f, 0.9646862745f,
	0.9734454751f, 0.9822505713f, 0.9911022186f, 1.0000000000f
};

} // namespace ce_detail

// ---------------------------------------------------------------------------
// Free scalar conversion functions
// Mirrors pbrt-v4 color.h: LinearToSRGB, LinearToSRGB8, SRGBToLinear, SRGB8ToLinear
// ---------------------------------------------------------------------------

// LinearToSRGB: linear float -> sRGB float
// Mirrors pbrt-v4 LinearToSRGB exactly -- no negative clamp in scalar fn.
// (Clamping occurs in LinearToSRGB8 via uint8 output range.)
CPU_GPU float LinearToSRGB(float value) {
	if (value <= 0.0031308f)
		return 12.92f * value;
	// Minimax rational approximation (pbrt-v4 / enoki)
	float s = std::sqrt(value);
	float p = -0.0016829072605308378f
			+ s * (  0.03453868659826638f
			+ s * (  0.7642611304733891f
			+ s * (  2.0041169284241644f
			+ s * (  0.7551545191665577f
			+ s * (-0.016202083165206348f)))));
	float q =  4.178892964897981e-7f
			+ s * (-0.00004375359692957097f
			+ s * (  0.03467195408529984f
			+ s * (  0.6085338522168684f
			+ s * (  1.8970238036421054f
			+ s))));
	float result = p / q * value;
	return result;
}

// LinearToSRGB8: linear float -> uint8, with optional dither in [-0.5, 0.5]
// Mirrors pbrt-v4 LinearToSRGB8(Float value, Float dither = 0).
CPU_GPU uint8_t LinearToSRGB8(float value, float dither = 0.f) {
	if (value <= 0.f) return 0;
	if (value >= 1.f) return 255;
	float v = LinearToSRGB(value) * 255.f + dither + 0.5f;
	int i = (int)v;
	return (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// SRGBToLinear: sRGB float -> linear float
// Minimax polynomial from enoki's color.h (same as pbrt-v4).
CPU_GPU float SRGBToLinear(float value) {
	if (value <= 0.04045f)
		return value * (1.f / 12.92f);
	float p = -0.0163933279112946f
			+ value * (  -0.7386328024653209f
			+ value * ( -11.199318357635072f
			+ value * ( -47.46726633009393f
			+ value *   -36.04572663838034f)));
	float q = -0.004261480793199332f
			+ value * ( -19.140923959601675f
			+ value * ( -59.096406619244426f
			+ value * ( -18.225745396846637f
			+ value)));
	return p / q * value;
}

// SRGB8ToLinear: sRGB uint8 -> linear float via precomputed LUT
// Mirrors pbrt-v4 SRGB8ToLinear(uint8_t value).
CPU_GPU float SRGB8ToLinear(uint8_t value) {
	return ce_detail::SRGBToLinearLUT[value];
}

// ---------------------------------------------------------------------------
// LinearColorEncoding
//
// Identity encoding: uint8 -> float via /255, float -> uint8 via *255+clamp.
// Mirrors pbrt-v4 LinearColorEncoding.
// ---------------------------------------------------------------------------
class LinearColorEncoding {
public:
	// Decode batch: uint8 -> linear float (divide by 255)
	void ToLinear(ce_detail::Span<const uint8_t> vin,
				  ce_detail::Span<float> vout) const {
		for (size_t i = 0; i < vin.size(); ++i)
			vout[i] = vin[i] / 255.f;
	}

	// Scalar version
	CPU_GPU float ToFloatLinear(float v) const { return v; }

	// Encode batch: linear float -> uint8 (clamp, *255, round)
	void FromLinear(ce_detail::Span<const float> vin,
					ce_detail::Span<uint8_t> vout) const {
		for (size_t i = 0; i < vin.size(); ++i) {
			float v = vin[i] * 255.f + 0.5f;
			int   iv = (int)v;
			vout[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
		}
	}
};

// ---------------------------------------------------------------------------
// sRGBColorEncoding
//
// sRGB OETF/EOTF: uses SRGBToLinearLUT for fast uint8 decode and
// LinearToSRGB8 for float encode.
// Mirrors pbrt-v4 sRGBColorEncoding.
// ---------------------------------------------------------------------------
class sRGBColorEncoding {
public:
	// Decode batch: sRGB uint8 -> linear float via LUT
	void ToLinear(ce_detail::Span<const uint8_t> vin,
				  ce_detail::Span<float> vout) const {
		for (size_t i = 0; i < vin.size(); ++i)
			vout[i] = SRGB8ToLinear(vin[i]);
	}

	// Scalar decode: sRGB float -> linear float
	CPU_GPU float ToFloatLinear(float v) const { return SRGBToLinear(v); }

	// Encode batch: linear float -> sRGB uint8
	void FromLinear(ce_detail::Span<const float> vin,
					ce_detail::Span<uint8_t> vout) const {
		for (size_t i = 0; i < vin.size(); ++i)
			vout[i] = LinearToSRGB8(vin[i]);
	}
};

// ---------------------------------------------------------------------------
// GammaColorEncoding
//
// Arbitrary gamma curve with precomputed LUTs for fast batch conversion.
// applyLUT[256]: uint8 -> float (forward, gamma applied)
// inverseLUT[1024]: float -> uint8 (inverse, for FromLinear)
// Mirrors pbrt-v4 GammaColorEncoding.
// ---------------------------------------------------------------------------
class GammaColorEncoding {
public:
	static constexpr int kInverseLUTSize = 1024;

	explicit GammaColorEncoding(float gamma) : gamma_(gamma) {
		// Build forward LUT: applyLUT[i] = (i/255)^gamma
		for (int i = 0; i < 256; ++i) {
			float v = float(i) / 255.f;
			applyLUT_[i] = std::pow(v, gamma_);
		}
		// Build inverse LUT: inverseLUT[i] = round(255 * (i/(size-1))^(1/gamma))
		for (int i = 0; i < kInverseLUTSize; ++i) {
			float v = float(i) / float(kInverseLUTSize - 1);
			float enc = std::pow(v, 1.f / gamma_) * 255.f + 0.5f;
			int iv = (int)enc;
			inverseLUT_[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
		}
	}

	// Decode batch: gamma-encoded uint8 -> linear float via LUT
	void ToLinear(ce_detail::Span<const uint8_t> vin,
				  ce_detail::Span<float> vout) const {
		for (size_t i = 0; i < vin.size(); ++i)
			vout[i] = applyLUT_[vin[i]];
	}

	// Scalar decode
	CPU_GPU float ToFloatLinear(float v) const {
		return (v <= 0.f) ? 0.f : std::pow(v, gamma_);
	}

	// Encode batch: linear float -> gamma-encoded uint8 (analytical, matches pbrt-v4)
	void FromLinear(ce_detail::Span<const float> vin,
					ce_detail::Span<uint8_t> vout) const {
		for (size_t i = 0; i < vin.size(); ++i) {
			float v = vin[i];
			float enc = (v <= 0.f) ? 0.f : std::pow(v, 1.f / gamma_) * 255.f + 0.5f;
			int   iv  = (int)enc;
			vout[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
		}
	}

	float gamma() const { return gamma_; }

	// Test accessor for the forward LUT
	const std::array<float, 256>& applyLUT() const { return applyLUT_; }

private:
	float   gamma_;
	std::array<float,   256>             applyLUT_;
	std::array<uint8_t, kInverseLUTSize> inverseLUT_;
};

// ---------------------------------------------------------------------------
// ColorEncoding
//
// std::variant dispatcher over the three concrete encoding types.
// Mirrors pbrt-v4 ColorEncoding (replaces TaggedPointer with std::variant).
//
// Named singletons:
//   ColorEncoding::Linear()  -- LinearColorEncoding
//   ColorEncoding::sRGB()    -- sRGBColorEncoding
//   ColorEncoding::Gamma(g)  -- GammaColorEncoding(g)
// ---------------------------------------------------------------------------
class ColorEncoding {
public:
	using VariantType = std::variant<LinearColorEncoding,
									 sRGBColorEncoding,
									 GammaColorEncoding>;

	/* implicit */ ColorEncoding(LinearColorEncoding e) : v_(std::move(e)) {}
	/* implicit */ ColorEncoding(sRGBColorEncoding   e) : v_(std::move(e)) {}
	/* implicit */ ColorEncoding(GammaColorEncoding  e) : v_(std::move(e)) {}

	// Named constructors
	static ColorEncoding Linear() { return ColorEncoding{LinearColorEncoding{}}; }
	static ColorEncoding sRGB()   { return ColorEncoding{sRGBColorEncoding{}}; }
	static ColorEncoding Gamma(float g) { return ColorEncoding{GammaColorEncoding{g}}; }

	// Decode batch: encoding-specific uint8 -> linear float
	void ToLinear(ce_detail::Span<const uint8_t> vin,
				  ce_detail::Span<float> vout) const {
		std::visit([&](const auto& enc) { enc.ToLinear(vin, vout); }, v_);
	}

	// Encode batch: linear float -> encoding-specific uint8
	void FromLinear(ce_detail::Span<const float> vin,
					ce_detail::Span<uint8_t> vout) const {
		std::visit([&](const auto& enc) { enc.FromLinear(vin, vout); }, v_);
	}

	// Scalar decode: encoding float -> linear float
	CPU_GPU float ToFloatLinear(float v) const {
		return std::visit([&](const auto& enc) { return enc.ToFloatLinear(v); }, v_);
	}

	const VariantType& AsVariant() const { return v_; }

private:
	VariantType v_;
};
