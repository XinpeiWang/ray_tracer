#pragma once

// ---------------------------------------------------------------------------
// Conductor RGB IOR presets (Î· + iÂ·k per channel)
//
// Values sampled from pbrt-v4's piecewise-linear spectral tables
// (src/pbrt/util/spectrum.cpp: Au_eta/Au_k, Ag_eta/Ag_k, Al_eta/Al_k, Cu_eta/Cu_k)
// at the three sRGB primary wavelengths:
//   R = 630 nm,  G = 532 nm,  B = 467 nm
//
// Sampling method: linear interpolation between the two nearest tabulated
// {wavelength, value} pairs in each array â€” identical to pbrt-v4's
// PiecewiseLinearSpectrum::operator()(lambda).
//
// Reference: pbrt-v4 Â§9.6 "Conductor BRDF", spectrum data in spectrum.cpp
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

struct ConductorPreset {
	const char* name;
	// Real part of complex IOR (Î·) per RGB channel
	float eta_r, eta_g, eta_b;
	// Imaginary part of complex IOR (k, extinction coefficient) per RGB channel
	float k_r, k_g, k_b;
};

// ---------------------------------------------------------------------------
// Table â€” sourced directly from pbrt-v4 spectrum.cpp spectral arrays
// ---------------------------------------------------------------------------

// Gold (Au) â€” warm yellow reflectance; strongly wavelength-dependent
// Î· drops sharply in the blue/green, k rises through the spectrum
static constexpr ConductorPreset kConductorAu = {
	"Au",
	0.184f,  0.457f,  1.354f,   // eta R/G/B  (630/532/467 nm)
	3.070f,  2.408f,  1.818f    // k   R/G/B
};

// Silver (Ag) â€” near-achromatic; highest reflectance of any metal
static constexpr ConductorPreset kConductorAg = {
	"Ag",
	0.130f,  0.130f,  0.137f,   // eta R/G/B
	3.963f,  3.194f,  2.640f    // k   R/G/B
};

// Aluminium (Al) â€” bright bluish-white; used for brushed metal
static constexpr ConductorPreset kConductorAl = {
	"Al",
	1.357f,  0.884f,  0.669f,   // eta R/G/B
	7.588f,  6.470f,  5.690f    // k   R/G/B
};

// Copper (Cu) â€” warm orange/red tint; k rises into the blue
static constexpr ConductorPreset kConductorCu = {
	"Cu",
	0.246f,  1.072f,  1.155f,   // eta R/G/B
	3.378f,  2.591f,  2.469f    // k   R/G/B
};

// Iron (Fe) â€” neutral dark grey; placeholder using typical published values
// (not directly in pbrt-v4 tables; values from Palik Handbook at 630/532/467 nm)
static constexpr ConductorPreset kConductorFe = {
	"Fe",
	2.940f,  2.915f,  2.773f,   // eta R/G/B
	3.080f,  2.950f,  2.813f    // k   R/G/B
};

// Lookup by name â€” returns nullptr if not found
CPU_GPU const ConductorPreset* FindConductorPreset(const char* name) {
	static const ConductorPreset table[] = {
		kConductorAu, kConductorAg, kConductorAl, kConductorCu, kConductorFe
	};
#if defined(__CUDACC__)
	// Simple strcmp equivalent for device code
	auto streq = [](const char* a, const char* b) {
		while (*a && *b) { if (*a++ != *b++) return false; }
		return *a == *b;
	};
#else
	auto streq = [](const char* a, const char* b) {
		return std::string(a) == std::string(b);
	};
#endif
	for (int i = 0; i < 5; ++i)
		if (streq(table[i].name, name)) return &table[i];
	return nullptr;
}
