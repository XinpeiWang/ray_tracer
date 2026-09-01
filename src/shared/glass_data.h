#pragma once

// ---------------------------------------------------------------------------
// Named dielectric ("glass") index-of-refraction presets
//
// pbrt-v4 ships exactly 7 named glass spectra a scene can bind via
// `"spectrum eta" "glass-<name>"` on a Dielectric/CoatedDiffuse/
// ThinDielectric material (spectrum.cpp's GlassBK7_eta/GlassBAF10_eta/
// GlassFK51A_eta/GlassLASF9_eta/GlassSF5_eta/GlassSF10_eta/GlassSF11_eta
// piecewise-linear tables) - note pbrt-v4's own registered STRING drops the
// "S" from Schott's "SF" series ("glass-F5"/"glass-F10"/"glass-F11", not
// "glass-SF5" etc.), even though the underlying glass and C array name both
// really are Schott SF5/SF10/SF11.
//
// This codebase's dielectric BSDF (bxdfs_dielectric.h, both GPU backends)
// takes a single scalar IOR, not a full per-wavelength spectrum - no
// spectral upsample is needed here, matching conductor_data.h's own
// RGB-flattened (not full-spectral) precedent for named metal presets. Each
// value below is the refractive index at the visual "d-line" reference
// wavelength (587.6 nm, the standard glass-catalog convention), obtained by
// linearly interpolating pbrt-v4's own tabulated {wavelength, value} pairs
// at that wavelength - cross-checked against published Schott/Ohara catalog
// nd values (e.g. BK7's interpolated 1.5168 matches the universally-cited
// N-BK7 catalog value exactly).
//
// Reference: pbrt-v4 spectrum.cpp (github.com/mmp/pbrt-v4)
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

struct GlassPreset {
	const char* name;
	float nd;   // refractive index at 587.6 nm
};

// Real Schott/Ohara glass this preset represents (only relevant to this
// comment - the string a scene actually uses is pbrt-v4's own registered
// name, kept exactly as pbrt-v4 spells it below).
static constexpr GlassPreset kGlassBK7   = {"BK7",   1.5168f};   // Schott N-BK7 (crown)
static constexpr GlassPreset kGlassBAF10 = {"BAF10", 1.6700f};   // Schott N-BAF10 (barium flint)
static constexpr GlassPreset kGlassFK51A = {"FK51A", 1.4866f};   // Schott N-FK51A (fluorophosphate crown)
static constexpr GlassPreset kGlassLASF9 = {"LASF9", 1.8503f};   // Schott N-LASF9 (lanthanum dense flint)
static constexpr GlassPreset kGlassF5    = {"F5",    1.6727f};   // Schott SF5 (dense flint)
static constexpr GlassPreset kGlassF10   = {"F10",   1.7283f};   // Schott SF10 (dense flint)
static constexpr GlassPreset kGlassF11   = {"F11",   1.7847f};   // Schott SF11 (dense flint)

// Lookup by name (the part after "glass-", e.g. "BK7" from "glass-BK7") -
// returns nullptr if not found, same convention as FindConductorPreset().
CPU_GPU const GlassPreset* FindGlassPreset(const char* name) {
	static const GlassPreset table[] = {
		kGlassBK7, kGlassBAF10, kGlassFK51A, kGlassLASF9, kGlassF5, kGlassF10, kGlassF11
	};
#if defined(__CUDACC__)
	auto streq = [](const char* a, const char* b) {
		while (*a && *b) { if (*a++ != *b++) return false; }
		return *a == *b;
	};
#else
	auto streq = [](const char* a, const char* b) {
		return std::string(a) == std::string(b);
	};
#endif
	constexpr int kCount = sizeof(table) / sizeof(table[0]);
	for (int i = 0; i < kCount; ++i)
		if (streq(table[i].name, name)) return &table[i];
	return nullptr;
}
