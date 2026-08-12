#ifndef PALETTE_DATA_H
#define PALETTE_DATA_H

#include <cstddef>
#include <string>
#include <vector>

// ============================================================================
// Palette data - deliberately Qt-free
// ============================================================================
// The colour tables live here, in plain structs, rather than in theme.cpp with
// the rest of the Qt code. That is not tidiness; it is what makes them
// testable.
//
// The Qt GUI is built with MinGW and the gtest binary with MSVC (see
// RayTracerGUI.pro vs tests/ray_tracer_tests.vcxproj), so anything that
// #includes a Qt header is unreachable from the test suite. Twelve palettes of
// thirty-two colours each is 384 hand-typed values, and before this split not
// one of them was checked by anything. The same reasoning already applies to
// render_output_parser.h, camera_math.h and error_codes.h - each kept Qt-free
// specifically so its rules could be pinned by tests.
//
// theme.h wraps this in QColor for the GUI's use; nothing here knows Qt exists.
// See tests/unit/palette_data_tests.cpp for what is now enforced: no unset
// field, unique ids, and a WCAG contrast floor on every text-on-surface pair.
// ============================================================================
namespace palette_data {

struct Rgb {
	unsigned char r = 0, g = 0, b = 0;
};

inline bool operator==(const Rgb &a, const Rgb &b) {
	return a.r == b.r && a.g == b.g && a.b == b.b;
}

// A colour that no palette uses on purpose (checked - none of the twelve
// contains pure black), so it doubles as the "nobody assigned this" sentinel
// that a value-initialised Rgb produces.
constexpr Rgb kUnset{0, 0, 0};

struct PaletteData {
	// std::string rather than const char*: a palette loaded from a file at
	// runtime owns its text, and making loaded and built-in palettes the same
	// type is what lets them share one registry, one lookup and one test suite.
	std::string id;
	std::string name;
	std::string origin;

	// Surface ramp, deepest first.
	Rgb surface0, surface1, surface2, surface3;

	// Text.
	Rgb textBody, textMuted, textDisabled;

	// Accents. accentPrimary drives the primary action and selection;
	// accentSecondary is the focus/active marker.
	Rgb accentPrimary, accentSecondary, accentDim;

	// The primary button's gradient, explicit rather than derived from the
	// accent: deriving it looked plausible on the purple scheme it was tuned
	// against and wrong on every other one.
	Rgb primaryTop, primaryBottom, primaryTopHover, primaryBottomHover;

	// Lines.
	Rgb border, borderStrong, hoverRow;

	// Outcome colours for the progress bar.
	Rgb success, error;

	// One colour per log severity. Explicit per theme because a palette that
	// reads well on near-black is often unreadable on white, and the log is the
	// densest coloured surface in the app.
	Rgb logInfo, logError, logWarning, logSuccess, logGpu, logCpu,
		logPerformance, logScene, logInit, logTechnique, logCommand,
		logDebug, logSeparator;

	// Optional decorative motif. Empty = none.
	std::string backgroundImage;
	bool backgroundTiled = false;
	std::string backgroundPosition = "bottom right";
};

// Every built-in palette, in menu order. The first entry is the default.
const std::vector<PaletteData> &builtins();

// ---------------------------------------------------------------------------
// Contrast, per WCAG 2.1
// ---------------------------------------------------------------------------
// Here rather than in the GUI because this is the part worth testing: two
// shipped bugs came from text that failed contrast against what it actually
// sat on - a dark olive "100%" on a dark olive progress fill, and a gold
// warning on cream. Both were found by eye, late. These make the check
// mechanical.

// Relative luminance, 0.0 (black) to 1.0 (white).
double relativeLuminance(const Rgb &c);

// Contrast ratio between two colours, 1.0 (identical) to 21.0 (black/white).
// Order does not matter. WCAG AA wants >= 4.5 for body text and >= 3.0 for
// large text and non-text indicators.
double contrastRatio(const Rgb &a, const Rgb &b);

} // namespace palette_data

#endif // PALETTE_DATA_H
