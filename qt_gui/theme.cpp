#include "theme.h"

namespace theme {

QColor Palette::colourFor(render_output::LogSeverity severity) const {
	using S = render_output::LogSeverity;
	switch (severity) {
	case S::Info:        return logInfo;
	case S::Error:       return logError;
	case S::Warning:     return logWarning;
	case S::Success:     return logSuccess;
	case S::Gpu:         return logGpu;
	case S::Cpu:         return logCpu;
	case S::Performance: return logPerformance;
	case S::Scene:       return logScene;
	case S::Init:        return logInit;
	case S::Technique:   return logTechnique;
	case S::Command:     return logCommand;
	case S::Debug:       return logDebug;
	case S::Separator:   return logSeparator;
	}
	return logInfo;
}

// ============================================================================
// Palettes
// ============================================================================
// Values are transcribed from each scheme's own published palette, verified
// against the upstream source rather than a blog post. Only NUMBERS are used -
// no files, artwork or code from those projects - which matters for Breeze
// (LGPL-2.0-or-later) and Qt Creator (GPL-3.0 + Qt Company exception), whose
// palette FILES could not be vendored into this project but whose colour
// values can be retyped. See THEMES.md for per-scheme attribution.
//
// Several schemes genuinely lack a value for a role this UI needs - Solarized
// ships only two background tones per mode and no disabled text; Dracula has
// nothing darker than its single background; Breeze computes disabled text and
// borders at paint time rather than storing them. Wherever that happens the
// value is DERIVED here and marked as such, so nobody later mistakes an
// invention of mine for part of the upstream scheme.
// ============================================================================

namespace {

// The scheme this project shipped with. Preserved value-for-value so switching
// away and back is a genuine round trip rather than an approximation.
Palette cyberpunk() {
	Palette p;
	p.id = "cyberpunk";
	p.name = "Cyberpunk";
	p.origin = "This project's original palette";
	p.isDark = true;

	p.surface0 = "#0E0E14";
	p.surface1 = "#16161F";
	p.surface2 = "#1E1E2A";
	p.surface3 = "#2A2A3A";

	p.textBody = "#E6E6F0";
	p.textMuted = "#9A9AB0";
	p.textDisabled = "#5A5A70";

	p.accentPrimary = "#FF3DFF";
	p.accentSecondary = "#00E5FF";
	p.accentDim = "#C93FE8";
	p.primaryTop = "#3A1050";
	p.primaryBottom = "#240C38";
	p.primaryTopHover = "#4E1668";
	p.primaryBottomHover = "#341048";

	p.border = "#2A2A3A";
	p.borderStrong = "#3A3A50";
	p.hoverRow = "#3A2E56";

	p.success = "#5AAA3C";
	p.error = "#DF4F4F";

	p.logInfo = "#D0D0D0";
	p.logError = "#FF6B6B";
	p.logWarning = "#FFD700";
	p.logSuccess = "#51CF66";
	p.logGpu = "#A9E34B";
	p.logCpu = "#74C0FC";
	p.logPerformance = "#FFA94D";
	p.logScene = "#CC99FF";
	p.logInit = "#63E6BE";
	p.logTechnique = "#E599F7";
	p.logCommand = "#CCCCCC";
	p.logDebug = "#A8A8A8";
	p.logSeparator = "#555555";
	return p;
}

// Dracula - MIT (c) 2023 Dracula Theme. Values from the project README's
// "Color Palette (OSS)" table. The commercial Dracula PRO palette is a
// separate, non-MIT thing and is deliberately not used.
Palette dracula() {
	Palette p;
	p.id = "dracula";
	p.name = "Dracula";
	p.origin = "Dracula (MIT) - draculatheme.com";
	p.isDark = true;

	// DERIVED: the spec has exactly one background (#282a36) and nothing
	// darker, but this UI needs a window tone beneath its panels. Darkened
	// from the canonical background rather than borrowing #21222c, which
	// several Dracula UI ports use but which does NOT appear in the spec.
	p.surface0 = QColor("#282a36").darker(118);
	p.surface1 = "#282a36";   // Background
	p.surface2 = "#44475a";   // Current Line / Selection
	p.surface3 = QColor("#44475a").lighter(115);  // DERIVED

	p.textBody = "#f8f8f2";   // Foreground
	p.textMuted = "#6272a4";  // Comment
	p.textDisabled = QColor("#6272a4").darker(130);  // DERIVED: no such value

	p.accentPrimary = "#bd93f9";   // Purple
	p.accentSecondary = "#8be9fd"; // Cyan
	p.accentDim = "#ff79c6";       // Pink
	p.primaryTop = QColor("#bd93f9").darker(260);
	p.primaryBottom = QColor("#bd93f9").darker(330);
	p.primaryTopHover = QColor("#bd93f9").darker(210);
	p.primaryBottomHover = QColor("#bd93f9").darker(270);

	p.border = "#44475a";
	p.borderStrong = "#6272a4";
	p.hoverRow = QColor("#44475a").lighter(120);

	p.success = "#50fa7b";
	p.error = "#ff5555";

	p.logInfo = "#f8f8f2";
	p.logError = "#ff5555";
	p.logWarning = "#f1fa8c";
	p.logSuccess = "#50fa7b";
	p.logGpu = "#50fa7b";
	p.logCpu = "#8be9fd";
	p.logPerformance = "#ffb86c";
	p.logScene = "#bd93f9";
	p.logInit = "#8be9fd";
	p.logTechnique = "#ff79c6";
	p.logCommand = "#f8f8f2";
	p.logDebug = "#6272a4";
	p.logSeparator = "#6272a4";
	return p;
}

// Nord - MIT (c) 2016-present Sven Greb. Values from src/nord.scss, whose own
// doc comments assign nord0/1/2 to backgrounds, nord3 to disabled UI, nord4 to
// text, and explicitly name nord8 "the accent color of the color palette".
Palette nord() {
	Palette p;
	p.id = "nord";
	p.name = "Nord";
	p.origin = "Nord (MIT) - nordtheme.com";
	p.isDark = true;

	p.surface0 = "#2e3440";   // nord0
	p.surface1 = "#3b4252";   // nord1
	p.surface2 = "#434c5e";   // nord2
	p.surface3 = "#4c566a";   // nord3

	p.textBody = "#d8dee9";   // nord4
	// DERIVED: Nord assigns nord3 to BOTH comments and disabled UI, so muted
	// and disabled would otherwise be the same colour and the text hierarchy
	// would collapse. Muted is lifted toward nord4.
	p.textMuted = QColor("#4c566a").lighter(150);
	p.textDisabled = "#4c566a";  // nord3, its documented disabled role

	p.accentPrimary = "#88c0d0";   // nord8, the documented accent
	p.accentSecondary = "#8fbcbb"; // nord7
	p.accentDim = "#5e81ac";       // nord10
	p.primaryTop = "#3b4252";
	p.primaryBottom = "#2e3440";
	p.primaryTopHover = "#434c5e";
	p.primaryBottomHover = "#3b4252";

	p.border = "#434c5e";
	p.borderStrong = "#4c566a";
	p.hoverRow = "#434c5e";

	p.success = "#a3be8c";   // nord14
	p.error = "#bf616a";     // nord11

	p.logInfo = "#d8dee9";
	p.logError = "#bf616a";
	p.logWarning = "#ebcb8b";   // nord13
	p.logSuccess = "#a3be8c";
	p.logGpu = "#a3be8c";
	p.logCpu = "#88c0d0";
	p.logPerformance = "#d08770";  // nord12
	p.logScene = "#b48ead";        // nord15
	p.logInit = "#8fbcbb";
	p.logTechnique = "#81a1c1";    // nord9
	p.logCommand = "#d8dee9";
	p.logDebug = QColor("#4c566a").lighter(150);
	p.logSeparator = "#4c566a";
	return p;
}

// Gruvbox Dark - MIT/X11, Pavel Pertsev. Values from colors/gruvbox.vim.
// The best-covered of these schemes: it has a real value for every role.
Palette gruvbox() {
	Palette p;
	p.id = "gruvbox";
	p.name = "Gruvbox Dark";
	p.origin = "Gruvbox (MIT/X11) - github.com/morhetz/gruvbox";
	p.isDark = true;

	p.surface0 = "#1d2021";   // dark0_hard
	p.surface1 = "#282828";   // dark0 / bg0
	p.surface2 = "#3c3836";   // dark1 / bg1
	p.surface3 = "#504945";   // dark2 / bg2

	p.textBody = "#ebdbb2";     // light1 / fg1, the Normal foreground
	p.textMuted = "#928374";    // gray_245, the Comment colour
	p.textDisabled = "#7c6f64"; // dark4 / bg4, used for LineNr

	p.accentPrimary = "#fabd2f";   // bright_yellow
	p.accentSecondary = "#83a598"; // bright_blue
	p.accentDim = "#fe8019";       // bright_orange
	p.primaryTop = "#504945";
	p.primaryBottom = "#3c3836";
	p.primaryTopHover = "#665c54";
	p.primaryBottomHover = "#504945";

	p.border = "#504945";       // bg2
	p.borderStrong = "#665c54"; // bg3
	p.hoverRow = "#504945";

	p.success = "#b8bb26";   // bright_green
	p.error = "#fb4934";     // bright_red

	p.logInfo = "#ebdbb2";
	p.logError = "#fb4934";
	p.logWarning = "#fabd2f";
	p.logSuccess = "#b8bb26";
	p.logGpu = "#b8bb26";
	p.logCpu = "#83a598";
	p.logPerformance = "#fe8019";
	p.logScene = "#d3869b";   // bright_purple
	p.logInit = "#8ec07c";    // bright_aqua
	p.logTechnique = "#d3869b";
	p.logCommand = "#ebdbb2";
	p.logDebug = "#928374";
	p.logSeparator = "#665c54";
	return p;
}

// Solarized Dark - MIT (c) 2011 Ethan Schoonover. Values from the repo README
// palette table. Per the spec, body text on a dark background is base0 (NOT
// base00); base00/base01 are used here for the muted and disabled steps, which
// keeps the whole text ramp inside the canonical palette.
Palette solarizedDark() {
	Palette p;
	p.id = "solarized-dark";
	p.name = "Solarized Dark";
	p.origin = "Solarized (MIT) - Ethan Schoonover";
	p.isDark = true;

	p.surface0 = "#002b36";   // base03
	p.surface1 = "#073642";   // base02
	// DERIVED: Solarized ships exactly two background tones per mode. There is
	// no third "raised surface" value; going further leaves the palette.
	p.surface2 = QColor("#073642").lighter(125);
	p.surface3 = QColor("#073642").lighter(150);

	p.textBody = "#839496";     // base0
	p.textMuted = "#657b83";    // base00
	p.textDisabled = "#586e75"; // base01

	p.accentPrimary = "#268bd2";   // blue
	p.accentSecondary = "#2aa198"; // cyan
	p.accentDim = "#6c71c4";       // violet
	p.primaryTop = "#073642";
	p.primaryBottom = "#002b36";
	p.primaryTopHover = QColor("#073642").lighter(130);
	p.primaryBottomHover = "#073642";

	p.border = "#073642";
	p.borderStrong = "#586e75";
	p.hoverRow = QColor("#073642").lighter(135);

	p.success = "#859900";   // green
	p.error = "#dc322f";     // red

	p.logInfo = "#839496";
	p.logError = "#dc322f";
	p.logWarning = "#b58900";   // yellow
	p.logSuccess = "#859900";
	p.logGpu = "#859900";
	p.logCpu = "#268bd2";
	p.logPerformance = "#cb4b16";  // orange
	p.logScene = "#6c71c4";        // violet
	p.logInit = "#2aa198";         // cyan
	p.logTechnique = "#d33682";    // magenta
	p.logCommand = "#93a1a1";      // base1
	p.logDebug = "#657b83";
	p.logSeparator = "#586e75";
	return p;
}

// Solarized Light - same MIT palette, base ramp inverted. Solarized's accents
// are deliberately shared between the two modes, so only the greys flip.
// The log colours matter most here: this is the one scheme where a bright
// on-black log palette would be unreadable.
Palette solarizedLight() {
	Palette p;
	p.id = "solarized-light";
	p.name = "Solarized Light";
	p.origin = "Solarized (MIT) - Ethan Schoonover";
	p.isDark = false;

	p.surface0 = "#fdf6e3";   // base3
	p.surface1 = "#eee8d5";   // base2
	p.surface2 = QColor("#eee8d5").darker(104);  // DERIVED, see Solarized Dark
	p.surface3 = QColor("#eee8d5").darker(110);

	p.textBody = "#657b83";     // base00
	p.textMuted = "#839496";    // base0
	p.textDisabled = "#93a1a1"; // base1

	p.accentPrimary = "#268bd2";
	p.accentSecondary = "#2aa198";
	p.accentDim = "#6c71c4";
	p.primaryTop = "#eee8d5";
	p.primaryBottom = QColor("#eee8d5").darker(106);
	p.primaryTopHover = "#fdf6e3";
	p.primaryBottomHover = "#eee8d5";

	p.border = QColor("#eee8d5").darker(108);
	p.borderStrong = "#93a1a1";
	// DERIVED: Solarized defines no hover tone, so this moves away from the
	// surface the way the other schemes do - which on a light scheme means
	// darker rather than lighter.
	p.hoverRow = QColor("#eee8d5").darker(106);

	p.success = "#859900";
	p.error = "#dc322f";

	p.logInfo = "#657b83";
	p.logError = "#dc322f";
	p.logWarning = "#b58900";
	p.logSuccess = "#859900";
	p.logGpu = "#859900";
	p.logCpu = "#268bd2";
	p.logPerformance = "#cb4b16";
	p.logScene = "#6c71c4";
	p.logInit = "#2aa198";
	p.logTechnique = "#d33682";
	p.logCommand = "#586e75";
	p.logDebug = "#839496";
	p.logSeparator = "#93a1a1";
	return p;
}

// Breeze Dark - colour VALUES transcribed from KDE Plasma's BreezeDark.colors
// (LGPL-2.0-or-later). The file itself is not copied or shipped. Breeze stores
// decimal RGB triplets; these are the hex equivalents.
Palette breezeDark() {
	Palette p;
	p.id = "breeze-dark";
	p.name = "Breeze Dark (KDE)";
	p.origin = "Colour values from KDE Plasma Breeze (LGPL-2.0-or-later); values transcribed, no KDE code included";
	p.isDark = true;

	p.surface0 = "#141618";   // View/BackgroundNormal
	p.surface1 = "#202326";   // Window/BackgroundNormal
	p.surface2 = "#292c30";   // Window/BackgroundAlternate, Button
	p.surface3 = QColor("#292c30").lighter(120);  // DERIVED

	p.textBody = "#fcfcfc";     // ForegroundNormal
	p.textMuted = "#a1a9b1";    // ForegroundInactive
	// DERIVED: Breeze has NO literal disabled-text colour - it defines a
	// paint-time transform that the widget style applies, which a stylesheet
	// cannot reproduce.
	p.textDisabled = QColor("#a1a9b1").darker(165);

	p.accentPrimary = "#3daee9";   // the Breeze blue
	p.accentSecondary = "#1d99f3"; // ForegroundLink
	p.accentDim = "#1e5774";       // Selection/BackgroundAlternate
	p.primaryTop = "#1e5774";
	p.primaryBottom = QColor("#1e5774").darker(130);
	p.primaryTopHover = "#3daee9";
	p.primaryBottomHover = "#1e5774";

	// DERIVED: Breeze defines no border colour at all; its widget style draws
	// frames from computed blends of the background.
	p.border = "#292c30";
	p.borderStrong = QColor("#292c30").lighter(140);
	p.hoverRow = "#1e5774";

	p.success = "#27ae60";   // ForegroundPositive
	p.error = "#da4453";     // ForegroundNegative

	p.logInfo = "#fcfcfc";
	p.logError = "#da4453";
	p.logWarning = "#f67400";   // ForegroundNeutral
	p.logSuccess = "#27ae60";
	p.logGpu = "#27ae60";
	p.logCpu = "#3daee9";
	p.logPerformance = "#f67400";
	p.logScene = "#9b59b6";     // ForegroundVisited
	p.logInit = "#1d99f3";
	p.logTechnique = "#9b59b6";
	p.logCommand = "#fcfcfc";
	p.logDebug = "#a1a9b1";
	p.logSeparator = QColor("#a1a9b1").darker(180);
	return p;
}

// Qt Creator "Dark (2024)" - colour VALUES transcribed from the theme's token
// chain (dark-2024.creatortheme -> dark.figmatokens -> primitive-colors.inc),
// which is GPL-3.0 with the Qt Company exception. The files are not copied.
// Worth noting: Qt Creator's modern accent is GREEN, not the blue people
// expect - IconsRunColor and FancyToolButtonHighlightColor both resolve to it.
// This is the only one of these schemes with a literal value for every role.
Palette qtCreatorDark() {
	Palette p;
	p.id = "qtcreator-dark";
	p.name = "Qt Creator Dark";
	p.origin = "Colour values from Qt Creator's Dark (2024) theme (GPL-3.0 with Qt Company exception); values transcribed, no Qt Creator code included";
	p.isDark = true;

	p.surface0 = "#1f1f1f";   // Token_Background_Default
	p.surface1 = "#262626";   // Token_Background_Muted
	p.surface2 = "#2d2d2d";   // Token_Background_Subtle
	p.surface3 = "#353535";   // BackgroundColorHover

	p.textBody = "#f2f2f2";     // Token_Text_Default
	p.textMuted = "#aeaeae";    // Token_Text_Muted
	p.textDisabled = "#595959"; // Token_Text_Subtle / IconsDisabledColor

	p.accentPrimary = "#27bf73";   // Token_Text_Accent
	p.accentSecondary = "#1f9b5d"; // Token_Accent_Default
	p.accentDim = "#1f9b5d";
	p.primaryTop = QColor("#1f9b5d").darker(220);
	p.primaryBottom = QColor("#1f9b5d").darker(290);
	p.primaryTopHover = QColor("#1f9b5d").darker(170);
	p.primaryBottomHover = QColor("#1f9b5d").darker(230);

	p.border = "#3f3f3f";       // Token_Stroke_Subtle / SplitterColor
	p.borderStrong = "#909090"; // Token_Stroke_Muted
	p.hoverRow = "#353535";

	p.success = "#27bf73";
	p.error = "#e34269";        // Token_Notification_Danger_Default

	p.logInfo = "#f2f2f2";
	p.logError = "#e34269";
	p.logWarning = "#efad4c";   // Token_Notification_Alert_Default
	p.logSuccess = "#27bf73";
	p.logGpu = "#27bf73";
	p.logCpu = "#8671ec";       // Token_Notification_Neutral_Default
	p.logPerformance = "#efad4c";
	p.logScene = "#8671ec";
	p.logInit = "#1f9b5d";
	p.logTechnique = "#8671ec";
	p.logCommand = "#f2f2f2";
	p.logDebug = "#aeaeae";
	p.logSeparator = "#3f3f3f";
	return p;
}

// ===========================================================================
// Topical themes
// ===========================================================================
// Unlike the eight above, these are not transcriptions of an existing scheme -
// they are original palettes built around a subject, each with its own motif
// painted on the tab pane's background (see Palette::backgroundImage).
//
// They are named for a GENRE, not for any particular game or film, and their
// artwork is drawn from scratch in qt_gui/backgrounds/. Naming a theme after a
// real franchise, or shipping its art, would be someone else's trademark and
// copyright in a public repository; the genre carries the same feel without
// borrowing anything.
//
// All four are light schemes on purpose. A faint motif needs a light ground to
// read against at all - at 7% ink on a near-black surface it is either
// invisible or, once boosted enough to see, glare.
// ===========================================================================

// Fantasy Parchment - aged paper and iron-gall ink, the palette of a tabletop
// campaign map. Warm neutrals only; the accents are leather and wax-seal red
// rather than anything that would look printed.
Palette fantasyParchment() {
	Palette p;
	p.id = "fantasy-parchment";
	p.name = "Fantasy Parchment";
	p.origin = "Original palette - aged paper and iron-gall ink";
	p.isDark = false;

	p.surface0 = "#f6ecd9";   // the page
	p.surface1 = "#efe3cc";   // panels, a shade deeper than the page
	p.surface2 = "#e6d7bb";
	p.surface3 = "#dbc9a8";

	p.textBody = "#4a3b2a";     // iron-gall ink, browner than black
	p.textMuted = "#7a6650";
	p.textDisabled = "#a6947c";

	p.accentPrimary = "#8b3a2f";   // wax seal
	p.accentSecondary = "#8b5a2b"; // tooled leather
	p.accentDim = "#b08a5e";
	p.primaryTop = "#efe3cc";
	p.primaryBottom = "#e2d2b4";
	p.primaryTopHover = "#f6ecd9";
	p.primaryBottomHover = "#eadcc2";

	p.border = "#d6c4a4";
	p.borderStrong = "#b8a284";
	p.hoverRow = "#e4d5b8";

	p.success = "#5c7a3a";   // moss
	p.error = "#a33227";

	p.logInfo = "#4a3b2a";
	p.logError = "#a33227";
	p.logWarning = "#9a6a1c";
	p.logSuccess = "#5c7a3a";
	p.logGpu = "#5c7a3a";
	p.logCpu = "#7a5b2e";
	p.logPerformance = "#a85c1e";
	p.logScene = "#6a4a7a";
	p.logInit = "#3d6b62";
	p.logTechnique = "#8b3a5e";
	p.logCommand = "#6b5b45";
	p.logDebug = "#8c7a62";
	p.logSeparator = "#c0ad8d";

	p.backgroundImage = ":/backgrounds/parchment.svg";
	p.backgroundTiled = false;
	p.backgroundPosition = "bottom right";
	return p;
}

// Sci-Fi Blueprint - drafting paper. Cool near-white ground with a single
// saturated blue doing all the accent work, which is how real blueprints read:
// one ink, many line weights.
Palette sciFiBlueprint() {
	Palette p;
	p.id = "scifi-blueprint";
	p.name = "Sci-Fi Blueprint";
	p.origin = "Original palette - drafting paper and blueprint ink";
	p.isDark = false;

	p.surface0 = "#eef4f8";
	p.surface1 = "#e4edf3";
	p.surface2 = "#d8e4ed";
	p.surface3 = "#c8d8e4";

	p.textBody = "#24404d";
	p.textMuted = "#5b7787";
	p.textDisabled = "#8ba4b2";

	p.accentPrimary = "#0b6f8f";
	p.accentSecondary = "#127d9e";
	p.accentDim = "#4a90a8";
	p.primaryTop = "#e4edf3";
	p.primaryBottom = "#d4e2ea";
	p.primaryTopHover = "#eef4f8";
	p.primaryBottomHover = "#dfeaf1";

	p.border = "#c9d9e3";
	p.borderStrong = "#a3bccb";
	p.hoverRow = "#d6e3ec";

	p.success = "#2e7d5b";
	p.error = "#b3382f";

	p.logInfo = "#24404d";
	p.logError = "#b3382f";
	p.logWarning = "#a06a10";
	p.logSuccess = "#2e7d5b";
	p.logGpu = "#2e7d5b";
	p.logCpu = "#0b6f8f";
	p.logPerformance = "#b25f00";
	p.logScene = "#5a5a9e";
	p.logInit = "#127d9e";
	p.logTechnique = "#8e3d78";
	p.logCommand = "#4a6472";
	p.logDebug = "#7a94a2";
	p.logSeparator = "#a8bfcb";

	p.backgroundImage = ":/backgrounds/blueprint.svg";
	p.backgroundTiled = true;
	return p;
}

// Retro Arcade - eighties cabinet art with the saturation pulled right down.
// The hues are the period ones (coral, teal, violet); making them pastel is
// what lets them sit on a light ground without becoming neon.
Palette retroArcade() {
	Palette p;
	p.id = "retro-arcade";
	p.name = "Retro Arcade";
	p.origin = "Original palette - eighties cabinet art, desaturated";
	p.isDark = false;

	p.surface0 = "#fbf3e6";
	p.surface1 = "#f4e9d8";
	p.surface2 = "#ebdcc6";
	p.surface3 = "#dfcbb0";

	p.textBody = "#3d3550";
	p.textMuted = "#6e6484";
	p.textDisabled = "#9c93ad";

	p.accentPrimary = "#d94f70";
	p.accentSecondary = "#2f8f8f";
	p.accentDim = "#8a6ba8";
	p.primaryTop = "#f4e9d8";
	p.primaryBottom = "#e8d9c0";
	p.primaryTopHover = "#fbf3e6";
	p.primaryBottomHover = "#f0e3ce";

	p.border = "#dfcdb4";
	p.borderStrong = "#bfa98c";
	p.hoverRow = "#ecdcc4";

	p.success = "#3f8f5c";
	p.error = "#c4372f";

	p.logInfo = "#3d3550";
	p.logError = "#c4372f";
	p.logWarning = "#c17a10";
	p.logSuccess = "#3f8f5c";
	p.logGpu = "#2f8f8f";
	p.logCpu = "#5a5fa8";
	p.logPerformance = "#e08a1e";
	p.logScene = "#8a4fa8";
	p.logInit = "#2f8f8f";
	p.logTechnique = "#d94f70";
	p.logCommand = "#5e5570";
	p.logDebug = "#8b829c";
	p.logSeparator = "#c3b39a";

	p.backgroundImage = ":/backgrounds/arcade.svg";
	p.backgroundTiled = true;
	return p;
}

// Cherry Blossom - the softest of the four. Its accents are the only ones here
// that are genuinely low-contrast against their ground, so the log colours
// deliberately run darker than the UI accents rather than matching them.
Palette cherryBlossom() {
	Palette p;
	p.id = "cherry-blossom";
	p.name = "Cherry Blossom";
	p.origin = "Original palette - sakura and warm grey";
	p.isDark = false;

	p.surface0 = "#fdf2f4";
	p.surface1 = "#f8e9ec";
	p.surface2 = "#f1dde2";
	p.surface3 = "#e6cbd3";

	p.textBody = "#4a4448";
	p.textMuted = "#7d7379";
	p.textDisabled = "#a89ea4";

	p.accentPrimary = "#b2536f";
	p.accentSecondary = "#5f8a6a";   // new leaf, the counterpoint to the pink
	p.accentDim = "#c4708c";
	p.primaryTop = "#f8e9ec";
	p.primaryBottom = "#eedae0";
	p.primaryTopHover = "#fdf2f4";
	p.primaryBottomHover = "#f5e3e8";

	p.border = "#ecd7dd";
	p.borderStrong = "#d3b3bd";
	p.hoverRow = "#f0dbe1";

	p.success = "#4f7d5c";
	p.error = "#b53a3a";

	p.logInfo = "#4a4448";
	p.logError = "#b53a3a";
	p.logWarning = "#a4761c";
	p.logSuccess = "#4f7d5c";
	p.logGpu = "#4f7d5c";
	p.logCpu = "#5a6f9e";
	p.logPerformance = "#b8862b";
	p.logScene = "#8a5f9e";
	p.logInit = "#41827d";
	p.logTechnique = "#b2536f";
	p.logCommand = "#6a5f66";
	p.logDebug = "#94888f";
	p.logSeparator = "#cdb8c0";

	p.backgroundImage = ":/backgrounds/blossom.svg";
	p.backgroundTiled = false;
	p.backgroundPosition = "top left";
	return p;
}

const QVector<Palette> &registry() {
	static const QVector<Palette> themes = {
		cyberpunk(),
		dracula(),
		nord(),
		gruvbox(),
		solarizedDark(),
		solarizedLight(),
		breezeDark(),
		qtCreatorDark(),
		fantasyParchment(),
		sciFiBlueprint(),
		retroArcade(),
		cherryBlossom(),
	};
	return themes;
}

} // namespace

const QVector<Palette> &all() {
	return registry();
}

const Palette &defaultPalette() {
	return registry().first();
}

const Palette &byId(const QString &id) {
	for (const Palette &p : registry()) {
		if (p.id == id) return p;
	}
	// Unknown id: settings written by a newer build, or a theme that has since
	// been removed. Falling back beats refusing to start.
	return defaultPalette();
}

} // namespace theme
