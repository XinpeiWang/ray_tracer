#include "palette_data.h"

#include <cmath>

// ==========================================================================
// Built-in palettes
// ==========================================================================
// Generated once from the Qt implementation this replaced and then verified
// byte-identical against it, so no value here was retyped by hand.
//
// Values are literal. Several were originally authored as a Qt expression such
// as QColor("#eee8d5").darker(104); those now carry the expression in a comment
// instead. Deriving one colour from another was an authoring convenience, never
// a runtime need, and keeping it would have meant reimplementing Qt's HSV
// rounding here exactly - a real risk of silently shifting a shipped colour.
//
// A DERIVED note means the upstream scheme genuinely has no value for that role
// and this one was invented; without the marker, a reader cannot tell which
// numbers are Solarized's and which are mine.
// ==========================================================================
namespace palette_data {
namespace {

const std::vector<PaletteData> kBuiltins = {

// The scheme this project shipped with. Preserved value-for-value so switching
// away and back is a genuine round trip rather than an approximation.
{
	"cyberpunk", "Cyberpunk", "This project's original palette",
	{0x0e, 0x0e, 0x14},      // surface0
	{0x16, 0x16, 0x1f},      // surface1
	{0x1e, 0x1e, 0x2a},      // surface2
	{0x2a, 0x2a, 0x3a},      // surface3
	{0xe6, 0xe6, 0xf0},      // textBody
	{0x9a, 0x9a, 0xb0},      // textMuted
	{0x5a, 0x5a, 0x70},      // textDisabled
	{0xff, 0x3d, 0xff},      // accentPrimary
	{0x00, 0xe5, 0xff},      // accentSecondary
	{0xc9, 0x3f, 0xe8},      // accentDim
	{0x3a, 0x10, 0x50},      // primaryTop
	{0x24, 0x0c, 0x38},      // primaryBottom
	{0x4e, 0x16, 0x68},      // primaryTopHover
	{0x34, 0x10, 0x48},      // primaryBottomHover
	{0x2a, 0x2a, 0x3a},      // border
	{0x3a, 0x3a, 0x50},      // borderStrong
	{0x3a, 0x2e, 0x56},      // hoverRow
	{0x5a, 0xaa, 0x3c},      // success
	{0xdf, 0x4f, 0x4f},      // error
	{0xd0, 0xd0, 0xd0},      // logInfo
	{0xff, 0x6b, 0x6b},      // logError
	{0xff, 0xd7, 0x00},      // logWarning
	{0x51, 0xcf, 0x66},      // logSuccess
	{0xa9, 0xe3, 0x4b},      // logGpu
	{0x74, 0xc0, 0xfc},      // logCpu
	{0xff, 0xa9, 0x4d},      // logPerformance
	{0xcc, 0x99, 0xff},      // logScene
	{0x63, 0xe6, 0xbe},      // logInit
	{0xe5, 0x99, 0xf7},      // logTechnique
	{0xcc, 0xcc, 0xcc},      // logCommand
	{0xa8, 0xa8, 0xa8},      // logDebug
	{0x55, 0x55, 0x55},      // logSeparator
},

// Dracula - MIT (c) 2023 Dracula Theme. Values from the project README's
// "Color Palette (OSS)" table. The commercial Dracula PRO palette is a
// separate, non-MIT thing and is deliberately not used.
{
	"dracula", "Dracula", "Dracula (MIT) - draculatheme.com",
	{0x22, 0x24, 0x2e},      // surface0 - was QColor("#282a36").darker(118)
	{0x28, 0x2a, 0x36},      // surface1 - Background
	{0x44, 0x47, 0x5a},      // surface2 - Current Line / Selection
	{0x4e, 0x52, 0x67},      // surface3 - was QColor("#44475a").lighter(115); DERIVED
	{0xf8, 0xf8, 0xf2},      // textBody - Foreground
	{0x62, 0x72, 0xa4},      // textMuted - Comment
	{0x4b, 0x58, 0x7e},      // textDisabled - was QColor("#6272a4").darker(130); DERIVED: no such value
	{0xbd, 0x93, 0xf9},      // accentPrimary - Purple
	{0x8b, 0xe9, 0xfd},      // accentSecondary - Cyan
	{0xff, 0x79, 0xc6},      // accentDim - Pink
	{0x49, 0x39, 0x60},      // primaryTop - was QColor("#bd93f9").darker(260)
	{0x39, 0x2d, 0x4b},      // primaryBottom - was QColor("#bd93f9").darker(330)
	{0x5a, 0x46, 0x77},      // primaryTopHover - was QColor("#bd93f9").darker(210)
	{0x46, 0x36, 0x5c},      // primaryBottomHover - was QColor("#bd93f9").darker(270)
	{0x44, 0x47, 0x5a},      // border
	{0x62, 0x72, 0xa4},      // borderStrong
	{0x52, 0x55, 0x6c},      // hoverRow - was QColor("#44475a").lighter(120)
	{0x50, 0xfa, 0x7b},      // success
	{0xff, 0x55, 0x55},      // error
	{0xf8, 0xf8, 0xf2},      // logInfo
	{0xff, 0x55, 0x55},      // logError
	{0xf1, 0xfa, 0x8c},      // logWarning
	{0x50, 0xfa, 0x7b},      // logSuccess
	{0x50, 0xfa, 0x7b},      // logGpu
	{0x8b, 0xe9, 0xfd},      // logCpu
	{0xff, 0xb8, 0x6c},      // logPerformance
	{0xbd, 0x93, 0xf9},      // logScene
	{0x8b, 0xe9, 0xfd},      // logInit
	{0xff, 0x79, 0xc6},      // logTechnique
	{0xf8, 0xf8, 0xf2},      // logCommand
	{0x62, 0x72, 0xa4},      // logDebug
	{0x62, 0x72, 0xa4},      // logSeparator
},

// Nord - MIT (c) 2016-present Sven Greb. Values from src/nord.scss, whose own
// doc comments assign nord0/1/2 to backgrounds, nord3 to disabled UI, nord4 to
// text, and explicitly name nord8 "the accent color of the color palette".
{
	"nord", "Nord", "Nord (MIT) - nordtheme.com",
	{0x2e, 0x34, 0x40},      // surface0 - nord0
	{0x3b, 0x42, 0x52},      // surface1 - nord1
	{0x43, 0x4c, 0x5e},      // surface2 - nord2
	{0x4c, 0x56, 0x6a},      // surface3 - nord3
	{0xd8, 0xde, 0xe9},      // textBody - nord4
	{0x72, 0x81, 0x9f},      // textMuted - was QColor("#4c566a").lighter(150)
	{0x4c, 0x56, 0x6a},      // textDisabled - nord3, its documented disabled role
	{0x88, 0xc0, 0xd0},      // accentPrimary - nord8, the documented accent
	{0x8f, 0xbc, 0xbb},      // accentSecondary - nord7
	{0x5e, 0x81, 0xac},      // accentDim - nord10
	{0x3b, 0x42, 0x52},      // primaryTop
	{0x2e, 0x34, 0x40},      // primaryBottom
	{0x43, 0x4c, 0x5e},      // primaryTopHover
	{0x3b, 0x42, 0x52},      // primaryBottomHover
	{0x43, 0x4c, 0x5e},      // border
	{0x4c, 0x56, 0x6a},      // borderStrong
	{0x43, 0x4c, 0x5e},      // hoverRow
	{0xa3, 0xbe, 0x8c},      // success - nord14
	{0xbf, 0x61, 0x6a},      // error - nord11
	{0xd8, 0xde, 0xe9},      // logInfo
	{0xbf, 0x61, 0x6a},      // logError
	{0xeb, 0xcb, 0x8b},      // logWarning - nord13
	{0xa3, 0xbe, 0x8c},      // logSuccess
	{0xa3, 0xbe, 0x8c},      // logGpu
	{0x88, 0xc0, 0xd0},      // logCpu
	{0xd0, 0x87, 0x70},      // logPerformance - nord12
	{0xb4, 0x8e, 0xad},      // logScene - nord15
	{0x8f, 0xbc, 0xbb},      // logInit
	{0x81, 0xa1, 0xc1},      // logTechnique - nord9
	{0xd8, 0xde, 0xe9},      // logCommand
	{0x72, 0x81, 0x9f},      // logDebug - was QColor("#4c566a").lighter(150)
	{0x4c, 0x56, 0x6a},      // logSeparator
},

// Gruvbox Dark - MIT/X11, Pavel Pertsev. Values from colors/gruvbox.vim.
// The best-covered of these schemes: it has a real value for every role.
{
	"gruvbox", "Gruvbox Dark", "Gruvbox (MIT/X11) - github.com/morhetz/gruvbox",
	{0x1d, 0x20, 0x21},      // surface0 - dark0_hard
	{0x28, 0x28, 0x28},      // surface1 - dark0 / bg0
	{0x3c, 0x38, 0x36},      // surface2 - dark1 / bg1
	{0x50, 0x49, 0x45},      // surface3 - dark2 / bg2
	{0xeb, 0xdb, 0xb2},      // textBody - light1 / fg1, the Normal foreground
	{0x92, 0x83, 0x74},      // textMuted - gray_245, the Comment colour
	{0x7c, 0x6f, 0x64},      // textDisabled - dark4 / bg4, used for LineNr
	{0xfa, 0xbd, 0x2f},      // accentPrimary - bright_yellow
	{0x83, 0xa5, 0x98},      // accentSecondary - bright_blue
	{0xfe, 0x80, 0x19},      // accentDim - bright_orange
	{0x50, 0x49, 0x45},      // primaryTop
	{0x3c, 0x38, 0x36},      // primaryBottom
	{0x66, 0x5c, 0x54},      // primaryTopHover
	{0x50, 0x49, 0x45},      // primaryBottomHover
	{0x50, 0x49, 0x45},      // border - bg2
	{0x66, 0x5c, 0x54},      // borderStrong - bg3
	{0x50, 0x49, 0x45},      // hoverRow
	{0xb8, 0xbb, 0x26},      // success - bright_green
	{0xfb, 0x49, 0x34},      // error - bright_red
	{0xeb, 0xdb, 0xb2},      // logInfo
	{0xfb, 0x49, 0x34},      // logError
	{0xfa, 0xbd, 0x2f},      // logWarning
	{0xb8, 0xbb, 0x26},      // logSuccess
	{0xb8, 0xbb, 0x26},      // logGpu
	{0x83, 0xa5, 0x98},      // logCpu
	{0xfe, 0x80, 0x19},      // logPerformance
	{0xd3, 0x86, 0x9b},      // logScene - bright_purple
	{0x8e, 0xc0, 0x7c},      // logInit - bright_aqua
	{0xd3, 0x86, 0x9b},      // logTechnique
	{0xeb, 0xdb, 0xb2},      // logCommand
	{0x92, 0x83, 0x74},      // logDebug
	{0x66, 0x5c, 0x54},      // logSeparator
},

// Solarized Dark - MIT (c) 2011 Ethan Schoonover. Values from the repo README
// palette table. Per the spec, body text on a dark background is base0 (NOT
// base00); base00/base01 are used here for the muted and disabled steps, which
// keeps the whole text ramp inside the canonical palette.
{
	"solarized-dark", "Solarized Dark", "Solarized (MIT) - Ethan Schoonover",
	{0x00, 0x2b, 0x36},      // surface0 - base03
	{0x07, 0x36, 0x42},      // surface1 - base02
	{0x09, 0x44, 0x52},      // surface2 - was QColor("#073642").lighter(125)
	{0x0b, 0x51, 0x63},      // surface3 - was QColor("#073642").lighter(150)
	{0x83, 0x94, 0x96},      // textBody - base0
	{0x65, 0x7b, 0x83},      // textMuted - base00
	{0x58, 0x6e, 0x75},      // textDisabled - base01
	{0x26, 0x8b, 0xd2},      // accentPrimary - blue
	{0x2a, 0xa1, 0x98},      // accentSecondary - cyan
	{0x6c, 0x71, 0xc4},      // accentDim - violet
	{0x07, 0x36, 0x42},      // primaryTop
	{0x00, 0x2b, 0x36},      // primaryBottom
	{0x09, 0x46, 0x56},      // primaryTopHover - was QColor("#073642").lighter(130)
	{0x07, 0x36, 0x42},      // primaryBottomHover
	{0x07, 0x36, 0x42},      // border
	{0x58, 0x6e, 0x75},      // borderStrong
	{0x09, 0x49, 0x59},      // hoverRow - was QColor("#073642").lighter(135)
	{0x85, 0x99, 0x00},      // success - green
	{0xdc, 0x32, 0x2f},      // error - red
	{0x83, 0x94, 0x96},      // logInfo
	{0xdc, 0x32, 0x2f},      // logError
	{0xb5, 0x89, 0x00},      // logWarning - yellow
	{0x85, 0x99, 0x00},      // logSuccess
	{0x85, 0x99, 0x00},      // logGpu
	{0x26, 0x8b, 0xd2},      // logCpu
	{0xcb, 0x4b, 0x16},      // logPerformance - orange
	{0x6c, 0x71, 0xc4},      // logScene - violet
	{0x2a, 0xa1, 0x98},      // logInit - cyan
	{0xd3, 0x36, 0x82},      // logTechnique - magenta
	{0x93, 0xa1, 0xa1},      // logCommand - base1
	{0x65, 0x7b, 0x83},      // logDebug
	{0x58, 0x6e, 0x75},      // logSeparator
},

// Solarized Light - same MIT palette, base ramp inverted. Solarized's accents
// are deliberately shared between the two modes, so only the greys flip.
// The log colours matter most here: this is the one scheme where a bright
// on-black log palette would be unreadable.
{
	"solarized-light", "Solarized Light", "Solarized (MIT) - Ethan Schoonover",
	{0xfd, 0xf6, 0xe3},      // surface0 - base3
	{0xee, 0xe8, 0xd5},      // surface1 - base2
	{0xe5, 0xdf, 0xcd},      // surface2 - was QColor("#eee8d5").darker(104); DERIVED, see Solarized Dark
	{0xd8, 0xd3, 0xc2},      // surface3 - was QColor("#eee8d5").darker(110)
	{0x65, 0x7b, 0x83},      // textBody - base00
	{0x83, 0x94, 0x96},      // textMuted - base0
	{0x93, 0xa1, 0xa1},      // textDisabled - base1
	{0x26, 0x8b, 0xd2},      // accentPrimary
	{0x2a, 0xa1, 0x98},      // accentSecondary
	{0x6c, 0x71, 0xc4},      // accentDim
	{0xee, 0xe8, 0xd5},      // primaryTop
	{0xe1, 0xdb, 0xc9},      // primaryBottom - was QColor("#eee8d5").darker(106)
	{0xfd, 0xf6, 0xe3},      // primaryTopHover
	{0xee, 0xe8, 0xd5},      // primaryBottomHover
	{0xdc, 0xd7, 0xc5},      // border - was QColor("#eee8d5").darker(108)
	{0x93, 0xa1, 0xa1},      // borderStrong
	{0xe1, 0xdb, 0xc9},      // hoverRow - was QColor("#eee8d5").darker(106)
	{0x85, 0x99, 0x00},      // success
	{0xdc, 0x32, 0x2f},      // error
	{0x65, 0x7b, 0x83},      // logInfo
	{0xdc, 0x32, 0x2f},      // logError
	{0xb5, 0x89, 0x00},      // logWarning
	{0x85, 0x99, 0x00},      // logSuccess
	{0x85, 0x99, 0x00},      // logGpu
	{0x26, 0x8b, 0xd2},      // logCpu
	{0xcb, 0x4b, 0x16},      // logPerformance
	{0x6c, 0x71, 0xc4},      // logScene
	{0x2a, 0xa1, 0x98},      // logInit
	{0xd3, 0x36, 0x82},      // logTechnique
	{0x58, 0x6e, 0x75},      // logCommand
	{0x83, 0x94, 0x96},      // logDebug
	{0x93, 0xa1, 0xa1},      // logSeparator
},

// Breeze Dark - colour VALUES transcribed from KDE Plasma's BreezeDark.colors
// (LGPL-2.0-or-later). The file itself is not copied or shipped. Breeze stores
// decimal RGB triplets; these are the hex equivalents.
{
	"breeze-dark", "Breeze Dark (KDE)", "Colour values from KDE Plasma Breeze (LGPL-2.0-or-later); values transcribed, no KDE code included",
	{0x14, 0x16, 0x18},      // surface0 - View/BackgroundNormal
	{0x20, 0x23, 0x26},      // surface1 - Window/BackgroundNormal
	{0x29, 0x2c, 0x30},      // surface2 - Window/BackgroundAlternate, Button
	{0x31, 0x35, 0x3a},      // surface3 - was QColor("#292c30").lighter(120); DERIVED
	{0xfc, 0xfc, 0xfc},      // textBody - ForegroundNormal
	{0xa1, 0xa9, 0xb1},      // textMuted - ForegroundInactive
	{0x62, 0x66, 0x6b},      // textDisabled - was QColor("#a1a9b1").darker(165)
	{0x3d, 0xae, 0xe9},      // accentPrimary - the Breeze blue
	{0x1d, 0x99, 0xf3},      // accentSecondary - ForegroundLink
	{0x1e, 0x57, 0x74},      // accentDim - Selection/BackgroundAlternate
	{0x1e, 0x57, 0x74},      // primaryTop
	{0x17, 0x43, 0x59},      // primaryBottom - was QColor("#1e5774").darker(130)
	{0x3d, 0xae, 0xe9},      // primaryTopHover
	{0x1e, 0x57, 0x74},      // primaryBottomHover
	{0x29, 0x2c, 0x30},      // border
	{0x39, 0x3e, 0x43},      // borderStrong - was QColor("#292c30").lighter(140)
	{0x1e, 0x57, 0x74},      // hoverRow
	{0x27, 0xae, 0x60},      // success - ForegroundPositive
	{0xda, 0x44, 0x53},      // error - ForegroundNegative
	{0xfc, 0xfc, 0xfc},      // logInfo
	{0xda, 0x44, 0x53},      // logError
	{0xf6, 0x74, 0x00},      // logWarning - ForegroundNeutral
	{0x27, 0xae, 0x60},      // logSuccess
	{0x27, 0xae, 0x60},      // logGpu
	{0x3d, 0xae, 0xe9},      // logCpu
	{0xf6, 0x74, 0x00},      // logPerformance
	{0x9b, 0x59, 0xb6},      // logScene - ForegroundVisited
	{0x1d, 0x99, 0xf3},      // logInit
	{0x9b, 0x59, 0xb6},      // logTechnique
	{0xfc, 0xfc, 0xfc},      // logCommand
	{0xa1, 0xa9, 0xb1},      // logDebug
	{0x59, 0x5e, 0x62},      // logSeparator - was QColor("#a1a9b1").darker(180)
},

// Qt Creator "Dark (2024)" - colour VALUES transcribed from the theme's token
// chain (dark-2024.creatortheme -> dark.figmatokens -> primitive-colors.inc),
// which is GPL-3.0 with the Qt Company exception. The files are not copied.
// Worth noting: Qt Creator's modern accent is GREEN, not the blue people
// expect - IconsRunColor and FancyToolButtonHighlightColor both resolve to it.
// This is the only one of these schemes with a literal value for every role.
{
	"qtcreator-dark", "Qt Creator Dark", "Colour values from Qt Creator's Dark (2024) theme (GPL-3.0 with Qt Company exception); values transcribed, no Qt Creator code included",
	{0x1f, 0x1f, 0x1f},      // surface0 - Token_Background_Default
	{0x26, 0x26, 0x26},      // surface1 - Token_Background_Muted
	{0x2d, 0x2d, 0x2d},      // surface2 - Token_Background_Subtle
	{0x35, 0x35, 0x35},      // surface3 - BackgroundColorHover
	{0xf2, 0xf2, 0xf2},      // textBody - Token_Text_Default
	{0xae, 0xae, 0xae},      // textMuted - Token_Text_Muted
	{0x59, 0x59, 0x59},      // textDisabled - Token_Text_Subtle / IconsDisabledColor
	{0x27, 0xbf, 0x73},      // accentPrimary - Token_Text_Accent
	{0x1f, 0x9b, 0x5d},      // accentSecondary - Token_Accent_Default
	{0x1f, 0x9b, 0x5d},      // accentDim
	{0x0e, 0x46, 0x2a},      // primaryTop - was QColor("#1f9b5d").darker(220)
	{0x0b, 0x35, 0x20},      // primaryBottom - was QColor("#1f9b5d").darker(290)
	{0x12, 0x5b, 0x37},      // primaryTopHover - was QColor("#1f9b5d").darker(170)
	{0x0d, 0x43, 0x28},      // primaryBottomHover - was QColor("#1f9b5d").darker(230)
	{0x3f, 0x3f, 0x3f},      // border - Token_Stroke_Subtle / SplitterColor
	{0x90, 0x90, 0x90},      // borderStrong - Token_Stroke_Muted
	{0x35, 0x35, 0x35},      // hoverRow
	{0x27, 0xbf, 0x73},      // success
	{0xe3, 0x42, 0x69},      // error - Token_Notification_Danger_Default
	{0xf2, 0xf2, 0xf2},      // logInfo
	{0xe3, 0x42, 0x69},      // logError
	{0xef, 0xad, 0x4c},      // logWarning - Token_Notification_Alert_Default
	{0x27, 0xbf, 0x73},      // logSuccess
	{0x27, 0xbf, 0x73},      // logGpu
	{0x86, 0x71, 0xec},      // logCpu - Token_Notification_Neutral_Default
	{0xef, 0xad, 0x4c},      // logPerformance
	{0x86, 0x71, 0xec},      // logScene
	{0x1f, 0x9b, 0x5d},      // logInit
	{0x86, 0x71, 0xec},      // logTechnique
	{0xf2, 0xf2, 0xf2},      // logCommand
	{0xae, 0xae, 0xae},      // logDebug
	{0x3f, 0x3f, 0x3f},      // logSeparator
},

// Fantasy Parchment - aged paper and iron-gall ink, the palette of a tabletop
// campaign map. Warm neutrals only; the accents are leather and wax-seal red
// rather than anything that would look printed.
{
	"fantasy-parchment", "Fantasy Parchment", "Original palette - aged paper and iron-gall ink",
	{0xf6, 0xec, 0xd9},      // surface0 - the page
	{0xef, 0xe3, 0xcc},      // surface1 - panels, a shade deeper than the page
	{0xe6, 0xd7, 0xbb},      // surface2
	{0xdb, 0xc9, 0xa8},      // surface3
	{0x4a, 0x3b, 0x2a},      // textBody - iron-gall ink, browner than black
	{0x7a, 0x66, 0x50},      // textMuted
	{0xa6, 0x94, 0x7c},      // textDisabled
	{0x8b, 0x3a, 0x2f},      // accentPrimary - wax seal
	{0x8b, 0x5a, 0x2b},      // accentSecondary - tooled leather
	{0xb0, 0x8a, 0x5e},      // accentDim
	{0xef, 0xe3, 0xcc},      // primaryTop
	{0xe2, 0xd2, 0xb4},      // primaryBottom
	{0xf6, 0xec, 0xd9},      // primaryTopHover
	{0xea, 0xdc, 0xc2},      // primaryBottomHover
	{0xd6, 0xc4, 0xa4},      // border
	{0xb8, 0xa2, 0x84},      // borderStrong
	{0xe4, 0xd5, 0xb8},      // hoverRow
	{0x5c, 0x7a, 0x3a},      // success - moss
	{0xa3, 0x32, 0x27},      // error
	{0x4a, 0x3b, 0x2a},      // logInfo
	{0xa3, 0x32, 0x27},      // logError
	{0x9a, 0x6a, 0x1c},      // logWarning
	{0x5c, 0x7a, 0x3a},      // logSuccess
	{0x5c, 0x7a, 0x3a},      // logGpu
	{0x7a, 0x5b, 0x2e},      // logCpu
	{0xa8, 0x5c, 0x1e},      // logPerformance
	{0x6a, 0x4a, 0x7a},      // logScene
	{0x3d, 0x6b, 0x62},      // logInit
	{0x8b, 0x3a, 0x5e},      // logTechnique
	{0x6b, 0x5b, 0x45},      // logCommand
	{0x8c, 0x7a, 0x62},      // logDebug
	{0xc0, 0xad, 0x8d},      // logSeparator
	":/backgrounds/parchment.svg", false, "bottom right",
},

// Sci-Fi Blueprint - drafting paper. Cool near-white ground with a single
// saturated blue doing all the accent work, which is how real blueprints read:
// one ink, many line weights.
{
	"scifi-blueprint", "Sci-Fi Blueprint", "Original palette - drafting paper and blueprint ink",
	{0xee, 0xf4, 0xf8},      // surface0
	{0xe4, 0xed, 0xf3},      // surface1
	{0xd8, 0xe4, 0xed},      // surface2
	{0xc8, 0xd8, 0xe4},      // surface3
	{0x24, 0x40, 0x4d},      // textBody
	{0x5b, 0x77, 0x87},      // textMuted
	{0x8b, 0xa4, 0xb2},      // textDisabled
	{0x0b, 0x6f, 0x8f},      // accentPrimary
	{0x12, 0x7d, 0x9e},      // accentSecondary
	{0x4a, 0x90, 0xa8},      // accentDim
	{0xe4, 0xed, 0xf3},      // primaryTop
	{0xd4, 0xe2, 0xea},      // primaryBottom
	{0xee, 0xf4, 0xf8},      // primaryTopHover
	{0xdf, 0xea, 0xf1},      // primaryBottomHover
	{0xc9, 0xd9, 0xe3},      // border
	{0xa3, 0xbc, 0xcb},      // borderStrong
	{0xd6, 0xe3, 0xec},      // hoverRow
	{0x2e, 0x7d, 0x5b},      // success
	{0xb3, 0x38, 0x2f},      // error
	{0x24, 0x40, 0x4d},      // logInfo
	{0xb3, 0x38, 0x2f},      // logError
	{0xa0, 0x6a, 0x10},      // logWarning
	{0x2e, 0x7d, 0x5b},      // logSuccess
	{0x2e, 0x7d, 0x5b},      // logGpu
	{0x0b, 0x6f, 0x8f},      // logCpu
	{0xb2, 0x5f, 0x00},      // logPerformance
	{0x5a, 0x5a, 0x9e},      // logScene
	{0x12, 0x7d, 0x9e},      // logInit
	{0x8e, 0x3d, 0x78},      // logTechnique
	{0x4a, 0x64, 0x72},      // logCommand
	{0x6f, 0x88, 0x94},      // logDebug - darkened from #7a94a2, which measured 2.87:1 on surface0 (ContrastTest)
	{0xa8, 0xbf, 0xcb},      // logSeparator
	":/backgrounds/blueprint.svg", true, "bottom right",
},

// Retro Arcade - eighties cabinet art with the saturation pulled right down.
// The hues are the period ones (coral, teal, violet); making them pastel is
// what lets them sit on a light ground without becoming neon.
{
	"retro-arcade", "Retro Arcade", "Original palette - eighties cabinet art, desaturated",
	{0xfb, 0xf3, 0xe6},      // surface0
	{0xf4, 0xe9, 0xd8},      // surface1
	{0xeb, 0xdc, 0xc6},      // surface2
	{0xdf, 0xcb, 0xb0},      // surface3
	{0x3d, 0x35, 0x50},      // textBody
	{0x6e, 0x64, 0x84},      // textMuted
	{0x9c, 0x93, 0xad},      // textDisabled
	{0xc2, 0x46, 0x5f},      // accentPrimary - deepened from #d94f70, which left the primary button's label at 2.85:1 on its own gradient
	{0x2f, 0x8f, 0x8f},      // accentSecondary
	{0x8a, 0x6b, 0xa8},      // accentDim
	{0xf4, 0xe9, 0xd8},      // primaryTop
	{0xe8, 0xd9, 0xc0},      // primaryBottom
	{0xfb, 0xf3, 0xe6},      // primaryTopHover
	{0xf0, 0xe3, 0xce},      // primaryBottomHover
	{0xdf, 0xcd, 0xb4},      // border
	{0xbf, 0xa9, 0x8c},      // borderStrong
	{0xec, 0xdc, 0xc4},      // hoverRow
	{0x3f, 0x8f, 0x5c},      // success
	{0xc4, 0x37, 0x2f},      // error
	{0x3d, 0x35, 0x50},      // logInfo
	{0xc4, 0x37, 0x2f},      // logError
	{0xc1, 0x7a, 0x10},      // logWarning
	{0x3f, 0x8f, 0x5c},      // logSuccess
	{0x2f, 0x8f, 0x8f},      // logGpu
	{0x5a, 0x5f, 0xa8},      // logCpu
	{0xb8, 0x6f, 0x18},      // logPerformance - darkened from #e08a1e, which measured 2.44:1 on surface0
	{0x8a, 0x4f, 0xa8},      // logScene
	{0x2f, 0x8f, 0x8f},      // logInit
	{0xd9, 0x4f, 0x70},      // logTechnique
	{0x5e, 0x55, 0x70},      // logCommand
	{0x8b, 0x82, 0x9c},      // logDebug
	{0xc3, 0xb3, 0x9a},      // logSeparator
	":/backgrounds/arcade.svg", true, "bottom right",
},

// Cherry Blossom - the softest of the four. Its accents are the only ones here
// that are genuinely low-contrast against their ground, so the log colours
// deliberately run darker than the UI accents rather than matching them.
{
	"cherry-blossom", "Cherry Blossom", "Original palette - sakura and warm grey",
	{0xfd, 0xf2, 0xf4},      // surface0
	{0xf8, 0xe9, 0xec},      // surface1
	{0xf1, 0xdd, 0xe2},      // surface2
	{0xe6, 0xcb, 0xd3},      // surface3
	{0x4a, 0x44, 0x48},      // textBody
	{0x7d, 0x73, 0x79},      // textMuted
	{0xa8, 0x9e, 0xa4},      // textDisabled
	{0xb2, 0x53, 0x6f},      // accentPrimary
	{0x5f, 0x8a, 0x6a},      // accentSecondary - new leaf, the counterpoint to the pink
	{0xc4, 0x70, 0x8c},      // accentDim
	{0xf8, 0xe9, 0xec},      // primaryTop
	{0xee, 0xda, 0xe0},      // primaryBottom
	{0xfd, 0xf2, 0xf4},      // primaryTopHover
	{0xf5, 0xe3, 0xe8},      // primaryBottomHover
	{0xec, 0xd7, 0xdd},      // border
	{0xd3, 0xb3, 0xbd},      // borderStrong
	{0xf0, 0xdb, 0xe1},      // hoverRow
	{0x4f, 0x7d, 0x5c},      // success
	{0xb5, 0x3a, 0x3a},      // error
	{0x4a, 0x44, 0x48},      // logInfo
	{0xb5, 0x3a, 0x3a},      // logError
	{0xa4, 0x76, 0x1c},      // logWarning
	{0x4f, 0x7d, 0x5c},      // logSuccess
	{0x4f, 0x7d, 0x5c},      // logGpu
	{0x5a, 0x6f, 0x9e},      // logCpu
	{0xa8, 0x76, 0x1f},      // logPerformance - darkened from #b8862b, which measured 2.96:1 on surface0
	{0x8a, 0x5f, 0x9e},      // logScene
	{0x41, 0x82, 0x7d},      // logInit
	{0xb2, 0x53, 0x6f},      // logTechnique
	{0x6a, 0x5f, 0x66},      // logCommand
	{0x94, 0x88, 0x8f},      // logDebug
	{0xcd, 0xb8, 0xc0},      // logSeparator
	":/backgrounds/blossom.svg", false, "top left",
},

};

} // namespace

const std::vector<PaletteData> &builtins() { return kBuiltins; }

// WCAG 2.1 relative luminance: linearise each channel, then weight by the eye's
// sensitivity to it. The 0.03928 knee and 1.055/2.4 curve are the sRGB transfer
// function, not arbitrary constants.
double relativeLuminance(const Rgb &c) {
	auto lin = [](unsigned char v) {
		const double s = v / 255.0;
		return s <= 0.03928 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
	};
	return 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
}

double contrastRatio(const Rgb &a, const Rgb &b) {
	const double la = relativeLuminance(a);
	const double lb = relativeLuminance(b);
	const double hi = la > lb ? la : lb;
	const double lo = la > lb ? lb : la;
	return (hi + 0.05) / (lo + 0.05);
}

} // namespace palette_data
