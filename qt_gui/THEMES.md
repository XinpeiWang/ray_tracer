# Theme attribution

The GUI ships eight colour schemes (`qt_gui/theme.cpp`). Seven of them take
their numbers from established, widely-used developer colour schemes rather
than from values invented here — they have been eye-tested by far more people
than this project has, and they land somewhere users already recognise.

## What was and was not taken

**Only colour values were used.** No source file, palette file, stylesheet,
icon, or other asset from any of these projects is included in this repository
or was copied into it. Each palette is a hand-typed table of hex values in
`theme.cpp`, transcribed from the upstream project's own published palette.

This distinction matters for Breeze and Qt Creator, whose licences are
copyleft: their palette *files* could not be vendored here, but a colour value
is a fact about a scheme, not an expressive work, and retyping the numbers does
not carry the file's licence along with it. Both are credited below regardless.

**Where a scheme genuinely lacks a value** for a role this UI needs, the value
is derived here and marked `// DERIVED:` at its definition, with the reason.
Those are inventions of mine, not part of the upstream scheme, and the markers
exist so nobody later mistakes one for the other. Common causes: Solarized
publishes only two background tones per mode and no disabled-text colour;
Dracula has nothing darker than its single background; Breeze computes disabled
text and border tints at paint time rather than storing them.

## Schemes

| Scheme | Source | Licence |
| --- | --- | --- |
| Cyberpunk | This project's original palette | — |
| Dracula | [draculatheme.com](https://draculatheme.com) | MIT |
| Nord | [nordtheme.com](https://www.nordtheme.com) | MIT |
| Gruvbox Dark | [github.com/morhetz/gruvbox](https://github.com/morhetz/gruvbox) | MIT/X11 |
| Solarized Dark | Ethan Schoonover, [ethanschoonover.com/solarized](https://ethanschoonover.com/solarized/) | MIT |
| Solarized Light | Ethan Schoonover, as above | MIT |
| Breeze Dark (KDE) | KDE Plasma Breeze colour scheme | LGPL-2.0-or-later |
| Qt Creator Dark | Qt Creator's Dark (2024) theme | GPL-3.0 with Qt Company exception |

Cyberpunk is the project's own scheme and remains the default; its values are
unchanged from before the theme system existed, so the app looks the same on
first run as it always did.

## Icons

The SVGs under `qt_gui/icons/` are original to this project. They are
monochrome silhouettes with real transparency (holes cut with
`fill-rule="evenodd"`, never painted in a background colour) and are recoloured
at runtime from the active palette — see `qt_gui/icon_tint.h`.
