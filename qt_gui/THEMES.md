# Theme attribution

The GUI ships twelve colour schemes (`qt_gui/theme.cpp`). Seven of them take
their numbers from established, widely-used developer colour schemes rather
than from values invented here — they have been eye-tested by far more people
than this project has, and they land somewhere users already recognise. The
remaining five (Cyberpunk plus the four topical schemes below) are original.

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

## Topical themes

Four further schemes are not transcriptions of anything - they are original
palettes built around a subject, each with its own decorative motif:

| Scheme | Motif | Placement |
| --- | --- | --- |
| Fantasy Parchment | compass rose, contour lines, sea-serpent flourish | bottom right |
| Sci-Fi Blueprint | drafting grid with registration crosses | tiled |
| Retro Arcade | dot matrix with sparkles and a scanline | tiled |
| Cherry Blossom | branch, blossoms and drifting petals | top left |

**These are named for a genre, never for a franchise, and every motif is drawn
from scratch** (`qt_gui/backgrounds/*.svg`). No game or film artwork, screenshot,
logo, or name is used. Naming a theme after a real property, or shipping its
art, would be someone else's trademark and copyright — the genre carries the
same feel without borrowing anything.

All four are light schemes deliberately: a faint motif needs a light ground to
read against at all. At the opacities used here it would be either invisible on
a near-black surface or, once boosted enough to see, glare.

### How the motif is painted

On the tab pages' `QScrollArea` background — that is, the area *around* the
group boxes. Panels stay fully opaque, so the motif is never underneath body
text or log lines; ghosting a pattern through those costs legibility for very
little character. Opacity is baked into each SVG rather than applied at runtime,
because Qt stylesheets have no opacity property for `background-image`.

It cannot go on `QTabWidget::pane`: the scroll area covers the pane edge to
edge, so a background set there is never seen (confirmed by probing with an
opaque colour — nothing rendered). The scroll area's own viewport and content
widget also fill themselves opaquely by default and have to be punched through,
which is what the `QScrollArea#tabScroll > QWidget` rules do.

## Icons

The SVGs under `qt_gui/icons/` are original to this project. They are
monochrome silhouettes with real transparency (holes cut with
`fill-rule="evenodd"`, never painted in a background colour) and are recoloured
at runtime from the active palette — see `qt_gui/icon_tint.h`.
