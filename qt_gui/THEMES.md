# Theme attribution

The GUI ships thirteen colour schemes (`qt_gui/theme.cpp`). Seven of them take
their numbers from established, widely-used developer colour schemes rather
than from values invented here — they have been eye-tested by far more people
than this project has, and they land somewhere users already recognise. The
remaining six (Cyberpunk plus the five topical schemes below) are original.

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

Five further schemes are not transcriptions of anything - they are original
palettes built around a subject, each with its own decorative motif:

| Scheme | Motif | Placement |
| --- | --- | --- |
| Fantasy Parchment | compass rose, contour lines, sea-serpent flourish | bottom right |
| Sci-Fi Blueprint | drafting grid with registration crosses | tiled |
| Retro Arcade | dot matrix with sparkles and a scanline | tiled |
| Cherry Blossom | branch, blossoms and drifting petals | top left |
| Comic Pop | halftone dot field with ink sparkles and an impact burst | tiled |

**These are named for a genre, never for a franchise, and every motif is drawn
from scratch** (`qt_gui/backgrounds/*.svg`). No game or film artwork, screenshot,
logo, or name is used. Naming a theme after a real property, or shipping its
art, would be someone else's trademark and copyright — the genre carries the
same feel without borrowing anything.

All five are light schemes deliberately: a faint motif needs a light ground to
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
which is what the `QScrollArea#tabScroll > QWidget` rules (mainwindow_style.cpp)
do for the transparent colour.

The `background-image` itself, though, is applied separately — by
`ThemedScrollArea` (`mainwindow.h`) calling `viewport()->setStyleSheet(...)`
directly on its own viewport, a per-widget stylesheet, not a rule in the
app-wide one. That split exists because the app-wide stylesheet never
reliably applied a `background-image` to this specific nested selector at
runtime (confirmed by direct experiment - see `ThemedScrollArea`'s own class
comment for what was tried and ruled out); a stylesheet set directly on the
widget does.

## Your own themes

Themes are no longer compile-time only. Drop a `.theme` file (INI syntax) in
either of these and it appears in the Theme menu alongside the built-ins, with
no rebuild:

    <folder containing RayTracerGUI.exe>/themes/
    %APPDATA%/Ray Tracer Project/Ray Tracer/themes/

`qt_gui/themes/example.theme` is a complete, commented starting point — copy it
into one of the folders above and edit. Every colour role is required; `id` must
not match a built-in.

A loaded theme is not a second-class citizen: it parses into the same
`PaletteData` struct the built-ins use, so it goes through the identical
registry, lookup, menu and settings persistence with no special cases.

A file that is malformed — a missing role, a value that is not `#rrggbb`, an id
that shadows a built-in — is skipped with a warning naming the file and the
offending key. It is never fatal and never silent: a theme file is user content
and the app has to start without it. Parsing lives in the Qt-free
`palette_file.h` so those failure modes are unit-tested rather than hoped for.

## Contrast

Every palette is checked against WCAG 2.1 ratios by
`tests/unit/palette_data_tests.cpp` — body text, muted text, all twelve log
severities, the progress bar's outcome text and the primary button's label,
each against the surface it actually sits on. This exists because two bugs had
already shipped that it catches instantly: a dark-olive "100%" on a dark-olive
success fill, and gold warnings on cream.

Two deliberate exceptions:

- **Separators are decoration**, not information. WCAG exempts decorative
  content, and a divider forced to 3:1 stops dividing and starts competing with
  the text either side of it. They are checked for being perceptible, not
  readable.
- **Solarized and Nord miss AA for body text.** Both are famously low-contrast
  by design, and these are transcriptions. Editing them until they pass would
  ship something labelled Solarized that is not Solarized — someone choosing
  Solarized is asking for Solarized. They are held to a lower floor set just
  under what they measure today, so a regression still fails the build.

## Icons

The SVGs under `qt_gui/icons/` are original to this project. They are
monochrome silhouettes with real transparency (holes cut with
`fill-rule="evenodd"`, never painted in a background colour) and are recoloured
at runtime from the active palette — see `qt_gui/icon_tint.h`.
