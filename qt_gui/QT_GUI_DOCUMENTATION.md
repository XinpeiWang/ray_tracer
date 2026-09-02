# Qt GUI Documentation

## Overview

A Qt6-based graphical front-end for the ray tracer, offering:
- Scene selection across all built-in scenes (category tabs, search,
  self-contained vs. requires-external-files filtering) plus live-loaded
  `.pbrt` custom scenes
- CPU/GPU (recursive or wavefront) render mode selection, and alternate
  integrators (SPPM, BDPT, MLT) with their own option panels
- 13 built-in color themes (see [`THEMES.md`](THEMES.md)) plus multi-language
  UI support
- Single-image and video-generation output modes, with an in-app Preview
  tab for finished renders
- Real-time progress, render queueing, a log/output tab, and a
  diagnostics tab

## Project Structure

```
qt_gui/
├── main.cpp                     # Application entry point
├── mainwindow.h                 # MainWindow class declaration
├── mainwindow_widgets.h         # Small reusable UI widget classes (filters,
│                                 #   toast notifications, scaled image label, ...)
├── mainwindow_jobtypes.h        # Render job/controller/diagnostics data types
├── mainwindow.cpp                     # MainWindow construction, top-level wiring
├── mainwindow_tabs.cpp                # Basic Settings + Advanced Settings tabs
├── mainwindow_tabs_render.cpp         # Render Options, Preview, Video Settings tabs
├── mainwindow_tabs_output.cpp         # Progress, Log Output, Diagnostics tabs
├── mainwindow_actions.cpp             # Menu actions
├── mainwindow_slots.cpp               # Render lifecycle slots (start/stop/complete)
├── mainwindow_style.cpp               # Stylesheet application
├── theme.h / theme.cpp / theme_load.cpp / theme_switch.cpp  # Theme system
├── palette_data.h / palette_data.cpp  # Built-in theme color palettes
├── palette_file.h / palette_file.cpp  # User-custom palette load/save
├── scene_metadata_client.h/.cpp       # Live scene metadata via scene_metadata.dll
├── scene_technique_notes.h            # Per-technique explanatory text (Preview tab)
├── recent_renders.cpp                 # Recent Renders list persistence
├── icon_tint.h/.cpp                   # SVG icon recoloring per theme
├── font_switch.cpp / language_switch.cpp  # Font and UI-language switching
├── win_taskbar.h/.cpp                 # Windows taskbar progress button (ITaskbarList3)
├── camera_math.h, render_output_parser.h, error_handler.h, settings_keys.h
├── RayTracerGUI.pro             # Qt project file (qmake)
└── build/                       # Build outputs (generated; Makefile/moc files
                                  #   under build/release are tracked in git -
                                  #   see .gitignore's own exception for why)
```

`mainwindow.h`/`mainwindow_tabs.cpp` were each split from a single much
larger file into the pieces above as the GUI grew - see git history for the
splits (`mainwindow_widgets.h`/`mainwindow_jobtypes.h` out of `mainwindow.h`;
`mainwindow_tabs_render.cpp`/`mainwindow_tabs_output.cpp` out of
`mainwindow_tabs.cpp`). Any new header containing a `Q_OBJECT` class must be
added to `RayTracerGUI.pro`'s `HEADERS +=` and carry its own full Qt
`#include` list - moc compiles each listed header as an independent
translation unit, so it can't rely on borrowing includes from whatever
`.cpp` happens to include it.

Deployed package (`RayTracer_Package/`):
```
RayTracer_Package/
├── RayTracerGUI.exe    # Qt GUI executable
├── ray_tracer.exe      # Console renderer, spawned as a subprocess by the GUI
├── scene_metadata.dll  # Scene metadata, loaded live by the GUI (Windows only)
├── Qt6*.dll            # Qt runtime libraries
├── libgcc_s_seh-1.dll, libstdc++-6.dll  # MinGW runtime
└── platforms/, styles/ # Qt plugins
```

## Building

See [`QT6_INSTALLATION_GUIDE.md`](QT6_INSTALLATION_GUIDE.md) for installing
Qt itself. To build:

```powershell
$env:Path = "C:\Qt\Tools\mingw1310_64\bin;C:\Qt\6.11.1\mingw_64\bin;$env:Path"
cd qt_gui
qmake RayTracerGUI.pro -o build\Makefile
cd build
mingw32-make -f Makefile.Release -j8
```

Or via `scripts\build_all.ps1` / `scripts\build_and_deploy.ps1` from the
repo root, which build the GUI alongside the CLI/GPU renderer and (for the
`_deploy` variant) run `windeployqt` automatically. If `RayTracerGUI.pro`'s
`HEADERS`/`SOURCES` change, re-run `qmake` before `mingw32-make` so the
Makefile picks up the new moc/compile rules.

`RayTracerGUI.exe` holds a lock on `scene_metadata.dll` while running -
close it before rebuilding, or the DLL copy step will fail.

## Integration with the Render Backend

The GUI spawns `ray_tracer.exe` as a separate process via `QProcess`
(`RenderController`, `mainwindow_jobtypes.h`/`mainwindow_slots.cpp`) rather
than linking the renderer in-process - this keeps a crash in native
CUDA/OptiX code from taking down the GUI, and lets the CLI's own stdout
progress reporting drive the progress bar directly. Command construction
follows the CLI's real argument format (`--cpu`/`--gpu`, `--output`,
width/spp/depth, the scene id, integrator flags, etc. - see
`render_flag_names.h` in `src/shared/` for the canonical flag strings both
sides share) rather than a GUI-specific wire format.

Scene metadata (name, description, performance hint, recommended SPP,
GPU-compatibility) is loaded live from `scene_metadata.dll`
(`scene_metadata_client.h`/`.cpp`), never hardcoded in the GUI - see
[`SCENE_SELECTION.md`](../docs/SCENE_SELECTION.md) for why, and rebuild
that DLL whenever `scene_registry.h` changes.

## Running

```powershell
cd RayTracer_Package
.\RayTracerGUI.exe
```

## Troubleshooting

See [`QT_GUI_TROUBLESHOOTING.md`](QT_GUI_TROUBLESHOOTING.md) for
previously-hit issues and their resolutions.

**"Application failed to start"** - missing Qt DLLs/plugins; re-run
`windeployqt RayTracerGUI.exe --no-translations` in `RayTracer_Package/`.

**"Cannot find ray_tracer.exe"** - the console renderer isn't in the same
directory as `RayTracerGUI.exe`; both are deployed together by the normal
build/packaging scripts.

**Scene list looks stale (missing a scene, wrong description)** -
`scene_metadata.dll` wasn't rebuilt after a `scene_registry.h` change, or
`RayTracerGUI.exe` was still running and holding the old DLL locked during
a rebuild attempt.
