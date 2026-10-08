# Project structure

Where things are in the repository. The README's [Building from Source](../README.md#-building-from-source) and [BUILD.md](../BUILD.md) say how to build it.

```
ray_tracer/
├── src/                          # Ray tracing library
│   ├── TheRestOfYourLife/        # The primary CPU path tracer (materials, lights, cameras, scenes,
│   │                              #   pbrt-v4 builders, BDPT/MLT/SPPM integrators). Named for its origin
│   │                              #   in the "Ray Tracing in One Weekend" book series - it has long since
│   │                              #   outgrown that book's own scope; see the directory's own README.
│   ├── shared/                   # CPU/GPU-shared headers: pbrt-v4-style BxDFs, cameras, lights, sampling,
│   │                              #   the pbrt scene loader/flattener, scene_slugs.h (scene names),
│   │                              #   scene_document.h (the Scene Builder's model)
│   ├── data/                     # Precomputed lookup tables (sampling sequences, spectral data)
│   └── external/                 # Third-party headers (stb_image, NanoVDB, etc.)
│
├── launcher/                     # Unified launcher: the ray_tracer.exe entry point
│   ├── main.cpp                  # CPU/GPU/SPPM/video dispatch
│   ├── launcher_args.h           # Command-line parsing
│   ├── camera_path.h             # Video camera paths
│   └── launcher.vcxproj          # Visual Studio project (auto-deploys to RayTracer_Package/)
│
├── cpu_renderer/                  # CPU path tracer (static library)
│   ├── cpu_interface.cpp/.h      # C API for CPU rendering
│   └── cpu_renderer.vcxproj      # Visual Studio project
│
├── optix_renderer/                # OptiX GPU renderer (static library, thin VS-project wrapper -
│   └── optix_renderer.vcxproj    #   the real GPU implementation lives in gpu/optix/ below)
│
├── gpu/metal/                     # Metal GPU backend (macOS): runtime-compiled .metal shaders, the pbrt
│                                  #   loader, Live Preview, and the CPU-vs-Metal parity harness
│                                  #   (metal_cpu_gpu_parity_check.cpp + parity_golden.txt)
│
├── realtime_renderer/             # realtime_renderer.dll/.dylib - what the GUI's Live Preview talks to
├── scene_metadata/                # scene_metadata.dll/.dylib - the scene registry the GUI loads at runtime
│
├── gpu/optix/                     # OptiX GPU implementation - all three GPU backends share this one
│   │                              #   flat directory (a single OptiX pipeline/PTX build), distinguished
│   │                              #   by filename prefix rather than subdirectory:
│   ├── optix_programs.cu         # Recursive (mega-kernel) backend - bare names, no prefix
│   ├── optix_device_helpers.h    # Recursive backend's shared __device__ helpers (material shading, NEE)
│   ├── wavefront_*.cu/.h         # Queue-based wavefront path tracer (alt. GPU backend)
│   ├── sppm_*.cu/.h              # GPU SPPM (photon mapping) backend
│   ├── optix_renderer*.cpp/.h    # OptiX host-side renderer (init/scene/render split across 3 files)
│   ├── optix_interface.cpp/.h    # C API wrapper
│   ├── scene_builder.cpp/.h      # Native demo-scene + pbrt-loaded-scene conversion to OptiX format
│   ├── pbrt_gpu_builder.h        # Flattened pbrt scene -> GPU SceneData (the loader's GPU-side half)
│   └── optix_types.h             # Shared structures (materials, geometry, launch params)
│
├── qt_gui/                        # Qt 6 graphical interface
│   ├── RayTracerGUI.pro          # Qt project file
│   ├── mainwindow.h/.cpp         # Main window class + construction
│   ├── mainwindow_tabs.cpp       # Settings, Render Options and Preview tab construction
│   ├── mainwindow_slots.cpp      # Signal/slot handlers (stop/pause, output mode, presets); the rest are split by topic:
│   │                              #   mainwindow_queue / _render_events / _scene_info / _thumbnails / _downloads / _diagnostics_log .cpp
│   ├── mainwindow_tabs_render*.cpp  # Render Options + Preview tabs (the option groups are _groups.cpp); mainwindow_live_preview / _live_settings .cpp are the Live Preview session and its saved settings
│   ├── mainwindow_style.cpp      # Theme/QSS application
│   ├── scene_builder_*.cpp/.h    # The Scene Builder tab (widget, property panel, shared helpers); the
│   │                              #   scene model, JSON, checks and pbrt writer are src/shared/scene_{model,json,validate,pbrt_writer}.h (scene_document.h includes them)
│   ├── scene_layout_view.*       # The Scene Builder's 2D layout view
│   ├── scene_technique_notes.h   # Per-scene "what technique does this show" text (translated)
│   ├── translations/             # raytracer_{es,fr,ja,zh_CN}.ts - UI translations (lupdate/lrelease)
│   └── (Qt build output)         # Builds to RayTracer_Package/
│
├── models/                        # Mesh (.obj) and texture assets, Git LFS for the large ones
├── images/                        # Texture images used by the built-in scenes (earth map, normal/bump maps)
├── pbrt_scenes/                   # Scene files: the 58 bundled test scenes (category L) and your own (see pbrt_scenes/README.md)
├── resources/                     # Application icon and Windows resource files
│
├── tests/                         # Google Test suite (unit/ and integration/)
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── scripts/                       # Build and deployment scripts
│   ├── build_all.bat/.ps1        # Build all components
│   ├── deploy_launcher.ps1       # Verify the launcher deployed to RayTracer_Package/
│   ├── build_and_deploy.ps1      # One-command build + deploy
│   ├── deploy_qt_gui.ps1         # Qt dependency deployment
│   └── setup_env.bat/.ps1        # Environment setup
│
├── docs/                          # Guides and references - start at docs/README.md
│
├── RayTracer_Package/              # Deployment output (single canonical location)
│   ├── RayTracerGUI.exe          # Qt GUI (built from qt_gui/)
│   ├── ray_tracer.exe            # Console launcher (auto-deployed from launcher/)
│   ├── optix_programs.ptx        # GPU shader (auto-deployed from optix_renderer/)
│   └── Qt6*.dll + plugins        # Qt dependencies (deployed by scripts/deploy_qt_gui.ps1)
│
├── README.md                      # The project's front page
├── BUILD.md                       # Build instructions (Windows and macOS)
├── INSTALL.md                     # Using the portable release
├── CONTRIBUTING.md, CODING_STANDARDS.md, CHANGELOG.md
└── ray_tracer.sln                 # Visual Studio solution
```

**Key Directories:**
- **src/TheRestOfYourLife/** and **src/shared/** - Active production codebase (materials, lights, cameras, scenes)
- **gpu/optix/** - GPU implementation, mirrors most of the CPU feature set (see [Known Limitations](../README.md#-known-limitations) for gaps)
- **models/** - External mesh/texture assets (Git LFS)
- **RayTracer_Package/** - Single canonical deployment directory (auto-populated by builds)
- **scripts/** - All build/deploy automation
- **docs/** - Feature guides and architecture notes
