# Third-party notices

The original source code in this repository is under the [MIT licence](LICENSE).
Everything below was written by someone else, or is derived from their work, and
keeps its own licence. If you redistribute this project (or part of it), these
are the terms you inherit.

## Code that is in the repository

| Component | Where | Licence |
|---|---|---|
| **pbrt-v4** (Matt Pharr, Wenzel Jakob, Greg Humphreys) | Ported or adapted code in `src/shared/` (BDPT, MLT, SPPM and the other integrators, light samplers and light BVH, BxDFs, shapes, cameras, spectral code, sampling and hashing helpers, ReSTIR, BSSRDF, ...) and in the GPU kernels that mirror it. Roughly 45 files carry the pbrt-v4 attribution header, and each says what it was ported from. | Apache-2.0, full text in [`licenses/Apache-2.0-pbrt-v4.txt`](licenses/Apache-2.0-pbrt-v4.txt). Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. The ports are modified: they are rewritten as CPU/GPU templates against this project's scene interfaces, and several are adjusted where the original behaviour differs (each file's header and comments say so). Those files stay under Apache-2.0. |
| **Ray Tracing in One Weekend** series (Peter Shirley and others) | The first scenes, camera, vector and material code in `src/TheRestOfYourLife/` (each file's header says so) | CC0 1.0 Universal (public domain) |
| **NanoVDB** (Contributors to the OpenVDB Project) | `src/external/nanovdb/` | Apache-2.0 (SPDX header in the sources) |
| **tinyexr** (Syoyo Fujita and contributors; contains code derived from OpenEXR, Industrial Light & Magic) | `src/external/tinyexr.h` and `exr_reader.hh`, `streamreader.hh` | BSD-3-Clause; the licence text is at the top of the header |
| **Hosek-Wilkie sky model** (Lukas Hosek and Alexander Wilkie; coefficient tables as shipped in Blender's Cycles) | `src/shared/hosek_sky.h` (a re-write of the evaluation code) and `src/external/hosek_sky_data.h` (the tables, copied unchanged) | BSD-3-Clause, full text in [`licenses/BSD-3-Clause-Hosek-Wilkie.txt`](licenses/BSD-3-Clause-Hosek-Wilkie.txt) |
| **stb_image / stb_image_write** (Sean Barrett and contributors) | `src/external/stb_image.h`, `stb_image_write.h` | Public domain (or MIT, at your choice); see the end of each header |
| **miniz** (Rich Geldreich and contributors) | `src/external/miniz.c`, `miniz.h` | Public domain ("unlicense" statement at the end of the file) |
| **Noto Sans SC** (the Source Han Sans design: Copyright 2014-2021 Adobe, Reserved Font Name 'Source'; distributed by Google as Noto Sans CJK) | `qt_gui/fonts/NotoSansSC-Regular.ttf`, compiled into the GUI so Simplified Chinese text renders on a machine without a CJK font | SIL Open Font License 1.1 (`qt_gui/fonts/OFL.txt`; the release packages carry it as `licenses/NotoSansSC-OFL.txt`) |

## Tools and libraries used but not stored here

| Component | How it is used | Licence |
|---|---|---|
| **Qt** | The desktop GUI (`qt_gui/`) is built against your own Qt install and linked dynamically in the release packages | LGPL-3.0 (and commercial). Qt's source and licence: https://www.qt.io/ |
| **NVIDIA CUDA Toolkit** | Needed to build the OptiX backend. The runtime DLLs that the Windows release packages include are NVIDIA's redistributable files | NVIDIA CUDA Toolkit EULA |
| **NVIDIA OptiX SDK** | Needed to build the OptiX backend. **Not included in this repository or in the releases**: install it yourself and accept NVIDIA's licence | NVIDIA OptiX SDK licence |
| **Apple Metal / Foundation frameworks** | The macOS GPU backend uses the system frameworks | Apple's system licence |
| **GoogleTest** | The unit tests (fetched by CMake / vcpkg at configure time, not stored here) | BSD-3-Clause |
| **FFmpeg** (the `avcodec`, `avformat`, `avutil`, `swresample` and `swscale` DLLs) | Shipped in the Medium and Full packages because Qt's multimedia backend (the Live Preview video and the in-app video preview) needs them; we do not modify them | LGPL-2.1-or-later. Source and licence: https://ffmpeg.org/ |
| **Intel Open Image Denoise** | The AI denoiser for the CPU and Metal renderers. **Not in the repository or the packages**: the app offers to download Intel's own release once (about 57 MB on Windows), checked against a pinned SHA-256, and loads it at run time | Apache-2.0. https://www.openimagedenoise.org/ |
| **TripoSR** (Stability AI and Tripo) | The optional "Object from a photo" helper. **Not in the repository or the packages**: `scripts/setup_photo_to_mesh.ps1` downloads the model on request (about 5 GB) | MIT (its repository and model card; check them before you redistribute the weights). https://github.com/VAST-AI-Research/TripoSR |
| **Microsoft Visual C++ runtime** (`vcruntime140*.dll`, `msvcp140.dll`) | Redistributed in the Windows packages | Microsoft's redistributable terms |

## Meshes, textures and scenes

These are data, not code, and the MIT licence **does not apply to them**.

* **`models/`** holds test meshes from public 3D test collections: the Stanford 3D
  Scanning Repository, the McGuire Computer Graphics Archive (Crytek Sponza, Amazon
  Lumberyard Bistro, Rungholt) and the common-3d-test-models collection. Each model
  keeps the licence and attribution terms of its source. Some require attribution,
  some allow research use only, and some are non-commercial, so check the source
  before you redistribute a model or ship it in a product. A per-model provenance
  table has not been written yet; until it is, treat the sources above as authoritative.
  The larger scenes (Power Plant and others) are deliberately not bundled in the
  installers for this reason.
* **`images/`**: `earthmap.jpg` is NASA's Blue Marble in equirectangular projection
  (from Wikimedia Commons, public domain: created solely by NASA). `bump-marble.png`
  and `normal-checker.png` are small textures generated for this project's tests.
* **`pbrt_scenes/`**: a scene whose header says "Original, not derived from any
  distributed scene" is original to this project and under its MIT licence. Scenes
  taken or derived from the pbrt-v4 scene collection (https://github.com/mmp/pbrt-v4-scenes)
  keep the licence of that collection and of each scene's author. Several of those
  are non-commercial or no-derivatives, so check before redistributing anything you
  download into that folder.

If you find an attribution that is missing or wrong, please open an issue.
