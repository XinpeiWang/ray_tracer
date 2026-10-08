# Changelog

Release notes for the next tagged release, summarising the large correctness effort of October 2026.
Before this, see `git log`. Each item has a commit with the measurements behind it, and
[docs/PBRT_SUPPORT.md](docs/PBRT_SUPPORT.md) has the per-feature, per-backend detail.

## Unreleased

### New

* **Scene Builder** (GUI tab): build a scene from shapes, materials and lights, drag things around in a layout view, preview it, and save an ordinary `.pbrt` file. [docs/SCENE_BUILDER.md](docs/SCENE_BUILDER.md).
* **A 3D view in the Scene Builder** (the new **3D** button beside Top / Front / Side): orbit, pan and zoom around the scene, click to select, then Move (drag it on the floor, Shift-drag to lift it, or drag a coloured arrow along one axis), Rotate (drag a ring to turn it about a world axis) or Scale (drag a square to stretch it along one of its axes). Meshes are drawn at their real size (bounding box and a sample of their points). It is drawn with the Qt painter, so it needs no OpenGL or new libraries.
* **AI denoise for Live Preview on a Mac** (off by default, a checkbox in the Live Preview settings; ticking it offers to download the denoiser if it is not installed yet, and works at once after the install): Intel Open Image Denoise on the picture while it gathers samples, so it is clean right after a camera move instead of speckled; it fades out by 512 samples so a settled picture is exactly what was rendered. About 10 ms per frame. Needs the denoiser library from the Diagnostics tab. [docs/DENOISING.md](docs/DENOISING.md).
* **Live Preview on a Mac dispatches a normal frame as one command buffer** instead of about four row bands (5-6% faster frames); heavy frames are still split to stay under the GPU watchdog. [docs/MAC_LIVE_PREVIEW.md](docs/MAC_LIVE_PREVIEW.md) also now says where a GUI frame's time goes.
* **The Metal kernel is split up**: `primaryRayKernel` (1,631 lines) is now about 300 lines plus a dozen steps in `metal_poc_kernel_*.metal`, sharing state through three small bundles. No behaviour change: the Metal tests and the 95-scene parity sweep pass and render times are unchanged (pictures agree within noise; not bit-identical, see [docs/METAL_BACKEND.md](docs/METAL_BACKEND.md)).
* **The main render form remembers the scene and the image size** between runs (not samples, exposure, camera or renderer: samples, exposure and camera follow each scene's recommendation, and the GPU may not be there next time). A scene that no longer exists is skipped.
* **Live Preview on a Mac converges about 25% faster once the picture is still**: after three still frames each frame renders 2, 4 or up to 8 batches of samples (kept under about 100 ms), and goes back to one batch the moment the camera moves. `RT_LIVE_SCHEDULER=0` turns it off. [docs/MAC_LIVE_PREVIEW.md](docs/MAC_LIVE_PREVIEW.md).
* **One height for every control**: buttons, drop-downs, line edits and spin boxes are all 46px tall (a 40px box with a 3px margin above and below), on every tab. Buttons were 52px, spin boxes 51px, the Log/Diagnostics buttons 28px and a few controls set their own; they now come from one rule in the stylesheet, the Grid toggle is a real button, and a checked button (Top/Front/Side/3D, Grid) shows it. The GUI self-test checks the height of every visible control.
* **Sun & sky in the Scene Builder**: tick it on a sky light for a physically based clear sky at a chosen time of day (Hosek-Wilkie, the model Blender's Cycles uses): a sky picture and a matching Sun light (colour and strength from how low the sun is) made from the sun height, direction, haze and ground brightness, with presets from sunrise to sunset. Ordinary lights in the saved scene, so every renderer reads it; CPU and Metal agree within 0.2% and a closed-form test checks the units. [docs/SCENE_BUILDER.md](docs/SCENE_BUILDER.md).
* **Material presets in the Scene Builder**: a Preset list at the top of a material's properties sets gold, copper, chrome, brushed steel, clear or frosted glass, water, diamond, red or blue plastic, car paint, ceramic, wax, leaf, paper and more in one step; the values stay editable afterwards. [docs/SCENE_BUILDER.md](docs/SCENE_BUILDER.md).
* **Pictures in the Scene Builder**: a diffuse or glossy-paint material can take an image instead of a colour, on any shape (on a quad the picture is upright and the quad takes its shape). Meshes can be `.obj` as well as `.ply`.
* **Object from a photo** (Scene Builder, Add menu): one photo becomes a textured 3D mesh, made on your own computer by the open TripoSR model (nothing is uploaded). It is an optional helper set up once with `scripts/setup_photo_to_mesh.ps1` (about 5 GB; an NVIDIA card makes a photo take about half a minute), and the Diagnostics tab lists what it has and what is missing, with an **Install Photo Helper...** button that downloads and installs the missing parts (Windows). The shape is a guess, the back is invented. [docs/PHOTO_TO_SCENE.md](docs/PHOTO_TO_SCENE.md) also covers bringing in a many-photo scan from COLMAP or Meshroom.
* **AI denoising on the CPU** (`--cpu --denoise`, or the **AI denoiser** box with the CPU renderer): Intel's Open Image Denoise runs on the linear image before tone mapping, so `.exr` output is denoised too. It works on Windows as well as a Mac: the GUI offers to download the library (about 57 MB, checked against a pinned checksum) the first time, and the command line finds it in your user folder or `RT_OIDN_DIR`. Measured at 8 samples per pixel against a 1024-sample reference, the error drops 2.8x to 6x on the Cornell scenes. It is also given each pixel's surface colour and normal when every visible surface is matte (checkers 1.8x lower error); `--denoise-guides` forces that on for any scene. [docs/DENOISING.md](docs/DENOISING.md).
* The command line takes a `.pbrt` path where a scene id goes (`ray_tracer.exe --cpu 800 64 8 my-scene.pbrt`) and `--height N` for a non-square picture.

### Changed

* **Metal PNGs no longer get a vignette or a colour shift.** They were added to every Mac render (corners 18% darker, red and blue shifted apart) although the CPU and OptiX renderers add neither, so the same scene looked different on a Mac. They are opt-in now (`RT_METAL_POST_EFFECTS=vignette,aberration`); the light bilateral blur stays on by default until the Mac has a real denoiser.
* **AI denoising on a Mac.** The Metal renderer can now denoise with Intel's Open Image Denoise (`--denoise`, or the **AI denoiser** box in Render Options): a render with a handful of samples per pixel comes out clean. The library (~50 MB) is not inside the app: ticking the box offers to download it once from OIDN's own release page (checksum-verified). See [docs/DENOISING.md](docs/DENOISING.md).
* `--maxcomponentvalue` (the explicit firefly clamp) now works on Metal too.
* **Render passes (AOVs).** `--aovs` (or **Also write render passes** in Render Options) writes albedo, normal, depth, uv and a coverage alpha next to the image: merged into the same file for an `.exr` output (a multilayer EXR with the render's R G B), else into `<name>.aovs.exr`. Works with every renderer. See [docs/RENDER_PASSES.md](docs/RENDER_PASSES.md).

* **One table of what each renderer reads** (`src/shared/backend_capabilities.h`): the command line's "has no effect" warnings and the greyed-out Render Options controls both come from it. This fixed `--adaptive` being treated as ignored on Metal, which does read it. [docs/BACKEND_SUPPORT.md](docs/BACKEND_SUPPORT.md) is the same facts for people: CPU, both OptiX renderers and Metal in one table.
* **Live Preview controls follow the renderer's real features**, not the platform: the library says which optional features it has (`realtime_backend_features()`), and its C interface is declared once (`src/shared/realtime_api.h`) instead of copied into the GUI and each library.
* **The Scene Builder log is more complete**: a deleted object is named correctly, the last edit is written before a render and on quit, restoring the unsaved scene and a failing backup write are logged, and so are selections, the 3D tool keys, scene problems and meshes the 3D view cannot read. [docs/LOGGING.md](docs/LOGGING.md).

### Scenes

* **Every scene has a stable name** (`cornell-box`, a scene file's name) next to its short id (`A1`, `K37`). Ids still work everywhere, but they move when a category is added or a file is added to `pbrt_scenes/`; names do not. The GUI saves names (recent renders, thumbnails, notes), and the command line accepts either.
* **The scene list is tidier.** A file that a built-in scene already uses is listed once (Custom Scenes went from 218 entries to 76, most of them copies). The 58 bundled test fixtures are a **Test Scenes** category; Custom Scenes holds your own and downloaded scenes. Names say what a scene shows ("Cornell Rough Glass") instead of how it is stored ("(pbrt example)"), and a scene file can carry its own category, description and performance in header comments.
* **Descriptions and performance are real**: every test scene and 54 built-in scenes have new descriptions, and the performance word is a measured CPU time for a 400 x 400 picture (Fast under 10 s, Slow under 2 min, and so on) with its range shown.

### Rendering correctness

* **Path tracers add the last continuation ray.** The CPU and both OptiX backends stopped one segment early, so the emission or sky seen by the last
  bounce was lost: a diffuse sphere (albedo 0.5) under a white sky read 0.09 at depth 1 instead of 0.5, and a Cornell box was 1% / 3% / 10% dark at depth 8 / 4 / 2.
* **Measured BRDFs** now use pbrt's path weight `f * cos / pdf` (a white-furnace sphere read 7.8 / 8.1 / 3.5 instead of 0.1 / 0.2 / 0.4) and take direct-light samples with MIS.
* **Diffuse transmission** was wrong in three ways (lobe probability counted twice on the CPU, missing on the GPU, shadow rays passing through the shell) and the GPU never lit it from a
  point, spot or distant light. It matches the closed form of a closed shell (0.65 in the limit) on every backend, and a plate under a point light reads its closed form.
* **Coloured participating media** are sampled per colour channel on the CPU and both OptiX backends (they used one luminance extinction: a blue-heavy absorber read grey 0.47 instead of 0.82 / 0.45 / 0.17).
* **GPU sphere lights far from the shading point** no longer lose light to float32 cancellation (a light 480 units away read 65% / 73% of its closed form; now within 0.1%).
* **Cone and paraboloid lights**: the path tracer's light-sampling density missed the second crossing of a ray (3.4x too bright next to a cone light), and the paraboloid's surface area was wrong.
* **Image sky orientation** was inconsistent between the CPU's lookup and its own importance-sampling table, and between backends; all now put image row 0 straight up.

### Alternative integrators

* **BDPT and MLT**: fixed a never-set reverse density (8-25% dark in closed rooms), sphere and two-sided emitter densities, distant-light and infinite-light vertices, glossy and refracting BSDF handling, and
  pixel-filter use; scattering fog read 0.59x too dark and now matches. Disk, cylinder, cone and paraboloid lights now work. MLT was confirmed unbiased.
* **Coloured fog** under BDPT, MLT, SPPM and the debug integrators is rendered as three per-channel passes (three times the render time; they say so).
* **SPPM**: photon passes use the adjoint BSDF, the CPU gather radius scales with the scene (a fixed 10 units read 0.30 instead of 0.65 in an 8-unit room), and it warns about media (it has no volume model, like pbrt-v4's).
* **GPU SPPM** samples textures, handles a uniform sky, varies its samples per iteration, supports orthographic and equirectangular cameras and depth of field, and rebuilds its shader table when the scene
  changes (it read 0.33-0.47x of the path tracer on textured or sky-lit scenes and could crash the CUDA context).
* **`--simplepath`, `--randomwalk`, `--lightpath`** follow the same conventions as the main integrators; a visible sphere or cylinder light in `--lightpath` read 0.42x / 0.52x and now matches.

### Testing and project

* Over 5,000 automated tests, including closed-form scenes; the CPU integrator agreement tests also build as a GPU-free CMake target and are wired into GitHub CI (not yet seen to pass there).
* The slow test tier uses 8.5 GB instead of 23 GB; incremental builds and a fast test tier were added.
* A `LICENSE` (MIT), [third-party notices](THIRD_PARTY_NOTICES.md), a README gallery, a CPU-only quick start, [CONTRIBUTING](CONTRIBUTING.md), issue and pull request templates, and a code of conduct.
* A fresh clone is clean again (stale Git LFS rules removed).
* **`scripts/backend_parity.py`** compares the CPU with both OptiX renderers the way the Mac's Metal sweep does (same seed, small image, whole-image, per-channel and a 6x6 block grid): 195 of 206 scene/backend pairs pass, 5 are marginal, and 3 scenes are documented known gaps (`scripts/backend_parity_known_gaps.txt`).
* **A Windows GUI job** (`.github/workflows/windows-gui.yml`) builds the CPU-only GUI and runs the headless GUI smoke test, like the macOS release workflow.
* **The big files are split by topic** (no code changed; seeded renders are byte-identical on the CPU, both OptiX renderers): no source file is over 2,000 lines now, and the largest functions went from about 1,400 lines to a few hundred.

### Known gaps

* GPU SPPM's gather radius is a fixed 5 units; SPPM emits photons from area lights only (sky-lit scenes read 2-3% low); one SPPM scene (A7, Perlin spheres) reads about 10% low on CPU and GPU alike.
* `--lightpath` has not been checked in participating media. BDPT, MLT and SPPM do not support camera media; BDPT/MLT/SPPM do not support portal lights.
* GPU BDPT/MLT do not exist. Linux and a native arm64 macOS GUI are not supported yet.
