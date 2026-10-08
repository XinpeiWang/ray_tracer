# Changelog

Release notes for the next tagged release, summarising the large correctness effort of October 2026.
Before this, see `git log`. Each item has a commit with the measurements behind it, and
[docs/PBRT_SUPPORT.md](docs/PBRT_SUPPORT.md) has the per-feature, per-backend detail.

## Unreleased

### New

* **Scene Builder** (GUI tab): build a scene from shapes, materials and lights, drag things around in a layout view, preview it, and save an ordinary `.pbrt` file. [docs/SCENE_BUILDER.md](docs/SCENE_BUILDER.md).
* **A 3D view in the Scene Builder** (the new **3D** button beside Top / Front / Side): orbit, pan and zoom around the scene, click to select, drag an object on the floor (Shift-drag to lift it) or drag a coloured arrow to move it along one axis. It is drawn with the Qt painter, so it needs no OpenGL or new libraries.
* **Pictures in the Scene Builder**: a diffuse or glossy-paint material can take an image instead of a colour, on any shape (on a quad the picture is upright and the quad takes its shape). Meshes can be `.obj` as well as `.ply`.
* **Object from a photo** (Scene Builder, Add menu): one photo becomes a textured 3D mesh, made on your own computer by the open TripoSR model (nothing is uploaded). It is an optional helper set up once with `scripts/setup_photo_to_mesh.ps1` (about 5 GB; an NVIDIA card makes a photo take about half a minute), and the Diagnostics tab lists what it has and what is missing, with an **Install Photo Helper...** button that downloads and installs the missing parts (Windows). The shape is a guess, the back is invented. [docs/PHOTO_TO_SCENE.md](docs/PHOTO_TO_SCENE.md) also covers bringing in a many-photo scan from COLMAP or Meshroom.
* The command line takes a `.pbrt` path where a scene id goes (`ray_tracer.exe --cpu 800 64 8 my-scene.pbrt`) and `--height N` for a non-square picture.

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

### Known gaps

* GPU SPPM's gather radius is a fixed 5 units; SPPM emits photons from area lights only (sky-lit scenes read 2-3% low); one SPPM scene (A7, Perlin spheres) reads about 10% low on CPU and GPU alike.
* `--lightpath` has not been checked in participating media. BDPT, MLT and SPPM do not support camera media; BDPT/MLT/SPPM do not support portal lights.
* GPU BDPT/MLT do not exist. Linux and a native arm64 macOS GUI are not supported yet.
