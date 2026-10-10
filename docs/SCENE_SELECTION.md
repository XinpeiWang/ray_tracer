# Scene Selection System

## Overview

The renderer has 151 built-in scenes: the original "Ray Tracing in One Weekend" series, a wide sweep of pbrt-v4 materials, lights, cameras, volumes and shapes, imported mesh models, full textured environments, and curated demos of specific render options. Alongside them are the **Test Scenes** (the closed-form and regression scenes this project's own tests render) and any scene of your own: a `.pbrt` file dropped into `pbrt_scenes/`, or one made in the GUI's [Scene Builder](SCENE_BUILDER.md), appears without a rebuild.

Every scene renders on the CPU renderer. The GUI's scene info panel and the registry's `gpu_compatible` field say which scenes the GPU backends also support; a few scenes are CPU-only because a feature they use is not ported to the GPU.

## Names and ids

A scene has two keys.

* **A name (its "slug")**, such as `cornell-box` or `depth-of-field-pbrt-file`. It is lower case with hyphens and does not change when a category is added, when a file is added to `pbrt_scenes/`, or when a scene is renamed for display. Use it for anything you save or type: scripts, the command line, notes. For a scene found on disk the name is the file's name (`My Scene.pbrt` is `my-scene`; a scene in a downloaded collection's own folder is prefixed with the folder, `zero-day/frame25.pbrt` is `zero-day-frame25`; a clash gets `-2`). The names of the built-in scenes are written out in `src/shared/scene_slugs.h`.
* **An id**, a category letter plus a number, such as `A1` or `B10`. It is a short code, and still accepted everywhere a name is, but it is only as stable as the order of the registry: adding a category moves letters, and adding a file that sorts earlier moves the numbers of scenes found on disk.

The category letters, in GUI tab order:

| Letter | Category | Contents |
|--------|----------|----------|
| A | Basics | The original "Ray Tracing in One Weekend"/"Next Week" book scenes |
| B | Materials | BxDF / surface appearance sweep |
| C | Lights | Light types and sampling strategies |
| D | Cameras | Projection and lens models |
| E | Volumes | Participating media |
| F | Geometry | Shape primitives |
| G | Models | Single imported meshes |
| H | Large Scenes | Full textured environments |
| I | Education | Curated demos of specific Render Options controls (Sampler, Spectral rendering, Exposure, Tone mapping, Integrator) |
| J | Textures | Texture-system demos (encoding, wrap, invert, procedural texture classes, nested textures) |
| K | Custom Scenes | Your scenes and downloaded collections: `.pbrt` files found on disk that do not name another category |
| L | Test Scenes | The bundled fixtures: furnaces with exact answers, light-transport probes, one-feature regression scenes. Worth browsing (each isolates one thing) but not demos |
| M | My Scenes | Scenes you made: the ones saved from the Scene Builder with **Add to scene list** (they go to your own folder, so they survive reinstalling the program), and any scene file carrying a Scene Builder title. Shown right after Textures in the GUI, though its id letter is the last one so no existing id changed |

A name in a scene's list is what the scene shows ("Cornell Rough Glass"). A scene found on disk is listed under the title the Scene Builder saved in it, or its file name made readable (`bdpt-box-room` is "BDPT Box Room"). A "(pbrt file)" qualifier appears on six names only, where a compiled-in scene already has the plain name.

The full table (id, name, description, performance, recommended samples, GPU compatibility, camera) is `get_builtin_scene_registry()` in `src/TheRestOfYourLife/scene_registry_data.h`; this document does not copy it, because a copy would drift.

## What a scene's info says

* **Description:** one or two sentences on what the scene shows and what to look for.
* **Performance:** how long one 400 x 400 picture takes on the CPU at the scene's recommended samples (depth 8), measured on a 16-core desktop: **Fast** under 10 seconds, **Medium** 10 to 30 seconds, **Slow** 30 seconds to 2 minutes, **Very Slow** longer (the measurement stopped at 2 minutes). The CPU is used because every scene renders there; the GPU backends are several times faster, so every scene is quick on a GPU. Larger pictures cost roughly in proportion to their pixels. **Unknown** means the scene has not been measured (a scene file of your own).
* **Recommended samples** (applied when you select the scene), **GPU support**, and any assets the scene needs.

## GUI usage

The scene selector is on the **Settings** tab, in the "Scene" group box:

1. **Availability tabs:** "Self-Contained" and "Requires External Files" (mesh and texture scenes that need assets not guaranteed to be on disk).
2. **Category tabs:** one per category above.
3. **Search box:** filters the dropdown by name, id or description.
4. **Scene dropdown:** the selector.
5. **Info panel:** description, performance, recommended samples and GPU support, read live from `scene_metadata.dll`.

The GUI does not hardcode scene names or descriptions. It loads them from `scene_metadata.dll`, which wraps the same registry the renderer uses. The DLL must be rebuilt when the registry changes, and `RayTracerGUI.exe` holds a lock on it while running, so close the GUI first.

The GUI saves scenes by name: render file names (`render_cornell-box_<time>.png`), recent renders, the thumbnail cache and the technique notes all use the name, so they keep pointing at the same scene when ids move.

## Command-line usage

```
ray_tracer.exe [--cpu|--gpu] [--output PATH] [--height N] width spp max_depth SCENE [cam_x cam_y cam_z]
```

`SCENE` is a name (`cornell-box`), an id (`A1`), or the path of a `.pbrt` file. An unknown name says which scenes it might have meant.

```bash
# Cornell Box, GPU
ray_tracer.exe --gpu --output out.png 800 100 50 cornell-box

# The same scene by id, on the CPU
ray_tracer.exe --cpu --output out.png 800 200 50 A1

# A scene file of your own
ray_tracer.exe --cpu --output out.png 800 64 8 my-scene.pbrt

# Override the camera position (the scene's own recommended camera is used if omitted)
ray_tracer.exe --cpu --output out.png 800 100 50 cornell-box 278 278 -800
```

## Scene files and their header tags

Any `.pbrt` file placed in `pbrt_scenes/` (or in the folder named by the `RAY_TRACER_PBRT_DIR` environment variable) is found at startup and listed under **Custom Scenes**. A file can say more about itself in comment lines before `WorldBegin`:

```
# @rt-category Test Scenes
# @rt-description A fog ball that scatters but does not absorb, under a white sky.
# @rt-performance Fast
# @rt-gpu no
# @rt-size 555
```

`@rt-category` must be one of the categories above (an unknown one is reported and the scene stays a Custom Scene), `@rt-description` may be repeated (the lines are joined), `@rt-performance` is one of the four words, and `@rt-gpu no` marks a scene that uses something the GPU backends lack (cone and paraboloid shapes), so it is not offered a GPU render. Without a description, a scene file gets one that gives its resolution and samples. `@rt-size` is how big the scene is: the largest side of its bounding box, in the scene's own units (555 for the Cornell box, 6.4 for a small room). The Live Preview moves the camera by 2% of it per keyboard press (W, A, S, D, Up, Down), so a press is a sensible distance in a 6-unit room and in a 555-unit one; a scene that does not say gets the camera's distance to its target instead. `python scripts/stamp_scene_sizes.py` measures the scenes in `pbrt_scenes/` and writes the line (`--check` lists any that are missing or out of date, and `ray_tracer --print-scene-size <scene id, name or file>` prints one scene's size); the Scene Builder writes it for its own scenes. A `.pbrt` file that a built-in scene already uses is listed once, as the built-in scene. See [`PBRT_SUPPORT.md`](PBRT_SUPPORT.md) for which pbrt-v4 directives the loader reads, and [`pbrt_scenes/README.md`](../pbrt_scenes/README.md) for the folder.

## Adding a built-in scene

1. Put the scene's `.pbrt` file in `pbrt_scenes/`.
2. Add a `SceneNames::` constant in `src/shared/scene_descriptor.h` (the name says what the scene shows, not how it is stored).
3. Add its row to `src/shared/scene_slugs.h` (`kBuiltin`): the name that will not change.
4. Add a `build_curated_pbrt_scene_descriptor(...)` row in `src/TheRestOfYourLife/scene_registry_data.h`: id, name, category, description, performance, file.
5. Add its technique note to `qt_gui/scene_technique_notes.h`, keyed by the name. The notes are translated GUI text, so they stay in the GUI rather than in the (Qt-free) registry; `RT_GUI_SELFTEST=scenekeys` (see `qt_gui/mainwindow_selftest.cpp`) fails when a self-contained scene has no note or a note belongs to no scene.
6. Update the pinned counts in `tests/unit/scene_registry_tests.cpp`; the slug-table and name tests fail until steps 3 and 2 are done.

A test fixture needs none of this: give the file the tags above and it is listed.

## Scenes that need files

A scene flagged "requires external files" (the Models and Large Scenes, mostly) reads meshes and textures that are not part of every checkout, and a missing mesh stops the render with an error (exit code 3, "file not found") naming the files, since a statue scene without its statue is not a result. A scene that is not flagged and names a mesh it cannot read skips it with a warning ("could not be read" in the Log tab) and draws the rest. The GUI shows which folder the files belong in as soon as the scene is selected.

a self-contained scene that names a mesh which cannot be read skips it with a warning (see "could not be read" in the Log tab) and draws the rest. **Download missing files.** For the statue models (the "Models" scenes whose files live in this repository's `models/` folder) the GUI also offers a **Download missing files** button: it fetches them from this repository, checks each file's size and SHA-256, and saves them in a per-user folder the renderer also searches (`~/Library/Application Support/Ray Tracer/user_assets` on macOS, set through `RAY_TRACER_USER_ASSETS`), so it works from a read-only disk image too (the list is `qt_gui/downloadable_assets.txt`, regenerated by `scripts/update_asset_manifest.sh`). 

The large third-party scenes are not hosted by this project, so for those the button fetches the original files straight from their authors' sites: the McGuire Computer Graphics Archive scenes (Sponza, Bistro, Rungholt, San Miguel, ...; each archive is read member by member over HTTP range requests and every member is CRC-32 checked) and the folders of the official [pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes) repository (Contemporary Bathroom, Barcelona Pavilion, Subsurface Dragon, Ganesha, Sports Car, Crown, Villa, ...; pinned to a commit and checked against git blob SHA-1). The button shows the scene's copyright and licence text, the download size and the disk space needed before anything is fetched, stops if the disk is too small, resumes by skipping files that are already complete, and gives up with an error if no data arrives for a minute. Everything lands in the same per-user folder. The catalogue is `qt_gui/scene_packs.txt`, generated by `python3 scripts/gen_scene_packs.py` (`--check` re-verifies it against the upstream sites); these scenes stay under their own licences, none of which this project grants.

## Troubleshooting

**"Requires external assets" warning:** the scene needs mesh or texture files not present on disk. The scene's info panel says what is needed, and the GUI can download the large third-party scenes; some assets are tracked with Git LFS (`git lfs pull` if they show up as small pointer files).

**A scene renders too slowly:** reduce samples or resolution for a preview, and look at the info panel's performance first; volumes and high-poly meshes are inherently slower.

**The GPU renders differently from the CPU:** a few scenes are CPU-only or GPU-only for feature-specific reasons (the GPU-support line in the info panel and the registry's `gpu_compatible` field flag them). Everything else should agree across the CPU, GPU-recursive and GPU-wavefront renderers within normal Monte Carlo noise.
