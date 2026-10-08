# From a picture to a scene

Three ways to get a real-world picture into a rendered scene, from simple to ambitious. The first two are in the
[Scene Builder](SCENE_BUILDER.md); the third is a workflow using free tools, with the renderer at the end of it.

| You have | You want | Use |
|---|---|---|
| any picture | it on a wall, a floor, a ball, a statue | **Pictures** (built in, nothing to install) |
| **one** photo of an object | a 3D version of it to light and render | **Object from a photo** (optional helper, runs on your computer) |
| **many** photos of an object or place | an accurate 3D copy | **A photo scan** with COLMAP or Meshroom, then bring the mesh in |

## Pictures

A diffuse or glossy-paint material can be coloured by a picture instead of a flat colour: pick the material's
**Picture** in the Scene Builder's properties (**Clear** goes back to a colour). PNG, JPEG, BMP, TGA, EXR and HDR work.

- On a **quad**, choosing a picture also gives the quad the picture's shape, and the quad is written so the picture is the
  right way up and not mirrored when you stand the quad up as a wall (rotate it 90 degrees about X; it then faces +Z, towards
  the default camera).
- On a **sphere**, the picture wraps around it (its poles are on the Z axis, so with the default camera you look at a pole).
- On a **mesh**, the mesh's own texture coordinates decide where the picture goes. A mesh without them shows one colour.
- The picture is read as an ordinary (sRGB) picture, so what you see is the colour you put in.
- The scene file stores the picture's path. Keep the picture where it is, or choose it again after moving the scene. The Scene Builder warns when a
  picture, mesh or sky file it points to is no longer there (the renderer would otherwise just render without it).
- A phone photo is often stored sideways with a rotation tag that the renderer ignores. When a picture has one, the Scene Builder saves an upright PNG
  copy under its data folder (`pictures\`) and uses that, so the picture is the right way up and a quad takes the right shape.

For a sky, give the **Sky** light an image instead (an equirectangular panorama).

## Object from a photo

**Add > Object from a photo...** turns one photo into a textured 3D mesh and puts it in the scene, standing on the floor and
about two units tall. A model called [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (MIT licence) guesses the
shape. It runs on your own computer; **the photo is not uploaded anywhere**.

It is a guess, not a scan. Expect:

- the front to look right and the **back to be invented**, so check the object from every side before relying on it;
- soft detail: a face, text or thin parts (a chair's slats, a plant) come out smooth or blobby;
- the colour to be baked into the surface, so highlights and shadows in the photo stay in the texture. Light the object
  evenly (soft light, no hard shadows) and it re-lights well; a photo with strong sun on one side keeps that side bright.
- one object per photo, on a plain background if you can. The background is removed automatically (a PNG that already has
  a transparent background is used as it is).

### Setting it up (once)

The helper is **optional**: it needs PyTorch and about 5 GB of disk, so it is not part of the program. From the program's
folder, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_photo_to_mesh.ps1
```

This needs Python 3.10, 3.11 or 3.12 and git on your PATH. It makes a private Python environment in
`%LOCALAPPDATA%\RayTracerPhoto` (it never touches your other Python installs), installs PyTorch (the NVIDIA build when it
finds an NVIDIA card, otherwise the processor-only build), installs TripoSR at the version this was tested with, and downloads the
model weights (about 1.7 GB, from Hugging Face) and the background-removal model (U2-Net, Apache-2.0, about 176 MB). Then use the menu as above.

An NVIDIA card with about 6 GB of memory makes a photo take seconds; on the processor it takes several minutes. On a Mac see the next section.

**From the program (Windows).** Open the **Diagnostics** tab, press **Run Diagnostics**, and if its *Photo helper* section
lists something missing, press **Install Photo Helper...**. It says what will be downloaded and from where, shows the exact script it will run, asks you to confirm, runs the same
setup script with its output in a window (Cancel stops it and everything it started), and runs the diagnostics again when it is done, so the
report shows the result. It needs Python 3.10 to 3.12 and git on your PATH; on macOS and Linux the button is disabled and the guide's other steps apply.

**Checking what is installed.** Run the **Diagnostics** tab: its last section, *Photo helper*, lists the helper script, the
Python environment, each package with its version (PyTorch, Transformers, rembg, xatlas, scikit-image, SciPy, trimesh, NumPy,
Pillow, einops, OmegaConf, Hugging Face Hub, ONNX Runtime), the graphics card PyTorch can use, the TripoSR code, and whether the two
model files are already downloaded. Anything missing or unusable is flagged (and the setup script is named), so a broken install
shows up there instead of as a failed import halfway through a photo. The same list is printed by
`python tools\photo_to_mesh\photo_to_mesh.py --check`.

If you keep the environment somewhere else, set `RAY_TRACER_PHOTO3D_PYTHON` to its `python.exe` and pass `-Folder` to the
setup script. Keep the folder name short: PyTorch's folders are deep and Windows stops at 260 characters.

### On a Mac

The same helper works on macOS; the installer is a bash script that ships inside the app (`Contents/MacOS/scripts/setup_photo_to_mesh.sh`). The easiest way is the **Diagnostics** tab: *Run Diagnostics*, then *Install Photo Helper* (it asks first, shows the output while it runs, and Cancel stops it and everything it started). Or run it yourself:

```bash
bash "/Applications/RayTracerGUI.app/Contents/MacOS/scripts/setup_photo_to_mesh.sh"
```

It needs **Python 3.10, 3.11 or 3.12** and **git**, which a Mac does not have by default: `brew install python.12`, or the installer from python.org, and `xcode-select --install` for git. The environment goes to `~/Library/Application Support/RayTracerPhoto` (set `RAY_TRACER_PHOTO3D_PYTHON` to its `venv/bin/python` if you keep it elsewhere).

- **Apple silicon** uses the Apple GPU through PyTorch's MPS backend; if MPS cannot run something the helper says so and finishes on the processor (several minutes). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set for it.
- **Intel Macs** get PyTorch 2.2.2, the last version built for them, and always run on the processor (several minutes per photo).
- The results folder is `~/Library/Application Support/Ray Tracer Project/Ray Tracer/photo_meshes`.

### Where things go

Each photo's result is kept in the program's per-user data folder, in `photo_meshes\<photo name>-<date and time>\`
(on Windows `%APPDATA%\Ray Tracer Project\Ray Tracer\photo_meshes`), as `mesh.obj` and `texture.png`. A saved scene points at
those files, so keep that folder (or copy the two files and choose them again) if you move the scene to another computer.

The Diagnostics report's *Saved Photo Objects* line says how many of these folders there are and how much space they take. Nothing deletes them
automatically (a saved scene may point into one), so delete the ones no scene uses when you want the space back.

You can also run the helper by itself:

```powershell
%LOCALAPPDATA%\RayTracerPhoto\venv\Scripts\python.exe tools\photo_to_mesh\photo_to_mesh.py photo.jpg --out out_folder
```

`--resolution 384` gives a finer mesh (slower), `--texture-size 2048` a sharper texture, `--no-remove-bg` skips the
background removal for a photo that already has a plain background.

### Taking a good photo

One object, in the middle, filling about two thirds of the picture, shot from slightly above and from the front (three-quarter
views work well), against a plain background, in even soft light. Avoid glass, mirrors and very shiny things (the model sees
the reflections as part of the object), and objects cut off by the edge of the photo.

### If it does not work

| Message or symptom | What to do |
|---|---|
| "The photo helper has not been set up" | Run the setup script above. The Diagnostics tab shows what is missing. |
| "found no object in the photo" | Use a clearer photo of one object on a plain background. |
| runs out of graphics memory | Close other programs that use the GPU, or run the helper by hand with `--device cpu` (slow). |
| the object comes out lying down or upside down | Use the Rotation fields on the object (the model assumes the photo shows it upright). |
| setup fails with a "path too long" error | Pass a shorter `-Folder`. |

## A photo scan (many photos)

For an accurate copy of a real object or place, take 30 to 100 overlapping photos while moving around it and let
photogrammetry software reconstruct it. Free choices:

- **[COLMAP](https://colmap.github.io/)** (command line and GUI): finds the camera positions and a dense point cloud, then
  *Poisson meshing* gives a mesh.
- **[Meshroom](https://alicevision.org/#meshroom)**: a friendlier interface over the same kind of pipeline, and it can bake
  a texture for you.
- **Apple's Object Capture** (macOS) and phone apps (Polycam, Scaniverse, RealityScan, Luma) do the whole job and export a mesh.

Export a **mesh with a texture** as `.obj` (or `.ply`), then in the Scene Builder:

1. **Add > Mesh...** and choose the file. Set **Scale** until it is the size you want (scans come out in arbitrary units).
2. Set the material's **Picture** to the texture image the scan came with.
3. Add lights and a sky, and render.

Things that do not carry over yet:

- A scan with **several textures or materials** (a mesh made of separate pieces, each with its own picture) shows one picture
  over the whole mesh; use one texture atlas when you export.
- A mesh that has only **per-vertex colours** (no texture) renders in one flat colour. Bake the colours to a texture in
  MeshLab or Blender, or export with a texture.
- Scans are **not relightable** beyond the colour they carry: shadows and highlights from the day they were photographed are
  in the texture. Photograph in even, overcast light to keep that small.

## Why not turn a photo into a whole scene?

One photo does not hold enough information to rebuild a room: depth is ambiguous and everything hidden is unknown. Research
systems guess room layouts, but they are not reliable enough to ship. Neural scene models (NeRF, Gaussian splatting) give
beautiful new views of a scene but are not geometry and materials a path tracer can re-light, so they are not offered here.

## For developers

- The helper's pure parts have tests that need no model: `%LOCALAPPDATA%\RayTracerPhoto\venv\Scripts\python.exe -m unittest tools/photo_to_mesh/test_photo_to_mesh.py`
  (mesh orientation, the colour baker, the `--check` report; they skip without numpy and scipy).
- The Qt-free text handling (`src/shared/photo_helper_report.h`: progress lines, the missing-facts parser, the line splitter that copes with
  `\n`, `\r\n`, PowerShell's `\r\r\n` and pip's lone `\r`) is in `tests/unit/photo_helper_report_tests.cpp`.
- The GUI self-test modes `photo` (needs the helper installed) and `installphoto` (stand-in installer scripts, incl. Cancel; `python3 scripts/gui_selftest.py APP --modes installphoto`) drive the whole flow without dialogs; see
  `qt_gui/mainwindow_selftest_photo.cpp`.
