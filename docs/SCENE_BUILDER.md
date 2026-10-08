# Scene Builder

Build a scene from shapes, materials and lights, move things around, see a picture, and save it as an ordinary pbrt-v4 file. It is the **Scene Builder** tab of the Qt GUI (`RayTracerGUI`), and it needs no pbrt knowledge.

![The Scene Builder tab: the scene list, a top view with the camera and its field of view, a preview render, and the properties of the selected gold ball](gallery/scene-builder.jpg)

## A first scene

1. Open the **Scene Builder** tab. It starts with an example (a checkered floor, glass and gold balls, a red box, a ceiling light and a pale sky).
2. Press **Preview**. A small picture appears in a second or two (Draft quality, on the CPU).
3. Click the gold ball in the layout view and drag it. The properties on the right follow; press **Preview** again.
4. **Add** a shape or a light, pick a colour or a material in the properties, **Preview**.
5. **Save** writes a `.pbrt` file. **Render picture...** renders at the size and sample count set under *Camera and image* and saves a PNG.

The layout view shows the scene from above (**Top**), the front (**Front**) or the side (**Side**). Drag to move, mouse wheel to zoom, right-drag to pan, **Frame all** to fit everything. Dragging snaps to a 0.25 grid; hold Alt for free movement. The camera has two handles: its position and the point it looks at. Spotlights and the sun have a handle for the point they aim at. Delete removes the selected item, Ctrl+Z / Ctrl+Y undo and redo (also the buttons).

An unsaved scene is kept when you close the program and comes back when you reopen it.

## What the world looks like

* **+Y is up.** Units are whatever you like; the example is a few units across.
* **Rotation** is three angles in degrees, about the world X, Y and Z axes, applied in that order.
* A **sphere** is centred on its position. A **box**, **cylinder** and **cone** are centred on their bounding box. A **quad** is a flat rectangle in the XZ plane, a **disk** is a flat circle; a cylinder and a cone stand along +Y.
* A shape that **gives off light** lights its outer side: a box, sphere, cylinder or cone outward, a quad or disk from the side facing up. To hang a light panel from a ceiling, rotate it 180 degrees about X (the *Light panel* in the Add menu already is). Tick *Both sides* to light both.
* **Colours** are picked as ordinary (sRGB) colours and stored as linear values for the renderer. A light's *Strength* multiplies its colour, so a white panel of strength 12 has a radiance of 12.

## Materials

| In the tab | What it is | Settings |
|---|---|---|
| Matte (diffuse) | pbrt `diffuse`; optional checker pattern of two colours (every shape but a box; on a disk it makes rings), or a picture | colour or **Picture**; checker colour B and how many checks across |
| Metal | pbrt `conductor` | colour, roughness (0 is a mirror) |
| Glass | pbrt `dielectric` | index of refraction (1.5 for glass, 1.33 for water), roughness |
| Glossy paint (coated) | pbrt `coateddiffuse`: a diffuse colour (or a picture) under a clear coat | paint colour or **Picture**, coat index of refraction, coat roughness |
| Translucent | pbrt `diffusetransmission`: paper, leaves, lampshades | what it reflects, what it lets through |

A **Picture** (matte and glossy paint) replaces the colour with an image on any shape; **Clear** goes back to a colour. On a quad it also gives the quad the picture's shape and the picture stands upright when the quad is rotated 90 degrees about X (a wall). On a mesh the mesh's own texture coordinates decide where it goes. See [PHOTO_TO_SCENE.md](PHOTO_TO_SCENE.md).

**Add > Object from a photo...** turns one photo into a textured mesh with an optional helper that runs on your computer (an AI model; it has to be set up once, about 5 GB). The shape is a guess. It, and how to bring in a many-photo scan, are in [PHOTO_TO_SCENE.md](PHOTO_TO_SCENE.md).

## Lights

| In the tab | What it is |
|---|---|
| Point light | a tiny bulb: position, colour, strength |
| Spotlight | a cone of light: position, the point it aims at, cone angle, soft edge |
| Sun (distant light) | parallel light from a direction: where it shines from and towards |
| Sky | light from every direction: a colour, or an equirectangular panorama image (`.exr`, `.hdr`, `.png`, `.jpg`) |
| any shape with *Gives off light* | an area light; a panel or a ball makes soft shadows |

A scene with no light renders black; the tab says so under the properties.

## Rendering

**Preview** renders a small picture (*Draft* 320 pixels wide and 16 samples per pixel, *Good* 480 and 64, *Best* 640 and 256) with the same renderer the Render tab uses, on the scene saved to a temporary file. Tick **Use the GPU** to use OptiX (Windows) or Metal (macOS); it needs a supported card and is much faster for large pictures. The button reads *Cancel* while a render runs.

**Render picture...** uses the width, height, samples and light bounces under *Camera and image*, and asks where to save the PNG.

Problems that would stop a render (the camera at its target, a radius of zero, a mesh with no file, a spotlight aiming at itself) are listed under the properties and disable the render buttons until they are fixed.

## The saved file

*Save* writes plain pbrt-v4 text. Anything that reads pbrt can use it, including this renderer's command line:

```bash
ray_tracer.exe --cpu --output out.ppm --height 600 800 128 8 my-scene.pbrt
```

(`--height` gives a non-square picture; the arguments are width, samples, depth, then the scene, which may be a `.pbrt` path instead of a scene id.)

The file starts with a few comment lines, one of them a copy of the whole scene as JSON (`# @rt-builder-doc {...}`). That line is how the Scene Builder opens the file again for editing. Two consequences:

* **Open** only opens files the Scene Builder saved. Any other `.pbrt` renders fine from the command line or the Render tab, but has no editable form here.
* If you edit the pbrt text by hand, the Scene Builder ignores your changes when it reopens the file (it reads the comment, not the directives) and overwrites them on the next save. Keep hand-edited scenes under a different name.

**Add to scene list** saves the scene into the program's `pbrt_scenes` folder, so it appears in the Settings tab under **My Scenes** (a category for the scenes you made) **straight away, with no restart** (and is selected there). The scene list grows while the program runs: the library behind it lists new files (`refresh_user_scenes()` in `scene_registry.h`), and a library that already built its list - the Live Preview one, the renderer the GUI starts - finds the new scene the first time it is asked for it. That works because a scene saved there has a *persistent id*: `M<10000 + n>`, `n` remembered in the folder's `.scene_ids` file the first time any process sees the file, so the GUI, the renderer and Live Preview all call it the same thing and it never changes when other scenes come and go (an older scene's id is untouched, and ids of scenes not saved this way are numbered in the order found, as before). If the program sits somewhere it should not write - inside a macOS `.app` bundle (writing there breaks the app's seal) or on a read-only disk image - or is not writable, the scene goes instead to a per-user folder, `<your data folder>/user_scenes` (on a Mac `~/Library/Application Support/Ray Tracer Project/Ray Tracer/user_assets/user_scenes`), which the scene list also reads; those scenes are numbered after the built-in ones, so adding one never changes another scene's id. The program never creates a new `pbrt_scenes` folder, because a new folder in the wrong place would hide the scenes already there.

## What it does not do (yet)

* Bump or normal maps, hair, subsurface, participating media, the principled material and instanced copies: use a hand-written pbrt file for those ([PBRT_SUPPORT.md](PBRT_SUPPORT.md) lists what the renderer accepts).
* A mesh (`.ply`) is shown in the layout view only as a small marker at its position, because the view does not read the file; its scale and rotation apply when it renders.
* No animation or camera paths.
* The layout view is two-dimensional. Use the three views together, or type exact numbers in the properties.

## How it is checked

The scene model, the pbrt writer, the checks and the JSON are in `src/shared/scene_document.h` (standard library only) with tests in `tests/unit/scene_builder_tests.cpp`:

* a document survives a round trip through JSON and through the pbrt text;
* every shape, material and light loads in the renderer;
* a name with a newline in it cannot add a directive to the file;
* closed-form renders: a diffuse sphere of albedo (0.5, 0.25, 0.75) under a white sky of strength 1 reads exactly that at depth 1; a sphere light of radiance 20 and radius 0.5 four units over a quad of albedo 0.6 lights the point under it by `rho * L * (r/d)^2`; and each shape emits on the side this page says it does (and not from behind).

The tab itself has a headless self-test (`RT_GUI_SELFTEST=builder`, see `qt_gui/mainwindow_selftest.cpp`): an edit, a real mouse drag in the layout view, undo and redo, a save and re-open, and a preview render. On a machine with a GPU (`scripts/gui_selftest.py --live-preview`) it also renders a preview through *Use the GPU* (Metal on a Mac) and checks it is lit and about as bright as the CPU one, and checks that *Add to scene list* from inside a `.app` writes to the per-user folder. It also checks that the scene is in the list at once under its persistent id, that the `ray_tracer` the GUI starts renders it by that id, and that the Settings tab's picker shows it selected.
