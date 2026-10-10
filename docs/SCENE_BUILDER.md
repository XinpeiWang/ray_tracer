# Scene Builder

Build a scene from shapes, materials and lights, move things around, see a picture, and save it as an ordinary pbrt-v4 file. It is the **Scene Builder** tab of the Qt GUI (`RayTracerGUI`), and it needs no pbrt knowledge.

![The Scene Builder tab: the scene list, a top view with the camera and its field of view, a preview render, and the properties of the selected gold ball](gallery/scene-builder.jpg)

## A first scene

1. Open the **Scene Builder** tab. It starts with an example (a checkered floor, glass and gold balls, a red box, a ceiling light and a pale sky).
2. Press **Preview**. A small picture appears in a second or two (Draft quality, on the CPU).
3. Click the gold ball in the layout view and drag it. The properties on the right follow; press **Preview** again.
4. **Add** a shape or a light, pick a colour or a material in the properties, **Preview**.
5. **Save** writes a `.pbrt` file. **Render picture...** renders at the size and sample count set under *Camera and image* and saves a PNG.

The layout view shows the scene from above (**Top**), the front (**Front**) or the side (**Side**). Drag to move, mouse wheel to zoom, right-drag (or drag the background) to pan, **Frame all** to fit everything. Dragging snaps to a 0.25 grid; hold Alt (Option on a Mac) for free movement. The camera has two handles: its position and the point it looks at. Spotlights and the sun have a handle for the point they aim at. Delete removes the selected item, Ctrl+Z / Ctrl+Y undo and redo (also the buttons).

An unsaved scene is kept when you close the program and comes back when you reopen it.

## Looking at and moving things

Above the preview are four views of the scene: **Top**, **Front** and **Side** (flat views, good for exact placement) and **3D**.

In every view a click selects, and dragging an object moves it; the position is snapped to steps of 0.25 unless *Snap to grid* is off or you hold Alt (Option on a Mac).
Each drag is one undo step. In the flat views, the wheel zooms and right-drag (or a drag on the background) pans. In the **3D** view:

| Do this | To |
|---|---|
| Drag the background | Orbit around the scene |
| Right- or middle-drag, or Shift-drag the background | Pan (the Shift-drag is for a trackpad, where a right-drag is awkward) |
| Wheel | Zoom |
| Click an object, light or the camera | Select it (the camera is the little box with its view frame; a spotlight, sun or the camera also has a target you can drag) |
| **Move** tool (W): drag a selected item | Move it on the floor (Shift-drag: up and down) |
| **Move**: drag the white dot in the middle of the selected item, or hold Ctrl (Command on a Mac) while dragging | Move it **freely, in any direction**: it follows the pointer on the plane that faces the camera, so up, down, sideways, nearer or further as the mouse goes. Orbit the view to change which way "sideways" and "nearer" are |
| **Move**: drag the outer half of one of its coloured arrows (X red, Y green, Z blue) | Move it along that axis only |
| **Rotate** tool (E): drag a coloured ring | Turn the object about that world axis (in steps of 5 degrees; Alt or Option for free); the three Rotation numbers are worked out for you |
| **Scale** tool (R): drag a square handle | Stretch the object along that one of its own axes (steps of 5 %): a box's size, a cylinder's or cone's height or radius, a quad's width or depth. A sphere, a disk and a mesh scale all round |
| **Frame all** | Bring everything back into view |

Rotate and Scale apply to objects; for a light or the camera the Move tool stays.  In the Move tool only the outer part of an arrow takes hold (from about half way to its tip); a press anywhere else on the selected object, the inner part of the arrows included, slides it on the floor, not an arrow (the white dot in the middle is the free move); in the
Scale tool only the square at the end of a handle takes hold (so the scale starts at 1 and does not jump); a ring seen almost edge-on holds still instead of
spinning the object. The Rotate and Scale buttons are greyed while a light or the camera is selected, and Move shows pressed. A new object is dropped on the
floor under the middle of the view, or under what the camera looks at when the camera is nearly level (so it never lands far off-screen). A mesh file is read in
the background: it shows as the small marker until its box and points are ready, and the view does not wait for it. Big flat panels (a wall, a ceiling) are
drawn by their farthest point, so they no longer cover things standing in front of them. Pressing the Move / Rotate / Scale buttons (above the view) or the W / E / R keys
(when the view has focus) switches tool.

The 3D view draws flat-shaded shapes with the same size, position and rotation the renderer uses; it shows shape and placement, not materials or lighting (press
**Preview** for that). A mesh is drawn at its real size: its bounding box (at the object's Scale) with a sample of its vertices, so you can see how big it is and which way it faces; a file that cannot be read, or one over 80 MB, shows a small marker instead.

## Ready-made shapes

Besides the sphere, box, quad, disk, cylinder, cone and mesh file, the Add menu has shapes with no pbrt primitive of their own. They are written into the scene as triangle meshes with smooth or sharp normals as fits the shape, so they render the same on the CPU, Metal and OptiX.

| Shape | Settings | Notes |
|---|---|---|
| Pyramid | base width (X), base depth (Z), height | four sloping sides, apex above the middle |
| Wedge (ramp) | size X, Y, Z | a box cut along its diagonal: low at the front (-Z), full height at the back |
| Stairs | size X, Y, Z, steps (1 to 100) | climbs from the front (-Z) to the back |
| Torus (ring) | ring radius, tube radius | lies flat; the tube must be thinner than the ring radius |
| Capsule | radius, height | a cylinder with rounded ends; the height includes the ends (never less than twice the radius) |
| Dome (half sphere) | radius | flat bottom, `radius` tall |
| Tube (pipe) | outer radius, hole radius, height | a hollow cylinder, open at both ends |

Each one has a picture-friendly set of texture coordinates, so the checker pattern works on them.

## Keyboard

Delete or Backspace removes the selected item (from the list or a view; in a text box they edit the text). Ctrl+D (Command+D on a Mac) duplicates it. Undo and Redo use the platform's usual keys (Ctrl+Z / Ctrl+Y on Windows, Command+Z / Shift+Command+Z on a Mac); the buttons' tooltips show the ones for your keyboard. W, E and R pick the 3D view's Move, Rotate and Scale tools.

## Copies: duplicate, array and scatter

**Duplicate** (the button under the scene list, or Ctrl+D / Command+D) makes a numbered copy of the selected object or light ("Chair" gives "Chair 2") beside the original, moved by about its own width, and selects it. One undo step.

**Array...** (select an object first) makes many copies of it at once, in the spirit of Blender's Array modifier and particle scatter. It opens a window with three pages and a line that says how many objects the settings would add:

| Page | Makes | Settings |
|---|---|---|
| **Grid** | a line, a rectangle or a block of copies | how many along X, Y (up) and Z (the original is one of them), and the distance between neighbours |
| **Ring** | copies spaced evenly round a point on the floor, as chairs round a table | how many in the whole ring (the original counts as one and stays where it is), radius, centre X and Z, a turn of the whole ring, and whether each copy turns round with the ring so it faces the centre the way the original does |
| **Scatter** | copies placed at random in a disk or a rectangle | how many, the area (centre, width, depth), the floor height, the smallest and largest size (as a percentage), a random turn about the vertical, a random lean, whether footprints may overlap (and an extra gap), and an **Arrangement** number: the same number always gives the same scatter, **Another arrangement** tries the next |

By default an array copies the one object. To copy a thing made of several objects (a tree's trunk and crown, a table's top and legs, a prop or a blocky creature) tick **Copy other objects together with it** and tick its parts in the list: the window ticks the objects with the same first word in their name that stand close by, and you can change that. Each copy keeps the parts' places relative to the selected object (the one you picked is the anchor); a ring turns the whole thing, and a scatter scales and turns it as one, standing it on the floor height you set.

The copies are ordinary objects with their own numbered names: move, recolour or delete them one by one. The whole array is one undo step, and the original is never changed. One array makes at most 500 copies and 2,000 objects in all (fifty copies of a forty-block creature, say); the window says when it has cut it short, and, for a scatter that keeps copies apart, when the area is too small for them all. The turn about the vertical adds to the object's own turn, so it suits upright objects. The numbers come from a small generator written into the program, so a scatter looks the same on every platform and in every build.

A scatter is not an instance: each copy is a separate object in the saved file (a few hundred bytes of text), which is why the limit exists.

## Props

The **Props** section of the Add menu puts a few ordinary objects in at once: a **table** (top and four legs), a **chair**, a **tree** (trunk and two cones), a **snowman**, a stone **column** and a **street lamp** (its bulb is an emitting sphere). They stand on the floor with the middle of their footprint where the new item would drop, as one undo step, and the first part is selected. From then on they are plain objects: move, recolour or delete the parts one at a time (nothing is grouped), or Duplicate a part. A second copy of a prop is named apart ("Table top 2").

## Blocky objects

The **Blocky** part of the Add menu has three submenus of colourful block-style objects (in the look of block-building games; not affiliated with any of them): **Blocks** (grass, dirt, stone, planks, log, sand, glass, water, TNT, gold, diamond ore, a glowing block, lava, a pumpkin and a lantern pumpkin), **Creatures** (a green monster, a pig, a sheep, an explorer, a skeleton) and **Things** (an oak tree, a cottage, a torch, a chest, a crafting table, a furnace, a bed, two flowers, a fence, a row of rainbow-coloured wool, a glowing purple portal). A block is half a scene unit across, so a creature stands about 0.8 tall and the cottage about 4.

Each one goes in as a few dozen ordinary **Box** objects (a creature's body, legs, head and face are separate boxes; the pieces are merged so none overlap), standing on the floor under the middle of the view, as one undo step. From then on they are plain boxes: recolour a creeper's face, delete a window, move one wall. Glass and water blocks use the glass and see-through materials, gold is metal, and the glowing block, lava, torches, the portal, the lantern pumpkin and the furnace fire are light-giving boxes, so they light what is near them.

## Model library

**Add > Model library...** opens a window of ready-made meshes with a picture of each (the Stanford bunny, armadillo, Lucy and dragon, a horse, a cow, a teapot, busts, a beetle car and more: 22 in a source checkout, about 100 thousand triangles each for the big ones). Pick one and press **Add** (or double-click it): it stands on the floor under the middle of the view, about 1.6 units across, as one undo step; **Scale** in its properties resizes it and the Rotate tool turns it. The search box narrows the list by name. **Choose another file...** adds any `.obj` or `.ply` of your own instead (that one keeps its own size, so a big scan may need a smaller Scale).

The installers carry only Spot, Suzanne and the Utah teapot; the rest need a source checkout. Each model keeps the licence of its source: [MODELS.md](MODELS.md) says where each comes from and what to check before publishing.

## What the world looks like

* **+Y is up.** Units are whatever you like; the example is a few units across.
* **Rotation** is three angles in degrees, about the world X, Y and Z axes, applied in that order.
* A **sphere** is centred on its position. A **box**, **cylinder**, **cone** and the ready-made shapes below are centred on their bounding box. A **quad** is a flat rectangle in the XZ plane, a **disk** is a flat circle; a cylinder and a cone stand along +Y.
* A shape that **gives off light** lights its outer side: a box, sphere, cylinder, cone or any ready-made shape outward, a quad or disk from the side facing up. To hang a light panel from a ceiling, rotate it 180 degrees about X (the *Light panel* in the Add menu already is). Tick *Both sides* to light both.
* **Colours** are picked as ordinary (sRGB) colours and stored as linear values for the renderer. A light's *Strength* multiplies its colour, so a white panel of strength 12 has a radiance of 12.

## Sun & sky

Tick **Sun & sky** in a sky light's properties for a clear outdoor sky at a chosen time of day: **Preset** (Sunrise, Morning, Noon, Afternoon, Sunset) or **Sun height** and **Sun direction**, **Haze** (1.7 very clear, 3 ordinary, 6 and up hazy) and **Ground brightness**. Direction 0 is towards +X, 90 away from the starting camera, 180 towards -X, 270 behind it.

What it makes: the sky is the Hosek-Wilkie analytic sky model (the one Blender's Cycles offers; `src/shared/hosek_sky.h`) written as a 512 x 256 equirectangular HDR picture, `sky_v1_e<height>_a<direction>_t<haze>_g<ground>.hdr`, in the `skies` folder beside your My Scenes (`~/Library/Application Support/Ray Tracer Project/Ray Tracer/user_assets/skies` on a Mac); below the horizon the picture is the ground, its brightness times the light that falls on level ground. The scene's **Sun** (the first Distant light, or a new one named "Sun") is aimed along the sun with its colour and strength from the atmosphere the light crosses (Rayleigh and aerosol extinction): a sun on the horizon is dim and deep orange, a high one nearly white. Both are ordinary lights in the saved `.pbrt` (the picture is referenced by its full path, like any picture you pick), so every renderer reads them with nothing special.

Units: an unobstructed sun gives an irradiance of 3 on a surface facing it (what the starter scene's Sun uses) and the sky is scaled at the real ratio of sky to sun light, so **Brightness 1** is a daylight scene. A level plate under the sun and sky renders to albedo / pi times the sun's irradiance times sin(height) plus the sky picture's irradiance, which `tests/unit/scene_sky_tests.cpp` checks on the CPU renderer (within 5%). CPU and Metal agree on the three test scenes within 0.2%.

The sun follows these settings only when they change; a Sun light you move by hand stays where you put it until you change the sky again. Changing a setting makes a new picture (a few tens of milliseconds); the old ones stay in the folder.

## Materials

| In the tab | What it is | Settings |
|---|---|---|
| Matte (diffuse) | pbrt `diffuse`; optional checker pattern of two colours (every shape but a box; on a disk it makes rings), or a picture | colour or **Picture**; checker colour B and how many checks across |
| Metal | pbrt `conductor` | colour, roughness (0 is a mirror) |
| Glass | pbrt `dielectric` | index of refraction (1.5 for glass, 1.33 for water), roughness |
| Glossy paint (coated) | pbrt `coateddiffuse`: a diffuse colour (or a picture) under a clear coat | paint colour or **Picture**, coat index of refraction, coat roughness |
| Translucent | pbrt `diffusetransmission`: paper, leaves, lampshades | what it reflects, what it lets through |

**Preset.** The first row of a material's properties is a **Preset** list of ready-made materials: matte (chalk, black rubber, terracotta, concrete), plastic (red, blue, white ceramic, car paint), metal (gold, copper, silver, aluminium, chrome, brushed steel), glass (clear, frosted, water, diamond) and translucent (wax, leaf, paper). Choosing one sets the material's type, colour, roughness and index of refraction in one step (and clears a picture or checker); nothing about the preset is remembered afterwards, so the values can be tuned further. One undo step reverts it. The metal colours are the real reflectance of each metal, the glass indices the real ones. The table is `src/shared/scene_materials.h`.

A **Picture** (matte and glossy paint) replaces the colour with an image on any shape; **Clear** goes back to a colour. On a quad it also gives the quad the picture's shape and the picture stands upright when the quad is rotated 90 degrees about X (a wall). On a mesh the mesh's own texture coordinates decide where it goes. See [PHOTO_TO_SCENE.md](PHOTO_TO_SCENE.md).

**Add > Object from a photo...** turns one photo into a textured mesh with an optional helper that runs on your computer (an AI model; it has to be set up once, about 5 GB). The shape is a guess. It, and how to bring in a many-photo scan, are in [PHOTO_TO_SCENE.md](PHOTO_TO_SCENE.md).

## Lights

| In the tab | What it is |
|---|---|
| Point light | a tiny bulb: position, colour, strength |
| Spotlight | a cone of light: position, the point it aims at, cone angle, soft edge |
| Sun (distant light) | parallel light from a direction: where it shines from and towards |
| Sky | light from every direction: a colour, an equirectangular panorama image (`.exr`, `.hdr`, `.png`, `.jpg`), or **Sun & sky** (below) |
| any shape with *Gives off light* | an area light; a panel or a ball makes soft shadows |

A scene with no light renders black; the tab says so under the properties.

## Rendering

**Preview** renders a small picture (*Draft* 480 pixels wide and 16 samples per pixel, *Good* 720 and 64, *Best* 960 and 256) and shows it as large as its pane allows with the same renderer the Render tab uses, on the scene saved to a temporary file. Tick **Use the GPU** to use OptiX (Windows) or Metal (macOS); it needs a supported card and is much faster for large pictures. The button reads *Cancel* while a render runs.

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

The scene's **name** is the field at the top right of the tab (it is also the *Title* under *Camera and image*); it is what the scene list shows. **Add to scene list** never overwrites a scene: if the name is taken the copy is saved as "my-scene-2", "my-scene-3", .... A scene that is already in the list (added before, or opened from it) asks whether to *Add as a new scene* or *Update the existing one* (the list then shows the new name straight away).

**Preview live** (next to it, when Live Preview is available) saves a copy titled "<name> (live preview)" as `<name>-live-preview.pbrt` in the same folder and opens it in Live Preview; the copy is replaced each time you press the button, and the scene itself and its own listing are never touched. After moving things around there, Live Preview's **Edit in Builder** brings the arrangement back here (see [LIVE_PREVIEW.md](LIVE_PREVIEW.md)).

**Add to scene list** saves the scene into a per-user folder, `<your data folder>/user_scenes` (on Windows `%APPDATA%\Ray Tracer Project\Ray Tracer\user_assets\user_scenes`, on a Mac `~/Library/Application Support/Ray Tracer Project/Ray Tracer/user_assets/user_scenes`), so it appears in the Settings tab under **My Scenes** (a category for the scenes you made) **straight away, with no restart** (and is selected there). The scene list grows while the program runs: the library behind it lists new files (`refresh_user_scenes()` in `scene_registry.h`), and a library that already built its list - the Live Preview one, the renderer the GUI starts - finds the new scene the first time it is asked for it. That works because a scene saved there has a *persistent id*: `M<10000 + n>`, `n` remembered in the folder's `.scene_ids` file the first time any process sees the file, so the GUI, the renderer and Live Preview all call it the same thing and it never changes when other scenes come and go (an older scene's id is untouched, and ids of scenes not saved this way are numbered in the order found, as before). It is always the per-user folder, never the program's own `pbrt_scenes` folder: the scene list can only grow while the program runs from the per-user folder (a file saved into `pbrt_scenes` would show up only after a restart, and would shift the ids of the scenes found after it), it is always writable, and writing into a macOS `.app` bundle breaks the app's seal. To keep a scene somewhere else, use **Save As**; setting the environment variable `RAY_TRACER_PBRT_DIR` to a writable folder makes *Add to scene list* save there instead (listed after a restart).

**Deleting the scenes you made.** In the Settings tab, under the *My Scenes* category, **Delete Scene...** moves the selected scene's file to the Trash (the Recycle Bin on Windows) and **Delete All My Scenes...** does that for every scene you made; both ask first. The scene leaves the list at once and the selection moves to a neighbouring scene; the scenes in the other categories are never touched, and a file in the Trash can be put back. Only scenes the Scene Builder saved into its scene-list folder can be deleted here.

## What it does not do (yet)

* Several objects cannot be selected at once: an array's *Copy other objects together with it* list is how a thing made of parts is copied; moving a group of objects together is not possible yet.
* Bump or normal maps, hair, subsurface, participating media, the principled material and instanced copies: use a hand-written pbrt file for those ([PBRT_SUPPORT.md](PBRT_SUPPORT.md) lists what the renderer accepts).
* A mesh (`.ply` or `.obj`) is shown in the layout view only as a small marker at its position, because the view does not read the file; its scale and rotation apply when it renders.
* No animation or camera paths.
* The 3D view shows shapes, not materials or lighting, and a mesh as its bounding box and a sample of its points. Use it with the flat views, or type exact numbers in the properties.

## How it is checked

The scene model (`scene_model.h`), the JSON (`scene_json.h`), the checks (`scene_validate.h`) and the pbrt writer (`scene_pbrt_writer.h`) are in `src/shared/`, all pulled in by `scene_document.h` (standard library only); the undo and redo lists are `snapshot_history.h` with tests in `tests/unit/scene_builder_tests.cpp`:

* a document survives a round trip through JSON and through the pbrt text;
* every shape, material and light loads in the renderer;
* a name with a newline in it cannot add a directive to the file;
* closed-form renders: a diffuse sphere of albedo (0.5, 0.25, 0.75) under a white sky of strength 1 reads exactly that at depth 1; a sphere light of radiance 20 and radius 0.5 four units over a quad of albedo 0.6 lights the point under it by `rho * L * (r/d)^2`; and each shape emits on the side this page says it does (and not from behind).

The tab itself has a headless self-test (`RT_GUI_SELFTEST=builder`, see `qt_gui/mainwindow_selftest.cpp`): an edit, a real mouse drag in the layout view, undo and redo, a save and re-open, and a preview render. On a machine with a GPU (`scripts/gui_selftest.py --live-preview`) it also renders a preview through *Use the GPU* (Metal on a Mac) and checks it is lit and about as bright as the CPU one, and checks that *Add to scene list* from inside a `.app` writes to the per-user folder. It also checks that the scene is in the list at once under its persistent id, that the `ray_tracer` the GUI starts renders it by that id, and that the Settings tab's picker shows it selected.
