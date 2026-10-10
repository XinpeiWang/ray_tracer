# Live Preview: a user's guide

Live Preview shows a scene while you move around in it. It renders over and over at a small size, and the picture sharpens as long as you hold still: move the camera and it goes noisy again at once, stop and it clears up in a second or two. It is for choosing a view, checking lighting and trying out an arrangement before you spend minutes on a full render. (The developer's notes are in [MAC_LIVE_PREVIEW.md](MAC_LIVE_PREVIEW.md).)

## Starting it

1. Open the **Settings** tab, pick a scene, and set **Output Mode** to **Live Preview (interactive)**.
2. Press **Render**: in this mode it starts the preview instead of a render. A **Live Preview** tab opens inside the Preview tab and fills in. The first time ever on a Mac the program compiles its shaders, which takes about 15 seconds; later starts are quick.
3. **Stop Render** (or Esc) ends it and leaves the last picture on screen.

The preview has the proportions of the image size in Settings (its longer side is 400 pixels), so it frames the scene the way the final render will. A scene that needs files you have not downloaded (Sponza, Bistro, large environments) says so instead of starting: use the **Download missing files** button under the scene's description in the Settings tab.

## Moving the camera

| To | Do this |
|---|---|
| Orbit around the scene | drag with the mouse, or press Left / Right |
| Zoom | scroll the wheel, or press + / - |
| Fly | W / S forward and back, A / D left and right, Up / Down up and down |

Orbit and zoom turn around a point in front of the camera; flying moves that point with the camera. The Settings tab has **Mouse Sensitivity** and **Keyboard Sensitivity** (how far one drag or one key press goes). A key press is a fraction of the scene's own size, so one press is a sensible step in a small room and in a large scene alike.

## Settings that change the picture

These are in the **Render Options** tab, in the group for Live Preview.

* **Exposure**, and **Auto exposure** (on by default on a Mac): brightens a scene that would otherwise come up nearly black.
* **Samples/Frame** and **Max Bounces**: more samples per frame is smoother but slower to react; fewer bounces is faster but darker in closed rooms.
* **Firefly Clamp**: limits the very bright single pixels that noise produces.
* **Smooth noisy pixels** (on by default on a Mac): hides the noise of pixels that have few samples yet. It only changes what is shown, never the accumulated picture.
* **Depth of Field**, with **Aperture** and **Focus Distance**: blur by distance, in the scene's own units.
* **AI denoise** (Mac, off by default): cleans the picture while it gathers samples; it offers to download the denoiser the first time you tick it.

## Moving objects (Mac)

Under the picture there are three buttons. **Move objects** switches the mouse over to the objects in the scene; the other two work on what you have moved.

* **Pick**: move the cursor over an object and a thin white box shows what a click would take. Click it and a yellow box marks it.
* **Drag** to slide it across the floor. Hold **Shift** while dragging to lift or lower it.
* **Keys**: with an object selected, W / S, A / D and Up / Down move it instead of the camera, in the directions the picture shows (forward is away from you, along the floor). You can orbit, zoom or fly with it selected: the yellow box follows.
* **Click on empty space** to let go of the object. Dragging there orbits the camera as usual.
* **Reset objects** puts everything back where the scene file has it.
* **Save arrangement** keeps what you have done: it saves the scene, with the objects where they are now, as a new scene in the scene list under **My Scenes**. Your original scene is not changed. A scene you made in the Scene Builder opens in the Builder with the objects in their new places, so you can carry on editing it there.

What counts as "an object": everything in one group of the scene file (a box written as six rectangles moves as one box), or a single shape that stands on its own. Things that are not plain shapes (hair, instanced copies, curves) cannot be picked; a click on one orbits instead. Moves are not saved unless you press **Save arrangement**, and every new Live Preview starts from the scene as its file has it. Each move rebuilds the scene (about a tenth of a second for a small one), so on a heavy scene the picture follows a drag a few times a second rather than smoothly.

## If something looks wrong

* **Black or very dark**: raise Exposure, or leave Auto exposure on. A scene lit only by a small lamp can need more bounces.
* **Very noisy or flickering**: hold still; it clears. Turn **Smooth noisy pixels** on, or raise Samples/Frame.
* **It will not start on a scene**: a few scenes use features the Mac's GPU renderer lacks (see [BACKEND_SUPPORT.md](BACKEND_SUPPORT.md)); render those with an ordinary Render (Output Mode: Render Single Image) instead.
* **Slow on a large scene**: use fewer Samples/Frame or fewer Max Bounces (the image size in Settings changes the preview's proportions, not how big it is).
