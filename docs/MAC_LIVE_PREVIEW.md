# Live Preview on macOS (Metal)

Live Preview (Output Mode -> "Live Preview (interactive)") renders continuously at a small fixed size so you can drag to orbit
the camera and watch the image sharpen. On Windows it is backed by OptiX; on macOS it is backed by the Metal path tracer.

## How it fits together

* The GUI (`qt_gui/realtime_preview_session.cpp`) runs a worker thread that calls one function per frame,
  `realtime_render_frame(...)`, from `realtime_renderer.dll` / **`realtime_renderer.dylib`**, and accumulates the frames itself
  (running mean per pixel, reset when the camera moves).
* `realtime_renderer/realtime_renderer_mac.cpp` exports that exact C ABI (34 arguments, hand-duplicated by the GUI) and forwards to
  `gpu/metal/metal_live_preview.h`.
* `metal_live_preview.mm` keeps ONE persistent session per (scene, size): the pbrt scene, acceleration structures and shader
  pipeline are built on the first frame (~0.25 s, ~15 s the very first time ever while macOS compiles the shaders); every later
  frame only updates the camera, frame seed and sample count and dispatches. This works because
  `MetalPocApp::compileShaderAndDispatch()` now keeps its per-frame dispatch as a re-runnable function (`liveRender`) that owns every
  GPU resource it needs; a normal single-image render is unchanged (the golden snapshot shows no drift).
* The library returns the plain linear mean of `samples_per_pixel` samples, plus (for the GUI's temporal reprojection) the world position of
  the ray through each pixel's centre and the camera basis, and honours the depth-of-field override. The other OptiX-only flags in the
  signature (AI denoiser, SVGF, ReSTIR, radiance cache, path guiding, temporal/neural upscale, NRC, adaptive sampling) are accepted and
  ignored. The GUI hides those controls on macOS: the Live Preview Settings group shows only Depth of Field, Aperture/Focus Distance,
  Exposure, Samples/Frame, Max Bounces and Firefly Clamp, and the Live denoiser group is hidden.

## Performance (M2, measured with `build/metal_live_bench`)

| scene | size | spp/frame | frame time |
|---|---|---|---|
| A1 (Cornell, glass sphere) | 480x480 | 2 | ~35 ms (28 fps) |
| A1 | 640x640 | 4 | ~120 ms (8 fps) |
| G1 (69k-triangle mesh) | 480x480 | 2 | ~26 ms (39 fps) |
| A1 inside the GUI (400x300 preview) | 400x300 | 1 | ~17 ms (58 fps) |

## Finding the scene files

Live Preview runs inside the GUI process, so unlike a render job (a subprocess the GUI starts with the right working directory) it
cannot count on the current directory: a Finder/Dock launch starts in `/`. `metal_live_preview.mm` therefore resolves the registry's
relative scene path against the current directory, the executable's directory (`Contents/MacOS`), the library's directory and their
parents, and reports where it looked if the file is missing. (v12 shipped without this and showed "the pbrt scene failed to load";
`scripts/gui_selftest_macos.sh` now launches the app from `/` like Finder does, and `ctest -R metal_realtime_dylib_other_cwd` calls the
library from a directory without scenes.)

## Limits

* Reprojection is the GUI's nearest-pixel kind (the same as with OptiX): after a camera move the previous accumulation is remapped onto
  the new view wherever the surface point matches, so the picture stays recognisable instead of restarting from noise. It shows the usual
  artefacts of that method (streaks on bright emitters, stale history on a surface seen from behind, which takes tens of frames to wash
  out) and costs one extra primary ray per pixel per frame. Pinhole cameras only.
* Depth of field is the scene's thin-lens model with the GUI's Aperture (lens diameter) and Focus Distance in scene units. There is no
  denoising or ReSTIR yet. Scenes the Metal backend cannot render (see METAL_PARITY_STATUS.md) fail to start.

## Testing

* `ctest -R metal_realtime_dylib` calls `realtime_renderer.dylib` through the GUI's own function-pointer type (dlopen), renders
  frames and checks that the picture is not black, that two frames with the same camera differ (the seed changes), that moving the
  camera changes the picture, that the output is finite, that every returned world position projects through the returned camera basis
  back onto its own pixel (the relation the GUI's reprojection relies on), and that an aperture override changes the picture.
* `scripts/gui_selftest_macos.sh [App.app]` smoke-tests the real GUI, headless (Qt `offscreen`, a throwaway HOME): the Output Mode
  list must contain an enabled "Live Preview (interactive)", and starting it must produce frames (>20 in 10 s) that change when
  the camera is orbited, that the picture fills the tile, and that switching depth of field on changes it; it also logs how close the
  tile is to the settled picture 8 frames after a camera move (reprojection keeps it close). It saves screenshots of the app's own window. The hook behind it is `RT_GUI_SELFTEST=<ui|livepreview|options>`
  (`qt_gui/mainwindow_selftest.cpp`), a no-op unless that variable is set. Do NOT run the app without the throwaway HOME from a
  tool: macOS asks for permission to read ~/Pictures and an unanswered prompt blocks startup with no visible window.
* `build/metal_live_bench [scene] [w] [h] [spp] [frames]` drives the live API with an orbiting camera, prints frame times and writes PNGs.
