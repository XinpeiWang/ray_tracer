# Render passes (AOVs)

A render pass shows one property of the scene instead of the finished lit picture. Compositing programs (Blender, Nuke, After Effects, Fusion) use them to adjust a render after the fact: relight with the normals, add fog or depth of field from the depth, recolour a surface from the albedo.

`--aovs` (or **Also write render passes** in the Render Options tab) writes these channels next to the image:

| Channel | What it holds |
|---|---|
| `albedo.R/G/B` | the surface's own colour, without lighting: a matte surface gives its colour, glass and mirrors their tint; 0 on a pure light and where nothing is hit |
| `normal.X/Y/Z` | the surface direction in world space, turned to face the camera, unit length; 0 where nothing is hit |
| `depth.Z` | the distance from the camera to the surface along the ray (not along the view axis); 0 where nothing is hit |
| `uv.U/V` | the surface's texture coordinates |
| `A` | coverage: 1 where the camera sees a surface, 0 where it sees only the background, in between on silhouettes |

Where a pixel is part surface and part background, the surface channels are averaged over the surface samples only, so an object's edge keeps its own value instead of fading towards 0 (use `A` to blend).

**Where they go.** If the image is an `.exr` (`ray_tracer ... --output shot.exr --aovs`), the passes are merged into it: one multilayer EXR with `R G B` (the render) and the passes above. For any other image type they are written to `<name>.aovs.exr` beside it. Single images only (not video).

**How they are made.** The beauty image comes from whichever renderer you chose (CPU, OptiX or Metal). The passes come from a separate first-hit pass over the camera's rays on the CPU scene, with up to 16 samples per pixel for anti-aliasing, so they cost very little, work the same for every renderer, and never change the picture. `src/TheRestOfYourLife/aov_pass.h` is the pass; `cpu_render_main_aovs()` writes it; `cpu_merge_exr_passes()` merges it into an image's EXR; `src/shared/exr_writer.h` has the multi-channel EXR reader and writer (channels are written sorted by name, as the format requires).

**Not (yet) there:** object / material ids, direct-versus-indirect light splits, a shadow catcher, light groups. The Metal renderer writes no passes of its own, and the denoiser on a Mac sees colour only (the passes could guide it; see DENOISING.md).
