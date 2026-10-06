// metal_live_preview.h - Live Preview on the Metal backend.
//
// The GUI's Live Preview calls a renderer library once per frame (a few samples per pixel, an orbiting camera) and
// accumulates the results itself. On Windows that library is OptiX; this is the Metal one. The expensive work - loading the
// pbrt scene, building the acceleration structures, compiling the shader pipeline - happens ONCE, on the first frame of a
// scene/size; every later frame only updates the camera/seed/sample-count uniforms and dispatches, which is what makes
// interactive frame rates possible.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Renders one frame of `scene_id` at width x height with `spp` samples per pixel and writes the LINEAR (not tone-mapped)
// mean radiance to out_rgb (width*height*3 floats, row-major, top row first). The camera is at (cam_x, cam_y, cam_z) in the
// scene's own units, looking at (lookat_x, lookat_y, lookat_z) if has_custom_lookat, else at the scene's own look-at point.
// A different scene or size rebuilds the session. out_world_pos (width*height*4: x, y, z in the scene's own units, w = 1 where the
// pixel's first ray hit something, 0 where it missed) and out_camera_basis (12 floats: origin, lower-left corner, horizontal, vertical
// of the view plane at distance 1, in the same units) feed the caller's temporal reprojection; pass nullptr to skip them.
// aperture (lens DIAMETER) and focus_distance, both in the scene's own units, override the thin-lens depth of field; aperture < 0 keeps
// the scene's own camera. Returns false on failure; see metal_live_last_error().
// `frame_seed` should change every call (the caller's frame counter) so successive frames get different noise.
bool metal_live_render_frame(const char* scene_id, int width, int height, int spp, int max_depth,
                             double cam_x, double cam_y, double cam_z,
                             bool has_custom_lookat, double lookat_x, double lookat_y, double lookat_z,
                             float max_component_value, unsigned int frame_seed,
                             float* out_rgb, float* out_world_pos, float* out_camera_basis,
                             double aperture, double focus_distance);

// Detail for the most recent failed metal_live_render_frame() on this thread.
const char* metal_live_last_error();

// Drops the session and its GPU resources (the next frame rebuilds it).
void metal_live_shutdown();

#ifdef __cplusplus
}
#endif
