#ifndef CPU_INTERFACE_H
#define CPU_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Render the Cornell box scene using CPU path tracing with importance sampling.
 * 
 * @param width         Image width in pixels (height = width for square aspect)
 * @param height        Image height in pixels
 * @param spp           Samples per pixel
 * @param max_depth     Maximum ray bounce depth
 * @param output_path   Output PPM file path (e.g., "C:/path/to/image.ppm")
 * @return 0 on success, non-zero on failure
 */
int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path);

#ifdef __cplusplus
}
#endif

#endif // CPU_INTERFACE_H
