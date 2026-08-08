# Video Generation Feature - Testing Guide

This guide explains how to use the video generation feature to create animated ray-traced videos.

## Overview

The video generation feature allows you to:
- Render multiple frames with animated camera movement
- Choose from 4 camera animation paths (orbit, linear, figure8, spiral)
- Automatically assemble frames into an MP4 video using ffmpeg

## Prerequisites

1. **Build the project** (if not already built):
   ```powershell
   .\build_all.ps1
   ```

2. **ffmpeg** (external dependency, not bundled):
   - Install from https://ffmpeg.org/download.html and make sure `ffmpeg` is on PATH
   - The launcher invokes it automatically as the last step of a video render
     (`libx264` codec, `yuv420p` pixel format) - if it's missing, the render
     still completes and the rendered/converted frames are kept on disk, but
     the launcher exits with `ERR_VIDEO_ASSEMBLY_FAILED` and no video is written

## Quick Start

### Render Video (Automatic Assembly)

Run the ray tracer in video mode - the MP4 is created automatically:

```powershell
# GPU mode (recommended) - 60 frames, orbit path
.\ray_tracer.exe --video --frames 60 --fps 30 --camera-path orbit 600 100 50

# CPU mode - 30 frames, spiral path
.\ray_tracer.exe --cpu --video --frames 30 --fps 30 --camera-path spiral 400 50 20
```

This will:
1. Create frames in `output/frames/` directory
2. Automatically assemble them into an MP4 video
3. Output the final video to `output/<name>_video.mp4`
- `frame_0002.ppm`
- ...
- `frame_0060.ppm`

## Command-Line Options

### Video Rendering (`ray_tracer.exe`)

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Enable video generation mode | (disabled) |
| `--frames`, `-f` | Number of frames to render | 120 |
| `--fps` | Target frames per second | 30 |
| `--speed` | Camera movement speed multiplier | 1.0 |
| `--camera-path`, `-p` | Camera animation path | orbit |
| `--gpu` | Use GPU renderer (OptiX) | ✓ |
| `--cpu` | Use CPU renderer | |
| `--output`, `-o` | Output path (affects frame dir) | `./output/image.ppm` |

**Positional arguments** (same as single-frame mode):
```
ray_tracer.exe [--video options] [width] [spp] [max_depth]
```

Example:
```powershell
# 800x800 resolution, 100 samples per pixel, 50 ray depth
.\ray_tracer.exe --video --frames 90 --camera-path figure8 800 100 50
```

The video file will be created automatically as `output/<name>_video.mp4`.

### Movement Speed

`--speed` does not change the camera path itself - the camera always
completes the exact same full sweep (one full rotation for orbit/figure8,
two for spiral, the whole start→end traversal for linear) no matter the
speed, so it always swings all the way around/through the scene. Instead,
`--speed` scales how many frames that sweep is spread across: `--frames` is
the "1.0x" baseline frame count, and the actual number of rendered frames is
`frames / speed` (capped at 5000). `0.5x` renders twice as many frames,
spreading the same journey over more of them - and more real video time at
the same `--fps` - so it looks slower without ever cutting the path short.
`2.0x` renders half as many frames, covering the same journey faster (and
in a shorter video).

```powershell
# Half-speed orbit - renders 2x the frames (240 instead of 120), same full
# rotation, twice the video length at the same fps
.\ray_tracer.exe --video --camera-path orbit --frames 120 --speed 0.5

# Double-speed spiral - renders half the frames (60 instead of 120), same
# 2 full rotations, half the video length
.\ray_tracer.exe --video --camera-path spiral --frames 120 --speed 2.0
```

## Camera Animation Paths

### 1. Orbit (Default)
Circular motion around the scene on the XZ plane.

```powershell
.\ray_tracer.exe --video --camera-path orbit
```

**Best for:** Full scene overview, 360° rotation

### 2. Linear
Straight-line movement from start to end position.

```powershell
.\ray_tracer.exe --video --camera-path linear
```

**Best for:** Fly-through effects, cinematic reveals

### 3. Figure-8
Lemniscate pattern (figure-8 shape) motion.

```powershell
.\ray_tracer.exe --video --camera-path figure8
```

**Best for:** Dynamic motion, artistic effect

### 4. Spiral
Spiraling inward while rotating around the scene.

```powershell
.\ray_tracer.exe --video --camera-path spiral
```

**Best for:** Zoom-in effect, dramatic intro/outro

## Examples

### Example 1: Quick Preview (Low Quality)
Fast render for testing camera paths:

```powershell
# 30 frames, 200x200 resolution, 10 samples/pixel - video assembles automatically
.\ray_tracer.exe --video --frames 30 --fps 30 --camera-path orbit 200 10 20
```

Estimated time: ~1-2 minutes (GPU), ~5-10 minutes (CPU)

### Example 2: High Quality Production
Full quality video for final output:

```powershell
# 120 frames, 800x800 resolution, 500 samples/pixel
.\ray_tracer.exe --gpu --video --frames 120 --fps 30 --camera-path orbit 800 500 50
```

Estimated time: ~10-30 minutes (GPU), several hours (CPU)

### Example 3: Multiple Camera Paths
Create different videos from the same scene (each run clears and re-populates `output/frames/`, then assembles its own `output/<name>_video.mp4`):

```powershell
# Orbit path
.\ray_tracer.exe --video --frames 60 --camera-path orbit 600 100 50

# Spiral path (re-renders frames; --output picks the video's name/location -
# it's derived from the output stem as <stem>_video.mp4, so without --output
# both runs above would overwrite the same output/image_video.mp4)
.\ray_tracer.exe --video --frames 60 --camera-path spiral --output .\output\spiral.ppm 600 100 50
```

## Performance Tips

### GPU Mode (Recommended)
- **Much faster** for video generation (10-100x speedup)
- Use higher sample counts for quality (100-500 spp)
- Requires CUDA-capable GPU + OptiX 9.1

### CPU Mode
- Good for testing and low-frame-count videos
- Use lower sample counts (10-50 spp) for faster renders
- Multi-threaded (uses all CPU cores)

### Quality vs. Speed Trade-offs

| Setting | Preview | Production |
|---------|---------|------------|
| Frames | 30-60 | 120-240 |
| Resolution | 200-400 | 600-1200 |
| Samples/Pixel | 10-50 | 100-1000 |
| Render Time (GPU) | 1-5 min | 10-60 min |

## Troubleshooting

### "No frame_*.ppm files found"
Check that video rendering completed successfully. Look for `output/frames/` directory.

### Video is too fast/slow
Adjust FPS during rendering:
```powershell
# Render at 60 FPS
.\ray_tracer.exe --video --fps 60 --frames 120
```

### Video file not created
Check the render log for ffmpeg errors (missing from PATH, unsupported codec, etc. - the launcher prints the exact ffmpeg command it ran, which you can also run manually to debug). Also confirm the renderer completed successfully and didn't crash during frame rendering.

### Frames look different between renders
- Video mode uses **animated camera positions**
- Single-frame mode uses **static camera position**
- This is expected behavior

### Out of memory during rendering
Reduce resolution or samples per pixel:
```powershell
# Lower settings
.\ray_tracer.exe --video --frames 60 400 50 30
```

## File Structure

```
output/
├── frames/                  # Cleared and repopulated by each --video run
│   ├── frame_0001.ppm       # Rendered frames, original frame index
│   ├── frame_0002.ppm
│   ├── ...
│   ├── enc_0000.png         # PNG conversions, renumbered contiguously
│   ├── enc_0001.png         # (skips any frame that failed to render)
│   └── ...
├── <stem>_video.mp4         # Final video, assembled by ffmpeg (<stem> from --output)
└── image.ppm                # Single-frame renders (normal mode)
```

## Next Steps

1. **Experiment with camera paths** to find the best view
2. **Adjust render quality** (resolution, samples/pixel) for your needs
3. **Try different FPS values** (24 for film look, 30 for smooth, 60 for very smooth)
4. **Customize camera paths** by editing `launcher/camera_path.h` (advanced)

## See Also

- [BUILD.md](../docs/BUILD.md) - Build instructions
- [README.md](../README.md) - Project overview
- [launcher/camera_path.h](../launcher/camera_path.h) - Camera path implementation
