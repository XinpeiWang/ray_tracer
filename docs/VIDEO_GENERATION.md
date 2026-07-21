# Video Generation Feature - Testing Guide

This guide explains how to use the video generation feature to create animated ray-traced videos.

## Overview

The video generation feature allows you to:
- Render multiple frames with animated camera movement
- Choose from 4 camera animation paths (orbit, linear, figure8, spiral)
- Automatically assemble frames into an MP4 video using OpenCV (no external tools required)

## Prerequisites

1. **Build the project** (if not already built):
   ```powershell
   .\scripts\build_all.ps1
   ```

2. **OpenCV** (built-in via vcpkg):
   - OpenCV 4 is automatically available through vcpkg integration
   - No manual installation or PATH configuration needed
   - Video encoding uses the built-in MP4V codec

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
# 30 frames, 200x200 resolution, 10 samples/pixel
.\ray_tracer.exe --video --frames 30 --fps 30 --camera-path orbit 200 10 20
.\scripts\assemble_video.ps1 -FramesDir ".\output\frames" -FPS 30
```

Estimated time: ~1-2 minutes (GPU), ~5-10 minutes (CPU)

### Example 2: High Quality Production
Full quality video for final output:

```powershell
# 120 frames, 800x800 resolution, 500 samples/pixel
.\ray_tracer.exe --gpu --video --frames 120 --fps 30 --camera-path orbit 800 500 50
.\scripts\assemble_video.ps1 -FramesDir ".\output\frames" -FPS 30 -Quality 18
```

Estimated time: ~10-30 minutes (GPU), several hours (CPU)

### Example 3: Multiple Camera Paths
Create different videos from the same scene:

```powershell
# Orbit path
.\ray_tracer.exe --video --frames 60 --camera-path orbit 600 100 50
.\scripts\assemble_video.ps1 -FramesDir ".\output\frames" -OutputPath ".\output\orbit.mp4" -FPS 30

# Spiral path (re-renders frames)
.\ray_tracer.exe --video --frames 60 --camera-path spiral 600 100 50
.\scripts\assemble_video.ps1 -FramesDir ".\output\frames" -OutputPath ".\output\spiral.mp4" -FPS 30
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
Check the render log for OpenCV errors. Ensure the renderer completed successfully and didn't crash during frame rendering.

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
├── frames/              # Rendered frames (created by ray_tracer.exe --video)
│   ├── frame_0001.ppm
│   ├── frame_0002.ppm
│   └── ...
├── video.mp4            # Final video (created by assemble_video.ps1)
└── image.ppm            # Single-frame renders (normal mode)
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
