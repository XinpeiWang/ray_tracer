# Error Code Reference Guide

**Complete debugging reference for the ray tracer error code system**

## Overview

The ray tracer uses a comprehensive error code system to make debugging easier. Every error has:
- **Numeric code** - Integer identifying the specific error
- **Category** - General, CPU Renderer, or GPU Renderer
- **Message** - Human-readable description
- **Troubleshooting hint** - Actionable steps to resolve the issue

## Error Code Ranges

| Range | Category | Description |
|-------|----------|-------------|
| 0 | Success | Render completed successfully |
| 1-99 | General | File I/O, parameters, validation |
| 100-199 | CPU Renderer | CPU-specific rendering errors |
| 200-299 | GPU Renderer | CUDA/GPU-specific errors |
| 999 | User Action | User-initiated cancellation |

---

## Success Code

### 0 - SUCCESS
✅ **Message:** Success  
**Meaning:** Render completed successfully  
**Action:** None needed

---

## General Errors (1-99)

### 1 - ERR_UNKNOWN
**Message:** Unknown error occurred  
**Meaning:** An unidentified error happened  
**Troubleshooting:**
- Check the Log Output tab for detailed error information
- Look for stack traces or exception messages
- Report the issue if it persists

### 2 - ERR_INVALID_ARGUMENTS
**Message:** Invalid command-line arguments  
**Meaning:** The renderer was called with incorrect parameters  
**Troubleshooting:**
- Check command syntax: `ray_tracer.exe [--cpu|--gpu] [--output path] <width> <spp> <depth> <scene_id> <cam_x> <cam_y> <cam_z>`
- Ensure all numeric arguments are positive integers
- For GUI users: this usually indicates a GUI bug, please report it

### 3 - ERR_FILE_NOT_FOUND
**Message:** Required file not found  
**Meaning:** A texture or resource file is missing  
**Troubleshooting:**
- For Earth scene (scene 3), ensure `earthmap.jpg` is in the correct location
- Check that all texture files are present and readable
- Verify file paths are correct

### 5 - ERR_FILE_WRITE_FAILED
**Message:** Failed to write output file  
**Meaning:** Cannot write the rendered image to disk  
**Troubleshooting:**
- Check output directory permissions and disk space
- Ensure no other program is using the output file
- Try a different output path
- Verify disk is not full

### 6 - ERR_FILE_COPY_FAILED
**Message:** Failed to copy output file  
**Meaning:** Render succeeded but couldn't copy to requested location  
**Troubleshooting:**
- Check destination folder permissions
- Ensure enough disk space
- Original file may still exist at default location (OneDrive/Desktop)

### 7 - ERR_DIRECTORY_CREATE_FAILED
**Message:** Failed to create output directory  
**Meaning:** Cannot create the directory for output files  
**Troubleshooting:**
- Check parent directory permissions
- Verify path is valid
- Ensure no file exists with the same name as the directory

### 8 - ERR_INVALID_DIMENSIONS
**Message:** Invalid image dimensions (must be > 0)  
**Meaning:** Width or height is zero or negative  
**Troubleshooting:**
- Width and height must be positive integers
- Recommended: 400-1920 pixels
- Common resolutions: 800×800, 1920×1080, 3840×2160

### 9 - ERR_INVALID_SAMPLE_COUNT
**Message:** Invalid sample count (must be > 0)  
**Meaning:** Samples per pixel is zero or negative  
**Troubleshooting:**
- For quick previews: 10-50 samples
- For final renders: 100-500 samples
- More samples = better quality but slower render

### 10 - ERR_INVALID_MAX_DEPTH
**Message:** Invalid max depth (must be > 0)  
**Meaning:** Maximum ray bounce depth is zero or negative  
**Troubleshooting:**
- Recommended: 10-100 bounces
- Lower values = faster but less realistic lighting
- Higher values = more realistic reflections/refractions

### 11 - ERR_INVALID_SCENE_ID
**Message:** Invalid scene ID (must be 0-8)  
**Meaning:** Scene ID is out of valid range  
**Troubleshooting:**
- Valid scene IDs: 0 (Cornell Box) through 8 (Final Scene)
- Check `docs/SCENE_SELECTION.md` for scene list
- Use the GUI scene selector to choose valid scenes

### 12 - ERR_INVALID_CAMERA_POSITION
**Message:** Invalid camera position coordinates  
**Meaning:** Camera X/Y/Z values are invalid (NaN or infinity)  
**Troubleshooting:**
- Camera coordinates should be finite numbers
- Default Cornell Box: (278, 278, -800)
- Check that camera inputs are valid numbers

### 13 - ERR_OUTPUT_PATH_INVALID
**Message:** Output path is invalid or not writable  
**Meaning:** The specified output path cannot be used  
**Troubleshooting:**
- Check that the path exists and is writable
- Avoid special characters in path
- Use absolute paths for reliability

---

## CPU Renderer Errors (100-199)

### 100 - ERR_CPU_SCENE_BUILD_FAILED
**Message:** CPU: Failed to build scene  
**Meaning:** Scene construction threw an exception  
**Troubleshooting:**
- Check scene ID and texture file availability
- For Earth scene, ensure earthmap.jpg exists
- Check Log Output for stack trace

### 101 - ERR_CPU_SCENE_EMPTY
**Message:** CPU: Scene contains no objects  
**Meaning:** Scene builder returned zero geometry  
**Troubleshooting:**
- This indicates a bug in the scene builder
- Check that the scene ID is valid (0-8)
- Report this error if it persists

### 102 - ERR_CPU_CAMERA_INIT_FAILED
**Message:** CPU: Failed to initialize camera  
**Meaning:** Camera configuration is invalid  
**Troubleshooting:**
- Check camera position values
- Ensure lookfrom and lookat are different points
- Verify field of view (vfov) is reasonable (10-120 degrees)

### 103 - ERR_CPU_RENDER_FAILED
**Message:** CPU: Rendering failed during execution  
**Meaning:** An error occurred during the actual rendering loop  
**Troubleshooting:**
- Check Log Output for exception details
- May indicate a bug in material or geometry code
- Try reducing scene complexity

### 104 - ERR_CPU_THREAD_FAILED
**Message:** CPU: Worker thread encountered an error  
**Meaning:** One of the rendering threads crashed  
**Troubleshooting:**
- Check for race conditions or thread-safety issues
- Try reducing thread count (set RAY_TRACER_THREADS environment variable)
- Report this error with scene details

### 105 - ERR_CPU_MEMORY_ALLOCATION
**Message:** CPU: Out of memory  
**Meaning:** System ran out of RAM during rendering  
**Troubleshooting:**
- **Reduce resolution:** Try 800×800 instead of 1920×1080
- **Reduce samples:** Try 50 instead of 500
- **Close other apps:** Free up system memory
- **Scene complexity:** Some scenes (Final Scene) are very memory-intensive

### 106 - ERR_CPU_BVH_BUILD_FAILED
**Message:** CPU: BVH acceleration structure build failed  
**Meaning:** Failed to build the spatial acceleration structure  
**Troubleshooting:**
- This may indicate invalid geometry
- Try disabling BVH for debugging
- Report this error with scene details

### 107 - ERR_CPU_TEXTURE_LOAD_FAILED
**Message:** CPU: Failed to load texture file  
**Meaning:** Cannot read a required texture image  
**Troubleshooting:**
- **Earth scene:** Ensure earthmap.jpg is in the correct folder
- Check file permissions and format
- Verify file is not corrupted
- JPEG and PNG formats are supported

### 108 - ERR_CPU_LIGHTS_EMPTY
**Message:** CPU: Scene has no lights for importance sampling  
**Meaning:** PDF sampling requires at least one light source  
**Troubleshooting:**
- This should not occur in current version (dummy lights added automatically)
- If you see this, report it as a bug

### 109 - ERR_CPU_MATERIAL_INVALID
**Message:** CPU: Invalid material configuration  
**Meaning:** Material parameters are invalid (NaN, negative values, etc.)  
**Troubleshooting:**
- Check scene definition for invalid material properties
- Refractive index should be > 0
- Colors should be in [0,1] range

---

## GPU Renderer Errors (200-299)

### 200 - ERR_GPU_NO_DEVICE
**Message:** GPU: No CUDA-capable device found  
**Meaning:** System has no NVIDIA GPU or CUDA is not installed  
**Troubleshooting:**
- **Switch to CPU mode** - CPU works on all systems
- Check that GPU is NVIDIA (AMD/Intel GPUs don't support CUDA)
- Verify CUDA drivers are installed
- Update GPU drivers to latest version

### 201 - ERR_GPU_DEVICE_INIT_FAILED
**Message:** GPU: Failed to initialize CUDA device  
**Meaning:** CUDA initialization or device setup failed  
**Troubleshooting:**
- Update NVIDIA drivers
- Check CUDA installation
- Verify GPU is functioning correctly
- Try rebooting the system

### 202 - ERR_GPU_MEMORY_ALLOCATION
**Message:** GPU: Failed to allocate device memory  
**Meaning:** GPU ran out of VRAM  
**Troubleshooting:**
- **Reduce resolution:** Try 800×800 instead of 1920×1080
- **Reduce samples:** Try 50 instead of 500
- **Close other apps:** Free up GPU memory (games, browsers, etc.)
- **Switch to CPU mode** for large renders

### 203 - ERR_GPU_MEMORY_COPY_FAILED
**Message:** GPU: Failed to copy data to/from device  
**Meaning:** Communication with GPU failed  
**Troubleshooting:**
- GPU may be unstable or overheating
- Try reducing workload size
- Check GPU health and cooling
- Update GPU drivers

### 204 - ERR_GPU_KERNEL_LAUNCH_FAILED
**Message:** GPU: Failed to launch kernel  
**Meaning:** Could not start GPU rendering kernel  
**Troubleshooting:**
- Kernel may exceed GPU resource limits
- Try smaller resolution
- Update GPU drivers
- Switch to CPU mode

### 205 - ERR_GPU_KERNEL_EXECUTION_FAILED
**Message:** GPU: Kernel execution failed  
**Meaning:** GPU kernel crashed during execution  
**Troubleshooting:**
- May indicate GPU instability
- Check GPU temperature and cooling
- Try reducing sample count or resolution
- Update GPU drivers

### 206 - ERR_GPU_SCENE_SERIALIZATION_FAILED
**Message:** GPU: Failed to serialize scene data  
**Meaning:** Cannot convert scene to GPU-compatible format  
**Troubleshooting:**
- This indicates a bug in scene serialization
- Try a different scene
- Use CPU mode for this scene
- Report the error with scene ID

### 207 - ERR_GPU_DEVICE_SYNCHRONIZATION_FAILED
**Message:** GPU: Device synchronization failed  
**Meaning:** Could not wait for GPU to finish work  
**Troubleshooting:**
- GPU may have hung or crashed
- Try rebooting
- Check GPU health
- Update drivers

### 208 - ERR_GPU_OUT_OF_MEMORY
**Message:** GPU: Out of device memory  
**Meaning:** GPU VRAM exhausted during rendering  
**Troubleshooting:**
- **Reduce resolution:** 800×800 uses ~2MB VRAM per frame
- **Reduce samples:** Fewer samples = less memory
- **Close GPU apps:** Free VRAM by closing games/browsers
- **Use CPU mode** for memory-intensive renders

### 209 - ERR_GPU_INVALID_CONFIGURATION
**Message:** GPU: Invalid kernel configuration  
**Meaning:** Block size or grid dimensions are invalid  
**Troubleshooting:**
- This should auto-fallback to smaller block sizes
- If you see this, report it as a bug
- Try switching to CPU mode

### 210 - ERR_GPU_TEXTURE_BINDING_FAILED
**Message:** GPU: Failed to bind texture  
**Meaning:** Cannot load texture into GPU texture memory  
**Troubleshooting:**
- Texture may be too large for GPU
- Try a smaller texture
- Use CPU mode for high-res textures

### 211 - ERR_GPU_UNSUPPORTED_SCENE
**Message:** GPU: Scene not supported on GPU (use CPU mode)  
**Meaning:** GPU renderer only supports Cornell Box (scene 0)  
**Troubleshooting:**
- **Switch to CPU mode** to render this scene
- CPU supports all 9 scenes
- GPU support for other scenes is planned for future releases

---

## User Action (999)

### 999 - ERR_USER_CANCELLED
**Message:** Render cancelled by user  
**Meaning:** User clicked Stop or closed the application  
**Action:** None needed - this is intentional

---

## How to Use This Guide

### From Command Line
When a render fails, the error code and message are printed to console:
```
[CPU ERROR 105] CPU: Out of memory
  Hint: Try reducing resolution or sample count
```

### From GUI
The GUI shows:
1. **Log Output tab** - Detailed error with code, category, and hint
2. **Error dialog** - User-friendly message with troubleshooting steps
3. **Status message** - Brief error summary in status bar

### Exit Codes
The ray_tracer.exe process returns the error code as its exit code:
```bash
ray_tracer.exe --cpu 800 100 50 1 13 2 3
echo Exit code: %ERRORLEVEL%  # Windows
echo Exit code: $?             # Linux/Mac
```

Exit code 0 = success, any other value = specific error code.

---

## Quick Troubleshooting by Category

### Out of Memory Errors (105, 202, 208)
1. Lower resolution (800×800 → 400×400)
2. Reduce samples (500 → 100 → 50)
3. Close other applications
4. Choose simpler scenes
5. Switch CPU ↔ GPU mode

### File Errors (3, 5, 6, 7, 13)
1. Check file/folder permissions
2. Verify disk space
3. Ensure paths are valid
4. Check for special characters in paths
5. Try different output location

### GPU Errors (200-211)
1. Update NVIDIA drivers
2. Switch to CPU mode
3. Reduce resolution/samples
4. Check GPU health (temperature, stability)
5. Use CPU for unsupported scenes

### Parameter Errors (8, 9, 10, 11, 12)
1. Use GUI scene selector
2. Check numeric values are positive
3. Use recommended ranges (see error messages)
4. Verify scene ID is 0-8

---

## Adding New Error Codes

If you're extending the renderer, follow these steps:

1. **Add to error_codes.h**
   ```cpp
   ERR_YOUR_NEW_ERROR = 110,  // Choose next available code
   ```

2. **Add message**
   ```cpp
   {ERR_YOUR_NEW_ERROR, "Description of your error"},
   ```

3. **Add troubleshooting hint**
   ```cpp
   {ERR_YOUR_NEW_ERROR, "Steps to fix this error"},
   ```

4. **Update qt_gui/error_handler.h** with Qt-friendly messages

5. **Update this documentation** with the new error

6. **Return the error code** from your function:
   ```cpp
   return ERR_YOUR_NEW_ERROR;
   ```

---

## Error Code Best Practices

### For Developers
- Always return specific error codes, never generic -1 or 1
- Add context logging before returning error
- Include error details in exception messages
- Test error paths to ensure codes are returned correctly

### For Users
- Check the Log Output tab for detailed diagnostics
- Note the error code when reporting issues
- Try troubleshooting hints before reporting bugs
- Include error code in bug reports

---

## See Also

- `src/TheRestOfYourLife/error_codes.h` - C++ error code definitions
- `qt_gui/error_handler.h` - Qt GUI error mappings
- `docs/SCENE_SELECTION.md` - Scene IDs and descriptions
- `LOG_TAB_ADDED.md` - How to read GUI logs

---

**Last Updated:** Error code system v1.0  
**Total Error Codes:** 40+ distinct error codes  
**Error Coverage:** General (13), CPU (10), GPU (12)
