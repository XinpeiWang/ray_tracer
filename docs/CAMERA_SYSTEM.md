# Camera System Documentation

## Overview

The ray tracer supports configurable camera positioning for the Cornell box scene. The camera can be positioned anywhere in 3D space (inside or outside the box) while always looking at the center of the scene.

## Cornell Box Scene Geometry

```
Box Dimensions: X[0, 555], Y[0, 555], Z[0, 555]
Center Point: (278, 278, 278)

Walls:
  - Front:   Z=0 (open, no wall - allows viewing from outside)
  - Back:    Z=555 (white)
  - Left:    X=0 (red)
  - Right:   X=555 (green)
  - Floor:   Y=0 (white)
  - Ceiling: Y=555 (white, with light source at center)
```

## Camera Parameters

### lookfrom (Camera Position)
- **Configurable**: Set by user via GUI presets or custom X/Y/Z values
- **Range**: Can be positioned anywhere (-2000 to +2000 in each axis)
- **Default**: (278, 278, -300) - outside the box looking in

### lookat (Camera Target)
- **Fixed**: Always points to (278, 278, 278) - the center of the Cornell box
- **Not user-configurable**: Hardcoded in both CPU and GPU renderers

### vup (View Up Vector)
- **Fixed**: (0, 1, 0) - defines "up" direction
- **Not user-configurable**

### vfov (Vertical Field of View)
- **Fixed**: 40 degrees
- **Not user-configurable**

## GUI Camera Presets

The GUI provides 8 camera presets for common viewpoints:

| Preset | Position (X, Y, Z) | Description | Distance from Center |
|--------|-------------------|-------------|---------------------|
| **Front View (Outside)** | (278, 278, -300) | Classic view from outside the open front | ~578 units |
| **Inside Front** | (278, 278, 50) | Just inside the front opening | ~228 units |
| **Inside Back** | (278, 278, 500) | Near the back wall, looking toward front | ~222 units |
| **Right Wall (Green)** | (500, 278, 278) | Near the green right wall | ~222 units |
| **Left Wall (Red)** | (50, 278, 278) | Near the red left wall | ~228 units |
| **Floor Corner** | (100, 50, 100) | Low angle from front corner | ~307 units |
| **Ceiling Corner** | (450, 500, 450) | High angle from back corner | ~305 units |
| **Custom** | (user input) | Manually set X/Y/Z via spinboxes | variable |

All "inside" presets maintain approximately the same distance from center (~220-230 units) for consistent viewing perspectives.

## Code Flow

### 1. GUI Layer (`qt_gui/mainwindow.cpp` & `mainwindow.h`)

**User Interaction:**
- User selects a preset from `m_cameraPresetCombo` OR
- User selects "Custom" and manually adjusts `m_cameraPosX`, `m_cameraPosY`, `m_cameraPosZ`

**Preset Change Handler:**
```cpp
void MainWindow::onCameraPresetChanged(int index)
```
- Checks if "Custom" (index 7) is selected
- Enables/disables spinboxes accordingly
- Updates spinbox values to show preset coordinates

**Render Trigger:**
```cpp
void MainWindow::onRenderClicked()
```
- Collects camera position from spinboxes: `camX`, `camY`, `camZ`
- Passes to `RenderThread::setParameters(..., camX, camY, camZ, ...)`

### 2. Render Thread (`qt_gui/mainwindow.cpp` - RenderThread class)

**Command Line Construction:**
```cpp
void RenderThread::run()
```
- Builds command: `ray_tracer.exe [--gpu|--cpu] [--output path] width samples depth cam_x cam_y cam_z`
- Launches subprocess with camera coordinates as last 3 arguments

### 3. Launcher (`main.cpp`)

**Argument Parsing:**
```cpp
// Numeric argument order: width, spp, max_depth, cam_x, cam_y, cam_z
std::vector<double> numeric_args;
// Parse using std::stod for floating point support
```

**Renderer Invocation:**
```cpp
if (use_gpu) {
	gpu_render_main(..., cam_x, cam_y, cam_z);
} else {
	cpu_render_main(..., cam_x, cam_y, cam_z);
}
```

### 4. CPU Renderer (`cpu_renderer/cpu_interface.cpp`)

**Camera Setup:**
```cpp
extern "C" int cpu_render_main(..., double cam_x, double cam_y, double cam_z) {
	camera cam;
	cam.lookfrom = point3(cam_x, cam_y, cam_z);  // User-specified position
	cam.lookat   = point3(278, 278, 278);        // Fixed: center of Cornell box
	cam.vup      = vec3(0, 1, 0);                // Fixed: up is +Y
	cam.vfov     = 40;                           // Fixed field of view

	cam.render(world, lights);
}
```

### 5. GPU Renderer (`gpu/cuda/gpu_interface.cu` & `scene_serializer.cpp`)

**Scene Serialization:**
```cpp
// gpu/cuda/scene_serializer.cpp
void serialize_scene_arrays(..., double cam_x, double cam_y, double cam_z) {
	camera cam;
	cam.lookfrom = point3(cam_x, cam_y, cam_z);  // User-specified position
	cam.lookat   = point3(278, 278, 278);        // Fixed: center of Cornell box
	cam.vup      = vec3(0, 1, 0);                // Fixed: up is +Y
	cam.vfov     = 40;                           // Fixed field of view

	// Convert to CameraPOD for GPU kernel
	*out_camera = to_camera_pod(cam);
}
```

## Key Implementation Files

| File | Purpose |
|------|---------|
| `qt_gui/mainwindow.h` | Camera UI widget declarations and RenderThread interface |
| `qt_gui/mainwindow.cpp` | Camera preset setup, spinbox handling, command-line generation |
| `main.cpp` | Command-line parsing and renderer dispatch |
| `cpu_renderer/cpu_interface.h` | CPU renderer C API with camera parameters |
| `cpu_renderer/cpu_interface.cpp` | CPU camera setup and render invocation |
| `gpu/cuda/gpu_interface.h` | GPU renderer C API with camera parameters |
| `gpu/cuda/gpu_interface.cu` | GPU render coordination |
| `gpu/cuda/scene_serializer.h` | GPU scene serialization interface |
| `gpu/cuda/scene_serializer.cpp` | GPU camera setup and POD conversion |

## Adding New Camera Presets

To add a new preset to the GUI:

1. **Edit `qt_gui/mainwindow.cpp`** in the `createAdvancedTab()` function:
```cpp
// Add before the "Custom" entry
m_cameraPresetCombo->addItem("My New View", QVariant::fromValue(QVector3D(x, y, z)));
```

2. **Update the Custom index** in `onCameraPresetChanged()`:
```cpp
// If you now have 9 total items, Custom is at index 8
bool isCustom = (index == 8); // Update from 7 to 8
```

3. **Rebuild** the Qt GUI:
```bash
cd qt_gui
mingw32-make
```

## Testing Camera Positions

### Via GUI:
1. Launch `RayTracerGUI.exe`
2. Go to Advanced Settings tab
3. Select a camera preset or choose "Custom"
4. Click "Render"
5. View output image (timestamped to avoid overwrites)

### Via Command Line:
```bash
# Test a specific camera position
ray_tracer.exe --gpu 800 100 50 278 500 200
#               mode width spp depth x  y   z

# The command format is:
# ray_tracer.exe [--gpu|--cpu] [--output path] width samples depth cam_x cam_y cam_z
```

## Troubleshooting

### Camera appears not to move
- **Check timestamped output**: GUI uses timestamped filenames by default to prevent overwriting
- **Verify coordinates**: Open Advanced Settings and check the spinbox values
- **Check renderer logs**: Look for `camera=(x,y,z)` in the log output

### View looks wrong from inside the box
- **Check distance from center**: Inside views work best ~200-250 units from center
- **Avoid extreme positions**: Camera too close to walls may show mostly one color
- **Check for obstructions**: Camera inside objects will create unusual views

### Custom preset not enabling spinboxes
- **Verify index**: Ensure `onCameraPresetChanged()` uses correct index for "Custom"
- **Check signal connection**: Verify `connect()` call is present and after widget creation

## Future Enhancements

Possible camera system improvements:

1. **Configurable lookat**: Allow user to set target point instead of fixed center
2. **Animation paths**: Define keyframes and render camera path interpolation
3. **Camera rotation**: Allow roll/pitch/yaw controls in addition to position
4. **FOV adjustment**: Make vfov user-configurable for zoom effects
5. **Orbit controls**: Maintain distance from center while allowing rotation
6. **Save/load presets**: Allow users to save custom positions as named presets
