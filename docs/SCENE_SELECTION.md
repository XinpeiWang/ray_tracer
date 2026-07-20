# Scene Selection System

## Overview

The ray tracer now supports multiple pre-built scenes from the "Ray Tracing in One Weekend" book series. Users can select different scenes through the Qt GUI, and the appropriate scene will be rendered by the CPU or GPU renderer.

## Available Scenes

The scene library is defined in `src/TheRestOfYourLife/scenes.h` and includes:

| ID | Scene Name | Description | Performance | GPU Support | Notes |
|----|------------|-------------|-------------|-------------|-------|
| 0 | Cornell Box | Classic Cornell box with glass sphere and white box | Medium | ✅ Yes | Default scene |
| 1 | Bouncing Spheres | Random spheres with checker ground (final scene from "In One Weekend") | Slow | ❌ CPU only | Uses moving spheres |
| 2 | Checkered Spheres | Two spheres with procedural checker texture | Fast | ❌ CPU only | Simple test scene |
| 3 | Earth | Globe with earth texture mapping | Fast | ❌ CPU only | **Requires earthmap.jpg** |
| 4 | Perlin Spheres | Spheres with Perlin noise marble texture | Fast | ❌ CPU only | Procedural textures |
| 5 | Colored Quads | Five colored quad primitives | Fast | ❌ CPU only | Simple geometry |
| 6 | Simple Light | Perlin spheres with emissive light sources | Fast | ❌ CPU only | Emissive materials |
| 7 | Cornell Smoke | Cornell box with volumetric fog | Slow | ❌ CPU only | Requires 200+ spp |
| 8 | Final Scene | Complex scene from "The Next Week" | Very Slow | ❌ CPU only | Requires 500+ spp |

## GUI Usage

### Scene Selector Location
The scene selector is located in the **Advanced Settings** tab of the GUI, in the "Scene Selection" group box.

### Components
1. **Scene Dropdown** - Select from the available scenes
2. **Scene Info Label** - Shows scene description, performance characteristics, recommended samples per pixel (SPP), and GPU compatibility

### Features
- **Automatic SPP Adjustment**: When you change scenes, the GUI automatically updates the samples per pixel to the recommended value for that scene
- **GPU Warning**: Scenes that don't support GPU rendering will show a warning in the scene info panel
- **File Requirements**: Scenes that require external files (like Earth) will display a warning

## Technical Implementation

### Architecture Flow

```
GUI (Qt)
  └─> RenderThread::setParameters(..., scene_id, ...)
	  └─> ray_tracer.exe [--gpu|--cpu] width spp depth scene_id cam_x cam_y cam_z
		  ├─> GPU Path: gpu_render_main(..., scene_id, ...) [Cornell Box only]
		  └─> CPU Path: cpu_render_main(..., scene_id, ...)
			  └─> Scene dispatch in cpu_interface.cpp
				  ├─> build_cornell_box()
				  ├─> build_bouncing_spheres()
				  ├─> build_checkered_spheres()
				  ├─> build_earth()
				  ├─> build_perlin_spheres()
				  ├─> build_quads()
				  ├─> build_simple_light()
				  ├─> build_cornell_smoke()
				  └─> build_final_scene()
```

### Key Files Modified

#### GUI Layer
- `qt_gui/mainwindow.h` - Added scene selector controls and scene change handler
- `qt_gui/mainwindow.cpp` - Added scene dropdown, scene info label, and `onSceneChanged()` handler

#### Launcher
- `main.cpp` - Updated argument parsing to include scene_id parameter

#### Renderer Interfaces
- `cpu_renderer/cpu_interface.h` - Added scene_id parameter to `cpu_render_main()`
- `cpu_renderer/cpu_interface.cpp` - Added scene dispatch switch statement
- `gpu/cuda/gpu_interface.h` - Added scene_id parameter to `gpu_render_main()`
- `gpu/cuda/gpu_interface.cu` - Added scene_id parameter with Cornell Box fallback

#### Scene Library
- `src/TheRestOfYourLife/scenes.h` - **New file** with centralized scene builders

## Command-Line Usage

The scene ID can be passed via command line:

```bash
# Cornell Box (default)
ray_tracer.exe --cpu 800 100 50 0 278 278 -800

# Bouncing Spheres
ray_tracer.exe --cpu 800 100 50 1 13 2 3

# Final Scene (very slow!)
ray_tracer.exe --cpu 800 500 50 8 478 278 -600
```

### Command Format
```
ray_tracer.exe [--cpu|--gpu] [--output PATH] width spp max_depth scene_id cam_x cam_y cam_z
```

## Scene-Specific Notes

### Earth Scene (ID 3)
- **Requires** `earthmap.jpg` in the same directory as the executable
- You can download earth texture maps from https://www.solarsystemscope.com/textures/
- Place the image file next to `ray_tracer.exe`

### Bouncing Spheres (ID 1)
- This is the final scene from "Ray Tracing in One Weekend"
- Contains ~400 small spheres + 3 large spheres
- Uses moving spheres (motion blur)
- BVH acceleration is applied automatically
- Recommended camera: `(13, 2, 3)` looking at `(0, 0, 0)`

### Cornell Smoke (ID 7)
- Uses constant medium volumes for volumetric effects
- Requires at least 200 samples per pixel for clean results
- Computationally expensive due to volume scattering

### Final Scene (ID 8)
- The most complex scene from "The Next Week"
- Contains 1000+ objects including boxes, spheres, and volumes
- Requires 500+ samples per pixel
- Render time can be several minutes to hours depending on resolution
- BVH acceleration is critical for acceptable performance

## GPU Limitations

Currently, only the Cornell Box scene (ID 0) is supported on the GPU renderer because:
1. The GPU path uses a simplified scene serialization system
2. Some scene features (like volumes, moving spheres, image textures) are not yet implemented on GPU
3. The GPU renderer uses naive path tracing instead of importance sampling

**Fallback Behavior**: If you select a non-Cornell Box scene with GPU mode enabled, the GPU renderer will automatically fall back to rendering the Cornell Box and display a warning message.

**Recommendation**: Use CPU renderer for all scenes except Cornell Box.

## Camera Presets

The camera presets in the GUI are designed for the Cornell Box. When switching to other scenes, you may want to use custom camera positions:

| Scene | Recommended Camera Position | Recommended Lookat |
|-------|----------------------------|-------------------|
| Bouncing Spheres | (13, 2, 3) | (0, 0, 0) |
| Checkered Spheres | (13, 2, 3) | (0, 0, 0) |
| Earth | (0, 0, 12) | (0, 0, 0) |
| Perlin Spheres | (13, 2, 3) | (0, 2, 0) |
| Quads | (0, 0, 9) | (0, 0, 0) |
| Simple Light | (26, 3, 6) | (0, 2, 0) |
| Final Scene | (478, 278, -600) | (278, 278, 0) |

**Note**: Currently, the `lookat` target is fixed at `(278, 278, 278)` (Cornell Box center) in the renderer code. This is acceptable for Cornell Box and Cornell Smoke but may not be ideal for other scenes. Future improvements could make the lookat point scene-dependent.

## Future Enhancements

### Short-term
- [ ] Make camera `lookat` point configurable per scene
- [ ] Add scene-specific camera presets in the GUI
- [ ] Support image textures on GPU (for Earth scene)

### Long-term
- [ ] Full GPU support for all scene primitives (volumes, moving spheres, etc.)
- [ ] User-created scene file format
- [ ] Scene preview thumbnails in the GUI
- [ ] Real-time viewport preview (low-quality interactive view)

## Troubleshooting

### "Scene requires external files" warning
- For Earth scene: Download and place `earthmap.jpg` next to `ray_tracer.exe`

### GPU shows Cornell Box for other scenes
- This is expected behavior; use CPU renderer for non-Cornell Box scenes

### Scene renders too slowly
- Check the recommended SPP in the scene info panel
- Reduce samples per pixel for faster preview renders
- Reduce resolution (e.g., 400x400 instead of 800x800)

### Scene looks wrong or has artifacts
- Increase samples per pixel (especially for Cornell Smoke and Final Scene)
- Check that you're using recommended camera positions
- Ensure all required external files are present

## Code Example: Adding a New Scene

To add a new scene to the library:

1. **Define the scene builder in `scenes.h`**:
```cpp
inline hittable_list build_my_new_scene() {
	hittable_list world;

	// Add geometry here
	auto ground = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground));

	return world;
}
```

2. **Add to the SceneType enum**:
```cpp
enum class SceneType {
	// ... existing scenes ...
	MY_NEW_SCENE = 9
};
```

3. **Add configuration entry**:
```cpp
static const SceneConfig SCENE_CONFIGS[] = {
	// ... existing configs ...
	{ SceneType::MY_NEW_SCENE, "My New Scene", 
	  "Description of my scene", 
	  false, false, 100, "Fast" }
};
```

4. **Update the GUI dropdown** in `qt_gui/mainwindow.cpp`:
```cpp
m_sceneCombo->addItem("My New Scene", 9);
```

5. **Add to the CPU renderer dispatch** in `cpu_renderer/cpu_interface.cpp`:
```cpp
case 9:  // My New Scene
	world = build_my_new_scene();
	break;
```

6. **Update scene info arrays** in `mainwindow.cpp`:
```cpp
static const SceneInfo sceneInfos[] = {
	// ... existing entries ...
	{"Description of my scene", "Fast", 100, false, false}
};
```

That's it! The scene will now be available in the GUI dropdown.
