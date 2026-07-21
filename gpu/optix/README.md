# OptiX GPU Renderer

This directory contains the OptiX-based GPU renderer for the ray tracer.

## Architecture

**OptiX** is NVIDIA's high-performance ray tracing API that provides:
- Hardware-accelerated ray tracing via RT cores
- Automatic BVH construction and traversal
- Programmable ray generation, intersection, and shading

## File Structure

- `optix_types.h` - Shared data structures between host and device
- `optix_interface.h` - C API matching `gpu/cuda/gpu_interface.h`
- `optix_renderer.h` - Main OptiXRenderer class (host-side)
- `optix_renderer.cpp` - OptiXRenderer implementation
- `optix_programs.cu` - OptiX device programs (raygen, miss, hit, intersection)
- `scene_builder.cpp` - Convert scene objects to OptiX geometry

## Build System

OptiX programs (`.cu` files) are compiled to PTX at build time via `build_optix.targets`. The host code loads PTX modules at runtime and creates the OptiX pipeline.

## OptiX Pipeline

1. **Ray Generation**: Creates primary rays from camera
2. **Intersection**: Tests rays against custom primitives (spheres, quads)
3. **Closest Hit**: Shades the hit point based on material
4. **Miss**: Returns sky/background color
5. **Any Hit**: (Not used - could be used for transparency)

## Materials

Supported materials match the CPU renderer:
- Lambertian (diffuse)
- Metal (reflective with optional fuzz)
- Dielectric (glass, refraction)
- Diffuse Light (emissive)

## Requirements

- OptiX SDK 7.0+ (tested with 9.1.0)
- CUDA 11.0+ (compiled with 13.2)
- NVIDIA GPU with RT cores (Turing/Ampere/Ada/Hopper/Blackwell)
- Windows 10/11 with recent NVIDIA drivers (535.00+)
