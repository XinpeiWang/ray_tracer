CUDA prototype renderer
=======================

This folder contains a minimal CUDA prototype that renders a simple gradient image
on the GPU and writes a PPM file. It's intended as a proof-of-concept to verify the
GPU pipeline on machines with an NVIDIA GPU.

Prerequisites
-------------
- NVIDIA GPU and driver
- CUDA Toolkit (nvcc) installed and available on PATH

Build
-----
Open a Developer PowerShell or regular shell where nvcc is on PATH and run:

	cd gpu/cuda
	nvcc host.cu -O3 -std=c++17 -o cuda_renderer.exe

If compilation fails, install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
and restart your shell.

Run
---
	gpu/cuda/cuda_renderer.exe [width] [height]

Example:
	gpu/cuda/cuda_renderer.exe 800 450

The program writes image_cuda.ppm to your Desktop (OneDrive Desktop preferred when available).

Notes
-----
- This is a minimal demo (no ray-sphere intersections). It verifies that kernels can run
  and data can be copied back to the host. You can extend the kernel to implement ray
  generation, intersections, and shading.
- For a full path tracer on the GPU you'll need to implement device-side RNG, BVH,
  material evaluation, and accumulate samples per pixel across kernel launches.
