#pragma once
// ---------------------------------------------------------------------------
// cpu_gpu.h -- shared CPU_GPU / CPU_GPU_RESTRICT macro definitions
//
// CPU_GPU marks a function so it compiles for both host (CPU) and CUDA
// device (GPU) code: __host__ __device__ __forceinline__ under NVCC,
// plain inline otherwise. CPU_GPU_RESTRICT is the matching __restrict__
// spelling for each toolchain.
//
// Every src/shared header used to define these macros locally, which had
// drifted out of sync: most headers used `inline` on the non-CUDA path,
// but a subset expanded CPU_GPU to nothing at all there, silently turning
// header-defined free functions into non-inline definitions -- an ODR
// violation (LNK2005) if the header is ever included from more than one
// translation unit. Centralizing here keeps all headers on one definition.
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU inline
#  endif
#endif

#ifndef CPU_GPU_RESTRICT
#  if defined(__CUDACC__)
#    define CPU_GPU_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define CPU_GPU_RESTRICT __restrict
#  else
#    define CPU_GPU_RESTRICT __restrict__
#  endif
#endif

// CPU_GPU_CONST marks namespace-scope constexpr DATA (not functions) that
// a CPU_GPU-tagged function reads - e.g. a lookup table. Unlike functions,
// __host__ __device__ isn't legal on a single constexpr array definition,
// so under NVCC this is device-only (`static __device__ constexpr` - the
// leading `static` keeps it internal-linkage, so including this header
// from more than one nvcc-compiled translation unit can't collide at
// device-link time); on a plain host compile it's the original
// `static constexpr`, unchanged. A function tagged CPU_GPU that reads
// CPU_GPU_CONST data therefore still compiles correctly for both host and
// device - the two never need to see the "other side"'s definition,
// because __CUDACC__ is only ever defined for the device compile pass.
#ifndef CPU_GPU_CONST
#  if defined(__CUDACC__)
#    define CPU_GPU_CONST static __device__ constexpr
#  else
#    define CPU_GPU_CONST static constexpr
#  endif
#endif

// Shared safety cap for "advance through a medium-boundary/null-BSDF hit
// (interface_material / MaterialType::Interface - a real pbrt-v4 "no BSDF"
// pass-through) without consuming the path's depth/bounce budget". Every
// integrator that supports this (CPU default path tracer, BDPT, SPPM, the
// debug light/simple/simple-vol path integrators, and all 3 GPU backends)
// needs its OWN loop-local crossing counter, since each has its own loop
// state - but they should all bound it against the same value, so a
// degenerate/self-intersecting scene can't hang one integrator while every
// other integrator stays safely bounded. Named "Crossings" not "Skips" -
// use this spelling everywhere for grep-ability.
CPU_GPU_CONST int kMaxMediumBoundaryCrossings = 32;
