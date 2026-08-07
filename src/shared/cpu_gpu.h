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
