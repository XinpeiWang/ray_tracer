@echo off
:: Setup script for Ray Tracer build environment
:: This sets environment variables for the current session

echo ========================================
echo Ray Tracer Build Environment Setup
echo ========================================
echo.

:: Detect CUDA Toolkit
set "CUDA_BASE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if exist "%CUDA_BASE%\v13.3" (
	set "CudaToolkitPath=%CUDA_BASE%\v13.3"
	echo [OK] Found CUDA Toolkit 13.3
) else if exist "%CUDA_BASE%\v13.2" (
	set "CudaToolkitPath=%CUDA_BASE%\v13.2"
	echo [OK] Found CUDA Toolkit 13.2
) else (
	echo [WARNING] CUDA Toolkit not found!
	echo Please install CUDA Toolkit 12.x or 13.x
)

:: Detect OptiX SDK
set "OPTIX_BASE=C:\ProgramData\NVIDIA Corporation"
if exist "%OPTIX_BASE%\OptiX SDK 9.1.0" (
	set "OptixSdkPath=%OPTIX_BASE%\OptiX SDK 9.1.0"
	echo [OK] Found OptiX SDK 9.1.0
) else (
	echo [WARNING] OptiX SDK not found!
	echo Please install OptiX SDK 9.1+
)

echo.
echo Environment variables set for this session:
echo   CudaToolkitPath = %CudaToolkitPath%
echo   OptixSdkPath    = %OptixSdkPath%
echo.
echo ========================================
echo Ready to build!
echo ========================================
echo.
echo Run: build_all.bat
echo.
