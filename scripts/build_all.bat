@echo off
:: Build all components of the Ray Tracer project
:: Usage: build_all.bat [Debug|Release]

setlocal enabledelayedexpansion

:: Get configuration (default: Release)
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

echo ========================================
echo Building Ray Tracer - %CONFIG% Configuration
echo ========================================
echo.

:: Check if we're in a VS Developer environment
where msbuild >nul 2>&1
if errorlevel 1 (
	echo Error: MSBuild not found!
	echo Please run this from a Visual Studio Developer Command Prompt.
	exit /b 1
)

:: Build the entire solution
echo Building C++ solution...
msbuild ray_tracer.sln /p:Configuration=%CONFIG% /p:Platform=x64 /v:minimal /m

if errorlevel 1 (
	echo.
	echo Build FAILED!
	exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Outputs:
echo   - Launcher:       launcher\x64\%CONFIG%\ray_tracer.exe
echo   - CPU Renderer:   cpu_renderer\x64\%CONFIG%\cpu_renderer.lib
echo   - OptiX Renderer: x64\%CONFIG%\optix_renderer.lib
echo   - OptiX PTX:      gpu\optix\optix_programs.ptx
echo   - Tests:          tests\x64\%CONFIG%\ray_tracer_tests.exe
echo.
echo Run the launcher:
echo   launcher\x64\%CONFIG%\ray_tracer.exe --help
echo.

endlocal
