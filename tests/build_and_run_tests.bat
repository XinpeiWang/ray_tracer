@echo off
REM Build and run Ray Tracer unit tests
REM This script downloads Google Test and builds all tests

echo ========================================
echo Ray Tracer Unit Tests Build Script
echo ========================================
echo.

REM Check if we're in the tests directory
if not exist "unit" (
	echo ERROR: Please run this script from the tests/ directory
	exit /b 1
)

REM Create build directory
if not exist "build" mkdir build
cd build

echo [1/4] Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
	echo ERROR: CMake configuration failed
	echo Make sure you have CMake and Visual Studio installed
	cd ..
	exit /b 1
)

echo.
echo [2/4] Building tests (Debug)...
cmake --build . --config Debug
if errorlevel 1 (
	echo ERROR: Debug build failed
	cd ..
	exit /b 1
)

echo.
echo [3/4] Building tests (Release)...
cmake --build . --config Release
if errorlevel 1 (
	echo ERROR: Release build failed
	cd ..
	exit /b 1
)

echo.
echo [4/4] Running tests...
echo.
echo ========================================
echo Running Debug Tests
echo ========================================
Debug\ray_tracer_tests.exe --gtest_color=yes
set DEBUG_RESULT=%errorlevel%

echo.
echo ========================================
echo Running Release Tests
echo ========================================
Release\ray_tracer_tests.exe --gtest_color=yes
set RELEASE_RESULT=%errorlevel%

cd ..

echo.
echo ========================================
echo Test Results Summary
echo ========================================
echo Debug tests:   %DEBUG_RESULT%
echo Release tests: %RELEASE_RESULT%
echo.

if %DEBUG_RESULT%==0 if %RELEASE_RESULT%==0 (
	echo ✓ All tests passed!
	exit /b 0
) else (
	echo ✗ Some tests failed
	exit /b 1
)
