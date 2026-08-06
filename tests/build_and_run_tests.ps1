#!/usr/bin/env pwsh
# PowerShell script to build and run Ray Tracer unit tests
# Alternative to batch script with better error handling and output

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray Tracer Unit Tests Build Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if we're in the tests directory
if (-not (Test-Path "unit")) {
	Write-Host "ERROR: Please run this script from the tests/ directory" -ForegroundColor Red
	exit 1
}

# Create build directory
if (-not (Test-Path "build")) {
	New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location build

# Step 1: Configure
Write-Host "[1/4] Configuring CMake..." -ForegroundColor Yellow
cmake .. -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) {
	Write-Host "ERROR: CMake configuration failed" -ForegroundColor Red
	Write-Host "Make sure you have CMake and Visual Studio installed" -ForegroundColor Red
	Set-Location ..
	exit 1
}

# Step 2: Build Debug
Write-Host "`n[2/4] Building tests (Debug)..." -ForegroundColor Yellow
cmake --build . --config Debug
if ($LASTEXITCODE -ne 0) {
	Write-Host "ERROR: Debug build failed" -ForegroundColor Red
	Set-Location ..
	exit 1
}

# Step 3: Build Release
Write-Host "`n[3/4] Building tests (Release)..." -ForegroundColor Yellow
cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
	Write-Host "ERROR: Release build failed" -ForegroundColor Red
	Set-Location ..
	exit 1
}

# Step 4: Run tests
Write-Host "`n[4/4] Running tests...`n" -ForegroundColor Yellow

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Debug Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
& "Debug\ray_tracer_tests.exe" --gtest_color=yes
$debugResult = $LASTEXITCODE

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Running Release Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
& "Release\ray_tracer_tests.exe" --gtest_color=yes
$releaseResult = $LASTEXITCODE

Set-Location ..

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Test Results Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($debugResult -eq 0) {
	Write-Host "Debug tests:   ✓ PASSED" -ForegroundColor Green
} else {
	Write-Host "Debug tests:   ✗ FAILED (exit code: $debugResult)" -ForegroundColor Red
}

if ($releaseResult -eq 0) {
	Write-Host "Release tests: ✓ PASSED" -ForegroundColor Green
} else {
	Write-Host "Release tests: ✗ FAILED (exit code: $releaseResult)" -ForegroundColor Red
}

Write-Host ""

if ($debugResult -eq 0 -and $releaseResult -eq 0) {
	Write-Host "✓ All tests passed!" -ForegroundColor Green
	exit 0
} else {
	Write-Host "✗ Some tests failed" -ForegroundColor Red
	exit 1
}
